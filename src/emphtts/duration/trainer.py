from __future__ import annotations

import gc
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from emphtts.duration.duration_predictor import calculate_remaining_lengths
from emphtts.tts.model.dataset import collate_fn
from emphtts.tts.model.trainer import wandb_available
from emphtts.tts.model.utils import default, exists, list_str_to_idx


def masked_l1_loss(est_lengths, tar_lengths):
    first_zero_idx = (tar_lengths == 0).int().argmax(dim=1)
    B, L = tar_lengths.shape
    range_tensor = torch.arange(L, device=tar_lengths.device).expand(B, L)
    mask = range_tensor <= first_zero_idx[:, None]  # Include the first 0
    loss = F.l1_loss(est_lengths, tar_lengths, reduction='none')  # (B, L)
    loss = loss * mask  # Zero out ignored positions
    loss = loss.sum() / mask.sum()  # Normalize by valid elements
    return loss


def masked_cross_entropy_loss(est_length_logits, tar_length_labels):
    first_zero_idx = (tar_length_labels == 0).int().argmax(dim=1)
    B, L = tar_length_labels.shape
    range_tensor = torch.arange(L, device=tar_length_labels.device).expand(B, L)
    mask = range_tensor <= first_zero_idx[:, None]  # Include the first 0
    loss = F.cross_entropy(
        est_length_logits.reshape(-1, est_length_logits.size(-1)),
        tar_length_labels.reshape(-1),
        reduction='none'
    ).reshape(B, L)
    loss = loss * mask
    loss = loss.sum() / mask.sum()
    return loss


class Trainer:
    """Duration-predictor trainer.

    ``train_streaming`` runs a fixed number of updates over a streamed (iterable) dataset, as in pretraining;
    ``train`` runs epochs over a map-style dataset, as in fine-tuning.
    """

    def __init__(
        self,
        model,
        vocab_size,
        vocab_char_map,
        loss_fn='L1',
        lambda_L1=1,
        gumbel_tau=0.5,
        n_class=301,
        n_frame_per_class=10,
        total_updates=85_000,  # train_streaming
        epochs=21,  # train
        learning_rate=1e-4,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger: str | None = "wandb",  # "wandb" | None
        wandb_project="EmphTTS",
        wandb_run_name="duration_predictor",
        wandb_resume_id: str = None,
        last_per_updates=None,
        accelerate_kwargs: dict | None = None,
        ema_kwargs: dict | None = None,
        use_ema: bool = False,
        hop_length: int = 256,
        sample_rate: int = 24_000,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

        if logger == "wandb" and not wandb_available():
            logger = None
        print(f"Using logger: {logger}")

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **(accelerate_kwargs or {}),
        )

        if logger == "wandb":
            init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "group": wandb_run_name}}
            if exists(wandb_resume_id):
                init_kwargs["wandb"]["id"] = wandb_resume_id
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "total_updates": total_updates,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                },
            )

        self.model = model
        self.vocab_size = vocab_size
        self.vocab_char_map = vocab_char_map
        assert loss_fn in ['L1', 'CE', 'L1_and_CE']
        self.loss_fn = loss_fn
        self.lambda_L1 = lambda_L1
        self.n_class = n_class
        self.n_frame_per_class = n_frame_per_class
        self.gumbel_tau = gumbel_tau

        self.use_ema = use_ema
        if use_ema and self.is_main:
            self.ema = EMA(model, include_online_model=False, **(ema_kwargs or {}))
            self.ema.to(self.accelerator.device)

        self.total_updates = total_updates
        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/duration_predictor")
        self.frame_seconds = hop_length / sample_rate

        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if self.use_ema:
                checkpoint['ema_model_state_dict'] = self.ema.state_dict()
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at step {step}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")
                if self.keep_last_n_checkpoints > 0:
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_") and f.endswith(".pt") and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith(".pt") for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            latest_checkpoint = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")],
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )[-1]

        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")
        print(f"Resuming from {latest_checkpoint}")

        if "step" in checkpoint:
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if self.use_ema:
                self.ema.load_state_dict(checkpoint['ema_model_state_dict'])
            step = checkpoint["step"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            step = 0

        del checkpoint
        gc.collect()
        return step

    def _text_ids(self, text, device):
        text_ids = list_str_to_idx(text, self.vocab_char_map).to(device)
        return text_ids.masked_fill(text_ids == -1, self.vocab_size)

    def _losses(self, predictions, tar_lengths):
        """Training loss, the duration error in frames, and the loss terms to log."""
        if self.loss_fn == 'L1':
            loss = masked_l1_loss(est_lengths=predictions, tar_lengths=tar_lengths)
            return loss, loss.detach(), {'loss_L1': loss.item()}

        tar_length_labels = (tar_lengths // self.n_frame_per_class) \
            .clamp(min=0, max=self.n_class-1)  # [0, 1, ..., n_class-1]
        loss_CE = masked_cross_entropy_loss(
            est_length_logits=predictions, tar_length_labels=tar_length_labels
        )

        if self.loss_fn == 'CE':
            with torch.no_grad():
                est_lengths = torch.argmax(predictions, dim=-1) * self.n_frame_per_class
                frame_error = masked_l1_loss(est_lengths=est_lengths, tar_lengths=tar_lengths)
            return loss_CE, frame_error, {'loss_CE': loss_CE.item()}

        # L1_and_CE: L1 on a straight-through Gumbel-softmax sample of the class
        est_length_1hots = F.gumbel_softmax(predictions, tau=self.gumbel_tau, hard=True, dim=-1)
        length_values = torch.arange(self.n_class, device=predictions.device).float() * self.n_frame_per_class
        est_lengths = (est_length_1hots * length_values).sum(-1)
        loss_L1 = masked_l1_loss(est_lengths=est_lengths, tar_lengths=tar_lengths)
        loss = loss_CE + self.lambda_L1 * loss_L1
        return loss, loss_L1.detach(), {'loss_L1': loss_L1.item(), 'loss_CE': loss_CE.item()}

    def validate(self, valid_dataloader, global_step, eval_ema=False):
        """Log the average validation loss and duration error in seconds."""
        self.model.eval()
        total_valid_loss = 0.0
        total_sec_error = 0.0
        count = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                mel = batch['mel'].permute(0, 2, 1)  # (B, L_mel, D)
                text_ids = self._text_ids(batch['text'], mel.device)
                tar_lengths = calculate_remaining_lengths(batch['mel_lengths'])
                if eval_ema:
                    predictions = self.ema.forward_eval(text_ids=text_ids, mel=mel)
                else:
                    predictions = self.model(text_ids=text_ids, mel=mel)

                loss, frame_error, _ = self._losses(predictions, tar_lengths)
                total_sec_error += (frame_error * self.frame_seconds).item()
                total_valid_loss += loss.item()
                count += 1

        ema_suffix = "_ema" if eval_ema else ""
        self.accelerator.log(
            {
                f"valid_loss{ema_suffix}": total_valid_loss / count if count > 0 else 0.0,
                f"valid_sec_error{ema_suffix}": total_sec_error / count if count > 0 else 0.0,
            },
            step=global_step,
        )
        self.model.train()

    def _valid_dataloader(self, valid_dataset, num_workers):
        return self.accelerator.prepare(DataLoader(
            valid_dataset,
            collate_fn=collate_fn,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        ))

    def _build_scheduler(self, total_steps):
        """Linear warmup then linear decay; steps count scheduler calls, which accelerate scales by process."""
        warmup_steps = self.num_warmup_updates * self.accelerator.num_processes
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        return SequentialLR(self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])

    def _update(self, batch):
        """One optimization step on a collated batch; returns the values to log."""
        device = self.accelerator.device
        with self.accelerator.accumulate(self.model):
            mel = batch['mel'].to(device).permute(0, 2, 1)  # (B, L_mel, D)
            text_ids = self._text_ids(batch['text'], device)
            tar_lengths = calculate_remaining_lengths(batch['mel_lengths'].to(device))
            predictions = self.model(text_ids=text_ids, mel=mel)

            loss, frame_error, loss_terms = self._losses(predictions, tar_lengths)
            log_dict = {
                'loss': loss.item(),
                **loss_terms,
                'sec_error': (frame_error * self.frame_seconds).item(),
                'lr': self.scheduler.get_last_lr()[0],
            }

            self.accelerator.backward(loss)

            if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        if self.is_main and self.use_ema and self.accelerator.sync_gradients:
            self.ema.update()
        return log_dict

    def _after_update(self, global_step, log_dict, valid_dataloader, progress_bar, epoch):
        """Log, then save checkpoints and validate on their schedule."""
        if self.accelerator.is_local_main_process:
            self.accelerator.log(log_dict, step=global_step)
            progress_bar.set_postfix(loss=f"{log_dict['loss']:.4f}", epoch=epoch)
            progress_bar.update(1)

        if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
            self.save_checkpoint(global_step)
            if self.accelerator.is_local_main_process:
                self.validate(valid_dataloader, global_step)
                if self.use_ema:
                    self.validate(valid_dataloader, global_step, eval_ema=True)
            self.accelerator.wait_for_everyone()

        if global_step % (self.last_per_updates * self.grad_accumulation_steps) == 0:
            self.save_checkpoint(global_step, last=True)

    def _finish(self, global_step):
        if global_step % (self.last_per_updates * self.grad_accumulation_steps) != 0:
            self.save_checkpoint(global_step, last=True)
        self.accelerator.end_training()

    def train_streaming(self, train_dataset: IterableDataset, valid_dataset: Dataset, num_workers=4):
        """Train for ``total_updates`` steps, re-reading the stream (with a new shuffle) whenever it ends.

        ``train_dataset`` must already be split for this process (see ``build_streaming_dataset``), so its
        loader is not passed through ``accelerator.prepare``, which would shard the stream a second time.
        """
        valid_dataloader = self._valid_dataloader(valid_dataset, num_workers)
        self.scheduler = self.accelerator.prepare(
            self._build_scheduler(self.total_updates * self.accelerator.num_processes)
        )

        # Resuming restores the step count, optimizer and scheduler; the stream restarts from its beginning.
        global_step = self.load_checkpoint()
        if global_step >= self.total_updates:
            self.accelerator.print(f"Already reached {self.total_updates} updates, nothing to do.")
            return

        progress_bar = tqdm(
            total=self.total_updates,
            initial=global_step,
            desc="Training",
            unit="update",
            disable=not self.accelerator.is_local_main_process,
        )

        epoch = 0
        while global_step < self.total_updates:
            self.model.train()
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)  # reshuffles shards and the shuffle buffer
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
            )

            num_batches = 0
            for batch in train_dataloader:
                if global_step >= self.total_updates:
                    break
                num_batches += 1
                log_dict = self._update(batch)
                global_step += 1
                self._after_update(global_step, log_dict, valid_dataloader, progress_bar, epoch)

            if num_batches == 0:
                raise RuntimeError("The training stream produced no batches; check the dataset sources and filters.")
            epoch += 1

        progress_bar.close()
        self._finish(global_step)

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, num_workers=4):
        """Train for ``epochs`` passes over a map-style dataset, shuffled every epoch."""
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            batch_size=self.batch_size,
            shuffle=True,
        )
        valid_dataloader = self._valid_dataloader(valid_dataset, num_workers)

        # The dataloader length is taken before accelerate shards it across processes.
        total_steps = len(train_dataloader) * self.epochs / self.grad_accumulation_steps
        self.scheduler = self._build_scheduler(total_steps)
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)

        # Resuming restores the step count, optimizer and scheduler, and restarts from the first epoch.
        global_step = self.load_checkpoint()

        for epoch in range(self.epochs):
            self.model.train()
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="step",
                disable=not self.accelerator.is_local_main_process,
            )
            for batch in train_dataloader:
                log_dict = self._update(batch)
                global_step += 1
                self._after_update(global_step, log_dict, valid_dataloader, progress_bar, epoch)
            progress_bar.close()

        self._finish(global_step)
