import copy
import gc
import os
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from emphtts.duration.grpo_data_utils import grpo_collate_fn as collate_fn
from emphtts.tts.model.trainer import wandb_available
from emphtts.tts.model.utils import list_str_to_idx


class GRPODurationTrainer:
    """
    GRPO (Group Relative Policy Optimization) for the duration predictor of a frozen TTS model.

    For each prompt, ``num_pre_samples`` durations are sampled from the predictor, synthesized with
    ``inference_fn`` and scored with ``reward_fn``; rewards are normalized within the group and the
    predictor is updated with a clipped policy-gradient loss plus a KL penalty to the initial model.
    """
    def __init__(
        self,
        model,                      # Duration predictor model
        inference_fn,               # Function to generate speech
        reward_fn,                  # Function to compute rewards from generated speech
        
        vocab_size: int,            # Size of the vocabulary
        vocab_char_map: dict,       # Mapping from characters to token IDs

        # Duration model parameters
        n_class: int = 301,         # Number of duration classes
        n_frame_per_class: int = 10, # Number of frames per class
        gumbel_tau: float = 0.7,
        
        # GRPO parameters
        beta: float = 0.04,         # KL regularization weight
        clip_param: float = 0.2,    # PPO clip parameter
        num_pre_samples: int = 8,   # Number of samples per prompt
        compute_gen_logps: bool = True, # Whether to compute generation log probabilities
        asr_reward_type: str = "loss_ctc", # "loss_ctc" or "wer"
        sim_weight: float = 3.0,    # Weight for speaker similarity reward
        asr_weight: float = 1.0,
        use_stress_metric: bool = False, # add the WhiStress stress reward
        stress_balanced_acc_weight: float = 1.0,
        
        # Training parameters
        learning_rate: float = 5e-6,
        num_warmup_updates: int = 10000,
        save_per_updates: int = 10000,
        checkpoint_path: Optional[str] = None,
        all_steps: int = 100000,     # Total training steps
        
        # Batch parameters
        batch_size: int = 8,
        grad_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0,
        
        # Logging parameters
        logger: Optional[str] = "wandb",
        wandb_project: str = "EmphTTS",
        wandb_run_name: str = "durpred_grpo",
        wandb_resume_id: Optional[str] = None,

        accelerate_kwargs: dict | None = None,
    ):
        # Initialize accelerator for distributed training
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

        self.logger = logger
        if self.logger == "wandb":
            if wandb_resume_id:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id, "group": wandb_run_name}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "group": wandb_run_name}}

            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "beta": beta,
                    "clip_param": clip_param,
                    "num_pre_samples": num_pre_samples,
                    "n_class": n_class,
                    "n_frame_per_class": n_frame_per_class,
                    "all_steps": all_steps,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                },
            )
        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        # Store model, inference function, and reward function
        self.model = model

        # Reference model: frozen clone of the initial predictor for the KL term. The frozen modules are
        # only moved to the device, not wrapped by accelerate (they never receive gradients).
        self.ref_model = copy.deepcopy(model).requires_grad_(False).eval().to(self.accelerator.device)
        self.inference_fn = inference_fn.requires_grad_(False).eval().to(self.accelerator.device)
        self.reward_fn = reward_fn
        self.reward_fn.vocoder   = self.reward_fn.vocoder.to(device=self.accelerator.device)
        self.reward_fn.sv_model  = self.reward_fn.sv_model.to(device=self.accelerator.device)
        if use_stress_metric:
            assert asr_reward_type == 'wer', "Stress metric can only be used with WER metric"
            self.reward_fn.whistress_client.whistress = self.reward_fn.whistress_client.whistress.to(self.accelerator.device)
        else:
            self.reward_fn.asr_model = self.reward_fn.asr_model.to(device=self.accelerator.device)
        
        # Store vocabulary and mapping
        self.vocab_size = vocab_size
        self.vocab_char_map = vocab_char_map

        # Store duration model parameters
        self.n_class = n_class
        self.n_frame_per_class = n_frame_per_class
        self.gumbel_tau = gumbel_tau
        
        # Store GRPO parameters
        self.beta = beta
        self.asr_reward_type = asr_reward_type
        self.sim_weight = sim_weight
        self.asr_weight = asr_weight
        self.clip_param = clip_param
        self.num_pre_samples = num_pre_samples
        self.compute_gen_logps = compute_gen_logps
        self.use_stress_metric = use_stress_metric
        self.stress_balanced_acc_weight = stress_balanced_acc_weight
        
        # Store training parameters
        self.learning_rate = learning_rate
        self.num_warmup_updates: int = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.checkpoint_path = checkpoint_path or f"ckpts/{wandb_run_name}"
        self.all_steps = all_steps
        
        # Store batch parameters
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Initialize optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Prepare model and optimizer with accelerator
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # GRPO batch queue
        self.batch_queue = []

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    def save_checkpoint(self, step, last=False):
        """Save model and optimizer state"""
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")
    
    def load_checkpoint(self):
        """Load latest checkpoint if available"""
        if (
            not self.checkpoint_path
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

        print(f'Loading checkpoint: {latest_checkpoint}')
        checkpoint = torch.load(
            f"{self.checkpoint_path}/{latest_checkpoint}", 
            weights_only=True, 
            map_location="cpu"
        )

        if "step" in checkpoint:
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if hasattr(self, 'scheduler') and checkpoint["scheduler_state_dict"]:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
        else:
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            step = 0
        
        del checkpoint
        gc.collect()
        
        print(f'Successfully loaded checkpoint at step {step}')
        return step
    
    @torch.no_grad()
    def get_ref_logps(self, text_ids, mel, sampled_classes):
        """
        Get log probabilities from the reference model for the sampled classes
        """
        B = text_ids.shape[0]
        K = self.num_pre_samples
        with self.accelerator.autocast():  # same precision as the policy model
            ref_logits = self.ref_model(text_ids=text_ids, mel=mel)[:, -1, :]
            ref_logits = ref_logits.unsqueeze(1).repeat(1, K, 1).view(B*K, -1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_logps = torch.gather(
                ref_log_probs, 
                dim=-1, 
                index=sampled_classes.unsqueeze(-1)
            ).squeeze(-1)
        return ref_logps
    
    @torch.no_grad()
    def generate_duration_samples(self, batch_inputs):
        """
        Generate multiple duration predictions from the model for each input
        and evaluate them using the inference function and reward model
        
        Args:
            batch_inputs: Dictionary with text, prompt audio, etc.
            
        Returns:
            Dictionary with duration samples, rewards, and reference logits
        """
        all_sampled_classes = []
        all_durations = []
        all_rewards = []
        all_gen_logps = []

        all_ctc_loss = []
        all_sv_loss = []
        all_stress_balanced_acc = []

        prompt_mel = batch_inputs['mel'].permute(0, 2, 1) # (B, T, 100)
        prompt_text = batch_inputs['text']

        # The target text comes from a different utterance than the prompt audio; it is what gets synthesized
        target_text = batch_inputs['target_text']
        full_text = [prompt + ' ' + target for prompt, target in zip(prompt_text, target_text)]

        # Goes to duration model and TTS
        full_text_ids = list_str_to_idx(full_text, self.vocab_char_map).to(self.accelerator.device)

        # Deepcopy to separate text_ids for SLP and TTS
        slp_text_ids = full_text_ids.detach().clone()
        slp_text_ids = slp_text_ids.masked_fill(slp_text_ids==-1, self.vocab_size) # (B, L)

        # Pre-compute duration logits
        K = self.num_pre_samples
        # Run model once for B inputs
        old_logits = self.model(
            text_ids=slp_text_ids, # (B, L)
            mel=prompt_mel         # (B, T, 100)
        )[:, -1, :]  # (B, n_class)

        # Repeat each result K times along batch dimension
        old_logits = old_logits.unsqueeze(1).repeat(1, K, 1) # (B, K, n_class)

        for _full_text_ids, _target_text, _prompt_mel, _old_logits in zip(
            full_text_ids, target_text, prompt_mel, old_logits
        ):

            duration_sample = F.gumbel_softmax(_old_logits, tau=self.gumbel_tau, hard=True, dim=-1)
            duration2frames = torch.arange(self.n_class).float().to(self.accelerator.device) * self.n_frame_per_class
            est_frames = (duration_sample * duration2frames).sum(-1) # (K, )

            # Compute log probabilities of the samples
            sampled_classes = duration_sample.argmax(dim=-1)
            log_probs = F.log_softmax(_old_logits, dim=-1)
            gen_logps = torch.gather(
                log_probs, 
                dim=-1, 
                index=sampled_classes.unsqueeze(-1)
            ).squeeze(-1)  # (K,)

            # Generate speech using the sampled durations
            sampled_rewards = []

            for i in range(K):
                cur_duration = est_frames[i]
                if cur_duration.isnan() or cur_duration.isinf() or cur_duration == 0:
                    cur_duration = torch.tensor(100.0, device=cur_duration.device)
                infer_full_text_ids = _full_text_ids.unsqueeze(0)
                infer_prompt_mel = _prompt_mel.unsqueeze(0)
                cur_duration = cur_duration.unsqueeze(0)
                with torch.inference_mode():
                    _est_mel = self.inference_fn(
                        full_text_ids=infer_full_text_ids,
                        prompt_mel=infer_prompt_mel,
                        target_duration=cur_duration,
                    )
                    _est_mel = _est_mel.permute(0, 2, 1) # (1, T, 100)

                    loss_dict = self.reward_fn(
                        prompt_mel=infer_prompt_mel,
                        est_mel=_est_mel,
                        target_text=[_target_text],
                    )
                    reward_sim = loss_dict['loss_sim']   # 0 to 1, higher = better
                    reward_asr = loss_dict[self.asr_reward_type]  # lower = better
                    reward = self.sim_weight * reward_sim - self.asr_weight * reward_asr
                    if self.use_stress_metric:
                        reward += self.stress_balanced_acc_weight * loss_dict['stress_balanced_acc']

                        all_stress_balanced_acc.append(
                            loss_dict['stress_balanced_acc']
                        )

                    if reward.isnan() or reward.isinf():
                        reward = torch.tensor(-1.0, device=cur_duration.device)
                    all_ctc_loss.append(reward_asr.nan_to_num(0.0))
                    all_sv_loss.append(reward_sim.nan_to_num(0.0))
                    sampled_rewards.append(reward)
            sampled_rewards = torch.stack(sampled_rewards)  # (K, )
            # Normalize rewards within the group
            if (sampled_rewards.max() - sampled_rewards.min()).item() > 1e-6:
                sampled_rewards = (sampled_rewards - sampled_rewards.mean()) / (sampled_rewards.std() + 1e-8)

            all_sampled_classes.append(sampled_classes)
            all_durations.append(est_frames)
            all_gen_logps.append(gen_logps)
            all_rewards.extend(sampled_rewards)  # list with length of B*K
        
        # Concatenate all data
        sampled_classes = torch.cat(all_sampled_classes, dim=0)
        durations = torch.cat(all_durations, dim=0)
        rewards = torch.stack(all_rewards)    # use stack to keep the same device of elements
        gen_logps = torch.cat(all_gen_logps, dim=0)

        ctc_losses = torch.stack(all_ctc_loss)
        sv_losses = torch.stack(all_sv_loss)

        if self.use_stress_metric:
            stress_bal_acc = torch.stack(all_stress_balanced_acc)
        
        if self.is_main:
            self.accelerator.log({
                "ctc_loss": ctc_losses.mean().item(),
                "sv_loss": sv_losses.mean().item(),
                "reward": rewards.mean().item(),
                "reward_min": rewards.min().item(),
                "reward_max": rewards.max().item(),
            }, step=self.global_step)

            if self.use_stress_metric:
                self.accelerator.log(
                    {
                        'stress_balanced_acc': stress_bal_acc.mean().item(),
                    },
                    step=self.global_step
                )

        ref_logps = self.get_ref_logps(slp_text_ids, prompt_mel, sampled_classes)

        batch_outputs = {
            "text_ids": slp_text_ids,
            "prompt_mel": prompt_mel,
            "rewards": rewards,
            "refs": ref_logps,
            "sampled_classes": sampled_classes,
            "durations": durations,
        }
        
        if self.compute_gen_logps:
            batch_outputs["gen_logps"] = gen_logps

        return batch_outputs
    
    def GRPO_step(self, batch):
        """
        Perform a GRPO update step
        
        Args:
            batch: Dictionary with inputs, rewards, reference logits, etc.
            
        Returns:
            Loss value
        """
        # Extract batch data; every per-sample tensor has B*K entries
        rewards = batch['rewards']
        ref_logps = batch['refs']
        sampled_classes = batch['sampled_classes']
        prompt_mel = batch['prompt_mel']
        text_ids = batch['text_ids']

        # Forward pass to get current model logits
        K = self.num_pre_samples
        B, _, _ = prompt_mel.shape
        cur_logits = self.model(
            text_ids=text_ids, # (B, L)
            mel=prompt_mel         # (B, T, 100)
        )[:, -1, :]
        cur_logits = cur_logits.unsqueeze(1).repeat(1, K, 1).view(B*K, -1) 

        # Compute current log probabilities for sampled actions
        log_probs = F.log_softmax(cur_logits, dim=-1)
        cur_logps = torch.gather(
            log_probs, 
            dim=-1, 
            index=sampled_classes.unsqueeze(-1)
        ).squeeze(-1)  # (B)

        # k3 estimator of KL(cur || ref): exp(ref - cur) - (ref - cur) - 1
        # Clamp before exp: cur_logps ≈ -inf when model assigns near-zero prob to sampled class,
        # causing exp(+inf) - inf - 1 = NaN
        kl_diff = (ref_logps - cur_logps).clamp(-20, 20)
        kl_div = torch.exp(kl_diff) - kl_diff - 1 # (B)

        # Compute probability ratio for PPO
        if "gen_logps" in batch:
            gen_logps = batch['gen_logps']
            ratio = torch.exp((cur_logps - gen_logps).clamp(-20, 20))
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
            loss = torch.min(ratio * rewards, clipped_ratio * rewards)
        else:
            # Simplification if gen_logps not available
            loss = torch.exp(cur_logps - cur_logps.detach()) * rewards
        
        # Final GRPO loss with KL regularization
        loss = -(loss - self.beta * kl_div) # (B)
        loss = loss.mean()
        
        return loss
    
    def get_batch(self):
        """Get a batch from the queue or return None if empty"""
        if not self.batch_queue:
            return None
        return self.batch_queue.pop(0)
    
    def generate_mode(self, num_batches=8):
        """Generate ``num_batches`` batches of scored rollouts and add them to the batch queue."""
        for _ in range(num_batches):
            try:
                batch_inputs = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_dataloader)
                batch_inputs = next(self.train_iterator)

            # Generate samples and compute rewards
            batch_outputs = self.generate_duration_samples(batch_inputs)
            # Check if batch has sufficient reward diversity
            rewards = batch_outputs["rewards"]
            if (rewards.max() - rewards.min()).item() < 0.01:
                if self.is_main:
                    print("Skipping batch with low reward diversity")
                continue
            # Add batch to queue
            self.batch_queue.append(batch_outputs)

    def train(self, train_dataset, num_workers=8):
        """
        Train the model using GRPO

        Args:
            train_dataset: Training dataset
            num_workers: Number of workers for data loading
        """
        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        self.train_iterator = iter(self.train_dataloader)

        # Setup schedulers
        warmup_steps = self.num_warmup_updates * self.accelerator.num_processes
        total_steps = self.all_steps
        decay_steps = total_steps - warmup_steps
        
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        
        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, decay_scheduler], 
            milestones=[warmup_steps]
        )
        
        self.scheduler = self.accelerator.prepare(self.scheduler)
        
        # Load checkpoint if available
        start_step = self.load_checkpoint()
        self.global_step = start_step
        
        # Generate initial batches
        self.generate_mode()
        
        # Training loop
        progress = range(1, self.all_steps * self.grad_accumulation_steps + 1)
        
        # Skip steps that are already done
        progress = [step for step in progress if step > start_step]
        if self.is_main:
            progress = tqdm(progress, desc="Training", unit="step")
        
        for step in progress:
            # Get batch from queue or generate more
            batch = self.get_batch()
            while batch is None:
                self.generate_mode()
                batch = self.get_batch()
            
            # GRPO update
            with self.accelerator.accumulate(self.model):
                loss = self.GRPO_step(batch)
                if loss.isnan() or loss.isinf():
                    self.optimizer.zero_grad()
                    continue
                self.accelerator.backward(loss)
                
                if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                    total_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    total_norm = torch.norm(
                        torch.stack([
                            torch.norm(p.grad.detach(), 2)
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]),
                        2
                    )
                
                self.accelerator.log({
                    "grad_norm": total_norm.item()
                }, step=self.global_step)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # Log metrics
            if self.is_main:
                self.accelerator.log({
                    "loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }, step=self.global_step)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.8f}"
                )
            
            # Save checkpoint
            if self.global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                self.save_checkpoint(self.global_step)

        # Save final checkpoint
        self.save_checkpoint(self.global_step, last=True)
        self.accelerator.end_training()