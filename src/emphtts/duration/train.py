"""Pretrain the duration predictor on a streamed dataset (Emilia EN from the Hugging Face Hub by default)."""
import hydra
import torch
from accelerate import PartialState

from emphtts.duration.data import build_streaming_dataset, load_validation_dataset
from emphtts.duration.trainer import Trainer
from emphtts.tts.infer.utils_infer import build_duration_model
from emphtts.tts.model.utils import get_tokenizer


def build_trainer(cfg):
    """Duration predictor (optionally initialized from ``ckpts.pretrain``) and its trainer."""
    vocab_id_map, vocab_size = get_tokenizer(cfg.datasets.vocab_path, cfg.datasets.tokenizer)
    duration_predictor = build_duration_model(cfg, vocab_size)

    if cfg.ckpts.get("pretrain"):
        checkpoint = torch.load(cfg.ckpts.pretrain, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
        duration_predictor.load_state_dict(state)
        print(f"Initialized from {cfg.ckpts.pretrain}")

    return Trainer(
        model=duration_predictor,
        vocab_size=vocab_size,
        vocab_char_map=vocab_id_map,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        total_updates=cfg.optim.get("total_updates", 0),
        epochs=cfg.optim.get("epochs", 0),
        batch_size=cfg.datasets.batch_size,
        last_per_updates=cfg.ckpts.last_per_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=cfg.ckpts.keep_last_n_checkpoints,
        max_grad_norm=cfg.optim.max_grad_norm,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        checkpoint_path=cfg.ckpts.save_dir,
        wandb_project=cfg.wandb.project,
        wandb_run_name=cfg.wandb.run_name,
        loss_fn=cfg.loss.loss_fn,
        lambda_L1=cfg.loss.lambda_L1,
        gumbel_tau=cfg.loss.gumbel_tau,
        n_class=cfg.loss.n_class,
        n_frame_per_class=cfg.loss.n_frame_per_class,
        use_ema=cfg.optim.use_ema,
        hop_length=cfg.model.mel_spec.hop_length,
        sample_rate=cfg.model.mel_spec.target_sample_rate,
    )


@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="duration_predictor"
)
def main(cfg):
    trainer = build_trainer(cfg)

    state = PartialState()
    train_dataset = build_streaming_dataset(
        list(cfg.datasets.train.values()),
        cfg.model.mel_spec,
        probabilities=cfg.datasets.get("probabilities"),
        seed=cfg.datasets.seed,
        shuffle_buffer=cfg.datasets.shuffle_buffer,
        rank=state.process_index,
        world_size=state.num_processes,
    )
    trainer.train_streaming(train_dataset, load_validation_dataset(cfg), num_workers=cfg.datasets.num_workers)


if __name__ == "__main__":
    main()
