import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

from emphtts.duration.grpo_data_utils import GRPOParquetDataset
from emphtts.duration.grpo_trainer import GRPODurationTrainer
from emphtts.duration.grpo_utils import TTSInferenceFn, TTSRewardFn
from emphtts.tts.infer.utils_infer import build_duration_model
from emphtts.tts.model.utils import get_tokenizer, seed_everything


@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="grpo",
)
def main(cfg):
    # ── Tokenizer ────────────────────────────────────────────────────────────
    vocab_char_map, vocab_size = get_tokenizer(cfg.tts.vocab_file, "custom")

    # ── Duration predictor ───────────────────────────────────────────────────
    durpred_cfg = OmegaConf.load(cfg.durpred.config)
    model = build_duration_model(durpred_cfg, vocab_size)

    ckpt = torch.load(cfg.durpred.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print(f"Loaded DurPred checkpoint: {cfg.durpred.ckpt}")

    # ── TTS inference function ───────────────────────────────────────────────
    inference_fn = TTSInferenceFn(
        tts_config=cfg.tts.config,
        tts_ckpt_path=cfg.tts.ckpt,
        vocab_char_map=vocab_char_map,
        cfg_strength=cfg.tts.cfg_strength,
        steps=cfg.tts.steps,
        use_ema=cfg.tts.use_ema,
    )

    # ── Reward function ──────────────────────────────────────────────────────
    reward_fn = TTSRewardFn(
        campplus_ckpt_path=cfg.reward.campplus_ckpt,
        vocos_local_path=cfg.reward.vocos_local_path,
        tts_sample_rate=durpred_cfg.model.mel_spec.target_sample_rate,
        asr_model_id=cfg.reward.asr_model_id,
        use_stress_metric=cfg.grpo.get('use_stress_metric', False)
    )

    # ── Training dataset ─────────────────────────────────────────────────────
    mel_kwargs = dict(
        target_sample_rate=durpred_cfg.model.mel_spec.target_sample_rate,
        n_mel_channels=durpred_cfg.model.mel_spec.n_mel_channels,
        hop_length=durpred_cfg.model.mel_spec.hop_length,
        n_fft=durpred_cfg.model.mel_spec.n_fft,
        win_length=durpred_cfg.model.mel_spec.win_length,
        mel_spec_type=durpred_cfg.model.mel_spec.mel_spec_type,
    )
    sub_datasets = []
    for src in cfg.datasets.sources.values():
        ds_kwargs = dict(path=src.path, split=src.split,
                         audio_column=src.audio_column, text_column=src.text_column,
                         **mel_kwargs)
        if src.name:
            ds_kwargs["name"] = src.name
        if src.data_files:
            ds_kwargs["data_files"] = src.data_files
        if src.num_sample is not None:
            ds_kwargs["num_sample"] = src.num_sample
        ds = GRPOParquetDataset(**ds_kwargs)
        print(f"Loaded dataset '{src.path}' with {len(ds)} samples")
        sub_datasets.append(ds)

    train_dataset = sub_datasets[0] if len(sub_datasets) == 1 else ConcatDataset(sub_datasets)
    print(f"Total training samples: {len(train_dataset)}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = GRPODurationTrainer(
        model=model,
        inference_fn=inference_fn,
        reward_fn=reward_fn,
        vocab_size=vocab_size,
        vocab_char_map=vocab_char_map,
        # duration model
        n_class=durpred_cfg.loss.n_class,
        n_frame_per_class=durpred_cfg.loss.n_frame_per_class,
        gumbel_tau=cfg.grpo.gumbel_tau,
        # GRPO
        beta=cfg.grpo.beta,
        clip_param=cfg.grpo.clip_param,
        num_pre_samples=cfg.grpo.num_pre_samples,
        compute_gen_logps=cfg.grpo.compute_gen_logps,
        asr_reward_type=cfg.grpo.asr_reward_type,
        sim_weight=cfg.grpo.sim_weight,
        asr_weight=cfg.grpo.asr_weight,
        use_stress_metric=cfg.grpo.get('use_stress_metric', False),
        stress_balanced_acc_weight=cfg.grpo.get('stress_balanced_acc_weight', 1.0),
        # optimisation
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        all_steps=cfg.optim.all_steps,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        # checkpointing
        save_per_updates=cfg.ckpts.save_per_updates,
        checkpoint_path=cfg.ckpts.checkpoint_path,
        # batch
        batch_size=cfg.datasets.batch_size,
        # logging
        wandb_project=cfg.wandb.project,
        wandb_run_name=cfg.wandb.run_name,
        wandb_resume_id=cfg.wandb.resume_id,
    )

    trainer.train(train_dataset, num_workers=cfg.datasets.num_workers)


if __name__ == "__main__":
    seed_everything(666)
    main()
