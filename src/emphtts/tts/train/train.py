# training script.

import os
import shutil

import hydra
from omegaconf import OmegaConf

from emphtts.tts.infer.utils_infer import build_f5_model
from emphtts.tts.model import Trainer
from emphtts.tts.model.dataset import load_dataset
from emphtts.tts.model.utils import get_tokenizer, seed_everything


@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(model_cfg):
    exp_name = model_cfg.ckpts.save_dir.split('/')[-1]

    vocab_char_map, _ = get_tokenizer(model_cfg.model.tokenizer_path, model_cfg.model.tokenizer)
    model = build_f5_model(model_cfg, vocab_char_map)

    # Finetuning starts from an explicit checkpoint copied into this run's directory.
    if model_cfg.ckpts.get("pretrain"):
        save_dir = model_cfg.ckpts.save_dir
        os.makedirs(save_dir, exist_ok=True)
        source = model_cfg.ckpts.pretrain
        shutil.copy2(source, os.path.join(save_dir, "pretrained_" + os.path.basename(source)))

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_total_updates=model_cfg.optim.get('num_total_updates', None),
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=model_cfg.ckpts.save_dir,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project=model_cfg.ckpts.get('logger_proj', "EmphTTS"),
        wandb_run_name=exp_name,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        finetune=bool(model_cfg.ckpts.get("pretrain")),
    )

    train_dataset = load_dataset(
        model_cfg.datasets.name,
        dataset_type=model_cfg.datasets.dataset_type,
        mel_spec_kwargs=model_cfg.model.mel_spec,
        index_path=model_cfg.datasets.get("index_path"),
    )
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    seed_everything(666)
    main()
