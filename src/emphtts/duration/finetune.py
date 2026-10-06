"""Fine-tune a pretrained duration predictor on map-style data (Expresso + a fixed 5% of an Emilia subset)."""
import hydra

from emphtts.duration.data import build_map_dataset, load_validation_dataset
from emphtts.duration.train import build_trainer


@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="duration_finetune"
)
def main(cfg):
    trainer = build_trainer(cfg)
    train_dataset = build_map_dataset(
        list(cfg.datasets.finetune.values()),
        cfg.model.mel_spec,
        seed=cfg.datasets.seed,
        num_proc=cfg.datasets.get("num_proc"),
    )
    trainer.train(train_dataset, load_validation_dataset(cfg), num_workers=cfg.datasets.num_workers)


if __name__ == "__main__":
    main()
