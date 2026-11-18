from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import torch

from trainer_accum_target_earlystop import PredictorTrainConfig, AlignmentPredictorTrainer
from model import FitnessPrecisionPredictor
from cached_variant_a_dataset import CachedVariantADataset, load_split_records


GRAPH_CACHE_DIR = Path(
    os.environ.get("GRAPH_CACHE_DIR", "graph_cache/pgnn-sync-context-behavior-stats-recomputed-dataset")
)

SAVE_MODELS_DIR = Path(
    os.environ.get("SAVE_MODELS_DIR", "models")
)

SPLIT_DIR = Path("splits_recomputed_dataset/grouped")


def main():
    train_records = load_split_records(SPLIT_DIR / "train.csv")
    validation_records = load_split_records(SPLIT_DIR / "val.csv")

    train_ds = CachedVariantADataset(
        train_records,
        cache_dir=GRAPH_CACHE_DIR,
        label="train",
    )

    valid_ds = CachedVariantADataset(
        validation_records,
        cache_dir=GRAPH_CACHE_DIR,
        label="validation",
    )

    model_config = {
        "enode_in_dim": 5,
        "pnode_in_dim": 4,
        "tnode_in_dim": 5,
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.1,
        "q_dim": 32,
        "mismatch_dim": 32
    }

    model = FitnessPrecisionPredictor(**model_config)

    train_config = PredictorTrainConfig(
        lr=3e-4,
        batch_size=32,
        epochs=300,
        num_workers=0,
        grad_accum_steps=1,
        use_bin_weighted_loss=False,
        early_stopping_metric="loss",
        early_stopping_patience=70,
        restore_best_weights=True,
    )

    trainer = AlignmentPredictorTrainer(
        model,
        config=train_config,
    )

    # ---------------------------------------------------------
    # Print model information BEFORE training
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Grouped unweighted: pgnn-sync-context-no-behavior-stats-expanded")

    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(model)

    print("\n" + "=" * 80)
    print("MODEL HYPERPARAMETERS")
    print("=" * 80)
    for key, value in model_config.items():
        print(f"{key:30s}: {value}")

    print("\n" + "=" * 80)
    print("TRAINING HYPERPARAMETERS")
    print("=" * 80)
    print(train_config)

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 80)
    print("MODEL PARAMETERS")
    print("=" * 80)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Non-trainable params : {total_params - trainable_params:,}")
    print("=" * 80 + "\n")
    # ---------------------------------------------------------

    history = trainer.fit(train_ds, valid_ds)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    save_dir = SAVE_MODELS_DIR / "grouped-unweighted" / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    for item in history["models"]:
        epoch = item["epoch"]
        model_save_path = save_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": item["state_dict"],
                "config": trainer.cfg,
            },
            model_save_path,
        )

        print("Saved model to", model_save_path)

    model_save_path = save_dir / "model_best_restored.pt"

    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "config": trainer.cfg,
            "best_epoch": history["best_epoch"],
            "best_valid_loss": history["best_valid_loss"],
            "history": history,
        },
        model_save_path,
    )

    print("Saved model to", model_save_path)


if __name__ == "__main__":
    main()
