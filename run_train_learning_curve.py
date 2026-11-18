from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from trainer_accum_target_earlystop import (
    PredictorTrainConfig,
    AlignmentPredictorTrainer,
)
from model import FitnessPrecisionPredictor
from cached_variant_a_dataset import CachedVariantADataset, load_split_records


# ============================================================================
# Paths / experiment configuration
# ============================================================================

GRAPH_CACHE_DIR = Path(
    os.environ.get(
        "GRAPH_CACHE_DIR",
        "graph_cache/variant_a_v11_pgnn_sync_context_behavior_stats_recomputed",
    )
)

SAVE_MODELS_DIR = Path(
    os.environ.get(
        "SAVE_MODELS_DIR",
        "models/learning_curve",
    )
)

SPLIT_DIR = Path(
    os.environ.get(
        "SPLIT_DIR",
        "splits_recomputed_dataset/grouped",
    )
)

EXPERIMENT_NAME = os.environ.get(
    "EXPERIMENT_NAME",
    "experiment",
)


# ============================================================================
# Model configuration
# ============================================================================

MODEL_CONFIG = {
    "enode_in_dim": 5,
    "pnode_in_dim": 4,
    "tnode_in_dim": 5,
    "hidden_dim": 128,
    "num_layers": 4,
    "dropout": 0.1,
    "q_dim": 32,
    "mismatch_dim": 32,
    "behavior_in_dim": 16,
    "behavior_dim": 32,
    "match_weighting": "uniform"
}


# ============================================================================
# Training configuration
# ============================================================================

TRAIN_CONFIG = {
    "lr": 3e-4,
    "batch_size": 32,
    "epochs": 400,
    "num_workers": 0,
    "grad_accum_steps": 1,
    "use_bin_weighted_loss": False,
    "early_stopping_metric": "loss",
    "early_stopping_patience": 50,
    "restore_best_weights": True,
}


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train one point of a nested learning-curve experiment."
        )
    )

    parser.add_argument(
        "--train-size",
        type=str,
        required=True,
        help=(
            "Exact number of training records to use "
            "(e.g. 2000, 4000, 8000) or 'full'."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Repeat seed. For a fixed seed, all training sizes are "
            "nested prefixes of the same permutation."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing result directory for the same "
            "training size and seed."
        ),
    )

    return parser.parse_args()


# ============================================================================
# Random seeds
# ============================================================================

def set_training_seed(seed: int) -> None:
    """
    Set random seeds used for model initialization/training.

    This is intentionally called AFTER constructing the training subset,
    so subset generation does not consume global RNG state used by training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ============================================================================
# Nested learning-curve subset construction
# ============================================================================

def make_nested_subset(
    records,
    train_size: int,
    seed: int,
):
    """
    Construct a deterministic nested subset.

    A single seed-specific permutation of the COMPLETE training split
    is generated:

        permutation(seed)

    Each learning-curve dataset is simply a prefix of that permutation:

        2k   = permutation[:2000]
        4k   = permutation[:4000]
        8k   = permutation[:8000]
        16k  = permutation[:16000]
        32k  = permutation[:32000]
        full = permutation[:N]

    Therefore, for the same seed:

        2k ⊂ 4k ⊂ 8k ⊂ 16k ⊂ 32k ⊂ full

    This relationship is exact.
    """
    n_total = len(records)

    if n_total == 0:
        raise ValueError("Training split is empty.")

    if train_size <= 0:
        raise ValueError(
            f"train_size must be > 0, got {train_size}."
        )

    if train_size > n_total:
        raise ValueError(
            f"Requested train_size={train_size:,}, but the full "
            f"training split contains only {n_total:,} records."
        )

    # IMPORTANT:
    # For the same seed and same training split, this produces exactly
    # the same permutation for every invocation.
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)

    # Prefix selection guarantees nesting.
    selected_indices = order[:train_size]

    subset = [
        records[int(i)]
        for i in selected_indices
    ]

    return subset, selected_indices


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    seed = int(args.seed)
    train_size_arg = args.train_size.strip().lower()

    # ------------------------------------------------------------------------
    # Load the FIXED train/validation split.
    #
    # Only the number of training examples changes.
    # Validation remains identical for every experiment.
    # ------------------------------------------------------------------------

    train_records_full = load_split_records(
        SPLIT_DIR / "train.csv"
    )

    validation_records = load_split_records(
        SPLIT_DIR / "val.csv"
    )

    n_train_full = len(train_records_full)

    if n_train_full == 0:
        raise ValueError("Full training split is empty.")

    # ------------------------------------------------------------------------
    # Resolve requested training size.
    # ------------------------------------------------------------------------

    if train_size_arg == "full":
        train_size = n_train_full
        is_full_training_set = True
        size_label = "full"

    else:
        try:
            train_size = int(train_size_arg)
        except ValueError as exc:
            raise ValueError(
                "--train-size must be an integer such as 2000, "
                "4000, 8000, 16000, 32000, or the string 'full'."
            ) from exc

        is_full_training_set = False
        size_label = f"{train_size:06d}"

    if train_size > n_train_full:
        raise ValueError(
            f"Requested training size {train_size:,}, but the "
            f"training split contains only {n_train_full:,} records."
        )

    # ------------------------------------------------------------------------
    # Construct nested training subset.
    #
    # Every size for the same seed uses the same seed-specific permutation.
    # ------------------------------------------------------------------------

    train_records, selected_indices = make_nested_subset(
        train_records_full,
        train_size=train_size,
        seed=seed,
    )

    # Training/model RNG is reset AFTER subset construction.
    set_training_seed(seed)

    effective_fraction = train_size / n_train_full

    # ------------------------------------------------------------------------
    # Output directory
    #
    # Example:
    #
    # learning_curve/
    #   pgnn-sync-context/
    #       size_002000/
    #           seed_43/
    #       size_004000/
    #           seed_43/
    #       ...
    #       size_full/
    #           seed_43/
    # ------------------------------------------------------------------------

    save_dir = (
        SAVE_MODELS_DIR
        / "learning_curve"
        / EXPERIMENT_NAME
        / f"size_{size_label}"
        / f"seed_{seed}"
    )

    if save_dir.exists():
        if args.overwrite:
            print(
                f"Removing existing result directory:\n"
                f"  {save_dir}"
            )
            shutil.rmtree(save_dir)

        else:
            raise FileExistsError(
                f"Result directory already exists:\n"
                f"  {save_dir}\n\n"
                "Use --overwrite only if you intentionally want "
                "to replace this run."
            )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Save exact training indices.
    #
    # These files also allow you to verify the nesting later:
    #
    # set(indices_2k).issubset(indices_4k)
    # etc.
    # ------------------------------------------------------------------------

    np.save(
        save_dir / "train_subset_indices.npy",
        selected_indices,
    )

    # ------------------------------------------------------------------------
    # Reproducibility metadata
    # ------------------------------------------------------------------------

    run_info = {
        "experiment_name": EXPERIMENT_NAME,

        "requested_train_size": train_size_arg,
        "train_size": train_size,
        "is_full_training_set": is_full_training_set,
        "effective_train_fraction": effective_fraction,

        "seed": seed,

        "num_train_full": n_train_full,
        "num_train_used": len(train_records),
        "num_validation": len(validation_records),

        "subset_method": (
            "prefix_of_seed_specific_permutation"
        ),

        "split_dir": str(SPLIT_DIR),
        "graph_cache_dir": str(GRAPH_CACHE_DIR),

        "model_config": MODEL_CONFIG,
        "train_config": TRAIN_CONFIG,
    }

    with (
        save_dir / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_info,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Print experiment information
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LEARNING-CURVE RUN")
    print("=" * 80)

    print(
        f"Experiment            : {EXPERIMENT_NAME}"
    )
    print(
        f"Requested train size  : {train_size_arg}"
    )
    print(
        f"Actual training size  : {train_size:,}"
    )
    print(
        f"Full training size    : {n_train_full:,}"
    )
    print(
        f"Effective fraction    : "
        f"{effective_fraction:.6f} "
        f"({effective_fraction * 100:.2f}%)"
    )
    print(
        f"Seed                  : {seed}"
    )
    print(
        f"Validation size       : "
        f"{len(validation_records):,}"
    )
    print(
        f"Full-data run         : "
        f"{is_full_training_set}"
    )
    print(
        f"Output directory      : {save_dir}"
    )

    print("=" * 80)

    print("\nNested subset construction:")
    print(
        f"  permutation seed = {seed}"
    )
    print(
        f"  selected prefix  = first {train_size:,} / "
        f"{n_train_full:,} records"
    )

    print("=" * 80 + "\n")

    # ------------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------------

    train_ds = CachedVariantADataset(
        train_records,
        cache_dir=GRAPH_CACHE_DIR,
        label=(
            f"train-{size_label}-seed-{seed}"
        ),
    )

    valid_ds = CachedVariantADataset(
        validation_records,
        cache_dir=GRAPH_CACHE_DIR,
        label="validation",
    )

    # ------------------------------------------------------------------------
    # Model + trainer
    # ------------------------------------------------------------------------

    model = FitnessPrecisionPredictor(
        **MODEL_CONFIG
    )

    train_config = PredictorTrainConfig(
        **TRAIN_CONFIG
    )

    trainer = AlignmentPredictorTrainer(
        model,
        config=train_config,
    )

    # ------------------------------------------------------------------------
    # Model information
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(model)

    print("\n" + "=" * 80)
    print("MODEL HYPERPARAMETERS")
    print("=" * 80)

    for key, value in MODEL_CONFIG.items():
        print(
            f"{key:30s}: {value}"
        )

    print("\n" + "=" * 80)
    print("TRAINING HYPERPARAMETERS")
    print("=" * 80)
    print(train_config)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n" + "=" * 80)
    print("MODEL PARAMETERS")
    print("=" * 80)

    print(
        f"Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    print(
        f"Non-trainable params : "
        f"{total_params - trainable_params:,}"
    )

    print("=" * 80 + "\n")

    # ------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------

    history = trainer.fit(
        train_ds,
        valid_ds,
    )

    # ------------------------------------------------------------------------
    # Avoid duplicating every checkpoint state in the final checkpoint.
    # ------------------------------------------------------------------------

    history_compact = {
        key: value
        for key, value in history.items()
        if key != "models"
    }

    # ------------------------------------------------------------------------
    # Save final best-restored model
    # ------------------------------------------------------------------------

    model_save_path = (
        save_dir
        / "model_best_restored.pt"
    )

    torch.save(
        {
            "model_state_dict":
                trainer.model.state_dict(),

            "model_config":
                MODEL_CONFIG,

            "train_config":
                TRAIN_CONFIG,

            "requested_train_size":
                train_size_arg,

            "train_size":
                train_size,

            "is_full_training_set":
                is_full_training_set,

            "effective_train_fraction":
                effective_fraction,

            "seed":
                seed,

            "num_train_full":
                n_train_full,

            "num_train_used":
                len(train_records),

            "num_validation":
                len(validation_records),

            "subset_method":
                "prefix_of_seed_specific_permutation",

            "best_epoch":
                history.get("best_epoch"),

            "best_valid_loss":
                history.get("best_valid_loss"),

            "history":
                history_compact,
        },
        model_save_path,
    )

    print("\n" + "=" * 80)
    print("RUN COMPLETED")
    print("=" * 80)

    print(
        f"Training size : {train_size:,}"
    )
    print(
        f"Seed          : {seed}"
    )
    print(
        f"Model saved   : {model_save_path}"
    )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()