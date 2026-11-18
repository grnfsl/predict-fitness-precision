from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer

from model import FitnessPrecisionPredictor
from cached_variant_a_dataset import CachedVariantADataset, load_split_records

from graph import (
    AlignmentPairDataset,
    collate_alignment_pairs,
    prepare_pair_specs,
)

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None


# =============================================================================
# Configuration
# =============================================================================

MODEL_ROOT = Path(
    os.environ.get(
        "LEARNING_CURVE_MODEL_ROOT",
        "models/learning_curve/pgnn-sync-context-behavior-stats-recomputed-dataset",
    )
)

SYNTHETIC_TEST_CSV = Path(
    os.environ.get(
        "SYNTHETIC_TEST_CSV",
        "splits_recomputed_dataset/grouped/test.csv",
    )
)

SYNTHETIC_GRAPH_CACHE_DIR = Path(
    os.environ.get(
        "GRAPH_CACHE_DIR",
        "graph_cache/variant_a_v11_pgnn_sync_context_behavior_stats_recomputed",
    )
)

# Real-life layout expected:
#
# data_prep/exact_results/
#   fit_prec_results.csv
#   logs/
#   models/<log_id>/*.pnml
#
REAL_LIFE_DIR = Path(
    os.environ.get(
        "REAL_LIFE_DIR",
        "data_prep/exact_results",
    )
)

REAL_LIFE_CSV = REAL_LIFE_DIR / "fit_prec_results1.csv"
REAL_LIFE_LOG_DIR = REAL_LIFE_DIR / "all_logs"
REAL_LIFE_MODEL_DIR = REAL_LIFE_DIR / "all_models"


# IMPORTANT:
# The training target called "fitness" corresponds to average_trace_fitness
# in the extracted dataset. Therefore use avg_fitness for the real-life set.
REAL_LIFE_FITNESS_COLUMN = "avg_fitness"
REAL_LIFE_PRECISION_COLUMN = "precision"


# Exact learning-curve sizes used during training.
TRAIN_SIZES = [
    2000,
    4000,
    8000,
    16000,
    32000,
    "full",
]

SEEDS = [
    41,
    51,
    61,
    71,
    81,
]


BATCH_SIZE_SYNTHETIC = 32
BATCH_SIZE_REAL_LIFE = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "LEARNING_CURVE_OUTPUT_DIR",
        "learning_curve_results/pgnn-sync-context-behavior-stats-recomputed",
    )
)

SAVE_RESULTS = True
SAVE_PLOTS = True
SHOW_PLOTS = False

# "ci95" or "std"
UNCERTAINTY_BAND = "ci95"


DEFAULT_MODEL_CONFIG = {
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
    # "match_weighting": "frequency"
}


# =============================================================================
# Learning-curve size labels
# =============================================================================


def make_size_label(train_size) -> str:
    """
    Reproduce exactly the directory labels used by
    run_train_learning_curve.py.

    Examples:
        2000   -> "002000"
        32000  -> "032000"
        "full" -> "full"
    """
    if str(train_size).lower() == "full":
        return "full"

    return f"{int(train_size):06d}"


# =============================================================================
# Real-life data preparation
# =============================================================================


def read_log_as_variants(
    log_path: Path,
    activity_key: str = "concept:name",
):
    """
    Read one XES log and compress it to unique trace variants.

    Returns:
        [
            ([activity_1, activity_2, ...], count),
            ...
        ]
    """
    log = xes_importer.apply(
        str(log_path),
        parameters={
            xes_importer.Variants.ITERPARSE.value.Parameters.SHOW_PROGRESS_BAR:
                False
        },
    )

    counter = Counter()

    for trace in log:
        seq = tuple(
            str(event[activity_key])
            for event in trace
        )
        counter[seq] += 1

    return [
        (list(seq), count)
        for seq, count in counter.items()
    ]


def build_real_life_records(csv_path: Path):
    """
    Load the fixed real-life benchmark.

    Absolute Mac paths stored in the CSV are ignored. Files are instead
    resolved from REAL_LIFE_DIR so the same CSV works on Spartan.
    """
    df = pd.read_csv(csv_path)

    if "status" in df.columns:
        df = df[
            df["status"] == "success"
        ].copy()

    required = {
        "log_id",
        "model_name",
        REAL_LIFE_FITNESS_COLUMN,
        REAL_LIFE_PRECISION_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required real-life CSV columns: "
            f"{sorted(missing)}"
        )

    records = []

    for idx, row in df.iterrows():
        log_id = str(row["log_id"])
        model_name = str(row["model_name"])

        # -------------------------------------------------------------
        # Resolve log path.
        #
        # The script is run from the project root (the same directory
        # that contains data_prep/), so all benchmark paths are resolved
        # relative to REAL_LIFE_LOG_DIR / REAL_LIFE_MODEL_DIR.
        # If the CSV stores an absolute or machine-specific path, keep
        # only its filename.
        # -------------------------------------------------------------
        if (
            "log_path" in df.columns
            and pd.notna(row.get("log_path"))
        ):
            log_filename = Path(
                str(row["log_path"])
            ).name
        else:
            log_filename = (
                log_id
                if log_id.lower().endswith(".xes")
                else f"{log_id}.xes"
            )

        log_path = (
            REAL_LIFE_LOG_DIR
            / log_filename
        )

        # -------------------------------------------------------------
        # Resolve model path.
        #
        # IMPORTANT: model_name is not guaranteed to be the physical
        # PNML filename. For example, the heuristics-miner rows in the
        # CSV may have a descriptive model_name while model_path contains
        # the actual filename stored under all_models/. Therefore prefer
        # the basename of model_path whenever that column is available.
        # -------------------------------------------------------------
        if (
            "model_path" in df.columns
            and pd.notna(row.get("model_path"))
        ):
            model_filename = Path(
                str(row["model_path"])
            ).name
        else:
            model_filename = model_name

        # Primary layout used by the supplied benchmark CSV:
        #
        # data_prep/exact_results/all_models/<actual-pnml-filename>
        model_path = (
            REAL_LIFE_MODEL_DIR
            / model_filename
        )

        # Optional fallback for an older nested layout:
        # all_models/<log_id>/<filename>
        if not model_path.exists():
            nested_model_path = (
                REAL_LIFE_MODEL_DIR
                / log_id
                / model_filename
            )

            if nested_model_path.exists():
                model_path = nested_model_path

        if not log_path.exists():
            raise FileNotFoundError(
                "Real-life log not found:\n"
                f"  {log_path}\n"
                f"Current working directory: {Path.cwd()}"
            )

        if not model_path.exists():
            raise FileNotFoundError(
                "Real-life model not found. Tried:\n"
                f"  {REAL_LIFE_MODEL_DIR / model_filename}\n"
                f"  {REAL_LIFE_MODEL_DIR / log_id / model_filename}\n"
                f"CSV model_name: {model_name}\n"
                f"CSV model_path: {row.get('model_path', '<missing>')}\n"
                f"Current working directory: {Path.cwd()}"
            )

        threshold = (
            row["threshold"]
            if "threshold" in df.columns
            else idx
        )

        records.append(
            {
                "name":
                    f"{log_id}_thr_{threshold}",

                "log_id":
                    log_id,

                "threshold":
                    threshold,

                "model_name":
                    model_name,

                "log_path":
                    log_path,

                "model_path":
                    model_path,

                "fitness":
                    float(
                        row[
                            REAL_LIFE_FITNESS_COLUMN
                        ]
                    ),

                "precision":
                    float(
                        row[
                            REAL_LIFE_PRECISION_COLUMN
                        ]
                    ),
            }
        )

    print(
        f"Loaded {len(records)} real-life pairs "
        f"from {df['log_id'].nunique()} logs."
    )

    return records


def build_real_life_dataset(records):
    """
    Build the fixed real-life graphs once and reuse them for every
    learning-curve checkpoint.

    IMPORTANT:
    These graphs are constructed using the same current
    PGNN-sync-context-behavior-stats graph implementation used by the
    trained model.
    """
    log_variant_cache = {}
    raw_pairs = []
    targets = []

    print("\nPreparing fixed real-life graph set...")

    for i, record in enumerate(
        records,
        start=1,
    ):
        log_path = Path(
            record["log_path"]
        )

        model_path = Path(
            record["model_path"]
        )

        if log_path not in log_variant_cache:
            print(
                f"  Reading log: "
                f"{log_path.name}"
            )

            log_variant_cache[
                log_path
            ] = read_log_as_variants(
                log_path
            )

        variants = log_variant_cache[
            log_path
        ]

        net, im, fm = pnml_importer.apply(
            str(model_path)
        )

        raw_pairs.append(
            (
                variants,
                net,
                im,
                fm,
            )
        )

        targets.append(
            (
                record["fitness"],
                record["precision"],
            )
        )

        print(
            f"  Prepared "
            f"{i:02d}/{len(records):02d}: "
            f"{record['log_id']} | "
            f"{record['model_name']}"
        )

    specs = prepare_pair_specs(
        raw_pairs,
        precomputed_targets=targets,
        compute_targets_if_missing=False,
    )

    dataset = AlignmentPairDataset(
        specs,
        variant="A",
        random_permute_onehot=False,
        trace_freq_mode="normalized",
        prebuild_graphs=True,
    )

    print(
        f"Finished building "
        f"{len(dataset)} real-life graphs.\n"
    )

    return dataset


# =============================================================================
# Model and evaluation helpers
# =============================================================================


def load_model(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model_config = checkpoint.get(
        "model_config",
        DEFAULT_MODEL_CONFIG,
    )

    model = FitnessPrecisionPredictor(
        **model_config
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_dataset(
    model,
    dataset,
    batch_size: int,
    device: torch.device,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_alignment_pairs,
    )

    predictions = []
    targets = []

    model.eval()

    with torch.inference_mode():
        for batch in loader:

            if len(batch) == 3:
                (
                    g_batch,
                    y_batch,
                    _sample_weight,
                ) = batch

            else:
                (
                    g_batch,
                    y_batch,
                ) = batch

            g_batch = g_batch.to(
                device
            )

            output = model(
                g_batch
            )

            pred = output["pred"]

            predictions.append(
                pred.detach().cpu()
            )

            targets.append(
                y_batch.detach().cpu()
            )

    preds = torch.cat(
        predictions,
        dim=0,
    ).numpy()

    tgts = torch.cat(
        targets,
        dim=0,
    ).numpy()

    fit_mae = float(
        np.mean(
            np.abs(
                preds[:, 0]
                - tgts[:, 0]
            )
        )
    )

    prec_mae = float(
        np.mean(
            np.abs(
                preds[:, 1]
                - tgts[:, 1]
            )
        )
    )

    mean_mae = (
        0.5
        * (
            fit_mae
            + prec_mae
        )
    )

    return (
        fit_mae,
        prec_mae,
        mean_mae,
    )


# =============================================================================
# Summary statistics
# =============================================================================


def ci95_half_width(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n < 2:
        return float("nan")

    sd = float(
        np.std(
            values,
            ddof=1,
        )
    )

    se = sd / math.sqrt(n)

    if student_t is not None:
        critical = float(
            student_t.ppf(
                0.975,
                df=n - 1,
            )
        )
    else:
        critical = 1.96

    return critical * se


def summarize_results(
    results_df: pd.DataFrame,
):
    rows = []

    for keys, group in results_df.groupby(
        [
            "test_set",
            "size_label",
        ],
        sort=False,
    ):
        (
            test_set,
            size_label,
        ) = keys

        n_train = int(
            round(
                group[
                    "n_train"
                ].mean()
            )
        )

        effective_fraction = float(
            group[
                "effective_fraction"
            ].mean()
        )

        row = {
            "test_set":
                test_set,

            "size_label":
                size_label,

            "n_train":
                n_train,

            "effective_fraction":
                effective_fraction,

            "effective_fraction_pct":
                effective_fraction * 100.0,

            "n_runs":
                len(group),
        }

        for metric in [
            "fitness_mae",
            "precision_mae",
            "mean_mae",
        ]:
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            row[
                f"{metric}_std"
            ] = (
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if len(values) > 1
                else float("nan")
            )

            row[
                f"{metric}_ci95"
            ] = ci95_half_width(
                values
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "test_set",
                "n_train",
            ],
            ignore_index=True,
        )
    )


# =============================================================================
# Main learning-curve evaluation
# =============================================================================


def evaluate_learning_curve():
    print("=" * 80)
    print("LEARNING-CURVE EVALUATION")
    print("=" * 80)

    print(
        f"Device                 : "
        f"{DEVICE}"
    )

    print(
        f"Model root             : "
        f"{MODEL_ROOT}"
    )

    print(
        f"Synthetic test CSV     : "
        f"{SYNTHETIC_TEST_CSV}"
    )

    print(
        f"Synthetic graph cache  : "
        f"{SYNTHETIC_GRAPH_CACHE_DIR}"
    )

    print(
        f"Real-life CSV          : "
        f"{REAL_LIFE_CSV}"
    )

    print(
        f"Real-life fitness      : "
        f"{REAL_LIFE_FITNESS_COLUMN}"
    )

    print(
        f"Real-life precision    : "
        f"{REAL_LIFE_PRECISION_COLUMN}"
    )

    print(
        f"Training sizes         : "
        f"{TRAIN_SIZES}"
    )

    print(
        f"Seeds                  : "
        f"{SEEDS}"
    )

    print("=" * 80)

    if not SYNTHETIC_TEST_CSV.exists():
        raise FileNotFoundError(
            SYNTHETIC_TEST_CSV
        )

    if not SYNTHETIC_GRAPH_CACHE_DIR.exists():
        raise FileNotFoundError(
            "Synthetic graph cache not found: "
            f"{SYNTHETIC_GRAPH_CACHE_DIR}"
        )

    if not REAL_LIFE_CSV.exists():
        raise FileNotFoundError(
            REAL_LIFE_CSV
        )

    # -----------------------------------------------------------------
    # Fixed synthetic test set.
    # -----------------------------------------------------------------

    synthetic_records = load_split_records(
        SYNTHETIC_TEST_CSV
    )

    synthetic_test_ds = CachedVariantADataset(
        synthetic_records,
        cache_dir=SYNTHETIC_GRAPH_CACHE_DIR,
        label="synthetic-test",
    )

    print(
        f"\nSynthetic test pairs: "
        f"{len(synthetic_test_ds):,}"
    )

    # -----------------------------------------------------------------
    # Fixed real-life test set.
    # -----------------------------------------------------------------

    real_life_records = build_real_life_records(
        REAL_LIFE_CSV
    )

    real_life_ds = build_real_life_dataset(
        real_life_records
    )

    # -----------------------------------------------------------------
    # Evaluate every learning-curve checkpoint.
    # -----------------------------------------------------------------

    rows = []

    for seed in SEEDS:

        for train_size_request in TRAIN_SIZES:

            size_label = make_size_label(
                train_size_request
            )

            checkpoint_path = (
                MODEL_ROOT
                / f"size_{size_label}"
                / f"seed_{seed}"
                / "model_best_restored.pt"
            )

            if not checkpoint_path.exists():
                print(
                    "\nSkipping missing checkpoint:"
                )
                print(
                    f"  {checkpoint_path}"
                )
                continue

            print(
                "\n"
                + "=" * 80
            )

            print(
                f"Training size = "
                f"{train_size_request} | "
                f"seed = {seed}"
            )

            print(
                checkpoint_path
            )

            model, checkpoint = load_model(
                checkpoint_path,
                DEVICE,
            )

            n_train = checkpoint.get(
                "num_train_used",
                np.nan,
            )

            effective_fraction = checkpoint.get(
                "effective_train_fraction",
                np.nan,
            )

            print(
                f"Actual n_train        : "
                f"{n_train}"
            )

            print(
                f"Effective fraction    : "
                f"{effective_fraction}"
            )

            if np.isfinite(
                float(effective_fraction)
            ):
                print(
                    f"Effective percentage  : "
                    f"{float(effective_fraction) * 100:.2f}%"
                )

            # ---------------------------------------------------------
            # Synthetic test set
            # ---------------------------------------------------------

            (
                syn_fit,
                syn_prec,
                syn_mean,
            ) = evaluate_dataset(
                model,
                synthetic_test_ds,
                BATCH_SIZE_SYNTHETIC,
                DEVICE,
            )

            # ---------------------------------------------------------
            # Real-life test set
            # ---------------------------------------------------------

            (
                real_fit,
                real_prec,
                real_mean,
            ) = evaluate_dataset(
                model,
                real_life_ds,
                BATCH_SIZE_REAL_LIFE,
                DEVICE,
            )

            common = {
                "requested_train_size":
                    checkpoint.get(
                        "requested_train_size",
                        str(train_size_request),
                    ),

                "size_label":
                    size_label,

                "seed":
                    seed,

                "n_train":
                    n_train,

                "effective_fraction":
                    effective_fraction,

                "best_epoch":
                    checkpoint.get(
                        "best_epoch",
                        np.nan,
                    ),

                "best_valid_loss":
                    checkpoint.get(
                        "best_valid_loss",
                        np.nan,
                    ),
            }

            rows.append(
                {
                    **common,

                    "test_set":
                        "synthetic",

                    "fitness_mae":
                        syn_fit,

                    "precision_mae":
                        syn_prec,

                    "mean_mae":
                        syn_mean,
                }
            )

            rows.append(
                {
                    **common,

                    "test_set":
                        "real_life",

                    "fitness_mae":
                        real_fit,

                    "precision_mae":
                        real_prec,

                    "mean_mae":
                        real_mean,
                }
            )

            print(
                f"Synthetic | "
                f"fit={syn_fit:.6f} "
                f"prec={syn_prec:.6f} "
                f"mean={syn_mean:.6f}"
            )

            print(
                f"Real life | "
                f"fit={real_fit:.6f} "
                f"prec={real_prec:.6f} "
                f"mean={real_mean:.6f}"
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        rows
    )

    if results_df.empty:
        raise RuntimeError(
            "No checkpoints were evaluated."
        )

    summary_df = summarize_results(
        results_df
    )

    if SAVE_RESULTS:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_df.to_csv(
            OUTPUT_DIR
            / "learning_curve_runs.csv",
            index=False,
        )

        summary_df.to_csv(
            OUTPUT_DIR
            / "learning_curve_summary.csv",
            index=False,
        )

        print(
            "\nSaved CSV results to: "
            f"{OUTPUT_DIR.resolve()}"
        )

    return (
        results_df,
        summary_df,
    )


# =============================================================================
# Plotting
# =============================================================================


def plot_learning_curves(
    summary_df: pd.DataFrame,
    *,
    show: bool = SHOW_PLOTS,
    save: bool = SAVE_PLOTS,
    uncertainty: str = UNCERTAINTY_BAND,
):
    if uncertainty not in {
        "std",
        "ci95",
    }:
        raise ValueError(
            "uncertainty must be "
            "'std' or 'ci95'."
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.6),
    )

    panels = [
        (
            "synthetic",
            "Synthetic test set",
        ),
        (
            "real_life",
            "Real-life test set",
        ),
    ]

    metrics = [
        (
            "fitness_mae",
            "Fitness MAE",
        ),
        (
            "precision_mae",
            "Precision MAE",
        ),
        (
            "mean_mae",
            "Mean MAE",
        ),
    ]

    for (
        ax,
        (
            test_set,
            title,
        ),
    ) in zip(
        axes,
        panels,
    ):

        panel = (
            summary_df[
                summary_df[
                    "test_set"
                ] == test_set
            ]
            .sort_values(
                "n_train"
            )
        )

        x = panel[
            "n_train"
        ].to_numpy(
            dtype=float
        )

        for (
            metric,
            label,
        ) in metrics:

            y = panel[
                f"{metric}_mean"
            ].to_numpy(
                dtype=float
            )

            band = panel[
                f"{metric}_{uncertainty}"
            ].to_numpy(
                dtype=float
            )

            line = ax.plot(
                x,
                y,
                marker="o",
                label=label,
            )[0]

            ax.fill_between(
                x,
                y - band,
                y + band,
                alpha=0.15,
                color=line.get_color(),
            )

        tick_labels = [
            (
                f"{int(n):,}\n"
                f"({pct:.1f}%)"
            )
            for n, pct in zip(
                panel["n_train"],
                panel[
                    "effective_fraction_pct"
                ],
            )
        ]

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            tick_labels
        )

        ax.set_xlabel(
            "Training samples"
        )

        ax.set_ylabel(
            "MAE"
        )

        ax.set_title(
            title
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend()

    band_label = (
        "95% CI"
        if uncertainty == "ci95"
        else "±1 SD"
    )

    fig.suptitle(
        "Learning curves "
        "(mean across seeds; "
        f"shaded band = {band_label})"
    )

    fig.tight_layout()

    if save:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        png_path = (
            OUTPUT_DIR
            / "learning_curve_mae.png"
        )

        pdf_path = (
            OUTPUT_DIR
            / "learning_curve_mae.pdf"
        )

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        print(
            "Saved plot to",
            png_path.resolve(),
        )

        print(
            "Saved plot to",
            pdf_path.resolve(),
        )

    if show:
        plt.show()
    else:
        plt.close(
            fig
        )

    return fig


# =============================================================================
# Entry point
# =============================================================================


def main():
    (
        results_df,
        summary_df,
    ) = evaluate_learning_curve()

    print(
        "\nPer-run results"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSummary"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    plot_learning_curves(
        summary_df
    )

    return (
        results_df,
        summary_df,
    )


if __name__ == "__main__":
    main()