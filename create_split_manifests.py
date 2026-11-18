import csv
import random
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("data_prep/local_dataset_clean1")
OUTPUT_ROOT = Path("splits_clean")

SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

RANDOM_SEED = 43
GROUP_COLUMN = "split_group"


# ============================================================
# Split helpers
# ============================================================

def assign_groups_to_splits(groups, split_ratios, random_seed):
    rng = random.Random(random_seed)

    groups = list(groups)
    rng.shuffle(groups)

    n = len(groups)

    n_train = int(round(n * split_ratios["train"]))
    n_val = int(round(n * split_ratios["val"]))

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    group_to_split = {}

    for group in train_groups:
        group_to_split[group] = "train"

    for group in val_groups:
        group_to_split[group] = "val"

    for group in test_groups:
        group_to_split[group] = "test"

    return group_to_split


def create_split_manifests(
    dataset_dir,
    output_dir,
    split_mode,
    split_ratios=None,
    group_column="model_name",
    random_seed=42,
):
    """
    Create train/val/test CSV manifests only.

    No XES or PNML files are copied.

    split_mode="group":
        all rows with the same group_column stay in the same split.

    split_mode="random":
        rows are shuffled directly and then split.
    """

    if split_ratios is None:
        split_ratios = {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
        }

    if split_mode not in {"group", "random"}:
        raise ValueError('split_mode must be either "group" or "random".')

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    results_path = dataset_dir / "results.csv"

    if not results_path.exists():
        raise FileNotFoundError(results_path)

    # --------------------------------------------------------
    # Read canonical results.csv
    # --------------------------------------------------------

    with results_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames

    if not rows:
        raise ValueError("results.csv has no rows.")

    if columns is None:
        raise ValueError("Could not read CSV columns.")

    if "model_name" not in columns:
        raise ValueError(
            "results.csv must contain model_name column."
        )

    if "log_name" not in columns:
        raise ValueError(
            "results.csv must contain log_name column."
        )

    if split_mode == "group" and group_column not in columns:
        raise ValueError(
            f"group_column does not exist: {group_column}"
        )

    # --------------------------------------------------------
    # Attach stable graph-cache index
    # --------------------------------------------------------
    #
    # cache_index is the row position in the canonical
    # final_dataset/results.csv.
    #
    # Graph cache entry N corresponds to cache_index N.
    # --------------------------------------------------------

    indexed_rows = []

    for cache_index, row in enumerate(rows):
        row_copy = dict(row)
        row_copy["cache_index"] = str(cache_index)
        indexed_rows.append(row_copy)

    output_columns = list(columns)

    if "cache_index" not in output_columns:
        output_columns.append("cache_index")

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    rng = random.Random(random_seed)

    if split_mode == "random":

        shuffled_rows = list(indexed_rows)
        rng.shuffle(shuffled_rows)

        n = len(shuffled_rows)

        n_train = int(round(n * split_ratios["train"]))
        n_val = int(round(n * split_ratios["val"]))

        split_rows = {
            "train": shuffled_rows[:n_train],
            "val": shuffled_rows[
                n_train:n_train + n_val
            ],
            "test": shuffled_rows[
                n_train + n_val:
            ],
        }

    else:

        groups = sorted({
            row[group_column]
            for row in indexed_rows
        })

        group_to_split = assign_groups_to_splits(
            groups=groups,
            split_ratios=split_ratios,
            random_seed=random_seed,
        )

        split_rows = {
            "train": [],
            "val": [],
            "test": [],
        }

        for row in indexed_rows:
            split_name = group_to_split[
                row[group_column]
            ]

            split_rows[split_name].append(row)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    total_split_rows = sum(
        len(split_rows[name])
        for name in ["train", "val", "test"]
    )

    if total_split_rows != len(indexed_rows):
        raise RuntimeError(
            f"Split row count mismatch: "
            f"{total_split_rows} != {len(indexed_rows)}"
        )

    cache_indices = []

    for split_name in ["train", "val", "test"]:
        cache_indices.extend(
            int(row["cache_index"])
            for row in split_rows[split_name]
        )

    if len(cache_indices) != len(set(cache_indices)):
        raise RuntimeError(
            "Duplicate cache_index found across splits."
        )

    if set(cache_indices) != set(range(len(indexed_rows))):
        raise RuntimeError(
            "Some canonical dataset rows are missing "
            "from the split manifests."
        )

    # --------------------------------------------------------
    # Save manifests
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name in ["train", "val", "test"]:

        split_path = (
            output_dir / f"{split_name}.csv"
        )

        with split_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=output_columns,
            )

            writer.writeheader()
            writer.writerows(
                split_rows[split_name]
            )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    total = len(indexed_rows)

    print("=" * 70)
    print(f"Split mode     : {split_mode}")
    print(f"Random seed    : {random_seed}")

    if split_mode == "group":
        print(f"Group column   : {group_column}")
        print(f"Unique groups  : {len(groups)}")

    print(f"Total rows     : {total}")
    print()

    for split_name in ["train", "val", "test"]:

        count = len(split_rows[split_name])
        percentage = 100.0 * count / total

        print(
            f"{split_name:5s}: "
            f"{count:6d} rows "
            f"({percentage:.2f}%)"
        )

    print()
    print(f"Saved to: {output_dir}")
    print("=" * 70)
    print()


# ============================================================
# Main
# ============================================================

def main():

    # Grouped split
    create_split_manifests(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_ROOT / "grouped",
        split_mode="group",
        split_ratios=SPLIT_RATIOS,
        group_column=GROUP_COLUMN,
        random_seed=RANDOM_SEED,
    )

    # Random split
    create_split_manifests(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_ROOT / "random",
        split_mode="random",
        split_ratios=SPLIT_RATIOS,
        group_column=GROUP_COLUMN,
        random_seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    main()
