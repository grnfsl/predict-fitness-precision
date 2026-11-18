import gc
import hashlib
import os
import pickle
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer

from graph import (
    prepare_pair_spec,
    build_alignment_pair_graph,
)

from filelock import FileLock

def read_log_as_variants(log_path, activity_key="concept:name"):
    """
    Converts a .xes file to a compact variant representation:
        [
            (["a", "b", "c"], 13),
            (["a", "c", "b"], 7),
            ...
        ]
    """
    log = xes_importer.apply(
        str(log_path),
        parameters={
            "show_progress_bar": False
        },
    )

    counter = Counter()

    for trace_idx, trace in enumerate(log):
        seq = []
        for event in trace:
            if activity_key not in event:
                raise KeyError(
                    f"Activity key {activity_key!r} not found in event. "
                    f"Available keys: {list(event.keys())}"
                )
            seq.append(str(event[activity_key]))
        counter[tuple(seq)] += 1

    variants = sorted(
        [(list(seq), int(count)) for seq, count in counter.items()],
        key=lambda x: (-x[1], tuple(x[0])),
    )

    # del log
    # gc.collect()

    return variants


def _variant_cache_path(log_path, cache_dir):
    """
    Build a cache filename that avoids collisions when two folders contain files
    with the same stem and invalidates automatically if the source file changes.
    """
    log_path = Path(log_path)
    stat = log_path.stat()
    digest = hashlib.sha1(str(log_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / f"{log_path.stem}.{digest}.{stat.st_size}.{stat.st_mtime_ns}.variants.pkl"


def read_log_as_variants_cached_old(log_path, cache_dir=None): # unable to handle multiple caches with multiple processes
    """
    Parses XES once, saves compact variants, and reuses them in later epochs.
    """
    log_path = Path(log_path)

    if cache_dir is None:
        return read_log_as_variants(log_path)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = _variant_cache_path(log_path, cache_dir)

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    variants = read_log_as_variants(log_path)

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(variants, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)

    return variants

def read_log_as_variants_cached(log_path, cache_dir=None):
    """
    Parses XES once, saves compact variants, and reuses them in later epochs.

    Cache creation is protected by a file lock so multiple Slurm jobs/processes
    can safely share the same cache directory.
    """
    log_path = Path(log_path)

    if cache_dir is None:
        return read_log_as_variants(log_path)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = _variant_cache_path(log_path, cache_dir)
    lock_path = Path(str(cache_path) + ".lock")

    # Fast path: already cached
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Only one process/job may create this particular cache file at a time.
    with FileLock(str(lock_path)):

        # Another process may have created it while we waited.
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        variants = read_log_as_variants(log_path)

        tmp_path = Path(
            str(cache_path) + f".{os.getpid()}.tmp"
        )

        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(
                    variants,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            tmp_path.replace(cache_path)

        finally:
            # Clean up if writing/renaming failed.
            if tmp_path.exists():
                tmp_path.unlink()

        return variants

def _target_bin(value, num_bins=10):
    """
    Return the bin index for values in [0, 1].

    Bins are:
        [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    so the exact value 1.0 is placed in the last bin.
    """
    value = float(value)
    if value >= 1.0:
        return num_bins - 1
    if value <= 0.0:
        return 0
    return max(0, min(num_bins - 1, int(value * num_bins)))


def print_bin_weight_grid(counts, num_bins=10, min_weight=0.25, max_weight=5.0):
    """
    Print 10x10 grids showing:
      1. number of samples in each fitness/precision bin
      2. weight assigned to each bin

    Rows    = fitness bins
    Columns = precision bins
    """

    non_empty_counts = list(counts.values())
    mean_count = sum(non_empty_counts) / max(1, len(non_empty_counts))

    print("\nBin count grid")
    print("Rows = fitness bins, Columns = precision bins")
    print("      " + " ".join([f"p{j}" for j in range(num_bins)]))

    for fb in range(num_bins):
        row = []
        for pb in range(num_bins):
            row.append(f"{counts.get((fb, pb), 0):4d}")
        print(f"f{fb}  " + " ".join(row))

    print("\nBin weight grid")
    print("Rows = fitness bins, Columns = precision bins")
    print("      " + " ".join([f"p{j}" for j in range(num_bins)]))

    for fb in range(num_bins):
        row = []
        for pb in range(num_bins):
            c = counts.get((fb, pb), 0)
            if c == 0:
                row.append("   .")
            else:
                w = mean_count / float(c)
                w = max(float(min_weight), min(float(max_weight), w))
                row.append(f"{w:4.1f}")
        print(f"f{fb}  " + " ".join(row))


def add_bin_weights_to_records(
    records,
    *,
    fitness_key="fitness",
    precision_key="precision",
    weight_key="sample_weight",
    fitness_bin_key="fitness_bin",
    precision_bin_key="precision_bin",
    num_bins=10,
    min_weight=0.25,
    max_weight=5.0,
    inplace=False,
    print_summary=True,
    print_grid=True,
):
    """
    Add one bin-based sample weight to every training record.

    The weights are computed from the training records only. Each record is
    assigned to a 2D bin using its true fitness and precision. Rare bins get
    larger weights; common bins get smaller weights.

    Weight formula:
        weight = mean_non_empty_bin_count / count_of_sample_bin

    The result is clipped to [min_weight, max_weight].
    """
    if not records:
        return records if inplace else []

    weighted_records = records if inplace else [dict(record) for record in records]

    counts = [[0 for _ in range(num_bins)] for _ in range(num_bins)]

    for record in weighted_records:
        fbin = _target_bin(record[fitness_key], num_bins=num_bins)
        pbin = _target_bin(record[precision_key], num_bins=num_bins)
        record[fitness_bin_key] = fbin
        record[precision_bin_key] = pbin
        counts[fbin][pbin] += 1

    non_empty_counts = [
        counts[i][j]
        for i in range(num_bins)
        for j in range(num_bins)
        if counts[i][j] > 0
    ]

    if not non_empty_counts:
        mean_count = 1.0
    else:
        mean_count = sum(non_empty_counts) / float(len(non_empty_counts))

    for record in weighted_records:
        fbin = record[fitness_bin_key]
        pbin = record[precision_bin_key]
        bin_count = max(1, counts[fbin][pbin])
        weight = mean_count / float(bin_count)
        weight = max(float(min_weight), min(float(max_weight), weight))
        record[weight_key] = float(weight)

    if print_summary:
        weights = [float(record[weight_key]) for record in weighted_records]
        print("Bin-weighted training records")
        print(f"  records: {len(weighted_records)}")
        print(f"  non-empty bins: {len(non_empty_counts)} / {num_bins * num_bins}")
        print(f"  mean non-empty bin count: {mean_count:.2f}")
        print(f"  min weight: {min(weights):.4f}")
        print(f"  max weight: {max(weights):.4f}")

    if print_grid:
        def _bin_label(i):
            left = i / num_bins
            right = (i + 1) / num_bins
            return f"[{left:.1f},{right:.1f})" if i < num_bins - 1 else f"[{left:.1f},{right:.1f}]"

        fitness_labels = [_bin_label(i) for i in range(num_bins)]

        print("\nBin count grid")
        print("fitness bins")
        print("precision bins")
        print()

        header = "precision \\ fitness".ljust(18)
        for label in fitness_labels:
            header += f"{label:>11s}"
        print(header)

        # Print precision from high to low, like the MAE heatmap
        for pbin in reversed(range(num_bins)):
            row_label = _bin_label(pbin).ljust(18)
            row = row_label

            for fbin in range(num_bins):
                row += f"{counts[fbin][pbin]:11d}"

            print(row)

        print("\nBin weight grid")
        print("fitness bins")
        print("precision bins")
        print()

        header = "precision \\ fitness".ljust(18)
        for label in fitness_labels:
            header += f"{label:>11s}"
        print(header)

        for pbin in reversed(range(num_bins)):
            row_label = _bin_label(pbin).ljust(18)
            row = row_label

            for fbin in range(num_bins):
                bin_count = counts[fbin][pbin]

                if bin_count == 0:
                    cell = "."
                else:
                    weight = mean_count / float(bin_count)
                    weight = max(float(min_weight), min(float(max_weight), weight))
                    cell = f"{weight:.1f}"

                row += f"{cell:>11s}"

            print(row)

    return weighted_records


def add_unit_weights_to_records(
    records,
    *,
    weight_key="sample_weight",
    inplace=False,
):
    """
    Add sample_weight=1.0 to records.

    Useful for validation/test records so the trainer can receive a weight
    tensor while validation remains unweighted.
    """
    if records is None:
        return None

    out = records if inplace else [dict(record) for record in records]
    for record in out:
        record[weight_key] = 1.0
    return out


class LazyAlignmentPairDataset(Dataset):
    """
    Keeps only file paths + targets in memory.

    __getitem__ loads one XES/PNML pair, converts the XES log to variants,
    builds one graph, returns it, and then releases intermediate objects.
    """

    def __init__(
        self,
        records,
        variant="A",
        max_local_onehot_dim=None,
        random_permute_onehot=False,
        trace_freq_mode="normalized",
        variant_cache_dir=".cache_variants",
    ):
        self.records = list(records)
        self.variant = variant.upper()
        self.max_local_onehot_dim = max_local_onehot_dim
        self.random_permute_onehot = bool(random_permute_onehot)
        self.trace_freq_mode = trace_freq_mode
        self.variant_cache_dir = variant_cache_dir

        if self.variant not in {"A", "C"}:
            raise ValueError("variant must be 'A' or 'C'")

        if self.variant == "C" and self.max_local_onehot_dim is None:
            raise ValueError(
                "Variant C needs max_local_onehot_dim. Compute it once with "
                "compute_max_local_activity_dim_from_records(train_records + valid_records)."
            )

        if self.variant == "A":
            self.max_local_onehot_dim = 0

    def __len__(self):
        return len(self.records)

    ################ Debugging code. Remove it and uncomment the other one for actual run ################

    def __getitem__(self, idx):
        import time

        record = self.records[idx]
        timing = {}

        t0 = time.perf_counter()
        log_variants = read_log_as_variants_cached(
            record["log_path"],
            cache_dir=self.variant_cache_dir,
        )
        timing["read_variants"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        net, im, fm = pnml_importer.apply(str(record["model_path"]))
        timing["load_pnml"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        spec = prepare_pair_spec(
            log_df=log_variants,
            net=net,
            im=im,
            fm=fm,
            fitness=float(record["fitness"]),
            precision=float(record["precision"]),
            compute_targets_if_missing=False,
        )
        timing["prepare_spec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        graph = build_alignment_pair_graph(
            spec=spec,
            variant=self.variant,
            max_local_onehot_dim=self.max_local_onehot_dim,
            random_permute_onehot=self.random_permute_onehot,
            trace_freq_mode=self.trace_freq_mode,
            rng=None,
        )
        timing["build_graph"] = time.perf_counter() - t0

        target = torch.tensor([spec.fitness, spec.precision], dtype=torch.float32)
        sample_weight = torch.tensor(
            float(record.get("sample_weight", 1.0)),
            dtype=torch.float32,
        )

        # print only occasionally
        # if idx % 200 == 0:
        #     total = sum(timing.values())
        #     print(f"[LazyDataset idx={idx}] total={total:.3f}s | " +
        #           " | ".join(f"{k}={v:.3f}s" for k, v in timing.items()))

        # del log_variants, net, im, fm, spec
        # gc.collect()

        return graph, target, sample_weight

    ########################################################################

    # def __getitem__(self, idx):
    #     record = self.records[idx]
    #
    #     log_variants = read_log_as_variants_cached(
    #         record["log_path"],
    #         cache_dir=self.variant_cache_dir,
    #     )
    #
    #     net, im, fm = pnml_importer.apply(str(record["model_path"]))
    #
    #     spec = prepare_pair_spec(
    #         log_df=log_variants,
    #         net=net,
    #         im=im,
    #         fm=fm,
    #         fitness=float(record["fitness"]),
    #         precision=float(record["precision"]),
    #         compute_targets_if_missing=False,
    #     )
    #
    #     graph = build_alignment_pair_graph(
    #         spec=spec,
    #         variant=self.variant,
    #         max_local_onehot_dim=self.max_local_onehot_dim,
    #         random_permute_onehot=self.random_permute_onehot,
    #         trace_freq_mode=self.trace_freq_mode,
    #         rng=None,
    #     )
    #
    #     target = torch.tensor([spec.fitness, spec.precision], dtype=torch.float32)
    #     sample_weight = torch.tensor(
    #         float(record.get("sample_weight", 1.0)),
    #         dtype=torch.float32,
    #     )
    #
    #     del log_variants, net, im, fm, spec
    #     gc.collect()
    #
    #     return graph, target, sample_weight


# def build_records_from_csv(
#     logs_dir,
#     models_dir,
#     csv_path,
#     test_size=400,
#     seed=42,
#     model_col="model (.pnml)",
#     fitness_col="fitness",
#     precision_col="precision",
# ):
#     logs_dir = Path(logs_dir)
#     models_dir = Path(models_dir)
#     csv_path = Path(csv_path)
#
#     df = pd.read_csv(csv_path)
#
#     df["stem"] = df[model_col].apply(
#         lambda x: Path(str(x).strip()).stem
#     )
#
#     xes_files = {
#         path.stem.strip(): path
#         for path in logs_dir.rglob("*.xes")
#     }
#
#     pnml_files = {
#         path.stem.strip(): path
#         for path in models_dir.rglob("*.pnml")
#     }
#
#     records = []
#     missing_logs = []
#     missing_models = []
#
#     for _, row in df.iterrows():
#         name = row["stem"]
#
#         if name not in xes_files:
#             missing_logs.append(name)
#             continue
#
#         if name not in pnml_files:
#             missing_models.append(name)
#             continue
#
#         records.append({
#             "name": name,
#             "log_path": xes_files[name],
#             "model_path": pnml_files[name],
#             "fitness": float(row[fitness_col]),
#             "precision": float(row[precision_col]),
#         })
#
#     print("Valid records:", len(records))
#     print("Missing logs:", len(missing_logs))
#     print("Missing models:", len(missing_models))
#
#     if missing_logs:
#         print("First missing logs:", missing_logs[:10])
#
#     if missing_models:
#         print("First missing models:", missing_models[:10])
#
#     rng = random.Random(seed)
#     rng.shuffle(records)
#
#     test_records = records[:test_size]
#     train_records = records[test_size:]
#
#     print("Train records:", len(train_records))
#     print("Test records:", len(test_records))
#
#     return train_records, test_records

import random
import re
from pathlib import Path

import pandas as pd


def infer_group_id(model_name, log_name):
    """
    Infer source/base model id for group split.

    Examples:
        model_0001.pnml                  -> model_0001
        model_0001_clean_001.xes         -> model_0001
        base_000001_thr_00.pnml          -> base_000001
        base_000001_mixed_001.xes        -> base_000001
        im_run_0001_base_000001_thr_00   -> base_000001
    """

    text = f"{model_name} {log_name}"

    match = re.search(r"(base_\d+)", text)
    if match:
        return match.group(1)

    match = re.search(r"(model_\d+)", text)
    if match:
        return match.group(1)

    return Path(str(model_name)).stem


def resolve_test_count(test_size, total):
    """
    test_size can be:
        400  -> exact number of test records/groups
        0.2  -> 20%
    """

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Float test_size must be between 0 and 1.")
        return round(total * test_size)

    return int(test_size)


def build_records_from_csv(
    logs_dir,
    models_dir,
    csv_path,
    test_size=0.2,
    seed=42,
    model_col="model (.pnml)",
    model_name_col="model_name",
    log_name_col="log_name",
    targets=None,
    split_by_group=True,
):
    logs_dir = Path(logs_dir)
    models_dir = Path(models_dir)
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    # Old format:
    # model (.pnml), fitness, precision
    has_old_format = model_col in df.columns

    # New format:
    # model_name, log_name, log_fitness, precision, ...
    has_new_format = (
        model_name_col in df.columns
        and log_name_col in df.columns
    )

    if not has_old_format and not has_new_format:
        raise ValueError(
            "CSV must contain either:\n"
            f"  old format: {model_col!r}\n"
            "or\n"
            f"  new format: {model_name_col!r}, {log_name_col!r}"
        )

    # Default target mapping.
    # Format:
    #     "CSV column" -> "record key"
    if targets is None:
        if "log_fitness" in df.columns:
            targets = {
                "log_fitness": "fitness",
                "precision": "precision",
            }
        else:
            targets = {
                "fitness": "fitness",
                "precision": "precision",
            }

    missing_target_cols = [
        csv_col for csv_col in targets
        if csv_col not in df.columns
    ]

    if missing_target_cols:
        raise ValueError(
            f"Missing target columns in CSV: {missing_target_cols}"
        )

    xes_files = {
        path.stem.strip(): path
        for path in logs_dir.rglob("*.xes")
    }

    pnml_files = {
        path.stem.strip(): path
        for path in models_dir.rglob("*.pnml")
    }

    records = []
    missing_logs = []
    missing_models = []

    for _, row in df.iterrows():

        if has_old_format:
            model_key = Path(str(row[model_col]).strip()).stem
            log_key = model_key
            name = model_key

        else:
            model_key = Path(str(row[model_name_col]).strip()).stem
            log_key = Path(str(row[log_name_col]).strip()).stem
            name = f"{model_key}__{log_key}"

        if log_key not in xes_files:
            missing_logs.append(log_key)
            continue

        if model_key not in pnml_files:
            missing_models.append(model_key)
            continue

        group_id = infer_group_id(model_key, log_key)

        record = {
            "name": name,
            "group_id": group_id,
            "log_path": xes_files[log_key],
            "model_path": pnml_files[model_key],
        }

        for csv_col, record_key in targets.items():
            record[record_key] = float(row[csv_col])

        records.append(record)

    print("Valid records:", len(records))
    print("Missing logs:", len(missing_logs))
    print("Missing models:", len(missing_models))

    if missing_logs:
        print("First missing logs:", missing_logs[:10])

    if missing_models:
        print("First missing models:", missing_models[:10])

    rng = random.Random(seed)

    if split_by_group:
        groups = sorted({record["group_id"] for record in records})
        rng.shuffle(groups)

        test_group_count = resolve_test_count(test_size, len(groups))
        test_groups = set(groups[:test_group_count])

        test_records = [
            record for record in records
            if record["group_id"] in test_groups
        ]

        train_records = [
            record for record in records
            if record["group_id"] not in test_groups
        ]

        rng.shuffle(train_records)
        rng.shuffle(test_records)

        print("Split mode: group split")
        print("Train groups:", len(set(r["group_id"] for r in train_records)))
        print("Test groups:", len(set(r["group_id"] for r in test_records)))

    else:
        rng.shuffle(records)

        test_count = resolve_test_count(test_size, len(records))
        test_records = records[:test_count]
        train_records = records[test_count:]

        print("Split mode: random row split")

    print("Train records:", len(train_records))
    print("Test records:", len(test_records))

    return train_records, test_records


def build_records_from_csv_old(
    logs_dir,
    models_dir,
    csv_path,
    test_size=400,
    seed=42,
    model_col="model (.pnml)",
    model_name_col="model_name",
    log_name_col="log_name",
    fitness_col="fitness",
    precision_col="precision",
):
    logs_dir = Path(logs_dir)
    models_dir = Path(models_dir)
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    # Existing behaviour:
    # CSV has: model (.pnml), fitness, precision
    # In this mode, the log and model are matched using the same stem.
    has_old_format = model_col in df.columns

    # New behaviour:
    # CSV has: model_name, log_name, fitness, precision
    # In this mode, model_name is looked up in models_dir,
    # and log_name is looked up in logs_dir.
    has_new_format = model_name_col in df.columns and log_name_col in df.columns

    if not has_old_format and not has_new_format:
        raise ValueError(
            "CSV must contain either:\n"
            f"  old format: {model_col!r}, {fitness_col!r}, {precision_col!r}\n"
            "or\n"
            f"  new format: {model_name_col!r}, {log_name_col!r}, "
            f"{fitness_col!r}, {precision_col!r}"
        )

    xes_files = {
        path.stem.strip(): path
        for path in logs_dir.rglob("*.xes")
    }

    pnml_files = {
        path.stem.strip(): path
        for path in models_dir.rglob("*.pnml")
    }

    records = []
    missing_logs = []
    missing_models = []

    for _, row in df.iterrows():
        if has_old_format:
            # Old CSV format:
            # model (.pnml), fitness, precision
            #
            # Preserve previous behaviour: use the model stem as both
            # the model name and the log name.
            model_key = Path(str(row[model_col]).strip()).stem
            log_key = model_key
            name = model_key

        else:
            # New CSV format:
            # model_name, log_name, fitness, precision
            #
            # model_name is resolved inside models_dir.
            # log_name is resolved inside logs_dir.
            model_key = Path(str(row[model_name_col]).strip()).stem
            log_key = Path(str(row[log_name_col]).strip()).stem
            name = f"{model_key}__{log_key}"

        if log_key not in xes_files:
            missing_logs.append(log_key)
            continue

        if model_key not in pnml_files:
            missing_models.append(model_key)
            continue

        records.append({
            "name": name,
            "log_path": xes_files[log_key],
            "model_path": pnml_files[model_key],
            "fitness": float(row[fitness_col]),
            "precision": float(row[precision_col]),
        })

    print("Valid records:", len(records))
    print("Missing logs:", len(missing_logs))
    print("Missing models:", len(missing_models))

    if missing_logs:
        print("First missing logs:", missing_logs[:10])

    if missing_models:
        print("First missing models:", missing_models[:10])

    rng = random.Random(seed)
    rng.shuffle(records)

    test_records = records[:test_size]
    train_records = records[test_size:]

    print("Train records:", len(train_records))
    print("Test records:", len(test_records))

    return train_records, test_records


def compute_max_local_activity_dim_from_records(records, variant_cache_dir=".cache_variants"):
    """
    Lightweight helper for Variant C.

    It only reads/caches XES variants and does not load PNML models.
    For Variant A you do not need this function.
    """
    max_dim = 0

    for i, record in enumerate(records):
        if i % 100 == 0:
            print(f"Scanning activity dimension: {i}/{len(records)}")

        variants = read_log_as_variants_cached(
            record["log_path"],
            cache_dir=variant_cache_dir,
        )

        acts = {str(act) for seq, _count in variants for act in seq}
        max_dim = max(max_dim, len(acts))

    print("max_local_onehot_dim:", max_dim)
    return max_dim
