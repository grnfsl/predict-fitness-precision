from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import dgl
import pandas as pd
import torch
from torch.utils.data import Dataset



def _current_rss_gib():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kib = float(line.split()[1])
                    return kib / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def load_split_records(split_csv: str | Path):
    """
    Load one saved split manifest.

    The manifest must contain cache_index plus the original results.csv columns.
    Fitness is normalized to record['fitness'] for the existing trainer code.
    """
    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)

    df = pd.read_csv(split_csv)

    if "cache_index" not in df.columns:
        raise ValueError(
            f"{split_csv} has no cache_index column. "
            "Run create_split_manifests.py first."
        )

    if "precision" not in df.columns:
        raise ValueError(f"{split_csv} has no precision column.")

    if "fitness" in df.columns:
        fitness_col = "fitness"
    elif "log_fitness" in df.columns:
        fitness_col = "log_fitness"
    else:
        raise ValueError(
            f"{split_csv} needs either a fitness or log_fitness column."
        )

    records = df.to_dict(orient="records")
    for record in records:
        record["cache_index"] = int(record["cache_index"])
        record["fitness"] = float(record[fitness_col])
        record["precision"] = float(record["precision"])

    return records


class CachedVariantADataset(Dataset):
    """
    Load the selected cached Variant A graphs into this Slurm job's RAM.

    The graph cache is shared on disk, but each job has its own in-memory copy.
    sample_weight remains experiment-specific and comes from the supplied records,
    not from the persistent graph cache.
    """

    def __init__(
        self,
        records: Sequence[dict],
        *,
        cache_dir: str | Path,
        label: str = "dataset",
        validate_targets: bool = True,
    ):
        self.records = list(records)
        self.cache_dir = Path(cache_dir)

        index_path = self.cache_dir / "index.csv"
        metadata_path = self.cache_dir / "metadata.json"
        shard_dir = self.cache_dir / "shards"

        if not index_path.exists():
            raise FileNotFoundError(index_path)
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        if not shard_dir.exists():
            raise FileNotFoundError(shard_dir)

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata.get("variant") != "A":
            raise ValueError(
                f"Expected Variant A cache, got {metadata.get('variant')!r}."
            )

        cache_index_df = pd.read_csv(index_path)
        cache_lookup = {
            int(row.cache_index): row
            for row in cache_index_df.itertuples(index=False)
        }

        total = len(self.records)
        self.graphs = [None] * total
        self.targets = [None] * total
        self.sample_weights = [None] * total

        requests_by_shard = defaultdict(list)

        for dataset_pos, record in enumerate(self.records):
            cache_index = int(record["cache_index"])

            if cache_index not in cache_lookup:
                raise KeyError(
                    f"cache_index={cache_index} from {label} is not present in "
                    f"{index_path}."
                )

            cache_row = cache_lookup[cache_index]

            if validate_targets:
                if abs(float(record["fitness"]) - float(cache_row.fitness)) > 1e-7:
                    raise ValueError(
                        f"Fitness mismatch for cache_index={cache_index}: "
                        f"split={record['fitness']}, cache={cache_row.fitness}"
                    )
                if abs(float(record["precision"]) - float(cache_row.precision)) > 1e-7:
                    raise ValueError(
                        f"Precision mismatch for cache_index={cache_index}: "
                        f"split={record['precision']}, cache={cache_row.precision}"
                    )

            requests_by_shard[str(cache_row.shard)].append(
                (dataset_pos, int(cache_row.graph_offset))
            )

            self.targets[dataset_pos] = torch.tensor(
                [float(record["fitness"]), float(record["precision"])],
                dtype=torch.float32,
            )
            self.sample_weights[dataset_pos] = torch.tensor(
                float(record.get("sample_weight", 1.0)),
                dtype=torch.float32,
            )

        print(
            f"[{label}] Loading {total} cached Variant A graphs from "
            f"{len(requests_by_shard)} shard(s) into RAM...",
            flush=True,
        )

        start = time.perf_counter()
        loaded = 0

        for shard_number, (shard_name, requests) in enumerate(
            requests_by_shard.items(),
            start=1,
        ):
            # Sort offsets for deterministic reads. DGL can load only the requested
            # graph indices from a multi-graph .bin file.
            requests = sorted(requests, key=lambda x: x[1])
            offsets = [graph_offset for _, graph_offset in requests]

            shard_path = shard_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(shard_path)

            graphs, _ = dgl.load_graphs(
                str(shard_path),
                idx_list=offsets,
            )

            if len(graphs) != len(requests):
                raise RuntimeError(
                    f"Loaded {len(graphs)} graphs from {shard_name}, "
                    f"expected {len(requests)}."
                )

            for (dataset_pos, _), graph in zip(requests, graphs):
                self.graphs[dataset_pos] = graph

            loaded += len(graphs)

            if shard_number % 10 == 0 or shard_number == len(requests_by_shard):
                elapsed = time.perf_counter() - start
                rate = loaded / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{label}] loaded {loaded}/{total} graphs | "
                    f"{rate:.2f} graphs/s",
                    flush=True,
                )

        if any(graph is None for graph in self.graphs):
            raise RuntimeError(f"[{label}] finished with missing graph entries.")

        elapsed = time.perf_counter() - start
        rss = _current_rss_gib()

        msg = f"[{label}] Finished loading {total} graphs in {elapsed:.1f}s"
        if rss is not None:
            msg += f" | process RSS={rss:.2f} GiB"
        print(msg, flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            self.targets[idx],
            self.sample_weights[idx],
        )
