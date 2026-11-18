from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import dgl
import pandas as pd
import torch
from pm4py.objects.petri_net.importer import importer as pnml_importer

from graph import prepare_pair_spec, build_alignment_pair_graph
from lazy_dataset import read_log_as_variants


VARIANT = "A"
TRACE_FREQ_MODE = "normalized"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _build_records(dataset_dir: Path):
    results_path = dataset_dir / "results.csv"
    models_dir = dataset_dir / "models"
    logs_dir = dataset_dir / "logs"

    if not results_path.exists():
        raise FileNotFoundError(results_path)
    if not models_dir.exists():
        raise FileNotFoundError(models_dir)
    if not logs_dir.exists():
        raise FileNotFoundError(logs_dir)

    df = pd.read_csv(results_path)

    required = {"model_name", "log_name", "precision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results.csv missing required columns: {sorted(missing)}")

    if "fitness" in df.columns:
        fitness_col = "fitness"
    elif "log_fitness" in df.columns:
        fitness_col = "log_fitness"
    else:
        raise ValueError("results.csv needs either 'fitness' or 'log_fitness'.")

    records = []
    missing_models = []
    missing_logs = []

    for cache_index, row in df.iterrows():
        model_name = str(row["model_name"])
        log_name = str(row["log_name"])

        model_path = models_dir / model_name
        log_path = logs_dir / log_name

        if not model_path.exists():
            missing_models.append(model_name)
            continue
        if not log_path.exists():
            missing_logs.append(log_name)
            continue

        records.append({
            "cache_index": int(cache_index),
            "model_name": model_name,
            "log_name": log_name,
            "model_path": str(model_path.resolve()),
            "log_path": str(log_path.resolve()),
            "fitness": float(row[fitness_col]),
            "precision": float(row["precision"]),
        })

    if missing_models or missing_logs:
        raise FileNotFoundError(
            "Canonical results.csv references missing files. "
            f"Missing models={len(missing_models)}, logs={len(missing_logs)}. "
            f"First models={missing_models[:5]}, first logs={missing_logs[:5]}"
        )

    if len(records) != len(df):
        raise RuntimeError("Internal record-count mismatch.")

    return records, results_path, fitness_col


def _make_shards(records, target_graphs_per_shard: int):
    """
    Pack whole model groups into shards where practical.

    This makes one worker reuse one parsed PNML model for all of that model's
    log/model pairs instead of reparsing the model for every graph.
    """
    groups = OrderedDict()
    for record in records:
        groups.setdefault(record["model_path"], []).append(record)

    shards = []
    current = []

    for group_records in groups.values():
        # Extremely large model groups are allowed to span multiple shards.
        if len(group_records) > target_graphs_per_shard:
            if current:
                shards.append(current)
                current = []

            for start in range(0, len(group_records), target_graphs_per_shard):
                shards.append(group_records[start:start + target_graphs_per_shard])
            continue

        if current and len(current) + len(group_records) > target_graphs_per_shard:
            shards.append(current)
            current = []

        current.extend(group_records)

    if current:
        shards.append(current)

    return shards


def _build_one_shard(shard_id: int, records, build_dir: str):
    # Keep each process single-threaded internally. Parallelism comes from the
    # process pool, not nested BLAS/OpenMP/PyTorch thread pools.
    torch.set_num_threads(1)

    build_dir = Path(build_dir)
    shard_dir = build_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    model_cache = {}
    log_cache = {}
    graphs = []
    index_rows = []

    for local_offset, record in enumerate(records):
        model_path = record["model_path"]
        log_path = record["log_path"]

        if model_path not in model_cache:
            model_cache[model_path] = pnml_importer.apply(model_path)
        net, im, fm = model_cache[model_path]

        if log_path not in log_cache:
            log_cache[log_path] = read_log_as_variants(log_path)
        log_variants = log_cache[log_path]

        spec = prepare_pair_spec(
            log_df=log_variants,
            net=net,
            im=im,
            fm=fm,
            fitness=record["fitness"],
            precision=record["precision"],
            compute_targets_if_missing=False,
            sample_weight=1.0,
        )

        graph = build_alignment_pair_graph(
            spec=spec,
            variant=VARIANT,
            max_local_onehot_dim=0,
            random_permute_onehot=False,
            trace_freq_mode=TRACE_FREQ_MODE,
            rng=None,
        )

        graphs.append(graph)
        index_rows.append({
            "cache_index": int(record["cache_index"]),
            "shard": f"shard_{shard_id:05d}.bin",
            "graph_offset": int(local_offset),
            "model_name": record["model_name"],
            "log_name": record["log_name"],
            "fitness": float(record["fitness"]),
            "precision": float(record["precision"]),
        })

    shard_path = shard_dir / f"shard_{shard_id:05d}.bin"
    tmp_path = shard_dir / f".{shard_path.name}.{os.getpid()}.tmp"

    dgl.save_graphs(str(tmp_path), graphs)
    os.replace(tmp_path, shard_path)

    elapsed = time.perf_counter() - start

    # Release large per-shard objects before this worker accepts another task.
    del graphs, model_cache, log_cache
    gc.collect()

    return {
        "shard_id": shard_id,
        "count": len(records),
        "elapsed": elapsed,
        "index_rows": index_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build the reusable master Variant A DGL graph cache."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data_prep/local_dataset_clean1"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "GRAPH_CACHE_DIR",
                "graph_cache/pgnn_sync_context_behavior_stats_without_pm4py_bug",
            )
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("GRAPH_BUILD_WORKERS", "4")),
    )
    parser.add_argument(
        "--graphs-per-shard",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.graphs_per_shard < 1:
        raise ValueError("--graphs-per-shard must be >= 1")

    dataset_dir = args.dataset_dir.resolve()
    cache_dir = args.cache_dir.resolve()

    records, results_path, fitness_col = _build_records(dataset_dir)
    shards = _make_shards(records, args.graphs_per_shard)

    print("=" * 80)
    print("MASTER VARIANT A GRAPH CACHE BUILD")
    print("=" * 80)
    print("Dataset:", dataset_dir)
    print("Results:", results_path)
    print("Graphs:", len(records))
    print("Shards:", len(shards))
    print("Workers:", args.workers)
    print("Cache:", cache_dir)
    print(flush=True)

    if cache_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Cache already exists: {cache_dir}\n"
                "Use --overwrite only when you intentionally want to rebuild it."
            )
        shutil.rmtree(cache_dir)

    build_dir = Path(str(cache_dir) + f".building_{os.getpid()}")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    start_all = time.perf_counter()
    index_rows = []
    completed_graphs = 0

    try:
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
        ) as executor:
            futures = {
                executor.submit(
                    _build_one_shard,
                    shard_id,
                    shard_records,
                    str(build_dir),
                ): shard_id
                for shard_id, shard_records in enumerate(shards)
            }

            for future in as_completed(futures):
                result = future.result()
                index_rows.extend(result["index_rows"])
                completed_graphs += result["count"]

                elapsed = time.perf_counter() - start_all
                rate = completed_graphs / elapsed if elapsed > 0 else 0.0

                print(
                    f"shard {result['shard_id']:05d} finished | "
                    f"{completed_graphs}/{len(records)} graphs | "
                    f"{rate:.2f} graphs/s | "
                    f"shard time={result['elapsed']:.1f}s",
                    flush=True,
                )

        index_df = pd.DataFrame(index_rows).sort_values("cache_index")

        if len(index_df) != len(records):
            raise RuntimeError(
                f"Cache index contains {len(index_df)} rows, expected {len(records)}."
            )

        if index_df["cache_index"].tolist() != list(range(len(records))):
            raise RuntimeError("Cache index is missing or duplicating canonical indices.")

        index_df.to_csv(build_dir / "index.csv", index=False)

        metadata = {
            "cache_version": 1,
            "variant": VARIANT,
            "trace_freq_mode": TRACE_FREQ_MODE,
            "random_permute_onehot": False,
            "max_local_onehot_dim": 0,
            "graph_count": len(records),
            "shard_count": len(shards),
            "graphs_per_shard_target": args.graphs_per_shard,
            "fitness_column": fitness_col,
            "canonical_results_sha256": _sha256(results_path),
            "canonical_results_path": str(results_path.resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        with (build_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        build_dir.rename(cache_dir)

    except Exception:
        print(f"Build failed. Partial cache left at: {build_dir}", flush=True)
        raise

    elapsed = time.perf_counter() - start_all

    print()
    print("=" * 80)
    print("CACHE BUILD COMPLETE")
    print("=" * 80)
    print("Cache:", cache_dir)
    print("Graphs:", len(records))
    print("Total time:", f"{elapsed:.1f}s")
    print("Average rate:", f"{len(records) / elapsed:.2f} graphs/s")
    print("Index:", cache_dir / "index.csv")
    print("Metadata:", cache_dir / "metadata.json")


if __name__ == "__main__":
    main()
