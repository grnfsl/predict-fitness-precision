"""Generate IM-discovered process-model/log pairs and evaluate fitness/precision.

The program keeps the algorithm visible:

1. Generate base process trees M_base.
2. Generate base logs L_base from M_base.
3. Run ProM Inductive Miner for configured thresholds.
4. Assign variant profiles to final IM model/log pair slots.
5. Save each final model-log pair manifest.
6. Evaluate every missing pair.
7. Save each completed result and preserve the run for resume/append.

This IM version keeps the ProM PNML model for normal pairs. If a variant
profile contains model_mutations, it parses the ProM EfficientTree string,
applies the same PM4Py process-tree mutations used by the mutation-only
generator, converts the mutated tree to PNML, and evaluates that final pair.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import queue
import random
import re
import signal
import shutil
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

import numpy as np
import pm4py

from pm4py.util import constants as pm4py_constants
from pm4py.util import xes_constants

# PM4Py 2.7.23.3 multiprocessing-alignment compatibility fix
if not hasattr(pm4py_constants, "DEFAULT_TIMESTAMP_KEY"):
    pm4py_constants.DEFAULT_TIMESTAMP_KEY = (
        xes_constants.DEFAULT_TIMESTAMP_KEY
    )

from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.simulation.playout.process_tree import algorithm as tree_playout
from pm4py.algo.simulation.tree_generator import algorithm as tree_generator
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.process_tree import converter as tree_converter
from pm4py.objects.log.obj import Event, EventLog, Trace

from log_mutations import apply_log_mutations
from model_mutations import apply_model_mutations
from prom_tree_parser import parse_prom_tree_string

ACTIVITY_KEY = "concept:name"
CASE_ID_KEY = "concept:name"
THRESHOLD_PATTERN = re.compile(r"_thr_(\d{2})\.pnml$")


class EvaluationTimeout(TimeoutError):
    """Raised when one fitness/precision evaluation exceeds its time limit."""


def _raise_evaluation_timeout(signum: int, frame: Any) -> None:
    del signum, frame
    raise EvaluationTimeout("Alignment evaluation exceeded its time limit.")


@contextmanager
def evaluation_timeout(seconds: float):
    """
    Limit one alignment fitness + precision evaluation.

    The timer runs inside the worker process. A value <= 0 disables it.
    """
    seconds = float(seconds)
    if seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("timeout_seconds_per_pair requires Linux or macOS.")

    previous_handler = signal.signal(signal.SIGALRM, _raise_evaluation_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


RESULT_COLUMNS = [
    "run_id",
    "model_name",
    "log_name",
    "model_index",
    "base_id",
    "threshold",
    "log_variant_type",
    "log_variant_index",
    "mutation_order",
    "mutation_parameters",
    "seed",
    "duplicate_retry",
    "algorithm_name",
    "algorithm_parameters",
    "average_trace_fitness",
    "log_fitness",
    "precision",
    "percFitTraces",
    "percentage_of_fitting_traces",
    "num_traces",
    "num_trace_variants",
    "num_model_activities",
    "num_log_activities",
    "process_tree",
    "base_process_tree",
    "evaluation_seconds",
    "status",
    "error",
]


# ---------------------------------------------------------------------------
# Configuration and run folders
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load the complete configuration from JSON without hidden defaults."""
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_run_ids(runs_dir: Path) -> List[str]:
    return sorted(
        path.name
        for path in runs_dir.glob("run_*")
        if path.is_dir() and path.name[4:].isdigit()
    )


def next_run_id(runs_dir: Path) -> str:
    run_ids = list_run_ids(runs_dir)
    next_number = max((int(run_id[4:]) for run_id in run_ids), default=0) + 1
    return f"run_{next_number:04d}"


def comparable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ignore operational settings that should not invalidate resume."""
    comparable = copy.deepcopy(config)
    comparable.pop("run", None)
    comparable.pop("overwrite_output", None)
    comparable.pop("evaluation", None)
    comparable.pop("paths", None)

    # Discovery execution settings may differ during resume. The thresholds and
    # algorithm metadata still matter, so keep them.
    prom = comparable.get("prom")
    if isinstance(prom, dict):
        prom.pop("run_java", None)
        prom.pop("ant_executable", None)
        prom.pop("prom_project_dir", None)

    return comparable


def prepare_run(config: Dict[str, Any]) -> Tuple[str, Dict[str, Path], bool]:
    output_dir = Path(config["paths"]["output_dir"]).resolve()
    runs_dir = output_dir / "runs"
    mode = str(config["run"].get("mode", "new")).lower()
    requested_run_id = config["run"].get("run_id")

    if mode == "new" and config.get("overwrite_output", False) and output_dir.exists():
        shutil.rmtree(output_dir)

    runs_dir.mkdir(parents=True, exist_ok=True)
    existing = list_run_ids(runs_dir)
    legacy_results = output_dir / "results.csv"
    if (
        mode == "new"
        and not existing
        and legacy_results.exists()
        and not config.get("overwrite_output", False)
    ):
        raise FileExistsError(
            f"Legacy output already exists at {legacy_results}. "
            "Choose a new output directory or set overwrite_output=true once."
        )

    if mode == "new":
        if existing:
            raise FileExistsError(
                "Existing runs were found. Use run.mode='append' to add a new run "
                "or run.mode='resume' to continue one."
            )
        run_id = str(requested_run_id or "run_0001")
        is_resume = False
    elif mode == "append":
        run_id = str(requested_run_id or next_run_id(runs_dir))
        if (runs_dir / run_id).exists():
            raise FileExistsError(f"Run already exists: {run_id}")
        is_resume = False
    elif mode == "resume":
        if not existing:
            raise FileNotFoundError("No existing run is available to resume.")
        run_id = str(requested_run_id or existing[-1])
        if not (runs_dir / run_id).is_dir():
            raise FileNotFoundError(f"Run not found: {run_id}")
        is_resume = True
    else:
        raise ValueError("run.mode must be 'new', 'append', or 'resume'.")

    run_dir = runs_dir / run_id
    paths = {
        "output_dir": output_dir,
        "runs_dir": runs_dir,
        "run_dir": run_dir,
        "base_models_dir": run_dir / "base_models",
        "base_logs_dir": run_dir / "base_logs",
        "discovered_models_dir": run_dir / "discovered_models",
        "final_models_dir": run_dir / "final_models",
        "log_variants_dir": run_dir / "log_variants",
        "manifests_dir": run_dir / "manifests",
        "results_path": run_dir / "results.csv",
        "used_config_path": run_dir / "used_config.json",
        "state_path": run_dir / "run_state.json",
        "combined_results_path": output_dir / "results.csv",
    }

    for key in (
        "base_models_dir",
        "base_logs_dir",
        "discovered_models_dir",
        "final_models_dir",
        "log_variants_dir",
        "manifests_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)

    if is_resume:
        with paths["used_config_path"].open("r", encoding="utf-8") as file:
            previous_config = json.load(file)
        if comparable_config(previous_config) != comparable_config(config):
            raise ValueError(
                "Generation settings differ from the saved run. "
                "Use run.mode='append' for a different configuration."
            )
    else:
        with paths["used_config_path"].open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

    save_state(paths["state_path"], status="running")
    return run_id, paths, is_resume


def save_state(state_path: Path, **values: Any) -> None:
    state: Dict[str, Any] = {}
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as file:
            state = json.load(file)
    state.update(values)
    with state_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def base_id_for_index(model_index: int) -> str:
    # Keep old IM naming so ProM output folders match existing conventions.
    return f"base_{model_index - 1:06d}"


def threshold_to_token(threshold: float) -> str:
    return f"thr_{int(round(float(threshold) * 10.0)):02d}"


def get_tree_activities(tree: Any) -> List[str]:
    activities: set[str] = set()

    def visit(node: Any) -> None:
        if node.operator is None:
            if node.label is not None:
                activities.add(str(node.label))
            return
        for child in node.children:
            visit(child)

    visit(tree)
    return sorted(activities)


def get_net_activities(net: Any) -> List[str]:
    return sorted({str(t.label) for t in net.transitions if t.label is not None})


def log_to_sequences(log: EventLog) -> List[List[str]]:
    return [
        [str(event[ACTIVITY_KEY]) for event in trace]
        for trace in log
        if len(trace) > 0
    ]


def sequences_to_log(sequences: Sequence[Sequence[str]], case_prefix: str) -> EventLog:
    log = EventLog()
    for trace_index, sequence in enumerate(sequences, start=1):
        if not sequence:
            continue
        trace = Trace(attributes={CASE_ID_KEY: f"{case_prefix}_{trace_index:06d}"})
        for activity in sequence:
            trace.append(Event({ACTIVITY_KEY: activity}))
        log.append(trace)
    return log


def trace_variant(sequence: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sequence)


def log_fingerprint(
    sequences: Sequence[Sequence[str]],
) -> Tuple[Tuple[Tuple[str, ...], int], ...]:
    counts = Counter(trace_variant(sequence) for sequence in sequences)
    return tuple(sorted(counts.items()))


def count_log_activities(sequences: Sequence[Sequence[str]]) -> int:
    return len({activity for sequence in sequences for activity in sequence})


def make_foreign_activity_pool(
    model_activities: Sequence[str], pool_size: int
) -> List[str]:
    existing = set(model_activities)
    pool: List[str] = []
    candidate_number = 1
    while len(pool) < pool_size:
        label = f"__FOREIGN_{candidate_number:03d}__"
        if label not in existing:
            pool.append(label)
        candidate_number += 1
    return pool


def format_mutation_order(
    log_order: Sequence[str],
    model_order: Sequence[str],
) -> str:
    parts: List[str] = []
    if log_order:
        parts.append("log:" + ">".join(log_order))
    if model_order:
        parts.append("model:" + ">".join(model_order))
    return " | ".join(parts)


def mutation_order_to_string(value: Any) -> str:
    if isinstance(value, list):
        return ">".join(str(item) for item in value)
    return str(value or "")


def is_allocated_variant_sampling(config: Dict[str, Any]) -> bool:
    sampling = config["log_variants"].get("variant_sampling", {})
    return str(sampling.get("mode", "all")).lower() == "allocated"


def allocate_counts(weights: Sequence[float], total: int) -> List[int]:
    """Allocate integer counts using largest remainders."""
    if total <= 0:
        return [0 for _ in weights]
    if not weights:
        return []

    weight_sum = sum(max(0.0, float(weight)) for weight in weights)
    if weight_sum <= 0:
        weights = [1.0 for _ in weights]
        weight_sum = float(len(weights))

    raw = [max(0.0, float(weight)) / weight_sum * total for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)

    order = sorted(
        range(len(raw)),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    return counts


# ---------------------------------------------------------------------------
# Variant-profile config
# ---------------------------------------------------------------------------


def build_variant_entries(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return configured log-variant profile entries.

    New format:
        log_variants.variants = [
            {target_fraction: 0.1, log_mutations: {...}, model_mutations: {}},
            ...
        ]

    Legacy mutation config is also accepted when log_variants["variants"] is absent.
    """
    log_variants_config = config["log_variants"]

    if "variants" in log_variants_config:
        entries: List[Dict[str, Any]] = []
        for entry in log_variants_config.get("variants", []):
            if not entry.get("enabled", True):
                continue
            log_mutations = dict(entry.get("log_mutations", {}))
            model_mutations = dict(entry.get("model_mutations", {}))
            entries.append(
                {
                    "format": "new",
                    "variant_type": (
                        "clean" if not log_mutations and not model_mutations else "variant"
                    ),
                    "num_variants": int(entry.get("num_variants", 1)),
                    "log_mutations": log_mutations,
                    "model_mutations": model_mutations,
                    "log_mutation_order": entry.get("log_mutation_order"),
                    "model_mutation_order": entry.get("model_mutation_order"),
                    "shuffle_log_mutations": bool(entry.get("shuffle_log_mutations", False)),
                    "shuffle_model_mutations": bool(entry.get("shuffle_model_mutations", False)),
                    "target_fraction": float(entry.get("target_fraction", 1.0)),
                }
            )
        return entries

    # Backward-compatible legacy format.
    entries = []
    for mutation_name, mutation_config in log_variants_config.items():
        if mutation_name in {"clean", "variants", "variant_sampling"}:
            continue
        if not mutation_config.get("enabled", False):
            continue

        if mutation_name == "mixed":
            entries.append(
                {
                    "format": "legacy",
                    "variant_type": "mixed",
                    "num_variants": int(mutation_config.get("num_variants", 1)),
                    "log_mutations": dict(mutation_config["operations"]),
                    "model_mutations": {},
                    "log_mutation_order": list(mutation_config["mutation_order"]),
                    "model_mutation_order": None,
                    "shuffle_log_mutations": bool(mutation_config.get("shuffle_order", False)),
                    "shuffle_model_mutations": False,
                    "target_fraction": float(mutation_config.get("target_fraction", 1.0)),
                }
            )
            continue

        parameters = {
            key: value
            for key, value in mutation_config.items()
            if key not in {"enabled", "num_variants", "target_fraction"}
        }
        entries.append(
            {
                "format": "legacy",
                "variant_type": mutation_name,
                "num_variants": int(mutation_config.get("num_variants", 1)),
                "log_mutations": {mutation_name: parameters},
                "model_mutations": {},
                "log_mutation_order": [mutation_name],
                "model_mutation_order": None,
                "shuffle_log_mutations": False,
                "shuffle_model_mutations": False,
                "target_fraction": float(mutation_config.get("target_fraction", 1.0)),
            }
        )

    return entries


def build_clean_entry() -> Dict[str, Any]:
    return {
        "format": "new",
        "variant_type": "clean",
        "num_variants": 1,
        "log_mutations": {},
        "model_mutations": {},
        "log_mutation_order": [],
        "model_mutation_order": [],
        "shuffle_log_mutations": False,
        "shuffle_model_mutations": False,
        "target_fraction": 1.0,
    }


def build_variant_plan(
    config: Dict[str, Any],
    discovered_models: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    """
    Optionally allocate variant profiles across final IM model/log pair slots.

    In the IM version, allocation_unit must be model_log_pair. One slot is:
        one discovered IM model threshold + its base log.
    """
    if not is_allocated_variant_sampling(config):
        return None

    sampling = config["log_variants"].get("variant_sampling", {})
    allocation_unit = str(sampling.get("allocation_unit", "model_log_pair"))
    if allocation_unit != "model_log_pair":
        raise ValueError("IM variant_sampling.allocation_unit must be 'model_log_pair'.")

    entries = build_variant_entries(config)
    slots: List[Tuple[str, str]] = []
    for base_id in sorted(discovered_models):
        for model in discovered_models[base_id]:
            slots.append((base_id, str(model["model_name"])))

    if not entries:
        return {slot: [] for slot in slots}

    variants_per_slot = int(
        sampling.get(
            "variants_per_model_log_pair",
            sampling.get("variants_per_base_model", 1),
        )
    )
    variants_per_slot = max(1, variants_per_slot)
    total_slots = len(slots) * variants_per_slot

    weights = [float(entry.get("target_fraction", 1.0)) for entry in entries]
    counts = allocate_counts(weights, total_slots)

    assigned_entries: List[Dict[str, Any]] = []
    for entry, count in zip(entries, counts):
        for _ in range(count):
            slot_entry = copy.deepcopy(entry)
            slot_entry["num_variants"] = 1
            assigned_entries.append(slot_entry)

    rng = random.Random(int(config["seed"]) + 900_000)
    if sampling.get("shuffle_assignments", True):
        rng.shuffle(assigned_entries)

    plan: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    offset = 0
    for slot in slots:
        plan[slot] = assigned_entries[offset : offset + variants_per_slot]
        offset += variants_per_slot

    return plan


# ---------------------------------------------------------------------------
# Steps 1-2: Generate base models/logs
# ---------------------------------------------------------------------------


def generate_or_load_base_item(
    config: Dict[str, Any],
    paths: Dict[str, Path],
    model_index: int,
) -> Dict[str, Any]:
    base_id = base_id_for_index(model_index)
    tree_path = paths["base_models_dir"] / f"{base_id}.ptml"
    log_path = paths["base_logs_dir"] / f"{base_id}.xes"

    if tree_path.exists() and log_path.exists():
        tree = pm4py.read_ptml(str(tree_path))
        base_log = xes_importer.apply(
            str(log_path),
            parameters={"show_progress_bar": False},
        )
        return {
            "base_id": base_id,
            "model_index": model_index,
            "process_tree": tree,
            "process_tree_path": tree_path,
            "model_activities": get_tree_activities(tree),
            "base_log": base_log,
            "base_log_path": log_path,
        }

    model_seed = int(config["seed"]) + model_index
    seed_everything(model_seed)

    parameters = dict(config["base_generation"]["tree_parameters"])
    parameters["no_models"] = 1
    tree = tree_generator.apply(
        variant=tree_generator.Variants.PTANDLOGGENERATOR,
        parameters=parameters,
    )
    pm4py.write_ptml(tree, str(tree_path))

    log_seed = int(config["seed"]) + 100_000 + model_index
    seed_everything(log_seed)

    num_traces = int(config["base_generation"]["num_traces_per_model"])
    base_log = tree_playout.apply(
        tree,
        variant=tree_playout.Variants.BASIC_PLAYOUT,
        parameters={"num_traces": num_traces},
    )
    xes_exporter.apply(
        base_log,
        str(log_path),
        parameters={"show_progress_bar": False},
    )

    return {
        "base_id": base_id,
        "model_index": model_index,
        "process_tree": tree,
        "process_tree_path": tree_path,
        "model_activities": get_tree_activities(tree),
        "base_log": base_log,
        "base_log_path": log_path,
    }


def generate_base_items(
    config: Dict[str, Any],
    paths: Dict[str, Path],
) -> List[Dict[str, Any]]:
    num_models = int(config["base_generation"]["num_models"])
    base_items: List[Dict[str, Any]] = []

    for model_index in range(1, num_models + 1):
        item = generate_or_load_base_item(config, paths, model_index)
        base_items.append(item)
        print(
            f"Base {item['base_id']} ({model_index}/{num_models}): "
            f"{len(item['model_activities'])} activities, {len(item['base_log'])} traces"
        )

    return base_items


# ---------------------------------------------------------------------------
# Step 3: Run ProM discovery
# ---------------------------------------------------------------------------


def run_prom_discovery(config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    prom_config = config["prom"]

    if not bool(prom_config.get("run_java", True)):
        print("Skipped Java discovery: prom.run_java is false.")
        return

    prom_project_dir_value = prom_config.get(
        "prom_project_dir",
        config.get("paths", {}).get("prom_project_dir"),
    )
    if not prom_project_dir_value:
        raise ValueError("Missing prom.prom_project_dir in config.")

    prom_project_dir = Path(prom_project_dir_value).resolve()
    if not (prom_project_dir / "build.xml").exists():
        raise FileNotFoundError(f"build.xml not found in {prom_project_dir}")

    thresholds = ",".join(str(value) for value in prom_config["thresholds"])
    command = [
        str(prom_config.get("ant_executable", "ant")),
        f"-Dinput.dir={paths['base_logs_dir']}",
        f"-Doutput.dir={paths['discovered_models_dir']}",
        f"-Dthresholds={thresholds}",
        "run",
    ]

    print("Running ProM discovery...")
    subprocess.run(command, cwd=prom_project_dir, check=True)


# ---------------------------------------------------------------------------
# Step 4: Load discovered PNML models
# ---------------------------------------------------------------------------


def load_discovered_models_from_status(
    config: Dict[str, Any],
    paths: Dict[str, Path],
) -> Dict[str, List[Dict[str, Any]]]:
    status_path = paths["discovered_models_dir"] / "discovery_status.csv"
    discovered: Dict[str, List[Dict[str, Any]]] = {}
    if not status_path.exists():
        return discovered

    prom_config = config["prom"]
    algorithm_name = str(prom_config["algorithm_name"])
    parameter_name = str(prom_config["parameter_name"])

    with status_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if str(row.get("status", "")).lower() != "success":
                continue

            model_path = Path(str(row.get("output_pnml", "")))
            if not model_path.exists():
                print(f"Warning: discovered PNML does not exist: {model_path}")
                continue

            threshold = float(row["threshold"])
            base_id = str(row["log_id"])
            net, _, _ = pm4py.read_pnml(str(model_path))
            model_activities = get_net_activities(net)
            discovered.setdefault(base_id, []).append(
                {
                    "base_id": base_id,
                    "model_name": model_path.name,
                    "model_path": model_path,
                    "threshold": threshold,
                    "threshold_token": threshold_to_token(threshold),
                    "model_activities": model_activities,
                    "num_model_activities": len(model_activities),
                    "process_tree": row.get("process_tree", ""),
                    "algorithm_name": algorithm_name,
                    "algorithm_parameters": {parameter_name: threshold},
                    "sort_value": threshold,
                }
            )

    for models in discovered.values():
        models.sort(key=lambda item: item["sort_value"])

    return discovered


def load_discovered_models_from_pnml_names(
    config: Dict[str, Any],
    paths: Dict[str, Path],
) -> Dict[str, List[Dict[str, Any]]]:
    discovered: Dict[str, List[Dict[str, Any]]] = {}
    prom_config = config["prom"]
    algorithm_name = str(prom_config["algorithm_name"])
    parameter_name = str(prom_config["parameter_name"])

    for model_path in sorted(paths["discovered_models_dir"].glob("*/*.pnml")):
        match = THRESHOLD_PATTERN.search(model_path.name)
        if match is None:
            print(f"Skipped unrecognised PNML name: {model_path.name}")
            continue

        threshold = int(match.group(1)) / 10.0
        base_id = model_path.parent.name
        net, _, _ = pm4py.read_pnml(str(model_path))
        model_activities = get_net_activities(net)
        discovered.setdefault(base_id, []).append(
            {
                "base_id": base_id,
                "model_name": model_path.name,
                "model_path": model_path,
                "threshold": threshold,
                "threshold_token": threshold_to_token(threshold),
                "model_activities": model_activities,
                "num_model_activities": len(model_activities),
                "process_tree": "",
                "algorithm_name": algorithm_name,
                "algorithm_parameters": {parameter_name: threshold},
                "sort_value": threshold,
            }
        )

    for models in discovered.values():
        models.sort(key=lambda item: item["sort_value"])

    return discovered


def load_discovered_models(
    config: Dict[str, Any],
    paths: Dict[str, Path],
) -> Dict[str, List[Dict[str, Any]]]:
    discovered = load_discovered_models_from_status(config, paths)
    if discovered:
        return discovered
    return load_discovered_models_from_pnml_names(config, paths)



def save_tree_as_pnml(tree: Any, model_path: Path) -> None:
    net, initial_marking, final_marking = tree_converter.apply(tree)
    pm4py.write_pnml(net, initial_marking, final_marking, str(model_path))


def load_prom_tree_for_model(model: Dict[str, Any]) -> Any:
    process_tree_text = str(model.get("process_tree", "") or "")
    if not process_tree_text.strip():
        raise ValueError(
            f"No process_tree string was found for {model['model_name']}. "
            "Use the updated PromIMBatch.java that writes process_tree in "
            "discovery_status.csv, then rerun discovery."
        )
    return parse_prom_tree_string(process_tree_text)


# ---------------------------------------------------------------------------
# Step 5: Generate final log/model pair manifests
# ---------------------------------------------------------------------------


def remove_incomplete_base_files(base_id: str, paths: Dict[str, Path]) -> None:
    (paths["manifests_dir"] / f"{base_id}.json").unlink(missing_ok=True)

    variant_dir = paths["log_variants_dir"] / base_id
    if variant_dir.exists():
        shutil.rmtree(variant_dir)

    final_model_dir = paths["final_models_dir"] / base_id
    if final_model_dir.exists():
        shutil.rmtree(final_model_dir)


def create_final_pair_for_slot(
    base_item: Dict[str, Any],
    model: Dict[str, Any],
    entry: Dict[str, Any],
    entry_number: int,
    requested_index: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create the final pair content for one IM model/log slot.

    If model_mutations is empty, the original ProM PNML is reused.
    If model_mutations is non-empty, the ProM EfficientTree string is parsed,
    mutated as a PM4Py ProcessTree, then converted to a new PNML later.
    """
    base_sequences = log_to_sequences(base_item["base_log"])
    if not base_sequences:
        raise ValueError(f"{base_item['base_id']}: base log is empty.")

    generation_config = config["variant_generation"]
    max_duplicate_retries = int(generation_config["max_duplicate_retries"])
    foreign_pool_size = int(generation_config["foreign_activity_pool_size"])

    original_model_activities = list(model.get("model_activities", []))
    if not original_model_activities:
        net, _, _ = pm4py.read_pnml(str(model["model_path"]))
        original_model_activities = get_net_activities(net)

    model_index = int(base_item["model_index"])
    threshold_index = int(round(float(model["threshold"]) * 1000))
    seed_base = (
        int(config["seed"])
        + model_index * 100_000_000
        + threshold_index * 1_000_000
        + entry_number * 10_000
        + requested_index * 100
    )

    log_mutations = dict(entry.get("log_mutations", {}))
    model_mutations = dict(entry.get("model_mutations", {}))
    is_clean_profile = not log_mutations and not model_mutations
    variant_type = "clean" if is_clean_profile else entry["variant_type"]

    accepted: Optional[Dict[str, Any]] = None
    base_log_fingerprint = log_fingerprint(base_sequences)

    for retry in range(max_duplicate_retries + 1):
        candidate_seed = seed_base + retry
        rng = random.Random(candidate_seed)

        # Model side.
        if model_mutations:
            discovered_tree = load_prom_tree_for_model(model)
            mutated_tree, model_order, model_parameters = apply_model_mutations(
                discovered_tree,
                model_mutations,
                rng,
                mutation_order=entry.get("model_mutation_order"),
                shuffle_mutations=entry.get("shuffle_model_mutations", False),
            )
            current_model_activities = get_tree_activities(mutated_tree)
            process_tree_string = str(mutated_tree)
        else:
            mutated_tree = None
            model_order = []
            model_parameters = {}
            current_model_activities = original_model_activities
            process_tree_string = str(model.get("process_tree", "") or "")

        foreign_activities = make_foreign_activity_pool(
            sorted(set(current_model_activities) | set(base_item["model_activities"])),
            foreign_pool_size,
        )

        # Log side.
        sequences, log_order, log_parameters = apply_log_mutations(
            base_sequences,
            log_mutations,
            rng,
            current_model_activities,
            foreign_activities,
            mutation_order=entry.get("log_mutation_order"),
            shuffle_mutations=entry.get("shuffle_log_mutations", False),
        )

        if not sequences:
            continue

        log_changed = log_fingerprint(sequences) != base_log_fingerprint
        model_changed = bool(model_order)

        if not is_clean_profile and not log_changed and not model_changed:
            continue

        accepted = {
            "variant_type": variant_type,
            "variant_index": requested_index,
            "tree": mutated_tree,
            "uses_mutated_model": model_changed,
            "sequences": sequences,
            "mutation_order": format_mutation_order(log_order, model_order),
            "mutation_parameters": {
                "log": log_parameters,
                "model": model_parameters,
            },
            "model_mutation_order": model_order,
            "seed": candidate_seed,
            "duplicate_retry": retry,
            "num_model_activities": len(current_model_activities),
            "process_tree": process_tree_string,
        }
        break

    if accepted is None:
        raise RuntimeError(
            f"Could not create a usable {entry['variant_type']} variant "
            f"for {base_item['base_id']} and {model['model_name']}."
        )

    return accepted

def selected_entries_for_model(
    config: Dict[str, Any],
    variant_plan: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]],
    base_id: str,
    model_name: str,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    clean_settings = config["log_variants"].get("clean", {})
    if bool(clean_settings.get("enabled", False)):
        entries.append(build_clean_entry())

    if variant_plan is not None:
        entries.extend(variant_plan.get((base_id, model_name), []))
    else:
        entries.extend(build_variant_entries(config))

    return entries


def generate_manifest_for_base(
    run_id: str,
    base_item: Dict[str, Any],
    models: List[Dict[str, Any]],
    config: Dict[str, Any],
    paths: Dict[str, Path],
    variant_plan: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    base_id = base_item["base_id"]
    remove_incomplete_base_files(base_id, paths)
    variant_dir = paths["log_variants_dir"] / base_id
    variant_dir.mkdir(parents=True, exist_ok=True)

    final_model_dir = paths["final_models_dir"] / base_id
    final_model_dir.mkdir(parents=True, exist_ok=True)

    final_pairs: List[Dict[str, Any]] = []

    for model in models:
        entries = selected_entries_for_model(
            config,
            variant_plan,
            base_id,
            str(model["model_name"]),
        )
        new_format_index = 0

        for entry_number, entry in enumerate(entries, start=1):
            requested_variants = int(entry["num_variants"])
            for requested_index in range(1, requested_variants + 1):
                if entry["format"] == "new":
                    new_format_index += 1
                    variant_index = new_format_index
                else:
                    variant_index = requested_index

                try:
                    variant = create_final_pair_for_slot(
                        base_item=base_item,
                        model=model,
                        entry=entry,
                        entry_number=entry_number,
                        requested_index=variant_index,
                        config=config,
                    )
                except Exception as error:
                    print(f"  Warning: {error} Skipped.")
                    continue

                log_name = (
                    f"{base_id}_{model['threshold_token']}_"
                    f"{variant['variant_type']}_{variant_index:03d}.xes"
                )
                log_path = variant_dir / log_name
                event_log = sequences_to_log(
                    variant["sequences"],
                    case_prefix=log_name.removesuffix(".xes"),
                )
                xes_exporter.apply(
                    event_log,
                    str(log_path),
                    parameters={"show_progress_bar": False},
                )

                if variant["uses_mutated_model"]:
                    model_name = (
                        f"{base_id}_{model['threshold_token']}_"
                        f"{variant['variant_type']}_{variant_index:03d}.pnml"
                    )
                    model_path = final_model_dir / model_name
                    save_tree_as_pnml(variant["tree"], model_path)
                else:
                    model_name = model["model_name"]
                    model_path = Path(model["model_path"])

                sequences = variant["sequences"]
                final_pairs.append(
                    {
                        "run_id": run_id,
                        "base_id": base_id,
                        "model_name": model_name,
                        "model_path": str(model_path),
                        "log_name": log_name,
                        "log_path": str(log_path),
                        "model_index": base_item["model_index"],
                        "threshold": model["threshold"],
                        "log_variant_type": variant["variant_type"],
                        "log_variant_index": variant_index,
                        "mutation_order": variant["mutation_order"],
                        "mutation_parameters": variant["mutation_parameters"],
                        "seed": variant["seed"],
                        "duplicate_retry": variant["duplicate_retry"],
                        "algorithm_name": model["algorithm_name"],
                        "algorithm_parameters": model["algorithm_parameters"],
                        "num_traces": len(sequences),
                        "num_trace_variants": len(
                            {trace_variant(sequence) for sequence in sequences}
                        ),
                        "num_model_activities": variant["num_model_activities"],
                        "num_log_activities": count_log_activities(sequences),
                        "process_tree": variant["process_tree"],
                        "base_process_tree": str(base_item["process_tree"]),
                    }
                )

    manifest = {
        "base_id": base_id,
        "model_index": base_item["model_index"],
        "base_model_name": f"{base_id}.ptml",
        "base_log_name": f"{base_id}.xes",
        "base_process_tree": str(base_item["process_tree"]),
        "final_pairs": final_pairs,
    }

    manifest_path = paths["manifests_dir"] / f"{base_id}.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return manifest


def load_complete_manifest(base_id: str, paths: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    manifest_path = paths["manifests_dir"] / f"{base_id}.json"
    if not manifest_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    for pair in manifest.get("final_pairs", []):
        if not Path(pair["model_path"]).exists():
            return None
        if not Path(pair["log_path"]).exists():
            return None

    return manifest


def build_tasks_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []

    for pair in manifest.get("final_pairs", []):
        metadata = {
            "run_id": pair["run_id"],
            "model_name": pair["model_name"],
            "log_name": pair["log_name"],
            "model_index": pair["model_index"],
            "base_id": pair["base_id"],
            "threshold": pair["threshold"],
            "log_variant_type": pair["log_variant_type"],
            "log_variant_index": pair["log_variant_index"],
            "mutation_order": mutation_order_to_string(pair["mutation_order"]),
            "mutation_parameters": json.dumps(pair["mutation_parameters"], sort_keys=True),
            "seed": pair["seed"],
            "duplicate_retry": pair["duplicate_retry"],
            "algorithm_name": pair["algorithm_name"],
            "algorithm_parameters": json.dumps(pair["algorithm_parameters"], sort_keys=True),
            "num_traces": pair["num_traces"],
            "num_trace_variants": pair["num_trace_variants"],
            "num_model_activities": pair["num_model_activities"],
            "num_log_activities": pair["num_log_activities"],
            "process_tree": pair.get("process_tree", ""),
            "base_process_tree": pair.get("base_process_tree", ""),
        }
        tasks.append(
            {
                "model_path": pair["model_path"],
                "log_path": pair["log_path"],
                "metadata": metadata,
            }
        )

    return tasks


# ---------------------------------------------------------------------------
# Step 6: Evaluate each model-log pair
# ---------------------------------------------------------------------------


def evaluate_saved_pair(task: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one saved PNML/XES pair inside a worker process."""
    result = dict(task["metadata"])
    timeout_seconds = float(task.get("timeout_seconds", 0) or 0)
    started = time.perf_counter()

    try:
        net, initial_marking, final_marking = pm4py.read_pnml(task["model_path"])
        log = xes_importer.apply(
            task["log_path"],
            parameters={"show_progress_bar": False},
        )
        common_parameters = {
            "show_progress_bar": False,
            "multiprocessing": True,
            "cores": 4
        }

        with evaluation_timeout(timeout_seconds):
            fitness = fitness_evaluator.apply(
                log,
                net,
                initial_marking,
                final_marking,
                variant=fitness_evaluator.Variants.ALIGNMENT_BASED,
                parameters=common_parameters,
            )
            precision = precision_evaluator.apply(
                log,
                net,
                initial_marking,
                final_marking,
                variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE,
                parameters=common_parameters,
            )

        fitting_percentage = fitness.get(
            "percFitTraces",
            fitness.get("percentage_of_fitting_traces"),
        )
        result.update(
            {
                "average_trace_fitness": fitness.get("average_trace_fitness"),
                "log_fitness": fitness.get("log_fitness"),
                "precision": precision,
                "percFitTraces": fitting_percentage,
                "percentage_of_fitting_traces": fitting_percentage,
                "status": "ok",
                "error": "",
            }
        )
    except EvaluationTimeout as error:
        result.update(
            {
                "average_trace_fitness": None,
                "log_fitness": None,
                "precision": None,
                "percFitTraces": None,
                "percentage_of_fitting_traces": None,
                "status": "timeout",
                "error": str(error),
            }
        )
    except Exception as error:
        result.update(
            {
                "average_trace_fitness": None,
                "log_fitness": None,
                "precision": None,
                "percFitTraces": None,
                "percentage_of_fitting_traces": None,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        )

    result["evaluation_seconds"] = round(time.perf_counter() - started, 6)
    return result


# ---------------------------------------------------------------------------
# Step 7: Save each completed result
# ---------------------------------------------------------------------------


def pair_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return str(row["run_id"]), str(row["model_name"]), str(row["log_name"])


def prepare_results_for_resume(
    results_path: Path,
    retry_timeouts: bool = True,
) -> set[Tuple[str, str, str]]:
    """Keep completed rows; retry errors and optionally retry timeouts."""
    completed: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    if results_path.exists():
        with results_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                status = row.get("status")
                keep_row = status == "ok" or (
                    status == "timeout" and not retry_timeouts
                )
                if keep_row:
                    completed[pair_key(row)] = row

    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(completed.values())

    return set(completed)


def append_result(writer: csv.DictWriter, file: Any, row: Dict[str, Any]) -> None:
    writer.writerow(row)
    file.flush()


def make_empty_result_row(
    task: Dict[str, Any],
    status: str,
    error: str,
    evaluation_seconds: float,
) -> Dict[str, Any]:
    row = dict(task["metadata"])
    row.update(
        {
            "average_trace_fitness": None,
            "log_fitness": None,
            "precision": None,
            "percFitTraces": None,
            "percentage_of_fitting_traces": None,
            "evaluation_seconds": round(evaluation_seconds, 6),
            "status": status,
            "error": error,
        }
    )
    return row


def pair_worker_loop(
    worker_id: int,
    generation: int,
    task_queue: Any,
    result_queue: Any,
) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            break

        row = evaluate_saved_pair(task)
        result_queue.put(
            {
                "worker_id": worker_id,
                "generation": generation,
                "row": row,
            }
        )


def run_tasks(
    tasks: List[Dict[str, Any]],
    num_workers: int,
    timeout_seconds: float,
    grace_seconds: float,
    writer: csv.DictWriter,
    file: Any,
) -> List[Dict[str, Any]]:
    """Run pair tasks using reusable workers and parent-side hard timeout."""
    context = mp.get_context("spawn")
    result_queue = context.Queue()

    pending = list(tasks)
    completed: List[Dict[str, Any]] = []
    workers: Dict[int, Dict[str, Any]] = {}
    generations: Dict[int, int] = {}

    hard_limit = timeout_seconds + grace_seconds if timeout_seconds > 0 else 0

    def start_worker(worker_id: int) -> None:
        generation = generations.get(worker_id, 0) + 1
        generations[worker_id] = generation

        task_queue = context.Queue()
        process = context.Process(
            target=pair_worker_loop,
            args=(worker_id, generation, task_queue, result_queue),
        )
        process.start()

        workers[worker_id] = {
            "process": process,
            "task_queue": task_queue,
            "generation": generation,
            "current_task": None,
            "start_time": None,
        }

    def send_next_task(worker_id: int) -> None:
        if not pending:
            return
        task = pending.pop(0)
        worker = workers[worker_id]
        worker["current_task"] = task
        worker["start_time"] = time.perf_counter()
        worker["task_queue"].put(task)

    def stop_worker(worker_id: int) -> None:
        worker = workers[worker_id]
        process = worker["process"]

        if process.is_alive():
            worker["task_queue"].put(None)
            process.join(timeout=5)

        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

        if process.is_alive():
            process.kill()
            process.join()

        try:
            worker["task_queue"].close()
            worker["task_queue"].join_thread()
        except Exception:
            pass

    def restart_worker(worker_id: int) -> None:
        stop_worker(worker_id)
        start_worker(worker_id)

    def save_row(row: Dict[str, Any]) -> None:
        append_result(writer, file, row)
        completed.append(row)

    if not tasks:
        return completed

    num_workers = max(1, min(num_workers, len(tasks)))

    for worker_id in range(num_workers):
        start_worker(worker_id)
        send_next_task(worker_id)

    try:
        with tqdm(total=len(tasks), desc="Evaluating pairs", unit="pair") as progress:
            while len(completed) < len(tasks):
                for _ in range(1000):
                    try:
                        message = result_queue.get_nowait()
                    except queue.Empty:
                        break

                    worker_id = int(message["worker_id"])
                    if worker_id not in workers:
                        continue

                    worker = workers[worker_id]
                    if message["generation"] != worker["generation"]:
                        continue

                    row = message["row"]
                    worker["current_task"] = None
                    worker["start_time"] = None

                    save_row(row)
                    progress.update(1)
                    send_next_task(worker_id)

                for worker_id, worker in list(workers.items()):
                    task = worker["current_task"]
                    if task is None or hard_limit <= 0:
                        continue

                    elapsed = time.perf_counter() - worker["start_time"]
                    if elapsed <= hard_limit:
                        continue

                    row = make_empty_result_row(
                        task=task,
                        status="timeout",
                        error=(
                            f"Hard timeout after {elapsed:.2f} seconds "
                            f"(limit={timeout_seconds}, grace={grace_seconds})."
                        ),
                        evaluation_seconds=elapsed,
                    )
                    save_row(row)
                    progress.update(1)

                    restart_worker(worker_id)
                    send_next_task(worker_id)

                for worker_id, worker in list(workers.items()):
                    task = worker["current_task"]
                    process = worker["process"]
                    if task is None or process.is_alive():
                        continue

                    elapsed = time.perf_counter() - worker["start_time"]
                    row = make_empty_result_row(
                        task=task,
                        status="error",
                        error="Worker exited without returning a result.",
                        evaluation_seconds=elapsed,
                    )
                    save_row(row)
                    progress.update(1)

                    restart_worker(worker_id)
                    send_next_task(worker_id)

                time.sleep(0.2)
    finally:
        for worker_id in list(workers):
            stop_worker(worker_id)

        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

    return completed


def evaluate_and_save(
    tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    results_path: Path,
) -> List[Dict[str, Any]]:
    completed_rows: List[Dict[str, Any]] = []
    if not tasks:
        return completed_rows

    evaluation = config["evaluation"]
    use_multiprocessing = bool(evaluation.get("use_multiprocessing", False))
    timeout_seconds = float(evaluation.get("timeout_seconds_per_pair", 0) or 0)
    grace_seconds = float(evaluation.get("hard_timeout_grace_seconds", 0) or 0)

    worker_tasks = [{**task, "timeout_seconds": timeout_seconds} for task in tasks]

    with results_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_COLUMNS,
            extrasaction="ignore",
        )

        if not use_multiprocessing:
            for task in tqdm(worker_tasks, desc="Evaluating pairs", unit="pair"):
                row = evaluate_saved_pair(task)
                append_result(writer, file, row)
                completed_rows.append(row)
            return completed_rows

        configured_workers = int(evaluation.get("num_workers", 1))
        num_workers = max(1, min(configured_workers, len(worker_tasks)))
        completed_rows = run_tasks(
            tasks=worker_tasks,
            num_workers=num_workers,
            timeout_seconds=timeout_seconds,
            grace_seconds=grace_seconds,
            writer=writer,
            file=file,
        )

    return completed_rows


def rebuild_combined_results(paths: Dict[str, Path]) -> None:
    with paths["combined_results_path"].open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=RESULT_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for run_id in list_run_ids(paths["runs_dir"]):
            run_results = paths["runs_dir"] / run_id / "results.csv"
            if not run_results.exists():
                continue
            with run_results.open("r", newline="", encoding="utf-8") as input_file:
                writer.writerows(csv.DictReader(input_file))


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------


def main(config_path: Path = Path("v3/config.json")) -> None:
    config = load_config(config_path)
    run_id, paths, is_resume = prepare_run(config)

    print(f"Run: {run_id} ({'resume' if is_resume else config['run']['mode']})")

    # 1-2. Generate/reuse base process trees and base logs.
    base_items = generate_base_items(config, paths)
    base_by_id = {item["base_id"]: item for item in base_items}

    # 3. Discover IM models at all configured thresholds.
    run_prom_discovery(config, paths)

    # 4. Load discovered PNML models and discovered process-tree strings.
    discovered_models = load_discovered_models(config, paths)
    if not discovered_models:
        raise RuntimeError("No discovered IM models were found.")

    # 5. Build allocated profile plan after discovery, because the final slots
    # are discovered model/log pairs.
    variant_plan = build_variant_plan(config, discovered_models)

    all_tasks: List[Dict[str, Any]] = []
    for base_id in sorted(base_by_id):
        base_item = base_by_id[base_id]
        models = discovered_models.get(base_id, [])
        if not models:
            print(f"Warning: no discovered models found for {base_id}; skipped.")
            continue

        manifest = load_complete_manifest(base_id, paths)
        if manifest is None:
            manifest = generate_manifest_for_base(
                run_id=run_id,
                base_item=base_item,
                models=models,
                config=config,
                paths=paths,
                variant_plan=variant_plan,
            )
            print(f"Generated {len(manifest['final_pairs'])} final pairs for {base_id}")
        else:
            print(f"Reused complete manifest for {base_id}")

        all_tasks.extend(build_tasks_from_manifest(manifest))

    # 6. Evaluate every missing model-log pair.
    retry_timeouts = bool(config["evaluation"].get("retry_timeouts", True))
    completed_pairs = prepare_results_for_resume(
        paths["results_path"],
        retry_timeouts=retry_timeouts,
    )
    pending_tasks = [
        task for task in all_tasks
        if pair_key(task["metadata"]) not in completed_pairs
    ]
    print(
        f"\n6. Evaluating {len(pending_tasks)} missing pairs "
        f"({len(completed_pairs)} already completed)"
    )

    try:
        new_rows = evaluate_and_save(
            pending_tasks,
            config,
            paths["results_path"],
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        print("Rebuilding combined results from completed rows...")
        rebuild_combined_results(paths)
        save_state(paths["state_path"], status="interrupted")
        print(f"Partial results are saved in {paths['results_path']}")
        print(f"Combined results rebuilt at {paths['combined_results_path']}")
        return

    # 7. Results were saved pair-by-pair; rebuild all-runs CSV.
    rebuild_combined_results(paths)
    save_state(paths["state_path"], status="complete")
    successful_new = sum(row["status"] == "ok" for row in new_rows)
    print(f"7. Results saved immediately to {paths['results_path']}")
    print(f"   New successful evaluations: {successful_new}/{len(new_rows)}")
    print(f"   Combined results: {paths['combined_results_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, nargs="?", default=Path("config.json"))
    arguments = parser.parse_args()
    main(arguments.config)