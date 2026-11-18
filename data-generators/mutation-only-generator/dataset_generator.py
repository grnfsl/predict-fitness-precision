"""Generate process-model/log pairs and evaluate fitness and precision.

The program keeps the algorithm visible:

1. Generate model M_base.
2. Generate log L_base from M_base.
3. Generate log set L_variants from L_base.
4. Evaluate every model-log pair.
5. Save each completed result and preserve the run for resume/append.
"""

from __future__ import annotations

import os
import argparse
import copy
import csv
import json
import multiprocessing as mp
import queue
import random
import signal
import shutil
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import pm4py
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.simulation.playout.process_tree import algorithm as tree_playout
from pm4py.algo.simulation.tree_generator import algorithm as tree_generator
from pm4py.objects.conversion.process_tree import converter as tree_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import Event, EventLog, Trace

ACTIVITY_KEY = "concept:name"
CASE_ID_KEY = "concept:name"


class EvaluationTimeout(TimeoutError):
    """Raised when one fitness/precision evaluation exceeds its time limit."""


def _raise_evaluation_timeout(signum: int, frame: Any) -> None:
    del signum, frame
    raise EvaluationTimeout("Alignment evaluation exceeded its time limit.")


@contextmanager
def evaluation_timeout(seconds: float):
    """
    Limit one alignment fitness + precision evaluation.

    The timer runs inside the current worker process, so the existing
    multiprocessing structure is unchanged. A value <= 0 disables it.
    """
    seconds = float(seconds)
    if seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError(
            "timeout_seconds_per_pair requires Linux or macOS."
        )

    previous_handler = signal.signal(
        signal.SIGALRM,
        _raise_evaluation_timeout,
    )
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
    "log_variant_type",
    "log_variant_index",
    "mutation_order",
    "mutation_parameters",
    "seed",
    "duplicate_retry",
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
    comparable = copy.deepcopy(config)
    comparable.pop("run", None)
    comparable.pop("overwrite_output", None)
    comparable.pop("evaluation", None)  # Workers may change during resume.
    comparable.pop("paths", None)
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
        "models_dir": run_dir / "models",
        "logs_dir": run_dir / "logs",
        "manifests_dir": run_dir / "manifests",
        "results_path": run_dir / "results.csv",
        "used_config_path": run_dir / "used_config.json",
        "state_path": run_dir / "run_state.json",
        "combined_results_path": output_dir / "results.csv",
    }

    for key in ("models_dir", "logs_dir", "manifests_dir"):
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


def get_model_activities(tree: Any) -> List[str]:
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


# ---------------------------------------------------------------------------
# Step 1: Generate model M_base
# ---------------------------------------------------------------------------


def generate_base_model(config: Dict[str, Any], model_index: int) -> Tuple[Any, int]:
    model_seed = int(config["seed"]) + model_index
    seed_everything(model_seed)

    parameters = dict(config["base_generation"]["tree_parameters"])
    parameters["no_models"] = 1
    tree = tree_generator.apply(
        variant=tree_generator.Variants.PTANDLOGGENERATOR,
        parameters=parameters,
    )
    return tree, model_seed


# ---------------------------------------------------------------------------
# Step 2: Generate log L_base from M_base
# ---------------------------------------------------------------------------


def generate_base_log(
    tree: Any, config: Dict[str, Any], model_index: int
) -> Tuple[EventLog, int]:
    log_seed = int(config["seed"]) + 100_000 + model_index
    seed_everything(log_seed)

    num_traces = int(config["base_generation"]["num_traces_per_model"])
    log = tree_playout.apply(
        tree,
        variant=tree_playout.Variants.BASIC_PLAYOUT,
        parameters={"num_traces": num_traces},
    )
    return log, log_seed

# ---------------------------------------------------------------------------
# Step 3: Generate log set L_variants from L_base
# ---------------------------------------------------------------------------


def mutate_biased(
    sequences: Sequence[Sequence[str]], parameters: Dict[str, Any], rng: random.Random
) -> List[List[str]]:
    variants = sorted({trace_variant(sequence) for sequence in sequences})
    if len(variants) < 2:
        return [list(sequence) for sequence in sequences]

    favored_variant_fraction = float(parameters["favored_variant_fraction"])
    favored_trace_fraction = float(parameters["favored_trace_fraction"])

    num_favored = max(1, round(len(variants) * favored_variant_fraction))
    num_favored = min(num_favored, len(variants) - 1)

    favored = rng.sample(variants, num_favored)
    non_favored = [variant for variant in variants if variant not in set(favored)]

    total_traces = len(sequences)
    num_favored_traces = round(total_traces * favored_trace_fraction)
    num_favored_traces = min(max(1, num_favored_traces), total_traces - 1)

    result = [list(rng.choice(favored)) for _ in range(num_favored_traces)]
    result.extend(
        list(rng.choice(non_favored))
        for _ in range(total_traces - num_favored_traces)
    )
    rng.shuffle(result)
    return result


def mutate_downsampled(
    sequences: Sequence[Sequence[str]], parameters: Dict[str, Any], rng: random.Random
) -> List[List[str]]:
    keep_fraction = float(parameters["keep_trace_fraction"])
    keep_count = max(1, round(len(sequences) * keep_fraction))
    keep_count = min(keep_count, len(sequences))
    selected_indices = sorted(rng.sample(range(len(sequences)), keep_count))
    return [list(sequences[index]) for index in selected_indices]


def mutate_delete_events(
    sequences: Sequence[Sequence[str]], parameters: Dict[str, Any], rng: random.Random
) -> List[List[str]]:
    probability = float(parameters["event_probability"])
    result = []
    for sequence in sequences:
        changed = [activity for activity in sequence if rng.random() >= probability]
        if changed:  # Empty traces are intentionally discarded.
            result.append(changed)
    return result


def mutate_replace_events(
    sequences: Sequence[Sequence[str]],
    parameters: Dict[str, Any],
    rng: random.Random,
    model_activities: Sequence[str],
) -> List[List[str]]:
    probability = float(parameters["event_probability"])
    result: List[List[str]] = []

    for sequence in sequences:
        changed: List[str] = []
        for activity in sequence:
            alternatives = [candidate for candidate in model_activities if candidate != activity]
            if alternatives and rng.random() < probability:
                changed.append(rng.choice(alternatives))
            else:
                changed.append(activity)
        result.append(changed)
    return result


def mutate_foreign_replace_events(
    sequences: Sequence[Sequence[str]],
    parameters: Dict[str, Any],
    rng: random.Random,
    foreign_activities: Sequence[str],
) -> List[List[str]]:
    """
    Replace events with activities that do not exist in the model.

    This preserves the number of traces and the length of every trace.
    """
    probability = float(parameters["event_probability"])
    result: list[list[str]] = []

    for sequence in sequences:
        changed: list[str] = []

        for activity in sequence:
            if rng.random() < probability:
                changed.append(rng.choice(foreign_activities))
            else:
                changed.append(activity)

        result.append(changed)

    return result


def mutate_insert_events(
    sequences: Sequence[Sequence[str]],
    parameters: Dict[str, Any],
    rng: random.Random,
    foreign_activities: Sequence[str],
) -> List[List[str]]:
    probability = float(parameters["event_probability"])
    result: List[List[str]] = []

    for sequence in sequences:
        changed: List[str] = []
        for activity in sequence:
            foreign = rng.choice(foreign_activities)
            should_insert = rng.random() < probability
            insert_before = rng.random() < 0.5

            if should_insert and insert_before:
                changed.append(foreign)
            changed.append(activity)
            if should_insert and not insert_before:
                changed.append(foreign)
        result.append(changed)
    return result


def mutate_reorder_events(
    sequences: Sequence[Sequence[str]], parameters: Dict[str, Any], rng: random.Random
) -> List[List[str]]:
    trace_probability = float(parameters["trace_probability"])
    window_fraction = float(parameters["window_fraction"])
    result: List[List[str]] = []

    for sequence in sequences:
        changed = list(sequence)
        if len(changed) >= 2 and rng.random() < trace_probability:
            window_size = max(2, round(len(changed) * window_fraction))
            window_size = min(window_size, len(changed))
            start = rng.randint(0, len(changed) - window_size)
            original_window = changed[start : start + window_size]
            shuffled_window = list(original_window)

            for _ in range(10):
                rng.shuffle(shuffled_window)
                if shuffled_window != original_window:
                    break

            changed[start : start + window_size] = shuffled_window
        result.append(changed)
    return result


def mutate_truncate_traces(
    sequences: Sequence[Sequence[str]], parameters: Dict[str, Any], rng: random.Random
) -> List[List[str]]:
    trace_probability = float(parameters["trace_probability"])
    keep_fraction = float(parameters["keep_fraction"])
    minimum_length = int(parameters.get("minimum_trace_length", 1))
    result: List[List[str]] = []

    for sequence in sequences:
        changed = list(sequence)
        if len(changed) > minimum_length and rng.random() < trace_probability:
            new_length = max(minimum_length, round(len(changed) * keep_fraction))
            new_length = min(new_length, len(changed) - 1)
            changed = changed[:new_length]
        if changed:
            result.append(changed)
    return result


def mutate_duplicate_events(
    sequences: Sequence[Sequence[str]], parameters: Dict[str, Any], rng: random.Random
) -> List[List[str]]:
    probability = float(parameters["event_probability"])
    result: List[List[str]] = []

    for sequence in sequences:
        changed: List[str] = []
        for activity in sequence:
            changed.append(activity)
            if rng.random() < probability:
                changed.append(activity)
        result.append(changed)
    return result


def apply_mutation(
    sequences: Sequence[Sequence[str]],
    mutation_name: str,
    parameters: Dict[str, Any],
    rng: random.Random,
    model_activities: Sequence[str],
    foreign_activities: Sequence[str],
) -> List[List[str]]:
    if mutation_name == "biased":
        return mutate_biased(sequences, parameters, rng)
    if mutation_name == "downsampled":
        return mutate_downsampled(sequences, parameters, rng)
    if mutation_name == "delete_events":
        return mutate_delete_events(sequences, parameters, rng)
    if mutation_name == "replace_events":
        return mutate_replace_events(sequences, parameters, rng, model_activities)
    if mutation_name == "foreign_replace_events":
        return mutate_foreign_replace_events(sequences, parameters, rng, foreign_activities)
    if mutation_name == "insert_events":
        return mutate_insert_events(sequences, parameters, rng, foreign_activities)
    if mutation_name == "reorder_events":
        return mutate_reorder_events(sequences, parameters, rng)
    if mutation_name == "truncate_traces":
        return mutate_truncate_traces(sequences, parameters, rng)
    if mutation_name == "duplicate_events":
        return mutate_duplicate_events(sequences, parameters, rng)
    raise ValueError(f"Unknown mutation: {mutation_name}")


def create_mutated_candidate(
    base_sequences: Sequence[Sequence[str]],
    mutation_name: str,
    mutation_config: Dict[str, Any],
    rng: random.Random,
    model_activities: Sequence[str],
    foreign_activities: Sequence[str],
) -> Tuple[List[List[str]], List[str], Dict[str, Any]]:
    if mutation_name != "mixed":
        parameters = {
            key: value
            for key, value in mutation_config.items()
            if key not in {"enabled", "num_variants"}
        }
        mutated = apply_mutation(
            base_sequences,
            mutation_name,
            parameters,
            rng,
            model_activities,
            foreign_activities,
        )
        return mutated, [mutation_name], parameters

    mutation_order = list(mutation_config["mutation_order"])
    if mutation_config.get("shuffle_order", False):
        rng.shuffle(mutation_order)

    operation_configs = mutation_config["operations"]
    mutated = [list(sequence) for sequence in base_sequences]
    applied_parameters: Dict[str, Any] = {}

    for operation in mutation_order:
        parameters = dict(operation_configs[operation])
        applied_parameters[operation] = parameters
        mutated = apply_mutation(
            mutated,
            operation,
            parameters,
            rng,
            model_activities,
            foreign_activities,
        )
        if not mutated:
            break

    return mutated, mutation_order, applied_parameters


def generate_log_variants(
    base_log: EventLog,
    model_activities: Sequence[str],
    config: Dict[str, Any],
    model_index: int,
) -> List[Dict[str, Any]]:
    base_sequences = log_to_sequences(base_log)
    if not base_sequences:
        raise ValueError("The generated base log is empty.")

    generation_config = config["variant_generation"]
    max_duplicate_retries = int(generation_config["max_duplicate_retries"])
    foreign_pool_size = int(generation_config["foreign_activity_pool_size"])
    foreign_activities = make_foreign_activity_pool(model_activities, foreign_pool_size)

    variants: List[Dict[str, Any]] = []

    # Reserve the original fingerprint even when the clean log is disabled.
    # This prevents an unchanged mutation from being accepted as a new variant.
    fingerprints = {log_fingerprint(base_sequences)}

    clean_settings = config["log_variants"].get("clean", {})

    if clean_settings.get("enabled", False):
        variants.append(
            {
                "variant_type": "clean",
                "variant_index": 1,
                "sequences": [
                    list(sequence)
                    for sequence in base_sequences
                ],
                "mutation_order": [],
                "mutation_parameters": {},
                "seed": int(config["seed"]) + 200_000 + model_index,
                "duplicate_retry": 0,
            }
        )

    mutation_number = 0
    for mutation_name, mutation_config in config["log_variants"].items():
        if mutation_name == "clean" or not mutation_config.get("enabled", False):
            continue

        mutation_number += 1
        requested_variants = int(mutation_config.get("num_variants", 1))

        for variant_index in range(1, requested_variants + 1):
            seed_base = (
                int(config["seed"])
                + model_index * 1_000_000
                + mutation_number * 10_000
                + variant_index * 100
            )

            accepted: Optional[Dict[str, Any]] = None
            for retry in range(max_duplicate_retries + 1):
                candidate_seed = seed_base + retry
                rng = random.Random(candidate_seed)
                sequences, order, parameters = create_mutated_candidate(
                    base_sequences,
                    mutation_name,
                    mutation_config,
                    rng,
                    model_activities,
                    foreign_activities,
                )

                if not sequences:
                    continue

                fingerprint = log_fingerprint(sequences)
                if fingerprint in fingerprints:
                    continue

                fingerprints.add(fingerprint)
                accepted = {
                    "variant_type": mutation_name,
                    "variant_index": variant_index,
                    "sequences": sequences,
                    "mutation_order": order,
                    "mutation_parameters": parameters,
                    "seed": candidate_seed,
                    "duplicate_retry": retry,
                }
                break

            if accepted is None:
                print(
                    f"  Warning: could not create a unique {mutation_name} "
                    f"variant {variant_index}; it was skipped."
                )
            else:
                variants.append(accepted)

    return variants



# ---------------------------------------------------------------------------
# Step 4: Evaluate each model-log pair
# ---------------------------------------------------------------------------


def evaluate_saved_pair(task: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one saved PNML/XES pair inside a worker process."""
    # print(
    #     f"[worker {os.getpid()}] started pair "
    #     f"{task['metadata']['model_name']} × {task['metadata']['log_name']}",
    #     flush=True,
    # )

    result = dict(task["metadata"])
    timeout_seconds = float(task.get("timeout_seconds", 0) or 0)
    started = time.perf_counter()

    try:
        net, initial_marking, final_marking = pm4py.read_pnml(
            task["model_path"]
        )
        log = xes_importer.apply(
            task["log_path"],
            parameters={"show_progress_bar": False},
        )
        common_parameters = {
            "show_progress_bar": False,
            "multiprocessing": False,
        }

        # Only the expensive conformance computation is timed.
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
                "average_trace_fitness": fitness.get(
                    "average_trace_fitness"
                ),
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

    result["evaluation_seconds"] = round(
        time.perf_counter() - started,
        6,
    )
    return result


# ---------------------------------------------------------------------------
# Step 5: Save artifacts and each completed result
# ---------------------------------------------------------------------------


def remove_incomplete_model_files(model_index: int, paths: Dict[str, Path]) -> None:
    model_name = f"model_{model_index:04d}.pnml"
    (paths["models_dir"] / model_name).unlink(missing_ok=True)
    (paths["manifests_dir"] / f"model_{model_index:04d}.json").unlink(
        missing_ok=True
    )
    for log_path in paths["logs_dir"].glob(f"model_{model_index:04d}_*.xes"):
        log_path.unlink()


def save_model_and_logs(
    tree: Any,
    variants: List[Dict[str, Any]],
    model_index: int,
    model_activities: Sequence[str],
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    """Write the manifest last; its presence marks this model as complete."""
    remove_incomplete_model_files(model_index, paths)

    model_name = f"model_{model_index:04d}.pnml"
    model_path = paths["models_dir"] / model_name
    net, initial_marking, final_marking = tree_converter.apply(tree)
    pm4py.write_pnml(
        net, initial_marking, final_marking, str(model_path)
    )

    saved_variants: List[Dict[str, Any]] = []
    for variant in variants:
        variant_type = str(variant["variant_type"])
        variant_index = int(variant["variant_index"])
        log_name = f"model_{model_index:04d}_{variant_type}_{variant_index:03d}.xes"
        log_path = paths["logs_dir"] / log_name
        sequences = variant["sequences"]
        event_log = sequences_to_log(
            sequences, case_prefix=log_name.removesuffix(".xes")
        )
        xes_exporter.apply(
            event_log, str(log_path), parameters={"show_progress_bar": False}
        )

        saved_variants.append(
            {
                "log_name": log_name,
                "variant_type": variant_type,
                "variant_index": variant_index,
                "mutation_order": variant["mutation_order"],
                "mutation_parameters": variant["mutation_parameters"],
                "seed": variant["seed"],
                "duplicate_retry": variant["duplicate_retry"],
                "num_traces": len(sequences),
                "num_trace_variants": len(
                    {trace_variant(sequence) for sequence in sequences}
                ),
                "num_log_activities": count_log_activities(sequences),
            }
        )

    manifest = {
        "model_index": model_index,
        "model_name": model_name,
        "num_model_activities": len(model_activities),
        "process_tree": str(tree),
        "variants": saved_variants,
    }
    manifest_path = paths["manifests_dir"] / f"model_{model_index:04d}.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def load_complete_manifest(
    model_index: int, paths: Dict[str, Path]
) -> Optional[Dict[str, Any]]:
    manifest_path = paths["manifests_dir"] / f"model_{model_index:04d}.json"
    if not manifest_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not (paths["models_dir"] / manifest["model_name"]).exists():
        return None
    if any(
        not (paths["logs_dir"] / variant["log_name"]).exists()
        for variant in manifest["variants"]
    ):
        return None
    return manifest


def build_tasks_from_manifest(
    run_id: str, manifest: Dict[str, Any], paths: Dict[str, Path]
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    model_name = manifest["model_name"]

    for variant in manifest["variants"]:
        metadata = {
            "run_id": run_id,
            "model_name": model_name,
            "log_name": variant["log_name"],
            "model_index": manifest["model_index"],
            "log_variant_type": variant["variant_type"],
            "log_variant_index": variant["variant_index"],
            "mutation_order": ">".join(variant["mutation_order"]),
            "mutation_parameters": json.dumps(
                variant["mutation_parameters"], sort_keys=True
            ),
            "seed": variant["seed"],
            "duplicate_retry": variant["duplicate_retry"],
            "num_traces": variant["num_traces"],
            "num_trace_variants": variant["num_trace_variants"],
            "num_model_activities": manifest["num_model_activities"],
            "num_log_activities": variant["num_log_activities"],
            "process_tree": manifest["process_tree"],
        }
        tasks.append(
            {
                "model_path": str((paths["models_dir"] / model_name).resolve()),
                "log_path": str(
                    (paths["logs_dir"] / variant["log_name"]).resolve()
                ),
                "metadata": metadata,
            }
        )
    return tasks


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
    """Create a result row when the parent kills or loses a worker."""
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
    """Reusable worker. It processes one pair at a time."""
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

    hard_limit = (
        timeout_seconds + grace_seconds
        if timeout_seconds > 0
        else 0
    )

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

    def restart_worker(worker_id: int) -> None:
        stop_worker(worker_id)
        start_worker(worker_id)

    def save_row(row: Dict[str, Any]) -> None:
        append_result(writer, file, row)
        completed.append(row)

    num_workers = max(1, min(num_workers, len(tasks)))

    for worker_id in range(num_workers):
        start_worker(worker_id)
        send_next_task(worker_id)

    try:
        with tqdm(total=len(tasks), desc="Evaluating pairs", unit="pair") as progress:
            while len(completed) < len(tasks):
                # Read available results without blocking.
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

                # Kill workers whose current pair exceeded the hard limit.
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
                            f"(limit={timeout_seconds}, "
                            f"grace={grace_seconds})."
                        ),
                        evaluation_seconds=elapsed,
                    )
                    save_row(row)
                    progress.update(1)

                    restart_worker(worker_id)
                    send_next_task(worker_id)

                # Detect unexpected worker death.
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
    """Only the parent process writes, immediately after each pair finishes."""
    completed_rows: List[Dict[str, Any]] = []
    if not tasks:
        return completed_rows

    evaluation = config["evaluation"]
    use_multiprocessing = bool(
        evaluation.get("use_multiprocessing", False)
    )
    timeout_seconds = float(
        evaluation.get("timeout_seconds_per_pair", 0) or 0
    )
    grace_seconds = float(
        evaluation.get("hard_timeout_grace_seconds", 0) or 0
    )

    worker_tasks = [
        {**task, "timeout_seconds": timeout_seconds}
        for task in tasks
    ]

    with results_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_COLUMNS,
            extrasaction="ignore",
        )

        if not use_multiprocessing:
            for task in tqdm(
                worker_tasks,
                desc="Evaluating pairs",
                unit="pair",
            ):
                row = evaluate_saved_pair(task)
                append_result(writer, file, row)
                completed_rows.append(row)

            return completed_rows

        configured_workers = int(evaluation.get("num_workers", 1))
        num_workers = max(
            1,
            min(configured_workers, len(worker_tasks)),
        )

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
            output_file, fieldnames=RESULT_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        for run_id in list_run_ids(paths["runs_dir"]):
            run_results = paths["runs_dir"] / run_id / "results.csv"
            if not run_results.exists():
                continue
            with run_results.open("r", newline="", encoding="utf-8") as input_file:
                writer.writerows(csv.DictReader(input_file))


# ---------------------------------------------------------------------------
# Main program: the five steps in order
# ---------------------------------------------------------------------------


def main(config_path: Path = Path("config.json")) -> None:
    config = load_config(config_path)
    run_id, paths, is_resume = prepare_run(config)
    num_models = int(config["base_generation"]["num_models"])
    all_tasks: List[Dict[str, Any]] = []

    print(f"Run: {run_id} ({'resume' if is_resume else config['run']['mode']})")

    for model_index in range(1, num_models + 1):
        print(f"\nBase model {model_index}/{num_models}")
        manifest = load_complete_manifest(model_index, paths)

        if manifest is None:
            # 1. Generate model M_base.
            tree, _ = generate_base_model(config, model_index)
            model_activities = get_model_activities(tree)
            print(f"  1. Generated model with {len(model_activities)} activities")

            # 2. Generate log L_base from M_base.
            base_log, _ = generate_base_log(tree, config, model_index)
            print(f"  2. Generated base log with {len(base_log)} traces")

            # 3. Generate log set L_variants from L_base.
            variants = generate_log_variants(
                base_log, model_activities, config, model_index
            )
            print(f"  3. Generated {len(variants)} logs including the clean log")

            manifest = save_model_and_logs(
                tree, variants, model_index, model_activities, paths
            )
        else:
            print("  1-3. Reused complete saved model and logs")

        all_tasks.extend(build_tasks_from_manifest(run_id, manifest, paths))

    # 4. Evaluate every missing model-log pair.
    retry_timeouts = bool(
        config["evaluation"].get("retry_timeouts", True)
    )
    completed_pairs = prepare_results_for_resume(
        paths["results_path"],
        retry_timeouts=retry_timeouts,
    )
    pending_tasks = [
        task for task in all_tasks
        if pair_key(task["metadata"]) not in completed_pairs
    ]
    print(
        f"\n4. Evaluating {len(pending_tasks)} missing pairs "
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

    # 5. Results were saved pair-by-pair; rebuild the all-runs CSV.
    rebuild_combined_results(paths)
    save_state(paths["state_path"], status="complete")
    successful_new = sum(row["status"] == "ok" for row in new_rows)
    print(f"5. Results saved immediately to {paths['results_path']}")
    print(f"   New successful evaluations: {successful_new}/{len(new_rows)}")
    print(f"   Combined results: {paths['combined_results_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=Path, nargs="?", default=Path("config.json")
    )
    arguments = parser.parse_args()
    main(arguments.config)