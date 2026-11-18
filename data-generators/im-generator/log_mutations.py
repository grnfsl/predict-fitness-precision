"""Log-side mutations for the mutation-only dataset generator."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def trace_variant(sequence: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sequence)


def mutate_biased(
    sequences: Sequence[Sequence[str]],
    parameters: Dict[str, Any],
    rng: random.Random,
) -> List[List[str]]:
    variants = sorted({trace_variant(sequence) for sequence in sequences})
    if len(variants) < 2:
        return [list(sequence) for sequence in sequences]

    favored_variant_fraction = float(parameters["favored_variant_fraction"])
    favored_trace_fraction = float(parameters["favored_trace_fraction"])

    num_favored = max(1, round(len(variants) * favored_variant_fraction))
    num_favored = min(num_favored, len(variants) - 1)

    favored = rng.sample(variants, num_favored)
    favored_set = set(favored)
    non_favored = [variant for variant in variants if variant not in favored_set]

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
    sequences: Sequence[Sequence[str]],
    parameters: Dict[str, Any],
    rng: random.Random,
) -> List[List[str]]:
    keep_fraction = float(parameters["keep_trace_fraction"])
    keep_count = max(1, round(len(sequences) * keep_fraction))
    keep_count = min(keep_count, len(sequences))
    selected_indices = sorted(rng.sample(range(len(sequences)), keep_count))
    return [list(sequences[index]) for index in selected_indices]


def mutate_delete_events(
    sequences: Sequence[Sequence[str]],
    parameters: Dict[str, Any],
    rng: random.Random,
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
    result: List[List[str]] = []

    for sequence in sequences:
        changed: List[str] = []
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
    sequences: Sequence[Sequence[str]],
    parameters: Dict[str, Any],
    rng: random.Random,
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
    sequences: Sequence[Sequence[str]],
    parameters: Dict[str, Any],
    rng: random.Random,
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
    sequences: Sequence[Sequence[str]],
    parameters: Dict[str, Any],
    rng: random.Random,
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


def apply_one_log_mutation(
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
    raise ValueError(f"Unknown log mutation: {mutation_name}")


def apply_log_mutations(
    base_sequences: Sequence[Sequence[str]],
    mutations_config: Dict[str, Dict[str, Any]],
    rng: random.Random,
    model_activities: Sequence[str],
    foreign_activities: Sequence[str],
    mutation_order: Optional[Sequence[str]] = None,
    shuffle_mutations: bool = False,
) -> Tuple[List[List[str]], List[str], Dict[str, Any]]:
    """Apply the configured log mutations in order and return sequences/order/params."""
    if not mutations_config:
        return [list(sequence) for sequence in base_sequences], [], {}

    order = list(mutation_order) if mutation_order is not None else list(mutations_config)
    if shuffle_mutations:
        rng.shuffle(order)

    mutated = [list(sequence) for sequence in base_sequences]
    applied_parameters: Dict[str, Any] = {}
    applied_order: List[str] = []

    for operation in order:
        if operation not in mutations_config:
            raise ValueError(f"Missing config for log mutation: {operation}")

        parameters = dict(mutations_config[operation])
        applied_parameters[operation] = parameters
        mutated = apply_one_log_mutation(
            mutated,
            operation,
            parameters,
            rng,
            model_activities,
            foreign_activities,
        )
        applied_order.append(operation)
        if not mutated:
            break

    return mutated, applied_order, applied_parameters