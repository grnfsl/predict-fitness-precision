from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import random

import dgl
import torch
import pandas as pd

from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_eval


CASE_COL_CANDIDATES = ("case:concept:name", "case_id", "caseid", "trace_id")
ACT_COL_CANDIDATES = ("concept:name", "activity", "event_name")


@dataclass
class PairExampleSpec:
    '''
    Prepared specification for one (log, Petri net) pair.

    The graph can be rebuilt deterministically from this spec.
    '''
    variants: List[Tuple[Tuple[str, ...], int]]
    total_cases: int
    transition_labels: List[Optional[str]]
    transition_visible: List[bool]
    place_initial: List[float]
    place_final: List[float]
    pt_edges: List[Tuple[int, int]]   # place -> transition
    tp_edges: List[Tuple[int, int]]   # transition -> place
    fitness: float
    precision: float
    max_local_activity_count: int
    activity_order: Tuple[str, ...]
    sample_weight: float = 1.0


def _find_case_col(log_df: pd.DataFrame) -> str:
    for col in CASE_COL_CANDIDATES:
        if col in log_df.columns:
            return col
    raise KeyError(
        f"Could not find a case id column. Tried: {CASE_COL_CANDIDATES}. "
        f"Available columns: {list(log_df.columns)}"
    )


def _find_act_col(log_df: pd.DataFrame) -> str:
    for col in ACT_COL_CANDIDATES:
        if col in log_df.columns:
            return col
    raise KeyError(
        f"Could not find an activity column. Tried: {ACT_COL_CANDIDATES}. "
        f"Available columns: {list(log_df.columns)}"
    )


def extract_trace_variants(log_df: pd.DataFrame) -> Tuple[List[Tuple[Tuple[str, ...], int]], int]:
    '''
    Convert an event log dataframe into unique trace variants with frequencies.
    Keeps only unique variants; repeated traces are represented by counts.
    '''
    case_col = _find_case_col(log_df)
    act_col = _find_act_col(log_df)

    variants_counter: Counter[Tuple[str, ...]] = Counter()
    for _, grp in log_df.groupby(case_col, sort=False):
        seq = tuple(str(x) for x in grp[act_col].tolist())
        variants_counter[seq] += 1

    variants = sorted(variants_counter.items(), key=lambda x: (-x[1], x[0]))
    total_cases = int(sum(variants_counter.values()))
    return variants, total_cases



def normalize_precomputed_variants(
    variants_input: Sequence[Tuple[Sequence[str], int]],
) -> Tuple[List[Tuple[Tuple[str, ...], int]], int]:
    '''
    Normalize an already-compressed log representation into the same internal
    format returned by extract_trace_variants().

    Expected input format:
        [
            (['a', 'b', 'c'], 13),
            (['a', 'c', 'b'], 7),
            ...
        ]

    Duplicate variants are merged, activities are converted to strings,
    non-positive counts are ignored, and the output is sorted deterministically.
    '''
    variants_counter: Counter[Tuple[str, ...]] = Counter()

    for seq, count in variants_input:
        seq_tuple = tuple(str(x) for x in seq)
        count_int = int(count)
        if count_int <= 0:
            continue
        variants_counter[seq_tuple] += count_int

    variants = sorted(variants_counter.items(), key=lambda x: (-x[1], x[0]))
    total_cases = int(sum(variants_counter.values()))
    return variants, total_cases

def compute_alignment_targets(log_df: pd.DataFrame, net, im, fm) -> Tuple[float, float]:
    '''
    Alignment-based replay fitness and alignment-based precision.
    '''
    params = {"show_progress_bar": False}
    fit = replay_fitness.apply(
        log_df,
        net,
        im,
        fm,
        variant=replay_fitness.Variants.ALIGNMENT_BASED,
        parameters=params,
    )
    prec = precision_eval.apply(
        log_df,
        net,
        im,
        fm,
        variant=precision_eval.Variants.ALIGN_ETCONFORMANCE,
        parameters=params,
    )
    return float(fit["average_trace_fitness"]), float(prec)


def prepare_pair_spec(
    log_df: Any,
    net,
    im,
    fm,
    fitness: Optional[float] = None,
    precision: Optional[float] = None,
    compute_targets_if_missing: bool = True,
    sample_weight: float = 1.0,
) -> PairExampleSpec:
    '''
    Prepare a serializable spec for one (log, net) pair.
    '''
    # ``log_df`` may be either:
    #   1. a pandas DataFrame event log, or
    #   2. a precomputed variant list: [([activity, ...], count), ...].
    # The lazy dataset uses option 2 to avoid keeping full XES logs in RAM.
    log_df_for_metrics = None
    if isinstance(log_df, pd.DataFrame):
        variants, total_cases = extract_trace_variants(log_df)
        log_df_for_metrics = log_df
    else:
        variants, total_cases = normalize_precomputed_variants(log_df)


    place_to_idx = {p: i for i, p in enumerate(net.places)}
    trans_to_idx = {t: i for i, t in enumerate(net.transitions)}

    transition_labels: List[Optional[str]] = []
    transition_visible: List[bool] = []
    for t in net.transitions:
        label = None if t.label is None else str(t.label)
        transition_labels.append(label)
        transition_visible.append(label is not None)

    place_initial = [0.0 for _ in net.places]
    place_final = [0.0 for _ in net.places]
    for p, w in im.items():
        place_initial[place_to_idx[p]] = float(w)
    for p, w in fm.items():
        place_final[place_to_idx[p]] = float(w)

    pt_edges: List[Tuple[int, int]] = []
    tp_edges: List[Tuple[int, int]] = []
    for p in net.places:
        for arc in p.out_arcs:
            t = arc.target
            pt_edges.append((place_to_idx[p], trans_to_idx[t]))
        for arc in p.in_arcs:
            t = arc.source
            tp_edges.append((trans_to_idx[t], place_to_idx[p]))

    visible_acts = sorted({a for var, _ in variants for a in var})
    if fitness is None or precision is None:
        if not compute_targets_if_missing:
            raise ValueError("fitness/precision missing and compute_targets_if_missing=False")
        if log_df_for_metrics is None:
            raise ValueError(
                "Cannot compute fitness/precision from precomputed variants only. "
                "Pass precomputed_targets or pass a pandas DataFrame log."
            )
        fitness, precision = compute_alignment_targets(log_df_for_metrics, net, im, fm)

    return PairExampleSpec(
        variants=variants,
        total_cases=total_cases,
        transition_labels=transition_labels,
        transition_visible=transition_visible,
        place_initial=place_initial,
        place_final=place_final,
        pt_edges=pt_edges,
        tp_edges=tp_edges,
        fitness=float(fitness),
        precision=float(precision),
        max_local_activity_count=len(visible_acts),
        activity_order=tuple(visible_acts),
        sample_weight=float(sample_weight),
    )


def prepare_pair_specs(
    raw_pairs: Sequence[Tuple[Any, Any, Any, Any]],
    precomputed_targets: Optional[Sequence[Tuple[float, float]]] = None,
    compute_targets_if_missing: bool = True,
    precomputed_sample_weights: Optional[Sequence[float]] = None,
) -> List[PairExampleSpec]:
    specs: List[PairExampleSpec] = []
    for i, (log_df, net, im, fm) in enumerate(raw_pairs):
        fit = prec = None
        if precomputed_targets is not None:
            fit, prec = precomputed_targets[i]

        sample_weight = 1.0
        if precomputed_sample_weights is not None:
            sample_weight = float(precomputed_sample_weights[i])

        specs.append(
            prepare_pair_spec(
                log_df=log_df,
                net=net,
                im=im,
                fm=fm,
                fitness=fit,
                precision=prec,
                compute_targets_if_missing=compute_targets_if_missing,
                sample_weight=sample_weight,
            )
        )
    return specs


def dataset_max_local_activity_dim(specs: Sequence[PairExampleSpec]) -> int:
    return max((spec.max_local_activity_count for spec in specs), default=0)


def _trace_freq_feature(variant_count: int, total_cases: int, mode: str) -> float:
    if total_cases <= 0:
        return 0.0
    if mode == "normalized":
        return float(variant_count) / float(total_cases)
    if mode == "log":
        return math.log1p(float(variant_count))
    raise ValueError(f"Unknown trace_freq_mode={mode!r}")


def _build_local_activity_basis(
    activity_order: Sequence[str],
    max_local_onehot_dim: int,
    random_permute_onehot: bool,
    rng: Optional[random.Random] = None,
) -> Dict[str, int]:
    acts = list(activity_order)
    if len(acts) > max_local_onehot_dim:
        raise ValueError(
            f"Local activity count {len(acts)} exceeds max_local_onehot_dim={max_local_onehot_dim}"
        )

    slot_ids = list(range(len(acts)))
    if random_permute_onehot:
        if rng is None:
            rng = random
        rng.shuffle(slot_ids)

    return {act: slot_ids[i] for i, act in enumerate(acts)}


def _directed_transition_structural_features(
    spec: PairExampleSpec,
) -> List[List[float]]:
    """
    Compute the raw 5-D q_t structural vector for every transition:

        [
            is_silent,
            distance_from_start_norm,
            distance_to_end_norm,
            reachable_from_start,
            can_reach_final,
        ]

    Distances are ordinary directed shortest-path hop distances on the
    bipartite Petri-net graph:

        place -> transition -> place

    They are very cheap to compute: one multi-source BFS from the initial
    places and one reverse multi-source BFS from the final places.

    For an unreachable transition, the corresponding normalized distance is
    0 and the reachability flag is 0, so "unreachable" is distinguishable
    from a genuinely short distance.
    """
    num_pnodes = len(spec.place_initial)
    num_tnodes = len(spec.transition_labels)
    num_nodes = num_pnodes + num_tnodes

    def p_gid(pidx: int) -> int:
        return int(pidx)

    def t_gid(tidx: int) -> int:
        return num_pnodes + int(tidx)

    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    rev_adj: List[List[int]] = [[] for _ in range(num_nodes)]

    for pidx, tidx in spec.pt_edges:
        u = p_gid(pidx)
        v = t_gid(tidx)
        adj[u].append(v)
        rev_adj[v].append(u)

    for tidx, pidx in spec.tp_edges:
        u = t_gid(tidx)
        v = p_gid(pidx)
        adj[u].append(v)
        rev_adj[v].append(u)

    def multi_source_bfs(
        graph_adj: List[List[int]],
        sources: List[int],
    ) -> List[Optional[int]]:
        dist: List[Optional[int]] = [None] * num_nodes
        q = deque()

        for s in sources:
            if 0 <= s < num_nodes and dist[s] is None:
                dist[s] = 0
                q.append(s)

        while q:
            u = q.popleft()
            next_dist = int(dist[u]) + 1

            for v in graph_adj[u]:
                if dist[v] is None:
                    dist[v] = next_dist
                    q.append(v)

        return dist

    initial_places = [
        p_gid(i)
        for i, value in enumerate(spec.place_initial)
        if float(value) > 0.0
    ]

    final_places = [
        p_gid(i)
        for i, value in enumerate(spec.place_final)
        if float(value) > 0.0
    ]

    dist_from_start = multi_source_bfs(
        adj,
        initial_places,
    )

    # Reverse traversal from final places gives directed distance-to-final.
    dist_to_end = multi_source_bfs(
        rev_adj,
        final_places,
    )

    finite_start = [
        dist_from_start[t_gid(i)]
        for i in range(num_tnodes)
        if dist_from_start[t_gid(i)] is not None
    ]

    finite_end = [
        dist_to_end[t_gid(i)]
        for i in range(num_tnodes)
        if dist_to_end[t_gid(i)] is not None
    ]

    max_start = max([1] + [int(x) for x in finite_start])
    max_end = max([1] + [int(x) for x in finite_end])

    features: List[List[float]] = []

    for tidx in range(num_tnodes):
        ds = dist_from_start[t_gid(tidx)]
        de = dist_to_end[t_gid(tidx)]

        reachable_start = 1.0 if ds is not None else 0.0
        reachable_end = 1.0 if de is not None else 0.0

        ds_norm = (
            float(ds) / float(max_start)
            if ds is not None
            else 0.0
        )

        de_norm = (
            float(de) / float(max_end)
            if de is not None
            else 0.0
        )

        is_silent = (
            0.0
            if spec.transition_visible[tidx]
            else 1.0
        )

        features.append(
            [
                is_silent,
                ds_norm,
                de_norm,
                reachable_start,
                reachable_end,
            ]
        )

    return features



# ---------------------------------------------------------------------------
# Global behavioral compatibility statistics (late-fusion branch)
# ---------------------------------------------------------------------------

BEHAVIOR_FEATURE_NAMES = (
    "ef_log_coverage",
    "ef_model_extra_ratio",
    "ef_mean_distance_error",
    "ef_min_distance_error",
    "start_coverage",
    "end_coverage",
    "model_extra_start_ratio",
    "model_extra_end_ratio",
    "loop_coverage",
    "model_extra_loop_ratio",
    "scc_coverage",
    "model_extra_scc_ratio",
    "concurrency_coverage",
    "model_extra_concurrency_ratio",
    "conflict_coverage",
    "model_extra_conflict_ratio",
)

BEHAVIOR_FEATURE_DIM = len(BEHAVIOR_FEATURE_NAMES)


def _tarjan_scc(adj: List[List[int]]) -> Tuple[List[int], List[List[int]]]:
    """Return node->SCC ids and SCC member lists for a directed graph."""
    n = len(adj)
    index = 0
    stack: List[int] = []
    on_stack = [False] * n
    indices = [-1] * n
    lowlink = [0] * n
    components: List[List[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            component: List[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)

    node_to_scc = [-1] * n
    for cid, component in enumerate(components):
        for v in component:
            node_to_scc[v] = cid

    return node_to_scc, components


def _zero_one_bfs(
    adj_cost: List[List[Tuple[int, int]]],
    sources: Sequence[int],
) -> List[Optional[int]]:
    """0-1 BFS used for visible-transition distance calculations."""
    dist: List[Optional[int]] = [None] * len(adj_cost)
    q = deque()

    for source in sources:
        source = int(source)
        if 0 <= source < len(adj_cost):
            if dist[source] is None or int(dist[source]) > 0:
                dist[source] = 0
                q.appendleft(source)

    while q:
        u = q.popleft()
        du = int(dist[u])

        for v, cost in adj_cost[u]:
            nd = du + int(cost)
            if dist[v] is None or nd < int(dist[v]):
                dist[v] = nd
                if cost == 0:
                    q.appendleft(v)
                else:
                    q.append(v)

    return dist


def _log_behavior_statistics(
    spec: PairExampleSpec,
    activity_order: Sequence[str],
):
    """
    Cheap log-side behavioral summaries computed directly from trace variants.

    For each ordered pair (A, B), the nearest later B is used for every
    occurrence of A. This avoids combinatorial over-counting in looping traces.
    No Petri-net replay is involved.
    """
    activities = list(activity_order)
    total_cases = max(1, int(spec.total_cases))
    distance_scale = max(1, len(activities))

    start_freq = {a: 0.0 for a in activities}
    end_freq = {a: 0.0 for a in activities}
    repeat_freq = {a: 0.0 for a in activities}
    activity_case_mass = {a: 0.0 for a in activities}

    pair_case_mass: Dict[Tuple[str, str], float] = {}
    pair_cooccur_mass: Dict[Tuple[str, str], float] = {}
    pair_min_dist: Dict[Tuple[str, str], float] = {}
    pair_dist_sum: Dict[Tuple[str, str], float] = {}
    pair_dist_weight: Dict[Tuple[str, str], float] = {}

    for trace_raw, count_raw in spec.variants:
        trace = [str(a) for a in trace_raw]
        count = float(count_raw)
        if not trace or count <= 0.0:
            continue

        start_freq[trace[0]] = start_freq.get(trace[0], 0.0) + count
        end_freq[trace[-1]] = end_freq.get(trace[-1], 0.0) + count

        present = set(trace)
        for act in present:
            activity_case_mass[act] = activity_case_mass.get(act, 0.0) + count

        counts = Counter(trace)
        for act, n in counts.items():
            if n > 1:
                repeat_freq[act] = repeat_freq.get(act, 0.0) + count

        present_sorted = sorted(present)
        for i, a in enumerate(present_sorted):
            for b in present_sorted[i + 1:]:
                pair_cooccur_mass[(a, b)] = pair_cooccur_mass.get((a, b), 0.0) + count
                pair_cooccur_mass[(b, a)] = pair_cooccur_mass.get((b, a), 0.0) + count

        next_pos: Dict[str, int] = {}
        trace_pairs_seen = set()

        for i in range(len(trace) - 1, -1, -1):
            a = trace[i]

            for b, j in next_pos.items():
                pair = (a, b)
                distance = float(j - i)
                trace_pairs_seen.add(pair)
                pair_min_dist[pair] = min(pair_min_dist.get(pair, distance), distance)
                pair_dist_sum[pair] = pair_dist_sum.get(pair, 0.0) + count * distance
                pair_dist_weight[pair] = pair_dist_weight.get(pair, 0.0) + count

            next_pos[a] = i

        for pair in trace_pairs_seen:
            pair_case_mass[pair] = pair_case_mass.get(pair, 0.0) + count

    for table in (start_freq, end_freq, repeat_freq, activity_case_mass):
        for act in list(table):
            table[act] = float(table[act]) / float(total_cases)

    pair_stats: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for pair, case_mass in pair_case_mass.items():
        min_dist = pair_min_dist[pair]
        mean_dist = pair_dist_sum[pair] / max(pair_dist_weight[pair], 1e-8)
        pair_stats[pair] = (
            float(case_mass) / float(total_cases),
            min(1.0, float(min_dist) / float(distance_scale)),
            min(1.0, float(mean_dist) / float(distance_scale)),
        )

    concurrency: Dict[Tuple[str, str], float] = {}
    exclusivity: Dict[Tuple[str, str], float] = {}

    for i, a in enumerate(activities):
        for b in activities[i + 1:]:
            ab = float(pair_case_mass.get((a, b), 0.0))
            ba = float(pair_case_mass.get((b, a), 0.0))
            order_mass = ab + ba
            conc = 0.0 if order_mass <= 0.0 else 2.0 * min(ab, ba) / order_mass

            support_a = float(activity_case_mass.get(a, 0.0))
            support_b = float(activity_case_mass.get(b, 0.0))
            cooccur = float(pair_cooccur_mass.get((a, b), 0.0)) / float(total_cases)
            union = support_a + support_b - cooccur
            excl = 0.0 if union <= 0.0 else max(0.0, min(1.0, 1.0 - cooccur / union))

            concurrency[(a, b)] = conc
            concurrency[(b, a)] = conc
            exclusivity[(a, b)] = excl
            exclusivity[(b, a)] = excl

    return (
        start_freq,
        end_freq,
        repeat_freq,
        activity_case_mass,
        pair_stats,
        concurrency,
        exclusivity,
    )


def _model_behavior_statistics(
    spec: PairExampleSpec,
    activity_order: Sequence[str],
):
    """
    Cheap static Petri-net behavioral proxies.

    Visible-distance convention:
      entering a visible transition costs 1; places and silent transitions cost 0.
    Thus a direct visible successor has distance 1 even when silent transitions
    lie between the two visible activities.
    """
    activities = list(activity_order)
    num_p = len(spec.place_initial)
    num_t = len(spec.transition_labels)
    num_nodes = num_p + num_t

    def p_gid(pidx: int) -> int:
        return int(pidx)

    def t_gid(tidx: int) -> int:
        return num_p + int(tidx)

    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    adj_cost: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    rev_cost: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    presets = [set() for _ in range(num_t)]
    postsets = [set() for _ in range(num_t)]

    for pidx, tidx in spec.pt_edges:
        u, v = p_gid(pidx), t_gid(tidx)
        adj[u].append(v)
        cost = 1 if spec.transition_visible[tidx] else 0
        adj_cost[u].append((v, cost))
        rev_cost[v].append((u, 0))
        presets[tidx].add(int(pidx))

    for tidx, pidx in spec.tp_edges:
        u, v = t_gid(tidx), p_gid(pidx)
        adj[u].append(v)
        adj_cost[u].append((v, 0))
        cost = 1 if spec.transition_visible[tidx] else 0
        rev_cost[v].append((u, cost))
        postsets[tidx].add(int(pidx))

    label_to_tidx: Dict[str, List[int]] = {}
    for tidx, (label, visible) in enumerate(zip(spec.transition_labels, spec.transition_visible)):
        if visible and label is not None:
            label_to_tidx.setdefault(str(label), []).append(tidx)

    initial_nodes = [p_gid(i) for i, x in enumerate(spec.place_initial) if float(x) > 0.0]
    final_nodes = [p_gid(i) for i, x in enumerate(spec.place_final) if float(x) > 0.0]
    dist_from_start = _zero_one_bfs(adj_cost, initial_nodes)
    dist_to_end = _zero_one_bfs(rev_cost, final_nodes)

    start_flag = {a: 0.0 for a in activities}
    end_flag = {a: 0.0 for a in activities}

    for act, tids in label_to_tidx.items():
        ds = [dist_from_start[t_gid(t)] for t in tids if dist_from_start[t_gid(t)] is not None]
        de = [dist_to_end[t_gid(t)] for t in tids if dist_to_end[t_gid(t)] is not None]
        if ds and min(int(x) for x in ds) == 1:
            start_flag[act] = 1.0
        if de and min(int(x) for x in de) == 1:
            end_flag[act] = 1.0

    node_to_scc, components = _tarjan_scc(adj)
    component_is_cycle: List[bool] = []
    component_visible_labels: List[set] = []

    for component in components:
        labels = set()
        for node in component:
            if node >= num_p:
                tidx = node - num_p
                label = spec.transition_labels[tidx]
                if spec.transition_visible[tidx] and label is not None:
                    labels.add(str(label))
        component_visible_labels.append(labels)
        has_self_loop = any(v in adj[v] for v in component)
        component_is_cycle.append(len(component) > 1 or has_self_loop)

    in_cycle = {a: 0.0 for a in activities}
    for act, tids in label_to_tidx.items():
        for tidx in tids:
            cid = node_to_scc[t_gid(tidx)]
            if cid >= 0 and component_is_cycle[cid]:
                in_cycle[act] = 1.0
                break

    pair_reach: Dict[Tuple[str, str], Tuple[float, float]] = {}
    transition_distances: Dict[int, List[Optional[int]]] = {}

    for tids in label_to_tidx.values():
        for tidx in tids:
            transition_distances[tidx] = _zero_one_bfs(adj_cost, [t_gid(tidx)])

    distance_scale = max(1, len(activities))

    for a, tids_a in label_to_tidx.items():
        for b, tids_b in label_to_tidx.items():
            best: Optional[int] = None
            for ta in tids_a:
                dist = transition_distances[ta]
                for tb in tids_b:
                    if ta == tb:
                        continue
                    value = dist[t_gid(tb)]
                    if value is not None and int(value) > 0:
                        best = int(value) if best is None else min(best, int(value))
            if best is not None:
                pair_reach[(a, b)] = (
                    1.0,
                    min(1.0, float(best) / float(distance_scale)),
                )

    same_scc: Dict[Tuple[str, str], float] = {}
    concurrency_proxy: Dict[Tuple[str, str], float] = {}
    conflict_proxy: Dict[Tuple[str, str], float] = {}

    for i, a in enumerate(activities):
        tids_a = label_to_tidx.get(a, [])
        for b in activities[i + 1:]:
            tids_b = label_to_tidx.get(b, [])
            same_cycle = 0.0
            conflict = 0.0
            independent = 0.0

            for ta in tids_a:
                for tb in tids_b:
                    ca = node_to_scc[t_gid(ta)] if num_nodes > 0 else -1
                    cb = node_to_scc[t_gid(tb)] if num_nodes > 0 else -1
                    if ca >= 0 and ca == cb and component_is_cycle[ca]:
                        same_cycle = 1.0

                    if presets[ta] & presets[tb]:
                        conflict = 1.0

                    no_order = (
                        (a, b) not in pair_reach
                        and (b, a) not in pair_reach
                    )
                    disjoint = (
                        not (presets[ta] & presets[tb])
                        and not (postsets[ta] & postsets[tb])
                    )
                    if no_order and disjoint:
                        independent = 1.0

            same_scc[(a, b)] = same_cycle
            same_scc[(b, a)] = same_cycle
            concurrency_proxy[(a, b)] = independent
            concurrency_proxy[(b, a)] = independent
            conflict_proxy[(a, b)] = conflict
            conflict_proxy[(b, a)] = conflict

    return (
        start_flag,
        end_flag,
        in_cycle,
        pair_reach,
        same_scc,
        concurrency_proxy,
        conflict_proxy,
    )


def _safe_weighted_coverage(
    weights: Sequence[float],
    supported: Sequence[float],
    empty_value: float = 1.0,
) -> float:
    denom = float(sum(weights))
    if denom <= 1e-12:
        return float(empty_value)
    return float(sum(w * s for w, s in zip(weights, supported))) / denom


def _build_global_behavior_stats(
    spec: PairExampleSpec,
    activity_order: Sequence[str],
) -> List[float]:
    """
    Build the 16-D global behavioral compatibility vector used only at final fusion.

    These are explicit statistics, not node embeddings. No behavior GNN is used.
    Every value is normalized to approximately [0, 1].
    """
    activities = list(activity_order)

    (
        log_start,
        log_end,
        log_repeat,
        log_activity_support,
        log_pair,
        log_concurrency,
        log_exclusivity,
    ) = _log_behavior_statistics(spec, activities)

    (
        model_start,
        model_end,
        model_cycle,
        model_pair,
        model_same_scc,
        model_concurrency,
        model_conflict,
    ) = _model_behavior_statistics(spec, activities)

    # ---------------------------------------------------------------
    # Eventually-follows: observed coverage, model extras, and distance.
    # ---------------------------------------------------------------
    log_ef_weights: List[float] = []
    log_ef_supported: List[float] = []
    common_weights: List[float] = []
    mean_distance_errors: List[float] = []
    min_distance_errors: List[float] = []

    for pair, (support, log_min_dist, log_mean_dist) in log_pair.items():
        if pair[0] == pair[1] or support <= 0.0:
            continue
        model_vals = model_pair.get(pair)
        model_exists = 1.0 if model_vals is not None else 0.0
        log_ef_weights.append(float(support))
        log_ef_supported.append(model_exists)

        if model_vals is not None:
            model_dist = float(model_vals[1])
            common_weights.append(float(support))
            mean_distance_errors.append(abs(float(log_mean_dist) - model_dist))
            min_distance_errors.append(abs(float(log_min_dist) - model_dist))

    ef_log_coverage = _safe_weighted_coverage(log_ef_weights, log_ef_supported)

    model_ef_pairs = [pair for pair in model_pair if pair[0] != pair[1]]
    if model_ef_pairs:
        ef_model_extra_ratio = sum(
            1.0 - float(log_pair.get(pair, (0.0, 0.0, 0.0))[0])
            for pair in model_ef_pairs
        ) / float(len(model_ef_pairs))
    else:
        ef_model_extra_ratio = 0.0

    common_denom = float(sum(common_weights))
    if common_denom > 1e-12:
        ef_mean_distance_error = sum(
            w * e for w, e in zip(common_weights, mean_distance_errors)
        ) / common_denom
        ef_min_distance_error = sum(
            w * e for w, e in zip(common_weights, min_distance_errors)
        ) / common_denom
    else:
        ef_mean_distance_error = 0.0
        ef_min_distance_error = 0.0

    # ---------------------------------------------------------------
    # Start / end behavior.
    # ---------------------------------------------------------------
    start_weights = [float(log_start.get(a, 0.0)) for a in activities]
    start_supported = [float(model_start.get(a, 0.0)) for a in activities]
    start_coverage = _safe_weighted_coverage(start_weights, start_supported)

    end_weights = [float(log_end.get(a, 0.0)) for a in activities]
    end_supported = [float(model_end.get(a, 0.0)) for a in activities]
    end_coverage = _safe_weighted_coverage(end_weights, end_supported)

    model_start_acts = [a for a in activities if float(model_start.get(a, 0.0)) > 0.0]
    model_end_acts = [a for a in activities if float(model_end.get(a, 0.0)) > 0.0]

    model_extra_start_ratio = (
        sum(1.0 for a in model_start_acts if float(log_start.get(a, 0.0)) <= 0.0)
        / float(len(model_start_acts))
        if model_start_acts else 0.0
    )
    model_extra_end_ratio = (
        sum(1.0 for a in model_end_acts if float(log_end.get(a, 0.0)) <= 0.0)
        / float(len(model_end_acts))
        if model_end_acts else 0.0
    )

    # ---------------------------------------------------------------
    # Individual loops and pairwise SCC/cycle agreement.
    # ---------------------------------------------------------------
    loop_weights = [float(log_repeat.get(a, 0.0)) for a in activities]
    loop_supported = [float(model_cycle.get(a, 0.0)) for a in activities]
    loop_coverage = _safe_weighted_coverage(loop_weights, loop_supported)

    model_cycle_acts = [a for a in activities if float(model_cycle.get(a, 0.0)) > 0.0]
    model_extra_loop_ratio = (
        sum(1.0 - float(log_repeat.get(a, 0.0)) for a in model_cycle_acts)
        / float(len(model_cycle_acts))
        if model_cycle_acts else 0.0
    )

    unordered_pairs = [
        (activities[i], activities[j])
        for i in range(len(activities))
        for j in range(i + 1, len(activities))
    ]

    scc_weights: List[float] = []
    scc_supported: List[float] = []
    for a, b in unordered_pairs:
        ab = float(log_pair.get((a, b), (0.0, 0.0, 0.0))[0])
        ba = float(log_pair.get((b, a), (0.0, 0.0, 0.0))[0])
        pair_loop_strength = min(ab, ba)
        if pair_loop_strength > 0.0:
            scc_weights.append(pair_loop_strength)
            scc_supported.append(float(model_same_scc.get((a, b), 0.0)))

    scc_coverage = _safe_weighted_coverage(scc_weights, scc_supported)

    model_scc_pairs = [
        (a, b)
        for a, b in unordered_pairs
        if float(model_same_scc.get((a, b), 0.0)) > 0.0
    ]
    model_extra_scc_ratio = (
        sum(
            1.0 - min(
                float(log_pair.get((a, b), (0.0, 0.0, 0.0))[0]),
                float(log_pair.get((b, a), (0.0, 0.0, 0.0))[0]),
            )
            for a, b in model_scc_pairs
        ) / float(len(model_scc_pairs))
        if model_scc_pairs else 0.0
    )

    # ---------------------------------------------------------------
    # Cheap concurrency / conflict proxy agreement.
    # ---------------------------------------------------------------
    conc_weights: List[float] = []
    conc_supported: List[float] = []
    conflict_weights: List[float] = []
    conflict_supported: List[float] = []

    for a, b in unordered_pairs:
        pair_support = min(
            float(log_activity_support.get(a, 0.0)),
            float(log_activity_support.get(b, 0.0)),
        )

        conc_strength = float(log_concurrency.get((a, b), 0.0))
        if conc_strength > 0.0 and pair_support > 0.0:
            conc_weights.append(pair_support * conc_strength)
            conc_supported.append(float(model_concurrency.get((a, b), 0.0)))

        conflict_strength = float(log_exclusivity.get((a, b), 0.0))
        if conflict_strength > 0.0 and pair_support > 0.0:
            conflict_weights.append(pair_support * conflict_strength)
            conflict_supported.append(float(model_conflict.get((a, b), 0.0)))

    concurrency_coverage = _safe_weighted_coverage(conc_weights, conc_supported)
    conflict_coverage = _safe_weighted_coverage(conflict_weights, conflict_supported)

    model_conc_pairs = [
        (a, b)
        for a, b in unordered_pairs
        if float(model_concurrency.get((a, b), 0.0)) > 0.0
    ]
    model_conflict_pairs = [
        (a, b)
        for a, b in unordered_pairs
        if float(model_conflict.get((a, b), 0.0)) > 0.0
    ]

    model_extra_concurrency_ratio = (
        sum(1.0 - float(log_concurrency.get((a, b), 0.0)) for a, b in model_conc_pairs)
        / float(len(model_conc_pairs))
        if model_conc_pairs else 0.0
    )

    model_extra_conflict_ratio = (
        sum(1.0 - float(log_exclusivity.get((a, b), 0.0)) for a, b in model_conflict_pairs)
        / float(len(model_conflict_pairs))
        if model_conflict_pairs else 0.0
    )

    values = [
        ef_log_coverage,
        ef_model_extra_ratio,
        ef_mean_distance_error,
        ef_min_distance_error,
        start_coverage,
        end_coverage,
        model_extra_start_ratio,
        model_extra_end_ratio,
        loop_coverage,
        model_extra_loop_ratio,
        scc_coverage,
        model_extra_scc_ratio,
        concurrency_coverage,
        model_extra_concurrency_ratio,
        conflict_coverage,
        model_extra_conflict_ratio,
    ]

    return [max(0.0, min(1.0, float(v))) for v in values]

def build_alignment_pair_graph(
    spec: PairExampleSpec,
    variant: str = "A",
    max_local_onehot_dim: Optional[int] = None,
    random_permute_onehot: bool = False,
    trace_freq_mode: str = "normalized",
    rng: Optional[random.Random] = None,
) -> dgl.DGLHeteroGraph:
    """
    Build one graph for synchronous Log-GNN -> PGNN + context plus global behavioral statistics.

    Intended experiment
    -------------------
    Variant A (label-independent node features).

    Event-log side:
        enode --next/next_rev--> enode

    Petri-net side:
        pnode --pt--> tnode --tp--> pnode

    Cross relation:
        enode --match--> tnode

    There is deliberately NO match_rev relation.

    A single cnode is connected bidirectionally ONLY to tnodes. Its hidden state still starts at zero.
The same cnode also stores a fixed 16-D behavior_stats vector used ONLY at final fusion.

    Every event occurrence is connected to every visible transition having
    the same activity label. The model uses these edges through dedicated
    learned message transformation + mean aggregation:

        c_t = mean_{e -> t} W_match h_e

    Labels therefore define correspondence only; label identity itself is
    never supplied as a neural feature.

    Variant A neural inputs
    -----------------------
    enode.x:
        [pos_norm, is_first, is_last, trace_freq, trace_len_norm]

    pnode.x:
        [initial, final, in_degree_norm, out_degree_norm]

    tnode.x:
        raw structural q_t:
        [
            is_silent,
            distance_from_start_norm,
            distance_to_end_norm,
            reachable_from_start,
            can_reach_final,
        ]

    Additional metadata
    -------------------
    enode.activity_idx:
        local union-activity index used for mismatch bookkeeping.

    tnode.activity_idx:
        same local index for visible transitions; -1 for silent transitions.

    enode.variant_idx:
        local trace-variant index.

    enode.variant_pool_weight:
        normalized trace-variant frequency divided by variant length.
        Summing it over one variant recovers that variant's normalized
        case frequency. It is used ONLY by hierarchical global log readout.

    enode.event_mass:
        normalized trace-variant frequency for each event occurrence.
        Used only to compute the explicit log-only mismatch ratio.

    cnode.behavior_stats:
        16-D fixed global log-model behavioral compatibility vector:
        EF coverage/extras/distance, start/end, loops/SCC, and cheap
        concurrency/conflict proxy agreement. Used only at final fusion.

    No explicit trace-frequency weight is used on event -> transition
    cross edges. trace_freq is already part of the event-node neural input.
    """
    variant = variant.upper()
    if variant not in {"A", "C"}:
        raise ValueError("variant must be 'A' or 'C'")

    use_onehot = variant == "C"

    if use_onehot:
        if max_local_onehot_dim is None:
            raise ValueError(
                "Variant C requires max_local_onehot_dim"
            )

        act_to_slot = _build_local_activity_basis(
            spec.activity_order,
            max_local_onehot_dim=max_local_onehot_dim,
            random_permute_onehot=random_permute_onehot,
            rng=rng,
        )
    else:
        max_local_onehot_dim = 0
        act_to_slot = {}

    # ------------------------------------------------------------------
    # Local union activity index.
    # ------------------------------------------------------------------
    log_activities = {
        str(act)
        for trace, _count in spec.variants
        for act in trace
    }

    model_activities = {
        str(lbl)
        for lbl, visible in zip(
            spec.transition_labels,
            spec.transition_visible,
        )
        if visible and lbl is not None
    }

    pair_activity_order = sorted(
        log_activities | model_activities
    )

    pair_activity_to_idx = {
        act: idx
        for idx, act in enumerate(pair_activity_order)
    }

    # Explicit log-model behavioral compatibility statistics.
    # These are fixed global features; there is no behavior GNN.
    behavior_stats = _build_global_behavior_stats(
        spec,
        pair_activity_order,
    )

    # ------------------------------------------------------------------
    # Event-log nodes + trace metadata.
    # ------------------------------------------------------------------
    enode_feats: List[List[float]] = []
    enode_activity_idx: List[int] = []
    enode_variant_idx: List[int] = []
    enode_variant_pool_weight: List[float] = []
    enode_event_mass: List[float] = []

    enode_next_src: List[int] = []
    enode_next_dst: List[int] = []

    # Needed to materialize event -> matching-transition edges.
    activity_to_event_nodes: Dict[str, List[int]] = {}

    event_idx = 0

    max_trace_len = max(
        (len(v[0]) for v in spec.variants),
        default=1,
    )

    for variant_idx, (trace, count) in enumerate(spec.variants):
        trace_len = len(trace)

        tfreq = _trace_freq_feature(
            count,
            spec.total_cases,
            trace_freq_mode,
        )

        normalized_variant_frequency = (
            float(count) / float(spec.total_cases)
            if spec.total_cases > 0
            else 0.0
        )

        # Summed over the events of exactly one variant, this equals the
        # normalized case frequency of that variant.
        per_event_variant_pool_weight = (
            normalized_variant_frequency / float(trace_len)
            if trace_len > 0
            else 0.0
        )

        for pos, act in enumerate(trace):
            act = str(act)

            pos_norm = (
                0.0
                if trace_len <= 1
                else float(pos) / float(trace_len - 1)
            )

            is_first = 1.0 if pos == 0 else 0.0
            is_last = 1.0 if pos == trace_len - 1 else 0.0

            trace_len_norm = (
                float(trace_len)
                / float(max(1, max_trace_len))
            )

            base = [
                pos_norm,
                is_first,
                is_last,
                tfreq,
                trace_len_norm,
            ]

            if use_onehot:
                oh = [0.0] * max_local_onehot_dim
                if act in act_to_slot:
                    oh[act_to_slot[act]] = 1.0
                feat = base + oh
            else:
                feat = base

            enode_feats.append(feat)
            enode_activity_idx.append(
                pair_activity_to_idx[act]
            )
            enode_variant_idx.append(variant_idx)
            enode_variant_pool_weight.append(
                per_event_variant_pool_weight
            )
            enode_event_mass.append(
                normalized_variant_frequency
            )

            activity_to_event_nodes.setdefault(
                act,
                [],
            ).append(event_idx)

            if pos > 0:
                enode_next_src.append(event_idx - 1)
                enode_next_dst.append(event_idx)

            event_idx += 1

    num_enodes = len(enode_feats)
    num_pnodes = len(spec.place_initial)
    num_tnodes = len(spec.transition_labels)

    # ------------------------------------------------------------------
    # Place features remain unchanged.
    # ------------------------------------------------------------------
    pnode_feats: List[List[float]] = []

    p_in_deg = [0.0 for _ in range(num_pnodes)]
    p_out_deg = [0.0 for _ in range(num_pnodes)]

    for pidx, _tidx in spec.pt_edges:
        p_out_deg[pidx] += 1.0

    for _tidx, pidx in spec.tp_edges:
        p_in_deg[pidx] += 1.0

    max_pdeg = max(
        [1.0] + p_in_deg + p_out_deg
    )

    for i in range(num_pnodes):
        pnode_feats.append(
            [
                float(spec.place_initial[i]),
                float(spec.place_final[i]),
                p_in_deg[i] / max_pdeg,
                p_out_deg[i] / max_pdeg,
            ]
        )

    # ------------------------------------------------------------------
    # New transition structural features q_t.
    # ------------------------------------------------------------------
    raw_transition_structural = (
        _directed_transition_structural_features(spec)
    )

    tnode_feats: List[List[float]] = []
    tnode_activity_idx: List[int] = []

    # Materialized event -> transition matching edges.
    match_src: List[int] = []
    match_dst: List[int] = []

    for tidx in range(num_tnodes):
        lbl = spec.transition_labels[tidx]
        visible = spec.transition_visible[tidx]

        base = raw_transition_structural[tidx]

        if use_onehot:
            oh = [0.0] * max_local_onehot_dim
            if (
                visible
                and lbl is not None
                and str(lbl) in act_to_slot
            ):
                oh[act_to_slot[str(lbl)]] = 1.0
            feat = base + oh
        else:
            feat = base

        tnode_feats.append(feat)

        if visible and lbl is not None:
            label = str(lbl)
            activity_idx = pair_activity_to_idx[label]
            tnode_activity_idx.append(activity_idx)

            for eidx in activity_to_event_nodes.get(
                label,
                [],
            ):
                match_src.append(eidx)
                match_dst.append(tidx)
        else:
            tnode_activity_idx.append(-1)

    # ------------------------------------------------------------------
    # DGL graph.
    # ------------------------------------------------------------------
    def _tensor_pairs(pairs, idx):
        if len(pairs) == 0:
            return torch.zeros(
                0,
                dtype=torch.int64,
            )

        return torch.tensor(
            [x[idx] for x in pairs],
            dtype=torch.int64,
        )

    next_pairs = list(
        zip(
            enode_next_src,
            enode_next_dst,
        )
    )

    next_rev_pairs = list(
        zip(
            enode_next_dst,
            enode_next_src,
        )
    )

    match_pairs = list(
        zip(
            match_src,
            match_dst,
        )
    )

    # One featureless global context node for the Petri-net side.
    # IMPORTANT: it connects ONLY to transitions, never directly to places.
    tnode_ids = torch.arange(num_tnodes, dtype=torch.int64)
    context_ids = torch.zeros(num_tnodes, dtype=torch.int64)

    data_dict = {
        ("enode", "next", "enode"): (
            _tensor_pairs(next_pairs, 0),
            _tensor_pairs(next_pairs, 1),
        ),
        ("enode", "next_rev", "enode"): (
            _tensor_pairs(next_rev_pairs, 0),
            _tensor_pairs(next_rev_pairs, 1),
        ),
        ("pnode", "pt", "tnode"): (
            _tensor_pairs(spec.pt_edges, 0),
            _tensor_pairs(spec.pt_edges, 1),
        ),
        ("tnode", "tp", "pnode"): (
            _tensor_pairs(spec.tp_edges, 0),
            _tensor_pairs(spec.tp_edges, 1),
        ),
        ("enode", "match", "tnode"): (
            _tensor_pairs(match_pairs, 0),
            _tensor_pairs(match_pairs, 1),
        ),
        ("tnode", "t_to_context", "cnode"): (
            tnode_ids,
            context_ids,
        ),
        ("cnode", "context_to_t", "tnode"): (
            context_ids,
            tnode_ids,
        ),
    }

    g = dgl.heterograph(
        data_dict=data_dict,
        num_nodes_dict={
            "enode": num_enodes,
            "pnode": num_pnodes,
            "tnode": num_tnodes,
            "cnode": 1,
        },
    )

    # ------------------------------------------------------------------
    # Neural inputs.
    # ------------------------------------------------------------------
    g.nodes["enode"].data["x"] = (
        torch.tensor(
            enode_feats,
            dtype=torch.float32,
        )
        if num_enodes > 0
        else torch.zeros(
            0,
            5 + max_local_onehot_dim,
            dtype=torch.float32,
        )
    )

    g.nodes["pnode"].data["x"] = (
        torch.tensor(
            pnode_feats,
            dtype=torch.float32,
        )
        if num_pnodes > 0
        else torch.zeros(
            0,
            4,
            dtype=torch.float32,
        )
    )

    g.nodes["tnode"].data["x"] = (
        torch.tensor(
            tnode_feats,
            dtype=torch.float32,
        )
        if num_tnodes > 0
        else torch.zeros(
            0,
            5 + max_local_onehot_dim,
            dtype=torch.float32,
        )
    )

    # The cnode hidden state is still initialized to zero by the model and
    # updated only from PGNN transition messages. behavior_stats is a separate
    # fixed late-fusion feature vector and is NOT fed into context propagation.
    g.nodes["cnode"].data["behavior_stats"] = torch.tensor(
        [behavior_stats],
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Matching / pooling metadata.
    # ------------------------------------------------------------------
    g.nodes["enode"].data["activity_idx"] = torch.tensor(
        enode_activity_idx,
        dtype=torch.int64,
    )

    g.nodes["tnode"].data["activity_idx"] = torch.tensor(
        tnode_activity_idx,
        dtype=torch.int64,
    )

    g.nodes["enode"].data["variant_idx"] = torch.tensor(
        enode_variant_idx,
        dtype=torch.int64,
    )

    g.nodes["enode"].data["variant_pool_weight"] = torch.tensor(
        enode_variant_pool_weight,
        dtype=torch.float32,
    )

    g.nodes["enode"].data["event_mass"] = torch.tensor(
        enode_event_mass,
        dtype=torch.float32,
    )

    return g


class AlignmentPairDataset(torch.utils.data.Dataset):
    '''
    Dataset for batched training of the alignment surrogate.

    Variant A:
      - label-independent event/place features
      - structural transition q_t features
      - one-way event->transition match edges plus transition-only context edges
      - 16-D global behavior_stats stored on the single cnode for late fusion

    Variant C:
      - same as A, plus local one-hot identity on event nodes and visible transitions

    Notes:
      - For Variant C, max_local_onehot_dim must stay fixed across the dataset so that
        the input projection dimensions remain static.
      - If random_permute_onehot=True, graphs are rebuilt on-the-fly in __getitem__.
    '''

    def __init__(
        self,
        specs: Sequence[PairExampleSpec],
        variant: str = "A",
        max_local_onehot_dim: Optional[int] = None,
        random_permute_onehot: bool = False,
        trace_freq_mode: str = "normalized",
        prebuild_graphs: bool = True,
        seed: int = 0,
    ):
        self.specs = list(specs)
        self.variant = variant.upper()
        self.random_permute_onehot = random_permute_onehot
        self.trace_freq_mode = trace_freq_mode
        self.seed = int(seed)

        if self.variant == "C":
            if max_local_onehot_dim is None:
                max_local_onehot_dim = dataset_max_local_activity_dim(self.specs)
        else:
            max_local_onehot_dim = 0

        self.max_local_onehot_dim = int(max_local_onehot_dim)
        self._prebuilt_graphs: Optional[List[dgl.DGLHeteroGraph]] = None

        if prebuild_graphs and not self.random_permute_onehot:
            self._prebuilt_graphs = [
                build_alignment_pair_graph(
                    spec=spec,
                    variant=self.variant,
                    max_local_onehot_dim=self.max_local_onehot_dim,
                    random_permute_onehot=False,
                    trace_freq_mode=self.trace_freq_mode,
                    rng=None,
                )
                for spec in self.specs
            ]

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        spec = self.specs[idx]
        if self._prebuilt_graphs is not None:
            g = self._prebuilt_graphs[idx]
        else:
            rng = random.Random(self.seed + idx)
            g = build_alignment_pair_graph(
                spec=spec,
                variant=self.variant,
                max_local_onehot_dim=self.max_local_onehot_dim,
                random_permute_onehot=self.random_permute_onehot,
                trace_freq_mode=self.trace_freq_mode,
                rng=rng,
            )

        y = torch.tensor([spec.fitness, spec.precision], dtype=torch.float32)
        sample_weight = torch.tensor(float(spec.sample_weight), dtype=torch.float32)
        return g, y, sample_weight


def collate_alignment_pairs(batch):
    """
    Collate graph/target batches.

    Supports both old dataset items:
        (graph, target)
    and weighted dataset items:
        (graph, target, sample_weight)
    """
    if len(batch[0]) == 2:
        graphs, ys = zip(*batch)
        g_batch = dgl.batch(list(graphs))
        y_batch = torch.stack(list(ys), dim=0)
        return g_batch, y_batch

    if len(batch[0]) == 3:
        graphs, ys, sample_weights = zip(*batch)
        g_batch = dgl.batch(list(graphs))
        y_batch = torch.stack(list(ys), dim=0)
        sample_weight_batch = torch.stack(list(sample_weights), dim=0).view(-1)
        return g_batch, y_batch, sample_weight_batch

    raise ValueError(
        "Each dataset item must be either (graph, target) or "
        "(graph, target, sample_weight)."
    )
