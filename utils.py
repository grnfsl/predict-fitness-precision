from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta, timezone
from itertools import count
from xml.sax.saxutils import escape
from pathlib import Path

import json
from pathlib import Path

def generate_xes(
    sequences,
    *,
    write_file: bool = False,
    out_path: str | None = None,
    return_df: bool = True,
    log_name: str | None = None,
    base_time: datetime | None = None,
    overwrite: bool = False,
):
    """
    Generate a log from (sequence, frequency) pairs.

    - If write_file=True: writes XES to disk
    - If return_df=True: returns a Pandas DataFrame like pm4py.read_xes()
    - Both can be enabled together

    Returns:
        pandas.DataFrame if return_df=True, else None
    """
    if base_time is None:
        base_time = datetime.now(timezone.utc)

    if log_name is None:
        log_name = base_time.strftime("log_%Y%m%d_%H%M%S_%f")

    if write_file and out_path is None:
        raise ValueError("out_path must be provided when write_file=True")

    rows = [] if return_df else None

    trace_id = count(1)

    # Optional file setup
    f = None
    if write_file:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File exists: {out_path}")
        f = path.open("w", encoding="utf-8", newline="\n")
        w = f.write

        # XES header
        w('<?xml version="1.0" encoding="UTF-8" ?>\n')
        w('<log xes.version="1.0" xes.features="nested-attributes"\n')
        w('     xmlns="http://www.xes-standard.org/">\n')
        w('  <extension name="Concept" prefix="concept" uri="http://www.xes-standard.org/concept.xesext"/>\n')
        w('  <extension name="Time" prefix="time" uri="http://www.xes-standard.org/time.xesext"/>\n')
        w('  <classifier name="Activity" keys="concept:name"/>\n')
        w('  <global scope="trace">\n')
        w('    <string key="concept:name" value=""/>\n')
        w('  </global>\n')
        w('  <global scope="event">\n')
        w('    <string key="concept:name" value=""/>\n')
        w('    <date key="time:timestamp" value="1970-01-01T00:00:00.000+00:00"/>\n')
        w('  </global>\n')
        w(f'  <string key="concept:name" value="{escape(log_name)}"/>\n')

    try:
        for activities, freq in sequences:
            if freq <= 0:
                continue

            for _ in range(freq):
                tid = next(trace_id)
                case_id = f"trace_{tid}"
                t0 = base_time + timedelta(seconds=tid)

                if write_file:
                    w("  <trace>\n")
                    w(f'    <string key="concept:name" value="{case_id}"/>\n')

                for i, act in enumerate(activities):
                    ts = t0 + timedelta(seconds=i)

                    # Build DataFrame row
                    if return_df:
                        rows.append({
                            "case:concept:name": case_id,
                            "concept:name": act,
                            "time:timestamp": ts
                        })

                    # Write XES event
                    if write_file:
                        w("    <event>\n")
                        w(f'      <string key="concept:name" value="{escape(act)}"/>\n')
                        w(f'      <date key="time:timestamp" value="{ts.isoformat(timespec="milliseconds")}"/>\n')
                        w("    </event>\n")

                if write_file:
                    w("  </trace>\n")

        if write_file:
            w("</log>\n")

    finally:
        if f is not None:
            f.close()

    if return_df:
        df = pd.DataFrame(rows)
        return df

    return None


# Excludes models
def save_history(hist, path):
    path = Path(path)

    hist_to_save = {k: v for k, v in hist.items() if k != "models"}

    with open(path, "w") as f:
        json.dump(hist_to_save, f, indent=2)


def load_history(path):
    path = Path(path)

    with open(path, "r") as f:
        hist = json.load(f)

    return hist


import matplotlib.pyplot as plt
import networkx as nx

from graph import (
    prepare_pair_spec,
    build_alignment_pair_graph,
)

def plot_alignment_pair_graph(
    log_df,
    net,
    im,
    fm,
    figsize=(16, 9),
    show_reverse=False,
):
    """
    Plot the complete event-log + Petri-net + context-node graph
    used by the context-node experiment.

    Parameters
    ----------
    log_df:
        Event log as pandas DataFrame.

    net, im, fm:
        PM4Py Petri net, initial marking, and final marking.

    figsize:
        Matplotlib figure size.

    show_reverse:
        If True, also display the explicit reverse relations used
        by the GNN. Usually False is much easier to read.
    """

    # ============================================================
    # 1. Build the actual graph used by the model
    # ============================================================

    spec = prepare_pair_spec(
        log_df=log_df,
        net=net,
        im=im,
        fm=fm,
        fitness=0.0,
        precision=0.0,
        compute_targets_if_missing=False,
    )

    g = build_alignment_pair_graph(
        spec,
        variant="A",
    )

    # ============================================================
    # 2. Build readable node labels + positions
    # ============================================================

    G = nx.MultiDiGraph()

    labels = {}
    positions = {}

    enodes = []
    pnodes = []
    tnodes = []
    cnodes = []

    # ============================================================
    # EVENT LOG -- TOP
    # ============================================================

    event_idx = 0

    num_variants = len(spec.variants)

    event_y_start = 6.0
    variant_spacing = 1.5
    event_x_spacing = 1.8

    for variant_idx, (trace, count) in enumerate(spec.variants):

        y = event_y_start - variant_idx * variant_spacing

        for pos_idx, activity in enumerate(trace):
            node = ("enode", event_idx)

            G.add_node(node)

            labels[node] = (
                f"{activity}\n"
                f"v{variant_idx + 1}, n={count}"
            )

            positions[node] = (
                pos_idx * event_x_spacing,
                y,
            )

            enodes.append(node)

            event_idx += 1

    # ============================================================
    # PETRI NET -- BOTTOM
    # ============================================================

    place_objects = list(net.places)
    transition_objects = list(net.transitions)

    pn_y = 0.0
    pn_x_spacing = 2.0

    # Put places and transitions approximately left-to-right.
    # This simple placement is mainly intended for small examples.

    for i, place in enumerate(place_objects):
        node = ("pnode", i)

        G.add_node(node)

        labels[node] = str(place.name)

        positions[node] = (
            i * pn_x_spacing,
            pn_y,
        )

        pnodes.append(node)

    for i, transition in enumerate(transition_objects):

        node = ("tnode", i)

        G.add_node(node)

        if transition.label is None:
            transition_name = "τ"
        else:
            transition_name = str(transition.label)

        labels[node] = transition_name

        positions[node] = (
            i * pn_x_spacing + 1.0,
            pn_y - 1.3,
        )

        tnodes.append(node)

    # ============================================================
    # CONTEXT NODE -- RIGHT SIDE
    # ============================================================

    all_x = [
        x
        for x, y in positions.values()
    ]

    max_x = max(all_x) if all_x else 0.0

    cnode = ("cnode", 0)

    G.add_node(cnode)

    labels[cnode] = "CONTEXT"

    positions[cnode] = (
        max_x + 4.0,
        pn_y - 0.5,
    )

    cnodes.append(cnode)

    # ============================================================
    # 3. Copy relations from the actual DGL graph
    # ============================================================

    relation_edges = {}

    for canonical_etype in g.canonical_etypes:

        src_type, relation, dst_type = canonical_etype

        # Hide explicit reverse copies unless requested.
        if not show_reverse and relation in {
            "next_rev",
            "pt_rev",
            "tp_rev",
            "match_rev",
            "context_to_p",
            "context_to_t",
        }:
            continue

        src, dst = g.edges(etype=canonical_etype)

        edges = []

        for s, d in zip(src.tolist(), dst.tolist()):

            u = (src_type, s)
            v = (dst_type, d)

            G.add_edge(
                u,
                v,
                relation=relation,
            )

            edges.append((u, v))

        relation_edges[relation] = edges

    # ============================================================
    # 4. Plot nodes
    # ============================================================

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=enodes,
        node_shape="o",
        node_size=1500,
        alpha=0.9,
        label="Event",
    )

    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=pnodes,
        node_shape="o",
        node_size=1500,
        alpha=0.9,
        label="Place",
    )

    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=tnodes,
        node_shape="s",
        node_size=1400,
        alpha=0.9,
        label="Transition",
    )

    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=cnodes,
        node_shape="D",
        node_size=2200,
        alpha=0.9,
        label="Context",
    )

    # ============================================================
    # 5. Plot different edge types
    # ============================================================

    # Event-log sequence
    if "next" in relation_edges:
        nx.draw_networkx_edges(
            G,
            positions,
            edgelist=relation_edges["next"],
            arrows=True,
            arrowsize=18,
            width=1.8,
        )

    # Petri-net arcs
    for relation in ["pt", "tp"]:
        if relation in relation_edges:
            nx.draw_networkx_edges(
                G,
                positions,
                edgelist=relation_edges[relation],
                arrows=True,
                arrowsize=18,
                width=1.8,
            )

    # Event -> transition matching
    if "match" in relation_edges:
        nx.draw_networkx_edges(
            G,
            positions,
            edgelist=relation_edges["match"],
            arrows=True,
            arrowsize=13,
            width=1.0,
            style="dashed",
            alpha=0.45,
            connectionstyle="arc3,rad=0.08",
        )

    # PN -> context
    context_edges = []

    for relation in [
        "p_to_context",
        "t_to_context",
    ]:
        context_edges.extend(
            relation_edges.get(relation, [])
        )

    if context_edges:
        nx.draw_networkx_edges(
            G,
            positions,
            edgelist=context_edges,
            arrows=True,
            arrowsize=15,
            width=1.3,
            style="dotted",
            alpha=0.65,
        )

    # Optional explicit reverse edges
    if show_reverse:

        reverse_edges = []

        for relation in [
            "next_rev",
            "pt_rev",
            "tp_rev",
            "match_rev",
            "context_to_p",
            "context_to_t",
        ]:
            reverse_edges.extend(
                relation_edges.get(relation, [])
            )

        nx.draw_networkx_edges(
            G,
            positions,
            edgelist=reverse_edges,
            arrows=True,
            arrowsize=10,
            width=0.7,
            style="dashed",
            alpha=0.25,
            connectionstyle="arc3,rad=-0.08",
        )

    # ============================================================
    # 6. Labels
    # ============================================================

    nx.draw_networkx_labels(
        G,
        positions,
        labels=labels,
        font_size=9,
    )

    plt.title(
        "Event Log + Petri Net + Global Context Node"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.show()

    return g