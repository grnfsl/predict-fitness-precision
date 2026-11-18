from __future__ import annotations

from typing import Dict, List, Tuple

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_RELATIONS = (
    ("enode", "next", "enode"),
    ("enode", "next_rev", "enode"),
)

EXPECTED_RELATIONS = {
    ("enode", "next", "enode"),
    ("enode", "next_rev", "enode"),
    ("pnode", "pt", "tnode"),
    ("tnode", "tp", "pnode"),
    ("enode", "match", "tnode"),
    ("tnode", "t_to_context", "cnode"),
    ("cnode", "context_to_t", "tnode"),
}

BEHAVIOR_FEATURE_DIM = 16


def _batch_sizes(
    g: dgl.DGLHeteroGraph,
    ntype: str,
) -> List[int]:
    try:
        return [
            int(x)
            for x in g.batch_num_nodes(ntype).tolist()
        ]
    except Exception:
        return [int(g.num_nodes(ntype))]


def _node_graph_ids(
    sizes: List[int],
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(sizes)

    return torch.repeat_interleave(
        torch.arange(
            batch_size,
            device=device,
            dtype=torch.long,
        ),
        torch.tensor(
            sizes,
            device=device,
            dtype=torch.long,
        ),
    )


def _mean_pool_by_batch(
    feats: torch.Tensor,
    sizes: List[int],
    hidden_dim: int,
) -> torch.Tensor:
    """
    Mean-pool node embeddings independently for each graph of a DGL batch.
    """
    batch_size = len(sizes)
    device = feats.device
    dtype = feats.dtype

    graph_ids = _node_graph_ids(
        sizes,
        device,
    )

    out = torch.zeros(
        batch_size,
        hidden_dim,
        device=device,
        dtype=dtype,
    )

    counts = torch.zeros(
        batch_size,
        1,
        device=device,
        dtype=dtype,
    )

    if feats.size(0) > 0:
        out.index_add_(
            0,
            graph_ids,
            feats,
        )

        counts.index_add_(
            0,
            graph_ids,
            torch.ones(
                feats.size(0),
                1,
                device=device,
                dtype=dtype,
            ),
        )

    return (
        out
        / counts.clamp_min(1.0)
    )


class LogGNNLayer(nn.Module):
    """
    One event-log GNN layer.

    Relations:
        next
        next_rev

    Plus the learned self transformation used by Original+self.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        h = int(hidden_dim)

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_feats=h,
                    out_feats=h,
                    norm="both",
                    weight=True,
                    bias=True,
                    allow_zero_in_degree=True,
                )
                for rel in LOG_RELATIONS
            },
            aggregate="sum",
        )

        self.self_linear = nn.Linear(
            h,
            h,
            bias=False,
        )

        self.dropout = nn.Dropout(
            float(dropout)
        )

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h_event: torch.Tensor,
    ) -> torch.Tensor:
        # The full heterograph also contains `match`, `pt`, `tp`, and context relations.
        # This Log-GNN layer must see ONLY the event-log relations.
        # The `match` relation is reserved for the cross-edge message passing module.
        log_g = dgl.edge_type_subgraph(
            g,
            list(LOG_RELATIONS),
        )

        msg = self.conv(
            log_g,
            {"enode": h_event},
        )["enode"]

        return F.relu(
            self.dropout(
                msg
                + self.self_linear(h_event)
            )
        )


class MatchingCrossEdgeMean(nn.Module):
    """
    One-way event -> transition cross-edge message passing.

    Cross edges exist only when:
        label(event) == label(visible transition)

    There is deliberately NO match_rev relation.

    Each matching event sends:
        W_match h_e

    Supported aggregation methods:

    uniform (original):
        c_t = sum(W_match h_e) / number_of_matching_events

    frequency (ablation):
        c_t = sum(w_e * W_match h_e) / sum(w_e)

    where w_e is the normalized trace-variant frequency
    stored in enode.event_mass.

    No attention score is learned and no explicit trace-frequency weighting
    is applied here. `trace_freq` is already part of the event-node input.

    Therefore, transitions having the same activity label receive the same
    event-side aggregate at a given layer. They are still distinguished by
    their different PGNN preset/postset structure and q_t.
    """

    def __init__(
        self,
        hidden_dim: int,
        match_weighting: str = "uniform",
    ):
        super().__init__()

        h = int(hidden_dim)

        # Default to the original method for backward compatibility.
        if match_weighting not in {"uniform", "frequency"}:
            raise ValueError(
                "match_weighting must be 'uniform' or 'frequency'"
            )

        self.match_weighting = match_weighting

        self.message_linear = nn.Linear(
            h,
            h,
            bias=False,
        )

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h_event: torch.Tensor,
        num_transitions: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns
        -------
        c_transition:
            Aggregated transformed event message on incoming
            match edges, [num_tnodes, H].

        match_flag:
            1 if transition has at least one matching event node,
            otherwise 0, [num_tnodes, 1].
        """

        device = h_event.device
        dtype = h_event.dtype
        hidden_dim = h_event.size(-1)

        relation_g = g[
            ("enode", "match", "tnode")
        ]

        src, dst = relation_g.edges()

        c_transition = torch.zeros(
            num_transitions,
            hidden_dim,
            device=device,
            dtype=dtype,
        )

        match_count = torch.zeros(
            num_transitions,
            1,
            device=device,
            dtype=dtype,
        )

        if src.numel() == 0:
            return (
                c_transition,
                match_count,
            )

        messages = self.message_linear(
            h_event[src]
        )

        # ------------------------------------------------------------
        # Event -> transition message aggregation.
        # ------------------------------------------------------------

        if self.match_weighting == "frequency":

            # Use the existing event_mass feature: normalized
            # trace-variant frequency for each event occurrence.
            event_mass = (
                g.nodes["enode"].data["event_mass"][src]
                .to(device=device, dtype=dtype)
                .reshape(-1, 1)
            )

            # Weight messages before aggregation so frequent
            # trace variants contribute proportionally more.
            c_transition.index_add_(
                0,
                dst,
                messages * event_mass,
            )

            # Accumulate weights separately to compute a
            # weighted MEAN rather than a weighted sum.
            match_weight_sum = torch.zeros(
                num_transitions,
                1,
                device=device,
                dtype=dtype,
            )

            match_weight_sum.index_add_(
                0,
                dst,
                event_mass,
            )

        else:

            # Original unweighted message aggregation.
            c_transition.index_add_(
                0,
                dst,
                messages,
            )

        # ------------------------------------------------------------
        # Matching counts.
        # ------------------------------------------------------------

        # Keep the original count calculation because match_flag
        # must indicate edge existence, not total frequency.
        match_count.index_add_(
            0,
            dst,
            torch.ones(
                src.numel(),
                1,
                device=device,
                dtype=dtype,
            ),
        )

        # ------------------------------------------------------------
        # Normalize aggregated messages.
        # ------------------------------------------------------------

        if self.match_weighting == "frequency":

            # Divide by total matching weight; use 1 only where
            # the weight sum is zero to avoid division by zero.
            denominator = torch.where(
                match_weight_sum > 0,
                match_weight_sum,
                torch.ones_like(match_weight_sum),
            )

            c_transition = (
                c_transition / denominator
            )

        else:

            # Preserve the original mean normalization exactly.
            c_transition = (
                c_transition
                / match_count.clamp_min(1.0)
            )

        match_flag = (
            match_count > 0
        ).to(dtype)

        return (
            c_transition,
            match_flag,
        )



class ContextConditionedPGNNLayer(nn.Module):
    """
    One synchronous PGNN layer with:
      - normal one-way event -> transition matching messages
      - one global context node connected ONLY to transitions

    All persistent layer-l states are read before the layer commits:
        h_event^l, h_place^l, h_context^l

    Transition message:

        m_t^(l+1) = Psi_l([
            preset_t^l
            || postset_t^l
            || q_t
            || c_t^l
            || context_t^l
            || match_flag_t
        ])

    where:
        c_t^l = mean_{e -> t} W_match h_e^l
        context_t^l = W_CtoT h_context^l

    The same transition messages update:
      1. output places through genuine tnode -> pnode arcs
      2. the single context node through mean transition aggregation

    Context update:

        h_context^(l+1) = ReLU(Dropout(
            mean_t(W_TtoC m_t^(l+1))
            + W_self_context h_context^l
        ))

    The context node is NOT connected directly to places and is NOT included
    directly in the final readout.

    Default transition-input dimensions:
        preset       128
        postset      128
        q_t           32
        log context  128
        PN context   128
        flag           1
        ----------------
        total        545
    """

    def __init__(
        self,
        hidden_dim: int,
        q_dim: int,
        dropout: float,
        match_weighting: str = "uniform",
    ):
        super().__init__()

        h = int(hidden_dim)
        q = int(q_dim)

        self.hidden_dim = h

        self.pre_place = nn.Linear(
            h,
            h,
            bias=False,
        )

        self.post_place = nn.Linear(
            h,
            h,
            bias=False,
        )

        self.cross_match = MatchingCrossEdgeMean(
            hidden_dim=h,
            match_weighting=match_weighting,
        )

        # Global transition-level Petri-net context.
        self.context_to_transition = nn.Linear(
            h,
            h,
            bias=False,
        )

        self.transition_to_context = nn.Linear(
            h,
            h,
            bias=False,
        )

        # Context-node self term, analogous to the learned self transformation
        # used on the event side. The context starts from zero.
        self.context_self = nn.Linear(
            h,
            h,
            bias=False,
        )

        self.transition_message = nn.Sequential(
            nn.Linear(
                4 * h + q + 1,
                h,
            ),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                h,
                h,
            ),
        )

        self.transition_to_place = nn.Linear(
            h,
            h,
            bias=False,
        )

        self.place_update = nn.Sequential(
            nn.Linear(
                2 * h,
                h,
            ),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                h,
                h,
            ),
        )

        self.dropout = nn.Dropout(
            float(dropout)
        )

    @staticmethod
    def _sum_to_index(
        values: torch.Tensor,
        index: torch.Tensor,
        out_size: int,
    ) -> torch.Tensor:
        out = torch.zeros(
            out_size,
            values.size(-1),
            device=values.device,
            dtype=values.dtype,
        )

        if index.numel() > 0:
            out.index_add_(
                0,
                index,
                values,
            )

        return out

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h_event_old: torch.Tensor,
        h_place_old: torch.Tensor,
        h_context_old: torch.Tensor,
        q_transition: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        h_event_old, h_place_old, and h_context_old are all layer-l states.
        The caller computes the Log-GNN event update independently from this
        PGNN/context update, preserving synchronous Log -> PN coupling.
        """
        num_places = h_place_old.size(0)
        num_transitions = q_transition.size(0)

        # ------------------------------------------------------------
        # Whole preset from OLD place states.
        # ------------------------------------------------------------
        pre_pidx, pre_tidx = g.edges(
            etype=("pnode", "pt", "tnode")
        )

        if pre_pidx.numel() > 0:
            pre_values = self.pre_place(
                h_place_old[pre_pidx]
            )
        else:
            pre_values = h_place_old.new_zeros(
                (0, self.hidden_dim)
            )

        preset = self._sum_to_index(
            pre_values,
            pre_tidx,
            num_transitions,
        )

        # ------------------------------------------------------------
        # Whole postset from OLD place states.
        # ------------------------------------------------------------
        post_tidx, post_pidx = g.edges(
            etype=("tnode", "tp", "pnode")
        )

        if post_pidx.numel() > 0:
            post_values = self.post_place(
                h_place_old[post_pidx]
            )
        else:
            post_values = h_place_old.new_zeros(
                (0, self.hidden_dim)
            )

        postset = self._sum_to_index(
            post_values,
            post_tidx,
            num_transitions,
        )

        # ------------------------------------------------------------
        # NORMAL one-way cross-edge message passing.
        #
        # Critically, this uses h_event_old, NOT the event states just
        # produced by the Log-GNN layer of the same depth.
        # ------------------------------------------------------------
        (
            c_transition,
            match_flag,
        ) = self.cross_match(
            g,
            h_event_old,
            num_transitions,
        )

        # ------------------------------------------------------------
        # Global context -> transition messages.
        #
        # There is exactly one cnode per graph and it connects ONLY to
        # transitions. With batched graphs, DGL keeps separate cnode ids.
        # ------------------------------------------------------------
        context_transition = torch.zeros(
            num_transitions,
            self.hidden_dim,
            device=h_place_old.device,
            dtype=h_place_old.dtype,
        )

        context_src, context_tdst = g.edges(
            etype=("cnode", "context_to_t", "tnode")
        )

        if context_src.numel() > 0:
            context_values = self.context_to_transition(
                h_context_old[context_src]
            )
            context_transition.index_add_(
                0,
                context_tdst,
                context_values,
            )

        # ------------------------------------------------------------
        # Higher-order log- and context-conditioned PGNN transition message.
        # ------------------------------------------------------------
        transition_input = torch.cat(
            [
                preset,
                postset,
                q_transition,
                c_transition,
                context_transition,
                match_flag,
            ],
            dim=-1,
        )

        m_transition = self.transition_message(
            transition_input
        )

        # ------------------------------------------------------------
        # Transition -> global context update.
        #
        # Aggregate transformed transition messages by MEAN so PN size does
        # not automatically scale the context magnitude.
        # ------------------------------------------------------------
        num_context_nodes = h_context_old.size(0)

        context_messages = torch.zeros(
            num_context_nodes,
            self.hidden_dim,
            device=h_place_old.device,
            dtype=h_place_old.dtype,
        )

        context_counts = torch.zeros(
            num_context_nodes,
            1,
            device=h_place_old.device,
            dtype=h_place_old.dtype,
        )

        context_tsrc, context_dst = g.edges(
            etype=("tnode", "t_to_context", "cnode")
        )

        if context_tsrc.numel() > 0:
            context_values = self.transition_to_context(
                m_transition[context_tsrc]
            )

            context_messages.index_add_(
                0,
                context_dst,
                context_values,
            )

            context_counts.index_add_(
                0,
                context_dst,
                torch.ones(
                    context_tsrc.numel(),
                    1,
                    device=h_place_old.device,
                    dtype=h_place_old.dtype,
                ),
            )

        context_messages = (
            context_messages
            / context_counts.clamp_min(1.0)
        )

        h_context_next = F.relu(
            self.dropout(
                context_messages
                + self.context_self(h_context_old)
            )
        )

        # ------------------------------------------------------------
        # PGNN place update through genuine transition -> place arcs.
        # ------------------------------------------------------------
        if post_tidx.numel() > 0:
            place_values = self.transition_to_place(
                m_transition[post_tidx]
            )
        else:
            place_values = h_place_old.new_zeros(
                (0, self.hidden_dim)
            )

        place_messages = self._sum_to_index(
            place_values,
            post_pidx,
            num_places,
        )

        h_place_next = self.place_update(
            torch.cat(
                [
                    h_place_old,
                    place_messages,
                ],
                dim=-1,
            )
        )

        h_place_next = F.relu(
            self.dropout(
                h_place_next
            )
        )

        return (
            h_place_next,
            h_context_next,
            m_transition,
            match_flag,
        )


class HierarchicalLogReadout(nn.Module):
    """
    Global log readout consistent with the compressed trace-variant input:

        events within one trace variant
            -> mean
        trace-variant embeddings
            -> frequency-weighted mean

    This avoids letting a rare long variant dominate merely because it has
    more event nodes.

    The variant frequency is used here intentionally at the LOG readout
    level. It is NOT reused as an explicit weight inside cross-edge message passing.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        h = int(hidden_dim)

        self.hidden_dim = h

        self.readout_proj = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(h, h),
        )

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h_event: torch.Tensor,
    ) -> torch.Tensor:
        device = h_event.device
        dtype = h_event.dtype

        e_sizes = _batch_sizes(
            g,
            "enode",
        )

        batch_size = len(e_sizes)

        e_graph_id = _node_graph_ids(
            e_sizes,
            device,
        )

        variant_idx = (
            g.nodes["enode"]
            .data["variant_idx"]
            .to(device)
            .long()
        )

        per_event_variant_weight = (
            g.nodes["enode"]
            .data["variant_pool_weight"]
            .to(device)
            .to(dtype)
        )

        # local variant_idx < number of event nodes in the graph
        stride = max(
            1,
            max(e_sizes, default=1),
        )

        slots = (
            e_graph_id * stride
            + variant_idx
        )

        num_slots = batch_size * stride

        variant_sum = torch.zeros(
            num_slots,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )

        variant_count = torch.zeros(
            num_slots,
            1,
            device=device,
            dtype=dtype,
        )

        variant_weight = torch.zeros(
            num_slots,
            1,
            device=device,
            dtype=dtype,
        )

        if h_event.size(0) > 0:
            variant_sum.index_add_(
                0,
                slots,
                h_event,
            )

            variant_count.index_add_(
                0,
                slots,
                torch.ones(
                    h_event.size(0),
                    1,
                    device=device,
                    dtype=dtype,
                ),
            )

            # Sum of (variant_frequency / trace_length) over all events in
            # that variant = normalized variant frequency.
            variant_weight.index_add_(
                0,
                slots,
                per_event_variant_weight.unsqueeze(-1),
            )

        variant_present = (
            variant_count.squeeze(-1) > 0
        )

        present_slots = torch.nonzero(
            variant_present,
            as_tuple=False,
        ).squeeze(-1)

        h_log_raw = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )

        total_weight = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )

        if present_slots.numel() > 0:
            variant_mean = (
                variant_sum[present_slots]
                / variant_count[
                    present_slots
                ].clamp_min(1.0)
            )

            weights = variant_weight[
                present_slots
            ]

            graph_id = (
                present_slots // stride
            )

            h_log_raw.index_add_(
                0,
                graph_id,
                variant_mean * weights,
            )

            total_weight.index_add_(
                0,
                graph_id,
                weights,
            )

        h_log_raw = (
            h_log_raw
            / total_weight.clamp_min(1e-8)
        )

        return self.readout_proj(
            h_log_raw
        )


class MismatchReadout(nn.Module):
    """
    Compact symmetric mismatch branches.

    LOG-ONLY
    --------
    Activity exists in the log but has no visible model transition.

    Representation:
        mean final event embeddings of those event nodes
        -> 128 -> 32
        -> multiplied by log_only_ratio

    log_only_ratio:
        frequency mass of log-only event occurrences
        / total event frequency mass

    MODEL-ONLY
    ----------
    Visible transition activity exists in the model but nowhere in the log.

    Representation:
        mean final m_t of unmatched visible transitions
        -> 128 -> 32
        -> multiplied by model_only_ratio

    model_only_ratio:
        number of unique visible model activities absent from log
        / number of unique visible model activities

    A zero ratio forces the corresponding embedding contribution to zero.
    """

    def __init__(
        self,
        hidden_dim: int,
        mismatch_dim: int,
        dropout: float,
    ):
        super().__init__()

        h = int(hidden_dim)
        m = int(mismatch_dim)

        self.hidden_dim = h
        self.mismatch_dim = m

        self.log_only_proj = nn.Sequential(
            nn.Linear(h, m),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(m, m),
        )

        self.model_only_proj = nn.Sequential(
            nn.Linear(h, m),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(m, m),
        )

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h_event: torch.Tensor,
        m_transition: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        device = h_event.device
        dtype = h_event.dtype

        e_sizes = _batch_sizes(
            g,
            "enode",
        )

        t_sizes = _batch_sizes(
            g,
            "tnode",
        )

        batch_size = len(e_sizes)

        e_graph_id = _node_graph_ids(
            e_sizes,
            device,
        )

        t_graph_id = _node_graph_ids(
            t_sizes,
            device,
        )

        e_activity = (
            g.nodes["enode"]
            .data["activity_idx"]
            .to(device)
            .long()
        )

        t_activity = (
            g.nodes["tnode"]
            .data["activity_idx"]
            .to(device)
            .long()
        )

        event_mass = (
            g.nodes["enode"]
            .data["event_mass"]
            .to(device)
            .to(dtype)
        )

        stride = max(
            1,
            max(
                (
                    e + t
                    for e, t in zip(
                        e_sizes,
                        t_sizes,
                    )
                ),
                default=1,
            ),
        )

        num_slots = batch_size * stride

        log_present = torch.zeros(
            num_slots,
            device=device,
            dtype=torch.bool,
        )

        model_present = torch.zeros(
            num_slots,
            device=device,
            dtype=torch.bool,
        )

        e_slots = (
            e_graph_id * stride
            + e_activity
        )

        if e_slots.numel() > 0:
            log_present[
                torch.unique(e_slots)
            ] = True

        t_valid = t_activity >= 0

        t_slots_valid = (
            t_graph_id[t_valid] * stride
            + t_activity[t_valid]
        )

        if t_slots_valid.numel() > 0:
            model_present[
                torch.unique(t_slots_valid)
            ] = True

        # ------------------------------------------------------------
        # Log-only branch.
        # ------------------------------------------------------------
        e_model_present = model_present[
            e_slots
        ]

        log_only_event_mask = (
            ~e_model_present
        )

        log_only_sum = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )

        log_only_count = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )

        if log_only_event_mask.any():
            gids = e_graph_id[
                log_only_event_mask
            ]

            log_only_sum.index_add_(
                0,
                gids,
                h_event[
                    log_only_event_mask
                ],
            )

            log_only_count.index_add_(
                0,
                gids,
                torch.ones(
                    gids.numel(),
                    1,
                    device=device,
                    dtype=dtype,
                ),
            )

        log_only_raw = (
            log_only_sum
            / log_only_count.clamp_min(1.0)
        )

        total_event_mass = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )

        log_only_event_mass = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )

        if event_mass.numel() > 0:
            total_event_mass.index_add_(
                0,
                e_graph_id,
                event_mass.unsqueeze(-1),
            )

        if log_only_event_mask.any():
            log_only_event_mass.index_add_(
                0,
                e_graph_id[
                    log_only_event_mask
                ],
                event_mass[
                    log_only_event_mask
                ].unsqueeze(-1),
            )

        log_only_ratio = (
            log_only_event_mass
            / total_event_mass.clamp_min(1e-8)
        ).clamp(
            0.0,
            1.0,
        )

        h_log_only = (
            self.log_only_proj(
                log_only_raw
            )
            * log_only_ratio
        )

        # ------------------------------------------------------------
        # Model-only branch.
        # ------------------------------------------------------------
        model_only_transition_mask = torch.zeros(
            t_activity.numel(),
            device=device,
            dtype=torch.bool,
        )

        if t_valid.any():
            local_slots = (
                t_graph_id[t_valid] * stride
                + t_activity[t_valid]
            )

            unmatched_visible = (
                ~log_present[
                    local_slots
                ]
            )

            valid_indices = torch.nonzero(
                t_valid,
                as_tuple=False,
            ).squeeze(-1)

            model_only_transition_mask[
                valid_indices[
                    unmatched_visible
                ]
            ] = True

        model_only_sum = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )

        model_only_count = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )

        if model_only_transition_mask.any():
            gids = t_graph_id[
                model_only_transition_mask
            ]

            model_only_sum.index_add_(
                0,
                gids,
                m_transition[
                    model_only_transition_mask
                ],
            )

            model_only_count.index_add_(
                0,
                gids,
                torch.ones(
                    gids.numel(),
                    1,
                    device=device,
                    dtype=dtype,
                ),
            )

        model_only_raw = (
            model_only_sum
            / model_only_count.clamp_min(1.0)
        )

        # Ratio based on UNIQUE visible activity labels, not transition count.
        slot_graph_id = (
            torch.arange(
                num_slots,
                device=device,
                dtype=torch.long,
            )
            // stride
        )

        unique_model_slots = torch.nonzero(
            model_present,
            as_tuple=False,
        ).squeeze(-1)

        unique_model_only_slots = torch.nonzero(
            model_present & ~log_present,
            as_tuple=False,
        ).squeeze(-1)

        model_activity_count = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )

        model_only_activity_count = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )

        if unique_model_slots.numel() > 0:
            model_activity_count.index_add_(
                0,
                slot_graph_id[
                    unique_model_slots
                ],
                torch.ones(
                    unique_model_slots.numel(),
                    1,
                    device=device,
                    dtype=dtype,
                ),
            )

        if unique_model_only_slots.numel() > 0:
            model_only_activity_count.index_add_(
                0,
                slot_graph_id[
                    unique_model_only_slots
                ],
                torch.ones(
                    unique_model_only_slots.numel(),
                    1,
                    device=device,
                    dtype=dtype,
                ),
            )

        model_only_ratio = (
            model_only_activity_count
            / model_activity_count.clamp_min(1.0)
        ).clamp(
            0.0,
            1.0,
        )

        h_model_only = (
            self.model_only_proj(
                model_only_raw
            )
            * model_only_ratio
        )

        return (
            h_log_only,
            log_only_ratio,
            h_model_only,
            model_only_ratio,
        )


class FitnessPrecisionPredictor(nn.Module):
    """
    Synchronous one-way Event-Log -> PGNN with a transition-only global
    Petri-net context node and fixed global behavioral compatibility features.

    This combines the useful pieces of the previous experiments while keeping
    the one-way Log -> PN design:
      - event Log-GNN with learned self transformation
      - one-way event -> transition matching messages
      - PGNN preset/postset transition interactions
      - one featureless global cnode connected ONLY to transitions
      - hierarchical frequency-weighted trace-variant log pooling
      - final-place-only model pooling
      - existing log-only/model-only mismatch branches
      - 16-D fixed behavioral statistics projected by a small MLP at late fusion
      - cnode hidden state excluded from final readout

    Layer l:
        h_event^(l+1) = LogGNN_l(h_event^l)

        m_t^(l+1) = Psi_l([
            preset_t^l || postset_t^l || q_t || log_context_t^l
            || pn_context_t^l || match_flag_t
        ])

        h_place^(l+1) = PGNNPlaceUpdate_l(h_place^l, m_t^(l+1))

        h_context^(l+1) = ContextUpdate_l(h_context^l, mean_t(m_t^(l+1)))

    The context node starts from zero, has a learned self transformation,
    never connects directly to places, and its hidden state is not concatenated
    into final fusion. The separate cnode.behavior_stats tensor is projected and
    concatenated only after h_log and h_model have been computed.
    """

    def __init__(
        self,
        enode_in_dim: int,
        pnode_in_dim: int,
        tnode_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        target: str = "both",
        q_dim: int = 32,
        mismatch_dim: int = 32,
        behavior_in_dim: int = BEHAVIOR_FEATURE_DIM,
        behavior_dim: int = 32,
        match_weighting: str = "uniform",
    ):
        super().__init__()

        self.target = str(target).lower()

        if self.target not in {
            "both",
            "fitness",
            "precision",
        }:
            raise ValueError(
                "target must be 'both', 'fitness', or 'precision'"
            )

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.q_dim = int(q_dim)
        self.mismatch_dim = int(mismatch_dim)
        self.behavior_in_dim = int(behavior_in_dim)
        self.behavior_dim = int(behavior_dim)
        self.match_weighting = match_weighting

        if self.behavior_in_dim != BEHAVIOR_FEATURE_DIM:
            raise ValueError(
                f"This architecture expects behavior_in_dim={BEHAVIOR_FEATURE_DIM}. "
                "Rebuild graphs with graph_pgnn_sync_context_behavior_stats.py."
            )

        if int(tnode_in_dim) != 5:
            raise ValueError(
                "This architecture expects tnode_in_dim=5: "
                "[is_silent, distance_from_start_norm, distance_to_end_norm, "
                "reachable_from_start, can_reach_final]."
            )

        self.event_input_proj = nn.Linear(
            int(enode_in_dim),
            self.hidden_dim,
        )

        self.place_input_proj = nn.Linear(
            int(pnode_in_dim),
            self.hidden_dim,
        )

        self.transition_q_proj = nn.Sequential(
            nn.Linear(
                int(tnode_in_dim),
                self.q_dim,
            ),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                self.q_dim,
                self.q_dim,
            ),
        )

        self.log_layers = nn.ModuleList(
            [
                LogGNNLayer(
                    hidden_dim=self.hidden_dim,
                    dropout=float(dropout),
                )
                for _ in range(
                    self.num_layers
                )
            ]
        )

        self.pgnn_layers = nn.ModuleList(
            [
                ContextConditionedPGNNLayer(
                    hidden_dim=self.hidden_dim,
                    q_dim=self.q_dim,
                    dropout=float(dropout),
                    match_weighting=self.match_weighting,
                )
                for _ in range(
                    self.num_layers
                )
            ]
        )

        self.log_readout = HierarchicalLogReadout(
            hidden_dim=self.hidden_dim,
            dropout=float(dropout),
        )

        self.model_readout = nn.Sequential(
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
        )

        self.behavior_proj = nn.Sequential(
            nn.Linear(
                self.behavior_in_dim,
                self.behavior_dim,
            ),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                self.behavior_dim,
                self.behavior_dim,
            ),
        )

        self.mismatch_readout = MismatchReadout(
            hidden_dim=self.hidden_dim,
            mismatch_dim=self.mismatch_dim,
            dropout=float(dropout),
        )

        fusion_in_dim = (
            2 * self.hidden_dim
            + 2 * self.mismatch_dim
            + self.behavior_dim
            + 2
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_in_dim,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
        )

        if self.target in {
            "both",
            "fitness",
        }:
            self.fitness_head = nn.Sequential(
                nn.Linear(
                    self.hidden_dim,
                    self.hidden_dim,
                ),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(
                    self.hidden_dim,
                    1,
                ),
            )

        if self.target in {
            "both",
            "precision",
        }:
            self.precision_head = nn.Sequential(
                nn.Linear(
                    self.hidden_dim,
                    self.hidden_dim,
                ),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(
                    self.hidden_dim,
                    1,
                ),
            )

    def _check_graph_schema(
        self,
        g: dgl.DGLHeteroGraph,
    ) -> None:
        actual = set(
            g.canonical_etypes
        )

        if actual != EXPECTED_RELATIONS:
            raise ValueError(
                "Graph schema does not match model_pgnn_sync_context_behavior_stats.py.\n"
                "Expected relations:\n"
                f"  {sorted(EXPECTED_RELATIONS)}\n"
                "Actual relations:\n"
                f"  {sorted(actual)}\n"
                "Rebuild the graph cache using graph_pgnn_sync_context_behavior_stats.py."
            )

        required = (
            ("enode", "activity_idx"),
            ("enode", "variant_idx"),
            ("enode", "variant_pool_weight"),
            ("enode", "event_mass"),
            ("tnode", "activity_idx"),
            ("cnode", "behavior_stats"),
        )

        for ntype, field in required:
            if field not in g.nodes[ntype].data:
                raise ValueError(
                    f"Missing {ntype}.{field}. "
                    "Rebuild graphs using graph_pgnn_sync_context_behavior_stats.py."
                )

        if g.nodes["tnode"].data["x"].shape[-1] != 5:
            raise ValueError(
                "Expected 5-D transition structural features. "
                "Rebuild graph cache using graph_pgnn_sync_context_behavior_stats.py."
            )

        if g.nodes["cnode"].data["behavior_stats"].shape[-1] != self.behavior_in_dim:
            raise ValueError(
                f"Expected {self.behavior_in_dim}-D cnode.behavior_stats. "
                "Rebuild graph cache using graph_pgnn_sync_context_behavior_stats.py."
            )

    def forward(
        self,
        g_batch: dgl.DGLHeteroGraph,
    ) -> Dict[str, torch.Tensor]:
        self._check_graph_schema(
            g_batch
        )

        h_event = F.relu(
            self.event_input_proj(
                g_batch.nodes["enode"]
                .data["x"]
            )
        )

        h_place = F.relu(
            self.place_input_proj(
                g_batch.nodes["pnode"]
                .data["x"]
            )
        )

        q_transition = self.transition_q_proj(
            g_batch.nodes["tnode"]
            .data["x"]
        )

        # One featureless context node per graph. DGL batching concatenates
        # them, so this tensor has shape [batch_size, hidden_dim].
        h_context = torch.zeros(
            g_batch.num_nodes("cnode"),
            self.hidden_dim,
            device=h_event.device,
            dtype=h_event.dtype,
        )

        # Valid even if num_layers=0.
        m_transition = torch.zeros(
            g_batch.num_nodes("tnode"),
            self.hidden_dim,
            device=h_event.device,
            dtype=h_event.dtype,
        )

        last_match_flag = torch.zeros(
            g_batch.num_nodes("tnode"),
            1,
            device=h_event.device,
            dtype=h_event.dtype,
        )

        # ------------------------------------------------------------
        # SYNCHRONOUS layerwise coupling.
        #
        # At layer l:
        #
        #   Log GNN reads h_event^l
        #   Cross edges read h_event^l
        #   PGNN reads h_place^l
        #
        # Neither branch sees the other branch's newly-created l+1 state
        # until the NEXT layer.
        # ------------------------------------------------------------
        for log_layer, pgnn_layer in zip(
            self.log_layers,
            self.pgnn_layers,
        ):
            h_event_old = h_event
            h_place_old = h_place
            h_context_old = h_context

            # Event-side update from OLD event states.
            h_event_next = log_layer(
                g_batch,
                h_event_old,
            )

            # Model/context-side update from the SAME old event/place/context
            # states. There is no PN -> event reverse path.
            (
                h_place_next,
                h_context_next,
                m_transition,
                last_match_flag,
            ) = pgnn_layer(
                g_batch,
                h_event_old,
                h_place_old,
                h_context_old,
                q_transition,
            )

            # Commit layer-l+1 persistent states together.
            h_event = h_event_next
            h_place = h_place_next
            h_context = h_context_next

        # ------------------------------------------------------------
        # Main global representations.
        # ------------------------------------------------------------
        h_log = self.log_readout(
            g_batch,
            h_event,
        )

        h_model_raw = _mean_pool_by_batch(
            h_place,
            _batch_sizes(
                g_batch,
                "pnode",
            ),
            self.hidden_dim,
        )

        h_model = self.model_readout(
            h_model_raw
        )

        # ------------------------------------------------------------
        # Fixed global behavioral compatibility features.
        #
        # There is exactly one cnode per graph, so after DGL batching this
        # tensor is [batch_size, 16]. It is NOT used in PGNN/context message
        # passing; it enters only here at late fusion.
        # ------------------------------------------------------------
        behavior_stats = (
            g_batch.nodes["cnode"]
            .data["behavior_stats"]
            .to(h_log.device)
            .to(h_log.dtype)
        )

        h_behavior = self.behavior_proj(
            behavior_stats
        )

        # ------------------------------------------------------------
        # Compact asymmetric mismatch signals.
        # ------------------------------------------------------------
        (
            h_log_only,
            log_only_ratio,
            h_model_only,
            model_only_ratio,
        ) = self.mismatch_readout(
            g_batch,
            h_event,
            m_transition,
        )

        joint = torch.cat(
            [
                h_log,
                h_model,
                h_behavior,
                h_log_only,
                log_only_ratio,
                h_model_only,
                model_only_ratio,
            ],
            dim=-1,
        )

        g_ctx = self.fusion(
            joint
        )

        out: Dict[str, torch.Tensor] = {}

        if self.target in {
            "both",
            "fitness",
        }:
            out["fitness"] = torch.sigmoid(
                self.fitness_head(
                    g_ctx
                )
            ).squeeze(-1)

        if self.target in {
            "both",
            "precision",
        }:
            out["precision"] = torch.sigmoid(
                self.precision_head(
                    g_ctx
                )
            ).squeeze(-1)

        if self.target == "both":
            out["pred"] = torch.stack(
                [
                    out["fitness"],
                    out["precision"],
                ],
                dim=-1,
            )
        elif self.target == "fitness":
            out["pred"] = (
                out["fitness"]
                .unsqueeze(-1)
            )
        else:
            out["pred"] = (
                out["precision"]
                .unsqueeze(-1)
            )

        return out