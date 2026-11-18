"""Process-tree model mutations for the mutation-only dataset generator.

The functions intentionally do not canonicalize the tree after mutation. The
saved process tree is meant to show how the PNML was produced.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pm4py.objects.process_tree.obj import Operator, ProcessTree


# ---------------------------------------------------------------------------
# Process-tree helpers
# ---------------------------------------------------------------------------


def clone_tree(node: Any, parent: Optional[Any] = None) -> Any:
    """Deep-copy a PM4Py process tree while restoring parent pointers."""
    copied = ProcessTree()
    copied.operator = node.operator
    copied.label = node.label
    copied.parent = parent
    copied.children = [
        clone_tree(child, copied)
        for child in getattr(node, "children", [])
    ]
    return copied


def make_leaf(label: Optional[str], parent: Optional[Any] = None) -> Any:
    node = ProcessTree()
    node.operator = None
    node.label = label
    node.parent = parent
    node.children = []
    return node


def make_operator(
    operator: Operator,
    children: Sequence[Any],
    parent: Optional[Any] = None,
) -> Any:
    node = ProcessTree()
    node.operator = operator
    node.label = None
    node.parent = parent
    node.children = list(children)
    for child in node.children:
        child.parent = node
    return node


def all_nodes(root: Any) -> List[Any]:
    nodes = [root]
    for child in getattr(root, "children", []):
        nodes.extend(all_nodes(child))
    return nodes


def visible_leaves(root: Any) -> List[Any]:
    return [
        node
        for node in all_nodes(root)
        if node.operator is None and node.label is not None
    ]


def visible_labels(root: Any) -> List[str]:
    return sorted({str(node.label) for node in visible_leaves(root)})


def operator_nodes(
    root: Any,
    operator: Operator,
    min_children: int = 0,
) -> List[Any]:
    return [
        node
        for node in all_nodes(root)
        if node.operator == operator
        and len(getattr(node, "children", [])) >= min_children
    ]


def replace_node(root: Any, old_node: Any, new_node: Any) -> Any:
    parent = old_node.parent
    if parent is None:
        new_node.parent = None
        return new_node

    index = parent.children.index(old_node)
    parent.children[index] = new_node
    new_node.parent = parent
    return root


def is_xor_with_tau_parent(node: Any) -> bool:
    parent = getattr(node, "parent", None)
    if parent is None or parent.operator != Operator.XOR:
        return False
    return any(
        child.operator is None and child.label is None
        for child in getattr(parent, "children", [])
    )


def activity_labels_in_subtree(node: Any) -> List[str]:
    labels: List[str] = []
    seen: set[str] = set()

    for leaf in visible_leaves(node):
        label = str(leaf.label)
        if label not in seen:
            labels.append(label)
            seen.add(label)

    return labels


# ---------------------------------------------------------------------------
# Eligible targets
# ---------------------------------------------------------------------------


def candidates_add_tau_skip(root: Any, parameters: Dict[str, Any]) -> List[Any]:
    """
    Visible leaves that can be made optional.

    By default this targets leaves directly inside SEQUENCE nodes. This keeps
    the mutation more local and realistic. Set within_sequence_only=false to
    allow any visible leaf that is not already under XOR(..., tau).
    """
    within_sequence_only = bool(parameters.get("within_sequence_only", True))
    candidates: List[Any] = []

    for leaf in visible_leaves(root):
        if is_xor_with_tau_parent(leaf):
            continue
        parent = getattr(leaf, "parent", None)
        if within_sequence_only and (parent is None or parent.operator != Operator.SEQUENCE):
            continue
        candidates.append(leaf)

    return candidates


def candidates_add_xor_existing_activity(root: Any, parameters: Dict[str, Any]) -> List[Any]:
    del parameters
    labels = visible_labels(root)
    if len(labels) < 2:
        return []

    # Avoid repeatedly wrapping leaves that are already inside an XOR.
    return [
        leaf
        for leaf in visible_leaves(root)
        if getattr(leaf, "parent", None) is None
        or leaf.parent.operator != Operator.XOR
    ]


def candidates_relax_sequence_to_parallel(root: Any, parameters: Dict[str, Any]) -> List[Any]:
    del parameters
    return operator_nodes(root, Operator.SEQUENCE, min_children=2)


def candidates_parallel_to_sequence(root: Any, parameters: Dict[str, Any]) -> List[Any]:
    del parameters
    return operator_nodes(root, Operator.PARALLEL, min_children=2)


def candidates_restrict_xor_choice(root: Any, parameters: Dict[str, Any]) -> List[Any]:
    del parameters
    return operator_nodes(root, Operator.XOR, min_children=2)


def candidates_swap_sequence_children(root: Any, parameters: Dict[str, Any]) -> List[Any]:
    del parameters
    return operator_nodes(root, Operator.SEQUENCE, min_children=2)


def candidates_replace_subtree_with_optional_activity_loops(
    root: Any,
    parameters: Dict[str, Any],
) -> List[Any]:
    min_activities = int(parameters.get("min_activities", 2))
    max_activities = int(parameters.get("max_activities", 6))
    allow_root = bool(parameters.get("allow_root", False))

    candidates: List[Any] = []
    for node in all_nodes(root):
        if node is root and not allow_root:
            continue

        labels = activity_labels_in_subtree(node)
        if min_activities <= len(labels) <= max_activities:
            candidates.append(node)

    return candidates


CANDIDATE_FUNCTIONS = {
    "add_tau_skip": candidates_add_tau_skip,
    "add_xor_existing_activity": candidates_add_xor_existing_activity,
    "relax_sequence_to_parallel": candidates_relax_sequence_to_parallel,
    "parallel_to_sequence": candidates_parallel_to_sequence,
    "restrict_xor_choice": candidates_restrict_xor_choice,
    "swap_sequence_children": candidates_swap_sequence_children,
    "replace_subtree_with_optional_activity_loops": (
        candidates_replace_subtree_with_optional_activity_loops
    ),
}


# ---------------------------------------------------------------------------
# Model mutations
# ---------------------------------------------------------------------------


def mutate_add_tau_skip(
    root: Any,
    parameters: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Any, bool]:
    """Replace a visible leaf a with XOR(a, tau)."""
    candidates = candidates_add_tau_skip(root, parameters)
    if not candidates:
        return root, False

    target = rng.choice(candidates)
    wrapper = make_operator(
        Operator.XOR,
        [target, make_leaf(None)],
        parent=target.parent,
    )
    root = replace_node(root, target, wrapper)
    return root, True


def mutate_add_xor_existing_activity(
    root: Any,
    parameters: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Any, bool]:
    """Replace a visible leaf a with XOR(a, b), where b already exists."""
    candidates = candidates_add_xor_existing_activity(root, parameters)
    labels = visible_labels(root)
    if not candidates or len(labels) < 2:
        return root, False

    target = rng.choice(candidates)
    alternatives = [label for label in labels if label != str(target.label)]
    if not alternatives:
        return root, False

    extra_leaf = make_leaf(rng.choice(alternatives))
    wrapper = make_operator(
        Operator.XOR,
        [target, extra_leaf],
        parent=target.parent,
    )
    root = replace_node(root, target, wrapper)
    return root, True


def mutate_relax_sequence_to_parallel(
    root: Any,
    parameters: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Any, bool]:
    """Change one SEQUENCE node into PARALLEL, allowing extra orderings."""
    candidates = candidates_relax_sequence_to_parallel(root, parameters)
    if not candidates:
        return root, False

    target = rng.choice(candidates)
    target.operator = Operator.PARALLEL
    return root, True


def mutate_parallel_to_sequence(
    root: Any,
    parameters: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Any, bool]:
    """Change one PARALLEL node into SEQUENCE, restricting orderings."""
    candidates = candidates_parallel_to_sequence(root, parameters)
    if not candidates:
        return root, False

    target = rng.choice(candidates)
    target.operator = Operator.SEQUENCE
    if bool(parameters.get("shuffle_children", True)):
        rng.shuffle(target.children)
    return root, True


def mutate_restrict_xor_choice(
    root: Any,
    parameters: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Any, bool]:
    """Replace one XOR choice by one of its children."""
    candidates = candidates_restrict_xor_choice(root, parameters)
    if not candidates:
        return root, False

    target = rng.choice(candidates)
    kept_child = rng.choice(target.children)
    root = replace_node(root, target, kept_child)
    kept_child.parent = target.parent
    return root, True


def mutate_swap_sequence_children(
    root: Any,
    parameters: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Any, bool]:
    """Swap two direct children under one SEQUENCE node."""
    candidates = candidates_swap_sequence_children(root, parameters)
    if not candidates:
        return root, False

    target = rng.choice(candidates)
    i, j = rng.sample(range(len(target.children)), 2)
    target.children[i], target.children[j] = target.children[j], target.children[i]
    return root, True


def make_optional_activity_loop(label: str) -> Any:
    """Build XOR(LOOP(label, tau), tau)."""
    loop = make_operator(
        Operator.LOOP,
        [make_leaf(label), make_leaf(None)],
    )
    return make_operator(
        Operator.XOR,
        [loop, make_leaf(None)],
    )


def mutate_replace_subtree_with_optional_activity_loops(
    root: Any,
    parameters: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Any, bool]:
    """
    Replace one subtree by a permissive local flower-like block.

    Example:
        SEQ(a, b, c)
    can become:
        PARALLEL(XOR(LOOP(a, tau), tau),
                 XOR(LOOP(b, tau), tau),
                 XOR(LOOP(c, tau), tau))
    """
    candidates = candidates_replace_subtree_with_optional_activity_loops(
        root,
        parameters,
    )
    if not candidates:
        return root, False

    target = rng.choice(candidates)
    labels = activity_labels_in_subtree(target)
    if len(labels) < int(parameters.get("min_activities", 2)):
        return root, False

    rng.shuffle(labels)
    children = [make_optional_activity_loop(label) for label in labels]
    replacement = make_operator(
        Operator.PARALLEL,
        children,
        parent=target.parent,
    )
    root = replace_node(root, target, replacement)
    return root, True


MUTATION_FUNCTIONS = {
    "add_tau_skip": mutate_add_tau_skip,
    "add_xor_existing_activity": mutate_add_xor_existing_activity,
    "relax_sequence_to_parallel": mutate_relax_sequence_to_parallel,
    "parallel_to_sequence": mutate_parallel_to_sequence,
    "restrict_xor_choice": mutate_restrict_xor_choice,
    "swap_sequence_children": mutate_swap_sequence_children,
    "replace_subtree_with_optional_activity_loops": (
        mutate_replace_subtree_with_optional_activity_loops
    ),
}


def apply_one_model_mutation(
    root: Any,
    mutation_name: str,
    parameters: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Any, bool]:
    if mutation_name not in MUTATION_FUNCTIONS:
        raise ValueError(f"Unknown model mutation: {mutation_name}")

    return MUTATION_FUNCTIONS[mutation_name](root, parameters, rng)


def count_eligible_targets(
    root: Any,
    mutation_name: str,
    parameters: Dict[str, Any],
) -> int:
    if mutation_name not in CANDIDATE_FUNCTIONS:
        raise ValueError(f"Unknown model mutation: {mutation_name}")
    return len(CANDIDATE_FUNCTIONS[mutation_name](root, parameters))


def resolve_num_applications(
    root: Any,
    mutation_name: str,
    parameters: Dict[str, Any],
) -> int:
    """
    Resolve mutation strength.

    Backward compatibility:
        num_applications is still accepted if present.

    New preferred style:
        application_fraction × eligible targets, clamped by
        min_applications and max_applications.
    """
    eligible = count_eligible_targets(root, mutation_name, parameters)
    if eligible <= 0:
        return 0

    if "num_applications" in parameters:
        return max(0, int(parameters["num_applications"]))

    fraction = float(parameters.get("application_fraction", 1.0))
    minimum = int(parameters.get("min_applications", 0))
    maximum = int(parameters.get("max_applications", eligible))

    count = round(fraction * eligible)
    count = max(minimum, count)
    count = min(maximum, count)
    count = min(eligible, count)
    return max(0, count)


def apply_model_mutations(
    base_tree: Any,
    mutations_config: Dict[str, Dict[str, Any]],
    rng: random.Random,
    mutation_order: Optional[Sequence[str]] = None,
    shuffle_mutations: bool = False,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """Apply model mutations to a copy of base_tree and return tree/order/params."""
    mutated_tree = clone_tree(base_tree)

    if not mutations_config:
        return mutated_tree, [], {}

    order = list(mutation_order) if mutation_order is not None else list(mutations_config)
    if shuffle_mutations:
        rng.shuffle(order)

    applied_order: List[str] = []
    applied_parameters: Dict[str, Any] = {}

    for operation in order:
        if operation not in mutations_config:
            raise ValueError(f"Missing config for model mutation: {operation}")

        parameters = dict(mutations_config[operation])
        if parameters.get("enabled", True) is False:
            continue

        probability = float(parameters.get("probability", 1.0))
        num_applications = resolve_num_applications(
            mutated_tree,
            operation,
            parameters,
        )
        applied_parameters[operation] = {
            **parameters,
            "resolved_num_applications": num_applications,
        }

        for _ in range(num_applications):
            if rng.random() > probability:
                continue

            mutated_tree, changed = apply_one_model_mutation(
                mutated_tree,
                operation,
                parameters,
                rng,
            )
            if changed:
                applied_order.append(operation)
            else:
                break

    return mutated_tree, applied_order, applied_parameters