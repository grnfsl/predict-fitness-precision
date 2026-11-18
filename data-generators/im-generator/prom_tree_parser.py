"""Parse ProM EfficientTree strings into PM4Py ProcessTree objects.

Expected ProM-style examples:
    seq(a,b,c)
    xor(a,tau)
    and(seq(a,b),c)
    loop(a,b,tau)

The parser is intentionally small and strict. It is meant for the strings
written by PromIMBatch.java from EfficientTree.toString().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from pm4py.objects.process_tree.obj import Operator, ProcessTree


_OPERATOR_MAP = {
    "seq": Operator.SEQUENCE,
    "sequence": Operator.SEQUENCE,
    "xor": Operator.XOR,
    "x": Operator.XOR,
    "and": Operator.PARALLEL,
    "parallel": Operator.PARALLEL,
    "loop": Operator.LOOP,
}

# Optional operators. Some PM4Py versions may not have these enum values.
if hasattr(Operator, "OR"):
    _OPERATOR_MAP["or"] = Operator.OR
if hasattr(Operator, "INTERLEAVING"):
    _OPERATOR_MAP["int"] = Operator.INTERLEAVING
    _OPERATOR_MAP["interleaved"] = Operator.INTERLEAVING
    _OPERATOR_MAP["interleaving"] = Operator.INTERLEAVING


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int


def make_leaf(label: Optional[str], parent: Optional[ProcessTree] = None) -> ProcessTree:
    node = ProcessTree()
    node.operator = None
    node.label = label
    node.parent = parent
    node.children = []
    return node


def make_operator(
    operator: Operator,
    children: Sequence[ProcessTree],
    parent: Optional[ProcessTree] = None,
) -> ProcessTree:
    node = ProcessTree()
    node.operator = operator
    node.label = None
    node.parent = parent
    node.children = list(children)
    for child in node.children:
        child.parent = node
    return node


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0

    while i < len(text):
        char = text[i]

        if char.isspace():
            i += 1
            continue

        if char in "(),":
            tokens.append(Token(char, char, i))
            i += 1
            continue

        if char in {"'", '"'}:
            quote = char
            start = i
            i += 1
            value_chars: List[str] = []
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    value_chars.append(text[i + 1])
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                value_chars.append(text[i])
                i += 1
            else:
                raise ValueError(f"Unterminated quoted label at position {start}.")

            tokens.append(Token("NAME", "".join(value_chars), start))
            continue

        start = i
        value_chars = []
        while i < len(text) and text[i] not in "()," and not text[i].isspace():
            value_chars.append(text[i])
            i += 1

        if not value_chars:
            raise ValueError(f"Unexpected character {text[i]!r} at position {i}.")

        tokens.append(Token("NAME", "".join(value_chars), start))

    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        self.index = 0

    def current(self) -> Optional[Token]:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def peek(self, offset: int = 0) -> Optional[Token]:
        position = self.index + offset
        if position >= len(self.tokens):
            return None
        return self.tokens[position]

    def consume(self, kind: str) -> Token:
        token = self.current()
        if token is None:
            raise ValueError(f"Expected {kind}, reached end of input.")
        if token.kind != kind:
            raise ValueError(
                f"Expected {kind} at position {token.position}, got {token.value!r}."
            )
        self.index += 1
        return token

    def parse_tree(self) -> ProcessTree:
        token = self.consume("NAME")
        name = token.value
        lower_name = name.lower()

        if self.current() is not None and self.current().kind == "(":
            operator = _OPERATOR_MAP.get(lower_name)
            if operator is None:
                raise ValueError(
                    f"Unknown ProM tree operator {name!r} at position {token.position}."
                )

            self.consume("(")
            children: List[ProcessTree] = []
            if self.current() is not None and self.current().kind != ")":
                while True:
                    children.append(self.parse_tree())
                    if self.current() is None:
                        raise ValueError("Expected ',' or ')', reached end of input.")
                    if self.current().kind == ",":
                        self.consume(",")
                        continue
                    break
            self.consume(")")

            if not children:
                raise ValueError(f"Operator {name!r} has no children.")
            return make_operator(operator, children)

        if lower_name == "tau":
            return make_leaf(None)
        return make_leaf(name)


def parse_prom_tree_string(text: str) -> ProcessTree:
    """Parse one ProM EfficientTree string into a PM4Py ProcessTree."""
    if text is None or not str(text).strip():
        raise ValueError("Cannot parse an empty ProM process-tree string.")

    tokens = tokenize(str(text).strip())
    parser = _Parser(tokens)
    tree = parser.parse_tree()

    if parser.current() is not None:
        token = parser.current()
        raise ValueError(
            f"Unexpected extra token {token.value!r} at position {token.position}."
        )

    return tree


def visible_labels(root: ProcessTree) -> List[str]:
    """Return sorted visible activity labels from a PM4Py ProcessTree."""
    labels: set[str] = set()

    def visit(node: ProcessTree) -> None:
        if node.operator is None:
            if node.label is not None:
                labels.add(str(node.label))
            return
        for child in node.children:
            visit(child)

    visit(root)
    return sorted(labels)