from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tree_sitter_languages import get_parser as _get_ts_parser

_LOG = logging.getLogger(__name__)

Language = Literal["python", "javascript", "typescript", "go", "rust"]

EXTENSION_MAP: dict[str, Language] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}

# Node types that represent functions (context-dependent: function vs method)
_FUNCTION_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(["function_definition", "async_function_definition"]),
    "javascript": frozenset(["function_declaration", "generator_function_declaration"]),
    "typescript": frozenset(["function_declaration", "generator_function_declaration"]),
    "go": frozenset(["function_declaration"]),
    "rust": frozenset(["function_item"]),
}

# Node types that are always methods regardless of context
_METHOD_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(),
    "javascript": frozenset(["method_definition"]),
    "typescript": frozenset(["method_definition", "abstract_method_signature"]),
    "go": frozenset(["method_declaration"]),
    "rust": frozenset(),
}

# Node types that introduce a class-level context
_CLASS_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(["class_definition"]),
    "javascript": frozenset(["class_declaration"]),
    "typescript": frozenset(["class_declaration"]),
    "go": frozenset(["type_spec"]),
    "rust": frozenset(["struct_item", "enum_item", "impl_item"]),
}

# Call expression node types per language (used for reference detection)
_CALL_NODE_TYPES: dict[str, str] = {
    "python": "call",
    "javascript": "call_expression",
    "typescript": "call_expression",
    "go": "call_expression",
    "rust": "call_expression",
}


@dataclass
class ParsedNode:
    kind: Literal["function", "method", "class", "module"]
    name: str | None
    start_line: int  # 0-indexed (tree-sitter convention)
    end_line: int    # 0-indexed
    start_byte: int
    end_byte: int


@dataclass
class RefNode:
    target_name: str
    source_line: int   # 1-indexed
    line_content: str


def language_for_path(path: Path) -> Language | None:
    return EXTENSION_MAP.get(path.suffix.lower())


def parse_file(path: Path) -> tuple[Language, list[ParsedNode]] | None:
    """Parse a source file and return (language, nodes). Returns None if unsupported."""
    lang = language_for_path(path)
    if lang is None:
        return None
    try:
        source = path.read_bytes()
    except OSError as exc:
        _LOG.warning("Cannot read %s: %s", path, exc)
        return None
    return lang, parse_source(source, lang)


def parse_source(source: bytes, lang: Language) -> list[ParsedNode]:
    """Parse source bytes and return the list of structural nodes."""
    try:
        parser = _get_ts_parser(lang)
        tree = parser.parse(source)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("tree-sitter parse error (%s): %s — file skipped", lang, exc)
        return []

    nodes: list[ParsedNode] = []
    _traverse(tree.root_node, source, lang, in_class=False, results=nodes)
    return nodes


def extract_references(
    source: bytes,
    lang: Language,
    known_symbols: set[str],
    source_path: str,
) -> list[RefNode]:
    """Find call-site references to known symbols in a source file."""
    try:
        parser = _get_ts_parser(lang)
        tree = parser.parse(source)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("Reference scan parse error (%s): %s", lang, exc)
        return []

    call_type = _CALL_NODE_TYPES.get(lang)
    if not call_type:
        return []

    lines = source.decode("utf-8", errors="replace").splitlines()
    refs: list[RefNode] = []
    _find_call_refs(tree.root_node, source, call_type, known_symbols, lines, refs)
    return refs


# ---------------------------------------------------------------------------
# Internal traversal helpers
# ---------------------------------------------------------------------------


def _get_name(node: Any, source: bytes) -> str | None:
    """Extract the symbol name from a structural node."""
    for field in ("name", "type"):
        child = node.child_by_field_name(field)
        if child is not None:
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return None


def _traverse(
    node: Any,
    source: bytes,
    lang: str,
    in_class: bool,
    results: list[ParsedNode],
) -> None:
    func_types = _FUNCTION_NODE_TYPES.get(lang, frozenset())
    method_types = _METHOD_NODE_TYPES.get(lang, frozenset())
    class_types = _CLASS_NODE_TYPES.get(lang, frozenset())

    if node.type in method_types:
        name = _get_name(node, source)
        results.append(ParsedNode(
            kind="method",
            name=name,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        ))
        # Do not recurse into method bodies to keep extraction flat

    elif node.type in func_types:
        kind: Literal["function", "method", "class", "module"] = (
            "method" if in_class else "function"
        )
        name = _get_name(node, source)
        results.append(ParsedNode(
            kind=kind,
            name=name,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        ))
        # Do not recurse into function bodies

    elif node.type in class_types:
        name = _get_name(node, source)
        results.append(ParsedNode(
            kind="class",
            name=name,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        ))
        # Recurse into class body with in_class=True so nested functions → methods
        for child in node.children:
            _traverse(child, source, lang, in_class=True, results=results)

    else:
        for child in node.children:
            _traverse(child, source, lang, in_class=in_class, results=results)


def _find_call_refs(
    node: Any,
    source: bytes,
    call_type: str,
    known: set[str],
    lines: list[str],
    refs: list[RefNode],
) -> None:
    if node.type == call_type:
        func_field = node.child_by_field_name("function")
        if func_field is not None:
            ident = _leaf_identifier(func_field, source)
            if ident and ident in known:
                line_idx = func_field.start_point[0]
                content = lines[line_idx] if line_idx < len(lines) else ""
                refs.append(RefNode(
                    target_name=ident,
                    source_line=line_idx + 1,
                    line_content=content.strip(),
                ))

    for child in node.children:
        _find_call_refs(child, source, call_type, known, lines, refs)


def _leaf_identifier(node: Any, source: bytes) -> str | None:
    """Return the last identifier in a (possibly dotted) expression node."""
    identifier_types = {"identifier", "field_identifier", "type_identifier", "property_identifier"}
    if node.type in identifier_types:
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    # For attribute / member access: take the rightmost identifier child
    for child in reversed(node.children):
        if child.type in identifier_types:
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return None
