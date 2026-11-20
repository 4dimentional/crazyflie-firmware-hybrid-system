"""Print the libclang AST of a firmware source file as an indented tree.

Usage (from the repo root, with the venv active):
    python -m ha_extract.dump_ast src/modules/src/controller/controller_pid.c
    python -m ha_extract.dump_ast <file> -f controllerPid      # one function
    python -m ha_extract.dump_ast <file> -f controllerPid -d 4 # limit depth
"""
from __future__ import annotations

import argparse
import signal
from pathlib import Path

import clang.cindex as ci

from ha_extract.ast_util import MacroIndex, binary_op, source_text, unary_op
from ha_extract.parse import parse_file


def _label(cursor: ci.Cursor, show_source: bool, macros: MacroIndex) -> str:
    parts = [cursor.kind.name]
    if cursor.spelling:
        parts.append(repr(cursor.spelling))
    op = binary_op(cursor) or unary_op(cursor)
    if op:
        parts.append(f"op '{op}'")
    if cursor.type.kind != ci.TypeKind.INVALID and cursor.kind.is_expression():
        parts.append(f"<{cursor.type.spelling}>")
    parts.append(f"L{cursor.location.line}")
    use = macros.containing(cursor)
    if use is not None:
        parts.append(f"[macro {use.name}]")
    elif show_source and (cursor.kind.is_expression() or cursor.kind.is_statement()):
        parts.append(f"| {source_text(cursor, max_len=60)}")
    return "  ".join(parts)


def dump(cursor: ci.Cursor, main_file: str, max_depth: int | None, show_source: bool,
         macros: MacroIndex, depth: int = 0) -> None:
    print("  " * depth + _label(cursor, show_source, macros))
    if max_depth is not None and depth >= max_depth:
        return
    for child in cursor.get_children():
        if child.location.file and child.location.file.name != main_file:
            continue  # skip nodes that come from included headers
        dump(child, main_file, max_depth, show_source, macros, depth + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-f", "--function", help="only dump this function definition")
    ap.add_argument("-d", "--depth", type=int, help="maximum tree depth")
    ap.add_argument("--no-source", action="store_true", help="hide the source text column")
    args = ap.parse_args()
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet exit when piped into head/less

    r = parse_file(args.file)
    for d in r.errors:
        print(f"[error] {d.location.file}:{d.location.line}: {d.spelling}")
    main_file = r.tu.spelling
    macros = MacroIndex(r.tu)

    roots = [c for c in r.tu.cursor.get_children()
             if c.location.file and c.location.file.name == main_file]
    if args.function:
        roots = [c for c in roots if c.kind == ci.CursorKind.FUNCTION_DECL
                 and c.spelling == args.function and c.is_definition()]
        if not roots:
            raise SystemExit(f"function {args.function!r} not defined in {Path(args.file).name}")
    for c in roots:
        dump(c, main_file, args.depth, not args.no_source, macros)


if __name__ == "__main__":
    main()
