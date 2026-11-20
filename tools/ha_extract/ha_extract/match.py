"""Run pattern sets on the functions of a C file and print the instance trees.

Usage:
    python -m ha_extract.match src/modules/src/controller/controller_mellinger.c
    python -m ha_extract.match <file> -f controllerPid
    python -m ha_extract.match <file> -f controllerPid --json out.json
    python -m ha_extract.match <file> --patterns statemachine
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import clang.cindex as ci

from ha_extract.ast_util import MacroIndex
from ha_extract.cdg import FunctionGraphs, UnsupportedConstruct, function_definitions
from ha_extract.parse import parse_file
from ha_extract.pattern import Instance, PatternSet, format_instance, match_function


def pattern_set(name: str) -> PatternSet:
    if name == "controller":
        from ha_extract.patterns_controller import CONTROLLER_PATTERNS
        return CONTROLLER_PATTERNS
    if name == "statemachine":
        from ha_extract.patterns_statemachine import STATEMACHINE_PATTERNS
        return STATEMACHINE_PATTERNS
    raise ValueError(f"unknown pattern set {name!r}")


def match_tu(tu: ci.TranslationUnit, patterns: PatternSet,
             only: str | None = None) -> dict[str, list[Instance]]:
    macros = MacroIndex(tu)
    out: dict[str, list[Instance]] = {}
    for fn in function_definitions(tu):
        if only and fn.spelling != only:
            continue
        try:
            g = FunctionGraphs(fn)
        except UnsupportedConstruct as e:
            print(f"[skip] {fn.spelling}: {e}", file=sys.stderr)
            continue
        out[fn.spelling] = match_function(g, macros, patterns)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-f", "--function")
    ap.add_argument("--patterns", default="controller", choices=["controller", "statemachine"])
    ap.add_argument("--all", action="store_true", help="also show functions with no nested instances")
    ap.add_argument("--json", help="write the instance trees as JSON to this path")
    args = ap.parse_args()
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    r = parse_file(args.file)
    for d in r.errors:
        print(f"[parse error] {d.location.file}:{d.location.line}: {d.spelling}", file=sys.stderr)
    results = match_tu(r.tu, pattern_set(args.patterns), args.function)
    if args.function and args.function not in results:
        raise SystemExit(f"function {args.function!r} not defined in {Path(args.file).name}")

    for fn, roots in results.items():
        roots = [x for x in roots if args.all or x.children]
        for root in roots:
            print(f"== {fn}")
            print("\n".join(format_instance(root)))
            print()
    if args.json:
        Path(args.json).write_text(json.dumps(
            {fn: [x.to_dict() for x in roots] for fn, roots in results.items()}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
