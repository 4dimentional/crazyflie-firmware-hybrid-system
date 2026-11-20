"""Export every intermediate result of one function as Graphviz DOT + SVG.

    python -m ha_extract.export src/modules/src/controller/controller_mellinger.c -f controllerMellinger

writes to tools/ha_extract/out/<file>/<function>/ (or -o DIR):

    1_ast.dot / .svg          AST of the function (implicit casts and parentheses
                              collapsed; --raw keeps them)
    2_cfg.dot / .svg          control flow graph (T/F/case edge labels; straight-line
                              statements merged into basic blocks, --full to split)
    3_cdg.dot / .svg          control dependence graph, predicates coloured by the
                              pattern that matched them (statement runs with the same
                              dependence merged into one block, --full to split)
    4_instances.dot / .svg    pattern instance tree with metavariable bindings
    4_instances.json / .txt   the same tree as JSON and as indented text
    index.html                all four SVGs on one page

Open index.html in a browser (or the SVGs in VS Code) and zoom in.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

import clang.cindex as ci

from ha_extract.ast_util import MacroIndex, binary_op, source_text, strip, unary_op
from ha_extract.cdg import FunctionGraphs, describe, function_definitions
from ha_extract.match import pattern_set
from ha_extract.parse import TOOL_ROOT, parse_file
from ha_extract.pattern import Instance, format_instance, match_function

K = ci.CursorKind

PATTERN_COLORS = {
    "PModeBranch": "#cfe2ff", "PContGuard": "#ffe5b4", "PTiming": "#d9f2d9",
    "PReset": "#f8d7da", "PResetCall": "#f8d7da", "PLoop": "#e2d9f3", "POtherIf": "#eeeeee",
    "PStateCheck": "#cfe2ff", "PGuard": "#ffe5b4", "PTransition": "#d9f2d9",
}


def q(s: str) -> str:
    """Quote a DOT label (\\n stays a line break)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\\\\n", "\\n") + '"'


def _header(name: str, rankdir: str = "TB") -> list[str]:
    return [f"digraph {q(name)} {{", f"  rankdir={rankdir};",
            '  node [shape=box, fontname="monospace", fontsize=10, style=filled, fillcolor=white];',
            '  edge [fontname="monospace", fontsize=9];']


# ---------------------------------------------------------------- AST
def ast_dot(fn: ci.Cursor, macros: MacroIndex, raw: bool = False) -> str:
    lines = _header(f"AST {fn.spelling}", rankdir="LR")
    counter = [0]
    main = fn.translation_unit.spelling

    def visit(c: ci.Cursor) -> str:
        if not raw:
            c = strip(c)
        nid = f"a{counter[0]}"; counter[0] += 1
        label = [c.kind.name]
        if c.spelling:
            label.append(repr(c.spelling))
        op = binary_op(c) or unary_op(c)
        if op:
            label.append(f"op {op}")
        label.append(f"L{c.location.line}")
        use = macros.containing(c)
        if use is not None:
            label.append(f"[macro {use.name}]")
        elif c.kind.is_expression() and not list(c.get_children()):
            label.append(source_text(c, max_len=30))
        color = ("#d9f2d9" if c.kind.is_statement() else
                 "#cfe2ff" if c.kind.is_expression() else "#eeeeee")
        if use is not None:
            color = "#f0f0f0"
        text = "\\n".join(label)
        lines.append(f"  {nid} [label={q(text)}, fillcolor={q(color)}];")
        for k in c.get_children():
            if k.location.file and k.location.file.name != main:
                continue
            lines.append(f"  {nid} -> {visit(k)};")
        return nid

    visit(fn)
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- CFG / CDG
def _node_style(n, macros, claimed: dict) -> str:
    shape = {"entry": "oval", "exit": "oval", "return": "box"}.get(n.kind, "diamond" if n.is_predicate else "box")
    text = "ENTRY" if n.kind == "entry" else "EXIT" if n.kind == "exit" else describe(n, macros, 45)
    pat = claimed.get(n.id)
    if pat:
        text += f"\\n«{pat}»"
    color = PATTERN_COLORS.get(pat, "white")
    return f"label={q(text)}, shape={shape}, fillcolor={q(color)}"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _block_style(nodes, macros) -> str:
    """A box listing several statements, one left-aligned line each."""
    rows = [f"L{n.line:<4} {source_text(n.cursor, macros, 60)}" for n in nodes]
    if len(rows) > 12:
        rows = rows[:5] + [f"... {len(rows) - 10} more ..."] + rows[-5:]
    label = '"' + "".join(_esc(r) + "\\l" for r in rows) + '"'
    return f"label={label}, shape=box, fillcolor=\"#fafafa\""


def _edge_label(l) -> str:
    return "" if l is None or l.kind == "entry" else str(l)


def _edge_color(lab: str) -> str:
    return {"T": "#2e7d32", "F": "#c62828"}.get(lab, "black")


def _is_plain(n) -> bool:
    return n.kind in ("stmt", "return")


def cfg_dot(g: FunctionGraphs, macros: MacroIndex, full: bool = False) -> str:
    """CFG; by default straight-line statement runs are merged into basic blocks."""
    cfg = g.cfg

    def leader(n) -> bool:
        if not _is_plain(n) or cfg.in_degree(n) != 1:
            return True
        p = next(iter(cfg.predecessors(n)))
        return not _is_plain(p) or cfg.out_degree(p) != 1

    blocks: dict[int, list] = {}
    for n in sorted(cfg, key=lambda n: (n.kind != "entry", n.kind == "exit", n.offset)):
        if not full and not leader(n):
            continue
        chain = [n]
        while not full and _is_plain(chain[-1]) and cfg.out_degree(chain[-1]) == 1:
            nxt = next(iter(cfg.successors(chain[-1])))
            if leader(nxt):
                break
            chain.append(nxt)
        blocks[n.id] = chain

    lines = _header(f"CFG {g.name}") + ["  ordering=out;"]
    for first_id, chain in blocks.items():
        n = chain[0]
        style = _block_style(chain, macros) if _is_plain(n) else _node_style(n, macros, {})
        lines.append(f"  n{first_id} [{style}];")
    for first_id, chain in blocks.items():
        last = chain[-1]
        for _, b, l in sorted(cfg.out_edges(last, data="label"), key=lambda e: e[1].offset):
            lab = _edge_label(l)
            back = b.kind in ("while", "for", "do") and 0 <= b.offset < last.offset
            extra = ", style=dashed, constraint=false" if back else ""
            lines.append(f"  n{last.id if full else first_id} -> n{b.id} "
                         f"[label={q(lab)}, color={q(_edge_color(lab))}, fontcolor={q(_edge_color(lab))}{extra}];")
    lines.append("}")
    return "\n".join(lines)


def cdg_dot(g: FunctionGraphs, macros: MacroIndex, claimed: dict, full: bool = False) -> str:
    """CDG; by default consecutive plain statements with the same single
    dependence are drawn as one block (like the paper's [B1], [B2] nodes)."""
    rep: dict[int, str] = {}                    # CDG node id -> DOT node id
    decls: list[tuple[int, str]] = []           # (offset, declaration)
    for parent in g.cdg:
        for label, kids in g.groups(parent).items():
            run: list = []
            for k in kids + [None]:
                groupable = (k is not None and not full and _is_plain(k) and k.id not in claimed
                             and len(g.dependences(k)) == 1)
                if groupable:
                    run.append(k)
                    continue
                if run:
                    bid = f"b{run[0].id}"
                    for r in run:
                        rep[r.id] = bid
                    decls.append((run[0].offset, f"  {bid} [{_block_style(run, macros)}];"))
                    run = []
    for n in g.cdg:
        if n.id not in rep:
            rep[n.id] = f"n{n.id}"
            decls.append((n.offset, f"  n{n.id} [{_node_style(n, macros, claimed)}];"))

    lines = _header(f"CDG {g.name}", rankdir="LR") + ["  ordering=out;"]
    lines += [d for _, d in sorted(decls, key=lambda x: x[0])]
    seen = set()
    for a, b, l in sorted(g.cdg.edges(data="label"), key=lambda e: (e[0].offset, e[1].offset)):
        key = (rep[a.id], rep[b.id], _edge_label(l))
        if key in seen:
            continue
        seen.add(key)
        lab = key[2]
        lines.append(f"  {key[0]} -> {key[1]} [label={q(lab)}, color={q(_edge_color(lab))}, "
                     f"fontcolor={q(_edge_color(lab))}];")
    lines.append("}")
    return "\n".join(lines)


def claimed_nodes(roots: list[Instance]) -> dict[int, str]:
    """CDG node id -> name of the innermost pattern whose predicate/statement it is."""
    out: dict[int, str] = {}
    for r in roots:
        for inst in r.walk():
            for tag, n in inst.nodes:
                if tag in ("if", "switch", "while", "for", "do", "stmt"):
                    out[n.id] = inst.pattern
    return out


# ---------------------------------------------------------------- instance tree
def instances_dot(fn_name: str, roots: list[Instance]) -> str:
    lines = _header(f"instances {fn_name}", rankdir="LR")
    counter = [0]

    def visit(inst: Instance) -> str:
        nid = f"i{counter[0]}"; counter[0] += 1
        head = inst.pattern + (f"  <{inst.slot}>" if inst.slot else "") + (f"  L{inst.line}" if inst.line else "")
        binds = {**(inst.globals or {}), **inst.bindings}
        body = [f"{k}: {v.text}" for k, v in binds.items()]
        label = "\\n".join([head] + body)
        lines.append(f"  {nid} [label={q(label)}, fillcolor={q(PATTERN_COLORS.get(inst.pattern, 'white'))}];")
        for c in inst.children:
            lines.append(f"  {nid} -> {visit(c)} [label={q(c.slot or '')}];")
        return nid

    for r in roots:
        visit(r)
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- driver
def render(dot_path: Path) -> Path | None:
    if shutil.which("dot") is None:
        return None
    svg = dot_path.with_suffix(".svg")
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg)], check=True)
    return svg


def export_function(tu: ci.TranslationUnit, fn: ci.Cursor, out: Path, patterns: str = "controller",
                    raw_ast: bool = False, full: bool = False) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    macros = MacroIndex(tu)
    g = FunctionGraphs(fn)
    roots = match_function(g, macros, pattern_set(patterns))

    files = {
        "1_ast.dot": ast_dot(fn, macros, raw_ast),
        "2_cfg.dot": cfg_dot(g, macros, full),
        "3_cdg.dot": cdg_dot(g, macros, claimed_nodes(roots), full),
        "4_instances.dot": instances_dot(fn.spelling, roots),
    }
    svgs = []
    for name, text in files.items():
        p = out / name
        p.write_text(text)
        svgs.append(render(p))
    (out / "4_instances.json").write_text(json.dumps([r.to_dict() for r in roots], indent=2))
    (out / "4_instances.txt").write_text("\n\n".join("\n".join(format_instance(r)) for r in roots) + "\n")

    titles = ["1. AST", "2. CFG", "3. CDG (predicates coloured by matched pattern)", "4. Pattern instance tree"]
    sections = "".join(
        f'<h2>{t}</h2><p><a href="{s.name}">open {s.name} alone</a></p>'
        f'<div class="g"><img src="{s.name}"></div>' if s else f"<h2>{t}</h2><p>(install graphviz to render)</p>"
        for t, s in zip(titles, svgs))
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(fn.spelling)}</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fff}} .g{{overflow:auto;border:1px solid #ccc;max-height:85vh}}
img{{max-width:none}}</style>
<h1>{html.escape(fn.spelling)} <small>({html.escape(Path(tu.spelling).name)})</small></h1>
<p>Instance tree as text: <a href="4_instances.txt">4_instances.txt</a> · JSON: <a href="4_instances.json">4_instances.json</a></p>
{sections}
""")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-f", "--function", action="append", help="function(s) to export (default: all)")
    ap.add_argument("-o", "--out", help="output directory (default tools/ha_extract/out/<file>/)")
    ap.add_argument("--patterns", default="controller", choices=["controller", "statemachine"])
    ap.add_argument("--raw", action="store_true", help="keep implicit casts / parentheses in the AST")
    ap.add_argument("--full", action="store_true",
                    help="one CFG/CDG node per statement instead of merged statement blocks")
    args = ap.parse_args()

    r = parse_file(args.file)
    base = Path(args.out) if args.out else TOOL_ROOT / "out" / Path(args.file).stem
    fns = [c for c in function_definitions(r.tu) if not args.function or c.spelling in args.function]
    if not fns:
        raise SystemExit("no matching function definitions")
    for fn in fns:
        d = export_function(r.tu, fn, base / fn.spelling, args.patterns, args.raw, args.full)
        print(f"{fn.spelling}: {d / 'index.html'}")


if __name__ == "__main__":
    main()
