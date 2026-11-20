"""Control flow and control dependence graphs for one C function.

The CFG has one node per simple statement and one per predicate
(if / switch / while / for / do-while). The CDG is computed from the CFG with
the classic post-dominator construction (Ferrante, Ottenstein & Warren 1987),
so it reflects actual control flow: statements after `if (c) return;` are
control dependent on that `if` (label F), not on the function entry, and a
statement reachable through switch fall-through depends on several case labels.

Usage:
    python -m ha_extract.cdg <file.c> -f <function> [--dot out.dot]
"""
from __future__ import annotations

import argparse
import signal
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import clang.cindex as ci
import networkx as nx

from ha_extract.ast_util import MacroIndex, children, source_text

K = ci.CursorKind


class UnsupportedConstruct(Exception):
    pass


@dataclass(frozen=True)
class Label:
    """Label of a control dependence edge."""
    kind: str                      # 'entry' | 'T' | 'F' | 'case' | 'default'
    text: str = ""                 # case value text
    cursor: ci.Cursor | None = field(default=None, compare=False, hash=False)

    def __str__(self) -> str:
        return f"case {self.text}" if self.kind == "case" else self.kind


ENTRY_L, T, F, DEFAULT = Label("entry"), Label("T"), Label("F"), Label("default")


@dataclass(eq=False)
class Node:
    id: int
    kind: str                      # entry | exit | stmt | return | if | switch | while | for | do
    cursor: ci.Cursor | None = None
    cond: ci.Cursor | None = None  # condition expression of predicates

    PREDICATES = ("if", "switch", "while", "for", "do")

    @property
    def is_predicate(self) -> bool:
        return self.kind in self.PREDICATES

    @property
    def line(self) -> int:
        return self.cursor.location.line if self.cursor is not None else 0

    @property
    def offset(self) -> int:
        return self.cursor.extent.start.offset if self.cursor is not None else -1

    def __repr__(self) -> str:
        return f"<{self.kind} L{self.line}>" if self.cursor is not None else f"<{self.kind}>"


@dataclass
class _Ctx:
    brk: Node | None = None
    cont: Node | None = None
    cases: list | None = None      # [(Label, entry Node)] of the innermost switch


class FunctionGraphs:
    def __init__(self, fn: ci.Cursor):
        if fn.kind != K.FUNCTION_DECL or not fn.is_definition():
            raise ValueError(f"{fn.spelling} is not a function definition")
        self.fn = fn
        self.name = fn.spelling
        self.cfg = nx.MultiDiGraph()
        self._next_id = 0
        self.entry = self._new("entry")
        self.exit = self._new("exit")
        body = next(c for c in fn.get_children() if c.kind == K.COMPOUND_STMT)
        first = self._stmt(body, self.exit, _Ctx())
        self._edge(self.entry, first, ENTRY_L)
        self._connect_infinite_loops()
        self.ipdom = self._postdominators()
        self.cdg = self._control_dependence()
        self._groups_cache: dict[int, dict[Label, list[Node]]] = {}

    # ---- CFG construction (backwards: each statement is wired to its successor)
    def _new(self, kind: str, cursor=None, cond=None) -> Node:
        n = Node(self._next_id, kind, cursor, cond)
        self._next_id += 1
        self.cfg.add_node(n)
        return n

    def _edge(self, a: Node, b: Node, label: Label | None = None) -> None:
        self.cfg.add_edge(a, b, label=label)

    def _stmt(self, c: ci.Cursor, nxt: Node, ctx: _Ctx) -> Node:
        """Add statement `c` to the CFG; return the node control enters it at."""
        k = c.kind
        kids = children(c)
        if k == K.COMPOUND_STMT:
            for s in reversed(kids):
                nxt = self._stmt(s, nxt, ctx)
            return nxt
        if k == K.IF_STMT:
            n = self._new("if", c, kids[0])
            self._edge(n, self._stmt(kids[1], nxt, ctx), T)
            self._edge(n, self._stmt(kids[2], nxt, ctx) if len(kids) > 2 else nxt, F)
            return n
        if k == K.WHILE_STMT:
            n = self._new("while", c, kids[0])
            self._edge(n, self._stmt(kids[1], n, _Ctx(nxt, n, ctx.cases)), T)
            self._edge(n, nxt, F)
            return n
        if k == K.DO_STMT:
            n = self._new("do", c, kids[1])
            body = self._stmt(kids[0], n, _Ctx(nxt, n, ctx.cases))
            self._edge(n, body, T)
            self._edge(n, nxt, F)
            return body
        if k == K.FOR_STMT:
            return self._for(c, kids, nxt, ctx)
        if k == K.SWITCH_STMT:
            n = self._new("switch", c, kids[0])
            inner = _Ctx(nxt, ctx.cont, [])
            self._stmt(kids[1], nxt, inner)
            for label, target in inner.cases:
                self._edge(n, target, label)
            if not any(l.kind == "default" for l, _ in inner.cases):
                self._edge(n, nxt, DEFAULT)
            return n
        if k == K.CASE_STMT:
            target = self._stmt(kids[-1], nxt, ctx)
            if ctx.cases is None:
                raise UnsupportedConstruct(f"case outside switch at line {c.location.line}")
            ctx.cases.append((Label("case", " ".join(t.spelling for t in kids[0].get_tokens()), kids[0]), target))
            return target
        if k == K.DEFAULT_STMT:
            target = self._stmt(kids[-1], nxt, ctx)
            ctx.cases.append((DEFAULT, target))
            return target
        if k == K.BREAK_STMT:
            return ctx.brk
        if k == K.CONTINUE_STMT:
            return ctx.cont
        if k == K.RETURN_STMT:
            n = self._new("return", c)
            self._edge(n, self.exit)
            return n
        if k == K.NULL_STMT:
            return nxt
        if k == K.LABEL_STMT:
            return self._stmt(kids[-1], nxt, ctx)
        if k in (K.GOTO_STMT, K.INDIRECT_GOTO_STMT):
            raise UnsupportedConstruct(f"goto at line {c.location.line}")
        n = self._new("stmt", c)
        self._edge(n, nxt)
        return n

    def _for(self, c: ci.Cursor, kids: list, nxt: Node, ctx: _Ctx) -> Node:
        # libclang omits absent for-clauses from the children, so find which
        # of init/cond/inc exist from the tokens between the parentheses.
        has_init, has_cond, has_inc = _for_clauses(c)
        it = iter(kids)
        init = next(it) if has_init else None
        cond = next(it) if has_cond else None
        inc = next(it) if has_inc else None
        body = next(it)
        n = self._new("for", c, cond)
        back = n
        if inc is not None:
            back = self._new("stmt", inc)
            self._edge(back, n)
        self._edge(n, self._stmt(body, back, _Ctx(nxt, back, ctx.cases)), T)
        self._edge(n, nxt, F)
        if init is not None:
            first = self._stmt(init, n, ctx) if init.kind == K.DECL_STMT else self._new("stmt", init)
            if init.kind != K.DECL_STMT:
                self._edge(first, n)
            return first
        return n

    def _connect_infinite_loops(self) -> None:
        """Give loops that never reach exit (e.g. task `while (1)`) a pseudo exit edge."""
        while True:
            reaches = nx.ancestors(self.cfg, self.exit) | {self.exit}
            stuck = [n for n in self.cfg if n not in reaches and n.kind in ("while", "for", "do")]
            if not stuck:
                return
            self._edge(min(stuck, key=lambda n: n.id), self.exit, Label("F"))

    # ---- post-dominators and control dependence
    def _postdominators(self) -> dict[Node, Node]:
        rev = nx.DiGraph()
        rev.add_nodes_from(self.cfg)
        rev.add_edges_from((b, a) for a, b in self.cfg.edges())
        rev.add_edge(self.exit, self.entry)        # augmented ENTRY -> EXIT edge
        idom = nx.immediate_dominators(rev, self.exit)
        idom.pop(self.exit, None)
        return idom

    def _postdominates(self, b: Node, a: Node) -> bool:
        x = a
        while x in self.ipdom:
            x = self.ipdom[x]
            if x is b:
                return True
        return False

    def _control_dependence(self) -> nx.MultiDiGraph:
        cdg = nx.MultiDiGraph()
        cdg.add_nodes_from(n for n in self.cfg if n is not self.exit)
        seen = set()
        # For each CFG edge a->b where b does not post-dominate a, every node on
        # the post-dominator tree path from b up to (excluding) ipdom(a) is
        # control dependent on a with that edge's label.
        for a, b, label in self.cfg.edges(data="label"):
            if a not in self.ipdom or self._postdominates(b, a):
                continue
            stop = self.ipdom[a]
            x = b
            while x is not stop and x is not self.exit:
                key = (a.id, x.id, label)
                if key not in seen:
                    seen.add(key)
                    cdg.add_edge(a, x, label=label)
                x = self.ipdom[x]
        live = nx.descendants(self.cfg, self.entry) | {self.entry}
        cdg.remove_nodes_from([n for n in list(cdg) if n not in live])   # dead code
        return cdg

    # ---- queries used by the matcher and printers
    def groups(self, node: Node) -> dict[Label, list[Node]]:
        """CDG children of `node` grouped by edge label, in source order."""
        g = self._groups_cache.get(node.id)
        if g is None:
            tmp: dict[Label, list[Node]] = defaultdict(list)
            if node.is_predicate:            # keep labels with no dependents (e.g. `case X: break;`)
                for _, _, label in self.cfg.out_edges(node, data="label"):
                    tmp[label]
            for _, child, label in self.cdg.out_edges(node, data="label"):
                if child is not node:                       # loop self-dependence
                    tmp[label].append(child)
            g = {l: sorted(ns, key=lambda n: n.offset) for l, ns in tmp.items()}
            self._groups_cache[node.id] = g
        return g

    def region(self, node: Node, label: Label) -> list[Node]:
        return self.groups(node).get(label, [])

    def top_level(self) -> list[Node]:
        return self.region(self.entry, ENTRY_L)

    def dependences(self, node: Node) -> list[tuple[Node, Label]]:
        return [(p, l) for p, _, l in self.cdg.in_edges(node, data="label")]


def _for_clauses(c: ci.Cursor) -> tuple[bool, bool, bool]:
    toks = [t.spelling for t in c.get_tokens()]
    i = toks.index("(") + 1
    depth, parts, cur = 0, [], 0
    for t in toks[i:]:
        if t == "(":
            depth += 1
        elif t == ")":
            if depth == 0:
                parts.append(cur)
                break
            depth -= 1
        if t == ";" and depth == 0:
            parts.append(cur); cur = 0
        elif t != ")" or depth:
            cur += 1
    parts += [0] * (3 - len(parts))
    return parts[0] > 0, parts[1] > 0, parts[2] > 0


def function_definitions(tu: ci.TranslationUnit) -> list[ci.Cursor]:
    main = tu.spelling
    return [c for c in tu.cursor.get_children()
            if c.kind == K.FUNCTION_DECL and c.is_definition()
            and c.location.file and c.location.file.name == main]


# ---- printing ------------------------------------------------------------
def describe(n: Node, macros: MacroIndex | None = None, width: int = 70) -> str:
    if n.kind == "entry":
        return "ENTRY"
    shown = n.cond if n.is_predicate and n.cond is not None else n.cursor
    return f"L{n.line:<4} {n.kind:<6} {source_text(shown, macros, width)}"


def print_cdg(g: FunctionGraphs, macros: MacroIndex | None = None) -> None:
    printed: set[int] = set()

    def walk(n: Node, depth: int, label: Label | None) -> None:
        prefix = "  " * depth + (f"[{label}] " if label is not None and label.kind != "entry" else "")
        again = n.id in printed
        print(prefix + describe(n, macros) + ("   (see above)" if again else ""))
        if again:
            return
        printed.add(n.id)
        for lab, kids in g.groups(n).items():
            for k in kids:
                walk(k, depth + 1, lab)

    walk(g.entry, 0, None)


def to_dot(g: FunctionGraphs, macros: MacroIndex | None = None) -> str:
    lines = [f'digraph "{g.name}" {{', "  node [shape=box, fontname=monospace];"]
    for n in g.cdg:
        shape = "diamond" if n.is_predicate else ("oval" if n.kind == "entry" else "box")
        text = describe(n, macros, 50).replace('"', r'\"')
        lines.append(f'  n{n.id} [label="{text}", shape={shape}];')
    for a, b, l in g.cdg.edges(data="label"):
        lab = "" if l is None or l.kind == "entry" else str(l)
        lines.append(f'  n{a.id} -> n{b.id} [label="{lab}"];')
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    from ha_extract.parse import parse_file

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-f", "--function", required=True)
    ap.add_argument("--dot", help="also write the CDG as Graphviz DOT to this path")
    args = ap.parse_args()
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    r = parse_file(args.file)
    fns = [c for c in function_definitions(r.tu) if c.spelling == args.function]
    if not fns:
        raise SystemExit(f"function {args.function!r} not defined in {Path(args.file).name}")
    macros = MacroIndex(r.tu)
    g = FunctionGraphs(fns[0])
    print_cdg(g, macros)
    if args.dot:
        Path(args.dot).write_text(to_dot(g, macros))
        print(f"\nwrote {args.dot}")


if __name__ == "__main__":
    main()
