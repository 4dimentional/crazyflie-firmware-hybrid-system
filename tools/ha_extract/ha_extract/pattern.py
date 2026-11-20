"""Pattern language and matcher over control dependence graphs.

A pattern is a code template with metavariables (holes). Matching overlays
the pattern on a function's CDG top-down; at each CDG node the pattern's
expression template is compared with the node's AST depth-first, binding
metavariables. A reference to another pattern creates a child *pattern
instance*, so a successful match yields a pattern-instance tree.

Metavariables have
  - a kind: Var (variables), Const (enum constants / literals), Field
    (struct member names), Expr (any expression), and
  - a scope: 'global' bindings are shared by the whole instance tree of one
    @main match (the same value everywhere), 'local' bindings are fresh for
    every pattern instance.

Patterns are written as Python objects rather than parsed from C-like
templates; see patterns_controller.py and patterns_statemachine.py.

Matching is a backtracking search written with generators: every `match`
yields each state (bindings + partial instance tree) under which the pattern
fits. Regions commit to the first successful alternative per statement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Iterator, Sequence

import clang.cindex as ci

from ha_extract.ast_util import MacroIndex, binary_op, children, source_text, strip, unary_op
from ha_extract.cdg import FunctionGraphs, Label, Node, F, T

K = ci.CursorKind


# ---------------------------------------------------------------- results
@dataclass(frozen=True)
class Bound:
    """A metavariable's value. `key` decides equality; `text` is for display."""
    key: tuple
    text: str
    cursor: ci.Cursor | None = field(default=None, compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class Instance:
    pattern: str
    bindings: dict                       # local metavariables of this instance
    nodes: tuple                         # ((tag, Node), ...) pattern<->CDG correspondences
    children: tuple                      # child Instances
    slot: str | None = None              # where the parent placed it (cond / then / else / ...)
    globals: dict | None = None          # only on @main roots

    @property
    def line(self) -> int:
        for _, n in self.nodes:
            if n.line:
                return n.line
        return 0

    def walk(self) -> Iterator["Instance"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, pattern: str) -> list["Instance"]:
        return [i for i in self.walk() if i.pattern == pattern]

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "slot": self.slot,
            "line": self.line,
            "bindings": {k: v.text for k, v in self.bindings.items()},
            **({"globals": {k: v.text for k, v in self.globals.items()}} if self.globals else {}),
            "nodes": [{"tag": t, "line": n.line, "kind": n.kind} for t, n in self.nodes],
            "children": [c.to_dict() for c in self.children],
        }


@dataclass(frozen=True)
class St:
    """Matching state threaded through the search (persistent / copy-on-write)."""
    globals: dict
    locals: dict
    children: tuple = ()
    nodes: tuple = ()

    def bind(self, name: str, scope: str, value: Bound) -> "St | None":
        table = self.globals if scope == "global" else self.locals
        old = table.get(name)
        if old is not None:
            return self if old.key == value.key else None
        if scope == "global":
            return replace(self, globals={**self.globals, name: value})
        return replace(self, locals={**self.locals, name: value})

    def add_node(self, tag: str, node: Node) -> "St":
        return replace(self, nodes=self.nodes + ((tag, node),))

    def tag_children_from(self, start: int, slot: str) -> "St":
        if len(self.children) == start:
            return self
        tagged = tuple(c if c.slot else replace(c, slot=slot) for c in self.children[start:])
        return replace(self, children=self.children[:start] + tagged)


@dataclass
class Ctx:
    graphs: FunctionGraphs
    macros: MacroIndex
    registry: "PatternSet"


class Pat:
    def match(self, target, st: St, cx: Ctx) -> Iterator[St]:
        raise NotImplementedError


# ---------------------------------------------------------------- metavariables
Where = Callable[[ci.Cursor], bool] | None


class _Meta(Pat):
    def __init__(self, name: str, scope: str = "local", where: Where = None):
        assert scope in ("global", "local")
        self.name, self.scope, self.where = name, scope, where

    def value(self, c: ci.Cursor, cx: Ctx) -> Bound | None:
        raise NotImplementedError

    def match(self, target, st, cx):
        if not isinstance(target, ci.Cursor):
            return
        c = strip(target)
        if self.where is not None and not self.where(c):
            return
        v = self.value(c, cx)
        if v is not None:
            s = st.bind(self.name, self.scope, v)
            if s is not None:
                yield s


class Var(_Meta):
    """A variable or parameter reference."""
    def value(self, c, cx):
        if c.kind != K.DECL_REF_EXPR or c.referenced is None:
            return None
        ref = c.referenced
        if ref.kind not in (K.VAR_DECL, K.PARM_DECL):
            return None
        return Bound(("var", ref.get_usr() or ref.spelling), c.spelling, c)


class Const(_Meta):
    """An enum constant or a numeric literal (possibly negated)."""
    def value(self, c, cx):
        if c.kind == K.DECL_REF_EXPR and c.referenced is not None and c.referenced.kind == K.ENUM_CONSTANT_DECL:
            return Bound(("enum", c.spelling), c.spelling, c)
        if c.kind in (K.INTEGER_LITERAL, K.FLOATING_LITERAL, K.CHARACTER_LITERAL):
            text = source_text(c, cx.macros)
            return Bound(("lit", text), text, c)
        if unary_op(c) == "-":
            inner = self.value(strip(children(c)[0]), cx)
            if inner is not None and inner.key[0] == "lit":
                return Bound(("lit", "-" + inner.text), "-" + inner.text, c)
        return None


class Expr(_Meta):
    """Any expression, compared by its source text."""
    def value(self, c, cx):
        if not c.kind.is_expression():
            return None
        text = source_text(c, cx.macros)
        return Bound(("expr", text), text, c)


class Field:
    """Metavariable for a struct member name (used inside Member)."""
    def __init__(self, name: str, scope: str = "local"):
        self.name, self.scope = name, scope


# ---------------------------------------------------------------- expression templates
class Any_(Pat):
    def match(self, target, st, cx):
        yield st


class Or(Pat):
    def __init__(self, *alts: Pat):
        self.alts = alts

    def match(self, target, st, cx):
        for a in self.alts:
            yield from a.match(target, st, cx)


class Member(Pat):
    """`base->field` or `base.field`; field is a literal name or a Field metavariable."""
    def __init__(self, base: Pat, fld: "str | Field"):
        self.base, self.fld = base, fld

    def match(self, target, st, cx):
        c = strip(target)
        if c.kind != K.MEMBER_REF_EXPR:
            return
        kids = children(c)
        if not kids:
            return
        if isinstance(self.fld, Field):
            st = st.bind(self.fld.name, self.fld.scope, Bound(("field", c.spelling), c.spelling, c))
            if st is None:
                return
        elif c.spelling != self.fld:
            return
        yield from self.base.match(kids[0], st, cx)


class BinOp(Pat):
    def __init__(self, ops: "str | Sequence[str]", lhs: Pat, rhs: Pat, *, commutative=False,
                 op_var: str | None = None):
        self.ops = {ops} if isinstance(ops, str) else set(ops)
        self.lhs, self.rhs, self.commutative, self.op_var = lhs, rhs, commutative, op_var

    def match(self, target, st, cx):
        c = strip(target)
        op = binary_op(c)
        if op not in self.ops:
            return
        if self.op_var:
            st = st.bind(self.op_var, "local", Bound(("op", op), op))
            if st is None:
                return
        a, b = children(c)
        orders = [(a, b), (b, a)] if self.commutative else [(a, b)]
        for x, y in orders:
            for s1 in self.lhs.match(x, st, cx):
                yield from self.rhs.match(y, s1, cx)


class UnOp(Pat):
    def __init__(self, ops: "str | Sequence[str]", operand: Pat):
        self.ops = {ops} if isinstance(ops, str) else set(ops)
        self.operand = operand

    def match(self, target, st, cx):
        c = strip(target)
        if unary_op(c) in self.ops:
            yield from self.operand.match(children(c)[0], st, cx)


class Call(Pat):
    """Function call whose callee name matches a regex (fullmatch); optional callee metavariable."""
    def __init__(self, callee: str = r".*", callee_var: str | None = None, scope: str = "local"):
        self.rx, self.callee_var, self.scope = re.compile(callee), callee_var, scope

    def match(self, target, st, cx):
        c = strip(target)
        if c.kind != K.CALL_EXPR or not self.rx.fullmatch(c.spelling or ""):
            return
        if self.callee_var:
            st = st.bind(self.callee_var, self.scope, Bound(("fn", c.spelling), c.spelling, c))
            if st is None:
                return
        yield st


class Macro(Pat):
    """An expression produced by expanding a macro whose name matches `name`.

    `args` names metavariables (local) bound to the macro's argument texts.
    """
    def __init__(self, name: str, args: Sequence[str] = ()):
        self.rx, self.args = re.compile(name), tuple(args)

    def match(self, target, st, cx):
        use = cx.macros.containing(strip(target))
        if use is None or not self.rx.fullmatch(use.name):
            return
        for var, text in zip(self.args, use.args):
            st = st.bind(var, "local", Bound(("macroarg", text), text))
            if st is None:
                return
        st = st.bind("macro", "local", Bound(("macro", use.name), use.name))
        if st is not None:
            yield st


# ---------------------------------------------------------------- statement / CDG templates
class If(Pat):
    """An `if` predicate node; `then`/`else_` are region patterns over its T/F CDG children."""
    def __init__(self, cond: Pat, then: Pat | None = None, else_: Pat | None = None):
        self.cond, self.then, self.else_ = cond, then, else_

    def match(self, node, st, cx):
        if not isinstance(node, Node) or node.kind != "if":
            return
        st = st.add_node("if", node)
        n0 = len(st.children)
        for s1 in self.cond.match(node.cond, st, cx):
            s1 = s1.tag_children_from(n0, "cond")
            for s2 in _region(self.then, cx.graphs.region(node, T), s1, cx):
                yield from _region(self.else_, cx.graphs.region(node, F), s2, cx)


class Loop(Pat):
    """A while / for / do-while predicate; `body` is a region pattern over its T children."""
    def __init__(self, cond: Pat | None = None, body: Pat | None = None):
        self.cond, self.body = cond or Any_(), body

    def match(self, node, st, cx):
        if not isinstance(node, Node) or node.kind not in ("while", "for", "do"):
            return
        st = st.add_node(node.kind, node)
        target = node.cond if node.cond is not None else None
        for s1 in (self.cond.match(target, st, cx) if target is not None else [st]):
            yield from _region(self.body, cx.graphs.region(node, T), s1, cx)


@dataclass
class CaseTarget:
    switch: Node
    label: Label
    nodes: list


class Switch(Pat):
    """A switch node; `case` is matched against each case label group (cases that do
    not fit are skipped, at least one must fit)."""
    def __init__(self, cond: Pat, case: Pat):
        self.cond, self.case = cond, case

    def match(self, node, st, cx):
        if not isinstance(node, Node) or node.kind != "switch":
            return
        st = st.add_node("switch", node)
        for s1 in self.cond.match(node.cond, st, cx):
            matched = 0
            for label, nodes in cx.graphs.groups(node).items():
                m = next(self.case.match(CaseTarget(node, label, nodes), s1, cx), None)
                if m is not None:
                    s1, matched = m, matched + 1
            if matched:
                yield s1


class Case(Pat):
    """One case group of a Switch: `label` is matched against the case expression
    (a `default:` group only fits Any_()), `body` against its CDG children."""
    def __init__(self, label: Pat, body: Pat | None = None):
        self.label, self.body = label, body

    def match(self, target, st, cx):
        if not isinstance(target, CaseTarget):
            return
        if target.label.kind == "case":
            labels = self.label.match(target.label.cursor, st, cx)
        elif isinstance(self.label, Any_):
            labels = [st]
        else:
            return
        for s1 in labels:
            yield from _region(self.body, target.nodes, s1, cx)


class Stmt(Pat):
    """A simple statement whose expression matches `expr`."""
    def __init__(self, expr: Pat):
        self.expr = expr

    def match(self, node, st, cx):
        if not isinstance(node, Node) or node.kind not in ("stmt", "return"):
            return
        c = node.cursor
        if node.kind == "return":
            kids = children(c)
            if not kids:
                return
            c = kids[0]
        if not c.kind.is_expression():
            return
        yield from self.expr.match(c, st.add_node("stmt", node), cx)


def Assign(lhs: Pat, rhs: Pat) -> Stmt:
    return Stmt(BinOp("=", lhs, rhs))


# ---------------------------------------------------------------- regions (lists of CDG siblings)
class Region(Pat):
    """The statements under one CDG label.

    nested:   alternatives tried, in order, on every statement; the first one
              that fits is kept (statements matching none stay plain body).
    require:  patterns that must each fit at least one statement.
    coverage: 'full' fails if some statement fits no nested alternative;
              'partial' (default) allows unmatched statements.
    """
    def __init__(self, name: str = "body", nested: Sequence[Pat] = (), require: Sequence[Pat] = (),
                 coverage: str = "partial"):
        assert coverage in ("partial", "full")
        self.name, self.nested, self.require, self.coverage = name, tuple(nested), tuple(require), coverage

    def match(self, nodes, st, cx):
        if not isinstance(nodes, list):
            return
        for s1, used in self._require(list(self.require), nodes, st, frozenset(), cx):
            ok = True
            for n in nodes:
                s1 = s1.add_node(self.name, n)
                if n.id in used:
                    continue
                for alt in self.nested:
                    n0 = len(s1.children)
                    m = next(alt.match(n, s1, cx), None)
                    if m is not None:
                        s1 = m.tag_children_from(n0, self.name)
                        break
                else:
                    if self.coverage == "full":
                        ok = False
                        break
            if ok:
                yield s1

    def _require(self, reqs, nodes, st, used, cx):
        """Backtrack over which statement satisfies each required pattern."""
        if not reqs:
            yield st, used
            return
        first, rest = reqs[0], reqs[1:]
        for n in nodes:
            if n.id in used:
                continue
            n0 = len(st.children)
            for s1 in first.match(n, st, cx):
                yield from self._require(rest, nodes, s1.tag_children_from(n0, self.name),
                                         used | {n.id}, cx)


class Only(Pat):
    """A region consisting of exactly one statement that matches `pat`."""
    def __init__(self, pat: Pat, name: str = "body"):
        self.pat, self.name = pat, name

    def match(self, nodes, st, cx):
        if isinstance(nodes, list) and len(nodes) == 1:
            n0 = len(st.children)
            for s in self.pat.match(nodes[0], st, cx):
                yield s.tag_children_from(n0, self.name)


def _region(pat: Pat | None, nodes: list, st: St, cx: Ctx) -> Iterator[St]:
    if pat is None:
        yield st
    else:
        yield from pat.match(nodes, st, cx)


# ---------------------------------------------------------------- pattern references and sets
class Ref(Pat):
    """Use of another named pattern: matching it creates a child instance."""
    def __init__(self, name: str):
        self.name = name

    def match(self, target, st, cx):
        pdef = cx.registry[self.name]
        inner = St(st.globals, {})
        for s in pdef.body.match(target, inner, cx):
            inst = Instance(pdef.name, s.locals, s.nodes, s.children)
            yield St(s.globals, st.locals, st.children + (inst,), st.nodes)


@dataclass
class PatternDef:
    """A named pattern. Annotations (only meaningful on @main patterns):
    main   - tried on every function definition;
    level  - CDG depth at which matching may start (0 = the function's
             top-level statements, None = any region);
    coverage - 'partialtoplevel' lets the top-level region contain statements
               the pattern does not account for, 'full' requires all of them.
    """
    name: str
    body: Pat
    main: bool = False
    level: int | None = None
    coverage: str = "partialtoplevel"


class PatternSet(dict):
    def __init__(self, *defs: PatternDef):
        super().__init__((d.name, d) for d in defs)
        for d in defs:
            if d.main and isinstance(d.body, Region):
                d.body.coverage = "partial" if d.coverage == "partialtoplevel" else "full"

    @property
    def mains(self) -> list[PatternDef]:
        return [d for d in self.values() if d.main]


def match_function(g: FunctionGraphs, macros: MacroIndex, patterns: PatternSet) -> list[Instance]:
    """Run every @main pattern on one function; return the root instances.

    A region-shaped @main pattern (Region / Only) is matched once per start
    region; a statement-shaped one (If, Switch, ...) is tried at every
    statement of the start regions, giving one root per match, like the
    paper's "try the @main pattern at every position".
    """
    cx = Ctx(g, macros, patterns)
    roots = []
    for d in patterns.mains:
        if d.level == 0:
            starts = [g.top_level()]
        else:
            starts = [g.region(n, l) for n in g.cdg for l in g.groups(n)]
        region_shaped = isinstance(d.body, (Region, Only))
        for nodes in starts:
            if not region_shaped and d.coverage == "full" and len(nodes) != 1:
                continue
            for target in ([nodes] if region_shaped else nodes):
                st0 = St({"fn": Bound(("fn", g.name), g.name)}, {})
                m = next(d.body.match(target, st0, cx), None)
                if m is not None:
                    roots.append(Instance(d.name, m.locals, m.nodes, m.children, globals=m.globals))
    return roots


# ---------------------------------------------------------------- printing
def format_instance(inst: Instance, macros: MacroIndex | None = None, indent: int = 0) -> list[str]:
    pad = "  " * indent
    head = f"{pad}{inst.pattern}"
    if inst.slot:
        head += f"  <{inst.slot}>"
    if inst.line:
        head += f"  L{inst.line}"
    binds = {**(inst.globals or {}), **inst.bindings}
    if binds:
        head += "  {" + ", ".join(f"{k}: {v.text}" for k, v in binds.items()) + "}"
    lines = [head]
    for c in inst.children:
        lines += format_instance(c, macros, indent + 1)
    return lines
