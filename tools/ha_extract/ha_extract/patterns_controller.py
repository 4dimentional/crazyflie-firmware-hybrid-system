"""Patterns for mode-dependent computation in Crazyflie controllers.

Instead of the paper's "state variable + transition" templates, controllers
select a control law by the setpoint's mode fields:

  @main @level 0 @coverage partialtoplevel
  PController:   { ...statements, each possibly one of BODY... }

  PModeBranch:   if (PModePred) { BODY } else { PModeBranch | BODY }   (else-if chains)
  PModePred:     ⟦sp⟧->mode.⟦field⟧ ==/!= ⟦modeConst⟧                 sp global; field, modeConst local
               | PModePred && PModePred | PModePred || PModePred | !PModePred
  PContGuard:    if (PContPred) { BODY } else { BODY }                 guard over continuous values
  PContPred:     ⟦lhs⟧ <op> ⟦rhs⟧  (both numeric, non-enum)  | && | ||
  PTiming:       if (RATE_DO_EXECUTE(⟦rate⟧, ⟦tick⟧)) { BODY }       sub-rate execution, also the
               | if (!RATE_DO_EXECUTE(...)) { ... } else { BODY }    early-return form (body under F)
  PReset:        if (⟦g⟧) { ... ⟦…Reset/Reinit…⟧(...) ... }           reset jump on an event
  PResetCall:    ⟦…Reset/Reinit…⟧(...);
  PLoop / POtherIf: anything else with a body, so nested patterns are still found.

BODY is the ordered alternative list below: each statement of a body is tried
against these in turn and the first that fits becomes a child instance.
"""
from __future__ import annotations

from ha_extract.ast_util import is_numeric
from ha_extract.pattern import (
    Any_, BinOp, Call, Const, Expr, Field, If, Loop, Macro, Member, Only, Or, PatternDef,
    PatternSet, Ref, Region, Stmt, UnOp, Var,
)

RESET_RX = r".*(Reset|Reinit|reset|reinit).*"
CMP_OPS = ("<", ">", "<=", ">=", "==", "!=")

BODY = [
    Ref("PTiming"),
    Ref("PModeBranch"),
    Ref("PContGuard"),
    Ref("PReset"),
    Ref("PLoop"),
    Ref("POtherIf"),
    Ref("PResetCall"),
]


def body(name: str) -> Region:
    return Region(name, nested=BODY)


_rate = Macro(r"RATE_DO_EXECUTE(_HZ)?", args=("rate", "tick"))

CONTROLLER_PATTERNS = PatternSet(
    PatternDef("PController", Region("top", nested=BODY), main=True, level=0,
               coverage="partialtoplevel"),

    PatternDef("PTiming", Or(
        If(_rate, then=body("then")),
        If(UnOp("!", _rate), else_=body("else")),
    )),

    PatternDef("PModeBranch", If(
        Ref("PModePred"),
        then=body("then"),
        else_=Or(Only(Ref("PModeBranch"), "elif"), body("else")),
    )),

    PatternDef("PModePred", Or(
        BinOp(("==", "!="),
              Member(Member(Var("sp", "global"), "mode"), Field("field")),
              Const("modeConst"),
              commutative=True, op_var="op"),
        BinOp("&&", Ref("PModePred"), Ref("PModePred"), op_var="op"),
        BinOp("||", Ref("PModePred"), Ref("PModePred"), op_var="op"),
        UnOp("!", Ref("PModePred")),
    )),

    PatternDef("PContGuard", If(
        Ref("PContPred"),
        then=body("then"),
        else_=Or(Only(Ref("PContGuard"), "elif"), body("else")),
    )),

    PatternDef("PContPred", Or(
        BinOp(CMP_OPS, Expr("lhs", where=is_numeric), Expr("rhs", where=is_numeric), op_var="op"),
        BinOp("&&", Ref("PContPred"), Ref("PContPred"), op_var="op"),
        BinOp("||", Ref("PContPred"), Ref("PContPred"), op_var="op"),
    )),

    PatternDef("PReset", If(
        Expr("g"),
        then=Region("then", nested=BODY, require=[Ref("PResetCall")]),
    )),

    PatternDef("PResetCall", Stmt(Call(RESET_RX, callee_var="fn"))),

    PatternDef("PLoop", Loop(Expr("cond"), body("body"))),

    PatternDef("POtherIf", If(Expr("cond"), then=body("then"), else_=body("else"))),
)
