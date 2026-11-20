"""The paper's switch-based state machine patterns.

  @main @level 0 @coverage partialtoplevel
  PStateCheck:  switch (⟦stateVar⟧) { case ⟦stateConst⟧: PStateCase }    stateVar global
  PStateCase:   { ...PGuard | PTransition... }                           stateConst local
  PGuard:       if (⟦guard⟧) ⟦stateVar⟧ = ⟦target⟧;                     guard, target local
  PTransition:  ⟦stateVar⟧ = ⟦target⟧;                                   unconditional

Because stateVar is global, an assignment to another variable inside a case
(`mode = X`) is not taken as a transition of this machine.

Reading the instance tree: each PStateCase's stateConst is a state, each
PGuard below it a transition (guard -> target).
"""
from __future__ import annotations

from ha_extract.pattern import Assign, Case, Const, Expr, If, PatternDef, PatternSet, Ref, Region, Switch, Var

STATEMACHINE_PATTERNS = PatternSet(
    PatternDef("PStateCheck", Switch(Var("stateVar", "global"), case=Ref("PStateCase")),
               main=True, level=0, coverage="partialtoplevel"),
    PatternDef("PStateCase", Case(Const("stateConst"),
                                  Region("body", nested=[Ref("PGuard"), Ref("PTransition")]))),
    PatternDef("PGuard", If(Expr("guard"),
                            then=Region("then", require=[Assign(Var("stateVar", "global"),
                                                                Const("target"))]))),
    PatternDef("PTransition", Assign(Var("stateVar", "global"), Const("target"))),
)
