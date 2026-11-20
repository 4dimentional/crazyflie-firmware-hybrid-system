"""Pattern matching: the paper's state machine example and the two controllers."""
from ha_extract.pattern import match_function
from ha_extract.patterns_controller import CONTROLLER_PATTERNS
from ha_extract.patterns_statemachine import STATEMACHINE_PATTERNS

MELLINGER = "src/modules/src/controller/controller_mellinger.c"
PID = "src/modules/src/controller/controller_pid.c"


def run(fn, patterns):
    roots = match_function(fn.g, fn.macros, patterns)
    return roots


def b(inst, name):
    return inst.bindings[name].text


def mode_atoms(pred):
    """Leaf (field, op, modeConst) triples of a PModePred instance tree."""
    return [(b(i, "field"), b(i, "op"), b(i, "modeConst")) for i in pred.walk() if "field" in i.bindings]


def child(inst, pattern, slot=None):
    found = [c for c in inst.children if c.pattern == pattern and (slot is None or c.slot == slot)]
    assert len(found) == 1, (pattern, slot, [(c.pattern, c.slot) for c in inst.children])
    return found[0]


# ---------------------------------------------------------------- paper example
CYCLIC = """\
typedef enum { WRITING, WAITING_FOR_ACK, IDLE } st_t;
static st_t state, mode;
static int done, ack;
void Cyclic10ms(void) {
  switch (state) {
    case WRITING:
      if (done) state = WAITING_FOR_ACK;
      if (ack) mode = IDLE;
      break;
    case WAITING_FOR_ACK:
      if (ack) state = IDLE;
      break;
    case IDLE:
      break;
  }
}
"""


def test_paper_state_machine(snippet):
    f = snippet(CYCLIC, "Cyclic10ms")
    (root,) = run(f, STATEMACHINE_PATTERNS)
    assert root.globals["stateVar"].text == "state"
    cases = {b(c, "stateConst"): c for c in root.children}
    assert set(cases) == {"WRITING", "WAITING_FOR_ACK", "IDLE"}
    # `mode = IDLE` assigns another variable, so the global stateVar rejects it
    assert [(b(g, "guard"), b(g, "target")) for g in cases["WRITING"].children] == [("done", "WAITING_FOR_ACK")]
    assert [(b(g, "guard"), b(g, "target")) for g in cases["WAITING_FOR_ACK"].children] == [("ack", "IDLE")]
    assert cases["IDLE"].children == ()


def test_global_metavariable_keeps_one_setpoint(snippet):
    code = """\
typedef enum { modeDisable, modeAbs } m_t;
typedef struct { struct { m_t x, z; } mode; float thrust; } sp_t;
float out;
void ctl(const sp_t *setpoint, const sp_t *other) {
  if (setpoint->mode.x == modeAbs) {
    out = 1;
  } else if (other->mode.z == modeAbs) {
    out = 2;
  }
}
"""
    f = snippet(code, "ctl")
    (root,) = run(f, CONTROLLER_PATTERNS)
    branch = child(root, "PModeBranch")
    assert root.globals["sp"].text == "setpoint"
    # the else-if tests a different struct: not a branch of the same mode switch
    assert [c.pattern for c in branch.children if c.slot != "cond"] == ["POtherIf"]


# ---------------------------------------------------------------- Mellinger
def test_mellinger_mode_tree(firmware):
    f = firmware(MELLINGER, "controllerMellinger")
    (root,) = run(f, CONTROLLER_PATTERNS)
    assert root.globals["sp"].text == "setpoint"

    timing = child(root, "PTiming")                      # if (!RATE_DO_EXECUTE(...)) return;
    assert (b(timing, "rate"), timing.slot) == ("ATTITUDE_RATE", "top")
    branches = [c for c in timing.children if c.pattern == "PModeBranch"]
    assert [x.line for x in branches] == [152, 181]      # yaw chain, main mode chain

    yaw = branches[0]
    chain = []
    x = yaw
    while x is not None:
        chain.append(mode_atoms(child(x, "PModePred", "cond")))
        x = next((c for c in x.children if c.slot == "elif"), None)
    assert chain == [[("yaw", "==", "modeVelocity")], [("yaw", "==", "modeAbs")], [("quat", "==", "modeAbs")]]

    position = branches[1]
    assert mode_atoms(child(position, "PModePred", "cond")) == [
        ("x", "==", "modeAbs"), ("y", "==", "modeAbs"), ("z", "==", "modeAbs")]
    manual = child(position, "PModeBranch", "elif")
    assert mode_atoms(child(manual, "PModePred", "cond")) == [
        ("pitch", "==", "modeAbs"), ("roll", "==", "modeAbs"), ("z", "==", "modeDisable")]
    fallback_z = child(manual, "PModeBranch", "else")    # nested inside the fallback body
    assert mode_atoms(child(fallback_z, "PModePred", "cond")) == [("z", "==", "modeDisable")]

    guards = {g.line: g for g in timing.children if g.pattern == "PContGuard"}
    airborne = guards[306]                               # if (control->thrust > 0)
    pred = child(airborne, "PContPred", "cond")
    assert (b(pred, "lhs"), b(pred, "op"), b(pred, "rhs")) == ("control -> thrust", ">", "0")
    assert b(child(airborne, "PResetCall", "else"), "fn") == "controllerMellingerReset"


# ---------------------------------------------------------------- PID
def test_pid_mode_tree(firmware):
    f = firmware(PID, "controllerPid")
    (root,) = run(f, CONTROLLER_PATTERNS)
    tops = [(c.pattern, c.line) for c in root.children]
    assert tops == [("PReset", 81), ("PTiming", 85), ("PTiming", 115), ("PTiming", 119), ("PContGuard", 163)]

    reset = root.children[0]
    assert b(reset, "g") == "setpointModeChanged ( setpoint )"
    assert b(child(reset, "PResetCall", "then"), "fn") == "controllerPidReinitialize"

    yaw = child(root.children[1], "PModeBranch")
    atoms = []
    x = yaw
    while x is not None:
        atoms.append(mode_atoms(child(x, "PModePred", "cond")))
        x = next((c for c in x.children if c.slot == "elif"), None)
    assert atoms == [[("yaw", "==", "modeVelocity")], [("yaw", "==", "modeAbs")], [("quat", "==", "modeAbs")]]
    # the yawMaxDelta clamp is a continuous guard inside the velocity-mode body
    assert [g.line for g in yaw.children if g.pattern == "PContGuard"] == [91]

    att = root.children[3]
    preds = [mode_atoms(child(x, "PModePred", "cond")) for x in att.children if x.pattern == "PModeBranch"]
    assert preds == [
        [("z", "==", "modeDisable")],
        [("x", "==", "modeDisable"), ("y", "==", "modeDisable")],
        [("roll", "==", "modeVelocity")],
        [("pitch", "==", "modeVelocity")],
    ]

    idle = root.children[4]                              # if (control->thrust == 0) { ...reinit }
    assert b(child(idle, "PResetCall", "then"), "fn") == "controllerPidReinitialize"


def test_local_state_var_would_accept_foreign_assignment(snippet):
    """Control for test_paper_state_machine: with a *local* stateVar the
    unrelated `mode = IDLE` is wrongly taken as a transition."""
    from ha_extract.pattern import (Assign, Case, Const, Expr, If, PatternDef, PatternSet, Ref,
                                    Region, Switch, Var)
    local = PatternSet(
        PatternDef("PStateCheck", Switch(Var("stateVar", "local"), case=Ref("PStateCase")),
                   main=True, level=0),
        PatternDef("PStateCase", Case(Const("stateConst"), Region("body", nested=[Ref("PGuard")]))),
        PatternDef("PGuard", If(Expr("guard"), then=Region(
            "then", require=[Assign(Var("stateVar", "local"), Const("target"))]))),
    )
    f = snippet(CYCLIC, "Cyclic10ms")
    (root,) = run(f, local)
    writing = next(c for c in root.children if b(c, "stateConst") == "WRITING")
    assert [(b(g, "guard"), b(g, "target")) for g in writing.children] == [
        ("done", "WAITING_FOR_ACK"), ("ack", "IDLE")]
