"""Control dependence on small snippets (line numbers refer to the snippet text)."""

IF_ELSE = """\
int a, b, c, x;
void f(void) {
  a = 1;
  if (x) {
    b = 2;
  } else {
    c = 3;
  }
  a = 4;
}
"""


def test_if_else(snippet):
    f = snippet(IF_ELSE, "f")
    assert f.deps(3) == {("entry", "entry")}
    assert f.deps(4, "if") == {("entry", "entry")}
    assert f.deps(5) == {(4, "T")}
    assert f.deps(7) == {(4, "F")}
    assert f.deps(9) == {("entry", "entry")}          # after the if: unconditional


EARLY_RETURN = """\
int a, b, x;
void f(void) {
  a = 1;
  if (!x) {
    return;
  }
  a = 2;
  b = 3;
}
"""


def test_early_return_makes_rest_dependent_on_F(snippet):
    f = snippet(EARLY_RETURN, "f")
    assert f.deps(5, "return") == {(4, "T")}
    assert f.deps(7) == {(4, "F")}
    assert f.deps(8) == {(4, "F")}


ELSE_IF = """\
int a, x, y;
void f(void) {
  if (x == 1) {
    a = 1;
  } else if (y == 2) {
    a = 2;
  } else {
    a = 3;
  }
}
"""


def test_else_if_chain_nests_under_F(snippet):
    f = snippet(ELSE_IF, "f")
    assert f.deps(5, "if") == {(3, "F")}
    assert f.deps(6) == {(5, "T")}
    assert f.deps(8) == {(5, "F")}


SWITCH = """\
enum { A, B, C, D };
int s, x, y, z;
void f(void) {
  switch (s) {
    case A:
      x = 1;
    case B:
    case C:
      y = 2;
      break;
    case D:
      z = 3;
      break;
  }
  x = 0;
}
"""


def test_switch_fallthrough_gives_multiple_labels(snippet):
    f = snippet(SWITCH, "f")
    assert f.deps(6) == {(4, "case A")}
    # reached by falling through from A and directly from B and C: matched once per label
    assert f.deps(9) == {(4, "case A"), (4, "case B"), (4, "case C")}
    assert f.deps(12) == {(4, "case D")}
    assert f.deps(15) == {("entry", "entry")}
    labels = {str(l) for l in f.g.groups(f.at(4, "switch"))}
    assert labels == {"case A", "case B", "case C", "case D", "default"}


LOOPS = """\
int i, n, s;
void f(void) {
  for (i = 0; i < n; i++) {
    if (i == 3) continue;
    s += i;
  }
  while (1) {
    s--;
  }
}
"""


def test_loops(snippet):
    f = snippet(LOOPS, "f")
    assert f.deps(4, "if") == {(3, "T")}
    assert f.deps(5) == {(4, "F")}
    assert (3, "T") in f.deps(3, "for")                 # loop predicate depends on itself
    assert f.deps(8) == {(7, "T")}                       # infinite loop still handled
