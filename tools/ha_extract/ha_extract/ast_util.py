"""Small helpers over libclang cursors shared by the CDG builder and the matcher."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import clang.cindex as ci

# --- operator kinds -------------------------------------------------------
# The Python bindings shipped with libclang 18 do not expose these, but the
# C library does (LLVM >= 17). They work for nodes inside macro expansions,
# where token-based guessing fails.
_lib = ci.conf.lib
_lib.clang_getCursorBinaryOperatorKind.argtypes = [ci.Cursor]
_lib.clang_getCursorBinaryOperatorKind.restype = ctypes.c_uint
_lib.clang_getBinaryOperatorKindSpelling.argtypes = [ctypes.c_uint]
_lib.clang_getBinaryOperatorKindSpelling.restype = ci._CXString
_lib.clang_getBinaryOperatorKindSpelling.errcheck = ci._CXString.from_result
_lib.clang_getCursorUnaryOperatorKind.argtypes = [ci.Cursor]
_lib.clang_getCursorUnaryOperatorKind.restype = ctypes.c_uint
_lib.clang_getUnaryOperatorKindSpelling.argtypes = [ctypes.c_uint]
_lib.clang_getUnaryOperatorKindSpelling.restype = ci._CXString
_lib.clang_getUnaryOperatorKindSpelling.errcheck = ci._CXString.from_result

_BINARY_KINDS = (ci.CursorKind.BINARY_OPERATOR, ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR)


def binary_op(c: ci.Cursor) -> str | None:
    """'==', '&&', '=', '+=', ... for binary operator cursors, else None."""
    if c.kind not in _BINARY_KINDS:
        return None
    k = _lib.clang_getCursorBinaryOperatorKind(c)
    return _lib.clang_getBinaryOperatorKindSpelling(k) if k else None


def unary_op(c: ci.Cursor) -> str | None:
    """'!', '-', '*', '&', ... for unary operator cursors, else None."""
    if c.kind != ci.CursorKind.UNARY_OPERATOR:
        return None
    k = _lib.clang_getCursorUnaryOperatorKind(c)
    return _lib.clang_getUnaryOperatorKindSpelling(k) if k else None


# --- normalisation --------------------------------------------------------
def strip(c: ci.Cursor) -> ci.Cursor:
    """Drop implicit casts (UNEXPOSED_EXPR) and parentheses."""
    while c.kind in (ci.CursorKind.UNEXPOSED_EXPR, ci.CursorKind.PAREN_EXPR):
        kids = list(c.get_children())
        if len(kids) != 1:
            break
        c = kids[0]
    return c


def children(c: ci.Cursor) -> list[ci.Cursor]:
    return list(c.get_children())


def is_numeric(c: ci.Cursor) -> bool:
    """Arithmetic (int/float) and not an enum: a continuous-ish quantity."""
    t = strip(c).type.get_canonical()
    return t.kind in _NUMERIC_TYPES


_NUMERIC_TYPES = {
    ci.TypeKind.FLOAT, ci.TypeKind.DOUBLE, ci.TypeKind.LONGDOUBLE,
    ci.TypeKind.INT, ci.TypeKind.UINT, ci.TypeKind.SHORT, ci.TypeKind.USHORT,
    ci.TypeKind.LONG, ci.TypeKind.ULONG, ci.TypeKind.LONGLONG, ci.TypeKind.ULONGLONG,
    ci.TypeKind.CHAR_S, ci.TypeKind.UCHAR, ci.TypeKind.SCHAR, ci.TypeKind.CHAR_U,
}


# --- macros ---------------------------------------------------------------
@dataclass(frozen=True)
class MacroUse:
    name: str
    start: int
    end: int
    args: tuple[str, ...]
    text: str


class MacroIndex:
    """Macro expansions in the main file.

    Nodes produced by a macro expansion have extents pointing at the macro
    call site, so their tokens are the call's tokens, not the node's own.
    """

    def __init__(self, tu: ci.TranslationUnit):
        main_file = tu.spelling
        self.uses: list[MacroUse] = []
        for c in tu.cursor.get_children():
            if c.kind != ci.CursorKind.MACRO_INSTANTIATION:
                continue
            if not (c.location.file and c.location.file.name == main_file):
                continue
            toks = [t.spelling for t in c.get_tokens()]
            self.uses.append(MacroUse(c.spelling, c.extent.start.offset, c.extent.end.offset,
                                      _macro_args(toks), " ".join(toks)))

    def containing(self, c: ci.Cursor) -> MacroUse | None:
        a, b = c.extent.start.offset, c.extent.end.offset
        best = None
        for u in self.uses:
            if u.start <= a and b <= u.end and (best is None or u.end - u.start < best.end - best.start):
                best = u
        return best


def _macro_args(toks: list[str]) -> tuple[str, ...]:
    """Split `NAME ( a , f ( b , c ) )` into ('a', 'f ( b , c )')."""
    if len(toks) < 3 or toks[1] != "(":
        return ()
    args, cur, depth = [], [], 0
    for t in toks[2:]:
        if t in "([{":
            depth += 1
        elif t in ")]}":
            if depth == 0:
                break
            depth -= 1
        if t == "," and depth == 0:
            args.append(" ".join(cur)); cur = []
        else:
            cur.append(t)
    args.append(" ".join(cur))
    return tuple(a for a in args if a)


def source_text(c: ci.Cursor, macros: MacroIndex | None = None, max_len: int | None = None) -> str:
    """Source text of a cursor; for macro-expanded nodes, the macro call text."""
    if macros is not None:
        use = macros.containing(c)
        if use is not None:
            return use.text
    text = " ".join(t.spelling for t in c.get_tokens())
    if max_len is not None and len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text
