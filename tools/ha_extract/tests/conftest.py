import pytest

from ha_extract.ast_util import MacroIndex
from ha_extract.cdg import FunctionGraphs, function_definitions
from ha_extract.parse import parse_file, parse_source


class Fn:
    """A parsed function with its graphs; keeps the translation unit alive."""

    def __init__(self, result, name):
        assert result.ok, [d.spelling for d in result.errors]
        self.tu = result.tu
        self.macros = MacroIndex(self.tu)
        fn = next(c for c in function_definitions(self.tu) if c.spelling == name)
        self.g = FunctionGraphs(fn)

    def at(self, line, kind=None):
        found = [n for n in self.g.cdg if n.line == line and (kind is None or n.kind == kind)]
        assert len(found) == 1, (line, kind, found)
        return found[0]

    def deps(self, line, kind=None):
        """{(parent line or 'entry', label text)} for the node at `line`."""
        n = self.at(line, kind)
        return {(p.line if p.kind != "entry" else "entry", str(l)) for p, l in self.g.dependences(n)}


@pytest.fixture
def snippet():
    def make(code, name):
        return Fn(parse_source(code), name)
    return make


@pytest.fixture(scope="session")
def firmware():
    cache = {}

    def get(path, name):
        if (path, name) not in cache:
            cache[(path, name)] = Fn(parse_file(path), name)
        return cache[(path, name)]
    return get
