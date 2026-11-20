import pytest

from ha_extract.parse import parse_file

CONTROLLERS = [
    "src/modules/src/controller/controller_pid.c",
    "src/modules/src/controller/controller_mellinger.c",
]


@pytest.mark.parametrize("path", CONTROLLERS)
def test_controller_parses_without_errors(path):
    r = parse_file(path)
    assert r.ok, [f"{d.location.line}: {d.spelling}" for d in r.errors]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
