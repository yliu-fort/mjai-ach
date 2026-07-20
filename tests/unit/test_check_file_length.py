"""Unit test for the AST line-count guard (tools/check_file_length.py).

Verifies the guard counts AST-bearing lines (not blank/comment padding) and
flags files over the cap. This guards the guard, since it's a local hook.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_file_length.py"


def _load_tool():
    """Load the standalone tool script as a module (no package needed)."""
    spec = importlib.util.spec_from_file_location("check_file_length", _TOOL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_file_length"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_short_file_passes(tmp_path):
    mod = _load_tool()
    p = _write(
        tmp_path,
        "short.py",
        """
        def f(x):
            return x + 1
        """,
    )
    assert mod.code_line_count(p) <= mod.MAX_LINES
    assert mod.scan([p], mod.MAX_LINES) == []


def test_comments_and_blanks_do_not_count(tmp_path):
    """1000 comment lines must not trip the cap — only logic lines count."""
    mod = _load_tool()
    body = "\n".join(["# comment " * 1 for _ in range(1000)] + ["x = 1"])
    p = _write(tmp_path, "comments.py", body)
    assert mod.code_line_count(p) == 1


def test_long_file_flagged(tmp_path):
    """A file with >cap AST lines is reported as an offender."""
    mod = _load_tool()
    cap = 5
    # 10 distinct statements -> >cap AST-bearing lines.
    body = "\n".join(f"x{i} = {i}" for i in range(10))
    p = _write(tmp_path, "long.py", body)
    offenders = mod.scan([p], cap)
    assert len(offenders) == 1
    assert offenders[0][0] == p
    assert offenders[0][1] > cap


def test_init_gets_lower_cap(tmp_path):
    """__init__.py aggregators get the smaller INIT_MAX_LINES cap."""
    mod = _load_tool()
    # Write an __init__.py with more than INIT_MAX_LINES statements.
    body = "\n".join(f"x{i} = {i}" for i in range(mod.INIT_MAX_LINES + 5))
    p = _write(tmp_path, "__init__.py", body)
    offenders = mod.scan([p], mod.MAX_LINES)  # default cap is MAX_LINES
    # Should still be flagged because it exceeds INIT_MAX_LINES, even though
    # under MAX_LINES.
    assert len(offenders) == 1


def test_main_exit_code(tmp_path, monkeypatch, capsys):
    """main() returns nonzero and prints offenders when violations exist."""
    mod = _load_tool()
    monkeypatch.chdir(tmp_path)
    p = _write(tmp_path, "long.py", "\n".join(f"x{i}={i}" for i in range(10)))
    monkeypatch.setattr(sys, "argv", ["check_file_length.py", "--max", "3", str(p)])
    code = mod.main()
    assert code == 1
    captured = capsys.readouterr()
    assert "long.py" in captured.err
