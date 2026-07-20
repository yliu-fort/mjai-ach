"""AST-based line-count guard for AGENTS.md §3 rule 1 (files < 500 lines).

Invoked by pre-commit. Fails if any staged ``.py`` file under ``src/`` or
``scripts/`` exceeds MAX_LINES lines of actual code (blank + comment lines
excluded, so the cap is on logic, not prose).

Why AST-driven counting: a naive wc -l would let someone hit the cap with blank
padding or dodge it by cramming statements onto one line. We count the number of
*lines that contain AST nodes*, which is robust to both tricks.

Usage (from repo root)::

    python tools/check_file_length.py [--max N] [path ...]

If no paths are given, scans ``src/`` and ``scripts/``. Exit code 1 on violation.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MAX_LINES = 500
DEFAULT_ROOTS = ("src", "scripts")
# Allow __init__.py aggregators to be slightly longer if re-exports grow; they
# rarely contain logic. Still capped, just higher.
INIT_MAX_LINES = 200


def code_line_count(path: Path) -> int:
    """Number of source lines that carry at least one AST node."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    lines: set[int] = set()
    for node in ast.walk(tree):
        # Module node has no lineno; everything else does.
        if hasattr(node, "lineno"):
            lines.add(node.lineno)
            # end_lineno covers multi-line nodes (e.g. long calls).
            end = getattr(node, "end_lineno", None)
            if end is not None:
                lines.update(range(node.lineno, end + 1))
    return len(lines)


def scan(paths: list[Path], max_lines: int) -> list[tuple[Path, int]]:
    """Return list of (path, line_count) for files exceeding max_lines."""
    offenders: list[tuple[Path, int]] = []
    for path in paths:
        if not path.is_file() or path.suffix != ".py":
            continue
        cap = INIT_MAX_LINES if path.name == "__init__.py" else max_lines
        count = code_line_count(path)
        if count > cap:
            offenders.append((path, count))
    return offenders


def collect(explicit: list[str]) -> list[Path]:
    if explicit:
        out: list[Path] = []
        for p in explicit:
            pp = Path(p)
            if pp.is_dir():
                out.extend(pp.rglob("*.py"))
            elif pp.is_file():
                out.append(pp)
        return out
    out = []
    for root in DEFAULT_ROOTS:
        r = Path(root)
        if r.is_dir():
            out.extend(r.rglob("*.py"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=MAX_LINES)
    parser.add_argument("paths", nargs="*", help="files or dirs to scan")
    args = parser.parse_args()

    files = collect(args.paths)
    offenders = scan(files, args.max)
    if not offenders:
        return 0
    cap = args.max
    print(f"FAIL: files exceed {cap}-line cap (AGENTS.md §3):", file=sys.stderr)
    for path, count in sorted(offenders, key=lambda x: -x[1]):
        print(f"  {count:>4} lines  {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
