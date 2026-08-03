"""Compact status printing and contextual error helpers shared by src modules.

Preserves the existing notebook-facing print style (flush=True, indented notes,
``[status] name`` items, repo-relative paths). Not a logging framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

_REPO_ROOT: Optional[Path] = None


def set_project_root(root: PathLike) -> None:
    """Set the repo root used for relative status paths."""
    global _REPO_ROOT
    _REPO_ROOT = Path(root).resolve()


def detect_repo_root(hint: Optional[PathLike] = None) -> Path:
    """Return repo root (directory containing ``src/`` and usually ``data/``)."""
    if _REPO_ROOT is not None:
        return _REPO_ROOT
    if hint is not None:
        start = Path(hint).resolve()
        for parent in [start, *start.parents]:
            if (parent / "src").is_dir() and (parent / "data").is_dir():
                return parent
    # Prefer the package location (``.../repo/src/status_io.py`` -> repo).
    here = Path(__file__).resolve().parent
    if here.name == "src":
        return here.parent
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    if (cwd / "src").is_dir():
        return cwd
    if (cwd.parent / "src").is_dir():
        return cwd.parent
    return cwd


def rel(path: PathLike) -> str:
    """Path relative to the repo root when possible; else the basename."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(detect_repo_root(path))).replace("\\", "/")
    except ValueError:
        return path.name


def announce(msg: str) -> None:
    print(msg, flush=True)


def note(msg: str) -> None:
    announce(f"  {msg}")


def item(name: str, status: str = "ok") -> None:
    announce(f"  [{status}] {name}")


def section(title: str, directory: PathLike) -> None:
    announce(title)
    announce(f"  dir: {rel(directory).rstrip('/')}/")


def raise_ctx(
    exc_type: type,
    message: str,
    *,
    cause: Optional[BaseException] = None,
) -> None:
    """Raise ``exc_type(message)``, optionally chained from *cause*."""
    if cause is not None:
        raise exc_type(message) from cause
    raise exc_type(message)


def format_batch_failures(
    desc: str,
    errors: Sequence[Tuple[Any, BaseException]],
    n_total: int,
    *,
    limit: int = 5,
) -> str:
    """One-line summary for aggregated item failures (``N/M failed: ...``)."""
    detail = "; ".join(f"{item_}: {exc}" for item_, exc in errors[:limit])
    more = f" (+{len(errors) - limit} more)" if len(errors) > limit else ""
    return f"{desc}: {len(errors)}/{n_total} failed: {detail}{more}"


def summarize_skipped(
    label: str,
    n_failed: int,
    n_total: int,
    *,
    examples: Optional[Iterable[Any]] = None,
    limit: int = 5,
) -> None:
    """Print a compact skipped/failed batch note (no raise)."""
    if n_failed <= 0:
        return
    msg = f"{n_failed}/{n_total} {label} skipped"
    if examples is not None:
        xs = list(examples)[:limit]
        if xs:
            msg += f" (e.g. {', '.join(map(str, xs))}"
            if n_failed > limit:
                msg += ", ..."
            msg += ")"
    note(msg)
