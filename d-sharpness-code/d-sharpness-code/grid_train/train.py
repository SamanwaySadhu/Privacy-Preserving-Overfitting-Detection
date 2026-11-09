from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
import datetime
import itertools
import json

import pandas as pd


@dataclass
class GridPaths:
    """Resolved output paths for a grid run."""

    out_dir: Path
    csv_path: Path


def timestamp_now() -> str:
    """Return a timestamp string suitable for directory names."""
    return datetime.datetime.now().strftime("%m%d-%H%M%S")


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_grid_paths(
    *,
    out_root: str,
    experiment_name: str,
    append_csv_path: Optional[str] = None,
    force_append: bool = False,
) -> GridPaths:
    """
    Create output directories and resolve the CSV path.

    Args:
        out_root: Base directory for all grid outputs.
        experiment_name: Name prefix for the run directory.
        append_csv_path: Optional existing CSV to append to.
        force_append: If True, always append to append_csv_path.

    Returns:
        GridPaths with the resolved output directory and CSV path.
    """
    out_dir = ensure_dir(Path(out_root) / f"{experiment_name}_{timestamp_now()}")
    if force_append and append_csv_path:
        csv_path = Path(append_csv_path)
        ensure_dir(csv_path.parent)
    else:
        csv_path = out_dir / "runs.csv"
    return GridPaths(out_dir=out_dir, csv_path=csv_path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Write a JSON payload to disk with indentation.

    Args:
        path: File path to write.
        payload: JSON-serializable dictionary.
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def csv_has_rows(path: Path) -> bool:
    """Return True if a CSV file exists and has data rows."""
    return path.exists() and path.stat().st_size > 0


def append_csv_row(path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    """
    Append a single row to a CSV file, creating it with a header if needed.

    Args:
        path: CSV file path.
        row: Row dictionary to append.
        columns: Column order for the CSV.
    """
    ensure_dir(path.parent)
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(path, mode="a", header=not csv_has_rows(path), index=False)


def grid_product(grid: Dict[str, Iterable[Any]]) -> Iterator[Dict[str, Any]]:
    """
    Iterate over the cartesian product of a grid dictionary.

    Args:
        grid: Mapping of parameter name to iterable of values.

    Yields:
        Dictionary of one concrete hyperparameter setting.
    """
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))
