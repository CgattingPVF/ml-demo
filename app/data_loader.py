from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv", ".xlsx", ".xls"}
ENCODING_CANDIDATES = ("utf-16", "utf-8-sig", "utf-8", "latin-1")


def _guess_delimiter(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="ignore") as handle:
        first_line = handle.readline()

    scores = {
        "\t": first_line.count("\t"),
        ",": first_line.count(","),
        ";": first_line.count(";"),
        "|": first_line.count("|"),
    }
    return max(scores, key=scores.get)


def _read_text_dataset(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODING_CANDIDATES:
        try:
            delimiter = _guess_delimiter(path, encoding)
            return pd.read_csv(
                path,
                sep=delimiter,
                encoding=encoding,
                low_memory=False,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
            continue

    raise ValueError(f"Could not read {path.name}: {last_error}") from last_error


def load_dataset(path: str | Path) -> pd.DataFrame:
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported dataset type {source_path.suffix}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if source_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(source_path)

    return _read_text_dataset(source_path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    copy_df = df.copy()
    copy_df.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        for col in copy_df.columns
    ]
    return copy_df


def dataset_profile(df: pd.DataFrame, top_n: int = 25) -> dict[str, Any]:
    missing_pct = (
        (df.isna().mean() * 100)
        .sort_values(ascending=False)
        .head(top_n)
        .round(2)
        .to_dict()
    )
    dtype_breakdown = df.dtypes.astype(str).value_counts().to_dict()
    cardinality = df.nunique(dropna=True).sort_values(ascending=False).head(top_n).to_dict()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtype_breakdown": {str(k): int(v) for k, v in dtype_breakdown.items()},
        "top_missing_pct": {str(k): float(v) for k, v in missing_pct.items()},
        "top_cardinality": {str(k): int(v) for k, v in cardinality.items()},
    }

