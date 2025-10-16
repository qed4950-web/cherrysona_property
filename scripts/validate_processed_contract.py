#!/usr/bin/env python3
"""Validate processed dataset contract and emit schema/quality summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

EXPECTED_FILES = {
    "transactions": "transactions.csv",
    "monthly_basics": "monthly_basics.parquet",
    "monthly_momentum": "monthly_momentum.parquet",
    "monthly_volatility": "monthly_volatility.parquet",
    "quality_report": "quality_report.json",
}

NUMERIC_HEALTH_CHECK_COLUMNS: Dict[str, List[str]] = {
    "transactions": [
        "price_per_m2",
        "Yield_%",
        "거래금액_만원",
        "전용면적_㎡",
        "추정매입가_만원",
    ],
    "monthly_basics": [
        "avg_price_per_m2",
        "txn_cnt",
        "avg_yield_pct",
        "cancel_rate",
    ],
    "monthly_momentum": [
        "avg_p_per_m2",
        "avg_p_per_m2_lag3",
    ],
    "monthly_volatility": [
        "cv_p_per_m2",
        "std_p_per_m2",
        "avg_p_per_m2",
    ],
}


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def compute_null_ratio(frame: pd.DataFrame) -> Dict[str, float]:
    total = len(frame)
    if total == 0:
        return {column: 1.0 for column in frame.columns}
    return {
        column: float(frame[column].isna().sum()) / total
        for column in frame.columns
    }


def compute_negative_counts(frame: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for column in columns:
        if column not in frame.columns:
            counts[column] = -1  # mark as missing
            continue
        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        counts[column] = int((numeric_series < 0).sum())
    return counts


def infer_schema(frame: pd.DataFrame) -> Dict[str, str]:
    return {
        column: str(dtype)
        for column, dtype in frame.dtypes.items()
    }


def build_summary(input_dir: Path) -> Dict[str, object]:
    summary: Dict[str, object] = {"input_dir": str(input_dir.resolve()), "files": {}}
    for key, relative_path in EXPECTED_FILES.items():
        target_path = input_dir / relative_path
        entry: Dict[str, object] = {
            "path": str(target_path),
            "exists": target_path.exists(),
        }
        if not target_path.exists():
            summary["files"][key] = entry
            continue

        if target_path.suffix in {".csv", ".parquet"}:
            frame = load_dataframe(target_path)
            entry.update(
                {
                    "rows": int(len(frame)),
                    "columns": list(frame.columns),
                    "schema": infer_schema(frame),
                    "null_ratio": compute_null_ratio(frame),
                    "negative_counts": compute_negative_counts(
                        frame,
                        NUMERIC_HEALTH_CHECK_COLUMNS.get(key, []),
                    ),
                }
            )
        elif target_path.suffix == ".json":
            entry["preview"] = target_path.read_text(encoding="utf-8")[:1000]
        summary["files"][key] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Processed data directory (expects transactions.csv and monthly_* files)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write validation summary JSON",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    summary = build_summary(input_dir)

    output_path = Path(args.out).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Validation summary saved to {output_path}")


if __name__ == "__main__":
    main()
