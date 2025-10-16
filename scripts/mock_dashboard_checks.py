#!/usr/bin/env python3
"""Run CLI-based dashboard contract checks without launching Streamlit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

REQUIRED_COLUMNS: Sequence[str] = (
    "sido",
    "sigungu",
    "eupmyeondong",
    "area_bucket",
    "floor_bucket",
    "price_per_m2",
    "Yield_%",
    "floor_insight_enabled",
)

AGG_COLUMNS: Sequence[str] = (
    "sido",
    "sigungu",
    "area_bucket",
)


def null_ratio(series: pd.Series) -> float:
    return float(series.isna().sum()) / float(len(series)) if len(series) else 0.0


def load_transactions(path: Path, limit: int | None) -> pd.DataFrame:
    kwargs = {}
    if limit:
        kwargs["nrows"] = limit
    df = pd.read_csv(path, low_memory=False, **kwargs)
    return df


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, bool]:
    return {column: column in df.columns for column in columns}


def compute_combos(df: pd.DataFrame, sample_size: int = 10) -> pd.DataFrame:
    if not set(AGG_COLUMNS).issubset(df.columns):
        return pd.DataFrame()
    grouped = (
        df.groupby(list(AGG_COLUMNS))
        .agg(
            txn_count=("src_type", "count"),
            avg_price_per_m2=("price_per_m2", "mean"),
            avg_yield=("Yield_%", "mean"),
        )
        .reset_index()
        .sort_values("txn_count", ascending=False)
    )
    return grouped.head(sample_size)


def write_markdown(
    output_path: Path,
    column_presence: Dict[str, bool],
    null_stats: Dict[str, float],
    combo_stats: pd.DataFrame,
    monthly_checks: Dict[str, bool],
) -> None:
    lines: List[str] = []
    lines.append("# Dashboard Contract Checks")
    lines.append("")
    lines.append("## Column Presence")
    for column, exists in column_presence.items():
        lines.append(f"- `{column}`: {'OK' if exists else 'MISSING'}")
    lines.append("")
    lines.append("## Null Ratios")
    for column, ratio in null_stats.items():
        lines.append(f"- `{column}`: {ratio:.4f}")
    lines.append("")
    lines.append("## Aggregated Sample (Top Combos)")
    if combo_stats.empty:
        lines.append("- No aggregation available (required columns missing)")
    else:
        lines.append("")
        lines.append("| sido | sigungu | area_bucket | txn_count | avg_price_per_m2 | avg_yield |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for _, row in combo_stats.iterrows():
            lines.append(
                "| {sido} | {sigungu} | {area} | {txn:,.0f} | {price:,.2f} | {yield_:,.2f} |".format(
                    sido=row.get("sido", ""),
                    sigungu=row.get("sigungu", ""),
                    area=row.get("area_bucket", ""),
                    txn=row.get("txn_count", 0),
                    price=row.get("avg_price_per_m2", 0.0),
                    yield_=row.get("avg_yield", 0.0),
                )
            )
    lines.append("")
    lines.append("## Monthly Artifacts")
    for artifact, present in monthly_checks.items():
        lines.append(f"- `{artifact}`: {'OK' if present else 'MISSING'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_combo_samples(path: Path, combo_stats: pd.DataFrame) -> None:
    if combo_stats.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    combo_stats.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True, help="Processed data directory")
    parser.add_argument("--out", required=True, help="Output markdown path for summary")
    parser.add_argument("--samples-out", help="Optional CSV path for aggregated sample output")
    parser.add_argument("--limit", type=int, help="Optional row limit when loading transactions")
    args = parser.parse_args()

    processed_dir = Path(args.processed).expanduser().resolve()
    transactions_path = processed_dir / "transactions.csv"
    if not transactions_path.exists():
        raise SystemExit(f"transactions.csv not found under {processed_dir}")

    transactions = load_transactions(transactions_path, args.limit)
    column_presence = ensure_columns(transactions, REQUIRED_COLUMNS)
    null_stats = {
        column: null_ratio(transactions[column]) if column in transactions.columns else 1.0
        for column in REQUIRED_COLUMNS
    }
    combo_stats = compute_combos(transactions)

    monthly_checks = {}
    for artifact in [
        "monthly_basics.parquet",
        "monthly_momentum.parquet",
        "monthly_volatility.parquet",
    ]:
        monthly_checks[artifact] = (processed_dir / artifact).exists()

    write_markdown(Path(args.out).expanduser(), column_presence, null_stats, combo_stats, monthly_checks)

    if args.samples_out:
        write_combo_samples(Path(args.samples_out).expanduser(), combo_stats)

    print("Dashboard contract check completed.")


if __name__ == "__main__":
    main()
