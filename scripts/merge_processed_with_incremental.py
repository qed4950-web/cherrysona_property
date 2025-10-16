#!/usr/bin/env python3
"""Merge baseline processed outputs with incremental outputs and recompute aggregates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

AGG_KEYS: Sequence[str] = [
    "src_type",
    "sido",
    "sigungu",
    "eupmyeondong",
    "YYYYMM",
]

TRANSACTION_KEY: Sequence[str] = [
    "src_type",
    "sido",
    "sigungu",
    "eupmyeondong",
    "계약일자",
    "YYYYMM",
    "거래금액_만원",
    "전용면적_㎡",
    "geo_hash",
    "date_key",
]


def load_transactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"transactions.csv not found: {path}")
    return pd.read_csv(path, low_memory=False)


def merge_transactions(base: pd.DataFrame, incremental: pd.DataFrame) -> pd.DataFrame:
    for col in TRANSACTION_KEY:
        if col not in base.columns or col not in incremental.columns:
            raise SystemExit(f"Missing column '{col}' in processed datasets")
    merged = pd.concat([base, incremental], ignore_index=True)
    merged = merged.drop_duplicates(subset=list(TRANSACTION_KEY), keep="last")
    merged.sort_values(
        ["src_type", "YYYYMM", "sido", "sigungu", "eupmyeondong", "계약일자"],
        inplace=True,
    )
    merged.reset_index(drop=True, inplace=True)
    return merged


def compute_monthly_basics(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=list(AGG_KEYS) + [
            "txn_cnt",
            "avg_p_per_m2",
            "std_p_per_m2",
            "avg_yield_pct",
            "cancel_rate",
        ])
    grouped = (
        transactions.groupby(list(AGG_KEYS), dropna=False)
        .agg(
            txn_cnt=("거래금액_만원", "count"),
            avg_p_per_m2=("가격_per_㎡", "mean"),
            std_p_per_m2=("가격_per_㎡", "std"),
            avg_yield_pct=("Yield_%", "mean"),
            cancel_rate=("취소여부", "mean"),
        )
        .reset_index()
    )
    grouped["std_p_per_m2"] = grouped["std_p_per_m2"].fillna(0)
    grouped["cancel_rate"] = grouped["cancel_rate"].fillna(0)
    return grouped.sort_values(AGG_KEYS).reset_index(drop=True)


def compute_momentum(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    monthly = monthly.sort_values(AGG_KEYS)
    monthly["avg_p_per_m2_lag3"] = (
        monthly.groupby(["src_type", "sido", "sigungu", "eupmyeondong"])["avg_p_per_m2"]
        .shift(3)
    )
    monthly["mom_3m"] = monthly["avg_p_per_m2"] / monthly["avg_p_per_m2_lag3"] - 1
    monthly.loc[monthly["avg_p_per_m2_lag3"].isna(), "mom_3m"] = 0
    monthly.loc[
        monthly["mom_3m"].replace([float("inf"), float("-inf")], pd.NA).isna(),
        "mom_3m",
    ] = 0
    return monthly.reset_index(drop=True)


def compute_volatility(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    vol = monthly.copy()
    vol["cv_p_per_m2"] = vol["std_p_per_m2"] / vol["avg_p_per_m2"].replace(0, pd.NA)
    vol["cv_p_per_m2"] = vol["cv_p_per_m2"].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    return vol.reset_index(drop=True)


def persist_outputs(transactions: pd.DataFrame, monthly: pd.DataFrame, momentum: pd.DataFrame, volatility: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(output_dir / "transactions.csv", index=False)
    monthly.to_parquet(output_dir / "monthly_basics.parquet", index=False)
    momentum.to_parquet(output_dir / "monthly_momentum.parquet", index=False)
    volatility.to_parquet(output_dir / "monthly_volatility.parquet", index=False)


def merge_quality_reports(base: Path, incremental: Path, output: Path) -> None:
    contents = []
    for path in (base, incremental):
        if path.exists():
            contents.append(path.read_text(encoding="utf-8"))
    if contents:
        output.write_text("\n".join(contents), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Baseline processed directory")
    parser.add_argument("--incremental", required=True, help="Incremental processed directory")
    parser.add_argument("--output", required=True, help="Output directory for merged dataset")
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser().resolve()
    inc_dir = Path(args.incremental).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()

    base_transactions = load_transactions(base_dir / "transactions.csv")
    incremental_transactions = load_transactions(inc_dir / "transactions.csv")

    merged_transactions = merge_transactions(base_transactions, incremental_transactions)
    monthly_basics = compute_monthly_basics(merged_transactions)
    monthly_momentum = compute_momentum(monthly_basics.copy())
    monthly_volatility = compute_volatility(monthly_basics.copy())

    persist_outputs(
        transactions=merged_transactions,
        monthly=monthly_basics,
        momentum=monthly_momentum,
        volatility=monthly_volatility,
        output_dir=out_dir,
    )

    merge_quality_reports(
        base_dir / "quality_report.json",
        inc_dir / "quality_report.json",
        out_dir / "quality_report.json",
    )

    print(f"Merged processed dataset written to {out_dir}")


if __name__ == "__main__":
    main()
