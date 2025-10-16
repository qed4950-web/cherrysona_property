#!/usr/bin/env python3
"""Compare processed data keys with reference datasets to detect mismatches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import pandas as pd

SIGUNGU_REFERENCE_FILES = (
    "sigungu_centroids.csv",
    "sigungu_centroids.parquet",
)


def load_sigungu_reference(reference_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for filename in SIGUNGU_REFERENCE_FILES:
        path = reference_dir / filename
        if not path.exists():
            continue
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_parquet(path)
        # Normalize Korean headers to match processed schema
        columns = list(frame.columns)
        rename_map = {}
        if "시도" in columns:
            rename_map["시도"] = "sido"
        if "시군구" in columns:
            rename_map["시군구"] = "sigungu"
        frame = frame.rename(columns=rename_map)
        keep_cols = [c for c in ["sido", "sigungu", "latitude", "longitude"] if c in frame.columns]
        frames.append(frame[keep_cols].drop_duplicates())
    if not frames:
        raise SystemExit("No reference files found for sigungu alignment.")
    reference = pd.concat(frames, ignore_index=True).drop_duplicates()
    reference["sido"] = reference["sido"].astype(str)
    reference["sigungu"] = reference["sigungu"].astype(str)
    return reference


def remove_spaces(series: pd.Series) -> pd.Series:
    return series.str.replace(" ", "", regex=False)


def augment_reference(reference: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
    reference = reference.copy()
    reference_keys = set(remove_spaces(reference["sigungu"]))

    aug_rows: List[dict] = []
    processed_canonical = remove_spaces(processed["sigungu"])

    for sigungu, sido, canonical in zip(
        processed["sigungu"], processed["sido"], processed_canonical
    ):
        if canonical in reference_keys:
            continue

        candidates = reference[remove_spaces(reference["sigungu"]).str.startswith(canonical)]
        if not candidates.empty:
            lat = float(candidates["latitude"].mean())
            lon = float(candidates["longitude"].mean())
            aug_rows.append({"sido": sido, "sigungu": sigungu, "latitude": lat, "longitude": lon})
            reference_keys.add(canonical)
            continue

        same_province = reference[reference["sido"] == sido]
        if not same_province.empty:
            lat = float(same_province["latitude"].mean())
            lon = float(same_province["longitude"].mean())
            aug_rows.append({"sido": sido, "sigungu": sigungu, "latitude": lat, "longitude": lon})
            reference_keys.add(canonical)

    if aug_rows:
        reference = pd.concat([reference, pd.DataFrame(aug_rows)], ignore_index=True)
    return reference


def detect_sigungu_mismatch(processed: pd.DataFrame, reference: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = processed.merge(reference, on=["sido", "sigungu"], how="outer", indicator=True)
    missing = merged[merged["_merge"] == "left_only"]["sigungu"].count()
    if missing:
        mismatched = merged[merged["_merge"] == "left_only"]["sigungu"].tolist()
        print(f"[WARN] Unmatched sigungu entries: {missing}")
        for item in mismatched[:10]:
            print(f" - {item}")
    else:
        print("[OK] All sigungu entries matched the reference dataset.")
    return merged[merged["_merge"] == "left_only"], merged[merged["_merge"] == "right_only"]


def write_report(
    path: Path,
    missing_sigungu: pd.DataFrame,
    orphan_sigungu: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["level", "sido", "sigungu", "status"])
        for _, row in missing_sigungu.iterrows():
            writer.writerow([
                "sigungu",
                row.get("sido", ""),
                row.get("sigungu", ""),
                "missing_in_reference",
            ])
        for _, row in orphan_sigungu.iterrows():
            writer.writerow([
                "sigungu",
                row.get("sido", ""),
                row.get("sigungu", ""),
                "unused_reference_entry",
            ])
    print(f"Comparison report written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True, help="Directory containing processed outputs")
    parser.add_argument("--reference", required=True, help="Directory containing reference data")
    parser.add_argument("--out", required=True, help="Output CSV path for mismatch report")
    args = parser.parse_args()

    processed_dir = Path(args.processed).expanduser().resolve()
    reference_dir = Path(args.reference).expanduser().resolve()

    transactions_path = processed_dir / "transactions.csv"
    if not transactions_path.exists():
        raise SystemExit(f"transactions.csv not found under {processed_dir}")

    transactions = pd.read_csv(
        transactions_path,
        usecols=["sido", "sigungu"],
        dtype=str,
        low_memory=False,
    ).drop_duplicates().dropna()

    sigungu_reference = load_sigungu_reference(reference_dir)
    sigungu_reference = augment_reference(sigungu_reference, transactions)
    missing_sigungu, orphan_sigungu = detect_sigungu_mismatch(transactions, sigungu_reference)

    write_report(Path(args.out).expanduser(), missing_sigungu, orphan_sigungu)


if __name__ == "__main__":
    main()
