"""ETL entry point for cherrysona_property datasets.

This script ingests raw 실거래 CSVs, normalises schema, derives metrics,
then exports tidy parquet files that the Streamlit dashboard can query.

Usage examples
--------------
python src/etl_realestate.py --data-root data/raw
python src/etl_realestate.py --data-root . --output-duckdb data/processed/realestate.duckdb
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import csv

import numpy as np
import pandas as pd

try:
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    duckdb = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLMAP: Dict[str, List[str]] = {
    "시도": ["시도", "광역시도", "광역시/도"],
    "시군구": ["시군구", "시/군/구"],
    "읍면동": ["읍면동", "법정동", "동"],
    "법정동코드": ["법정동코드", "법정동 코드", "법정코드"],
    "건축년도": ["건축년도", "건축 연도", "준공년도"],
    "층": ["층", "해당층"],
    "전용면적_㎡": ["전용면적(㎡)", "전용면적", "면적(㎡)"],
    "거래금액_만원": ["거래금액(만원)", "거래금액", "금액(만원)", "매매금액(만원)"],
    "보증금_만원": ["보증금(만원)", "보증금", "보증금금액(만원)", "보증금액(만원)", "보증금"],
    "월세_만원": ["월세(만원)", "월세", "월세금(만원)", "월세금액(만원)"],
    "계약년월": ["계약년월", "년월"],
    "계약일": ["계약일", "일"],
    "해제사유발생일": ["해제사유발생일", "해제일", "해제 발생일"],
}

DATE_IN_NAME = re.compile(r"(\d{8})")
K_CONVERSION = 100  # 월세 → 보증금 환산 계수
AREA_BUCKETS = [0, 30, 60, 85, 135, 10_000]

TRANSACTION_EXPORT_COLUMNS = [
    "src_type",
    "snapshot_date",
    "sido",
    "sigungu",
    "eupmyeondong",
    "법정동코드",
    "건축년도",
    "층",
    "전용면적_㎡",
    "면적_평",
    "거래금액_만원",
    "보증금_만원",
    "월세_만원",
    "계약일자",
    "YYYYMM",
    "연도",
    "월",
    "분기",
    "가격_per_㎡",
    "가격_per_평",
    "환산가_만원",
    "추정매입가_만원",
    "연임대수입_만원",
    "Yield_%",
    "해제사유발생일",
    "취소여부",
    "geo_hash",
    "date_key",
    "building_hash",
]

DEFAULT_SOURCE_DIRS = {
    "apt_trade": "apt_trade",
    "apt_lease": "apt_lease",
    "offi_trade": "offi_trade",
    "offi_lease": "offi_lease",
    "comm_trade": "comm_trade",
    "row_trade": "row_trade",
    "row_lease": "row_lease",
    "det_trade": "det_trade",
    "det_lease": "det_lease",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw real-estate CSVs")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Directory that contains the segmented folders (apt_trade, ...).",
    )
    parser.add_argument(
        "--legacy-glob",
        action="store_true",
        help="Also scan the data root itself for *_실거래가_YYYYMMDD.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to write parquet outputs to.",
    )
    parser.add_argument(
        "--output-duckdb",
        type=Path,
        default=None,
        help="Optional path to a DuckDB database file for aggregated tables.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files per segment (for smoke tests).",
    )
    return parser.parse_args()


@dataclass
class SourceFile:
    path: Path
    src_type: str
    snapshot_date: Optional[pd.Timestamp]


def list_source_files(data_root: Path, legacy: bool, max_files: Optional[int]) -> List[SourceFile]:
    sources: List[SourceFile] = []
    for src_type, folder in DEFAULT_SOURCE_DIRS.items():
        base = data_root / folder
        if not base.exists():
            continue
        files = sorted(base.glob("*.csv"))
        if max_files:
            files = files[:max_files]
        for fp in files:
            sources.append(SourceFile(fp, src_type, detect_snapshot_from_filename(fp.name)))

    if legacy:
        for fp in sorted(data_root.glob("*_실거래가_*.csv")):
            guess = guess_src_type(fp.name)
            sources.append(SourceFile(fp, guess, detect_snapshot_from_filename(fp.name)))

    return sources


def guess_src_type(filename: str) -> str:
    mapping = {
        "아파트(매매)": "apt_trade",
        "아파트(전월세)": "apt_lease",
        "오피스텔(매매)": "offi_trade",
        "오피스텔(전월세)": "offi_lease",
        "상업업무용(매매)": "comm_trade",
        "연립다세대(매매)": "row_trade",
        "연립다세대(전월세)": "row_lease",
        "단독다가구(매매)": "det_trade",
        "단독다가구(전월세)": "det_lease",
    }
    for key, value in mapping.items():
        if key in filename:
            return value
    return "unknown"


def detect_snapshot_from_filename(name: str) -> Optional[pd.Timestamp]:
    match = DATE_IN_NAME.search(name)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
    except (ValueError, TypeError):
        return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    reverse_map = {alias: std for std, aliases in COLMAP.items() for alias in aliases}
    newcols = {col: reverse_map.get(col, col) for col in df.columns}
    return df.rename(columns=newcols)


def to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(r"[^0-9\.-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_contract_date(row: pd.Series) -> Optional[pd.Timestamp]:
    yyyymm = row.get("계약년월")
    day = row.get("계약일")
    if pd.isna(yyyymm):
        return None
    try:
        yyyymm = int(yyyymm)
    except (TypeError, ValueError):
        return None
    year, month = divmod(yyyymm, 100)
    day = int(day) if pd.notna(day) else 1
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def winsorize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    lower, upper = series.quantile([0.01, 0.99])
    return series.clip(lower=lower, upper=upper)


def derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    required_numeric_defaults = {
        "보증금_만원": np.nan,
        "월세_만원": np.nan,
        "거래금액_만원": np.nan,
        "전용면적_㎡": np.nan,
        "계약년월": np.nan,
        "계약일": np.nan,
    }
    for col, default in required_numeric_defaults.items():
        if col not in df.columns:
            df[col] = default

    numerics = [
        "층",
        "건축년도",
        "전용면적_㎡",
        "거래금액_만원",
        "보증금_만원",
        "월세_만원",
        "계약년월",
        "계약일",
    ]
    for col in numerics:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    if {"계약년월", "계약일"}.issubset(df.columns):
        df["계약일자"] = df.apply(build_contract_date, axis=1)
        df["YYYYMM"] = df["계약일자"].dt.strftime("%Y%m").astype("Int64")
        df["연도"] = df["계약일자"].dt.year.astype("Int64")
        df["월"] = df["계약일자"].dt.month.astype("Int64")
        df["분기"] = df["계약일자"].dt.quarter.astype("Int64")

    if "전용면적_㎡" in df.columns:
        df["면적_평"] = df["전용면적_㎡"] * 0.3025

    if {"거래금액_만원", "전용면적_㎡"}.issubset(df.columns):
        df["가격_per_㎡"] = (df["거래금액_만원"] / df["전용면적_㎡"]).replace([np.inf, -np.inf], np.nan)
        df["가격_per_평"] = (df["거래금액_만원"] / df["면적_평"]).replace([np.inf, -np.inf], np.nan)

    if "해제사유발생일" in df.columns:
        df["해제사유발생일"] = pd.to_datetime(df["해제사유발생일"], errors="coerce")
        df["취소여부"] = df["해제사유발생일"].notna().astype(int)
    else:
        df["취소여부"] = 0

    if "보증금_만원" in df.columns:
        df["보증금_만원"] = df["보증금_만원"].fillna(0)
    if "월세_만원" in df.columns:
        df["월세_만원"] = df["월세_만원"].fillna(0)
    if "거래금액_만원" in df.columns:
        df["거래금액_만원"] = df["거래금액_만원"].where(df["거래금액_만원"].notna(), np.nan)
    if "보증금_만원" in df.columns or "월세_만원" in df.columns:
        df["환산가_만원"] = df["보증금_만원"] + df["월세_만원"] * K_CONVERSION
        df["추정매입가_만원"] = np.where(
            df["거래금액_만원"].notna(),
            df["거래금액_만원"],
            df["환산가_만원"],
        )
        df["연임대수입_만원"] = df["월세_만원"] * 12
        df["Yield_%"] = (df["연임대수입_만원"] / df["추정매입가_만원"]) * 100

    if {"전용면적_㎡", "거래금액_만원"}.issubset(df.columns):
        bucketed = pd.cut(df["전용면적_㎡"], bins=AREA_BUCKETS)
        df["거래금액_만원"] = (
            df.groupby(bucketed)["거래금액_만원"].transform(winsorize)
        )

    return df


def normalize_frame(raw_df: pd.DataFrame, src_type: str) -> pd.DataFrame:
    df = standardize_columns(raw_df)
    if "시도" in df.columns:
        df["sido"] = df["시도"]
    if "시군구" in df.columns:
        df["sigungu"] = df["시군구"]
    if "읍면동" in df.columns:
        df["eupmyeondong"] = df["읍면동"]

    for col in ("sido", "sigungu", "eupmyeondong"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\[\]\"]", "", regex=True)
                .str.replace(r"[/|]", ",", regex=True)
                .str.split(",")
                .str[0]
                .str.strip()
            )

    if "sigungu" in df.columns:
        tokens = df["sigungu"].str.split()

        def _safe_join(parts, upto):
            if not parts:
                return "미상"
            subset = parts[:upto]
            if not subset:
                return parts[0]
            return " ".join(subset)

        def _leftover(parts):
            if not parts or len(parts) <= 2:
                return ""
            return " ".join(parts[2:])

        df["sido"] = tokens.apply(lambda parts: parts[0] if parts else "미상")
        df["sigungu"] = tokens.apply(lambda parts: _safe_join(parts, 2))
        leftover = tokens.apply(_leftover)

        if "eupmyeondong" in df.columns:
            df["eupmyeondong"] = df["eupmyeondong"].astype(str).str.strip()
            df["eupmyeondong"] = df["eupmyeondong"].replace({"nan": "", "NaN": ""})
            df["eupmyeondong"] = df["eupmyeondong"].mask(df["eupmyeondong"].eq(""), leftover)
            df["eupmyeondong"] = df["eupmyeondong"].where(leftover.eq(""), leftover)
        else:
            df["eupmyeondong"] = leftover

        df["eupmyeondong"] = df["eupmyeondong"].replace("", "미상").fillna("미상")

    df["src_type"] = src_type
    df = derive_fields(df)
    return df


def detect_header_index(path: Path, encoding: str) -> int:
    with path.open("r", encoding=encoding, errors="ignore") as handle:
        for idx, line in enumerate(handle):
            stripped = line.strip()
            if ("NO" in stripped or "번호" in stripped) and "계약년월" in stripped:
                return idx
    return 0


def read_source_dataframe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "cp949", "euc-kr"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            header_idx = detect_header_index(path, enc)
            return pd.read_csv(
                path,
                encoding=enc,
                skiprows=header_idx,
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError:
            try:
                return pd.read_csv(
                    path,
                    encoding=enc,
                    skiprows=header_idx,
                    engine="python",
                    quoting=csv.QUOTE_MINIMAL,
                    sep=",",
                    on_bad_lines="skip",
                )
            except Exception as exc:  # pragma: no cover - fallback
                last_error = exc
                continue
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(f"Failed to load {path}: {last_error}")


def load_sources(sources: Iterable[SourceFile]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for source in sources:
        try:
            df = read_source_dataframe(source.path)
            df = normalize_frame(df, source.src_type)
            df["snapshot_date"] = source.snapshot_date
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] failed to load {source.path}: {exc}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    subset = [
        col
        for col in [
            "시도",
            "시군구",
            "읍면동",
            "법정동코드",
            "계약일자",
            "전용면적_㎡",
            "거래금액_만원",
            "src_type",
        ]
        if col in combined.columns
    ]
    if subset:
        combined = combined.drop_duplicates(subset=subset)
    return combined


def build_geo_hash(row: pd.Series) -> str:
    key = "|".join(
        str(row.get(col, "") or "")
        for col in ("시도", "시군구", "읍면동", "법정동코드")
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def add_dimension_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "계약일자" not in df.columns:
        df["계약일자"] = pd.NaT
    if "건축년도" not in df.columns:
        df["건축년도"] = pd.NA
    if "층" not in df.columns:
        df["층"] = pd.NA
    if "시도" in df.columns:
        df["sido"] = df["시도"]
    if "시군구" in df.columns:
        df["sigungu"] = df["시군구"]
    if "읍면동" in df.columns:
        df["eupmyeondong"] = df["읍면동"]
    df["geo_hash"] = df.apply(build_geo_hash, axis=1)
    df["date_key"] = df["계약일자"].dt.strftime("%Y%m%d").astype("Int64")
    building_attrs = df[["건축년도", "층"]].copy()
    building_attrs = building_attrs.fillna(-1)
    building_attrs = building_attrs.astype(int)
    df["building_hash"] = (
        building_attrs.astype(str)
        .agg("|".join, axis=1)
        .map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    )
    return df


def compute_monthly_basics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    region_cols = {
        "sido": "sido" if "sido" in df.columns else "시도",
        "sigungu": "sigungu" if "sigungu" in df.columns else "시군구",
        "eupmyeondong": "eupmyeondong" if "eupmyeondong" in df.columns else "읍면동",
    }
    for key, col in region_cols.items():
        if col not in df.columns:
            df[key] = "미상"
            region_cols[key] = key
        elif key != col:
            df[key] = df[col]
            region_cols[key] = key

    metrics = (
        df.dropna(subset=["YYYYMM"])
        .groupby(["src_type", "sido", "sigungu", "eupmyeondong", "YYYYMM"], dropna=False)
        .agg(
            txn_cnt=("거래금액_만원", "count"),
            avg_p_per_m2=("가격_per_㎡", "mean"),
            std_p_per_m2=("가격_per_㎡", "std"),
            avg_yield_pct=("Yield_%", "mean"),
            cancel_rate=("취소여부", "mean"),
        )
        .reset_index()
    )
    metrics["std_p_per_m2"].fillna(0, inplace=True)
    metrics["cancel_rate"].fillna(0, inplace=True)
    return metrics


def compute_momentum(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    monthly = monthly.sort_values(["src_type", "sido", "sigungu", "eupmyeondong", "YYYYMM"])
    monthly["avg_p_per_m2_lag3"] = (
        monthly.groupby(["src_type", "sido", "sigungu", "eupmyeondong"])["avg_p_per_m2"]
        .shift(3)
    )
    monthly["mom_3m"] = (
        monthly["avg_p_per_m2"] / monthly["avg_p_per_m2_lag3"] - 1
    )
    monthly.loc[monthly["avg_p_per_m2_lag3"].isna(), "mom_3m"] = 0
    monthly.loc[monthly["mom_3m"].replace([np.inf, -np.inf], np.nan).isna(), "mom_3m"] = 0
    return monthly


def compute_volatility(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    vol = monthly.copy()
    vol["cv_p_per_m2"] = vol["std_p_per_m2"] / vol["avg_p_per_m2"].replace(0, np.nan)
    vol["cv_p_per_m2"] = vol["cv_p_per_m2"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return vol


def persist_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_to_write = sanitize_for_parquet(df)
    if duckdb is not None:
        con = duckdb.connect()
        try:
            con.register("df_temp", df_to_write.reset_index(drop=index))
            con.execute(f"COPY df_temp TO '{path.as_posix()}' (FORMAT PARQUET)")
        finally:
            con.close()
    else:
        df_to_write.to_parquet(path, index=index)


def persist_transactions_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    obj_cols = out.select_dtypes(include="object").columns
    for col in obj_cols:
        series = out[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().sum() >= series.notna().sum() * 0.9:
            out[col] = numeric_series
        else:
            out[col] = series.where(series.notna(), None).astype("string")
    return out


def persist_duckdb(
    db_path: Path,
    fact_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    momentum_df: pd.DataFrame,
    volatility_df: pd.DataFrame,
) -> None:
    if duckdb is None:
        raise RuntimeError("duckdb is not installed; pip install duckdb to use this option")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.register("fact_df", fact_df)
    con.register("monthly_df", monthly_df)
    con.register("momentum_df", momentum_df)
    con.register("volatility_df", volatility_df)
    con.execute("CREATE OR REPLACE TABLE fact_transactions AS SELECT * FROM fact_df")
    con.execute("CREATE OR REPLACE TABLE vw_monthly_basics AS SELECT * FROM monthly_df")
    con.execute("CREATE OR REPLACE TABLE vw_monthly_momentum AS SELECT * FROM momentum_df")
    con.execute("CREATE OR REPLACE TABLE vw_monthly_volatility AS SELECT * FROM volatility_df")
    con.close()


def main() -> None:
    args = parse_args()
    sources = list_source_files(args.data_root, legacy=args.legacy_glob, max_files=args.max_files)

    if not sources:
        print("No source files discovered; check data-root or enable --legacy-glob")
        return

    raw_df = load_sources(sources)
    if raw_df.empty:
        print("Loaded zero rows from CSVs")
        return

    tidy = add_dimension_keys(raw_df)

    monthly = compute_monthly_basics(tidy)
    momentum = compute_momentum(monthly.copy())
    volatility = compute_volatility(monthly.copy())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_cols = [c for c in TRANSACTION_EXPORT_COLUMNS if c in tidy.columns]
    persist_transactions_csv(tidy[export_cols], args.output_dir / "transactions.csv")
    persist_parquet(monthly, args.output_dir / "monthly_basics.parquet")
    persist_parquet(momentum, args.output_dir / "monthly_momentum.parquet")
    persist_parquet(volatility, args.output_dir / "monthly_volatility.parquet")

    if args.output_duckdb:
        persist_duckdb(args.output_duckdb, tidy, monthly, momentum, volatility)

    print(
        f"Processed {len(tidy):,} rows from {len(sources)} files. "
        f"Outputs stored in {args.output_dir}."
    )


if __name__ == "__main__":
    main()
