# -*- coding: utf-8 -*-
from __future__ import annotations

# Streamlit Unified Real-Estate Dashboard
# - 데이터: ETL 산출물 (Parquet / CSV)
# - 기능: dashboard.py (KPI, VC, RO, Persona, Explorer) + streamlit_realestate_app.py (Top5 Insights)

import os
from pathlib import Path
import math
from datetime import datetime
from typing import Callable, Optional, Set, Union, List, Tuple

import altair as alt
import pandas as pd, numpy as np
from pandas.tseries.offsets import MonthEnd
import pydeck as pdk
import streamlit as st

try:  # optional dependency for DuckDB backend
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    duckdb = None

try:  # optional dependency for ML 추천 기능
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.tree import DecisionTreeRegressor, plot_tree  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    KMeans = None
    StandardScaler = None
    RandomForestRegressor = None
    DecisionTreeRegressor = None
    plot_tree = None

try:  # optional dependency for Decision Tree 시각화
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while avoiding divide-by-zero and infinite results."""

    denom = denominator.where(denominator > 0)
    result = numerator.divide(denom)
    return result.replace([np.inf, -np.inf], np.nan)


def _missing_dependencies(deps: dict[str, object]) -> list[str]:
    return [name for name, module in deps.items() if module is None]


GROUP_COL = "시군구"
SIGUNGU_ALIASES = ("시군구", "sigungu")
LOG_STORAGE_KEY = "__sidebar_logs__"

NEW_BUILD_YEARS = 5
OLD_BUILD_YEARS = 20
MIN_NEW_SAMPLE_COUNT = 10
MIN_OLD_SAMPLE_COUNT = 10
MIN_OLD_PRICE_PER_M2 = 1.0


def push_log(message: str, level: str = "info") -> None:
    logs = st.session_state.setdefault(LOG_STORAGE_KEY, [])
    logs.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level.upper(),
        "message": message,
    })


def warn_missing_resource(path: Union[str, Path], label: str) -> None:
    """Show a single warning per missing resource to avoid log spam."""

    key = f"missing::{Path(path)}"
    notified = st.session_state.setdefault("__missing_resource__", set())
    if key in notified:
        return
    notified.add(key)
    message = f"{label} 데이터를 찾을 수 없습니다. 경로를 확인하세요: {path}"
    push_log(message, "warning")
    st.warning(message, icon="⚠️")


def _to_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if stripped == "":
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value, decimals: int = 1, suffix: str = "") -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number:.{decimals}f}{suffix}"


def simplify_region_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        return "기타"
    parts = name.split()
    if not parts:
        return name
    if parts[0] == "서울특별시":
        return "서울특별시"
    if len(parts) >= 2 and (parts[1].endswith("시") or parts[1].endswith("군")):
        return parts[1]
    return parts[0]


def normalize_sigungu_label(name: object) -> str:
    """Return the "sido sigungu" pair so boundary datasets align."""
    if not isinstance(name, str):
        return ""
    stripped = name.strip()
    if not stripped:
        return ""
    tokens = stripped.split()
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0]

def aggregate_regions_for_chart(df: pd.DataFrame, aggregate: bool) -> pd.DataFrame:
    if not aggregate or df.empty or GROUP_COL not in df.columns:
        return df
    work = df.copy()
    work[GROUP_COL] = work[GROUP_COL].apply(simplify_region_name)
    group_cols = [GROUP_COL]
    if "YYYYMM" in work.columns:
        group_cols.append("YYYYMM")
    numeric_cols = work.select_dtypes(include=[np.number]).columns.difference(group_cols)
    if not numeric_cols.any():
        return work[group_cols].drop_duplicates()
    aggregated = work.groupby(group_cols, as_index=False)[list(numeric_cols)].mean()
    return aggregated


def aggregate_summary_by_region(df: pd.DataFrame, aggregate: bool) -> pd.DataFrame:
    if not aggregate or df.empty or GROUP_COL not in df.columns:
        return df
    work = df.copy()
    work[GROUP_COL] = work[GROUP_COL].apply(simplify_region_name)
    numeric_cols = work.select_dtypes(include=[np.number]).columns.difference([GROUP_COL])
    aggregated = work.groupby(GROUP_COL, as_index=False)[list(numeric_cols)].mean()
    for col in work.columns:
        if col not in aggregated.columns and col != GROUP_COL:
            aggregated[col] = (
                work.groupby(GROUP_COL)[col].first().reindex(aggregated[GROUP_COL]).reset_index(drop=True)
            )
    return aggregated


def resolve_sigungu(df: pd.DataFrame) -> Optional[str]:
    for col in SIGUNGU_ALIASES:
        if col in df.columns:
            return col
    return None


def ensure_sigungu_named(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where the 시군구 컬럼명이 통일되어 있음."""
    sig_col = resolve_sigungu(df)
    if sig_col is None:
        return df.copy()
    if sig_col == GROUP_COL:
        return df.copy()
    out = df.copy()
    out.rename(columns={sig_col: GROUP_COL}, inplace=True)
    return out

# =============================
# 데이터 경로
# =============================
DATA_DIR = Path("data/processed")
MONTHLY_PATH = DATA_DIR / "monthly_basics.parquet"
MONTHLY_PATH_COMBINED = DATA_DIR / "monthly_basics_combined.parquet"
MOMENTUM_PATH = DATA_DIR / "monthly_momentum.parquet"
MOMENTUM_PATH_COMBINED = DATA_DIR / "monthly_momentum_combined.parquet"
VOLATILITY_PATH = DATA_DIR / "monthly_volatility.parquet"
VOLATILITY_PATH_COMBINED = DATA_DIR / "monthly_volatility_combined.parquet"
FACT_PATH = DATA_DIR / "transactions.csv"
FACT_PATH_COMBINED = DATA_DIR / "transactions_combined.csv"
CENTROID_PARQUET = Path("data/reference/sigungu_centroids.parquet")
CENTROID_CSV = Path("data/reference/sigungu_centroids.csv")
POLICY_PATH = Path("data/reference/policy_risk.csv")
BOUNDARY_PATH = Path("data/raw/HangJeongDong_ver20250401.geojson")
LIFESTYLE_PATH = Path("data/reference/lifestyle_scores.csv")
DUCKDB_ENV_PATH = os.environ.get("CHERRYSONA_DUCKDB", "")

DUCKDB_TABLES = {
    "monthly_basics": "vw_monthly_basics",
    "monthly_momentum": "vw_monthly_momentum",
    "monthly_volatility": "vw_monthly_volatility",
    "transactions": "fact_transactions",
}

# =============================
# 로딩 함수
# =============================


def _can_use_duckdb(path: Optional[str]) -> bool:
    return bool(path and duckdb is not None and Path(path).exists())


@st.cache_resource(show_spinner=False)
def get_duckdb_connection(path: str) -> Optional[object]:
    if not _can_use_duckdb(path):  # pragma: no cover - guarded by caller
        return None
    try:
        return duckdb.connect(path, read_only=True)  # type: ignore[arg-type]
    except Exception:
        return None


def fetch_duckdb_table(path: str, table: str) -> pd.DataFrame:
    if not _can_use_duckdb(path):
        return pd.DataFrame()
    conn = get_duckdb_connection(path)
    if conn is None:
        push_log(f"DuckDB 연결 실패: {path}", "error")
        return pd.DataFrame()
    try:
        return conn.execute(f"SELECT * FROM {table}").fetch_df()
    except Exception as exc:
        push_log(f"DuckDB 쿼리 실패({table}): {exc}", "error")
        return pd.DataFrame()


def _resolve_processed_path(primary: Path, combined: Path, prefer_combined: bool) -> tuple[Path, bool]:
    if prefer_combined and combined.exists():
        return combined, True
    if prefer_combined and not combined.exists() and primary.exists():
        return primary, False
    if primary.exists():
        return primary, False
    if combined.exists():
        return combined, True
    return (combined if prefer_combined else primary, False)


@st.cache_data(show_spinner=False, ttl=3600)
def load_monthly_basics(
    backend: str = "Parquet",
    duckdb_path: Optional[str] = None,
    *,
    prefer_combined: bool = False,
):
    if backend == "DuckDB" and duckdb_path:
        return fetch_duckdb_table(duckdb_path, DUCKDB_TABLES["monthly_basics"])
    path, used_combined = _resolve_processed_path(MONTHLY_PATH, MONTHLY_PATH_COMBINED, prefer_combined)
    if not path.exists():
        return pd.DataFrame()
    if prefer_combined and not used_combined:
        push_log("Combined 월간 기본 지표가 없어 기본 데이터를 사용합니다.", "warning")
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=3600)
def load_momentum(
    backend: str = "Parquet",
    duckdb_path: Optional[str] = None,
    *,
    prefer_combined: bool = False,
):
    if backend == "DuckDB" and duckdb_path:
        return fetch_duckdb_table(duckdb_path, DUCKDB_TABLES["monthly_momentum"])
    path, used_combined = _resolve_processed_path(MOMENTUM_PATH, MOMENTUM_PATH_COMBINED, prefer_combined)
    if not path.exists():
        return pd.DataFrame()
    if prefer_combined and not used_combined:
        push_log("Combined 모멘텀 지표가 없어 기본 데이터를 사용합니다.", "warning")
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=3600)
def load_volatility(
    backend: str = "Parquet",
    duckdb_path: Optional[str] = None,
    *,
    prefer_combined: bool = False,
):
    if backend == "DuckDB" and duckdb_path:
        return fetch_duckdb_table(duckdb_path, DUCKDB_TABLES["monthly_volatility"])
    path, used_combined = _resolve_processed_path(VOLATILITY_PATH, VOLATILITY_PATH_COMBINED, prefer_combined)
    if not path.exists():
        return pd.DataFrame()
    if prefer_combined and not used_combined:
        push_log("Combined 변동성 지표가 없어 기본 데이터를 사용합니다.", "warning")
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=600)
def load_transactions(
    backend: str = "Parquet",
    duckdb_path: Optional[str] = None,
    *,
    prefer_combined: bool = False,
):
    if backend == "DuckDB" and duckdb_path:
        return fetch_duckdb_table(duckdb_path, DUCKDB_TABLES["transactions"])
    path, used_combined = _resolve_processed_path(FACT_PATH, FACT_PATH_COMBINED, prefer_combined)
    if not path.exists():
        return pd.DataFrame()
    if prefer_combined and not used_combined:
        push_log("Combined 거래 원장을 찾을 수 없어 기본 데이터를 사용합니다.", "warning")
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        push_log(f"거래 원장 로딩 실패({path.name}): {exc}", "error")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=86400)
def load_geo_centroids() -> pd.DataFrame:
    if CENTROID_PARQUET.exists():
        df = pd.read_parquet(CENTROID_PARQUET)
    elif CENTROID_CSV.exists():
        df = pd.read_csv(CENTROID_CSV)
    else:
        return pd.DataFrame()

    rename_map = {}
    if "sigungu" in df.columns and GROUP_COL not in df.columns:
        rename_map["sigungu"] = GROUP_COL
    if "lat" in df.columns and "latitude" not in df.columns:
        rename_map["lat"] = "latitude"
    if "lon" in df.columns and "longitude" not in df.columns:
        rename_map["lon"] = "longitude"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {GROUP_COL, "latitude", "longitude"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df[[GROUP_COL, "latitude", "longitude"]].dropna()


@st.cache_data(show_spinner=False, ttl=3600)
def load_policy_risk() -> pd.DataFrame:
    if not POLICY_PATH.exists():
        return pd.DataFrame(columns=["YYYYMM", GROUP_COL, "interest_rate", "regulation_flag", "policy_comment"])
    df = pd.read_csv(POLICY_PATH)
    rename_map = {}
    if "sigungu" in df.columns and GROUP_COL not in df.columns:
        rename_map["sigungu"] = GROUP_COL
    if rename_map:
        df = df.rename(columns=rename_map)
    required = {GROUP_COL, "YYYYMM"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["YYYYMM", GROUP_COL, "interest_rate", "regulation_flag", "policy_comment"])
    df["YYYYMM"] = pd.to_numeric(df["YYYYMM"], errors="coerce").astype("Int64")
    if "interest_rate" in df.columns:
        df["interest_rate"] = pd.to_numeric(df["interest_rate"], errors="coerce")
    if "regulation_flag" in df.columns:
        df["regulation_flag"] = df["regulation_flag"].astype(str)
    return df


@st.cache_data(show_spinner=False, ttl=86400)
def load_lifestyle_scores() -> pd.DataFrame:
    if not LIFESTYLE_PATH.exists():
        return pd.DataFrame(columns=[GROUP_COL, "school_score", "amenity_score", "lifestyle_comment"])

    df = pd.read_csv(LIFESTYLE_PATH)
    rename_map = {}
    if "sigungu" in df.columns and GROUP_COL not in df.columns:
        rename_map["sigungu"] = GROUP_COL
    if rename_map:
        df = df.rename(columns=rename_map)

    if GROUP_COL not in df.columns:
        return pd.DataFrame(columns=[GROUP_COL, "school_score", "amenity_score", "lifestyle_comment"])

    numeric_cols = ["school_score", "amenity_score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    columns = [GROUP_COL] + [c for c in ["school_score", "amenity_score", "lifestyle_comment"] if c in df.columns]
    deduped = df[columns].drop_duplicates(subset=[GROUP_COL])
    return deduped


@st.cache_resource(show_spinner=False)
def load_sigungu_boundaries() -> dict:
    if not BOUNDARY_PATH.exists():
        return {}
    try:
        import json
    except ImportError:  # pragma: no cover
        return {}

    with BOUNDARY_PATH.open("r", encoding="utf-8") as fh:
        geo = json.load(fh)

    boundaries: dict[str, dict] = {}
    for feature in geo.get("features", []):
        props = feature.get("properties", {})
        sido = props.get("sidonm") or props.get("sido")
        sgg = props.get("sggnm") or props.get("sgg")
        if not sido or not sgg:
            continue
        sigungu_name = f"{sido} {sgg}".strip()
        if sigungu_name not in boundaries:
            boundaries[sigungu_name] = feature
        else:
            # Merge geometries by accumulating coordinates list
            existing = boundaries[sigungu_name]["geometry"]
            incoming = feature.get("geometry", {})
            if existing.get("type") == "MultiPolygon" and incoming.get("type") == "MultiPolygon":
                existing.setdefault("coordinates", []).extend(incoming.get("coordinates", []))
            elif existing.get("type") == "Polygon" and incoming.get("type") == "Polygon":
                existing.setdefault("coordinates", []).extend(incoming.get("coordinates", []))
            # Skip if geometry types differ; more robust merge would require shapely
    return boundaries


def filter_by_src(df: pd.DataFrame, src: str) -> pd.DataFrame:
    if df.empty or "src_type" not in df.columns:
        return df.copy()
    return df[df["src_type"] == src].copy()


CONTRACT_SRC_FALLBACK = {
    "apt": "apt_lease",
    "offi": "offi_lease",
    "row": "row_lease",
    "det": "det_lease",
}


def resolve_contract_source(src: str) -> str:
    if not isinstance(src, str) or "_" not in src:
        return src
    asset, variant = src.split("_", 1)
    if variant == "lease":
        return src
    fallback = CONTRACT_SRC_FALLBACK.get(asset)
    return fallback or src


def ensure_yyyymm_numeric(df: pd.DataFrame, col: str = "YYYYMM") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df.copy()
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def filter_recent_months(df: pd.DataFrame, months: Optional[int], col: str = "YYYYMM") -> pd.DataFrame:
    if df.empty or col not in df.columns or not months or months <= 0:
        return df.copy()
    numeric = pd.to_numeric(df[col], errors="coerce")
    series = numeric.dropna()
    if series.empty:
        return df.copy()
    periods = pd.PeriodIndex(series.astype(int).astype(str), freq="M")
    latest = periods.max()
    threshold = latest - (months - 1)
    keep_index = series.index[periods >= threshold]
    return df.loc[keep_index].copy()


def filter_by_regions(df: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = ensure_sigungu_named(df)
    if GROUP_COL not in work.columns or not regions:
        return work
    cleaned = [r for r in regions if r != "전체"]
    if not cleaned:
        return work
    return work[work[GROUP_COL].isin(cleaned)].copy()


def filter_by_neighborhoods(df: pd.DataFrame, neighborhoods: list[str]) -> pd.DataFrame:
    if df.empty or "eupmyeondong" not in df.columns or not neighborhoods:
        return df.copy()
    cleaned = [item for item in neighborhoods if item and item != "전체"]
    if not cleaned:
        return df.copy()
    return df[df["eupmyeondong"].isin(cleaned)].copy()


def filter_by_properties(df: pd.DataFrame, property_keys: list[str]) -> pd.DataFrame:
    if df.empty or "property_key" not in df.columns or not property_keys:
        return df.copy()
    cleaned = [key for key in property_keys if key]
    if not cleaned:
        return df.copy()
    return df[df["property_key"].isin(cleaned)].copy()


def format_property_label(row: pd.Series) -> str:
    sigungu = str(row.get("sigungu") or "").strip()
    dong = str(row.get("eupmyeondong") or "").strip()

    def _format_lot(value: object) -> str:
        if pd.isna(value):
            return ""
        try:
            number = int(float(value))
            return str(number)
        except (TypeError, ValueError):
            return str(value).strip()

    lot_main = _format_lot(row.get("lot_main"))
    lot_sub = _format_lot(row.get("lot_sub"))
    lot = lot_main
    if lot and lot_sub and lot_sub not in {"0", "0.0"}:
        lot = f"{lot}-{lot_sub}"

    parts = [part for part in (sigungu, dong, lot) if part]
    if not parts:
        fallback = str(row.get("property_key") or "").strip()
        return fallback
    return " ".join(parts)


def prepare_property_summary_table(
    fact_df: pd.DataFrame,
    rent_df: pd.DataFrame,
    annual_rate: float,
    label_lookup: dict[str, str],
) -> pd.DataFrame:
    if fact_df.empty or "property_key" not in fact_df.columns:
        return pd.DataFrame()

    work = fact_df.copy()
    work["계약일자"] = pd.to_datetime(work.get("계약일자"), errors="coerce")

    agg = (
        work.groupby("property_key")
        .agg(
            sigungu=("sigungu", "first"),
            eupmyeondong=("eupmyeondong", "first"),
            latest_contract=("계약일자", "max"),
            txn_cnt=("가격_per_㎡", "count"),
            avg_price_per_m2=("가격_per_㎡", "mean"),
            avg_area=("전용면적_㎡", "mean"),
            avg_floor=("층", "mean"),
            new_ratio=("is_new", "mean"),
            old_ratio=("is_old", "mean"),
        )
        .reset_index()
    )

    if agg.empty:
        return pd.DataFrame()

    rent_summary = pd.DataFrame()
    if not rent_df.empty and "property_key" in rent_df.columns:
        rent_summary = compute_rent_metrics(rent_df, annual_rate, group_field="property_key")
        if not rent_summary.empty:
            rent_summary = rent_summary.rename(
                columns={
                    "rent_gap_mean": "rent_gap_mean",
                    "avg_monthly_rent": "avg_monthly_rent",
                    "rent_undervalue_rate": "rent_undervalue_rate",
                }
            )

    if not rent_summary.empty:
        agg = agg.merge(rent_summary, on="property_key", how="left")

    agg["매물"] = agg["property_key"].map(label_lookup).fillna(agg["property_key"])
    agg["최신 계약일"] = agg["latest_contract"].dt.strftime("%Y-%m-%d").fillna("-")
    agg["거래 건수"] = agg["txn_cnt"].astype(int)
    agg["평균 ㎡당가"] = agg["avg_price_per_m2"].apply(lambda v: _format_float(v, 1))
    agg["평균 전용(㎡)"] = agg["avg_area"].apply(lambda v: _format_float(v, 1))
    agg["평균 층"] = agg["avg_floor"].apply(lambda v: _format_float(v, 1))
    agg["신축 비율"] = agg["new_ratio"].apply(lambda v: format_percent(v))
    agg["구축 비율"] = agg["old_ratio"].apply(lambda v: format_percent(v))

    if "rent_gap_mean" in agg.columns:
        agg["Rent Gap(만원)"] = agg["rent_gap_mean"].apply(lambda v: _format_float(v, 1))
    if "avg_monthly_rent" in agg.columns:
        agg["평균 월세(만원)"] = agg["avg_monthly_rent"].apply(lambda v: _format_float(v, 1))
    if "rent_undervalue_rate" in agg.columns:
        agg["월세 저평가율"] = agg["rent_undervalue_rate"].apply(format_percent)

    display_cols = [
        "매물",
        "sigungu",
        "eupmyeondong",
        "최신 계약일",
        "거래 건수",
        "평균 ㎡당가",
        "평균 전용(㎡)",
        "평균 층",
        "신축 비율",
        "구축 비율",
    ]

    optional_cols = ["Rent Gap(만원)", "평균 월세(만원)", "월세 저평가율"]
    display_cols.extend([col for col in optional_cols if col in agg.columns])

    rename_map = {
        "sigungu": "시군구",
        "eupmyeondong": "읍·동",
    }
    agg = agg.rename(columns=rename_map)

    return agg[display_cols]


@st.cache_data(show_spinner=False, ttl=900, max_entries=32)
def compute_property_recommendations(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty or "property_key" not in trade_df.columns:
        return pd.DataFrame()

    work = trade_df.copy()
    for column in ["가격_per_㎡", "Yield_%", "전용면적_㎡", "층", "거래금액_만원"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    if "계약일자" in work.columns:
        work["계약일자"] = pd.to_datetime(work["계약일자"], errors="coerce")

    agg_mapping: dict[str, tuple[str, str]] = {}
    agg_mapping["sigungu"] = (GROUP_COL, "first") if GROUP_COL in work.columns else ("property_key", "first")
    if "eupmyeondong" in work.columns:
        agg_mapping["eupmyeondong"] = ("eupmyeondong", "first")
    if "lot_main" in work.columns:
        agg_mapping["lot_main"] = ("lot_main", "first")
    if "lot_sub" in work.columns:
        agg_mapping["lot_sub"] = ("lot_sub", "first")
    if "계약일자" in work.columns:
        agg_mapping["latest_contract"] = ("계약일자", "max")
    if "가격_per_㎡" in work.columns:
        agg_mapping["avg_price_per_m2"] = ("가격_per_㎡", "mean")
    if "거래금액_만원" in work.columns:
        agg_mapping["avg_trade_amount"] = ("거래금액_만원", "mean")
    if "Yield_%" in work.columns:
        agg_mapping["avg_yield_pct"] = ("Yield_%", "mean")
    if "전용면적_㎡" in work.columns:
        agg_mapping["avg_area_sqm"] = ("전용면적_㎡", "mean")
    if "층" in work.columns:
        agg_mapping["avg_floor"] = ("층", "mean")
    if "is_new" in work.columns:
        agg_mapping["new_ratio"] = ("is_new", "mean")

    grouped = work.groupby("property_key").agg(**agg_mapping)
    grouped["transaction_count"] = work.groupby("property_key")["property_key"].size()

    for col in ["avg_price_per_m2", "avg_yield_pct"]:
        if col not in grouped.columns:
            grouped[col] = np.nan

    def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            scaled = pd.Series(0.5, index=series.index)
        else:
            min_val = valid.min()
            max_val = valid.max()
            if np.isclose(min_val, max_val, equal_nan=True):
                scaled = pd.Series(0.5, index=series.index)
            else:
                scaled = (numeric - min_val) / (max_val - min_val)
                scaled = scaled.clip(0.0, 1.0)
        if invert:
            scaled = 1.0 - scaled
        return scaled

    score = pd.Series(0.0, index=grouped.index, dtype=float)
    if "avg_price_per_m2" in grouped.columns:
        price_component = _minmax(grouped["avg_price_per_m2"], invert=True)
        score = score + 0.4 * price_component.fillna(0.0)
    if "avg_yield_pct" in grouped.columns:
        yield_component = _minmax(grouped["avg_yield_pct"], invert=False)
        score = score + 0.6 * yield_component.fillna(0.0)

    grouped["score"] = score * 100
    grouped.sort_values("score", ascending=False, inplace=True)

    if "sigungu" not in grouped.columns:
        grouped["sigungu"] = ""
    if "eupmyeondong" not in grouped.columns:
        grouped["eupmyeondong"] = ""
    if "latest_contract" not in grouped.columns:
        grouped["latest_contract"] = pd.NaT

    grouped.reset_index(inplace=True)
    for col in ["lot_main", "lot_sub"]:
        if col not in grouped.columns:
            grouped[col] = ""

    grouped["property_label"] = grouped.apply(format_property_label, axis=1)
    grouped.drop(columns=[col for col in ["lot_main", "lot_sub"] if col in grouped.columns], inplace=True)

    result_cols = [
        "property_key",
        "property_label",
        "sigungu",
        "eupmyeondong",
    ]
    if "avg_price_per_m2" in grouped.columns:
        result_cols.append("avg_price_per_m2")
    if "avg_yield_pct" in grouped.columns:
        result_cols.append("avg_yield_pct")
    if "avg_trade_amount" in grouped.columns:
        result_cols.append("avg_trade_amount")
    if "avg_area_sqm" in grouped.columns:
        result_cols.append("avg_area_sqm")
    if "transaction_count" in grouped.columns:
        result_cols.append("transaction_count")
    result_cols.append("latest_contract")
    result_cols.append("score")

    return grouped[result_cols]

def restrict_to_regions(df: pd.DataFrame, regions: List[str]) -> pd.DataFrame:
    if df.empty or not regions:
        return df.copy()
    work = ensure_sigungu_named(df)
    if GROUP_COL not in work.columns:
        return df.copy()
    mask = work[GROUP_COL].isin(regions)
    if mask.empty:
        return df.copy()
    return df.loc[mask.values].copy()


STRATEGY_PRESETS = {
    "안정형": {"w1": 0.2, "w2": 0.2, "w3": 0.6},
    "성장형": {"w1": 0.3, "w2": 0.5, "w3": 0.2},
    "균형형": {"w1": 0.4, "w2": 0.3, "w3": 0.3},
}


def collect_months(*dfs: pd.DataFrame) -> list[int]:
    months: set[int] = set()
    for df in dfs:
        if df.empty or "YYYYMM" not in df.columns:
            continue
        vals = pd.to_numeric(df["YYYYMM"], errors="coerce").dropna().astype(int)
        months.update(vals.tolist())
    return sorted(months)


def format_percent(value: float) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number * 100:+.1f}%"


def format_score(value: float) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number:.1f}"


def render_time_series_chart(df: pd.DataFrame, value_col: str, label: str, axis_format: Optional[str] = None) -> None:
    if df.empty or value_col not in df.columns or "YYYYMM" not in df.columns:
        st.info(f"{label} 시계열 데이터를 찾을 수 없습니다.")
        return

    work = df[[GROUP_COL, "YYYYMM", value_col]].dropna().copy()
    if work.empty:
        st.info(f"{label} 시계열 데이터를 찾을 수 없습니다.")
        return

    numeric_values = pd.to_numeric(work[value_col], errors="coerce")
    if numeric_values.isna().all():
        st.info(f"{label} 시계열 데이터를 찾을 수 없습니다.")
        return
    if numeric_values.abs().sum() == 0:
        st.info(f"{label} 지표가 0으로만 집계되어 시각화를 생략합니다.")
        return
    work[value_col] = numeric_values

    work["YYYYMM"] = work["YYYYMM"].astype(str)
    axis_kwargs = {"title": label}
    if axis_format:
        axis_kwargs["format"] = axis_format

    legend_select = alt.selection_multi(
        fields=[GROUP_COL],
        bind="legend",
        empty="all",
    )

    chart = (
        alt.Chart(work)
        .mark_line(point=True)
        .encode(
            x=alt.X("YYYYMM:O", title="계약월"),
            y=alt.Y(f"{value_col}:Q", axis=alt.Axis(**axis_kwargs)),
            color=alt.Color(f"{GROUP_COL}:N", title="시군구"),
            opacity=alt.condition(legend_select, alt.value(1.0), alt.value(0.1)),
        )
        .add_params(legend_select)
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def _compute_metric_domain(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return 0.0, 1.0
    min_val, max_val = float(values.min()), float(values.max())
    if math.isclose(min_val, max_val, rel_tol=1e-6, abs_tol=1e-6):
        buffer = abs(min_val) * 0.05 or 0.01
        return min_val - buffer, max_val + buffer
    return min_val, max_val


def _build_altair_color_scale(metric_key: str, data: pd.DataFrame, percent_metrics: set[str]) -> alt.Scale:
    values = pd.to_numeric(data.get(metric_key, pd.Series(dtype=float)), errors="coerce").dropna()
    domain_min, domain_max = _compute_metric_domain(values)
    if metric_key in percent_metrics or (domain_min < 0.0 < domain_max):
        bound = max(abs(domain_min), abs(domain_max)) or 1e-6
        domain_min, domain_max = -bound, bound
        return alt.Scale(
            domain=[domain_min, domain_max],
            domainMid=0,
            range=["#b2182b", "#f7f7f7", "#2166ac"],
            clamp=True,
        )
    return alt.Scale(
        domain=[domain_min, domain_max],
        range=["#fee08b", "#f46d43"],
        clamp=True,
    )


def _lerp_rgba(start: list[int], end: list[int], ratio: float) -> list[int]:
    ratio = max(0.0, min(1.0, float(ratio)))
    return [int(start[i] + (end[i] - start[i]) * ratio) for i in range(len(start))]


def _colorize_rgba(value: float, domain_min: float, domain_max: float, diverging: bool) -> list[int]:
    neutral = [240, 240, 240, 80]
    if diverging:
        span = max(abs(domain_min), abs(domain_max), 1e-6)
        if value >= 0:
            return _lerp_rgba(neutral, [33, 113, 181, 220], value / span)
        return _lerp_rgba(neutral, [178, 24, 43, 220], abs(value) / span)
    span = max(domain_max - domain_min, 1e-6)
    return _lerp_rgba([237, 248, 251, 150], [8, 81, 156, 220], (value - domain_min) / span)


def render_rent_gap_chart(df: pd.DataFrame) -> None:
    if df.empty or "rent_gap_mean" not in df.columns:
        st.info("전월세 Rent Gap 데이터를 찾을 수 없습니다.")
        return

    chart_data = df[[GROUP_COL, "rent_gap_mean"]].dropna().copy()
    if chart_data.empty:
        st.info("전월세 Rent Gap 데이터를 찾을 수 없습니다.")
        return

    chart_data["gap_direction"] = np.where(chart_data["rent_gap_mean"] >= 0, "환산가 우위", "월세 우위")
    chart = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("rent_gap_mean:Q", title="월 Rent Gap(만원)"),
        y=alt.Y(f"{GROUP_COL}:N", sort='-x', title=""),
        color=alt.Color("gap_direction:N", title="상태", scale=alt.Scale(domain=["환산가 우위", "월세 우위"], range=["#4E79A7", "#F28E2B"]))
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)


def prepare_map_data(base: pd.DataFrame, centroids: pd.DataFrame, metric: str) -> pd.DataFrame:
    if base.empty or centroids.empty or metric not in base.columns:
        return pd.DataFrame()
    map_base = base[[GROUP_COL, metric]].copy()
    map_base[GROUP_COL] = map_base[GROUP_COL].apply(normalize_sigungu_label)
    map_base = (
        map_base.dropna(subset=[GROUP_COL, metric])
        .groupby(GROUP_COL, as_index=False)[metric]
        .mean()
    )
    merged = map_base.merge(centroids, on=GROUP_COL, how="inner")
    merged = merged.dropna(subset=[metric, "latitude", "longitude"])
    return merged


def compute_choropleth_dataset(
    boundaries: dict[str, dict],
    metrics: pd.DataFrame,
    metric: str,
    diverging_metrics: Optional[Set[str]] = None,
) -> pd.DataFrame:
    if not boundaries or metrics.empty or metric not in metrics.columns:
        return pd.DataFrame()

    if GROUP_COL not in metrics.columns:
        return pd.DataFrame()

    normalized = metrics.copy()
    normalized[GROUP_COL] = normalized[GROUP_COL].apply(normalize_sigungu_label)
    normalized = normalized.dropna(subset=[GROUP_COL])
    metric_series = (
        normalized.groupby(GROUP_COL)[metric]
        .mean()
        .dropna()
    )
    data: list[dict[str, object]] = []
    values: list[float] = []

    for name, feature in boundaries.items():
        if name not in metric_series.index:
            continue
        geometry = feature.get("geometry", {})
        gtype = geometry.get("type")
        coords = geometry.get("coordinates", [])
        rings: list[list[list[float]]] = []
        if gtype == "Polygon":
            rings = coords
        elif gtype == "MultiPolygon":
            for poly in coords:
                rings.extend(poly)
        else:
            continue
        if not rings:
            continue
        value = metric_series.loc[name]
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        values.append(value_float)
        data.append({
            "시군구": name,
            "value": value_float,
            "polygon": rings,
        })

    if not data:
        return pd.DataFrame()

    values_series = pd.Series(values)
    domain_min, domain_max = _compute_metric_domain(values_series)
    diverging = (diverging_metrics and metric in diverging_metrics) or (domain_min < 0.0 < domain_max)

    for row in data:
        val = row.get("value")
        if pd.isna(val):
            row["color"] = [200, 200, 200, 60]
        else:
            row["color"] = _colorize_rgba(float(val), domain_min, domain_max, diverging)

    return pd.DataFrame(data)


def compute_view_state_from_data(point_df: Optional[pd.DataFrame] = None, poly_df: Optional[pd.DataFrame] = None) -> pdk.ViewState:
    latitudes: list[float] = []
    longitudes: list[float] = []

    if point_df is not None and not point_df.empty:
        if "latitude" in point_df.columns and "longitude" in point_df.columns:
            latitudes.extend(point_df["latitude"].dropna().tolist())
            longitudes.extend(point_df["longitude"].dropna().tolist())

    if poly_df is not None and not poly_df.empty and "polygon" in poly_df.columns:
        for rings in poly_df["polygon"]:
            if not isinstance(rings, list):
                continue
            for ring in rings:
                if not isinstance(ring, list):
                    continue
                for coords in ring:
                    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                        continue
                    lon, lat = coords[0], coords[1]
                    longitudes.append(lon)
                    latitudes.append(lat)

    lat_series = pd.Series(latitudes)
    lon_series = pd.Series(longitudes)
    lat_center = float(lat_series.mean()) if not lat_series.empty else 36.2
    lon_center = float(lon_series.mean()) if not lon_series.empty else 127.8
    return pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=7.0)


@st.cache_data(show_spinner=False, ttl=900, max_entries=32)
def compute_dom_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            GROUP_COL,
            "YYYYMM",
            "dom_median",
            "dom_p75",
            "cancel_rate_dom",
            "txn_cnt_dom",
            "vacancy_proxy",
            "contract_months_avg",
            "contract_months_median",
            "contract_renewal_rate",
            "contract_extension_rate",
            "contract_stability",
        ])

    work = ensure_sigungu_named(df)
    if GROUP_COL not in work.columns:
        return pd.DataFrame(columns=[
            GROUP_COL,
            "YYYYMM",
            "dom_median",
            "dom_p75",
            "cancel_rate_dom",
            "txn_cnt_dom",
            "vacancy_proxy",
            "contract_months_avg",
            "contract_months_median",
            "contract_renewal_rate",
            "contract_extension_rate",
            "contract_stability",
        ])

    required = {"YYYYMM", "취소여부", "계약일자"}
    if not required.issubset(work.columns):
        return pd.DataFrame(columns=[
            GROUP_COL,
            "YYYYMM",
            "dom_median",
            "dom_p75",
            "cancel_rate_dom",
            "txn_cnt_dom",
            "vacancy_proxy",
            "contract_months_avg",
            "contract_months_median",
            "contract_renewal_rate",
            "contract_extension_rate",
            "contract_stability",
        ])

    work = work.copy()
    work["snapshot_date"] = pd.to_datetime(work.get("snapshot_date"), errors="coerce")

    if "YYYYMM" in work.columns:
        yyyymm_numeric = pd.to_numeric(work["YYYYMM"], errors="coerce").astype("Int64")
        fallback_snapshot = pd.to_datetime(yyyymm_numeric.astype(str), format="%Y%m", errors="coerce")
        if fallback_snapshot.notna().any():
            fallback_snapshot = fallback_snapshot + MonthEnd(0)
            missing_mask = work["snapshot_date"].isna()
            if missing_mask.any():
                work.loc[missing_mask & fallback_snapshot.notna(), "snapshot_date"] = fallback_snapshot[missing_mask & fallback_snapshot.notna()]

    work["계약일자"] = pd.to_datetime(work["계약일자"], errors="coerce")
    work["취소여부"] = pd.to_numeric(work["취소여부"], errors="coerce")

    contract_defaults = {
        "contract_months": np.nan,
        "contract_is_renewal": np.nan,
        "contract_extension_flag": np.nan,
        "contract_stability_score": np.nan,
    }
    for column, default in contract_defaults.items():
        if column not in work.columns:
            work[column] = default

    bool_like = {
        "true": 1,
        "false": 0,
        True: 1,
        False: 0,
        "Y": 1,
        "N": 0,
        "y": 1,
        "n": 0,
    }
    for column in contract_defaults:
        if column not in work.columns:
            continue
        series = work[column]
        if series.dtype == object:
            normalized = series.replace(bool_like)
        else:
            normalized = series
        work[column] = pd.to_numeric(normalized, errors="coerce")

    work["dom_days"] = (work["snapshot_date"] - work["계약일자"]).dt.days
    work.loc[work["dom_days"] < 0, "dom_days"] = np.nan

    agg = (
        work.groupby([GROUP_COL, "YYYYMM"], as_index=False)
        .agg(
            dom_median=("dom_days", "median"),
            dom_p75=("dom_days", lambda s: s.quantile(0.75)),
            cancel_rate_dom=("취소여부", "mean"),
            txn_cnt_dom=("취소여부", "count"),
            contract_months_avg=("contract_months", "mean"),
            contract_months_median=("contract_months", "median"),
            contract_renewal_rate=("contract_is_renewal", "mean"),
            contract_extension_rate=("contract_extension_flag", "mean"),
            contract_stability=("contract_stability_score", "mean"),
        )
    )

    agg.sort_values([GROUP_COL, "YYYYMM"], inplace=True)
    agg["vacancy_proxy"] = agg.groupby(GROUP_COL)["txn_cnt_dom"].transform(
        lambda s: safe_divide(s.rolling(3, min_periods=1).mean(), s.rolling(12, min_periods=3).mean()) - 1
    )

    for col in [
        "dom_median",
        "dom_p75",
        "cancel_rate_dom",
        "contract_months_avg",
        "contract_months_median",
        "contract_renewal_rate",
        "contract_extension_rate",
        "contract_stability",
    ]:
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)

    agg["YYYYMM"] = pd.to_numeric(agg["YYYYMM"], errors="coerce").astype("Int64")
    return agg


def minmax_scale(series: pd.Series, invert: bool = False) -> pd.Series:
    work = pd.to_numeric(series, errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    valid = work.dropna()
    if valid.empty:
        scaled = pd.Series(0.0, index=series.index, dtype=float)
    else:
        min_val = valid.min()
        max_val = valid.max()
        if pd.isna(min_val) or pd.isna(max_val) or np.isclose(max_val, min_val):
            scaled = pd.Series(0.5, index=series.index, dtype=float)
        else:
            scaled = (work - min_val) / (max_val - min_val)
            scaled = scaled.fillna(0.0).clip(0.0, 1.0)
    if invert:
        scaled = 1.0 - scaled
    return scaled


@st.cache_data(show_spinner=False, ttl=600, max_entries=32)
def compute_persona_profiles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or GROUP_COL not in df.columns:
        return pd.DataFrame(columns=[GROUP_COL, "income_score", "growth_score", "capital_score"])

    persona_components: dict[str, list[tuple[str, bool]]] = {
        "income": [
            ("avg_yield_pct", False),
            ("stability", False),
            ("cancel_rate", True),
            ("cv_p_per_m2", True),
            ("dom_median", True),
            ("vacancy_proxy", True),
            ("interest_rate", True),
            ("contract_stability", False),
            ("contract_renewal_rate", False),
        ],
        "growth": [
            ("new_premium", False),
            ("floor_premium", False),
            ("mom_3m", False),
            ("rent_gap_mean", True),
            ("avg_p_per_m2", True),
            ("contract_months_avg", False),
            ("school_score", False),
            ("amenity_score", False),
        ],
        "capital": [
            ("mom_3m", False),
            ("undervalue_rate", False),
            ("composite_score", False),
            ("rent_undervalue_rate", False),
            ("vacancy_proxy", True),
            ("contract_stability", False),
        ],
    }

    persona_weights: dict[str, dict[str, float]] = {
        "income": {
            "avg_yield_pct": 0.35,
            "stability": 0.25,
            "cancel_rate": 0.05,
            "cv_p_per_m2": 0.05,
            "dom_median": 0.05,
            "vacancy_proxy": 0.1,
            "interest_rate": 0.05,
            "contract_stability": 0.05,
            "contract_renewal_rate": 0.05,
        },
        "growth": {
            "new_premium": 0.25,
            "floor_premium": 0.15,
            "mom_3m": 0.2,
            "rent_gap_mean": 0.1,
            "avg_p_per_m2": 0.05,
            "contract_months_avg": 0.05,
            "school_score": 0.1,
            "amenity_score": 0.1,
        },
        "capital": {
            "mom_3m": 0.3,
            "undervalue_rate": 0.25,
            "composite_score": 0.2,
            "rent_undervalue_rate": 0.15,
            "vacancy_proxy": 0.05,
            "contract_stability": 0.05,
        },
    }

    base = df.copy()
    base = base.groupby(GROUP_COL, as_index=False).first()
    result = base[[GROUP_COL]].copy()

    for persona_key, components in persona_components.items():
        scaled_parts: dict[str, pd.Series] = {}
        for col, invert in components:
            if col in base.columns:
                scaled_parts[col] = minmax_scale(base[col], invert=invert)

        if not scaled_parts:
            score = pd.Series(0.0, index=base.index, dtype=float)
        else:
            stacked = pd.DataFrame(scaled_parts)
            weights_map = persona_weights.get(persona_key, {})
            weights = pd.Series({col: weights_map.get(col, 1.0) for col in stacked.columns}, dtype=float)
            total_weight = float(weights.sum()) if float(weights.sum()) > 0 else float(len(stacked.columns))
            weighted = stacked.multiply(weights, axis=1)
            score = weighted.sum(axis=1) / total_weight
            score = score.fillna(0.0)

        result[f"{persona_key}_score"] = score.values

    columns_to_include = [
        "avg_yield_pct",
        "stability",
        "cancel_rate",
        "cv_p_per_m2",
        "new_premium",
        "floor_premium",
        "mom_3m",
        "undervalue_rate",
        "composite_score",
        "dom_median",
        "cancel_rate_dom",
        "rent_undervalue_rate",
        "rent_gap_mean",
        "avg_p_per_m2",
        "interest_rate",
        "policy_comment",
        "vacancy_proxy",
        "contract_months",
        "contract_stability_score",
        "contract_is_renewal",
        "contract_extension_flag",
        "contract_months_avg",
        "contract_months_median",
        "contract_renewal_rate",
        "contract_extension_rate",
        "contract_stability",
        "school_score",
        "amenity_score",
        "lifestyle_comment",
    ]
    for col in columns_to_include:
        if col in base.columns and col not in result.columns:
            result[col] = base[col]

    return result


def prepare_ml_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty or GROUP_COL not in df.columns:
        return pd.DataFrame()

    usable = [col for col in feature_cols if col in df.columns]
    if target_col:
        if target_col not in df.columns:
            return pd.DataFrame()
        usable.append(target_col)
    if not usable:
        return pd.DataFrame()

    cols = [GROUP_COL] + usable
    work = df[cols].copy()
    for col in usable:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if target_col:
        drop_cols = [col for col in usable if col != target_col]
        work.dropna(subset=drop_cols + [target_col], inplace=True)
    else:
        work.dropna(subset=usable, inplace=True)
    work = work.replace([np.inf, -np.inf], np.nan)
    if work.empty:
        return pd.DataFrame()
    return work


@st.cache_data(show_spinner=False, ttl=600, max_entries=16)
def run_kmeans_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    clusters: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deps = _missing_dependencies({
        "KMeans": KMeans,
        "StandardScaler": StandardScaler,
    })
    if deps:
        return pd.DataFrame(), pd.DataFrame()

    dataset = prepare_ml_dataset(df, feature_cols)
    if dataset.empty or clusters <= 1:
        return pd.DataFrame(), pd.DataFrame()
    if dataset.shape[0] < clusters:
        return pd.DataFrame(), pd.DataFrame()

    features = [col for col in feature_cols if col in dataset.columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataset[features])
    model = KMeans(n_clusters=clusters, n_init=10, random_state=42)
    labels = model.fit_predict(scaled)

    clustered = dataset.copy()
    clustered["cluster"] = labels

    scaled_df = pd.DataFrame(scaled, columns=features, index=clustered.index)
    scaled_df["cluster"] = labels
    cluster_scaled_stats = scaled_df.groupby("cluster")[features].mean()

    cluster_summary = clustered.groupby("cluster")[features].mean().reset_index()
    score = np.zeros(len(cluster_summary), dtype=float)
    for idx, row in cluster_summary.iterrows():
        val = 0.0
        if "mom_3m" in cluster_scaled_stats.columns:
            val += cluster_scaled_stats.loc[row["cluster"], "mom_3m"]
        if "stability" in cluster_scaled_stats.columns:
            val += 0.5 * cluster_scaled_stats.loc[row["cluster"], "stability"]
        if "avg_p_per_m2" in cluster_scaled_stats.columns:
            val -= cluster_scaled_stats.loc[row["cluster"], "avg_p_per_m2"]
        cluster_summary.loc[idx, "kmeans_score"] = val

    return clustered, cluster_summary.sort_values("kmeans_score", ascending=False)


def compute_random_forest_undervaluation(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, Optional[RandomForestRegressor], list[str]]:
    deps = _missing_dependencies({
        "RandomForestRegressor": RandomForestRegressor,
    })
    if deps:
        return pd.DataFrame(), None, []

    dataset = prepare_ml_dataset(df, feature_cols, target_col)
    if dataset.empty:
        return pd.DataFrame(), None, []

    features = [col for col in feature_cols if col in dataset.columns]
    if not features:
        return pd.DataFrame(), None, []

    X = dataset[features]
    y = dataset[target_col]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    result = dataset.copy()
    result["predicted"] = preds
    result["ml_undervalue_rate"] = np.where(
        preds == 0,
        0.0,
        (preds - y) / preds,
    )
    return result, model, features


def build_decision_tree_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[Optional[DecisionTreeRegressor], list[str]]:
    deps = _missing_dependencies({
        "DecisionTreeRegressor": DecisionTreeRegressor,
    })
    if deps:
        return None, []

    dataset = prepare_ml_dataset(df, feature_cols, target_col)
    if dataset.empty:
        return None, []

    features = [col for col in feature_cols if col in dataset.columns]
    if not features:
        return None, []

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(dataset[features], dataset[target_col])
    return model, features


def build_compare_comments(df: pd.DataFrame) -> str:
    if df.empty or GROUP_COL not in df.columns:
        return ""

    work = df.copy()
    highlight_specs = [
        {"key": "composite_score", "label": "종합 점수", "better": "high", "formatter": lambda v: f"{v:.1f}"},
        {"key": "undervalue_rate", "label": "저평가율", "better": "high", "formatter": format_percent},
        {"key": "mom_3m", "label": "모멘텀", "better": "high", "formatter": format_percent},
        {"key": "school_score", "label": "학군", "better": "high", "formatter": lambda v: f"{v:.1f}"},
        {"key": "amenity_score", "label": "생활편의", "better": "high", "formatter": lambda v: f"{v:.1f}"},
        {"key": "vacancy_proxy", "label": "공실 위험", "better": "low", "formatter": format_percent},
    ]

    def render_value(raw: float, formatter: Callable[[float], str]) -> str:
        if pd.isna(raw):
            return "-"
        try:
            return formatter(float(raw))
        except Exception:  # pragma: no cover - 안전 장치
            return "-"

    highlights: list[str] = []
    for spec in highlight_specs:
        key = spec["key"]
        if key not in work.columns:
            continue
        series = pd.to_numeric(work[key], errors="coerce").dropna()
        if series.empty:
            continue
        if spec.get("better") == "low":
            best_idx = series.idxmin()
            worst_idx = series.idxmax()
        else:
            best_idx = series.idxmax()
            worst_idx = series.idxmin()
        best_region = work.loc[best_idx, GROUP_COL]
        best_val = work.loc[best_idx, key]
        worst_region = work.loc[worst_idx, GROUP_COL]
        worst_val = work.loc[worst_idx, key]
        formatter = spec["formatter"]
        sentence = f"{spec['label']} 최고 {best_region}({render_value(best_val, formatter)})"
        if len(series) > 1 and best_region != worst_region:
            sentence += f" · 주의 {worst_region}({render_value(worst_val, formatter)})"
        highlights.append(sentence)

    top_highlights = highlights[:3]

    metric_details = [
        ("undervalue_rate", "저평가", format_percent),
        ("mom_3m", "모멘텀", format_percent),
        ("stability", "가격 안정성", lambda v: f"{v * 100:.0f}%"),
        ("contract_stability", "계약 안정성", format_percent),
        ("school_score", "학군", lambda v: f"{v:.1f}"),
        ("amenity_score", "생활편의", lambda v: f"{v:.1f}"),
        ("dom_median", "중위 DOM", lambda v: f"{v:.0f}일"),
        ("vacancy_proxy", "공실", format_percent),
        ("rent_gap_mean", "Rent Gap", lambda v: f"{v:.1f}"),
        ("composite_score", "종합", lambda v: f"{v:.1f}"),
    ]

    detail_lines: list[str] = []
    for _, row in work.iterrows():
        pieces: list[str] = []
        for key, label, formatter in metric_details:
            if key not in row:
                continue
            formatted = render_value(row.get(key), formatter)
            if formatted == "-":
                continue
            pieces.append(f"{label} {formatted}")
            if len(pieces) >= 4:
                break
        comment_tail = row.get("lifestyle_comment") if "lifestyle_comment" in row else ""
        if comment_tail and isinstance(comment_tail, str) and comment_tail.strip():
            pieces.append(comment_tail.strip())
        if not pieces:
            pieces.append("지표 없음")
        detail_lines.append(f"- {row.get(GROUP_COL, '미지정')}: " + ", ".join(pieces))

    blocks = []
    if top_highlights:
        blocks.append("\n".join(f"- {text}" for text in top_highlights))
    if detail_lines:
        blocks.append("\n".join(detail_lines))
    return "\n\n".join(blocks)


def prepare_radar_frame(df: pd.DataFrame, config: list[dict[str, Union[str, bool]]]) -> pd.DataFrame:
    if df.empty or GROUP_COL not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for order, item in enumerate(config):
        metric = item.get("key")
        label = item.get("label", metric)
        invert = bool(item.get("invert", False))
        if metric not in df.columns:
            continue
        scaled = minmax_scale(df[metric], invert=invert)
        for idx in df.index:
            region = df.at[idx, GROUP_COL]
            scaled_val = scaled.loc[idx] if idx in scaled.index else np.nan
            raw_val = df.at[idx, metric]
            rows.append(
                {
                    GROUP_COL: region,
                    "metric": label,
                    "metric_key": metric,
                    "order": order,
                    "value": float(scaled_val) if not pd.isna(scaled_val) else 0.0,
                    "raw": raw_val,
                }
            )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame.sort_values([GROUP_COL, "order"], inplace=True)

    repeats = frame.groupby(GROUP_COL).head(1).copy()
    repeats["order"] = repeats["order"] + len(config)
    frame = pd.concat([frame, repeats], ignore_index=True)

    return frame


def render_ranked_list(
    df: pd.DataFrame,
    value_col: str,
    label: str,
    ascending: bool = False,
    formatter: Optional[Callable[[float], str]] = None,
    axis_format: str = ".0%",
) -> None:
    if df.empty or value_col not in df.columns:
        st.info(f"{label} 데이터를 계산할 수 없습니다.")
        return

    top = df.sort_values(value_col, ascending=ascending).head(5).reset_index(drop=True)
    if top.empty:
        st.info(f"{label} 데이터를 계산할 수 없습니다.")
        return

    display = top[[GROUP_COL, value_col]].copy()
    display.insert(0, "순위", display.index + 1)
    fmt = formatter or format_percent
    display[label] = display[value_col].apply(fmt)
    display.drop(columns=[value_col], inplace=True)
    display.rename(columns={GROUP_COL: "시군구"}, inplace=True)
    card_count = min(3, len(display))
    if card_count > 0:
        card_cols = st.columns(card_count)
        for col, (_, row) in zip(card_cols, display.head(card_count).iterrows()):
            col.metric(
                f"#{int(row['순위'])} {row['시군구']}",
                row[label],
            )

    with st.expander("Top5 상세", expanded=False):
        st.table(display)

    chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(value_col, title=label, axis=alt.Axis(format=axis_format)),
        y=alt.Y(f"{GROUP_COL}:N", sort='-x' if not ascending else 'x', title=""),
        color=alt.value("#4E79A7")
    ).properties(height=150)
    st.altair_chart(chart, use_container_width=True)


# =============================
# Top5 인사이트 계산 함수
# =============================
@st.cache_data(show_spinner=False, ttl=900, max_entries=32)
def compute_undervaluation(df):
    base_cols = [GROUP_COL, "undervalue_rate", "avg_price_per_m2"]
    if "가격_per_㎡" not in df.columns:
        return pd.DataFrame(columns=base_cols)
    work = ensure_sigungu_named(df)
    if GROUP_COL not in work.columns:
        return pd.DataFrame(columns=base_cols)
    tmp = work[[GROUP_COL, "가격_per_㎡"]].copy()
    tmp["시도"] = tmp[GROUP_COL].astype(str).str.split().str[0]
    region_avg = (
        tmp.groupby([GROUP_COL, "시도"], dropna=True)["가격_per_㎡"].mean().rename("avg_price_per_m2").reset_index()
    )
    if region_avg.empty:
        return pd.DataFrame(columns=base_cols)
    sido_avg = region_avg.groupby("시도")["avg_price_per_m2"].mean().rename("sido_avg").reset_index()
    out = region_avg.merge(sido_avg, on="시도", how="left")
    out = out.dropna(subset=["sido_avg", "avg_price_per_m2"])  # ensure we have baseline
    out = out[out["sido_avg"].abs() > 1e-9]
    if out.empty:
        return pd.DataFrame(columns=base_cols)
    out["undervalue_rate"] = (out["sido_avg"] - out["avg_price_per_m2"]) / out["sido_avg"]
    return out[[GROUP_COL, "undervalue_rate", "avg_price_per_m2"]]

@st.cache_data(show_spinner=False, ttl=900, max_entries=32)
def compute_new_premium(df):
    base_cols = [
        GROUP_COL,
        "new_avg",
        "new_cnt",
        "old_avg",
        "old_cnt",
        "new_premium",
    ]
    required_cols = ["건축년도", "가격_per_㎡"]
    if not set(required_cols).issubset(df.columns):
        return pd.DataFrame(columns=base_cols)

    work = ensure_sigungu_named(df)
    if GROUP_COL not in work.columns:
        return pd.DataFrame(columns=base_cols)

    work = work[[GROUP_COL] + required_cols].copy()
    work["건축년도"] = pd.to_numeric(work["건축년도"], errors="coerce")
    work["가격_per_㎡"] = pd.to_numeric(work["가격_per_㎡"], errors="coerce")
    work = work.dropna(subset=["건축년도", "가격_per_㎡"])
    if work.empty:
        return pd.DataFrame(columns=base_cols)

    current_year = pd.Timestamp.today().year
    age = current_year - work["건축년도"]
    work["is_new"] = age <= NEW_BUILD_YEARS
    work["is_old"] = age >= OLD_BUILD_YEARS

    new_stats = (
        work[work["is_new"]]
        .groupby(GROUP_COL)["가격_per_㎡"]
        .agg(new_avg="mean", new_cnt="count")
    )
    old_stats = (
        work[work["is_old"]]
        .groupby(GROUP_COL)["가격_per_㎡"]
        .agg(old_avg="mean", old_cnt="count")
    )

    if new_stats.empty or old_stats.empty:
        return pd.DataFrame(columns=base_cols)

    out = new_stats.join(old_stats, how="inner").dropna().reset_index()
    if out.empty:
        return pd.DataFrame(columns=base_cols)

    mask = (
        (out["new_cnt"] >= MIN_NEW_SAMPLE_COUNT)
        & (out["old_cnt"] >= MIN_OLD_SAMPLE_COUNT)
        & (out["old_avg"] >= MIN_OLD_PRICE_PER_M2)
    )
    out = out.loc[mask]
    if out.empty:
        return pd.DataFrame(columns=base_cols)

    out["new_premium"] = (out["new_avg"] - out["old_avg"]) / out["old_avg"]
    return out.reindex(columns=base_cols)

@st.cache_data(show_spinner=False, ttl=900, max_entries=32)
def compute_floor_premium(df):
    base_cols = [GROUP_COL, "high_avg", "low_avg", "floor_premium"]
    if not {"층","가격_per_㎡"}.issubset(df.columns):
        return pd.DataFrame(columns=base_cols)
    work = ensure_sigungu_named(df)
    if GROUP_COL not in work.columns:
        return pd.DataFrame(columns=base_cols)
    work["floor_bucket"]=pd.cut(work["층"].fillna(-1),[-1,5,15,np.inf],labels=["저층","중층","고층"])
    hi=work[work["floor_bucket"]=="고층"].groupby(GROUP_COL)["가격_per_㎡"].mean().rename("high_avg")
    lo=work[work["floor_bucket"]=="저층"].groupby(GROUP_COL)["가격_per_㎡"].mean().rename("low_avg")
    out=pd.concat([hi,lo],axis=1).dropna().reset_index()
    out["floor_premium"]=(out["high_avg"]-out["low_avg"])/out["low_avg"]
    return out.reindex(columns=base_cols)

@st.cache_data(show_spinner=False, ttl=600, max_entries=32)
def compute_composite_score(uv,momo,vol,w1=0.4,w2=0.3,w3=0.3):
    if uv.empty or momo.empty or vol.empty:
        return pd.DataFrame()
    uv_work = ensure_sigungu_named(uv)
    momo_work = ensure_sigungu_named(momo)
    vol_work = ensure_sigungu_named(vol)
    if GROUP_COL not in uv_work.columns or GROUP_COL not in momo_work.columns or GROUP_COL not in vol_work.columns:
        return pd.DataFrame()
    momo_last = momo_work.groupby(GROUP_COL).tail(1)[[GROUP_COL,"mom_3m"]]
    uv_mean = uv_work.groupby(GROUP_COL)["undervalue_rate"].mean().reset_index()
    vol_stat = vol_work.groupby(GROUP_COL).agg(avg=("avg_p_per_m2","mean"),std=("avg_p_per_m2","std")).reset_index()
    vol_stat["stability"] = (1 - vol_stat["std"] / vol_stat["avg"]).fillna(0)
    comp = uv_mean.merge(momo_last,on=GROUP_COL).merge(vol_stat[[GROUP_COL,"stability"]],on=GROUP_COL)
    scale = lambda s,invert=False: ((1 - s.rank(pct=True)) if invert else s.rank(pct=True)) * 100
    comp["score_undervalue"] = scale(comp["undervalue_rate"])
    comp["score_momentum"] = scale(comp["mom_3m"])
    comp["score_stability"] = scale(comp["stability"])
    comp["composite_score"] = w1 * comp["score_undervalue"] + w2 * comp["score_momentum"] + w3 * comp["score_stability"]
    return comp.sort_values("composite_score",ascending=False)

@st.cache_data(show_spinner=False, ttl=900, max_entries=32)
def compute_rent_metrics(df, annual_rate: float = 0.055, *, group_field: str = GROUP_COL):
    if df.empty or not {"보증금_만원", "월세_만원"}.issubset(df.columns):
        return pd.DataFrame()
    work = ensure_sigungu_named(df)
    target_group = group_field if group_field else GROUP_COL
    if target_group not in work.columns:
        return pd.DataFrame()
    work = work.copy()
    if "src_type" in work.columns:
        lease_mask = work["src_type"].astype(str).str.contains("lease", case=False, na=False)
        if not lease_mask.any():
            return pd.DataFrame()
        work = work[lease_mask]
    if work[["보증금_만원", "월세_만원"]].sum().sum() == 0:
        return pd.DataFrame()
    work["lease_equiv_monthly"] = work["보증금_만원"] * (annual_rate / 12.0)
    work["rent_gap_monthly"] = work["lease_equiv_monthly"] - work["월세_만원"]

    summary = (
        work.groupby(target_group)
        .agg(
            rent_gap_mean=("rent_gap_monthly", "mean"),
            avg_monthly_rent=("월세_만원", "mean"),
        )
        .reset_index()
    )

    overall_avg_rent = summary["avg_monthly_rent"].mean()
    if pd.isna(overall_avg_rent) or overall_avg_rent == 0:
        summary["rent_undervalue_rate"] = 0.0
    else:
        summary["rent_undervalue_rate"] = (
            (overall_avg_rent - summary["avg_monthly_rent"]) / overall_avg_rent
        )
    return summary
# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="부동산 대시보드 (최종 통합판)", layout="wide")
st.title("🏠 부동산 인사이트 대시보드")

with st.sidebar:
    st.header("⚙️ 분석 설정")
    st.markdown("### 데이터 소스")
    backend_options = ["Parquet"]
    backend_default = 0
    if duckdb is not None:
        backend_options.append("DuckDB")
        if _can_use_duckdb(DUCKDB_ENV_PATH):
            backend_default = 1
    data_backend = st.selectbox(
        "데이터 소스",
        backend_options,
        index=min(backend_default, len(backend_options) - 1),
        help="Parquet/CSV 파일 또는 DuckDB 파일에서 데이터를 불러옵니다.",
    )
    duckdb_path_input = ""
    if data_backend == "DuckDB":
        duckdb_path_input = st.text_input(
            "DuckDB 파일 경로",
            DUCKDB_ENV_PATH,
            help="ETL 단계에서 생성한 DuckDB 파일 경로를 입력하세요.",
        ).strip()
        if not _can_use_duckdb(duckdb_path_input):
            warning_msg = "DuckDB 파일이 없거나 duckdb 패키지가 설치되지 않았습니다."
            st.warning(warning_msg, icon="⚠️")
            push_log(warning_msg, "warning")

    use_combined_sources = False
    if data_backend == "Parquet":
        use_combined_sources = st.checkbox(
            "매칭된 combined 산출물 우선 사용",
            value=True,
            help="ETL에서 생성된 *_combined 파일이 존재하면 해당 데이터를 우선 사용합니다.",
        )

    basics_raw = ensure_yyyymm_numeric(
        load_monthly_basics(
            data_backend,
            duckdb_path_input,
            prefer_combined=use_combined_sources,
        )
    )
    momo_raw = ensure_yyyymm_numeric(
        load_momentum(
            data_backend,
            duckdb_path_input,
            prefer_combined=use_combined_sources,
        )
    )
    vol_raw = ensure_yyyymm_numeric(
        load_volatility(
            data_backend,
            duckdb_path_input,
            prefer_combined=use_combined_sources,
        )
    )
    fact_raw = ensure_yyyymm_numeric(
        load_transactions(
            data_backend,
            duckdb_path_input,
            prefer_combined=use_combined_sources,
        )
    )

    if data_backend == "Parquet":
        combined_files = [
            MONTHLY_PATH_COMBINED,
            MOMENTUM_PATH_COMBINED,
            VOLATILITY_PATH_COMBINED,
            FACT_PATH_COMBINED,
        ]
        if use_combined_sources:
            if all(path.exists() for path in combined_files):
                st.caption("Combined 산출물을 사용 중입니다.")
            elif any(path.exists() for path in combined_files):
                st.info(
                    "일부 combined 산출물이 없어 사용 가능한 파일은 기본 데이터로 대체합니다.",
                    icon="ℹ️",
                )
            else:
                st.warning(
                    "combined 산출물이 존재하지 않아 기본 processed 데이터를 사용합니다.",
                    icon="⚠️",
                )

        if basics_raw.empty and not MONTHLY_PATH.exists() and not MONTHLY_PATH_COMBINED.exists():
            warn_missing_resource(MONTHLY_PATH, "월간 기본 지표 (monthly_basics.parquet)")
        if momo_raw.empty and not MOMENTUM_PATH.exists() and not MOMENTUM_PATH_COMBINED.exists():
            warn_missing_resource(MOMENTUM_PATH, "모멘텀 데이터 (monthly_momentum.parquet)")
        if vol_raw.empty and not VOLATILITY_PATH.exists() and not VOLATILITY_PATH_COMBINED.exists():
            warn_missing_resource(VOLATILITY_PATH, "변동성 데이터 (monthly_volatility.parquet)")
        if fact_raw.empty and not FACT_PATH.exists() and not FACT_PATH_COMBINED.exists():
            warn_missing_resource(FACT_PATH, "거래 원장 (transactions.csv)")

    st.markdown("### 📂 데이터 선택")
    src_type = st.selectbox(
        "자산 유형",
        [
            "apt_trade", "apt_lease", "offi_trade", "offi_lease",
            "comm_trade", "row_trade", "row_lease", "det_trade", "det_lease",
        ],
        help="분석할 자산 유형을 선택하세요."
    )

    basics_src = filter_by_src(basics_raw, src_type)
    momo_src = filter_by_src(momo_raw, src_type)
    vol_src = filter_by_src(vol_raw, src_type)
    fact_src = filter_by_src(fact_raw, src_type)
    contract_src_type = resolve_contract_source(src_type)
    contract_src = filter_by_src(fact_raw, contract_src_type) if contract_src_type != src_type else fact_src
    if contract_src.empty and contract_src_type != src_type:
        contract_src = fact_src.copy()

    available_months = collect_months(basics_src, momo_src, vol_src, fact_src)
    months_cap = max(1, min(24, len(available_months) if available_months else 1))
    period_choice = st.selectbox(
        "기간",
        ["최근 6개월", "최근 1년", "전체"],
        index=1,
        help="분석에 사용할 거래 기간을 선택하세요."
    )
    period_map = {"최근 6개월": 6, "최근 1년": 12, "전체": None}
    base_months = period_map.get(period_choice)
    recent_months = None if base_months is None else min(base_months, months_cap)

    dual_assets = {"apt", "offi", "row", "det"}
    asset_prefix = src_type.split("_", 1)[0] if isinstance(src_type, str) and "_" in src_type else src_type
    enforce_dual_regions = False
    common_regions: List[str] = []
    if asset_prefix in dual_assets:
        trade_type = f"{asset_prefix}_trade"
        lease_type = f"{asset_prefix}_lease"
        trade_recent = filter_recent_months(filter_by_src(fact_raw, trade_type), recent_months)
        lease_recent = filter_recent_months(filter_by_src(fact_raw, lease_type), recent_months)
        if not trade_recent.empty and not lease_recent.empty:
            enforce_dual_regions = st.checkbox(
                "매매·전월세 데이터가 모두 있는 시군구만 분석",
                value=True,
                key=f"dual_region_toggle_{asset_prefix}",
                help="선택 자산의 매매와 전월세 데이터가 모두 존재하는 시군구만 포함합니다.",
            )
            if enforce_dual_regions:
                trade_regions = set(ensure_sigungu_named(trade_recent)[GROUP_COL].dropna().unique().tolist())
                lease_regions = set(ensure_sigungu_named(lease_recent)[GROUP_COL].dropna().unique().tolist())
                common_regions = sorted(trade_regions & lease_regions)
                if common_regions:
                    filtered_basics = restrict_to_regions(basics_src, common_regions)
                    filtered_momo = restrict_to_regions(momo_src, common_regions)
                    filtered_vol = restrict_to_regions(vol_src, common_regions)
                    filtered_fact = restrict_to_regions(fact_src, common_regions)
                    filtered_contract = restrict_to_regions(contract_src, common_regions)
                    if filtered_fact.empty:
                        st.warning("선택한 조건에 해당하는 거래 데이터를 찾을 수 없습니다. 전체 지역을 사용합니다.")
                        enforce_dual_regions = False
                    else:
                        basics_src = filtered_basics
                        momo_src = filtered_momo
                        vol_src = filtered_vol
                        fact_src = filtered_fact
                        contract_src = filtered_contract
                        st.caption(
                            "교집합 적용 · "
                            + ", ".join(common_regions[:5])
                            + (" 외" if len(common_regions) > 5 else "")
                        )
                else:
                    st.warning("선택한 기간에는 매매와 전월세가 모두 있는 시군구가 없습니다. 전체 지역을 사용합니다.")
                    enforce_dual_regions = False

    region_source = ensure_sigungu_named(fact_src)
    region_values = [] if region_source.empty else sorted(region_source[GROUP_COL].dropna().unique().tolist())
    region_options = ["전체"] + region_values if region_values else ["전체"]
    region_mode = st.radio(
        "지역 선택 모드",
        ("맞춤 선택", "전체"),
        index=0,
        horizontal=True,
        help="전체 데이터를 보거나 원하는 시군구만 골라 비교하세요.",
    )

    quick_selection: list[str] = []
    region_selection: list[str] = ["전체"]
    if region_mode == "맞춤 선택" and region_values:
        region_counts = (
            region_source[GROUP_COL]
            .value_counts()
            .reindex(region_values)
            .dropna()
        )
        top_regions = region_counts.head(12).index.tolist()
        if top_regions:
            quick_selection = st.multiselect(
                "빠른 선택 (최근 거래 상위)",
                options=top_regions,
                help="자주 분석하는 시군구를 빠르게 선택하세요.",
            )
        preferred_alias_groups = [
            ["서초구 전지역", "서울특별시 서초구", "서초구"],
            ["영등포구 전지역", "서울특별시 영등포구", "영등포구"],
            ["관악구 전지역", "서울특별시 관악구", "관악구"],
        ]
        region_candidates = region_options[1:]
        available_defaults: list[str] = []
        for aliases in preferred_alias_groups:
            direct_match = next((alias for alias in aliases if alias in region_candidates), None)
            if direct_match:
                target = direct_match
            else:
                matching_options = [
                    option
                    for option in region_candidates
                    if any(alias in option for alias in aliases)
                ]
                matching_options.sort(key=lambda opt: ("전지역" not in opt, len(opt)))
                target = matching_options[0] if matching_options else None
            if target and target not in available_defaults:
                available_defaults.append(target)
        seoul_defaults_map: dict[str, str] = {}
        for option in region_candidates:
            if not option.startswith("서울특별시"):
                continue
            parts = option.split()
            district = next(
                (token for token in parts[1:] if token.endswith("구")),
                parts[1] if len(parts) > 1 else option,
            )
            current = seoul_defaults_map.get(district)
            if current is None or ("전지역" in option and "전지역" not in current):
                seoul_defaults_map[district] = option
        seoul_default_list = [seoul_defaults_map[key] for key in sorted(seoul_defaults_map.keys())]

        default_regions = list(quick_selection) if quick_selection else list(available_defaults)
        if default_regions:
            default_regions.extend(opt for opt in seoul_default_list if opt not in default_regions)
        else:
            default_regions = seoul_default_list if seoul_default_list else region_candidates[:3]

        region_selection = st.multiselect(
            "지역 (시군구)",
            region_options[1:],
            default=default_regions,
            placeholder="시군구를 검색해 선택하세요.",
            help="비교할 시군구를 한 번에 선택하거나 검색해 추가하세요.",
        )
        region_selection = sorted(set(region_selection + quick_selection)) or ["전체"]

    if region_mode == "전체":
        region_selection = ["전체"]

    neighborhood_selection: list[str] = []
    selected_property_keys: list[str] = []
    property_label_map: dict[str, str] = {}

    if region_selection != ["전체"] and not fact_src.empty:
        option_source = filter_by_regions(fact_src, region_selection)
        if "eupmyeondong" in option_source.columns:
            neighborhoods_available = (
                option_source["eupmyeondong"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
            )
            neighborhoods_available = sorted(neighborhoods_available)
            if neighborhoods_available:
                neighborhood_selection = st.multiselect(
                    "상세 지역 (읍·동)",
                    options=["전체"] + neighborhoods_available,
                    default=["전체"],
                    key="neighborhood_selector",
                    help="선택한 시군구 내에서 분석할 읍·동을 좁혀보세요.",
                )

        cleaned_neighborhoods = [item for item in neighborhood_selection if item != "전체"]
        property_candidates = option_source
        if cleaned_neighborhoods:
            property_candidates = filter_by_neighborhoods(property_candidates, cleaned_neighborhoods)

        if "sigungu" not in property_candidates.columns and GROUP_COL in property_candidates.columns:
            property_candidates = property_candidates.copy()
            property_candidates["sigungu"] = property_candidates[GROUP_COL]

        required_property_columns = ["sigungu", "eupmyeondong", "lot_main", "lot_sub", "property_key"]
        available_property_columns = [col for col in required_property_columns if col in property_candidates.columns]
        missing_property_columns = [col for col in required_property_columns if col not in property_candidates.columns]
        if missing_property_columns:
            st.info(f"매물 정보를 만들기 위한 열이 부족합니다 ({', '.join(missing_property_columns)}). 매물 선택 기능을 생략합니다.")
        elif not property_candidates.empty:
            property_records = (
                property_candidates[available_property_columns]
                .dropna(subset=["property_key"])
                .drop_duplicates()
            )
            if not property_records.empty:
                property_records["property_label"] = property_records.apply(format_property_label, axis=1)
                property_records = property_records[property_records["property_label"].str.strip() != ""]
                if not property_records.empty:
                    property_records = property_records.sort_values("property_label")
                    property_label_map = dict(zip(property_records["property_label"], property_records["property_key"]))
                    property_selection = st.multiselect(
                        "개별 매물 (지번)",
                        options=["전체"] + list(property_label_map.keys()),
                        default=["전체"],
                        key="property_selector",
                        help="읍·동 내 특정 지번(동·호)을 선택해 세부 분석을 수행합니다.",
                    )
                    selected_property_keys = [property_label_map[label] for label in property_selection if label != "전체"]

    # Normalize selections to avoid treating "전체" as an active filter
    neighborhood_filters = [item for item in neighborhood_selection if item != "전체"]
    if property_label_map:
        property_label_lookup = {value: key for key, value in property_label_map.items()}
    else:
        property_label_lookup = {}

    st.markdown("### ⚖️ 전략 설정")
    strategy_mode = st.selectbox(
        "전략 프리셋",
        list(STRATEGY_PRESETS.keys()),
        index=2,
        help="선호하는 전략 유형을 선택하면 가중치가 자동 설정됩니다."
    )
    annual_rate = st.slider(
        "전월세 환산율(%)",
        1.0,
        10.0,
        5.5,
        0.1,
        help="보증금을 월세로 환산할 때 사용할 연 환산율입니다."
    )
    preset_weights = STRATEGY_PRESETS[strategy_mode]
    st.caption(
        f"현재 프리셋 가중치 · 저평가 {preset_weights['w1']:.2f} · 모멘텀 {preset_weights['w2']:.2f} · 안정성 {preset_weights['w3']:.2f}\n"
        "→ 종합 점수와 추천 Top3 순위에 즉시 반영됩니다."
    )
    st.caption(
        f"전월세 환산율 {annual_rate:.1f}% → Rent Gap과 월세 저평가율 계산에 사용됩니다."
    )

    st.markdown("### 🛠️ 기타 옵션")
    enable_alerts = st.checkbox("알림 조건 저장", value=False, help="설정한 조건을 저장해 이후 알림 기능과 연동합니다.")
    export_ready = st.checkbox("내보내기 옵션 표시", value=False, help="데이터 내보내기 기능을 미리 확인합니다.")

geo_centroids = load_geo_centroids()
policy_raw = load_policy_risk()
lifestyle_raw = load_lifestyle_scores()
sigungu_boundaries = load_sigungu_boundaries()

if geo_centroids.empty and not (CENTROID_PARQUET.exists() or CENTROID_CSV.exists()):
    warn_missing_resource(CENTROID_PARQUET if CENTROID_PARQUET.exists() else CENTROID_CSV, "시군구 좌표 데이터")
if policy_raw.empty and not POLICY_PATH.exists():
    warn_missing_resource(POLICY_PATH, "정책/금리 레퍼런스 (policy_risk.csv)")
if lifestyle_raw.empty and not LIFESTYLE_PATH.exists():
    warn_missing_resource(LIFESTYLE_PATH, "생활점수 레퍼런스 (lifestyle_scores.csv)")
if not sigungu_boundaries and not BOUNDARY_PATH.exists():
    warn_missing_resource(BOUNDARY_PATH, "시군구 경계 GeoJSON")

lifestyle = filter_by_regions(lifestyle_raw, region_selection)
policy_filtered = filter_by_regions(policy_raw, region_selection)
lifestyle = filter_by_neighborhoods(lifestyle, neighborhood_filters)
lifestyle = filter_by_properties(lifestyle, selected_property_keys)
policy_filtered = filter_by_neighborhoods(policy_filtered, neighborhood_filters)
policy_filtered = filter_by_properties(policy_filtered, selected_property_keys)
aggregate_charts = region_mode == "전체"

if export_ready:
    st.sidebar.info("내보내기 기능은 업데이트 준비 중입니다.")
    push_log("내보내기 기능은 아직 준비되지 않았습니다.", "info")
if enable_alerts:
    st.sidebar.info("알림 조건 저장은 곧 제공될 예정입니다.")
    push_log("알림 조건 저장 기능은 곧 제공될 예정입니다.", "info")

with st.sidebar.expander("📄 메시지 로그", expanded=False):
    logs = st.session_state.get(LOG_STORAGE_KEY, [])
    if logs:
        for entry in logs[-50:]:
            st.markdown(f"`{entry['time']}` [{entry['level']}] {entry['message']}")
    else:
        st.caption("최근 메시지가 없습니다.")

if "strategy_marker" not in st.session_state or st.session_state["strategy_marker"] != strategy_mode:
    for key, val in STRATEGY_PRESETS[strategy_mode].items():
        st.session_state[f"weight_{key}"] = val
    st.session_state["strategy_marker"] = strategy_mode

current_w1 = st.session_state.get("weight_w1", STRATEGY_PRESETS[strategy_mode]["w1"])
current_w2 = st.session_state.get("weight_w2", STRATEGY_PRESETS[strategy_mode]["w2"])
current_w3 = st.session_state.get("weight_w3", STRATEGY_PRESETS[strategy_mode]["w3"])

basics = filter_by_regions(filter_recent_months(basics_src, recent_months), region_selection)
momo = filter_by_regions(filter_recent_months(momo_src, recent_months), region_selection)
vol = filter_by_regions(filter_recent_months(vol_src, recent_months), region_selection)
fact = filter_by_regions(filter_recent_months(fact_src, recent_months), region_selection)
contract_fact = filter_by_regions(filter_recent_months(contract_src, recent_months), region_selection)

basics = filter_by_neighborhoods(basics, neighborhood_filters)
basics = filter_by_properties(basics, selected_property_keys)
momo = filter_by_neighborhoods(momo, neighborhood_filters)
momo = filter_by_properties(momo, selected_property_keys)
vol = filter_by_neighborhoods(vol, neighborhood_filters)
vol = filter_by_properties(vol, selected_property_keys)
fact = filter_by_neighborhoods(fact, neighborhood_filters)
fact = filter_by_properties(fact, selected_property_keys)
contract_fact = filter_by_neighborhoods(contract_fact, neighborhood_filters)
contract_fact = filter_by_properties(contract_fact, selected_property_keys)

if contract_fact.empty and not contract_src.empty:
    # 최근 N개월에 해당하는 전월세 데이터가 없을 때는 기간 필터를 완화해 전체 계약 데이터로 대체한다.
    contract_fact = filter_by_regions(contract_src, region_selection)
    contract_fact = filter_by_neighborhoods(contract_fact, neighborhood_filters)
    contract_fact = filter_by_properties(contract_fact, selected_property_keys)

for frame in (basics, momo, vol, contract_fact):
    if "index" in frame.columns:
        frame.drop(columns=["index"], inplace=True)

latest_momo = pd.DataFrame()
if not momo.empty:
    if "YYYYMM" in momo.columns:
        latest_momo = momo.sort_values("YYYYMM").groupby(GROUP_COL).tail(1)
    else:
        latest_momo = momo.copy()

latest_basics = pd.DataFrame()
if not basics.empty:
    if "YYYYMM" in basics.columns:
        latest_basics = basics.sort_values("YYYYMM").groupby(GROUP_COL).tail(1)
    else:
        latest_basics = basics.copy()

latest_vol = pd.DataFrame()
if not vol.empty:
    if "YYYYMM" in vol.columns:
        latest_vol = vol.sort_values("YYYYMM").groupby(GROUP_COL).tail(1)
    else:
        latest_vol = vol.copy()

uv = compute_undervaluation(fact)
npremium = compute_new_premium(fact)
floor = compute_floor_premium(fact)
contract_metrics_source = contract_fact if not contract_fact.empty else fact
rent_summary = compute_rent_metrics(contract_metrics_source, annual_rate)
risk_timeseries = compute_dom_metrics(contract_metrics_source)
comp_score = compute_composite_score(uv, momo, vol, current_w1, current_w2, current_w3)

risk_filtered = filter_by_regions(risk_timeseries, region_selection)

comp_columns = [
    GROUP_COL,
    "composite_score",
    "score_undervalue",
    "score_momentum",
    "score_stability",
    "stability",
]
compare_base = comp_score[comp_columns].copy() if not comp_score.empty else pd.DataFrame(columns=comp_columns)
if not uv.empty:
    compare_base = compare_base.merge(uv[[GROUP_COL,"undervalue_rate"]], on=GROUP_COL, how="outer")
if not latest_momo.empty and "mom_3m" in latest_momo.columns:
    compare_base = compare_base.merge(latest_momo[[GROUP_COL,"mom_3m"]], on=GROUP_COL, how="outer")
if not npremium.empty:
    compare_base = compare_base.merge(npremium[[GROUP_COL,"new_premium"]], on=GROUP_COL, how="outer")
if not floor.empty:
    compare_base = compare_base.merge(floor[[GROUP_COL,"floor_premium"]], on=GROUP_COL, how="outer")
if not rent_summary.empty:
    compare_base = compare_base.merge(rent_summary[[GROUP_COL,"rent_undervalue_rate","rent_gap_mean"]], on=GROUP_COL, how="outer")
if not lifestyle.empty:
    life_cols = [GROUP_COL]
    for col in ["school_score", "amenity_score", "lifestyle_comment"]:
        if col in lifestyle.columns and col not in life_cols:
            life_cols.append(col)
    compare_base = compare_base.merge(lifestyle[life_cols], on=GROUP_COL, how="outer")
if not risk_timeseries.empty:
    latest_risk = risk_timeseries.sort_values("YYYYMM").groupby(GROUP_COL).tail(1)
    risk_cols = [
        "dom_median",
        "cancel_rate_dom",
        "vacancy_proxy",
        "contract_months_avg",
        "contract_months_median",
        "contract_renewal_rate",
        "contract_extension_rate",
        "contract_stability",
    ]
    available_cols = [GROUP_COL] + [col for col in risk_cols if col in latest_risk.columns]
    compare_base = compare_base.merge(latest_risk[available_cols], on=GROUP_COL, how="outer")
if not policy_raw.empty:
    latest_policy = policy_raw.sort_values("YYYYMM").groupby(GROUP_COL).tail(1)
    compare_base = compare_base.merge(
        latest_policy[[GROUP_COL, "interest_rate", "regulation_flag", "policy_comment"]],
        on=GROUP_COL,
        how="outer",
    )
if not latest_basics.empty:
    basics_cols = [c for c in ["avg_yield_pct", "avg_p_per_m2", "cancel_rate", "txn_cnt"] if c in latest_basics.columns]
    if basics_cols:
        compare_base = compare_base.merge(
            latest_basics[[GROUP_COL] + basics_cols].rename(columns={"cancel_rate": "cancel_rate_basics"}),
            on=GROUP_COL,
            how="outer",
        )
if not latest_vol.empty:
    vol_cols = [c for c in ["cv_p_per_m2", "cancel_rate"] if c in latest_vol.columns]
    if vol_cols:
        compare_base = compare_base.merge(
            latest_vol[[GROUP_COL] + vol_cols].rename(columns={"cancel_rate": "cancel_rate_vol"}),
            on=GROUP_COL,
            how="outer",
        )
if not contract_metrics_source.empty and GROUP_COL in contract_metrics_source.columns:
    contract_summary = (
        contract_metrics_source.groupby(GROUP_COL, as_index=False)[
            ["contract_stability_score", "contract_is_renewal", "contract_extension_flag"]
        ]
        .mean()
        .rename(
            columns={
                "contract_stability_score": "contract_stability_smooth",
                "contract_is_renewal": "contract_renewal_ratio",
                "contract_extension_flag": "contract_extension_ratio",
            }
        )
    )
    compare_base = compare_base.merge(contract_summary, on=GROUP_COL, how="outer")

if "cancel_rate" not in compare_base.columns:
    compare_base["cancel_rate"] = np.nan
if "cancel_rate_vol" in compare_base.columns:
    compare_base["cancel_rate"] = compare_base["cancel_rate"].fillna(compare_base["cancel_rate_vol"])
if "cancel_rate_basics" in compare_base.columns:
    compare_base["cancel_rate"] = compare_base["cancel_rate"].fillna(compare_base["cancel_rate_basics"])
if "cancel_rate_dom" in compare_base.columns:
    compare_base["cancel_rate"] = compare_base["cancel_rate"].fillna(compare_base["cancel_rate_dom"])
numeric_columns = compare_base.select_dtypes(include=[np.number]).columns
if "vacancy_proxy" in numeric_columns:
    compare_base["vacancy_proxy"] = compare_base["vacancy_proxy"].fillna(0.0)

compare_base = compare_base.dropna(how="all", subset=compare_base.columns.difference([GROUP_COL]))

if "contract_stability" in compare_base.columns:
    compare_base["contract_stability"] = compare_base["contract_stability"].replace(0.0, np.nan)
if "contract_stability_smooth" in compare_base.columns:
    base_series = compare_base.get("contract_stability")
    if base_series is None:
        base_series = pd.Series(np.nan, index=compare_base.index, dtype=float)
    compare_base["contract_stability"] = base_series.fillna(compare_base["contract_stability_smooth"])
    compare_base.drop(columns=["contract_stability_smooth"], inplace=True, errors="ignore")
if "contract_renewal_ratio" in compare_base.columns and "contract_renewal_rate" in compare_base.columns:
    compare_base["contract_renewal_rate"] = compare_base["contract_renewal_rate"].replace(0.0, np.nan).fillna(compare_base["contract_renewal_ratio"])
    compare_base.drop(columns=["contract_renewal_ratio"], inplace=True, errors="ignore")
if "contract_extension_ratio" in compare_base.columns and "contract_extension_rate" in compare_base.columns:
    compare_base["contract_extension_rate"] = compare_base["contract_extension_rate"].replace(0.0, np.nan).fillna(compare_base["contract_extension_ratio"])
    compare_base.drop(columns=["contract_extension_ratio"], inplace=True, errors="ignore")

persona_source_raw = compare_base.copy()

compare_base = compare_base.copy()
numeric_columns = [col for col in numeric_columns if col in compare_base.columns]
compare_base[numeric_columns] = compare_base[numeric_columns].fillna(0.0)
non_numeric_columns = compare_base.columns.difference(numeric_columns)
compare_base[non_numeric_columns] = compare_base[non_numeric_columns].fillna("")

baseline_reference: dict[str, float] = {}
for metric in ["undervalue_rate", "mom_3m", "new_premium", "floor_premium", "rent_undervalue_rate"]:
    if metric in compare_base.columns:
        baseline = compare_base[metric].mean()
        baseline_reference[metric] = baseline
        compare_base[f"{metric}_delta"] = compare_base[metric] - baseline

persona_profiles = compute_persona_profiles(persona_source_raw)
if persona_profiles.empty:
    persona_source = compare_base.copy()
else:
    persona_source = persona_profiles.copy()

setup_tab, explore_tab, refine_tab, validate_tab, decide_tab = st.tabs([
    "🛠️ Setup",
    "🧭 Explore",
    "💡 Refine",
    "🧪 Validate",
    "🏁 Decide",
])

with setup_tab:
    st.subheader("분석 설정 요약")
    summary_cols = st.columns(3)
    summary_cols[0].metric("자산 유형", src_type.replace("_", " "))
    summary_cols[1].metric("선택 기간", period_choice)
    summary_cols[2].metric("전략 프리셋", strategy_mode)
    st.caption(f"전월세 환산율 {annual_rate:.1f}% · Rent Gap 계산에 적용됩니다.")

    region_label = "전체" if region_selection == ["전체"] else ", ".join(region_selection[:5]) + (" 외" if len(region_selection) > 5 else "")
    neighborhood_label = "전체" if not neighborhood_filters else ", ".join(neighborhood_filters[:5]) + (" 외" if len(neighborhood_filters) > 5 else "")
    property_label = "전체" if not selected_property_keys else f"{len(selected_property_keys)}개 선택"

    st.markdown("**선택 필터**")
    st.write(f"- 지역 모드: {region_mode}")
    st.write(f"- 시군구: {region_label}")
    st.write(f"- 읍·동: {neighborhood_label}")
    st.write(f"- 개별 매물: {property_label}")

    st.markdown("**데이터 현황 (필터 적용 후)**")
    data_cols = st.columns(4)
    data_cols[0].metric("거래 건수", f"{len(fact):,}")
    data_cols[1].metric("월간 지표", f"{len(basics):,}")
    data_cols[2].metric("모멘텀", f"{len(momo):,}")
    data_cols[3].metric("변동성", f"{len(vol):,}")

    missing_msgs: list[str] = []
    if fact.empty:
        missing_msgs.append("거래 데이터가 없습니다.")
    if basics.empty:
        missing_msgs.append("월간 기본 지표가 없습니다.")
    if momo.empty:
        missing_msgs.append("모멘텀 지표가 없습니다.")
    if vol.empty:
        missing_msgs.append("변동성 지표가 없습니다.")
    if missing_msgs:
        st.warning(" / ".join(missing_msgs))
    else:
        st.success("데이터 준비 완료! 다음 단계인 Explore에서 패턴을 확인하세요.")

    st.caption("필터를 바꾸면 이 요약이 즉시 갱신됩니다.")

with explore_tab:
    st.header("① 시장 흐름 탐색")
    st.caption("가격·모멘텀·계약 지표를 확인해 후보 지역의 흐름을 파악하세요.")
    sub_tabs = st.tabs(["가격 흐름", "모멘텀", "리스크 & 계약", "거래 미리보기"])
    with sub_tabs[0]:
        st.subheader("가격 흐름")
        st.caption("선택한 시군구의 월별 평균 ㎡당 가격 흐름으로, 최신 기간 데이터를 집계합니다.")
        basics_display = aggregate_regions_for_chart(basics, aggregate_charts)
        render_time_series_chart(basics_display, "avg_p_per_m2", "평균 ㎡당 가격")
        with st.expander("원본 데이터 보기", expanded=False):
            st.dataframe(basics.head() if not basics.empty else pd.DataFrame())

        st.markdown("### Top 인사이트 목록")
        st.caption("핵심 지표별 상위 지역을 최신 값 기준으로 정렬했습니다.")
        insight_cols = st.columns(2)
        with insight_cols[0]:
            st.markdown("**저평가 Top5**")
            render_ranked_list(uv, "undervalue_rate", "저평가율")
        with insight_cols[1]:
            st.markdown("**모멘텀 Top5**")
            render_ranked_list(latest_momo, "mom_3m", "3개월 모멘텀")
        insight_cols = st.columns(2)
        with insight_cols[0]:
            st.markdown("**신축 프리미엄 Top5**")
            render_ranked_list(npremium, "new_premium", "신축 프리미엄")
        with insight_cols[1]:
            st.markdown("**층 프리미엄 Top5**")
            render_ranked_list(floor, "floor_premium", "층 프리미엄")

        st.markdown("### 지역 Heatmap (β)")
        st.caption("선택한 지표 값을 지도 색상 또는 버블 크기로 시각화합니다.")
        map_metric_labels = {
            "undervalue_rate": "저평가율",
            "mom_3m": "3개월 모멘텀",
            "new_premium": "신축 프리미엄",
            "floor_premium": "층 프리미엄",
            "rent_undervalue_rate": "월세 저평가율",
            "composite_score": "종합 점수",
            "contract_stability": "계약 안정성",
            "contract_renewal_rate": "갱신 비율",
        }
        percent_metrics = {
            "undervalue_rate",
            "mom_3m",
            "new_premium",
            "floor_premium",
            "rent_undervalue_rate",
            "contract_stability",
            "contract_renewal_rate",
        }
        available_map_options = {k: v for k, v in map_metric_labels.items() if k in compare_base.columns}
        if geo_centroids.empty and not sigungu_boundaries:
            st.info("시군구 좌표/경계 데이터가 필요합니다. `data/reference/sigungu_centroids.(csv|parquet)` 또는 행정동 GeoJSON을 준비하세요.")
        elif not available_map_options:
            st.info("지도에 표시할 지표를 찾을 수 없습니다.")
        else:
            map_metric_key = st.selectbox(
                "지도에 표시할 지표",
                list(available_map_options.keys()),
                format_func=lambda k: available_map_options[k],
                key="map_metric_selector",
            )
            map_label = available_map_options[map_metric_key]

            map_modes: list[str] = []
            if not geo_centroids.empty:
                map_modes.append("버블")
            if sigungu_boundaries:
                map_modes.append("경계 Choropleth")

            if not map_modes:
                st.info("필요한 지도 데이터가 부족합니다.")
            else:
                map_style_choice = None
                map_style_lookup = {
                    "라이트": "mapbox://styles/mapbox/light-v11",
                    "다크": "mapbox://styles/mapbox/dark-v11",
                    "스트리트": "mapbox://styles/mapbox/streets-v12",
                    "위성": "mapbox://styles/mapbox/satellite-streets-v12",
                }
                if "경계 Choropleth" in map_modes:
                    map_style_choice = st.selectbox(
                        "배경 지도",
                        list(map_style_lookup.keys()),
                        index=0,
                        key="map_style_selector",
                    )

                default_index = 0
                map_mode = map_modes[default_index] if len(map_modes) == 1 else st.radio(
                    "지도 형태",
                    map_modes,
                    horizontal=True,
                    key="map_mode_selector",
                )

                if map_mode == "버블" or (len(map_modes) == 1 and map_modes[0] == "버블"):
                    map_data = prepare_map_data(compare_base, geo_centroids, map_metric_key)
                    if map_data.empty:
                        st.info("선택한 지표에 대해 버블 지도를 계산할 수 없습니다.")
                    else:
                        map_data = map_data.copy()
                        metric_values = pd.to_numeric(map_data[map_metric_key], errors="coerce")
                        domain_min, domain_max = _compute_metric_domain(metric_values.dropna())
                        diverging = (map_metric_key in percent_metrics) or (domain_min < 0.0 < domain_max)
                        display_func = format_percent if map_metric_key in percent_metrics else (lambda v: f"{v:.1f}")

                        color_values = metric_values.fillna(0.0)
                        map_data["color"] = [
                            _colorize_rgba(val, domain_min, domain_max, diverging)
                            if not pd.isna(orig)
                            else [200, 200, 200, 80]
                            for val, orig in zip(color_values, metric_values)
                        ]

                        use_abs_size = (map_metric_key in percent_metrics) or (domain_min < 0.0 < domain_max)
                        size_field = map_metric_key
                        if use_abs_size:
                            size_field = "_metric_size"
                            map_data[size_field] = metric_values.abs()

                        size_series = pd.to_numeric(map_data[size_field], errors="coerce")
                        if size_series.dropna().empty:
                            map_data["radius"] = 6000.0
                        else:
                            size_domain_min, size_domain_max = _compute_metric_domain(size_series.dropna())
                            if math.isclose(size_domain_min, size_domain_max, rel_tol=1e-9, abs_tol=1e-9):
                                map_data["radius"] = 6000.0
                            else:
                                normalized = (size_series - size_domain_min) / (size_domain_max - size_domain_min)
                                normalized = normalized.clip(0.0, 1.0).fillna(0.3)
                                map_data["radius"] = 3000.0 + normalized * 9000.0

                        map_data["tooltip_value"] = metric_values.apply(lambda v: display_func(v) if not pd.isna(v) else "-")

                        view_state = compute_view_state_from_data(point_df=map_data)
                        scatter_layer = pdk.Layer(
                            "ScatterplotLayer",
                            map_data,
                            get_position="[longitude, latitude]",
                            get_radius="radius",
                            get_fill_color="color",
                            get_line_color=[255, 255, 255, 200],
                            pickable=True,
                            auto_highlight=True,
                        )

                        layers = [scatter_layer]
                        if sigungu_boundaries:
                            boundary_geojson = {
                                "type": "FeatureCollection",
                                "features": list(sigungu_boundaries.values()),
                            }
                            boundary_layer = pdk.Layer(
                                "GeoJsonLayer",
                                boundary_geojson,
                                stroked=True,
                                filled=False,
                                get_line_color=[90, 90, 90, 160],
                                line_width_min_pixels=1.2,
                            )
                            layers.append(boundary_layer)

                        tooltip = {
                            "html": f"<b>{{{GROUP_COL}}}</b><br>{map_label}: {{tooltip_value}}",
                            "style": {"backgroundColor": "#2c3e50", "color": "white"},
                        }
                        deck = pdk.Deck(
                            layers=layers,
                            initial_view_state=view_state,
                            tooltip=tooltip,
                            map_style=map_style_lookup.get(map_style_choice or "라이트"),
                        )
                        st.pydeck_chart(deck)
                else:
                    choropleth_df = compute_choropleth_dataset(
                        sigungu_boundaries,
                        compare_base,
                        map_metric_key,
                        diverging_metrics=percent_metrics,
                    )
                    if choropleth_df.empty:
                        st.info("선택한 지표에 대해 경계 데이터를 계산할 수 없습니다.")
                    else:
                        display_func = format_percent if map_metric_key in percent_metrics else (lambda v: f"{v:.1f}")
                        choropleth_df = choropleth_df.copy()
                        choropleth_df["tooltip"] = choropleth_df["value"].apply(display_func)
                        view_state = compute_view_state_from_data(poly_df=choropleth_df)
                        layer = pdk.Layer(
                            "PolygonLayer",
                            choropleth_df,
                            get_polygon="polygon",
                            get_fill_color="color",
                            get_line_color=[255, 255, 255],
                            opacity=0.65,
                            pickable=True,
                            auto_highlight=True,
                        )
                        tooltip = {
                            "html": f"<b>{{시군구}}</b><br>{map_label}: {{tooltip}}",
                            "style": {"backgroundColor": "#2c3e50", "color": "white"},
                        }
                        deck = pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            tooltip=tooltip,
                            map_style=map_style_lookup.get(map_style_choice or "라이트"),
                        )
                        st.pydeck_chart(deck)
    with sub_tabs[1]:
        st.subheader("모멘텀 추이")
        st.caption("`monthly_momentum`의 3개월 모멘텀 지표(`mom_3m`)를 시군구별로 비교합니다.")
        momo_display = aggregate_regions_for_chart(momo, aggregate_charts)
        render_time_series_chart(momo_display, "mom_3m", "3개월 모멘텀")
        if not momo.empty and {"YYYYMM", GROUP_COL}.issubset(momo.columns):
            z_source = ensure_sigungu_named(momo)
            if "mom_3m" in z_source.columns:
                z_base = z_source.dropna(subset=["mom_3m"]).copy()
                if not z_base.empty:
                    z_base["YYYYMM"] = z_base["YYYYMM"].astype("Int64").astype(str)
                    z_base["zscore"] = (
                        z_base.groupby(GROUP_COL)["mom_3m"].transform(
                            lambda s: ((s - s.mean()) / s.std(ddof=0)) if s.std(ddof=0) else 0
                        )
                    )
                    z_base["zscore"] = z_base["zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    if not z_base.empty:
                        st.markdown("**지역별 모멘텀 z-score Heatmap**")
                        heat = z_base[[GROUP_COL, "YYYYMM", "zscore"]]
                        heat_chart = alt.Chart(heat).mark_rect().encode(
                            x=alt.X("YYYYMM:O", title="계약월"),
                            y=alt.Y(f"{GROUP_COL}:N", title="지역"),
                            color=alt.Color(
                                "zscore:Q",
                                scale=alt.Scale(scheme="redblue", domainMid=0),
                                title="z-score",
                            ),
                            tooltip=[GROUP_COL, "YYYYMM", alt.Tooltip("zscore:Q", format=".2f")],
                        )
                        st.altair_chart(heat_chart, use_container_width=True)
        with st.expander("원본 데이터 보기", expanded=False):
            st.dataframe(momo.head() if not momo.empty else pd.DataFrame())
    with sub_tabs[2]:
        st.subheader("변동성 / 취소율")
        st.caption("가격 변동계수와 계약 데이터에서 계산한 DOM·취소율·공실 지표의 추세를 보여줍니다.")
        vol_display = aggregate_regions_for_chart(vol, aggregate_charts)
        if "cv_p_per_m2" in vol_display.columns:
            render_time_series_chart(vol_display, "cv_p_per_m2", "가격 변동계수")
        else:
            st.info("가격 변동계수 데이터를 찾을 수 없습니다.")
        risk_display = aggregate_regions_for_chart(risk_filtered, aggregate_charts)
        if not risk_display.empty:
            if "dom_median" in risk_display.columns:
                render_time_series_chart(risk_display, "dom_median", "중위 DOM(일)", axis_format=".0f")
            if "cancel_rate_dom" in risk_display.columns:
                render_time_series_chart(risk_display, "cancel_rate_dom", "취소율(계약 취소)", axis_format=".0%")
            if "vacancy_proxy" in risk_display.columns:
                render_time_series_chart(risk_display, "vacancy_proxy", "공실 Proxy", axis_format=".0%")
            if "contract_stability" in risk_display.columns:
                render_time_series_chart(risk_display, "contract_stability", "계약 안정성", axis_format=".0%")
            if "contract_months_avg" in risk_display.columns:
                render_time_series_chart(risk_display, "contract_months_avg", "평균 계약기간(월)", axis_format=".1f")
            if "contract_renewal_rate" in risk_display.columns:
                render_time_series_chart(risk_display, "contract_renewal_rate", "갱신 비율", axis_format=".0%")
            if "contract_extension_rate" in risk_display.columns:
                render_time_series_chart(risk_display, "contract_extension_rate", "갱신요구권 사용률", axis_format=".0%")
        else:
            st.info("DOM/취소율 시계열 데이터를 찾을 수 없습니다.")

        if not policy_filtered.empty:
            st.markdown("**정책·금리 모니터**")
            st.caption("선택 지역의 최신 금리·규제 현황과 정책 메모를 함께 확인하세요.")
            policy_latest = policy_filtered.sort_values("YYYYMM").groupby(GROUP_COL).tail(1)
            if not policy_latest.empty:
                policy_view = policy_latest[[GROUP_COL, "YYYYMM", "interest_rate", "regulation_flag", "policy_comment"]].copy()
                policy_view["YYYYMM"] = policy_view["YYYYMM"].astype("Int64").astype(str)
                if "interest_rate" in policy_view.columns:
                    policy_view["interest_rate"] = policy_view["interest_rate"].apply(
                        lambda v: f"{v:.2f}%" if not pd.isna(v) else "-"
                    )
                if "regulation_flag" in policy_view.columns:
                    policy_view["regulation_flag"] = policy_view["regulation_flag"].replace(
                        {
                            "1": "규제",
                            "0": "비규제",
                            "True": "규제",
                            "False": "비규제",
                        }
                    )
                policy_view.rename(columns={
                    GROUP_COL: "시군구",
                    "YYYYMM": "기준월",
                    "interest_rate": "금리(%)",
                    "regulation_flag": "규제 여부",
                    "policy_comment": "정책 메모",
                }, inplace=True)
                st.table(policy_view)

            policy_timeline = policy_filtered.dropna(subset=["interest_rate"]).copy()
            if not policy_timeline.empty:
                policy_timeline["YYYYMM"] = policy_timeline["YYYYMM"].astype("Int64").astype(str)
                policy_timeline["regulation_flag"] = policy_timeline["regulation_flag"].replace(
                    {
                        "1": "규제",
                        "0": "비규제",
                        "True": "규제",
                        "False": "비규제",
                    }
                )
                policy_chart = alt.Chart(policy_timeline).mark_line(point=True).encode(
                    x=alt.X("YYYYMM:O", title="정책월"),
                    y=alt.Y("interest_rate:Q", title="금리(%)"),
                    color=alt.Color(f"{GROUP_COL}:N", title="시군구"),
                    tooltip=[
                        alt.Tooltip(f"{GROUP_COL}:N", title="시군구"),
                        alt.Tooltip("YYYYMM:O", title="정책월"),
                        alt.Tooltip("interest_rate:Q", title="금리(%)", format=".2f"),
                        alt.Tooltip("regulation_flag:N", title="규제"),
                        alt.Tooltip("policy_comment:N", title="정책 메모"),
                    ],
                ).properties(height=260)
                st.altair_chart(policy_chart, use_container_width=True)
            else:
                st.info("금리 정보를 포함한 정책 데이터가 없습니다.")

        with st.expander("원본 데이터 보기", expanded=False):
            st.dataframe(vol.head() if not vol.empty else pd.DataFrame())
            st.dataframe(risk_filtered.head() if not risk_filtered.empty else pd.DataFrame())
            st.dataframe(policy_filtered.head() if not policy_filtered.empty else pd.DataFrame())
    with sub_tabs[3]:
        st.subheader("거래 원본 미리보기")
        st.caption("현재 필터 조건이 적용된 거래 데이터를 최대 100건까지 확인합니다.")
        st.dataframe(fact.head(100) if not fact.empty else pd.DataFrame())

    

    st.info("후보를 좁히려면 **'💡 Refine'** 탭으로 이동하세요.")

with refine_tab:
    persona_focus_candidates: list[dict[str, object]] = []
    composite_scores = pd.DataFrame()
    cluster_scores = pd.DataFrame()
    ml_scores = pd.DataFrame()
    rent_scores = pd.DataFrame()

    st.markdown("### 1️⃣ 종합 점수 기반 추천")
    st.caption("저평가·모멘텀·안정성 가중 합산으로 계산한 `composite_score` 상위 지역입니다.")
    if comp_score.empty:
        st.info("필요한 데이터가 부족해 종합 점수를 계산할 수 없습니다.")
    else:
        st.caption(
            f"가중치 설정 · 저평가 {current_w1:.2f}, 모멘텀 {current_w2:.2f}, 안정성 {current_w3:.2f}")
        render_ranked_list(comp_score, "composite_score", "종합 점수", formatter=format_score, axis_format=".1f")
        composite_scores = comp_score[[GROUP_COL, "composite_score"]].copy()

    st.markdown("### 2️⃣ 군집 분석 기반 추천")
    cluster_features = [col for col in ["avg_p_per_m2", "mom_3m", "stability"] if col in persona_source.columns]
    clustered_preview = pd.DataFrame()
    summary_preview = pd.DataFrame()
    if len(cluster_features) >= 2:
        available_rows = persona_source.dropna(subset=cluster_features)
        unique_regions = available_rows[GROUP_COL].nunique() if GROUP_COL in available_rows.columns else 0
        if unique_regions >= 2:
            cluster_count = st.slider(
                "군집 개수",
                min_value=2,
                max_value=min(8, unique_regions),
                value=min(4, unique_regions),
                step=1,
                key="reco_kmeans_cluster_count",
            )
            clustered_preview, summary_preview = run_kmeans_clustering(
                persona_source,
                cluster_features,
                clusters=cluster_count,
            )
        if clustered_preview.empty or summary_preview.empty:
            st.info("군집을 계산할 수 없습니다. 필터 범위를 넓혀보거나 입력 지표를 확인하세요.")
        else:
            axis_options = cluster_features.copy()
            label_map = {
                "avg_p_per_m2": "평균 ㎡당가",
                "mom_3m": "3개월 모멘텀",
                "stability": "안정성",
            }
            x_default = axis_options.index("avg_p_per_m2") if "avg_p_per_m2" in axis_options else 0
            x_feature = st.selectbox("군집 산점도 X축", axis_options, index=x_default, key="reco_kmeans_axis_x", format_func=lambda c: label_map.get(c, c))
            y_candidates = [f for f in axis_options if f != x_feature] or axis_options
            y_feature = st.selectbox("Y축", y_candidates, index=0, key="reco_kmeans_axis_y", format_func=lambda c: label_map.get(c, c))

            chart = (
                alt.Chart(clustered_preview)
                .mark_circle(size=80, opacity=0.85, stroke="#ffffff", strokeWidth=0.4)
                .encode(
                    x=alt.X(f"{x_feature}:Q", title=label_map.get(x_feature, x_feature)),
                    y=alt.Y(f"{y_feature}:Q", title=label_map.get(y_feature, y_feature)),
                    color=alt.Color("cluster:N", title="클러스터"),
                    tooltip=[GROUP_COL, "cluster"] + cluster_features,
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)

            summary_display = summary_preview[["cluster"] + cluster_features + ["kmeans_score"]].copy()
            summary_display.rename(columns={
                "cluster": "클러스터",
                "avg_p_per_m2": "평균 ㎡당가",
                "mom_3m": "최근 3개월 모멘텀",
                "stability": "안정성",
                "kmeans_score": "추천 점수",
            }, inplace=True)
            st.dataframe(summary_display.reset_index(drop=True), use_container_width=True, key="reco_kmeans_summary")

            score_map = summary_preview.set_index("cluster")["kmeans_score"]
            cluster_scores = clustered_preview[[GROUP_COL]].copy()
            cluster_scores["kmeans_score"] = clustered_preview["cluster"].map(score_map)

            top_cluster_row = summary_preview.sort_values("kmeans_score", ascending=False).iloc[0]
            best_cluster_id = int(top_cluster_row["cluster"])
            cluster_points = clustered_preview[clustered_preview["cluster"] == best_cluster_id]
            if not cluster_points.empty:
                top_candidate = cluster_points.sort_values("avg_p_per_m2").iloc[0]
                notes: list[str] = []
                if "mom_3m" in top_candidate:
                    notes.append(f"모멘텀 {format_percent(top_candidate['mom_3m'])}")
                if "stability" in top_candidate:
                    notes.append(f"안정성 {format_percent(top_candidate['stability'])}")
                if "avg_p_per_m2" in top_candidate:
                    notes.append(f"㎡당 {_format_float(top_candidate['avg_p_per_m2'], 1)}만원")
                best_candidates = cluster_points.sort_values("avg_p_per_m2").head(5)[[GROUP_COL] + cluster_features].copy()
                best_candidates.rename(columns={
                    "avg_p_per_m2": "평균 ㎡당가",
                    "mom_3m": "3개월 모멘텀",
                    "stability": "안정성",
                }, inplace=True)
                for col in best_candidates.columns:
                    if col in {"3개월 모멘텀", "안정성"}:
                        best_candidates[col] = best_candidates[col].apply(format_percent)
                    elif col == "평균 ㎡당가":
                        best_candidates[col] = best_candidates[col].apply(lambda v: _format_float(v, 1))
                st.dataframe(best_candidates.reset_index(drop=True), use_container_width=True, key="reco_kmeans_candidates")

    st.markdown("### 3️⃣ ML 저평가 탐지")
    st.caption("랜덤포레스트가 예측한 ㎡당가와 실제값 차이를 기반으로 저평가 가능성이 높은 지역을 제시합니다.")
    rf_features_all = [
        "undervalue_rate",
        "mom_3m",
        "stability",
        "avg_yield_pct",
        "rent_undervalue_rate",
        "vacancy_proxy",
        "composite_score",
    ]
    rf_target = "avg_p_per_m2"
    rf_data, rf_model, rf_features = compute_random_forest_undervaluation(persona_source, rf_features_all, rf_target)
    if rf_data.empty or rf_model is None or not rf_features:
        st.info("모델 학습을 위한 지표가 부족합니다. 평균 ㎡당가와 모멘텀·안정성 지표를 확인하세요.")
    else:
        ml_scores = rf_data[[GROUP_COL, "ml_undervalue_rate"]].copy()
        top_limit = st.slider("추천 개수", min_value=3, max_value=10, value=5, step=1, key="rf_reco_topn")
        rf_view = rf_data.sort_values("ml_undervalue_rate", ascending=False).head(top_limit).copy()
        rename_cols = {
            GROUP_COL: "시군구",
            "avg_p_per_m2": "실제 ㎡당가",
            "predicted": "모델 예측 ㎡당가",
            "ml_undervalue_rate": "저평가율(ML)",
            "mom_3m": "3개월 모멘텀",
            "stability": "안정성",
            "avg_yield_pct": "평균 Yield",
        }
        if "avg_p_per_m2" in rf_view.columns:
            rf_view["avg_p_per_m2"] = rf_view["avg_p_per_m2"].apply(lambda v: _format_float(v, 1))
        if "predicted" in rf_view.columns:
            rf_view["predicted"] = rf_view["predicted"].apply(lambda v: _format_float(v, 1))
        if "ml_undervalue_rate" in rf_view.columns:
            rf_view["ml_undervalue_rate"] = rf_view["ml_undervalue_rate"].apply(format_percent)
        if "mom_3m" in rf_view.columns:
            rf_view["mom_3m"] = rf_view["mom_3m"].apply(format_percent)
        if "avg_yield_pct" in rf_view.columns:
            rf_view["avg_yield_pct"] = rf_view["avg_yield_pct"].apply(lambda v: _format_float(v, 1))
        if "stability" in rf_view.columns:
            rf_view["stability"] = rf_view["stability"].apply(lambda v: _format_float(v, 2))
        st.dataframe(rf_view.rename(columns=rename_cols).reset_index(drop=True))

        importance_df = pd.DataFrame(
            {
                "feature": rf_features,
                "importance": rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        if not importance_df.empty:
            label_map = {
                "undervalue_rate": "저평가율",
                "mom_3m": "3개월 모멘텀",
                "stability": "안정성",
                "avg_yield_pct": "평균 Yield",
                "rent_undervalue_rate": "월세 저평가율",
                "vacancy_proxy": "공실 Proxy",
                "composite_score": "종합 점수",
            }
            importance_df["label"] = importance_df["feature"].map(label_map).fillna(importance_df["feature"])
            importance_chart = (
                alt.Chart(importance_df)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("importance:Q", title="중요도"),
                    y=alt.Y("label:N", title="특성", sort="-x"),
                    tooltip=[
                        alt.Tooltip("label:N", title="특성"),
                        alt.Tooltip("importance:Q", title="중요도", format=".3f"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(importance_chart, use_container_width=True)

    st.markdown("### 4️⃣ 전월세 적정성 & 리스크")
    st.caption(
        f"보증금을 환산해 Rent Gap과 월세 저평가율을 계산해 전월세 대비 수익성을 점검합니다. (현재 환산율 {annual_rate:.1f}%)"
    )
    rent_period_choice = st.selectbox(
        "Rent Gap 분석 기간",
        ["최근 3개월", "최근 6개월", "최근 1년", "전체"],
        index=1,
        key="rent_gap_period_selector",
        help="Rent Gap 차트에 사용할 기간을 선택하세요.",
    )
    rent_period_map = {"최근 3개월": 3, "최근 6개월": 6, "최근 1년": 12, "전체": None}
    rent_months = rent_period_map.get(rent_period_choice)
    rent_base = contract_src if not contract_src.empty else fact_src
    rent_filtered = filter_recent_months(rent_base, rent_months)
    rent_fact = filter_by_regions(rent_filtered, region_selection)
    rent_fact = filter_by_neighborhoods(rent_fact, neighborhood_filters)
    rent_fact = filter_by_properties(rent_fact, selected_property_keys)
    fallback_to_all_months = False
    if rent_fact.empty and not rent_base.empty and rent_months is not None:
        rent_fact = filter_by_regions(filter_recent_months(rent_base, None), region_selection)
        rent_fact = filter_by_neighborhoods(rent_fact, neighborhood_filters)
        rent_fact = filter_by_properties(rent_fact, selected_property_keys)
        fallback_to_all_months = not rent_fact.empty
    rent_summary_period = compute_rent_metrics(rent_fact, annual_rate)

    if rent_summary_period.empty:
        st.info("전월세 적정성 분석에 사용할 데이터를 찾을 수 없습니다. 기간이나 지역을 조정해 보세요.")
    else:
        rent_scores = rent_summary_period[[GROUP_COL, "rent_undervalue_rate"]].copy()
        if fallback_to_all_months:
            st.caption("선택한 기간에 데이터가 없어 전체 기간으로 대체했습니다.")
        rent_view = aggregate_summary_by_region(rent_summary_period, aggregate_charts)
        rent_view["월세 저평가율"] = rent_view["rent_undervalue_rate"].apply(format_percent)
        rent_view.drop(columns=["rent_undervalue_rate"], inplace=True)
        st.table(rent_view.rename(columns={
            GROUP_COL: "시군구",
            "rent_gap_mean": "월 Rent Gap(만원)",
            "avg_monthly_rent": "평균 월세(만원)",
        }))
        rent_chart_data = aggregate_summary_by_region(rent_summary_period, aggregate_charts)
        render_rent_gap_chart(rent_chart_data)

    region_components = []
    if not composite_scores.empty:
        region_components.append(composite_scores.rename(columns={"composite_score": "comp_raw"}))
    if not cluster_scores.empty:
        region_components.append(cluster_scores.rename(columns={"kmeans_score": "cluster_raw"}))
    if not ml_scores.empty:
        region_components.append(ml_scores.rename(columns={"ml_undervalue_rate": "ml_raw"}))
    if not rent_scores.empty:
        region_components.append(rent_scores.rename(columns={"rent_undervalue_rate": "rent_raw"}))

    st.markdown("### 🔝 최종 추천 지역 (Top3)")
    if not region_components:
        st.info("추천 점수를 계산할 수 없습니다. 상단 지표를 먼저 확인하세요.")
    else:
        merged = region_components[0]
        for df in region_components[1:]:
            merged = merged.merge(df, on=GROUP_COL, how="outer")

        def _normalize(values: pd.Series | object, invert: bool = False) -> pd.Series:
            if not isinstance(values, pd.Series):
                return pd.Series(dtype=float)
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.empty:
                return pd.Series(np.nan, index=values.index)
            valid = numeric.dropna()
            if valid.empty:
                return pd.Series(np.nan, index=values.index)
            min_val = valid.min()
            max_val = valid.max()
            if np.isclose(min_val, max_val):
                scaled = pd.Series(50.0, index=values.index)
            else:
                scaled = (numeric - min_val) / (max_val - min_val)
                if invert:
                    scaled = 1.0 - scaled
                scaled = scaled * 100
            return scaled

        merged["score_composite"] = _normalize(merged.get("comp_raw"))
        merged["score_cluster"] = _normalize(merged.get("cluster_raw"))
        merged["score_ml"] = _normalize(merged.get("ml_raw"))
        merged["score_rent"] = _normalize(merged.get("rent_raw"))

        weight_map = {
            "score_composite": 0.45,
            "score_cluster": 0.2,
            "score_ml": 0.25,
            "score_rent": 0.1,
        }
        total_weight = sum(w for key, w in weight_map.items() if key in merged.columns and not merged[key].isna().all())
        if total_weight == 0:
            merged["final_score"] = np.nan
        else:
            merged["final_score"] = sum(
                merged.get(key, 0).fillna(0) * weight
                for key, weight in weight_map.items()
            ) / total_weight

        candidate_pool = merged.dropna(subset=[GROUP_COL]).copy()
        primary_rank = candidate_pool.dropna(subset=["final_score"]).sort_values("final_score", ascending=False)
        ordered = primary_rank.copy()
        if len(ordered) < 3:
            fallback_columns = ["score_composite", "score_cluster", "score_ml", "score_rent"]
            for column in fallback_columns:
                fallback = candidate_pool.dropna(subset=[column])
                if fallback.empty:
                    continue
                fallback_sorted = fallback.sort_values(column, ascending=False)
                ordered = pd.concat([ordered, fallback_sorted], ignore_index=True)
                ordered = ordered.drop_duplicates(subset=[GROUP_COL], keep="first")
                if len(ordered) >= 3:
                    break
        top_regions = ordered.head(3).copy()
        if top_regions.empty or top_regions[GROUP_COL].isna().all():
            st.info("추천 대상 지역을 계산할 수 없습니다. 필터 조건을 확인하세요.")
        else:
            display_cols = [GROUP_COL, "final_score", "score_composite", "score_cluster", "score_ml", "score_rent"]
            readable = top_regions[display_cols].rename(columns={
                GROUP_COL: "시군구",
                "final_score": "최종 점수",
                "score_composite": "종합",
                "score_cluster": "군집",
                "score_ml": "ML",
                "score_rent": "전월세",
            })
            for col in ["최종 점수", "종합", "군집", "ML", "전월세"]:
                if col in readable.columns:
                    readable[col] = readable[col].apply(lambda v: _format_float(v, 1, "") if not pd.isna(v) else "-")
            st.table(readable.reset_index(drop=True))

            persona_focus_candidates = []
            for rank, (_, row) in enumerate(top_regions.iterrows(), start=1):
                region = str(row.get(GROUP_COL))
                if not region or region.strip() == "" or pd.isna(region):
                    continue
                parts = []
                if not pd.isna(row.get("score_composite")):
                    parts.append(f"종합 {row['score_composite']:.0f}")
                if not pd.isna(row.get("score_cluster")):
                    parts.append(f"군집 {row['score_cluster']:.0f}")
                if not pd.isna(row.get("score_ml")):
                    parts.append(f"ML {row['score_ml']:.0f}")
                if not pd.isna(row.get("score_rent")):
                    parts.append(f"전월세 {row['score_rent']:.0f}")
                persona_focus_candidates.append({
                    "label": f"최종 추천 #{rank} · {region}",
                    "region": region,
                    "summary": " / ".join(parts),
                    "payload": {
                        "source": "final",
                        "region": region,
                        "final_score": float(row.get("final_score", np.nan)) if not pd.isna(row.get("final_score", np.nan)) else None,
                        "score_composite": float(row.get("score_composite", np.nan)) if not pd.isna(row.get("score_composite", np.nan)) else None,
                        "score_cluster": float(row.get("score_cluster", np.nan)) if not pd.isna(row.get("score_cluster", np.nan)) else None,
                        "score_ml": float(row.get("score_ml", np.nan)) if not pd.isna(row.get("score_ml", np.nan)) else None,
                        "score_rent": float(row.get("score_rent", np.nan)) if not pd.isna(row.get("score_rent", np.nan)) else None,
                    },
                })

            if persona_focus_candidates:
                st.markdown("#### 추천 포커스 지역 미리보기")
                preview_cols = st.columns(len(persona_focus_candidates))
                for idx, (col, candidate) in enumerate(zip(preview_cols, persona_focus_candidates), start=1):
                    payload = candidate.get("payload", {})
                    final_score = payload.get("final_score")
                    with col:
                        st.markdown(f"**#{idx} {candidate['region']}**")
                        score_text = "-" if final_score is None or pd.isna(final_score) else f"{final_score:.1f}"
                        st.metric("최종 점수", score_text)
                        summary_text = str(candidate.get("summary") or "").strip()
                        if summary_text:
                            st.caption(summary_text)
                        else:
                            st.caption("세부 점수는 준비 중입니다.")

                labels = [item["label"] for item in persona_focus_candidates]
                current_label = st.session_state.get("persona_focus_choice")
                if current_label not in labels:
                    current_label = labels[0]
                selected_label = st.radio(
                    "포커스 후보 선택",
                    labels,
                    index=labels.index(current_label),
                    horizontal=True,
                    key="persona_focus_choice",
                    help="선택한 포커스 지역은 Persona·전략 비교·매물 추천에 연동됩니다.",
                )
                selected_candidate = next(item for item in persona_focus_candidates if item["label"] == selected_label)
                st.session_state["persona_reco_highlights"] = persona_focus_candidates
                st.session_state["persona_focus_choice_persona"] = selected_label
                st.session_state["persona_focus_region"] = selected_candidate["region"]
                st.session_state["persona_focus_payload"] = dict(selected_candidate.get("payload", {}))
                st.session_state["persona_focus_summary"] = selected_candidate.get("summary", "")
            else:
                st.session_state.pop("persona_reco_highlights", None)
                st.session_state.pop("persona_focus_choice", None)
                st.session_state.pop("persona_focus_choice_persona", None)
                st.session_state.pop("persona_focus_region", None)
                st.session_state.pop("persona_focus_payload", None)
                st.session_state.pop("persona_focus_summary", None)

    st.info("추천 후보를 비교하려면 **'🧪 Validate'** 탭에서 지표를 확인하세요.")

with validate_tab:
    try:
        with st.expander("가중치 조정", expanded=False):
            st.caption("전략 프리셋을 기반으로 저평가·모멘텀·안정성 가중치를 미세 조정하세요.")
            st.slider("저평가 가중치", 0.0, 1.0, float(current_w1), 0.05, key="weight_w1")
            st.slider("모멘텀 가중치", 0.0, 1.0, float(current_w2), 0.05, key="weight_w2")
            st.slider("안정성 가중치", 0.0, 1.0, float(current_w3), 0.05, key="weight_w3")
            total = st.session_state["weight_w1"] + st.session_state["weight_w2"] + st.session_state["weight_w3"]
            if not np.isclose(total, 1.0):
                st.warning("가중치 합이 1에 가깝도록 조정하면 비교가 더 명확해집니다.")

        current_w1 = st.session_state.get("weight_w1", current_w1)
        current_w2 = st.session_state.get("weight_w2", current_w2)
        current_w3 = st.session_state.get("weight_w3", current_w3)

        highlight_queue = st.session_state.get("persona_reco_highlights", [])
        if highlight_queue:
            labels = [item["label"] for item in highlight_queue]
            current_choice = st.session_state.get("persona_focus_choice")
            if current_choice not in labels:
                current_choice = labels[0]
                st.session_state["persona_focus_choice"] = current_choice
            stored_secondary = st.session_state.get("persona_focus_choice_persona", current_choice)
            if stored_secondary not in labels:
                stored_secondary = current_choice
                st.session_state["persona_focus_choice_persona"] = stored_secondary
            persona_choice = st.selectbox(
                "추천 후보",
                labels,
                key="persona_focus_choice_persona",
                help="추천 탭에서 선택한 포커스 지역을 바꿀 수 있습니다.",
            )
            selected_label = persona_choice
            if selected_label != st.session_state.get("persona_focus_choice"):
                st.session_state["persona_focus_choice"] = selected_label
                selected_candidate = next(item for item in highlight_queue if item["label"] == selected_label)
                st.session_state["persona_focus_region"] = selected_candidate["region"]
                st.session_state["persona_focus_payload"] = dict(selected_candidate.get("payload", {}))
                st.session_state["persona_focus_summary"] = selected_candidate.get("summary", "")
                st.experimental_rerun()
            selected_candidate = next(item for item in highlight_queue if item["label"] == selected_label)
            summary_text = str(selected_candidate.get("summary") or "").strip()
            if summary_text:
                st.caption(summary_text)
            else:
                st.caption("추천 후보를 선택하면 아래 설명이 자동으로 갱신됩니다.")
        else:
            st.session_state.pop("persona_focus_choice", None)
            st.session_state.pop("persona_focus_choice_persona", None)
            st.session_state.pop("persona_focus_region", None)
            st.session_state.pop("persona_focus_payload", None)
            st.session_state.pop("persona_focus_summary", None)

        focus_region = st.session_state.get("persona_focus_region")
        focus_payload = st.session_state.get("persona_focus_payload", {})
        if focus_region and GROUP_COL in persona_source.columns:
            focus_data = persona_source[persona_source[GROUP_COL] == focus_region]
            if focus_data.empty:
                st.info(f"{focus_region} 데이터가 없어 추천 포커스를 표시할 수 없습니다.")
                st.session_state.pop("persona_focus_payload", None)
            else:
                st.subheader("추천 포커스 지역")
                st.caption("Refine 탭에서 선택한 지역의 주요 지표입니다.")
                focus_row = focus_data.iloc[0]
                metric_cols = st.columns(3)
                metric_cols[0].metric("종합 점수", format_score(focus_row.get("composite_score")))
                metric_cols[1].metric("저평가율", format_percent(focus_row.get("undervalue_rate")))
                metric_cols[2].metric("모멘텀", format_percent(focus_row.get("mom_3m")))

                summary_cols = [
                    GROUP_COL,
                    "avg_p_per_m2",
                    "ml_undervalue_rate",
                    "kmeans_score",
                    "stability",
                    "avg_yield_pct",
                ]
                available_summary = [col for col in summary_cols if col in focus_data.columns]
                if available_summary:
                    focus_view = focus_data[available_summary].copy()
                    rename_map = {
                        GROUP_COL: "시군구",
                        "avg_p_per_m2": "평균 ㎡당가",
                        "ml_undervalue_rate": "ML 저평가율",
                        "kmeans_score": "군집 점수",
                        "stability": "안정성",
                        "avg_yield_pct": "평균 Yield",
                    }
                    focus_view.rename(columns={k: v for k, v in rename_map.items() if k in focus_view.columns}, inplace=True)
                    for col in focus_view.columns:
                        if "저평가율" in col:
                            focus_view[col] = focus_view[col].apply(format_percent)
                        elif col in {"군집 점수", "안정성"}:
                            focus_view[col] = focus_view[col].apply(lambda v: _format_float(v, 2))
                        elif col in {"평균 ㎡당가"}:
                            focus_view[col] = focus_view[col].apply(lambda v: _format_float(v, 1))
                        elif col == "평균 Yield":
                            focus_view[col] = focus_view[col].apply(lambda v: _format_float(v, 1))
                    st.table(focus_view.reset_index(drop=True))

                payload_source = focus_payload.get("source")
                if focus_payload.get("region") != focus_region:
                    payload_source = None
                if payload_source == "ml":
                    ml_cols = st.columns(3)
                    ml_cols[0].metric("모델 예측 ㎡당가", _format_float(focus_payload.get("ml_predicted"), 1, "만원"))
                    ml_cols[1].metric("실제 ㎡당가", _format_float(focus_payload.get("ml_actual"), 1, "만원"))
                    ml_cols[2].metric("ML 저평가율", format_percent(focus_payload.get("ml_rate")))
                    st.caption("랜덤포레스트 기반 저평가 분석 결과입니다.")
                elif payload_source == "cluster":
                    st.caption("군집 기반 추천에서 선택한 지역입니다.")
                elif payload_source == "final":
                    final_cols = st.columns(4)
                    final_cols[0].metric("최종 점수", _format_float(focus_payload.get("final_score"), 1))
                    final_cols[1].metric("종합", _format_float(focus_payload.get("score_composite"), 1))
                    final_cols[2].metric("군집", _format_float(focus_payload.get("score_cluster"), 1))
                    final_cols[3].metric("ML", _format_float(focus_payload.get("score_ml"), 1))
                    extra_text = []
                    if focus_payload.get("score_rent") is not None:
                        extra_text.append(f"전월세 {focus_payload['score_rent']:.1f}")
                    if extra_text:
                        st.caption(" · ".join(extra_text))
                    else:
                        st.caption("최종 점수는 여러 지표를 통합해 계산되었습니다.")
                elif payload_source == "composite":
                    st.caption(f"종합 추천 점수 {format_score(focus_payload.get('composite_score'))}로 선택됐습니다.")

                if st.button("포커스 해제", key="persona_focus_clear"):
                    st.session_state.pop("persona_focus_region", None)
                    st.session_state.pop("persona_focus_payload", None)
        else:
            st.info("추천 후보가 아직 없습니다. **'💡 Refine'** 탭에서 후보를 선택해 주세요.")

        st.subheader("전략 기반 비교")
        st.caption("최근 월의 주요 지표와 기준선 대비 편차를 활용해 선택 지역을 나란히 비교합니다.")
        available_regions = sorted(compare_base[GROUP_COL].dropna().unique().tolist())

        default_compare: list[str] = []
        recommended_regions = [
            item["region"]
            for item in st.session_state.get("persona_reco_highlights", persona_focus_candidates)
            if isinstance(item, dict) and item.get("region") in available_regions
        ]
        for region in recommended_regions:
            if region not in default_compare:
                default_compare.append(region)

        fallback_priority = [
            "서울특별시 서초구",
            "서울특별시 관악구",
            "서울특별시 영등포구",
        ]
        if len(default_compare) < 2:
            for region in fallback_priority:
                if region in available_regions and region not in default_compare:
                    default_compare.append(region)
                if len(default_compare) >= 2:
                    break

        if len(default_compare) < 2 and "composite_score" in compare_base.columns:
            ranked = (
                compare_base[[GROUP_COL, "composite_score"]]
                .dropna(subset=[GROUP_COL])
                .sort_values("composite_score", ascending=False)
            )
            ranked_list = ranked[GROUP_COL].dropna().drop_duplicates().tolist()
            for region in ranked_list:
                if region not in default_compare:
                    default_compare.append(region)
                if len(default_compare) >= 2:
                    break

        if len(default_compare) < 2:
            for region in available_regions:
                if region not in default_compare:
                    default_compare.append(region)
                if len(default_compare) >= 2:
                    break

        if recommended_regions:
            st.caption("추천 Top3 지역을 비교 기본값에 자동으로 채워뒀습니다. 필요하면 다른 지역으로 바꿔보세요.")

        compare_targets = st.multiselect(
            "비교할 지역 선택", available_regions, default=default_compare,
            help="두 개 이상 선택하면 지표를 나란히 비교합니다.")

        if len(compare_targets) < 2:
            st.info("2개 이상의 시군구를 선택하면 비교 테이블이 표시됩니다.")
            compare_view = pd.DataFrame()
        else:
            compare_view = (
                compare_base
                .set_index(GROUP_COL)
                .reindex(compare_targets)
                .reset_index()
                .rename(columns={"index": GROUP_COL})
            )
            available_set = set(compare_base[GROUP_COL].dropna().unique().tolist())
            missing_targets = [region for region in compare_targets if region not in available_set]
            if missing_targets:
                st.warning("비교 데이터가 부족한 지역: " + ", ".join(missing_targets))
            non_empty_view = compare_view.drop(columns=[GROUP_COL], errors="ignore").dropna(how="all")
            if non_empty_view.empty:
                st.info("선택한 지역에 대한 비교 데이터를 찾을 수 없습니다.")
            else:
                compare_display = compare_view.copy()
            chart_ready = compare_view[
                compare_view.drop(columns=[GROUP_COL], errors="ignore").notna().any(axis=1)
            ].copy()

            percent_cols = {
                "undervalue_rate",
                "undervalue_rate_delta",
                "mom_3m",
                "mom_3m_delta",
                "new_premium",
                "new_premium_delta",
                "floor_premium",
                "floor_premium_delta",
                "rent_undervalue_rate",
                "rent_undervalue_rate_delta",
                "cancel_rate",
                "cancel_rate_dom",
                "vacancy_proxy",
                "contract_renewal_rate",
                "contract_extension_rate",
                "contract_stability",
            }
            number_renderers = {
                "dom_median": lambda v: _format_float(v, 0),
                "contract_months_avg": lambda v: _format_float(v, 1),
                "contract_months_median": lambda v: _format_float(v, 1),
                "interest_rate": lambda v: _format_float(v, 2, "%"),
                "school_score": lambda v: _format_float(v, 1),
                "amenity_score": lambda v: _format_float(v, 1),
            }

            for col in percent_cols:
                if col in compare_display.columns:
                    compare_display[col] = compare_display[col].apply(format_percent)

            for col, renderer in number_renderers.items():
                if col in compare_display.columns:
                    compare_display[col] = compare_display[col].apply(renderer)

            rename_map = {
                GROUP_COL: "시군구",
                "composite_score": "종합 점수",
                "undervalue_rate": "저평가율",
                "undervalue_rate_delta": "저평가율-평균편차",
                "mom_3m": "3개월 모멘텀",
                "mom_3m_delta": "3개월 모멘텀-평균편차",
                "new_premium": "신축 프리미엄",
                "new_premium_delta": "신축 프리미엄-평균편차",
                "floor_premium": "층 프리미엄",
                "floor_premium_delta": "층 프리미엄-평균편차",
                "rent_undervalue_rate": "월세 저평가율",
                "rent_undervalue_rate_delta": "월세 저평가율-평균편차",
                "rent_gap_mean": "Rent Gap(만원)",
                "dom_median": "중위 DOM(일)",
                "cancel_rate": "취소율",
                "cancel_rate_dom": "취소율(계약)",
                "vacancy_proxy": "공실 Proxy",
                "contract_months_avg": "평균 계약기간(월)",
                "contract_months_median": "중위 계약기간(월)",
                "contract_renewal_rate": "갱신 비율",
                "contract_extension_rate": "갱신요구권 사용률",
                "contract_stability": "계약 안정성",
                "interest_rate": "금리(%)",
                "regulation_flag": "규제여부",
                "policy_comment": "정책 메모",
                "school_score": "학군 점수",
                "amenity_score": "생활편의 점수",
                "lifestyle_comment": "생활 메모",
            }
            st.table(compare_display.rename(columns=rename_map))

            comment_text = build_compare_comments(chart_ready)
            if comment_text:
                st.markdown(f"**자동 코멘트**\n{comment_text}")

            if baseline_reference:
                label_map = {
                    "undervalue_rate": "저평가율",
                    "mom_3m": "3개월 모멘텀",
                    "new_premium": "신축 프리미엄",
                    "floor_premium": "층 프리미엄",
                    "rent_undervalue_rate": "월세 저평가율",
                }
                parts = []
                for key, label in label_map.items():
                    if key in baseline_reference and not pd.isna(baseline_reference[key]):
                        parts.append(f"{label}: {format_percent(baseline_reference[key])}")
                if parts:
                    st.caption("전체 평균 · " + " · ".join(parts))

            radar_config = [
                {"key": "undervalue_rate", "label": "저평가율"},
                {"key": "mom_3m", "label": "모멘텀"},
                {"key": "new_premium", "label": "신축 프리미엄"},
                {"key": "avg_yield_pct", "label": "Yield"},
                {"key": "stability", "label": "안정성"},
                {"key": "rent_undervalue_rate", "label": "월세 저평가"},
                {"key": "vacancy_proxy", "label": "공실 Proxy", "invert": True},
                {"key": "interest_rate", "label": "금리(%)", "invert": True},
                {"key": "school_score", "label": "학군"},
                {"key": "amenity_score", "label": "생활편의"},
            ]
            present_config = [item for item in radar_config if item["key"] in chart_ready.columns]
            radar_data = prepare_radar_frame(chart_ready, present_config)
            if not radar_data.empty:
                sort_labels = [item["label"] for item in present_config]
                radar_chart = alt.Chart(radar_data).mark_line(point=True).encode(
                    theta=alt.Theta("metric:N", sort=sort_labels if sort_labels else None),
                    radius=alt.Radius("value:Q", scale=alt.Scale(domain=[0, 1], range=[20, 120])),
                    color=alt.Color(f"{GROUP_COL}:N", title="시군구"),
                    detail=f"{GROUP_COL}:N",
                    order="order:Q",
                    tooltip=[
                        alt.Tooltip(f"{GROUP_COL}:N", title="시군구"),
                        alt.Tooltip("metric:N", title="지표"),
                        alt.Tooltip("raw:Q", title="원본", format=".2f"),
                    ],
                ).properties(height=360)
                st.altair_chart(radar_chart, use_container_width=True)

            if "composite_score" in chart_ready.columns:
                bar_source = chart_ready.dropna(subset=["composite_score"])
                if not bar_source.empty:
                    chart = alt.Chart(bar_source).mark_bar(cornerRadiusEnd=4).encode(
                        x=alt.X("composite_score", title="종합 점수"),
                        y=alt.Y(f"{GROUP_COL}:N", sort='-x', title=""),
                        color=alt.value("#F28E2B")
                    ).properties(height=180)
                    st.altair_chart(chart, use_container_width=True)
        st.subheader("🧠 ML 추천 근거")
        st.caption("결정트리 시각화로 추천 근거가 된 분기 조건과 특징 중요도를 확인합니다.")
        tree_deps = _missing_dependencies({
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "plot_tree": plot_tree,
            "matplotlib": plt,
        })
        if tree_deps:
            missing = ", ".join(tree_deps)
            st.info(f"설명형 모델을 그리려면 추가 패키지가 필요합니다 ({missing})")
        else:
            tree_features_all = [
                "undervalue_rate",
                "mom_3m",
                "stability",
                "avg_yield_pct",
                "rent_undervalue_rate",
                "vacancy_proxy",
                "composite_score",
            ]
            tree_target = "avg_p_per_m2"
            tree_dataset = prepare_ml_dataset(persona_source, tree_features_all, tree_target)
            tree_model, tree_features = build_decision_tree_model(persona_source, tree_features_all, tree_target)
            if tree_model is None or not tree_features or tree_dataset.empty:
                st.info("결정트리를 그릴 데이터가 부족합니다. 선택한 기간과 지역을 조정해 보세요.")
            else:
                fig, ax = plt.subplots(figsize=(9, 5))
                plot_tree(
                    tree_model,
                    feature_names=tree_features,
                    filled=True,
                    rounded=True,
                    fontsize=8,
                    ax=ax,
                )
                st.pyplot(fig)
                plt.close(fig)

                importances = pd.Series(tree_model.feature_importances_, index=tree_features)
                importances = importances[importances > 0].sort_values(ascending=False)
                if not importances.empty:
                    st.markdown(
                        "**주요 분기 기준** : "
                        + " · ".join(f"{feat}({weight:.2f})" for feat, weight in importances.items())
                    )
    except Exception as exc:
        push_log(f"Persona 탭 렌더링 오류: {exc}", "error")
        st.error("페르소나 도표를 렌더링하는 중 오류가 발생했습니다. 메시지 로그를 확인하세요.")

    st.info("매물 확정 단계는 **'🏁 Decide'** 탭에서 이어집니다.")

with decide_tab:
    st.subheader("추천 지역 · 매물 확정")
    focus_candidates_state = st.session_state.get("persona_reco_highlights", [])
    base_candidates = persona_focus_candidates if "persona_focus_candidates" in locals() else []
    focus_candidates = focus_candidates_state or base_candidates

    if focus_candidates:
        summary_cols = st.columns(min(3, len(focus_candidates)))
        for col, candidate in zip(summary_cols, focus_candidates):
            payload = candidate.get("payload", {})
            final_score = payload.get("final_score")
            with col:
                st.markdown(f"**{candidate['region']}**")
                score_text = "-" if final_score is None or pd.isna(final_score) else f"{final_score:.1f}"
                st.metric("최종 점수", score_text)
                summary_text = str(candidate.get("summary") or "").strip()
                if summary_text:
                    st.caption(summary_text)
    else:
        st.info("Refine 탭에서 추천 후보를 먼저 생성하면 매물 추천을 자동으로 이어받을 수 있습니다.")

    if selected_property_keys:
        st.subheader("선택한 매물 요약")
        st.caption("사이드바에서 고른 지번의 거래·임대 지표를 모았습니다.")
        property_summary_table = prepare_property_summary_table(
            fact,
            contract_metrics_source,
            annual_rate,
            property_label_lookup,
        )
        if property_summary_table.empty:
            st.info("선택한 매물의 상세 통계를 계산할 수 없습니다. 거래 데이터가 부족합니다.")
        else:
            st.dataframe(
                property_summary_table.reset_index(drop=True),
                use_container_width=True,
                key="property_summary_decide",
            )

    st.markdown("### 추천 매물 Drill-down")
    candidate_regions = [item["region"] for item in focus_candidates if isinstance(item, dict)]
    if not candidate_regions and GROUP_COL in fact.columns:
        candidate_regions = sorted(fact[GROUP_COL].dropna().unique().tolist())
    candidate_regions = list(dict.fromkeys(candidate_regions))

    if not candidate_regions:
        st.info("추천 지역을 먼저 선택하면 매물 후보를 제안할 수 있습니다.")
    else:
        default_region = st.session_state.get("property_reco_region")
        focus_region_state = st.session_state.get("persona_focus_region")
        if not default_region and focus_region_state in candidate_regions:
            default_region = focus_region_state
        if default_region not in candidate_regions:
            default_region = candidate_regions[0]

        selected_property_region = st.selectbox(
            "매물 추천 대상 시군구",
            candidate_regions,
            index=candidate_regions.index(default_region),
            key="property_reco_region",
        )

        region_trade = fact[fact[GROUP_COL] == selected_property_region] if GROUP_COL in fact.columns else pd.DataFrame()
        if region_trade.empty:
            st.info(f"{selected_property_region} 지역의 거래 데이터를 찾을 수 없어 매물 추천을 계산할 수 없습니다.")
        else:
            property_reco = compute_property_recommendations(region_trade)
            if property_reco.empty:
                st.info("매물 정보를 계산할 수 없습니다. 지번/층 정보가 포함된 데이터를 확인해 주세요.")
            else:
                top_property_count = st.slider("추천 매물 개수", min_value=3, max_value=15, value=5, step=1, key="property_reco_topn")
                display_reco = property_reco.head(top_property_count).copy()
                if "avg_price_per_m2" in display_reco.columns:
                    display_reco["avg_price_per_m2"] = display_reco["avg_price_per_m2"].apply(lambda v: _format_float(v, 1))
                if "avg_yield_pct" in display_reco.columns:
                    display_reco["avg_yield_pct"] = display_reco["avg_yield_pct"].apply(lambda v: _format_float(v, 2))
                if "avg_area_sqm" in display_reco.columns:
                    display_reco["avg_area_sqm"] = display_reco["avg_area_sqm"].apply(lambda v: _format_float(v, 1))
                if "avg_trade_amount" in display_reco.columns:
                    display_reco["avg_trade_amount"] = display_reco["avg_trade_amount"].apply(lambda v: _format_float(v, 0))
                if "latest_contract" in display_reco.columns:
                    display_reco["latest_contract"] = display_reco["latest_contract"].apply(
                        lambda dt: dt.strftime("%Y-%m-%d") if isinstance(dt, pd.Timestamp) and not pd.isna(dt) else "-"
                    )
                display_reco["score"] = display_reco["score"].apply(lambda v: _format_float(v, 1, "%"))

                column_rename = {
                    "property_key": "매물 ID",
                    "property_label": "매물",
                    "sigungu": "시군구",
                    "eupmyeondong": "읍·동",
                    "avg_price_per_m2": "평균 ㎡당가(만원)",
                    "avg_yield_pct": "평균 Yield(%)",
                    "avg_area_sqm": "평균 전용(㎡)",
                    "avg_trade_amount": "평균 거래금액(만원)",
                    "transaction_count": "거래 건수",
                    "latest_contract": "최근 계약일",
                    "score": "추천 점수",
                }
                display_reco.rename(columns={k: v for k, v in column_rename.items() if k in display_reco.columns}, inplace=True)
                st.dataframe(display_reco.reset_index(drop=True), use_container_width=True, key="property_reco_table")
                st.caption("관심 매물을 확정하려면 사이드바 **개별 매물 (지번)**에서 동일한 라벨을 선택해 비교해 보세요.")

                persona_base = property_reco.copy()

                def _normalize(values: pd.Series | object, invert: bool = False) -> pd.Series:
                    if not isinstance(values, pd.Series) or values.empty:
                        if persona_base.empty:
                            return pd.Series(dtype=float)
                        return pd.Series(0.0, index=persona_base.index, dtype=float)

                    numeric = pd.to_numeric(values, errors="coerce")
                    valid = numeric.dropna()
                    if valid.empty:
                        norm = pd.Series(0.0, index=values.index, dtype=float)
                    else:
                        min_val = float(valid.min())
                        max_val = float(valid.max())
                        if math.isclose(min_val, max_val, rel_tol=1e-9, abs_tol=1e-9):
                            norm = pd.Series(0.5, index=values.index, dtype=float)
                        else:
                            norm = (numeric - min_val) / (max_val - min_val)
                    norm = norm.fillna(0.0).clip(0.0, 1.0)
                    if invert:
                        norm = 1.0 - norm
                    return norm

                persona_mapping = {
                    "income": ("연금형(인컴)", "수익률 중심 안정형"),
                    "growth": ("실거주 성장", "신축·성장성 우선"),
                    "capital": ("시세차익형", "가격 메리트·거래량"),
                }

                income_score = (
                    0.7 * _normalize(persona_base.get("avg_yield_pct"))
                    + 0.3 * _normalize(persona_base.get("avg_price_per_m2"), invert=True)
                )
                growth_score = (
                    0.6 * _normalize(persona_base.get("new_ratio"))
                    + 0.4 * _normalize(persona_base.get("avg_price_per_m2"), invert=True)
                )
                capital_score = (
                    0.5 * _normalize(persona_base.get("avg_price_per_m2"), invert=True)
                    + 0.5 * _normalize(persona_base.get("transaction_count"))
                )

                persona_base["income_score"] = income_score
                persona_base["growth_score"] = growth_score
                persona_base["capital_score"] = capital_score

                st.markdown("### 페르소나별 추천 매물 Top5")
                persona_tabs = st.tabs([label for label, _ in persona_mapping.values()])
                for (key, (label, desc)), tab in zip(persona_mapping.items(), persona_tabs):
                    with tab:
                        st.caption(desc)
                        score_col = f"{key}_score"
                        if score_col not in persona_base.columns or persona_base[score_col].dropna().empty:
                            st.info("필요한 지표가 부족해 매물 추천을 계산할 수 없습니다.")
                            continue
                        persona_rank = (
                            persona_base.sort_values(score_col, ascending=False)
                            .head(5)
                            .copy()
                        )
                        if persona_rank.empty:
                            st.info("추천 결과가 없습니다.")
                            continue

                        persona_rank["추천 점수"] = persona_rank[score_col].apply(lambda v: _format_float(v * 100, 1))
                        display_cols = [
                            "property_label",
                            "sigungu",
                            "eupmyeondong",
                            "avg_price_per_m2",
                            "avg_yield_pct",
                            "avg_area_sqm",
                            "avg_trade_amount",
                            "latest_contract",
                            "transaction_count",
                            "추천 점수",
                        ]
                        available_cols = [col for col in display_cols if col in persona_rank.columns]
                        view = persona_rank[available_cols].copy()
                        rename_map = {
                            "property_label": "매물",
                            "sigungu": "시군구",
                            "eupmyeondong": "읍·동",
                            "avg_price_per_m2": "평균 ㎡당가(만원)",
                            "avg_yield_pct": "평균 Yield(%)",
                            "avg_area_sqm": "평균 전용(㎡)",
                            "avg_trade_amount": "평균 거래금액(만원)",
                            "latest_contract": "최근 계약일",
                            "transaction_count": "거래 건수",
                        }
                        view.rename(columns={k: v for k, v in rename_map.items() if k in view.columns}, inplace=True)
                        if "평균 ㎡당가(만원)" in view.columns:
                            view["평균 ㎡당가(만원)"] = view["평균 ㎡당가(만원)"].apply(lambda v: _format_float(v, 1))
                        if "평균 Yield(%)" in view.columns:
                            view["평균 Yield(%)"] = view["평균 Yield(%)"].apply(lambda v: _format_float(v, 2))
                        if "평균 전용(㎡)" in view.columns:
                            view["평균 전용(㎡)"] = view["평균 전용(㎡)"].apply(lambda v: _format_float(v, 1))
                        if "평균 거래금액(만원)" in view.columns:
                            view["평균 거래금액(만원)"] = view["평균 거래금액(만원)"].apply(lambda v: _format_float(v, 0))
                        if "최근 계약일" in view.columns:
                            view["최근 계약일"] = view["최근 계약일"].apply(
                                lambda dt: dt.strftime("%Y-%m-%d") if isinstance(dt, pd.Timestamp) and not pd.isna(dt) else "-"
                            )
                        st.table(view.reset_index(drop=True))
