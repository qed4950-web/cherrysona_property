# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Streamlit Unified Real-Estate Dashboard
- 데이터: ETL 산출물 (Parquet / CSV)
- 기능: dashboard.py (KPI, VC, RO, Persona, Explorer) + streamlit_realestate_app.py (Top5 Insights)
"""

import os
from pathlib import Path
import math
from datetime import datetime
from typing import Callable, Optional, Set, Union

import altair as alt
import pandas as pd, numpy as np
import pydeck as pdk
import streamlit as st

try:  # optional dependency for DuckDB backend
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    duckdb = None

GROUP_COL = "시군구"
SIGUNGU_ALIASES = ("시군구", "sigungu")
LOG_STORAGE_KEY = "__sidebar_logs__"


def push_log(message: str, level: str = "info") -> None:
    logs = st.session_state.setdefault(LOG_STORAGE_KEY, [])
    logs.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level.upper(),
        "message": message,
    })


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
MOMENTUM_PATH = DATA_DIR / "monthly_momentum.parquet"
VOLATILITY_PATH = DATA_DIR / "monthly_volatility.parquet"
FACT_PATH = DATA_DIR / "transactions.csv"
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


@st.cache_data(show_spinner=False, ttl=3600)
def load_monthly_basics(backend: str = "Parquet", duckdb_path: Optional[str] = None):
    if backend == "DuckDB" and duckdb_path:
        return fetch_duckdb_table(duckdb_path, DUCKDB_TABLES["monthly_basics"])
    return pd.read_parquet(MONTHLY_PATH) if MONTHLY_PATH.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_momentum(backend: str = "Parquet", duckdb_path: Optional[str] = None):
    if backend == "DuckDB" and duckdb_path:
        return fetch_duckdb_table(duckdb_path, DUCKDB_TABLES["monthly_momentum"])
    return pd.read_parquet(MOMENTUM_PATH) if MOMENTUM_PATH.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_volatility(backend: str = "Parquet", duckdb_path: Optional[str] = None):
    if backend == "DuckDB" and duckdb_path:
        return fetch_duckdb_table(duckdb_path, DUCKDB_TABLES["monthly_volatility"])
    return pd.read_parquet(VOLATILITY_PATH) if VOLATILITY_PATH.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600)
def load_transactions(backend: str = "Parquet", duckdb_path: Optional[str] = None):
    if backend == "DuckDB" and duckdb_path:
        return fetch_duckdb_table(duckdb_path, DUCKDB_TABLES["transactions"])
    return pd.read_csv(FACT_PATH) if FACT_PATH.exists() else pd.DataFrame()


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

    work["YYYYMM"] = work["YYYYMM"].astype(str)
    axis_kwargs = {"title": label}
    if axis_format:
        axis_kwargs["format"] = axis_format

    chart = alt.Chart(work).mark_line(point=True).encode(
        x=alt.X("YYYYMM:O", title="계약월"),
        y=alt.Y(f"{value_col}:Q", axis=alt.Axis(**axis_kwargs)),
        color=alt.Color(f"{GROUP_COL}:N", title="시군구")
    ).properties(height=280)
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
    merged = base[[GROUP_COL, metric]].merge(centroids, on=GROUP_COL, how="inner")
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

    metric_series = metrics.set_index(GROUP_COL)[metric]
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


def compute_dom_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required = {GROUP_COL, "YYYYMM", "취소여부", "계약일자"}
    if df.empty or not required.issubset(df.columns):
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
        return pd.DataFrame()

    work = work.copy()
    work["snapshot_date"] = pd.to_datetime(work.get("snapshot_date"), errors="coerce")
    work["계약일자"] = pd.to_datetime(work["계약일자"], errors="coerce")
    work["취소여부"] = pd.to_numeric(work["취소여부"], errors="coerce")

    for column, default in [
        ("contract_months", np.nan),
        ("contract_is_renewal", 0),
        ("contract_extension_flag", 0),
        ("contract_stability_score", np.nan),
    ]:
        if column not in work.columns:
            work[column] = default

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

    base = df.copy()
    base = base.groupby(GROUP_COL, as_index=False).first()
    result = base[[GROUP_COL]].copy()

    for persona_key, components in persona_components.items():
        scaled_parts: list[pd.Series] = []
        for col, invert in components:
            if col in base.columns:
                scaled_parts.append(minmax_scale(base[col], invert=invert))
        if not scaled_parts:
            score = pd.Series(0.0, index=base.index, dtype=float)
        else:
            stacked = pd.concat(scaled_parts, axis=1)
            score = stacked.mean(axis=1)
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
        comment_tail = row.get("lifestyle_comment") if "lifestyle_comment" in row else ""
        if comment_tail and isinstance(comment_tail, str) and comment_tail.strip():
            pieces.append(comment_tail.strip())
        if not pieces:
            pieces.append("지표 없음")
        detail_lines.append(f"{row.get(GROUP_COL, '미지정')}: " + ", ".join(pieces))

    blocks = []
    if highlights:
        blocks.append(" · ".join(highlights))
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
def compute_undervaluation(df):
    base_cols = [GROUP_COL, "undervalue_rate", "avg_price_per_m2"]
    if "가격_per_㎡" not in df.columns:
        return pd.DataFrame(columns=base_cols)
    work = ensure_sigungu_named(df)
    if GROUP_COL not in work.columns:
        return pd.DataFrame(columns=base_cols)
    group_avg = work.groupby(GROUP_COL)["가격_per_㎡"].mean().rename("avg_price_per_m2")
    overall_avg = group_avg.mean()
    if pd.isna(overall_avg) or overall_avg == 0:
        return pd.DataFrame(columns=base_cols)
    out = group_avg.reset_index()
    out["undervalue_rate"] = (overall_avg - out["avg_price_per_m2"]) / overall_avg
    return out.reindex(columns=base_cols)

def compute_new_premium(df):
    base_cols = [GROUP_COL, "new_avg", "old_avg", "new_premium"]
    if not {"건축년도","가격_per_㎡"}.issubset(df.columns):
        return pd.DataFrame(columns=base_cols)
    work = ensure_sigungu_named(df)
    if GROUP_COL not in work.columns:
        return pd.DataFrame(columns=base_cols)
    cy=pd.Timestamp.today().year
    work["is_new"]=(cy-work["건축년도"])<=5
    work["is_old"]=(cy-work["건축년도"])>=20
    new=work[work["is_new"]].groupby(GROUP_COL)["가격_per_㎡"].mean().rename("new_avg")
    old=work[work["is_old"]].groupby(GROUP_COL)["가격_per_㎡"].mean().rename("old_avg")
    out=pd.concat([new,old],axis=1).dropna().reset_index()
    out["new_premium"]=(out["new_avg"]-out["old_avg"])/out["old_avg"]
    return out.reindex(columns=base_cols)

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

def compute_rent_metrics(df,annual_rate=0.055):
    if not {"보증금_만원","월세_만원"}.issubset(df.columns):
        return pd.DataFrame()
    work = ensure_sigungu_named(df)
    if GROUP_COL not in work.columns:
        return pd.DataFrame()
    work = work.copy()
    work["lease_equiv_monthly"] = work["보증금_만원"] * (annual_rate / 12.0)
    work["rent_gap_monthly"] = work["lease_equiv_monthly"] - work["월세_만원"]

    summary = (work.groupby(GROUP_COL)
               .agg(rent_gap_mean=("rent_gap_monthly", "mean"),
                    avg_monthly_rent=("월세_만원", "mean"))
               .reset_index())

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

    basics_raw = ensure_yyyymm_numeric(load_monthly_basics(data_backend, duckdb_path_input))
    momo_raw = ensure_yyyymm_numeric(load_momentum(data_backend, duckdb_path_input))
    vol_raw = ensure_yyyymm_numeric(load_volatility(data_backend, duckdb_path_input))
    fact_raw = ensure_yyyymm_numeric(load_transactions(data_backend, duckdb_path_input))

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
        region_selection = st.multiselect(
            "지역 (시군구)",
            region_options[1:],
            default=quick_selection or region_options[1:3],
            placeholder="시군구를 검색해 선택하세요.",
            help="비교할 시군구를 한 번에 선택하거나 검색해 추가하세요.",
        )
        region_selection = sorted(set(region_selection + quick_selection)) or ["전체"]

    if region_mode == "전체":
        region_selection = ["전체"]

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

    st.markdown("### 🛠️ 기타 옵션")
    enable_alerts = st.checkbox("알림 조건 저장", value=False, help="설정한 조건을 저장해 이후 알림 기능과 연동합니다.")
    export_ready = st.checkbox("내보내기 옵션 표시", value=False, help="데이터 내보내기 기능을 미리 확인합니다.")

geo_centroids = load_geo_centroids()
policy_raw = load_policy_risk()
lifestyle_raw = load_lifestyle_scores()
sigungu_boundaries = load_sigungu_boundaries()
lifestyle = filter_by_regions(lifestyle_raw, region_selection)
policy_filtered = filter_by_regions(policy_raw, region_selection)
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

for frame in (basics, momo, vol):
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
rent_summary = compute_rent_metrics(fact, annual_rate)
risk_timeseries = compute_dom_metrics(fact)
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
compare_base[numeric_columns] = compare_base[numeric_columns].fillna(0.0)
non_numeric_columns = compare_base.columns.difference(numeric_columns)
compare_base[non_numeric_columns] = compare_base[non_numeric_columns].fillna("")

baseline_reference: dict[str, float] = {}
for metric in ["undervalue_rate", "mom_3m", "new_premium", "floor_premium", "rent_undervalue_rate"]:
    if metric in compare_base.columns:
        baseline = compare_base[metric].mean()
        baseline_reference[metric] = baseline
        compare_base[f"{metric}_delta"] = compare_base[metric] - baseline

persona_profiles = compute_persona_profiles(compare_base)

trend_tab, insights_tab, reco_tab, persona_tab = st.tabs([
    "📊 Trend & KPI",
    "🔎 Insights",
    "💡 추천 & 리스크",
    "👥 Persona & Compare",
])

with trend_tab:
    sub_tabs = st.tabs(["Overview", "모멘텀", "변동성", "Explorer"])
    with sub_tabs[0]:
        st.subheader("월간 KPI")
        basics_display = aggregate_regions_for_chart(basics, aggregate_charts)
        render_time_series_chart(basics_display, "avg_p_per_m2", "평균 ㎡당 가격")
        with st.expander("원본 데이터 보기", expanded=False):
            st.dataframe(basics.head() if not basics.empty else pd.DataFrame())
    with sub_tabs[1]:
        st.subheader("모멘텀 추이")
        momo_display = aggregate_regions_for_chart(momo, aggregate_charts)
        render_time_series_chart(momo_display, "mom_3m", "3개월 모멘텀")
        with st.expander("원본 데이터 보기", expanded=False):
            st.dataframe(momo.head() if not momo.empty else pd.DataFrame())
    with sub_tabs[2]:
        st.subheader("변동성 / 취소율")
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
        st.dataframe(fact.head(100) if not fact.empty else pd.DataFrame())

with insights_tab:
    st.subheader("Top5 인사이트")
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

    st.markdown("**지역 Heatmap (β)**")
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
                    metric_values = pd.to_numeric(map_data[map_metric_key], errors="coerce")
                    domain_min, domain_max = _compute_metric_domain(metric_values.dropna())
                    use_abs_size = (map_metric_key in percent_metrics) or (domain_min < 0.0 < domain_max)
                    size_field = map_metric_key
                    size_label = map_label
                    if use_abs_size:
                        size_field = "_metric_size"
                        size_label = f"{map_label} (절댓값)"
                        map_data = map_data.assign(_metric_size=metric_values.abs())
                    size_domain_min, size_domain_max = _compute_metric_domain(pd.to_numeric(map_data[size_field], errors="coerce").dropna())
                    color_scale = _build_altair_color_scale(map_metric_key, map_data, percent_metrics)
                    chart = alt.Chart(map_data).mark_circle(opacity=0.85, stroke="#FFFFFF", strokeWidth=0.5).encode(
                        longitude="longitude:Q",
                        latitude="latitude:Q",
                        size=alt.Size(
                            f"{size_field}:Q",
                            title=size_label,
                            scale=alt.Scale(range=[60, 800], domain=[size_domain_min, size_domain_max], zero=False),
                        ),
                        color=alt.Color(
                            f"{map_metric_key}:Q",
                            title=map_label,
                            scale=color_scale,
                        ),
                        tooltip=[
                            alt.Tooltip(f"{GROUP_COL}:N", title="시군구"),
                            alt.Tooltip(
                                f"{map_metric_key}:Q",
                                title=map_label,
                                format=".1%" if map_metric_key in percent_metrics else ".1f",
                            ),
                        ],
                    ).properties(height=360).project("mercator")
                    st.altair_chart(chart, use_container_width=True)
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

with reco_tab:
    st.subheader("종합 추천 Top5")
    if comp_score.empty:
        st.info("필요한 데이터가 부족해 종합 점수를 계산할 수 없습니다.")
    else:
        st.caption(
            f"가중치 설정 · 저평가 {current_w1:.2f}, 모멘텀 {current_w2:.2f}, 안정성 {current_w3:.2f}")
        render_ranked_list(comp_score, "composite_score", "종합 점수", formatter=format_score, axis_format=".1f")

    st.subheader("전월세 적정성 & 리스크")
    rent_period_choice = st.selectbox(
        "Rent Gap 분석 기간",
        ["최근 3개월", "최근 6개월", "최근 1년", "전체"],
        index=1,
        key="rent_gap_period_selector",
        help="Rent Gap 차트에 사용할 기간을 선택하세요.",
    )
    rent_period_map = {"최근 3개월": 3, "최근 6개월": 6, "최근 1년": 12, "전체": None}
    rent_months = rent_period_map.get(rent_period_choice)
    rent_fact = filter_by_regions(filter_recent_months(fact_src, rent_months), region_selection)
    rent_summary_period = compute_rent_metrics(rent_fact, annual_rate)

    if rent_summary_period.empty:
        st.info("선택한 기간에 대한 전월세 데이터를 계산할 수 없습니다.")
    else:
        rent_view = aggregate_summary_by_region(rent_summary_period, aggregate_charts)
        rent_view["평균 월세 대비"] = rent_view["rent_undervalue_rate"].apply(format_percent)
        rent_view.drop(columns=["rent_undervalue_rate"], inplace=True)
        st.table(rent_view.rename(columns={
            GROUP_COL: "시군구",
            "rent_gap_mean": "월 Rent Gap(만원)",
            "avg_monthly_rent": "평균 월세(만원)",
        }))
        rent_chart_data = aggregate_summary_by_region(rent_summary_period, aggregate_charts)
        render_rent_gap_chart(rent_chart_data)

with persona_tab:
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

        st.subheader("전략 기반 비교")
        available_regions = sorted(compare_base[GROUP_COL].dropna().unique().tolist())
        default_compare = available_regions[:2]
        compare_targets = st.multiselect(
            "비교할 지역 선택", available_regions, default=default_compare,
            help="두 개 이상 선택하면 지표를 나란히 비교합니다.")

        if len(compare_targets) < 2:
            st.info("2개 이상의 시군구를 선택하면 비교 테이블이 표시됩니다.")
            compare_view = pd.DataFrame()
        else:
            compare_view = compare_base[compare_base[GROUP_COL].isin(compare_targets)].copy()
            if compare_view.empty:
                st.info("선택한 지역에 대한 비교 데이터를 찾을 수 없습니다.")
            else:
                compare_display = compare_view.copy()
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

                comment_text = build_compare_comments(compare_view)
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
                present_config = [item for item in radar_config if item["key"] in compare_view.columns]
                radar_data = prepare_radar_frame(compare_view, present_config)
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

                if "composite_score" in compare_view.columns:
                    chart = alt.Chart(compare_view).mark_bar(cornerRadiusEnd=4).encode(
                        x=alt.X("composite_score", title="종합 점수"),
                        y=alt.Y(f"{GROUP_COL}:N", sort='-x', title=""),
                        color=alt.value("#F28E2B")
                    ).properties(height=180)
                    st.altair_chart(chart, use_container_width=True)

        st.subheader("Persona 추천")
        if persona_profiles.empty:
            st.info("페르소나 추천을 계산할 수 없습니다. 비교 지표를 확인하세요.")
        else:
            persona_defs = [
                ("income", "연금형", "안정성과 Yield를 중시한 장기 보유"),
                ("growth", "실거주 성장", "신축·생활 프리미엄 중심"),
                ("capital", "시세차익형", "모멘텀·저평가 결합"),
            ]
            tabs = st.tabs([label for _, label, _ in persona_defs])
            for (key, label, desc), tab in zip(persona_defs, tabs):
                with tab:
                    st.caption(desc)
                    score_col = f"{key}_score"
                    if score_col not in persona_profiles.columns:
                        st.info("필요한 지표가 부족해 추천을 만들 수 없습니다.")
                        continue
                    top = persona_profiles.sort_values(score_col, ascending=False).head(3)
                    if top.empty:
                        st.info("추천 결과가 없습니다.")
                        continue
                    display_cols = [GROUP_COL, score_col]
                    metric_map = {
                        "income": ["avg_yield_pct", "stability", "cancel_rate", "dom_median", "vacancy_proxy", "interest_rate"],
                        "growth": [
                            "new_premium",
                            "mom_3m",
                            "floor_premium",
                            "rent_gap_mean",
                            "avg_p_per_m2",
                            "school_score",
                            "amenity_score",
                        ],
                        "capital": ["mom_3m", "undervalue_rate", "rent_undervalue_rate", "composite_score", "vacancy_proxy"],
                    }
                    for metric in metric_map.get(key, []):
                        if metric in top.columns:
                            display_cols.append(metric)
                    for extra in ["policy_comment"]:
                        if extra in top.columns and extra not in display_cols:
                            display_cols.append(extra)
                    view = top[display_cols].copy()
                    view.rename(columns={
                    score_col: "적합도(0~1)",
                    "avg_yield_pct": "평균 Yield(%)",
                    "stability": "안정성",
                    "cancel_rate": "취소율",
                    "new_premium": "신축 프리미엄",
                    "mom_3m": "3개월 모멘텀",
                    "floor_premium": "층 프리미엄",
                    "undervalue_rate": "저평가율",
                    "composite_score": "종합 점수",
                    "dom_median": "중위 DOM(일)",
                    "rent_undervalue_rate": "월세 저평가율",
                    "vacancy_proxy": "공실 Proxy",
                    "interest_rate": "금리(%)",
                    "rent_gap_mean": "Rent Gap(만원)",
                    "avg_p_per_m2": "㎡당 가격(만원)",
                    "policy_comment": "정책 메모",
                    "school_score": "학군 점수",
                    "amenity_score": "생활편의 점수",
                }, inplace=True)
                for col in view.columns:
                    if col.endswith("율") or col in {
                        "신축 프리미엄",
                        "3개월 모멘텀",
                        "층 프리미엄",
                        "저평가율",
                        "월세 저평가율",
                        "공실 Proxy",
                        "계약 안정성",
                        "갱신 비율",
                        "갱신요구권 사용률",
                    }:
                        view[col] = view[col].apply(format_percent)
                    elif col in {"평균 Yield(%)", "안정성"}:
                        view[col] = view[col].apply(lambda v: _format_float(v, 1))
                    elif col == "중위 DOM(일)":
                        view[col] = view[col].apply(lambda v: _format_float(v, 0))
                    elif col == "Rent Gap(만원)":
                        view[col] = view[col].apply(lambda v: _format_float(v, 1))
                    elif col == "㎡당 가격(만원)":
                        view[col] = view[col].apply(lambda v: _format_float(v, 1))
                    elif col == "금리(%)":
                        view[col] = view[col].apply(lambda v: _format_float(v, 2, "%"))
                    elif col in {"학군 점수", "생활편의 점수"}:
                        view[col] = view[col].apply(lambda v: _format_float(v, 1))
                    elif col == "적합도(0~1)":
                        view[col] = view[col].apply(
                            lambda v: f"{_to_float(v) * 100:.0f}%" if _to_float(v) is not None else "-"
                        )
                st.table(view.rename(columns={GROUP_COL: "시군구"}))

                top_region = top.iloc[0]
                bullet = []
                if "rent_gap_mean" in top.columns:
                    rg = _format_float(top_region.get("rent_gap_mean"), 1)
                    if rg != "-":
                        bullet.append(f"Rent Gap {rg}만원")
                if "avg_p_per_m2" in top.columns:
                    price = _format_float(top_region.get("avg_p_per_m2"), 1)
                    if price != "-":
                        bullet.append(f"㎡당 {price}만원")
                if "vacancy_proxy" in top.columns and not pd.isna(top_region.get("vacancy_proxy")):
                    bullet.append(f"공실 {format_percent(top_region['vacancy_proxy'])}")
                if "interest_rate" in top.columns:
                    rate = _format_float(top_region.get("interest_rate"), 2, "%")
                    if rate != "-":
                        bullet.append(f"금리 {rate}")
                if "policy_comment" in top.columns and isinstance(top_region.get("policy_comment"), str) and top_region["policy_comment"].strip():
                    bullet.append(top_region["policy_comment"].strip())
                if bullet:
                    st.markdown(f"**Top 지역 코멘트** · {top_region[GROUP_COL]} · " + " · ".join(bullet))
    except Exception as exc:
        push_log(f"Persona 탭 렌더링 오류: {exc}", "error")
        st.error("페르소나 도표를 렌더링하는 중 오류가 발생했습니다. 메시지 로그를 확인하세요.")
