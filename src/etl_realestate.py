"""새로운 cherrysona_property ETL 파이프라인 진입점.

원본 실거래 CSV를 읽어 컬럼 스키마를 정규화하고, 핵심 파생 컬럼과
지역/날짜 차원 키를 생성한 뒤 요약 지표를 산출합니다. 기존 스크립트를
전면 개편하여 파생 로직을 명시적으로 분리하고, 사용자 정의 전처리
요구사항(가격/면적 윈저라이즈, 면적·층 버킷, 신축/구축 플래그 등)을
포함합니다.

예시 실행
--------
python src/etl_realestate.py --data-root data/raw
python src/etl_realestate.py --data-root data/raw --output-duckdb data/processed/realestate.duckdb
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pipelines.example_split_raw import split_by_contract_month

try:  # pragma: no cover - 선택 종속성
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover
    duckdb = None

# ---------------------------------------------------------------------------
# 컬럼 매핑 및 상수
# ---------------------------------------------------------------------------
COL_ALIASES: Dict[str, str] = {
    alias: canonical
    for canonical, aliases in {
        "시도": ["시도", "광역시도", "광역시/도"],
        "시군구": ["시군구", "시/군/구"],
        "읍면동": ["읍면동", "법정동", "동"],
        "법정동코드": ["법정동코드", "법정동 코드", "법정코드"],
        "건축년도": ["건축년도", "건축 연도", "준공년도"],
        "층": ["층", "해당층"],
        "전용면적_㎡": [
            "전용면적(㎡)",
            "전용면적",
            "면적(㎡)",
            "계약면적",
            "계약면적(㎡)",
            "계약면적_㎡",
            "전용/연면적(㎡)",
            "전용/연면적",
        ],
        "연면적_㎡": ["연면적", "연면적(㎡)", "연면적_㎡"],
        "대지면적_㎡": ["대지면적", "대지면적(㎡)", "대지 면적"],
        "거래금액_만원": ["거래금액(만원)", "거래금액", "금액(만원)", "매매금액(만원)"],
        "보증금_만원": ["보증금(만원)", "보증금", "보증금금액(만원)", "보증금액(만원)", "보증금"],
        "월세_만원": ["월세(만원)", "월세", "월세금(만원)", "월세금액(만원)"],
        "계약년월": ["계약년월", "년월"],
        "계약일": ["계약일", "일"],
        "해제사유발생일": ["해제사유발생일", "해제일", "해제 발생일"],
        "계약기간": ["계약기간", "계약 기간"],
        "계약구분": ["계약구분", "계약 구분"],
        "갱신요구권": ["갱신요구권", "갱신 요구권"],
    }.items()
    for alias in aliases
}

NUMERIC_COLUMNS: Sequence[str] = (
    "층",
    "건축년도",
    "전용면적_㎡",
    "연면적_㎡",
    "대지면적_㎡",
    "거래금액_만원",
    "보증금_만원",
    "월세_만원",
    "계약년월",
    "계약일",
)

ESSENTIAL_COLUMNS: Sequence[str] = (
    "거래금액_만원",
    "보증금_만원",
    "월세_만원",
    "전용면적_㎡",
    "계약년월",
    "계약일",
)

TRANSACTION_EXPORT_COLUMNS: Sequence[str] = (
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
    "area_bucket",
    "floor_bucket",
    "거래금액_만원",
    "보증금_만원",
    "월세_만원",
    "환산가_만원",
    "추정매입가_만원",
    "연임대수입_만원",
    "가격_per_㎡",
    "가격_per_평",
    "price_per_m2",
    "price_per_pyeong",
    "Yield_%",
    "yield_imputed",
    "transaction_amount_imputed",
    "transaction_amount_from_trade",
    "transaction_amount_trade_diff_days",
    "floor_insight_enabled",
    "계약일자",
    "YYYYMM",
    "연도",
    "월",
    "분기",
    "해제사유발생일",
    "취소여부",
    "계약기간",
    "계약구분",
    "갱신요구권",
    "contract_start",
    "contract_end",
    "contract_months",
    "contract_is_renewal",
    "contract_extension_flag",
    "contract_stability_score",
    "is_new",
    "is_old",
    "lot_main",
    "lot_sub",
    "property_key",
    "geo_hash",
    "date_key",
    "building_hash",
)

DEFAULT_SOURCE_DIRS: Dict[str, str] = {
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

AREA_BUCKET_BINS = [0, 40, 85, 135, np.inf]
AREA_BUCKET_LABELS = ["소형", "중형", "중대형", "대형"]
FLOOR_BUCKET_BINS = [-np.inf, 5, 15, np.inf]
FLOOR_BUCKET_LABELS = ["저층", "중층", "고층"]
K_CONVERSION = 100
YIELD_PRIMARY_WINDOW_DAYS = 90
YIELD_SECONDARY_WINDOW_DAYS = 180

AREA_FALLBACK_COLUMNS = (
    "연면적_㎡",
    "대지면적_㎡",
)

LOT_NUMBER_COLUMN = "번지"
PROPERTY_KEY_REQUIRED_COMPONENTS = ("sido", "sigungu", "eupmyeondong", "lot_main", "lot_sub")
PROPERTY_OPTIONAL_SOURCE_COLUMNS = ("동", "호")
PROPERTY_KEY_MISSING_VALUE = "missing"
DUAL_PROPERTY_PREFIXES = {"apt", "offi", "row", "det"}
PROPERTY_MATCH_TOLERANCE = 1.5

ADDRESS_TOKEN_PATTERN = re.compile(r"(?:\d{1,2})?월|(?:\d{1,2})?일|주민등록|생년월일|도로명", re.IGNORECASE)


def _init_metadata(df: pd.DataFrame) -> Dict[str, list]:
    df.attrs.setdefault("imputations", [])
    df.attrs.setdefault("drops", [])
    df.attrs.setdefault("issues", [])
    return df.attrs


def _log_imputation(df: pd.DataFrame, column: str, method: str, count: int) -> None:
    if count <= 0:
        return
    meta = _init_metadata(df)
    meta["imputations"].append({"column": column, "method": method, "count": int(count)})


def _log_drop(df: pd.DataFrame, reason: str, count: int) -> None:
    if count <= 0:
        return
    meta = _init_metadata(df)
    meta["drops"].append({"reason": reason, "count": int(count)})


def _assign_meta(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Copy attrs from source dataframe to target and return target."""

    target.attrs = source.attrs
    return target


def _log_issue(df: pd.DataFrame, issue: str, count: int) -> None:
    if count <= 0:
        return
    meta = _init_metadata(df)
    meta["issues"].append({"issue": issue, "count": int(count)})


def fill_with_group_median(
    df: pd.DataFrame,
    column: str,
    *,
    group_cols: Sequence[str],
    default_value: Optional[float] = None,
    include_zero_as_missing: bool = False,
) -> pd.DataFrame:
    if column not in df.columns:
        return df

    series = df[column]
    if include_zero_as_missing:
        missing_mask = series.isna() | (series == 0)
    else:
        missing_mask = series.isna()
    if not missing_mask.any():
        return df

    valid_groups = [col for col in group_cols if col in df.columns]
    if valid_groups:
        group_medians = df.groupby(valid_groups)[column].transform("median")
    else:
        group_medians = pd.Series(np.nan, index=df.index)

    replacements = missing_mask & group_medians.notna()
    if replacements.any():
        df.loc[replacements, column] = group_medians[replacements]
        group_label = "+".join(valid_groups) if valid_groups else "global"
        _log_imputation(df, column, f"median:{group_label}", int(replacements.sum()))

    if default_value is not None:
        remaining = df[column].isna()
        if include_zero_as_missing:
            remaining |= df[column] == 0
        if remaining.any():
            df.loc[remaining, column] = default_value
            _log_imputation(df, column, f"default:{default_value}", int(remaining.sum()))

    return df


def clean_address_tokens(df: pd.DataFrame, column: str = "번지") -> pd.DataFrame:
    if column not in df.columns:
        return df
    series = df[column].astype(str)
    mask = series.str.contains(ADDRESS_TOKEN_PATTERN)
    if mask.any():
        df.loc[mask, column] = pd.NA
        _log_issue(df, f"address_token_pattern:{column}", int(mask.sum()))
    return df


def _normalize_property_component(series: pd.Series) -> pd.Series:
    normalized = series.fillna("")
    normalized = normalized.astype(str).str.strip().str.lower()
    normalized = normalized.replace({"nan": "", "none": ""})
    return normalized


def _split_lot_number(value: object) -> Tuple[str, str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "", ""
    text = str(value).strip()
    if not text:
        return "", ""
    cleaned = re.sub(r"[^0-9-]", "", text)
    if not cleaned:
        return "", ""
    if "-" in cleaned:
        main, sub = cleaned.split("-", 1)
    else:
        main, sub = cleaned, "0"
    main = main.strip()
    sub = sub.strip()
    main_digits = re.sub(r"[^0-9]", "", main)
    sub_digits = re.sub(r"[^0-9]", "", sub)
    if main_digits:
        try:
            main_digits = str(int(main_digits))
        except ValueError:
            main_digits = main_digits.lstrip("0") or "0"
    if sub_digits:
        try:
            sub_digits = str(int(sub_digits))
        except ValueError:
            sub_digits = sub_digits.lstrip("0") or "0"
    return main_digits, sub_digits or "0"


def add_lot_components(df: pd.DataFrame, column: str = LOT_NUMBER_COLUMN) -> pd.DataFrame:
    if column not in df.columns:
        df["lot_main"] = ""
        df["lot_sub"] = ""
        return df

    series = df[column].astype(str).fillna("").str.strip()
    # remove known noise before splitting
    cleaned = series.replace({"nan": "", "None": "", "미상": ""})
    raw_values = cleaned.apply(_split_lot_number)
    lot_pairs = raw_values.tolist()
    if len(lot_pairs) == 0:
        df["lot_main"] = ""
        df["lot_sub"] = ""
        return df
    lot_main, lot_sub = zip(*lot_pairs)
    df["lot_main"] = pd.Series(lot_main, index=df.index, dtype="string")
    df["lot_sub"] = pd.Series(lot_sub, index=df.index, dtype="string")
    df["lot_main"] = df["lot_main"].replace({"<NA>": ""}).fillna("")
    df["lot_sub"] = df["lot_sub"].replace({"<NA>": "0"}).fillna("0")
    return df


def append_property_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        if "property_key" not in df.columns:
            df["property_key"] = PROPERTY_KEY_MISSING_VALUE
        if "lot_main" not in df.columns:
            df["lot_main"] = PROPERTY_KEY_MISSING_VALUE
        if "lot_sub" not in df.columns:
            df["lot_sub"] = "0"
        return df

    if "lot_main" not in df.columns or "lot_sub" not in df.columns:
        df = add_lot_components(df)

    normalized_components: Dict[str, pd.Series] = {}
    for col in PROPERTY_KEY_REQUIRED_COMPONENTS:
        if col not in df.columns:
            df[col] = ""
        series = _normalize_property_component(df[col])
        if col == "lot_main":
            series = series.replace("", PROPERTY_KEY_MISSING_VALUE)
        if col == "lot_sub":
            series = series.replace("", "0")
        normalized_components[col] = series

    key = normalized_components[PROPERTY_KEY_REQUIRED_COMPONENTS[0]].copy()
    for col in PROPERTY_KEY_REQUIRED_COMPONENTS[1:]:
        key = key + "|" + normalized_components[col]

    for optional_col in PROPERTY_OPTIONAL_SOURCE_COLUMNS:
        if optional_col in df.columns:
            optional_series = _normalize_property_component(df[optional_col])
            key = key + "|" + optional_series
        else:
            key = key + "|"

    key = key.str.replace(r"\|{2,}", "|", regex=True)
    key = key.str.replace(r"^\|+", "", regex=True)
    key = key.str.replace(r"\|+$", "", regex=True)
    key = key.replace("", PROPERTY_KEY_MISSING_VALUE)
    key = key.fillna(PROPERTY_KEY_MISSING_VALUE)

    df["property_key"] = key.astype("string")
    df["lot_main"] = normalized_components["lot_main"].astype("string")
    df["lot_sub"] = normalized_components["lot_sub"].astype("string")
    return df


def _compute_property_match_stats(
    df: pd.DataFrame,
    tolerance: float,
) -> Tuple[set[str], List[Dict[str, Any]]]:
    if df.empty or "property_key" not in df.columns:
        return set(), []

    property_keys = df["property_key"].fillna(PROPERTY_KEY_MISSING_VALUE)
    asset_prefix = df["src_type"].astype(str).str.split("_", n=1).str[0]

    matched_keys: set[str] = set()
    stats: List[Dict[str, Any]] = []

    for asset in sorted(DUAL_PROPERTY_PREFIXES):
        asset_mask = asset_prefix == asset
        if not asset_mask.any():
            continue

        asset_df = df.loc[asset_mask].copy()
        asset_df["property_key"] = property_keys.loc[asset_mask]

        valid_mask = asset_df["property_key"] != PROPERTY_KEY_MISSING_VALUE
        trade_df = asset_df[asset_df["src_type"].str.endswith("trade") & valid_mask]
        lease_df = asset_df[asset_df["src_type"].str.endswith("lease") & valid_mask]

        trade_props = set(trade_df["property_key"].unique())
        lease_props = set(lease_df["property_key"].unique())
        common_props = trade_props & lease_props

        area_filtered_props: set[str] = set()
        matched_props = set(common_props)

        if "전용면적_㎡" in trade_df.columns and "전용면적_㎡" in lease_df.columns and common_props:
            trade_medians = (
                trade_df.groupby("property_key")["전용면적_㎡"].median()
            )
            lease_medians = (
                lease_df.groupby("property_key")["전용면적_㎡"].median()
            )
            area_join = pd.concat(
                [trade_medians.rename("trade_median"), lease_medians.rename("lease_median")],
                axis=1,
                join="inner",
            )

            def _area_ok(row: pd.Series) -> bool:
                t = row.get("trade_median")
                l = row.get("lease_median")
                if pd.isna(t) or pd.isna(l):
                    return True
                if min(t, l) <= 0:
                    return True
                ratio = max(t, l) / min(t, l)
                return ratio <= tolerance

            area_join["area_ok"] = area_join.apply(_area_ok, axis=1)
            matched_props = set(area_join.index[area_join["area_ok"]])
            area_filtered_props = set(area_join.index[~area_join["area_ok"]])

        matched_keys.update(matched_props)

        stats.append(
            {
                "asset": asset,
                "rows_total": int(asset_mask.sum()),
                "rows_missing_property": int((asset_mask & (property_keys == PROPERTY_KEY_MISSING_VALUE)).sum()),
                "properties_with_trade": len(trade_props),
                "properties_with_lease": len(lease_props),
                "properties_with_both": len(common_props),
                "properties_area_filtered": len(area_filtered_props),
                "matched_properties": len(matched_props),
            }
        )

    return matched_keys, stats

# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceFile:
    path: Path
    src_type: str
    snapshot_date: Optional[pd.Timestamp]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw real-estate CSVs")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="원본 CSV가 저장된 루트 디렉터리",
    )
    parser.add_argument(
        "--legacy-glob",
        action="store_true",
        help="루트 디렉터리 바로 아래 *_실거래가_YYYYMMDD.csv 파일도 함께 스캔",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="가공 결과를 저장할 디렉터리",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="계약년월 단위 분할만 수행하고 나머지 ETL 단계는 생략",
    )
    parser.add_argument(
        "--split-format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help="split-only 모드에서 생성할 산출물 포맷",
    )
    parser.add_argument(
        "--output-duckdb",
        type=Path,
        default=None,
        help="DuckDB DB 파일 경로(선택)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="각 세그먼트별 처리할 최대 파일 수(스모크 테스트용)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 파일 탐색 및 로딩 도우미
# ---------------------------------------------------------------------------
def detect_snapshot_from_filename(name: str) -> Optional[pd.Timestamp]:
    digits = "".join(ch for ch in name if ch.isdigit())
    if len(digits) >= 8:
        try:
            return pd.to_datetime(digits[:8], format="%Y%m%d", errors="coerce")
        except (ValueError, TypeError):
            return None
    return None


FILENAME_SRC_MAP = {
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


def guess_src_type(filename: str) -> str:
    for key, value in FILENAME_SRC_MAP.items():
        if key in filename:
            return value
    return "unknown"


def list_source_files(
    data_root: Path,
    legacy: bool,
    max_files: Optional[int],
) -> List[SourceFile]:
    sources: List[SourceFile] = []
    for src_type, folder in DEFAULT_SOURCE_DIRS.items():
        base = data_root / folder
        if not base.exists():
            continue
        files = sorted(base.glob("*.csv"))
        if max_files is not None:
            files = files[:max_files]
        for fp in files:
            sources.append(
                SourceFile(
                    path=fp,
                    src_type=src_type,
                    snapshot_date=detect_snapshot_from_filename(fp.name),
                )
            )

    if legacy:
        for fp in sorted(data_root.glob("*_실거래가_*.csv")):
            sources.append(
                SourceFile(
                    path=fp,
                    src_type=guess_src_type(fp.name),
                    snapshot_date=detect_snapshot_from_filename(fp.name),
                )
            )

    return sources


# ---------------------------------------------------------------------------
# CSV 입력 처리
# ---------------------------------------------------------------------------

def resolve_split_formats(option: str) -> Tuple[str, ...]:
    if option == "parquet":
        return ("parquet",)
    if option == "both":
        return ("csv", "parquet")
    return ("csv",)


def run_split_only_mode(
    sources: List[SourceFile], output_dir: Path, export_formats: Tuple[str, ...]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_contracts = 0
    for source in sources:
        snapshot = (
            source.snapshot_date.strftime("%Y%m%d") if source.snapshot_date is not None else None
        )
        try:
            results = split_by_contract_month(
                raw_path=source.path,
                processed_dir=output_dir,
                snapshot=snapshot,
                export_formats=export_formats,
            )
        except Exception as exc:  # pragma: no cover - 방어 로그
            print(f"[split-only] 실패: {source.path.name} -> {exc}")
            continue

        total_contracts += len(results)
        preview = ", ".join(r.contract_ym for r in results[:3]) or "-"
        print(
            f"[split-only] {source.path.name} -> {len(results)}건 (미리보기: {preview})"
        )

    print(
        f"[split-only] 총 {len(sources)}개 원본에서 {total_contracts}건 계약년월 파일 생성 완료"
    )


def detect_header_index(path: Path, encoding: str) -> int:
    with path.open("r", encoding=encoding, errors="ignore") as handle:
        for idx, line in enumerate(handle):
            stripped = line.strip()
            if ("NO" in stripped or "번호" in stripped) and "계약년월" in stripped:
                return idx
    return 0


def read_source_dataframe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
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
            except Exception as exc:  # pragma: no cover
                last_error = exc
                continue
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(f"Failed to load {path}: {last_error}")


# ---------------------------------------------------------------------------
# 전처리 및 파생 로직
# ---------------------------------------------------------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_columns = {col: COL_ALIASES.get(col, col) for col in df.columns}
    return df.rename(columns=new_columns)


def to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(r"[^0-9\.-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if series.empty:
        return series
    quantiles = series.quantile([lower, upper])
    lower_bound = quantiles.iloc[0]
    upper_bound = quantiles.iloc[1]
    if pd.isna(lower_bound) or pd.isna(upper_bound):
        return series
    return series.clip(lower=lower_bound, upper=upper_bound)


def ensure_area_column(df: pd.DataFrame) -> pd.DataFrame:
    if "전용면적_㎡" not in df.columns:
        df["전용면적_㎡"] = pd.NA
    mask = df["전용면적_㎡"].isna() | (df["전용면적_㎡"] <= 0)
    for alt in AREA_FALLBACK_COLUMNS:
        if alt in df.columns:
            candidates = mask & df[alt].notna()
            if candidates.any():
                df.loc[candidates, "전용면적_㎡"] = df.loc[candidates, alt]
                _log_imputation(df, "전용면적_㎡", f"fallback:{alt}", int(candidates.sum()))
            mask = df["전용면적_㎡"].isna() | (df["전용면적_㎡"] <= 0)
            if not mask.any():
                break
    return df


def ensure_transaction_amount(df: pd.DataFrame) -> pd.DataFrame:
    if "거래금액_만원" not in df.columns:
        df["거래금액_만원"] = pd.NA
    df["transaction_amount_imputed"] = pd.Series(False, index=df.index, dtype="boolean")
    df["transaction_amount_from_trade"] = pd.Series(False, index=df.index, dtype="boolean")
    df["transaction_amount_trade_diff_days"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    mask_amount = df["거래금액_만원"].isna() | (df["거래금액_만원"] <= 0)
    if mask_amount.any():
        deposit = df.get("보증금_만원")
        monthly = df.get("월세_만원")
        if deposit is not None or monthly is not None:
            deposit_filled = deposit.fillna(0) if deposit is not None else 0
            monthly_filled = monthly.fillna(0) if monthly is not None else 0
            fallback_amount = deposit_filled + monthly_filled * K_CONVERSION
            if isinstance(fallback_amount, pd.Series):
                replacements = mask_amount & fallback_amount.notna()
                if replacements.any():
                    df.loc[replacements, "거래금액_만원"] = fallback_amount.loc[replacements]
                    df.loc[replacements, "transaction_amount_imputed"] = True
                    _log_imputation(df, "거래금액_만원", "fallback:deposit_plus_rent", int(replacements.sum()))
    return df


def mark_floor_eligible(df: pd.DataFrame) -> pd.DataFrame:
    eligible_types = {
        "apt_trade",
        "apt_lease",
        "offi_trade",
        "offi_lease",
    }
    exclude_types = {"comm_trade", "det_lease"}
    eligible = df["src_type"].isin(eligible_types) & (~df["src_type"].isin(exclude_types))
    df["floor_insight_enabled"] = eligible.astype(int)
    return df


def enrich_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    period_series = df.get("계약기간")
    contract_months = pd.Series(np.nan, index=df.index, dtype="float")
    contract_start = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    contract_end = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    if period_series is not None:
        period_clean = period_series.astype(str).str.strip()
        for idx, value in period_clean.items():
            if not value or value in {"nan", "-", ""}:
                continue
            parts = re.findall(r"\d{6}", value)
            if len(parts) < 2:
                continue
            start_raw, end_raw = parts[0], parts[-1]
            start_ts = pd.to_datetime(start_raw + "01", format="%Y%m%d", errors="coerce")
            end_ts = pd.to_datetime(end_raw + "01", format="%Y%m%d", errors="coerce")
            if pd.isna(start_ts) or pd.isna(end_ts):
                continue
            months = (end_ts.year - start_ts.year) * 12 + (end_ts.month - start_ts.month)
            if months <= 0:
                continue
            contract_months.at[idx] = float(months)
            contract_start.at[idx] = start_ts
            contract_end.at[idx] = end_ts

    df["contract_start"] = contract_start
    df["contract_end"] = contract_end
    df["contract_months"] = contract_months

    contract_type_raw = df.get("계약구분", pd.Series("", index=df.index)).astype(str).str.strip()
    renewal_flag = contract_type_raw.str.contains("갱신", na=False)

    extension_raw = df.get("갱신요구권", pd.Series("", index=df.index)).astype(str).str.strip()
    extension_flag = extension_raw.str.contains("사용|행사", na=False)
    extension_flag &= ~extension_raw.str.contains("미사용|미행사|거절", na=False)
    extension_flag |= extension_raw.str.fullmatch("Y", case=False, na=False)

    length_component = contract_months.fillna(0) / 24.0
    length_component = length_component.clip(lower=0.0, upper=1.0)

    stability_score = (
        0.6 * length_component
        + 0.25 * renewal_flag.astype(float)
        + 0.15 * extension_flag.astype(float)
    )
    stability_score = stability_score.clip(lower=0.0, upper=1.0)

    df["contract_is_renewal"] = renewal_flag.astype(int)
    df["contract_extension_flag"] = extension_flag.astype(int)
    df["contract_stability_score"] = stability_score.astype(float)

    return df


def propagate_contract_metrics_from_lease(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "contract_stability_score" not in df.columns:
        return df
    if "property_key" not in df.columns or "YYYYMM" not in df.columns or "src_type" not in df.columns:
        return df

    work = df.copy()
    # Ensure comparable key types
    work["property_key"] = work["property_key"].astype(str)
    work["YYYYMM"] = pd.to_numeric(work["YYYYMM"], errors="coerce").astype("Int64")

    lease_mask = work["src_type"].astype(str).str.endswith("_lease", na=False)
    trade_mask = work["src_type"].astype(str).str.endswith("_trade", na=False)
    if not lease_mask.any() or not trade_mask.any():
        return df

    metrics_cols = [
        "contract_stability_score",
        "contract_is_renewal",
        "contract_extension_flag",
    ]
    lease_metrics = (
        work.loc[lease_mask & work["property_key"].notna(), ["property_key", "YYYYMM", *metrics_cols]]
        .groupby(["property_key", "YYYYMM"], dropna=False)
        .mean()
        .reset_index()
    )
    if lease_metrics.empty:
        return df

    rename_map = {col: f"{col}_lease_ref" for col in metrics_cols}
    lease_metrics.rename(columns=rename_map, inplace=True)

    work = work.merge(lease_metrics, on=["property_key", "YYYYMM"], how="left")
    trade_mask = work["src_type"].astype(str).str.endswith("_trade", na=False)
    for col in metrics_cols:
        ref_col = f"{col}_lease_ref"
        if ref_col not in work.columns:
            continue
        mask = trade_mask & work[ref_col].notna()
        if mask.any():
            if col in {"contract_is_renewal", "contract_extension_flag"}:
                work.loc[mask, col] = (
                    work.loc[mask, ref_col]
                    .round()
                    .astype("Int64")
                )
            else:
                work.loc[mask, col] = work.loc[mask, ref_col]
        work.drop(columns=[ref_col], inplace=True, errors="ignore")

    return work


# ---------------------------------------------------------------------------
# 수익률 보강 유틸
# ---------------------------------------------------------------------------


def _compute_effective_monthly_rent(
    deposit: Optional[pd.Series],
    monthly: Optional[pd.Series],
) -> pd.Series:
    if deposit is not None:
        deposit_series = deposit.fillna(0).astype(float)
        index = deposit_series.index
    else:
        deposit_series = None
        index = None

    if monthly is not None:
        monthly_series = monthly.fillna(0).astype(float)
        index = monthly_series.index if index is None else index
    else:
        monthly_series = None

    if index is None:
        return pd.Series(dtype=float)

    if deposit_series is None:
        deposit_component = pd.Series(0.0, index=index)
    else:
        deposit_component = (deposit_series / K_CONVERSION).reindex(index, fill_value=0.0)

    if monthly_series is None:
        monthly_component = pd.Series(0.0, index=index)
    else:
        monthly_component = monthly_series.reindex(index, fill_value=0.0)

    return deposit_component + monthly_component


def _match_trade_with_lease(
    trade_df: pd.DataFrame,
    lease_df: pd.DataFrame,
    *,
    max_days: int,
) -> pd.DataFrame:
    if trade_df.empty or lease_df.empty:
        return pd.DataFrame(columns=["trade_index", "lease_annual_rent", "date_diff_days"])

    trade_cols = ["trade_index", "property_key", "asset_type", "계약일자_trade"]
    lease_cols = ["property_key", "asset_type", "계약일자_lease", "lease_annual_rent"]
    trade = trade_df.loc[:, trade_cols].copy()
    lease = lease_df.loc[:, lease_cols].copy()

    trade = trade.sort_values(["property_key", "asset_type", "계약일자_trade"]).reset_index(drop=True)
    lease = lease.sort_values(["property_key", "asset_type", "계약일자_lease"]).reset_index(drop=True)
    if trade.empty or lease.empty:
        return pd.DataFrame(columns=["trade_index", "lease_annual_rent", "date_diff_days"])

    lease_groups = {}
    for key, sub in lease.groupby(["property_key", "asset_type"]):
        sub = sub.dropna(subset=["계약일자_lease", "lease_annual_rent"])
        if sub.empty:
            continue
        sub_sorted = sub.sort_values("계약일자_lease")
        lease_groups[key] = (
            sub_sorted["계약일자_lease"].to_numpy(dtype="datetime64[ns]"),
            sub_sorted["lease_annual_rent"].to_numpy(dtype=float),
        )

    if not lease_groups:
        return pd.DataFrame(columns=["trade_index", "lease_annual_rent", "date_diff_days"])

    matches = []
    for key, trades in trade.groupby(["property_key", "asset_type"]):
        payload = lease_groups.get(key)
        if payload is None:
            continue
        lease_dates, lease_rents = payload
        if lease_dates.size == 0:
            continue
        for row in trades.itertuples(index=False):
            trade_date = np.datetime64(row.계약일자_trade.to_datetime64())
            delta_days = np.abs((lease_dates - trade_date).astype('timedelta64[D]')).astype(int)
            within = delta_days <= max_days
            if not np.any(within):
                continue
            min_delta = delta_days[within].min()
            candidate_idx = np.flatnonzero(delta_days == min_delta)
            chosen = candidate_idx[0]
            matches.append((int(row.trade_index), float(lease_rents[chosen]), int(min_delta)))

    if not matches:
        return pd.DataFrame(columns=["trade_index", "lease_annual_rent", "date_diff_days"])

    result = pd.DataFrame(matches, columns=["trade_index", "lease_annual_rent", "date_diff_days"])
    return result

def _match_lease_with_trade(
    lease_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    *,
    max_days: int,
) -> pd.DataFrame:
    if lease_df.empty or trade_df.empty:
        return pd.DataFrame(columns=["lease_index", "trade_amount", "date_diff_days"])

    lease_cols = ["lease_index", "property_key", "asset_type", "계약일자_lease"]
    trade_cols = ["property_key", "asset_type", "계약일자_trade", "trade_amount"]
    lease = lease_df.loc[:, lease_cols].copy()
    trade = trade_df.loc[:, trade_cols].copy()

    trade_groups: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for key, sub in trade.groupby(["property_key", "asset_type"]):
        sub = sub.dropna(subset=["계약일자_trade", "trade_amount"])
        if sub.empty:
            continue
        sub_sorted = sub.sort_values("계약일자_trade")
        trade_groups[key] = (
            sub_sorted["계약일자_trade"].to_numpy(dtype="datetime64[ns]"),
            sub_sorted["trade_amount"].to_numpy(dtype=float),
        )

    if not trade_groups:
        return pd.DataFrame(columns=["lease_index", "trade_amount", "date_diff_days"])

    matches: List[Tuple[int, float, int]] = []
    for row in lease.itertuples(index=False):
        key = (row.property_key, row.asset_type)
        payload = trade_groups.get(key)
        if payload is None:
            continue
        trade_dates, trade_amounts = payload
        if trade_dates.size == 0:
            continue
        lease_date = getattr(row, "계약일자_lease", None)
        if pd.isna(lease_date):
            continue
        lease_np = np.datetime64(lease_date.to_datetime64())
        delta_days = np.abs(trade_dates - lease_np).astype('timedelta64[D]').astype(int)
        within = delta_days <= max_days
        if not np.any(within):
            continue
        min_delta = delta_days[within].min()
        candidate_idx = np.flatnonzero(delta_days == min_delta)
        chosen = candidate_idx[0]
        matches.append((int(row.lease_index), float(trade_amounts[chosen]), int(min_delta)))

    if not matches:
        return pd.DataFrame(columns=["lease_index", "trade_amount", "date_diff_days"])

    return pd.DataFrame(matches, columns=["lease_index", "trade_amount", "date_diff_days"])


def enrich_lease_amount_from_trade(
    df: pd.DataFrame,
    *,
    primary_days: int = YIELD_PRIMARY_WINDOW_DAYS,
    secondary_days: int = YIELD_SECONDARY_WINDOW_DAYS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty or "거래금액_만원" not in df.columns:
        return df, {}
    if "src_type" not in df.columns or "property_key" not in df.columns:
        return df, {}

    if "transaction_amount_imputed" not in df.columns:
        df["transaction_amount_imputed"] = pd.Series(False, index=df.index, dtype="boolean")
    if "transaction_amount_from_trade" not in df.columns:
        df["transaction_amount_from_trade"] = pd.Series(False, index=df.index, dtype="boolean")
    if "transaction_amount_trade_diff_days" not in df.columns:
        df["transaction_amount_trade_diff_days"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    src_types = df["src_type"].astype("string")
    asset_types = src_types.str.split("_", n=1).str[0]

    lease_mask = src_types.str.endswith("lease")
    trade_mask = src_types.str.endswith("trade")
    if not lease_mask.any() or not trade_mask.any():
        return df, {}

    amount = df["거래금액_만원"]
    imputed = df["transaction_amount_imputed"].fillna(False)
    needs_mask = lease_mask & (amount.isna() | (amount <= 0) | imputed)
    lease_candidates = df.loc[needs_mask].copy()
    lease_candidates = lease_candidates[
        lease_candidates["property_key"].notna()
        & (lease_candidates["property_key"] != PROPERTY_KEY_MISSING_VALUE)
        & lease_candidates["계약일자"].notna()
    ]
    if lease_candidates.empty:
        return df, {}

    trade_df = df.loc[trade_mask].copy()
    trade_df = trade_df[
        trade_df["거래금액_만원"].notna()
        & (trade_df["거래금액_만원"] > 0)
        & trade_df["property_key"].notna()
        & (trade_df["property_key"] != PROPERTY_KEY_MISSING_VALUE)
        & trade_df["계약일자"].notna()
    ]
    if trade_df.empty:
        return df, {}

    lease_candidates = lease_candidates.assign(
        lease_index=lease_candidates.index,
        asset_type=asset_types.loc[lease_candidates.index].values,
        계약일자_lease=lease_candidates["계약일자"],
    )
    trade_df = trade_df.assign(
        asset_type=asset_types.loc[trade_df.index].values,
        계약일자_trade=trade_df["계약일자"],
        trade_amount=trade_df["거래금액_만원"],
    )

    summary: Dict[str, Any] = {
        "lease_candidates": int(len(lease_candidates)),
        "window_primary_days": primary_days,
        "window_secondary_days": secondary_days,
    }

    def _apply_matches(match_df: pd.DataFrame) -> int:
        if match_df.empty:
            return 0
        idx = pd.Index(match_df["lease_index"].astype(df.index.dtype, copy=False))
        df.loc[idx, "거래금액_만원"] = match_df["trade_amount"].values
        df.loc[idx, "transaction_amount_imputed"] = False
        df.loc[idx, "transaction_amount_from_trade"] = True
        df.loc[idx, "transaction_amount_trade_diff_days"] = match_df["date_diff_days"].values
        return int(len(match_df))

    matches_primary = _match_lease_with_trade(lease_candidates, trade_df, max_days=primary_days)
    resolved_primary = _apply_matches(matches_primary)
    if resolved_primary:
        summary["matched_trade_primary"] = resolved_primary

    remaining_mask = lease_mask & df["transaction_amount_imputed"].fillna(False)
    if remaining_mask.any():
        secondary_candidates = df.loc[remaining_mask].copy()
        secondary_candidates = secondary_candidates[
            secondary_candidates["property_key"].notna()
            & (secondary_candidates["property_key"] != PROPERTY_KEY_MISSING_VALUE)
            & secondary_candidates["계약일자"].notna()
        ]
        if not secondary_candidates.empty:
            secondary_candidates = secondary_candidates.assign(
                lease_index=secondary_candidates.index,
                asset_type=asset_types.loc[secondary_candidates.index].values,
                계약일자_lease=secondary_candidates["계약일자"],
            )
            matches_secondary = _match_lease_with_trade(secondary_candidates, trade_df, max_days=secondary_days)
            resolved_secondary = _apply_matches(matches_secondary)
            if resolved_secondary:
                summary["matched_trade_secondary"] = resolved_secondary

    still_imputed = int(df.loc[lease_mask, "transaction_amount_imputed"].fillna(False).sum())
    summary["lease_amount_from_trade"] = int(df.loc[lease_mask, "transaction_amount_from_trade"].fillna(False).sum())
    summary["lease_amount_still_imputed"] = still_imputed

    return df, summary


def _prepare_group_rent_stats(lease_df: pd.DataFrame) -> Dict[str, Dict[tuple, float]]:
    if lease_df.empty:
        return {}

    lease_df = lease_df.copy()
    lease_df["sigungu_key"] = lease_df["sigungu"].fillna("미상").astype(str)
    lease_df["sido_key"] = lease_df["sido"].fillna("미상").astype(str)
    lease_df["area_bucket_key"] = lease_df["area_bucket"].astype("string").fillna("미상")

    stats: Dict[str, Dict[tuple, float]] = {}
    stats["asset_sigungu_area"] = (
        lease_df.groupby(["asset_type", "sigungu_key", "area_bucket_key"], dropna=False)["lease_annual_rent"].mean().to_dict()
    )
    stats["asset_sido_area"] = (
        lease_df.groupby(["asset_type", "sido_key", "area_bucket_key"], dropna=False)["lease_annual_rent"].mean().to_dict()
    )
    stats["asset_sigungu"] = (
        lease_df.groupby(["asset_type", "sigungu_key"], dropna=False)["lease_annual_rent"].mean().to_dict()
    )
    stats["asset_sido"] = (
        lease_df.groupby(["asset_type", "sido_key"], dropna=False)["lease_annual_rent"].mean().to_dict()
    )
    stats["asset"] = lease_df.groupby(["asset_type"], dropna=False)["lease_annual_rent"].mean().to_dict()
    return stats


def _lookup_group_annual_rent(
    row: pd.Series,
    stats: Dict[str, Dict[tuple, float]],
) -> float | None:
    asset = row.get("asset_type")
    if not asset:
        return None

    sigungu = str(row.get("sigungu") or "미상")
    sido = str(row.get("sido") or "미상")
    area_bucket = row.get("area_bucket")
    area_key = str(area_bucket) if pd.notna(area_bucket) else "미상"

    keys = [
        ("asset_sigungu_area", (asset, sigungu, area_key)),
        ("asset_sido_area", (asset, sido, area_key)),
        ("asset_sigungu", (asset, sigungu)),
        ("asset_sido", (asset, sido)),
        ("asset", (asset,)),
    ]

    for bucket, key in keys:
        value = stats.get(bucket, {}).get(key)
        if value and value > 0:
            return float(value)
    return None


def enrich_trade_yield_from_rent(
    df: pd.DataFrame,
    *,
    primary_days: int = YIELD_PRIMARY_WINDOW_DAYS,
    secondary_days: int = YIELD_SECONDARY_WINDOW_DAYS,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    summary: Dict[str, int] = {}
    required_cols = {"src_type", "property_key", "계약일자", "추정매입가_만원", "Yield_%"}
    if df.empty or not required_cols.issubset(df.columns):
        return df, summary

    if "yield_imputed" not in df.columns:
        df["yield_imputed"] = pd.Series(False, index=df.index, dtype="boolean")
    else:
        df["yield_imputed"] = df["yield_imputed"].astype("boolean")

    src_types = df["src_type"].astype(str)
    asset_types = src_types.str.split("_", n=1).str[0]
    trade_mask = src_types.str.endswith("trade")
    lease_mask = src_types.str.endswith("lease")

    trade_indices = df.index[trade_mask]
    if trade_indices.empty:
        return df, summary

    lease_indices = df.index[lease_mask]
    lease_df = df.loc[lease_indices].copy()
    lease_df = lease_df[lease_df["property_key"].notna()]
    lease_df = lease_df[lease_df["property_key"] != PROPERTY_KEY_MISSING_VALUE]
    lease_df = lease_df[lease_df["계약일자"].notna()]

    if lease_df.empty:
        return df, summary

    lease_df = lease_df.assign(
        asset_type=asset_types.loc[lease_df.index].values,
        effective_monthly=_compute_effective_monthly_rent(
            lease_df.get("보증금_만원"),
            lease_df.get("월세_만원"),
        ),
    )
    lease_df["lease_annual_rent"] = lease_df["effective_monthly"] * 12
    lease_df = lease_df[lease_df["lease_annual_rent"] > 0]
    lease_df = lease_df.rename(columns={"계약일자": "계약일자_lease"})
    lease_info = lease_df[[
        "property_key",
        "asset_type",
        "계약일자_lease",
        "lease_annual_rent",
        "sido",
        "sigungu",
        "area_bucket",
    ]].copy()

    if lease_info.empty:
        return df, summary

    trade_df = df.loc[trade_indices].copy()
    trade_df = trade_df[trade_df["추정매입가_만원"].notna() & (trade_df["추정매입가_만원"] > 0)]
    trade_df = trade_df[trade_df["계약일자"].notna()]
    if trade_df.empty:
        return df, summary

    trade_df = trade_df.assign(
        asset_type=asset_types.loc[trade_df.index].values,
        trade_index=trade_df.index,
    )
    trade_df = trade_df[trade_df["property_key"].notna()]
    trade_df = trade_df[trade_df["property_key"] != PROPERTY_KEY_MISSING_VALUE]

    if trade_df.empty:
        return df, summary

    existing_yield = df.loc[trade_df.index, "Yield_%"].fillna(0)
    trade_needs = trade_df[existing_yield <= 0].copy()
    if trade_needs.empty:
        return df, summary

    summary["trade_candidates"] = int(len(trade_needs))

    trade_needs = trade_needs.rename(columns={"계약일자": "계약일자_trade"})

    def _assign_yield_from_matches(matches: pd.DataFrame, *, imputed: bool) -> int:
        if matches.empty:
            return 0
        idx = pd.Index(matches["trade_index"].astype(df.index.dtype, copy=False))
        numerator = pd.Series(matches["lease_annual_rent"].values, index=idx)
        denominator = df.loc[idx, "추정매입가_만원"]
        yields = safe_divide(numerator, denominator) * 100
        valid = yields.notna() & (yields > 0)
        if not valid.any():
            return 0
        valid_indices = valid.index[valid]
        df.loc[valid_indices, "Yield_%"] = yields.loc[valid]
        df.loc[valid_indices, "yield_imputed"] = imputed
        return int(valid.sum())

    matched_primary = _match_trade_with_lease(trade_needs, lease_info, max_days=primary_days)
    resolved_primary = _assign_yield_from_matches(matched_primary, imputed=False)
    if resolved_primary:
        summary["matched_property_primary"] = resolved_primary

    unresolved_mask = df.loc[trade_needs["trade_index"], "Yield_%"].fillna(0) <= 0
    unresolved_indices = pd.Index(trade_needs.loc[unresolved_mask.values, "trade_index"])

    if not unresolved_indices.empty:
        trade_secondary = trade_needs.loc[trade_needs["trade_index"].isin(unresolved_indices)].copy()
        matched_secondary = _match_trade_with_lease(trade_secondary, lease_info, max_days=secondary_days)
        resolved_secondary = _assign_yield_from_matches(matched_secondary, imputed=False)
        if resolved_secondary:
            summary["matched_property_secondary"] = resolved_secondary

        unresolved_mask = df.loc[trade_needs["trade_index"], "Yield_%"].fillna(0) <= 0
        unresolved_indices = pd.Index(trade_needs.loc[unresolved_mask.values, "trade_index"])

    if not unresolved_indices.empty:
        trade_final = trade_needs.loc[trade_needs["trade_index"].isin(unresolved_indices)].copy()
        group_stats = _prepare_group_rent_stats(lease_info)
        aggregated: Dict[int, float] = {}
        for _, row in trade_final.iterrows():
            annual = _lookup_group_annual_rent(row, group_stats)
            if annual and annual > 0:
                aggregated[int(row["trade_index"])] = float(annual)
        if aggregated:
            idx = pd.Index(list(aggregated.keys()))
            numerator = pd.Series(list(aggregated.values()), index=idx)
            denominator = df.loc[idx, "추정매입가_만원"]
            yields = safe_divide(numerator, denominator) * 100
            valid = yields.notna() & (yields > 0)
            if valid.any():
                valid_indices = valid.index[valid]
                df.loc[valid_indices, "Yield_%"] = yields.loc[valid]
                df.loc[valid_indices, "yield_imputed"] = True
                summary["imputed_group_average"] = int(valid.sum())

        unresolved_mask = df.loc[trade_needs["trade_index"], "Yield_%"].fillna(0) <= 0
        unresolved_indices = pd.Index(trade_needs.loc[unresolved_mask.values, "trade_index"])

    if not unresolved_indices.empty:
        summary["unresolved_trade"] = int(len(unresolved_indices))
        df.loc[unresolved_indices, "Yield_%"] = np.nan

    post_yield = df.loc[trade_needs["trade_index"], "Yield_%"]
    resolved_mask = post_yield.notna() & (post_yield > 0)
    resolved_count = int(resolved_mask.sum())
    if resolved_count:
        summary["resolved_trade"] = resolved_count
    trade_candidates = summary.get("trade_candidates", 0)
    if trade_candidates:
        coverage = resolved_count / trade_candidates
        summary["yield_calc_coverage"] = round(float(coverage), 4)
    else:
        summary["yield_calc_coverage"] = 0.0
    summary["yield_imputed_true"] = int(
        df.loc[trade_needs["trade_index"], "yield_imputed"].fillna(False).sum()
    )

    return df, summary


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while protecting against zero/negative denominators."""

    denom = denominator.where(denominator > 0)
    result = numerator.divide(denom)
    return result.replace([np.inf, -np.inf], np.nan)


def build_contract_date(row: pd.Series) -> Optional[pd.Timestamp]:
    yyyymm = row.get("계약년월")
    day = row.get("계약일")
    if pd.isna(yyyymm):
        return None
    try:
        yyyymm_int = int(float(yyyymm))
    except (TypeError, ValueError):
        return None
    year, month = divmod(yyyymm_int, 100)
    try:
        day_int = int(float(day)) if pd.notna(day) else 1
    except (TypeError, ValueError):
        day_int = 1
    try:
        return pd.Timestamp(year=year, month=month, day=day_int)
    except ValueError:
        # 비정상 일자는 해당 월의 마지막 날짜로 보정
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def normalize_region_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        tokens = df["sigungu"].apply(
            lambda value: value.split() if isinstance(value, str) else []
        )

        def _safe_join(parts: List[str], upto: int) -> str:
            if not parts:
                return "미상"
            subset = parts[:upto]
            if not subset:
                return parts[0]
            return " ".join(subset)

        def _leftover(parts: List[str]) -> str:
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

    for col in ("sido", "sigungu", "eupmyeondong"):
        if col in df.columns:
            df[col] = df[col].replace({"nan": "미상", "NaN": "미상"}).fillna("미상")

    return df


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ESSENTIAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    if "해제사유발생일" not in df.columns:
        df["해제사유발생일"] = pd.NA
    for optional in ("계약기간", "계약구분", "갱신요구권"):
        if optional not in df.columns:
            df[optional] = pd.NA
    return df


def preprocess_dataframe(raw_df: pd.DataFrame, *, src_type: str) -> pd.DataFrame:
    df = standardize_columns(raw_df).copy()
    _init_metadata(df)
    df = ensure_required_columns(df)
    df = clean_address_tokens(df)
    df = add_lot_components(df)
    df = normalize_region_columns(df)
    df["src_type"] = src_type

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    if "층" in df.columns:
        invalid_floor = df["층"].notna() & ((df["층"] < -5) | (df["층"] > 200))
        if invalid_floor.any():
            df.loc[invalid_floor, "층"] = np.nan
            _log_imputation(df, "층", "drop_out_of_range", int(invalid_floor.sum()))

    if "건축년도" in df.columns:
        current_year = pd.Timestamp.today().year
        invalid_year = df["건축년도"].notna() & (
            (df["건축년도"] < 1960) | (df["건축년도"] > current_year)
        )
        if invalid_year.any():
            df.loc[invalid_year, "건축년도"] = np.nan
            _log_imputation(df, "건축년도", "drop_out_of_range", int(invalid_year.sum()))

    if "보증금_만원" in df.columns:
        df = fill_with_group_median(
            df,
            "보증금_만원",
            group_cols=("sigungu", "src_type"),
            default_value=0.0,
        )
    if "월세_만원" in df.columns:
        df = fill_with_group_median(
            df,
            "월세_만원",
            group_cols=("sigungu", "src_type"),
            default_value=0.0,
        )

    df = ensure_area_column(df)
    df = ensure_transaction_amount(df)
    df = mark_floor_eligible(df)
    df = enrich_contract_features(df)

    if "거래금액_만원" in df.columns:
        df["거래금액_만원"] = winsorize_series(df["거래금액_만원"])
    if "전용면적_㎡" in df.columns:
        df["전용면적_㎡"] = winsorize_series(df["전용면적_㎡"])
        df["전용면적_㎡"] = df["전용면적_㎡"].where(df["전용면적_㎡"] > 0)

    if {"계약년월", "계약일"}.issubset(df.columns):
        df["계약일자"] = df.apply(build_contract_date, axis=1)
        df["YYYYMM"] = df["계약일자"].dt.strftime("%Y%m").astype("Int64")
        df["연도"] = df["계약일자"].dt.year.astype("Int64")
        df["월"] = df["계약일자"].dt.month.astype("Int64")
        df["분기"] = df["계약일자"].dt.quarter.astype("Int64")
    else:
        df["계약일자"] = pd.NaT
        df["YYYYMM"] = pd.NA
        df["연도"] = pd.NA
        df["월"] = pd.NA
        df["분기"] = pd.NA

    if "전용면적_㎡" in df.columns:
        df["면적_평"] = df["전용면적_㎡"] * 0.3025
    else:
        df["면적_평"] = pd.NA

    if "전용면적_㎡" in df.columns:
        df["전용면적_㎡"] = df["전용면적_㎡"].where(df["전용면적_㎡"] > 0)
    if "면적_평" in df.columns:
        df["면적_평"] = df["면적_평"].where(df["면적_평"] > 0)

    if {"거래금액_만원", "전용면적_㎡"}.issubset(df.columns):
        per_m2 = safe_divide(df["거래금액_만원"], df["전용면적_㎡"])
        per_pyeong = safe_divide(df["거래금액_만원"], df["면적_평"])
        df["가격_per_㎡"] = per_m2
        df["가격_per_평"] = per_pyeong
    else:
        df["가격_per_㎡"] = pd.NA
        df["가격_per_평"] = pd.NA

    df["price_per_m2"] = df["가격_per_㎡"]
    df["price_per_pyeong"] = df["가격_per_평"]

    if "전용면적_㎡" in df.columns:
        df["area_bucket"] = pd.cut(
            df["전용면적_㎡"],
            bins=AREA_BUCKET_BINS,
            labels=AREA_BUCKET_LABELS,
            include_lowest=True,
        )
    else:
        df["area_bucket"] = pd.Categorical([np.nan] * len(df), categories=AREA_BUCKET_LABELS)

    if "층" in df.columns:
        df["floor_bucket"] = pd.cut(
            df["층"].fillna(0),
            bins=FLOOR_BUCKET_BINS,
            labels=FLOOR_BUCKET_LABELS,
            include_lowest=True,
        )
        if "floor_insight_enabled" in df.columns:
            df["floor_bucket"] = df["floor_bucket"].where(df["floor_insight_enabled"] == 1)
        if "src_type" in df.columns:
            exclude_types = {"comm_trade", "det_lease"}
            mask_exclude = df["src_type"].isin(exclude_types)
            df.loc[mask_exclude, "floor_bucket"] = pd.NA
    else:
        df["floor_bucket"] = pd.Categorical([np.nan] * len(df), categories=FLOOR_BUCKET_LABELS)

    if "해제사유발생일" in df.columns:
        cleaned_cancel = (
            df["해제사유발생일"].astype(str)
            .str.strip()
            .replace({"": pd.NA, "-": pd.NA, "nan": pd.NA, "None": pd.NA})
            .str.replace(r"\.0$", "", regex=True)
        )
        df["해제사유발생일"] = pd.to_datetime(cleaned_cancel, format="%Y%m%d", errors="coerce")
        df["취소여부"] = df["해제사유발생일"].notna().astype(int)
    else:
        df["해제사유발생일"] = pd.NaT
        df["취소여부"] = 0

    if "보증금_만원" in df.columns or "월세_만원" in df.columns:
        df["환산가_만원"] = df.get("보증금_만원", 0) + df.get("월세_만원", 0) * K_CONVERSION
    else:
        df["환산가_만원"] = pd.NA

    if "거래금액_만원" in df.columns:
        df["추정매입가_만원"] = np.where(
            df["거래금액_만원"].notna(),
            df["거래금액_만원"],
            df["환산가_만원"],
        )
    else:
        df["추정매입가_만원"] = df["환산가_만원"]

    if "월세_만원" in df.columns:
        df["연임대수입_만원"] = df["월세_만원"] * 12
    else:
        df["연임대수입_만원"] = pd.NA

    df["추정매입가_만원"] = df["추정매입가_만원"].where(df["추정매입가_만원"] > 0)
    df["연임대수입_만원"] = df["연임대수입_만원"].where(df["연임대수입_만원"] >= 0)

    df["Yield_%"] = safe_divide(df["연임대수입_만원"], df["추정매입가_만원"]) * 100

    current_year = pd.Timestamp.today().year
    if "건축년도" in df.columns:
        df["is_new"] = (current_year - df["건축년도"]) <= 5
        df["is_old"] = (current_year - df["건축년도"]) >= 20
        df.loc[df["건축년도"].isna(), ["is_new", "is_old"]] = False
    else:
        df["is_new"] = False
        df["is_old"] = False

    date_missing = df["계약일자"].isna().sum()
    if date_missing:
        _log_issue(df, "missing_contract_date", int(date_missing))

    essential_mask = df["전용면적_㎡"].notna() & (df["전용면적_㎡"] > 0)
    essential_mask &= df["추정매입가_만원"].notna() & (df["추정매입가_만원"] > 0)
    removed = (~essential_mask).sum()
    if removed:
        _log_drop(df, "missing_core_metrics", int(removed))
    df = _assign_meta(df, df[essential_mask].copy())

    df.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)
    return df


def drop_duplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    original_len = len(df)
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
        if col in df.columns
    ]
    if subset:
        deduped = df.drop_duplicates(subset=subset)
        removed = original_len - len(deduped)
        if removed:
            _log_drop(df, "duplicate_transactions", removed)
        return _assign_meta(df, deduped)
    return df


def build_geo_hash(row: pd.Series) -> str:
    key = "|".join(
        str(row.get(col, "") or "") for col in ("시도", "시군구", "읍면동", "법정동코드")
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
    # ?? ??? ?? ??? ??? ??? ????, ?? ?? ?? ?? ????.
    df = normalize_region_columns(df)

    region_fallbacks = (
        ("sido", "시도"),
        ("sigungu", "시군구"),
        ("eupmyeondong", "읍면동"),
    )
    for normalized, original in region_fallbacks:
        if normalized not in df.columns and original in df.columns:
            df[normalized] = df[original]
        if normalized not in df.columns:
            df[normalized] = ""
        series = df[normalized].astype(str).str.strip()
        series = series.replace({"nan": "", "NaN": "", "None": ""})
        if original in df.columns:
            fallback = df[original].astype(str).str.strip()
            series = series.mask(series.eq(""), fallback)
        df[normalized] = series.replace("", "미상").fillna("미상")
    df["geo_hash"] = df.apply(build_geo_hash, axis=1)
    df["date_key"] = df["계약일자"].dt.strftime("%Y%m%d").astype("Int64")
    building_attrs = df[["건축년도", "층"]].copy().fillna(-1).astype(int)
    df["building_hash"] = (
        building_attrs.astype(str)
        .agg("|".join, axis=1)
        .map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    )
    df = append_property_key(df)
    return df


# ---------------------------------------------------------------------------
# 집계 및 산출물 유지
# ---------------------------------------------------------------------------

def compute_monthly_basics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "YYYYMM" not in df.columns:
        return pd.DataFrame()
    monthly = (
        df.dropna(subset=["YYYYMM"])
        .groupby(["src_type", "sido", "sigungu", "eupmyeondong", "YYYYMM"], dropna=False)
        .agg(
            txn_cnt=("거래금액_만원", "count"),
            avg_p_per_m2=("가격_per_㎡", "mean"),
            std_p_per_m2=("가격_per_㎡", "std"),
            avg_yield_pct=("Yield_%", "mean"),
            avg_contract_stability=("contract_stability_score", "mean"),
            cancel_rate=("취소여부", "mean"),
        )
        .reset_index()
    )
    monthly["std_p_per_m2"] = monthly["std_p_per_m2"].fillna(0)
    if "avg_contract_stability" in monthly.columns:
        monthly["avg_contract_stability"] = monthly["avg_contract_stability"].fillna(0)
    monthly["cancel_rate"] = monthly["cancel_rate"].fillna(0)
    return monthly.loc[:, ~monthly.columns.str.fullmatch("index")].reset_index(drop=True)


def compute_momentum(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    monthly = monthly.sort_values(["src_type", "sido", "sigungu", "eupmyeondong", "YYYYMM"])
    monthly["avg_p_per_m2_lag3"] = (
        monthly.groupby(["src_type", "sido", "sigungu", "eupmyeondong"])["avg_p_per_m2"].shift(3)
    )
    monthly["mom_3m"] = monthly["avg_p_per_m2"] / monthly["avg_p_per_m2_lag3"] - 1
    monthly.loc[monthly["avg_p_per_m2_lag3"].isna(), "mom_3m"] = 0
    monthly.loc[monthly["mom_3m"].replace([np.inf, -np.inf], np.nan).isna(), "mom_3m"] = 0
    monthly = monthly.loc[:, ~monthly.columns.str.fullmatch("index")]  # drop stray index column
    return monthly.reset_index(drop=True)


def compute_volatility(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    vol = monthly.copy()
    vol["cv_p_per_m2"] = vol["std_p_per_m2"] / vol["avg_p_per_m2"].replace(0, np.nan)
    vol["cv_p_per_m2"] = vol["cv_p_per_m2"].replace([np.inf, -np.inf], np.nan).fillna(0)
    if "avg_contract_stability" in vol.columns:
        vol["stability"] = vol["avg_contract_stability"].fillna(0)
    elif "contract_stability_score" in vol.columns:
        vol["stability"] = vol["contract_stability_score"].fillna(0)
    else:
        vol["stability"] = 0.0
    vol = vol.loc[:, ~vol.columns.str.fullmatch("index")]  # ensure index column removed
    return vol.reset_index(drop=True)


def build_property_matched_dataset(
    df: pd.DataFrame, *, tolerance: float = PROPERTY_MATCH_TOLERANCE
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty or "property_key" not in df.columns:
        return pd.DataFrame(), {}

    property_keys = df["property_key"].fillna(PROPERTY_KEY_MISSING_VALUE)
    asset_prefix = df["src_type"].astype(str).str.split("_", n=1).str[0]

    matched_keys, asset_stats = _compute_property_match_stats(df, tolerance)
    if not asset_stats:
        return pd.DataFrame(), {}

    dual_mask = asset_prefix.isin(DUAL_PROPERTY_PREFIXES)
    matched_mask = df["property_key"].isin(matched_keys)
    retain_mask = ~dual_mask | (dual_mask & matched_mask)

    combined_df = df.loc[retain_mask].copy()

    for entry in asset_stats:
        asset = entry["asset"]
        asset_mask = asset_prefix == asset
        entry["row_retained"] = int((asset_mask & retain_mask).sum())
        entry["row_dropped"] = int(entry["rows_total"] - entry["row_retained"])

    summary: Dict[str, Any] = {
        "tolerance_ratio": tolerance,
        "rows_total": int(dual_mask.sum()),
        "rows_retained": int((dual_mask & retain_mask).sum()),
        "rows_dropped": int((dual_mask & (~retain_mask)).sum()),
        "rows_missing_property": int((dual_mask & (property_keys == PROPERTY_KEY_MISSING_VALUE)).sum()),
        "matched_properties": int(sum(entry.get("matched_properties", 0) for entry in asset_stats)),
        "assets": asset_stats,
    }

    return combined_df, summary


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


def persist_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_to_write = sanitize_for_parquet(df)
    df_to_write = df_to_write.drop(columns=["index"], errors="ignore")
    if duckdb is not None:
        con = duckdb.connect()
        try:
            con.register("df_temp", df_to_write.reset_index(drop=True))
            con.execute(f"COPY df_temp TO '{path.as_posix()}' (FORMAT PARQUET)")
        finally:
            con.close()
    else:
        df_to_write.reset_index(drop=True).to_parquet(path, index=False)


def persist_transactions_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    datetime_cols = out.select_dtypes(include="datetime64[ns]").columns
    for col in datetime_cols:
        out[col] = out[col].dt.strftime("%Y-%m-%d")
    string_like = out.select_dtypes(include="object").columns
    if len(string_like) > 0:
        out[string_like] = out[string_like].fillna("")
    out.to_csv(path, index=False, na_rep="")


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


# ---------------------------------------------------------------------------
# 품질 경고 및 로깅 유틸
# ---------------------------------------------------------------------------

QUALITY_KEY_COLUMNS = (
    "거래금액_만원",
    "전용면적_㎡",
    "면적_평",
    "가격_per_㎡",
    "가격_per_평",
    "층",
    "건축년도",
    "보증금_만원",
    "월세_만원",
)


QUALITY_LOGS: List[Dict[str, Any]] = []


def assess_data_quality(
    df: pd.DataFrame,
    source_path: Path,
    *,
    null_threshold: float = 0.4,
    src_type: Optional[str] = None,
) -> None:
    """Collect data-quality stats and print warnings when thresholds are exceeded."""

    entry: Dict[str, Any] = {
        "file": source_path.name,
        "rows": int(len(df)),
        "issues": [],
        "null_ratio": {},
    }
    if df.empty:
        print(f"[WARN] {source_path.name}: preprocessing produced 0 rows")
        entry["issues"].append("no_rows")
        QUALITY_LOGS.append(entry)
        return

    non_floor_types = {"comm_trade", "comm_lease", "det_trade", "det_lease"}
    skip_floor_check = src_type in non_floor_types

    for col in QUALITY_KEY_COLUMNS:
        if col not in df.columns:
            continue
        if col == "층":
            if skip_floor_check:
                continue
            if "src_type" in df.columns:
                unique_types = set(df["src_type"].dropna().unique())
                if unique_types and unique_types.issubset(non_floor_types):
                    continue
            if "floor_insight_enabled" in df.columns and df["floor_insight_enabled"].sum() == 0:
                continue
        ratio = df[col].isna().mean()
        entry["null_ratio"][col] = round(float(ratio), 4)
        if ratio >= null_threshold:
            entry["issues"].append({"column": col, "null_ratio": round(float(ratio), 4)})

    if entry["issues"]:
        joined = ", ".join(
            f"{issue['column']} {issue['null_ratio']*100:.0f}% NaN"
            if isinstance(issue, dict)
            else str(issue)
            for issue in entry["issues"]
        )
        print(f"[WARN] {source_path.name}: quality issues -> {joined}")

    meta = getattr(df, "attrs", {})
    if meta.get("imputations"):
        entry["imputations"] = meta["imputations"]
    if meta.get("drops"):
        entry["drops"] = meta["drops"]
    if meta.get("issues"):
        entry["extra_issues"] = meta["issues"]

    QUALITY_LOGS.append(entry)


def persist_quality_report(path: Path, report: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 파이프라인 orchestrator
# ---------------------------------------------------------------------------

def load_sources(sources: Iterable[SourceFile]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for source in sources:
        try:
            df_raw = read_source_dataframe(source.path)
            df_norm = preprocess_dataframe(df_raw, src_type=source.src_type)
            df_norm["snapshot_date"] = source.snapshot_date
            assess_data_quality(df_norm, source.path, src_type=source.src_type)
            frames.append(df_norm)
        except Exception as exc:
            print(f"[WARN] failed to load {source.path}: {exc}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = drop_duplicate_transactions(combined)
    return combined


def main() -> None:
    args = parse_args()
    sources = list_source_files(args.data_root, legacy=args.legacy_glob, max_files=args.max_files)

    if not sources:
        print("No source files discovered; check data-root or enable --legacy-glob")
        return

    if args.split_only:
        formats = resolve_split_formats(args.split_format)
        run_split_only_mode(sources, args.output_dir, formats)
        return

    tidy = load_sources(sources)
    if tidy.empty:
        print("Loaded zero rows from CSVs")
        return

    tidy = add_dimension_keys(tidy)

    tidy, lease_amount_summary = enrich_lease_amount_from_trade(tidy)
    if lease_amount_summary:
        QUALITY_LOGS.append({"file": "__lease_amount_enrichment__", **lease_amount_summary})

    tidy, yield_summary = enrich_trade_yield_from_rent(tidy)
    if yield_summary:
        QUALITY_LOGS.append({"file": "__yield_enrichment__", **yield_summary})

    tidy = propagate_contract_metrics_from_lease(tidy)

    combined_tidy, match_summary = build_property_matched_dataset(tidy)
    if not combined_tidy.empty:
        combined_tidy = propagate_contract_metrics_from_lease(combined_tidy)

    monthly = compute_monthly_basics(tidy)
    momentum = compute_momentum(monthly.copy())
    volatility = compute_volatility(monthly.copy())

    combined_monthly = pd.DataFrame()
    combined_momentum = pd.DataFrame()
    combined_volatility = pd.DataFrame()
    if not combined_tidy.empty:
        combined_monthly = compute_monthly_basics(combined_tidy)
        combined_momentum = compute_momentum(combined_monthly.copy())
        combined_volatility = compute_volatility(combined_monthly.copy())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_cols = [c for c in TRANSACTION_EXPORT_COLUMNS if c in tidy.columns]
    persist_transactions_csv(tidy[export_cols], args.output_dir / "transactions.csv")
    persist_parquet(monthly, args.output_dir / "monthly_basics.parquet")
    persist_parquet(momentum, args.output_dir / "monthly_momentum.parquet")
    persist_parquet(volatility, args.output_dir / "monthly_volatility.parquet")

    if not combined_tidy.empty:
        combined_export_cols = [c for c in TRANSACTION_EXPORT_COLUMNS if c in combined_tidy.columns]
        persist_transactions_csv(
            combined_tidy[combined_export_cols],
            args.output_dir / "transactions_combined.csv",
        )
        persist_parquet(combined_monthly, args.output_dir / "monthly_basics_combined.parquet")
        persist_parquet(combined_momentum, args.output_dir / "monthly_momentum_combined.parquet")
        persist_parquet(
            combined_volatility,
            args.output_dir / "monthly_volatility_combined.parquet",
        )

    if match_summary:
        QUALITY_LOGS.append(
            {
                "file": "__property_matching__",
                "rows": int(len(tidy)),
                "retained_rows": int(len(combined_tidy)),
                "match": match_summary,
            }
        )

    if QUALITY_LOGS:
        persist_quality_report(args.output_dir / "quality_report.json", QUALITY_LOGS)
    else:
        persist_quality_report(args.output_dir / "quality_report.json", [])

    if args.output_duckdb:
        persist_duckdb(args.output_duckdb, tidy, monthly, momentum, volatility)

    print(
        f"Processed {len(tidy):,} rows from {len(sources)} files. "
        f"Outputs stored in {args.output_dir}."
    )


if __name__ == "__main__":
    main()
