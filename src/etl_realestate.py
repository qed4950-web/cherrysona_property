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
        "전용면적_㎡": ["전용면적(㎡)", "전용면적", "면적(㎡)", "계약면적", "계약면적(㎡)", "계약면적_㎡"],
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
AREA_FALLBACK_COLUMNS = (
    "연면적_㎡",
    "대지면적_㎡",
)

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
            df.loc[mask, "전용면적_㎡"] = df.loc[mask, alt]
            mask = df["전용면적_㎡"].isna() | (df["전용면적_㎡"] <= 0)
            if not mask.any():
                break
    return df


def ensure_transaction_amount(df: pd.DataFrame) -> pd.DataFrame:
    if "거래금액_만원" not in df.columns:
        df["거래금액_만원"] = pd.NA
    mask_amount = df["거래금액_만원"].isna() | (df["거래금액_만원"] <= 0)
    if mask_amount.any():
        deposit = df.get("보증금_만원")
        monthly = df.get("월세_만원")
        if deposit is not None or monthly is not None:
            deposit_filled = deposit.fillna(0) if deposit is not None else 0
            monthly_filled = monthly.fillna(0) if monthly is not None else 0
            fallback_amount = deposit_filled + monthly_filled * K_CONVERSION
            if isinstance(fallback_amount, pd.Series):
                df.loc[mask_amount, "거래금액_만원"] = fallback_amount.loc[mask_amount]
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
    df = ensure_required_columns(df)
    df = normalize_region_columns(df)
    df["src_type"] = src_type

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    if "보증금_만원" in df.columns:
        df["보증금_만원"] = df["보증금_만원"].fillna(0)
    if "월세_만원" in df.columns:
        df["월세_만원"] = df["월세_만원"].fillna(0)

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

    essential_mask = df["전용면적_㎡"].notna() & (df["전용면적_㎡"] > 0)
    essential_mask &= df["추정매입가_만원"].notna() & (df["추정매입가_만원"] > 0)
    df = df[essential_mask].copy()

    df.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)
    return df


def drop_duplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
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
        return df.drop_duplicates(subset=subset)
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
    if "시도" in df.columns:
        df["sido"] = df["시도"]
    if "시군구" in df.columns:
        df["sigungu"] = df["시군구"]
    if "읍면동" in df.columns:
        df["eupmyeondong"] = df["읍면동"]
    df["geo_hash"] = df.apply(build_geo_hash, axis=1)
    df["date_key"] = df["계약일자"].dt.strftime("%Y%m%d").astype("Int64")
    building_attrs = df[["건축년도", "층"]].copy().fillna(-1).astype(int)
    df["building_hash"] = (
        building_attrs.astype(str)
        .agg("|".join, axis=1)
        .map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    )
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
            cancel_rate=("취소여부", "mean"),
        )
        .reset_index()
    )
    monthly["std_p_per_m2"] = monthly["std_p_per_m2"].fillna(0)
    monthly["cancel_rate"] = monthly["cancel_rate"].fillna(0)
    return monthly


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
    return monthly


def compute_volatility(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    vol = monthly.copy()
    vol["cv_p_per_m2"] = vol["std_p_per_m2"] / vol["avg_p_per_m2"].replace(0, np.nan)
    vol["cv_p_per_m2"] = vol["cv_p_per_m2"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return vol


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

    monthly = compute_monthly_basics(tidy)
    momentum = compute_momentum(monthly.copy())
    volatility = compute_volatility(monthly.copy())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_cols = [c for c in TRANSACTION_EXPORT_COLUMNS if c in tidy.columns]
    persist_transactions_csv(tidy[export_cols], args.output_dir / "transactions.csv")
    persist_parquet(monthly, args.output_dir / "monthly_basics.parquet")
    persist_parquet(momentum, args.output_dir / "monthly_momentum.parquet")
    persist_parquet(volatility, args.output_dir / "monthly_volatility.parquet")

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
