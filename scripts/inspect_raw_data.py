#!/usr/bin/env python
"""Raw data quality inspector for cherrysona_property."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

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

ESSENTIAL_COLUMNS = (
    "거래금액_만원",
    "보증금_만원",
    "월세_만원",
    "전용면적_㎡",
    "계약년월",
    "계약일",
)

NUMERIC_COLUMNS = (
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


def read_source_dataframe(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949", "utf-8"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                dtype=str,
                na_values=["", "-", " "],
                keep_default_na=True,
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise ValueError(f"Failed to load {path}: {last_error}")


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


def detect_address_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    anomalies: Dict[str, Any] = {}
    column = "번지"
    if column not in df.columns:
        return anomalies
    series = df[column].astype(str).fillna("")
    pattern_mask = series.str.contains(r"월|일|시|분", na=False)
    count = int(pattern_mask.sum())
    if count:
        anomalies["address_tokens"] = {
            "column": column,
            "count": count,
            "sample": series[pattern_mask].head(3).tolist(),
        }
    return anomalies


def inspect_file(path: Path) -> Dict[str, Any]:
    raw = read_source_dataframe(path)
    df = standardize_columns(raw)
    entry: Dict[str, Any] = {
        "file": path.name,
        "rows": int(len(df)),
        "missing_columns": [],
        "null_ratio": {},
        "numeric_parse_failure": {},
        "anomalies": {},
    }

    missing = [col for col in ESSENTIAL_COLUMNS if col not in df.columns]
    if missing:
        entry["missing_columns"] = missing

    for col in df.columns:
        entry["null_ratio"][col] = round(float(df[col].isna().mean()), 4)

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col]
        converted = to_numeric(series)
        failures = int(series.notna().sum() - converted.notna().sum())
        if failures > 0:
            entry["numeric_parse_failure"][col] = {
                "count": failures,
                "ratio": round(float(failures / max(1, len(series))), 4),
            }

    entry["anomalies"].update(detect_address_anomalies(df))
    return entry


def collect_files(data_root: Path) -> List[Path]:
    files: List[Path] = []
    for folder in DEFAULT_SOURCE_DIRS.values():
        base = data_root / folder
        if not base.exists():
            continue
        files.extend(sorted(base.glob("*.csv")))
    return files


def run_inspection(data_root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "data_root": str(data_root),
        "files": [],
        "summary": {
            "total_rows": 0,
            "total_files": 0,
            "columns_missing": {},
        },
    }
    for csv_path in collect_files(data_root):
        entry = inspect_file(csv_path)
        report["files"].append(entry)
        report["summary"]["total_files"] += 1
        report["summary"]["total_rows"] += entry["rows"]
        for col in entry.get("missing_columns", []):
            report["summary"]["columns_missing"].setdefault(col, 0)
            report["summary"]["columns_missing"][col] += 1
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw real-estate CSV quality")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="원본 CSV 디렉터리 루트",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/raw_data_report.json"),
        help="진단 결과를 저장할 JSON 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_inspection(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"[INFO] Raw data report saved to {args.output}")
    print(f"[INFO] 총 {report['summary']['total_files']}개 파일, {report['summary']['total_rows']:,}건 데이터 검사 완료")
if __name__ == "__main__":
    main()
