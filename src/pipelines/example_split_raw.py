"""원본 실거래 CSV를 계약년월 단위로 잘라 처리하는 간단한 파이프라인 예시.

한 번에 모든 파일을 돌리다가 중간에 멈추는 상황을 피하려면, 이 스크립트로
원본 CSV를 하나씩 지정해서 가공하면 됩니다. 기본값은 아파트 매매 데이터지만
``--raw`` 옵션으로 다른 유형(오피스텔, 상업용 등)을 순차적으로 처리할 수
있습니다.

예시 실행
---------
.venv/bin/python src/pipelines/example_split_raw.py \
    --raw data/raw/apt_trade/아파트(매매)_실거래가.csv \
    --format both
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

MONEY_KEYWORDS = ("금액", "보증금", "월세", "가격")


@dataclass(frozen=True)
class SplitResult:
    """처리된 산출물의 기본 메타데이터."""

    contract_ym: str
    snapshot: str
    outputs: Dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="계약년월 기준으로 원본 실거래 CSV를 나눠서 저장합니다."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/apt_trade/아파트(매매)_실거래가.csv"),
        help="처리할 원본 CSV 경로",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="생성될 산출물을 저장할 폴더",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="파일명에 들어갈 스냅샷 라벨 (미지정 시 파일명에서 추출)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help="산출물 저장 포맷",
    )
    return parser.parse_args()


def _ensure_processed_dir(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)


def load_raw_dataframe(raw_path: Path) -> pd.DataFrame:
    """BOM이 포함된 CSV를 안전하게 읽고, 금액 관련 컬럼의 쉼표를 제거한다."""

    df = pd.read_csv(raw_path, encoding="utf-8-sig", low_memory=False)
    money_columns = [col for col in df.columns if any(key in col for key in MONEY_KEYWORDS)]
    for col in money_columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .replace({"-": pd.NA, "": pd.NA, "nan": pd.NA})
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce")
    return df


def detect_snapshot(raw_path: Path) -> str:
    """파일명에서 숫자만 뽑아 스냅샷 라벨을 추측한다."""

    digits = "".join(ch for ch in raw_path.stem if ch.isdigit())
    return digits if digits else "manual"


def _normalise_contract(contract_value: object) -> str:
    value = str(contract_value)
    return value.strip().split(".")[0]


def _export_name(raw_path: Path, contract_ym: str, snapshot: str, suffix: str) -> str:
    src_token = raw_path.parent.name or "raw"
    stem_token = "".join(ch if ch.isalnum() else "_" for ch in raw_path.stem).strip("_")
    if stem_token and stem_token.lower() not in src_token.lower():
        base = f"{src_token}_{stem_token}"
    else:
        base = src_token
    return f"{base}_{contract_ym}_snapshot_{snapshot}.{suffix}"


def split_by_contract_month(
    raw_path: Path,
    processed_dir: Path,
    snapshot: str | None = None,
    export_formats: Iterable[str] = ("csv",),
) -> List[SplitResult]:
    """원본 한 개를 계약년월별 파일로 분리한다."""

    _ensure_processed_dir(processed_dir)

    formats: Set[str] = {fmt.lower() for fmt in export_formats}
    invalid = formats - {"csv", "parquet"}
    if invalid:
        raise ValueError(f"지원하지 않는 포맷: {sorted(invalid)}")

    resolved_snapshot = snapshot or detect_snapshot(raw_path)
    df = load_raw_dataframe(raw_path)

    if "계약년월" not in df.columns:
        raise ValueError("원본 데이터에 '계약년월' 컬럼이 필요합니다.")

    results: List[SplitResult] = []
    for contract_ym, group in df.groupby("계약년월"):
        contract_ym_str = _normalise_contract(contract_ym)
        export_base_kwargs = {
            "raw_path": raw_path,
            "contract_ym": contract_ym_str,
            "snapshot": resolved_snapshot,
        }

        group = group.copy()
        group["snapshot_date"] = resolved_snapshot
        group["contract_yyyymm"] = contract_ym_str

        outputs: Dict[str, Path] = {}
        if "csv" in formats:
            csv_name = _export_name(suffix="csv", **export_base_kwargs)
            csv_path = processed_dir / csv_name
            group.to_csv(csv_path, index=False)
            outputs["csv"] = csv_path
        if "parquet" in formats:
            parquet_name = _export_name(suffix="parquet", **export_base_kwargs)
            parquet_path = processed_dir / parquet_name
            group.to_parquet(parquet_path, engine="pyarrow", index=False)
            outputs["parquet"] = parquet_path

        results.append(
            SplitResult(
                contract_ym=contract_ym_str,
                snapshot=resolved_snapshot,
                outputs=outputs,
            )
        )

    return results


def main() -> None:
    args = parse_args()
    formats = ("csv",)
    if args.format == "parquet":
        formats = ("parquet",)
    elif args.format == "both":
        formats = ("csv", "parquet")

    results = split_by_contract_month(
        raw_path=args.raw,
        processed_dir=args.processed_dir,
        snapshot=args.snapshot,
        export_formats=formats,
    )

    preview = results[:3]
    for result in preview:
        targets = ", ".join(f"{kind}:{path.name}" for kind, path in result.outputs.items())
        print(
            f"saved ({targets}) "
            f"[계약년월={result.contract_ym}, snapshot={result.snapshot}]"
        )
    if len(results) > len(preview):
        print(f"... {len(results) - len(preview)} more 계약년월 처리 완료")


if __name__ == "__main__":
    main()
