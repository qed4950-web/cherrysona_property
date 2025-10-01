"""Reusable DuckDB query builders for dashboard and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

EXPLORER_COLUMNS: Tuple[str, ...] = (
    "src_type",
    "snapshot_date",
    "sido",
    "sigungu",
    "eupmyeondong",
    "YYYYMM",
    "계약일자",
    "거래금액_만원",
    "보증금_만원",
    "월세_만원",
    "전용면적_㎡",
    "층",
    "건축년도",
    "환산가_만원",
    "Yield_%",
)


@dataclass(frozen=True)
class QuerySpec:
    """Encapsulates a SQL statement with parameters for execution."""

    sql: str
    params: Sequence[object]


def _build_common_filters(
    base_sql: str,
    src_type: str,
    sido: Optional[str],
    sigungu: Optional[str],
    ym_from: Optional[int],
    ym_to: Optional[int],
) -> QuerySpec:
    sql = [base_sql]
    params: List[object] = [src_type]
    if sido:
        sql.append("AND sido = ?")
        params.append(sido)
    if sigungu:
        sql.append("AND sigungu = ?")
        params.append(sigungu)
    if ym_from:
        sql.append("AND yyyymm >= ?")
        params.append(int(ym_from))
    if ym_to:
        sql.append("AND yyyymm <= ?")
        params.append(int(ym_to))
    return QuerySpec("\n".join(sql), params)


def monthly_basics(
    src_type: str,
    *,
    sido: Optional[str] = None,
    sigungu: Optional[str] = None,
    ym_from: Optional[int] = None,
    ym_to: Optional[int] = None,
) -> QuerySpec:
    base = """
        SELECT * FROM vw_monthly_basics
        WHERE src_type = ?
    """.strip()
    return _build_common_filters(base, src_type, sido, sigungu, ym_from, ym_to)


def monthly_momentum(
    src_type: str,
    *,
    sido: Optional[str] = None,
    sigungu: Optional[str] = None,
    ym_from: Optional[int] = None,
    ym_to: Optional[int] = None,
) -> QuerySpec:
    base = """
        SELECT * FROM vw_monthly_momentum
        WHERE src_type = ?
    """.strip()
    return _build_common_filters(base, src_type, sido, sigungu, ym_from, ym_to)


def monthly_volatility(
    src_type: str,
    *,
    sido: Optional[str] = None,
    sigungu: Optional[str] = None,
    ym_from: Optional[int] = None,
    ym_to: Optional[int] = None,
) -> QuerySpec:
    base = """
        SELECT *
        FROM vw_monthly_volatility
        WHERE src_type = ?
    """.strip()
    spec = _build_common_filters(base, src_type, sido, sigungu, ym_from, ym_to)
    sql = "\n".join([spec.sql, "ORDER BY yyyymm"])
    return QuerySpec(sql, spec.params)


def fact_transactions(
    src_type: str,
    *,
    sido: Optional[str] = None,
    sigungu: Optional[str] = None,
    ym_from: Optional[int] = None,
    ym_to: Optional[int] = None,
    limit: int = 2000,
    offset: int = 0,
    sample: bool = True,
    columns: Iterable[str] = EXPLORER_COLUMNS,
) -> QuerySpec:
    safe_limit = max(1, min(int(limit), 5000))
    safe_offset = max(0, int(offset))
    quoted_columns = ", ".join(f'"{col}"' for col in columns)
    base_sql = [
        "SELECT",
        quoted_columns,
        "FROM fact_transactions",
        "WHERE src_type = ?",
    ]
    params: List[object] = [src_type]
    if sido:
        base_sql.append("AND sido = ?")
        params.append(sido)
    if sigungu:
        base_sql.append("AND sigungu = ?")
        params.append(sigungu)
    if ym_from:
        base_sql.append("AND yyyymm >= ?")
        params.append(int(ym_from))
    if ym_to:
        base_sql.append("AND yyyymm <= ?")
        params.append(int(ym_to))

    if sample:
        base_sql.append(f"USING SAMPLE {safe_limit} ROWS")
    else:
        base_sql.append("ORDER BY yyyymm")
        base_sql.append("LIMIT ?")
        base_sql.append("OFFSET ?")
        params.extend([safe_limit, safe_offset])

    return QuerySpec("\n".join(base_sql), params)


def region_options() -> QuerySpec:
    sql = """
        SELECT DISTINCT src_type, sido AS 시도, sigungu AS 시군구
        FROM fact_transactions
    """.strip()
    return QuerySpec(sql, [])
