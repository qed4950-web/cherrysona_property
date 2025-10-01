#!/usr/bin/env python3
"""CLI diagnostics for DuckDB queries used by the Streamlit dashboard."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import duckdb

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.duckdb_queries import (  # noqa: E402
    QuerySpec,
    EXPLORER_COLUMNS,
    fact_transactions,
    monthly_basics,
    monthly_momentum,
    monthly_volatility,
    region_options,
)


@dataclass
class QueryMetrics:
    name: str
    iteration: int
    row_count: Optional[int] = None
    count_ms: Optional[float] = None
    sample_ms: Optional[float] = None
    sample_rows: Optional[int] = None
    sample_path: Optional[str] = None
    explain_path: Optional[str] = None
    error: Optional[str] = None


def build_query(name: str, args: argparse.Namespace) -> QuerySpec:
    if name == "monthly_basics":
        return monthly_basics(
            args.src_type,
            sido=args.sido,
            sigungu=args.sigungu,
            ym_from=args.ym_from,
            ym_to=args.ym_to,
        )
    if name == "monthly_momentum":
        return monthly_momentum(
            args.src_type,
            sido=args.sido,
            sigungu=args.sigungu,
            ym_from=args.ym_from,
            ym_to=args.ym_to,
        )
    if name == "monthly_volatility":
        return monthly_volatility(
            args.src_type,
            sido=args.sido,
            sigungu=args.sigungu,
            ym_from=args.ym_from,
            ym_to=args.ym_to,
        )
    if name == "explorer":
        return fact_transactions(
            args.src_type,
            sido=args.sido,
            sigungu=args.sigungu,
            ym_from=args.ym_from,
            ym_to=args.ym_to,
            limit=args.limit,
            offset=args.offset,
            sample=not args.no_sample,
            columns=EXPLORER_COLUMNS,
        )
    if name == "region_options":
        return region_options()
    raise ValueError(f"Unknown query name: {name}")


def run_query(
    con: duckdb.DuckDBPyConnection,
    name: str,
    iteration: int,
    spec: QuerySpec,
    *,
    sample_limit: Optional[int],
    output_dir: Optional[Path],
    explain: bool,
) -> QueryMetrics:
    metrics = QueryMetrics(name=name, iteration=iteration)
    params = list(spec.params)
    try:
        count_sql = f"SELECT count(*) AS cnt FROM ({spec.sql})"
        start = time.perf_counter()
        metrics.row_count = int(con.execute(count_sql, params).fetchone()[0])
        metrics.count_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:  # noqa: BLE001
        metrics.error = f"COUNT failed: {exc}"
        return metrics

    if sample_limit and sample_limit > 0:
        sample_sql = f"SELECT * FROM ({spec.sql}) LIMIT {int(sample_limit)}"
        try:
            start = time.perf_counter()
            relation = con.execute(sample_sql, params)
            metrics.sample_ms = (time.perf_counter() - start) * 1000
            if pd is not None:
                sample_df = relation.fetch_df()
                metrics.sample_rows = len(sample_df)
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    sample_path = output_dir / f"{iteration:02d}_{name}.csv"
                    sample_df.to_csv(sample_path, index=False)
                    metrics.sample_path = str(sample_path)
            else:
                sample_rows = relation.fetchall()
                metrics.sample_rows = len(sample_rows)
        except Exception as exc:  # noqa: BLE001
            metrics.error = f"Sample fetch failed: {exc}"
            return metrics

    if explain:
        try:
            explain_sql = f"EXPLAIN ANALYZE {spec.sql}"
            output_dir = output_dir or REPO_ROOT / "diagnostics"
            output_dir.mkdir(parents=True, exist_ok=True)
            explain_path = output_dir / f"{iteration:02d}_{name}_explain.txt"
            with explain_path.open("w", encoding="utf-8") as fh:
                fh.write(con.execute(explain_sql, params).fetchone()[0])
            metrics.explain_path = str(explain_path)
        except Exception as exc:  # noqa: BLE001
            metrics.error = f"EXPLAIN failed: {exc}"

    return metrics


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, default=None, help="Path to DuckDB file. Default: env CHERRYSONA_DUCKDB")
    parser.add_argument("--src-type", required=True, help="Source type (e.g. apt_trade)")
    parser.add_argument("--sido", help="Filter by 시도")
    parser.add_argument("--sigungu", help="Filter by 시군구")
    parser.add_argument("--ym-from", type=int, dest="ym_from", help="Filter lower bound (YYYYMM)")
    parser.add_argument("--ym-to", type=int, dest="ym_to", help="Filter upper bound (YYYYMM)")
    parser.add_argument("--limit", type=int, default=2000, help="Explorer limit/sample size (default 2000)")
    parser.add_argument("--offset", type=int, default=0, help="Explorer offset when not sampling")
    parser.add_argument("--no-sample", action="store_true", help="Disable sampling for explorer query")
    parser.add_argument("--queries", nargs="+", default=["monthly_basics", "monthly_momentum", "monthly_volatility", "explorer", "region_options"], choices=["monthly_basics", "monthly_momentum", "monthly_volatility", "explorer", "region_options"], help="Queries to execute")
    parser.add_argument("--sample-limit", type=int, default=200, help="Rows to fetch for sample preview (0 to skip)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the full query set N times")
    parser.add_argument("--output-json", type=Path, help="Persist metrics as JSON")
    parser.add_argument("--output-dir", type=Path, help="Directory to store CSV samples/explain plans")
    parser.add_argument("--explain", action="store_true", help="Also capture EXPLAIN ANALYZE output")
    return parser.parse_args(argv)


def resolve_duckdb_path(args: argparse.Namespace) -> Path:
    if args.duckdb:
        return args.duckdb
    env_path = os.environ.get("CHERRYSONA_DUCKDB")
    if env_path:
        return Path(env_path)
    raise SystemExit("DuckDB path not provided. Use --duckdb or set CHERRYSONA_DUCKDB.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    db_path = resolve_duckdb_path(args)
    if not db_path.exists():
        raise SystemExit(f"DuckDB file does not exist: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)
    results: List[QueryMetrics] = []
    try:
        for iteration in range(1, args.repeat + 1):
            print(f"\nIteration {iteration}/{args.repeat}")
            for name in args.queries:
                spec = build_query(name, args)
                print(f"  -> {name}", end="", flush=True)
                metrics = run_query(
                    con,
                    name,
                    iteration,
                    spec,
                    sample_limit=args.sample_limit,
                    output_dir=args.output_dir,
                    explain=args.explain,
                )
                results.append(metrics)
                if metrics.error:
                    print(f" ... ERROR: {metrics.error}")
                else:
                    info = [
                        f"rows={metrics.row_count}",
                        f"count_ms={metrics.count_ms:.2f}",
                    ]
                    if metrics.sample_ms is not None:
                        info.append(f"sample_ms={metrics.sample_ms:.2f}")
                        info.append(f"sample_rows={metrics.sample_rows}")
                    print(" ... " + ", ".join(info))
    finally:
        con.close()

    if args.output_json:
        payload: List[Dict[str, object]] = [asdict(r) for r in results]
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved metrics to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
