#!/usr/bin/env python3
"""Compare ETL performance metrics between two log files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

DURATION_PATTERNS = [
    re.compile(r"elapsed time[:=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"total runtime[:=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"real\s*([0-9]+\.?[0-9]*)s"),
]

ELAPSED_WALL_PATTERN = re.compile(
    r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:]+\.?[0-9]*)",
    re.IGNORECASE,
)

CPU_TIME_PATTERNS = {
    "user_seconds": re.compile(r"User time \(seconds\):\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
    "system_seconds": re.compile(r"System time \(seconds\):\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
    "max_rss_kb": re.compile(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", re.IGNORECASE),
}


def parse_wall_clock(value: str) -> Optional[float]:
    if not value:
        return None
    parts = value.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 1:
        return parts[0]
    return None


def extract_metrics(path: Path) -> Dict[str, Optional[float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    warn_count = sum(1 for line in lines if "[WARN]" in line)
    error_count = sum(1 for line in lines if "[ERROR]" in line)
    traceback_count = sum(1 for line in lines if "Traceback (most recent call last)" in line)

    elapsed_seconds: Optional[float] = None
    for pattern in DURATION_PATTERNS:
        for match in pattern.finditer(text):
            try:
                elapsed_seconds = float(match.group(1))
            except ValueError:
                continue
    if elapsed_seconds is None:
        wall_match = ELAPSED_WALL_PATTERN.search(text)
        if wall_match:
            elapsed_seconds = parse_wall_clock(wall_match.group(1))

    cpu_stats: Dict[str, Optional[float]] = {key: None for key in CPU_TIME_PATTERNS}
    for key, pattern in CPU_TIME_PATTERNS.items():
        match = pattern.search(text)
        if match:
            try:
                cpu_stats[key] = float(match.group(1))
            except ValueError:
                cpu_stats[key] = None

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "line_count": len(lines),
        "warn_count": warn_count,
        "error_count": error_count,
        "traceback_count": traceback_count,
        "elapsed_seconds": elapsed_seconds,
        **cpu_stats,
    }


def diff_metrics(current: Dict[str, Optional[float]], baseline: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    diff: Dict[str, Optional[float]] = {}
    keys = {
        "size_bytes",
        "line_count",
        "warn_count",
        "error_count",
        "traceback_count",
        "elapsed_seconds",
        "user_seconds",
        "system_seconds",
        "max_rss_kb",
    }
    for key in keys:
        current_value = current.get(key)
        baseline_value = baseline.get(key)
        if current_value is None or baseline_value is None:
            diff[key] = None
        else:
            diff[key] = float(current_value) - float(baseline_value)
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, help="Path to current ETL log")
    parser.add_argument("--baseline", required=True, help="Path to baseline ETL log")
    parser.add_argument("--out", help="Optional path to write comparison JSON")
    args = parser.parse_args()

    current_path = Path(args.current).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()

    for path in (current_path, baseline_path):
        if not path.exists():
            raise SystemExit(f"Log file not found: {path}")

    current_metrics = extract_metrics(current_path)
    baseline_metrics = extract_metrics(baseline_path)
    diff = diff_metrics(current_metrics, baseline_metrics)

    report = {
        "current": current_metrics,
        "baseline": baseline_metrics,
        "diff": diff,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out:
        output_path = Path(args.out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
