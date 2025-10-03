#!/usr/bin/env bash
set -euo pipefail
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
STREAMLIT_BROWSER="${STREAMLIT_BROWSER:-0}"
STREAMLIT_DATA_DIR="${STREAMLIT_DATA_DIR:-app/cache}"

export STREAMLIT_BROWSER
export STREAMLIT_DATA_DIR

exec streamlit run app/dashboard_unified.py --server.port="$STREAMLIT_PORT"
