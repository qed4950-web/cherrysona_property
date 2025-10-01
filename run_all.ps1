# 실행 위치: C:\python\github\부동산1차\cherrysona_property

# 1. 가상환경 활성화
.\.venv\Scripts\activate

# 2. ETL 실행
python src\etl_realestate.py --data-root data\raw --output-dir data\processed --output-duckdb data\processed\realestate.duckdb

# 3. 캐시 삭제
Remove-Item -Recurse -Force app\cache\* -ErrorAction SilentlyContinue
streamlit cache clear

# 4. 대시보드 실행
streamlit run app\dashboard.py

#사용법 경로 바꾸고 .\run_all.ps1

# 윈도우 PowerShell에서 실행 → 자동으로 WSL 내부 진입

wsl -e bash -lic "
cd /mnt/c/python/github/부동산1차/cherrysona_property && \
source .venv/bin/activate && \
python src/etl_realestate.py --data-root data/raw --output-dir data/processed --output-duckdb data/processed/realestate.duckdb && \
rm -rf app/cache/* && \
streamlit cache clear && \
streamlit run app/dashboard.py
"
# 1. 가상환경 활성화 (이미 되어있으면 생략)
.\.venv\Scripts\activate

# 2. streamlit 실행
streamlit run dashboard.py