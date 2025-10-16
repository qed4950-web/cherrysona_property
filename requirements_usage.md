# 환경 설정 & 실행 가이드

## 1. 가상환경 생성 (선택)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. 필수 패키지 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

DuckDB 연결을 사용할 계획이면 `duckdb` 패키지가 이미 requirements에 포함돼 있으므로 별도 설치가 필요 없습니다. DuckDB 데이터 파일이 없다면 Parquet/CSV 모드로 그대로 사용하면 됩니다.

## 3. 대시보드 실행
```bash
scripts/run_dashboard.sh
```
환경변수:
- `STREAMLIT_PORT` (기본 8501)
- `STREAMLIT_DATA_DIR` (기본 `app/cache`)
- `CHERRYSONA_DUCKDB` (선택: DuckDB 데이터 파일 경로 지정 시 사용)

## 4. 데이터 소스 선택
대시보드 사이드바의 **데이터 소스** 토글에서 `Parquet` 또는 `DuckDB`를 선택합니다. DuckDB 경로를 입력하지 않으면 Parquet/CSV 파일을 사용합니다.

## 5. 로그 확인
사이드바 하단의 **📄 메시지 로그** 익스팬더에서 경고/오류 로그를 확인할 수 있습니다.

## 6. 추가 참고 문서
- `docs/visualization_setup.md`: 데이터 소스 토글, 빠른 지역 선택, 리스크/정책 모니터 사용법 요약.
- `docs/todo_cycles.md`: Cycle 1~4 진행 현황 및 남은 TODO 확인.

## 7. 데이터 샘플 주의
Git 설정상 `data/` 디렉터리는 기본적으로 추적하지 않습니다. `data/reference/lifestyle_scores.csv` 같은 샘플이 필요하면 새 환경에서 직접 생성하거나, `git add -f`로 수동 관리하세요.
