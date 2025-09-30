# Repository Guidelines

## data/ 디렉터리
- `data/raw/`에 부동산 실거래 CSV를 유형별 하위폴더(`apt_trade/`, `offi_lease/` 등)로 정리하면 ETL이 글롭 패턴으로 로드할 수 있습니다.
- `data/reference/`에는 `ref_법정동코드.csv`처럼 정적 참조 테이블과 매핑 파일을 보관하고, `.gitignore`에 등록해 민감 데이터가 추적되지 않도록 합니다.
- 전처리된 산출물은 `data/processed/`에 저장하고, 파일명에 스냅샷 일자를 포함해 증분 적재 시 분기점을 명확히 합니다.

## src/ 모듈
- ETL 파이프라인은 `src/etl_realestate.py`처럼 단일 진입 스크립트와 `src/pipelines/` 하위 단계별 모듈로 분리합니다.
- 공통 유틸은 `src/utils/`에 묶고, 타입 힌트와 독스트링을 사용해 데이터 스키마(`pandas.DataFrame`)를 명시합니다.
- 환경 변수는 `src/config.py`에서 `python-dotenv`로 로드하며, 기본값과 예외 메시지를 함께 정의합니다.

## app/ 대시보드
- Streamlit 스켈레톤은 `app/dashboard.py`에 유지하고, 페이지별 구성 요소는 `app/pages/`로 분할합니다.
- `streamlit run app/dashboard.py` 실행 전 `.streamlit/secrets.toml`을 준비해 DB 접속 정보를 주입하세요.
- 시각화에 사용되는 캐시 데이터는 `app/cache/`에 저장하고, 용량이 큰 파일은 주기적으로 정리합니다.

## tests/와 fixtures/
- 단위 테스트는 `tests/`에서 모듈 경로를 반영한 이름(`test_etl_realestate.py`)으로 생성합니다.
- `tests/fixtures/`에는 축약된 샘플 CSV와 기대 결과 JSON을 보관하고, `pytest` 픽스처로 로드합니다.
- 푸시 전 `pytest --maxfail=1 --disable-warnings --cov=src`를 실행해 커버리지 80% 이상을 유지합니다.

## 관리 문서 및 자동화
- 문서화는 `docs/`에 저장하고, 실험 노트북은 `notebooks/`로 이동해 코드와 분리합니다.
- 커밋 메시지는 현재 시제, 60자 이내, `ETL:`, `Dash:` 등 스코프 접두어를 사용합니다.
- PR에는 변경 요약, 데이터 영향, 검증 결과, 필요 시 대시보드 스크린샷을 포함하고 관련 이슈를 링크합니다.

## 보안 및 설정 팁
- `.env.example`을 제공해 필수 환경 변수를 안내하고, 실제 `.env`는 커밋하지 않습니다.
- 실거래 데이터는 업로드 전에 PII 제거, 행 수 검증, 이상값 스팟 체크를 진행해 배포 안정성을 확보합니다.
