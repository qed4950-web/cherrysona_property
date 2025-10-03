# 2025-03-18 CSV → ETL → Dashboard 정합성 개선 기록

## 1. 문제 요약

| 구분 | 발견된 현상 | 영향 | 대응 코드 |
|------|-------------|------|-----------|
| Rent Gap 왜곡 | 매매 데이터에 `보증금/월세`가 0으로 채워져 Rent Gap 계산이 모두 0으로 표시 | 추천·페르소나 탭이 실제 사례와 다르게 노출 | `app/dashboard_unified.py:989` |
| 주소 필드 오염 | `번지` 열에 “01월 11일” 등 날짜·신분 관련 문자열이 다수 포함 | 위치 기반 분석, 좌표 조인 시 실패 위험 | `src/etl_realestate.py:217` |
| 층/건축년도 이상치 | `층` 음수·200층 초과, `건축년도` 미래 값 존재 | 층/신축 프리미엄, 안정성 지표 왜곡 | `src/etl_realestate.py:688` |
| 보증금·월세 결측 | 전월세 CSV 외 자산에 결측 다수 → 자동 0 채움 | Yield, Rent Gap 계산 오차 | `src/etl_realestate.py:703` |
| 거래금액 누락 | 전월세 데이터에 실제 거래금액 없음 | 추정 매입가, 가격 단위 지표 계산 불가 | `src/etl_realestate.py:472` |
| 핵심 값 미보유 행 | 면적/추정매입가가 비어 있어도 통과 | KPI, 모멘텀/변동성 집계 시 NaN 누적 | `src/etl_realestate.py:840` |
| 중복 거래 | 동일 계약이 여러 CSV에 존재 | 거래량 집계 과대 | `src/etl_realestate.py:851` |
| Parquet index 유입 | 월간 산출물에 `index` 열이 남음 | Altair 등에서 예기치 않은 축 생성 | `src/etl_realestate.py:1003` |
| CSV dtype 경고 | `pd.read_csv` 기본 옵션으로 혼합 타입 경고 | Dashboard 실행 시 경고 스팸 | `src/etl_realestate.py:1012`, `app/dashboard_unified.py:215` |
| 필수 산출물 누락시 사용자 인지 어려움 | 데이터 미생성 상태로 대시보드 띄우면 빈 화면 | 디버깅 난항 | `app/dashboard_unified.py:49`, `app/dashboard_unified.py:1072` | 

## 2. 세부 개선 내역

### 2.1 전처리 파이프라인 강화 (src/etl_realestate.py)

1. **메타데이터 로깅 프레임워크** (`_init_metadata`, `_log_imputation`, `_log_drop`, `_log_issue`)
   - 결측 보정과 행 제거가 어디서 얼마나 일어났는지 기록해 `quality_report.json`에 직관적으로 노출합니다 (`src/etl_realestate.py:156`, `src/etl_realestate.py:1034`).

2. **주소 토큰 정리** (`clean_address_tokens`)
   - `번지` 문자열에 “월/일/주민등록/도로명” 등이 포함되면 `NaN`으로 전환 후 `extra_issues`에 기록합니다. 향후 좌표 매핑 실패를 예방합니다 (`src/etl_realestate.py:217`).

3. **층·건축년도 이상치 제거**
   - `층`값이 -5 미만·200 초과, `건축년도`가 1960 이전 또는 현재 연도 초과 시 결측으로 전환합니다. 페르소나/프리미엄 계산 안정성을 확보합니다 (`src/etl_realestate.py:688`).

4. **보증금·월세 중앙값 보정**
   - 시군구+자산유형 조합으로 중앙값을 추정하고, 여전히 결측이면 0으로 채웁니다(`fill_with_group_median`). 전월세 데이터는 의미 있는 값, 매매 데이터는 안전한 0 처리로 구분됩니다 (`src/etl_realestate.py:703`).

5. **면적·거래금액 보강**
   - 면적은 연면적/대지면적을 순차적으로 활용하고, 거래금액은 `보증금 + 월세×환산율`로 대체합니다. 이 과정 역시 `imputations`로 기록됩니다 (`src/etl_realestate.py:472`, `src/etl_realestate.py:488`).

6. **핵심 지표 누락 행 제거**
   - `전용면적_㎡` 또는 `추정매입가_만원`이 없는 행은 분석이 불가능하므로 제거하고, 제거 수량을 `drops`에 남깁니다 (`src/etl_realestate.py:840`).

7. **중복 거래 제거 추적**
   - 중복 제거 시 제거된 건수를 기록해 데이터 신뢰도를 추적합니다 (`src/etl_realestate.py:851`).

8. **산출물 형식 정리**
   - `persist_parquet`에서 인덱스를 완전히 제거하고 (`src/etl_realestate.py:1003`), `transactions.csv`는 날짜를 ISO 문자열로 직렬화해 Pandas dtype 경고를 제거합니다 (`src/etl_realestate.py:1012`).

### 2.2 대시보드 측 보완 (app/dashboard_unified.py)

1. **안전한 나눗셈 유틸**
   - `_safe_divide`를 대시보드 내부에 정의해 분모 0 발생 시 NaN으로 처리합니다 (`app/dashboard_unified.py:27`).

2. **Rent Gap 계산 보호**
   - `src_type`에 `lease`가 없거나 보증금/월세 합계가 0이면 Rent Gap 계산을 건너뛰어 매매 데이터가 추천 지표를 왜곡하지 않도록 했습니다 (`app/dashboard_unified.py:989`).

3. **데이터 누락 경고**
   - 필수 Parquet/CSV/레퍼런스 파일이 없으면 Streamlit 경고와 메시지 로그를 한 번만 출력해 사용자가 즉시 알아차리도록 했습니다 (`app/dashboard_unified.py:49`, `app/dashboard_unified.py:1072`, `app/dashboard_unified.py:1173`).

4. **CSV 로딩 옵션 조정**
   - `pd.read_csv(..., low_memory=False)`로 읽어 dtype 경고를 제거하고, 추후 dtype 통일이 용이하도록 했습니다 (`app/dashboard_unified.py:215`).

### 2.3 원시 데이터 진단 자동화
- `scripts/inspect_raw_data.py`를 추가해 원시 CSV의 필수 열 누락, 숫자 파싱 실패, 주소 이상치 등을 JSON 리포트로 기록합니다. 실행 결과는 `diagnostics/raw_data_report.json`에서 확인할 수 있습니다.

## 3. 결과 요약

- ETL 재실행 결과 9개 CSV, 5,032,807건을 처리하여 Parquet·CSV·DuckDB 산출물을 생성했고, `quality_report.json`에는 보정/제거 내역과 주소 패턴 이상치까지 포함됩니다.
- 대시보드는 필수 데이터가 존재할 때 경고 없이 구동되며, Rent Gap, 페르소나 스코어 등 주요 지표가 실제 값에 기반해 산출됩니다.
- 원시 CSV 진단 보고서에는 `번지` 필드의 문자열 패턴, 필수 열 누락 여부 등이 명시되어 클린징 정책 수립에 활용할 수 있습니다.

## 4. 향후 권장 조치

1. `diagnostics/raw_data_report.json`의 `extra_issues`(주소 토큰)와 `missing_columns`를 검토해 원 데이터 공급처와의 정합성을 확보합니다.
2. `quality_report.json`에서 `imputations`/`drops` 비율이 높은 열을 중심으로 전처리 파라미터(중앙값 기준, 윈저라이즈 범위 등)를 재검토합니다.
3. 레퍼런스 데이터(`policy_risk.csv`, `lifestyle_scores.csv`, `sigungu_centroids.*`, GeoJSON)를 운영 환경과 동기화하고, 주기적인 업데이트(금리, 정책 변경 등)를 위한 자동화 절차를 마련합니다.
4. 대시보드를 실행해 시각적으로 지표가 기대대로 출력되는지 확인하고, 필요 시 추가 UI/UX 개선(예: Rent Gap 기간 슬라이더, 지도 강조 옵션 등)을 반영합니다.

본 문서는 CSV → ETL → Streamlit Dashboard 전 과정을 검토하며 발견한 이슈와 대응 내역을 정리한 것으로, 추후 파이프라인 변경 시 지속적으로 업데이트해야 합니다.
