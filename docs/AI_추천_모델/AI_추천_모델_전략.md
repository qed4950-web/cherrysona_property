# AI_추천_모델_전략

## 문서 개요

- [문서 목차](../문서_목차.md)
- [데이터 파이프라인 통합가이드](../데이터_파이프라인/데이터_파이프라인_통합가이드.md)
- [대시보드 · 시각화 통합가이드](../대시보드_시각화/대시보드_시각화_통합가이드.md)
- [운영 프로세스 통합가이드](../운영_프로세스/운영_프로세스_통합가이드.md)

---

## 1. ML 확장 전략

| 작업 사이클 | 목적 | 적용 모델 | 역할/설명 |
|-------------|------|-----------|-----------|
| Cycle 1: Insights | 가격·모멘텀 기반 “좋은 구간” 자동 탐색 | `KMeans` 군집화 | 기존 4분면(가격 vs 모멘텀)을 ML로 확장. `avg_p_per_m2`, `mom_3m`, `stability` 등을 정규화 후 군집화하여 Momentum↑ · 가격↓ 구역을 추천 후보군으로 표시. |
| Cycle 2: 추천 스코어 | 저평가율 자동 산출 | `RandomForestRegressor` 회귀 | 핵심 특성(전용면적, 층, 건축년도, `avg_p_per_m2`, `mom_3m` 등)으로 적정 거래금액을 예측하고, `예측금액` 대비 실제 `거래금액` 편차로 ML 저평가율을 계산해 Top N 후보를 제시. |
| Cycle 3: 근거 설명 | 추천 결과 해석 | `DecisionTreeRegressor` + Feature Importance | Cycle 2 학습 데이터를 활용해 얕은 트리(예: depth 3)를 학습하고 `plot_tree`/feature importance로 “왜 이 지역을 추천했는가”를 시각적으로 설명. 페르소나/리스크 탭의 하단 분석에 배치. |

### 구현 포인트

- **전처리 공통화**: `compare_base`/`persona_source`에서 필요한 수치 열을 `StandardScaler`로 정규화하고, 결측치는 직전에 계산한 방식(`fillna`)을 재사용.
- **모델 캐싱**: `st.cache_resource`로 학습 모델을 감싸 재실행 시 반복 학습을 줄이고, 사이드바 조건(기간·지역·자산군) 변경 시에만 재학습하도록 키 구성.
- **시각화 연계**: KMeans 결과는 Altair 산점도와 색상/툴팁 연동, RandomForest 결과는 `st.dataframe`/`st.metric`, DecisionTree는 `st.pyplot`으로 트리 + `feature_importances_` 막대차트를 병행.
- **사용자 흐름**: KPI → Insights(KMeans) → 추천(RandomForest) → Persona/Risk(Tree) 순으로 자연스러운 “탐색 → 검증 → 설명” 사이클을 완성.
- **확장 아이디어**: 각 사이클의 출력값을 `st.session_state`로 공유해 이후 탭에서 후보 리스트, 설명 노트, 알림 트리거(저평가율 임계값) 등에 재사용 가능.

### 기존 기능과의 충돌 가능성 체크

- **데이터 집합 재사용**: `compare_base`는 다른 탭에서도 활용 중이므로, ML 학습 시 복사본(`df.copy()`)을 사용하고 원본을 변형하지 않아야 합니다. 현재 설계는 이를 전제로 하므로 충돌 없음.
- **의존성 추가**: scikit-learn, matplotlib 등의 패키지가 요구됩니다. `requirements.txt`에 버전을 명시하지 않으면 배포 시 ImportError가 발생할 수 있으니 반드시 추가 필요.
- **계산 비용**: 사이드바 조건 변경 시 재학습이 일어날 수 있어, 대용량 데이터에서는 연산 지연 가능. `st.cache_resource` 키 설계와 `n_jobs`(가능한 모델의 경우) 조정으로 완화.
- **모델 입력 컬럼 충돌**: 선택한 특성이 결측이 많거나 타입이 혼재되면 다른 지표 계산에도 영향이 갈 수 있음. ML 파이프라인 진입 전에 `dropna`/`to_numeric`을 수행하고, 실패 시 graceful fallback 메시지를 띄워야 다른 기능(same 탭의 표/차트)과 겹치지 않습니다.
- **시각화 중복**: Insights 탭의 기존 Altair 차트와 KMeans 산점도가 동일 영역을 쓸 경우 UI가 과밀해질 수 있음. 기존 차트를 유지하려면 새 차트를 expander나 별도 컬럼으로 배치하는 것이 안전합니다.
- **세션 상태 키**: 새 모델 결과를 `st.session_state`에 보관할 경우 기존 키(`REGION_SELECTION_KEY` 등)와 중복되지 않도록 고유 접두어를 사용해야 예상치 못한 조건 변경이 발생하지 않습니다.
- **ETL 파이프라인 연계**: ML 학습에 활용할 열(`avg_p_per_m2`, `mom_3m`, `stability`, `전용면적`, `층`, `건축년도` 등)이 ETL 단계에서 일관된 스키마로 보장되는지 확인해야 합니다. ETL 수정 전에는 컬럼 추가/이름 변경이 다른 대시보드 지표와 ML 입력 모두에 영향을 줄 수 있으므로, `etl_realestate.py` 수정 시 해당 열 유지 여부를 먼저 점검하세요.

### 사전 점검 체크리스트 (코드 작업 전)

1. **ETL 산출물 스키마 확인**  
   - `data/processed/monthly_basics.parquet`: `avg_p_per_m2`, `txn_cnt`, `avg_yield_pct`, `cancel_rate` 존재 여부 확인.  
   - `data/processed/monthly_momentum.parquet`: `mom_3m` 컬럼 존재 여부 및 최근 월 데이터 확인.  
   - `data/processed/monthly_volatility.parquet`: `cv_p_per_m2`, `stability` 산출 여부 확인.  
   - `data/processed/transactions.csv`: ML 입력에 사용할 `전용면적`, `층`, `건축년도`, `가격_per_㎡`, `거래금액`(또는 등가 열) 준비 여부 확인.

2. **ETL 코드 영향도 점검 (`src/etl_realestate.py`)**  
   - 상기 컬럼을 생성/유지하는 로직이 동작 중인지, 필터링이나 집계 과정에서 값이 삭제되는 구간이 없는지 확인.  
   - 향후 컬럼명 변경 시 `app/dashboard_unified.py`와 ML 파이프라인 코드에서 동일하게 반영될 계획 수립.

3. **대시보드 전처리 충돌 여부 확인**  
   - `filter_by_src`, `filter_recent_months`, `filter_by_regions` 호출 이후에도 ML 학습에 필요한 컬럼이 유지되는지 샘플 데이터를 통해 검증.  
   - `compare_base` 생성 이후 결측 보정(`fillna`)이 ML 학습에 악영향을 주지 않는지 점검. 필요 시 학습용 복사본(`persona_source`)에서 추가적인 `dropna` 로직을 준비.

4. **레퍼런스/보조 데이터 상태**  
   - `data/reference/lifestyle_scores.csv`, `policy_risk.csv`, `sigungu_centroids.(parquet|csv)`가 최신 데이터와 일치하는지 확인. KMeans 시각화나 추천 근거 설명에 보조 지표가 필요할 경우 결측 여부를 미리 파악.

5. **성능·리소스 계획**  
   - 기대 최대 데이터 크기에서 KMeans/RandomForest 학습 소요 시간 측정.  
   - 필요 시 배치 학습(사전 학습 후 결과 저장)과 실시간 학습(사용자 인터랙션 기반) 중 어떤 전략을 택할지 결정.

위 항목을 모두 통과한 후에 ML 관련 코드를 탭별로 삽입해야 ETL·대시보드·데이터 파일과의 충돌을 최소화할 수 있습니다.

### 2025-10-05 구현 정리

- **Insights 탭**에 `avg_p_per_m2`·`mom_3m`·`stability` 기반 KMeans 군집 시각화를 추가해 자동 추천 클러스터를 표시합니다.
- **추천 탭**에 RandomForest 회귀를 연동해 `avg_p_per_m2` 대비 ML 저평가율을 계산하고 Top N 후보와 특성 중요도를 보여줍니다.
- **Persona/Risk 탭** 하단에 DecisionTree 시각화를 배치해 추천 판단 근거(주요 분기 기준)를 설명합니다.
- 모든 ML 파이프라인은 `scikit-learn`/`matplotlib` 미설치 시 우회 안내 메시지를 출력하도록 가드 처리했습니다.
