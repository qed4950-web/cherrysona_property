# Visualization Enhancements (Cycle 1)

## Time-Series Overview
- Location: `Trend & KPI > Overview/모멘텀/변동성`
- Usage: Altair 라인 차트가 선택된 시군구의 월별 `avg_p_per_m2`, `mom_3m`, `cv_p_per_m2`를 표시합니다.
- Tip: 차트 하단의 "원본 데이터 보기" 익스팬더에서 원천 테이블을 확인할 수 있습니다.

## Rent Gap Bar Chart
- Location: `💡 추천 & 리스크` 탭의 전월세 섹션
- Definition: `Rent Gap = (보증금×환산율/12) - 월세`
- Interpretation: 파란색(환산가 우위) → 전세 기준이 유리, 주황색(월세 우위) → 월세가 더 큼.

## Risk (DOM · Cancellation · Vacancy · Stability)
- Location: `📊 Trend & KPI > 변동성 / 취소율`
- DOM: `snapshot_date - 계약일자`로 계산한 중위 계약 체류기간(일)
- 취소율: `취소여부` 평균값(=계약 취소 비율)을 월 단위로 시각화
- Vacancy Proxy: 최근 3개월 거래량 평균 ÷ 최근 12개월 평균 – 1 (음수면 거래량 둔화 → 공실 위험↑)
- 계약 안정성: 계약기간·갱신 여부·갱신요구권을 가중해 0~1 스코어로 환산
- Policy Risk: `data/reference/policy_risk.csv`에 금리(`interest_rate`), 규제 여부(`regulation_flag`), 코멘트(`policy_comment`)를 월/지역 단위로 기록하면 Compare/Persona 탭에 표시됨
- Policy Monitor: `📊 Trend & KPI > 변동성 / 취소율` 탭에서 최신 금리/규제 요약 테이블과 금리 추이 라인 차트를 함께 확인 가능
- Note: 원본 탭 아래 익스팬더에서 `risk_filtered` 미리보기 확인 가능

## Regional Heatmap (Beta)
- Location: `🔎 Insights` 탭 하단
- Bubble mode: `data/reference/sigungu_centroids.(csv|parquet)`에서 시군구 중심 좌표를 사용.
- Choropleth mode: `data/raw/HangJeongDong_ver20250401.geojson`와 같이 행정동 경계 GeoJSON을 제공하면 pydeck PolygonLayer로 경계 채색 가능.
- Metric selector: 저평가율, 모멘텀, 신축·층 프리미엄, 월세 저평가율, 종합 점수 등 가용 지표만 노출.
- Notes: 버블/경계 모드는 라디오 버튼으로 전환하며, 경계 데이터가 없는 경우 자동으로 버블 모드만 노출.

## TODO Follow-up
- GeoJSON 경계 시각화 및 배경지도 추가 검토
- 지도용 색상 스케일/임계치 튜닝
- Rent Gap 그래프에 기간 선택 옵션 추가
  
## 데이터 소스 참고
- `data/reference/lifestyle_scores.csv`: 시군구별 학군(`school_score`)·생활편의(`amenity_score`) 점수와 메모를 입력하면 Compare/Persona 탭에서 실거주 성장 지표에 반영됨
- Streamlit 사이드바의 **데이터 소스** 토글에서 Parquet/CSV 또는 DuckDB 파일을 선택할 수 있습니다. DuckDB를 고르면 `CHERRYSONA_DUCKDB` 환경변수(또는 직접 입력한 경로)를 통해 `fact_transactions`, `vw_monthly_*` 테이블을 바로 조회합니다.
- `scripts/run_dashboard.sh` 실행 스크립트는 `STREAMLIT_PORT`(기본 8501)와 `STREAMLIT_DATA_DIR`(기본 `app/cache`) 환경변수를 인식해 배포 시 설정을 단순화합니다.
- 지역 필터는 `지역 선택 모드`(전체/맞춤)와 빠른 선택(최근 거래 상위 12개) 기능을 제공하므로, 넓은 범위/특정 지역 모두 쉽게 전환할 수 있습니다.
