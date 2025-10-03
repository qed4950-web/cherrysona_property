# Dashboard TODO Roadmap

## Cycle 1 · Visualization Upgrade
- 데이터 준비: 지역 좌표/경계 GeoJSON 정리하고 시군구 키 매핑 확인
- Heatmap/버블: Altair Layer 또는 Folium 선택, 저평가율·모멘텀·프리미엄 토글 구현
- 시계열 라인차트: 지역 멀티선 그래프, 최근 6/12/전체 구간 스위치 제공
- 전월세 환산 차익: Rent Gap 분포 Bar Chart + 환산율 슬라이더 연동
- 캡션/툴팁: 시각화마다 지표 정의와 해석 코멘트 추가

## Cycle 2 · Persona & Compare 확장
- ✅ 연금형 지표: 안정성, Yield, 금리·공실까지 반영한 적합도 계산
- ✅ 실거주 성장 지표: 신축 프리미엄 + `data/reference/lifestyle_scores.csv` 학군·생활편의 점수 통합
- ✅ 시세차익형 지표: 모멘텀·저평가·공실 Proxy 중심 스코어링 정비
- ✅ Compare 레이더 차트: 핵심 지표 표준화, Altair 폴라 시각화 적용
- ✅ 자동 코멘트: 비교 결과에서 상위/하위 지표 강조 문장 생성 로직 추가
- ✅ Top5 KPI 카드: 상단 metric 카드 + 상세 테이블 익스팬더 방식 도입

## Cycle 3 · 리스크 분석 심화
- ✅ 변동성(CV)·취소율: Trend 탭 라인차트 추가, 계약 취소율 시각화 완료
- ✅ DOM(체류기간) 데이터: 계약일자 대비 스냅샷 기반 중위 DOM 계산 및 대시보드 반영
- ✅ 공실률 Proxy: 최근 3개월/12개월 거래량 변화를 사용한 공실 위험 지표 추가
- ✅ 전월세 계약 안정성: 계약기간/계약구분/갱신요구권 기반 안정성 스코어 산출 및 대시보드 반영
- ✅ 정책/금리: `data/reference/policy_risk.csv` 구조 정리, 리스크 탭 금리 추이 라인 및 최신 규제/코멘트 테이블 노출 (외부 API 연동은 후속)

## Cycle 4 · 성능 & UX 개선
- ✅ DuckDB/Parquet 토글: 사이드바에서 데이터 소스 전환(Parquet ↔️ DuckDB) 지원, `scripts/run_dashboard.sh` 배포 스크립트 추가
- ✅ 데이터 캐시: `st.cache_data` TTL 조정으로 월간/거래/레퍼런스 데이터 재사용 최적화
- ✅ 필터 UX: `지역 선택 모드` 토글과 거래량 기반 빠른 선택 도입, 맞춤 지역 분석 UX 개선
- KPI 카드/탭 레이아웃: 반응형 컬럼 폭 조정, 모바일 뷰 테스트
- 배포 준비: requirements/venv 정리, Streamlit Cloud 또는 내부 서버 배포 가이드
