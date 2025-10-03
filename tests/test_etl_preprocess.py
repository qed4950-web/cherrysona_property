import pandas as pd
import numpy as np
import pytest

from src.etl_realestate import preprocess_dataframe, safe_divide


def test_safe_divide_handles_zero_and_negative():
    numer = pd.Series([100, 50, 25])
    denom = pd.Series([10, 0, -5])
    result = safe_divide(numer, denom)
    assert result.iloc[0] == 10
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


def test_preprocess_dataframe_parses_cancel_date_and_yield_guard():
    raw = pd.DataFrame(
        {
            "시도": ["서울특별시"],
            "시군구": ["강남구"],
            "읍면동": ["삼성동"],
            "거래금액(만원)": ["10,000"],
            "전용면적(㎡)": [50],
            "계약년월": [202201],
            "계약일": [15],
            "해제사유발생일": ["20220421"],
            "월세_만원": [0],
            "보증금_만원": [0],
            "건축년도": [2018],
            "층": [10],
        }
    )

    processed = preprocess_dataframe(raw, src_type="apt_trade")

    assert "해제사유발생일" in processed.columns
    assert pd.api.types.is_datetime64_any_dtype(processed["해제사유발생일"])  # parsed as datetime
    assert processed["취소여부"].iloc[0] == 1

    assert processed["가격_per_㎡"].iloc[0] == pytest.approx(200.0)
    assert not np.isinf(processed["Yield_%"].iloc[0])
    assert processed["floor_insight_enabled"].iloc[0] == 1


def test_preprocess_fallback_for_rent_only_and_area_alternative():
    raw = pd.DataFrame(
        {
            "시도": ["서울특별시"],
            "시군구": ["마포구"],
            "읍면동": ["합정동"],
            "보증금(만원)": ["5,000"],
            "월세(만원)": [50],
            "연면적(㎡)": [80],
            "계약년월": [202203],
            "계약일": [3],
            "건축년도": [2010],
            "층": [12],
        }
    )

    processed = preprocess_dataframe(raw, src_type="apt_lease")

    assert processed["거래금액_만원"].iloc[0] == pytest.approx(5_000 + 50 * 100)
    assert processed["전용면적_㎡"].iloc[0] == pytest.approx(80.0)
    assert processed["가격_per_㎡"].iloc[0] == pytest.approx((5_000 + 50 * 100) / 80)
    assert processed["floor_insight_enabled"].iloc[0] == 1


def test_floor_insight_disabled_for_comm_trade():
    raw = pd.DataFrame(
        {
            "시도": ["서울특별시"],
            "시군구": ["중구"],
            "읍면동": ["명동"],
            "거래금액(만원)": ["25,000"],
            "연면적": [120.0],
            "계약년월": [202201],
            "계약일": [10],
            "층": [5],
        }
    )

    processed = preprocess_dataframe(raw, src_type="comm_trade")

    assert processed["floor_insight_enabled"].iloc[0] == 0
    assert processed["floor_bucket"].isna().all()
