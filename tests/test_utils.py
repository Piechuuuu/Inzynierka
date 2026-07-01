"""Unit tests for utils.py utility functions."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from utils import (
    build_prediction_row,
    clip_udzial,
    compute_city_stats,
    extract_miasto,
    filter_price_quantiles,
    parse_udzial,
    safe_encode,
)


# ──────────────────────────────────────────────
#  extract_miasto
# ──────────────────────────────────────────────
class TestExtractMiasto:
    def test_standard_address(self):
        assert extract_miasto("MSC:Bydgoszcz;ULC:Gdanska") == "BYDGOSZCZ"

    def test_multiple_segments(self):
        result = extract_miasto("MSC:Torun;ULC:Main|MSC:Poznan;ULC:Other")
        assert result == "TORUN"

    def test_lowercase_city(self):
        assert extract_miasto("MSC:warszawa;ULC:foo") == "WARSZAWA"

    def test_city_with_spaces(self):
        assert extract_miasto("MSC: Nowe Miasto ;ULC:X") == "NOWE MIASTO"

    def test_nan_returns_nieznany(self):
        assert extract_miasto(np.nan) == "NIEZNANY"
        assert extract_miasto(None) == "NIEZNANY"
        assert extract_miasto(float("nan")) == "NIEZNANY"

    def test_no_msc_prefix(self):
        assert extract_miasto("ULC:Gdanska;NR:5") == "NIEZNANY"

    def test_empty_string(self):
        assert extract_miasto("") == "NIEZNANY"

    def test_only_msc_prefix(self):
        assert extract_miasto("MSC:Krakow") == "KRAKOW"

    def test_numeric_input(self):
        assert extract_miasto(12345) == "NIEZNANY"

    def test_pipe_separated_takes_first(self):
        assert extract_miasto("MSC:Gdansk;X|MSC:Sopot;Y") == "GDANSK"

    def test_msc_with_semicolon_boundary(self):
        assert extract_miasto("MSC:Wroclaw;") == "WROCLAW"


# ──────────────────────────────────────────────
#  parse_udzial
# ──────────────────────────────────────────────
class TestParseUdzial:
    def test_full_share(self):
        assert parse_udzial("1/1") == 1.0

    def test_half_share(self):
        assert parse_udzial("1/2") == 0.5

    def test_third_share(self):
        assert parse_udzial("1/3") == pytest.approx(1 / 3)

    def test_quarter_share(self):
        assert parse_udzial("1/4") == 0.25

    def test_two_thirds(self):
        assert parse_udzial("2/3") == pytest.approx(2 / 3)

    def test_three_quarters(self):
        assert parse_udzial("3/4") == 0.75

    def test_nan_returns_one(self):
        assert parse_udzial(np.nan) == 1.0
        assert parse_udzial(None) == 1.0

    def test_plain_float_string(self):
        assert parse_udzial("0.75") == 0.75

    def test_plain_int_string(self):
        assert parse_udzial("1") == 1.0

    def test_whitespace_around_fraction(self):
        assert parse_udzial("  1/2  ") == 0.5

    def test_invalid_fraction(self):
        assert parse_udzial("abc/def") == 1.0

    def test_division_by_zero(self):
        assert parse_udzial("1/0") == 1.0

    def test_empty_string(self):
        assert parse_udzial("") == 1.0

    def test_non_numeric_string(self):
        assert parse_udzial("unknown") == 1.0

    def test_zero_numerator(self):
        assert parse_udzial("0/1") == 0.0

    def test_large_fraction(self):
        assert parse_udzial("999/1000") == pytest.approx(0.999)


# ──────────────────────────────────────────────
#  safe_encode
# ──────────────────────────────────────────────
class TestSafeEncode:
    @pytest.fixture()
    def encoders(self):
        enc = {}
        le = LabelEncoder()
        le.fit(["mieszkalny", "biurowy", "przemyslowy", "nieznany"])
        enc["bud_rodzaj"] = le

        le2 = LabelEncoder()
        le2.fit(["wtorny", "pierwotny", "NIEZNANY"])
        enc["rynek"] = le2

        le3 = LabelEncoder()
        le3.fit(["A", "B", "C"])
        enc["no_fallback"] = le3

        return enc

    def test_known_value(self, encoders):
        result = safe_encode("bud_rodzaj", "mieszkalny", encoders)
        expected = int(encoders["bud_rodzaj"].transform(["mieszkalny"])[0])
        assert result == expected

    def test_another_known_value(self, encoders):
        result = safe_encode("bud_rodzaj", "biurowy", encoders)
        expected = int(encoders["bud_rodzaj"].transform(["biurowy"])[0])
        assert result == expected

    def test_unknown_value_falls_back_to_nieznany(self, encoders):
        result = safe_encode("bud_rodzaj", "UNKNOWN_TYPE", encoders)
        expected = int(encoders["bud_rodzaj"].transform(["nieznany"])[0])
        assert result == expected

    def test_unknown_value_falls_back_to_NIEZNANY(self, encoders):
        result = safe_encode("rynek", "NONEXISTENT", encoders)
        expected = int(encoders["rynek"].transform(["NIEZNANY"])[0])
        assert result == expected

    def test_missing_column_returns_zero(self, encoders):
        assert safe_encode("nonexistent_column", "foo", encoders) == 0

    def test_no_fallback_available_returns_zero(self, encoders):
        assert safe_encode("no_fallback", "UNKNOWN", encoders) == 0

    def test_empty_encoders(self):
        assert safe_encode("col", "val", {}) == 0

    def test_all_classes_accessible(self, encoders):
        for cls in ["mieszkalny", "biurowy", "przemyslowy", "nieznany"]:
            result = safe_encode("bud_rodzaj", cls, encoders)
            assert isinstance(result, int)


# ──────────────────────────────────────────────
#  clip_udzial
# ──────────────────────────────────────────────
class TestClipUdzial:
    def test_values_within_range(self):
        s = pd.Series([0.1, 0.5, 0.9])
        result = clip_udzial(s)
        pd.testing.assert_series_equal(result, s)

    def test_clips_low_values(self):
        s = pd.Series([0.0, -1.0, 0.0001])
        result = clip_udzial(s)
        assert (result >= 0.001).all()

    def test_clips_high_values(self):
        s = pd.Series([1.5, 2.0, 100.0])
        result = clip_udzial(s)
        assert (result <= 1.0).all()

    def test_boundary_values(self):
        s = pd.Series([0.001, 1.0])
        result = clip_udzial(s)
        assert result.iloc[0] == 0.001
        assert result.iloc[1] == 1.0

    def test_custom_range(self):
        s = pd.Series([0.0, 5.0, 10.0])
        result = clip_udzial(s, low=1.0, high=8.0)
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 5.0
        assert result.iloc[2] == 8.0


# ──────────────────────────────────────────────
#  filter_price_quantiles
# ──────────────────────────────────────────────
class TestFilterPriceQuantiles:
    def test_filters_outliers(self):
        prices = list(range(1, 101))
        df = pd.DataFrame({"price": prices})
        result = filter_price_quantiles(df, "price", q_low=0.1, q_high=0.9)
        assert result["price"].min() >= 10
        assert result["price"].max() <= 90

    def test_default_quantiles(self):
        np.random.seed(42)
        prices = np.random.lognormal(mean=12, sigma=1, size=1000)
        df = pd.DataFrame({"price": prices})
        result = filter_price_quantiles(df, "price")
        assert len(result) < len(df)
        assert result["price"].min() >= df["price"].quantile(0.01)
        assert result["price"].max() <= df["price"].quantile(0.95)

    def test_preserves_columns(self):
        df = pd.DataFrame({"price": [1, 2, 3, 4, 5], "city": list("abcde")})
        result = filter_price_quantiles(df, "price", q_low=0.0, q_high=1.0)
        assert list(result.columns) == ["price", "city"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"price": pd.Series([], dtype=float)})
        result = filter_price_quantiles(df, "price")
        assert len(result) == 0


# ──────────────────────────────────────────────
#  compute_city_stats
# ──────────────────────────────────────────────
class TestComputeCityStats:
    def test_basic_aggregation(self):
        df = pd.DataFrame({
            "miasto": ["A", "A", "B", "B", "B"],
            "cena": [100, 200, 300, 400, 500],
        })
        result = compute_city_stats(df, "miasto", "cena")
        assert len(result) == 2
        assert set(result.columns) == {"miasto", "n", "mediana", "srednia"}

    def test_counts_correct(self):
        df = pd.DataFrame({
            "miasto": ["X", "X", "X", "Y"],
            "cena": [10, 20, 30, 100],
        })
        result = compute_city_stats(df, "miasto", "cena")
        x_row = result[result["miasto"] == "X"]
        assert x_row["n"].iloc[0] == 3

    def test_median_correct(self):
        df = pd.DataFrame({
            "miasto": ["A", "A", "A"],
            "cena": [100, 200, 300],
        })
        result = compute_city_stats(df, "miasto", "cena")
        assert result["mediana"].iloc[0] == 200.0

    def test_mean_correct(self):
        df = pd.DataFrame({
            "miasto": ["A", "A", "A"],
            "cena": [100, 200, 300],
        })
        result = compute_city_stats(df, "miasto", "cena")
        assert result["srednia"].iloc[0] == 200.0

    def test_sorted_descending_by_count(self):
        df = pd.DataFrame({
            "miasto": ["A"] * 5 + ["B"] * 10 + ["C"] * 3,
            "cena": list(range(18)),
        })
        result = compute_city_stats(df, "miasto", "cena")
        counts = result["n"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_single_city(self):
        df = pd.DataFrame({"miasto": ["Only"], "cena": [42]})
        result = compute_city_stats(df, "miasto", "cena")
        assert len(result) == 1
        assert result["n"].iloc[0] == 1


# ──────────────────────────────────────────────
#  build_prediction_row
# ──────────────────────────────────────────────
class TestBuildPredictionRow:
    @pytest.fixture()
    def features_order(self):
        return [
            "bud_rodzaj", "tran_rodzaj_rynku", "tran_rodzaj_trans",
            "tran_sprzedajacy", "tran_kupujacy", "nier_rodzaj", "nier_prawo",
            "miasto", "bud_pow_uzyt", "nier_pow_gruntu", "udzial_float",
            "teryt", "rok", "miesiac", "ma_pow_uzyt",
            "miasto_med", "teryt_med", "nier_cena_brutto", "ma_nier_cena",
        ]

    @pytest.fixture()
    def encoded_cats(self):
        return {
            "bud_rodzaj": 1,
            "tran_rodzaj_rynku": 2,
            "tran_rodzaj_trans": 0,
            "tran_sprzedajacy": 3,
            "tran_kupujacy": 0,
            "nier_rodzaj": 1,
            "nier_prawo": 2,
            "miasto": 5,
        }

    def test_output_shape(self, encoded_cats, features_order):
        row = build_prediction_row(
            encoded_cats=encoded_cats,
            bud_pow_uzyt=65.0,
            nier_pow_gruntu=500.0,
            udzial_float=1.0,
            teryt=461,
            rok=2024,
            miesiac=6,
            miasto_med_val=300000.0,
            teryt_med_val=250000.0,
            nier_cena_val=350000.0,
            ma_nier_cena_val=1,
            features_order=features_order,
        )
        assert row.shape == (1, 19)

    def test_column_order(self, encoded_cats, features_order):
        row = build_prediction_row(
            encoded_cats=encoded_cats,
            bud_pow_uzyt=65.0,
            nier_pow_gruntu=500.0,
            udzial_float=1.0,
            teryt=461,
            rok=2024,
            miesiac=6,
            miasto_med_val=300000.0,
            teryt_med_val=250000.0,
            nier_cena_val=350000.0,
            ma_nier_cena_val=1,
            features_order=features_order,
        )
        assert list(row.columns) == features_order

    def test_ma_pow_uzyt_flag_positive(self, encoded_cats, features_order):
        row = build_prediction_row(
            encoded_cats=encoded_cats,
            bud_pow_uzyt=65.0,
            nier_pow_gruntu=0.0,
            udzial_float=1.0,
            teryt=461,
            rok=2024,
            miesiac=6,
            miasto_med_val=300000.0,
            teryt_med_val=250000.0,
            nier_cena_val=0.0,
            ma_nier_cena_val=0,
            features_order=features_order,
        )
        assert row["ma_pow_uzyt"].iloc[0] == 1

    def test_ma_pow_uzyt_flag_zero(self, encoded_cats, features_order):
        row = build_prediction_row(
            encoded_cats=encoded_cats,
            bud_pow_uzyt=0.0,
            nier_pow_gruntu=0.0,
            udzial_float=1.0,
            teryt=461,
            rok=2024,
            miesiac=6,
            miasto_med_val=300000.0,
            teryt_med_val=250000.0,
            nier_cena_val=0.0,
            ma_nier_cena_val=0,
            features_order=features_order,
        )
        assert row["ma_pow_uzyt"].iloc[0] == 0

    def test_numeric_values_preserved(self, encoded_cats, features_order):
        row = build_prediction_row(
            encoded_cats=encoded_cats,
            bud_pow_uzyt=120.5,
            nier_pow_gruntu=800.0,
            udzial_float=0.5,
            teryt=463,
            rok=2023,
            miesiac=11,
            miasto_med_val=400000.0,
            teryt_med_val=350000.0,
            nier_cena_val=500000.0,
            ma_nier_cena_val=1,
            features_order=features_order,
        )
        assert row["bud_pow_uzyt"].iloc[0] == 120.5
        assert row["nier_pow_gruntu"].iloc[0] == 800.0
        assert row["udzial_float"].iloc[0] == 0.5
        assert row["teryt"].iloc[0] == 463
        assert row["rok"].iloc[0] == 2023
        assert row["miesiac"].iloc[0] == 11

    def test_encoded_cats_preserved(self, encoded_cats, features_order):
        row = build_prediction_row(
            encoded_cats=encoded_cats,
            bud_pow_uzyt=65.0,
            nier_pow_gruntu=0.0,
            udzial_float=1.0,
            teryt=461,
            rok=2024,
            miesiac=6,
            miasto_med_val=300000.0,
            teryt_med_val=250000.0,
            nier_cena_val=0.0,
            ma_nier_cena_val=0,
            features_order=features_order,
        )
        for col, val in encoded_cats.items():
            assert row[col].iloc[0] == val
