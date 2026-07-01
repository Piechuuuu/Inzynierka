"""
Utility functions for the real-estate price estimation project.

Extracted from analiza_nieruchomosci.py and app_streamlit.py
to enable unit testing and reuse.
"""

import re

import numpy as np
import pandas as pd


def extract_miasto(adres):
    """Extracts the first city name from a bud_adres field.

    Expected format: "MSC:NazwaMiasta;..." with segments separated by "|".
    Returns the city name in uppercase, or "NIEZNANY" if not found.
    """
    if pd.isna(adres):
        return "NIEZNANY"
    segment = str(adres).split("|")[0]
    match = re.search(r"MSC:([^;|]+)", segment)
    if match:
        return match.group(1).strip().upper()
    return "NIEZNANY"


def parse_udzial(val):
    """Converts an ownership-share string to a float.

    Handles fractions ("1/2" -> 0.5), plain numbers, NaN, and invalid values.
    Returns 1.0 as default for missing/unparseable input.
    """
    if pd.isna(val):
        return 1.0
    val = str(val).strip()
    if "/" in val:
        try:
            parts = val.split("/")
            return float(parts[0]) / float(parts[1])
        except Exception:
            return 1.0
    try:
        return float(val)
    except Exception:
        return 1.0


def safe_encode(col, val, encoders):
    """Encodes a categorical value to its integer label.

    Falls back through "nieznany", "NIEZNANY", "INNE" if *val* is unknown,
    then returns 0 as an absolute fallback.

    Parameters
    ----------
    col : str
        Column name (key into *encoders*).
    val : str
        Raw categorical value.
    encoders : dict
        Mapping of column names to fitted ``LabelEncoder`` instances.
    """
    if col not in encoders:
        return 0
    le = encoders[col]
    if val in le.classes_:
        return int(le.transform([val])[0])
    for fallback in ["nieznany", "NIEZNANY", "INNE"]:
        if fallback in le.classes_:
            return int(le.transform([fallback])[0])
    return 0


def clip_udzial(series, low=0.001, high=1.0):
    """Clip an ownership-share series to a sensible range."""
    return series.clip(low, high)


def filter_price_quantiles(df, col, q_low=0.01, q_high=0.95):
    """Keep rows where *col* falls between the given quantiles."""
    lo = df[col].quantile(q_low)
    hi = df[col].quantile(q_high)
    return df[(df[col] >= lo) & (df[col] <= hi)]


def compute_city_stats(df, city_col, price_col):
    """Aggregate per-city transaction statistics (count, median, mean)."""
    return (
        df.groupby(city_col)[price_col]
        .agg(n="count", mediana="median", srednia="mean")
        .reset_index()
        .sort_values("n", ascending=False)
    )


def build_prediction_row(
    encoded_cats,
    bud_pow_uzyt,
    nier_pow_gruntu,
    udzial_float,
    teryt,
    rok,
    miesiac,
    miasto_med_val,
    teryt_med_val,
    nier_cena_val,
    ma_nier_cena_val,
    features_order,
):
    """Construct a single-row DataFrame ready for model.predict().

    Parameters
    ----------
    encoded_cats : dict
        Already-encoded categorical column values.
    features_order : list[str]
        Column order the model expects (from metryki["_features"]).
    """
    row_dict = {
        **encoded_cats,
        "bud_pow_uzyt": bud_pow_uzyt,
        "nier_pow_gruntu": nier_pow_gruntu,
        "udzial_float": udzial_float,
        "teryt": teryt,
        "rok": rok,
        "miesiac": miesiac,
        "ma_pow_uzyt": int(bud_pow_uzyt > 0),
        "miasto_med": miasto_med_val,
        "teryt_med": teryt_med_val,
        "nier_cena_brutto": nier_cena_val,
        "ma_nier_cena": ma_nier_cena_val,
    }
    return pd.DataFrame([row_dict])[features_order]
