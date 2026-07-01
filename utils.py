"""
Utility functions for the real-estate price estimation project.

Extracted from analiza_nieruchomosci.py and app_streamlit.py
to enable unit testing and reuse.

Contains:
  - parse_udzial: ownership-share fraction parser
  - extract_miasto: city-name extraction from address field
  - safe_encode: robust LabelEncoder lookup with fallback
  - clip_udzial: clip ownership-share series
  - filter_price_quantiles / filter_by_percentile: quantile-based row filtering
  - compute_city_stats: per-city transaction statistics
  - build_prediction_row: construct single-row DataFrame for prediction
  - print_section_header: formatted console section divider
  - save_plot: matplotlib save-and-close helper
  - train_and_evaluate: fit model, predict, compute MAE/RMSE/R2
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Parsing helpers ─────────────────────────────────────────


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


# ── Encoding helper ─────────────────────────────────────────


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


# ── Data helpers ────────────────────────────────────────────


def clip_udzial(series, low=0.001, high=1.0):
    """Clip an ownership-share series to a sensible range."""
    return series.clip(low, high)


def filter_price_quantiles(df, col, q_low=0.01, q_high=0.95):
    """Keep rows where *col* falls between the given quantiles."""
    lo = df[col].quantile(q_low)
    hi = df[col].quantile(q_high)
    return df[(df[col] >= lo) & (df[col] <= hi)]


# Alias used by analiza_nieruchomosci.py
filter_by_percentile = filter_price_quantiles


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


# ── Console / IO helpers ────────────────────────────────────


def print_section_header(number, title):
    """Print a numbered section divider to stdout."""
    print("\n" + "=" * 60)
    print(f"{number}. {title}")
    print("=" * 60)


def save_plot(output_dir, filename, dpi=150):
    """``tight_layout`` + save the current matplotlib figure, then close it."""
    plt.tight_layout()
    path = f"{output_dir}/{filename}"
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"Zapisano: {path}")


# ── Modelling helpers ───────────────────────────────────────


def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    """Fit *model*, predict on the test set, and return a metrics dict.

    Returns ``{"MAE": …, "RMSE": …, "R2": …, "model": …, "preds": …}``.
    """
    print(f"\nTrenuję: {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    print(f"  MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "model": model,
        "preds": y_pred,
    }
