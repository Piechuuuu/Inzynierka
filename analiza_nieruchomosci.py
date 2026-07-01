"""
Analiza danych transakcji nieruchomościami (transakcje_budynki.csv)
Praca inżynierska – Jakub Piechowiak

Zawartość:
  1. Wczytanie i eksploracja danych (EDA)
  2. Czyszczenie i przygotowanie danych
  3. Feature engineering (miasto, rok, miesiąc, udziały, mediany)
  4. Modele: Regresja liniowa, Random Forest, XGBoost
  5. Porównanie modeli (MAE, RMSE, R²)
  6. Zapis modelu i wszystkich artefaktów wymaganych przez app_streamlit.py

Wymagane biblioteki:
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from utils import (
    extract_miasto,
    filter_by_percentile,
    parse_udzial,
    print_section_header,
    save_plot,
    train_and_evaluate,
)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Uwaga: xgboost nie jest zainstalowany. Zainstaluj: pip install xgboost")
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_PATH   = "transakcje_budynki.csv"
OUTPUT_DIR  = "wyniki"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. WCZYTANIE DANYCH
# ─────────────────────────────────────────────
print_section_header(1, "WCZYTANIE DANYCH")

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Rozmiar zbioru: {df.shape[0]:,} wierszy, {df.shape[1]} kolumn")
print("\nKolumny:")
print(df.dtypes)

# ─────────────────────────────────────────────
# 2. EDA – EKSPLORACYJNA ANALIZA DANYCH
# ─────────────────────────────────────────────
print_section_header(2, "EDA")

missing     = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(1)
missing_df  = pd.DataFrame({"Braki": missing, "Procent": missing_pct})
print("\nBraki danych (top 15):")
print(missing_df[missing_df["Braki"] > 0].head(15))

print("\nRozkład rodzaju budynku:")
print(df["bud_rodzaj"].value_counts())

print("\nRozkład rodzaju rynku:")
print(df["tran_rodzaj_rynku"].value_counts())

print("\nStatystyki tran_cena_brutto:")
print(df["tran_cena_brutto"].describe())

# --- Wykresy EDA ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ceny_ogr = df["tran_cena_brutto"].dropna()
ceny_ogr = ceny_ogr[(ceny_ogr > 1000) & (ceny_ogr < ceny_ogr.quantile(0.99))]
axes[0].hist(ceny_ogr, bins=80, color="#2196F3", edgecolor="white")
axes[0].set_title("Rozkład cen transakcji (bez outlierów)")
axes[0].set_xlabel("Cena brutto [PLN]")
axes[0].set_ylabel("Liczba transakcji")

top_rodzaje = df["bud_rodzaj"].value_counts().head(5).index
df_top = df[df["bud_rodzaj"].isin(top_rodzaje)].dropna(subset=["tran_cena_brutto"])
df_top = df_top[df_top["tran_cena_brutto"].between(1000, df_top["tran_cena_brutto"].quantile(0.98))]
df_top.boxplot(column="tran_cena_brutto", by="bud_rodzaj", ax=axes[1], rot=20)
axes[1].set_title("Cena wg rodzaju budynku (5 najczęstszych)")
axes[1].set_xlabel("")
axes[1].set_ylabel("Cena brutto [PLN]")
plt.suptitle("")
save_plot(OUTPUT_DIR, "eda_ceny.png")

fig, ax = plt.subplots(figsize=(7, 5))
df_rynek = df.dropna(subset=["tran_cena_brutto", "tran_rodzaj_rynku"])
df_rynek = df_rynek[df_rynek["tran_cena_brutto"].between(1000, df_rynek["tran_cena_brutto"].quantile(0.98))]
df_rynek.boxplot(column="tran_cena_brutto", by="tran_rodzaj_rynku", ax=ax)
ax.set_title("Cena wg rodzaju rynku")
ax.set_xlabel("")
ax.set_ylabel("Cena brutto [PLN]")
plt.suptitle("")
save_plot(OUTPUT_DIR, "eda_rynek.png")

num_cols = ["tran_cena_brutto", "nier_pow_gruntu", "bud_pow_uzyt", "nier_cena_brutto"]
corr = df[num_cols].dropna().corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Macierz korelacji cech numerycznych")
save_plot(OUTPUT_DIR, "eda_korelacja.png")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print_section_header(3, "FEATURE ENGINEERING")

df2 = df.copy()

# ── 3a. Miasto z bud_adres (format "MSC:NazwaMiasta;...") ──
df2["miasto"] = df2["bud_adres"].apply(extract_miasto)
print(f"Unikalne miasta: {df2['miasto'].nunique()}")
print("Top 15 miast:")
print(df2["miasto"].value_counts().head(15))

# ── 3b. Rok i miesiąc z dok_data ──
df2["dok_data_dt"] = pd.to_datetime(df2["dok_data"], errors="coerce", utc=True)
df2["rok"]         = df2["dok_data_dt"].dt.year
df2["miesiac"]     = df2["dok_data_dt"].dt.month

print(f"\nRozkład lat transakcji:")
print(df2["rok"].value_counts().sort_index())

# ── 3c. Udział własności jako float ──
df2["udzial_float"] = df2["nier_udzial"].apply(parse_udzial)
# Przytnij do sensownego zakresu
df2["udzial_float"] = df2["udzial_float"].clip(0.001, 1.0)

# ── 3d. Filtr dat – tylko 2020–2025 ──
df2 = df2[(df2["rok"] >= 2020) & (df2["rok"] <= 2025)].copy()
print(f"\nRekordy po filtrze dat 2020–2025: {len(df2):,}")

# ─────────────────────────────────────────────
# 4. CZYSZCZENIE I PRZYGOTOWANIE DO MODELOWANIA
# ─────────────────────────────────────────────
print_section_header(4, "PRZYGOTOWANIE DANYCH DO MODELOWANIA")

TARGET = "tran_cena_brutto"

# Kolumny potrzebne do modelu + target
RAW_COLS = [
    "bud_rodzaj", "tran_rodzaj_rynku", "tran_rodzaj_trans",
    "tran_sprzedajacy", "tran_kupujacy", "nier_rodzaj", "nier_prawo",
    "miasto", "bud_pow_uzyt", "nier_pow_gruntu", "udzial_float",
    "teryt", "rok", "miesiac", "nier_cena_brutto", TARGET,
]

df_model = df2[RAW_COLS].copy()

# Usuń wiersze bez ceny docelowej
df_model = df_model.dropna(subset=[TARGET])

# Filtruj nieprawdopodobne ceny (percentyle 1–95)
df_model = filter_by_percentile(df_model, TARGET, lower=0.01, upper=0.95)
print(f"Rekordy po filtrowaniu cen (Q1–Q95): {len(df_model):,}")

# Uzupełnij braki numeryczne medianą
for col in ["bud_pow_uzyt", "nier_pow_gruntu", "nier_cena_brutto"]:
    med = df_model[col].median()
    df_model[col] = df_model[col].fillna(med)

# Uzupełnij braki kategoryczne
cat_cols = [
    "bud_rodzaj", "tran_rodzaj_rynku", "tran_rodzaj_trans",
    "tran_sprzedajacy", "tran_kupujacy", "nier_rodzaj", "nier_prawo",
]
for col in cat_cols:
    df_model[col] = df_model[col].fillna("nieznany")

df_model["miasto"] = df_model["miasto"].fillna("NIEZNANY")

# ── 4a. Flagi ──
df_model["ma_pow_uzyt"]  = (df_model["bud_pow_uzyt"] > 0).astype(int)
df_model["ma_nier_cena"] = (df_model["nier_cena_brutto"] > 0).astype(int)

# ── 4b. Target encoding: mediana ceny wg TERYT i wg miasta ──
teryt_med   = df_model.groupby("teryt")[TARGET].median().to_dict()
miasto_med  = df_model.groupby("miasto")[TARGET].median().to_dict()

df_model["teryt_med"]  = df_model["teryt"].map(teryt_med)
df_model["miasto_med"] = df_model["miasto"].map(miasto_med)

# Zapis target encodings
target_encodings = {
    "teryt_med":  teryt_med,
    "miasto_med": miasto_med,
}
joblib.dump(target_encodings, f"{OUTPUT_DIR}/target_encodings.pkl")
print(f"Zapisano: {OUTPUT_DIR}/target_encodings.pkl")

# ── 4c. Label encoding zmiennych kategorycznych ──
encoders = {}
for col in cat_cols + ["miasto"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, f"{OUTPUT_DIR}/encoders.pkl")
print(f"Zapisano: {OUTPUT_DIR}/encoders.pkl  (klucze: {list(encoders.keys())})")

# ── 4d. Statystyki miast (do app – zakładka Analiza rynku) ──
# Używamy oryginalnych nazw miast (przed encodingiem) z df2
df_city = df2[["miasto", TARGET]].copy()
df_city = df_city.dropna(subset=[TARGET])
df_city = filter_by_percentile(df_city, TARGET, lower=0.01, upper=0.95)

city_stats = (
    df_city.groupby("miasto")[TARGET]
    .agg(n="count", mediana="median", srednia="mean")
    .reset_index()
    .sort_values("n", ascending=False)
)
city_stats.to_csv(f"{OUTPUT_DIR}/city_stats.csv", index=False)
print(f"Zapisano: {OUTPUT_DIR}/city_stats.csv  ({len(city_stats)} miast)")

# ── 4e. Ceny roczne (do app – zakładka Analiza rynku) ──
yearly_prices = (
    df_model.copy()
    .assign(rok_orig=df2.loc[df_model.index, "rok"] if "rok" in df2.columns else df_model["rok"])
    .groupby("rok")[TARGET]
    .median()
    .reset_index()
)
yearly_prices.to_csv(f"{OUTPUT_DIR}/yearly_prices.csv", index=False)
print(f"Zapisano: {OUTPUT_DIR}/yearly_prices.csv")

# ── 4f. Definicja finalnych cech modelu ──
FEATURES = [
    "bud_rodzaj", "tran_rodzaj_rynku", "tran_rodzaj_trans",
    "tran_sprzedajacy", "tran_kupujacy", "nier_rodzaj", "nier_prawo",
    "miasto", "bud_pow_uzyt", "nier_pow_gruntu", "udzial_float",
    "teryt", "rok", "miesiac", "ma_pow_uzyt",
    "miasto_med", "teryt_med", "nier_cena_brutto", "ma_nier_cena",
]

print(f"\nCechy w modelu ({len(FEATURES)}): {FEATURES}")
print(f"Zmienna docelowa: {TARGET}")

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTreningowy: {len(X_train):,} próbek | Testowy: {len(X_test):,} próbek")

# ─────────────────────────────────────────────
# 5. TRENOWANIE MODELI
# ─────────────────────────────────────────────
print_section_header(5, "TRENOWANIE MODELI")

results = {}

# ── 5a. Regresja liniowa ──
results["Regresja liniowa"] = train_and_evaluate(
    LinearRegression(), "Regresja liniowa", X_train, y_train, X_test, y_test
)

# ── 5b. Random Forest ──
results["Random Forest"] = train_and_evaluate(
    RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    "Random Forest", X_train, y_train, X_test, y_test
)

# ── 5c. XGBoost ──
if XGBOOST_AVAILABLE:
    results["XGBoost"] = train_and_evaluate(
        XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "XGBoost", X_train, y_train, X_test, y_test
    )

# ─────────────────────────────────────────────
# 6. PORÓWNANIE MODELI
# ─────────────────────────────────────────────
print_section_header(6, "PORÓWNANIE MODELI")

summary = pd.DataFrame(
    {name: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]}
     for name, v in results.items()}
).T.sort_values("R²", ascending=False)
print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

best_name  = summary.index[0]
best_preds = results[best_name]["preds"]
print(f"\nNajlepszy model: {best_name}")

# Wykres porównania metryk
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metric_keys = [("MAE", "MAE"), ("RMSE", "RMSE"), ("R²", "R2")]
colors = ["#42A5F5", "#EF5350", "#66BB6A"]
for i, ((label, key), color) in enumerate(zip(metric_keys, colors)):
    vals  = [results[m][key] for m in results]
    names = list(results.keys())
    bars  = axes[i].bar(names, vals, color=color)
    axes[i].set_title(label)
    axes[i].set_ylabel(label)
    axes[i].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, vals):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:,.0f}" if label != "R²" else f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
plt.suptitle("Porównanie modeli predykcyjnych")
save_plot(OUTPUT_DIR, "porownanie_modeli.png")

# Wykres: wartości rzeczywiste vs przewidywane
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, best_preds, alpha=0.2, s=5, color="#1565C0")
lim = max(float(y_test.max()), float(best_preds.max()))
ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Idealna predykcja")
ax.set_xlabel("Cena rzeczywista [PLN]")
ax.set_ylabel("Cena przewidywana [PLN]")
ax.set_title(f"Rzeczywiste vs Przewidywane – {best_name}")
ax.legend()
save_plot(OUTPUT_DIR, "predykcja_vs_rzeczywiste.png")

# Ważność cech – Random Forest
rf = results["Random Forest"]["model"]
rf_importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
rf_importances.plot(kind="barh", ax=ax, color="#7E57C2")
ax.set_title("Ważność cech – Random Forest")
ax.set_xlabel("Ważność")
save_plot(OUTPUT_DIR, "waznosc_cech.png")

# Zapisz feature importance jako CSV (wymagane przez app)
fi_df = pd.DataFrame({
    "cecha": FEATURES,
    "waga":  rf.feature_importances_,
})
fi_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
print(f"Zapisano: {OUTPUT_DIR}/feature_importance.csv")

# ─────────────────────────────────────────────
# 7. ZAPIS MODELU I METRYK
# ─────────────────────────────────────────────
print_section_header(7, "ZAPIS WYTRENOWANEGO MODELU I METRYK")

best_model = results[best_name]["model"]
joblib.dump(best_model, f"{OUTPUT_DIR}/model_najlepszy.pkl")
print(f"Model '{best_name}' zapisany do: {OUTPUT_DIR}/model_najlepszy.pkl")

# Metryki – format wymagany przez app_streamlit.py
metryki = {
    name: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"]}
    for name, v in results.items()
}
metryki["_best"]     = best_name
metryki["_features"] = FEATURES
joblib.dump(metryki, f"{OUTPUT_DIR}/metryki.pkl")
print(f"Metryki zapisane do: {OUTPUT_DIR}/metryki.pkl")

# ── Podsumowanie zapisanych plików ──
print("\n" + "=" * 60)
print("PLIKI WYNIKOWE W FOLDERZE:", OUTPUT_DIR)
print("=" * 60)
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size  = os.path.getsize(fpath)
    print(f"  {fname:<40} {size/1024:>8.1f} KB")

print("\nGotowe! Uruchom teraz: streamlit run app_streamlit.py")