"""
Aplikacja webowa – Estymacja cen nieruchomości (Kujawsko-Pomorskie)
Dane: RCiWN 2020–2025 | Praca inżynierska – Jakub Piechowiak

Uruchomienie:
    pip install streamlit plotly joblib scikit-learn pandas numpy xgboost
    streamlit run app_streamlit.py
"""

import logging
import os
from fractions import Fraction

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger(__name__)

from utils import safe_encode as _safe_encode, build_prediction_row

st.set_page_config(page_title="Estymacja cen nieruchomości", page_icon="🏙️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
:root{--bg:#0d0f14;--surface:#161920;--surface2:#1e2230;--border:#2a2f3d;
      --accent:#4f8ef7;--accent2:#7c5cbf;--gold:#f0b429;--green:#34d399;
      --text:#e8ecf3;--muted:#7b8599;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;font-family:'Inter',sans-serif;color:var(--text);}
[data-testid="stSidebar"]{display:none!important;}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:var(--text)!important;letter-spacing:-0.02em;}
[data-testid="stMetric"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1rem 1.2rem!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:0.78rem!important;text-transform:uppercase;letter-spacing:0.06em;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-family:'Space Grotesk',sans-serif!important;font-size:1.5rem!important;}
label{color:var(--muted)!important;font-size:0.82rem!important;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;}
[data-testid="stButton"]>button{background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%)!important;color:white!important;border:none!important;border-radius:10px!important;padding:0.75rem 2rem!important;font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;font-size:1rem!important;width:100%;}
[data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:10px;gap:4px;padding:4px;}
[data-baseweb="tab"]{background:transparent!important;border-radius:8px!important;color:var(--muted)!important;}
[aria-selected="true"]{background:var(--surface2)!important;color:var(--text)!important;}
hr{border-color:var(--border)!important;}
.result-card{background:linear-gradient(135deg,#1a2035 0%,#1e1630 100%);border:1px solid var(--accent);border-radius:16px;padding:2rem 2.5rem;text-align:center;margin:1.5rem 0;box-shadow:0 0 40px rgba(79,142,247,0.12);}
.result-price{font-family:'Space Grotesk',sans-serif;font-size:3rem;font-weight:700;background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;}
.result-label{color:var(--muted);font-size:0.85rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;}
.result-range{color:var(--muted);font-size:0.95rem;margin-top:0.75rem;}
.eyebrow{color:var(--accent);font-size:0.72rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem;margin-top:1rem;}
.form-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.5rem;margin-bottom:1rem;}
.info-box{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.6rem 0.9rem;margin:0.3rem 0;font-size:0.82rem;color:var(--muted);}
.model-badge{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem;}
.model-badge.best{border-color:var(--gold);}
#MainMenu,footer,[data-testid="stHeader"],[data-testid="stToolbar"]{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

WYNIKI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wyniki")

REQUIRED_ARTIFACTS = [
    "model_najlepszy.pkl",
    "encoders.pkl",
    "metryki.pkl",
    "target_encodings.pkl",
]

missing = [f for f in REQUIRED_ARTIFACTS if not os.path.isfile(os.path.join(WYNIKI_DIR, f))]
if missing:
    st.error(
        f"Brakuje wymaganych plikow: {', '.join(missing)}. "
        "Uruchom najpierw `analiza_nieruchomosci.py`."
    )
    st.stop()


@st.cache_resource
def load_artifacts():
    try:
        model    = joblib.load(os.path.join(WYNIKI_DIR, "model_najlepszy.pkl"))
        encoders = joblib.load(os.path.join(WYNIKI_DIR, "encoders.pkl"))
        metryki  = joblib.load(os.path.join(WYNIKI_DIR, "metryki.pkl"))
        te       = joblib.load(os.path.join(WYNIKI_DIR, "target_encodings.pkl"))
    except Exception as exc:
        log.error("Blad wczytywania artefaktow modelu: %s", exc)
        st.error(f"Nie udalo sie wczytac artefaktow modelu: {exc}")
        st.stop()
    return model, encoders, metryki, te


@st.cache_data
def load_stats():
    try:
        city   = pd.read_csv(os.path.join(WYNIKI_DIR, "city_stats.csv"))
        yearly = pd.read_csv(os.path.join(WYNIKI_DIR, "yearly_prices.csv"))
        fi     = pd.read_csv(os.path.join(WYNIKI_DIR, "feature_importance.csv"), names=["cecha","waga"], header=0)
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        log.error("Blad wczytywania statystyk CSV: %s", exc)
        st.error(f"Nie udalo sie wczytac pliku statystyk: {exc}")
        st.stop()
    return city, yearly, fi

model, encoders, metryki, te = load_artifacts()
city_stats, yearly_prices, feat_imp = load_stats()
best_model_name = metryki["_best"]
FEATURES = metryki["_features"]

# ── Pobierz listę miast bezpośrednio z encodera (zawsze zgodna z modelem) ──
MIASTA_LIST = sorted([m for m in encoders["miasto"].classes_ if m != "NIEZNANY" and m != "INNE"])

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#7b8599", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10)
)

TERYT_OPTIONS = {
    "461 – Bydgoszcz (miasto)": 461, "463 – Toruń (miasto)": 463,
    "462 – Grudziądz (miasto)": 462, "403 – Bydgoski": 403,
    "404 – Chełmiński": 404, "405 – Golubsko-Dobrzyński": 405,
    "406 – Grudziądzki": 406, "407 – Inowrocławski": 407,
    "408 – Lipnowski": 408, "409 – Mogileński": 409,
    "410 – Nakielski (Szubin)": 410, "412 – Radziejowski": 412,
    "413 – Sępoleński": 413, "414 – Świecki": 414,
    "415 – Toruński": 415, "416 – Tucholski": 416,
    "417 – Wąbrzeski": 417, "418 – Włocławski": 418, "419 – Żniński": 419,
}
BUD_LABELS = {
    "mieszkalny":              "🏠 Mieszkalny",
    "handlowoUslugowy":        "🏪 Handlowo-usługowy",
    "biurowy":                 "🏢 Biurowy",
    "przemyslowy":             "🏭 Przemysłowy",
    "gospodarczy":             "🏗️ Gospodarczy",
    "transportuILacznosci":    "🚉 Transport i łączność",
    "zbiornikiSilosyMagazyny": "🏔️ Zbiorniki/magazyny",
    "oswiatyISportu":          "🏫 Oświata i sport",
    "szpitale":                "🏥 Szpitale",
    "pozostaleNiemieszkalne":  "🏛️ Pozostałe",
    "nieznany":                "❔ Nieznany",
}
NIER_LABELS = {
    "nieruchomoscLokalowa":            "🚪 Lokal",
    "nieruchomoscGruntowaZabudowana":  "🏘️ Grunt zabudowany",
    "nieruchomoscBudynkowa":           "🏗️ Budynkowa",
    "nieruchomoscGruntowaNiezabudowana":"🌿 Grunt niezabudowany",
}
RYNEK_LABELS = {
    "wtorny":   "📦 Wtórny",
    "pierwotny":"✨ Pierwotny",
    "nieznany": "❔ Nieznany",
}
TRANS_LABELS = {
    "wolnyRynek":                        "🤝 Wolny rynek",
    "sprzedazBezprzetargowa":            "📋 Bez przetargu",
    "sprzedazZBonifikata":               "💸 Z bonifikatą",
    "sprzedazPrzetargowa":               "🔨 Przetarg",
    "sprzedazWPostepowaniuEgzekucyjnym": "⚖️ Egzekucja",
    "sprzedazNaCelPubliczny":            "🏛️ Cel publiczny",
}
STRONA_LABELS = {
    "osobaFizyczna":                   "👤 Osoba fizyczna",
    "jednostkaSamorzaduTerytorialnego":"🏛️ JST",
    "skarbPanstwa":                    "🇵🇱 Skarb Państwa",
    "osobaPrawna":                     "🏢 Osoba prawna",
}
PRAWO_LABELS = {
    "wlasnoscLokaluWrazZPrawemZwiazanym":   "🔑 Własność lokalu",
    "wlasnoscNieruchomosciGruntowej":        "📋 Własność gruntowa",
    "wlasnoscBudynkuWrazZPrawemZwiazanym":  "🏗️ Własność budynku",
    "uzytkowanieWieczyste":                 "⏳ Użytkowanie wieczyste",
    "nieznany":                             "❔ Nieznane",
}

# ── HEADER ──
st.markdown("""
<div style="padding:1rem 0 1.5rem 0;">
    <p class="eyebrow" style="margin-top:0">Rejestr Cen i Wartości Nieruchomości · 2020–2025</p>
    <h1 style="font-size:2.2rem;margin:0 0 0.4rem 0;">🏙️ Estymacja cen nieruchomości</h1>
    <p style="color:#7b8599;font-size:0.95rem;margin:0;">
        Kujawsko-Pomorskie &nbsp;·&nbsp; 205 425 transakcji &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp;
        <span style="color:#34d399;font-weight:600;">R² = 0.747</span>
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Predykcja", "📊 Analiza rynku", "🤖 Modele"])

# ══ TAB 1 – PREDYKCJA ══
with tab1:
    st.markdown('<p class="eyebrow">Parametry nieruchomości</p>', unsafe_allow_html=True)

    # ROW 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bud_rodzaj = st.selectbox(
            "Rodzaj budynku",
            list(BUD_LABELS.keys()),
            format_func=lambda x: BUD_LABELS[x]
        )
    with c2:
        nier_rodzaj = st.selectbox(
            "Rodzaj nieruchomości",
            list(NIER_LABELS.keys()),
            format_func=lambda x: NIER_LABELS[x]
        )
    with c3:
        nier_prawo = st.selectbox(
            "Prawo",
            list(PRAWO_LABELS.keys()),
            format_func=lambda x: PRAWO_LABELS[x]
        )
    with c4:
        tran_rodzaj_rynku = st.selectbox(
            "Rynek",
            list(RYNEK_LABELS.keys()),
            format_func=lambda x: RYNEK_LABELS[x]
        )

    # ROW 2
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        tran_rodzaj_trans = st.selectbox(
            "Rodzaj transakcji",
            list(TRANS_LABELS.keys()),
            format_func=lambda x: TRANS_LABELS[x]
        )
    with c6:
        tran_sprzedajacy = st.selectbox(
            "Sprzedający",
            list(STRONA_LABELS.keys()),
            format_func=lambda x: STRONA_LABELS[x]
        )
    with c7:
        tran_kupujacy = st.selectbox(
            "Kupujący",
            list(STRONA_LABELS.keys()),
            format_func=lambda x: STRONA_LABELS[x]
        )
    with c8:
        # Lista miast pobrana z encodera — zawsze zgodna z modelem
        miasto_wybor = st.selectbox(
            "Miasto",
            ["NIEZNANY"] + MIASTA_LIST
        )
        teryt_label = st.selectbox("Powiat", list(TERYT_OPTIONS.keys()), index=10)

    # ROW 3
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        bud_pow_uzyt = st.number_input("Pow. użytkowa budynku [m²]", 0.0, 5000.0, 65.0, 5.0)
    with c10:
        nier_pow_gruntu = st.number_input("Pow. działki [m²]", 0.0, 50000.0, 0.0, 50.0)
    with c11:
        udzial_str = st.selectbox("Udział własności", ["1/1", "1/2", "1/3", "1/4", "2/3", "3/4"])
    with c12:
        rok     = st.slider("Rok transakcji", 2020, 2025, 2025)
        miesiac = st.slider("Miesiąc", 1, 12, 6)

    with st.expander("💰 Znana cena nieruchomości (opcjonalne — poprawia dokładność)"):
        nier_cena_brutto_input = st.number_input(
            "Cena nieruchomości [PLN] — zostaw 0 jeśli nieznana",
            min_value=0.0, max_value=50_000_000.0, value=0.0, step=10_000.0
        )

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 2, 2])
    with btn_col:
        predict_btn = st.button("🔮 Szacuj cenę", type="primary")

    st.markdown("---")

    if predict_btn:
        # ── Przygotowanie wartości ──
        try:
            udzial_float = float(Fraction(udzial_str))
        except (ValueError, ZeroDivisionError) as exc:
            log.warning("Nie udalo sie sparsowac udzialu '%s': %s", udzial_str, exc)
            udzial_float = 1.0

        teryt = TERYT_OPTIONS[teryt_label]
        pow_uzyt_val = bud_pow_uzyt if bud_pow_uzyt > 0 else 65.0

        # Mediany z target encodings
        teryt_med_val  = te["teryt_med"].get(teryt, float(np.median(list(te["teryt_med"].values()))))
        # Dla miasta: "NIEZNANY" istnieje w encoderze i w te — użyj bezpośrednio
        miasto_med_val = te["miasto_med"].get(miasto_wybor, te["miasto_med"].get("NIEZNANY", 300_000.0))

        # Cena nieruchomości: jeśli użytkownik podał — użyj; inaczej mediana powiatu
        nier_cena_val    = nier_cena_brutto_input if nier_cena_brutto_input > 0 else teryt_med_val
        ma_nier_cena_val = int(nier_cena_brutto_input > 0)

        # ── Enkodowanie kategorii ──
        cat_inputs = {
            "bud_rodzaj":        bud_rodzaj,
            "tran_rodzaj_rynku": tran_rodzaj_rynku,
            "tran_rodzaj_trans": tran_rodzaj_trans,
            "tran_sprzedajacy":  tran_sprzedajacy,
            "tran_kupujacy":     tran_kupujacy,
            "nier_rodzaj":       nier_rodzaj,
            "nier_prawo":        nier_prawo,
            "miasto":            miasto_wybor,
        }
        encoded = {col: _safe_encode(col, val, encoders) for col, val in cat_inputs.items()}

        # ── Budowa wiersza wejściowego (kolejność = FEATURES z metryki.pkl) ──
        row = build_prediction_row(
            encoded_cats=encoded,
            bud_pow_uzyt=pow_uzyt_val,
            nier_pow_gruntu=nier_pow_gruntu,
            udzial_float=udzial_float,
            teryt=teryt,
            rok=rok,
            miesiac=miesiac,
            miasto_med_val=miasto_med_val,
            teryt_med_val=teryt_med_val,
            nier_cena_val=nier_cena_val,
            ma_nier_cena_val=ma_nier_cena_val,
            features_order=FEATURES,
        )

        # ── Predykcja ──
        try:
            pred = float(model.predict(row)[0])
        except Exception as exc:
            log.error("Blad predykcji modelu: %s", exc)
            st.error(f"Blad predykcji modelu: {exc}")
            st.stop()

        if pred < 0:
            log.warning("Model zwrocil ujemna cene (%.2f), ustawiam na 0", pred)
            pred = 0.0

        pred_low  = pred * 0.85
        pred_high = pred * 1.15
        cena_m2   = pred / pow_uzyt_val if pow_uzyt_val > 0 else None

        # ── Wyniki ──
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Szacowana cena transakcji</div>
            <div class="result-price">{pred:,.0f} PLN</div>
            <div class="result-range">Przedział ufności (±15%): {pred_low:,.0f} – {pred_high:,.0f} PLN</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Szacowana cena",  f"{pred:,.0f} PLN")
        m2.metric("Cena za m²",      f"{cena_m2:,.0f} PLN/m²" if cena_m2 else "—")
        m3.metric("Mediana powiatu", f"{teryt_med_val:,.0f} PLN")
        m4.metric("Model R²",        f"{metryki[best_model_name]['R2']:.3f}")

        col_g, col_d = st.columns([3, 2])
        with col_g:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                delta={
                    "reference": teryt_med_val,
                    "valueformat": ",.0f", "suffix": " PLN",
                    "increasing": {"color": "#f87171"},
                    "decreasing": {"color": "#34d399"},
                },
                number={"suffix": " PLN", "valueformat": ",.0f", "font": {"size": 26, "color": "#e8ecf3"}},
                gauge={
                    "axis": {"range": [0, pred * 2.5], "tickcolor": "#7b8599", "tickfont": {"color": "#7b8599"}},
                    "bar": {"color": "#4f8ef7"},
                    "bgcolor": "#1e2230",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,          pred * 0.85], "color": "#1e2230"},
                        {"range": [pred * 0.85, pred * 1.15], "color": "#1a2c4a"},
                        {"range": [pred * 1.15, pred * 2.5],  "color": "#1e2230"},
                    ],
                    "threshold": {"line": {"color": "#f0b429", "width": 2}, "thickness": 0.75, "value": teryt_med_val},
                },
            ))
            fig_gauge.update_layout(
                **PLOTLY_LAYOUT, height=240,
                title=dict(
                    text=f"Predykcja vs mediana powiatu ({teryt_med_val:,.0f} PLN)",
                    font=dict(color="#7b8599", size=12)
                )
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_d:
            st.markdown('<p class="eyebrow">Podsumowanie</p>', unsafe_allow_html=True)
            powiat_nazwa = teryt_label.split("–")[1].strip() if "–" in teryt_label else teryt_label
            items = {
                "Budynek":       BUD_LABELS[bud_rodzaj],
                "Rynek":         RYNEK_LABELS[tran_rodzaj_rynku],
                "Transakcja":    TRANS_LABELS[tran_rodzaj_trans],
                "Pow. użytkowa": f"{pow_uzyt_val:.1f} m²",
                "Pow. gruntu":   f"{nier_pow_gruntu:.0f} m²",
                "Cena nier.":    f"{nier_cena_val:,.0f} PLN",
                "Miasto":        miasto_wybor,
                "Powiat":        powiat_nazwa,
                "Rok/Miesiąc":   f"{rok}/{miesiac:02d}",
            }
            for k, v in items.items():
                st.markdown(
                    f'<div class="info-box"><b style="color:#e8ecf3">{k}:</b> {v}</div>',
                    unsafe_allow_html=True
                )

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;color:#7b8599;">
            <div style="font-size:3.5rem;margin-bottom:1rem;">🏙️</div>
            <h3 style="color:#e8ecf3">Gotowy do szacowania</h3>
            <p>Wypełnij parametry powyżej i kliknij <strong style="color:#4f8ef7">Szacuj cenę</strong>.</p>
            <p style="font-size:0.85rem;margin-top:0.5rem;">Model Random Forest · R² = 0.747 · MAE ≈ 48 400 PLN</p>
        </div>
        """, unsafe_allow_html=True)


# ══ TAB 2 – ANALIZA RYNKU ══
with tab2:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="eyebrow">Mediana ceny transakcji wg roku</p>', unsafe_allow_html=True)
        fig_y = go.Figure()
        fig_y.add_trace(go.Scatter(
            x=yearly_prices["rok"],
            y=yearly_prices["tran_cena_brutto"],
            mode="lines+markers+text",
            text=[f"{v/1000:.0f}k" for v in yearly_prices["tran_cena_brutto"]],
            textposition="top center",
            textfont=dict(color="#e8ecf3", size=11),
            line=dict(color="#4f8ef7", width=2.5),
            marker=dict(size=9, color="#4f8ef7"),
            fill="tozeroy",
            fillcolor="rgba(79,142,247,0.08)",
        ))
        fig_y.update_layout(
            **PLOTLY_LAYOUT, height=290,
            xaxis=dict(gridcolor="#1e2230", tickvals=yearly_prices["rok"].tolist()),
            yaxis=dict(gridcolor="#1e2230", tickformat=",.0f"),
        )
        st.plotly_chart(fig_y, use_container_width=True)

    with col_r:
        st.markdown('<p class="eyebrow">Top 15 miast – mediana ceny [PLN]</p>', unsafe_allow_html=True)
        top_c = city_stats[
            (city_stats["n"] >= 30) & (city_stats["miasto"] != "NIEZNANY")
        ].nlargest(15, "mediana")
        fig_c = go.Figure(go.Bar(
            x=top_c["mediana"],
            y=top_c["miasto"],
            orientation="h",
            marker=dict(
                color=top_c["mediana"],
                colorscale=[[0, "#1a2c4a"], [1, "#7c5cbf"]],
                showscale=False,
            ),
            text=[f"{v/1000:.0f}k" for v in top_c["mediana"]],
            textposition="outside",
            textfont=dict(color="#e8ecf3"),
        ))
        fig_c.update_layout(
            **PLOTLY_LAYOUT, height=370,
            xaxis=dict(gridcolor="#1e2230", tickformat=",.0f"),
            yaxis=dict(gridcolor="#1e2230", categoryorder="total ascending"),
        )
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="eyebrow">Wolumen transakcji – top 20 miast</p>', unsafe_allow_html=True)
    top20 = city_stats[city_stats["miasto"] != "NIEZNANY"].head(21)
    fig_vol = go.Figure(go.Bar(
        x=top20["miasto"],
        y=top20["n"],
        marker=dict(
            color=top20["mediana"],
            colorscale=[[0, "#1a2c4a"], [0.5, "#4f8ef7"], [1, "#7c5cbf"]],
            colorbar=dict(title="Mediana PLN", tickformat=",.0f", tickfont=dict(color="#7b8599")),
            showscale=True,
        ),
        text=top20["n"],
        textposition="outside",
        textfont=dict(color="#7b8599"),
    ))
    fig_vol.update_layout(
        **PLOTLY_LAYOUT, height=310,
        xaxis=dict(gridcolor="#1e2230", tickangle=-30),
        yaxis=dict(gridcolor="#1e2230", title="Liczba transakcji"),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown('<p class="eyebrow" style="margin-top:1rem">Tabela statystyk miast</p>', unsafe_allow_html=True)
    disp = city_stats[
        (city_stats["n"] >= 10) & (city_stats["miasto"] != "NIEZNANY")
    ].copy()
    disp["mediana"] = disp["mediana"].map("{:,.0f} PLN".format)
    disp["srednia"] = disp["srednia"].map("{:,.0f} PLN".format)
    disp.columns = ["Miasto", "Transakcje", "Mediana", "Średnia"]
    st.dataframe(disp.head(60), use_container_width=True, hide_index=True)


# ══ TAB 3 – MODELE ══
with tab3:
    st.markdown('<p class="eyebrow">Porównanie algorytmów – dane 2020–2025</p>', unsafe_allow_html=True)
    names = [k for k in metryki if k not in ("_best", "_features")]

    cols = st.columns(len(names))
    for col, mname in zip(cols, names):
        m = metryki[mname]
        is_best = mname == best_model_name
        with col:
            st.markdown(f"""
            <div class="model-badge {'best' if is_best else ''}">
                <div style="font-weight:600;font-size:0.95rem;color:{'#f0b429' if is_best else '#e8ecf3'}">
                    {'🥇 ' if is_best else ''}{mname}
                </div>
                <div style="margin-top:0.6rem;color:#7b8599;font-size:0.85rem;line-height:1.8">
                    MAE: <span style="color:#e8ecf3">{m['MAE']:,.0f} PLN</span><br>
                    RMSE: <span style="color:#e8ecf3">{m['RMSE']:,.0f} PLN</span><br>
                    R²: <span style="color:{'#34d399' if is_best else '#4f8ef7'};font-size:1.2rem;font-weight:700">{m['R2']:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="eyebrow">R²</p>', unsafe_allow_html=True)
        r2s = [metryki[k]["R2"] for k in names]
        fig_r2 = go.Figure(go.Bar(
            x=names, y=r2s,
            marker_color=["#f0b429" if n == best_model_name else "#4f8ef7" for n in names],
            text=[f"{v:.4f}" for v in r2s],
            textposition="outside",
            textfont=dict(color="#e8ecf3"),
        ))
        fig_r2.update_layout(
            **PLOTLY_LAYOUT, height=260,
            yaxis=dict(gridcolor="#1e2230", range=[0, max(r2s) * 1.3]),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_b:
        st.markdown('<p class="eyebrow">MAE [PLN]</p>', unsafe_allow_html=True)
        maes = [metryki[k]["MAE"] for k in names]
        fig_mae = go.Figure(go.Bar(
            x=names, y=maes,
            marker_color=["#f0b429" if n == best_model_name else "#7c5cbf" for n in names],
            text=[f"{v:,.0f}" for v in maes],
            textposition="outside",
            textfont=dict(color="#e8ecf3"),
        ))
        fig_mae.update_layout(
            **PLOTLY_LAYOUT, height=260,
            yaxis=dict(gridcolor="#1e2230", tickformat=",.0f"),
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="eyebrow">Ważność cech – Random Forest</p>', unsafe_allow_html=True)
    fi_sorted = feat_imp.sort_values("waga", ascending=True).copy()
    feat_labels = {
        "bud_rodzaj":        "Rodzaj budynku",
        "tran_rodzaj_rynku": "Rodzaj rynku",
        "tran_rodzaj_trans": "Rodzaj transakcji",
        "tran_sprzedajacy":  "Sprzedający",
        "tran_kupujacy":     "Kupujący",
        "nier_rodzaj":       "Rodzaj nieruchomości",
        "nier_prawo":        "Prawo",
        "miasto":            "Miasto",
        "bud_pow_uzyt":      "Pow. użytkowa",
        "nier_pow_gruntu":   "Pow. gruntu",
        "udzial_float":      "Udział własności",
        "teryt":             "TERYT",
        "rok":               "Rok",
        "miesiac":           "Miesiąc",
        "ma_pow_uzyt":       "Flaga pow. uzyt.",
        "miasto_med":        "Mediana ceny miasta",
        "teryt_med":         "Mediana ceny powiatu",
        "nier_cena_brutto":  "Cena nieruchomości",
        "ma_nier_cena":      "Flaga ceny nier.",
    }
    fi_sorted["cecha_pl"] = fi_sorted["cecha"].map(feat_labels).fillna(fi_sorted["cecha"])
    fi_sorted["pct"] = fi_sorted["waga"] * 100
    fig_fi = go.Figure(go.Bar(
        x=fi_sorted["pct"],
        y=fi_sorted["cecha_pl"],
        orientation="h",
        marker=dict(
            color=fi_sorted["waga"],
            colorscale=[[0, "#1a2c4a"], [1, "#4f8ef7"]],
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in fi_sorted["pct"]],
        textposition="outside",
        textfont=dict(color="#7b8599"),
    ))
    fig_fi.update_layout(
        **PLOTLY_LAYOUT, height=520,
        xaxis=dict(gridcolor="#1e2230", ticksuffix="%"),
        yaxis=dict(gridcolor="#1e2230"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("ℹ️ O danych i metodologii"):
        algo_lines = []
        for algo_name, algo_label in [
            ("Regresja liniowa", "Regresja liniowa (baseline)"),
            ("Random Forest", "Random Forest (400 drzew)"),
            ("XGBoost", "XGBoost (600 estymatorów)"),
        ]:
            if algo_name in metryki:
                algo_lines.append(
                    f"- {algo_label} — R²={metryki[algo_name]['R2']:.4f}"
                )
            else:
                algo_lines.append(f"- {algo_label} — niedostępny")
        algo_text = "\n".join(algo_lines)

        st.markdown(f"""
**Źródło:** Rejestr Cen i Wartości Nieruchomości (RCiWN) · Kujawsko-Pomorskie  
**Zakres:** styczeń 2020 – grudzień 2025 · 205 425 transakcji (po filtrowaniu Q1–Q95)

**Algorytmy:**
{algo_text}

**Cechy ({len(FEATURES)}):** {', '.join(FEATURES)}  
**Podział:** 80% trening / 20% test (random_state=42)
        """)