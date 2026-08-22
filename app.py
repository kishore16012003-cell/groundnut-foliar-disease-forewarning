
"""
Groundnut Fungal Disease Forewarning System
-------------------------------------------
District-specific:
    Coimbatore
    Cuddalore

Diseases:
    Groundnut Rust (Puccinia arachidis)
    Late Leaf Spot (Tikka disease)

Forecasts:
    Current Week
    Week 1
    Week 2

The regression equations are trained automatically from the
Excel files in the data/ directory. No coefficients are hard-coded.
"""

import os
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Groundnut Fungal Disease Forewarning",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 0.7rem; }

    .hero {
        padding: 65px 35px;
        border-radius: 22px;
        margin-bottom: 25px;
        background: linear-gradient(135deg, #173d24, #2d6a3f);
        color: white;
        text-align: center;
    }

    .hero h1 { font-size: 42px; margin-bottom: 8px; }
    .hero p { font-size: 19px; margin: 0; }

    .result-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-bottom: 12px;
    }

    .scope-note {
        background: #f1f6ff;
        padding: 12px 16px;
        border-radius: 10px;
        border-left: 5px solid #4777c7;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# DATA CONFIGURATION
# ============================================================

DATASETS = {
    "Coimbatore": {
        "Rust": {
            "file": DATA_DIR / "coimbatore_rust.xlsx",
            "disease_col": "SE SCORE",
            "season": None,
        },
        "Late Leaf Spot": {
            "file": DATA_DIR / "coimbatore_lls.xlsx",
            "disease_col": "SE SCORE",
            "season": None,
        },
    },
    "Cuddalore": {
        "Rust": {
            "file": DATA_DIR / "cuddalore.xlsx",
            "disease_col": "RUST",
            "season": "KHARIF",
        },
        "Late Leaf Spot": {
            "file": DATA_DIR / "cuddalore.xlsx",
            "disease_col": "LLS",
            "season": "KHARIF",
        },
    },
}

# Use variables available in BOTH district datasets.
FEATURES = [
    "MAXIMUM TEMPERATURE",
    "MINIMUM TEMPERATURE",
    "MORNING RH",
    "EVENING  RH",
    "RAINFALL",
    "WIND SPEED",
    "DEW POINT",
]

FEATURE_LABELS = {
    "MAXIMUM TEMPERATURE": "Maximum Temperature (°C)",
    "MINIMUM TEMPERATURE": "Minimum Temperature (°C)",
    "MORNING RH": "Morning Relative Humidity (%)",
    "EVENING  RH": "Evening Relative Humidity (%)",
    "RAINFALL": "Rainfall (mm)",
    "WIND SPEED": "Wind Speed",
    "DEW POINT": "Dew Point (°C)",
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_excel(path: str, disease_col: str, season: str | None):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    # Excel files contain year/season/month values only at
    # the first row of each block, so carry them forward.
    for col in ["YEARS", "SEASONS", "MONTHS"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    if season and "SEASONS" in df.columns:
        df["SEASONS"] = df["SEASONS"].astype(str).str.upper().str.strip()
        df = df[df["SEASONS"] == season].copy()

    for col in FEATURES + [disease_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort chronologically where possible.
    sort_cols = [c for c in ["YEARS", "WEEKS"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")

    return df.reset_index(drop=True)


@st.cache_resource
def train_model(district: str, disease: str, horizon: int):
    cfg = DATASETS[district][disease]
    df = load_excel(str(cfg["file"]), cfg["disease_col"], cfg["season"])

    target_col = cfg["disease_col"]

    # Horizon 0 = current week.
    # Horizon 1 = disease score one week later.
    # Horizon 2 = disease score two weeks later.
    if horizon == 0:
        target = df[target_col].copy()
    else:
        target = df[target_col].shift(-horizon)

    work = df[FEATURES].copy()
    work["TARGET"] = target

    work = work.dropna(subset=FEATURES + ["TARGET"])

    if len(work) < 15:
        raise ValueError(
            f"Not enough observations for {district} / {disease} / "
            f"horizon {horizon}. Available: {len(work)}"
        )

    X = work[FEATURES]
    y = work["TARGET"]

    # Chronological 80/20 validation.
    split = max(1, int(len(work) * 0.80))
    if split >= len(work):
        split = len(work) - 1

    model = LinearRegression()
    model.fit(X.iloc[:split], y.iloc[:split])

    test_pred = model.predict(X.iloc[split:])

    metrics = {
        "n": len(work),
        "r2": float(r2_score(y.iloc[split:], test_pred))
        if len(y.iloc[split:]) > 1 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y.iloc[split:], test_pred)))
        if len(y.iloc[split:]) > 0 else np.nan,
        "mae": float(mean_absolute_error(y.iloc[split:], test_pred))
        if len(y.iloc[split:]) > 0 else np.nan,
    }

    # Refit on all available historical observations for production use.
    final_model = LinearRegression()
    final_model.fit(X, y)

    # Historical distribution used for data-driven risk bands.
    historical_scores = y.to_numpy(dtype=float)

    return {
        "model": final_model,
        "metrics": metrics,
        "historical_scores": historical_scores,
        "coefficients": final_model.coef_,
        "intercept": float(final_model.intercept_),
        "n": len(work),
    }


# ============================================================
# PREDICTION / EQUATION HELPERS
# ============================================================

def make_equation(model_info):
    intercept = model_info["intercept"]
    coef = model_info["coefficients"]

    equation = f"Y = {intercept:.4f}"

    for feature, b in zip(FEATURES, coef):
        sign = "+" if b >= 0 else "-"
        equation += f" {sign} {abs(b):.4f}×{feature_short(feature)}"

    return equation


def feature_short(feature):
    names = {
        "MAXIMUM TEMPERATURE": "Tmax",
        "MINIMUM TEMPERATURE": "Tmin",
        "MORNING RH": "RHm",
        "EVENING  RH": "RHe",
        "RAINFALL": "Rain",
        "WIND SPEED": "Wind",
        "DEW POINT": "Dew",
    }
    return names.get(feature, feature)


def predict_score(model_info, values):
    X = pd.DataFrame([values], columns=FEATURES)
    prediction = float(model_info["model"].predict(X)[0])

    # Disease scores cannot be negative.
    # Upper limit is taken from historical observed maximum.
    historical_max = float(np.nanmax(model_info["historical_scores"]))
    prediction = max(0.0, min(prediction, historical_max))

    return prediction


def risk_from_score(score, historical_scores):
    """
    Data-driven provisional risk classification:
    Q1 / median / Q3 of the district-disease historical
    disease-score distribution.

    This is NOT a substitute for an experimentally validated
    epidemiological severity scale.
    """
    q1, q2, q3 = np.nanpercentile(historical_scores, [25, 50, 75])

    if score <= q1:
        return "Low Risk", "green", (q1, q2, q3)
    elif score <= q2:
        return "Moderate Risk", "orange", (q1, q2, q3)
    elif score <= q3:
        return "High Risk", "red", (q1, q2, q3)
    else:
        return "Very High Risk", "darkred", (q1, q2, q3)


# ============================================================
# FAVOURABLE WEATHER CONDITIONS
# ============================================================

def rust_conditions(values):
    tmean = (values["MAXIMUM TEMPERATURE"] + values["MINIMUM TEMPERATURE"]) / 2
    rhmean = (values["MORNING RH"] + values["EVENING  RH"]) / 2
    rain = values["RAINFALL"]
    wind = values["WIND SPEED"]

    return {
        "Mean temperature": tmean,
        "Mean RH": rhmean,
        "Temperature 25–30°C": 25 <= tmean <= 30,
        "RH >85%": rhmean > 85,
        "Rainfall present": rain > 0,
        "Wind + rainfall": rain > 0 and wind > 0,
    }


def lls_conditions(values):
    tmean = (values["MAXIMUM TEMPERATURE"] + values["MINIMUM TEMPERATURE"]) / 2
    rhmean = (values["MORNING RH"] + values["EVENING  RH"]) / 2
    rain = values["RAINFALL"]

    return {
        "Mean temperature": tmean,
        "Mean RH": rhmean,
        "Temperature 20–30°C": 20 <= tmean <= 30,
        "RH >90%": rhmean > 90,
        "Rainfall present": rain > 0,
    }


# ============================================================
# UI: HOME
# ============================================================

st.sidebar.title("🌱 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Disease Selection",
        "Disease Prediction",
        "Symptoms Information",
        "IDM Practices",
        "About",
    ],
)


if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <h1>🌱 Groundnut Fungal Disease Forewarning System</h1>
            <p>
                District-specific weather-based forecasting for
                Rust and Late Leaf Spot
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Welcome

        This decision-support system provides weather-based
        forewarning of two important groundnut fungal diseases:

        - **Rust — Puccinia arachidis**
        - **Late Leaf Spot — Tikka disease**

        The forecasting system provides:

        - Current-week disease score
        - One-week-ahead forecast
        - Two-week-ahead forecast
        - District-specific models
        - Weather-favourability information
        - Farmer advisory
        - WhatsApp alert support
        """
    )

    st.info(
        "Models are developed separately for Coimbatore and Cuddalore using their respective historical weather and disease observations."
    )


# ============================================================
# UI: DISEASE SELECTION
# ============================================================

elif page == "Disease Selection":

    st.header("🦠 Select Disease")

    st.write("Choose the disease for which you want to view information or forecast risk.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟠 Groundnut Rust")
        st.write("Caused by *Puccinia arachidis*.")
        if st.button("Select Rust", use_container_width=True):
            st.session_state["disease"] = "Rust"
            st.success("Rust selected. Go to Disease Prediction.")

    with col2:
        st.subheader("🟤 Late Leaf Spot")
        st.write("Also known as Tikka disease.")
        if st.button("Select Late Leaf Spot", use_container_width=True):
            st.session_state["disease"] = "Late Leaf Spot"
            st.success("Late Leaf Spot selected. Go to Disease Prediction.")

    st.divider()

    selected = st.session_state.get("disease", "Rust")
    st.info(f"Current selection: **{selected}**")


# ============================================================
# UI: DISEASE PREDICTION
# ============================================================

elif page == "Disease Prediction":

    st.header("🔮 Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox(
            "Select District",
            ["Coimbatore", "Cuddalore"],
        )

        disease = st.selectbox(
            "Select Disease",
            ["Rust", "Late Leaf Spot"],
            index=0 if st.session_state.get("disease", "Rust") == "Rust" else 1,
        )

        if district == "Cuddalore":
        variety = st.selectbox(
            "Groundnut Variety Type",
            ["Short Duration", "Medium Duration", "Long Duration"],
        )

    with col2:
        forecast = st.selectbox(
            "Forecast Period",
            ["Current Week", "Week 1 Forecast", "Week 2 Forecast"],
        )

        phone = st.text_input(
            "Farmer WhatsApp Number (with country code)",
            placeholder="e.g. 919876543210",
        )

    st.markdown("### 🌦 Current Week Weather")

    c1, c2 = st.columns(2)

    with c1:
        x1 = st.number_input(
            "Maximum Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=30.0,
            step=0.1,
        )

        x2 = st.number_input(
            "Minimum Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=22.0,
            step=0.1,
        )

        x3 = st.number_input(
            "Morning Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
        )

        x4 = st.number_input(
            "Evening Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
        )

    with c2:
        x5 = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=10.0,
            step=0.1,
        )

        x6 = st.number_input(
            "Wind Speed",
            min_value=0.0,
            value=5.0,
            step=0.1,
        )

        x7 = st.number_input(
            "Dew Point (°C)",
            min_value=0.0,
            max_value=50.0,
            value=20.0,
            step=0.1,
        )

    st.caption(
        "Variety type is collected as crop context. It is not included as a "
        "regression variable because the uploaded historical datasets do not "
        "contain a variety-type column."
    )

    values = {
        "MAXIMUM TEMPERATURE": x1,
        "MINIMUM TEMPERATURE": x2,
        "MORNING RH": x3,
        "EVENING  RH": x4,
        "RAINFALL": x5,
        "WIND SPEED": x6,
        "DEW POINT": x7,
    }

    if st.button("🔍 Predict Disease", type="primary", use_container_width=True):

        horizon_map = {
            "Current Week": 0,
            "Week 1 Forecast": 1,
            "Week 2 Forecast": 2,
        }

        horizon = horizon_map[forecast]

        try:
            info = train_model(district, disease, horizon)
            score = predict_score(info, values)

            risk, color, quartiles = risk_from_score(
                score,
                info["historical_scores"],
            )

            st.success(
                f"### Predicted {disease} Score: {score:.2f}"
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <h2 style="color:{color};">{risk}</h2>
                    <p><b>District:</b> {district}</p>
                    <p><b>Disease:</b> {disease}</p>
                    <p><b>Forecast:</b> {forecast}</p>
                    <p><b>Variety type:</b> {variety}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --------------------------------------------
            # FAVOURABLE CONDITIONS
            # --------------------------------------------

            st.subheader("🌦 Favourable Weather Conditions")

            if disease == "Rust":
                conditions = rust_conditions(values)
            else:
                conditions = lls_conditions(values)

            a, b, c, d = st.columns(4)

            condition_items = list(conditions.items())
            flags = [
                item for item in condition_items
                if isinstance(item[1], (bool, np.bool_))
            ]

            for col, (label, flag) in zip(
                [a, b, c, d],
                flags[:4],
            ):
                with col:
                    if flag:
                        st.success("✓ " + label)
                    else:
                        st.warning("✗ " + label)

            st.write(
                f"**Mean temperature:** "
                f"{conditions['Mean temperature']:.2f} °C"
            )

            st.write(
                f"**Mean relative humidity:** "
                f"{conditions['Mean RH']:.2f} %"
            )

            # --------------------------------------------
            # GAUGE
            # --------------------------------------------

            historical_max = max(
                float(np.nanmax(info["historical_scores"])),
                1.0,
            )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": f"{disease} Disease Score"},
                    gauge={
                        "axis": {"range": [0, historical_max]},
                        "bar": {"color": color},
                    },
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            # --------------------------------------------
            # ALL THREE FORECASTS
            # --------------------------------------------

            st.subheader("📊 Current / Week 1 / Week 2 Forecast")

            all_scores = []

            for h, label in [
                (0, "Current Week"),
                (1, "Week 1"),
                (2, "Week 2"),
            ]:
                model_info = train_model(
                    district,
                    disease,
                    h,
                )

                s = predict_score(
                    model_info,
                    values,
                )

                r, _, _ = risk_from_score(
                    s,
                    model_info["historical_scores"],
                )

                all_scores.append(
                    {
                        "Forecast": label,
                        "Predicted Score": round(s, 2),
                        "Risk": r,
                    }
                )

            result_df = pd.DataFrame(all_scores)

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

            fig_trend = px.line(
                result_df,
                x="Forecast",
                y="Predicted Score",
                markers=True,
                title=f"{disease} Forecast Trend",
            )

            st.plotly_chart(
                fig_trend,
                use_container_width=True,
            )

            # --------------------------------------------
            # WEATHER CHART
            # --------------------------------------------

            weather_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Max Temp",
                        "Min Temp",
                        "Morning RH",
                        "Evening RH",
                        "Rainfall",
                        "Wind",
                        "Dew Point",
                    ],
                    "Value": [
                        x1,
                        x2,
                        x3,
                        x4,
                        x5,
                        x6,
                        x7,
                    ],
                }
            )

            fig_weather = px.bar(
                weather_df,
                x="Parameter",
                y="Value",
                title="Weather Parameters Used",
            )

            st.plotly_chart(
                fig_weather,
                use_container_width=True,
            )

            # --------------------------------------------
            # EQUATION
            # --------------------------------------------

            st.subheader("🧮 Forecast Equation")

            st.code(
                make_equation(info),
                language="text",
            )

            st.caption(
                f"Training observations: {info['n']} | "
                f"Chronological validation R²: {info['metrics']['r2']:.3f} | "
                f"RMSE: {info['metrics']['rmse']:.3f} | "
                f"MAE: {info['metrics']['mae']:.3f}"
            )

            # --------------------------------------------
            # ADVISORY
            # --------------------------------------------

            if disease == "Rust":
                advisory = (
                    "Monitor the crop closely when temperature is 25–30°C, "
                    "relative humidity is above 85%, and rainy/windy "
                    "conditions persist. Follow locally recommended "
                    "integrated disease-management practices."
                )
            else:
                advisory = (
                    "Monitor the crop closely under 20–30°C, very high "
                    "relative humidity, frequent rain/dew and prolonged "
                    "leaf-wetness conditions. Follow locally recommended "
                    "integrated disease-management practices."
                )

            st.warning("**Advisory:** " + advisory)

            # --------------------------------------------
            # WHATSAPP
            # --------------------------------------------

            selected_row = result_df[
                result_df["Forecast"] == forecast
            ].iloc[0]

            selected_score = float(
                selected_row["Predicted Score"]
            )

            selected_risk = selected_row["Risk"]

            if phone and selected_risk in ["High Risk", "Very High Risk"]:

                message = f"""
🌱 Groundnut Disease Forewarning

District: {district}
Disease: {disease}
Forecast: {forecast}

Predicted Disease Score: {selected_score:.2f}
Risk Level: {selected_risk}

Weather:
Maximum Temperature: {x1:.1f} °C
Minimum Temperature: {x2:.1f} °C
Morning RH: {x3:.1f} %
Evening RH: {x4:.1f} %
Rainfall: {x5:.1f} mm
Wind Speed: {x6:.1f}
Dew Point: {x7:.1f} °C

Advisory:
{advisory}

- Groundnut Disease Forewarning System
"""

                encoded = urllib.parse.quote(message)
                whatsapp_url = (
                    f"https://wa.me/{phone}?text={encoded}"
                )

                st.markdown(
                    f"""
                    <a href="{whatsapp_url}" target="_blank">
                        <button style="
                            background:#25D366;
                            color:white;
                            padding:12px 22px;
                            border:none;
                            border-radius:8px;
                            font-size:16px;
                            font-weight:bold;">
                            📲 Send WhatsApp Alert
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

            elif phone:
                st.info(
                    "No high-risk WhatsApp alert is required "
                    "for the selected forecast."
                )

        except Exception as exc:
            st.error(
                "Prediction could not be completed. "
                "Check the Excel file and required columns."
            )
            st.exception(exc)


# ============================================================
# SYMPTOMS
# ============================================================

elif page == "Symptoms Information":

    st.header("🌿 Disease Symptoms")

    disease = st.selectbox(
        "Select Disease",
        ["Rust", "Late Leaf Spot"],
    )

    if disease == "Rust":

        st.subheader("Groundnut Rust — Puccinia arachidis")

        st.write(
            """
            Rust produces small rust-coloured pustules on the leaves.
            Under favourable conditions, lesions and pustules can increase
            rapidly and contribute to premature deterioration of foliage.
            """
        )

        st.markdown("### Typical symptoms")
        st.markdown(
            """
            - Rust-coloured or reddish-brown pustules
            - Pustules mainly visible on leaves
            - Increased severity under humid and rainy weather
            - Premature deterioration of foliage in severe infection
            """
        )

        st.info(
            "Add your own approved rust symptom image as "
            "`assets/rust_symptom.jpg` in the GitHub repository."
        )

    else:

        st.subheader("Late Leaf Spot — Tikka Disease")

        st.write(
            """
            Late Leaf Spot produces circular dark lesions on groundnut
            leaves and may cause premature leaf loss when disease becomes
            severe.
            """
        )

        st.markdown("### Typical symptoms")
        st.markdown(
            """
            - Circular dark leaf spots
            - Yellowing around affected areas may occur
            - Progressive leaf damage
            - Premature defoliation under severe disease
            """
        )

        st.info(
            "Add your own approved LLS symptom image as "
            "`assets/lls_symptom.jpg` in the GitHub repository."
        )


# ============================================================
# IDM
# ============================================================

elif page == "IDM Practices":

    st.header("🛡️ Integrated Disease Management")

    disease = st.selectbox(
        "Select Disease",
        ["Rust", "Late Leaf Spot"],
    )

    st.subheader(f"{disease} — Recommended IDM Principles")

    st.markdown(
        """
        - Use locally recommended and suitable varieties.
        - Maintain good field sanitation.
        - Avoid unnecessary prolonged leaf wetness where practical.
        - Monitor disease development regularly.
        - Use weather-based warning information for timely scouting.
        - Apply fungicides only according to locally approved labels,
          extension recommendations and resistance-management guidance.
        - Follow recommended spray intervals and safety precautions.
        """
    )

    st.warning(
        "The forewarning system is a decision-support tool. "
        "It should support field scouting and local agricultural "
        "recommendations rather than replace them."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.header("ℹ️ About the System")

    st.markdown(
        """
        ### Groundnut Fungal Disease Forewarning System

        **Diseases**

        - Groundnut Rust (*Puccinia arachidis*)
        - Late Leaf Spot (Tikka disease)

        **District models**

        - Coimbatore
        - Cuddalore

        **Forecast horizons**

        - Current Week
        - Week 1
        - Week 2

        ### Model approach

        District-specific regression models are trained from historical
        weather and disease-score observations. The disease target is
        shifted by 0, 1 and 2 weeks to produce current, one-week-ahead
        and two-week-ahead models.

        Coimbatore and Cuddalore observations are not mixed during
        district-specific model training.
        """
    )

    st.success(
        "Model scope should always be reported with the district and "
        "season covered by the training data."
    )

    st.write("**Developer:** Kishor Kumar")
    st.write("**Year:** 2026")
