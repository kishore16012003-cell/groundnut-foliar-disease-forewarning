"""
Groundnut Foliar Disease Forewarning System
--------------------------------------------
Fixed MLR equation-based prediction system.

Farmer selects:
    District
    Disease
    Crop Stage
    Forecast Period
    Weather parameters

The application DOES NOT retrain an MLR model at runtime.
It uses the pre-developed fixed MLR equations below.

Weather variables:
    X1 = Maximum Temperature (°C)
    X2 = Minimum Temperature (°C)
    X3 = Morning RH (%)
    X4 = Evening RH (%)
    X5 = Wind Speed

Farmer-facing output:
    Predicted Severity (%)
    Severity Category
    Farmer Advisory

Current fixed equations available in this version:
    Aliyarnagar - Rust
    Aliyarnagar - Late Leaf Spot
    Vridhachalam - Rust
    Vridhachalam - Late Leaf Spot

IMPORTANT:
The percentage conversion assumes a 0-9 disease severity score:
    Severity (%) = Predicted score / 9 * 100

Verify the scoring scale before final research deployment.
"""

import urllib.parse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Groundnut Foliar Disease Forewarning",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.7rem;
    }

    .hero {
        padding: 50px 28px;
        border-radius: 22px;
        margin-bottom: 25px;
        background: linear-gradient(135deg, #17452a, #39734a);
        color: white;
        text-align: center;
    }

    .hero h1 {
        font-size: 40px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        margin: 0;
    }

    .result-card {
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #dddddd;
        text-align: center;
        margin: 15px 0;
    }

    .severity-number {
        font-size: 48px;
        font-weight: 700;
        margin: 8px 0;
    }

    .severity-category {
        font-size: 27px;
        font-weight: 700;
    }

    .stage-card {
        padding: 15px 18px;
        border-radius: 12px;
        background: #f3f7ff;
        border-left: 5px solid #4675bd;
        margin: 10px 0 20px 0;
    }

    .small-note {
        font-size: 13px;
        color: #666666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FIXED MLR EQUATIONS
#
# Each equation is:
# Y = intercept + b1*X1 + b2*X2 + b3*X3 + b4*X4 + b5*X5
#
# No Excel file is used for runtime prediction.
# ============================================================

MLR_MODELS = {

    "Aliyarnagar": {

        "Rust": {

            "Current Week": {
                "intercept": -35.0291,
                "coefficients": [
                    1.3226,   # X1 Max Temperature
                    0.0387,   # X2 Min Temperature
                    -0.0264,  # X3 Morning RH
                    0.0236,   # X4 Evening RH
                    -0.4046,  # X5 Wind Speed
                ],
                "r2": 0.0582,
            },

            "Week 1 Forecast": {
                "intercept": -21.5995,
                "coefficients": [
                    1.0124,
                    -0.1004,
                    -0.0393,
                    0.0296,
                    -0.2442,
                ],
                "r2": 0.0310,
            },

            "Week 2 Forecast": {
                "intercept": -5.4809,
                "coefficients": [
                    0.6547,
                    -0.2231,
                    -0.0624,
                    0.0295,
                    -0.1420,
                ],
                "r2": 0.0145,
            },
        },

        "Late Leaf Spot": {

            "Current Week": {
                "intercept": -88.94,
                "coefficients": [
                    3.39,
                    1.16,
                    0.368,
                    -0.0503,
                    -1.98,
                ],
                "r2": None,
            },

            "Week 1 Forecast": {
                "intercept": -69.84,
                "coefficients": [
                    2.86,
                    1.20,
                    0.294,
                    -0.0192,
                    0.213,
                ],
                "r2": None,
            },

            "Week 2 Forecast": {
                "intercept": -76.11,
                "coefficients": [
                    3.08,
                    1.12,
                    0.0873,
                    -0.0892,
                    1.89,
                ],
                "r2": None,
            },
        },
    },

    "Vridhachalam": {

        "Rust": {

            "Current Week": {
                "intercept": 5.7859,
                "coefficients": [
                    -0.3740,
                    0.3648,
                    0.0141,
                    -0.0088,
                    -0.4083,
                ],
                "r2": 0.5404,
            },

            "Week 1 Forecast": {
                "intercept": 3.1233,
                "coefficients": [
                    -0.2952,
                    0.3381,
                    0.0124,
                    0.0046,
                    -0.4336,
                ],
                "r2": 0.3871,
            },

            "Week 2 Forecast": {
                "intercept": 3.5604,
                "coefficients": [
                    -0.2651,
                    0.2589,
                    0.0021,
                    0.0232,
                    -0.3434,
                ],
                "r2": 0.3535,
            },
        },

        "Late Leaf Spot": {

            "Current Week": {
                "intercept": 5.2539,
                "coefficients": [
                    -0.5621,
                    0.5064,
                    0.0738,
                    0.0327,
                    -1.2936,
                ],
                "r2": 0.5140,
            },

            "Week 1 Forecast": {
                "intercept": -1.0280,
                "coefficients": [
                    -0.4247,
                    0.5228,
                    0.0834,
                    0.0449,
                    -1.3799,
                ],
                "r2": 0.4603,
            },

            "Week 2 Forecast": {
                "intercept": -0.2727,
                "coefficients": [
                    -0.3595,
                    0.4042,
                    0.0674,
                    0.0630,
                    -1.2077,
                ],
                "r2": 0.3937,
            },
        },
    },
}


# ============================================================
# CROP STAGES
# ============================================================

CROP_STAGES = [
    "Germination & Emergence",
    "Vegetative Growth",
    "Flowering",
    "Pegging",
    "Pod & Seed Filling",
    "Maturity & Harvest",
]

STAGE_DESCRIPTION = {
    "Germination & Emergence":
        "0–20 days: crop establishment and emergence.",
    "Vegetative Growth":
        "20–35 days: leaf and canopy development.",
    "Flowering":
        "30–40 days: flowering stage.",
    "Pegging":
        "35–60 days: pegging stage; disease monitoring becomes important.",
    "Pod & Seed Filling":
        "60–100 days: pod development and seed filling.",
    "Maturity & Harvest":
        "100–120+ days: maturity and harvesting period.",
}

# Weather-based foliar disease forewarning is activated
# from flowering onwards in this application.
ACTIVE_STAGES = {
    "Flowering",
    "Pegging",
    "Pod & Seed Filling",
    "Maturity & Harvest",
}


# ============================================================
# MLR CALCULATION
# ============================================================

FEATURE_NAMES = [
    "Maximum Temperature",
    "Minimum Temperature",
    "Morning RH",
    "Evening RH",
    "Wind Speed",
]


def calculate_mlr(model_info, x1, x2, x3, x4, x5):
    """Calculate Y using the fixed MLR equation."""
    intercept = model_info["intercept"]
    b1, b2, b3, b4, b5 = model_info["coefficients"]

    y = (
        intercept
        + b1 * x1
        + b2 * x2
        + b3 * x3
        + b4 * x4
        + b5 * x5
    )

    return float(y)


def equation_text(model_info):
    """Create a readable MLR equation."""
    variables = ["X1", "X2", "X3", "X4", "X5"]

    equation = f"Y = {model_info['intercept']:.4f}"

    for variable, coefficient in zip(
        variables,
        model_info["coefficients"],
    ):
        sign = "+" if coefficient >= 0 else "-"
        equation += (
            f" {sign} {abs(coefficient):.4f}{variable}"
        )

    return equation


# ============================================================
# SEVERITY CONVERSION
# ============================================================

# Disease score scale used for percentage conversion.
# Change ONLY if your validated field scoring scale is different.
MAX_DISEASE_SCORE = 9.0


def score_to_percentage(score):
    """
    Convert predicted disease score to percentage.

    Assumes a 0-9 disease severity scale.
    """
    score = max(0.0, min(float(score), MAX_DISEASE_SCORE))
    return (score / MAX_DISEASE_SCORE) * 100.0


def severity_category(percentage):
    """
    Farmer-friendly category based on predicted severity percentage.
    """
    if percentage < 20:
        return "Very Low"
    if percentage < 40:
        return "Low"
    if percentage < 60:
        return "Moderate"
    if percentage < 80:
        return "High"
    return "Severe"


def category_color(category):
    return {
        "Very Low": "green",
        "Low": "green",
        "Moderate": "orange",
        "High": "red",
        "Severe": "darkred",
    }[category]


# ============================================================
# FAVOURABLE WEATHER
# ============================================================

def weather_conditions(disease, x1, x2, x3, x4, rainfall):
    mean_temperature = (x1 + x2) / 2.0
    mean_rh = (x3 + x4) / 2.0

    if disease == "Rust":
        return {
            "Temperature 25–30°C":
                25 <= mean_temperature <= 30,
            "Relative humidity >85%":
                mean_rh > 85,
            "Rainfall / rainy conditions":
                rainfall > 0,
        }

    return {
        "Temperature 20–30°C":
            20 <= mean_temperature <= 30,
        "Relative humidity >90%":
            mean_rh > 90,
        "Rainfall / wet conditions":
            rainfall > 0,
    }


# ============================================================
# ADVISORY
# ============================================================

def get_advisory(
    disease,
    category,
    crop_stage,
    favourable_weather,
):
    if crop_stage not in ACTIVE_STAGES:
        return (
            "Disease forewarning is not activated at the selected "
            "crop stage. Continue regular crop monitoring."
        )

    if category == "Very Low":
        return (
            "Disease pressure is very low. Continue regular field "
            "monitoring."
        )

    if category == "Low":
        return (
            "Low disease pressure is indicated. Monitor the crop "
            "regularly, especially older leaves."
        )

    if category == "Moderate":
        if favourable_weather:
            return (
                "Moderate disease pressure is indicated and weather "
                "conditions are favourable. Increase field monitoring "
                "and follow locally recommended disease-management practices."
            )
        return (
            "Moderate disease pressure is indicated. Continue close "
            "field monitoring."
        )

    if category == "High":
        return (
            "High disease pressure is forecast. Inspect the crop closely "
            "and take timely disease-management action according to "
            "local agricultural recommendations and approved product labels."
        )

    return (
        "Severe disease pressure is forecast. Immediate field inspection "
        "and timely disease-management action are recommended according "
        "to local agricultural recommendations and approved product labels."
    )


# ============================================================
# PREDICTION FOR ALL THREE PERIODS
# ============================================================

def get_predictions(
    district,
    disease,
    x1,
    x2,
    x3,
    x4,
    x5,
):
    results = []

    for forecast in [
        "Current Week",
        "Week 1 Forecast",
        "Week 2 Forecast",
    ]:
        model_info = MLR_MODELS[district][disease][forecast]

        score = calculate_mlr(
            model_info,
            x1,
            x2,
            x3,
            x4,
            x5,
        )

        # The raw score is kept internal.
        percentage = score_to_percentage(score)
        category = severity_category(percentage)

        results.append(
            {
                "Forecast": forecast,
                "Predicted Severity (%)": round(
                    percentage,
                    1,
                ),
                "Severity Category": category,
            }
        )

    return pd.DataFrame(results)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Disease Prediction",
        "Disease Information",
        "About Developer",
    ],
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <h1>🌱 Groundnut Foliar Disease Forewarning</h1>
            <p>
                Fixed MLR Equation-Based Weather Forewarning System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        """
        This system provides weather-based forewarning for
        Groundnut Rust and Late Leaf Spot.

        The application uses pre-developed, district-specific
        Multiple Linear Regression (MLR) equations. Historical
        Excel data are not retrained during farmer prediction.
        """
    )

    st.info(
        "Farmer selects the current crop stage first, then enters "
        "weather information for prediction."
    )

    st.markdown("### Prediction workflow")

    st.write(
        """
        **District → Disease → Crop Stage → Forecast Period → "
        "Weather → Fixed MLR Equation → Predicted Severity (%) → "
        "Severity Category → Farmer Advisory**
        """
    )


# ============================================================
# DISEASE PREDICTION
# ============================================================

elif page == "Disease Prediction":

    st.header("🔮 Disease Forewarning")

    # --------------------------------------------------------
    # DISTRICT / DISEASE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox(
            "📍 Select District",
            list(MLR_MODELS.keys()),
        )

    with col2:
        disease = st.selectbox(
            "🦠 Select Disease",
            ["Rust", "Late Leaf Spot"],
        )

    # --------------------------------------------------------
    # CROP STAGE
    # --------------------------------------------------------

    st.subheader("🌱 Select Current Groundnut Crop Stage")

    crop_stage = st.radio(
        "Choose the stage that matches your field:",
        CROP_STAGES,
        horizontal=True,
    )

    st.markdown(
        f"""
        <div class="stage-card">
            <b>Selected crop stage:</b> {crop_stage}<br>
            {STAGE_DESCRIPTION[crop_stage]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast = st.selectbox(
        "🔮 Select Forecast Period",
        [
            "Current Week",
            "Week 1 Forecast",
            "Week 2 Forecast",
        ],
    )

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    st.subheader("🌦 Weather Parameters")

    c1, c2 = st.columns(2)

    with c1:
        x1 = st.number_input(
            "X1 — Maximum Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=30.0,
            step=0.1,
        )

        x2 = st.number_input(
            "X2 — Minimum Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=22.0,
            step=0.1,
        )

        x3 = st.number_input(
            "X3 — Morning RH (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
        )

    with c2:
        x4 = st.number_input(
            "X4 — Evening RH (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
        )

        x5 = st.number_input(
            "X5 — Wind Speed",
            min_value=0.0,
            value=5.0,
            step=0.1,
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=10.0,
            step=0.1,
        )

    phone = st.text_input(
        "Farmer WhatsApp Number (optional)",
        placeholder="e.g. 919876543210",
    )

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 PREDICT DISEASE",
        type="primary",
        use_container_width=True,
    ):

        # --------------------------------------------
        # CROP-STAGE FILTER
        # --------------------------------------------

        if crop_stage not in ACTIVE_STAGES:

            st.info(
                "🌱 Disease forewarning is not activated at the "
                "selected crop stage."
            )

            st.write(
                f"**Selected Crop Stage:** {crop_stage}"
            )

            st.write(
                "Continue regular crop monitoring. Weather-based "
                "foliar disease forewarning becomes active from "
                "the relevant crop growth stages."
            )

        else:

            # --------------------------------------------
            # FIXED MLR PREDICTION
            # --------------------------------------------

            model_info = MLR_MODELS[
                district
            ][
                disease
            ][
                forecast
            ]

            raw_score = calculate_mlr(
                model_info,
                x1,
                x2,
                x3,
                x4,
                x5,
            )

            percentage = score_to_percentage(
                raw_score
            )

            category = severity_category(
                percentage
            )

            color = category_color(
                category
            )

            # --------------------------------------------
            # WEATHER CONDITIONS
            # --------------------------------------------

            conditions = weather_conditions(
                disease,
                x1,
                x2,
                x3,
                x4,
                rainfall,
            )

            favourable_count = sum(
                conditions.values()
            )

            favourable_weather = (
                favourable_count >= 2
            )

            # --------------------------------------------
            # MAIN FARMER RESULT
            # --------------------------------------------

            st.markdown(
                f"""
                <div class="result-card">
                    <div>Predicted Severity</div>

                    <div class="severity-number"
                         style="color:{color};">
                        {percentage:.1f}%
                    </div>

                    <div>Severity Category</div>

                    <div class="severity-category"
                         style="color:{color};">
                        {category.upper()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --------------------------------------------
            # FORECAST SUMMARY
            # --------------------------------------------

            st.write(
                f"**District:** {district}  |  "
                f"**Disease:** {disease}  |  "
                f"**Forecast:** {forecast}  |  "
                f"**Crop Stage:** {crop_stage}"
            )

            # --------------------------------------------
            # WEATHER FAVOURABILITY
            # --------------------------------------------

            st.subheader(
                "🌦 Favourable Weather Conditions"
            )

            weather_cols = st.columns(
                len(conditions)
            )

            for col, (condition, present) in zip(
                weather_cols,
                conditions.items(),
            ):
                with col:
                    if present:
                        st.success(
                            "✓ " + condition
                        )
                    else:
                        st.warning(
                            "✗ " + condition
                        )

            # --------------------------------------------
            # GAUGE
            # --------------------------------------------

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    number={
                        "suffix": "%"
                    },
                    title={
                        "text":
                        "Predicted Severity"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "bar": {
                            "color": color
                        },
                        "steps": [
                            {
                                "range": [0, 20],
                                "color": "#d9f2d9",
                            },
                            {
                                "range": [20, 40],
                                "color": "#fff2cc",
                            },
                            {
                                "range": [40, 60],
                                "color": "#ffe0b2",
                            },
                            {
                                "range": [60, 80],
                                "color": "#f4b183",
                            },
                            {
                                "range": [80, 100],
                                "color": "#f4cccc",
                            },
                        ],
                    },
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            # --------------------------------------------
            # CURRENT / WEEK 1 / WEEK 2
            # --------------------------------------------

            st.subheader(
                "📊 Current Week, Week 1 & Week 2"
            )

            results_df = get_predictions(
                district,
                disease,
                x1,
                x2,
                x3,
                x4,
                x5,
            )

            # Farmer-friendly table only.
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            fig_trend = px.line(
                results_df,
                x="Forecast",
                y="Predicted Severity (%)",
                markers=True,
                range_y=[0, 100],
                title=(
                    f"{disease} Severity Forecast"
                ),
            )

            st.plotly_chart(
                fig_trend,
                use_container_width=True,
            )

            # --------------------------------------------
            # FARMER ADVISORY
            # --------------------------------------------

            st.subheader(
                "👨‍🌾 Farmer Advisory"
            )

            advisory = get_advisory(
                disease,
                category,
                crop_stage,
                favourable_weather,
            )

            if category in ["High", "Severe"]:
                st.warning(advisory)

            elif category == "Moderate":
                st.info(advisory)

            else:
                st.success(advisory)

            # --------------------------------------------
            # TECHNICAL MODEL DETAILS
            # Not shown in the main farmer result.
            # --------------------------------------------

            with st.expander(
                "Technical Model Details"
            ):

                st.write(
                    "**Fixed MLR equation used:**"
                )

                st.code(
                    equation_text(model_info),
                    language="text",
                )

                st.write(
                    "X1 = Maximum Temperature"
                )
                st.write(
                    "X2 = Minimum Temperature"
                )
                st.write(
                    "X3 = Morning RH"
                )
                st.write(
                    "X4 = Evening RH"
                )
                st.write(
                    "X5 = Wind Speed"
                )

                if model_info["r2"] is not None:
                    st.write(
                        f"Model R²: "
                        f"{model_info['r2']:.4f}"
                    )
                else:
                    st.write(
                        "R²: Not entered for this reference equation."
                    )

                st.caption(
                    "The application uses this fixed equation directly. "
                    "The historical Excel data are not retrained during "
                    "runtime prediction."
                )

            # --------------------------------------------
            # WHATSAPP
            # --------------------------------------------

            if (
                phone
                and category in ["High", "Severe"]
            ):

                message = f"""
🌱 Groundnut Foliar Disease Forewarning

District: {district}
Disease: {disease}
Crop Stage: {crop_stage}
Forecast: {forecast}

Predicted Severity: {percentage:.1f}%
Severity Category: {category}

Advisory:
{advisory}

- Groundnut Foliar Disease Forewarning System
"""

                encoded_message = urllib.parse.quote(
                    message
                )

                whatsapp_url = (
                    f"https://wa.me/{phone}"
                    f"?text={encoded_message}"
                )

                st.markdown(
                    f"""
                    <a href="{whatsapp_url}"
                       target="_blank">
                        <button style="
                            background-color:#25D366;
                            color:white;
                            padding:12px 22px;
                            border:none;
                            border-radius:8px;
                            font-size:16px;
                            font-weight:bold;
                            cursor:pointer;">
                            📲 Send WhatsApp Alert
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )


# ===================================================
# DISEASE INFORMATION
# ===================================================

if disease == "Rust":

    st.subheader("Groundnut Rust - Puccinia arachidis")

    st.write("""
    Rust is a fungal disease of groundnut that produces
    rust-coloured pustules on leaves. Disease development
    can increase under favourable humid and rainy conditions.
    """)

    st.image(
        "Rust.png",
        caption="Groundnut Rust Disease Symptoms",
        use_container_width=True
    )

    st.markdown("### Symptoms")

    st.markdown("""
    - Rust-coloured or reddish-brown pustules
    - Mainly visible on leaves
    - Increased disease development under humid conditions
    - Severe infection can damage foliage
    """)

else:

    st.subheader("Late Leaf Spot - Tikka Disease")

    st.write("""
    Late Leaf Spot produces circular dark lesions on
    groundnut leaves and may cause premature defoliation
    under severe disease pressure.
    """)

    st.image(
        "LLS.png",
        caption="Groundnut Late Leaf Spot Disease Symptoms",
        use_container_width=True
    )

    st.markdown("### Symptoms")

    st.markdown("""
    - Circular dark leaf spots
    - Lesions on older leaves
    - Progressive leaf damage
    - Premature defoliation under severe infection
    """)



# ============================================================
# ABOUT
# ============================================================

elif page == "About Developer":

    st.header("ℹ️ About the Developer")

    st.write(
        """
        **Developed by:** Kishor Kumar

        **Project:** Groundnut Foliar Disease Forewarning System

        **Year:** 2026
        """
    )

    st.markdown(
        """
        ### Model approach

        The application uses fixed district-specific Multiple Linear
        Regression equations developed from historical weather and
        disease observations.

        The production application does not retrain the MLR model
        when a farmer enters weather data.
        """
    )

    st.markdown(
        """
        ### Farmer-facing output

        The prediction page displays:

        - Predicted Severity (%)
        - Severity Category
        - Farmer Advisory
        """
    )

    st.caption(
        "Current fixed-equation models in this version: "
        "Aliyarnagar and Vridhachalam for Rust and Late Leaf Spot."
    )
