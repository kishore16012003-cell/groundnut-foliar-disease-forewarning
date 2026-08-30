"""
================================================================
TNAU Disease Prediction System
================================================================
A modern, professional Streamlit application for weather-based
forewarning of Groundnut foliar diseases (Rust and Late Leaf Spot).

Developed for: Tamil Nadu Agricultural University (TNAU)
Project      : Groundnut Foliar Disease Forewarning
Developer    : Kishor Kumar
Year         : 2026

----------------------------------------------------------------
Backend note (kept internal — NOT surfaced in the UI):
----------------------------------------------------------------
The prediction logic uses pre-developed, district-specific
equations. The end-user simply sees a clean "Predict Disease"
experience; the underlying model details are intentionally
hidden from the farmer-facing interface.

Weather variables used internally:
    X1 = Maximum Temperature (deg C)
    X2 = Minimum Temperature (deg C)
    X3 = Morning RH (%)
    X4 = Evening RH (%)
    X5 = Wind Speed

The percentage conversion assumes a 0-9 disease severity score:
    Severity (%) = Predicted score / 9 * 100
================================================================
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
    page_title="TNAU Disease Prediction System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME / STYLES  —  Green & White, modern, professional
# ============================================================

CUSTOM_CSS = """
<style>
/* ----------  Design tokens  ---------- */
:root {
    --tnau-green-darkest: #0f3d1f;
    --tnau-green-dark:    #1b5e20;
    --tnau-green:         #2e7d32;
    --tnau-green-mid:     #388e3c;
    --tnau-green-light:   #43a047;
    --tnau-green-accent:  #66bb6a;
    --tnau-green-soft:    #a5d6a7;
    --tnau-green-bg:      #e8f5e9;
    --tnau-green-border:  #c8e6c9;
    --tnau-green-tint:    #f1f8f2;
    --tnau-text-dark:     #16311c;
    --tnau-text:          #2f4a36;
    --tnau-text-muted:    #5f6b63;
    --tnau-white:         #ffffff;
    --tnau-bg:            #f8fbf8;
    --tnau-shadow:        0 6px 22px rgba(27, 94, 32, 0.08);
    --tnau-shadow-hover:  0 14px 36px rgba(27, 94, 32, 0.16);
    --tnau-radius:        16px;
    --tnau-radius-sm:     10px;
}

/* ----------  Global / page background  ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--tnau-bg);
    color: var(--tnau-text);
    font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', system-ui, -apple-system, sans-serif;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* ----------  Top header bar  ---------- */
.tnau-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 14px 22px;
    margin-bottom: 22px;
    background: linear-gradient(135deg, #ffffff 0%, #f1f8f2 100%);
    border: 1px solid var(--tnau-green-border);
    border-radius: var(--tnau-radius);
    box-shadow: var(--tnau-shadow);
    animation: fadeInDown 0.6s ease-out;
}
.tnau-header img.tnau-logo {
    height: 64px;
    width: 64px;
    object-fit: contain;
    flex-shrink: 0;
    border-radius: 50%;
    background: #ffffff;
    padding: 4px;
    border: 2px solid var(--tnau-green-border);
    transition: transform 0.3s ease;
}
.tnau-header:hover img.tnau-logo { transform: rotate(6deg) scale(1.04); }
.tnau-header .tnau-title-wrap { flex: 1; min-width: 0; }
.tnau-header .tnau-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 0.2px;
    color: var(--tnau-green-darkest);
    line-height: 1.15;
    margin: 0;
}
.tnau-header .tnau-subtitle {
    font-size: 13px;
    color: var(--tnau-text-muted);
    margin-top: 4px;
    font-weight: 500;
}
.tnau-header .tnau-badge {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: var(--tnau-green-dark);
    background: var(--tnau-green-bg);
    border: 1px solid var(--tnau-green-border);
    padding: 6px 12px;
    border-radius: 999px;
    white-space: nowrap;
}

/* ----------  Hero  ---------- */
.hero {
    position: relative;
    padding: 54px 36px;
    border-radius: 24px;
    margin-bottom: 28px;
    background:
        radial-gradient(circle at 12% 18%, rgba(255,255,255,0.16) 0%, transparent 38%),
        radial-gradient(circle at 88% 82%, rgba(255,255,255,0.10) 0%, transparent 42%),
        linear-gradient(135deg, #0f3d1f 0%, #1b5e20 45%, #2e7d32 100%);
    color: #ffffff;
    text-align: center;
    overflow: hidden;
    box-shadow: 0 18px 40px rgba(15, 61, 31, 0.25);
    animation: fadeIn 0.7s ease-out;
}
.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(circle at 20% 80%, rgba(165, 214, 167, 0.15) 0%, transparent 28%);
    pointer-events: none;
}
.hero h1 {
    font-size: 38px;
    font-weight: 800;
    margin: 0 0 10px 0;
    letter-spacing: 0.3px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.18);
}
.hero p {
    font-size: 17px;
    margin: 0;
    color: #e8f5e9;
    font-weight: 400;
    opacity: 0.96;
}
.hero .hero-chips {
    margin-top: 22px;
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}
.hero .hero-chip {
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.28);
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 500;
    backdrop-filter: blur(4px);
}

/* ----------  Cards / sections  ---------- */
.section-card {
    background: #ffffff;
    border: 1px solid var(--tnau-green-border);
    border-radius: var(--tnau-radius);
    padding: 24px 26px;
    margin: 18px 0;
    box-shadow: var(--tnau-shadow);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.section-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--tnau-shadow-hover);
}

.section-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--tnau-green-dark);
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::before {
    content: "";
    width: 4px;
    height: 18px;
    background: linear-gradient(180deg, var(--tnau-green-light), var(--tnau-green-dark));
    border-radius: 2px;
    display: inline-block;
}

/* ----------  Stage card  ---------- */
.stage-card {
    padding: 16px 20px;
    border-radius: var(--tnau-radius-sm);
    background: linear-gradient(135deg, var(--tnau-green-bg) 0%, #ffffff 100%);
    border-left: 5px solid var(--tnau-green-light);
    margin: 12px 0 18px 0;
    color: var(--tnau-text-dark);
    transition: border-left-color 0.25s ease, transform 0.25s ease;
}
.stage-card:hover { transform: translateX(3px); border-left-color: var(--tnau-green-dark); }
.stage-card b { color: var(--tnau-green-dark); }

/* ----------  Result card  ---------- */
.result-card {
    padding: 30px 24px;
    border-radius: 20px;
    border: 1px solid var(--tnau-green-border);
    text-align: center;
    margin: 18px 0;
    background:
        radial-gradient(circle at 50% 0%, var(--tnau-green-bg) 0%, #ffffff 70%);
    box-shadow: var(--tnau-shadow);
    animation: popIn 0.55s cubic-bezier(0.2, 0.9, 0.3, 1.4);
}
.result-card .result-label {
    font-size: 14px;
    font-weight: 600;
    color: var(--tnau-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.severity-number {
    font-size: 56px;
    font-weight: 800;
    margin: 6px 0 4px 0;
    line-height: 1;
    text-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.severity-category {
    font-size: 22px;
    font-weight: 700;
    margin-top: 6px;
    letter-spacing: 0.6px;
}
.result-summary {
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px dashed var(--tnau-green-border);
    font-size: 14px;
    color: var(--tnau-text-muted);
}
.result-summary b { color: var(--tnau-green-dark); }

/* ----------  Info / advisory tiles  ---------- */
.info-tile {
    background: #ffffff;
    border: 1px solid var(--tnau-green-border);
    border-radius: var(--tnau-radius-sm);
    padding: 16px 18px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    height: 100%;
}
.info-tile:hover {
    transform: translateY(-3px);
    box-shadow: var(--tnau-shadow-hover);
}
.info-tile.ok  { background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 100%); border-color: var(--tnau-green-soft); }
.info-tile.no  { background: linear-gradient(135deg, #fff8e1 0%, #ffffff 100%); border-color: #ffe082; }
.info-tile .tile-icon { font-size: 22px; margin-bottom: 6px; }
.info-tile .tile-text { font-size: 13px; color: var(--tnau-text); font-weight: 500; line-height: 1.4; }

/* ----------  Feature cards (Home)  ---------- */
.feature-card {
    background: #ffffff;
    border: 1px solid var(--tnau-green-border);
    border-radius: var(--tnau-radius);
    padding: 22px 20px;
    text-align: center;
    height: 100%;
    transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;
    box-shadow: var(--tnau-shadow);
}
.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: var(--tnau-shadow-hover);
    border-color: var(--tnau-green-light);
}
.feature-card .feature-icon {
    width: 56px; height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--tnau-green-bg) 0%, var(--tnau-green-soft) 100%);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 14px auto;
    font-size: 26px;
    transition: transform 0.3s ease;
}
.feature-card:hover .feature-icon { transform: rotate(-8deg) scale(1.06); }
.feature-card h3 {
    font-size: 16px; font-weight: 700; color: var(--tnau-green-dark);
    margin: 0 0 8px 0;
}
.feature-card p {
    font-size: 13px; color: var(--tnau-text-muted);
    margin: 0; line-height: 1.55;
}

/* ----------  Buttons  ---------- */
.stButton > button {
    background: linear-gradient(135deg, var(--tnau-green) 0%, var(--tnau-green-dark) 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.4px;
    box-shadow: 0 6px 16px rgba(27, 94, 32, 0.28) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 24px rgba(27, 94, 32, 0.38) !important;
    filter: brightness(1.05) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ----------  Inputs / selectboxes  ---------- */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border-color: var(--tnau-green-border) !important;
    background: #ffffff !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--tnau-green-light) !important;
    box-shadow: 0 0 0 3px rgba(67, 160, 71, 0.16) !important;
}

/* ----------  Radio (horizontal stage selector)  ---------- */
.stRadio > div[role="radiogroup"] label {
    padding: 8px 14px !important;
    border-radius: 10px !important;
    border: 1px solid var(--tnau-green-border) !important;
    margin: 4px !important;
    transition: all 0.18s ease !important;
    background: #ffffff;
}
.stRadio > div[role="radiogroup"] label:hover {
    background: var(--tnau-green-bg) !important;
    border-color: var(--tnau-green-light) !important;
}
.stRadio > div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, var(--tnau-green-bg) 0%, #ffffff 100%) !important;
    border-color: var(--tnau-green) !important;
    box-shadow: 0 4px 12px rgba(27, 94, 32, 0.12);
}

/* ----------  Sidebar  ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, var(--tnau-green-tint) 100%);
    border-right: 1px solid var(--tnau-green-border);
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {
    border-radius: 10px;
    margin: 4px 0;
    transition: all 0.18s ease;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label:hover {
    background: var(--tnau-green-bg);
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, var(--tnau-green) 0%, var(--tnau-green-dark) 100%);
    color: #ffffff;
    border-color: var(--tnau-green-dark);
    box-shadow: 0 4px 12px rgba(27, 94, 32, 0.22);
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-checked="true"] span {
    color: #ffffff;
}

/* ----------  Streamlit native element tweaks  ---------- */
h1, h2, h3, h4 { color: var(--tnau-green-dark) !important; }
h2 { border-bottom: 2px solid var(--tnau-green-border); padding-bottom: 8px; }
.stAlert { border-radius: 12px !important; }
.stAlert > div { border-radius: 12px !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid var(--tnau-green-border); }
.stImage > img { border-radius: 12px; box-shadow: var(--tnau-shadow); }
hr { border-color: var(--tnau-green-border) !important; }

/* ----------  Footer  ---------- */
.app-footer {
    margin-top: 40px;
    padding: 20px 24px;
    text-align: center;
    background: linear-gradient(135deg, var(--tnau-green-tint) 0%, #ffffff 100%);
    border: 1px solid var(--tnau-green-border);
    border-radius: var(--tnau-radius);
    color: var(--tnau-text-muted);
    font-size: 13px;
}
.app-footer b { color: var(--tnau-green-dark); }

/* ----------  Animations  ---------- */
@keyframes fadeIn     { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeInDown { from { opacity: 0; transform: translateY(-12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes popIn      { from { opacity: 0; transform: scale(0.94); } to { opacity: 1; transform: scale(1); } }

/* ----------  Responsive  ---------- */
@media (max-width: 768px) {
    .hero { padding: 36px 18px; }
    .hero h1 { font-size: 26px; }
    .hero p { font-size: 14px; }
    .tnau-header { flex-wrap: wrap; }
    .tnau-header img.tnau-logo { height: 52px; width: 52px; }
    .tnau-header .tnau-title { font-size: 18px; }
    .tnau-header .tnau-badge { display: none; }
    .severity-number { font-size: 42px; }
    .block-container { padding-left: 14px; padding-right: 14px; }
}

/* hide streamlit's default "made with streamlit" footer & menu clutter if desired */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# TNAU HEADER  —  logo + title bar shown on every page
# ============================================================

def render_header(active_page: str = "") -> None:
    """Render the green-and-white TNAU branded header bar."""
    # Resolve logo path. Look next to this script first, then in /download,
    # finally fall back to nothing (the alt text shows).
    logo_candidates = [
        Path(__file__).parent / "tnau_logo.png",
        Path("tnau_logo.png"),
        Path("/home/z/my-project/download/tnau_logo.png"),
    ]
    logo_src = ""
    for cand in logo_candidates:
        if cand.exists():
            try:
                import base64
                data = base64.b64encode(cand.read_bytes()).decode()
                logo_src = f"data:image/png;base64,{data}"
            except Exception:
                logo_src = str(cand)
            break

    logo_html = (
        f'<img class="tnau-logo" src="{logo_src}" alt="TNAU Logo" />'
        if logo_src
        else '<div class="tnau-logo" style="display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--tnau-green-dark);">TNAU</div>'
    )

    badge = f"· {active_page}" if active_page else ""
    st.markdown(
        f"""
        <div class="tnau-header">
            {logo_html}
            <div class="tnau-title-wrap">
                <div class="tnau-title">TNAU Disease Prediction System</div>
                <div class="tnau-subtitle">
                    Tamil Nadu Agricultural University &nbsp;|&nbsp;
                    Groundnut Foliar Disease Forewarning
                </div>
            </div>
            <div class="tnau-badge">🌱 {active_page or 'Home'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BACKEND PREDICTION ENGINE
#
# Kept intact from the original application. The MLR terminology is
# intentionally NOT exposed anywhere in the UI — the user only sees a
# clean "Predict Disease" experience.
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

# Internal default prediction key used by the backend equation lookup.
# The end-user never sees this string — the UI presents a single
# "Predict Disease" experience with no period selection.
DEFAULT_FORECAST = "Current Week"


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
# PREDICTION CALCULATION
# ============================================================

FEATURE_NAMES = [
    "Maximum Temperature",
    "Minimum Temperature",
    "Morning RH",
    "Evening RH",
    "Wind Speed",
]


def calculate_mlr(model_info, x1, x2, x3, x4, x5):
    """Calculate the predicted severity score using the fixed equation."""
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


# ============================================================
# SEVERITY CONVERSION
# ============================================================

# Disease score scale used for percentage conversion.
MAX_DISEASE_SCORE = 9.0


def score_to_percentage(score):
    """Convert predicted disease score to percentage (0–9 scale)."""
    score = max(0.0, min(float(score), MAX_DISEASE_SCORE))
    return (score / MAX_DISEASE_SCORE) * 100.0


def severity_category(percentage):
    """Farmer-friendly category based on predicted severity percentage."""
    if percentage < 20:
        return "Very Low"
    if percentage < 40:
        return "Low"
    if percentage < 60:
        return "Moderate"
    if percentage < 80:
        return "High"
    return "Severe"


# Theme-aligned category colors (greens for low, ambers/reds for higher risk)
def category_color(category):
    return {
        "Very Low":  "#2e7d32",   # green
        "Low":       "#43a047",   # lighter green
        "Moderate":  "#f9a825",   # amber
        "High":      "#ef6c00",   # orange
        "Severe":    "#c62828",   # red
    }[category]


def category_emoji(category):
    return {
        "Very Low":  "🟢",
        "Low":       "🟢",
        "Moderate":  "🟡",
        "High":      "🟠",
        "Severe":    "🔴",
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

def get_advisory(disease, category, crop_stage, favourable_weather):
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
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 6px 4px 14px 4px;">
            <div style="font-size: 15px; font-weight: 800; color: var(--tnau-green-dark);">
                🌱 Navigation
            </div>
            <div style="font-size: 12px; color: var(--tnau-text-muted); margin-top: 2px;">
                TNAU Disease Prediction System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Go to",
        [
            "Home",
            "Disease Prediction",
            "Disease Information",
            "About Developer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("🌱  TNAU · 2026")


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    render_header("Home")

    st.markdown(
        """
        <div class="hero">
            <h1>🌱 TNAU Disease Prediction System</h1>
            <p>
                Weather-based forewarning for Groundnut Rust and Late Leaf Spot —
                empowering farmers with timely, actionable disease advisories.
            </p>
            <div class="hero-chips">
                <span class="hero-chip">🌾 Groundnut</span>
                <span class="hero-chip">🌦️ Weather-Driven</span>
                <span class="hero-chip">📍 District-Specific</span>
                <span class="hero-chip">👨‍🌾 Farmer-Friendly</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### About the System")

    st.markdown(
        """
        <div class="section-card">
            <p style="font-size:15px; line-height:1.7; margin:0; color:var(--tnau-text);">
                This system provides <b style="color:var(--tnau-green-dark);">weather-based
                forewarning</b> for two major foliar diseases of groundnut —
                <b>Rust</b> and <b>Late Leaf Spot</b>. By combining current
                weather parameters with district-specific knowledge, the
                application delivers a clear severity prediction and a
                practical farmer advisory that can help guide field-level
                decisions.
            </p>
            <p style="font-size:15px; line-height:1.7; margin:14px 0 0 0; color:var(--tnau-text);">
                The interface is intentionally simple: select your district,
                choose the disease of interest, indicate the current crop
                stage, enter the local weather values, and click
                <b style="color:var(--tnau-green-dark);">Predict Disease</b>.
                The system responds with the predicted severity, an
                easy-to-read category, and a recommended course of action.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What You Get")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Predicted Severity</h3>
                <p>A clear percentage and category that tells you how strong the disease pressure is expected to be.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌦️</div>
                <h3>Weather Favourability</h3>
                <p>See which current weather conditions are conducive to disease development in your field.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">👨‍🌾</div>
                <h3>Farmer Advisory</h3>
                <p>Receive a practical, locally-relevant recommendation on monitoring and disease management.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Quick Start")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Three Simple Steps</div>
            <div style="display:flex; gap:14px; margin-top:14px; flex-wrap:wrap;">
                <div style="flex:1; min-width:200px; padding:14px 16px; background:var(--tnau-green-bg); border-radius:12px; border-left:4px solid var(--tnau-green-light);">
                    <div style="font-size:13px; color:var(--tnau-text-muted); font-weight:600;">STEP 1</div>
                    <div style="font-size:15px; color:var(--tnau-green-dark); font-weight:700; margin-top:4px;">Select District & Disease</div>
                    <div style="font-size:13px; color:var(--tnau-text-muted); margin-top:4px;">Choose your location and the disease you want to assess.</div>
                </div>
                <div style="flex:1; min-width:200px; padding:14px 16px; background:var(--tnau-green-bg); border-radius:12px; border-left:4px solid var(--tnau-green-light);">
                    <div style="font-size:13px; color:var(--tnau-text-muted); font-weight:600;">STEP 2</div>
                    <div style="font-size:15px; color:var(--tnau-green-dark); font-weight:700; margin-top:4px;">Enter Crop Stage & Weather</div>
                    <div style="font-size:13px; color:var(--tnau-text-muted); margin-top:4px;">Pick the current growth stage and fill in the local weather values.</div>
                </div>
                <div style="flex:1; min-width:200px; padding:14px 16px; background:var(--tnau-green-bg); border-radius:12px; border-left:4px solid var(--tnau-green-light);">
                    <div style="font-size:13px; color:var(--tnau-text-muted); font-weight:600;">STEP 3</div>
                    <div style="font-size:15px; color:var(--tnau-green-dark); font-weight:700; margin-top:4px;">Click Predict Disease</div>
                    <div style="font-size:13px; color:var(--tnau-text-muted); margin-top:4px;">Get your severity prediction, weather favourability, and advisory.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-footer">
            🌱 <b>TNAU Disease Prediction System</b> &nbsp;·&nbsp;
            Tamil Nadu Agricultural University &nbsp;·&nbsp; 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DISEASE PREDICTION PAGE
# ============================================================

elif page == "Disease Prediction":

    render_header("Disease Prediction")

    st.markdown(
        """
        <div class="hero" style="padding: 32px 28px;">
            <h1 style="font-size:28px;">🔮 Disease Prediction</h1>
            <p>Enter your field conditions and click <b>Predict Disease</b> to receive an instant forecast.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # DISTRICT / DISEASE
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Location & Disease</div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CROP STAGE
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🌱 Current Groundnut Crop Stage</div>
            <div style="font-size:13px; color:var(--tnau-text-muted); margin-bottom:10px;">
                Choose the stage that best matches your field.
            </div>
        """,
        unsafe_allow_html=True,
    )

    crop_stage = st.radio(
        "Crop stage",
        CROP_STAGES,
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div class="stage-card">
            <b>Selected crop stage:</b> {crop_stage}<br>
            <span style="font-size:13px; color:var(--tnau-text-muted);">
            {STAGE_DESCRIPTION[crop_stage]}
            {" ⚠️ Forewarning is active from the Flowering stage onwards." if crop_stage not in ACTIVE_STAGES else " ✅ Forewarning is active at this stage."}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # WEATHER PARAMETERS
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🌦️ Weather Parameters</div>
            <div style="font-size:13px; color:var(--tnau-text-muted); margin-bottom:14px;">
                Enter the current weather values observed in your field.
            </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        x1 = st.number_input(
            "🌡️ Maximum Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=30.0,
            step=0.1,
        )

        x2 = st.number_input(
            "🌡️ Minimum Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=22.0,
            step=0.1,
        )

        x3 = st.number_input(
            "💧 Morning Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
        )

    with c2:
        x4 = st.number_input(
            "💧 Evening Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
        )

        x5 = st.number_input(
            "💨 Wind Speed (km/h)",
            min_value=0.0,
            value=5.0,
            step=0.1,
        )

        rainfall = st.number_input(
            "🌧️ Rainfall (mm)",
            min_value=0.0,
            value=10.0,
            step=0.1,
        )

    phone = st.text_input(
        "📱 Farmer WhatsApp Number (optional — used only to send an alert for High/Severe predictions)",
        placeholder="e.g. 919876543210",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------
    predict_clicked = st.button(
        "🔍 PREDICT DISEASE",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:

        # --------------------------------------------
        # CROP-STAGE FILTER
        # --------------------------------------------
        if crop_stage not in ACTIVE_STAGES:

            st.markdown(
                """
                <div class="result-card" style="background:linear-gradient(135deg,#fff8e1 0%,#ffffff 70%); border-color:#ffe082;">
                    <div class="result-label">Forewarning Not Active</div>
                    <div style="font-size:32px; margin:8px 0;">🌱</div>
                    <div style="font-size:18px; font-weight:700; color:#ef6c00;">
                        Disease forewarning is not activated at this crop stage.
                    </div>
                    <div class="result-summary">
                        <b>Selected Crop Stage:</b> """ + crop_stage + """<br>
                        Continue regular crop monitoring. Weather-based foliar disease
                        forewarning becomes active from the <b>Flowering</b> stage onwards.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            # --------------------------------------------
            # PREDICTION  (uses the internal default equation)
            # --------------------------------------------
            model_info = MLR_MODELS[district][disease][DEFAULT_FORECAST]

            raw_score = calculate_mlr(
                model_info, x1, x2, x3, x4, x5,
            )

            percentage = score_to_percentage(raw_score)
            category = severity_category(percentage)
            color = category_color(category)
            emoji = category_emoji(category)

            # --------------------------------------------
            # WEATHER CONDITIONS
            # --------------------------------------------
            conditions = weather_conditions(
                disease, x1, x2, x3, x4, rainfall,
            )
            favourable_count = sum(conditions.values())
            favourable_weather = favourable_count >= 2

            # --------------------------------------------
            # MAIN RESULT CARD
            # --------------------------------------------
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Severity</div>
                    <div class="severity-number" style="color:{color};">
                        {percentage:.1f}%
                    </div>
                    <div class="result-label" style="margin-top:8px;">Severity Category</div>
                    <div class="severity-category" style="color:{color};">
                        {emoji} {category.upper()}
                    </div>
                    <div class="result-summary">
                        <b>District:</b> {district} &nbsp;·&nbsp;
                        <b>Disease:</b> {disease} &nbsp;·&nbsp;
                        <b>Crop Stage:</b> {crop_stage}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --------------------------------------------
            # WEATHER FAVOURABILITY
            # --------------------------------------------
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">🌦️ Favourable Weather Conditions</div>
                    <div style="font-size:13px; color:var(--tnau-text-muted); margin-bottom:14px;">
                        Conditions that favour disease development in your field right now.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            weather_cols = st.columns(len(conditions))
            for col, (condition, present) in zip(weather_cols, conditions.items()):
                with col:
                    cls = "ok" if present else "no"
                    mark = "✓" if present else "✗"
                    st.markdown(
                        f"""
                        <div class="info-tile {cls}">
                            <div class="tile-icon">{mark}</div>
                            <div class="tile-text">{condition}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            # --------------------------------------------
            # GAUGE
            # --------------------------------------------
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">📊 Severity Gauge</div>
                """,
                unsafe_allow_html=True,
            )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    number={"suffix": "%"},
                    title={"text": "Predicted Severity"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 20],   "color": "#e8f5e9"},
                            {"range": [20, 40],  "color": "#dcedc8"},
                            {"range": [40, 60],  "color": "#fff9c4"},
                            {"range": [60, 80],  "color": "#ffe0b2"},
                            {"range": [80, 100], "color": "#ffcdd2"},
                        ],
                        "threshold": {
                            "line": {"color": color, "width": 4},
                            "thickness": 0.85,
                            "value": percentage,
                        },
                    },
                )
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#1b3a23", "family": "Segoe UI, sans-serif"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # --------------------------------------------
            # FARMER ADVISORY
            # --------------------------------------------
            advisory = get_advisory(
                disease, category, crop_stage, favourable_weather,
            )

            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">👨‍🌾 Farmer Advisory</div>
                """,
                unsafe_allow_html=True,
            )

            if category in ["High", "Severe"]:
                st.warning(advisory)
            elif category == "Moderate":
                st.info(advisory)
            else:
                st.success(advisory)

            st.markdown("</div>", unsafe_allow_html=True)

            # --------------------------------------------
            # WHATSAPP ALERT (High / Severe only)
            # --------------------------------------------
            if phone and category in ["High", "Severe"]:

                message = f"""
🌱 TNAU Disease Prediction System

District: {district}
Disease: {disease}
Crop Stage: {crop_stage}

Predicted Severity: {percentage:.1f}%
Severity Category: {category}

Advisory:
{advisory}

- TNAU Disease Prediction System
"""
                encoded_message = urllib.parse.quote(message)
                whatsapp_url = (
                    f"https://wa.me/{phone}?text={encoded_message}"
                )

                st.markdown(
                    f"""
                    <a href="{whatsapp_url}" target="_blank"
                       style="text-decoration:none; display:block;">
                        <button style="
                            background:linear-gradient(135deg,#25D366,#128C7E);
                            color:white;
                            padding:14px 24px;
                            border:none;
                            border-radius:12px;
                            font-size:15px;
                            font-weight:700;
                            cursor:pointer;
                            width:100%;
                            box-shadow:0 6px 16px rgba(37,211,102,0.32);
                            transition:transform 0.18s ease, box-shadow 0.18s ease;">
                            📲 Send WhatsApp Alert
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
        <div class="app-footer">
            🌱 <b>TNAU Disease Prediction System</b> &nbsp;·&nbsp;
            Always validate field conditions with local agricultural officers.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DISEASE INFORMATION PAGE
# ============================================================

elif page == "Disease Information":

    render_header("Disease Information")

    st.markdown(
        """
        <div class="hero" style="padding: 32px 28px;">
            <h1 style="font-size:28px;">🌱 Disease Information</h1>
            <p>Learn about the major foliar diseases of groundnut covered by this system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    disease = st.selectbox("Select Disease", ["Rust", "Late Leaf Spot"])

    if disease == "Rust":

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Groundnut Rust — Puccinia arachidis</div>
                <p style="font-size:15px; line-height:1.7; color:var(--tnau-text);">
                    Rust is a fungal disease of groundnut that produces
                    rust-coloured pustules on leaves. Disease development
                    can increase rapidly under favourable humid and rainy
                    conditions, and severe infection may lead to significant
                    defoliation and yield loss if not managed in time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            "Rust.png",
            caption="Groundnut Rust Disease Symptoms",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Symptoms</div>
                <ul style="font-size:15px; line-height:1.8; color:var(--tnau-text); padding-left:22px; margin:0;">
                    <li>Rust-coloured or reddish-brown pustules on leaves</li>
                    <li>Pustules mainly visible on the lower leaf surface</li>
                    <li>Increased disease development under humid conditions</li>
                    <li>Severe infection can damage foliage and reduce yield</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Late Leaf Spot — Tikka Disease</div>
                <p style="font-size:15px; line-height:1.7; color:var(--tnau-text);">
                    Late Leaf Spot produces circular dark lesions on
                    groundnut leaves and may cause premature defoliation
                    under severe disease pressure. It is one of the most
                    destructive foliar diseases of groundnut and requires
                    timely monitoring and management.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            "LLS.png",
            caption="Groundnut Late Leaf Spot Disease Symptoms",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Symptoms</div>
                <ul style="font-size:15px; line-height:1.8; color:var(--tnau-text); padding-left:22px; margin:0;">
                    <li>Circular dark leaf spots with characteristic halos</li>
                    <li>Lesions typically appear on older leaves first</li>
                    <li>Progressive leaf damage as the disease advances</li>
                    <li>Premature defoliation under severe infection</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="app-footer">
            🌱 <b>TNAU Disease Prediction System</b> &nbsp;·&nbsp;
            Consult local agricultural officers for region-specific management practices.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ABOUT DEVELOPER PAGE
# ============================================================

elif page == "About Developer":

    render_header("About Developer")

    st.markdown(
        """
        <div class="hero" style="padding: 32px 28px;">
            <h1 style="font-size:28px;">ℹ️ About the Developer</h1>
            <p>Meet the team behind the TNAU Disease Prediction System.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Developer</div>
            <p style="font-size:15px; line-height:1.8; color:var(--tnau-text); margin:0;">
                <b style="color:var(--tnau-green-dark);">Developed by:</b> Kishor Kumar<br>
                <b style="color:var(--tnau-green-dark);">Project:</b> TNAU Disease Prediction System —
                Groundnut Foliar Disease Forewarning<br>
                <b style="color:var(--tnau-green-dark);">Institution:</b> Tamil Nadu Agricultural University (TNAU)<br>
                <b style="color:var(--tnau-green-dark);">Year:</b> 2026
            </p>
        </div>

        <div class="section-card">
            <div class="section-title">About the System</div>
            <p style="font-size:15px; line-height:1.7; color:var(--tnau-text); margin:0;">
                The TNAU Disease Prediction System is a farmer-friendly
                decision-support tool that provides weather-based
                forewarning for two major foliar diseases of groundnut —
                <b>Rust</b> and <b>Late Leaf Spot</b>. By combining
                district-specific knowledge with current weather
                observations, the system delivers a clear severity
                prediction and a practical farmer advisory that can
                help guide timely field-level decisions.
            </p>
        </div>

        <div class="section-card">
            <div class="section-title">Farmer-Facing Output</div>
            <p style="font-size:15px; line-height:1.7; color:var(--tnau-text); margin:0 0 10px 0;">
                The prediction page presents the following information in a
                clean, easy-to-understand format:
            </p>
            <ul style="font-size:15px; line-height:1.8; color:var(--tnau-text); padding-left:22px; margin:0;">
                <li>Predicted Severity (%)</li>
                <li>Severity Category (Very Low / Low / Moderate / High / Severe)</li>
                <li>Favourable Weather Conditions</li>
                <li>Farmer Advisory</li>
            </ul>
        </div>

        <div class="section-card">
            <div class="section-title">Coverage</div>
            <p style="font-size:15px; line-height:1.7; color:var(--tnau-text); margin:0;">
                The current version of the system covers the
                <b>Aliyarnagar</b> and <b>Vridhachalam</b> districts for
                both <b>Rust</b> and <b>Late Leaf Spot</b> diseases.
                Additional districts and diseases can be incorporated as
                further validated data become available.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-footer">
            🌱 <b>TNAU Disease Prediction System</b> &nbsp;·&nbsp;
            Tamil Nadu Agricultural University &nbsp;·&nbsp; 2026
        </div>
        """,
        unsafe_allow_html=True,
    )
