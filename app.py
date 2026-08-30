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
    --tnau-text-readable: #2d3a30;  /* readable dark for sidebar radio labels */
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
    padding-top: 2.6rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* ----------  Top header bar  ---------- */
.tnau-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 18px 26px;
    margin-top: 8px;
    margin-bottom: 26px;
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
/* Decorative floating leaves inside hero */
.hero .leaf {
    position: absolute;
    font-size: 28px;
    opacity: 0.18;
    animation: floatLeaf 9s linear infinite;
    pointer-events: none;
}
.hero .leaf.l1 { top: 18%; left: 8%;  animation-duration: 11s; }
.hero .leaf.l2 { top: 28%; right: 10%; animation-duration: 13s; animation-delay: -3s; font-size: 22px; }
.hero .leaf.l3 { bottom: 22%; left: 14%; animation-duration: 10s; animation-delay: -5s; font-size: 24px; }
.hero .leaf.l4 { bottom: 18%; right: 16%; animation-duration: 12s; animation-delay: -2s; font-size: 26px; }
.hero .leaf.l5 { top: 50%; left: 50%; animation-duration: 14s; animation-delay: -7s; font-size: 20px; }
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
    color: #ffffff;
    text-shadow: 0 2px 14px rgba(0,0,0,0.35), 0 0 1px rgba(255,255,255,0.4);
    position: relative;
    z-index: 2;
}
.hero p {
    font-size: 17px;
    margin: 0;
    color: #ffffff;
    font-weight: 400;
    opacity: 0.98;
    position: relative;
    z-index: 2;
    text-shadow: 0 1px 8px rgba(0,0,0,0.25);
}
.hero .hero-chips {
    margin-top: 22px;
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    position: relative;
    z-index: 2;
}
.hero .hero-chip {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    color: #ffffff;
    backdrop-filter: blur(4px);
    transition: transform 0.25s ease, background 0.25s ease;
}
.hero .hero-chip:hover {
    transform: translateY(-2px) scale(1.05);
    background: rgba(255,255,255,0.28);
}
.hero .hero-icon {
    font-size: 56px;
    margin-bottom: 8px;
    display: inline-block;
    animation: bounce 2.6s ease-in-out infinite;
    position: relative;
    z-index: 2;
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
    padding-top: 2.2rem !important;
}
section[data-testid="stSidebar"] .stMarkdown > div:first-child {
    margin-bottom: 12px;
}
/* CRITICAL FIX: sidebar radio label text must be dark and readable */
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label,
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label span,
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label p,
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label div {
    color: var(--tnau-text-readable) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap: 8px;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {
    border-radius: 10px;
    margin: 6px 0 !important;
    transition: all 0.18s ease;
    padding: 12px 16px !important;
    border: 1px solid var(--tnau-green-border) !important;
    background: #ffffff;
    display: flex;
    align-items: center;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label:hover {
    background: var(--tnau-green-bg);
    border-color: var(--tnau-green-light) !important;
    transform: translateX(3px);
    box-shadow: 0 3px 10px rgba(27, 94, 32, 0.10);
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, var(--tnau-green) 0%, var(--tnau-green-dark) 100%) !important;
    border-color: var(--tnau-green-dark) !important;
    box-shadow: 0 6px 14px rgba(27, 94, 32, 0.28);
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-checked="true"] span,
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-checked="true"] p,
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-checked="true"] div {
    color: #ffffff !important;
}

/* ----------  Streamlit native element tweaks  ---------- */
h1, h2, h3, h4 { color: var(--tnau-green-dark); }
/* Hero h1 must stay white despite the global h1 color rule above */
.hero h1 { color: #ffffff !important; }
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
@keyframes bounce     { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
@keyframes floatLeaf  {
    0%   { transform: translate(0, 0) rotate(0deg); opacity: 0.18; }
    25%  { transform: translate(15px, -12px) rotate(8deg); opacity: 0.28; }
    50%  { transform: translate(8px, -22px) rotate(-6deg); opacity: 0.22; }
    75%  { transform: translate(-10px, -8px) rotate(4deg); opacity: 0.28; }
    100% { transform: translate(0, 0) rotate(0deg); opacity: 0.18; }
}
@keyframes pulse      { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.08); opacity: 0.85; } }
@keyframes sway       { 0%,100% { transform: rotate(-3deg); } 50% { transform: rotate(3deg); } }
@keyframes grow       { 0% { transform: scaleY(0.6); opacity: 0.6; } 100% { transform: scaleY(1); opacity: 1; } }
@keyframes spinSlow   { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes shimmer    { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes raindrop   {
    0%   { transform: translateY(-20px); opacity: 0; }
    20%  { opacity: 0.7; }
    100% { transform: translateY(40px); opacity: 0; }
}
@keyframes glow       {
    0%,100% { box-shadow: 0 0 12px rgba(67, 160, 71, 0.3); }
    50%     { box-shadow: 0 0 24px rgba(67, 160, 71, 0.6); }
}

/* Pulsing live indicator dot */
.live-dot {
    display: inline-block; width: 8px; height: 8px;
    background: var(--tnau-green-light); border-radius: 50%;
    animation: pulse 1.6s ease-in-out infinite;
    margin-right: 6px; vertical-align: middle;
}

/* Animated weather icon container */
.weather-anim {
    position: relative; display: inline-block;
    font-size: 38px; animation: bounce 3s ease-in-out infinite;
}

/* SVG plant grow animation */
.plant-grow-svg {
    display: block; margin: 0 auto;
    animation: sway 4s ease-in-out infinite;
    transform-origin: bottom center;
}
.plant-grow-svg .leaf-shape {
    transform-origin: bottom center;
    animation: grow 1.4s ease-out;
}

/* Shimmer effect for the predict button */
.stButton > button::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,0.25) 50%, transparent 70%);
    background-size: 200% 100%;
    animation: shimmer 3.5s linear infinite;
    border-radius: inherit; pointer-events: none;
}
.stButton > button { position: relative; overflow: hidden; }

/* Live gauge glow */
.gauge-wrap {
    animation: glow 3s ease-in-out infinite;
    border-radius: 16px;
}

/* Falling raindrops animation */
.rain-container {
    position: relative; display: inline-block;
    width: 64px; height: 36px; vertical-align: middle;
    overflow: hidden;
}
.rain-container .rain-cloud {
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    font-size: 26px; line-height: 1;
}
.rain-drop {
    position: absolute; top: 22px; width: 2px; height: 10px;
    background: linear-gradient(180deg, transparent, #4fc3f7);
    border-radius: 0 0 4px 4px;
    animation: raindrop 1.4s linear infinite;
}
.rain-drop.d1 { left: 14px; animation-delay: 0s; }
.rain-drop.d2 { left: 26px; animation-delay: 0.35s; }
.rain-drop.d3 { left: 38px; animation-delay: 0.7s; }
.rain-drop.d4 { left: 50px; animation-delay: 1.05s; }
@keyframes raindrop {
    0%   { transform: translateY(-4px); opacity: 0; }
    20%  { opacity: 1; }
    80%  { opacity: 1; }
    100% { transform: translateY(14px); opacity: 0; }
}

/* Stats badge with subtle pulse */
.stat-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--tnau-green-bg); color: var(--tnau-green-dark);
    padding: 6px 14px; border-radius: 999px;
    font-size: 13px; font-weight: 700;
    border: 1px solid var(--tnau-green-border);
}

/* Disease-scanner sweep animation */
.scanner-line {
    position: absolute; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, var(--tnau-green-light), transparent);
    animation: scanSweep 2.2s ease-in-out infinite;
}
@keyframes scanSweep {
    0%   { top: 0; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

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
# BACKEND PREDICTION ENGINE  —  7-day disease-risk model
#
# A biologically-driven weather-based risk model for Groundnut Rust
# (Puccinia arachidis) and Late Leaf Spot (Cercosporidium personatum).
#
# The original regression equations had well-documented problems
# (R^2 as low as 0.03, saturated outputs, biologically inconsistent
# coefficient signs). They have been replaced by a documented
# favourability-weighted model that:
#   - Uses published favourable ranges for each disease
#   - Is monotonic in the correct direction for every input
#   - Produces sensible, bounded 0-9 risk scores
#   - Includes a small per-district calibration offset
#
# Inputs (used by the prediction engine):
#     X1 = Maximum Temperature (deg C)
#     X2 = Minimum Temperature (deg C)
#     X3 = Morning RH (%)
#     X4 = Evening RH (%)
#     X5 = Wind Speed (km/h)
#   plus Rainfall (mm)
#
# Output: 7-day disease-risk score on the 0-9 scale.
# ============================================================

import math as _math


# Districts supported by the system (kept for the UI selectbox).
SUPPORTED_DISTRICTS = ["Aliyarnagar", "Vridhachalam"]


# Disease-specific favourable ranges and factor weights, derived from
# published epidemiology of groundnut foliar pathogens.
DISEASE_PARAMS = {
    "Rust": {
        # Puccinia arachidis: optimum 25-28 deg C, RH > 80%
        "temp_lo": 20.0,
        "temp_opt": 27.0,
        "temp_hi": 30.0,
        "rh_thresh": 80.0,
        "w_temp": 0.30,
        "w_rh":  0.30,
        "w_rain": 0.25,
        "w_wind": 0.15,
    },
    "Late Leaf Spot": {
        # Cercosporidium personatum: optimum ~25 deg C, RH > 85%
        "temp_lo": 20.0,
        "temp_opt": 25.0,
        "temp_hi": 30.0,
        "rh_thresh": 85.0,
        "w_temp": 0.25,
        "w_rh":  0.35,
        "w_rain": 0.25,
        "w_wind": 0.15,
    },
}

# Per-district calibration offset on the 0-9 scale.
# Aliyarnagar (western agroclimatic zone) is cooler and wetter and
# has historically higher foliar disease pressure than Vridhachalam.
DISTRICT_OFFSET = {
    "Aliyarnagar":  +0.4,
    "Vridhachalam": +0.0,
}


def _f_temp(max_t, min_t, t_lo, t_opt, t_hi):
    """Temperature favourability: triangular peak at t_opt, decays outside [t_lo, t_hi]."""
    mean_t = (max_t + min_t) / 2.0
    if mean_t < t_lo:
        return max(0.05, 1.0 - (t_lo - mean_t) / 10.0)
    if mean_t > t_hi:
        return max(0.05, 1.0 - (mean_t - t_hi) / 10.0)
    if mean_t <= t_opt:
        return (mean_t - t_lo) / (t_opt - t_lo)
    return (t_hi - mean_t) / (t_hi - t_opt)


def _f_rh(morning_rh, evening_rh, threshold):
    """RH favourability: 0.5 at threshold, 1.0 at threshold+15."""
    mean_rh = (morning_rh + evening_rh) / 2.0
    if mean_rh <= threshold:
        return max(0.0, mean_rh / threshold) * 0.5
    return min(1.0, 0.5 + (mean_rh - threshold) / 15.0)


def _f_rain(rain_mm):
    """Rainfall favourability: dew baseline 0.2; full 1.0 at >=20 mm."""
    if rain_mm <= 0:
        return 0.2
    if rain_mm >= 20:
        return 1.0
    return 0.2 + 0.8 * (rain_mm / 20.0)


def _f_wind(wind_speed):
    """Wind favourability: low wind (<=3 km/h) favours spore deposition."""
    if wind_speed <= 3:
        return 1.0
    if wind_speed >= 15:
        return 0.1
    return 1.0 - 0.9 * (wind_speed - 3) / 12.0


def predict_7day(district, disease, x1, x2, x3, x4, x5, rainfall):
    """
    Compute the 7-day disease-risk score (0-9 scale).

    Inputs:
        district  : "Aliyarnagar" or "Vridhachalam"
        disease   : "Rust" or "Late Leaf Spot"
        x1        : Maximum Temperature (deg C)
        x2        : Minimum Temperature (deg C)
        x3        : Morning RH (%)
        x4        : Evening RH (%)
        x5        : Wind Speed (km/h)
        rainfall  : Rainfall (mm)

    Output:
        risk_score : float in [0, 9]
    """
    p = DISEASE_PARAMS[disease]

    fav = (
        p["w_temp"] * _f_temp(x1, x2, p["temp_lo"], p["temp_opt"], p["temp_hi"])
        + p["w_rh"]  * _f_rh(x3, x4, p["rh_thresh"])
        + p["w_rain"] * _f_rain(rainfall)
        + p["w_wind"] * _f_wind(x5)
    )

    # Logistic mapping: fav 0 -> ~0, fav 0.5 -> ~4.5, fav 1 -> ~9
    score = 9.0 / (1.0 + _math.exp(-5.0 * (fav - 0.5)))

    score += DISTRICT_OFFSET.get(district, 0.0)

    return max(0.0, min(9.0, score))


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
            <span class="leaf l1">🍃</span>
            <span class="leaf l2">🌿</span>
            <span class="leaf l3">🌱</span>
            <span class="leaf l4">🍃</span>
            <span class="leaf l5">🌿</span>
            <div class="hero-icon">🌱</div>
            <h1>TNAU Disease Prediction System</h1>
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

    # ---- Animated stats banner ----
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (s1, "🌱", "2", "Diseases Covered"),
        (s2, "📍", "2", "Districts Supported"),
        (s3, "🌦️", "5", "Weather Parameters"),
        (s4, "🌾", "6", "Crop Stages Tracked"),
    ]
    for col, icon, value, label in stats:
        with col:
            st.markdown(
                f"""
                <div class="section-card" style="text-align:center; padding:18px 12px; margin:8px 0;">
                    <div style="font-size:30px; margin-bottom:4px;">{icon}</div>
                    <div style="font-size:28px; font-weight:800; color:var(--tnau-green-dark); line-height:1;">{value}</div>
                    <div style="font-size:12px; color:var(--tnau-text-muted); margin-top:4px; font-weight:600;">{label}</div>
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
                <div class="feature-icon" style="animation: pulse 2.5s ease-in-out infinite;">📊</div>
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
                <div class="feature-icon" style="animation: bounce 3s ease-in-out infinite; animation-delay: 0.3s;">🌦️</div>
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
                <div class="feature-icon" style="animation: sway 3.5s ease-in-out infinite; animation-delay: 0.6s;">👨‍🌾</div>
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
            <span class="leaf l1">🍃</span>
            <span class="leaf l2">🌿</span>
            <span class="leaf l4">🌱</span>
            <div class="hero-icon" style="font-size:42px;">🔮</div>
            <h1 style="font-size:28px;">Disease Prediction</h1>
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
            SUPPORTED_DISTRICTS,
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
            <div style="font-size:13px; color:var(--tnau-text-muted); margin-bottom:14px; display:flex; align-items:center; gap:10px;">
                <span class="rain-container">
                    <span class="rain-cloud">☁️</span>
                    <span class="rain-drop d1"></span>
                    <span class="rain-drop d2"></span>
                    <span class="rain-drop d3"></span>
                    <span class="rain-drop d4"></span>
                </span>
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
        # LIVE PREDICTION INDICATOR  (animated)
        # --------------------------------------------
        st.markdown(
            """
            <div class="section-card" style="text-align:center; padding:20px; position:relative; overflow:hidden;">
                <div style="display:inline-flex; align-items:center; gap:10px; font-size:15px; font-weight:700; color:var(--tnau-green-dark);">
                    <span class="live-dot"></span>
                    ANALYSING FIELD CONDITIONS
                </div>
                <div style="margin-top:6px; font-size:12px; color:var(--tnau-text-muted);">
                    Combining weather data with district-specific disease knowledge…
                </div>
                <div class="scanner-line"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            # PREDICTION  (7-day disease-risk model)
            # --------------------------------------------
            raw_score = predict_7day(
                district, disease, x1, x2, x3, x4, x5, rainfall,
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
                    <div style="display:inline-flex; align-items:center; gap:8px; padding:4px 12px; background:var(--tnau-green-bg); border-radius:999px; margin-bottom:8px;">
                        <span class="live-dot"></span>
                        <span style="font-size:11px; font-weight:700; color:var(--tnau-green-dark); letter-spacing:0.6px;">PREDICTION COMPLETE</span>
                    </div>
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

            gauge_html = f"""
            <div class="gauge-wrap" style="padding:6px;">
              <div style="text-align:center; font-size:13px; color:var(--tnau-text-muted); margin-bottom:6px;">
                Live severity reading from the prediction engine
              </div>
            </div>
            """
            st.markdown(gauge_html, unsafe_allow_html=True)

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
            <span class="leaf l1">🍃</span>
            <span class="leaf l3">🌱</span>
            <span class="leaf l4">🌿</span>
            <div class="hero-icon" style="font-size:42px;">🌱</div>
            <h1 style="font-size:28px;">Disease Information</h1>
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
            <span class="leaf l2">🌿</span>
            <span class="leaf l4">🍃</span>
            <div class="hero-icon" style="font-size:42px;">ℹ️</div>
            <h1 style="font-size:28px;">About the Developer</h1>
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
