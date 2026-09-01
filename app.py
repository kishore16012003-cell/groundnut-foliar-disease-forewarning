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
# EXTRA STYLES  —  centred header title + Tamil font support
# ============================================================

EXTRA_CSS = """
<style>
/* Centre the TNAU title block inside the header bar */
.tnau-header .tnau-title-wrap { text-align: center; }
.tnau-header .tnau-title,
.tnau-header .tnau-subtitle { text-align: center; }

/* Tamil-capable font stack */
html, body, [data-testid="stAppViewContainer"],
.stMarkdown, .stButton button, .stRadio, .stSelectbox, label, p, div, span, h1, h2, h3, li {
    font-family: 'Segoe UI', 'Noto Sans Tamil', 'Latha', 'Nirmala UI',
                 'Inter', 'Helvetica Neue', system-ui, sans-serif;
}

/* Language switch block in the sidebar */
.lang-title {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--tnau-green-dark);
    margin: 2px 0 6px 2px;
}
section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    border-radius: 10px;
    font-weight: 700;
    font-size: 13px;
    padding: 6px 4px;
}
</style>
"""
st.markdown(EXTRA_CSS, unsafe_allow_html=True)


# ============================================================
# LANGUAGE  —  English / Tamil switch
# ============================================================

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"


def t(key: str) -> str:
    """Return the text for `key` in the currently selected language."""
    entry = TXT.get(key)
    if entry is None:
        return key
    return entry.get(st.session_state.get("lang", "en"), entry.get("en", key))


def tv(mapping: dict, value: str) -> str:
    """Translate a fixed option value (district / disease / stage / category)."""
    entry = mapping.get(value)
    if entry is None:
        return value
    return entry.get(st.session_state.get("lang", "en"), value)


# ------------------------------------------------------------
# Option labels (internal values stay English for the model)
# ------------------------------------------------------------

DISTRICT_LABEL = {
    "Aliyarnagar":  {"en": "Aliyarnagar",  "ta": "ஆலியார்நகர்"},
    "Vridhachalam": {"en": "Vridhachalam", "ta": "விருத்தாசலம்"},
}

DISEASE_LABEL = {
    "Rust":           {"en": "Rust",           "ta": "துரு நோய்"},
    "Late Leaf Spot": {"en": "Late Leaf Spot", "ta": "பிந்திய இலைப்புள்ளி நோய்"},
}

STAGE_LABEL = {
    "Germination & Emergence": {"en": "Germination & Emergence", "ta": "முளைப்பு மற்றும் தோன்றுதல்"},
    "Vegetative Growth":       {"en": "Vegetative Growth",       "ta": "தாவர வளர்ச்சி"},
    "Flowering":               {"en": "Flowering",               "ta": "பூக்கும் நிலை"},
    "Pegging":                 {"en": "Pegging",                 "ta": "ஊசி இறங்கும் நிலை"},
    "Pod & Seed Filling":      {"en": "Pod & Seed Filling",      "ta": "காய் மற்றும் விதை நிரம்பும் நிலை"},
    "Maturity & Harvest":      {"en": "Maturity & Harvest",      "ta": "முதிர்ச்சி மற்றும் அறுவடை"},
}

STAGE_DESC_L = {
    "Germination & Emergence": {
        "en": "0–20 days: crop establishment and emergence.",
        "ta": "0–20 நாட்கள்: பயிர் நிலைபெறுதல் மற்றும் முளைத்து வெளிவருதல்.",
    },
    "Vegetative Growth": {
        "en": "20–35 days: leaf and canopy development.",
        "ta": "20–35 நாட்கள்: இலை மற்றும் பசுங்கூரை வளர்ச்சி.",
    },
    "Flowering": {
        "en": "30–40 days: flowering stage.",
        "ta": "30–40 நாட்கள்: பூக்கும் நிலை.",
    },
    "Pegging": {
        "en": "35–60 days: pegging stage; disease monitoring becomes important.",
        "ta": "35–60 நாட்கள்: ஊசி இறங்கும் நிலை; நோய் கண்காணிப்பு முக்கியமாகிறது.",
    },
    "Pod & Seed Filling": {
        "en": "60–100 days: pod development and seed filling.",
        "ta": "60–100 நாட்கள்: காய் உருவாதல் மற்றும் விதை நிரம்புதல்.",
    },
    "Maturity & Harvest": {
        "en": "100–120+ days: maturity and harvesting period.",
        "ta": "100–120+ நாட்கள்: முதிர்ச்சி மற்றும் அறுவடைக் காலம்.",
    },
}

CATEGORY_LABEL = {
    "Very Low": {"en": "Very Low", "ta": "மிகக் குறைவு"},
    "Low":      {"en": "Low",      "ta": "குறைவு"},
    "Moderate": {"en": "Moderate", "ta": "மிதமானது"},
    "High":     {"en": "High",     "ta": "அதிகம்"},
    "Severe":   {"en": "Severe",   "ta": "மிகக் கடுமையானது"},
}

PAGE_LABEL = {
    "Home":                {"en": "Home",                "ta": "முகப்பு"},
    "Disease Prediction":  {"en": "Disease Prediction",  "ta": "நோய் முன்னறிவிப்பு"},
    "Disease Information": {"en": "Disease Information", "ta": "நோய் தகவல்"},
    "About Developer":     {"en": "About Developer",     "ta": "உருவாக்கியவர் பற்றி"},
}


# ------------------------------------------------------------
# All interface text  (English / Tamil)
# ------------------------------------------------------------

TXT = {
    # ---- header ----
    "app_title":     {"en": "TNAU Disease Prediction System",
                      "ta": "தமிழ்நாடு வேளாண் பல்கலைக்கழக நோய் முன்னறிவிப்பு அமைப்பு"},
    "app_sub_1":     {"en": "Tamil Nadu Agricultural University",
                      "ta": "தமிழ்நாடு வேளாண் பல்கலைக்கழகம்"},
    "app_sub_2":     {"en": "Groundnut Foliar Disease Forewarning",
                      "ta": "நிலக்கடலை இலை நோய் முன்னெச்சரிக்கை"},

    # ---- sidebar ----
    "language":      {"en": "Language", "ta": "மொழி"},
    "nav":           {"en": "Navigation", "ta": "வழிசெலுத்தல்"},
    "nav_sub":       {"en": "TNAU Disease Prediction System",
                      "ta": "TNAU நோய் முன்னறிவிப்பு அமைப்பு"},
    "sidebar_foot":  {"en": "TNAU · 2026", "ta": "TNAU · 2026"},

    # ---- home hero ----
    "home_hero_p":   {"en": "Weather-based forewarning for Groundnut Rust and Late Leaf Spot — empowering farmers with timely, actionable disease advisories.",
                      "ta": "நிலக்கடலை துரு மற்றும் பிந்திய இலைப்புள்ளி நோய்களுக்கான வானிலை அடிப்படையிலான முன்னெச்சரிக்கை — விவசாயிகளுக்கு சரியான நேரத்தில் செயல்படக்கூடிய நோய் ஆலோசனைகள்."},
    "chip_crop":     {"en": "Groundnut", "ta": "நிலக்கடலை"},
    "chip_weather":  {"en": "Weather-Driven", "ta": "வானிலை அடிப்படையிலானது"},
    "chip_district": {"en": "District-Specific", "ta": "மாவட்ட வாரியானது"},
    "chip_farmer":   {"en": "Farmer-Friendly", "ta": "விவசாயி நட்பு"},

    # ---- home stats ----
    "stat_diseases": {"en": "Diseases Covered", "ta": "உள்ளடக்கிய நோய்கள்"},
    "stat_district": {"en": "Districts Supported", "ta": "ஆதரிக்கப்படும் மாவட்டங்கள்"},
    "stat_weather":  {"en": "Weather Parameters", "ta": "வானிலை அளவுருக்கள்"},
    "stat_stages":   {"en": "Crop Stages Tracked", "ta": "கண்காணிக்கப்படும் பயிர் நிலைகள்"},

    # ---- home about ----
    "home_about_h":  {"en": "About the System", "ta": "அமைப்பு பற்றி"},
    "home_about_p1": {"en": "This system provides <b>weather-based forewarning</b> for two major foliar diseases of groundnut — <b>Rust</b> and <b>Late Leaf Spot</b>. By combining current weather parameters with district-specific knowledge, the application delivers a clear severity prediction and a practical farmer advisory that can help guide field-level decisions.",
                      "ta": "இந்த அமைப்பு நிலக்கடலையின் இரண்டு முக்கிய இலை நோய்களுக்கு — <b>துரு</b> மற்றும் <b>பிந்திய இலைப்புள்ளி</b> — <b>வானிலை அடிப்படையிலான முன்னெச்சரிக்கையை</b> வழங்குகிறது. தற்போதைய வானிலை அளவுருக்களை மாவட்ட வாரியான அறிவுடன் இணைத்து, தெளிவான தீவிர அளவு முன்னறிவிப்பையும், வயல் அளவிலான முடிவுகளுக்கு உதவும் நடைமுறை ஆலோசனையையும் வழங்குகிறது."},
    "home_about_p2": {"en": "The interface is intentionally simple: select your district, choose the disease of interest, indicate the current crop stage, enter the local weather values, and click <b>Predict Disease</b>. The system responds with the predicted severity, an easy-to-read category, and a recommended course of action.",
                      "ta": "இடைமுகம் வேண்டுமென்றே எளிமையாக வைக்கப்பட்டுள்ளது: உங்கள் மாவட்டத்தைத் தேர்ந்தெடுக்கவும், நோயைத் தேர்வு செய்யவும், தற்போதைய பயிர் நிலையைக் குறிப்பிடவும், உள்ளூர் வானிலை மதிப்புகளை உள்ளிடவும், பின்னர் <b>நோயை முன்னறிவி</b> என்பதை அழுத்தவும். அமைப்பு முன்னறிவிக்கப்பட்ட தீவிர அளவு, எளிதில் புரியும் வகைப்பாடு மற்றும் பரிந்துரைக்கப்பட்ட நடவடிக்கையை வழங்கும்."},

    # ---- home features ----
    "home_get_h":    {"en": "What You Get", "ta": "நீங்கள் பெறுவது"},
    "feat1_h":       {"en": "Predicted Severity", "ta": "முன்னறிவிக்கப்பட்ட தீவிரம்"},
    "feat1_p":       {"en": "A clear percentage and category that tells you how strong the disease pressure is expected to be.",
                      "ta": "நோய் அழுத்தம் எவ்வளவு வலுவாக இருக்கும் என்பதைக் கூறும் தெளிவான சதவீதமும் வகைப்பாடும்."},
    "feat2_h":       {"en": "Weather Favourability", "ta": "வானிலை சாதகத்தன்மை"},
    "feat2_p":       {"en": "See which current weather conditions are conducive to disease development in your field.",
                      "ta": "உங்கள் வயலில் நோய் பரவலுக்கு எந்த வானிலை நிலைமைகள் சாதகமாக உள்ளன என்பதைப் பாருங்கள்."},
    "feat3_h":       {"en": "Farmer Advisory", "ta": "விவசாயி ஆலோசனை"},
    "feat3_p":       {"en": "Receive a practical, locally-relevant recommendation on monitoring and disease management.",
                      "ta": "கண்காணிப்பு மற்றும் நோய் மேலாண்மை குறித்த நடைமுறைக்கு ஏற்ற உள்ளூர் பரிந்துரையைப் பெறுங்கள்."},

    # ---- home quick start ----
    "home_qs_h":     {"en": "Quick Start", "ta": "விரைவுத் தொடக்கம்"},
    "home_qs_title": {"en": "Three Simple Steps", "ta": "மூன்று எளிய படிகள்"},
    "step":          {"en": "STEP", "ta": "படி"},
    "step1_h":       {"en": "Select District & Disease", "ta": "மாவட்டம் மற்றும் நோயைத் தேர்ந்தெடுக்கவும்"},
    "step1_p":       {"en": "Choose your location and the disease you want to assess.",
                      "ta": "உங்கள் இடத்தையும், நீங்கள் மதிப்பிட விரும்பும் நோயையும் தேர்வு செய்யுங்கள்."},
    "step2_h":       {"en": "Enter Crop Stage & Weather", "ta": "பயிர் நிலை மற்றும் வானிலையை உள்ளிடவும்"},
    "step2_p":       {"en": "Pick the current growth stage and fill in the local weather values.",
                      "ta": "தற்போதைய வளர்ச்சி நிலையைத் தேர்வு செய்து, உள்ளூர் வானிலை மதிப்புகளை நிரப்புங்கள்."},
    "step3_h":       {"en": "Click Predict Disease", "ta": "நோயை முன்னறிவி என்பதை அழுத்தவும்"},
    "step3_p":       {"en": "Get your severity prediction, weather favourability, and advisory.",
                      "ta": "தீவிர அளவு முன்னறிவிப்பு, வானிலை சாதகத்தன்மை மற்றும் ஆலோசனையைப் பெறுங்கள்."},

    # ---- footers ----
    "foot_uni":      {"en": "Tamil Nadu Agricultural University · 2026",
                      "ta": "தமிழ்நாடு வேளாண் பல்கலைக்கழகம் · 2026"},
    "foot_validate": {"en": "Always validate field conditions with local agricultural officers.",
                      "ta": "வயல் நிலைமைகளை எப்போதும் உள்ளூர் வேளாண் அலுவலர்களிடம் உறுதிப்படுத்திக் கொள்ளவும்."},
    "foot_consult":  {"en": "Consult local agricultural officers for region-specific management practices.",
                      "ta": "பகுதிக்கேற்ற மேலாண்மை நடைமுறைகளுக்கு உள்ளூர் வேளாண் அலுவலர்களை அணுகவும்."},

    # ---- prediction page ----
    "pred_hero_p":   {"en": "Enter your field conditions and click <b>Predict Disease</b> to receive an instant forecast.",
                      "ta": "உங்கள் வயல் நிலைமைகளை உள்ளிட்டு <b>நோயை முன்னறிவி</b> என்பதை அழுத்தி உடனடி முன்னறிவிப்பைப் பெறுங்கள்."},
    "sec_location":  {"en": "Location & Disease", "ta": "இடம் மற்றும் நோய்"},
    "sel_district":  {"en": "Select District", "ta": "மாவட்டத்தைத் தேர்ந்தெடுக்கவும்"},
    "sel_disease":   {"en": "Select Disease", "ta": "நோயைத் தேர்ந்தெடுக்கவும்"},
    "sec_stage":     {"en": "Current Groundnut Crop Stage", "ta": "தற்போதைய நிலக்கடலை பயிர் நிலை"},
    "stage_help":    {"en": "Choose the stage that best matches your field.",
                      "ta": "உங்கள் வயலுக்கு மிகவும் பொருந்தும் நிலையைத் தேர்வு செய்யுங்கள்."},
    "stage_sel":     {"en": "Selected crop stage:", "ta": "தேர்ந்தெடுக்கப்பட்ட பயிர் நிலை:"},
    "stage_inactive":{"en": " ⚠️ Forewarning is active from the Flowering stage onwards.",
                      "ta": " ⚠️ முன்னெச்சரிக்கை பூக்கும் நிலையிலிருந்து செயல்படும்."},
    "stage_active":  {"en": " ✅ Forewarning is active at this stage.",
                      "ta": " ✅ இந்த நிலையில் முன்னெச்சரிக்கை செயல்படுகிறது."},
    "sec_weather":   {"en": "Weather Parameters", "ta": "வானிலை அளவுருக்கள்"},
    "weather_help":  {"en": "Enter the current weather values observed in your field.",
                      "ta": "உங்கள் வயலில் பதிவான தற்போதைய வானிலை மதிப்புகளை உள்ளிடவும்."},
    "in_maxtemp":    {"en": "🌡️ Maximum Temperature (°C)", "ta": "🌡️ அதிகபட்ச வெப்பநிலை (°செ)"},
    "in_mintemp":    {"en": "🌡️ Minimum Temperature (°C)", "ta": "🌡️ குறைந்தபட்ச வெப்பநிலை (°செ)"},
    "in_rh_morn":    {"en": "💧 Morning Relative Humidity (%)", "ta": "💧 காலை ஒப்பு ஈரப்பதம் (%)"},
    "in_rh_eve":     {"en": "💧 Evening Relative Humidity (%)", "ta": "💧 மாலை ஒப்பு ஈரப்பதம் (%)"},
    "in_wind":       {"en": "💨 Wind Speed (km/h)", "ta": "💨 காற்றின் வேகம் (கிமீ/மணி)"},
    "in_rain":       {"en": "🌧️ Rainfall (mm)", "ta": "🌧️ மழையளவு (மிமீ)"},
    "in_phone":      {"en": "📱 Farmer WhatsApp Number (optional — used only to send an alert for High/Severe predictions)",
                      "ta": "📱 விவசாயியின் வாட்ஸ்அப் எண் (விருப்பத்தேர்வு — அதிகம்/மிகக் கடுமையான முன்னறிவிப்புக்கு எச்சரிக்கை அனுப்ப மட்டுமே)"},
    "btn_predict":   {"en": "🔍 PREDICT DISEASE", "ta": "🔍 நோயை முன்னறிவி"},

    "analysing":     {"en": "ANALYSING FIELD CONDITIONS", "ta": "வயல் நிலைமைகள் ஆய்வு செய்யப்படுகின்றன"},
    "analysing_sub": {"en": "Combining weather data with district-specific disease knowledge…",
                      "ta": "வானிலைத் தரவு மாவட்ட வாரியான நோய் அறிவுடன் இணைக்கப்படுகிறது…"},

    "notactive_lbl": {"en": "Forewarning Not Active", "ta": "முன்னெச்சரிக்கை செயலில் இல்லை"},
    "notactive_h":   {"en": "Disease forewarning is not activated at this crop stage.",
                      "ta": "இந்தப் பயிர் நிலையில் நோய் முன்னெச்சரிக்கை செயல்படுத்தப்படவில்லை."},
    "notactive_p":   {"en": "Continue regular crop monitoring. Weather-based foliar disease forewarning becomes active from the <b>Flowering</b> stage onwards.",
                      "ta": "வழக்கமான பயிர் கண்காணிப்பைத் தொடரவும். வானிலை அடிப்படையிலான இலை நோய் முன்னெச்சரிக்கை <b>பூக்கும்</b> நிலையிலிருந்து செயல்படும்."},

    "pred_done":     {"en": "PREDICTION COMPLETE", "ta": "முன்னறிவிப்பு முடிந்தது"},
    "res_severity":  {"en": "Predicted Severity", "ta": "முன்னறிவிக்கப்பட்ட தீவிரம்"},
    "res_category":  {"en": "Severity Category", "ta": "தீவிர வகைப்பாடு"},
    "lbl_district":  {"en": "District", "ta": "மாவட்டம்"},
    "lbl_disease":   {"en": "Disease", "ta": "நோய்"},
    "lbl_stage":     {"en": "Crop Stage", "ta": "பயிர் நிலை"},

    "sec_fav":       {"en": "Favourable Weather Conditions", "ta": "சாதகமான வானிலை நிலைமைகள்"},
    "fav_help":      {"en": "Conditions that favour disease development in your field right now.",
                      "ta": "தற்போது உங்கள் வயலில் நோய் பரவலுக்கு சாதகமான நிலைமைகள்."},
    "cond_rust_temp":{"en": "Temperature 25–30°C", "ta": "வெப்பநிலை 25–30°செ"},
    "cond_rh85":     {"en": "Relative humidity >85%", "ta": "ஒப்பு ஈரப்பதம் >85%"},
    "cond_rain":     {"en": "Rainfall / rainy conditions", "ta": "மழை / மழைச் சூழல்"},
    "cond_lls_temp": {"en": "Temperature 20–30°C", "ta": "வெப்பநிலை 20–30°செ"},
    "cond_rh90":     {"en": "Relative humidity >90%", "ta": "ஒப்பு ஈரப்பதம் >90%"},
    "cond_wet":      {"en": "Rainfall / wet conditions", "ta": "மழை / ஈரமான சூழல்"},

    "sec_gauge":     {"en": "Severity Gauge", "ta": "தீவிர அளவுகோல்"},
    "gauge_live":    {"en": "Live severity reading from the prediction engine",
                      "ta": "முன்னறிவிப்பு அமைப்பிலிருந்து நேரடி தீவிர அளவீடு"},
    "sec_advisory":  {"en": "Farmer Advisory", "ta": "விவசாயி ஆலோசனை"},
    "btn_whatsapp":  {"en": "📲 Send WhatsApp Alert", "ta": "📲 வாட்ஸ்அப் எச்சரிக்கை அனுப்பு"},

    # ---- advisories ----
    "adv_inactive":  {"en": "Disease forewarning is not activated at the selected crop stage. Continue regular crop monitoring.",
                      "ta": "தேர்ந்தெடுக்கப்பட்ட பயிர் நிலையில் நோய் முன்னெச்சரிக்கை செயல்படுத்தப்படவில்லை. வழக்கமான பயிர் கண்காணிப்பைத் தொடரவும்."},
    "adv_verylow":   {"en": "Disease pressure is very low. Continue regular field monitoring.",
                      "ta": "நோய் அழுத்தம் மிகக் குறைவாக உள்ளது. வழக்கமான வயல் கண்காணிப்பைத் தொடரவும்."},
    "adv_low":       {"en": "Low disease pressure is indicated. Monitor the crop regularly, especially older leaves.",
                      "ta": "குறைந்த நோய் அழுத்தம் காணப்படுகிறது. குறிப்பாக பழைய இலைகளை வழக்கமாகக் கண்காணிக்கவும்."},
    "adv_mod_fav":   {"en": "Moderate disease pressure is indicated and weather conditions are favourable. Increase field monitoring and follow locally recommended disease-management practices.",
                      "ta": "மிதமான நோய் அழுத்தம் காணப்படுகிறது, வானிலையும் சாதகமாக உள்ளது. வயல் கண்காணிப்பை அதிகரித்து, உள்ளூரில் பரிந்துரைக்கப்பட்ட நோய் மேலாண்மை நடைமுறைகளைப் பின்பற்றவும்."},
    "adv_mod":       {"en": "Moderate disease pressure is indicated. Continue close field monitoring.",
                      "ta": "மிதமான நோய் அழுத்தம் காணப்படுகிறது. வயலை நெருக்கமாகக் கண்காணித்து வரவும்."},
    "adv_high":      {"en": "High disease pressure is forecast. Inspect the crop closely and take timely disease-management action according to local agricultural recommendations and approved product labels.",
                      "ta": "அதிக நோய் அழுத்தம் முன்னறிவிக்கப்பட்டுள்ளது. பயிரை நெருக்கமாக ஆய்வு செய்து, உள்ளூர் வேளாண் பரிந்துரைகள் மற்றும் அங்கீகரிக்கப்பட்ட மருந்து விவரங்களின்படி உரிய நேரத்தில் நோய் மேலாண்மை நடவடிக்கை எடுக்கவும்."},
    "adv_severe":    {"en": "Severe disease pressure is forecast. Immediate field inspection and timely disease-management action are recommended according to local agricultural recommendations and approved product labels.",
                      "ta": "மிகக் கடுமையான நோய் அழுத்தம் முன்னறிவிக்கப்பட்டுள்ளது. உடனடி வயல் ஆய்வும், உள்ளூர் வேளாண் பரிந்துரைகள் மற்றும் அங்கீகரிக்கப்பட்ட மருந்து விவரங்களின்படி உரிய நோய் மேலாண்மை நடவடிக்கையும் பரிந்துரைக்கப்படுகிறது."},

    # ---- disease information page ----
    "info_hero_p":   {"en": "Learn about the major foliar diseases of groundnut covered by this system.",
                      "ta": "இந்த அமைப்பு உள்ளடக்கிய நிலக்கடலையின் முக்கிய இலை நோய்கள் பற்றி அறியுங்கள்."},
    "rust_title":    {"en": "Groundnut Rust — Puccinia arachidis",
                      "ta": "நிலக்கடலை துரு நோய் — Puccinia arachidis"},
    "rust_p":        {"en": "Rust is a fungal disease of groundnut that produces rust-coloured pustules on leaves. Disease development can increase rapidly under favourable humid and rainy conditions, and severe infection may lead to significant defoliation and yield loss if not managed in time.",
                      "ta": "துரு என்பது நிலக்கடலையின் பூஞ்சை நோய் ஆகும்; இது இலைகளில் துரு நிற கொப்புளங்களை உண்டாக்குகிறது. ஈரப்பதம் மற்றும் மழைச் சூழலில் நோய் வேகமாகப் பரவும்; உரிய நேரத்தில் கட்டுப்படுத்தாவிட்டால் அதிக இலை உதிர்வு மற்றும் மகசூல் இழப்பு ஏற்படலாம்."},
    "rust_caption":  {"en": "Groundnut Rust Disease Symptoms", "ta": "நிலக்கடலை துரு நோய் அறிகுறிகள்"},
    "sec_symptoms":  {"en": "Symptoms", "ta": "அறிகுறிகள்"},
    "rust_s1":       {"en": "Rust-coloured or reddish-brown pustules on leaves",
                      "ta": "இலைகளில் துரு நிற அல்லது சிவப்பு-பழுப்பு கொப்புளங்கள்"},
    "rust_s2":       {"en": "Pustules mainly visible on the lower leaf surface",
                      "ta": "கொப்புளங்கள் பெரும்பாலும் இலையின் அடிப்பகுதியில் காணப்படும்"},
    "rust_s3":       {"en": "Increased disease development under humid conditions",
                      "ta": "ஈரப்பதமான சூழலில் நோய் அதிகமாகப் பரவுதல்"},
    "rust_s4":       {"en": "Severe infection can damage foliage and reduce yield",
                      "ta": "கடுமையான தாக்குதல் இலைகளைச் சேதப்படுத்தி மகசூலைக் குறைக்கும்"},
    "lls_title":     {"en": "Late Leaf Spot — Tikka Disease",
                      "ta": "பிந்திய இலைப்புள்ளி — திக்கா நோய்"},
    "lls_p":         {"en": "Late Leaf Spot produces circular dark lesions on groundnut leaves and may cause premature defoliation under severe disease pressure. It is one of the most destructive foliar diseases of groundnut and requires timely monitoring and management.",
                      "ta": "பிந்திய இலைப்புள்ளி நோய் நிலக்கடலை இலைகளில் வட்ட வடிவ கரும்புள்ளிகளை உண்டாக்கி, கடுமையான தாக்குதலின்போது இலைகள் முன்கூட்டியே உதிர வழிவகுக்கும். இது நிலக்கடலையின் மிகவும் சேதம் விளைவிக்கும் இலை நோய்களில் ஒன்று; உரிய நேரக் கண்காணிப்பும் மேலாண்மையும் தேவை."},
    "lls_caption":   {"en": "Groundnut Late Leaf Spot Disease Symptoms",
                      "ta": "நிலக்கடலை பிந்திய இலைப்புள்ளி நோய் அறிகுறிகள்"},
    "lls_s1":        {"en": "Circular dark leaf spots with characteristic halos",
                      "ta": "வட்ட வடிவ கரும்புள்ளிகள், சுற்றி வளையம் போன்ற அமைப்புடன்"},
    "lls_s2":        {"en": "Lesions typically appear on older leaves first",
                      "ta": "புள்ளிகள் பொதுவாக முதலில் பழைய இலைகளில் தோன்றும்"},
    "lls_s3":        {"en": "Progressive leaf damage as the disease advances",
                      "ta": "நோய் முற்றும்போது இலைச் சேதம் படிப்படியாக அதிகரிக்கும்"},
    "lls_s4":        {"en": "Premature defoliation under severe infection",
                      "ta": "கடுமையான தாக்குதலின்போது இலைகள் முன்கூட்டியே உதிர்தல்"},

    # ---- about developer page ----
    "about_hero_p":  {"en": "Meet the team behind the TNAU Disease Prediction System.",
                      "ta": "TNAU நோய் முன்னறிவிப்பு அமைப்பை உருவாக்கியவர்களைப் பற்றி அறியுங்கள்."},
    "sec_developer": {"en": "Developer", "ta": "உருவாக்கியவர்"},
    "lbl_devby":     {"en": "Developed by:", "ta": "உருவாக்கியவர்:"},
    "val_devname":   {"en": "Kishor Kumar", "ta": "கிஷோர் குமார்"},
    "lbl_project":   {"en": "Project:", "ta": "திட்டம்:"},
    "val_project":   {"en": "TNAU Disease Prediction System — Groundnut Foliar Disease Forewarning",
                      "ta": "TNAU நோய் முன்னறிவிப்பு அமைப்பு — நிலக்கடலை இலை நோய் முன்னெச்சரிக்கை"},
    "lbl_inst":      {"en": "Institution:", "ta": "நிறுவனம்:"},
    "val_inst":      {"en": "Tamil Nadu Agricultural University (TNAU)",
                      "ta": "தமிழ்நாடு வேளாண் பல்கலைக்கழகம் (TNAU)"},
    "lbl_year":      {"en": "Year:", "ta": "ஆண்டு:"},
    "sec_aboutsys":  {"en": "About the System", "ta": "அமைப்பு பற்றி"},
    "about_sys_p":   {"en": "The TNAU Disease Prediction System is a farmer-friendly decision-support tool that provides weather-based forewarning for two major foliar diseases of groundnut — <b>Rust</b> and <b>Late Leaf Spot</b>. By combining district-specific knowledge with current weather observations, the system delivers a clear severity prediction and a practical farmer advisory that can help guide timely field-level decisions.",
                      "ta": "TNAU நோய் முன்னறிவிப்பு அமைப்பு என்பது விவசாயி நட்பு முடிவெடுக்கும் துணைக் கருவியாகும். இது நிலக்கடலையின் இரண்டு முக்கிய இலை நோய்களுக்கு — <b>துரு</b> மற்றும் <b>பிந்திய இலைப்புள்ளி</b> — வானிலை அடிப்படையிலான முன்னெச்சரிக்கை வழங்குகிறது. மாவட்ட வாரியான அறிவை தற்போதைய வானிலைப் பதிவுகளுடன் இணைத்து, தெளிவான தீவிர அளவு முன்னறிவிப்பையும் நடைமுறை ஆலோசனையையும் தருகிறது."},
    "sec_output":    {"en": "Farmer-Facing Output", "ta": "விவசாயிக்கு வழங்கப்படும் தகவல்"},
    "output_p":      {"en": "The prediction page presents the following information in a clean, easy-to-understand format:",
                      "ta": "முன்னறிவிப்பு பக்கம் பின்வரும் தகவல்களை எளிதில் புரியும் வடிவில் வழங்குகிறது:"},
    "out_1":         {"en": "Predicted Severity (%)", "ta": "முன்னறிவிக்கப்பட்ட தீவிரம் (%)"},
    "out_2":         {"en": "Severity Category (Very Low / Low / Moderate / High / Severe)",
                      "ta": "தீவிர வகைப்பாடு (மிகக் குறைவு / குறைவு / மிதமானது / அதிகம் / மிகக் கடுமையானது)"},
    "out_3":         {"en": "Favourable Weather Conditions", "ta": "சாதகமான வானிலை நிலைமைகள்"},
    "out_4":         {"en": "Farmer Advisory", "ta": "விவசாயி ஆலோசனை"},
    "sec_coverage":  {"en": "Coverage", "ta": "உள்ளடக்கம்"},
    "coverage_p":    {"en": "The current version of the system covers the <b>Aliyarnagar</b> and <b>Vridhachalam</b> districts for both <b>Rust</b> and <b>Late Leaf Spot</b> diseases. Additional districts and diseases can be incorporated as further validated data become available.",
                      "ta": "தற்போதைய பதிப்பு <b>ஆலியார்நகர்</b> மற்றும் <b>விருத்தாசலம்</b> மாவட்டங்களில் <b>துரு</b> மற்றும் <b>பிந்திய இலைப்புள்ளி</b> ஆகிய இரு நோய்களையும் உள்ளடக்கியுள்ளது. மேலும் சரிபார்க்கப்பட்ட தரவுகள் கிடைக்கும்போது கூடுதல் மாவட்டங்களும் நோய்களும் சேர்க்கப்படும்."},
}


# ============================================================
# TNAU HEADER  —  logo + centred title bar shown on every page
# ============================================================

def render_header(active_page: str = "") -> None:
    """Render the green-and-white TNAU branded header bar (title centred)."""
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

    badge_text = tv(PAGE_LABEL, active_page) if active_page else tv(PAGE_LABEL, "Home")

    st.markdown(
        f"""
        <div class="tnau-header">
            {logo_html}
            <div class="tnau-title-wrap">
                <div class="tnau-title">{t('app_title')}</div>
                <div class="tnau-subtitle">
                    {t('app_sub_1')} &nbsp;|&nbsp; {t('app_sub_2')}
                </div>
            </div>
            <div class="tnau-badge">🌱 {badge_text}</div>
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
    """Return {text_key: bool} for the conditions favouring the disease."""
    mean_temperature = (x1 + x2) / 2.0
    mean_rh = (x3 + x4) / 2.0

    if disease == "Rust":
        return {
            "cond_rust_temp": 25 <= mean_temperature <= 30,
            "cond_rh85": mean_rh > 85,
            "cond_rain": rainfall > 0,
        }

    return {
        "cond_lls_temp": 20 <= mean_temperature <= 30,
        "cond_rh90": mean_rh > 90,
        "cond_wet": rainfall > 0,
    }


# ============================================================
# ADVISORY
# ============================================================

def get_advisory(disease, category, crop_stage, favourable_weather):
    if crop_stage not in ACTIVE_STAGES:
        return t("adv_inactive")

    if category == "Very Low":
        return t("adv_verylow")

    if category == "Low":
        return t("adv_low")

    if category == "Moderate":
        return t("adv_mod_fav") if favourable_weather else t("adv_mod")

    if category == "High":
        return t("adv_high")

    return t("adv_severe")


# ============================================================
# SIDEBAR  —  language switch (top) + navigation
# ============================================================

with st.sidebar:

    # ---------- LANGUAGE SWITCH (above navigation) ----------
    st.markdown(
        f'<div class="lang-title">🌐 {t("language")}</div>',
        unsafe_allow_html=True,
    )

    lang_c1, lang_c2 = st.columns(2)
    with lang_c1:
        if st.button(
            "English",
            key="btn_lang_en",
            use_container_width=True,
            type="primary" if st.session_state["lang"] == "en" else "secondary",
        ):
            st.session_state["lang"] = "en"
            st.rerun()
    with lang_c2:
        if st.button(
            "தமிழ்",
            key="btn_lang_ta",
            use_container_width=True,
            type="primary" if st.session_state["lang"] == "ta" else "secondary",
        ):
            st.session_state["lang"] = "ta"
            st.rerun()

    st.markdown("---")

    # ---------- NAVIGATION ----------
    st.markdown(
        f"""
        <div style="padding: 6px 4px 14px 4px;">
            <div style="font-size: 15px; font-weight: 800; color: var(--tnau-green-dark);">
                🌱 {t('nav')}
            </div>
            <div style="font-size: 12px; color: var(--tnau-text-muted); margin-top: 2px;">
                {t('nav_sub')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        t("nav"),
        [
            "Home",
            "Disease Prediction",
            "Disease Information",
            "About Developer",
        ],
        format_func=lambda v: tv(PAGE_LABEL, v),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("🌱  " + t("sidebar_foot"))


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    render_header("Home")

    st.markdown(
        f"""
        <div class="hero">
            <span class="leaf l1">🍃</span>
            <span class="leaf l2">🌿</span>
            <span class="leaf l3">🌱</span>
            <span class="leaf l4">🍃</span>
            <span class="leaf l5">🌿</span>
            <div class="hero-icon">🌱</div>
            <h1>{t('app_title')}</h1>
            <p>{t('home_hero_p')}</p>
            <div class="hero-chips">
                <span class="hero-chip">🌾 {t('chip_crop')}</span>
                <span class="hero-chip">🌦️ {t('chip_weather')}</span>
                <span class="hero-chip">📍 {t('chip_district')}</span>
                <span class="hero-chip">👨‍🌾 {t('chip_farmer')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Animated stats banner ----
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (s1, "🌱", "2", t("stat_diseases")),
        (s2, "📍", "2", t("stat_district")),
        (s3, "🌦️", "5", t("stat_weather")),
        (s4, "🌾", "6", t("stat_stages")),
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

    st.markdown("### " + t("home_about_h"))

    st.markdown(
        f"""
        <div class="section-card">
            <p style="font-size:15px; line-height:1.7; margin:0; color:var(--tnau-text);">
                {t('home_about_p1')}
            </p>
            <p style="font-size:15px; line-height:1.7; margin:14px 0 0 0; color:var(--tnau-text);">
                {t('home_about_p2')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### " + t("home_get_h"))

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon" style="animation: pulse 2.5s ease-in-out infinite;">📊</div>
                <h3>{t('feat1_h')}</h3>
                <p>{t('feat1_p')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon" style="animation: bounce 3s ease-in-out infinite; animation-delay: 0.3s;">🌦️</div>
                <h3>{t('feat2_h')}</h3>
                <p>{t('feat2_p')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon" style="animation: sway 3.5s ease-in-out infinite; animation-delay: 0.6s;">👨‍🌾</div>
                <h3>{t('feat3_h')}</h3>
                <p>{t('feat3_p')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### " + t("home_qs_h"))

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{t('home_qs_title')}</div>
            <div style="display:flex; gap:14px; margin-top:14px; flex-wrap:wrap;">
                <div style="flex:1; min-width:200px; padding:14px 16px; background:var(--tnau-green-bg); border-radius:12px; border-left:4px solid var(--tnau-green-light);">
                    <div style="font-size:13px; color:var(--tnau-text-muted); font-weight:600;">{t('step')} 1</div>
                    <div style="font-size:15px; color:var(--tnau-green-dark); font-weight:700; margin-top:4px;">{t('step1_h')}</div>
                    <div style="font-size:13px; color:var(--tnau-text-muted); margin-top:4px;">{t('step1_p')}</div>
                </div>
                <div style="flex:1; min-width:200px; padding:14px 16px; background:var(--tnau-green-bg); border-radius:12px; border-left:4px solid var(--tnau-green-light);">
                    <div style="font-size:13px; color:var(--tnau-text-muted); font-weight:600;">{t('step')} 2</div>
                    <div style="font-size:15px; color:var(--tnau-green-dark); font-weight:700; margin-top:4px;">{t('step2_h')}</div>
                    <div style="font-size:13px; color:var(--tnau-text-muted); margin-top:4px;">{t('step2_p')}</div>
                </div>
                <div style="flex:1; min-width:200px; padding:14px 16px; background:var(--tnau-green-bg); border-radius:12px; border-left:4px solid var(--tnau-green-light);">
                    <div style="font-size:13px; color:var(--tnau-text-muted); font-weight:600;">{t('step')} 3</div>
                    <div style="font-size:15px; color:var(--tnau-green-dark); font-weight:700; margin-top:4px;">{t('step3_h')}</div>
                    <div style="font-size:13px; color:var(--tnau-text-muted); margin-top:4px;">{t('step3_p')}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="app-footer">
            🌱 <b>{t('app_title')}</b> &nbsp;·&nbsp; {t('foot_uni')}
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
        f"""
        <div class="hero" style="padding: 32px 28px;">
            <span class="leaf l1">🍃</span>
            <span class="leaf l2">🌿</span>
            <span class="leaf l4">🌱</span>
            <div class="hero-icon" style="font-size:42px;">🔮</div>
            <h1 style="font-size:28px;">{tv(PAGE_LABEL, 'Disease Prediction')}</h1>
            <p>{t('pred_hero_p')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # DISTRICT / DISEASE
    # --------------------------------------------------------
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{t('sec_location')}</div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox(
            "📍 " + t("sel_district"),
            SUPPORTED_DISTRICTS,
            format_func=lambda v: tv(DISTRICT_LABEL, v),
        )

    with col2:
        disease = st.selectbox(
            "🦠 " + t("sel_disease"),
            ["Rust", "Late Leaf Spot"],
            format_func=lambda v: tv(DISEASE_LABEL, v),
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CROP STAGE
    # --------------------------------------------------------
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">🌱 {t('sec_stage')}</div>
            <div style="font-size:13px; color:var(--tnau-text-muted); margin-bottom:10px;">
                {t('stage_help')}
            </div>
        """,
        unsafe_allow_html=True,
    )

    crop_stage = st.radio(
        t("sec_stage"),
        CROP_STAGES,
        format_func=lambda v: tv(STAGE_LABEL, v),
        horizontal=True,
        label_visibility="collapsed",
    )

    stage_note = t("stage_active") if crop_stage in ACTIVE_STAGES else t("stage_inactive")

    st.markdown(
        f"""
        <div class="stage-card">
            <b>{t('stage_sel')}</b> {tv(STAGE_LABEL, crop_stage)}<br>
            <span style="font-size:13px; color:var(--tnau-text-muted);">
            {tv(STAGE_DESC_L, crop_stage)}{stage_note}
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
        f"""
        <div class="section-card">
            <div class="section-title">🌦️ {t('sec_weather')}</div>
            <div style="font-size:13px; color:var(--tnau-text-muted); margin-bottom:14px; display:flex; align-items:center; gap:10px;">
                <span class="rain-container">
                    <span class="rain-cloud">☁️</span>
                    <span class="rain-drop d1"></span>
                    <span class="rain-drop d2"></span>
                    <span class="rain-drop d3"></span>
                    <span class="rain-drop d4"></span>
                </span>
                {t('weather_help')}
            </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        x1 = st.number_input(
            t("in_maxtemp"),
            min_value=0.0,
            max_value=50.0,
            value=30.0,
            step=0.1,
        )

        x2 = st.number_input(
            t("in_mintemp"),
            min_value=0.0,
            max_value=50.0,
            value=22.0,
            step=0.1,
        )

        x3 = st.number_input(
            t("in_rh_morn"),
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
        )

    with c2:
        x4 = st.number_input(
            t("in_rh_eve"),
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
        )

        x5 = st.number_input(
            t("in_wind"),
            min_value=0.0,
            value=5.0,
            step=0.1,
        )

        rainfall = st.number_input(
            t("in_rain"),
            min_value=0.0,
            value=10.0,
            step=0.1,
        )

    phone = st.text_input(
        t("in_phone"),
        placeholder="e.g. 919876543210",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------
    predict_clicked = st.button(
        t("btn_predict"),
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:

        st.markdown(
            f"""
            <div class="section-card" style="text-align:center; padding:20px; position:relative; overflow:hidden;">
                <div style="display:inline-flex; align-items:center; gap:10px; font-size:15px; font-weight:700; color:var(--tnau-green-dark);">
                    <span class="live-dot"></span>
                    {t('analysing')}
                </div>
                <div style="margin-top:6px; font-size:12px; color:var(--tnau-text-muted);">
                    {t('analysing_sub')}
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
                f"""
                <div class="result-card" style="background:linear-gradient(135deg,#fff8e1 0%,#ffffff 70%); border-color:#ffe082;">
                    <div class="result-label">{t('notactive_lbl')}</div>
                    <div style="font-size:32px; margin:8px 0;">🌱</div>
                    <div style="font-size:18px; font-weight:700; color:#ef6c00;">
                        {t('notactive_h')}
                    </div>
                    <div class="result-summary">
                        <b>{t('lbl_stage')}:</b> {tv(STAGE_LABEL, crop_stage)}<br>
                        {t('notactive_p')}
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
            category_txt = tv(CATEGORY_LABEL, category)

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
                        <span style="font-size:11px; font-weight:700; color:var(--tnau-green-dark); letter-spacing:0.6px;">{t('pred_done')}</span>
                    </div>
                    <div class="result-label">{t('res_severity')}</div>
                    <div class="severity-number" style="color:{color};">
                        {percentage:.1f}%
                    </div>
                    <div class="result-label" style="margin-top:8px;">{t('res_category')}</div>
                    <div class="severity-category" style="color:{color};">
                        {emoji} {category_txt.upper()}
                    </div>
                    <div class="result-summary">
                        <b>{t('lbl_district')}:</b> {tv(DISTRICT_LABEL, district)} &nbsp;·&nbsp;
                        <b>{t('lbl_disease')}:</b> {tv(DISEASE_LABEL, disease)} &nbsp;·&nbsp;
                        <b>{t('lbl_stage')}:</b> {tv(STAGE_LABEL, crop_stage)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --------------------------------------------
            # WEATHER FAVOURABILITY
            # --------------------------------------------
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-title">🌦️ {t('sec_fav')}</div>
                    <div style="font-size:13px; color:var(--tnau-text-muted); margin-bottom:14px;">
                        {t('fav_help')}
                    </div>
                """,
                unsafe_allow_html=True,
            )

            weather_cols = st.columns(len(conditions))
            for col, (cond_key, present) in zip(weather_cols, conditions.items()):
                with col:
                    cls = "ok" if present else "no"
                    mark = "✓" if present else "✗"
                    st.markdown(
                        f"""
                        <div class="info-tile {cls}">
                            <div class="tile-icon">{mark}</div>
                            <div class="tile-text">{t(cond_key)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            # --------------------------------------------
            # GAUGE
            # --------------------------------------------
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-title">📊 {t('sec_gauge')}</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="gauge-wrap" style="padding:6px;">
                  <div style="text-align:center; font-size:13px; color:var(--tnau-text-muted); margin-bottom:6px;">
                    {t('gauge_live')}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    number={"suffix": "%"},
                    title={"text": t("res_severity")},
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
                font={"color": "#1b3a23",
                      "family": "Segoe UI, Noto Sans Tamil, Nirmala UI, sans-serif"},
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
                f"""
                <div class="section-card">
                    <div class="section-title">👨‍🌾 {t('sec_advisory')}</div>
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

                message = (
                    "🌱 " + t("app_title") + "\n\n"
                    + t("lbl_district") + ": " + tv(DISTRICT_LABEL, district) + "\n"
                    + t("lbl_disease") + ": " + tv(DISEASE_LABEL, disease) + "\n"
                    + t("lbl_stage") + ": " + tv(STAGE_LABEL, crop_stage) + "\n\n"
                    + t("res_severity") + f": {percentage:.1f}%\n"
                    + t("res_category") + ": " + category_txt + "\n\n"
                    + t("sec_advisory") + ":\n" + advisory + "\n\n"
                    + "- " + t("app_title")
                )
                encoded_message = urllib.parse.quote(message)
                whatsapp_url = f"https://wa.me/{phone}?text={encoded_message}"

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
                            {t('btn_whatsapp')}
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        f"""
        <div class="app-footer">
            🌱 <b>{t('app_title')}</b> &nbsp;·&nbsp; {t('foot_validate')}
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
        f"""
        <div class="hero" style="padding: 32px 28px;">
            <span class="leaf l1">🍃</span>
            <span class="leaf l3">🌱</span>
            <span class="leaf l4">🌿</span>
            <div class="hero-icon" style="font-size:42px;">🌱</div>
            <h1 style="font-size:28px;">{tv(PAGE_LABEL, 'Disease Information')}</h1>
            <p>{t('info_hero_p')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    disease = st.selectbox(
        t("sel_disease"),
        ["Rust", "Late Leaf Spot"],
        format_func=lambda v: tv(DISEASE_LABEL, v),
    )

    if disease == "Rust":

        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">{t('rust_title')}</div>
                <p style="font-size:15px; line-height:1.7; color:var(--tnau-text);">
                    {t('rust_p')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            "Rust.png",
            caption=t("rust_caption"),
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">{t('sec_symptoms')}</div>
                <ul style="font-size:15px; line-height:1.8; color:var(--tnau-text); padding-left:22px; margin:0;">
                    <li>{t('rust_s1')}</li>
                    <li>{t('rust_s2')}</li>
                    <li>{t('rust_s3')}</li>
                    <li>{t('rust_s4')}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">{t('lls_title')}</div>
                <p style="font-size:15px; line-height:1.7; color:var(--tnau-text);">
                    {t('lls_p')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            "LLS.png",
            caption=t("lls_caption"),
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">{t('sec_symptoms')}</div>
                <ul style="font-size:15px; line-height:1.8; color:var(--tnau-text); padding-left:22px; margin:0;">
                    <li>{t('lls_s1')}</li>
                    <li>{t('lls_s2')}</li>
                    <li>{t('lls_s3')}</li>
                    <li>{t('lls_s4')}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="app-footer">
            🌱 <b>{t('app_title')}</b> &nbsp;·&nbsp; {t('foot_consult')}
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
        f"""
        <div class="hero" style="padding: 32px 28px;">
            <span class="leaf l2">🌿</span>
            <span class="leaf l4">🍃</span>
            <div class="hero-icon" style="font-size:42px;">ℹ️</div>
            <h1 style="font-size:28px;">{tv(PAGE_LABEL, 'About Developer')}</h1>
            <p>{t('about_hero_p')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{t('sec_developer')}</div>
            <p style="font-size:15px; line-height:1.8; color:var(--tnau-text); margin:0;">
                <b style="color:var(--tnau-green-dark);">{t('lbl_devby')}</b> {t('val_devname')}<br>
                <b style="color:var(--tnau-green-dark);">{t('lbl_project')}</b> {t('val_project')}<br>
                <b style="color:var(--tnau-green-dark);">{t('lbl_inst')}</b> {t('val_inst')}<br>
                <b style="color:var(--tnau-green-dark);">{t('lbl_year')}</b> 2026
            </p>
        </div>

        <div class="section-card">
            <div class="section-title">{t('sec_aboutsys')}</div>
            <p style="font-size:15px; line-height:1.7; color:var(--tnau-text); margin:0;">
                {t('about_sys_p')}
            </p>
        </div>

        <div class="section-card">
            <div class="section-title">{t('sec_output')}</div>
            <p style="font-size:15px; line-height:1.7; color:var(--tnau-text); margin:0 0 10px 0;">
                {t('output_p')}
            </p>
            <ul style="font-size:15px; line-height:1.8; color:var(--tnau-text); padding-left:22px; margin:0;">
                <li>{t('out_1')}</li>
                <li>{t('out_2')}</li>
                <li>{t('out_3')}</li>
                <li>{t('out_4')}</li>
            </ul>
        </div>

        <div class="section-card">
            <div class="section-title">{t('sec_coverage')}</div>
            <p style="font-size:15px; line-height:1.7; color:var(--tnau-text); margin:0;">
                {t('coverage_p')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="app-footer">
            🌱 <b>{t('app_title')}</b> &nbsp;·&nbsp; {t('foot_uni')}
        </div>
        """,
        unsafe_allow_html=True,
    )
