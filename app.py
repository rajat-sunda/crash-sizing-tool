"""
BIW Crash Material Selection Tool  —  v3
==========================================
New in this version:
  1. Export Report  — download a PDF with charts + results summary
  2. Scatter Overlay — grid of standard grade × thickness combos,
     coloured green/red by pass/fail, overlaid on the boundary chart
  3. Multi-Scenario Comparison — lock in named scenarios and compare
     them in a table below the charts

Run with:
    streamlit run app.py
"""

import io
import math
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage,
                                 HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BIW Crash Material Selector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
.main .block-container {
    background-color: #f4f6f9 !important;
    color: #1a1f2e !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.app-header {
    background: linear-gradient(90deg, #1b3a6b 0%, #2a5298 60%, #1e6fbf 100%);
    border-bottom: 3px solid #1b3a6b;
    padding: 18px 32px 14px 32px;
    margin-bottom: 24px;
    border-radius: 8px;
}
.app-header h1 { margin:0; font-size:1.55rem; font-weight:700; letter-spacing:.04em; color:#fff; }
.app-header p  { margin:4px 0 0 0; font-size:.83rem; color:#c8daf7; letter-spacing:.02em; }
.metric-row { display:flex; gap:12px; margin-bottom:22px; flex-wrap:wrap; }
.metric-card {
    flex:1; background:#fff; border:1px solid #d0daea; border-top:3px solid #2a5298;
    border-radius:8px; padding:14px 18px; min-width:110px; box-shadow:0 1px 4px rgba(0,0,0,.07);
}
.metric-card .label { font-size:.68rem; color:#6b7a99; letter-spacing:.07em; text-transform:uppercase; margin-bottom:5px; }
.metric-card .value { font-size:1.4rem; font-weight:700; color:#1b3a6b; font-variant-numeric:tabular-nums; }
.metric-card .unit  { font-size:.70rem; color:#8896b0; margin-left:3px; }
.result-card {
    background:#fff; border:1px solid #d0daea; border-radius:8px;
    padding:18px 24px; margin-top:20px; box-shadow:0 1px 4px rgba(0,0,0,.07);
}
.result-card h4 {
    margin:0 0 12px 0; font-size:.78rem; color:#5a6a8a; letter-spacing:.07em;
    text-transform:uppercase; border-bottom:1px solid #e8ecf4; padding-bottom:8px;
}
.result-row { display:flex; gap:32px; flex-wrap:wrap; align-items:flex-start; }
.result-item .rlabel { font-size:.70rem; color:#6b7a99; letter-spacing:.04em; text-transform:uppercase; margin-bottom:2px; }
.result-item .rvalue { font-size:1.1rem; font-weight:600; color:#1a1f2e; }
.badge-pass { background:#e6f9ee; color:#1a7a3c; border:1px solid #5abf7e; padding:4px 14px; border-radius:5px; font-size:.85rem; font-weight:700; }
.badge-fail { background:#fdecea; color:#c0392b; border:1px solid #e07070; padding:4px 14px; border-radius:5px; font-size:.85rem; font-weight:700; }
.badge-ok   { background:#e8f0fe; color:#1b3a9e; border:1px solid #7096e8; padding:4px 14px; border-radius:5px; font-size:.85rem; font-weight:700; }
[data-testid="stSidebar"] { background-color:#ffffff !important; border-right:1px solid #d0daea; }
[data-testid="stSidebar"] * { color:#1a1f2e !important; }
.sidebar-title {
    font-size:.95rem; font-weight:700; color:#1b3a6b !important; letter-spacing:.04em;
    margin-bottom:8px; padding-bottom:6px; border-bottom:2px solid #2a5298;
}
footer {visibility:hidden;} #MainMenu {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>⚡ BIW Crash Material Selector</h1>
    <p>Concept-stage Body-in-White crash sizing tool &nbsp;·&nbsp; CCI = RM₁·L₁·t₁ + RM₂·L₂·t₂</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seg_len(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def compute_wall_lengths(pts):
    P1,P2,P3,P4 = pts["P1"],pts["P2"],pts["P3"],pts["P4"]
    P5,P6,P7,P8 = pts["P5"],pts["P6"],pts["P7"],pts["P8"]
    L1 = seg_len(P7,P5)+seg_len(P5,P1)+seg_len(P1,P4)+seg_len(P4,P6)+seg_len(P6,P8)
    L2 = seg_len(P7,P5)+seg_len(P5,P2)+seg_len(P2,P3)+seg_len(P3,P6)+seg_len(P6,P8)
    return L1, L2

# Standard grades and thickness steps used for scatter overlay & comparisons
STD_GRADES      = [420, 590, 780, 980, 1180, 1310]
STD_THICKNESSES = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]

# ---------------------------------------------------------------------------
# Session state for multi-scenario list
# ---------------------------------------------------------------------------
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []   # list of dicts

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🔧 Input Parameters</div>', unsafe_allow_html=True)

    default_pts = {
        "P1":(0.0,100.0),"P2":(100.0,100.0),
        "P3":(100.0,0.0),"P4":(0.0,0.0),
        "P5":(40.0,100.0),"P6":(40.0,0.0),
        "P7":(40.0,120.0),"P8":(40.0,-20.0),
    }
    with st.expander("📐 Geometry Points (mm)", expanded=False):
        pts = {}
        for key,(dx,dy) in default_pts.items():
            c1,c2 = st.columns(2)
            with c1: px = st.number_input(f"{key} x", value=dx, step=1.0, key=f"{key}_x", format="%.1f")
            with c2: py = st.number_input(f"{key} y", value=dy, step=1.0, key=f"{key}_y", format="%.1f")
            pts[key] = (px, py)

    with st.expander("💥 Crash Scenario", expanded=True):
        v1 = st.number_input("Initial speed v₁ (km/h)", value=56.0, min_value=1.0, step=1.0)
        v2 = st.number_input("Target speed v₂ (km/h)",  value=64.0, min_value=1.0, step=1.0)

    with st.expander("📋 Baseline Design", expanded=False):
        RM1_base = st.number_input("RM₁ baseline (MPa)", value=590.0, min_value=100.0, step=10.0)
        t1_base  = st.number_input("t₁ baseline (mm)",   value=1.4,   min_value=0.1,  step=0.05)
        RM2_base = st.number_input("RM₂ baseline (MPa)", value=780.0, min_value=100.0, step=10.0)
        t2_base  = st.number_input("t₂ baseline (mm)",   value=1.2,   min_value=0.1,  step=0.05)
        EA_base  = st.number_input("EA baseline (kJ)",   value=8.5,   min_value=0.01, step=0.1)

    st.markdown("---")

    # Mode toggle
    st.markdown("**🔀 Analysis Mode**")
    mode = st.radio(
        "Fix which material?",
        options=["Fix Material 1 → explore Mat 2", "Fix Material 2 → explore Mat 1"],
        index=0,
    )
    fixing_mat1 = mode.startswith("Fix Material 1")

    st.markdown("---")

    if fixing_mat1:
        st.markdown("**Material 1 — Fixed inputs**")
        RM1_sel = st.slider("RM₁ (MPa)", 400, 1400, 780, 10)
        t1_sel  = st.slider("t₁ (mm)",  0.8, 2.5, 1.4, 0.05)
        st.markdown("---")
        st.markdown("**Material 2 — Optional pin**")
        st.caption("Pin RM₂ to find minimum t₂.")
        fix_other = st.checkbox("Pin RM₂", value=False)
        RM2_pin   = st.slider("RM₂ (MPa)", 400, 1400, 980, 10, disabled=not fix_other)
        RM1_pin   = None; t2_sel = None
    else:
        st.markdown("**Material 2 — Fixed inputs**")
        RM2_sel = st.slider("RM₂ (MPa)", 400, 1400, 980, 10)
        t2_sel  = st.slider("t₂ (mm)",  0.8, 2.5, 1.2, 0.05)
        st.markdown("---")
        st.markdown("**Material 1 — Optional pin**")
        st.caption("Pin RM₁ to find minimum t₁.")
        fix_other = st.checkbox("Pin RM₁", value=False)
        RM1_pin   = st.slider("RM₁ (MPa)", 400, 1400, 780, 10, disabled=not fix_other)
        RM2_pin   = None; RM1_sel = None; t1_sel = None

    st.markdown("---")

    # ── Feature 2: Scatter overlay toggle ───────────────────────────────────
    st.markdown("**🔵 Scatter Overlay**")
    show_scatter = st.checkbox("Show all grade × thickness combos", value=False,
                               help="Plots every standard grade/thickness combo as a coloured dot (green=PASS, red=FAIL) on the boundary chart.")

# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------
L1, L2 = compute_wall_lengths(pts)
R            = (v2/v1)**2
CCI_baseline = RM1_base*L1*t1_base + RM2_base*L2*t2_base
CCI_target   = CCI_baseline * R
C            = EA_base/CCI_baseline if CCI_baseline > 0 else 0.0
EA_target    = C * CCI_target

if fixing_mat1:
    S_fixed   = RM1_sel * L1 * t1_sel
    Remaining = CCI_target - S_fixed
    mat_sufficient = Remaining <= 0
    t_pin_min = None
    if fix_other and not mat_sufficient and RM2_pin > 0 and L2 > 0:
        t_pin_min = Remaining / (RM2_pin * L2)
else:
    S_fixed   = RM2_sel * L2 * t2_sel
    Remaining = CCI_target - S_fixed
    mat_sufficient = Remaining <= 0
    t_pin_min = None
    if fix_other and not mat_sufficient and RM1_pin > 0 and L1 > 0:
        t_pin_min = Remaining / (RM1_pin * L1)

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card"><div class="label">Wall Length L₁</div>
        <div class="value">{L1:.1f}<span class="unit">mm</span></div></div>
    <div class="metric-card"><div class="label">Wall Length L₂</div>
        <div class="value">{L2:.1f}<span class="unit">mm</span></div></div>
    <div class="metric-card"><div class="label">Energy Ratio R</div>
        <div class="value">{R:.3f}</div></div>
    <div class="metric-card"><div class="label">CCI Baseline</div>
        <div class="value">{CCI_baseline/1000:.2f}<span class="unit">×10³</span></div></div>
    <div class="metric-card"><div class="label">CCI Target</div>
        <div class="value">{CCI_target/1000:.2f}<span class="unit">×10³</span></div></div>
    <div class="metric-card"><div class="label">Energy Target</div>
        <div class="value">{EA_target:.2f}<span class="unit">kJ</span></div></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
GRADES      = [590, 780, 980, 1180]
GRADE_COLOR = "rgba(80,110,180,0.45)"
T_RANGE     = np.linspace(0.8, 2.5, 300)
Y_MIN, Y_MAX = 350, 1450

BASE_LAYOUT = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafd",
    font=dict(color="#1a1f2e", size=11, family="Inter, Segoe UI, sans-serif"),
    margin=dict(l=60, r=70, t=50, b=50),
    showlegend=True,
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#d0daea",
                borderwidth=1, font=dict(size=9, color="#3a4a6a")),
)
AXIS_STYLE = dict(
    gridcolor="#dde4f0", zerolinecolor="#c8d0e0", linecolor="#c0cbdd",
    tickcolor="#8896b0", tickfont=dict(size=10, color="#3a4a6a"),
)

def add_grade_lines(fig):
    for g in GRADES:
        fig.add_hline(y=g, line=dict(color=GRADE_COLOR, dash="dash", width=1.2),
                      annotation_text=f"{g} MPa", annotation_position="right",
                      annotation_font=dict(size=9, color="#5a6a9a"))

def add_boundary_zones(fig, t_range, bnd_cl):
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_range, t_range[::-1]]),
        y=np.concatenate([bnd_cl, np.full(len(t_range), Y_MAX)]),
        fill="toself", fillcolor="rgba(34,139,34,0.10)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_range, t_range[::-1]]),
        y=np.concatenate([bnd_cl, np.full(len(t_range), Y_MIN)]),
        fill="toself", fillcolor="rgba(200,50,40,0.08)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))

def add_selected_dot(fig, x_val, y_val, label):
    fig.add_trace(go.Scatter(
        x=[x_val], y=[y_val], mode="markers",
        marker=dict(color="#0d8c4e", size=14, symbol="circle",
                    line=dict(color="#1b3a6b", width=2.5)),
        name=label,
        hovertemplate=f"{label}<extra></extra>",
    ))

def add_pin_marker(fig, t_val, rm_val, label):
    in_range = 0.8 <= t_val <= 2.5
    mc = "#0d8c4e" if in_range else "#c0392b"
    fig.add_trace(go.Scatter(
        x=[t_val], y=[rm_val], mode="markers+text",
        marker=dict(color=mc, size=14, symbol="diamond",
                    line=dict(color="#1b3a6b", width=2)),
        text=[f"  t_min={t_val:.2f}mm"], textposition="middle right",
        textfont=dict(size=10, color=mc),
        name=label,
        hovertemplate=f"t_min={t_val:.3f}mm<extra></extra>",
    ))

def add_scatter_overlay(fig, remaining, L_explore, axis="t2"):
    """
    Feature 2: plot all standard grade × thickness combinations.
    Each point is green (PASS: CCI_from_this_combo >= Remaining)
    or red (FAIL) on the boundary chart.
    remaining  — how much CCI the explore material must provide
    L_explore  — wall length of the explore material
    axis       — "t1" or "t2" (just for hover label)
    """
    pass_x, pass_y = [], []
    fail_x, fail_y = [], []
    for rm, t in itertools.product(STD_GRADES, STD_THICKNESSES):
        cci = rm * L_explore * t
        if cci >= remaining:
            pass_x.append(t); pass_y.append(rm)
        else:
            fail_x.append(t); fail_y.append(rm)
    lbl = "t₁" if axis == "t1" else "t₂"
    ax  = "RM₁" if axis == "t1" else "RM₂"
    if pass_x:
        fig.add_trace(go.Scatter(
            x=pass_x, y=pass_y, mode="markers",
            marker=dict(color="rgba(22,160,80,0.75)", size=9, symbol="circle",
                        line=dict(color="rgba(22,160,80,1)", width=1)),
            name="PASS combo", hovertemplate=f"{ax}=%{{y}} MPa  {lbl}=%{{x}} mm<extra>PASS</extra>",
        ))
    if fail_x:
        fig.add_trace(go.Scatter(
            x=fail_x, y=fail_y, mode="markers",
            marker=dict(color="rgba(210,50,40,0.65)", size=9, symbol="circle",
                        line=dict(color="rgba(210,50,40,1)", width=1)),
            name="FAIL combo", hovertemplate=f"{ax}=%{{y}} MPa  {lbl}=%{{x}} mm<extra>FAIL</extra>",
        ))

# ---------------------------------------------------------------------------
# Build Chart 1 (RM1 vs t1)
# ---------------------------------------------------------------------------
fig1 = go.Figure()
add_grade_lines(fig1)

if fixing_mat1:
    add_selected_dot(fig1, t1_sel, RM1_sel, f"RM₁={RM1_sel} MPa, t₁={t1_sel:.2f}mm")
    chart1_title = "Chart 1 — Material 1 (Fixed Selection)"
else:
    chart1_title = "Chart 1 — Material 1 Feasible Region"
    if mat_sufficient:
        fig1.add_annotation(x=1.65, y=900, text="✅ Material 2 alone<br>is sufficient",
            showarrow=False, font=dict(size=15, color="#1a7a3c"), align="center",
            bgcolor="rgba(220,255,235,0.85)", bordercolor="#5abf7e",
            borderwidth=1.5, borderpad=12)
    elif L1 > 0 and Remaining > 0:
        bnd    = Remaining / (L1 * T_RANGE)
        bnd_cl = np.clip(bnd, Y_MIN, Y_MAX)
        add_boundary_zones(fig1, T_RANGE, bnd_cl)
        # Feature 2 — scatter overlay on exploration chart
        if show_scatter:
            add_scatter_overlay(fig1, Remaining, L1, axis="t1")
        fig1.add_trace(go.Scatter(
            x=T_RANGE, y=bnd_cl, mode="lines",
            line=dict(color="#e05c1a", width=2.5), name="Min RM₁ boundary",
            hovertemplate="t₁:%{x:.2f}mm  Min RM₁:%{y:.0f}MPa<extra></extra>",
        ))
        if fix_other and t_pin_min is not None and t_pin_min > 0:
            add_pin_marker(fig1, t_pin_min, RM1_pin, f"t₁_min @ RM₁={RM1_pin}MPa")

fig1.update_layout(
    **BASE_LAYOUT,
    title=dict(text=chart1_title, font=dict(size=13, color="#1b3a6b"), x=0),
    xaxis=dict(**AXIS_STYLE, title="Thickness t₁ (mm)", range=[0.75, 2.55]),
    yaxis=dict(**AXIS_STYLE, title="Grade RM₁ (MPa)", range=[Y_MIN, Y_MAX]),
)

# ---------------------------------------------------------------------------
# Build Chart 2 (RM2 vs t2)
# ---------------------------------------------------------------------------
fig2 = go.Figure()
add_grade_lines(fig2)

if fixing_mat1:
    chart2_title = "Chart 2 — Material 2 Feasible Region"
    if mat_sufficient:
        fig2.add_annotation(x=1.65, y=900, text="✅ Material 1 alone<br>is sufficient",
            showarrow=False, font=dict(size=15, color="#1a7a3c"), align="center",
            bgcolor="rgba(220,255,235,0.85)", bordercolor="#5abf7e",
            borderwidth=1.5, borderpad=12)
    elif L2 > 0 and Remaining > 0:
        bnd    = Remaining / (L2 * T_RANGE)
        bnd_cl = np.clip(bnd, Y_MIN, Y_MAX)
        add_boundary_zones(fig2, T_RANGE, bnd_cl)
        # Feature 2 — scatter overlay on exploration chart
        if show_scatter:
            add_scatter_overlay(fig2, Remaining, L2, axis="t2")
        fig2.add_trace(go.Scatter(
            x=T_RANGE, y=bnd_cl, mode="lines",
            line=dict(color="#e05c1a", width=2.5), name="Min RM₂ boundary",
            hovertemplate="t₂:%{x:.2f}mm  Min RM₂:%{y:.0f}MPa<extra></extra>",
        ))
        if fix_other and t_pin_min is not None and t_pin_min > 0:
            add_pin_marker(fig2, t_pin_min, RM2_pin, f"t₂_min @ RM₂={RM2_pin}MPa")
else:
    chart2_title = "Chart 2 — Material 2 (Fixed Selection)"
    add_selected_dot(fig2, t2_sel, RM2_sel, f"RM₂={RM2_sel}MPa, t₂={t2_sel:.2f}mm")

fig2.update_layout(
    **BASE_LAYOUT,
    title=dict(text=chart2_title, font=dict(size=13, color="#1b3a6b"), x=0),
    xaxis=dict(**AXIS_STYLE, title="Thickness t₂ (mm)", range=[0.75, 2.55]),
    yaxis=dict(**AXIS_STYLE, title="Grade RM₂ (MPa)", range=[Y_MIN, Y_MAX]),
)

# ---------------------------------------------------------------------------
# Render charts
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
with col2:
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Results card
# ---------------------------------------------------------------------------
if mat_sufficient:
    fixed_name   = "Material 1" if fixing_mat1 else "Material 2"
    status_badge = f'<span class="badge-ok">✅ {fixed_name.upper()} ALONE SUFFICIENT</span>'
    status_note  = f"{fixed_name} contribution exceeds CCI target."
    t_min_display = "—"; remaining_display = "—"
elif fix_other and t_pin_min is not None:
    passes        = 0.8 <= t_pin_min <= 2.5
    pin_label     = f"RM₂={RM2_pin}" if fixing_mat1 else f"RM₁={RM1_pin}"
    status_badge  = ('<span class="badge-pass">✅ PASS</span>' if passes
                     else '<span class="badge-fail">❌ FAIL — t_min out of range</span>')
    status_note   = f"With {pin_label} MPa → t_min = {t_pin_min:.3f} mm"
    t_min_display = f"{t_pin_min:.3f} mm"
    remaining_display = f"{Remaining/1000:.3f} ×10³"
else:
    status_badge      = '<span style="color:#6b7a99;font-size:.85rem;">Pin the other material to compute t_min</span>'
    status_note       = ""
    t_min_display     = "—"
    remaining_display = "—" if mat_sufficient else f"{Remaining/1000:.3f} ×10³"

st.markdown(f"""
<div class="result-card">
    <h4>📊 Results Summary</h4>
    <div class="result-row">
        <div class="result-item"><div class="rlabel">Fixed Material CCI</div>
            <div class="rvalue">{S_fixed/1000:.3f} ×10³</div></div>
        <div class="result-item"><div class="rlabel">CCI Target</div>
            <div class="rvalue">{CCI_target/1000:.3f} ×10³</div></div>
        <div class="result-item"><div class="rlabel">Remaining CCI</div>
            <div class="rvalue">{remaining_display}</div></div>
        <div class="result-item"><div class="rlabel">t_min (pinned)</div>
            <div class="rvalue">{t_min_display}</div></div>
        <div class="result-item"><div class="rlabel">Energy Target</div>
            <div class="rvalue">{EA_target:.2f} kJ</div></div>
        <div class="result-item" style="margin-left:auto;align-self:center;text-align:right;">
            {status_badge}
            <div class="rlabel" style="margin-top:5px;">{status_note}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# FEATURE 3 — Multi-Scenario Comparison
# ===========================================================================
st.markdown("---")
st.markdown("### 📋 Multi-Scenario Comparison")

with st.expander("➕ Add current selection as a scenario", expanded=True):
    sc_name = st.text_input("Scenario name", value=f"Scenario {len(st.session_state.scenarios)+1}")
    if st.button("💾 Save scenario"):
        if fixing_mat1:
            sc = dict(
                name=sc_name, mode="Fix Mat1",
                RM1=RM1_sel, t1=t1_sel,
                RM2=RM2_pin if fix_other else "—",
                t2=f"{t_pin_min:.3f}" if (fix_other and t_pin_min is not None) else "—",
                S_fixed=S_fixed, Remaining=Remaining,
                CCI_target=CCI_target,
                t_min=t_pin_min if (fix_other and t_pin_min is not None) else None,
                passes=(0.8 <= t_pin_min <= 2.5) if (fix_other and t_pin_min is not None) else None,
                mat_sufficient=mat_sufficient,
            )
        else:
            sc = dict(
                name=sc_name, mode="Fix Mat2",
                RM1=RM1_pin if fix_other else "—",
                t1=f"{t_pin_min:.3f}" if (fix_other and t_pin_min is not None) else "—",
                RM2=RM2_sel, t2=t2_sel,
                S_fixed=S_fixed, Remaining=Remaining,
                CCI_target=CCI_target,
                t_min=t_pin_min if (fix_other and t_pin_min is not None) else None,
                passes=(0.8 <= t_pin_min <= 2.5) if (fix_other and t_pin_min is not None) else None,
                mat_sufficient=mat_sufficient,
            )
        st.session_state.scenarios.append(sc)
        st.success(f"'{sc_name}' saved!")

if st.session_state.scenarios:
    # Build display dataframe
    rows = []
    for s in st.session_state.scenarios:
        if s["mat_sufficient"]:
            verdict = "✅ Mat alone OK"
        elif s["passes"] is True:
            verdict = "✅ PASS"
        elif s["passes"] is False:
            verdict = "❌ FAIL"
        else:
            verdict = "— (no pin)"
        rows.append({
            "Name":       s["name"],
            "Mode":       s["mode"],
            "RM₁ (MPa)":  s["RM1"],
            "t₁ (mm)":    s["t1"],
            "RM₂ (MPa)":  s["RM2"],
            "t₂ (mm)":    s["t2"],
            "S_fixed ×10³": f"{s['S_fixed']/1000:.3f}",
            "Remaining ×10³": ("—" if s["mat_sufficient"]
                                else f"{s['Remaining']/1000:.3f}"),
            "t_min (mm)": f"{s['t_min']:.3f}" if s["t_min"] is not None else "—",
            "Verdict":    verdict,
        })
    df_sc = pd.DataFrame(rows)

    # Colour verdict column
    def colour_verdict(val):
        if "PASS" in val or "OK" in val:
            return "background-color:#e6f9ee; color:#1a7a3c; font-weight:700"
        if "FAIL" in val:
            return "background-color:#fdecea; color:#c0392b; font-weight:700"
        return ""

    styled = df_sc.style.applymap(colour_verdict, subset=["Verdict"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    col_dl, col_clr = st.columns([1,1])
    with col_clr:
        if st.button("🗑 Clear all scenarios"):
            st.session_state.scenarios = []
            st.rerun()

    # ── Download as CSV ──────────────────────────────────────────────────────
    with col_dl:
        csv_bytes = df_sc.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download table (CSV)",
            data=csv_bytes,
            file_name="biw_scenarios.csv",
            mime="text/csv",
        )
else:
    st.info("No scenarios saved yet. Adjust sliders above and click **Save scenario**.")

# ===========================================================================
# FEATURE 1 — Export Report (PDF)
# ===========================================================================
st.markdown("---")
st.markdown("### 📄 Export Report")

def build_pdf(fig1, fig2, scenarios, meta):
    """
    Generate a PDF report containing:
      - Title and run parameters
      - Both Plotly charts as embedded images
      - Results summary table
      - Scenarios comparison table (if any)
    Returns bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story  = []

    # ── Title block ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                  fontSize=16, textColor=colors.HexColor("#1b3a6b"),
                                  spaceAfter=4)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  fontSize=9, textColor=colors.HexColor("#5a6a8a"),
                                  spaceAfter=10)
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                  fontSize=9, leading=13,
                                  textColor=colors.HexColor("#1a1f2e"))
    hdr_style   = ParagraphStyle("hdr", parent=styles["Heading2"],
                                  fontSize=11, textColor=colors.HexColor("#1b3a6b"),
                                  spaceBefore=10, spaceAfter=4)

    story.append(Paragraph("⚡ BIW Crash Material Selector — Report", title_style))
    story.append(Paragraph(
        f"v₁ = {meta['v1']} km/h  →  v₂ = {meta['v2']} km/h  |  "
        f"R = {meta['R']:.3f}  |  CCI target = {meta['CCI_target']/1000:.3f} ×10³  |  "
        f"Energy target = {meta['EA_target']:.2f} kJ",
        sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d0daea"),
                             spaceAfter=8))

    # ── Parameters table ────────────────────────────────────────────────────
    story.append(Paragraph("Run Parameters", hdr_style))
    param_data = [
        ["Parameter", "Value"],
        ["L₁ (mm)", f"{meta['L1']:.2f}"],
        ["L₂ (mm)", f"{meta['L2']:.2f}"],
        ["CCI Baseline ×10³", f"{meta['CCI_baseline']/1000:.3f}"],
        ["CCI Target ×10³",   f"{meta['CCI_target']/1000:.3f}"],
        ["Energy Ratio R",    f"{meta['R']:.4f}"],
        ["Energy Baseline (kJ)", f"{meta['EA_base']:.2f}"],
        ["Energy Target (kJ)",   f"{meta['EA_target']:.2f}"],
    ]
    ts = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1b3a6b")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f4f6f9"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#d0daea")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ])
    story.append(Table(param_data, colWidths=[90*mm, 60*mm], style=ts))
    story.append(Spacer(1, 8*mm))

    # ── Charts ───────────────────────────────────────────────────────────────
    story.append(Paragraph("Charts", hdr_style))
    chart_w = 85*mm
    chart_h = 62*mm
    for fig, label in [(fig1, "Chart 1"), (fig2, "Chart 2")]:
        try:
            img_bytes = fig.to_image(format="png", width=700, height=500, scale=2)
            img_buf   = io.BytesIO(img_bytes)
            story.append(Paragraph(label, body_style))
            story.append(RLImage(img_buf, width=chart_w, height=chart_h))
            story.append(Spacer(1, 4*mm))
        except Exception:
            story.append(Paragraph(f"[{label} image unavailable — install kaleido]", body_style))

    # ── Scenarios table ──────────────────────────────────────────────────────
    if scenarios:
        story.append(Paragraph("Scenario Comparison", hdr_style))
        hdr = ["Name","Mode","RM₁","t₁","RM₂","t₂","Rem ×10³","t_min","Verdict"]
        tbl_data = [hdr]
        for s in scenarios:
            if s["mat_sufficient"]: v = "Mat OK"
            elif s["passes"] is True:  v = "PASS"
            elif s["passes"] is False: v = "FAIL"
            else: v = "—"
            tbl_data.append([
                s["name"], s["mode"],
                str(s["RM1"]), str(s["t1"]),
                str(s["RM2"]), str(s["t2"]),
                ("—" if s["mat_sufficient"] else f"{s['Remaining']/1000:.3f}"),
                f"{s['t_min']:.3f}" if s["t_min"] else "—",
                v,
            ])
        col_w = [28,18,16,14,16,14,20,18,18]
        col_w_mm = [w*mm for w in col_w]
        sc_ts = TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1b3a6b")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTSIZE",   (0,0), (-1,-1), 7),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f4f6f9"), colors.white]),
            ("GRID",(0,0),(-1,-1),0.3, colors.HexColor("#d0daea")),
            ("TOPPADDING",  (0,0),(-1,-1), 2),
            ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ])
        # Colour verdict cells
        for i, s in enumerate(scenarios, start=1):
            if s["passes"] is True or s["mat_sufficient"]:
                sc_ts.add("BACKGROUND", (8,i),(8,i), colors.HexColor("#d4edda"))
                sc_ts.add("TEXTCOLOR",  (8,i),(8,i), colors.HexColor("#1a7a3c"))
            elif s["passes"] is False:
                sc_ts.add("BACKGROUND", (8,i),(8,i), colors.HexColor("#fdecea"))
                sc_ts.add("TEXTCOLOR",  (8,i),(8,i), colors.HexColor("#c0392b"))
        story.append(Table(tbl_data, colWidths=col_w_mm, style=sc_ts))

    doc.build(story)
    buf.seek(0)
    return buf.read()

with st.expander("📄 Generate & Download PDF Report", expanded=False):
    st.caption("Charts are embedded as images. Requires **kaleido** (`pip install kaleido`).")
    if st.button("📥 Build PDF now"):
        with st.spinner("Rendering charts and building PDF…"):
            meta = dict(v1=v1, v2=v2, R=R, L1=L1, L2=L2,
                        CCI_baseline=CCI_baseline, CCI_target=CCI_target,
                        EA_base=EA_base, EA_target=EA_target)
            try:
                pdf_bytes = build_pdf(fig1, fig2, st.session_state.scenarios, meta)
                st.download_button(
                    label="⬇️ Download Report PDF",
                    data=pdf_bytes,
                    file_name="biw_crash_report.pdf",
                    mime="application/pdf",
                )
                st.success("PDF ready — click the button above to download.")
            except ImportError as e:
                st.error(f"Missing library: {e}. Run: pip install kaleido reportlab")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
