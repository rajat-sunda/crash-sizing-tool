"""
BIW Crash Material Selection Tool
==========================================
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
.weight-pos { color:#c0392b; font-weight:700; }   /* heavier = red  */
.weight-neg { color:#1a7a3c; font-weight:700; }   /* lighter = green */
.weight-neu { color:#5a6a8a; }
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
    <p>Concept-stage Body-in-White crash sizing tool &nbsp;·&nbsp; CCI (Crash Capacity Index) = RM₁·L₁·t₁ + RM₂·L₂·t₂</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
STEEL_DENSITY   = 7850.0          # kg/m³
STD_GRADES      = [420, 590, 780, 980, 1180, 1310]
STD_THICKNESSES = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]

def seg_len(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def compute_wall_lengths(pts):
    P1,P2,P3,P4 = pts["P1"],pts["P2"],pts["P3"],pts["P4"]
    P5,P6,P7,P8 = pts["P5"],pts["P6"],pts["P7"],pts["P8"]
    L1 = seg_len(P7,P5)+seg_len(P5,P1)+seg_len(P1,P4)+seg_len(P4,P6)+seg_len(P6,P8)
    L2 = seg_len(P7,P5)+seg_len(P5,P2)+seg_len(P2,P3)+seg_len(P3,P6)+seg_len(P6,P8)
    return L1, L2

def weight_penalty(L_mm, t_new_mm, t_base_mm):
    """
    Mass delta per unit rail length (kg/m) for one material side.
    L_mm      : wall length in mm
    t_new_mm  : new thickness in mm
    t_base_mm : baseline thickness in mm
    Returns signed kg/m value (positive = heavier, negative = lighter).
    """
    return (L_mm / 1000.0) * ((t_new_mm - t_base_mm) / 1000.0) * STEEL_DENSITY

def fmt_weight(w_kg_per_m):
    """Format weight penalty with sign and colour class."""
    sign   = "+" if w_kg_per_m > 0 else ""
    cls    = "weight-pos" if w_kg_per_m > 0 else ("weight-neg" if w_kg_per_m < 0 else "weight-neu")
    return f'<span class="{cls}">{sign}{w_kg_per_m:.3f} kg/m</span>'

def fmt_weight_plain(w_kg_per_m):
    """Plain text version for PDF/CSV."""
    sign = "+" if w_kg_per_m > 0 else ""
    return f"{sign}{w_kg_per_m:.3f} kg/m"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

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
    st.markdown("**🔵 Scatter Overlay**")
    show_scatter = st.checkbox("Show all grade × thickness combos", value=False,
                               help="Green = PASS, Red = FAIL. Hover shows weight penalty.")

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
    # Fixed side = Mat1; explore side = Mat2
    t_fixed_base  = t1_base
    t_explore_base= t2_base
    L_explore     = L2
    L_fixed       = L1
    S_fixed        = RM1_sel * L1 * t1_sel
    Remaining      = CCI_target - S_fixed
    mat_sufficient = Remaining <= 0
    t_pin_min      = None
    if fix_other and not mat_sufficient and RM2_pin > 0 and L2 > 0:
        t_pin_min = Remaining / (RM2_pin * L2)
    # Weight penalty for the FIXED side (Mat 1)
    wp_fixed = weight_penalty(L1, t1_sel, t1_base)
else:
    # Fixed side = Mat2; explore side = Mat1
    t_fixed_base  = t2_base
    t_explore_base= t1_base
    L_explore     = L1
    L_fixed       = L2
    S_fixed        = RM2_sel * L2 * t2_sel
    Remaining      = CCI_target - S_fixed
    mat_sufficient = Remaining <= 0
    t_pin_min      = None
    if fix_other and not mat_sufficient and RM1_pin > 0 and L1 > 0:
        t_pin_min = Remaining / (RM1_pin * L1)
    # Weight penalty for the FIXED side (Mat 2)
    wp_fixed = weight_penalty(L2, t2_sel, t2_base)

# Weight penalty for pin (explore side), if pinned
wp_pin = None
if t_pin_min is not None and t_pin_min > 0:
    wp_pin = weight_penalty(L_explore, t_pin_min, t_explore_base)

# Total weight penalty (fixed + explore sides combined), only if both known
wp_total = None
if t_pin_min is not None and t_pin_min > 0:
    wp_total = wp_fixed + wp_pin

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
GRADES       = [590, 780, 980, 1180]
GRADE_COLOR  = "rgba(80,110,180,0.45)"
T_RANGE      = np.linspace(0.8, 2.5, 300)
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

def add_selected_dot(fig, x_val, y_val, label, wp_kg=None):
    """Dot for the fixed material. Optionally show weight penalty in hover."""
    hover = label
    if wp_kg is not None:
        sign = "+" if wp_kg > 0 else ""
        hover += f"<br>ΔWeight = {sign}{wp_kg:.3f} kg/m"
    fig.add_trace(go.Scatter(
        x=[x_val], y=[y_val], mode="markers",
        marker=dict(color="#0d8c4e", size=14, symbol="circle",
                    line=dict(color="#1b3a6b", width=2.5)),
        name=label,
        hovertemplate=hover + "<extra></extra>",
    ))

def add_boundary_curve_with_weight(fig, t_range, bnd_cl, L_explore, t_base_mm, axis="t2"):
    """
    Boundary curve where hover shows:
      - thickness, min RM
      - weight penalty at that thickness vs baseline
    """
    lbl = "t₁" if axis == "t1" else "t₂"
    ax  = "RM₁" if axis == "t1" else "RM₂"
    # Pre-compute weight penalty array for every point on the curve
    wp_arr = np.array([weight_penalty(L_explore, float(t), t_base_mm) for t in t_range])
    sign_arr = np.where(wp_arr >= 0, "+", "")
    custom = np.stack([wp_arr, bnd_cl], axis=-1)   # shape (N, 2)
    fig.add_trace(go.Scatter(
        x=t_range, y=bnd_cl, mode="lines",
        line=dict(color="#e05c1a", width=2.5),
        name=f"Min {ax} boundary",
        customdata=custom,
        hovertemplate=(
            f"{lbl}: %{{x:.2f}} mm<br>"
            f"Min {ax}: %{{customdata[1]:.0f}} MPa<br>"
            f"ΔWeight: %{{customdata[0]:+.3f}} kg/m"
            "<extra></extra>"
        ),
    ))

def add_pin_marker(fig, t_val, rm_val, label, wp_kg=None):
    in_range = 0.8 <= t_val <= 2.5
    mc = "#0d8c4e" if in_range else "#c0392b"
    hover = f"t_min={t_val:.3f}mm"
    if wp_kg is not None:
        sign = "+" if wp_kg > 0 else ""
        hover += f"<br>ΔWeight = {sign}{wp_kg:.3f} kg/m"
    fig.add_trace(go.Scatter(
        x=[t_val], y=[rm_val], mode="markers+text",
        marker=dict(color=mc, size=14, symbol="diamond",
                    line=dict(color="#1b3a6b", width=2)),
        text=[f"  t_min={t_val:.2f}mm"], textposition="middle right",
        textfont=dict(size=10, color=mc),
        name=label,
        hovertemplate=hover + "<extra></extra>",
    ))

def add_scatter_overlay(fig, remaining, L_explore, t_base_mm, axis="t2"):
    """
    Scatter of all standard grade × thickness combos.
    Hover shows: grade, thickness, pass/fail, weight penalty.
    """
    pass_x, pass_y, pass_wp = [], [], []
    fail_x, fail_y, fail_wp = [], [], []
    for rm, t in itertools.product(STD_GRADES, STD_THICKNESSES):
        cci = rm * L_explore * t
        wp  = weight_penalty(L_explore, t, t_base_mm)
        if cci >= remaining:
            pass_x.append(t); pass_y.append(rm); pass_wp.append(wp)
        else:
            fail_x.append(t); fail_y.append(rm); fail_wp.append(wp)

    lbl = "t₁" if axis == "t1" else "t₂"
    ax  = "RM₁" if axis == "t1" else "RM₂"

    def hover_tmpl(status):
        return (
            f"{ax}=%{{y}} MPa  {lbl}=%{{x:.2f}} mm<br>"
            f"ΔWeight=%{{customdata:+.3f}} kg/m"
            f"<extra>{status}</extra>"
        )

    if pass_x:
        fig.add_trace(go.Scatter(
            x=pass_x, y=pass_y, mode="markers",
            marker=dict(color="rgba(22,160,80,0.75)", size=9, symbol="circle",
                        line=dict(color="rgba(22,160,80,1)", width=1)),
            customdata=pass_wp,
            name="PASS combo",
            hovertemplate=hover_tmpl("PASS"),
        ))
    if fail_x:
        fig.add_trace(go.Scatter(
            x=fail_x, y=fail_y, mode="markers",
            marker=dict(color="rgba(210,50,40,0.65)", size=9, symbol="circle",
                        line=dict(color="rgba(210,50,40,1)", width=1)),
            customdata=fail_wp,
            name="FAIL combo",
            hovertemplate=hover_tmpl("FAIL"),
        ))

# ---------------------------------------------------------------------------
# Build Chart 1 (RM1 vs t1)
# ---------------------------------------------------------------------------
fig1 = go.Figure()
add_grade_lines(fig1)

if fixing_mat1:
    # Fixed: show selected dot with weight penalty of Mat1
    add_selected_dot(fig1, t1_sel, RM1_sel,
                     f"RM₁={RM1_sel} MPa, t₁={t1_sel:.2f}mm",
                     wp_kg=wp_fixed)
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
        if show_scatter:
            add_scatter_overlay(fig1, Remaining, L1, t1_base, axis="t1")
        # Boundary curve with weight penalty in hover
        add_boundary_curve_with_weight(fig1, T_RANGE, bnd_cl, L1, t1_base, axis="t1")
        if fix_other and t_pin_min is not None and t_pin_min > 0:
            add_pin_marker(fig1, t_pin_min, RM1_pin,
                           f"t₁_min @ RM₁={RM1_pin}MPa", wp_kg=wp_pin)

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
        if show_scatter:
            add_scatter_overlay(fig2, Remaining, L2, t2_base, axis="t2")
        # Boundary curve with weight penalty in hover
        add_boundary_curve_with_weight(fig2, T_RANGE, bnd_cl, L2, t2_base, axis="t2")
        if fix_other and t_pin_min is not None and t_pin_min > 0:
            add_pin_marker(fig2, t_pin_min, RM2_pin,
                           f"t₂_min @ RM₂={RM2_pin}MPa", wp_kg=wp_pin)
else:
    chart2_title = "Chart 2 — Material 2 (Fixed Selection)"
    add_selected_dot(fig2, t2_sel, RM2_sel,
                     f"RM₂={RM2_sel}MPa, t₂={t2_sel:.2f}mm",
                     wp_kg=wp_fixed)

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
# Results card  (includes weight penalty section)
# ---------------------------------------------------------------------------
if mat_sufficient:
    fixed_name    = "Material 1" if fixing_mat1 else "Material 2"
    status_badge  = f'<span class="badge-ok">✅ {fixed_name.upper()} ALONE SUFFICIENT</span>'
    status_note   = f"{fixed_name} contribution exceeds CCI target."
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

# Weight penalty display strings
fixed_side_lbl   = "Mat 1" if fixing_mat1 else "Mat 2"
explore_side_lbl = "Mat 2" if fixing_mat1 else "Mat 1"

wp_fixed_html = fmt_weight(wp_fixed)
wp_pin_html   = fmt_weight(wp_pin)   if wp_pin   is not None else "<span class='weight-neu'>—</span>"
wp_total_html = fmt_weight(wp_total) if wp_total is not None else "<span class='weight-neu'>—</span>"

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

    <div style="border-top:1px solid #e8ecf4; margin-top:14px; padding-top:12px;">
        <div class="rlabel" style="margin-bottom:8px;">⚖️ WEIGHT PENALTY vs BASELINE &nbsp;
            <span style="font-size:.68rem; color:#8896b0;">(kg per metre of rail)</span>
        </div>
        <div class="result-row">
            <div class="result-item">
                <div class="rlabel">{fixed_side_lbl} ΔWeight</div>
                <div class="rvalue">{wp_fixed_html}</div>
            </div>
            <div class="result-item">
                <div class="rlabel">{explore_side_lbl} ΔWeight (pinned t_min)</div>
                <div class="rvalue">{wp_pin_html}</div>
            </div>
            <div class="result-item">
                <div class="rlabel">Total ΔWeight (both sides)</div>
                <div class="rvalue">{wp_total_html}</div>
            </div>
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
                wp_fixed=wp_fixed,
                wp_pin=wp_pin,
                wp_total=wp_total,
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
                wp_fixed=wp_fixed,
                wp_pin=wp_pin,
                wp_total=wp_total,
            )
        st.session_state.scenarios.append(sc)
        st.success(f"'{sc_name}' saved!")

if st.session_state.scenarios:
    rows = []
    for s in st.session_state.scenarios:
        if s["mat_sufficient"]:   verdict = "✅ Mat alone OK"
        elif s["passes"] is True: verdict = "✅ PASS"
        elif s["passes"] is False:verdict = "❌ FAIL"
        else:                     verdict = "— (no pin)"
        rows.append({
            "Name":             s["name"],
            "Mode":             s["mode"],
            "RM₁ (MPa)":        s["RM1"],
            "t₁ (mm)":          s["t1"],
            "RM₂ (MPa)":        s["RM2"],
            "t₂ (mm)":          s["t2"],
            "S_fixed ×10³":     f"{s['S_fixed']/1000:.3f}",
            "Remaining ×10³":   ("—" if s["mat_sufficient"] else f"{s['Remaining']/1000:.3f}"),
            "t_min (mm)":       f"{s['t_min']:.3f}" if s["t_min"] is not None else "—",
            "ΔW Fixed (kg/m)":  fmt_weight_plain(s["wp_fixed"]),
            "ΔW Explore (kg/m)":fmt_weight_plain(s["wp_pin"]) if s["wp_pin"] is not None else "—",
            "ΔW Total (kg/m)":  fmt_weight_plain(s["wp_total"]) if s["wp_total"] is not None else "—",
            "Verdict":          verdict,
        })
    df_sc = pd.DataFrame(rows)

    def colour_row(val, col):
        if col == "Verdict":
            if "PASS" in str(val) or "OK" in str(val):
                return "background-color:#e6f9ee; color:#1a7a3c; font-weight:700"
            if "FAIL" in str(val):
                return "background-color:#fdecea; color:#c0392b; font-weight:700"
        if col in ("ΔW Fixed (kg/m)", "ΔW Explore (kg/m)", "ΔW Total (kg/m)"):
            try:
                v = float(str(val).replace("+",""))
                if v > 0: return "color:#c0392b; font-weight:600"
                if v < 0: return "color:#1a7a3c; font-weight:600"
            except Exception: pass
        return ""

    def style_df(df):
        styled = df.style
        for col in df.columns:
            try:
                styled = styled.map(lambda v, c=col: colour_row(v, c), subset=[col])
            except AttributeError:
                styled = styled.applymap(lambda v, c=col: colour_row(v, c), subset=[col])
        return styled

    st.dataframe(style_df(df_sc), use_container_width=True, hide_index=True)

    col_dl, col_clr = st.columns([1,1])
    with col_clr:
        if st.button("🗑 Clear all scenarios"):
            st.session_state.scenarios = []
            st.rerun()
    with col_dl:
        csv_bytes = df_sc.to_csv(index=False).encode()
        st.download_button("⬇️ Download table (CSV)", csv_bytes,
                           "biw_scenarios.csv", "text/csv")
else:
    st.info("No scenarios saved yet. Adjust sliders above and click **Save scenario**.")

# ===========================================================================
# FEATURE 1 — Export Report (PDF via matplotlib, no kaleido)
# ===========================================================================

def render_chart_matplotlib(chart_data):
    """Draw chart using matplotlib Agg — no Chrome or kaleido needed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GRADES_MPL  = [590, 780, 980, 1180]
    NAV   = "#1b3a6b"
    ORANGE= "#e05c1a"
    Y_MIN_M, Y_MAX_M = 350, 1450
    T_MIN, T_MAX = 0.75, 2.55

    fig_mpl, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    ax.set_facecolor("#f8fafd")
    fig_mpl.patch.set_facecolor("white")

    for g in GRADES_MPL:
        ax.axhline(g, color="#8096c8", lw=0.9, ls="--", alpha=0.7)
        ax.text(T_MAX + 0.02, g, f"{g} MPa", va="center", ha="left",
                fontsize=7, color="#5a6a9a")

    if chart_data.get("sufficient_msg"):
        ax.text(1.65, 900, chart_data["sufficient_msg"], ha="center", va="center",
                fontsize=12, color="#1a7a3c",
                bbox=dict(boxstyle="round,pad=0.5", fc="#dcffe4", ec="#5abf7e", lw=1.2))
    else:
        bnd = chart_data.get("boundary")
        if bnd is not None:
            t_arr, rm_arr = bnd
            if chart_data.get("pass_fill"):
                ax.fill_between(t_arr, rm_arr, Y_MAX_M, color="#22a846", alpha=0.10, zorder=1)
                ax.fill_between(t_arr, Y_MIN_M, rm_arr, color="#c83228", alpha=0.08, zorder=1)
            ax.plot(t_arr, rm_arr, color=ORANGE, lw=2.2, zorder=3, label="Min boundary")

        sp = chart_data.get("scatter_pass")
        sf = chart_data.get("scatter_fail")
        if sp:
            xs, ys = zip(*sp)
            ax.scatter(xs, ys, c="#16a050", s=28, alpha=0.75, zorder=4,
                       label="PASS combo", edgecolors="#16a050", linewidths=0.5)
        if sf:
            xs, ys = zip(*sf)
            ax.scatter(xs, ys, c="#d23228", s=28, alpha=0.65, zorder=4,
                       label="FAIL combo", edgecolors="#d23228", linewidths=0.5)

        dot = chart_data.get("selected_dot")
        if dot:
            x, y, lbl, wp_val = dot
            sign = f" (ΔW {wp_val:+.3f} kg/m)" if wp_val is not None else ""
            ax.scatter([x], [y], c="#0d8c4e", s=120, zorder=6,
                       edgecolors=NAV, linewidths=1.8, label=lbl + sign)

        pin = chart_data.get("pin_marker")
        if pin:
            x, y, lbl, wp_val = pin
            ok = 0.8 <= x <= 2.5
            mc = "#0d8c4e" if ok else "#c0392b"
            sign = f" (ΔW {wp_val:+.3f} kg/m)" if wp_val is not None else ""
            ax.scatter([x], [y], c=mc, s=120, marker="D", zorder=6,
                       edgecolors=NAV, linewidths=1.8, label=lbl + sign)
            ax.annotate(f"t_min={x:.2f}mm", (x, y),
                        xytext=(8, 4), textcoords="offset points", fontsize=7, color=mc)

    ax.set_xlim(T_MIN, T_MAX + 0.18)
    ax.set_ylim(Y_MIN_M, Y_MAX_M)
    ax.set_xlabel(chart_data["xlabel"], fontsize=9, color="#3a4a6a")
    ax.set_ylabel(chart_data["ylabel"], fontsize=9, color="#3a4a6a")
    ax.set_title(chart_data["title"], fontsize=10, color=NAV, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8, colors="#3a4a6a")
    ax.grid(True, color="#dde4f0", lw=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#c0cbdd")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="upper right",
                  framealpha=0.9, edgecolor="#d0daea")

    fig_mpl.tight_layout()
    buf_img = io.BytesIO()
    fig_mpl.savefig(buf_img, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _plt; _plt.close(fig_mpl)
    buf_img.seek(0)
    return buf_img.read()


def build_pdf(chart1_data, chart2_data, scenarios, meta):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                  fontSize=16, textColor=colors.HexColor("#1b3a6b"), spaceAfter=4)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  fontSize=9, textColor=colors.HexColor("#5a6a8a"), spaceAfter=10)
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                  fontSize=9, leading=13, textColor=colors.HexColor("#1a1f2e"))
    hdr_style   = ParagraphStyle("hdr", parent=styles["Heading2"],
                                  fontSize=11, textColor=colors.HexColor("#1b3a6b"),
                                  spaceBefore=10, spaceAfter=4)

    story.append(Paragraph("BIW Crash Material Selector - Report", title_style))
    story.append(Paragraph(
        f"v1={meta['v1']} km/h -> v2={meta['v2']} km/h  |  "
        f"R={meta['R']:.3f}  |  CCI target={meta['CCI_target']/1000:.3f}x10^3  |  "
        f"EA target={meta['EA_target']:.2f} kJ",
        sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#d0daea"), spaceAfter=8))

    story.append(Paragraph("Run Parameters", hdr_style))
    param_data = [
        ["Parameter", "Value"],
        ["L1 (mm)",              f"{meta['L1']:.2f}"],
        ["L2 (mm)",              f"{meta['L2']:.2f}"],
        ["CCI Baseline x10^3",   f"{meta['CCI_baseline']/1000:.3f}"],
        ["CCI Target x10^3",     f"{meta['CCI_target']/1000:.3f}"],
        ["Energy Ratio R",       f"{meta['R']:.4f}"],
        ["Energy Baseline (kJ)", f"{meta['EA_base']:.2f}"],
        ["Energy Target (kJ)",   f"{meta['EA_target']:.2f}"],
        ["Steel density (kg/m3)","7850"],
    ]
    ts = TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#1b3a6b")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f4f6f9"), colors.white]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#d0daea")),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
    ])
    story.append(Table(param_data, colWidths=[90*mm, 60*mm], style=ts))
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("Charts", hdr_style))
    chart_w = 168*mm; chart_h = 112*mm
    for cdata, label in [(chart1_data, "Chart 1 - Material 1"),
                         (chart2_data, "Chart 2 - Material 2")]:
        story.append(Paragraph(label, body_style))
        img_bytes = render_chart_matplotlib(cdata)
        story.append(RLImage(io.BytesIO(img_bytes), width=chart_w, height=chart_h))
        story.append(Spacer(1, 5*mm))

    if scenarios:
        story.append(Paragraph("Scenario Comparison (with Weight Penalty)", hdr_style))
        hdr = ["Name","Mode","RM1","t1","RM2","t2","Rem x10^3",
               "t_min","DW Fix","DW Exp","DW Tot","Verdict"]
        tbl_data = [hdr]
        for s in scenarios:
            if s["mat_sufficient"]:   v = "Mat OK"
            elif s["passes"] is True: v = "PASS"
            elif s["passes"] is False:v = "FAIL"
            else:                     v = "—"
            tbl_data.append([
                s["name"], s["mode"],
                str(s["RM1"]), str(s["t1"]),
                str(s["RM2"]), str(s["t2"]),
                ("—" if s["mat_sufficient"] else f"{s['Remaining']/1000:.3f}"),
                f"{s['t_min']:.3f}" if s["t_min"] else "—",
                fmt_weight_plain(s["wp_fixed"]),
                fmt_weight_plain(s["wp_pin"])   if s["wp_pin"]   is not None else "—",
                fmt_weight_plain(s["wp_total"]) if s["wp_total"] is not None else "—",
                v,
            ])
        col_w_mm = [w*mm for w in [24,15,14,12,14,12,18,16,18,18,18,15]]
        sc_ts = TableStyle([
            ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#1b3a6b")),
            ("TEXTCOLOR",  (0,0),(-1,0), colors.white),
            ("FONTSIZE",   (0,0),(-1,-1), 7),
            ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f4f6f9"), colors.white]),
            ("GRID",(0,0),(-1,-1),0.3, colors.HexColor("#d0daea")),
            ("TOPPADDING",  (0,0),(-1,-1), 2),
            ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ])
        for i, s in enumerate(scenarios, start=1):
            if s["passes"] is True or s["mat_sufficient"]:
                sc_ts.add("BACKGROUND",(11,i),(11,i), colors.HexColor("#d4edda"))
                sc_ts.add("TEXTCOLOR", (11,i),(11,i), colors.HexColor("#1a7a3c"))
            elif s["passes"] is False:
                sc_ts.add("BACKGROUND",(11,i),(11,i), colors.HexColor("#fdecea"))
                sc_ts.add("TEXTCOLOR", (11,i),(11,i), colors.HexColor("#c0392b"))
            # Colour weight cells
            for col_idx, wp_key in [(8,"wp_fixed"),(9,"wp_pin"),(10,"wp_total")]:
                wp = s.get(wp_key)
                if wp is not None:
                    c = colors.HexColor("#fdecea") if wp > 0 else (colors.HexColor("#d4edda") if wp < 0 else colors.white)
                    sc_ts.add("BACKGROUND",(col_idx,i),(col_idx,i), c)
        story.append(Table(tbl_data, colWidths=col_w_mm, style=sc_ts))

    doc.build(story)
    buf.seek(0)
    return buf.read()


st.markdown("---")
st.markdown("### 📄 Export Report")

with st.expander("📄 Generate & Download PDF Report", expanded=False):
    if st.button("📥 Build PDF now"):
        with st.spinner("Rendering charts and building PDF…"):
            meta = dict(v1=v1, v2=v2, R=R, L1=L1, L2=L2,
                        CCI_baseline=CCI_baseline, CCI_target=CCI_target,
                        EA_base=EA_base, EA_target=EA_target)

            def scatter_combos_pdf(remaining, L_side, t_base_mm):
                sp, sf = [], []
                for rm, t in itertools.product(STD_GRADES, STD_THICKNESSES):
                    (sp if rm*L_side*t >= remaining else sf).append((t, rm))
                return sp, sf

            if fixing_mat1:
                c1 = dict(
                    title="Chart 1 - Material 1 (Fixed)",
                    xlabel="Thickness t1 (mm)", ylabel="Grade RM1 (MPa)",
                    boundary=None, pass_fill=False,
                    selected_dot=(t1_sel, RM1_sel,
                                  f"RM1={RM1_sel}MPa t1={t1_sel:.2f}mm", wp_fixed),
                    pin_marker=None, scatter_pass=None, scatter_fail=None,
                    sufficient_msg=None,
                )
                if mat_sufficient:
                    c2 = dict(title="Chart 2 - Material 2 Feasible Region",
                              xlabel="Thickness t2 (mm)", ylabel="Grade RM2 (MPa)",
                              boundary=None, pass_fill=False, selected_dot=None,
                              pin_marker=None, scatter_pass=None, scatter_fail=None,
                              sufficient_msg="Material 1 alone is sufficient")
                else:
                    bnd_t = T_RANGE
                    bnd_y = np.clip(Remaining/(L2*bnd_t), 350, 1450)
                    sp, sf = scatter_combos_pdf(Remaining, L2, t2_base) if show_scatter else (None, None)
                    pin = ((t_pin_min, RM2_pin,
                            f"t2_min @ RM2={RM2_pin}MPa", wp_pin)
                           if (fix_other and t_pin_min and t_pin_min > 0) else None)
                    c2 = dict(title="Chart 2 - Material 2 Feasible Region",
                              xlabel="Thickness t2 (mm)", ylabel="Grade RM2 (MPa)",
                              boundary=(bnd_t, bnd_y), pass_fill=True,
                              selected_dot=None, pin_marker=pin,
                              scatter_pass=sp, scatter_fail=sf, sufficient_msg=None)
            else:
                c2 = dict(
                    title="Chart 2 - Material 2 (Fixed)",
                    xlabel="Thickness t2 (mm)", ylabel="Grade RM2 (MPa)",
                    boundary=None, pass_fill=False,
                    selected_dot=(t2_sel, RM2_sel,
                                  f"RM2={RM2_sel}MPa t2={t2_sel:.2f}mm", wp_fixed),
                    pin_marker=None, scatter_pass=None, scatter_fail=None,
                    sufficient_msg=None,
                )
                if mat_sufficient:
                    c1 = dict(title="Chart 1 - Material 1 Feasible Region",
                              xlabel="Thickness t1 (mm)", ylabel="Grade RM1 (MPa)",
                              boundary=None, pass_fill=False, selected_dot=None,
                              pin_marker=None, scatter_pass=None, scatter_fail=None,
                              sufficient_msg="Material 2 alone is sufficient")
                else:
                    bnd_t = T_RANGE
                    bnd_y = np.clip(Remaining/(L1*bnd_t), 350, 1450)
                    sp, sf = scatter_combos_pdf(Remaining, L1, t1_base) if show_scatter else (None, None)
                    pin = ((t_pin_min, RM1_pin,
                            f"t1_min @ RM1={RM1_pin}MPa", wp_pin)
                           if (fix_other and t_pin_min and t_pin_min > 0) else None)
                    c1 = dict(title="Chart 1 - Material 1 Feasible Region",
                              xlabel="Thickness t1 (mm)", ylabel="Grade RM1 (MPa)",
                              boundary=(bnd_t, bnd_y), pass_fill=True,
                              selected_dot=None, pin_marker=pin,
                              scatter_pass=sp, scatter_fail=sf, sufficient_msg=None)

            try:
                pdf_bytes = build_pdf(c1, c2, st.session_state.scenarios, meta)
                st.download_button(
                    label="⬇️ Download Report PDF",
                    data=pdf_bytes,
                    file_name="biw_crash_report.pdf",
                    mime="application/pdf",
                )
                st.success("PDF ready — click the button above to download.")
            except Exception as e:
                import traceback
                st.error("PDF generation failed. See details below:")
                st.code(traceback.format_exc(), language="text")
