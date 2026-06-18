"""
BIW Crash Material Selection Tool  —  v5
==========================================
Formula:
    Mass Penalty (kg/m) = (L_mm/1000) × (t_new_mm - t_base_mm)/1000 × 7850
    Cost Penalty (Rs/m) = (L_mm/1000) × t_new_mm/1000 × rate_new  (Rs/m³)
                        - (L_mm/1000) × t_base_mm/1000 × rate_base (Rs/m³)

Run with:
    streamlit run app.py
"""

import io, math, itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, Image as RLImage, HRFlowable)

# ===========================================================================
# ★★★  MATERIAL DATASHEET  ★★★
# ---------------------------------------------------------------------------
# Replace this section with your actual datasheet once you receive it.
#
# MATERIAL_DATA is a list of dicts, one per grade. Each dict has:
#   "grade"      : int   — RM value in MPa
#   "rate"       : float — material cost in Rs/m³
#   "thicknesses": list  — available thicknesses in mm for this grade
#
# HOW TO UPDATE:
#   1. Find the block below that starts with "MATERIAL_DATA = ["
#   2. For each grade your supplier offers, add/edit a dict entry.
#   3. Set "rate" to the Rs/m³ price from your datasheet.
#   4. Set "thicknesses" to only the gauges your supplier stocks for that grade.
#
# Current values are APPROXIMATE/ILLUSTRATIVE — sourced from publicly available
# Indian steel market data (SAIL/JSW price lists, ~2024). Rates are indicative
# and must be replaced with your actual vendor quotes.
# ---------------------------------------------------------------------------
MATERIAL_DATA = [
    # grade  rate(Rs/m³)   available thicknesses (mm)
    {"grade": 420,  "rate":  68_000, "thicknesses": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]},
    {"grade": 590,  "rate":  82_000, "thicknesses": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]},
    {"grade": 780,  "rate":  98_000, "thicknesses": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]},
    {"grade": 980,  "rate": 118_000, "thicknesses": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]},
    {"grade": 1180, "rate": 145_000, "thicknesses": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]},
    {"grade": 1310, "rate": 172_000, "thicknesses": [1.0, 1.2, 1.4, 1.6, 1.8]},
]
# ★★★  END OF DATASHEET SECTION  ★★★
# ===========================================================================

# Build lookup: grade → rate, and flat list of (grade, thickness) pairs
GRADE_RATE    = {d["grade"]: d["rate"] for d in MATERIAL_DATA}
DATASHEET_COMBOS = [(d["grade"], t) for d in MATERIAL_DATA for t in d["thicknesses"]]
# Unique sorted grades and thicknesses (for sliders / dropdowns)
ALL_GRADES      = sorted(GRADE_RATE.keys())
ALL_THICKNESSES = sorted(set(t for d in MATERIAL_DATA for t in d["thicknesses"]))

STEEL_DENSITY = 7850.0   # kg/m³

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="BIW Crash Material Selector", page_icon="🚗",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
.main .block-container {
    background-color:#f4f6f9 !important; color:#1a1f2e !important;
    font-family:'Inter','Segoe UI',sans-serif;
}
.app-header {
    background:linear-gradient(90deg,#1b3a6b 0%,#2a5298 60%,#1e6fbf 100%);
    border-bottom:3px solid #1b3a6b; padding:18px 32px 14px; margin-bottom:24px; border-radius:8px;
}
.app-header h1{margin:0;font-size:1.55rem;font-weight:700;letter-spacing:.04em;color:#fff;}
.app-header p{margin:4px 0 0;font-size:.83rem;color:#c8daf7;letter-spacing:.02em;}
.metric-row{display:flex;gap:12px;margin-bottom:22px;flex-wrap:wrap;}
.metric-card{
    flex:1;background:#fff;border:1px solid #d0daea;border-top:3px solid #2a5298;
    border-radius:8px;padding:14px 18px;min-width:110px;box-shadow:0 1px 4px rgba(0,0,0,.07);
}
.metric-card .label{font-size:.68rem;color:#6b7a99;letter-spacing:.07em;text-transform:uppercase;margin-bottom:5px;}
.metric-card .value{font-size:1.4rem;font-weight:700;color:#1b3a6b;font-variant-numeric:tabular-nums;}
.metric-card .unit{font-size:.70rem;color:#8896b0;margin-left:3px;}
.result-card{
    background:#fff;border:1px solid #d0daea;border-radius:8px;
    padding:18px 24px;margin-top:20px;box-shadow:0 1px 4px rgba(0,0,0,.07);
}
.result-card h4{
    margin:0 0 12px;font-size:.78rem;color:#5a6a8a;letter-spacing:.07em;
    text-transform:uppercase;border-bottom:1px solid #e8ecf4;padding-bottom:8px;
}
.result-row{display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start;}
.result-item .rlabel{font-size:.70rem;color:#6b7a99;letter-spacing:.04em;text-transform:uppercase;margin-bottom:2px;}
.result-item .rvalue{font-size:1.1rem;font-weight:600;color:#1a1f2e;}
.badge-pass{background:#e6f9ee;color:#1a7a3c;border:1px solid #5abf7e;padding:4px 14px;border-radius:5px;font-size:.85rem;font-weight:700;}
.badge-fail{background:#fdecea;color:#c0392b;border:1px solid #e07070;padding:4px 14px;border-radius:5px;font-size:.85rem;font-weight:700;}
.badge-ok{background:#e8f0fe;color:#1b3a9e;border:1px solid #7096e8;padding:4px 14px;border-radius:5px;font-size:.85rem;font-weight:700;}
[data-testid="stSidebar"]{background-color:#ffffff !important;border-right:1px solid #d0daea;}
[data-testid="stSidebar"] *{color:#1a1f2e !important;}
.sidebar-title{font-size:.95rem;font-weight:700;color:#1b3a6b !important;letter-spacing:.04em;
    margin-bottom:8px;padding-bottom:6px;border-bottom:2px solid #2a5298;}
footer{visibility:hidden;} #MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>⚡ BIW Crash Material Selector</h1>
    <p>Concept-stage BIW crash sizing tool &nbsp;·&nbsp; CCI (Crash Capacity Indicator) = RM₁·L₁·t₁ + RM₂·L₂·t₂</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seg_len(a, b):
    return math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)

def compute_wall_lengths(pts):
    P1,P2,P3,P4=pts["P1"],pts["P2"],pts["P3"],pts["P4"]
    P5,P6,P7,P8=pts["P5"],pts["P6"],pts["P7"],pts["P8"]
    L1=seg_len(P7,P5)+seg_len(P5,P1)+seg_len(P1,P4)+seg_len(P4,P6)+seg_len(P6,P8)
    L2=seg_len(P7,P5)+seg_len(P5,P2)+seg_len(P2,P3)+seg_len(P3,P6)+seg_len(P6,P8)
    return L1, L2

def mass_penalty(L_mm, t_new_mm, t_base_mm):
    """kg/m — positive = heavier, negative = lighter."""
    return (L_mm/1000)*((t_new_mm-t_base_mm)/1000)*STEEL_DENSITY

def cost_penalty(L_mm, t_new_mm, rate_new, t_base_mm, rate_base):
    """
    Rs/m — cost of new design minus cost of baseline for this wall.
    cost = L(m) × t(m) × rate(Rs/m³)
    """
    cost_new  = (L_mm/1000)*(t_new_mm/1000)*rate_new
    cost_base = (L_mm/1000)*(t_base_mm/1000)*rate_base
    return cost_new - cost_base

def rate_for(grade):
    """Look up Rs/m³ rate for a grade; return 0 if not in datasheet."""
    return GRADE_RATE.get(int(grade), 0)

def fmt_plain(val, unit):
    if val is None: return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f} {unit}"

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
        "P1":(0.,100.),"P2":(100.,100.),"P3":(100.,0.),"P4":(0.,0.),
        "P5":(40.,100.),"P6":(40.,0.),"P7":(40.,120.),"P8":(40.,-20.),
    }
    with st.expander("📐 Geometry Points (mm)", expanded=False):
        pts = {}
        for key,(dx,dy) in default_pts.items():
            c1,c2 = st.columns(2)
            with c1: px=st.number_input(f"{key} x",value=dx,step=1.,key=f"{key}_x",format="%.1f")
            with c2: py=st.number_input(f"{key} y",value=dy,step=1.,key=f"{key}_y",format="%.1f")
            pts[key]=(px,py)

    with st.expander("💥 Crash Scenario", expanded=True):
        v1=st.number_input("Initial speed v₁ (km/h)",value=56.,min_value=1.,step=1.)
        v2=st.number_input("Target speed v₂ (km/h)", value=64.,min_value=1.,step=1.)

    with st.expander("📋 Baseline Design", expanded=False):
        RM1_base=st.number_input("RM₁ baseline (MPa)",value=590.,min_value=100.,step=10.)
        t1_base =st.number_input("t₁ baseline (mm)",  value=1.4, min_value=0.1, step=0.05)
        RM2_base=st.number_input("RM₂ baseline (MPa)",value=780.,min_value=100.,step=10.)
        t2_base =st.number_input("t₂ baseline (mm)",  value=1.2, min_value=0.1, step=0.05)
        EA_base =st.number_input("EA baseline (kJ)",  value=8.5, min_value=0.01,step=0.1)

    st.markdown("---")
    st.markdown("**🔀 Analysis Mode**")
    mode = st.radio("Fix which material?",
                    ["Fix Material 1 → explore Mat 2","Fix Material 2 → explore Mat 1"],index=0)
    fixing_mat1 = mode.startswith("Fix Material 1")

    st.markdown("---")
    if fixing_mat1:
        st.markdown("**Material 1 — Fixed inputs**")
        RM1_sel=st.slider("RM₁ (MPa)",min(ALL_GRADES),max(ALL_GRADES),780,10)
        t1_sel =st.slider("t₁ (mm)",min(ALL_THICKNESSES),max(ALL_THICKNESSES),1.4,0.05)
        st.markdown("---")
        st.markdown("**Material 2 — Optional pin**")
        st.caption("Pin RM₂ to find minimum t₂.")
        fix_other=st.checkbox("Pin RM₂",value=False)
        RM2_pin=st.slider("RM₂ (MPa)",min(ALL_GRADES),max(ALL_GRADES),980,10,disabled=not fix_other)
        RM1_pin=None; t2_sel=None
    else:
        st.markdown("**Material 2 — Fixed inputs**")
        RM2_sel=st.slider("RM₂ (MPa)",min(ALL_GRADES),max(ALL_GRADES),980,10)
        t2_sel =st.slider("t₂ (mm)",min(ALL_THICKNESSES),max(ALL_THICKNESSES),1.2,0.05)
        st.markdown("---")
        st.markdown("**Material 1 — Optional pin**")
        st.caption("Pin RM₁ to find minimum t₁.")
        fix_other=st.checkbox("Pin RM₁",value=False)
        RM1_pin=st.slider("RM₁ (MPa)",min(ALL_GRADES),max(ALL_GRADES),780,10,disabled=not fix_other)
        RM2_pin=None; RM1_sel=None; t1_sel=None

    st.markdown("---")
    st.markdown("**🔵 Scatter Overlay**")
    show_scatter=st.checkbox("Show datasheet grade × thickness combos",value=False,
        help="Shows only combinations from your material datasheet. Green=PASS, Red=FAIL. Hover shows mass & cost penalty.")

# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------
L1,L2 = compute_wall_lengths(pts)
R             = (v2/v1)**2
CCI_baseline  = RM1_base*L1*t1_base + RM2_base*L2*t2_base
CCI_target    = CCI_baseline*R
C             = EA_base/CCI_baseline if CCI_baseline>0 else 0.
EA_target     = C*CCI_target

if fixing_mat1:
    t_fixed_base=t1_base; t_explore_base=t2_base
    L_fixed=L1; L_explore=L2
    RM_fixed=RM1_sel; t_fixed=t1_sel
    S_fixed=RM1_sel*L1*t1_sel; Remaining=CCI_target-S_fixed
    mat_sufficient=Remaining<=0
    t_pin_min=None
    if fix_other and not mat_sufficient and RM2_pin>0 and L2>0:
        t_pin_min=Remaining/(RM2_pin*L2)
    rate_fixed = rate_for(RM1_sel)
    rate_fixed_base = rate_for(RM1_base)
    rate_pin   = rate_for(RM2_pin) if fix_other else 0
    rate_explore_base = rate_for(RM2_base)
else:
    t_fixed_base=t2_base; t_explore_base=t1_base
    L_fixed=L2; L_explore=L1
    RM_fixed=RM2_sel; t_fixed=t2_sel
    S_fixed=RM2_sel*L2*t2_sel; Remaining=CCI_target-S_fixed
    mat_sufficient=Remaining<=0
    t_pin_min=None
    if fix_other and not mat_sufficient and RM1_pin>0 and L1>0:
        t_pin_min=Remaining/(RM1_pin*L1)
    rate_fixed = rate_for(RM2_sel)
    rate_fixed_base = rate_for(RM2_base)
    rate_pin   = rate_for(RM1_pin) if fix_other else 0
    rate_explore_base = rate_for(RM1_base)

# Mass & cost penalty — FIXED side
mp_fixed = mass_penalty(L_fixed, t_fixed, t_fixed_base)
cp_fixed = cost_penalty(L_fixed, t_fixed, rate_fixed, t_fixed_base, rate_fixed_base)

# Mass & cost penalty — EXPLORE side (only if pinned)
mp_pin = cp_pin = mp_total = cp_total = None
if t_pin_min is not None and t_pin_min>0:
    mp_pin   = mass_penalty(L_explore, t_pin_min, t_explore_base)
    cp_pin   = cost_penalty(L_explore, t_pin_min, rate_pin, t_explore_base, rate_explore_base)
    mp_total = mp_fixed + mp_pin
    cp_total = cp_fixed + cp_pin

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
GRADES_REF   = [590,780,980,1180]
GRADE_COLOR  = "rgba(80,110,180,0.45)"
T_RANGE      = np.linspace(min(ALL_THICKNESSES),max(ALL_THICKNESSES),300)
Y_MIN,Y_MAX  = 350,1450

BASE_LAYOUT = dict(
    paper_bgcolor="#ffffff",plot_bgcolor="#f8fafd",
    font=dict(color="#1a1f2e",size=11,family="Inter,Segoe UI,sans-serif"),
    margin=dict(l=60,r=70,t=50,b=50),showlegend=True,
    legend=dict(bgcolor="rgba(255,255,255,0.85)",bordercolor="#d0daea",
                borderwidth=1,font=dict(size=9,color="#3a4a6a")),
)
AXIS_STYLE=dict(gridcolor="#dde4f0",zerolinecolor="#c8d0e0",linecolor="#c0cbdd",
                tickcolor="#8896b0",tickfont=dict(size=10,color="#3a4a6a"))

def add_grade_lines(fig):
    for g in GRADES_REF:
        fig.add_hline(y=g,line=dict(color=GRADE_COLOR,dash="dash",width=1.2),
                      annotation_text=f"{g} MPa",annotation_position="right",
                      annotation_font=dict(size=9,color="#5a6a9a"))

def add_boundary_zones(fig,t_range,bnd_cl):
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_range,t_range[::-1]]),
        y=np.concatenate([bnd_cl,np.full(len(t_range),Y_MAX)]),
        fill="toself",fillcolor="rgba(34,139,34,0.10)",
        line=dict(width=0),hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_range,t_range[::-1]]),
        y=np.concatenate([bnd_cl,np.full(len(t_range),Y_MIN)]),
        fill="toself",fillcolor="rgba(200,50,40,0.08)",
        line=dict(width=0),hoverinfo="skip",showlegend=False))

def add_selected_dot(fig,x,y,label,mp_kg=None,cp_rs=None):
    hover=label
    if mp_kg is not None: hover+=f"<br>Mass Penalty={mp_kg:+.3f} kg/m"
    if cp_rs is not None: hover+=f"<br>Cost Penalty={cp_rs:+.2f} Rs/m"
    fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers",
        marker=dict(color="#0d8c4e",size=14,symbol="circle",
                    line=dict(color="#1b3a6b",width=2.5)),
        name=label,hovertemplate=hover+"<extra></extra>"))

def add_boundary_curve(fig,t_range,bnd_cl,L_ex,t_base_mm,rate_ex,rate_base_ex,axis="t2"):
    lbl="t₁" if axis=="t1" else "t₂"
    ax ="RM₁" if axis=="t1" else "RM₂"
    mp_arr=np.round([mass_penalty(L_ex,float(t),t_base_mm) for t in t_range],3)
    cp_arr=np.round([cost_penalty(L_ex,float(t),rate_ex,t_base_mm,rate_base_ex) for t in t_range],2)
    custom=np.stack([mp_arr,bnd_cl,cp_arr],axis=-1)
    fig.add_trace(go.Scatter(x=t_range,y=bnd_cl,mode="lines",
        line=dict(color="#e05c1a",width=2.5),name=f"Min {ax} boundary",
        customdata=custom,
        hovertemplate=(f"{lbl}: %{{x:.2f}} mm<br>"
                       f"Min {ax}: %{{customdata[1]:.0f}} MPa<br>"
                       f"Mass Penalty: %{{customdata[0]:+.3f}} kg/m<br>"
                       f"Cost Penalty: %{{customdata[2]:+.2f}} Rs/m"
                       "<extra></extra>")))

def add_pin_marker(fig,t_val,rm_val,label,mp_kg=None,cp_rs=None):
    ok=(0.8<=t_val<=2.5); mc="#0d8c4e" if ok else "#c0392b"
    hover=f"t_min={t_val:.3f}mm"
    if mp_kg is not None: hover+=f"<br>Mass Penalty={mp_kg:+.3f} kg/m"
    if cp_rs is not None: hover+=f"<br>Cost Penalty={cp_rs:+.2f} Rs/m"
    fig.add_trace(go.Scatter(x=[t_val],y=[rm_val],mode="markers+text",
        marker=dict(color=mc,size=14,symbol="diamond",line=dict(color="#1b3a6b",width=2)),
        text=[f"  t_min={t_val:.2f}mm"],textposition="middle right",
        textfont=dict(size=10,color=mc),name=label,
        hovertemplate=hover+"<extra></extra>"))

def add_scatter_overlay(fig,remaining,L_ex,t_base_mm,rate_ex,rate_base_ex,axis="t2"):
    """
    Only plots datasheet combos (grade+thickness from MATERIAL_DATA).
    Hover shows grade, thickness, pass/fail, mass penalty, cost penalty.
    """
    pass_x,pass_y,pass_cd=[],[],[]
    fail_x,fail_y,fail_cd=[],[],[]
    for grade,t in DATASHEET_COMBOS:
        cci  = grade*L_ex*t
        rate = rate_for(grade)
        mp   = round(mass_penalty(L_ex,t,t_base_mm),3)
        cp   = round(cost_penalty(L_ex,t,rate,t_base_mm,rate_base_ex),2)
        cd   = [mp,cp]
        if cci>=remaining:
            pass_x.append(t); pass_y.append(grade); pass_cd.append(cd)
        else:
            fail_x.append(t); fail_y.append(grade); fail_cd.append(cd)

    lbl="t₁" if axis=="t1" else "t₂"
    ax ="RM₁" if axis=="t1" else "RM₂"
    htmpl = lambda status: (f"{ax}=%{{y}} MPa  {lbl}=%{{x:.2f}} mm<br>"
                             f"Mass Penalty=%{{customdata[0]:+.3f}} kg/m<br>"
                             f"Cost Penalty=%{{customdata[1]:+.2f}} Rs/m"
                             f"<extra>{status}</extra>")
    if pass_x:
        fig.add_trace(go.Scatter(x=pass_x,y=pass_y,mode="markers",
            marker=dict(color="rgba(22,160,80,0.75)",size=9,symbol="circle",
                        line=dict(color="rgba(22,160,80,1)",width=1)),
            customdata=pass_cd,name="PASS combo",hovertemplate=htmpl("PASS")))
    if fail_x:
        fig.add_trace(go.Scatter(x=fail_x,y=fail_y,mode="markers",
            marker=dict(color="rgba(210,50,40,0.65)",size=9,symbol="circle",
                        line=dict(color="rgba(210,50,40,1)",width=1)),
            customdata=fail_cd,name="FAIL combo",hovertemplate=htmpl("FAIL")))

# ---------------------------------------------------------------------------
# Build Chart 1
# ---------------------------------------------------------------------------
fig1=go.Figure(); add_grade_lines(fig1)

if fixing_mat1:
    add_selected_dot(fig1,t1_sel,RM1_sel,f"RM₁={RM1_sel}MPa t₁={t1_sel:.2f}mm",
                     mp_kg=mp_fixed,cp_rs=cp_fixed)
    chart1_title="Chart 1 — Material 1 (Fixed)"
else:
    chart1_title="Chart 1 — Material 1 Feasible Region"
    if mat_sufficient:
        fig1.add_annotation(x=1.65,y=900,text="✅ Material 2 alone<br>is sufficient",
            showarrow=False,font=dict(size=15,color="#1a7a3c"),align="center",
            bgcolor="rgba(220,255,235,0.85)",bordercolor="#5abf7e",borderwidth=1.5,borderpad=12)
    elif L1>0 and Remaining>0:
        bnd=Remaining/(L1*T_RANGE); bnd_cl=np.clip(bnd,Y_MIN,Y_MAX)
        add_boundary_zones(fig1,T_RANGE,bnd_cl)
        if show_scatter:
            add_scatter_overlay(fig1,Remaining,L1,t1_base,
                                rate_for(RM1_pin if RM1_pin else 780),rate_for(RM1_base),axis="t1")
        add_boundary_curve(fig1,T_RANGE,bnd_cl,L1,t1_base,
                           rate_for(RM1_pin if RM1_pin else 780),rate_for(RM1_base),axis="t1")
        if fix_other and t_pin_min is not None and t_pin_min>0:
            add_pin_marker(fig1,t_pin_min,RM1_pin,f"t₁_min @ RM₁={RM1_pin}MPa",
                           mp_kg=mp_pin,cp_rs=cp_pin)

fig1.update_layout(**BASE_LAYOUT,
    title=dict(text=chart1_title,font=dict(size=13,color="#1b3a6b"),x=0),
    xaxis=dict(**AXIS_STYLE,title="Thickness t₁ (mm)",range=[0.75,2.55]),
    yaxis=dict(**AXIS_STYLE,title="Grade RM₁ (MPa)",range=[Y_MIN,Y_MAX]))

# ---------------------------------------------------------------------------
# Build Chart 2
# ---------------------------------------------------------------------------
fig2=go.Figure(); add_grade_lines(fig2)

if fixing_mat1:
    chart2_title="Chart 2 — Material 2 Feasible Region"
    if mat_sufficient:
        fig2.add_annotation(x=1.65,y=900,text="✅ Material 1 alone<br>is sufficient",
            showarrow=False,font=dict(size=15,color="#1a7a3c"),align="center",
            bgcolor="rgba(220,255,235,0.85)",bordercolor="#5abf7e",borderwidth=1.5,borderpad=12)
    elif L2>0 and Remaining>0:
        bnd=Remaining/(L2*T_RANGE); bnd_cl=np.clip(bnd,Y_MIN,Y_MAX)
        add_boundary_zones(fig2,T_RANGE,bnd_cl)
        if show_scatter:
            add_scatter_overlay(fig2,Remaining,L2,t2_base,
                                rate_for(RM2_pin),rate_for(RM2_base),axis="t2")
        add_boundary_curve(fig2,T_RANGE,bnd_cl,L2,t2_base,
                           rate_for(RM2_pin),rate_for(RM2_base),axis="t2")
        if fix_other and t_pin_min is not None and t_pin_min>0:
            add_pin_marker(fig2,t_pin_min,RM2_pin,f"t₂_min @ RM₂={RM2_pin}MPa",
                           mp_kg=mp_pin,cp_rs=cp_pin)
else:
    chart2_title="Chart 2 — Material 2 (Fixed)"
    add_selected_dot(fig2,t2_sel,RM2_sel,f"RM₂={RM2_sel}MPa t₂={t2_sel:.2f}mm",
                     mp_kg=mp_fixed,cp_rs=cp_fixed)

fig2.update_layout(**BASE_LAYOUT,
    title=dict(text=chart2_title,font=dict(size=13,color="#1b3a6b"),x=0),
    xaxis=dict(**AXIS_STYLE,title="Thickness t₂ (mm)",range=[0.75,2.55]),
    yaxis=dict(**AXIS_STYLE,title="Grade RM₂ (MPa)",range=[Y_MIN,Y_MAX]))

# ---------------------------------------------------------------------------
# Render charts
# ---------------------------------------------------------------------------
col1,col2=st.columns(2)
with col1: st.plotly_chart(fig1,use_container_width=True,config={"displayModeBar":False})
with col2: st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

# ---------------------------------------------------------------------------
# Results card
# ---------------------------------------------------------------------------
if mat_sufficient:
    fn=("Material 1" if fixing_mat1 else "Material 2")
    status_badge=f'<span class="badge-ok">✅ {fn.upper()} ALONE SUFFICIENT</span>'
    status_note=f"{fn} alone exceeds CCI target."
    t_min_display=remaining_display="—"
elif fix_other and t_pin_min is not None:
    passes=(0.8<=t_pin_min<=2.5)
    pl=(f"RM₂={RM2_pin}" if fixing_mat1 else f"RM₁={RM1_pin}")
    status_badge=('<span class="badge-pass">✅ PASS</span>' if passes
                  else '<span class="badge-fail">❌ FAIL — t_min out of range</span>')
    status_note=f"With {pl} MPa → t_min={t_pin_min:.3f}mm"
    t_min_display=f"{t_pin_min:.3f} mm"
    remaining_display=f"{Remaining/1000:.3f} ×10³"
else:
    status_badge='<span style="color:#6b7a99;font-size:.85rem;">Pin the other material to compute t_min</span>'
    status_note=""
    t_min_display="—"
    remaining_display="—" if mat_sufficient else f"{Remaining/1000:.3f} ×10³"

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

# ── Mass Penalty ─────────────────────────────────────────────────────────────
fixed_lbl   = "Mat 1" if fixing_mat1 else "Mat 2"
explore_lbl = "Mat 2" if fixing_mat1 else "Mat 1"

def metric_delta(val,unit):
    if val is None: return "—", None
    sign="+" if val>0 else ""
    return f"{sign}{val:.3f} {unit}", val

st.markdown("#### ⚖️ Mass Penalty vs Baseline  "
            "<span style='font-size:.78rem;color:#8896b0;font-weight:400'>"
            "(kg/m — positive=heavier, negative=lighter)</span>",unsafe_allow_html=True)
mc1,mc2,mc3=st.columns(3)
v,d=metric_delta(mp_fixed,"kg/m")
with mc1: st.metric(f"{fixed_lbl} Mass Penalty",v,f"{d:+.3f} kg/m" if d is not None else None,delta_color="inverse")
v,d=metric_delta(mp_pin,"kg/m")
with mc2: st.metric(f"{explore_lbl} Mass Penalty (t_min)",v,f"{d:+.3f} kg/m" if d is not None else None,delta_color="inverse")
v,d=metric_delta(mp_total,"kg/m")
with mc3: st.metric("Total Mass Penalty",v,f"{d:+.3f} kg/m" if d is not None else None,delta_color="inverse")

# ── Cost Penalty ─────────────────────────────────────────────────────────────
st.markdown("#### 💰 Cost Penalty vs Baseline  "
            "<span style='font-size:.78rem;color:#8896b0;font-weight:400'>"
            "(Rs/m — positive=more expensive, negative=cheaper)</span>",unsafe_allow_html=True)
cc1,cc2,cc3=st.columns(3)
v,d=metric_delta(cp_fixed,"Rs/m")
with cc1: st.metric(f"{fixed_lbl} Cost Penalty",v,f"{d:+.2f} Rs/m" if d is not None else None,delta_color="inverse")
v,d=metric_delta(cp_pin,"Rs/m")
with cc2: st.metric(f"{explore_lbl} Cost Penalty (t_min)",v,f"{d:+.2f} Rs/m" if d is not None else None,delta_color="inverse")
v,d=metric_delta(cp_total,"Rs/m")
with cc3: st.metric("Total Cost Penalty",v,f"{d:+.2f} Rs/m" if d is not None else None,delta_color="inverse")

# ===========================================================================
# Multi-Scenario Comparison
# ===========================================================================
st.markdown("---")
st.markdown("### 📋 Multi-Scenario Comparison")

with st.expander("➕ Add current selection as a scenario",expanded=True):
    sc_name=st.text_input("Scenario name",value=f"Scenario {len(st.session_state.scenarios)+1}")
    if st.button("💾 Save scenario"):
        sc=dict(
            name=sc_name,
            mode="Fix Mat1" if fixing_mat1 else "Fix Mat2",
            RM1=RM1_sel if fixing_mat1 else (RM1_pin if fix_other else "—"),
            t1 =t1_sel  if fixing_mat1 else (f"{t_pin_min:.3f}" if (fix_other and t_pin_min) else "—"),
            RM2=RM2_pin if (fixing_mat1 and fix_other) else (RM2_sel if not fixing_mat1 else "—"),
            t2 =f"{t_pin_min:.3f}" if (fixing_mat1 and fix_other and t_pin_min) else (t2_sel if not fixing_mat1 else "—"),
            S_fixed=S_fixed, Remaining=Remaining, CCI_target=CCI_target,
            t_min=t_pin_min if (fix_other and t_pin_min) else None,
            passes=(0.8<=t_pin_min<=2.5) if (fix_other and t_pin_min) else None,
            mat_sufficient=mat_sufficient,
            mp_fixed=mp_fixed, mp_pin=mp_pin, mp_total=mp_total,
            cp_fixed=cp_fixed, cp_pin=cp_pin, cp_total=cp_total,
        )
        st.session_state.scenarios.append(sc)
        st.success(f"'{sc_name}' saved!")

if st.session_state.scenarios:
    rows=[]
    for s in st.session_state.scenarios:
        verdict=("✅ Mat OK" if s["mat_sufficient"] else
                 "✅ PASS"  if s["passes"] is True  else
                 "❌ FAIL"  if s["passes"] is False  else "— (no pin)")
        rows.append({
            "Name":s["name"],"Mode":s["mode"],
            "RM₁":s["RM1"],"t₁":s["t1"],"RM₂":s["RM2"],"t₂":s["t2"],
            "Rem×10³":"—" if s["mat_sufficient"] else f"{s['Remaining']/1000:.3f}",
            "t_min":f"{s['t_min']:.3f}" if s["t_min"] else "—",
            "MP Fix":fmt_plain(s["mp_fixed"],"kg/m"),
            "MP Exp":fmt_plain(s["mp_pin"],"kg/m"),
            "MP Tot":fmt_plain(s["mp_total"],"kg/m"),
            "CP Fix":fmt_plain(s["cp_fixed"],"Rs/m"),
            "CP Exp":fmt_plain(s["cp_pin"],"Rs/m"),
            "CP Tot":fmt_plain(s["cp_total"],"Rs/m"),
            "Verdict":verdict,
        })
    df_sc=pd.DataFrame(rows)

    def _style_col(val,col):
        if col=="Verdict":
            if "PASS" in str(val) or "OK" in str(val): return "background-color:#e6f9ee;color:#1a7a3c;font-weight:700"
            if "FAIL" in str(val): return "background-color:#fdecea;color:#c0392b;font-weight:700"
        if col in("MP Fix","MP Exp","MP Tot","CP Fix","CP Exp","CP Tot"):
            try:
                v=float(str(val).replace("+","").split()[0])
                if v>0: return "color:#c0392b;font-weight:600"
                if v<0: return "color:#1a7a3c;font-weight:600"
            except: pass
        return ""

    styled=df_sc.style
    for col in df_sc.columns:
        try:    styled=styled.map(lambda v,c=col:_style_col(v,c),subset=[col])
        except: styled=styled.applymap(lambda v,c=col:_style_col(v,c),subset=[col])
    st.dataframe(styled,use_container_width=True,hide_index=True)

    cdl,cclr=st.columns(2)
    with cclr:
        if st.button("🗑 Clear all scenarios"):
            st.session_state.scenarios=[]; st.rerun()
    with cdl:
        st.download_button("⬇️ Download (CSV)",df_sc.to_csv(index=False).encode(),
                           "biw_scenarios.csv","text/csv")
else:
    st.info("No scenarios saved yet. Adjust sliders and click Save scenario.")

# ===========================================================================
# PDF Export
# ===========================================================================
def render_chart_mpl(cdata):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAV="#1b3a6b"; ORANGE="#e05c1a"
    fig_m,ax=plt.subplots(figsize=(7,4.8),dpi=150)
    ax.set_facecolor("#f8fafd"); fig_m.patch.set_facecolor("white")
    for g in GRADES_REF:
        ax.axhline(g,color="#8096c8",lw=0.9,ls="--",alpha=0.7)
        ax.text(2.58,g,f"{g}MPa",va="center",ha="left",fontsize=7,color="#5a6a9a")
    if cdata.get("sufficient_msg"):
        ax.text(1.65,900,cdata["sufficient_msg"],ha="center",va="center",fontsize=12,color="#1a7a3c",
                bbox=dict(boxstyle="round,pad=0.5",fc="#dcffe4",ec="#5abf7e",lw=1.2))
    else:
        bnd=cdata.get("boundary")
        if bnd:
            t_arr,rm_arr=bnd
            if cdata.get("pass_fill"):
                ax.fill_between(t_arr,rm_arr,Y_MAX,color="#22a846",alpha=0.10,zorder=1)
                ax.fill_between(t_arr,Y_MIN,rm_arr,color="#c83228",alpha=0.08,zorder=1)
            ax.plot(t_arr,rm_arr,color=ORANGE,lw=2.2,zorder=3,label="Min boundary")
        for (xs,ys,lbl,mc) in [(cdata.get("scatter_pass",[]),"#16a050","PASS combo"),
                                (cdata.get("scatter_fail",[]),"#d23228","FAIL combo")]:
            if xs and isinstance(xs[0],tuple):
                xx,yy=zip(*xs); ax.scatter(xx,yy,c=mc,s=28,alpha=0.75,zorder=4,label=lbl,edgecolors=mc,linewidths=0.5)
        dot=cdata.get("selected_dot")
        if dot:
            x,y,lbl_d,_mp,_cp=dot
            ax.scatter([x],[y],c="#0d8c4e",s=120,zorder=6,edgecolors=NAV,linewidths=1.8,label=lbl_d)
        pin=cdata.get("pin_marker")
        if pin:
            x,y,lbl_p,_mp,_cp=pin
            mc="#0d8c4e" if 0.8<=x<=2.5 else "#c0392b"
            ax.scatter([x],[y],c=mc,s=120,marker="D",zorder=6,edgecolors=NAV,linewidths=1.8,label=lbl_p)
            ax.annotate(f"t_min={x:.2f}mm",(x,y),xytext=(8,4),textcoords="offset points",fontsize=7,color=mc)
    ax.set_xlim(0.75,2.73); ax.set_ylim(Y_MIN,Y_MAX)
    ax.set_xlabel(cdata["xlabel"],fontsize=9,color="#3a4a6a")
    ax.set_ylabel(cdata["ylabel"],fontsize=9,color="#3a4a6a")
    ax.set_title(cdata["title"],fontsize=10,color=NAV,fontweight="bold",pad=8)
    ax.tick_params(labelsize=8,colors="#3a4a6a"); ax.grid(True,color="#dde4f0",lw=0.6,zorder=0)
    for sp in ax.spines.values(): sp.set_edgecolor("#c0cbdd")
    hn,hl=ax.get_legend_handles_labels()
    if hn: ax.legend(hn,hl,fontsize=7,loc="upper right",framealpha=0.9,edgecolor="#d0daea")
    fig_m.tight_layout()
    buf2=io.BytesIO(); fig_m.savefig(buf2,format="png",dpi=150,bbox_inches="tight")
    plt.close(fig_m); buf2.seek(0); return buf2.read()

def build_pdf(c1data,c2data,scenarios,meta):
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,
                          topMargin=18*mm,bottomMargin=18*mm)
    sty=getSampleStyleSheet()
    T=lambda txt,style: Paragraph(txt,style)
    title_s=ParagraphStyle("ti",parent=sty["Heading1"],fontSize=16,
                            textColor=colors.HexColor("#1b3a6b"),spaceAfter=4)
    sub_s  =ParagraphStyle("su",parent=sty["Normal"],fontSize=9,
                            textColor=colors.HexColor("#5a6a8a"),spaceAfter=10)
    body_s =ParagraphStyle("bo",parent=sty["Normal"],fontSize=9,leading=13,
                            textColor=colors.HexColor("#1a1f2e"))
    hdr_s  =ParagraphStyle("hd",parent=sty["Heading2"],fontSize=11,
                            textColor=colors.HexColor("#1b3a6b"),spaceBefore=10,spaceAfter=4)
    story=[]
    story.append(T("BIW Crash Material Selector - Report",title_s))
    story.append(T(f"v1={meta['v1']}->v2={meta['v2']} km/h | R={meta['R']:.3f} | "
                   f"CCI_target={meta['CCI_target']/1000:.3f}x10^3 | EA_target={meta['EA_target']:.2f}kJ",sub_s))
    story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor("#d0daea"),spaceAfter=8))
    story.append(T("Run Parameters",hdr_s))
    pdata=[["Parameter","Value"],
           ["L1 (mm)",f"{meta['L1']:.2f}"],["L2 (mm)",f"{meta['L2']:.2f}"],
           ["CCI Baseline x10^3",f"{meta['CCI_baseline']/1000:.3f}"],
           ["CCI Target x10^3",f"{meta['CCI_target']/1000:.3f}"],
           ["Energy Ratio R",f"{meta['R']:.4f}"],
           ["EA Baseline (kJ)",f"{meta['EA_base']:.2f}"],
           ["EA Target (kJ)",f"{meta['EA_target']:.2f}"],
           ["Steel density (kg/m3)","7850"]]
    ts_p=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1b3a6b")),
                     ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                     ("FONTSIZE",(0,0),(-1,-1),8),
                     ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f4f6f9"),colors.white]),
                     ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d0daea")),
                     ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                     ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)])
    story.append(Table(pdata,colWidths=[90*mm,60*mm],style=ts_p))
    story.append(Spacer(1,6*mm))
    story.append(T("Charts",hdr_s))
    for cd,lbl in[(c1data,"Chart 1 - Material 1"),(c2data,"Chart 2 - Material 2")]:
        story.append(T(lbl,body_s))
        story.append(RLImage(io.BytesIO(render_chart_mpl(cd)),width=168*mm,height=112*mm))
        story.append(Spacer(1,5*mm))
    if scenarios:
        story.append(T("Scenario Comparison",hdr_s))
        hdr=["Name","Mode","RM1","t1","RM2","t2","Rem","t_min",
             "MP Fix","MP Exp","MP Tot","CP Fix","CP Exp","CP Tot","Verdict"]
        td=[hdr]
        for s in scenarios:
            v=("MatOK" if s["mat_sufficient"] else "PASS" if s["passes"] is True
               else "FAIL" if s["passes"] is False else "—")
            td.append([s["name"],s["mode"],str(s["RM1"]),str(s["t1"]),str(s["RM2"]),str(s["t2"]),
                       "—" if s["mat_sufficient"] else f"{s['Remaining']/1000:.3f}",
                       f"{s['t_min']:.3f}" if s["t_min"] else "—",
                       fmt_plain(s["mp_fixed"],""),fmt_plain(s["mp_pin"],""),fmt_plain(s["mp_total"],""),
                       fmt_plain(s["cp_fixed"],""),fmt_plain(s["cp_pin"],""),fmt_plain(s["cp_total"],""),v])
        cw=[w*mm for w in [22,14,12,11,12,11,16,14,16,16,16,16,16,16,14]]
        sc_ts=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1b3a6b")),
                          ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                          ("FONTSIZE",(0,0),(-1,-1),6),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                          ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f4f6f9"),colors.white]),
                          ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#d0daea")),
                          ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)])
        for i,s in enumerate(scenarios,1):
            vc=colors.HexColor("#d4edda") if (s["passes"] or s["mat_sufficient"]) else colors.HexColor("#fdecea")
            sc_ts.add("BACKGROUND",(14,i),(14,i),vc)
            for ci,key in[(8,"mp_fixed"),(9,"mp_pin"),(10,"mp_total"),
                          (11,"cp_fixed"),(12,"cp_pin"),(13,"cp_total")]:
                val=s.get(key)
                if val is not None:
                    sc_ts.add("BACKGROUND",(ci,i),(ci,i),
                              colors.HexColor("#fdecea") if val>0 else colors.HexColor("#d4edda"))
        story.append(Table(td,colWidths=cw,style=sc_ts))
    doc.build(story); buf.seek(0); return buf.read()

st.markdown("---")
st.markdown("### 📄 Export Report")
with st.expander("📄 Generate & Download PDF Report",expanded=False):
    if st.button("📥 Build PDF now"):
        with st.spinner("Building PDF…"):
            meta=dict(v1=v1,v2=v2,R=R,L1=L1,L2=L2,
                      CCI_baseline=CCI_baseline,CCI_target=CCI_target,
                      EA_base=EA_base,EA_target=EA_target)
            def sc_combos(rem,L_s,t_b):
                sp,sf=[],[]
                for gd,t in DATASHEET_COMBOS:
                    (sp if gd*L_s*t>=rem else sf).append((t,gd))
                return sp,sf
            if fixing_mat1:
                c1=dict(title="Chart 1 - Material 1 (Fixed)",xlabel="t1 (mm)",ylabel="RM1 (MPa)",
                        boundary=None,pass_fill=False,sufficient_msg=None,
                        selected_dot=(t1_sel,RM1_sel,f"RM1={RM1_sel}MPa",mp_fixed,cp_fixed),
                        pin_marker=None,scatter_pass=None,scatter_fail=None)
                if mat_sufficient:
                    c2=dict(title="Chart 2 - Material 2",xlabel="t2 (mm)",ylabel="RM2 (MPa)",
                            boundary=None,pass_fill=False,sufficient_msg="Mat1 alone sufficient",
                            selected_dot=None,pin_marker=None,scatter_pass=None,scatter_fail=None)
                else:
                    bt=T_RANGE; by=np.clip(Remaining/(L2*bt),350,1450)
                    sp,sf=sc_combos(Remaining,L2,t2_base) if show_scatter else ([],[])
                    pm=((t_pin_min,RM2_pin,f"t2_min@RM2={RM2_pin}",mp_pin,cp_pin)
                        if (fix_other and t_pin_min and t_pin_min>0) else None)
                    c2=dict(title="Chart 2 - Material 2",xlabel="t2 (mm)",ylabel="RM2 (MPa)",
                            boundary=(bt,by),pass_fill=True,sufficient_msg=None,
                            selected_dot=None,pin_marker=pm,scatter_pass=sp,scatter_fail=sf)
            else:
                c2=dict(title="Chart 2 - Material 2 (Fixed)",xlabel="t2 (mm)",ylabel="RM2 (MPa)",
                        boundary=None,pass_fill=False,sufficient_msg=None,
                        selected_dot=(t2_sel,RM2_sel,f"RM2={RM2_sel}MPa",mp_fixed,cp_fixed),
                        pin_marker=None,scatter_pass=None,scatter_fail=None)
                if mat_sufficient:
                    c1=dict(title="Chart 1 - Material 1",xlabel="t1 (mm)",ylabel="RM1 (MPa)",
                            boundary=None,pass_fill=False,sufficient_msg="Mat2 alone sufficient",
                            selected_dot=None,pin_marker=None,scatter_pass=None,scatter_fail=None)
                else:
                    bt=T_RANGE; by=np.clip(Remaining/(L1*bt),350,1450)
                    sp,sf=sc_combos(Remaining,L1,t1_base) if show_scatter else ([],[])
                    pm=((t_pin_min,RM1_pin,f"t1_min@RM1={RM1_pin}",mp_pin,cp_pin)
                        if (fix_other and t_pin_min and t_pin_min>0) else None)
                    c1=dict(title="Chart 1 - Material 1",xlabel="t1 (mm)",ylabel="RM1 (MPa)",
                            boundary=(bt,by),pass_fill=True,sufficient_msg=None,
                            selected_dot=None,pin_marker=pm,scatter_pass=sp,scatter_fail=sf)
            try:
                pdf_bytes=build_pdf(c1,c2,st.session_state.scenarios,meta)
                st.download_button("⬇️ Download Report PDF",pdf_bytes,
                                   "biw_crash_report.pdf","application/pdf")
                st.success("PDF ready — click above to download.")
            except Exception:
                import traceback
                st.error("PDF generation failed:")
                st.code(traceback.format_exc(),language="text")
