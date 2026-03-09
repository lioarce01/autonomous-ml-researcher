"""
dashboard.py — Streamlit live dashboard for experiment progress.

Run in a separate terminal:
    streamlit run dashboard.py
"""

import time
import os
import streamlit as st
import plotly.graph_objects as go

# Import db from the same directory
import sys
sys.path.insert(0, os.path.dirname(__file__))
import db

st.set_page_config(
    page_title="ATR — Autonomous ML Trainer",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Autonomous ML Trainer — Live Dashboard")

# Check for pause state
paused = os.path.exists(os.path.join(os.path.dirname(__file__), ".pause"))
if paused:
    st.warning("⏸ Agent is PAUSED (`.pause` file exists). Remove it to resume.")
else:
    st.success("▶ Agent is RUNNING")

# Load data
all_exps = db.get_all()
stats = db.get_stats()

if not all_exps:
    st.info("No experiments logged yet. Run the agent to get started.")
    time.sleep(30)
    st.rerun()

# ── Stats row ──────────────────────────────────────────────────────────────
total = stats["total"] or 0
kept = int(stats["kept_count"] or 0)
best = stats["best_val_loss"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Experiments", total)
col2.metric("Improvements (kept)", kept)
col3.metric("Best Val Loss", f"{best:.6f}" if best is not None else "—")
col4.metric("Improvement Rate", f"{kept/total*100:.0f}%" if total > 0 else "—")

st.divider()

# ── Progress chart ─────────────────────────────────────────────────────────
st.subheader("Val Loss Progress")

ids = [e["id"] for e in all_exps]
losses = [e["val_loss"] for e in all_exps]
names = [e["name"] for e in all_exps]
kept_flags = [bool(e["kept"]) for e in all_exps]
notes = [e["notes"] or "" for e in all_exps]

# Running best line
running_best = []
current_best = float("inf")
for loss in losses:
    if loss is not None and loss < current_best:
        current_best = loss
    running_best.append(current_best if current_best < float("inf") else None)

fig = go.Figure()

# Non-improvements (grey)
fig.add_trace(go.Scatter(
    x=[ids[i] for i in range(len(ids)) if not kept_flags[i]],
    y=[losses[i] for i in range(len(losses)) if not kept_flags[i]],
    mode="markers",
    name="No improvement",
    marker=dict(color="#888888", size=8, symbol="circle"),
    text=[names[i] for i in range(len(names)) if not kept_flags[i]],
    hovertemplate="<b>%{text}</b><br>Val Loss: %{y:.6f}<br>%{customdata}",
    customdata=[notes[i] for i in range(len(notes)) if not kept_flags[i]],
))

# Improvements (green)
fig.add_trace(go.Scatter(
    x=[ids[i] for i in range(len(ids)) if kept_flags[i]],
    y=[losses[i] for i in range(len(losses)) if kept_flags[i]],
    mode="markers",
    name="Improvement",
    marker=dict(color="#00cc44", size=10, symbol="circle"),
    text=[names[i] for i in range(len(names)) if kept_flags[i]],
    hovertemplate="<b>%{text}</b><br>Val Loss: %{y:.6f}<br>%{customdata}",
    customdata=[notes[i] for i in range(len(notes)) if kept_flags[i]],
))

# Running best line
fig.add_trace(go.Scatter(
    x=ids,
    y=running_best,
    mode="lines",
    name="Running best",
    line=dict(color="#00cc44", width=2, dash="solid"),
    hoverinfo="skip",
))

fig.update_layout(
    xaxis_title="Experiment #",
    yaxis_title="Val Loss",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    height=400,
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Leaderboard table ──────────────────────────────────────────────────────
st.subheader("All Experiments")

import pandas as pd

df = pd.DataFrame(all_exps)
df = df.sort_values("val_loss", ascending=True).reset_index(drop=True)
df["rank"] = df.index + 1
df["kept"] = df["kept"].apply(lambda x: "✅" if x else "")
df["val_loss"] = df["val_loss"].apply(lambda x: f"{x:.6f}" if x is not None else "—")
df["timestamp"] = df["timestamp"].str[:16]
df = df[["rank", "name", "val_loss", "notes", "timestamp", "kept", "git_hash"]]
df.columns = ["Rank", "Name", "Val Loss", "Notes", "When", "Kept", "Git"]

st.dataframe(df, use_container_width=True, hide_index=True)

# ── Auto-refresh ───────────────────────────────────────────────────────────
st.caption(f"Auto-refreshes every 30s. Last updated: {time.strftime('%H:%M:%S')}")
time.sleep(30)
st.rerun()
