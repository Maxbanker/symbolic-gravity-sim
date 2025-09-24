# app.py — Streamlit UI for Symbolic Gravity Simulator (11D-enabled)
from __future__ import annotations

import os
import json
import streamlit as st

from symbolic_gravity_sim.symbolic_collapse_sim import run

st.set_page_config(page_title="Symbolic Gravity Simulator (11D)", layout="centered")
st.title("🌀 Symbolic Gravity Simulator — 11D Upgrade")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Seed", min_value=0, value=42)
    n_agents = st.number_input("Number of Agents", min_value=1, value=4)
    steps = st.number_input("Steps", min_value=10, value=300)
    field = st.selectbox("Field Type", ["hillvalley", "gaussian", "uniform", "gradient"], index=0)
    overlay_curvature = st.checkbox("Overlay Curvature", value=False)

    st.markdown("### 11D")
    dump_11d = st.checkbox("Save 11D means CSV", value=True)

    st.markdown("---")
    save = st.text_input("Save Prefix (under runs/)", value="run_streamlit")
    st.caption("Outputs will be saved to `runs/<prefix>.png/.json[/_11d_means.csv]`")

# Ensure output directory exists
os.makedirs("runs", exist_ok=True)

# -----------------------------
# Run button
# -----------------------------
if st.button("Run Simulation", type="primary"):
    prefix = f"runs/{save}"

    # Execute simulation (no GUI window)
    run(
        seed=int(seed),
        n_agents=int(n_agents),
        steps=int(steps),
        field=str(field),
        overlay_curvature=bool(overlay_curvature),
        dump_11d_means=bool(dump_11d),
        save=prefix,
        no_show=True,
    )

    # Display image
    img_path = f"{prefix}.png"
    if os.path.exists(img_path):
        st.image(img_path, caption=os.path.basename(img_path))
    else:
        st.warning(f"Image not found at {img_path}")

    # Display JSON (pretty)
    json_path = f"{prefix}.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                st.json(json.load(f))
            except Exception:
                f.seek(0)
                st.json(f.read())
    else:
        st.warning(f"JSON not found at {json_path}")

    # CSV download (11D means)
    if dump_11d:
        csv_path = f"{prefix}_11d_means.csv"
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as fh:
                st.download_button(
                    "Download 11D means",
                    data=fh,
                    file_name=f"{save}_11d_means.csv",
                    mime="text/csv",
                )
        else:
            st.info("11D means CSV not found (it will appear after a successful run with the checkbox enabled).")
