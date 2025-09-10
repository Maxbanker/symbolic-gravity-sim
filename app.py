# app.py
import streamlit as st
import numpy as np
from symbolic_gravity_sim.symbolic_collapse_sim import run

st.title("Symbolic Gravity Simulator")
st.markdown("Adjust parameters to visualize agent trajectories in the Ψ_eff field.")

with st.sidebar:
    seed = st.number_input("Seed", min_value=0, value=42)
    n_agents = st.number_input("Number of Agents", min_value=1, value=4)
    steps = st.number_input("Steps", min_value=10, value=300)
    field = st.selectbox("Field Type", ["hillvalley", "gaussian", "uniform", "gradient"], index=0)
    overlay_curvature = st.checkbox("Overlay Curvature", value=False)
    save = st.text_input("Save Prefix", value="run_streamlit")

if st.button("Run Simulation"):
    run(
        seed=seed,
        n_agents=n_agents,
        steps=steps,
        field=field,
        overlay_curvature=overlay_curvature,
        save=save,
        no_show=True  # Prevent plt.show() in Streamlit
    )
    st.image(f"runs/{save}.png")
    with open(f"runs/{save}.json") as f:
        st.json(f.read())