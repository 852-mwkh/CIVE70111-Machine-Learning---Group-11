# ============================================================
# Solar PV Machine Learning Dashboard - Single File Streamlit App
# Author: ChatGPT (based on your specifications)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os

import plotly.express as px
import plotly.graph_objects as go

# Local modules
import config
import utils


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="☀️",
    layout="wide"
)

# Sidebar Title
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    [
        "🏠 Home",
        "⚙️ Task 4 – Suboptimal Classifier",
        "🌲 Task 5 – Random Forest Analysis",
        "📈 Task 6 – LSTM Forecasting"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":
    st.title("☀️ Solar PV Machine Learning Dashboard")
    st.subheader("Unified Interface for Classification, Forecasting & Feature Analysis")

    st.markdown("""
    ---
    ## **📘 Project Overview**
    This dashboard provides:
    - **Task 4** – Suboptimal operating condition detection (Linear SVC)
    - **Task 5** – Random Forest feature importance, PDP, and SHAP analysis  
    - **Task 6** – LSTM forecasting (with live forecasting + precomputed results)

    Models are trained on two solar PV plants using SCADA and weather datasets.

    ---

    ## **📂 Data Pipeline Summary**
    - Timestamp correction  
    - Irradiance cleaning (day/night rules)  
    - AC/DC cleaning and interpolation  
    - Daily & total yield reconstruction  
    - Feature engineering:
        - Efficiency ratios  
        - Temp deltas  
        - Cyclical time encoding  

    ---

    ## **📦 Saved Artifacts Used by This App**
    - **SVC decision models & ALE plots**  
    - **Random Forest feature importance PKL**  
    - **LSTM forecasting PKL + `.pt` weights per inverter**

    Navigate using the sidebar to explore each model.
    """)

# ============================================================
# TASK 4 – SUBOPTIMAL CLASSIFIER
# ============================================================

elif page == "⚙️ Task 4 – Suboptimal Classifier":

    st.title("⚙️ Suboptimal Condition Classification – Task 4")

    st.markdown("""
    This section allows you to:
    - Review SVC classifier outputs  
    - Run predictions manually  
    - Explore PR-AUC, F1, confusion matrices  
    """)

    st.info("⚠️ NOTE: This dashboard uses *precomputed SVC results* only, not live model loading.")

    # Try to load SVC results (if available)
    try:
        svc_results = utils.load_pickle(os.path.join(config.SVC_RESULTS, "SVC_results.pkl"))
        st.success("Loaded SVC results successfully.")
    except:
        st.warning("SVC results file not found — showing demo interface only.")
        svc_results = None

    st.subheader("🔍 Manual Prediction (Demo UI)")

    irr = st.number_input("Irradiation", 0.0, 2000.0, 500.0)
    ac = st.number_input("AC Power", 0.0, 3000.0, 500.0)
    dc = st.number_input("DC Power", 0.0, 3000.0, 500.0)
    amb = st.number_input("Ambient Temperature", -10.0, 60.0, 25.0)
    mod = st.number_input("Module Temperature", -10.0, 90.0, 35.0)

    if st.button("Predict Operating Condition"):
        st.info("⚠ Classifier not loaded — this is a UI preview.")
        st.write("Prediction: **Optimal (0)** (demo)")

    st.subheader("📊 Classification Results")
    if svc_results is not None:
        st.write(svc_results)
    else:
        st.write("No results available.")

# ============================================================
# TASK 5 – RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

elif page == "🌲 Task 5 – Random Forest Analysis":
    st.title("🌲 Random Forest Feature Importance – Task 5")

    # Load PKL
    st.subheader("📁 Loading Feature Importance Data...")
    try:
        rf_results = utils.load_rf_feature_importance()
        st.success("Random Forest feature importance loaded.")
    except Exception as e:
        st.error(f"Could not load RF importance data: {e}")
        st.stop()

    plant = st.selectbox("Select Plant", ["Plant1", "Plant2"])
    target = st.selectbox("Select Target", ["DC", "AC"])

    df_pi = rf_results[plant][target]["permutation_importance"]
    df_residual = rf_results[plant][target].get("residual_coefficients")
    df_pdp = rf_results[plant][target].get("pdp_coefficients")

    st.subheader("📌 Permutation Importance")
    st.dataframe(df_pi)

    fig_pi = px.bar(
        df_pi.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Permutation Importance"
    )
    st.plotly_chart(fig_pi, use_container_width=True)

    if df_residual is not None:
        st.subheader("📌 Residual Coefficients")
        st.dataframe(df_residual)

        fig_res = px.bar(
            df_residual.sort_values("abs_beta"),
            x="abs_beta",
            y="feature",
            orientation="h",
            title="Residual Coefficient Magnitude"
        )
        st.plotly_chart(fig_res, use_container_width=True)

    if df_pdp is not None:
        st.subheader("📌 PDP Slope Importance")
        st.dataframe(df_pdp)

        fig_pdp = px.bar(
            df_pdp.sort_values("abs_pdp_slope"),
            x="abs_pdp_slope",
            y=df_pdp.index,
            orientation="h",
            title="PDP Slope Importance"
        )
        st.plotly_chart(fig_pdp, use_container_width=True)

# ============================================================
# TASK 6 – LSTM FORECASTING
# ============================================================

elif page == "📈 Task 6 – LSTM Forecasting":

    st.title("📈 LSTM 1-Hour Ahead Forecasting – Task 6")

    # Load results
    try:
        lstm_results = utils.load_lstm_results()
        st.success("Loaded LSTM forecasting results.")
    except Exception as e:
        st.error(f"Could not load LSTM results: {e}")
        st.stop()

    st.sidebar.subheader("⚙ LSTM Controls")

    plant = st.sidebar.selectbox("Choose Plant", ["Plant1", "Plant2"])
    target = st.sidebar.selectbox("Choose Target", ["AC", "DC"])

    key_base = f"{plant}_{target}"
    per_inv = lstm_results[key_base]["per_inverter"]
    raw_list = lstm_results[key_base]["raw_outputs"]

    st.subheader("📊 Per-Inverter Forecasting Performance")
    st.dataframe(per_inv)

    inv = st.selectbox("Select Inverter", per_inv.index.tolist())

    # Fetch raw data for this inverter
    raw_entry = [x for x in raw_list if x["inverter"] == inv][0]

    y_true = raw_entry["y_true"]
    y_pred = raw_entry["y_pred_lstm"]
    persist = raw_entry["persist"]
    movavg = raw_entry["movavg"]
    t = raw_entry["test_time"]

    # Forecast Plot
    st.subheader("📈 Forecast vs Actual")
    fig = utils.plot_forecast(t, y_true, y_pred, persist, movavg,
                              title=f"{plant} – {inv} – {target} Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # Error plot
    st.subheader("📉 Error Plot")
    fig2 = utils.plot_error(t, y_true, y_pred, title="Forecast Error")
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------------
    # LIVE FORECASTING SECTION
    # -----------------------------------------------------------

    st.header("🔮 Live 1-Hour Forecast (LSTM)")

    st.markdown("""
    Provide the **last 24 hours (96 steps)** of inverter history.
    Each row must contain these 5 features:
    - AC_CLEAN  
    - DC_CLEAN  
    - IRRADIATION_CLEAN  
    - AMBIENT_TEMPERATURE  
    - MODULE_TEMPERATURE  
    """)

    uploaded = st.file_uploader("Upload 96×5 CSV for Live Forecast", type=["csv"])

    if uploaded:
        seq_df = pd.read_csv(uploaded)

        if seq_df.shape != (96, 5):
            st.error("❌ CSV must be exactly shape (96, 5).")
            st.stop()

        # Load LSTM model
        model, input_dim, hidden_dim = utils.load_lstm_model(plant, target, inv)

        # Prepare sequence
        seq_tensor = utils.prepare_lstm_sequence(seq_df.values)

        # Predict
        pred = utils.lstm_predict(model, seq_tensor)

        st.success(f"🌞 **Predicted {target} Power (1h ahead): {pred:.3f}**")

