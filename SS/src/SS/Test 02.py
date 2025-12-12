import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go

# Local modules
import config
import utils


# =============================================================
# Streamlit Layout Config
# =============================================================
st.set_page_config(
    page_title=config.APP_TITLE,
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "⚡ Task 4 — Suboptimal Classifier",
        "🌲 Task 5 — RF Feature Importance",
        "📈 Task 6 — LSTM Forecasting"
    ]
)


# =============================================================
# HOME PAGE
# =============================================================
if page == "🏠 Home":
    st.title("🌞 Solar PV Machine Learning Dashboard")
    st.markdown(f"### Version: {config.APP_VERSION}")

    st.markdown("""
    ## 📘 Project Overview
    This dashboard integrates ML models across 4 tasks:

    ### **Task 3 — Model Evaluation Framework**
    - Full pipeline for AC/DC forecasting  
    - Feature engineering  
    - Per-inverter & per-day modelling  
    - NN learning curves  
    - Bias–variance diagnostics  

    ### **Task 4 — Operating Condition Classification**
    - Linear SVC  
    - Optimised threshold (max F1 for Suboptimal class)  
    - Time-aware train/validation/test splits  
    - ALE, drop-column importance  

    ### **Task 5 — Random Forest Feature Importance**
    - Permutation importance  
    - Residual coefficients  
    - PDP slopes  
    - PDP curves  

    ### **Task 6 — LSTM Forecasting**
    - 24-hour window → 1-hour ahead  
    - PyTorch LSTM models  
    - Persistence & moving-average baselines  
    - Interactive visualisation  
    - Live forecasting mode  

    Use the sidebar to navigate.
    """)


# =============================================================
# TASK 4 — SUBOPTIMAL CLASSIFIER
# =============================================================
elif page == "⚡ Task 4 — Suboptimal Classifier":

    st.title("⚡ Suboptimal Operating Condition Classifier")

    st.markdown("""
    This page loads the **Linear SVC model outputs** and provides:
    - User-input prediction  
    - Thresholded Suboptimal/Optimal classification  
    - Model diagnostics  
    """)

    st.header("🔧 Load Classification Results")

    # Load classification results (PKL(s) expected in SVC_RESULTS)
    results_files = [
        f for f in os.listdir(config.SVC_RESULTS)
        if f.endswith(".pkl")
    ]

    if not results_files:
        st.error("No classification result PKL files found.")
    else:
        chosen = st.selectbox("Choose result file:", results_files)
        svc_results_path = os.path.join(config.SVC_RESULTS, chosen)
        svc_data = utils.load_pickle(svc_results_path)
        st.success(f"Loaded {chosen}")

        # Display keys (e.g., thresholds, metrics)
        st.json({k: str(type(v)) for k, v in svc_data.items()})

    st.subheader("🔮 Try a Live Prediction")
    irr = st.number_input("Irradiation", 0.0, 1500.0, 500.0)
    ac = st.number_input("AC Power", 0.0, 5000.0, 2500.0)
    dc = st.number_input("DC Power", 0.0, 5000.0, 2700.0)
    amb = st.number_input("Ambient Temperature", -10.0, 80.0, 35.0)
    mod = st.number_input("Module Temperature", -10.0, 120.0, 55.0)

    # Derived features
    dc_irra = dc / (irr + 1e-3)
    ac_irra = ac / (irr + 1e-3)
    temp_delta = mod - amb

    st.write(f"DC/IRRA = {dc_irra:.3f}")
    st.write(f"AC/IRRA = {ac_irra:.3f}")
    st.write(f"Temp_Delta = {temp_delta:.3f}")

    st.warning("⚠ Classifier prediction interface placeholder — final integration depends on exact SVC pipeline saved.")


# =============================================================
# TASK 5 — RF FEATURE IMPORTANCE
# =============================================================
elif page == "🌲 Task 5 — RF Feature Importance":

    st.title("🌲 Random Forest Feature Importance Explorer")

    st.markdown("""
    This page displays the feature importance outputs computed in Task 5:
    - Permutation importance  
    - Residual coefficients  
    - PDP slopes  
    """)

    # Load RF results
    try:
        rf_data = utils.load_rf_feature_importance()
        st.success("Loaded feature_importance_resultsRF.pkl")
    except Exception as e:
        st.error(f"Could not load RF results: {e}")
        st.stop()

    plant_choice = st.selectbox("Select Plant:", ["Plant1", "Plant2"])
    target_choice = st.selectbox("Target:", ["DC", "AC"])

    results = rf_data.get(plant_choice, {}).get(target_choice, {})

    if not results:
        st.error("No data available for selection.")
        st.stop()

    # Show Permutation Importance
    st.subheader("🔹 Permutation Importance")
    if results["permutation_importance"] is not None:
        st.dataframe(results["permutation_importance"])
    else:
        st.info("No permutation importance available.")

    # Residual Coefficients
    st.subheader("🔹 Residual Coefficients")
    if "residual_coefficients" in results:
        st.dataframe(results["residual_coefficients"])
    else:
        st.info("Residual coefficient data not available.")

    # PDP Slopes
    st.subheader("🔹 PDP Coefficients")
    if "pdp_coefficients" in results:
        st.dataframe(results["pdp_coefficients"])
    else:
        st.info("PDP coefficient data not available.")


# =============================================================
# TASK 6 — LSTM FORECASTING
# =============================================================
elif page == "📈 Task 6 — LSTM Forecasting":

    st.title("📈 LSTM Forecasting Dashboard")

    # Load LSTM results
    try:
        lstm_data = utils.load_lstm_results()
    except Exception as e:
        st.error(f"Failed to load LSTM results: {e}")
        st.stop()

    mode = st.radio("Choose Mode:", ["Precomputed Results", "Live Forecasting"])

    # ---------------------------------------------------------
    # MODE A — PRECOMPUTED RESULTS
    # ---------------------------------------------------------
    if mode == "Precomputed Results":

        st.header("📊 Precomputed LSTM Forecast Results")

        plant = st.selectbox("Plant:", ["Plant1", "Plant2"])
        target = st.selectbox("Target:", ["AC", "DC"])

        key = f"{plant}_{target}"
        plant_data = lstm_data.get(key, {})

        if not plant_data:
            st.error("Data not available.")
            st.stop()

        df = plant_data["per_inverter"]
        st.subheader("Per-Inverter Metrics")
        st.dataframe(df)

        inverter = st.selectbox("Choose inverter:", df.index.tolist())

        # Extract raw results
        raw_list = plant_data["raw_outputs"]
        entry = [e for e in raw_list if e["inverter"] == inverter][0]

        y_true = entry["y_true"]
        y_pred = entry["y_pred_lstm"]
        persist = entry["persist"]
        movavg = entry["movavg"]
        time_index = entry["test_time"]

        # Forecast plot
        fig1 = utils.plot_forecast(
            time_index, y_true, y_pred,
            persist=persist, movavg=movavg,
            title=f"{plant} {target} — {inverter} Forecast"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Error plot
        fig2 = utils.plot_error(
            time_index, y_true, y_pred,
            title="Forecast Error"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------------
    # MODE B — LIVE FORECASTING
    # ---------------------------------------------------------
    else:
        st.header("🔮 Live LSTM Forecasting (1-hour Ahead)")

        plant = st.selectbox("Plant:", ["Plant1", "Plant2"])
        target = st.selectbox("Target:", ["AC", "DC"])

        # Load LSTM result summary to get list of inverters
        key = f"{plant}_{target}"
        plant_data = lstm_data.get(key, {})
        df = plant_data.get("per_inverter", pd.DataFrame())

        if df.empty:
            st.error("No inverters available.")
            st.stop()

        inverter = st.selectbox("Choose inverter:", df.index.tolist())

        st.info("Enter 24 hours (96 rows) of feature data.")

        # Create input place for CSV upload
        uploaded = st.file_uploader("Upload 24-hour CSV file (96 × 5)", type=["csv"])

        if uploaded:
            data = pd.read_csv(uploaded)
            if data.shape != (96, 5):
                st.error("CSV must be exactly 96 rows × 5 columns.")
            else:
                seq = utils.prepare_lstm_sequence(data.values)

                # Load corresponding model
                model, input_dim, hidden_dim = utils.load_lstm_model(
                    plant, target, inverter
                )

                # Run prediction
                pred = utils.lstm_predict(model, seq)

                st.success(f"Predicted {target} Power 1 hour ahead: **{pred:.2f}**")

                st.write("Model Details:")
                st.json({
                    "inverter": inverter,
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim
                })
