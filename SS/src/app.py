# Streamlit dashboard for Tasks 2–7

import os
import glob
import pickle
import shutil
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import nbformat
import plotly.graph_objects as go

from config import (
    PLANT_CONFIG,
    FEATURES_CONFIG,
    FEATURE_NAMES,
    MODEL_NAMES,
    TASK4_FOLDER,
    TASK4_RESULTS_GLOB,
    TASK5_RF_AC_DC,
    TASK5_RF_SIMPLE,
    TASK6_FOLDER,
    TASK6_LSTM_PKL,
)

from utils import (
    get_inverters_for_plant,
    get_trained_models_for_inverter,
    make_feature_input_defaults,
    compute_feature_importance,
    load_all_results,
    build_metrics_dataframe,
    compute_bias_variance,
    extract_nn_loss_curves,
    train_single_inverter,
    train_all_inverters,
)

# --------------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------------------

def sidebar_controls():
    st.sidebar.title("Controls")

    plant = st.sidebar.selectbox("Plant", list(PLANT_CONFIG.keys()), index=0)
    inverters = get_inverters_for_plant(plant)
    inverter_id = st.sidebar.selectbox("Inverter", inverters, index=0)

    target_side = st.sidebar.radio(
        "Target to predict",
        ["DC power (DC_CLEAN)", "AC power (AC_CLEAN)"],
        index=0,
    )
    target_key = "dc" if target_side.startswith("DC") else "ac"

    # Primary models: exclude NeuralNet (fast retrain may not support it)
    primary_options = [m for m in MODEL_NAMES if m != "NeuralNet"]
    primary_model = st.sidebar.selectbox("Primary model", primary_options, index=0)

    # Compare models: include ALL, including NeuralNet
    compare_models = st.sidebar.multiselect(
        "Models to compare",
        options=MODEL_NAMES,
        default=MODEL_NAMES,
    )

    return plant, inverter_id, target_key, primary_model, compare_models



def sidebar_feature_inputs(feature_ui_cfg):
    st.sidebar.header("Environmental conditions")

    user_values = {}
    for feat in FEATURES_CONFIG:
        name = feat["name"]
        label = feat["label"]
        cfg = feature_ui_cfg.get(name)
        if not cfg:
            continue
        user_values[name] = st.sidebar.slider(
            label,
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]) if cfg["step"] > 0 else 0.1,
        )
    return user_values


# --------------------------------------------------------------------------------------
# PAGE 1 – INTERACTIVE PREDICTIONS (FAST RETRAIN)
# --------------------------------------------------------------------------------------

def page_predictions(
    plant: str,
    inverter_id: str,
    target_key: str,
    primary_model_name: str,
    compare_models: list[str],
):
    st.header("🔮 Interactive Predictions")

    # 1) Train lightweight models for this inverter
    try:
        with st.spinner("Training lightweight models for selected inverter..."):
            train_result = get_trained_models_for_inverter(plant, inverter_id)
    except Exception as e:
        st.error(f"Could not train fast models for {plant} – {inverter_id}.\n\n{e}")
        return

    models_dc = train_result["models_dc"]
    models_ac = train_result["models_ac"]
    metrics_dc = train_result["metrics_dc"]
    metrics_ac = train_result["metrics_ac"]
    feature_stats = train_result["feature_stats"]

    # 2) Sidebar sliders from feature stats
    feature_ui_cfg = make_feature_input_defaults(feature_stats)
    user_inputs = sidebar_feature_inputs(feature_ui_cfg)

    if not user_inputs:
        st.error("No feature inputs available.")
        return

    X_user = np.array([[user_inputs[f] for f in FEATURE_NAMES]], dtype=float)

    if target_key == "dc":
        models_for_target = models_dc
        metrics_for_target = metrics_dc
        target_label = "DC_CLEAN"
    else:
        models_for_target = models_ac
        metrics_for_target = metrics_ac
        target_label = "AC_CLEAN"

    st.markdown(
        f"**Plant:** `{plant}` · **Inverter:** `{inverter_id}` · **Target:** `{target_label}`"
    )

    st.markdown("---")
    st.subheader("Primary model prediction")

    if primary_model_name not in models_for_target:
        st.error(f"Primary model `{primary_model_name}` not available.")
        st.write("Available models:", ", ".join(sorted(models_for_target.keys())))
        return

    primary_model = models_for_target[primary_model_name]
    y_pred_primary = float(primary_model.predict(X_user)[0])

    col_pred, col_info = st.columns([2, 1])
    with col_pred:
        st.metric(
            label=f"{target_label} — {primary_model_name}",
            value=f"{y_pred_primary:,.3f}",
        )
        st.caption("Current input values")
        st.write(pd.DataFrame([user_inputs]))
    with col_info:
        st.markdown("**Test metrics (fast retrain)**")
        m = metrics_for_target.get(primary_model_name, {})
        if m:
            st.write(f"- RMSE: `{m['rmse']:.4f}`")
            st.write(f"- MAE: `{m['mae']:.4f}`")
        else:
            st.write("No metrics available.")

    # 3) Comparison plot across models
    st.markdown("---")
    st.subheader("Model comparison for current conditions")

    compare_models = compare_models or [primary_model_name]              # <---------------------------------------

    available = set(models_for_target.keys())
    missing = [m for m in compare_models if m not in available]

    if missing:
        st.info(
            "These models are only available in the full Task 3 training "
            "(no fast retrain for this tab): "
            + ", ".join(missing)
        )

    rows = []
    for name in compare_models:
        if name not in available:
            continue
        model = models_for_target[name]
        y_hat = float(model.predict(X_user)[0])
        row = {"Model": name, "Prediction": y_hat}
        m = metrics_for_target.get(name, {})
        row["RMSE"] = m.get("rmse", np.nan)
        row["MAE"] = m.get("mae", np.nan)
        rows.append(row)                                                 # <---------------------------------------

    if rows:
        df_compare = pd.DataFrame(rows)
        col1, col2 = st.columns(2)

        with col1:
            fig_pred = px.bar(
                df_compare,
                x="Model",
                y="Prediction",
                text="Prediction",
                title=f"Predicted {target_label} by model",
            )
            fig_pred.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_pred.update_layout(yaxis_title="Prediction")
            st.plotly_chart(fig_pred, use_container_width=True)

        with col2:
            fig_rmse = px.bar(
                df_compare,
                x="Model",
                y="RMSE",
                text="RMSE",
                title=f"Test RMSE (fast retrain) — {target_label}",
            )
            fig_rmse.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_rmse.update_layout(yaxis_title="RMSE")
            st.plotly_chart(fig_rmse, use_container_width=True)

        st.write("Detailed table")
        st.dataframe(df_compare.set_index("Model"))
    else:
        st.info("No models selected for comparison.")

    # 4) Feature importance for primary model
    st.markdown("---")
    st.subheader("📊 Feature importance / coefficients")

    importance = compute_feature_importance(primary_model, FEATURE_NAMES)
    if not importance:
        st.info("Feature importances not available for this model.")
    else:
        df_imp = (
            pd.DataFrame(
                {
                    "Feature": list(importance.keys()),
                    "Importance": list(importance.values()),
                }
            )
            .sort_values("Importance", ascending=False)
        )

        fig_imp = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Feature importance / |coefficients| — {primary_model_name}",
        )
        fig_imp.update_layout(xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.write("Numeric values")
        st.dataframe(df_imp.set_index("Feature"))

    st.caption(
        "Note: This tab retrains simpler models quickly for interactive predictions. "
        "The full Task 3 training (with parallel metrics, NN diagnostics, etc.) is "
        "visualised in the other tabs from your pre-computed *_results.pkl files."
    )


# --------------------------------------------------------------------------------------
# PAGE 2 – TASK 3 RESULTS (FROM *_results.pkl)
# --------------------------------------------------------------------------------------

def page_task3_results(plant: str, compare_models: list[str]):    # <------------------------------------------------
    st.header("📊 Task 3 — Full Inverter Results (from *_results.pkl)")

    try:
        results = load_all_results(plant)
    except Exception as e:
        st.error(f"Error loading Task 3 results for {plant}:\n\n{e}")
        return

    if not results:
        st.warning("No *_results.pkl files found for this plant. Retrain first.")
        return

    df_metrics = build_metrics_dataframe(results) # <======================================================================

    # Only keep selected models, if any selection given
    if compare_models:
        allowed_models = set(compare_models)

        # Map UI names → names used in Task 3 results
        if "LinearRegression" in allowed_models:
            allowed_models.add("Linear")  # Task 3 uses "Linear"

        # (add any other mappings here if needed)

        df_metrics = df_metrics[df_metrics["model"].isin(allowed_models)] # <======================================================================


    st.subheader("Combined RMSE/MAE per model")

    inv_options = ["All inverters"] + sorted(df_metrics["inverter"].unique())
    inv_choice = st.selectbox("Scope", inv_options, index=0)

    if inv_choice == "All inverters":
        df_summary = (
            df_metrics.groupby(["model", "side"])
            .agg(rmse=("rmse", "mean"), mae=("mae", "mean"))
            .reset_index()
        )
        st.write("Average across all inverters")
    else:
        df_summary = df_metrics[df_metrics["inverter"] == inv_choice].copy()
        st.write(f"Metrics for inverter `{inv_choice}`")

    st.dataframe(df_summary)

    col1, col2 = st.columns(2)
    with col1:
        fig_rmse = px.bar(
            df_summary,
            x="model",
            y="rmse",
            color="side",
            barmode="group",
            title="RMSE by model & side",
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col2:
        fig_mae = px.bar(
            df_summary,
            x="model",
            y="mae",
            color="side",
            barmode="group",
            title="MAE by model & side",
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    # Bias–variance
    st.markdown("---")
    st.subheader("Bias–Variance proxies (from parallel per-day training)")

    df_dc, df_ac = compute_bias_variance(results) # <======================================================

    if compare_models:
        allowed_models = set(compare_models)
        if "LinearRegression" in allowed_models:
            allowed_models.add("Linear")

        df_dc = df_dc[df_dc["model"].isin(allowed_models)]
        df_ac = df_ac[df_ac["model"].isin(allowed_models)]  # <======================================================



    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**DC side**")
        if df_dc.empty:
            st.info("No DC parallel metrics found.")
        else:
            st.dataframe(df_dc.set_index("model"))
            fig_dc = px.scatter(
                df_dc,
                x="variance",
                y="bias",
                text="model",
                title="Bias–variance proxy — DC",
            )
            fig_dc.update_traces(textposition="top center")
            fig_dc.update_layout(xaxis_title="Variance proxy", yaxis_title="Bias proxy")
            st.plotly_chart(fig_dc, use_container_width=True)

    with col4:
        st.markdown("**AC side**")
        if df_ac.empty:
            st.info("No AC parallel metrics found.")
        else:
            st.dataframe(df_ac.set_index("model"))
            fig_ac = px.scatter(
                df_ac,
                x="variance",
                y="bias",
                text="model",
                title="Bias–variance proxy — AC",
            )
            fig_ac.update_traces(textposition="top center")
            fig_ac.update_layout(xaxis_title="Variance proxy", yaxis_title="Bias proxy")
            st.plotly_chart(fig_ac, use_container_width=True)

    # NN loss curves
    st.markdown("---")   # <----------------------------------------------------------------------
    st.subheader("Neural Network training loss (mean DC vs AC)")

    # Optional: only show if NeuralNet is selected
    NN_NAME = "NeuralNet"
    if compare_models and NN_NAME not in compare_models:
        st.info(
            f"Include `{NN_NAME}` in **Models to compare** to view the NN loss curves."
        )
        return

    loss_dc, loss_ac = extract_nn_loss_curves(results)

    if not loss_dc and not loss_ac:
        st.info("No NN loss curves found in nn_diag.")
        return         # <----------------------------------------------------------------------


    curves_dc = list(loss_dc.values())
    curves_ac = list(loss_ac.values())

    max_len_dc = max(len(c) for c in curves_dc) if curves_dc else 0
    max_len_ac = max(len(c) for c in curves_ac) if curves_ac else 0
    L = min(max_len_dc, max_len_ac) if max_len_dc and max_len_ac else max(
        max_len_dc, max_len_ac
    )

    mean_dc = None
    mean_ac = None

    if curves_dc:
        mat_dc = np.full((len(curves_dc), max_len_dc), np.nan)
        for i, c in enumerate(curves_dc):
            mat_dc[i, : len(c)] = c
        mean_dc = np.nanmean(mat_dc, axis=0)[:L]

    if curves_ac:
        mat_ac = np.full((len(curves_ac), max_len_ac), np.nan)
        for i, c in enumerate(curves_ac):
            mat_ac[i, : len(c)] = c
        mean_ac = np.nanmean(mat_ac, axis=0)[:L]

    epochs = np.arange(L)
    data_loss = []
    if mean_dc is not None:
        data_loss.append(pd.DataFrame({"epoch": epochs, "loss": mean_dc, "side": "DC"}))
    if mean_ac is not None:
        data_loss.append(pd.DataFrame({"epoch": epochs, "loss": mean_ac, "side": "AC"}))


####################################################################################################
    if data_loss:
        df_loss = pd.concat(data_loss, ignore_index=True)

        # No Streamlit slider anymore – always use full range of epochs
        max_epoch = int(df_loss["epoch"].max())
        df_loss_plot = df_loss.copy()

        if df_loss_plot.empty:
            st.info("No NN loss curves found in nn_diag.")
            return

        sides = df_loss_plot["side"].unique()
        start_epoch = int(df_loss_plot["epoch"].min())

        # 1) Base figure + initial traces
        fig_loss = go.Figure()
        for side in sides:
            df0 = df_loss_plot[
                (df_loss_plot["side"] == side)
                & (df_loss_plot["epoch"] <= start_epoch)
            ]
            fig_loss.add_trace(
                go.Scatter(
                    x=df0["epoch"],
                    y=df0["loss"],
                    mode="lines",
                    name=side,
                )
            )

        # Fixed y-axis for stability
        y_max = float(df_loss_plot["loss"].max()) * 1.05

        fig_loss.update_layout(
            title="Mean NN training loss — DC vs AC",
            xaxis_title="epoch",
            yaxis_title="loss",
            xaxis=dict(range=[0, start_epoch], autorange=False),
            yaxis=dict(range=[0, y_max], autorange=False),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=1.0,
                    y=1.15,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 40, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
        )

        # 2) Frames: extend lines and x-axis up to each epoch
        frames = []
        for e in range(start_epoch + 1, max_epoch + 1):
            frame_data = []
            for side in sides:
                dfs = df_loss_plot[
                    (df_loss_plot["side"] == side)
                    & (df_loss_plot["epoch"] <= e)
                ]
                frame_data.append(
                    go.Scatter(
                        x=dfs["epoch"],
                        y=dfs["loss"],
                        mode="lines",
                        name=side,
                    )
                )

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(e),
                    layout=go.Layout(
                        xaxis=dict(range=[0, e], autorange=False),
                        yaxis=dict(range=[0, y_max], autorange=False),
                    ),
                )
            )

        fig_loss.frames = frames

        # 3) White Plotly slider at the BOTTOM of the plot
        steps = []
        for e in range(start_epoch, max_epoch + 1):
            steps.append(
                dict(
                    method="animate",
                    args=[
                        [str(e)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                    label=str(e),
                )
            )

        sliders = [
            dict(
                active=len(steps) - 1,
                currentvalue={"prefix": "Epoch: "},
                pad={"t": 0, "b": 100},
                x=0.0,
                xanchor="left",
                len=1.0,
                y=0,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.05)",   # 20% opacity white
                # bordercolor="rgba(255, 255, 255, 0.2)",  # optional
                steps=steps,
            )
        ]

        fig_loss.update_layout(
            sliders=sliders,
            margin=dict(t=80, b=160),
        )

        st.plotly_chart(fig_loss, use_container_width=True)

####################################################################################################
# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Task 7 – PV Plant Model Dashboard",
    layout="wide",
)

plant, inverter_id, target_key, primary_model, compare_models = sidebar_controls()

tabs = st.tabs(
    [
        "Predictions",
        "Task 3 Results",
    ]
)

with tabs[0]:
    page_predictions(plant, inverter_id, target_key, primary_model, compare_models)

with tabs[1]:                                        # <-------------------------------------------------------------
    page_task3_results(plant, compare_models)


# def main():
#     st.set_page_config(
#         page_title="Task 7 – PV Plant Model Dashboard",
#         layout="wide",
#     )

#     plant, inverter_id, target_key, primary_model, compare_models = sidebar_controls()

#     tabs = st.tabs(
#         [
#             "Predictions",
#             "Task 3 Results",
#         ]
#     )

#     with tabs[0]:
#         page_predictions(plant, inverter_id, target_key, primary_model, compare_models)

#     with tabs[1]:
#         page_task3_results(plant)


# if __name__ == "__main__":
#     main()
