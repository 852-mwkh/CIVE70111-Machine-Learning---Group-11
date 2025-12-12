# ================================================================
# Utilities.py  (Unified Utilities for Plant 1 & 2 pipelines)
# ================================================================
import matplotlib
matplotlib.use("Agg")      # MUST come before pyplot is imported

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import datetime as dt
import time
import glob

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# -------------------------------------------------------------------
#                       RAW FILE LOADING
# -------------------------------------------------------------------

def load_raw_files(folder: str):
    """Load the 4 CSV datasets from the folder."""
    plant_1 = pd.read_csv(os.path.join(folder, "Plant_1_Generation_Data_updated.csv"))
    weather_1 = pd.read_csv(os.path.join(folder, "Plant_1_Weather_Sensor_Data.csv"))
    plant_2 = pd.read_csv(os.path.join(folder, "Plant_2_Generation_Data.csv"))
    weather_2 = pd.read_csv(os.path.join(folder, "Plant_2_Weather_Sensor_Data.csv"))
    return plant_1, weather_1, plant_2, weather_2


# ===================================================================
#     ML + FEATURE ENGINEERING UTILITIES (Aligned with Cleaned Data)
# ===================================================================

def add_time_features(df: pd.DataFrame):
    """Add time-of-day + cyclical encodings to cleaned dataframe."""
    df = df.copy()
    df["HOUR"] = df.index.hour
    df["DAY_OF_WEEK"] = df.index.dayofweek

    df["HOUR_SIN"] = np.sin(2 * np.pi * df["HOUR"] / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["HOUR"] / 24)

    return df


def build_X_y(df_clean: pd.DataFrame, target_col: str):
    """
    Build feature matrix X and target y from CLEANED df_ps1 inverter data.
    """
    df = df_clean.copy()

    # remove nighttime zero-irradiation rows if predicting AC/DC
    if "IRRADIATION_CLEAN" in df.columns:
        df = df[df["IRRADIATION_CLEAN"] > 0]

    # add time features (index must be datetime)
    df = add_time_features(df)

    feature_cols = [
        c for c in [
            "IRRADIATION_CLEAN",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "NUM_OPT",
            "NUM_SUBOPT",
            "HOUR", "DAY_OF_WEEK", "HOUR_SIN", "HOUR_COS"
        ] if c in df.columns
    ]

    X = df[feature_cols].values
    y = df[target_col].values

    return X, y, feature_cols


def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1)": Ridge(alpha=1.0),
        "RandomForest(200)": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }


def train_models(X_train, y_train):
    models = get_models()
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def evaluate_models(fitted, X_train, y_train, X_test, y_test):
    results = []

    for name, model in fitted.items():
        pred_tr = model.predict(X_train)
        pred_te = model.predict(X_test)

        results.append({
            "model": name,
            "train_RMSE": mean_squared_error(y_train, pred_tr, squared=False),
            "test_RMSE": mean_squared_error(y_test, pred_te, squared=False),
            "train_MAE": mean_absolute_error(y_train, pred_tr),
            "test_MAE": mean_absolute_error(y_test, pred_te),
            "train_R2": r2_score(y_train, pred_tr),
            "test_R2": r2_score(y_test, pred_te),
        })

    return results


def plot_learning_curve(model, X, y, title):
    sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=5,
        scoring="neg_root_mean_squared_error",
        shuffle=True,
        random_state=42,
    )

    train_rmse = -np.mean(train_scores, axis=1)
    test_rmse = -np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(sizes, train_rmse, marker="o", label="Train RMSE")
    plt.plot(sizes, test_rmse, marker="s", label="Validation RMSE")
    plt.xlabel("Training Size")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_mlp(X_train, y_train):
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    return mlp


def plot_loss_curve(mlp_model, title):
    if not hasattr(mlp_model, "loss_curve_"):
        print("MLP model has no loss curve.")
        return

    plt.figure()
    plt.plot(mlp_model.loss_curve_)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compress_cleaned_15min(df: pd.DataFrame):
    """
    Safe 15-minute resampling for CLEANED inverter data.
    """
    df = df.copy()
    df = df.sort_values(df.index.name)

    # Resample to 15 min using first observation within window
    df_15 = df.resample("15T").first()

    # Forward-fill ONLY yield columns (if needed)
    for col in ["DAILY_YIELD_CLEAN", "TOTAL_YIELD_CLEAN"]:
        if col in df_15.columns:
            df_15[col] = df_15[col].fillna(method="ffill")

    return df_15


# ===============================================================
#                  PLANT 1 PIPELINE
# ===============================================================

def fix_plant_1_datetime(plant_1_raw: pd.DataFrame) -> pd.DataFrame:
    df = plant_1_raw.copy()

    start = pd.Timestamp('2020-05-15')
    end = pd.Timestamp('2020-06-18')

    df['parsed'] = pd.to_datetime(
        df['DATE_TIME'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )

    invalid = df['parsed'].isna() | (~df['parsed'].between(start, end))

    df.loc[invalid, 'parsed'] = pd.to_datetime(
        df.loc[invalid, 'DATE_TIME'],
        format='%Y-%d-%m %H:%M:%S',
        errors='coerce'
    )

    df['DATE_TIME'] = df['parsed']
    return df.drop(columns=['parsed'])


def preprocess_plant_1(plant_1_df: pd.DataFrame):
    df = plant_1_df.copy()
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

    df = df.drop(columns=['day'], errors='ignore')
    df.set_index('DATE_TIME', inplace=True)

    print("Plant 1 missing values:\n", df.isnull().sum())
    print("Plant 1 shape:", df.shape)

    p1_gp = df.groupby('SOURCE_KEY')
    inv_1 = {sk: g for sk, g in p1_gp}
    source_key_1 = df['SOURCE_KEY'].unique().tolist()

    print("Number of inverters:", len(source_key_1))
    print("Source keys:", source_key_1)

    return df, inv_1, source_key_1


def check_missing_per_inverter(inv_1: dict):
    for sk, df in inv_1.items():
        print(f"\nInverter {sk} missing values:")
        print(df.isnull().sum())
        print("Shape:", df.shape)


def check_constancy(inv_1: dict, source_key_1: list):
    cols = ['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']
    for sk in source_key_1:
        g = inv_1[sk].groupby('DATE_TIME')
        check = g[cols].nunique() == 1
        not_constant = (~check).sum()
        print(f"\nConstancy check for {sk}:")
        print(not_constant)


def aggregate_inverters(inv_1: dict) -> dict:
    agg_inv_1 = {}

    for sk, df in inv_1.items():
        agg_df = df.groupby('DATE_TIME').agg(
            PLANT_ID=('PLANT_ID', 'first'),
            SOURCE_KEY=('SOURCE_KEY', 'first'),
            DC_POWER=('DC_POWER', 'first'),
            AC_POWER=('AC_POWER', 'first'),
            DAILY_YIELD=('DAILY_YIELD', 'first'),
            TOTAL_YIELD=('TOTAL_YIELD', 'first'),
            NUM_OPT=('Operating_Condition', lambda x: (x == 'Optimal').sum()),
            NUM_SUBOPT=('Operating_Condition', lambda x: (x == 'Suboptimal').sum())
        ).reset_index()

        agg_inv_1[sk] = agg_df

    return agg_inv_1


def preprocess_weather_1(weather_1_raw: pd.DataFrame) -> pd.DataFrame:
    df = weather_1_raw.copy()
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

    df = df.drop(columns=['PLANT_ID', 'SOURCE_KEY'], errors='ignore')
    df.set_index('DATE_TIME', inplace=True)

    print("\nWeather missing values:\n", df.isnull().sum())
    return df


def clean_irradiation(weather_1_raw: pd.DataFrame) -> pd.DataFrame:
    df = weather_1_raw.copy()
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

    day_start = dt.time(6, 0)
    day_end = dt.time(18, 30)

    df['expected_day'] = df['DATE_TIME'].dt.time.between(day_start, day_end)

    df['IRRADIATION_CLEAN'] = df['IRRADIATION'].copy()
    df.loc[(~df['expected_day']) & (df['IRRADIATION'] > 0), 'IRRADIATION_CLEAN'] = 0
    df.loc[(df['expected_day']) & (df['IRRADIATION'] == 0), 'IRRADIATION_CLEAN'] = np.nan

    df['IRRADIATION_CLEAN'] = df['IRRADIATION_CLEAN'].interpolate()
    df['IRRADIATION_CLEAN'] = df['IRRADIATION_CLEAN'].fillna(0)

    df = df.set_index('DATE_TIME')
    return df.drop(columns=['SOURCE_KEY'], errors='ignore')


def print_time_differences(agg_inv_1: dict, s1_c: pd.DataFrame):
    for sk, df in agg_inv_1.items():
        df1 = df.set_index('DATE_TIME')
        diff_1_not_2 = df1.index.difference(s1_c.index)
        diff_2_not_1 = s1_c.index.difference(df1.index)

        print(f"\n{sk}:")
        print("  In inverter but not weather:", len(diff_1_not_2))
        print("  In weather but not inverter:", len(diff_2_not_1))


def join_inverter_weather(agg_inv_1: dict, s1_c: pd.DataFrame) -> dict:
    wea_inv_1 = {}
    s1_c_clean = s1_c.drop(columns=['PLANT_ID'], errors='ignore')

    for sk, df in agg_inv_1.items():
        df = df.set_index('DATE_TIME')
        join_df = df.join(s1_c_clean, how='inner')
        wea_inv_1[sk] = join_df

    return wea_inv_1


def clean_ac_dc(wea_inv_1: dict) -> dict:
    df_step_1 = {}

    for sk, df in wea_inv_1.items():
        df = df.copy()

        df['AC_CLEAN'] = df['AC_POWER'].copy()
        df['DC_CLEAN'] = df['DC_POWER'].copy()

        night = df['IRRADIATION_CLEAN'] == 0
        df.loc[night & (df['AC_CLEAN'] > 0), 'AC_CLEAN'] = 0
        df.loc[night & (df['DC_CLEAN'] > 0), 'DC_CLEAN'] = 0

        day = df['IRRADIATION_CLEAN'] > 0
        df.loc[day & (df['AC_CLEAN'] == 0), 'AC_CLEAN'] = np.nan
        df.loc[day & (df['DC_CLEAN'] == 0), 'DC_CLEAN'] = np.nan

        df['AC_CLEAN'] = df['AC_CLEAN'].interpolate().fillna(0)
        df['DC_CLEAN'] = df['DC_CLEAN'].interpolate().fillna(0)

        df_step_1[sk] = df

    return df_step_1


def clean_daily_yield(df_step_1: dict) -> dict:
    df_step_2 = {}

    for sk, df in df_step_1.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        df['DAILY_YIELD_CLEAN'] = df['DAILY_YIELD'].copy()

        all_days = np.unique(df.index.date)
        for d in all_days:
            day_mask = df.index.date == d
            df_day = df.loc[day_mask]

            irr_pos = df_day['IRRADIATION_CLEAN'] > 0

            if not irr_pos.any():
                df.loc[day_mask, 'DAILY_YIELD_CLEAN'] = 0
                continue

            t_start = df_day[irr_pos].index[0]
            t_end = df_day[irr_pos].index[-1]

            night = day_mask & (df.index < t_start)
            evening = day_mask & (df.index > t_end)
            mid = day_mask & (df.index >= t_start) & (df.index <= t_end)

            df.loc[night, 'DAILY_YIELD_CLEAN'] = 0
            df.loc[evening, 'DAILY_YIELD_CLEAN'] = df.at[t_end, 'DAILY_YIELD']

            vals = df.loc[mid, 'DAILY_YIELD_CLEAN'].values.astype(float)
            invalid = vals <= 0

            if len(vals) > 1:
                drops = np.diff(vals) < 0
                invalid[1:][drops] = True

            idx = df.loc[mid].index
            df.loc[idx[invalid], 'DAILY_YIELD_CLEAN'] = np.nan
            df.loc[idx, 'DAILY_YIELD_CLEAN'] = df.loc[idx, 'DAILY_YIELD_CLEAN'].interpolate()

            prev = df.at[idx[0], 'DAILY_YIELD_CLEAN']
            for t in idx[1:]:
                cur = df.at[t, 'DAILY_YIELD_CLEAN']
                if pd.isna(cur) or cur < prev:
                    df.at[t, 'DAILY_YIELD_CLEAN'] = prev
                else:
                    prev = cur

        df_step_2[sk] = df

    return df_step_2


def clean_total_yield(df_step_2: dict) -> dict:
    df_ps1 = {}

    for sk, df in df_step_2.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        df['TOTAL_YIELD_CLEAN'] = df['TOTAL_YIELD'].copy()

        ts = df.index
        for i in range(1, len(ts)):
            t_prev = ts[i - 1]
            t = ts[i]

            new_day = t.date() != t_prev.date()

            TY_prev = df.at[t_prev, 'TOTAL_YIELD_CLEAN']
            TY_now = df.at[t, 'TOTAL_YIELD']

            DY_prev = df.at[t_prev, 'DAILY_YIELD_CLEAN']
            DY_now = df.at[t, 'DAILY_YIELD_CLEAN']

            if new_day:
                df.at[t, 'TOTAL_YIELD_CLEAN'] = TY_prev
                continue

            expected = TY_prev + (DY_now - DY_prev)
            df.at[t, 'TOTAL_YIELD_CLEAN'] = expected if TY_now < TY_prev else TY_now

        cols = [
            'PLANT_ID', 'SOURCE_KEY',
            'AC_CLEAN', 'DC_CLEAN',
            'DAILY_YIELD_CLEAN', 'TOTAL_YIELD_CLEAN',
            'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE',
            'IRRADIATION_CLEAN', 'NUM_OPT', 'NUM_SUBOPT'
        ]

        df_ps1[sk] = df[[c for c in cols if c in df.columns]]

    return df_ps1


def plot_inverter_cleaned(df_ps1: dict, source_key_1: list, idx: int = 0):
    sk = source_key_1[idx]
    df = df_ps1[sk]

    for column in df.columns:
        plt.figure()
        plt.plot(df.index, df[column])
        plt.title(f'{sk} — {column}')
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.tight_layout()


def run_plant_1_pipeline(folder: str):
    plant_1_raw, weather_1_raw, plant_2_raw, weather_2_raw = load_raw_files(folder)
    plant_1_fixed = fix_plant_1_datetime(plant_1_raw)
    plant_1_idx, inv_1, source_key_1 = preprocess_plant_1(plant_1_fixed)

    check_missing_per_inverter(inv_1)
    check_constancy(inv_1, source_key_1)

    agg_inv_1 = aggregate_inverters(inv_1)
    weather_1_idx = preprocess_weather_1(weather_1_raw)
    s1_c = clean_irradiation(weather_1_raw)

    print_time_differences(agg_inv_1, s1_c)
    wea_inv_1 = join_inverter_weather(agg_inv_1, s1_c)

    step1 = clean_ac_dc(wea_inv_1)
    step2 = clean_daily_yield(step1)
    df_ps1 = clean_total_yield(step2)

    return {
        "plant_1_indexed": plant_1_idx,
        "inv_1": inv_1,
        "agg_inv_1": agg_inv_1,
        "weather_1_indexed": weather_1_idx,
        "sensor_clean": s1_c,
        "wea_inv_1": wea_inv_1,
        "df_step_1": step1,
        "df_step_2": step2,
        "df_ps1": df_ps1,
        "source_key_1": source_key_1,
    }


# ===============================================================
#                  PLANT 2 PIPELINE
# ===============================================================

def preprocess_plant_2(plant_2_raw: pd.DataFrame):
    df = plant_2_raw.copy()
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df = df.drop(columns=["PLANT_ID"], errors="ignore")
    df.set_index("DATE_TIME", inplace=True)

    print("Plant 2 missing values:\n", df.isnull().sum())
    print("Plant 2 shape:", df.shape)

    p2_gp = df.groupby("SOURCE_KEY")
    inv_2 = {sk: g for sk, g in p2_gp}
    source_key_2 = df["SOURCE_KEY"].unique().tolist()

    print("Number of Plant 2 inverters:", len(source_key_2))
    print("Source keys:", source_key_2)

    return df, inv_2, source_key_2


def aggregate_inverters_2(inv_2: dict) -> dict:
    agg_inv_2 = {}

    for sk, df in inv_2.items():
        agg_df = df.groupby("DATE_TIME").agg(
            SOURCE_KEY=("SOURCE_KEY", "first"),
            DC_POWER=("DC_POWER", "first"),
            AC_POWER=("AC_POWER", "first"),
            DAILY_YIELD=("DAILY_YIELD", "first"),
            TOTAL_YIELD=("TOTAL_YIELD", "first"),
            NUM_OPT=("Operating_Condition", lambda x: (x == "Optimal").sum()),
            NUM_SUBOPT=("Operating_Condition", lambda x: (x == "Suboptimal").sum()),
        ).reset_index()

        agg_inv_2[sk] = agg_df

    return agg_inv_2


def clean_irradiation_2(weather_2_raw: pd.DataFrame) -> pd.DataFrame:
    df = weather_2_raw.copy()
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df = df.drop(columns=["PLANT_ID", "SOURCE_KEY"], errors="ignore")

    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["EXPECTED_DAY"] = df["HOUR"].between(6, 18)

    df["IRRADIATION_CLEAN"] = df["IRRADIATION"].copy()

    df.loc[(~df["EXPECTED_DAY"]) & (df["IRRADIATION_CLEAN"] > 0), "IRRADIATION_CLEAN"] = 0
    df.loc[(df["EXPECTED_DAY"]) & (df["IRRADIATION_CLEAN"] == 0), "IRRADIATION_CLEAN"] = np.nan

    df["IRRADIATION_CLEAN"] = df["IRRADIATION_CLEAN"].interpolate().fillna(0)

    df = df.set_index("DATE_TIME")
    df = df.drop(columns=["IRRADIATION", "HOUR", "EXPECTED_DAY"], errors="ignore")

    return df


def join_inverter_weather_2(agg_inv_2: dict, s2_c: pd.DataFrame) -> dict:
    wea_inv_2 = {}

    for sk, df in agg_inv_2.items():
        df = df.set_index("DATE_TIME")
        join_df = df.join(s2_c, how="inner")
        wea_inv_2[sk] = join_df

    return wea_inv_2


def run_plant_2_pipeline(folder: str):
    plant_1_raw, weather_1_raw, plant_2_raw, weather_2_raw = load_raw_files(folder)

    plant_2_idx, inv_2, source_key_2 = preprocess_plant_2(plant_2_raw)
    agg_inv_2 = aggregate_inverters_2(inv_2)
    s2_c = clean_irradiation_2(weather_2_raw)
    wea_inv_2 = join_inverter_weather_2(agg_inv_2, s2_c)

    step1 = clean_ac_dc(wea_inv_2)
    step2 = clean_daily_yield(step1)
    df_ps2 = clean_total_yield(step2)

    return {
        "plant_2_indexed": plant_2_idx,
        "inv_2": inv_2,
        "agg_inv_2": agg_inv_2,
        "weather_2_clean": s2_c,
        "wea_inv_2": wea_inv_2,
        "df_step_1_p2": step1,
        "df_step_2_p2": step2,
        "df_ps2": df_ps2,
        "source_key_2": source_key_2,
    }


# Helper: Plant 2 plotting (for Script 2)
def plot_inverter_cleaned_plant2(df_ps2: dict, source_key_2: list, idx: int = 0):
    sk = source_key_2[idx]
    df = df_ps2[sk]

    for column in df.columns:
        plt.figure()
        plt.plot(df.index, df[column])
        plt.title(f'{sk} — {column}')
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.tight_layout()


# ===============================================================
#                  Training PIPELINE (run_inverter_experiment)
# ===============================================================

def run_inverter_experiment(
    inverter_id: str,
    daily_folder: str,
    start_date_str: str,
    end_date_str: str,
    verbose: bool = True,
    save_plots: bool = False,
    plot_folder: str | None = None,
):
    """
    Train multiple regression models for one inverter over multiple days.

    - Loads all daily CSV files in `daily_folder`
    - Filters rows between start_date_str and end_date_str (inclusive)
    - Fixes missing values via interpolation + ffill/bfill
    - Trains 5 models (Linear, Ridge, Lasso, RandomForest, NeuralNet)
      on the combined dataset (all days merged)
    - Trains the same 5 models per-day (per CSV) with a train/test split
    - Computes RMSE and MAE for DC_CLEAN and AC_CLEAN
    - Trains additional NeuralNets (on combined DC & AC) for diagnostics
      (iterations, learning rate, momentum, total weights, loss curve, time)
    - Builds cost/loss curves per model:

        results["loss_curves"]["dc"][model_name] → DC cost per iteration/tree
        results["loss_curves"]["ac"][model_name] → AC cost per iteration/tree

      For:
        - Linear / Ridge / Lasso: single-point MSE (list of length 1)
        - RandomForest: per-tree MSE (growing forest)
        - NeuralNet: MLPRegressor.loss_curve_ (per-iteration loss)

    Parameters
    ----------
    inverter_id : str
        ID of the inverter (used in printouts and plot filenames).
    daily_folder : str
        Folder containing daily CSVs for this inverter.
    start_date_str : str
        e.g. "2020-05-15"
    end_date_str : str
        e.g. "2020-06-17"
    verbose : bool
        If True, print progress and metrics.
    save_plots : bool
        If True, save all plots as PNG files into plot_folder.
    plot_folder : str | None
        Destination folder for plots if save_plots is True.

    Returns
    -------
    results : dict
        {
          "inverter_id": ...,
          "combined": {
              "dc": {model: {"rmse": float, "mae": float}},
              "ac": {model: {"rmse": float, "mae": float}},
              "predictions": {
                  "ModelName_DC": {"y_true": np.array, "y_pred": np.array},
                  "ModelName_AC": {"y_true": np.array, "y_pred": np.array},
              }
          },
          "parallel": {
              "days": [list_of_date_strings_with_enough_samples],
              "dc_rmse": {model: [rmse_per_day...]},
              "ac_rmse": {model: [rmse_per_day...]},
              "dc_mae":  {model: [mae_per_day...]},
              "ac_mae":  {model: [mae_per_day...]},
              "avg_dc_rmse": {model: float},
              "avg_ac_rmse": {model: float},
              "avg_dc_mae":  {model: float},
              "avg_ac_mae":  {model: float},
          },
          "loss_curves": {
              "dc": {model: [cost_values...]},
              "ac": {model: [cost_values...]},
          },
          "nn_diag": {
              "dc": {
                  "iterations": int,
                  "learning_rate": float,
                  "momentum": float,
                  "total_weights": int,
                  "train_time": float,
                  "loss_curve": list_of_floats
              },
              "ac": {
                  "iterations": int,
                  "learning_rate": float,
                  "momentum": float,
                  "total_weights": int,
                  "train_time": float,
                  "loss_curve": list_of_floats
              }
          }
        }
    """

    # ------------------------------------------------------------------
    # 0. CONFIG
    # ------------------------------------------------------------------
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    features = [
        "IRRADIATION_CLEAN",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "DAILY_YIELD_CLEAN",
    ]
    target_dc = "DC_CLEAN"
    target_ac = "AC_CLEAN"

    if save_plots and plot_folder is not None:
        os.makedirs(plot_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. HELPER: LOAD & PREPROCESS ONE CSV
    # ------------------------------------------------------------------
    def load_and_preprocess_csv(path):
        df = pd.read_csv(path)
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

        # Restrict to date range (defensive)
        df = df[(df["DATE_TIME"] >= start_date) & (df["DATE_TIME"] <= end_date)]
        if df.empty:
            return df

        # Interpolate numerics & fill NaNs
        df = df.interpolate(method="linear")
        df = df.fillna(method="bfill").fillna(method="ffill")

        return df

    # ------------------------------------------------------------------
    # 2. LOAD ALL DAILY FILES
    # ------------------------------------------------------------------
    csv_files = sorted(glob.glob(os.path.join(daily_folder, "*.csv")))

    if verbose:
        print(f"[{inverter_id}] Found {len(csv_files)} CSV files in {daily_folder}")

    daily_dfs = []
    day_labels = []

    for f in csv_files:
        df_day = load_and_preprocess_csv(f)
        if df_day.empty:
            continue

        day_date = df_day["DATE_TIME"].dt.date.iloc[0]
        day_labels.append(str(day_date))
        daily_dfs.append(df_day)

    if not daily_dfs:
        raise ValueError(
            f"[{inverter_id}] No daily data loaded after filtering. "
            f"Check folder and date range."
        )

    # Combined dataframe (all days)
    combined_df = pd.concat(daily_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # 3. FEATURES + TARGETS (COMBINED)
    # ------------------------------------------------------------------
    X_combined = combined_df[features]
    y_combined_dc = combined_df[target_dc]
    y_combined_ac = combined_df[target_ac]

    # Single train–test split (same indices for DC & AC)
    X_train, X_test, y_train_dc, y_test_dc = train_test_split(
        X_combined, y_combined_dc, test_size=0.2, shuffle=True, random_state=42
    )
    _, _, y_train_ac, y_test_ac = train_test_split(
        X_combined, y_combined_ac, test_size=0.2, shuffle=True, random_state=42
    )

    # ------------------------------------------------------------------
    # 4. DEFINE MODELS (5 TYPES)
    # ------------------------------------------------------------------
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=10000, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "NeuralNet": MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            learning_rate_init=0.001,
            momentum=0.9,
            max_iter=2000,
            random_state=42,
        ),
    }

    # Helper: RF per-tree loss curve (MSE)
    def compute_rf_loss_curve(rf_model, X, y_true):
        """
        Compute cost per tree for a RandomForest by incrementally
        averaging trees and measuring MSE on X, y_true.
        """
        n_trees = len(rf_model.estimators_)
        if n_trees == 0:
            return []

        curves = []
        # running sum of predictions
        running_sum = None
        for i, tree in enumerate(rf_model.estimators_):
            pred_i = tree.predict(X)
            if running_sum is None:
                running_sum = pred_i
            else:
                running_sum += pred_i
            y_hat = running_sum / (i + 1)
            mse_i = mean_squared_error(y_true, y_hat)
            curves.append(mse_i)
        return curves

    # ------------------------------------------------------------------
    # 5. TRAIN + EVALUATE ON COMBINED DATA
    # ------------------------------------------------------------------
    combined_results_dc = {}      # model -> {"rmse": ..., "mae": ...}
    combined_results_ac = {}      # model -> {"rmse": ..., "mae": ...}
    combined_pred_store = {}      # "Model_DC/AC" -> {"y_true": arr, "y_pred": arr}

    # cost curves (loss_curves) per model / per target
    loss_curves_dc = {name: [] for name in models.keys()}
    loss_curves_ac = {name: [] for name in models.keys()}

    if verbose:
        print(f"\n[{inverter_id}] ================== COMBINED DATA TRAINING ==================")

    for name, model in models.items():
        # ---- DC ----
        mdl_dc = model
        mdl_dc.fit(X_train, y_train_dc)
        pred_dc = mdl_dc.predict(X_test)

        rmse_dc = np.sqrt(mean_squared_error(y_test_dc, pred_dc))
        mae_dc = mean_absolute_error(y_test_dc, pred_dc)
        mse_dc = mean_squared_error(y_test_dc, pred_dc)

        combined_results_dc[name] = {"rmse": rmse_dc, "mae": mae_dc}
        combined_pred_store[name + "_DC"] = {
            "y_true": y_test_dc.to_numpy(),
            "y_pred": pred_dc,
        }

        # default DC cost curve: single MSE value
        loss_curves_dc[name] = [mse_dc]

        # ---- AC ---- (fresh instance per target)
        if name == "Linear":
            mdl_ac = LinearRegression()
        elif name == "Ridge":
            mdl_ac = Ridge(alpha=1.0)
        elif name == "Lasso":
            mdl_ac = Lasso(alpha=0.0005, max_iter=10000, random_state=42)
        elif name == "RandomForest":
            mdl_ac = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            )
        else:  # NeuralNet
            mdl_ac = MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                learning_rate_init=0.001,
                momentum=0.9,
                max_iter=2000,
                random_state=42,
            )

        mdl_ac.fit(X_train, y_train_ac)
        pred_ac = mdl_ac.predict(X_test)

        rmse_ac = np.sqrt(mean_squared_error(y_test_ac, pred_ac))
        mae_ac = mean_absolute_error(y_test_ac, pred_ac)
        mse_ac = mean_squared_error(y_test_ac, pred_ac)

        combined_results_ac[name] = {"rmse": rmse_ac, "mae": mae_ac}
        combined_pred_store[name + "_AC"] = {
            "y_true": y_test_ac.to_numpy(),
            "y_pred": pred_ac,
        }

        # default AC cost curve: single MSE value
        loss_curves_ac[name] = [mse_ac]

        # For RandomForest, override single-point cost with per-tree curve
        if name == "RandomForest":
            loss_curves_dc[name] = compute_rf_loss_curve(mdl_dc, X_test, y_test_dc)
            loss_curves_ac[name] = compute_rf_loss_curve(mdl_ac, X_test, y_test_ac)

        if verbose:
            print(
                f"[{inverter_id}] {name:12s} | "
                f"DC  RMSE={rmse_dc:8.3f}, MAE={mae_dc:8.3f} | "
                f"AC  RMSE={rmse_ac:8.3f}, MAE={mae_ac:8.3f}"
            )

    if verbose:
        print(f"[{inverter_id}] ============================================================\n")

    # ------------------------------------------------------------------
    # 6. NEURAL NETWORK DIAGNOSTICS (COMBINED, DC & AC)
    # ------------------------------------------------------------------
    nn_diag = {"dc": {}, "ac": {}}

    # ---- DC diagnostics ----
    nn_dc = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        learning_rate_init=0.001,
        momentum=0.9,
        max_iter=2000,
        random_state=42,
    )
    start_time_dc = time.time()
    nn_dc.fit(X_train, y_train_dc)
    end_time_dc = time.time()
    training_time_dc = end_time_dc - start_time_dc
    total_weights_dc = sum(w.size for w in nn_dc.coefs_)

    nn_diag["dc"] = {
        "iterations": nn_dc.n_iter_,
        "learning_rate": nn_dc.learning_rate_init,
        "momentum": nn_dc.momentum,
        "total_weights": int(total_weights_dc),
        "train_time": float(training_time_dc),
        "loss_curve": nn_dc.loss_curve_.copy(),
    }

    # ensure NeuralNet DC loss curve uses full iterative cost
    loss_curves_dc["NeuralNet"] = list(nn_dc.loss_curve_.copy())

    # ---- AC diagnostics ----
    nn_ac = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        learning_rate_init=0.001,
        momentum=0.9,
        max_iter=2000,
        random_state=42,
    )
    start_time_ac = time.time()
    nn_ac.fit(X_train, y_train_ac)
    end_time_ac = time.time()
    training_time_ac = end_time_ac - start_time_ac
    total_weights_ac = sum(w.size for w in nn_ac.coefs_)

    nn_diag["ac"] = {
        "iterations": nn_ac.n_iter_,
        "learning_rate": nn_ac.learning_rate_init,
        "momentum": nn_ac.momentum,
        "total_weights": int(total_weights_ac),
        "train_time": float(training_time_ac),
        "loss_curve": nn_ac.loss_curve_.copy(),
    }

    # ensure NeuralNet AC loss curve uses full iterative cost
    loss_curves_ac["NeuralNet"] = list(nn_ac.loss_curve_.copy())

    if verbose:
        print(f"[{inverter_id}] ====== NEURAL NETWORK DIAGNOSTICS (COMBINED DC) ======")
        print(f"Iterations completed : {nn_diag['dc']['iterations']}")
        print(f"Learning rate (init) : {nn_diag['dc']['learning_rate']}")
        print(f"Momentum             : {nn_diag['dc']['momentum']}")
        print(f"Total weights        : {nn_diag['dc']['total_weights']}")
        print(f"Training time (sec)  : {nn_diag['dc']['train_time']:.4f}")
        print("--------------------------------------------------------------")
        print(f"[{inverter_id}] ====== NEURAL NETWORK DIAGNOSTICS (COMBINED AC) ======")
        print(f"Iterations completed : {nn_diag['ac']['iterations']}")
        print(f"Learning rate (init) : {nn_diag['ac']['learning_rate']}")
        print(f"Momentum             : {nn_diag['ac']['momentum']}")
        print(f"Total weights        : {nn_diag['ac']['total_weights']}")
        print(f"Training time (sec)  : {nn_diag['ac']['train_time']:.4f}")
        print("==============================================================\n")

    # ------------------------------------------------------------------
    # 7. PER-DAY (“PARALLEL”) TRAINING
    # ------------------------------------------------------------------
    parallel_rmse_dc = {name: [] for name in models.keys()}
    parallel_rmse_ac = {name: [] for name in models.keys()}
    parallel_mae_dc = {name: [] for name in models.keys()}
    parallel_mae_ac = {name: [] for name in models.keys()}

    valid_day_labels = []  # only days with enough samples

    if verbose:
        print(f"[{inverter_id}] =============== PER-DAY (“PARALLEL”) TRAINING ===============")

    for df_day, day_label in zip(daily_dfs, day_labels):
        # Ensure enough samples to split
        if len(df_day) < 3:
            if verbose:
                print(f"[{inverter_id}] Skipping {day_label}: not enough samples ({len(df_day)})")
            continue

        X_day = df_day[features]
        y_day_dc = df_day[target_dc]
        y_day_ac = df_day[target_ac]

        Xtr, Xte, ytr_dc, yte_dc = train_test_split(
            X_day, y_day_dc, test_size=0.2, shuffle=True, random_state=42
        )
        _, _, ytr_ac, yte_ac = train_test_split(
            X_day, y_day_ac, test_size=0.2, shuffle=True, random_state=42
        )

        valid_day_labels.append(day_label)

        for name in models.keys():
            # Fresh instance per day / target
            if name == "Linear":
                mdl_dc = LinearRegression()
                mdl_ac = LinearRegression()
            elif name == "Ridge":
                mdl_dc = Ridge(alpha=1.0)
                mdl_ac = Ridge(alpha=1.0)
            elif name == "Lasso":
                mdl_dc = Lasso(alpha=0.0005, max_iter=10000, random_state=42)
                mdl_ac = Lasso(alpha=0.0005, max_iter=10000, random_state=42)
            elif name == "RandomForest":
                mdl_dc = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                )
                mdl_ac = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                )
            else:  # NeuralNet
                mdl_dc = MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    learning_rate_init=0.001,
                    momentum=0.9,
                    max_iter=2000,
                    random_state=42,
                )
                mdl_ac = MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    learning_rate_init=0.001,
                    momentum=0.9,
                    max_iter=2000,
                    random_state=42,
                )

            # DC
            mdl_dc.fit(Xtr, ytr_dc)
            pred_dc = mdl_dc.predict(Xte)
            rmse_dc = np.sqrt(mean_squared_error(yte_dc, pred_dc))
            mae_dc = mean_absolute_error(yte_dc, pred_dc)
            parallel_rmse_dc[name].append(rmse_dc)
            parallel_mae_dc[name].append(mae_dc)

            # AC
            mdl_ac.fit(Xtr, ytr_ac)
            pred_ac = mdl_ac.predict(Xte)
            rmse_ac = np.sqrt(mean_squared_error(yte_ac, pred_ac))
            mae_ac = mean_absolute_error(yte_ac, pred_ac)
            parallel_rmse_ac[name].append(rmse_ac)
            parallel_mae_ac[name].append(mae_ac)

    # Average per-day metrics
    avg_parallel_rmse_dc = {
        name: float(np.mean(vals)) for name, vals in parallel_rmse_dc.items() if len(vals) > 0
    }
    avg_parallel_rmse_ac = {
        name: float(np.mean(vals)) for name, vals in parallel_rmse_ac.items() if len(vals) > 0
    }
    avg_parallel_mae_dc = {
        name: float(np.mean(vals)) for name, vals in parallel_mae_dc.items() if len(vals) > 0
    }
    avg_parallel_mae_ac = {
        name: float(np.mean(vals)) for name, vals in parallel_mae_ac.items() if len(vals) > 0
    }

    if verbose:
        print(f"\n[{inverter_id}] ===== AVERAGE PER-DAY (“PARALLEL”) RESULTS =====")
        for name in models.keys():
            if name in avg_parallel_rmse_dc:
                print(
                    f"{name:12s} | DC  RMSE={avg_parallel_rmse_dc[name]:8.3f}, "
                    f"MAE={avg_parallel_mae_dc[name]:8.3f} | "
                    f"AC  RMSE={avg_parallel_rmse_ac[name]:8.3f}, "
                    f"MAE={avg_parallel_mae_ac[name]:8.3f}"
                )
        print("===================================================================\n")

    # ------------------------------------------------------------------
    # 8. PACK RESULTS INTO A SINGLE DICTIONARY
    # ------------------------------------------------------------------
    results = {
        "inverter_id": inverter_id,
        "combined": {
            "dc": combined_results_dc,
            "ac": combined_results_ac,
            "predictions": combined_pred_store,
        },
        "parallel": {
            "days": valid_day_labels,
            "dc_rmse": parallel_rmse_dc,
            "ac_rmse": parallel_rmse_ac,
            "dc_mae": parallel_mae_dc,
            "ac_mae": parallel_mae_ac,
            "avg_dc_rmse": avg_parallel_rmse_dc,
            "avg_ac_rmse": avg_parallel_rmse_ac,
            "avg_dc_mae": avg_parallel_mae_dc,
            "avg_ac_mae": avg_parallel_mae_ac,
        },
        "loss_curves": {
            "dc": loss_curves_dc,
            "ac": loss_curves_ac,
        },
        "nn_diag": nn_diag,
    }

    # ------------------------------------------------------------------
    # 9. SAVE PLOTS (NO plt.show())
    # ------------------------------------------------------------------
    if save_plots and plot_folder is not None:
        # ---------- A. Combined predictions: Actual vs Predicted + Residuals ----------
        for key, vals in combined_pred_store.items():
            y_true = vals["y_true"]
            y_pred = vals["y_pred"]

            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"{inverter_id} — {key} — Combined Model")

            # Scatter: Actual vs Predicted
            ax[0].scatter(y_true, y_pred, alpha=0.7)
            mn = min(y_true.min(), y_pred.min())
            mx = max(y_true.max(), y_pred.max())
            ax[0].plot([mn, mx], [mn, mx], "r--")
            ax[0].set_title("Actual vs Predicted")
            ax[0].set_xlabel("Actual")
            ax[0].set_ylabel("Predicted")
            ax[0].grid(True)

            # Residuals vs Predicted
            residuals = y_true - y_pred
            ax[1].scatter(y_pred, residuals, alpha=0.6)
            ax[1].axhline(0, color="red", linestyle="--")
            ax[1].set_title("Residuals vs Predicted")
            ax[1].set_xlabel("Predicted")
            ax[1].set_ylabel("Residual")
            ax[1].grid(True)

            # Residual distribution
            ax[2].hist(residuals, bins=20, alpha=0.8)
            ax[2].set_title("Residual Distribution")
            ax[2].set_xlabel("Residual")
            ax[2].set_ylabel("Frequency")
            ax[2].grid(True)

            fname = f"{inverter_id}_combined_{key}_performance.png"
            fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ---------- B. RMSE bar: Combined DC vs AC ----------
        labels = list(models.keys())
        rmse_dc_vals = [combined_results_dc[m]["rmse"] for m in labels]
        rmse_ac_vals = [combined_results_ac[m]["rmse"] for m in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, rmse_dc_vals, width, label="DC RMSE")
        ax.bar(x + width / 2, rmse_ac_vals, width, label="AC RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("RMSE")
        ax.set_title(f"{inverter_id} — Combined Model RMSE (DC vs AC)")
        ax.legend()
        ax.grid(axis="y")

        fname = f"{inverter_id}_combined_rmse_dc_vs_ac.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---------- C. Neural Network Loss Curve (DC) ----------
        fig, ax = plt.subplots(figsize=(10, 5))
        loss_arr_dc = np.array(nn_diag["dc"]["loss_curve"])
        ax.plot(loss_arr_dc, label="Training Loss (DC)")

        ax.set_title(f"{inverter_id} — Neural Network Loss Curve (DC, Combined)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()

        fname = f"{inverter_id}_nn_loss_curve_DC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---------- C2. Neural Network Loss Curve (AC) ----------
        fig, ax = plt.subplots(figsize=(10, 5))
        loss_arr_ac = np.array(nn_diag["ac"]["loss_curve"])
        ax.plot(loss_arr_ac, label="Training Loss (AC)")

        ax.set_title(f"{inverter_id} — Neural Network Loss Curve (AC, Combined)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()

        fname = f"{inverter_id}_nn_loss_curve_AC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---------- D. NN diagnostic bars (use DC stats) ----------
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.ravel()

        axs[0].bar(["Iterations (DC)"], [nn_diag["dc"]["iterations"]])
        axs[0].set_title("Training Iterations (DC)")
        axs[0].grid(axis="y")

        axs[1].bar(["Learning Rate (DC)"], [nn_diag["dc"]["learning_rate"]])
        axs[1].set_title("Learning Rate (DC)")
        axs[1].grid(axis="y")

        axs[2].bar(["Total Weights (DC)"], [nn_diag["dc"]["total_weights"]])
        axs[2].set_title("Model Size (Total Weights, DC)")
        axs[2].grid(axis="y")

        axs[3].bar(["Training Time (s, DC)"], [nn_diag["dc"]["train_time"]])
        axs[3].set_title("Training Time (DC)")
        axs[3].grid(axis="y")

        fig.suptitle(f"{inverter_id} — Neural Network Diagnostics (DC)", y=1.02)
        fig.tight_layout()

        fname = f"{inverter_id}_nn_diagnostics_DC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---------- E. Combined vs Average Parallel metrics (RMSE / MAE) ----------
        def plot_combined_vs_parallel(
            metric_combined_dc,
            metric_combined_ac,
            metric_parallel_dc,
            metric_parallel_ac,
            title_suffix,
            suffix_file,
        ):
            labels_loc = list(models.keys())
            x_loc = np.arange(len(labels_loc))
            width_loc = 0.18

            fig_loc, ax_loc = plt.subplots(figsize=(12, 6))

            c_dc = [metric_combined_dc[m] for m in labels_loc]
            c_ac = [metric_combined_ac[m] for m in labels_loc]
            p_dc = [metric_parallel_dc.get(m, np.nan) for m in labels_loc]
            p_ac = [metric_parallel_ac.get(m, np.nan) for m in labels_loc]

            ax_loc.bar(x_loc - 1.5 * width_loc, c_dc, width_loc, label="Combined DC")
            ax_loc.bar(x_loc - 0.5 * width_loc, c_ac, width_loc, label="Combined AC")
            ax_loc.bar(x_loc + 0.5 * width_loc, p_dc, width_loc, label="Avg Parallel DC")
            ax_loc.bar(x_loc + 1.5 * width_loc, p_ac, width_loc, label="Avg Parallel AC")

            ax_loc.set_xticks(x_loc)
            ax_loc.set_xticklabels(labels_loc)
            ax_loc.set_ylabel(title_suffix)
            ax_loc.set_title(f"{inverter_id} — Combined vs Average Parallel ({title_suffix})")
            ax_loc.legend()
            ax_loc.grid(axis="y")

            fig_loc.tight_layout()
            fig_loc.savefig(os.path.join(plot_folder, suffix_file),
                            dpi=150, bbox_inches="tight")
            plt.close(fig_loc)

        # RMSE comparison
        plot_combined_vs_parallel(
            {m: combined_results_dc[m]["rmse"] for m in models.keys()},
            {m: combined_results_ac[m]["rmse"] for m in models.keys()},
            avg_parallel_rmse_dc,
            avg_parallel_rmse_ac,
            "RMSE",
            f"{inverter_id}_combined_vs_parallel_RMSE.png",
        )

        # MAE comparison
        plot_combined_vs_parallel(
            {m: combined_results_dc[m]["mae"] for m in models.keys()},
            {m: combined_results_ac[m]["mae"] for m in models.keys()},
            avg_parallel_mae_dc,
            avg_parallel_mae_ac,
            "MAE",
            f"{inverter_id}_combined_vs_parallel_MAE.png",
        )

        # ---------- F. Per-day (“parallel”) RMSE over time ----------
        days_idx = np.arange(len(valid_day_labels))

        # DC
        fig, ax = plt.subplots(figsize=(14, 6))
        for name in models.keys():
            if len(parallel_rmse_dc[name]) == len(valid_day_labels):
                ax.plot(days_idx, parallel_rmse_dc[name], marker="o", label=name)
        ax.set_xticks(days_idx)
        ax.set_xticklabels(valid_day_labels, rotation=45, ha="right")
        ax.set_ylabel("RMSE (DC)")
        ax.set_xlabel("Day")
        ax.set_title(f"{inverter_id} — Per-Day RMSE (DC, Parallel Training)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        fname = f"{inverter_id}_per_day_rmse_DC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # AC
        fig, ax = plt.subplots(figsize=(14, 6))
        for name in models.keys():
            if len(parallel_rmse_ac[name]) == len(valid_day_labels):
                ax.plot(days_idx, parallel_rmse_ac[name], marker="o", label=name)
        ax.set_xticks(days_idx)
        ax.set_xticklabels(valid_day_labels, rotation=45, ha="right")
        ax.set_ylabel("RMSE (AC)")
        ax.set_xlabel("Day")
        ax.set_title(f"{inverter_id} — Per-Day RMSE (AC, Parallel Training)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        fname = f"{inverter_id}_per_day_rmse_AC.png"
        fig.savefig(os.path.join(plot_folder, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 10. RETURN RESULTS (your main script pickles per inverter)
    # ------------------------------------------------------------------
    return results


import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# from Utilities import run_plant_1_pipeline, plot_inverter_cleaned

if __name__ == "__main__":
    folder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\In"  # <-- update to your actual path
    outfolder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\00 Excel clean file\Plant 1"

    os.makedirs(outfolder, exist_ok=True)

    results = run_plant_1_pipeline(folder)

    df_ps1 = results["df_ps1"]
    source_keys = results["source_key_1"]

    print(f"\nDetected {len(source_keys)} inverters: {source_keys}")

    print("\nSaving cleaned inverter CSV files...")

    for sk in source_keys:
        df = df_ps1[sk]
        outfile = os.path.join(outfolder, f"Plant1_{sk}_clean.csv")
        df.to_csv(outfile)
        print(f"Saved: {outfile}")

    exported_files = [
        f for f in os.listdir(outfolder)
        if f.startswith("Plant1_") and f.endswith("_clean.csv")
    ]

    print("\n-----------------------------------------")
    print("CSV EXPORT SUMMARY")
    print("-----------------------------------------")
    print("Expected: 22 CSV")
    print(f"Found:    {len(exported_files)} CSV\n")

    if len(exported_files) == 22:
        print("✅ SUCCESS — all 22 inverter CSV files exported correctly.")
    else:
        print("❌ ERROR — missing inverter CSV files!")
        print("Files found:", exported_files)

    plot_inverter_cleaned(
        df_ps1=df_ps1,
        source_key_1=source_keys,
        idx=0
    )
    plt.show()


import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# from Utilities import run_plant_2_pipeline, plot_inverter_cleaned_plant2

if __name__ == "__main__":
    folder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\In"
    outfolder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\00 Excel clean file\Plant 2"

    os.makedirs(outfolder, exist_ok=True)

    results = run_plant_2_pipeline(folder)

    df_ps2 = results["df_ps2"]
    source_keys = results["source_key_2"]

    print(f"\nDetected {len(source_keys)} Plant 2 inverters: {source_keys}")

    print("\nSaving cleaned Plant 2 inverter CSV files...")

    for sk in source_keys:
        df = df_ps2[sk]
        outfile = os.path.join(outfolder, f"Plant2_{sk}_clean.csv")
        df.to_csv(outfile)
        print(f"Saved: {outfile}")

    exported_files = [
        f for f in os.listdir(outfolder)
        if f.startswith("Plant2_") and f.endswith("_clean.csv")
    ]

    print("\n-----------------------------------------")
    print("PLANT 2 CSV EXPORT SUMMARY")
    print("-----------------------------------------")
    print(f"Expected: {len(source_keys)} CSV")
    print(f"Found:    {len(exported_files)} CSV\n")

    if len(exported_files) == len(source_keys):
        print("✅ SUCCESS — all cleaned Plant 2 inverter CSV files exported correctly.")
    else:
        print("❌ ERROR — missing inverter CSV files!")
        print("Files found:", exported_files)

    plot_inverter_cleaned_plant2(
        df_ps2=df_ps2,
        source_key_2=source_keys,
        idx=0
    )
    plt.show()


import pandas as pd
import os

input_folder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\00 Excel clean file\Plant 1" 
output_root  = input_folder + "/Daily_Inverter_Data"

os.makedirs(output_root, exist_ok=True)

inverters = [
    '1BY6WEcLGh8j5v7', '1IF53ai7Xc0U56Y', '3PZuoBAID5Wc2HD', '7JYdWkrLSPkdwr4',
    'McdE0feGgRqW7Ca', 'VHMLBKoKgIrUVDU', 'WRmjgnKYAwPKWDb', 'ZnxXDlPa8U1GXgE',
    'ZoEaEvLYb1n2sOq', 'adLQvlD726eNBSB', 'bvBOhCH3iADSZry', 'iCRJl6heRkivqQ3',
    'ih0vzX44oOqAx2f', 'pkci93gMrogZuBj', 'rGa61gmuvPhdLxV', 'sjndEbLyjtCKgGv',
    'uHbuxQJl8lW7ozc', 'wCURE6d3bPkepu2', 'z9Y9gH1T5YWrNuG', 'zBIq5rxdHJRwDNY',
    'zVJPv84UY57bAof', 'YxYtjZvoooNbGkE'
]

keep_cols = [
    "DATE_TIME",
    "IRRADIATION_CLEAN",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "DAILY_YIELD_CLEAN",
    "DC_CLEAN",
    "AC_CLEAN"
]

start_date = pd.to_datetime("15/05/2020", dayfirst=True)
end_date   = pd.to_datetime("17/06/2020 23:59", dayfirst=True)

print(f"Processing {len(inverters)} inverters...\n")

for inv in inverters:
    filename = f"Plant1_{inv}_clean.csv"
    csv_path = os.path.join(input_folder, filename)

    if not os.path.exists(csv_path):
        print(f"⚠ WARNING: {filename} not found — skipping.")
        continue

    print(f"→ Processing inverter {inv}")

    df = pd.read_csv(csv_path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True)
    df = df[(df["DATE_TIME"] >= start_date) & (df["DATE_TIME"] <= end_date)]

    df["DATE"] = df["DATE_TIME"].dt.date
    df = df[keep_cols + ["DATE"]]

    inverter_folder = os.path.join(output_root, inv)
    os.makedirs(inverter_folder, exist_ok=True)

    for date, group in df.groupby("DATE"):
        group = group[keep_cols]
        outname = f"{date.strftime('%Y-%m-%d')}.csv"
        save_path = os.path.join(inverter_folder, outname)
        group.reset_index(drop=True).to_csv(save_path, index=False)
        print(f"   Saved {outname}")

print("\n✓ All 22 inverters processed successfully.")


import os
import pickle
# from Utilities import run_inverter_experiment

BASE_DAILY_FOLDER = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\00 Excel clean file\Plant 1\Daily_Inverter_Data"
SAVE_PLOTS_BASE = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\01 Plant1_Inverter_Models"

os.makedirs(SAVE_PLOTS_BASE, exist_ok=True)

inverters = [
    '1BY6WEcLGh8j5v7', '1IF53ai7Xc0U56Y', '3PZuoBAID5Wc2HD', '7JYdWkrLSPkdwr4',
    'McdE0feGgRqW7Ca', 'VHMLBKoKgIrUVDU', 'WRmjgnKYAwPKWDb', 'ZnxXDlPa8U1GXgE',
    'ZoEaEvLYb1n2sOq', 'adLQvlD726eNBSB', 'bvBOhCH3iADSZry', 'iCRJl6heRkivqQ3',
    'ih0vzX44oOqAx2f', 'pkci93gMrogZuBj', 'rGa61gmuvPhdLxV', 'sjndEbLyjtCKgGv',
    'uHbuxQJl8lW7ozc', 'wCURE6d3bPkepu2', 'z9Y9gH1T5YWrNuG', 'zBIq5rxdHJRwDNY',
    'zVJPv84UY57bAof', 'YxYtjZvoooNbGkE'
]

all_results = {}

for inv in inverters:
    print(f"\n======================")
    print(f" TRAINING INVERTER: {inv}")
    print(f"======================\n")

    inverter_daily_path = os.path.join(BASE_DAILY_FOLDER, inv)
    inverter_plot_path = os.path.join(SAVE_PLOTS_BASE, inv)
    os.makedirs(inverter_plot_path, exist_ok=True)

    results = run_inverter_experiment(
        inverter_id=inv,
        daily_folder=inverter_daily_path,
        start_date_str="2020-05-15",
        end_date_str="2020-06-17",
        verbose=True,
        save_plots=True,
        plot_folder=inverter_plot_path
    )

    all_results[inv] = results

    results_file = os.path.join(SAVE_PLOTS_BASE, f"{inv}_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"✓ Saved results for {inv}")
    print(f"✓ Plots saved to: {inverter_plot_path}")
    print("------------------------------------------------------")

master_save_path = os.path.join(SAVE_PLOTS_BASE, "ALL_INVERTER_RESULTS.pkl")
with open(master_save_path, "wb") as f:
    pickle.dump(all_results, f)

print("\n======================================================")
print("   FINISHED TRAINING ALL 22 INVERTERS — SUCCESS 🎉")
print("======================================================")


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
RESULTS_FOLDER = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\01 Plant1_Inverter_Models"
PLOTS_FOLDER   = os.path.join(RESULTS_FOLDER, "00 Training_Visualization_Plots")

os.makedirs(PLOTS_FOLDER, exist_ok=True)

# ============================================================
# LOAD ALL NN LOSS CURVES FROM PKL FILES
# ============================================================

loss_dc = {}   # inverter → loss array
loss_ac = {}   # inverter → loss array

for fname in os.listdir(RESULTS_FOLDER):
    if not fname.endswith("_results.pkl"):
        continue

    fpath = os.path.join(RESULTS_FOLDER, fname)
    with open(fpath, "rb") as f:
        res = pickle.load(f)

    inv_id = res.get("inverter_id", fname.replace("_results.pkl", ""))

    diag = res.get("nn_diag", {})

    if "dc" in diag and "loss_curve" in diag["dc"]:
        loss_dc[inv_id] = np.array(diag["dc"]["loss_curve"], dtype=float)

    if "ac" in diag and "loss_curve" in diag["ac"]:
        loss_ac[inv_id] = np.array(diag["ac"]["loss_curve"], dtype=float)

print(f"Loaded DC loss curves from {len(loss_dc)} inverters")
print(f"Loaded AC loss curves from {len(loss_ac)} inverters")


# ============================================================
# A1. PLOT ALL DC LOSS CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
max_len_dc = max(len(v) for v in loss_dc.values())
all_dc = np.full((len(loss_dc), max_len_dc), np.nan)

for i, (inv, curve) in enumerate(loss_dc.items()):
    ax.plot(curve, alpha=0.3, label=inv)
    all_dc[i, :len(curve)] = curve

mean_dc = np.nanmean(all_dc, axis=0)
std_dc  = np.nanstd(all_dc, axis=0)

ax.plot(mean_dc, color="black", linewidth=2, label="Mean DC Loss")

ax.fill_between(
    np.arange(len(mean_dc)),
    mean_dc - std_dc,
    mean_dc + std_dc,
    alpha=0.15,
    label="±1 std"
)

ax.set_title("Neural Network DC Loss Curves — All Inverters")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "DC_Loss_All.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================
# A2. PLOT ALL AC LOSS CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
max_len_ac = max(len(v) for v in loss_ac.values())
all_ac = np.full((len(loss_ac), max_len_ac), np.nan)

for i, (inv, curve) in enumerate(loss_ac.items()):
    ax.plot(curve, alpha=0.3, label=inv)
    all_ac[i, :len(curve)] = curve

mean_ac = np.nanmean(all_ac, axis=0)
std_ac  = np.nanstd(all_ac, axis=0)

ax.plot(mean_ac, color="black", linewidth=2, label="Mean AC Loss")
ax.fill_between(
    np.arange(len(mean_ac)),
    mean_ac - std_ac,
    mean_ac + std_ac,
    alpha=0.15
)

ax.set_title("Neural Network AC Loss Curves — All Inverters")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "AC_Loss_All.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================
# B. DC vs AC Mean Loss Comparison
# ============================================================

L = min(len(mean_dc), len(mean_ac))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mean_dc[:L], label="Mean DC Loss", linewidth=2)
ax.plot(mean_ac[:L], label="Mean AC Loss", linewidth=2)

ax.set_title("Mean Loss Comparison: DC vs AC")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "Mean_DC_vs_AC.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================
# C. Convergence Speed (per inverter)
# ============================================================

def get_convergence_epoch(loss, tol=1e-4, patience=10):
    """Returns epoch where improvement slows down."""
    best = loss[0]
    count = 0
    for i in range(1, len(loss)):
        if loss[i] < best - tol:
            best = loss[i]
            count = 0
        else:
            count += 1
        if count >= patience:
            return i
    return len(loss)

conv_dc = {inv: get_convergence_epoch(curve) for inv, curve in loss_dc.items()}
conv_ac = {inv: get_convergence_epoch(curve) for inv, curve in loss_ac.items()}

# Bar plot
fig, ax = plt.subplots(figsize=(14, 6))
invs = list(conv_dc.keys())
ax.bar(invs, [conv_dc[i] for i in invs], alpha=0.6, label="DC")
ax.bar(invs, [conv_ac.get(i, np.nan) for i in invs], alpha=0.6, label="AC")

ax.set_title("Convergence Epoch per Inverter")
ax.set_ylabel("Epoch")
ax.set_xticklabels(invs, rotation=45, ha="right")
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "Convergence_Epochs.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n✅ Training visualization complete.")
print(f"Plots saved in: {PLOTS_FOLDER}")


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# CONFIG
# ==============================================================
RESULTS_FOLDER = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\01 Plant1_Inverter_Models"
BIASVAR_FOLDER = os.path.join(RESULTS_FOLDER, "00 BiasVariance")

os.makedirs(BIASVAR_FOLDER, exist_ok=True)

MODELS = ["Linear", "Ridge", "Lasso", "RandomForest", "NeuralNet"]

# ==============================================================
# 1. LOAD ALL PER-INVERTER RESULTS (.pkl files)
# ==============================================================

inverter_results = {}   # inverter_id -> results dict

# Neural net loss curves per inverter
nn_loss_dc = {}         # inverter_id -> np.array loss curve (DC)
nn_loss_ac = {}         # inverter_id -> np.array loss curve (AC)

# Neural net diagnostics (per inverter)
nn_diag_dc = {
    "iterations": {},
    "learning_rate": {},
    "momentum": {},
    "total_weights": {},
    "train_time": {},
}
nn_diag_ac = {
    "iterations": {},
    "learning_rate": {},
    "momentum": {},
    "total_weights": {},
    "train_time": {},
}

for fname in os.listdir(RESULTS_FOLDER):
    if not fname.endswith("_results.pkl"):
        # skip master file or other pkl's
        continue

    fpath = os.path.join(RESULTS_FOLDER, fname)
    with open(fpath, "rb") as f:
        res = pickle.load(f)

    inverter_id = res.get("inverter_id", fname.replace("_results.pkl", ""))
    inverter_results[inverter_id] = res

    # Collect NN loss curves + diagnostics, if present
    diag = res.get("nn_diag", {})

    # Expected structure (newer version):
    # nn_diag = {
    #   "dc": {"iterations":..., "learning_rate":..., "momentum":...,
    #          "total_weights":..., "train_time":..., "loss_curve":[...]},
    #   "ac": {...}
    # }
    if isinstance(diag, dict) and "dc" in diag:
        dc_diag = diag["dc"]
        if "loss_curve" in dc_diag:
            nn_loss_dc[inverter_id] = np.array(dc_diag["loss_curve"], dtype=float)
        for key in nn_diag_dc.keys():
            if key in dc_diag:
                nn_diag_dc[key][inverter_id] = dc_diag[key]

    # AC side
    if isinstance(diag, dict) and "ac" in diag:
        ac_diag = diag["ac"]
        if "loss_curve" in ac_diag:
            nn_loss_ac[inverter_id] = np.array(ac_diag["loss_curve"], dtype=float)
        for key in nn_diag_ac.keys():
            if key in ac_diag:
                nn_diag_ac[key][inverter_id] = ac_diag[key]

print(f"Loaded {len(inverter_results)} inverter result files.")


# ==============================================================
# 2. COLLECT METRICS ACROSS INVERTERS
# ==============================================================

# Combined (all days merged)
combined_dc_rmse = {m: [] for m in MODELS}
combined_ac_rmse = {m: [] for m in MODELS}
combined_dc_mae  = {m: [] for m in MODELS}
combined_ac_mae  = {m: [] for m in MODELS}

# Parallel (average per-day metrics stored in avg_* fields)
parallel_dc_rmse = {m: [] for m in MODELS}
parallel_ac_rmse = {m: [] for m in MODELS}
parallel_dc_mae  = {m: [] for m in MODELS}
parallel_ac_mae  = {m: [] for m in MODELS}

# We also keep *per-inverter* per-model parallel RMSE lists
# for variance estimation
parallel_dc_rmse_full = {m: [] for m in MODELS}  # each element = list of per-day RMSE for that inverter
parallel_ac_rmse_full = {m: [] for m in MODELS}

for inv_id, res in inverter_results.items():
    comb = res["combined"]
    par  = res["parallel"]

    # --- combined ---
    dc_comb = comb["dc"]
    ac_comb = comb["ac"]

    for m in MODELS:
        if m in dc_comb:
            combined_dc_rmse[m].append(dc_comb[m]["rmse"])
            combined_dc_mae[m].append(dc_comb[m]["mae"])
        if m in ac_comb:
            combined_ac_rmse[m].append(ac_comb[m]["rmse"])
            combined_ac_mae[m].append(ac_comb[m]["mae"])

    # --- parallel (per-day info + averages) ---
    avg_dc_rmse = par.get("avg_dc_rmse", {})
    avg_ac_rmse = par.get("avg_ac_rmse", {})
    avg_dc_mae  = par.get("avg_dc_mae", {})
    avg_ac_mae  = par.get("avg_ac_mae", {})

    dc_rmse_days = par.get("dc_rmse", {})
    ac_rmse_days = par.get("ac_rmse", {})

    for m in MODELS:
        # parallel averages
        if m in avg_dc_rmse:
            parallel_dc_rmse[m].append(avg_dc_rmse[m])
        if m in avg_ac_rmse:
            parallel_ac_rmse[m].append(avg_ac_rmse[m])
        if m in avg_dc_mae:
            parallel_dc_mae[m].append(avg_dc_mae[m])
        if m in avg_ac_mae:
            parallel_ac_mae[m].append(avg_ac_mae[m])

        # full per-day RMSE lists (for variance proxy)
        if m in dc_rmse_days:
            # store the list for this inverter and model
            parallel_dc_rmse_full[m].append(dc_rmse_days[m])
        if m in ac_rmse_days:
            parallel_ac_rmse_full[m].append(ac_rmse_days[m])


# Helper to compute mean & std, ignoring empty lists
def mean_std(arr):
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))


# ==============================================================
# 3. BIAS–VARIANCE PROXIES
# ==============================================================

# "Bias" proxy: mean combined RMSE across inverters
# "Variance" proxy: std of *per-day* parallel RMSE across days & inverters

bias_proxy_dc = {}
bias_proxy_ac = {}
var_proxy_dc  = {}
var_proxy_ac  = {}

for m in MODELS:
    # bias proxies
    bias_proxy_dc[m] = mean_std(combined_dc_rmse[m])[0]
    bias_proxy_ac[m] = mean_std(combined_ac_rmse[m])[0]

    # variance proxies: flatten all per-day RMSE for this model
    all_dc_days = []
    for lst in parallel_dc_rmse_full[m]:
        all_dc_days.extend(lst)
    all_ac_days = []
    for lst in parallel_ac_rmse_full[m]:
        all_ac_days.extend(lst)

    var_proxy_dc[m] = float(np.std(all_dc_days)) if len(all_dc_days) > 0 else np.nan
    var_proxy_ac[m] = float(np.std(all_ac_days)) if len(all_ac_days) > 0 else np.nan


# ==============================================================
# 4. BIAS–VARIANCE SCATTER PLOTS (DC & AC)
# ==============================================================

labels = MODELS
x_dc = [var_proxy_dc[m] for m in labels]
y_dc = [bias_proxy_dc[m] for m in labels]

fig, ax = plt.subplots(figsize=(8, 6))
for i, m in enumerate(labels):
    ax.scatter(x_dc[i], y_dc[i])
    ax.text(x_dc[i] * 1.01, y_dc[i] * 1.01, m)

ax.set_xlabel("Variance proxy (std of per-day RMSE, DC)")
ax.set_ylabel("Bias proxy (mean combined RMSE, DC)")
ax.set_title("Bias–Variance Proxy Plane — DC")
ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(BIASVAR_FOLDER, "bias_variance_scatter_DC.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# AC version
x_ac = [var_proxy_ac[m] for m in labels]
y_ac = [bias_proxy_ac[m] for m in labels]

fig, ax = plt.subplots(figsize=(8, 6))
for i, m in enumerate(labels):
    ax.scatter(x_ac[i], y_ac[i])
    ax.text(x_ac[i] * 1.01, y_ac[i] * 1.01, m)

ax.set_xlabel("Variance proxy (std of per-day RMSE, AC)")
ax.set_ylabel("Bias proxy (mean combined RMSE, AC)")
ax.set_title("Bias–Variance Proxy Plane — AC")
ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(BIASVAR_FOLDER, "bias_variance_scatter_AC.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================
# 5. BAR PLOTS: BIAS & VARIANCE (DC & AC)
# ==============================================================

x = np.arange(len(labels))
width = 0.35

# DC
fig, ax = plt.subplots(figsize=(10, 5))
bias_dc_bar = [bias_proxy_dc[m] for m in labels]
var_dc_bar  = [var_proxy_dc[m] for m in labels]

ax.bar(x - width/2, bias_dc_bar, width, label="Bias proxy (mean combined RMSE)")
ax.bar(x + width/2, var_dc_bar,  width, label="Variance proxy (std per-day RMSE)")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Error / Std")
ax.set_title("Bias vs Variance proxy — DC")
ax.legend()
ax.grid(axis="y")

fig.tight_layout()
fig.savefig(os.path.join(BIASVAR_FOLDER, "bias_variance_bar_DC.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# AC
fig, ax = plt.subplots(figsize=(10, 5))
bias_ac_bar = [bias_proxy_ac[m] for m in labels]
var_ac_bar  = [var_proxy_ac[m] for m in labels]

ax.bar(x - width/2, bias_ac_bar, width, label="Bias proxy (mean combined RMSE)")
ax.bar(x + width/2, var_ac_bar,  width, label="Variance proxy (std per-day RMSE)")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Error / Std")
ax.set_title("Bias vs Variance proxy — AC")
ax.legend()
ax.grid(axis="y")

fig.tight_layout()
fig.savefig(os.path.join(BIASVAR_FOLDER, "bias_variance_bar_AC.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================
# 6. NEURAL NET LEARNING CURVES (COST vs ITERATION) — DC & AC
# ==============================================================

# ----- DC cost function -----
if len(nn_loss_dc) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    max_len_dc = max(len(curve) for curve in nn_loss_dc.values())
    all_curves_dc = np.full((len(nn_loss_dc), max_len_dc), np.nan)

    for idx, (inv_id, curve) in enumerate(nn_loss_dc.items()):
        epochs = np.arange(len(curve))
        ax.plot(epochs, curve, alpha=0.3, label=inv_id)
        all_curves_dc[idx, :len(curve)] = curve

    mean_curve_dc = np.nanmean(all_curves_dc, axis=0)
    ax.plot(np.arange(len(mean_curve_dc)), mean_curve_dc,
            linewidth=2.5, label="Mean across inverters")

    ax.set_xlabel("Iteration / Epoch")
    ax.set_ylabel("Loss (Cost Function)")
    ax.set_title("Neural Network Training Loss — DC (Combined data)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(BIASVAR_FOLDER, "nn_learning_curves_DC.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
else:
    print("No DC nn_diag loss_curve found; skipping DC cost-function plot.")

# ----- AC cost function -----
if len(nn_loss_ac) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    max_len_ac = max(len(curve) for curve in nn_loss_ac.values())
    all_curves_ac = np.full((len(nn_loss_ac), max_len_ac), np.nan)

    for idx, (inv_id, curve) in enumerate(nn_loss_ac.items()):
        epochs = np.arange(len(curve))
        ax.plot(epochs, curve, alpha=0.3, label=inv_id)
        all_curves_ac[idx, :len(curve)] = curve

    mean_curve_ac = np.nanmean(all_curves_ac, axis=0)
    ax.plot(np.arange(len(mean_curve_ac)), mean_curve_ac,
            linewidth=2.5, label="Mean across inverters")

    ax.set_xlabel("Iteration / Epoch")
    ax.set_ylabel("Loss (Cost Function)")
    ax.set_title("Neural Network Training Loss — AC (Combined data)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(BIASVAR_FOLDER, "nn_learning_curves_AC.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
else:
    print("No AC nn_diag loss_curve found; skipping AC cost-function plot.")

# ----- DC vs AC mean loss in a single figure -----
if len(nn_loss_dc) > 0 and len(nn_loss_ac) > 0:
    # Compute mean DC
    max_len_dc = max(len(c) for c in nn_loss_dc.values())
    dc_mat = np.full((len(nn_loss_dc), max_len_dc), np.nan)
    for i, c in enumerate(nn_loss_dc.values()):
        dc_mat[i, :len(c)] = c
    mean_dc = np.nanmean(dc_mat, axis=0)

    # Compute mean AC
    max_len_ac = max(len(c) for c in nn_loss_ac.values())
    ac_mat = np.full((len(nn_loss_ac), max_len_ac), np.nan)
    for i, c in enumerate(nn_loss_ac.values()):
        ac_mat[i, :len(c)] = c
    mean_ac = np.nanmean(ac_mat, axis=0)

    # Align lengths
    L = min(len(mean_dc), len(mean_ac))
    mean_dc = mean_dc[:L]
    mean_ac = mean_ac[:L]

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(L)
    ax.plot(epochs, mean_dc, label="Mean DC Loss")
    ax.plot(epochs, mean_ac, label="Mean AC Loss")
    ax.set_xlabel("Iteration / Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Neural Net Cost Function Comparison — DC vs AC (mean over inverters)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(BIASVAR_FOLDER, "nn_mean_loss_DC_vs_AC.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# 7. PRINT SOME TEXTUAL BIAS–VARIANCE DIAGNOSIS
# ==============================================================

print("\n===== Bias–Variance Diagnosis (proxies) =====")
print("Model         | Bias_DC  | Var_DC   | Bias_AC  | Var_AC")
print("-------------------------------------------------------")
for m in MODELS:
    print(f"{m:12s} | "
          f"{bias_proxy_dc[m]:7.3f} | {var_proxy_dc[m]:7.3f} | "
          f"{bias_proxy_ac[m]:7.3f} | {var_proxy_ac[m]:7.3f}")
print("=======================================================")

print("\n✅ Bias–Variance analysis complete.")
print(f"All bias–variance and learning-curve plots saved in: {BIASVAR_FOLDER}")


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
RESULTS_FOLDER = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\01 Plant1_Inverter_Models"
PLOTS_FOLDER   = os.path.join(RESULTS_FOLDER, "00 Training_Visualization_Plots")

os.makedirs(PLOTS_FOLDER, exist_ok=True)

# ============================================================
# LOAD ALL NN LOSS CURVES FROM PKL FILES
# ============================================================

loss_dc = {}   # inverter → loss array
loss_ac = {}   # inverter → loss array

for fname in os.listdir(RESULTS_FOLDER):
    if not fname.endswith("_results.pkl"):
        continue

    fpath = os.path.join(RESULTS_FOLDER, fname)
    with open(fpath, "rb") as f:
        res = pickle.load(f)

    inv_id = res.get("inverter_id", fname.replace("_results.pkl", ""))

    diag = res.get("nn_diag", {})

    if "dc" in diag and "loss_curve" in diag["dc"]:
        loss_dc[inv_id] = np.array(diag["dc"]["loss_curve"], dtype=float)

    if "ac" in diag and "loss_curve" in diag["ac"]:
        loss_ac[inv_id] = np.array(diag["ac"]["loss_curve"], dtype=float)

print(f"Loaded DC loss curves from {len(loss_dc)} inverters")
print(f"Loaded AC loss curves from {len(loss_ac)} inverters")


# ============================================================
# A1. PLOT ALL DC LOSS CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
max_len_dc = max(len(v) for v in loss_dc.values())
all_dc = np.full((len(loss_dc), max_len_dc), np.nan)

for i, (inv, curve) in enumerate(loss_dc.items()):
    ax.plot(curve, alpha=0.3, label=inv)
    all_dc[i, :len(curve)] = curve

mean_dc = np.nanmean(all_dc, axis=0)
std_dc  = np.nanstd(all_dc, axis=0)

ax.plot(mean_dc, color="black", linewidth=2, label="Mean DC Loss")

ax.fill_between(
    np.arange(len(mean_dc)),
    mean_dc - std_dc,
    mean_dc + std_dc,
    alpha=0.15,
    label="±1 std"
)

ax.set_title("Neural Network DC Loss Curves — All Inverters")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "DC_Loss_All.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================
# A2. PLOT ALL AC LOSS CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
max_len_ac = max(len(v) for v in loss_ac.values())
all_ac = np.full((len(loss_ac), max_len_ac), np.nan)

for i, (inv, curve) in enumerate(loss_ac.items()):
    ax.plot(curve, alpha=0.3, label=inv)
    all_ac[i, :len(curve)] = curve

mean_ac = np.nanmean(all_ac, axis=0)
std_ac  = np.nanstd(all_ac, axis=0)

ax.plot(mean_ac, color="black", linewidth=2, label="Mean AC Loss")
ax.fill_between(
    np.arange(len(mean_ac)),
    mean_ac - std_ac,
    mean_ac + std_ac,
    alpha=0.15
)

ax.set_title("Neural Network AC Loss Curves — All Inverters")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "AC_Loss_All.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================
# B. DC vs AC Mean Loss Comparison
# ============================================================

L = min(len(mean_dc), len(mean_ac))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mean_dc[:L], label="Mean DC Loss", linewidth=2)
ax.plot(mean_ac[:L], label="Mean AC Loss", linewidth=2)

ax.set_title("Mean Loss Comparison: DC vs AC")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "Mean_DC_vs_AC.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================
# C. Convergence Speed (per inverter)
# ============================================================

def get_convergence_epoch(loss, tol=1e-4, patience=10):
    """Returns epoch where improvement slows down."""
    best = loss[0]
    count = 0
    for i in range(1, len(loss)):
        if loss[i] < best - tol:
            best = loss[i]
            count = 0
        else:
            count += 1
        if count >= patience:
            return i
    return len(loss)

conv_dc = {inv: get_convergence_epoch(curve) for inv, curve in loss_dc.items()}
conv_ac = {inv: get_convergence_epoch(curve) for inv, curve in loss_ac.items()}

# Bar plot
fig, ax = plt.subplots(figsize=(14, 6))
invs = list(conv_dc.keys())
ax.bar(invs, [conv_dc[i] for i in invs], alpha=0.6, label="DC")
ax.bar(invs, [conv_ac.get(i, np.nan) for i in invs], alpha=0.6, label="AC")

ax.set_title("Convergence Epoch per Inverter")
ax.set_ylabel("Epoch")
ax.set_xticklabels(invs, rotation=45, ha="right")
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "Convergence_Epochs.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n✅ Training visualization complete.")
print(f"Plots saved in: {PLOTS_FOLDER}")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Folder where all inverter CSV files are stored
input_folder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\00 Excel clean file\Plant 1"

# (Optional) Folder where plots will be saved
plot_output = input_folder + "/Plots"
os.makedirs(plot_output, exist_ok=True)

# 22 inverter IDs
inverters = [
    '1BY6WEcLGh8j5v7', '1IF53ai7Xc0U56Y', '3PZuoBAID5Wc2HD', '7JYdWkrLSPkdwr4',
    'McdE0feGgRqW7Ca', 'VHMLBKoKgIrUVDU', 'WRmjgnKYAwPKWDb', 'ZnxXDlPa8U1GXgE',
    'ZoEaEvLYb1n2sOq', 'adLQvlD726eNBSB', 'bvBOhCH3iADSZry', 'iCRJl6heRkivqQ3',
    'ih0vzX44oOqAx2f', 'pkci93gMrogZuBj', 'rGa61gmuvPhdLxV', 'sjndEbLyjtCKgGv',
    'uHbuxQJl8lW7ozc', 'wCURE6d3bPkepu2', 'z9Y9gH1T5YWrNuG', 'zBIq5rxdHJRwDNY',
    'zVJPv84UY57bAof', 'YxYtjZvoooNbGkE'
]

# Variables to plot
variables = [
    ("IRRADIATION_CLEAN", "Irradiation"),
    ("AMBIENT_TEMPERATURE", "Ambient Temp (°C)"),
    ("MODULE_TEMPERATURE", "Module Temp (°C)"),
    ("DAILY_YIELD_CLEAN", "Daily Yield"),
    ("DC_CLEAN", "DC Power"),
    ("AC_CLEAN", "AC Power"),
]

# Time tick labels (full day 00:00 → 23:00)
tick_minutes = [h * 60 for h in range(0, 24)]
tick_labels = [f"{h:02d}:00" for h in range(0, 24)]

# Date range filter
start_date = pd.to_datetime("15/05/2020", dayfirst=True)
end_date   = pd.to_datetime("17/06/2020 23:45", dayfirst=True)

# -------------------------------------------------------------------
# MAIN LOOP — PROCESS EACH INVERTER
# -------------------------------------------------------------------
for inv in inverters:
    csv_path = os.path.join(input_folder, f"Plant1_{inv}_clean.csv")

    if not os.path.exists(csv_path):
        print(f"⚠ Missing file skipped: {csv_path}")
        continue

    print(f"📈 Plotting inverter: {inv}")

    # Load data
    df = pd.read_csv(csv_path)

    # Convert to datetime
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True)

    # Filter date range
    df = df[(df["DATE_TIME"] >= start_date) & (df["DATE_TIME"] <= end_date)]

    # Extract date + minutes from midnight
    df["DATE"] = df["DATE_TIME"].dt.date
    df["MINUTES"] = df["DATE_TIME"].dt.hour * 60 + df["DATE_TIME"].dt.minute

    # -------------------------------------------------------------
    # BUILD FIGURE
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(6, 7, figure=fig, width_ratios=[1,1,1,1,1,1,0.35])

    axes = []
    for i in range(6):
        ax = fig.add_subplot(gs[i, :6])
        axes.append(ax)

    legend_ax = fig.add_subplot(gs[:, 6])
    legend_ax.axis("off")

    # -------------------------------------------------------------
    # PLOT VARIABLES
    # -------------------------------------------------------------
    for ax, (col, ylabel) in zip(axes, variables):
        for date, group in df.groupby("DATE"):
            ax.plot(group["MINUTES"], group[col], label=str(date))
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_xticks(tick_minutes)
        ax.set_xticklabels(tick_labels)

    axes[-1].set_xlabel("Time of Day")

    # -------------------------------------------------------------
    # LEGEND PANEL
    # -------------------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        title="Dates",
        loc="center",
        frameon=True,
        fontsize=8,
    )

    plt.tight_layout()

    # OPTIONAL: Save figure automatically
    plot_path = os.path.join(plot_output, f"{inv}_plot.png")
    plt.savefig(plot_path, dpi=200)

    print(f"   ✓ Saved plot: {plot_path}")

    # Show each inverter's plot
    plt.show()


input_folder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\00 Excel clean file\Plant 2" 
output_root  = input_folder + "/Daily_Inverter_Data"

os.makedirs(output_root, exist_ok=True)

inverters = [
    "4UPUqMRk7TRMgml", "9kRcWv60rDACzjR", "81aHJ1q11NBPMrL", "Et9kgGMDl729KT4",
    "IQ2d7wF4YD8zU1Q", "LlT2YUhhzqhg5Sw", "LYwnQax7tkwH5Cb", "mqwcsP2rE7J0TFp",
    "Mx2yZCDsyf6DPfv", "NgDl19wMapZy17u", "oZ35aAeoifZaQzV", "oZZkBaNadn6DNKz",
    "PeE6FRyGXUgsRhN", "q49J1IKaHRwDQnt", "Qf4GUc1pJu5T6c6", "Quc1TzYxW2pYoWX",
    "rrq4fwE8jgrTyWY", "V94E5Ben1TlhnDV", "vOuJvMaM2sgwLmb", "WcxssY2VbP4hApt",
    "xMbIugepa2P7lBB", "xoJJ8DcxJEcupym"
]

keep_cols = [
    "DATE_TIME",
    "IRRADIATION_CLEAN",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "DAILY_YIELD_CLEAN",
    "DC_CLEAN",
    "AC_CLEAN"
]

start_date = pd.to_datetime("15/05/2020", dayfirst=True)
end_date   = pd.to_datetime("17/06/2020 23:59", dayfirst=True)

print(f"Processing {len(inverters)} inverters...\n")

for inv in inverters:
    filename = f"Plant2_{inv}_clean.csv"
    csv_path = os.path.join(input_folder, filename)

    if not os.path.exists(csv_path):
        print(f"⚠ WARNING: {filename} not found — skipping.")
        continue

    print(f"→ Processing inverter {inv}")

    df = pd.read_csv(csv_path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True)
    df = df[(df["DATE_TIME"] >= start_date) & (df["DATE_TIME"] <= end_date)]

    df["DATE"] = df["DATE_TIME"].dt.date
    df = df[keep_cols + ["DATE"]]

    inverter_folder = os.path.join(output_root, inv)
    os.makedirs(inverter_folder, exist_ok=True)

    for date, group in df.groupby("DATE"):
        group = group[keep_cols]
        outname = f"{date.strftime('%Y-%m-%d')}.csv"
        save_path = os.path.join(inverter_folder, outname)
        group.reset_index(drop=True).to_csv(save_path, index=False)
        print(f"   Saved {outname}")

print("\n✓ All 22 inverters processed successfully.")

import os
import pickle
# from Utilities import run_inverter_experiment   # your main training function

# ============================================================
# PATHS FOR PLANT 2 (TASK 3)
# ============================================================
BASE_DAILY_FOLDER = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\00 Excel clean file\Plant 2\Daily_Inverter_Data"
SAVE_PLOTS_BASE   = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\02 Plant2_Inverter_Models"

os.makedirs(SAVE_PLOTS_BASE, exist_ok=True)

# ============================================================
# 22 PLANT 2 INVERTERS (auto-detected from your folder)
# ============================================================
inverters = [
    "4UPUqMRk7TRMgml", "9kRcWv60rDACzjR", "81aHJ1q11NBPMrL", "Et9kgGMDl729KT4",
    "IQ2d7wF4YD8zU1Q", "LlT2YUhhzqhg5Sw", "LYwnQax7tkwH5Cb", "mqwcsP2rE7J0TFp",
    "Mx2yZCDsyf6DPfv", "NgDl19wMapZy17u", "oZ35aAeoifZaQzV", "oZZkBaNadn6DNKz",
    "PeE6FRyGXUgsRhN", "q49J1IKaHRwDQnt", "Qf4GUc1pJu5T6c6", "Quc1TzYxW2pYoWX",
    "rrq4fwE8jgrTyWY", "V94E5Ben1TlhnDV", "vOuJvMaM2sgwLmb", "WcxssY2VbP4hApt",
    "xMbIugepa2P7lBB", "xoJJ8DcxJEcupym"
]

# ============================================================
# TRAINING LOOP FOR ALL 22 INVERTERS — PLANT 2
# ============================================================
all_results = {}

for inv in inverters:

    print("\n===================================")
    print(f"   TRAINING PLANT 2 INVERTER: {inv}")
    print("===================================\n")

    # Daily folder for this inverter
    inverter_daily_path = os.path.join(BASE_DAILY_FOLDER, inv)

    # Create plot folder
    inverter_plot_path = os.path.join(SAVE_PLOTS_BASE, inv)
    os.makedirs(inverter_plot_path, exist_ok=True)

    # -------------------------------------------------------
    # RUN THE TRAINING EXPERIMENT FOR THIS INVERTER
    # -------------------------------------------------------
    results = run_inverter_experiment(
        inverter_id=inv,
        daily_folder=inverter_daily_path,
        start_date_str="2020-05-15",
        end_date_str="2020-06-17",
        verbose=True,
        save_plots=True,
        plot_folder=inverter_plot_path
    )

    # Store results in dictionary
    all_results[inv] = results

    # Save single inverter results
    results_file = os.path.join(SAVE_PLOTS_BASE, f"{inv}_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"✔ Saved results → {results_file}")
    print(f"✔ Saved plots   → {inverter_plot_path}")
    print("-----------------------------------------------")

# ============================================================
# SAVE MASTER RESULTS FILE FOR PLANT 2
# ============================================================
master_save_path = os.path.join(SAVE_PLOTS_BASE, "PLANT2_ALL_INVERTER_RESULTS.pkl")
with open(master_save_path, "wb") as f:
    pickle.dump(all_results, f)

print("\n======================================================")
print("   FINISHED TRAINING ALL 22 PLANT 2 INVERTERS 🎉")
print(f"   MASTER RESULTS SAVED TO:\n   {master_save_path}")
print("======================================================")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG  (PLANT 2 VERSION)
# ============================================================
RESULTS_FOLDER = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\02 Plant2_Inverter_Models"
PLOTS_FOLDER   = os.path.join(RESULTS_FOLDER, "00 Training_Visualization_Plots")

os.makedirs(PLOTS_FOLDER, exist_ok=True)

# ============================================================
# LOAD ALL NN LOSS CURVES FROM PKL FILES
# ============================================================

loss_dc = {}   # inverter → loss array
loss_ac = {}   # inverter → loss array

for fname in os.listdir(RESULTS_FOLDER):
    if not fname.endswith("_results.pkl"):
        continue

    fpath = os.path.join(RESULTS_FOLDER, fname)
    with open(fpath, "rb") as f:
        res = pickle.load(f)

    inv_id = res.get("inverter_id", fname.replace("_results.pkl", ""))
    diag   = res.get("nn_diag", {})

    # ----- DC -----
    if "dc" in diag and "loss_curve" in diag["dc"]:
        loss_dc[inv_id] = np.array(diag["dc"]["loss_curve"], dtype=float)

    # ----- AC -----
    if "ac" in diag and "loss_curve" in diag["ac"]:
        loss_ac[inv_id] = np.array(diag["ac"]["loss_curve"], dtype=float)

print(f"Loaded DC loss curves from {len(loss_dc)} inverters")
print(f"Loaded AC loss curves from {len(loss_ac)} inverters")

# ============================================================
# A1. PLOT ALL DC LOSS CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
max_len_dc = max(len(v) for v in loss_dc.values())
all_dc = np.full((len(loss_dc), max_len_dc), np.nan)

for i, (inv, curve) in enumerate(loss_dc.items()):
    ax.plot(curve, alpha=0.3, label=inv)
    all_dc[i, :len(curve)] = curve

mean_dc = np.nanmean(all_dc, axis=0)
std_dc  = np.nanstd(all_dc, axis=0)

ax.plot(mean_dc, color="black", linewidth=2, label="Mean DC Loss")
ax.fill_between(
    np.arange(len(mean_dc)),
    mean_dc - std_dc,
    mean_dc + std_dc,
    alpha=0.15,
    label="±1 std"
)

ax.set_title("Neural Network DC Loss Curves — All Inverters (Plant 2)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "DC_Loss_All.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# A2. PLOT ALL AC LOSS CURVES
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
max_len_ac = max(len(v) for v in loss_ac.values())
all_ac = np.full((len(loss_ac), max_len_ac), np.nan)

for i, (inv, curve) in enumerate(loss_ac.items()):
    ax.plot(curve, alpha=0.3, label=inv)
    all_ac[i, :len(curve)] = curve

mean_ac = np.nanmean(all_ac, axis=0)
std_ac  = np.nanstd(all_ac, axis=0)

ax.plot(mean_ac, color="black", linewidth=2, label="Mean AC Loss")
ax.fill_between(
    np.arange(len(mean_ac)),
    mean_ac - std_ac,
    mean_ac + std_ac,
    alpha=0.15
)

ax.set_title("Neural Network AC Loss Curves — All Inverters (Plant 2)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "AC_Loss_All.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# B. DC vs AC Mean Loss Comparison
# ============================================================

L = min(len(mean_dc), len(mean_ac))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mean_dc[:L], label="Mean DC Loss", linewidth=2)
ax.plot(mean_ac[:L], label="Mean AC Loss", linewidth=2)

ax.set_title("Mean Loss Comparison: DC vs AC — Plant 2")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "Mean_DC_vs_AC.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# C. Convergence Speed (per inverter)
# ============================================================

def get_convergence_epoch(loss, tol=1e-4, patience=10):
    """Returns epoch where improvement slows down."""
    best = loss[0]
    count = 0
    for i in range(1, len(loss)):
        if loss[i] < best - tol:
            best = loss[i]
            count = 0
        else:
            count += 1
        if count >= patience:
            return i
    return len(loss)

conv_dc = {inv: get_convergence_epoch(curve) for inv, curve in loss_dc.items()}
conv_ac = {inv: get_convergence_epoch(curve) for inv, curve in loss_ac.items()}

fig, ax = plt.subplots(figsize=(14, 6))
invs = list(conv_dc.keys())

ax.bar(invs, [conv_dc[i] for i in invs], alpha=0.6, label="DC")
ax.bar(invs, [conv_ac.get(i, np.nan) for i in invs], alpha=0.6, label="AC")

ax.set_title("Convergence Epoch per Inverter — Plant 2")
ax.set_ylabel("Epoch")
ax.set_xticklabels(invs, rotation=45, ha="right")
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "Convergence_Epochs.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n✅ Training visualization complete (Plant 2).")
print(f"Plots saved in: {PLOTS_FOLDER}")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# CONFIG  —  PLANT 2 (Using Plant-1 style trimming)
# ==============================================================
RESULTS_FOLDER = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\02 Plant2_Inverter_Models"
BIASVAR_FOLDER = os.path.join(RESULTS_FOLDER, "00_BiasVariance")

os.makedirs(BIASVAR_FOLDER, exist_ok=True)

MODELS = ["Linear", "Ridge", "Lasso", "RandomForest", "NeuralNet"]


# ==============================================================
# 1. LOAD ALL PER-INVERTER RESULTS (.pkl files)
# ==============================================================

inverter_results = {}

nn_loss_dc = {}
nn_loss_ac = {}

nn_diag_dc = {"iterations": {}, "learning_rate": {}, "momentum": {}, "total_weights": {}, "train_time": {}}
nn_diag_ac = {"iterations": {}, "learning_rate": {}, "momentum": {}, "total_weights": {}, "train_time": {}}

for fname in os.listdir(RESULTS_FOLDER):
    if not fname.endswith("_results.pkl"):
        continue

    fpath = os.path.join(RESULTS_FOLDER, fname)
    with open(fpath, "rb") as f:
        res = pickle.load(f)

    inverter_id = res.get("inverter_id", fname.replace("_results.pkl", ""))
    inverter_results[inverter_id] = res

    diag = res.get("nn_diag", {})

    # DC
    if isinstance(diag, dict) and "dc" in diag:
        dc_diag = diag["dc"]
        if "loss_curve" in dc_diag:
            nn_loss_dc[inverter_id] = np.array(dc_diag["loss_curve"], dtype=float)
        for key in nn_diag_dc.keys():
            if key in dc_diag:
                nn_diag_dc[key][inverter_id] = dc_diag[key]

    # AC
    if isinstance(diag, dict) and "ac" in diag:
        ac_diag = diag["ac"]
        if "loss_curve" in ac_diag:
            nn_loss_ac[inverter_id] = np.array(ac_diag["loss_curve"], dtype=float)
        for key in nn_diag_ac.keys():
            if key in ac_diag:
                nn_diag_ac[key][inverter_id] = ac_diag[key]

print(f"Loaded {len(inverter_results)} inverter result files from Plant 2.")


# ==============================================================
# 2. COLLECT METRICS
# ==============================================================

combined_dc_rmse = {m: [] for m in MODELS}
combined_ac_rmse = {m: [] for m in MODELS}
combined_dc_mae  = {m: [] for m in MODELS}
combined_ac_mae  = {m: [] for m in MODELS}

parallel_dc_rmse = {m: [] for m in MODELS}
parallel_ac_rmse = {m: [] for m in MODELS}
parallel_dc_mae  = {m: [] for m in MODELS}
parallel_ac_mae  = {m: [] for m in MODELS}

parallel_dc_rmse_full = {m: [] for m in MODELS}
parallel_ac_rmse_full = {m: [] for m in MODELS}

for inv_id, res in inverter_results.items():

    comb = res["combined"]
    par  = res["parallel"]

    dc_comb = comb["dc"]
    ac_comb = comb["ac"]

    for m in MODELS:
        if m in dc_comb:
            combined_dc_rmse[m].append(dc_comb[m]["rmse"])
            combined_dc_mae[m].append(dc_comb[m]["mae"])

        if m in ac_comb:
            combined_ac_rmse[m].append(ac_comb[m]["rmse"])
            combined_ac_mae[m].append(ac_comb[m]["mae"])

    avg_dc_rmse = par.get("avg_dc_rmse", {})
    avg_ac_rmse = par.get("avg_ac_rmse", {})
    avg_dc_mae  = par.get("avg_dc_mae", {})
    avg_ac_mae  = par.get("avg_ac_mae", {})

    dc_rmse_days = par.get("dc_rmse", {})
    ac_rmse_days = par.get("ac_rmse", {})

    for m in MODELS:

        if m in avg_dc_rmse:
            parallel_dc_rmse[m].append(avg_dc_rmse[m])

        if m in avg_ac_rmse:
            parallel_ac_rmse[m].append(avg_ac_rmse[m])

        if m in avg_dc_mae:
            parallel_dc_mae[m].append(avg_dc_mae[m])

        if m in avg_ac_mae:
            parallel_ac_mae[m].append(avg_ac_mae[m])

        if m in dc_rmse_days:
            parallel_dc_rmse_full[m].append(dc_rmse_days[m])

        if m in ac_rmse_days:
            parallel_ac_rmse_full[m].append(ac_rmse_days[m])


def mean_std(arr):
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))


# ==============================================================
# 3. BIAS–VARIANCE PROXIES
# ==============================================================

bias_proxy_dc = {}
bias_proxy_ac = {}
var_proxy_dc  = {}
var_proxy_ac  = {}

for m in MODELS:

    bias_proxy_dc[m] = mean_std(combined_dc_rmse[m])[0]
    bias_proxy_ac[m] = mean_std(combined_ac_rmse[m])[0]

    all_dc_days = []
    for lst in parallel_dc_rmse_full[m]:
        all_dc_days.extend(lst)

    all_ac_days = []
    for lst in parallel_ac_rmse_full[m]:
        all_ac_days.extend(lst)

    var_proxy_dc[m] = float(np.std(all_dc_days)) if all_dc_days else np.nan
    var_proxy_ac[m] = float(np.std(all_ac_days)) if all_ac_days else np.nan


# ==============================================================
# 4. SCATTER PLOTS
# ==============================================================

labels = MODELS

# DC scatter
x_dc = [var_proxy_dc[m] for m in labels]
y_dc = [bias_proxy_dc[m] for m in labels]

fig, ax = plt.subplots(figsize=(8, 6))
for i, m in enumerate(labels):
    ax.scatter(x_dc[i], y_dc[i])
    ax.text(x_dc[i]*1.01, y_dc[i]*1.01, m)

ax.set_xlabel("Variance proxy (std per-day RMSE, DC)")
ax.set_ylabel("Bias proxy (mean combined RMSE, DC)")
ax.set_title("Bias–Variance Proxy Plane — DC (Plant 2)")
ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(BIASVAR_FOLDER, "bias_variance_scatter_DC.png"))
plt.close(fig)


# AC scatter
x_ac = [var_proxy_ac[m] for m in labels]
y_ac = [bias_proxy_ac[m] for m in labels]

fig, ax = plt.subplots(figsize=(8, 6))
for i, m in enumerate(labels):
    ax.scatter(x_ac[i], y_ac[i])
    ax.text(x_ac[i]*1.01, y_ac[i]*1.01, m)

ax.set_xlabel("Variance proxy (std per-day RMSE, AC)")
ax.set_ylabel("Bias proxy (mean combined RMSE, AC)")
ax.set_title("Bias–Variance Proxy Plane — AC (Plant 2)")
ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(BIASVAR_FOLDER, "bias_variance_scatter_AC.png"))
plt.close(fig)


# ==============================================================
# 5. BAR PLOTS
# ==============================================================

x = np.arange(len(labels))
width = 0.35

# DC bars
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, [bias_proxy_dc[m] for m in labels], width, label="Bias")
ax.bar(x + width/2, [var_proxy_dc[m] for m in labels], width, label="Variance")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title("Bias vs Variance — DC (Plant 2)")
ax.grid(axis="y")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(BIASVAR_FOLDER, "bias_variance_bar_DC.png"))
plt.close(fig)

# AC bars
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, [bias_proxy_ac[m] for m in labels], width, label="Bias")
ax.bar(x + width/2, [var_proxy_ac[m] for m in labels], width, label="Variance")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title("Bias vs Variance — AC (Plant 2)")
ax.grid(axis="y")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(BIASVAR_FOLDER, "bias_variance_bar_AC.png"))
plt.close(fig)


# ==============================================================
# 6. NEURAL NETWORK LEARNING CURVES
# ==============================================================

# --- DC learning curves ---
if len(nn_loss_dc) > 0:

    fig, ax = plt.subplots(figsize=(10, 6))

    max_len_dc = max(len(c) for c in nn_loss_dc.values())
    all_dc = np.full((len(nn_loss_dc), max_len_dc), np.nan)

    for i, (inv, curve) in enumerate(nn_loss_dc.items()):
        ax.plot(np.arange(len(curve)), curve, alpha=0.3)
        all_dc[i, :len(curve)] = curve

    mean_dc_curve = np.nanmean(all_dc, axis=0)
    ax.plot(np.arange(len(mean_dc_curve)), mean_dc_curve, linewidth=2.5, label="Mean")

    ax.set_title("Neural Network Loss — DC (Plant 2)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(BIASVAR_FOLDER, "nn_learning_curves_DC.png"))
    plt.close(fig)


# --- AC learning curves ---
if len(nn_loss_ac) > 0:

    fig, ax = plt.subplots(figsize=(10, 6))

    max_len_ac = max(len(c) for c in nn_loss_ac.values())
    all_ac = np.full((len(nn_loss_ac), max_len_ac), np.nan)

    for i, (inv, curve) in enumerate(nn_loss_ac.items()):
        ax.plot(np.arange(len(curve)), curve, alpha=0.3)
        all_ac[i, :len(curve)] = curve

    mean_ac_curve = np.nanmean(all_ac, axis=0)
    ax.plot(np.arange(len(mean_ac_curve)), mean_ac_curve, linewidth=2.5, label="Mean")

    ax.set_title("Neural Network Loss — AC (Plant 2)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(BIASVAR_FOLDER, "nn_learning_curves_AC.png"))
    plt.close(fig)


# --- DC vs AC MEAN curves (Plant-1 style TRIMMING) ---
if len(nn_loss_dc) > 0 and len(nn_loss_ac) > 0:

    # DC mean
    max_len_dc = max(len(c) for c in nn_loss_dc.values())
    dc_mat = np.full((len(nn_loss_dc), max_len_dc), np.nan)
    for i, c in enumerate(nn_loss_dc.values()):
        dc_mat[i, :len(c)] = c
    mean_dc = np.nanmean(dc_mat, axis=0)

    # AC mean
    max_len_ac = max(len(c) for c in nn_loss_ac.values())
    ac_mat = np.full((len(nn_loss_ac), max_len_ac), np.nan)
    for i, c in enumerate(nn_loss_ac.values()):
        ac_mat[i, :len(c)] = c
    mean_ac = np.nanmean(ac_mat, axis=0)

    # Trim to same length
    L = min(len(mean_dc), len(mean_ac))
    mean_dc = mean_dc[:L]
    mean_ac = mean_ac[:L]

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(L)

    ax.plot(epochs, mean_dc, label="Mean DC")
    ax.plot(epochs, mean_ac, label="Mean AC")

    ax.set_title("Mean NN Loss — DC vs AC (Plant 2)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(BIASVAR_FOLDER, "nn_mean_loss_DC_vs_AC.png"))
    plt.close(fig)


# ==============================================================
# DONE
# ==============================================================

print("\n==============================================")
print(" Bias–Variance analysis for Plant 2 completed.")
print(" Saved plots in:", BIASVAR_FOLDER)
print("==============================================")


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# CONFIG  – PLANT 2
# ==============================================================
RESULTS_FOLDER = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\02 Plant2_Inverter_Models"
PLOTS_FOLDER   = os.path.join(RESULTS_FOLDER, "00_Comparison_Plots")

os.makedirs(PLOTS_FOLDER, exist_ok=True)

# We know we used these 5 models
MODELS = ["Linear", "Ridge", "Lasso", "RandomForest", "NeuralNet"]

# ==============================================================
# 1. LOAD ALL PER-INVERTER RESULTS
# ==============================================================

inverter_results = {}   # inverter_id -> results dict
nn_loss_dc       = {}   # inverter_id -> np.array loss curve (DC)
nn_loss_ac       = {}   # inverter_id -> np.array loss curve (AC)

# global NN diagnostics containers
nn_diag_dc = {
    "iterations": {},
    "learning_rate": {},
    "momentum": {},
    "total_weights": {},
    "train_time": {},
}
nn_diag_ac = {
    "iterations": {},
    "learning_rate": {},
    "momentum": {},
    "total_weights": {},
    "train_time": {},
}

for fname in os.listdir(RESULTS_FOLDER):
    if not fname.endswith("_results.pkl"):
        # skip master file or other pkl's
        continue

    fpath = os.path.join(RESULTS_FOLDER, fname)
    with open(fpath, "rb") as f:
        res = pickle.load(f)

    inverter_id = res.get("inverter_id", fname.replace("_results.pkl", ""))
    inverter_results[inverter_id] = res

    # Collect NN loss curves + diagnostics
    diag = res.get("nn_diag", {})

    # Expected structure:
    # nn_diag = {
    #   "dc": {"iterations":..., "learning_rate":..., "momentum":...,
    #          "total_weights":..., "train_time":..., "loss_curve":[...]},
    #   "ac": {... same keys ...}
    # }
    if "dc" in diag:
        dc_diag = diag["dc"]
        if "loss_curve" in dc_diag:
            nn_loss_dc[inverter_id] = np.array(dc_diag["loss_curve"], dtype=float)
        for key in nn_diag_dc.keys():
            if key in dc_diag:
                nn_diag_dc[key][inverter_id] = dc_diag[key]

    if "ac" in diag:
        ac_diag = diag["ac"]
        if "loss_curve" in ac_diag:
            nn_loss_ac[inverter_id] = np.array(ac_diag["loss_curve"], dtype=float)
        for key in nn_diag_ac.keys():
            if key in ac_diag:
                nn_diag_ac[key][inverter_id] = ac_diag[key]

print(f"Loaded {len(inverter_results)} Plant 2 inverter result files.")


# ==============================================================
# 2. COLLECT METRICS ACROSS INVERTERS
# ==============================================================

# Combined (all days merged)
combined_dc_rmse = {m: [] for m in MODELS}
combined_ac_rmse = {m: [] for m in MODELS}
combined_dc_mae  = {m: [] for m in MODELS}
combined_ac_mae  = {m: [] for m in MODELS}

# Parallel (average per-day metrics stored in avg_* fields)
parallel_dc_rmse = {m: [] for m in MODELS}
parallel_ac_rmse = {m: [] for m in MODELS}
parallel_dc_mae  = {m: [] for m in MODELS}
parallel_ac_mae  = {m: [] for m in MODELS}

for inv_id, res in inverter_results.items():
    comb = res["combined"]
    par  = res["parallel"]

    # --- combined ---
    dc_comb = comb["dc"]
    ac_comb = comb["ac"]

    for m in MODELS:
        if m in dc_comb:
            combined_dc_rmse[m].append(dc_comb[m]["rmse"])
            combined_dc_mae[m].append(dc_comb[m]["mae"])
        if m in ac_comb:
            combined_ac_rmse[m].append(ac_comb[m]["rmse"])
            combined_ac_mae[m].append(ac_comb[m]["mae"])

    # --- parallel (use avg_* dicts) ---
    avg_dc_rmse = par.get("avg_dc_rmse", {})
    avg_ac_rmse = par.get("avg_ac_rmse", {})
    avg_dc_mae  = par.get("avg_dc_mae", {})
    avg_ac_mae  = par.get("avg_ac_mae", {})

    for m in MODELS:
        if m in avg_dc_rmse:
            parallel_dc_rmse[m].append(avg_dc_rmse[m])
        if m in avg_ac_rmse:
            parallel_ac_rmse[m].append(avg_ac_rmse[m])
        if m in avg_dc_mae:
            parallel_dc_mae[m].append(avg_dc_mae[m])
        if m in avg_ac_mae:
            parallel_ac_mae[m].append(avg_ac_mae[m])

# Helper to compute mean & std, ignoring empty lists
def mean_std(arr):
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))


# ==============================================================
# 3. PRINT SUMMARY TABLES (COMBINED vs PARALLEL)
# ==============================================================

print("\n================== PLANT 2 — MODEL COMPARISON: COMBINED DATA ==================")
print("Model         | DC_RMSE(mean±std)    | AC_RMSE(mean±std)    | DC_MAE(mean±std)     | AC_MAE(mean±std)")
print("-----------------------------------------------------------------------------------------------")
for m in MODELS:
    dc_rmse_mean, dc_rmse_std = mean_std(combined_dc_rmse[m])
    ac_rmse_mean, ac_rmse_std = mean_std(combined_ac_rmse[m])
    dc_mae_mean,  dc_mae_std  = mean_std(combined_dc_mae[m])
    ac_mae_mean,  ac_mae_std  = mean_std(combined_ac_mae[m])

    print(f"{m:12s} | "
          f"{dc_rmse_mean:8.3f}±{dc_rmse_std:6.3f} | "
          f"{ac_rmse_mean:8.3f}±{ac_rmse_std:6.3f} | "
          f"{dc_mae_mean:8.3f}±{dc_mae_std:6.3f} | "
          f"{ac_mae_mean:8.3f}±{ac_mae_std:6.3f}")
print("=====================================================================\n")


print("================== PLANT 2 — MODEL COMPARISON: PARALLEL (PER-DAY AVG) ==================")
print("Model         | DC_RMSE(mean±std)    | AC_RMSE(mean±std)    | DC_MAE(mean±std)     | AC_MAE(mean±std)")
print("-----------------------------------------------------------------------------------------------")
for m in MODELS:
    dc_rmse_mean, dc_rmse_std = mean_std(parallel_dc_rmse[m])
    ac_rmse_mean, ac_rmse_std = mean_std(parallel_ac_rmse[m])
    dc_mae_mean,  dc_mae_std  = mean_std(parallel_dc_mae[m])
    ac_mae_mean,  ac_mae_std  = mean_std(parallel_ac_mae[m])

    print(f"{m:12s} | "
          f"{dc_rmse_mean:8.3f}±{dc_rmse_std:6.3f} | "
          f"{ac_rmse_mean:8.3f}±{ac_rmse_std:6.3f} | "
          f"{dc_mae_mean:8.3f}±{dc_mae_std:6.3f} | "
          f"{ac_mae_mean:8.3f}±{ac_mae_std:6.3f}")
print("==========================================================================\n")


# ==============================================================
# 4. PLOT: COMBINED DC vs AC (AVERAGE ACROSS INVERTERS)
# ==============================================================

labels = MODELS
x = np.arange(len(labels))
width = 0.35

avg_combined_dc_rmse = [mean_std(combined_dc_rmse[m])[0] for m in labels]
avg_combined_ac_rmse = [mean_std(combined_ac_rmse[m])[0] for m in labels]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, avg_combined_dc_rmse, width, label="DC RMSE (Combined)")
ax.bar(x + width/2, avg_combined_ac_rmse, width, label="AC RMSE (Combined)")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("RMSE")
ax.set_title("Plant 2 — Average Combined RMSE — DC vs AC across inverters")
ax.legend()
ax.grid(axis="y")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "combined_avg_rmse_dc_vs_ac_P2.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================
# 5. PLOT: PARALLEL DC vs AC (AVERAGE PER-DAY ACROSS INVERTERS)
# ==============================================================

avg_parallel_dc_rmse = [mean_std(parallel_dc_rmse[m])[0] for m in labels]
avg_parallel_ac_rmse = [mean_std(parallel_ac_rmse[m])[0] for m in labels]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, avg_parallel_dc_rmse, width, label="DC RMSE (Parallel avg)")
ax.bar(x + width/2, avg_parallel_ac_rmse, width, label="AC RMSE (Parallel avg)")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("RMSE")
ax.set_title("Plant 2 — Average Per-Day Parallel RMSE — DC vs AC across inverters")
ax.legend()
ax.grid(axis="y")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "parallel_avg_rmse_dc_vs_ac_P2.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================
# 6. PLOT: COMBINED vs PARALLEL (RMSE) PER MODEL
# ==============================================================

fig, ax = plt.subplots(figsize=(12, 6))
width = 0.18
x = np.arange(len(labels))

c_dc = [mean_std(combined_dc_rmse[m])[0] for m in labels]
c_ac = [mean_std(combined_ac_rmse[m])[0] for m in labels]
p_dc = [mean_std(parallel_dc_rmse[m])[0] for m in labels]
p_ac = [mean_std(parallel_ac_rmse[m])[0] for m in labels]

ax.bar(x - 1.5*width, c_dc, width, label="Combined DC")
ax.bar(x - 0.5*width, c_ac, width, label="Combined AC")
ax.bar(x + 0.5*width, p_dc, width, label="Parallel DC")
ax.bar(x + 1.5*width, p_ac, width, label="Parallel AC")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("RMSE")
ax.set_title("Plant 2 — Combined vs Parallel RMSE — DC & AC")
ax.legend()
ax.grid(axis="y")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "combined_vs_parallel_rmse_P2.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================
# 7. BOX PLOTS (DISTRIBUTION ACROSS INVERTERS)
# ==============================================================

# DC combined
fig, ax = plt.subplots(figsize=(10, 5))
data = [combined_dc_rmse[m] for m in labels]
ax.boxplot(data, labels=labels, showmeans=True)
ax.set_ylabel("RMSE")
ax.set_title("Plant 2 — Distribution of DC RMSE (Combined) across inverters")
ax.grid(axis="y")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "boxplot_combined_dc_rmse_P2.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# AC combined
fig, ax = plt.subplots(figsize=(10, 5))
data = [combined_ac_rmse[m] for m in labels]
ax.boxplot(data, labels=labels, showmeans=True)
ax.set_ylabel("RMSE")
ax.set_title("Plant 2 — Distribution of AC RMSE (Combined) across inverters")
ax.grid(axis="y")

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_FOLDER, "boxplot_combined_ac_rmse_P2.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================
# 8. COST FUNCTION PER ITERATION (NEURAL NET) — LOSS CURVES DC & AC
# ==============================================================

# ----- DC cost function -----
if len(nn_loss_dc) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    max_len_dc = max(len(curve) for curve in nn_loss_dc.values())
    all_curves_dc = np.full((len(nn_loss_dc), max_len_dc), np.nan)

    for idx, (inv_id, curve) in enumerate(nn_loss_dc.items()):
        epochs = np.arange(len(curve))
        ax.plot(epochs, curve, alpha=0.3, label=inv_id)
        all_curves_dc[idx, :len(curve)] = curve

    mean_curve_dc = np.nanmean(all_curves_dc, axis=0)
    ax.plot(np.arange(len(mean_curve_dc)), mean_curve_dc,
            linewidth=2.5, label="Mean across inverters")

    ax.set_xlabel("Iteration / Epoch")
    ax.set_ylabel("Loss (Cost Function)")
    ax.set_title("Plant 2 — Neural Network Training Loss — DC (Combined data)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_FOLDER, "nn_loss_curves_all_inverters_DC_P2.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
else:
    print("No DC nn_diag loss_curve found; skipping DC cost-function plot.")

# ----- AC cost function -----
if len(nn_loss_ac) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    max_len_ac = max(len(curve) for curve in nn_loss_ac.values())
    all_curves_ac = np.full((len(nn_loss_ac), max_len_ac), np.nan)

    for idx, (inv_id, curve) in enumerate(nn_loss_ac.items()):
        epochs = np.arange(len(curve))
        ax.plot(epochs, curve, alpha=0.3, label=inv_id)
        all_curves_ac[idx, :len(curve)] = curve

    mean_curve_ac = np.nanmean(all_curves_ac, axis=0)
    ax.plot(np.arange(len(mean_curve_ac)), mean_curve_ac,
            linewidth=2.5, label="Mean across inverters")

    ax.set_xlabel("Iteration / Epoch")
    ax.set_ylabel("Loss (Cost Function)")
    ax.set_title("Plant 2 — Neural Network Training Loss — AC (Combined data)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_FOLDER, "nn_loss_curves_all_inverters_AC_P2.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
else:
    print("No AC nn_diag loss_curve found; skipping AC cost-function plot.")

# ----- DC vs AC mean loss in a single figure -----
if len(nn_loss_dc) > 0 and len(nn_loss_ac) > 0:
    # Compute mean DC
    max_len_dc = max(len(c) for c in nn_loss_dc.values())
    dc_mat = np.full((len(nn_loss_dc), max_len_dc), np.nan)
    for i, c in enumerate(nn_loss_dc.values()):
        dc_mat[i, :len(c)] = c
    mean_dc = np.nanmean(dc_mat, axis=0)

    # Compute mean AC
    max_len_ac = max(len(c) for c in nn_loss_ac.values())
    ac_mat = np.full((len(nn_loss_ac), max_len_ac), np.nan)
    for i, c in enumerate(nn_loss_ac.values()):
        ac_mat[i, :len(c)] = c
    mean_ac = np.nanmean(ac_mat, axis=0)

    # Align lengths
    L = min(len(mean_dc), len(mean_ac))
    mean_dc = mean_dc[:L]
    mean_ac = mean_ac[:L]

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(L)
    ax.plot(epochs, mean_dc, label="Mean DC Loss")
    ax.plot(epochs, mean_ac, label="Mean AC Loss")
    ax.set_xlabel("Iteration / Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Plant 2 — Neural Net Cost Function Comparison — DC vs AC (mean over inverters)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_FOLDER, "nn_mean_loss_DC_vs_AC_P2.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# 9. GLOBAL NN DIAGNOSTICS DISTRIBUTIONS (DC & AC)
# ==============================================================

def plot_nn_diag_histograms(diag_dict_dc, diag_dict_ac, key, pretty_name):
    """
    diag_dict_*[key] = {inverter_id -> value}
    """
    vals_dc = list(diag_dict_dc[key].values())
    vals_ac = list(diag_dict_ac[key].values())

    if len(vals_dc) == 0 and len(vals_ac) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(vals_dc) > 0:
        ax.hist(vals_dc, bins=10, alpha=0.5, label="DC")
    if len(vals_ac) > 0:
        ax.hist(vals_ac, bins=10, alpha=0.5, label="AC")

    ax.set_xlabel(pretty_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Plant 2 — Distribution of Neural Net {pretty_name} across inverters")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fname = f"nn_diag_hist_{key}_P2.png"
    fig.savefig(os.path.join(PLOTS_FOLDER, fname),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nn_diag_bar_means(diag_dict_dc, diag_dict_ac, key, pretty_name):
    vals_dc = list(diag_dict_dc[key].values())
    vals_ac = list(diag_dict_ac[key].values())

    if len(vals_dc) == 0 and len(vals_ac) == 0:
        return

    mean_dc, std_dc = mean_std(vals_dc)
    mean_ac, std_ac = mean_std(vals_ac)

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(2)
    means = [mean_dc, mean_ac]
    stds  = [std_dc, std_ac]

    ax.bar(x, means, yerr=stds, capsize=5, tick_label=["DC", "AC"])
    ax.set_ylabel(pretty_name)
    ax.set_title(f"Plant 2 — Neural Net {pretty_name} — DC vs AC (mean ± std)")
    ax.grid(axis="y")

    fig.tight_layout()
    fname = f"nn_diag_bar_{key}_P2.png"
    fig.savefig(os.path.join(PLOTS_FOLDER, fname),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


pretty_names = {
    "iterations": "Iterations",
    "learning_rate": "Learning Rate",
    "momentum": "Momentum",
    "total_weights": "Total Weights",
    "train_time": "Training Time (s)",
}

for k, nm in pretty_names.items():
    plot_nn_diag_histograms(nn_diag_dc, nn_diag_ac, k, nm)
    plot_nn_diag_bar_means(nn_diag_dc, nn_diag_ac, k, nm)


# ==============================================================
# 10. FINAL MESSAGE
# ==============================================================

print("\n✅ Plant 2 model comparison complete.")
print(f"All Plant 2 comparison plots saved in: {PLOTS_FOLDER}")
