import os
import math
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import pickle  # <-- ADD THIS LINE

from sklearn.metrics import (
    auc, precision_recall_curve, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, average_precision_score,
    mean_absolute_error, mean_squared_error
)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


folder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\In"
gen_path_1 = os.path.join(folder, "Plant_1_Generation_Data_updated.csv")
weather_path_1 = os.path.join(folder, "Plant_1_Weather_Sensor_Data.csv")
gen_path_2 = os.path.join(folder, "Plant_2_Generation_Data.csv")
weather_path_2 = os.path.join(folder, "Plant_2_Weather_Sensor_Data.csv")


### ============================================
### Plant 1: Clean and merge generation + weather
### ============================================

# Correct time issue for Plant 1 generation data
df = pd.read_csv(gen_path_1)
start = pd.Timestamp("2020-05-15")
end = pd.Timestamp("2020-06-18")

df["parsed"] = pd.to_datetime(df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
invalid = df["parsed"].isna() | (~df["parsed"].between(start, end))
df.loc[invalid, "parsed"] = pd.to_datetime(
    df.loc[invalid, "DATE_TIME"], format="%Y-%d-%m %H:%M:%S", errors="coerce"
)
df["DATE_TIME"] = df["parsed"]
dfc = df.drop(columns=["parsed"])

# Drop rows with missing operating condition
plant_1_c = dfc.dropna()
plant_1_c = plant_1_c.drop(columns=["PLANT_ID", "day"])
plant_1_c.set_index("DATE_TIME", inplace=True)

# Separate into inverter dataframes
source_key_1 = plant_1_c["SOURCE_KEY"].unique().tolist()
p1c_gp = plant_1_c.groupby("SOURCE_KEY")
inv_1 = {SOURCE_KEY: group for SOURCE_KEY, group in p1c_gp}

# Aggregate by time for each inverter
agg_inv_1 = {}
for sk, df_inv in inv_1.items():
    agg_df = df_inv.groupby("DATE_TIME").agg(
        SOURCE_KEY=("SOURCE_KEY", "first"),
        DC_POWER=("DC_POWER", "first"),
        AC_POWER=("AC_POWER", "first"),
        DAILY_YIELD=("DAILY_YIELD", "first"),
        TOTAL_YIELD=("TOTAL_YIELD", "first"),
        NUM_OPT=("Operating_Condition", lambda x: (x == "Optimal").sum()),
        NUM_SUBOPT=("Operating_Condition", lambda x: (x == "Suboptimal").sum()),
    ).reset_index()
    agg_inv_1[sk] = agg_df

# Load Plant 1 weather
df = pd.read_csv(weather_path_1)
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

# Day/night rule
day_start = dt.time(6, 0)
day_end = dt.time(18, 30)
df["expected_day"] = df["DATE_TIME"].dt.time.between(day_start, day_end)

# Clean irradiation data
df["IRRADIATION_CLEAN"] = df["IRRADIATION"].copy()
df.loc[(~df["expected_day"]) & (df["IRRADIATION_CLEAN"] > 0), "IRRADIATION_CLEAN"] = 0
df.loc[(df["expected_day"]) & (df["IRRADIATION_CLEAN"] == 0), "IRRADIATION_CLEAN"] = float("nan")
df["IRRADIATION_CLEAN"] = df["IRRADIATION_CLEAN"].interpolate(method="linear")
df["IRRADIATION_CLEAN"] = df["IRRADIATION_CLEAN"].fillna(0)

s1_c = df.copy()
s1_c.set_index("DATE_TIME", inplace=True)
s1_c = s1_c.drop(columns=["SOURCE_KEY"])

# Join inverter data with weather
wea_inv_1 = {}
for sk, df_inv in agg_inv_1.items():
    df_inv = df_inv.set_index("DATE_TIME")
    join_df = df_inv.join(s1_c, how="inner")
    wea_inv_1[sk] = join_df

# Clean AC/DC and DAILY_YIELD for each inverter
df_step_1 = {}
for sk, df_inv in wea_inv_1.items():
    df_clean = df_inv.copy()
    df_clean["AC_CLEAN"] = df_clean["AC_POWER"].copy()
    df_clean["DC_CLEAN"] = df_clean["DC_POWER"].copy()

    night_mask = df_clean["IRRADIATION_CLEAN"] == 0
    df_clean.loc[night_mask & (df_clean["AC_CLEAN"] > 0), "AC_CLEAN"] = 0
    df_clean.loc[night_mask & (df_clean["DC_CLEAN"] > 0), "DC_CLEAN"] = 0

    day_mask = df_clean["IRRADIATION_CLEAN"] > 0
    df_clean.loc[day_mask & (df_clean["AC_CLEAN"] == 0), "AC_CLEAN"] = float("nan")
    df_clean.loc[day_mask & (df_clean["DC_CLEAN"] == 0), "DC_CLEAN"] = float("nan")

    df_clean["AC_CLEAN"] = df_clean["AC_CLEAN"].interpolate(method="linear")
    df_clean["DC_CLEAN"] = df_clean["DC_CLEAN"].interpolate(method="linear")
    df_clean["AC_CLEAN"] = df_clean["AC_CLEAN"].fillna(0)
    df_clean["DC_CLEAN"] = df_clean["DC_CLEAN"].fillna(0)

    df_step_1[sk] = df_clean

# DAILY_YIELD_CLEAN reconstruction
df_step_2 = {}
for sk, df_inv in df_step_1.items():
    df_clean = df_inv.copy()
    df_clean.index = pd.to_datetime(df_clean.index)
    df_clean["DAILY_YIELD_CLEAN"] = df_clean["DAILY_YIELD"].copy()

    dates = np.unique(df_clean.index.date)
    for d in dates:
        day_mask_full = df_clean.index.date == d
        df_day = df_clean.loc[day_mask_full]
        irr_pos = df_day["IRRADIATION_CLEAN"] > 0

        if not irr_pos.any():
            df_clean.loc[day_mask_full, "DAILY_YIELD_CLEAN"] = 0.0
            continue

        day_start_idx = df_day[irr_pos].index[0]
        day_end_idx = df_day[irr_pos].index[-1]

        night_mask = day_mask_full & (df_clean.index < day_start_idx)
        day_mask = day_mask_full & (df_clean.index >= day_start_idx) & (df_clean.index <= day_end_idx)
        evening_mask = day_mask_full & (df_clean.index > day_end_idx)

        df_clean.loc[night_mask, "DAILY_YIELD_CLEAN"] = 0.0
        val_end = df_clean.at[day_end_idx, "DAILY_YIELD"]
        df_clean.loc[evening_mask, "DAILY_YIELD_CLEAN"] = val_end

        day_idx = df_clean.loc[day_mask].index
        if len(day_idx) == 0:
            continue

        raw_vals = df_clean.loc[day_idx, "DAILY_YIELD_CLEAN"].values.astype(float)
        invalid = np.zeros(len(raw_vals), dtype=bool)
        invalid |= raw_vals <= 0

        if len(raw_vals) > 1:
            drops = np.diff(raw_vals) < 0
            invalid[1:][drops] = True

        df_clean.loc[day_idx[invalid], "DAILY_YIELD_CLEAN"] = np.nan
        df_clean.loc[day_idx, "DAILY_YIELD_CLEAN"] = (
            df_clean.loc[day_idx, "DAILY_YIELD_CLEAN"]
            .interpolate(method="linear", limit_direction="both")
        )

        prev_val = df_clean.at[day_idx[0], "DAILY_YIELD_CLEAN"]
        for t in day_idx[1:]:
            cur = df_clean.at[t, "DAILY_YIELD_CLEAN"]
            if pd.isna(cur) or cur < prev_val:
                df_clean.at[t, "DAILY_YIELD_CLEAN"] = prev_val
            else:
                prev_val = cur

        df_clean.loc[night_mask, "DAILY_YIELD_CLEAN"] = 0.0
        df_clean.loc[evening_mask, "DAILY_YIELD_CLEAN"] = val_end

    df_step_2[sk] = df_clean

# TOTAL_YIELD_CLEAN reconstruction
df_ps1 = {}
for sk, df_inv in df_step_2.items():
    df_clean = df_inv.copy()
    df_clean["TOTAL_YIELD_CLEAN"] = df_clean["TOTAL_YIELD"].copy()
    timestamps = df_clean.index

    for i in range(1, len(timestamps)):
        t_prev = timestamps[i - 1]
        t = timestamps[i]

        TY_prev = df_clean.at[t_prev, "TOTAL_YIELD_CLEAN"]
        TY_now = df_clean.at[t, "TOTAL_YIELD"]
        DY_prev = df_clean.at[t_prev, "DAILY_YIELD_CLEAN"]
        DY_now = df_clean.at[t, "DAILY_YIELD_CLEAN"]

        if t.date() != t_prev.date():
            df_clean.at[t, "TOTAL_YIELD_CLEAN"] = TY_prev
            continue

        delta_dy = DY_now - DY_prev
        TY_expected = TY_prev + delta_dy

        if TY_now < TY_prev:
            df_clean.at[t, "TOTAL_YIELD_CLEAN"] = TY_expected
        else:
            df_clean.at[t, "TOTAL_YIELD_CLEAN"] = TY_now

    df_clean = df_clean[
        [
            "PLANT_ID", "SOURCE_KEY",
            "AC_CLEAN", "DC_CLEAN",
            "DAILY_YIELD_CLEAN", "TOTAL_YIELD_CLEAN",
            "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
            "IRRADIATION_CLEAN", "NUM_OPT", "NUM_SUBOPT",
        ]
    ]

    df_clean["OPERATING_CONDITION_CLEAN"] = np.where(
        df_clean["NUM_OPT"] > df_clean["NUM_SUBOPT"],
        "Optimal", "Suboptimal"
    )

    df_clean = df_clean.drop(columns=["NUM_OPT", "NUM_SUBOPT"])
    df_ps1[sk] = df_clean


### ============================================
### Plant 2: Clean and merge generation + weather
### ============================================

plant_2 = pd.read_csv(gen_path_2, parse_dates=["DATE_TIME"])
plant_2 = plant_2.drop(columns=["PLANT_ID"])
plant_2.set_index("DATE_TIME", inplace=True)

p2_gp = plant_2.groupby("SOURCE_KEY")
inv_2 = {SOURCE_KEY: group for SOURCE_KEY, group in p2_gp}
source_key_2 = plant_2["SOURCE_KEY"].unique().tolist()

agg_inv_2 = {}
for sk, df_inv in inv_2.items():
    agg_df = df_inv.groupby("DATE_TIME").agg(
        SOURCE_KEY=("SOURCE_KEY", "first"),
        DC_POWER=("DC_POWER", "first"),
        AC_POWER=("AC_POWER", "first"),
        DAILY_YIELD=("DAILY_YIELD", "first"),
        TOTAL_YIELD=("TOTAL_YIELD", "first"),
        NUM_OPT=("Operating_Condition", lambda x: (x == "Optimal").sum()),
        NUM_SUBOPT=("Operating_Condition", lambda x: (x == "Suboptimal").sum()),
    ).reset_index()
    agg_inv_2[sk] = agg_df

df = pd.read_csv(weather_path_2)
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

day_start = dt.time(6, 0)
day_end = dt.time(18, 30)
df["expected_day"] = df["DATE_TIME"].dt.time.between(day_start, day_end)

df["IRRADIATION_CLEAN"] = df["IRRADIATION"].copy()
df.loc[(~df["expected_day"]) & (df["IRRADIATION_CLEAN"] > 0), "IRRADIATION_CLEAN"] = 0
df.loc[(df["expected_day"]) & (df["IRRADIATION_CLEAN"] == 0), "IRRADIATION_CLEAN"] = float("nan")
df["IRRADIATION_CLEAN"] = df["IRRADIATION_CLEAN"].interpolate(method="linear")
df["IRRADIATION_CLEAN"] = df["IRRADIATION_CLEAN"].fillna(0)

s2_c = df.copy()
s2_c.set_index("DATE_TIME", inplace=True)
s2_c = s2_c.drop(columns=["SOURCE_KEY"])

wea_inv_2 = {}
for sk, df_inv in agg_inv_2.items():
    df_inv = df_inv.set_index("DATE_TIME")
    join_df = df_inv.join(s2_c, how="inner")
    wea_inv_2[sk] = join_df

df_step_1 = {}
for sk, df_inv in wea_inv_2.items():
    df_clean = df_inv.copy()
    df_clean["AC_CLEAN"] = df_clean["AC_POWER"].copy()
    df_clean["DC_CLEAN"] = df_clean["DC_POWER"].copy()

    night_mask = df_clean["IRRADIATION_CLEAN"] == 0
    df_clean.loc[night_mask & (df_clean["AC_CLEAN"] > 0), "AC_CLEAN"] = 0
    df_clean.loc[night_mask & (df_clean["DC_CLEAN"] > 0), "DC_CLEAN"] = 0

    day_mask = df_clean["IRRADIATION_CLEAN"] > 0
    df_clean.loc[day_mask & (df_clean["AC_CLEAN"] == 0), "AC_CLEAN"] = float("nan")
    df_clean.loc[day_mask & (df_clean["DC_CLEAN"] == 0), "DC_CLEAN"] = float("nan")

    df_clean["AC_CLEAN"] = df_clean["AC_CLEAN"].interpolate(method="linear")
    df_clean["DC_CLEAN"] = df_clean["DC_CLEAN"].interpolate(method="linear")
    df_clean["AC_CLEAN"] = df_clean["AC_CLEAN"].fillna(0)
    df_clean["DC_CLEAN"] = df_clean["DC_CLEAN"].fillna(0)

    df_step_1[sk] = df_clean

df_step_2 = {}
for sk, df_inv in df_step_1.items():
    df_clean = df_inv.copy()
    df_clean.index = pd.to_datetime(df_clean.index)
    df_clean["DAILY_YIELD_CLEAN"] = df_clean["DAILY_YIELD"].copy()

    dates = np.unique(df_clean.index.date)
    for d in dates:
        day_mask_full = df_clean.index.date == d
        df_day = df_clean.loc[day_mask_full]
        irr_pos = df_day["IRRADIATION_CLEAN"] > 0

        if not irr_pos.any():
            df_clean.loc[day_mask_full, "DAILY_YIELD_CLEAN"] = 0.0
            continue

        day_start_idx = df_day[irr_pos].index[0]
        day_end_idx = df_day[irr_pos].index[-1]

        night_mask = day_mask_full & (df_clean.index < day_start_idx)
        day_mask = day_mask_full & (df_clean.index >= day_start_idx) & (df_clean.index <= day_end_idx)
        evening_mask = day_mask_full & (df_clean.index > day_end_idx)

        df_clean.loc[night_mask, "DAILY_YIELD_CLEAN"] = 0.0
        val_end = df_clean.at[day_end_idx, "DAILY_YIELD"]
        df_clean.loc[evening_mask, "DAILY_YIELD_CLEAN"] = val_end

        day_idx = df_clean.loc[day_mask].index
        if len(day_idx) == 0:
            continue

        raw_vals = df_clean.loc[day_idx, "DAILY_YIELD_CLEAN"].values.astype(float)
        invalid = np.zeros(len(raw_vals), dtype=bool)
        invalid |= raw_vals <= 0

        if len(raw_vals) > 1:
            drops = np.diff(raw_vals) < 0
            invalid[1:][drops] = True

        df_clean.loc[day_idx[invalid], "DAILY_YIELD_CLEAN"] = np.nan
        df_clean.loc[day_idx, "DAILY_YIELD_CLEAN"] = (
            df_clean.loc[day_idx, "DAILY_YIELD_CLEAN"]
            .interpolate(method="linear", limit_direction="both")
        )

        prev_val = df_clean.at[day_idx[0], "DAILY_YIELD_CLEAN"]
        for t in day_idx[1:]:
            cur = df_clean.at[t, "DAILY_YIELD_CLEAN"]
            if pd.isna(cur) or cur < prev_val:
                df_clean.at[t, "DAILY_YIELD_CLEAN"] = prev_val
            else:
                prev_val = cur

        df_clean.loc[night_mask, "DAILY_YIELD_CLEAN"] = 0.0
        df_clean.loc[evening_mask, "DAILY_YIELD_CLEAN"] = val_end

    df_step_2[sk] = df_clean

df_ps2 = {}
for sk, df_inv in df_step_2.items():
    df_clean = df_inv.copy()
    df_clean["TOTAL_YIELD_CLEAN"] = df_clean["TOTAL_YIELD"].copy()
    timestamps = df_clean.index

    for i in range(1, len(timestamps)):
        t_prev = timestamps[i - 1]
        t = timestamps[i]

        TY_prev = df_clean.at[t_prev, "TOTAL_YIELD_CLEAN"]
        TY_now = df_clean.at[t, "TOTAL_YIELD"]
        DY_prev = df_clean.at[t_prev, "DAILY_YIELD_CLEAN"]
        DY_now = df_clean.at[t, "DAILY_YIELD_CLEAN"]

        if t.date() != t_prev.date():
            df_clean.at[t, "TOTAL_YIELD_CLEAN"] = TY_prev
            continue

        delta_dy = DY_now - DY_prev
        TY_expected = TY_prev + delta_dy

        if TY_now < TY_prev:
            df_clean.at[t, "TOTAL_YIELD_CLEAN"] = TY_expected
        else:
            df_clean.at[t, "TOTAL_YIELD_CLEAN"] = TY_now

    df_clean = df_clean[
        [
            "PLANT_ID", "SOURCE_KEY",
            "AC_CLEAN", "DC_CLEAN",
            "DAILY_YIELD_CLEAN", "TOTAL_YIELD_CLEAN",
            "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
            "IRRADIATION_CLEAN", "NUM_OPT", "NUM_SUBOPT",
        ]
    ]

    df_clean["OPERATING_CONDITION_CLEAN"] = np.where(
        df_clean["NUM_OPT"] > df_clean["NUM_SUBOPT"],
        "Optimal", "Suboptimal"
    )

    df_clean = df_clean.drop(columns=["NUM_OPT", "NUM_SUBOPT"])
    df_ps2[sk] = df_clean


### ============================================
### Task 4 – Linear SVC model
### ============================================

def make_label(df):
    return (df["OPERATING_CONDITION_CLEAN"].str.lower() == "suboptimal").astype(int)

def engineer_features(df):
    df = df.groupby("SOURCE_KEY", group_keys=False).apply(lambda g: g.sort_values("DATE_TIME"))
    if {"DC_CLEAN", "IRRADIATION_CLEAN"}.issubset(df.columns):
        df["DC/IRRA"] = df["DC_CLEAN"] / (df["IRRADIATION_CLEAN"] + 1e-3)
    if {"AC_CLEAN", "IRRADIATION_CLEAN"}.issubset(df.columns):
        df["AC/IRRA"] = df["AC_CLEAN"] / (df["IRRADIATION_CLEAN"] + 1e-3)
    if {"MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"}.issubset(df.columns):
        df["Temp_Delta"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
    if "DATE_TIME" in df.columns:
        t = df["DATE_TIME"]
        hour = t.dt.hour + t.dt.minute / 60
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df

def assemble_all_from_df_ps(df_ps):
    parts = []
    for key, df_inv in df_ps.items():
        df_inv = df_inv.reset_index().rename(columns=lambda x: "DATE_TIME" if x == "index" else x)
        df_inv["SOURCE_KEY"] = key
        df_inv["DATE_TIME"] = pd.to_datetime(df_inv["DATE_TIME"])
        parts.append(df_inv)

    df_all = pd.concat(parts, ignore_index=True).drop_duplicates()
    m = (~df_all["OPERATING_CONDITION_CLEAN"].isna()) & (~df_all["IRRADIATION_CLEAN"].isna())

    print("\n=== Operating Condition Counts ===")
    counts = df_all["OPERATING_CONDITION_CLEAN"].value_counts()
    print(f"Number of Optimal (0):     {counts.get('Optimal', 0)}")
    print(f"Number of Suboptimal (1):  {counts.get('Suboptimal', 0)}")

    return df_all[m]

def time_split(df, y, test_days=10, val_days=3):
    last = df["DATE_TIME"].max()
    test_start = last - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    m_te = df["DATE_TIME"] >= test_start
    m_val = (df["DATE_TIME"] >= val_start) & (~m_te)
    m_tr = df["DATE_TIME"] < val_start

    return df[m_tr], df[m_val], df[m_te], y[m_tr], y[m_val], y[m_te]

def make_preprocessor(df):
    drop = ["OPERATING_CONDITION_CLEAN", "DATE_TIME", "PLANT_ID", "SOURCE_KEY"]
    num_cols = [c for c in df.columns if c not in drop and df[c].dtype.kind in "fcui"]

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        ]
    )
    return pre

def Suboptimal_f1_threshold(y, score_suboptimal):
    p, r, thr = precision_recall_curve(y, score_suboptimal)
    f1 = 2 * p[1:] * r[1:] / (p[1:] + r[1:] + 1e-12)
    return 0.5 if len(thr) == 0 else float(thr[np.nanargmax(f1)])

def Suboptimal_evaluate(name, y, score_suboptimal, thr, tag):
    pred = (score_suboptimal >= thr).astype(int)
    ap = average_precision_score(y, score_suboptimal)

    print(f"\n==== {name} | {tag} ====")
    print(f"Suboptimal Threshold: {thr:.4f} | PR-AUC: {ap:.4f}")
    print(classification_report(y, pred, digits=3))
    print(confusion_matrix(y, pred))

def run_classification_on_df_ps(df_ps, test_days=10, val_days=3):
    df = assemble_all_from_df_ps(df_ps)
    y = make_label(df)
    df_feat = engineer_features(df)

    X_tr, X_val, X_te, y_tr, y_val, y_te = time_split(df_feat, y, test_days, val_days)
    pre = make_preprocessor(df_feat)

    svc = Pipeline(
        [
            ("pre", pre),
            ("clf", SVC(kernel="linear", class_weight="balanced", probability=True)),
        ]
    )
    svc.fit(X_tr, y_tr)

    val_scores = svc.predict_proba(X_val)[:, 1]
    thr = Suboptimal_f1_threshold(y_val, val_scores)

    test_scores = svc.predict_proba(X_te)[:, 1]
    Suboptimal_evaluate("Linear SVC", y_te, test_scores, thr, "Test Set")



### ============================================
### Task 6 – LSTM model (PyTorch)
### ============================================

def make_seq(X, y, L):
    xs, ys = [], []
    for i in range(len(X) - L + 1):
        xs.append(X[i : i + L])
        ys.append(y[i + L - 1])
    return np.array(xs), np.array(ys)

def seq_X_only(arr, L):
    return np.array([arr[i : i + L] for i in range(len(arr) - L + 1)])

def eval_model(y, yhat):
    mae = mean_absolute_error(y, yhat)
    rmse = math.sqrt(mean_squared_error(y, yhat))
    return mae, rmse

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def run_lstm_for_one_df(
    df_raw,
    feature_cols,
    target_col="DC_CLEAN",
    horizon_steps=4,
    window_steps=96,
    inverter_name="",
):
    df = df_raw.sort_index().copy()
    df.index = pd.to_datetime(df.index)
    df["TARGET"] = df[target_col].shift(-horizon_steps)
    df_model = df.dropna(subset=["TARGET"]).copy()

    if len(df_model) <= window_steps + 10:
        print(f"{inverter_name}: insufficient data")
        return None

    n = len(df_model)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df_model.iloc[:train_end]
    val = df_model.iloc[train_end:val_end]
    test = df_model.iloc[val_end:]

    if len(test) <= window_steps:
        print(f"{inverter_name}: test too short")
        return None

    scX = MinMaxScaler().fit(train[feature_cols])
    scY = MinMaxScaler().fit(train[["TARGET"]])

    def scale(df_part):
        return scX.transform(df_part[feature_cols]), scY.transform(df_part[["TARGET"]]).ravel()

    X_train, y_train = scale(train)
    X_val, y_val = scale(val)
    X_test, y_test = scale(test)

    Xtr, ytr = make_seq(X_train, y_train, window_steps)
    Xv, yv = make_seq(X_val, y_val, window_steps)
    Xte, yte = make_seq(X_test, y_test, window_steps)

    test_time = test.index[window_steps - 1 :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1).to(device)
    Xv_t = torch.tensor(Xv, dtype=torch.float32).to(device)
    yv_t = torch.tensor(yv, dtype=torch.float32).view(-1, 1).to(device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True)

    model = LSTMRegressor(input_dim=Xtr.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    streak = 0

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t)
            val_loss = loss_fn(val_pred, yv_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            streak = 0
        else:
            streak += 1
            if streak >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    with torch.no_grad():
        pred_test = model(Xte_t).cpu().numpy().ravel()

    y_pred = scY.inverse_transform(pred_test.reshape(-1, 1)).ravel()
    y_true = scY.inverse_transform(yte.reshape(-1, 1)).ravel()

    tgt_raw = test[target_col].values.reshape(-1, 1)
    tgt_seq = seq_X_only(tgt_raw, window_steps)

    persist = tgt_seq[:, -1, 0]
    movavg = tgt_seq[:, -4:, 0].mean(axis=1)

    mae_lstm, rmse_lstm = eval_model(y_true, y_pred)
    mae_pers, rmse_pers = eval_model(y_true, persist)
    mae_ma, rmse_ma = eval_model(y_true, movavg)

    return {
        "inverter": inverter_name,
        "LSTM_MAE": mae_lstm,
        "LSTM_RMSE": rmse_lstm,
        "Pers_MAE": mae_pers,
        "Pers_RMSE": rmse_pers,
        "MA_MAE": mae_ma,
        "MA_RMSE": rmse_ma,
        "y_true": y_true,
        "y_pred_lstm": y_pred,
        "persist": persist,
        "movavg": movavg,
        "test_time": test_time,
        # >>> NEW: information needed to reload the model <<<
        "model_state_dict": model.state_dict(),
        "input_dim": Xtr.shape[2],
        "hidden_dim": model.lstm.hidden_size,
    }
def run_lstm_for_plant(
    df_dict,
    source_keys,
    feature_cols,
    target_type="DC",
    window_hours=24,
    horizon_minutes=60,
    step_minutes=15,
):
    step_per_hour = int(60 / step_minutes)
    window_steps = window_hours * step_per_hour
    horizon_steps = int(horizon_minutes / step_minutes)

    target_col = f"{target_type}_CLEAN"

    all_results = []

    for key in source_keys:
        df_inv = df_dict[key]
        res = run_lstm_for_one_df(
            df_raw=df_inv,
            feature_cols=feature_cols,
            target_col=target_col,
            horizon_steps=horizon_steps,
            window_steps=window_steps,
            inverter_name=key,
        )
        if res is not None:
            all_results.append(res)

    if not all_results:
        return pd.DataFrame(), pd.DataFrame(), []

    results_df = pd.DataFrame(all_results).set_index("inverter")

    avg_results = pd.DataFrame(
        {
            "LSTM_MAE_avg": [results_df["LSTM_MAE"].mean()],
            "LSTM_RMSE_avg": [results_df["LSTM_RMSE"].mean()],
            "Pers_MAE_avg": [results_df["Pers_MAE"].mean()],
            "Pers_RMSE_avg": [results_df["Pers_RMSE"].mean()],
            "MA_MAE_avg": [results_df["MA_MAE"].mean()],
            "MA_RMSE_avg": [results_df["MA_RMSE"].mean()],
        }
    )

    return results_df, avg_results, all_results


### ============================================
### Run SVC and LSTM for Plants 1 and 2
### ============================================

# ---------------------------
# Task 4 – Linear SVC
# ---------------------------
run_classification_on_df_ps(df_ps1)
run_classification_on_df_ps(df_ps2)

# ---------------------------
# Task 6 – LSTM Forecasting
# ---------------------------
feature_cols = [
    "AC_CLEAN",
    "DC_CLEAN",
    "IRRADIATION_CLEAN",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
]

# Plant 1 – AC
results_p1_ac_df, avg_p1_ac_df, raw_p1_ac = run_lstm_for_plant(
    df_dict=df_ps1,
    source_keys=source_key_1,
    feature_cols=feature_cols,
    target_type="AC",
    window_hours=24,
    horizon_minutes=60,
    step_minutes=15,
)
display(results_p1_ac_df)
display(avg_p1_ac_df)

# Plant 1 – DC
results_p1_dc_df, avg_p1_dc_df, raw_p1_dc = run_lstm_for_plant(
    df_dict=df_ps1,
    source_keys=source_key_1,
    feature_cols=feature_cols,
    target_type="DC",
    window_hours=24,
    horizon_minutes=60,
    step_minutes=15,
)
display(results_p1_dc_df)
display(avg_p1_dc_df)

# Plant 2 – AC
results_p2_ac_df, avg_p2_ac_df, raw_p2_ac = run_lstm_for_plant(
    df_dict=df_ps2,
    source_keys=source_key_2,
    feature_cols=feature_cols,
    target_type="AC",
    window_hours=24,
    horizon_minutes=60,
    step_minutes=15,
)
display(results_p2_ac_df)
display(avg_p2_ac_df)

# Plant 2 – DC
results_p2_dc_df, avg_p2_dc_df, raw_p2_dc = run_lstm_for_plant(
    df_dict=df_ps2,
    source_keys=source_key_2,
    feature_cols=feature_cols,
    target_type="DC",
    window_hours=24,
    horizon_minutes=60,
    step_minutes=15,
)
display(results_p2_dc_df)
display(avg_p2_dc_df)


# ============================================================
# SAVE FORECASTING RESULTS + TRAINED LSTM MODELS
# ============================================================

save_folder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\06 LSTM Forecasting Model"
os.makedirs(save_folder, exist_ok=True)

# ---------- 4.1 Save all results in one PKL ----------

save_path = os.path.join(save_folder, "LSTM Forecasting Model.pkl")

LSTM_results = {
    "Plant1_AC": {
        "per_inverter": results_p1_ac_df,
        "average": avg_p1_ac_df,
        "raw_outputs": raw_p1_ac,
    },
    "Plant1_DC": {
        "per_inverter": results_p1_dc_df,
        "average": avg_p1_dc_df,
        "raw_outputs": raw_p1_dc,
    },
    "Plant2_AC": {
        "per_inverter": results_p2_ac_df,
        "average": avg_p2_ac_df,
        "raw_outputs": raw_p2_ac,
    },
    "Plant2_DC": {
        "per_inverter": results_p2_dc_df,
        "average": avg_p2_dc_df,
        "raw_outputs": raw_p2_dc,
    },
}

with open(save_path, "wb") as f:
    pickle.dump(LSTM_results, f)

print(f"\n✅ LSTM forecasting results saved to:\n{save_path}")

# ---------- 4.2 Save each trained LSTM model as a .pt file ----------

model_dir = os.path.join(save_folder, "models")
os.makedirs(model_dir, exist_ok=True)

def save_model_group(results_list, plant_name, target_type):
    for entry in results_list:
        inv = entry["inverter"]
        state = entry["model_state_dict"]
        input_dim = entry["input_dim"]
        hidden_dim = entry["hidden_dim"]

        filename = f"{plant_name}_{target_type}_{inv}_LSTM.pt"
        file_path = os.path.join(model_dir, filename)

        torch.save(
            {
                "state_dict": state,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
            },
            file_path,
        )

        print(f"📁 Saved model → {file_path}")

# Save Plant 1 models
save_model_group(raw_p1_ac, "Plant1", "AC")
save_model_group(raw_p1_dc, "Plant1", "DC")

# Save Plant 2 models
save_model_group(raw_p2_ac, "Plant2", "AC")
save_model_group(raw_p2_dc, "Plant2", "DC")

print("\n✅ All trained LSTM models saved successfully!")


def visualize_lstm_results(
    df_dict,
    results_df,
    all_results,
    target_type="DC",
    power_threshold=50.0,
    figsize=(16, 6),
    plant_name="Plant"
):
    """
    Visualize AND SAVE LSTM forecast results for each inverter.

    Parameters
    ----------
    df_dict : dict
        Dictionary of inverter dataframes (df_ps1 or df_ps2).
    results_df : DataFrame
        Per-inverter summary metrics returned by run_lstm_for_plant().
    all_results : list of dict
        Raw results list returned by run_lstm_for_plant().
    target_type : str
        "AC" or "DC".
    power_threshold : float
        Filter out low-power nighttime noise.
    figsize : tuple
        Figure size.
    plant_name : str
        "Plant1" or "Plant2".
    """

    # ------------------------
    # Create output folder
    # ------------------------
    save_dir = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\06 LSTM Forecasting Model\Plots"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n==============================")
    print(f" Saving LSTM Plots for {plant_name} ({target_type})")
    print(f"==============================\n")

    for entry in all_results:
        inv = entry["inverter"]

        y_true = entry["y_true"]
        y_pred = entry["y_pred_lstm"]
        persist = entry["persist"]
        movavg = entry["movavg"]
        time_index = entry["test_time"]

        # Filter small values
        valid = y_true > power_threshold
        if valid.sum() < 5:
            print(f"Skipping {inv}: insufficient high-power samples.")
            continue

        t = time_index[valid]
        yt = y_true[valid]
        yp = y_pred[valid]
        yp_persist = persist[valid]
        yp_ma = movavg[valid]

        # ------------------------------
        # Plot 1: True vs Predicted
        # ------------------------------
        plt.figure(figsize=figsize)
        plt.plot(t, yt, label="True Power", linewidth=2)
        plt.plot(t, yp, label="LSTM Forecast", alpha=0.85)
        plt.plot(t, yp_persist, label="Persistence", linestyle="--", alpha=0.7)
        plt.plot(t, yp_ma, label="Moving Avg", linestyle="--", alpha=0.7)

        plt.title(f"{plant_name} – {inv} – {target_type} Forecast")
        plt.xlabel("Time")
        plt.ylabel(f"{target_type} Power")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save forecast plot
        fpath1 = os.path.join(save_dir, f"{plant_name}_{target_type}_{inv}_forecast.png")
        plt.savefig(fpath1, dpi=200)
        plt.close()

        # ------------------------------
        # Plot 2: Error plot
        # ------------------------------
        errors = yt - yp

        plt.figure(figsize=figsize)
        plt.plot(t, errors, label="LSTM Error", linewidth=1.5)
        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"{plant_name} – {inv} – {target_type} Error")
        plt.xlabel("Time")
        plt.ylabel("Error (True - Predicted)")
        plt.grid(True)
        plt.tight_layout()

        # Save error plot
        fpath2 = os.path.join(save_dir, f"{plant_name}_{target_type}_{inv}_error.png")
        plt.savefig(fpath2, dpi=200)
        plt.close()

        print(f"📁 Saved plots for {inv}:")
        print(f"   → {fpath1}")
        print(f"   → {fpath2}")

        # ------------------------------
        # Print Metrics
        # ------------------------------
        print(f"\n----- Metrics for {inv} ({target_type}) -----")
        print(results_df.loc[inv][[
            "LSTM_MAE", "LSTM_RMSE",
            "Pers_MAE", "Pers_RMSE",
            "MA_MAE", "MA_RMSE"
        ]])
        print("\n----------------------------------------------\n")


visualize_lstm_results(
    df_ps1,
    results_p1_dc_df,
    raw_p1_dc,
    "DC",
    50,
    plant_name="Plant1"
)


visualize_lstm_results(
    df_ps1,
    results_p1_ac_df,
    raw_p1_ac,
    "AC",
    50,
    plant_name="Plant1"
)


visualize_lstm_results(
    df_ps2,
    results_p2_ac_df,
    raw_p2_ac,
    "AC",
    50,
    plant_name="Plant2"
)


visualize_lstm_results(
    df_ps2,
    results_p2_ac_df,
    raw_p2_ac,
    "AC",
    50,
    plant_name="Plant2"
)
