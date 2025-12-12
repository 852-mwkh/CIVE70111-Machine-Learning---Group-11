import os
import datetime as dt

import numpy as np
import pandas as pd

# Disable all plot display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_recall_curve, classification_report, confusion_matrix,
    f1_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

from PyALE import ale

import pickle
from tqdm import tqdm
import logging
logging.getLogger("PyALE").setLevel(logging.WARNING)


import pickle

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def regression_outlier_detection_graph(df, x_col="IRRADIATION_CLEAN",
                                       y_col="AC_CLEAN", z_thresh=3, plot=True):
    df = df.copy()
    mask_valid = df[[x_col, y_col]].notna().all(axis=1)
    if mask_valid.sum() < 10:
        return df

    X = df.loc[mask_valid, [x_col]].values
    y = df.loc[mask_valid, y_col].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    residuals = y - y_pred
    z = (residuals - residuals.mean()) / residuals.std(ddof=0)
    outlier_mask = np.abs(z) > z_thresh

    df_valid = df.loc[mask_valid].copy()
    df_valid["outlier_reg"] = outlier_mask

    df_clean = df_valid.loc[~df_valid["outlier_reg"]].drop(columns=["outlier_reg"])
    df_rest = df.loc[~mask_valid]
    df_result = pd.concat([df_clean, df_rest], axis=0).sort_index()
    return df_result


def clean_weather(df_weather_raw):
    """
    Create IRRADIATION_CLEAN using simple 6:00–18:30 day/night rule,
    drop SOURCE_KEY, and set DATE_TIME as index.
    """
    dfw = df_weather_raw.copy()
    dfw["DATE_TIME"] = pd.to_datetime(dfw["DATE_TIME"])

    day_start = dt.time(6, 0)
    day_end   = dt.time(18, 30)
    dfw["expected_day"] = dfw["DATE_TIME"].dt.time.between(day_start, day_end)

    dfw["IRRADIATION_CLEAN"] = dfw["IRRADIATION"].copy()
    dfw.loc[(~dfw["expected_day"]) & (dfw["IRRADIATION_CLEAN"] > 0), "IRRADIATION_CLEAN"] = 0

    dfw.set_index("DATE_TIME", inplace=True)
    if "SOURCE_KEY" in dfw.columns:
        dfw = dfw.drop(columns=["SOURCE_KEY"])

    return dfw


def aggregate_inverters(df_gen_clean):
    """
    Aggregate generation data per inverter and time, and count Optimal/Suboptimal.
    Returns dict: {source_key: aggregated_df}
    """
    agg_dict = {}
    grouped = df_gen_clean.groupby("SOURCE_KEY")
    for sk, g in grouped:
        agg_df = g.groupby("DATE_TIME").agg(
            SOURCE_KEY=("SOURCE_KEY", "first"),
            DC_POWER=("DC_POWER", "first"),
            AC_POWER=("AC_POWER", "first"),
            DAILY_YIELD=("DAILY_YIELD", "first"),
            TOTAL_YIELD=("TOTAL_YIELD", "first"),
            NUM_OPT=("Operating_Condition", lambda x: (x == "Optimal").sum()),
            NUM_SUBOPT=("Operating_Condition", lambda x: (x == "Suboptimal").sum())
        ).reset_index()
        agg_dict[sk] = agg_df
    return agg_dict


def merge_inverter_weather(agg_inv_dict, df_weather_clean):
    """
    Inner-join each inverter df with weather df on matching DATE_TIME index.
    Returns dict: {source_key: joined_df}
    """
    joined = {}
    for sk, inv_df in agg_inv_dict.items():
        d = inv_df.copy()
        d["DATE_TIME"] = pd.to_datetime(d["DATE_TIME"])
        d.set_index("DATE_TIME", inplace=True)
        join_df = d.join(df_weather_clean, how="inner")
        joined[sk] = join_df
    return joined


def clean_ac_dc_dict(wea_inv_dict):
    """
    Clean AC_POWER and DC_POWER into AC_CLEAN/DC_CLEAN based on IRRADIATION_CLEAN.
    Returns dict on the same keys.
    """
    cleaned = {}
    for sk, df_join in wea_inv_dict.items():
        d = df_join.copy()
        d["AC_CLEAN"] = d["AC_POWER"].copy()
        d["DC_CLEAN"] = d["DC_POWER"].copy()

        night_mask = d["IRRADIATION_CLEAN"] == 0
        d.loc[night_mask & (d["AC_CLEAN"] > 0), "AC_CLEAN"] = 0
        d.loc[night_mask & (d["DC_CLEAN"] > 0), "DC_CLEAN"] = 0

        day_mask = d["IRRADIATION_CLEAN"] > 0
        d.loc[day_mask & (d["AC_CLEAN"] == 0), "AC_CLEAN"] = float("nan")
        d.loc[day_mask & (d["DC_CLEAN"] == 0), "DC_CLEAN"] = float("nan")

        d["AC_CLEAN"] = d["AC_CLEAN"].interpolate(method="linear")
        d["DC_CLEAN"] = d["DC_CLEAN"].interpolate(method="linear")

        d["AC_CLEAN"] = d["AC_CLEAN"].fillna(0)
        d["DC_CLEAN"] = d["DC_CLEAN"].fillna(0)

        cleaned[sk] = d
    return cleaned


def clean_daily_yield_dict(acdc_dict):
    """
    Enforce DAILY_YIELD_CLEAN:
      - 0 at night
      - monotonic increasing during daytime
      - flat after sunset
    Returns dict with DAILY_YIELD_CLEAN added.
    """
    cleaned = {}
    for sk, df_in in acdc_dict.items():
        d = df_in.copy()
        d.index = pd.to_datetime(d.index)
        d["DAILY_YIELD_CLEAN"] = d["DAILY_YIELD"].copy()

        dates = np.unique(d.index.date)
        for day in dates:
            mask_day_full = d.index.date == day
            df_day = d.loc[mask_day_full]

            irr_pos = df_day["IRRADIATION_CLEAN"] > 0
            if not irr_pos.any():
                d.loc[mask_day_full, "DAILY_YIELD_CLEAN"] = 0.0
                continue

            day_start_idx = df_day[irr_pos].index[0]
            day_end_idx   = df_day[irr_pos].index[-1]

            night_mask   = mask_day_full & (d.index < day_start_idx)
            day_mask     = mask_day_full & (d.index >= day_start_idx) & (d.index <= day_end_idx)
            evening_mask = mask_day_full & (d.index > day_end_idx)

            d.loc[night_mask, "DAILY_YIELD_CLEAN"] = 0.0
            val_end = d.at[day_end_idx, "DAILY_YIELD"]
            d.loc[evening_mask, "DAILY_YIELD_CLEAN"] = val_end

            day_idx = d.loc[day_mask].index
            if len(day_idx) == 0:
                continue

            raw_vals = d.loc[day_idx, "DAILY_YIELD_CLEAN"].values.astype(float)
            invalid = np.zeros(len(raw_vals), dtype=bool)

            invalid |= raw_vals <= 0
            if len(raw_vals) > 1:
                drops = np.diff(raw_vals) < 0
                invalid[1:][drops] = True

            d.loc[day_idx[invalid], "DAILY_YIELD_CLEAN"] = np.nan
            d.loc[day_idx, "DAILY_YIELD_CLEAN"] = (
                d.loc[day_idx, "DAILY_YIELD_CLEAN"]
                .interpolate(method="linear", limit_direction="both")
            )

            prev_val = d.at[day_idx[0], "DAILY_YIELD_CLEAN"]
            for t in day_idx[1:]:
                cur = d.at[t, "DAILY_YIELD_CLEAN"]
                if pd.isna(cur) or cur < prev_val:
                    d.at[t, "DAILY_YIELD_CLEAN"] = prev_val
                else:
                    prev_val = cur

            d.loc[night_mask, "DAILY_YIELD_CLEAN"] = 0.0
            d.loc[evening_mask, "DAILY_YIELD_CLEAN"] = val_end

        cleaned[sk] = d
    return cleaned

def clean_total_yield_dict(daily_dict):
    """
    Clean TOTAL_YIELD into TOTAL_YIELD_CLEAN using increments in DAILY_YIELD_CLEAN.
    Returns dict with TOTAL_YIELD_CLEAN added, and trimmed columns + OPERATING_CONDITION_CLEAN.
    """
    cleaned = {}
    for sk, df_in in daily_dict.items():
        d = df_in.copy()
        d["TOTAL_YIELD_CLEAN"] = d["TOTAL_YIELD"].copy()
        timestamps = d.index

        for i in range(1, len(timestamps)):
            t_prev = timestamps[i - 1]
            t_curr = timestamps[i]

            TY_prev = d.at[t_prev, "TOTAL_YIELD_CLEAN"]
            TY_now  = d.at[t_curr, "TOTAL_YIELD"]
            DY_prev = d.at[t_prev, "DAILY_YIELD_CLEAN"]
            DY_now  = d.at[t_curr, "DAILY_YIELD_CLEAN"]

            is_new_day = t_curr.date() != t_prev.date()
            if is_new_day:
                d.at[t_curr, "TOTAL_YIELD_CLEAN"] = TY_prev
                continue

            delta_dy = DY_now - DY_prev
            TY_expected = TY_prev + delta_dy

            if TY_now < TY_prev:
                d.at[t_curr, "TOTAL_YIELD_CLEAN"] = TY_expected
            else:
                d.at[t_curr, "TOTAL_YIELD_CLEAN"] = TY_now

        cols_keep = [
            "PLANT_ID", "SOURCE_KEY",
            "AC_CLEAN", "DC_CLEAN",
            "DAILY_YIELD_CLEAN", "TOTAL_YIELD_CLEAN",
            "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
            "IRRADIATION_CLEAN", "NUM_OPT", "NUM_SUBOPT"
        ]
        cols_keep = [c for c in cols_keep if c in d.columns]
        d = d[cols_keep]

        d["OPERATING_CONDITION_CLEAN"] = np.where(
            d["NUM_OPT"] > d["NUM_SUBOPT"], "Optimal", "Suboptimal"
        )
        d = d.drop(columns=["NUM_OPT", "NUM_SUBOPT"])

        cleaned[sk] = d
    return cleaned


def remove_outliers_ps_dict(df_ps_dict):
    """
    Apply regression_outlier_detection_graph to each inverter df.
    """
    out_dict = {}
    for sk, df_in in df_ps_dict.items():
        out_dict[sk] = regression_outlier_detection_graph(
            df_in, x_col="IRRADIATION_CLEAN", y_col="AC_CLEAN",
            z_thresh=3, plot=False
        )
    return out_dict


def make_label(df_all):
    """
    Label: Optimal -> 0, Suboptimal -> 1
    """
    return (df_all["OPERATING_CONDITION_CLEAN"].str.lower() == "suboptimal").astype(int)


def engineer_features(df_all):
    """
    Sort by DATE_TIME per SOURCE_KEY and add AC/IRRA, DC/IRRA.
    """
    df_feat = df_all.groupby("SOURCE_KEY", group_keys=False).apply(
        lambda g: g.sort_values("DATE_TIME")
    )
    df_feat["DC/IRRA"] = df_feat["DC_CLEAN"] / (df_feat["IRRADIATION_CLEAN"] + 1e-3)
    df_feat["AC/IRRA"] = df_feat["AC_CLEAN"] / (df_feat["IRRADIATION_CLEAN"] + 1e-3)
    return df_feat


def assemble_all_from_df_ps(df_ps_dict):
    """
    Combine all inverter dfs into one dataframe.
    """
    parts = []
    for sk, df_inv in df_ps_dict.items():
        d = df_inv.copy()
        d = d.reset_index()  # bring DATE_TIME back as a column
        parts.append(d)

    df_all = pd.concat(parts, ignore_index=True).drop_duplicates()
    df_all["DATE_TIME"] = pd.to_datetime(df_all["DATE_TIME"])

    mask = (~df_all["OPERATING_CONDITION_CLEAN"].isna()) & (~df_all["IRRADIATION_CLEAN"].isna())
    df_all = df_all[mask]

    counts = df_all["OPERATING_CONDITION_CLEAN"].value_counts()
    print("\n=== Operating Condition Counts ===")
    print(f"Number of Optimal (0):     {counts.get('Optimal', 0)}")
    print(f"Number of Suboptimal (1):  {counts.get('Suboptimal', 0)}")

    return df_all


def time_split(df_feat, y, test_days=10, val_days=3):
    """
    Chronological split into train/val/test.
    """
    last_time = df_feat["DATE_TIME"].max()
    test_start = last_time - pd.Timedelta(days=test_days)
    val_start  = test_start - pd.Timedelta(days=val_days)

    mask_test = df_feat["DATE_TIME"] >= test_start
    mask_val  = (df_feat["DATE_TIME"] >= val_start) & (~mask_test)
    mask_train = df_feat["DATE_TIME"] < val_start

    X_tr = df_feat[mask_train]
    X_val = df_feat[mask_val]
    X_te = df_feat[mask_test]

    y_tr = y[mask_train]
    y_val = y[mask_val]
    y_te = y[mask_test]

    return X_tr, X_val, X_te, y_tr, y_val, y_te

def make_preprocessor(df_feat, drop_col):
    """
    StandardScaler on numeric columns not in drop_col.
    """
    num_cols = [
        c for c in df_feat.columns
        if c not in drop_col and df_feat[c].dtype.kind in "fcui"
    ]
    pre = ColumnTransformer(
        [("num", Pipeline([("scaler", StandardScaler())]), num_cols)]
    )
    return pre


def Suboptimal_f1_threshold(y_true, scores_suboptimal):
    """
    Pick threshold that maximises F1 for the Suboptimal (1) class.
    """
    p, r, thr = precision_recall_curve(y_true, scores_suboptimal)
    if len(thr) == 0:
        return 0.0

    f1 = 2 * p[1:] * r[1:] / (p[1:] + r[1:] + 1e-12)
    best_ix = np.nanargmax(f1)
    return float(thr[best_ix])


def Suboptimal_evaluate(name, y_true, scores_suboptimal, thr, tag):
    """
    Print confusion matrix + classification report + PR-AUC focused on suboptimal.
    """
    preds = (scores_suboptimal >= thr).astype(int)
    ap = average_precision_score(y_true, scores_suboptimal)
    print(f"\n==== {name} | {tag} ====")
    print(f"Suboptimal focused Threshold: {thr:.4f} | PR-AUC: {ap:.4f}")
    print(classification_report(y_true, preds, digits=3))
    print("Suboptimal focused Confusion Matrix:\n", confusion_matrix(y_true, preds))


def f1_threshold_scorer(model, X, y_true, thr):
    """
    Compute F1 (Suboptimal=1) for a given model and threshold.
    """
    try:
        scores = model.predict_proba(X)[:, 1]
    except Exception:
        scores = model.decision_function(X)
    preds = (scores > thr).astype(int)
    return f1_score(y_true, preds, pos_label=1)


def plot_ale_1d(model, X, feature, bins=20, save_path=None):
    # Run ALE
    ale(X=X, model=model, feature=[feature], include_CI=False, grid_size=bins)

    # Sanitize filename
    safe_feature = str(feature)
    for bad in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
        safe_feature = safe_feature.replace(bad, "_")

    plt.title(f"ALE for {feature}")
    plt.tight_layout()

    if save_path:
        file = os.path.join(save_path, f"ALE_{safe_feature}.png")
        plt.savefig(file)

    plt.show()  # prevents display


def drop_column_importance(df_feat, baseline_f1, drop_col,
                           X_tr, y_tr, X_val, y_val, X_te, y_te):
    """
    Drop-column importance using LinearSVC: importance = baseline_f1 - dropped_f1.
    """
    importances = {}
    base_drop_cols = set(drop_col)

    for col in X_tr.columns:
        if col in base_drop_cols:
            continue

        X_tr_d = X_tr.drop(columns=[col])
        X_val_d = X_val.drop(columns=[col])
        X_te_d = X_te.drop(columns=[col])

        df_feat_d = df_feat.drop(columns=[col])
        pre_d = make_preprocessor(df_feat_d, drop_col)

        svm_d = Pipeline([
            ("pre", pre_d),
            ("clf", LinearSVC(class_weight="balanced", max_iter=5000))
        ])
        svm_d.fit(X_tr_d, y_tr)

        thr_d = Suboptimal_f1_threshold(y_val, svm_d.decision_function(X_val_d))
        dropped_f1 = f1_threshold_scorer(svm_d, X_te_d, y_te, thr_d)

        importances[col] = baseline_f1 - dropped_f1

    return importances

def run_classification_on_df_inv(df_ps_dict, test_days=10, val_days=3, drop_col=None):
    """
    Full pipeline + SAVE plots + SHOW plots + unique filenames per run.
    ALE is computed on TRAINING data (correct theoretical usage).
    """

    run_count = 0

    # global run_count
    # run_count += 1   # increment unique run ID

    # ================================================================
    # FOLDER SETUP
    # ================================================================

############################################################################################################################################
    
    # Change here 
    # base_path = r"C:\Users\MSI-NB\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data"
    base_path = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data"

######################################################################################################################################################################
    
    folder_main = os.path.join(base_path, "03 ALE SVM Decision")
    folder_plots = os.path.join(folder_main, "Plots")
    folder_ale = os.path.join(folder_plots, "ALE")
    folder_svm = os.path.join(folder_plots, "SVM")

    ensure_dir(folder_main)
    ensure_dir(folder_plots)
    ensure_dir(folder_ale)
    ensure_dir(folder_svm)

    # ================================================================
    # DATA PREPARATION
    # ================================================================
    if drop_col is None:
        drop_col = ["OPERATING_CONDITION_CLEAN", "DATE_TIME", "PLANT_ID", "SOURCE_KEY"]

    
    df = df_ps.reset_index().rename(columns={"index": "DATE_TIME"})
    df["SOURCE_KEY"] = sk
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    
    counts = df["OPERATING_CONDITION_CLEAN"].value_counts()
    print("\n=== Operating Condition Counts ===")
    print(f"Number of Optimal (0):     {counts.get('Optimal', 0)}")
    print(f"Number of Suboptimal (1):  {counts.get('Suboptimal', 0)}")
    
    y = make_label(df)
    
    df_feat = engineer_features(df)

    X_tr, X_val, X_te, y_tr, y_val, y_te = time_split(df_feat, y, test_days, val_days)
    X_tr = X_tr.drop(columns = drop_col)
    X_te = X_te.drop(columns = drop_col)
    X_val = X_val.drop(columns = drop_col)

    pre = make_preprocessor(df_feat,drop_col)

    # By making the weights of optimal prediction in the loss function larger (based on the level of imbalance), the model is better at recongizing the optimal condition despite of the class imbalance
    cw = compute_class_weight("balanced", classes= np.array([0,1]), y=y_tr)
    class_w = {0: cw[0], 1: cw[1]}

    # Logistic Regression 
    lr = Pipeline([("pre", pre),("clf", LogisticRegression(max_iter=5000, class_weight=class_w))])
    lr.fit(X_tr, y_tr)
    thr_lr_sub = Suboptimal_f1_threshold(y_val, lr.predict_proba(X_val)[:,1])
    Suboptimal_evaluate("LogReg - max suboptimal f1 score ", y_te, lr.predict_proba(X_te)[:,1], thr_lr_sub, "Full Test")

    # SVM 
    svm = Pipeline([("pre", pre),("clf", LinearSVC(class_weight="balanced", max_iter=5000))])
    svm.fit(X_tr, y_tr)
    thr_svm_sub = Suboptimal_f1_threshold(y_val, svm.decision_function(X_val)) 
    Suboptimal_evaluate("LinearSVM - max suboptimal f1 score ", y_te, svm.decision_function(X_te), thr_svm_sub, "Full Test")


    # Results suggested that Linear SVC (SVM) has better performance on predicting operating condition compared to logistic regression
    # Thus, SVM is the model used for further interpretation (ALE and Drop-Column Importance) 
    
    # Feature effect on prediction result (ALE curves)(what variation in feature can cause plant to become optimal or suboptimal)
    print("\n=== ALE for SVM ===")
    for feat in X_te.columns:
        plot_ale_1d(svm, X_te, feat)
    
    # Feature importance (Drop-Column Importance, to assess the model performance due to the presence of a feature)
    print("\n=== Baseline F1 Score of SVM ===")
    baseline = f1_threshold_scorer(svm, X_te, y_te, thr_svm_sub)
    print(baseline)
    
    print("\n=== Drop Column Importance for SVM (change of F1 score)===")
    svm_importance = drop_column_importance(df_feat, baseline, drop_col, X_tr, y_tr, X_val, y_val, X_te, y_te)
    for k, v in sorted(svm_importance.items(), key=lambda x: -x[1]):
        print(f"{k:25s}: {v:.4f}")
    
    # Besides class imbalance, the performance of classification model is also based on how well separated the data is for a given set of features, thus features are removed based on Drop-Column Importance to improve separatibility (feature selection)
    # Overlapping 
    scores = svm.decision_function(X_te)
    plt.hist(scores[y_te==0], bins=50, alpha=0.6, label="Optimal")
    plt.hist(scores[y_te==1], bins=50, alpha=0.6, label="Suboptimal")
    plt.axvline(thr_svm_sub, color='k', linestyle='--', label='boundary')
    plt.xlabel("SVM decision function")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    with open(pkl_file, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Saved results to → {pkl_file}\n")


    # Convenience function for running on a single inverter for calculating feature importance
drop = ["OPERATING_CONDITION_CLEAN","DATE_TIME","PLANT_ID","SOURCE_KEY"]
def run_classification_on_df_importance(df_dict, sk, test_days=10, val_days=3, drop_col=drop):

    # df_dict is the dictionary, sk is the inverter key
    df = df_dict[sk].copy()

    df = df.reset_index().rename(columns={"index": "DATE_TIME"})
    df["SOURCE_KEY"] = sk
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

    y = make_label(df)
    df_feat = engineer_features(df)
    
    X_tr, X_val, X_te, y_tr, y_val, y_te = time_split(df_feat, y, test_days, val_days)
    X_tr = X_tr.drop(columns=drop_col)
    X_val = X_val.drop(columns=drop_col)
    X_te = X_te.drop(columns=drop_col)

    feature_list = list(X_tr.columns)

    pre = make_preprocessor(df_feat, drop_col)

    svm = Pipeline([
        ("pre", pre),
        ("clf", LinearSVC(class_weight="balanced", max_iter=5000))
    ])
    svm.fit(X_tr, y_tr)

    thr = Suboptimal_f1_threshold(y_val, svm.decision_function(X_val))
    f1_full = f1_threshold_scorer(svm, X_te, y_te, thr)

    svm_importance = drop_column_importance(
        df_feat=df_feat,
        baseline=f1_full,
        drop_col=drop_col,
        X_tr=X_tr, y_tr=y_tr,
        X_val=X_val, y_val=y_val,
        X_te=X_te, y_te=y_te,
    )

    return {
        "svm": svm,
        "features": feature_list,
        "baseline_f1": f1_full,
        "svm_importance": svm_importance
    }

# ============================================================
# 0. PATHS
# ============================================================

############################################################################################################################################
# Change here 

# folder = r"C:\Users\MSI-NB\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\In"
folder = r"C:\Users\B.KING\OneDrive - Imperial College London\CIVE70111 Machine Learning\CouseWork\Group-11\data\In"
############################################################################################################################################

gen_path_1     = os.path.join(folder, "Plant_1_Generation_Data_updated.csv")   # Plant 1 generation
weather_path_1 = os.path.join(folder, "Plant_1_Weather_Sensor_Data.csv")       # Plant 1 weather

gen_path_2     = os.path.join(folder, "Plant_2_Generation_Data.csv")           # Plant 2 generation
weather_path_2 = os.path.join(folder, "Plant_2_Weather_Sensor_Data.csv")       # Plant 2 weather


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

# ------------------ Plant 1 ------------------

print("\n=== PLANT 1: LOADING DATA ===")
df_p1_gen_raw = pd.read_csv(gen_path_1, parse_dates=["DATE_TIME"])
df_p1_weather_raw = pd.read_csv(weather_path_1, parse_dates=["DATE_TIME"])

# Drop rows with missing Operating_Condition, drop PLANT_ID and 'day' as in original
df_p1_gen = df_p1_gen_raw.dropna().copy()
for col_drop in ["PLANT_ID", "day"]:
    if col_drop in df_p1_gen.columns:
        df_p1_gen = df_p1_gen.drop(columns=[col_drop])
df_p1_gen.set_index("DATE_TIME", inplace=True)

# Aggregate by inverter
df_p1_gen.reset_index(inplace=True)
agg_inv_p1 = aggregate_inverters(df_p1_gen)

# Clean weather
df_p1_weather = clean_weather(df_p1_weather_raw)

# Join inverter + weather
wea_inv_p1 = merge_inverter_weather(agg_inv_p1, df_p1_weather)

# Clean AC/DC, DAILY_YIELD, TOTAL_YIELD
p1_step1 = clean_ac_dc_dict(wea_inv_p1)
p1_step2 = clean_daily_yield_dict(p1_step1)
df_ps1 = clean_total_yield_dict(p1_step2)

# Outlier removal
df_ps1_outlier = remove_outliers_ps_dict(df_ps1)

# ------------------ Plant 2 ------------------

print("\n=== PLANT 2: LOADING DATA ===")
df_p2_gen_raw = pd.read_csv(gen_path_2, parse_dates=["DATE_TIME"])
df_p2_weather_raw = pd.read_csv(weather_path_2, parse_dates=["DATE_TIME"])

# Drop PLANT_ID from generation (as in original)
if "PLANT_ID" in df_p2_gen_raw.columns:
    df_p2_gen = df_p2_gen_raw.drop(columns=["PLANT_ID"]).copy()
else:
    df_p2_gen = df_p2_gen_raw.copy()

df_p2_gen.set_index("DATE_TIME", inplace=True)
df_p2_gen.reset_index(inplace=True)

agg_inv_p2 = aggregate_inverters(df_p2_gen)
df_p2_weather = clean_weather(df_p2_weather_raw)
wea_inv_p2 = merge_inverter_weather(agg_inv_p2, df_p2_weather)

p2_step1 = clean_ac_dc_dict(wea_inv_p2)
p2_step2 = clean_daily_yield_dict(p2_step1)
df_ps2 = clean_total_yield_dict(p2_step2)

df_ps2_outlier = remove_outliers_ps_dict(df_ps2)





# Plant 1 models before feature selection 
for sk in source_key_1:
    run_classification_on_df_inv(df_ps1[sk], drop_col = drop)

# Mean importance across all inverters for Plant 1
drop1 = drop + ['AC_CLEAN','DC_CLEAN','DAILY_YIELD_CLEAN','AC/IRRA','DC/IRRA','MODULE_TEMPERATURE']
all_imp1 = []
for sk in source_key_1:
    out = run_classification_on_df_importance(df_ps1, sk, drop_col = drop1)

    # store as row, not as dict of dicts
    s = []
    s = pd.Series(out["svm_importance"], name=sk)
    all_imp1.append(s)

importance_df1 = pd.DataFrame(all_imp1)
importance_df1.describe().T.sort_values(by="mean", ascending=False)

# Plant 1 models after feature selection 
for sk in source_key_1:
    run_classification_on_df_inv(df_ps1[sk], drop_col = drop1)


# Plant 2 models before feature selection 
for sk in source_key_2:
    run_classification_on_df_inv(df_ps2[sk], drop_col = drop)

# Mean importance across all inverters for Plant 1
drop2 = drop + ['DAILY_YIELD_CLEAN', 'AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','AC_CLEAN', 'TOTAL_YIELD_CLEAN','DC/IRRA'] 
all_imp2 = []
for sk in source_key_2:
    out = run_classification_on_df_importance(df_ps2, sk, drop_col = drop2)

    # store as row, not as dict of dicts
    s = []
    s = pd.Series(out["svm_importance"], name=sk)
    all_imp2.append(s)

importance_df2 = pd.DataFrame(all_imp2)
importance_df2.describe().T.sort_values(by="mean", ascending=False)

# Plant 2 models after feature selection  
for sk in source_key_2:
    run_classification_on_df_inv(df_ps2[sk], drop_col = drop2)