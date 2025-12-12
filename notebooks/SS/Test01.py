import os
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

plt.rcParams["figure.figsize"] = (8, 4)


########################################################################################################################################################
# Change Path

folder = r"C:/Users/B.KING/OneDrive - Imperial College London/CIVE70111 Machine Learning/CouseWork/In/"

########################################################################################################################################################
gen_path = os.path.join(folder, "Plant_1_Generation_Data_updated.csv")
weather_path = os.path.join(folder, "Plant_1_Weather_Sensor_Data.csv")


def load_and_fix_generation(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    start = pd.Timestamp("2020-05-15")
    end = pd.Timestamp("2020-06-18")

    df["parsed"] = pd.to_datetime(
        df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    invalid = df["parsed"].isna() | (~df["parsed"].between(start, end))

    df.loc[invalid, "parsed"] = pd.to_datetime(
        df.loc[invalid, "DATE_TIME"], format="%Y-%d-%m %H:%M:%S", errors="coerce"
    )

    df["DATE_TIME"] = df["parsed"]
    df = df.drop(columns=["parsed", "day"], errors="ignore")
    df = df.set_index("DATE_TIME")

    print("Nulls before cleaning:\n", df.isnull().sum())
    df = df.dropna()
    print("\nNulls after dropna:\n", df.isnull().sum())
    return df

plant_1 = load_and_fix_generation(gen_path)
source_key_1 = plant_1["SOURCE_KEY"].unique().tolist()
inv_1 = {sk: g for sk, g in plant_1.groupby("SOURCE_KEY")}


def load_and_clean_weather(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

    day_start, day_end = dt.time(6, 0), dt.time(18, 30)
    df["expected_day"] = df["DATE_TIME"].dt.time.between(day_start, day_end)
    df["irr_anom"] = (~df["expected_day"]) & (df["IRRADIATION"] > 0)

    df["IRRADIATION_CLEAN"] = df["IRRADIATION"].copy()
    df.loc[df["irr_anom"], "IRRADIATION_CLEAN"] = 0

    df = df.set_index("DATE_TIME")
    df = df.drop(columns=["PLANT_ID", "SOURCE_KEY", "expected_day", "irr_anom"], errors="ignore")
    print("\nWeather nulls:\n", df.isnull().sum())
    return df

weather_1 = load_and_clean_weather(weather_path)


def build_agg_inverters(inv_dict):
    agg_inv = {}
    for sk, df in inv_dict.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        agg_inv[sk] = df.groupby(df.index).agg(
            SOURCE_KEY=('SOURCE_KEY', 'first'),
            DC_POWER=('DC_POWER', 'first'),
            AC_POWER=('AC_POWER', 'first'),
            DAILY_YIELD=('DAILY_YIELD', 'first'),
            TOTAL_YIELD=('TOTAL_YIELD', 'first'),
            NUM_OPT=('Operating_Condition', lambda x: (x == "Optimal").sum()),
            NUM_SUBOPT=('Operating_Condition', lambda x: (x == "Suboptimal").sum())
        )
    return agg_inv

agg_inv_1 = build_agg_inverters(inv_1)

def merge_with_weather(agg_inv, weather_df):
    return {sk: df.join(weather_df, how="inner") for sk, df in agg_inv.items()}

wea_inv_1 = merge_with_weather(agg_inv_1, weather_1)


def clean_ac_dc(wea_inv):
    out = {}
    for sk, df in wea_inv.items():
        df = df.copy()
        df["AC_CLEAN"] = df["AC_POWER"].astype(float)
        df["DC_CLEAN"] = df["DC_POWER"].astype(float)

        night = df["IRRADIATION_CLEAN"] == 0
        day = df["IRRADIATION_CLEAN"] > 0

        df.loc[night & (df["AC_CLEAN"] > 0), "AC_CLEAN"] = 0
        df.loc[night & (df["DC_CLEAN"] > 0), "DC_CLEAN"] = 0

        df.loc[day & (df["AC_CLEAN"] == 0), "AC_CLEAN"] = np.nan
        df.loc[day & (df["DC_CLEAN"] == 0), "DC_CLEAN"] = np.nan

        df["AC_CLEAN"] = df["AC_CLEAN"].interpolate().fillna(0)
        df["DC_CLEAN"] = df["DC_CLEAN"].interpolate().fillna(0)
        out[sk] = df
    return out

df_step_1 = clean_ac_dc(wea_inv_1)


def clean_daily_yield(inv_dict):
    out = {}
    for sk, df in inv_dict.items():
        df = df.copy()
        df["DAILY_YIELD_CLEAN"] = df["DAILY_YIELD"].astype(float)
        days = np.unique(df.index.date)

        for d in days:
            m = df.index.date == d
            df_day = df.loc[m]
            irr_pos = df_day["IRRADIATION_CLEAN"] > 0

            if not irr_pos.any():
                df.loc[m, "DAILY_YIELD_CLEAN"] = 0
                continue

            start, end = df_day[irr_pos].index[[0, -1]]

            night = m & (df.index < start)
            daymask = m & (df.index >= start) & (df.index <= end)
            evening = m & (df.index > end)

            df.loc[night, "DAILY_YIELD_CLEAN"] = 0
            end_val = df.at[end, "DAILY_YIELD_CLEAN"]
            df.loc[evening, "DAILY_YIELD_CLEAN"] = end_val

            vals = df.loc[daymask, "DAILY_YIELD_CLEAN"]
            bad = (vals <= 0) | (vals.diff() < 0)
            df.loc[bad.index[bad], "DAILY_YIELD_CLEAN"] = np.nan

            df.loc[daymask, "DAILY_YIELD_CLEAN"] = df.loc[daymask, "DAILY_YIELD_CLEAN"].interpolate()

        out[sk] = df
    return out

df_step_2 = clean_daily_yield(df_step_1)


def clean_total_yield(inv_dict):
    out = {}
    for sk, df in inv_dict.items():
        df = df.copy()
        df["TOTAL_YIELD_CLEAN"] = df["TOTAL_YIELD"].astype(float)

        ts = df.index
        for i in range(1, len(ts)):
            t0, t = ts[i-1], ts[i]
            new_day = t.date() != t0.date()

            if new_day:
                df.at[t, "TOTAL_YIELD_CLEAN"] = df.at[t0, "TOTAL_YIELD_CLEAN"]
                continue

            delta = df.at[t, "DAILY_YIELD_CLEAN"] - df.at[t0, "DAILY_YIELD_CLEAN"]
            expected = df.at[t0, "TOTAL_YIELD_CLEAN"] + delta

            if df.at[t, "TOTAL_YIELD"] < df.at[t0, "TOTAL_YIELD_CLEAN"]:
                df.at[t, "TOTAL_YIELD_CLEAN"] = expected

        out[sk] = df
    return out

df_step_3 = clean_total_yield(df_step_2)


df_ps1 = {}
for sk, df in df_step_3.items():
    df = df.copy()
    df["OPERATING_CONDITION_CLEAN"] = np.where(
        df["NUM_OPT"] > df["NUM_SUBOPT"], "Optimal", "Suboptimal"
    )
    df_ps1[sk] = df[
        [
            "SOURCE_KEY", "AC_CLEAN", "DC_CLEAN",
            "DAILY_YIELD_CLEAN", "TOTAL_YIELD_CLEAN",
            "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
            "IRRADIATION_CLEAN", "OPERATING_CONDITION_CLEAN"
        ]
    ]

print(df_ps1[source_key_1[0]].head())

def regression_outlier_detection_graph(df, x_col="IRRADIATION_CLEAN",
                                       y_col="AC_CLEAN", z_thresh=3, plot=True):
    df = df.copy().dropna(subset=[x_col, y_col])
    X = df[[x_col]].values
    y = df[y_col].values

    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    z = (residuals - residuals.mean()) / residuals.std()

    df["outlier"] = np.abs(z) > z_thresh
    df_clean = df.loc[~df["outlier"]].copy().drop(columns=["outlier"])

    if plot:
        plt.scatter(df[x_col], df[y_col], label="Normal", alpha=0.3)
        plt.scatter(df.loc[df["outlier"], x_col],
                    df.loc[df["outlier"], y_col],
                    color="red", label="Outliers")
        plt.legend()
        plt.show()

    return df_clean

df_ps1_outlier = {
    sk: regression_outlier_detection_graph(df, plot=False)
    for sk, df in df_ps1.items()
}


def plot_raw_plant_time_series(df, max_cols=6):
    cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    for c in cols:
        plt.plot(df.index, df[c]); plt.title(c); plt.show()

def plot_inverter_raw(inv, sk, max_cols=6):
    df = inv[sk].select_dtypes(include=[np.number]).iloc[:, :max_cols]
    for c in df.columns:
        plt.plot(df.index, df[c]); plt.title(f"{sk} - {c}"); plt.show()

def plot_weather_time_series(df):
    for c in df.columns:
        plt.plot(df.index, df[c]); plt.title(c); plt.show()

def plot_clean_inverter_time_series(df_ps1, sk):
    df = df_ps1[sk].select_dtypes(include=[np.number])
    for c in df.columns:
        plt.plot(df.index, df[c]); plt.title(c); plt.show()

def plot_zoom_5_days(df_ps1, sk, days=5):
    df = df_ps1[sk]
    start = df.index.min()
    end = start + pd.Timedelta(days=days)
    df2 = df.loc[start:end].select_dtypes(include=[np.number])
    for c in df2.columns:
        plt.plot(df2.index, df2[c]); plt.title(c); plt.show()

def scatter_ac_vs_others(df_ps1, sk):
    df = df_ps1[sk].select_dtypes(include=[np.number])
    for c in df.columns:
        if c != "AC_CLEAN":
            plt.scatter(df["AC_CLEAN"], df[c]); plt.title(c); plt.show()

def plot_histograms_for_inverter(df_ps1, sk):
    df = df_ps1[sk].select_dtypes(include=[np.number])
    for c in df.columns:
        sns.histplot(df[c], kde=True); plt.title(c); plt.show()

def plot_correlation_heatmap(df_ps1, sk):
    df = df_ps1[sk].select_dtypes(include=[np.number])
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm"); plt.show()

def plot_outlier_residuals_for_inverter(df_ps1, sk, 
                                        x_col="IRRADIATION_CLEAN",
                                        y_col="AC_CLEAN",
                                        z_thresh=3):
    """
    Visualize regression-based outliers for a given inverter dataframe.
    Outliers = |z-score of residual| > z_thresh.
    """

    df = df_ps1[sk].copy()
    df = df.dropna(subset=[x_col, y_col])

    X = df[[x_col]].values
    y = df[y_col].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Compute residual z-scores
    residuals = y - y_pred
    z = (residuals - residuals.mean()) / residuals.std()

    df["outlier"] = np.abs(z) > z_thresh

    # Plot
    plt.figure(figsize=(8, 5))
    
    # Non-outliers
    plt.scatter(
        df.loc[~df["outlier"], x_col],
        df.loc[~df["outlier"], y_col],
        alpha=0.4,
        label="Normal"
    )

    # Outliers
    plt.scatter(
        df.loc[df["outlier"], x_col],
        df.loc[df["outlier"], y_col],
        color="red",
        label="Outliers"
    )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Outlier Detection (Inverter {sk}) — z > {z_thresh}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Statistical summary of cleaned data
for sk , df in df_ps1.items():
    df = df.drop(columns = ['PLANT_ID','SOURCE_KEY'])
    print('Plant 1 Inverter ' + sk)
    print(df.describe().T)


# statistic summary of one inverter
eg1 = df_ps1[source_key_1[0]].drop(columns = ['PLANT_ID', 'SOURCE_KEY'])
eg1.describe().T

# Time series plots of the entire plant
plot_raw_plant_time_series(plant_1)
# Time series plots of one invertre after handling DATE_TIME Issues
plot_inverter_raw(inv_1, source_key_1[0])
# Time series plots of the weather data before cleaning
plot_weather_time_series(weather_1)
# Time series plots of one inverter after cleaning
plot_clean_inverter_time_series(df_ps1, source_key_1[0])
# Zoomed in time series plots of one inverter
plot_zoom_5_days(df_ps1, source_key_1[0])
# Histograms of one inverter
plot_histograms_for_inverter(df_ps1, source_key_1[0])
# Similar correlation matrix for every inverter
for sk , df in df_ps1.items():
    print('Plant 1 Inverter ' + sk)
    df = df.drop(columns = ['PLANT_ID','SOURCE_KEY','OPERATING_CONDITION_CLEAN'])
    print(df.corr())

# Sample correlation matrix for one of the inverter
eg4 = df_ps1[source_key_1[0]].drop(columns = ['PLANT_ID','SOURCE_KEY','OPERATING_CONDITION_CLEAN'])
eg4.corr()

# Find correlation between features of one inverter
plot_correlation_heatmap(df_ps1, source_key_1[0])

# Find correlation between features of one inverter
plot_correlation_heatmap(df_ps1, source_key_1[0])

# Scatter plot of AC vs other features to check for outliers
scatter_ac_vs_others(df_ps1, source_key_1[0])

# Identify outliers using regression residuals
plot_outlier_residuals_for_inverter(df_ps1, source_key_1[0])   