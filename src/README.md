# CIVE70111 Machine Learning Coursework


## Description of Streamlit dashboards

| **Course** | CIVE70111 Machine Learning |
| **Academic Year** | 2025/2026 |
| **Submission Date** | 12/12/2025 |


## Structure of src folder

src/
├── README.md                         # Instruction and description of dashboards
├── requirements.txt                  # Python package dependencies
├── app.py                            # Streamlit codes
├── config.py                         # Configuration setting
├── utils.py                          # Helper functions  
├── lresults/                         # Results from LSTM model
│    ├── pl_ac_ts.csv
│    ...
│    └── results_p2_dc.csv 
├── cresults/                         # Results from Classificatino model
│    ├── ale.plant_fs 
│    ...
│    └── confusion_svm_plant1.csv
├── cmodels/                          # Trained classification model
│    ├── svm_plant1_fs.pkl
│    ...
│    └── svm_plant2_nofs.pkl
├── 02 Plant2_Inverter_Models/        # Trained models in task 3
│    ├── 4UPUqMRk7TRMgml_results.pkl
│    ...
│    └── xoJJ8DcxJEcupym_results.pkl
├── 01 Plant 1_Inverter_Models/       # Trained models in task 3
│    ├── 1BY6WEcLGh8j5v7_results.pkl
│    ...
│    └── zVJPv84UY57bAof_results.pkl
├── 00 Excel clean file/              # Results from task3     
│    ├── Plant 1
│    └── PLant 2  
└── .streamlit                        # Control dark theme
      └──config.toml                   


## How to Run Streamlit

1. Download src folder to desktop

2. Copy the path of the src folder in the desktop

3. Change:
    
  - 3.1 DATA_ROOT in config.py to your current src folder path

    Here:
    # =====================================================================
    # 🗂 BASE DIRECTORY FOR ALL DATA
    # =====================================================================
    DATA_ROOT = r"C:\Users\MSI-NB\Desktop\src"

  - 3.2 BASE_DIR in app.py to your current lresults folder path 

    Here:
    # -------------------------
    # DASHBOARD 3
    # -------------------------
    BASE_DIR = r"C:\Users\MSI-NB\Desktop\src\lresults"

4. Open terminal

5. Open the file in terminal (In my case, I enter cd C:\Users\MSI-NB\Desktop\src)

6. Create environment by entering python -m venv venv

7. Activate environment by entering venv\Scripts\activate.bat

8. Install required modules by entering pip install -r requirements.txt

9. Retrieve link to dashboard by entering streamlit run app.py

10. Copy the link in the terminal and paste it browser

11. Enjoy the dashboard


## Description of the dashboards

---

# Regression model

The regression dashboard provides an interactive interface to explore and compare the regression models developed in Task 3 for both PV plants. It focuses on inverter-level prediction of DC and AC power.

---

## Page 1 – Interactive predictions (fast retrain)

### Sidebars

- **Plant and inverter selection**
  - Select a **plant**: `Plant 1` or `Plant 2`.
  - Select an **inverter**, automatically discovered from cleaned CSV filenames (e.g. `Plant1_1BY6WEcLGh8j5v7_clean.csv`).

- **Target selection**
  - Radio button to choose the prediction target:
    - **DC power (`DC_CLEAN`)**
    - **AC power (`AC_CLEAN`)**

- **Model selection**
  - **Primary model**
    - Single model selected from `MODEL_NAMES`, excluding `"NeuralNet"` (fast retrain uses simpler models).
  - **Models to compare**
    - Multi-select over all models, **including** `"NeuralNet"` (for consistency with full Task 3 results).

- **Environmental conditions**
  - Sliders for each input feature defined in `FEATURES_CONFIG`, e.g.:
    - `IRRADIATION_CLEAN` – irradiance
    - `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`
    - `DAILY_YIELD_CLEAN`, `TOTAL_YIELD_CLEAN`
    - `NUM_OPT`, `NUM_SUBOPT`
    - `hour_cos`, `hour_sin` (time-of-day encoding)
  - Slider ranges (`min`, `max`, `default`, `step`) are derived automatically from historical data for the selected inverter.

### Main content

- **Fast retraining for selected inverter**
  - Loads the cleaned inverter CSV and applies engineered features (time-of-day features, normalised power ratios, temperature delta).
  - Trains lightweight models for both targets:
    - `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `MLPRegressor`
  - Creates a `"NeuralNet"` alias for the MLP model so it can be compared consistently with full Task 3 results.
  - Computes test **RMSE** and **MAE** for DC and AC.

- **Primary model prediction**
  - Uses the current slider values to form a single input vector.
  - The selected **primary model** predicts either `DC_CLEAN` or `AC_CLEAN`.
  - Displays:
    - A **metric tile** with the predicted power.
    - A small table summarising the **current input feature values**.
    - The primary model’s **RMSE** and **MAE** on a held-out test set for the chosen target.

- **Model comparison for current conditions**
  - For all selected comparison models that are available in fast retrain:
    - Computes predictions for the current input.
    - Retrieves their RMSE and MAE.
  - Shows:
    - Bar chart of **predicted power by model**.
    - Bar chart of **test RMSE by model** for the chosen target (AC or DC).
    - A **detailed table** with model name, prediction, RMSE and MAE.
  - If some selected models are only available in full Task 3 results (not fast retrain), the dashboard notes these explicitly.

- **Feature importance of the primary model**
  - Computes feature importance using a unified interface:
    - Linear/Ridge/Lasso → absolute coefficients.
    - RandomForest → `feature_importances_`.
    - MLPRegressor → sum of absolute input-layer weights per feature.
  - Visualises:
    - Horizontal bar chart of **feature importance**, sorted from most to least important.
    - Table of numeric importance values for each feature.
  - Helps interpret which features most influence the power prediction.

- **Notes**
  - A caption clarifies that this page uses **quick retraining** for interactivity.
  - The full Task 3 training (including parallel metrics and neural network diagnostics) is visualised on **Page 2** using pre-computed `*_results.pkl` files.

---

## Page 2 – Task 3 results (from `*_results.pkl`)

This page visualises the full offline results of Task 3 across inverters and models.

### Loading and filtering results

- Loads all `*_results.pkl` files for the selected plant.
- Builds a combined metrics DataFrame with:
  - Inverter ID
  - Model name
  - Side (`DC` / `AC`)
  - RMSE and MAE from the **combined** training.
- Optionally filters to only those models selected in **Models to compare**, with mapping:
  - `"LinearRegression"` (UI) → `"Linear"` (in stored results).

### Combined RMSE/MAE per model

- **Scope selection**
  - Dropdown to view:
    - **All inverters** – aggregated metrics.
    - A **single inverter** – per-model metrics for that inverter.
- **Metrics**
  - For **All inverters**:
    - Aggregates RMSE and MAE by model and side (mean across inverters).
  - For a **single inverter**:
    - Displays the corresponding rows directly.
- **Visualisation**
  - Dataframe of summarised metrics.
  - Two grouped bar charts:
    - **RMSE by model & side** (AC vs DC).
    - **MAE by model & side** (AC vs DC).

### Bias–variance proxies (from parallel per-day training)

- Uses per-day results from the `"parallel"` section of each `*_results.pkl`:
  - `parallel["dc_rmse"][model]` and `parallel["ac_rmse"][model]`.
- For each model and side:
  - **Bias proxy** = mean per-day RMSE.
  - **Variance proxy** = variance of per-day RMSE.
- **DC side**
  - Dataframe of bias/variance proxies indexed by model.
  - Scatter plot:
    - x-axis: variance proxy.
    - y-axis: bias proxy.
    - Labels show model names.
- **AC side**
  - Same structure for AC.
- Helps diagnose **underfitting vs overfitting** patterns for each model.

### Neural network training loss (mean DC vs AC)

- Only shown if `"NeuralNet"` is among **Models to compare**.
- Extracts neural network diagnostics from `nn_diag` for each inverter:
  - DC and AC training loss curves.
- Steps:
  - Collects DC and AC loss curves across inverters.
  - Pads them to a common length and computes the **mean loss curve** per side.
- **Visualisation**
  - Animated Plotly figure with:
    - Two lines: **mean DC loss** and **mean AC loss** as functions of epoch.
    - **Play/Pause** controls to animate training over epochs.
    - A **slider** at the bottom to scrub through epochs manually.
  - Fixed y-axis range for stability.
- Shows how the neural network converges on DC vs AC targets and highlights any differences in learning dynamics.

---

# Classification model (SVM dashboard)

The classification dashboard visualises the performance of SVM-based models that classify inverter operation into optimal vs suboptimal states. It combines prediction, interpretability and threshold analysis.

---

## Sidebar controls

- **Plant selection**
  - Select `Plant 1` or `Plant 2` (mapped internally to `plant1`/`plant2`).

- **Model version selection**
  - Choose between:
    - **No feature selection** – baseline SVM.
    - **Feature selection** – SVM trained on a reduced feature set.
  - Determines which pre-trained SVM, feature list, threshold and ALE data are loaded.

- The current **decision threshold** is displayed in the sidebar as a reference value.

---

## Interactive prediction section

- Users specify feature values through sliders, including:
  - `AC_CLEAN`, `DC_CLEAN`, `IRRADIATION_CLEAN`, `DAILY_YIELD_CLEAN`,
    `TOTAL_YIELD_CLEAN`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, etc.
- Slider ranges are adapted to the chosen plant, especially for `TOTAL_YIELD_CLEAN`.
- Additional features are computed internally:
  - `DC/IRRA = DC_CLEAN / IRRADIATION_CLEAN`
  - `AC/IRRA = AC_CLEAN / IRRADIATION_CLEAN`
- For each feature:
  - A **mini ALE plot** shows how the SVM’s local effect (ALE value) changes with the current slider value.
  - The ALE plot is cached and only recomputed when the slider value changes, to maintain responsiveness.

- **Prediction button**
  - When the user clicks **“Predict”**:
    - Builds a single-row input DataFrame with all selected features.
    - Computes the SVM decision function score.
    - Applies the threshold:
      - Score ≥ threshold → label = **1 (Suboptimal)**.
      - Score < threshold → label = **0 (Optimal)**.
  - Displays:
    - Predicted label (`Optimal (0)` or `Suboptimal (1)`).
    - Numerical score and threshold.
    - A **gauge indicator** showing where the score lies relative to the threshold.

---

## Tabs

### 1. Confusion Matrix

- Shows the confusion matrix of the selected SVM model on its test set.
- Visualised with:
  - x-axis: predicted classes (`Predict Optimal`, `Predict Suboptimal`).
  - y-axis: actual classes (`Actual Optimal`, `Actual Suboptimal`).
  - Counts displayed in each cell, with a distinct grid.

### 2. Metrics

- Displays key performance metrics as metric tiles:
  - **Precision**
  - **Recall**
  - **F1-score**
- Shows a **full metrics table** (e.g. per run, per configuration) to provide more detail.

### 3. SVM score distribution

- Histogram of SVM decision function scores on the test set.
- Bars are coloured by true label:
  - `Optimal`
  - `Suboptimal`
- A vertical line shows the current decision threshold.
- Used to visualise class overlap and separation around the threshold.

### 4. Feature importance (Drop-Column)

- Uses precomputed **drop-column importance**:
  - For each feature, importance = change in F1-score when the feature is removed.
- Bar chart:
  - x-axis: feature.
  - y-axis: importance (ΔF1-score).
  - Sorted in descending order of importance.
- Highlights which features have the largest impact on SVM performance.

---

## Threshold explorer (interactive)

- Separate section below the main tabs.
- Uses SVM scores and true labels from the test set.
- User interacts via a **threshold slider**:
  - Range: min–max of observed SVM scores.
  - Each new threshold produces updated predictions.

- For the chosen threshold, the dashboard:
  - Recomputes predictions and evaluates:
    - **Precision**, **Recall**, **F1-score** (shown as metric tiles).
  - Updates:
    - **Score distribution histogram** with a new threshold line.
    - **Confusion matrix** for the new threshold.

- This allows exploration of trade-offs between **false positives** and **false negatives**, supporting threshold tuning.

---

# LSTM model (Task 6 dashboard)

The LSTM dashboard visualises the performance of Task 6 time-series forecasting models (LSTM) and compares them with simpler baselines such as persistence and moving average, for both AC and DC power.

---

## Sidebar controls

- **Plant selection**
  - Choose `Plant 1` or `Plant 2`.

- **Target type**
  - Radio button: `AC` or `DC`.

- A note in the sidebar clarifies that metrics and plots are derived from the **Task 6 LSTM + baseline results**.

---

## Offline performance – metrics and bar charts

### Raw metrics per inverter

- The dashboard loads per-inverter results from CSV files for the chosen plant and target type.
- Displays a **dataframe** with:
  - Inverter IDs.
  - MAE and RMSE for:
    - `LSTM`
    - `Persistence` baseline
    - `Moving average` baseline

### Average error across inverters

- Computes average MAE and RMSE across all inverters:
  - Columns:
    - `LSTM_MAE`, `Pers_MAE`, `MA_MAE`
    - `LSTM_RMSE`, `Pers_RMSE`, `MA_RMSE`
- Displays metric tiles for:
  - **Average MAE** for LSTM, Persistence, Moving Average.
  - **Average RMSE** for LSTM, Persistence, Moving Average.
- Provides a high-level comparison of model accuracy across the entire plant.

### MAE and RMSE by inverter and model

- **MAE comparison**
  - Reshapes results to long format and plots grouped bar charts:
    - x-axis: inverter.
    - y-axis: MAE.
    - Colour: model (LSTM, Persistence, Moving Average).
- **RMSE comparison**
  - Same style of grouped bar chart, but for RMSE.
- These plots highlight how different inverters vary in difficulty and how the LSTM compares to baselines per inverter.

---

## Time-series visualisation for best inverter

### Interactive multi-line time-series

- Optional section controlled by **“Show interactive time-series plot”** checkbox.
- Loads a time-series dataset with columns such as:
  - `AC_TRUE` / `DC_TRUE` (actual power),
  - `AC_LSTM` / `DC_LSTM` (LSTM forecast),
  - `AC_PERSIST` / `DC_PERSIST` (persistence baseline),
  - `AC_MA` / `DC_MA` (moving average baseline),
  - `IRRADIATION`,
  - `date`,
  - indexed by `Time`.
- Provides checkboxes to toggle lines:
  - Actual power.
  - LSTM forecast.
  - Persistence forecast.
  - Moving average forecast.
  - Irradiation.
- Plots an interactive line chart:
  - X-axis: time.
  - Y-axis: power (and optionally irradiation).
- Enables qualitative inspection of how closely the LSTM follows actual power and how it compares to the baselines over time.

---

## Representative sunny and cloudy days

- Additional section focusing on representative day-level behaviour.

### Day selection based on power variability

- User specifies a **daytime power threshold** (e.g. 50 kW) to define when the system is “on”.
- Filters the time series to daytime periods where actual power exceeds the threshold.
- Groups daytime data by date and computes the **standard deviation of actual power** per day.
  - Lower std → **sunny and stable day**.
  - Higher std → **cloudy and variable day**.
- Selects one or two representative days:
  - A **sunny & stable day** (minimum std).
  - A **cloudy & variable day** (maximum std).
  - If only one day is available, treats it as a single representative day.

### Per-day time-series plots

- For each representative day:
  - Displays checkboxes to toggle:
    - Actual power.
    - LSTM forecast.
    - Persistence baseline.
    - Moving average baseline.
  - Plots time-series lines over that day with:
    - X-axis: time.
    - Y-axis: AC or DC power.
- Allows direct visual comparison of model performance under:
  - Clear-sky, stable conditions.
  - Cloudy, highly variable conditions.
- A caption explains that day classification is based on power variability, mirroring the approach used in the original Task 6 visualisation functions.

---

## Data Sources -->

- **SCADA / Plant measurements**
  - Inverter-level SCADA data for **Plant 1** and **Plant 2**.
  - Key variables include:
    - **Power**: `DC_CLEAN`, `AC_CLEAN`
    - **Energy**: `DAILY_YIELD_CLEAN`, `TOTAL_YIELD_CLEAN`
    - **Status / operation counters**: `NUM_OPT`, `NUM_SUBOPT`
    - Timestamps: `DATE_TIME`
  - Cleaned and stored as one CSV per inverter:
    - e.g. `Plant1_<INVERTER_ID>_clean.csv` in  
      `...\data\00 Excel clean file\Plant 1` and `Plant 2`.

- **Environmental / weather-related features**
  - Irradiation: `IRRADIATION_CLEAN`
  - Temperatures:
    - `AMBIENT_TEMPERATURE`
    - `MODULE_TEMPERATURE`
  - Used directly as model inputs and to form engineered features such as:
    - `DC/IRRA`, `AC/IRRA`
    - `Temp_Delta = MODULE_TEMPERATURE – AMBIENT_TEMPERATURE`
    - Time-of-day encodings `hour_cos`, `hour_sin` from `DATE_TIME`.

- **Engineered features and preprocessing**
  - Additional features created in the pipeline:
    - Cyclic hour features `hour_cos`, `hour_sin`.
    - Ratios `DC/IRRA`, `AC/IRRA` for normalised performance.
    - `Temp_Delta` for temperature-related effects.
  - Preprocessing steps include:
    - Conversion of `DATE_TIME` to `datetime`.
    - Interpolation of numeric variables and forward/backward filling of missing values.
    - Removal of rows with missing values in required feature/target columns.
  - These processed DataFrames feed:
    - **Fast retrain models** in the regression dashboard.
    - **Full Task 3 training** for the offline results.
    - **SVM classification** features and ALE analysis.
    - **LSTM time-series forecasting**.

- **Task-specific result files**
  - **Task 3 – Regression models**
    - Per-inverter result pickles: `*_results.pkl` under:
      - `...\data\01 Plant1_Inverter_Models`
      - `...\data\02 Plant2_Inverter_Models`
    - Each file stores:
      - Combined metrics (`rmse`, `mae` for DC and AC).
      - Per-day (“parallel”) metrics for bias–variance analysis.
      - Neural network diagnostics and loss curves.

  - **Task 4 – Classification (SVM + ALE)**
    - Stored under:  
      `...\data\03 ALE SVM Decision`
    - Files: `results_Run_*.pkl`, containing:
      - Trained SVM models and selected feature sets.
      - Test splits (`X_te`, `y_te`), confusion matrices and metrics.
      - ALE results per feature.
      - Drop-column feature importance values.

  - **Task 5 – Random Forest feature importance**
    - Stored under:  
      `...\data\04 AC DC`
    - Files:
      - `feature_importance_results_AC_DC.pkl` – RF importance for AC vs DC targets.
      - `feature_importance_resultsRF.pkl` – simplified RF importance results.
    - Used to interpret which features are most influential for AC/DC prediction.

  - **Task 6 – LSTM forecasting**
    - Model outputs:  
      `...\data\06 LSTM Forecasting Model\LSTM Forecasting Model.pkl`
    - Aggregated per-inverter metrics:
      - `results_p1_ac.csv`, `results_p1_dc.csv`
      - `results_p2_ac.csv`, `results_p2_dc.csv`
    - Time-series exports (for best inverters and visualisation):  
      `p1_ac_ts.csv`, `p1_dc_ts.csv`, `p2_ac_ts.csv`, `p2_dc_ts.csv` in  
      `...\src\lresults`
    - Contain actual power, LSTM forecasts, persistence and moving-average baselines, and irradiation.

- **Train / validation / test usage**
  - **Regression (Task 3 & dashboard)**
    - Uses train–test splits (typically 80/20) on combined inverter data and per-day (“parallel”) splits.
  - **Classification (Task 4)**
    - Uses dedicated training and test sets to compute confusion matrices and metrics; the test set is reused for threshold exploration.
  - **LSTM (Task 6)**
    - Time-series are split into training/validation/test sequences; only test-set results and forecasts are exposed in the dashboard.
