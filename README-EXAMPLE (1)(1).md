# CIVE70111 Machine Learning Coursework

> **⚠️ IMPORTANT: Your repository must be set to PRIVATE before submission. Go to Settings → Change visibility → Make private. Delete this warning once you have confirmed your repository is private.**

## Project Information

| **Course** | CIVE70111 Machine Learning |
| **Academic Year** | 2025/2026 |
| **Submission Date** | 09/12/2025 |

## Student Information

| Name                        | CID      | GitHub Username                |
|-----------------------------|----------|--------------------------------|
| Peng Yongxi                 | 06050061 | john.peng25@imperial.ac.uk     |
| Yuanqi Zhao                 | 06043840 | yuanqi.zhao25@imperial.ac.uk   |
| King Hung Michael Wong      | 06070058 | kk2725@ic.ac.uk                |
| Haojiang Wang               | 06063691 | haojiangwang25@imperial.ac.uk  |
| Weiyi Bao                   | 06059246 | weiyi.bao25@imperial.ac.uk     |
| Steven Wong                 | 06042971 |steven.sze-yin25@imperial.ac.uk |

## Executive Summary

This project investigates two solar photovoltaic (PV) plants using machine-learning methods to analyse inverter-level power generation and operational conditions. The dataset contains 34 days of 15-minute measurements, combining AC/DC output from individual inverters with plant-level weather sensor recordings. Substantial preprocessing was required to prepare the data for modelling, including reconstruction of missing timestamps, filtering of night-time noise, correction of counter resets, smoothing of irradiation signals and engineering of features such as temperature differences and cyclical encodings of time.

A structured, task-based workflow was followed. The analysis began with exploratory data assessment and feature engineering, which formed the foundation for model development. Models such as linear regression, random forest, neural neworks were compared to forecast AC and DC power, several classification methods were explored to identify inverter operating conditions. Interpretability tools were then applied to the selected models, and short-horizon temporal forecasting was implemented using a simple LSTM architecture. The project concludes with the creation of an interactive Streamlit dashboard that allows users to input environmental conditions, generate model predictions and visualise model behaviour in real time.

This work demonstrates an integrated data-processing and modelling pipeline for PV performance analysis, combining statistical exploration, supervised learning, model interpretation and applied deployment tools.

## Project Structure

```
your-project/
├── README.md
├── requirements.txt
│
├── data
│   └── In
│       ├── Plant_1_Generation_Data_updated.csv
│       ├── Plant_1_Weather_Sensor_Data.csv
│       ├── Plant_2_Generation_Data.csv
│       └── Plant_2_Weather_Sensor_Data.csv
│
├── notebooks
│   ├── 01_Task 1.ipynb
│   ├── 02_Task_2_Plant_1.ipynb
│   ├── 02_Task_2_Plant_2.ipynb
│   ├── 03_Task_3_Plant_1_and_2.ipynb
│   ├── 04_Task_4_and_5_Classifcation_model_and_model_interpretation.ipynb
│   ├── 05_Task_5_AC&DC.ipynb
│   ├── 05_Task_5_Random_Forest.ipynb
│   └── 06_Task_6.ipynb
│
├── src/
│   ├── app.py
│   ├── config.py
│   ├── Utilities.py
│   └── utils.py
│
└── docs/
    └── …


```

## How to Run
1. Install Dependencies 
pip install -r requirements.txt

Group-11
├── data
│   └── In
│       ├── Plant_1_Generation_Data_updated.csv
│       ├── Plant_1_Weather_Sensor_Data.csv
│       ├── Plant_2_Generation_Data.csv
│       └── Plant_2_Weather_Sensor_Data.csv
├── docs
├── notebooks
│   ├── 01_Task 1.ipynb
│   ├── 02_Task_2_Plant_1.ipynb
│   ├── 02_Task_2_Plant_2.ipynb
│   ├── 03_Task_3_Plant_1_and_2.ipynb
│   ├── 04_Task_4_and_5_Classifcation_model_and_model_interpretation.ipynb
│   ├── 05_Task_5_AC&DC.ipynb
│   ├── 05_Task_5_Random_Forest.ipynb
│   └── 06_Task_6.ipynb
├── src
└── SS


2. Organise Data
Place the four CSVs into data/.
- Plant_1_Generation_Data.csv

- Plant_1_Weather_Sensor_Data.csv

- Plant_2_Generation_Data.csv

- Plant_2_Weather_Sensor_Data.csv

3. Run Report Notebook

Open the full workflow summary:

jupyter notebook notebooks/00_report.ipynb

4. Run each Task Notebooks Individually from 01 to 08 from top to bottom, step by step

5. Launch the Streamlit Dashboard
streamlit run src/app.py


## Data Sources

- **Plant_1_Generation_Data.csv**
  - Source: Provided by course teaching team (Imperial College London) via Blackboard
  - Access Date: 2025-11-05
  - Description: Generation data for Plant 1, including AC/DC power, daily yield, total yield, and inverter operating conditions.
  - Citation: Imperial College London, CIVE70111 Machine Learning Coursework Dataset.

- **Plant_1_Weather_Sensor_Data.csv**
  - Source: Provided by course teaching team (Imperial College London) via Blackboard
  - Access Date: 2025-11-05
  - Description: Weather sensor data for Plant 1, including irradiation, ambient temperature, and module temperature.
  - Citation: Imperial College London, CIVE70111 Machine Learning Coursework Dataset.

- **Plant_2_Generation_Data.csv**
  - Source: Provided by course teaching team (Imperial College London) via Blackboard
  - Access Date: 2025-11-05
  - Description: Generation data for Plant 2, containing AC/DC power, yields, and operating condition labels.
  - Citation: Imperial College London, CIVE70111 Machine Learning Coursework Dataset.

- **Plant_2_Weather_Sensor_Data.csv**
  - Source: Provided by course teaching team (Imperial College London) via Blackboard
  - Access Date: 2025-11-05
  - Description: Weather sensor data for Plant 2, including irradiation and temperature readings.
  - Citation: Imperial College London, CIVE70111 Machine Learning Coursework Dataset.


## Methodology

[Provide a brief overview of your approach and methods]

1. **Data Collection:**  Four datasets from the course containing inverter-level generation and weather data.

2. **Data Cleaning:** Missing timestamps were identified and filled to ensure a uniform time index. Irradiation signals were cleaned using day–night filtering and interpolation to create a reliable `IRRADIATION_CLEAN` variable. AC and DC power were corrected to remove night-time noise and to enforce physical consistency (e.g., zero output at zero irradiation). Daily and total yields were reconstructed to eliminate counter resets. Temperature readings were smoothed, and a derived feature (`Temp_Delta`) was computed to capture thermal behaviour. Outliers were removed or capped based on statistical and regression-based diagnostics.

3. **Analysis:** 
The analytical work for this project was carried out in a task-based structure, with each task focusing on a different modelling objective. The methods applied in each task are summarised below:

● Task 2 

Data cleaning:

- Identify feature patterns, inverter-level variability, missing values and anomalies in the data set

- Correct Plant 1 DATE_TIME issue (eg. correct 2020/1/6 to 2020/6/1 )

- Separate plant generation dataframe into 22 inveter dataframes.

- Reduce the size of each inveter dataframe by variation in operating condition within the same time interval

- Remove noises in irradiation in weather data (small positive irradiation value during night time and early morning)

- Merge the each inverter dataframe with weather data

- Clean and correct anomalies in power data based on their relationship with irradiaiton as well as their original properties.

Data analysis:

- Use df.describe().T to output statistic summaries of irradiation, temperature and AC/DC output after cleaning

- Identify trends of power and weather data after cleaning

- Construct histograms of data to invetigate their distribution

- Identify outliers in AC and DC using AC vs Irradiaiton and DC vs Irradiation graphs

- Constructed correlation matrix and heatmap to identify candidate predictors for power output


● Task 3 

- Implemented multiple model to predict AC/DC output of each inverter in each plant

- Linear Regression (baseline)

- Polynomial Regression (higher-order baseline)

- Random Forest Regression (Non-linear model)

- Performance of models was assessed and compared using R², RMSE, MAE and residual diagnostics.

- Constructed learning curves for each inverter and each plant to determine the bias - variance of the model



● Task 4 & part of Task 5 – Classification of Inverter Operating Conditions

- Built Logistic regression and to predict inverter operating condition 

- Four versions of each model is included to investigate the effect of outliers and data scaling (with or without outliers/ with or without scaling)

- Handled class imbalance by increasing the weight of errors that are due to wrong prediciton of optimal condition in the loss function

- Select the threshold of each model that gives the best F1 score

- Evaluated model performance using F1 score and confusion matrices

- Analysed feature importance using Drop-Column Importance and feature effect using ALE plots.

- Construct prediction score distribution histogram to visualize the separatibility of the data.


● Task 5 

- Computed Partial Dependence Plots (PDP) for feature effects on AC/DC output.

- Generated SHAP (SHapley Additive exPlanations) values to quantify feature importance.

- Produced PDP-slope coefficient tables to approximate effect of each feature on model output.



● Task 6 

For each inverter:

- Built single layer LSTM model containing 32 units to predict AC/DC one hour after using 24 hrs lookback window

- Constructed Persistence and Moving average models as baseline models for comparison with LSTM model

- Evaluated performance of LSTM model by comparing the RMSE and MAE with the other two baseline models

- Identify the inverter for which LSTM achieves the lowest RMSE and MAE

- For the selected inverter, graphs containing actual power output as well as prediction by all three models on one stable day and one cloudy day are constructed.


● Task 8 – Final Aggregation and Visualisation

- Produced consolidated plots to present final model behaviour, including:

- Predicted vs actual AC/DC power

- Efficiency curves

- Feature-importance dashboards (SHAP + PDP)

- Model comparison summaries

- Created summary scripts integrating outputs from previous tasks to support final reporting.

4. **Visualization:** Few visualization plots are used:

- Use seaborn module to contruct correlation heatmap for one inverter in each plant

- Use Matplotlib.pyplot to construct time-series plots of power and weather data for each plant

- Use partial_dependence and PyALE modules to construct feature effect on model prediction for regression and classification models respectively

- Use shap module to calculate shape values of each feature and construct feature importance graph using matplotlib.pyplot

- Use matplotlib as well as predicted results from trained LSTM, Moving Average and Persistent models to construct graphs in Task 6

- Regression residual plots



## Results

- Random Forest provided the highest accuracy for AC/DC power prediction

- LinearSVC yielded the most reliable classification of inverter operating conditions

- Feature selection based on feature importance can improve model performance

- Feature scaling such as Standardscaler can slightly improve the performance of LinearSVC in identifying the correct operating condition of the inverter 

- Irradiation was consistently the most influential feature in both Random Forest Regression and LinearSVC models

- AC and DC power output are important features in LinearSVC models

- LSTM forecasting captured varaiation in AC and DC power output better than Moving Average and Persistence models.


# See the analysis notebooks in the /notebooks folder (03, 04, 05, 06) for detailed results and visualisations.


## Conclusions

The results of this project show that machine-learning methods can effectively describe and predict how solar PV systems behave. Random Forest was the most accurate model for forecasting AC and DC power because it can capture nonlinear relationships between irradiation, temperature and inverter performance. Logistic Regression worked best for classifying inverter operating conditions, providing stable predictions and clear explanations of how different environmental factors affect the likelihood of Suboptimal operation.

The use of SHAP, PDP and ALE confirmed that the models follow physical trends seen in real PV systems. Irradiation is the strongest driver of power generation, while higher temperature differences between the module and the environment are linked to lower inverter performance. The LSTM forecasting model also showed that time-dependent patterns can be captured, especially on days with rapidly changing weather. Overall, the models give useful insights into plant behaviour and can support performance monitoring and diagnostics.

Limitations:
- The dataset is short (34 days) and may not fully represent seasonal effects.
- Sensor noise and missing timestamps required significant cleaning, which may introduce uncertainty.
- Only a few environmental variables were available (irradiation, ambient temperature, module temperature).
- No inverter-specific metadata (age, position, maintenance history) was included.
- The LSTM model was trained on limited data, restricting its long-term performance.

Future Work:
- Extend the dataset to include more months and seasonal variation.
- Incorporate additional environmental data such as cloud cover, humidity or wind.
- Explore more advanced modelling approaches, including transformer-based time-series models.


## Individual Contributions (Group Projects Only)

<!-- Delete this section if individual project -->

> **Note:** Individual contributions are tracked through Git commit history. This section provides a high-level summary.

**[Peng Yongxi]:**
- [Contribution area 1,  "Data collection and cleaning (notebooks/01-data-loading.ipynb)"]
- [Contribution area 2,  "Statistical analysis (src/analysis.py)"]
- [Contribution area 3,  "Documentation (README.md, methodology)"]

**[Yuanqi Zhao]:**
- []
- []
- []

**[King Hung Michael Wong]:**
- [Contribution area 1]
- [Contribution area 2]
- [Contribution area 3]

**[Haojiang Wang]:**
- [Contribution area 1]
- [Contribution area 2]
- [Contribution area 3]

**[Weiyi Bao]:**
- [Contribution area 1]
- [Contribution area 2]
- [Contribution area 3]

**[Weiyi Bao]:**
- [Contribution area 1]
- [Contribution area 2]
- [Contribution area 3]

**Joint contributions:**
- [Areas where team members collaborated, e.g., "Code review, testing, and integration"]

## Use of AI Tools

> **Important:** Disclose any use of AI tools (ChatGPT, GitHub Copilot, Claude, etc.) as per course policy.

<!-- Choose one of the following statements and delete the others: -->

 AI tools used:**
The following AI tools were used in this project:

- **[ChatGPT]**
  - Purpose: [ Used to debug Python code errors, assist with the establishment of the framework of streamlit, and documentation]
  - Extent: [Approximately 10% of code development]

**Important:** All code generated or suggested by AI tools was thoroughly reviewed, understood, and tested by the project team. We take full responsibility for all submitted work.

## Dependencies

See [requirements.txt](requirements.txt) for a complete list of dependencies.

Key packages:
- Python 3.11
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- PyALE
- shap
- tensorflow / keras
- streamlit


## References

[List any references, citations, or resources used]

1. [Author(s). (Year). Title. Source. URL]
2. [Reference 2]
3. [Reference 3]

[Include academic papers, documentation, Stack Overflow answers, tutorials, etc.]

## Acknowledgments

- Professor Panagiotis Angeloudiss 
- TAs for CIVE 70111 Machine Learning module

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Repository:** [Link to your GitHub repository]
**Last Updated:** [Date]
