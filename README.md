

[![Code](https://img.shields.io/badge/GitHub-View%20Code-blue)](https://github.com/astral-fate/palm-analyzer)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/FatimahEmadEldin/Farm-Performance-Score)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FYOUR_USERNAME%2FYOUR_SPACE_NAME&label=Visitors&countColor=%23697689)

# Palm Farm Analytics: Predictive Models for the Tamra-Thon Hackathon

This project presents a multi-faceted data solution for the **تمرة ثون (Tamra-Thon) Hackathon**, leveraging satellite imagery to address key challenges in the date palm industry. The core of our solution is a pipeline of machine learning models that transform raw satellite data into actionable intelligence, including a novel **"Farm Performance Score,"** real-time anomaly detection, and highly accurate 7-day health forecasts.

## 1. The Problem: A Data Gap in Date Farming

The date palm industry faces significant challenges in optimizing production, managing waste, and improving supply chain efficiency. While rich in potential, the sector often lacks the granular, forward-looking data needed to drive decisions. Our project tackles this fundamental data gap by building predictive models to create the valuable business intelligence that is currently missing.

## 2. Our Solution: A Predictive Intelligence Pipeline

Instead of focusing on a single prediction, we developed a complete machine learning pipeline that transforms raw historical data into a suite of predictive tools. This pipeline consists of three distinct models, each solving a specific problem.

<img width="1919" height="900" alt="Screenshot of the Streamlit Dashboard" src="https://github.com/user-attachments/assets/b388b56a-2a3f-4dcc-a4a8-3cb3f566d3d9" />

## 3. Technology and Data

*   **Tech Stack:** `Gradio`, `Hugging Face Spaces`, `Pandas`, `Scikit-learn`, `LightGBM`, `Optuna`, `Plotly`.
*   **Dataset:** `consolidated_historical_2015_2025.csv`, containing time-series satellite data for multiple palm farms.
    *   **Source:** Sentinel-2 Satellite Imagery, processed via Google Earth Engine.
    *   **Key Indices:** NDVI (vegetation health), NDWI (water content), SAVI (soil-adjusted vegetation index).
    *   **Time Range:** 2018 to 2025.
    *   **Weather Data:** Historical daily weather for Madinah (`temperature`, `precipitation`, `solar_radiation`) was fetched from the Open-Meteo API and integrated into our final forecasting model to provide crucial environmental context.

## 4. The Machine Learning Pipeline

We developed a pipeline of unsupervised and supervised models to transform raw data into actionable insights.

### Part 1: Unsupervised Learning - The "Farm Performance Score"

This model is the foundation of our strategic analysis. It objectively categorizes farms into performance tiers without any pre-existing labels.

*   **Problem Solved:** How can we rank and compare the overall seasonal performance of different farms in a standardized way?
*   **Methodology:** We use **K-Means Clustering** to group farms based on their unique seasonal characteristics.
*   **Engineered Features (What the Model Learns From):** We transformed the raw daily NDVI data into powerful yearly summaries that represent a full growing season:
    *   `peak_ndvi`: The maximum vegetation health a farm achieves, a proxy for peak quality.
    *   `season_duration`: The length of the primary growth period in days, indicating efficiency.
    *   `avg_ndwi_stress`: The average water content during the hottest summer months, a key indicator of irrigation effectiveness.
    *   `seasonal_integral`: The total "greenness" over the entire season (the area under the NDVI curve), serving as a powerful proxy for total biomass and potential yield.
    *   `ndvi_std_dev`: The stability of crop health. A lower value indicates more uniform and consistent growth.
*   **Results:** The model successfully identified three distinct performance profiles. The centroids below show the average characteristics of each tier:




<img width="1776" height="1020" alt="image" src="https://github.com/user-attachments/assets/232e84b3-c790-45b3-b849-82879f821324" />




## Performance Tier Clustering

Farms were clustered into three performance tiers (**High**, **Medium**, **Low**) based on key vegetation metrics like **NDVI** (Normalized Difference Vegetation Index) and **EVI** (Enhanced Vegetation Index). This clustering helps identify the overall health and productivity of each farm.

---

## Forecasting Model Evaluation

A forecasting model was trained for each farm to predict future vegetation index values. The model's accuracy was evaluated using two key metrics:

* **R-squared ($R^2$)**: Indicates the proportion of variance in the dependent variable that is predictable from the independent variable(s). A value closer to **1.0** signifies a better fit.
* **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction. A lower value indicates higher accuracy.

### Individual Farm Forecast Scores

| Farm Name      | R² Score | MAE   |
| :------------- | :------: | :---: |
| alosba         |  0.980   | 0.008 |
| Abdula altazi  |  0.973   | 0.007 |
| abuonoq        |  0.964   | 0.011 |
| albadr         |  0.960   | 0.008 |
| alhabibah      |  0.956   | 0.008 |
| wahaa 2        |  0.948   | 0.010 |
| alia           |  0.942   | 0.007 |
| alia almadinah |  0.941   | 0.006 |
| wahaa nakeel   |  0.937   | 0.009 |
| almarbad       |  0.922   | 0.005 |

---

## Final Performance Report

The table below provides a comprehensive summary, ranking farms by their assigned performance tier and the R² score from their forecasting model. This offers a dual perspective on both **current quality** (Performance Tier) and **future predictability** (Forecast R²).

| Farm Name      | Performance Tier | Mean NDVI | Mean EVI | Std NDVI | Forecast R² | Forecast MAE |
| :------------- | :--------------: | :-------: | :------: | :------: | :---------: | :----------: |
| Abdula altazi  |  Tier 1 (High)   |   0.345   |  0.797   |  0.077   |    0.973    |    0.007     |
| abuonoq        |  Tier 1 (High)   |   0.399   |  0.980   |  0.084   |    0.964    |    0.011     |
| alosba         | Tier 2 (Medium)  |   0.262   |  0.470   |  0.075   |    0.980    |    0.008     |
| albadr         | Tier 2 (Medium)  |   0.248   |  0.608   |  0.062   |    0.960    |    0.008     |
| alhabibah      | Tier 2 (Medium)  |   0.260   |  0.634   |  0.064   |    0.956    |    0.008     |
| wahaa 2        | Tier 2 (Medium)  |   0.288   |  0.783   |  0.064   |    0.948    |    0.010     |
| alia           | Tier 2 (Medium)  |   0.306   |  0.771   |  0.065   |    0.942    |    0.007     |
| wahaa nakeel   | Tier 2 (Medium)  |   0.264   |  0.730   |  0.056   |    0.937    |    0.009     |
| alia almadinah |   Tier 3 (Low)   |   0.218   |  0.566   |  0.046   |    0.941    |    0.006     |
| almarbad       |   Tier 3 (Low)   |   0.149   |  0.374   |  0.028   |    0.922    |    0.005     |


### Part 2: Supervised Learning - 7-Day NDVI Forecasting

This model provides an operational forecast, helping farmers and managers move from being reactive to proactive.

*   **Problem Solved:** What will the health (NDVI) of a specific farm be **7 days from now**?
*   **Methodology:** We use a **LightGBM Regressor**, a powerful gradient-boosting model. To maximize its accuracy, we used the **Optuna** library to perform hyperparameter optimization, automatically finding the best settings for the model.
*   **Feature Engineering (No Data Leakage):** To ensure a realistic forecast, the model is trained to predict a future value (`NDVI_target_7d`) using **only information that would be known today or in the past**. The key features include:
    *   **Lag Features:** The NDVI values from 1, 7, 14, and 30 days ago.
    *   **Rolling Window Features:** The average and standard deviation of NDVI over the past 7, 14, and 30 days.
    *   **Cyclical Time Features:** `sin` and `cos` transformations of the month and day of the year, which help the model understand seasonality (e.g., that December is close to January).
    *   **Weather Data:** Historical daily `temperature`, `precipitation`, and `solar radiation` from Madinah.
*   **Results: Forecasting Accuracy:** Through a rigorous backtesting process, the optimized model demonstrated exceptional accuracy.
    *   **R-squared (R²):** **0.9988**
    *   *This means the model can explain 99.88% of the variation in NDVI a week into the future, making it a highly reliable tool for short-term planning.*

### Part 3: Supervised Learning - Anomaly Detection

This model acts as an early warning system for farmers.

*   **Problem Solved:** Has there been a sudden, unexpected change in a farm's health that requires immediate attention?
*   **Methodology:** We use an **Isolation Forest** model, which is excellent at identifying rare and unusual data points.
*   **How it Works:** The model learns the "normal" patterns of NDVI for each farm. Any data point that deviates significantly from this normal pattern is flagged as an anomaly.
*   **Results:** The model successfully identified 1,129 potential anomalies, allowing us to pinpoint specific dates and farms where health readings were statistically unusual and warranted further investigation.


![anamlydetection-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/de7b033b-249b-4ebb-b66e-b7e966469e4d)


<img width="1918" height="1078" alt="forcasting" src="https://github.com/user-attachments/assets/3f05eb0b-8f53-48e0-8e19-ab3d592dc211" />



## 5. Hackathon Track Applications

Our model pipeline directly feeds into solutions for all three Tamra-Thon tracks:

#### 🥇 Track 1: Smart Agriculture
*   **Proactive Alerts:** The **Anomaly Detection** model provides real-time alerts for sudden drops in NDVI, allowing farmers to investigate potential issues like irrigation failures or pest infestations.
*   **Resource Planning:** The **7-Day Forecast** allows for smarter scheduling of irrigation and fertilization, applying resources when they will be most effective.

#### ♻️ Track 2: Sustainability
*   **Biomass Prediction:** The **`seasonal_integral`** feature from our clustering model serves as a powerful proxy for the total biomass (palm fronds, fibers) a farm will produce. This allows us to predict the **Biomass Waste Tier** for each farm.
*   **Logistics Optimization:** This prediction enables logistics companies to plan efficient collection routes for recycling palm waste, turning a liability into a valuable resource.

#### 🚚 Track 3: Supply Chain
*   **Harvest Forecasting:** The combination of our models provides a comprehensive forecast for the market:
    *   **Harvest Timing:** Predicted using the `peak_day` feature.
    *   **Harvest Quality & Yield:** Predicted using the **"Farm Performance Score"**.
    *   **Short-Term Trajectory:** The **7-Day Forecast** helps predict if a farm's health is trending up or down leading into the harvest period.
*   **Market Transparency:** This data provides unprecedented transparency for buyers, helps farmers negotiate fair prices, and allows logistics operators to optimize their fleet for the predicted yield and timing.


```
