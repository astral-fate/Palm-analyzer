 
### **Updated `README.md` Content**

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

| Tier             | peak_ndvi | peak_day (Day of Year) | season_duration (Days) | avg_ndwi_stress | seasonal_integral (Total Vigor) | ndvi_std_dev (Volatility) |
| ---------------- | --------- | ---------------------- | ---------------------- | --------------- | ------------------------------- | ------------------------- |
| **Premium Tier** | **0.574** | 324                    | **285**                | **-0.422**      | **18.65**                       | 0.193                     |
| Standard Tier    | 0.072     | 46                     | 35                     | -0.197          | 2.46                            | **0.015**                 |
| Economy Tier     | 0.066     | 352                    | 205                    | -0.217          | 0.32                            | 0.018                     |

*The "Premium Tier" is clearly identifiable by its significantly higher peak health and overall vigor.*

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

## Citation

If this model helped your research, please cite our work:

```bibtex
@misc{tamra-thon-palm-analytics-2025,
    author = {[Fatimah Emad Eldin, Al-Jawara Al-Harbi]},
    title  = {{Palm Farm Analytics: Predictive Models for Smart Agriculture, Sustainability, and Supply Chain}},
    month  = sep,
    year   = {2025},
    url    = {https://huggingface.co/spaces/FatimahEmadEldin/Farm-Performance-Score}
}
```
