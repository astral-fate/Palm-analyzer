

[![Code](https://img.shields.io/badge/GitHub-View%20Code-blue)](https://github.com/astral-fate/palm-analyzer)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/FatimahEmadEldin/Farm-Performance-Score?logs=build)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fhuggingface.co%2Fspaces%2FYOUR_USERNAME%2FYOUR_SPACE_NAME&label=Visitors&countColor=%23697689)

# Palm Farm Analytics: Predictive Models for the Tamra-Thon Hackathon

This project presents a multi-faceted data solution for the **تمرة ثون (Tamra-Thon) Hackathon**, leveraging satellite imagery to address key challenges in the date palm industry. The core of our solution is a novel **"Farm Performance Score"** derived from historical satellite data, which serves as a powerful predictive indicator for applications across all three hackathon tracks: Smart Agriculture, Sustainability, and Supply Chain.

## 1. The Problem: A Data Gap in Date Farming

The date palm industry faces significant challenges, including optimizing production, managing waste, and improving supply chain efficiency. While rich in potential, the sector often lacks granular, real-time data to drive decisions. The Tamra-Thon hackathon highlights these issues:

*   **Smart Agriculture:** Farmers need tools to monitor crop health proactively, predict issues, and optimize resource usage.
*   **Sustainability:** The industry struggles with the under-utilization of biomass waste (fronds, fibers), lacking the data to build efficient recycling logistics.
*   **Supply Chain:** A weak link between producers and consumers, coupled with a lack of price transparency, makes it difficult to plan logistics and fairly value harvests.

Our project tackles the fundamental data gap at the heart of these problems. We start with time-series satellite data and build predictive models to create the valuable business intelligence that is currently missing.

## 2. Our Solution: A Unified "Farm Performance Score"

Instead of focusing on a single prediction, we developed a foundational model that creates a **"Farm Performance Score"** for each farm, for each year. This is achieved through a two-step machine learning process:

1.  **Feature Engineering:** We process raw time-series satellite data (NDVI, NDWI, SAVI) to extract meaningful agricultural indicators for each growing season.
2.  **Unsupervised Clustering:** We use a K-Means clustering model to group these indicators into distinct performance tiers: **Premium, Standard, and Economy**.

This score becomes a powerful proxy for yield, quality, and biomass, enabling a suite of predictive applications.

## 3. Technology and Data

*   **Tech Stack:** `Gradio`, `Hugging Face Spaces`, `Pandas`, `Scikit-learn`, `LightGBM`, `Plotly`.
*   **Dataset:** The `consolidated_historical_2015_2025.csv` dataset, containing time-series satellite data for 10+ palm farms.
    *   **Source:** Sentinel-2 Satellite Imagery.
    *   **Key Indices:** NDVI (vegetation health), NDWI (water content), SAVI (soil-adjusted vegetation index).
    *   **Time Range:** 2018 to 2025.

<img width="1919" height="900" alt="Screenshot 2025-09-05 075335" src="https://github.com/user-attachments/assets/b388b56a-2a3f-4dcc-a4a8-3cb3f566d3d9" />



## 4. Models and Methodology

We developed a pipeline of unsupervised and supervised models to transform raw data into actionable insights.

### Part 1: Unsupervised Learning - Farm Performance Clustering

This model is the core of our solution. It analyzes the unique characteristics of each farm's growing season to assign a performance score.

*   **Engineered Features:**
    *   `peak_ndvi`: The maximum vegetation health achieved.
    *   `season_duration`: The length of the primary growth period.
    *   `avg_ndwi_stress`: Average water stress during the hot summer months.
    *   `seasonal_integral`: The total "greenness" or biomass produced over the season.
    *   `ndvi_std_dev`: The stability and uniformity of crop health.
*   **Model:** K-Means Clustering (`n_clusters=3`).

#### Results: Cluster Analysis

The model successfully identified three distinct performance profiles. The centroids below show the average characteristics of each tier:

| Tier             | peak_ndvi | peak_day (Day of Year) | season_duration (Days) | avg_ndwi_stress | seasonal_integral (Total Vigor) | ndvi_std_dev (Volatility) |
| ---------------- | --------- | ---------------------- | ---------------------- | --------------- | ------------------------------- | ------------------------- |
| **Premium Tier** | **0.574** | 324                    | **285**                | **-0.422**      | **18.65**                       | 0.193                     |
| Standard Tier    | 0.072     | 46                     | 35                     | -0.197          | 2.46                            | **0.015**                 |
| Economy Tier     | 0.066     | 352                    | 205                    | -0.217          | 0.32                            | 0.018                     |

*The "Premium Tier" is clearly identifiable by its significantly higher peak health and overall vigor.*

### Part 2: Supervised Learning - Anomaly Detection

This model provides real-time alerts for farmers.

*   **Model:** Isolation Forest.
*   **Goal:** To identify individual data points that represent a sudden, unexpected change in a farm's vegetation health.

#### Results: Anomaly Identification

The model successfully identified 1,129 potential anomalies in the dataset. The top 10 most significant anomalies are listed below, indicating moments of unusually high (or low) NDVI readings that warrant investigation.

| timestamp                 | farm_name    | NDVI     | anomaly_score |
| ------------------------- | ------------ | -------- | ------------- |
| 2022-08-29 08:03:09.912   | wahaa nakeel | 0.634719 | -0.244034     |
| 2022-08-04 08:03:20.172   | abuonoq      | 0.640611 | -0.238517     |
| 2022-08-29 08:03:09.912   | alosba       | 0.616073 | -0.233538     |
| ...                       | ...          | ...      | ...           |

### Part 3: Supervised Learning - NDVI Forecasting

This model predicts future NDVI values to help anticipate growth trends.

*   **Model:** LightGBM Regressor.
*   **Features:** Time-based features (day of year, week, month), lag features (NDVI from previous days), and rolling averages.

#### Results: Forecasting Accuracy

The model achieved a very high level of accuracy in predicting NDVI values for the next 6 months.



*   **Root Mean Squared Error (RMSE):** **0.0018**

*This extremely low error indicates that the model can reliably forecast the health trajectory of the farms.*

## 5. Hackathon Track Applications

Our models directly feed into solutions for all three Tamra-Thon tracks:


<img width="1233" height="547" alt="download" src="https://github.com/user-attachments/assets/096774bc-91a2-4b35-9199-f53adba2b12c" />

<img width="1242" height="547" alt="download" src="https://github.com/user-attachments/assets/f01d118d-c69d-49d6-8559-4639fe20f75b" />


#### 🥇 Track 1: Smart Agriculture
Our **Anomaly Detection** model provides proactive alerts, allowing farmers to investigate potential issues like irrigation failures or pest infestations before they impact the entire crop. This moves from reactive to predictive farm management.

#### ♻️ Track 2: Sustainability
The **`seasonal_integral`** (Total Vigor) feature from our clustering model serves as a powerful proxy for the total biomass produced. This allows us to predict the **Biomass Waste Tier** for each farm, enabling logistics companies to plan efficient collection routes for recycling palm fronds and fibers.

#### 🚚 Track 3: Supply Chain
The combination of our models provides a comprehensive forecast for the market:
*   **Harvest Timing:** Predicted using the `peak_day` feature.
*   **Harvest Quality:** Predicted using the **"Farm Performance Score"**.

This data provides unprecedented transparency for buyers, helps farmers negotiate fair prices, and allows logistics operators to optimize their fleet.

## Citation

If this model helped your research, please cite our work:

```bibtex
@misc{tamra-thon-palm-analytics-2025,
    author = {[Fatimah Emad Eldin, Al-Jawara Al-Harbi]},
    title  = {{Palm Farm Analytics: Predictive Models for Smart Agriculture, Sustainability, and Supply Chain}},
    month  = sep,
    year   = {2025},
    url    = {https://huggingface.co/spaces/FatimahEmadEldin/Farm-Performance-Score?logs=build}
}
