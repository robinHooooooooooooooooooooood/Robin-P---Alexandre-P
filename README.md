# Robin-P---Alexandre-P
# Vehicle CO₂ Emission Prediction (Machine Learning Project)

## Business objective
Predict vehicle **CO₂ emissions (g/km)** from technical characteristics (engine, cylinders, transmission, fuel type, vehicle class, model year).
The objective is to support cleaner vehicle design and understand the main factors influencing emissions.

## Dataset
Dataset: **MY1995–2023 Fuel Consumption Ratings** (Government of Canada / Kaggle mirror)  
Target: `co2emission_g_km`

**Key issue: data leakage**
Fuel consumption variables (`comb_l100km`, `comb_mpg`, `fuelcons*`) are near-direct proxies for CO₂ emissions.
This creates an unrealistically high R². We detect it using correlation analysis and permutation importance, then remove these variables to make the task realistic.

## Methodology
- Data cleaning: missing values, duplicates, standardized column names
- EDA: distribution plots + correlation heatmap
- Baseline: Linear Regression with a preprocessing pipeline (scaling + one-hot encoding)
- Obstacle: leakage detected (R² too high)
- Fix: drop leakage columns + drop high-cardinality `model`
- Advanced models (clean dataset): Random Forest, SVM (+ TruncatedSVD), XGBoost
- Metrics: MAE, RMSE, R²

## Results
Fill this section with your final numbers:
- Baseline (leaky) R² / MAE / RMSE: **...**
- Clean dataset best model R² / MAE / RMSE: **...**

Metrics table: `results/metrics/`  
Figures: `results/figures/`

## How to run
Open `notebooks/final.ipynb` and run all cells.

## References
- Breiman (2001) Random Forests
- Chen & Guestrin (2016) XGBoost
