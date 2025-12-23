# Robin Piot & Alexandre Portier. — Vehicle CO₂ Emission Prediction (ML Project)

## Business objective
Predict vehicle **CO₂ emissions (g/km)** from technical characteristics (engine size, cylinders, transmission, fuel type, vehicle class, model year).  
The objective is to support cleaner vehicle design and identify the main factors driving emissions.

## Dataset
Dataset: **MY1995–2023 Fuel Consumption Ratings** (Government of Canada / Kaggle mirror)  
Target: `co2emission_g_km`

### Key issue: data leakage
Fuel consumption variables (`comb_l100km`, `comb_mpg`, `fuelcons*`) are near-direct proxies for CO₂ emissions.  
Including them leads to an unrealistically high R². We detect this using correlation analysis and permutation importance, then remove these variables to make the task realistic.

## Methodology
1. **Data cleaning**
   - Standardized column names
   - Removed missing values and duplicates
2. **Exploratory Data Analysis (EDA)**
   - Target distribution and group comparisons (e.g., fuel type)
   - Correlation heatmap to highlight leakage/proxy variables
3. **Baseline model**
   - Linear Regression with a preprocessing pipeline:
     - `StandardScaler` for numerical features
     - `OneHotEncoder(handle_unknown="ignore")` for categorical features
4. **Obstacle & project evolution**
   - Suspiciously high baseline performance → leakage suspected
   - Permutation importance confirms fuel consumption dominates
   - Fix: drop leakage columns and high-cardinality `model`
5. **Models on clean dataset**
   - Linear Regression (clean baseline)
   - Random Forest (ensemble)
   - SVM + `TruncatedSVD` (dimensionality reduction for sparse one-hot features)
   - XGBoost (advanced model, if available)
6. **Evaluation**
   - Metrics: **MAE**, **RMSE**, **R²**
   - Plots: actual vs predicted, residual distribution

## Results
**Fill with your final numbers (from `results/metrics/metrics_summary.csv`):**
- **Baseline with simple Linear Regression (with leakage):** R² = **0.928**, MAE = **11.71 g/km**, RMSE = **17.79 g/km**
- **Clean dataset (best model : Random Forrest):** R² = **0.95329**, MAE = **9.628238 g/km**, RMSE = **14.332936 g/km**

Metrics: `results/metrics/`  
Figures: `results/figures/`
## Feature importance (model interpretability)

To understand **which variables drive the predictions**, we exported feature influence tables to `results/metrics/`:

- `rf_permutation_importance.json`
- `xgb_permutation_importance.json`
- `svm_permutation_importance.json`
- `linear_regression_coefficients.json`

### Random Forest (best model) — permutation importance (ΔR² after shuffling)
Permutation importance measures how much the model performance drops when a feature is randomly shuffled (higher = more important).

| feature | importance_mean | importance_std |
|---|---|---|
| enginesize_l | 0.997249 | 0.012199 |
| cylinders | 0.162143 | 0.002036 |
| modelyear | 0.161842 | 0.002548 |
| vehicleclass | 0.127299 | 0.001747 |
| make | 0.062831 | 0.002029 |
| transmission | 0.052297 | 0.001827 |
| fuel_type_decoded | 0.011407 | 0.000518 |
| fueltype | 0.011178 | 0.000493 |

**Interpretation:** the model mainly relies on **engine size**, then **cylinders** and **model year**, followed by vehicle category and manufacturer.

### XGBoost — permutation importance (ΔR²)
| feature | importance_mean | importance_std |
|---|---|---|
| enginesize_l | 0.696741 | 0.010231 |
| vehicleclass | 0.126222 | 0.001955 |
| cylinders | 0.104071 | 0.001833 |
| modelyear | 0.099371 | 0.002085 |
| make | 0.054686 | 0.001216 |
| transmission | 0.047920 | 0.001212 |
| fueltype | 0.020213 | 0.000793 |
| fuel_type_decoded | 0.001864 | 0.000113 |

### SVM (+ TruncatedSVD) — permutation importance (ΔR²)
| feature | importance_mean | importance_std |
|---|---|---|
| enginesize_l | 0.469848 | 0.008047 |
| cylinders | 0.202885 | 0.002596 |
| modelyear | 0.107816 | 0.003114 |
| vehicleclass | 0.085452 | 0.001653 |
| transmission | 0.025562 | 0.000718 |
| make | 0.022991 | 0.000562 |
| fueltype | 0.010768 | 0.000518 |
| fuel_type_decoded | 0.010768 | 0.000518 |

### Linear Regression (baseline) — coefficients (true “weights” after one-hot encoding)
A linear model provides explicit coefficients after preprocessing (one-hot encoding). These are not directly comparable to permutation importance but they give a direction (+/-) and magnitude.

Top positive coefficients:
| feature | coefficient | abs_coefficient |
|---|---|---|
| cat__make_FERRARI | 124.253 | 124.253 |
| cat__make_BUGATTI | 123.261 | 123.261 |
| cat__make_Bugatti | 118.000 | 118.000 |
| cat__make_LAMBORGHINI | 74.278 | 74.278 |
| cat__vehicleclass_Van: Passenger | 67.413 | 67.413 |
| cat__make_MASERATI | 67.064 | 67.064 |
| cat__make_Lamborghini | 64.738 | 64.738 |
| cat__make_BENTLEY | 62.592 | 62.592 |

Top negative coefficients:
| feature | coefficient | abs_coefficient |
|---|---|---|
| cat__make_SMART | -49.560 | 49.560 |
| cat__make_GEO | -48.803 | 48.803 |
| cat__transmission_AV6 | -38.341 | 38.341 |
| cat__make_Mazda | -35.632 | 35.632 |
| cat__make_Toyota | -33.882 | 33.882 |
| cat__vehicleclass_SUBCOMPACT | -33.386 | 33.386 |
| cat__vehicleclass_MINICOMPACT | -32.944 | 32.944 |
| cat__vehicleclass_COMPACT | -32.394 | 32.394 |


## How to run
### Option A — Google Colab (recommended)
1. Open `notebooks/final.ipynb` in Colab
2. Upload the dataset CSV when prompted (or place it in `data/`)
3. Run all cells

### Option B — Local
```bash
pip install -r requirements.txt
