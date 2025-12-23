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
**Best model (clean dataset): Random Forest**

- **Baseline (Linear Regression with leakage):** R² = **0.928**, MAE = **11.71 g/km**, RMSE = **17.79 g/km**
- **Clean dataset (Random Forest):** R² = **0.95329**, MAE = **9.628238 g/km**, RMSE = **14.332936 g/km**

Metrics: `results/metrics/`  
Figures: `results/figures/`

## Feature importance (model interpretability)
We used **permutation importance** (drop in R² after shuffling a feature) to compare feature influence across models.

### Random Forest (best model) — permutation importance (ΔR²)
| feature | importance_mean | importance_std |
|---|---:|---:|
| enginesize_l | 0.997249 | 0.012199 |
| cylinders | 0.162143 | 0.002036 |
| modelyear | 0.161842 | 0.002548 |
| vehicleclass | 0.127299 | 0.001747 |
| make | 0.062831 | 0.002029 |
| transmission | 0.052297 | 0.001827 |
| fuel_type_decoded | 0.011407 | 0.000518 |
| fueltype | 0.011178 | 0.000493 |

### XGBoost — permutation importance (ΔR²)
| feature | importance_mean | importance_std |
|---|---:|---:|
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
|---|---:|---:|
| enginesize_l | 0.469848 | 0.008047 |
| cylinders | 0.202885 | 0.002596 |
| modelyear | 0.107816 | 0.003114 |
| vehicleclass | 0.085452 | 0.001653 |
| transmission | 0.025562 | 0.000718 |
| make | 0.022991 | 0.000562 |
| fueltype | 0.010768 | 0.000518 |
| fuel_type_decoded | 0.010768 | 0.000518 |

**Conclusion (features):** Across models, `enginesize_l` is by far the most influential feature, followed by `cylinders` and `modelyear`. Categorical variables (`vehicleclass`, `make`, `transmission`, `fueltype`) contribute less but consistently refine predictions through vehicle segment differences.

## How to run
### Option A — Google Colab (recommended)
1. Open `notebooks/final.ipynb` in Colab
2. Upload the dataset CSV when prompted (or place it in `data/`)
3. Run all cells

### Option B — Local
```bash
pip install -r requirements.txt
