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

## How to run
### Option A — Google Colab (recommended)
1. Open `notebooks/final.ipynb` in Colab
2. Upload the dataset CSV when prompted (or place it in `data/`)
3. Run all cells

### Option B — Local
```bash
pip install -r requirements.txt
