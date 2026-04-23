import os
import pandas as pd
import numpy as np
import pickle
import glob
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_2025 = os.path.join(BASE_DIR, "data_2025_time_cleaned.csv")
DATA_2026 = os.path.join(BASE_DIR, "2026_clustered_dataset.csv")
EXCEL_DATA_PATH = os.path.join(BASE_DIR, "CPIdata")

"""
Advanced feature: Index-based Adjustment - RPPI (Residential Property Price Index)
 
METHOD:
1. Mapping: Use pub_month and pub_year columns as keys to assign
   macroeconomic indicators to each row of apartment data.
   Indicators:
       - Consumer price indexes (CPI general)
       - Housing & Construction materials price index
       - Gold price indexes
2. Lag Features: Property purchase decisions are often influenced by
   macro conditions from 1-2 months prior, not current values.
 
PIPELINE OVERVIEW:
    - One model trained on 2025 data (with macro features)
    - One unified RPPI call covering all months (Jun 2025 – Mar 2026)
    - RPPI map split after the call → consistent Dec 2025 base for both years
    - 2025 and 2026 prices adjusted independently, then concatenated
"""

def extract_macro_indicators(path):
    if os.path.exists(path):
        files = glob.glob(os.path.join(path, "*.xlsx"))
        df_list = []
        # index_col=0 lets us find rows by their name (e.g., 'GOLD PRICE INDEXES')
        for f in files:
            month_match = re.search(r'T(\d+)_(\d{4})', os.path.basename(f))
            if month_match:
                month = int(month_match.group(1))
                year = int(month_match.group(2))
                try:
                    temp_df = pd.read_excel(f, sheet_name='Dia phuong', usecols="A:B", skiprows=7, header=None)
                    temp_df = temp_df.dropna(how='all') # drop the empty row 9
                    temp_df.index = temp_df.index.astype(str).str.replace('\n', ' ').str.strip()
                    temp_df.columns = temp_df.columns.astype(str).str.strip()

                    hanoi_cpi_general = temp_df.iloc[1, 1]  
                    hanoi_cpi_housing = temp_df.iloc[8, 1] 
                    cpi_gold_index = temp_df.iloc[18, 1]
                    
                    df_list.append({
                        'pub_month': month,
                        'pub_year': year,
                        'macro_cpi_general': hanoi_cpi_general,
                        'macro_cpi_housing': hanoi_cpi_housing,
                        'macro_gold_index': cpi_gold_index
                    })
                except KeyError as e:
                    print(f"Skipping {f}: Could not find expected row/column. Error: {e}")
        final_df = pd.DataFrame(df_list)
        final_df = final_df.sort_values(['pub_year', 'pub_month']).reset_index(drop=True)

        # Calculate lag features
        lag_cols = ['macro_cpi_general', 'macro_cpi_housing', 'macro_gold_index']
        for col in lag_cols:
            final_df[f'{col}_lag1'] = final_df[col].shift(1)
        
        # Backfill for the first month (6-2025)
        return final_df.ffill()

"""
Create a standard apartment (independent of time)
"""

def create_standard_apartment(df, model_features):
    
    standard_row = {}
    for col in model_features:
        if col in df.columns:
            # Check if column is numeric (int or float)
            if pd.api.types.is_numeric_dtype(df[col]):
                # Use median for numbers to avoid outlier bias
                standard_row[col] = df[col].median()
            else:
                # Use mode for categories/strings (most frequent value)
                standard_row[col] = df[col].mode()[0]
    return pd.DataFrame([standard_row])

"""
RPPI constructor
Inject macro indicators to predictions
"""
def get_rppi_from_model(standard_df, lgbm_model, model_features, df_macro, scenarios):

    BASE_MONTH, BASE_YEAR = 12, 2025
    if (BASE_MONTH, BASE_YEAR) not in scenarios:
        raise ValueError(
            f"Base month ({BASE_MONTH}/{BASE_YEAR}) must be included in scenarios "
            f"to compute a consistent RPPI anchor.")
    
    monthly_scenarios = []
    for m, y in scenarios:
        temp_df = standard_df.copy()
        temp_df['pub_month'] = int(m)
        temp_df['pub_year'] = int(y)
        
        macro_match = df_macro[(df_macro['pub_month'] == m) & (df_macro['pub_year'] == y)]
        if macro_match.empty:
            print(f"Warning: No macro data found for {m}/{y} — row will use standard_df values.")
        else:
            for col in [c for c in model_features if c in macro_match.columns]:
                temp_df[col] = macro_match[col].values[0]
        monthly_scenarios.append(temp_df)

    # Combine into one batch for the model
    predict_df = pd.concat(monthly_scenarios, ignore_index=True)
    predict_df = predict_df[model_features]

    # Predict, Inverse logarit if the target was log_price
    if 'Cluster' in predict_df.columns:
        predict_df['Cluster'] = predict_df['Cluster'].astype('category')

    # Model was trained on log_price → predictions are in log scale
    # expm1 reverses log1p transform → gives actual predicted price in VND
    preds = lgbm_model.predict(predict_df)
    log_preds = np.expm1(preds)
    
    # Create index map (set December = 1.0)
    base_idx = scenarios.index((BASE_MONTH, BASE_YEAR))
    base_price = log_preds[base_idx]
    rppi_map = {}
    for i, (m, y) in enumerate(scenarios):
        rppi_map[(m, y)] = log_preds[i] / base_price

    return rppi_map

# ─────────────────────────────────────────────────────────────────
# 1. ORIGINAL CONFIGURATION PARAMETERS (RETAINED FROM STEP 5)
# ─────────────────────────────────────────────────────────────────
LGB_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
N_ESTIMATORS = 800
TEST_SIZE = 0.2
SPLIT_SEED = 42

# ─────────────────────────────────────────────────────────────────
# 2. SEPARATE TRAINING FUNCTIONS (Reusable for Macros)
# ─────────────────────────────────────────────────────────────────
def run_lightgbm_training(df, target_col='log_price', extra_features=None):
    """
    The model training function is based on the original logic of Step 5b.
    extra_features: A list of columns to add (e.g., macro columns).
    """
    # 1. Define basic features (exactly the same as the original code)
    features_to_drop = ['price', 'log_price', 'price_per_m2', 'log_price_per_m2',
                        'district_name', 'ward_name', 'street_name', 'project_name',
                        'district_zone', 'published_at', 'house_direction']
    
    # Remove leak columns and old scaled columns.
    X_cols = [c for c in df.columns if c not in features_to_drop and not c.startswith('scaled_')]
    
    # Ensure that 'Cluster' and 'extra_features' are included in the training list.
    if extra_features:
        for feat in extra_features:
            if feat not in X_cols and feat in df.columns:
                X_cols.append(feat)

    X = df[X_cols].copy()
    y = df[target_col]

    # Processing categorical variables
    cat_features = []
    if 'Cluster' in X.columns:
        X['Cluster'] = X['Cluster'].astype('category')
        cat_features.append('Cluster')
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )

    # Training model
    print(f"-> Training LightGBM with {len(X_cols)} features...")
    model = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=N_ESTIMATORS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        categorical_feature=cat_features if cat_features else 'auto'
    )

    # Quick metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"   R2 on log_price (test set): {r2:.4f}")

    return model, X_cols

# ─────────────────────────────────────────────────────────────────
# 3. IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────

def build_model_v2_with_macro(df_2025_with_macro, save_dir):
    """
    The first (or second) model creation process contains macros..
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Identify the macro columns present in the dataframe.
    macro_cols = [c for c in df_2025_with_macro.columns if c.startswith('macro_')]
    
    model, final_features = run_lightgbm_training(
        df_2025_with_macro, 
        extra_features=macro_cols
    )
    
    # Export artifacts
    model_path = os.path.join(save_dir, "lgbm_model_v2_2025.pkl")
    feature_path = os.path.join(save_dir, "feature_names_v2_2025.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(feature_path, "wb") as f:
        pickle.dump(final_features, f)
        
    print(f"✅ Model and feature names saved to: {save_dir}")
    return model, final_features

# ─────────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────
df_2025 = pd.read_csv(DATA_2025)    # 37 columns
df_macro_full = extract_macro_indicators(EXCEL_DATA_PATH)

# Only merge 2025 macro into 2025 training data — no 2026 leakage
df_macro_2025  = df_macro_full[df_macro_full['pub_year'] == 2025]
df_2025_ready  = df_2025.merge(df_macro_2025, on=['pub_month', 'pub_year'], how='left')

# --- 4b. Train model on 2025 data ---
save_dir = os.path.join(BASE_DIR, "../app_models_v2")
model_2025_v2, features_2025_v2 = build_model_v2_with_macro(
    df_2025_with_macro=df_2025_ready,
    save_dir=save_dir
)
df_2025_ready.to_csv(os.path.join(BASE_DIR, "2025_ready.csv"), index=False, encoding='utf-8-sig')

# --- 4c. Load saved model artifacts ---
MODEL_V2_PATH    = os.path.join(save_dir, "lgbm_model_v2_2025.pkl")
FEATURES_V2_PATH = os.path.join(save_dir, "feature_names_v2_2025.pkl")
 
with open(MODEL_V2_PATH, 'rb') as f:
    model_2025_v2 = pickle.load(f)
with open(FEATURES_V2_PATH, 'rb') as f:
    features_2025_v2 = pickle.load(f)

# --- 4d. Create standard apartment from 2025 data only ---
# IMPORTANT: Must use 2025 distribution to keep the RPPI basket stable.
# Using 2026 data here would shift the benchmark and distort the index.
base_aptm_2025 = create_standard_apartment(df_2025_ready, features_2025_v2)

# --- 4e. Single unified RPPI call covering all months ---
# FIX: One call → one base price → consistent Dec 2025 = 1.0 for both years.
# df_macro_full is passed so Jan–Mar 2026 can access Dec 2025 lag values.
SCENARIOS_ALL = [
    (6,2025),(7,2025),(8,2025),(9,2025),(10,2025),(11,2025),(12,2025),
    (1,2026),(2,2026),(3,2026)
]

rppi_map_total = get_rppi_from_model(
    standard_df=base_aptm_2025,
    lgbm_model=model_2025_v2,
    model_features=features_2025_v2,
    df_macro=df_macro_full,
    scenarios=SCENARIOS_ALL
)
print("RPPI map total:\n", rppi_map_total)

# --- 4f. Split RPPI map after the call (same base guaranteed) ---
rppi_map_2025 = {k: v for k, v in rppi_map_total.items() if k[1] == 2025}
rppi_map_2026 = {k: v for k, v in rppi_map_total.items() if k[1] == 2026}
print("RPPI 2025:", rppi_map_2025)
print("RPPI 2026:", rppi_map_2026)

# --- 4g. Apply RPPI adjustment to 2025 data ---
df_2025_ready['temp_key'] = list(zip(df_2025_ready['pub_month'], df_2025_ready['pub_year']))
df_2025_ready['price_index_adjusted'] = (
    df_2025_ready['price'] / df_2025_ready['temp_key'].map(rppi_map_2025)
)
df_2025_ready['log_price_adj'] = np.log1p(df_2025_ready['price_index_adjusted'])
df_2025_ready.drop(columns=['temp_key'], inplace=True)

unmapped_2025 = df_2025_ready['price_index_adjusted'].isnull().sum()
if unmapped_2025 > 0:
    print(f"{unmapped_2025} rows in 2025 data could not be mapped to RPPI.")

# --- 4h. Load 2026 data and apply RPPI adjustment ---
df_2026 = pd.read_csv(DATA_2026)

# Merge 2026 macro (Jan–Mar 2026 rows only)
df_macro_2026  = df_macro_full[df_macro_full['pub_year'] == 2026]
df_2026_ready  = df_2026.merge(df_macro_2026, on=['pub_month', 'pub_year'], how='left')

df_2026_ready['temp_key'] = list(zip(df_2026_ready['pub_month'], df_2026_ready['pub_year']))
df_2026_ready['price_index_adjusted'] = (
    df_2026_ready['price'] / df_2026_ready['temp_key'].map(rppi_map_2026)
)
df_2026_ready['log_price_adj'] = np.log1p(df_2026_ready['price_index_adjusted'])
df_2026_ready.drop(columns=['temp_key'], inplace=True)
 
unmapped_2026 = df_2026_ready['price_index_adjusted'].isnull().sum()
if unmapped_2026 > 0:
    print(f"{unmapped_2026} rows in 2026 data could not be mapped to RPPI.")

# --- 4i. Merge and export final dataset ---
# Both datasets now share the same target variable (log_price_adj)
# anchored to Dec 2025 = 1.0, making them safe to concatenate for final model training.
df_master_final = pd.concat([df_2025_ready, df_2026_ready], ignore_index=True)
df_master_final.to_csv(
    os.path.join(BASE_DIR, "master_data_final_v6.csv"),
    index=False, encoding='utf-8-sig'
)
print("✅ master_data_final_v6.csv saved.")
