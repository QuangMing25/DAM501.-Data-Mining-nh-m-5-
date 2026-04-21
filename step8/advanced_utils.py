import os
import pandas as pd
import numpy as np
import pickle
import glob
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_2025 = os.path.join(BASE_DIR, "data_2025_time_cleaned.csv")
DATA_2026 = os.path.join(BASE_DIR, "2026_clustered_dataset.csv")
LGBM_PATH = os.path.join(BASE_DIR, "../app_models/lgbm_model.pkl")
LGBM_FEATURE_PATH = os.path.join(BASE_DIR, "../app_models/feature_names.pkl")
EXCEL_DATA_PATH = os.path.join(BASE_DIR, "CPIdata")

"""
Advanced feature: Index-based Adjustment - RPPI (Residential Property Price Index)
Inflation/CPI: This represents the devaluation of the currency and the increase in the cost of construction materials.
METHOD:
1. Mapping: Use the pub_month and pub_year columns as "keys" to assign 
corresponding macroeconomic indicators to each row of apartment data.
Choose indicators: 
    -   Consumer price indexes
    -   Housing & Construction materials
    -   Gold price indexes
2. Lag Features: The decision to buy a house is often not influenced by today's interest rate, 
but by the interest rate from 1-2 months ago.

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
        return final_df.bfill()

"""
Create a standard apartment (independent of time)
"""

def create_standard_aparment(df, model_features):
    # cols_to_use = [col for col in df.columns if col not in ['price', 'price_per_m2']]
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
def get_rppi_from_model(standard_df, lgbm_model, model_features, df_macro):
 
    scenarios = [
        (6, 2025), (7, 2025), (8, 2025), (9, 2025), (10, 2025), (11, 2025), (12, 2025),
        (1, 2026), (2, 2026), (3, 2026)
    ]
    monthly_scenarios = []
    for m, y in scenarios:
        temp_df = standard_df.copy()
        temp_df['pub_month'] = int(m)
        temp_df['pub_year'] = int(y)
        
        macro_match = df_macro[(df_macro['pub_month'] == m) & (df_macro['pub_year'] == y)]
        if not macro_match.empty:
            for col in [c for c in model_features if c in macro_match.columns]:
                temp_df[col] = macro_match[col].values[0]
        monthly_scenarios.append(temp_df)

    # Combine into one batch for the model
    predict_df = pd.concat(monthly_scenarios, ignore_index=True)
    predict_df = predict_df[model_features]

    # Predict, Inverse logarit if the target was log_price
    if 'Cluster' in predict_df.columns:
        predict_df['Cluster'] = predict_df['Cluster'].astype('category')
    preds = lgbm_model.predict(predict_df)
    log_preds = np.expm1(preds) # if use log_price
    
    # Create index map (set December = 1.0)
    dec_2025_price = log_preds[6]
    rppi_map = {}
    for i, (m, y) in enumerate(scenarios):
        rppi_map[(m, y)] = log_preds[i] / dec_2025_price
    return rppi_map

import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────────────────────────────
# 1. CẤU HÌNH THÔNG SỐ GỐC (GIỮ NGUYÊN TỪ STEP 5)
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
# 2. HÀM HUẤN LUYỆN TÁCH BIỆT (Tái sử dụng được cho Macro)
# ─────────────────────────────────────────────────────────────────
def run_lightgbm_training(df, target_col='log_price', extra_features=None):
    """
    Hàm huấn luyện mô hình dựa trên logic gốc của Step 5b.
    extra_features: Danh sách các cột muốn thêm vào (ví dụ: các cột macro).
    """
    # 1. Định nghĩa features cơ bản (giống hệt code gốc)
    features_to_drop = ['price', 'log_price', 'price_per_m2', 'log_price_per_m2',
                        'district_name', 'ward_name', 'street_name', 'project_name',
                        'district_zone', 'published_at', 'house_direction']
    
    # Loại bỏ các cột leak và các cột scaled cũ
    X_cols = [c for c in df.columns if c not in features_to_drop and not c.startswith('scaled_')]
    
    # Đảm bảo 'Cluster' và các 'extra_features' nằm trong danh sách huấn luyện
    if extra_features:
        for feat in extra_features:
            if feat not in X_cols and feat in df.columns:
                X_cols.append(feat)

    X = df[X_cols].copy()
    y = df[target_col]

    # Xử lý biến phân loại (Categorical)
    cat_features = []
    if 'Cluster' in X.columns:
        X['Cluster'] = X['Cluster'].astype('category')
        cat_features.append('Cluster')
    
    # Tách tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )

    # Huấn luyện
    print(f"-> Đang huấn luyện LightGBM với {len(X_cols)} features...")
    model = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=N_ESTIMATORS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        categorical_feature=cat_features if cat_features else 'auto'
    )

    # Đánh giá nhanh
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"   Kết quả R2 (trên log_price): {r2:.4f}")

    return model, X_cols

# ─────────────────────────────────────────────────────────────────
# 3. QUY TRÌNH THỰC THI (Áp dụng cho logic của bạn)
# ─────────────────────────────────────────────────────────────────

def build_model_v2_with_macro(df_2025_with_macro, save_dir):
    """
    Quy trình tạo Model lần 1 (hoặc lần 2) có chứa Macro.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Xác định các cột macro đang có trong dataframe
    macro_cols = [c for c in df_2025_with_macro.columns if c.startswith('macro_')]
    
    # Chạy huấn luyện
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
        
    print(f"✅ Đã lưu Model và Feature Names vào: {save_dir}")
    return model, final_features

df_2025 = pd.read_csv(DATA_2025)
df_macro_full = extract_macro_indicators(EXCEL_DATA_PATH)
df_macro_2025 = df_macro_full.iloc[:-3]
df_2025_ready = df_2025.merge(df_macro_2025, on=['pub_month', 'pub_year'], how='left')
df_2025_ready = df_2025_ready.bfill()
model_2025_v2, features_2025_v2 = build_model_v2_with_macro(
    df_2025_with_macro = df_2025_ready, 
    save_dir = os.path.join(BASE_DIR, "../app_models_v2")
)
df_2025_ready.to_csv(os.path.join(BASE_DIR, "2025_ready.csv"), index=False, encoding='utf-8-sig')

MODEL_V2_PATH = os.path.join(BASE_DIR, "../app_models_v2/lgbm_model_v2_2025.pkl")
FEATURES_V2_PATH = os.path.join(BASE_DIR, "../app_models_v2/feature_names_v2_2025.pkl")

with open(MODEL_V2_PATH, 'rb') as f:
    model_2025_v2 = pickle.load(f)
with open(FEATURES_V2_PATH, 'rb') as f:
    features_2025_v2 = pickle.load(f)

# I hack this part (another file to avoid too long & duplicated code)
df_2025_final = pd.read_csv(os.path.join(BASE_DIR, "df_2025_FINAL_ADJUSTED.csv"))
base_aptm_2025 = create_standard_aparment(df_2025_final, features_2025_v2)
df_2026 = pd.read_csv(DATA_2026)

rppi_map_total = get_rppi_from_model(
    base_aptm_2025, 
    model_2025_v2, 
    features_2025_v2, 
    df_macro_full
)
print("RPPI map total = \n", rppi_map_total)
rppi_map_2026 = {k: v for k, v in rppi_map_total.items() if k[1] == 2026}
print("RPPI Map 2026 chuẩn (đã tính từ mốc T12/2025):", rppi_map_2026)
# Bước C: Áp dụng điều chỉnh cho 2026
df_2026_ready = df_2026.merge(df_macro_full, on=['pub_month', 'pub_year'], how='left')
df_2026_ready['temp_key'] = list(zip(df_2026_ready['pub_month'], df_2026_ready['pub_year']))
df_2026_ready['price_index_adjusted'] = df_2026_ready['price'] / df_2026_ready['temp_key'].map(rppi_map_2026)
df_2026_ready['log_price_adj'] = np.log1p(df_2026_ready['price_index_adjusted'])
df_2026_ready.drop(columns=['temp_key'], inplace=True)

# Bước D: Hợp nhất (Concatenate) - Phá vỡ hoàn toàn vòng lặp
df_2025_final = pd.read_csv(os.path.join(BASE_DIR, "df_2025_FINAL_ADJUSTED.csv"))

# Đảm bảo các cột trùng khớp hoàn toàn
df_master_final = pd.concat([df_2025_final, df_2026_ready], ignore_index=True)

# Xuất file cuối cùng
df_master_final.to_csv(os.path.join(BASE_DIR, "master_data_final_v5.csv"), index=False, encoding='utf-8-sig')
