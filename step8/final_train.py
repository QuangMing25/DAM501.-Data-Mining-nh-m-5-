import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# --- 1. Load Master Data  ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_DATA_PATH = os.path.join(BASE_DIR, "master_data_final_v6.csv")
df_final = pd.read_csv(MASTER_DATA_PATH)

# Change the cluster to the correct category so LightGBM can process it properly
if 'Cluster' in df_final.columns:
    df_final['Cluster'] = df_final['Cluster'].astype('category')

# --- 2. Prepare for the Features (X) and Target (y) exercises ---
# Use log_price_adj as a training target
target_col = 'log_price_adj'

# Columns to be removed (Leakage or irrelevant)
drop_cols = [
    'price', 'log_price', 'price_per_m2', 'log_price_per_m2', 
    'price_index_adjusted', 'log_price_adj',
    'district_name', 'ward_name', 'street_name', 'project_name',
    'district_zone', 'published_at', 'house_direction'
]
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
X_cols = [c for c in df_final.columns if c not in drop_cols and not c.startswith('scaled_')]
X = df_final[X_cols]
y = df_final[target_col]

# --- 3. Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Training ---
print(f"-> Training the final model {len(X)} samples with {len(X_cols)} features...")
final_model = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=1000)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    categorical_feature=['Cluster'] if 'Cluster' in X_cols else 'auto'
)

# --- 5. Metric Evaluation ---
y_pred_log = final_model.predict(X_test)

# Convert the logarithm values ​​back to real monetary values ​
# ​(billion VND) to make the MAE calculation easier to understand
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

r2 = r2_score(y_test, y_pred_log)
mae_real = mean_absolute_error(y_test_real, y_pred_real)
rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100

print("\n" + "="*50)
print("THỐNG KÊ MODEL CUỐI CÙNG (FINAL MODEL)")
print("-" * 50)
print(f"1. R2 Score (trên log_price_adj): {r2:.4f}")
print(f"2. MAE (Sai số trung bình):        {mae_real:.4f} Tỷ VNĐ")
print(f"3. RMSE (Độ lệch chuẩn sai số):   {rmse_real:.4f} Tỷ VNĐ")
print(f"4. Tổng số features sử dụng:       {len(X_cols)}")
print(f"5.  MAPE : {mape:.2f}%")
print("="*50)

# --- 6. Lưu Artifacts cuối cùng ---
FINAL_EXPORT_DIR = os.path.join(BASE_DIR, "../final_production_model")
os.makedirs(FINAL_EXPORT_DIR, exist_ok=True)

with open(os.path.join(FINAL_EXPORT_DIR, "final_lgbm_model.pkl"), "wb") as f:
    pickle.dump(final_model, f)
with open(os.path.join(FINAL_EXPORT_DIR, "final_features.pkl"), "wb") as f:
    pickle.dump(X_cols, f)

print(f"✅ Đã đóng gói Model Final tại: {FINAL_EXPORT_DIR}")
