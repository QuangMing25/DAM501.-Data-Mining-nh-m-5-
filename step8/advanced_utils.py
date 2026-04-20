import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "clustered_dataset.csv")
LGBM_PATH = os.path.join(BASE_DIR, "lgbm_model_2.pkl")
LGBM_FEATURE_PATH = os.path.join(BASE_DIR, "model_features_2.pkl")

df = pd.read_csv(DATA_PATH)

"""
Index-based Adjustment - RPPI (Residential Property Price Index)
1. Calculate 
"""

def create_standard_aparment(df, model_features):
    # cols_to_use = [col for col in df.columns if col not in ['price', 'price_per_m2']]
    standard_row = {}
    for col in model_features:
    
        # Check if column is numeric (int or float)
        if pd.api.types.is_numeric_dtype(df[col]):
            # Use median for numbers to avoid outlier bias
            standard_row[col] = df[col].median()
        else:
            # Use mode for categories/strings (most frequent value)
            standard_row[col] = df[col].mode()[0]
    return pd.DataFrame([standard_row])

"""
Generate monthly predictions
"""
def get_rppi_from_model(standard_df, lgbm_model, model_features):
    months = [6, 7, 8, 9, 10, 11, 12]
    monthly_scenarios = []

    for m in months:
        temp_df = standard_df.copy()
        temp_df['pub_month'] = m
        if 'pub_year' in temp_df.columns:
            temp_df['pub_year'] = 2025.0
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
    dec_price = log_preds[-1]
    return {month: (p / dec_price) for month, p in zip(months, log_preds)}

"""
Inflation/CPI: This represents the devaluation of the currency and the increase in the cost of construction materials.
METHOD:
1. Mapping: Use the pub_month and pub_year columns as "keys" to assign 
corresponding macroeconomic indicators to each row of apartment data.
2. Lag Features: The decision to buy a house is often not influenced by today's interest rate, 
but by the interest rate from 1-2 months ago.
3. Interaction Features: 
3.1 Real Interest Rate: This index reflects the actual borrowing costs that buyers have to bear.
3.2 Affordability Index: Combine the adjusted price_index with the loan interest rate to see whether, 
at a given time, the apartment becomes "more expensive" or "cheaper" for the borrower.
"""
import glob
import re

EXCEL_DATA_PATH = os.path.join(BASE_DIR, "CPIdata")

def extract_macro_indicators(path):
    if os.path.exists(path):
        files = glob.glob(os.path.join(path, "T*.-CPI-web-Eng.xlsx"))
        df_list = []
        # index_col=0 lets us find rows by their name (e.g., 'GOLD PRICE INDEXES')
        for f in files:
            month_match = re.search(r'T(\d+)', os.path.basename(f))
            month = int(month_match.group(1)) if month_match else None
            temp_df = pd.read_excel(f, sheet_name='Dia phuong', usecols="A:B", skiprows=7, header=None)
            temp_df = temp_df.dropna(how='all') # drop the empty row 9
            temp_df.index = temp_df.index.astype(str).str.replace('\n', ' ').str.strip()
            temp_df.columns = temp_df.columns.astype(str).str.strip()

            try:
                hanoi_cpi_general = temp_df.iloc[1, 1]  
                hanoi_cpi_housing = temp_df.iloc[8, 1] 
                cpi_gold_index = temp_df.iloc[18, 1]
                
                if not temp_df.empty:
                    df_list.append({
                        'pub_month': month,
                        'macro_cpi_general': hanoi_cpi_general,
                        'macro_cpi_housing': hanoi_cpi_housing,
                        'macro_gold_index': cpi_gold_index
                    })
            except KeyError as e:
                print(f"Skipping {f}: Could not find expected row/column. Error: {e}")
        
        if not df_list:
            print("No data found! Check if file names match and folder path is correct.")
            return pd.DataFrame(columns=['pub_month', 'macro_cpi_general', 'macro_cpi_housing', 'macro_gold_index'])
        final_df = pd.DataFrame(df_list)
        final_df = final_df.sort_values('pub_month').reset_index(drop=True)
    else:
        raise ValueError("Path doesn't exist")
    return final_df

"""
Test
"""
with open(LGBM_PATH, 'rb') as f:
    model = pickle.load(f)
with open(LGBM_FEATURE_PATH, 'rb') as f:
    model_features = pickle.load(f)

# # Generate the 'Typical' apartment row
base_aptm = create_standard_aparment(df=df, model_features=model_features)

# Get the RPPI factors from existing LightGBM model
rppi_map = get_rppi_from_model(base_aptm, model, model_features)

# Calculate the adjusted price
df['price_index_adjusted'] = df['price'] / df['pub_month'].map(rppi_map)
print("RPPI Multipliers:", rppi_map)
if not os.path.exists(os.path.join(BASE_DIR, "price_adj_dataset_updated_log.csv")):
    df.to_csv(os.path.join(BASE_DIR, "price_adj_dataset_updated_log.csv"), index=False)
else:
    print("No need to generate dataset again!")

# Fuse excel CPI data
df_macro = extract_macro_indicators(EXCEL_DATA_PATH)
if not os.path.exists(os.path.join(BASE_DIR, "pub_month_cpi.csv")):
    df_macro.to_csv(os.path.join(BASE_DIR, "pub_month_cpi.csv"), index=False)
else:
    print("No need to generate CPI again!")

df_macro_lag = df_macro.copy()
lag_cols = ['macro_cpi_general', 'macro_cpi_housing', 'macro_gold_index']
for col in lag_cols:
    df_macro_lag[f'{col}_lag1'] = df_macro_lag[col].shift(1)

# Merge into the original dataset
df_final = df.merge(df_macro_lag, on='pub_month', how='left')
df_final = df_final.bfill()  # backfill to handle blank (NaN) values ​​for June due to lack of May data (lag)
print("--- Danh sách các cột vĩ mô mới đã thêm ---")
new_macro_cols = [c for c in df_final.columns if c.startswith('macro_')]
print(new_macro_cols)

if not os.path.exists(os.path.join(BASE_DIR, "price_adj_log_cpi.csv")):
    df_final.to_csv(os.path.join(BASE_DIR, "price_adj_log_cpi.csv"), index=False)
else:
    print("No need to generate final data again!")

"""
Run lgbm again
"""
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# SPLIT_SEED = 42
# TEST_SIZE  = 0.2
# lgb_params = {
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': 'rmse',
#     'learning_rate': 0.05,
#     'num_leaves': 63,
#     'max_depth': -1,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': -1,
#     'random_state': 42
# }
# N_ESTIMATORS = 800
# def train_and_evaluatee(X, y, model_name, cat_features=None):
#     """Huấn luyện LightGBM và trả về dict metrics + predictions."""
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED
#     )

#     model = lgb.LGBMRegressor(**lgb_params, n_estimators=N_ESTIMATORS)

#     fit_params = {
#         'eval_set': [(X_test, y_test)],
#         'eval_metric': 'rmse',
#     }
#     if cat_features:
#         fit_params['categorical_feature'] = cat_features

#     model.fit(X_train, y_train, **fit_params)

#     # Dự đoán
#     y_pred_log = model.predict(X_test)
#     y_pred_real = np.expm1(y_pred_log)
#     y_test_real = np.expm1(y_test)

#     # Metrics
#     r2   = r2_score(y_test_real, y_pred_real)
#     rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
#     mae  = mean_absolute_error(y_test_real, y_pred_real)
#     mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100

#     results = {
#         'name': model_name,
#         'X_train': X_train,
#         'model': model,
#         'R2': r2,
#         'RMSE_ty': rmse / 1e9,
#         'MAE_ty': mae / 1e9,
#         'MAPE': mape,
#         'y_test_real': y_test_real,
#         'y_pred_real': y_pred_real,
#         'X_cols': list(X.columns),
#     }

#     print(f"\n{'─'*50}")
#     print(f"  {model_name}")
#     print(f"{'─'*50}")
#     print(f"  R²   : {r2:.4f}")
#     print(f"  RMSE : {rmse/1e9:.4f} Tỷ VND")
#     print(f"  MAE  : {mae/1e9:.4f} Tỷ VND")
#     print(f"  MAPE : {mape:.2f}%")

#     return results

# features_to_drop = [
#     # Direct price leaks / targets
#     'price', 
#     'log_price', 
#     'price_per_m2', 
#     'log_price_per_m2',
#     'price_index_adjusted',        # Very important to drop
    
#     # Identification / non-feature columns
#     'district_name', 
#     'ward_name', 
#     'street_name', 
#     'project_name',
#     'published_at',
    
#     # Redundant after adjustment
#     'pub_month',                   # Safe to drop now
#     'pub_year'
# ]
# X_cols_full = [c for c in df_final.columns if c not in features_to_drop]
# wanted_extra = ['Cluster', 'macro_cpi_general', 'macro_cpi_housing', 
#                 'macro_gold_index', 'macro_cpi_general_lag1', 
#                 'macro_cpi_housing_lag1', 'macro_gold_index_lag1']

# X_cols_full = [c for c in X_cols_full if c in wanted_extra or c not in features_to_drop]
# categorical_cols = ['district_zone', 'Cluster']

# X_full = df_final[X_cols_full].copy()
# for col in categorical_cols:
#     if col in X_full.columns:
#         X_full[col] = X_full[col].astype('category')
# y = np.log(df_final['price_index_adjusted'])
# result_final = train_and_evaluatee(X_full, y, "Model with Macro & Adjusted Price", cat_features=['Cluster'])

# print(f"R2 after adjustment at step 8 = {result_final['R2']:.4f}")
# model_best = result_final['model']
# importance_gain = model_best.booster_.feature_importance(importance_type='gain')
# importance_split = model_best.booster_.feature_importance(importance_type='split')

# df_imp = pd.DataFrame({
#     'Feature': result_final['X_cols'],
#     'Split': importance_split,
#     'Gain': importance_gain
# }).sort_values(by='Gain', ascending=False).head(15)

# print("\n[TOP 15 FEATURE IMPORTANCE - MÔ HÌNH TÍCH HỢP K-MEANS]")
# print(df_imp[['Feature', 'Gain']].to_string(index=False))

# """
# Intermediate statistics
# 1. Compare values before and after adjustment
# """
# summary_district = df.groupby('district_name').agg({
#     'price': 'mean',
#     'price_index_adjusted': 'mean'
# }).reset_index()

# # Calculate the skew ratio (before and after adjustment)
# summary_district['diff_percentage'] = (
#     (summary_district['price_index_adjusted'] - summary_district['price']) / summary_district['price'] * 100
# )

# # Ascending sort
# summary_district = summary_district.sort_values(by='price_index_adjusted', ascending=False)

# print("\n--- TABLE COMPARING VALUES BEFORE AND AFTER ADJUSTMENT (VNĐ) ---")
# print(summary_district.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# """
# 2. Plot
# """
# # Use median as raw method to compare with hedonic RPPI adjustment
# # Normalize to december = 1.0 to match RPPI metric
# monthly_median = df.groupby('pub_month')['price_per_m2'].median()
# actual_index = monthly_median / monthly_median.loc[12]
# rppi_series = pd.Series(rppi_map)
# plt.figure(figsize=(10, 6))
# sns.set_style("whitegrid")

# plt.plot(actual_index.index, actual_index.values, marker='o', linestyle='--', color='red', label='Chỉ số Median thực tế (Bị nhiễu)')
# plt.plot(rppi_series.index, rppi_series.values, marker='s', linestyle='-', color='blue', linewidth=2, label='Hedonic RPPI (LightGBM - Chuẩn hóa)')

# # Trang trí biểu đồ
# plt.title('So sánh Chỉ số Giá Bất động sản: Thực tế vs. Mô hình Hedonic', fontsize=14, fontweight='bold')
# plt.xlabel('Tháng (năm 2025)', fontsize=12)
# plt.ylabel('Giá trị Index (Tháng 12 = 1.0)', fontsize=12)
# plt.xticks(range(6, 13))
# plt.legend()

# # Add explaination
# plt.annotate('Sự chênh lệch cho thấy giá tăng\ndo chất lượng căn hộ thay đổi', 
#              xy=(8, actual_index[8]), xytext=(6.5, actual_index[8]+0.05),
#              arrowprops=dict(facecolor='black', shrink=0.05))

# plt.tight_layout()
# plt.savefig(os.path.join(BASE_DIR, "RPPI comparison plot 2.png"), dpi=300)
# plt.show()

# """
# 3. HeatMap 
# """
# macro_cols = [c for c in df_final.columns if c.startswith('macro_')]
# target_col = ['price_index_adjusted']
# analysis_cols = target_col + macro_cols

# # 2. Calculate correlation matrix (Pearson correlation)
# corr_matrix = df_final[analysis_cols].corr()

# # 3. Draw Heatmap
# plt.figure(figsize=(12, 8))
# sns.set_context("paper", font_scale=1.2)

# # Draw heatmap with coolwarm hue (green is neg, red is positive)
# sns.heatmap(corr_matrix, 
#             annot=True,          # Display specific numbers
#             fmt=".2f",           # floating point
#             cmap='coolwarm',     # colors
#             linewidths=0.5)

# plt.title('Ma trận tương quan giữa Giá nhà và Chỉ số Vĩ mô', fontsize=15, fontweight='bold')
# plt.tight_layout()
# plt.savefig(os.path.join(BASE_DIR, "cormatrix_price_macro_indicators.png"), dpi=300)
# plt.show()

# =============================================
# Step 5: Trend & Evolution Analysis
# =============================================
print("\n" + "="*60)
print("STEP 5: TREND & EVOLUTION ANALYSIS")
print("="*60)

# 1. Evolution of Adjusted Price over months
monthly_evolution = df_final.groupby('pub_month')['price_index_adjusted'].agg(['median', 'mean', 'count']).round(2)
print("\nMonthly Evolution of price_index_adjusted:")
print(monthly_evolution)

# 2. Evolution by Cluster
cluster_evolution = df_final.groupby(['pub_month', 'Cluster'])['price_index_adjusted'].median().unstack()
print("\nMedian price_index_adjusted by Cluster over months:")
print(cluster_evolution)

# Optional: Save plots
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_final, x='pub_month', y='price_index_adjusted', estimator='median', errorbar=None)
plt.title('Evolution of Adjusted Price (price_index_adjusted) over Time')
plt.xlabel('Month (2025)')
plt.ylabel('Price Index Adjusted')
plt.xticks(range(6,13))
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(BASE_DIR, "trend_evolution_adjusted_price.png"), dpi=200)
plt.close()

print("→ Saved trend evolution plots.")
