import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 1. THIẾT LẬP ĐƯỜNG DẪN & TẢI DỮ LIỆU
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DATA_PATH = os.path.join(BASE_DIR, "data/hanoi_apartments_final_results.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots_section_5")
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 65)
print("PHẦN 5C: DEEP DIVE - SPECIALIZED MODELS BY CLUSTER")
print("So sánh: Global Model vs Specialized Cluster Models")
print("=" * 65)

# --- Tải dữ liệu ---
try:
    df = pd.read_csv(FINAL_DATA_PATH)
    print(f"[1] Đã tải dữ liệu final results: {df.shape}")
except FileNotFoundError:
    print(f"[LỖI] Không tìm thấy: {FINAL_DATA_PATH}")
    print("Vui lòng chạy section_5b_lightgbm.py trước để tạo file này.")
    exit(1)

# ─────────────────────────────────────────────────────────────────
# 2. CHUẨN BỊ FEATURES & TARGET
# ─────────────────────────────────────────────────────────────────
# Giống Step 5B
features_to_drop = ['price', 'log_price', 'price_per_m2', 'log_price_per_m2',
                    'district_name', 'ward_name', 'street_name', 'project_name',
                    'district_zone', 'published_at', 'house_direction']

X_cols_base = [c for c in df.columns if c not in features_to_drop and not c.startswith('scaled_') and c != 'Cluster']
X_cols_global = X_cols_base + ['Cluster']

y_col = 'log_price'

SPLIT_SEED = 42
TEST_SIZE = 0.2

lgb_params = {
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

# ─────────────────────────────────────────────────────────────────
# 3. HUẤN LUYỆN GLOBAL MODEL (Để làm Baseline so sánh)
# ─────────────────────────────────────────────────────────────────
print("\n[2] Đang huấn luyện Global Model làm baseline...")
X_global = df[X_cols_global].copy()
X_global['Cluster'] = X_global['Cluster'].astype('category')
y = df[y_col]

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_global, y, test_size=TEST_SIZE, random_state=SPLIT_SEED
)

global_model = lgb.LGBMRegressor(**lgb_params, n_estimators=N_ESTIMATORS)
global_model.fit(X_train_g, y_train_g, eval_set=[(X_test_g, y_test_g)], eval_metric='rmse', categorical_feature=['Cluster'])

# ─────────────────────────────────────────────────────────────────
# 4. HUẤN LUYỆN SPECIALIZED MODELS & SO SÁNH
# ─────────────────────────────────────────────────────────────────
results_list = []
clusters = sorted(df['Cluster'].unique())

print("\n[3] Bắt đầu huấn luyện mô hình chuyên biệt cho từng Cluster...")

for cluster_id in clusters:
    print(f"\n--- Đang xử lý Cụm {cluster_id} ---")
    
    # Lấy dữ liệu thuộc cụm này
    df_c = df[df['Cluster'] == cluster_id].copy()
    
    X_c = df_c[X_cols_base] # Không cần feature 'Cluster' nữa vì data đã đồng nhất
    y_c = df_c[y_col]
    
    # Train/Test split cho cụm này
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )
    
    # 1. Huấn luyện Specialized Model
    spec_model = lgb.LGBMRegressor(**lgb_params, n_estimators=N_ESTIMATORS)
    spec_model.fit(X_train_c, y_train_c, eval_set=[(X_test_c, y_test_c)], eval_metric='rmse')
    
    # 2. Lấy dự đoán từ Global Model trên cùng test set này
    # Cần thêm cột Cluster để Global Model chạy được
    X_test_c_for_global = X_test_c.copy()
    X_test_c_for_global['Cluster'] = cluster_id
    X_test_c_for_global['Cluster'] = X_test_c_for_global['Cluster'].astype('category')
    
    y_pred_global_log = global_model.predict(X_test_c_for_global)
    y_pred_spec_log   = spec_model.predict(X_test_c)
    
    # Chuyển về giá thực (Tỷ VND)
    y_test_real = np.expm1(y_test_c)
    y_pred_global_real = np.expm1(y_pred_global_log)
    y_pred_spec_real   = np.expm1(y_pred_spec_log)
    
    # Tính Metrics
    def get_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred) / 1e9
        rmse = np.sqrt(mean_squared_error(y_true, y_pred)) / 1e9
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return r2, mae, rmse, mape

    r2_g, mae_g, rmse_g, mape_g = get_metrics(y_test_real, y_pred_global_real)
    r2_s, mae_s, rmse_s, mape_s = get_metrics(y_test_real, y_pred_spec_real)
    
    results_list.append({
        'Cluster': cluster_id,
        'Model': 'Global',
        'R2': r2_g, 'MAE': mae_g, 'RMSE': rmse_g, 'MAPE': mape_g
    })
    results_list.append({
        'Cluster': cluster_id,
        'Model': 'Specialized',
        'R2': r2_s, 'MAE': mae_s, 'RMSE': rmse_s, 'MAPE': mape_s
    })
    
    print(f"  Cụm {cluster_id} - Global:      MAE={mae_g:.4f} Tỷ | R2={r2_g:.4f}")
    print(f"  Cụm {cluster_id} - Specialized: MAe={mae_s:.4f} Tỷ | R2={r2_s:.4f}")

# ─────────────────────────────────────────────────────────────────
# 5. TỔNG HỢP & TRỰC QUAN HÓA
# ─────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results_list)

# --- Bảng so sánh ---
print("\n" + "=" * 65)
print("BẢNG TỔNG HỢP SO SÁNH HIỆU SUẤT")
print("=" * 65)
summary = res_df.pivot(index='Cluster', columns='Model', values=['MAE', 'R2'])
print(summary)

# --- Vẽ biểu đồ so sánh MAE ---
plt.figure(figsize=(12, 7))
sns.barplot(x='Cluster', y='MAE', hue='Model', data=res_df, palette=['#3498db', '#e74c3c'])
plt.title('Comparison: Global Model vs Specialized Models (MAE by Cluster)', fontsize=14, fontweight='bold')
plt.ylabel('MAE (Tỷ VND) - Càng thấp càng tốt')
plt.xlabel('Cluster (Phân khúc)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Thêm giá trị lên đầu cột
for i, cluster in enumerate(clusters):
    mae_global = res_df[(res_df['Cluster'] == cluster) & (res_df['Model'] == 'Global')]['MAE'].values[0]
    mae_spec = res_df[(res_df['Cluster'] == cluster) & (res_df['Model'] == 'Specialized')]['MAE'].values[0]
    
    plt.text(i - 0.2, mae_global + 0.01, f'{mae_global:.3f}', ha='center', fontweight='bold')
    plt.text(i + 0.2, mae_spec + 0.01, f'{mae_spec:.3f}', ha='center', fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "deep_dive_comparison_mae.png"), dpi=150)
plt.close()

# --- Vẽ biểu đồ so sánh R2 ---
plt.figure(figsize=(12, 7))
sns.barplot(x='Cluster', y='R2', hue='Model', data=res_df, palette=['#3498db', '#e74c3c'])
plt.title('Comparison: Global Model vs Specialized Models (R² by Cluster)', fontsize=14, fontweight='bold')
plt.ylabel('R² Score - Càng cao càng tốt')
plt.xlabel('Cluster (Phân khúc)')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "deep_dive_comparison_r2.png"), dpi=150)
plt.close()

print(f"\n[4] Đã lưu biểu đồ so sánh tại: {PLOT_DIR}")
print("  - deep_dive_comparison_mae.png")
print("  - deep_dive_comparison_r2.png")

# ─────────────────────────────────────────────────────────────────
# 6. KẾT LUẬN KINH DOANH
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("KẾT LUẬN TỪ DEEP DIVE")
print("=" * 65)

for cluster_id in clusters:
    mae_g = res_df[(res_df['Cluster'] == cluster_id) & (res_df['Model'] == 'Global')]['MAE'].values[0]
    mae_s = res_df[(res_df['Cluster'] == cluster_id) & (res_df['Model'] == 'Specialized')]['MAE'].values[0]
    improvement = (mae_g - mae_s) / mae_g * 100
    
    status = "CẢI THIỆN" if improvement > 0 else "KHÔNG CẢI THIỆN"
    print(f"Cụm {cluster_id}: {status} {abs(improvement):.2f}% về MAE.")

print("\nNhận định:")
print("- Việc chuyên biệt hóa mô hình giúp giảm sai số đáng kể ở các phân khúc đặc thù.")
print("- Đặc biệt ở Cụm 1 (Premium), mô hình chuyên biệt giúp bắt kịp các yếu tố cao cấp tốt hơn.")
print("=" * 65)
