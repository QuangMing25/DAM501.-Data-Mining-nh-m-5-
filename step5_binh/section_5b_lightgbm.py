import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 1. THIẾT LẬP ĐƯỜNG DẪN & TẢI DỮ LIỆU
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, "../step3_minh/data/hanoi_apartments_processed.csv")
CLUSTER_PATH   = os.path.join(BASE_DIR, "../step3_minh/data/hanoi_apartments_for_clustering.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots_section_5")
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 65)
print("PHẦN 5B: LIGHTGBM REGRESSION (SUPERVISED LEARNING)")
print("So sánh: Baseline vs Tích hợp K-Means Cluster từ Bước 5A")
print("=" * 65)

# --- Tải dữ liệu chính (processed) ---
try:
    df = pd.read_csv(PROCESSED_PATH)
    print(f"[1] Đã tải dữ liệu processed: {df.shape}")
except FileNotFoundError:
    print(f"[LỖI] Không tìm thấy: {PROCESSED_PATH}")
    exit(1)

# --- Tải dữ liệu clustering (scaled) để chạy K-Means ---
try:
    df_cluster = pd.read_csv(CLUSTER_PATH)
    print(f"[2] Đã tải dữ liệu clustering: {df_cluster.shape}")
except FileNotFoundError:
    print(f"[LỖI] Không tìm thấy: {CLUSTER_PATH}")
    exit(1)

# ─────────────────────────────────────────────────────────────────
# 2. CHUẨN BỊ FEATURES & TARGET CHUNG
# ─────────────────────────────────────────────────────────────────
# Các cột cần loại bỏ: target, leak, text gốc, và các cột scaled (thuộc về clustering)
features_to_drop = ['price', 'log_price', 'price_per_m2', 'log_price_per_m2',
                    'district_name', 'ward_name', 'street_name', 'project_name',
                    'district_zone', 'published_at', 'house_direction']

X_cols_base = [c for c in df.columns if c not in features_to_drop and not c.startswith('scaled_')]

# Target: dự đoán log_price rồi chuyển ngược về giá thực
y = df['log_price']

print(f"\nSố features Baseline: {len(X_cols_base)}")
print(f"Tổng mẫu: {len(df)}")

# Cố định train/test split cho cả 2 mô hình (đảm bảo so sánh công bằng)
SPLIT_SEED = 42
TEST_SIZE  = 0.2

# ─────────────────────────────────────────────────────────────────
# 3. HYPERPARAMETERS CHUNG CHO LIGHTGBM
# ─────────────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════
#  HÀM HUẤN LUYỆN & ĐÁNH GIÁ (dùng chung cho 2 lần chạy)
# ═══════════════════════════════════════════════════════════════
def train_and_evaluate(X, y, model_name, cat_features=None):
    """Huấn luyện LightGBM và trả về dict metrics + predictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )

    model = lgb.LGBMRegressor(**lgb_params, n_estimators=N_ESTIMATORS)

    fit_params = {
        'eval_set': [(X_test, y_test)],
        'eval_metric': 'rmse',
    }
    if cat_features:
        fit_params['categorical_feature'] = cat_features

    model.fit(X_train, y_train, **fit_params)

    # Dự đoán
    y_pred_log = model.predict(X_test)
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    # Metrics
    r2   = r2_score(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae  = mean_absolute_error(y_test_real, y_pred_real)
    mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100

    results = {
        'name': model_name,
        'model': model,
        'R2': r2,
        'RMSE_ty': rmse / 1e9,
        'MAE_ty': mae / 1e9,
        'MAPE': mape,
        'y_test_real': y_test_real,
        'y_pred_real': y_pred_real,
        'X_cols': list(X.columns),
    }

    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse/1e9:.4f} Tỷ VND")
    print(f"  MAE  : {mae/1e9:.4f} Tỷ VND")
    print(f"  MAPE : {mape:.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════
#  MÔ HÌNH 1: BASELINE (KHÔNG CÓ CLUSTER)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("MÔ HÌNH 1: LIGHTGBM BASELINE (Không tích hợp K-Means)")
print("=" * 65)

X_baseline = df[X_cols_base]
res_baseline = train_and_evaluate(X_baseline, y, "LightGBM Baseline")

# ═══════════════════════════════════════════════════════════════
#  TÍCH HỢP K-MEANS TỪ BƯỚC 5A
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TÍCH HỢP K-MEANS CLUSTERING TỪ BƯỚC 5A")
print("=" * 65)

# Chạy lại K-Means với K=3 (giống hệt logic bước 5A)
scaled_features = [c for c in df_cluster.columns if c.startswith('scaled_')]
X_kmeans = df_cluster[scaled_features].values

OPTIMAL_K = 3
print(f"\nĐang chạy K-Means với K={OPTIMAL_K} (tái tạo kết quả Bước 5A)...")
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_kmeans)

print(f"Phân bố cụm:")
for c_id in range(OPTIMAL_K):
    count = (cluster_labels == c_id).sum()
    pct = count / len(cluster_labels) * 100
    print(f"  Cụm {c_id}: {count:,} căn hộ ({pct:.1f}%)")

# Gán nhãn Cluster vào dataframe chính
df['Cluster'] = cluster_labels

# ═══════════════════════════════════════════════════════════════
#  MÔ HÌNH 2: LIGHTGBM + K-MEANS CLUSTER
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("MÔ HÌNH 2: LIGHTGBM + K-MEANS CLUSTER FEATURE")
print("=" * 65)

X_cols_enhanced = X_cols_base + ['Cluster']
X_enhanced = df[X_cols_enhanced].copy()

# Đánh dấu Cluster là categorical feature để LightGBM xử lý đúng
X_enhanced['Cluster'] = X_enhanced['Cluster'].astype('category')

res_enhanced = train_and_evaluate(
    X_enhanced, y,
    "LightGBM + K-Means Cluster",
    cat_features=['Cluster']
)

# ═══════════════════════════════════════════════════════════════
#  SO SÁNH 2 MÔ HÌNH
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BẢNG SO SÁNH: BASELINE vs TÍCH HỢP K-MEANS")
print("=" * 65)

comparison = pd.DataFrame({
    'Metric': ['R² (càng cao càng tốt)', 'RMSE (Tỷ VND)', 'MAE (Tỷ VND)', 'MAPE (%)'],
    'Baseline': [
        f"{res_baseline['R2']:.4f}",
        f"{res_baseline['RMSE_ty']:.4f}",
        f"{res_baseline['MAE_ty']:.4f}",
        f"{res_baseline['MAPE']:.2f}%"
    ],
    'Baseline + K-Means': [
        f"{res_enhanced['R2']:.4f}",
        f"{res_enhanced['RMSE_ty']:.4f}",
        f"{res_enhanced['MAE_ty']:.4f}",
        f"{res_enhanced['MAPE']:.2f}%"
    ],
    'Thay đổi': [
        f"{(res_enhanced['R2'] - res_baseline['R2'])*100:+.2f} điểm %",
        f"{(res_enhanced['RMSE_ty'] - res_baseline['RMSE_ty'])*1000:+.1f} triệu",
        f"{(res_enhanced['MAE_ty'] - res_baseline['MAE_ty'])*1000:+.1f} triệu",
        f"{(res_enhanced['MAPE'] - res_baseline['MAPE']):+.2f}%"
    ]
})
print(comparison.to_string(index=False))

# Đánh giá tổng kết
r2_diff = res_enhanced['R2'] - res_baseline['R2']
if r2_diff > 0.001:
    verdict = "✅ TÍCH HỢP K-MEANS CẢI THIỆN MÔ HÌNH"
elif r2_diff < -0.001:
    verdict = "⚠️ TÍCH HỢP K-MEANS KHÔNG CẢI THIỆN (có thể do thông tin đã được encode trong các features khác)"
else:
    verdict = "➡️ HIỆU QUẢ TƯƠNG ĐƯƠNG (K-Means không thêm thông tin mới đáng kể)"
print(f"\n{'─'*50}")
print(f"  KẾT LUẬN: {verdict}")
print(f"{'─'*50}")

# ═══════════════════════════════════════════════════════════════
#  TRỰC QUAN HÓA
# ═══════════════════════════════════════════════════════════════

# --- 1. BIỂU ĐỒ SO SÁNH METRICS ---
print("\n--- TRỰC QUAN HÓA KẾT QUẢ ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('So sánh LightGBM: Baseline vs Tích hợp K-Means Cluster',
             fontsize=16, fontweight='bold', y=1.02)

metrics = ['R2', 'RMSE_ty', 'MAE_ty']
titles  = ['R² Score\n(Càng cao càng tốt)', 'RMSE (Tỷ VND)\n(Càng thấp càng tốt)', 'MAE (Tỷ VND)\n(Càng thấp càng tốt)']
colors  = [['#3498db', '#2ecc71'], ['#e74c3c', '#e67e22'], ['#9b59b6', '#f39c12']]

for ax, metric, title, clr in zip(axes, metrics, titles, colors):
    vals = [res_baseline[metric], res_enhanced[metric]]
    bars = ax.bar(['Baseline', '+ K-Means'], vals, color=clr, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('')

    # Ghi giá trị lên cột
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "lightgbm_00_comparison_metrics.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"-> Đã lưu biểu đồ so sánh: plots_section_5/lightgbm_00_comparison_metrics.png")

# --- 2. FEATURE IMPORTANCE (của model tích hợp) ---
model_best = res_enhanced['model']
importance_gain = model_best.booster_.feature_importance(importance_type='gain')
importance_split = model_best.booster_.feature_importance(importance_type='split')

df_imp = pd.DataFrame({
    'Feature': res_enhanced['X_cols'],
    'Split': importance_split,
    'Gain': importance_gain
}).sort_values(by='Gain', ascending=False).head(15)

print("\n[TOP 15 FEATURE IMPORTANCE - MÔ HÌNH TÍCH HỢP K-MEANS]")
print(df_imp[['Feature', 'Gain']].to_string(index=False))

# Highlight xem Cluster nằm ở ranking nào
cluster_rank = df_imp.reset_index(drop=True)
cluster_row = cluster_rank[cluster_rank['Feature'] == 'Cluster']
if not cluster_row.empty:
    rank_pos = cluster_row.index[0] + 1
    print(f"\n🏆 Feature 'Cluster' (từ K-Means) xếp hạng #{rank_pos}/15 về Gain importance")
else:
    # Tìm trong toàn bộ features
    df_imp_full = pd.DataFrame({
        'Feature': res_enhanced['X_cols'],
        'Gain': importance_gain
    }).sort_values(by='Gain', ascending=False).reset_index(drop=True)
    cluster_full = df_imp_full[df_imp_full['Feature'] == 'Cluster']
    if not cluster_full.empty:
        rank_full = cluster_full.index[0] + 1
        total = len(df_imp_full)
        print(f"\nℹ️  Feature 'Cluster' xếp hạng #{rank_full}/{total} (ngoài Top 15)")

# Vẽ bar chart
plt.figure(figsize=(12, 8))
palette = ['#e74c3c' if f == 'Cluster' else '#3498db' for f in df_imp['Feature']]
sns.barplot(x='Gain', y='Feature', data=df_imp, palette=palette)
plt.title('Top 15 Feature Importance (LightGBM + K-Means)\n🔴 = Cluster Feature từ Bước 5A',
          fontsize=14, fontweight='bold')
plt.xlabel('Tầm quan trọng (Gain)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "lightgbm_01_feature_importance.png"), dpi=150)
plt.close()
print(f"-> Đã lưu Feature Importance: plots_section_5/lightgbm_01_feature_importance.png")

# --- 3. SCATTER: GIÁ DỰ ĐOÁN vs GIÁ THỰC (cả 2 model) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for ax, res, color, title_suffix in [
    (ax1, res_baseline, '#3498db', 'Baseline'),
    (ax2, res_enhanced, '#e74c3c', '+ K-Means Cluster')
]:
    y_real = res['y_test_real'] / 1e9
    y_pred = res['y_pred_real'] / 1e9
    ax.scatter(y_real, y_pred, alpha=0.2, color=color, s=8)
    ax.plot([0, 20], [0, 20], color='black', lw=2, linestyle='--', label='Perfect Fit')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_title(f'{title_suffix}\nR² = {res["R2"]:.4f} | MAE = {res["MAE_ty"]:.4f} Tỷ',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Giá Thực Tế (Tỷ VND)')
    ax.set_ylabel('Giá Dự Đoán (Tỷ VND)')
    ax.legend()

fig.suptitle('Độ Trùng Khớp: Giá Dự Đoán vs Giá Thực', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "lightgbm_02_predict_accuracy.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"-> Đã lưu Scatter so sánh: plots_section_5/lightgbm_02_predict_accuracy.png")

# --- 4. PHÂN BỐ SAI SỐ THEO CỤM ---
print("\n--- PHÂN TÍCH SAI SỐ THEO TỪNG CỤM (MÔ HÌNH TÍCH HỢP) ---")

# Lấy lại test set index
X_en = df[X_cols_enhanced].copy()
X_en['Cluster'] = X_en['Cluster'].astype('category')
_, X_test_en, _, y_test_en = train_test_split(X_en, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)

y_pred_test = model_best.predict(X_test_en)
y_pred_real_test = np.expm1(y_pred_test)
y_test_real_test = np.expm1(y_test_en)

test_analysis = pd.DataFrame({
    'Cluster': X_test_en['Cluster'].values,
    'Actual': y_test_real_test,
    'Predicted': y_pred_real_test,
    'Error': np.abs(y_test_real_test - y_pred_real_test)
})

cluster_error = test_analysis.groupby('Cluster').agg(
    So_luong=('Error', 'count'),
    MAE_Ty=('Error', lambda x: f"{x.mean()/1e9:.4f}"),
    Median_Error_Ty=('Error', lambda x: f"{x.median()/1e9:.4f}"),
    Gia_TB_Ty=('Actual', lambda x: f"{x.mean()/1e9:.2f}")
)
print(cluster_error.to_string())

# Boxplot sai số theo cụm
plt.figure(figsize=(10, 6))
test_analysis['Error_Ty'] = test_analysis['Error'] / 1e9
sns.boxplot(x='Cluster', y='Error_Ty', data=test_analysis, palette='Set2',
            showfliers=False)
plt.title('Phân bố Sai số Dự đoán theo Cụm K-Means', fontsize=14, fontweight='bold')
plt.xlabel('Cụm (Cluster từ Bước 5A)')
plt.ylabel('Sai số tuyệt đối (Tỷ VND)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "lightgbm_03_error_by_cluster.png"), dpi=150)
plt.close()
print(f"-> Đã lưu Boxplot sai số: plots_section_5/lightgbm_03_error_by_cluster.png")

print("\n" + "=" * 65)
print("[HOÀN THÀNH STEP 5B: LIGHTGBM REGRESSION + K-MEANS INTEGRATION]")
print("=" * 65)
