"""
DAM501 – DATA MINING
Section 5B: LightGBM Regression (Supervised Learning) — TUNED VERSION

Cải tiến so với v1:
  - Early Stopping (patience=50) tự tìm n_estimators tối ưu
  - 5-Fold Cross Validation → R² mean ± std
  - Thêm MAPE + MAPE by Zone metric
  - Tăng num_leaves 63→127, thêm min_child_samples=20
  - Residual Error Plot by Zone
  - Split vs Gain dual-bar chart
  - Dynamic range cho Scatter plot

Input:
  step3_minh/data/hanoi_apartments_processed.csv  (72.604 × 37 cols)

Output:
  plots_section_5/lightgbm_01_feature_importance.png
  plots_section_5/lightgbm_02_predict_accuracy.png
  plots_section_5/lightgbm_03_residual_by_zone.png
  plots_section_5/lightgbm_04_split_vs_gain.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. ĐỊA CHỈ DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../step3_minh/data/hanoi_apartments_processed.csv")
PLOT_DIR  = os.path.join(BASE_DIR, "plots_section_5")
os.makedirs(PLOT_DIR, exist_ok=True)

SEP = "=" * 70
print(SEP)
print("PHẦN 5B: LIGHTGBM REGRESSION (SUPERVISED) — TUNED VERSION")
print("Dự đoán giá chung cư Hà Nội & Khai phá Feature Importance")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TẢI DỮ LIỆU & CHUẨN BỊ FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 1. TẢI DỮ LIỆU ---")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"  [OK] Dataset: {df.shape}")
except FileNotFoundError:
    print(f"  [LỖI] Không tìm thấy: {DATA_PATH}")
    exit(1)

# Các cột loại bỏ: target leak, ID text, scaled (đã được encode sẵn)
features_to_drop = [
    'price', 'log_price', 'price_per_m2', 'log_price_per_m2',
    'district_name', 'ward_name', 'street_name', 'project_name',
    'district_zone', 'published_at', 'house_direction'
]
X_cols = [c for c in df.columns
          if c not in features_to_drop and not c.startswith('scaled_')]
X = df[X_cols].copy()
y = df['log_price']

print(f"  Features: {len(X_cols)} biến")
print(f"  Target  : log_price (→ chuyển ngược exp-1 để ra Tỷ VND)")

# Giữ lại zone để phân tích residual sau
zone_col = df['district_zone'] if 'district_zone' in df.columns else None

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

if zone_col is not None:
    zone_test = zone_col.iloc[y_test.index]
else:
    zone_test = None

# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETERS (TUNED)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 2. CẤU HÌNH HYPERPARAMETERS (TUNED) ---")

params = {
    'boosting_type'   : 'gbdt',
    'objective'       : 'regression',
    'metric'          : 'rmse',
    'learning_rate'   : 0.05,
    'num_leaves'      : 127,         # Tăng từ 63 → 127 (học pattern phức tạp hơn)
    'max_depth'       : -1,
    'min_child_samples': 20,         # Kiểm soát overfitting (mới thêm)
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq'    : 5,
    'reg_alpha'       : 0.1,         # L1 regularization (mới thêm)
    'reg_lambda'      : 0.1,         # L2 regularization (mới thêm)
    'verbose'         : -1,
    'random_state'    : 42,
}

print(f"  num_leaves       : {params['num_leaves']}")
print(f"  min_child_samples: {params['min_child_samples']}")
print(f"  learning_rate    : {params['learning_rate']}")
print(f"  reg_alpha        : {params['reg_alpha']}")
print(f"  reg_lambda       : {params['reg_lambda']}")
print(f"  Early Stopping   : patience = 50 rounds")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING VỚI EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3. TRAINING VỚI EARLY STOPPING ---")

model = lgb.LGBMRegressor(**params, n_estimators=2000)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

best_iter = model.best_iteration_
print(f"\n  → Early Stopping dừng tại iteration: {best_iter}")
print(f"    (Tiết kiệm {2000 - best_iter} trees không cần thiết)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. ĐÁNH GIÁ MÔ HÌNH — CÁC METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 4. ĐÁNH GIÁ ĐỘ CHÍNH XÁC ---")

y_pred_log  = model.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

r2   = r2_score(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae  = mean_absolute_error(y_test_real, y_pred_real)

# MAPE — tính trên các mẫu có giá > 0 để tránh chia 0
mask_nonzero = y_test_real > 0
mape = (np.abs((y_test_real[mask_nonzero] - y_pred_real[mask_nonzero])
               / y_test_real[mask_nonzero])).mean() * 100

print(f"  R-squared (R²) : {r2:.4f}  — Mô hình giải thích {r2*100:.2f}% biến động giá")
print(f"  RMSE           : {rmse/1e9:.3f} Tỷ VND  (Lệch chuẩn có trọng số lớn)")
print(f"  MAE            : {mae/1e9:.3f} Tỷ VND  (Sai số tuyệt đối trung bình)")
print(f"  MAPE           : {mape:.2f}%  (Sai số tương đối — quan trọng nhất)")

# MAPE by Zone
if zone_test is not None:
    print("\n  [MAPE theo Vùng Địa lý]")
    for zone in ['inner', 'middle', 'outer']:
        mask_zone = (zone_test == zone).values & mask_nonzero.values
        if mask_zone.sum() > 0:
            mape_z = (np.abs((y_test_real.values[mask_zone] - y_pred_real[mask_zone])
                             / y_test_real.values[mask_zone])).mean() * 100
            print(f"    {zone.capitalize():8s}: MAPE = {mape_z:.2f}%  (n={mask_zone.sum():,})")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CROSS-VALIDATION (5-FOLD)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 5. CROSS-VALIDATION (5-FOLD) ---")
print("  [Đang chạy CV — có thể mất 1-2 phút...]")

cv_model = lgb.LGBMRegressor(**params, n_estimators=best_iter)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_model, X, y, cv=kf, scoring='r2', n_jobs=-1)

print(f"  CV R² scores  : {cv_scores.round(4)}")
print(f"  CV R² mean    : {cv_scores.mean():.4f}")
print(f"  CV R² std     : {cv_scores.std():.4f}")
print(f"  → Model {'ổn định' if cv_scores.std() < 0.02 else 'có dao động'} "
      f"(std {'<' if cv_scores.std() < 0.02 else '>='} 0.02)")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Scatter — Giá Thực vs Giá Dự Đoán (Dynamic Range)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 6. VẼ BIỂU ĐỒ ---")

# Dynamic clamp: hiển thị đến p99 để tránh outlier làm scale lệch
p99 = float(np.percentile(y_test_real, 99))
clamp = min(p99, 25)  # cap tối đa 25 tỷ

fig, ax = plt.subplots(figsize=(8, 8))
x_vals = np.clip(y_test_real / 1e9, 0, clamp)
y_vals = np.clip(y_pred_real / 1e9, 0, clamp)

scatter = ax.scatter(x_vals, y_vals, alpha=0.25, s=8, color='#c0392b', rasterized=True)
ax.plot([0, clamp], [0, clamp], 'k--', lw=1.5, label='Dự đoán hoàn hảo')
ax.set_xlim(0, clamp)
ax.set_ylim(0, clamp)
ax.set_xlabel('Giá Thực Tế (Tỷ VND)', fontsize=12)
ax.set_ylabel('Giá Dự Đoán (Tỷ VND)', fontsize=12)
ax.set_title(
    f'Độ Trùng Khớp: Dự Đoán vs Thực Tế\n'
    f'R²={r2:.4f} | MAE={mae/1e9:.3f} Tỷ | MAPE={mape:.1f}%',
    fontweight='bold', fontsize=12
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "lightgbm_02_predict_accuracy.png"), dpi=150)
plt.close()
print(f"  → Đã lưu: plots_section_5/lightgbm_02_predict_accuracy.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Feature Importance — Split vs Gain (Dual Bar)
# ─────────────────────────────────────────────────────────────────────────────
importance_split = model.booster_.feature_importance(importance_type='split')
importance_gain  = model.booster_.feature_importance(importance_type='gain')

df_imp = pd.DataFrame({
    'Feature': X_cols,
    'Split'  : importance_split,
    'Gain'   : importance_gain,
}).sort_values('Gain', ascending=False).head(15)

print("\n[TOP 15 FEATURE IMPORTANCE BY GAIN]")
print(df_imp[['Feature', 'Gain', 'Split']].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Top 15 Feature Importance — LightGBM Tuned', fontsize=14, fontweight='bold')

# Panel trái: GAIN
sns.barplot(x='Gain', y='Feature', data=df_imp, palette='magma', ax=axes[0])
axes[0].set_title('Tầm Quan Trọng theo GAIN\n(Tổng thông tin đóng góp)', fontsize=12)
axes[0].set_xlabel('Gain')
axes[0].set_ylabel('Feature')

# Panel phải: SPLIT (top 15 theo Split)
df_imp_split = pd.DataFrame({
    'Feature': X_cols,
    'Split'  : importance_split,
}).sort_values('Split', ascending=False).head(15)

sns.barplot(x='Split', y='Feature', data=df_imp_split, palette='viridis', ax=axes[1])
axes[1].set_title('Tầm Quan Trọng theo SPLIT\n(Số lần được dùng để phân nhánh)', fontsize=12)
axes[1].set_xlabel('Split Count')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "lightgbm_01_feature_importance.png"), dpi=150)
plt.close()
print(f"  → Đã lưu: plots_section_5/lightgbm_01_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Residual Distribution by Zone
# ─────────────────────────────────────────────────────────────────────────────
if zone_test is not None:
    residuals    = (y_test_real.values - y_pred_real) / 1e9   # Tỷ VND
    zone_arr     = zone_test.values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle('Phân Phối Sai Số Dự Đoán (Residuals) theo Vùng Địa lý',
                 fontsize=14, fontweight='bold')

    zone_colors = {'inner': '#e74c3c', 'middle': '#3498db', 'outer': '#2ecc71'}
    ZONE_ORDER  = ['inner', 'middle', 'outer']

    for i, zone in enumerate(ZONE_ORDER):
        mask_z = zone_arr == zone
        res_z  = residuals[mask_z]
        med_z  = np.median(res_z)
        std_z  = np.std(res_z)

        axes[i].hist(res_z, bins=60, color=zone_colors[zone], alpha=0.75, edgecolor='white', lw=0.3)
        axes[i].axvline(0,     color='black', lw=1.5, linestyle='--', label='Zero error')
        axes[i].axvline(med_z, color='red',   lw=1.5, linestyle='-',  label=f'Median={med_z:+.2f} Tỷ')
        axes[i].set_title(f'{zone.capitalize()} Zone\n(n={mask_z.sum():,} | std={std_z:.2f} Tỷ)',
                          fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Sai số (Thực - Dự Đoán) Tỷ VND')
        axes[i].set_ylabel('Số lượng')
        axes[i].legend(fontsize=9)
        axes[i].set_xlim(-10, 10)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lightgbm_03_residual_by_zone.png"), dpi=150)
    plt.close()
    print(f"  → Đã lưu: plots_section_5/lightgbm_03_residual_by_zone.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: CV Score Distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fold_labels = [f'Fold {i+1}' for i in range(len(cv_scores))]
bars = ax.bar(fold_labels, cv_scores, color='#2980b9', edgecolor='white', lw=0.5, width=0.6)
ax.axhline(cv_scores.mean(), color='red', linestyle='--', lw=1.5,
           label=f'Mean R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.002, f'{score:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(max(0, cv_scores.min() - 0.05), min(1.0, cv_scores.max() + 0.05))
ax.set_xlabel('Fold')
ax.set_ylabel('R² Score')
ax.set_title('5-Fold Cross Validation — LightGBM\n(Đánh giá độ ổn định mô hình)',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "lightgbm_04_cross_validation.png"), dpi=150)
plt.close()
print(f"  → Đã lưu: plots_section_5/lightgbm_04_cross_validation.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. KẾT QUẢ CUỐI
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("[KẾT QUẢ CUỐI — LIGHTGBM TUNED]")
print(f"  Best Iteration     : {best_iter} trees")
print(f"  R² (test set)      : {r2:.4f}")
print(f"  MAE                : {mae/1e9:.3f} Tỷ VND")
print(f"  MAPE               : {mape:.2f}%")
print(f"  CV R² mean ± std   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  num_leaves         : {params['num_leaves']}")
print(f"  reg_alpha/lambda   : {params['reg_alpha']} / {params['reg_lambda']}")
print(f"\n[HOÀN THÀNH STEP 5B: LIGHTGBM REGRESSION — TUNED]")
print("=" * 70)
