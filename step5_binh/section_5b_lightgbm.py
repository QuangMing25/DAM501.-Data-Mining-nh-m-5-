import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 1. THIẾT LẬP ĐƯỜNG DẪN & TẢI DỮ LIỆU
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../step3_minh/data/hanoi_apartments_processed.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots_section_5")
os.makedirs(PLOT_DIR, exist_ok=True)

print("="*65)
print("PHẦN 5B: LIGHTGBM REGRESSION (SUPERVISED LEARNING)")
print("Thuật toán dự đoán giá nhà & Tìm kiếm quyền lực biến số")
print("="*65)

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Đã tải thành công Dữ liệu: {df.shape}")
except FileNotFoundError:
    print(f"[LỖI] Không tìm thấy dữ liệu tại: {DATA_PATH}")
    exit(1)

# Lựa chọn Features. Bỏ đi các cột Leak Model hoặc ID rác.
features_to_drop = ['price', 'log_price', 'price_per_m2', 'log_price_per_m2', 
                    'district_name', 'ward_name', 'street_name', 'project_name', 
                    'district_zone', 'published_at', 'house_direction'] 
# Các cột categoric text string (tên) đã được encoded ở bước 3, nên xóa các bản text
X_cols = [c for c in df.columns if c not in features_to_drop and not c.startswith('scaled_')]
X = df[X_cols]

# Target (Dự đoán giá thực - tỷ VND, ta nên dự đoán Log Price cho model mượt rồi luân hồi ngược lại)
y = df['log_price']

print(f"Số lượng Feature huấn luyện: {len(X.columns)}")

# ─────────────────────────────────────────────────────────────────
# 2. KHỞI TẠO TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Tập Huấn luyện Train: {X_train.shape[0]} | Tập Đánh giá Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────
# 3. FIT MÔ HÌNH BẰNG LIGHTGBM
# ─────────────────────────────────────────────────────────────────
print("\n--- BẮT ĐẦU TRAINING MÔ HÌNH LIGHTGBM ---")
# Các hyperparameter này được set chuẩn cho độ hội tụ dữ liệu > 50.000 dòng
params = {
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

model = lgb.LGBMRegressor(**params, n_estimators=800)
# Ở phiên bản mới LightGBM, ko can callback ma dua qua eval_metric
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse'
)

# ─────────────────────────────────────────────────────────────────
# 4. ĐÁNH GIÁ MÔ HÌNH (MODEL EVALUATION)
# ─────────────────────────────────────────────────────────────────
print("\n--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC ---")
y_pred_log = model.predict(X_test)

# Chuyển ngược lại đơn vị gốc (exp(log_price) - 1)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

r2 = r2_score(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)

print(f"R-squared (R²): {r2:.4f} (Mức độ học được của Model)")
print(f"RMSE: {rmse/1e9:.2f} Tỷ VND (Lệch chuẩn lớn thô)")
print(f"MAE:  {mae/1e9:.2f} Tỷ VND (Lệch chuẩn cốt lõi)")

# ─────────────────────────────────────────────────────────────────
# 5. TRỰC QUAN HÓA (FEATURE IMPORTANCE) - "SPLIT" vs "GAIN"
# ─────────────────────────────────────────────────────────────────
print("\n--- TRÍCH XUẤT QUYỀN LỰC BIẾN (FEATURE IMPORTANCE) ---")

importance_split = model.booster_.feature_importance(importance_type='split')
importance_gain = model.booster_.feature_importance(importance_type='gain')

df_imp = pd.DataFrame({
    'Feature': X_cols,
    'Split': importance_split,
    'Gain': importance_gain
}).sort_values(by='Gain', ascending=False).head(15)

# Bảng xếp hạng in console
print(df_imp[['Feature', 'Gain']].to_string(index=False))

# Đồ thị Bar chart ngang top 15 Gain
plt.figure(figsize=(12, 8))
sns.barplot(x='Gain', y='Feature', data=df_imp, palette='magma')
plt.title('Top 15 Yếu Tố Quyết Định Giá Nhà (Tính theo Thông tin Tăng cường - Gain)', fontsize=14, fontweight='bold')
plt.xlabel('Tầm quan trọng (Gain)')
plt.ylabel('Tính năng')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "lightgbm_01_feature_importance.png"))
plt.close()
print(f"-> Đã lưu Bảng Vàng Feature Importance tại: plots_section_5/lightgbm_01_feature_importance.png")

# Đồ thị Scatter So sánh Giá Thực vs Giá Dự Đoán
plt.figure(figsize=(8, 8))
plt.scatter(y_test_real/1e9, y_pred_real/1e9, alpha=0.3, color='#c0392b')
plt.plot([0, 20], [0, 20], color='black', lw=2, linestyle='--')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.title('Độ Trùng Khớp: Giá Dự Đoán vs Giá Thực (Tỷ VND)')
plt.xlabel('Giá Thực Tế (Tỷ VND)')
plt.ylabel('Mô Hình Dự Đoán (Tỷ VND)')
plt.savefig(os.path.join(PLOT_DIR, "lightgbm_02_predict_accuracy.png"))
plt.close()
print(f"-> Đã lưu Scatter Trùng khớp tại: plots_section_5/lightgbm_02_predict_accuracy.png")

print("\n[HOÀN THÀNH STEP 5B: LIGHTGBM REGRESSION]")
print("="*65)
