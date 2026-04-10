import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Đọc dữ liệu
print("1. Đang tải dữ liệu...")
df = pd.read_csv('processed_hanoi_apartments_with_clusters.csv')

# Khai báo các Features hợp lệ để tránh rò rỉ dữ liệu (Data Leakage)
features = ['area', 'bedroom_count', 'bathroom_count', 'district_encoded', 'avg_district_price_sqm']

X = df[features]
y = df['log_price'] 

# 2. Chia tập Train / Test (80% huấn luyện, 20% kiểm thử)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Huấn luyện mô hình LightGBM
print("2. Đang huấn luyện mô hình LightGBM...")
model = lgb.LGBMRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=8, 
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 4. Dự đoán trên tập Test
y_pred_log = model.predict(X_test)

# Chuyển ngược từ log_price về giá tiền thật (VNĐ)
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)

# 5. Đánh giá mô hình (Metrics)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TẬP TEST) ---")
print(f"R-squared (R2): {r2:.4f}")
print(f"MAE  : {mae / 1e9:.3f} Tỷ VNĐ")
print(f"RMSE : {rmse / 1e9:.3f} Tỷ VNĐ")


# ==========================================
# VẼ VÀ LƯU BIỂU ĐỒ 1: FEATURE IMPORTANCE
# ==========================================
plt.figure(figsize=(10, 6))
# Đổi tên feature cho biểu đồ dễ nhìn
feature_names = ['Diện tích (area)', 'Số phòng ngủ', 'Số phòng tắm', 'Mã Quận', 'Giá trung bình Quận']
importance = model.feature_importances_
# Sắp xếp
sorted_idx = np.argsort(importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, importance[sorted_idx], align='center', color='teal')
plt.yticks(pos, np.array(feature_names)[sorted_idx])
plt.xlabel('Mức độ quan trọng (Feature Importance)')
plt.title('Đâu là yếu tố quyết định Giá Căn Hộ nhiều nhất? (LightGBM)')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('lgbm_feature_importance.png', dpi=300)
print("\n -> Đã lưu biểu đồ: lgbm_feature_importance.png")
plt.close()

# ==========================================
# VẼ VÀ LƯU BIỂU ĐỒ 2: THỰC TẾ vs DỰ ĐOÁN
# ==========================================
plt.figure(figsize=(8, 8))
# Lấy ngẫu nhiên 2000 điểm để vẽ 
sample_idx = np.random.choice(len(y_test_actual), 2000, replace=False)

plt.scatter(y_test_actual.iloc[sample_idx] / 1e9, y_pred_actual[sample_idx] / 1e9, alpha=0.4, color='purple', s=15)
# Đường thẳng lý tưởng (y = x)
max_val = max(y_test_actual.iloc[sample_idx].max(), y_pred_actual[sample_idx].max()) / 1e9
plt.plot([0, max_val], [0, max_val], '--r', linewidth=2, label='Đường dự đoán hoàn hảo')

plt.title('So sánh: Giá Thực Tế vs Giá Dự Đoán (Tỷ VNĐ)')
plt.xlabel('Giá Thực Tế (Tỷ VNĐ)')
plt.ylabel('Giá Dự Đoán bởi LightGBM (Tỷ VNĐ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lgbm_actual_vs_predicted.png', dpi=300)
print(" -> Đã lưu biểu đồ: lgbm_actual_vs_predicted.png")
plt.close()