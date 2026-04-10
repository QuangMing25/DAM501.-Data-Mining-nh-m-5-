import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.ticker as ticker

# 1. Đọc dữ liệu đã qua tiền xử lý
print("1. Đang tải dữ liệu...")
df = pd.read_csv('processed_hanoi_apartments.csv')

# 2. Chọn features để gom cụm (Dùng dữ liệu đã chuẩn hóa StandardScaler)
X = df[['area_scaled', 'log_price_scaled']]

# Lấy mẫu 10,000 dòng để tính Silhouette Score 
sample_df = df.sample(n=10000, random_state=42)
X_sample = sample_df[['area_scaled', 'log_price_scaled']]

wcss = []
sil_scores = []
K_range = range(2, 8)

# 3. Tính toán WCSS (Elbow) và Silhouette
print("2. Đang tính toán Elbow và Silhouette Score (Vui lòng đợi vài giây)...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_sample)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_sample, kmeans.labels_))

# ==========================================
# VẼ VÀ LƯU BIỂU ĐỒ 1: ELBOW & SILHOUETTE
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Plot
axes[0].plot(K_range, wcss, marker='o', linestyle='--', color='b')
axes[0].set_title('Phuong phap Elbow (WCSS)')
axes[0].set_xlabel('So luong cum (K)')
axes[0].set_ylabel('WCSS')
axes[0].grid(True, alpha=0.3)

# Silhouette Plot
axes[1].plot(K_range, sil_scores, marker='s', linestyle='-', color='orange')
axes[1].set_title('Diem Silhouette Score')
axes[1].set_xlabel('So luong cum (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_evaluation.png', dpi=300) 
print(" -> Đã lưu biểu đồ: kmeans_evaluation.png")
plt.close() # Đóng plot để vẽ plot tiếp theo


# 4. Huấn luyện mô hình K-Means chính thức với K=3 trên toàn bộ dữ liệu
print("3. Tiến hành gom cụm với K=3...")
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X)

# Phân tích Profile của từng cụm (sử dụng Median)
cluster_profile = df.groupby('cluster')[['price', 'area', 'price_per_sqm']].median().round(0)
cluster_profile['count'] = df['cluster'].value_counts()
cluster_profile['price_billion'] = cluster_profile['price'] / 1e9

print("\n--- ĐẶC ĐIỂM CÁC PHÂN KHÚC (CLUSTERS) ---")
print(cluster_profile[['price_billion', 'area', 'price_per_sqm', 'count']])


# ==========================================
# VẼ VÀ LƯU BIỂU ĐỒ 2: SCATTER PLOT PHÂN KHÚC
# ==========================================
plt.figure(figsize=(10, 6))
# Lấy mẫu 5000 điểm để vẽ 
plot_df = df.sample(5000, random_state=42)

colors = ['green', 'blue', 'red'] # Màu sắc cho 3 cụm
for i in range(optimal_k):
    cluster_data = plot_df[plot_df['cluster'] == i]
    plt.scatter(cluster_data['area'], cluster_data['price'], 
                c=colors[i], label=f'Cluster {i}', alpha=0.5, s=20)

plt.title('Phan khuc Can ho: Dien tich vs Gia ban')
plt.xlabel('Dien tich (m2)')
plt.ylabel('Gia ban (VND)')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('kmeans_clusters_scatter.png', dpi=300) 
print(" -> Đã lưu biểu đồ: kmeans_clusters_scatter.png")
plt.close()


# 5. Lưu file kết quả
df.to_csv('processed_hanoi_apartments_with_clusters.csv', index=False, encoding='utf-8-sig')
print("\n4. Đã lưu dữ liệu hoàn chỉnh vào: processed_hanoi_apartments_with_clusters.csv")