import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 1. THIẾT LẬP ĐƯỜNG DẪN & TẢI DỮ LIỆU
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../step3_minh/data/hanoi_apartments_for_clustering.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots_section_5")
os.makedirs(PLOT_DIR, exist_ok=True)

print("="*65)
print("PHẦN 5A: K-MEANS CLUSTERING (UNSUPERVISED LEARNING)")
print("Thuật toán khai phá phân khúc ngầm của hệ sinh thái Chung cư Hà Nội")
print("="*65)

print("\n--- 1. TẢI DỮ LIỆU ---")
try:
    df_cluster = pd.read_csv(DATA_PATH)
    print(f"Đã tải thành công ma trận: {df_cluster.shape}")
except FileNotFoundError:
    print(f"[LỖI] Không tìm thấy dữ liệu tại: {DATA_PATH}")
    print("Vui lòng chạy lại file section_3_preprocessing.py trước.")
    exit(1)

# Các features dùng để scale và chạy model
scaled_features = [c for c in df_cluster.columns if c.startswith('scaled_')]
X = df_cluster[scaled_features].values

# ─────────────────────────────────────────────────────────────────
# 2. XÁC ĐỊNH SỐ CỤM TỐI ƯU (ELBOW & SILHOUETTE)
# ─────────────────────────────────────────────────────────────────
print("\n--- 2. TÌM SỐ CỤM (K) TỐI ƯU ---")
# Do dữ liệu lớn (72k+ dòng), tính Silhouette sẽ bị tràn RAM/chậm, ta dùng sample 10%
X_sample_idx = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.1), replace=False)
X_sample = X[X_sample_idx]

inertia = []
sil_scores = []
K_range = range(2, 8)

print("Đang khởi chạy thuật toán để đo Inertia và Silhouette (Từ K=2 đến K=7)...")
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X)
    inertia.append(kmeans_temp.inertia_)
    
    # Tính điểm Silhouette trên mẫu ngẫu nhiên
    labels_sample = kmeans_temp.predict(X_sample)
    sil = silhouette_score(X_sample, labels_sample)
    sil_scores.append(sil)
    print(f"  K = {k}: Inertia = {kmeans_temp.inertia_:,.0f} | Silhouette Score = {sil:.4f}")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(K_range, inertia, 'bx-', label='Inertia (Elbow)')
ax1.set_xlabel('Số lượng Cụm (K)')
ax1.set_ylabel('Inertia (WCSS)', color='b')
ax1.tick_params('y', colors='b')
ax1.set_title('Elbow Method và Silhouette Score phân rã Dữ liệu')

ax2 = ax1.twinx()
ax2.plot(K_range, sil_scores, 'ro-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')
ax2.tick_params('y', colors='r')
plt.savefig(os.path.join(PLOT_DIR, "kmeans_01_elbow_silhouette.png"))
plt.close()
print(f"-> Đã lưu biểu đồ Elbow Method tại: plots_section_5/kmeans_01_elbow_silhouette.png")

# ─────────────────────────────────────────────────────────────────
# 3. FIT MÔ HÌNH CHÍNH THỨC VỚI K = 3
# ─────────────────────────────────────────────────────────────────
# Tại EDA Step 4, PCA và cấu trúc 3 Zones (Inner/Middle/Outer) đã chứng minh K=3 là hợp lý nhất cho kinh doanh
OPTIMAL_K = 3
print(f"\n--- 3. HUẤN LUYỆN K-MEANS VỚI K = {OPTIMAL_K} ---")
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(X)
print(f"Đã gán nhãn {OPTIMAL_K} cụm cho {len(df_cluster)} căn hộ.")

# ─────────────────────────────────────────────────────────────────
# 4. TRỰC QUAN HÓA BẰNG PCA 2-CHIỀU
# ─────────────────────────────────────────────────────────────────
print("\n--- 4. TRỰC QUAN HÓA (PCA 2D SCATTER) ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_cluster['PCA_1'] = X_pca[:, 0]
df_cluster['PCA_2'] = X_pca[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA_1', y='PCA_2',
    hue='Cluster',
    palette='viridis',
    data=df_cluster.sample(10000) # Chỉ vẽ 10000 điểm để khỏi lag hình
)
plt.title(f'K-Means Clusters Visualization (PCA - {pca.explained_variance_ratio_.sum()*100:.1f}% Variance)', fontweight='bold')
plt.savefig(os.path.join(PLOT_DIR, "kmeans_02_pca_clusters.png"))
plt.close()
print(f"-> Đã lưu Scatter Plot tại: plots_section_5/kmeans_02_pca_clusters.png")

# ─────────────────────────────────────────────────────────────────
# 5. KHAI PHÁ PROFILE CỦA CÁC CỤM (CLUSTER CHARACTERISTICS)
# ─────────────────────────────────────────────────────────────────
print("\n--- 5. KHAI PHÁ ĐẶC TRƯNG TỪNG CỤM (BẢN CHẤT KINH TẾ) ---")

# Chúng ta cần load lại dữ liệu chưa bị scaled để đọc thông số tiền (Tỷ đồng) và diện tích
RAW_DATA_PATH = os.path.join(BASE_DIR, "../step3_minh/data/hanoi_apartments_processed.csv")
df_raw = pd.read_csv(RAW_DATA_PATH)
df_raw['Cluster'] = df_cluster['Cluster']

cluster_profile = df_raw.groupby('Cluster').agg(
    So_luong=('price', 'count'),
    Ty_le=('price', lambda x: f"{(len(x)/len(df_raw))*100:.1f}%"),
    Gia_Trung_Vi=('price', lambda x: f"{x.median()/1e9:.2f} Tỷ"),
    Gia_M2_Trung_vi=('price_per_m2', lambda x: f"{x.median()/1e6:.1f} Tr/m²"),
    Dien_tich_TB=('area', 'mean'),
    So_Phong_Ngu_TB=('bedroom_count', 'mean'),
).round(1)

print("\n[HỒ SƠ CỤM K-MEANS - KNOWLEDGE DISCOVERY]")
print(cluster_profile.to_string())

# Cross-check với vị trí địa lý
zone_cross = pd.crosstab(df_raw['Cluster'], df_raw['district_zone'], normalize='index') * 100
print("\n[TỶ LỆ PHÂN BỐ ĐỊA LÝ TRONG CÁC CỤM (%)]")
print(zone_cross.round(1).to_string())

# Phân loại và gán tên cụm tự động bằng code (Dựa trên Giá/m2)
cluster_centers_price = cluster_profile['Gia_M2_Trung_vi'].str.replace(' Tr/m²', '').astype(float)
sorted_idx = cluster_centers_price.sort_values().index
labels = ["Phổ Thông Ngoại Ô", "Tầm Trung Lõi", "Cao Cấp (Premium)"]

print("\n--- DÁN NHÃN KINH DOANH CHO CỤM ---")
for i, c_id in enumerate(sorted_idx):
    print(f"Cụm {c_id}: {labels[i]} (Giá Median: {cluster_profile.loc[c_id, 'Gia_M2_Trung_vi']})")

print("\n[HOÀN THÀNH STEP 5A: K-MEANS CLUSTERING]")
print("="*65)
