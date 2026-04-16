"""
DAM501 – DATA MINING
Section 5A: K-Means Clustering (Unsupervised Learning) — TUNED VERSION

Cải tiến so với v1:
  - Mở rộng K_range từ 2–7 → 2–10
  - Thêm Davies-Bouldin Index (thấp hơn = tốt hơn)
  - Tự động chọn OPTIMAL_K từ Silhouette Score cao nhất
  - Tăng n_init=15 để ổn định kết quả
  - Thêm Radar Chart visualize profile từng cụm
  - Lưu model bằng joblib để tái sử dụng

Input:
  step3_minh/data/hanoi_apartments_for_clustering.csv   (72.604 × 8 scaled cols)
  step3_minh/data/hanoi_apartments_processed.csv        (72.604 × 37 cols)

Output:
  plots_section_5/kmeans_01_elbow_silhouette.png
  plots_section_5/kmeans_02_pca_clusters.png
  plots_section_5/kmeans_03_radar_profile.png
  models/kmeans_k{K}.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. ĐỊA CHỈ DỮ LIỆU & KHỞI TẠO
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "../step3_minh/data/hanoi_apartments_for_clustering.csv")
RAW_PATH   = os.path.join(BASE_DIR, "../step3_minh/data/hanoi_apartments_processed.csv")
PLOT_DIR   = os.path.join(BASE_DIR, "plots_section_5")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEP = "=" * 70
print(SEP)
print("PHẦN 5A: K-MEANS CLUSTERING (UNSUPERVISED) — TUNED VERSION")
print("Phân khúc thị trường chung cư Hà Nội bằng K-Means")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TẢI DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 1. TẢI DỮ LIỆU ---")
try:
    df_cluster = pd.read_csv(DATA_PATH)
    print(f"  [OK] Clustering dataset: {df_cluster.shape}")
except FileNotFoundError:
    print(f"  [LỖI] Không tìm thấy: {DATA_PATH}")
    exit(1)

try:
    df_raw = pd.read_csv(RAW_PATH)
    print(f"  [OK] Raw/processed dataset: {df_raw.shape}")
except FileNotFoundError:
    print(f"  [LỖI] Không tìm thấy: {RAW_PATH}")
    exit(1)

# Lấy ma trận scaled features
scaled_features = [c for c in df_cluster.columns if c.startswith('scaled_')]
X = df_cluster[scaled_features].values
print(f"  Số features scaled: {len(scaled_features)} → {scaled_features}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TÌM K TỐI ƯU — ELBOW + SILHOUETTE + DAVIES-BOULDIN
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 2. TÌM SỐ CỤM K TỐI ƯU (K = 2..10) ---")
print("   [Metrics] Inertia (↓) | Silhouette (↑) | Davies-Bouldin (↓)")

# Sample 10% để tính Silhouette & DB nhanh hơn (nhất quán)
np.random.seed(42)
sample_idx = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.10), replace=False)
X_sample = X[sample_idx]

K_range    = range(2, 11)
inertia    = []
sil_scores = []
db_scores  = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=15, init='k-means++')
    km.fit(X)
    labels_full   = km.labels_
    labels_sample = km.predict(X_sample)

    inertia.append(km.inertia_)

    sil = silhouette_score(X_sample, labels_sample)
    sil_scores.append(sil)

    db = davies_bouldin_score(X_sample, labels_sample)
    db_scores.append(db)

    print(f"  K = {k:2d} | Inertia = {km.inertia_:>12,.0f} | "
          f"Silhouette = {sil:.4f} | Davies-Bouldin = {db:.4f}")

# ── Tự động chọn K tối ưu ────────────────────────────────────────────────────
# Chiến lược kép:
#  1) Elbow Method: tìm K tại điểm gấp của Inertia (2nd derivative cực đại)
#  2) Silhouette: K có Silhouette cao nhất trong K ≤ MAX_BUSINESS_K
#  3) Nếu 2 phương pháp không đồng thuận và EDA Step 4 đã xác nhận K=3
#     thì ưu tiên K=3 (knowledge-driven override)
MAX_BUSINESS_K = 6
k_list_full    = list(K_range)

# Elbow: 2nd difference của Inertia → điểm cực đại là elbow
inertia_arr = np.array(inertia)
d1 = np.diff(inertia_arr)          # 1st difference
d2 = np.diff(d1)                    # 2nd difference (inflection point)
# K có giá trị 2nd diff lớn nhất (điểm gấp mạnh nhất)
elbow_idx = int(np.argmax(np.abs(d2))) + 1  # +1 vì mất 2 phần tử sau diff2
elbow_k   = k_list_full[elbow_idx]

# Silhouette tốt nhất trong [2, MAX_BUSINESS_K]
business_range = [k for k in k_list_full if k <= MAX_BUSINESS_K]
best_sil_idx   = int(np.argmax(sil_scores[:len(business_range)]))
best_sil_k     = business_range[best_sil_idx]

# EDA Step 4 đã xác nhận 3 Zone rõ ràng (Inner/Middle/Outer)
EDA_CONFIRMED_K = 3
eda_sil   = sil_scores[k_list_full.index(EDA_CONFIRMED_K)]
best_biz_sil = sil_scores[best_sil_idx]

# So sánh: nếu EDA K=3 không tệ hơn > 3% so với best Silhouette → ưu tiên K=3
silhouette_gap = (best_biz_sil - eda_sil) / best_biz_sil

if silhouette_gap <= 0.03:  # trong 3%
    OPTIMAL_K = EDA_CONFIRMED_K
    reason = f"EDA override (K={EDA_CONFIRMED_K} vs best Sil K={best_sil_k}, gap={silhouette_gap*100:.1f}%)"
elif elbow_k == best_sil_k:
    OPTIMAL_K = elbow_k
    reason = f"Elbow & Silhouette đồng thuận (K={elbow_k})"
else:
    OPTIMAL_K = elbow_k  # ưu tiên Elbow về mặt hình học
    reason = f"Elbow (K={elbow_k}) được chọn, Silhouette best K={best_sil_k}"

print(f"\n  → Elbow Method    : K = {elbow_k}")
print(f"  → Silhouette Best  : K = {best_sil_k} (score = {best_biz_sil:.4f})")
print(f"  → EDA Step 4 xác nhận: K = {EDA_CONFIRMED_K} (score = {eda_sil:.4f})")
print(f"  ✓ OPTIMAL_K = {OPTIMAL_K}  ({reason})")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Elbow + Silhouette + Davies-Bouldin
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Xác Định Số Cụm K Tối Ưu (K-Means Tuning)", fontsize=14, fontweight='bold')

k_list = list(K_range)

# Elbow
axes[0].plot(k_list, inertia, 'bo-', lw=2, markersize=7, label='Inertia')
axes[0].axvline(OPTIMAL_K, color='red', linestyle='--', alpha=0.7, label=f'Chọn K={OPTIMAL_K}')
axes[0].set_xlabel('Số Cụm K')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('Elbow Method')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Silhouette
axes[1].plot(k_list, sil_scores, 'gs-', lw=2, markersize=7, label='Silhouette Score')
axes[1].axvline(OPTIMAL_K, color='red', linestyle='--', alpha=0.7, label=f'Best K={OPTIMAL_K}')
axes[1].set_xlabel('Số Cụm K')
axes[1].set_ylabel('Silhouette Score (↑ tốt hơn)')
axes[1].set_title('Silhouette Score')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Davies-Bouldin
axes[2].plot(k_list, db_scores, 'r^-', lw=2, markersize=7, label='Davies-Bouldin Index')
axes[2].axvline(OPTIMAL_K, color='blue', linestyle='--', alpha=0.7, label=f'Best K={OPTIMAL_K}')
axes[2].set_xlabel('Số Cụm K')
axes[2].set_ylabel('Davies-Bouldin Index (↓ tốt hơn)')
axes[2].set_title('Davies-Bouldin Index')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "kmeans_01_elbow_silhouette.png"))
plt.close()
print(f"  → Đã lưu: plots_section_5/kmeans_01_elbow_silhouette.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIT MÔ HÌNH CHÍNH THỨC VỚI K = OPTIMAL_K
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- 3. HUẤN LUYỆN K-MEANS VỚI K = {OPTIMAL_K} (n_init=15, k-means++) ---")
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=15, init='k-means++')
df_cluster['Cluster'] = kmeans.fit_predict(X)
df_raw['Cluster'] = df_cluster['Cluster'].values
print(f"  Đã gán nhãn {OPTIMAL_K} cụm cho {len(df_cluster):,} căn hộ.")

# Lưu model
model_path = os.path.join(MODEL_DIR, f"kmeans_k{OPTIMAL_K}.pkl")
joblib.dump(kmeans, model_path)
print(f"  → Model đã lưu tại: models/kmeans_k{OPTIMAL_K}.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRỰC QUAN HÓA PCA 2D
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 4. TRỰC QUAN HÓA PCA 2D ---")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
df_cluster['PCA_1'] = X_pca[:, 0]
df_cluster['PCA_2'] = X_pca[:, 1]

explained = pca.explained_variance_ratio_.sum() * 100
print(f"  PCA giữ lại: {explained:.1f}% phương sai")

plt.figure(figsize=(10, 8))
palette = sns.color_palette("tab10", OPTIMAL_K)
sns.scatterplot(
    x='PCA_1', y='PCA_2',
    hue='Cluster',
    palette=palette,
    data=df_cluster.sample(min(10000, len(df_cluster)), random_state=42),
    alpha=0.5, s=15
)
plt.title(
    f'K-Means K={OPTIMAL_K} — PCA Scatter\n'
    f'({explained:.1f}% variance explained)',
    fontweight='bold', fontsize=13
)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "kmeans_02_pca_clusters.png"))
plt.close()
print(f"  → Đã lưu: plots_section_5/kmeans_02_pca_clusters.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. KHAI PHÁ PROFILE CỤM
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 5. KHAI PHÁ ĐẶC TRƯNG TỪNG CỤM ---")

cluster_profile = df_raw.groupby('Cluster').agg(
    So_luong    = ('price', 'count'),
    Gia_TB      = ('price', lambda x: x.median() / 1e9),
    Gia_M2_Med  = ('price_per_m2', lambda x: x.median() / 1e6),
    Dien_tich   = ('area', 'median'),
    PN_TB       = ('bedroom_count', 'mean'),
    WC_TB       = ('bathroom_count', 'mean'),
).round(2)
cluster_profile['Ty_le'] = (cluster_profile['So_luong'] / len(df_raw) * 100).round(1)

# Gán nhãn tự động theo Giá/m² median (tăng dần)
sorted_idx = cluster_profile['Gia_M2_Med'].sort_values().index.tolist()
business_labels = {
    0: "Phổ Thông / Ngoại Ô",
    1: "Tầm Trung Lõi Đô",
    2: "Cao Cấp (Premium)",
}
if OPTIMAL_K == 4:
    business_labels[3] = "Siêu Cao cấp / Penthouse"
elif OPTIMAL_K > 4:
    for i in range(3, OPTIMAL_K):
        business_labels[i] = f"Phân Khúc {i+1}"

print("\n[BẢNG PROFILE CỤM — KNOWLEDGE DISCOVERY]")
print(cluster_profile.to_string())

print("\n[NHÃN KINH DOANH TỰ ĐỘNG (theo Giá/m² Median)]")
cluster_names = {}
for rank, c_id in enumerate(sorted_idx):
    label = business_labels.get(rank, f"Phân khúc {rank+1}")
    cluster_names[c_id] = label
    print(f"  Cụm {c_id} → '{label}' | "
          f"Giá/m²: {cluster_profile.loc[c_id, 'Gia_M2_Med']:.1f} Tr/m² | "
          f"Thị phần: {cluster_profile.loc[c_id, 'Ty_le']:.1f}%")

# Cross-check Zone
if 'district_zone' in df_raw.columns:
    zone_cross = pd.crosstab(df_raw['Cluster'], df_raw['district_zone'], normalize='index') * 100
    print("\n[PHÂN BỐ ĐỊA LÝ TRONG TỪNG CỤM (%)]")
    print(zone_cross.round(1).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Radar Chart - Cluster Profile
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 6. RADAR CHART — PROFILE TỪNG CỤM ---")

radar_cols = ['Gia_TB', 'Gia_M2_Med', 'Dien_tich', 'PN_TB', 'WC_TB']
radar_labels = ['Giá TB\n(Tỷ)', 'Giá/m²\n(Tr)', 'Diện tích\n(m²)', 'Phòng ngủ', 'Phòng tắm']
N = len(radar_cols)

# Normalize 0-1 cho mỗi dimension
radar_data = cluster_profile[radar_cols].copy()
for col in radar_cols:
    col_min, col_max = radar_data[col].min(), radar_data[col].max()
    if col_max > col_min:
        radar_data[col] = (radar_data[col] - col_min) / (col_max - col_min)
    else:
        radar_data[col] = 0.5

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

colors_radar = plt.cm.tab10(np.linspace(0, 1, OPTIMAL_K))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, (c_id, row) in enumerate(radar_data.iterrows()):
    values = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, values, 'o-', lw=2, color=colors_radar[i],
            label=f"Cụm {c_id}: {cluster_names.get(c_id, '')}")
    ax.fill(angles, values, alpha=0.15, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
ax.set_title(f'Profile So sánh {OPTIMAL_K} Cụm K-Means\n(Normalized 0–1 theo mỗi chiều)',
             fontweight='bold', fontsize=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "kmeans_03_radar_profile.png"), bbox_inches='tight')
plt.close()
print(f"  → Đã lưu: plots_section_5/kmeans_03_radar_profile.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. KẾT QUẢ CUỐI
# ─────────────────────────────────────────────────────────────────────────────
opt_k_idx = list(K_range).index(OPTIMAL_K)
best_sil  = sil_scores[opt_k_idx]
best_db   = db_scores[opt_k_idx]
print(f"\n{'='*70}")
print(f"[KẾT QUẢ CUỐI — K-MEANS TUNED]")
print(f"  OPTIMAL K      : {OPTIMAL_K}")
print(f"  Silhouette     : {best_sil:.4f}")
print(f"  Davies-Bouldin : {best_db:.4f}")
print(f"  Model lưu tại  : models/kmeans_k{OPTIMAL_K}.pkl")
print(f"  Plots output   : plots_section_5/")
print(f"\n[HOÀN THÀNH STEP 5A: K-MEANS CLUSTERING — TUNED]")
print("=" * 70)
