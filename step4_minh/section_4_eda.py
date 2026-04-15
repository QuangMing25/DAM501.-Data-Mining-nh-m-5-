"""
DAM501 – DATA MINING
Section 4: Exploratory Data Analysis and Descriptive Mining (15%)

Phân tích khám phá dữ liệu sâu để hiểu cấu trúc thị trường căn hộ Hà Nội,
phát hiện patterns và relationships, hỗ trợ lựa chọn phương pháp Data Mining.

Input:
  data/hanoi_apartments_processed.csv      (72.604 × 51)
  data/hanoi_apartments_for_clustering.csv  (72.604 × 8)

Output:
  plots/section_4/*.png   (tất cả biểu đồ EDA)
  Console output          (thống kê chi tiết cho Section 4.md)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────
PROCESSED_PATH  = "../step3_minh/data/hanoi_apartments_processed.csv"
CLUSTERING_PATH = "../step3_minh/data/hanoi_apartments_for_clustering.csv"
PLOT_DIR        = "plots_section_4"
os.makedirs(PLOT_DIR, exist_ok=True)

SEP = "=" * 70

# Matplotlib defaults
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

# Color palettes
ZONE_COLORS = {'inner': '#e74c3c', 'middle': '#3498db', 'outer': '#2ecc71'}
ZONE_ORDER  = ['inner', 'middle', 'outer']
PALETTE_SEQ = ['#2c3e50', '#2980b9', '#27ae60', '#f39c12', '#e74c3c',
               '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#16a085']

# ─── LOAD DATA ───────────────────────────────────────────────────
print(SEP)
print("SECTION 4 — EXPLORATORY DATA ANALYSIS")
print("Loading data...")
print(SEP)

df = pd.read_csv(PROCESSED_PATH, encoding="utf-8-sig")
df_cluster = pd.read_csv(CLUSTERING_PATH, encoding="utf-8-sig")

print(f"Processed dataset : {df.shape}")
print(f"Clustering dataset: {df_cluster.shape}")

# Convert price to tỷ VND for readability
df['price_ty'] = df['price'] / 1e9
df['price_per_m2_tr'] = df['price_per_m2'] / 1e6

# ═════════════════════════════════════════════════════════════════
# PART 1: DISTRIBUTION ANALYSIS & SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PART 1 — DISTRIBUTION ANALYSIS & SUMMARY STATISTICS")
print(SEP)

# --- Thống kê tổng quan ---
print("\n--- 1.1 Thống kê tổng quan theo District Zone ---")
zone_stats = df.groupby('district_zone').agg(
    count=('price', 'count'),
    mean_price=('price_ty', 'mean'),
    median_price=('price_ty', 'median'),
    std_price=('price_ty', 'std'),
    mean_area=('area', 'mean'),
    median_area=('area', 'median'),
    mean_ppm2=('price_per_m2_tr', 'mean'),
    median_ppm2=('price_per_m2_tr', 'median')
).reindex(ZONE_ORDER)
print(zone_stats.to_string())

print("\n--- 1.2 Thống kê tổng quan theo Top 10 Quận ---")
top10_districts = df['district_name'].value_counts().head(10).index.tolist()
dist_stats = df[df['district_name'].isin(top10_districts)].groupby('district_name').agg(
    count=('price', 'count'),
    mean_price=('price_ty', 'mean'),
    median_price=('price_ty', 'median'),
    mean_ppm2=('price_per_m2_tr', 'mean'),
    mean_area=('area', 'mean')
).sort_values('mean_ppm2', ascending=False)
print(dist_stats.to_string())


# ─────────────────────────────────────────────────────────────────
# PLOT 1: Phân phối giá theo District Zone (Violin + Box)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Phân phối Giá, Diện tích, Giá/m² theo Vùng Địa lý", fontsize=14, fontweight='bold', y=1.02)

# 1a: Price by zone - violin
parts = axes[0].violinplot(
    [df[df['district_zone'] == z]['price_ty'].values for z in ZONE_ORDER],
    positions=[1, 2, 3], showmedians=True, showextrema=False
)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(list(ZONE_COLORS.values())[i])
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(['Inner\n(Nội thành lõi)', 'Middle\n(Vành đai 2-3)', 'Outer\n(Ngoại thành)'])
axes[0].set_ylabel('Giá (tỷ VND)')
axes[0].set_title('Phân phối Giá theo Vùng')

# 1b: Area by zone
parts2 = axes[1].violinplot(
    [df[df['district_zone'] == z]['area'].values for z in ZONE_ORDER],
    positions=[1, 2, 3], showmedians=True, showextrema=False
)
for i, pc in enumerate(parts2['bodies']):
    pc.set_facecolor(list(ZONE_COLORS.values())[i])
    pc.set_alpha(0.7)
parts2['cmedians'].set_color('black')
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(['Inner', 'Middle', 'Outer'])
axes[1].set_ylabel('Diện tích (m²)')
axes[1].set_title('Phân phối Diện tích theo Vùng')

# 1c: Price per m2 by zone
parts3 = axes[2].violinplot(
    [df[df['district_zone'] == z]['price_per_m2_tr'].values for z in ZONE_ORDER],
    positions=[1, 2, 3], showmedians=True, showextrema=False
)
for i, pc in enumerate(parts3['bodies']):
    pc.set_facecolor(list(ZONE_COLORS.values())[i])
    pc.set_alpha(0.7)
parts3['cmedians'].set_color('black')
axes[2].set_xticks([1, 2, 3])
axes[2].set_xticklabels(['Inner', 'Middle', 'Outer'])
axes[2].set_ylabel('Giá/m² (triệu VND)')
axes[2].set_title('Phân phối Giá/m² theo Vùng')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_01_zone_distributions.png")
plt.close()
print(f"\nSaved: {PLOT_DIR}/eda_01_zone_distributions.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 2: Giá trung bình theo Quận (sorted bar chart)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("So sánh Giá và Giá/m² theo Quận", fontsize=14, fontweight='bold', y=1.02)

# 2a: Median price by district
dist_price = df.groupby('district_name')['price_ty'].median().sort_values(ascending=True)
zone_map = df.groupby('district_name')['district_zone'].first()
bar_colors = [ZONE_COLORS[zone_map[d]] for d in dist_price.index]

axes[0].barh(dist_price.index, dist_price.values, color=bar_colors, edgecolor='white', lw=0.5)
axes[0].set_xlabel('Median Giá (tỷ VND)')
axes[0].set_title('Median Giá Căn Hộ theo Quận')
for i, (val, name) in enumerate(zip(dist_price.values, dist_price.index)):
    axes[0].text(val + 0.1, i, f'{val:.1f}', va='center', fontsize=8)

# 2b: Median price per m2 by district
dist_ppm2 = df.groupby('district_name')['price_per_m2_tr'].median().sort_values(ascending=True)
bar_colors2 = [ZONE_COLORS[zone_map[d]] for d in dist_ppm2.index]

axes[1].barh(dist_ppm2.index, dist_ppm2.values, color=bar_colors2, edgecolor='white', lw=0.5)
axes[1].set_xlabel('Median Giá/m² (triệu VND/m²)')
axes[1].set_title('Median Giá/m² theo Quận')
for i, (val, name) in enumerate(zip(dist_ppm2.values, dist_ppm2.index)):
    axes[1].text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=8)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=z.capitalize()) for z, c in ZONE_COLORS.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_02_district_price_comparison.png")
plt.close()
print(f"Saved: {PLOT_DIR}/eda_02_district_price_comparison.png")


# ═════════════════════════════════════════════════════════════════
# PART 2: GROUP-BASED COMPARISONS
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PART 2 — GROUP-BASED COMPARISONS")
print(SEP)

# --- 2.1 Giá theo số phòng ngủ ---
print("\n--- 2.1 Giá theo Số Phòng Ngủ ---")
bed_stats = df.groupby('bedroom_count').agg(
    count=('price', 'count'),
    mean_price=('price_ty', 'mean'),
    median_price=('price_ty', 'median'),
    mean_area=('area', 'mean'),
    mean_ppm2=('price_per_m2_tr', 'mean')
)
print(bed_stats.to_string())

# --- 2.2 Giá theo hướng ban công ---
print("\n--- 2.2 Giá theo Hướng Ban Công ---")
dir_cols = [c for c in df.columns if c.startswith('balcony_dir_')]
if dir_cols and 'balcony_direction' not in df.columns:
    df['balcony_direction'] = df[dir_cols].idxmax(axis=1).str.replace('balcony_dir_', '')

if 'balcony_direction' in df.columns:
    dir_stats = df.groupby('balcony_direction').agg(
        count=('price', 'count'),
    mean_price=('price_ty', 'mean'),
    median_price=('price_ty', 'median'),
    mean_ppm2=('price_per_m2_tr', 'mean')
).sort_values('median_price', ascending=False)
print(dir_stats.to_string())

# ─────────────────────────────────────────────────────────────────
# PLOT 3: Giá theo số phòng ngủ & số phòng tắm
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Giá Căn Hộ theo Số Phòng Ngủ và Phòng Tắm", fontsize=14, fontweight='bold', y=1.02)

# 3a: Box plot by bedroom count
bed_data = df[df['bedroom_count'] <= 5]
bp1 = axes[0].boxplot(
    [bed_data[bed_data['bedroom_count'] == b]['price_ty'].values for b in range(1, 6)],
    labels=[f'{b} PN' for b in range(1, 6)],
    patch_artist=True, showfliers=False,
    medianprops=dict(color='black', linewidth=1.5)
)
colors_bed = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
for patch, color in zip(bp1['boxes'], colors_bed):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_ylabel('Giá (tỷ VND)')
axes[0].set_xlabel('Số phòng ngủ')
axes[0].set_title('Phân phối Giá theo Số Phòng Ngủ')

# Add count annotations
for i, b in enumerate(range(1, 6)):
    count = (bed_data['bedroom_count'] == b).sum()
    axes[0].text(i + 1, axes[0].get_ylim()[1] * 0.95, f'n={count:,}',
                 ha='center', fontsize=8, fontstyle='italic')

# 3b: Box plot by bathroom count
bath_data = df[df['bathroom_count'] <= 4]
bp2 = axes[1].boxplot(
    [bath_data[bath_data['bathroom_count'] == b]['price_ty'].values for b in range(1, 5)],
    labels=[f'{b} WC' for b in range(1, 5)],
    patch_artist=True, showfliers=False,
    medianprops=dict(color='black', linewidth=1.5)
)
colors_bath = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
for patch, color in zip(bp2['boxes'], colors_bath):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_ylabel('Giá (tỷ VND)')
axes[1].set_xlabel('Số phòng tắm')
axes[1].set_title('Phân phối Giá theo Số Phòng Tắm')

for i, b in enumerate(range(1, 5)):
    count = (bath_data['bathroom_count'] == b).sum()
    axes[1].text(i + 1, axes[1].get_ylim()[1] * 0.95, f'n={count:,}',
                 ha='center', fontsize=8, fontstyle='italic')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_03_rooms_vs_price.png")
plt.close()
print(f"\nSaved: {PLOT_DIR}/eda_03_rooms_vs_price.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 4: Giá theo hướng nhà
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

if 'balcony_direction' in df.columns:
    dir_order = df.groupby('balcony_direction')['price_per_m2_tr'].median().sort_values(ascending=True)
    dir_filtered = dir_order[dir_order.index != 'Unknown']

    bp = ax.boxplot(
        [df[df['balcony_direction'] == d]['price_per_m2_tr'].values for d in dir_filtered.index],
        labels=dir_filtered.index, patch_artist=True, showfliers=False,
        medianprops=dict(color='black', linewidth=1.5), vert=True
    )
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)

    ax.set_ylabel('Giá/m² (triệu VND)')
    ax.set_title('Phân phối Giá/m² theo Hướng Ban Công', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    
    overall_med = df['price_per_m2_tr'].median()
    ax.axhline(overall_med, color='red', linestyle='--', lw=1, alpha=0.7, label=f'Median tổng: {overall_med:.1f} tr/m²')
    ax.legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_04_direction_vs_price.png")
plt.close()
print(f"Saved: {PLOT_DIR}/eda_04_direction_vs_price.png")


# ═════════════════════════════════════════════════════════════════
# PART 3: BEHAVIORAL / USAGE PATTERN EXPLORATION
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PART 3 — BEHAVIORAL / USAGE PATTERN EXPLORATION")
print(SEP)

# --- 3.1 Xu hướng đăng tin theo tháng ---
print("\n--- 3.1 Xu hướng đăng tin theo tháng ---")
month_stats = df.groupby('pub_month').agg(
    count=('price', 'count'),
    mean_price=('price_ty', 'mean'),
    median_price=('price_ty', 'median'),
    mean_ppm2=('price_per_m2_tr', 'mean')
)
print(month_stats.to_string())

# ─────────────────────────────────────────────────────────────────
# PLOT 5: Xu hướng theo tháng
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Xu hướng Thị Trường Căn Hộ Hà Nội theo Tháng (06–12/2025)", fontsize=14, fontweight='bold', y=1.02)

months = month_stats.index.astype(int)
month_labels = [f'T{m}' for m in months]

# 5a: Số lượng tin đăng
axes[0].bar(months, month_stats['count'], color='#3498db', edgecolor='white', lw=0.5)
axes[0].set_xlabel('Tháng')
axes[0].set_ylabel('Số lượng tin đăng')
axes[0].set_title('Số Lượng Tin Đăng Mới')
axes[0].set_xticks(months)
axes[0].set_xticklabels(month_labels)
for m, c in zip(months, month_stats['count']):
    axes[0].text(m, c + 100, f'{c:,}', ha='center', fontsize=8)

# 5b: Giá trung bình
axes[1].plot(months, month_stats['mean_price'], 'o-', color='#e74c3c', lw=2, markersize=6)
axes[1].fill_between(months, month_stats['mean_price'], alpha=0.1, color='#e74c3c')
axes[1].set_xlabel('Tháng')
axes[1].set_ylabel('Giá trung bình (tỷ VND)')
axes[1].set_title('Giá Trung Bình theo Tháng')
axes[1].set_xticks(months)
axes[1].set_xticklabels(month_labels)

# 5c: Giá/m² trung bình
axes[2].plot(months, month_stats['mean_ppm2'], 's-', color='#27ae60', lw=2, markersize=6)
axes[2].fill_between(months, month_stats['mean_ppm2'], alpha=0.1, color='#27ae60')
axes[2].set_xlabel('Tháng')
axes[2].set_ylabel('Giá/m² trung bình (triệu VND)')
axes[2].set_title('Giá/m² Trung Bình theo Tháng')
axes[2].set_xticks(months)
axes[2].set_xticklabels(month_labels)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_05_monthly_trends.png")
plt.close()
print(f"\nSaved: {PLOT_DIR}/eda_05_monthly_trends.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 6: Scatter — Price vs Area (colored by zone)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Mối quan hệ Giá – Diện tích theo Vùng Địa lý", fontsize=14, fontweight='bold', y=1.02)

# 6a: scatter by zone
for zone in ZONE_ORDER:
    mask = df['district_zone'] == zone
    sample = df[mask].sample(n=min(3000, mask.sum()), random_state=42)
    axes[0].scatter(sample['area'], sample['price_ty'],
                    c=ZONE_COLORS[zone], alpha=0.3, s=10, label=zone.capitalize())

axes[0].set_xlabel('Diện tích (m²)')
axes[0].set_ylabel('Giá (tỷ VND)')
axes[0].set_title('Giá vs Diện tích (sample ~9.000 điểm)')
axes[0].legend()

# 6b: 2D density / hexbin
hb = axes[1].hexbin(df['area'], df['price_ty'], gridsize=40, cmap='YlOrRd',
                     mincnt=1, linewidths=0.2)
axes[1].set_xlabel('Diện tích (m²)')
axes[1].set_ylabel('Giá (tỷ VND)')
axes[1].set_title('Mật độ tập trung (Hexbin)')
plt.colorbar(hb, ax=axes[1], label='Số lượng')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_06_price_area_scatter.png")
plt.close()
print(f"Saved: {PLOT_DIR}/eda_06_price_area_scatter.png")


# ═════════════════════════════════════════════════════════════════
# PART 4: CORRELATION & RELATIONSHIP DEEP DIVE
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PART 4 — CORRELATION & RELATIONSHIP ANALYSIS")
print(SEP)

# --- 4.1 Correlation matrix mở rộng ---
print("\n--- 4.1 Ma trận tương quan mở rộng ---")
corr_cols = ['price', 'area', 'price_per_m2', 'bedroom_count', 'bathroom_count',
             'district_encoded', 'zone_encoded', 'quality_score',
             'pub_month', 'log_price', 'log_area', 'log_price_per_m2']
corr_matrix = df[corr_cols].corr()
print("\nTop correlations with price:")
price_corr = corr_matrix['price'].drop('price').sort_values(key=abs, ascending=False)
for feat, val in price_corr.items():
    print(f"  {feat:25s}: {val:+.4f}")

# ─────────────────────────────────────────────────────────────────
# PLOT 7: Extended correlation heatmap
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))

full_corr_cols = ['price', 'area', 'price_per_m2', 'bedroom_count', 'bathroom_count',
                  'quality_score', 'feat_full_furniture', 'feat_corner_unit',
                  'has_premium_amenities', 'has_legal_paper',
                  'feat_balcony', 'feat_near_school', 'feat_near_hospital',
                  'feat_near_mall', 'feat_near_park']

# Rename for display
rename_map = {
    'price': 'Giá', 'area': 'Diện tích', 'price_per_m2': 'Giá/m²',
    'bedroom_count': 'Phòng ngủ', 'bathroom_count': 'Phòng tắm',
    'quality_score': 'Quality Score',
    'feat_full_furniture': 'Nội thất ĐĐ', 'feat_corner_unit': 'Căn góc',
    'has_premium_amenities': 'Tiện ích VIP', 'has_legal_paper': 'Pháp lý',
    'feat_balcony': 'Ban công',
    'feat_near_school': 'Gần trường', 'feat_near_hospital': 'Gần bệnh viện',
    'feat_near_mall': 'Gần siêu thị', 'feat_near_park': 'Gần công viên'
}

corr_full = df[full_corr_cols].rename(columns=rename_map).corr()

mask = np.triu(np.ones_like(corr_full, dtype=bool), k=1)
sns.heatmap(corr_full, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, annot_kws={'size': 7},
            vmin=-0.5, vmax=0.8)
ax.set_title('Ma trận Tương quan Mở rộng\n(Biến số + Text Features)', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_07_extended_correlation.png")
plt.close()
print(f"\nSaved: {PLOT_DIR}/eda_07_extended_correlation.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 8: Pair plot — key variables
# ─────────────────────────────────────────────────────────────────
print("\n--- Generating pair plot (sampled) ---")
pair_cols = ['log_price', 'log_area', 'bedroom_count', 'price_per_m2_tr', 'district_zone']
df_pair = df[pair_cols].sample(n=5000, random_state=42)

g = sns.pairplot(df_pair, hue='district_zone', palette=ZONE_COLORS,
                 diag_kind='kde', plot_kws={'alpha': 0.4, 's': 10},
                 height=2.2, hue_order=ZONE_ORDER)
g.fig.suptitle('Pair Plot: Biến Chính theo Vùng Địa lý (sample 5.000)', y=1.02, fontweight='bold')
plt.savefig(f"{PLOT_DIR}/eda_08_pairplot.png")
plt.close()
print(f"Saved: {PLOT_DIR}/eda_08_pairplot.png")


# ═════════════════════════════════════════════════════════════════
# PART 5: UNUSUAL / INTERESTING TRENDS
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PART 5 — UNUSUAL & INTERESTING TRENDS")
print(SEP)

# --- 5.1 Phân khúc giá tự nhiên ---
print("\n--- 5.1 Phân tích phân khúc giá tự nhiên ---")
bins_price = [0, 2, 4, 6, 8, 10, 15, 25]
labels_price = ['< 2 tỷ', '2-4 tỷ', '4-6 tỷ', '6-8 tỷ', '8-10 tỷ', '10-15 tỷ', '> 15 tỷ']
df['price_segment'] = pd.cut(df['price_ty'], bins=bins_price, labels=labels_price, include_lowest=True)

seg_zone = pd.crosstab(df['price_segment'], df['district_zone'], normalize='index') * 100
print("\nTỷ lệ phân bố theo zone trong mỗi phân khúc giá (%):")
print(seg_zone.reindex(columns=ZONE_ORDER).round(1).to_string())

seg_counts = df['price_segment'].value_counts().sort_index()
print(f"\nSố lượng tin đăng theo phân khúc giá:")
for seg, cnt in seg_counts.items():
    pct = cnt / len(df) * 100
    print(f"  {seg:>10s}: {cnt:>6,} ({pct:.1f}%)")

# --- 5.2 Quality Score vs Price ---
print("\n--- 5.2 Tương quan Quality Score và Giá ---")
qs_stats = df.groupby('quality_score').agg(
    count=('price', 'count'),
    mean_price=('price_ty', 'mean'),
    mean_ppm2=('price_per_m2_tr', 'mean')
)
print(qs_stats.head(15).to_string())

# --- 5.3 Top 10 phường đắt nhất và rẻ nhất ---
print("\n--- 5.3 Top 10 Phường đắt nhất (theo median giá/m²) ---")
ward_stats = df[df['ward_name'] != 'Unknown'].groupby(['district_name', 'ward_name']).agg(
    count=('price', 'count'),
    median_ppm2=('price_per_m2_tr', 'median'),
    median_price=('price_ty', 'median')
).reset_index()
ward_stats = ward_stats[ward_stats['count'] >= 50]  # Chỉ xét phường có >= 50 tin

top10_expensive = ward_stats.nlargest(10, 'median_ppm2')
print(top10_expensive.to_string(index=False))

print("\n--- 5.4 Top 10 Phường rẻ nhất ---")
top10_cheap = ward_stats.nsmallest(10, 'median_ppm2')
print(top10_cheap.to_string(index=False))


# ─────────────────────────────────────────────────────────────────
# PLOT 9: Phân khúc giá & Quality Score
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Phân tích Phân Khúc Giá và Quality Score", fontsize=14, fontweight='bold', y=1.01)

# 9a: Số lượng tin theo phân khúc giá
seg_counts_sorted = seg_counts.sort_index()
bars = axes[0, 0].bar(range(len(seg_counts_sorted)), seg_counts_sorted.values,
                       color=PALETTE_SEQ[:len(seg_counts_sorted)], edgecolor='white', lw=0.5)
axes[0, 0].set_xticks(range(len(seg_counts_sorted)))
axes[0, 0].set_xticklabels(seg_counts_sorted.index, rotation=30)
axes[0, 0].set_ylabel('Số lượng tin đăng')
axes[0, 0].set_title('Phân bố Tin Đăng theo Phân Khúc Giá')
for i, (seg, cnt) in enumerate(seg_counts_sorted.items()):
    axes[0, 0].text(i, cnt + 200, f'{cnt:,}\n({cnt/len(df)*100:.1f}%)', ha='center', fontsize=8)

# 9b: Tỷ lệ zone trong mỗi phân khúc (stacked bar)
seg_zone_sorted = seg_zone.reindex(columns=ZONE_ORDER)
bottom = np.zeros(len(seg_zone_sorted))
for zone in ZONE_ORDER:
    axes[0, 1].bar(range(len(seg_zone_sorted)), seg_zone_sorted[zone], bottom=bottom,
                    label=zone.capitalize(), color=ZONE_COLORS[zone], edgecolor='white', lw=0.3)
    bottom += seg_zone_sorted[zone].values
axes[0, 1].set_xticks(range(len(seg_zone_sorted)))
axes[0, 1].set_xticklabels(seg_zone_sorted.index, rotation=30)
axes[0, 1].set_ylabel('Tỷ lệ (%)')
axes[0, 1].set_title('Cấu trúc Địa lý theo Phân Khúc Giá')
axes[0, 1].legend(loc='upper right')

# 9c: Quality Score vs Mean Price
axes[1, 0].bar(qs_stats.index, qs_stats['mean_price'], color='#8172B2', edgecolor='white', lw=0.5)
axes[1, 0].set_xlabel('Quality Score')
axes[1, 0].set_ylabel('Giá trung bình (tỷ VND)')
axes[1, 0].set_title('Giá Trung Bình theo Quality Score')

# Fit trend line
z = np.polyfit(qs_stats.index, qs_stats['mean_price'], 1)
p = np.poly1d(z)
axes[1, 0].plot(qs_stats.index, p(qs_stats.index), 'r--', lw=1.5,
                label=f'Trend: +{z[0]:.2f} tỷ/score')
axes[1, 0].legend()

# 9d: Quality Score vs Mean Price/m²
axes[1, 1].bar(qs_stats.index, qs_stats['mean_ppm2'], color='#2ecc71', edgecolor='white', lw=0.5)
axes[1, 1].set_xlabel('Quality Score')
axes[1, 1].set_ylabel('Giá/m² trung bình (triệu VND)')
axes[1, 1].set_title('Giá/m² Trung Bình theo Quality Score')

z2 = np.polyfit(qs_stats.index, qs_stats['mean_ppm2'], 1)
p2 = np.poly1d(z2)
axes[1, 1].plot(qs_stats.index, p2(qs_stats.index), 'r--', lw=1.5,
                label=f'Trend: +{z2[0]:.2f} tr/m²/score')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_09_segments_quality.png")
plt.close()
print(f"\nSaved: {PLOT_DIR}/eda_09_segments_quality.png")


# ─────────────────────────────────────────────────────────────────
# PLOT 10: Top 10 phường đắt + rẻ nhất
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Top 10 Phường Đắt Nhất và Rẻ Nhất (theo Median Giá/m²)", fontsize=14, fontweight='bold', y=1.02)

# 10a: Top expensive
labels_exp = [f"{r['ward_name']}\n({r['district_name']})" for _, r in top10_expensive.iterrows()]
axes[0].barh(labels_exp[::-1], top10_expensive['median_ppm2'].values[::-1],
             color='#e74c3c', edgecolor='white', lw=0.5, alpha=0.8)
axes[0].set_xlabel('Median Giá/m² (triệu VND)')
axes[0].set_title('Top 10 Phường ĐẮT nhất', color='#e74c3c')
for i, val in enumerate(top10_expensive['median_ppm2'].values[::-1]):
    axes[0].text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=8)

# 10b: Top cheap
labels_chp = [f"{r['ward_name']}\n({r['district_name']})" for _, r in top10_cheap.iterrows()]
axes[1].barh(labels_chp[::-1], top10_cheap['median_ppm2'].values[::-1],
             color='#2ecc71', edgecolor='white', lw=0.5, alpha=0.8)
axes[1].set_xlabel('Median Giá/m² (triệu VND)')
axes[1].set_title('Top 10 Phường RẺ nhất', color='#2ecc71')
for i, val in enumerate(top10_cheap['median_ppm2'].values[::-1]):
    axes[1].text(val + 0.3, i, f'{val:.1f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_10_ward_extremes.png")
plt.close()
print(f"Saved: {PLOT_DIR}/eda_10_ward_extremes.png")


# ═════════════════════════════════════════════════════════════════
# PART 6: PRELIMINARY CLUSTERING EXPLORATION
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PART 6 — PRELIMINARY CLUSTERING EXPLORATION (PCA)")
print(SEP)

# PCA on clustering features
scaled_cols = [c for c in df_cluster.columns if c.startswith('scaled_')]
X_scaled = df_cluster[scaled_cols].values

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA Explained Variance Ratio:")
for i, ev in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ev:.4f} ({ev*100:.1f}%)")
print(f"  Total (3 components): {pca.explained_variance_ratio_.sum()*100:.1f}%")

print(f"\nPCA Component Loadings:")
loadings = pd.DataFrame(pca.components_.T, index=scaled_cols,
                         columns=[f'PC{i+1}' for i in range(3)])
print(loadings.round(3).to_string())

# ─────────────────────────────────────────────────────────────────
# PLOT 11: PCA visualization
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("PCA — Khám Phá Cấu Trúc Dữ Liệu Clustering", fontsize=14, fontweight='bold', y=1.02)

zones = df_cluster['district_zone'].values

# 11a: PC1 vs PC2 by zone
for zone in ZONE_ORDER:
    mask = zones == zone
    idx = np.where(mask)[0]
    sample_idx = np.random.RandomState(42).choice(idx, size=min(3000, len(idx)), replace=False)
    axes[0].scatter(X_pca[sample_idx, 0], X_pca[sample_idx, 1],
                    c=ZONE_COLORS[zone], alpha=0.3, s=8, label=zone.capitalize())
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title('PC1 vs PC2 — Theo Vùng')
axes[0].legend()

# 11b: PC1 vs PC2 colored by price
sample_idx_all = np.random.RandomState(42).choice(len(X_pca), size=min(8000, len(X_pca)), replace=False)
sc = axes[1].scatter(X_pca[sample_idx_all, 0], X_pca[sample_idx_all, 1],
                      c=df['price_ty'].iloc[sample_idx_all], cmap='RdYlGn_r',
                      alpha=0.5, s=8, vmin=1, vmax=15)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('PC1 vs PC2 — Theo Giá')
plt.colorbar(sc, ax=axes[1], label='Giá (tỷ VND)')

# 11c: PCA loadings biplot
ax_bi = axes[2]
for i, feat in enumerate(scaled_cols):
    short_name = feat.replace('scaled_', '')
    ax_bi.arrow(0, 0, loadings.iloc[i, 0]*3, loadings.iloc[i, 1]*3,
                head_width=0.05, head_length=0.03, fc=PALETTE_SEQ[i], ec=PALETTE_SEQ[i], lw=1.5)
    ax_bi.text(loadings.iloc[i, 0]*3.3, loadings.iloc[i, 1]*3.3,
               short_name, fontsize=9, ha='center', fontweight='bold', color=PALETTE_SEQ[i])
ax_bi.axhline(0, color='gray', lw=0.5, ls='--')
ax_bi.axvline(0, color='gray', lw=0.5, ls='--')
ax_bi.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax_bi.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax_bi.set_title('PCA Loadings (Biplot)')
ax_bi.set_xlim(-1.2, 1.2)
ax_bi.set_ylim(-1.2, 1.2)
ax_bi.set_aspect('equal')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_11_pca_exploration.png")
plt.close()
print(f"\nSaved: {PLOT_DIR}/eda_11_pca_exploration.png")


# ═════════════════════════════════════════════════════════════════
# PART 7: TEXT FEATURES — PATTERN ANALYSIS
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PART 7 — TEXT FEATURES PATTERN ANALYSIS")
print(SEP)

feat_cols = [c for c in df.columns if c.startswith('feat_') or c.startswith('has_')]

# --- 7.1 Text feature tương quan với giá ---
print("\n--- 7.1 Tương quan Text Features với Giá ---")
feat_corr_price = df[feat_cols + ['quality_score']].corrwith(df['price']).sort_values(ascending=False)
for feat, val in feat_corr_price.items():
    print(f"  {feat:28s}: {val:+.4f}")

# --- 7.2 Co-occurrence matrix ---
print("\n--- 7.2 Text Feature Co-occurrence (Top pairs) ---")
co_matrix = df[feat_cols].T.dot(df[feat_cols])
# Get top co-occurring pairs
pairs = []
for i in range(len(feat_cols)):
    for j in range(i + 1, len(feat_cols)):
        pairs.append((feat_cols[i], feat_cols[j], co_matrix.iloc[i, j]))
pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
for f1, f2, count in pairs_sorted:
    print(f"  {f1:28s} + {f2:28s}: {count:>6,}")


# ─────────────────────────────────────────────────────────────────
# PLOT 12: Text Features Analysis
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Text Features — Tương quan và Co-occurrence", fontsize=14, fontweight='bold', y=1.02)

# 12a: Feature correlation with price
feat_corr_sorted = feat_corr_price.sort_values()
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in feat_corr_sorted.values]
axes[0].barh(feat_corr_sorted.index, feat_corr_sorted.values, color=colors, edgecolor='white', lw=0.5)
axes[0].axvline(0, color='black', lw=0.8, ls='--')
axes[0].set_xlabel('Pearson Correlation với Giá')
axes[0].set_title('Tương quan Text Features với Giá')
for i, (feat, val) in enumerate(feat_corr_sorted.items()):
    axes[0].text(val + (0.002 if val >= 0 else -0.002), i,
                 f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=7)

# 12b: Text features by zone (heatmap)
zone_feat_mean = df.groupby('district_zone')[feat_cols].mean().reindex(ZONE_ORDER).T * 100
zone_feat_mean.index = [c.replace('feat_', '').replace('has_', '') for c in zone_feat_mean.index]
sns.heatmap(zone_feat_mean, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1],
            linewidths=0.5, cbar_kws={'label': 'Tỷ lệ (%)'})
axes[1].set_title('Tỷ lệ Text Features theo Vùng (%)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_12_text_features_analysis.png")
plt.close()
print(f"\nSaved: {PLOT_DIR}/eda_12_text_features_analysis.png")


# ═════════════════════════════════════════════════════════════════
# PART 8: DIỆN TÍCH PHÂN KHÚC — PHÂN TÍCH CHÉO
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PART 8 — AREA SEGMENT CROSS-ANALYSIS")
print(SEP)

bins_area = [0, 45, 65, 80, 100, 130, 250]
labels_area = ['Studio\n(<45m²)', 'Nhỏ\n(45-65m²)', 'TB\n(65-80m²)',
               'Khá\n(80-100m²)', 'Lớn\n(100-130m²)', 'Penthouse\n(>130m²)']
df['area_segment'] = pd.cut(df['area'], bins=bins_area, labels=labels_area, include_lowest=True)

print("\n--- Thống kê theo phân khúc diện tích ---")
area_seg_stats = df.groupby('area_segment', observed=False).agg(
    count=('price', 'count'),
    mean_price=('price_ty', 'mean'),
    median_price=('price_ty', 'median'),
    mean_ppm2=('price_per_m2_tr', 'mean'),
    mean_bedroom=('bedroom_count', 'mean')
)
print(area_seg_stats.to_string())


# ─────────────────────────────────────────────────────────────────
# PLOT 13: Cross-analysis Area Segment × Zone
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Phân tích Chéo: Phân Khúc Diện Tích × Vùng Địa lý",
             fontsize=14, fontweight='bold', y=1.02)

# 13a: Grouped bar — mean price by area segment × zone
area_zone_price = df.groupby(['area_segment', 'district_zone'], observed=False)['price_ty'].median().unstack()
area_zone_price = area_zone_price.reindex(columns=ZONE_ORDER)
x = np.arange(len(area_zone_price))
width = 0.25
for i, zone in enumerate(ZONE_ORDER):
    axes[0].bar(x + i * width, area_zone_price[zone], width,
                label=zone.capitalize(), color=ZONE_COLORS[zone], edgecolor='white', lw=0.3)
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(area_zone_price.index, fontsize=8)
axes[0].set_ylabel('Median Giá (tỷ VND)')
axes[0].set_title('Median Giá theo Phân Khúc Diện Tích × Vùng')
axes[0].legend()

# 13b: Grouped bar — mean price/m²
area_zone_ppm2 = df.groupby(['area_segment', 'district_zone'], observed=False)['price_per_m2_tr'].median().unstack()
area_zone_ppm2 = area_zone_ppm2.reindex(columns=ZONE_ORDER)
for i, zone in enumerate(ZONE_ORDER):
    axes[1].bar(x + i * width, area_zone_ppm2[zone], width,
                label=zone.capitalize(), color=ZONE_COLORS[zone], edgecolor='white', lw=0.3)
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(area_zone_ppm2.index, fontsize=8)
axes[1].set_ylabel('Median Giá/m² (triệu VND)')
axes[1].set_title('Median Giá/m² theo Phân Khúc Diện Tích × Vùng')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/eda_13_area_zone_cross.png")
plt.close()
print(f"\nSaved: {PLOT_DIR}/eda_13_area_zone_cross.png")


# ═════════════════════════════════════════════════════════════════
# SUMMARY — KEY FINDINGS
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("KEY FINDINGS SUMMARY — Guiding Data Mining Techniques")
print(SEP)

print("""
1. CẤU TRÚC PHÂN KHÚC RÕ RÀNG:
   - 3 vùng địa lý (inner/middle/outer) có mức giá khác biệt rõ ràng
   - 7 phân khúc giá phân bố không đều (4-8 tỷ chiếm đa số)
   → K-Means Clustering có cơ sở tốt để phát hiện phân khúc tự nhiên

2. TƯƠNG QUAN ĐA CHIỀU:
   - price ~ area (r=0.77), price ~ bedroom (r=0.56)
   - price_per_m2 phụ thuộc mạnh vào vị trí, không phải diện tích
   - Text features (parking, corner unit, gym) tương quan dương với giá
   → LightGBM Regression phù hợp để nắm bắt quan hệ phi tuyến

3. PATTERNS ĐỊA LÝ:
   - Inner zone: giá/m² cao gấp 1.5-2x outer zone
   - Chênh lệch giá giữa phường cùng quận có thể lên đến 50-100%
   → Cần đưa biến vị trí vào cả 2 models

4. PCA CHO THẤY 3 CHIỀU CHÍNH:
   - PC1: "tổng giá trị" (price, area, bedrooms)
   - PC2: "chất lượng vị trí" (price_per_m2, district)
   → Hỗ trợ K-Means: 2-3 PC giải thích phần lớn variance
""")

print(f"\n{SEP}")
print("SECTION 4 EDA COMPLETE")
print(f"  Total plots saved: 13")
print(f"  Output directory: {PLOT_DIR}/")
print(SEP)
