"""
DAM501 – DATA MINING
Section 3: Data Pre-processing and Transformation (full pipeline)

Bao gồm:
  3a — Cleaning, Feature Engineering, Encoding, Scaling
  3b — Text Feature Extraction từ cột 'description'

Input:  data/hanoi_apartments_cleaned.csv
Output:
  data/hanoi_apartments_processed.csv        (EDA + LightGBM)
  data/hanoi_apartments_for_clustering.csv   (K-Means)
  plots/section_3/*.png
"""

import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/hanoi_apartments_cleaned.csv")
PLOT_DIR  = os.path.join(BASE_DIR, "plots/section_3")
os.makedirs(PLOT_DIR, exist_ok=True)

SEP = "=" * 65

# ─────────────────────────────────────────────────────────────────
# CONFIG: TEXT FEATURES (keyword matching từ cột description)
# ─────────────────────────────────────────────────────────────────
TEXT_FEATURES = [
    ("feat_full_furniture",  ["nội thất đầy đủ", "full nội thất", "full đồ",
                              "đầy đủ nội thất", "xách vali vào", "nhà đủ đồ",
                              "full nội thất cao cấp"]),
    ("feat_corner_unit",     ["căn góc"]),
    ("has_legal_paper",      ["sổ đỏ", "sổ hồng", "pháp lý đầy đủ", "pháp lý rõ ràng",
                              "pháp lý hoàn chỉnh", "pháp lý sạch"]),
    ("has_premium_amenities",["bể bơi", "hồ bơi", "swimming pool", "gym", "phòng tập", 
                              "fitness", "sân chơi", "khu vui chơi", "khu trẻ em", "playground"]),
    ("feat_near_school",     ["trường học", "gần trường", "trường tiểu học",
                              "trường thpt", "trường mầm non", "trường quốc tế"]),
    ("feat_near_hospital",   ["bệnh viện", "gần bệnh viện"]),
    ("feat_near_mall",       ["siêu thị", "trung tâm thương mại",
                              "vinmart", "aeon", "lotte", "shophouse"]),
    ("feat_near_park",       ["công viên"]),
    ("feat_balcony",         ["ban công"]),
]
FEAT_NAMES = [f[0] for f in TEXT_FEATURES]

LABEL_MAP = {
    "feat_full_furniture":   "Nội thất đầy đủ",
    "feat_corner_unit":      "Căn góc",
    "has_legal_paper":       "Pháp lý rõ ràng",
    "has_premium_amenities": "Tiện ích cao cấp",
    "feat_near_school":      "Gần trường học",
    "feat_near_hospital":    "Gần bệnh viện",
    "feat_near_mall":        "Gần siêu thị/TTTM",
    "feat_near_park":        "Gần công viên",
    "feat_balcony":          "Ban công",
}

# ─────────────────────────────────────────────────────────────────
# STEP 0 — LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────
print(SEP)
print("STEP 0 — LOAD RAW DATA")
print(SEP)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"Shape (raw): {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────────
# STEP 1 — MISSING VALUES ANALYSIS
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 1 — MISSING VALUES ANALYSIS")
print(SEP)

missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df  = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_df  = missing_df[missing_df["missing_count"] > 0].sort_values("missing_pct", ascending=False)
print(missing_df.to_string())

# ─────────────────────────────────────────────────────────────────
# STEP 2 — DROP COLUMNS WITH >= 95% MISSING
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 2 — DROP COLUMNS WITH >= 95% MISSING")
print(SEP)

drop_cols = missing_pct[missing_pct >= 95].index.tolist()
print(f"Columns to drop ({len(drop_cols)}): {drop_cols}")
df.drop(columns=drop_cols, inplace=True)
print(f"Shape after drop: {df.shape}")

# ─────────────────────────────────────────────────────────────────
# STEP 3a — EXTRACT DESCRIPTION BEFORE DROPPING IT
# (giữ lại description để dùng ở Step 3b, xóa sau)
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 3a — TÁCH DESCRIPTION ĐỂ DÙNG CHO TEXT FEATURES SAU")
print(SEP)

descriptions_raw = df["description"].copy()
print(f"Saved {descriptions_raw.notna().sum()} non-null descriptions for text extraction.")

# ─────────────────────────────────────────────────────────────────
# STEP 3b — DROP NON-ANALYTICAL COLUMNS
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 3b — DROP NON-ANALYTICAL COLUMNS")
print(SEP)

# province_name, property_type_name: zero variance
# name, description: free text — giá trị sẽ được trích xuất thành features ở Step 10
# house_direction: user requested to shift focus to balcony_direction instead
non_analytic = ["name", "description", "province_name", "property_type_name", "house_direction"]
existing = [c for c in non_analytic if c in df.columns]
df.drop(columns=existing, inplace=True)
print(f"Dropped: {existing}")
print(f"Shape after drop: {df.shape}")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 4 — HANDLE MISSING VALUES")
print(SEP)

# Sync descriptions_raw with df index trước khi drop
descriptions_raw = descriptions_raw.loc[df.index]

n_before = len(df)
keep_mask_price = df["price"].notna()
df = df[keep_mask_price].copy()
descriptions_raw = descriptions_raw[keep_mask_price].copy()
print(f"Dropped rows with missing price: {n_before - len(df)} rows removed")
print(f"Shape: {df.shape}")

for col in ["bedroom_count", "bathroom_count"]:
    med_by_dist = df.groupby("district_name")[col].transform("median")
    global_med  = df[col].median()
    df[col] = df[col].fillna(med_by_dist).fillna(global_med)
    print(f"  {col}: filled with district median (global fallback = {global_med:.1f})")

print(f"  area: 0% missing — no action needed")

for col in ["ward_name", "street_name", "project_name", "balcony_direction"]:
    if col in df.columns:
        n_fill = df[col].isnull().sum()
        df[col] = df[col].fillna("Unknown")
        print(f"  {col}: filled {n_fill} with 'Unknown'")

print(f"\nMissing after treatment:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — OUTLIER DETECTION & REMOVAL
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 5 — OUTLIER DETECTION & REMOVAL")
print(SEP)

def iqr_bounds(series, factor=3.0):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    return Q1 - factor * (Q3 - Q1), Q3 + factor * (Q3 - Q1)

lo_price, hi_price = iqr_bounds(df["price"])
lo_price = max(lo_price, 1e8)
hi_price = min(hi_price, 100e9)
print(f"Price — bounds: [{lo_price/1e9:.2f}B, {hi_price/1e9:.2f}B] VND  "
      f"(outliers: {((df['price']<lo_price)|(df['price']>hi_price)).sum()})")

lo_area, hi_area = iqr_bounds(df["area"])
lo_area = max(lo_area, 15)
hi_area = min(hi_area, 500)
print(f"Area  — bounds: [{lo_area:.1f}, {hi_area:.1f}] m²  "
      f"(outliers: {((df['area']<lo_area)|(df['area']>hi_area)).sum()})")

lo_bed, hi_bed = 1, 10
print(f"Bedroom — bounds: [{lo_bed}, {hi_bed}]  "
      f"(outliers: {((df['bedroom_count']<lo_bed)|(df['bedroom_count']>hi_bed)).sum()})")

n_before = len(df)
outlier_mask = (
    (df["price"] >= lo_price) & (df["price"] <= hi_price) &
    (df["area"]  >= lo_area)  & (df["area"]  <= hi_area)  &
    (df["bedroom_count"] >= lo_bed) & (df["bedroom_count"] <= hi_bed)
)
df               = df[outlier_mask].copy()
descriptions_raw = descriptions_raw[outlier_mask].copy()
print(f"\nTotal removed: {n_before - len(df)} rows ({(n_before-len(df))/n_before*100:.1f}%)")
print(f"Shape after outlier removal: {df.shape}")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 6 — FEATURE ENGINEERING")
print(SEP)

df["price_per_m2"] = df["price"] / df["area"]
print(f"Created: price_per_m2 — mean {df['price_per_m2'].mean()/1e6:.1f}M VND/m²")

df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
df["pub_month"]    = df["published_at"].dt.month
df["pub_year"]     = df["published_at"].dt.year
print(f"Created: pub_month, pub_year")

inner  = ["Hoàn Kiếm", "Ba Đình", "Đống Đa", "Hai Bà Trưng"]
middle = ["Tây Hồ", "Cầu Giấy", "Thanh Xuân", "Hoàng Mai",
          "Long Biên", "Bắc Từ Liêm", "Nam Từ Liêm", "Hà Đông"]

df["district_zone"] = df["district_name"].apply(
    lambda d: "inner" if d in inner else ("middle" if d in middle else "outer")
)
print(f"Created: district_zone — {df['district_zone'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────
# STEP 7 — LOG TRANSFORM
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 7 — LOG TRANSFORM")
print(SEP)

for col in ["price", "area", "price_per_m2"]:
    skew_before = df[col].skew()
    df[f"log_{col}"] = np.log1p(df[col])
    print(f"  {col}: skew {skew_before:.2f} → log_{col} skew {df[f'log_{col}'].skew():.2f}")

# ─────────────────────────────────────────────────────────────────
# STEP 8 — ENCODING CATEGORICAL VARIABLES
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 8 — ENCODING CATEGORICAL VARIABLES")
print(SEP)

le_district = LabelEncoder()
df["district_encoded"] = le_district.fit_transform(df["district_name"])
print(f"Label encoded: district_name → district_encoded ({len(le_district.classes_)} classes)")

le_zone = LabelEncoder()
df["zone_encoded"] = le_zone.fit_transform(df["district_zone"])
print(f"Label encoded: district_zone → zone_encoded ({len(le_zone.classes_)} classes)")

if "balcony_direction" in df.columns:
    df["balcony_direction"] = df["balcony_direction"].str.replace(" - ", " ")
    direction_dummies = pd.get_dummies(df["balcony_direction"], prefix="balcony_dir")
    
    # Xóa cột 'balcony_dir_Unknown' để tránh dư thừa (dummy variable trap)
    # Các hàng có 'Unknown' sẽ nhận giá trị 0 ở tất cả các cột hướng khác
    if "balcony_dir_Unknown" in direction_dummies.columns:
        direction_dummies.drop(columns=["balcony_dir_Unknown"], inplace=True)
        
    df = pd.concat([df, direction_dummies], axis=1)
    df.drop(columns=["balcony_direction"], inplace=True)
    print(f"One-hot encoded: balcony_direction → {direction_dummies.shape[1]} columns (dropped 'Unknown')")

# ─────────────────────────────────────────────────────────────────
# STEP 9 — STANDARDIZATION FOR K-MEANS
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 9 — STANDARDIZATION (for K-Means clustering)")
print(SEP)

cluster_features = ["log_price", "log_area", "bedroom_count", "bathroom_count",
                    "log_price_per_m2", "district_encoded"]
scaler   = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[cluster_features]),
    columns=[f"scaled_{c}" for c in cluster_features],
    index=df.index
)
print(f"Scaled features: {cluster_features}")
print(f"Shape of scaled matrix: {df_scaled.shape}")

# ─────────────────────────────────────────────────────────────────
# STEP 10 — TEXT FEATURE EXTRACTION FROM DESCRIPTION
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 10 — TEXT FEATURE EXTRACTION (keyword matching từ description)")
print(SEP)

descriptions = descriptions_raw.reset_index(drop=True).fillna("").str.lower()

text_results = {}
for feat_name, keywords in TEXT_FEATURES:
    pattern = "|".join(re.escape(kw) for kw in keywords)
    matches = descriptions.str.contains(pattern, regex=True, na=False).astype(int)
    text_results[feat_name] = matches.values
    print(f"  {feat_name:<28}  positive={matches.sum():>6}  ({matches.mean()*100:.1f}%)")

df_text = pd.DataFrame(text_results, index=df.index)
df_text["quality_score"] = df_text[FEAT_NAMES].sum(axis=1)

print(f"\nquality_score — mean: {df_text['quality_score'].mean():.2f}, "
      f"max: {df_text['quality_score'].max()}")

# Ghép vào df chính
df = pd.concat([df, df_text], axis=1)
print(f"Shape after text features: {df.shape}")

# ─────────────────────────────────────────────────────────────────
# STEP 11 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 11 — FINAL DATASET SUMMARY")
print(SEP)

print(f"Final shape: {df.shape}")
key_num = ["price", "area", "price_per_m2", "bedroom_count", "bathroom_count"]
print(f"\nNumeric summary:")
print(df[key_num].describe().map(lambda x: f"{x:,.1f}").to_string())
print(f"\ndistrict_zone: {df['district_zone'].value_counts().to_dict()}")
print(f"quality_score distribution:\n{df['quality_score'].value_counts().sort_index().to_string()}")

# ─────────────────────────────────────────────────────────────────
# STEP 12 — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 12 — SAVE OUTPUTS & MODELS FOR WEB APP")
print(SEP)

# Tạo thư mục app_models nếu chưa có
MODEL_EXPORT_DIR = os.path.join(os.path.dirname(BASE_DIR), "app_models")
os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)

# Lưu Encoders và Scaler
with open(os.path.join(MODEL_EXPORT_DIR, "le_district.pkl"), "wb") as f:
    pickle.dump(le_district, f)
with open(os.path.join(MODEL_EXPORT_DIR, "le_zone.pkl"), "wb") as f:
    pickle.dump(le_zone, f)
with open(os.path.join(MODEL_EXPORT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print(f"Saved encoders and scaler to: {MODEL_EXPORT_DIR}")

# processed (cho EDA + LightGBM): bỏ cột scaled_*
df_out = df.drop(columns=[c for c in df.columns if c.startswith("scaled_")])
out_processed = os.path.join(BASE_DIR, "data/hanoi_apartments_processed.csv")
df_out.to_csv(out_processed, index=False, encoding="utf-8-sig")
print(f"Saved: data/hanoi_apartments_processed.csv  {df_out.shape}")

# clustering (cho K-Means): scaled features + zone labels
df_cluster = pd.concat(
    [df_scaled.reset_index(drop=True),
     df[["district_name", "district_zone"]].reset_index(drop=True)],
    axis=1
)
out_cluster = os.path.join(BASE_DIR, "data/hanoi_apartments_for_clustering.csv")
df_cluster.to_csv(out_cluster, index=False, encoding="utf-8-sig")
print(f"Saved: data/hanoi_apartments_for_clustering.csv  {df_cluster.shape}")

# ─────────────────────────────────────────────────────────────────
# STEP 13 — PLOTS (3a: distributions + 3b: text feature analysis)
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 13 — GENERATE PLOTS")
print(SEP)

# --- Plot 1: Key distributions ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Section 3 — Data Pre-processing: Key Distributions", fontsize=14, fontweight="bold")

axes[0,0].hist(df["price"]/1e9, bins=60, color="#4C72B0", edgecolor="white", lw=0.4)
axes[0,0].set_title("Price Distribution (tỷ VND)")
axes[0,0].set_xlabel("Giá (tỷ VND)")

axes[0,1].hist(df["log_price"], bins=60, color="#55A868", edgecolor="white", lw=0.4)
axes[0,1].set_title("Log(Price) Distribution")
axes[0,1].set_xlabel("log(Price)")

axes[0,2].hist(df["area"], bins=60, color="#C44E52", edgecolor="white", lw=0.4)
axes[0,2].set_title("Area Distribution (m²)")
axes[0,2].set_xlabel("Diện tích (m²)")

zone_order = ["inner", "middle", "outer"]
zone_data  = [df[df["district_zone"]==z]["price_per_m2"]/1e6 for z in zone_order]
axes[1,0].boxplot(zone_data, labels=zone_order, patch_artist=True,
                  boxprops=dict(facecolor="#4C72B0", alpha=0.6))
axes[1,0].set_title("Giá/m² theo Zone (triệu VND/m²)")

bed_counts = df["bedroom_count"].value_counts().sort_index()
axes[1,1].bar(bed_counts.index.astype(int), bed_counts.values, color="#8172B2", edgecolor="white")
axes[1,1].set_title("Phân phối số phòng ngủ")
axes[1,1].set_xlabel("Số phòng ngủ")

dist_counts = df["district_name"].value_counts().head(12)
axes[1,2].barh(dist_counts.index[::-1], dist_counts.values[::-1], color="#64B5CD")
axes[1,2].set_title("Top 12 Quận theo số tin đăng")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/section3_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOT_DIR}/section3_distributions.png")

# --- Plot 2: Correlation heatmap ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
corr_cols = ["price", "area", "price_per_m2", "bedroom_count", "bathroom_count", "district_encoded"]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax2, annot_kws={"size": 9})
ax2.set_title("Correlation Matrix (after preprocessing)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/section3_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOT_DIR}/section3_correlation.png")

# --- Plot 3: Text feature price impact ---
diffs = []
for feat, label in LABEL_MAP.items():
    p_yes = df[df[feat]==1]["price"].mean() / 1e9
    p_no  = df[df[feat]==0]["price"].mean() / 1e9
    diff  = (p_yes - p_no) / p_no * 100
    diffs.append((label, diff))

diffs_sorted = sorted(diffs, key=lambda x: x[1], reverse=True)
labels = [d[0] for d in diffs_sorted]
pcts   = [d[1] for d in diffs_sorted]
colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pcts]

fig3, ax3 = plt.subplots(figsize=(10, 7))
ax3.barh(labels[::-1], pcts[::-1], color=colors[::-1], edgecolor="white", lw=0.5)
ax3.axvline(0, color="black", lw=0.8, linestyle="--")
ax3.set_xlabel("Chênh lệch giá trung bình so với căn không có feature (%)")
ax3.set_title("Tác động của Text Features đến Giá Căn Hộ", fontweight="bold")
for i, pct in enumerate(pcts[::-1]):
    ax3.text(pct + (0.3 if pct>=0 else -0.3), i,
             f"{pct:+.1f}%", va="center", ha="left" if pct>=0 else "right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/section3b_feature_price_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOT_DIR}/section3b_feature_price_impact.png")

# --- Plot 4: Text feature frequency ---
feat_pcts = [(LABEL_MAP[f], df[f].mean()*100) for f in FEAT_NAMES]
feat_pcts_sorted = sorted(feat_pcts, key=lambda x: x[1], reverse=True)

fig4, ax4 = plt.subplots(figsize=(10, 7))
ax4.barh([x[0] for x in feat_pcts_sorted[::-1]],
         [x[1] for x in feat_pcts_sorted[::-1]],
         color="#4C72B0", edgecolor="white", lw=0.5)
ax4.set_xlabel("Tỷ lệ tin đăng có mention feature (%)")
ax4.set_title(f"Tần suất xuất hiện của Text Features trong Mô Tả\n({len(df):,} tin đăng)", fontweight="bold")
for i, (_, pct) in enumerate(feat_pcts_sorted[::-1]):
    ax4.text(pct+0.3, i, f"{pct:.1f}%", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/section3b_feature_frequency.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOT_DIR}/section3b_feature_frequency.png")

# --- Plot 5: Quality score distribution ---
fig5, ax5 = plt.subplots(figsize=(9, 5))
score_dist = df["quality_score"].value_counts().sort_index()
ax5.bar(score_dist.index, score_dist.values, color="#8172B2", edgecolor="white")
ax5.set_xlabel("Quality Score (số features được đề cập)")
ax5.set_ylabel("Số lượng tin đăng")
ax5.set_title("Phân phối Quality Score", fontweight="bold")
for score, count in score_dist.items():
    ax5.text(score, count+50, f"{count:,}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/section3b_quality_score.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOT_DIR}/section3b_quality_score.png")

# ─────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 3 PREPROCESSING COMPLETE")
print(f"  hanoi_apartments_processed.csv      : {df_out.shape}")
print(f"  hanoi_apartments_for_clustering.csv : {df_cluster.shape}")
print(f"  Plots saved to: {PLOT_DIR}/")
print(SEP)
