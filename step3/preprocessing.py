import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load Data
df = pd.read_csv('hanoi_apartments2.csv', sep=';')

# Focus vào các cột quan trọng
cols = ['district_name', 'area', 'price', 'bedroom_count', 'bathroom_count']
df = df[cols].copy()

# ========================================================
# BƯỚC 3: DATA CLEANING & NOISE REDUCTION
# ========================================================
# Làm sạch tên quận huyện
df['district_name'] = df['district_name'].astype(str).str.lower().str.replace('quận', '').str.replace('huyện', '').str.strip()

# Xử lý nhiễu text trong giá, ép về Float
df['price'] = pd.to_numeric(df['price'], errors='coerce')


# ========================================================
# BƯỚC 1: HANDLING MISSING VALUES
# ========================================================
# Bỏ các dòng thiếu Giá hoặc Diện tích
df = df.dropna(subset=['price', 'area'])

# Điền Missing Value bằng Median
df['bedroom_count'] = df['bedroom_count'].fillna(df['bedroom_count'].median())
df['bathroom_count'] = df['bathroom_count'].fillna(df['bathroom_count'].median())


# ========================================================
# BƯỚC 2: DETECTING AND TREATING OUTLIERS
# ========================================================
df['price_per_sqm'] = df['price'] / df['area']

# Dùng IQR (Interquartile Range) mở rộng để cắt đỉnh/đáy outlier
Q1 = df['price_per_sqm'].quantile(0.05)
Q3 = df['price_per_sqm'].quantile(0.95)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Lọc Outliers 
df = df[(df['price_per_sqm'] >= lower_bound) & (df['price_per_sqm'] <= upper_bound)]
df = df[(df['area'] >= 15) & (df['area'] <= 300)] # Loại bỏ nhà diện tích ảo


# ========================================================
# BƯỚC 4: FEATURE TRANSFORMATION
# ========================================================
# Log-transform làm mượt độ lệch phân phối giá
df['log_price'] = np.log1p(df['price'])

# Chuẩn hoá Z-Score cho K-Means
scaler = StandardScaler()
df[['area_scaled', 'log_price_scaled', 'bedroom_scaled']] = scaler.fit_transform(df[['area', 'log_price', 'bedroom_count']])


# ========================================================
# BƯỚC 5: DATA AGGREGATION & NEW FEATURES
# ========================================================
# Tạo context feature (Giá trung bình theo quận)
district_avg_price = df.groupby('district_name')['price_per_sqm'].mean().reset_index()
district_avg_price.rename(columns={'price_per_sqm': 'avg_district_price_sqm'}, inplace=True)
df = df.merge(district_avg_price, on='district_name', how='left')

# Chỉ số mức độ "Đắt/Rẻ" so với mặt bằng chung quận
df['price_ratio_to_district'] = df['price_per_sqm'] / df['avg_district_price_sqm']


# ========================================================
# BƯỚC 6: ENCODING CATEGORICAL VARIABLES
# ========================================================
# Mã hóa District sang ID số
le = LabelEncoder()
df['district_encoded'] = le.fit_transform(df['district_name'])

# Lưu file hoàn chỉnh
df.to_csv('processed_hanoi_apartments.csv', index=False, encoding='utf-8-sig')