import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Chi Tiết Định Giá Chung Cư", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
<style>
    .big-price {
        font-size: 4rem;
        font-weight: bold;
        color: #00b894;
        margin-bottom: -10px;
    }
    .price-unit {
        font-size: 1.5rem;
        color: #636e72;
    }
    .address-text {
        color: #636e72;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .cluster-tag {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Sử dụng file đã tiền xử lý từ Step 3
    df = pd.read_csv('step3_minh/data/hanoi_apartments_processed.csv')
    return df

df_full = load_data()

# Lấy từ điển map giữa tên quận và mã code
district_mapping = df_full.drop_duplicates(subset=['district_name', 'district_encoded'])[['district_name', 'district_encoded']].sort_values('district_name')

# ==================== PIPELINE STEP 5 (TRAINING) ====================
@st.cache_resource
def train_pipeline(data):
    # 1. K-Means (Step 5a)
    # Các features dùng cho clustering (theo section_3_preprocessing.py)
    cluster_features = ["log_price", "log_area", "bedroom_count", "bathroom_count",
                        "log_price_per_m2", "district_encoded"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[cluster_features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 2. LightGBM (Step 5b)
    # Features baseline từ Step 3 + Cluster từ Step 5
    features_to_drop = ['price', 'log_price', 'price_per_m2', 'log_price_per_m2',
                        'district_name', 'ward_name', 'street_name', 'project_name',
                        'district_zone', 'published_at']
    
    X_cols = [c for c in data.columns if c not in features_to_drop and not c.startswith('scaled_')]
    X = data[X_cols].copy()
    X['Cluster'] = X['Cluster'].astype('category')
    y = data['log_price']
    
    # Huấn luyện model (Parameters từ Step 5b)
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X, label=y, categorical_feature=['Cluster'])
    model = lgb.train(lgb_params, train_data, num_boost_round=800)
    
    # Lưu lại centroid và scaler để inference
    centroids = kmeans.cluster_centers_
    
    return model, X_cols, scaler, cluster_features, centroids, kmeans

model, model_features, scaler, cluster_features, centroids, kmeans_model = train_pipeline(df_full)

# ==================== SIDEBAR (NHẬP LIỆU) ====================
st.sidebar.title("Thông số Căn hộ ")

sel_district = st.sidebar.selectbox("Quận/Huyện", district_mapping['district_name'].tolist())
sel_district_code = district_mapping[district_mapping['district_name'] == sel_district]['district_encoded'].values[0]

sel_area = st.sidebar.slider("Diện tích (m²)", min_value=20, max_value=250, value=80, step=1)
sel_bed = st.sidebar.number_input("Số phòng ngủ", min_value=1, max_value=6, value=2, step=1)
sel_bath = st.sidebar.number_input("Số phòng tắm", min_value=1, max_value=5, value=2, step=1)
sel_month = st.sidebar.slider("Tháng dự đoán", min_value=1, max_value=12, value=4)

st.sidebar.markdown("---")
st.sidebar.subheader("Tiện ích & Pháp lý")
sel_legal = st.sidebar.checkbox("Sổ đỏ / Sổ hồng", value=True)
sel_furniture = st.sidebar.checkbox("Đầy đủ nội thất", value=True)
sel_amenities = st.sidebar.checkbox("Tiện ích VIP (Bể bơi, Gym...)", value=False)
sel_corner = st.sidebar.checkbox("Căn góc", value=False)
sel_balcony = st.sidebar.checkbox("Có ban công", value=True)

# Các tiện ích gần nhà (Text features)
st.sidebar.subheader("Vị trí gần")
sel_school = st.sidebar.checkbox("Trường học", value=True)
sel_hospital = st.sidebar.checkbox("Bệnh viện", value=False)
sel_mall = st.sidebar.checkbox("Siêu thị / TTTM", value=True)
sel_park = st.sidebar.checkbox("Công viên", value=False)

# Tính toán quality_score giống Step 3
text_feat_values = [
    sel_furniture, sel_corner, sel_legal, sel_amenities, 
    sel_school, sel_hospital, sel_mall, sel_park, sel_balcony
]
quality_score = sum([1 for f in text_feat_values if f])

# ==================== DỰ ĐOÁN CLUSTER (STAGE 1) ====================
# Vì không có giá để tính cluster chính xác, ta dùng kỹ thuật "Gần đúng" (Euclidean trên các biến có sẵn)
# Mapping input sang không gian scaled của K-Means
# cluster_features = ["log_price", "log_area", "bedroom_count", "bathroom_count", "log_price_per_m2", "district_encoded"]

# Giả định log_price và log_price_per_m2 ở mức trung bình của Quận để tìm Cluster phù hợp nhất
district_avg_log_price = df_full[df_full['district_name'] == sel_district]['log_price'].mean()
district_avg_log_ppm2  = df_full[df_full['district_name'] == sel_district]['log_price_per_m2'].mean()

input_for_scaling = pd.DataFrame([[
    district_avg_log_price,
    np.log1p(sel_area),
    sel_bed,
    sel_bath,
    district_avg_log_ppm2,
    sel_district_code
]], columns=cluster_features)

input_scaled = scaler.transform(input_for_scaling)
# Dự đoán Cluster dựa trên model K-Means đã train
pred_cluster = kmeans_model.predict(input_scaled)[0]

# Tên cụm theo logic Step 5
cluster_names = {
    0: ("Phân khúc Chủ Lực", "#3498db"),
    1: ("Phân khúc Premium", "#e74c3c"),
    2: ("Phân khúc Studio/Phổ thông", "#f1c40f")
}
c_name, c_color = cluster_names.get(pred_cluster, ("Không xác định", "gray"))

# ==================== DỰ ĐOÁN GIÁ (STAGE 2) ====================
# Chuẩn bị input cho LightGBM (Phải khớp 100% với model_features)
input_row = {col: 0 for col in model_features} # Khởi tạo 0

input_row['area'] = sel_area
input_row['bedroom_count'] = sel_bed
input_row['bathroom_count'] = sel_bath
input_row['pub_month'] = sel_month
input_row['pub_year'] = 2026 # Giả định tương lai
input_row['log_area'] = np.log1p(sel_area)
input_row['district_encoded'] = sel_district_code
input_row['zone_encoded'] = df_full[df_full['district_name'] == sel_district]['zone_encoded'].values[0]
input_row['feat_full_furniture'] = 1 if sel_furniture else 0
input_row['feat_corner_unit'] = 1 if sel_corner else 0
input_row['has_legal_paper'] = 1 if sel_legal else 0
input_row['has_premium_amenities'] = 1 if sel_amenities else 0
input_row['feat_near_school'] = 1 if sel_school else 0
input_row['feat_near_hospital'] = 1 if sel_hospital else 0
input_row['feat_near_mall'] = 1 if sel_mall else 0
input_row['feat_near_park'] = 1 if sel_park else 0
input_row['feat_balcony'] = 1 if sel_balcony else 0
input_row['quality_score'] = quality_score
input_row['Cluster'] = pred_cluster

# Chuyển sang DataFrame
input_df = pd.DataFrame([input_row])
input_df['Cluster'] = input_df['Cluster'].astype('category')

pred_log_price = model.predict(input_df)[0]
pred_price_vnd = np.exp(pred_log_price) - 1 # Giá gốc đơn vị VNĐ
pred_price_ty = pred_price_vnd / 1e9        # Đổi sang Tỷ để hiển thị
pred_price_per_m2 = (pred_price_vnd / sel_area) / 1e6 # Đổi sang Triệu/m2 để hiển thị

# Thống kê khu vực (quy đổi sang Triệu/m2)
district_df = df_full[df_full['district_name'] == sel_district]
min_price_m2 = district_df['price_per_m2'].quantile(0.05) / 1e6
max_price_m2 = district_df['price_per_m2'].quantile(0.95) / 1e6

# ==================== MAIN UI ====================
st.title("DỮ LIỆU KHAI PHÁ THỊ TRƯỜNG 2026")
st.markdown("## 🤖 Chi tiết định giá chung cư")
st.markdown(f"📍 <span class='address-text'>Căn hộ Quận {sel_district}, Hà Nội</span>", unsafe_allow_html=True)

# Hiển thị nhãn Cluster
st.markdown(f"<div class='cluster-tag' style='background-color:{c_color};'>Cụm {pred_cluster}: {c_name}</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # Header giá
    st.markdown(f"GIÁ ƯỚC TÍNH HIỆN TẠI")
    st.markdown(f"<span class='big-price'>{pred_price_ty:.2f}</span> <span class='price-unit'>Tỷ VNĐ</span>", unsafe_allow_html=True)
    st.markdown(f"📐 **{sel_area} m²** • **{pred_price_per_m2:.2f} Triệu/m²**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Thanh Gauge
    progress_val = (pred_price_per_m2 - min_price_m2) / (max_price_m2 - min_price_m2) * 100
    progress_val = max(0, min(100, progress_val))
    st.progress(int(progress_val))
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-box' style='border-color:#00b894; color:#00b894;'>Giá sàn khu vực<br><b>{:.2f} Tr/m²</b></div>".format(min_price_m2), unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-box' style='border-color:#00b894; color:#00b894;'>Giá dự báo<br><b>{:.2f} Tr/m²</b></div>".format(pred_price_per_m2), unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-box' style='border-color:#00b894; color:#00b894;'>Giá trần khu vực<br><b>{:.2f} Tr/m²</b></div>".format(max_price_m2), unsafe_allow_html=True)

    st.success(f"💡 **Insight từ Model:** Căn hộ của bạn thuộc **{c_name}**. Hệ thống LightGBM đã áp dụng quy luật giá riêng cho phân khúc này để đưa ra kết quả chính xác hơn.")

    # Biểu đồ xu hướng
    st.markdown(f"### 📈 BIẾN ĐỘNG GIÁ TẠI {sel_district.upper()}")
    trend_df = district_df.groupby('pub_month')['price_per_m2'].agg(['mean', 'std']).reset_index()
    trend_df = trend_df.sort_values('pub_month')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['pub_month'], y=trend_df['mean'], mode='lines+markers', name='Trung bình', line=dict(color='#00b894', width=3)))
    fig.update_layout(xaxis_title="Tháng", yaxis_title="Triệu/m²", margin=dict(l=0, r=0, t=30, b=0), height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Radar Chart
    st.markdown("### 💠 Phân tích Chất lượng")
    
    # Radar Scores
    scores = [
        min(10, 5 + (pred_price_per_m2 / 25)), # Vị trí
        10 if sel_legal else 3,                # Pháp lý
        quality_score * 1.1,                   # Tiện ích (tối đa 9.9)
        8,                                     # Tiềm năng
        max(2, 10 - (progress_val/10))         # Độ hời (giá càng cao điểm càng thấp)
    ]
    categories = ['Vị trí', 'Pháp lý', 'Tiện ích', 'Tiềm năng', 'Giá cả']
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=scores + [scores[0]], theta=categories + [categories[0]], fill='toself', fillcolor='rgba(0, 184, 148, 0.4)', line=dict(color='#00b894')))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 10])), showlegend=False, margin=dict(l=40, r=40, t=40, b=40), height=400)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.info(f"**Quality Score: {quality_score}/9**\n\nBạn đã chọn {quality_score} tiêu chí chất lượng cho căn hộ này.")

st.sidebar.markdown("---")












