
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(page_title="Hệ Thống Định Giá Chung Cư Hà Nội 2026", layout="wide", initial_sidebar_state="expanded")

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .big-price { font-size: 4rem; font-weight: bold; color: #00b894; margin-bottom: -10px; }
    .price-unit { font-size: 1.5rem; color: #636e72; }
    .metric-box { background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e9ecef; }
    .cluster-tag { display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; color: white; margin-bottom: 15px; }
    .sidebar-header { color: #2d3436; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: THÔNG SỐ NHẬP LIỆU ---
st.sidebar.markdown("<div class='sidebar-header'>📂 CẤU HÌNH DỮ LIỆU</div>", unsafe_allow_html=True)
dataset_option = st.sidebar.selectbox(
    "Bộ dữ liệu phân tích",
    ["Dữ liệu Tiêu chuẩn (Gốc)", "Dữ liệu Nâng cao (2025 Adjusted)", "Dữ liệu Final"],
    help="Chọn nguồn dữ liệu và mô hình dự báo tương ứng."
)
is_dataset_2 = dataset_option == "Dữ liệu Nâng cao (2025 Adjusted)"
is_dataset_minh = dataset_option == "Dữ liệu Final"

# --- LOAD ARTIFACTS (BRAIN) ---
@st.cache_resource
def load_prediction_engine(dataset_type):
    path = "app_models/"
    artifacts = {}
    # Các thành phần dùng chung
    files = {
        'kmeans': 'kmeans_model.pkl',
        'scaler': 'scaler.pkl',
        'le_district': 'le_district.pkl',
        'le_zone': 'le_zone.pkl'
    }
    for key, filename in files.items():
        with open(os.path.join(path, filename), 'rb') as f:
            artifacts[key] = pickle.load(f)
    
    # Thành phần thay đổi theo bộ dữ liệu
    if dataset_type == "Dữ liệu Nâng cao (2025 Adjusted)":
        model_path = "data_v2/2025/lgbm_model_v2_2025.pkl"
        feat_path = "data_v2/2025/feature_names_v2_2025.pkl"
        with open(model_path, 'rb') as f:
            artifacts['model'] = pickle.load(f)
        with open(feat_path, 'rb') as f:
            artifacts['feature_names'] = pickle.load(f)
    else:
        # Cả Dataset 1 và Dataset Minh đều dùng mô hình chuẩn
        with open(os.path.join(path, 'lgbm_model.pkl'), 'rb') as f:
            artifacts['model'] = pickle.load(f)
        artifacts['feature_names'] = artifacts['model'].feature_name_
        
    return artifacts

try:
    engine = load_prediction_engine(dataset_option)
except Exception as e:
    st.error(f"Lỗi khởi động bộ não dự báo: {e}")
    st.stop()

# --- LOAD DATA SOURCE (FOR STATS & PROJECTS) ---
@st.cache_data
def load_data_for_stats(dataset_type):
    if dataset_type == "Dữ liệu Nâng cao (2025 Adjusted)":
        path = 'data_v2/2025/df_2025_FINAL_ADJUSTED.csv'
    elif dataset_type == "Dữ liệu Final)":
        path = 'step3_minh/data/hanoi_apartments_processed.csv'
    else:
        path = 'step5_binh/data/hanoi_apartments_final_results.csv'
        
    if not os.path.exists(path):
        return pd.DataFrame(), []
    
    df = pd.read_csv(path)
    
    # Nếu là dữ liệu của Minh, cần dự đoán Cluster vì file chưa có
    if dataset_type == "Dữ liệu Final" and 'Cluster' not in df.columns:
        try:
            # Chuẩn bị dữ liệu cho KMeans (phải khớp với logic huấn luyện)
            cluster_cols = ["log_price", "log_area", "bedroom_count", "bathroom_count", "log_price_per_m2", "district_encoded"]
            X_cluster = df[cluster_cols]
            X_scaled = engine['scaler'].transform(X_cluster)
            df['Cluster'] = engine['kmeans'].predict(X_scaled)
        except Exception as e:
            st.warning(f"Không thể tạo cột Cluster cho dữ liệu Minh: {e}")

    # Chuẩn hóa Quận để khớp với LabelEncoder
    df['district_name'] = df['district_name'].str.title()
    valid_le_districts = list(engine['le_district'].classes_)
    df = df[df['district_name'].isin(valid_le_districts)]
    
    available_districts = sorted(df['district_name'].unique().tolist())
    return df, available_districts

try:
    df_stats, valid_districts = load_data_for_stats(is_dataset_2)
except Exception as e:
    st.error(f"Lỗi tải dữ liệu thống kê: {e}")
    df_stats = pd.DataFrame()
    valid_districts = sorted(list(engine['le_district'].classes_))

if df_stats.empty:
    valid_districts = sorted(list(engine['le_district'].classes_))

# --- SIDEBAR: CĂN HỘ ---
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='sidebar-header'>🏢 THÔNG SỐ CĂN HỘ</div>", unsafe_allow_html=True)

# 1. Địa điểm
sel_district = st.sidebar.selectbox("Quận/Huyện", valid_districts)
sel_district_code = engine['le_district'].transform([sel_district])[0]

# 2. Dự án (Lấy từ dữ liệu thống kê)
projects = []
if not df_stats.empty:
    projects = sorted(df_stats[df_stats['district_name'] == sel_district]['project_name'].dropna().unique().tolist())

if not projects: 
    projects = ["Dự án khác"]
else: 
    projects = ["Tất cả dự án"] + projects + ["Dự án khác"]

sel_project = st.sidebar.selectbox("Dự án / Chủ đầu tư", projects)

# 3. Thông số vật lý
st.sidebar.markdown("---")
sel_area = st.sidebar.slider("Diện tích (m²)", 20, 300, 75)
sel_bed = st.sidebar.number_input("Số phòng ngủ", 1, 10, 2)
sel_bath = st.sidebar.number_input("Số phòng vệ sinh", 1, 10, 2)
sel_month = st.sidebar.slider("Tháng dự báo", 1, 12, 4)

# 4. Tiện ích & Hướng
st.sidebar.markdown("---")
st.sidebar.subheader("Tiện ích & Hướng")
col1, col2 = st.sidebar.columns(2)
with col1:
    sel_legal = st.sidebar.checkbox("Sổ hồng", True)
    sel_furniture = st.sidebar.checkbox("Full nội thất", True)
    sel_balcony = st.sidebar.checkbox("Có ban công", True)
with col2:
    sel_corner = st.sidebar.checkbox("Căn góc", False)
    sel_amenities = st.sidebar.checkbox("Tiện ích VIP", False)

balcony_dirs = ['Bắc', 'Nam', 'Tây', 'Tây Bắc', 'Tây Nam', 'Đông', 'Đông Bắc', 'Đông Nam']
sel_dir = st.sidebar.selectbox("Hướng chính", balcony_dirs)

# 5. Tiện ích xung quanh
st.sidebar.markdown("---")
st.sidebar.subheader("Tiện ích xung quanh (1km)")
col3, col4 = st.sidebar.columns(2)
with col3:
    sel_school = st.sidebar.checkbox("Gần Trường học", True)
    sel_hospital = st.sidebar.checkbox("Gần Bệnh viện", False)
with col4:
    sel_mall = st.sidebar.checkbox("Siêu thị/TTTM", True)
    sel_park = st.sidebar.checkbox("Công viên/Hồ", False)

quality_score = sum([sel_furniture, sel_corner, sel_legal, sel_amenities, sel_school, sel_hospital, sel_mall, sel_park, sel_balcony])

# --- INFERENCE ---

# 1. District & Zone
inner = ['Hoàn Kiếm', 'Ba Đình', 'Hai Bà Trưng', 'Đống Đa', 'Cầu Giấy', 'Thanh Xuân', 'Tây Hồ']
outer = ['Đông Anh', 'Gia Lâm', 'Thanh Trì', 'Hoài Đức', 'Thanh Oai', 'Thạch Thất', 'Chương Mỹ', 'Sóc Sơn']
zone_name = 'inner' if sel_district in inner else ('outer' if sel_district in outer else 'middle')
sel_zone_code = engine['le_zone'].transform([zone_name])[0]

# 2. Dự đoán Cluster (Phân khúc)
# Sử dụng dữ liệu trung bình quận từ df_stats để mô hình KMeans hoạt động chính xác
if not df_stats.empty:
    dist_data = df_stats[df_stats['district_name'] == sel_district]
    avg_log_p = dist_data['log_price'].mean() if not dist_data.empty else 22.0
    avg_log_ppm2 = dist_data['log_price_per_m2'].mean() if not dist_data.empty else 18.0
else:
    avg_log_p, avg_log_ppm2 = 22.0, 18.0

cluster_in = pd.DataFrame([[
    avg_log_p, np.log1p(sel_area), sel_bed, sel_bath, avg_log_ppm2, sel_district_code
]], columns=["log_price", "log_area", "bedroom_count", "bathroom_count", "log_price_per_m2", "district_encoded"])

pred_cluster = int(engine['kmeans'].predict(engine['scaler'].transform(cluster_in))[0])

# 3. Dự đoán Giá (LightGBM)
input_row = {col: 0.0 for col in engine['feature_names']}
input_row.update({
    'area': float(sel_area), 'log_area': float(np.log1p(sel_area)),
    'bedroom_count': float(sel_bed), 'bathroom_count': float(sel_bath),
    'district_encoded': float(sel_district_code), 'zone_encoded': float(sel_zone_code),
    'quality_score': float(quality_score), 'Cluster': pred_cluster,
    'pub_month': float(sel_month), 'pub_year': 2026.0,
    'has_legal_paper': 1.0 if sel_legal else 0.0,
    'feat_full_furniture': 1.0 if sel_furniture else 0.0,
    'feat_corner_unit': 1.0 if sel_corner else 0.0,
    'has_premium_amenities': 1.0 if sel_amenities else 0.0,
    'feat_near_school': 1.0 if sel_school else 0.0,
    'feat_near_hospital': 1.0 if sel_hospital else 0.0,
    'feat_near_mall': 1.0 if sel_mall else 0.0,
    'feat_near_park': 1.0 if sel_park else 0.0,
    'feat_balcony': 1.0 if sel_balcony else 0.0
})

# Bổ sung chỉ số vĩ mô cho Bộ dữ liệu 2
if is_dataset_2:
    input_row.update({
        'macro_cpi_general': 100.1116,
        'macro_cpi_housing': 100.0593,
        'macro_gold_index': 103.0035,
        'macro_cpi_general_lag1': 100.2815,
        'macro_cpi_housing_lag1': 99.7713,
        'macro_gold_index_lag1': 103.0288
    })

# Hướng ban công (Khớp format: balcony_dir_Tây_Bắc)
formatted_dir = sel_dir.replace(' ', '_')
dir_col = f"balcony_dir_{formatted_dir}"
if dir_col in input_row: input_row[dir_col] = 1.0

input_df = pd.DataFrame([input_row])[engine['feature_names']]
input_df['Cluster'] = pd.Categorical(input_df['Cluster'], categories=[0, 1, 2])

# Dự báo
try:
    pred_log = engine['model'].predict(input_df)[0]
    res_ty = np.expm1(pred_log) / 1e9
    res_ppm2 = (res_ty * 1e3) / sel_area
except Exception as e:
    st.error(f"Lỗi dự báo: {e}")
    st.stop()

# --- UI CHÍNH ---
st.title("🤖 ĐỊNH GIÁ BẤT ĐỘNG SẢN THÔNG MINH 2026")

if is_dataset_2:
    st.info("📍 Đang sử dụng: **Dữ liệu Nâng cao ** | Mô hình vĩ mô tích hợp.")
else:
    st.info("📍 Đang sử dụng: **Dữ liệu Tiêu chuẩn ** | Mô hình suy diễn cơ bản.")

c_info = {0: ("Phổ thông", "#3498db"), 1: ("Premium", "#e74c3c"), 2: ("Trung cấp", "#f1c40f")}
name, color = c_info.get(pred_cluster, ("N/A", "gray"))
st.markdown(f"<div class='cluster-tag' style='background-color:{color};'>Phân khúc: {name}</div>", unsafe_allow_html=True)

col_main, col_stat = st.columns([2, 1])

with col_main:
    st.markdown("### GIÁ DỰ BÁO HIỆN TẠI")
    st.markdown(f"<span class='big-price'>{res_ty:.2f}</span> <span class='price-unit'>Tỷ VNĐ</span>", unsafe_allow_html=True)
    st.markdown(f"📐 **{sel_area} m²** | 💰 **{res_ppm2:.1f} Triệu/m²**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- THỐNG KÊ SO SÁNH (PHẢN HỒI NGAY LẬP TỨC) ---
    if not df_stats.empty:
        # Lọc dữ liệu để tính q1, q3
        s_df = df_stats[df_stats['district_name'] == sel_district]
        if sel_project not in ["Tất cả dự án", "Dự án khác"]:
            s_df = s_df[s_df['project_name'] == sel_project]
        
        # Lọc theo diện tích tương đồng (+/- 20%)
        s_df = s_df[(s_df['area'] >= sel_area*0.8) & (s_df['area'] <= sel_area*1.2)]
        
        if s_df.empty: # Fallback về Quận nếu lọc quá sâu
            s_df = df_stats[df_stats['district_name'] == sel_district]
            label_prefix = f"Quận {sel_district}"
        else:
            label_prefix = f"Dự án {sel_project}" if sel_project not in ["Tất cả dự án", "Dự án khác"] else f"Quận {sel_district}"

        real_ppm2_stats = s_df['price_per_m2'] / 1e6
        q1 = real_ppm2_stats.quantile(0.1) if not real_ppm2_stats.empty else res_ppm2 * 0.8
        q3 = real_ppm2_stats.quantile(0.9) if not real_ppm2_stats.empty else res_ppm2 * 1.2
        
        st.markdown(f"**So sánh với thị giá tại {label_prefix}:**")
        st.progress(max(0.0, min(1.0, (res_ppm2 - q1)/(q3 - q1) if q3 != q1 else 0.5)))
        
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metric-box' style='border-color:#00b894; color:#00b894;'>Giá sàn<br><b>{q1:.1f} Tr</b></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-box' style='border-color:#00b894; color:#00b894;'>Dự báo của bạn<br><b>{res_ppm2:.1f} Tr</b></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-box' style='border-color:#00b894; color:#00b894;'>Giá trần<br><b>{q3:.1f} Tr</b></div>", unsafe_allow_html=True)

        # Biểu đồ xu hướng Quận
        st.markdown(f"### 📈 Xu hướng giá thực tế tại {sel_district.upper()}")
        trend_df = df_stats[df_stats['district_name'] == sel_district]
        if not trend_df.empty:
            trend = trend_df.groupby('pub_month')['price_per_m2'].mean().reset_index()
            trend['price_per_m2'] /= 1e6
            fig = go.Figure(go.Scatter(x=trend['pub_month'], y=trend['price_per_m2'], mode='lines+markers', line=dict(color='#00b894')))
            fig.update_layout(xaxis_title="Tháng", yaxis_title="Triệu/m²", height=300, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

with col_stat:
    st.markdown("### 💠 Phân tích chất lượng")
    radar_scores = [min(10, 4 + (res_ppm2/20)), 10 if sel_legal else 3, quality_score * 1.1, 8, 7]
    radar_labels = ['Vị trí', 'Pháp lý', 'Tiện ích', 'Tiềm năng', 'Độ hời']
    fig_radar = go.Figure(go.Scatterpolar(r=radar_scores + [radar_scores[0]], theta=radar_labels + [radar_labels[0]], fill='toself', fillcolor='rgba(0, 184, 148, 0.3)', line=dict(color='#00b894')))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 10])), showlegend=False, height=350, margin=dict(l=40,r=40,t=40,b=40))
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.info(f"**Điểm chất lượng: {quality_score}/9**\n\nCăn hộ đạt {quality_score} tiêu chí về hạ tầng và pháp lý.")

st.sidebar.markdown("---")
st.sidebar.caption(f"© 2026 BĐS Team - Stable Release")
