# Hướng dẫn sử dụng Mô hình Dự báo Giá Chung cư Hà Nội (Web App)

Thư mục này chứa toàn bộ các "artifact" cần thiết để xây dựng một ứng dụng demo khai phá dữ liệu và dự báo giá bất động sản.

## 1. Danh sách các file và ý nghĩa

| File | Thành phần | Mô tả |
| :--- | :--- | :--- |
| `le_district.pkl` | LabelEncoder | Chuyển tên Quận (VD: 'Cầu Giấy') thành số nguyên tương ứng. |
| `le_zone.pkl` | LabelEncoder | Chuyển phân vùng (inner, middle, outer) thành số. |
| `scaler.pkl` | StandardScaler | Chuẩn hóa các đặc trưng số trước khi đưa vào mô hình Clustering. |
| `kmeans_model.pkl` | K-Means | Phân loại căn hộ vào 1 trong 3 nhóm (Phổ thông, Tầm trung, Cao cấp). |
| `lgbm_model.pkl` | LightGBM | Mô hình dự báo giá chính đã được huấn luyện. |
| `feature_names.pkl` | List | Danh sách và thứ tự các cột mà LightGBM yêu cầu. |

## 2. Quy trình dự báo (Inference Pipeline)

Để dự báo cho một căn hộ mới, Web App cần thực hiện theo đúng trình tự sau:

1.  **Input:** Nhận dữ liệu từ người dùng (Quận, Diện tích, Số phòng ngủ, Số vệ sinh, Hướng ban công, Mô tả).
2.  **Xử lý Logic (Feature Engineering):**
    - Sử dụng `le_district` để encode tên Quận.
    - Ánh xạ Quận sang Zone (inner/middle/outer) dựa theo logic trong Step 3 và dùng `le_zone` để encode.
    - **Log Transform:** Chuyển `area` thành `log_area` bằng `np.log1p(area)`.
3.  **Text Features:** Chạy keyword matching trên cột mô tả để tạo ra các cột binary (`feat_balcony`, `has_legal_paper`, v.v.) và tính `quality_score`.
4.  **Clustering (Phân khúc):** 
    - Vì mô hình K-Means hiện tại sử dụng cả thông tin giá để phân cụm (Knowledge Discovery), trong Web App Demo, bạn nên cho phép người dùng chọn phân khúc:
        - Cluster 0: Phổ thông
        - Cluster 1: Cao cấp
        - Cluster 2: Tầm trung
5.  **Predict:** 
    - Sắp xếp tất cả các đặc trưng theo đúng thứ tự trong `feature_names.pkl`.
    - Dùng `lgbm_model.predict(input_data)`.
6.  **Transform:** Kết quả đầu ra là `log_price`. Tính giá thực bằng: `price = np.expm1(y_pred)`.

## 3. Code mẫu (Python)

```python
import pickle
import pandas as pd
import numpy as np

# Load artifacts
with open('app_models/lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('app_models/le_district.pkl', 'rb') as f:
    le_district = pickle.load(f)
with open('app_models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Giả lập dữ liệu input
input_data = {
    'log_area': np.log1p(75.0),
    'bedroom_count': 2,
    'bathroom_count': 2,
    'district_encoded': le_district.transform(['Cầu Giấy'])[0],
    'Cluster': 0, # Phân khúc Phổ thông
    # ... bổ sung các features khác ...
}

df_input = pd.DataFrame([input_data])
df_input = df_input.reindex(columns=feature_names, fill_value=0)

# Dự báo
y_log = model.predict(df_input)
price_vnd = np.expm1(y_log)[0]
print(f"Giá dự báo: {price_vnd/1e9:.2f} Tỷ VND")
```

## 4. Lưu ý quan trọng
- **Độ lệch (Skewness):** Mô hình sử dụng Log Transform cho Target (Price) và Feature (Area) để giảm độ lệch dữ liệu. Luôn nhớ `log1p` khi input và `expm1` khi output.
- **Thứ tự cột:** Tuyệt đối phải dùng `feature_names.pkl` để sắp xếp cột dữ liệu trước khi predict, nếu không kết quả sẽ bị sai lệch hoàn toàn.
