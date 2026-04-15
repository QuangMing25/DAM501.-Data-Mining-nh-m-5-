# Section 5 — Data Mining Methods & Pattern Discovery (25%)

> **Scripts:**
> - `section_5a_kmeans.py` — Thuật toán Unsupervised: K-Means Clustering
> - `section_5b_lightgbm.py` — Thuật toán Supervised: LightGBM Regression
>
> **Input:**
> - `step3_minh/data/hanoi_apartments_for_clustering.csv` — 72.604 bản ghi × 8 cột scaled (Dùng cho K-Means)
> - `step3_minh/data/hanoi_apartments_processed.csv` — 72.604 bản ghi × 37 cột (Dùng cho LightGBM)
>
> **Output:** 4 biểu đồ tại `step5_minh/plots_section_5/`

---

## Tổng quan Chiến lược Kỹ thuật

Bước 5 là sân khấu cuối cùng — nơi các thuật toán học máy thực sự "khai phá" ra tri thức ẩn giấu trong dữ liệu thô. Theo yêu cầu đề bài, nhóm phải sử dụng **ít nhất 2 kỹ thuật Data Mining khác nhau**, trong đó bắt buộc có **ít nhất 1 phương pháp Unsupervised hoặc Pattern-based**.

| # | Phương pháp | Loại | Mục tiêu khai phá |
|---|---|---|---|
| 1 | **K-Means Clustering** | Unsupervised Learning | Phát hiện các phân khúc thị trường ẩn |
| 2 | **LightGBM Regression** | Supervised Learning | Dự đoán giá nhà & Đo lường tầm quan trọng biến |

---

## Phần 5A — K-Means Clustering (Phân Cụm Không Giám Sát)

### 5A.1 Bản chất Thuật toán
K-Means là thuật toán học **không cần nhãn (Unsupervised)** — có nghĩa là ta không hề mách cho mô hình biết giá nhà hay phân khúc nào ra sao. Thuật toán sẽ tự mình "nhìn" vào 6 đặc trưng của 72.604 căn hộ và tìm cách **gom các căn hộ giống nhau vào cùng một nhóm (Cluster)** dựa trên khoảng cách Euclid trong không gian đa chiều.

**Quy trình hoạt động:**
1. Khởi tạo ngẫu nhiên K điểm trung tâm (Centroids).
2. Gán mỗi căn hộ vào Cluster gần nhất.
3. Tính lại Centroid mới (trung bình của tất cả điểm trong Cluster).
4. Lặp lại cho đến khi Centroid không dịch chuyển thêm.

### 5A.2 Tìm K Tối ưu (Elbow Method & Silhouette Score)

Không phải lúc nào cũng biết cần bao nhiêu cụm. Hai phương pháp được dùng đồng thời:

| K | Inertia (WCSS) | Silhouette Score |
|---|---|---|
| 2 | 299.678 | 0.2782 |
| **3** | **246.960** | **0.2604** |
| 4 | 216.197 | 0.2579 |
| 5 | 192.314 | 0.2638 |
| 6 | 172.911 | 0.2633 |
| 7 | 155.039 | 0.2736 |

**Quyết định chọn K = 3** vì:
- **Elbow Method**: Đường cong Inertia gấp khúc rõ nét ở K=3, sau đó tiếp tục giảm chậm dần (Diminishing returns).
- **Kết quả EDA Step 4**: Phân tích PCA đã xác nhận 3 Zone địa lý (Inner/Middle/Outer) tạo ra 3 cụm tự nhiên rõ ràng trong không gian PCA, cho ta cơ sở kinh doanh để tin vào K=3.

![Elbow & Silhouette](plots_section_5/kmeans_01_elbow_silhouette.png)

### 5A.3 Kết Quả Phân Cụm — Knowledge Discovery

Sau khi huấn luyện K-Means với K=3 trên toàn bộ 72.604 căn hộ:

| Cluster | Thị phần | Giá Median | Giá/m² Median | Diện tích TB | Phòng ngủ TB | Đặc điểm Địa lý |
|---|---|---|---|---|---|---|
| **Cụm 0** | 45.5% (33.069 căn) | 5.20 Tỷ | 71.0 Tr/m² | 73.1 m² | 2.2 PN | Middle 74%, Outer 19% |
| **Cụm 1** | 39.3% (28.524 căn) | 9.30 Tỷ | 86.2 Tr/m² | 112.4 m² | 3.0 PN | Middle 86%, Inner 10% |
| **Cụm 2** | 15.2% (11.011 căn) | 3.65 Tỷ | 74.2 Tr/m² | 50.7 m² | 1.6 PN | Middle 56%, Outer 33% |

### 5A.4 Diễn giải Kinh Doanh (Business Interpretation)

Thuật toán đã "tự nhiên" khám phá ra 3 phân khúc khách hàng mà thị trường chưa đặt tên chính thức:

**Cụm 0 — "Phân khúc Chủ Lực Tầm Trung" (45.5% thị phần)**
Đây là khối lượng xương sống của thị trường. Căn hộ 2 phòng ngủ, 73m², giá 5.2 tỷ — nhắm vào hộ gia đình làm công ăn lương ổn định với ngân sách lý tưởng của Hà Nội. Tập trung đông nhất ở phân khu Vành đai 2-3 là nơi đang bùng nổ nguồn cung chung cư mới (Thanh Xuân, Cầu Giấy, Nam Từ Liêm).

**Cụm 1 — "Căn Hộ Premium & Gia Đình Lớn" (39.3% thị phần)**
Phân khúc "hạng sang" của thị trường. Diện tích 112m², 3 phòng ngủ thoải mái, đơn giá 86 Tr/m². Gần 10% tập khách hàng nằm ở khu vực nội đô (Inner Zone). Đây là sản phẩm của các tập đoàn như Vinhomes, Masterise, Him Lam nhắm đến gia đình nhiều thế hệ có tài chính mạnh.

**Cụm 2 — "Căn Studio & Nhỏ Gọn" (15.2% thị phần)**
Căn nhỏ nhắn chỉ 50m², 1-2 phòng ngủ, giá vừa vặn 3.65 tỷ. Tập này có 33% ngoại ô, nhưng cũng có 11% nội đô (Studio đầu tư cho thuê). Phục vụ 2 đối tượng: người trẻ mua nhà lần đầu và nhà đầu tư cho thuê ngắn hạn (Airbnb).

![K-Means PCA Visualization](plots_section_5/kmeans_02_pca_clusters.png)

---

## Phần 5B — LightGBM Regression (Hồi quy Có Giám Sát)

### 5B.1 Bản chất Thuật toán
LightGBM là một thuật toán thuộc họ **Gradient Boosting Decision Trees** (Cây quyết định khuếch đại lỗi). Khác với Random Forest xây cây song song, LightGBM xây cây **tuần tự** — mỗi cây mới học từ lỗi sai của cây trước, tạo ra một ensemble siêu mạnh.

**Tại sao chọn LightGBM chứ không phải Linear Regression thông thường?**
- **Phi tuyến tính**: Trong EDA (eda_06), cùng diện tích 80m² nhưng giá dao động từ 2 tỷ đến 15 tỷ — Linear Regression không thể nắm bắt sự phức tạp này.
- **Interaction Effects**: Diện tích × Quận tạo ra ảnh hưởng chéo phi tuyến — LightGBM xứ lý tự nhiên thông qua phân nhánh cây.
- **Multicollinearity an toàn**: Dù `area` và `bedroom_count` tương quan cao (r=0.75), tree-based models không bị ảnh hưởng bởi đa cộng tuyến.

### 5B.2 Cấu hình Mô hình (Hyperparameters)

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',    # Hồi quy (Predict số thực)
    'metric': 'rmse',
    'learning_rate': 0.05,        # Nhỏ → hội tụ chậm nhưng chắc, tránh overfitting
    'num_leaves': 63,             # Cây sâu vừa phải
    'feature_fraction': 0.8,      # Mỗi cây chỉ dùng 80% features → đa dạng hóa
    'bagging_fraction': 0.8,      # Mỗi cây chỉ dùng 80% dữ liệu → tránh overfit
    'n_estimators': 800           # 800 cây quyết định
}
```

**Biến mục tiêu (Target):** `log_price` (Log của giá tỷ VND) → sau đó chuyển ngược lại `exp(log_price) - 1` để ra giá thực. Dùng log transform giúp hàm mất mát đối xứng và mô hình học ổn định hơn.

**Tập dữ liệu:**
- Train: 58.083 bản ghi (80%)
- Test: 14.521 bản ghi (20%)
- **Số lượng Features đưa vào:** 27 biến (Loại bỏ ID, text trùng, và các cột leak như `price_per_m2`)

### 5B.3 Kết Quả Đánh Giá (Model Evaluation)

| Chỉ số | Giá trị | Diễn giải |
|---|---|---|
| **R² Score** | **0.8106** | Mô hình giải thích được 81.06% biến động giá thị trường |
| **RMSE** | **1.47 Tỷ VND** | Độ lệch bình phương trung bình (nhạy với outlier) |
| **MAE** | **0.96 Tỷ VND** | Sai số tuyệt đối trung bình — lệch chưa đến 1 tỷ/căn |

**Đánh giá độ tốt của mô hình:**
- R² = 0.81 trong bài toán Bất động sản (vốn có vô vàn yếu tố nhiễu: tâm lý, thị trường, thành phần ngầm...) là **rất xuất sắc**.
- MAE = 0.96 Tỷ trên nền giá trung vị 6.25 Tỷ → Sai số tương đối ~15%.
- Để so sánh: Linear Regression thông thường trên cùng dữ liệu thường chỉ đạt R² = 0.55–0.65 do quan hệ phi tuyến.

![Predicted vs Actual](plots_section_5/lightgbm_02_predict_accuracy.png)

### 5B.4 Khai Phá Tầm Quan Trọng Biến (Feature Importance by Gain)

Thuật toán Gain đo lường tổng thông tin mà một biến đóng góp vào việc giảm lỗi dự đoán. Kết quả Top 15:

| Hạng | Feature | Gain | Diễn giải kinh doanh |
|---|---|---|---|
| 1 | `area` | 55,244 | **Diện tích là Vua** — quyết định tổng giá trị căn hộ |
| 2 | `log_area` | 15,918 | Phiên bản log xác nhận quan hệ phi tuyến của diện tích |
| 3 | `district_encoded` | 10,848 | **Quận là Chúa** — địa lý quyết định đơn giá/m² |
| 4 | `pub_month` | 4,029 | Thị trường tăng nóng theo tháng — time series có ý nghĩa |
| 5 | `zone_encoded` | 3,040 | Phân khu Inner/Middle/Outer tách biệt rõ ràng |
| 6 | `bathroom_count` | 1,678 | Proxy của quy mô và chất lượng |
| 7 | `bedroom_count` | 1,005 | Phòng ngủ đi sau phòng tắm — tương quan cao với diện tích |
| 8 | `quality_score` | 802 | **Text Feature #1** — chứng minh bước 3 đúng đắn |
| 9 | `has_premium_amenities` | 659 | Hồ bơi/Gym/Sân chơi tạo premium rõ rệt |
| 10 | `has_legal_paper` | 649 | Sổ đỏ/Pháp lý = Yên tâm = Giá cao hơn |
| 11 | `balcony_dir_Unknown` | 607 | Biến ban công ẩn — thiếu info = căn hộ phổ thông hơn |
| 12 | `feat_near_school` | 324 | Confounding: có nghĩa vì nhiều trường ở vùng ngoại thành rẻ hơn |
| 13 | `feat_full_furniture` | 303 | Nội thất đẩy giá bán thực tế |
| 14 | `feat_near_park` | 294 | Không gian xanh = premium nhỏ |
| 15 | `feat_balcony` | 273 | Ban công thực sự có ảnh hưởng đến giá |

![Feature Importance](plots_section_5/lightgbm_01_feature_importance.png)

**Phát hiện cốt lõi từ Feature Importance:**
1. **Diện tích & Vị trí chi phối 70%+ quyền lực** → Đúng với quy luật BĐS thực tế.
2. **Text Features (quality_score, amenities, legal_paper) TOP 8–10** → Xác nhận việc đầu tư kỹ thuật vào Feature Engineering ở Step 3 đã trả quả xứng đáng.
3. **Hướng ban công (`balcony_dir_Unknown`) góp mặt** → Việc chuyển từ house_direction sang balcony_direction là quyết định đúng đắn.

---

## Kết Luận Bước 5 & Định Hướng Tiếp Theo

### Thành tựu đã đạt được

| Tiêu chí | Kết quả | Đánh giá |
|---|---|---|
| Đáp ứng yêu cầu Unsupervised | ✅ K-Means Clustering | Phát hiện 3 phân khúc ẩn |
| Đáp ứng yêu cầu Pattern Discovery | ✅ Feature Importance Gain | Vạch trần cơ cấu định giá |
| Độ chính xác mô hình | R² = 0.81 | Xuất sắc cho bài toán BĐS |
| Giải thích được (Explainability) | ✅ Feature Importance | AI không còn là hộp đen |

### Định hướng tiếp theo (Nếu muốn nâng cấp thêm)

**Hướng 1 — Demo tương tác (Streamlit App):**
Lưu mô hình LightGBM thành file `.pkl`, xây dựng giao diện web đơn giản bằng Streamlit để giảng viên nhập thông số căn hộ và nhận ngay kết quả dự đoán giá.

**Hướng 2 — Báo cáo chuyên nghiệp (Business Report):**
Tổng hợp các phát hiện từ K-Means và Feature Importance thành báo cáo kinh doanh: Phân khúc nào đang tăng nóng? Yếu tố nào đang bị định giá thiếu? Cảnh báo bong bóng giá?

**Hướng 3 — Cross-Validation & Fine-tuning:**
Áp dụng K-Fold Cross Validation (k=5) cho LightGBM để đánh giá độ ổn định mô hình trên nhiều tập dữ liệu khác nhau, đảm bảo R² không phụ thuộc vào cách chia ngẫu nhiên Train/Test.
