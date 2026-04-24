# Section 5 — Data Mining Methods & Pattern Discovery (25%)

> **Scripts:**
> - `section_5a_kmeans.py` — Thuật toán Unsupervised: K-Means Clustering
> - `section_5b_lightgbm.py` — Thuật toán Supervised: LightGBM Regression (Tích hợp kết quả K-Means)
>
> **Input:**
> - `step3_minh/data/hanoi_apartments_for_clustering.csv` — 72.604 bản ghi × 8 cột scaled (Dùng cho K-Means)
> - `step3_minh/data/hanoi_apartments_processed.csv` — 72.604 bản ghi × 37 cột (Dùng cho LightGBM)
>
> **Output:** 6 biểu đồ tại `step5_binh/plots_section_5/`
>
> **Pipeline liên kết:** Bước 5A (K-Means) → Gán nhãn Cluster → Bước 5B (LightGBM sử dụng Cluster làm feature)

---

## Tổng quan Chiến lược Kỹ thuật

Bước 5 là sân khấu cuối cùng — nơi các thuật toán học máy thực sự "khai phá" ra tri thức ẩn giấu trong dữ liệu thô. Theo yêu cầu đề bài, nhóm phải sử dụng **ít nhất 2 kỹ thuật Data Mining khác nhau**, trong đó bắt buộc có **ít nhất 1 phương pháp Unsupervised hoặc Pattern-based**.

**Điểm đặc biệt của pipeline nhóm 5:** Hai thuật toán không chạy độc lập mà được **nối tiếp thành chuỗi tri thức (Knowledge-Driven Pipeline)**:

```
5A: K-Means Clustering → Phát hiện 3 phân khúc thị trường ẩn
        │
        ▼ (Gán nhãn Cluster làm feature mới)
        │
5B: LightGBM Regression → Dự đoán giá + Đánh giá giá trị của phân khúc
```

| # | Phương pháp | Loại | Mục tiêu khai phá | Liên kết |
|---|---|---|---|---|
| 1 | **K-Means Clustering** | Unsupervised Learning | Phát hiện các phân khúc thị trường ẩn | Output → Input cho 5B |
| 2 | **LightGBM Regression** | Supervised Learning | Dự đoán giá nhà & Đo lường tầm quan trọng biến | Nhận Cluster label từ 5A |

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

## Phần 5B — LightGBM Regression (Hồi quy Có Giám Sát + Tích hợp K-Means)

### 5B.1 Bản chất Thuật toán
LightGBM là một thuật toán thuộc họ **Gradient Boosting Decision Trees** (Cây quyết định khuếch đại lỗi). Khác với Random Forest xây cây song song, LightGBM xây cây **tuần tự** — mỗi cây mới học từ lỗi sai của cây trước, tạo ra một ensemble siêu mạnh.

**Tại sao chọn LightGBM chứ không phải Linear Regression thông thường?**
- **Phi tuyến tính**: Trong EDA (eda_06), cùng diện tích 80m² nhưng giá dao động từ 2 tỷ đến 15 tỷ — Linear Regression không thể nắm bắt sự phức tạp này.
- **Interaction Effects**: Diện tích × Quận tạo ra ảnh hưởng chéo phi tuyến — LightGBM xử lý tự nhiên thông qua phân nhánh cây.
- **Multicollinearity an toàn**: Dù `area` và `bedroom_count` tương quan cao (r=0.75), tree-based models không bị ảnh hưởng bởi đa cộng tuyến.

### 5B.2 Chiến lược Pipeline: Tích hợp K-Means → LightGBM

Điểm khác biệt cốt lõi so với cách tiếp cận truyền thống: **Bước 5B không chạy độc lập, mà kế thừa trực tiếp tri thức từ Bước 5A**.

Cụ thể, nhãn Cluster (0/1/2) từ K-Means được đưa vào LightGBM như một **categorical feature bổ sung**. Ý nghĩa:
- Cluster encode thông tin **tương tác đa biến** (giá × diện tích × vị trí × phòng ngủ) mà không biến đơn lẻ nào chứa được.
- LightGBM nhận thêm "bản đồ phân khúc" do K-Means vẽ ra, giúp model hiểu rằng: một căn hộ thuộc phân khúc Premium sẽ có quy luật giá khác phân khúc Phổ thông.

Để chứng minh giá trị thực sự, nhóm chạy **2 mô hình trên cùng điều kiện** và so sánh:

| Yếu tố | Mô hình 1 (Baseline) | Mô hình 2 (+ K-Means) |
|---|---|---|
| Số features | 27 | 28 (thêm cột `Cluster`) |
| Dữ liệu input | `hanoi_apartments_processed.csv` | Giống + nhãn K-Means từ 5A |
| Train/Test split | 80/20 (seed=42) | Giống hệt (đảm bảo công bằng) |
| Hyperparameters | Giống nhau | Giống nhau |

### 5B.3 Cấu hình Mô hình (Hyperparameters)

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
- **Baseline:** 27 features | **Enhanced:** 28 features (thêm `Cluster` categorical)

### 5B.4 Kết Quả So Sánh — Baseline vs Tích hợp K-Means

| Chỉ số | Baseline (Không K-Means) | + K-Means Cluster | Thay đổi |
|---|---|---|---|
| **R² Score** | 0.8106 | **0.8487** | **+3.81 điểm %** |
| **RMSE** | 1.4743 Tỷ VND | **1.3176 Tỷ VND** | **-156.6 triệu** ↓ |
| **MAE** | 0.9632 Tỷ VND | **0.8606 Tỷ VND** | **-102.5 triệu** ↓ |
| **MAPE** | 15.97% | **13.84%** | **-2.13%** ↓ |

> **✅ KẾT LUẬN: Tích hợp K-Means cải thiện mô hình toàn diện.** R² tăng từ 0.81 lên 0.85, sai số MAE giảm hơn 100 triệu VND/căn. Điều này chứng minh rằng tri thức phân khúc từ Unsupervised Learning (Bước 5A) có giá trị thực sự khi đưa vào Supervised Learning (Bước 5B).

![So sánh Metrics 2 mô hình](plots_section_5/lightgbm_00_comparison_metrics.png)

![Scatter: Giá Dự đoán vs Giá Thực — Baseline (trái) vs + K-Means (phải)](plots_section_5/lightgbm_02_predict_accuracy.png)

### 5B.5 Khai Phá Tầm Quan Trọng Biến (Feature Importance by Gain)

Thuật toán Gain đo lường tổng thông tin mà một biến đóng góp vào việc giảm lỗi dự đoán. Kết quả Top 15 từ **mô hình tích hợp K-Means**:

| Hạng | Feature | Gain | Diễn giải kinh doanh |
|---|---|---|---|
| 🥇 | **`Cluster`** | **44,318** | **🏆 Phân khúc K-Means là biến quyền lực #1** — vượt xa cả diện tích! |
| 2 | `area` | 27,287 | Diện tích — trụ cột truyền thống của định giá BĐS |
| 3 | `log_area` | 8,470 | Phiên bản log xác nhận quan hệ phi tuyến của diện tích |
| 4 | `district_encoded` | 8,313 | Quận huyện — địa lý quyết định đơn giá/m² |
| 5 | `pub_month` | 2,954 | Thị trường tăng nóng theo tháng — time series có ý nghĩa |
| 6 | `zone_encoded` | 2,348 | Phân khu Inner/Middle/Outer tách biệt rõ ràng |
| 7 | `bedroom_count` | 1,796 | Số phòng ngủ — proxy của quy mô căn hộ |
| 8 | `quality_score` | 577 | **Text Feature #1** — chứng minh bước 3 đúng đắn |
| 9 | `has_legal_paper` | 492 | Sổ đỏ/Pháp lý = Yên tâm = Giá cao hơn |
| 10 | `has_premium_amenities` | 476 | Hồ bơi/Gym/Sân chơi tạo premium rõ rệt |
| 11 | `balcony_dir_Unknown` | 430 | Biến ban công ẩn — thiếu info = căn hộ phổ thông hơn |
| 12 | `bathroom_count` | 331 | Proxy của quy mô và chất lượng |
| 13 | `feat_near_park` | 231 | Không gian xanh = premium nhỏ |
| 14 | `feat_full_furniture` | 229 | Nội thất đẩy giá bán thực tế |
| 15 | `feat_near_school` | 207 | Confounding: nhiều trường ở vùng ngoại thành rẻ hơn |

![Feature Importance — Cluster highlight đỏ](plots_section_5/lightgbm_01_feature_importance.png)

**Phát hiện cốt lõi từ Feature Importance:**
1. **`Cluster` xếp hạng #1 (Gain = 44,318)** — Đây là bằng chứng mạnh nhất rằng pipeline K-Means → LightGBM tạo ra giá trị thực. Nhãn phân khúc encode thông tin tổng hợp từ nhiều biến mà không feature đơn lẻ nào chứa được.
2. **Diện tích & Vị trí vẫn chi phối 50%+ quyền lực** truyền thống → Đúng với quy luật BĐS thực tế.
3. **Text Features (quality_score, amenities, legal_paper) TOP 8–10** → Xác nhận việc đầu tư kỹ thuật vào Feature Engineering ở Step 3 đã trả quả xứng đáng.
4. **Hướng ban công (`balcony_dir_Unknown`) góp mặt** → Việc chuyển từ house_direction sang balcony_direction là quyết định đúng đắn.

### 5B.6 Phân tích Sai số theo Cụm (Error Analysis by Cluster)

Để hiểu sâu hơn, nhóm phân tích sai số dự đoán của mô hình tích hợp theo từng cụm K-Means:

| Cụm | Số lượng test | MAE (Tỷ VND) | Sai số Median (Tỷ) | Giá TB (Tỷ) | Nhận xét |
|---|---|---|---|---|---|
| **Cụm 0** — Tầm Trung | 6.669 | 0.6291 | 0.4779 | 5.29 | Dự đoán tốt — thị trường ổn định |
| **Cụm 1** — Premium | 5.602 | 1.2972 | 0.9009 | 10.06 | Sai số lớn nhất — phân khúc cao cấp có nhiều biến động |
| **Cụm 2** — Phổ Thông | 2.250 | 0.4597 | 0.2786 | 3.82 | Dự đoán chính xác nhất — phân khúc đồng nhất |

**Diễn giải:** Cụm Premium (giá TB 10 tỷ) có sai số gấp gần 3 lần Cụm Phổ Thông. Điều này hợp lý vì căn hộ cao cấp chịu ảnh hưởng từ nhiều yếu tố phi dữ liệu hơn (thương hiệu CĐT, view tầng cao, nội thất cá nhân hóa...) mà dataset chưa thu thập được.

![Boxplot sai số theo cụm](plots_section_5/lightgbm_03_error_by_cluster.png)

### 5B.7 Deep Dive: Mô hình Toàn cục vs. Mô hình Chuyên biệt (Specialized Models)

Để giải quyết "điểm mù" tại phân khúc Premium, nhóm thực hiện một thử nghiệm chuyên sâu: **Chia để trị**. Thay vì dùng một mô hình cho tất cả, nhóm huấn luyện 3 mô hình LightGBM riêng biệt cho 3 cụm.

**Kết quả đối soát MAE (Tỷ VND):**

| Cluster | Mô hình Toàn cục (Global) | Mô hình Chuyên biệt (Specialized) | Kết quả |
|---|---|---|---|
| **Cụm 0** (Tầm trung) | **0.563** | 0.592 | Global tốt hơn |
| **Cụm 1** (Premium) | **1.149** | 1.228 | Global tốt hơn |
| **Cụm 2** (Phổ thông) | **0.405** | 0.449 | Global tốt hơn |

**Phát hiện thú vị (Knowledge Discovery):**
Trái với dự đoán ban đầu, **Mô hình Toàn cục (Global Model) chiến thắng tuyệt đối**. 
- **Lý do 1 (Data Volume):** Mô hình toàn cục được học từ 72.604 bản ghi, giúp nó nắm bắt được các quy luật chung của thị trường (ví dụ: ảnh hưởng của hướng nhà, tiện ích) mà các mô hình nhỏ lẻ không có đủ dữ liệu để học sâu.
- **Lý do 2 (Feature Engineering):** Việc đưa nhãn `Cluster` vào làm categorical feature đã là quá đủ để LightGBM tự điều chỉnh logic dự báo cho từng phân khúc mà không cần tách rời dữ liệu.

![MAE Comparison](plots_section_5/deep_dive_comparison_mae.png)

---

## Kết Luận Bước 5 & Định Hướng Tiếp Theo

### Thành tựu đã đạt được

| Tiêu chí | Kết quả | Đánh giá |
|---|---|---|
| Đáp ứng yêu cầu Unsupervised | ✅ K-Means Clustering | Phát hiện 3 phân khúc thị trường ẩn |
| Đáp ứng yêu cầu Supervised | ✅ LightGBM Regression | Dự đoán giá R² = 0.85 |
| **Pipeline liên kết 5A → 5B** | ✅ Cluster làm feature | **Cải thiện R² +3.81%, giảm MAE 103 triệu** |
| Deep Dive: Global vs Spec | ✅ Global Model Winner | **Dữ liệu lớn + Nhãn phân cụm mang lại hiệu quả cao nhất** |
| Feature Importance Discovery | ✅ Cluster xếp hạng #1 | K-Means tạo ra tri thức mạnh nhất cho model |
| Giải thích được (Explainability) | ✅ Feature Importance + Error by Cluster | AI không còn là hộp đen |

### Bài học phương pháp luận

> **"Unsupervised không phải điểm cuối — mà là điểm khởi đầu cho Supervised."**
>
> Việc K-Means phát hiện 3 phân khúc thị trường không chỉ có giá trị khám phá (discovery) mà còn có giá trị ứng dụng trực tiếp: khi được "chuyển giao" sang LightGBM dưới dạng categorical feature, nó trở thành biến quyền lực #1 — encode được thông tin tương tác đa chiều mà không biến đơn lẻ nào chứa nổi.
>
> Đây là minh chứng cho triết lý **Knowledge-Driven Pipeline**: mỗi bước trong quy trình không tồn tại cô lập mà phải kế thừa và khuếch đại tri thức từ bước trước.

### Định hướng tiếp theo (Nếu muốn nâng cấp thêm)

**Hướng 1 — Demo tương tác (Streamlit App):**
Lưu mô hình LightGBM thành file `.pkl`, xây dựng giao diện web đơn giản bằng Streamlit để giảng viên nhập thông số căn hộ và nhận ngay kết quả dự đoán giá.

**Hướng 2 — Chuyên biệt hóa Model theo Cluster:**
Thay vì 1 model chung, xây 3 model LightGBM riêng cho từng cụm → kiểm chứng xem model chuyên biệt có giảm sai số hơn model toàn cục hay không (đặc biệt ở Cụm Premium đang có MAE cao).

**Hướng 3 — Cross-Validation & Fine-tuning:**
Áp dụng K-Fold Cross Validation (k=5) cho LightGBM để đánh giá độ ổn định mô hình trên nhiều tập dữ liệu khác nhau, đảm bảo R² không phụ thuộc vào cách chia ngẫu nhiên Train/Test.
