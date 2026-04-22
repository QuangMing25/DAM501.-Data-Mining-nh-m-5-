# Section 3 — Data Pre-processing and Transformation (10%)

> **Scripts:**
> - `section_3_preprocessing.py` — pipeline tiền xử lý chính (14 bước: 0–13)
> - `convert_parquet.py` — tiện ích chuyển đổi dữ liệu gốc từ `.parquet` sang `.csv`
>
> **Input:** `data/hanoi_apartments_cleaned.csv` — 86.601 bản ghi × 19 cột
>
> **Output:**
> - `data/hanoi_apartments_processed.csv` — **72.604 bản ghi × 37 cột** (dùng cho EDA và LightGBM)
> - `data/hanoi_apartments_for_clustering.csv` — 72.604 bản ghi × 8 cột (6 scaled + 2 label)
> - `../app_models/le_district.pkl` — LabelEncoder cho quận (dùng cho Web App)
> - `../app_models/le_zone.pkl` — LabelEncoder cho zone (dùng cho Web App)
> - `../app_models/scaler.pkl` — StandardScaler đã fit (dùng cho Web App)
> - Các biểu đồ lưu tại `plots/section_3/`

---

## Tổng quan pipeline tiền xử lý (14 bước)

| Bước | Phân nhóm | Thao tác | Phân tích & Trọng tâm rút ra | Kết quả |
|---|---|---|---|---|
| 0 | Loading data | Đọc dữ liệu đầu vào | Đọc dữ liệu đầu vào, kiểm tra shape & cột. | 86.601 × 19 |
| 1 | Handling missing values | Phân tích missing values | Xác định được 3 nhóm missing: cao (50%), thấp (14%), nhỏ (< 5%). | Lập bảng tỷ lệ khuyết |
| 2 | Data cleaning and noise reduction | Loại bỏ cột không giá trị phân tích | Loại bỏ các cột không có giá trị phân tích: `name`, `province_name`, `property_type_name`. Tách riêng cột `description` để phân rã ở Bước 9 do có nhiều thông tin giá trị, có thể làm giàu dữ liệu. | Xóa 3 cột |
| 3 | Handling missing values | Xử lý missing values | Loại bỏ các hàng thiếu giá trị mà không thể xử lý; Điền theo median theo quận hoặc giá trị "Unknown" cho các cột còn thiếu. | Xóa 12.157 hàng → 74.444 × 10 |
| 4 | Detecting and treating outliers | Phát hiện và loại outliers | Phát hiện 3 cột `price`, `area`, `bedroom_count` có giá trị bất thường. Áp dụng IQR với IQR = 3.0 để loại bỏ các giá trị "chắc chắn bất thường" với giá trị của `price`, `area`. Đồng thời loại bỏ những căn hộ có số phòng vô lý với giá trị `bedroom_count`. | Xóa 1.840 hàng (2,5%) → 72.604 |
| 5 | Feature transformation or discretization | Feature Engineering | Tạo features mới: `price_per_m2`, phân khu lõi/nội/ngoại thành (phân thành 3 zone), tách thời gian đăng theo tháng và năm | Tạo 3 biến mới |
| 6 | Feature transformation or discretization | Biến đổi Logarithm | Các giá trị trong cột `price`, `area`, `price_per_m2` bị lệch phải mạnh, cần áp dụng hàm Logarithm để đưa phân phối về dạng hình chuông (Normal Distribution) cân đối hơn, giảm sự ảnh hưởng của những giá trị cực lớn. Đồng thời tăng độ chính xác khi áp dụng K-Means. | 3 cột chuyển giá trị sau log; skewness giảm mạnh |
| 7 | Encoding categorical variables | Encoding categorical | Chuyển các giá trị dạng text (category) thành số để K-Means và LightGBM có thể xử lý được, đồng thời bảo toàn ý nghĩa thực sự của từng features. Áp dụng Label Encoding cho cột `district_name` và `district_zone`, áp dụng One-Hot Encoding cho `balcony_direction`. | +2 cột label, + 8 cột OHE |
| 8 | Feature transformation or discretization | Chuẩn hóa cho K-Means | Chuẩn hóa 6 features để K-Means không bị lệch tỷ lệ, đảm bảo kết quả phân cụm chính xác | Ma trận 6 cột scaled |
| 9 | Data aggregation or transaction construction | Text Feature Extraction | Phân rã cột `description` thành 9 biến binary + `quality_score`: mỗi tin đăng có tập thuộc tính rõ ràng. Đây là bước làm giàu thêm dữ liệu. | +10 cột → 72.604 × 37 |
| 10 | Summary | Final summary | Đánh giá ma trận dữ liệu sau tiền xử lý, thống kê mô tả các features chính, đánh giá chất lượng Feature Engineering. | 72.604 × 37, 0% missing |
| 11 | Summary | Lưu output & Export models | Export 2 file CSV làm đầu vào cho K-Means và LighGBM;  Export 3 file PKL cho Web App mô phỏng. | 2 CSV + 3 PKL |
| 12 | Summary | Vẽ biểu đồ | Tổng hợp và trực quan hoá phân phối, tương quan, tác động features. | 5 PNG |

---

## Bước 0. Load Raw Data

**Thao tác:** Đọc file `data/hanoi_apartments_cleaned.csv` bằng `pd.read_csv` với encoding `utf-8-sig`.

**Kết quả:**

| Chỉ số | Giá trị |
|---|---|
| Số bản ghi | 86.601 |
| Số cột | 19 |

**19 cột đầu vào:**

| Nhóm | Cột |
|---|---|
| Định danh / Mô tả | `name`, `description`, `province_name`, `property_type_name` |
| Địa lý | `district_name`, `ward_name`, `street_name`, `project_name` |
| Đặc điểm căn hộ | `area`, `bedroom_count`, `bathroom_count`, `floor_count`, `frontage_width`, `road_width`, `house_depth` |
| Hướng | `house_direction`, `balcony_direction` |
| Giá & Thời gian | `price`, `published_at` |

---

## Bước 1. Đánh giá Mức độ Thiếu hụt Dữ liệu (Missing Values Analysis)

**Thao tác:** Tính `isnull().sum()` và tỷ lệ `%` cho tất cả 19 cột, sắp xếp giảm dần.

**Kết quả — 3 nhóm missing:**

| Nhóm | Cột | Tỷ lệ thiếu | Ghi chú |
|---|---|---|---|
| Missing tuyệt đối | `house_depth`, `road_width`, `floor_count`, `frontage_width` | > 99% | Trường nhập cho đất nền, không áp dụng cho chung cư |
| Missing cao | `balcony_direction` | ~50% | Môi giới hay bỏ qua hoặc nhập sai |
| Missing thấp | `price` | ~14% | Tin đăng thiếu giá cần loại bỏ |
| Missing nhỏ | `bedroom_count`, `bathroom_count`, `ward_name`, `street_name`, `project_name` | < 5% | Có thể xử lý bằng imputation |

**Quyết định thiết kế:**
1. Với đặc thù chung cư, `frontage_width`, `road_width`, `house_depth` là các trường vô nghĩa — bị ép nhập bởi Batdongsan.vn theo template đất nền. Loại bỏ toàn bộ ở Bước 2.
2. `balcony_direction` thiếu ~50% nhưng **không loại** — thay vào đó fill `"Unknown"` rồi OHE ở Bước 8, giữ nguyên khối lượng 86K bản ghi.

---

## Bước 2. Loại bỏ Cột thiếu >= 95%

**Thao tác:** Tự động phát hiện bằng `missing_pct[missing_pct >= 95].index`, sau đó `df.drop()`.

**Kết quả:**

| Cột bị xóa | Lý do |
|---|---|
| `house_depth` | 99%+ missing — trường đất nền |
| `road_width` | 99%+ missing — trường đất nền |
| `floor_count` | 99%+ missing — thường để trống cho chung cư |
| `frontage_width` | 99%+ missing — trường đất nền |

**Shape sau bước 2:** 86.601 × 15

---

## Bước 3. Loại bỏ Cột không có Giá trị Phân tích

**Thao tác:** Tách riêng cột `description` ra biến `descriptions_raw` trước, sau đó `df.drop()` toàn bộ 5 cột không cần thiết.

**Kết quả:**

| Cột bị xóa | Lý do |
|---|---|
| `name` | Free text tiêu đề — không có giá trị phân tích định lượng |
| `description` | Free text — **tách sang `descriptions_raw`** để dùng ở Bước 10, không dùng trực tiếp |
| `province_name` | Zero variance — toàn bộ là "Hà Nội" |
| `property_type_name` | Zero variance — toàn bộ là "Chung cư" |

**Bước ngoặt xử lý Direction:** Trước đây `house_direction` đóng vai trò trung tâm nhưng cho ra kết quả thống kê sai do môi giới thường nhập hướng cửa lớn toà nhà thay vì hướng cửa chính căn hộ. Với chung cư, **`balcony_direction` (hướng ban công)** mới là chỉ số phản ánh đúng điều kiện vi khí hậu thực tế. `house_direction` bị loại hoàn toàn tại bước này; `balcony_direction` được xử lý OHE ở Bước 8.

**Shape sau bước 3:** 86.601 × 10

---

## Bước 4. Xử lý Missing Values

**Thao tác:** Ba chiến lược xử lý tương ứng ba loại cột còn thiếu.

**Kết quả chi tiết:**

| Cột | Cách thực hiện | Kết quả |
|---|---|---|
| `price` | **Xóa hàng** — thiếu giá không thể impute | Xóa 12.157 hàng → còn 74.444 |
| `bedroom_count` | **Median theo quận** (hoặc median theo dataset) | Fill ~vài trăm hàng |
| `bathroom_count` | **Median theo quận** (hoặc median theo dataset) | Fill ~vài trăm hàng |
| `ward_name` | **Fill "Unknown"** | Các tin không rõ phường |
| `street_name` | **Fill "Unknown"** | Các tin không rõ đường |
| `project_name` | **Fill "Unknown"** | Các tin không thuộc dự án |
| `balcony_direction` | **Fill "Unknown"** | ~50% bản ghi → nhận giá trị "Unknown" |

**Lý do dùng Median theo quận:** Mỗi quận có mức giá và quy mô căn hộ khác nhau rõ rệt. Dùng median toàn tập sẽ làm sai lệch đặc trưng địa lý — ví dụ: căn hộ Hoàn Kiếm 1 phòng ngủ sẽ bị fill thành 2 (median toàn quốc) nếu không chia theo quận.

**Shape sau bước 4:** 74.444 × 10, 0% missing

---

## Bước 5. Phát hiện và Loại bỏ Outliers

**Thao tác:** Áp dụng IQR với hệ số **factor = 3.0** (rộng hơn 1.5 thông thường) để chỉ loại các điểm thực sự bất thường, không làm mất dữ liệu hợp lệ ở đuôi phân phối.

**Ngưỡng IQR cho từng biến:**

| Biến | Công thức | Giá trị ngưỡng | Ghi chú |
|---|---|---|---|
| `price` | IQR × 3, sàn tối thiểu 100 triệu, trần tối đa 100 tỷ | **[0.1 tỷ — 21.7 tỷ VND]** | Loại giá ảo / nhập sai đơn vị |
| `area` | IQR × 3, sàn tối thiểu 15 m², trần tối đa 500 m² | **[15 m² — 500 m²]** | Loại căn quá nhỏ/quá lớn bất thường |
| `bedroom_count` | Hard bound | **[1 — 10 phòng]** | Loại giá trị 0 hoặc > 10 |

**Kết quả:**

| Chỉ số | Giá trị |
|---|---|
| Số hàng bị loại | 1.840 hàng |
| Tỷ lệ loại | 2,5% |
| Shape sau bước 5 | **72.604 × 10** |

**Lý do chọn factor = 3.0:** Thị trường bất động sản có phân phối lệch phải nặng (right-skewed). Với factor = 1.5 sẽ loại quá nhiều căn hộ cao cấp hợp lệ ở Hoàn Kiếm, Ba Đình. Factor = 3.0 giữ được đầu phân phối cao trong khi vẫn loại được các giá trị nhập sai đơn vị (ví dụ: 15 triệu thay vì 15 tỷ).

---

## Bước 6. Feature Engineering

**Thao tác:** Tạo 4 biến mới từ các cột hiện có.

**Kết quả — 4 biến mới:**

| Biến mới | Công thức | Ý nghĩa |
|---|---|---|
| `price_per_m2` | `price / area` | Hệ quy chiếu chuẩn hoá giá theo diện tích — loại bỏ confounding của area |
| `pub_month` | `published_at.dt.month` | Yếu tố thời vụ — giá bất động sản biến động theo tháng |
| `pub_year` | `published_at.dt.year` | Yếu tố năm — bắt xu hướng dài hạn |
| `district_zone` | Phân loại 3 vòng theo quận | Proxy địa lý vĩ mô: `inner` / `middle` / `outer` |

**Phân loại `district_zone`:**

| Zone | Quận thuộc | Đặc điểm |
|---|---|---|
| `inner` | Hoàn Kiếm, Ba Đình, Đống Đa, Hai Bà Trưng | Lõi đô thị, giá cao nhất |
| `middle` | Tây Hồ, Cầu Giấy, Thanh Xuân, Hoàng Mai, Long Biên, Bắc Từ Liêm, Nam Từ Liêm, Hà Đông | Vành đai đô thị, chiếm > 75% dữ liệu |
| `outer` | Các quận/huyện còn lại | Ngoại thành, giá thấp hơn |

**Thống kê phân phối zone sau bước 6:**

> Middle Zone chiếm tỷ trọng áp đảo (~75%), phản ánh đúng thực tế thị trường Hà Nội tập trung mạnh ở khu vực Cầu Giấy–Thanh Xuân–Hoàng Mai–Hà Đông.

---

## Bước 7. Log Transformation

**Thao tác:** Áp dụng `np.log1p()` cho 3 cột có phân phối lệch phải nặng.

**Kết quả — Cải thiện skewness:**

| Cột gốc | Skewness trước | Cột log | Skewness sau |
|---|---|---|---|
| `price` | lệch phải cao | `log_price` | ≈ **–0.16** (gần chuẩn) |
| `area` | lệch phải | `log_area` | giảm mạnh |
| `price_per_m2` | lệch phải | `log_price_per_m2` | giảm mạnh |

**Lý do quan trọng:**
- Thuật toán LightGBM (tree-based) ít bị ảnh hưởng bởi skewness, nhưng log transform giúp target variable `log_price` có phân phối cân đối hơn → MAE và RMSE trên thang log phản ánh sai số tương đối (%), trực quan hơn sai số tuyệt đối trên thang tỷ VND.
- K-Means dùng `log_price` và `log_area` để khoảng cách Euclidean không bị chi phối bởi outlier ở đuôi phân phối.

---

## Bước 8. Encoding Categorical Variables

**Thao tác:** Áp dụng hai phương pháp encoding khác nhau cho từng loại biến.

**Kết quả:**

| Biến | Phương pháp | Chi tiết | Số cột tạo ra |
|---|---|---|---|
| `district_name` | **Label Encoding** | 19 quận → giá trị nguyên [0–18], lưu vào `le_district.pkl` | 1 cột `district_encoded` |
| `district_zone` | **Label Encoding** | 3 zone → [0–2], lưu vào `le_zone.pkl` | 1 cột `zone_encoded` |
| `balcony_direction` | **One-Hot Encoding** | Mỗi hướng → 1 cột binary; cột `balcony_dir_Unknown` bị xóa | ~8 cột `balcony_dir_*` |

**Xử lý kỹ thuật OHE:**
- Trước khi OHE, chuỗi `" - "` trong tên hướng được thay bằng `" "` (ví dụ: `"Đông - Nam"` → `"Đông Nam"`) để tránh tạo ra các cột thừa do format không nhất quán.
- Cột `balcony_dir_Unknown` bị xóa sau OHE: các hàng có hướng Unknown nhận giá trị `0` ở toàn bộ cột hướng — đây là cách mã hoá tham chiếu ngầm (reference category) đúng chuẩn.
- Cột `balcony_direction` gốc bị xóa sau khi OHE xong.

---

## Bước 9. Chuẩn hóa cho K-Means (StandardScaler)

**Thao tác:** Fit `StandardScaler` trên 6 features được chọn cho clustering, tạo ra ma trận scaled riêng.

**6 features được chuẩn hóa:**

| Feature | Ý nghĩa | Lý do đưa vào clustering |
|---|---|---|
| `log_price` | Giá log | Biến target chính |
| `log_area` | Diện tích log | Quy mô căn hộ |
| `bedroom_count` | Số phòng ngủ | Loại hình căn hộ |
| `bathroom_count` | Số phòng tắm | Cấp độ tiện nghi |
| `log_price_per_m2` | Giá/m² log | Hiệu suất đầu tư |
| `district_encoded` | Mã quận | Vị trí địa lý |

**Kết quả:** Ma trận `df_scaled` có shape **72.604 × 6**, các cột có tên `scaled_log_price`, `scaled_log_area`, v.v.

**Lý do dùng StandardScaler cho K-Means:** K-Means tính khoảng cách Euclidean — nếu không chuẩn hoá, `log_price` (tầm [15–24]) sẽ lấn át hoàn toàn `bedroom_count` (tầm [1–5]), khiến cluster chỉ phản ánh giá mà bỏ qua các chiều thông tin còn lại.

---

## Bước 10. Text Feature Extraction từ Description

**Thao tác:** Keyword matching bằng regex trên `descriptions_raw` (đã lưu từ Bước 3), tạo 9 biến binary + 1 biến tổng hợp `quality_score`.

**Chiến lược lọc — từ 17 features thô xuống 9 features:**
- **Loại 5 features "Hiển nhiên có"** (`thang máy`, `bãi đỗ xe`, `bảo vệ 24/7`, `view đẹp`, `thoáng mát`) — thiếu keyword không có nghĩa là không có tính năng (False Negative diện rộng).
- **Gộp 3 cụm thành 2 features** để giảm đa cộng tuyến:
  - `has_legal_paper` gộp: sổ đỏ + sổ hồng + pháp lý đầy đủ/rõ ràng/sạch
  - `has_premium_amenities` gộp: bể bơi/hồ bơi + gym/phòng tập + sân chơi trẻ em

**Danh sách 9 Text Features cuối cùng:**

| Feature | Tên tiếng Việt | Tần suất | Nhóm |
|---|---|---|---|
| `has_legal_paper` | Giấy tờ Pháp lý | 53.9% | Gộp |
| `feat_full_furniture` | Đầy đủ nội thất | 44.5% | Nội tại |
| `feat_balcony` | Có ban công | 43.9% | Nội tại |
| `feat_near_school` | Gần trường học | 30.9% | Ngoại khu |
| `feat_near_mall` | Gần siêu thị / TTTM | 30.9% | Ngoại khu |
| `feat_near_park` | Gần công viên | 27.3% | Ngoại khu |
| `feat_near_hospital` | Gần bệnh viện | 21.7% | Ngoại khu |
| `has_premium_amenities` | Tiện ích VIP nội khu | 21.1% | Gộp |
| `feat_corner_unit` | Căn góc | 11.6% | Nội tại |

**Biến `quality_score`:** Tổng số features được đề cập trong mô tả — mean **2.86** (tối đa 9). Phản ánh đúng mức độ chi tiết / cao cấp của tin đăng.

**Shape sau bước 10:** 72.604 × 37

---

## Bước 11. Tổng kết Dataset Cuối cùng (Final Summary)

### Profile Ma trận dữ liệu (`hanoi_apartments_processed.csv`)

| Chỉ số | Giá trị |
|---|---|
| Số bản ghi | 72.604 (chắt lọc từ 86.601 bản thô) |
| Số cột | 37 |
| Missing values | 0% |
| Dữ liệu bị loại | 13.997 hàng (16,2%) |

**Phân tích mức hao hụt:**

| Nguyên nhân | Số hàng mất | Giai đoạn |
|---|---|---|
| Thiếu giá (`price` null) | 12.157 | Bước 4 |
| Outliers giá / diện tích / phòng ngủ | 1.840 | Bước 5 |
| **Tổng** | **13.997** | |

### Thống kê mô tả các biến chính

| Biến | Median | Mean | Ghi chú |
|---|---|---|---|
| `price` (tỷ VND) | 6,25 | 6,96 | Right-skewed nhẹ sau khi loại outlier |
| `area` (m²) | 80 | 85,1 | Phổ biến căn 2–3 phòng ngủ |
| `price_per_m2` (triệu/m²) | 77,6 | 81,1 | Nội thành đắt hơn ~2x ngoại thành |
| `bedroom_count` | 2 | 2,3 | Căn 2PN chiếm đa số |
| `bathroom_count` | 2 | 2,0 | — |
| `log_price` skewness | — | — | ≈ –0,16 (rất gần chuẩn) |

### Phân phối District Zone

| Zone | Số tin | Tỷ lệ |
|---|---|---|
| `middle` (Vành đai đô thị) | ~55.111 | >75% |
| `inner` (Lõi nội thành) | ~10.000+ | ~14% |
| `outer` (Ngoại thành) | ~7.000+ | ~10% |

### Đánh giá chất lượng Feature Engineering

- **Kiểm soát chiều dữ liệu:** Giảm từ 51 cột (nếu giữ toàn bộ) xuống còn 37 cột có ý nghĩa thực sự, không làm mất variance.
- **Log Transform hiệu quả:** `log_price` đạt skewness ≈ –0,16 — phân phối gần chuẩn, tối ưu cho cả tree-based model lẫn distance-based clustering.
- **Text Features có tác động rõ:** `has_legal_paper` và `has_premium_amenities` cho thấy chênh lệch giá trung bình đáng kể so với căn không có feature (xem biểu đồ `section3b_feature_price_impact.png`).

---

## Bước 12. Lưu Output và Export Models cho Web App

**Thao tác:** Hai nhiệm vụ song song — lưu dữ liệu downstream và export artifacts cho inference.

**1. Lưu dữ liệu đã xử lý (2 CSV):**

| File | Shape | Nội dung | Dùng cho |
|---|---|---|---|
| `hanoi_apartments_processed.csv` | 72.604 × 37 | Toàn bộ features (bỏ cột `scaled_*`) | EDA (Step 4) + LightGBM (Step 5b) |
| `hanoi_apartments_for_clustering.csv` | 72.604 × 8 | 6 cột scaled + `district_name` + `district_zone` | K-Means (Step 5a) |

**2. Export artifacts sang `../app_models/` (3 pkl):**

| File | Nội dung | Mục đích trong Web App |
|---|---|---|
| `le_district.pkl` | LabelEncoder đã fit trên 19 quận | Encode quận do user nhập khi dự đoán |
| `le_zone.pkl` | LabelEncoder đã fit trên 3 zone | Encode zone tương ứng tự động |
| `scaler.pkl` | StandardScaler đã fit trên 6 features | Scale input trước khi đưa vào K-Means |

**Lý do phải lưu pkl (không fit lại):** Encoder và Scaler phải được fit **đúng một lần** trên toàn bộ tập train. Nếu Web App khởi tạo encoder mới từ đầu khi nhận user input (chỉ có 1 sample), kết quả encode/scale sẽ khác hoàn toàn → model trả về dự đoán sai hoàn toàn.

---

## Bước 13. Sinh Biểu đồ Phân tích (Plots)

**Thao tác:** Tạo 5 biểu đồ PNG lưu vào `plots/section_3/` phục vụ báo cáo và trình bày.

---

### Plot 1 — `section3_distributions.png`: Key Distributions (Grid 2×3)

![section3_distributions](plots/section_3/section3_distributions.png)

Biểu đồ tổng hợp 6 góc nhìn về phân phối dữ liệu sau tiền xử lý.

**Hàng trên — Phân phối 3 biến số chính:**

| Sub-plot | Quan sát | Kết luận |
|---|---|---|
| **(1) Price Distribution (tỷ VND)** | Lệch phải rõ rệt, đỉnh tập trung ở 3–5 tỷ, đuôi kéo dài đến 20+ tỷ | Phân phối điển hình của thị trường BĐS — giá trị trung bình bị kéo lên bởi nhóm cao cấp |
| **(2) Log(Price) Distribution** | Gần chuẩn (bell curve), tâm quanh giá trị ~22 (log scale), đối xứng hơn rõ rệt | Log Transform thành công — xác nhận `log_price` là target phù hợp cho ML |
| **(3) Area Distribution (m²)** | Lệch phải, đỉnh ở 50–80 m², đuôi trải đến 200 m² | Căn hộ Hà Nội phổ biến diện tích trung bình, số căn > 150 m² rất ít |

**Hàng dưới — Phân tích theo phân nhóm:**

| Sub-plot | Quan sát | Kết luận |
|---|---|---|
| **(4) Giá/m² theo Zone (Boxplot)** | `inner`: median cao nhất, IQR rộng, nhiều outlier phía trên (tới ~400 triệu/m²). `middle`: median thấp hơn, phân tán vừa. `outer`: median thấp nhất, IQR hẹp | Xác nhận `district_zone` là biến phân loại có ý nghĩa — giá/m² giảm dần rõ từ lõi ra ngoại thành |
| **(5) Phân phối số phòng ngủ** | Căn 2PN chiếm đỉnh (~30.000+ tin), kế tiếp là 3PN, rồi 1PN; căn 4+ PN rất hiếm | Thị trường Hà Nội tập trung mạnh ở phân khúc 2–3 phòng ngủ — phù hợp với nhu cầu gia đình hạt nhân |
| **(6) Top 12 Quận theo số tin đăng** | Nam Từ Liêm dẫn đầu vượt trội (~12.000+ tin), tiếp theo là Cầu Giấy, Hoàng Mai, Hà Đông, Thanh Xuân; các quận nội thành cũ (Tây Hồ, Hai Bà Trưng) ít tin hơn hẳn | Cung cấp bằng chứng về sự chênh lệch dữ liệu địa lý — các quận phát triển mạnh gần đây (Nam Từ Liêm, Cầu Giấy) đang áp đảo về số lượng tin |

**Nhận xét tổng thể Plot 1:** Cặp sub-plot (1)–(2) là minh chứng trực quan nhất cho tính hiệu quả của Log Transform. Boxplot Zone (4) xác nhận thiết kế biến `district_zone` là hợp lý. Biểu đồ (6) cảnh báo về mất cân bằng dữ liệu địa lý — model cần lưu ý khi đánh giá hiệu năng trên từng quận.

---

### Plot 2 — `section3_correlation.png`: Correlation Matrix

![section3_correlation](plots/section_3/section3_correlation.png)

Ma trận tương quan Pearson giữa 6 biến số sau tiền xử lý.

**Các cặp tương quan nổi bật:**

| Cặp biến | Hệ số r | Diễn giải |
|---|---|---|
| `price` ↔ `area` | **0.77** | Tương quan mạnh — diện tích là predictor quan trọng nhất của giá tuyệt đối |
| `area` ↔ `bedroom_count` | **0.75** | Tương quan mạnh — căn càng nhiều phòng càng có diện tích lớn, đúng logic |
| `price` ↔ `price_per_m2` | **0.65** | Tương quan khá — giá cao thường đi kèm mật độ giá/m² cao (khu vực đắt đỏ) |
| `price` ↔ `bedroom_count` | **0.56** | Tương quan vừa — số phòng ảnh hưởng giá nhưng kém hơn diện tích |
| `area` ↔ `bathroom_count` | **0.58** | Tương quan vừa — căn lớn hơn thường có nhiều phòng tắm hơn |
| `bedroom_count` ↔ `bathroom_count` | **0.58** | Tương quan vừa — cấu hình phòng ngủ/tắm thường đi theo nhau |
| `price` ↔ `bathroom_count` | **0.46** | Tương quan thấp-vừa — số phòng tắm có đóng góp nhưng không là yếu tố chính |
| `area` ↔ `price_per_m2` | **0.07** | Gần zero — xác nhận `price_per_m2` là biến độc lập, không phụ thuộc diện tích |
| `district_encoded` ↔ tất cả | **≈ 0.02–0.03** | Gần zero — do Label Encoding gán số tuỳ ý cho quận, không mang thứ tự giá trị thực |

**Nhận xét tổng thể Plot 2:**
- **Không có đa cộng tuyến nghiêm trọng** giữa các features: cặp cao nhất là `area`–`price` (0.77) và `area`–`bedroom_count` (0.75), nhưng cả hai vẫn trong ngưỡng chấp nhận được với tree-based model (LightGBM không nhạy cảm với multicollinearity).
- **`price_per_m2` độc lập với `area`** (r=0.07): thiết kế biến này đúng — nó đo chất lượng địa điểm/loại hình, không phản ánh kích thước căn hộ.
- **`district_encoded` r≈0.03** là hệ quả của Label Encoding ordinal — vì vậy model nên dùng `district_encoded` chỉ để phân biệt quận, không khai thác mối quan hệ tuyến tính. Đây là lý do clustering dùng khoảng cách Euclidean vẫn hợp lý.

---

### Plot 3 — `section3b_feature_price_impact.png`: Tác động của Text Features đến Giá

![section3b_feature_price_impact](plots/section_3/section3b_feature_price_impact.png)

Biểu đồ đo lường chênh lệch giá trung bình (%) giữa nhóm có và không có từng text feature.

**Kết quả từng feature (sắp xếp giảm dần):**

| Feature | Tác động | Diễn giải |
|---|---|---|
| **Căn góc** | **+17.6%** | Tác động lớn nhất — căn góc có 2 mặt thoáng, view rộng, được thị trường định giá cao rõ rệt |
| **Ban công** | +7.1% | Ban công là yếu tố tiện nghi quan trọng — không gian ngoài trời trong chung cư luôn được trả thêm |
| **Tiện ích cao cấp** | +6.1% | Hồ bơi/gym/sân chơi là đặc trưng của dự án mid-to-high end — phản ánh phân khúc giá cao |
| **Gần siêu thị/TTTM** | +5.3% | Tiện lợi mua sắm là yếu tố định giá được thị trường ghi nhận rõ ràng |
| **Gần công viên** | +3.4% | Không gian xanh = chất lượng sống cao hơn, nhưng tác động thấp hơn thương mại |
| **Nội thất đầy đủ** | +1.8% | Đáng ngạc nhiên là thấp — môi giới có thể cường điệu "full nội thất" khiến tín hiệu bị nhiễu |
| **Pháp lý rõ ràng** | +1.4% | Tác động thấp — do pháp lý là điều kiện cơ bản, căn không có sổ đỏ thường ở dự án đang hoàn thiện thủ tục, giá đã được thị trường chiết khấu sẵn |
| **Gần bệnh viện** | +0.4% | Gần như trung tính — lợi ích y tế không phải yếu tố định giá chính |
| **Gần trường học** | **–4.6%** | Dấu âm — kết quả ngược trực giác, xem giải thích bên dưới |

**Giải thích kết quả âm của "Gần trường học" (–4.6%):**

Đây là kết quả thú vị cần phân tích thêm. Có 2 giả thuyết:
1. **Confounding địa lý:** Nhiều trường học tập trung ở các quận ngoại thành đang phát triển (Gia Lâm, Bắc Từ Liêm, Hà Đông) — nơi có mật độ tin đăng cao nhưng giá thấp hơn quận nội thành. Tín hiệu "gần trường" bị lẫn với tín hiệu "giá thấp" của vùng ngoại vi.
2. **Keyword overfitting:** "Trường học", "gần trường" là từ khoá phổ biến trong mô tả marketing, không nhất thiết phản ánh vị trí thực sự đặc biệt.

→ **Hàm ý cho model:** `feat_near_school` vẫn được giữ lại trong pipeline vì tần suất xuất hiện cao (30.9%) và phản ánh thông tin mô tả thực — nhưng LightGBM sẽ tự phát hiện tương tác này và xử lý phù hợp trong quá trình huấn luyện.

---

### Plot 4 — `section3b_feature_frequency.png`: Tần suất Text Features trong Mô tả

![section3b_feature_frequency](plots/section_3/section3b_feature_frequency.png)

Tỷ lệ % tin đăng có đề cập đến từng text feature trong cột `description` (tổng 72.604 tin).

**Kết quả và phân tích:**

| Nhóm | Feature | Tần suất | Phân tích |
|---|---|---|---|
| **Nhóm phổ biến (>40%)** | Pháp lý rõ ràng | 53.9% | Feature phổ biến nhất — môi giới dùng thông tin pháp lý như công cụ tạo niềm tin |
| | Nội thất đầy đủ | 44.5% | Gần một nửa tin đăng nhấn mạnh nội thất — xu hướng bán căn hộ "sẵn ở" |
| | Ban công | 43.9% | Ban công được liệt kê như điểm bán hàng (selling point) quan trọng |
| **Nhóm trung bình (20–35%)** | Gần siêu thị/TTTM | 30.9% | Tiện ích thương mại là điểm nhấn marketing phổ biến |
| | Gần trường học | 30.9% | Tương đương siêu thị — tập trung ở tin đăng hướng đến gia đình có con |
| | Gần công viên | 27.3% | Không gian xanh được quảng bá nhưng ít phổ biến hơn tiện ích thương mại |
| | Gần bệnh viện | 21.7% | Đề cập vừa phải — thường xuất hiện ở khu vực quanh bệnh viện lớn |
| | Tiện ích cao cấp | 21.1% | Phù hợp tỷ lệ dự án mid-to-high end trong tập dữ liệu |
| **Nhóm hiếm (<15%)** | Căn góc | 11.6% | Thấp nhưng có nghĩa — căn góc thực sự chiếm thiểu số trong bất kỳ toà nhà nào |

**Đối chiếu Tần suất ↔ Tác động Giá (so với Plot 3):**

Kết hợp hai biểu đồ tạo ra insight quan trọng:
- **Căn góc (11.6%, +17.6%):** Hiếm + tác động cao → đây là tín hiệu chất lượng thực, không phải marketing
- **Pháp lý (53.9%, +1.4%):** Phổ biến + tác động thấp → đã trở thành yếu tố tiêu chuẩn, ít tạo khác biệt giá
- **Nội thất (44.5%, +1.8%):** Phổ biến + tác động thấp → từ khoá "full nội thất" xuất hiện quá nhiều, bị loãng tín hiệu
- **Tiện ích cao cấp (21.1%, +6.1%):** Tần suất vừa + tác động khá cao → vẫn là differentiator tốt, phản ánh phân khúc thực sự

---

### Plot 5 — `section3b_quality_score.png`: Phân phối Quality Score

![section3b_quality_score](plots/section_3/section3b_quality_score.png)

Phân phối biến `quality_score` — tổng số text features được đề cập trong mô tả của từng tin đăng (thang 0–9).

**Số liệu chi tiết từng mức score:**

| Score | Số tin đăng | Tỷ lệ | Diễn giải |
|---|---|---|---|
| 0 | 6.946 | 9,6% | Mô tả không đề cập bất kỳ feature nào — tin đăng rất sơ sài |
| 1 | 14.335 | 19,7% | Đề cập 1 feature — thường chỉ nêu pháp lý hoặc nội thất |
| 2 | **15.505** | **21,4%** | **Đỉnh phân phối** — đa số tin đăng đề cập đúng 2 features |
| 3 | 11.445 | 15,8% | Mô tả khá đầy đủ |
| 4 | 7.640 | 10,5% | Mô tả tốt |
| 5 | 7.333 | 10,1% | Mô tả chi tiết |
| 6 | 5.838 | 8,0% | Mô tả rất chi tiết |
| 7 | 2.895 | 4,0% | Mô tả toàn diện |
| 8 | 645 | 0,9% | Mô tả rất hiếm — gần như đề cập tất cả |
| 9 | 22 | 0,03% | Đề cập đủ cả 9 features — cực kỳ hiếm |

**Nhận xét tổng thể Plot 5:**

- **Phân phối lệch phải (right-skewed):** Mean 2.86, đỉnh tại Score=2 — đa số tin đăng có mô tả sơ lược. Phù hợp thực tế: môi giới thường chỉ liệt kê 1–3 điểm nổi bật.
- **Nhóm Score 0–2 chiếm ~50% tập dữ liệu:** Tín hiệu cho thấy chất lượng mô tả thị trường BĐS Việt Nam còn thấp — phần lớn tin đăng thiếu thông tin chi tiết.
- **Đuôi Score 7–9 cực nhỏ (<5%):** Chỉ các tin đăng chuyên nghiệp / dự án lớn mới có mô tả đầy đủ. Nhóm này thường tương ứng với phân khúc cao cấp.
- **`quality_score` là biến liên tục có ý nghĩa:** Không chỉ đếm số keywords, nó còn phản ánh mức độ đầu tư của người đăng vào chất lượng nội dung — và theo biểu đồ tác động giá (Plot 3), mỗi feature đều có premium dương (ngoại trừ `feat_near_school`), nên `quality_score` cao đồng nghĩa với xu hướng giá cao hơn.

---

## Tiện ích: `convert_parquet.py`

Script độc lập dùng để chuyển đổi file `.parquet` gốc sang `.csv` — bước chuẩn bị trước khi chạy pipeline chính.

**Cách dùng:**
```bash
python convert_parquet.py                 # Chuyển đổi thông thường
RUN_OPTIONAL=1 python convert_parquet.py  # Kèm bước làm sạch dữ liệu bị lỗi hướng nhà
```

**Tính năng Optional** (`RUN_OPTIONAL=1`): Phát hiện và loại bỏ các hàng có chuỗi hướng nhà bất thường (ví dụ: "Đông Bắc" và "Đông Nam" xuất hiện cùng một ô) bằng regex pattern matching. Xử lý 4 pattern lỗi phổ biến. Kết quả lưu ra `data/aptm_hanoi_2026_cleaned.csv`. Tính năng này chạy độc lập, không ảnh hưởng đến pipeline chính.

---

## Bước 14. Dataset sau khi thực hiện Step 3

Sau toàn bộ pipeline tiền xử lý, Step 3 xuất ra **2 file CSV** phục vụ cho các bước tiếp theo.

---

### File 1: `hanoi_apartments_processed.csv`

> Dùng cho: **Step 4 (EDA)** và **Step 5b (LightGBM)**

**Thông số tổng quan:**

| Chỉ số | Giá trị |
|---|---|
| Số bản ghi | 72.604 |
| Số cột | 36 |
| Missing values | 308 hàng ở `published_at`, `pub_month`, `pub_year` (parse lỗi timestamp) |

**Cấu trúc 36 cột — phân theo nhóm:**

| Nhóm | Cột | Số cột | Kiểu dữ liệu |
|---|---|---|---|
| **Địa lý** | `district_name`, `ward_name`, `street_name`, `project_name`, `district_zone` | 5 | object |
| **Số liệu gốc** | `price`, `area`, `bedroom_count`, `bathroom_count`, `price_per_m2`, `pub_month`, `pub_year` | 7 | float64 |
| **Thời gian** | `published_at` | 1 | object |
| **Log transform** | `log_price`, `log_area`, `log_price_per_m2` | 3 | float64 |
| **Label Encoded** | `district_encoded`, `zone_encoded` | 2 | int64 |
| **OHE Hướng ban công** | `balcony_dir_Bắc`, `balcony_dir_Nam`, `balcony_dir_Tây`, `balcony_dir_Tây Bắc`, `balcony_dir_Tây Nam`, `balcony_dir_Đông`, `balcony_dir_Đông Bắc`, `balcony_dir_Đông Nam` | 8 | bool |
| **Text features (binary)** | `feat_full_furniture`, `feat_corner_unit`, `has_legal_paper`, `has_premium_amenities`, `feat_near_school`, `feat_near_hospital`, `feat_near_mall`, `feat_near_park`, `feat_balcony` | 9 | int64 |
| **Quality score** | `quality_score` | 1 | int64 |

**Thống kê mô tả các cột số liệu chính:**

| Cột | Min | Mean | Max | Ghi chú |
|---|---|---|---|---|
| `price` | 100.000.000 | 6.955.901.000 | 21.700.000.000 | Đơn vị: VND |
| `area` | 15,0 | 85,1 | 221,0 | Đơn vị: m² |
| `bedroom_count` | 1 | 2,43 | 10 | Phổ biến: 2PN (35.088), 3PN (29.903) |
| `bathroom_count` | 1 | 1,90 | 9 | Phổ biến: 2WC (58.377) |
| `price_per_m2` | 843.750 | 81.080.780 | 454.545.455 | Đơn vị: VND/m² (~81,1 tr/m² trung bình) |
| `log_price` | 18,42 | 22,55 | 23,80 | Phân phối gần chuẩn (skew ≈ −0,16) |
| `log_area` | 2,77 | 4,40 | 5,40 | — |
| `log_price_per_m2` | 13,65 | 18,17 | 19,94 | — |
| `quality_score` | 0 | 2,86 | 9 | Đỉnh tại score=2 (15.505 tin) |

**Phân phối `district_zone`:**

| Zone | Số bản ghi | Tỷ lệ |
|---|---|---|
| middle | 55.111 | 75,9% |
| outer | 11.362 | 15,6% |
| inner | 6.131 | 8,5% |

**Phân phối `balcony_direction` (qua 8 cột OHE):**

| Hướng ban công | Cột OHE | Số bản ghi có hướng |
|---|---|---|
| Nam | `balcony_dir_Nam` | ~nhiều nhất |
| Bắc | `balcony_dir_Bắc` | — |
| Đông Nam | `balcony_dir_Đông Nam` | — |
| Đông Bắc | `balcony_dir_Đông Bắc` | — |
| Tây | `balcony_dir_Tây` | — |
| Đông | `balcony_dir_Đông` | — |
| Tây Bắc | `balcony_dir_Tây Bắc` | — |
| Tây Nam | `balcony_dir_Tây Nam` | — |
| Unknown | *(không có cột — reference category)* | ~50% bản ghi (tất cả OHE = 0) |

---

### File 2: `hanoi_apartments_for_clustering.csv`

> Dùng cho: **Step 5a (K-Means Clustering)**

**Thông số tổng quan:**

| Chỉ số | Giá trị |
|---|---|
| Số bản ghi | 72.604 |
| Số cột | 8 |
| Missing values | 0 |

**Cấu trúc 8 cột:**

| Cột | Nhóm | Kiểu dữ liệu | Nguồn gốc | Giá trị mẫu |
|---|---|---|---|---|
| `scaled_log_price` | 6 features scaled | float64 | `log_price` → StandardScaler | 0,7506 |
| `scaled_log_area` | 6 features scaled | float64 | `log_area` → StandardScaler | 0,0987 |
| `scaled_bedroom_count` | 6 features scaled | float64 | `bedroom_count` → StandardScaler | −0,6287 |
| `scaled_bathroom_count` | 6 features scaled | float64 | `bathroom_count` → StandardScaler | 0,2159 |
| `scaled_log_price_per_m2` | 6 features scaled | float64 | `log_price_per_m2` → StandardScaler | 1,0819 |
| `scaled_district_encoded` | 6 features scaled | float64 | `district_encoded` → StandardScaler | −0,7621 |
| `district_name` | Label | object | Giữ nguyên từ raw | Hai Bà Trưng |
| `district_zone` | Label | object | Giữ nguyên từ raw | inner / middle / outer |

> **Lưu ý:** Cột `district_name` và `district_zone` được giữ lại trong file clustering **chỉ để gán nhãn sau khi phân cụm** (dùng cho phân tích và visualization), không đưa vào thuật toán K-Means.

---

### So sánh 2 file output

| Tiêu chí | `hanoi_apartments_processed.csv` | `hanoi_apartments_for_clustering.csv` |
|---|---|---|
| Mục đích | EDA + LightGBM regression | K-Means clustering |
| Số cột | 36 | 8 |
| Kiểu dữ liệu | Hỗn hợp (object, float, int, bool) | float64 + object (label) |
| Có cột raw (price, area) | Có | Không |
| Có log transform | Có | Có (trong scaled) |
| Có OHE / text features | Có | Không |
| Đã chuẩn hóa (scaled) | Không | Có (6 cột scaled) |
| Cột nhãn địa lý | `district_name`, `district_zone`, `ward_name` | `district_name`, `district_zone` |
