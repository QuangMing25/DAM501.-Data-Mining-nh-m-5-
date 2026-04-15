# Section 3 — Data Pre-processing and Transformation (10%)

> **Script:** `section_3_preprocessing.py` *(1 file duy nhất, pipeline đầy đủ)*
> **Input:** `data/hanoi_apartments_cleaned.csv` — 86.601 bản ghi × 19 cột
> **Output:**
> - `data/hanoi_apartments_processed.csv` — **72.604 bản ghi × 37 cột** (dùng cho EDA và LightGBM)
> - `data/hanoi_apartments_for_clustering.csv` — 72.604 bản ghi × 8 cột scaled (dùng cho K-Means)
> - Các biểu đồ lưu tại `plots/section_3/`

---

## Tổng quan pipeline tiền xử lý (13 bước)

| Bước | Thao tác | Phân tích & Trọng tâm rút ra | Kết quả |
|---|---|---|---|
| 0 | Load raw data | Đọc dữ liệu đầu vào | 86.601 × 19 |
| 1 | Phân tích missing values | Tìm ra cột lỗi thiếu dữ liệu | — |
| 2 | Loại cột missing >= 95% | Loại bỏ thông tin ảo (chiều sâu, mặt tiền...), giảm nhiễu vật lý | Xóa 4 cột |
| 3 | Loại cột không giá trị KH | Loại bỏ `house_direction` do môi giới hay nhầm, chuyển trọng tâm sang hướng ban công. Giữ description lại để bóc rã. | Xóa 5 cột |
| 4 | Xử lý missing values | Fill khuyết bằng Median theo từng quận, giữ nguyên vẹn địa lý. | Xóa 12.157 dòng trống giá |
| 5 | Phát hiện và loại outliers | Cắt giá < 100tr, > 21.7 tỷ để làm sạch mỏ neo thị trường thực tế | Xóa 1.840 dòng (2,5%) |
| 6 | Feature Engineering | Đo lường hệ quy chiếu mới: Giá/m², phân khu chuẩn (Zone) | Tạo 4 biến mới |
| 7 | Log Transform | Đưa dữ liệu về phân phối lý tưởng cho Machine Learning | Giảm skewness rõ rệt |
| 8 | Encoding categorical | Đổi hướng nghiên cứu sang `balcony_direction` với 9 biến OHE | OHE hướng ban công |
| 9 | Chuẩn hóa (K-Means) | StandardScaler để dọn đường cho clustering | Ma trận 6 tính năng scaled |
| 10 | **Text Feature Extraction** | Gom lại 9 text features quan trọng nhất, gộp tính năng có chung hệ quy chiếu (VIP, Pháp lý) | Thêm 9 biến Binary + Score |
| 11 | Final summary | Xác nhận chất lượng ma trận | 72.604 × 37 sạch |
| 12 | Lưu output | Phân luồng cho Clustering và Regression | 2 CSV files |
| 13 | Vẽ biểu đồ | Dùng để phân tích bước 3 | 5 plots |

---

## Bước 1. Đánh giá Mức độ Thiếu hụt Dữ liệu (Null/Missing Data)

**Kết quả:**
*   Dữ liệu có nhóm missing tuyệt đối (>99%): `house_depth`, `road_width`, `floor_count`, `frontage_width`.
*   Dữ liệu có nhóm missing 50%: `balcony_direction`, `house_direction`.
*   Dữ liệu thiếu 14%: `price`.

**Quyết định thiết kế (Design Decision):** 
1. Với đặc thù Chung Cư (Apartment), mặt tiền hay đường vào (frontage_width, road_width) là các chi tiết vô nghĩa. Việc thiếu hụt >99% đến từ việc Batdongsan.vn ép nhập dữ liệu cho đất nền, không dùng cho chung cư. Buộc phải loại bỏ hoàn toàn các cột này ở Bước 2.
2. Thiết kế One-Hot Encoding cho các biến thiếu 50% bằng cách fill "Unknown" để không làm mất đi khối lượng dữ liệu quý giá.

---

## Bước 3 & Bước 8. Bước ngoặt xử lý Direction (Hướng nhà)

Trước đây, `house_direction` (Hướng cửa chính) đóng vai trò trung tâm nhưng lại đưa lại kết quả thống kê sai với nguyên lý đời thực. Nhóm nhận ra sự nhầm lẫn của môi giới lúc nhập liệu, môi giới hay lấy hướng sửa lớn làm chuẩn.
Thay vào đó, Model gạch bỏ biểu đồ hướng nhà và sử dụng duy nhất **`balcony_direction` (Hướng ban công)** cung cấp trực diện khí hậu thực tế của căn hộ.
**Kết quả:** Cột `house_direction` bị xóa sổ ở Bước 3. Cột `balcony_direction` được mang sang Bước 8 để One-Hot Encoding thành 9 biến phái sinh riêng biệt (dir_Đông Nam, dir_Tây Bắc,...).

---

## Bước 10. Tinh chỉnh Text Features Extraction

Bắt nguồn từ 17 features thô rác được trích ra từ trường `description`, thuật toán Bước 10 đã được nhóm tối ưu mạnh tay:
*   Loại 5 Text feature thuộc nhóm "Hiển nhiên có" như *thang máy*, *bãi để xe* để tránh hiệu ứng False Negative cho dữ liệu. Tránh mô hình học sai rằng chung cư Hà Nội... 90% phải đi bộ lên cầu thang.
*   Gộp (Merge) 2 cụm Tính năng lồng nhau để giảm đa cộng tuyến:
    1.  `has_legal_paper`: Cụm Sổ đỏ, Sổ hồng, Pháp lý rõ ràng. (Mức độ sở hữu 53.9%).
    2.  `has_premium_amenities`: Cụm Hồ bơi, Gym, Sân chơi trẻ em. (Mức độ sở hữu 21.1%).

**Danh sách 9 Text Features cuối cùng:**

| Feature (Biến) | Tên tiếng Việt | Tần suất |
| :--- | :--- | :--- |
| `has_legal_paper` | Giấy tờ Pháp lý | 53.9% |
| `feat_full_furniture` | Đầy đủ nội thất | 44.5% |
| `feat_balcony` | Có ban công | 43.9% |
| `feat_near_school` | Gần hệ thống trường | 30.9% |
| `feat_near_mall` | Gần siêu thị / TTTM | 30.9% |
| `feat_near_park` | Gần công viên | 27.3% |
| `feat_near_hospital` | Gần bệnh viện | 21.7% |
| `has_premium_amenities`| Tiện ích VIP nội khu | 21.1% |
| `feat_corner_unit` | Đặc trưng căn góc | 11.6% |

Biến `quality_score` cũng được hiệu chỉnh: Mean giảm xuống 2.86 (tối đa là 9). Phản ánh đúng mức độ cao cấp của một đoạn miêu tả môi giới.

---

## Tổng kết Bước 11 (Final Results)

### Profile Ma trận Data cuối cùng (`hanoi_apartments_processed.csv`)
*   **Tổng số dòng:** 72,604 (Chắt lọc từ 86,601 bản thô).
*   **Số lượng cột:** 37 Columns. 
*   **Sạch sẽ:** 0% Missing Values.

### Thống kê Toán học
*   **`price` (tỷ VND)**: Median 6.25 Tỷ | Mean 6.96 Tỷ 
*   **`area` (m²)**: Median 80m² | Mean 85.1m²
*   **`price_per_m2` (triệu/m²)**: Median 77.6 triệu | Mean 81.1 triệu
*   **Lệch sinh thái Địa lý**: Thị trường cực kì bất cân xứng tập trung tại phân khu Vành đai đô thị (Middle Zone) với lượng tin bài khổng lồ (55,111 tin - chiếm >75%).

### Đánh giá chất lượng của Feature Engineering:
- **Ngăn chặn Lời nguyền mảng (Curse of Dimensionality):** Do loại bỏ mạnh tay nhóm Text Rác và định hướng đúng Direction, mô hình giảm size từ 51 cột (cũ) xuống đúng quy mô 37 cột mà không làm mất % variance nào yếu vị.
- **Log Transformation cực kỳ hiệu quả:** Cải thiện phân phối xác suất hình chuông. Độ tiệm cận của mô hình (Dành cho Cây quyết định về sau) được hưởng lợi tuyệt đối với cột Log Price lệch cực kì nhỏ (-0.16).
- Sẵn sàng tích hợp sang **Step 4: Phân tách Khám phá Đặc tả (EDA)** và **Step 5: K-Means Regression Pipelines**.
