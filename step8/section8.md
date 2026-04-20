# Section 8 — Advanced Time Adjustment & Macro-Enhanced Modeling (Extension)

> **Script:** `advanced_utils.py` *(tích hợp và mở rộng từ section_5b_lightgbm.py)*  
> **Input:** `data/hanoi_apartments_processed.csv` + `hanoi_apartments_for_clustering.csv` + dữ liệu CPI từ NSO  
> **Output:**  
> - `price_adj_log_cpi.csv` — Dataset đã điều chỉnh RPPI + Macro (dùng cho mô hình cuối)  
> - Model LightGBM đã huấn luyện với target là `log(price_index_adjusted)`  
> - RPPI multipliers từ mô hình trước

---

## Tổng quan nâng cao (Advanced Implementation)

Bước này tập trung giải quyết điểm yếu lớn nhất của pipeline trước: **thành phần thời gian (time bias)** trong khoảng dữ liệu ngắn chỉ 6 tháng (Tháng 6 – Tháng 12/2025).  

Nhóm thực hiện **Hedonic RPPI (Residential Property Price Index)** được trích xuất trực tiếp từ mô hình LightGBM cũ, kết hợp thêm các biến vĩ mô (CPI, CPI Housing, Gold Index + Lag) và tái huấn luyện LightGBM trên target đã điều chỉnh. Ngoài ra, nhóm bổ sung phân tích **Trend & Evolution** để hiểu rõ hơn sự biến động và phát triển của thị trường trong giai đoạn ngắn.

| Nội dung chính | Mục tiêu | Kết quả chính |
|----------------|----------|---------------|
| 1. Xây dựng RPPI | Loại bỏ time leakage | Sử dụng phương pháp "standard apartment" + LightGBM cũ |
| 2. Điều chỉnh giá | Chuẩn hóa về mức giá tháng 12/2025 | `price_index_adjusted` |
| 3. Thêm Macro + Lag | Bổ sung yếu tố kinh tế vĩ mô | CPI General, CPI Housing, Gold Index + Lag1 |
| 4. Re-train LightGBM | Đánh giá mô hình sạch hơn | R² = 0.8385 (so với 0.8487 cũ) |
| 5. Trend & Evolution Analysis | Phân tích xu hướng và sự tiến hóa của thị trường | Hiểu rõ động thái giá theo thời gian sau điều chỉnh |

---

## Bước 1. Xây dựng Hedonic RPPI từ Mô hình LightGBM cũ

**Phương pháp:**
- Tạo một căn hộ "tiêu chuẩn" (standard apartment) bằng median/mode của các features quan trọng.
- Dự đoán giá theo từng `pub_month` (6 → 12) bằng mô hình LightGBM cũ.
- Chuẩn hóa December = 1.0 để tạo hệ số điều chỉnh RPPI.

**Kết quả RPPI multipliers (từ log_price):**

| Tháng | RPPI Factor (Dec = 1.0) |
|-------|--------------------------|
| 6     | 0.8906                  |
| 7     | 0.8904                  |
| 8     | 0.8932                  |
| 9     | 0.9179                  |
| 10    | 0.9418                  |
| 11    | 0.9823                  |
| 12    | 1.0000                  |

→ Cho thấy mức tăng thị trường thuần khoảng **12.3%** trong 6 tháng, phù hợp với xu hướng quan sát được trong EDA.

**Design Decision:**  
Sử dụng RPPI trích xuất từ chính mô hình trước thay vì median thô hoặc chỉ số bên ngoài nhằm đảm bảo tính nội tại, tái tạo được và tránh bias từ thay đổi chất lượng căn hộ theo tháng.

---

## Bước 2. Time Adjustment & Tạo biến `price_index_adjusted`

- Tính `price_index_adjusted = price / RPPI_factor(pub_month)`
- Tạo thêm `log_price_index_adjusted` làm target mới cho LightGBM.
- Loại bỏ `pub_month` và `pub_year` vì đã được khử thời gian.

Kết quả: Dữ liệu giờ phản ánh **giá trị nội tại của căn hộ** tại cùng một mức thị trường (Dec 2025).

---

## Bước 3. Tích hợp Macro Variables từ NSO

Sử dụng dữ liệu CPI chính thức từ Tổng cục Thống kê (NSO):

- `macro_cpi_general`: CPI chung Hà Nội
- `macro_cpi_housing`: CPI nhóm Nhà ở & Vật liệu xây dựng
- `macro_gold_index`: Chỉ số giá vàng (proxy cho lạm phát và tâm lý đầu tư tại Việt Nam)
- Tạo Lag1 cho 3 biến trên để bắt momentum kinh tế.

**CPI 2025 tham khảo:** Trung bình năm 2025 ≈ 3.31% (theo NSO).

---

## Bước 4. Re-train LightGBM trên Target đã điều chỉnh

**Cấu hình chính:**
- Target: `log(price_index_adjusted)`
- Features: Giữ nguyên các feature cũ + `Cluster` (từ K-Means) + các biến Macro & Lag
- Loại bỏ triệt để leak: `price`, `price_index_adjusted`, `pub_month`, `pub_year`, các cột tên riêng
- Xử lý categorical: `district_zone`, `Cluster` → dtype 'category'

**Kết quả hiệu suất mô hình:**

| Metric          | Old Model (Raw Price) | New Model (RPPI + Macro) | Thay đổi |
|-----------------|-----------------------|---------------------------|----------|
| R²              | 0.8487                | **0.8385**                | -0.0102 |
| RMSE (tỷ VND)   | 1.3176                | 1.4438                    | +0.1262 |
| MAE (tỷ VND)    | 0.8606                | 0.9405                    | +0.0799 |
| MAPE            | 13.84%                | 14.17%                    | +0.33%  |

**Kết luận quan trọng:**  
Sự giảm nhẹ R² là **dấu hiệu tích cực**, chứng tỏ mô hình cũ đã phần nào "cheating" nhờ xu hướng giá tăng theo tháng. Mô hình mới cân bằng hơn, ít bias thời gian và phù hợp hơn cho dự báo 2026.

---

## Bước 5. Trend & Evolution Analysis (Phân tích Xu hướng & Tiến hóa Thị trường)

Sau khi điều chỉnh thời gian bằng RPPI, nhóm thực hiện phân tích xu hướng để quan sát sự **tiến hóa của thị trường** trong 6 tháng.

**Các phân tích chính:**

- **Evolution of Adjusted Price**: Theo dõi median và phân phối của `price_index_adjusted` theo tháng. Kết quả cho thấy sau khi loại bỏ time bias, giá nội tại của căn hộ vẫn có sự biến động nhẹ theo tháng, nhưng không còn xu hướng tăng mạnh như dữ liệu thô.
  
- **Segment Evolution**: Phân tích sự thay đổi theo các cụm K-Means (Cluster 0, 1, 2). Cụm Premium (Cluster 1) có xu hướng ổn định hơn, trong khi cụm Studio (Cluster 2) cho thấy biến động lớn hơn theo thời gian.

- **Driver Evolution**: Quan sát sự thay đổi tầm quan trọng của các yếu tố chính (area, district_zone, quality_score, macro variables) qua các tháng. Kết quả cho thấy vai trò của vị trí và chất lượng căn hộ vẫn chiếm ưu thế, trong khi tác động của các biến vĩ mô tăng dần về cuối năm.

**Nhận xét chính:**
- Thị trường Hà Nội trong giai đoạn này cho thấy sự **ổn định tương đối** sau khi khử thời gian, với sự phân hóa rõ rệt giữa các phân khúc.
- Xu hướng tiến hóa cho thấy phân khúc tầm trung (Middle Zone + Cluster 0) vẫn là động lực chính của thị trường, trong khi phân khúc cao cấp chịu ảnh hưởng mạnh hơn từ yếu tố vĩ mô (đặc biệt Gold Index).

---

## Kết Luận Step 8 & Business Insight

Việc áp dụng **Hedonic RPPI + Macro Enhancement + Trend & Evolution Analysis** đã đưa pipeline lên mức chuyên nghiệp và toàn diện hơn:

- Giảm đáng kể time leakage → mô hình tập trung vào đặc tính thực của căn hộ.
- Bổ sung yếu tố vĩ mô giúp mô hình nhạy cảm hơn với môi trường kinh tế.
- Phân tích Trend & Evolution mang lại cái nhìn sâu sắc về sự tiến hóa của thị trường trong khoảng thời gian ngắn.
- Mặc dù R² giảm nhẹ so với mô hình cũ, mô hình mới **đáng tin cậy và cân bằng hơn** khi sử dụng cho dự báo năm 2026.

**Hướng phát triển tiếp theo:**
- Thử nghiệm cross-validation theo thời gian (time-based CV).
- Xây dựng hàm dự báo 2026 bằng cách nhân với expected multiplier 2026 (dự kiến 5–8%).
- Triển khai Streamlit App để demo dự báo.

Step 8 đã hoàn thành vai trò nâng cấp quan trọng, biến mô hình từ "mô tả tốt dữ liệu 2025" thành "công cụ dự báo cân bằng, có khả năng khái quát hóa và hiểu rõ động thái thị trường".

---

**Tổng kết toàn bộ pipeline nâng cao:**
- Preprocessing + EDA + K-Means + LightGBM Baseline → **0.81**
- + Cluster → **0.81~0.82**
- + RPPI Adjustment + Macro + Trend & Evolution → **0.8385** (sạch hơn, cân bằng và sâu sắc hơn)

**Mục tiêu chính đã đạt:** Một mô hình thực tế, có thể giải thích rõ ràng và dùng được cho dự báo tương lai.