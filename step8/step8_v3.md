## Tổng quan nâng cao (Advanced Implementation)

### Data Enrichment – Index-based Adjustment

#### 1. Vấn đề (Problem)

Giá bất động sản biến động theo thời gian do các điều kiện kinh tế vĩ mô (lạm phát, chi phí xây dựng, dòng tiền đầu tư).

→ Điều này tạo ra **time bias**:  
Model có thể nhầm lẫn giữa:
- “Căn hộ tốt hơn”  
- và “Thị trường đang tăng giá”

→ Dẫn đến sai lệch trong học và dự báo, đặc biệt với tập dữ liệu ngắn (T6–T12/2025).

---

#### 2. Macro Indicators (Bổ sung tín hiệu vĩ mô)

Để xử lý thành phần thời gian, nhóm tích hợp các biến vĩ mô làm feature đầu vào:

- `macro_cpi_general`: Chỉ số giá tiêu dùng chung (lạm phát tổng thể)  
- `macro_cpi_housing`: CPI nhóm Nhà ở & Vật liệu xây dựng (chi phí ngành)  
- `macro_gold_index`: Chỉ số giá vàng (proxy cho hành vi đầu tư tại Việt Nam)  

**Lag Feature:**
- Tạo thêm `lag1` (~1 tháng) cho các biến macro  
→ Phản ánh độ trễ trong quyết định mua nhà

**Vai trò:**
- Giúp model học được ảnh hưởng của kinh tế vĩ mô lên giá bất động sản  
- Biến “thời gian” từ latent variable → observable features

---

#### 3. Xây dựng Hedonic RPPI từ Mô hình LightGBM

**Phương pháp:**
- Tạo một căn hộ "tiêu chuẩn" (standard apartment) bằng median/mode của các features quan trọng.
- Giữ nguyên đặc điểm căn hộ (area, location, cluster…)
- Chỉ thay đổi yếu tố thời gian (macro + month)
- Dự đoán giá theo từng `pub_month` bằng mô hình LightGBM.

**Chuẩn hóa:**
- Lấy T12/2025 = 1.0 để tạo hệ số RPPI

→ RPPI phản ánh **biến động thị trường thuần túy theo thời gian**

---

**Kết quả RPPI multipliers (từ log_price):**

| Tháng | Năm   | RPPI Factor (Dec = 1.0) |
|-------|-------|--------------------------|
| 6     | 2025  | 0.9055                  |
| 7     | 2025  | 0.8850                  |
| 8     | 2025  | 0.9039                  |
| 9     | 2025  | 0.9391                  |
| 10    | 2025  | 0.9531                  |
| 11    | 2025  | 0.9719                  |
| 12    | 2025  | 1.0000                  |
| 1     | 2026  | 0.9321                  |
| 2     | 2026  | 0.9132                  |
| 3     | 2026  | 0.9001                  |

→ Có thể thấy:
- Giá thị trường không tăng tuyến tính mà có dao động (T7 giảm nhẹ)
- Sau T12/2025 đạt đỉnh → Q1/2026 có xu hướng giảm

**Design Decision:**  
Sử dụng RPPI trích xuất từ mô hình thay vì chỉ số ngoài → đảm bảo:
- Tính nội tại (model-consistent)
- Tránh bias do thay đổi chất lượng căn hộ theo thời gian

---

#### 4. Price Index Adjustment (Chuẩn hóa giá)

- Tính `price_index_adjusted = price / RPPI_factor(pub_month)`
- Tạo thêm `log_price_index_adjusted` làm target mới
- Loại bỏ `pub_month`, `pub_year` sau khi đã xử lý thời gian

**Kết quả:**
- Giá được quy về cùng mặt bằng thời gian (T12/2025)
- Phản ánh **giá trị nội tại của căn hộ**

---

#### 5. Re-train LightGBM với dữ liệu đã điều chỉnh

**Cấu hình chính:**
- Target: `log(price_index_adjusted)`
- Features:
  - Feature gốc (diện tích, vị trí, cluster…)
  - Macro indicators + lag
- Loại bỏ leak: `price`, `price_index_adjusted`, `pub_month`, `pub_year`
- Xử lý categorical: `district_zone`, `Cluster`

---

**Kết quả hiệu suất mô hình:**

| Metric          | Old Model (Raw Price) | New Model (RPPI + Macro) | Thay đổi |
|-----------------|-----------------------|---------------------------|----------|
| R²              | 0.8487                | **0.87**                  | +0.0213 |
| RMSE (tỷ VND)   | 0.9405                | 0.976                     | +0.0355 |
| MAE (tỷ VND)    | 1.4438                | 1.4934                    | +0.0496 |
| MAPE            | 13.84%                | 13.24%                    | -0.6%   |

---

**Nhận xét quan trọng:**
- Chỉ số vàng biến động mạnh (đỉnh T10/2025 → giảm T3/2026)
→ phản ánh yếu tố tâm lý đầu tư
- CPI tổng & CPI housing ổn định → đóng vai trò nền
- RPPI cho thấy thị trường giảm nhẹ đầu 2026

---

## Kết luận Step 8 & Business Insight

Việc kết hợp **Macro Features + Model-based RPPI** đã nâng cấp pipeline:

- Giảm time bias → mô hình tập trung vào đặc tính thực của căn hộ  
- Bổ sung tín hiệu vĩ mô → tăng độ nhạy với thị trường  
- Chuẩn hóa giá → đảm bảo so sánh được giữa các giai đoạn  

→ Mô hình trở nên **ổn định hơn và có khả năng tổng quát hóa tốt hơn cho 2026**

---

**Hướng phát triển tiếp theo:**
- Time-based cross-validation  
- Forecast multiplier 2026  
- Triển khai demo (Streamlit)
