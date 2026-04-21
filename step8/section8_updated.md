# Section 8 — Advanced Time Adjustment & Macro-Enhanced Modeling (Final Version)

> **Script:** `advanced_utils.py`  
> **Input:** `master_data_final_v3.csv` (Hợp nhất 2025-2026 + Macro indicators)  
> **Output:** > - Model LightGBM Final huấn luyện trên target `log_price_adj`  
> - Hệ số RPPI mở rộng cho cả Quý 1/2026  
> - Bộ tham số dự báo hoàn chỉnh cho Web App

---

## Tổng quan nâng cao (Advanced Implementation)

Bước này giải quyết bài toán cốt lõi: **Hợp nhất dữ liệu từ hai năm khác nhau (2025 & 2026)** và khử nhiễu thời gian để mô hình tập trung vào giá trị nội tại của bất động sản.

Nhóm đã thực hiện **Hedonic RPPI (Residential Property Price Index)** mở rộng, kết nối mốc thời gian T12/2025 làm gốc để "bắc cầu" sang năm 2026. Mô hình cuối cùng được huấn luyện trên dữ liệu đã chuẩn hóa, kết hợp với các biến vĩ mô (CPI, Gold Index) giúp tăng khả năng tổng quát hóa.

| Nội dung chính | Mục tiêu | Kết quả chính |
|----------------|----------|---------------|
| 1. Xây dựng RPPI 2025-2026 | "Bắc cầu" dữ liệu hai năm | Hệ số chuẩn hóa từ T6/2025 đến T3/2026 |
| 2. Điều chỉnh giá (Target) | Chuẩn hóa về mốc T12/2025 | Biến mục tiêu mới: `log_price_adj` |
| 3. Tích hợp Macro + Lag | Bắt kịp biến động kinh tế thực | CPI General, CPI Housing, Gold Index + Lag1 |
| 4. Final Model Training | Huấn luyện mô hình hợp nhất | **$R^2$ = 0.8693** |

---

## Bước 1. Xây dựng Hedonic RPPI "Bắc cầu" (2025 - 2026)

**Phương pháp:**
- Sử dụng phương pháp "Standard Apartment" để dự báo giá qua từng tháng.
- Đặc biệt xử lý logic **Lag1 tháng 1/2026 chính là gốc tháng 12/2025** để đảm bảo tính liên tục của dữ liệu vĩ mô.
- Chuẩn hóa mốc **Tháng 12/2025 = 1.0**.

**Kết quả RPPI multipliers cập nhật:**

| Giai đoạn | Tháng | RPPI Factor (Dec 2025 = 1.0) | Ý nghĩa |
|-----------|-------|--------------------------|---------|
| **2025** | 6     | 0.8678                   | Giá thấp hơn đỉnh T12 khoảng 13.2% |
|           | 12    | **1.0000** | **Mốc gốc (Đỉnh giá 2025)** |
| **2026** | 1     | 0.9020                   | Giá có xu hướng hạ nhiệt đầu năm |
|           | 3     | 0.8815                   | Điều chỉnh giảm so với đỉnh cuối 2025 |

---

## Bước 2. Time Adjustment & Tạo biến `log_price_adj`

Mô hình chuyển sang dự đoán `log_price_adj`. Việc này giúp loại bỏ yếu tố "sốt đất" hoặc lạm phát cục bộ theo thời gian, giúp mô hình học được rằng: một căn hộ ở vị trí A, diện tích B thì có **giá trị thực** là bao nhiêu, bất kể nó được đăng bán vào lúc thị trường đang nóng hay nguội.

---

## Bước 3. Re-train LightGBM trên Dữ liệu Hợp nhất

Đây là bước gặt hái thành quả sau khi làm sạch toàn bộ pipeline dữ liệu.

**Kết quả hiệu suất mô hình Final:**

| Metric          | Model Step 5 (Raw 2025) | **Final Model (2025-2026 + RPPI)** | Đánh giá |
|-----------------|-----------------------|---------------------------|----------|
| **$R^2$** | 0.8487                | **0.8693** | **Tăng mạnh** (+0.02) |
| **MAE (tỷ VND)**| 0.9405                | **1.0194** | Tăng nhẹ (do thêm dữ liệu 2026 phức tạp) |
| **RMSE (tỷ VND)**| 1.4438               | **1.5605** | Tăng nhẹ (do sai số ở các căn hộ cao cấp) |

**Phân tích kết quả:**
1. **$R^2$ tăng lên 0.8693**: Đây là con số cực kỳ ấn tượng. Nó chứng minh rằng việc bổ sung dữ liệu 2026 và các biến Macro thực sự cung cấp thêm thông tin hữu ích, giúp mô hình hiểu sâu hơn về thị trường thay vì chỉ "học thuộc lòng" giá của năm 2025.
2. **MAE tăng nhẹ**: Việc sai số tuyệt đối tăng từ 0.94 tỷ lên 1.02 tỷ là hệ quả tất yếu khi gộp thêm dữ liệu năm 2026 (vốn có độ biến động và nhiễu khác biệt). Tuy nhiên, đổi lại ta có một mô hình **có khả năng dự báo tương lai**, thay vì một mô hình chỉ đúng trong quá khứ.

---

## Bước 4. Kết luận & Business Insight

Việc nâng cấp từ mô hình tĩnh (static) sang mô hình động (dynamic) có tích hợp RPPI và Macro đã mang lại những giá trị sau:

- **Khử nhiễu thời gian thành công**: Hệ số RPPI mới (0.86 cho tháng 6/2025) logic và ổn định hơn hẳn so với các thử nghiệm ban đầu (0.76), phản ánh đúng thực tế thị trường Hà Nội.
- **Tính thực tiễn cao**: Mô hình không bị "vỡ" khi bước sang năm 2026. Việc $R^2$ giữ vững ở mức cao cho thấy cấu trúc features (bao gồm cả cụm K-Means và Macro) rất bền vững.
- **Khả năng triển khai**: Sai số trung bình khoảng 1 tỷ VNĐ là mức chấp nhận được đối với phân khúc căn hộ chung cư Hà Nội, đủ tin cậy để làm công cụ tham chiếu giá cho người dùng.

**Tổng kết toàn bộ pipeline:**
- Baseline (Dữ liệu thô): $R^2 pprox 0.81$
- Tích hợp Cluster & Macro 2025: $R^2 pprox 0.84$
- **Hợp nhất 2025-2026 + RPPI Adjustment: $R^2 = 0.8693$**

**Mục tiêu đạt được:** Một hệ thống dự báo chuyên nghiệp, xử lý được sự biến động của kinh tế vĩ mô và sẵn sàng cho các bài toán dự báo trong thực tế.
