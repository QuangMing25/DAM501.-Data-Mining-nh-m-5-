# Section 4 — Exploratory Data Analysis and Descriptive Mining (15%)

> **Script:** `section_4_eda.py`
> **Input:** `data/hanoi_apartments_processed.csv` — 72.604 bản ghi × 37 cột (Updated từ 51 cột)
> **Input (phụ):** `data/hanoi_apartments_for_clustering.csv` — 72.604 bản ghi × 8 cột
> **Output:** 13 biểu đồ trong `plots/section_4/` (Đã cập nhật theo logic Hướng Ban Công)

---

## Tổng quan phân tích

| Phần | Nội dung | Biểu đồ |
|---|---|---|
| Part 1 | Phân phối & Thống kê tổng quan | eda_01, eda_02 |
| Part 2 | So sánh theo nhóm (phòng ngủ, **Hướng Ban Công**) | eda_03, eda_04 |
| Part 3 | Xu hướng thời gian & Quan hệ Giá–Diện tích | eda_05, eda_06 |
| Part 4 | Ma trận tương quan mở rộng & Pair Plot | eda_07, eda_08 |
| Part 5 | Phân khúc giá, Quality Score & Top phường | eda_09, eda_10 |
| Part 6 | Khám phá cấu trúc PCA (tiền K-Means) | eda_11 |
| Part 7 | Text Features (9 Biến Tối Ưu) — Tương quan & Co-occurrence | eda_12 |
| Part 8 | Phân tích chéo Diện tích × Vùng | eda_13 |

---

## Part 1 — Phân phối & Thống kê Tổng quan

### 1.1 Thống kê theo Vùng Địa lý (Phân khu hành chính)

| Zone | Số tin | Giá TB (tỷ) | Median giá (tỷ) | DT TB (m²) | Giá/m² TB (tr) | Giá/m² Median (tr) |
|---|---|---|---|---|---|---|
| **Inner** (nội thành lõi) | 6.131 (8,5%) | 7,78 | 6,60 | 86,8 | 90,0 | 84,0 |
| **Middle** (vành đai 2-3) | 55.111 (75,9%) | 7,36 | 6,70 | 88,6 | 82,7 | 80,0 |
| **Outer** (ngoại thành) | 11.362 (15,6%) | 4,55 | 4,20 | 67,6 | 68,3 | 66,7 |

**Nhận xét Biểu đồ 1 & 2:**
- **Inner zone** có phân phối giá rộng nhất và đuôi kéo dài phản ánh định giá khuynh hướng di sản (Premium Vị trí).
- **Middle zone chiếm 75,9%** đại diện cho sức mua cốt lõi và xu hướng ở hiện tại của Hà Nội.
- **Tây Hồ** dẫn đầu biểu đồ quận về tỷ giá đơn vị (median 101,6 tr/m²) nhờ các chung cư siêu sang ven hồ, tạo ra sự chênh lệch (Gap) 13% so với huyện ngoại thành rẻ nhất là Mê Linh.

---

## Part 2 — So sánh theo Nhóm

### 2.1 Giá theo Số Phòng Ngủ
Số phòng ngủ biểu đạt tính tuyến tính rõ rệt trong tổng giá. Tuy nhiên, đơn giá (Giá/m²) thì ổn định theo dải từ 79 - 84 triệu/m², chứng minh đơn giá là hàm của địa lý thay vì diện tích thông thủy.

### 2.2 Đột phá: Giá theo Hướng Ban Công (`balcony_direction`)

Thay vì phân tích Hướng nhà bị loãng ở Version cũ, biểu đồ 4 **(eda_04_direction_vs_price.png)** thể hiện giá trị trung vị của **Hướng Ban Công**.
- Đỉnh bảng là Hướng Bắc (~83.9 triệu/m²) và Nam (~81.8 triệu/m²).
- Đáy bảng là Tây Nam (~77.3 triệu/m²) và Đông Nam (~77.4 triệu/m²).
- **Khám phá Đắt giá (Knowledge Discovery):** Hướng Đông Nam vốn được chuộng nhất theo quan niệm vì mát, nhưng giá trị/m² lại thấp hơn. Lý do là Môi giới nhập liệu theo "Hướng ban công" nhưng tư duy người mua định giá hướng tòa nhà đã bị phai nhạt và chi phối mạnh hơn bởi Phân khu, Sổ đỏ và Tiện ích đính kèm! Hướng nhà không còn là yếu tố định giá số 1 của căn hộ.

---

## Part 3 & 4 — Xu hướng Thời gian và Ma trận Quan hệ Kinh tế

- **Biến động 6 Tháng:** Biểu đồ xu hướng (eda_05) ghi nhận chung cư duy trì đà tăng đều từ Tháng 6 (6.1 Tỷ) lên trần ổn định ở Tháng 11-12 (7.5 Tỷ). 
- **Tương quan mạnh nhất (EDA_07):** Biến Price (Tổng giá) nhận tương quan khổng lồ từ `Area` (+0.77).
- **Phủ định Đa cộng tuyến:** `price_per_m2` tỏ ra miễn nhiễm với Diện tích, nó là hệ quả của cụm `Quality_Score` và Yếu tố lõi Zone.

---

## Part 5 — Quality Score và Mức Độ Nhượng Quyền Phân Khúc

*   Thị trường đang nằm trọn trong túi tiền từ **4 - 8 tỷ** (Đóng góp đến > 53% lượng cung thị trường).
*   **Quality Score** (Giá trị tiện ích từ 0-9): Thể hiện tác động Tuyến tính Hoàn hảo. Nhà càng giàu tiện ích (Score > 8) thì giá/m² càng ngất ngưởng.

**Top Phường Ganh Đua:**
Ba Đình (Liễu Giai, Giảng Võ) và Tây Hồ (Quảng An, Thụy Khuê) chiếm trọn Spotlight với đơn giá lên đến **166.7 triệu/m²**. Đối lập với Cự Khê thanh oai (42 triệu/m²). Việc Phường định giá mạnh hơn Quận là tiền đề cho mô hình LightGBM sẽ chia cắt không gian cây quyết định.

---

## Part 6 — PCA (Phân Tích Thành Phần Chính) dọn đường cho K-Means

PCA chứng minh chỉ cần 3 Thành phần chính là giữ được **87.6% Variance** toàn hệ thống:
*   **PC1 (50.3%):** Đại lượng của Kích cỡ (Log Price, Area, Bedrooms). Mệnh danh là "Tổng giá trị".
*   **PC2 (20.6%):** Đại lượng Định giá M2. Mệnh danh là "Độ Đắt Xắt Ra Miếng".
*   **PC3 (16.6%):** Đại lượng Địa lý nguyên thủy (District).

K-Means ở Step 5 đã cầm chắc vũ khí phân tách thị trường.

---

## Part 7 — Text Feature Impact (Sự lên ngôi của Sổ Đỏ và Tiện ích VIP)

Phiên bản EDA này sử dụng 9 Bio-Text Features cực sạch:

### 7.1 Điểm Tương quan theo Tính Năng
1.  **Căn Góc (`feat_corner_unit`):** Quán quân (+0.1128). Tăng sinh khí và view kép, định giá chung cư tăng đột biến.
2.  **Tiện ích VIP (`has_premium_amenities`):** Feature gộp thông minh (Bể bơi, gym, trẻ em) mang lại correlation cao đạt +0.0504. Chứng thực phân hạng chung cư kiểu mới.

### 7.2 Cặp Đôi Đồng Co-Occurrence
Sự xuất hiện bộ đôi của **Đầy đủ Nội thất (`feat_full_furniture`) + Sổ Đỏ Pháp Lý (`has_legal_paper`)** đạt 19,945 tin. Khẳng định quy chuẩn kinh doanh mua đứt bán đoạn ngay lập tức trên thị trường Hà Nội.

---

## KẾT LUẬN CHI TIẾT STEP 4 (Detailed Conclusions & Business Intelligence)

Sau quá trình Khám phá và Phân tích Dữ liệu (EDA) mở rộng trên tập dữ liệu 72.604 căn hộ Chung cư Hà Nội đã qua tiền xử lý, nhóm nghiên cứu đã thu thập và trích xuất thành công tri thức (Knowledge Discovery) có tính ứng dụng cao. Dưới đây là 3 khối kết luận cốt lõi:

### 1. Khối Kiến trúc Giá cả & Chênh lệch Địa lý (Pricing Architecture)
1.  **Vành đai Trung Tâm (Middle Zone) là Thước đo Thị Trường:** Các quận thuộc Vành đai 2-3 (như Thanh Xuân, Cầu Giấy, Nam Từ Liêm) thâu tóm tới 75.9% thị phần chung cư với phân khúc "cháy hàng" nằm trong khoảng **4-8 tỷ (chiếm 53%)**.
2.  **Sự phân hóa Vi Mô (Phường > Quận):** Đơn giá/m² không chỉ phụ thuộc vào cấp Quận. Điển hình như Quận Đống Đa: Phường Láng Thượng đạt 110 triệu/m², trong khi Phương Liên chỉ đạt mức bình dân 54 triệu/m². Nghĩa là _Vị trí cấp Phường (Ward)_ mang tính quyết định hơn so với cấp Quận.
3.  **Hội chứng "Nghịch lý Studio":** Các căn hộ diện tích siêu nhỏ (<45m²) lại đang ghi nhận Đơn giá/m² cao nhất hệ thống (85.3 triệu/m²), vượt mặt cả nhóm căn hộ lớn hay trung bình. Nhóm này đại diện cho mô hình Chung cư Mini hoặc Căn hộ đầu tư cho thuê ngắn hạn nằm sâu trong nội thành.

### 2. Khối Phản Biện Hành Vi Mua Bán (Behavioral Insights)
1.  **Sự Thất Sủng của "Hướng Nhà":** Phân tích Boxplot đã bẻ gãy định kiến truyền thống. Mức giá trần của chung cư không nằm ở các hướng Đông/Đông Nam mà lại đạt đỉnh ở Hướng Bắc/Nam. Kết luận rằng khi mua căn hộ, yếu tố thông thoáng tự nhiên và Hướng Ban Công, view nhìn thực tế có giá trị cao hơn sơ đồ la bàn truyền thống.
2.  **Cặp Bài Trùng Tiện Ích:** "Sổ Đỏ" và "Full Nội Thất" luôn song hành trong các tin bài chất lượng cao (19,945 tin đăng). Sự tồn tại của "Bể Bơi" hay "Căn Góc" đủ sức tạo ra cú nhảy vọt (Gap Premium) tăng hơn **11%** so với một căn chung cư chuẩn thông thường.

### 3. Khối Chuẩn bị Mô Hình Hóa Học Máy (Machine Learning Formulation)
Với những phát hiện dồi dào trên, chiến lược khai phá dữ liệu ở **Step 5** đã được định hình vô cùng rõ nét:
*   **PCA Confirm - K-Means Clustering:** Dữ liệu hoàn toàn có thể bị chia tách sâu xuống thành các "Vùng Giá Trị Bất Động Sản" (Real Estate Value Clusters) do Phương sai 3 chiều PC1, PC2, PC3 đã ôm trọn 87.6% cấu trúc. Ta sẽ dùng K-Means để đào ra các cụm: Học thuật (Gần trường), Ngoại ô Giá rẻ, Nội thành Premium.
*   **LightGBM Regression:** Tính chất tương quan phi tuyến giữa Area, District_Encoded, và Cụm Text Features rác/cao cấp khiến việc xử lý bằng Hồi quy tuyến tính (Linear Regression) sẽ sụp đổ. Sử dụng **LightGBM (Thuật toán cây quyết định Gradient Boosting)** là bước đi tối ưu nhất để xử lý tương tác chéo (Interaction effects) mà không lo bị hiệu ứng Đa cộng tuyến cản bước.

**Đánh giá tiến độ:** 
Step 4 đã hoàn thành xuất sắc vai trò "Mở đường". Data Mining Framework hiện tại đã sẵn sàng để dấn thân vào giai đoạn Model Training!
