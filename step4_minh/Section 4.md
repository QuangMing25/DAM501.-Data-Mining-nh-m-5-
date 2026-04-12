# Section 4 — Exploratory Data Analysis and Descriptive Mining (15%)

> **Script:** `section_4_eda.py`
> **Input:** `data/hanoi_apartments_processed.csv` — 72.604 bản ghi × 51 cột
> **Input (phụ):** `data/hanoi_apartments_for_clustering.csv` — 72.604 bản ghi × 8 cột
> **Output:** 13 biểu đồ trong `plots/section_4/`

---

## Tổng quan phân tích

| Phần | Nội dung | Biểu đồ |
|---|---|---|
| Part 1 | Phân phối & Thống kê tổng quan | eda_01, eda_02 |
| Part 2 | So sánh theo nhóm (phòng ngủ, hướng nhà) | eda_03, eda_04 |
| Part 3 | Xu hướng thời gian & Quan hệ Giá–Diện tích | eda_05, eda_06 |
| Part 4 | Ma trận tương quan mở rộng & Pair Plot | eda_07, eda_08 |
| Part 5 | Phân khúc giá, Quality Score & Top phường | eda_09, eda_10 |
| Part 6 | Khám phá cấu trúc PCA (tiền K-Means) | eda_11 |
| Part 7 | Text Features — Tương quan & Co-occurrence | eda_12 |
| Part 8 | Phân tích chéo Diện tích × Vùng | eda_13 |

---

## Part 1 — Phân phối & Thống kê Tổng quan

### 1.1 Thống kê theo Vùng Địa lý

| Zone | Số tin | Giá TB (tỷ) | Median giá (tỷ) | DT TB (m²) | Giá/m² TB (tr) | Giá/m² Median (tr) |
|---|---|---|---|---|---|---|
| **Inner** (nội thành lõi) | 6.131 (8,5%) | 7,78 | 6,60 | 86,8 | 90,0 | 84,0 |
| **Middle** (vành đai 2-3) | 55.111 (75,9%) | 7,36 | 6,70 | 88,6 | 82,7 | 80,0 |
| **Outer** (ngoại thành) | 11.362 (15,6%) | 4,55 | 4,20 | 67,6 | 68,3 | 66,7 |

### 1.2 Thống kê Top 10 Quận (theo giá/m²)

| Quận | Số tin | Giá TB (tỷ) | Median giá (tỷ) | Giá/m² TB (tr) | DT TB (m²) |
|---|---|---|---|---|---|
| Cầu Giấy | 8.568 | 9,02 | 8,83 | 94,6 | 96,8 |
| Bắc Từ Liêm | 4.807 | 8,65 | 7,90 | 90,0 | 95,3 |
| Thanh Xuân | 7.191 | 8,57 | 7,90 | 89,5 | 95,7 |
| Đống Đa | 2.518 | 7,57 | 6,40 | 86,1 | 88,9 |
| Hai Bà Trưng | 2.180 | 7,21 | 6,60 | 85,2 | 83,9 |
| Nam Từ Liêm | 13.001 | 7,10 | 6,60 | 84,4 | 85,0 |
| Long Biên | 3.083 | 6,11 | 5,50 | 72,0 | 84,5 |
| Hoàng Mai | 8.386 | 5,49 | 5,10 | 71,6 | 76,2 |
| Gia Lâm | 6.201 | 4,17 | 3,92 | 70,5 | 59,6 |
| Hà Đông | 7.903 | 5,59 | 5,30 | 64,8 | 86,8 |

### Biểu đồ 1 — Phân phối Giá, Diện tích, Giá/m² theo Vùng (Violin)

![eda_01_zone_distributions](plots/section_4/eda_01_zone_distributions.png)

**Nhận xét:**

- **Inner zone** có phân phối giá rộng nhất — giá median 6,6 tỷ nhưng đuôi trên kéo dài đến 20+ tỷ, phản ánh sự đa dạng từ căn hộ cũ đến căn hộ siêu cao cấp ở Ba Đình, Tây Hồ.

- **Outer zone** có phân phối tập trung hẹp ở mức 2–6 tỷ VND — thị trường ngoại thành đồng nhất hơn, ít phân hóa. Diện tích trung bình cũng thấp nhất (67,6 m²), chủ yếu là căn hộ nhỏ ở Gia Lâm, Đông Anh.

- **Giá/m² phân hóa rõ ràng**: Inner (median 84 tr/m²) > Middle (80 tr/m²) > Outer (66,7 tr/m²). Outer zone rẻ hơn inner zone ~25% về giá đơn vị, là tín hiệu quan trọng cho K-Means clustering.

- **Middle zone chiếm 75,9%** tổng tin đăng — đây là \"trọng tâm\" thị trường, và bất kỳ pattern nào từ clustering/regression đều sẽ bị ảnh hưởng mạnh bởi phân khúc này.

### Biểu đồ 2 — So sánh Giá theo Quận

![eda_02_district_price_comparison](plots/section_4/eda_02_district_price_comparison.png)

**Nhận xét:**

- **Cầu Giấy dẫn đầu** cả về giá tuyệt đối (median 8,8 tỷ) lẫn giá/m² (94,6 tr/m²) — phản ánh vị trí đắc địa gần khu công nghệ cao, nhiều dự án cao cấp (Vinhomes Metropolis, The Nine...).

- **Gap lớn giữa inner và outer**: Quận rẻ nhất (Gia Lâm: median 3,9 tỷ) chỉ bằng 44% quận đắt nhất (Cầu Giấy: 8,8 tỷ).

- **Hà Đông là trường hợp đặc biệt**: giá/m² thấp nhất trong middle zone (64,8 tr/m²) dù thuộc vành đai trung — do quỹ đất lớn, nhiều dự án giá rẻ. Diện tích trung bình lại cao (86,8 m²), cho thấy Hà Đông cung cấp "diện tích nhiều hơn với giá rẻ hơn".

- **Bắc Từ Liêm và Thanh Xuân** có giá/m² ngang nhau (~90 tr/m²) dù vị trí địa lý khác nhau — cho thấy giá không chỉ phụ thuộc khoảng cách trung tâm mà còn phụ thuộc vào chất lượng dự án.

---

## Part 2 — So sánh theo Nhóm

### 2.1 Giá theo Số Phòng Ngủ

| Số PN | Số tin | Giá TB (tỷ) | Median | DT TB (m²) | Giá/m² TB (tr) |
|---|---|---|---|---|---|
| 1 | 4.737 | 3,66 | 3,38 | 43,6 | 83,7 |
| 2 | 35.088 | 5,58 | 5,10 | 69,8 | 79,4 |
| 3 | 29.903 | 8,60 | 7,99 | 103,8 | 82,5 |
| 4 | 2.772 | 12,14 | 12,00 | 146,5 | 83,0 |
| 5+ | 104 | 11,47 | 10,00 | 155,1 | 77,0+ |

### 2.2 Giá theo Hướng Nhà (Top 5)

| Hướng | Số tin | Giá TB (tỷ) | Median | Giá/m² TB (tr) |
|---|---|---|---|---|
| Bắc | 2.070 | 7,85 | 7,20 | 87,1 |
| Nam | 2.352 | 7,41 | 6,80 | 85,5 |
| Tây | 1.413 | 7,35 | 6,80 | 81,8 |
| Tây - Bắc | 4.828 | 7,42 | 6,80 | 82,0 |
| Unknown | 35.313 | 6,79 | 6,00 | 80,1 |

### Biểu đồ 3 — Giá theo Số Phòng Ngủ và Phòng Tắm

![eda_03_rooms_vs_price](plots/section_4/eda_03_rooms_vs_price.png)

**Nhận xét:**

- **Giá tăng tuyến tính theo số phòng ngủ**: mỗi phòng ngủ thêm ~2,5-3 tỷ VND — từ 3,4 tỷ (1PN) lên 5,1 tỷ (2PN), 8,0 tỷ (3PN), 12,0 tỷ (4PN). Đây là quan hệ mạnh, dễ nắm bắt cho regression.

- **Phân khúc 2-3 phòng ngủ chiếm 89,5%** tổng tin đăng (35.088 + 29.903 = 65.000 tin). Đây là phân khúc chủ đạo của thị trường Hà Nội, nhắm đến hộ gia đình 3-4 người.

- **Căn 4PN** có mức giá premium rõ ràng (median 12 tỷ) nhưng chỉ chiếm 3,8% — phân khúc cao cấp nhỏ, giá/m² không cao hơn đáng kể (83 tr/m² vs 79 tr/m² ở 2PN), cho thấy premium đến từ diện tích chứ không từ đơn giá.

- **Giá/m² ổn định** ở mức 79–84 tr/m² bất kể số phòng ngủ — xác nhận `price_per_m2` phụ thuộc vào vị trí hơn là kích thước căn hộ.

### Biểu đồ 4 — Giá/m² theo Hướng Nhà

![eda_04_direction_vs_price](plots/section_4/eda_04_direction_vs_price.png)

**Nhận xét:**

- **Hướng Bắc có giá/m² cao nhất** (87 tr/m², cao hơn median 12,5%) — điều này hơi bất ngờ vì phong thủy Việt Nam truyền thống ưa hướng Nam. Tuy nhiên, có thể giải thích bởi confounding: nhiều dự án cao cấp trung tâm (Ba Đình, Tây Hồ) có căn hướng Bắc nhìn ra Hồ Tây.

- **Hướng Nam đứng thứ 2** (85,5 tr/m²) — phù hợp với ưa thích phong thủy, nhưng chênh lệch với hướng Bắc không lớn.

- **Tây Nam và Đông Bắc thấp nhất** (~80–81 tr/m²) — đây thường là hướng ít được ưa chuộng trong văn hóa phong thủy Việt Nam.

- **Nhóm "Unknown" (48,6% = 35.313 tin)** có giá thấp hơn trung bình — phản ánh việc thiếu thông tin hướng nhà thường đi kèm với tin đăng chất lượng thấp hoặc dự án phổ thông.

- **Chênh lệch tối đa ~8,7%** giữa hướng đắt nhất và rẻ nhất — tác động hướng nhà là thực nhưng yếu hơn nhiều so với vị trí địa lý.

---

## Part 3 — Xu hướng Thời gian & Quan hệ Giá–Diện tích

### 3.1 Xu hướng Đăng Tin theo Tháng (06–12/2025)

| Tháng | Số tin | Giá TB (tỷ) | Median giá (tỷ) | Giá/m² TB (tr) |
|---|---|---|---|---|
| T6 | 7.511 | 6,10 | 5,50 | 71,6 |
| T7 | 11.613 | 6,35 | 5,70 | 73,5 |
| T8 | 10.582 | 6,69 | 6,00 | 77,2 |
| T9 | 10.123 | 7,01 | 6,30 | 80,5 |
| T10 | 10.270 | 7,28 | 6,58 | 84,5 |
| T11 | 10.401 | 7,55 | 6,90 | 88,3 |
| T12 | 11.796 | 7,53 | 6,80 | 89,0 |

### Biểu đồ 5 — Xu hướng Thị Trường theo Tháng

![eda_05_monthly_trends](plots/section_4/eda_05_monthly_trends.png)

**Nhận xét:**

- **Giá tăng liên tục 6 tháng liên tiếp** (T6→T11): giá trung bình tăng từ 6,10 tỷ lên 7,55 tỷ (+23,8%). Giá/m² tăng từ 71,6 lên 89,0 tr/m² (+24,3%). Đây là xu hướng tăng giá rõ ràng trong nửa cuối năm 2025.

- **T12 chững lại nhẹ**: giá trung bình giảm nhẹ so với T11 (7,53 vs 7,55 tỷ) nhưng giá/m² vẫn tăng (89,0 vs 88,3 tr/m²) — cho thấy thị trường có dấu hiệu ổn định vào cuối năm.

- **Số lượng tin đăng biến động**: T6 thấp nhất (7.511), tăng mạnh ở T7 (11.613) rồi ổn định 10.000–11.800 tin/tháng. T7 cao do hiệu ứng "sau kỳ nghỉ hè" và start of Q3.

- **Biến `pub_month` có tương quan +0,15 với giá** — yếu nhưng nhất quán, cho thấy xu hướng thời gian nên được đưa vào LightGBM như một feature.

### Biểu đồ 6 — Scatter Giá vs Diện tích

![eda_06_price_area_scatter](plots/section_4/eda_06_price_area_scatter.png)

**Nhận xét:**

- **Scatter plot xác nhận tương quan mạnh price~area (r=0,77)**: các điểm phân bố theo hướng tuyến tính rõ ràng, nhưng có phương sai lớn — cùng 80 m² có thể có giá từ 2 tỷ đến 15 tỷ.

- **3 vùng tách biệt rõ ràng** trong scatter: Outer (xanh lá) tập trung ở góc dưới-trái (nhỏ, rẻ), Inner (đỏ) phân tán rộng ở phía trên, Middle (xanh dương) chiếm phần lớn.

- **Hexbin density map** cho thấy mật độ cao nhất tại vùng 60–100 m² × 3–8 tỷ VND — đây là "sweet spot" của thị trường căn hộ Hà Nội.

- **Phương sai lớn ở cùng diện tích** chứng tỏ diện tích đơn lẻ không đủ giải thích giá → cần thêm biến vị trí, tiện ích, chất lượng. Đây chính là lý do LightGBM (multi-feature) sẽ hiệu quả hơn regression đơn biến.

---

## Part 4 — Tương quan & Ma trận Quan hệ

### 4.1 Tương quan với Giá (xếp hạng)

| Feature | Pearson r |
|---|---|
| log_price | +0,949 |
| area | +0,771 |
| log_area | +0,745 |
| price_per_m2 | +0,652 |
| log_price_per_m2 | +0,635 |
| bedroom_count | +0,563 |
| bathroom_count | +0,463 |
| zone_encoded | -0,269 |
| pub_month | +0,149 |
| quality_score | +0,104 |
| district_encoded | +0,033 |

### Biểu đồ 7 — Ma trận Tương quan Mở rộng

![eda_07_extended_correlation](plots/section_4/eda_07_extended_correlation.png)

**Nhận xét:**

- **5 biến tương quan mạnh nhất với giá**: area (0,77), price_per_m2 (0,65), bedroom_count (0,56), bathroom_count (0,46), zone_encoded (-0,27). Đây là nhóm features chính cho LightGBM.

- **Text features có tương quan yếu nhưng dương** với giá: corner_unit (+0,11), parking (+0,11), quality_score (+0,10). Dù tương quan tuyến tính thấp, chúng có thể có interaction effects phi tuyến mà LightGBM có thể nắm bắt.

- **Đa cộng tuyến cần lưu ý**: area~bedroom_count (r=0,75), bathroom_count~bedroom_count (r=0,58). Với LightGBM (tree-based), đa cộng tuyến không ảnh hưởng trực tiếp đến dự đoán, nhưng cần cẩn trọng khi diễn giải feature importance.

- **district_encoded tương quan gần 0 với giá** — do label encoding tạo thứ tự tùy ý. Tuy nhiên LightGBM sẽ chia tại các threshold phù hợp, nên vẫn hữu ích.

### Biểu đồ 8 — Pair Plot (Biến chính × Vùng)

![eda_08_pairplot](plots/section_4/eda_08_pairplot.png)

**Nhận xét:**

- **KDE trên đường chéo** cho thấy phân phối log_price gần chuẩn ở mỗi zone, xác nhận log transform hiệu quả.

- **Outer zone (xanh lá)** tạo thành cluster riêng biệt trong hầu hết các cặp biến — đây là tín hiệu tốt cho K-Means: dữ liệu có cấu trúc phân cụm tự nhiên.

- **Inner và Middle overlap đáng kể** trong không gian log_price – log_area — cho thấy 2 zone này khó phân biệt chỉ dựa trên giá và diện tích, cần thêm price_per_m2 và district để tách biệt.

---

## Part 5 — Xu hướng Bất thường & Phân khúc Giá

### 5.1 Phân bố theo Phân Khúc Giá

| Phân khúc | Số tin | % | % Inner | % Middle | % Outer |
|---|---|---|---|---|---|
| < 2 tỷ | 663 | 0,9% | 4,4% | 44,9% | 50,7% |
| 2–4 tỷ | 12.684 | 17,5% | 3,3% | 55,3% | 41,5% |
| 4–6 tỷ | 21.267 | 29,3% | 5,7% | 79,1% | 15,2% |
| 6–8 tỷ | 17.308 | 23,8% | 9,0% | 85,2% | 5,8% |
| 8–10 tỷ | 9.601 | 13,2% | 9,6% | 85,8% | 4,6% |
| 10–15 tỷ | 8.776 | 12,1% | 13,6% | 84,4% | 2,0% |
| > 15 tỷ | 2.305 | 3,2% | 17,1% | 82,5% | 0,4% |

### 5.2 Quality Score vs Giá

| Score | Số tin | Giá TB (tỷ) | Giá/m² TB (tr) |
|---|---|---|---|
| 0 | 4.603 | 6,28 | 80,8 |
| 1 | 10.364 | 6,49 | 79,8 |
| 2 | 12.893 | 6,81 | 80,4 |
| 3 | 11.625 | 7,02 | 80,6 |
| 5 | 7.506 | 6,96 | 80,4 |
| 8 | 2.920 | 7,62 | 85,0 |
| 11 | 391 | 8,23 | 88,8 |
| 14 | 8 | 9,34 | 94,4 |

### 5.3–5.4 Top 10 Phường Đắt & Rẻ nhất

**Top 5 Phường ĐẮT nhất (median giá/m²):**

| Phường | Quận | Số tin | Median Giá/m² (tr) | Median Giá (tỷ) |
|---|---|---|---|---|
| Liễu Giai | Ba Đình | 99 | 166,7 | 13,20 |
| Quảng An | Tây Hồ | 100 | 147,3 | 16,50 |
| Thụy Khuê | Tây Hồ | 166 | 145,0 | 15,40 |
| Giảng Võ | Ba Đình | 81 | 119,0 | 10,00 |
| Thành Công | Ba Đình | 319 | 117,8 | 11,75 |

**Top 5 Phường RẺ nhất:**

| Phường | Quận | Số tin | Median Giá/m² (tr) | Median Giá (tỷ) |
|---|---|---|---|---|
| Cự Khê | Thanh Oai | 139 | 42,3 | 3,10 |
| Thạch Hòa | Thạch Thất | 243 | 43,6 | 2,48 |
| Kiến Hưng | Hà Đông | 761 | 50,0 | 3,39 |
| Tân Lập | Đan Phượng | 127 | 52,2 | 3,60 |
| Tả Thanh Oai | Thanh Trì | 726 | 53,2 | 2,79 |

### Biểu đồ 9 — Phân Khúc Giá & Quality Score

![eda_09_segments_quality](plots/section_4/eda_09_segments_quality.png)

**Nhận xét:**

- **Phân khúc 4–6 tỷ chiếm đa số (29,3%)**, tiếp theo 6–8 tỷ (23,8%) — hai phân khúc này cộng lại chiếm >53% thị trường. Đây là "mainstream market" của căn hộ chung cư Hà Nội.

- **Cấu trúc địa lý thay đổi rõ theo phân khúc giá**:
  - Phân khúc < 2 tỷ: 50,7% thuộc outer → đây là vùng giá rẻ ngoại thành
  - Phân khúc > 15 tỷ: 17,1% inner, 82,5% middle, chỉ 0,4% outer → cao cấp tập trung ở nội thành và cận nội thành
  - Tỷ lệ inner tăng dần từ 3,3% (2–4 tỷ) lên 17,1% (>15 tỷ)

- **Quality Score có quan hệ tuyến tính dương với giá**: mỗi điểm quality_score tăng thêm, giá trung bình tăng ~0,22 tỷ VND. Score 14 (cao nhất) có giá trung bình 9,34 tỷ — cao hơn 49% so với score 0 (6,28 tỷ).

- **Giá/m² cũng tăng theo quality_score**: từ 80 tr/m² (score 0–5) lên 94 tr/m² (score 14) — cho thấy chất lượng mô tả tin đăng tương quan thực sự với phân khúc giá chứ không chỉ là "marketing nổ".

### Biểu đồ 10 — Top 10 Phường Đắt & Rẻ nhất

![eda_10_ward_extremes](plots/section_4/eda_10_ward_extremes.png)

**Nhận xét:**

- **Chênh lệch giá/m² cực kỳ lớn**: Liễu Giai (Ba Đình) có giá 166,7 tr/m², cao gấp **3,9 lần** Cự Khê (Thanh Oai) tại 42,3 tr/m². Đây là khoảng cách lớn nhất trong cùng thành phố Hà Nội.

- **Top đắt tập trung ở Ba Đình và Tây Hồ**: 5/10 phường đắt nhất thuộc 2 quận này — khu vực ven Hồ Tây và trung tâm chính trị Ba Đình là "prime location" của Hà Nội.

- **Phát hiện bất ngờ**: Đống Đa có cả phường đắt (Láng Thượng: 110 tr/m²) và phường rẻ (Phương Liên–Trung Tự: 54 tr/m², Nam Đồng: 58 tr/m²) — cho thấy sự phân hóa mạnh **trong cùng một quận**. Điều này quan trọng cho K-Means: clustering nên xem xét đến cấp phường, không chỉ quận.

- **Hà Đông xuất hiện ở cả 2 danh sách**: Kiến Hưng (50 tr/m²) nằm trong top rẻ nhất — xác nhận Hà Đông là quận có phân hóa giá mạnh trong middle zone.

---

## Part 6 — Khám phá Cấu trúc PCA (Tiền K-Means)

### PCA Explained Variance

| Component | Variance (%) | Tích lũy (%) |
|---|---|---|
| PC1 | 50,3% | 50,3% |
| PC2 | 20,6% | 70,9% |
| PC3 | 16,6% | 87,6% |

### PCA Component Loadings

| Feature | PC1 | PC2 | PC3 |
|---|---|---|---|
| log_price | **0,519** | **0,356** | -0,012 |
| log_area | **0,520** | -0,213 | -0,007 |
| bedroom_count | **0,477** | -0,296 | -0,020 |
| bathroom_count | **0,431** | -0,253 | -0,014 |
| log_price_per_m2 | 0,214 | **0,822** | -0,011 |
| district_encoded | 0,029 | 0,002 | **1,000** |

### Biểu đồ 11 — PCA Visualization

![eda_11_pca_exploration](plots/section_4/eda_11_pca_exploration.png)

**Nhận xét:**

- **3 PC giải thích 87,6% tổng phương sai** — cấu trúc dữ liệu có thể được nén hiệu quả, hứa hẹn cho K-Means clustering.

- **PC1 (50,3%) = "Tổng giá trị căn hộ"**: loading cao ở log_price (0,52), log_area (0,52), bedroom (0,48), bathroom (0,43). PC1 đo lường kích thước và giá tổng thể — căn hộ lớn, nhiều phòng, giá cao đều có PC1 dương.

- **PC2 (20,6%) = "Chất lượng vị trí / Premium"**: loading rất cao ở log_price_per_m2 (0,82). PC2 tách biệt càn hộ có đơn giá cao (vị trí đắc địa) khỏi căn hộ có đơn giá thấp, bất kể kích thước.

- **PC3 (16,6%) = "Vị trí địa lý"**: loading gần tuyệt đối ở district_encoded (1,00). PC3 gần như hoàn toàn phản ánh quận — biến này orthogonal với price và area.

- **Biplot (hình phải)** cho thấy log_price nằm giữa nhóm "kích thước" (area, bedroom, bathroom) và "chất lượng" (price_per_m2) — xác nhận giá là hàm của cả hai yếu tố.

- **Scatter PC1 vs PC2 theo vùng**: Outer zone (xanh lá) tách biệt rõ ở vùng PC1 thấp (căn hộ nhỏ, rẻ), Inner (đỏ) phân tán ở vùng PC2 cao (đơn giá cao). Tín hiệu rất tích cực cho K-Means.

---

## Part 7 — Phân tích Text Features

### 7.1 Tương quan Text Features với Giá

| Feature | Pearson r |
|---|---|
| Căn góc | +0,113 |
| Bãi đỗ xe | +0,106 |
| Quality Score | +0,104 |
| View đẹp | +0,089 |
| Bể bơi | +0,080 |
| Gym | +0,079 |
| Ban công | +0,069 |
| Gần siêu thị | +0,049 |
| Bảo vệ 24/7 | +0,041 |
| Thang máy | +0,040 |
| Ánh sáng tự nhiên | +0,034 |
| Gần công viên | +0,031 |
| Sân chơi | +0,025 |
| Nội thất ĐĐ | +0,018 |
| Sổ đỏ | +0,005 |
| Pháp lý ĐĐ | +0,004 |
| Gần bệnh viện | +0,004 |
| **Gần trường học** | **-0,044** |

### 7.2 Top 10 Co-occurrence Pairs

| Feature 1 | Feature 2 | Số lần cùng xuất hiện |
|---|---|---|
| Nội thất ĐĐ | Ban công | 15.656 |
| Gần trường học | Gần siêu thị | 15.655 |
| Gần siêu thị | Gần công viên | 13.444 |
| Nội thất ĐĐ | Sổ đỏ | 13.131 |
| Gần trường học | Gần công viên | 13.099 |

### Biểu đồ 12 — Text Features Analysis

![eda_12_text_features_analysis](plots/section_4/eda_12_text_features_analysis.png)

**Nhận xét:**

- **Nhóm "physical features"** (căn góc +0,11, bãi đỗ xe +0,11) có tương quan mạnh nhất với giá — xác nhận đặc điểm vật lý căn hộ là yếu tố định giá quan trọng.

- **Nhóm "lifestyle amenities"** (bể bơi, gym, view đẹp: +0,08–0,09) tương quan trung bình — đây là đặc trưng của phân khúc trung-cao cấp.

- **"Gần trường học" là feature duy nhất tương quan âm** (-0,044) — hiện tượng confounding đã phân tích ở Section 3: khu vực có nhiều trường thường ở ngoại thành với giá thấp.

- **Co-occurrence patterns quan trọng**: "Nội thất ĐĐ + Ban công" và "Gần trường + Gần siêu thị" là 2 cặp phổ biến nhất — cho thấy người đăng tin thường mô tả theo "bộ" (nhóm tiện ích nội thất hoặc nhóm tiện ích vị trí).

- **Heatmap theo zone**: Inner zone có tỷ lệ cao hơn ở gym, bể bơi, parking — phản ánh chất lượng dự án cao cấp hơn. Outer zone lại cao hơn ở "gần trường học" — xác nhận confounding effect.

---

## Part 8 — Phân tích Chéo Diện tích × Vùng

### Thống kê theo Phân Khúc Diện Tích

| Phân khúc | Số tin | Giá TB (tỷ) | Median (tỷ) | Giá/m² TB (tr) | PN TB |
|---|---|---|---|---|---|
| Studio (<45m²) | 4.491 | 3,19 | 3,12 | 85,3 | 1,4 |
| Nhỏ (45–65m²) | 14.705 | 4,41 | 4,25 | 76,6 | 1,9 |
| TB (65–80m²) | 18.835 | 5,75 | 5,50 | 78,6 | 2,2 |
| Khá (80–100m²) | 16.005 | 7,42 | 7,00 | 82,1 | 2,7 |
| Lớn (100–130m²) | 12.886 | 9,88 | 9,27 | 86,5 | 3,0 |
| Penthouse (>130m²) | 5.682 | 12,58 | 12,06 | 82,4 | 3,4 |

### Biểu đồ 13 — Cross-analysis Diện Tích × Vùng

![eda_13_area_zone_cross](plots/section_4/eda_13_area_zone_cross.png)

**Nhận xét:**

- **Studio (<45m²) có giá/m² cao nhất** (85,3 tr/m²) — ngược trực giác! Điều này do căn studio thường nằm ở dự án cao cấp trung tâm (designed for investment/rental), nên đơn giá cao dù tổng giá thấp.

- **Phân khúc 45–65m² có giá/m² thấp nhất** (76,6 tr/m²) — đây là căn hộ "phổ thông" nhất, thường ở dự án mass-market ngoại thành.

- **Giá/m² tăng trở lại từ 80m² lên** — phân khúc lớn hơn thường ở dự án tốt hơn, nên cả tổng giá lẫn đơn giá tăng.

- **Cross-analysis: Gap inner vs outer tăng khi diện tích tăng**: ở Studio, inner chênh outer ~35%; ở Penthouse, inner chênh outer ~55%. Phân khúc cao cấp phân hóa mạnh hơn theo vị trí.

---

## Tổng hợp Key Findings — Hướng dẫn Lựa chọn Data Mining

### 1. K-Means Clustering có cơ sở vững chắc

- 3 vùng địa lý (inner/middle/outer) đã cho thấy phân khúc tự nhiên rõ ràng
- PCA xác nhận: 3 PC giải thích 87,6% variance, với 3 chiều ý nghĩa riêng biệt (tổng giá trị, chất lượng vị trí, vị trí địa lý)
- Outer zone tách biệt rõ trong không gian PCA → K-Means có thể phát hiện thêm các phân khúc mà phân loại thủ công chưa nắm bắt

### 2. LightGBM Regression phù hợp cho dữ liệu này

- Quan hệ price~features là **đa chiều và phi tuyến**: cùng 80m² có thể 2 tỷ hoặc 15 tỷ tùy vị trí
- Nhiều **interaction effects**: diện tích × zone, bedroom × zone, text features × vị trí
- **Đa cộng tuyến** (area~bedroom: r=0,75) không ảnh hưởng LightGBM nhưng ảnh hưởng linear regression → chọn tree-based model là đúng
- 51 features (bao gồm 17 text + quality_score) tạo feature space phong phú cho model

### 3. Variables quan trọng nhất cần đưa vào models

| Nhóm | Features | Lý do |
|---|---|---|
| Kích thước | area, bedroom_count, bathroom_count | Tương quan mạnh nhất với giá |
| Vị trí | district_encoded, zone_encoded, ward | Quyết định giá/m² |
| Chất lượng | quality_score, feat_parking, feat_corner_unit | Đặc trưng premium với effect rõ |
| Thời gian | pub_month | Xu hướng giá tăng theo tháng |

### 4. Patterns bất thường cần chú ý

- **Hà Đông**: middle zone nhưng giá/m² thấp hơn một số outer zone → có thể tạo cluster riêng
- **Studio cao cấp**: giá/m² cao hơn căn hộ lớn → phân khúc đặc biệt trong clustering
- **"Gần trường học" tương quan âm**: confounding effect, LightGBM cần kiểm soát biến vị trí

---
