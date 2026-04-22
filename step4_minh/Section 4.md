# Section 4 — Exploratory Data Analysis and Descriptive Mining (15%)

> **Script:** `section_4_eda.py`
>
> **Input:**
> - `../step3_minh/data/hanoi_apartments_processed.csv` — 72.604 bản ghi × 37 cột
> - `../step3_minh/data/hanoi_apartments_for_clustering.csv` — 72.604 bản ghi × 8 cột (6 scaled + 2 label)
>
> **Output:** 13 biểu đồ PNG trong `plots_section_4/`

---

## Tổng quan phân tích (8 Parts — 13 Plots)

| Part | Nội dung | Biểu đồ |
|---|---|---|
| Part 1 | Phân phối theo Vùng địa lý & So sánh theo Quận | eda_01, eda_02 |
| Part 2 | So sánh Giá theo Số phòng ngủ / phòng tắm & Hướng ban công | eda_03, eda_04 |
| Part 3 | Xu hướng thị trường theo tháng (T6–T12/2025) | eda_05 |
| Part 4 | Mối quan hệ Giá–Diện tích (Scatter + Hexbin) | eda_06 |
| Part 4 | Ma trận tương quan mở rộng & Pair Plot | eda_07, eda_08 |
| Part 5 | Phân khúc giá tự nhiên, Quality Score & Top phường | eda_09, eda_10 |
| Part 6 | Khám phá cấu trúc PCA (tiền K-Means) | eda_11 |
| Part 7 | Text Features — Tương quan với giá & phân bố theo vùng | eda_12 |
| Part 8 | Phân tích chéo Diện tích × Vùng địa lý | eda_13 |

---

## Part 1 — Phân phối theo Vùng & So sánh theo Quận

### Plot 1 — `eda_01_zone_distributions.png`: Violin Plot 3 Vùng

![eda_01](plots_section_4/eda_01_zone_distributions.png)

Biểu đồ violin hiển thị phân phối Giá, Diện tích và Giá/m² của 3 zone địa lý.

**Thống kê tổng quan theo Zone:**

| Zone | Số tin | Tỷ lệ | Giá TB (tỷ) | Median Giá (tỷ) | DT TB (m²) | Giá/m² TB (tr) | Giá/m² Median (tr) |
|---|---|---|---|---|---|---|---|
| **Inner** (nội thành lõi) | 6.131 | 8,5% | 7,78 | 6,60 | 86,8 | 90,0 | 84,0 |
| **Middle** (vành đai 2-3) | 55.111 | 75,9% | 7,36 | 6,70 | 88,6 | 82,7 | 80,0 |
| **Outer** (ngoại thành) | 11.362 | 15,6% | 4,55 | 4,20 | 67,6 | 68,3 | 66,7 |

**Phân tích từ biểu đồ:**

- **Giá (violin trái):** Inner có đuôi trên kéo dài tới ~22 tỷ và IQR rộng nhất — phản ánh sự hiện diện của phân khúc siêu cao cấp. Middle có hình dạng "eo thắt" đặc trưng — tập trung mạnh ở dải 4–8 tỷ, ít phân tán hơn. Outer hẹp và thấp rõ rệt, median ~4,2 tỷ.
- **Diện tích (violin giữa):** Inner và Middle có diện tích tương đương (median ~80 m²), Outer nhỏ hơn (~65 m²). Cả 3 zone đều có đuôi kéo đến >200 m² (căn penthouse).
- **Giá/m² (violin phải):** Inner cao nhất (median ~84 tr/m²), Middle (~80 tr/m²), Outer (~67 tr/m²). Inner có outlier cực cao tới ~450 tr/m² — các căn hộ siêu sang ven hồ Tây Bắc.
- **Nhận xét:** Inner và Middle gần nhau về median nhưng Inner có đuôi phân phối cao hơn hẳn — cho thấy Middle zone là thước đo thị trường đại chúng, Inner là thị trường premium.

---

### Plot 2 — `eda_02_district_price_comparison.png`: So sánh theo Quận

![eda_02](plots_section_4/eda_02_district_price_comparison.png)

Biểu đồ cột ngang so sánh Median Giá và Median Giá/m² toàn bộ các quận, tô màu theo zone.

**Quận dẫn đầu & đội sổ:**

| Hạng | Quận (Median Giá) | Giá (tỷ) | Quận (Median Giá/m²) | Giá/m² (tr) |
|---|---|---|---|---|
| Cao nhất | Tây Hồ | ~9,9 | Tây Hồ | ~101,4 |
| 2 | Cầu Giấy | ~8,6 | Đống Anh | ~93,1 |
| 3 | Bắc Từ Liêm | ~7,9 | Ba Đình | ~92,9 |
| 4 | Thanh Xuân | ~7,8 | Bắc Từ Liêm | ~90,0 |
| Thấp nhất | Mê Linh | ~1,9 | Mê Linh | ~19,2 |

**Phân tích:**
- **Tây Hồ dẫn đầu tuyệt đối** cả hai tiêu chí nhờ cụm căn hộ siêu sang ven hồ (Quảng An, Thụy Khuê). Khoảng cách giá/m² giữa Tây Hồ (101 tr) và Mê Linh (19 tr) là **hơn 5 lần** — minh chứng cho sự phân hóa địa lý cực đoan.
- **Đống Anh** (outer zone) đứng thứ 2 về giá/m² — bất thường so với kỳ vọng. Nguyên nhân: sự xuất hiện của các dự án chung cư cao cấp đang phát triển nhanh tại khu vực này.
- **Middle zone** (Cầu Giấy, Bắc Từ Liêm, Thanh Xuân, Nam Từ Liêm) chiếm phần lớn trung vị bảng xếp hạng — xác nhận đây là khu vực giao dịch cốt lõi của thị trường.
- Màu sắc (đỏ = inner, xanh dương = middle, xanh lá = outer) cho thấy outer zone nằm toàn bộ ở đáy bảng giá tuyệt đối, nhưng có một số ngoại lệ ở giá/m².

---

## Part 2 — So sánh theo Phòng & Hướng Ban Công

### Plot 3 — `eda_03_rooms_vs_price.png`: Giá theo Số Phòng Ngủ & Phòng Tắm

![eda_03](plots_section_4/eda_03_rooms_vs_price.png)

Boxplot so sánh phân phối giá theo số phòng ngủ (trái) và số phòng tắm (phải).

**Số liệu chi tiết — Phòng ngủ:**

| Số PN | Số tin | Median Giá (tỷ) | IQR | Nhận xét |
|---|---|---|---|---|
| 1 PN | 4.737 | ~3,5 | hẹp | Căn studio/1PN — phân khúc nhỏ |
| 2 PN | **35.088** | ~5,0 | hẹp-vừa | **Chiếm đa số (48%)** — thị trường cốt lõi |
| 3 PN | 29.903 | ~8,0 | vừa | Phổ biến thứ 2 (41%) |
| 4 PN | 2.772 | ~12,0 | rộng | Phân khúc cao cấp |
| 5 PN | 93 | ~10,0 | rất rộng | Cực hiếm, biến động lớn |

**Số liệu chi tiết — Phòng tắm:**

| Số WC | Số tin | Median Giá (tỷ) | Nhận xét |
|---|---|---|---|
| 1 WC | 10.721 | ~4,0 | Căn nhỏ/giá rẻ |
| 2 WC | **58.377** | ~6,5 | **Áp đảo (80%)** — cấu hình tiêu chuẩn |
| 3 WC | 3.284 | ~11,0 | Cao cấp |
| 4 WC | 205 | ~15,0 | Siêu cao cấp |

**Phân tích:**
- Giá tăng đơn điệu (monotonic) theo số phòng ngủ từ 1→4 PN — quan hệ tuyến tính rõ ràng.
- Căn 5 PN có IQR rực rộng và median (~10 tỷ) thấp hơn 4 PN — do mẫu nhỏ (n=93) và có thể bao gồm cả căn phân lô/nhà phố bị phân loại nhầm.
- Cấu hình **2PN + 2WC** là cấu hình tiêu chuẩn thị trường Hà Nội, chiếm tỷ trọng áp đảo.

---

### Plot 4 — `eda_04_direction_vs_price.png`: Giá/m² theo Hướng Ban Công

![eda_04](plots_section_4/eda_04_direction_vs_price.png)

Boxplot phân phối Giá/m² theo từng hướng ban công, sắp xếp từ thấp đến cao, đường đỏ là median tổng (77,6 tr/m²).

**Kết quả median Giá/m² từng hướng (thấp → cao):**

| Hướng | Median Giá/m² (tr) | So sánh median tổng |
|---|---|---|
| Tây Nam | ~75–76 | Dưới median |
| Tây Bắc | ~76–77 | Dưới / ngang median |
| Đông | ~77 | Ngang median |
| Đông Bắc | ~79 | Trên median |
| Đông Nam | ~79 | Trên median |
| Tây | ~81 | Trên median |
| Bắc | ~82–83 | Trên median |
| **Nam** | **~83–84** | **Cao nhất** |

**Phân tích & Phát hiện quan trọng:**
- Các hộp IQR của 8 hướng gần như **bằng nhau về độ rộng** (~65–100 tr/m²) — cho thấy hướng ban công có tác động biên (marginal effect), không phải yếu tố định giá chủ đạo.
- **Hướng Nam và Bắc cao nhất** (trực giác: thoáng mát và ít nắng gắt mùa hè). Trái với quan niệm dân gian, **Đông Nam không dẫn đầu** dù được ưa chuộng theo phong thủy truyền thống.
- **Lý giải:** Người mua định giá căn hộ dựa vào vị trí (Zone/Quận), pháp lý và tiện ích nhiều hơn là hướng. Hướng ban công chỉ là tín hiệu phụ trợ, bị confounding mạnh bởi vị trí địa lý.
- **Hàm ý cho model:** Các cột `balcony_dir_*` (OHE từ Step 3) sẽ có feature importance thấp trong LightGBM — đây là kết quả mong đợi, không phải lỗi thiết kế.

---

## Part 3 — Xu hướng Thị trường theo Tháng

### Plot 5 — `eda_05_monthly_trends.png`: Xu hướng T6–T12/2025

![eda_05](plots_section_4/eda_05_monthly_trends.png)

Ba sub-plot: Số lượng tin đăng mới, Giá trung bình, và Giá/m² trung bình theo từng tháng.

**Số liệu chi tiết:**

| Tháng | Số tin đăng | Giá TB (tỷ) | Giá/m² TB (tr) |
|---|---|---|---|
| T6/2025 | 7.511 | ~6,2 | ~70 |
| T7/2025 | 11.613 | ~6,5 | ~74 |
| T8/2025 | 10.582 | ~6,7 | ~78 |
| T9/2025 | 10.123 | ~7,0 | ~81 |
| T10/2025 | 10.270 | ~7,3 | ~84 |
| T11/2025 | 10.401 | ~7,5 | ~86 |
| T12/2025 | 11.796 | ~7,5–7,6 | ~88 |

**Phân tích:**
- **Số lượng tin đăng:** T6 thấp (~7.500) do tháng đầu thu thập, tăng vọt T7 (~11.600) rồi ổn định ~10.000–10.500/tháng. T12 tăng nhẹ trở lại (~11.800) — có thể do cuối năm nhiều người chốt giao dịch.
- **Giá trung bình tăng đều và liên tục** từ 6,2 tỷ (T6) lên 7,5 tỷ (T12) — tăng khoảng **21% trong 7 tháng**. Đây là xu hướng tăng thực, không phải bias mẫu.
- **Giá/m² tăng từ ~70 lên ~88 tr/m²** (+26%) — tốc độ tăng cao hơn giá tuyệt đối, cho thấy không chỉ diện tích mà chất lượng/vị trí của căn hộ được đăng cũng tăng dần theo thời gian (người bán đăng căn tốt hơn về cuối năm).
- **Hàm ý cho model:** Biến `pub_month` và `pub_year` (Feature Engineering Step 3) nắm bắt được xu hướng thời gian này — là predictor bổ trợ quan trọng cho LightGBM.

---

## Part 4 — Mối quan hệ Giá–Diện tích & Ma trận Tương quan

### Plot 6 — `eda_06_price_area_scatter.png`: Scatter và Hexbin Giá–Diện tích

![eda_06](plots_section_4/eda_06_price_area_scatter.png)

**Phân tích:**
- **Scatter (trái, ~9.000 điểm mẫu):** Mối quan hệ dương tuyến tính rõ ràng giữa diện tích và giá. Inner zone (đỏ) phân bố ở phần trên của đám mây điểm — cùng diện tích nhưng giá cao hơn Middle và Outer. Outer (xanh lá) tập trung phía dưới.
- **Hexbin (phải — toàn bộ 72.604 điểm):** Khối lượng cực lớn tập trung tại vùng **50–100 m² × 3–8 tỷ** (màu đỏ đậm, mật độ >1.200 điểm/ô). Vùng này đại diện cho "thị trường đại chúng" Hà Nội. Phần đuôi (>150 m² hoặc >15 tỷ) rất thưa thớt.
- **Nhận xét:** Quan hệ giá–diện tích là phi tuyến nhẹ (tuyến tính trên log scale) — xác nhận quyết định dùng `log_price` và `log_area` từ Step 3 là đúng đắn.

---

### Plot 7 — `eda_07_extended_correlation.png`: Ma trận Tương quan Mở rộng

![eda_07](plots_section_4/eda_07_extended_correlation.png)

Ma trận tương quan Pearson 15 biến: 5 biến số + 9 text features + quality_score (hiển thị tam giác dưới).

**Các cặp tương quan nổi bật với Giá:**

| Biến | r với Giá | Nhận xét |
|---|---|---|
| Diện tích | **+0.77** | Tương quan mạnh nhất — driver chính của giá tuyệt đối |
| Giá/m² | +0.65 | Tương quan khá — vị trí tốt vừa diện tích lớn vừa đắt/m² |
| Phòng ngủ | +0.56 | Tương quan vừa — size proxy |
| Phòng tắm | +0.46 | Tương quan vừa |
| Căn góc | **+0.11** | Tương quan cao nhất trong text features |
| Ban công | +0.07 | Tương quan dương nhỏ |
| Quality Score | +0.06 | Tổng tiện ích dương với giá |
| Tiện ích VIP | +0.05 | Tương quan dương — dự án cao cấp |
| Gần siêu thị | +0.05 | Tương quan dương nhỏ |
| Gần trường | **−0.04** | **Tương quan âm** (xem giải thích Part 7) |

**Tương quan giữa các biến số:**
- `Diện tích` ↔ `Phòng ngủ` = 0.75 — tương quan mạnh, chấp nhận được với tree-based model
- `Giá/m²` ↔ `Diện tích` ≈ 0.07 — gần zero, xác nhận `price_per_m2` là biến độc lập tốt
- Text features với nhau: hầu hết gần zero (< 0.15) — ít đa cộng tuyến giữa các text features

---

### Plot 8 — `eda_08_pairplot.png`: Pair Plot 4 Biến chính theo Vùng

![eda_08](plots_section_4/eda_08_pairplot.png)

Pair plot 4 biến (`log_price`, `log_area`, `bedroom_count`, `price_per_m2_tr`) với hue theo zone (sample 5.000).

**Phân tích:**
- **`log_price` vs `log_area` (góc trên-trái):** Tương quan tuyến tính rõ trên log scale. Inner (đỏ) nằm trên/phải — cùng diện tích log nhưng giá log cao hơn.
- **`log_price` vs `bedroom_count`:** Pattern bậc thang (step-wise) — số phòng là biến rời rạc. Mỗi bậc phòng ngủ ứng với một dải giá khác nhau.
- **`log_price` vs `price_per_m2_tr`:** Tương quan dương, inner zone có đuôi phải cao hơn — xác nhận giá cao đi kèm giá/m² cao tại nội thành.
- **KDE đường chéo:** `log_price` của 3 zone: inner và middle phân phối chồng lên nhau nhiều (nhiều inner có giá bằng middle), outer lệch trái rõ. `price_per_m2_tr` của inner có đuôi phải dài hơn — căn siêu cao cấp kéo phân phối.

---

## Part 5 — Phân khúc Giá & Quality Score

### Plot 9 — `eda_09_segments_quality.png`: Phân khúc Giá và Quality Score

![eda_09](plots_section_4/eda_09_segments_quality.png)

Bốn sub-plot: phân bố tin theo phân khúc giá, cấu trúc zone trong mỗi phân khúc, giá TB theo quality score, giá/m² TB theo quality score.

**Phân bố tin đăng theo phân khúc giá:**

| Phân khúc | Số tin | Tỷ lệ | Zone chủ đạo |
|---|---|---|---|
| < 2 tỷ | 663 | 0,9% | Outer ~65% |
| 2–4 tỷ | ~12.884 | ~17,8% | Outer + Middle |
| **4–6 tỷ** | **~15.308** | **~21,1%** | Middle áp đảo |
| **6–8 tỷ** | **~17.368** | **~21,8%** | Middle áp đảo |
| 8–10 tỷ | ~9.601 | ~13,2% | Middle + Inner |
| 10–15 tỷ | ~6.771 | ~9,3% | Inner tăng dần |
| > 15 tỷ | ~2.305 | ~3,2% | Inner dominant |

> **Phân khúc 4–8 tỷ chiếm ~43% tổng cung** — đây là vùng "cháy hàng" của thị trường Hà Nội.

**Cấu trúc zone theo phân khúc (stacked bar):** Càng lên phân khúc cao, tỷ lệ Inner (đỏ) tăng dần trong khi Outer (xanh lá) biến mất hoàn toàn ở phân khúc > 10 tỷ. Middle (xanh dương) chiếm > 80% ở phân khúc 4–8 tỷ.

**Quality Score vs Giá:**
- **Trend giá tổng:** +0,12 tỷ/điểm score — tăng tuyến tính nhẹ nhưng đều đặn
- **Trend giá/m²:** +0,30 tr/m²/điểm score — tương quan dương rõ hơn
- Từ Score 0 → Score 9: giá trung bình tăng từ ~6,7 tỷ lên ~8,0 tỷ (+19%), giá/m² tăng từ ~79 tr lên ~81+ tr

---

### Plot 10 — `eda_10_ward_extremes.png`: Top 10 Phường Đắt & Rẻ nhất

![eda_10](plots_section_4/eda_10_ward_extremes.png)

So sánh Median Giá/m² của Top 10 phường đắt nhất (đỏ) và rẻ nhất (xanh lá).

**Top 10 Phường ĐẮT nhất:**

| Hạng | Phường | Quận | Median Giá/m² (tr) |
|---|---|---|---|
| 1 | Liễu Giai | Ba Đình | **166,7** |
| 2 | Quảng An | Tây Hồ | 147,3 |
| 3 | Thụy Khuê | Tây Hồ | 145,0 |
| 4 | Giảng Võ | Ba Đình | 119,0 |
| 5 | Thành Công | Ba Đình | 117,8 |
| 6 | Dịch Vọng Hậu | Cầu Giấy | 117,2 |
| 7 | Láng Thượng | Đống Đa | 109,7 |
| 8 | Xuân Tảo | Bắc Từ Liêm | 107,1 |
| 9 | Vĩnh Tuy | Hai Bà Trưng | 103,8 |
| 10 | Xuân La | Tây Hồ | 100,9 |

**Top 10 Phường RẺ nhất:**

| Hạng | Phường | Quận | Median Giá/m² (tr) |
|---|---|---|---|
| 1 | Cự Khê | Thanh Oai | **42,3** |
| 2 | Thạch Hòa | Thạch Thất | 43,6 |
| 3 | Kiến Hưng | Hà Đông | 50,0 |
| 4 | Tân Lập | Đan Phượng | 52,2 |
| 5 | Tả Thanh Oai | Thanh Trì | 53,2 |
| 6 | Phương Liên–Trung Tự | Đống Đa | 54,2 |
| 7 | Phúc La | Hà Đông | 55,2 |
| 8 | Thạch Bàn | Long Biên | 57,0 |
| 9 | Nam Đồng | Đống Đa | 58,3 |
| 10 | Thanh Nhàn | Hai Bà Trưng | 58,5 |

**Phân tích:**
- **Khoảng cách đỉnh–đáy gần 4 lần:** Liễu Giai (166,7 tr/m²) vs Cự Khê (42,3 tr/m²) — khoảng cách này **lớn hơn nhiều** so với khoảng cách giữa các quận, chứng minh rằng **phường có tính quyết định hơn quận** trong việc định giá.
- **Ba Đình + Tây Hồ chiếm trọn Top 5** đắt — đây là các phường ven hồ (Hồ Tây) và quanh trung tâm chính trị Hà Nội.
- **Xuân Tảo (Bắc Từ Liêm) ở hạng 8 đắt nhất** — bất ngờ với một quận Middle Zone, giải thích bởi sự phát triển mạnh của khu Ciputra và các dự án cao cấp Tây Hà Nội.
- **Phương Liên–Trung Tự (Đống Đa) và Nam Đồng (Đống Đa) ở top rẻ** dù thuộc quận nội thành — cho thấy nội ô Hà Nội cũng có vùng giá thấp nếu dự án lớn chưa phát triển.
- **Hàm ý cho model:** `ward_name` (tên phường) là biến có granularity cao — LightGBM có thể học được phân biệt phường tốt hơn là chỉ dùng cấp quận.

---

## Part 6 — PCA: Tiền phân tích Clustering

### Plot 11 — `eda_11_pca_exploration.png`: PCA 3 thành phần chính

![eda_11](plots_section_4/eda_11_pca_exploration.png)

Ba sub-plot: Scatter PC1 vs PC2 tô màu theo Zone, tô màu theo Giá, và Biplot loadings.

**Phương sai giải thích được:**

| Thành phần | Phương sai giải thích | Tích lũy | Ý nghĩa vật lý |
|---|---|---|---|
| **PC1** | **50,3%** | 50,3% | "Tổng giá trị căn hộ" — log_price, log_area, bedroom_count cùng hướng |
| **PC2** | **20,6%** | 70,9% | "Chất lượng giá/m²" — log_price_per_m2 là loading chính |
| **PC3** | **16,6%** | **87,6%** | "Vị trí địa lý" — district_encoded |

**Phân tích từ biểu đồ:**

- **Scatter theo Zone (trái):** Outer zone (xanh lá) phân bố về phía âm của PC1 (giá trị nhỏ hơn), Inner (đỏ) và Middle (xanh dương) chồng nhau nhiều trên PC1 nhưng Inner có xu hướng cao hơn trên PC2.
- **Scatter theo Giá (giữa):** Gradient rõ ràng từ phải (xanh lá = rẻ) sang trái (đỏ/tối = đắt) trên trục PC1 — xác nhận PC1 là "trục giá trị". Màu tối tập trung ở góc trái-trên (PC1 thấp, PC2 cao) — đây là vùng căn nhỏ nhưng đắt/m².
- **Biplot loadings (phải):** 
  - `log_price` và `log_area` cùng hướng (→ phải trên): đây là trục PC1
  - `log_price_per_m2` hướng thẳng lên: trục PC2
  - `bedroom_count` và `log_area` gần song song: không tạo thêm dimension độc lập
  - `district_encoded` góc riêng biệt: xác nhận quận tạo ra dimension địa lý độc lập (PC3)

**Kết luận PCA:** 3 PC giải thích 87,6% variance — K-Means clustering trên 6 features scaled sẽ có cơ sở cấu trúc tốt để tạo ra các cụm có ý nghĩa kinh tế.

---

## Part 7 — Text Features: Tương quan & Phân bố theo Vùng

### Plot 12 — `eda_12_text_features_analysis.png`: Correlation và Heatmap theo Zone

![eda_12](plots_section_4/eda_12_text_features_analysis.png)

Trái: Tương quan Pearson từng text feature với Giá. Phải: Heatmap tỷ lệ xuất hiện của text features theo zone.

**Tương quan với Giá (sắp xếp giảm dần):**

| Feature | r với Giá | Nhận xét |
|---|---|---|
| `feat_corner_unit` | **+0,113** | Cao nhất — căn góc là signal chất lượng thực |
| `feat_balcony` | +0,069 | Ban công tương quan dương đáng kể |
| `quality_score` | +0,063 | Tổng tiện ích = proxy chất lượng tốt |
| `has_premium_amenities` | +0,050 | Hồ bơi/gym → phân khúc cao |
| `feat_near_mall` | +0,049 | Siêu thị/TTTM → vị trí trung tâm |
| `feat_near_park` | +0,031 | Công viên tương quan dương nhỏ |
| `feat_full_furniture` | +0,018 | Nội thất: tương quan yếu do keyword phổ biến |
| `has_legal_paper` | +0,014 | Pháp lý: yếu do đã là tiêu chuẩn thị trường |
| `feat_near_hospital` | +0,004 | Gần bệnh viện: gần như trung tính |
| `feat_near_school` | **−0,044** | **Âm** — xem phân tích chi tiết bên dưới |

**Heatmap tỷ lệ Text Features theo Zone (điểm nổi bật):**

| Feature | Inner | Middle | Outer | Nhận xét |
|---|---|---|---|---|
| `has_legal_paper` | **61,5%** | 56,8% | 36,1% | Inner cao nhất — người bán cao cấp nhấn mạnh pháp lý |
| `has_premium_amenities` | 12,8% | 20,3% | **29,2%** | **Outer cao nhất** — dự án ngoại thành cạnh tranh bằng tiện ích |
| `feat_full_furniture` | 40,9% | **46,5%** | 36,8% | Middle nhiều tin đăng full nội thất nhất |
| `feat_near_mall` | 23,9% | **33,4%** | 22,8% | Middle gần TTTM nhất — Vincom, Aeon tập trung vành đai 2 |
| `feat_balcony` | 34,8% | **45,6%** | 40,7% | Middle đề cập ban công nhiều nhất |

**Phân tích feat_near_school âm (−0,044):**
- Đây là kết quả confounding địa lý: "gần trường học" xuất hiện nhiều ở các quận Middle và Outer Zone đang phát triển (Nam Từ Liêm, Hà Đông, Hoàng Mai) — những nơi có giá thấp hơn Inner Zone. Model học được tín hiệu "gần trường = vùng ngoại vi = giá thấp hơn" thay vì học quan hệ nhân quả.
- Feature vẫn được giữ trong pipeline vì LightGBM sẽ học được tương tác ngữ cảnh (district × near_school) tốt hơn correlation đơn biến.
- **Insight quan trọng:** `has_premium_amenities` cao ở Outer Zone cho thấy các dự án mới ở ngoại thành đang dùng hồ bơi/gym như công cụ marketing để cạnh tranh với vị trí kém hơn. Đây là chiến lược phổ biến của các chủ đầu tư như Vinhomes, Ecopark.

---

## Part 8 — Phân tích Chéo: Diện tích × Vùng

### Plot 13 — `eda_13_area_zone_cross.png`: Median Giá & Giá/m² theo Phân khúc Diện tích × Zone

![eda_13](plots_section_4/eda_13_area_zone_cross.png)

Grouped bar chart so sánh Median Giá (trái) và Median Giá/m² (phải) theo 6 phân khúc diện tích × 3 zone.

**6 phân khúc diện tích:**

| Phân khúc | Diện tích | Inner (tỷ) | Middle (tỷ) | Outer (tỷ) |
|---|---|---|---|---|
| Studio | < 45 m² | ~3,6 | ~3,3 | ~2,7 |
| Nhỏ | 45–65 m² | ~4,2 | ~4,5 | ~3,7 |
| Trung bình | 65–80 m² | ~5,7 | ~5,8 | ~4,7 |
| Khá | 80–100 m² | ~8,5 | ~7,2 | ~6,0 |
| Lớn | 100–130 m² | ~10,5 | ~9,3 | ~7,1 |
| Penthouse | > 130 m² | ~13,0 | ~12,4 | ~9,0 |

**Giá/m² theo phân khúc — Phát hiện "Nghịch lý Studio":**

| Phân khúc | Inner (tr/m²) | Middle (tr/m²) | Outer (tr/m²) |
|---|---|---|---|
| **Studio** | **~107** | ~88 | ~74 |
| Nhỏ | ~70 | ~78 | ~66 |
| Trung bình | ~77 | ~79 | ~65 |
| Khá | **~110** | ~80 | ~65 |
| Lớn | ~93 | ~82 | ~62 |
| Penthouse | ~85 | ~81 | ~62 |

**Phân tích "Nghịch lý Studio" (Studio Inner = 107 tr/m²):**
- Studio tại Inner Zone có giá/m² **cao nhất hệ thống** (~107 tr/m²), vượt cả Lớn và Penthouse ở cùng zone.
- Lý giải: Studio tại Ba Đình, Hoàn Kiếm, Đống Đa là căn hộ mini nội thành — khan hiếm tuyệt đối, target đầu tư cho thuê ngắn hạn/Airbnb với tỷ suất cho thuê rất cao. Premium vị trí lấn át hoàn toàn hiệu ứng "căn nhỏ".
- Inner "Khá" (80–100 m²) cũng có spike ~110 tr/m² — các căn hộ cao cấp khu vực Liễu Giai, Quảng An trong phân khúc diện tích phổ biến.
- **Middle và Outer** không có nghịch lý này — giá/m² tăng đơn điệu theo diện tích, phản ánh thị trường đại chúng thông thường.

**Nhận xét tổng thể Plot 13:**
Inner zone luôn dẫn đầu giá tuyệt đối ở mọi phân khúc diện tích, nhưng khoảng cách so với Middle thu hẹp ở phân khúc lớn (Penthouse: Inner 13B vs Middle 12.4B = chỉ chênh 5%). Outer có khoảng cách lớn nhất ở phân khúc lớn và Penthouse (~4B thấp hơn Middle).

---

## Kết luận Step 4 — Knowledge Discovery

### 1. Kiến trúc Giá cả & Phân hóa Địa lý

- **Middle Zone là trụ cột thị trường** (75,9% lượng cung), phân khúc 4–8 tỷ chiếm ~43% — đây là "thị trường cốt lõi" Hà Nội. Mô hình cần dự đoán tốt nhất tại vùng này.
- **Phân hóa vi mô cấp Phường > Quận:** Khoảng cách giá/m² giữa phường đắt nhất (Liễu Giai 166,7 tr) và rẻ nhất (Cự Khê 42,3 tr) là **4x** — lớn hơn nhiều khoảng cách giữa Inner và Outer zone (1,3x). `ward_name` là signal địa lý quan trọng cho LightGBM.
- **Tây Hồ là outlier dương:** Median giá/m² 101 tr — vượt xa mọi quận Inner Zone khác nhờ hệ sinh thái hồ Tây.

### 2. Phát hiện Hành vi Thị trường

- **Hướng Ban công không phải yếu tố định giá chủ đạo:** IQR các hướng gần bằng nhau, Nam/Bắc dẫn đầu nhỏ (không phải Đông Nam theo truyền thống). Zone và chất lượng dự án mới là yếu tố quyết định.
- **Nghịch lý Studio Inner:** Studio (<45 m²) tại nội thành có giá/m² cao nhất (~107 tr) — thị trường đầu tư cho thuê ngắn hạn tạo ra premium vượt quy mô căn hộ.
- **Premium amenities chiến lược Outer Zone:** Outer zone đề cập tiện ích cao cấp (hồ bơi, gym) với tỷ lệ cao nhất (29,2%) — công cụ bù đắp vị trí kém.
- **Giá thị trường tăng liên tục T6→T12/2025:** +21% giá tuyệt đối, +26% giá/m² — xu hướng tăng mạnh cần được model học qua biến thời gian.

### 3. Cơ sở cho Model Step 5

- **PCA xác nhận K-Means:** 3 PC giải thích 87,6% variance, cấu trúc 3 chiều rõ ràng (Size, Quality, Location) → K-Means sẽ phát hiện được các cụm có ý nghĩa kinh tế thực.
- **Quan hệ phi tuyến + tương tác chéo:** Ví dụ Studio × Inner Zone, Outer Zone × premium amenities, ward × price — đây chính xác là những pattern LightGBM (tree-based) xử lý tốt nhất, Linear Regression không thể nắm bắt.
- **Feature importance dự kiến:** `area`, `district_name`, `ward_name`, `price_per_m2` (proxy), `bedroom_count`, `pub_month` sẽ dẫn đầu; `balcony_dir_*` và `has_legal_paper` sẽ có importance thấp — đúng với những gì EDA đã chỉ ra.
