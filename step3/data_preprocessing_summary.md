# Tiền xử lí và làm xạch dữ liệu


Dự án này ứng dụng các kỹ thuật Khai phá dữ liệu (Data Mining) để làm sạch dữ liệu bất động sản thực tế, từ đó phân khúc thị trường (K-Means Clustering) và dự đoán giá nhà (LightGBM Regression).

---

## 📊 1. Tổng quan Dữ liệu gốc (Trước khi xử lý)
Dữ liệu thô được thu thập (crawl) từ các trang web bất động sản với file gốc là `hanoi_apartments2.csv`. Vì là dữ liệu nhập tay từ người dùng (môi giới, chủ nhà), dữ liệu chứa rất nhiều nhiễu.

* **Số lượng mẫu:** 86,601 dòng.
* **Số lượng đặc trưng (Features):** 19 cột.
* **Các vấn đề gặp phải:**
  * Khuyết thiếu dữ liệu (Missing Values) ở các trường quan trọng như diện tích, giá, số phòng ngủ/tắm.
  * Tồn tại "giá ảo" (Outliers) như giá quá cao (100 tỷ) hoặc diện tích không có thực.
  * Định dạng text lộn xộn (VD: "Quận Cầu Giấy" và "cầu giấy" bị coi là 2 khu vực khác nhau).
  * Chứa text lẫn trong cột số (VD: Cột giá chứa chữ "Thỏa thuận").

---

## 🛠️ 2. Pipeline Tiền xử lý dữ liệu (Data Pre-processing)
Để chuẩn bị dữ liệu tốt nhất cho mô hình K-Means và LightGBM, nhóm đã áp dụng các kỹ thuật sau:

| Bước xử lý | Kỹ thuật áp dụng | Mục đích & Kết quả |
| :--- | :--- | :--- |
| **1. Handling Missing Values** | - Xóa các dòng thiếu Giá (`price`) hoặc Diện tích (`area`).<br>- Điền Missing Value bằng Median cho số phòng ngủ/tắm. | Đảm bảo không còn ô trống (NaN) để K-Means có thể tính toán khoảng cách, giữ lại tối đa dữ liệu. |
| **2. Treating Outliers** | - Tính `price_per_sqm` (Giá/m²).<br>- Dùng phương pháp **IQR** cắt đỉnh/đáy.<br>- Lọc diện tích (15m² - 300m²). | Loại bỏ hoàn toàn các tin đăng rác, giá ảo, diện tích ảo làm nhiễu mô hình. |
| **3. Data Cleaning** | - Ép kiểu cột Giá về dạng số (Float).<br>- Đưa Text quận/huyện về chữ thường, cắt bỏ tiền tố "Quận", "Huyện". | Đồng nhất dữ liệu categorical, loại bỏ nhiễu text. |
| **4. Feature Transformation** | - `Log-transform` cho biến Giá (`log_price`).<br>- Dùng `StandardScaler` chuẩn hóa diện tích và số phòng. | Kéo phân phối giá về dạng chuẩn giúp LightGBM học tốt hơn. Đưa các biến về cùng thang đo cho thuật toán K-Means. |
| **5. Data Aggregation** | - Tạo tính năng mới: Giá trung bình theo quận (`avg_district_price_sqm`).<br>- Tạo tỷ lệ: Giá căn hộ / Giá trung bình quận. | Tạo **ngữ cảnh thị trường** (Market context) giúp mô hình nhận biết căn hộ đắt hay rẻ so với mặt bằng chung. |
| **6. Encoding Categorical** | Dùng `LabelEncoder` mã hóa tên Quận sang dạng ID số. | Chuyển đổi toàn bộ Text sang Số hóa để tương thích với các thuật toán ML. |

---

## ✨ 3. Kết quả (Dữ liệu sau xử lý)
Sau khi chạy pipeline, tập dữ liệu đầu ra `processed_hanoi_apartments_fixed.csv` đã đạt chuẩn hoàn toàn:

* **Số lượng mẫu giữ lại:** 73,835 dòng (Dữ liệu sạch).
* **Trạng thái:** Không còn Missing Values (NaN), không còn Outliers nghiêm trọng.
* **Features mới:** Đã bổ sung thêm các Context Feature mang ý nghĩa kinh doanh cao.
* Tập dữ liệu này đã sẵn sàng 100% để đưa vào 2 bài toán:
  1. **Clustering:** Phân khúc thị trường bằng K-Means.
  2. **Regression:** Dự đoán giá nhà bằng LightGBM.

---

