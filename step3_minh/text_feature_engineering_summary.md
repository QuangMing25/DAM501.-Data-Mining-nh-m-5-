# Tổng hợp Đánh giá và Tối ưu Text Features (Feature Engineering)

Tài liệu này tổng hợp chiến lược trích xuất và tinh chỉnh các đặc trưng văn bản (Text Features) từ cột `description`. Việc tối ưu hóa giúp giảm nhiễu (noise), hạn chế đa cộng tuyến (multicollinearity), và ngăn chặn mô hình học phải các tín hiệu sai lệch.

## 1. Các Feature được sử dụng cho Mô hình (9 Features)

Các tính năng này mang ý nghĩa thiết thực, có tác động tích cực đến dự báo giá và đã được gộp/chỉnh sửa để giảm tối đa nhiễu lô-gic.

| STT | Tên Feature (Biến) | Tên tiếng Việt / Phân loại | Hành động xử lý | Nguồn gốc Keyword (Gộp từ đâu) |
| :--- | :--- | :--- | :--- | :--- |
| **Nhóm Nội tại** |||||
| 1 | `feat_full_furniture` | Nội thất đầy đủ | **Giữ nguyên** | Nội thất đầy đủ, full đồ, xách vali vào... |
| 2 | `feat_corner_unit` | Căn góc | **Giữ nguyên** | Căn góc |
| 3 | `feat_balcony` | Ban công | **Giữ nguyên** | Ban công |
| **Nhóm Gộp (Merged)** |||||
| 4 | `has_legal_paper` | Có giấy tờ pháp lý | **Gộp mới** | Đã gộp từ: `feat_red_book` + `feat_legal_full` |
| 5 | `has_premium_amenities`| Tiện ích nội khu cao cấp | **Gộp mới** | Đã gộp từ: `feat_swimming_pool` + `feat_gym` + `feat_playground` |
| **Nhóm Ngoại khu** |||||
| 6 | `feat_near_school` | Gần trường học | **Giữ nguyên** | Trường học, gần trường, trường quốc tế... |
| 7 | `feat_near_hospital` | Gần bệnh viện | **Giữ nguyên** | Bệnh viện, gần bệnh viện... |
| 8 | `feat_near_mall` | Gần siêu thị / TTTM | **Giữ nguyên** | Siêu thị, trung tâm thương mại, aeon, lotte... |
| 9 | `feat_near_park` | Gần công viên | **Giữ nguyên** | Công viên |

---

## 2. Các Feature bị LOẠI BỎ (Không đưa vào Model)

Tổng cộng 5 features bị loại khỏi pipeline do không đem lại lợi ích phân loại hoặc có thể trực tiếp làm sai mô hình học máy.

| STT | Tên Feature (Biến) | Ý nghĩa keyword | Lý do loại bỏ (Drop) |
| :--- | :--- | :--- | :--- |
| 1 | `feat_elevator` | Thang máy | **Hiển nhiên có:** Chung cư 99% phải có thang máy. Thiếu keyword không có nghĩa là chung cư đi bộ (gây False Negative). |
| 2 | `feat_parking` | Bãi đỗ xe / Hầm | **Hiển nhiên có:** Giống thang máy, bãi để xe là tiêu chuẩn bắt buộc của căn hộ. |
| 3 | `feat_security` | Bảo vệ 24/7 | **Hiển nhiên có:** Gần như không có chung cư hiện đại nào thiếu bảo vệ. |
| 4 | `feat_nice_view` | View đẹp / Toàn cảnh| **Chủ quan/Nhiễu:** Cụm từ "Văn mẫu" môi giới hay dùng để câu view, không đánh giá được giá trị thực. |
| 5 | `feat_natural_light`| Thoáng / Ánh sáng tự nhiên| **Chủ quan/Nhiễu:** Rất chung chung, không có chuẩn mực đo lường cụ thể cho toàn bộ căn hộ. |

---

## 3. Lý thuyết & Giải thích

### Tại sao loại bỏ feature "Hiển nhiên có"?
Việc một môi giới bất động sản KHÔNG ghi "Có thang máy" trong mô tả (description) của phần lớn tin đăng, không đồng nghĩa với việc tòa chung cư đó không có thang máy. 
Nếu giữ feature `feat_elevator`, mô hình sẽ gắn nhãn `0` cho hơn 95% tin đăng và "học" rằng hầu hết các chung cư tại Hà Nội là chung cư leo bộ. Tình trạng này gọi là **False Negative (Âm tính giả) diện rộng**.

### Tại sao Gộp (Merge) Feature?
1. **Pháp lý:** Từ khóa "sổ đỏ" và "pháp lý đầy đủ" có cùng mục tiêu bảo chứng pháp lý, tách làm 2 cột khiến thuật toán đối mặt với vấn đề đa cộng tuyến (Multicollinearity) vì chúng có sự tương quan nội tại.
2. **Tiện ích cao cấp:** "Hồ bơi", "Phòng Gym" và "Khu vui chơi trẻ em" thường hay xuất hiện theo cùng một nhóm tại các hệ sinh thái chung cư bậc trung trở lên. Việc gom thành `has_premium_amenities` tóm gọn đặc tính sang trọng của chung cư và giúp phân chia rõ ràng hơn ranh giới giá trị chung cư.

Về mặt Model, việc giảm từ 17 features rác xuống 9 features ý nghĩa giúp giảm bớt chiều dữ liệu, hỗ trợ thuật toán hội tụ nhanh hơn và giảm thiểu rủi ro Overfitting.

---

## 4. Xử lý Đặc trưng Phi văn bản (Non-text feature): Hướng Ban Công

Ngoài các text feature được tách từ mô tả, trong tập dữ liệu còn có sự nhầm lẫn lớn về thuộc tính: **Hướng Nhà (house_direction)** và **Hướng Ban Công (balcony_direction)**. Tại Step 3, nhóm đã đưa ra thay đổi kiến trúc quan trọng:

1.  **Loại bỏ hoàn toàn `house_direction`:** Dữ liệu thị trường cho thấy Môi giới thường nhầm lẫn điền hướng cửa chính chung cư vào trường này, trong khi đối với Căn hộ, "Hướng ban công" mới quyết định độ mát mẻ và ảnh hưởng lớn đến giá.
2.  **Tập trung One-hot Encoding `balcony_direction`:** Hệ thống giữ lại cột Hướng Ban công, điền khuyết (Unknown) và chuyển đổi thành 9 cột nhị phân (One-hot Encoding). Thao tác này giúp cung cấp cho Mô hình (LightGBM) thông tin chính xác về độ thông thoáng tự nhiên của căn hộ thay vì dùng hướng cửa bị sai lệch vật lý.
