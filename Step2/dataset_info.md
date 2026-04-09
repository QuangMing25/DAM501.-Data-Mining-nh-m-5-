# Thông tin bộ dữ liệu: Hanoi Apartments Cleaned

Tài liệu này cung cấp các thông tin tổng quan, cấu trúc và định dạng của bộ dữ liệu `vietnam_real_estate_full_2.parquet` được sử dụng trong dự án.

## 1. Nguồn dữ liệu (Data Source)
- **Đường dẫn tệp**: `data/vietnam_real_estate_full_2.parquet`
- **Nguồn gốc (Raw Data)**: Bộ dữ liệu gốc được tải xuống từ Hugging Face. Đây là bộ dữ liệu lớn chứa thông tin bất động sản toàn quốc. Được tổng hợp từ các trang tin rao bán bất động sản của Batdongsan.com và các trang rao bán bất động sản khác có các tin rao về căn hộ chung cư tại Hà Nội.

## 2. Kích thước dữ liệu (Data Size)
- **Số lượng bản ghi (Rows)**: 1000000 
- **Số lượng thuộc tính (Columns)**: 19

## 3. Các loại thuộc tính (Attributes / Columns)

Tổng cộng có 19 thuộc tính, chi tiết được tổng hợp trong bảng sau:

| Thuộc tính (Column) | Mô tả (Description) | Phân loại | Kiểu dữ liệu |
| :--- | :--- | :--- | :--- |
| `name` | Tiêu đề của tin đăng bán dự án / căn hộ. | Định tính | String |
| `description` | Nội dung mô tả chi tiết bằng văn bản tự do của người đăng. | Định tính | String |
| `property_type_name` | Loại hình bất động sản. | Định tính | String |
| `province_name` | Tên cấp Tỉnh/Thành phố (VD: Hà Nội). | Định tính | String |
| `district_name` | Tên cấp Quận/Huyện. | Định tính | String |
| `ward_name` | Tên cấp Phường/Xã. | Định tính | String |
| `street_name` | Tên Đường/Phố. | Định tính | String |
| `project_name` | Tên của dự án tòa nhà chung cư tương ứng. | Định tính | String |
| `house_direction` | Hướng cửa chính của căn hộ (VD: Đông, Nam,...). | Định tính | String |
| `balcony_direction` | Hướng ban công của căn hộ. | Định tính | String |
| `published_at` | Thời gian tin đăng được xuất bản. | Định tính | Datetime |
| `price` | Mức giá bán của căn hộ. | Định lượng | Float |
| `area` | Diện tích sử dụng (thường tính bằng mét vuông - $m^2$). | Định lượng | Float |
| `floor_count` | Số tầng của căn hộ (hoặc quy mô số tầng của cả tòa nhà). | Định lượng | Float |
| `frontage_width` | Chiều rộng mặt tiền. | Định lượng | Float |
| `house_depth` | Chiều sâu của bất động sản. | Định lượng | Float |
| `road_width` | Độ rộng của ngõ hoặc đường tiếp giáp phía trước. | Định lượng | Float |
| `bedroom_count` | Số lượng phòng ngủ. | Định lượng | Float |
| `bathroom_count` | Số lượng phòng vệ sinh / phòng tắm. | Định lượng | Float |
