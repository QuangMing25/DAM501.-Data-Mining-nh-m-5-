# 🚀 Project Pipeline: Data Mining (Hanoi Apartments Pricing)

Tài liệu này cấu trúc hóa quy trình (pipeline) chi tiết thực hiện đồ án môn học Data Mining. Pipeline đóng vai trò định hướng toàn bộ dự án từ lúc định nghĩa vấn đề kinh doanh, thu thập dữ liệu đến chạy thuật toán và đưa ra các đề xuất hành động thực tiễn.

---

## Phần I: Problem Definition & Business Understanding

**Mục tiêu:** Hiểu rõ vấn đề và giá trị nghiệp vụ của dữ liệu, tập trung vào việc khám phá cấu trúc, hành vi và các mẫu (patterns) tiềm ẩn của thị trường bất động sản.

- **1.1. Data Mining Objective:** 
  - Xác định rõ các bài toán Data Mining trong dự án (e.g., Phân cụm thị trường bằng K-Means, Phân tích yếu tố ảnh hưởng giá bằng LightGBM).
  - Giải thích lý do lựa chọn các bài toán này phù hợp với vấn đề kinh doanh đã đặt ra.
- **1.2. Business Value:** 
  - Chỉ ra các mẫu (patterns) phân tích được sẽ hỗ trợ doanh nghiệp ra quyết định như thế nào (VD: Định giá chính xác bất động sản, ra quyết định đầu tư, tối ưu hóa chiến lược MKT).
- **1.3. Assumptions and Limitations:** 
  - Liệt kê giả định (assumptions) về dữ liệu/bối cảnh.
  - Phân tích hạn chế (limitations) liên quan đến chất lượng dữ liệu đầu vào hoặc khả năng bao quát của kết quả.

---

## Phần II: Dataset Selection & Understanding

**Mục tiêu:** Trình bày rõ quá trình chọn lọc, đặc điểm gốc và nguồn thu thập bộ dữ liệu căn hộ thế giới thực.

- **2.1. Xác định tiêu chí đánh giá:**
  - Đặt target tối thiểu phục vụ train model (VD: 3,000–5,000 tin chung cư Hà Nội).
  - Các đặc tả tối thiểu: Phải có giá, có diện tích, có tên quận, số phòng ngủ, v.v. Không chọn nguồn có độ khó crawl quá khắc nghiệt.
- **2.2. Đánh giá và thử nghiệm chọn nguồn crawling:**
  - Các nền tảng cân nhắc: batdongsan.com.vn, nhatot.com, meeyland.com...
  - Lặp lại quá trình trích xuất thử nghiệm (bằng requests/JSON, BeautifulSoup) để nhận xét độ khó (Dễ/Trung Bình/Khó) và chọn lọc nền tảng có chất lượng thông tin thô tốt nhất.
- **2.3. Thống nhất chốt nguồn dữ liệu & features:**
  - Quyết định cuối cùng về nguồn dữ liệu thu thập, khối lượng tin sẽ cào về và danh sách toàn bộ các trường dữ liệu (features).

---

## Phần III: Data Pre-processing & Transformation

**Mục tiêu:** Quá trình tinh chỉnh dữ liệu thô phục vụ tối đa cho các Model Khai phá. Giai đoạn này bắt buộc phải giải thích lý do (why) của từng bước thực hiện thay vì chỉ show code.

- **3.1. Handling missing values (Xử lý Khuyết thiếu):**
  - Làm sạch ô trống (NaN). Lý do: K-Means gãy thuật toán đo khoảng cách nếu dính NaN. Đối với LightGBM cần chuẩn hóa để tuân thủ logic thực tế thị trường thay vì điền bù tự động một cách mù quáng.
- **3.2. Detecting and treating outliers (Khoanh vùng Ngoại lệ):**
  - Triệt tiêu các tin đăng giá áo, nhập ảo diện tích để Centroid của phân cụm (K-Means) không bị kéo lệch sai thực tế thị trường.
- **3.3. Data cleaning and noise reduction (Làm sạch văn bản):**
  - Chuẩn hóa format văn bản nhập tay (VD: Gộp "Q. Cầu Giấy" và "Quận Cầu Giấy") để bảo toàn tính minh bạch cho model đào tạo.
- **3.4. Feature transformation or discretization (Biến đổi/Rời rạc hóa):**
  - Log-transform cột giá (Price) để giảm thiểu phân phối lệch (Skewed Distribution).
  - Scale dữ liệu chuẩn hóa (StandardScaler) trước khi vào quy trình K-Means.
- **3.5. Data aggregation / transaction (Ngữ cảnh không gian):**
  - Xây dựng thêm Feature định lượng vị trí (Tốc độ tăng giá/Giá trung bình của khu vực, tọa độ không gian) nhằm cấu trúc sâu insight từng căn theo tính chất khu vực.
- **3.6. Encoding categorical variables (Mã hóa phân loại):**
  - Số hóa Text features, bảo tồn nguyên vẹn ý nghĩa thống kê của Feature để LightGBM và K-Means dễ đàng đọc hiểu phân lớp.

---

## Phần IV: Exploratory Data Analysis (EDA) & Descriptive Mining

**Mục tiêu:** Sục sạo cấu trúc sâu thẳm của dữ liệu để lôi ra các Pattern và quan hệ chằng chịt, KHÔNG đơn thuần chỉ là bước plot biểu đồ lấy lệ.

- **4.1. Distribution & Summary (Thống kê trung tâm):** Xem xét Mean/Median/Skewness của Price/m². Phát hiện phân khúc chung cư tập trung mạnh và biên độ giao động.
- **4.2. Group-based comparisons (Đối sánh chéo):** Nghiên cứu vách ngăn giá giữa vị trí địa lý khác nhau, chất lượng căn hộ khác nhau để xem thị trường phân mảnh ra sao.
- **4.3. Behavioral pattern exploration:** Nhận diện hệ quả của các hành vi thị trường (Quan hệ phi tuyến giữa các features).
- **4.4. Preliminary clustering (Mapping sơ bộ):** Tận dụng phân tích trực quan Scatter đa chiều để thử hình dung cấu trúc tự nhiên để dọn đường cho chỉ số "k" của cụm K-Means ở bước tiếp.
- **4.5. Unusual trends (Nhận diện Dị kỳ):** Phát hiện Knowledge Discovery: Xuyên thủng kiến thức có sẵn bằng các luồng thông tin đi ngược kỳ vọng thông thường tìm thấy trong Dataset.
- **4.6. TỔNG KẾT (Quan trọng):** Xâu chuỗi tất cả luận điểm thành mệnh đề chốt hạ giải thích: TẠI SAO tôi chọn triển khai K-Means, TẠI SAO tôi chọn Regress qua LightGBM cùng các Features này.

---

## Phần V: Data Mining Methods & Pattern Discovery

**Mục tiêu:** Diễn giải mạnh phần kiến trúc thuật toán, ý nghĩa ứng dụng thay vì đi giải thích các công thức độ khó thuật toán. Hai thuật toán được **nối tiếp thành chuỗi tri thức (Knowledge-Driven Pipeline)**, không chạy độc lập.

- **5.1. K-Means Clustering (Khai phá không giám sát):**
  - Nhóm các sản phẩm chung cư đang bán vào thành các Cluster bản chất để trả lời được câu hỏi tóm tắt: "Thị trường này chia làm mấy phân khúc tự nhiên, dung mạo từng khối ra sao?".
  - **Output:** Nhãn Cluster (0/1/2) cho 72.604 căn hộ → **chuyển giao sang bước 5.2**.
- **5.2. LightGBM Regression (Khai phá có giám sát + Tích hợp K-Means):**
  - Nhận nhãn Cluster từ bước 5.1 làm **categorical feature bổ sung**, tạo pipeline liên kết Unsupervised → Supervised.
  - Chạy **2 mô hình song song** (Baseline vs + K-Means) để chứng minh giá trị của phân khúc: R² tăng từ 0.81 → 0.85, MAE giảm 103 triệu VND/căn. Feature `Cluster` xếp hạng #1 trong Feature Importance.
  - Trả lời được câu hỏi: "Biến số nào chi phối quyền lực đến giá tiền? Quyền lực đó tác động xoay chiều như thế nào giữa các cụm phân khúc?".

---

## Phần VI: Evaluation & Interpretation of Results

**Mục tiêu:** Định hướng đánh giá 2 chiều: (1) Kết quả mặt kỹ thuật - (2) Giá trị ứng dụng Insight kết hợp năng lực tư duy phản biện.

- **6.1. Đánh giá chất lượng K-Means:** Xác nhận Cụm chạy ra là "thực thể thị trường thật" chứ không phải những cluster toán học vô tri.
- **6.2. Đánh giá LightGBM Regression:** Ưu tiên đo lường chất lượng thông điệp (Insight) qua biểu đồ SHAP và mức độ tin cậy được thể hiện qua các chỉ số hồi quy (RMSE / MAE).
- **6.3. Strength & Weakness Critique (Phản biện):** Thái độ đánh giá công bằng điểm mù hạn chế và sự ưu việt của 2 Model đã chọn nhằm đáp ứng kỳ vọng thảo luận khách quan từ Hội đồng đánh giá.
- **6.4. Toàn cục vs Local Cluster (Mở rộng):** So sánh tính hiệu quả giữa một đại mô hình (train tất cả) và việc chẻ nhỏ Model train chuyên biệt cho từng Cluster tìm được ở phần 5.

---

## Phần VII: Business Insights & Recommendations

**Mục tiêu:** Mọi thông số toán học phải được dịch ngược về ngôn ngữ quản trị, kinh doanh để tạo lợi thế chiến lược.

- **7.1. Translate to Actionable Insights:** Cụ thể hóa những Knowledge Discovery thu được thành lợi ích cho bộ phận quản trị nhà nước, nhà đầu tư, chủ thầu hay sàn giao dịch. Khung pattern nào mang lại giá trị cao nhất?
- **7.2. Concrete Recommendations:** Phác thảo cơ chế vận hành giá, chiến lược tiếp cận Customer Segments hoặc cảnh báo rủi ro ảo giá bám chặt vào Data Mining Insight.

---

## Phần VIII: (Bonus) Advanced Techniques

**Mục tiêu:** Thể hiện độ bao phủ kỹ năng thông qua kỹ thuật tầng sâu nhằm khai thác mạnh hơn chất lượng dự án.

- Tuỳ chọn 1: **PCA (Phân tích thành phần chính):** Giảm chiều tính toán và làm sắc nét sơ đồ Cluster Visualizations trực quan.
- Tuỳ chọn 2: **Anomaly Detection (Bắt tin dị thường):** Thể hiện năng lực bóc tách những rủi ro ngầm, tin "treo đầu dê bán thịt chó".
- Tuỳ chọn 3: **Advanced Feature Engineering:** Xây dựng thêm những tính năng tinh xảo (như index hạ tầng vỹ mô, bán kính tiếp cận) để khiến Insight phân tích rực rỡ hơn bản chất ban đầu, chứ không phải chỉ là để đua Top điểm R2 Model.
