# Business Requirements Document (BRD)
## Dự án: Ứng dụng Data Mining trong Dự đoán Giá Chung cư tại TP Hà Nội (Tại thời điểm hiện tại)

---

### 1. Tóm tắt dự án (Executive Summary)
Dự án nhằm xây dựng một hệ thống phân tích và ước lượng giá căn hộ chung cư tại thị trường Hà Nội **tại thời điểm hiện tại** bằng các kỹ thuật Data Mining (Khai thác dữ liệu). Thông qua việc thu thập và phân tích các yếu tố định lượng (vị trí, diện tích, tiện ích) ngay tại mốc thời gian này, dự án cung cấp một công cụ hỗ trợ định giá "ngay lập tức" một cách minh bạch, khoa học cho người mua, nhà đầu tư và các đơn vị kinh doanh bất động sản mà không dựa vào phân tích chuỗi thời gian hay dự báo biến động tương lai.

---

### 2. Bối cảnh kinh doanh (Business Context)
*   **Thị trường hiện tại:** Thị trường chung cư Hà Nội đang ở một mặt bằng giá mới sau các đợt tăng nóng thời gian qua. Nhu cầu nhà ở thực tại các đô thị lớn đang rất cao ở ngay thời điểm này.
*   **Sự phân hoá:** Thị trường ghi nhận sự tách biệt rõ rệt giữa các phân khúc (bình dân đến hạng sang), nhưng ở thời điểm hiện tại đang thiếu đi một thước đo chuẩn xác để đánh giá giá trị thực tế của một điểm bất động sản dựa trên các thuộc tính của nó giữa ma trận thông tin nhiễu loạn.

---

### 3. Vấn đề và Thách thức (Problem Statement)
*   **Định giá thiếu minh bạch "ngay lúc này":** Việc định giá một căn hộ khi rao bán hay mua vào phụ thuộc rất nhiều vào "cảm tính tức thời" hoặc so sánh thủ công một vài căn đang mở bán tương tự, dẫn đến ngáo giá hoặc bán hớ.
*   **Không nắm bắt được bức tranh hiện tại:** Khó có cái nhìn tổng thể về cấu trúc giá và các nhóm sản phẩm tương đồng đang cùng tồn tại trên thị trường ở cùng một thời điểm.
*   **Khó khăn thực chiến cho người mua/bán:** Không có một công cụ "sống" để nhập thông số căn hộ và biết liệu với thị trường ngày hôm nay, mức giá X là đắt hay rẻ so với mặt bằng chung.

---

### 4. Mục tiêu dự án (Project Objectives)
*   **Khám phá cấu trúc thị trường hiện tại:** Sử dụng các thuật toán phân cụm (Clustering) để tự động hóa việc phân loại các phân khúc chung cư dựa trên dữ liệu thu thập được ở mốc thời gian hiện tại.
*   **Xác định động lực giá hiện hành:** Phân tích xem ở thời điểm hiện tại, yếu tố nào (vị trí, diện tích, tiện ích...) đang đóng vai trò quyết định mạnh mẽ nhất đến giá trị căn hộ.
*   **Ước lượng giá tại chỗ (Point-in-time Prediction):** Xây dựng mô hình học máy (Machine Learning) có khả năng định lượng ngân sách/giá trị hợp lý nhất của một căn hộ chỉ dựa vào các tham số đầu vào ở thời điểm truy vấn.
*   **Hỗ trợ ra quyết định:** Cung cấp công cụ đánh giá độc lập cho người mua nhà và môi giới trước khi tiến hành giao dịch thực tế trên thị trường lúc này.

---

### 5. Phạm vi dự án (Project Scope)
#### 5.1. Bao gồm (In-Scope)
*   Dữ liệu căn hộ chung cư tại các quận/huyện thuộc TP Hà Nội được **thu thập mới nhất (tại thời điểm thực hiện dự án)**.
*   Dữ liệu thuộc tính tĩnh và động ở hiện tại: Diện tích, số phòng ngủ, số vệ sinh, hướng nhà, tầng, tên dự án, đơn vị chủ đầu tư, tiện ích, và giá rao bán hiện tại.
*   Dự đoán mức giá của một căn hộ ở **thời điểm hiện tại** (Point-in-time Cross-sectional Analysis).

#### 5.2. Không bao gồm (Out-of-Scope)
*   **Phân tích chuỗi thời gian (Time-series analysis) hoặc dự báo xu hướng giá trong tương lai (ví dụ: dự đoán giá vào năm sau).**
*   Dữ liệu lịch sử giá qua các năm cũ.
*   Các loại hình bất động sản khác (nhà đất thổ cư, biệt thự, đất nền).
*   Thực hiện các giao dịch đặt cọc hoặc thanh toán trực tuyến.
*   Tư vấn pháp lý.

---

### 6. Yêu cầu chức năng (Functional Requirements)
*   **FR1 - Thu thập & Tiền xử lý dữ liệu:** Crawler để thu thập dữ liệu cross-sectional từ các sàn BĐS uy tín ở thời điểm hiện hành, làm sạch dữ liệu nhiễu, loại bỏ các listing "ảo".
*   **FR2 - Phân tích Phân khúc Hiện tại:** Tự động nhóm các căn hộ trên thị trường hiện nay vào cùng một phân khúc tương xứng.
*   **FR3 - Mô hình Định giá Tức thời:** Người dùng nhập các thông số (vị trí, diện tích...) và hệ thống trả về mức giá hợp lý ước tính theo mặt bằng thị trường hiện tại cùng biên độ dao động.
*   **FR4 - Báo cáo Trực quan:** Hiển thị biểu đồ về các yếu tố đang tác động mạnh nhất đến giá căn hộ lúc này (Feature Importance).

---

### 7. Yêu cầu phi chức năng (Non-functional Requirements)
*   **Độ chính xác (Accuracy):** Mô hình dự báo cần đạt độ lỗi thấp (ví dụ: MAPE < 15% so với giá rao bán thực tế hiện tại).
*   **Khả năng giải thích (Explainability):** Hệ thống phải chỉ tóm lược được vì sao lại ra mức giá này dựa vào cấu trúc dữ liệu hiện tại (vd: do căn hộ ở quận Cầu Giấy, 2PN, có bể bơi).
*   **Hiệu năng:** Kết quả ước lượng trả về nhanh chóng nhằm nâng cao trải nghiệm người dùng.

---

### 8. Các bên liên quan (Stakeholders)
*   **Người mua nhà ở thực:** Cần công cụ thẩm định giá nhanh trước khi xuống tiền mua nhà ngay lúc này.
*   **Nhà đầu tư:** Quan tâm đến việc tìm ra các căn hộ bị định giá thấp (undervalued) so với thị trường hiện hành để lướt sóng.
*   **Môi giới BĐS:** Cần căn cứ dữ liệu để tư vấn cho khách mua/bán chốt giá hợp lý đẩy nhanh giao dịch.

---

### 9. Giả định và Rủi ro (Assumptions & Risks)
*   **Giả định:** Dữ liệu crawler được từ các cổng thông tin hiện tại (giá chào bán) có sự tương quan chặt chẽ với giá giao dịch thực tế tại thời điểm này.
*   **Rủi ro:** Thông tin listing ảo, giá bẫy do cò mồi đăng tải có thể làm méo mó mô hình nếu khâu tiền xử lý (FR1) không làm tốt.

---

### 10. Tiêu chí đánh giá thành công (Success Criteria)
*   Thu thập đủ lượng dữ liệu mẫu đại diện cho thị trường Hà Nội tại thời điểm hiện tại.
*   Mô hình mô phỏng/định giá có sai số nằm trong giới hạn cho phép khi test với các căn hộ đang rao bán thực.
*   Chỉ rõ được top các thuộc tính đang ảnh hưởng lớn nhất đến giá chung cư ở thời điểm hiện nay.
