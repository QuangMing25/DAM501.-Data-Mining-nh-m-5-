# Section 6 — Evaluation & Interpretation of Results (15%)

Tài liệu này tổng hợp các đánh giá đa chiều về chất lượng của hai mô hình Khai phá dữ liệu (K-Means và LightGBM) đã thực hiện ở Bước 5, đồng thời đưa ra các nhận định phản biện về giá trị thực tiễn của dự án.

---

## 6.1. Đánh giá chất lượng K-Means Clustering (Khai phá không giám sát)

Mục tiêu của K-Means là tìm ra các phân khúc thị trường "tự nhiên". Chất lượng được đánh giá qua hai lăng kính: Toán học và Nghiệp vụ.

### A. Đánh giá qua Chỉ số Kỹ thuật
- **Inertia (WCSS):** Đạt giá trị **246.960** tại K=3. Biểu đồ Elbow cho thấy đây là "điểm gãy" tối ưu nhất, nơi mà việc tăng thêm cụm không còn mang lại sự sụt giảm lỗi đáng kể (Diminishing returns).
- **Silhouette Score:** Đạt **0.2604**. Mặc dù con số này không quá cao (do dữ liệu BĐS có độ nhiễu và chồng lấn lớn giữa các ranh giới giá), nhưng nó đủ để khẳng định các cụm có sự phân tách rõ rệt trong không gian đa chiều.
- **Tính ổn định (Stability):** Kết quả phân cụm lặp lại nhất quán qua các lần chạy, cho thấy cấu trúc dữ liệu thực sự tồn tại 3 nhóm chính.

![Elbow & Silhouette](../step5_binh/plots_section_5/kmeans_01_elbow_silhouette.png)

### B. Đánh giá qua Giá trị Nghiệp vụ (Business Relevance)
Kết quả phân cụm không chỉ là những con số vô tri mà đã phác họa được 3 "thực thể" thực tế trên thị trường Hà Nội:
| **Cụm 0 (37.6%):** Phân khúc Cao Cấp (Premium) - Giá Median: 90.9 Tr/m². Tập trung các dự án giá trị cao, tiện ích đầy đủ.
| **Cụm 1 (46.4%):** Phân khúc Phổ Thông Ngoại Ô - Giá Median: 76.3 Tr/m². Trọng tâm nguồn cung mới dịch chuyển ra vành đai và ngoại ô.
| **Cụm 2 (16.0%):** Phân khúc Tầm Trung Lõi - Giá Median: 76.5 Tr/m². Các dự án nội đô ổn định, đáp ứng nhu cầu ở thực.

![K-Means PCA Visualization](../step5_binh/plots_section_5/kmeans_02_pca_clusters.png)

**=> Kết luận:** K-Means đã hoàn thành xuất sắc vai trò "người mở đường", tạo ra tri thức phân khúc (Segment Knowledge) mà không cần sự can thiệp của con người, đồng thời phản ánh đúng xu hướng dịch chuyển của thị trường giai đoạn 2025-2026.

---

## 6.2. Đánh giá LightGBM Regression (Khai phá có giám sát)

Mô hình LightGBM được đánh giá dựa trên khả năng dự báo và độ quan trọng của các biến số.

### A. Hiệu quả dự báo (Performance Metrics)
Việc tích hợp nhãn Cluster từ K-Means vào LightGBM đã mang lại sự bứt phá đáng kể so với mô hình cơ sở (Baseline):

| Chỉ số | Kết quả (+ K-Means) | Cải thiện so với Baseline |
|---|---|---|
| **R² Score** | **0.8525** | **+4.26%** (Giải thích được 85.25% biến động giá) |
| **MAE (Sai số tuyệt đối)** | **903.7 triệu VND** | **Giảm 110.5 triệu VND** |
| **MAPE (Phần trăm sai số)**| **13.19%** | **Giảm 2.64%** |

![Comparison Metrics](../step5_binh/plots_section_5/lightgbm_00_comparison_metrics.png)
![Predict Accuracy](../step5_binh/plots_section_5/lightgbm_02_predict_accuracy.png)

**Nhận xét:** Việc bổ sung dữ liệu 2026 đã đẩy độ chính xác của mô hình lên mức rất cao (R² ~ 0.85). Với một thị trường biến động mạnh, mức sai số ~13% là một kết quả xuất sắc, đặc biệt khi mô hình đã học được từ hơn 133,000 giao dịch.

### B. Đánh giá qua Feature Importance (Quyền lực của Biến)
- **Biến Cluster đứng vị trí #1:** Đây là phát hiện quan trọng nhất. Nhãn phân khúc từ K-Means chứa đựng thông tin tổng hợp (tương tác giữa diện tích, vị trí, giá) mạnh hơn bất kỳ biến đơn lẻ nào.
- **Sự đóng góp của Text Features:** Các biến như `quality_score`, `has_legal_paper`, `has_premium_amenities` đều nằm trong Top 10. Điều này chứng minh quy trình Tiền xử lý (Step 3) và trích xuất đặc trưng văn bản là hoàn toàn đúng đắn và mang lại giá trị thực.

![Feature Importance](../step5_binh/plots_section_5/lightgbm_01_feature_importance.png)

---

## 6.3. Strength & Weakness Critique (Phản biện mô hình)

Để có cái nhìn khách quan, nhóm thực hiện phản biện về các điểm mạnh và hạn chế của phương pháp tiếp cận:

### Điểm mạnh (Strengths)
1.  **Pipeline liên kết (Knowledge-Driven):** Sự kết hợp giữa Unsupervised (K-Means) và Supervised (LightGBM) tạo ra một chu trình tri thức khép kín, nơi bước sau kế thừa và khuếch đại thành quả của bước trước.
2.  **Xử lý phi tuyến tính:** LightGBM đã vượt qua được giới hạn của các mô hình tuyến tính, nắm bắt tốt các tương tác chéo giữa Diện tích, Quận huyện và Tiện ích.
3.  **Tính giải thích cao (Explainability):** Thông qua Feature Importance và phân tích sai số theo cụm, mô hình không còn là "hộp đen" mà cung cấp các Insight kinh doanh rõ ràng.

### Hạn chế (Weaknesses)
1.  **Độ nhiễu của phân khúc Cao cấp:** Sai số ở Cụm Cao Cấp (Cụm 0) cao nhất (MAE ~1.37 tỷ). Lý do: Giá nhà cao cấp phụ thuộc vào nhiều yếu tố "mềm" chưa thu thập được như: Thương hiệu chủ đầu tư, uy tín quản lý vận hành, hoặc nội thất cá nhân hóa đặc biệt.
2.  **Sự phụ thuộc vào chất lượng tin đăng:** Mô hình dựa trên mô tả của môi giới. Nếu môi giới nhập liệu sai hoặc cố tình "thổi giá", mô hình sẽ bị ảnh hưởng (Garbage in - Garbage out).
3.  **Điểm mù chu kỳ kinh tế:** Dù đã có dữ liệu 2025-2026, thị trường BĐS vẫn có những pha "sốt đất" mang tính tâm lý bầy đàn mà mô hình hồi quy thuần túy khó lòng bắt nhịp kịp thời nếu không có dữ liệu vĩ mô (lãi suất, tín dụng).

---

## 6.4. So sánh Mô hình Toàn cục vs Mô hình Chuyên biệt (Mở rộng)

Nhóm đã tiến hành phân tích sâu về sai số dự báo trên từng cụm K-Means bằng cách tách riêng các mô hình (Specialized Models):
- **Cụm 2 (Tầm Trung Lõi):** Dự báo chính xác nhất (MAE thấp nhất ~479 triệu). Thị trường này có tính đồng nhất cao, giá cả tuân theo quy luật diện tích/vị trí rất chặt chẽ.
- **Cụm 0 (Cao Cấp):** Khó dự báo nhất, nhưng mô hình chuyên biệt đã giúp **cải thiện sai số (giảm MAE)** so với mô hình toàn cục.

![Error by Cluster](../step5_binh/plots_section_5/lightgbm_03_error_by_cluster.png)

**Insight đề xuất:** Với tập dữ liệu đủ lớn (133k+), chiến lược "chia để trị" (sử dụng các mô hình chuyên biệt cho từng phân khúc) bắt đầu phát huy hiệu quả, đặc biệt cho phân khúc Cao Cấp và Phổ Thông Ngoại Ô. Trong tương lai, việc tách riêng đường ống dự đoán cho phân khúc Premium sẽ là hướng đi bắt buộc để cải thiện độ chính xác cho nhóm khách hàng cao cấp.

---

**Tổng kết:** Phần VI đã chứng minh được tính đúng đắn của toàn bộ quy trình Data Mining. Kết quả đạt được (R² = 0.8525) không chỉ mạnh về mặt kỹ thuật mà còn mang lại những giá trị hiểu biết sâu sắc về cấu trúc thị trường chung cư Hà Nội trong giai đoạn chuyển giao 2025-2026.
