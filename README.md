Dự án Phân loại Email Spam
Tổng quan
Dự án này triển khai và so sánh ba mô hình học máy—Random Forest, Logistic Regression và Long Short-Term Memory (LSTM)—để phân loại email thành spam hoặc không phải spam. Bộ dữ liệu sử dụng là combined_data.csv, chứa các email được gắn nhãn. Dự án bao gồm tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá và dự đoán trên các mẫu email mới.

Tính năng
Tiền xử lý dữ liệu: Làm sạch văn bản (chuyển thành chữ thường, loại bỏ dấu câu, loại bỏ từ dừng) bằng NLTK.
Trích xuất đặc trưng: Sử dụng TF-IDF cho Random Forest và Logistic Regression; mã hóa và đệm cho LSTM.
Mô hình:
Random Forest Classifier
Logistic Regression
Mạng nơ-ron LSTM
Đánh giá: Độ chính xác, Precision, Recall, F1-Score và Báo cáo phân loại cho từng mô hình.
Dự đoán: Phân loại email mới thành spam hoặc không spam bằng cả ba mô hình.
Lưu mô hình: Lưu các mô hình đã huấn luyện và bộ tiền xử lý để tái sử dụng.
Bộ dữ liệu
File: combined_data.csv
Cột:
text: Nội dung email
label: 0 (không phải spam) hoặc 1 (spam)
Kích thước: 83,448 email (43,910 spam, 39,538 không phải spam)
Yêu cầu
Để chạy dự án, cài đặt các gói Python sau:

bash

Sao chép
pip install pandas numpy scikit-learn tensorflow nltk joblib
Tải thêm stopwords của NLTK:

python

Sao chép
import nltk
nltk.download('stopwords')
Cài đặt
Tải repository hoặc các file dự án.
Đảm bảo file dữ liệu (combined_data.csv) nằm trong thư mục dự án.
Cài đặt các gói phụ thuộc được liệt kê ở phần Yêu cầu.
Hướng dẫn sử dụng
Mở file Jupyter Notebook A39948_nlp.ipynb.
Chạy lần lượt tất cả các ô để:
Tải và tiền xử lý dữ liệu
Huấn luyện và đánh giá mô hình Random Forest, Logistic Regression và LSTM
So sánh hiệu suất các mô hình
Lưu mô hình và bộ tiền xử lý
Dự đoán spam/không spam cho các email mẫu
Để phân loại email mới, chỉnh sửa danh sách emails trong ô cuối và chạy lại.
Ví dụ:

python

Sao chép
emails = [
    "Chúc mừng! Bạn đã trúng thẻ quà tặng Walmart trị giá $1000. Truy cập http://bit.ly/123456 để nhận ngay.",
    "Chào John, hy vọng bạn khỏe. Hãy sắp xếp một cuộc họp vào tuần tới để thảo luận về dự án."
]
Kết quả
Hiệu suất mô hình trên tập kiểm tra:

Mô hình	Độ chính xác	Precision	Recall	F1-Score
Random Forest	0.9854	0.9840	0.9882	0.9861
Logistic Regression	0.9835	0.9803	0.9883	0.9843
LSTM	0.9797	0.9842	0.9769	0.9806
File đã lưu
random_forest_model.pkl: Mô hình Random Forest đã huấn luyện
logistic_regression_model.pkl: Mô hình Logistic Regression đã huấn luyện
lstm_model.pkl: Mô hình LSTM đã huấn luyện
tfidf_vectorizer.pkl: Bộ vector hóa TF-IDF
tokenizer.pkl: Bộ mã hóa cho LSTM
Cải tiến trong tương lai
Thử nghiệm tinh chỉnh siêu tham số để cải thiện hiệu suất.
Bổ sung thêm đặc trưng (ví dụ: siêu dữ liệu email).
Kiểm tra với các bộ dữ liệu đa dạng hơn để tăng tính tổng quát.
