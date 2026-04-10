import pandas as pd

# Giả sử df là dữ liệu bạn đã đọc từ file parquet
df = pd.read_parquet('hanoi_apartments_cleaned.parquet')

# Khi lưu ra CSV, ép định dạng số không dùng ký hiệu khoa học
df.to_csv('hanoi_apartments_cleaned3.csv', 
          index=False, 
          encoding='utf-8-sig', 
          float_format='%.0f') # %.0f giúp loại bỏ .0 và ký hiệu e+

print("Đã lưu file với định dạng số đầy đủ.")