cd "/Users/quangminh/QuangMinh/MSE35HN/Ky_[II]/[2.4] DAM501. Khai pha du lieu/[4] Final Project"

# Cài thư viện
.venv/bin/pip install pandas pyarrow

# Chuyển đổi
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_parquet('/Users/quangminh/Downloads/hanoi_apartments_cleaned.parquet')
print(df.shape)
print(df.columns.tolist())
df.to_csv('/Users/quangminh/Downloads/hanoi_apartments_cleaned.csv', index=False, encoding='utf-8-sig')
print('Done!')
"
