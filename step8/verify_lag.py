import pandas as pd
import os

def verify_macro_consistency(df):
    print("="*60)
    print("KIỂM TRA TÍNH ĐỒNG NHẤT MACRO VÀ LAG (GIAO THOA 2025-2026)")
    print("="*60)
    
    # 1. Retrieve actual macro data for December 2025
    m12_2025 = df[(df['pub_month'] == 12) & (df['pub_year'] == 2025)].iloc[0]
    
    # 2. Retrieve actual macro data for January 2026
    m1_2026 = df[(df['pub_month'] == 1) & (df['pub_year'] == 2026)].iloc[0]
    
    macro_cols = [c for c in df.columns if c.startswith('macro_') and 'lag1' not in c]
    
    is_correct = True
    for col in macro_cols:
        val_dec = m12_2025[col]
        val_jan_lag = m1_2026[f"{col}_lag1"]
        
        diff = abs(val_dec - val_jan_lag)
        status = "✅ KHỚP" if diff < 1e-6 else "❌ LỆCH"
        
        if diff >= 1e-6: is_correct = False
        
        print(f"Chỉ số: {col}")
        print(f"  - Gốc T12/2025: {val_dec:>10.4f}")
        print(f"  - Lag1 T1/2026: {val_jan_lag:>10.4f}")
        print(f"  - Trạng thái:    {status}")
        print("-" * 30)

    if is_correct:
        print("\n=> KẾT LUẬN: Dữ liệu bắc cầu 2025-2026 hoàn hảo. Lag1 tháng 1 chính là gốc tháng 12.")
    else:
        print("\n=> CẢNH BÁO: Có sự sai lệch tại điểm giao thoa. Cần kiểm tra lại bước Merge Macro.")

# Sử dụng:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FINAL = os.path.join(BASE_DIR, "master_data_final_v3.csv")
df_master_final = pd.read_csv(DATA_FINAL)
verify_macro_consistency(df_master_final)
