# cd "/Users/quangminh/QuangMinh/MSE35HN/Ky_[II]/[2.4] DAM501. Khai pha du lieu/[4] Final Project"

# # Cài thư viện
# .venv/bin/pip install pandas pyarrow

# # Chuyển đổi
# .venv/bin/python3 -c "

"""
Cross-platform convert handler function
Put all raw parquet files to the data folder and call this function
regardless of the platforms (MAC, Windows, Linux)
"""
from pathlib import Path
import pandas as pd
import os

def convert_data(file_name):

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    input_path = DATA_DIR / f"{file_name}.parquet"
    output_path = input_path.with_suffix('.csv')

    if not input_path.exists():
        print(f"Error: Could not find {input_path}")
        return
    
    df = pd.read_parquet(input_path)
    print(f"Loaded: {input_path}")
    print(f"Shape: {df.shape}")

    if not os.path.exists(output_path):
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Saved to: {output_path}")
    else:
        print(f"dataset : {output_path} exists")
    print('Done!')

convert_data("aptm_hanoi_2026")

"""
Optional: For handling weird cases
Usage
Windows: `$env:RUN_OPTIONAL=1; python script.py` (for PowerShell)
Mac/Linux: `RUN_OPTIONAL=1 python script.py`
Otherwise: Run `python script.py` as usual
"""

if os.getenv("RUN_OPTIONAL") == "1":
    print("Running optional part of the code...")

    import unicodedata
    import re

    def clean_messy_sequences(df, sequences):
        compiled_pattern = [
            re.compile(r'\W+'.join(seq), re.IGNORECASE)
            for seq in sequences
        ]

        def is_messy(val):
            if not isinstance(val, str):
                return False
            norm_text = unicodedata.normalize('NFKC', val).lower()
            return any(p.search(norm_text) for p in compiled_pattern)
        
        mask = df.apply(lambda row: row.map(is_messy).any(), axis=1)
        df_cleaned = df[~mask].copy()
        print(f"Removed {mask.sum()} rows.")
        return df_cleaned
    
    sequences = [
        ['đông bắc', 'đông nam'],
        ['đông', 'đông nam', 'nam', 'bắc'],
        ['tây', 'đông bắc', 'tây nam', 'tây bắc'],
        ['tây bắc', 'đông nam']
    ]
    df = pd.read_csv("data/aptm_hanoi_2026.csv")
    df_orig_cleaned = clean_messy_sequences(df, sequences)
    df_orig_cleaned.to_csv("data/aptm_hanoi_2026_cleaned.csv", index=False)
    