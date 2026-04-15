import pandas as pd
df = pd.read_csv('../step3_minh/data/hanoi_apartments_processed.csv')

dir_cols = [c for c in df.columns if c.startswith('dir_')]
df['house_direction'] = df[dir_cols].idxmax(axis=1).str.replace('dir_', '')

dir_stats = df.groupby('house_direction').agg(
    count=('price', 'count'),
    median_price=('price', 'median'),
    mean_price=('price', 'mean'),
    median_ppm2=('price_per_m2', 'median')
).sort_values('median_ppm2', ascending=False)
print(dir_stats)
