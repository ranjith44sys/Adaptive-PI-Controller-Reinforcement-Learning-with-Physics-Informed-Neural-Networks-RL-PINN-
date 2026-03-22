import pandas as pd
import os

csv_path = r'e:\Caterpiller\ML_RL.csv'
df = pd.read_csv(csv_path)

# Check first 20 rows to see if there is a pattern
print("First 20 rows:")
print(df.head(20).to_string())

# Check if error changes smoothly
diff_error = df['error'].diff().abs()
print("\nMean absolute error change between rows:", diff_error.mean())
print("Max absolute error change between rows:", diff_error.max())
print("Min absolute error change between rows:", diff_error.min())

# Check for any reset points (large jumps in error)
large_jumps = df[diff_error > 100]
print(f"\nNumber of large jumps (>100): {len(large_jumps)}")
