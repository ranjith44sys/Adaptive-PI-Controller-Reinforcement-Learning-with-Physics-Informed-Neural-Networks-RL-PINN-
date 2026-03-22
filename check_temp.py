import pandas as pd
import matplotlib.pyplot as pd
import os

csv_path = r'e:\Caterpiller\ML_RL.csv'
df = pd.read_csv(csv_path)

# Check first 100 rows to see if there is a pattern
print("First 20 rows:")
print(df.head(20).to_string())

# Check if error changes smoothly
diff_error = df['error'].diff().abs()
print("\nMean absolute error change between rows:", diff_error.mean())
print("Max absolute error change between rows:", diff_error.max())
