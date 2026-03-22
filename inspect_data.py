import scipy.io
import os
import numpy as np

for mat_file in [r'e:\Caterpiller\Simulink_Data.mat', r'e:\Caterpiller\Simulink_Data (1).mat']:
    if os.path.exists(mat_file):
        try:
            data = scipy.io.loadmat(mat_file)
            print(f"\n--- {mat_file} ---")
            for k, v in data.items():
                if k.startswith('__'): continue
                print(f"Key: {k}, Type: {type(v)}")
                if isinstance(v, (np.ndarray, list)):
                    if hasattr(v, 'shape'):
                        print(f"  Shape: {v.shape}")
                    if v.size < 50:
                        print(f"  Value: {v}")
                else:
                    print(f"  Value: {v}")
        except Exception as e:
            print(f"Error reading {mat_file}: {e}")

# Also check CSV for correlation
import pandas as pd
df = pd.read_csv(r'e:\Caterpiller\ML_RL.csv')
print("\nCSV Head:")
print(df.head())
print("\nCSV Stats:")
print(df.describe())
