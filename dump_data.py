import scipy.io
import os
import numpy as np
import pandas as pd

with open(r'e:\Caterpiller\data_dump.txt', 'w', encoding='utf-8') as f:
    for mat_file in [r'e:\Caterpiller\Simulink_Data.mat', r'e:\Caterpiller\Simulink_Data (1).mat']:
        if os.path.exists(mat_file):
            try:
                data = scipy.io.loadmat(mat_file)
                f.write(f"\n--- {mat_file} ---\n")
                for k, v in data.items():
                    if k.startswith('__'): continue
                    f.write(f"Key: {k}, Type: {type(v)}\n")
                    if isinstance(v, (np.ndarray, list)):
                        if hasattr(v, 'shape'):
                            f.write(f"  Shape: {v.shape}\n")
                        if v.size < 100:
                            f.write(f"  Value: {v}\n")
                        elif k == 'ans' or 'data' in k.lower():
                             f.write(f"  First 10 values: {v.flatten()[:10]}\n")
                    else:
                        f.write(f"  Value: {v}\n")
            except Exception as e:
                f.write(f"Error reading {mat_file}: {e}\n")

    csv_path = r'e:\Caterpiller\ML_RL.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        f.write("\n--- ML_RL.csv ---\n")
        f.write(f"Columns: {df.columns.tolist()}\n")
        f.write("Describe:\n")
        f.write(df.describe().to_string() + "\n")
        f.write("Correlation:\n")
        f.write(df.corr().to_string() + "\n")
