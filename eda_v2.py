import pandas as pd
import numpy as np
import os

csv_path = r'e:\Caterpiller\ML_RL.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("CSV Shape:", df.shape)
    print("CSV Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nStatistics:")
    print(df.describe())
else:
    print(f"CSV not found at {csv_path}")

try:
    import PyPDF2
    pdf_path = r'e:\Caterpiller\Problem Statement 2.pdf'
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        print(f"\nPDF Pages: {len(reader.pages)}")
        text = reader.pages[0].extract_text()
        print("\nPDF First Page Text Snippet:")
        print(text[:2000])
except Exception as e:
    print(f"\nError reading PDF or PyPDF2 not installed: {e}")
