import pandas as pd
import PyPDF2
import os

# Read CSV stats
csv_path = r'e:\Caterpiller\ML_RL.csv'
df = pd.read_csv(csv_path)
print("CSV Columns:", df.columns.tolist())
print("CSV Mean Values:")
print(df.mean().to_string())

# Read PDF text
pdf_path = r'e:\Caterpiller\Problem Statement 2.pdf'
with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

# Search for gains
for line in full_text.split('\n'):
    if any(k in line for k in ['Kp', 'Ki', 'fixed', 'Kp_fixed', 'Ki_fixed', 'gain', 'setpoint', '900']):
        print(line)

with open(r'e:\Caterpiller\pdf_content.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)
