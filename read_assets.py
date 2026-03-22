import scipy.io
import os
import PyPDF2

# Read .mat files
for mat_file in [r'e:\Caterpiller\Simulink_Data.mat', r'e:\Caterpiller\Simulink_Data (1).mat']:
    if os.path.exists(mat_file):
        try:
            data = scipy.io.loadmat(mat_file)
            print(f"\n--- {mat_file} Keys ---")
            print([k for k in data.keys() if not k.startswith('__')])
            # Print a bit of data if possible
            for k in [k for k in data.keys() if not k.startswith('__')]:
                print(f"{k} shape/type: {type(data[k])}")
                if isinstance(data[k], (int, float, np.ndarray)) and data[k].size < 10:
                    print(f"{k} value: {data[k]}")
        except Exception as e:
            print(f"Error reading {mat_file}: {e}")

# Read PDF more carefully
pdf_path = r'e:\Caterpiller\Problem Statement 2.pdf'
if os.path.exists(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                print(f"\n--- PDF Page {i+1} ---")
                print(page.extract_text())
    except Exception as e:
        print(f"Error reading PDF: {e}")
