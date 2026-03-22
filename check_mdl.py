import zipfile
import os

mdl_path = r'e:\Caterpiller\ProblemState.mdl'
if os.path.exists(mdl_path):
    if zipfile.is_zipfile(mdl_path):
        print(f"{mdl_path} is a ZIP file.")
        with zipfile.ZipFile(mdl_path, 'r') as z:
            print("Files in ZIP:")
            print(z.namelist())
    else:
        print(f"{mdl_path} is NOT a ZIP file.")
        with open(mdl_path, 'r', errors='ignore') as f:
            print("First 1000 characters:")
            print(f.read(1000))
