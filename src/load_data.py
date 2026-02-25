import os
import shutil
import kagglehub
import pandas as pd

DATA_DIR = "data"

def fetch_data(dataset_handle="adarshsng/lending-club-loan-data-csv", filename="loan.csv"):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
    
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        print("file already exists, loading data...")
        data = pd.read_csv(file_path)
    else:
        print("file not found, downloading dataset...")
        path = kagglehub.dataset_download(dataset_handle)
        csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
        shutil.copy(os.path.join(path, csv_file), file_path)
        data = pd.read_csv(file_path)
    
    return data