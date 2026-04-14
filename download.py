import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --- Configuration ---
# Data will be saved in the ./DATA directory to match the project structure
BASE_DATA_DIR = "./DATA"

# NASA STEREO Data URLs (STA - Ahead)
URL_MAP = {
    "beacon_raw": "https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/secchi/img/cor2/",
    "science_raw": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/a/img/cor2/",
    "beacon_seq": "https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/secchi/seq/cor2/",
    "science_seq": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/a/seq/cor2/"
}

# Mapping to local paths (matching common_utils.py logic)
def get_local_path(data_type, date_str):
    if data_type == "beacon_raw":
        return os.path.join(BASE_DATA_DIR, "STEREO_DATA/STA/beacon", date_str)
    elif data_type == "science_raw":
        return os.path.join(BASE_DATA_DIR, "STEREO_DATA/STA/science", date_str)
    elif data_type == "beacon_seq":
        return os.path.join(BASE_DATA_DIR, "STEREO_DATA_seq/STA/Beacon_seq", date_str)
    elif data_type == "science_seq":
        return os.path.join(BASE_DATA_DIR, "STEREO_DATA_seq/STA/Science_seq", date_str)
    return None

def download_files(data_type, date_str):
    base_url = f"{URL_MAP[data_type]}{date_str}/"
    save_dir = get_local_path(data_type, date_str)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\nChecking {data_type.upper()} at: {base_url}")
    
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code != 200:
            print(f"Skipping: Directory not found on server.")
            return
            
        soup = BeautifulSoup(response.text, 'html.parser')
        # Filter for .fts or .fts.gz files
        files = [a['href'] for a in soup.find_all('a', href=True) 
                 if a['href'].endswith(('.fts', '.fts.gz'))]
        
        print(f"Found {len(files)} files. Starting download...")
        
        for i, file_name in enumerate(files):
            file_url = base_url + file_name
            save_path = os.path.join(save_dir, file_name)
            
            if os.path.exists(save_path):
                continue
                
            print(f"[{i+1}/{len(files)}] Downloading: {file_name}")
            file_res = requests.get(file_url, stream=True)
            with open(save_path, 'wb') as f:
                for chunk in file_res.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
    except Exception as e:
        print(f"Error downloading {data_type}: {e}")

if __name__ == "__main__":
    # Get target date from user
    target_date = input("Enter the date to download (YYYYMMDD, e.g., 20240510): ").strip()
    
    if len(target_date) != 8:
        print("Invalid date format. Please use YYYYMMDD.")
    else:
        print(f"Starting full data download for {target_date}...")
        for d_type in URL_MAP.keys():
            download_files(d_type, target_date)
        print("\nDownload process completed.")