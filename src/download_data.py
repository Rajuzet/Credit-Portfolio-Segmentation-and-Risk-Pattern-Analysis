import os
import requests
import sys

URL = "https://zenodo.org/api/records/11295916/files/LC_loans_granting_model_dataset.csv/content"
DEST_PATH = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\data\raw\loans_granting_model_dataset.csv"

os.makedirs(os.path.dirname(DEST_PATH), exist_ok=True)

print(f"Starting download from Zenodo to {DEST_PATH}...")
try:
    with requests.get(URL, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        last_percent = -1
        
        with open(DEST_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int((downloaded / total_size) * 100)
                        if percent != last_percent and percent % 10 == 0:
                            print(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent}%)")
                            last_percent = percent

    print(f"Download completed successfully! File size: {os.path.getsize(DEST_PATH) / (1024*1024):.2f} MB")
except Exception as e:
    print(f"Download failed: {e}", file=sys.stderr)
    sys.exit(1)
