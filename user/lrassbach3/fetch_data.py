"""
download tif files and csv files
"""

import os
import sys
import requests
import agripotential

from agripotential.dataset import PotentialDataset, download_dataset

# ---------------------------------------------------------------------------

SAVE_DIR = os.path.expandvars(f"$HOME/scratch/lrassbach3/agri_files") # change for val/test

os.makedirs(f"{SAVE_DIR}/tif", exist_ok=True)

if not os.path.isdir(SAVE_DIR):
    sys.exit(f"ERROR: save directory '{SAVE_DIR}' does not exist")
# ---------------------------------------------------------------------------

def download_file(src_url: str, dest_path: str):
    with requests.get(src_url, stream=True) as response:
        response.raise_for_status()  # Raise exception for HTTP errors
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:
                    f.write(chunk)

download_file(src_url="https://huggingface.co/datasets/m-sakka/agripotential/resolve/main/test.csv", dest_path=os.path.expandvars(f"{SAVE_DIR}/test.csv"))
download_file(src_url="https://huggingface.co/datasets/m-sakka/agripotential/resolve/main/val.csv", dest_path=os.path.expandvars(f"{SAVE_DIR}/val.csv"))
download_file(src_url="https://huggingface.co/datasets/m-sakka/agripotential/resolve/main/train.csv", dest_path=os.path.expandvars(f"{SAVE_DIR}/train.csv"))

download_dataset(os.path.expandvars(f"{SAVE_DIR}/tif"), download_images=True)
