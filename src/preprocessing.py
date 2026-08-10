"""
Fetch AgriPotential patches and convert them to precomputed torch tensors.
"""

import os
import sys

import requests
import torch
from tqdm import tqdm
from agripotential.dataset import PotentialDataset, download_dataset
import rasterio
from rasterio.transform import xy
from pyproj import Transformer

LABEL = 'viticulture'
SPLIT = 'test'
CHUNK_SIZE = 64
DOWNLOAD_DATASET = False
DOWNLOAD_IMAGES = False
DOWNLOAD_TEST = False

DATA_PATH = os.path.expandvars('$HOME/scratch/agripotential/')
SAVE_DIR = os.path.expandvars(f'$HOME/scratch/precomputed_tensors_2/{SPLIT}/')

if not os.path.isdir(SAVE_DIR):
    sys.exit(f"ERROR: save directory '{SAVE_DIR}' does not exist")


def download_file(src_url: str, dest_path: str):
    with requests.get(src_url, stream=True) as response:
        response.raise_for_status()
        with open(dest_path, 'wb') as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def get_patch_latlon(row, col, patch_size, transform, transformer):
    center_row = row + patch_size // 2
    center_col = col + patch_size // 2
    x, y = xy(transform, center_row, center_col)
    lon, lat = transformer.transform(x, y)
    return lat, lon


def main():
    print('initializing dataset...')

    if DOWNLOAD_TEST:
        download_file(
            src_url='https://huggingface.co/datasets/m-sakka/agripotential/resolve/main/test.csv',
            dest_path=os.path.expandvars('$HOME/scratch/agripotential/test.csv'),
        )

    if DOWNLOAD_DATASET:
        download_dataset(os.path.expandvars('$HOME/scratch'), download_images=DOWNLOAD_IMAGES)

    dataset = PotentialDataset(LABEL, SPLIT, data_path=DATA_PATH)

    with rasterio.open(dataset.label_path) as src:
        src_crs = src.crs.to_string()

    transformer = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)

    with rasterio.open(dataset.label_path) as ref_src:
        ref_transform = ref_src.transform

    data_chunk, label_chunk, id_chunk, latlon_chunk = [], [], [], []
    chunk_index = 0

    print(f'extracting dataset into chunks of {CHUNK_SIZE}...')

    for i in tqdm(range(len(dataset))):
        data, label, patch_id = dataset[i]
        patch_meta = dataset.patches.iloc[i]
        lat, lon = get_patch_latlon(
            patch_meta['row'], patch_meta['col'], patch_meta['patch_size'], ref_transform, transformer
        )

        data_chunk.append(torch.tensor(data, dtype=torch.float32))
        label_chunk.append(torch.tensor(label, dtype=torch.long))
        id_chunk.append(patch_id)
        latlon_chunk.append(torch.tensor([lat, lon], dtype=torch.float32))

        if len(data_chunk) == CHUNK_SIZE or i == len(dataset) - 1:
            fname = os.path.join(SAVE_DIR, f'chunk_{chunk_index:04d}.pt')
            torch.save({
                'data': torch.stack(data_chunk),
                'label': torch.stack(label_chunk),
                'patch_ids': id_chunk,
                'latlon': torch.stack(latlon_chunk),
            }, fname)
            data_chunk, label_chunk, id_chunk = [], [], []
            chunk_index += 1

    print('done')


if __name__ == '__main__':
    main()
