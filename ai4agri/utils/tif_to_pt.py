"""
Converts AgriPotential dataset to precomputed PyTorch tensors to be used by custom dataset
"""

import os

import torch
from tqdm import tqdm

from ai4agri.utils.potential_dataset import PotentialDataset

data_path = "data/agripotential"
dataset = PotentialDataset("viticulture", "train", data_path)
save_dir = "data/precomputed_tensors/train/"
os.makedirs(save_dir, exist_ok=True)

CHUNK_SIZE = 256
data_chunk, label_chunk, id_chunk = [], [], []
chunk_index = 0

print(f"Extracting dataset into chunks of {CHUNK_SIZE}...")

for i in tqdm(range(len(dataset))):
    data, label, patch_id = dataset[i]

    # Append to our running chunk list
    data_chunk.append(torch.tensor(data, dtype=torch.float32))
    label_chunk.append(torch.tensor(label, dtype=torch.long))
    id_chunk.append(patch_id)

    # Save the chunk if it is full OR if it is the very last patch in the dataset
    if len(data_chunk) == CHUNK_SIZE or i == len(dataset) - 1:
        torch.save(
            {
                "data": torch.stack(
                    data_chunk
                ),  # Shape becomes (256, 34, 10, 128, 128)
                "label": torch.stack(label_chunk),  # Shape becomes (256,)
                "patch_ids": id_chunk,
            },
            os.path.join(save_dir, f"chunk_{chunk_index:04d}.pt"),
        )

        # Reset lists for the next chunk
        data_chunk, label_chunk, id_chunk = [], [], []
        chunk_index += 1
