"""
Streams AgriPotential precomputed tensors as an IterableDataset.
"""

import os
import random

import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingChunkDataset(IterableDataset):
    def __init__(self, chunk_dir, shuffle=True, seed=42):
        super().__init__()
        self.chunk_dir = chunk_dir
        # Sort files to ensure all workers start with the exact same baseline list
        self.chunk_files = sorted(
            [f for f in os.listdir(chunk_dir) if f.endswith(".pt")]
        )
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # --- WORKER SHARDING ---
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Slice the file list so each worker gets unique files
        # e.g., Worker 0 gets files [0, 4, 8], Worker 1 gets [1, 5, 9]
        worker_files = self.chunk_files[worker_id::num_workers]

        # Set up a localized random generator to prevent identical worker shuffles
        rng = random.Random(self.seed + worker_id)
        # -----------------------------------------

        # 1. Shuffle the order in which we load the massive chunks
        if self.shuffle:
            rng.shuffle(worker_files)

        # 2. Process one chunk at a time
        for chunk_file in worker_files:
            file_path = os.path.join(self.chunk_dir, chunk_file)

            # memory-mapped load leaves the heavy array on disk until yielded
            payload = torch.load(file_path, weights_only=True, mmap=True)
            chunk_data = payload["data"]
            chunk_labels = payload["label"]

            num_patches_in_chunk = chunk_data.size(0)

            # 3. Create an index array to shuffle patches WITHIN the chunk
            indices = list(range(num_patches_in_chunk))
            if self.shuffle:
                rng.shuffle(indices)

            # 4. Yield individual patches one-by-one to the DataLoader
            for idx in indices:
                yield chunk_data[idx], chunk_labels[idx]


# Example

# # You can now save huge files to disk (e.g., 1000 patches per .pt file)
# dataset = StreamingChunkDataset("data/precomputed_tensors/train")
#
# dataloader = DataLoader(
#     dataset,
#     batch_size=128,
#     num_workers=4,  # 4 CPU workers will stream different chunks in parallel
#     pin_memory=True
# )
#
# for data, label in dataloader:
#     # Data shape will be perfectly [128, 34, 10, 128, 128]
#     print("Done")
#     break
