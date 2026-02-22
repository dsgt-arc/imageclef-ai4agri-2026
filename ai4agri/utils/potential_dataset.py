# Copied from https://github.com/MohammadElSakka/agripotential/blob/main/agripotential/dataset/potential_dataset.py
import os
from typing import Literal

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT_URL = "https://huggingface.co/datasets/m-sakka/agripotential/resolve/main/"


class PotentialDataset:
    def __init__(
        self,
        label_name: Literal["viticulture", "market", "field"],
        mode: Literal["train", "val", "test"],
        data_path: str | None = None,
    ):
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = ROOT_URL

        self.metadata_path = os.path.join(self.data_path, "metadata.csv")
        self.patch_csv_path = os.path.join(self.data_path, f"{mode}.csv")
        self.label_path = os.path.join(self.data_path, f"{label_name}.tif")

        self.sentinel2_paths: list[str] = []
        self.patches: pd.DataFrame = pd.DataFrame()
        self._setup()

    def _setup(self):
        metadata_df = pd.read_csv(self.metadata_path)
        self.sentinel2_paths = [
            os.path.join(self.data_path, f) for f in metadata_df["filename"]
        ]
        self.patches = pd.read_csv(self.patch_csv_path)

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, str]:
        patch_meta = self.patches.iloc[idx]
        row, col, patch_size, patch_id = (
            patch_meta["row"],
            patch_meta["col"],
            patch_meta["patch_size"],
            patch_meta["patch_id"],
        )
        window = Window(col, row, patch_size, patch_size)

        data = np.empty((34, 10, patch_size, patch_size), dtype=np.float32)
        for i, fp in enumerate(self.sentinel2_paths):
            with rasterio.open(fp) as src:
                data[i] = src.read(window=window)

        with rasterio.open(self.label_path) as src:
            label = src.read(window=window)[0].astype(np.uint8)

        return data, label, patch_id
