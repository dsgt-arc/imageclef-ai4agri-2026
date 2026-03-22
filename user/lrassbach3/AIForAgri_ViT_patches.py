#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# version 1.0
# %pip uninstall pandas --y
# %pip install pandas
#get_ipython().run_line_magic('pip', 'install --force-reinstall "numpy==1.21"')
#get_ipython().run_line_magic('pip', 'install --force-reinstall "pandas==1.5"')
# %pip install --force-reinstall "pandas==1.6.0"


# In[5]:


# %pip install --upgrade pandas scipy matplotlib numpy
# %pip install --upgrade python-dateutil pytz

import time
from tkinter import Image

import pandas as pd
import numpy as np
import sys

# root_path: The directory where the dataset is downloaded or hosted.
root_path = "https://huggingface.co/datasets/m-sakka/agripotential/resolve/main/"
path = root_path + "metadata.csv"
metadata = pd.read_csv(path)
metadata.head(50)


# In[4]:


# %pip install rasterio

from rasterio import rasterio
from rasterio.windows import Window

train_subset_path = root_path + "train.csv"
test_subset_path = root_path + "val.csv"
val_subset_path = root_path + "test.csv"
train_df = pd.read_csv(train_subset_path)
test_df = pd.read_csv(test_subset_path)
#val_df = pd.read_csv() #TODO

date1 = metadata.iloc[0]


# print(date1)


# In[ ]:


# to start, will focus on >T31TEJ_2017_01_03.tif,03,01,2017<


# In[ ]:


''' sources to read:
Vision Transformer (ViT)
Dosovitskiy et al., 2020

Attention Is All You Need
Vaswani et al., 2017

Spectral–Spatial Transformer for Hyperspectral Image Classification
He et al., 2021

SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers
Zhang et al., 2021
'''


# In[ ]:


# date1_data.meta

viticulture_label_path = root_path + "viticulture.tif"
viticulture_label_data = rasterio.open(viticulture_label_path)


# In[ ]:


#get_ipython().run_line_magic('pip', 'install torch')
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PotentialDataset(Dataset):
  """
  A PyTorch Dataset class for loading AgriPotential image patches
  and corresponding crop type labels, organized based on patch metadata.

  The dataset expects a root directory structure containing:
  - 'metadata.csv': A file listing Sentinel-2 file paths.
  - '{subset}.csv': A file defining the patch coordinates (row, col, patch_size)
                    for the specific subset (e.g., 'train', 'val', 'test').
  - '{crop_type}.tif': A GeoTIFF file containing the ground truth crop type
                       labels.
  - 34 Sentinel-2 GeoTIFF files referenced in 'metadata.csv'.

  Attributes:
    root_url (str): The root directory containing the data files.
                    For online loading use https://huggingface.co/datasets/m-sakka/agripotential/resolve/main/
    crop_type (str): Identifier for the label GeoTIFF file (e.g., 'crop_labels').
    subset (str): Identifier for the patch definition CSV file (e.g., 'train').
    metadata_df (pd.DataFrame): DataFrame loaded from 'metadata.csv' containing
                                Sentinel-2 file information.
    patch_df (pd.DataFrame): DataFrame loaded from '{subset}.csv' containing
                             patch details (row, col, patch_size).
    label_path (str): Full path to the crop type label GeoTIFF file.
    sentinel2_paths (list): List of full paths to the Sentinel-2 GeoTIFF files.
  """
  def __init__(self, root_url, crop_type, subset, localpath="$HOME/scratch/lrassbach3/agri_files/tif/"):
    """
    Initializes the dataset

    Args:
      root_url (str): The base directory path for the dataset.
      crop_type (str): The name of the label file (e.g., "viticulture", "market", "field")
      subset (str): The name of the subset (e.g., "train", "val", "test")
    """
    super().__init__()
    self.root_url = root_url
    self.subset = subset
    localpath = os.path.expandvars(localpath)

    # Load metadata and patch information
    self.metadata_df = pd.read_csv(self.root_url + "metadata.csv")
    self.patch_df = pd.read_csv(self.root_url + subset +".csv")

    # Define paths
    self.label_path = localpath + crop_type + ".tif"
    self.sentinel2_paths = []
    for f in self.metadata_df["filename"]:
      self.sentinel2_paths.append(os.path.join(localpath, f))

  def __len__(self):
    """
    Returns the total number of patches in the dataset subset.

    Returns:
      int: The number of rows in self.patch_df.
    """
    return len(self.patch_df)

  def __getitem__(self, idx):
    """
    Retrieves the Sentinel-2 data, corresponding label, and patch ID for a given index.

    The Sentinel-2 data is stacked across the time dimension (i.e., multiple GeoTIFFs),
    and then the bands from each image are implicitly stacked resulting in a
    shape of (34 Timeframes, 10 Bands, PatchSize, PatchSize)

    Args:
      idx (int): The index of the patch to retrieve (from 0 to len-1).

    Returns:
      tuple: A tuple containing:
        - data (np.ndarray): The stacked Sentinel-2 patch data. Shape: (34, 10, P, P) where P is patch_size.
        - label (np.ndarray): The corresponding crop type label patch. Shape: (P, P).
        - patch_id (int or str): The unique identifier for the patch.
    """
    patch = self.patch_df.iloc[idx]
    patch_row = patch["row"]
    patch_col = patch["col"]
    patch_size = patch["patch_size"]
    patch_id = patch["patch_id"]

    data = np.empty((34, 10, patch_size, patch_size), dtype=np.float32)
    window = Window(patch_col, patch_row, patch_size, patch_size)
    def safe_read(src, window, retries=3):
        for attempt in range(retries):
            try:
                return src.read(window=window)
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                print(f"[WARN] Read failed, retrying ({attempt+1}/{retries})...")
                time.sleep(0.1)

    label = None
    for i, fp in enumerate(self.sentinel2_paths):
      with rasterio.open(fp) as src:
        data[i] = safe_read(src,window=window)
    with rasterio.open(self.label_path) as src:
        label = safe_read(src,window)
        label = label[0]

    return data, label, patch_id

# In[ ]:

class SpectralViTPixel(nn.Module):
    def __init__(self, num_bands, num_classes,
                 patch_size=128, patch=4,
                 d_model=128, depth=4, nhead=4):
        super().__init__()

        self.num_bands = num_bands
        self.patch = patch
        self.patch_size = patch_size

        # Number of patches along each dimension
        self.grid_H = patch_size // patch
        self.grid_W = patch_size // patch
        num_patches = self.grid_H * self.grid_W

        # 1. Patch embedding MLP (replaces pixel_embed)
        self.patch_embed = nn.Sequential(
            nn.Linear(num_bands * patch * patch, 256),
            nn.GELU(),
            nn.Linear(256, d_model)
        )

        # Normalize embeddings
        self.embed_norm = nn.LayerNorm(d_model)

        # 2. 2D positional embeddings for patch grid
        self.row_embed = nn.Parameter(torch.randn(self.grid_H, d_model))
        self.col_embed = nn.Parameter(torch.randn(self.grid_W, d_model))

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Classification head (per patch)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = x.mean(dim=1)   # (B, C, H, W)
        B, C, H, W = x.shape
        patch = self.patch

        # -------------------------------
        # PATCHIFY
        # -------------------------------
        # x: (B, C, H, W)
        x = x.unfold(2, patch, patch).unfold(3, patch, patch)
        # shape: (B, C, grid_H, grid_W, patch, patch)

        x = x.permute(0, 2, 3, 1, 4, 5)
        # shape: (B, grid_H, grid_W, C, patch, patch)

        x = x.reshape(B, self.grid_H * self.grid_W, C * patch * patch)
        # shape: (B, N_patches, C * patch^2)

        # -------------------------------
        # PATCH EMBEDDING
        # -------------------------------
        x = self.patch_embed(x)
        x = self.embed_norm(x)

        # -------------------------------
        # POSITIONAL ENCODING
        # -------------------------------
        rows = self.row_embed.unsqueeze(1).expand(self.grid_H, self.grid_W, -1)
        cols = self.col_embed.unsqueeze(0).expand(self.grid_H, self.grid_W, -1)
        pos = (rows + cols).reshape(self.grid_H * self.grid_W, -1)

        x = x + pos.unsqueeze(0)

        # -------------------------------
        # TRANSFORMER
        # -------------------------------
        x = self.encoder(x)

        # -------------------------------
        # CLASSIFICATION
        # -------------------------------
        logits = self.cls_head(x)  # (B, N_patches, num_classes)

        # -------------------------------
        # RESHAPE BACK TO PATCH GRID
        # -------------------------------
        logits = logits.reshape(B, self.grid_H, self.grid_W, -1)
        logits = logits.permute(0, 3, 1, 2)  # (B, num_classes, grid_H, grid_W)

        return logits


# In[ ]:


#get_ipython().run_line_magic('pip', 'install tqdm')
# debug
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from tqdm import trange

# model init
model = SpectralViTPixel(num_bands=10,num_classes=5).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# get mean for normalization

print("computing mean & std")

channel_sum = None
channel_sq_sum = None
pixel_count = 0
'''
for ind in trange(34, desc='files', leave=False):
    date = metadata.iloc[ind]
    date_data = rasterio.open(root_path + date["filename"])
    batch_size = 16
    for n in range(395):
        for i in range(batch_size): 
            index = i + (n * batch_size)

            patch_row = train_df.iloc[index]["row"]
            patch_col = train_df.iloc[index]["col"]
            patch_size = train_df.iloc[index]["patch_size"]

            patch = date_data.read(window=Window(patch_col, patch_row, patch_size, patch_size))
            patch = torch.from_numpy(patch).float()  # (C, H, W)

            C, H, W = patch.shape
            pixels = H * W

            # Initialize accumulators on first patch
            if channel_sum is None:
                channel_sum = torch.zeros(C)
                channel_sq_sum = torch.zeros(C)

            # Sum over spatial dims
            patch_flat = patch.view(C, -1)
            channel_sum += patch_flat.sum(dim=1)
            channel_sq_sum += (patch_flat ** 2).sum(dim=1)
            pixel_count += pixels

    date_data.close()

# Final statistics
mean = channel_sum / pixel_count
std = torch.sqrt(channel_sq_sum / pixel_count - mean ** 2)

print("mean:", mean)
print("std:", std)
'''

total_loss = 0
total_correct = 0
total_pixels = 0
# train
# pull data from source
size = 800
# batch size of 1
# for iter in range(size):
train_mode = True
test = False
if train_mode:
    print("train mode")

    dataset = PotentialDataset(root_path, "viticulture", "train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    iterator = iter(dataloader)
    epochs = 80
    for i in trange(epochs, desc='epochs'):
        for x, y, pid in dataloader:
            # uncomment for normalization
            # x = (x - mean[None, :, None, None]) / std[None, :, None, None]

    #     print(f"xshape : {x.shape}")
            x = x.to(device)
            Y = y.to(device)

    #     valid_ratio = (labels != 0).float().mean().item() 
    #     print("Valid pixel ratio:", valid_ratio)
            optimizer.zero_grad()
        # run forward pass on model
            logits = model.forward(x)
        # loss
            B, K, grid_H, grid_W = logits.shape
            patch = model.patch

            # y: (B, H, W)
            patch_labels = (
                y.unfold(1, patch, patch)
                .unfold(2, patch, patch)
                .reshape(B, grid_H, grid_W, -1)
                .mode(dim=-1).values
            )  # (B, grid_H, grid_W)
            mask = patch_labels != 0
            patch_labels = patch_labels.to(device)
            mask = mask.to(device)
            patch_labels = patch_labels[mask] - 1
            logits = logits.permute(0, 2, 3, 1).reshape(-1, K)
            logits = logits[mask.reshape(-1)]
            patch_labels = patch_labels.long()


            # Compute loss
            loss = criterion(logits, patch_labels)

            loss.backward()
            optimizer.step()
            # Accuracy
            preds = logits.argmax(dim=1)  # (B, grid_H, grid_W)
            total_correct += (preds == patch_labels).sum().item()
            total_pixels  += patch_labels.numel()
            total_loss    += loss.item() * patch_labels.numel()

        # Cuda cleanup
            del x
            del Y


# In[ ]:

    torch.save(model.state_dict(), 'model_weights_viaDL.pth')


    avg_loss = total_loss / total_pixels
    accuracy = total_correct / total_pixels
    print(f"training avg_loss: {avg_loss}; training accuracy: {accuracy}")
elif test:
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    state_dict = torch.load('model_weights_full_norm.pth')
    model.load_state_dict(state_dict) 
    print("test mode")
    with torch.no_grad():
        for ind in trange(34, desc='files', leave=False):
            date = metadata.iloc[ind]
            date_data = rasterio.open(root_path+date["filename"])
            batch_size = 16 # TODO these will need to be revisited
            # 252 * 3 is max training
            for n in range(48):
                x = []
                Y = []
      #     print(f"file: {ind}; batch: {n}")
                for i in range(batch_size):
                    index = i + (n * batch_size)
                 # print(f"index: {index}")
                    patch_row = test_df.iloc[index]["row"]
                    patch_col = test_df.iloc[index]["col"]
                    patch_size = test_df.iloc[index]["patch_size"]
                    patch_id = test_df.iloc[index]["patch_id"]
                    image_x = date_data.read(window=Window(patch_col, patch_row, patch_size, patch_size))
                    x.append(torch.from_numpy(image_x))
                    image_y = viticulture_label_data.read(window=Window(patch_col, patch_row, patch_size, patch_size))
                    Y.append(torch.from_numpy(image_y))
                
                x = torch.stack(x, dim=0)
                x = (x - mean[None, :, None, None]) / std[None, :, None, None]
                #print(f"xshape : {x.shape}")
                x = x.to(torch.float32)
                x = x.to(device)
                Y = torch.stack(Y, dim=0)
                #print(f"yshape : {Y.shape}")
                Y = Y.to(torch.int64)
                Y = Y.to(device)
                logits = model.forward(x) 
                # (B, K, grid_H, grid_W) 
                B, K, grid_H, grid_W = logits.shape 
                patch = model.patch
                H = W = grid_H * patch 
                labels = Y.squeeze(1)
                patch_labels = labels.unfold(1, patch, patch).unfold(2, patch, patch)
                patch_labels = patch_labels.reshape(B, grid_H, grid_W, -1).mode(dim=-1).values
                
                patch_mask = patch_labels != 0

                logits = logits.permute(0,2,3,1).reshape(-1, K) 
                patch_labels = patch_labels.reshape(-1) 
                patch_mask = patch_mask.reshape(-1)
                
                logits = logits[patch_mask]
                patch_labels = patch_labels[patch_mask]

                patch_labels = patch_labels - 1

                preds = logits.argmax(dim=1) 
                total_correct += (preds == patch_labels).sum().item() 
                total_pixels += patch_labels.numel()

                del x
                del Y
        avg_loss = total_loss / total_pixels
        accuracy = total_correct / total_pixels
        print(f"test avg_loss: {avg_loss}; test accuracy: {accuracy}")
else:
    print("error - test or train not selected")
