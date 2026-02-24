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

import pandas as pd

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

train_df = pd.read_csv(train_subset_path)
print(train_df.head(50))



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


# In[ ]:

class HybridSpectralEncoder(nn.Module):
    def __init__(self, num_bands, d_model):
        super().__init__()

        # 1D CNN over spectral dimension
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU()
        )

        # MLP for global nonlinear mixing
        self.mlp = nn.Sequential(
            nn.Linear(64 * num_bands, 256),
            nn.GELU(),
            nn.Linear(256, d_model)
        )

    def forward(self, x):
        # x: (B, HW, C)
        B, N, C = x.shape

        # CNN expects (B*N, 1, C)
        x = x.reshape(B * N, 1, C)

        x = self.cnn(x)              # (B*N, 64, C)
        x = x.flatten(1)             # (B*N, 64*C)
        x = self.mlp(x)              # (B*N, d_model)

        return x.reshape(B, N, -1)

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

total_loss = 0
total_correct = 0
total_pixels = 0
# train
# pull data from source
size = 800
# batch size of 1
# for iter in range(size):
epochs = 1
for i in trange(epochs, desc='epochs'):
    for ind in trange(34, desc='files', leave=False):
        # for ind in range(34):
        #   print(f"file: {ind}")
        date = metadata.iloc[ind]
        date_data = rasterio.open(root_path+date["filename"])
        batch_size = 16
        # 252 * 3 is max training
        for n in range(48):
            x = []
            Y = []
        #     print(f"file: {ind}; batch: {n}")
            for i in range(batch_size):
                index = i + (n * batch_size)
                # print(f"index: {index}")
                patch_row = train_df.iloc[index]["row"]
                patch_col = train_df.iloc[index]["col"]
                patch_size = train_df.iloc[index]["patch_size"]
                patch_id = train_df.iloc[index]["patch_id"]
                image_x = date_data.read(window=Window(patch_col, patch_row, patch_size, patch_size))
                x.append(torch.from_numpy(image_x))
                image_y = viticulture_label_data.read(window=Window(patch_col, patch_row, patch_size, patch_size))
                Y.append(torch.from_numpy(image_y))

            x = torch.stack(x, dim=0)
        #     print(f"xshape : {x.shape}")
            x = x.to(torch.float32)
            x = x.to(device)

            Y = torch.stack(Y, dim=0)
            Y = Y.to(torch.int64)
            Y = Y.to(device)

        #     valid_ratio = (labels != 0).float().mean().item() 
        #     print("Valid pixel ratio:", valid_ratio)
            
            # run forward pass on model
            logits = model.forward(x)
            # loss
            B, K, grid_H, grid_W = logits.shape
            patch = model.patch
            H = W = grid_H * patch # 128
            
            labels = Y.reshape(B, H, W)
            
            patch_labels = labels.unfold(1, patch, patch).unfold(2, patch, patch) 
            # (B, grid_H, grid_W, patch, patch) # pick top-left pixel (or majority vote) 
            patch_labels = patch_labels.reshape(B, grid_H, grid_W, -1)
            patch_labels = patch_labels.mode(dim=-1).values
            # (B, grid_H, grid_W) 
            # --------------------------------------- 
            # 2. Build patch-level mask 
            # --------------------------------------- 
            patch_mask = patch_labels != 0 
            # same shape as patch grid # --------------------------------------- 
            #3. Flatten logits and labels # --------------------------------------- 
            logits = logits.permute(0, 2, 3, 1).reshape(-1, K) 
            patch_labels = patch_labels.reshape(-1) 
            patch_mask = patch_mask.reshape(-1) 
            # --------------------------------------- # 4. Apply mask # --------------------------------------- 
            logits = logits[patch_mask] 
            patch_labels = patch_labels[patch_mask] # --------------------------------------- 
            # 5. Shift labels if needed # --------------------------------------- 
            patch_labels = patch_labels - 1
        
            loss = criterion(logits, patch_labels).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * patch_labels.numel()

            # Accuracy
            preds = logits.argmax(dim=1)
            total_correct += (preds == patch_labels).sum().item()
            total_pixels+= patch_labels.numel()

            # Cuda cleanup
            del x
            del Y


# In[ ]:

torch.save(model.state_dict(), 'model_weights.pth')


avg_loss = total_loss / total_pixels
accuracy = total_correct / total_pixels
print(f"training avg_loss: {avg_loss}; training accuracy: {accuracy}")


# In[ ]:

'''
# save model
# from google.colab import drive
from datetime import datetime

current_time = datetime.now()
timestamp_string = current_time.strftime("%Y-%m-%d_%H:%M:%S")

basename = "_model_weights.pth"
filename = timestamp_string + basename

# drive.mount('/content/gdrive')
PATH = '/ps-clef2026_img_ai4agri-0/' + filename

torch.save(model.state_dict(), PATH)


# In[ ]:


test_subset_path = root_path + "test.csv"

test_df = pd.read_csv(test_subset_path)
test_df.head()


# In[ ]:


# run on test
total_loss = 0
total_correct = 0
total_pixels = 0

# test
for i in range():
  date1 = metadata.iloc[0]
  date1_data = rasterio.open(root_path+date1["filename"])
  training_iterations = 6
  train_size = 3
  total_batches_for_training = train_size * training_iterations
  print(f"full training size : {total_batches_for_training}")
  for i in range(training_iterations):
    x = []
    Y = []
    # pull data from source
    start_index = i * train_size
    end_index = (i * train_size) + train_size
    for index in range(start_index, end_index):
      patch_row = train_df.iloc[index]["row"]
      patch_col = train_df.iloc[index]["col"]
      patch_size = train_df.iloc[index]["patch_size"]
      patch_id = train_df.iloc[index]["patch_id"]
      image = date1_data.read(window=Window(patch_col, patch_row, patch_size, patch_size))
      x.append(torch.from_numpy(image))

    x = torch.stack(x, dim=0)
    print(f"xshape : {x.shape}")
    x = x.to(torch.float32)
    x = x.to(device)


    for index in range(start_index, end_index):
      patch_row = train_df.iloc[index]["row"]
      patch_col = train_df.iloc[index]["col"]
      patch_size = train_df.iloc[index]["patch_size"]
      patch_id = train_df.iloc[index]["patch_id"]
      image = viticulture_label_data.read(window=Window(patch_col, patch_row, patch_size, patch_size))
      Y.append(torch.from_numpy(image))

    Y = torch.stack(Y, dim=0)
    Y = Y.to(device)

    labels = Y.reshape(-1)

    # run forward pass on model
    logits = model.forward(x)
    # loss
    B, K, H, W = logits.shape
    logits = logits.permute(0, 2, 3, 1).reshape(-1, K)
    mask = labels != 0
    logits = logits[mask]
    labels = labels[mask]

    loss = criterion(logits, labels).to(device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * labels.numel()

    # Accuracy
    preds = logits.argmax(dim=1)
    total_correct += (preds == labels).sum().item()
    total_pixels += labels.numel()

    # Cuda cleanup
    del x
    del Y


# In[ ]:



'''
