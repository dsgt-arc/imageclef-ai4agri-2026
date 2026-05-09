import torch
import torch.nn as nn
import math
from tqdm import tqdm
import sys
import os
import torch
import copy
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import random
import threading
import time
from collections import defaultdict
from collections import OrderedDict
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import zipfile
from PIL import Image
import argparse

import tempfile
tempfile.tempdir = os.path.expandvars("$HOME/scratch/tmp/")

REFLECTANCE_SCALE = 10_000.0
CHUNKS_DIR = os.path.expandvars("$HOME/scratch/precomputed_tensors/")


def label_to_ordinal(targets, num_classes=5):
    # targets: (N,) values 1..5
    # returns: (N, 4) binary vectors
    thresholds = torch.arange(1, num_classes, device=targets.device)  # [1,2,3,4]
    return (targets.unsqueeze(1) > thresholds).float()


def loss_fn(logits, targets):
    B, C, H, W = logits.shape
    logits  = logits.permute(0, 2, 3, 1).reshape(-1, C)
    targets = targets.reshape(-1)

    mask = targets > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True), 0

    n               = mask.sum()
    ordinal_targets = label_to_ordinal(targets[mask])
    loss_sum        = F.binary_cross_entropy_with_logits(
        logits[mask], ordinal_targets, reduction='sum'
    )
    return loss_sum, n.item()

def ordinal_to_label(logits, threshold=0.5):
    # logits: [B, 4, H, W]
    probs = torch.sigmoid(logits)

    return (probs > threshold).sum(dim=1) + 1

def accuracy_pm1(logits, targets):
    preds   = ordinal_to_label(logits).reshape(-1)
    targets = targets.reshape(-1)
    mask    = targets > 0
    if mask.sum() == 0:
        return 0.0, 0
    correct = (torch.abs(preds[mask] - targets[mask]) <= 1).float().sum().item()
    n       = mask.sum().item()
    return correct, n

def accuracy_exact(logits, targets):
    preds   = ordinal_to_label(logits).reshape(-1)
    targets = targets.reshape(-1)
    mask    = targets > 0
    if mask.sum() == 0:
        return 0.0, 0
    correct = (preds[mask] == targets[mask]).float().sum().item()
    n       = mask.sum().item()
    return correct, n

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct     = 0.0
    total_correct_exact = 0.0
    total_pixels      = 0

    with torch.no_grad():
        for data, label, _ in loader:
            data, label = data.to(device), label.to(device)
            logits = model(data)
            loss_sum, n = loss_fn(logits, label)
            if n == 0:
                continue
    
            total_loss += loss_sum.item()

            correct, _       = accuracy_pm1(logits, label)
            correct_exact, _ = accuracy_exact(logits, label)
    
            total_correct       += correct
            total_correct_exact += correct_exact
            total_pixels        += n

    val_acc       = total_correct       / total_pixels
    val_acc_exact = total_correct_exact / total_pixels
    loss = total_loss / total_pixels
    
    return loss, val_acc, val_acc_exact

class ChunkedDataset(Dataset):
    def __init__(self, mode, add_indices=True, cache_size=8):
        self.mode = mode
        self.chunks_dir  = os.path.join(CHUNKS_DIR, mode)
        self.add_indices = add_indices
        self.cache_size  = cache_size
        self.index       = []

        chunk_files = sorted(f for f in os.listdir(self.chunks_dir) if f.endswith(".pt"))

        meta_path = os.path.join(self.chunks_dir, "_index.json")

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                chunk_lengths = json.load(f)
        else:
            print("Building index (first run only)...")
            chunk_lengths = {}
            for fname in chunk_files:
                payload = torch.load(
                    os.path.join(self.chunks_dir, fname),
                    map_location="cpu", weights_only=True, mmap=True
                )
                chunk_lengths[fname] = payload["data"].shape[0]
                del payload
            with open(meta_path, "w") as f:
                json.dump(chunk_lengths, f)

        for fname in chunk_files:
            n = chunk_lengths[fname]
            self.index.extend((fname, i) for i in range(n))

        self._cache: dict[str, dict] = {}
        self._cache_order = []

    def _load_chunk(self, path):
        if path in self._cache:
            self._cache_order.remove(path)
            self._cache_order.append(path)
            return self._cache[path]

        full_path = os.path.join(self.chunks_dir, path)

        payload = torch.load(full_path, map_location="cpu", weights_only=True)

        self._cache[path] = payload
        self._cache_order.append(path)

        if len(self._cache_order) > self.cache_size:
            evict = self._cache_order.pop(0)
            del self._cache[evict]

        return payload

    def shuffle(self):
        from collections import defaultdict
        import random

        chunks = defaultdict(list)
        for fname, local_idx in self.index:
            chunks[fname].append(local_idx)

        chunk_names = list(chunks.keys())
        random.shuffle(chunk_names)

        self.index = []
        for fname in chunk_names:
            local_indices = chunks[fname]
            random.shuffle(local_indices)
            self.index.extend((fname, i) for i in local_indices)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, patch_idx = self.index[idx]
        payload  = self._load_chunk(path)

        data     = payload["data"][patch_idx]   # [T, C, H, W]
        label    = payload["label"][patch_idx]  # [H, W]
        patch_id = payload["patch_ids"][patch_idx]

        data = data.to(torch.float32)
        data.div_(REFLECTANCE_SCALE).clamp_(0.0, 1.0)

        if self.add_indices:
            red, green, nir, swir = data[:, 2], data[:, 1], data[:, 6], data[:, 8]
            eps = 1e-6

            indices = torch.stack([
                (nir - red)   / (nir + red   + eps),
                (green - nir) / (green + nir + eps),
                (nir - swir)  / (nir + swir  + eps),
            ], dim=1)

            data = torch.cat([data, indices], dim=1)  # [T, C+3, H, W]

        if self.mode == "train":
            crop_size = 64
            _, _, H, W = data.shape
            min_valid  = 0.2  # at least 20% of pixels must be labeled

            h0, w0 = 0, 0
            for _ in range(10):
                _h0 = torch.randint(0, H - crop_size + 1, (1,)).item()
                _w0 = torch.randint(0, W - crop_size + 1, (1,)).item()

                crop_label = label[_h0:_h0+crop_size, _w0:_w0+crop_size]
                valid_frac = (crop_label > 0).float().mean().item()

                if valid_frac >= min_valid:
                    h0, w0 = _h0, _w0
                    break

            data  = data[:, :, h0:h0+crop_size, w0:w0+crop_size]
            label = label[h0:h0+crop_size, w0:w0+crop_size]

        return data, label, patch_id

class FlatUNet(nn.Module):
    """
    Baseline UNet: flatten all timesteps into channels.
    
    Input:  [B, T, C, H, W]
    Flatten: [B, T*C, H, W]
    Output: [B, num_classes, H, W]
    """
    def __init__(
        self,
        in_channels  = 13,
        num_timesteps= 34,
        num_classes  = 4,
        base_dim     = 64,
        dropout      = 0.2,
    ):
        super().__init__()
        flat_channels = in_channels * num_timesteps  # 13*34 = 442

        self.enc1       = self._block(flat_channels, base_dim,     dropout)
        self.enc2       = self._block(base_dim,      base_dim * 2, dropout)
        self.enc3       = self._block(base_dim * 2,  base_dim * 4, dropout)
        self.enc4       = self._block(base_dim * 4,  base_dim * 8, dropout)
        self.pool       = nn.MaxPool2d(2)

        self.bottleneck = self._block(base_dim * 8,  base_dim * 16, dropout)

        self.up4  = nn.ConvTranspose2d(base_dim * 16, base_dim * 8, 2, stride=2)
        self.dec4 = self._block(base_dim * 16, base_dim * 8, dropout)

        self.up3  = nn.ConvTranspose2d(base_dim * 8,  base_dim * 4, 2, stride=2)
        self.dec3 = self._block(base_dim * 8,  base_dim * 4, dropout)

        self.up2  = nn.ConvTranspose2d(base_dim * 4,  base_dim * 2, 2, stride=2)
        self.dec2 = self._block(base_dim * 4,  base_dim * 2, dropout)

        self.up1  = nn.ConvTranspose2d(base_dim * 2,  base_dim,     2, stride=2)
        self.dec1 = self._block(base_dim * 2,  base_dim,     dropout)

        self.out  = nn.Conv2d(base_dim, num_classes, kernel_size=1)

    def _block(self, in_ch, out_ch, dropout=0.2):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)   # [B, 442, H, W]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1) # [B, 4, H, W]

def visualize_failures(model, val_loader, device, output_dir, n_samples=10):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    count = 0

    # --- Colormaps ---
    gt_cmap = plt.cm.get_cmap('RdYlGn', 5)  # for ground truth (shows -1)

    pred_cmap = plt.cm.get_cmap('RdYlGn', 5).copy()
    pred_cmap.set_bad(color='lightgray')  # masked areas → gray

    error_cmap = plt.cm.get_cmap('RdYlGn_r').copy()
    error_cmap.set_bad(color='lightgray')

    binary_cmap = plt.cm.get_cmap('RdYlGn').copy()
    binary_cmap.set_bad(color='lightgray')

    with torch.no_grad():
        for data, label, patch_ids in val_loader:
            if count >= n_samples:
                break

            data, label = data.to(device), label.to(device)

            logits = model(data)
            preds  = ordinal_to_label(logits) - 1   # [B, H, W], 0–4
            label_0indexed = label - 1              # [B, H, W], 0–4, -1 unlabeled

            for i, pid in enumerate(patch_ids):
                pred   = preds[i].cpu().numpy()
                target = label_0indexed[i].cpu().numpy()

                mask = target >= 0  # valid pixels

                if mask.sum() == 0:
                    continue

                diff = np.abs(pred - target)

                # --- Masked arrays (KEY FIX) ---
                pred_masked   = np.ma.masked_where(~mask, pred)
                diff_masked   = np.ma.masked_where(~mask, diff)

                binary = (diff <= 1).astype(float)
                binary_masked = np.ma.masked_where(~mask, binary)

                # --- Plot ---
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                # Ground truth (keep -1 visible)
                im0 = axes[0].imshow(target, vmin=-1, vmax=4, cmap=gt_cmap)
                axes[0].set_title(f'Ground Truth\n{pid}')
                plt.colorbar(im0, ax=axes[0])

                # Prediction
                axes[1].imshow(pred_masked, vmin=0, vmax=4, cmap=pred_cmap)
                axes[1].set_title('Prediction')

                # Error magnitude
                axes[2].imshow(diff_masked, vmin=0, vmax=4, cmap=error_cmap)
                axes[2].set_title('Error magnitude')

                # Binary correctness
                acc = binary[mask].mean()
                axes[3].imshow(binary_masked, vmin=0, vmax=1, cmap=binary_cmap)
                axes[3].set_title(f'±1 correct\nacc={acc:.3f}')

                for ax in axes:
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{pid}_failure.png'), dpi=100)
                plt.close()

                count += 1

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses,   label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses,   label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (log scale)')
    plt.title('Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {save_path}")
    
if __name__ == "__main__":
    model_name = "unet_a0"

    # Check PyTorch and device
    print(f"PyTorch version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_dataset = ChunkedDataset("train", cache_size=6)
    val_dataset   = ChunkedDataset("val",   cache_size=2)

    batch_size = 32
    epochs = 300

    model = FlatUNet(
        in_channels   = 13,
        num_timesteps = 34,
        num_classes   = 4,
        base_dim      = 64,
        dropout       = 0.2,
    ).to(device)

   
    val_loader = DataLoader(
        val_dataset,
        batch_size=64, 
        shuffle=False,
    )

    model.load_state_dict(torch.load(f'{model_name}/best_model.pt', map_location=device))
    model.eval()

    visualize_failures(model, val_loader, device, f'{model_name}/failure_analysis')
