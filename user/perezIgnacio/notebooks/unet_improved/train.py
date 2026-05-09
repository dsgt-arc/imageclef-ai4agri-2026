# %%
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# %%
import os
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from PIL import Image
import random


# %%
from utils import ordinal_predict, accuracy_exact, accuracy_pm1, valid_mask, ordinal_loss, evaluate, plot_loss_curve

# %%
REFLECTANCE_SCALE = 10_000.0
CHUNKS_DIR = os.path.expandvars("$HOME/scratch/precomputed_tensors_2/")

# %%
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
        label    = payload["label"][patch_idx] - 1  # [H, W]
        patch_id = payload["patch_ids"][patch_idx]

        data = data.to(torch.float32)
        data = data / REFLECTANCE_SCALE

        # bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
        if self.add_indices:
            B4 = data[:, 2] # R
            B3 = data[:, 1] # G
            B6 = data[:, 4] # B6
            B8 = data[:, 6] # B8
            B11 = data[:, 7] # B11
            B12 = data[:, 9] # B12

            eps = 1e-6

            NDVI  = (B8 - B4) / (B8 + B4 + eps)
            NDMI  = (B8 - B11) / (B8 + B11 + eps)
            NDWI  = (B3 - B8) / (B3 + B8 + eps)
            NDRE  = (B8 - B6) / (B8 + B6 + eps)
            NBR   = (B8 - B12) / (B8 + B12 + eps)

            indices = torch.stack([NDVI, NDMI, NDWI, NDRE, NBR], dim=1)  # [T, 5, H, W]

            data = torch.cat([data, indices], dim=1)  # [T, C+5, H, W]

        return data, label, patch_id


# %%
# Check PyTorch and device
print(f"PyTorch version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# %%
train_dataset = ChunkedDataset("train", cache_size=6)
val_dataset   = ChunkedDataset("val",   cache_size=2)

batch_size = 32
epochs = 100

val_loader = DataLoader(
    val_dataset,
    batch_size=64, 
    shuffle=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    shuffle=False,
)

# %%
def generate_confusion_matrix(model, val_loader, device):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, label, _ in val_loader:
            data, label = data.to(device), label.to(device)

            logits = model(data)
            preds = ordinal_predict(logits)
            
            mask = label != -1

            labels = label[mask]
            preds  = preds[mask]

            # collect for confusion matrix
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    print("Unique preds:", torch.unique(torch.cat(all_preds)))
    print("Unique labels:", torch.unique(torch.cat(all_labels)))

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3,4])

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=[0,1,2,3,4], 
                yticklabels=[0,1,2,3,4],
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix.png')
    plt.close()


# %%
test_dataset = ChunkedDataset("test", cache_size=2)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# %%
def generate_submission(model, test_loader, device):
    output_dir = "submissions"
    count = 0

    total_counts = torch.zeros(5, dtype=torch.long)

    with torch.no_grad():
        for data, _, patch_ids in test_loader:
            data = data.to(device)
            B = data.shape[0]

            logits = model(data)
            preds = ordinal_predict(logits)

            for c in range(5):
                total_counts[c] += (preds == c).sum().item()

            for pred, pid in zip(preds.cpu().numpy(), patch_ids):
                img = Image.fromarray(pred.astype(np.uint8), mode='L')
                img.save(os.path.join(output_dir, f"{pid}.png"))
                count += 1

    print(f"Total predictions: {total_counts.sum().item()}")
    for c in range(5):
        pct = total_counts[c].item() / total_counts.sum().item() * 100
        print(f"  class {c}: {total_counts[c].item():>10,}  ({pct:.1f}%)")

    # Zip all predictions
    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith('.png'):
                zf.write(os.path.join(output_dir, fname), fname)

    print(f"Saved {count} predictions → {zip_path}")


# %%
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(dropout)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = self.relu(out)

        return out


# %%
class ResUNetOrdinal(nn.Module):
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

        # -------- Encoder (3 levels) --------
        self.enc1 = ResBlock(flat_channels, base_dim, dropout)
        self.enc2 = ResBlock(base_dim, base_dim * 2, dropout)
        self.enc3 = ResBlock(base_dim * 2, base_dim * 4, dropout)

        self.pool = nn.MaxPool2d(2)

        # -------- Bottleneck --------
        self.bottleneck = ResBlock(base_dim * 4, base_dim * 8, dropout)

        # -------- Decoder --------
        self.up3  = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, 2, stride=2)
        self.dec3 = ResBlock(base_dim * 8, base_dim * 4, dropout)

        self.up2  = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2)
        self.dec2 = ResBlock(base_dim * 4, base_dim * 2, dropout)

        self.up1  = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2)
        self.dec1 = ResBlock(base_dim * 2, base_dim, dropout)

        self.head = nn.Conv2d(base_dim, 1, kernel_size=1)

        # K-1 thresholds
        self.raw_bias = nn.Parameter(torch.zeros(num_classes))

    def _block(self, in_ch, out_ch, dropout=0.2):
        return ResBlock(in_ch, out_ch, dropout)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # -------- Encoder --------
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # -------- Bottleneck --------
        b = self.bottleneck(self.pool(e3))

        # -------- Decoder --------
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # -------- Ordinal Head --------
        features = self.head(d1)  # [B,1,H,W]

        thresholds = torch.cumsum(F.softplus(self.raw_bias), dim=0)
        thresholds = thresholds.view(1, -1, 1, 1)

        logits = features - thresholds  # [B, K-1, H, W]

        return logits


# %%
epochs = 100

model = ResUNetOrdinal(
    in_channels   = 15,
    num_timesteps = 34,
    num_classes   = 4,
    base_dim      = 128,
    dropout       = 0.2,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = 1e-4, # 1e-4 initially, changed to 1e-5
    weight_decay = 1e-2,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max  = epochs,
    eta_min= 1e-7,
)

# %%
train_losses = []
val_losses = []

best_val_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(epochs):
    train_dataset.shuffle()
    model.train()

    train_loss = 0.0
    train_acc_pm1 = 0
    train_acc_exact = 0
    num_batches = 0

    for data, label, _ in train_loader:
        data, label = data.cuda(), label.cuda()

        # -------------------
        # augmentation
        # -------------------
        k = random.randint(0, 3)
        if k > 0:
            data = torch.rot90(data, k, [-2, -1])
            label = torch.rot90(label, k, [-2, -1])

        if random.random() > 0.5:
            data = torch.flip(data, [-1])
            label = torch.flip(label, [-1])

        # -------------------
        # forward
        # -------------------
        
        optimizer.zero_grad()

        logits = model(data)

        n = valid_mask(label).sum().item()

        if n == 0:
            continue

        loss = ordinal_loss(logits, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        num_batches += 1
        train_loss += loss.item()

        # -------------------
        # metrics
        # -------------------
        with torch.no_grad():
            correct_pm1 = accuracy_pm1(logits, label)
            correct_exact = accuracy_exact(logits, label)
            
            if num_batches == 1:
                preds = ordinal_predict(logits)
                mask = valid_mask(label)
                valid_preds = preds[mask]  # only labeled pixels
                counts = torch.bincount(valid_preds.view(-1), minlength=5)

                print("Class counts:", counts.tolist())
                print("Class ratios:", (counts / counts.sum()).tolist())

        train_acc_pm1 += correct_pm1.item()
        train_acc_exact += correct_exact.item()

    train_loss = train_loss / num_batches
    train_acc_pm1 = train_acc_pm1 / num_batches
    train_acc_exact = train_acc_exact / num_batches

    # -------------------
    # validation
    # -------------------
    val_loss, val_acc_pm1, val_acc_exact = evaluate(model, val_loader, device)

    print(
        f"Epoch {epoch:3d} | "
        f"train loss {train_loss:.4f} pm1 {train_acc_pm1:.4f} exact {train_acc_exact:.4f} | "
        f"val loss {val_loss:.4f} pm1 {val_acc_pm1:.4f} exact {val_acc_exact:.4f}"
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # -------------------
    # checkpointing
    # -------------------
    if epoch % 10 == 0:
        plot_loss_curve(train_losses, val_losses, f'loss_curve.png')

    # -------------------
    # early stopping (loss-based)
    # -------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), f'best.pt')
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} — best val loss: {best_val_loss:.4f}")
            break

# final plot
plot_loss_curve(train_losses, val_losses, f'loss_curve.png')

# %%
model.eval()

val_loss, val_acc_pm1, val_acc_exact = evaluate(model, val_loader, device)
print(f"val loss {val_loss:.4f} pm1 {val_acc_pm1:.4f} exact {val_acc_exact:.4f}")

generate_confusion_matrix(model, val_loader, device)

generate_submission(model, test_loader, device)