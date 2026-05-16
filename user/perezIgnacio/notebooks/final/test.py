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
from tqdm import tqdm
from datetime import datetime


# %%
import terratorch
from terratorch.tasks import SemanticSegmentationTask

# %%
from utils import ordinal_predict, accuracy_exact, accuracy_pm1, valid_mask, ordinal_loss, evaluate, plot_loss_curve, ordinal_confidence, label_to_ordinal

# %%
REFLECTANCE_SCALE = 10_000.0
CHUNKS_DIR = os.path.expandvars("$HOME/scratch/precomputed_tensors_2/")
CHECKPOINTS_DIR = os.path.expandvars("$HOME/scratch/checkpoints/")

SEASON_INDICES = {
    'winter': [0, 12, 13, 22, 23, 24, 33],
    'spring': [1, 14, 25, 26, 27],
    'summer': [2, 3, 4, 5, 15, 16, 17, 18, 19, 28, 29, 30],
    'autumn': [6, 7, 8, 9, 10, 11, 20, 21, 31, 32],
}

CLASS_WEIGHTS = [0.8457, 1.0195, 0.8576, 1.1605, 1.2362]

# our bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
# prithvi:   B2, B3, B4,                 B8A, B11, B12
PRITHVI_BAND_INDICES = [0, 1, 2, 7, 8, 9]

MEANS = torch.tensor([494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962, 1739.579917])
STDS  = torch.tensor([284.925432, 357.84876,  575.566823, 896.601013,  951.900334,  921.407808])

# %%
def aggregate_seasons(data):
    # data: [T, C, H, W]
    data = data[:, PRITHVI_BAND_INDICES].float()  # [T, 6, H, W]
    
    means = MEANS.view(1, 6, 1, 1).to(data.device)
    stds  = STDS.view(1, 6, 1, 1).to(data.device)
    data  = (data - means) / stds
    
    frames = []
    for idxs in SEASON_INDICES.values():
        frames.append(data[idxs].mean(dim=0))  # [6, H, W]
    
    return torch.stack(frames, dim=0)  # [4, 6, H, W]

def date_to_doy(d, m, y):
    return datetime(y, m, d).timetuple().tm_yday - 1

# Fixed seasonal temporal coords — same for all patches since we use mean aggregation
# Order: Winter(Jan), Spring(Apr), Summer(Aug), Autumn(Sep)
FIXED_TEMPORAL_COORDS = torch.tensor([
    [2018, date_to_doy(28, 1, 2018)],   # Winter
    [2018, date_to_doy(18, 4, 2018)],   # Spring
    [2018, date_to_doy(26, 8, 2018)],   # Summer
    [2018, date_to_doy(20, 9, 2018)],   # Autumn
], dtype=torch.float32)  # [4, 2]

# %%
from collections import defaultdict
import random

class ChunkedDataset(Dataset):
    def __init__(self, mode, cache_size=8):
        self.mode       = mode
        self.chunks_dir = os.path.join(CHUNKS_DIR, mode)
        self.cache_size = cache_size
        self.index      = []

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

        self._cache       = {}
        self._cache_order = []

    def _load_chunk(self, path):
        if path in self._cache:
            self._cache_order.remove(path)
            self._cache_order.append(path)
            return self._cache[path]

        payload = torch.load(
            os.path.join(self.chunks_dir, path),
            map_location="cpu", weights_only=True
        )
        self._cache[path] = payload
        self._cache_order.append(path)

        if len(self._cache_order) > self.cache_size:
            evict = self._cache_order.pop(0)
            del self._cache[evict]

        return payload

    def shuffle(self):
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

    def _augment(self, data, label):
        """Apply identical random transforms to data [*, H, W] and label [H, W]."""
        if torch.rand(1) < 0.5:
            data  = torch.flip(data,  dims=[-1])
            label = torch.flip(label, dims=[-1])
        if torch.rand(1) < 0.5:
            data  = torch.flip(data,  dims=[-2])
            label = torch.flip(label, dims=[-2])
        if torch.rand(1) < 0.5:
            k     = torch.randint(1, 4, (1,)).item()
            data  = torch.rot90(data,  k, dims=[-2, -1])
            label = torch.rot90(label, k, dims=[-2, -1])
        return data, label

    def _aggregate_seasons(self, data):
        data = data[:, PRITHVI_BAND_INDICES].float()  # [T, 6, H, W]
        
        means = MEANS.view(1, 6, 1, 1).to(data.device)
        stds  = STDS.view(1, 6, 1, 1).to(data.device)
        data  = (data - means) / stds
        
        frames = []
        for idxs in SEASON_INDICES.values():
            frames.append(data[idxs].mean(dim=0))  # [6, H, W]
        
        return torch.stack(frames, dim=1) # [6, 4, H, W]
    
    def _add_indices(self, data):
        data = data / REFLECTANCE_SCALE

        B3  = data[:, 1]
        B4  = data[:, 2]
        B6  = data[:, 4]
        B8  = data[:, 6]
        B11 = data[:, 7]
        B12 = data[:, 9]
        eps = 1e-6

        indices = torch.stack([
            (B8 - B4)  / (B8 + B4  + eps),
            (B8 - B11) / (B8 + B11 + eps),
            (B3 - B8)  / (B3 + B8  + eps),
            (B8 - B6)  / (B8 + B6  + eps),
            (B8 - B12) / (B8 + B12 + eps),
        ], dim=1)

        return torch.cat([data, indices], dim=1)


    def __getitem__(self, idx):
        path, patch_idx = self.index[idx]
        payload  = self._load_chunk(path)

        raw      = payload["data"][patch_idx].to(torch.float32)  # [T, C, H, W]
        label    = payload["label"][patch_idx].long() - 1
        patch_id = payload["patch_ids"][patch_idx]
        latlon   = payload["latlon"][patch_idx]

        if self.mode == 'train':
            raw, label = self._augment(raw, label)

        # ---- Prithvi branch ----
        prithvi_data = self._aggregate_seasons(raw) # [4, 6, H, W]

        # ---- UNet branch ----
        unet_data = self._add_indices(raw) # [T, C+5, H, W]
            
        return {
            "image":           prithvi_data,
            "unet":            unet_data,
            "mask":            label,
            "filename":        patch_id,
            "temporal_coords": FIXED_TEMPORAL_COORDS,
            "location_coords": latlon,
        }
    
class OrdinalSegmentationTask(SemanticSegmentationTask):
    def __init__(self, epochs=40, *args, **kwargs):
        kwargs.pop("task", None)
        super().__init__(*args, **kwargs)
        self.raw_bias = nn.Parameter(torch.zeros(4))
        self.epochs = epochs

        self.save_hyperparameters(ignore=["model", "teacher", "teacher_model"])

    def _to_ordinal_logits(self, features):
        thresholds = torch.cumsum(F.softplus(self.raw_bias), dim=0)
        thresholds = thresholds.view(1, -1, 1, 1)
        return features - thresholds

    def _compute_loss(self, logits, y):
        return ordinal_loss(logits, y)
        
def generate_confusion_matrix(model, val_loader, device):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            data = batch["image"]
            label = batch["mask"]
            data_unet = batch["unet"]
            temporal = batch.get("temporal_coords")
            location = batch.get("location_coords")
            patch_ids = batch.get("filename")

            data, label = data.to(device), label.to(device)
            temporal, location = temporal.to(device), location.to(device)

            output = model(data, temporal_coords=temporal, location_coords=location)
            features = output.output.unsqueeze(1) # [B,1,H,W] because by default model does mask = self._check_for_single_channel_and_squeeze(mask)
            logits = model._to_ordinal_logits(features)
            
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
  
def evaluate_unet(model, loader, device):
    total_loss = 0
    total_correct = 0.0
    total_correct_exact = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            label = batch["mask"]
            data_unet = batch["unet"]
            
            data_unet, label = data_unet.to(device), label.to(device)

            # Unet
            logits = model(data_unet)

            total_loss += ordinal_loss(logits, label).item()
            total_correct += accuracy_pm1(logits, label)
            total_correct_exact += accuracy_exact(logits, label)
            num_batches += 1

    return total_loss / num_batches, total_correct / num_batches, total_correct_exact / num_batches

def evaluate_ensemble(model_prithvi, model_unet, loader, device):
    total_loss = 0
    total_correct = 0.0
    total_correct_exact = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            data = batch["image"].to(device)
            label = batch["mask"].to(device)
            data_unet = batch["unet"].to(device)
            temporal = batch.get("temporal_coords").to(device)
            location = batch.get("location_coords").to(device)
            patch_ids = batch.get("filename")

            # Unet
            logits_u = model_unet(data_unet)

            # Prithvi
            output = model_prithvi.model(data, temporal_coords=temporal, location_coords=location)
            features = output.output.unsqueeze(1)
            logits_p = model_prithvi._to_ordinal_logits(features)

            w = 0.35
            logits = w * logits_p + (1 - w) * logits_u
            
            total_loss += ordinal_loss(logits, label).item()
            total_correct += accuracy_pm1(logits, label)
            total_correct_exact += accuracy_exact(logits, label)
            num_batches += 1

    return total_loss / num_batches, total_correct / num_batches, total_correct_exact / num_batches

def evaluate_prithvi(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0.0
    total_correct_exact = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            data = batch["image"].to(device)
            label = batch["mask"].to(device)
            data_unet = batch["unet"].to(device)
            temporal = batch.get("temporal_coords").to(device)
            location = batch.get("location_coords").to(device)
            patch_ids = batch.get("filename")
            
            if valid_mask(label).sum().item() == 0:
                continue

            output = model(data, temporal_coords=temporal, location_coords=location)
            features = output.output.unsqueeze(1) # [B,1,H,W] because by default model does mask = self._check_for_single_channel_and_squeeze(mask)
            logits = model._to_ordinal_logits(features)
            
            total_loss += ordinal_loss(logits, label).item()
            total_correct += accuracy_pm1(logits, label)
            total_correct_exact += accuracy_exact(logits, label)
            num_batches += 1

    return total_loss / num_batches, total_correct / num_batches, total_correct_exact / num_batches

def generate_submission_ensemble(model_unet, model_prithvi, test_loader, device, w=0.5):
    model_unet.eval()
    model_prithvi.eval()
    
    output_dir = "submissions"
    count = 0

    total_counts = torch.zeros(5, dtype=torch.long)

    with torch.no_grad():
        for batch in test_loader:
            data = batch["image"].to(device)
            data_unet = batch["unet"].to(device)
            temporal = batch.get("temporal_coords").to(device)
            location = batch.get("location_coords").to(device)
            patch_ids = batch.get("filename")

            # Unet
            logits_u = model_unet(data_unet)

            # Prithvi
            output = model_prithvi.model(data, temporal_coords=temporal, location_coords=location)
            features = output.output.unsqueeze(1)
            logits_p = model_prithvi._to_ordinal_logits(features)

            logits = w * logits_p + (1 - w) * logits_u
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
# Check PyTorch and device
print(f"PyTorch version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %%
#### Unet
from unet import ResUNetOrdinal

model_unet = ResUNetOrdinal(
    in_channels   = 15,
    num_timesteps = 34,
    num_classes   = 4,
    base_dim      = 128,
    dropout       = 0.2,
).to(device)

model_unet.load_state_dict(torch.load('final_best_model-v2.pt', map_location=device))

# %%
#### Prithvi

from terratorch.models import EncoderDecoderFactory

model_factory = EncoderDecoderFactory()

model_prithvi = model_factory.build_model(
    task="regression",  # use regression since we apply ordinal loss
    backbone="prithvi_eo_v2_100_tl",
    backbone_pretrained=True,
    backbone_coords_encoding=["time", "location"],
    backbone_num_frames = 4,
    backbone_img_size = 128,
    backbone_bands=[
        "BLUE",
        "GREEN",
        "RED",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2",
    ],
    necks=[
        {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
        {"name": "ReshapeTokensToImage", "effective_time_dim": 4},
        {"name": "LearnedInterpolateToPyramidal"}
    ],
    decoder="UperNetDecoder",
    decoder_channels=128,
    head_dropout=0.1,
    head_channel_list=[128],
    head_final_act=None,
    head_num_outputs=1,
    rescale=True
)

model = OrdinalSegmentationTask.load_from_checkpoint(
    CHECKPOINTS_DIR + "prithvi_final/best5.ckpt",
    model=model_prithvi
)
model.cuda()

val_dataset = ChunkedDataset(mode="val", cache_size=2)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False
)

test_dataset = ChunkedDataset("test", cache_size=2)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

def evaluate(logits, labels):
    acc_exact = accuracy_exact(logits, labels)
    acc_pm1   = accuracy_pm1(logits, labels)
    return acc_exact.item(), acc_pm1.item()

model.eval()
model_unet.eval()

val_loss, val_acc_pm1, val_acc_exact = evaluate_unet(model_unet, val_loader, device)

print(f"UNet - val loss {val_loss:.4f} pm1 {val_acc_pm1:.4f} exact {val_acc_exact:.4f}")

val_loss, val_acc_pm1, val_acc_exact = evaluate_prithvi(model, val_loader, device)

print(f"Prithvi - val loss {val_loss:.4f} pm1 {val_acc_pm1:.4f} exact {val_acc_exact:.4f}")

val_loss, val_acc_pm1, val_acc_exact = evaluate_ensemble(model, model_unet, val_loader, device)

print(f"Ensemble - val loss {val_loss:.4f} pm1 {val_acc_pm1:.4f} exact {val_acc_exact:.4f}")

# tried w = 0.35, 0.1, 0.4
# other things tried: TTA, tuning thresholds
generate_submission_ensemble(model_unet, model, test_loader, device, w=0.35) # best one is w=0.35
