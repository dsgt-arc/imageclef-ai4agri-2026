import torch
import torch.nn as nn
import os
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import torch.nn.functional as F
import json
from datetime import datetime

import terratorch
from terratorch.tasks import SemanticSegmentationTask
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, EarlyStopping, ModelCheckpoint

class ShuffleDatasetCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.train_dataloader.dataset.shuffle()

class FreezeBackboneCallback(Callback):
    def __init__(self, unfreeze_epoch: int = 5):
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_start(self, trainer, pl_module):
        # Freeze backbone at the start
        for name, param in pl_module.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
        print(f"[FreezeBackbone] Backbone frozen. Will unfreeze at epoch {self.unfreeze_epoch}.")

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            for name, param in pl_module.named_parameters():
                if "backbone" in name:
                    param.requires_grad = True
            print(f"[FreezeBackbone] Backbone unfrozen at epoch {self.unfreeze_epoch}.")


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
# prithvi:   B2, B3, B4,                B8A, B11, B12
PRITHVI_BAND_INDICES = [0, 1, 2, 7, 8, 9]

MEANS = torch.tensor([494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962, 1739.579917])
STDS  = torch.tensor([284.925432, 357.84876,  575.566823, 896.601013,  951.900334,  921.407808])

LOSS_TYPE = "bce"  # "bce" or "ce"

def ordinal_to_label(logits, thresholds=None):
    probs = torch.sigmoid(logits)  # [B,4,H,W]

    if thresholds is None:
        thresholds = [0.5] * probs.shape[1]

    thresholds = torch.tensor(
        thresholds, device=probs.device
    ).view(1, -1, 1, 1)

    return (probs > thresholds).sum(dim=1)

def predictions_from_logits(logits):
    if LOSS_TYPE == "bce":
        return ordinal_to_label(logits)
    else:
        return logits.argmax(dim=1)

def loss_fn_bce(logits, targets, num_classes=5):
    B, C, H, W = logits.shape  # C should be num_classes - 1

    # [B, C, H, W] → [B*H*W, C]
    logits  = logits.permute(0, 2, 3, 1).reshape(-1, C)
    targets = targets.reshape(-1)

    # mask unlabeled
    mask = targets >= 0
    if mask.sum() == 0:
        return None

    logits  = logits[mask]
    targets = targets[mask]

    # build ordinal targets
    thresholds = torch.arange(num_classes - 1, device=targets.device)
    ordinal_targets = (targets.unsqueeze(1) > thresholds).float()

    loss = F.binary_cross_entropy_with_logits(
        logits, ordinal_targets, reduction='none'
    )  # [N_pixels, C]

    loss = loss.mean(dim=1)  # [N_pixels]

    return loss.mean()

def loss_fn_ce(logits, targets):
    return F.cross_entropy(logits, targets, ignore_index=-1, reduction='mean')

class OrdinalSegmentationTask(SemanticSegmentationTask):
    def on_train_epoch_start(self):
        opt = self.optimizers()
        lrs = [pg["lr"] for pg in opt.param_groups]
        print(f"Epoch {self.current_epoch} | LRs: {lrs}")

    def _compute_loss(self, logits, y):
        if LOSS_TYPE == "bce":
            loss = loss_fn_bce(logits, y)
        else:
            loss = loss_fn_ce(logits, y)
        if loss is None:
            return None, 0
        return loss, 1

    def training_step(self, batch, batch_idx):
        x        = batch["image"]
        y        = batch["mask"]
        temporal = batch.get("temporal_coords")
        location = batch.get("location_coords")

        output = self.model(x, temporal_coords=temporal, location_coords=location)
        logits = output.output

        loss, n = self._compute_loss(logits, y)
        if loss is None:
            return None

        self.log("train/loss", loss, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x        = batch["image"]
        y        = batch["mask"]
        temporal = batch.get("temporal_coords")
        location = batch.get("location_coords")

        output = self.model(x, temporal_coords=temporal, location_coords=location)
        logits = output.output

        loss, n = self._compute_loss(logits, y)
        if loss is None:
            return

        preds = predictions_from_logits(logits)

        valid_mask = y >= 0
        y_valid    = y[valid_mask]
        p_valid    = preds[valid_mask]

        diff      = torch.abs(p_valid.long() - y_valid.long())
        acc_pm1   = (diff <= 1).float().mean()
        acc_exact = (diff == 0).float().mean()

        self.log("val/loss",      loss,      prog_bar=True, sync_dist=True, on_epoch=True, on_step=False, batch_size=x.shape[0])
        self.log("val/acc_pm1",   acc_pm1,   prog_bar=True, sync_dist=True, on_epoch=True, on_step=False, batch_size=x.shape[0])
        self.log("val/acc_exact", acc_exact, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False, batch_size=x.shape[0])
        
def aggregate_seasons(data):
    data = data[:, PRITHVI_BAND_INDICES].float()  # [T, 6, H, W]
    
    means = MEANS.view(1, 6, 1, 1).to(data.device)
    stds  = STDS.view(1, 6, 1, 1).to(data.device)
    data  = (data - means) / stds
    
    frames = []
    for idxs in SEASON_INDICES.values():
        frames.append(data[idxs].mean(dim=0))
    return torch.stack(frames, dim=0) # [4, 6, H, W]

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

        data     = payload["data"][patch_idx].to(torch.float32)
        label    = payload["label"][patch_idx]
        patch_id = payload["patch_ids"][patch_idx]
        latlon   = payload["latlon"][patch_idx]

        data  = aggregate_seasons(data)
        label = label.long() - 1

        return data, label, patch_id, FIXED_TEMPORAL_COORDS, latlon

class LossCurveCallback(Callback):
    def __init__(self, save_path="loss_curve.png"):
        self.save_path    = save_path
        self.train_losses = []
        self.val_losses   = []
        self.val_accs     = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train/loss")
        if loss is not None:
            self.train_losses.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val/loss")
        acc  = trainer.callback_metrics.get("val/acc_pm1")

        if loss is not None:
            self.val_losses.append(loss.item())
        if acc is not None:
            self.val_accs.append(acc.item())
            print(f"  val loss: {loss.item():.4f} | val acc ±1: {acc.item():.4f}")

        self._plot()

    def _plot(self):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.train_losses, label="train loss")
        ax1.plot(self.val_losses,   label="val loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.val_accs, label="val acc ±1", color="green")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("accuracy")
        ax2.set_title("Val Accuracy ±1")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150)
        plt.close()
        
if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── model ──────────────────────────────────────────────────────────────────
    model = OrdinalSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=dict(
            backbone = "prithvi_eo_v2_100_tl",
            backbone_pretrained = True,
            backbone_bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            backbone_coords_encoding=["time", "location"],
            backbone_num_frames = 4,
            backbone_img_size = 128,
            necks = [
                {"name": "SelectIndices", "indices": [2, 5, 8, 11]}, # [5, 11, 17, 23] for 300
                {"name": "ReshapeTokensToImage", "effective_time_dim": 4},
            ],
            decoder = "UperNetDecoder",
            decoder_channels = 128,
            decoder_scale_modules = True,
            num_classes = 4,
            head_dropout = 0.1,
            rescale = True,
        ),
        ignore_index = -1,
        lr = 1e-5,
        optimizer="AdamW",
        optimizer_hparams=dict(weight_decay=0.1),
        freeze_backbone  = False,
        freeze_decoder   = False,
        plot_on_val      = False,
    )

    # ── dataloaders ────────────────────────────────────────────────────────────
    train_dataset = ChunkedDataset(mode="train", cache_size=4)
    val_dataset   = ChunkedDataset(mode="val", cache_size=2)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=False,
        num_workers=2, prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    # ── callbacks ──────────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        monitor    = "val/acc_exact",
        mode       = "max",
        dirpath    = CHECKPOINTS_DIR + "prithvi",
        filename   = "best",          # fixed name so it always overwrites
        save_top_k = 1,               # only keep the single best
        save_last  = False,           # don't save a separate 'last.ckpt'
    )

    loss_curve_callback = LossCurveCallback(
        save_path="./prithvi/prithvi_loss_curve.png"
    )

    # ── trainer ────────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator = "gpu",
        devices     = 1,
        precision   = "bf16-mixed",
        max_epochs  = 30,
        callbacks   = [
            FreezeBackboneCallback(unfreeze_epoch=5), 
            EarlyStopping(monitor="val/loss", patience=10, mode="min"),
            checkpoint_callback, 
            ShuffleDatasetCallback(), 
            loss_curve_callback
        ],
        log_every_n_steps = 10,
    )

    trainer.fit(model, train_loader, val_loader)
