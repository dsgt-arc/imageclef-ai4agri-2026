# %%
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import json
import random
from datetime import datetime
from collections import defaultdict
import random

import terratorch
from terratorch.tasks import SemanticSegmentationTask
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from utils import ordinal_predict, accuracy_exact, accuracy_pm1, valid_mask, ordinal_loss, evaluate, plot_loss_curve, ordinal_confidence, label_to_ordinal

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
        prithvi_data = self._aggregate_seasons(raw)   # [4, 6, H, W]

        # ---- UNet branch ----
        unet_data = self._add_indices(raw)  # [T, C+5, H, W]
            
        return {
            "image":           prithvi_data,
            "unet":            unet_data,
            "mask":            label,
            "filename":        patch_id,
            "temporal_coords": FIXED_TEMPORAL_COORDS,
            "location_coords": latlon,
        }

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
        pl_module.raw_bias.requires_grad = False
        
        print(f"[FreezeBackbone] Backbone frozen. Will unfreeze at epoch {self.unfreeze_epoch}.")

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            for name, param in pl_module.named_parameters():
                if "backbone" in name:
                    param.requires_grad = True
            pl_module.raw_bias.requires_grad = True

            print(f"[FreezeBackbone] Backbone unfrozen at epoch {self.unfreeze_epoch}.")

class MetricsCallback(Callback):
    def __init__(self, save_path="loss_curve.png"):
        self.save_path    = save_path
        self.train_losses = []
        self.val_losses   = []
        self.val_accs_pm1 = []
        self.val_accs_exact = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train/loss")
        self.train_losses.append(loss.item())

        ###
        # Teacher metrics
        ###
        loss_gt = trainer.callback_metrics.get("train/loss_gt")
        loss_pseudo = trainer.callback_metrics.get("train/loss_pseudo")
        pseudo_pct = trainer.callback_metrics.get("train/pseudo_pct")
        conf_mean = trainer.callback_metrics.get("train/conf_mean")
        print(f"  train loss: {loss.item():.4f} | gt: {loss_gt.item():.4f} | pseudo: {loss_pseudo.item():.4f} | pseudo %: {pseudo_pct.item()*100:.2f}% | conf mean: {conf_mean.item():.4f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val/loss")
        acc_pm1  = trainer.callback_metrics.get("val/acc_pm1")
        accuracy_exact = trainer.callback_metrics.get("val/acc_exact")

        self.val_losses.append(loss.item())
        self.val_accs_pm1.append(acc_pm1.item())
        self.val_accs_exact.append(accuracy_exact.item())
        
        print(f"  val loss: {loss.item():.4f} | val acc ±1: {acc_pm1.item():.4f} | val acc exact: {accuracy_exact.item():.4f}")

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

        ax2.plot(self.val_accs_pm1, label="val acc ±1", color="green")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("accuracy")
        ax2.set_title("Val Accuracy ±1")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150)
        plt.close()

class OrdinalSegmentationTask(SemanticSegmentationTask):
    def __init__(self, epochs=40, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.raw_bias = nn.Parameter(torch.zeros(4))

        self.epochs = epochs
        self.HIGH_CONF = 0.7
        self.LAMBDA = 0.3

        self.save_hyperparameters(ignore=["model", "teacher", "teacher_model"])

    def set_teacher(self, teacher_model):
        self.__dict__["teacher"] = teacher_model

        if self.teacher is not None:
            self.teacher.eval()

            for p in self.teacher.parameters():
                p.requires_grad = False

    def on_train_epoch_start(self):
        opt = self.optimizers()
        lrs = [pg["lr"] for pg in opt.param_groups]
        print(f"Epoch {self.current_epoch} | LRs: {lrs}")
        print("Bias: ", self.raw_bias)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in self.named_parameters() if "encoder" in n], "lr": 1e-5},
                {"params": [p for n, p in self.named_parameters() if "encoder" not in n], "lr": 5e-5},
            ],
            weight_decay=1e-2,
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=3
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs - 3,
            eta_min=1e-7,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[3],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _to_ordinal_logits(self, features):
        thresholds = torch.cumsum(F.softplus(self.raw_bias), dim=0)
        thresholds = thresholds.view(1, -1, 1, 1)
        return features - thresholds

    def _compute_loss(self, logits, y):
        return ordinal_loss(logits, y)

    def training_step(self, batch, batch_idx):
        x        = batch["image"]
        y        = batch["mask"]
        x_unet   = batch["unet"]
        temporal = batch.get("temporal_coords")
        location = batch.get("location_coords")

        # -----------------------
        # STUDENT FORWARD
        # -----------------------
        output = self.model(x, temporal_coords=temporal, location_coords=location)
        features = output.output.unsqueeze(1) # [B,1,H,W] because by default model does mask = self._check_for_single_channel_and_squeeze(mask)
        logits = self._to_ordinal_logits(features)
        # -----------------------
        # GT LOSS
        # -----------------------
        loss_gt = self._compute_loss(logits, y)

        # -----------------------
        # TEACHER PSEUDO-LABELS
        # -----------------------
        if self.teacher is not None:
            with torch.no_grad():
                logits_teacher = self.teacher(x_unet)
                preds_teacher = ordinal_predict(logits_teacher)
                conf_teacher  = ordinal_confidence(logits_teacher)

            pseudo_mask = (y == -1) & (conf_teacher > self.HIGH_CONF)

            if pseudo_mask.any():
                pseudo_labels = torch.full_like(y, -1)
                pseudo_labels[pseudo_mask] = preds_teacher[pseudo_mask]

                loss_pseudo = self._compute_loss(logits, pseudo_labels)
                
                if loss_pseudo is not None and not torch.isnan(loss_pseudo):
                    loss = loss_gt + self.LAMBDA * loss_pseudo

                    self.log("train/loss_pseudo", loss_pseudo, on_epoch=True, batch_size=x.shape[0])
                else:
                    print("Invalid loss (NaN or None) encountered in training step.", batch_idx) # Shouldn't happen
                    loss = loss_gt
            else:
                loss = loss_gt

            # metrics
            self.log("train/loss_gt", loss_gt, on_epoch=True, batch_size=x.shape[0])
            self.log("train/pseudo_pct", pseudo_mask.float().mean(), on_epoch=True)
            self.log("train/conf_mean", conf_teacher.mean(), on_epoch=True)
        else:
            loss = loss_gt

        # metrics
        with torch.no_grad():
            acc_pm1 = accuracy_pm1(logits, y)
            acc_exact = accuracy_exact(logits, y)

        self.log("train/loss", loss, prog_bar=True, on_epoch=True, batch_size=x.shape[0])
        self.log("train/acc_pm1",   acc_pm1,   prog_bar=False, batch_size=x.shape[0])
        self.log("train/acc_exact", acc_exact, prog_bar=False, batch_size=x.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        x        = batch["image"]
        y        = batch["mask"]
        temporal = batch.get("temporal_coords")
        location = batch.get("location_coords")

        output = self.model(x, temporal_coords=temporal, location_coords=location)

        features = output.output.unsqueeze(1)  # [B,1,H,W] because by default model does mask = self._check_for_single_channel_and_squeeze(mask)
        logits = self._to_ordinal_logits(features)
        

        with torch.no_grad():
            loss = self._compute_loss(logits, y)

            if loss is None or torch.isnan(loss):
                print("Invalid loss (NaN or None) encountered in validation step. Skipping this batch.", batch_idx)
                return  # skip this batch entirely
        
            acc_pm1 = accuracy_pm1(logits, y)
            acc_exact = accuracy_exact(logits, y)

        self.log("val/loss",      loss,      prog_bar=True,  sync_dist=True, on_epoch=True, on_step=False, batch_size=x.shape[0])
        self.log("val/acc_pm1",   acc_pm1,   prog_bar=True,  sync_dist=True, on_epoch=True, on_step=False, batch_size=x.shape[0])
        self.log("val/acc_exact", acc_exact, prog_bar=True,  sync_dist=True, on_epoch=True, on_step=False, batch_size=x.shape[0])

# Check PyTorch and device
print(f"PyTorch version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

#### Unet
from unet import ResUNetOrdinal

model_unet = ResUNetOrdinal(
    in_channels   = 15,
    num_timesteps = 34,
    num_classes   = 4,
    base_dim      = 128,
    dropout       = 0.2,
).to(device)

model_unet.load_state_dict(torch.load('best_unet.pt', map_location=device))

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

model = OrdinalSegmentationTask(
    epochs = 40,
    model = model_prithvi,
    model_args = dict(
        num_classes = 4,
    ),
    ignore_index = -1, 
    lr           = 2e-4,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=0.01),
    freeze_backbone  = False, 
    freeze_decoder   = False,
    plot_on_val      = False
)

model.set_teacher(model_unet)
# Instead of random, we initialize with the unet raw bias weights.
model.raw_bias.data = model_unet.raw_bias.data.clone()

# ckpt = torch.load(CHECKPOINTS_DIR + "prithvi_final/best.ckpt", map_location="cpu")
# model.load_state_dict(ckpt["state_dict"], strict=True)

# ── callbacks ──────────────────────────────────────────────────────────────
checkpoint_callback = ModelCheckpoint(
    monitor    = "val/loss",
    mode       = "min",
    dirpath    = CHECKPOINTS_DIR + "prithvi_final",
    filename   = "best",
    save_top_k = 1,
    save_last  = False,
)

metrics_callback = MetricsCallback(
    save_path="prithvi_loss_curve.png"
)

frozen_backbone_callback = FreezeBackboneCallback(unfreeze_epoch=7) # 5 or 7

early_stopping_callback = EarlyStopping(monitor="val/loss", patience=10, mode="min")

shuffle_dataset_callback = ShuffleDatasetCallback()

# ── trainer ────────────────────────────────────────────────────────────────
trainer = pl.Trainer(
    accelerator = "gpu",
    devices     = 1,
    precision   = "bf16-mixed",
    max_epochs  = 40,
    callbacks   = [
        shuffle_dataset_callback,
        frozen_backbone_callback, 
        early_stopping_callback,
        checkpoint_callback, 
        metrics_callback
    ],
    log_every_n_steps = 10,
    num_sanity_val_steps=0
)

# ── dataloaders ────────────────────────────────────────────────────────────
train_dataset = ChunkedDataset(mode="train", cache_size=4)
val_dataset   = ChunkedDataset(mode="val", cache_size=2)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=False,
    num_workers=4, prefetch_factor=2,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False
)

trainer.fit(model, train_loader, val_loader)
