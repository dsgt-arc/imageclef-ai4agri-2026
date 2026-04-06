import lightning.pytorch as pl
import torch
import torch.nn as nn
from terratorch.models import PixelwiseModel

def pm1_accuracy(pred, target, ignore_index=0):
    mask = target != ignore_index
    if not torch.any(mask):
        return 0.0
    err = (pred[mask] - target[mask]).abs()
    return (err <= 1).float().mean().item()

def ordinal_loss_step(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 0
) -> torch.Tensor:
    """Computes binary cross entropy over K-1 cumulative thresholds."""
    mask = target != ignore_index
    if mask.sum() == 0:
        return logits.sum() * 0.0

    # target is [1, 5], so target - 2 creates the thresholds offset
    ordinal_targets = torch.zeros(
        (target.size(0), target.size(1), target.size(2), logits.size(1)),
        dtype=torch.float32,
        device=target.device,
    )
    for k in range(logits.size(1)):
        ordinal_targets[:, :, :, k] = (target >= (k + 2)).float()

    ordinal_targets = ordinal_targets.permute(0, 3, 1, 2)  # (B, K-1, H, W)
    mask_expanded = mask.unsqueeze(1).expand_as(logits)

    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask_expanded], ordinal_targets[mask_expanded]
    )

class PrithviLightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # We construct PixelwiseModel from TerraTorch. 
        # By default we map our (B, T, C, H, W) to (B, C, T, H, W).
        # We will process in chunks of 4 frames or 2 frames to avoid OOM or 
        # position embedding mismatches if the backbone expects a specific num_frames.
        # prithvi_eo_v2_300 expects 6 bands.
        self.frames_per_chunk = 2
        
        self.model = PixelwiseModel(
            backbone=cfg.backbone,
            backbone_pretrained=True, # Automatically fetch from huggingface
            backbone_in_channels=6,
            backbone_num_frames=self.frames_per_chunk,
            num_classes=cfg.num_classes - 1, # K-1 for ordinal
            head_params={"in_channels": 1024, "decoder_channels": 256, "head_dropout": 0.2}, 
            # 1024 is standard for 300M parameters, using a FPN or UperNet head.
            # Using FC head is default in terratorch for some, we let PixelwiseModel infer.
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        # We will pad T=34 to T=34 or whatever to be divisible by frames_per_chunk
        B, T, C, H, W = x.shape
        padded_T = ((T + self.frames_per_chunk - 1) // self.frames_per_chunk) * self.frames_per_chunk
        
        if padded_T != T:
            # Pad by repeating the last frame
            pad_frames = padded_T - T
            last_frame = x[:, -1:, :, :, :].repeat(1, pad_frames, 1, 1, 1)
            x_padded = torch.cat([x, last_frame], dim=1)
        else:
            x_padded = x
            
        chunks = x_padded.chunk(padded_T // self.frames_per_chunk, dim=1)
        logits_list = []
        for chunk in chunks:
            # chunk: (B, T_chunk, C, H, W) -> we need (B, C, T_chunk, H, W) for TerraTorch
            chunk_input = chunk.permute(0, 2, 1, 3, 4)
            logits = self.model(chunk_input) # (B, K-1, H, W)
            logits_list.append(logits)
            
        # Average logits over temporal chunks
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return avg_logits

    def training_step(self, batch, batch_idx):
        x, y, _doys = batch
        logits = self(x)
        loss = ordinal_loss_step(logits, y, ignore_index=self.cfg.ignore_index)
        
        # log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _doys = batch
        logits = self(x)
        loss = ordinal_loss_step(logits, y, ignore_index=self.cfg.ignore_index)
        
        preds = (logits.sigmoid() > 0.5).sum(dim=1) + 1
        acc_pm1 = pm1_accuracy(preds, y, self.cfg.ignore_index)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_pm1", acc_pm1, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # We can implement a separate learning rate for backbone vs head
        head_params = []
        backbone_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.cfg.lr * 0.1},
                {"params": head_params, "lr": self.cfg.lr},
            ],
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
