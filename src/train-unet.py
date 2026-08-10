import argparse
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset

from unet import ResUNetOrdinal
from utils import accuracy_pm1, evaluate, loss_fn, plot_loss_curve

REFLECTANCE_SCALE = 10_000.0
CHUNKS_DIR = os.path.expandvars('$HOME/scratch/precomputed_tensors/')


class ChunkedDataset(Dataset):
    def __init__(self, mode, add_indices=True, cache_size=8):
        self.mode = mode
        self.chunks_dir = os.path.join(CHUNKS_DIR, mode)
        self.add_indices = add_indices
        self.cache_size = cache_size
        self.index = []

        for fname in sorted(f for f in os.listdir(self.chunks_dir) if f.endswith('.pt')):
            payload = torch.load(os.path.join(self.chunks_dir, fname), map_location='cpu', weights_only=True)
            self.index.extend((fname, i) for i in range(payload['data'].shape[0]))

    def __len__(self):
        return len(self.index)

    def shuffle(self):
        import random
        random.shuffle(self.index)

    def __getitem__(self, idx):
        path, patch_idx = self.index[idx]
        payload = torch.load(os.path.join(self.chunks_dir, path), map_location='cpu', weights_only=True)
        data = payload['data'][patch_idx].to(torch.float32)
        label = payload['label'][patch_idx]
        data = data / REFLECTANCE_SCALE
        return data, label, payload['patch_ids'][patch_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    train_dataset = ChunkedDataset('train', cache_size=6)
    val_dataset = ChunkedDataset('val', cache_size=2)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, persistent_workers=True, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = ResUNetOrdinal(in_channels=13, num_timesteps=34, num_classes=4, base_dim=64, dropout=0.2).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 30
    no_improve = 0

    for epoch in range(args.epochs):
        train_dataset.shuffle()
        model.train()
        total_loss = 0.0
        total_pixels = 0
        total_acc = 0.0

        for data, label, _ in train_loader:
            data, label = data.to(args.device), label.to(args.device)
            k = random.randint(0, 3)
            hflip = random.random() > 0.5
            if k > 0:
                data = torch.rot90(data, k, [-2, -1])
                label = torch.rot90(label, k, [-2, -1])
            if hflip:
                data = torch.flip(data, [-1])
                label = torch.flip(label, [-1])

            optimizer.zero_grad()
            logits = model(data)
            loss_sum, n = loss_fn(logits, label)
            if n == 0:
                continue
            (loss_sum / n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_sum.item()
            acc, _ = accuracy_pm1(logits, label)
            total_acc += acc
            total_pixels += n

        train_loss = total_loss / total_pixels
        train_acc = total_acc / total_pixels
        val_loss, val_acc, val_acc_exact = evaluate(model, val_loader, args.device)
        print(f'Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} | exact acc {val_acc_exact:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_unet.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    plot_loss_curve(train_losses, val_losses, 'unet_loss_curve.png')


if __name__ == '__main__':
    main()
