import os
import zipfile

import torch
from PIL import Image
from torch.utils.data import DataLoader

from prithvi import ChunkedDataset, build_prithvi_model, OrdinalSegmentationTask
from unet import ResUNetOrdinal
from utils import accuracy_exact, accuracy_pm1, ordinal_predict, ordinal_loss

CHECKPOINTS_DIR = os.path.expandvars('$HOME/scratch/checkpoints/')


def evaluate_unet(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_correct_exact = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            label = batch['mask'].to(device)
            data_unet = batch['unet'].to(device)
            logits = model(data_unet)
            total_loss += ordinal_loss(logits, label).item()
            total_correct += accuracy_pm1(logits, label)
            total_correct_exact += accuracy_exact(logits, label)
            num_batches += 1

    return total_loss / num_batches, total_correct / num_batches, total_correct_exact / num_batches


def evaluate_ensemble(model_prithvi, model_unet, loader, device, w=0.35):
    model_unet.eval()
    model_prithvi.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_correct_exact = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            data = batch['image'].to(device)
            label = batch['mask'].to(device)
            data_unet = batch['unet'].to(device)
            temporal = batch.get('temporal_coords').to(device)
            location = batch.get('location_coords').to(device)

            logits_u = model_unet(data_unet)
            output = model_prithvi.model(data, temporal_coords=temporal, location_coords=location)
            features = output.output.unsqueeze(1)
            logits_p = model_prithvi._to_ordinal_logits(features)
            logits = w * logits_p + (1 - w) * logits_u

            total_loss += ordinal_loss(logits, label).item()
            total_correct += accuracy_pm1(logits, label)
            total_correct_exact += accuracy_exact(logits, label)
            num_batches += 1

    return total_loss / num_batches, total_correct / num_batches, total_correct_exact / num_batches


def generate_submission_ensemble(model_unet, model_prithvi, test_loader, device, w=0.35, output_dir='submissions'):
    model_unet.eval()
    model_prithvi.eval()
    os.makedirs(output_dir, exist_ok=True)
    total_counts = torch.zeros(5, dtype=torch.long)
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            data = batch['image'].to(device)
            data_unet = batch['unet'].to(device)
            temporal = batch.get('temporal_coords').to(device)
            location = batch.get('location_coords').to(device)
            patch_ids = batch.get('filename')

            logits_u = model_unet(data_unet)
            output = model_prithvi.model(data, temporal_coords=temporal, location_coords=location)
            features = output.output.unsqueeze(1)
            logits_p = model_prithvi._to_ordinal_logits(features)
            logits = w * logits_p + (1 - w) * logits_u
            preds = ordinal_predict(logits)

            for class_idx in range(5):
                total_counts[class_idx] += (preds == class_idx).sum().item()

            for pred, pid in zip(preds.cpu().numpy(), patch_ids):
                img = Image.fromarray(pred.astype('uint8'), mode='L')
                img.save(os.path.join(output_dir, f'{pid}.png'))
                count += 1

    zip_path = f'{output_dir}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as handle:
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith('.png'):
                handle.write(os.path.join(output_dir, fname), fname)

    print(f'Saved {count} predictions → {zip_path}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model_unet = ResUNetOrdinal(in_channels=15, num_timesteps=34, num_classes=4, base_dim=128, dropout=0.2).to(device)
    model_unet.load_state_dict(torch.load('final_best_model-v2.pt', map_location=device))

    model_prithvi = build_prithvi_model()
    model = OrdinalSegmentationTask.load_from_checkpoint(CHECKPOINTS_DIR + 'prithvi_final/best5.ckpt', model=model_prithvi)
    model.to(device)

    val_dataset = ChunkedDataset(mode='val', cache_size=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataset = ChunkedDataset(mode='test', cache_size=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    val_loss, val_acc_pm1, val_acc_exact = evaluate_unet(model_unet, val_loader, device)
    print(f'UNet - val loss {val_loss:.4f} pm1 {val_acc_pm1:.4f} exact {val_acc_exact:.4f}')

    val_loss, val_acc_pm1, val_acc_exact = evaluate_ensemble(model, model_unet, val_loader, device, w=0.35)
    print(f'Ensemble - val loss {val_loss:.4f} pm1 {val_acc_pm1:.4f} exact {val_acc_exact:.4f}')

    generate_submission_ensemble(model_unet, model, test_loader, device, w=0.35)


if __name__ == '__main__':
    main()
