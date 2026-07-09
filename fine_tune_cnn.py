import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import glob
import re
import os

from config import Config
from datasets.hdr_dataset import HDRDataset
from optics.lens import LearnableLens, l2_laplacian_regularizer
from models.deconv_net import ReconNet
from losses.l2_gamma import l2_gamma_batch
from utils.image_ops import psf_convolve_rgb, save_png, to_numpy_img
from optics.sensor import sensor_model, simulate_sensor_capture

os.chdir('/home/projects/sipl-prj10826/DeepOpticsHDR_PyTorch')

global gt_image_singelton
gt_image_singelton = True

def evaluate(lens, cnn, loader, cfg, epoch):
    global gt_image_singelton
    lens.eval()
    cnn.eval()
    lens.is_training = False 

    total_loss = 0.0
    first_save = True

    with torch.no_grad():
        for hdr in loader:
            hdr = hdr.to(cfg.device)

            psfs_hr = lens()
            psfs = sensor_model(
                psfs_hr,
                sampling_factor=cfg.sampling_factor,
                patch_size=cfg.image_size
            )

            clipped_hdr = torch.clamp(hdr, cfg.hdr_min_val, cfg.hdr_max_val)
            blurred_linear = psf_convolve_rgb(clipped_hdr, psfs)
            x_in = simulate_sensor_capture(blurred_linear, min_val=cfg.hdr_min_val)
            restored_hdr = cnn(x_in)

            loss = l2_gamma_batch(restored_hdr, hdr)
            total_loss += loss.item()

            if first_save:
                first_save = False
                index = 0

                hdr_np = np.squeeze(to_numpy_img(hdr[index]))
                blur_np = np.squeeze(to_numpy_img(x_in[index]))
                rest_np = np.squeeze(to_numpy_img(restored_hdr[index]))

                hdr_np = np.power(np.maximum(hdr_np, 0.0), 0.5)
                blur_np = np.power(np.maximum(blur_np, 0.0), 0.5)
                rest_np = np.power(np.maximum(rest_np, 0.0), 0.5)

                out_dir = os.path.join(cfg.images_checkpoint, "fine_tune")
                os.makedirs(out_dir, exist_ok=True)

                if gt_image_singelton:
                    save_png(os.path.join(out_dir, f"epoch{epoch}_gt.png"), hdr_np)
                    gt_image_singelton = False
                save_png(os.path.join(out_dir, f"epoch{epoch}_blurred.png"), blur_np)
                save_png(os.path.join(out_dir, f"epoch{epoch}_resotred.png"), rest_np)

    cnn.train()
    # Note: Lens stays eval/frozen during fine-tuning, but keeping structure consistent
    return total_loss / len(loader)

def save_checkpoint(save_path, epoch, cnn, lens, optimizer, scheduler, val_loss=None):
    path = os.path.join(save_path, f"ft_ckpt_epoch_{epoch:03d}.pt")
    torch.save({
        "epoch": epoch,
        "cnn": cnn.state_dict(),
        "lens": lens.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss": val_loss
    }, path)
    print(f"[FINE-TUNE CHECKPOINT] saved: {path}")

# Load Configuration
cfg = Config()

os.makedirs(cfg.cnn_checkpoint, exist_ok=True)

# Prepare datasets and loaders
dataset = HDRDataset(cfg.data_root)
train_len = int(cfg.data_split * len(dataset))
val_len = int(len(dataset) - train_len)

train_set, val_set = random_split(
    dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Initialize modules 
# EXPLICITLY passing 0.02 for the sensor noise as requested
lens = LearnableLens(
    cfg.lens_resolution,
    cfg.wavelengths,
    cfg.refractive_indices,
    cfg.sub_pixel_pitch,
    cfg.focal_distance,
    0, # Height mao noise is 0 because we have already fixed the optics. 
).to(cfg.device)

cnn = ReconNet().to(cfg.device)

# --- LOAD LATEST JOINT-TRAINING CHECKPOINT ---
checkpoint_dir = cfg.check_point_dir
checkpoint_pattern = os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")
checkpoint_files = glob.glob(checkpoint_pattern)

if not checkpoint_files:
    raise FileNotFoundError(f"No joint-training checkpoints found in '{checkpoint_dir}' to fine-tune from.")

def get_epoch_num(filepath):
    match = re.search(r"epoch_?(\d+)", os.path.basename(filepath))
    return int(match.group(1)) if match else -1

checkpoint_files.sort(key=get_epoch_num)
latest_checkpoint_path = checkpoint_files[-1]

print("-" * 60)
print(f"Loading base checkpoint for fine-tuning: {latest_checkpoint_path}")
ckpt = torch.load(latest_checkpoint_path, map_location=cfg.device)
cnn.load_state_dict(ckpt["cnn"])
lens.load_state_dict(ckpt["lens"])
start_epoch = ckpt["epoch"] + 1
print(f"Loaded successfully. Commencing fine-tuning from Epoch {start_epoch}")
print("-" * 60)

# --- FREEZE LENS HEIGHT MAP ---
for param in lens.parameters():
    param.requires_grad = False
lens.eval() 
lens.is_training = False # Keeps height map noise behavior deterministic / frozen if used in forward

# --- OPTIMIZER & SCHEDULER FOR CNN ONLY ---
# Reducing the learning rate for fine-tuning (e.g., original lr divided by 10)
fine_tune_lr = cfg.lr_cnn * 0.1 

optimizer = torch.optim.Adam([
    {"params": cnn.parameters(), "lr": fine_tune_lr}
], betas=(0.9, 0.999), eps=1e-8)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.99
)

# --- FINE-TUNING LOOP ---
# Fine-tune for a designated number of epochs (e.g., adding 20 epochs past start_epoch)
fine_tune_epochs = start_epoch + 100 

for epoch in range(start_epoch, fine_tune_epochs):
    cnn.train()

    for i, hdr in enumerate(train_loader):
        hdr = hdr.to(cfg.device)
        
        # Static Lens Inference
        with torch.no_grad():
            psfs_hr = lens() 
            psfs = sensor_model(
                psfs_hr,
                sampling_factor=cfg.sampling_factor,
                patch_size=cfg.image_size
            )
            clipped_hdr = torch.clamp(hdr, cfg.hdr_min_val, cfg.hdr_max_val)
            blurred_linear = psf_convolve_rgb(clipped_hdr, psfs)
            x_in = simulate_sensor_capture(blurred_linear, noise_std=0.02, min_val=cfg.hdr_min_val)

        # Optimize CNN only
        restored_hdr = cnn(x_in)
        loss = l2_gamma_batch(restored_hdr, hdr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"[FT] Epoch {epoch} | Iter {i} | l2_loss: {loss.item():.5f}")

    val_loss = evaluate(lens, cnn, val_loader, cfg, epoch)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"==> FT Epoch={epoch} | val_loss={val_loss:.5f} | lr={current_lr}")

    scheduler.step()
    save_checkpoint(cfg.cnn_checkpoint, epoch, cnn, lens, optimizer, scheduler, val_loss)