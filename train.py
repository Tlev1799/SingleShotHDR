import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from datasets.hdr_dataset import HDRDataset
from optics.lens import LearnableLens, l2_laplacian_regularizer
from models.deconv_net import ReconNet
from losses.l2_gamma import l2_gamma_batch
from utils.image_ops import psf_convolve_rgb, save_exr, save_png, to_numpy_img
from optics.sensor import sensor_model, simulate_sensor_capture

import numpy as np
import glob
import re

import os
os.chdir('/home/projects/sipl-prj10826/DeepOpticsHDR_PyTorch')

# TODO: Add hdrplus dataset citation to the project report. https://www.hdrplusdata.org/dataset.html

global gt_image_singelton
gt_image_singelton = True

def evaluate(lens, cnn, loader, cfg, epoch):
    global gt_image_singelton
    lens.eval()
    cnn.eval()

    lens.is_training = False # So height map noise is not added.

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
            x_in = simulate_sensor_capture(blurred_linear, min_val=cfg.hdr_min_val) # torch.clamp(blurred_linear + 1e-8, cfg.hdr_min_val, 1.0)
            restored_hdr = cnn(x_in)

            loss = l2_gamma_batch(restored_hdr, hdr)
            total_loss += loss.item()

            if first_save:
                first_save = False
                index = 0 # random.randint(0, hdr.shape[0]-1)
                # import ipdb; ipdb.set_trace()

                hdr_np = np.squeeze(to_numpy_img(hdr[index]))
                blur_np = np.squeeze(to_numpy_img(x_in[index]))
                rest_np = np.squeeze(to_numpy_img(restored_hdr[index]))

                # Gamma correction
                hdr_np = np.power(np.maximum(hdr_np, 0.0), 0.5)
                blur_np = np.power(np.maximum(blur_np, 0.0), 0.5)
                rest_np = np.power(np.maximum(rest_np, 0.0), 0.5)

                out_dir = cfg.images_checkpoint
                os.makedirs(out_dir, exist_ok=True)

                # PNG (display-friendly) with exposure
                # save_png(os.path.join(out_dir, f"epoch{epoch}_gt_exp3.png"), hdr_np, -3)
                # save_png(os.path.join(out_dir, f"epoch{epoch}_blurred_exp3.png"), blur_np, -3)
                # save_png(os.path.join(out_dir, f"epoch{epoch}_restored_exp3.png"), rest_np, -3)

                # PNG (display-friendly) without exposure
                if gt_image_singelton:
                    save_png(os.path.join(out_dir, f"epoch{epoch}_gt.png"), hdr_np)
                    gt_image_singelton = False
                save_png(os.path.join(out_dir, f"epoch{epoch}_blurred.png"), blur_np)
                save_png(os.path.join(out_dir, f"epoch{epoch}_resotred.png"), rest_np)

                # EXR (true HDR linear data)
                # save_exr(os.path.join(out_dir, f"epoch{epoch}_gt.exr"), hdr_np)
                # save_exr(os.path.join(out_dir, f"epoch{epoch}_blurred.exr"), blur_np)
                # save_exr(os.path.join(out_dir, f"epoch{epoch}_restored.exr"), rest_np)

    lens.train()
    cnn.train()
    lens.is_training = True

    return total_loss / len(loader)

def save_checkpoint(save_path, epoch, cnn, lens, optimizer, scheduler, val_loss=None):
    path = os.path.join(save_path, f"ckpt_epoch_{epoch:03d}.pt")

    torch.save({
        "epoch": epoch,
        "cnn": cnn.state_dict(),
        "lens": lens.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss": val_loss
    }, path)

    print(f"[CHECKPOINT] saved: {path}")

def load_checkpoint(path, cnn, lens, optimizer, scheduler):
    ckpt = torch.load(path)

    cnn.load_state_dict(ckpt["cnn"])
    lens.load_state_dict(ckpt["lens"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt["epoch"], ckpt.get("val_loss", None)

cfg = Config()

# os.makedirs(cfg.cnn_checkpoint, exist_ok=True)
os.makedirs(cfg.optics_checkpoint, exist_ok=True)
os.makedirs(cfg.debug_checkpoint, exist_ok=True)

dataset = HDRDataset(cfg.data_root)

train_len = int(cfg.data_split * len(dataset))
val_len = int(len(dataset) - train_len)

# import ipdb; ipdb.set_trace()

train_set, val_set = random_split(
    dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

lens = LearnableLens(
    cfg.lens_resolution,
    cfg.wavelengths,
    cfg.refractive_indices,
    cfg.sub_pixel_pitch,
    cfg.focal_distance,
    cfg.height_map_noise,
).to(cfg.device)

cnn = ReconNet().to(cfg.device)

optimizer = torch.optim.Adam([
    {
        "params": cnn.parameters(),
        "lr": cfg.lr_cnn
    },
    {
        "params": lens.parameters(),
        "lr": cfg.lr_cnn * cfg.psf_lr_multiplier
    }
], betas=(0.9, 0.999), eps=1e-8)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.99
)

# TODO: This scheduler did not work as well for the full training set, try the original one again. 

# NOTE: GEMINI suggested scheduler
# scheduler = ReduceLROnPlateau(
#     optimizer, 
#     mode='min',       # We want to minimize loss
#     factor=0.2,       # Multiply LR by 0.2 when stalling (e.g., 1e-4 -> 2e-5)
#     patience=4,       # Wait 4 epochs of no improvement before dropping
#     threshold=1e-4,   # Minimum change to qualify as an improvement
#     min_lr=1e-6       # Don't let the learning rate drop below this floor
# )

# ==============================================================================
# AUTOMATIC CHECKPOINT RESUMPTION LOGIC
# ==============================================================================
checkpoint_dir = cfg.check_point_dir
os.makedirs(checkpoint_dir, exist_ok=True)

# Scan for any checkpoint files matching your saving pattern
checkpoint_pattern = os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")
checkpoint_files = glob.glob(checkpoint_pattern)

start_epoch = 0
if checkpoint_files:
    # Helper to extract the integer epoch number from the filename string
    # e.g., 'checkpoints/ckpt_epoch_073.pt' -> 73
    def get_epoch_num(filepath):
        match = re.search(r"epoch_?(\d+)", os.path.basename(filepath))
        return int(match.group(1)) if match else -1

    # Sort files naturally by epoch number to guarantee the true latest file is last
    checkpoint_files.sort(key=get_epoch_num)
    latest_checkpoint_path = checkpoint_files[-1]

    print("-" * 60)
    print("Automatic Resume Triggered!")
    print(f"Loading latest checkpoint file: {latest_checkpoint_path}")

    checkpoint_epoch, last_val_loss = load_checkpoint(latest_checkpoint_path, cnn, lens, optimizer, scheduler)
    start_epoch = checkpoint_epoch + 1

    print(f"Resuming pipeline execution from Epoch {start_epoch}, which had val_loss: {last_val_loss}")
    print("-" * 60)

else:
    print("-" * 60)
    print(f"No existing checkpoints found in '{checkpoint_dir}'.")
    print("Starting a completely fresh training run from Epoch 0.")
    print("-" * 60)


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
for epoch in range(start_epoch, cfg.epochs):

    cnn.train()
    lens.train()

    for i, hdr in enumerate(train_loader):

        hdr = hdr.to(cfg.device)
        
        psfs_hr = lens() # calls forward() of lens torch module. Returns shape (3, H, W)
        psfs = sensor_model(
            psfs_hr,
            sampling_factor=cfg.sampling_factor,
            patch_size=cfg.image_size
        )

        # Clip hdr before convolving
        clipped_hdr = torch.clamp(hdr, cfg.hdr_min_val, cfg.hdr_max_val)
        blurred_linear = psf_convolve_rgb(clipped_hdr, psfs)

        # Sensor Capture Simulation: Clamp to LDR [0, 1]
        x_in = simulate_sensor_capture(blurred_linear, min_val=cfg.hdr_min_val) # torch.clamp(blurred_linear + 1e-8, cfg.hdr_min_val, 1.0)

        # Process LDR image through the deconvolution network
        restored_hdr = cnn(x_in)

        # Main training loss
        train_loss = l2_gamma_batch(restored_hdr, hdr)

        # Regularization loss
        height_map_reg_loss = l2_laplacian_regularizer(lens.height)
        height_map_scaled_loss = cfg.lambda_height_map * height_map_reg_loss

        # Final loss
        loss = train_loss + height_map_scaled_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Iteration number {i}, l2_loss: {train_loss:.5f}, height map scaled loss: {height_map_scaled_loss:.5f}, total loss: {loss:.5f}")


    val_loss = evaluate(lens, cnn, val_loader, cfg, epoch)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"epoch={epoch} | train_loss={loss.item():.5f} | val_loss={val_loss:.5f} | lr={current_lr}")

    # scheduler.step(val_loss)
    scheduler.step()

    save_checkpoint(cfg.check_point_dir, epoch, cnn, lens, optimizer, scheduler, val_loss)
    if epoch % 5 == 0:
        torch.save(lens.sqrt_height_map.detach().cpu(), f"{cfg.optics_checkpoint}/lens_height_map_epoch{epoch:03d}.pt") # Save height map
        torch.save(lens().detach().cpu(), f"{cfg.debug_checkpoint}/psf_snapshot_epoch{epoch:03d}.pt") # save PSF