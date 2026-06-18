import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ExponentialLR

from config import Config
from datasets.hdr_dataset import HDRDataset
from optics.lens import LearnableLens, l2_laplacian_regularizer
from models.deconv_net import ReconNet
from losses.l2_gamma import l2_gamma_batch
from utils.image_ops import psf_convolve_rgb, save_exr, save_png, to_numpy_img, get_final_image
from optics.sensor import sensor_model

import numpy as np
import random

import os
os.chdir('/home/projects/sipl-prj10826/DeepOpticsHDR_PyTorch')

def evaluate(lens, cnn, loader, cfg, epoch):
    lens.eval()
    cnn.eval()

    lens.is_training = False # So height map noise is not added.

    total_loss = 0.0

    first_save = True

    # TODO:
    # CNN seems to be identical to original paper, problem might be in the handling of optics, psf calculation and that direction,
    # Possible issue is also just the display method but I doubt it. More likely its a mismatch between their code and ours.

    with torch.no_grad():
        for hdr in loader:

            hdr = hdr.to(cfg.device)

            psfs_hr = lens()
            psfs = sensor_model(
                psfs_hr,
                sampling_factor=cfg.sampling_factor,
                patch_size=cfg.image_size
            )

            blurred = psf_convolve_rgb(hdr, psfs)
            restored = cnn(blurred)

            loss = l2_gamma_batch(restored, hdr)
            total_loss += loss.item()

            if first_save:
                first_save = False
                index = random.randint(0, hdr.shape[0]-1)
                import ipdb; ipdb.set_trace()

                hdr_np = np.squeeze(to_numpy_img(hdr[index]))
                blur_np = np.squeeze(to_numpy_img(blurred[index]))
                rest_np = np.squeeze(to_numpy_img(restored[index]))

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
                save_png(os.path.join(out_dir, f"epoch{epoch}_gt.png"), hdr_np)
                save_png(os.path.join(out_dir, f"epoch{epoch}_blurred.png"), blur_np)
                save_png(os.path.join(out_dir, f"epoch{epoch}_resotred.png"), rest_np)

                # EXR (true HDR linear data)
                save_exr(os.path.join(out_dir, f"epoch{epoch}_gt.exr"), hdr_np)
                save_exr(os.path.join(out_dir, f"epoch{epoch}_blurred.exr"), blur_np)
                save_exr(os.path.join(out_dir, f"epoch{epoch}_restored.exr"), rest_np)

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
    cfg.pixel_pitch,
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

start_epoch = 0
if cfg.should_resume:
    start_epoch, _ = load_checkpoint(f"{cfg.check_point_dir}/ckpt_epoch_018.pt", cnn, lens, optimizer, scheduler)
    start_epoch += 1

for epoch in range(start_epoch, cfg.epochs):

    for i, hdr in enumerate(train_loader):

        hdr = hdr.to(cfg.device)
        
        psfs_hr = lens().to(cfg.device) # calls forward() of lens torch module. Returns shape (3, H, W)
        psfs = sensor_model(
            psfs_hr,
            sampling_factor=cfg.sampling_factor,
            patch_size=cfg.image_size
        )

        blurred = psf_convolve_rgb(hdr, psfs)
        restored = cnn(blurred)

        # Main training loss
        train_loss = l2_gamma_batch(restored, hdr)

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

    # Update learning rates
    scheduler.step()

    val_loss = evaluate(lens, cnn, val_loader, cfg, epoch)
    print(f"epoch={epoch} train_loss={loss.item():.5f} val_loss={val_loss:.5f}")

    save_checkpoint(cfg.check_point_dir, epoch, cnn, lens, optimizer, scheduler, val_loss)
    if epoch % 5 == 0:
        torch.save(lens.sqrt_height_map.detach().cpu(), f"{cfg.optics_checkpoint}/lens_height_map_epoch{epoch:03d}.pt") # Save height map
        torch.save(lens().detach().cpu(), f"{cfg.debug_checkpoint}/psf_snapshot_epoch{epoch:03d}.pt") # save PSF