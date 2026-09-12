from config import Config
from my_datasets.hdr_dataset import HDRDataset
from optics.lens import LearnableLens, l2_laplacian_regularizer
from models.deconv_net import ReconNet
from losses.patch_discriminator import PatchDiscriminator
from utils.image_ops import psf_convolve_rgb
from optics.sensor import sensor_model, simulate_sensor_capture
from losses.l2_gamma import l2_gamma_batch

from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

import torch.nn.functional as F
import argparse
import torch
import glob
import re
import os
import sys

os.chdir('/home/projects/sipl-prj10826/DeepOpticsHDR_PyTorch')

def parse_arguments():
    parser = argparse.ArgumentParser(description="SphereGAN Training Script")
    parser.add_argument('--base_run', action='store_true', help='Flag to train the base model.')
    parser.add_argument('--lambda_gan', type=float, help='The weight parameter for the GAN loss.')
    return parser.parse_args()

def save_checkpoint(save_path, epoch, cnn, lens, optimizer, scheduler, discriminator, opt_disc, disc_scheduler, lambda_gan, val_loss=None):
    path = os.path.join(save_path, f"ckpt_epoch_{epoch:03d}.pt")
    torch.save({
        "epoch": epoch,
        "cnn": cnn.state_dict(),
        "lens": lens.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_disc": opt_disc.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict(),
        "lambda_gan": lambda_gan,
        "val_loss": val_loss
    }, path)
    print(f"[CHECKPOINT] saved: {path}")

def load_checkpoint(path, cnn, lens, optimizer, scheduler, discriminator, opt_disc, disc_scheduler):
    ckpt = torch.load(path)
    cnn.load_state_dict(ckpt["cnn"])
    lens.load_state_dict(ckpt["lens"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    
    if "discriminator" in ckpt: discriminator.load_state_dict(ckpt["discriminator"])
    if "opt_disc" in ckpt: opt_disc.load_state_dict(ckpt["opt_disc"])
    if "disc_scheduler" in ckpt: disc_scheduler.load_state_dict(ckpt["disc_scheduler"])

    lam_gan = ckpt.get("lambda_gan", ckpt.get("lmabda_gan", "Unknown"))
    print(f"Loaded checkpoint was trained with lambda_gan: {lam_gan}")
    return ckpt["epoch"], ckpt.get("val_loss", None)

def compute_sphere_distance(q):
    norm_sq = q ** 2 
    val = (norm_sq - 1.0) / (norm_sq + 1.0)
    val = torch.clamp(val, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(val)

def evaluate(lens, cnn, loader, cfg):
    lens.eval()
    cnn.eval()
    lens.is_training = False
    total_loss = 0.0

    with torch.no_grad():
        for hdr in loader:
            hdr = hdr.to(cfg.device)
            psfs_hr = lens() 
            psfs = sensor_model(psfs_hr, sampling_factor=cfg.sampling_factor, patch_size=cfg.image_size)
            clipped_hdr = torch.clamp(hdr, cfg.hdr_min_val, cfg.hdr_max_val)
            blurred_linear = psf_convolve_rgb(clipped_hdr, psfs)
            x_in = simulate_sensor_capture(blurred_linear, min_val=cfg.hdr_min_val) 
            restored_hdr = cnn(x_in)
            loss = l2_gamma_batch(restored_hdr, clipped_hdr)
            total_loss += loss.item()  

    lens.train()
    cnn.train()
    lens.is_training = True
    return total_loss / len(loader)


args = parse_arguments()
cfg = Config()

is_base_run = args.base_run
lambda_gan = args.lambda_gan

base_model_to_use = cfg.base_model_path.format(lambda_gan=lambda_gan)

# ----------------------------------------------------------------------
# RUN MODE DYNAMICS & CHECKPOINT ROUTING
# ----------------------------------------------------------------------
if is_base_run:
    if os.path.exists(base_model_to_use):
        print(f"[INFO] Base model already exists at {base_model_to_use}.")
        print("If you intend to train a new base model, delete the file manually. Exiting.")
        sys.exit(0)
    
    total_epochs = cfg.base_epochs
    cfg.check_point_dir = os.path.join(cfg.check_point_dir, f"base_model_lambdaGAN_{lambda_gan}")

else:
    total_epochs = cfg.branch_epochs
    cfg.check_point_dir = os.path.join(cfg.check_point_dir, f"lambdaGAN_{lambda_gan}")


# Get training dataset.
dataset = HDRDataset(cfg.data_root)
train_len = int(cfg.data_split * len(dataset))
val_len = int(len(dataset) - train_len)

train_set, val_set = random_split(
    dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Create Networks
lens = LearnableLens(
    cfg.lens_resolution, cfg.wavelengths, cfg.refractive_indices,
    cfg.sub_pixel_pitch, cfg.focal_distance, cfg.height_map_noise,
).to(cfg.device)

cnn = ReconNet().to(cfg.device)
discriminator = PatchDiscriminator().to(cfg.device)

# Initialize FRESH Optimizers & Schedulers
optimizer = torch.optim.Adam([
    {"params": cnn.parameters(), "lr": cfg.lr_cnn},
    {"params": lens.parameters(), "lr": cfg.lr_cnn * cfg.psf_lr_multiplier}
], betas=(0.9, 0.999), eps=1e-8)

opt_disc = torch.optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
disc_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt_disc, gamma=0.99)

# ----------------------------------------------------------------------
# AUTOMATIC CHECKPOINT RESUMPTION OR FRESH START LOGIC
# ----------------------------------------------------------------------
checkpoint_dir = cfg.check_point_dir
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_pattern = os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")
checkpoint_files = glob.glob(checkpoint_pattern)

start_epoch = 0
val_loss = None

if checkpoint_files:
    # Resume flow for either the base run or a branch run
    def get_epoch_num(filepath):
        match = re.search(r"epoch_?(\d+)", os.path.basename(filepath))
        return int(match.group(1)) if match else -1

    checkpoint_files.sort(key=get_epoch_num)
    latest_checkpoint_path = checkpoint_files[-1]

    print("-" * 60)
    print(f"Automatic Resume Triggered in {checkpoint_dir}")
    print(f"Loading latest checkpoint file: {latest_checkpoint_path}")

    checkpoint_epoch, last_val_loss = load_checkpoint(latest_checkpoint_path, cnn, lens, optimizer, scheduler, discriminator, opt_disc, disc_scheduler)
    start_epoch = checkpoint_epoch + 1

    print(f"Resuming pipeline execution from Epoch {start_epoch}, which had val_loss: {last_val_loss}")
    print("-" * 60)

else:
    print("-" * 60)
    print(f"No existing checkpoints found in '{checkpoint_dir}'.")
    
    if is_base_run:
        print("Starting a completely fresh base training run from Epoch 0.")
        print("Saving initial untrained PSF to disk...")
        with torch.no_grad():
            lens.eval() 
            initial_psfs_hr = lens() 
            initial_psfs = sensor_model(initial_psfs_hr, sampling_factor=cfg.sampling_factor, patch_size=cfg.image_size)
            
            epsilon = 1e-6
            psf_norm = initial_psfs / (initial_psfs.max() + epsilon)
            psf_log = torch.log10(psf_norm + epsilon)
            psf_log_scaled = (psf_log - psf_log.min()) / (psf_log.max() - psf_log.min())
            
            save_image(psf_log_scaled, os.path.join(cfg.check_point_dir, "initial_psf_RGB_log.jpg"))
            save_image(psf_log_scaled[0:1], os.path.join(cfg.check_point_dir, "initial_psf_R_log.jpg"))
            save_image(psf_log_scaled[1:2], os.path.join(cfg.check_point_dir, "initial_psf_G_log.jpg"))
            save_image(psf_log_scaled[2:3], os.path.join(cfg.check_point_dir, "initial_psf_B_log.jpg"))

            height_map = lens.doe_height.detach() 
            hm_scaled = (height_map - height_map.min()) / (height_map.max() - height_map.min() + epsilon)
            save_image(hm_scaled, os.path.join(cfg.check_point_dir, "initial_height_map.jpg"))
        lens.train()
    else:
        # When building on a trained network, we always use lambda_gan=0 as starting point.
        base_model_to_use = cfg.base_model_path.format(lambda_gan=0.0)
        if not os.path.exists(base_model_to_use):
            print(f"Error: Required base model not found at {base_model_to_use}")
            sys.exit(1)
            
        print(f"Branching Mode: Loading pre-trained base model weights from {base_model_to_use}")
        ckpt = torch.load(base_model_to_use)
        cnn.load_state_dict(ckpt["cnn"])
        lens.load_state_dict(ckpt["lens"])
        print("Base weights successfully loaded. Fresh optimizers initialized. Starting branch at Epoch 0.")
        
    print("-" * 60)

print(f"Perception loss weight used: {lambda_gan}")

# ==============================================================================
#                                   MAIN TRAINING LOOP
# ==============================================================================
for epoch in range(start_epoch, total_epochs):

    cnn.train()
    lens.train()
    discriminator.train()

    for i, hdr in enumerate(train_loader):
        hdr = hdr.to(cfg.device)

        # Extract psf from current height map. 
        psfs_hr = lens() 
        psfs = sensor_model(psfs_hr, sampling_factor=cfg.sampling_factor, patch_size=cfg.image_size)

        # Pass images through the network: phase mask + sensor + cnn
        clipped_hdr = torch.clamp(hdr, cfg.hdr_min_val, cfg.hdr_max_val)
        blurred_linear = psf_convolve_rgb(clipped_hdr, psfs)
        x_in = simulate_sensor_capture(blurred_linear, noise_std=cfg.sensor_noise_std, min_val=cfg.hdr_min_val) 
        restored_hdr = cnn(x_in)

        # ----------------------------------------------------------------------
        #                           TRAIN DISCRIMINATOR
        # ----------------------------------------------------------------------
        opt_disc.zero_grad()

        scaled_hdr = torch.clamp(hdr / cfg.hdr_max_val, 0.0, 1.0)
        norm_hdr = torch.pow(torch.clamp(scaled_hdr, min=1e-8), 1.0 / 2.2)

        scaled_restored_hdr = torch.clamp(restored_hdr / cfg.hdr_max_val, 0.0, 1.0)
        norm_restored_hdr = torch.pow(torch.clamp(scaled_restored_hdr, min=1e-8), 1.0 / 2.2)

        pred_real = discriminator(norm_hdr)
        pred_fake = discriminator(norm_restored_hdr.detach())

        d_real = compute_sphere_distance(pred_real)
        d_fake = compute_sphere_distance(pred_fake)

        loss_disc = 0.0
        for r in [1, 2, 3]:
            loss_disc += (d_real ** r - d_fake ** r).mean()

        loss_disc.backward()
        opt_disc.step()

        # ----------------------------------------------------------------------
        #                           TRAIN GENERATOR (OPTICS + CNN)
        # ----------------------------------------------------------------------
        optimizer.zero_grad()

        # distortion_loss = l2_gamma_batch(restored_hdr, hdr)
        distortion_loss = l2_gamma_batch(restored_hdr, clipped_hdr)

        height_map_reg_loss = l2_laplacian_regularizer(lens.doe_height)
        height_map_scaled_loss = cfg.lambda_height_map * height_map_reg_loss

        pred_fake_for_gen = discriminator(norm_restored_hdr)
        d_fake_gen = compute_sphere_distance(pred_fake_for_gen)
        
        loss_gan = 0.0
        for r in [1, 2, 3]:
            loss_gan += (d_fake_gen ** r).mean()

        total_gen_loss = distortion_loss + height_map_scaled_loss + (lambda_gan * loss_gan)

        total_gen_loss.backward()      
        optimizer.step()

        with torch.no_grad():
            max_sqrt_val = (cfg.max_doe_height) ** 0.5
            lens.sqrt_height_map.clamp_(min=-max_sqrt_val, max=max_sqrt_val)

        if i % 200 == 0:
            print(f"Iteration {i} | Distortion loss: {distortion_loss:.5f} | Reg loss (scaled): {height_map_scaled_loss:.5f} | Disc_Loss: {loss_disc:.5f} | GAN_Loss: {loss_gan:.5f} | Total: {total_gen_loss:.5f}")

    val_loss = evaluate(lens, cnn, val_loader, cfg)
    print(f"epoch={epoch} | train_loss={total_gen_loss.item():.5f} | val_loss={val_loss:.5f}")

    scheduler.step()
    disc_scheduler.step()

    if epoch % 4 == 0:
        save_checkpoint(cfg.check_point_dir, epoch, cnn, lens,
                        optimizer, scheduler, discriminator, opt_disc,
                        disc_scheduler, lambda_gan, val_loss)

# Save final network.
if is_base_run:
    save_path = cfg.base_model_path.format(lambda_gan=lambda_gan)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": total_epochs,
        "cnn": cnn.state_dict(),
        "lens": lens.state_dict(),
        "val_loss": val_loss
    }, save_path)
    print(f"[SUCCESS] Base model saved to {save_path}")
else:
    results_path = cfg.RESULTS_DIR_FORMAT.format(lambda_gan=lambda_gan)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    torch.save({
        "epoch": total_epochs,
        "cnn": cnn.state_dict(),
        "lens": lens.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_disc": opt_disc.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict(),
        "val_loss": val_loss
    }, results_path)
    print(f"[SUCCESS] Branch model for lambda_gan={lambda_gan} saved to {results_path}")