import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ExponentialLR

from config import Config
from datasets.hdr_dataset import HDRDataset
from optics.lens import LearnableLens, l2_laplacian_regularizer
from models.deconv_net import ReconNet
from losses.l2_gamma import l2_gamma_batch
from utils.image_ops import psf_convolve_rgb
from optics.sensor import sensor_model

import matplotlib.pyplot as plt

import os
os.chdir('/home/projects/sipl-prj10826/DeepOpticsHDR_GPT')

def evaluate(lens, cnn, loader, cfg):
    lens.eval()
    cnn.eval()

    lens.is_training = False # So height map noise is not added.

    total_loss = 0.0
    display_images = True

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

            if display_images:

                import ipdb; ipdb.set_trace()

                num_examples = 3
                fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))

                for i in range(num_examples):

                    target = hdr[i].detach().cpu().permute(1, 2, 0).numpy()
                    inp = blurred[i].detach().cpu().permute(1, 2, 0).numpy()
                    output = restored[i].detach().cpu().permute(1, 2, 0).numpy()

                    axes[i, 0].imshow(target)
                    axes[i, 0].set_title("Target")

                    axes[i, 1].imshow(inp)
                    axes[i, 1].set_title("Blurred")

                    axes[i, 2].imshow(output)
                    axes[i, 2].set_title("Restored")

                    for j in range(3):
                        axes[i, j].axis("off")

                plt.suptitle(f"Epoch {epoch}")
                plt.tight_layout()
                plt.show()
                display_images = False

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
    start_epoch, _ = load_checkpoint(f"{cfg.check_point_dir}/ckpt_epoch_036.pt", cnn, lens, optimizer, scheduler)
    start_epoch += 1

for epoch in range(start_epoch, cfg.epochs):

    for i, hdr in enumerate(train_loader):

        hdr = hdr.to(cfg.device)
        # import ipdb; ipdb.set_trace()
        
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

        # Final loss
        loss = train_loss + cfg.lambda_height_map * height_map_reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Iteration number {i}, l2_loss: {train_loss}, regularizing loss: {height_map_reg_loss}, total loss: {loss}")

    # Update learning rates
    scheduler.step()

    val_loss = evaluate(lens, cnn, val_loader, cfg)
    print(f"epoch={epoch} train_loss={loss.item():.5f} val_loss={val_loss:.5f}")

    save_checkpoint(cfg.check_point_dir, epoch, cnn, lens, optimizer, scheduler, val_loss)
    if epoch % 5 == 0:
        torch.save(lens.sqrt_height_map.detach().cpu(), f"{cfg.optics_checkpoint}/lens_height_map_epoch{epoch:03d}.pt") # Save height map
        torch.save(lens().detach().cpu(), f"{cfg.debug_checkpoint}/psf_snapshot_epoch{epoch:03d}.pt") # save PSF