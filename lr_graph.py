import os
import glob
import re
import torch
import matplotlib.pyplot as plt
from config import Config  # Import your project's config file

import os
os.chdir('/home/projects/sipl-prj10826/DeepOpticsHDR_PyTorch')

def plot_lr_from_checkpoints():
    # Instantiate your config to get the correct save path
    cfg = Config()
    checkpoint_dir = cfg.check_point_dir

    # Look for files matching your ckpt_epoch_*.pt pattern
    search_pattern = os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")
    checkpoint_files = glob.glob(search_pattern)

    # Fallback: check subdirectories recursively if paths are nested
    if not checkpoint_files:
        search_pattern = os.path.join(checkpoint_dir, "**/ckpt_epoch_*.pt")
        checkpoint_files = glob.glob(search_pattern, recursive=True)

    if not checkpoint_files:
        print(f"Error: No checkpoint files found matching 'ckpt_epoch_*.pt' in {checkpoint_dir}")
        return

    epochs = []
    lrs = []
    val_losses = []

    print(f"Found {len(checkpoint_files)} checkpoint files. Extracting states...")

    for file_path in checkpoint_files:
        # Extract the epoch number from the filename string using regex
        match = re.search(r"epoch_?(\d+)", os.path.basename(file_path))
        if not match:
            continue
        epoch_num = int(match.group(1))

        try:
            # Map location to CPU to prevent filling up VRAM while reading files
            checkpoint = torch.load(file_path, map_location="cpu")

            # Extract the actual active learning rate from the optimizer's parameter groups
            if "optimizer" in checkpoint:
                lr = checkpoint["optimizer"]["param_groups"][0]["lr"]
            elif "scheduler" in checkpoint and "_last_lr" in checkpoint["scheduler"]:
                lr = checkpoint["scheduler"]["_last_lr"][0]
            else:
                continue

            print("yay")
            epochs.append(epoch_num)
            lrs.append(lr)

            # Extract the validation loss if saved alongside the state dictionary
            if "val_loss" in checkpoint:
                val_losses.append(checkpoint["val_loss"])
            else:
                val_losses.append(None)

        except Exception as e:
            print(f"Skipping corrupt or unreadable file {os.path.basename(file_path)}: {e}")
            continue

    if not epochs:
        print("No valid training state statistics could be parsed from the files.")
        return

    # Sort all extracted data sequentially by epoch number
    sorted_data = sorted(zip(epochs, lrs, val_losses))
    epochs, lrs, val_losses = zip(*sorted_data)

    # Build Dual-Axis Graph
    fig, ax1 = plt.subplots(figsize=(11, 5))

    # Left Y-Axis: Learning Rate (Logarithmic Scale)
    color_lr = "crimson"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Learning Rate", color=color_lr, fontsize=12)
    ax1.plot(epochs, lrs, marker="o", markersize=4, color=color_lr, linewidth=1.5, label="Learning Rate")
    ax1.tick_params(axis="y", labelcolor=color_lr)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # Right Y-Axis: Validation Loss (Linear Scale)
    if any(v is not None for v in val_losses):
        ax2 = ax1.twinx()
        color_loss = "royalblue"
        ax2.set_ylabel("Validation Loss", color=color_loss, fontsize=12)
        ax2.plot(epochs, val_losses, marker="s", markersize=3, linestyle="--", color=color_loss, linewidth=1.2, alpha=0.7, label="Val Loss")
        ax2.tick_params(axis="y", labelcolor=color_loss)
        plt.title("E2E Learning Rate & Loss Progression directly from Checkpoints", fontsize=13, fontweight="bold")
    else:
        plt.title("E2E Learning Rate Progression directly from Checkpoints", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_lr_from_checkpoints()