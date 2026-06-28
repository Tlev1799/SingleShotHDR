import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('/home/projects/sipl-prj10826/DeepOpticsHDR_PyTorch')

def visualize_learned_psf(psf_path):
    """
    Loads a saved PSF snapshot tensor and displays its spatial structure
    across RGB channels in both linear and log scales.
    
    Expected tensor shape: (3, H, W)
    """
    if not os.path.exists(psf_path):
        print(f"Error: Snapshot file not found at {psf_path}")
        return

    # Load tensor to CPU
    psf = torch.load(psf_path, map_location="cpu")
    
    # Handle dictionary packaging if saved as a state dictionary
    if isinstance(psf, dict):
        if "psf" in psf:
            psf = psf["psf"]
        else:
            print("Error: Dict found but no 'psf' key present.")
            return

    # Ensure shape is 3D (3, H, W)
    if psf.ndim == 4:  # Strip batch dimension if present (1, 3, H, W)
        psf = psf.squeeze(0)
        
    psf = psf.detach().numpy()
    channels = ['Red Channel', 'Green Channel', 'Blue Channel']
    colors = ['Reds_r', 'Greens_r', 'Blues_r']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    for c in range(3):
        channel_data = psf[c]
        
        # --- Row 1: Linear Intensity (Highlights the focused core) ---
        ax_lin = axes[0, c]
        im_lin = ax_lin.imshow(channel_data, cmap='viridis')
        ax_lin.set_title(f"{channels[c]} (Linear)", fontsize=12, fontweight='bold')
        fig.colorbar(im_lin, ax=ax_lin, fraction=0.046, pad=0.04)
        ax_lin.axis('off')
        
        # --- Row 2: Logarithmic Intensity (Exposes the diffractive wings) ---
        ax_log = axes[1, c]
        # Add a tiny epsilon floor to prevent log(0) errors
        log_data = np.log10(channel_data + 1e-8) 
        
        im_log = ax_log.imshow(log_data, cmap='magma')
        ax_log.set_title(f"{channels[c]} (Log10 Scale)", fontsize=12, fontweight='bold')
        fig.colorbar(im_log, ax=ax_log, fraction=0.046, pad=0.04)
        ax_log.axis('off')

    plt.suptitle(f"Visualizing Learned Optical PSF: {os.path.basename(psf_path)}", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Update this string to point directly to one of your saved PSF snapshots
    target_psf = "./checkpoints/debug/psf_snapshot_epoch095.pt"
    
    visualize_learned_psf(target_psf)