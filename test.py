import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Force working directory to resolve your absolute paths
os.chdir('/home/projects/sipl-prj10826/DeepOpticsHDR_PyTorch')

# Import your native custom modules
from config import Config
from datasets.hdr_dataset import HDRTestDataset
from optics.lens import LearnableLens
from models.deconv_net import ReconNet
from losses.l2_gamma import l2_gamma_batch
from utils.image_ops import psf_convolve_rgb, save_png, to_numpy_img
from optics.sensor import sensor_model, simulate_sensor_capture


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculates PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def evaluate_and_generate_triplets(lens, cnn, loader, cfg, output_dir, max_triplets=50):
    """
    Evaluates the model on the provided data loader, prints the average 
    l2_gamma loss, and saves side-by-side [GT | Blurred | Restored] triplet images
    along with their PSNR values.
    """
    lens.eval()
    cnn.eval()
    lens.is_training = False  # Deactivates height map noise for clean validation

    total_loss = 0.0
    triplet_count = 0
    psnr_values = []  # Array to hold PSNR metrics for saved triplets
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nEvaluating dataset ({len(loader.dataset)} images)...")
    
    with torch.no_grad():
        for batch_idx, hdr in enumerate(loader):
            hdr = hdr.to(cfg.device)

            # Generate optical blur using your custom optics setup
            psfs_hr = lens()
            psfs = sensor_model(
                psfs_hr,
                sampling_factor=cfg.sampling_factor,
                patch_size=cfg.image_size
            )

            clipped_hdr = torch.clamp(hdr, cfg.hdr_min_val, cfg.hdr_max_val)
            blurred_linear = psf_convolve_rgb(clipped_hdr, psfs)
            x_in = simulate_sensor_capture(blurred_linear, min_val=cfg.hdr_min_val)

            # Pass through the recovery network
            restored_hdr = cnn(x_in)

            # Calculate loss metric
            loss = l2_gamma_batch(restored_hdr, hdr)
            total_loss += loss.item()

            # Process visual samples for triplets
            if triplet_count < max_triplets:
                batch_size = hdr.shape[0]
                for sample_idx in range(batch_size):
                    if triplet_count >= max_triplets:
                        break

                    # Convert tensors to numpy using your custom utility
                    hdr_np = np.squeeze(to_numpy_img(hdr[sample_idx]))
                    blur_np = np.squeeze(to_numpy_img(x_in[sample_idx]))
                    rest_np = np.squeeze(to_numpy_img(restored_hdr[sample_idx]))

                    # Apply your custom Gamma correction (0.5 power)
                    hdr_np = np.power(np.maximum(hdr_np, 0.0), 0.5)
                    blur_np = np.power(np.maximum(blur_np, 0.0), 0.5)
                    rest_np = np.power(np.maximum(rest_np, 0.0), 0.5)

                    # --- PSNR Calculation ---
                    # We calculate PSNR on the gamma-corrected images (max_val=1.0)
                    current_psnr = calculate_psnr(hdr_np, rest_np, max_val=1.0)
                    psnr_values.append(current_psnr)

                    # Concatenate horizontally: GT | Blurred | Restored
                    triplet_img = np.concatenate([hdr_np, blur_np, rest_np], axis=1)

                    # Save the composite triplet using your native save_png tool
                    save_name = f"triplet_sample_{triplet_count:03d}.png"
                    save_path = os.path.join(output_dir, save_name)
                    save_png(save_path, triplet_img)
                    
                    triplet_count += 1

    avg_loss = total_loss / len(loader)
    
    # Save the PSNR array to disk
    psnr_array = np.array(psnr_values)
    psnr_save_path = os.path.join(output_dir, "psnr_values.npy")
    np.save(psnr_save_path, psnr_array)
    
    avg_psnr = np.mean(psnr_array) if len(psnr_array) > 0 else 0.0

    print("\n" + "="*40)
    print("           TEST EVALUATION            ")
    print("="*40)
    print(f"Average Test Loss (l2_gamma): {avg_loss:.5f}")
    print(f"Average PSNR (Saved Triplets): {avg_psnr:.2f} dB")
    print(f"Saved {triplet_count} visual triplets to: {output_dir}")
    print(f"Saved PSNR array to: {psnr_save_path}")
    print("="*40 + "\n")


def main():
    cfg = Config()

    # 1. Setup exact architecture
    lens = LearnableLens(
        cfg.lens_resolution,
        cfg.wavelengths,
        cfg.refractive_indices,
        cfg.sub_pixel_pitch,
        cfg.focal_distance,
        cfg.height_map_noise,
    ).to(cfg.device)

    cnn = ReconNet().to(cfg.device)

    # 2. Automatically discover latest checkpoint using your exact training regex
    checkpoint_dir = cfg.test_checkpoint_dir
    checkpoint_pattern = os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found matching pattern '{checkpoint_pattern}'")

    def get_epoch_num(filepath):
        match = re.search(r"epoch_?(\d+)", os.path.basename(filepath))
        return int(match.group(1)) if match else -1

    checkpoint_files.sort(key=get_epoch_num)
    latest_checkpoint_path = checkpoint_files[-1]

    print(f"Loading trained weights directly from: {latest_checkpoint_path}")
    ckpt = torch.load(latest_checkpoint_path, map_location=cfg.device)
    
    cnn.load_state_dict(ckpt["cnn"])
    lens.load_state_dict(ckpt["lens"])

    # 3. Recreate validation subset using your exact split specifications
    test_set = HDRTestDataset(cfg.test_root)

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 4. Set directory for test evaluation outputs
    test_output_dir = os.path.join(cfg.test_images_output, "test_evaluation_triplets")

    # 5. Run evaluation loop
    evaluate_and_generate_triplets(
        lens=lens, 
        cnn=cnn, 
        loader=test_loader, 
        cfg=cfg, 
        output_dir=test_output_dir, 
        max_triplets=100  # Adjust how many sample triplets you want saved
    )

if __name__ == "__main__":
    main()