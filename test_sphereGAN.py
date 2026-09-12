import argparse
from torch.utils.data import DataLoader
import torch
import math
import os
import random
from torchvision.utils import save_image

# Metric libraries
import lpips
import pyiqa
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.kid import KernelInceptionDistance

from config import Config
from my_datasets.hdr_dataset import HDRTestDataset, HDRDataset
from optics.lens import LearnableLens
from models.deconv_net import ReconNet
from optics.sensor import sensor_model, simulate_sensor_capture
from utils.image_ops import psf_convolve_rgb


class HDREvaluator:
    def __init__(self, device="cuda", dataset_size=3000):
        self.device = device
        # Standard metrics initialized on the correct device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # LPIPS uses VGG by default. Expects inputs in [-1, 1]
        self.lpips_metric = lpips.LPIPS(net='vgg').to(device)

        # Initialize NIQE metric on the correct device
        self.niqe_metric = pyiqa.create_metric('niqe', as_loss=False, device=device)
        self.musiq_metric = pyiqa.create_metric('musiq', as_loss=False, device=device)
        self.nima_metric = pyiqa.create_metric('nima', as_loss=False, device=device)

        # Technical No-Reference Metrics via pyiqa (Lower is better for all three)
        self.piqe_metric = pyiqa.create_metric('piqe', as_loss=False, device=device)
        self.brisque_metric = pyiqa.create_metric('brisque', as_loss=False, device=device)
        self.ilniqe_metric = pyiqa.create_metric('ilniqe', as_loss=False, device=device)

        # KID needs a subset size smaller than or equal to the dataset length.
        kid_subset_size = min(50, dataset_size)
        self.kid = KernelInceptionDistance(subset_size=kid_subset_size).to(device)
        
    def mu_law_compression(self, x, mu=5000.0):
        """Applies mu-law tonemapping to compress HDR to [0,1]."""
        return torch.log(1.0 + mu * x) / math.log(1.0 + mu)

    def gamma_compression(self, x, gamma=2.2):
        """Applies standard gamma compression."""
        return torch.pow(torch.clamp(x, min=1e-8), 1.0 / gamma)

    def evaluate(self, pred_linear, target_linear, max_val=64.0):
        """
        Calculates metrics across different HDR domains.
        Expects Tensors of shape (B, C, H, W).
        """
        # Normalize to [0, 1] based on the theoretical max of your HDR data
        p_lin = torch.clamp(pred_linear / max_val, 0.0, 1.0)
        t_lin = torch.clamp(target_linear / max_val, 0.0, 1.0)

        # 1. Linear Domain
        psnr_l = self.psnr(p_lin, t_lin).item()
        ssim_l = self.ssim(p_lin, t_lin).item()

        # 2. Gamma Domain
        p_gamma = self.gamma_compression(p_lin)
        t_gamma = self.gamma_compression(t_lin)
        psnr_gamma = self.psnr(p_gamma, t_gamma).item()

        # 3. Mu-Law Domain
        p_mu = self.mu_law_compression(p_lin)
        t_mu = self.mu_law_compression(t_lin)
        psnr_mu = self.psnr(p_mu, t_mu).item()
        ssim_mu = self.ssim(p_mu, t_mu).item()

        niqe_val = self.niqe_metric(p_gamma).mean().item() if hasattr(self, 'niqe_metric') else 0.0
        musiq_val = self.musiq_metric(p_gamma).mean().item() if hasattr(self, 'musiq_metric') else 0.0
        nima_val = self.nima_metric(p_gamma).mean().item() if hasattr(self, 'nima_metric') else 0.0

        # 4. Technical NR Metrics (Evaluated on Gamma domain, [0, 1] range)
        piqe_val = self.piqe_metric(p_gamma).mean().item() if hasattr(self, 'piqe_metric') else 0.0
        brisque_val = self.brisque_metric(p_gamma).mean().item() if hasattr(self, 'brisque_metric') else 0.0
        ilniqe_val = self.ilniqe_metric(p_gamma).mean().item() if hasattr(self, 'ilniqe_metric') else 0.0

        # 6. KID (Update state internally)
        # torchmetrics KID expects uint8 tensors [0, 255]
        p_uint8 = (p_gamma * 255.0).byte()
        t_uint8 = (t_gamma * 255.0).byte()
        self.kid.update(t_uint8, real=True)
        self.kid.update(p_uint8, real=False)

        # 4. LPIPS (Evaluated on Gamma domain, scaled to [-1, 1])
        p_lpips = (p_gamma * 2.0) - 1.0
        t_lpips = (t_gamma * 2.0) - 1.0
        lpips_val = self.lpips_metric(p_lpips, t_lpips).mean().item()

        return {
            "PSNR_L": psnr_l, "SSIM_L": ssim_l,
            "PSNR_gamma": psnr_gamma,
            "PSNR_mu": psnr_mu, "SSIM_mu": ssim_mu,
            "LPIPS": lpips_val,
            "NIQE": niqe_val,
            "MUSIQ": musiq_val,
            "NIMA": nima_val,
            "PIQE": piqe_val,
            "BRISQUE": brisque_val,
            "ILNIQE": ilniqe_val
        }

def parse_arguments():
    parser = argparse.ArgumentParser(description="SphereGAN Training Script")
    
    parser.add_argument('lambda_gan', type=float, help='The weight parameter (lambda_gan) for the GAN loss.')
    parser.add_argument('--base_run', action='store_true', help='Flag to train the base model.')

    # Percentage of images to save (0.0 to 100.0)
    parser.add_argument(
        '--save_jpg_percent', 
        type=float, 
        default=0.0, 
        help='Percentage of test images to save as JPGs (e.g., 10.5 for 10.5%)'
    )

    return parser.parse_args()

def load_trained_network(path, cnn, lens, device):
    network = torch.load(path, map_location=device)

    cnn.load_state_dict(network["cnn"])
    lens.load_state_dict(network["lens"])

# Get lambda_gan of network to use.
args = parse_arguments()
lambda_gan = args.lambda_gan
is_base_run = args.base_run

# Get configuration.
cfg = Config()

save_prob = args.save_jpg_percent / 100.0  # Convert percentage to a 0.0-1.0 probability

if is_base_run:
    trained_network_path = cfg.base_model_path.format(lambda_gan=lambda_gan)
else:
    trained_network_path = cfg.RESULTS_DIR_FORMAT.format(lambda_gan=lambda_gan)
output_path_dir = trained_network_path.replace(".pt", "_jpgs")

os.makedirs(output_path_dir, exist_ok=True)
print(f"Visual results will be saved to: {output_path_dir}")

# Get test dataset.
test_set = HDRTestDataset(cfg.test_root)

# Create loader.
test_loader = DataLoader(
    test_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Create the optics network.
lens = LearnableLens(
    cfg.lens_resolution,
    cfg.wavelengths,
    cfg.refractive_indices,
    cfg.sub_pixel_pitch,
    cfg.focal_distance,
    cfg.height_map_noise,
).to(cfg.device)

# Create the digital network.
cnn = ReconNet().to(cfg.device)

# Load the trained network.
load_trained_network(trained_network_path, cnn, lens, cfg.device)

################# Start Evaluation #################
lens.eval()
cnn.eval()
lens.is_training = False # Deactivates height map noise.

evaluator = HDREvaluator(device=cfg.device, dataset_size=len(test_set))

# Dictionaries to accumulate metrics
total_metrics = {
    "PSNR_L": 0.0, "SSIM_L": 0.0,
    "PSNR_gamma": 0.0, 
    "PSNR_mu": 0.0, "SSIM_mu": 0.0,
    "LPIPS": 0.0,
    "NIQE": 0.0,
    "MUSIQ": 0.0,
    "NIMA": 0.0,
    "PIQE": 0.0,
    "BRISQUE": 0.0,
    "ILNIQE": 0.0
}

num_batches = len(test_loader)
print(f"Starting evaluation on {len(test_set)} test images...")

# ------------------------------------------------------------------
# SAVE THE LEARNED PSF (COLOR-SEPARATED & LOG DOMAIN)
# ------------------------------------------------------------------
with torch.no_grad():
    psfs_hr = lens() # Shape: (3, H, W)
    psfs = sensor_model(
                psfs_hr,
                sampling_factor=cfg.sampling_factor,
                patch_size=cfg.image_size
            )
    
    # Transfer to log domain.
    epsilon = 1e-6
    psf_normalized = psfs / (psfs.max() + epsilon)
    psf_log = torch.log10(psf_normalized + epsilon)
    
    # Scale the log-domain tensor to [0, 1] for standard JPG saving
    psf_log_scaled = (psf_log - psf_log.min()) / (psf_log.max() - psf_log.min())
    
    # Extract individual channels (Shape becomes (1, H, W))
    psf_r = psf_log_scaled[0:1, :, :]
    psf_g = psf_log_scaled[1:2, :, :]
    psf_b = psf_log_scaled[2:3, :, :]
    
    # Save separately
    save_image(psf_r, os.path.join(output_path_dir, f"psf_R_log_lambda_{lambda_gan}.jpg"))
    save_image(psf_g, os.path.join(output_path_dir, f"psf_G_log_lambda_{lambda_gan}.jpg"))
    save_image(psf_b, os.path.join(output_path_dir, f"psf_B_log_lambda_{lambda_gan}.jpg"))

    # Extract, normalize, and save the DOE height map as a grayscale image
    height_map = lens.doe_height.detach() 
    hm_scaled = (height_map - height_map.min()) / (height_map.max() - height_map.min() + epsilon)
    save_image(hm_scaled, os.path.join(output_path_dir, "learned_height_map.jpg"))
    
print(f"Saved color-separated log-domain PSFs to: {output_path_dir}")

with torch.no_grad():
    for i, hdr in enumerate(test_loader):
        hdr = hdr.to(cfg.device)

        if torch.isnan(hdr).any() or torch.isinf(hdr).any():
            print("Corrupted input data found!")
        
        # Extract psfs of the learned height map.
        psfs_hr = lens() 
        psfs = sensor_model(
            psfs_hr,
            sampling_factor=cfg.sampling_factor,
            patch_size=cfg.image_size
        )

        # Pass images through the network.
        clipped_hdr = torch.clamp(hdr, cfg.hdr_min_val, cfg.hdr_max_val)
        blurred_linear = psf_convolve_rgb(clipped_hdr, psfs)
        x_in = simulate_sensor_capture(blurred_linear, min_val=cfg.hdr_min_val) 
        restored_hdr = cnn(x_in)

        # Evaluate
        batch_metrics = evaluator.evaluate(restored_hdr, clipped_hdr, max_val=cfg.hdr_max_val)
        
        for key in total_metrics:
            total_metrics[key] += batch_metrics[key]

        # ------------------------------------------------------------------
        # JPG SAVING LOGIC
        # ------------------------------------------------------------------
        if save_prob > 0.0:
            # We must tone-map the HDR to Gamma domain [0,1] so the JPG looks correct
            p_lin = torch.clamp(restored_hdr / cfg.hdr_max_val, 0.0, 1.0)
            t_lin = torch.clamp(hdr / cfg.hdr_max_val, 0.0, 1.0)
            
            # Use mu-law compression instead of gamma to tone-map the HDR
            # The high mu value (5000) will aggressively pull up the dark midtones
            p_vis = evaluator.mu_law_compression(p_lin, mu=5000.0)
            t_vis = evaluator.mu_law_compression(t_lin, mu=5000.0)
            
            # Loop through each image in the current batch
            for b in range(p_vis.size(0)):
                if random.random() < save_prob:
                    # Concatenate side-by-side: [Target (Left) | Prediction (Right)]
                    # dim=2 is the width dimension (C, H, W)
                    comparison = torch.cat([t_vis[b], p_vis[b]], dim=2)
                    
                    filename = os.path.join(output_path_dir, f"batch_{i:04d}_img_{b:02d}.jpg")
                    save_image(comparison, filename)
        # ------------------------------------------------------------------
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_batches} batches...")

# Calculate and print final averages
print("\n" + "="*50)
print(f"FINAL TEST RESULTS (lambda_gan = {lambda_gan})")
print("="*50)

# Print averaged batch metrics (PSNR, SSIM, LPIPS, NIQE)
for key in total_metrics:
    avg_val = total_metrics[key] / num_batches
    print(f"{key:>12}: {avg_val:.4f}")

# Compute the final dataset-level KID metric
try:
    kid_mean, kid_std = evaluator.kid.compute()
    print(f"{'KID_mean':>12}: {kid_mean.item():.4f}")
    print(f"{'KID_std':>12}: {kid_std.item():.4f}")
except Exception as e:
    print(f"{'KID':>12}: N/A (Dataset too small or missing data)")
    
print("="*50)