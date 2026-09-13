from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class Config:

    data_root = "./my_datasets/train_dataset"
    test_root = "./my_datasets/test_dataset"
    check_point_dir = "./checkpoints"

    base_epochs = 100
    branch_epochs = 100
    base_model_path = "./trained_networks/base_model_lambda_gan_{lambda_gan}.pt"
    RESULTS_DIR_FORMAT = "./trained_networks/lambda_gan_{lambda_gan}.pt"

    image_size = 320
    batch_size = 8 # 16
    data_split = 0.995

    lr_cnn = 1e-4
    psf_lr_multiplier = 1

    # RGB wavelengths (meters)
    wavelengths = {
        "r": 635e-9,
        "g": 530e-9,
        "b": 450e-9
    }

    # PDMS refractive indices (R, G, B)
    refractive_indices = torch.tensor([1.4295, 1.4349, 1.4421])

    # Physical values are in meters.
    pixel_pitch = 4.29e-6
    focal_distance = 35e-3
    pupil_diameter = 5e-3

    # Scale of height map laplacian regularizer (higher this is the smoother the final height map must be).
    lambda_height_map = 1e9

    max_doe_height = 1.55e-6

    # manufacturing constraint
    height_map_noise = 20e-9

    sampling_factor = 1 # 4
    sub_pixel_pitch = pixel_pitch / sampling_factor

    phase_mask_size = 5.6e-3
    raw_resolution = phase_mask_size / sub_pixel_pitch
    lens_resolution = int(np.ceil(raw_resolution / sampling_factor) * sampling_factor)  # Evaluates to 1306

    hdr_max_val = 64
    hdr_min_val = 1e-5

    sensor_noise_std = 0.0

    device = "cuda"