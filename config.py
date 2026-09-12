from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class Config:

    # data_root = "./my_datasets/train_dataset_temp_name" # "../Datasets_Preperation/train_bin_files"
    data_root = "../Datasets_Preperation/orig_datasets/train_processed/bin"
    # data_root = "../Datasets_Preperation/new_hdrplus_processed_dataset/train"
    # test_root = "./my_datasets/test_dataset" # "../Datasets_Preperation/test_exr_files"
    # test_root = "../Datasets_Preperation/test_exr_files"
    test_root = "../Datasets_Preperation/orig_datasets/test_processed_cpp/bin"
    # test_root = "../Datasets_Preperation/new_attempt_downsample/processed_hdr_320"

    # NOTE: Change this value for temporary testing with alternative my_datasets.
    temp_dataset_path = "./my_datasets/test_dataset"

    test_checkpoint_dir = "./model_to_test"
    test_images_output = f"{test_checkpoint_dir}/images"

    check_point_dir = "./checkpoints"

    RESULTS_DIR_FORMAT = "./trained_networks/lambda_gan_{lambda_gan}.pt"

    # --- New Configs for 60+60 Branching Strategy ---
    base_epochs = 100
    branch_epochs = 100
    base_model_path = "./trained_networks/lambda_gan_{lambda_gan}.pt" # TODO: Changed to run multiple lambda in simul.

    image_size = 320

    # TODO: This is the last main change. See if it improves the psfs learned image.
    batch_size = 8 # 16

    data_split = 0.995

    epochs = 100

    lr_cnn = 1e-4
    # lr_optics = 1e-5

    psf_lr_multiplier = 1 # 1e6

    # RGB wavelengths (meters)
    wavelengths = {
        "r": 635e-9,
        "g": 530e-9,
        "b": 450e-9
    }

    # PDMS refractive indices (R, G, B)
    refractive_indices = torch.tensor([1.4295, 1.4349, 1.4421])

    pixel_pitch = 4.29e-6
    focal_distance = 0.035

    pupil_diameter = 5e-3

    lambda_l2 = 1.0
    lambda_vgg = 0.05

    # TODO: Do not forget to change this back.
    lambda_height_map = 1e9

    max_doe_height = 1.55e-6

    # manufacturing constraint
    height_map_noise = 20e-9

    sampling_factor = 1 #4   # optical resolution / sensor resolution ratio
    sub_pixel_pitch = pixel_pitch / sampling_factor

    phase_mask_size = 5.6e-3
    raw_resolution = phase_mask_size / sub_pixel_pitch
    lens_resolution = int(np.ceil(raw_resolution / sampling_factor) * sampling_factor)  # Evaluates to 1306

    hdr_max_val = 64
    hdr_min_val = 1e-5

    sensor_noise_std = 0.0

    device = "cuda"