from dataclasses import dataclass
import torch

@dataclass
class Config:

    data_root = "./datasets/train_dataset"

    check_point_dir = "./checkpoints"
    #cnn_checkpoint = f"{check_point_dir}/cnns"
    optics_checkpoint = f"{check_point_dir}/optics"
    debug_checkpoint = f"{check_point_dir}/debug"

    images_checkpoint = f"{check_point_dir}/images"

    image_size = 320

    batch_size = 8

    data_split = 0.995

    epochs = 100

    lr_cnn = 1e-4
    # lr_optics = 1e-5

    psf_lr_multiplier = 1

    # RGB wavelengths (meters)
    wavelengths = {
        "r": 635e-9,
        "g": 530e-9,
        "b": 450e-9
    }

    # PDMS refractive indices (R, G, B)
    refractive_indices = torch.tensor([1.4295, 1.4349, 1.4421])

    pixel_pitch = 4e-6
    focal_distance = 0.035

    pupil_diameter = 2e-3

    lambda_l2 = 1.0
    lambda_vgg = 0.05

    lambda_height_map = 1e9

    # manufacturing constraint
    height_map_noise = 20e-9

    sampling_factor = 4   # optical resolution / sensor resolution ratio
    lens_resolution = 320 * sampling_factor

    should_resume = True

    device = "cuda"