import torch
import torch.nn as nn
import torch.nn.functional as F

from .fresnel import fresnel_propagate

def l2_laplacian_regularizer(height_map):
    """
    TF-equivalent:
    tf.contrib.layers.l2_regularizer applied via conv Laplacian
    """

    if height_map.dim() == 2:
        height_map = height_map.unsqueeze(0).unsqueeze(0)
    elif height_map.dim() == 3:
        height_map = height_map.unsqueeze(1)

    kernel = torch.tensor(
        [[0., 1., 0.],
         [1., -4., 1.],
         [0., 1., 0.]],
        device=height_map.device,
        dtype=height_map.dtype
    ).view(1, 1, 3, 3)

    lap = F.conv2d(height_map, kernel, padding=1)

    return (lap ** 2).mean()

def thin_lens_height(resolution, focal_length, sub_pixel_pitch, n):
    """
    Parabolic thin lens approximation:
    h(r) = r^2 / (2 * (n-1) * f)
    """

    # import ipdb; ipdb.set_trace()

    coords = torch.linspace(-1, 1, resolution)
    Y, X = torch.meshgrid(coords, coords, indexing="ij")

    # Transform pixels to meters.
    half_width = (resolution / 2) * sub_pixel_pitch
    X = X * half_width
    Y = Y * half_width

    r2 = X**2 + Y**2

    height = r2 / (2 * (n - 1.0) * focal_length)

    return height


class LearnableLens(nn.Module):

    def __init__(self, resolution, wavelengths, refractive_indices,
                 sub_pixel_pitch, focal_distance, height_noise, is_training=True):

        super().__init__()

        self.wavelengths = wavelengths
        self.sub_pixel_pitch = sub_pixel_pitch
        self.focal_distance = focal_distance
        self.height_noise = height_noise
        self.is_training = is_training

        # Register refractive indices as a buffer for automatic device/state tracking
        self.register_buffer("n", refractive_indices)

        # Initial values of height map will correspond to thinLens -and ideal surface lens.
        n_mean = float(self.n.mean().item())

        h_init = thin_lens_height(
            resolution,
            focal_distance,
            sub_pixel_pitch,
            n_mean
        )

        # learnable base structure
        self.sqrt_height_map = nn.Parameter(torch.sqrt(h_init.clamp(min=0)))

        coords = torch.linspace(-1, 1, resolution)
        Y, X = torch.meshgrid(coords, coords, indexing="ij")
        half_width = (resolution / 2) * sub_pixel_pitch
        X_meters = X * half_width
        Y_meters = Y * half_width
        physical_radius = torch.sqrt(X_meters**2 + Y_meters**2)

        self.register_buffer("pupil", (physical_radius < 1.0).float())

    @property
    def height(self):
        """
        Physical height map: h = (sqrt_h)^2
        Automatically computed during training.
        """
        return self.sqrt_height_map ** 2

    def compute_psf(self, wavelength, n):

        # enforce fabrication noise (important for realism)
        height = self.sqrt_height_map ** 2

        if self.is_training:
            # Add manufacturing constraint noise while training
            noise = torch.randn_like(height) * self.height_noise
            height = height + noise

        # physical phase model
        phase = (2 * torch.pi * (n - 1.0) * height) / wavelength

        field = self.pupil * torch.exp(1j * phase)

        out = fresnel_propagate(
            field,
            wavelength,
            self.sub_pixel_pitch,
            self.focal_distance
        )

        psf = torch.abs(out) ** 2
        psf = psf / (psf.sum() + 1e-8)

        return psf

    def forward(self):

        psf_r = self.compute_psf(self.wavelengths["r"], self.n[0])
        psf_g = self.compute_psf(self.wavelengths["g"], self.n[1])
        psf_b = self.compute_psf(self.wavelengths["b"], self.n[2])

        return torch.stack([psf_r, psf_g, psf_b], dim=0)