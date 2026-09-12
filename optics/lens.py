import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

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


class LearnableLens(nn.Module):
    def __init__(self, resolution, wavelengths, refractive_indices, sub_pixel_pitch, focal_distance, height_noise, is_training=True):
        super().__init__()

        self.wavelengths = wavelengths
        self.sub_pixel_pitch = sub_pixel_pitch
        self.focal_distance = focal_distance
        self.height_noise = height_noise
        self.is_training = is_training

        # Precompute high resolution fresnel transfer functions.
        self._compute_fresnel_transfer_functions(resolution, wavelengths, refractive_indices, sub_pixel_pitch, focal_distance)

        # Register refractive indices as a buffer for automatic device/state tracking
        self.register_buffer("n", refractive_indices)

        # ---------------------------------------------------------
        # 1. FIXED THIN LENS (Handles focus, NOT trainable)
        # ---------------------------------------------------------
        #n_mean = float(self.n.mean().item())
        coords = torch.linspace(-1, 1, resolution)
        Y, X = torch.meshgrid(coords, coords, indexing="ij")

        half_width = (resolution / 2) * sub_pixel_pitch
        X_meters = X * half_width
        Y_meters = Y * half_width
        r2 = X_meters**2 + Y_meters**2
        self.register_buffer("r2", r2)

        # ADDED NEGATIVE SIGN: Creates a converging (convex) phase profile
        #fixed_height = -r2 / (2 * (n_mean - 1.0) * focal_distance)
        #self.register_buffer("fixed_lens_height", fixed_height)

        # ---------------------------------------------------------
        # 2. FREEFORM DOE (Trainable, Square Root Parameterization)
        # ---------------------------------------------------------
        # Matches TF: init_height_map_sqrt_value = 1e-3 * np.random.rand(...)
        init_sqrt_val = 1e-3 * torch.rand(resolution, resolution)
        self.sqrt_height_map = nn.Parameter(init_sqrt_val)

        # Aperture pupil fixed to 1.2mm diameter to fit inside the tensor grid
        physical_radius = torch.sqrt(r2)
        self.register_buffer("pupil", (physical_radius < (5e-3 / 2.0)).float())


    def _compute_fresnel_transfer_functions(self, resolution, wavelengths, refractive_indices, sub_pixel_pitch, focal_distance):
        # The authors pad by 1/4 of the original resolution on each side
        pad = resolution // 4
        padded_res = resolution + 2 * pad
        
        fx = fft.fftfreq(padded_res, d=sub_pixel_pitch)
        fy = fft.fftfreq(padded_res, d=sub_pixel_pitch)

        FY, FX = torch.meshgrid(fy, fx, indexing="ij")

        squared_sum = (FX**2 + FY**2).to(refractive_indices.dtype)

        for key, wl in wavelengths.items():
            exponent = -1j * torch.pi * wl * focal_distance * squared_sum
            self.register_buffer(f"HF_{key}", torch.exp(exponent))

    @property
    def doe_height(self):
        """
        Enforces non-negativity constraint via squaring.
        Returns physical heights natively in meters (~1e-6 scale).
        """
        return self.sqrt_height_map ** 2

    # @property
    # def total_height(self):
    #     """
    #     The physical height map: fixed lens + trainable DOE.
    #     Used for the actual wave propagation simulation.
    #     """
    #     return self.doe_height + self.fixed_lens_height

    def compute_psf(self, wavelength, Hf, n):
        # 1. Trainable DOE height (shared or independent, but scaled by its specific channel parameters)
        doe_h = self.doe_height
        if self.is_training:
            noise = torch.randn_like(doe_h) * self.height_noise
            doe_h = doe_h + noise

        # 2. Channel-specific Thin Lens height map (replicating plano_convex_initializer)
        # convex_radius = (n - 1.0) * focal_distance
        convex_radius = (n - 1.0) * self.focal_distance
        thin_lens_height = -self.r2 / (2.0 * convex_radius)

        # 3. Total height for this specific wavelength channel
        total_h = doe_h + thin_lens_height

        # 4. Phase shift calculated using THIS channel's wavelength and refractive index
        phase = (2.0 * torch.pi * (n - 1.0) * total_h) / wavelength

        field = self.pupil * torch.exp(1j * phase)
        out = fresnel_propagate(field, Hf)

        psf = torch.abs(out) ** 2
        psf = psf / (psf.sum() + 1e-8)

        return psf

    # def compute_psf(self, wavelength, Hf, n):
    #     # enforce fabrication noise (important for realism)
    #     height = self.total_height

    #     if self.is_training:
    #         # Add manufacturing constraint noise while training
    #         noise = torch.randn_like(height) * self.height_noise
    #         height = height + noise

    #     # physical phase model
    #     phase = (2 * torch.pi * (n - 1.0) * height) / wavelength

    #     field = self.pupil * torch.exp(1j * phase)

    #     out = fresnel_propagate(field, Hf)

    #     psf = torch.abs(out) ** 2
    #     psf = psf / (psf.sum() + 1e-8)

    #     return psf

    def forward(self):
        psf_r = self.compute_psf(self.wavelengths["r"], self.HF_r, self.n[0])
        psf_g = self.compute_psf(self.wavelengths["g"], self.HF_g, self.n[1])
        psf_b = self.compute_psf(self.wavelengths["b"], self.HF_b, self.n[2])

        return torch.stack([psf_r, psf_g, psf_b], dim=0)