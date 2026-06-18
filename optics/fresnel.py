import torch
import torch.nn as nn
import torch.fft as fft


def fresnel_propagate(field, wavelength, pixel_pitch, distance):

    H, W = field.shape[-2:]

    fx = fft.fftfreq(W, d=pixel_pitch, device=field.device)
    fy = fft.fftfreq(H, d=pixel_pitch, device=field.device)

    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    Hf = torch.exp(
        -1j * torch.pi * wavelength * distance * (FX**2 + FY**2)
    )

    U = fft.fft2(field)

    return fft.ifft2(U * Hf)

def circular_aperture(input_field, r_cutoff, pixel_pitch):

    """
    input_field: (H, W) or (B, H, W) or (B, C, H, W)
    r_cutoff: physical radius (meters)
    pixel_pitch: meters per pixel
    """

    if input_field.dim() == 2:
        input_field = input_field.unsqueeze(0)

    B, H, W = input_field.shape[:3]

    device = input_field.device
    dtype = input_field.dtype

    # coordinate grid in meters (centered)
    x = (torch.arange(W, device=device, dtype=dtype) - W / 2) * pixel_pitch
    y = (torch.arange(H, device=device, dtype=dtype) - H / 2) * pixel_pitch

    Y, X = torch.meshgrid(y, x, indexing="ij")

    r = torch.sqrt(X**2 + Y**2)  # (H, W)

    # default cutoff = full aperture
    if r_cutoff is None:
        r_cutoff = r.max()

    aperture = (r < r_cutoff).to(dtype)

    # broadcast to batch
    aperture = aperture.unsqueeze(0).expand(B, H, W)

    return input_field * aperture
