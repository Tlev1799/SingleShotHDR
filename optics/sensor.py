import torch


def area_downsample(psf, factor):
    """
    psf: (3, H, W)
    """

    C, H, W = psf.shape

    psf = psf.view(
        C,
        H // factor, factor,
        W // factor, factor
    )

    psf = psf.mean(dim=(2, 4))  # (3, H/f, W/f)

    return psf


def crop_psf(psf, patch_size):

    C, H, W = psf.shape

    offset_h = (H - patch_size) // 2
    offset_w = (W - patch_size) // 2

    return psf[:, 
               offset_h:offset_h + patch_size,
               offset_w:offset_w + patch_size]

def noramlize_psf(psf):
    return psf / psf.sum(dim=(1, 2), keepdim=True)


def sensor_model(psf, sampling_factor, patch_size):

    """
    Input:
        psf: (3, H, W) from lens()
    Output:
        psf: (3, patch_size, patch_size)
    """

    psf = area_downsample(psf, sampling_factor)
    psf = crop_psf(psf, patch_size)
    psf = noramlize_psf(psf)

    return psf

# ---------------------------------------------------------
# NEW: Sensor Noise and Clipping Simulation (Linear Space)
# ---------------------------------------------------------
def simulate_sensor_capture(blurred_linear, noise_std=0.0, min_val=1e-5, max_val=1.0):
    """
    Simulates physical sensor degradation by adding Gaussian read noise 
    and clipping sensor values directly in linear space.

    Args:
        blurred_linear: (B, C, H, W) input image from optical convolution
        noise_std: Standard deviation of Gaussian sensor read noise
        min_val: Bottom clipping floor to prevent non-differentiable 0s
        max_val: Top saturation capacity floor of the LDR sensor
    """
    # 1. Add zero-mean Gaussian read noise matching the paper's tf.random_normal
    noise = torch.randn_like(blurred_linear) * noise_std
    noisy_linear = blurred_linear + noise

    # 2. Clip the saturated pixels directly to the linear [1e-5, 1.0] LDR range
    ldr_output = torch.clamp(noisy_linear, min=min_val, max=max_val)
    
    return ldr_output