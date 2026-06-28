import torch
import torch.nn.functional as F
import numpy as np
import imageio.v3 as iio
import cv2

# def psf_convolve_rgb(images, psfs):

#     # images: (B,3,H,W)
#     B, C, H, W = images.shape

#     out_channels = []
#     # import ipdb; ipdb.set_trace()

#     for c in range(3):

#         kernel = psfs[c].unsqueeze(0).unsqueeze(0)  # 1,1,H,W
#         kernel = kernel.expand(1, 1, H, W)

#         out_c = F.conv2d(
#             images[:, c:c+1],
#             kernel,
#             padding="same"
#         )

#         out_channels.append(out_c)

#     return torch.cat(out_channels, dim=1)

def psf_convolve_rgb(images, psfs):
    """
    Performs true linear convolution in the frequency domain via FFT,
    matching the paper's exact zero-padding and centering constraints.

    Args:
        images: (B, 3, H, W) linear HDR image tensor
        psfs: (3, H, W) sensor-resolution normalized PSF
    """
    B, C, H, W = images.shape

    # 1. Pad image boundaries to double size to completely prevent circular aliasing
    pad_top = H // 2
    pad_bottom = H // 2
    pad_left = W // 2
    pad_right = W // 2

    images_padded = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    H_pad, W_pad = images_padded.shape[-2:]

    # 2. Convert the PSF to an OTF matching the padded canvas size
    psfs_padded = F.pad(psfs, (0, W_pad - W, 0, H_pad - H), mode="constant", value=0.0)
    
    # CRITICAL: Roll the PSF so its center pixel anchors at (0,0) matching FFT origin assumptions
    psfs_shifted = torch.roll(psfs_padded, shifts=(-(H // 2), -(W // 2)), dims=(1, 2))
    otf = torch.fft.fft2(psfs_shifted)  # Shape: (3, H_pad, W_pad)

    # 3. Transform image to frequency domain
    img_fft = torch.fft.fft2(images_padded)  # Shape: (B, 3, H_pad, W_pad)

    # 4. Element-wise multiplication (OTF automatically broadcasts across the Batch dimension)
    result_fft = img_fft * otf.unsqueeze(0)

    # 5. Inverse FFT back to spatial domain
    result_padded = torch.fft.ifft2(result_fft).real

    # 6. Crop back to the original center window, cleanly discarding the padding
    result = result_padded[:, :, pad_top:pad_top + H, pad_left:pad_left + W]

    return result

def load_exr_image(file_path):
    """Loads a 32-bit floating-point HDR image from an EXR file,

    automatically converting OpenCV's default BGR format to RGB.
    """
    # IMREAD_UNCHANGED is critical to preserve the full float32 dynamic range
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(
            f"Failed to load EXR file: {file_path}. "
            "Ensure the file is not corrupt and opencv-python is properly installed."
        )

    # OpenCV reads images in BGR sequence by default; convert it to standard RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def load_and_resize_exr(file_path, target_size=(320, 320)):
    """Loads a 32-bit floating-point HDR image from an EXR file,

    resizes it to the target dimensions using area interpolation,
    and converts the channel sequence from BGR to RGB.
    """
    # IMREAD_UNCHANGED is critical to keep the full float32 dynamic range
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to load EXR file: {file_path}")

    # Resize the image. OpenCV expects (width, height) for target size
    # INTER_AREA is ideal for shrinking images without creating moiré or aliasing artifacts
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Convert from OpenCV's default BGR format to standard RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    return img_rgb

def load_and_resize_tiff(file_path, target_size=(320, 320)):
    """Loads a high-bit-depth or floating-point HDR image from a TIFF file,

    resizes it cleanly, and strictly normalizes it to a 3-channel RGB layout.
    """
    # IMREAD_UNCHANGED keeps the full float32/16 range and all channels intact
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to load TIFF file: {file_path}")

    # Resize using area interpolation to protect HDR point intensities from aliasing
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Inspect the channel layout and enforce 3-channel RGB
    if len(img_resized.shape) == 2:
        # If a grayscale image snuck into the dataset, replicate it across 3 channels
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
    elif len(img_resized.shape) == 3:
        num_channels = img_resized.shape[2]
        
        if num_channels == 4:
            # FIX: Convert BGRA -> RGB (drops the problematic alpha channel completely)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGB)
        elif num_channels == 3:
            # Standard BGR -> RGB conversion
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        elif num_channels == 1:
            # Grayscale trailing dimension -> RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            # Extreme fallback safeguard for unusual multi-spectral imagery
            img_rgb = img_resized[:, :, :3]
    else:
        raise ValueError(f"Unexpected array dimension setup {img.shape} found in: {file_path}")

    return img_rgb

def load_training_image(name_hdr):

    data = np.fromfile(name_hdr, dtype=np.float32)
    ss = len(data)
    
    if ss < 3:
        raise ValueError("Invalid HDR file")

    sz = np.floor(data[0:3]).astype(int)
    npix = sz[0]*sz[1]*sz[2]
    meta_length = ss - npix

    # Read binary HDR ground truth
    y = np.reshape(data[meta_length:meta_length+npix], (sz[0], sz[1], sz[2]))
    return y

# Read training data (HDR ground truth and LDR JPEG images)
def load_training_pair(name_hdr, name_jpg):

    data = np.fromfile(name_hdr, dtype=np.float32)
    ss = len(data)
    
    if ss < 3:
        return (False,0,0)

    sz = np.floor(data[0:3]).astype(int)
    npix = sz[0]*sz[1]*sz[2]
    meta_length = ss - npix

    # Read binary HDR ground truth
    y = np.reshape(data[meta_length:meta_length+npix], (sz[0], sz[1], sz[2]))

    # Read JPEG LDR image
    # x = scipy.misc.imread(name_jpg).astype(np.float32)/255.0

    # NOTE: This function was copied from original paper. Temporarily I return just the hdr image.
    return y # return (True, x, y)


def to_numpy_img(x):
    """
    x: (B,3,H,W) or (3,H,W)
    returns: (H,W,3) float32 numpy
    """
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float().permute(1, 2, 0)
    return x.numpy()


def save_png(path, img, exposure=0.0):
    img = np.squeeze(img)

    # exposure scaling (same idea as original)
    sc = np.sqrt(2.0 ** exposure)

    img = sc * img

    # clamp to display range
    img = np.clip(img, 0.0, 1.0)

    # convert to 8-bit
    img = (img * 255.0).astype(np.uint8)

    iio.imwrite(path, img)


def save_exr(path, img):
    """
    Save true HDR linear values (no clipping)
    """
    iio.imwrite(path, img.astype(np.float32), extension=".exr")
