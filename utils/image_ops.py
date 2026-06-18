import torch
import torch.nn.functional as F
import numpy as np
import imageio.v3 as iio


def psf_convolve_rgb(images, psfs):

    # images: (B,3,H,W)
    B, C, H, W = images.shape

    out_channels = []
    # import ipdb; ipdb.set_trace()

    for c in range(3):

        kernel = psfs[c].unsqueeze(0).unsqueeze(0)  # 1,1,H,W
        kernel = kernel.expand(1, 1, H, W)

        out_c = F.conv2d(
            images[:, c:c+1],
            kernel,
            padding="same"
        )

        out_channels.append(out_c)

    return torch.cat(out_channels, dim=1)

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


def get_final_image(network_output, x_in, thr=0.05):
    """
    PyTorch equivalent of TensorFlow get_final().

    Args:
        network_output: CNN output (restoration prediction)
        x_in: input blurred image (same scale as training input)
        thr: highlight threshold
    """

    # network prediction
    y_predict = network_output

    # highlight mask (max over channels)
    alpha = torch.amax(x_in, dim=1, keepdim=True)  # (B,1,H,W)

    alpha = torch.clamp((alpha - 1.0 + thr) / thr, 0.0, 1.0)
    alpha = alpha.repeat(1, 3, 1, 1)

    # "linearized" input
    x_lin = x_in ** 2

    # exp mapping (HDR reconstruction space)
    y_predict = torch.exp(y_predict) - 1.0 / 255.0

    # alpha blend
    y_final = (1 - alpha) * x_lin + alpha * y_predict

    return y_final