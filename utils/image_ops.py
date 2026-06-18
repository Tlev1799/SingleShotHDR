import torch
import torch.nn.functional as F
import numpy as np


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