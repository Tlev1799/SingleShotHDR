import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt

def read_exr(path):
    exr_file = OpenEXR.InputFile(path)

    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define pixel type (float32 is most common)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read channels (usually R, G, B)
    r = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
    g = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32)
    b = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32)

    # Reshape
    r = r.reshape(height, width)
    g = g.reshape(height, width)
    b = b.reshape(height, width)

    # Stack into image (H, W, 3)
    img = np.stack([r, g, b], axis=-1)

    return img

import os
print(os.getcwd())
out = read_exr("my_first_hdr_img.exr")
print(out.shape, out.dtype)

lo, hi = np.percentile(out, [1, 99])
vis = np.clip((out - lo) / (hi - lo), 0, 1)
print(vis.shape)

print(vis.min())
print(vis.max())
print(vis.mean())
print(vis.median())

plt.imshow(vis)
plt.savefig('extracted_img.png')

