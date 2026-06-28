import OpenEXR
import Imath
import numpy as np

def load_exr_rgb(path):
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    dw = header['dataWindow']

    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    r, g, b = exr.channels(["R", "G", "B"], pt)

    r = np.frombuffer(r, dtype=np.float32).reshape(h, w)
    g = np.frombuffer(g, dtype=np.float32).reshape(h, w)
    b = np.frombuffer(b, dtype=np.float32).reshape(h, w)

    return r, g, b


def stats(arr):
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }


r, g, b = load_exr_rgb("DeepOpticsHDR_PyTorch/Disco_1920x1080p_30_00002.exr")

print("R channel:", stats(r))
print("G channel:", stats(g))
print("B channel:", stats(b))

# Combined stats (all channels together)
all_pixels = np.stack([r, g, b], axis=-1)

print("\nOverall image:")
print({
    "min": float(np.min(all_pixels)),
    "max": float(np.max(all_pixels)),
    "mean": float(np.mean(all_pixels)),
    "median": float(np.median(all_pixels)),
})