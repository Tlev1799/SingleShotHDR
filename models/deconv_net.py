import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------
# Bilinear kernel init (TF equivalent)
# -------------------------
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)

    for i in range(min(in_channels, out_channels)):
        weight[i, i, :, :] = filt

    return torch.from_numpy(weight)


# -------------------------
# Deconv layer (TF equivalent)
# -------------------------
class DeconvLayer(nn.Module):
    def __init__(self, cin, cout, alpha=0.0):
        super().__init__()

        # scale = 2
        # k = 2 * scale - scale % 2  # 4

        # self.deconv = nn.ConvTranspose2d(
        #     cin, cout,
        #     kernel_size=k,
        #     stride=scale,
        #     padding=1,
        #     bias=False
        # )
        self.deconv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)

        # TF bilinear initialization
        #self.deconv.weight.data = bilinear_kernel(cin, cout, k)

        self.bn = nn.BatchNorm2d(cout)
        self.alpha = alpha

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.deconv(x)
        x = self.bn(x)
        return torch.maximum(self.alpha * x, x)


# -------------------------
# Skip fusion layer (TF equivalent)
# -------------------------
class SkipConnectionLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)

        self._init_identity(channels)

    def _init_identity(self, c):
        w = torch.zeros((c, c * 2, 1, 1))

        for i in range(c):
            w[i, i, 0, 0] = 1.0          # decoder path
            w[i, i + c, 0, 0] = 1.0      # skip path

        self.fuse.weight.data = w
        self.fuse.bias.data.zero_()

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


# -------------------------
# Encoder (TF equivalent)
# -------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s0 = x

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        e5 = self.enc5(p4)

        return e5, [s0, e1, e2, e3, e4]


# -------------------------
# Decoder (TF equivalent)
# -------------------------
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = DeconvLayer(64, 64)
        self.up2 = DeconvLayer(64, 64)
        self.up3 = DeconvLayer(64, 64)
        self.up4 = DeconvLayer(64, 64)

        self.skip4 = SkipConnectionLayer(64)
        self.skip3 = SkipConnectionLayer(64)
        self.skip2 = SkipConnectionLayer(64)
        self.skip1 = SkipConnectionLayer(64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

        nn.init.xavier_uniform_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

        self.final_bn = nn.BatchNorm2d(3)
        self.final_skip = SkipConnectionLayer(3)

    def forward(self, x, skips):
        s0, e1, e2, e3, e4 = skips

        x = self.up1(x)

        x = self.skip4(x, e4)
        x = self.up2(x)

        x = self.skip3(x, e3)
        x = self.up3(x)

        x = self.skip2(x, e2)
        x = self.up4(x)

        x = self.skip1(x, e1)

        x = self.final_conv(x)
        x = self.final_bn(x)

        x = self.final_skip(x, s0)

        return x


# -------------------------
# Full model (TF equivalent)
# -------------------------
class ReconNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        return self.decoder(bottleneck, skips)
    