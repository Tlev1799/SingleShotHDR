import torch
import torch.nn as nn

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights


class VGGLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1
        )

        self.features = (
            vgg.features[:16]
            .eval()
        )

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, pred, gt):

        f1 = self.features(pred)
        f2 = self.features(gt)

        return ((f1 - f2)**2).mean()