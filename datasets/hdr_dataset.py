from pathlib import Path

import cv2
import torch

import os
import utils.image_ops as img_ops

from torch.utils.data import Dataset

class HDRDataset(Dataset):

    def __init__(self, root):

        self.files = sorted(list(Path(root).glob("*.bin")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        #jpg_path = self.files[idx].with_suffix(".jpg")
        img = img_ops.load_training_image(self.files[idx])

        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)

        return img