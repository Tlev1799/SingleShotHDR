from pathlib import Path

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

class HDRTestDataset(Dataset):

    def __init__(self, root):
        # Scans for both standard .tif and .tiff extensions
        root_path = Path(root)
        self.files = sorted(list(root_path.rglob("*.exr")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = str(self.files[idx])
        
        # Load, resize, and convert to RGB
        img = img_ops.load_exr_image(file_path)

        # Convert the float32/float16 numpy array to a PyTorch tensor
        img = torch.from_numpy(img).float()

        # Permute from HWC (Height, Width, Channels) to CHW for PyTorch
        img = img.permute(2, 0, 1)

        return img