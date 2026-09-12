from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import utils.image_ops as img_ops

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
        root_path = Path(root)
        self.files = sorted(list(root_path.rglob("*.exr")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = str(self.files[idx])
        img = img_ops.load_exr_image(file_path)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img