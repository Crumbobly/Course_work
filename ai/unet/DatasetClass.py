import os
from PIL import Image
import numpy as np
import torch
from torch.onnx.symbolic_opset9 import tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform
        self.mask_resize = T.Resize((15, 1000), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.replace(".jpg", "_mask.png"))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.mask_resize(image)  # тот же размер, что и у маски
        if self.transform:
            image = self.transform(image)

        mask = self.mask_resize(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
