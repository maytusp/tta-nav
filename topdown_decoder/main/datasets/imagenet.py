import lmdb
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO

class miniImageNet(Dataset):
    def __init__(
        self,
        root,
        norm=True,
        transform=None,
    ):
        self.root = root
        self.transform = transform
        self.norm = norm
        self.lst = os.listdir(self.root)
        self.lst.sort()
        self.images = [
            os.path.join(root, img)
            for img in self.lst
            if img.endswith(".jpg")
        ]

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.transform is not None:
            img = self.transform(img)
            

        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0
        try:
            return torch.from_numpy(img).permute(2, 0, 1).float()
        except:
            print(img.shape)

    def __len__(self):
        return len(self.images)
