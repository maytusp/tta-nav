import lmdb
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
robustnav_corruption_types = ["Defocus Blur", "Lighting", "Spatter", "Speckle Noise", "Motion Blur"]
albumentations_corrupton_types = ["Jitter", "Glare", "Light Out", "Shadow", "Snow", "Rain", "Fog", "Occlusion",
"Zoom Blur",  "Glass Blur"]


class GibsonLmdbDataset(Dataset):
    def __init__(
        self,
        root,
        norm=True,
        split="train",
        transform=None,
        image_size=256,
        original_resolution=256,
    ):
        self.transform = transform
        self.env = lmdb.open(
            root,
            readonly=True,
            max_readers=32,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.norm = norm
        self.original_resolution = original_resolution
        self.image_size = image_size

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        if split is None:
            self.offset = 0
        elif split == "train":
            # last 28k
            self.length = self.length - 2000
            self.offset = 2000
        elif split == "test":
            # first 2k
            self.length = 2000
            self.offset = 0
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        index = index + self.offset

        with self.env.begin(write=False) as txn:
            key = f"{self.original_resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)

        if self.transform is not None:
            img = self.transform(img)

        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0

        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return self.length


class GibsonDataset(Dataset):
    def __init__(
        self,
        root,
        norm=True,
        transform=None,
        corruption_type=None,
        severity=None,
        return_orig=False,
    ):
        self.root = root
        self.transform = transform
        self.corruption_type = corruption_type
        self.norm = norm
        self.lst = os.listdir(self.root)
        self.lst.sort()
        self.images = [
            os.path.join(root, img)
            for img in self.lst
            if img.endswith(".png")
        ]
        self.return_orig = return_orig
        if corruption_type is not None:
            if corruption_type in robustnav_corruption_types:
                from corruptions.my_benchmark import corruptions
            elif corruption_type in albumentations_corrupton_types:
                from corruptions.albumentations_benchmark import corruptions
            else:
                print(f"The corruption named {corruption_type} is not in the list")
            self.corruptions = corruptions(corruption_type, severity)
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        orig_img = img
        if self.corruption_type is not None:
            img = self.corruptions.apply(img)
            
            
        if self.transform is not None:
            img = self.transform(img)
            orig_img = self.transform(orig_img)

        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
            orig_img = (np.asarray(orig_img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0
            orig_img = np.asarray(orig_img).astype(np.float) / 255.0
        if not(self.return_orig):
            return torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            return torch.from_numpy(orig_img).permute(2, 0, 1).float(), torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.images)
