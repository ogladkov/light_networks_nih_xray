import cv2
import numpy as np
import os
from pathlib import Path
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset


# Data augmentation and preprocessing using Albumentations
def get_transforms(cfg):
    train_transforms = A.Compose([
        A.Resize(cfg.width, cfg.height),
        A.HorizontalFlip(),
        # A.Rotate(limit=20),  # Rotate up to 20 degrees
        # A.RandomBrightnessContrast(p=0.2),
        # A.GaussianBlur(p=0.1),  # Apply Gaussian blur with a 10% probability
        A.Normalize(),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(cfg.width, cfg.height),  # DenseNet-121 expects 224x224 images
        A.Normalize(),
        ToTensorV2(),
    ])

    return train_transforms, val_transforms
class MNIHDataset(Dataset):

    def __init__(self, cfg, train=True, transforms=None):
        self.dataset_root = Path(cfg.root_path)
        self.train = train
        self.transforms = transforms

        if train:
            self.dataset_cls_path = self.dataset_root / 'train'

        else:
            self.dataset_cls_path = self.dataset_root / 'val'

        self.data, self.cls_weights, self.cls2label = self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, cls = self.data[idx]

        image = cv2.imread(img_path)
        image = image[..., ::-1]

        label = self.cls2label[cls]

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, label

    def get_data(self):
        # Gather fnames and class names
        data = []
        cls_cntr = {cls: 0 for cls in os.listdir(self.dataset_cls_path)}

        for cls in os.listdir(self.dataset_cls_path):

            for fname in os.listdir(self.dataset_cls_path / cls):
                fpath = str(self.dataset_cls_path / cls / fname)
                data.append([fpath, cls])
                cls_cntr[cls] += 1

        # Permute if train
        if self.train:
            random.shuffle(data)

        # Get weights of classes
        total_samples = np.sum([cls_cntr[cls] for cls in cls_cntr.keys()])
        cls_weights = [1 / (cls_cntr[cls] / total_samples) for cls in cls_cntr.keys()]

        # Get classes name-label dict
        cls2label = {}
        for n, cls in enumerate(cls_cntr.keys()):
            cls2label[cls] = n

        return data, torch.Tensor(cls_weights), cls2label

if __name__ == '__main__':
    cfg = OmegaConf.load('./config.yml')

    train_dataset = MNIHDataset(cfg.dataset)
