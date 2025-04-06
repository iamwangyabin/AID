import os
import cv2
import random
import json
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations import Normalize
from data.albu_aug import FrequencyPatterns

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

def check_transform_lib(transform):
    module_name = transform.__class__.__module__
    if module_name.startswith('albumentations'):
        return 'albumentations'
    elif module_name.startswith('torchvision'):
        return 'torchvision'
    else:
        return 'unknown'

class BinaryJsonDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataroot = data_root
        self.split = split
        self.image_pathes = []
        self.labels = []

        json_file = os.path.join(self.dataroot, f'{split}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        for img_rel_path, label in data[subset].items():
            img_full_path = os.path.join(self.dataroot, img_rel_path)
            self.image_pathes.append(img_full_path)
            self.labels.append(label)

        for idx, transform in enumerate(trsf):
            self.lib = check_transform_lib(transform)

        if self.lib == 'albumentations':
            self.transform_chain = A.Compose(trsf)
        elif self.lib == 'torchvision':
            self.transform_chain = transforms.Compose(trsf)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        label = self.labels[idx]
        if self.lib == 'albumentations':
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.transform_chain(image=image)["image"].float()
        elif self.lib == 'torchvision':
            image = Image.open(img_path).convert('RGB')
            image = self.transform_chain(image)
        return image, label



class AIDEBinaryJsonDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataroot = data_root
        self.split = split
        self.image_pathes = []
        self.labels = []

        json_file = os.path.join(self.dataroot, f'{split}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        for img_rel_path, label in data[subset].items():
            img_full_path = os.path.join(self.dataroot, img_rel_path)
            self.image_pathes.append(img_full_path)
            self.labels.append(label)

        for idx, transform in enumerate(trsf):
            self.lib = check_transform_lib(transform)

        if self.lib == 'albumentations':
            self.transform_chain = A.Compose(trsf)
        elif self.lib == 'torchvision':
            self.transform_chain = transforms.Compose(trsf)

        from .augmentations import DCT_base_Rec_Module
        self.dct = DCT_base_Rec_Module()

        self.transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        label = self.labels[idx]
        if self.lib == 'albumentations':
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            image = self.transform_chain(image=image)["image"].float()
        elif self.lib == 'torchvision':
            image = Image.open(img_path).convert('RGB')
            image = self.transform_chain(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)

        x_0 = self.transform_train(image)
        x_minmin = self.transform_train(x_minmin)
        x_maxmax = self.transform_train(x_maxmax)

        x_minmin1 = self.transform_train(x_minmin1)
        x_maxmax1 = self.transform_train(x_maxmax1)

        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(label))



