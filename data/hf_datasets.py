import os
import io
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A

from datasets import load_dataset

def check_transform_lib(transform):
    module_name = transform.__class__.__module__
    if module_name.startswith('albumentations'):
        return 'albumentations'
    elif module_name.startswith('torchvision'):
        return 'torchvision'
    else:
        return 'unknown'

class CC12MDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataset = load_dataset(data_root)[split]
        self.label = 0

        for idx, transform in enumerate(trsf):
            self.lib = check_transform_lib(transform)

        if self.lib == 'albumentations':
            self.transform_chain = A.Compose(trsf)
        elif self.lib == 'torchvision':
            self.transform_chain = transforms.Compose(trsf)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        # caption = item['txt']
        # with io.BytesIO(item['webp']) as buffer:
        #     image = Image.open(buffer).convert('RGB')
        #     if self.transform_chain:
        #         image = self.transform_chain(image)
        image = item['jpg'].convert('RGB')
        if self.transform_chain:
            image = self.transform_chain(image)

        del item
        return image, self.label


class SynthCLIPDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataset = load_dataset(data_root)[split]
        self.label = 1

        for idx, transform in enumerate(trsf):
            self.lib = check_transform_lib(transform)

        if self.lib == 'albumentations':
            self.transform_chain = A.Compose(trsf)
        elif self.lib == 'torchvision':
            self.transform_chain = transforms.Compose(trsf)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        # caption = item['txt']
        image = item['image'].convert('RGB')
        if self.transform_chain:
            image = self.transform_chain(image)
        del item
        return image, self.label




class HFIMLDDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataset = load_dataset(data_root)[subset]
        for idx, transform in enumerate(trsf):
            self.lib = check_transform_lib(transform)

        if self.lib == 'albumentations':
            self.transform_chain = A.Compose(trsf)
        elif self.lib == 'torchvision':
            self.transform_chain = transforms.Compose(trsf)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image'].convert('RGB')
        label = sample['label']

        if self.lib == 'albumentations':
            image = np.array(image)
            image = self.transform_chain(image=image)["image"].float()
        elif self.lib == 'torchvision':
            image = self.transform_chain(image)
        return image, label

