import os
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

class BinaryDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataroot = data_root
        self.split = split
        self.image_pathes = []
        self.labels = []
        self.label_mapping = {'0_real': 0, '1_fake': 1}
        image_extensions = ('.jpg', '.jpeg', '.png')

        root = os.path.join(self.dataroot, split, subset)
        for label in ['0_real', '1_fake']:
            label_dir = os.path.join(root, label)
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                if img_path.lower().endswith(image_extensions):
                    self.image_pathes.append(img_path)
                    self.labels.append(self.label_mapping[label])

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
            image = self.transform_chain(image=image)["image"].float()
        elif self.lib == 'torchvision':
            image = Image.open(img_path).convert('RGB')
            image = self.transform_chain(image)
        return image, label


