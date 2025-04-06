import os
import io
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A

from datasets import load_from_disk

def check_transform_lib(transform):
    module_name = transform.__class__.__module__
    if module_name.startswith('albumentations'):
        return 'albumentations'
    elif module_name.startswith('torchvision'):
        return 'torchvision'
    else:
        return 'unknown'

class ArrowDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataroot = data_root
        self.dataset = load_from_disk(data_root)

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

        with open(os.path.join(self.dataroot, 'mapping.json'), 'r') as f:
            mapping = json.load(f)
        self.mapping = {}
        for path, idx in mapping.items():
            img_full_path = os.path.join(self.dataroot, path)
            self.mapping[img_full_path] = idx


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
        index = self.mapping[img_path]
        example = self.dataset[index]
        image = Image.open(io.BytesIO(example['image'])).convert('RGB')
        label = self.labels[idx]
        if self.lib == 'albumentations':
            image = np.array(image)
            image = self.transform_chain(image=image)["image"].float()
        elif self.lib == 'torchvision':
            image = self.transform_chain(image)
        return image, label



class ArrowContrastDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataroot = data_root
        self.dataset = load_from_disk(data_root)

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

        with open(os.path.join(self.dataroot, 'mapping.json'), 'r') as f:
            mapping = json.load(f)
        self.mapping = {}
        for path, idx in mapping.items():
            img_full_path = os.path.join(self.dataroot, path)
            self.mapping[img_full_path] = idx

        from data import DataAugment
        self.clean_transform = transforms.Compose([
            transforms.RandomCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        self.noisy_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            DataAugment(
                blur_prob=1,
                blur_sig=[0.0, 3.0],
                jpg_prob=1,
                jpg_method=['cv2', 'pil'],
                jpg_qual=[30, 100]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])


    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        index = self.mapping[img_path]
        example = self.dataset[index]
        image = Image.open(io.BytesIO(example['image'])).convert('RGB')
        binary_label = self.labels[idx]
        clean_image = self.clean_transform(image)
        noisy_image = self.noisy_transform(image)
        return clean_image, noisy_image, binary_label


class ArrowTextImagePairDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataroot = data_root
        self.dataset = load_from_disk(data_root)

        self.split = split
        # self.image_pathes = []
        self.labels = []

        assert subset in ['real', 'fake'], "subset must be 'real' or 'fake'"
        if subset == 'real':
            label = 0
        else:
            label = 1

        with open(os.path.join(self.dataroot, 'mapping.json'), 'r') as f:
            mapping = json.load(f)

        json_file = os.path.join(self.dataroot, f'{split}.json')
        with open(json_file, 'r') as f:
            self.image_pathes = json.load(f)

        self.mapping = {}
        for path, idx in mapping.items():
            self.labels.append(label)
            # self.image_pathes.append(path)
            self.mapping[path] = idx

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
        index = self.mapping[img_path]
        example = self.dataset[index]
        image = Image.open(io.BytesIO(example['image'])).convert('RGB')
        label = self.labels[idx]
        if self.lib == 'albumentations':
            image = np.array(image)
            image = self.transform_chain(image=image)["image"].float()
        elif self.lib == 'torchvision':
            image = self.transform_chain(image)
        return image, label










#
#
#
# import json
# import random
#
# with open('./playground-v2.5-1024px-aesthetic/mapping.json', 'r') as file:
#     data = json.load(file)
#
# keys = data.keys()
#
# ids = []
# for k in keys:
#     ids.append(k.split('_')[0])
#
# test_ids = random.sample(ids, 3150)
#
# train_list = []
# test_list = []
#
# for ii in keys:
#     if ii.split('_')[0] in test_ids:
#         test_list.append(ii)
#     else:
#         train_list.append(ii)
#
# with open('./playground-v2.5-1024px-aesthetic/train.json', 'w') as train_file:
#     json.dump(train_list, train_file)
#
# with open('./playground-v2.5-1024px-aesthetic/test.json', 'w') as test_file:
#     json.dump(test_list[:3000], test_file)
#
#
# real_train_list = []
# real_test_list = []
#
# with open('./real/mapping.json', 'r') as file:
#     data = json.load(file)
#
# keys = data.keys()
#
# for ii in keys:
#     if ii.split('.')[0] in test_ids:
#         real_test_list.append(ii)
#     else:
#         real_train_list.append(ii)
#
#
# with open('./real/train.json', 'w') as train_file:
#     json.dump(real_train_list, train_file)
#
# with open('./real/test.json', 'w') as test_file:
#     json.dump(real_test_list[:3000], test_file)