import os
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import ImageFile, Image
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)



class DeepfakeMultiDatasets(Dataset):
    def __init__(self, opt, split='train'):
        self.dataroot = opt.dataroot
        self.split = split
        self.image_pathes = []
        self.labels = []
        self.label_mapping = {'0_real': 0, '1_fake': 1}
        image_extensions = ('.jpg', '.jpeg', '.png')

        if split == 'train':
            subfolder = opt.subfolder_names
            # classes = os.listdir(os.path.join(self.dataroot, subfolder, split))
            for id, cls in enumerate(opt.multicalss_names):
                root = os.path.join(self.dataroot, subfolder, split, cls)
                for label in ['0_real', '1_fake']:
                    label_dir = os.path.join(root, label)
                    for img_file in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_file)
                        if img_path.lower().endswith(image_extensions):
                            self.image_pathes.append(img_path)
                            self.labels.append(self.label_mapping[label] + id * 2)

            trsf = [
                transforms.Resize(opt.loadSize),
                transforms.RandomResizedCrop(opt.cropSize),
                transforms.RandomHorizontalFlip() if opt.random_flip else transforms.Lambda(lambda img: img),
                transforms.Lambda(lambda img: data_augment(img, opt)) if opt.augment else transforms.Lambda(lambda img: img),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]

        else:

            for id, subfolder in enumerate(opt.subfolder_names):
                if opt.multicalss_idx[id]:
                    classes = os.listdir(os.path.join(self.dataroot, subfolder, split))
                else:
                    classes = ['']
                for cls in classes:
                    root = os.path.join(self.dataroot, subfolder, split, cls)
                    for label in ['0_real', '1_fake']:
                        label_dir = os.path.join(root, label)
                        for img_file in os.listdir(label_dir):
                            img_path = os.path.join(label_dir, img_file)
                            if img_path.lower().endswith(image_extensions):
                                self.image_pathes.append(img_path)
                                self.labels.append(self.label_mapping[label] + id * 2)

            trsf = [
                transforms.Resize(opt.loadSize),
                transforms.CenterCrop(opt.cropSize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]

        self.transform_chain = transforms.Compose(trsf)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform_chain(image)
        return image, label

