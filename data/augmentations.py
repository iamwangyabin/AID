import io
from io import BytesIO
import cv2
import numbers
import numpy as np
from collections.abc import Sequence
from PIL import ImageFile, Image
from random import random, choice, randint
from scipy.fftpack import dct
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


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

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)
    return size

class RandomInterpolationResize(torch.nn.Module):
    def __init__(self, size, max_size=None, antialias=None):
        super().__init__()
        self.size = _setup_size(size, error_msg=" (h, w) as size.")
        self.interpolation = [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR,
                              transforms.InterpolationMode.BICUBIC, transforms.InterpolationMode.BOX,
                              transforms.InterpolationMode.HAMMING, transforms.InterpolationMode.LANCZOS,]
        if max_size is not None:
            if not (isinstance(max_size, int) and max_size > 0):
                raise ValueError("max_size must be an integer")
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, img):
        interpolation = random.choice(self.interpolation)
        return transforms.functional.resize(img, self.size, interpolation, self.max_size, self.antialias)






class DataAugment:
    def __init__(self, blur_prob, blur_sig, jpg_prob, jpg_method, jpg_qual):
        self.blur_prob = blur_prob
        self.blur_sig = blur_sig
        self.jpg_prob = jpg_prob
        self.jpg_method = jpg_method
        self.jpg_qual = jpg_qual

    def __call__(self, image):
        image = np.array(image)
        if random() < self.blur_prob:
            sig = sample_continuous(self.blur_sig)
            gaussian_blur(image, sig)

        if random() < self.jpg_prob:
            method = sample_discrete(self.jpg_method)
            qual = sample_discrete(self.jpg_qual)
            image = jpeg_from_key(image, qual, method)

        return Image.fromarray(image)


class DCTTransform:
    def __init__(self, mean_path, var_path, log_scale=True, epsilon=1e-12):
        self.log_scale = log_scale
        self.epsilon = epsilon

        # self.dct_mean = torch.load(mean_path).permute(1, 2, 0).numpy()
        # self.dct_var = torch.load(var_path).permute(1, 2, 0).numpy()
        self.dct_mean = np.load(mean_path)
        self.dct_var = np.load(var_path)

    def __call__(self, image):
        image = np.array(image)
        image = dct(image, type=2, norm="ortho", axis=0)
        image = dct(image, type=2, norm="ortho", axis=1)
        # log scale
        if self.log_scale:
            image = np.abs(image)
            image += self.epsilon  # no zero in log
            image = np.log(image)
        # normalize
        image = (image - self.dct_mean) / np.sqrt(self.dct_var)
        image = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float)
        return image


class RandomCompress:
    def __init__(self, method="JPEG", qf=[60, 100]):
        self.qf = qf
        self.method = method

    def __call__(self, image):
        outputIoStream = io.BytesIO()
        quality_factor = randint(int(self.qf[0]), int(self.qf[1]))
        image.save(outputIoStream, self.method, quality=quality_factor, optimize=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)


class Compress:
    def __init__(self, method="JPEG", qf=100):
        self.qf = qf
        self.method = method

    def __call__(self, image):
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, self.method, quality=self.qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)


def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class DCT_base_Rec_Module(nn.Module):
    """_summary_

    Args:
        x: [C, H, W] -> [C*level, output, output]
    """

    def __init__(self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]):
        super().__init__()

        assert output % window_size == 0
        assert len(level_fliter) > 0

        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)

        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1),
                                         requires_grad=False)

        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=stride
        )
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size),
            kernel_size=(window_size, window_size),
            stride=window_size
        )

        lm, mh = 2.82, 2
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]

        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList(
            [Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i + 1), norm=True) for i
             in range(grade_N)])

    def forward(self, x):

        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, W, H = x.shape
        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)

        _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size)
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=1)

        grade = torch.zeros(L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[1, 2, 3])
            grade += w * _x
            w *= k

        _, idx = torch.sort(grade)
        max_idx = torch.flip(idx, dims=[0])[:N]
        maxmax_idx = max_idx[0]
        if len(max_idx) == 1:
            maxmax_idx1 = max_idx[0]
        else:
            maxmax_idx1 = max_idx[1]

        min_idx = idx[:N]
        minmin_idx = idx[0]
        if len(min_idx) == 1:
            minmin_idx1 = idx[0]
        else:
            minmin_idx1 = idx[1]

        x_minmin = torch.index_select(level_x_unfold, 0, minmin_idx)
        x_maxmax = torch.index_select(level_x_unfold, 0, maxmax_idx)
        x_minmin1 = torch.index_select(level_x_unfold, 0, minmin_idx1)
        x_maxmax1 = torch.index_select(level_x_unfold, 0, maxmax_idx1)

        x_minmin = x_minmin.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)
        x_maxmax = x_maxmax.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)
        x_minmin1 = x_minmin1.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)
        x_maxmax1 = x_maxmax1.reshape(1, level_N * C * window_size * window_size).transpose(0, 1)

        x_minmin = self.fold0(x_minmin)
        x_maxmax = self.fold0(x_maxmax)
        x_minmin1 = self.fold0(x_minmin1)
        x_maxmax1 = self.fold0(x_maxmax1)

        return x_minmin, x_maxmax, x_minmin1, x_maxmax1


