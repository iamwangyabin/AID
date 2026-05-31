import random
from io import BytesIO

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def _pil_jpeg_in_memory(img, quality):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    out = Image.open(buffer).convert("RGB")
    out.load()
    return out


def _resize_random_interp(img, size):
    interpolation = random.choice([InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
    return F.resize(img, size, interpolation=interpolation)


class GAPLRandomStateAugmentation(torch.nn.Module):
    """GAPL-style random state augmentation used in the official stage-2 dataloader."""

    def __init__(
        self,
        resize_size=256,
        crop_size=224,
        auglist="JPEGinMemory,RandomResizeWithRandomIntpl,RandomCrop,RandomHorizontalFlip,RandomVerticalFlip,RRCWithRandomIntpl,RandomRotation,RandomTranslate,RandomShear,RandomPadding,RandomCutout",
        min_augs="0",
        max_augs="2",
    ):
        super().__init__()
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.aug_names = [name.strip() for name in auglist.split(",") if name.strip()]
        self.min_augs = self._parse_counts(min_augs)
        self.max_augs = self._parse_counts(max_augs)

    def _parse_counts(self, value):
        values = [int(v) for v in str(value).split(",")]
        if len(values) == 1:
            return values * len(self.aug_names)
        if len(values) != len(self.aug_names):
            raise ValueError("GAPLRandomStateAugmentation count list must match auglist length.")
        return values

    def _apply_one(self, img, name):
        if name == "JPEGinMemory":
            return _pil_jpeg_in_memory(img, random.randint(75, 100))
        if name == "RandomResizeWithRandomIntpl":
            max_size = round(self.crop_size * 1.228)
            return _resize_random_interp(img, random.randint(self.crop_size + 1, max_size))
        if name == "RandomCrop":
            return transforms.RandomCrop(self.crop_size)(img)
        if name == "RandomHorizontalFlip":
            return F.hflip(img) if random.random() < 0.5 else img
        if name == "RandomVerticalFlip":
            return F.vflip(img) if random.random() < 0.5 else img
        if name == "RRCWithRandomIntpl":
            interpolation = random.choice(
                [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC, InterpolationMode.LANCZOS]
            )
            return transforms.RandomResizedCrop(
                self.crop_size,
                scale=(0.9, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=interpolation,
            )(img)
        if name == "RandomRotation":
            return transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR)(img)
        if name == "RandomTranslate":
            return transforms.RandomAffine(0, translate=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR)(img)
        if name == "RandomShear":
            return transforms.RandomAffine(
                0,
                shear=(-15, 15, -15, 15),
                interpolation=InterpolationMode.BILINEAR,
            )(img)
        if name in {"RandomPadding", "RandomPaddingAndResize"}:
            width, height = img.size
            pad = max(1, int(min(width, height) * 0.1))
            fill = random.randint(0, 255)
            img = ImageOps.expand(img, border=pad, fill=(fill, fill, fill))
            return F.resize(img, self.resize_size, interpolation=InterpolationMode.BILINEAR)
        if name == "RandomCutout":
            width, height = img.size
            area = width * height
            cut_area = random.uniform(0.02, 0.06) * area
            aspect = random.uniform(0.3, 3.3)
            cut_w = max(1, min(width, int((cut_area * aspect) ** 0.5)))
            cut_h = max(1, min(height, int((cut_area / aspect) ** 0.5)))
            left = random.randint(0, max(0, width - cut_w))
            top = random.randint(0, max(0, height - cut_h))
            fill = random.randint(0, 255)
            img = img.copy()
            ImageDraw.Draw(img).rectangle([left, top, left + cut_w, top + cut_h], fill=(fill, fill, fill))
            return img
        return img

    def forward(self, img):
        if not F._is_pil_image(img):
            raise TypeError("GAPLRandomStateAugmentation expects a PIL image.")

        order = []
        for idx, name in enumerate(self.aug_names):
            order.extend([name] * random.randint(self.min_augs[idx], self.max_augs[idx]))
        random.shuffle(order)

        for name in order:
            img = self._apply_one(img, name)
        return img
