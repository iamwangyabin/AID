import csv
import os
import warnings

import albumentations as A
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


def check_transform_lib(transform):
    module_name = transform.__class__.__module__
    if module_name.startswith("albumentations"):
        return "albumentations"
    if module_name.startswith("torchvision"):
        return "torchvision"
    return "unknown"


BFREE_FAKE_FOLDERS = {
    "selfconditioned": ["SD2.1_selfconditioned"],
    "selfconditioned_origbg": ["SD2.1_selfconditioned_origBG"],
    "inpainted_samecat": ["SD2.1_inpainted_samecat"],
    "inpainted_samecat_origbg": ["SD2.1_inpainted_samecat_origBG"],
    "inpainted_diffcat": ["SD2.1_inpainted_diffcat"],
    "inpainted_diffcat_origbg": ["SD2.1_inpainted_diffcat_origBG"],
}
BFREE_FAKE_FOLDERS["all"] = [folder for folders in BFREE_FAKE_FOLDERS.values() for folder in folders]


class BFreeTrainingDataset(Dataset):
    """
    Dataset loader for the official B-Free training data release.

    Expected layout:
      data_root/train_list.csv
      data_root/valid_list.csv
      data_root/COCO_real_512/<id>.png
      data_root/SD2.1_selfconditioned/<id>.png
      ...
    """

    def __init__(self, data_root, trsf, subset="all", split="train"):
        self.dataroot = data_root
        self.split = split
        self.subset = subset or "all"
        self.image_pathes = []
        self.labels = []

        ids = self._read_ids(self._resolve_split_file(split))
        fake_folders = self._resolve_fake_folders(self.subset)
        for image_id in ids:
            self.image_pathes.append(os.path.join(self.dataroot, "COCO_real_512", f"{image_id}.png"))
            self.labels.append(0)
            for folder in fake_folders:
                self.image_pathes.append(os.path.join(self.dataroot, folder, f"{image_id}.png"))
                self.labels.append(1)

        missing = [path for path in self.image_pathes[:256] if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "B-Free dataset layout check failed. First missing file: "
                f"{missing[0]}. Expected official folders under {self.dataroot}."
            )

        self.lib = "torchvision"
        for transform in trsf:
            self.lib = check_transform_lib(transform)

        if self.lib == "albumentations":
            self.transform_chain = A.Compose(trsf)
        elif self.lib == "torchvision":
            self.transform_chain = transforms.Compose(trsf)
        else:
            raise ValueError("BFreeTrainingDataset supports torchvision or albumentations transforms.")

    def _resolve_split_file(self, split):
        split_name = "valid" if split in {"val", "valid", "validation"} else split
        split_file = os.path.join(self.dataroot, f"{split_name}_list.csv")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"B-Free split file not found: {split_file}")
        return split_file

    def _read_ids(self, split_file):
        with open(split_file, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "id" not in reader.fieldnames:
                raise ValueError(f"B-Free split file must contain an 'id' column: {split_file}")
            return [row["id"] for row in reader]

    def _resolve_fake_folders(self, subset):
        key = str(subset).lower()
        if key in BFREE_FAKE_FOLDERS:
            return BFREE_FAKE_FOLDERS[key]
        if subset in BFREE_FAKE_FOLDERS["all"]:
            return [subset]
        raise ValueError(
            f"Unsupported B-Free subset '{subset}'. Use one of "
            f"{sorted(BFREE_FAKE_FOLDERS.keys())} or an official fake folder name."
        )

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        image = Image.open(self.image_pathes[idx]).convert("RGB")
        label = self.labels[idx]
        if self.lib == "albumentations":
            image = np.array(image)
            image = self.transform_chain(image=image)["image"].float()
        else:
            image = self.transform_chain(image)
        return image, label
