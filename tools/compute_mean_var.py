# borrow from FreDect
# compute mean and var for DCT operatoin
# this method needs to compute mean & var before training and testing
import os
import torch
import argparse
import numpy as np
from PIL import Image
from scipy import fftpack
from tqdm import tqdm

def _welford_update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    if count is None:
        count, mean, M2 = 0, np.zeros_like(new_value), np.zeros_like(new_value)

    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2

    return (count, mean, M2)


def _welford_finalize(existing_aggregate):
    count, mean, M2 = existing_aggregate
    mean, variance, sample_variance = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return (float("nan"), float("nan"), float("nan"))
    else:
        return (mean, variance, sample_variance)


def welford(sample):
    """Calculates the mean, variance and sample variance along the first axis of an array.
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    existing_aggregate = (None, None, None)
    for data in sample:
        existing_aggregate = _welford_update(existing_aggregate, data)

    # sample variance only for calculation
    return _welford_finalize(existing_aggregate)[:-1]

def find_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                yield os.path.join(root, file)

def dct2(array):
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array


def scale_image(image):
    if not image.flags.writeable:
        image = np.copy(image)

    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    image /= 127.5
    image -= 1.
    return image

def log_scale(array, epsilon=1e-12):
    """Log scale the input array.
    """
    array = np.abs(array)
    array += epsilon  # no zero in log
    array = np.log(array)
    return array

def load_image(path):
    x = Image.open(path).convert('RGB')
    width, height = x.size
    new_width, new_height = 224, 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    x = x.crop((left, top, right, bottom))
    return np.asarray(x)

def main(args):
    paths = list(find_images(args.input))

    existing_aggregate = (None, None, None)
    # images = []
    for path in tqdm(paths):
        image = load_image(path)
        image = dct2(image)
        image = log_scale(image)
        # images.append(image)
        existing_aggregate = _welford_update(existing_aggregate, image)


    mean, var = _welford_finalize(existing_aggregate)[:-1]

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "mean"), mean)
    np.save(os.path.join(args.output, "var"), var)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DCT mean and variance of images in a directory")
    parser.add_argument("--input", type=str, help="Input directory containing images")
    parser.add_argument("--output", type=str, help="Output directory to save the mean and variance")
    args = parser.parse_args()

    main(args)



