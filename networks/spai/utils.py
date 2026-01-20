import torch
from torch.nn import functional as F


def patchify_image(
    x: torch.Tensor,
    patch_size: tuple[int, int],
    patch_stride: tuple[int, int],
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("Expected input of shape B x C x H x W")

    kernel_h, kernel_w = patch_size
    stride_h, stride_w = patch_stride

    patches = F.unfold(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
    b, c_kk, l = patches.shape
    c = x.size(1)
    patches = patches.transpose(1, 2).contiguous()
    patches = patches.view(b, l, c, kernel_h, kernel_w)
    return patches
