from typing import Optional

import torch
from torch import fft
from torch import linalg


def filter_image_frequencies(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = fft.fft2(image)
    image = fft.fftshift(image)

    filtered_image: torch.Tensor = image * mask
    residual_filtered_image: torch.Tensor = image * (1 - mask)

    filtered_image = fft.ifftshift(filtered_image)
    filtered_image = fft.ifft2(filtered_image).real
    residual_filtered_image = fft.ifftshift(residual_filtered_image)
    residual_filtered_image = fft.ifft2(residual_filtered_image).real

    return filtered_image, residual_filtered_image


def generate_circular_mask(
    input_size: int,
    mask_radius_start: int,
    mask_radius_stop: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    coordinates: torch.Tensor = generate_centered_2d_coordinates_grid(input_size, device)
    radius: torch.Tensor = linalg.vector_norm(coordinates, dim=-1)
    mask: torch.Tensor = torch.where(radius < mask_radius_start, 1, 0)
    if mask_radius_stop is not None:
        mask = torch.where(radius > mask_radius_stop, 1, mask)
    return mask


def generate_centered_2d_coordinates_grid(
    size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    assert size % 2 == 0, "Input size must be even."
    coords_values: torch.Tensor = torch.arange(0, size // 2, dtype=torch.float, device=device)
    coords_values = torch.cat([coords_values.flip(dims=(0,)), coords_values], dim=0)
    coordinates_x: torch.Tensor = coords_values.unsqueeze(dim=0).expand(size, -1)
    coordinates_y: torch.Tensor = torch.t(coordinates_x)
    coordinates: torch.Tensor = torch.stack([coordinates_x, coordinates_y], dim=2)
    return coordinates
