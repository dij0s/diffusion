from collections.abc import Callable
from typing import Literal

import torch as T
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from diffusion.pipeline.sampling import euler_sample, sample_model


def to_uint8_rgb(x: Tensor) -> Tensor:
    """Reshape and rescale an image into uint8 RGB.

    Args:
        x (Tensor): The input image of shape (B, C, H, W), and value range [-1,1].

    Returns:
        Tensor: The output image of shape (B, 3, H, W), and value range [0,255].
    """
    x = x.clamp(-1, 1).add(1).div(2).mul(255).round().to(T.uint8)
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


@T.no_grad()
def compute_fid(
    model: nn.Module,
    sigma_data: Tensor,
    loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    image_channels: int,
    image_size: int,
    n_samples: int = 2000,
    batch_size: int = 64,
    step_fn: Callable[[Tensor, Callable[[Tensor, Tensor | int], Tensor], tuple[Tensor, Tensor]], Tensor] = euler_sample,
    device: Literal["gpu", "cpu"] = "cpu",
) -> float:
    """Compute the FID between generated and real images.

    Args:
        model (nn.Module): The model to sample images from.
        sigma_data (Tensor): The standard deviation of the data distribution.
        loader (DataLoader): The DataLoader yielding real images to compare against.
        image_channels (int): The number of channels of the generated images.
        image_size (int): The height and width of the generated images.
        n_samples (int): The number of real and generated images used to estimate the statistics. Defaults to 64.
        batch_size (int): The number of images generated at once. Defaults to 64.
        step_fn (Callable[[Tensor, Callable[[Tensor, Tensor | int], Tensor], tuple[Tensor, Tensor]], Tensor]): The denoising step function. Defaults to Euler's method.
        device (Literal["gpu", "cpu"]): The device on which to perform Tensor operations. Defaults to "cpu".

    Returns:
        float: The FID, lower is better.
    """

    fid: FrechetInceptionDistance = FrechetInceptionDistance(feature=2048).to(device)

    # compute statistics on
    # real samples, then
    # generated statistics
    seen: int = 0
    for y, _ in iter(loader):
        fid.update(to_uint8_rgb(y.to(device)), real=True)  # pyright: ignore[reportAny]
        seen += y.shape[0]
        if seen >= n_samples:
            break

    generated: int = 0
    while generated < n_samples:
        b: int = min(batch_size, n_samples - generated)
        x: Tensor = sample_model(  # pyright: ignore[reportAssignmentType]
            model=model,
            sigma_data=sigma_data,
            n_samples=b,
            image_channels=image_channels,
            image_size=image_size,
            step_fn=step_fn,
            device=device,
        )
        fid.update(to_uint8_rgb(x), real=False)
        generated += b

    return float(fid.compute())
