from collections.abc import Callable
from typing import Literal

import torch as T
from torch import Tensor, nn

from diffusion.helpers.loss import c_in, c_noise, c_out, c_skip


def build_sigma_schedule(
    steps: int,
    rho: int = 7,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0
) -> Tensor:
    """Build sigma schedule for model sampling.

    Args:
        steps (int): The size of the sequence of noise levels.
        rho (int): The log-scale factor between noise levels.
        sigma_min (int): The lower bound of the noise level interval.
        sigma_max (int): The upper bound of the noise level interval.

    Returns:
        Tensor: The sequence of noise levels.
    """

    min_inv_rho: float = sigma_min ** (1 / rho)  # pyright: ignore[reportAny]
    max_inv_rho: float = sigma_max ** (1 / rho)  # pyright: ignore[reportAny]

    return (
        max_inv_rho +
        T.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)
    ) ** rho

def sample_model(
    model: nn.Module,
    sigma_data: Tensor,
    n_samples: int,
    image_channels: int,
    image_size: int,
    device: Literal["gpu", "cpu"] = "cpu",
) -> Tensor:
    """Sample a model, using Euler's method.

    Args:
        model (nn.Module): The model to sample.
        sigma_data (Tensor): The standard deviation of the data distribution.
        n_samples (int): The number of samples to generate.
        image_channels (int): The number of channels of the image to generate.
        image_size (int): The image size of the image to generate.
        device (Literal["gpu", "cpu"]): The device on which to perform Tensor operations. Defaults to "cpu".

    Returns:
        Tensor: The generated samples.
    """
    sigma_data = sigma_data.to(device)
    model = model.to(device)
    _ = model.eval()

    # create sampling lambda
    # function by composing
    # noise level transforms
    # and model output
    D: Callable[[Tensor, Tensor], Tensor] = lambda x, sigma: (  # pyright: ignore[reportAny]
        (c_skip(sigma_data, sigma) * x) +
        (c_out(sigma_data, sigma) * model(c_in(sigma_data, sigma) * x, c_noise(sigma)))
    )

    # create sequence of
    # decreasing sigmas
    sigmas: Tensor = build_sigma_schedule(steps=50, rho=7).to(device)

    # initialize with pure
    # normal noise, and generate
    # using sampled schedule
    x: Tensor = T.randn(n_samples, image_channels, image_size, image_size) * sigmas[0]
    x = x.to(device)

    for i, sigma in enumerate(sigmas):
        x_denoised: Tensor | None = None
        with T.no_grad():
            x_denoised = D(x, sigma)

        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d: Tensor = (x - x_denoised) / sigma  # pyright: ignore[reportOperatorIssue]
        x = (
            x + d * (sigma_next - sigma)
        )

    return x
