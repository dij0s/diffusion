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

def euler_sample(
    x: Tensor,
    D: Callable[[Tensor, Tensor | int], Tensor],
    sigmas: tuple[Tensor, Tensor],
) -> Tensor:
    """Perform one step of Euler's method.

    Args:
        x (Tensor): The noised samples.
        D (Callable[[Tensor, Tensor | int], Tensor]): The denoising function.
        sigmas (tuple[Tensor, Tensor]): The tuple of contiguous sigmas to denoise the image with.

    Returns:
        Tensor: The denoised samples.
    """
    x_denoised: Tensor | None = None
    sigma, sigma_next = sigmas

    with T.no_grad():
        x_denoised = D(x, sigma)

        d: Tensor = (x - x_denoised) / sigma
        return (
            x + d * (sigma_next - sigma)
        )

def heun_sample(
    x: Tensor,
    D: Callable[[Tensor, Tensor | int], Tensor],
    sigmas: tuple[Tensor, Tensor],
) -> Tensor:
    """Perform one step of Heun's 2nd order method.

    Args:
        x (Tensor): The noised samples.
        D (Callable[[Tensor, Tensor | int], Tensor]): The denoising function.
        sigmas (tuple[Tensor, Tensor]): The tuple of contiguous sigmas to denoise the image with.

    Returns:
        Tensor: The denoised samples.
    """
    sigma, sigma_next = sigmas

    with T.no_grad():
        # slope at current
        # noise level given
        # euler's step
        d: Tensor = (x - D(x, sigma)) / sigma
        x_euler: Tensor = x + d * (sigma_next - sigma)

        # fallback to euler
        # when last step goes
        # to sigma=0 as no
        # slope is defined
        if sigma_next == 0:
            return x_euler

        # 2nd order correction
        # using trapezoidal rule
        d_next: Tensor = (x_euler - D(x_euler, sigma_next)) / sigma_next
        return x + (sigma_next - sigma) * (d + d_next) / 2

def sample_model(
    model: nn.Module,
    sigma_data: Tensor,
    n_samples: int,
    image_channels: int,
    image_size: int,
    step_fn: Callable[[Tensor, Callable[[Tensor, Tensor | int], Tensor], tuple[Tensor, Tensor]], Tensor] = euler_sample,
    denoising_steps: int = 50,
    device: Literal["gpu", "cpu"] = "cpu",
) -> Tensor:
    """Sample a model.

    Args:
        model (nn.Module): The model to sample.
        sigma_data (Tensor): The standard deviation of the data distribution.
        n_samples (int): The number of samples to generate.
        image_channels (int): The number of channels of the image to generate.
        image_size (int): The image size of the image to generate.
        step_fn (Callable[[Tensor, Callable[[Tensor, Tensor | int], Tensor], tuple[Tensor, Tensor]], Tensor]): The denoising step function. Defaults to Euler's method.
        denoising_steps (int): The number of denoising steps. Defaults to 50.
        device (Literal["gpu", "cpu"]): The device on which to perform Tensor operations. Defaults to "cpu".

    Returns:
        Tensor: The generated samples.
    """
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

    sigma_data = sigma_data.to(device)
    # create sequence of
    # decreasing sigmas
    sigmas: Tensor = build_sigma_schedule(steps=denoising_steps, rho=7).to(device)

    # initialize with pure
    # normal noise, and generate
    # using sampled schedule
    x: Tensor = T.randn(n_samples, image_channels, image_size, image_size).to(device)
    x = x * sigmas[0]

    for i, sigma in enumerate(sigmas):
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        x = step_fn(x, D, (sigma, sigma_next))  # pyright: ignore[reportArgumentType]

    return x
