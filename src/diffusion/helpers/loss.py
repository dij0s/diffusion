from collections.abc import Callable
from typing import override

import torch as T
import torch.nn.modules.loss as l
from torch import Tensor, nn


def c_in(sigma_data: Tensor, sigma: Tensor) -> Tensor:
    """Compute network input scale coefficients.

    Args:
        sigma_data (Tensor): The standard deviation of the data distribution.
        sigma (Tensor): The noise level used to degrade sample image.

    Returns:
        Tensor: The network input scale coefficients.
    """
    return 1 / T.sqrt(T.pow(sigma_data, 2) + T.pow(sigma, 2))

def c_out(sigma_data: Tensor, sigma: Tensor) -> Tensor:
    """Compute network output scale coefficients.

    Args:
        sigma_data (Tensor): The standard deviation of the data distribution.
        sigma (Tensor): The noise level used to degrade sample image.

    Returns:
        Tensor: The network output scale coefficients.
    """
    return T.mul(sigma, sigma_data) / T.sqrt(T.pow(sigma_data, 2) + T.pow(sigma, 2))

def c_skip(sigma_data: Tensor, sigma: Tensor) -> Tensor:
    """Compute skip connection coefficient.

    Args:
        sigma_data (Tensor): The standard deviation of the data distribution.
        sigma (Tensor): The noise level used to degrade sample image.

    Returns:
        Tensor: The skip connection coefficient.
    """
    return T.pow(sigma_data, 2) / (T.pow(sigma_data, 2) + T.pow(sigma, 2))

def c_noise(sigma: Tensor) -> Tensor:
    """Transform noise level.

    Args:
        sigma (Tensor): The noise level used to degrade sample image.

    Returns:
        Tensor: The transformed noise level.
    """
    return T.log(sigma) / 4

class DenoisingLoss(l._Loss):  # pyright: ignore[reportPrivateUsage]
    """This criterion computes a variation of the Mean Squared Error loss, for an image diffusion task.

    Args:
        sigma_data (Tensor): The standard deviation of the data distribution.
    """
    def __init__(self, sigma_data: Tensor):
        super().__init__(size_average=None, reduce=None, reduction="mean")
        # register sigma_data as
        # part of the module state
        self.register_buffer("sigma_data", sigma_data.detach().clone())

        # parametrize noise level
        # function with data
        # distribution st. dev.
        self.c_in: Callable[[Tensor], Tensor] = lambda sigma: c_in(sigma_data, sigma)
        self.c_out: Callable[[Tensor], Tensor] = lambda sigma: c_out(sigma_data, sigma)
        self.c_skip: Callable[[Tensor], Tensor] = lambda sigma: c_skip(sigma_data, sigma)
        self.c_noise: Callable[[Tensor], Tensor] = c_noise

    @override
    def forward(self, model: nn.Module, x: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
        """Run the forward pass.

        Args:
            model (nn.Module): The model to compute the output from.
            x (Tensor): The input degraded image.
            sigma (Tensor): The sampled noise level.
            y (Tensor): The target original image.

        Returns:
            Tensor: The resulting denoising loss.
        """
        output: Tensor = model(
            T.mul(self.c_in(sigma), x),
            self.c_noise(sigma)
        )
        target: Tensor = T.div(
            T.sub(y, T.mul(
                self.c_skip(sigma),
                x
            )),
            self.c_out(sigma)
        )

        return T.mean(T.pow(output - target, 2))
