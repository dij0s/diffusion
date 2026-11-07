from typing import Literal

import numpy as np
import torch as T
from torch import Tensor, nn
from torch.utils.data import DataLoader

from diffusion.helpers.loss import DenoisingLoss


def sample_sigma(
    n: int,
    loc: float = -1.2,
    scale: float = 1.2,
    sigma_min: float = 2e-3,
    sigma_max: float = 80
) -> Tensor:
    """Sample sigma within a log-normal distribution.

    Args:
        n (int): The output Tensor size.
        loc (float): The underlying Gaussian's distribution mean.
        scale (float): The underlying Gaussian's distribution standard deviation.
        sigma_min (float): The lower bound to which a sampled sigma is clipped.
        sigma_max (float): The upper bound to which a sampled sigma is clipped.

    Returns:
        Tensor: A Tensor of sampled sigmas.
    """

    return ((T.randn(n) * scale + loc)
        .exp()
        .clip(sigma_min, sigma_max))

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    sigma_data: Tensor,
    nb_epochs: int,
    eta: float,
    device: Literal["gpu", "cpu"] = "cpu",
):
    """Train a model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The train DataLoader.
        sigma_data (Tensor): The standard deviation of the data distribution.
        nb_epochs (int): The number of epochs to train the model for.
        eta (float): The learning rate.
        device (Literal["gpu", "cpu"]): The device on which to perform Tensor operations. Defaults to "cpu".

    Raises:
        Exception: If loss is NaN.
    """
    model = model.to(device)

    criterion: DenoisingLoss = DenoisingLoss(sigma_data).to(device)
    optimizer: T.optim.Optimizer = T.optim.SGD(model.parameters(), lr=eta)

    for e in range(nb_epochs):
        # sample a batch of images
        for y, target in iter(train_loader):
            y = y.to(device)

            # sample noise levels
            # and unsqueeze dimensions
            sigmas: Tensor = sample_sigma(n=y.shape[0]).to(device)
            sigmas = sigmas[:, None, None, None]
            # sample gaussian noise
            epsilons: Tensor = T.randn_like(y) * sigmas
            # degrade image
            x: Tensor = T.add(y, epsilons)

            # compute loss and
            # make backward pass
            loss: Tensor = criterion(
                model=model,
                x=x,
                sigma=sigmas,
                y=y
            )
            if np.isnan(loss.item()):
                raise Exception(f"Loss at step {e} is nan.")
            print(f"% loss@{e} {loss.item()}")

            model.zero_grad()
            loss.backward()
            optimizer.step()
