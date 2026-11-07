from typing import Literal

import numpy as np
import torch as T
from torch import Tensor, nn
from torch.utils.data import DataLoader

import wandb
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
    train_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    valid_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    sigma_data: Tensor,
    nb_epochs: int,
    eta: float,
    run: wandb.Run,
    valid_loss_every: int = 100,
    device: Literal["gpu", "cpu"] = "cpu",
) -> nn.Module:
    """Train a model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The train DataLoader.
        valid_loader (DataLoader): The validation DataLoader.
        sigma_data (Tensor): The standard deviation of the data distribution.
        nb_epochs (int): The number of epochs to train the model for.
        eta (float): The learning rate.
        run (wandb.Run): The wandb-associated experiment run.
        valid_loss_every (int): The interval at which the validation loss should be computed, and logged. Defaults to 100.
        device (Literal["gpu", "cpu"]): The device on which to perform Tensor operations. Defaults to "cpu".

    Returns:
        nn.Module: The trained model.

    Raises:
        Exception: If loss is NaN.
    """
    model = model.to(device)

    criterion: DenoisingLoss = DenoisingLoss(sigma_data).to(device)
    optimizer: T.optim.Optimizer = T.optim.SGD(model.parameters(), lr=eta)

    for e in range(nb_epochs):
        loss_train: float = 0.0
        compute_valid_loss: bool = e % valid_loss_every == 0

        # sample a batch of images
        for y, _ in iter(train_loader):
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
            loss_train += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

        # compute average train
        # loss across batches
        average_loss_train: float = loss_train / len(train_loader)
        run.log({"train/loss": average_loss_train}, step=e+1, commit=(not compute_valid_loss))

        # compute validation loss
        # on 20% of samples
        if compute_valid_loss:
            print(f"% computing valid loss@{e}")
            iterations: int = len(valid_loader) // 5
            loss_valid: float = 0.0
            iterator = iter(valid_loader)
            with T.no_grad():
                for _ in range(iterations):
                    y_valid, _ = next(iterator)
                    y_valid = y_valid.to(device)

                    sigmas_valid: Tensor = sample_sigma(n=y_valid.shape[0]).to(device)
                    sigmas_valid = sigmas_valid[:, None, None, None]
                    # sample gaussian noise
                    epsilons_valid: Tensor = T.randn_like(y_valid) * sigmas_valid
                    # degrade image
                    x_valid: Tensor = T.add(y_valid, epsilons_valid)

                    # compute loss and
                    # make backward pass
                    loss_valid += criterion(  # pyright: ignore[reportAny]
                        model=model,
                        x=x_valid,
                        sigma=sigmas_valid,
                        y=y_valid
                    ).item()

            # log average loss
            # on valid subset
            average_loss_valid: float = loss_valid / iterations
            run.log({"valid/loss": average_loss_valid}, step=e+1, commit=True)

    return model
