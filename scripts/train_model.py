import torch as T

from diffusion.helpers.data import load_dataset_and_make_dataloaders
from diffusion.models import UNet
from diffusion.pipeline.training import train_model

BATCH_SIZE: int = 8

dataloaders, info = load_dataset_and_make_dataloaders(
    dataset_name="FashionMNIST",
    root_dir="data",
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True
)
print(f"% dataset information {info}")

device: str = "cuda" if T.cuda.is_available() else "cpu"
print(f"% training on device '{device}'")

train_model(
    model=UNet(
        image_channels=info.image_channels,
        nb_channels=4,
        num_blocks=4,
        cond_channels=4
    ),
    train_loader=dataloaders.train,
    sigma_data=info.sigma_data,
    nb_epochs=1,
    eta=1e-2,
    device=device  # pyright: ignore[reportArgumentType]
)
