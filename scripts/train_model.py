import argparse
from functools import reduce
from pathlib import Path

import torch as T
import yaml
from PIL import Image
from torchvision.utils import make_grid

import wandb
from diffusion.helpers.data import load_dataset_and_make_dataloaders
from diffusion.models import UNet
from diffusion.pipeline.sampling import sample_model
from diffusion.pipeline.training import train_model


def main(run: wandb.Run, config: dict[str, dict[str, str] | str]):
    """Train and evaluate diffusion model.

    Args:
        run (wandb.Run): The wandb-associated experiment run.
        config (dict[str, dict[str, str] | str]): The configuration of the experiment.
    """
    dataloaders, info = load_dataset_and_make_dataloaders(
        dataset_name="FashionMNIST",
        root_dir="data",
        batch_size=config["batch_size"],  # pyright: ignore[reportArgumentType]
        num_workers=4,
        pin_memory=True
    )
    print(f"% dataset information {info}")

    device: str = "cuda" if T.cuda.is_available() else "cpu"
    print(f"% training on device '{device}'")
    model = train_model(
        model=UNet(  # pyright: ignore[reportCallIssue]
            image_channels=info.image_channels,
            **config["model"]  # pyright: ignore[reportCallIssue]
        ),
        train_loader=dataloaders.train,
        valid_loader=dataloaders.valid,
        sigma_data=info.sigma_data,
        nb_epochs=config["nb_epochs"],  # pyright: ignore[reportArgumentType]
        eta=float(config["eta"]),  # pyright: ignore[reportArgumentType]
        run=run,
        valid_loss_every=max(config["nb_epochs"] // 10, 1),  # pyright: ignore[reportOperatorIssue, reportUnknownArgumentType]
        device=device  # pyright: ignore[reportArgumentType]
    )

    n_samples: int = 8
    print(f"% generating {n_samples} samples")
    samples: T.Tensor = sample_model(
        model=model,
        sigma_data=info.sigma_data,
        n_samples=n_samples,
        image_channels=info.image_channels,
        image_size=info.image_size,
        device=device  # pyright: ignore[reportArgumentType]
    )
    samples = samples.clamp(-1, 1).add(1).div(2).mul(255).byte()
    samples = make_grid(samples)
    memory: Image.Image = Image.fromarray(samples.permute(1, 2, 0).cpu().numpy())
    run.log({"generated_samples": wandb.Image(memory)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("yaml_configuration_filepath")
    _ = parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    config: dict[str, dict[str, str] | str] | None = yaml.safe_load(  # pyright: ignore[reportAny]
        Path(args.yaml_configuration_filepath)  # pyright: ignore[reportAny]
            .read_text(encoding="utf-8")
    )
    if config is None:
        raise Exception(f"Configuration file at {args.yaml_configuration_filepath} can not be opened.")  # noqa: TRY002  # pyright: ignore[reportAny]
    flat_config: dict[str, str] = reduce(  # pyright: ignore[reportUnknownVariableType]
        lambda res, e: (  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
            {**res, **e[1]}
            if isinstance(e[1], dict)
            else {**res, e[0]: e[1]}
        ),
        config.items(),
        {},  # pyright: ignore[reportUnknownArgumentType]
    )

    _ = wandb.login()
    with wandb.init(
        entity="osd1-garden", project="diffusion",
        config=flat_config,
        mode="online" if args.log else "offline",  # pyright: ignore[reportAny]
        settings=wandb.Settings(x_disable_stats=True)
    ) as run:
        main(run, config)

        run.finish()
