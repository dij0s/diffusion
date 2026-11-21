import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="Diffusion playground", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from time import perf_counter

    return


@app.cell
def _():
    import torch as T
    from torch import nn
    from torch import Tensor

    return T, Tensor, nn


@app.cell
def _():
    from PIL import Image
    from torchvision.utils import make_grid

    return Image, make_grid


@app.cell
def _():
    from diffusion.helpers.data import load_dataset_and_make_dataloaders
    from diffusion.models import NoiseConditionedUNet
    from diffusion.pipeline.sampling import (
        sample_model,
        build_sigma_schedule,
        euler_sample,
        heun_sample,
    )

    return (
        NoiseConditionedUNet,
        build_sigma_schedule,
        euler_sample,
        heun_sample,
        load_dataset_and_make_dataloaders,
        sample_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diffusion playground

    This notebook allows for a deeper understanding of diffusion models, produced in the context of this project.
    """)
    return


@app.cell
def _(T, load_dataset_and_make_dataloaders):
    device: str = "cuda" if T.cuda.is_available() else "cpu"
    dataloaders, info = load_dataset_and_make_dataloaders(
        dataset_name="FashionMNIST",
        root_dir="data",
        batch_size=32,
        num_workers=4,
        pin_memory=device == "cuda",
    )
    return device, info


@app.cell(hide_code=True)
def _(NoiseConditionedUNet, T, info, nn):
    def load_model(filepath: str) -> nn.Module:
        """Load model from checkpoint filepath, in inference mode."""

        model: nn.Module = NoiseConditionedUNet(
            image_channels=info.image_channels, nb_channels=12, num_blocks=12, cond_channels=20
        )
        model.load_state_dict(T.load(filepath, weights_only=True))
        print(f"% loaded model parameters at {filepath}")
        print(model)
        _ = model.eval()

        return model

    return (load_model,)


@app.cell(hide_code=True)
def _(mo):
    get_model, set_model = mo.state(None)
    return get_model, set_model


@app.cell(hide_code=True)
def _(load_model, mo, set_model):
    mo.ui.file_browser(
        initial_path="checkpoints",
        multiple=False,
        label="Select model checkpoint.",
        on_change=lambda metadata: set_model(load_model(metadata[0].path)),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that the model parameters are loaded, we can play around!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    get_step, set_step = mo.state(0)
    return get_step, set_step


@app.cell(hide_code=True)
def _(mo):
    n_steps = mo.ui.number(10, 100, 10, value=20, label="Number of sampling steps")
    seed = mo.ui.number(0, 10_000, value=42, label="Seed number")
    mo.hstack([n_steps, seed])
    return n_steps, seed


@app.cell(hide_code=True)
def _(
    T,
    Tensor,
    build_sigma_schedule,
    device: str,
    euler_sample,
    get_model,
    heun_sample,
    info,
    mo,
    n_steps,
    sample_model,
    seed,
    set_step,
):
    _model = get_model()
    mo.stop(_model is None, mo.md("Must load a checkpoint first."))

    _sigmas: Tensor = build_sigma_schedule(n_steps.value)
    _g: T.Generator = T.Generator().manual_seed(seed.value)
    _x_init: Tensor = (
        T.randn(5, info.image_channels, info.image_size, info.image_size, generator=_g)
        * _sigmas[0]
    )

    _common = dict(
        model=_model,
        sigma_data=info.sigma_data,
        n_samples=5,
        image_channels=info.image_channels,
        image_size=info.image_size,
        denoising_steps=n_steps.value,
        x_init=_x_init,
        return_trajectory=True,
        device=device,
    )
    _, traj_euler = sample_model(step_fn=euler_sample, **_common)
    _, traj_heun = sample_model(step_fn=heun_sample, **_common)
    sigmas = _sigmas
    set_step(0)
    return sigmas, traj_euler, traj_heun


@app.cell(hide_code=True)
def _(mo, n_steps, set_step):
    next_btn: mo.ui.button = mo.ui.button(
        label="Next", on_click=lambda _: set_step(lambda s: min(s + 1, n_steps.value))
    )
    prev_btn: mo.ui.button = mo.ui.button(
        label="Previous", on_click=lambda _: set_step(lambda s: max(s - 1, 0))
    )
    reset_btn: mo.ui.button = mo.ui.button(label="Reset", on_click=lambda _: set_step(0))
    return next_btn, prev_btn, reset_btn


@app.cell(hide_code=True)
def _(
    Image,
    T,
    get_step,
    make_grid,
    mo,
    n_steps,
    next_btn: "mo.ui.button",
    prev_btn: "mo.ui.button",
    reset_btn: "mo.ui.button",
    sigmas,
    traj_euler,
    traj_heun,
):
    def to_image(x: T.Tensor) -> Image.Image:
        x = x.clamp(-1, 1).add(1).div(2).mul(255).byte().cpu()
        return Image.fromarray(make_grid(x, nrow=x.shape[0]).permute(1, 2, 0).numpy())

    _n = n_steps.value
    _step = min(get_step(), _n)
    _sigma = sigmas[_step].item() if _step < _n else 0.0
    _nfe_heun = max(2 * _step - (1 if _step == _n else 0), 0)

    mo.vstack(
        [
            mo.hstack([prev_btn, next_btn, reset_btn], justify="start"),
            mo.md(f"% step {_step}/{_n}"),
            mo.center(
                mo.vstack(
                    [
                        mo.md(f"% Euler"),
                        mo.image(to_image(traj_euler[_step]), width="80vw"),
                        mo.md(f"% Heun"),
                        mo.image(to_image(traj_heun[_step]), width="80vw"),
                    ]
                )
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
