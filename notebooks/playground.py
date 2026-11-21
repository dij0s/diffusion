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

    return (perf_counter,)


@app.cell
def _():
    import torch as T
    from torch import nn

    return T, nn


@app.cell
def _():
    from PIL import Image
    from torchvision.utils import make_grid

    return Image, make_grid


@app.cell
def _():
    from diffusion.helpers.data import load_dataset_and_make_dataloaders
    from diffusion.models import NoiseConditionedUNet
    from diffusion.pipeline.sampling import sample_model

    return (
        NoiseConditionedUNet,
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
            image_channels=info.image_channels, nb_channels=1, num_blocks=1, cond_channels=2
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
def _(
    Image,
    T,
    device: str,
    get_model,
    info,
    make_grid,
    mo,
    perf_counter,
    sample_model,
):
    start: float = perf_counter()
    n_samples: int = 5
    samples: T.Tensor = sample_model(
        model=get_model(),
        sigma_data=info.sigma_data,
        n_samples=n_samples,
        image_channels=info.image_channels,
        image_size=info.image_size,
        device=device,
    )
    elapsed: float = perf_counter() - start
    print(f"% sampled {n_samples} images in {elapsed:.3f}s, on {device}")

    samples = samples.clamp(-1, 1).add(1).div(2).mul(255).byte()
    samples = make_grid(samples)
    memory: Image.Image = Image.fromarray(samples.permute(1, 2, 0).cpu().numpy())
    mo.center(mo.image(memory, width="50vw"))
    return


if __name__ == "__main__":
    app.run()
