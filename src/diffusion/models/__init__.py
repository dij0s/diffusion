from .noise_conditioned_unet import Model as NoiseConditionedUNet
from .unet import Model as UNet

MODELS = {
    "unet": UNet,
    "noise_conditioned_unet": NoiseConditionedUNet,
}

def build_model(name: str, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', choose from available: [{sorted(MODELS)}]")
    return MODELS[name](**kwargs)  # pyright: ignore[reportUnknownArgumentType]
