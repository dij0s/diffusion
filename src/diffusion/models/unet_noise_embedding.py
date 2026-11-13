import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(nb_channels, cond_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        # reshape batch noise
        # level, from training
        c_noise = c_noise.reshape(-1)
        cond = self.noise_emb(c_noise)
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, cond)
        return self.conv_out(x)


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int, cond_channels: int) -> None:
        super().__init__()
        # double the linear
        # layer output as we
        # want two parameters
        # a, b, per channel
        self.affine1 = nn.Linear(cond_channels, nb_channels*2)
        self.norm1 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.affine2 = nn.Linear(cond_channels, nb_channels*2)
        self.norm2 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _batch_cond_norm(x: torch.Tensor, norm: nn.Module, affine: nn.Module, cond: torch.Tensor) -> torch.Tensor:
        """Batch-compatible noise-conditioned BatchNorm2d utility function."""
        a, b = affine(cond).chunk(2, dim=1)
        a, b = a[:, :, None, None], b[:, :, None, None]
        # reparametrize scale param.
        # s.t. it is close to 1
        # after initialization
        return (1 + a) * norm(x) + b


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self._batch_cond_norm(x, self.norm1, self.affine1, cond)))
        y = self.conv2(F.relu(self._batch_cond_norm(y, self.norm2, self.affine2, cond)))
        return x + y
