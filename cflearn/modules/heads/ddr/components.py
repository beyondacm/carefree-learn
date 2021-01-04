import torch

from torch import Tensor
from typing import List
from typing import NamedTuple
from torch.nn import Module
from torch.nn import ModuleList

from ...blocks import MonotonousMapping


class MonoInteract(Module):
    def __init__(self, dim: int, num_units: List[int]):
        super().__init__()
        in_dim = 1
        self.blocks = ModuleList(
            MonotonousMapping.stack(
                in_dim,
                dim,
                num_units,
                ascent=True,
                return_blocks=True,
                use_couple_bias=False,
                bias=False,
            )
        )

    def forward(self, inp: Tensor, nets: List[Tensor]) -> Tensor:
        inp = self.blocks[0](inp)
        for i, block in enumerate(self.blocks[1:]):
            inp = inp * torch.sigmoid(nets[i])
            inp = block(inp)
        return inp


class AffineResults(NamedTuple):
    mul: Tensor
    add: Tensor
    out: Tensor


class AffineHead(Module):
    def __init__(self, dim: int):
        super().__init__()
        make = lambda: MonotonousMapping.stack(
            dim,
            1,
            [],
            ascent=True,
            use_couple_bias=False,
            bias=False,
        )
        self.mul_head = make()
        self.add_head = make()

    def forward(self, latent: Tensor, net: Tensor) -> AffineResults:
        mul = self.mul_head(latent)
        add = self.add_head(latent)
        return AffineResults(mul, add, mul * net + add)


class MedianOutputs(NamedTuple):
    nets: List[Tensor]
    median: Tensor
    pos_med_res: Tensor
    neg_med_res: Tensor


__all__ = [
    "MonoInteract",
    "AffineHead",
    "MedianOutputs",
]
