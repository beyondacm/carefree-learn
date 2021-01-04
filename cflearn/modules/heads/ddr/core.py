import math
import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional

from .components import *
from ..base import HeadBase
from ...blocks import MLP
from ....types import tensor_dict_type


@HeadBase.register("ddr")
class DDRHead(HeadBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        fetch_q: bool,
        fetch_cdf: bool,
        num_layers: int,
        latent_dim: int,
        mapping_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        final_mapping_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(in_dim, out_dim)
        assert out_dim == 1
        if not fetch_q and not fetch_cdf:
            raise ValueError("something must be fetched, either `q` or `cdf`")
        self.fetch_q = fetch_q
        self.fetch_cdf = fetch_cdf
        self.latent_dim = latent_dim
        num_units = [latent_dim] * num_layers
        if mapping_configs is None:
            mapping_configs = {}
        mapping_configs.setdefault("dropout", 0.0)
        mapping_configs.setdefault("batch_norm", False)
        self.median_backbone = MLP(
            in_dim,
            3,
            num_units,
            mapping_configs,
            final_mapping_config=final_mapping_config,
        )
        if not self.fetch_q:
            self.q_interact = None
            self.q_affine_head = None
        else:
            self.q_interact = MonoInteract(latent_dim, num_units)
            self.q_affine_head = AffineHead(latent_dim)
        if not self.fetch_cdf:
            self.y_interact = None
            self.y_affine_head = None
            self.y_logit_anchor = None
        else:
            self.y_interact = MonoInteract(latent_dim, num_units)
            self.y_affine_head = AffineHead(latent_dim)
            self.register_buffer("cdf_logit_anchor", torch.tensor([math.log(3.0)]))

    @property
    def q_fn(self) -> Callable[[Tensor], Tensor]:
        return lambda q: 2.0 * q - 1.0

    @property
    def q_inv_fn(self) -> Callable[[Tensor], Tensor]:
        return torch.sigmoid

    def _get_median_outputs(self, net: Tensor) -> MedianOutputs:
        nets = []
        for mapping in self.median_backbone.mappings:
            net = mapping(net)
            nets.append(net.detach())
        med, pos_med_res, neg_med_res = net.split(1, dim=1)
        return MedianOutputs(nets[:-1], med, pos_med_res, neg_med_res)

    def _q_results(
        self,
        q_batch: Tensor,
        median_outputs: MedianOutputs,
        do_inverse: bool = False,
    ) -> tensor_dict_type:
        q_batch = self.q_fn(q_batch)
        q_latent = self.q_interact(q_batch, median_outputs.nets)
        q_sign = q_batch > 0.5
        med_res = torch.where(
            q_sign,
            median_outputs.pos_med_res,
            median_outputs.neg_med_res,
        )
        affine = self.q_affine_head(q_latent, med_res.detach())
        y_res = affine.out
        results = {
            "q_sign": q_sign,
            "med_mul": affine.mul,
            "med_add": affine.add,
            "y_res": y_res,
        }
        q_inverse = None
        if do_inverse and self.fetch_cdf:
            inverse_results = self._y_results(y_res.detach(), median_outputs, False)
            q_inverse = inverse_results["q"]
        results["q_inverse"] = q_inverse
        return results

    def _y_results(
        self,
        y_res: Tensor,
        median_outputs: MedianOutputs,
        do_inverse: bool = False,
    ) -> tensor_dict_type:
        y_latent = self.y_interact(y_res, median_outputs.nets)
        affine = self.y_affine_head(y_latent, self.y_logit_anchor)
        q_logit = affine.out
        q = self.q_inv_fn(q_logit)
        results = {
            "q": q,
            "q_logit": q_logit,
            "q_logit_mul": affine.mul,
            "q_logit_add": affine.add,
        }
        y_inverse_res = None
        if do_inverse and self.fetch_q:
            inverse_results = self._q_results(q.detach(), median_outputs, False)
            y_inverse_res = inverse_results["y_res"]
        results["y_inverse_res"] = y_inverse_res
        return results

    def forward(
        self,
        net: Tensor,
        *,
        median: bool = False,
        do_inverse: bool = False,
        q_batch: Optional[Tensor] = None,
        y_batch: Optional[Tensor] = None,
    ) -> tensor_dict_type:
        median_outputs = self._get_median_outputs(net)
        results = {
            "median": median_outputs.median,
            "pos_med_res": median_outputs.pos_med_res,
            "neg_med_res": median_outputs.neg_med_res,
        }
        if self.fetch_q and not median and q_batch is not None:
            results.update(self._q_results(q_batch, median_outputs, do_inverse))
        if self.fetch_cdf and not median and y_batch is not None:
            results.update(self._y_results(y_batch, median_outputs, do_inverse))
        return results


__all__ = ["DDRHead"]
