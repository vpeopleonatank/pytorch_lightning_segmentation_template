from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from dataclasses import dataclass

from prostate_cancer_segmentation.config_parse.conf_utils import asdict_filtered, validate_config_group_generic

if TYPE_CHECKING:
    pydantic_dataclass = dataclass
else:
    from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(frozen=True)
class OptimConf(ABC):
    name: str

    @abstractmethod
    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        pass


@pydantic_dataclass(frozen=True)
class AdamConf(OptimConf):
    lr: float
    weight_decay: float

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=model_params, **asdict_filtered(self))


@pydantic_dataclass(frozen=True)
class SgdConf(OptimConf):
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        return torch.optim.SGD(params=model_params, **asdict_filtered(self))


valid_names = {"adam": AdamConf, "sgd": SgdConf}


def validate_config_group(cfg_subgroup: DictConfig) -> OptimConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="optimizer"
    )
    return validated_dataclass
