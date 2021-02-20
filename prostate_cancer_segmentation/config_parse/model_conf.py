from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from prostate_cancer_segmentation.config_parse.conf_utils import asdict_filtered, validate_config_group_generic
from prostate_cancer_segmentation.networks.deeplab.deeplab import DeepLab

if TYPE_CHECKING:
    pydantic_dataclass = dataclass
else:
    from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(frozen=True)
class ModelConf(ABC):
    name: str
    num_classes: int

    @abstractmethod
    def get_model(self):
        pass


@pydantic_dataclass(frozen=True)
class Deeplabv3Conf(ModelConf):
    backbone: str
    output_stride: int
    sync_bn: bool  # Can use PL to sync batchnorm. This enables custom batchnorm code.
    enable_amp: bool = False  # Should always be false, since PL takes case of 16bit training

    def get_model(self) -> torch.nn.Module:
        return DeepLab(**asdict_filtered(self))


valid_names = {
    "deeplabv3": Deeplabv3Conf,
}


def validate_config_group(cfg_subgroup: DictConfig) -> ModelConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="model"
    )
    return validated_dataclass
