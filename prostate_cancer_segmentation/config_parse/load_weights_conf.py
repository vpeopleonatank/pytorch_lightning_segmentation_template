from abc import ABC
from typing import Optional, TYPE_CHECKING

from omegaconf import DictConfig
from dataclasses import dataclass

from prostate_cancer_segmentation.config_parse.conf_utils import validate_config_group_generic

if TYPE_CHECKING:
    pydantic_dataclass = dataclass
else:
    from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(frozen=True)
class LoadWeightsConf(ABC):
    name: str
    path: Optional[str] = None


valid_names = {
    "disabled": LoadWeightsConf,
    "load_weights": LoadWeightsConf,
}


def validate_config_group(cfg_subgroup: DictConfig) -> LoadWeightsConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="load_weights"
    )
    return validated_dataclass
