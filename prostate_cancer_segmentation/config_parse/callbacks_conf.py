from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig
from dataclasses import dataclass
from pytorch_lightning.callbacks import Callback

from prostate_cancer_segmentation.config_parse.callbacks_available import (
    CheckpointConf,
    EarlyStopConf,
    LearningRateMonitorConf,
    LogMediaConf,
)
from prostate_cancer_segmentation.config_parse.conf_utils import validate_config_group_generic


# The Callbacks config cannot be directly initialized because it contains sub-entries for each callback, each
# of which describes a separate class.
# For each of the callbacks, we define a dataclass and use them to init the list of callbacks

if TYPE_CHECKING:
    pydantic_dataclass = dataclass
else:
    from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(frozen=True)
class CallbacksConf(ABC):
    name: str

    @abstractmethod
    def get_callbacks_list(self, *args) -> List:
        return []


@pydantic_dataclass(frozen=True)
class DisabledCallbacksConf(CallbacksConf):
    def get_callbacks_list(self, *args) -> List:
        return []


@pydantic_dataclass(frozen=True)
class StandardCallbacksConf(CallbacksConf):
    """Get a dictionary of all the callbacks."""

    early_stopping: Optional[Dict] = None
    checkpoints: Optional[Dict] = None
    log_media: Optional[Dict] = None
    lr_monitor: Optional[Dict] = None

    def get_callbacks_list(self, exp_dir: str, cfg: DictConfig) -> List[Callback]:
        """Get all available callbacks and the Callback Objects in list
        If a callback's entry is not present in the config file, it'll not be output in the list
        """
        callbacks_list = []
        if self.early_stopping is not None:
            early_stop = EarlyStopConf(**self.early_stopping).get_callback()
            callbacks_list.append(early_stop)

        if self.checkpoints is not None:
            checkpoint = CheckpointConf(**self.checkpoints).get_callback(exp_dir)
            callbacks_list.append(checkpoint)

        if self.log_media is not None:
            log_media = LogMediaConf(**self.log_media).get_callback(exp_dir, cfg)
            callbacks_list.append(log_media)

        if self.lr_monitor is not None:
            lr_monitor = LearningRateMonitorConf(**self.lr_monitor).get_callback()
            callbacks_list.append(lr_monitor)

        return callbacks_list


valid_names = {
    "disabled": DisabledCallbacksConf,
    "standard": StandardCallbacksConf,
}


def validate_config_group(cfg_subgroup: DictConfig) -> CallbacksConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="callback"
    )
    return validated_dataclass
