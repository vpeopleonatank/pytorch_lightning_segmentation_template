"""Dataclasses just to initialize and return Callback objects"""
from typing import Optional, TYPE_CHECKING

from omegaconf import DictConfig
from dataclasses import dataclass
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint

from prostate_cancer_segmentation.callbacks.log_media import LogMedia
from prostate_cancer_segmentation.config_parse.conf_utils import asdict_filtered

if TYPE_CHECKING:
    pydantic_dataclass = dataclass
else:
    from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(frozen=True)
class EarlyStopConf:
    monitor: str
    min_delta: float
    patience: int
    mode: str
    verbose: bool = False

    def get_callback(self) -> Callback:
        return EarlyStopping(**asdict_filtered(self))


@pydantic_dataclass(frozen=True)
class CheckpointConf:
    filename: Optional[str]
    monitor: Optional[str]
    mode: str
    save_last: Optional[bool]
    period: int
    save_top_k: Optional[int]
    verbose: bool = False

    def get_callback(self, logs_dir) -> Callback:
        return ModelCheckpoint(dirpath=logs_dir, **asdict_filtered(self))


@pydantic_dataclass(frozen=True)
class LogMediaConf:
    max_samples: int
    period_epoch: int
    period_step: int
    save_to_disk: bool
    save_latest_only: bool
    verbose: bool = False

    def get_callback(self, exp_dir: str, cfg: DictConfig) -> Callback:
        return LogMedia(exp_dir=exp_dir, cfg=cfg, **asdict_filtered(self))


@pydantic_dataclass(frozen=True)
class LearningRateMonitorConf:
    logging_interval: Optional[str]
    log_momentum: bool = False

    def get_callback(self) -> Callback:
        return LearningRateMonitor(**asdict_filtered(self))
