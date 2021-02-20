from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

from omegaconf import DictConfig

from prostate_cancer_segmentation.config_parse import (
    callbacks_conf,
    dataset_conf,
    load_weights_conf,
    logger_conf,
    model_conf,
    optimizer_conf,
    scheduler_conf,
    trainer_conf,
)
from prostate_cancer_segmentation.config_parse.callbacks_conf import CallbacksConf
from prostate_cancer_segmentation.config_parse.dataset_conf import DatasetConf
from prostate_cancer_segmentation.config_parse.load_weights_conf import LoadWeightsConf
from prostate_cancer_segmentation.config_parse.logger_conf import LoggerConf
from prostate_cancer_segmentation.config_parse.model_conf import ModelConf
from prostate_cancer_segmentation.config_parse.optimizer_conf import OptimConf
from prostate_cancer_segmentation.config_parse.scheduler_conf import SchedulerConf
from prostate_cancer_segmentation.config_parse.trainer_conf import TrainerConf

if TYPE_CHECKING:
    pydantic_dataclass = dataclass
else:
    from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(frozen=True)
class TrainConf:
    random_seed: Optional[int]
    logs_root_dir: str
    dataset: DatasetConf
    optimizer: OptimConf
    model: ModelConf
    trainer: TrainerConf
    scheduler: SchedulerConf
    logger: LoggerConf
    callbacks: CallbacksConf
    load_weights: LoadWeightsConf


class ParseConfig:
    @classmethod
    def parse_config(cls, cfg: DictConfig) -> TrainConf:
        """Parses the config file read from hydra to populate the TrainConfig dataclass"""
        config = TrainConf(
            random_seed=cfg.random_seed,
            logs_root_dir=cfg.logs_root_dir,
            dataset=dataset_conf.validate_config_group(cfg.dataset),
            model=model_conf.validate_config_group(cfg.model),
            optimizer=optimizer_conf.validate_config_group(cfg.optimizer),
            trainer=trainer_conf.validate_config_group(cfg.trainer),
            scheduler=scheduler_conf.validate_config_group(cfg.scheduler),
            logger=logger_conf.validate_config_group(cfg.logger),
            callbacks=callbacks_conf.validate_config_group(cfg.callbacks),
            load_weights=load_weights_conf.validate_config_group(cfg.load_weights),
        )

        return config
