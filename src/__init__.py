#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pacote principal do projeto YOLOv8 Training.

Este pacote contém os módulos necessários para download de datasets,
treinamento de modelos YOLOv8, avaliação e inferência.
"""

__version__ = "0.1.0"
__author__ = "YOLOv8 Training Team"

from src.utils.logger import setup_logging
from src.utils.config_loader import load_yaml_config, validate_config
from src.core.data_management import RoboflowDownloader
from src.core.training import TrainingConfig, YOLOv8Trainer, TrainingPipeline, ITrainer, ITrainingPipeline

# Configuração automática de logging
setup_logging()
