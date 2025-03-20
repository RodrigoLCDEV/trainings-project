#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para treinamento de modelos YOLOv8.

Este módulo implementa classes e funções para gerenciar
o treinamento e a configuração de modelos YOLOv8 usando
interfaces, injeção de dependência e validação com pydantic.
"""

# Interfaces
from src.core.training.interface import ITrainer, ITrainingPipeline

# Implementações com pydantic e injeção de dependência
from src.core.training.training_config_pydantic import TrainingConfig, YOLOv8Hyperparameters, TrainingPaths
from src.core.training.yolov8_trainer import YOLOv8Trainer
from src.core.training.training_pipeline import TrainingPipeline

__all__ = [
    # Interfaces
    "ITrainer",
    "ITrainingPipeline",
    
    # Configurações com pydantic
    "TrainingConfig",
    "YOLOv8Hyperparameters",
    "TrainingPaths",
    
    # Implementações com injeção de dependência
    "YOLOv8Trainer",
    "TrainingPipeline"
]
