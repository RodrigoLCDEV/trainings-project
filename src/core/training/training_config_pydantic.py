#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para configuração de treinamento de modelos YOLOv8 usando pydantic.

Este módulo implementa modelos para validação de configurações
e hiperparâmetros para treinamento de modelos YOLOv8 usando pydantic.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from pydantic import BaseModel, Field, validator, root_validator, PositiveInt, PositiveFloat

from src.utils.config_loader import load_yaml_config


class YOLOv8Hyperparameters(BaseModel):
    """
    Modelo pydantic para validação dos hiperparâmetros do YOLOv8.
    """
    model_size: str = Field(
        ...,
        description="Tamanho do modelo YOLOv8 (nano, small, medium, large, xlarge, ou n, s, m, l, x)"
    )
    batch_size: PositiveInt = Field(
        default=16,
        description="Tamanho do lote para treinamento"
    )
    epochs: PositiveInt = Field(
        default=100,
        description="Número de épocas para treinamento"
    )
    img_size: PositiveInt = Field(
        default=640,
        description="Tamanho da imagem para treinamento (múltiplo de 32)"
    )
    optimizer: str = Field(
        default="SGD",
        description="Otimizador a ser usado (SGD, Adam, etc.)"
    )
    lr0: PositiveFloat = Field(
        default=0.01,
        description="Taxa de aprendizado inicial"
    )
    patience: PositiveInt = Field(
        default=50,
        description="Paciência para early stopping"
    )
    device: str = Field(
        default="",
        description="Dispositivo para treinamento (vazio para autodetecção, ou '0' para primeira GPU)"
    )
    
    @validator('model_size')
    def validate_model_size(cls, v):
        valid_sizes = ["nano", "small", "medium", "large", "xlarge", "n", "s", "m", "l", "x"]
        if v.lower() not in valid_sizes:
            raise ValueError(f"Tamanho de modelo inválido: {v}. Valores válidos: {valid_sizes}")
        return v
    
    @validator('img_size')
    def validate_img_size(cls, v):
        if v % 32 != 0:
            raise ValueError(f"Tamanho de imagem deve ser múltiplo de 32: {v}")
        return v


class TrainingPaths(BaseModel):
    """
    Modelo pydantic para validação dos caminhos de treinamento.
    """
    data_dir: Path = Field(
        default=Path("Dataset_roboflow"),
        description="Diretório para dados de treinamento"
    )
    model_save_dir: Path = Field(
        default=Path("models/model_yolo"),
        description="Diretório para salvar os modelos treinados"
    )
    data_yaml_path: Optional[Path] = Field(
        default=None,
        description="Caminho para o arquivo data.yaml do dataset"
    )
    
    @validator('model_save_dir', 'data_dir')
    def create_if_not_exists(cls, v):
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('data_yaml_path')
    def validate_data_yaml(cls, v):
        if v is not None and not v.exists():
            raise ValueError(f"Arquivo data.yaml não encontrado: {v}")
        return v


class TrainingConfig(BaseModel):
    """
    Modelo pydantic para configuração completa de treinamento do YOLOv8.
    """
    hyperparameters: YOLOv8Hyperparameters
    paths: TrainingPaths
    
    class Config:
        """Configurações do modelo pydantic."""
        validate_assignment = True
        arbitrary_types_allowed = True
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/settings.yml") -> "TrainingConfig":
        """
        Carrega configuração de um arquivo YAML.
        
        Args:
            config_path (str): Caminho para o arquivo de configuração YAML.
            
        Returns:
            TrainingConfig: Instância de TrainingConfig com os valores do YAML.
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado.
            ValueError: Se o arquivo contiver configurações inválidas.
        """
        # Verificar se o arquivo existe
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
        
        # Carregar configurações do YAML
        config_data = load_yaml_config(config_path)
        
        # Extrair seções relevantes
        training_config = config_data.get("training", {})
        paths_config = config_data.get("paths", {})
        
        # Construir objetos do modelo
        hyperparams = YOLOv8Hyperparameters(
            model_size=training_config.get("model_size", "nano"),
            batch_size=training_config.get("batch_size", 16),
            epochs=training_config.get("epochs", 100),
            img_size=training_config.get("img_size", 640),
            optimizer=training_config.get("optimizer", "SGD"),
            lr0=training_config.get("lr0", 0.01),
            patience=training_config.get("patience", 50),
            device=training_config.get("device", "")
        )
        
        paths = TrainingPaths(
            data_dir=Path(paths_config.get("data_dir", "Dataset_roboflow")),
            model_save_dir=Path(paths_config.get("model_save_dir", "models/model_yolo"))
        )
        
        # Construir a configuração completa
        return cls(
            hyperparameters=hyperparams,
            paths=paths
        )
    
    def get_yolo_model_name(self) -> str:
        """
        Retorna o nome formatado do modelo YOLOv8 baseado no tamanho.
        
        Returns:
            str: Nome do modelo no formato "yolov8n.pt", "yolov8s.pt", etc.
        """
        # Converte model_size para o formato abreviado correto
        size_mapping = {
            "nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x",
        }
        
        # Se já for abreviado (n, s, m, l, x), mantém; caso contrário, converte
        size_code = self.hyperparameters.model_size.lower()
        if size_code in size_mapping.keys():
            size_code = size_mapping[size_code]
        
        return f"yolov8{size_code}.pt"
    
    def get_training_args(self) -> Dict[str, Any]:
        """
        Retorna os argumentos formatados para treinar o modelo com a API do YOLOv8.
        
        Returns:
            Dict[str, Any]: Dicionário com argumentos de treinamento prontos para ultralytics.
        """
        return {
            "data": str(self.paths.data_yaml_path) if self.paths.data_yaml_path else None,
            "epochs": self.hyperparameters.epochs,
            "patience": self.hyperparameters.patience,
            "batch": self.hyperparameters.batch_size,
            "imgsz": self.hyperparameters.img_size,
            "optimizer": self.hyperparameters.optimizer,
            "lr0": self.hyperparameters.lr0,
            "device": self.hyperparameters.device,
            "project": str(self.paths.model_save_dir),
            "name": f"yolov8_{self.hyperparameters.model_size}_{self.hyperparameters.img_size}px",
            "exist_ok": True,
            "pretrained": True,
            "verbose": True
        }
    
    def set_data_yaml_path(self, data_yaml_path: Union[str, Path]) -> None:
        """
        Define o caminho para o arquivo data.yaml do dataset.
        
        Args:
            data_yaml_path (Union[str, Path]): Caminho para o arquivo data.yaml
            
        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        data_yaml_path = Path(data_yaml_path)
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Arquivo data.yaml não encontrado: {data_yaml_path}")
        
        self.paths.data_yaml_path = data_yaml_path
    
    def __str__(self) -> str:
        """
        Retorna uma representação em string da configuração.
        
        Returns:
            str: Descrição da configuração.
        """
        hp = self.hyperparameters
        config_str = (
            f"TrainingConfig:\n"
            f"  Modelo: YOLOv8 {hp.model_size}\n"
            f"  Batch Size: {hp.batch_size}\n"
            f"  Epochs: {hp.epochs}\n"
            f"  Image Size: {hp.img_size}px\n"
            f"  Optimizer: {hp.optimizer}\n"
            f"  Learning Rate: {hp.lr0}\n"
        )
        
        if self.paths.data_yaml_path:
            config_str += f"  Dataset: {self.paths.data_yaml_path.parent.name}\n"
        
        return config_str 