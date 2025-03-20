#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testes para o módulo de configuração de treinamento.
"""

import os
import tempfile
import pytest
from pathlib import Path
import yaml
from pydantic import ValidationError
from src.core.training.training_config_pydantic import TrainingConfig, YOLOv8Hyperparameters, TrainingPaths


def test_training_config_initialization():
    """Testa a inicialização da configuração TrainingConfig com valores válidos."""
    # Criar um arquivo YAML temporário para teste
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp_file:
        yaml_content = """
        training:
          model_size: "nano"
          batch_size: 16
          epochs: 100
          img_size: 640
        paths:
          model_save_dir: "models/test"
          data_dir: "datasets/test"
        """
        temp_file.write(yaml_content.encode("utf-8"))
        temp_path = temp_file.name

    try:
        # Carregar a configuração
        config = TrainingConfig.from_yaml(temp_path)

        # Verificar se os valores foram carregados corretamente
        assert config.hyperparameters.model_size == "nano"
        assert config.hyperparameters.batch_size == 16
        assert config.hyperparameters.epochs == 100
        assert config.hyperparameters.img_size == 640
        assert config.paths.model_save_dir == Path("models/test")
        assert config.paths.data_dir == Path("datasets/test")
        
        # Verificar valores padrão opcionais
        assert config.hyperparameters.optimizer == "SGD"
        assert config.hyperparameters.lr0 == 0.01
        assert config.hyperparameters.patience == 50
    finally:
        # Limpar o arquivo temporário
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_hyperparameters_validation():
    """Testa a validação de parâmetros inválidos."""
    # Criar um hiperparâmetro com valores inválidos
    with pytest.raises(ValidationError):
        # Batch size negativo deve falhar
        YOLOv8Hyperparameters(
            model_size="nano",
            batch_size=-1,
            epochs=100,
            img_size=640
        )
    
    with pytest.raises(ValidationError):
        # Epochs negativo deve falhar
        YOLOv8Hyperparameters(
            model_size="nano",
            batch_size=16,
            epochs=-1,
            img_size=640
        )

    with pytest.raises(ValidationError):
        # Learning rate negativa deve falhar
        YOLOv8Hyperparameters(
            model_size="nano",
            batch_size=16,
            epochs=100,
            img_size=640,
            lr0=-0.01
        )


def test_get_yolo_model_name():
    """Testa a função que retorna o nome do modelo formatado."""
    # Criar hiperparâmetros básicos
    hp_nano = YOLOv8Hyperparameters(
        model_size="nano",
        batch_size=16,
        epochs=100,
        img_size=640
    )
    
    paths = TrainingPaths(
        model_save_dir=Path("models/test"),
        data_dir=Path("datasets/test")
    )
    
    # Criar config com diferentes tamanhos
    config = TrainingConfig(hyperparameters=hp_nano, paths=paths)
    assert config.get_yolo_model_name() == "yolov8n.pt"
    
    # Modificar tamanho para testar outras opções
    config.hyperparameters.model_size = "small"
    assert config.get_yolo_model_name() == "yolov8s.pt"
    
    config.hyperparameters.model_size = "m"  # Forma abreviada
    assert config.get_yolo_model_name() == "yolov8m.pt"


def test_get_training_args():
    """Testa a geração de argumentos de treinamento."""
    # Criar hiperparâmetros básicos
    hp = YOLOv8Hyperparameters(
        model_size="nano",
        batch_size=16,
        epochs=100,
        img_size=640
    )
    
    paths = TrainingPaths(
        model_save_dir=Path("models/test"),
        data_dir=Path("datasets/test")
    )
    
    config = TrainingConfig(hyperparameters=hp, paths=paths)
    
    # Criar um arquivo data.yaml temporário
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as data_file:
        data_file.write(b"# Test data.yaml")
        data_yaml_path = data_file.name
    
    try:
        # Configurar data_yaml_path
        with pytest.raises(FileNotFoundError):
            # Deve falhar ao tentar definir um caminho que não existe
            config.set_data_yaml_path("invalid_path.yaml")
        
        # Definir com um caminho que existe
        config.set_data_yaml_path(data_yaml_path)
        
        # Obter argumentos de treinamento
        args = config.get_training_args()
        
        # Verificar argumentos
        assert args["data"] == data_yaml_path
        assert args["epochs"] == 100
        assert args["batch"] == 16
        assert args["imgsz"] == 640
        assert args["name"] == "yolov8_nano_640px"
        assert args["project"] == str(Path("models/test"))
    finally:
        # Limpar arquivos temporários
        if os.path.exists(data_yaml_path):
            os.unlink(data_yaml_path) 