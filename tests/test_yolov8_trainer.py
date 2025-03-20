#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testes para o módulo de treinamento YOLOv8.
"""

import os
import sys
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.training.training_config_pydantic import TrainingConfig, YOLOv8Hyperparameters, TrainingPaths
from src.core.training.yolov8_trainer import YOLOv8Trainer


# Mock para a classe YOLO
class MockYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.best = "mock_best_model.pt"
        self.save_dir = "mock_results_dir"
    
    def train(self, **kwargs):
        # Simular um objeto de resultados de treinamento
        results = MagicMock()
        results.best = "mock_best_model.pt"
        results.save_dir = "mock_results_dir"
        results.results_dict = {
            "metrics/precision(B)": 0.85,
            "metrics/recall(B)": 0.82,
            "metrics/mAP50(B)": 0.90,
            "metrics/mAP50-95(B)": 0.75,
            "val/box_loss": 0.1,
            "epoch": 100
        }
        return results
    
    def val(self, **kwargs):
        # Simular resultados de validação
        results = MagicMock()
        results.results_dict = {
            "metrics/precision(B)": 0.84,
            "metrics/recall(B)": 0.81,
            "metrics/mAP50(B)": 0.89,
            "metrics/mAP50-95(B)": 0.74,
        }
        return results
    
    def export(self, **kwargs):
        # Simular exportação do modelo
        return "mock_exported_model.onnx"


@pytest.fixture
def mock_ultralytics(monkeypatch):
    """Mock do pacote ultralytics."""
    mock_pkg = MagicMock()
    mock_pkg.YOLO = MockYOLO
    monkeypatch.setattr("src.core.training.yolov8_trainer.YOLO", MockYOLO)
    monkeypatch.setattr("src.core.training.yolov8_trainer.ULTRALYTICS_AVAILABLE", True)
    return mock_pkg


@pytest.fixture
def training_config():
    """Configuração de teste para treinamento."""
    # Usar um diretório temporário
    temp_dir = tempfile.mkdtemp()
    
    # Criar hiperparâmetros de teste
    hyperparameters = YOLOv8Hyperparameters(
        model_size="nano",
        batch_size=2,
        epochs=1,
        img_size=640
    )
    
    # Criar configuração de caminhos
    paths = TrainingPaths(
        model_save_dir=Path(temp_dir) / "models",
        data_dir=Path(temp_dir) / "data"
    )
    
    # Criar arquivo data.yaml de teste
    data_dir = paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    data_yaml_path = data_dir / "data.yaml"
    
    with open(data_yaml_path, "w") as f:
        f.write("""
        path: ./data
        train: images/train
        val: images/val
        test: images/test
        
        names:
          0: class1
          1: class2
        """)
    
    # Configurar data_yaml_path
    paths.data_yaml_path = data_yaml_path
    
    # Criar configuração completa
    config = TrainingConfig(
        hyperparameters=hyperparameters,
        paths=paths
    )
    
    yield config
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_trainer_initialization(training_config, mock_ultralytics):
    """Testa a inicialização do treinador."""
    trainer = YOLOv8Trainer(training_config)
    assert trainer.config == training_config
    assert trainer.model is None
    assert trainer.model_path is None


def test_trainer_train(training_config, mock_ultralytics):
    """Testa o método de treinamento."""
    trainer = YOLOv8Trainer(training_config)
    
    # Treinar o modelo
    results = trainer.train()
    
    # Verificar resultados
    assert results["success"] is True
    assert results["model_path"] == "mock_best_model.pt"
    assert results["precision"] == 0.85
    assert results["mAP50"] == 0.90
    assert trainer.model_path == Path("mock_best_model.pt")


def test_trainer_validate(training_config, mock_ultralytics):
    """Testa o método de validação."""
    trainer = YOLOv8Trainer(training_config)
    
    # Definir model_path manualmente para simular modelo treinado
    trainer.model_path = Path("mock_best_model.pt")
    
    # Criar o arquivo falso para que Path.exists() retorne True
    Path("mock_best_model.pt").touch()
    
    try:
        # Validar o modelo
        metrics = trainer.validate()
        
        # Verificar resultados
        assert metrics["precision"] == 0.84
        assert metrics["recall"] == 0.81
        assert metrics["mAP50"] == 0.89
        assert metrics["mAP50-95"] == 0.74
    finally:
        # Limpar o arquivo criado
        if Path("mock_best_model.pt").exists():
            Path("mock_best_model.pt").unlink()


def test_trainer_export_model(training_config, mock_ultralytics, monkeypatch):
    """Testa o método de exportação do modelo."""
    # Mock para Path.stat() para o cálculo do tamanho do modelo
    class MockPathStat:
        @property
        def st_size(self):
            return 1024 * 1024 * 10  # 10 MB
    
    mock_path = MagicMock()
    mock_path.stat.return_value = MockPathStat()
    mock_path.parent = Path("mock_dir")
    mock_path.name = "mock_exported_model.onnx"
    
    # Mock para Path(str(exported))
    monkeypatch.setattr("pathlib.Path", lambda x: mock_path if x == "mock_exported_model.onnx" else Path(x))
    
    # Criar treinador
    trainer = YOLOv8Trainer(training_config)
    
    # Definir model_path manualmente para simular modelo treinado
    trainer.model_path = Path("mock_best_model.pt")
    
    # Criar o arquivo falso para que Path.exists() retorne True
    Path("mock_best_model.pt").touch()
    
    try:
        # Exportar o modelo
        result = trainer.export_model(format="onnx")
        
        # Verificar resultados
        assert result["success"] is True
        assert result["format"] == "onnx"
        assert result["model_size_mb"] == 10
    finally:
        # Limpar o arquivo criado
        if Path("mock_best_model.pt").exists():
            Path("mock_best_model.pt").unlink() 