#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testes para o módulo de validação de datasets.
"""

import os
import shutil
import tempfile
import pytest
import yaml
from pathlib import Path

from src.core.data_management.dataset_validator import DatasetValidator


@pytest.fixture
def valid_dataset():
    """Cria um dataset válido temporário para testes."""
    # Criar diretório temporário
    temp_dir = tempfile.mkdtemp()
    dataset_path = Path(temp_dir)
    
    # Criar estrutura do dataset
    for subset in ["train", "val", "test"]:
        # Criar diretórios de imagens
        images_dir = dataset_path / "images" / subset
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar diretórios de labels
        labels_dir = dataset_path / "labels" / subset
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar algumas imagens e labels
        for i in range(5):
            # Criar arquivo de imagem vazio
            image_file = images_dir / f"img_{i}.jpg"
            image_file.touch()
            
            # Criar arquivo de label correspondente
            label_file = labels_dir / f"img_{i}.txt"
            with open(label_file, "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
    
    # Criar arquivo data.yaml
    data_yaml = {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "class1",
            1: "class2"
        }
    }
    
    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    
    yield dataset_path
    
    # Limpar após os testes
    shutil.rmtree(temp_dir)


@pytest.fixture
def invalid_dataset():
    """Cria um dataset inválido temporário para testes."""
    # Criar diretório temporário
    temp_dir = tempfile.mkdtemp()
    dataset_path = Path(temp_dir)
    
    # Criar estrutura parcial (apenas train, sem test e val)
    images_dir = dataset_path / "images" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    labels_dir = dataset_path / "labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar algumas imagens e labels com discrepância
    for i in range(5):
        # Criar arquivo de imagem vazio
        image_file = images_dir / f"img_{i}.jpg"
        image_file.touch()
        
        # Criar apenas 3 labels (omitindo 2 para testar discrepância)
        if i < 3:
            label_file = labels_dir / f"img_{i}.txt"
            with open(label_file, "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
    
    # Criar arquivo data.yaml incompleto
    data_yaml = {
        "path": str(dataset_path),
        "train": "images/train",
        # Faltando "val" e "test"
        "names": {
            0: "class1"
        }
    }
    
    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    
    yield dataset_path
    
    # Limpar após os testes
    shutil.rmtree(temp_dir)


def test_validate_valid_dataset(valid_dataset):
    """Testa a validação de um dataset válido."""
    validator = DatasetValidator()
    
    # Executar validação
    valid, stats = validator.validate(valid_dataset)
    
    # Verificar resultado
    assert valid is True
    assert stats["train_images"] == 5
    assert stats["valid_images"] == 5
    assert stats["test_images"] == 5
    assert stats["classes"] == 2


def test_validate_invalid_dataset(invalid_dataset):
    """Testa a validação de um dataset inválido."""
    validator = DatasetValidator()
    
    # Executar validação (deve falhar na validação do YAML)
    valid, stats = validator.validate(invalid_dataset)
    
    # Verificar resultado
    assert valid is False
    assert "error" in stats


def test_check_yaml_valid(valid_dataset):
    """Testa a validação de um arquivo YAML válido."""
    validator = DatasetValidator()
    
    # Validar YAML
    yaml_path = valid_dataset / "data.yaml"
    valid = validator.check_yaml(yaml_path)
    
    # Verificar resultado
    assert valid is True


def test_check_yaml_invalid(invalid_dataset):
    """Testa a validação de um arquivo YAML inválido."""
    validator = DatasetValidator()
    
    # Validar YAML
    yaml_path = invalid_dataset / "data.yaml"
    valid = validator.check_yaml(yaml_path)
    
    # Verificar resultado
    assert valid is False


def test_check_images_annotations_valid(valid_dataset):
    """Testa a validação de correspondência entre imagens e anotações em um dataset válido."""
    validator = DatasetValidator()
    
    # Validar correspondência
    valid = validator.check_images_annotations(valid_dataset)
    
    # Verificar resultado
    assert valid is True


def test_check_images_annotations_invalid(invalid_dataset):
    """Testa a validação de correspondência entre imagens e anotações em um dataset inválido."""
    validator = DatasetValidator()
    
    # Validar correspondência
    valid = validator.check_images_annotations(invalid_dataset)
    
    # Verificar resultado
    assert valid is False 