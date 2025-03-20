#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para validação de datasets.

Este módulo implementa classes para validação de datasets,
verificando a integridade dos arquivos, correspondência entre
imagens e anotações, e validação do arquivo YAML.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Union, List, Set

from src.core.data_management.interface import IDatasetValidator


class DatasetValidator(IDatasetValidator):
    """
    Classe para validação de datasets do YOLOv8.
    
    Esta classe implementa métodos para validar a estrutura e integridade
    de datasets no formato YOLOv8.
    
    Attributes:
        logger (logging.Logger): Logger para registrar eventos.
    """
    
    def __init__(self):
        """Inicializa o validador de datasets."""
        self.logger = logging.getLogger(__name__)
    
    def validate(self, dataset_path: Union[str, Path]) -> Tuple[bool, Dict[str, Any]]:
        """
        Valida a estrutura e os arquivos de um dataset.
        
        Args:
            dataset_path (Union[str, Path]): Caminho para o diretório do dataset.
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Tupla (válido, estatísticas) com o resultado da validação.
        """
        dataset_path = Path(dataset_path)
        yaml_path = dataset_path / "data.yaml"
        
        # Verificar se o diretório existe
        if not dataset_path.exists() or not dataset_path.is_dir():
            self.logger.error(f"Diretório do dataset não encontrado: {dataset_path}")
            return False, {"error": f"Diretório do dataset não encontrado: {dataset_path}"}
        
        # Verificar o arquivo YAML
        if not self.check_yaml(yaml_path):
            return False, {"error": f"Arquivo YAML inválido: {yaml_path}"}
        
        # Verificar a correspondência entre imagens e anotações
        if not self.check_images_annotations(dataset_path):
            return False, {"error": "Discrepância entre imagens e anotações"}
        
        # Calcular estatísticas
        stats = self._calculate_stats(dataset_path, yaml_path)
        self.logger.info(f"Dataset validado: {stats}")
        
        return True, stats
    
    def check_yaml(self, yaml_path: Union[str, Path]) -> bool:
        """
        Verifica se o arquivo YAML do dataset é válido.
        
        Args:
            yaml_path (Union[str, Path]): Caminho para o arquivo YAML do dataset.
            
        Returns:
            bool: True se o arquivo YAML for válido, False caso contrário.
        """
        yaml_path = Path(yaml_path)
        
        # Verificar se o arquivo existe
        if not yaml_path.exists():
            self.logger.error(f"Arquivo YAML não encontrado: {yaml_path}")
            return False
        
        try:
            # Carregar o arquivo YAML
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            
            # Verificar campos obrigatórios
            required_fields = ["path", "train", "val", "test", "names"]
            for field in required_fields:
                if field not in data:
                    self.logger.error(f"Campo obrigatório '{field}' ausente no YAML")
                    return False
            
            # Verificar se há pelo menos uma classe
            if not data["names"] or not isinstance(data["names"], dict):
                self.logger.error("Nenhuma classe definida no arquivo YAML")
                return False
            
            self.logger.info(f"Arquivo YAML válido: {len(data['names'])} classes definidas")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao validar arquivo YAML: {str(e)}")
            return False
    
    def check_images_annotations(self, dataset_path: Union[str, Path]) -> bool:
        """
        Verifica se há correspondência entre imagens e anotações.
        
        Args:
            dataset_path (Union[str, Path]): Caminho para o diretório do dataset.
            
        Returns:
            bool: True se houver correspondência entre imagens e anotações, False caso contrário.
        """
        dataset_path = Path(dataset_path)
        
        # Verificar cada conjunto (train, val, test)
        for subset in ["train", "val", "test"]:
            images_dir = dataset_path / "images" / subset
            labels_dir = dataset_path / "labels" / subset
            
            # Verificar se os diretórios existem
            if not images_dir.exists() or not labels_dir.exists():
                self.logger.warning(f"Diretórios para {subset} não encontrados")
                continue
            
            # Obter listas de arquivos
            image_files = set(f.stem for f in images_dir.glob("*.jpg") if f.is_file())
            image_files.update(f.stem for f in images_dir.glob("*.png") if f.is_file())
            label_files = set(f.stem for f in labels_dir.glob("*.txt") if f.is_file())
            
            # Verificar correspondência
            if image_files != label_files:
                missing_labels = image_files - label_files
                missing_images = label_files - image_files
                
                if missing_labels:
                    self.logger.error(f"Imagens sem anotações em {subset}: {len(missing_labels)}")
                
                if missing_images:
                    self.logger.error(f"Anotações sem imagens em {subset}: {len(missing_images)}")
                
                return False
        
        self.logger.info("Correspondência entre imagens e anotações verificada")
        return True
    
    def _calculate_stats(self, dataset_path: Path, yaml_path: Path) -> Dict[str, Any]:
        """
        Calcula estatísticas do dataset.
        
        Args:
            dataset_path (Path): Caminho para o diretório do dataset.
            yaml_path (Path): Caminho para o arquivo YAML do dataset.
            
        Returns:
            Dict[str, Any]: Estatísticas do dataset.
        """
        stats = {
            "train_images": 0,
            "valid_images": 0,
            "test_images": 0,
            "classes": 0,
            "path": str(dataset_path)
        }
        
        # Contar imagens em cada conjunto
        train_dir = dataset_path / "images" / "train"
        val_dir = dataset_path / "images" / "val"
        test_dir = dataset_path / "images" / "test"
        
        if train_dir.exists():
            stats["train_images"] = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
        
        if val_dir.exists():
            stats["valid_images"] = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.png")))
        
        if test_dir.exists():
            stats["test_images"] = len(list(test_dir.glob("*.jpg"))) + len(list(test_dir.glob("*.png")))
        
        # Carregar número de classes do YAML
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                stats["classes"] = len(data.get("names", {}))
                stats["class_names"] = list(data.get("names", {}).values())
        except Exception:
            pass
        
        return stats 