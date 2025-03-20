#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfaces para o módulo de treinamento.

Este módulo define as interfaces abstratas para treinamento
e avaliação de modelos.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path


class ITrainer(ABC):
    """
    Interface para treinadores de modelos.
    
    Define os métodos abstratos que devem ser implementados por
    qualquer classe concreta que treine modelos.
    """
    
    @abstractmethod
    def train(self, data_yaml_path: Optional[Union[str, Path]] = None, resume: bool = False) -> Dict[str, Any]:
        """
        Treina um modelo com os parâmetros configurados.
        
        Args:
            data_yaml_path (Optional[Union[str, Path]]): Caminho para o arquivo data.yaml.
            resume (bool): Se True, tenta retomar um treinamento anterior.
            
        Returns:
            Dict[str, Any]: Métricas e resultados do treinamento.
        """
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """
        Valida o modelo treinado no conjunto de validação.
        
        Returns:
            Dict[str, Any]: Métricas de validação.
        """
        pass
    
    @abstractmethod
    def export_model(self, format: str = "onnx", output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Exporta o modelo treinado para outros formatos.
        
        Args:
            format (str): Formato para exportação (onnx, torchscript, etc).
            output_dir (Optional[Union[str, Path]]): Diretório para salvar o modelo exportado.
                
        Returns:
            Dict[str, Any]: Informações sobre o modelo exportado.
        """
        pass


class ITrainingPipeline(ABC):
    """
    Interface para pipelines de treinamento.
    
    Define métodos abstratos para executar um fluxo completo de treinamento,
    desde o download dos dados até a avaliação do modelo.
    """
    
    @abstractmethod
    def download_data(self, force: bool = False) -> bool:
        """
        Realiza o download dos dados necessários para o treinamento.
        
        Args:
            force (bool): Se True, força o download mesmo se os dados já existirem.
            
        Returns:
            bool: True se o download foi bem-sucedido, False caso contrário.
        """
        pass
    
    @abstractmethod
    def prepare_data(self) -> bool:
        """
        Prepara os dados para treinamento, realizando pré-processamentos necessários.
        
        Returns:
            bool: True se a preparação foi bem-sucedida, False caso contrário.
        """
        pass
    
    @abstractmethod
    def train_model(self) -> Dict[str, Any]:
        """
        Treina o modelo com os dados preparados.
        
        Returns:
            Dict[str, Any]: Métricas e resultados do treinamento.
        """
        pass
    
    @abstractmethod
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Avalia o modelo treinado.
        
        Returns:
            Dict[str, Any]: Métricas de avaliação.
        """
        pass
    
    @abstractmethod
    def deploy_model(self, format: str = "onnx") -> Dict[str, Any]:
        """
        Exporta o modelo para implantação.
        
        Args:
            format (str): Formato para exportação do modelo.
            
        Returns:
            Dict[str, Any]: Informações sobre o modelo exportado.
        """
        pass 