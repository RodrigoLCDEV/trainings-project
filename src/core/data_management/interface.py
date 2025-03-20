#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfaces para o gerenciamento de dados.

Este módulo define as interfaces abstratas para download, validação
e gerenciamento de datasets.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union
from pathlib import Path


class IDataDownloader(ABC):
    """
    Interface para classes que realizam download de datasets.
    
    Define os métodos abstratos que devem ser implementados por
    qualquer classe concreta que realizar download de datasets.
    """
    
    @abstractmethod
    def download_dataset(self, force_download: bool = False) -> Tuple[bool, str]:
        """
        Realiza o download de um dataset.
        
        Args:
            force_download (bool): Se True, força o download mesmo se os dados já existirem.
            
        Returns:
            Tuple[bool, str]: Tupla (sucesso, mensagem) indicando o resultado da operação.
        """
        pass
    
    @abstractmethod
    def validate_dataset(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Valida um dataset baixado, verificando arquivos e estrutura.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: Tupla (válido, estatísticas) com o resultado da validação.
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """
        Remove dados temporários e libera espaço em disco.
        
        Returns:
            bool: True se a limpeza foi bem-sucedida, False caso contrário.
        """
        pass


class IDatasetValidator(ABC):
    """
    Interface para classes que validam datasets.
    
    Define os métodos abstratos que devem ser implementados por
    qualquer classe concreta que valide datasets.
    """
    
    @abstractmethod
    def validate(self, dataset_path: Union[str, Path]) -> Tuple[bool, Dict[str, Any]]:
        """
        Valida a estrutura e os arquivos de um dataset.
        
        Args:
            dataset_path (Union[str, Path]): Caminho para o diretório do dataset.
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Tupla (válido, estatísticas) com o resultado da validação.
        """
        pass
    
    @abstractmethod
    def check_yaml(self, yaml_path: Union[str, Path]) -> bool:
        """
        Verifica se o arquivo YAML do dataset é válido.
        
        Args:
            yaml_path (Union[str, Path]): Caminho para o arquivo YAML do dataset.
            
        Returns:
            bool: True se o arquivo YAML for válido, False caso contrário.
        """
        pass
    
    @abstractmethod
    def check_images_annotations(self, dataset_path: Union[str, Path]) -> bool:
        """
        Verifica se há correspondência entre imagens e anotações.
        
        Args:
            dataset_path (Union[str, Path]): Caminho para o diretório do dataset.
            
        Returns:
            bool: True se houver correspondência entre imagens e anotações, False caso contrário.
        """
        pass 