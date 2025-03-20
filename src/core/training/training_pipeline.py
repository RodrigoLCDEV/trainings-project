#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para pipeline de treinamento de modelos.

Este módulo implementa um pipeline completo de treinamento,
desde o download dos dados até a exportação do modelo.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

from src.core.data_management.interface import IDataDownloader
from src.core.training.interface import ITrainer, ITrainingPipeline
from src.core.training.training_config_pydantic import TrainingConfig


class TrainingPipeline(ITrainingPipeline):
    """
    Pipeline completo para treinamento de modelos YOLOv8.
    
    Esta classe implementa um fluxo de trabalho completo para treinar,
    validar e exportar modelos YOLOv8, usando injeção de dependências.
    
    Attributes:
        config (TrainingConfig): Configuração para o treinamento.
        downloader (IDataDownloader): Downloader de datasets.
        trainer (ITrainer): Treinador de modelos.
        logger (logging.Logger): Logger para registrar eventos.
    """
    
    def __init__(self, 
                config: TrainingConfig, 
                downloader: IDataDownloader, 
                trainer: ITrainer):
        """
        Inicializa o pipeline com as dependências fornecidas.
        
        Args:
            config (TrainingConfig): Configuração para o treinamento.
            downloader (IDataDownloader): Downloader para obter os dados.
            trainer (ITrainer): Treinador para treinar o modelo.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.downloader = downloader
        self.trainer = trainer
        
        self.logger.info("Pipeline de treinamento inicializado")
    
    def download_data(self, force: bool = False) -> bool:
        """
        Realiza o download dos dados necessários para o treinamento.
        
        Args:
            force (bool): Se True, força o download mesmo se os dados já existirem.
            
        Returns:
            bool: True se o download foi bem-sucedido, False caso contrário.
        """
        self.logger.info("Iniciando download de dados")
        
        try:
            # Realizar download
            success, message = self.downloader.download_dataset(force_download=force)
            
            if not success:
                self.logger.error(f"Falha no download: {message}")
                return False
            
            self.logger.info(f"Download concluído: {message}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no download de dados: {str(e)}")
            return False
    
    def prepare_data(self) -> bool:
        """
        Prepara os dados para treinamento, realizando validação e configuração.
        
        Returns:
            bool: True se a preparação foi bem-sucedida, False caso contrário.
        """
        self.logger.info("Preparando dados para treinamento")
        
        try:
            # Validar dataset
            valid, stats = self.downloader.validate_dataset()
            
            if not valid:
                self.logger.error(f"Dataset inválido: {stats.get('error', 'Erro desconhecido')}")
                return False
            
            self.logger.info(f"Dataset válido com {stats.get('train_images')} imagens de treino")
            
            # Configurar data_yaml_path para o trainer
            if stats.get("path"):
                data_yaml_path = Path(stats["path"]) / "data.yaml"
                if data_yaml_path.exists():
                    self.config.set_data_yaml_path(data_yaml_path)
                    self.logger.info(f"data.yaml configurado: {data_yaml_path}")
                else:
                    self.logger.error(f"data.yaml não encontrado em {data_yaml_path}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na preparação de dados: {str(e)}")
            return False
    
    def train_model(self) -> Dict[str, Any]:
        """
        Treina o modelo com os dados preparados.
        
        Returns:
            Dict[str, Any]: Métricas e resultados do treinamento.
        """
        self.logger.info("Iniciando treinamento do modelo")
        
        try:
            # Treinar modelo
            metrics = self.trainer.train()
            self.logger.info(f"Treinamento concluído: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro no treinamento: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Avalia o modelo treinado.
        
        Returns:
            Dict[str, Any]: Métricas de avaliação.
        """
        self.logger.info("Avaliando modelo treinado")
        
        try:
            # Validar modelo
            metrics = self.trainer.validate()
            self.logger.info(f"Avaliação concluída: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro na avaliação: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def deploy_model(self, format: str = "onnx") -> Dict[str, Any]:
        """
        Exporta o modelo para implantação.
        
        Args:
            format (str): Formato para exportação do modelo.
            
        Returns:
            Dict[str, Any]: Informações sobre o modelo exportado.
        """
        self.logger.info(f"Exportando modelo para formato {format}")
        
        try:
            # Exportar modelo
            export_result = self.trainer.export_model(format=format)
            self.logger.info(f"Exportação concluída: {export_result}")
            return export_result
            
        except Exception as e:
            self.logger.error(f"Erro na exportação: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def run_full_pipeline(self, force_download: bool = False) -> Dict[str, Any]:
        """
        Executa o pipeline completo: download, preparação, treino, avaliação e exportação.
        
        Args:
            force_download (bool): Se True, força o download mesmo se os dados já existirem.
            
        Returns:
            Dict[str, Any]: Resultados de todas as etapas do pipeline.
        """
        self.logger.info("Iniciando pipeline completo de treinamento")
        
        results = {
            "download": False,
            "preparation": False,
            "training": None,
            "evaluation": None,
            "deployment": None
        }
        
        # Download de dados
        download_success = self.download_data(force=force_download)
        results["download"] = download_success
        
        if not download_success:
            self.logger.error("Pipeline interrompido: falha no download de dados")
            return results
        
        # Preparação de dados
        preparation_success = self.prepare_data()
        results["preparation"] = preparation_success
        
        if not preparation_success:
            self.logger.error("Pipeline interrompido: falha na preparação de dados")
            return results
        
        # Treinamento
        training_results = self.train_model()
        results["training"] = training_results
        
        if not training_results.get("success", False):
            self.logger.error("Pipeline interrompido: falha no treinamento")
            return results
        
        # Avaliação
        evaluation_results = self.evaluate_model()
        results["evaluation"] = evaluation_results
        
        # Exportação
        deployment_results = self.deploy_model(format="onnx")
        results["deployment"] = deployment_results
        
        self.logger.info("Pipeline de treinamento concluído com sucesso")
        return results
    
    def __str__(self) -> str:
        """
        Retorna uma representação em string do pipeline.
        
        Returns:
            str: Descrição do pipeline.
        """
        return (
            f"TrainingPipeline:\n"
            f"  Downloader: {type(self.downloader).__name__}\n"
            f"  Trainer: {type(self.trainer).__name__}\n"
            f"  Config: {type(self.config).__name__}\n"
        ) 