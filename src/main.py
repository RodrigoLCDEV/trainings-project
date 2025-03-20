#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para o projeto de treinamento YOLOv8.

Este script serve como ponto de entrada para as principais 
funcionalidades do projeto: download de dados, treinamento, validação e predição.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Adicionar o diretório raiz ao sys.path se necessário
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Importar componentes do projeto
from src.utils.logger import setup_logging
from src.utils.config_loader import load_yaml_config

# Importar interfaces
from src.core.data_management.interface import IDataDownloader
from src.core.training.interface import ITrainer, ITrainingPipeline

# Importar implementações concretas
from src.core.data_management.dataset_validator import DatasetValidator
from src.core.data_management.roboflow_downloader import RoboflowDownloader
from src.core.training.training_config_pydantic import TrainingConfig
from src.core.training.yolov8_trainer import YOLOv8Trainer
from src.core.training.training_pipeline import TrainingPipeline


def create_pipeline(config_path: str = "config/settings.yml") -> TrainingPipeline:
    """
    Cria o pipeline de treinamento com as dependências necessárias.
    
    Args:
        config_path (str): Caminho para o arquivo de configuração.
        
    Returns:
        TrainingPipeline: Pipeline de treinamento configurado.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Criando pipeline com configuração: {config_path}")
    
    # Carregar configuração
    config = TrainingConfig.from_yaml(config_path)
    logger.info(f"Configuração carregada: {config}")
    
    # Criar downloader
    downloader = RoboflowDownloader(config_path=config_path)
    logger.info(f"Downloader criado: {type(downloader).__name__}")
    
    # Criar trainer
    trainer = YOLOv8Trainer(config=config)
    logger.info(f"Trainer criado: {type(trainer).__name__}")
    
    # Criar pipeline
    pipeline = TrainingPipeline(config=config, downloader=downloader, trainer=trainer)
    logger.info(f"Pipeline criado: {pipeline}")
    
    return pipeline


def download_data(config_path: str = "config/settings.yml", force: bool = False) -> bool:
    """
    Realiza o download dos dados do Roboflow.
    
    Args:
        config_path (str): Caminho para o arquivo de configuração.
        force (bool): Se True, força o download mesmo se os dados já existirem.
        
    Returns:
        bool: True se o download foi bem-sucedido, False caso contrário.
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando download de dados do Roboflow")
    
    try:
        # Criar pipeline
        pipeline = create_pipeline(config_path)
        
        # Realizar download
        success = pipeline.download_data(force=force)
        if not success:
            logger.error("Falha no download de dados")
            return False
        
        # Validar dataset
        success = pipeline.prepare_data()
        if not success:
            logger.error("Falha na validação de dados")
            return False
        
        logger.info("Download e validação de dados concluídos com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro durante o download de dados: {str(e)}")
        return False


def train_model(config_path: str = "config/settings.yml", data_yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Treina um modelo YOLOv8 com as configurações especificadas.
    
    Args:
        config_path (str): Caminho para o arquivo de configuração.
        data_yaml_path (Optional[str]): Caminho para o arquivo data.yaml.
            Se None, tenta encontrar automaticamente no diretório de dataset.
            
    Returns:
        Dict[str, Any]: Métricas e resultados do treinamento.
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando treinamento de modelo YOLOv8")
    
    try:
        # Criar pipeline
        pipeline = create_pipeline(config_path)
        
        # Se data_yaml_path foi fornecido, configurar
        if data_yaml_path:
            pipeline.config.set_data_yaml_path(data_yaml_path)
            logger.info(f"Usando data.yaml em: {data_yaml_path}")
        
        # Treinar modelo
        metrics = pipeline.train_model()
        logger.info(f"Treinamento concluído. Métricas: {metrics}")
        
        # Validar o modelo treinado
        val_metrics = pipeline.evaluate_model()
        logger.info(f"Validação concluída. Métricas: {val_metrics}")
        
        # Exportar para ONNX
        export_result = pipeline.deploy_model(format="onnx")
        logger.info(f"Modelo exportado: {export_result}")
        
        return {
            "training": metrics,
            "validation": val_metrics,
            "export": export_result
        }
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        return {"error": str(e), "success": False}


def run_full_pipeline(config_path: str = "config/settings.yml", force_download: bool = False) -> Dict[str, Any]:
    """
    Executa o pipeline completo: download, treinamento, validação e exportação.
    
    Args:
        config_path (str): Caminho para o arquivo de configuração.
        force_download (bool): Se True, força o download mesmo se os dados já existirem.
        
    Returns:
        Dict[str, Any]: Resultados de todas as etapas do pipeline.
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando pipeline completo")
    
    try:
        # Criar pipeline
        pipeline = create_pipeline(config_path)
        
        # Executar o pipeline completo
        results = pipeline.run_full_pipeline(force_download=force_download)
        logger.info(f"Pipeline completo concluído. Resultados: {results}")
        
        return results
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {str(e)}")
        return {"error": str(e), "success": False}


def main():
    """Função principal que processa argumentos e executa os comandos."""
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Gerenciamento de treinamento YOLOv8")
    parser.add_argument("--config", type=str, default="config/settings.yml", 
                        help="Caminho para o arquivo de configuração")
    parser.add_argument("--mode", type=str, required=True, choices=["download", "train", "all"],
                        help="Modo de operação: download, train ou all")
    parser.add_argument("--data_yaml", type=str, default=None,
                        help="Caminho para o arquivo data.yaml (opcional)")
    parser.add_argument("--force", action="store_true",
                        help="Força o download mesmo se os dados já existirem")
    
    # Analisar argumentos
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Executar o modo selecionado
    if args.mode == "download":
        success = download_data(args.config, force=args.force)
        return 0 if success else 1
    
    elif args.mode == "train":
        results = train_model(args.config, args.data_yaml)
        return 0 if results.get("success", False) else 1
    
    elif args.mode == "all":
        results = run_full_pipeline(args.config, force_download=args.force)
        return 0 if not "error" in results else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 