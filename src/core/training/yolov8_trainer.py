#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para treinamento de modelos YOLOv8.

Este módulo implementa classes para treinar modelos YOLOv8
usando a biblioteca ultralytics e injeção de dependência.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from src.core.training.interface import ITrainer
from src.core.training.training_config_pydantic import TrainingConfig


class YOLOv8Trainer(ITrainer):
    """
    Classe para treinar modelos YOLOv8 usando injeção de dependência.

    Esta classe implementa a interface ITrainer para treinar modelos
    YOLOv8 com base em uma configuração fornecida externamente.

    Attributes:
        config (TrainingConfig): Configuração para o treinamento.
        logger (logging.Logger): Logger para registrar eventos.
        model_path (Optional[Path]): Caminho para o modelo pré-treinado ou treinado.
        results_dir (Optional[Path]): Diretório para os resultados do treinamento.
    """

    def __init__(self, config: TrainingConfig):
        """
        Inicializa o treinador com a configuração fornecida.

        Args:
            config (TrainingConfig): Configuração para o treinamento.
        
        Raises:
            ImportError: Se a biblioteca ultralytics não estiver instalada.
            ValueError: Se a configuração for inválida.
        """
        self.logger = logging.getLogger(__name__)
        
        # Verificar disponibilidade do ultralytics
        if not ULTRALYTICS_AVAILABLE:
            self.logger.error("Biblioteca 'ultralytics' não instalada. Use: pip install ultralytics")
            raise ImportError("Biblioteca 'ultralytics' é necessária para esta funcionalidade")
        
        # Validar e armazenar configuração
        if not isinstance(config, TrainingConfig):
            self.logger.error("A configuração fornecida não é uma instância de TrainingConfig")
            raise ValueError("A configuração deve ser uma instância de TrainingConfig")
        
        self.config = config
        self.model = None
        self.model_path = None
        self.results_dir = None
        
        self.logger.info(f"YOLOv8Trainer inicializado com configuração: {self.config}")
    
    def _load_model(self) -> YOLO:
        """
        Carrega o modelo YOLOv8 baseado na configuração.
        
        Returns:
            YOLO: Instância do modelo YOLOv8.
            
        Raises:
            FileNotFoundError: Se o modelo não for encontrado.
        """
        try:
            model_name = self.config.get_yolo_model_name()
            
            # Se já tivermos um model_path definido, use-o
            if self.model_path and Path(self.model_path).exists():
                self.logger.info(f"Carregando modelo treinado: {self.model_path}")
                model = YOLO(str(self.model_path))
            else:
                # Caso contrário, carregue o modelo padrão
                self.logger.info(f"Carregando modelo pré-treinado: {model_name}")
                model = YOLO(model_name)
            
            self.model = model
            return model
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise FileNotFoundError(f"Não foi possível carregar o modelo: {str(e)}")
    
    def train(self, data_yaml_path: Optional[Union[str, Path]] = None, resume: bool = False) -> Dict[str, Any]:
        """
        Treina o modelo YOLOv8 com os parâmetros configurados.
        
        Args:
            data_yaml_path (Optional[Union[str, Path]]): Caminho para o arquivo data.yaml.
                Se None, usa o configurado anteriormente.
            resume (bool): Se True, tenta retomar um treinamento anterior.
            
        Returns:
            Dict[str, Any]: Métricas e resultados do treinamento.
            
        Raises:
            ValueError: Se o data_yaml_path não estiver definido.
            RuntimeError: Se houver falha durante o treinamento.
        """
        # Definir data_yaml_path se fornecido
        if data_yaml_path:
            self.config.set_data_yaml_path(data_yaml_path)
        
        # Verificar se data_yaml_path está definido
        if not self.config.paths.data_yaml_path:
            self.logger.error("Data YAML não configurado")
            raise ValueError("Data YAML não configurado. Forneça data_yaml_path ou use config.set_data_yaml_path().")
        
        try:
            # Carregar o modelo
            model = self._load_model()
            
            # Obter argumentos de treinamento
            training_args = self.config.get_training_args()
            
            # Ajustar para resumir treinamento se solicitado
            if resume and self.results_dir and Path(self.results_dir).exists():
                training_args["resume"] = True
            
            # Log dos parâmetros de treinamento
            self.logger.info(f"Iniciando treinamento com parâmetros: {training_args}")
            
            # Executar treinamento
            results = model.train(**training_args)
            
            # Armazenar caminho para o melhor modelo
            if hasattr(results, "best") and Path(results.best).exists():
                self.model_path = Path(results.best)
                self.logger.info(f"Melhor modelo salvo em: {self.model_path}")
            
            # Armazenar diretório de resultados
            if hasattr(results, "save_dir") and Path(results.save_dir).exists():
                self.results_dir = Path(results.save_dir)
                self.logger.info(f"Resultados salvos em: {self.results_dir}")
            
            # Extrair e retornar métricas
            metrics = self._extract_metrics(results)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro durante o treinamento: {str(e)}")
            raise RuntimeError(f"Falha no treinamento: {str(e)}")
    
    def _extract_metrics(self, results: Any) -> Dict[str, Any]:
        """
        Extrai métricas relevantes dos resultados do treinamento.
        
        Args:
            results (Any): Resultados retornados pelo método train() do ultralytics.
            
        Returns:
            Dict[str, Any]: Dicionário com métricas de treinamento.
        """
        metrics = {
            "success": True,
            "model_path": str(self.model_path) if self.model_path else None,
            "results_dir": str(self.results_dir) if self.results_dir else None,
        }
        
        # Extrair métricas específicas se disponíveis
        if hasattr(results, "results_dict"):
            metrics.update({
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "val_loss": results.results_dict.get("val/box_loss", 0),
                "epochs_completed": results.results_dict.get("epoch", 0),
            })
        
        return metrics
    
    def validate(self) -> Dict[str, Any]:
        """
        Valida o modelo treinado no conjunto de validação.
        
        Returns:
            Dict[str, Any]: Métricas de validação.
            
        Raises:
            FileNotFoundError: Se o modelo treinado não for encontrado.
            ValueError: Se o data_yaml_path não estiver definido.
        """
        if not self.model_path or not Path(self.model_path).exists():
            self.logger.error("Modelo treinado não encontrado")
            raise FileNotFoundError("Modelo treinado não encontrado. Execute train() primeiro.")
        
        if not self.config.paths.data_yaml_path:
            self.logger.error("Data YAML não configurado")
            raise ValueError("Data YAML não configurado.")
        
        try:
            # Carregar o modelo se ainda não estiver carregado
            if not self.model:
                self.model = YOLO(str(self.model_path))
            
            # Executar validação
            self.logger.info(f"Validando modelo: {self.model_path}")
            results = self.model.val(data=str(self.config.paths.data_yaml_path))
            
            # Extrair métricas
            metrics = {
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
            }
            
            self.logger.info(f"Validação concluída: mAP50={metrics['mAP50']:.4f}, mAP50-95={metrics['mAP50-95']:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro durante a validação: {str(e)}")
            raise RuntimeError(f"Falha na validação: {str(e)}")
    
    def export_model(self, 
                    format: str = "onnx", 
                    output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Exporta o modelo treinado para outros formatos.
        
        Args:
            format (str): Formato de exportação (onnx, torchscript, coreml, etc).
            output_dir (Optional[Union[str, Path]]): Diretório para salvar o modelo exportado.
                Se None, usa o mesmo diretório do modelo original.
                
        Returns:
            Dict[str, Any]: Informações sobre o modelo exportado.
            
        Raises:
            FileNotFoundError: Se o modelo treinado não for encontrado.
            ValueError: Se o formato não for suportado.
        """
        if not self.model_path or not Path(self.model_path).exists():
            self.logger.error("Modelo treinado não encontrado")
            raise FileNotFoundError("Modelo treinado não encontrado. Execute train() primeiro.")
        
        # Validar formato
        supported_formats = ["onnx", "torchscript", "openvino", "coreml", "tflite", "saved_model", "pb", "trt"]
        if format.lower() not in supported_formats:
            self.logger.error(f"Formato não suportado: {format}")
            raise ValueError(f"Formato não suportado: {format}. Formatos válidos: {supported_formats}")
        
        try:
            # Carregar o modelo se ainda não estiver carregado
            if not self.model:
                self.model = YOLO(str(self.model_path))
            
            # Definir diretório de saída
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(self.model_path).parent
            
            # Exportar modelo
            self.logger.info(f"Exportando modelo para formato {format}")
            exported = self.model.export(format=format, imgsz=self.config.hyperparameters.img_size)
            
            # Verificar e mover o modelo exportado se necessário
            exported_path = Path(str(exported))
            if output_dir != exported_path.parent:
                target_path = output_dir / exported_path.name
                shutil.copy2(exported_path, target_path)
                self.logger.info(f"Modelo copiado para: {target_path}")
                exported_path = target_path
            
            return {
                "success": True,
                "format": format,
                "exported_path": str(exported_path),
                "model_size_mb": round(exported_path.stat().st_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Erro durante a exportação: {str(e)}")
            raise RuntimeError(f"Falha na exportação: {str(e)}")
    
    def __str__(self) -> str:
        """
        Retorna uma representação em string do treinador.
        
        Returns:
            str: Descrição do treinador e configuração.
        """
        trainer_str = (
            f"YOLOv8Trainer:\n"
            f"{str(self.config)}"
        )
        
        if self.model_path:
            trainer_str += f"  Modelo: {Path(self.model_path).name}\n"
        
        return trainer_str 