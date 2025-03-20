#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para gerenciamento de downloads de datasets do Roboflow.

Este módulo implementa a classe RoboflowDownloader para gerenciar
o download e validação de datasets do Roboflow via API.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

from src.utils.config_loader import load_yaml_config, validate_config


class RoboflowDownloader:
    """
    Classe para gerenciar o download de datasets do Roboflow.

    Esta classe implementa métodos para realizar o download de datasets
    do Roboflow, verificando dependências, validando diretórios e 
    processando os dados conforme as configurações.

    Attributes:
        config (Dict[str, Any]): Configurações do projeto.
        logger (logging.Logger): Logger para registrar eventos.
        dest_dir (Path): Diretório de destino para os datasets.
        api_key (str): Chave de API do Roboflow.
    """

    def __init__(self, config_path: str = "config/settings.yml"):
        """
        Inicializa o downloader com as configurações especificadas.

        Args:
            config_path (str): Caminho para o arquivo de configuração.
        
        Raises:
            ImportError: Se o pacote roboflow não estiver instalado.
            ValueError: Se as configurações estiverem incompletas.
            PermissionError: Se não houver permissões de escrita no diretório de destino.
        """
        self.logger = logging.getLogger(__name__)
        
        # Verificar se o pacote roboflow está instalado
        if not ROBOFLOW_AVAILABLE:
            self.logger.error("Pacote 'roboflow' não instalado. Use: pip install roboflow")
            raise ImportError("Pacote 'roboflow' é necessário para esta funcionalidade")
        
        # Carregar configurações
        self.config = load_yaml_config(config_path)
        
        # Validar configurações
        required_fields = {
            "roboflow.api_key": str,
            "roboflow.workspace": str,
            "roboflow.project": str,
            "roboflow.version": str,
            "roboflow.format": str,
            "paths.processed_data_dir": str
        }
        
        if not validate_config(self.config, required_fields):
            raise ValueError("Configurações incompletas para o downloader do Roboflow")
        
        # Configurar variáveis de instância
        self.api_key = self.config["roboflow"]["api_key"]
        self.dest_dir = Path(self.config["paths"]["processed_data_dir"])
        
        # Verificar permissões de escrita no diretório de destino
        self._check_directory_permissions()
    
    def _check_directory_permissions(self) -> None:
        """
        Verifica se o diretório de destino existe e tem permissões de escrita.
        
        Raises:
            PermissionError: Se o diretório não puder ser criado ou não tiver permissões de escrita.
        """
        # Criar diretório se não existir
        if not self.dest_dir.exists():
            try:
                self.dest_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Diretório criado: {self.dest_dir}")
            except Exception as e:
                self.logger.error(f"Falha ao criar diretório {self.dest_dir}: {str(e)}")
                raise PermissionError(f"Não foi possível criar o diretório de destino: {self.dest_dir}")
        
        # Verificar permissões de escrita
        test_file = self.dest_dir / ".write_test"
        try:
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()  # Remover arquivo de teste
        except Exception as e:
            self.logger.error(f"Sem permissão de escrita em {self.dest_dir}: {str(e)}")
            raise PermissionError(f"Sem permissão de escrita no diretório: {self.dest_dir}")
    
    def download_dataset(self, force_download: bool = False) -> Tuple[bool, str]:
        """
        Realiza o download do dataset do Roboflow.
        
        Args:
            force_download (bool): Se True, força o download mesmo se os dados já existirem.
            
        Returns:
            Tuple[bool, str]: Tupla (sucesso, mensagem) indicando o resultado da operação.
        """
        # Verificar se os dados já existem
        data_dir = self.dest_dir / self.config["roboflow"]["project"]
        if data_dir.exists() and not force_download:
            self.logger.info(f"Dataset já existe em {data_dir}. Use force_download=True para baixar novamente.")
            return True, f"Dataset já existe em {data_dir}"
        
        try:
            # Inicializar cliente do Roboflow
            rf = Roboflow(api_key=self.api_key)
            
            # Acessar o workspace
            workspace = rf.workspace(self.config["roboflow"]["workspace"])
            
            # Acessar o projeto
            project = workspace.project(self.config["roboflow"]["project"])
            
            # Baixar a versão específica do dataset
            dataset = project.version(
                self.config["roboflow"]["version"]
            ).download(
                self.config["roboflow"]["format"],
                location=str(self.dest_dir)
            )
            
            self.logger.info(
                f"Dataset baixado com sucesso: {self.config['roboflow']['project']} "
                f"v{self.config['roboflow']['version']}"
            )
            
            return True, f"Dataset baixado com sucesso para {self.dest_dir}"
            
        except Exception as e:
            error_msg = f"Erro ao baixar dataset: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def validate_dataset(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Valida o dataset baixado, verificando arquivos e estrutura.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: Tupla (válido, estatísticas) com o resultado da validação.
        """
        project_name = self.config["roboflow"]["project"]
        data_dir = self.dest_dir / project_name
        
        # Verificar se o diretório existe
        if not data_dir.exists():
            return False, {"error": f"Diretório do dataset não encontrado: {data_dir}"}
        
        # Verificar estrutura esperada do YOLOv8
        required_dirs = ["train", "valid", "test"]
        required_files = ["data.yaml"]
        
        # Estatísticas do dataset
        stats = {
            "valid": True,
            "project": project_name,
            "version": self.config["roboflow"]["version"],
            "format": self.config["roboflow"]["format"],
            "train_images": 0,
            "valid_images": 0,
            "test_images": 0,
            "classes": []
        }
        
        # Verificar diretórios e arquivos obrigatórios
        for dir_name in required_dirs:
            dir_path = data_dir / dir_name
            if not dir_path.exists():
                stats["valid"] = False
                stats["error"] = f"Diretório obrigatório não encontrado: {dir_name}"
                return False, stats
            
            # Contar imagens nos diretórios train/valid/test
            images_count = len(list(dir_path.glob("images/*.jpg"))) + len(list(dir_path.glob("images/*.png")))
            stats[f"{dir_name}_images"] = images_count
        
        for file_name in required_files:
            file_path = data_dir / file_name
            if not file_path.exists():
                stats["valid"] = False
                stats["error"] = f"Arquivo obrigatório não encontrado: {file_name}"
                return False, stats
        
        # Ler classes do arquivo data.yaml
        try:
            from yaml import safe_load
            with open(data_dir / "data.yaml", "r") as f:
                data_yaml = safe_load(f)
                if "names" in data_yaml:
                    stats["classes"] = data_yaml["names"]
        except Exception as e:
            self.logger.warning(f"Não foi possível ler as classes do arquivo data.yaml: {str(e)}")
        
        return stats["valid"], stats
    
    def cleanup(self) -> bool:
        """
        Remove dados temporários e libera espaço em disco.
        
        Returns:
            bool: True se a limpeza foi bem-sucedida, False caso contrário.
        """
        try:
            # Limpar apenas diretórios temporários, não os dados principais
            tmp_dirs = []
            for d in self.dest_dir.glob("*tmp*"):
                if d.is_dir() and "tmp" in d.name.lower():
                    tmp_dirs.append(d)
            
            for tmp_dir in tmp_dirs:
                self.logger.info(f"Removendo diretório temporário: {tmp_dir}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                
            return True
        except Exception as e:
            self.logger.error(f"Erro ao limpar dados temporários: {str(e)}")
            return False 