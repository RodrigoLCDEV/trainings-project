#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitário para configuração de logging.

Este módulo configura o sistema de logging a partir de um arquivo de configuração YAML,
fornecendo uma interface consistente para registro de logs em toda a aplicação.
"""

import os
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any

from .config_loader import load_yaml_config


def setup_logging(
    logging_config_path: str = "config/logging.yml",
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG",
) -> None:
    """
    Configura o logging a partir de um arquivo YAML.

    Args:
        logging_config_path: Caminho para o arquivo de configuração de logging (YAML)
        default_level: Nível de logging padrão se a configuração falhar
        env_key: Variável de ambiente que pode substituir o caminho do arquivo de configuração
    """
    # Verificar se existe uma variável de ambiente com o caminho de configuração
    path = os.getenv(env_key, logging_config_path)

    try:
        # Garantir que o diretório de logs existe
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Carregar configuração
        config = load_yaml_config(path)

        # Processar configurações de paths relativos
        _process_file_handlers(config)

        # Aplicar configuração
        logging.config.dictConfig(config)

        logging.info(f"Logging configurado com sucesso usando {path}")
    except Exception as e:
        # Em caso de falha, usar configuração básica
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.warning(f"Erro ao configurar logging: {str(e)}. Usando configuração básica.")


def _process_file_handlers(config: Dict[str, Any]) -> None:
    """
    Processa configurações de caminhos de arquivo nos handlers de logging.

    Substitui variáveis como ${logs_dir} nos caminhos de arquivo e cria os diretórios conforme necessário.

    Args:
        config: Configuração de logging carregada
    """
    if "handlers" not in config:
        return

    # Para cada handler, verificar se ele usa um arquivo
    for handler_config in config["handlers"].values():
        if "filename" in handler_config:
            filename = handler_config["filename"]

            # Substituir variáveis como ${logs_dir}
            if "${logs_dir}" in filename:
                filename = filename.replace("${logs_dir}", "logs")
                handler_config["filename"] = filename

            # Garantir que o diretório do arquivo existe
            log_file = Path(filename)
            log_file.parent.mkdir(exist_ok=True, parents=True)


def get_logger(name: str) -> logging.Logger:
    """
    Obtém um logger configurado para um módulo específico.

    Args:
        name: Nome do logger (geralmente __name__ do módulo)

    Returns:
        logging.Logger: Logger configurado
    """
    return logging.getLogger(name)
