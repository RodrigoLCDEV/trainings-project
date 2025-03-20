#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitário para carregar configurações a partir de arquivos YAML.

Este módulo implementa funções para carregar e processar configurações
a partir de arquivos YAML, substituindo variáveis de ambiente quando necessário.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configurar logger
logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Carrega um arquivo YAML de configuração e substitui variáveis de ambiente.

    Args:
        config_path: Caminho para o arquivo de configuração YAML.

    Returns:
        Dict[str, Any]: Configurações carregadas como um dicionário.

    Raises:
        FileNotFoundError: Se o arquivo de configuração não for encontrado.
        yaml.YAMLError: Se o arquivo YAML estiver mal formatado.
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            # Carregar o YAML
            config = yaml.safe_load(file)

            # Processar o dicionário para substituir referências a variáveis de ambiente
            config = _replace_env_vars(config)

            logger.info(f"Configurações carregadas com sucesso de {config_path}")
            return config
    except FileNotFoundError as e:
        logger.error(f"Erro ao carregar configuração: {str(e)}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Erro ao analisar o arquivo YAML: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar configuração: {str(e)}")
        raise


def _replace_env_vars(config: Any) -> Any:
    """
    Substitui referências a variáveis de ambiente em um dicionário de configuração.

    Formato de referência: ${ENV_VAR} ou ${ENV_VAR:default_value}

    Args:
        config: Configuração para processar (dict, list, str, etc.)

    Returns:
        Configuração com variáveis de ambiente substituídas
    """
    if isinstance(config, dict):
        return {key: _replace_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str) and "${" in config and "}" in config:
        # Substituir variáveis de ambiente na string
        result = config
        start_pos = result.find("${")
        while start_pos >= 0:
            end_pos = result.find("}", start_pos)
            if end_pos < 0:
                break

            env_var = result[start_pos + 2 : end_pos]
            default_value = None

            # Verificar se há um valor padrão
            if ":" in env_var:
                env_var, default_value = env_var.split(":", 1)

            env_value = os.environ.get(env_var, default_value)
            if env_value is None:
                logger.warning(f"Variável de ambiente '{env_var}' não encontrada e sem valor padrão")
                # Manter a referência original se não houver valor
                env_value = f"${{{env_var}}}"

            # Substituir na string
            result = result[:start_pos] + str(env_value) + result[end_pos + 1 :]
            # Procurar a próxima ocorrência
            start_pos = result.find("${", start_pos)

        return result
    else:
        return config


def validate_config(config: Dict[str, Any], required_fields: Optional[Dict[str, Any]] = None) -> bool:
    """
    Valida se a configuração contém todos os campos obrigatórios.

    Args:
        config: Configuração a ser validada
        required_fields: Estrutura com campos obrigatórios e seus tipos esperados

    Returns:
        bool: True se a configuração for válida, False caso contrário
    """
    if required_fields is None:
        return True

    for field, expected_type in required_fields.items():
        parts = field.split(".")
        current = config

        # Navegar pela estrutura aninhada
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                logger.error(f"Campo obrigatório não encontrado: {field}")
                return False
            current = current[part]

        # Verificar o tipo se esperado
        if expected_type is not None and not isinstance(current, expected_type):
            logger.error(f"Tipo incorreto para o campo {field}. Esperado: {expected_type}, Recebido: {type(current)}")
            return False

    return True
