#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testes para o módulo de carregamento de configurações.
"""

import os
import tempfile
from pathlib import Path
import pytest
from src.utils.config_loader import load_yaml_config, _replace_env_vars, validate_config


def test_load_yaml_config():
    """Testa o carregamento de configurações a partir de arquivos YAML."""
    # Criar um arquivo YAML temporário para teste
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp_file:
        yaml_content = """
        project:
          name: Test Project
          version: 1.0.0
        paths:
          data_dir: data
        """
        temp_file.write(yaml_content.encode("utf-8"))
        temp_path = temp_file.name

    try:
        # Carregar a configuração
        config = load_yaml_config(temp_path)

        # Verificar se os valores foram carregados corretamente
        assert config["project"]["name"] == "Test Project"
        assert config["project"]["version"] == "1.0.0"
        assert config["paths"]["data_dir"] == "data"
    finally:
        # Limpar o arquivo temporário
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_load_yaml_config_file_not_found():
    """Testa o comportamento quando o arquivo de configuração não é encontrado."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("non_existent_file.yml")


def test_replace_env_vars():
    """Testa a substituição de variáveis de ambiente nas configurações."""
    # Configurar variáveis de ambiente para teste
    os.environ["TEST_VAR"] = "test_value"
    os.environ["ANOTHER_VAR"] = "another_value"

    # Testar substituição em string simples
    assert _replace_env_vars("${TEST_VAR}") == "test_value"

    # Testar substituição em string com texto ao redor
    assert _replace_env_vars("prefix_${TEST_VAR}_suffix") == "prefix_test_value_suffix"

    # Testar substituição em dicionário
    test_dict = {
        "key1": "${TEST_VAR}",
        "key2": "prefix_${ANOTHER_VAR}_suffix",
        "key3": "no_replacement",
    }
    replaced_dict = _replace_env_vars(test_dict)
    assert replaced_dict["key1"] == "test_value"
    assert replaced_dict["key2"] == "prefix_another_value_suffix"
    assert replaced_dict["key3"] == "no_replacement"

    # Testar substituição em lista
    test_list = ["${TEST_VAR}", "prefix_${ANOTHER_VAR}_suffix", "no_replacement"]
    replaced_list = _replace_env_vars(test_list)
    assert replaced_list[0] == "test_value"
    assert replaced_list[1] == "prefix_another_value_suffix"
    assert replaced_list[2] == "no_replacement"


def test_validate_config():
    """Testa a validação de configurações."""
    # Configuração válida
    config = {
        "project": {"name": "Test", "version": "1.0.0"},
        "paths": {"data_dir": "data"},
        "training": {"epochs": 50},
    }

    # Validação básica (sem requisitos)
    assert validate_config(config) is True

    # Validação com requisitos existentes
    required_fields = {
        "project.name": str,
        "paths.data_dir": str,
        "training.epochs": int,
    }
    assert validate_config(config, required_fields) is True

    # Validação com campo faltando
    missing_fields = {
        "project.name": str,
        "paths.data_dir": str,
        "training.missing": int,
    }
    assert validate_config(config, missing_fields) is False

    # Validação com tipo incorreto
    wrong_types = {
        "project.name": str,
        "paths.data_dir": str,
        "training.epochs": str,  # Deveria ser int
    }
    assert validate_config(config, wrong_types) is False 