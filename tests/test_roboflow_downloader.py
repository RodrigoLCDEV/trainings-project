#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testes para o módulo de download do Roboflow.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.core.data_management.roboflow_downloader import RoboflowDownloader


# Configuração de mock para roboflow
@pytest.fixture
def mock_roboflow():
    """Mock para a biblioteca Roboflow."""
    with patch("src.core.data_management.roboflow_downloader.Roboflow") as mock_rf:
        # Configurar a estrutura mock completa
        mock_dataset = MagicMock()
        
        mock_version = MagicMock()
        mock_version.download.return_value = mock_dataset
        
        mock_project = MagicMock()
        mock_project.version.return_value = mock_version
        
        mock_workspace = MagicMock()
        mock_workspace.project.return_value = mock_project
        
        mock_rf_instance = MagicMock()
        mock_rf_instance.workspace.return_value = mock_workspace
        
        mock_rf.return_value = mock_rf_instance
        
        yield mock_rf


# Configuração de mock para as configurações
@pytest.fixture
def mock_config():
    """Mock para configurações."""
    config = {
        "roboflow": {
            "api_key": "dummy_api_key",
            "workspace": "dummy_workspace",
            "project": "dummy_project",
            "version": "1",
            "format": "yolov8"
        },
        "paths": {
            "processed_data_dir": "/tmp/dummy_data_dir"
        }
    }
    
    with patch("src.core.data_management.roboflow_downloader.load_yaml_config") as mock_load:
        mock_load.return_value = config
        yield config


# Configuração de mock para verificação de diretório
@pytest.fixture
def mock_directory_permissions():
    """Mock para verificação de permissões de diretório."""
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("pathlib.Path.unlink") as mock_unlink:
        
        mock_exists.return_value = True
        yield mock_exists, mock_mkdir, mock_file, mock_unlink


@pytest.fixture
def downloader(mock_config, mock_directory_permissions):
    """Cria uma instância do RoboflowDownloader com mocks."""
    return RoboflowDownloader()


def test_init(mock_config, mock_directory_permissions):
    """Testa a inicialização do downloader."""
    # Verificar se a inicialização ocorre corretamente
    downloader = RoboflowDownloader()
    
    assert downloader.api_key == "dummy_api_key"
    # Usar um caminho normalizado para comparação entre sistemas operacionais
    assert downloader.dest_dir.as_posix() == "/tmp/dummy_data_dir"


def test_check_directory_permissions_success(mock_config):
    """Testa a verificação de permissões quando tudo está correto."""
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("pathlib.Path.unlink") as mock_unlink:
        
        mock_exists.return_value = True
        
        downloader = RoboflowDownloader()
        # Se não lançar exceção, está funcionando
        assert downloader is not None


def test_check_directory_permissions_mkdir_failure(mock_config):
    """Testa a falha na criação do diretório."""
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.mkdir") as mock_mkdir:
        
        mock_exists.return_value = False
        mock_mkdir.side_effect = PermissionError("Sem permissão para criar diretório")
        
        with pytest.raises(PermissionError):
            RoboflowDownloader()


def test_check_directory_permissions_write_failure(mock_config):
    """Testa a falha na escrita no diretório."""
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open") as mock_file:
        
        mock_exists.return_value = True
        mock_file.side_effect = PermissionError("Sem permissão para escrever")
        
        with pytest.raises(PermissionError):
            RoboflowDownloader()


def test_download_dataset_success(downloader, mock_roboflow):
    """Testa o download bem-sucedido."""
    # Configurar mock para verificar se o dataset já existe
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False
        
        success, message = downloader.download_dataset()
        
        assert success is True
        assert "sucesso" in message
        # Verificar se o método foi chamado com os parâmetros corretos
        mock_roboflow.assert_called_once_with(api_key="dummy_api_key")


def test_download_dataset_already_exists(downloader):
    """Testa o caso onde o dataset já existe."""
    # Configurar mock para simular que o dataset já existe
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        
        success, message = downloader.download_dataset(force_download=False)
        
        assert success is True
        assert "já existe" in message


def test_download_dataset_force(downloader, mock_roboflow):
    """Testa o download forçado mesmo quando o dataset já existe."""
    # Configurar mock para simular que o dataset já existe
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        
        success, message = downloader.download_dataset(force_download=True)
        
        assert success is True
        assert "sucesso" in message
        # Verificar se o download foi chamado mesmo com o dataset existente
        downloader_rf = mock_roboflow.return_value
        workspace = downloader_rf.workspace.return_value
        project = workspace.project.return_value
        version = project.version.return_value
        assert version.download.called


def test_download_dataset_failure(downloader, mock_roboflow):
    """Testa falha no download."""
    # Configurar mock para simular falha no download
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False
        
        # Configurar exceção durante o download
        downloader_rf = mock_roboflow.return_value
        workspace = downloader_rf.workspace.return_value
        project = workspace.project.return_value
        version = project.version.return_value
        version.download.side_effect = Exception("Erro de API")
        
        success, message = downloader.download_dataset()
        
        assert success is False
        assert "Erro" in message


def test_validate_dataset_success(downloader):
    """Testa a validação bem-sucedida do dataset."""
    # Mockear estrutura de diretórios e arquivos
    with patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.glob") as mock_glob, \
         patch("builtins.open", mock_open(read_data="names: ['class1', 'class2']")), \
         patch("yaml.safe_load") as mock_yaml_load:
        
        mock_exists.return_value = True
        mock_glob.return_value = [Path("image1.jpg"), Path("image2.jpg")]
        mock_yaml_load.return_value = {"names": ["class1", "class2"]}
        
        valid, stats = downloader.validate_dataset()
        
        assert valid is True
        assert "classes" in stats
        assert len(stats["classes"]) == 2


def test_validate_dataset_missing_directory(downloader):
    """Testa a validação quando o diretório do dataset não existe."""
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False
        
        valid, stats = downloader.validate_dataset()
        
        assert valid is False
        assert "error" in stats
        assert "não encontrado" in stats["error"]


def test_cleanup_success(downloader):
    """Testa a limpeza bem-sucedida."""
    with patch("pathlib.Path.glob") as mock_glob, \
         patch("pathlib.Path.is_dir") as mock_is_dir, \
         patch("shutil.rmtree") as mock_rmtree:
        
        # Simular dois diretórios temporários
        tmp_dir1 = Path("/tmp/dummy_data_dir/tmp_123")
        tmp_dir2 = Path("/tmp/dummy_data_dir/some_tmp_dir")
        mock_glob.return_value = [tmp_dir1, tmp_dir2]
        
        # Configurar para que is_dir retorne True
        mock_is_dir.return_value = True
        
        result = downloader.cleanup()
        
        assert result is True
        assert mock_rmtree.call_count == 2


def test_cleanup_failure(downloader):
    """Testa falha na limpeza."""
    with patch("pathlib.Path.glob") as mock_glob, \
         patch("pathlib.Path.is_dir") as mock_is_dir, \
         patch("shutil.rmtree") as mock_rmtree:
        
        # Simular diretório temporário
        tmp_dir = Path("/tmp/dummy_data_dir/tmp_123")
        mock_glob.return_value = [tmp_dir]
        
        # Configurar para que is_dir retorne True
        mock_is_dir.return_value = True
        
        # Simular erro ao remover
        mock_rmtree.side_effect = PermissionError("Sem permissão para remover")
        
        result = downloader.cleanup()
        
        assert result is False 