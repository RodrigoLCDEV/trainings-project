#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pacote de utilitários para o projeto de treinamento YOLOv8.

Este pacote contém utilitários compartilhados em toda a aplicação,
como configurações, logging, manipulação de arquivos e outras funções comuns.
"""

from .config_loader import load_yaml_config, validate_config
from .logger import setup_logging, get_logger
