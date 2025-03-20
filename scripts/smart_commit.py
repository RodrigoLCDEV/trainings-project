#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para facilitar o processo de commit com documentação automática.
Instala o pre-commit se necessário e executa os hooks automaticamente.
"""

import os
import sys
import subprocess
import argparse


def run_command(command, show_output=True):
    """Executa um comando e retorna a saída."""
    print(f"Executando: {command}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
    )
    stdout, stderr = process.communicate()
    
    if show_output:
        if stdout:
            print(stdout)
        if stderr and process.returncode != 0:
            print(f"ERRO: {stderr}")
    
    return process.returncode, stdout, stderr


def check_pre_commit_installed():
    """Verifica se o pre-commit está instalado e o instala se necessário."""
    try:
        import pre_commit
        return True
    except ImportError:
        print("pre-commit não está instalado. Instalando...")
        code, _, _ = run_command("pip install pre-commit")
        if code != 0:
            print("Falha ao instalar pre-commit. Instale manualmente: pip install pre-commit")
            return False
        print("pre-commit instalado com sucesso!")
        return True


def setup_pre_commit():
    """Configura o pre-commit para o repositório."""
    # Verifica se o pre-commit já está instalado no repositório
    if os.path.exists(".git/hooks/pre-commit"):
        print("pre-commit já está configurado para este repositório.")
        return True
    
    print("Configurando pre-commit para o repositório...")
    code, _, _ = run_command("pre-commit install --hook-type pre-commit --hook-type prepare-commit-msg")
    
    if code != 0:
        print("Falha ao configurar pre-commit. Configure manualmente: pre-commit install")
        return False
        
    print("pre-commit configurado com sucesso!")
    return True


def commit_changes(message):
    """Faz o commit das alterações com a mensagem fornecida."""
    # Adiciona todas as alterações ao staging
    code, _, _ = run_command("git add .")
    if code != 0:
        print("Falha ao adicionar arquivos ao staging.")
        return False
        
    # Faz o commit com a mensagem fornecida
    code, _, _ = run_command(f'git commit -m "{message}"')
    if code != 0:
        print("O commit falhou. Verifique os hooks do pre-commit.")
        return False
        
    print("Commit realizado com sucesso!")
    return True


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Realiza commit com documentação automática")
    parser.add_argument("message", nargs="?", help="Mensagem de commit")
    args = parser.parse_args()
    
    if not args.message:
        print("Por favor, forneça uma mensagem de commit.")
        return 1
    
    # Verifica se o pré-commit está instalado
    if not check_pre_commit_installed():
        return 1
        
    # Configura o pre-commit
    if not setup_pre_commit():
        return 1
        
    # Faz o commit
    if not commit_changes(args.message):
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 