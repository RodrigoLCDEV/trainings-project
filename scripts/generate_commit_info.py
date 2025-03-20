#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para gerar automaticamente informações sobre as mudanças no commit.
Utilizado pelo hook de pre-commit para documentar alterações.
"""

import os
import sys
import subprocess
from datetime import datetime


def run_command(command):
    """Executa um comando e retorna a saída."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Erro ao executar comando: {command}")
        print(f"Erro: {stderr}")
        return None
    return stdout.strip()


def get_staged_files():
    """Retorna a lista de arquivos que serão incluídos no commit."""
    result = run_command("git diff --cached --name-status")
    if not result:
        return []
    
    files = []
    for line in result.split("\n"):
        if not line.strip():
            continue
        
        parts = line.split("\t")
        if len(parts) < 2:
            continue
            
        status = parts[0]
        filename = parts[1]
        
        # Traduz o status
        status_map = {
            "A": "Adicionado",
            "M": "Modificado",
            "D": "Removido",
            "R": "Renomeado",
        }
        
        status_text = status_map.get(status[0], status)
        files.append((status_text, filename))
    
    return files


def get_commit_message():
    """Obtém a mensagem de commit do arquivo COMMIT_EDITMSG."""
    try:
        git_dir = run_command("git rev-parse --git-dir")
        if not git_dir:
            return None
            
        commit_msg_file = os.path.join(git_dir, "COMMIT_EDITMSG")
        if not os.path.exists(commit_msg_file):
            return None
            
        with open(commit_msg_file, "r", encoding="utf-8") as f:
            msg = f.read().strip()
            # Remove comentários
            msg = "\n".join([line for line in msg.split("\n") 
                           if not line.startswith("#") and line.strip()])
            return msg
    except Exception as e:
        print(f"Erro ao obter mensagem de commit: {e}")
        return None


def generate_commit_info():
    """Gera informações sobre o commit atual."""
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    staged_files = get_staged_files()
    commit_message = get_commit_message()
    
    if not staged_files:
        print("Nenhum arquivo para commit.")
        return
    
    info = [
        f"Data: {today}",
        "Arquivos alterados:",
    ]
    
    for status, filename in staged_files:
        info.append(f"  - {status}: {filename}")
    
    if commit_message:
        info.append("\nMensagem do commit:")
        info.append(commit_message)
    
    info_text = "\n".join(info)
    
    # Adicionar à mensagem de commit
    commit_msg_file = os.path.join(run_command("git rev-parse --git-dir"), "COMMIT_EDITMSG")
    with open(commit_msg_file, "r", encoding="utf-8") as f:
        original_msg = f.read()
        
    # Verificar se já existe nossa seção
    if "# Informações do Commit" in original_msg:
        return
        
    with open(commit_msg_file, "w", encoding="utf-8") as f:
        # Encontra a linha de comentários e insere antes dela
        lines = original_msg.split("\n")
        comment_index = next((i for i, line in enumerate(lines) if line.startswith("#")), len(lines))
        
        new_content = "\n".join(lines[:comment_index])
        if new_content.strip():
            new_content += "\n\n"
            
        new_content += "# Informações do Commit\n" + info_text + "\n\n" + "\n".join(lines[comment_index:])
        f.write(new_content)


if __name__ == "__main__":
    generate_commit_info()
    sys.exit(0) 