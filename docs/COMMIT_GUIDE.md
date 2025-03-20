# Guia de Commit com Documentação Automática

Este projeto implementa um sistema de pre-commit que automaticamente documenta as alterações realizadas em cada commit. Isso facilita o acompanhamento do histórico de desenvolvimento e melhora a colaboração entre a equipe.

## Como Funciona

O sistema consiste em dois componentes principais:

1. **Hook de Pre-commit**: Executa verificações automáticas (linting, formatação, etc.) antes que o commit seja finalizado.
2. **Hook de Prepare-commit-msg**: Adiciona automaticamente informações sobre as alterações à mensagem de commit.

## Configuração

Para configurar o sistema de pre-commit:

```bash
# Instalar o pre-commit
pip install pre-commit

# Instalar os hooks no repositório
pre-commit install --hook-type pre-commit --hook-type prepare-commit-msg
```

## Usando o Script de Commit Inteligente

Para facilitar o processo, disponibilizamos um script que automatiza toda a configuração e execução do commit:

```bash
# Windows
python scripts/smart_commit.py "Sua mensagem de commit"

# Linux/Mac
python3 scripts/smart_commit.py "Sua mensagem de commit"
```

Este script:
- Verifica se o pre-commit está instalado (e o instala se necessário)
- Configura os hooks do pre-commit
- Adiciona todos os arquivos modificados ao staging
- Realiza o commit com a mensagem fornecida
- Executa automaticamente os hooks, incluindo a documentação das alterações

## Verificações Realizadas

O pre-commit executa as seguintes verificações antes de permitir um commit:

- **Black**: Formatação de código Python
- **Flake8**: Verificação de estilo e erros no código Python
- **isort**: Ordenação de imports
- **pyupgrade**: Modernização de código Python
- **Bandit**: Verificação de segurança
- E diversas verificações básicas (arquivos YAML, EOL, etc.)

## Documentação Automática

O sistema adiciona automaticamente à sua mensagem de commit:

- Data e hora do commit
- Lista de arquivos alterados (adicionados, modificados, removidos)
- Tipo de alteração para cada arquivo

## Dicas de Boas Práticas

1. **Mensagens claras**: Mesmo com a documentação automática, escreva mensagens de commit descritivas
2. **Commits pequenos**: Prefira vários commits pequenos e focados a um único commit grande
3. **Padrão de mensagens**: Use um prefixo para indicar o tipo de alteração (feat:, fix:, docs:, etc.)

## Resolução de Problemas

Se encontrar problemas com os hooks:

1. **Erros de linting**: Execute `pre-commit run` para ver detalhes dos erros
2. **Ignorar verificações (excepcional)**: Use `git commit -m "..." --no-verify` 
3. **Reinstalar hooks**: Execute `pre-commit uninstall` seguido de `pre-commit install --hook-type pre-commit --hook-type prepare-commit-msg` 