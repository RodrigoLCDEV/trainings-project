# Documentação do Módulo de Gerenciamento de Dados

Este documento descreve o módulo de Gerenciamento de Dados, responsável pelo download, validação e preparação de datasets do Roboflow para treinamento de modelos YOLOv8.

## 1. Visão Geral

O módulo `src/core/data_management` contém componentes para gerenciar datasets do Roboflow, garantindo que os dados necessários estejam disponíveis e validados para o treinamento de modelos.

### Componentes Principais:

- **RoboflowDownloader**: Classe responsável por baixar datasets do Roboflow via API e validar sua integridade.

## 2. RoboflowDownloader

### 2.1 Descrição

A classe `RoboflowDownloader` gerencia o download e validação de datasets do Roboflow, verificando dependências, validando diretórios e processando os dados conforme as configurações do projeto.

### 2.2 Requisitos

- Python 3.8+
- Pacote `roboflow` instalado
- Configurações válidas em `config/settings.yml`
- Permissões de escrita no diretório de destino

### 2.3 Configuração

A classe utiliza as seguintes configurações do arquivo `config/settings.yml`:

```yaml
# Configurações do Roboflow
roboflow:
  api_key: ${ROBOFLOW_API_KEY}     # Chave de API do Roboflow (de variável de ambiente)
  workspace: ${ROBOFLOW_WORKSPACE} # Nome do workspace
  project: ${ROBOFLOW_PROJECT}     # Nome do projeto
  version: ${ROBOFLOW_VERSION}     # Versão do dataset
  format: yolov8                   # Formato de saída

# Configurações de caminhos
paths:
  processed_data_dir: Dataset_roboflow/processed # Diretório de destino
```

### 2.4 Uso Básico

```python
from src.core.data_management import RoboflowDownloader

# Inicializar o downloader
downloader = RoboflowDownloader(config_path="config/settings.yml")

# Realizar o download do dataset
success, message = downloader.download_dataset(force_download=False)

if success:
    # Validar o dataset baixado
    valid, stats = downloader.validate_dataset()
    
    if valid:
        print(f"Dataset válido com {stats['train_images']} imagens de treino!")
    else:
        print(f"Erro na validação: {stats.get('error')}")

# Limpar arquivos temporários
downloader.cleanup()
```

### 2.5 Script de Exemplo

O projeto inclui um script de exemplo em `scripts/download_roboflow_dataset.py` para facilitar o download dos datasets:

```bash
# Download básico
python scripts/download_roboflow_dataset.py

# Forçar download (mesmo se já existir)
python scripts/download_roboflow_dataset.py --force

# Usar um arquivo de configuração alternativo
python scripts/download_roboflow_dataset.py --config path/to/config.yml
```

### 2.6 Métodos Principais

| Método | Descrição |
|--------|-----------|
| `__init__(config_path="config/settings.yml")` | Inicializa o downloader com as configurações especificadas |
| `download_dataset(force_download=False)` | Realiza o download do dataset, retorna (sucesso, mensagem) |
| `validate_dataset()` | Valida o dataset baixado, retorna (válido, estatísticas) |
| `cleanup()` | Remove arquivos temporários, retorna sucesso (bool) |

### 2.7 Exceções

| Exceção | Condição |
|---------|----------|
| `ImportError` | Se o pacote `roboflow` não estiver instalado |
| `ValueError` | Se as configurações estiverem incompletas |
| `PermissionError` | Se não houver permissões de escrita no diretório de destino |

## 3. Boas Práticas

### 3.1 Variáveis de Ambiente

As credenciais do Roboflow (API Key, workspace, etc.) devem ser armazenadas em variáveis de ambiente, não em código ou arquivos de configuração versionados:

```bash
# Windows
set ROBOFLOW_API_KEY=your_api_key
set ROBOFLOW_WORKSPACE=your_workspace
set ROBOFLOW_PROJECT=your_project
set ROBOFLOW_VERSION=1

# Linux/Mac
export ROBOFLOW_API_KEY=your_api_key
export ROBOFLOW_WORKSPACE=your_workspace
export ROBOFLOW_PROJECT=your_project
export ROBOFLOW_VERSION=1
```

### 3.2 Teste de Permissões

Verifique as permissões antes de executar downloads longos:

```python
try:
    downloader = RoboflowDownloader()
    # Permissões OK se não lançar exceção
except PermissionError as e:
    print(f"Erro de permissão: {e}")
```

### 3.3 Validação de Datasets

Sempre valide os datasets após o download para garantir que a estrutura está correta:

```python
valid, stats = downloader.validate_dataset()
if not valid:
    print(f"Erro na validação: {stats.get('error')}")
    exit(1)
```

## 4. Resolução de Problemas

### 4.1 ImportError: No module named 'roboflow'

Instale o pacote roboflow:

```bash
pip install roboflow
```

### 4.2 Erro de Autenticação

Verifique se a API Key está correta e se as variáveis de ambiente estão definidas:

```bash
echo %ROBOFLOW_API_KEY%  # Windows
echo $ROBOFLOW_API_KEY   # Linux/Mac
```

### 4.3 Permissão Negada ao Criar Diretório

Verifique se você tem permissões para escrever no diretório de destino ou use um diretório alternativo:

```python
downloader = RoboflowDownloader(config_path="config/local_settings.yml")
# Onde local_settings.yml tem paths.processed_data_dir apontando para um diretório com permissões corretas
```

## 5. Testes

O módulo inclui testes unitários que podem ser executados com pytest:

```bash
# Executar testes básicos
pytest tests/test_roboflow_downloader.py

# Executar testes com relatório de cobertura
pytest --cov=src.core.data_management tests/test_roboflow_downloader.py
```

## 6. Contribuindo

Ao contribuir para este módulo, siga estas diretrizes:

1. Mantenha o princípio da Responsabilidade Única (SRP) ao adicionar novas funcionalidades
2. Escreva testes unitários para qualquer nova função ou classe
3. Documente comportamentos e parâmetros usando docstrings no formato Google
4. Verifique a cobertura de código (mínimo 80% para novas funcionalidades)
5. Siga o estilo de código PEP 8 