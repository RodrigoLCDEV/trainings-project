# YOLOv8 Training Project

[![CI/CD](https://github.com/seu-usuario/trainings-project/actions/workflows/ci.yml/badge.svg)](https://github.com/seu-usuario/trainings-project/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descrição

Este projeto implementa um pipeline completo para treinamento de modelos YOLOv8 utilizando datasets do Roboflow. O projeto segue princípios SOLID, com módulos bem definidos e responsabilidades claras, além de automação para garantir qualidade de código e testes.

Desenvolvido como parte de um projeto de aprendizado e demonstração de boas práticas de engenharia de software.

## Estrutura do Projeto

```
trainings-project/  
├── docs/                      # Documentação detalhada
├── .github/workflows/         # Fluxos de CI/CD (GitHub Actions)
├── logs/                      # Registros de execução e treinamento
├── Dataset_roboflow/          # Datasets do Roboflow
├── models/                    # Modelos treinados
├── src/                       # Código-fonte
│   ├── core/                  # Componentes principais
│   │   ├── data_management/   # Gerenciamento de dados (Roboflow)
│   │   ├── training/          # Configuração e execução do treino
│   │   ├── evaluation/        # Métricas e avaliação do modelo
│   │   └── inference/         # Predição em novos dados
│   └── utils/                 # Funções auxiliares
├── tests/                     # Testes automatizados
├── notebooks/                 # Jupyter notebooks para exemplos
├── scripts/                   # Scripts auxiliares
└── config/                    # Arquivos de configuração
```

## Módulos Principais

### 1. Gerenciamento de Dados (data_management)

O módulo `data_management` é responsável por baixar, validar e preparar datasets do Roboflow para treinamento.

**Componentes:**

- **RoboflowDownloader**: Classe para gerenciar o download de datasets via API do Roboflow.

**Funcionalidades:**

- Download de datasets específicos do Roboflow
- Validação de integridade dos datasets
- Verificação de permissões de diretório
- Limpeza de arquivos temporários

**Uso básico:**

```python
from src.core.data_management import RoboflowDownloader

# Inicializar
downloader = RoboflowDownloader()

# Download e validação
success, _ = downloader.download_dataset()
valid, stats = downloader.validate_dataset()

print(f"Dataset com {stats['train_images']} imagens de treino!")
```

Para mais detalhes, consulte [Documentação de Gerenciamento de Dados](docs/DATA_MANAGEMENT.md).

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/trainings-project.git
   cd trainings-project
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente:
   ```bash
   cp .env.example .env
   # Edite o arquivo .env com suas credenciais
   ```

## Uso

### Download de Datasets do Roboflow:

```bash
python scripts/download_roboflow_dataset.py
```

### Execução de Testes:

```bash
pytest                           # Todos os testes
pytest tests/test_roboflow_downloader.py  # Testes específicos
pytest --cov=src                 # Com cobertura de código
```

### Pre-commit Hooks

O projeto utiliza pre-commit hooks para garantir qualidade de código:

```bash
pip install pre-commit
pre-commit install
```

## Documentação

- [Gerenciamento de Dados](docs/DATA_MANAGEMENT.md)
- [Guia de Commits](docs/COMMIT_GUIDE.md)

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature: `git checkout -b feature/nova-feature`
3. Commit suas mudanças: `python scripts/smart_commit.py "feat: Implementa nova feature"`
4. Push para a branch: `git push origin feature/nova-feature`
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
