# Guia Completo - Treinamento de Modelo YOLO com Roboflow

Este guia consolidado contém todas as informações necessárias para:
1. Configurar o ambiente
2. Treinar modelos YOLO com Roboflow
3. Resolver problemas comuns
4. Utilizar as funcionalidades avançadas do projeto

## Sumário
- [Introdução](#introdução)
- [Configuração Inicial](#configuração-inicial)
- [Formas de Uso](#formas-de-uso)
- [Treinamento Local (CPU)](#treinamento-local-cpu)
- [Treinamento no Google Colab (GPU)](#treinamento-no-google-colab-gpu)
- [Solução de Problemas Comuns](#solução-de-problemas-comuns)
- [Funcionalidades Avançadas](#funcionalidades-avançadas)
  - [Treinamento com Múltiplos Datasets](#treinamento-com-múltiplos-datasets)
  - [Configurações Avançadas para Hardware Específico](#configurações-avançadas-para-hardware-específico)

## Introdução

Este projeto fornece uma estrutura unificada para trabalhar com modelos YOLO e datasets do Roboflow. O script principal `main_universal.py` foi projetado para funcionar em qualquer ambiente (local ou Google Colab) e oferece tanto uma interface de linha de comando quanto um menu interativo.

## Configuração Inicial

### Pré-requisitos
- Python 3.8+
- Conta no Roboflow com um projeto configurado
- Arquivo `.env` com as credenciais do Roboflow

### Instalação

1. Configure o ambiente virtual e instale as dependências:
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual (Windows)
.\venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

2. Configure o arquivo `.env` na raiz do projeto:
```
PRIVATE_API_KEY=sua_chave_api_roboflow
PUBLISHABLE_API_KEY=sua_chave_publica
ID_PROJECT=seu_id_projeto
ROBOFLOW_WORKSPACE=seu_workspace
ROBOFLOW_VERSION=1
```

3. Execute o script de configuração:
```bash
python main_universal.py --action setup
# ou use o script de atalho:
.\scripts\execute_setup.bat
```

## Formas de Uso

### 1. Script Universal (Local e Colab)

O script `main_universal.py` detecta automaticamente o ambiente e se adapta:

```bash
# Usar o menu interativo
python main_universal.py

# Ações específicas via linha de comando
python main_universal.py --action train --epochs 10  # Treinar por 10 épocas
python main_universal.py --action test --confidence 0.5  # Testar com confiança 0.5
python main_universal.py --action all  # Executar todo o fluxo
```

### 2. Notebook Jupyter no Google Colab

Para treinar usando GPU gratuita no Google Colab:

1. Faça upload do arquivo `colab_runner.ipynb` para o Google Colab
2. Siga as instruções em cada célula do notebook
3. O notebook irá:
   - Configurar o ambiente
   - Verificar disponibilidade de GPU
   - Baixar o dataset
   - Treinar o modelo
   - Testar e validar o modelo
   - Salvar os resultados no Google Drive (opcional)

## Treinamento Local (CPU)

O treinamento em CPU é significativamente mais lento, mas ainda é viável para datasets pequenos e testes iniciais.

### Otimizações para CPU

As configurações recomendadas para treinar em CPU são:
- Modelo: YOLOv8n (o mais leve e rápido)
- Tamanho das imagens: 320px (reduzido para maior velocidade)
- Batch size: 1 (para compatibilidade com CPU)
- Épocas: 5-10 (para teste inicial)

### Executando o Treinamento Local

```bash
python main_universal.py --action train --device cpu --batch 1 --imgsz 320
```

### Limitações do Treinamento em CPU

- **Velocidade**: 10-50x mais lento que uma GPU
- **Tamanho das imagens**: Limitado a 320px (versus 640px+ com GPU)
- **Precisão**: Geralmente 25-35% após várias horas de treinamento
- **Batch size**: Limitado a 1, o que afeta o aprendizado

## Treinamento no Google Colab (GPU)

O treinamento no Google Colab com GPU é altamente recomendado para obter resultados melhores e mais rápidos.

### Vantagens do Colab
- GPU gratuita (NVIDIA T4 ou P100)
- 30-50x mais rápido que CPU
- Suporta imagens maiores (640px+)
- Batch size maior (8-16)
- Melhor precisão

### Usando o Notebook Colab

1. Abra o arquivo `colab_runner.ipynb` no Google Colab
2. Ative a aceleração por GPU:
   - Editar > Configurações do notebook > Acelerador de hardware > GPU
3. Execute as células do notebook em sequência

## Solução de Problemas Comuns

### Problema com PyTorch 2.6+

A partir do PyTorch 2.6, o comportamento padrão da função `torch.load()` foi alterado por questões de segurança, causando erros ao carregar modelos YOLO.

**Solução:**
```bash
# Definir variável de ambiente antes de executar o script
set TORCH_LOAD_WEIGHTS_ONLY=0  # Windows CMD
$env:TORCH_LOAD_WEIGHTS_ONLY = 0  # Windows PowerShell
TORCH_LOAD_WEIGHTS_ONLY=0 python main_universal.py  # Linux/macOS
```

Ou use o script `fix_torch_loading.py` que aplica automaticamente os patches necessários.

### Erros de Memória

Se encontrar erros de memória durante o treinamento:

1. Reduza o tamanho do batch:
   ```bash
   python main_universal.py --action train --batch 1
   ```

2. Reduza o tamanho das imagens:
   ```bash
   python main_universal.py --action train --imgsz 320
   ```

3. Desative as técnicas de aumento de dados no arquivo `src/utils/config.py`:
   ```python
   AUG_MOSAIC = 0.0  # Desativar mosaico
   AUG_MIXUP = 0.0  # Desativar mixup
   ```

### Lentidão no Treinamento

Se o treinamento estiver muito lento:

1. Use GPU (Google Colab) em vez de CPU
2. Feche outros programas que consomem recursos
3. Reduza o tamanho das imagens e do modelo
4. Desative técnicas de aumento de dados

## Funcionalidades Avançadas

### Treinamento com Múltiplos Datasets

O projeto suporta treinamento com múltiplos datasets do Roboflow de duas maneiras:

#### 1. Treinamento Sequencial
Treina o modelo em cada dataset sequencialmente (fine-tuning):

```bash
python main_universal.py --action train_multiple --multi-method sequential
```

#### 2. Treinamento Combinado
Combina todos os datasets e treina uma única vez:

```bash
python main_universal.py --action train_multiple --multi-method combined
```

#### Dicas para Múltiplos Datasets

- Use datasets complementares (diferentes iluminações, ângulos, fundos)
- No modo sequencial, comece com datasets genéricos e termine com específicos
- Verifique conflitos de classes entre datasets
- Considere o balanceamento entre datasets de tamanhos diferentes

### Configurações Avançadas para Hardware Específico

#### Para AMD Ryzen (CPU + GPU integrada)

Configurações recomendadas no `src/utils/config.py`:
```python
YOLO_MODEL = "yolov8n.pt"  # Modelo mais leve
IMG_SIZE = 384  # Tamanho reduzido
BATCH_SIZE = 2  # Mínimo para GPU integrada
EPOCHS = 25  # Equilibrado
PATIENCE = 5  # Early stopping
MULTI_SCALE = False  # Economiza memória
AUG_MIXUP = 0.0  # Desativado para economizar recursos
AUG_MOSAIC = 0.8  # Reduzido
```

#### Dicas para Treinar em Notebooks

1. Conecte à energia elétrica
2. Maximize a refrigeração (base refrigerada)
3. Feche programas em segundo plano
4. Monitore a temperatura (abaixo de 85°C)
5. Defina o plano de energia para "Alto Desempenho"
6. Aumente a memória virtual se possível

## Resultados e Arquivos Gerados

Após o treinamento, os resultados serão salvos em:

- **Modelo treinado**: `runs/train/exp/weights/best.pt`
- **Métricas e gráficos**: `runs/train/exp/`
- **Resultados de testes**: `runs/detect/predict/`

Para visualizar os resultados do treinamento:
```bash
# Visualizar métricas com TensorBoard
tensorboard --logdir runs/train
``` 