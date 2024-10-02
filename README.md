# Tópicos Avançados em Aprendizado de Máquina e Otimização - Trabalho 03
## Classificação de Atividades Utilizando Sinais de Acelerômetros com LSTM

Este projeto tem como objetivo identificar atividades físicas a partir de sinais de aceleração (eixos x, y e z) coletados de um dispositivo móvel Samsung Galaxy Nexus I9250. Os dados foram obtidos da base **UniMibSHAR** e utilizados para treinar uma rede neural recorrente do tipo **LSTM** (Long Short-Term Memory) para classificar as seguintes atividades:

### Atividades Classificadas:
1. Levantar-se de uma cadeira – `StandingUpFS`
2. Levantar-se da cama – `StandingUpFL`
3. Caminhar – `Walking`
4. Correr – `Running`
5. Descer escadas – `GoingDownS`
6. Pular – `Jumping`
7. Subir escadas – `GoingUpS`
8. Deitar-se na cama – `LyingDownFS`
9. Sentar-se numa cadeira – `SittingDown`

### Estrutura dos Dados:
O projeto utiliza dois arquivos principais:

1. **adl_data.mat**: Contém 7579 amostras de 453 valores cada, correspondentes aos eixos x, y e z dos acelerômetros:
   - Os primeiros 151 valores são do eixo **x**.
   - Os próximos 151 valores são do eixo **y**.
   - Os últimos 151 valores são do eixo **z**.
   
2. **adl_labels.mat**: Contém 7579 rótulos numéricos, de 1 a 9, correspondendo às atividades listadas acima. Cada linha de `adl_labels.mat` corresponde diretamente a uma linha de `adl_data.mat`.

2. **adl_names.mat**: Contém os nomes dos rótulos, correspondendo às atividades listadas acima. Cada linha de `adl_names.mat` corresponde diretamente a uma atividade na ordem listada.

### Objetivo do Projeto:
- Treinar e avaliar uma rede neural LSTM para a classificação das atividades com base nos dados dos acelerômetros.
- Dividir os dados em dois conjuntos: 
  - **Treinamento** (85%)
  - **Validação** (15%)

---

## Como Executar o Projeto

### 1. Requisitos
1. Clone o repositório:

    ```bash
    git clone https://github.com/hebertbruno/PGENE-613-Trabalho-03.git
    cd PGENE-613-Trabalho-03
   
2. Crie um ambiente virtual (opcional, mas recomendado):

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows `venv\Scripts\activate`
    ```

3. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```


### 2. Funções Principais

O projeto está organizado em diferentes módulos, com as seguintes funções:

- **``load_data.py``**: Carrega e processa os dados dos arquivos `adl_data.mat`, `adl_labels.mat` e `adl_names.mat`.
- **`model.py`**: Cria e configura a arquitetura do modelo LSTM.
- **`train.py`**: Treina o modelo utilizando o conjunto de treinamento.
- **`test.py`**: Avalia o modelo treinado no conjunto de validação.
- **`plot.py`**: Plota gráficos de perda e acurácia durante o treinamento e gera a matriz de confusão para o desempenho do modelo.
- **`save_predictions.py`**: Salva as previsões do modelo em um arquivo `.csv`.

### 3. Executar o Projeto

Para treinar e avaliar o modelo, basta rodar o script principal `main.py`:

```bash
python main.py
```
### 4. Resultados

Os resultados são armazenados na pasta `/results` contendo:
- Graficos de acuracia e perda do conjunto de validacao ao longo das epocas.
- CSV com os valores reais e preditos pelo modelo para comparação.

### Referencias:

[1] Dataset UniMibSHAR: https://www.mdpi.com/2076-3417/7/10/1101

## Autores

- **Alvaro Oliveira**
- **Bruno Hebert**