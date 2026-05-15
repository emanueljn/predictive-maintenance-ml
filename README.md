# Predictive Maintenance using Machine Learning

## Descrição do Projeto

Este projeto foi desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC) em Engenharia de Computação, com foco na aplicação de técnicas de aprendizado de máquina supervisionado para predição de falhas em máquinas industriais.

A proposta busca demonstrar a viabilidade da utilização de modelos de classificação supervisionada no contexto da manutenção preditiva, utilizando ferramentas de código aberto e bases de dados acessíveis, alinhando-se aos princípios da Indústria 4.0 e da manutenção orientada por dados.

O sistema realiza:

* pré-processamento de dados industriais;
* treinamento de modelos de machine learning;
* validação e teste dos algoritmos;
* geração de métricas de desempenho;
* análise de predições de falha em máquinas industriais.

---

# Objetivo

Desenvolver e avaliar modelos de aprendizado de máquina capazes de prever falhas industriais a partir de dados operacionais de máquinas, utilizando algoritmos supervisionados de classificação.

---

# Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

# Estrutura do Projeto

```text
predictive-maintenance-ml/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   ├── processed/
│   │   ├── X_train.csv
│   │   ├── X_val.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_val.csv
│   │   └── y_test.csv
│
├── src/
│   ├── preprocessamento.py
│   ├── treinamento.py
│   ├── avaliacao.py
│   ├── resultados.py
│   └── teste.py
│
├── results/
│   ├── matriz_confusao.png
│   ├── metricas.csv
│
├── requirements.txt
├── README.md
```

---

# Dataset Utilizado

O projeto utiliza o dataset público AI4I 2020 Predictive Maintenance Dataset, amplamente utilizado em pesquisas relacionadas à manutenção preditiva.

O conjunto de dados contém variáveis operacionais relacionadas a:

* temperatura do ar;
* temperatura do processo;
* velocidade rotacional;
* torque;
* desgaste da ferramenta;
* ocorrência de falhas.

---

# Fluxo Metodológico

O pipeline experimental foi estruturado nas seguintes etapas:

```text
Coleta dos Dados
        ↓
Pré-processamento
        ↓
Tratamento de Dados Ausentes
        ↓
Análise de Outliers
        ↓
Normalização
        ↓
Divisão Treino / Validação / Teste
        ↓
Treinamento dos Modelos
        ↓
Validação
        ↓
Predições
        ↓
Avaliação dos Resultados
```

---

# Modelos Utilizados

Foram implementados os seguintes algoritmos supervisionados:

* Logistic Regression
* Random Forest
* Multilayer Perceptron (MLP)

---

# Métricas de Avaliação

Os modelos foram avaliados utilizando:

* Acurácia;
* Precisão;
* Recall;
* F1-score;
* Matriz de confusão.

---

# Resultados Obtidos

Os resultados experimentais demonstraram elevado desempenho dos modelos aplicados ao conjunto de testes.

| Modelo              | Acurácia | Precisão | Recall   | F1-score |
| ------------------- | -------- | -------- | -------- | -------- |
| Logistic Regression | 0.999476 | 1.0      | 0.980769 | 0.990291 |
| Random Forest       | 0.999476 | 1.0      | 0.980769 | 0.990291 |
| MLP                 | 0.999476 | 1.0      | 0.980769 | 0.990291 |

A matriz de confusão obtida para o modelo Random Forest apresentou:

* elevada capacidade de generalização;
* ausência de falsos positivos;
* apenas um falso negativo no conjunto de testes.

---

# Como Executar o Projeto

## 1. Clonar o Repositório

```bash
git clone https://github.com/emanueljn/predictive-maintenance-ml.git
```

---

## 2. Acessar o Diretório

```bash
cd predictive-maintenance-ml
```

---

## 3. Criar Ambiente Virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

# Execução dos Scripts

## Pré-processamento

```bash
python src/preprocessamento.py
```

Responsável por:

* limpeza dos dados;
* tratamento de valores ausentes;
* normalização;
* divisão treino/validação/teste.

---

## Treinamento dos Modelos

```bash
python src/treinamento.py
```

Responsável por:

* treinamento dos algoritmos;
* ajuste de hiperparâmetros;
* geração das predições.

---

## Avaliação dos Modelos

```bash
python src/avaliacao.py
```

Responsável por:

* cálculo das métricas;
* matriz de confusão;
* avaliação final.

---

## Testes Práticos

```bash
python src/teste.py
```

Responsável por:

* simulação de novas máquinas;
* geração de predições;
* classificação de falha ou não falha.

---

# Possíveis Melhorias Futuras

* utilização de dados industriais reais;
* integração com sensores IoT;
* aprendizado profundo (Deep Learning);
* monitoramento em tempo real;
* dashboards analíticos;
* implantação em ambientes industriais reais.

---

# Autor

- Alex de Almeida Cruz;
- Emanuel de Jesus Nardes;
- José Edinaldo da Silva Junior;
- Leandro de Jesus Vitorino dos Santos;
- Robson dos Santos;
- Sarah Aparecida Valeriano;
- Rodrigues Silva;
- Victor Luis Gama Rodrigues;
- Willian Claudio Ferreira.


Trabalho de Conclusão de Curso – Engenharia de Computação pela UNIVESP – Universidade Virtual do Estado de São Paulo

---

# Licença

Este projeto possui finalidade acadêmica e educacional.
