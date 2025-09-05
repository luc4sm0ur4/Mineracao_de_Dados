# ⛏️ Mineração de Dados

Bem-vindo! Este repositório é dedicado à aplicação prática de técnicas de Mineração de Dados. Aqui você encontrará códigos e análises desenvolvidos com o objetivo de explorar datasets, construir modelos preditivos e extrair insights valiosos. O foco principal é a utilização das linguagens **R** e **Python**.

## 🎯 Objetivo

O propósito deste projeto é servir como um portfólio e uma base de estudos em Data Mining, demonstrando a implementação de algoritmos de Machine Learning para resolver problemas reais. Cada pasta representa um projeto ou um estudo de caso, abordando desde o pré-processamento dos dados até a avaliação do modelo final.

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte maneira para facilitar a navegação:

```
/Mineracao_de_Dados
|
├── 📁 datasets/
│   └── 📄 credit_risk.csv
│
├── 📁 Python/
│   └── 📓 Regressão Logística - Risco de Crédito.ipynb
│
├── 📁 R/
│   └── 📜 Regressão Logística com dados de Risco de Crédito.R
│
└── README.md
```

- **/datasets**: Contém os conjuntos de dados utilizados nos projetos.
- **/Python**: Scripts e notebooks desenvolvidos em Python.
- **/R**: Scripts e projetos desenvolvidos em R.

## ⭐ Projetos em Destaque

### Análise de Risco de Crédito com Regressão Logística

Este projeto aborda um problema clássico de classificação: **prever a probabilidade de inadimplência de um cliente**. Utilizando o dataset `credit_risk.csv`, foram desenvolvidos dois modelos de Regressão Logística, um em R e outro em Python.

O objetivo é classificar os clientes como "bons" ou "maus" pagadores com base em suas características (histórico de crédito, propósito do empréstimo, idade, etc.), ajudando na tomada de decisão para concessão de crédito.

**Técnicas e Conceitos Abordados:**
- Análise Exploratória de Dados (EDA)
- Pré-processamento e Limpeza de Dados
- Implementação de Regressão Logística
- Treinamento e Teste de Modelo
- Avaliação de performance do classificador

## 🛠️ Ferramentas e Bibliotecas

As principais tecnologias utilizadas neste repositório são:

- **Linguagens**:
  - ![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
  - ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

- **Bibliotecas Python**:
  - **Pandas** e **Numpy**: Para manipulação e análise de dados.
  - **Scikit-learn**: Para a construção do modelo de Machine Learning.
  - **Matplotlib** e **Seaborn**: Para visualização de dados.
  - **Jupyter Notebook**: Para o desenvolvimento interativo.

- **Pacotes R**:
  - **dplyr** e **readr**: Para manipulação de dados.
  - **ggplot2**: Para visualizações gráficas.
  - **caTools**: Para divisão dos dados em treino e teste.
  - **caret**: Para avaliação de modelos.

## 🚀 Como Executar os Projetos

Para rodar os projetos em sua máquina local, siga os passos:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/luc4sm0ur4/Mineracao_de_Dados.git](https://github.com/luc4sm0ur4/Mineracao_de_Dados.git)
    cd Mineracao_de_Dados
    ```

2.  **Para o projeto em Python:**
    - É recomendado criar um ambiente virtual.
    - Instale as bibliotecas necessárias:
      ```bash
      pip install pandas numpy scikit-learn matplotlib seaborn jupyter
      ```
    - Abra o Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Navegue até `Python/` e abra o arquivo `.ipynb`.

3.  **Para o projeto em R:**
    - Abra o RStudio.
    - Instale os pacotes necessários que estão no início do script:
      ```r
      install.packages(c("dplyr", "ggplot2", "caTools", "caret"))
      ```
    - Execute o script `.R` localizado na pasta `R/`.

## 🤝 Como Contribuir

Contribuições são sempre bem-vindas! Se você tiver sugestões para melhorar os projetos existentes ou quiser adicionar novos algoritmos e análises, sinta-se à vontade para:

1.  Fazer um **Fork** do projeto.
2.  Criar uma nova **Branch** (`git checkout -b feature/NovaAnalise`).
3.  Fazer **Commit** de suas mudanças (`git commit -m 'Adiciona nova análise de clusterização'`).
4.  Fazer **Push** para a Branch (`git push origin feature/NovaAnalise`).
5.  Abrir um **Pull Request**.

## 👤 Autor

- **Lucas Moura** - [luc4sm0ur4](https://github.com/luc4sm0ur4)