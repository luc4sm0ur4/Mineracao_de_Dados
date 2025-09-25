# ⛏️ Mineração de Dados: Um Portfólio Prático

![Status do Projeto](https://img.shields.io/badge/status-em%20desenvolvimento-green?style=for-the-badge)

Bem-vindo! Este repositório é uma coleção de projetos e estudos de caso em **Mineração de Dados** e **Machine Learning**, implementados principalmente em **R** e **Python**. O objetivo é aplicar técnicas e algoritmos para extrair insights valiosos e construir modelos preditivos a partir de diversos datasets.

## 🎯 Sobre o Projeto

Este projeto funciona como um portfólio e uma base de conhecimento prático em Data Mining. Cada pasta representa um desafio específico, abordando desde a análise exploratória e pré-processamento de dados até a implementação e avaliação de modelos. É um registro contínuo do meu aprendizado e desenvolvimento na área.

## 📚 Sumário
* [Tecnologias Utilizadas](#-tecnologias-utilizadas)
* [Estrutura do Repositório](#-estrutura-do-repositório)
* [Projetos em Destaque](#-projetos-em-destaque)
* [Como Executar os Projetos](#-como-executar-os-projetos)
* [Como Contribuir](#-como-contribuir)
* [Licença](#-licença)
* [Autor](#-autor)

## 🛠️ Tecnologias Utilizadas

As principais ferramentas e linguagens que você encontrará aqui são:

- **Linguagens**:
  - ![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
  - ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

- **Bibliotecas Python**:
  - **Análise de Dados**: `Pandas`, `NumPy`
  - **Machine Learning**: `Scikit-learn`, `Opacus`
  - **Visualização**: `Matplotlib`, `Seaborn`
  - **Desenvolvimento**: `Jupyter Notebook`

- **Pacotes R**:
  - **Manipulação de Dados**: `dplyr`, `readr`
  - **Visualização**: `ggplot2`
  - **Modelagem**: `caTools`, `caret`

## 📂 Estrutura do Repositório

O repositório está organizado em pastas, onde cada uma corresponde a uma atividade ou estudo de caso. Abaixo está um resumo do conteúdo de cada diretório:

| Pasta | Linguagem(ns) | Descrição |
| :--- | :--- | :--- |
| `Atividade1_-_...` | `Python` | Estudo sobre privacidade diferencial com **Variational Autoencoders (VAE)** e a biblioteca **Opacus**. |
| `Atividade2_-_...` | `R` | Foco em **Análise Exploratória de Dados (EDA)** para extrair insights iniciais de um dataset. |
| `3_Materials_...` | `R` | Início do estudo de caso sobre **detecção de transações fraudulentas**, com foco na verificação e correção de erros nos dados. |
| `Atividade4_-_...` | `R` / `Python` | Continuação do caso de detecção de fraudes, incluindo scripts para exportação de dados e criação de fluxogramas de processo. |
| `Atividade5_-_...` | `Python` | Estudo de caso completo sobre a **previsão de proliferação de algas** (Algae Bloom), implementando um fluxo de trabalho de Machine Learning. |

## ⭐ Projetos em Destaque

### 📌 Análise de Risco de Crédito com Regressão Logística

Este projeto aborda um problema clássico de classificação: **prever a probabilidade de inadimplência de um cliente**. Utilizando um dataset de risco de crédito, o objetivo é classificar os clientes como "bons" ou "maus" pagadores, auxiliando na tomada de decisão para concessão de crédito.

**Técnicas e Conceitos Abordados:**
- Análise Exploratória de Dados (EDA)
- Pré-processamento e Limpeza de Dados
- Implementação de Regressão Logística em R e Python
- Treinamento e Teste de Modelo (divisão treino/teste)
- Avaliação de performance do classificador (Matriz de Confusão, Acurácia, etc.)

## 🚀 Como Executar os Projetos

Para clonar e executar os projetos localmente, siga os passos abaixo.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/luc4sm0ur4/Mineracao_de_Dados.git](https://github.com/luc4sm0ur4/Mineracao_de_Dados.git)
    cd Mineracao_de_Dados
    ```

2.  **Para projetos em Python:**
    - É recomendado o uso de um ambiente virtual (`venv` ou `conda`).
    - Instale as dependências:
      ```bash
      pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab
      ```
    - Inicie o Jupyter:
      ```bash
      jupyter lab
      ```
    - Navegue até a pasta do projeto desejado e abra o notebook (`.ipynb`).

3.  **Para projetos em R:**
    - Abra o RStudio ou seu editor de preferência.
    - Instale os pacotes listados no início de cada script `.R`:
      ```r
      # Exemplo de pacotes a serem instalados
      install.packages(c("dplyr", "ggplot2", "caTools", "caret"))
      ```
    - Execute o script.

## 🙌 Como Contribuir

Contribuições são muito bem-vindas! Se você tem sugestões para melhorar os projetos ou quer adicionar novas análises, sinta-se à vontade para:

1.  Fazer um **Fork** do projeto.
2.  Criar uma nova **Branch** (`git checkout -b feature/NovaAnalise`).
3.  Fazer **Commit** de suas mudanças (`git commit -m 'Adiciona nova análise de clusterização'`).
4.  Fazer **Push** para a sua Branch (`git push origin feature/NovaAnalise`).
5.  Abrir um **Pull Request**.

## 📄 Licença

Este projeto está distribuído sob a licença GPL-v3. Veja o arquivo [LICENSE](https://www.gnu.org/licenses/gpl-3.0.html) para mais detalhes.

## 👤 Autor

Feito por **Lucas Moura**.

- **GitHub**: [@luc4sm0ur4](https://github.com/luc4sm0ur4)
- **LinkedIn**: [Lucas Moura](https://linkedin.com/in/lucasmoura112)