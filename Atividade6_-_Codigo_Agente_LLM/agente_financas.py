# Aluno: Lucas Carvalho da Luz Moura
# Matricula: 2020111815
# Codigo Feito e rodado em Linux

import os
import pandas as pd
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from io import StringIO

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Variável de ambiente GOOGLE_API_KEY não encontrada.")

    agente = Agent(
        model=Gemini(id="models/gemini-2.5-pro", api_key=api_key),
        tools=[YFinanceTools()],
        markdown=False
    )

    pergunta = (
        "Forneça os preços de fechamento diários da AAPL (Apple) nos últimos 10 dias de negociação. "
        "Retorne como uma tabela com as colunas 'Date' e 'Close'."
    )

    resposta = agente.run(pergunta)
    conteudo = resposta.content if hasattr(resposta, 'content') else resposta

    print("Resposta recebida do agente:")
    print(conteudo)

    try:
        # Usa StringIO para passar texto HTML para read_html
        tabelas = pd.read_html(StringIO(conteudo))
        tabela = tabelas[0]

        # Filtra os últimos 10 pregões
        ultimos_10 = tabela.tail(10)

        # Salva em arquivo txt com formatação legível
        arquivo_nome = "cotacoes_aapl.txt"
        with open(arquivo_nome, "w", encoding="utf-8") as f:
            f.write("Cotações diárias da Apple (AAPL) - Últimos 10 pregões\n")
            f.write("=" * 50 + "\n\n")
            f.write(ultimos_10.to_string(index=False))
            f.write("\n\nDados fornecidos pelo YFinanceTools via Agente Gemini.\n")

        print(f"Cotações dos últimos 10 dias úteis salvas em '{os.path.abspath(arquivo_nome)}'.")

    except ValueError:
        print("Não foi possível extrair tabela da resposta do agente.")
        arquivo_nome = "cotacoes_aapl.txt"
        with open(arquivo_nome, "w", encoding="utf-8") as f:
            f.write("Resposta recebida (não em tabela):\n\n")
            f.write(conteudo)

        print(f"Resposta completa salva em '{os.path.abspath(arquivo_nome)}' para análise manual.")

    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()
