#Aluno: Lucas Carvalho da Luz Moura
#Matricula: 2020111815

import os
import asyncio
import pandas as pd
import re
from io import StringIO
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.google import Gemini

load_dotenv()

def markdown_to_csv(markdown_text: str, csv_filename="notebooks_kabum.csv"):
    try:
        linhas = []
        for line in markdown_text.splitlines():
            if re.match(r"^\s*\|.*\|\s*$", line):
                linhas.append(line.strip())

        if len(linhas) < 2:
            print("Tabela Markdown inválida ou não encontrada.")
            return
        
        linhas_sem_separador = [l for l in linhas if not re.match(r"^\s*\|?[-:\s|]+\|?[-:\s|]*\|?\s*$", l)]
        texto_csv = "\n".join([linha.strip("|").replace("|", ";") for linha in linhas_sem_separador])

        df = pd.read_csv(StringIO(texto_csv), sep=";")
        df.to_csv(csv_filename, sep=";", index=False)
        print(f"CSV salvo com sucesso em: {csv_filename}")

    except Exception as e:
        print(f"Erro ao converter markdown para CSV: {e}")

async def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Variável de ambiente GOOGLE_API_KEY não encontrada.")

    agente = Agent(
        model=Gemini(id="models/gemini-2.5-pro", api_key=api_key),
        markdown=True,
    )

    pergunta = (
        "Liste o nome e preço dos notebooks encontrados na primeira página do site "
        "https://www.kabum.com.br/busca/notebook em formato de tabela markdown. "
        "Não dê explicações, apenas retorne a tabela nome x preço."
    )

    resposta = await agente.arun(pergunta)

    conteudo = resposta.content if hasattr(resposta, "content") else resposta
    print("Resposta do Gemini via API:")
    print(conteudo)

    markdown_to_csv(conteudo)

if __name__ == "__main__":
    asyncio.run(main())
