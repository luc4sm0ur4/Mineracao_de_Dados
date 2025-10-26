# Aluno: Lucas Carvalho da Luz Moura
# Matricula: 2020111815
# Codigo Feito e rodado em Linux

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools import tool
from datetime import datetime
import json
import time

class PerguntaAgente:
    def __init__(self):
        self.agente = Agent(model=Ollama(id="gemma3n:e4b"), markdown=False)

    def responde(self, pergunta):
        inicio_total = time.perf_counter() #Contagem inicial Local
        inicio_llm = time.perf_counter() #Contagem inicial do LLM

        resposta = self.agente.run(pergunta)

        fim_llm = time.perf_counter() #Contagem final LLM
        fim_total = time.perf_counter() #Contagem final Local

        tempo_exato_llm = fim_llm - inicio_llm #Resultado LLM
        tempo_total = fim_total - inicio_total #Resultado Local

        return {
            "pergunta": pergunta,
            "resposta": resposta.content if hasattr(resposta, 'content') else resposta,
            "tempo_exato_llm": tempo_exato_llm,
            "tempo_de_resposta": tempo_total
        }

class RespostaJSONAgente:
    def gerar_json(self, resultado):
        resultado_formatado = resultado.copy()
        resultado_formatado["tempo_exato_llm"] = f"{resultado['tempo_exato_llm']:.2f} segundos" #string com 2 casas decimais
        resultado_formatado["tempo_de_resposta"] = f"{resultado['tempo_de_resposta']:.2f} segundos" #string com 2 casas decimais
        return json.dumps(resultado_formatado, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    pergunta_agente = PerguntaAgente()
    json_agente = RespostaJSONAgente()

    resultado1 = pergunta_agente.responde("Qual foi os primeiros jogadores do Flamengo?")
    print("Resposta 1 em JSON:")
    print(json_agente.gerar_json(resultado1))

    resultado2 = pergunta_agente.responde("Qual é a capital do Tocantins?")
    print("\nResposta 2 em JSON:")
    print(json_agente.gerar_json(resultado2))

    # Calcula a diferença de tempo entre operação total e tempo exato da LLM para cada resposta
    diferenca_1 = resultado1["tempo_de_resposta"] - resultado1["tempo_exato_llm"]
    diferenca_2 = resultado2["tempo_de_resposta"] - resultado2["tempo_exato_llm"]

    print(f"\nDiferença entre tempo de operação e tempo exato da LLM para a Resposta 1: {diferenca_1:.2f} segundos")
    print(f"Diferença entre tempo de operação e tempo exato da LLM para a Resposta 2: {diferenca_2:.2f} segundos")
    print("-------------------------------------------------------------------------------------------------------")