#Aluno: Lucas Carvalho da Luz Moura
#Matricula: 2020111815
#Codigo Feito e rodado em Linux

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools import tool
from datetime import datetime
import json
import time

class PerguntaAgente:
    def __init__(self):
        self.agente = Agent(model=Ollama(id="gemma3n:e4b"), markdown=False)

    def responder(self, pergunta):
        resposta = self.agente.run(pergunta)
        return resposta.content if hasattr(resposta, 'content') else resposta

class RespostaJSONAgente:
    def gerar_json(self, pergunta, resposta, tempo_decorrido):
        resultado = {
            "pergunta": pergunta,
            "resposta": resposta,
            "tempo_de_resposta": f"{tempo_decorrido:.2f} segundos"
        }
        return json.dumps(resultado, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    pergunta_agente = PerguntaAgente()
    json_agente = RespostaJSONAgente()

    pergunta1 = "Qual foi os primeiros jogadores do Flamengo?"
    inicio_1 = time.perf_counter()
    resp1 = pergunta_agente.responder(pergunta1)
    fim_1 = time.perf_counter()
    tempo_1 = fim_1 - inicio_1
    print("Resposta 1 em JSON:")
    print(json_agente.gerar_json(pergunta1, resp1, tempo_1))

    pergunta2 = "Qual é a capital do Tocantins?"
    inicio_2 = time.perf_counter()
    resp2 = pergunta_agente.responder(pergunta2)
    fim_2 = time.perf_counter()
    tempo_2 = fim_2 - inicio_2
    print("\nResposta 2 em JSON:")
    print(json_agente.gerar_json(pergunta2, resp2, tempo_2))

    # Diferença real do tempo de resposta
    diferenca_tempos = abs(tempo_1 - tempo_2)
    print(f"\nDiferença de tempo entre as respostas: {diferenca_tempos:.2f} segundos")