import os
import nest_asyncio
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from agno.agent import Agent
from agno.models.google import Gemini

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Histórico simples da conversa
history = []

# Instancia o agente Gemini
agent = Agent(
    model=Gemini(id="models/gemini-2.5-pro", api_key=GOOGLE_API_KEY),
    markdown=True
)

async def processar_mensagem(texto_usuario: str) -> str:
    history.append(f"Usuário: {texto_usuario}")
    prompt = "\n".join(history)
    resposta = await agent.arun(prompt)
    texto_resposta = resposta.content if hasattr(resposta, "content") else resposta
    history.append(f"Agente: {texto_resposta}")
    return texto_resposta

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    texto_usuario = update.message.text
    if not texto_usuario:
        return
    resposta = await processar_mensagem(texto_usuario)
    await update.message.reply_text(resposta)

async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot rodando...")
    await app.run_polling()

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())