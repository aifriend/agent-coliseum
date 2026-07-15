"""
Agent Coliseum -- BCP -- Colab 03: Naive Baseline

Converted from 03_naive_baseline.ipynb (source:
github.com/omar-florez/agent-coliseum, branch bcp) into a plain module.
Importing this file has no side effects; run it directly to join the arena.
"""
import os
import re
import threading
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

from agent_base import Agent, MatchContext
from agent_server import serve_and_register

load_dotenv()

AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL", "https://YOUR-RESOURCE.openai.azure.com/openai/v1/")
MODEL = "gpt-4o-mini"
ARENA_URL = "https://agent-coliseum.onrender.com"
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")

client = OpenAI(base_url=AZURE_BASE_URL, api_key=AZURE_API_KEY)


class NaiveAgent(Agent):
    name = "JAL - Bot Basico"
    avatar = "🤖"
    description = "Un agente simple sin memoria ni recuperacion de conocimiento"

    def think(self, ctx: MatchContext) -> str:
        role_instruction = (
            f"Generate a sharp, specific question about {ctx.topic} to ask your opponent."
            if ctx.role == "asker"
            else f"Answer this question in ONE short sentence only: {ctx.current_question}"
        )
        prompt = f"""You are competing in a Latin America knowledge tournament.
Topic: {ctx.topic} | Role: {ctx.role}

Task: {role_instruction}

Reason step by step. Write concrete responses (do NOT leave blanks):

SITUATION [Wei et al. 2022 -- Chain-of-Thought]:
OPPONENT [Langner et al. 2023 -- Theory of Mind]:
GOAL [Yao et al. 2022 -- ReAct]:
DRAFT [Nye et al. 2021 -- Scratchpad]:
CRITIQUE [Madaan et al. 2023 -- Self-Refine]:
FINAL [Shinn et al. 2023 -- Reflexion]: <your {'question' if ctx.role == 'asker' else 'answer'}, ONE sentence only>"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()

    def ask(self, ctx: MatchContext) -> str:
        return self._extract_final(self.think(ctx))

    def answer(self, ctx: MatchContext) -> str:
        return self._extract_final(self.think(ctx))

    def _extract_final(self, text: str) -> str:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            clean = line.replace("**", "").strip()
            if clean.upper().startswith("FINAL"):
                after = clean[5:].lstrip(":").strip()
                after = re.sub(r"^\[.*?\]\s*:?\s*", "", after).strip()
                after = re.sub(r"^(Question|Answer)\s*:\s*", "", after, flags=re.IGNORECASE).strip()
                if after:
                    return after
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        return lines[j].strip()
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith("#"):
                return line.strip()
        return text.strip()


def _keepalive(arena_url):
    while True:
        time.sleep(60)
        try:
            requests.get(f"{arena_url}/health", timeout=5)
        except Exception:
            pass


agent = NaiveAgent()


if __name__ == "__main__":
    threading.Thread(target=_keepalive, args=(ARENA_URL,), daemon=True).start()
    serve_and_register(agent=agent, arena_url=ARENA_URL, port=5002, ngrok_token=NGROK_TOKEN)
