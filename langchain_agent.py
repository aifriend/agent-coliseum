"""
Agent Coliseum -- BCP -- Colab 02: LangChain-style Agent

Converted from 02_langchain_agent.ipynb (source:
github.com/omar-florez/agent-coliseum, branch bcp) into a plain module.
Importing this file has no side effects; run it directly to join the arena.
"""
import os
import random
import re
import threading
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

from agent_base import Agent, MatchContext, MatchResult, Position, WorldContext
from agent_server import serve_and_register

load_dotenv()

AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL", "https://rsgd15-foundry.openai.azure.com/openai/v1/")
MODEL = "gpt-4o-mini"
ARENA_URL = "https://agent-coliseum.onrender.com"
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")

client = OpenAI(base_url=AZURE_BASE_URL, api_key=AZURE_API_KEY)

SYSTEM_PROMPT = """You are a competitive Latin America knowledge agent.
Reason step by step. Write concrete responses under each label (do NOT leave blanks):

SITUATION [Wei et al. 2022 -- Chain-of-Thought]:
OPPONENT [Langner et al. 2023 -- Theory of Mind]:
GOAL [Yao et al. 2022 -- ReAct]:
DRAFT [Nye et al. 2021 -- Scratchpad]:
CRITIQUE [Madaan et al. 2023 -- Self-Refine]:
FINAL [Shinn et al. 2023 -- Reflexion]: <your question or answer, concise>"""


def think_chain_invoke(input_text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ],
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


class LangChainLatAmAgent(Agent):
    name = "JAL - LangChain Puma"
    avatar = "🐆"
    description = "Agente con memoria de conversacion y razonamiento estructurado"

    def __init__(self):
        self._match_memory = {}
        self._opponent_notes = {}

    def on_arena_start(self, ctx: WorldContext) -> None:
        self._match_memory = {}
        self._opponent_notes = {}

    def on_match_start(self, ctx: MatchContext) -> None:
        self._match_memory[ctx.match_id] = []

    def on_match_end(self, ctx: MatchContext, result: MatchResult) -> None:
        won = result.winner_id == ctx.my_agent_id
        self._opponent_notes[ctx.opponent_agent_id] = {
            "name": ctx.opponent_name,
            "result": "won" if won else "lost",
            "topic": ctx.topic,
        }

    def move(self, ctx: WorldContext) -> Position:
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])
        return Position(
            x=max(0, min(ctx.map_width - 1, ctx.my_position.x + dx)),
            y=max(0, min(ctx.map_height - 1, ctx.my_position.y + dy)),
        )

    def think(self, ctx: MatchContext) -> str:
        history = ""
        for t in ctx.history[-3:]:
            history += f"\n  Turn {t['turn_number']}: Q={t['question'][:50]} A={t['answer'][:50]} Score={t['score']}"
        opp = self._opponent_notes.get(ctx.opponent_agent_id, {})
        opp_txt = (
            f"Previously: {opp.get('result', 'unknown')} on {opp.get('topic', '?')}"
            if opp
            else "First encounter."
        )
        mem = self._match_memory.get(ctx.match_id, [])
        mem_txt = "".join(f"\n  Turn {e['turn']} ({e['role']}): {e['summary']}" for e in mem[-2:])

        role_instruction = (
            f"Generate a sharp, specific question about {ctx.topic} that will be hard for {ctx.opponent_name} to answer."
            if ctx.role == "asker"
            else f"Answer this question in ONE short sentence only: {ctx.current_question}"
        )

        input_text = f"""Topic: {ctx.topic} | Role: {ctx.role} | Turn {ctx.turn}/{ctx.total_turns}
My score: {sum(ctx.my_scores)} pts | Opponent: {sum(ctx.opponent_scores)} pts | Opponent: {ctx.opponent_name}
Opponent history: {opp_txt}
Match history:{history if history else " (first turn)"}
Memory:{mem_txt if mem_txt else " (empty)"}

Task: {role_instruction}"""

        result = think_chain_invoke(input_text)
        mem.append({"turn": ctx.turn, "role": ctx.role, "summary": result[:150]})
        self._match_memory[ctx.match_id] = mem
        return result

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


agent = LangChainLatAmAgent()


if __name__ == "__main__":
    threading.Thread(target=_keepalive, args=(ARENA_URL,), daemon=True).start()
    serve_and_register(agent=agent, arena_url=ARENA_URL, port=5001, ngrok_token=NGROK_TOKEN)
