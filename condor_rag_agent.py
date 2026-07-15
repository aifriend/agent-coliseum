"""
Agent Coliseum -- BCP -- Colab 01: Condor RAG Agent

Converted from 01_condor_rag_agent.ipynb (source:
github.com/omar-florez/agent-coliseum, branch bcp) into a plain module.
Importing this file has no side effects; run it directly to join the arena.
"""
import json
import os
import random
import re
import threading
import time

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent_base import Agent, MatchContext, MatchResult, Position, WorldContext
from agent_server import serve_and_register

load_dotenv()

AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL", "https://rsgd15-foundry.openai.azure.com/openai/v1/")
MODEL = "gpt-4o-mini"
ARENA_URL = "https://agent-coliseum.onrender.com"
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")

client = OpenAI(base_url=AZURE_BASE_URL, api_key=AZURE_API_KEY)


def build_rag_index(facts_path="latam_facts.jsonl", max_facts=50):
    if not os.path.exists(facts_path):
        facts_path = os.path.join("..", "data", "latam_facts.jsonl")
    facts = []
    with open(facts_path) as f:
        for line in f:
            if line.strip():
                facts.append(json.loads(line))
                if len(facts) >= max_facts:
                    break
    texts = [f["text"] for f in facts]
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(texts)
    print(f"RAG index ready: {len(facts)} facts, {matrix.shape[1]} TF-IDF features")
    return vectorizer, matrix, facts


rag_vectorizer, rag_matrix, rag_facts = build_rag_index(max_facts=50)


def search_rag(query: str, top_k: int = 3) -> list:
    vec = rag_vectorizer.transform([query])
    scores = cosine_similarity(vec, rag_matrix).flatten()
    top = np.argsort(scores)[::-1][:top_k]
    return [rag_facts[i]["text"] for i in top if scores[i] > 0]


class CondorAgent(Agent):
    name = "JAL - Condor RAG"
    avatar = "🦅"
    description = "Agente con memoria y busqueda semantica de hechos latinoamericanos"

    def __init__(self):
        self._memory = {}

    def on_arena_start(self, ctx: WorldContext) -> None:
        self._memory = {}

    def on_match_start(self, ctx: MatchContext) -> None:
        if ctx.opponent_agent_id not in self._memory:
            self._memory[ctx.opponent_agent_id] = {"name": ctx.opponent_name, "wins": 0, "losses": 0}

    def on_match_end(self, ctx: MatchContext, result: MatchResult) -> None:
        opp = ctx.opponent_agent_id
        won = result.winner_id == ctx.my_agent_id
        if opp in self._memory:
            if won:
                self._memory[opp]["wins"] += 1
            else:
                self._memory[opp]["losses"] += 1

    def move(self, ctx: WorldContext) -> Position:
        active = [a for a in ctx.agents if a.status == "active" and a.agent_id != ctx.my_agent_id]
        if not active:
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            return Position(
                x=max(0, min(ctx.map_width - 1, ctx.my_position.x + dx)),
                y=max(0, min(ctx.map_height - 1, ctx.my_position.y + dy)),
            )
        target = min(active, key=lambda a: a.score)
        dx = 1 if target.position.x > ctx.my_position.x else -1 if target.position.x < ctx.my_position.x else 0
        dy = 1 if target.position.y > ctx.my_position.y else -1 if target.position.y < ctx.my_position.y else 0
        return Position(
            x=max(0, min(ctx.map_width - 1, ctx.my_position.x + dx)),
            y=max(0, min(ctx.map_height - 1, ctx.my_position.y + dy)),
        )

    def think(self, ctx: MatchContext) -> str:
        rag_hits = search_rag(ctx.topic + " " + ctx.current_question, top_k=3)
        rag_context = "\n".join(f"- {f}" for f in rag_hits) or "(no relevant facts found)"
        opp_mem = self._memory.get(ctx.opponent_agent_id, {})
        opp_summary = (
            f"Wins: {opp_mem.get('wins', 0)}, Losses: {opp_mem.get('losses', 0)}"
            if opp_mem
            else "No prior history."
        )
        history_text = ""
        for t in ctx.history[-3:]:
            history_text += f"\n  Turn {t['turn_number']}: Q={t['question'][:60]} A={t['answer'][:60]} Score={t['score']}"

        role_instruction = (
            f"Generate a sharp, specific question about {ctx.topic} that will be hard for {ctx.opponent_name} to answer."
            if ctx.role == "asker"
            else f"Answer this question in ONE short sentence only: {ctx.current_question}"
        )

        prompt = f"""You are competing in a Latin America knowledge tournament.
Topic: {ctx.topic} | Role: {ctx.role} | Turn: {ctx.turn}/{ctx.total_turns}
Your score: {sum(ctx.my_scores)} pts | Opponent: {sum(ctx.opponent_scores)} pts
Opponent: {ctx.opponent_name} | Profile: {opp_summary}
History:{history_text if history_text else " (first turn)"}
Knowledge base:
{rag_context}

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


agent = CondorAgent()


if __name__ == "__main__":
    threading.Thread(target=_keepalive, args=(ARENA_URL,), daemon=True).start()
    serve_and_register(agent=agent, arena_url=ARENA_URL, port=5000, ngrok_token=NGROK_TOKEN)
