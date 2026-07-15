"""
Agent Coliseum — showcase app.

A small local web UI to watch the three converted agents (Condor RAG,
LangChain Puma, Naive Baseline) duel on a topic side by side: same
question, same knowledge, three different reasoning styles.

This talks to the agents directly in-process (agent.think() / agent.ask() /
agent.answer()) — it does NOT touch serve_and_register(), so no ngrok
tunnel opens and nothing registers with the live arena.

Run with:  uv run python showcase_app.py
"""
from flask import Flask, jsonify, render_template, request

from agent_base import MatchContext
import condor_rag_agent
import langchain_agent
import naive_baseline

app = Flask(__name__)

FIGHTERS = [
    {"id": "condor", "module": condor_rag_agent, "color": "#6fa8d1", "tag": "RAG + memory"},
    {"id": "puma", "module": langchain_agent, "color": "#d2691e", "tag": "memory + reasoning"},
    {"id": "bot", "module": naive_baseline, "color": "#8a8f95", "tag": "no memory, no retrieval"},
]

for _f in FIGHTERS:
    _agent = _f["module"].agent
    _f["name"] = _agent.name
    _f["avatar"] = _agent.avatar
    _f["description"] = _agent.description


def _duel_one(fighter: dict, topic: str) -> dict:
    agent = fighter["module"].agent
    try:
        asker_ctx = MatchContext(
            match_id="showcase", topic=topic, turn=1, total_turns=1, role="asker",
            history=[], my_agent_id="me", opponent_agent_id="opp",
            opponent_name="Rival", my_scores=[], opponent_scores=[],
        )
        question_trace = agent.think(asker_ctx)
        question = agent._extract_final(question_trace, "asker")

        answerer_ctx = MatchContext(
            match_id="showcase", topic=topic, turn=1, total_turns=1, role="answerer",
            history=[], my_agent_id="me", opponent_agent_id="opp",
            opponent_name="Rival", my_scores=[], opponent_scores=[],
            current_question=question,
        )
        answer_trace = agent.think(answerer_ctx)
        answer = agent._extract_final(answer_trace, "answerer")

        return {
            "id": fighter["id"],
            "color": fighter["color"],
            "name": agent.name,
            "avatar": agent.avatar,
            "description": agent.description,
            "question": question,
            "question_trace": question_trace,
            "answer": answer,
            "answer_trace": answer_trace,
            "error": None,
        }
    except Exception as e:
        return {
            "id": fighter["id"],
            "color": fighter["color"],
            "name": agent.name,
            "avatar": agent.avatar,
            "description": agent.description,
            "question": None,
            "question_trace": None,
            "answer": None,
            "answer_trace": None,
            "error": str(e),
        }


@app.route("/")
def index():
    return render_template("index.html", fighters=FIGHTERS)


@app.route("/api/duel/<fighter_id>", methods=["POST"])
def duel_one(fighter_id):
    topic = (request.json or {}).get("topic", "").strip()
    if not topic:
        return jsonify({"error": "topic is required"}), 400
    fighter = next((f for f in FIGHTERS if f["id"] == fighter_id), None)
    if fighter is None:
        return jsonify({"error": f"unknown fighter '{fighter_id}'"}), 404

    return jsonify(_duel_one(fighter, topic))


if __name__ == "__main__":
    print("\nAgent Coliseum showcase — http://127.0.0.1:8000\n")
    app.run(port=8000, debug=False, threaded=True)
