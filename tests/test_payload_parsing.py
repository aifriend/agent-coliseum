import pytest

from agent_base import AgentInfo, MatchContext, Position, WorldContext
from agent_server import _parse_match_ctx, _parse_world_ctx


def _match_payload(**extra):
    data = {
        "match_id": "m1", "topic": "history", "turn": 2, "total_turns": 6,
        "role": "asker", "history": [{"turn_number": 1}], "my_agent_id": "a",
        "opponent_agent_id": "b", "opponent_name": "Bob",
        "my_scores": [3], "opponent_scores": [5],
    }
    data.update(extra)
    return data


def test_match_ctx_maps_fields_and_defaults():
    ctx = _parse_match_ctx(_match_payload())
    assert isinstance(ctx, MatchContext)
    assert (ctx.match_id, ctx.turn, ctx.role) == ("m1", 2, "asker")
    assert ctx.opponent_scores == [5]
    assert ctx.scratchpad == "" and ctx.current_question == ""


def test_match_ctx_optional_fields():
    ctx = _parse_match_ctx(_match_payload(scratchpad="n", current_question="Q?"))
    assert ctx.scratchpad == "n" and ctx.current_question == "Q?"


def test_match_ctx_missing_required_key():
    data = _match_payload()
    del data["topic"]
    with pytest.raises(KeyError):
        _parse_match_ctx(data)


def test_world_ctx_with_agents():
    data = {
        "phase": "running", "my_agent_id": "a", "my_position": {"x": 1, "y": 2},
        "my_score": 7, "map_width": 20, "map_height": 15,
        "agents": [{"agent_id": "b", "name": "B", "avatar": "x", "score": 4,
                    "position": {"x": 3, "y": 4}, "status": "active"}],
    }
    ctx = _parse_world_ctx(data)
    assert isinstance(ctx, WorldContext)
    assert ctx.my_position == Position(1, 2)
    assert isinstance(ctx.agents[0], AgentInfo)
    assert ctx.agents[0].position == Position(3, 4)


def test_world_ctx_without_agents_key():
    data = {"phase": "lobby", "my_agent_id": "a", "my_position": {"x": 0, "y": 0},
            "my_score": 0, "map_width": 5, "map_height": 5}
    assert _parse_world_ctx(data).agents == []
