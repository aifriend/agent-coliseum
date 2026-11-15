import pytest

import condor_rag_agent
import langchain_agent
import naive_baseline

def _classes():
    found = []
    for mod in (condor_rag_agent, langchain_agent, naive_baseline):
        for obj in vars(mod).values():
            if isinstance(obj, type) and hasattr(obj, "_extract_final") and obj.__module__ == mod.__name__:
                found.append(obj)
    return found


@pytest.fixture(params=_classes(), ids=lambda c: c.__name__)
def extract(request):
    return request.param.__new__(request.param)._extract_final


def test_final_on_same_line(extract):
    assert extract("SITUATION: x\nFINAL: Lima is the capital of Peru.") == "Lima is the capital of Peru."


def test_strips_bold_and_citation_bracket(extract):
    text = "**FINAL [Shinn et al. 2023 -- Reflexion]:** Bogota"
    assert extract(text) == "Bogota"


def test_strips_answer_prefix(extract):
    assert extract("FINAL: Answer: Quito") == "Quito"


def test_final_value_on_next_line(extract):
    assert extract("FINAL:\n\n  Montevideo\nextra") == "Montevideo"


def test_falls_back_to_last_non_heading_line(extract):
    assert extract("some reasoning\nthe answer is Cusco\n# heading\n") == "the answer is Cusco"


def test_plain_text_is_stripped(extract):
    assert extract("  just text  ") == "just text"
