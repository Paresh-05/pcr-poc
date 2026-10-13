"""Graph behaviour with a scripted fake LLM (no database, no network)."""
import asyncio

from app.diagnosis_graph import (
    MAX_CANDIDATES, MAX_QUESTIONS, MIN_QUESTIONS, Candidate, Conclusion, StepOut, build_graph, normalize_candidates, should_conclude,
)


def _step(cands: list[tuple[str, int]], question: str, ready: bool = False) -> StepOut:
    return StepOut(candidates=[Candidate(cause=c, likelihood=n, reasoning="r") for c, n in cands], next_question=question, ready_to_conclude=ready)


CONCLUSION = Conclusion(
    problem="A", confidence="high", evidence=["e"], fix_steps=["s1", "s2"], parts=["p"], verify_fix=["v"],
    safety_note="n", alternatives=[],
)


class ScriptedLLM:
    def __init__(self, steps: list[StepOut]):
        self.steps = list(steps)
        self.calls: list[tuple[str, str]] = []

    async def summarize(self, system, user, purpose="summarize"):
        return "summary"

    async def structured(self, system, user, schema, purpose):
        self.calls.append((purpose, user))
        return CONCLUSION if schema is Conclusion else self.steps.pop(0)


def _start(llm):
    g = build_graph(llm)
    return g, asyncio.run(g.ainvoke({"mode": "start", "case_file": "CF", "candidates": [], "history": []}))


def _answer(g, state, answer="yes", note=""):
    return asyncio.run(g.ainvoke({
        "mode": "answer", "case_file": "CF", "candidates": state["candidates"], "history": state["history"],
        "pending_question": state["pending_question"], "answer": answer, "note": note,
    }))


def test_candidates_are_normalized_to_100_and_sorted():
    out = normalize_candidates([Candidate(cause="b", likelihood=1, reasoning=""), Candidate(cause="a", likelihood=3, reasoning="")])
    assert [c["cause"] for c in out] == ["a", "b"] and sum(c["likelihood"] for c in out) == 100
    zeros = normalize_candidates([Candidate(cause=x, likelihood=0, reasoning="") for x in "abc"])
    assert sum(c["likelihood"] for c in zeros) == 100
    assert len(normalize_candidates([Candidate(cause=str(i), likelihood=10, reasoning="") for i in range(9)])) == MAX_CANDIDATES


def test_conclusion_rules():
    lead = [{"likelihood": 80}, {"likelihood": 10}]
    assert not should_conclude(lead, MIN_QUESTIONS - 1, True)  # never before the minimum number of questions
    assert should_conclude(lead, MIN_QUESTIONS, False)
    assert not should_conclude([{"likelihood": 50}, {"likelihood": 40}], MIN_QUESTIONS, False)
    assert should_conclude([{"likelihood": 65}, {"likelihood": 35}], MIN_QUESTIONS, True)  # model says ready and top is solid
    assert not should_conclude([{"likelihood": 55}, {"likelihood": 45}], MIN_QUESTIONS, True)
    assert should_conclude([{"likelihood": 40}, {"likelihood": 35}], MAX_QUESTIONS, False)
    assert should_conclude([], 0, False)


def test_start_asks_first_question_and_waits():
    llm = ScriptedLLM([_step([("A", 45), ("B", 35), ("C", 20)], "Is the connector damaged?")])
    _, state = _start(llm)
    assert state["status"] == "asking" and state["pending_question"] == "Is the connector damaged?"
    assert [c["cause"] for c in state["candidates"]] == ["A", "B", "C"]
    assert [p for p, _ in llm.calls] == ["propose"]


def test_answers_narrow_down_and_finish_with_a_structured_conclusion():
    llm = ScriptedLLM([
        _step([("A", 45), ("B", 35), ("C", 20)], "Q1?"),
        _step([("A", 55), ("B", 30), ("C", 15)], "Q2?"),
        _step([("A", 70), ("B", 20), ("C", 10)], "Q3?"),
        _step([("A", 90), ("B", 6), ("C", 4)], "Q4?"),
    ])
    g, state = _start(llm)
    state = _answer(g, state, "yes")
    assert state["status"] == "asking" and state["pending_question"] == "Q2?" and len(state["history"]) == 1
    state = _answer(g, state, "no", "looked fine")
    assert state["status"] == "asking" and len(state["history"]) == 2
    state = _answer(g, state, "yes")  # third answer: the leader is now far ahead, so it concludes
    assert state["status"] == "concluded" and state["pending_question"] is None
    assert state["conclusion"]["fix_steps"] == ["s1", "s2"]
    assert [h["answer"] for h in state["history"]] == ["yes", "no", "yes"] and state["history"][1]["note"] == "looked fine"
    assert [p for p, _ in llm.calls] == ["propose", "step", "step", "step", "conclude"]
    assert "Q1?" in llm.calls[1][1] and "not_sure" not in llm.calls[1][1]


def test_repeated_question_forces_a_conclusion_instead_of_looping():
    llm = ScriptedLLM([_step([("A", 50), ("B", 50)], "Q1?"), _step([("A", 52), ("B", 48)], "q1?")])
    g, state = _start(llm)
    state = _answer(g, state, "not_sure")
    assert state["status"] == "concluded" and state["conclusion"] is not None


def test_stops_after_the_maximum_number_of_questions():
    steps = [_step([("A", 40), ("B", 35), ("C", 25)], f"Q{i}?") for i in range(MAX_QUESTIONS + 1)]
    g, state = _start(ScriptedLLM(steps))
    for _ in range(MAX_QUESTIONS):
        assert state["status"] == "asking"
        state = _answer(g, state, "yes")
    assert state["status"] == "concluded" and len(state["history"]) == MAX_QUESTIONS
