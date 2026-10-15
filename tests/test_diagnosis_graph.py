"""Graph behaviour with a scripted fake LLM (no database, no network)."""
import asyncio

from app.diagnosis_graph import (
    MAX_CANDIDATES, MAX_QUESTIONS, MIN_INFORMATIVE, Candidate, Conclusion, StepOut, build_graph, cap_confidence,
    informative_answers, normalize_candidates, opening, pick_question, should_conclude,
)

PACK = {
    "causes": [
        {"cause": "A", "origin": "both", "share_pct": 60, "cases_counted": 60},
        {"cause": "B", "origin": "past_repairs", "share_pct": 20, "cases_counted": 20},
        {"cause": "C", "origin": "manual", "share_pct": None, "cases_counted": None},
    ],
    "question_bank": [
        {"question": "Is the connector damaged?", "source": "manual", "yes_points_to": ["C"], "no_points_to": []},
        {"question": "Is the 12V battery below 11.5 V?", "source": "past_repairs", "yes_points_to": ["A"], "no_points_to": ["B"]},
    ],
}


def _step(cands: list[tuple[str, int]], question: str, ready: bool = False, showed: str = "it moved") -> StepOut:
    return StepOut(candidates=[Candidate(cause=c, likelihood=n, reasoning="r") for c, n in cands],
                   what_it_showed=showed, next_question=question, ready_to_conclude=ready)


CONCLUSION = Conclusion(
    problem="A", confidence="high", evidence=["e"], how_we_got_here=["h"], ruled_out=[], uncertain=["u"],
    fix_steps=["s1", "s2"], parts=["p"], verify_fix=["v"], safety_note="n", alternatives=[],
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


def _start(llm, pack=PACK):
    g = build_graph(llm)
    return g, asyncio.run(g.ainvoke({"mode": "start", "case_file": "CF", "pack": pack, "candidates": [], "history": []}))


def _answer(g, state, answer="yes"):
    return asyncio.run(g.ainvoke({
        "mode": "answer", "case_file": "CF", "candidates": state["candidates"], "history": state["history"],
        "start_candidates": state.get("start_candidates", []), "forced": state.get("forced", False),
        "pending_question": state["pending_question"], "answer": answer,
    }))


def test_candidates_are_normalized_to_100_and_sorted():
    out = normalize_candidates([Candidate(cause="b", likelihood=1, reasoning=""), Candidate(cause="a", likelihood=3, reasoning="")])
    assert [c["cause"] for c in out] == ["a", "b"] and sum(c["likelihood"] for c in out) == 100
    zeros = normalize_candidates([Candidate(cause=x, likelihood=0, reasoning="") for x in "abc"])
    assert sum(c["likelihood"] for c in zeros) == 100
    assert len(normalize_candidates([Candidate(cause=str(i), likelihood=10, reasoning="") for i in range(9)])) == MAX_CANDIDATES


def test_conclusion_rules_count_only_yes_no_answers():
    lead = [{"likelihood": 80}, {"likelihood": 10}]
    assert not should_conclude(lead, 2, True, MIN_INFORMATIVE - 1)  # "not sure" answers do not count toward the minimum
    assert should_conclude(lead, 3, False, MIN_INFORMATIVE)
    assert not should_conclude([{"likelihood": 50}, {"likelihood": 40}], 3, False, 3)
    assert should_conclude([{"likelihood": 65}, {"likelihood": 35}], 3, True, 3)  # model says ready and top is solid
    assert not should_conclude([{"likelihood": 55}, {"likelihood": 45}], 3, True, 3)
    assert should_conclude([{"likelihood": 40}, {"likelihood": 35}], MAX_QUESTIONS, False, 1)
    assert should_conclude([], 0, False, 0)
    assert informative_answers([{"answer": "yes"}, {"answer": "not_sure"}, {"answer": "no"}]) == 2


def test_confidence_is_capped_by_the_evidence_not_the_model():
    cands = [{"likelihood": 48}, {"likelihood": 35}]
    assert cap_confidence("high", cands, 4, False)[0] == "low"  # leader under 60%
    assert cap_confidence("high", [{"likelihood": 80}], 2, False)[0] == "low"  # too few informative answers
    assert cap_confidence("high", [{"likelihood": 80}], 4, True)[0] == "low"  # forced stop
    assert cap_confidence("high", [{"likelihood": 65}], 4, False)[0] == "medium"
    assert cap_confidence("high", [{"likelihood": 80}], 4, False)[0] == "high"
    assert cap_confidence("low", [{"likelihood": 90}], 5, False)[0] == "low"  # the model can lower it, never raise it
    assert "48%" in cap_confidence("high", cands, 4, False)[1]


def test_opening_ranks_from_the_pack_and_picks_the_question_that_separates_the_leaders():
    candidates, question = opening(PACK)
    assert [c["cause"] for c in candidates] == ["A", "B", "C"] and sum(c["likelihood"] for c in candidates) == 100
    assert candidates[0]["likelihood"] <= 60  # capped: the opening must not look like a verdict
    assert question == "Is the 12V battery below 11.5 V?"  # separates A from B, unlike the connector question
    assert pick_question(PACK["question_bank"], [{"question": question}], candidates) == "Is the connector damaged?"
    assert opening({}) == ([], "")


def test_start_needs_no_ai_call_when_the_pack_has_questions():
    llm = ScriptedLLM([])
    _, state = _start(llm)
    assert state["status"] == "asking" and state["pending_question"] == "Is the 12V battery below 11.5 V?"
    assert llm.calls == [] and state["start_candidates"] == state["candidates"]


def test_start_falls_back_to_the_model_when_the_pack_has_no_questions():
    llm = ScriptedLLM([_step([("A", 45), ("B", 35), ("C", 20)], "Is the connector damaged?")])
    _, state = _start(llm, pack={"causes": [], "question_bank": []})
    assert state["status"] == "asking" and state["pending_question"] == "Is the connector damaged?"
    assert [p for p, _ in llm.calls] == ["propose"]


def test_answers_narrow_down_and_finish_with_a_capped_conclusion():
    llm = ScriptedLLM([
        _step([("A", 55), ("B", 30), ("C", 15)], "Q2?", showed="12V was low, A up"),
        _step([("A", 70), ("B", 20), ("C", 10)], "Q3?"),
        _step([("A", 90), ("B", 6), ("C", 4)], "Q4?"),
    ])
    g, state = _start(llm)
    state = _answer(g, state, "yes")
    assert state["status"] == "asking" and state["pending_question"] == "Q2?" and state["history"][0]["showed"] == "12V was low, A up"
    state = _answer(g, state, "no")
    assert state["status"] == "asking" and len(state["history"]) == 2
    state = _answer(g, state, "yes")  # third yes/no answer and the leader is far ahead: concludes
    assert state["status"] == "concluded" and state["pending_question"] is None
    assert state["conclusion"]["fix_steps"] == ["s1", "s2"] and state["conclusion"]["confidence"] == "high"
    assert "85%" in state["conclusion"]["confidence_note"]  # the model said 90%, but one answer may not cut a cause by more than half
    assert [p for p, _ in llm.calls] == ["step", "step", "step", "conclude"]
    assert "Opening candidates" in llm.calls[-1][1] and "showed: 12V was low" in llm.calls[-1][1]


def test_not_sure_answers_do_not_end_the_questions_early():
    llm = ScriptedLLM([_step([("A", 90), ("B", 6), ("C", 4)], f"Q{i}?") for i in range(2, 8)])
    g, state = _start(llm)
    state = _answer(g, state, "not_sure")
    state = _answer(g, state, "not_sure")
    state = _answer(g, state, "yes")
    assert state["status"] == "asking"  # leader is far ahead but only one Yes/No answer so far
    state = _answer(g, state, "no")
    state = _answer(g, state, "yes")
    assert state["status"] == "concluded" and informative_answers(state["history"]) == 3


def test_a_repeated_question_gets_one_retry_and_then_a_low_confidence_conclusion():
    llm = ScriptedLLM([_step([("A", 50), ("B", 50)], "Is the 12V battery below 11.5 V?"),
                       _step([("A", 52), ("B", 48)], "is the 12v battery below 11.5 v")])
    g, state = _start(llm)
    state = _answer(g, state, "yes")
    assert [p for p, _ in llm.calls].count("step") == 2 and "DIFFERENT" in llm.calls[1][1]
    assert state["status"] == "concluded" and state["conclusion"]["confidence"] == "low"


def test_a_repeated_question_that_is_fixed_on_retry_keeps_asking():
    llm = ScriptedLLM([_step([("A", 50), ("B", 50)], "Is the 12V battery below 11.5 V?"),
                       _step([("A", 52), ("B", 48)], "Is the connector damaged?")])
    g, state = _start(llm)
    state = _answer(g, state, "yes")
    assert state["status"] == "asking" and state["pending_question"] == "Is the connector damaged?"


def test_stops_after_the_maximum_number_of_questions():
    steps = [_step([("A", 40), ("B", 35), ("C", 25)], f"Q{i}?") for i in range(MAX_QUESTIONS + 1)]
    g, state = _start(ScriptedLLM(steps))
    for _ in range(MAX_QUESTIONS):
        assert state["status"] == "asking"
        state = _answer(g, state, "not_sure")
    assert state["status"] == "concluded" and len(state["history"]) == MAX_QUESTIONS
    assert state["conclusion"]["confidence"] == "low"


def test_reaching_the_question_limit_is_not_reported_as_a_forced_stop():
    # nine different questions, then a repeat exactly at the limit: the limit ended it, not the repeat
    steps = [_step([("A", 70), ("B", 25), ("C", 5)], f"Q{i}?") for i in range(MAX_QUESTIONS - 1)]
    steps.append(_step([("A", 70), ("B", 25), ("C", 5)], "Q0?"))
    g, state = _start(ScriptedLLM(steps))
    for i in range(MAX_QUESTIONS):
        state = _answer(g, state, "yes" if i % 2 else "no")
    assert state["status"] == "concluded" and state["forced"] is False
    assert state["conclusion"]["confidence"] == "medium"  # 70% leader and 10 Yes/No answers: not forced down to low
    assert "limit of 10 questions" in state["conclusion"]["confidence_note"]
    assert "no new question" not in state["conclusion"]["confidence_note"]


def test_near_duplicate_questions_count_as_repeats():
    from app.diagnosis_graph import _is_repeat
    asked = [{"question": "Did TACO identify a faulty BDU or another battery switching component in addition to the negative contactor?"}]
    assert _is_repeat("Does TACO identify a faulty BDU or another battery switching component in addition to the negative contactor?", asked)
    assert not _is_repeat("Is the 12V battery voltage above 13 V?", [{"question": "Is the 12V battery voltage below 11.5 V?"}])


def test_swing_limits_not_sure_changes_nothing_and_one_answer_cannot_erase_a_cause():
    from app.diagnosis_graph import limit_swing
    prev = [{"cause": "A", "likelihood": 60, "reasoning": "x"}, {"cause": "B", "likelihood": 30, "reasoning": "x"}, {"cause": "C", "likelihood": 10, "reasoning": "x"}]
    new = [{"cause": "A", "likelihood": 94, "reasoning": "y"}, {"cause": "B", "likelihood": 4, "reasoning": "y"}, {"cause": "C", "likelihood": 2, "reasoning": "y"}]
    assert limit_swing(prev, new, "not_sure") == prev
    out = {c["cause"]: c["likelihood"] for c in limit_swing(prev, new, "no")}
    assert sum(out.values()) == 100 and out["B"] >= 13 and out["C"] >= 4  # about half of 30 and of 10 survives after rescaling
    assert limit_swing([], new, "yes") == new
