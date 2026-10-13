"""Cleaning, exact counting and the knowledge-pack checks. No database and no AI."""
from app.cases import Case
from app.pack import Pack, PackCause, PackQuestion, Quote, verify_pack
from app.prepare import clean_manual, clean_note, part_name, prepare_cases


def _case(i, note, parts):
    return Case(i, "MG_COMET_EV", "c", note, parts)


def test_notes_lose_greetings_and_fillers_but_keep_findings():
    out = clean_note("Dear sir, Greetings from MG Attapur Service….!   We found the 12V battery at 6.5 V.   ")
    assert out.startswith("We found the 12V battery") and "Dear" not in out and "Greetings" not in out


def test_manual_page_markers_and_broken_lines_are_fixed():
    text = "Circuit/System\ninspection continues\n--- Page 155 ---\n─ Yes.\nGo to step 2：now"
    out = clean_manual(text)
    assert "Page 155" not in out and "Circuit/System inspection continues" in out and "- Yes." in out and "step 2:" in out


def test_part_codes_become_names():
    assert part_name("000000000023574986-GASKET-EXH MUFF") == "GASKET-EXH MUFF"
    assert part_name("MOT CONT") == "MOT CONT"


def test_copies_are_counted_once_and_parts_are_counted_exactly():
    cases = [
        _case(1, "We replaced the motor controller.", ["000000000000000001-MOT CONT"]),
        _case(2, "We  replaced the motor CONTROLLER!!", ["000000000000000001-MOT CONT"]),  # same note, other spelling
        _case(3, "Nothing found, code did not return.", ["NOT APPLICABLE"]),
        _case(4, "12V battery was weak, replaced.", ["000000000000000002-BATTERY 35AH"]),
    ]
    p = prepare_cases(cases)
    assert (p.raw_cases, p.distinct_cases) == (4, 3)
    assert p.no_part_cases == 1 and p.pct(p.no_part_cases) == 33
    assert dict(p.part_counts) == {"MOT CONT": 1, "BATTERY 35AH": 1}
    assert p.cases_with_parts({"MOT CONT"}) == (1, 2)  # one distinct note, two raw cases


def _prepared():
    return prepare_cases([
        _case(1, "We found the 12V battery at 6.5 V and replaced the 12V battery.", ["000000000000000002-BATTERY 35AH"]),
        _case(2, "Motor controller faulty, replaced the controller.", ["000000000000000001-MOT CONT"]),
        _case(3, "No fault found, code cleared and did not return.", ["NOT APPLICABLE"]),
    ])


def _pack(causes, questions=()):
    return Pack(fault_meaning="m", setting_conditions="s", causes=causes, question_bank=list(questions), gaps=[])


def test_verify_drops_invented_quotes_parts_and_unsupported_causes_and_computes_counts():
    good = PackCause(cause="Weak 12V battery", origin="past_repairs", matching_parts=["BATTERY 35AH", "MADE UP PART"], replaces_nothing=False,
                     quotes=[Quote(source="case", case_id=1, text="the 12V battery at 6.5 V"),
                             Quote(source="case", case_id=1, text="this sentence is not in the note")],
                     how_to_tell=["12V under 11.5 V"], repair_steps=[])
    unsupported = PackCause(cause="Wiring harness", origin="past_repairs", matching_parts=["NOPE"], replaces_nothing=False,
                            quotes=[Quote(source="case", case_id=99, text="harness chafed")], how_to_tell=[], repair_steps=[])
    manual_only = PackCause(cause="Relay", origin="manual", matching_parts=[], replaces_nothing=False, how_to_tell=[], repair_steps=[],
                            quotes=[Quote(source="manual", text="main negative relay is not closed")])
    q = PackQuestion(question="Is the 12V battery below 11.5 V?", source="past_repairs",
                     yes_points_to=["Weak 12V battery", "Invented cause"], no_points_to=["Relay"])
    out, checks = verify_pack(_pack([good, unsupported, manual_only], [q]), _prepared(), "The Main Negative relay is not closed.")
    names = [c["cause"] for c in out["causes"]]
    assert names == ["Weak 12V battery", "Relay"] and checks["causes_dropped"] == 1
    battery = out["causes"][0]
    assert battery["matching_parts"] == ["BATTERY 35AH"] and battery["cases_counted"] == 1 and battery["share_pct"] == 33
    assert len(battery["quotes"]) == 1 and checks["quotes_dropped"] == 2 and checks["parts_dropped"] == 2
    assert out["question_bank"][0]["yes_points_to"] == ["Weak 12V battery"]  # the invented cause is removed
    assert out["causes"][1]["quotes"][0]["source"] == "manual" and out["causes"][1]["share_pct"] is None  # not countable


def test_a_no_part_cause_gets_the_exact_no_part_share():
    c = PackCause(cause="Intermittent, nothing found", origin="past_repairs", matching_parts=[], replaces_nothing=True,
                  quotes=[Quote(source="case", case_id=3, text="No fault found")], how_to_tell=[], repair_steps=[])
    out, _ = verify_pack(_pack([c]), _prepared(), "")
    assert out["causes"][0]["cases_counted"] == 1 and out["causes"][0]["share_pct"] == 33


def test_verify_flags_missing_manual_and_thin_data():
    out, _ = verify_pack(_pack([]), _prepared(), "")
    assert any("No complete manual" in g for g in out["gaps"]) and any("distinct past repairs" in g for g in out["gaps"])
