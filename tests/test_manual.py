import json
from pathlib import Path

from app.manual import is_complete_procedure


ROOT = Path(__file__).resolve().parent.parent


def test_completeness_rejects_cut_sections_and_accepts_terminal_procedure():
    complete = (
        "DTC P100016 Diagnostic Description. Circuit/System Inspection. "
        "1. Is voltage correct? Yes. Proceed. No. Repair the circuit. "
        "2. Is the DTC still set? Yes. Replace the module. No. Diagnosis completed. Maintenance Guide."
    )
    assert is_complete_procedure(complete, "P100016")
    assert not is_complete_procedure(complete.replace("Diagnosis completed", "Proceed to the next inspection"), "P100016")
    assert not is_complete_procedure("18-20 DTC Repair Instructions 11. Replace module. " + complete, "P100016")


def test_repository_manual_allowlist_matches_audited_complete_procedures():
    data = json.loads((ROOT / "dtc_requested.json").read_text(encoding="utf-8"))
    complete_counts = {
        (entry["model"], entry["dtc_code"]): sum(
            section.get("complete") is True
            and section.get("end_of_procedure") is True
            and is_complete_procedure(section.get("content", ""), entry["dtc_code"])
            for section in entry["subsystems"]
        )
        for entry in data
    }
    assert complete_counts == {
        ("Windsor EV/Pro", "U140687"): 6,
        ("MG Comet EV", "U200487"): 6,
        ("Hector MCE", "P100016"): 3,
        ("Hector MCE", "U140787"): 4,
        ("ZS EV", "U0245"): 0,
        ("Hector BS4/BS6", "P100016"): 2,
        ("M9", "U014087"): 0,
    }
