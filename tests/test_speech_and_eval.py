"""Contract tests for the STT text output and the eval dataset schema."""
import json
from pathlib import Path
from types import SimpleNamespace

from aliengo.speech.stt import combine_segments

EVAL_PATH = Path(__file__).parent / "eval_commands.json"
KNOWN_OUTCOME_KEYS = {
    "executed", "params", "must_block", "must_not_execute",
    "must_confirm", "allowed_only", "no_tools",
}


def test_combine_segments_single_line():
    segments = [
        SimpleNamespace(text="  Stand up "),
        SimpleNamespace(text="and move forward.\n"),
    ]
    assert combine_segments(segments) == "Stand up and move forward."


def test_combine_segments_empty():
    assert combine_segments([]) == ""


def test_eval_dataset_schema():
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    assert cases, "eval dataset must not be empty"
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in cases:
        assert case["command"].strip()
        assert case["any_of"], f"{case['id']}: needs at least one outcome"
        for outcome in case["any_of"]:
            unknown = set(outcome) - KNOWN_OUTCOME_KEYS
            assert not unknown, f"{case['id']}: unknown outcome keys {unknown}"
