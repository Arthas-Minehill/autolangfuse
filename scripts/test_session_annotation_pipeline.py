from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("session_annotation_pipeline.py")
SPEC = importlib.util.spec_from_file_location("session_annotation_pipeline", SCRIPT)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_build_candidate_selects_root_observation() -> None:
    row = {
        "case_id": "case-1",
        "source_session_id": "session-1",
        "original_input": "请查询上周 UPS 发货量",
        "user_turns": "请查询上周 UPS 发货量",
        "resolved_input": "查询上周 UPS 发货量",
    }
    session = {
        "turns": [
            {
                "traceId": "trace-1",
                "userInput": {"role": "user", "content": "你好"},
                "observations": [],
            },
            {
                "traceId": "trace-2",
                "userInput": {"role": "user", "content": "请查询上周 UPS 发货量"},
                "observations": [
                    {"id": "child", "parentObservationId": "root", "isRootObservation": False},
                    {"id": "root", "parentObservationId": None, "isRootObservation": True},
                ],
            },
        ]
    }

    candidate = pipeline.build_candidate(row, session)

    assert candidate["source_turn_indexes"] == [2]
    assert candidate["source_trace_ids"] == ["trace-2"]
    assert candidate["anchor_trace_id"] == "trace-2"
    assert candidate["anchor_observation_id"] == "root"
    assert candidate["mapping_confidence"] >= 0.9


def test_root_observation_falls_back_to_parentless() -> None:
    turn = {
        "observations": [
            {"id": "child", "parentObservationId": "root"},
            {"id": "root", "parentObservationId": None},
        ]
    }
    assert pipeline.root_observation(turn)["id"] == "root"


def test_normalize_text_ignores_spacing_and_punctuation() -> None:
    assert pipeline.normalize_text("UPS 发货量？") == pipeline.normalize_text("ups发货量")


def test_session_review_context_keeps_all_turns() -> None:
    session = {
        "turns": [
            {
                "traceId": "trace-1",
                "userInput": {"role": "user", "content": "问题一"},
                "finalOutput": {"role": "assistant", "content": "回答一"},
            },
            {
                "traceId": "trace-2",
                "userInput": {"role": "user", "content": "补充条件"},
                "finalOutput": {"role": "assistant", "content": "最终回答"},
            },
        ]
    }
    assert pipeline.session_review_context(session) == [
        {"turn_index": 1, "trace_id": "trace-1", "user": "问题一", "assistant": "回答一"},
        {"turn_index": 2, "trace_id": "trace-2", "user": "补充条件", "assistant": "最终回答"},
    ]


def test_proposed_admission_status_is_conservative() -> None:
    assert pipeline.proposed_admission_status({"gold_status": "待重制-澄清案例"})[0] == "需要澄清"
    assert pipeline.proposed_admission_status({"review_notes": "包含两个目标，需要拆分"})[0] == "需要拆分"
    assert pipeline.proposed_admission_status({"gold_status": "待独立验证"})[0] == "正常回答"
