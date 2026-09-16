from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("session_recovery_launcher.py")
SPEC = importlib.util.spec_from_file_location("session_recovery_launcher", SCRIPT)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_time_range_uses_configured_offset() -> None:
    assert launcher.resolve_time_range(
        "2026-09-01 00:00:00",
        "2026-09-02 00:00:00",
        all_sessions=False,
        offset_text="+08:00",
    ) == ("2026-08-31T16:00:00Z", "2026-09-01T16:00:00Z")


def test_all_sessions_ignores_empty_time_fields() -> None:
    assert launcher.resolve_time_range(
        "", "", all_sessions=True, offset_text="+08:00"
    ) == (None, None)


def test_parse_ai_cases_accepts_json_fence() -> None:
    assert launcher.parse_ai_cases(
        '```json\n{"cases":[{"resolved_input":"查询库存"}]}\n```'
    ) == [{"resolved_input": "查询库存"}]


def test_build_sessions_skips_observations_without_session_id() -> None:
    observations = [
        {
            "sessionId": "s1",
            "traceId": "t1",
            "id": "root",
            "parentObservationId": None,
            "startTime": "2026-09-01T00:00:00Z",
            "input": '{"role":"user","content":"问题"}',
            "output": '{"role":"assistant","content":"回答"}',
        },
        {"traceId": "t2", "id": "orphan"},
    ]
    sessions, skipped = launcher.build_sessions(observations)
    assert len(sessions) == 1
    assert sessions[0]["sessionId"] == "s1"
    assert sessions[0]["turnCount"] == 1
    assert skipped == 1


def test_run_id_is_preserved_in_normalized_case() -> None:
    row = launcher.normalize_case(
        {"original_input": "查库存", "resolved_input": "查询当前库存"},
        run_id="recovery_1",
        case_number=1,
        session_id="s1",
        snapshot_at="2026-09-01T00:00:00Z",
    )
    assert row["recovery_run_id"] == "recovery_1"
    assert row["case_id"] == "case_0001"
    assert row["source_session_id"] == "s1"


def test_redaction_covers_all_credentials() -> None:
    redacted = launcher.redact_config(
        {
            "LANGFUSE_PUBLIC_KEY": "public",
            "LANGFUSE_SECRET_KEY": "secret",
            "LANGFUSE_LOGIN_PASSWORD": "password",
            "AI_API_KEY": "ai-secret",
        }
    )
    assert set(redacted.values()) == {"<已配置>"}


def test_end_to_end_dry_run_does_not_write_remote(tmp_path, monkeypatch) -> None:
    observations = [
        {
            "projectId": "p1",
            "sessionId": "s1",
            "traceId": "t1",
            "id": "o1",
            "isRootObservation": True,
            "parentObservationId": None,
            "startTime": "2026-09-01T00:00:00Z",
            "input": {"role": "user", "content": "查询当前库存"},
            "output": {"role": "assistant", "content": "库存为 10"},
            "type": "SPAN",
        }
    ]
    monkeypatch.setattr(
        launcher,
        "fetch_observations",
        lambda config, start_utc, end_utc, log: observations,
    )
    monkeypatch.setattr(
        launcher,
        "review_session_with_ai",
        lambda config, prompt, session: [
            {
                "original_input": "查询当前库存",
                "original_output": "库存为 10",
                "user_turns": "查询当前库存",
                "resolved_input": "查询当前库存",
                "primary_intent": "数据查询与交付",
                "secondary_intent": "单指标取数",
                "execution_status": "成功",
                "clarification_status": "无需澄清",
                "context_required": "不需要",
                "gold_status": "待独立验证",
            }
        ],
    )
    result = launcher.run_workflow(
        {
            "LANGFUSE_PROJECT_ID": "p1",
            "SESSION_TIMEZONE_OFFSET": "+08:00",
            "SESSION_OUTPUT_ROOT": str(tmp_path),
            "AI_REVIEW_PROMPT_FILE": str(launcher.DEFAULT_PROMPT_PATH),
        },
        launcher.RunOptions(
            start_text="2026-09-01 00:00:00",
            end_text="2026-09-02 00:00:00",
            all_sessions=False,
            limit=1,
            write_remote=False,
        ),
        lambda message: None,
    )
    run_dir = Path(result["run_dir"])
    assert result["session_count"] == 1
    assert result["case_count"] == 1
    assert (run_dir / "candidate_cases.jsonl").exists()
    assert not (run_dir / "review_observations.jsonl").exists()
