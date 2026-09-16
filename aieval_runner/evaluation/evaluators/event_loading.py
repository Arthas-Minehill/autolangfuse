from __future__ import annotations

from pathlib import Path

from aieval_runner.evaluation.events import EventSummary, load_event_summary
from aieval_runner.core.models import AgentExecution


def load_execution_events(execution: AgentExecution) -> EventSummary:
    output = execution.output
    event_path = output.get("event_path") if isinstance(output, dict) else None
    if not isinstance(event_path, str) or not event_path.strip():
        return EventSummary(errors=["missing_event_path"])
    path = Path(event_path)
    if not path.is_file():
        return EventSummary(errors=["event_file_missing"])
    try:
        return load_event_summary(path)
    except (OSError, ValueError) as exc:
        return EventSummary(errors=[f"invalid_event_file:{type(exc).__name__}"])
