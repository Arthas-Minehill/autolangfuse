from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol

from aieval_runner.core.models import AgentExecution, EvalCase, EvalRunContext, EvaluatorResult
from aieval_runner.evaluation.events import EventSummary


JsonObject = Dict[str, object]


@dataclass(frozen=True)
class EvaluationInput:
    case: EvalCase
    execution: AgentExecution
    ctx: EvalRunContext
    events: EventSummary

    @property
    def calls(self) -> List[JsonObject]:
        return self.events.tool_calls


class CaseEvaluator(Protocol):
    name: str

    def evaluate(self, data: EvaluationInput) -> EvaluatorResult:
        ...


EvaluatorFunction = Callable[[EvaluationInput], EvaluatorResult]


@dataclass(frozen=True)
class FunctionCaseEvaluator:
    name: str
    function: EvaluatorFunction

    def evaluate(self, data: EvaluationInput) -> EvaluatorResult:
        return self.function(data)
