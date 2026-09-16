from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from aieval_runner.core.json_utils import json_default
from aieval_runner.core.models import EvalScore, LangfuseApiError, RunnerConfig


class LangfuseClient:
    def __init__(self, *, host: str, public_key: str, secret_key: str, timeout_seconds: int = 30) -> None:
        self.host = host.rstrip("/")
        self.public_key = public_key
        self.secret_key = secret_key
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.public_key or not self.secret_key:
            raise LangfuseApiError("缺少 LANGFUSE_PUBLIC_KEY 或 LANGFUSE_SECRET_KEY")

        url = self._public_api_url(path, query)
        payload = None
        headers = {
            "Accept": "application/json; charset=utf-8",
            "Content-Type": "application/json; charset=utf-8",
        }
        token = f"{self.public_key}:{self.secret_key}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(token).decode("ascii")
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False, default=json_default).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LangfuseApiError(f"Langfuse API 返回 {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LangfuseApiError(f"Langfuse API 连接失败: {exc}") from exc

        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LangfuseApiError(f"Langfuse API 返回非 JSON: {raw[:200]}") from exc
        if isinstance(parsed, dict):
            return parsed
        return {"data": parsed}

    def _public_api_url(self, path: str, query: Optional[Dict[str, Any]]) -> str:
        clean_path = path if path.startswith("/") else f"/{path}"
        url = f"{self.host}/api/public{clean_path}"
        clean_query = {key: value for key, value in (query or {}).items() if value not in (None, "")}
        if clean_query:
            url += "?" + urllib.parse.urlencode(clean_query)
        return url


_LANGFUSE_SDK: Optional[Any] = None


def api_from_config(config: RunnerConfig) -> Optional[LangfuseClient]:
    if not config.langfuse_enabled:
        return None
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        return None
    return LangfuseClient(
        host=config.langfuse_host,
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
    )


def get_langfuse_sdk(config: RunnerConfig) -> Any:
    global _LANGFUSE_SDK
    if _LANGFUSE_SDK is not None:
        return _LANGFUSE_SDK
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", config.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", config.langfuse_host)
    try:
        from langfuse import Langfuse
    except ImportError as exc:
        raise LangfuseApiError("写入 Score 需要安装 langfuse: pip install -r requirements.txt") from exc
    _LANGFUSE_SDK = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
    )
    return _LANGFUSE_SDK


def write_score(score: EvalScore, *, config: RunnerConfig, trace_id: str, observation_id: Optional[str]) -> None:
    lf = get_langfuse_sdk(config)
    score_id = ":".join([
        "aieval",
        str(score.metadata.get("eval_run_id")),
        str(score.metadata.get("case_id")),
        score.name,
    ])
    payload = {
        "name": score.name,
        "value": score.value,
        "trace_id": trace_id,
        "score_id": score_id,
        "data_type": score.data_type,
        "comment": score.comment,
        "metadata": score.metadata,
    }
    if observation_id:
        payload["observation_id"] = observation_id
    lf.create_score(**payload)


def flush_scores() -> None:
    if _LANGFUSE_SDK is None:
        return
    try:
        _LANGFUSE_SDK.flush()
    except Exception:
        pass
