from __future__ import annotations

import json

from aieval_runner.core.constants import (
    EXPECTED_OUTPUT_SCHEMA,
    INPUT_SCHEMA,
    METADATA_SCHEMA,
    SCORE_METADATA_KEYS,
    TRACE_METADATA_SCHEMA,
)


def print_protocol_schemas() -> None:
    schemas = {
        "input_schema": INPUT_SCHEMA,
        "metadata_schema": METADATA_SCHEMA,
        "expected_output_schema": EXPECTED_OUTPUT_SCHEMA,
        "trace_metadata_schema": TRACE_METADATA_SCHEMA,
        "score_metadata_keys": SCORE_METADATA_KEYS,
    }
    print(json.dumps(schemas, ensure_ascii=False, indent=2))
