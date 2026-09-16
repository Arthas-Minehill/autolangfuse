"""
Complete rebuild of optimized_dataset.json using MCP results for ALL 10 cases.
Handles both saved-file results and inline results.
"""
import json, sys
from pathlib import Path

TOOL_RESULTS = Path(r"C:\Users\Administrator\.claude\projects\D--code-aieval\a8438ac5-71c6-4bdb-bb72-f58ae6e53352\tool-results")
OPTIMIZED_JSON = Path(r"D:\code\aieval\docs\optimized_dataset.json")

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    data = json.loads(content)
    # Handle wrapped format
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'text' in data[0]:
        data = json.loads(data[0]['text'])
    return data

def mcp_data_to_table(mcp_result):
    """Convert MCP result's data array to columns/rows format."""
    data = mcp_result.get('data', [])
    if not data:
        return None
    columns = []
    seen = set()
    for row in data:
        if isinstance(row, dict):
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    rows = []
    for row in data:
        if isinstance(row, dict):
            rows.append([row.get(col) for col in columns])
    return {"columns": columns, "rows": rows}

# File mapping for ALL 10 cases
CASE_FILES = {
    "L1-METRIC-CUSTOMER-ACTIVE-002": "mcp-metric-mcp-remote-searchMetricAppQueryResult-1781601378577.txt",
    "L1-METRIC-CUSTOMER-ACTIVE-003": "mcp-metric-mcp-remote-searchMetricAppQueryResult-1781601380330.txt",
    "L1-METRIC-CUSTOMER-ACTIVE-005": "call_01_yWN2uBhq4HOjKEoQmUKY2003.json",
    "L1-METRIC-CUSTOMER-ACTIVE-007": "mcp-metric-mcp-remote-searchMetricAppQueryResult-1781601383136.txt",
    "L1-METRIC-CUSTOMER-ACTIVE-010": "mcp-metric-mcp-remote-searchMetricAppQueryResult-1781600675961.txt",
    # These will use inline data files we create below
    "L1-METRIC-CUSTOMER-ACTIVE-004": "inline_004.json",
    "L1-METRIC-CUSTOMER-ACTIVE-006": "inline_006.json",
    "L1-METRIC-CUSTOMER-ACTIVE-008": "inline_008.json",
    "L1-METRIC-CUSTOMER-ACTIVE-009": "inline_009.json",
    "L1-METRIC-CUSTOMER-ACTIVE-011": "inline_011.json",
}

def main():
    # Load existing dataset (has correct main_query values from user edits)
    with open(OPTIMIZED_JSON, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    updated = 0
    for item in dataset['items']:
        case_id = item['case_id']
        filename = CASE_FILES.get(case_id)
        if not filename:
            print(f"SKIP {case_id}: no file mapping")
            continue

        filepath = TOOL_RESULTS / filename
        if not filepath.exists():
            print(f"SKIP {case_id}: file not found: {filepath}")
            continue

        try:
            mcp_result = load_json_file(filepath)
        except Exception as e:
            print(f"SKIP {case_id}: failed to load: {e}")
            continue

        if not isinstance(mcp_result, dict) or 'data' not in mcp_result:
            print(f"SKIP {case_id}: no data key, type={type(mcp_result).__name__}")
            continue

        table = mcp_data_to_table(mcp_result)
        if not table:
            print(f"SKIP {case_id}: empty data")
            continue

        item['expectedOutput']['expected_result'] = table
        print(f"OK {case_id}: {len(table['columns'])} cols x {len(table['rows'])} rows")
        updated += 1

    with open(OPTIMIZED_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(f"\nTotal updated: {updated}/10")
    return 0 if updated == 10 else 1

if __name__ == '__main__':
    sys.exit(main())
