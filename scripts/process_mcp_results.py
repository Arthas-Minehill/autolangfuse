"""
Processing script: Extract MCP tool results and rebuild optimized_dataset.json.
Run after all searchMetricAppQueryResult calls are complete.
"""
import json, os, sys
from pathlib import Path

TOOL_RESULTS_DIR = Path(r"C:\Users\Administrator\.claude\projects\D--code-aieval\a8438ac5-71c6-4bdb-bb72-f58ae6e53352\tool-results")
OPTIMIZED_JSON = Path(r"D:\code\aieval\docs\optimized_dataset.json")

# Map case_id -> (main_query, mcp_result_file_name)
CASE_FILE_MAP = {
    "L1-METRIC-CUSTOMER-ACTIVE-002": "mcp-metric-mcp-remote-searchMetricAppQueryResult-1781601378577.txt",
    "L1-METRIC-CUSTOMER-ACTIVE-003": "mcp-metric-mcp-remote-searchMetricAppQueryResult-1781601380330.txt",
    "L1-METRIC-CUSTOMER-ACTIVE-005": "call_01_yWN2uBhq4HOjKEoQmUKY2003.json",
    "L1-METRIC-CUSTOMER-ACTIVE-007": "mcp-metric-mcp-remote-searchMetricAppQueryResult-1781601383136.txt",
    "L1-METRIC-CUSTOMER-ACTIVE-010": "mcp-metric-mcp-remote-searchMetricAppQueryResult-1781600675961.txt",
}

def load_mcp_result(filename):
    """Load MCP result from file. Handles both raw JSON and wrapped (array with text) formats."""
    filepath = TOOL_RESULTS_DIR / filename
    if not filepath.exists():
        print(f"  WARNING: File not found: {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print(f"  WARNING: Invalid JSON in {filename}")
        return None
    # Handle wrapped format (array with {"type": "text", "text": "..."})
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'text' in data[0]:
        try:
            data = json.loads(data[0]['text'])
        except (json.JSONDecodeError, KeyError):
            print(f"  WARNING: Could not unwrap {filename}")
            return None
    # Result should have a 'data' key with the rows
    if isinstance(data, dict) and 'data' in data:
        return data
    print(f"  WARNING: No 'data' key in {filename}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    return None

def data_to_table(mcp_result):
    """Convert MCP result's data array (list of dicts) to columns/rows table format."""
    data = mcp_result.get('data', [])
    if not data:
        return None
    # Collect all unique column names in order of first appearance
    columns = []
    seen = set()
    for row in data:
        if isinstance(row, dict):
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    # Build rows
    rows = []
    for row in data:
        if isinstance(row, dict):
            rows.append([row.get(col) for col in columns])
    return {"columns": columns, "rows": rows}

def main():
    # Load optimized dataset
    with open(OPTIMIZED_JSON, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_updated = 0

    for item in dataset['items']:
        case_id = item['case_id']
        filename = CASE_FILE_MAP.get(case_id)

        if filename:
            print(f"Processing {case_id} from {filename}...")
            mcp_result = load_mcp_result(filename)
            if mcp_result:
                table = data_to_table(mcp_result)
                if table:
                    item['expectedOutput']['expected_result'] = table
                    print(f"  -> {len(table['columns'])} columns, {len(table['rows'])} rows")
                    total_updated += 1
                else:
                    print(f"  -> No data rows found!")
            else:
                print(f"  -> Failed to load MCP result!")
        else:
            print(f"Skipping {case_id} (no saved file, will be handled separately)")

    # Save updated dataset
    with open(OPTIMIZED_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(f"\nUpdated {total_updated} cases. Saved to {OPTIMIZED_JSON}")

if __name__ == '__main__':
    main()
