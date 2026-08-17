from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def render_html(trace: dict) -> str:
    nodes = trace.get("nodes", {})
    summary = trace.get("summary", {})
    rows = []
    for node_id, node in sorted(nodes.items()):
        depth = html.escape(str(node.get("depth")))
        rows.append(
            f'<tr data-depth="{depth}">'
            f"<td>{html.escape(str(node_id))}</td>"
            f"<td>{html.escape(str(node.get('parent_id') or ''))}</td>"
            f"<td>{depth}</td>"
            f"<td>{html.escape(str(node.get('status')))}</td>"
            f"<td>{html.escape(_fmt(node.get('latency_seconds')))}</td>"
            f"<td>{html.escape(str(node.get('total_tokens') or ''))}</td>"
            f"<td>{html.escape(str(node.get('provider') or ''))}</td>"
            f"<td>{html.escape(str(node.get('model') or ''))}</td>"
            f"<td>{html.escape(str(node.get('stopping_reason') or ''))}</td>"
            f"<td>{html.escape(str(node.get('aggregation_status') or ''))}</td>"
            f"<td><details><summary>preview</summary>{html.escape(str(node.get('task_preview') or ''))}</details></td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Efficient RLM Trace {html.escape(str(trace.get('run_id', '')))}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #17202a;
    }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 20px 0;
    }}
    .metric {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 10px; background: #f6f8fa; }}
    .metric strong {{ display: block; font-size: 12px; color: #57606a; margin-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f8fa; text-align: left; }}
    tr[data-depth="1"] td:first-child {{ padding-left: 24px; }}
    tr[data-depth="2"] td:first-child {{ padding-left: 40px; }}
  </style>
</head>
<body>
  <h1>Efficient RLM Trace</h1>
  <p>Run ID: <code>{html.escape(str(trace.get('run_id', '')))}</code></p>
  <section class="summary">
    {_metric('Mode', summary.get('mode'))}
    {_metric('Calls', summary.get('calls'))}
    {_metric('Tasks', summary.get('tasks'))}
    {_metric('Max depth', summary.get('max_depth_reached'))}
    {_metric('Wall time', _fmt(summary.get('wall_time_seconds')))}
    {_metric('Total tokens', summary.get('total_tokens'))}
  </section>
  <table>
    <thead>
      <tr>
        <th>Node</th><th>Parent</th><th>Depth</th><th>Status</th><th>Latency</th>
        <th>Tokens</th><th>Provider</th><th>Model</th><th>Stop reason</th><th>Aggregation</th><th>Task</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def _metric(label: str, value) -> str:
    escaped_label = html.escape(label)
    escaped_value = html.escape(str(value if value is not None else ""))
    return f'<div class="metric"><strong>{escaped_label}</strong>{escaped_value}</div>'


def _fmt(value) -> str:
    return f"{value:.4f}s" if isinstance(value, (int, float)) else ""


def render_file(input_path: str | Path, output_path: str | Path) -> str:
    trace = json.loads(Path(input_path).read_text(encoding="utf-8"))
    html_text = render_html(trace)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_text, encoding="utf-8")
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an Efficient RLM trace JSON file as self-contained HTML.")
    parser.add_argument("trace_json")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = args.output or str(Path(args.trace_json).with_suffix(".html"))
    print(render_file(args.trace_json, output))


if __name__ == "__main__":
    main()
