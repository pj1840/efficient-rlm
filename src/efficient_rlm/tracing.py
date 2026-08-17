from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from uuid import uuid4

from efficient_rlm.models import Trace, TraceNode


def preview(text: str, limit: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


class TraceRecorder:
    def __init__(self, config: dict, run_id: str | None = None) -> None:
        self.trace = Trace(run_id=run_id or uuid4().hex[:12], started_at=time.time(), config=config)
        self._lock = Lock()

    def add_node(self, node_id: str, parent_id: str | None, depth: int, task: str) -> None:
        with self._lock:
            self.trace.nodes[node_id] = TraceNode(
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
                task_preview=preview(task),
                start_time=time.time(),
            )
            if parent_id and parent_id in self.trace.nodes:
                self.trace.nodes[parent_id].child_node_ids.append(node_id)
                self.trace.nodes[parent_id].child_node_ids.sort()

    def mark_decomposed(self, node_id: str) -> None:
        with self._lock:
            self.trace.nodes[node_id].decomposed = True

    def finish_node(
        self,
        node_id: str,
        status: str,
        output: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        stopping_reason: str | None = None,
        aggregation_status: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            node = self.trace.nodes[node_id]
            node.status = status
            node.end_time = time.time()
            if node.start_time is not None:
                node.latency_seconds = node.end_time - node.start_time
            node.output_preview = preview(output or "") if output is not None else None
            node.provider = provider
            node.model = model
            node.prompt_tokens = prompt_tokens
            node.completion_tokens = completion_tokens
            node.total_tokens = total_tokens
            node.stopping_reason = stopping_reason
            node.aggregation_status = aggregation_status
            node.error = error

    def finalize(self, final_answer: str, summary: dict) -> None:
        self.trace.final_answer_preview = preview(final_answer)
        self.trace.summary = summary

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self.trace)
        data["nodes"] = {node_id: asdict(node) for node_id, node in self.trace.nodes.items()}
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(target)


def render_trace(path: str | Path) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = data.get("nodes", {})
    roots = [node for node in nodes.values() if node.get("parent_id") is None]
    lines = [f"Trace {data.get('run_id', '<unknown>')}"]
    if data.get("summary"):
        summary = data["summary"]
        lines.append(
            "summary: "
            f"mode={summary.get('mode')} calls={summary.get('calls')} "
            f"tasks={summary.get('tasks')} seconds={summary.get('wall_time_seconds')}"
        )

    def walk(node: dict, indent: str = "") -> None:
        status = node.get("status")
        latency = node.get("latency_seconds")
        latency_text = f" {latency:.4f}s" if isinstance(latency, (int, float)) else ""
        reason = f" reason={node.get('stopping_reason')}" if node.get("stopping_reason") else ""
        lines.append(f"{indent}- {node.get('node_id')} depth={node.get('depth')} status={status}{latency_text}{reason}")
        if node.get("task_preview"):
            lines.append(f"{indent}  task: {node['task_preview']}")
        for child_id in node.get("child_node_ids", []):
            child = nodes.get(child_id)
            if child:
                walk(child, indent + "  ")

    for root in roots:
        walk(root)
    return "\n".join(lines)


def trace_to_mermaid(path: str | Path) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = data.get("nodes", {})
    lines = ["graph TD"]
    for node_id, node in nodes.items():
        label = f"{node_id}<br/>depth={node.get('depth')}<br/>{node.get('status')}"
        lines.append(f'  {node_id}["{label}"]')
        for child_id in node.get("child_node_ids", []):
            lines.append(f"  {node_id} --> {child_id}")
    return "\n".join(lines)
