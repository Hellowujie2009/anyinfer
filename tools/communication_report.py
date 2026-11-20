"""Summarize torch-npu's per-step HCCL communication.json on each rank."""

import json
import math
import re
from collections import defaultdict


OP = re.compile(r"^(?:hcom|hccl)[_:/-]?(alltoallvc|alltoallv|alltoall|allreduce|allgather|"
                r"reducescatter|batchsendrecv|broadcast|scatter|receive|reduce|send)", re.I)


def accumulate(totals, field, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(value) and value >= 0:
        statistic = totals.setdefault(field, {"sum": 0.0, "records": 0})
        statistic["sum"] += value
        statistic["records"] += 1


def rank_communication(directory):
    paths = sorted(directory.rglob("communication.json"))
    # torch-npu retains raw CANN inputs with a different schema.
    exported = [p for p in paths if p.parent.name == "ASCEND_PROFILER_OUTPUT"]
    candidates = exported or paths
    if not candidates:
        return {"status": "not_exported"}
    if len(candidates) != 1:
        return {"status": "ambiguous", "files": [str(p.relative_to(directory)) for p in candidates]}
    path, = candidates
    groups = {}
    for step, categories in json.loads(path.read_text(encoding="utf-8-sig")).items():
        if not step.startswith("step") or not isinstance(categories, dict):
            continue
        for category in ("collective", "p2p"):
            for name, op in categories.get(category, {}).items():
                if name.lower().startswith("total") or not isinstance(op, dict):
                    continue
                timings = op.get("Communication Time Info", {})
                links = op.get("Communication Bandwidth Info", {})
                if not timings and not links:
                    continue
                match = OP.match(name)
                kind = match[1].lower() if match else "other"
                row = groups.setdefault((category, kind), {"category": category, "operation": kind,
                    "records": 0, "example": name, "times_ms": {}, "transport": defaultdict(dict)})
                row["records"] += 1
                for key, field in (("elapsed", "Elapse Time(ms)"), ("wait", "Wait Time(ms)"),
                                   ("synchronization", "Synchronization Time(ms)"), ("transit", "Transit Time(ms)")):
                    accumulate(row["times_ms"], key, timings.get(field))
                for transport, measurements in links.items():
                    if transport.lower().startswith("total") or not isinstance(measurements, dict):
                        continue
                    for key, field in (("transit_ms", "Transit Time(ms)"), ("transit_mb", "Transit Size(MB)")):
                        accumulate(row["transport"][transport], key, measurements.get(field))
    status = "no_matching_operations"
    if groups:
        has_values = any(row["times_ms"] or any(row["transport"].values()) for row in groups.values())
        status = "exported" if has_values else "no_numeric_fields"
    return {"status": status, "file": str(path.relative_to(directory)), "operations": list(groups.values())}


def communication_summary(trace_root, expected_ranks):
    ranks = []
    for rank in range(expected_ranks):
        try:
            data = rank_communication(trace_root / f"rank-{rank}")
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            data = {"status": "unreadable", "error": str(exc)}
        ranks.append({"rank": rank, **data})
    return {"ranks": ranks, "ranks_with_measurements": sum(r["status"] == "exported" for r in ranks),
            "scope": "primary operator replay; per-rank HCCL records; time fields may overlap"}
