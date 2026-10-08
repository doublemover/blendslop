"""Append-only result indexes and leaderboards for refinement runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .contracts import ExperimentResult, json_safe, stable_hash
from .parameter_search import promotion_decision, rank_results, score_result
try:
    from blender_blocking.metrics.namespaces import get_metric_path
    from blender_blocking.utils.json_io import write_json, write_jsonl
except ImportError:  # pragma: no cover
    from metrics.namespaces import get_metric_path
    from utils.json_io import write_json, write_jsonl


class ResultIndex:
    def __init__(self, run_root: Path, *, objective: str = "quality_win") -> None:
        self.run_root = Path(run_root)
        self.path = self.run_root / "index.jsonl"
        self.objective = objective
        self.malformed_lines = 0

    def append(self, result: ExperimentResult) -> None:
        payload = _relativize_result(result.to_dict(), self.run_root)
        write_jsonl(self.path, (payload,), append=True)

    def append_many(self, results: Iterable[ExperimentResult]) -> None:
        rows = [
            _relativize_result(result.to_dict(), self.run_root)
            for result in results
        ]
        write_jsonl(self.path, rows, append=True)

    def load(self) -> list[ExperimentResult]:
        results, malformed = load_index(self.path, run_root=self.run_root)
        self.malformed_lines = malformed
        return results

    def write_leaderboards(self, results: Iterable[ExperimentResult]) -> None:
        write_leaderboard_json(
            results,
            self.run_root / "leaderboard.json",
            objective=self.objective,
        )
        write_leaderboard_md(
            results,
            self.run_root / "leaderboard.md",
            objective=self.objective,
        )


def load_index(path: Path, *, run_root: Path | None = None) -> tuple[list[ExperimentResult], int]:
    path = Path(path)
    results: list[ExperimentResult] = []
    malformed = 0
    if not path.exists():
        return results, malformed
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
            if run_root is not None:
                payload = _absolutize_result(payload, Path(run_root))
            results.append(ExperimentResult.from_dict(payload))
        except Exception:
            malformed += 1
    return results, malformed


def append_global_index(result: ExperimentResult, global_path: Path, *, run_root: Path) -> None:
    append_global_index_many((result,), global_path, run_root=run_root)


def append_global_index_many(
    results: Iterable[ExperimentResult],
    global_path: Path,
    *,
    run_root: Path,
) -> None:
    rows = [
        _global_index_payload(result, run_root=run_root)
        for result in results
    ]
    write_jsonl(global_path, rows, append=True)


def _global_index_payload(result: ExperimentResult, *, run_root: Path) -> dict[str, object]:
    payload = result.to_dict()
    payload["run_root"] = Path(run_root).as_posix()
    return payload


def query_global_index(
    global_path: Path,
    *,
    suite: str | None = None,
    track: str | None = None,
    mode: str | None = None,
) -> list[Mapping[str, object]]:
    if not Path(global_path).exists():
        return []
    rows = []
    for line in Path(global_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if suite is not None and row.get("suite") != suite:
            continue
        if track is not None and row.get("track") != track:
            continue
        if mode is not None and row.get("mode") != mode:
            continue
        rows.append(row)
    return rows


def write_leaderboard_json(
    results: Iterable[ExperimentResult],
    path: Path,
    *,
    objective: str,
) -> Path:
    scored = rank_results(results, objective=objective)
    rows = [
        _leaderboard_row(index + 1, result, score)
        for index, (result, score) in enumerate(scored)
    ]
    duplicate_groups = _annotate_duplicate_quality_groups(rows)
    moonshot_summary = _moonshot_leaderboard_summary(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    payload = {
        "schema_version": "refinement_leaderboard_v1",
        "objective": objective,
        "counts": counts,
        "duplicate_quality_group_count": len(duplicate_groups),
        "duplicate_quality_duplicate_row_count": sum(
            int(group["count"]) for group in duplicate_groups
        ),
        "duplicate_quality_groups": duplicate_groups,
        "moonshot_summary": moonshot_summary,
        "top_by_objective": rows[0] if rows else None,
        "top_by_min_iou": max(
            rows,
            key=lambda row: float(row.get("min_iou") or 0.0),
            default=None,
        ),
        "fastest_acceptable": _fastest_acceptable(rows),
        "rows": rows,
    }
    write_json(path, payload)
    return path


def write_leaderboard_md(
    results: Iterable[ExperimentResult],
    path: Path,
    *,
    objective: str,
) -> Path:
    scored = rank_results(results, objective=objective)
    lines = [
        "# Refinement Leaderboard",
        "",
        f"Objective: `{objective}`",
        "",
        "| Rank | Case | Variant | Mode | Status | Promotion | Score | Avg IoU | Min IoU | Front | Side | Top | Topology | Editability | Moonshots | Portfolio | Elapsed | Autopsy | Result |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    rows: list[dict[str, object]] = []
    for index, (result, score) in enumerate(scored, start=1):
        row = _leaderboard_row(index, result, score)
        rows.append(row)
    duplicate_groups = _annotate_duplicate_quality_groups(rows)
    for row in rows:
        lines.append(
            "| {rank} | {case_id} | {variant_id} | {mode} | {status} | {promotion_tier} | {score:.3f} | "
            "{average_iou:.3f} | {min_iou:.3f} | {front_iou:.3f} | {side_iou:.3f} | "
            "{top_iou:.3f} | {topology_score:.3f} | {editability_score:.3f} | "
            "{moonshot_ran_count:.0f} | {moonshot_portfolio_kind} | {elapsed_s:.3f} | {autopsy_category} | {result_json} |".format(
                **row
            )
        )
    if duplicate_groups:
        lines.extend(
            [
                "",
                "## Duplicate Quality Groups",
                "",
                "| Fingerprint | Count | Examples |",
                "|---|---:|---|",
            ]
        )
        for group in duplicate_groups[:20]:
            lines.append(
                "| {fingerprint} | {count} | {examples} |".format(
                    fingerprint=group["fingerprint"],
                    count=group["count"],
                    examples=", ".join(str(item) for item in group["examples"]),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _leaderboard_row(
    rank: int,
    result: ExperimentResult,
    score: Mapping[str, object] | None = None,
) -> dict[str, object]:
    score = score or score_result(result)
    autopsy = result.autopsy if isinstance(result.autopsy, Mapping) else {}
    promotion = promotion_decision(result)
    return {
        "rank": rank,
        "case_id": result.case_id,
        "variant_id": result.variant_id,
        "mode": result.mode,
        "status": result.status,
        "promotion_tier": promotion.tier,
        "promotable": promotion.promotable,
        "parent_selectable": promotion.parent_selectable,
        "promotion_blockers": list(promotion.blockers),
        "parent_selection_blockers": list(promotion.blocking_for_parent_selection),
        "backend_status": promotion.backend_status,
        "cache_hit": bool(get_metric_path(result.metrics, "cache.hit")),
        "cache_source": str(get_metric_path(result.metrics, "cache.source") or ""),
        "quality_fingerprint": _quality_fingerprint(result, promotion),
        "moonshot_status_counts": _moonshot_status_counts(result),
        "moonshot_ran_count": float(_moonshot_status_counts(result).get("ran", 0)),
        "moonshot_error_count": float(_moonshot_status_counts(result).get("error", 0)),
        "moonshot_top_delta": _moonshot_top_delta(result),
        "moonshot_active_view": _moonshot_active_view(result),
        "moonshot_active_sequence": _moonshot_active_sequence(result),
        "moonshot_portfolio_action": _moonshot_portfolio_action(result),
        "moonshot_portfolio_available_actions": _moonshot_portfolio_available_actions(result),
        "moonshot_portfolio_rejected_actions": _moonshot_portfolio_rejected_actions(result),
        "moonshot_portfolio_dependency_edges": _moonshot_portfolio_dependency_edges(result),
        "moonshot_portfolio_execution_plan": _moonshot_portfolio_execution_plan(result),
        "moonshot_portfolio_risks": _moonshot_portfolio_risks(result),
        "moonshot_portfolio_kind": _moonshot_portfolio_kind(result),
        "score": float(score.get("total", 0.0)),
        "average_iou": result.avg_iou,
        "min_iou": result.min_iou,
        "front_iou": result.view_iou("front") or 0.0,
        "side_iou": result.view_iou("side") or 0.0,
        "top_iou": result.view_iou("top") or 0.0,
        "topology_score": _metric(result, "topology_score"),
        "editability_score": _metric(result, "editability_score"),
        "elapsed_s": result.elapsed_s,
        "autopsy_category": autopsy.get("category", ""),
        "result_json": result.result_json.as_posix() if result.result_json else "",
        "score_terms": json_safe(score.get("terms", ())),
    }


def _annotate_duplicate_quality_groups(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        fingerprint = str(row.get("quality_fingerprint", ""))
        if fingerprint:
            groups.setdefault(fingerprint, []).append(row)
    duplicate_groups = [
        {
            "fingerprint": fingerprint,
            "count": len(items),
            "examples": [
                f"{item.get('case_id')}:{item.get('variant_id')}"
                for item in items[:8]
            ],
            "modes": sorted({str(item.get("mode", "")) for item in items}),
            "statuses": sorted({str(item.get("status", "")) for item in items}),
        }
        for fingerprint, items in groups.items()
        if len(items) > 1
    ]
    duplicate_groups.sort(
        key=lambda item: (int(item["count"]), str(item["fingerprint"])),
        reverse=True,
    )
    counts = {
        str(group["fingerprint"]): int(group["count"])
        for group in duplicate_groups
    }
    group_ids = {
        str(group["fingerprint"]): index + 1
        for index, group in enumerate(duplicate_groups)
    }
    for row in rows:
        fingerprint = str(row.get("quality_fingerprint", ""))
        count = counts.get(fingerprint, 1)
        row["duplicate_quality_count"] = count
        row["duplicate_quality_group_id"] = group_ids.get(fingerprint)
        row["duplicate_quality"] = count > 1
    return duplicate_groups


def _moonshot_leaderboard_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    top_deltas: list[Mapping[str, object]] = []
    active_views: list[Mapping[str, object]] = []
    active_sequences: list[Mapping[str, object]] = []
    portfolio_available_actions: list[Mapping[str, object]] = []
    portfolio_actions: list[Mapping[str, object]] = []
    portfolio_rejected_actions: list[Mapping[str, object]] = []
    portfolio_dependency_edges: list[Mapping[str, object]] = []
    portfolio_execution_plan: list[Mapping[str, object]] = []
    portfolio_risks: list[Mapping[str, object]] = []
    for row in rows:
        status_counts = row.get("moonshot_status_counts")
        if isinstance(status_counts, Mapping):
            for status, count in status_counts.items():
                counts[str(status)] = counts.get(str(status), 0) + int(count or 0)
        top_delta = row.get("moonshot_top_delta")
        if isinstance(top_delta, Mapping) and top_delta:
            top_deltas.append(
                {
                    **dict(top_delta),
                    "case_id": row.get("case_id", ""),
                    "variant_id": row.get("variant_id", ""),
                }
            )
        active_view = row.get("moonshot_active_view")
        if isinstance(active_view, Mapping) and active_view:
            active_views.append(
                {
                    **dict(active_view),
                    "case_id": row.get("case_id", ""),
                    "variant_id": row.get("variant_id", ""),
                }
            )
        active_sequence = row.get("moonshot_active_sequence")
        if isinstance(active_sequence, Sequence) and not isinstance(active_sequence, (str, bytes)):
            for item in active_sequence:
                if isinstance(item, Mapping):
                    active_sequences.append(
                        {
                            **dict(item),
                            "case_id": row.get("case_id", ""),
                            "variant_id": row.get("variant_id", ""),
                        }
                    )
        portfolio_action = row.get("moonshot_portfolio_action")
        if isinstance(portfolio_action, Mapping) and portfolio_action:
            portfolio_actions.append(
                {
                    **dict(portfolio_action),
                    "case_id": row.get("case_id", ""),
                    "variant_id": row.get("variant_id", ""),
                }
            )
        for key, target in (
            ("moonshot_portfolio_available_actions", portfolio_available_actions),
            ("moonshot_portfolio_rejected_actions", portfolio_rejected_actions),
        ):
            items = row.get(key)
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                for item in items:
                    if isinstance(item, Mapping):
                        target.append(
                            {
                                **dict(item),
                                "case_id": row.get("case_id", ""),
                                "variant_id": row.get("variant_id", ""),
                            }
                        )
        for key, target in (
            ("moonshot_portfolio_dependency_edges", portfolio_dependency_edges),
            ("moonshot_portfolio_execution_plan", portfolio_execution_plan),
            ("moonshot_portfolio_risks", portfolio_risks),
        ):
            items = row.get(key)
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                for item in items:
                    if isinstance(item, Mapping):
                        target.append(
                            {
                                **dict(item),
                                "case_id": row.get("case_id", ""),
                                "variant_id": row.get("variant_id", ""),
                            }
                        )
    top_deltas.sort(
        key=lambda item: abs(float(item.get("delta", item.get("expected_metric_delta", 0.0)) or 0.0)),
        reverse=True,
    )
    active_views.sort(
        key=lambda item: float(item.get("expected_metric_delta", item.get("score", 0.0)) or 0.0),
        reverse=True,
    )
    portfolio_actions.sort(
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )
    portfolio_available_actions.sort(
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )
    portfolio_rejected_actions.sort(
        key=lambda item: (
            str(item.get("rejection", "")),
            -float(item.get("score", 0.0) or 0.0),
        )
    )
    active_sequences.sort(key=lambda item: int(item.get("order", 999) or 999))
    portfolio_risks.sort(
        key=lambda item: float(item.get("risk", 0.0) or 0.0),
        reverse=True,
    )
    return {
        "status_counts": dict(sorted(counts.items())),
        "ran_count": int(counts.get("ran", 0)),
        "error_count": int(counts.get("error", 0)),
        "top_candidate_deltas": top_deltas[:8],
        "active_view_suggestions": active_views[:8],
        "active_view_sequence": active_sequences[:8],
        "portfolio_available_actions": portfolio_available_actions[:12],
        "portfolio_actions": portfolio_actions[:8],
        "portfolio_rejected_actions": portfolio_rejected_actions[:12],
        "portfolio_dependency_edges": portfolio_dependency_edges[:12],
        "portfolio_execution_plan": portfolio_execution_plan[:8],
        "portfolio_risks": portfolio_risks[:8],
    }


def _quality_fingerprint(result: ExperimentResult, promotion: object) -> str:
    payload = {
        "schema": "refinement_quality_fingerprint_v1",
        "mode": result.mode,
        "result_status": result.status,
        "backend_status": getattr(promotion, "backend_status", ""),
        "promotion_tier": getattr(promotion, "tier", ""),
        "render": {
            view: {
                "area_iou": _rounded(
                    get_metric_path(
                        result.metrics,
                        f"render.per_view.{view}.area_iou",
                    )
                ),
                "boundary_iou": _rounded(
                    get_metric_path(
                        result.metrics,
                        f"render.per_view.{view}.boundary_iou",
                    )
                ),
                "signed_distance_loss": _rounded(
                    get_metric_path(
                        result.metrics,
                        f"render.per_view.{view}.signed_distance_loss",
                    )
                ),
            }
            for view in ("front", "side", "top")
        },
        "aggregates": {
            "average_iou": _rounded(result.avg_iou),
            "min_view_iou": _rounded(result.min_iou),
            "boundary_iou_mean": _rounded(
                get_metric_path(result.metrics, "render.boundary_iou_mean")
            ),
            "signed_distance_loss_mean": _rounded(
                get_metric_path(result.metrics, "render.signed_distance_loss_mean")
            ),
            "topology_score": _rounded(_metric(result, "topology_score")),
            "editability_score": _rounded(_metric(result, "editability_score")),
        },
        "backend": {
            "selected": _selected_backend_name(result),
            "area_iou_min": _rounded(
                get_metric_path(result.metrics, "backend.area_iou_min")
            ),
            "area_iou_mean": _rounded(
                get_metric_path(result.metrics, "backend.area_iou_mean")
            ),
            "boundary_iou_mean": _rounded(
                get_metric_path(result.metrics, "backend.boundary_iou_mean")
            ),
        },
        "blockers": tuple(sorted(getattr(promotion, "blockers", ()))),
    }
    return stable_hash(payload, length=16)


def _rounded(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return round(parsed, 6)


def _selected_backend_name(result: ExperimentResult) -> str:
    backend = result.backend_result if isinstance(result.backend_result, Mapping) else {}
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    source = selected if isinstance(selected, Mapping) else backend
    if not isinstance(source, Mapping):
        return ""
    for key in ("backend_name", "name", "mode", "selected_backend"):
        value = source.get(key)
        if value:
            return str(value)
    return ""


def _fastest_acceptable(rows: list[Mapping[str, object]]) -> Mapping[str, object] | None:
    acceptable = [
        row
        for row in rows
        if row.get("status") == "pass"
        and bool(row.get("promotable"))
        and (float(row.get("min_iou") or 0.0) >= 0.7 or float(row.get("average_iou") or 0.0) >= 0.85)
    ]
    return min(acceptable, key=lambda row: float(row.get("elapsed_s") or 0.0), default=None)


def _moonshot_payload(result: ExperimentResult) -> Mapping[str, object]:
    payload = get_metric_path(result.metrics, "moonshots")
    if not isinstance(payload, Mapping):
        payload = result.metrics.get("moonshots")
    if isinstance(payload, Mapping):
        return payload
    backend = result.backend_result if isinstance(result.backend_result, Mapping) else {}
    payload = backend.get("moonshot_evidence")
    return payload if isinstance(payload, Mapping) else {}


def _moonshot_status_counts(result: ExperimentResult) -> dict[str, int]:
    payload = _moonshot_payload(result)
    counts = payload.get("status_counts")
    if not isinstance(counts, Mapping):
        return {}
    output: dict[str, int] = {}
    for key, value in counts.items():
        try:
            output[str(key)] = int(value or 0)
        except (TypeError, ValueError):
            output[str(key)] = 0
    return output


def _moonshot_top_delta(result: ExperimentResult) -> Mapping[str, object]:
    payload = _moonshot_payload(result)
    deltas = payload.get("top_candidate_deltas")
    if isinstance(deltas, list) and deltas:
        first = deltas[0]
        return first if isinstance(first, Mapping) else {}
    return {}


def _moonshot_active_view(result: ExperimentResult) -> Mapping[str, object]:
    payload = _moonshot_payload(result)
    suggestions = payload.get("active_view_suggestions")
    if isinstance(suggestions, list) and suggestions:
        first = suggestions[0]
        return first if isinstance(first, Mapping) else {}
    return {}


def _moonshot_active_sequence(result: ExperimentResult) -> list[Mapping[str, object]]:
    payload = _moonshot_payload(result)
    sequence = payload.get("active_view_sequence")
    if not isinstance(sequence, list):
        return []
    return [item for item in sequence if isinstance(item, Mapping)][:8]


def _moonshot_portfolio_action(result: ExperimentResult) -> Mapping[str, object]:
    payload = _moonshot_payload(result)
    actions = payload.get("portfolio_actions")
    if isinstance(actions, list) and actions:
        first = actions[0]
        return first if isinstance(first, Mapping) else {}
    return {}


def _moonshot_portfolio_available_actions(result: ExperimentResult) -> list[Mapping[str, object]]:
    payload = _moonshot_payload(result)
    actions = payload.get("portfolio_available_actions")
    if not isinstance(actions, list):
        return []
    return [item for item in actions if isinstance(item, Mapping)][:12]


def _moonshot_portfolio_rejected_actions(result: ExperimentResult) -> list[Mapping[str, object]]:
    payload = _moonshot_payload(result)
    actions = payload.get("portfolio_rejected_actions")
    if not isinstance(actions, list):
        return []
    return [item for item in actions if isinstance(item, Mapping)][:12]


def _moonshot_portfolio_dependency_edges(result: ExperimentResult) -> list[Mapping[str, object]]:
    payload = _moonshot_payload(result)
    edges = payload.get("portfolio_dependency_edges")
    if not isinstance(edges, list):
        return []
    return [item for item in edges if isinstance(item, Mapping)][:12]


def _moonshot_portfolio_execution_plan(result: ExperimentResult) -> list[Mapping[str, object]]:
    payload = _moonshot_payload(result)
    plan = payload.get("portfolio_execution_plan")
    if not isinstance(plan, list):
        return []
    return [item for item in plan if isinstance(item, Mapping)][:8]


def _moonshot_portfolio_risks(result: ExperimentResult) -> list[Mapping[str, object]]:
    payload = _moonshot_payload(result)
    risks = payload.get("portfolio_risks")
    if not isinstance(risks, list):
        return []
    return [item for item in risks if isinstance(item, Mapping)][:8]


def _moonshot_portfolio_kind(result: ExperimentResult) -> str:
    action = _moonshot_portfolio_action(result)
    if not action:
        return ""
    return str(action.get("kind", ""))


def _metric(result: ExperimentResult, key: str) -> float:
    aliases = {
        "topology_score": "topology.score",
        "editability_score": "editability.qa_score",
    }
    value = result.metrics.get(key)
    if value is None and key in aliases:
        value = get_metric_path(result.metrics, aliases[key])
    if value is None and isinstance(result.backend_result, Mapping):
        selected = result.backend_result.get("selected")
        source = selected if isinstance(selected, Mapping) else result.backend_result
        metric = source.get("metric_result", {}) if isinstance(source, Mapping) else {}
        if isinstance(metric, Mapping):
            value = metric.get(key)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _relativize_result(payload: Mapping[str, object], run_root: Path) -> dict[str, object]:
    return _convert_paths(payload, run_root, make_relative=True)


def _absolutize_result(payload: Mapping[str, object], run_root: Path) -> dict[str, object]:
    return _convert_paths(payload, run_root, make_relative=False)


def _convert_paths(payload: Mapping[str, object], run_root: Path, *, make_relative: bool) -> dict[str, object]:
    path_keys = {"result_json"}
    path_map_keys = {"render_paths", "reference_paths", "artifacts"}
    output = dict(payload)
    for key in path_keys:
        value = output.get(key)
        if isinstance(value, str) and value:
            output[key] = _convert_path_string(value, run_root, make_relative)
    for key in path_map_keys:
        mapping = output.get(key)
        if isinstance(mapping, Mapping):
            output[key] = {
                name: _convert_path_string(str(value), run_root, make_relative)
                for name, value in mapping.items()
            }
    return output


def _convert_path_string(value: str, run_root: Path, make_relative: bool) -> str:
    path = Path(value)
    if make_relative:
        try:
            return path.resolve(strict=False).relative_to(run_root.resolve(strict=False)).as_posix()
        except ValueError:
            return path.as_posix()
    if path.is_absolute():
        return path.as_posix()
    return (run_root / path).as_posix()
