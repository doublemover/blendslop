"""Append-only result indexes and leaderboards for refinement runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .contracts import ExperimentResult, json_safe
from .parameter_search import promotion_decision, rank_results, score_result


class ResultIndex:
    def __init__(self, run_root: Path, *, objective: str = "quality_win") -> None:
        self.run_root = Path(run_root)
        self.path = self.run_root / "index.jsonl"
        self.objective = objective
        self.malformed_lines = 0

    def append(self, result: ExperimentResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _relativize_result(result.to_dict(), self.run_root)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            handle.flush()

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
    global_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["run_root"] = Path(run_root).as_posix()
    with global_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


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
    rows = [_leaderboard_row(index + 1, result, score) for index, (result, score) in enumerate(scored)]
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    payload = {
        "schema_version": "refinement_leaderboard_v1",
        "objective": objective,
        "counts": counts,
        "top_by_objective": rows[0] if rows else None,
        "top_by_min_iou": max(rows, key=lambda row: float(row.get("min_iou") or 0.0), default=None),
        "fastest_acceptable": _fastest_acceptable(rows),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
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
        "| Rank | Case | Variant | Mode | Status | Promotion | Score | Avg IoU | Min IoU | Front | Side | Top | Topology | Editability | Elapsed | Autopsy | Result |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for index, (result, score) in enumerate(scored, start=1):
        row = _leaderboard_row(index, result, score)
        lines.append(
            "| {rank} | {case_id} | {variant_id} | {mode} | {status} | {promotion_tier} | {score:.3f} | "
            "{average_iou:.3f} | {min_iou:.3f} | {front_iou:.3f} | {side_iou:.3f} | "
            "{top_iou:.3f} | {topology_score:.3f} | {editability_score:.3f} | "
            "{elapsed_s:.3f} | {autopsy_category} | {result_json} |".format(
                **row
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
        "promotion_blockers": list(promotion.blockers),
        "backend_status": promotion.backend_status,
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


def _fastest_acceptable(rows: list[Mapping[str, object]]) -> Mapping[str, object] | None:
    acceptable = [
        row
        for row in rows
        if row.get("status") == "pass"
        and bool(row.get("promotable"))
        and (float(row.get("min_iou") or 0.0) >= 0.7 or float(row.get("average_iou") or 0.0) >= 0.85)
    ]
    return min(acceptable, key=lambda row: float(row.get("elapsed_s") or 0.0), default=None)


def _metric(result: ExperimentResult, key: str) -> float:
    value = result.metrics.get(key)
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
