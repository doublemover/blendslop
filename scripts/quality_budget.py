"""Quality and performance budget comparison for JSON artifacts.

This is intentionally repository-harness code: it reads benchmark/e2e JSON
outputs and evaluates declarative thresholds without importing Blender APIs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = "quality_perf_budget_v1"
THRESHOLD_MODES = {"min", "max", "equal"}
COMPARISON_MODES = {
    "max_percent_increase",
    "max_percent_decrease",
    "max_absolute_increase",
    "max_absolute_decrease",
}
OPERATOR_MODES = THRESHOLD_MODES | COMPARISON_MODES


@dataclass(frozen=True)
class BudgetRecord:
    artifact: str
    case: str
    name: str
    mode: str = ""
    shape_id: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (self.artifact, self.case, self.name, self.mode, self.shape_id)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_budget_files(
    *,
    current_path: str | Path,
    budget_path: str | Path,
    baseline_path: str | Path | None = None,
) -> dict[str, Any]:
    current = read_json(current_path)
    budget = read_json(budget_path)
    baseline = read_json(baseline_path) if baseline_path else None
    return evaluate_budget_payloads(
        current,
        budget,
        baseline_payload=baseline,
        artifact_path=str(current_path),
        baseline_path=str(baseline_path) if baseline_path else None,
    )


def evaluate_budget_payloads(
    current_payload: Mapping[str, Any],
    budget_payload: Mapping[str, Any],
    *,
    baseline_payload: Optional[Mapping[str, Any]] = None,
    artifact_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
) -> dict[str, Any]:
    thresholds = tuple(budget_payload.get("thresholds", ()))
    comparisons = tuple(budget_payload.get("comparisons", ()))
    current_records = tuple(_records_from_payload(current_payload, artifact_path))
    baseline_records = tuple(_records_from_payload(baseline_payload, baseline_path))
    baseline_by_identity = {record.identity: record for record in baseline_records}

    threshold_checks = []
    for threshold in thresholds:
        threshold_checks.extend(_evaluate_threshold(threshold, current_records))
    threshold_checks = _prioritize_checks(threshold_checks)

    comparison_checks = []
    if comparisons:
        for comparison in comparisons:
            comparison_checks.extend(
                _evaluate_comparison(
                    comparison,
                    current_records,
                    baseline_by_identity,
                    has_baseline=baseline_payload is not None,
                )
            )
    comparison_checks = _prioritize_checks(comparison_checks)

    threshold_passed = all(
        check["passed"] or not check.get("required", True) for check in threshold_checks
    )
    comparison_passed = all(
        check["passed"] or not check.get("required", True)
        for check in comparison_checks
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "budget": {
            "name": budget_payload.get("name", ""),
            "schema_version": budget_payload.get("schema_version", ""),
        },
        "artifact_path": artifact_path,
        "baseline_path": baseline_path,
        "record_count": len(current_records),
        "threshold_passed": threshold_passed,
        "comparison_passed": comparison_passed,
        "passed": threshold_passed and comparison_passed,
        "checks": threshold_checks,
        "comparison_checks": comparison_checks,
    }


def _evaluate_threshold(
    threshold: Mapping[str, Any],
    records: Sequence[BudgetRecord],
) -> list[dict[str, Any]]:
    required = bool(threshold.get("required", True))
    matches = [record for record in records if _selector_matches(threshold, record)]
    if not matches:
        if bool(threshold.get("allow_missing", False)):
            return [_base_check(threshold, None, required, True, "no matching record")]
        if _selector_artifact_absent(threshold, records):
            return [
                _base_check(threshold, None, required, True, "artifact not present")
            ]
        return [_base_check(threshold, None, required, False, "no matching record")]

    checks = []
    for record in matches:
        metric = str(threshold["metric"])
        value = _coerce_number(_lookup_metric(record.data, metric))
        operator_mode = _threshold_operator_mode(threshold)
        passed, message = _compare_value(
            value,
            float(threshold["threshold"]),
            operator_mode,
        )
        check = {
                **_record_selector(record),
                "id": threshold.get("id", ""),
                "metric": metric,
                "value": value,
                "threshold": float(threshold["threshold"]),
                "mode": operator_mode,
                "required": required,
                "passed": passed,
                "message": message,
            }
        check["category"] = _check_category(check, record.data)
        check["impact_rank"] = _impact_rank(check)
        checks.append(check)
    return checks


def _evaluate_comparison(
    comparison: Mapping[str, Any],
    current_records: Sequence[BudgetRecord],
    baseline_by_identity: Mapping[tuple[str, str, str, str, str], BudgetRecord],
    *,
    has_baseline: bool,
) -> list[dict[str, Any]]:
    required = bool(comparison.get("required", False))
    matches = [
        record for record in current_records if _selector_matches(comparison, record)
    ]
    if not has_baseline:
        return [
            _base_check(
                comparison, None, required, not required, "baseline not provided"
            )
        ]
    if not matches:
        if _selector_artifact_absent(comparison, current_records):
            return [
                _base_check(comparison, None, required, True, "artifact not present")
            ]
        return [_base_check(comparison, None, required, False, "no matching record")]

    checks = []
    for current in matches:
        baseline = baseline_by_identity.get(current.identity)
        metric = str(comparison["metric"])
        if baseline is None:
            checks.append(
                {
                    **_record_selector(current),
                    "id": comparison.get("id", ""),
                    "metric": metric,
                    "required": required,
                    "passed": not required,
                    "message": "no matching baseline record",
                }
            )
            continue
        before = _coerce_number(_lookup_metric(baseline.data, metric))
        after = _coerce_number(_lookup_metric(current.data, metric))
        operator_mode = _comparison_operator_mode(comparison)
        passed, delta, message = _compare_delta(
            before,
            after,
            float(comparison["threshold"]),
            operator_mode,
        )
        checks.append(
            {
                **_record_selector(current),
                "id": comparison.get("id", ""),
                "metric": metric,
                "baseline": before,
                "current": after,
                "delta": delta,
                "threshold": float(comparison["threshold"]),
                "mode": operator_mode,
                "required": required,
                "passed": passed,
                "message": message,
            }
        )
    return checks


def _records_from_payload(
    payload: Optional[Mapping[str, Any]],
    source_path: Optional[str],
) -> Iterable[BudgetRecord]:
    if not payload:
        return ()
    records: list[BudgetRecord] = []
    if isinstance(payload.get("results"), list):
        records.extend(_benchmark_records(payload, source_path))
    if isinstance(payload.get("matrix"), list):
        records.extend(_matrix_records(payload, source_path))
    if isinstance(payload.get("bundles"), list):
        records.extend(
            _evaluation_bundle_records(payload.get("bundles", ()), source_path)
        )
    if isinstance(payload.get("evaluation_bundles"), list):
        records.extend(
            _evaluation_bundle_records(
                payload.get("evaluation_bundles", ()), source_path
            )
        )
    if isinstance(payload.get("evaluation_bundle"), Mapping):
        records.append(
            _evaluation_bundle_record(payload["evaluation_bundle"], source_path)
        )
    backend = payload.get("backend_result")
    if isinstance(backend, Mapping):
        records.extend(_records_from_payload(backend, source_path))
    if _is_evaluation_bundle(payload):
        records.append(_evaluation_bundle_record(payload, source_path))
    if "validation_mode" in payload or "average_iou" in payload:
        records.append(
            BudgetRecord(
                artifact="e2e",
                case=str(payload.get("mode", "single")),
                name=str(payload.get("mode", "single")),
                mode=str(payload.get("mode", "")),
                data=dict(payload),
                source_path=source_path,
            )
        )
    if records:
        return tuple(_dedupe_records(records))
    return (
        BudgetRecord(
            artifact=str(payload.get("artifact", "json")),
            case=str(payload.get("case", "default")),
            name=str(payload.get("name", "payload")),
            data=dict(payload),
            source_path=source_path,
        ),
    )


def _dedupe_records(records: Sequence[BudgetRecord]) -> Iterable[BudgetRecord]:
    seen: set[tuple[tuple[str, str, str, str, str], str | None]] = set()
    for record in records:
        key = (record.identity, record.source_path)
        if key in seen:
            continue
        seen.add(key)
        yield record


def _evaluation_bundle_records(
    bundles: Iterable[Any],
    source_path: Optional[str],
) -> Iterable[BudgetRecord]:
    for item in bundles:
        if isinstance(item, Mapping) and _is_evaluation_bundle(item):
            yield _evaluation_bundle_record(item, source_path)


def _evaluation_bundle_record(
    payload: Mapping[str, Any],
    source_path: Optional[str],
) -> BudgetRecord:
    data = dict(payload)
    data["metrics"] = _bundle_metric_payload(payload)
    return BudgetRecord(
        artifact="evaluation",
        case=str(payload.get("suite", "default")),
        name=str(payload.get("candidate_id", payload.get("name", "bundle"))),
        mode=str(payload.get("mode", "")),
        shape_id=str(payload.get("target_id", "")),
        data=data,
        source_path=source_path,
    )


def _is_evaluation_bundle(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("schema_version", "")).startswith("evaluation-bundle") or (
        isinstance(payload.get("metric_groups"), list)
        and "candidate_id" in payload
        and "mode" in payload
    )


def _bundle_metric_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for group in payload.get("metric_groups", ()) or ():
        if not isinstance(group, Mapping):
            continue
        for metric in group.get("metrics", ()) or ():
            if not isinstance(metric, Mapping):
                continue
            name = str(metric.get("name", ""))
            if not name:
                continue
            _set_dotted_metric(metrics, name, metric.get("value"))
    return metrics


def _set_dotted_metric(target: dict[str, Any], name: str, value: Any) -> None:
    current = target
    parts = [part for part in name.split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _benchmark_records(
    payload: Mapping[str, Any],
    source_path: Optional[str],
) -> Iterable[BudgetRecord]:
    for item in payload.get("results", ()):
        if not isinstance(item, Mapping):
            continue
        yield BudgetRecord(
            artifact="benchmark",
            case=str(item.get("case", "ad-hoc")),
            name=str(item.get("name", "")),
            data=dict(item),
            source_path=source_path,
        )


def _matrix_records(
    payload: Mapping[str, Any],
    source_path: Optional[str],
) -> Iterable[BudgetRecord]:
    for item in payload.get("matrix", ()):
        if not isinstance(item, Mapping):
            continue
        mode = str(item.get("mode", ""))
        shape_id = str(item.get("shape_id", ""))
        yield BudgetRecord(
            artifact="e2e",
            case=str(item.get("case", item.get("suite", "synthetic"))),
            name=str(item.get("name", f"{shape_id}/{mode}".strip("/"))),
            mode=mode,
            shape_id=shape_id,
            data=dict(item),
            source_path=source_path,
        )


def _selector_matches(selector: Mapping[str, Any], record: BudgetRecord) -> bool:
    for key, actual in _record_selector(record).items():
        expected = selector.get(key)
        if expected is None or expected == "*":
            continue
        if key == "mode" and str(expected) in OPERATOR_MODES:
            continue
        if str(expected) != str(actual):
            return False
    return True


def _threshold_operator_mode(threshold: Mapping[str, Any]) -> str:
    explicit = threshold.get("threshold_mode", threshold.get("operator"))
    if explicit is not None:
        return str(explicit)
    raw_mode = str(threshold.get("mode", "min"))
    if raw_mode in THRESHOLD_MODES:
        return raw_mode
    return "min"


def _comparison_operator_mode(comparison: Mapping[str, Any]) -> str:
    explicit = comparison.get("comparison_mode", comparison.get("operator"))
    if explicit is not None:
        return str(explicit)
    raw_mode = str(comparison.get("mode", "max_percent_increase"))
    if raw_mode in COMPARISON_MODES:
        return raw_mode
    return "max_percent_increase"


def _selector_artifact_absent(
    selector: Mapping[str, Any],
    records: Sequence[BudgetRecord],
) -> bool:
    expected = selector.get("artifact")
    if expected is None or expected == "*":
        return False
    return all(record.artifact != str(expected) for record in records)


def _record_selector(record: BudgetRecord) -> dict[str, str]:
    return {
        "artifact": record.artifact,
        "case": record.case,
        "name": record.name,
        "mode": record.mode,
        "shape_id": record.shape_id,
    }


def _lookup_metric(data: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _compare_value(
    value: Optional[float],
    threshold: float,
    mode: str,
) -> tuple[bool, str]:
    if value is None:
        return False, "missing metric"
    if mode == "min":
        return value >= threshold, "" if value >= threshold else "below minimum"
    if mode == "max":
        return value <= threshold, "" if value <= threshold else "above maximum"
    if mode == "equal":
        return value == threshold, "" if value == threshold else "not equal"
    raise ValueError(f"unknown threshold mode {mode!r}")


def _compare_delta(
    before: Optional[float],
    after: Optional[float],
    threshold: float,
    mode: str,
) -> tuple[bool, Optional[float], str]:
    if before is None or after is None:
        return False, None, "missing baseline or current metric"
    if mode == "max_percent_increase":
        delta = _percent_delta(before, after)
        return (
            delta <= threshold,
            delta,
            "" if delta <= threshold else "percent increase regression",
        )
    if mode == "max_percent_decrease":
        delta = ((before - after) / max(abs(before), 1e-12)) * 100.0
        return (
            delta <= threshold,
            delta,
            "" if delta <= threshold else "percent decrease regression",
        )
    if mode == "max_absolute_increase":
        delta = after - before
        return (
            delta <= threshold,
            delta,
            "" if delta <= threshold else "absolute increase regression",
        )
    if mode == "max_absolute_decrease":
        delta = before - after
        return (
            delta <= threshold,
            delta,
            "" if delta <= threshold else "absolute decrease regression",
        )
    raise ValueError(f"unknown comparison mode {mode!r}")


def _percent_delta(before: float, after: float) -> float:
    denominator = max(abs(before), 1e-12)
    return ((after - before) / denominator) * 100.0


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _base_check(
    selector: Mapping[str, Any],
    value: Optional[float],
    required: bool,
    passed: bool,
    message: str,
) -> dict[str, Any]:
    check = {
        "id": selector.get("id", ""),
        "artifact": str(selector.get("artifact", "*")),
        "case": str(selector.get("case", "*")),
        "name": str(selector.get("name", "*")),
        "mode": str(selector.get("mode", "*")),
        "shape_id": str(selector.get("shape_id", "*")),
        "metric": selector.get("metric", ""),
        "value": value,
        "required": required,
        "passed": passed,
        "message": message,
    }
    check["category"] = _check_category(check, {})
    check["impact_rank"] = _impact_rank(check)
    return check


def _prioritize_checks(checks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(check) for check in checks),
        key=lambda check: (
            bool(check.get("passed", False)),
            int(check.get("impact_rank", 99)),
            str(check.get("mode", "")),
            str(check.get("case", "")),
            str(check.get("metric", "")),
        ),
    )


def _check_category(check: Mapping[str, Any], record_data: Mapping[str, Any]) -> str:
    if check.get("passed"):
        return "passed"
    metric = str(check.get("metric", ""))
    message = str(check.get("message", ""))
    value = check.get("value")
    status = str(record_data.get("status", ""))
    row_passed = bool(record_data.get("passed", False))
    if row_passed and status == "pass" and metric and value is not None:
        return "contract_mismatch"
    if value is None or "missing" in message:
        if _is_optional_or_research_missing(check):
            return "research_optional_missing"
        return "coverage_missing"
    if "topology" in metric or "editability" in metric:
        return "topology_editability_failure"
    return "quality_below_floor"


def _is_optional_or_research_missing(check: Mapping[str, Any]) -> bool:
    if not bool(check.get("required", True)):
        return True
    metric = str(check.get("metric", ""))
    return any(part in metric for part in ("lpips", "geometry.recoverable"))


def _impact_rank(check: Mapping[str, Any]) -> int:
    category = str(check.get("category", ""))
    metric = str(check.get("metric", ""))
    if category == "contract_mismatch":
        return 0
    if "per_view" in metric or "min_view_iou" in metric:
        return 1
    if category == "topology_editability_failure":
        return 2
    if "geometry" in metric:
        return 3
    if category == "coverage_missing":
        return 4
    if category == "research_optional_missing":
        return 5
    if category == "quality_below_floor":
        return 1
    return 9


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate quality/perf budget JSON.")
    parser.add_argument(
        "--current", required=True, help="Current benchmark/e2e JSON artifact."
    )
    parser.add_argument("--budget", required=True, help="Budget JSON file.")
    parser.add_argument(
        "--baseline", default=None, help="Optional baseline JSON artifact."
    )
    parser.add_argument("--report", default=None, help="Write budget report JSON.")
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Always exit 0 after writing/printing the report.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = evaluate_budget_files(
        current_path=args.current,
        budget_path=args.budget,
        baseline_path=args.baseline,
    )
    if args.report:
        write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.warn_only:
        return 0
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
