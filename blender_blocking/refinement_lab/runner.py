"""Execution engine for refinement experiment plans."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
import copy
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .adaptive import merge_proposals, proposals_from_result_payload
from .artifact_report import ReportOptions, generate_report
from .bounds_debug import build_bounds_debug_report
from .candidate_autopsy import autopsy_candidate
from .contracts import (
    ExperimentCase,
    ExperimentPlan,
    ExperimentResult,
    ExperimentVariant,
    RefinementRunManifest,
    compact_path_segment,
    json_safe,
    stable_hash,
    utc_now,
)
from .parameter_search import promotion_decision, score_result
from .parameters import apply_variant_parameter_to_config
from .result_index import ResultIndex, append_global_index, append_global_index_many

try:
    from blender_blocking.config import BlockingConfig
    from blender_blocking.metrics.namespaces import (
        namespace_metric_key,
        set_metric_path,
        set_render_aggregate_metrics,
    )
    from blender_blocking.utils.json_io import load_json as _load_json_payload
    from blender_blocking.utils.json_io import write_json as _write_json_payload
    from blender_blocking.utils.optional_deps import dependency_report
except ImportError:  # pragma: no cover
    from config import BlockingConfig
    from metrics.namespaces import (
        namespace_metric_key,
        set_metric_path,
        set_render_aggregate_metrics,
    )
    from utils.json_io import load_json as _load_json_payload
    from utils.json_io import write_json as _write_json_payload
    from utils.optional_deps import dependency_report


@dataclass(frozen=True)
class RunOptions:
    html_report: bool = True
    write_overlays: bool = True
    write_bounds_debug: bool = True
    write_autopsy: bool = True
    copy_references: bool = False
    stop_on_first_error: bool = False
    fail_on_all_failed: bool = True
    append_global_index: bool = True
    report_failures: str = "top"
    write_lineage: bool = True
    write_adaptive_proposals: bool = True
    adaptive_max_proposals: int = 12
    subprocess_blender: bool = False
    blender_executable: str | None = None
    progress: bool = False
    cache_root: Path | None = None
    reference_cache: bool = False
    candidate_cache: bool = False
    resume_candidates: bool = False
    debug_artifact_policy: str = "all"
    batch_index_writes: bool = True
    moonshot_sidecars: bool = False
    moonshot_experiments: tuple[str, ...] = ()


class BaseRunner:
    def __init__(
        self,
        *,
        plan: ExperimentPlan,
        options: RunOptions = RunOptions(),
        base_config: BlockingConfig | None = None,
    ) -> None:
        self.plan = plan
        self.options = options
        self.base_config = base_config or BlockingConfig()
        self.run_root = Path(plan.output_root).resolve(strict=False)
        self.cache_root = (
            Path(options.cache_root).resolve(strict=False)
            if options.cache_root is not None
            else (self.run_root / ".cache").resolve(strict=False)
        )
        self.index = ResultIndex(self.run_root, objective=plan.objective)
        self.cache_stats: dict[str, Any] = {
            "reference_cache_hits": 0,
            "reference_cache_misses": 0,
            "reference_cache_writes": 0,
            "candidate_cache_hits": 0,
            "candidate_cache_misses": 0,
            "candidate_cache_writes": 0,
            "candidate_cache_local_resume_writes": 0,
            "candidate_cache_shared_writes": 0,
            "candidate_cache_sources": {},
            "candidate_cache_skipped_hits": {},
            "candidate_cache_no_hit_reasons": {},
            "candidate_cache_key_schema": "refinement_candidate_effective_state_v2",
        }

    def run(self) -> tuple[bool, list[ExperimentResult]]:
        self._prepare_run_root()
        manifest = self._write_manifest()
        results: list[ExperimentResult] = []
        index_buffer: list[ExperimentResult] = []
        total = len(self.plan.cases) * len(self.plan.variants)
        counter = 0
        for case in self.plan.cases:
            try:
                references = self._prepare_case_references(case)
            except Exception as exc:
                for variant in self.plan.variants:
                    counter += 1
                    print(
                        f"[{counter:03d}/{total:03d}] {variant.variant_id} ({variant.mode})"
                    )
                    result = self._reference_generation_error_result(
                        case,
                        variant,
                        exc,
                    )
                    result = self._postprocess_result(case, variant, result)
                    self._append_index_result(result, index_buffer)
                    results.append(result)
                    self._print_result_row(result)
                if self.options.stop_on_first_error:
                    break
                continue
            for variant in self.plan.variants:
                counter += 1
                print(
                    f"[{counter:03d}/{total:03d}] {variant.variant_id} ({variant.mode})"
                )
                candidate_started_utc = utc_now()
                candidate_started_monotonic = time.perf_counter()
                try:
                    result = self._load_reusable_candidate(case, variant, references)
                    if result is None:
                        result = self._run_one(case, variant, references)
                        result = self._postprocess_result(case, variant, result)
                        self._write_candidate_state(case, variant, references, result)
                    else:
                        print("  reused cached candidate result")
                except Exception as exc:
                    result = self._candidate_execution_error_result(
                        case,
                        variant,
                        references,
                        exc,
                        started_utc=candidate_started_utc,
                        elapsed_s=time.perf_counter() - candidate_started_monotonic,
                    )
                self._append_index_result(result, index_buffer)
                results.append(result)
                self._print_result_row(result)
                if self.options.stop_on_first_error and result.status == "error":
                    break
            if (
                self.options.stop_on_first_error
                and results
                and results[-1].status == "error"
            ):
                break
        self._flush_index_results(index_buffer)
        self.index.write_leaderboards(results)
        if self.options.append_global_index:
            global_path = self.run_root.parent / "global-index.jsonl"
            if self.options.batch_index_writes:
                append_global_index_many(results, global_path, run_root=self.run_root)
            else:
                for result in results:
                    append_global_index(result, global_path, run_root=self.run_root)
        if self.options.html_report:
            manifest = self._write_manifest(results=results)
            generate_report(
                run_root=self.run_root,
                results=results,
                manifest=manifest,
                objective=self.plan.objective,
                options=ReportOptions(
                    top_k=self.plan.top_k,
                    include_failures=self.options.report_failures,
                    write_overlays=self.options.write_overlays,
                ),
            )
        if self.options.write_adaptive_proposals:
            self._write_adaptive_outputs(results)
        if self.options.write_lineage:
            self._write_lineage_outputs(results)
        self._write_cache_stats()
        passed_any = any(result.status == "pass" for result in results)
        return (passed_any or not self.options.fail_on_all_failed), results

    def _append_index_result(
        self,
        result: ExperimentResult,
        buffer: list[ExperimentResult],
    ) -> None:
        if self.options.batch_index_writes:
            buffer.append(result)
        else:
            self.index.append(result)

    def _flush_index_results(self, buffer: list[ExperimentResult]) -> None:
        if not buffer:
            return
        self.index.append_many(buffer)
        buffer.clear()

    def _reference_generation_error_result(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        exc: Exception,
    ) -> ExperimentResult:
        started = utc_now()
        variant_dir = self._case_variant_dir(case, variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        result_json = variant_dir / "result.json"
        spec_payload = (
            case.metadata.get("spec", {}) if isinstance(case.metadata, Mapping) else {}
        )
        payload = {
            "schema_version": "refinement_reference_generation_error_v1",
            "status": "error",
            "failure_code": "reference_generation_failed",
            "case_id": case.case_id,
            "variant_id": variant.variant_id,
            "mode": variant.mode,
            "synthetic_family": spec_payload.get("family"),
            "synthetic_shape_id": spec_payload.get("shape_id"),
            "error": str(exc),
            "exception_type": type(exc).__name__,
        }
        result_json.write_text(
            json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ExperimentResult(
            run_id=self.plan.run_id,
            case_id=case.case_id,
            variant_id=variant.variant_id,
            mode=variant.mode,
            status="error",
            exit_code=2,
            started_utc=started,
            finished_utc=utc_now(),
            elapsed_s=0.0,
            result_json=result_json,
            reference_paths={},
            metrics={
                "validation_mode": "render-iou",
                "failure_code": "reference_generation_failed",
            },
            artifacts={"result": result_json},
            errors=(str(exc),),
        )

    def _candidate_execution_error_result(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
        exc: Exception,
        *,
        started_utc: str,
        elapsed_s: float,
    ) -> ExperimentResult:
        variant_dir = self._case_variant_dir(case, variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        result_json = variant_dir / "result.json"
        error = f"{type(exc).__name__}: {exc}"
        payload = {
            "schema_version": "refinement_candidate_execution_error_v1",
            "status": "error",
            "failure_code": "candidate_execution_failed",
            "case_id": case.case_id,
            "variant_id": variant.variant_id,
            "mode": variant.mode,
            "suite": self.plan.suite,
            "track": self.plan.track,
            "objective": self.plan.objective,
            "validation_mode": variant.validation_mode,
            "error": str(exc),
            "exception_type": type(exc).__name__,
        }
        result_json.write_text(
            json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ExperimentResult(
            run_id=self.plan.run_id,
            case_id=case.case_id,
            variant_id=variant.variant_id,
            mode=variant.mode,
            status="error",
            exit_code=2,
            started_utc=started_utc,
            finished_utc=utc_now(),
            elapsed_s=max(0.0, float(elapsed_s)),
            result_json=result_json,
            reference_paths=reference_paths,
            metrics={
                "validation_mode": variant.validation_mode,
                "failure_code": "candidate_execution_failed",
                "candidate": {
                    "failure_code": "candidate_execution_failed",
                    "exception_type": type(exc).__name__,
                },
            },
            artifacts={"result": result_json},
            errors=(error,),
        )

    def _run_one(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
    ) -> ExperimentResult:
        raise NotImplementedError

    def _prepare_run_root(self) -> None:
        for directory in (
            self.run_root,
            self.run_root / "c",
            self.run_root / "assets",
            self.run_root / "assets" / "o",
            self.run_root / "assets" / "d",
            self.run_root / "assets" / "t",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.plan.write(self.run_root / "plan.json")

    def _write_manifest(
        self,
        *,
        results: Sequence[ExperimentResult] = (),
    ) -> RefinementRunManifest:
        counts: dict[str, int] = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        manifest = RefinementRunManifest(
            run_id=self.plan.run_id,
            created_utc=utc_now(),
            suite=self.plan.suite,
            track=self.plan.track,
            search=self.plan.search,
            objective=self.plan.objective,
            output_root=self.run_root,
            git=_cached_git_info(),
            environment=_cached_environment_info(),
            dependency_report=_cached_dependency_report(),
            plan_path=self.run_root / "plan.json",
            index_path=self.run_root / "index.jsonl",
            html_report_path=self.run_root / "report.html",
            cases=self.plan.cases,
            variants=self.plan.variants,
            result_counts=counts,
        )
        manifest.write(self.run_root / "manifest.json")
        return manifest

    def _prepare_case_references(self, case: ExperimentCase) -> Mapping[str, Path]:
        if case.source != "synthetic":
            return case.reference_paths
        return self._generate_synthetic_references(case)

    def _generate_synthetic_references(
        self, case: ExperimentCase
    ) -> Mapping[str, Path]:
        try:
            from blender_blocking.synthetic.blender_builders import render_views
            from blender_blocking.synthetic.specs import SyntheticShapeSpec
        except ImportError:  # pragma: no cover
            from synthetic.blender_builders import render_views
            from synthetic.specs import SyntheticShapeSpec

        spec_payload = (
            case.metadata.get("spec", {}) if isinstance(case.metadata, Mapping) else {}
        )
        spec = SyntheticShapeSpec.from_dict(spec_payload)
        output = self._case_dir(case) / "ref"
        resolution = tuple(self.base_config.render_silhouette.resolution)
        if self.options.reference_cache:
            cache_dir = self._reference_cache_dir(case, resolution)
            if _reference_views_ready(cache_dir):
                self.cache_stats["reference_cache_hits"] += 1
            else:
                self.cache_stats["reference_cache_misses"] += 1
                cache_dir.mkdir(parents=True, exist_ok=True)
                render_views(
                    spec,
                    cache_dir,
                    resolution=resolution,
                    include_orbit=False,
                )
                self.cache_stats["reference_cache_writes"] += 1
                _require_reference_views(cache_dir, context=f"cache:{cache_dir}")
            rendered, copy_error = _copy_reference_views(cache_dir, output)
            if copy_error:
                raise RuntimeError(copy_error)
        else:
            rendered = render_views(
                spec,
                output,
                resolution=resolution,
                include_orbit=False,
            )
        references = {
            view: Path(rendered[view])
            for view in ("front", "side", "top")
            if view in rendered
        }
        _require_reference_mapping(references, context=f"case:{case.case_id}")
        return references

    def _reference_cache_dir(
        self,
        case: ExperimentCase,
        resolution: tuple[int, int],
    ) -> Path:
        spec_payload = (
            case.metadata.get("spec", {}) if isinstance(case.metadata, Mapping) else {}
        )
        key = stable_hash(
            {
                "schema": "synthetic_reference_cache_v1",
                "case_id": case.case_id,
                "suite": case.suite,
                "synthetic_definition": case.synthetic_definition,
                "spec": spec_payload,
                "resolution": resolution,
            },
            length=24,
        )
        return self.cache_root / "references" / key[:2] / key

    def _candidate_state_key(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
    ) -> str:
        return stable_hash(
            self._candidate_state_key_material(case, variant, reference_paths),
            length=32,
        )

    def _candidate_state_key_material(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
    ) -> Mapping[str, Any]:
        case_payload = dict(case.to_dict())
        case_payload.pop("reference_paths", None)
        return {
            "schema": "refinement_candidate_effective_state_v2",
            "suite": self.plan.suite,
            "track": self.plan.track,
            "objective": self.plan.objective,
            "case": case_payload,
            "variant": _candidate_cache_variant_payload(variant),
            "base_config": self.base_config.to_dict(),
            "references": _reference_hashes(reference_paths),
        }

    def _local_candidate_state_path(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
    ) -> Path:
        return self._case_variant_dir(case, variant) / "candidate-state.json"

    def _shared_candidate_state_path(
        self,
        key: str,
    ) -> Path:
        return self.cache_root / "candidates" / key[:2] / key / "candidate-state.json"

    def _load_reusable_candidate(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
    ) -> ExperimentResult | None:
        if not (self.options.resume_candidates or self.options.candidate_cache):
            return None
        key = self._candidate_state_key(case, variant, reference_paths)
        candidates: list[tuple[Path, str]] = []
        if self.options.resume_candidates:
            candidates.append((self._local_candidate_state_path(case, variant), "local_resume"))
        if self.options.candidate_cache:
            candidates.append((self._shared_candidate_state_path(key), "shared_cache"))
        for path, source in candidates:
            if not path.exists():
                self._record_candidate_cache_no_hit(f"{source}:missing_state")
                continue
            result = self._load_candidate_state(path, key)
            if result is not None:
                self._record_candidate_cache_hit(source)
                return self._adapt_reused_candidate_result(
                    result,
                    case,
                    variant,
                    source=source,
                )
        self.cache_stats["candidate_cache_misses"] += 1
        return None

    def _load_candidate_state(
        self,
        path: Path,
        key: str,
    ) -> ExperimentResult | None:
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._record_candidate_cache_skip("unreadable_state")
            return None
        if state.get("schema_version") != "refinement_candidate_state_v1":
            self._record_candidate_cache_skip("schema_mismatch")
            return None
        if state.get("state_key") != key:
            self._record_candidate_cache_skip("key_mismatch")
            return None
        result_payload = state.get("result")
        if not isinstance(result_payload, Mapping):
            self._record_candidate_cache_skip("missing_result")
            return None
        try:
            result = ExperimentResult.from_dict(
                {**dict(result_payload), "run_id": self.plan.run_id}
            )
        except Exception:
            self._record_candidate_cache_skip("invalid_result")
            return None
        if not _result_artifacts_ready(result):
            self._record_candidate_cache_skip("artifacts_missing")
            return None
        return result

    def _adapt_reused_candidate_result(
        self,
        result: ExperimentResult,
        case: ExperimentCase,
        variant: ExperimentVariant,
        *,
        source: str,
    ) -> ExperimentResult:
        metrics = copy.deepcopy(dict(result.metrics))
        set_metric_path(metrics, "variant.diagnostic_only", bool(variant.diagnostic_only))
        set_metric_path(metrics, "cache.hit", True)
        set_metric_path(metrics, "cache.source", source)
        warnings = tuple(result.warnings)
        if result.case_id != case.case_id or result.variant_id != variant.variant_id:
            warnings = warnings + (
                f"reused_candidate_result:{result.case_id}:{result.variant_id}",
            )
        return ExperimentResult.from_dict(
            {
                **result.to_dict(),
                "run_id": self.plan.run_id,
                "case_id": case.case_id,
                "variant_id": variant.variant_id,
                "mode": variant.mode,
                "metrics": metrics,
                "warnings": warnings,
            }
        )

    def _write_candidate_state(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
        result: ExperimentResult,
    ) -> None:
        if not (self.options.resume_candidates or self.options.candidate_cache):
            return
        key = self._candidate_state_key(case, variant, reference_paths)
        payload = {
            "schema_version": "refinement_candidate_state_v1",
            "state_key": key,
            "case_id": case.case_id,
            "variant_id": variant.variant_id,
            "status": result.status,
            "written_utc": utc_now(),
            "result": result.to_dict(),
        }
        paths: list[tuple[Path, str]] = []
        if self.options.resume_candidates:
            paths.append((self._local_candidate_state_path(case, variant), "local_resume"))
        if self.options.candidate_cache:
            paths.append((self._shared_candidate_state_path(key), "shared_cache"))
        for path, destination in paths:
            _write_json(path, payload)
            self.cache_stats["candidate_cache_writes"] += 1
            if destination == "local_resume":
                self.cache_stats["candidate_cache_local_resume_writes"] += 1
            elif destination == "shared_cache":
                self.cache_stats["candidate_cache_shared_writes"] += 1

    def _record_candidate_cache_hit(self, source: str) -> None:
        self.cache_stats["candidate_cache_hits"] += 1
        sources = self.cache_stats["candidate_cache_sources"]
        if isinstance(sources, dict):
            sources[source] = int(sources.get(source, 0)) + 1

    def _record_candidate_cache_skip(self, reason: str) -> None:
        skipped = self.cache_stats["candidate_cache_skipped_hits"]
        if isinstance(skipped, dict):
            skipped[reason] = int(skipped.get(reason, 0)) + 1

    def _record_candidate_cache_no_hit(self, reason: str) -> None:
        no_hits = self.cache_stats["candidate_cache_no_hit_reasons"]
        if isinstance(no_hits, dict):
            no_hits[reason] = int(no_hits.get(reason, 0)) + 1

    def _write_cache_stats(self) -> Path | None:
        if not (
            self.options.resume_candidates
            or self.options.candidate_cache
            or self.options.reference_cache
        ):
            return None
        payload = {
            "schema_version": "refinement_cache_stats_v1",
            "run_id": self.plan.run_id,
            "reference_cache_enabled": bool(self.options.reference_cache),
            "candidate_cache_enabled": bool(self.options.candidate_cache),
            "resume_candidates_enabled": bool(self.options.resume_candidates),
            "cache_root": self.cache_root.as_posix(),
            **self.cache_stats,
        }
        path = self.run_root / "cache-stats.json"
        _write_json(path, payload)
        return path

    def _case_dir(self, case: ExperimentCase) -> Path:
        return self.run_root / "c" / compact_path_segment(
            case.case_id,
            max_length=32,
            fallback="case",
        )

    def _case_variant_dir(
        self, case: ExperimentCase, variant: ExperimentVariant
    ) -> Path:
        return self._case_dir(case) / "v" / compact_path_segment(
            variant.variant_id,
            max_length=40,
            fallback="variant",
        )

    def _postprocess_result(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        result: ExperimentResult,
    ) -> ExperimentResult:
        variant_dir = self._case_variant_dir(case, variant)
        bounds_payload: Mapping[str, Any] = {}
        if self.options.write_bounds_debug:
            try:
                bounds_payload = build_bounds_debug_report(
                    result,
                    output_dir=variant_dir,
                    reference_paths=result.reference_paths,
                    render_paths=result.render_paths,
                    write_files=True,
                )
            except Exception as exc:
                bounds_payload = {"error": str(exc)}
        autopsy_payload: Mapping[str, Any] = {}
        if self.options.write_autopsy:
            autopsy_payload = autopsy_candidate(
                result,
                run_root=self.run_root,
                bounds_debug=bounds_payload,
            )
            (variant_dir / "autopsy.json").write_text(
                json.dumps(autopsy_payload, indent=2, sort_keys=True, default=str)
                + "\n",
                encoding="utf-8",
            )
        scored_result = ExperimentResult.from_dict(
            {
                **result.to_dict(),
                "bounds_debug": bounds_payload,
                "autopsy": autopsy_payload,
            }
        )
        moonshot_payload: Mapping[str, Any] = {}
        moonshot_artifacts: dict[str, Path] = {}
        if self.options.moonshot_sidecars:
            moonshot_payload, moonshot_artifacts = self._run_moonshot_sidecars(
                case,
                variant,
                scored_result,
                bounds_debug=bounds_payload,
                autopsy=autopsy_payload,
            )
            metrics = copy.deepcopy(dict(scored_result.metrics))
            metrics["moonshots"] = json_safe(moonshot_payload)
            set_metric_path(
                metrics,
                "moonshot.ran_count",
                float(moonshot_payload.get("ran_count", 0) or 0),
            )
            set_metric_path(
                metrics,
                "moonshot.error_count",
                float(moonshot_payload.get("error_count", 0) or 0),
            )
            for item in moonshot_payload.get("results", ()) or ():
                if not isinstance(item, Mapping):
                    continue
                experiment_id = str(item.get("experiment_id", ""))
                status = str(item.get("status", ""))
                if experiment_id:
                    set_metric_path(
                        metrics,
                        f"moonshot.experiments.{experiment_id}.ran",
                        1.0 if status == "ran" else 0.0,
                    )
            backend_result = copy.deepcopy(dict(scored_result.backend_result))
            backend_result["moonshot_evidence"] = json_safe(moonshot_payload)
            scored_result = ExperimentResult.from_dict(
                {
                    **scored_result.to_dict(),
                    "metrics": metrics,
                    "backend_result": backend_result,
                    "artifacts": {
                        **scored_result.to_dict().get("artifacts", {}),
                        **{key: path.as_posix() for key, path in moonshot_artifacts.items()},
                    },
                }
            )
        score_payload = score_result(scored_result, objective=self.plan.objective)
        return ExperimentResult(
            run_id=result.run_id,
            case_id=result.case_id,
            variant_id=result.variant_id,
            mode=result.mode,
            status=result.status,
            exit_code=result.exit_code,
            started_utc=result.started_utc,
            finished_utc=result.finished_utc,
            elapsed_s=result.elapsed_s,
            command=result.command,
            result_json=result.result_json,
            render_paths=result.render_paths,
            reference_paths=result.reference_paths,
            backend_result=scored_result.backend_result,
            metrics=scored_result.metrics,
            score=score_payload,
            artifacts={**dict(scored_result.artifacts), **moonshot_artifacts},
            autopsy=autopsy_payload,
            bounds_debug=bounds_payload,
            warnings=result.warnings,
            errors=result.errors,
        )

    def _run_moonshot_sidecars(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        result: ExperimentResult,
        *,
        bounds_debug: Mapping[str, Any],
        autopsy: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], dict[str, Path]]:
        try:
            from blender_blocking.moonshots import MoonshotRequest, list_experiments, run_experiment
        except Exception:  # pragma: no cover - script-style imports
            from moonshots import MoonshotRequest, list_experiments, run_experiment  # type: ignore

        experiment_ids = self.options.moonshot_experiments or tuple(
            experiment.experiment_id for experiment in list_experiments()
        )
        variant_dir = self._case_variant_dir(case, variant)
        moonshot_root = variant_dir / "moonshots"
        case_payload = case.to_dict()
        candidate_payload = {
            **result.to_dict(),
            "case": case_payload,
            "variant": variant.to_dict(),
            "bounds_debug": json_safe(bounds_debug),
            "autopsy": json_safe(autopsy),
        }
        results = []
        status_counts: dict[str, int] = {}
        artifacts: dict[str, Path] = {}
        for experiment_id in experiment_ids:
            prior_results = tuple(dict(item) for item in results)
            request = MoonshotRequest(
                experiment_id=experiment_id,
                target=None,
                candidate=candidate_payload,
                config={
                    "case": case_payload,
                    "variant": variant.to_dict(),
                    "target_signals": _signals_from_case(case),
                    "refinement_result": result.to_dict(),
                    "bounds_debug": json_safe(bounds_debug),
                    "autopsy": json_safe(autopsy),
                    "prior_moonshot_results": prior_results,
                },
                artifact_root=moonshot_root.as_posix(),
                allow_research_execution=True,
            )
            try:
                sidecar_result = run_experiment(request)
            except Exception as exc:
                try:
                    from blender_blocking.moonshots.contracts import error_result
                except Exception:  # pragma: no cover
                    from moonshots.contracts import error_result  # type: ignore

                sidecar_result = error_result(
                    request,
                    error=f"{type(exc).__name__}: {exc}",
                )
            payload = sidecar_result.to_dict()
            results.append(payload)
            status = str(payload.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            for key, value in payload.get("artifacts", {}).items():
                if value:
                    artifacts[f"moonshot_{experiment_id}_{key}"] = Path(str(value))
        summary = {
            "schema_version": "refinement_moonshot_evidence_v1",
            "enabled": True,
            "case_id": case.case_id,
            "variant_id": variant.variant_id,
            "experiment_count": len(results),
            "status_counts": dict(sorted(status_counts.items())),
            "ran_count": int(status_counts.get("ran", 0)),
            "skipped_count": int(status_counts.get("skipped", 0)),
            "unsupported_count": int(status_counts.get("unsupported", 0)),
            "error_count": int(status_counts.get("error", 0)),
            "top_candidate_deltas": _moonshot_top_candidate_deltas(results),
            "active_view_suggestions": _moonshot_active_view_suggestions(results),
            "active_view_sequence": _moonshot_active_view_sequence(results),
            "sdf_extraction_plans": _moonshot_sdf_extraction_plans(results),
            "retopology_phase_plans": _moonshot_retopology_phase_plans(results),
            "portfolio_available_actions": _moonshot_portfolio_available_actions(results),
            "portfolio_actions": _moonshot_portfolio_actions(results),
            "portfolio_rejected_actions": _moonshot_portfolio_rejected_actions(results),
            "portfolio_dependency_edges": _moonshot_portfolio_dependency_edges(results),
            "portfolio_execution_plan": _moonshot_portfolio_execution_plan(results),
            "portfolio_risks": _moonshot_portfolio_risks(results),
            "results": results,
        }
        summary_path = moonshot_root / "summary.json"
        _write_json(summary_path, summary)
        artifacts["moonshot_summary"] = summary_path
        return summary, artifacts

    def _print_result_row(self, result: ExperimentResult) -> None:
        category = ""
        if isinstance(result.autopsy, Mapping):
            category = str(result.autopsy.get("category", ""))
        print(
            f"  {result.status:<5} avg={result.avg_iou:.3f} min={result.min_iou:.3f} "
            f"elapsed={result.elapsed_s:.2f}s {category}"
        )

    def _write_adaptive_outputs(
        self,
        results: Sequence[ExperimentResult],
    ) -> tuple[Path, Path]:
        proposals = []
        sources_by_proposal: dict[str, list[ExperimentResult]] = {}
        for result in results:
            payload = {
                "status": result.status,
                "metrics": result.metrics,
                "backend_result": result.backend_result,
            }
            result_proposals = proposals_from_result_payload(
                payload,
                max_proposals=self.options.adaptive_max_proposals,
            )
            proposals.extend(result_proposals)
            for proposal in result_proposals:
                sources_by_proposal.setdefault(proposal.proposal_id, []).append(result)
        ranked = merge_proposals(
            proposals,
            max_proposals=self.options.adaptive_max_proposals,
        )
        proposal_path = self.run_root / "adaptive-proposals.json"
        variant_path = self.run_root / "adaptive-variants.json"
        proposal_payload = {
            "schema_version": "refinement_run_adaptive_proposals_v1",
            "run_id": self.plan.run_id,
            "suite": self.plan.suite,
            "track": self.plan.track,
            "objective": self.plan.objective,
            "source_result_count": len(results),
            "proposal_count": len(ranked),
            "proposals": [proposal.to_dict() for proposal in ranked],
        }
        variant_payload = {
            "schema_version": "refinement_run_adaptive_variants_v1",
            "run_id": self.plan.run_id,
            "source_result_count": len(results),
            "variant_count": len(ranked),
            "variants": [
                _variant_from_adaptive_proposal(
                    proposal,
                    sources=sources_by_proposal.get(proposal.proposal_id, ()),
                ).to_dict()
                for proposal in ranked
            ],
        }
        _write_json(proposal_path, proposal_payload)
        _write_json(variant_path, variant_payload)
        return proposal_path, variant_path

    def _write_lineage_outputs(
        self,
        results: Sequence[ExperimentResult],
    ) -> tuple[Path, Path]:
        try:
            from blender_blocking.evaluation.lineage import (
                ArtifactRecord,
                RunLineage,
                capture_environment,
                dirty_worktree,
                hash_file,
                hash_json_payload,
                repo_revision,
                write_reproduce_script,
                write_run_lineage,
            )
        except ImportError:  # pragma: no cover
            from evaluation.lineage import (
                ArtifactRecord,
                RunLineage,
                capture_environment,
                dirty_worktree,
                hash_file,
                hash_json_payload,
                repo_revision,
                write_reproduce_script,
                write_run_lineage,
            )

        root = self.run_root.resolve(strict=False)
        artifacts: list[ArtifactRecord] = []
        for key, path in _run_artifact_paths(root).items():
            if path.exists():
                artifacts.append(
                    ArtifactRecord.from_path(
                        path,
                        key=key,
                        root=root,
                        generated=True,
                        committed_allowed=False,
                    )
                )
        for result in results:
            result_key = f"{result.case_id}.{result.variant_id}"
            if result.result_json and result.result_json.exists():
                artifacts.append(
                    ArtifactRecord.from_path(
                        result.result_json,
                        key=f"{result_key}.result_json",
                        root=root,
                    )
                )
            for view, path in result.render_paths.items():
                if path.exists():
                    artifacts.append(
                        ArtifactRecord.from_path(
                            path,
                            key=f"{result_key}.render.{view}",
                            root=root,
                        )
                    )
            for artifact_key, path in result.artifacts.items():
                if path.exists():
                    artifacts.append(
                        ArtifactRecord.from_path(
                            path,
                            key=f"{result_key}.artifact.{artifact_key}",
                            root=root,
                        )
                    )
        input_hashes = {
            f"{case.case_id}.{view}": hash_file(path)
            for case in self.plan.cases
            for view, path in case.reference_paths.items()
            if Path(path).exists()
        }
        command = _lineage_reproduce_command(self.plan)
        reproduce_path = root / "reproduce.ps1"
        write_reproduce_script(command, reproduce_path, cwd=Path(__file__).resolve().parents[2])
        artifacts.append(
            ArtifactRecord.from_path(
                reproduce_path,
                key="reproduce_script",
                root=root,
                generated=True,
                committed_allowed=False,
            )
        )
        lineage = RunLineage(
            schema_version="run_lineage_v1",
            run_id=self.plan.run_id,
            parent_run_id=_parent_run_id(self.plan.metadata),
            repo_revision=repo_revision(Path(__file__).resolve().parents[2]),
            dirty_worktree=dirty_worktree(Path(__file__).resolve().parents[2]),
            command=command,
            environment=capture_environment(),
            input_hashes=input_hashes,
            config_hashes={
                "plan": hash_json_payload(self.plan.to_dict()),
                "base_config": hash_json_payload(self.base_config.to_dict()),
            },
            random_seeds={"plan_seed": int(self.plan.seed)},
            artifacts=tuple(artifacts),
        )
        lineage_path = write_run_lineage(lineage, root / "lineage.json")
        return lineage_path, reproduce_path


class InProcessBlenderRunner(BaseRunner):
    def _run_one(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
    ) -> ExperimentResult:
        started = utc_now()
        started_monotonic = time.perf_counter()
        variant_dir = self._case_variant_dir(case, variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        render_dir = variant_dir / "r"
        artifact_root = variant_dir / "a"
        result_json = variant_dir / "result.json"
        command = _variant_command(
            variant,
            result_json=result_json,
            render_dir=render_dir,
            artifact_root=artifact_root,
            reference_paths=reference_paths,
        )
        _write_text(variant_dir / "command.txt", " ".join(command) + "\n")
        config = copy.deepcopy(self.base_config)
        _apply_variant_to_config(config, variant)
        self._apply_run_option_config(config)
        config.validate()
        _write_json(variant_dir / "config.json", config.to_dict())
        try:
            from blender_blocking.test_e2e_validation import test_with_custom_images

            passed = test_with_custom_images(
                str(reference_paths["front"]),
                str(reference_paths["side"]),
                str(reference_paths["top"]),
                num_slices=config.reconstruction.num_slices,
                iou_threshold=0.7,
                render_config=config.render_silhouette,
                workflow_config=config,
                config_label=variant.variant_id,
                validation_mode=variant.validation_mode,
                render_output_dir=render_dir,
                artifact_root=artifact_root,
                result_json=result_json,
                run_id=f"{self.plan.run_id}-{case.case_id}-{variant.variant_id}",
                progress=self.options.progress,
                debug_output_dir=variant_dir / "dbg",
                debug_artifact_policy=self.options.debug_artifact_policy,
            )
            status = "pass" if passed else "fail"
            exit_code = 0 if passed else 1
            errors: tuple[str, ...] = ()
        except TypeError:
            try:
                from blender_blocking.test_e2e_validation import test_with_custom_images

                passed = test_with_custom_images(
                    str(reference_paths["front"]),
                    str(reference_paths["side"]),
                    str(reference_paths["top"]),
                    num_slices=config.reconstruction.num_slices,
                    iou_threshold=0.7,
                    render_config=config.render_silhouette,
                    workflow_config=config,
                    config_label=variant.variant_id,
                    validation_mode=variant.validation_mode,
                    render_output_dir=render_dir,
                    artifact_root=artifact_root,
                    result_json=result_json,
                    run_id=f"{self.plan.run_id}-{case.case_id}-{variant.variant_id}",
                    progress=self.options.progress,
                )
                status = "pass" if passed else "fail"
                exit_code = 0 if passed else 1
                errors = ()
            except Exception as exc:
                status = "error"
                exit_code = 2
                errors = (str(exc),)
        except Exception as exc:
            status = "error"
            exit_code = 2
            errors = (str(exc),)
        elapsed = time.perf_counter() - started_monotonic
        payload = _load_json(result_json)
        return _result_from_payload(
            plan_id=self.plan.run_id,
            case=case,
            variant=variant,
            status=status,
            exit_code=exit_code,
            started_utc=started,
            finished_utc=utc_now(),
            elapsed_s=elapsed,
            command=command,
            result_json=result_json,
            payload=payload,
            reference_paths=reference_paths,
            errors=errors,
        )

    def _apply_run_option_config(self, config: BlockingConfig) -> None:
        if self.options.cache_root is None:
            return
        if not (self.options.reference_cache or self.options.candidate_cache):
            return
        cache_dir = Path(self.options.cache_root) / "visual-hull-volumes"
        config.visual_hull.enable_cache = True
        config.visual_hull.cache_directory = str(cache_dir)
        config.visual_hull.cache_namespace = stable_hash(
            {
                "schema": "visual_hull_cache_namespace_v1",
                "suite": self.plan.suite,
                "track": self.plan.track,
                "base_config": self.base_config.to_dict(),
            },
            length=16,
        )
        config.visual_hull.cache_read = True
        config.visual_hull.cache_write = True


class SubprocessRunner(BaseRunner):
    def _run_one(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
    ) -> ExperimentResult:
        if not self.options.blender_executable:
            raise ValueError("SubprocessRunner requires blender_executable")
        started = utc_now()
        started_monotonic = time.perf_counter()
        variant_dir = self._case_variant_dir(case, variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        render_dir = (variant_dir / "r").resolve(strict=False)
        artifact_root = (variant_dir / "a").resolve(strict=False)
        result_json = (variant_dir / "result.json").resolve(strict=False)
        command = _variant_command(
            variant,
            result_json=result_json,
            render_dir=render_dir,
            artifact_root=artifact_root,
            reference_paths=reference_paths,
            blender_executable=self.options.blender_executable,
        )
        _write_text(variant_dir / "command.txt", " ".join(command) + "\n")
        stdout_path = variant_dir / "stdout.txt"
        stderr_path = variant_dir / "stderr.txt"
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        payload = _load_json(result_json)
        status = "pass" if completed.returncode == 0 else "fail"
        errors = (
            (completed.stderr.strip(),)
            if completed.returncode and completed.stderr.strip()
            else ()
        )
        return _result_from_payload(
            plan_id=self.plan.run_id,
            case=case,
            variant=variant,
            status=status,
            exit_code=completed.returncode,
            started_utc=started,
            finished_utc=utc_now(),
            elapsed_s=time.perf_counter() - started_monotonic,
            command=tuple(command),
            result_json=result_json,
            payload=payload,
            reference_paths=reference_paths,
            errors=errors,
        )


def runner_for_plan(
    plan: ExperimentPlan,
    *,
    options: RunOptions,
    base_config: BlockingConfig | None = None,
) -> BaseRunner:
    if options.subprocess_blender:
        return SubprocessRunner(plan=plan, options=options, base_config=base_config)
    return InProcessBlenderRunner(plan=plan, options=options, base_config=base_config)


def _variant_from_adaptive_proposal(
    proposal: Any,
    *,
    sources: Sequence[ExperimentResult],
) -> ExperimentVariant:
    variant = proposal.to_variant()
    if not sources:
        return variant
    source_rows = tuple(str(result.variant_id) for result in sources)
    has_promotable_source = any(promotion_decision(result).promotable for result in sources)
    if has_promotable_source:
        return replace(
            variant,
            parameters={
                **dict(variant.parameters),
                "adaptive_source_results": source_rows,
            },
        )
    return replace(
        variant,
        diagnostic_only=True,
        tags=tuple(dict.fromkeys(tuple(variant.tags) + ("diagnostic-only",))),
        parameters={
            **dict(variant.parameters),
            "adaptive_source_results": source_rows,
            "diagnostic_reason": "source_results_not_promotable",
        },
    )


def _result_from_payload(
    *,
    plan_id: str,
    case: ExperimentCase,
    variant: ExperimentVariant,
    status: str,
    exit_code: int,
    started_utc: str,
    finished_utc: str,
    elapsed_s: float,
    command: Sequence[str],
    result_json: Path,
    payload: Mapping[str, Any],
    reference_paths: Mapping[str, Path],
    errors: Sequence[str] = (),
) -> ExperimentResult:
    backend_result = _backend_result_from_payload(payload)
    metrics = _metrics_from_payload(payload)
    if not metrics.get("validation_mode"):
        metrics["validation_mode"] = variant.validation_mode
    set_metric_path(metrics, "variant.diagnostic_only", bool(variant.diagnostic_only))
    artifacts = _artifacts_from_payload(payload)
    render_paths = {
        key: Path(value)
        for key, value in dict(
            payload.get("rendered_paths", {}) if isinstance(payload, Mapping) else {}
        ).items()
    }
    return ExperimentResult(
        run_id=plan_id,
        case_id=case.case_id,
        variant_id=variant.variant_id,
        mode=variant.mode,
        status=status,
        exit_code=exit_code,
        started_utc=started_utc,
        finished_utc=finished_utc,
        elapsed_s=elapsed_s,
        command=tuple(command),
        result_json=result_json,
        render_paths=render_paths,
        reference_paths={key: Path(value) for key, value in reference_paths.items()},
        backend_result=backend_result,
        metrics=metrics,
        artifacts=artifacts,
        warnings=tuple(
            payload.get("warnings", ()) if isinstance(payload, Mapping) else ()
        ),
        errors=tuple(errors),
    )


def _metrics_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not isinstance(payload, Mapping):
        return metrics
    validation_mode = str(payload.get("validation_mode") or "")
    metrics["validation_mode"] = validation_mode
    if validation_mode == "render-iou" and isinstance(payload.get("average_iou"), (int, float)):
        value = float(payload["average_iou"])
        set_metric_path(metrics, "render.average_iou", value)
        metrics["average_iou"] = value
    if validation_mode == "render-iou" and isinstance(payload.get("min_view_iou"), (int, float)):
        value = float(payload["min_view_iou"])
        set_metric_path(metrics, "render.min_view_iou", value)
        metrics["min_view_iou"] = value
    views = payload.get("views", {})
    if isinstance(views, Mapping):
        metrics["views"] = json_safe(views)
        for view, item in views.items():
            if not isinstance(item, Mapping):
                continue
            area_value = item.get("iou", item.get("area_iou"))
            if isinstance(area_value, (int, float)):
                value = float(area_value)
                set_metric_path(metrics, f"render.per_view.{view}.area_iou", value)
                metrics[f"{view}_iou"] = value
            for source_key, target_key in (
                ("boundary_iou", f"render.per_view.{view}.boundary_iou"),
                (
                    "signed_distance_loss",
                    f"render.per_view.{view}.signed_distance_loss",
                ),
            ):
                raw = item.get(source_key)
                if isinstance(raw, (int, float)):
                    set_metric_path(metrics, target_key, float(raw))
        required_values = [
            metrics.get(f"{view}_iou")
            for view in ("front", "side", "top")
            if isinstance(metrics.get(f"{view}_iou"), (int, float))
        ]
        if len(required_values) == 3:
            min_required = float(min(required_values))
            set_metric_path(metrics, "render.min_view_iou", min_required)
            if validation_mode == "render-iou":
                metrics["min_view_iou"] = min_required
        set_render_aggregate_metrics(metrics)
    backend = _backend_result_from_payload(payload)
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    source = selected if isinstance(selected, Mapping) else backend
    metric_result = (
        source.get("metric_result", {}) if isinstance(source, Mapping) else {}
    )
    for bundle in _evaluation_bundles_from_payload(payload):
        for group in bundle.get("metric_groups", ()) or ():
            if not isinstance(group, Mapping):
                continue
            for metric in group.get("metrics", ()) or ():
                if not isinstance(metric, Mapping):
                    continue
                name = str(metric.get("name", ""))
                value = metric.get("value")
                if not name or not isinstance(value, (int, float, bool)):
                    continue
                numeric = float(value)
                metrics[name.replace(".", "_")] = numeric
                set_metric_path(metrics, name, numeric)
                alias = _bundle_metric_alias(name)
                if alias:
                    set_metric_path(metrics, alias, numeric)
                if name == "editability.editable_reconstruction_index":
                    metrics["editability_score"] = numeric
                elif name == "topology.score":
                    metrics["topology_score"] = numeric
    if isinstance(metric_result, Mapping):
        _set_metric_result_metrics(metrics, metric_result)
    return metrics


def _backend_result_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    backend = payload.get("backend_result")
    if isinstance(backend, Mapping):
        return backend
    if any(
        key in payload
        for key in (
            "selected",
            "candidates",
            "scores",
            "policy",
            "pareto_report",
            "cost_report",
        )
    ):
        return payload
    return {}


def _set_metric_result_per_view(
    metrics: dict[str, Any],
    metric_result: Mapping[str, Any],
) -> None:
    per_view = metric_result.get("per_view")
    if not isinstance(per_view, Mapping):
        return
    for view, item in per_view.items():
        if not isinstance(item, Mapping):
            continue
        for source_key, target_key in (
            ("area_iou", f"backend.per_view.{view}.area_iou"),
            ("boundary_iou", f"backend.per_view.{view}.boundary_iou"),
            (
                "signed_distance_loss",
                f"backend.per_view.{view}.signed_distance_loss",
            ),
        ):
            raw = item.get(source_key)
            if isinstance(raw, (int, float)):
                set_metric_path(metrics, target_key, float(raw))


def _set_metric_result_metrics(
    metrics: dict[str, Any],
    metric_result: Mapping[str, Any],
) -> None:
    for key in (
        "area_iou_mean",
        "area_iou_min",
        "boundary_iou_mean",
        "topology_score",
        "editability_score",
        "complexity_penalty",
        "elapsed_s",
    ):
        if isinstance(metric_result.get(key), (int, float)):
            target_key = namespace_metric_key(key)
            set_metric_path(metrics, target_key, float(metric_result[key]))
            if key == "topology_score":
                metrics["topology_score"] = float(metric_result[key])
            elif key == "editability_score":
                metrics["editability_score"] = float(metric_result[key])
    _set_metric_result_per_view(metrics, metric_result)
    _set_metric_result_render_qa(metrics, metric_result)


def _set_metric_result_render_qa(
    metrics: dict[str, Any],
    metric_result: Mapping[str, Any],
) -> None:
    extras = metric_result.get("extras")
    if not isinstance(extras, Mapping):
        return
    render_qa = extras.get("render_qa")
    if not isinstance(render_qa, Mapping):
        return
    status = render_qa.get("status")
    if status is not None:
        set_metric_path(metrics, "render.qa.status", str(status))
    missing = render_qa.get("missing_required_metrics")
    if isinstance(missing, bool):
        set_metric_path(metrics, "render.qa.missing_required_metrics", missing)
    per_view = render_qa.get("per_view")
    if not isinstance(per_view, Mapping):
        return
    for view, item in per_view.items():
        if not isinstance(item, Mapping):
            continue
        area_value = item.get("area_iou", item.get("iou"))
        if isinstance(area_value, (int, float)):
            value = float(area_value)
            set_metric_path(metrics, f"render.per_view.{view}.area_iou", value)
            metrics[f"{view}_iou"] = value
        for source_key, target_key in (
            ("boundary_iou", f"render.per_view.{view}.boundary_iou"),
            (
                "signed_distance_loss",
                f"render.per_view.{view}.signed_distance_loss",
            ),
        ):
            raw = item.get(source_key)
            if isinstance(raw, (int, float)):
                set_metric_path(metrics, target_key, float(raw))
    required_values = [
        metrics.get(f"{view}_iou")
        for view in ("front", "side", "top")
        if isinstance(metrics.get(f"{view}_iou"), (int, float))
    ]
    if len(required_values) == 3:
        min_required = float(min(required_values))
        set_metric_path(metrics, "render.min_view_iou", min_required)
        metrics["min_view_iou"] = min_required
        set_metric_path(metrics, "render.average_iou", sum(required_values) / 3.0)
        metrics["average_iou"] = sum(required_values) / 3.0
    set_render_aggregate_metrics(metrics)


def _evaluation_bundles_from_payload(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    bundles = []
    direct = payload.get("evaluation_bundles")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        bundles.extend(item for item in direct if isinstance(item, Mapping))
    single = payload.get("evaluation_bundle")
    if isinstance(single, Mapping):
        bundles.append(single)
    backend = payload.get("backend_result")
    if isinstance(backend, Mapping):
        bundles.extend(_evaluation_bundles_from_payload(backend))
    return tuple(bundles)


def _bundle_metric_alias(name: str) -> str:
    aliases = {
        "topology.score": "topology_score",
        "topology.penalty": "topology_penalty",
        "editability.editable_reconstruction_index": "editability.qa_score",
        "editability.complexity_penalty": "complexity_penalty",
        "geometry.fscore_tau": "geometry_fscore_tau",
        "geometry.volumetric_iou": "geometry_volumetric_iou",
        "geometry.chamfer_l2": "geometry_chamfer_l2",
        "geometry.chamfer_l1_normalized": "geometry_chamfer_l1_normalized",
        "geometry.chamfer_l2_normalized": "geometry_chamfer_l2_normalized",
        "geometry.surface_coverage": "geometry_surface_coverage",
    }
    return aliases.get(name, "")


def _artifacts_from_payload(payload: Mapping[str, Any]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    if not isinstance(payload, Mapping):
        return artifacts
    backend = _backend_result_from_payload(payload)
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    sources = []
    if isinstance(selected, Mapping):
        sources.append(selected)
    if isinstance(backend, Mapping):
        sources.append(backend)
    for source in sources:
        for key in ("mesh_path", "primitive_path", "volume_path"):
            value = source.get(key)
            if value:
                artifacts[key.replace("_path", "")] = Path(str(value))
        nested = source.get("artifacts", {})
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                if value:
                    artifacts[str(key)] = Path(str(value))
    return artifacts


def _variant_command(
    variant: ExperimentVariant,
    *,
    result_json: Path,
    render_dir: Path,
    artifact_root: Path | None = None,
    reference_paths: Mapping[str, Path] | None = None,
    blender_executable: str | None = None,
) -> tuple[str, ...]:
    args = list(variant.cli_args)
    if reference_paths:
        args.extend(
            [
                "--front",
                str(Path(reference_paths["front"]).resolve(strict=False)),
                "--side",
                str(Path(reference_paths["side"]).resolve(strict=False)),
                "--top",
                str(Path(reference_paths["top"]).resolve(strict=False)),
            ]
        )
    args.extend(["--result-json", str(Path(result_json).resolve(strict=False))])
    args.extend(["--render-output-dir", str(Path(render_dir).resolve(strict=False))])
    if artifact_root is not None:
        args.extend(
            [
                "--artifact-output-root",
                str(Path(artifact_root).resolve(strict=False)),
            ]
        )
    args.append("--no-progress")
    if blender_executable:
        return (
            blender_executable,
            "--background",
            "--python",
            "blender_blocking/test_e2e_validation.py",
            "--",
            *args,
        )
    return tuple(args)


def _apply_variant_to_config(cfg: BlockingConfig, variant: ExperimentVariant) -> None:
    cfg.reconstruction.reconstruction_mode = variant.mode
    for key, value in variant.parameters.items():
        apply_variant_parameter_to_config(cfg, key, value)
    if variant.config_overrides:
        _apply_config_overrides(cfg, variant.config_overrides)
    cfg.validate()


def _apply_config_overrides(target: object, overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise ValueError(f"unknown config override {type(target).__name__}.{key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply_config_overrides(current, value)
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(target, key, value)


def _candidate_cache_variant_payload(variant: ExperimentVariant) -> Mapping[str, Any]:
    ignored_parameters = {
        "adaptive_loop_generation",
        "adaptive_loop_parent_result",
        "adaptive_loop_parent_mode",
        "adaptive_loop_contributing_parent_results",
    }
    parameters = {
        key: value
        for key, value in dict(variant.parameters).items()
        if key not in ignored_parameters
    }
    return {
        "schema": "refinement_candidate_effective_variant_v2",
        "mode": variant.mode,
        "validation_mode": variant.validation_mode,
        "parameters": json_safe(parameters),
        "cli_args": tuple(variant.cli_args),
        "config_overrides": json_safe(variant.config_overrides),
    }


def _signals_from_case(case: ExperimentCase) -> Mapping[str, Mapping[str, Any]]:
    spec = case.metadata.get("spec", {}) if isinstance(case.metadata, Mapping) else {}
    if not isinstance(spec, Mapping):
        spec = {}
    parameters = spec.get("parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    family = str(
        parameters.get(
            "primitive",
            parameters.get(
                "profile_kind",
                parameters.get("blockout_kind", case.synthetic_definition),
            ),
        )
        or ""
    )
    complexity = 0.25
    if family in {"chair", "vehicle", "adversarial", "checkerboard_breakup"}:
        complexity = 0.65
    elif family in {"box", "cube", "rect", "block"}:
        complexity = 0.18
    hole_count = 1 if family in {"chair", "torus", "checkerboard_breakup"} else 0
    band_samples = int(parameters.get("profile_samples", 24) or 24)
    return {
        "surface": {
            "available": bool(spec),
            "constraint_count": len(tuple(case.required_views)),
            "surface_point_count": int(parameters.get("surface_point_count", 0) or 0),
            "target_views": tuple(case.required_views),
        },
        "profile": {
            "available": bool(spec),
            "view_count": len(tuple(case.required_views)),
            "band_samples": band_samples,
            "interval_count": max(band_samples, int(band_samples * (1.0 + complexity))),
            "hole_count": hole_count,
            "mean_width": float(parameters.get("width", parameters.get("radius", 1.0)) or 1.0),
            "max_width": float(parameters.get("width", parameters.get("radius", 1.0)) or 1.0),
            "complexity": complexity,
        },
        "constraints": {
            "available": bool(case.required_views),
            "constraint_count": len(tuple(case.required_views)),
            "constraint_views": {view: 1 for view in case.required_views},
        },
        "uncertainty": {
            "available": bool(case.known_ambiguity_notes),
            "overall_confidence_mean": 0.82 if case.known_ambiguity_notes else 0.95,
            "overall_boundary_uncertainty_mean": 0.22 if case.known_ambiguity_notes else 0.05,
            "consistency": 0.72 if case.known_ambiguity_notes else 0.88,
            "view_details": {},
        },
        "topology": {
            "available": bool(spec),
            "score": max(0.35, 1.0 - complexity * 0.35),
            "complexity": complexity,
            "detail": "synthetic_case_metadata",
        },
    }


def _moonshot_top_candidate_deltas(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    deltas = []
    for result in results:
        experiment_id = str(result.get("experiment_id", ""))
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for key in (
            "volumetric_iou_delta",
            "topology_score_delta",
            "editability_score_delta",
            "expected_objective_delta",
            "best_expected_metric_delta",
        ):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                deltas.append(
                    {
                        "experiment_id": experiment_id,
                        "metric": key,
                        "delta": float(value),
                    }
                )
    deltas.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return deltas[:8]


def _moonshot_active_view_suggestions(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    suggestions = []
    for result in results:
        if result.get("experiment_id") != "active_view_planning":
            continue
        evidence = result.get("degradation", {}).get("evidence") if isinstance(result.get("degradation"), Mapping) else None
        if not isinstance(evidence, Mapping):
            continue
        requests = evidence.get("requests")
        if isinstance(requests, Sequence) and not isinstance(requests, (str, bytes)):
            suggestions.extend(item for item in requests if isinstance(item, Mapping))
    suggestions.sort(
        key=lambda item: float(item.get("expected_metric_delta", item.get("score", 0.0)) or 0.0),
        reverse=True,
    )
    return suggestions[:5]


def _moonshot_active_view_sequence(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    sequence = []
    for result in results:
        if result.get("experiment_id") != "active_view_planning":
            continue
        evidence = _moonshot_evidence(result)
        rows = evidence.get("sequence_plan")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            sequence.extend(item for item in rows if isinstance(item, Mapping))
    sequence.sort(key=lambda item: int(item.get("order", 999) or 999))
    return sequence[:8]


def _moonshot_sdf_extraction_plans(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    plans = []
    for result in results:
        if result.get("experiment_id") != "implicit_sdf_proxy":
            continue
        plan = _moonshot_evidence(result).get("extraction_plan")
        if isinstance(plan, Mapping):
            plans.append(plan)
    return plans[:4]


def _moonshot_retopology_phase_plans(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    phases = []
    for result in results:
        if result.get("experiment_id") != "editable_retopology":
            continue
        rows = _moonshot_evidence(result).get("phase_plan")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            phases.extend(item for item in rows if isinstance(item, Mapping))
    return phases[:8]


def _moonshot_portfolio_actions(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    actions = []
    for result in results:
        if result.get("experiment_id") != "moonshot_portfolio_optimizer":
            continue
        evidence = result.get("degradation", {}).get("evidence") if isinstance(result.get("degradation"), Mapping) else None
        if not isinstance(evidence, Mapping):
            continue
        selected = evidence.get("selected_actions")
        if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
            actions.extend(item for item in selected if isinstance(item, Mapping))
    actions.sort(
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )
    return actions[:8]


def _moonshot_portfolio_available_actions(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    actions = []
    for result in results:
        if result.get("experiment_id") != "moonshot_portfolio_optimizer":
            continue
        evidence = _moonshot_evidence(result)
        rows = evidence.get("available_actions")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            actions.extend(item for item in rows if isinstance(item, Mapping))
    actions.sort(
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )
    return actions[:12]


def _moonshot_portfolio_rejected_actions(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    actions = []
    for result in results:
        if result.get("experiment_id") != "moonshot_portfolio_optimizer":
            continue
        evidence = _moonshot_evidence(result)
        rows = evidence.get("rejected_actions")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            actions.extend(item for item in rows if isinstance(item, Mapping))
    actions.sort(
        key=lambda item: (str(item.get("rejection", "")), -float(item.get("score", 0.0) or 0.0)),
    )
    return actions[:12]


def _moonshot_portfolio_dependency_edges(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    edges = []
    for result in results:
        if result.get("experiment_id") != "moonshot_portfolio_optimizer":
            continue
        rows = _moonshot_evidence(result).get("dependency_edges")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            edges.extend(item for item in rows if isinstance(item, Mapping))
    return edges[:12]


def _moonshot_portfolio_execution_plan(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    plan = []
    for result in results:
        if result.get("experiment_id") != "moonshot_portfolio_optimizer":
            continue
        rows = _moonshot_evidence(result).get("execution_plan")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            plan.extend(item for item in rows if isinstance(item, Mapping))
    return plan[:8]


def _moonshot_portfolio_risks(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    risks = []
    for result in results:
        if result.get("experiment_id") != "moonshot_portfolio_optimizer":
            continue
        rows = _moonshot_evidence(result).get("risk_register")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            risks.extend(item for item in rows if isinstance(item, Mapping))
    risks.sort(key=lambda item: float(item.get("risk", 0.0) or 0.0), reverse=True)
    return risks[:8]


def _moonshot_evidence(result: Mapping[str, Any]) -> Mapping[str, Any]:
    degradation = result.get("degradation")
    if isinstance(degradation, Mapping):
        evidence = degradation.get("evidence")
        if isinstance(evidence, Mapping):
            return evidence
    return {}


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = _load_json_payload(path, default={})
    return payload if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json_payload(path, payload)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _reference_views_ready(directory: Path) -> bool:
    return all((Path(directory) / f"{view}.png").exists() for view in ("front", "side", "top"))


def _copy_reference_views(source_dir: Path, output_dir: Path) -> tuple[dict[str, Path], str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Path] = {}
    missing: list[str] = []
    for view in ("front", "side", "top"):
        source = Path(source_dir) / f"{view}.png"
        if not source.exists():
            missing.append(view)
            continue
        target = output_dir / source.name
        _link_or_copy(source, target)
        rendered[view] = target
    missing.extend(
        view
        for view, path in rendered.items()
        if not Path(path).exists()
    )
    if missing:
        return rendered, f"reference_generation_missing_views:{','.join(sorted(set(missing)))}"
    return rendered, None


def _require_reference_views(directory: Path, *, context: str) -> None:
    missing = [
        view
        for view in ("front", "side", "top")
        if not (Path(directory) / f"{view}.png").exists()
    ]
    if missing:
        raise RuntimeError(
            f"reference_generation_missing_views:{','.join(missing)}:{context}"
        )


def _require_reference_mapping(reference_paths: Mapping[str, Path], *, context: str) -> None:
    missing = [
        view
        for view in ("front", "side", "top")
        if view not in reference_paths or not Path(reference_paths[view]).exists()
    ]
    if missing:
        raise RuntimeError(
            f"reference_generation_missing_views:{','.join(missing)}:{context}"
        )


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        try:
            if target.stat().st_size == source.stat().st_size:
                return
        except OSError:
            pass
        target.unlink()
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _reference_hashes(reference_paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        str(view): _file_hash(Path(path))
        for view, path in sorted(reference_paths.items())
    }


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _result_artifacts_ready(result: ExperimentResult) -> bool:
    paths = []
    if result.result_json is not None:
        paths.append(result.result_json)
    paths.extend(result.render_paths.values())
    paths.extend(result.artifacts.values())
    return all(Path(path).exists() for path in paths)


_CACHED_DEPENDENCY_REPORT: dict[str, object] | None = None
_CACHED_GIT_INFO: dict[str, object] | None = None
_CACHED_ENVIRONMENT_INFO: dict[str, object] | None = None


def _cached_dependency_report() -> dict[str, object]:
    global _CACHED_DEPENDENCY_REPORT
    if _CACHED_DEPENDENCY_REPORT is None:
        _CACHED_DEPENDENCY_REPORT = dict(
            dependency_report(
                (
                    "numpy",
                    "cv2",
                    "PIL",
                    "scipy",
                    "skimage",
                    "open3d",
                    "trimesh",
                    "torch",
                    "torchvision",
                    "lpips",
                    "openvdb",
                    "nvdiffrast",
                )
            )
        )
    return dict(_CACHED_DEPENDENCY_REPORT)


def _cached_git_info() -> dict[str, object]:
    global _CACHED_GIT_INFO
    if _CACHED_GIT_INFO is None:
        _CACHED_GIT_INFO = _git_info()
    return dict(_CACHED_GIT_INFO)


def _cached_environment_info() -> dict[str, object]:
    global _CACHED_ENVIRONMENT_INFO
    if _CACHED_ENVIRONMENT_INFO is None:
        _CACHED_ENVIRONMENT_INFO = _environment_info()
    return dict(_CACHED_ENVIRONMENT_INFO)


def _run_artifact_paths(run_root: Path) -> Mapping[str, Path]:
    return {
        "plan": run_root / "plan.json",
        "manifest": run_root / "manifest.json",
        "index": run_root / "index.jsonl",
        "leaderboard_json": run_root / "leaderboard.json",
        "leaderboard_md": run_root / "leaderboard.md",
        "report_html": run_root / "report.html",
        "adaptive_proposals": run_root / "adaptive-proposals.json",
        "adaptive_variants": run_root / "adaptive-variants.json",
        "global_index": run_root.parent / "global-index.jsonl",
    }


def _lineage_reproduce_command(plan: ExperimentPlan) -> tuple[str, ...]:
    command = [
        "python",
        "-m",
        "blender_blocking.refinement_lab.cli",
        "run",
        "--suite",
        plan.suite,
        "--track",
        plan.track,
        "--search",
        plan.search,
        "--objective",
        plan.objective,
        "--result-root",
        str(plan.output_root),
        "--seed",
        str(plan.seed),
        "--top-k",
        str(plan.top_k),
    ]
    if plan.max_runs is not None:
        command.extend(("--max-runs", str(plan.max_runs)))
    return tuple(command)


def _parent_run_id(metadata: Mapping[str, Any]) -> str | None:
    for key in ("parent_run_id", "source_run_id", "previous_run_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _environment_info() -> dict[str, object]:
    info = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }
    try:
        import bpy

        info["blender_version"] = bpy.app.version_string
    except Exception:
        info["blender_version"] = None
    return info


def _git_info() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_count": len([line for line in status.splitlines() if line.strip()]),
    }
