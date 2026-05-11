"""Execution engine for refinement experiment plans."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
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
from .parameter_search import score_result
from .parameters import apply_variant_parameter_to_config
from .result_index import ResultIndex, append_global_index

try:
    from blender_blocking.config import BlockingConfig
    from blender_blocking.metrics.namespaces import (
        namespace_metric_key,
        set_metric_path,
        set_render_aggregate_metrics,
    )
    from blender_blocking.utils.optional_deps import dependency_report
except ImportError:  # pragma: no cover
    from config import BlockingConfig
    from metrics.namespaces import (
        namespace_metric_key,
        set_metric_path,
        set_render_aggregate_metrics,
    )
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
        self.index = ResultIndex(self.run_root, objective=plan.objective)

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
                result = self._load_reusable_candidate(case, variant, references)
                if result is None:
                    result = self._run_one(case, variant, references)
                    result = self._postprocess_result(case, variant, result)
                    self._write_candidate_state(case, variant, references, result)
                else:
                    print("  reused cached candidate result")
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
        if self.options.reference_cache and self.options.cache_root is not None:
            cache_dir = self._reference_cache_dir(case, resolution)
            if not _reference_views_ready(cache_dir):
                cache_dir.mkdir(parents=True, exist_ok=True)
                render_views(
                    spec,
                    cache_dir,
                    resolution=resolution,
                    include_orbit=False,
                )
            rendered = _copy_reference_views(cache_dir, output)
        else:
            rendered = render_views(
                spec,
                output,
                resolution=resolution,
                include_orbit=False,
            )
        return {
            view: Path(rendered[view])
            for view in ("front", "side", "top")
            if view in rendered
        }

    def _reference_cache_dir(
        self,
        case: ExperimentCase,
        resolution: tuple[int, int],
    ) -> Path:
        root = Path(self.options.cache_root or (self.run_root / ".cache"))
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
        return root / "references" / key[:2] / key

    def _candidate_state_key(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
    ) -> str:
        return stable_hash(
            {
                "schema": "refinement_candidate_state_v1",
                "suite": self.plan.suite,
                "track": self.plan.track,
                "objective": self.plan.objective,
                "case": case.to_dict(),
                "variant": {
                    "schema": "refinement_candidate_effective_variant_v1",
                    "mode": variant.mode,
                    "validation_mode": variant.validation_mode,
                    "variant_hash": variant.variant_hash(),
                },
                "base_config": self.base_config.to_dict(),
                "references": _reference_hashes(reference_paths),
            },
            length=32,
        )

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
        root = Path(self.options.cache_root or (self.run_root / ".cache"))
        return root / "candidates" / key[:2] / key / "candidate-state.json"

    def _load_reusable_candidate(
        self,
        case: ExperimentCase,
        variant: ExperimentVariant,
        reference_paths: Mapping[str, Path],
    ) -> ExperimentResult | None:
        if not (self.options.resume_candidates or self.options.candidate_cache):
            return None
        key = self._candidate_state_key(case, variant, reference_paths)
        candidates: list[Path] = []
        if self.options.resume_candidates:
            candidates.append(self._local_candidate_state_path(case, variant))
        if self.options.candidate_cache:
            candidates.append(self._shared_candidate_state_path(key))
        for path in candidates:
            result = self._load_candidate_state(path, key)
            if result is not None:
                return self._adapt_reused_candidate_result(result, case, variant)
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
            return None
        if state.get("schema_version") != "refinement_candidate_state_v1":
            return None
        if state.get("state_key") != key:
            return None
        result_payload = state.get("result")
        if not isinstance(result_payload, Mapping):
            return None
        try:
            result = ExperimentResult.from_dict(
                {**dict(result_payload), "run_id": self.plan.run_id}
            )
        except Exception:
            return None
        if not _result_artifacts_ready(result):
            return None
        return result

    def _adapt_reused_candidate_result(
        self,
        result: ExperimentResult,
        case: ExperimentCase,
        variant: ExperimentVariant,
    ) -> ExperimentResult:
        metrics = copy.deepcopy(dict(result.metrics))
        set_metric_path(metrics, "variant.diagnostic_only", bool(variant.diagnostic_only))
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
        paths = []
        if self.options.resume_candidates:
            paths.append(self._local_candidate_state_path(case, variant))
        if self.options.candidate_cache:
            paths.append(self._shared_candidate_state_path(key))
        for path in paths:
            _write_json(path, payload)

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
            backend_result=result.backend_result,
            metrics=result.metrics,
            score=score_payload,
            artifacts=result.artifacts,
            autopsy=autopsy_payload,
            bounds_debug=bounds_payload,
            warnings=result.warnings,
            errors=result.errors,
        )

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
        for result in results:
            payload = {
                "status": result.status,
                "metrics": result.metrics,
                "backend_result": result.backend_result,
            }
            proposals.extend(
                proposals_from_result_payload(
                    payload,
                    max_proposals=self.options.adaptive_max_proposals,
                )
            )
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
            "variants": [proposal.to_variant().to_dict() for proposal in ranked],
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
    backend_result = (
        payload.get("backend_result", {}) if isinstance(payload, Mapping) else {}
    )
    metrics = _metrics_from_payload(payload)
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
        backend_result=backend_result if isinstance(backend_result, Mapping) else {},
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
    backend = payload.get("backend_result", {})
    selected = backend.get("selected") if isinstance(backend, Mapping) else None
    source = selected if isinstance(selected, Mapping) else backend
    metric_result = (
        source.get("metric_result", {}) if isinstance(source, Mapping) else {}
    )
    if isinstance(metric_result, Mapping):
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
    return metrics


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
    backend = payload.get("backend_result", {})
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


def _load_json(path: Path) -> Mapping[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _reference_views_ready(directory: Path) -> bool:
    return all((Path(directory) / f"{view}.png").exists() for view in ("front", "side", "top"))


def _copy_reference_views(source_dir: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Path] = {}
    for view in ("front", "side", "top"):
        source = Path(source_dir) / f"{view}.png"
        if not source.exists():
            continue
        target = output_dir / source.name
        _link_or_copy(source, target)
        rendered[view] = target
    return rendered


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
