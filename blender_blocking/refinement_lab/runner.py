"""Execution engine for refinement experiment plans."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
import copy
import json
import platform
from pathlib import Path
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
    utc_now,
)
from .parameter_search import score_result
from .parameters import apply_variant_parameter_to_config
from .result_index import ResultIndex, append_global_index

try:
    from blender_blocking.config import BlockingConfig
    from blender_blocking.utils.optional_deps import dependency_report
except ImportError:  # pragma: no cover
    from config import BlockingConfig
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
        total = len(self.plan.cases) * len(self.plan.variants)
        counter = 0
        for case in self.plan.cases:
            references = self._prepare_case_references(case)
            for variant in self.plan.variants:
                counter += 1
                print(
                    f"[{counter:03d}/{total:03d}] {variant.variant_id} ({variant.mode})"
                )
                result = self._run_one(case, variant, references)
                result = self._postprocess_result(case, variant, result)
                self.index.append(result)
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
            git=_git_info(),
            environment=_environment_info(),
            dependency_report=dependency_report(
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
            ),
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
        rendered = render_views(
            spec,
            output,
            resolution=tuple(self.base_config.render_silhouette.resolution),
            include_orbit=False,
        )
        return {
            view: Path(rendered[view])
            for view in ("front", "side", "top")
            if view in rendered
        }

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
    metrics["validation_mode"] = payload.get("validation_mode")
    if isinstance(payload.get("average_iou"), (int, float)):
        metrics["average_iou"] = float(payload["average_iou"])
    views = payload.get("views", {})
    if isinstance(views, Mapping):
        metrics["views"] = json_safe(views)
        for view, item in views.items():
            if isinstance(item, Mapping) and isinstance(item.get("iou"), (int, float)):
                metrics[f"{view}_iou"] = float(item["iou"])
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
                metrics[key] = float(metric_result[key])
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
                alias = _bundle_metric_alias(name)
                if alias:
                    metrics.setdefault(alias, numeric)
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
        "silhouette.min_view_iou": "area_iou_min",
        "silhouette.average_iou": "area_iou_mean",
        "silhouette.mean_boundary_iou": "boundary_iou_mean",
        "topology.score": "topology_score",
        "topology.penalty": "topology_penalty",
        "editability.editable_reconstruction_index": "editability_score",
        "editability.complexity_penalty": "complexity_penalty",
        "geometry.fscore_tau": "geometry_fscore_tau",
        "geometry.volumetric_iou": "geometry_volumetric_iou",
        "geometry.chamfer_l2": "geometry_chamfer_l2",
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
