"""Command line entry point for the synthetic shape factory."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .artifact_writer import validate_manifest_tree, write_artifact_set
from .ground_truth import build_pure_artifacts
from .registry import get_definition, list_definitions, list_suites, specs_for_suite
from .specs import json_safe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m blender_blocking.synthetic.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered synthetic shapes and suites.")
    list_parser.add_argument("--family", default=None, help="Optional family filter.")

    generate_parser = subparsers.add_parser("generate", help="Generate a synthetic suite.")
    generate_parser.add_argument("--suite", default="smoke", help="Suite name.")
    generate_parser.add_argument("--out", default="test_output/synthetic", help="Output root.")
    generate_parser.add_argument("--count", type=int, default=None, help="Optional number of generated specs.")
    generate_parser.add_argument("--seed", type=int, default=0, help="Base deterministic seed.")
    generate_parser.add_argument("--volume-resolution", type=int, default=64, help="Analytic volume resolution.")
    generate_parser.add_argument(
        "--blender",
        action="store_true",
        help="Build Blender meshes for Blender-supported specs. Requires running inside Blender.",
    )

    validate_parser = subparsers.add_parser("validate-manifest", help="Validate one manifest or a manifest tree.")
    validate_parser.add_argument("path", help="Manifest path or directory containing manifests.")

    args = parser.parse_args(argv)

    if args.command == "list":
        return _cmd_list(args.family)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "validate-manifest":
        return _cmd_validate(Path(args.path))
    parser.error(f"unknown command {args.command}")
    return 2


def _cmd_list(family: str | None) -> int:
    payload = {
        "suites": list(list_suites()),
        "definitions": [
            {
                "name": definition.name,
                "family": definition.family,
                "pure_python": definition.pure_python,
                "blender_supported": definition.blender_supported,
                "description": definition.description,
            }
            for definition in list_definitions(family)
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    out_root = Path(args.out)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_seed_{args.seed:04d}_{args.suite}"
    run_root = out_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    specs = specs_for_suite(args.suite, seed=args.seed, count=args.count)
    generated = []
    for spec in specs:
        artifacts = build_pure_artifacts(spec, volume_resolution=args.volume_resolution)
        mesh_paths = {}
        if args.blender:
            definition = get_definition(_definition_name_from_spec(spec))
            if not definition.blender_supported:
                raise SystemExit(f"{spec.shape_id} is not Blender-supported; omit --blender or choose another suite.")
            from .blender_builders import export_mesh

            mesh_paths = export_mesh(spec, run_root / spec.shape_id / "mesh")
            artifacts["mesh_paths"] = mesh_paths
        artifact_set = write_artifact_set(spec, artifacts, run_root)
        generated.append(
            {
                "shape_id": spec.shape_id,
                "family": spec.family,
                "manifest": artifact_set.manifest_path.as_posix(),
                "mesh_paths": json_safe(mesh_paths),
            }
        )

    index_path = run_root / "manifest.json"
    index_payload = {
        "suite": args.suite,
        "seed": args.seed,
        "count": len(generated),
        "run_root": run_root.as_posix(),
        "generated": generated,
    }
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(index_payload, indent=2, sort_keys=True))
    return 0


def _cmd_validate(path: Path) -> int:
    result = validate_manifest_tree(path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _definition_name_from_spec(spec: object) -> str:
    parameters = getattr(spec, "parameters")
    for key in ("primitive", "profile_kind", "blockout_kind", "mask_kind", "degradation"):
        if key in parameters:
            return str(parameters[key])
    raise ValueError(f"Cannot infer registry definition for {getattr(spec, 'shape_id', '<unknown>')}")


if __name__ == "__main__":
    sys.exit(main())
