r"""
Run a quick silhouette_intersection build + render in Blender.

# TODO(silhouette_intersection): Keep this helper until boolean intersection
# outputs are no longer boxy/empty for car/star inputs.

Usage (PowerShell):
  $blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
  & $blender --background --python scripts/debug_silhouette_intersection.py -- `
    --base car

  # Or with explicit images + config
  & $blender --background --python scripts/debug_silhouette_intersection.py -- `
    --front blender_blocking/test_images/car_front.png `
    --side blender_blocking/test_images/car_side.png `
    --top blender_blocking/test_images/car_top.png `
    --config-path configs/silhouette_intersection-ultra.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

try:
    import bpy  # noqa: F401  # type: ignore

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_python_paths(root: Path) -> None:
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "blender_blocking"))
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        site = Path(userprofile) / "blender_python_packages"
        if site.exists():
            sys.path.insert(0, str(site))


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug silhouette_intersection in Blender"
    )
    parser.add_argument(
        "--base", type=str, default=None, help="Base name in test_images (e.g., car)"
    )
    parser.add_argument("--front", type=str, default=None, help="Front image path")
    parser.add_argument("--side", type=str, default=None, help="Side image path")
    parser.add_argument("--top", type=str, default=None, help="Top image path")
    parser.add_argument(
        "--config-path", type=str, default=None, help="BlockingConfig override JSON"
    )
    parser.add_argument(
        "--render-dir",
        type=str,
        default="temp/debug_intersection",
        help="Output directory for renders",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Render resolution (square)",
    )
    return parser.parse_args(argv)


def main() -> int:
    if not BLENDER_AVAILABLE:
        print("Error: Must run inside Blender.")
        return 2

    root = _repo_root()
    _add_python_paths(root)

    from blender_blocking.config import BlockingConfig
    from blender_blocking.main_integration import BlockingWorkflow
    from blender_blocking.integration.blender_ops.render_utils import (
        render_orthogonal_views,
    )

    try:
        from blender_blocking.test_e2e_validation import _apply_overrides
    except Exception:
        _apply_overrides = None

    args = _parse_args()

    if args.base:
        img_dir = root / "blender_blocking" / "test_images"
        front = img_dir / f"{args.base}_front.png"
        side = img_dir / f"{args.base}_side.png"
        top = img_dir / f"{args.base}_top.png"
    else:
        front = Path(args.front) if args.front else None
        side = Path(args.side) if args.side else None
        top = Path(args.top) if args.top else None

    if not front or not side:
        print("Error: Provide --base or --front/--side (and optional --top)")
        return 2

    cfg = BlockingConfig()
    cfg.reconstruction.reconstruction_mode = "silhouette_intersection"

    overrides: Dict[str, Any] = {}
    if args.config_path:
        overrides = json.loads(Path(args.config_path).read_text(encoding="utf-8"))
        if _apply_overrides:
            _apply_overrides(cfg, overrides)
        else:
            print("Warning: _apply_overrides not available; config overrides ignored")

    cfg.validate()

    workflow = BlockingWorkflow(
        front_path=str(front),
        side_path=str(side),
        top_path=str(top) if top else None,
        config=cfg,
    )
    mesh = workflow.run_full_workflow(num_slices=cfg.reconstruction.num_slices)
    if mesh is None:
        print("Error: No mesh generated.")
        return 1

    print(f"Mesh: {mesh.name}")
    print(f"  verts: {len(mesh.data.vertices)} faces: {len(mesh.data.polygons)}")

    out_dir = root / args.render_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    render_orthogonal_views(
        str(out_dir),
        views=["front", "side", "top"],
        target_objects=[mesh],
        resolution=(args.resolution, args.resolution),
        transparent_bg=True,
    )
    print(f"Rendered to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
