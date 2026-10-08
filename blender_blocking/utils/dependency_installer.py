"""Pip install plans for optional Blender Python dependencies.

The runtime probes in :mod:`utils.optional_deps` intentionally do not install
packages.  This module contains the explicit repair/install commands that setup
tools can run when a user wants optional research dependencies available inside
Blender's Python process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


TORCH_CPU_VERSION = "2.5.1"
TORCHVISION_CPU_VERSION = "0.20.1"
LPIPS_VERSION = "0.1.4"
SYMPY_VERSION = "1.13.1"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


@dataclass(frozen=True)
class PipInstallStep:
    """One pip command in an optional dependency install plan."""

    reason: str
    args: tuple[str, ...]

    def command_line(self) -> str:
        return " ".join(_quote_arg(arg) for arg in self.args)

    def to_dict(self) -> dict[str, object]:
        return {"reason": self.reason, "args": list(self.args)}


@dataclass(frozen=True)
class DependencyInstallPlan:
    """A deterministic sequence of pip install commands."""

    name: str
    steps: tuple[PipInstallStep, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
            "notes": list(self.notes),
        }


def blender_research_cpu_plan(
    *,
    python_executable: str | Path | None = None,
    user: bool = True,
    force_reinstall: bool = True,
) -> DependencyInstallPlan:
    """Return the supported CPU torch/torchvision/lpips install plan.

    The versions are pinned because Blender's Windows process can be sensitive
    to native runtime changes in torch wheels.  This stack is intentionally
    CPU-only and therefore works on AMD systems without pretending to provide
    ROCm or CUDA support.
    """

    python = str(python_executable or sys.executable)
    user_args = ("--user",) if user else ()
    force_args = ("--force-reinstall",) if force_reinstall else ()
    pip = (python, "-m", "pip", "install")
    return DependencyInstallPlan(
        name="blender-research-cpu",
        notes=(
            "Installs CPU-only PyTorch for Blender Python; this is suitable for "
            "LPIPS and CPU research paths on AMD/non-CUDA machines.",
            "nvdiffrast remains NVIDIA/CUDA-only and is not installed by this plan.",
        ),
        steps=(
            PipInstallStep(
                reason=(
                    "Install the known-good CPU torch/torchvision wheel pair for "
                    "Blender's Python 3.11 runtime."
                ),
                args=(
                    *pip,
                    *user_args,
                    *force_args,
                    "--no-deps",
                    "--index-url",
                    PYTORCH_CPU_INDEX,
                    f"torch=={TORCH_CPU_VERSION}+cpu",
                    f"torchvision=={TORCHVISION_CPU_VERSION}+cpu",
                ),
            ),
            PipInstallStep(
                reason=(
                    "Install torch runtime dependencies, including the sympy "
                    "version pinned by torch 2.5.1."
                ),
                args=(
                    *pip,
                    *user_args,
                    *force_args,
                    "filelock",
                    "fsspec",
                    "jinja2",
                    "networkx",
                    f"sympy=={SYMPY_VERSION}",
                    "typing-extensions",
                ),
            ),
            PipInstallStep(
                reason="Install LPIPS without allowing it to replace the pinned torch stack.",
                args=(
                    *pip,
                    *user_args,
                    "--no-deps",
                    f"lpips=={LPIPS_VERSION}",
                ),
            ),
        ),
    )


def execute_install_plan(
    plan: DependencyInstallPlan,
    *,
    dry_run: bool = False,
) -> int:
    """Execute a dependency install plan and return a process-style exit code."""

    for index, step in enumerate(plan.steps, start=1):
        print(f"\n[{index}/{len(plan.steps)}] {step.reason}")
        print(step.command_line())
        if dry_run:
            continue
        completed = subprocess.run(step.args)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def print_install_plan(plan: DependencyInstallPlan) -> None:
    """Print a human-readable install plan."""

    print(f"Install plan: {plan.name}")
    for note in plan.notes:
        print(f"  - {note}")
    for index, step in enumerate(plan.steps, start=1):
        print(f"\n[{index}/{len(plan.steps)}] {step.reason}")
        print(f"  {step.command_line()}")


def _quote_arg(arg: object) -> str:
    text = str(arg)
    if not text:
        return '""'
    if any(char.isspace() for char in text):
        return f'"{text}"'
    return text


__all__ = [
    "DependencyInstallPlan",
    "LPIPS_VERSION",
    "PYTORCH_CPU_INDEX",
    "PipInstallStep",
    "SYMPY_VERSION",
    "TORCH_CPU_VERSION",
    "TORCHVISION_CPU_VERSION",
    "blender_research_cpu_plan",
    "execute_install_plan",
    "print_install_plan",
]
