"""Run context and determinism helpers for blocking workflows."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import random
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config import BlockingConfig


SCHEMA_VERSION = "blocktool_manifest_v1"


@dataclass
class StageTiming:
    """Timing information for a workflow stage."""

    stage: str
    elapsed_ms: float
    started_utc: str

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "stage": self.stage,
            "elapsed_ms": self.elapsed_ms,
            "started_utc": self.started_utc,
        }


@dataclass
class GenerationContext:
    """Execution context capturing determinism and manifest metadata."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seed: Optional[int] = None
    unit_scale: float = 0.01
    num_slices: int = 10
    reconstruction_mode: str = "legacy"
    dry_run: bool = False
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = SCHEMA_VERSION
    stages: List[StageTiming] = field(default_factory=list)
    logs: List[Dict[str, object]] = field(default_factory=list)
    config: Optional["BlockingConfig"] = None
    artifact_root: Optional[str] = None

    def apply_seed(self) -> None:
        """Seed RNGs for deterministic runs when a seed is provided."""
        if self.seed is None:
            return
        random.seed(self.seed)
        try:
            import numpy as np

            np.random.seed(self.seed)
        except ImportError:
            pass

    def log(self, level: str, message: str, **fields: Any) -> str:
        """Emit a structured log line tagged with the current run_id."""
        ordered_fields = " ".join(f"{key}={fields[key]}" for key in sorted(fields))
        line = f"[blocktool] run_id={self.run_id} level={level} msg={message}"
        if ordered_fields:
            line = f"{line} {ordered_fields}"
        print(line)
        self.logs.append({"level": level, "message": message, **fields})
        return line

    @contextlib.contextmanager
    def time_block(self, stage: str) -> Iterator[None]:
        """Context manager that records elapsed time for a stage."""
        start = time.perf_counter()
        started_utc = datetime.now(timezone.utc).isoformat()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.stages.append(
                StageTiming(stage=stage, elapsed_ms=elapsed_ms, started_utc=started_utc)
            )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict describing this context."""
        data: Dict[str, object] = {
            "run_id": self.run_id,
            "seed": self.seed,
            "unit_scale": self.unit_scale,
            "num_slices": self.num_slices,
            "reconstruction_mode": self.reconstruction_mode,
            "dry_run": self.dry_run,
            "created_utc": self.created_utc,
            "schema_version": self.schema_version,
        }
        if self.config is not None:
            data["config"] = self.config.to_dict()
        if self.artifact_root is not None:
            data["artifact_root"] = self.artifact_root
        try:
            json.dumps(data)
        except TypeError as exc:
            raise ValueError(
                "GenerationContext contains non-serializable values"
            ) from exc
        return data
