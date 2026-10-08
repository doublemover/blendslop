"""External benchmark and SOTA comparison metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .schemas import json_safe


@dataclass(frozen=True)
class ExternalBaselineRecord:
    name: str
    method_family: str
    source_url: str
    paper_year: int | None
    input_assumptions: tuple[str, ...]
    output_representation: str
    editable_output: bool
    metrics_reported: Mapping[str, float | str] = field(default_factory=dict)
    reproduced_locally: bool = False
    local_command: str | None = None
    task_mismatch_notes: tuple[str, ...] = ()
    citation: str = ""
    evidence_level: str = "reported"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "method_family": self.method_family,
            "source_url": self.source_url,
            "paper_year": self.paper_year,
            "input_assumptions": list(self.input_assumptions),
            "output_representation": self.output_representation,
            "editable_output": self.editable_output,
            "metrics_reported": json_safe(self.metrics_reported),
            "reproduced_locally": self.reproduced_locally,
            "local_command": self.local_command,
            "task_mismatch_notes": list(self.task_mismatch_notes),
            "citation": self.citation,
            "evidence_level": self.evidence_level,
        }


DEFAULT_EXTERNAL_BASELINES = (
    ExternalBaselineRecord(
        "Visual Hull",
        "shape_from_silhouette",
        "https://www.researchgate.net/publication/3192242_The_Visual_Hull_Concept_for_Silhouette-Based_Image_Understanding",
        1994,
        ("calibrated silhouettes",),
        "visual hull volume/mesh",
        True,
        task_mismatch_notes=("directly relevant to silhouette-only recoverability",),
    ),
    ExternalBaselineRecord(
        "NeRF",
        "radiance_field",
        "https://arxiv.org/abs/2003.08934",
        2020,
        ("posed RGB images", "many views"),
        "radiance field",
        False,
        task_mismatch_notes=("strong image metric baseline, not an editable mesh baseline",),
    ),
    ExternalBaselineRecord(
        "3D Gaussian Splatting",
        "radiance_splats",
        "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/",
        2023,
        ("posed RGB images", "SfM initialization"),
        "gaussian splats",
        False,
        task_mismatch_notes=("must be distilled to mesh before editable comparison",),
    ),
)
