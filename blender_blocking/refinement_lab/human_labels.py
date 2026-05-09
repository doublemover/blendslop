"""Human review labels for refinement candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import json_safe, utc_now


ALLOWED_LABELS = {
    "sculptable",
    "useful_proxy",
    "too_blobby",
    "wrong_silhouette",
    "bad_topology",
    "overfit_silhouette",
    "underfit_silhouette",
    "reject",
}


@dataclass(frozen=True)
class HumanLabel:
    run_id: str
    case_id: str
    variant_id: str
    label: str
    score: int
    result_json: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""
    created_utc: str = ""
    schema_version: str = "human_label_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        if not self.created_utc:
            object.__setattr__(self, "created_utc", utc_now())
        self.validate()

    def validate(self) -> None:
        if not self.variant_id:
            raise ValueError("variant_id is required")
        if self.label not in ALLOWED_LABELS:
            raise ValueError(f"label must be one of {sorted(ALLOWED_LABELS)}")
        if not (1 <= int(self.score) <= 5):
            raise ValueError("score must be in [1, 5]")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "variant_id": self.variant_id,
            "result_json": self.result_json,
            "label": self.label,
            "score": int(self.score),
            "tags": list(self.tags),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HumanLabel":
        payload = dict(data)
        payload.pop("schema_version", None)
        payload["tags"] = tuple(payload.get("tags") or ())
        return cls(**payload)


def append_label(path: Path, label: HumanLabel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label.to_dict(), sort_keys=True, default=str) + "\n")


def load_labels(path: Path) -> list[HumanLabel]:
    if not Path(path).exists():
        return []
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            labels.append(HumanLabel.from_dict(json.loads(line)))
        except Exception:
            continue
    return labels


def labels_by_variant(labels: Sequence[HumanLabel | Mapping[str, Any]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for label in labels:
        payload = label.to_dict() if hasattr(label, "to_dict") else json_safe(label)
        variant_id = str(payload.get("variant_id", ""))
        grouped.setdefault(variant_id, []).append(payload)
    return grouped


def label_score(label: Mapping[str, Any]) -> float:
    weights = {
        "sculptable": 1.0,
        "useful_proxy": 0.7,
        "too_blobby": -0.4,
        "wrong_silhouette": -1.0,
        "bad_topology": -0.9,
        "overfit_silhouette": -0.35,
        "underfit_silhouette": -0.5,
        "reject": -2.0,
    }
    base = weights.get(str(label.get("label", "")), 0.0)
    try:
        score = float(label.get("score", 3)) / 5.0
    except (TypeError, ValueError):
        score = 0.6
    return base * score
