"""Lightweight learned surrogate for refinement-lab prioritization."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import ExperimentResult, ExperimentVariant, json_safe
from .parameter_search import score_result


DEFAULT_BASE_FEATURES = {
    "result.pass": 0.0,
    "result.status_pass": 0.0,
    "metric.average_iou": 0.0,
    "metric.min_view_iou": 0.0,
    "metric.front_iou": 0.0,
    "metric.side_iou": 0.0,
    "metric.top_iou": 0.0,
    "metric.boundary_iou_mean": 0.0,
    "metric.topology_score": 0.0,
    "metric.editability_score": 0.0,
    "metric.cost_combined_total_wall_ms": 0.0,
    "metric.elapsed_s": 0.0,
}


@dataclass(frozen=True)
class SurrogateTrainingExample:
    result: ExperimentResult
    target: float
    features: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.result.run_id,
            "case_id": self.result.case_id,
            "variant_id": self.result.variant_id,
            "mode": self.result.mode,
            "target": self.target,
            "features": dict(self.features),
        }


@dataclass(frozen=True)
class SurrogatePrediction:
    variant_id: str
    mode: str
    predicted_score: float
    uncertainty: float
    rank: int = 0
    feature_contributions: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "mode": self.mode,
            "predicted_score": self.predicted_score,
            "uncertainty": self.uncertainty,
            "rank": self.rank,
            "feature_contributions": dict(self.feature_contributions),
        }


@dataclass(frozen=True)
class SurrogateModel:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    feature_means: Mapping[str, float]
    feature_scales: Mapping[str, float]
    objective: str
    example_count: int
    residual_rmse: float
    residual_mae: float
    target_mean: float
    target_std: float
    ridge: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def predict_features(self, features: Mapping[str, float]) -> SurrogatePrediction:
        vector = _standardized_vector(features, self.feature_names, self.feature_means, self.feature_scales)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        predicted = float(np.dot(vector, coefficients) + float(self.intercept))
        contributions = {
            name: float(value * coefficients[index])
            for index, (name, value) in enumerate(zip(self.feature_names, vector))
            if abs(float(value * coefficients[index])) > 1e-9
        }
        uncertainty = float(
            max(
                self.residual_rmse,
                _distance_uncertainty(vector) * max(1.0, self.target_std) * 0.1,
            )
        )
        return SurrogatePrediction(
            variant_id=str(features.get("variant_id", "")),
            mode=str(features.get("mode", "")),
            predicted_score=predicted,
            uncertainty=uncertainty,
            feature_contributions=contributions,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "refinement_surrogate_model_v1",
            "objective": self.objective,
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "feature_means": dict(self.feature_means),
            "feature_scales": dict(self.feature_scales),
            "example_count": self.example_count,
            "residual_rmse": self.residual_rmse,
            "residual_mae": self.residual_mae,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "ridge": self.ridge,
            "diagnostics": json_safe(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SurrogateModel":
        payload = dict(data)
        payload.pop("schema_version", None)
        payload["feature_names"] = tuple(str(name) for name in payload["feature_names"])
        payload["coefficients"] = tuple(float(value) for value in payload["coefficients"])
        payload["feature_means"] = {
            str(key): float(value) for key, value in payload.get("feature_means", {}).items()
        }
        payload["feature_scales"] = {
            str(key): float(value) for key, value in payload.get("feature_scales", {}).items()
        }
        return cls(**payload)


def build_training_examples(
    results: Iterable[ExperimentResult],
    *,
    objective: str = "quality_win",
    variants_by_id: Mapping[str, ExperimentVariant] | None = None,
) -> tuple[SurrogateTrainingExample, ...]:
    """Build scored surrogate examples from completed refinement results."""

    examples = []
    variants = variants_by_id or {}
    for result in results:
        try:
            target = float(score_result(result, objective=objective)["total"])
        except Exception:
            continue
        variant = variants.get(result.variant_id)
        features = (
            features_from_variant(variant)
            if variant is not None
            else features_from_result(result)
        )
        examples.append(SurrogateTrainingExample(result, target, features))
    return tuple(examples)


def train_surrogate(
    results: Iterable[ExperimentResult],
    *,
    objective: str = "quality_win",
    variants_by_id: Mapping[str, ExperimentVariant] | None = None,
    ridge: float = 1e-6,
) -> SurrogateModel:
    """Train a deterministic ridge-regression surrogate over result features."""

    examples = build_training_examples(
        results,
        objective=objective,
        variants_by_id=variants_by_id,
    )
    if not examples:
        return SurrogateModel(
            feature_names=(),
            coefficients=(),
            intercept=0.0,
            feature_means={},
            feature_scales={},
            objective=objective,
            example_count=0,
            residual_rmse=0.0,
            residual_mae=0.0,
            target_mean=0.0,
            target_std=0.0,
            ridge=float(ridge),
            diagnostics={"status": "empty_training_set"},
        )

    feature_names = _feature_names(examples)
    matrix = np.asarray(
        [_feature_vector(example.features, feature_names) for example in examples],
        dtype=np.float64,
    )
    target = np.asarray([example.target for example in examples], dtype=np.float64)
    means = matrix.mean(axis=0) if matrix.size else np.zeros(len(feature_names))
    scales = matrix.std(axis=0) if matrix.size else np.ones(len(feature_names))
    scales = np.where(scales > 1e-9, scales, 1.0)
    standardized = (matrix - means) / scales
    design = np.column_stack([np.ones(len(examples)), standardized])
    regularizer = np.eye(design.shape[1], dtype=np.float64) * max(0.0, float(ridge))
    regularizer[0, 0] = 0.0
    try:
        weights = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(design.T @ design + regularizer) @ design.T @ target
    fitted = design @ weights
    residuals = target - fitted
    return SurrogateModel(
        feature_names=feature_names,
        coefficients=tuple(float(value) for value in weights[1:]),
        intercept=float(weights[0]),
        feature_means={name: float(means[index]) for index, name in enumerate(feature_names)},
        feature_scales={name: float(scales[index]) for index, name in enumerate(feature_names)},
        objective=objective,
        example_count=len(examples),
        residual_rmse=float(math.sqrt(float(np.mean(residuals**2)))) if residuals.size else 0.0,
        residual_mae=float(np.mean(np.abs(residuals))) if residuals.size else 0.0,
        target_mean=float(np.mean(target)) if target.size else 0.0,
        target_std=float(np.std(target)) if target.size else 0.0,
        ridge=float(ridge),
        diagnostics={
            "status": "trained",
            "feature_count": len(feature_names),
            "target_min": float(np.min(target)) if target.size else 0.0,
            "target_max": float(np.max(target)) if target.size else 0.0,
        },
    )


def features_from_result(
    result: ExperimentResult,
    *,
    variant: ExperimentVariant | None = None,
) -> Mapping[str, float]:
    """Extract numeric features from a result and optional variant definition."""

    features = dict(DEFAULT_BASE_FEATURES)
    features.update(
        {
            "result.pass": 1.0 if result.passed else 0.0,
            "result.status_pass": 1.0 if result.status == "pass" else 0.0,
            "result.exit_code": float(result.exit_code),
            "metric.average_iou": _float(result.avg_iou),
            "metric.min_view_iou": _float(result.min_iou),
            "metric.front_iou": _float(result.view_iou("front")),
            "metric.side_iou": _float(result.view_iou("side")),
            "metric.top_iou": _float(result.view_iou("top")),
            "metric.elapsed_s": _float(result.elapsed_s),
            "warning.count": float(len(result.warnings)),
            "error.count": float(len(result.errors)),
        }
    )
    for key, value in result.metrics.items():
        if isinstance(value, (int, float, bool)):
            features[f"metric.{key}"] = _float(value)
    backend = result.backend_result if isinstance(result.backend_result, Mapping) else {}
    _flatten_numeric("backend", backend, features, max_depth=3)
    features[f"mode.{result.mode}"] = 1.0
    features[f"status.{result.status}"] = 1.0
    if variant is not None:
        features.update(features_from_variant(variant))
    return features


def features_from_variant(variant: ExperimentVariant) -> Mapping[str, float]:
    """Extract pre-run variant/config features for surrogate prioritization."""

    features: dict[str, float] = {
        f"mode.{variant.mode}": 1.0,
        f"validation_mode.{variant.validation_mode}": 1.0,
        "variant.diagnostic_only": 1.0 if variant.diagnostic_only else 0.0,
        "variant.cli_arg_count": float(len(variant.cli_args)),
        "variant.tag_count": float(len(variant.tags)),
    }
    for tag in variant.tags:
        features[f"tag.{tag}"] = 1.0
    _flatten_numeric("param", variant.parameters, features, max_depth=4)
    _flatten_numeric("config", variant.config_overrides, features, max_depth=4)
    return features


def prioritize_variants(
    model: SurrogateModel,
    variants: Sequence[ExperimentVariant],
    *,
    base_features: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> tuple[SurrogatePrediction, ...]:
    """Predict and rank variants before expensive reconstruction runs."""

    base = {
        name: float(model.feature_means.get(name, value))
        for name, value in DEFAULT_BASE_FEATURES.items()
    }
    base.update(base_features or {})
    predictions = []
    for variant in variants:
        features = {**base, **features_from_variant(variant)}
        prediction = model.predict_features(features)
        predictions.append(
            SurrogatePrediction(
                variant_id=variant.variant_id,
                mode=variant.mode,
                predicted_score=prediction.predicted_score,
                uncertainty=prediction.uncertainty,
                feature_contributions=prediction.feature_contributions,
            )
        )
    sorted_predictions = sorted(
        predictions,
        key=lambda item: (item.predicted_score - item.uncertainty, item.predicted_score),
        reverse=True,
    )
    if top_k is not None:
        sorted_predictions = sorted_predictions[: max(1, int(top_k))]
    return tuple(
        SurrogatePrediction(
            variant_id=item.variant_id,
            mode=item.mode,
            predicted_score=item.predicted_score,
            uncertainty=item.uncertainty,
            rank=index,
            feature_contributions=item.feature_contributions,
        )
        for index, item in enumerate(sorted_predictions, start=1)
    )


def surrogate_report(
    results: Iterable[ExperimentResult],
    variants: Sequence[ExperimentVariant],
    *,
    objective: str = "quality_win",
    ridge: float = 1e-6,
    top_k: int | None = None,
) -> dict[str, object]:
    """Train a surrogate and return a JSON-ready prioritization report."""

    variants_by_id = {variant.variant_id: variant for variant in variants}
    model = train_surrogate(
        results,
        objective=objective,
        variants_by_id=variants_by_id,
        ridge=ridge,
    )
    predictions = prioritize_variants(model, variants, top_k=top_k)
    return {
        "schema_version": "refinement_surrogate_report_v1",
        "objective": objective,
        "model": model.to_dict(),
        "prediction_count": len(predictions),
        "predictions": [prediction.to_dict() for prediction in predictions],
    }


def _feature_names(examples: Sequence[SurrogateTrainingExample]) -> tuple[str, ...]:
    names: set[str] = set()
    for example in examples:
        names.update(example.features)
    names.discard("variant_id")
    names.discard("mode")
    return tuple(sorted(names))


def _feature_vector(features: Mapping[str, float], names: Sequence[str]) -> list[float]:
    return [_float(features.get(name)) for name in names]


def _standardized_vector(
    features: Mapping[str, float],
    names: Sequence[str],
    means: Mapping[str, float],
    scales: Mapping[str, float],
) -> np.ndarray:
    values = []
    for name in names:
        scale = float(scales.get(name, 1.0) or 1.0)
        if abs(scale) <= 1e-9:
            scale = 1.0
        values.append((_float(features.get(name)) - float(means.get(name, 0.0))) / scale)
    return np.asarray(values, dtype=np.float64)


def _flatten_numeric(
    prefix: str,
    value: Any,
    features: dict[str, float],
    *,
    max_depth: int,
) -> None:
    if max_depth <= 0:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten_numeric(f"{prefix}.{key}", item, features, max_depth=max_depth - 1)
        return
    if isinstance(value, (list, tuple)):
        numeric = [_float(item, default=math.nan) for item in value]
        finite = [item for item in numeric if math.isfinite(item)]
        if finite:
            features[f"{prefix}.mean"] = float(sum(finite) / len(finite))
            features[f"{prefix}.count"] = float(len(finite))
        return
    if isinstance(value, bool):
        features[prefix] = 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        features[prefix] = _float(value)
    elif isinstance(value, str) and len(value) <= 64:
        features[f"{prefix}.{value}"] = 1.0


def _distance_uncertainty(vector: np.ndarray) -> float:
    if vector.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(vector))))


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)
