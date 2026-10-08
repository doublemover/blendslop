"""Analytic ellipses and continuous projected-union contour proposals."""
from copy import deepcopy
import hashlib
import json

import numpy as np

from .contour_silhouette import contour_footprint


def is_ellipse(part):
    return type(part).__name__ in {'EllipsoidPrimitive', 'AnisotropicGaussianPrimitive'}


def mesh_footprint(part, camera, softness=24., resolution=None):
    return contour_footprint(part, camera, softness, resolution)


class ShapeAwareSilhouetteCache:
    def __init__(self, cameras, softness=24., min_variance=1e-6):
        from .soft_silhouette import ProjectedSilhouetteCache
        self.cameras = tuple(cameras)
        if len({camera.name for camera in self.cameras}) != len(self.cameras):
            raise ValueError('shape-aware camera names must be unique')
        self.softness, self.min_variance = softness, min_variance
        self.ellipses = ProjectedSilhouetteCache(self.cameras, softness, min_variance)
        self.entries = {}
        self.generation = 0
        self.signature = None
        self.recomputed_components = self.reused_components = 0
        self.derivative_evaluations = 0
        self.derivative_cursor = 0
        self.derivative_failures = []
        self.approximation_reports = {}

    def render(self, parts):
        self.parts = list(parts)
        self.ellipse_ids = [i for i, p in enumerate(self.parts) if is_ellipse(p)]
        ellipse_parts = [self.parts[i] for i in self.ellipse_ids]
        self.ellipse_masks = self.ellipses.render(ellipse_parts)
        self.transmittance, self.component_masks = {}, {}
        output, keys, reports = {}, [], {}
        for camera in self.cameras:
            transmittance = np.ones(camera.image_size[::-1])
            components = {}
            for index, part in enumerate(self.parts):
                if is_ellipse(part):
                    continue
                payload = part.to_dict() if hasattr(part, 'to_dict') else vars(part)
                signature = hashlib.sha256(json.dumps(
                    payload, sort_keys=True,
                    default=lambda value: np.asarray(value).tolist()).encode()).hexdigest()
                key = (camera.name, tuple(camera.axes), tuple(camera.image_size),
                       tuple(camera.world_bounds), float(self.softness), signature)
                keys.append(key)
                if key not in self.entries:
                    self.entries[key] = contour_footprint(
                        part, camera, self.softness, return_metadata=True)
                    self.recomputed_components += 1
                else:
                    self.reused_components += 1
                alpha, report = self.entries[key]
                components[index] = alpha
                reports[(camera.name, index)] = report
                transmittance *= 1-alpha
            self.transmittance[camera.name] = transmittance
            self.component_masks[camera.name] = components
            output[camera.name] = 1-(1-self.ellipse_masks[camera.name])*transmittance
        signature = tuple(keys), self.ellipses.generation
        if signature != self.signature:
            self.generation += 1
            self.signature = signature
        if len(self.entries) > 128:
            self.entries = {key: self.entries[key] for key in keys}
        self.approximation_reports = reports
        return output

    def _ellipse_world_gradients(self, gradients):
        weighted = {name: value*self.transmittance[name]
                    for name, value in gradients.items()}
        return self.ellipses.backward(weighted)

    def backward(self, gradients):
        if len(self.ellipse_ids) != len(self.parts):
            raise ValueError('nonellipse contours require family parameter pullbacks, not ellipse covariance gradients')
        return self._ellipse_world_gradients(gradients)

    def parameter_gradients(self, gradients, bounds, *, budget=None,
                            maximum_controls=8, difference_step=1e-4):
        """Analytic ellipse gradients plus bounded local contour differences.

        The numerical family fallback contracts the mask derivative before any
        hard admission gate. Active contour/topology changes can be nonsmooth.
        Perturbations consume the same objective allowance as proposal scoring.
        A rotating prefix visits all family controls across successive calls.
        """
        from blender_blocking.placement.resfit_objective import ResFitObjectiveResult
        from blender_blocking.placement.resfit_parameters import (
            apply_parameter_increment, discover_primitive_parameters,
            pullback_render_gradients,
        )
        if not np.isfinite(difference_step) or difference_step <= 0:
            raise ValueError('contour difference step must be finite and positive')
        if not isinstance(maximum_controls, int) or maximum_controls <= 0:
            raise ValueError('contour maximum controls must be a positive integer')
        for camera in self.cameras:
            values = np.asarray(gradients.get(camera.name, np.zeros(camera.image_size[::-1])), float)
            if values.shape != camera.image_size[::-1] or not np.isfinite(values).all():
                raise ValueError('contour mask gradients must be finite and match each camera')
        complete_gradients = {camera.name: np.asarray(
            gradients.get(camera.name, np.zeros(camera.image_size[::-1])), float)
            for camera in self.cameras}
        ellipse_parts = [self.parts[i] for i in self.ellipse_ids]
        analytic = pullback_render_gradients(ellipse_parts,
                                            *self._ellipse_world_gradients(complete_gradients))
        result = {(self.ellipse_ids[index], attr, axis): value
                  for (index, attr, axis), value in analytic.items()}
        refs = [ref for ref in discover_primitive_parameters(self.parts)
                if ref[0] not in self.ellipse_ids]
        if not refs:
            return result
        start = self.derivative_cursor % len(refs)
        refs = refs[start:]+refs[:start]
        completed = 0
        for ref in refs[:maximum_controls]:
            # Leave one score slot for an actual admitted proposal. Never start
            # one side of a pair when its mate has no count allowance.
            remaining = None if budget is None else budget.remaining_evaluations()
            if budget is not None and (budget.reason() is not None
                                      or (remaining is not None and remaining < 3)):
                break
            index = ref[0]
            other_transmittance = {}
            for camera in self.cameras:
                value = 1-self.ellipse_masks[camera.name]
                for other, alpha in self.component_masks[camera.name].items():
                    if other != index:
                        value = value*(1-alpha)
                other_transmittance[camera.name] = value

            def contraction(items):
                self.derivative_evaluations += 1
                total = 0.
                for camera in self.cameras:
                    resolution = self.approximation_reports[(camera.name, index)]['resolution']
                    alpha = contour_footprint(items[index], camera, self.softness, resolution)
                    total += float(np.sum(complete_gradients[camera.name]
                                          *other_transmittance[camera.name]*alpha))
                return total

            try:
                values = []
                for sign in (1., -1.):
                    trial = deepcopy(self.parts)
                    apply_parameter_increment(trial, ref, sign*difference_step, bounds)
                    if budget is None:
                        values.append(contraction(trial))
                    else:
                        scored = budget.evaluate(
                            lambda items: ResFitObjectiveResult(contraction(items),
                                {"scope": "contour derivative contraction; no admission"}), trial)
                        values.append(scored.total)
                result[ref] = (values[0]-values[1])/(2*difference_step)
            except Exception as exc:
                self.derivative_failures.append(f'{ref}: {type(exc).__name__}: {exc}')
                if budget is not None and budget.reason() is not None:
                    break
            completed += 1
        self.derivative_cursor = (start+completed) % len(refs)
        return result

    def render_coordinate_batch(self, base, candidates):
        raise ValueError('contour footprints use independent cached coordinate evaluations')
