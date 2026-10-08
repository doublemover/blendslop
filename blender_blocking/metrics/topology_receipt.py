"""Reusable index-only diagnostics bound to actual triangle connectivity.

This is an in-process reuse contract, not a native-solid qualification receipt.
Coordinate changes do not invalidate index-only topology. Geometric degeneracy,
orientation and intersection checks must still evaluate the actual coordinates.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from .topology import TopologyReport, mesh_topology_report

TOPOLOGY_EVALUATOR = "index_connectivity_v1"


def connectivity_hash(vertex_count, faces):
    faces = np.asarray(faces, dtype=np.int64, order="C")
    return hashlib.sha256(str(int(vertex_count)).encode() + faces.tobytes()).hexdigest()


@dataclass(frozen=True)
class ConnectivityTopologyReceipt:
    evaluator: str
    connectivity_hash: str
    report: TopologyReport

    @classmethod
    def capture(cls, vertices, faces):
        faces = np.asarray(faces, dtype=np.int64, order="C")
        report = mesh_topology_report(vertices, faces)
        return cls(TOPOLOGY_EVALUATOR, connectivity_hash(len(vertices), faces), report)

    def matches(self, data):
        return (self.evaluator == TOPOLOGY_EVALUATOR
                and self.connectivity_hash == data.connectivity_hash
                and self.report.vertex_count == len(data.vertices)
                and self.report.face_count == len(data.faces))

    def identity(self):
        return {"evaluator": self.evaluator,
                "connectivity_hash": self.connectivity_hash,
                "scope": "index connectivity only; no geometric-solid qualification"}

    def summary(self):
        result = self.report.to_dict()
        result.update(topology_style="mesh", has_faces=bool(self.report.face_count))
        return result


def topology_for_geometry(data, receipt=None):
    if isinstance(receipt, ConnectivityTopologyReceipt) and receipt.matches(data):
        return receipt.report.to_dict(), True
    return mesh_topology_report(data.vertices, data.faces).to_dict(), False
