"""Existing Open3D CPU qualification helper; no installation or benchmark."""
from pathlib import Path
import argparse, json, hashlib, sys
import numpy as np


def main():
    import open3d as o3d
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--identity', required=True)
    args = p.parse_args()
    with np.load(args.input, allow_pickle=False) as values:
        vertices, faces = values['vertices'], values['faces']
        content_hash = str(values['content_hash'])
    connectivity = hashlib.sha256(str(len(vertices)).encode()+np.asarray(faces, np.int64).tobytes()).hexdigest()
    expected = hashlib.sha256(np.asarray(vertices, np.float64).tobytes()+connectivity.encode()).hexdigest()
    if expected != content_hash:
        raise ValueError('qualification geometry hash does not match actual input buffers')
    if len(faces) > 60000:
        raise ValueError('qualification exceeds the explicit 60000-triangle bound')
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    pairs = np.asarray(mesh.get_self_intersecting_triangles())
    parents = list(range(len(vertices)))
    def component(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for face in faces:
        for index in face[1:]:
            parents[component(int(index))] = component(int(face[0]))
    face_components = [component(int(face[0])) for face in faces]
    between = sum(face_components[int(a)] != face_components[int(b)] for a, b in pairs)
    provenance = [{'faces': [int(a), int(b)], 'left_component': face_components[int(a)],
                   'right_component': face_components[int(b)],
                   'relation': 'between_components' if face_components[int(a)] != face_components[int(b)] else 'within_component'}
                  for a, b in pairs[:20]]
    # Load only this numeric module; no Blender/image package imports in the helper.
    import importlib.util
    predicate_path = Path(__file__).resolve().parents[1]/'blender_blocking/evaluation/triangle_contacts.py'
    spec = importlib.util.spec_from_file_location('_blendslop_triangle_contacts', predicate_path)
    predicates = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predicates)
    verification = predicates.verify_reported_pairs(vertices, faces, pairs, max_pairs=1000)
    screened = bool(mesh.is_edge_manifold(allow_boundary_edges=False) and mesh.is_vertex_manifold()
                    and mesh.is_orientable() and verification['complete'] and verification['non_disjoint_pairs'] == 0)
    part_guard = predicates.within_part_boundary_guard(vertices, faces) if screened else {
        'status': 'not_required', 'passed': False, 'reason': 'reported contacts or topology already block qualification'}
    manifold = bool(screened and part_guard['passed'])
    receipt = {'geometry_content_hash': content_hash, 'toolchain_identity': args.identity,
               'manifold_validated': manifold,
               'self_intersections': verification['non_disjoint_pairs'] if verification['complete'] else None,
               'reported_self_intersections': int(len(pairs)), 'exact_pair_verification': verification,
               'within_part_boundary_guard': part_guard,
               'self_intersections_scope': 'exact non-disjoint reported triangle pairs; null when incomplete',
               'triangle_predicate_sha256': hashlib.sha256(predicate_path.read_bytes()).hexdigest(),
               'intersection_sample': pairs[:20].tolist(), 'intersection_pair_provenance': provenance,
               'between_component_pairs': int(between), 'within_component_pairs': int(len(pairs)-between),
               'backend': 'Open3D '+o3d.__version__,
               'helper_python': sys.version, 'helper_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'triangle_count': len(faces), 'qualification': 'Open3D edge/vertex/orientability screen plus exact verification of reported triangle pairs'}
    args.output.write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
