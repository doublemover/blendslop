"""One output mesh contract for evaluation, camera framing and validation."""
from __future__ import annotations


def output_mesh_targets(roots, *, artist_preview=False):
    """Keep intended multipart descendants and omit explicitly retained sources.

    Visibility is scene state, not output membership. Exclusion of an artist
    parent applies to its entire subtree, including unmarked mesh descendants.
    """
    meshes, seen = [], set()
    for root in roots:
        for obj in [root] + list(getattr(root, "children_recursive", ())):
            if getattr(obj, "type", None) != "MESH" or id(obj) in seen:
                continue
            if not artist_preview:
                ancestor, excluded, ancestry = obj, False, set()
                while ancestor is not None and id(ancestor) not in ancestry:
                    ancestry.add(id(ancestor))
                    if ancestor.get("blendslop_export_exclude", False):
                        excluded = True
                        break
                    ancestor = getattr(ancestor, "parent", None)
                if excluded:
                    continue
            seen.add(id(obj))
            meshes.append(obj)
    return meshes
