"""Content-scoped qualification of actual delivered Boolean geometry."""
from __future__ import annotations


def qualify_retained_output(data, options):
    """Never inherit operand certificates or assume surface components are bodies."""
    from .grouped_solids import solid_guard
    guard = solid_guard(data)
    python = options.get("native_qualification_python")
    if not python:
        return {"status": "unchecked_candidate", "geometry_content_hash": data.content_hash,
                "boundary_qualified": False, "single_solid_qualified": False,
                "solid_guard": guard, "reason": "no_existing_qualification_interpreter",
                "certificate_scope": "actual retained coordinates and connectivity; edits invalidate receipt"}
    from .native_qualification import qualify_geometry
    receipt = qualify_geometry(data, python=python,
                               timeout_s=float(options.get("native_qualification_timeout_s", 15.)),
                               ownership_root=options.get("native_run_ownership_root"))
    return {**receipt, "boundary_qualified": bool(receipt.get("manifold_validated")),
            "material_body_count": None,
            "certificate_scope": "actual retained coordinates and connectivity; edits invalidate receipt",
            "body_count_limitation": "surface component count does not establish material body count for cavities"}
