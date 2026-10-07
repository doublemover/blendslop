"""Process-owned, optional pinned CPU transport for signed-field jobs."""
import atexit
import os
from pathlib import Path

_SESSIONS = {}


def implicit_session(executable):
    from ..differentiable.helper_session import NumericHelperSession
    root = Path(__file__).resolve().parents[3]
    key = os.path.abspath(executable)
    if key not in _SESSIONS or _SESSIONS[key].closed:
        sources = [Path(__file__).with_name(name) for name in ('numeric_solver.py', 'field_model.py', 'mesh_objective.py')]
        sources.append(Path(__file__).resolve().parents[1]/'differentiable'/'projected_mesh_rays.py')
        _SESSIONS[key] = NumericHelperSession(key, root/'scripts/implicit_worker.py', source_paths=sources)
    return _SESSIONS[key]


def close_implicit_sessions():
    for session in list(_SESSIONS.values()):
        session.close()
    _SESSIONS.clear()


atexit.register(close_implicit_sessions)
