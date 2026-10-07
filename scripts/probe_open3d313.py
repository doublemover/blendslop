import sys,json,importlib.util,importlib.metadata as md
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]
from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()
import bpy
modules={'numpy':'numpy','dash':'dash','werkzeug':'Werkzeug','flask':'Flask','nbformat':'nbformat','configargparse':'ConfigArgParse','open3d':'open3d'}
result={'blender':bpy.app.version_string,'python':sys.version,'modules':{}}
for module,dist in modules.items():
    spec=importlib.util.find_spec(module)
    try:version=md.version(dist)
    except md.PackageNotFoundError:version=None
    result['modules'][module]={'available':spec is not None,'version':version,'origin':spec.origin if spec else None}
path=ROOT/'temp/improvement-phase-20261006/open3d313-proposal-preflight.json'
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
