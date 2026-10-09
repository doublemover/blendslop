#!/usr/bin/env python3
"""Three exact saved reconstructed families, cosmetic color/lit presentation only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"blender_blocking"), str(ROOT), str(ROOT/"scripts")]
import test_runner
from run_surface_quality_check import _write
from run_family_source_edit_check import _compile_exact, _cleanup, _publish_receipt
from run_quality_coverage_check import save_mesh
from run_reference_neutral_inspection import shading_state


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _material(name, color):
    import bpy
    mat = bpy.data.materials.new(name+"SatinCeramic")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = .31
    shader.inputs["Metallic"].default_value = .05
    shader.inputs["Coat Weight"].default_value = .22
    shader.inputs["Coat Roughness"].default_value = .23
    coords = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 95.
    noise.inputs["Detail"].default_value = 2.
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*[v*.88 for v in color],1.)
    ramp.color_ramp.elements[1].color = (*color,1.)
    rough = nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value = 0.
    rough.inputs["From Max"].default_value = 1.
    rough.inputs["To Min"].default_value = .27
    rough.inputs["To Max"].default_value = .35
    links.new(coords.outputs["Generated"],noise.inputs["Vector"])
    links.new(noise.outputs["Fac"],ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"],shader.inputs["Base Color"])
    links.new(noise.outputs["Fac"],rough.inputs["Value"])
    links.new(rough.outputs["Result"],shader.inputs["Roughness"])
    links.new(shader.outputs["BSDF"],output.inputs["Surface"])
    return mat


def _material_record(material):
    def plain(value):
        if isinstance(value,(str,bool,int,float)):
            return value
        try:
            return list(value)
        except TypeError:
            return str(value)
    return {"name": material.name, "nodes": [{"name": n.name, "type": n.bl_idname,
        "inputs": {s.name: plain(s.default_value) for s in n.inputs if hasattr(s,"default_value")},
        **({"ramp": [{"position": e.position,"color": list(e.color)} for e in n.color_ramp.elements]} if hasattr(n,"color_ramp") else {})}
        for n in material.node_tree.nodes],
        "links": [{"from_node": l.from_node.name,"from_socket": l.from_socket.name,"to_node": l.to_node.name,"to_socket": l.to_socket.name}
        for l in material.node_tree.links], "displacement": "none", "bump": "none", "scope": "cosmetic generated-coordinate color/roughness only"}


def _stage(source, data, name, color):
    import bpy
    import numpy as np
    from mathutils import Vector
    scene = bpy.context.scene
    low, high = data.vertices.min(axis=0), data.vertices.max(axis=0)
    center, extent = (low+high)*.5, float(np.ptp(data.vertices,axis=0).max())
    owned = []
    floor_mat = bpy.data.materials.new(name+"MatteFloor")
    floor_mat.diffuse_color = (.71,.74,.79,1.)
    floor_mat.use_nodes = True
    floor_shader = floor_mat.node_tree.nodes.get("Principled BSDF")
    floor_shader.inputs["Base Color"].default_value = (.71,.74,.79,1.)
    floor_shader.inputs["Roughness"].default_value = .82
    bpy.ops.mesh.primitive_plane_add(size=extent*200,location=(center[0],center[1],low[2]-extent*.025))
    floor = bpy.context.object
    floor.name = name+"PresentationFloor"
    floor.data.materials.append(floor_mat)
    owned.append(floor)
    camera_data = bpy.data.cameras.new(name+"BeautyCamera")
    camera = bpy.data.objects.new(name+"BeautyCamera",camera_data)
    scene.collection.objects.link(camera)
    owned.append(camera)
    direction = (3.7,-5.8,8.) if name == "rounded_triangle_dot" else (3.7,-5.8,3.5)
    camera.location = Vector(center)+Vector(direction).normalized()*extent*4.
    camera.rotation_euler = (Vector(center)-camera.location).to_track_quat("-Z","Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = extent*1.55
    camera_data.clip_start, camera_data.clip_end = .01, extent*100.
    scene.camera = camera
    lights = []
    for label, direction, power, size, tint in (
            ("Key",(-3.,-4.,5.),1200.,3.5,(1.,.88,.76)),
            ("Fill",(4.,-1.,2.5),650.,4.,(.69,.83,1.)),
            ("Rim",(1.,4.,4.5),1700.,2.,(1.,.97,.89))):
        light_data = bpy.data.lights.new(name+label,"AREA")
        light_data.energy = power*extent**2
        light_data.shape = "DISK"
        light_data.size = size*extent
        light_data.color = tint
        light = bpy.data.objects.new(name+label,light_data)
        scene.collection.objects.link(light)
        light.location = Vector(center)+Vector(direction)*extent
        light.rotation_euler = (Vector(center)-light.location).to_track_quat("-Z","Y").to_euler()
        owned.append(light)
        lights.append({"name": light.name,"type": light_data.type,"energy": light_data.energy,
                       "size": light_data.size,"color": list(light_data.color),"matrix_world": [list(row) for row in light.matrix_world]})
    mat = _material(name,color)
    source.data.materials.clear()
    source.data.materials.append(mat)
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (.71,.74,.79,1.)
    background.inputs["Strength"].default_value = .35
    # Isolate presentation from the factory cube/camera/light and other rows.
    keep = {source,*owned}
    ancestor = source.parent
    while ancestor is not None:
        keep.add(ancestor)
        ancestor = ancestor.parent
    visibility = {obj: obj.hide_render for obj in scene.objects}
    for obj in scene.objects:
        if obj not in keep:
            obj.hide_render = True
    bpy.context.view_layer.update()
    lights = [{"name": light.name, "type": light.data.type, "energy": light.data.energy,
               "size": light.data.size, "color": list(light.data.color),
               "matrix_world": [list(row) for row in light.matrix_world]}
              for light in owned if light.type == "LIGHT"]
    return owned, visibility, {"material": _material_record(mat), "lights": lights,
        "floor": {"plane_z": floor.location.z,"roughness": .82,"size": extent*200},
        "world": {"color": [.71,.74,.79,1.],"strength": .35}}, camera


def main():
    import bpy
    import numpy as np
    from reconstruction.native_geometry import GeometryArrays,evaluated_arrays
    from evaluation.canonical_artifacts import camera_record,camera_frame_sha256
    from utils.run_ownership import OwnedRun
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-receipt",type=Path,required=True)
    parser.add_argument("--rounded-style-receipt",type=Path,required=True)
    parser.add_argument("--triangle-receipt",type=Path,required=True)
    parser.add_argument("--rounded-experiment",type=Path,
                        help="Explicit parent-selected admitted refined recipe; no legacy face reuse")
    parser.add_argument("--families",nargs="+",choices=("rounded_box","capsule","rounded_triangle_dot"),default=("rounded_box","capsule","rounded_triangle_dot"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if not args.output.resolve().is_relative_to(ROOT/"temp/tasks"):
        raise ValueError("new presentation output must stay below isolated temp/tasks")
    family = json.loads(args.family_receipt.read_text())
    box = json.loads(args.rounded_style_receipt.read_text())
    triangle = json.loads(args.triangle_receipt.read_text())
    rows = {"rounded_box": (args.rounded_style_receipt,box,args.rounded_style_receipt.parent),
            "capsule": (args.family_receipt,family["cases"]["capsule"],Path(family["cases"]["capsule"]["artifact_directory"])),
            "rounded_triangle_dot": (args.triangle_receipt,triangle,args.triangle_receipt.parent)}
    box_selection = {"source": "verified original corrected style", "fallback": "none; guard failure is reported"}
    if args.rounded_experiment is not None:
        experiment = json.loads(args.rounded_experiment.read_text())
        if experiment.get("crop_admitted") is not True:
            raise ValueError("parent-selected refined experiment is not admitted")
        folder = args.rounded_experiment.parent/"refined"
        refined = experiment["refined"]
        if json.loads((folder/"program.json").read_text()) != refined["program"]:
            raise ValueError("refined recipe changed from experiment receipt")
        rows["rounded_box"] = (args.rounded_experiment,
            {"geometry_hash": refined["geometry_hash"], "npz_sha256": _sha(folder/"evaluated-exact.npz"),
             "obj_sha256": None}, folder)
        box_selection = {"source": "explicit parent-selected admitted refined experiment", "fallback": "none; guard failure is reported",
                         "receipt": str(args.rounded_experiment.resolve()), "sha256": _sha(args.rounded_experiment),
                         "family_surface_limits": experiment.get("family_surface_limits"), "family_acceptance": "unqualified"}
    if len(set(args.families)) != len(args.families):
        raise ValueError("presentation families must be distinct")
    rows = {name: rows[name] for name in args.families}
    inputs = {}
    for name,(receipt,row,folder) in rows.items():
        if _sha(folder/"evaluated-exact.npz") != row["npz_sha256"] or (
                row["obj_sha256"] is not None and _sha(folder/"evaluated.obj") != row["obj_sha256"]):
            raise ValueError("saved presentation geometry changed: "+name)
        inputs[name] = {"receipt": str(receipt.resolve()),"receipt_sha256": _sha(receipt),
            "directory": str(folder.resolve()),"required_indexed_hash": row["geometry_hash"],
            "program_sha256": _sha(folder/"program.json"),"npz_sha256": row["npz_sha256"],"obj_sha256": row["obj_sha256"],
            "source_obj_status": "unavailable" if row["obj_sha256"] is None else "retained"}
    colors = {"rounded_box": [.43,.11,.065],"capsule": [.055,.24,.21],"rounded_triangle_dot": [.075,.28,.48]}
    frozen = {"protocol": "retained_families_cosmetic_presentation_v1","inputs": inputs,"colors_linear": colors,
        "box_selection": box_selection,
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in
            (Path(__file__),ROOT/"scripts/run_family_source_edit_check.py",ROOT/"scripts/run_reference_neutral_inspection.py",
             ROOT/"blender_blocking/primitives/shape_program_compiler.py",ROOT/"blender_blocking/primitives/rounded_triangle.py")},
        "frames": len(inputs),"resolution": [1536,1536],"engine": "BLENDER_EEVEE","render_samples": 64,
        "deadline_seconds": 120,"memory_limit_bytes": 8*1024**3,"threads": 2,
        "fits": 0,"measurements": 0,"raw_comparisons": 0,"qualification_children": 0,
        "displacement": "none","scope": "color/lit presentation only; no family acceptance or source inspection claim",
        "environment": {"blender": bpy.app.version_string,"python": sys.version,"numpy": np.__version__}}
    started = time.monotonic()
    owner = OwnedRun(args.output.resolve(),producer="retained_family_cosmetic_presentation",max_generated_bytes=67108864,
                     shared_inputs={name: item["receipt"] for name,item in inputs.items()})
    with owner:
        _write(owner.root/"frozen-workload.json",frozen)
        owner.register_file("frozen-workload.json","diagnostic")
        receipt = {"protocol": frozen["protocol"],"status": "running","cases": {},"aggregate_accepted": False}
        _publish_receipt(owner,receipt)
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE"
        if not hasattr(scene.eevee,"taa_render_samples"):
            raise RuntimeError("native EEVEE render sample control unavailable; no substitute contract")
        scene.eevee.taa_render_samples = 64
        scene.render.resolution_x = scene.render.resolution_y = 1536
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.film_transparent = False
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.exposure = 0.
        scene.view_settings.gamma = 1.
        for name,item in inputs.items():
            print("presentation "+name,flush=True)
            folder = owner.root/name
            folder.mkdir()
            compiled, owned, visibility = None, [], {}
            try:
                if time.monotonic()-started > 120:
                    raise TimeoutError("presentation deadline exceeded")
                origin = Path(item["directory"])
                with np.load(origin/"evaluated-exact.npz",allow_pickle=False) as archive:
                    captured = GeometryArrays.capture(archive["vertices"],archive["faces"])
                wire = json.loads((origin/"program.json").read_text())
                compiled, source, before = _compile_exact(wire,captured)
                if before.content_hash != item["required_indexed_hash"]:
                    raise ValueError("saved recipe cannot reproduce exact indexed geometry")
                _write(folder/"program.json",wire)
                _,hashes = save_mesh(source,folder)
                style = shading_state(source)
                owned,visibility,details,camera = _stage(source,before,name,colors[name])
                record = camera_record(camera,resolution=(1536,1536),
                    pixel_aspect=(scene.render.pixel_aspect_x,scene.render.pixel_aspect_y))
                path = folder/"beauty.png"
                scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)
                after = evaluated_arrays(source)
                if after.content_hash != before.content_hash:
                    raise ValueError("cosmetic preview changed candidate geometry")
                actual = camera_record(camera,resolution=(1536,1536),
                    pixel_aspect=(scene.render.pixel_aspect_x,scene.render.pixel_aspect_y))
                if camera_frame_sha256(record,resolution=(1536,1536)) != camera_frame_sha256(actual,resolution=(1536,1536)):
                    raise ValueError("presentation render changed camera")
                receipt["cases"][name] = {"status": "display_preview_retained",**hashes,
                    "geometry_unchanged": True,"camera": actual,"camera_sha256": camera_frame_sha256(actual,resolution=(1536,1536)),
                    "beauty": {"path": str(path),"sha256": _sha(path),"resolution": [1536,1536],
                        "geometry_hash": before.content_hash,"camera_sha256": camera_frame_sha256(actual,resolution=(1536,1536)),
                        "geometry_unchanged_after_render": True,"binding": "verified_by_producer"},
                    "render_settings": {"engine": scene.render.engine,"samples": scene.eevee.taa_render_samples,
                        "view_transform": scene.view_settings.view_transform,"look": scene.view_settings.look,
                        "exposure": scene.view_settings.exposure,"gamma": scene.view_settings.gamma,"color_mode": "RGB"},
                    "presentation": details,"retained_source_style": style,
                    "metric_admission": "none; separate cosmetic presentation", "artifact_directory": str(folder)}
                _write(folder/"presentation-provenance.json",receipt["cases"][name])
            except Exception as exc:
                receipt["cases"][name] = {"status": "blocked", "reason": type(exc).__name__+": "+str(exc),
                    "required_indexed_hash": item["required_indexed_hash"],"presentation_substitute": "none"}
            finally:
                for obj,value in visibility.items():
                    if obj.name in bpy.data.objects:
                        obj.hide_render = value
                scene.camera = None
                for obj in reversed(owned):
                    if obj.name in bpy.data.objects:
                        mesh = obj.data if obj.type=="MESH" else None
                        bpy.data.objects.remove(obj,do_unlink=True)
                        if mesh is not None and mesh.users==0:
                            bpy.data.meshes.remove(mesh)
                if compiled is not None:
                    _cleanup(compiled)
            _publish_receipt(owner,receipt)
            for path in folder.iterdir():
                if path.is_file():
                    owner.register_file(path.relative_to(owner.root),"diagnostic" if path.suffix==".png" else "final_output")
        receipt.update(status="display_previews_retained" if all(row["status"] == "display_preview_retained" for row in receipt["cases"].values()) else "required_preview_blocked",elapsed_seconds=time.monotonic()-started,run_root=str(owner.root),
            retained_frames=sum(r["status"]=="display_preview_retained" for r in receipt["cases"].values()))
        _publish_receipt(owner,receipt)
        print("PRESENTATION_RESULT="+str(owner.root/"results.json"),flush=True)
    return 1 if any(r["status"]=="blocked" for r in receipt["cases"].values()) else 0


if __name__=="__main__":
    raise SystemExit(main())
