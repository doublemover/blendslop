"""Shared silhouette rendering utilities for Blender."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import bpy
    from mathutils import Vector

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    Vector = None
    BLENDER_AVAILABLE = False


@dataclass
class RenderSettingsSnapshot:
    resolution_x: int
    resolution_y: int
    resolution_percentage: int
    file_format: str
    color_mode: str
    engine: str
    film_transparent: bool


@dataclass
class SilhouetteRenderSession:
    scene: bpy.types.Scene
    target_objects: List[bpy.types.Object]
    camera: bpy.types.Object
    light: Optional[bpy.types.Object]
    created_camera: bool
    created_light: bool
    original_camera: Optional[bpy.types.Object]
    render_settings: RenderSettingsSnapshot
    hidden_objects: Dict[bpy.types.Object, bool]
    original_materials: Dict[bpy.types.Object, List[Optional[bpy.types.Material]]]
    world: Optional[bpy.types.World]
    created_world: bool
    background_node: Optional[bpy.types.Node]
    background_created: bool
    background_color: Optional[Tuple[float, float, float, float]]
    world_use_nodes: Optional[bool]
    silhouette_material: bpy.types.Material
    extra_lights: List[bpy.types.Object]

    def restore(self) -> None:
        """Restore scene state and cleanup temporary objects."""
        if not BLENDER_AVAILABLE:
            return

        for obj, prev in self.hidden_objects.items():
            if hasattr(obj, "hide_render"):
                obj.hide_render = prev

        for obj, mats in self.original_materials.items():
            if getattr(obj, "type", None) != "MESH":
                continue
            obj.data.materials.clear()
            for mat in mats:
                if mat is None:
                    continue
                obj.data.materials.append(mat)

        if self.world is not None:
            if self.background_node is not None and self.background_color is not None:
                self.background_node.inputs[0].default_value = self.background_color
            if self.background_created and self.background_node is not None:
                try:
                    self.world.node_tree.nodes.remove(self.background_node)
                except Exception:
                    pass
            if self.world_use_nodes is not None:
                self.world.use_nodes = self.world_use_nodes

        if self.created_world and self.world is not None:
            if self.scene.world == self.world:
                self.scene.world = None
            try:
                bpy.data.worlds.remove(self.world, do_unlink=True)
            except Exception:
                pass

        # Always clear or restore the scene camera before deleting a temporary
        # camera. Blender can otherwise leave scene.camera pointing at a removed
        # object, causing the next headless render to produce an empty frame.
        self.scene.camera = self.original_camera

        if self.created_light and self.light is not None:
            bpy.data.objects.remove(self.light, do_unlink=True)

        if self.created_camera and self.camera is not None:
            bpy.data.objects.remove(self.camera, do_unlink=True)

        for light in self.extra_lights:
            if light is None:
                continue
            try:
                bpy.data.objects.remove(light, do_unlink=True)
            except Exception:
                pass

        rs = self.render_settings
        self.scene.render.resolution_x = rs.resolution_x
        self.scene.render.resolution_y = rs.resolution_y
        self.scene.render.resolution_percentage = rs.resolution_percentage
        self.scene.render.image_settings.file_format = rs.file_format
        self.scene.render.image_settings.color_mode = rs.color_mode
        self.scene.render.engine = rs.engine
        self.scene.render.film_transparent = rs.film_transparent
        _update_view_layer()


def collect_target_objects(
    scene: bpy.types.Scene,
    target_objects: Optional[Iterable[bpy.types.Object]] = None,
) -> List[bpy.types.Object]:
    """Collect renderable mesh targets."""
    if not BLENDER_AVAILABLE:
        return []
    if target_objects is None:
        mesh_objects = [obj for obj in scene.objects if obj.type == "MESH"]
        tagged = [obj for obj in mesh_objects if obj.get("blocktool_role") == "final"]
        return tagged if tagged else mesh_objects

    return [obj for obj in target_objects if getattr(obj, "type", None) == "MESH"]


def _mesh_objects(scene: bpy.types.Scene) -> List[bpy.types.Object]:
    """Return all mesh objects in the scene."""
    return [obj for obj in scene.objects if getattr(obj, "type", None) == "MESH"]


def _update_view_layer() -> None:
    """Flush visibility and material changes before render or after restore."""
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass


def ensure_world_background(
    scene: bpy.types.Scene, color: Tuple[float, float, float, float]
) -> Tuple[
    Optional[bpy.types.World],
    bool,
    Optional[bpy.types.Node],
    bool,
    Optional[Tuple[float, float, float, float]],
    Optional[bool],
]:
    """Ensure a background node exists, returning original state for restoration."""
    if scene.world is None:
        world = bpy.data.worlds.new(name="BlocktoolWorld")
        scene.world = world
        created_world = True
    else:
        world = scene.world
        created_world = False

    world_use_nodes = world.use_nodes
    world.use_nodes = True
    nodes = world.node_tree.nodes
    background_node = nodes.get("Background")
    background_created = False
    if background_node is None:
        background_node = nodes.new(type="ShaderNodeBackground")
        background_created = True

    original_color = tuple(background_node.inputs[0].default_value)
    background_node.inputs[0].default_value = color
    return (
        world,
        created_world,
        background_node,
        background_created,
        original_color,
        world_use_nodes,
    )


def ensure_silhouette_material(
    name: str = "Blocktool_Silhouette",
    color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    strength: float = 1.0,
) -> bpy.types.Material:
    """Create or reuse a black emission material for silhouettes."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def _create_party_materials() -> List[bpy.types.Material]:
    presets = [
        ("Blocktool_Party_Disco", "principled", {"metallic": 1.0, "roughness": 0.2}),
        ("Blocktool_Party_Diamond", "principled", {"metallic": 0.0, "roughness": 0.05}),
        ("Blocktool_Party_Glass", "glass", {"roughness": 0.0}),
    ]
    mats: List[bpy.types.Material] = []
    for name, kind, params in presets:
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output = nodes.new(type="ShaderNodeOutputMaterial")
        if kind == "glass":
            shader = nodes.new(type="ShaderNodeBsdfGlass")
            shader.inputs["Roughness"].default_value = params.get("roughness", 0.0)
            links.new(shader.outputs["BSDF"], output.inputs["Surface"])
        else:
            shader = nodes.new(type="ShaderNodeBsdfPrincipled")
            shader.inputs["Metallic"].default_value = params.get("metallic", 0.0)
            shader.inputs["Roughness"].default_value = params.get("roughness", 0.5)
            links.new(shader.outputs["BSDF"], output.inputs["Surface"])
        mats.append(mat)
    return mats


def _apply_party_mode(
    scene: bpy.types.Scene,
    targets: List[bpy.types.Object],
    seed: int = 1337,
    light_count: int = 3,
) -> List[bpy.types.Object]:
    import random
    from integration.blender_ops.camera_framing import compute_bounds_world

    if not targets:
        return []
    rng = random.Random(seed)
    bounds_min, bounds_max = compute_bounds_world(targets)
    center = (bounds_min + bounds_max) / 2.0
    radius = max((bounds_max - bounds_min).length / 2.0, 1.0)

    mats = _create_party_materials()
    for idx, obj in enumerate(targets):
        if getattr(obj, "type", None) != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mats[idx % len(mats)])

    created_lights: List[bpy.types.Object] = []
    for i in range(light_count):
        light_data = bpy.data.lights.new(name=f"PartyLight_{i}", type="POINT")
        light_obj = bpy.data.objects.new(name=f"PartyLight_{i}", object_data=light_data)
        scene.collection.objects.link(light_obj)
        light_data.energy = rng.uniform(200.0, 600.0)
        light_data.color = (
            rng.random(),
            rng.random(),
            rng.random(),
        )
        offset = Vector(
            (
                rng.uniform(-radius, radius),
                rng.uniform(-radius, radius),
                rng.uniform(-radius, radius),
            )
        )
        light_obj.location = center + offset
        created_lights.append(light_obj)

    return created_lights


def set_render_settings(
    scene: bpy.types.Scene,
    resolution: Tuple[int, int],
    color_mode: str,
    transparent_bg: bool,
    engine: Optional[str],
) -> RenderSettingsSnapshot:
    """Apply render settings and return snapshot for restoration."""
    snapshot = RenderSettingsSnapshot(
        resolution_x=scene.render.resolution_x,
        resolution_y=scene.render.resolution_y,
        resolution_percentage=scene.render.resolution_percentage,
        file_format=scene.render.image_settings.file_format,
        color_mode=scene.render.image_settings.color_mode,
        engine=scene.render.engine,
        film_transparent=scene.render.film_transparent,
    )

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = color_mode
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = transparent_bg
    if engine is not None:
        scene.render.engine = engine
    return snapshot


def ensure_camera(
    scene: bpy.types.Scene, camera: Optional[bpy.types.Object] = None
) -> Tuple[bpy.types.Object, bool, Optional[bpy.types.Object]]:
    """Return a camera to use, creating one if needed."""
    original_camera = scene.camera
    if camera is not None:
        if scene.camera != camera:
            scene.camera = camera
        return camera, False, original_camera
    if scene.camera is not None:
        return scene.camera, False, original_camera
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    scene.camera = camera
    return camera, True, original_camera


def ensure_light(
    scene: bpy.types.Scene,
    light: Optional[bpy.types.Object] = None,
    ensure: bool = True,
) -> Tuple[Optional[bpy.types.Object], bool]:
    """Return a light to use, optionally creating one if none exists."""
    if not ensure:
        return light, False
    if light is not None:
        return light, False
    existing = next((obj for obj in scene.objects if obj.type == "LIGHT"), None)
    if existing is not None:
        return existing, False
    bpy.ops.object.light_add(type="SUN", location=(5, 5, 5))
    return bpy.context.active_object, True


def set_camera_orbit(
    camera: bpy.types.Object,
    center: Vector,
    distance: float,
    angle_rad: float,
    ortho_scale: float,
) -> None:
    """Position the camera at an orbit angle around the object."""
    camera.location.x = center.x + distance * math.cos(angle_rad)
    camera.location.y = center.y + distance * math.sin(angle_rad)
    camera.location.z = center.z
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def set_camera_top(
    camera: bpy.types.Object,
    center: Vector,
    distance: float,
    ortho_scale: float,
) -> None:
    """Position the camera for a top-down orthographic view."""
    camera.location = (center.x, center.y, center.z + distance)
    camera.rotation_euler = (0, 0, 0)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale


def render_silhouette_frame(
    session: SilhouetteRenderSession, output_path: Path
) -> None:
    """Render a single frame using the active scene camera."""
    session.scene.render.filepath = str(output_path)
    with _suppress_blender_render_stdout():
        bpy.ops.render.render(write_still=True)


@contextmanager
def _suppress_blender_render_stdout() -> Any:
    """Keep Blender's per-file render log lines out of smoke-run output."""
    stdout = getattr(sys, "stdout", None)
    fileno = getattr(stdout, "fileno", None)
    if not callable(fileno):
        yield
        return
    try:
        fd = fileno()
    except (OSError, ValueError):
        yield
        return

    try:
        stdout.flush()
    except Exception:
        pass
    saved_fd = os.dup(fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        try:
            if stdout is not None:
                stdout.flush()
        except Exception:
            pass
        os.dup2(saved_fd, fd)
        os.close(saved_fd)


@contextmanager
def silhouette_session(
    *,
    scene: Optional[bpy.types.Scene] = None,
    target_objects: Optional[Iterable[bpy.types.Object]] = None,
    resolution: Tuple[int, int] = (512, 512),
    color_mode: str = "BW",
    transparent_bg: bool = False,
    engine: Optional[str] = None,
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    silhouette_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    force_material: bool = True,
    party_mode: bool = False,
    hide_non_targets: bool = True,
    camera: Optional[bpy.types.Object] = None,
    light: Optional[bpy.types.Object] = None,
    ensure_light_obj: Optional[bool] = None,
    **kwargs: Any,
) -> SilhouetteRenderSession:
    """Context manager for silhouette rendering with cleanup."""
    if ensure_light_obj is None:
        ensure_light_obj = kwargs.pop("ensure_light", True)
    elif "ensure_light" in kwargs:
        raise TypeError("Use ensure_light_obj instead of ensure_light")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available for silhouette rendering")

    scene = scene or bpy.context.scene
    targets = collect_target_objects(scene, target_objects)
    if not targets:
        raise ValueError("No renderable mesh targets found for silhouette rendering")

    render_settings = set_render_settings(
        scene, resolution, color_mode, transparent_bg, engine
    )

    (
        world,
        created_world,
        background_node,
        background_created,
        original_bg_color,
        world_use_nodes,
    ) = ensure_world_background(scene, background_color)

    silhouette_material = ensure_silhouette_material(color=silhouette_color)

    mesh_objects = _mesh_objects(scene)
    hidden_objects: Dict[bpy.types.Object, bool] = {}
    for obj in mesh_objects:
        hidden_objects[obj] = getattr(obj, "hide_render", False)
        if obj in targets:
            obj.hide_render = False
        elif hide_non_targets:
            obj.hide_render = True
        else:
            obj.hide_render = False

    original_materials: Dict[bpy.types.Object, List[Optional[bpy.types.Material]]] = {}
    if party_mode and force_material:
        print("Warning: party_mode ignored because force_material is enabled.")
        party_mode = False
    material_objects = targets if hide_non_targets else mesh_objects
    if force_material or party_mode:
        for obj in material_objects:
            if getattr(obj, "type", None) != "MESH":
                continue
            if hasattr(obj, "hide_set"):
                obj.hide_set(False)
            if hasattr(obj, "hide_viewport"):
                obj.hide_viewport = False
            original_materials[obj] = list(obj.data.materials)
            obj.data.materials.clear()
            if force_material:
                obj.data.materials.append(silhouette_material)

    camera_obj, created_camera, original_camera = ensure_camera(scene, camera)
    light_obj, created_light = ensure_light(scene, light, ensure=ensure_light_obj)
    extra_lights: List[bpy.types.Object] = []
    if party_mode:
        extra_lights = _apply_party_mode(scene, targets)
    _update_view_layer()

    session = SilhouetteRenderSession(
        scene=scene,
        target_objects=targets,
        camera=camera_obj,
        light=light_obj,
        created_camera=created_camera,
        created_light=created_light,
        original_camera=original_camera,
        render_settings=render_settings,
        hidden_objects=hidden_objects,
        original_materials=original_materials,
        world=world,
        created_world=created_world,
        background_node=background_node,
        background_created=background_created,
        background_color=original_bg_color,
        world_use_nodes=world_use_nodes,
        silhouette_material=silhouette_material,
        extra_lights=extra_lights,
    )

    try:
        yield session
    finally:
        session.restore()
