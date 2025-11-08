import argparse
import json
import os
import random
from typing import Iterable, Union

import bpy
import bmesh
import imageio
import numpy as np

import blenderproc as bproc


context = bpy.context
scene = context.scene
render = scene.render


bproc.init()
bproc.renderer.set_output_format(file_format="PNG", enable_transparency=True)

render.engine = "CYCLES"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100


MeshObjType = Union[bproc.types.MeshObject, bpy.types.Object]


def _ensure_consistent_normals(objects: Iterable[MeshObjType]) -> None:
    for obj in objects:
        blender_obj = getattr(obj, "blender_obj", obj)
        if blender_obj.type != "MESH":
            continue

        mesh = blender_obj.data
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(mesh)
        finally:
            bm.free()
        mesh.calc_normals()
        mesh.update()

        if hasattr(obj, "shade_smooth"):
            obj.shade_smooth()
        else:
            bpy.ops.object.select_all(action='DESELECT')
            blender_obj.select_set(True)
            bpy.context.view_layer.objects.active = blender_obj
            bpy.ops.object.shade_smooth()
            blender_obj.select_set(False)


def _add_lighting() -> None:
    bpy.ops.object.light_add(type="AREA")
    light_obj = bpy.data.objects["Area"]
    light = light_obj.data
    light.energy = 30000
    light.use_nodes = True
    light_obj.location[2] = 1.3
    light_obj.scale = (30.0, 30.0, 30.0)

    bpy.ops.object.light_add(type="AREA")
    fill_obj = bpy.data.objects["Area.001"]
    fill_light = fill_obj.data
    fill_light.energy = 8000
    fill_light.use_nodes = True
    fill_obj.location = (2.0, -2.0, 1.0)
    fill_obj.rotation_euler = (np.deg2rad(45), np.deg2rad(15), np.deg2rad(-35))

    bpy.data.worlds["World"].use_nodes = True
    bg = bpy.data.worlds["World"].node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.2


def _setup_camera() -> None:
    cam = scene.objects["Camera"]
    cam.location = (0, 3, 0)
    bpy.data.cameras["Camera"].lens_unit = "FOV"
    bpy.data.cameras["Camera"].angle = 40 * np.pi / 180

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty


def _sample_camera_loc(phi: float, theta: float, r: float = 3.0) -> np.ndarray:
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z], dtype=np.float32)


def _load_objs(tree, data_root: str) -> None:
    loaded_objects = []
    for node in tree:
        semantic_id = int(node.get("id", 0))
        for obj_path in node["objs"]:
            objs = bproc.loader.load_obj(os.path.join(data_root, obj_path))
            for obj in objs:
                try:
                    obj.set_cp("category_id", semantic_id)
                except AttributeError:
                    obj["category_id"] = semantic_id
                if hasattr(obj, "set_cp"):
                    obj.set_cp("semantic_id", semantic_id)
                else:
                    obj["semantic_id"] = semantic_id
                loaded_objects.append(obj)
    _ensure_consistent_normals(loaded_objects)


def _render_scene(n_imgs: int = 10):
    _setup_camera()
    _add_lighting()

    for material in bpy.data.materials:
        material.use_backface_culling = False

    phi_seg = np.linspace(np.pi / 3, np.pi / 2, n_imgs + 1)
    theta_seg = np.linspace(-5 * np.pi / 6, -np.pi / 6, n_imgs + 1)

    phis = [np.random.uniform(phi_seg[i], phi_seg[i + 1]) for i in range(n_imgs)]
    thetas = [np.random.uniform(theta_seg[i], theta_seg[i + 1]) for i in range(n_imgs)]
    random.shuffle(phis)
    random.shuffle(thetas)

    for i in range(n_imgs):
        r = np.random.uniform(3, 3.5)
        location = _sample_camera_loc(phis[i], thetas[i], r)
        rotation_matrix = bproc.camera.rotation_from_forward_vec([0, 0, 0] - location)
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

    data = bproc.renderer.render()
    data.update(bproc.renderer.render_segmap(map_by=["instance", "class"]))
    return data


def _write_imgs(src_dir: str, data) -> None:
    save_img_dir = os.path.join(src_dir, "imgs")
    os.makedirs(save_img_dir, exist_ok=True)

    for i, rgb in enumerate(data["colors"]):
        fname = str(i).zfill(2)
        imageio.imwrite(f"{save_img_dir}/{fname}.png", rgb)

    if "class_segmaps" in data:
        semantic_masks = data["class_segmaps"]
        semantic_dir = os.path.join(save_img_dir, "semantic_masks_merge_fixed")
        os.makedirs(semantic_dir, exist_ok=True)
        for i, semantic_mask in enumerate(semantic_masks):
            fname = str(i).zfill(2)
            np.savez_compressed(f"{semantic_dir}/{fname}.npz", semantic_mask=semantic_mask)
            mask = semantic_mask.astype(np.float32)
            mask -= mask.min()
            if mask.max() > 0:
                mask /= mask.max()
            semantic_img = (mask * 255).astype(np.uint8)
            imageio.imwrite(f"{semantic_dir}/{fname}.png", semantic_img)


def render_imgs(src_dir: str, n_imgs: int = 20, incremental: bool = False) -> None:
    if incremental:
        img_dir = os.path.join(src_dir, "imgs")
        if not os.path.exists(img_dir):
            return
        if len(os.listdir(img_dir)) < n_imgs:
            return

    print(f"Rendering images for {src_dir} ...")

    with open(os.path.join(src_dir, "object_merge_fixed.json"), "r") as f:
        src = json.load(f)

    _load_objs(src["diffuse_tree"], src_dir)

    raw_data = _render_scene(n_imgs)
    _write_imgs(src_dir, raw_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to the data directory")
    parser.add_argument("--n_imgs", type=int, default=40, help="number of images to render for each model")
    parser.add_argument("--incremental", action="store_true", help="whether to render images incrementally")

    args = parser.parse_args()

    try:
        render_imgs(args.data, args.n_imgs, args.incremental)
    except Exception as e:  # pragma: no cover
        print(e)
        with open("render_err.log", "a", encoding="utf-8") as f:
            f.write(f"{args.data}\n")
            f.write(f"{e}\n")
