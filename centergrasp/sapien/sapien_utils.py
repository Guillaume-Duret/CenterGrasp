import pathlib
import numpy as np
import spatialmath as sm
from dataclasses import dataclass
from typing import Tuple
import sapien.core as sapien
from sapien.utils.viewer import Viewer
from centergrasp.cameras import CameraParams
from centergrasp.mesh_utils import SceneObject, AmbientCGTexture
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
from pathlib import Path

@dataclass
class CameraObsConfig:
    rgb: bool = True
    depth_gt: bool = False
    depth_real: bool = False
    segmentation: bool = False
    normal: bool = False


@dataclass
class CameraObs:
    rgb: np.ndarray = None
    depth_gt: np.ndarray = None
    depth_real: np.ndarray = None
    segmentation: np.ndarray = None
    normal: np.ndarray = None

    @property
    def depth(self):
        return self.depth_real if self.depth_real is not None else self.depth_gt


@dataclass
class Obs:
    camera: CameraObs = None
    camera_pose: sm.SE3 = None
    joint_state: np.ndarray = None
    wTbase: sm.SE3 = sm.SE3()


@dataclass
class Trajectory:
    time: np.ndarray = None
    position: np.ndarray = None
    velocity: np.ndarray = None
    acceleration: np.ndarray = None


def init_sapien(
    headless: bool, physics_dt: float
) -> Tuple[sapien.Engine, sapien.SapienRenderer, sapien.Scene]:
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(offscreen_only=headless)
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    scene.set_timestep(physics_dt)
    return engine, renderer, scene


def enable_raytracing():
    # Call this before creating a camera or a viewer
    sapien.render_config.camera_shader_dir = "rt"
    sapien.render_config.viewer_shader_dir = "rt"
    sapien.render_config.rt_samples_per_pixel = 32
    sapien.render_config.rt_max_path_depth = 8
    sapien.render_config.rt_use_denoiser = True
    return


def init_viewer(scene: sapien.Scene, renderer: sapien.SapienRenderer, show_axes=False) -> Viewer:
    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=1.5, y=0.0, z=1.0)
    viewer.set_camera_rpy(y=3.14, p=-0.5, r=0)
    viewer.toggle_axes(show_axes)
    viewer.render()
    return viewer


def init_default_material(scene: sapien.Scene):
    physical_material = scene.create_physical_material(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    )
    scene.default_physical_material = physical_material
    return


def init_lights(scene: sapien.Scene):
    scene.set_ambient_light([0.4, 0.4, 0.4])
    lights = [
        scene.add_point_light(position=[2, 2, 2], color=[1, 1, 1], shadow=True),
        scene.add_point_light(position=[2, -2, 2], color=[1, 1, 1], shadow=True),
        scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3]),
        scene.add_directional_light([0, 0, -1], [1, 1, 1]),
    ]
    return lights


def render_material_from_ambient_cg_texture(
    renderer: sapien.SapienRenderer, texture: AmbientCGTexture
) -> sapien.RenderMaterial:
    render_material = renderer.create_material()
    if texture.color_fpath is not None:
        render_material.set_diffuse_texture_from_file(str(texture.color_fpath))
    if texture.roughness_fpath is not None:
        render_material.set_roughness_texture_from_file(str(texture.roughness_fpath))
    if texture.metallic_fpath is not None:
        render_material.set_metallic_texture_from_file(str(texture.metallic_fpath))
    if texture.normal_fpath is not None:
        render_material.set_normal_texture_from_file(str(texture.normal_fpath))
    return render_material


def random_render_material(renderer: sapien.SapienRenderer) -> sapien.RenderMaterial:
    render_material = renderer.create_material()
    render_material.set_base_color([np.random.rand(), np.random.rand(), np.random.rand(), 1.0])
    render_material.set_roughness(np.random.rand())
    render_material.set_metallic(np.random.rand())
    return render_material


def add_table(
    scene: sapien.Scene,
    half_size: np.ndarray,
    position: np.ndarray,
    material: sapien.RenderMaterial = None,
    color: np.ndarray = np.array([0.3, 0.3, 0.6]),
) -> sapien.Actor:
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size)
    print("IN aDD table")
    if material is not None:
        print("IN TABLE MATERIAL")
        builder.add_box_visual(half_size=half_size, material=material)
    else:
        print("IN NO TABLE MATERIAL")
        builder.add_box_visual(half_size=half_size, color=color)
    table = builder.build_kinematic(name="table")
    table.set_pose(sapien.Pose(position))
    return table


def _get_object_builder(
    scene: sapien.Scene,
    obj: SceneObject,
    render_material: sapien.RenderMaterial = None,
) -> sapien.ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_collision_from_file(
        filename=str(obj.collision_fpath), scale=obj.scale, density=obj.density
    )
    builder.add_visual_from_file(
        filename=str(obj.visual_fpath), scale=obj.scale, material=render_material
    )
    return builder


def _get_object_builder_nonconvex(
    scene: sapien.Scene,
    obj: SceneObject,
    render_material: sapien.RenderMaterial = None,
) -> sapien.ActorBuilder:
    builder = scene.create_actor_builder()
    builder.add_nonconvex_collision_from_file(filename=str(obj.collision_fpath), scale=obj.scale)
    builder.add_visual_from_file(
        filename=str(obj.visual_fpath), scale=obj.scale, material=render_material
    )
    return builder


def add_object_nonconvex(
    scene: sapien.Scene,
    obj: SceneObject,
    render_material: sapien.RenderMaterial = None,
) -> sapien.Actor:
    builder = _get_object_builder_nonconvex(scene, obj, render_material)
    actor = builder.build_kinematic(name=obj.name)
    actor.set_pose(sapien.Pose.from_transformation_matrix(obj.pose4x4))
    return actor


def load_materials_from_mtl(renderer, mtl_path):
    """Load materials from MTL file without _visual suffix"""
    materials = {}
    current_mtl = None
    
    try:
        with open(mtl_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                key = parts[0].lower()
                
                if key == 'newmtl':
                    current_mtl = renderer.create_material()
                    materials[parts[1]] = current_mtl
                elif current_mtl and key == 'map_kd':
                    texture_path = mtl_path.parent / parts[1]
                    if texture_path.exists():
                        try:
                            current_mtl.set_diffuse_texture_from_file(str(texture_path))
                            print(f"Loaded texture: {texture_path.name}")
                        except Exception as e:
                            print(f"Failed to load texture {texture_path}: {str(e)}")
    
    except Exception as e:
        print(f"Error loading MTL {mtl_path}: {str(e)}")
    
    return materials
def add_object_kinematic(
    scene: sapien.Scene,
    obj: SceneObject,
    render_material: sapien.RenderMaterial = None,
) -> sapien.Actor:
    print(f"Loading object: {obj.name}")
    print(f"Visual file: {obj.visual_fpath}")
    print(f"Using material: {'custom' if render_material else 'original'}")
    # Use original materials if availablle

    obj_path = Path(obj.visual_fpath)
    # Get base name without _visual
    base_name = obj_path.stem.replace('_visual', '')
    mtl_path = obj_path.with_name(f"{base_name}.mtl")
    
    if mtl_path.exists() and render_material is None:
        try:
            materials = load_materials_from_mtl(scene.engine.renderer, mtl_path)
            if materials:
                render_material = next(iter(materials.values()))
                # Debug print (safe version)
                print(f"Loaded material for {obj.name}")
        except Exception as e:
            print(f"Error loading materials for {obj.name}: {str(e)}")

    builder = _get_object_builder(scene, obj, render_material)
    actor = builder.build_kinematic(name=obj.name)
    actor.set_pose(sapien.Pose.from_transformation_matrix(obj.pose4x4))
    return actor


def add_object_dynamic(
    scene: sapien.Scene,
    obj: SceneObject,
    render_material: sapien.RenderMaterial = None,
) -> sapien.Actor:
    builder = _get_object_builder(scene, obj, render_material)
    actor = builder.build(name=obj.name)
    actor.set_pose(sapien.Pose.from_transformation_matrix(obj.pose4x4))
    return actor


def add_robot(
    scene: sapien.Scene,
    urdf_path: pathlib.Path,
    pose: np.ndarray = np.eye(4),
    fix_root_link: bool = True,
) -> sapien.Articulation:
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    robot = loader.load(str(urdf_path))
    robot.set_name(urdf_path.stem)
    robot.set_root_pose(sapien.Pose.from_transformation_matrix(pose))
    return robot


def add_camera(
    scene: sapien.Scene,
    camera_params: CameraParams,
    near: float = 0.01,
    far: float = 10.0,
    name: str = "",
) -> sapien.CameraEntity:
    camera = scene.add_camera(
        name=name,
        width=camera_params.width,
        height=camera_params.height,
        fovy=camera_params.fov_y_rad,
        near=near,
        far=far,
    )
    return camera


def attach_to_link(camera: sapien.CameraEntity, link: sapien.Link, local_pose: np.ndarray):
    camera.set_parent(parent=link, keep_pose=False)
    camera.set_local_pose(sapien.Pose.from_transformation_matrix(local_pose))
    return


def add_sensor(
    scene: sapien.Scene,
    camera_params: CameraParams,
    near: float = 0.01,
    far: float = 10.0,
    name: str = "",
) -> StereoDepthSensor:
    config = StereoDepthSensorConfig()
    config.rgb_resolution = (camera_params.width, camera_params.height)
    config.ir_resolution = (camera_params.width, camera_params.height)
    config.rgb_intrinsic = camera_params.K
    config.ir_intrinsic = camera_params.K
    config.min_depth = near
    config.max_depth = far
    sensor = StereoDepthSensor(name, scene, config)
    return sensor


def get_sensor_obs(sensor: StereoDepthSensor, camera_obs_config: CameraObsConfig) -> CameraObs:
    camera = sensor._cam_rgb
    cam_obs = get_camera_obs(camera, camera_obs_config)
    if camera_obs_config.depth_real:
        sensor.take_picture(infrared_only=True)
        sensor.compute_depth()
        cam_obs.depth_real = sensor.get_depth()
    return cam_obs


def get_camera_obs(camera: sapien.CameraEntity, camera_obs_config: CameraObsConfig) -> CameraObs:
    cam_obs = CameraObs()
    camera.take_picture()
    if camera_obs_config.rgb:
        rgb_float = camera.get_color_rgba()[..., :-1]  # [H, W, 4]
        cam_obs.rgb = (rgb_float * 255).astype(np.uint8)
    if camera_obs_config.depth_gt:
        # Each pixel is (x, y, z, z-buffer) in camera space (OpenGL/Blender)
        position_rgba = camera.get_position_rgba()  # [H, W, 4]
        depth = -position_rgba[..., 2]
        cam_obs.depth_gt = depth[..., np.newaxis]
    if camera_obs_config.normal:
        # [H, W, 4] (last channel just zeros)
        normal_rgb = camera.get_normal_rgba()[..., :-1]
        cam_obs.normal = normal_rgb
    if camera_obs_config.segmentation:
        # [H, W, 4] (last 2 channels zeros)
        segmentation = camera.get_visual_actor_segmentation()[..., :-2]
        actor_level_segm = segmentation[..., 1]
        cam_obs.segmentation = actor_level_segm
    return cam_obs
