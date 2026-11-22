from pathlib import Path
from typing import Optional, List
from xml.dom import minidom

import numpy as np
import yaml
from urdf_parser_py.urdf import URDF, Joint, Link, Visual, Collision, Box, Pose

from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_to_euler
from torch_robotics.torch_kinematics_tree.models.robot_tree import DifferentiableTree
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path, get_configs_path
from xml.etree import ElementTree as ET

from torch_robotics.torch_utils.torch_utils import to_numpy


class DifferentiableKUKAiiwa(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'kuka_iiwa' / 'urdf' / 'iiwa7.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_kuka_iiwa"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


def modidy_franka_panda_urdf_grasped_object(robot_file, grasped_object):
    robot_urdf = URDF.from_xml_file(robot_file)
    joint = Joint(
        name='grasped_object_fixed_joint',
        parent='panda_hand',
        child='grasped_object',
        joint_type='fixed',
        origin=Pose(xyz=to_numpy(grasped_object.pos.squeeze()),
                    rpy=to_numpy(q_to_euler(grasped_object.ori).squeeze())
                    )
    )
    robot_urdf.add_joint(joint)

    geometry_grasped_object = grasped_object.geometry_urdf
    link = Link(
        name='grasped_object',
        visual=Visual(geometry_grasped_object),
        # inertial=None,
        collision=Collision(geometry_grasped_object),
        origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
    )
    robot_urdf.add_link(link)

    # replace the robots file
    robot_file = Path(str(robot_file).replace('.urdf', '_grasped_object.urdf'))
    xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")
    with open(str(robot_file), "w") as f:
        f.write(xmlstr)

    return robot_file


def modidy_franka_panda_urdf_collision_model(robot_file):
    collision_spheres = 'panda/panda_sphere_config.yaml'
    # load collision file:
    coll_yml = (get_configs_path() / collision_spheres).as_posix()
    with open(coll_yml) as file:
        coll_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_urdf = URDF.from_xml_file(robot_file)

    link_collision_names = []
    link_collision_margins = []
    for link_name, spheres_l in coll_params.items():
        for i, sphere in enumerate(spheres_l):
            joint = Joint(
                name=f'joint_{link_name}_sphere_{i}',
                parent=f'{link_name}',
                child=f'{link_name}_{i}',
                joint_type='fixed',
                origin=Pose(xyz=to_numpy(sphere[:3]))
            )
            robot_urdf.add_joint(joint)

            link_collision = f'{link_name}_{i}'
            link = Link(
                name=link_collision,
                origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
            )
            robot_urdf.add_link(link)

            link_collision_names.append(link_collision)
            link_collision_margins.append(sphere[-1])

    # replace the robots file
    robot_file = Path(str(robot_file).replace('.urdf', '_collision_model.urdf'))
    xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")
    with open(str(robot_file), "w") as f:
        f.write(xmlstr)

    return robot_file, link_collision_names, link_collision_margins


class DifferentiableFrankaPanda(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, gripper=False, device='cpu', grasped_object=None,
                 use_collision_spheres=False):
        if gripper:
            robot_file = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_hand.urdf'
        else:
            robot_file = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_no_gripper.urdf'

        # Modify the urdf to append links of the collision model
        if use_collision_spheres:
            robot_file, link_collision_names, link_collision_margins = modidy_franka_panda_urdf_collision_model(robot_file)
            self.link_collision_names = link_collision_names
            self.link_collision_margins = link_collision_margins

        # Modify the urdf to append the link of the grasped object
        if grasped_object is not None:
            robot_file = modidy_franka_panda_urdf_grasped_object(robot_file, grasped_object)

        self.model_path = robot_file.as_posix()
        self.name = "differentiable_franka_panda"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableUR10(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, attach_gripper=False, device='cpu'):
        robot_path = get_robot_path()
        if attach_gripper:
            robot_file = robot_path / 'ur10' / 'urdf' / 'ur10_suction.urdf'
        else:
            robot_file = robot_path / 'ur10' / 'urdf' / 'ur10.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_ur10"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableHabitatStretch(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_path = get_robot_path()
        robot_file = robot_path / 'habitat_stretch' / 'urdf' / 'hab_stretch.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_stretch"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableTiagoDualHolo(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'tiago_dual_description' / 'tiago_dual_holobase_minimal.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_tiago_dual_holo"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableTiagoDualHoloMove(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'tiago_dual_description' / 'tiago_dual_holobase_minimal_holonomic.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_tiago_dual_holo_move"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)

    def get_link_names(self):  # pop those hacky frames for moving base
        return super().get_link_names()[3:]


class DifferentiableShadowHand(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'shadow_hand' / 'shadow_hand.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_shadow_hand"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableAllegroHand(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'allegro_hand' / 'allegro_hand.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_allegro_hand"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class Differentiable2LinkPlanar(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'planar_manipulators' / 'urdf' / '2_link_planar.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_2_link_planar"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


def modify_urdf_add_sphere_links(robot_file, spheres):
    """Append one fixed child link per collision sphere (named ``<link>_<i>``) to a URDF.

    Args:
        robot_file: Path of the source URDF.
        spheres: ``{link_name: [[x, y, z, radius], ...]}`` in the link frame.

    Returns:
        (path of the written ``*_collision_model.urdf``, sphere link names, sphere radii)
    """
    robot_urdf = URDF.from_xml_file(robot_file)
    link_collision_names = []
    link_collision_margins = []
    for link_name, spheres_l in spheres.items():
        for i, sphere in enumerate(spheres_l):
            link_collision = f'{link_name}_{i}'
            robot_urdf.add_joint(Joint(
                name=f'joint_{link_name}_sphere_{i}', parent=link_name, child=link_collision,
                joint_type='fixed', origin=Pose(xyz=[float(v) for v in sphere[:3]])
            ))
            robot_urdf.add_link(Link(name=link_collision, origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])))
            link_collision_names.append(link_collision)
            link_collision_margins.append(float(sphere[3]))

    robot_file = Path(str(robot_file).replace('.urdf', '_collision_model.urdf'))
    xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")
    with open(str(robot_file), "w") as f:
        f.write(xmlstr)
    return robot_file, link_collision_names, link_collision_margins


class DifferentiableGimXL(DifferentiableTree):
    """GIM XL 6-DoF arm (nominal CAD URDF of the hardware bundle, 2026-11-17).

    Collision spheres come from ``data/configs/gim_xl/gim_xl_sphere_config.yaml``, exported from
    the cuRobo sphere fit, so torch_robotics and cuRobo share one collision model.
    """
    sphere_config = 'gim_xl/gim_xl_sphere_config.yaml'

    def __init__(self, link_list: Optional[str] = None, device='cpu', use_collision_spheres=True):
        robot_file = get_robot_path() / 'gim_xl_description' / 'gim_arm_xl_6dof.urdf'
        with open((get_configs_path() / self.sphere_config).as_posix()) as file:
            self.collision_config = yaml.safe_load(file)
        self.spheres = self.collision_config['spheres']
        self.self_collision_pairs = self.collision_config['self_collision_pairs']

        if use_collision_spheres:
            robot_file, self.link_collision_names, self.link_collision_margins = \
                modify_urdf_add_sphere_links(robot_file, self.spheres)

        self.model_path = robot_file.as_posix()
        self.name = "differentiable_gim_xl"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)
