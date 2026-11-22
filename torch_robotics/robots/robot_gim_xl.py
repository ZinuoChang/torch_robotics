"""GIM XL 6-DoF arm (hardware experiment robot).

Kinematics: nominal CAD URDF of the hardware bundle (gim_xl_sim_resources_20261117), base frame
``arm6_base_link`` at the world origin, tool frame ``arm6_tool0`` (link-6 origin, NOT a calibrated TCP).

Collision model: the cuRobo sphere fit exported to ``data/configs/gim_xl/gim_xl_sphere_config.yaml``
(see curobo-v2-fast ``benchmark/fsd_mp/tools``), so stomp-mpd and cuRobo share one geometry.

- Object collision: one point per sphere center of link1..link6 with the sphere radius as margin
  (the base link is fixed and is not checked against objects, like panda_link0/1 for the Panda).
- Self collision: every sphere pair of the active link pairs (pairs not in cuRobo's ignore matrix),
  in collision if the center distance < r_i + r_j + ``self_collision_margin``. Unlike the Panda model
  this test always uses the real sphere radii, even when callers pass ``margin=`` (e.g.
  ``PlanningTask.get_trajs_collision_and_free`` passes margin=0.01 to every field).
"""
from collections import OrderedDict

import einops
import numpy as np
import torch

from torch_robotics.environments.primitives import MultiSphereField
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, \
    link_rot_from_link_tensor, link_quat_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robot_tree import convert_link_dict_to_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableGimXL
from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionSelfField
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame


class CollisionSelfFieldSpheres(CollisionSelfField):
    """Sphere-pair self collision that keeps its per-pair radii margins when a ``margin`` is passed."""

    def compute_embodiment_collision(self, q, link_pos, **kwargs):
        distances = self.compute_embodiment_signed_distances(q, link_pos, **kwargs)
        return torch.any(distances < self.cutoff_margin, dim=-1)

    def compute_costs_impl(self, q, link_pos, **kwargs):
        kwargs.pop('margin', None)
        return super().compute_costs_impl(q, link_pos, **kwargs)


class RobotGimXL(RobotBase):

    ARM_LINKS = ['arm6_base_link'] + [f'arm6_link{i}' for i in range(1, 7)] + ['arm6_tool0']
    OBJECT_COLLISION_LINKS = [f'arm6_link{i}' for i in range(1, 7)]

    def __init__(self,
                 self_collision_margin=0.0,
                 tensor_args=None,
                 **kwargs):
        kwargs.pop('use_collision_spheres', None)
        kwargs.pop('grasped_object', None)  # not supported (no gripper DOF in the model)

        self.link_name_ee = 'arm6_tool0'
        self.diff_gim = DifferentiableGimXL(device=tensor_args['device'], use_collision_spheres=True)
        self.jl_lower, self.jl_upper, _, _ = self.diff_gim.get_joint_limit_array()
        q_limits = torch.tensor(np.array([self.jl_lower, self.jl_upper]), **tensor_args)

        spheres = self.diff_gim.spheres
        name_to_idx = self.diff_gim._name_to_idx_map

        def sphere_links(link):
            return [f'{link}_{i}' for i in range(len(spheres.get(link, [])))]

        def sphere_radius(sphere_link):
            link, i = sphere_link.rsplit('_', 1)
            return float(spheres[link][int(i)][3])

        # ---- object collision: sphere centers of the moving links
        link_names_obj = [s for link in self.OBJECT_COLLISION_LINKS for s in sphere_links(link)]
        link_margins_obj = [sphere_radius(s) for s in link_names_obj]
        link_idxs_obj = [name_to_idx[s] for s in link_names_obj]

        # ---- self collision: all sphere pairs of the active link pairs
        pairs_sphere = OrderedDict()
        for link_a, link_b in self.diff_gim.self_collision_pairs:
            for sa in sphere_links(link_a):
                pairs_sphere.setdefault(sa, []).extend(sphere_links(link_b))
        link_names_self = sorted({s for k, v in pairs_sphere.items() for s in [k] + v})
        link_idxs_self = [name_to_idx[s] for s in link_names_self]

        super().__init__(
            name='RobotGimXL',
            q_limits=q_limits,
            grasped_object=None,
            link_names_for_object_collision_checking=link_names_obj,
            link_margins_for_object_collision_checking=link_margins_obj,
            link_idxs_for_object_collision_checking=link_idxs_obj,
            num_interpolated_points_for_object_collision_checking=len(link_names_obj),  # = sphere centers
            link_names_for_self_collision_checking=link_names_self,
            link_names_pairs_for_self_collision_checking=pairs_sphere,
            link_idxs_for_self_collision_checking=link_idxs_self,
            num_interpolated_points_for_self_collision_checking=len(link_names_self),  # = sphere centers
            self_collision_margin_robot=self_collision_margin,
            use_collision_spheres=True,
            tensor_args=tensor_args,
            **kwargs
        )

        # Per-pair threshold r_i + r_j + margin, in the order RobotBase built the pair indices.
        margins = []
        for name_a in self.link_names_for_self_collision_checking:
            for name_b in pairs_sphere.get(name_a, []):
                margins.append(sphere_radius(name_a) + sphere_radius(name_b) + self_collision_margin)
        old = self.df_collision_self
        self.df_collision_self = CollisionSelfFieldSpheres(
            self,
            link_idxs_for_collision_checking=old.link_idxs_for_collision_checking,
            idxs_links_distance_matrix=old.idxs_links_distance_matrix,
            num_interpolated_points=old.num_interpolated_points,
            cutoff_margin=to_torch(margins, **self.tensor_args),
            tensor_args=self.tensor_args,
        )
        assert len(margins) == len(old.idxs_links_distance_matrix)
        self.n_self_collision_pairs = len(margins)

        # tensors for rendering
        self.arm_link_names = [n for n in self.ARM_LINKS if n in name_to_idx]
        self.all_sphere_link_names = [s for link in spheres for s in sphere_links(link)]
        self.all_sphere_radii = to_torch([sphere_radius(s) for s in self.all_sphere_link_names], **self.tensor_args)

    # ------------------------------------------------------------------------------------------------
    def fk_map_collision_impl(self, q, **kwargs):
        q_orig_shape = q.shape
        if len(q_orig_shape) == 3:
            b, h, d = q_orig_shape
            q = einops.rearrange(q, 'b h d -> (b h) d')
        elif len(q_orig_shape) != 2:
            raise NotImplementedError
        link_pose_dict = self.diff_gim.compute_forward_kinematics_all_links(q, return_dict=True)
        link_all_tensor = convert_link_dict_to_tensor(link_pose_dict, self.diff_gim.get_link_names())
        if len(q_orig_shape) == 3:
            link_all_tensor = einops.rearrange(link_all_tensor, "(b h) t d1 d2 -> b h t d1 d2", b=b, h=h)
        return link_pos_from_link_tensor(link_all_tensor)

    def get_EE_pose(self, q):
        return self.diff_gim.compute_forward_kinematics_all_links(q, link_list=[self.link_name_ee])

    def get_EE_position(self, q):
        return link_pos_from_link_tensor(self.get_EE_pose(q))

    def get_EE_orientation(self, q, rotation_matrix=True):
        ee_pose = self.get_EE_pose(q)
        return link_rot_from_link_tensor(ee_pose) if rotation_matrix else link_quat_from_link_tensor(ee_pose)

    def get_sphere_centers(self, q):
        """World positions of all collision spheres (incl. base), shape (..., n_spheres, 3)."""
        fks = self.diff_gim.compute_forward_kinematics_all_links(q.reshape(-1, self.q_dim), return_dict=True)
        pos = link_pos_from_link_tensor(convert_link_dict_to_tensor(fks, self.all_sphere_link_names))
        return pos.reshape(*q.shape[:-1], -1, 3)

    # ------------------------------------------------------------------------------------------------
    def render(self, ax, q=None, color='blue', arrow_length=0.1, arrow_alpha=1.0, arrow_linewidth=2.0,
               draw_links_spheres=False, **kwargs):
        skeleton = get_skeleton_from_model(self.diff_gim, q, self.arm_link_names)
        skeleton.draw_skeleton(ax=ax, color=color)
        fks_dict = self.diff_gim.compute_forward_kinematics_all_links(q.unsqueeze(0), return_dict=True)
        if draw_links_spheres:
            centers = self.get_sphere_centers(q.unsqueeze(0)).squeeze(0)
            spheres = MultiSphereField(centers, self.all_sphere_radii.view(-1, 1), tensor_args=self.tensor_args)
            spheres.render(ax, color='red', cmap='Reds', **kwargs)
        plot_coordinate_frame(ax, fks_dict[self.link_name_ee], tensor_args=self.tensor_args,
                              arrow_length=arrow_length, arrow_alpha=arrow_alpha, arrow_linewidth=arrow_linewidth)

    def get_isaac_asset(self):
        """(asset_root, asset_file) of a URDF for Isaac Gym: CAD meshes as visuals and the collision
        spheres as collision geometry, so Isaac contacts use the same model as planning."""
        from pathlib import Path
        from xml.dom import minidom
        from xml.etree import ElementTree as ET
        from urdf_parser_py.urdf import URDF, Collision, Sphere, Pose
        from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
        root = get_robot_path()
        src = root / 'gim_xl_description' / 'gim_arm_xl_6dof.urdf'
        dst = src.with_name('gim_arm_xl_6dof_isaac_spheres.urdf')
        urdf = URDF.from_xml_file(src.as_posix())
        for link in urdf.links:
            link.collisions = [Collision(geometry=Sphere(radius=float(s[3])), origin=Pose(xyz=[float(v) for v in s[:3]]))
                               for s in self.diff_gim.spheres.get(link.name, [])]
        dst.write_text(minidom.parseString(ET.tostring(urdf.to_xml())).toprettyxml(indent='  '))
        return str(root), str(dst.relative_to(root))

    def _load_visual_meshes(self):
        if getattr(self, '_visual_meshes', None) is None:
            import trimesh
            from pathlib import Path
            from urdf_parser_py.urdf import URDF
            from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
            urdf_file = get_robot_path() / 'gim_xl_description' / 'gim_arm_xl_6dof.urdf'
            urdf = URDF.from_xml_file(urdf_file.as_posix())
            self._visual_meshes = {}
            for link in urdf.links:
                if link.visual is not None and hasattr(link.visual.geometry, 'filename'):
                    mesh = trimesh.load(Path(urdf_file).parent / link.visual.geometry.filename, force='mesh')
                    self._visual_meshes[link.name] = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
        return self._visual_meshes

    def render_meshes(self, ax, q, color=(0.75, 0.78, 0.85), alpha=1.0, face_fraction=0.15, seed=0, **kwargs):
        """Draw the CAD meshes at configuration q (a random ``face_fraction`` of the faces, for speed)."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        rng = np.random.default_rng(seed)
        fks = self.diff_gim.compute_forward_kinematics_all_links(q.reshape(1, -1), return_dict=True)
        for link, (verts, faces) in self._load_visual_meshes().items():
            H = to_numpy(fks[link].get_transform_matrix().squeeze(0))
            v = verts @ H[:3, :3].T + H[:3, 3]
            if face_fraction < 1.0:
                faces = faces[rng.random(len(faces)) < face_fraction]
            tri = v[faces]
            # simple Lambert shading so the shape is readable
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
            shade = 0.45 + 0.55 * np.abs(n @ np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75]))
            fc = np.clip(np.asarray(color)[None, :3] * shade[:, None], 0, 1)
            coll = Poly3DCollection(tri, facecolors=fc, edgecolors='none', alpha=alpha)
            ax.add_collection3d(coll)

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            for traj, color in zip(trajs_pos, colors):
                for t in range(traj.shape[0]):
                    self.render(ax, traj[t], color, **kwargs, arrow_length=0.05, arrow_alpha=0.5, arrow_linewidth=1.)
        if start_state is not None:
            self.render(ax, start_state, color='green')
        if goal_state is not None:
            self.render(ax, goal_state, color='purple')
