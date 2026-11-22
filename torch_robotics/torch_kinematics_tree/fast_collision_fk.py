"""
Fast forward kinematics of link positions + analytic position Jacobian-vector products for revolute-joint trees
(e.g. the Panda), used for collision checking / gradient-based planning.

DifferentiableTree.compute_forward_kinematics_all_links composes a transform for every link with all its descendants
(hundreds of tiny kernels for the Panda) and autograd then has to differentiate through all of it. For collision
checking only the *positions* of the link frames are needed, and only the controlled joints move: every other link is
rigidly attached to the nearest controlled ancestor. So:

  * forward: chain the controlled joint frames (n_dof small products), then
        p_link = t_anchor + R_anchor @ c_link,   c_link = constant offset to the anchor frame
  * backward (VJP): for a revolute joint j with world axis w_j and origin o_j,
        d p_s / d q_j = w_j x (p_s - o_j)   for every link s below joint j, so
        grad_q_j = w_j . ( sum_s p_s x g_s - o_j x sum_s g_s )
    which is a few masked sums, no autograd graph.

Constant offsets are taken from the existing FK at a reference configuration, so conventions (fixed rotations, joint
clamping, base pose) are inherited from the original implementation. Equivalence with FK + autograd is covered by
tests/test_fast_collision_fk.py.
"""
import torch

from torch_robotics.torch_kinematics_tree.geometrics.utils import bmm3, bmv3


def _is_supported(tree):
    for idx in tree._controlled_joints:
        if tree._bodies[idx].joint_type not in ('revolute', 'continuous'):
            return False
    return len(tree._controlled_joints) > 0


class FastCollisionFK:

    @staticmethod
    def build_if_supported(tree, tensor_args):
        if not _is_supported(tree):
            return None
        return FastCollisionFK(tree, tensor_args)

    def __init__(self, tree, tensor_args):
        self.tensor_args = tensor_args
        device, dtype = tensor_args['device'], tensor_args['dtype']
        bodies = list(tree._bodies)
        names = [b.name for b in bodies]
        name_to_row = {n: i for i, n in enumerate(names)}
        self.n_links = len(bodies)
        self.n_dof = len(tree._controlled_joints)

        # reference configuration (any q works: offsets between rigidly connected frames do not depend on q)
        with torch.no_grad():
            pose_ref = tree.compute_forward_kinematics_all_links(
                torch.zeros(1, tree._n_dofs, device=device), return_dict=True)
        T0 = {n: pose_ref[n].get_transform_matrix()[0].double().cpu() for n in names}

        def parent_of(b):
            return getattr(b, '_parent', None) or getattr(b, '_parents', None)

        controlled_rows = [name_to_row[bodies[i].name] for i in tree._controlled_joints]
        row_to_joint = {r: k for k, r in enumerate(controlled_rows)}

        def anchor_row(row):
            # nearest controlled ancestor-or-self, else the root
            while True:
                if row in row_to_joint:
                    return row
                p = parent_of(bodies[row])
                if p is None:
                    return row
                row = name_to_row[p.name]

        anchors = [name_to_row[bodies[0].name]] + controlled_rows       # anchor 0: root, anchor k+1: joint k
        anchor_index = {r: a for a, r in enumerate(anchors)}
        self.root_R = T0[names[0]][:3, :3].to(device=device, dtype=dtype)
        self.root_t = T0[names[0]][:3, 3].to(device=device, dtype=dtype)

        # per link: anchor and constant position offset in the anchor frame
        link_anchor = torch.zeros(self.n_links, dtype=torch.long)
        offsets = torch.zeros(self.n_links, 3, dtype=torch.float64)
        for r in range(self.n_links):
            a_row = anchor_row(r)
            a = anchor_index[a_row]
            link_anchor[r] = a
            Tc = torch.linalg.inv(T0[names[a_row]]) @ T0[names[r]]
            offsets[r] = Tc[:3, 3]
        # sort links by anchor so each anchor owns a contiguous slice; keep the inverse permutation
        order = torch.argsort(link_anchor, stable=True)
        self.inv_order = torch.argsort(order).to(device)   # tree row -> sorted position
        self.order = order.to(device)                       # sorted position -> tree row
        self.group_sizes = [int((link_anchor == a).sum()) for a in range(len(anchors))]
        self.offsets_sorted = offsets[order].to(device=device, dtype=dtype)           # (L, 3)
        self.link_anchor_sorted = link_anchor[order]

        # joints: constants between the anchors, joint transform parameters
        self.parent_anchor, self.C_R, self.C_t = [], [], []
        fixed_R, trans, axes, lower, upper = [], [], [], [], []
        for k, i in enumerate(tree._controlled_joints):
            body = bodies[i]
            p = parent_of(body)
            p_row = name_to_row[p.name]
            pa_row = anchor_row(p_row)
            self.parent_anchor.append(anchor_index[pa_row])
            Tc = torch.linalg.inv(T0[names[pa_row]]) @ T0[names[p_row]]
            self.C_R.append(Tc[:3, :3].to(device=device, dtype=dtype))
            self.C_t.append(Tc[:3, 3].to(device=device, dtype=dtype))
            fixed_R.append(body.fixed_rotation.reshape(3, 3).to(device=device, dtype=dtype))
            trans.append(body.trans.reshape(3).to(device=device, dtype=dtype))
            axes.append(body.joint_axis.reshape(3).to(device=device, dtype=dtype))
            if body.joint_limits is not None:
                lower.append(float(body.joint_limits['lower']))
                upper.append(float(body.joint_limits['upper']))
            else:
                lower.append(-float('inf'))
                upper.append(float('inf'))
        self.fixed_R = torch.stack(fixed_R)                                        # (n_dof, 3, 3)
        self.trans = torch.stack(trans)                                            # (n_dof, 3)
        self.axes = torch.stack(axes)                                              # (n_dof, 3), sign included
        self.lower = torch.tensor(lower, device=device, dtype=dtype)
        self.upper = torch.tensor(upper, device=device, dtype=dtype)

        # Rodrigues terms: R(q) = c I + s [w]x + (1 - c) w w^T  (rotation by q about the signed axis w)
        w = self.axes
        K = torch.zeros(self.n_dof, 3, 3, device=device, dtype=dtype)
        K[:, 0, 1], K[:, 0, 2] = -w[:, 2], w[:, 1]
        K[:, 1, 0], K[:, 1, 2] = w[:, 2], -w[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -w[:, 1], w[:, 0]
        self.rod_K = K
        self.rod_ww = w.unsqueeze(-1) * w.unsqueeze(-2)
        self.eye3 = torch.eye(3, device=device, dtype=dtype)

        # masks: link s is below joint k  <=>  joint k's anchor is an ancestor-or-self of s's anchor
        anchor_parent = {}
        for k in range(self.n_dof):
            anchor_parent[k + 1] = self.parent_anchor[k]
        below = torch.zeros(self.n_dof, self.n_links)
        for s in range(self.n_links):
            a = int(self.link_anchor_sorted[s])
            while a != 0:
                below[a - 1, s] = 1.0
                a = anchor_parent[a]
        self.below_mask = below.to(device=device, dtype=dtype)                    # (n_dof, L) in sorted link order

    # ---------------------------------------------------------------------------------------------------------
    def _chain(self, q):
        """q: (..., n_dof) -> world rotations (..., n_anchors, 3, 3) and translations (..., n_anchors, 3)"""
        shape = q.shape[:-1]
        q_c = torch.clamp(q, min=self.lower, max=self.upper)
        c, s = torch.cos(q_c), torch.sin(q_c)
        rot = (c[..., None, None] * self.eye3 + s[..., None, None] * self.rod_K
               + (1. - c)[..., None, None] * self.rod_ww)                          # (..., n_dof, 3, 3)
        R_J = bmm3(self.fixed_R, rot)                                              # fixed rotation * joint rotation
        R_w = [self.root_R.expand(*shape, 3, 3)]
        t_w = [self.root_t.expand(*shape, 3)]
        for k in range(self.n_dof):
            pa = self.parent_anchor[k]
            Rp, tp = R_w[pa], t_w[pa]
            Rpar = bmm3(Rp, self.C_R[k])
            tpar = bmv3(Rp, self.C_t[k]) + tp
            R_w.append(bmm3(Rpar, R_J[..., k, :, :]))
            t_w.append(bmv3(Rpar, self.trans[k]) + tpar)
        return torch.stack(R_w, dim=-3), torch.stack(t_w, dim=-2), q_c

    @torch.no_grad()
    def forward(self, q):
        """
        q: (..., n_dof). Returns link positions (..., n_links, 3) in the link order of the tree and a frames tuple
        for vjp().
        """
        R_w, t_w, q_c = self._chain(q)
        pos_groups, start = [], 0
        for a, n in enumerate(self.group_sizes):
            if n == 0:
                continue
            off = self.offsets_sorted[start:start + n]                              # (n, 3)
            pos_groups.append(bmv3(R_w[..., a, :, :].unsqueeze(-3), off) + t_w[..., a, :].unsqueeze(-2))
            start += n
        pos = torch.cat(pos_groups, dim=-2)[..., self.inv_order, :]
        return pos, (R_w, t_w, q, pos)

    @torch.no_grad()
    def vjp(self, frames, g):
        """
        g: (..., n_links, 3) cotangent of the link positions (any extra leading dims, e.g. one per cost field, are
        broadcast against the frames). Returns the gradient w.r.t. q: (..., n_dof).
        """
        R_w, t_w, q, pos = frames
        n_extra = g.dim() - pos.dim()
        pos_s = pos                                                                        # tree link order
        # joint world axes / origins
        w_world = bmv3(R_w[..., 1:, :, :], self.axes)                                     # (..., n_dof, 3)
        o_world = t_w[..., 1:, :]
        for _ in range(n_extra):
            pos_s, w_world, o_world = pos_s.unsqueeze(0), w_world.unsqueeze(0), o_world.unsqueeze(0)
        # masks are in sorted link order -> bring g / positions to sorted order
        g_s = g[..., self.order, :]
        p_s = pos_s[..., self.order, :]
        moment = torch.cross(p_s, g_s, dim=-1)                                             # (..., L, 3)
        M = torch.einsum('kl,...lc->...kc', self.below_mask, moment)                       # sum_s p_s x g_s
        Fsum = torch.einsum('kl,...lc->...kc', self.below_mask, g_s)                       # sum_s g_s
        tau = (w_world * (M - torch.cross(o_world, Fsum, dim=-1))).sum(-1)                 # (..., n_dof)
        inside = (q >= self.lower) & (q <= self.upper)                                      # clamp passes gradient
        for _ in range(n_extra):
            inside = inside.unsqueeze(0)
        return tau * inside.to(tau.dtype)
