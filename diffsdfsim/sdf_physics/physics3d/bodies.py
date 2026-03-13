#
# Copyright 2024 Max-Planck-Gesellschaft
# Code author: Michael Strecke, michael.strecke@tuebingen.mpg.de
# Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math

import numpy as np
import ode
import pyrender
import torch
import trimesh
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, so3_exponential_map, quaternion_multiply, \
    quaternion_apply
from ev_sdf_utils import marching_cubes, grid_interp
from torch.autograd import Function
from torch.nn.functional import normalize

from diffsdfsim.lcp_physics.physics import Body
from diffsdfsim.lcp_physics.physics.utils import polar_to_cart
from .utils import Defaults3D, get_tensor, spherical_to_cart, quat

DIM = Defaults3D.DIM


def box_sdf(pts, dims):
    half_dims = dims / 2

    q = torch.abs(pts) - half_dims

    max_dist = torch.max(q, dim=1)[0]

    m = torch.clamp(q, min=0.)
    sdfs = m.norm(dim=1) + torch.clamp(max_dist, max=0.)

    return sdfs


def box_sdf_grad(pts, dims):
    half_dims = dims / 2

    q = torch.abs(pts) - half_dims
    signs = torch.sign(pts)
    signs[signs == 0] = 1

    max_dist = torch.max(q, dim=1)[0]
    max_dirs = max_dist.new_zeros(pts.shape)
    # The box SDF has gradient discontinuities on the surface and the inside when more than one dim of q has the same
    # value (e.g. corners, edges on the surface). The line below generates a gradient which is pointing along the
    # corresponding 2D/3D diagonal. This should be provide a (possibly inaccurate) "failsafe" normal that pushes objects
    # apart in edge-edge collisions.
    max_dirs[q == max_dist.unsqueeze(1)] = 1.

    m = torch.max(q, q.new_zeros(1))

    grads_ov = (normalize(m, dim=1) + (max_dist <= 0).type_as(max_dirs).unsqueeze(1) * max_dirs) * signs

    grads = normalize(grads_ov, dim=1)

    return grads


def sphere_sdf(pts, rad):
    pts_norm = pts.norm(dim=1)

    sdfs = pts_norm - rad

    return sdfs


def sphere_sdf_grad(pts, rad):
    return normalize(pts, dim=1)


def cylinder_sdf(pts, rad, height):
    half_height = height / 2

    ps = torch.stack([pts[:, :2].norm(dim=1), pts[:, 2]], dim=1)

    q = torch.abs(ps) - torch.stack([rad, half_height])

    max_dist = torch.max(q, dim=1)[0]

    m = torch.clamp(q, min=0.)

    sdfs = m.norm(dim=1) + torch.clamp(max_dist, max=0.)

    return sdfs


def cylinder_sdf_grad(pts, rad, height):
    half_height = height / 2

    ps = torch.stack([pts[:, :2].norm(dim=1), pts[:, 2]], dim=1)

    q = torch.abs(ps) - torch.stack([rad, half_height])
    signs = torch.sign(pts[:, 2])
    signs[signs == 0] = 1

    max_dist = torch.max(q, dim=1)[0]
    max_dirs = max_dist.new_zeros(q.shape)
    # "Failsafe", see box grad.
    max_dirs[q == max_dist.unsqueeze(1)] = 1.

    m = torch.clamp(q, min=0.)

    grads = normalize(m, dim=1) + (max_dist <= 0).type_as(max_dirs).unsqueeze(1) * max_dirs
    grads = torch.cat([grads[:, 0].unsqueeze(1) * normalize(pts[:, :2], dim=1),
                       (grads[:, 1] * signs).unsqueeze(1)], dim=1)

    grads = normalize(grads, dim=1)

    return grads


def bowl_sdf(pts, r, d):
    pts[:, 2] = pts[:, 2] - r / 2

    ps = torch.stack([pts[:, :2].norm(dim=1), pts[:, 2]], dim=1)

    ps_norm = ps.norm(dim=1)

    ps = ps.clone()
    ps[ps[:, 1] < 0, 0] = ps_norm[ps[:, 1] < 0]

    ps[:, 0] = (ps[:, 0] - r).abs() - d

    sdfs = torch.max(ps, ps.new_zeros(1)).norm(dim=1) + torch.min(ps.new_zeros(1), ps.max(dim=1)[0])

    return sdfs


def bowl_sdf_grad(pts, r, d):
    pts[:, 2] = pts[:, 2] - r / 2

    ps = torch.stack([pts[:, :2].norm(dim=1), pts[:, 2]], dim=1)

    ps_norm = ps.norm(dim=1)

    ps = ps.clone()
    ps[ps[:, 1] < 0, 0] = ps_norm[ps[:, 1] < 0]

    ps[:, 0] = (ps[:, 0] - r).abs() - d

    grads = pts * (ps_norm - r).sign().unsqueeze(1)
    grads[(ps[:, 1] >= 0) & (ps[:, 0] < 0), :2] = 0
    grads[ps[:, 1] >= 0, 2] = grads[ps[:, 1] >= 0, 2].abs()

    grads = normalize(grads, dim=1)

    return grads


def rounded_sdf(base_func):
    def round_corners(pts, *params):
        r = params[0]
        base_params = params[1:]
        return base_func(pts, *base_params) - r

    return round_corners


def rounded_sdf_grad(base_grad_func):
    def round_grad(pts, *params):
        r = params[0]
        base_params = params[1:]
        return base_grad_func(pts, *base_params)

    return round_grad


def brick_sdf(pts, dims, r):
    half_dims = dims / 2
    half_dims[:2] -= r

    # "in-plane" sdf for first two dimensions: with rounded corners
    q = pts.abs() - half_dims
    max_dist_01 = q[:, :2].max(dim=1)[0]
    m_01 = q[:, :2].clamp(min=0.)
    sdfs_01 = m_01.norm(dim=1) + max_dist_01.clamp(max=0.) - r

    # Remaining sdfs built from "in-plane" sdf and box for last dim
    q = torch.stack([sdfs_01, q[:, 2]], dim=1)
    max_dist = q.max(dim=1)[0]
    m = q.clamp(min=0.)
    sdfs = m.norm(dim=1) + max_dist.clamp(max=0.)

    return sdfs


def grid_sdf(pts, sdf):
    inds = (pts + 1.) * 0.5 * (pts.new_tensor(sdf.shape) - 1)
    valid_mask = torch.all((inds <= (inds.new_tensor(sdf.shape) - 1)) & (inds >= 0), dim=1)

    sdfs = sdf.new_ones(pts.shape[0])

    sdfs[valid_mask] = grid_interp(sdf, inds[valid_mask])

    return sdfs


def grid_sdf_grad(pts, sdf):
    # Forward differences
    # sdf_grads = torch.stack([torch.cat([sdf[1:, :, :] - sdf[:-1, :, :],
    #                                     sdf.new_zeros((1, sdf.shape[1], sdf.shape[2]))], dim=0),
    #                          torch.cat([sdf[:, 1:, :] - sdf[:, :-1, :],
    #                                     sdf.new_zeros((sdf.shape[0], 1, sdf.shape[2]))], dim=1),
    #                          torch.cat([sdf[:, :, 1:] - sdf[:, :, :-1],
    #                                     sdf.new_zeros((sdf.shape[0], sdf.shape[1], 1))], dim=2),
    #                          ], dim=0)

    # Central differences
    sdf_grads = torch.stack([torch.cat([sdf.new_zeros((1, sdf.shape[1], sdf.shape[2])),
                                        (sdf[2:, :, :] - sdf[:-2, :, :]) / 2,
                                        sdf.new_zeros((1, sdf.shape[1], sdf.shape[2]))], dim=0),
                             torch.cat([sdf.new_zeros((sdf.shape[0], 1, sdf.shape[2])),
                                        (sdf[:, 2:, :] - sdf[:, :-2, :]) / 2,
                                        sdf.new_zeros((sdf.shape[0], 1, sdf.shape[2]))], dim=1),
                             torch.cat([sdf.new_zeros((sdf.shape[0], sdf.shape[1], 1)),
                                        (sdf[:, :, 2:] - sdf[:, :, :-2]) / 2,
                                        sdf.new_zeros((sdf.shape[0], sdf.shape[1], 1))], dim=2),
                             ], dim=0)

    inds = (pts + 1.) * 0.5 * (pts.new_tensor(sdf.shape) - 1)
    valid_mask = torch.all((inds <= (inds.new_tensor(sdf.shape) - 1)) & (inds >= 0), dim=1)

    grads = sdf.new_zeros(pts.shape)

    grads[valid_mask] = normalize(grid_interp(sdf_grads, inds[valid_mask]), dim=1)

    return grads


class DiffGridSDF(Function):
    @staticmethod
    def forward(ctx, pts, sdf):
        ctx.save_for_backward(pts, sdf)
        ctx.mark_non_differentiable(sdf)
        return grid_sdf(pts, sdf)

    @staticmethod
    def backward(ctx, d_sdf):
        pts, sdf = ctx.saved_tensors
        grads = grid_sdf_grad(pts, sdf)
        return grads * d_sdf.unsqueeze(1), None


def comp_projection_integrals(verts, faces, A, B):
    a0 = verts[faces][torch.arange(faces.shape[0]), :, A]
    b0 = verts[faces][torch.arange(faces.shape[0]), :, B]
    a1 = verts[faces[:, [1, 2, 0]]][torch.arange(faces.shape[0]), :, A]
    b1 = verts[faces[:, [1, 2, 0]]][torch.arange(faces.shape[0]), :, B]
    da = a1 - a0
    db = b1 - b0
    a0_2 = a0 * a0
    a0_3 = a0_2 * a0
    a0_4 = a0_3 * a0
    b0_2 = b0 * b0
    b0_3 = b0_2 * b0
    b0_4 = b0_3 * b0
    a1_2 = a1 * a1
    a1_3 = a1_2 * a1
    b1_2 = b1 * b1
    b1_3 = b1_2 * b1

    C1 = a1 + a0
    Ca = a1 * C1 + a0_2
    Caa = a1 * Ca + a0_3
    Caaa = a1 * Caa + a0_4
    Cb = b1 * (b1 + b0) + b0_2
    Cbb = b1 * Cb + b0_3
    Cbbb = b1 * Cbb + b0_4
    Cab = 3 * a1_2 + 2 * a1 * a0 + a0_2
    Kab = a1_2 + 2 * a1 * a0 + 3 * a0_2
    Caab = a0 * Cab + 4 * a1_3
    Kaab = a1 * Kab + 4 * a0_3
    Cabb = 4 * b1_3 + 3 * b1_2 * b0 + 2 * b1 * b0_2 + b0_3
    Kabb = b1_3 + 2 * b1_2 * b0 + 3 * b1 * b0_2 + 4 * b0_3

    P1 = (db * C1).sum(dim=1) / 2.0
    Pa = (db * Ca).sum(dim=1) / 6.0
    Paa = (db * Caa).sum(dim=1) / 12.0
    Paaa = (db * Caaa).sum(dim=1) / 20.0
    Pb = (da * Cb).sum(dim=1) / -6.0
    Pbb = (da * Cbb).sum(dim=1) / -12.0
    Pbbb = (da * Cbbb).sum(dim=1) / -20.0
    Pab = (db * (b1 * Cab + b0 * Kab)).sum(dim=1) / 24.0
    Paab = (db * (b1 * Caab + b0 * Kaab)).sum(dim=1) / 60.0
    Pabb = (da * (a1 * Cabb + a0 * Kabb)).sum(dim=1) / -60.0

    return P1, Pa, Paa, Paaa, Pb, Pbb, Pbbb, Pab, Paab, Pabb


def comp_face_integrals(verts, faces, normals, w, A, B, C):
    P1, Pa, Paa, Paaa, Pb, Pbb, Pbbb, Pab, Paab, Pabb = comp_projection_integrals(verts, faces, A, B)

    k1 = 1 / normals[torch.arange(normals.shape[0]), C]
    k2 = k1 * k1
    k3 = k2 * k1
    k4 = k3 * k1

    nA = normals[torch.arange(normals.shape[0]), A]
    nB = normals[torch.arange(normals.shape[0]), B]

    Fa = k1 * Pa
    Fb = k1 * Pb
    Fc = -k2 * (nA * Pa + nB * Pb + w * P1)

    Faa = k1 * Paa
    Fbb = k1 * Pbb
    Fcc = k3 * (nA * nA * Paa + 2 * nA * nB * Pab + nB * nB * Pbb
                + w * (2 * (nA * Pa + nB * Pb) + w * P1))

    Faaa = k1 * Paaa
    Fbbb = k1 * Pbbb
    Fccc = -k4 * (nA ** 3 * Paaa + 3 * nA * nA * nB * Paab
                  + 3 * nA * nB * nB * Pabb + nB * nB * nB * Pbbb
                  + 3 * w * (nA * nA * Paa + 2 * nA * nB * Pab + nB * nB * Pbb)
                  + w * w * (3 * (nA * Pa + nB * Pb) + w * P1))

    Faab = k1 * Paab
    Fbbc = -k2 * (nA * Pabb + nB * Pbbb + w * Pbb)
    Fcca = k3 * (nA * nA * Paaa + 2 * nA * nB * Paab + nB * nB * Pabb
                 + w * (2 * (nA * Paa + nB * Pab) + w * Pa))

    return Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca


def comp_volume_integrals(verts, faces, normals, w):
    C = torch.argmax(normals.abs(), dim=1)
    A = (C + 1) % 3
    B = (A + 1) % 3

    Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca = comp_face_integrals(verts, faces, normals, w, A, B,
                                                                                        C)
    T0 = verts.new_zeros(faces.shape[0])
    T0[A == 0] = normals[A == 0, 0] * Fa[A == 0]
    T0[B == 0] = normals[B == 0, 0] * Fb[B == 0]
    T0[C == 0] = normals[C == 0, 0] * Fc[C == 0]

    T0 = T0.sum()

    normA = normals[torch.arange(normals.shape[0]), A]
    normB = normals[torch.arange(normals.shape[0]), B]
    normC = normals[torch.arange(normals.shape[0]), C]

    T1 = verts.new_zeros(faces.shape[0], 3)
    T1[torch.arange(faces.shape[0]), A] = normA * Faa
    T1[torch.arange(faces.shape[0]), B] = normB * Fbb
    T1[torch.arange(faces.shape[0]), C] = normC * Fcc
    T1 = T1.sum(dim=0) / 2

    T2 = verts.new_zeros(faces.shape[0], 3)
    T2[torch.arange(faces.shape[0]), A] = normA * Faaa
    T2[torch.arange(faces.shape[0]), B] = normB * Fbbb
    T2[torch.arange(faces.shape[0]), C] = normC * Fccc
    T2 = T2.sum(dim=0) / 3

    TP = verts.new_zeros(faces.shape[0], 3)
    TP[torch.arange(faces.shape[0]), A] = normA * Faab
    TP[torch.arange(faces.shape[0]), B] = normB * Fbbc
    TP[torch.arange(faces.shape[0]), C] = normC * Fcca
    TP = TP.sum(dim=0) / 2

    return T0, T1, T2, TP


def get_ang_inertia(verts, faces, mass):
    # https://github.com/OpenFOAM/OpenFOAM-2.1.x/blob/master/src/meshTools/momentOfInertia/volumeIntegration/volInt.c
    normals = torch.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 1]], dim=1)
    normals = normals / normals.norm(dim=1).unsqueeze(1)
    w = (-normals * verts[faces[:, 0]]).sum(dim=1)

    T0, T1, T2, TP = comp_volume_integrals(verts, faces, normals, w)

    density = mass / T0

    J = torch.diag(density * (T2[[1, 2, 0]] + T2[[2, 0, 1]]))
    J[0, 1] = J[1, 0] = -density * TP[0]
    J[1, 2] = J[2, 1] = -density * TP[1]
    J[2, 0] = J[0, 2] = -density * TP[2]

    return J


class Body3D(Body):
    """Base class for bodies in 3D
    """

    def __init__(self, pos, vel=(0, 0, 0, 0, 0, 0), mass=1, restitution=Defaults3D.RESTITUTION,
                 fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON, COM = (0,0,0),
                 col=(255, 255, 255), thickness=1, smooth=False, is_transparent=False):
        # get base tensor to define dtype, device and layout for others
        self._set_base_tensor(locals().values())

        self.eps = get_tensor(eps, base_tensor=self._base_tensor)
        # rotation & position vectors
        pos = get_tensor(pos, base_tensor=self._base_tensor)
        if pos.size(0) == 3:
            self.p = torch.cat([pos.new_ones(1), pos.new_zeros(3), pos])
        elif pos.size(0) == 6:
            q = quat(pos[:3], "wxyz")
            self.p = torch.cat([q, pos[3:]])
        else:
            self.p = pos
        self.rot = self.p[:4]
        self.pos = self.p[4:]

        # linear and angular velocity vector
        vel = get_tensor(vel, base_tensor=self._base_tensor)
        if vel.size(0) == 3:
            self.v = torch.cat([vel.new_zeros(3), vel])
        else:
            self.v = vel

        self.mass = get_tensor(mass, self._base_tensor)
        self.ang_inertia = self._get_ang_inertia(self.mass)
        # M can change if object rotates, not the case for now
        self.M = self.v.new_zeros(len(self.v), len(self.v))
        ang_sizes = [3, 3]
        rot_mat = quaternion_to_matrix(self.rot)
        self.M[:ang_sizes[0], :ang_sizes[1]] = rot_mat @ self.ang_inertia @ rot_mat.t()
        self.M[ang_sizes[0]:, ang_sizes[1]:] = torch.eye(DIM).type_as(self.M) * self.mass

        self.fric_coeff = get_tensor(fric_coeff, base_tensor=self._base_tensor)
        self.restitution = get_tensor(restitution, base_tensor=self._base_tensor)
        self.forces = []

        self.col = col
        self.thickness = thickness

        ## new center of mass

        self.com = get_tensor(COM, base_tensor=self._base_tensor)

        if all([isinstance(c, int) for c in col]):
            color = [c / 255. for c in col]
        else:
            color = col
        alpha_mode = 'OPAQUE'
        if is_transparent:
            color += [0.5]
            alpha_mode = 'BLEND'

        self.mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(self.verts.detach().cpu(), self.faces.detach().cpu(),
                                                               vertex_colors=color),
                                               smooth=smooth,
                                               material=pyrender.MetallicRoughnessMaterial(metallicFactor=0.5,
                                                                                           roughnessFactor=0.75,
                                                                                           baseColorFactor=color,
                                                                                           alphaMode=alpha_mode))

        self._create_geom()

    def _set_base_tensor(self, args):
        """Check if any tensor provided and if so set as base tensor to
           use as base for other tensors' dtype, device and layout.
        """
        if hasattr(self, '_base_tensor') and self._base_tensor is not None:
            return

        for arg in args:
            if isinstance(arg, torch.Tensor):
                self._base_tensor = arg
                return

        # if no tensor provided, use defaults
        self._base_tensor = get_tensor(0, base_tensor=None)
        return

    def _create_geom(self):
        raise NotImplementedError

    def _get_ang_inertia(self, mass):
        raise NotImplementedError

    def get_surface(self):
        raise NotImplementedError

    def move(self, dt, update_geom_rotation=True):
        new_p = torch.cat([quaternion_multiply(matrix_to_quaternion(so3_exponential_map(self.v[:3].unsqueeze(0) * dt)),
                                               self.p[:4]).squeeze(),
                           self.p[4:] + self.v[3:] * dt])
        # new_p = torch.cat([matrix_to_quaternion(
        #     so3_exponential_map(self.v[:3].unsqueeze(0) * dt) @ quaternion_to_matrix(self.p[:4])).squeeze(),
        #                    self.p[4:] + self.v[3:] * dt])
        # new_p = torch.cat([quat_cross(quat_exp(self.v[:3] * 0.5 * dt), self.p[:4]), self.p[4:] + self.v[3:] * dt])
        self.set_p(new_p, update_geom_rotation)

    def set_p(self, new_p, update_geom_rotation=True, update_ang_inertia=True):
        self.p = new_p
        # Reset memory pointers
        self.rot = self.p[0:4]
        self.pos = self.p[4:]

        self.geom.setPosition(self.pos)
        if update_geom_rotation:
            # q = torch.cat([self.rot[-1:], self.rot[:-1]])
            self.geom.setQuaternion(self.rot)

        if update_ang_inertia:
            rot_mat = quaternion_to_matrix(self.rot)
            self.M = torch.block_diag(rot_mat @ self.ang_inertia @ rot_mat.t(), self.M[3:, 3:])

    def draw(self, screen, pixels_per_meter=1):
        pass

    def render(self, scene, t=0, render_forces=False, render_torques=False, force_col=(1., 1., 1., 0.5),
               torque_col=(1., 1., 1., 0.5), force_scale=1., torque_scale=1.):
        rot_mat = quaternion_to_matrix(self.rot)
        T = torch.eye(DIM + 1, dtype=Defaults3D.DTYPE)
        T[:3, :3] = rot_mat
        T[:3, 3] = self.pos
        # Render bounding box
        # scene.add(
        #     pyrender.Mesh.from_trimesh(trimesh.creation.box(self.scale.detach().expand(DIM).cpu() * 2, T.detach()),
        #                                wireframe=True))

        # Render mesh
        mesh_node = scene.add(self.mesh, pose=T.detach())
        if render_forces or render_torques:
            T[:3, :3] = torch.eye(DIM, dtype=T.dtype)
            force = self.forces[-1].force(t) if self.forces else self.apply_forces(t)
            if render_forces and force[-3:].norm() > 0.:
                rot = trimesh.geometry.align_vectors(force[-3:].cpu().detach(),
                                                     np.array([0, 0, force[-3:].norm().item()]))
                cylinder = pyrender.Mesh.from_trimesh(
                    trimesh.creation.cylinder(0.1, force[-3:].norm().item() * force_scale,
                                              transform=rot.T @ np.array([[1, 0, 0, 0],
                                                                          [0, 1, 0, 0],
                                                                          [0, 0, 1,
                                                                           force[-3:].norm().item() * force_scale / 2],
                                                                          [0, 0, 0, 1]]),
                                              vertex_colors=force_col),
                    smooth=True,
                    material=pyrender.MetallicRoughnessMaterial(metallicFactor=0.5,
                                                                roughnessFactor=0.75,
                                                                baseColorFactor=force_col,
                                                                alphaMode='BLEND'))
                scene.add(cylinder, pose=T.detach())
                cone = pyrender.Mesh.from_trimesh(
                    trimesh.creation.cone(0.2, 0.3, transform=rot.T @ np.array([[1, 0, 0, 0],
                                                                                [0, 1, 0, 0],
                                                                                [0, 0, 1,
                                                                                 force[
                                                                                 -3:].norm().item() * force_scale],
                                                                                [0, 0, 0, 1]]),
                                          vertex_colors=force_col),
                    smooth=True,
                    material=pyrender.MetallicRoughnessMaterial(metallicFactor=0.5,
                                                                roughnessFactor=0.75,
                                                                baseColorFactor=force_col,
                                                                alphaMode='BLEND'))
                scene.add(cone, pose=T.detach())
            if render_torques and force[:3].norm() > 0:
                rot = trimesh.geometry.align_vectors(force[:3].cpu().detach(),
                                                     np.array([0, 0, force[:3].norm().item()]))
                cylinder = pyrender.Mesh.from_trimesh(
                    trimesh.creation.cylinder(0.1, force[:3].norm().item() * torque_scale,
                                              transform=rot.T @ np.array([[1, 0, 0, 0],
                                                                          [0, 1, 0, 0],
                                                                          [0, 0, 1,
                                                                           force[:3].norm().item() * torque_scale / 2],
                                                                          [0, 0, 0, 1]]),
                                              vertex_colors=torque_col),
                    smooth=True,
                    material=pyrender.MetallicRoughnessMaterial(metallicFactor=0.5,
                                                                roughnessFactor=0.75,
                                                                baseColorFactor=torque_col,
                                                                alphaMode='BLEND'))
                scene.add(cylinder, pose=T.detach())
                cone = pyrender.Mesh.from_trimesh(
                    trimesh.creation.cone(0.2, 0.3, transform=rot.T @ np.array([[1, 0, 0, 0],
                                                                                [0, 1, 0, 0],
                                                                                [0, 0, 1,
                                                                                 force[:3].norm().item() * torque_scale],
                                                                                [0, 0, 0, 1]]),
                                          vertex_colors=torque_col),
                    smooth=True,
                    material=pyrender.MetallicRoughnessMaterial(metallicFactor=0.5,
                                                                roughnessFactor=0.75,
                                                                baseColorFactor=torque_col,
                                                                alphaMode='BLEND'))
                scene.add(cone, pose=T.detach())
        return mesh_node


class Mesh3D(Body3D):
    def __init__(self, pos, verts, faces, vel=(0, 0, 0, 0, 0, 0), mass=1,
                 restitution=Defaults3D.RESTITUTION, fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON, COM = (0,0,0),
                 col=(255, 255, 255), thickness=1, smooth=False, is_transparent=False):
        # self._set_base_tensor(locals().values())
        self._base_tensor = get_tensor(0, base_tensor=None)
        pos = get_tensor(pos, base_tensor=self._base_tensor)
        self.verts = verts
        self.faces = faces

        super().__init__(pos=pos, vel=vel, mass=mass, restitution=restitution, fric_coeff=fric_coeff, eps=eps, COM = COM,
                         col=col, thickness=thickness, smooth=smooth, is_transparent=is_transparent)

        if pos.size(0) != 3:
            self.set_p(self.p)

    def _create_geom(self):
        scale = self._base_tensor.new_tensor(self.mesh.bounds.max())
        # Use bounding box for broadphase
        self.geom = ode.GeomBox(None, scale.expand(3) * 2 + 2 * self.eps.item())
        self.geom.setPosition(self.pos)
        self.geom.no_contact = set()

    def _get_ang_inertia(self, mass):
        return get_ang_inertia(self.verts, self.faces, mass)

    def get_surface(self):
        verts = quaternion_apply(self.rot, self.verts) + self.pos
        return verts, self.faces


class SDF3D(Body3D):
    def __init__(self, pos, scale, sdf_func, params, grad_func=None, vel=(0, 0, 0, 0, 0, 0), mass=1,
                 restitution=Defaults3D.RESTITUTION, fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON, COM=(0, 0, 0),
                 col=(255, 255, 255), thickness=1, smooth=False, is_transparent=False):
        self._set_base_tensor(locals().values())
        self.scale = get_tensor(scale, base_tensor=self._base_tensor)
        pos = get_tensor(pos, base_tensor=self._base_tensor)
        self.sdf_func = sdf_func
        self.grad_func = grad_func
        self.params = params

        self.verts, self.faces = self._create_mesh()

        super().__init__(pos=pos, vel=vel, mass=mass, restitution=restitution, fric_coeff=fric_coeff, eps=eps, COM=COM,
                         col=col, thickness=thickness, smooth=smooth, is_transparent=is_transparent)

        if pos.size(0) != 3:
            self.set_p(self.p)

    def _create_geom(self):
        # Use bounding box for broadphase
        self.geom = ode.GeomBox(None, self.scale.expand(3) * 2 + 2 * self.eps.item())
        self.geom.setPosition(self.pos)
        self.geom.no_contact = set()

    @staticmethod
    def _diff_marching_cubes(sdf_func, res=128):
        class MeshSDF(Function):
            @staticmethod
            def forward(ctx, *sdf_params):
                samp_space = torch.linspace(-1., 1., res, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)
                samples = torch.stack(torch.meshgrid(samp_space, samp_space, samp_space), dim=3).reshape(-1, 3)

                sdfs = sdf_func(samples, *sdf_params)

                sdfs = sdfs.reshape(res, res, res)

                verts, faces = marching_cubes(sdfs, 0.0)

                # Transform vertices to local coordinate system
                verts = verts / (res - 1) * 2. - 1.

                ctx.save_for_backward(verts.detach(), *sdf_params)
                ctx.mark_non_differentiable(faces)

                return verts, faces

            @staticmethod
            def backward(ctx, grad_v, grad_f):
                params = ctx.saved_tensors

                with torch.enable_grad():
                    for param in params:
                        if not param.requires_grad:
                            param.requires_grad = True
                        if not param.is_leaf:
                            param.retain_grad()

                    sdfs = sdf_func(*params)

                    verts_grads = torch.autograd.grad(sdfs, params[0], sdfs.new_ones(sdfs.shape), retain_graph=True)[0]

                    normals = normalize(verts_grads, dim=1)

                    dL_ds_i = - (grad_v.unsqueeze(1) @ normals.unsqueeze(-1)).squeeze(-1)

                    loss_dz = (dL_ds_i * sdfs.view(sdfs.shape[0], -1)).sum()
                    dL_dz = torch.autograd.grad(loss_dz, params[1:], loss_dz.new_tensor(1.))

                    # The following lines would be the exact formulation from the MeshSDF paper. The above is equivalent
                    # and much more efficient. A similar implementation was used in MeshSDF reference code at
                    # https://github.com/cvlab-epfl/MeshSDF/blob/6d34bd73839fea29c7f45756b2d43c40fb84cf2f/demo_optimizer.py#L192
                    # param_grads = torch.autograd.functional.jacobian(lambda *p: sdf_func(params[0], *p), params[1:])
                    # dL_dz = [(dL_ds_i * param_grad.view(sdfs.shape[0], -1)).sum(dim=0) for param_grad in param_grads]

                return dL_dz

        return MeshSDF.apply

    def _create_mesh(self, max_vox_size=0.03):
        # TODO: mcubes seems not to work on CUDA for some sizes.
        # vol_size = self.scale * 2
        # res = (vol_size / max_vox_size).ceil().int()
        verts, faces = self._diff_marching_cubes(self.sdf_func)(*self.params)
        verts = verts * self.scale
        return verts, faces

    def _get_ang_inertia(self, mass):
        return get_ang_inertia(self.verts, self.faces, mass)

    def get_surface(self):
        verts = quaternion_apply(self.rot, self.verts) + self.pos
        return verts, self.faces

    def query_sdfs(self, pts_loc, return_grads=True, return_overlapmask=False):
        overlap_mask = torch.all(torch.abs(pts_loc) <= self.scale, dim=1)

        sdfs = self._base_tensor.new_ones(pts_loc.shape[0])
        grads_loc = self._base_tensor.new_zeros(pts_loc.shape)

        if torch.any(overlap_mask):
            pts_in = pts_loc[overlap_mask] / self.scale

            if return_grads and self.grad_func is None:
                with torch.enable_grad():
                    if not pts_in.requires_grad:
                        pts_in.requires_grad = True
                    if not pts_in.is_leaf:
                        pts_in.retain_grad()

                    sdfs[overlap_mask] = self.sdf_func(pts_in, *self.params)

                    grads_ov = torch.autograd.grad(sdfs[overlap_mask], pts_in, sdfs.new_ones(sdfs[overlap_mask].shape),
                                                   retain_graph=not pts_in.is_leaf)[0]

                    grads_loc[overlap_mask] = normalize(grads_ov, dim=1)

                    if pts_in.is_leaf:
                        sdfs = sdfs.detach()
            else:
                sdfs[overlap_mask] = self.sdf_func(pts_in, *self.params)
                if return_grads:
                    grads_loc[overlap_mask] = normalize(self.grad_func(pts_in, *self.params), dim=1)

        sdfs = sdfs * self.scale

        if return_grads and return_overlapmask:
            return sdfs, grads_loc, overlap_mask
        elif return_overlapmask:
            return sdfs, overlap_mask
        elif return_grads:
            return sdfs, grads_loc
        else:
            return sdfs


class SDFGrid3D(SDF3D):
    def __init__(self, pos, scale, sdf, vel=(0, 0, 0), mass=1, restitution=Defaults3D.RESTITUTION,
                 fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON, COM=(0, 0, 0),
                 col=(255, 255, 255), thickness=1, smooth=False, is_transparent=False):
        self._set_base_tensor(locals().values())
        self.sdf = get_tensor(sdf, base_tensor=self._base_tensor)

        super().__init__(pos, scale, sdf_func=DiffGridSDF().apply, params=[self.sdf], grad_func=grid_sdf_grad, vel=vel,
                         mass=mass, restitution=restitution, fric_coeff=fric_coeff, eps=eps, col=col, COM=COM,
                         thickness=thickness, smooth=smooth, is_transparent=is_transparent)

    def _diff_marching_cubes(self, sdf_func):
        return super()._diff_marching_cubes(sdf_func, res=self.sdf.shape[0])


class SDFBox(SDF3D):
    def __init__(self, pos, dims, vel=(0, 0, 0, 0, 0, 0), mass=1, restitution=Defaults3D.RESTITUTION,
                 fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON, custom_mesh=Defaults3D.CUSTOM_MESH,
                 custom_inertia=Defaults3D.CUSTOM_INERTIA, col=(255, 255, 255), thickness=1, smooth=False,
                 is_transparent=False):
        self._set_base_tensor(locals().values())
        self.dims = get_tensor(dims, base_tensor=self._base_tensor)
        scale = torch.max(self.dims) * 1.5 / 2

        if custom_inertia:
            self._get_ang_inertia = self._custom_get_ang_inertia
        if custom_mesh:
            self._create_mesh = self._custom_create_mesh

        super().__init__(pos, scale, sdf_func=box_sdf, params=[self.dims / scale], grad_func=box_sdf_grad, vel=vel,
                         mass=mass, restitution=restitution, fric_coeff=fric_coeff, eps=eps, col=col,
                         thickness=thickness, smooth=smooth, is_transparent=is_transparent)

    def _custom_get_ang_inertia(self, mass):
        return mass * torch.diag(self.dims[[1, 0, 0]] ** 2 + self.dims[[2, 2, 1]] ** 2) / 12

    def _custom_create_mesh(self, max_tri_length=0.1):
        half_dims = self.dims / 2
        nverts = (self.dims / max_tri_length).ceil().int() + 1

        w = torch.linspace(-half_dims[0].item(), half_dims[0].item(), nverts[0], dtype=Defaults3D.DTYPE,
                           device=Defaults3D.DEVICE)
        h = torch.linspace(-half_dims[1].item(), half_dims[1].item(), nverts[1], dtype=Defaults3D.DTYPE,
                           device=Defaults3D.DEVICE)
        d = torch.linspace(-half_dims[2].item(), half_dims[2].item(), nverts[2], dtype=Defaults3D.DTYPE,
                           device=Defaults3D.DEVICE)
        # Replace ends of linspace with original coordinates for gradients
        w[0] = -half_dims[0]
        w[-1] = half_dims[0]
        h[0] = -half_dims[1]
        h[-1] = half_dims[1]
        d[0] = -half_dims[2]
        d[-1] = half_dims[2]

        fb = torch.stack(torch.meshgrid(w, h), dim=2).reshape(-1, 2)
        lr = torch.stack(torch.meshgrid(h, d), dim=2).reshape(-1, 2)
        tb = torch.stack(torch.meshgrid(w, d), dim=2).reshape(-1, 2)

        fb_inds = torch.arange(fb.shape[0]).reshape(nverts[0], nverts[1])
        lr_inds = torch.arange(lr.shape[0]).reshape(nverts[1], nverts[2])
        tb_inds = torch.arange(tb.shape[0]).reshape(nverts[0], nverts[2])

        f_faces = torch.cat(
            [torch.stack([fb_inds[:-1, :-1], fb_inds[1:, :-1], fb_inds[:-1, 1:]], dim=2).reshape(-1, 3),
             torch.stack([fb_inds[1:, :-1], fb_inds[1:, 1:], fb_inds[:-1, 1:]], dim=2).reshape(-1, 3)])

        l_faces = torch.cat(
            [torch.stack([lr_inds[:-1, :-1], lr_inds[1:, :-1], lr_inds[:-1, 1:]], dim=2).reshape(-1, 3),
             torch.stack([lr_inds[1:, :-1], lr_inds[1:, 1:], lr_inds[:-1, 1:]], dim=2).reshape(-1, 3)])

        t_faces = torch.cat(
            [torch.stack([tb_inds[:-1, :-1], tb_inds[1:, :-1], tb_inds[:-1, 1:]], dim=2).reshape(-1, 3),
             torch.stack([tb_inds[1:, :-1], tb_inds[1:, 1:], tb_inds[:-1, 1:]], dim=2).reshape(-1, 3)])

        f = torch.cat([fb, half_dims[2] * fb.new_ones(fb.shape[0]).unsqueeze(1)], dim=1)
        ba = torch.cat([fb, -half_dims[2] * fb.new_ones(fb.shape[0]).unsqueeze(1)], dim=1)
        l = torch.cat([half_dims[0] * lr.new_ones(lr.shape[0]).unsqueeze(1), lr], dim=1)
        r = torch.cat([-half_dims[0] * lr.new_ones(lr.shape[0]).unsqueeze(1), lr], dim=1)
        t = torch.stack([tb[:, 0], half_dims[1] * tb.new_ones(tb.shape[0]), tb[:, 1]], dim=1)
        bo = torch.stack([tb[:, 0], -half_dims[1] * tb.new_ones(tb.shape[0]), tb[:, 1]], dim=1)

        verts = torch.cat([f, ba, l, r, t, bo])

        faces = torch.cat([f_faces, f_faces.flip(1) + f.shape[0],
                           l_faces + 2 * f.shape[0], l_faces.flip(1) + 2 * f.shape[0] + l.shape[0],
                           t_faces.flip(1) + 2 * f.shape[0] + 2 * l.shape[0],
                           t_faces + 2 * f.shape[0] + 2 * l.shape[0] + t.shape[0]])

        # verts, inds = verts.unique(dim=0, return_inverse=True)
        # faces = inds[faces]

        return verts, faces


class SDFBoxRounded(SDF3D):
    def __init__(self, pos, dims, r, vel=(0, 0, 0, 0, 0, 0), mass=1, restitution=Defaults3D.RESTITUTION,
                 fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON,
                 col=(255, 255, 255), thickness=1, smooth=False, is_transparent=False):
        self._set_base_tensor(locals().values())
        self.dims = get_tensor(dims, base_tensor=self._base_tensor)
        self.r = get_tensor(r, base_tensor=self._base_tensor)
        scale = torch.max(self.dims) * 1.5 / 2

        super().__init__(pos, scale, sdf_func=rounded_sdf(box_sdf),
                         params=[self.r / scale, (self.dims - 2 * self.r) / scale],
                         grad_func=rounded_sdf_grad(box_sdf_grad), vel=vel,
                         mass=mass, restitution=restitution, fric_coeff=fric_coeff, eps=eps, col=col,
                         thickness=thickness, smooth=smooth, is_transparent=is_transparent)


class SDFBrick(SDF3D):
    def __init__(self, pos, dims, r, vel=(0, 0, 0, 0, 0, 0), mass=1, restitution=Defaults3D.RESTITUTION,
                 fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON,
                 col=(255, 255, 255), thickness=1, smooth=False, is_transparent=False):
        self._set_base_tensor(locals().values())
        self.dims = get_tensor(dims, base_tensor=self._base_tensor)
        self.r = get_tensor(r, base_tensor=self._base_tensor)
        scale = torch.max(self.dims) * 1.5 / 2

        super().__init__(pos, scale, sdf_func=brick_sdf,
                         params=[self.dims / scale, r / scale],
                         grad_func=rounded_sdf_grad(box_sdf_grad), vel=vel,
                         mass=mass, restitution=restitution, fric_coeff=fric_coeff, eps=eps, col=col,
                         thickness=thickness, smooth=smooth, is_transparent=is_transparent)


class SDFCylinder(SDF3D):
    def __init__(self, pos, rad, height, vel=(0, 0, 0, 0, 0, 0), mass=1, restitution=Defaults3D.RESTITUTION,
                 fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON, custom_mesh=Defaults3D.CUSTOM_MESH,
                 custom_inertia=Defaults3D.CUSTOM_INERTIA, col=(255, 255, 255), thickness=1, smooth=False,
                 is_transparent=False):
        self._set_base_tensor(locals().values())
        self.height = get_tensor(height, base_tensor=self._base_tensor)
        self.rad = get_tensor(rad, base_tensor=self._base_tensor)

        scale = max(self.rad, self.height / 2) * 1.5

        if custom_inertia:
            self._get_ang_inertia = self._custom_get_ang_inertia
        if custom_mesh:
            self._create_mesh = self._custom_create_mesh

        super().__init__(pos, scale, sdf_func=cylinder_sdf, params=[self.rad / scale, self.height / scale],
                         grad_func=cylinder_sdf_grad, vel=vel, mass=mass, restitution=restitution,
                         fric_coeff=fric_coeff, eps=eps, col=col, thickness=thickness, smooth=smooth,
                         is_transparent=is_transparent)

    def _custom_get_ang_inertia(self, mass):
        return mass * torch.diag(torch.cat([((3 * self.rad ** 2 + self.height ** 2) / 12).repeat(2),
                                            (self.rad ** 2 / 2).unsqueeze(0)]))

    def _custom_create_mesh(self, numsegs=32, max_tri_length=0.1):
        thetas = torch.linspace(0, 2 * math.pi * (numsegs - 1) / numsegs, int(numsegs), dtype=self._base_tensor.dtype,
                                device=self._base_tensor.device)

        half_height = self.height / 2
        num_v_verts = (self.height / max_tri_length).ceil().int() + 1

        vert_heights = torch.linspace(-half_height.item(), half_height.item(), num_v_verts,
                                      dtype=self._base_tensor.dtype, device=self._base_tensor.device)
        vert_heights[0] = -half_height
        vert_heights[-1] = half_height

        theta_grid, height_grid = torch.meshgrid(thetas, vert_heights)

        verts = torch.cat([polar_to_cart(self.rad, theta_grid), height_grid.unsqueeze(0)]).reshape(3, -1).t()

        verts = torch.cat([verts, verts.new_zeros(2, verts.shape[1])])
        verts[-2, -1] = half_height
        verts[-1, -1] = -half_height

        inds = torch.arange(0, theta_grid.nelement(), device=self._base_tensor.device).reshape(theta_grid.shape)
        # Repeat first column to wrap full circle
        inds = torch.cat([inds, inds[0, :].unsqueeze(0)], dim=0)

        faces = torch.cat([torch.stack([inds[:-1, :-1], inds[1:, 1:], inds[:-1, 1:]], dim=2).reshape(-1, 3),
                           torch.stack([inds[:-1, :-1], inds[1:, :-1], inds[1:, 1:]], dim=2).reshape(-1, 3)])
        faces_top = torch.stack(
            [torch.tensor(verts.shape[0] - 2, device=self._base_tensor.device).repeat(inds.shape[0] - 1),
             inds[:-1, -1], inds[1:, -1]], dim=1)
        faces_bottom = torch.stack(
            [torch.tensor(verts.shape[0] - 1, device=self._base_tensor.device).repeat(inds.shape[0] - 1),
             inds[1:, 0], inds[:-1, 0]], dim=1)

        faces = torch.cat([faces, faces_top, faces_bottom])

        return verts, faces


class SDFSphere(SDF3D):
    def __init__(self, pos, rad, vel=(0, 0, 0, 0, 0, 0), mass=1, restitution=Defaults3D.RESTITUTION,
                 fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON, custom_mesh=Defaults3D.CUSTOM_MESH,
                 custom_inertia=Defaults3D.CUSTOM_INERTIA, col=(255, 255, 255), thickness=1, smooth=True,
                 is_transparent=False):
        self._set_base_tensor(locals().values())
        self.rad = get_tensor(rad, base_tensor=self._base_tensor)
        scale = self.rad * 1.5

        if custom_inertia:
            self._get_ang_inertia = self._custom_get_ang_inertia
        if custom_mesh:
            self._create_mesh = self._custom_create_mesh

        super().__init__(pos, scale, sdf_func=sphere_sdf, params=[self.rad / scale], grad_func=sphere_sdf_grad, vel=vel,
                         mass=mass, restitution=restitution, fric_coeff=fric_coeff, eps=eps, col=col,
                         thickness=thickness, smooth=smooth, is_transparent=is_transparent)

    def _custom_get_ang_inertia(self, mass):
        return 2 / 5 * mass * self.rad ** 2 * torch.eye(DIM, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)

    def _custom_create_mesh(self, max_tri_length=0.1):
        # thetas = torch.linspace(-math.pi / 2, math.pi / 2, int(numsegs / 2), dtype=Defaults3D.DTYPE,
        #                         device=Defaults3D.DEVICE)
        # phis = torch.linspace(0, 2 * math.pi * (numsegs - 1) / numsegs, int(numsegs), dtype=Defaults3D.DTYPE,
        #                       device=Defaults3D.DEVICE)
        #
        # thetas, phis = torch.meshgrid(thetas, phis)
        #
        # verts = spherical_to_cart(self.rad, thetas, phis).reshape(3, -1).t()
        #
        # inds = torch.tensor(range(numsegs * int(numsegs / 2)), device=Defaults3D.DEVICE).reshape(int(numsegs / 2),
        #                                                                                          numsegs)
        #
        # # Filter duplicate verts
        # verts = verts[inds[0, -1]:inds[-1, 1]]
        # inds[0] = 0
        # inds[1:] -= inds.shape[1] - 1
        # inds[-1] = inds[-1, 0]
        # # Repeat first column to wrap full circle
        # inds = torch.cat([inds, inds[:, 0].unsqueeze(1)], dim=1)
        #
        # faces = torch.cat([torch.stack([inds[:-1, :-1], inds[:-1, 1:], inds[1:, 1:]], dim=2).reshape(-1, 3),
        #                    torch.stack([inds[:-1, :-1], inds[1:, 1:], inds[1:, :-1]], dim=2).reshape(-1, 3)])

        # tri_length ~ 1 / 2^subdivisions
        # => 2^subdivisions ~ 1 / tri_length
        # => subdivisions ~ log2(1 / tri_length)
        # subdivisions = math.ceil(math.log2(self.rad / max_tri_length))
        subdivisions = 4
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions)

        verts = get_tensor(mesh.vertices, base_tensor=self._base_tensor)
        faces = torch.tensor(mesh.faces, device=self._base_tensor.device)

        verts = verts * self.rad

        return verts, faces


class SDFBowl(SDF3D):
    def __init__(self, pos, r, d, vel=(0, 0, 0), mass=1, restitution=Defaults3D.RESTITUTION,
                 fric_coeff=Defaults3D.FRIC_COEFF, eps=Defaults3D.EPSILON, custom_mesh=Defaults3D.CUSTOM_MESH,
                 col=(255, 255, 255), thickness=1, smooth=False, is_transparent=False):
        self._set_base_tensor(locals().values())
        self.r = get_tensor(r, base_tensor=self._base_tensor)
        self.d = get_tensor(d, base_tensor=self._base_tensor)
        scale = (self.r + self.d) * 1.3333

        if custom_mesh:
            self._create_mesh = self._custom_create_mesh

        super().__init__(pos, scale, sdf_func=bowl_sdf, params=[self.r / scale, self.d / scale],
                         grad_func=bowl_sdf_grad, vel=vel, mass=mass, restitution=restitution, fric_coeff=fric_coeff,
                         eps=eps, col=col, thickness=thickness, smooth=smooth, is_transparent=is_transparent)

    def _custom_create_mesh(self, numsegs=32):
        thetas = torch.linspace(0, -math.pi / 2, int(numsegs / 4), dtype=Defaults3D.DTYPE,
                                device=Defaults3D.DEVICE)
        phis = torch.linspace(0, 2 * math.pi * (numsegs - 1) / numsegs, int(numsegs), dtype=Defaults3D.DTYPE,
                              device=Defaults3D.DEVICE)

        thetas, phis = torch.meshgrid(thetas, phis)

        verts0 = spherical_to_cart(self.r - self.d, thetas, phis).reshape(3, -1).t()
        verts1 = spherical_to_cart(self.r + self.d, thetas, phis).reshape(3, -1).t()

        inds = torch.tensor(range(numsegs * int(numsegs / 4)), device=Defaults3D.DEVICE).reshape(int(numsegs / 4),
                                                                                                 numsegs)

        # Filter duplicate verts
        verts0 = verts0[:inds[-1, 1]]
        verts1 = verts1[:inds[-1, 1]]
        inds[-1] = inds[-1, 0]
        # Repeat first column to wrap full circle
        inds = torch.cat([inds, inds[:, 0].unsqueeze(1)], dim=1)

        faces_bowl = torch.cat([torch.stack([inds[1:, 1:], inds[:-1, 1:], inds[:-1, :-1]], dim=2).reshape(-1, 3),
                                torch.stack([inds[1:-1, :-1], inds[1:-1, 1:], inds[:-2, :-1]], dim=2).reshape(-1, 3)])
        faces_top = torch.cat([
            torch.stack([inds[0, :-1] + verts0.shape[0], inds[0, 1:] + verts0.shape[0], inds[0, :-1]], dim=1),
            torch.stack([inds[0, 1:] + verts0.shape[0], inds[0, 1:], inds[0, :-1]], dim=1)], dim=0)

        verts = torch.cat([verts0, verts1])
        faces = torch.cat([faces_bowl.flip(1), faces_bowl + verts0.shape[0], faces_top])

        verts[:, 2] = verts[:, 2] + self.r / 2

        return verts, faces
