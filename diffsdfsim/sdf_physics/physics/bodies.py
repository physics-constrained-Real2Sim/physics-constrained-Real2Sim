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

import ode
import pygame
import torch

from lcp_physics.physics.bodies import Body
from lcp_physics.physics.utils import Defaults, get_tensor, rotation_matrix, cross_2d, polar_to_cart


class SDF(Body):
    def __init__(self, pos, scale, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff=Defaults.FRIC_COEFF, eps=Defaults.EPSILON,
                 col=(255, 0, 0), thickness=1):
        self._set_base_tensor(locals().values())
        self.scale = get_tensor(scale, base_tensor=self._base_tensor)
        pos = get_tensor(pos, base_tensor=self._base_tensor)
        super().__init__(pos=pos, vel=vel, mass=mass, restitution=restitution, fric_coeff=fric_coeff, eps=eps,
                         col=col, thickness=thickness)
        if pos.size(0) == 3:
            self.set_p(pos)

    def _create_geom(self):
        # Use bounding box for broadphase
        self.geom = ode.GeomBox(None, torch.cat([self.scale.expand(2) + 2 * self.eps.item(),
                                                 self.scale.expand(2).new_ones(1)]))
        self.geom.setPosition(torch.cat([self.pos, self.pos.new_zeros(1)]))
        self.geom.no_contact = set()

    def _get_ang_inertia(self, mass):
        # TODO: probably only valid for convex shapes...
        numerator = 0
        denominator = 0
        verts, edges = self.get_surface()
        verts = (verts - self.pos) @ rotation_matrix(self.rot)
        for edge in edges:
            v1 = verts[edge[0]]
            v2 = verts[edge[1]]
            norm_cross = torch.norm(cross_2d(v2, v1))
            numerator = numerator + norm_cross * \
                        (torch.dot(v1, v1) + torch.dot(v1, v2) + torch.dot(v2, v2))
            denominator = denominator + norm_cross
        return 1 / 6 * mass * numerator / denominator

    def draw(self, screen, pixels_per_meter=1):
        # TODO: draw bounding box for now, better extract surface
        half_dims = self.scale.expand(2) / 2
        v0, v1 = half_dims, half_dims * half_dims.new_tensor([-1, 1])
        verts = [v0, v1, -v0, -v1]
        rot_mat = rotation_matrix(self.rot)
        verts = [(self.pos + rot_mat @ v).detach().cpu().numpy() * pixels_per_meter for v in verts]

        # draw hull
        p = pygame.draw.polygon(screen, (0, 255, 0), verts, self.thickness)

        return [p]

    def get_surface(self):
        """
        Get a mesh describing the object surface.
        :return: vertices and faces describing the mesh
        """
        raise NotImplementedError

    def query_sdfs(self, pts):
        """
        Query SDF values at the query points in world coordinates.
        :param pts: query points
        :return: sdf values in the current volume at the query points (scaled to world coordinates)
        """
        raise NotImplementedError


class SDFGrid(SDF):
    def __init__(self, pos, scale, sdf, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff=Defaults.FRIC_COEFF, eps=Defaults.EPSILON,
                 col=(255, 0, 0), thickness=1):
        self._set_base_tensor(locals().values())
        self.sdf = get_tensor(sdf, base_tensor=self._base_tensor)
        # Forward differences
        # self.sdf_grads = torch.stack([torch.cat([self.sdf[1:, :] - self.sdf[:-1, :],
        #                                          self.sdf.new_zeros((1, self.sdf.shape[1]))], dim=0),
        #                               torch.cat([self.sdf[:, 1:] - self.sdf[:, :-1],
        #                                          self.sdf.new_zeros((self.sdf.shape[0], 1))], dim=1)], dim=2)

        # Central differences
        self.sdf_grads = torch.stack([torch.cat([self.sdf.new_zeros((1, self.sdf.shape[1])),
                                                 (self.sdf[2:, :] - self.sdf[:-2, :]) / 2,
                                                 self.sdf.new_zeros((1, self.sdf.shape[1]))], dim=0),
                                      torch.cat([self.sdf.new_zeros((self.sdf.shape[0], 1)),
                                                 (self.sdf[:, 2:] - self.sdf[:, :-2]) / 2,
                                                 self.sdf.new_zeros((self.sdf.shape[0], 1))], dim=1)], dim=2)
        self.scale = get_tensor(scale, base_tensor=self._base_tensor)

        self.verts, self.edges = self.marching_squares()

        super().__init__(pos, scale, vel=vel, mass=mass, restitution=restitution,
                         fric_coeff=fric_coeff, eps=eps, col=col, thickness=thickness)

    def marching_squares(self):
        edgeTable = torch.tensor([[-1, -1, -1, -1],
                                  [2, 3, -1, -1],
                                  [1, 2, -1, -1],
                                  [1, 3, -1, -1],
                                  [0, 1, -1, -1],
                                  [0, 3, 1, 2],
                                  [0, 2, -1, -1],
                                  [0, 3, -1, -1],
                                  [0, 3, -1, -1],
                                  [0, 2, -1, -1],
                                  [0, 1, 2, 3],
                                  [0, 1, -1, -1],
                                  [1, 3, -1, -1],
                                  [1, 2, -1, -1],
                                  [2, 3, -1, -1],
                                  [-1, -1, -1, -1]], dtype=torch.int64, device=Defaults.DEVICE)

        inner = self.sdf < 0

        tl = inner[:-1, :-1]
        tr = inner[:-1, 1:]
        bl = inner[1:, :-1]
        br = inner[1:, 1:]

        cubeclasses = torch.sum(torch.stack([tl.flatten(), tr.flatten(), br.flatten(), bl.flatten()],
                                            dim=1).long() * torch.tensor([8, 4, 2, 1], device=Defaults.DEVICE), dim=1)

        vertIdxBuffer = ((cubeclasses != 0) & (cubeclasses != 15)).long() * 2 \
                        + ((cubeclasses == 5) | (cubeclasses == 10)).long() * 2
        num_verts = vertIdxBuffer.sum()
        vertIdxBuffer = vertIdxBuffer.cumsum(0) - vertIdxBuffer
        verts = torch.zeros((num_verts, 2), dtype=self._base_tensor.dtype, device=Defaults.DEVICE)

        edgeIdxBuffer = ((cubeclasses != 0) & (cubeclasses != 15)).long() \
                        + ((cubeclasses == 5) | (cubeclasses == 10)).long()
        num_edges = edgeIdxBuffer.sum()
        edgeIdxBuffer = edgeIdxBuffer.cumsum(0) - edgeIdxBuffer
        edges = torch.zeros((num_edges, 2), dtype=torch.int64, device=Defaults.DEVICE)

        pts_loc = torch.stack(torch.meshgrid(torch.linspace(-0.5, 0.5, self.sdf.shape[0],
                                                            dtype=Defaults.DTYPE, device=Defaults.DEVICE),
                                             torch.linspace(-0.5, 0.5, self.sdf.shape[1],
                                                            dtype=Defaults.DTYPE, device=Defaults.DEVICE)),
                              dim=2)

        ps = torch.stack([pts_loc[:-1, :-1], pts_loc[:-1, 1:], pts_loc[1:, 1:], pts_loc[1:, :-1]])
        ps = ps.reshape(4, -1, 2)
        vals = torch.stack([self.sdf[:-1, :-1], self.sdf[:-1, 1:], self.sdf[1:, 1:], self.sdf[1:, :-1]]).reshape(4, -1)

        offsets = vertIdxBuffer.new_zeros((4, cubeclasses.shape[0]))
        offset = vertIdxBuffer.new_zeros(cubeclasses.shape[0])

        masks = [(cubeclasses >= 4) & (cubeclasses <= 11),
                 ((cubeclasses >= 2) & (cubeclasses <= 5)) | ((cubeclasses >= 10) & (cubeclasses <= 13)),
                 ((cubeclasses % 4) == 1) | ((cubeclasses % 4) == 2),
                 ((cubeclasses < 8) & ((cubeclasses % 2) == 1)) | ((cubeclasses >= 8) & ((cubeclasses % 2) == 0))]

        for i, mask in enumerate(masks):
            verts[(vertIdxBuffer + offset)[mask]] = self.vertex_interp(ps[i], ps[(i + 1) % 4],
                                                                       vals[i], vals[(i + 1) % 4])[mask]
            offsets[i][mask] = offset[mask]
            offset[mask] += 1

        for i in range(2):
            edgemask = edgeTable[cubeclasses, 2 * i] != -1
            edgeidx0 = edgeTable[cubeclasses, 2 * i][edgemask].unsqueeze(0)
            edgeidx1 = edgeTable[cubeclasses, 2 * i + 1][edgemask].unsqueeze(0)
            edges[edgeIdxBuffer[edgemask], 0] = vertIdxBuffer[edgemask] + offsets[:, edgemask].gather(0, edgeidx0)
            edges[edgeIdxBuffer[edgemask], 1] = vertIdxBuffer[edgemask] + offsets[:, edgemask].gather(0, edgeidx1)

        # Make vertices unique and remove degenerate edges
        verts, inds = verts.unique(dim=0, return_inverse=True)
        edges = inds[edges]
        edges = edges[edges[:, 0] != edges[:, 1]]

        verts = verts * self.scale

        return verts, edges

    @staticmethod
    def vertex_interp(p0, p1, val0, val1):
        retval = torch.zeros_like(p0)
        retval[(torch.abs(val0) < 1e-5) | (torch.abs(val0 - val1) < 1e-5)] \
            = p0[(torch.abs(val0) < 1e-5) | (torch.abs(val0 - val1) < 1e-5)]
        retval[torch.abs(val1) < 1e-5] = p1[torch.abs(val1) < 1e-5]

        mu = -val0 / (val1 - val0)
        retval[(torch.abs(val0) >= 1e-5) & (torch.abs(val1) >= 1e-5) & (torch.abs(val0 - val1) >= 1e-5)] \
            = (p0 + mu.unsqueeze(1) * (p1 - p0))[[(torch.abs(val0) >= 1e-5) & (torch.abs(val1) >= 1e-5)
                                                  & (torch.abs(val0 - val1) >= 1e-5)]]

        return retval

    def query_sdfs(self, pts):
        rot_mat = rotation_matrix(self.rot)
        pts_loc = (pts - self.pos) @ rot_mat
        overlap_mask = torch.all(torch.abs(pts_loc) < self.scale / 2, dim=1)

        sdfs = self.sdf.new_ones(pts_loc.shape[0])
        grads_loc = self.sdf.new_zeros(pts.shape)

        if torch.any(overlap_mask):
            inds = (pts_loc[overlap_mask] / self.scale + 0.5) * (get_tensor(self.sdf.shape,
                                                                            base_tensor=self._base_tensor) - 1)

            sdfs[overlap_mask] = self.interp_bilinear(self.sdf, inds)
            grads_loc[overlap_mask] = self.interp_bilinear(self.sdf_grads, inds)

        grad_norm = grads_loc.norm(dim=1)
        grads_norm = grads_loc.clone()
        grads_norm[grad_norm != 0] = grads_loc[grad_norm != 0] / grad_norm[grad_norm != 0].unsqueeze(1)

        sdfs *= self.scale

        grads = grads_norm @ rot_mat.t()

        return sdfs, grads

    @staticmethod
    def interp_bilinear(vol, inds):
        x0 = torch.floor(inds[:, 0]).long()
        x1 = x0 + 1

        y0 = torch.floor(inds[:, 1]).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, vol.shape[0] - 1)
        x1 = torch.clamp(x1, 0, vol.shape[0] - 1)
        y0 = torch.clamp(y0, 0, vol.shape[1] - 1)
        y1 = torch.clamp(y1, 0, vol.shape[1] - 1)

        va = vol[x0, y0]
        vb = vol[x0, y1]
        vc = vol[x1, y0]
        vd = vol[x1, y1]

        wa = (x1.type(inds.type()) - inds[:, 0]) * (y1.type(inds.type()) - inds[:, 1])
        wb = (x1.type(inds.type()) - inds[:, 0]) * (inds[:, 1] - y0.type(inds.type()))
        wc = (inds[:, 0] - x0.type(inds.type())) * (y1.type(inds.type()) - inds[:, 1])
        wd = (inds[:, 0] - x0.type(inds.type())) * (inds[:, 1] - y0.type(inds.type()))

        if len(vol.shape) <= 2:
            return (va * wa) + (vb * wb) + (vc * wc) + (vd * wd)
        else:
            return (va.t() * wa).t() + (vb.t() * wb).t() + (vc.t() * wc).t() + (vd.t() * wd).t()

    def get_surface(self):
        rot_mat = rotation_matrix(self.rot)
        verts_loc, edges = self.verts, self.edges
        # verts_loc, edges = self.marching_squares()
        verts = verts_loc @ rot_mat.t() + self.pos

        return verts, edges

    def draw(self, screen, pixels_per_meter=1):
        p = super().draw(screen, pixels_per_meter=1)
        rot_mat = rotation_matrix(self.rot)
        verts = self.verts @ rot_mat.t() + self.pos

        ls = []
        for edge in self.edges:
            ls.append(pygame.draw.line(screen, self.col, verts[edge[0]].detach().cpu().numpy(),
                                       verts[edge[1]].detach().cpu().numpy()))

        return ls + p


class SDFRect(SDF):
    def __init__(self, pos, dims, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff=Defaults.FRIC_COEFF, eps=Defaults.EPSILON,
                 col=(255, 0, 0), thickness=1):
        self._set_base_tensor(locals().values())
        self.dims = get_tensor(dims, base_tensor=self._base_tensor)
        scale = torch.max(self.dims) * 1.5

        super().__init__(pos, scale, vel=vel, mass=mass, restitution=restitution,
                         fric_coeff=fric_coeff, eps=eps, col=col, thickness=thickness)

    def get_surface(self):
        half_dims = self.dims / 2
        v0, v1 = half_dims, half_dims * half_dims.new_tensor([-1, 1])
        verts_loc = torch.stack([v0, v1, -v0, -v1])
        edges = torch.tensor([[v, (v + 1) % len(verts_loc)] for v in range(len(verts_loc))], device=Defaults.DEVICE)

        rot_mat = rotation_matrix(self.rot)
        verts = verts_loc @ rot_mat.t() + self.pos

        return verts, edges

    def query_sdfs(self, pts):
        rot_mat = rotation_matrix(self.rot)
        pts_loc = (pts - self.pos) @ rot_mat
        overlap_mask = torch.all(torch.abs(pts_loc) < self.scale / 2, dim=1)

        sdfs = self._base_tensor.new_ones(pts_loc.shape[0]) * self.scale
        grads_loc = self._base_tensor.new_zeros(pts.shape)

        if torch.any(overlap_mask):
            half_dims = self.dims / 2

            dist = torch.abs(pts_loc[overlap_mask]) - half_dims
            outer = torch.any(dist > 0, dim=1)
            signs = torch.sign(pts_loc[overlap_mask])
            signs[signs == 0] = 1

            max_dist, max_dist_ind = torch.max(dist, dim=1)

            q = torch.max(dist, get_tensor(0))

            sdfs[overlap_mask] = q.norm(dim=1) + torch.min(max_dist, get_tensor(0))

            grads_ov = self._base_tensor.new_zeros(dist.shape)
            grads_ov[torch.arange(dist.shape[0]), max_dist_ind] = 1

            grads_ov[outer] = q[outer]
            grads_ov = grads_ov * signs
            grads_loc[overlap_mask] = grads_ov / grads_ov.norm(dim=1).unsqueeze(1)

        grads = grads_loc @ rot_mat.t()

        return sdfs, grads

    def _get_ang_inertia(self, mass):
        return mass * torch.sum(self.dims ** 2) / 12

    def draw(self, screen, pixels_per_meter=1):
        p = super().draw(screen, pixels_per_meter=1)

        half_dims = self.dims / 2
        v0, v1 = half_dims, half_dims * half_dims.new_tensor([-1, 1])
        verts = [v0, v1, -v0, -v1]
        rot_mat = rotation_matrix(self.rot)

        verts = [(rot_mat @ v + self.pos).detach().cpu().numpy() * pixels_per_meter
                 for v in verts]

        l1 = pygame.draw.line(screen, (0, 0, 255), verts[0], verts[2])
        l2 = pygame.draw.line(screen, (0, 0, 255), verts[1], verts[3])

        pRect = pygame.draw.polygon(screen, self.col, verts, self.thickness)

        return [pRect] + p


class SDFCircle(SDF):
    def __init__(self, pos, rad, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff=Defaults.FRIC_COEFF, eps=Defaults.EPSILON,
                 col=(255, 0, 0), thickness=1):
        self._set_base_tensor(locals().values())
        self.rad = get_tensor(rad, base_tensor=self._base_tensor)
        scale = self.rad * 2 * 1.3333

        super().__init__(pos, scale, vel=vel, mass=mass, restitution=restitution,
                         fric_coeff=fric_coeff, eps=eps, col=col, thickness=thickness)

    def get_surface(self, num_verts=64):
        angles = torch.linspace(0, (2 * math.pi / num_verts) * (num_verts - 1), num_verts, dtype=Defaults.DTYPE,
                                device=Defaults.DEVICE)
        verts_loc = polar_to_cart(self.rad, angles).t()
        edges = torch.tensor([[v, (v + 1) % len(verts_loc)] for v in range(len(verts_loc))], device=Defaults.DEVICE)

        rot_mat = rotation_matrix(self.rot)
        verts = verts_loc @ rot_mat.t() + self.pos

        return verts, edges

    def query_sdfs(self, pts):
        rot_mat = rotation_matrix(self.rot)
        pts_loc = (pts - self.pos) @ rot_mat
        overlap_mask = torch.all(torch.abs(pts_loc) < self.scale / 2, dim=1)

        sdfs = self._base_tensor.new_ones(pts_loc.shape[0]) * self.scale
        grads_loc = self._base_tensor.new_zeros(pts.shape)

        if torch.any(overlap_mask):
            pts_norm = pts_loc[overlap_mask].norm(dim=1)

            sdfs[overlap_mask] = pts_norm - self.rad
            grads_loc[overlap_mask] = pts_loc[overlap_mask] / pts_norm.unsqueeze(1)

        grads = grads_loc @ rot_mat.t()

        return sdfs, grads

    def _get_ang_inertia(self, mass):
        return mass * self.rad * self.rad / 2

    def draw(self, screen, pixels_per_meter=1):
        center = (self.pos.detach().cpu().numpy() * pixels_per_meter).astype(int)
        rad = int(self.rad.item() * pixels_per_meter)
        # draw radius to visualize orientation
        r = pygame.draw.line(screen, (0, 0, 255), center,
                             center + [math.cos(self.rot.item()) * rad,
                                       math.sin(self.rot.item()) * rad],
                             self.thickness)
        # draw circle
        c = pygame.draw.circle(screen, self.col, center,
                               rad, self.thickness)

        p = super().draw(screen, pixels_per_meter=pixels_per_meter)

        return [c, r] + p


class SDFBowl(SDF):
    def __init__(self, pos, r, d, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff=Defaults.FRIC_COEFF, eps=Defaults.EPSILON,
                 col=(255, 0, 0), thickness=1):
        self._set_base_tensor(locals().values())
        self.r = get_tensor(r, base_tensor=self._base_tensor)
        self.d = get_tensor(d, base_tensor=self._base_tensor)
        scale = (r + d) * 2 * 1.3333

        super().__init__(pos, scale, vel=vel, mass=mass, restitution=restitution,
                         fric_coeff=fric_coeff, eps=eps, col=col, thickness=thickness)

    def get_surface(self, num_verts=64):
        angles = torch.linspace(-math.pi, 0, int(num_verts / 2), dtype=Defaults.DTYPE,
                                device=Defaults.DEVICE)
        verts0_loc = polar_to_cart(self.r - self.d, angles).t()
        verts1_loc = polar_to_cart(self.r + self.d, angles).t()
        verts_loc = torch.cat([verts0_loc, verts1_loc.flip(0)])
        edges = torch.tensor([[v, (v + 1) % len(verts_loc)] for v in range(len(verts_loc))], device=Defaults.DEVICE)

        verts_loc[:, 1] += self.r / 2

        rot_mat = rotation_matrix(self.rot)
        verts = verts_loc @ rot_mat.t() + self.pos

        return verts, edges

    def query_sdfs(self, pts):
        rot_mat = rotation_matrix(self.rot)
        pts_loc = (pts - self.pos) @ rot_mat

        overlap_mask = torch.all(torch.abs(pts_loc) < self.scale / 2, dim=1)

        sdfs = pts.new_ones(pts.shape[0]) * self.scale
        grads_loc = pts.new_zeros(pts.shape)

        if torch.any(overlap_mask):
            pts_in = pts_loc[overlap_mask]
            pts_in[:, 1] -= self.r / 2

            ps = pts_in.clone()
            ps[:, 0] = ps[:, 0].abs()

            ps_norm = ps.norm(dim=1)

            ps[ps[:, 1] < 0, 0] = ps_norm[ps[:, 1] < 0]

            ps[:, 0] = (ps[:, 0] - self.r).abs() - self.d

            sdfs[overlap_mask] = torch.max(ps, get_tensor(0)).norm(dim=1) + torch.min(get_tensor(0), ps.max(dim=1)[0])

            grads_ov = pts_in * (ps_norm - self.r).sign().unsqueeze(1)
            grads_ov[ps[:, 1] >= 0] = torch.max(ps[ps[:, 1] >= 0], get_tensor(0))
            grads_ov[ps[:, 1] >= 0, 0] = grads_ov[ps[:, 1] >= 0, 0] * pts_in[ps[:, 1] >= 0, 0].sign()\
                                         * ((pts_in[ps[:, 1] >= 0, 0].abs() - self.r).sign())

            grads_loc[overlap_mask] = grads_ov / grads_ov.norm(dim=1).unsqueeze(1)

        grads = grads_loc @ rot_mat.t()

        return sdfs, grads

    def draw(self, screen, pixels_per_meter=1):
        verts = self.get_surface()[0]
        pts = [v.detach().cpu().numpy() * pixels_per_meter for v in verts]

        p = pygame.draw.polygon(screen, self.col, pts, self.thickness)

        r = super().draw(screen, pixels_per_meter)

        return [p] + r
