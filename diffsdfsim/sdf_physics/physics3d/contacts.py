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
import torch
from diffsdfsim.lcp_physics.physics.contacts import ContactHandler, OdeContactHandler
from pytorch3d.transforms import quaternion_apply, quaternion_invert, quaternion_multiply
from scipy.spatial.qhull import ConvexHull, QhullError

from .bodies import SDF3D, Body3D
from .utils import Defaults3D


def _overlap(b1, b2):
    v1 = b1.get_surface()[0]
    v2 = b2.get_surface()[0]

    v1_b2 = quaternion_apply(quaternion_invert(b2.rot), v1 - b2.pos)
    v2_b1 = quaternion_apply(quaternion_invert(b1.rot), v2 - b1.pos)

    ov_v1_b2 = torch.any(torch.all((-b2.scale <= v1_b2) & (v1_b2 <= b2.scale), dim=1))
    ov_v2_b1 = torch.any(torch.all((-b1.scale <= v2_b1) & (v2_b1 <= b1.scale), dim=1))
    return ov_v1_b2 and ov_v2_b1


def _frank_wolfe(b1, b2, eps=Defaults3D.EPSILON, tol=Defaults3D.TOL):
    verts, faces = b1.get_surface()
    # Convert vertices to b2's coordinate frame for easier SDF querying
    verts = quaternion_apply(quaternion_invert(b2.rot), verts - b2.pos)

    x = verts[faces].mean(dim=1).squeeze()

    centr_sdfs, centr_grads = b2.query_sdfs(x)
    rads = x.new_zeros(x.shape[0])
    for i in range(3):
        new_rads = (x - verts[faces[:, i]]).norm(dim=1)
        rads[new_rads > rads] = new_rads[new_rads > rads]

    cand_mask = (centr_sdfs < rads + eps) & (centr_grads.norm(dim=1) > 1e-12)
    pqr = verts[faces[cand_mask]]
    if not torch.any(cand_mask):
        return pqr.new_empty((0, 3)), torch.nonzero(cand_mask, as_tuple=False).squeeze(1)

    sdfs = b2.query_sdfs(pqr.reshape(-1, 3), return_grads=False).reshape(pqr.shape[0], -1)
    abc = sdfs.new_zeros(sdfs.shape)
    inds = sdfs.argmin(dim=1)
    x = pqr[torch.arange(pqr.shape[0]), inds, :]
    abc[torch.arange(abc.shape[0]), inds] = 1.

    for iter in range(32):
        sdfs, grads = b2.query_sdfs(x)

        dpqr = pqr @ grads.unsqueeze(2)

        ind = dpqr.argmin(dim=1)
        s = pqr[torch.arange(pqr.shape[0]), ind.squeeze()]

        gamma = 2.0 / (iter + 2.0)

        impr = ((x - s).unsqueeze(1) @ grads.unsqueeze(2)).squeeze(2)
        gamma = gamma * (impr.abs() > tol)
        if torch.all(gamma == 0) or torch.any(sdfs < -tol):
            # Stop if improvement for all points too small or
            # if we found a point that will cause step rejection
            break

        x = (1.0 - gamma) * x + gamma * s
        abc *= (1.0 - gamma)
        abc[torch.arange(abc.shape[0]), ind.squeeze()] += gamma.squeeze()

    if isinstance(b1, SDF3D):
        # Push x to actual surface from triangle
        x_b1 = (b1.verts[b1.faces[cand_mask]] * abc.unsqueeze(2)).sum(dim=1)
        sdfs1, grads1 = b1.query_sdfs(x_b1)
        x = x - sdfs1.unsqueeze(1) * quaternion_apply(quaternion_multiply(quaternion_invert(b2.rot), b1.rot), grads1)
    sdfs = b2.query_sdfs(x, return_grads=False)

    contact_mask = (sdfs <= eps)  # & (sdfs - sdfs.min() < 1e-5)
    cand_mask[cand_mask.clone()] &= contact_mask

    return abc[contact_mask], torch.nonzero(cand_mask, as_tuple=False).squeeze(1)


def _filter_contacts(normals, p1, eps=Defaults3D.EPSILON):
    contact_inds = torch.arange(normals.shape[0], device=normals.device)
    if normals.shape[0] <= 1:
        return contact_inds

    # At SDF singularities, normals might be zero
    valid_mask = normals.norm(dim=1) > 1e-12
    normals = normals[valid_mask]
    p1 = p1[valid_mask]
    contact_inds = contact_inds[valid_mask]

    clusters = []
    while normals.shape[0] > 0:
        n = normals[0]

        angles = torch.acos(torch.min(normals @ n, n.new_tensor(1.)))
        cluster_mask = (angles < 1e-2)

        c_p1s = p1[cluster_mask]
        c_inds = contact_inds[cluster_mask]

        clusters.append((c_p1s, c_inds))

        normals = normals[~cluster_mask]
        p1 = p1[~cluster_mask]
        contact_inds = contact_inds[~cluster_mask]

    clusters_filtered = []
    # Filter further points based by generating a (possibly lower-dimensional) convex hull
    for p1, cluster_inds in clusters:
        completed = False
        ps = p1.detach()
        while not completed:
            if ps.shape[1] > 1:
                try:
                    hull = ConvexHull(ps.cpu())
                    inds = torch.tensor(hull.vertices).long().to(normals.device)
                    completed = True
                except QhullError:
                    # Qhull didn't work, likely because the points only span a lower-dim space
                    # => remove dim with smallest variance
                    var = ps.var(dim=0)
                    mask = torch.ones(ps.shape[1]).bool()
                    mask[var.argmin()] = False
                    ps = ps[:, mask]
            else:
                # If we reduced the points to 1D convex hull is just min and max
                ps_min, ps_argmin = ps.min(0)
                ps_max, ps_argmax = ps.max(0)
                if ps_max - ps_min > eps:
                    inds = torch.stack([ps.argmin(), ps.argmax()])
                else:
                    inds = torch.stack([ps.argmin()])
                completed = True

        clusters_filtered.append(cluster_inds[inds])

    filtered_inds = torch.empty((0), dtype=torch.int64, device=normals.device)
    for cluster_inds in clusters_filtered:
        filtered_inds = torch.cat([filtered_inds, cluster_inds], dim=0)

    return filtered_inds


def _compute_contacts(b1, b2, abc, contact_inds, eps=Defaults3D.EPSILON, detach_contact_b2=True):
    if contact_inds.nelement() > 0:
        # verts, faces = b1.get_surface()
        verts, faces = b1.verts, b1.faces

        cp_b1 = (verts[faces[contact_inds]] * abc.unsqueeze(2)).sum(dim=1)
        if isinstance(b1, SDF3D):
            # Contact points on triangle might not be on surface, correct for this.
            dists1, normals1 = b1.query_sdfs(cp_b1)
            cp_b1 = cp_b1 - dists1.unsqueeze(1) * normals1
            dists1, normals1 = b1.query_sdfs(cp_b1)

        contact_points = quaternion_apply(b1.rot, cp_b1) + b1.pos

        if detach_contact_b2:
            # TODO: This detach() stop gradients needed e.g. for block tower example (when contact points need to be
            #       moved consistently for optimization).
            cp_b2 = quaternion_apply(quaternion_invert(b2.rot), contact_points - b2.pos).detach()
        else:
            cp_b2 = quaternion_apply(quaternion_invert(b2.rot), contact_points - b2.pos)

        dists2, normals2 = b2.query_sdfs(cp_b2)

        if isinstance(b1, SDF3D):
            laplacian1 = cp_b1.new_zeros(cp_b1.shape[0])
            for i in range(3):
                shift = cp_b1.new_zeros(3)
                shift[i] = eps
                laplacian1 += b1.query_sdfs(cp_b1 + shift, return_grads=False) - 2 * dists1 \
                              + b1.query_sdfs(cp_b1 - shift, return_grads=False)

            laplacian2 = cp_b2.new_zeros(cp_b2.shape[0])
            for i in range(3):
                shift = cp_b2.new_zeros(3)
                shift[i] = eps
                laplacian2 += b2.query_sdfs(cp_b2 + shift, return_grads=False) - 2 * dists2 \
                              + b2.query_sdfs(cp_b2 - shift, return_grads=False)

            stable_mask = laplacian2.abs() < laplacian1.abs()

            normals = quaternion_apply(b2.rot, normals2) * stable_mask.unsqueeze(1) \
                      - quaternion_apply(b1.rot, normals1) * (~stable_mask).unsqueeze(1)
        else:
            normals = quaternion_apply(b2.rot, normals2)
        dists = dists2

        # p2 = contact_points - b2.pos - dists.unsqueeze(1) / 2. * normals
        p2 = quaternion_apply(b2.rot, cp_b2 - dists2.unsqueeze(1) * normals2)
        p1 = quaternion_apply(b1.rot, cp_b1)
        pen = -dists

        return normals, p1, p2, pen

    return abc.new_tensor([]), abc.new_tensor([]), abc.new_tensor([]), abc.new_tensor([])


class FWContactHandler(ContactHandler):
    def __init__(self):
        self.debug_callback = OdeContactHandler()

    def __call__(self, args, geom1, geom2):
        # self.debug_callback(args, geom1, geom2)

        if geom1 in geom2.no_contact:
            return
        world = args[0]

        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]

        assert (isinstance(b1, SDF3D) and isinstance(b2, Body3D)) or (isinstance(b2, SDF3D) and isinstance(b1, Body3D))

        if isinstance(b1, SDF3D) and isinstance(b2, SDF3D):
            if not _overlap(b1, b2):
                return

            # TODO: compare to using only one direction
            validstep = self._search_contacts(geom1, geom2, world)
            if validstep:
                self._search_contacts(geom2, geom1, world)
        elif isinstance(b1, SDF3D):
            self._search_contacts(geom2, geom1, world)
        else:
            self._search_contacts(geom1, geom2, world)

        # world.contacts_debug = world.contacts  # XXX

    @staticmethod
    def _search_contacts(geom1, geom2, world):
        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]

        assert (isinstance(b1, Body3D))
        assert (isinstance(b2, SDF3D))
        with torch.no_grad():
            abc, contact_inds = _frank_wolfe(b1, b2, world.eps, world.tol)
            normals, p1, p2, pens = _compute_contacts(b1, b2, abc, contact_inds,
                                                      detach_contact_b2=world.detach_contact_b2)
            if torch.all(pens <= world.tol):
                filtered_inds = _filter_contacts(normals, p1, eps=world.eps)

        if torch.all(pens <= world.tol):
            normals, p1, p2, pens = _compute_contacts(b1, b2, abc[filtered_inds], contact_inds[filtered_inds],
                                                      detach_contact_b2=world.detach_contact_b2)

        pts = []
        for normal, pt1, pt2, pen in zip(normals, p1, p2, pens):
            pts.append((normal, pt1, pt2, pen))
        for p in pts:
            world.contacts.append((p, geom1.body, geom2.body))

        return torch.all(pens <= world.tol)

