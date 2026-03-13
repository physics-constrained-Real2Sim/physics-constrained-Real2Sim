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

import torch
from lcp_physics.physics.contacts import ContactHandler, OdeContactHandler
from lcp_physics.physics.utils import rotation_matrix, Defaults
from scipy.spatial.qhull import ConvexHull

from .bodies import SDF

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2


class SDFContactHandler(ContactHandler):
    def __init__(self):
        self.debug_callback = OdeContactHandler()

    def __call__(self, args, geom1, geom2):
        # self.debug_callback(args, geom1, geom2)

        if geom1 in geom2.no_contact:
            return
        world = args[0]

        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]

        assert (isinstance(b1, SDF))
        assert (isinstance(b2, SDF))

        verts1, edges1 = b1.get_surface()
        verts2, edges2 = b2.get_surface()

        sdfs1, grads1, contact_verts1 = self._search_contacts(verts1, edges1, b2)
        sdfs2, grads2, contact_verts2 = self._search_contacts(verts2, edges2, b1)

        # sdfs1, grads1 = b2.query_sdfs(verts1)
        # sdfs2, grads2 = b1.query_sdfs(verts2)
        # contact_verts1 = verts1
        # contact_verts2 = verts2

        contacts1 = sdfs1 <= world.eps
        contacts2 = sdfs2 <= world.eps

        pts = []
        if torch.any(contacts1):
            normals = grads1[contacts1]
            p1 = (contact_verts1[contacts1] - b1.pos) - sdfs1[contacts1].unsqueeze(1) * grads1[contacts1]
            p2 = (contact_verts1[contacts1] - b2.pos) - sdfs1[contacts1].unsqueeze(1) * grads1[contacts1]
            dists = sdfs1[contacts1]

            for normal, pt1, pt2, dist in zip(normals, p1, p2, dists):
                pts.append((normal, pt1, pt2, -dist))

        if torch.any(contacts2):
            normals = -grads2[contacts2]
            p2 = (contact_verts2[contacts2] - b2.pos) - sdfs2[contacts2].unsqueeze(1) * grads2[contacts2]
            p1 = (contact_verts2[contacts2] - b1.pos) - sdfs2[contacts2].unsqueeze(1) * grads2[contacts2]
            dists = sdfs2[contacts2]

            for normal, pt1, pt2, dist in zip(normals, p1, p2, dists):
                pts.append((normal, pt1, pt2, -dist))

        # # Filter out points that are less important for contact based on penetration distance
        # if len(pts) > 1:
        #     dists = torch.stack([p[3] for p in pts])
        #     mask = dists.max() - dists < 1e-5
        #
        #     pts = [pts[i] for i in mask.nonzero(as_tuple=False)]

        # Filter further points based by generating a (possibly lower-dimensional) convex hull
        if len(pts) > 1:
            ps = torch.stack([p[1] for p in pts])

            u, s, v = torch.pca_lowrank(ps)

            dim = (s > 1e-5).sum()

            ps_trans = ps @ v[:, :dim]

            if dim > 1:
                hull = ConvexHull(ps_trans.cpu())
                inds = hull.vertices
            elif dim == 1:
                inds = [ps_trans.argmin(), ps_trans.argmax()]
            else:
                inds = [0]

            pts = [pts[i] for i in inds]

        for p in pts:
            world.contacts.append((p, geom1.body, geom2.body))
        world.contacts_debug = world.contacts # XXX

    # Frank-Wolfe algorithm
    def _search_contacts(self, verts, edges, body):
        ab = verts[edges]

        x = ab.mean(dim=1).squeeze()

        start_sdfs, start_grads = body.query_sdfs(x)
        rads = (x.unsqueeze(1) - ab).norm(dim=2).max(dim=1)[0]

        x = x[(start_sdfs < rads) & ~torch.all(start_grads == 0, dim=1)]
        ab = ab[(start_sdfs < rads) & ~torch.all(start_grads == 0, dim=1)]
        if x.nelement() == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        for iter in range(32):
            _, grads = body.query_sdfs(x)

            dab = ab @ grads.unsqueeze(2)

            ind = dab.argmin(dim=1)
            s = ab[torch.arange(ab.shape[0]), ind.squeeze()]

            gamma = 2.0 / (iter + 2.0)

            x = (1.0 - gamma) * x + gamma * s

        sdfs, grads = body.query_sdfs(x)

        return sdfs, grads, x
