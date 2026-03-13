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
from diffsdfsim.lcp_physics.physics import TotalConstraint, RotConstraint
from pytorch3d.transforms import quaternion_invert, quaternion_multiply, quaternion_apply
from torch.nn.functional import normalize

from diffsdfsim.sdf_physics.physics3d.utils import (cart_to_spherical, spherical_to_cart, Indices, Defaults3D, skew_symmetric_mat,
                                         orthogonal, get_tensor)

X = Indices.X
Y = Indices.Y
Z = Indices.Z
DIM = Defaults3D.DIM


class ZConstraint:
    """Prevents motion in the Z axis.
    """
    def __init__(self, body1):
        self.static = True
        self.num_constraints = 1
        self.body1 = body1
        self.pos = body1.pos
        self.rot1 = self.body1.p[:4]
        self.body2 = self.rot2 = None

    def J(self):
        J = self.pos.new_tensor([0, 0, 0, 0, 0, 1]).unsqueeze(0)
        return J, None

    def move(self, dt):
        self.update_pos()

    def update_pos(self):
        self.pos = self.body1.pos
        self.rot1 = self.body1.p[:4]

    def draw(self, screen, pixels_per_meter=1):
        pass


class YConstraint:
    """Prevents motion in the Y axis.
    """
    def __init__(self, body1):
        self.static = True
        self.num_constraints = 1
        self.body1 = body1
        self.pos = body1.pos
        self.rot1 = self.body1.p[:4]
        self.body2 = self.rot2 = None

    def J(self):
        J = self.pos.new_tensor([0, 0, 0, 0, 1, 0]).unsqueeze(0)
        return J, None

    def move(self, dt):
        self.update_pos()

    def update_pos(self):
        self.pos = self.body1.pos
        self.rot1 = self.body1.p[:4]

    def draw(self, screen, pixels_per_meter=1):
        pass


class XConstraint:
    """Prevents motion in the X axis.
    """
    def __init__(self, body1):
        self.static = True
        self.num_constraints = 1
        self.body1 = body1
        self.pos = body1.pos
        self.rot1 = self.body1.p[:4]
        self.body2 = self.rot2 = None

    def J(self):
        J = self.pos.new_tensor([0, 0, 0, 1, 0, 0]).unsqueeze(0)
        return J, None

    def move(self, dt):
        self.update_pos()

    def update_pos(self):
        self.pos = self.body1.pos
        self.rot1 = self.body1.p[:4]

    def draw(self, screen, pixels_per_meter=1):
        pass


class RotConstraint3D(RotConstraint):
    def __init__(self, body1):
        super().__init__(body1)

        self.num_constraints = 3
        self.rot1 = self.body1.p[:4]

    def J(self):
        J_rot = torch.eye(3).type_as(self.pos)
        J_trans = self.pos.new_zeros([3, 3])
        J = torch.cat([J_rot, J_trans], dim=1)
        return J, None

    def move(self, dt):
        self.update_pos()

    def update_pos(self):
        self.pos = self.body1.pos
        self.rot1 = self.body1.p[:4]


class TotalConstraint3D(TotalConstraint):
    def __init__(self, body1):
        super().__init__(body1)
        self.num_constraints = 6
        self.r1, theta, phi = cart_to_spherical(self.pos1)
        self.rot1 = torch.stack([theta, phi])
        self.eye = torch.eye(self.num_constraints).type_as(self.pos)

    def move(self, dt):
        self.rot1 = self.rot1 + self.body1.v[:2] * dt
        self.update_pos()

    def update_pos(self):
        self.pos1 = spherical_to_cart(self.r1, *self.rot1)
        self.pos = self.body1.pos + self.pos1


class GripperJoint:
    """Gripper Joint, only allows linear motion between the bodies directly towards/away from each other"""
    def __init__(self, body1, body2, axis=[1, 0, 0]):
        self.static = False
        self.num_constraints = 5
        self.body1 = body1
        self.body2 = body2
        self.pos = body1.pos
        self.pos1 = self.pos - body1.pos
        self.rot1 = body1.rot
        self.rot2 = None
        self.pos2 = self.pos - self.body2.pos
        self.rot2 = quaternion_multiply(self.body2.rot, quaternion_invert(self.body1.rot))
        self.axis = get_tensor(axis, base_tensor=self.pos)

    def J(self):
        J1 = self.pos.new_zeros(self.num_constraints, 6)
        J2 = self.pos.new_zeros(self.num_constraints, 6)

        # Rotational velocities should be the same
        J1[:3, :3] = torch.eye(3).type_as(self.pos)
        J2[:3, :3] = -torch.eye(3).type_as(self.pos)

        # Directions in which we want to prevent movement (orthogonals to axis)
        ax = quaternion_apply(self.rot1, self.axis)
        dir1 = orthogonal(ax)
        dir2 = torch.cross(dir1, ax)

        dirs = normalize(torch.stack([dir1, dir2]), dim=1)

        # These matrices will make put equality constraints on the velocity components along these directions
        # (direct projection for linear part, projection of cross-product for angular part).
        J1[3:] = dirs @ torch.cat([-skew_symmetric_mat(self.pos1), torch.eye(DIM).type_as(self.pos)], dim=1)
        J2[3:] = dirs @ torch.cat([skew_symmetric_mat(self.pos2), -torch.eye(DIM).type_as(self.pos)], dim=1)

        return J1, J2

    def move(self, dt):
        self.update_pos()

    def update_pos(self):
        self.pos = self.body1.pos
        self.pos1 = self.pos - self.body1.pos
        self.rot1 = self.body1.rot
        if self.body2 is not None:
            # keep position on body1 as reference
            self.pos2 = self.pos - self.body2.pos
            self.rot2 = quaternion_multiply(self.body2.rot, quaternion_invert(self.body1.rot))
