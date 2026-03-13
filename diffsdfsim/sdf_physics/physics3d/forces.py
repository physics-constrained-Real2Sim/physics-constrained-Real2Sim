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
from diffsdfsim.lcp_physics.physics import ExternalForce

from .utils import get_tensor


def down_force(t):
    return ExternalForce3D.DOWN


def vert_impulse(t):
    if t < 0.1:
        return ExternalForce3D.DOWN
    else:
        return ExternalForce3D.ZEROS


def hor_impulse(t):
    if t < 0.1:
        return ExternalForce3D.RIGHT
    else:
        return ExternalForce3D.ZEROS


def rot_impulse(t):
    if t < 0.1:
        return ExternalForce3D.ROT
    else:
        return ExternalForce3D.ZEROS


class ExternalForce3D(ExternalForce):
    """Generic external force to be added to objects.
       Takes in a force_function which returns a force vector as a function of time,
       and a multiplier that multiplies such vector.
    """
    # Pre-store basic forces
    UP = get_tensor([0, 0, 0, 0, 1, 0])
    DOWN = get_tensor([0, 0, 0, 0, -1, 0])
    RIGHT = get_tensor([0, 0, 0, 1, 0, 0])
    LEFT = get_tensor([0, 0, 0, -1, 0, 0])
    FRONT = get_tensor([0, 0, 0, 0, 0,  1])
    BACK = get_tensor([0, 0, 0, 0, 0, -1])
    ROTX = get_tensor([1, 0, 0, 0, 0, 0])
    ROTY = get_tensor([0, 1, 0, 0, 0, 0])
    ROTZ = get_tensor([0, 0, 1, 0, 0, 0])
    ZEROS = get_tensor([0, 0, 0, 0, 0, 0])

    def __init__(self, force_func=down_force, multiplier=1.):
        super().__init__(force_func=force_func, multiplier=multiplier)


class Gravity3D(ExternalForce3D):
    """Gravity force object, constantly returns a downwards pointing force of
       magnitude body.mass * g.
    """

    def __init__(self, g=10.0):
        self.multiplier = g
        self.body = None
        self.cached_force = None

    def force(self, t):
        return self.cached_force

    def set_body(self, body):
        super().set_body(body)
        down_tensor = ExternalForce3D.DOWN.type_as(body._base_tensor).to(body._base_tensor)
        self.cached_force = down_tensor * self.body.mass * self.multiplier
