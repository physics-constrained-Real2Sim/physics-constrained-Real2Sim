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
import time

import pyrender
import torch
from diffsdfsim.lcp_physics.physics import World
from diffsdfsim.lcp_physics.physics.utils import get_instance
from torch.nn.functional import normalize
from pytorch3d.transforms import quaternion_apply

from . import contacts as contacts_module
from .utils import Defaults3D, orthogonal, get_colormap

DIM = Defaults3D.DIM


class World3D(World):
    def __init__(self, bodies, constraints=[], dt=Defaults3D.DT, engine=Defaults3D.ENGINE,
                 contact_callback=Defaults3D.CONTACT, eps=Defaults3D.EPSILON,
                 tol=Defaults3D.TOL, fric_dirs=Defaults3D.FRIC_DIRS,
                 post_stab=Defaults3D.POST_STABILIZATION, strict_no_penetration=True, time_of_contact_diff=True,
                 stop_contact_grad=False, stop_friction_grad=False, detach_contact_b2=False):
        contact_callback_id = get_instance(contacts_module, contact_callback).__class__
        self.detach_contact_b2 = detach_contact_b2
        super().__init__(bodies, constraints=constraints, dt=dt, engine=engine,
                         contact_callback=contact_callback_id, eps=eps,
                         tol=tol, fric_dirs=fric_dirs,
                         post_stab=post_stab, strict_no_penetration=strict_no_penetration,
                         time_of_contact_diff=time_of_contact_diff, stop_contact_grad=stop_contact_grad,
                         stop_friction_grad=stop_friction_grad)
        self.static_inverse = False

    def M(self): ## Mass INERTIA matrix of the whole system
        self._M = torch.block_diag(*[b.M for b in self.bodies])
        return self._M

    def set_p(self, new_p):
        for i, b in enumerate(self.bodies):
            b.set_p(new_p[i * (self.vec_len + 1):(i + 1) * (self.vec_len + 1)])

    # def Jc(self): ## contact jacobin
    #     Jc = self._M.new_zeros(len(self.contacts), self.vec_len * len(self.bodies))
    #     for i, contact in enumerate(self.contacts):
    #         if self.stop_contact_grad:
    #             c = [c.detach() for c in contact[0]]
    #         else:
    #             c = contact[0]
    #         i1 = contact[1]
    #         i2 = contact[2]
    #         J1 = torch.cat([torch.cross(c[1], c[0]), c[0]])
    #         J2 = -torch.cat([torch.cross(c[2], c[0]), c[0]])
    #         Jc[i, i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
    #         Jc[i, i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2

    #     return Jc

    def Jc(self):
        Jc = self._M.new_zeros(len(self.contacts), self.vec_len * len(self.bodies))
        for i, contact in enumerate(self.contacts):
            if self.stop_contact_grad:
                c = [c.detach() for c in contact[0]]
            else:
                c = contact[0]
            i1 = contact[1]
            i2 = contact[2]
            J1 = torch.cat([torch.cross(c[1] - quaternion_apply(self.bodies[i1].rot, self.bodies[i1].com), c[0]), c[0]])
            J2 = -torch.cat([torch.cross(c[2]- quaternion_apply(self.bodies[i1].rot, self.bodies[i2].com), c[0]), c[0]])
            Jc[i, i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            Jc[i, i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
        return Jc
    
    # def Jf(self): ## friction Jacobian

    #     Jf = self._M.new_zeros(len(self.contacts) * self.fric_dirs,
    #                            self.vec_len * len(self.bodies))
    #     for i, contact in enumerate(self.contacts):
    #         if self.stop_friction_grad:
    #             c = [c.detach() for c in contact[0]]  # c = (normal, contact_pt_1, contact_pt_2)
    #         else:
    #             c = contact[0]
    #         i1 = contact[1]  # body 1 index
    #         i2 = contact[2]  # body 2 index

    #         dir1 = normalize(orthogonal(c[0]), dim=0)
    #         dir2 = normalize(torch.cross(dir1, c[0]), dim=0)
    #         dirs = torch.stack([dir1, dir2])
    #         if self.fric_dirs == 8:
    #             dir3 = normalize(dir1 + dir2, dim=0)
    #             dir4 = normalize(torch.cross(dir3, c[0]), dim=0)

    #             dirs = torch.cat([dirs,
    #                               torch.stack([dir3, dir4])
    #                               ], dim=0)
    #         dirs = torch.cat([dirs, -dirs], dim=0)

    #         J1 = torch.cat([torch.cross(c[1].expand(self.fric_dirs, -1), dirs), dirs], dim=1)
    #         J2 = torch.cat([torch.cross(c[2].expand(self.fric_dirs, -1), dirs), dirs], dim=1)

    #         Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs, i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
    #         Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs, i2 * self.vec_len:(i2 + 1) * self.vec_len] = -J2
    #     return Jf

    def Jf(self):

        Jf = self._M.new_zeros(len(self.contacts) * self.fric_dirs,
                               self.vec_len * len(self.bodies))
        for i, contact in enumerate(self.contacts):
            if self.stop_friction_grad:
                c = [c.detach() for c in contact[0]]  # c = (normal, contact_pt_1, contact_pt_2)
            else:
                c = contact[0]
            i1 = contact[1]  # body 1 index
            i2 = contact[2]  # body 2 index

            dir1 = normalize(orthogonal(c[0]), dim=0)
            dir2 = normalize(torch.cross(dir1, c[0]), dim=0)
            dirs = torch.stack([dir1, dir2])
            if self.fric_dirs == 8:
                dir3 = normalize(dir1 + dir2, dim=0)
                dir4 = normalize(torch.cross(dir3, c[0]), dim=0)

                dirs = torch.cat([dirs,
                                  torch.stack([dir3, dir4])
                                  ], dim=0)
            dirs = torch.cat([dirs, -dirs], dim=0)
            J1 = torch.cat([torch.cross((c[1]- quaternion_apply(self.bodies[i1].rot, self.bodies[i1].com)).expand(self.fric_dirs, -1), dirs), dirs], dim=1)
            J2 = torch.cat([torch.cross((c[2]- quaternion_apply(self.bodies[i1].rot, self.bodies[i2].com)).expand(self.fric_dirs, -1), dirs), dirs], dim=1)

            Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs, i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs, i2 * self.vec_len:(i2 + 1) * self.vec_len] = -J2
        return Jf


    def save_state(self):
        raise NotImplementedError

    def load_state(self, state_dict):
        raise NotImplementedError

    def reset_engine(self):
        raise NotImplementedError


def run_world(world, fixed_dt=False, animation_dt=None, run_time=10, print_time=True,
              scene=None, recorder=None, render_forces=False, render_torques=False, force_col=(1., 1., 1., 0.5),
              torque_col=(1., 1., 1., 0.5), force_scale=1., torque_scale=1., on_step=None):
    """Helper function to run a simulation forward once a world is created.
    """


    total_loss = None ## new callback function outputs
    # If in batched mode don't display simulation
    if hasattr(world, 'worlds'):
        scene = None

    v = None
    # if scene is not None and recorder is None:
    #     v = pyrender.Viewer(scene, run_in_thread=True, viewport_size=(640, 480), point_size=10,
    #                         # shadows=True
    #                         )

    if animation_dt is None:
        animation_dt = float(world.dt)
    elapsed_time = 0. #真实世界流逝的时间（挂钟时间）
    prev_frame_time = -animation_dt
    start_time = time.time()#真实世界开始模拟的时间（挂钟时间）

    seg_node_map = {}
    colormap = get_colormap()

    def render_step(elapsed_time, prev_frame_time):
        if elapsed_time - prev_frame_time >= animation_dt:
            if v is not None:
                v.render_lock.acquire()
            prev_frame_time = elapsed_time

            for node in scene.mesh_nodes.copy():
                scene.remove_node(node)

            for body in world.bodies:
                node = body.render(scene, world.t, render_forces, render_torques, force_col, torque_col,
                                   force_scale, torque_scale)
                seg_node_map[node] = body.col

            # c = 0
            # for node in scene.mesh_nodes.copy():
            #     seg_node_map[node] = (colormap[c][0], colormap[c][1], colormap[c][2])
            #     c += 1

            # Visualize contact points and normal for debug
            # (Uncomment contacts_debug line in contacts handler):
            # if world.contacts_debug:
            #     p1 = torch.stack([p + world.bodies[b1].pos for (_, p, _, _), b1, _ in world.contacts_debug])
            #     p2 = torch.stack([p + world.bodies[b2].pos for (_, _, p, _), _, b2 in world.contacts_debug])
            #     m1 = pyrender.Mesh.from_points(p1.detach().cpu(), colors=[0, 255, 0, 1])
            #     m2 = pyrender.Mesh.from_points(p2.detach().cpu(), colors=[0, 0, 255, 1])
            #     scene.add(m1)
            #     scene.add(m2)

            if recorder is not None:
                color_img, depth_img, pc, segmentation_mask, camera_poses = recorder.record(world.t, seg_node_map)
                world.observations.append((world.t, color_img, depth_img, pc, segmentation_mask, camera_poses))

            if v is not None:
                v.render_lock.release()

        elapsed_time = time.time() - start_time
        if recorder is None:
            # Adjust frame rate dynamically to keep real time
            wait_time = world.t - elapsed_time
            if wait_time >= 0:  # and not recorder:
                wait_time += animation_dt  # XXX
                time.sleep(max(wait_time - animation_dt, 0))
            #     animation_dt -= 0.005 * wait_time
            # elif wait_time < 0:
            #     animation_dt += 0.005 * -wait_time
            # elapsed_time = time.time() - start_time

        return elapsed_time, prev_frame_time

    if scene is not None:
        elapsed_time, prev_frame_time = render_step(elapsed_time, prev_frame_time)

    i = 0 ## step counter
    while world.t < run_time:
        i += 1
        # print("step counter:", i)
        world.step(fixed_dt=fixed_dt)
        ## new callback function
        if on_step is not None:
            loss_t = on_step(world)      # 每步一个很小的标量
            if loss_t is not None:
                total_loss = loss_t if total_loss is None else (total_loss + loss_t)

        if scene is not None:
            if v is not None and not v.is_active:
                break
            elapsed_time, prev_frame_time = render_step(elapsed_time, prev_frame_time)

        elapsed_time = time.time() - start_time
        if print_time:
            print(f"\rStep: {i:<6}  World time: {world.t:.3f}  Elapsed: {elapsed_time:.2f}s", end='', flush=True)
    if v is not None:
        v.close_external()
        for node in scene.mesh_nodes.copy():
            scene.remove_node(node)

    return total_loss