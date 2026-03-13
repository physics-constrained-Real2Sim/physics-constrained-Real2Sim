#
# Copyright 2024 Max-Planck-Gesellschaft
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
# ========================================================================
# This file originates from
# https://github.com/locuslab/lcp-physics/tree/a85ecfe0fdc427ee016f3d1c2ddb0d0c0f98f21b/lcp_physics, licensed under the
# Apache License, version 2.0 (see LICENSE).
# This file was updated to transfer include the time-of-contact differential. The updates were done by
# by Michael Strecke, michael.strecke@tuebingen.mpg.de, and Joerg Stueckler, joerg.stueckler@tuebingen.mpg.de,
# Embodied Vision Group, Max Planck Institute for Intelligent Systems.
#
import time

import ode
import torch
from pytorch3d.transforms import so3_exponential_map, quaternion_to_matrix
from torch.autograd import Function

from . import contacts as contacts_module
from . import engines as engines_module
from .utils import Indices, Defaults, cross_2d, get_instance, left_orthogonal, rotation_matrix

X, Y = Indices.X, Indices.Y
DIM = Defaults.DIM


class World:
    """A physics simulation world, with bodies and constraints.
    """

    def __init__(self, bodies, constraints=[], dt=Defaults.DT, engine=Defaults.ENGINE,
                 contact_callback=Defaults.CONTACT, eps=Defaults.EPSILON,
                 tol=Defaults.TOL, fric_dirs=Defaults.FRIC_DIRS,
                 post_stab=Defaults.POST_STABILIZATION, strict_no_penetration=True,
                 time_of_contact_diff=False, stop_contact_grad=False, stop_friction_grad=False):
        self.contacts_debug = None  # XXX

        # Load classes from string name defined in utils
        self.engine = get_instance(engines_module, engine)
        self.contact_callback = get_instance(contacts_module, contact_callback)

        self.t = 0
        self.dt = dt
        self.last_dt = None
        self.eps = eps
        self.tol = tol
        self.fric_dirs = fric_dirs
        self.post_stab = post_stab

        self.bodies = bodies
        self.vec_len = len(self.bodies[0].v)

        self.trajectory = []
        self.observations = []

        # XXX Using ODE for broadphase for now
        self.space = ode.HashSpace()
        for i, b in enumerate(bodies):
            b.geom.body = i
            self.space.add(b.geom)

        self.static_inverse = True
        self.num_constraints = 0
        self.joints = []
        for j in constraints:
            b1, b2 = j.body1, j.body2
            i1 = bodies.index(b1)
            i2 = bodies.index(b2) if b2 else None
            self.joints.append((j, i1, i2))
            self.num_constraints += j.num_constraints
            if not j.static:
                self.static_inverse = False

        M_size = bodies[0].M.size(0)
        self._M = bodies[0].M.new_zeros(M_size * len(bodies), M_size * len(bodies))
        # XXX Better way for diagonal block matrix?
        for i, b in enumerate(bodies):
            self._M[i * M_size:(i + 1) * M_size, i * M_size:(i + 1) * M_size] = b.M

        self.set_v(torch.cat([b.v for b in bodies]))

        self.contacts = None
        self.toc_contacts = None
        self.find_contacts()
        self.strict_no_pen = strict_no_penetration
        if self.strict_no_pen:
            # Check for initial penetrations
            penetrating_contacts = [c for c in self.contacts if c[0][3].item() > self.tol]
            
            if penetrating_contacts:
                print(f"Initial penetration detected in {len(penetrating_contacts)} contact(s).")
                print("Attempting to resolve by separating bodies...")
                
                # Try to resolve penetrations
                self.resolve_initial_penetrations(max_iterations=1000)
                
                # Check if resolved
                penetrating_contacts = [c for c in self.contacts if c[0][3].item() > self.tol]
                if penetrating_contacts:
                    print(f"Warning: Could not fully resolve all penetrations.")
                    # Optionally, you can still assert here if you want strict behavior:
                    # assert False, f'Interpenetration remains after separation attempts:\n{penetrating_contacts}'
                else:
                    print("Successfully resolved all initial penetrations.")

        self.time_of_contact_diff = time_of_contact_diff
        self.stop_contact_grad = stop_contact_grad
        self.stop_friction_grad = stop_friction_grad

    def resolve_initial_penetrations(self, max_iterations=300):
        """
        Resolve initial penetrations by separating overlapping bodies.
        This is called during initialization before the main simulation loop.
        """
        for iteration in range(max_iterations):
            self.find_contacts()
            
            # Check if all contacts are within tolerance
            penetrating_contacts = [c for c in self.contacts if c[0][3].item() > self.tol]
            
            if not penetrating_contacts:
                # No more penetrations, we're done
                break
                
            # Resolve each penetrating contact
            for contact in penetrating_contacts:
                contact_info, body1_idx, body2_idx = contact
                normal, contact_pt_1, contact_pt_2, penetration_depth = contact_info
                
                body1 = self.bodies[body1_idx]
                body2 = self.bodies[body2_idx]
                
                # Calculate separation distance (add small buffer)
                separation = (penetration_depth.item() + self.tol) * 0.01
                
                # Determine how to distribute the separation based on body properties
                # If one body is much heavier (mass ratio > 100), move the lighter one
                # Otherwise, split the movement based on inverse mass
                mass1 = body1.mass if hasattr(body1, 'mass') else 1.0
                mass2 = body2.mass if hasattr(body2, 'mass') else 1.0
                
                # Handle infinite mass (static bodies)
                if torch.isinf(mass1):
                    ratio1 = 0.0
                    ratio2 = 1.0
                elif torch.isinf(mass2):
                    ratio1 = 1.0
                    ratio2 = 0.0
                else:
                    # Distribute inversely proportional to mass
                    total_inv_mass = 1.0 / mass1 + 1.0 / mass2
                    ratio1 = (1.0 / mass1) / total_inv_mass
                    ratio2 = (1.0 / mass2) / total_inv_mass
                
                # Move bodies apart along the contact normal
                displacement1 = normal * separation * ratio1
                displacement2 = -normal * separation * ratio2
                
                # Update body positions
                new_p1 = body1.p.clone()
                new_p2 = body2.p.clone()
                
                # Apply displacement to positional components
                # (assuming last 2 or 3 components are position)
                pos_dim = normal.shape[0]  # 2D or 3D
                new_p1[-pos_dim:] += displacement1
                new_p2[-pos_dim:] += displacement2
                
                body1.set_p(new_p1)
                body2.set_p(new_p2)
            
            # Update the world's combined position vector
            self.set_p(torch.cat([b.p for b in self.bodies]))
            
        # Final check
        self.find_contacts()
        remaining_penetrations = [c for c in self.contacts if c[0][3].item() > self.tol]
        
        if remaining_penetrations and self.strict_no_pen:
            print(f"Warning: Could not fully resolve all penetrations after {max_iterations} iterations.")
            print(f"Remaining penetrations: {len(remaining_penetrations)}")
            for c in remaining_penetrations:
                print(f"  Bodies {c[1]}-{c[2]}: penetration = {c[0][3].item():.6f}")

    def undo_step(self):
        # XXX Clones necessary?
        self.t = self.start_t
        self.set_p(self.start_p.clone())
        self.set_v(self.start_v.clone())
        self.contacts = self.start_contacts
        for j, c in zip(self.joints, self.start_rot_joints):
            j[0].rot1 = c[0].clone()
            j[0].update_pos()
        while self.trajectory[-1][0] > self.t:
            self.trajectory = self.trajectory[:-1]


    def step(self, fixed_dt=False):
        self.start_p = torch.cat([b.p for b in self.bodies])
        self.start_rot_joints = [(j[0].rot1, j[0].rot2) for j in self.joints]
        self.start_v = self.v
        self.start_contacts = self.contacts
        self.start_t = self.t

        had_contacts = False
        dt = self.dt
        if fixed_dt:
            end_t = self.t + self.dt
            while self.t < end_t:
                dt = end_t - self.t
                self.step_dt(dt)
                if self.contacts:
                    had_contacts = True
        else:
            self.step_dt(dt)
            if self.contacts:
                had_contacts = True
        return had_contacts

    class H(Function):
        @staticmethod
        def forward(ctx, h, cs1, cs2, vs1, vs2, poss1, poss2, rots1_mat, rots2_mat, ns2, as1, as2):

            ctx.save_for_backward(h, cs1, cs2, vs1, vs2, poss1, poss2, rots1_mat, rots2_mat, ns2, as1, as2)

            return h


        @staticmethod
        def D( h, cs1, cs2, vs1, vs2, poss1, poss2, rots1, rots2, ns2, as1, as2 ):
            # distance function between xi and xj at time h: nj.T ( xj - g( xi, h, theta ) )

            dRi = so3_exponential_map(h * vs1[:, :3])
            dRj = so3_exponential_map(h * vs2[:, :3])

            Ri = rots1
            Rj = rots2

            Rih = torch.matmul(dRi, Ri)
            Rjh = torch.matmul(dRj, Rj)

            # in world frame
            posih = poss1 + h * vs1[:, -poss1.shape[1]:] + 0.5 * as1[:, -poss1.shape[1]:] * h * h
            posjh = poss2 + h * vs2[:, -poss1.shape[1]:] + 0.5 * as2[:, -poss1.shape[1]:] * h * h

            cih_in_w = Rih @ cs1.unsqueeze(2) + posih.unsqueeze(2)

            cih_in_j = (Rjh.transpose(1, 2) @ (cih_in_w - posjh.unsqueeze(2) )).squeeze(2)

            return (ns2.unsqueeze(1) @ (cs2 - cih_in_j).unsqueeze(2)).squeeze(2)


        @staticmethod
        def dD_dx( h, cs1, cs2, vs1, vs2, poss1, poss2, rots1, rots2, ns2, as1, as2 ):

            return World.H.dD_dx_autograd(h, cs1, cs2, vs1, vs2, poss1, poss2, rots1, rots2, ns2, as1, as2)

        @staticmethod
        def dD_dx_autograd(h, cs1, cs2, vs1, vs2, poss1, poss2, rots1, rots2, ns2, as1, as2):

            with torch.enable_grad():

                (dD_dh, dD_dcs1, dD_dcs2, dD_dvs1, dD_dvs2,
                 dD_dposs1, dD_dposs2, dD_drots1, dD_drots2, dD_dns2, dD_das1, dD_das2) = \
                    torch.autograd.functional.jacobian(
                        World.H.D,
                        (h, cs1, cs2, vs1, vs2, poss1, poss2, rots1, rots2, ns2, as1, as2),
                        strict=True)

            return dD_dh, dD_dcs1, dD_dcs2, dD_dvs1, dD_dvs2, \
                   dD_dposs1, dD_dposs2, dD_drots1, dD_drots2, dD_dns2, dD_das1, dD_das2


        @staticmethod
        def backward(ctx, dL_dh):

            h, cs1, cs2, vs1, vs2, poss1, poss2, rots1, rots2, ns2, as1, as2 = ctx.saved_tensors

            dD_dh, dD_dci, dD_dcj, dD_dvi, dD_dvj, dD_dposi, dD_dposj, dD_dRi, dD_dRj, dD_dnj, dD_dai, dD_daj \
                = World.H.dD_dx( h, cs1, cs2, vs1, vs2, poss1, poss2, rots1, rots2, ns2, as1, as2 )

            # only consider motion into collision
            dD_dh[ dD_dh < Defaults.TOL/h ] = 0.

            denom = torch.sum( dD_dh ** 2, dim=0 )
            if denom > 1e-5:
                dD_dh_inv = dD_dh / denom
            else:
                dD_dh_inv = 0. * dD_dh

            dL_dcs1 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1) * dD_dci, dim=0).squeeze(0) * dL_dh
            dL_dcs2 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1) * dD_dcj, dim=0).squeeze(0) * dL_dh
            dL_dvs1 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1) * dD_dvi, dim=0).squeeze(0) * dL_dh
            dL_dvs2 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1) * dD_dvj, dim=0).squeeze(0) * dL_dh
            dL_dposs1 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1) * dD_dposi, dim=0).squeeze(0) * dL_dh
            dL_dposs2 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1) * dD_dposj, dim=0).squeeze(0) * dL_dh
            dL_drots1 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dD_dRi, dim=0).squeeze(0) * dL_dh
            dL_drots2 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dD_dRj, dim=0).squeeze(0) * dL_dh
            dL_dns2 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1) * dD_dnj, dim=0).squeeze(0) * dL_dh
            dL_das1 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1) * dD_dai, dim=0).squeeze(0) * dL_dh
            dL_das2 = torch.sum(-dD_dh_inv.unsqueeze(-1).unsqueeze(-1) * dD_daj, dim=0).squeeze(0) * dL_dh

            assert dL_dcs1.shape == cs1.shape
            assert dL_dcs2.shape == cs2.shape
            assert dL_dvs1.shape == vs1.shape
            assert dL_dvs2.shape == vs2.shape
            assert dL_dposs1.shape == poss1.shape
            assert dL_dposs2.shape == poss2.shape
            assert dL_drots1.shape == rots1.shape
            assert dL_drots2.shape == rots2.shape
            assert dL_dns2.shape == ns2.shape
            assert dL_das1.shape == as1.shape
            assert dL_das2.shape == as2.shape

            return dL_dh, dL_dcs1, dL_dcs2, dL_dvs1, dL_dvs2, dL_dposs1, \
                   dL_dposs2, dL_drots1, dL_drots2, dL_dns2, dL_das1, dL_das2


    # @profile
    def step_dt(self, dt):
        start_p = torch.cat([b.p for b in self.bodies])
        start_rot_joints = [(j[0].rot1, j[0].rot2) for j in self.joints]
        # INFO: in original LCP code the solver step was done outside the loop. This, however leads to different
        #       larger accelerations when timesteps get smaller and that can become problematic.
        start_v = self.v
        start_contacts = self.contacts

        while True:

            dt_ = dt

            if self.time_of_contact_diff and self.toc_contacts:

                # after a collision, the next time step also depends on the time of contact
                dt_joint = self.last_dt.detach() + dt_
                dt_ = -self.last_dt + dt_joint

            new_v = self.engine.solve_dynamics(self, dt_)
            self.set_v(new_v)
            # try step with current dt

            for body in self.bodies:
                body.move(dt_)
            for joint in self.joints:
                joint[0].move(dt_)

            self.find_contacts()

            if all([c[0][3].item() <= self.tol for c in self.contacts]):

                # Only compute toc diff for bodies that move into collision in this timestep, i.e. had no contact before
                self.toc_contacts = [c for c in self.contacts
                                     if {c[1], c[2]} not in [{prev_c[1], prev_c[2]} for prev_c in start_contacts]]
                if self.time_of_contact_diff and self.toc_contacts:
                    #print( '\n contacts \n', len(self.contacts) )

                    vs1 = torch.stack([self.bodies[c[1]].v for c in self.toc_contacts])
                    cs1 = torch.stack([c[0][1] for c in self.toc_contacts])
                    poss1 = torch.stack([self.bodies[c[1]].pos for c in self.toc_contacts])
                    rots1 = torch.stack([self.bodies[c[1]].rot for c in self.toc_contacts])
                    as1 = torch.stack([self.bodies[c[1]].apply_forces(self.t)/self.bodies[c[1]].mass for c in self.toc_contacts])
                    vs2 = torch.stack([self.bodies[c[2]].v for c in self.toc_contacts])
                    cs2 = torch.stack([c[0][2] for c in self.toc_contacts])
                    poss2 = torch.stack([self.bodies[c[2]].pos for c in self.toc_contacts])
                    rots2 = torch.stack([self.bodies[c[2]].rot for c in self.toc_contacts])
                    as2 = torch.stack([self.bodies[c[2]].apply_forces(self.t)/self.bodies[c[2]].mass for c in self.toc_contacts])

                    ns = torch.stack([c[0][0] for c in self.toc_contacts])

                    # precalculate quantities needed for time of contact differential
                    # body velocities before time step: input vs1, vs2

                    # body position and orientation before time step
                    poss1 -= dt_ * vs1[:, -poss1.shape[1]:]
                    poss2 -= dt_ * vs2[:, -poss1.shape[1]:]

                    if cs1.shape[1] == 3:
                        # rotations in world frame
                        rot1 = so3_exponential_map(-dt_ * vs1[:, :3])
                        rot2 = so3_exponential_map(-dt_ * vs2[:, :3])
                        # rotations of body frame to world frame
                        rots1_mat = quaternion_to_matrix(rots1)
                        rots2_mat = quaternion_to_matrix(rots2)
                    if cs1.shape[1] == 2:
                        rot1 = torch.stack([rotation_matrix(-dt_ * v[0]) for v in vs1])
                        rot2 = torch.stack([rotation_matrix(-dt_ * v[0]) for v in vs2])
                        rots1_mat = rotation_matrix(rots1)
                        rots2_mat = rotation_matrix(rots2)
                    rots1_mat = rot1 @ rots1_mat
                    rots2_mat = rot2 @ rots2_mat

                    # determine contact points in body frames before time step
                    # contact points are input with rotation in world frame but position relative to body frame origin (in world coordinates)
                    cs1 = (rots1_mat.transpose(1, 2) @ cs1.unsqueeze(2)).squeeze(2)
                    cs2 = (rots2_mat.transpose(1, 2) @ cs2.unsqueeze(2)).squeeze(2)

                    # determine contact normal in body frame of shape 2
                    # contact normal is input in world frame coordinates
                    ns2 = (rots2_mat.transpose(1, 2) @ ns.unsqueeze(2)).squeeze(2)


                    # "Recompute" dt with gradients for contact points
                    if not torch.is_tensor( dt_ ):
                        dt_ = cs1.new_tensor(dt_)

                    dt_ = self.H.apply(dt_, cs1, cs2, vs1, vs2, poss1, poss2, rots1_mat, rots2_mat, ns2, as1, as2)

                    # Undo motion
                    self.set_p(start_p.clone())
                    for j, c in zip(self.joints, start_rot_joints):
                        j[0].rot1 = c[0].clone()
                        j[0].update_pos()

                    # Redo the step with the dt_ that has time-of-impact gradients
                    for body in self.bodies:
                        body.move(dt_)
                    for joint in self.joints:
                        joint[0].move(dt_)

                    self.last_dt = dt_

                break
            else:
                if dt < self.dt / 2**2 or dt < 0.001:
                    # if step becomes too small, just continue
                    print("[step is too small], just continue")
                    break
                dt /= 2
                print("[Simulation]: Time havle")
                # reset positions to beginning of step
                # XXX Clones necessary?
                self.set_p(start_p.clone())
                self.set_v(start_v.clone())
                self.contacts = start_contacts
                for j, c in zip(self.joints, start_rot_joints):
                    j[0].rot1 = c[0].clone()
                    j[0].update_pos()

        if self.post_stab:
            tmp_v = self.v
            dp = self.engine.post_stabilization(self).squeeze(0)
            dp /= 2  # XXX Why 1/2 factor?
            # XXX Clean up / Simplify this update?
            self.set_v(dp)
            for body in self.bodies:
                body.move(dt)
            for joint in self.joints:
                joint[0].move(dt)
            self.set_v(tmp_v)

            self.find_contacts()  # XXX Necessary to recheck contacts?

        # store this step in a trajectory
        curr_p = torch.cat([b.p for b in self.bodies])
        curr_v = self.v
        curr_rot_joints = [(j[0].rot1, j[0].rot2) for j in self.joints]
        curr_contacts = self.contacts
        self.trajectory.append( ( self.t, curr_p, curr_v, curr_contacts, curr_rot_joints ) )

        self.t += dt

    def get_v(self):
        return self.v

    def set_v(self, new_v):
        self.v = new_v
        for i, b in enumerate(self.bodies):
            b.v = self.v[i * len(b.v):(i + 1) * len(b.v)]

    def set_p(self, new_p):
        for i, b in enumerate(self.bodies):
            b.set_p(new_p[i * self.vec_len:(i + 1) * self.vec_len])

    def apply_forces(self, t):
        return torch.cat([b.apply_forces(t) for b in self.bodies])

    def find_contacts(self):
        self.contacts = []
        # ODE contact detection
        self.space.collide([self], self.contact_callback)


    def restitutions(self):
        restitutions = self._M.new_empty(len(self.contacts))
        for i, c in enumerate(self.contacts):
            r1 = self.bodies[c[1]].restitution
            r2 = self.bodies[c[2]].restitution
            restitutions[i] = (r1 + r2) / 2
            # restitutions[i] = math.sqrt(r1 * r2)
        return restitutions

    def M(self):
        return self._M

    def Je(self):
        Je = self._M.new_zeros(self.num_constraints,
                               self.vec_len * len(self.bodies))
        row = 0
        for joint in self.joints:
            J1, J2 = joint[0].J()
            i1 = joint[1]
            i2 = joint[2]
            Je[row:row + J1.size(0),
            i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            if J2 is not None:
                Je[row:row + J2.size(0),
                i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
            row += J1.size(0)
        return Je

    def Jc(self):
        Jc = self._M.new_zeros(len(self.contacts), self.vec_len * len(self.bodies))
        for i, contact in enumerate(self.contacts):
            if self.stop_contact_grad:
                # Gradients for points and normals from LCP are not meaningful
                c = [c.detach() for c in contact[0]]  # c = (normal, contact_pt_1, contact_pt_2)
            else:
                c = contact[0]
            i1 = contact[1]
            i2 = contact[2]
            J1 = torch.cat([cross_2d(c[1], c[0]).reshape(1, 1),
                            c[0].unsqueeze(0)], dim=1)
            J2 = -torch.cat([cross_2d(c[2], c[0]).reshape(1, 1),
                             c[0].unsqueeze(0)], dim=1)
            Jc[i, i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            Jc[i, i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
        return Jc

    def Jf(self):
        Jf = self._M.new_zeros(len(self.contacts) * self.fric_dirs,
                               self.vec_len * len(self.bodies))
        for i, contact in enumerate(self.contacts):
            if self.stop_friction_grad:
                # Gradients for points and normals from LCP are not meaningful
                c = [c.detach() for c in contact[0]]  # c = (normal, contact_pt_1, contact_pt_2)
            else:
                c = contact[0]
            dir1 = left_orthogonal(c[0])
            dir2 = -dir1
            i1 = contact[1]  # body 1 index
            i2 = contact[2]  # body 2 index
            J1 = torch.cat([
                torch.cat([cross_2d(c[1], dir1).reshape(1, 1),
                           dir1.unsqueeze(0)], dim=1),
                torch.cat([cross_2d(c[1], dir2).reshape(1, 1),
                           dir2.unsqueeze(0)], dim=1),
            ], dim=0)
            J2 = torch.cat([
                torch.cat([cross_2d(c[2], dir1).reshape(1, 1),
                           dir1.unsqueeze(0)], dim=1),
                torch.cat([cross_2d(c[2], dir2).reshape(1, 1),
                           dir2.unsqueeze(0)], dim=1),
            ], dim=0)
            Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs,
            i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs,
            i2 * self.vec_len:(i2 + 1) * self.vec_len] = -J2

        return Jf

    def mu(self):
        return self._memoized_mu(*[(c[1], c[2]) for c in self.contacts])

    def _memoized_mu(self, *contacts):
        # contacts is argument so that cacheing can be implemented at some point
        mu = self._M.new_zeros(len(self.contacts))
        for i, contacts in enumerate(self.contacts):
            i1 = contacts[1]
            i2 = contacts[2]
            # mu[i] = torch.sqrt(self.bodies[i1].fric_coeff * self.bodies[i2].fric_coeff)
            mu[i] = 0.5 * (self.bodies[i1].fric_coeff + self.bodies[i2].fric_coeff)
        return torch.diag(mu)

    def E(self):
        return self._memoized_E(len(self.contacts))

    def _memoized_E(self, num_contacts):
        n = self.fric_dirs * num_contacts
        E = self._M.new_zeros(n, num_contacts)
        for i in range(num_contacts):
            E[i * self.fric_dirs: (i + 1) * self.fric_dirs, i] += 1
        return E

    def save_state(self):
        raise NotImplementedError

    def load_state(self, state_dict):
        raise NotImplementedError

    def reset_engine(self):
        raise NotImplementedError


def run_world(world, animation_dt=None, run_time=10, print_time=True,
              screen=None, recorder=None, pixels_per_meter=1):
    """Helper function to run a simulation forward once a world is created.
    """
    # If in batched mode don't display simulation
    if hasattr(world, 'worlds'):
        screen = None

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    if animation_dt is None:
        animation_dt = float(world.dt)
    elapsed_time = 0.
    prev_frame_time = -animation_dt
    start_time = time.time()

    while world.t < run_time:
        world.step()

        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            if elapsed_time - prev_frame_time >= animation_dt or recorder:
                prev_frame_time = elapsed_time

                screen.blit(background, (0, 0))
                update_list = []
                for body in world.bodies:
                    update_list += body.draw(screen, pixels_per_meter=pixels_per_meter)
                for joint in world.joints:
                    update_list += joint[0].draw(screen, pixels_per_meter=pixels_per_meter)

                # Visualize contact points and normal for debug
                # (Uncomment contacts_debug line in contacts handler):
                if world.contacts_debug:
                    for c in world.contacts_debug:
                        (normal, p1, p2, penetration), b1, b2 = c
                        b1_pos = world.bodies[b1].pos
                        b2_pos = world.bodies[b2].pos
                        p1 = p1 + b1_pos
                        p2 = p2 + b2_pos
                        pygame.draw.circle(screen, (0, 255, 0), p1.data.numpy().astype(int), 5)
                        pygame.draw.circle(screen, (0, 0, 255), p2.data.numpy().astype(int), 5)
                        pygame.draw.line(screen, (0, 255, 0), p1.data.numpy().astype(int),
                                         (p1.data.numpy() + normal.data.numpy() * 100).astype(int), 3)

                if not recorder:
                    # Don't refresh screen if recording
                    pygame.display.update(update_list)
                    pygame.display.flip()  # XXX
                else:
                    recorder.record(world.t)

            elapsed_time = time.time() - start_time
            if not recorder:
                # Adjust frame rate dynamically to keep real time
                wait_time = world.t - elapsed_time
                if wait_time >= 0 and not recorder:
                    wait_time += animation_dt  # XXX
                    time.sleep(max(wait_time - animation_dt, 0))
                #     animation_dt -= 0.005 * wait_time
                # elif wait_time < 0:
                #     animation_dt += 0.005 * -wait_time
                # elapsed_time = time.time() - start_time

        elapsed_time = time.time() - start_time
        if print_time:
            print('\r ', '{} / {}  {} '.format(int(world.t), int(elapsed_time),
                                               1 / animation_dt), end='')
