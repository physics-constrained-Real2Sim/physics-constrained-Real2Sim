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
import os
import pickle
from pathlib import Path

import pyrender
import sacred
import torch
from pytorch3d.loss import chamfer_distance
from sacred.utils import apply_backspaces_and_linefeeds
from torch.utils.tensorboard import SummaryWriter

from sdf_physics.physics3d.bodies import SDFBox, SDFSphere
from sdf_physics.physics3d.constraints import TotalConstraint3D
from sdf_physics.physics3d.forces import Gravity3D
from sdf_physics.physics3d.utils import Defaults3D, get_tensor, Rx, Ry, Recorder3D
from sdf_physics.physics3d.world import World3D, run_world

basedir = Path(__file__).resolve().parent.parent.parent.parent
experiment_basedir = basedir.joinpath('experiments', 'trajectory_fitting_sphere')
ex = sacred.Experiment(name='trajectory_fitting_sphere')
ex.observers.append(sacred.observers.FileStorageObserver(experiment_basedir))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    run_time = 1.5
    max_iter = 100
    lr = 1e-1
    conv_thresh = 1e-5
    optimizer = 'GD'

    mesh_freq = 10

    min_dim = 0.4
    max_dim = 2.0

    # resolution of the target trajectory, can be higher than the default dt
    target_dt = Defaults3D.DT
    # target_dt = 1.0/300.0

    use_toc_diff = True
    detach_2nd_bounce = True

    custom_mesh = False
    custom_inertia = False

    use_wall = True
    use_floor = True

    use_gravity = True
    use_friction = True


def random_uniform(dim, min, max):
    return torch.rand(dim, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE) * (max - min) + min


@ex.capture
def make_world(rad, use_toc_diff, use_friction, use_wall, use_floor, use_gravity, custom_mesh, custom_inertia,
               sphere_col=(0, 255, 0), sphere_pose=(1.0, 0, 0, 0, 0, 5, 0), sphere_vel=(0, 0, 0, 5, 0, 0),
               dt=Defaults3D.DT):
    bodies = []
    joints = []

    restitution = Defaults3D.RESTITUTION
    if use_friction:
        fric_coeff = 0.25  # Defaults3D.FRIC_COEFF
    else:
        fric_coeff = 0.

    if use_floor:
        floor = SDFBox([0, -.5, 0], [20, 1, 20], restitution=restitution, fric_coeff=fric_coeff, col=(255, 255, 255),
                       custom_mesh=custom_mesh, custom_inertia=custom_inertia)
        bodies.append(floor)
        joints.append(TotalConstraint3D(floor))

    if use_wall:
        wall = SDFBox([5, 5, 0], [1, 10, 10], restitution=restitution, fric_coeff=fric_coeff, col=(0, 0, 0),
                      custom_mesh=custom_mesh, custom_inertia=custom_inertia)
        joints.append(TotalConstraint3D(wall))
        bodies.append(wall)
        if use_floor:
            wall.add_no_contact(floor)

    sphere = SDFSphere(sphere_pose, rad, vel=sphere_vel, restitution=restitution, fric_coeff=fric_coeff, col=sphere_col,
                       custom_mesh=custom_mesh, custom_inertia=custom_inertia)
    if use_gravity:
        sphere.add_force(Gravity3D())
    bodies.append(sphere)

    world = World3D(bodies, joints, time_of_contact_diff=use_toc_diff, dt=dt)
    return world, sphere


def trajectory_loss(world, world_target):
    loss_x = 0.
    loss_y = 0.
    loss_z = 0.
    loss_v = 0.
    loss_w = 0.

    last_j = 0
    for i, s in enumerate(world.trajectory):

        # find closest time in target trajectory
        min_diff = 1e100
        last_diff = 1e100
        min_s_target = None
        new_j = 0
        for j, s_target in enumerate(world_target.trajectory[last_j:]):

            diff = abs(s[0] - s_target[0])
            if diff <= min_diff:
                min_diff = diff
                min_s_target = s_target
                new_j = last_j + j

            if diff > last_diff:
                break  # ascending time order
            last_diff = diff

        s_pos = s[1][-3:]
        s_vel = s[2]

        s_target_pos = min_s_target[1][-3:]
        s_target_vel = min_s_target[2]

        loss_x += ((s_pos[0] - s_target_pos[0]) ** 2).sum()
        loss_y += ((s_pos[1] - s_target_pos[1]) ** 2).sum()
        loss_z += ((s_pos[2] - s_target_pos[2]) ** 2).sum()
        loss_v += ((s_vel[3:] - s_target_vel[3:]) ** 2).sum()
        loss_w += ((s_vel[0:3] - s_target_vel[0:3]) ** 2).sum()

        last_j = new_j

    loss = (loss_x + loss_y + loss_z)
    # loss = (loss_x+loss_y+loss_z+0.001*loss_v+0.001*loss_w)
    # loss = 0.1*(loss_x+loss_y+loss_z)+0.01*loss_v+0.01*loss_w
    # loss = 0.1*loss_v
    loss /= len(world.trajectory)
    return loss


def run_world_fixed_dt(world, run_time, detach_2nd_bounce=False):
    num_contact_steps = 0
    while world.t < run_time:
        had_contacts = world.step(fixed_dt=True)
        print('\r', world.t, end='')
        if detach_2nd_bounce and had_contacts:
            num_contact_steps += 1

        if detach_2nd_bounce and had_contacts and num_contact_steps > 1:
            world.undo_step()
            world.v = world.v.detach().clone()
            world.set_v(world.v)
            world.set_p(torch.cat([b.p.detach().clone() for b in world.bodies]))
            num_contact_steps = 0
    print('\n')


@ex.command(unobserved=True)
def record_results(run_dir, output_dir, run_time):
    with open(os.path.join(run_dir, 'output.pkl'), 'rb') as f:
        data = pickle.load(f)

    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    cam = pyrender.PerspectiveCamera(yfov=math.pi / 3, aspectRatio=4 / 3)
    # cam = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=1500)
    cam_pose = get_tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 15],
                           [0, 0, 0, 1]])
    theta = math.pi / 4
    cam_pose = Rx(-theta) @ cam_pose
    scene.add(cam, pose=cam_pose.cpu())
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5)
    scene.add(light, pose=(Ry(math.pi / 2) @ Rx(-math.pi / 4)).cpu())

    colors = {
        'start': (0, 0, 255),  # blue
        'final': (0, 255, 0),  # green
        'target': (255, 255, 255)  # white
    }

    for name in ['start', 'final', 'target']:
        world = make_world(data[name + '_rad'], sphere_col=colors[name])[0]
        recorder = Recorder3D(0, scene, os.path.join(output_dir, name), save_to_disk=True)
        run_world(world, fixed_dt=True, scene=scene, run_time=run_time, recorder=recorder)


@ex.automain
def optim_sphere_trajectory(min_dim, max_dim, run_time, lr, max_iter, optimizer, detach_2nd_bounce, target_dt, _run,
                            seed):
    if _run._id:
        writer = SummaryWriter(os.path.join(experiment_basedir, str(_run._id)))
    
    target_rad = random_uniform(1, min_dim, max_dim)
    # target_rad = get_tensor(0.5)

    start_rad = random_uniform(1, min_dim, max_dim)
    # start_rad = get_tensor(0.4)
    rad = start_rad.clone()
    rad.requires_grad = True

    if optimizer == 'GD':
        optim = torch.optim.SGD([rad], lr=lr)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam([rad], lr=lr)

    world_target, sphere_target = make_world(target_rad, dt=target_dt)
    run_world_fixed_dt(world_target, run_time)

    last_loss = 1e10
    loss_hist = []
    dist_hist = []
    rad_hist = []
    for e in range(max_iter):
        optim.zero_grad()
        world, sphere = make_world(rad)

        run_world_fixed_dt(world, run_time, detach_2nd_bounce=detach_2nd_bounce)

        loss = trajectory_loss(world, world_target)

        dist = chamfer_distance(sphere.verts.unsqueeze(0), sphere_target.verts.unsqueeze(0))[0]
        #loss = dist

        loss.backward()

        loss_hist.append(loss.item())
        dist_hist.append(dist.item())
        rad_hist.append(rad.item())

        ex.log_scalar('chamfer_dist', dist.item())
        ex.log_scalar('loss', loss.item())

        if _run._id:
            writer.add_scalar('objective', loss, e)
            writer.add_scalar('chamfer_dist', dist, e)
            writer.add_scalars('rad_{}'.format(seed),
                               {'rad': rad, 'target': target_rad}, e)

        print(e, '/', max_iter)
        print('chamfer dist:', dist.item())
        print('loss:', loss.item())
        print('start rad: ', start_rad.item())
        print('rad: ', rad.item())
        print('target rad: ', target_rad.item())
        print('rad grad: ', rad.grad.item())
        print('==================================')

        if abs((last_loss - loss).item()) < 1e-5:
            break

        optim.step()
        rad.data[rad < min_dim] = min_dim
        rad.data[rad > max_dim] = max_dim
        last_loss = loss

    world, sphere = make_world(rad)
    dist = chamfer_distance(sphere.verts.unsqueeze(0), sphere_target.verts.unsqueeze(0))[0]
    run_world_fixed_dt(world, run_time, detach_2nd_bounce=detach_2nd_bounce)
    loss = trajectory_loss(world, world_target)
    ex.log_scalar('chamfer_dist', dist.item())
    ex.log_scalar('loss', loss.item())
    dist_hist.append(dist.item())
    loss_hist.append(loss.item())
    print('Final loss:', loss.item())
    print('Final chamfer dist:', dist.item())

    if _run._id:
        output_dir = os.path.join(experiment_basedir, 'outputs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, str(_run._id) + '.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump({'start_rad': start_rad,
                         'final_rad': rad,
                         'target_rad': target_rad,
                         'loss_hist': loss_hist,
                         'rad_hist': rad_hist,
                         'dist_hist': dist_hist,
                         'final_loss': loss,
                         'trajectory': world.trajectory,
                         'target_trajectory': world_target.trajectory}, f)
        _run.add_artifact(output_file, 'output.pkl')
        writer.close()
