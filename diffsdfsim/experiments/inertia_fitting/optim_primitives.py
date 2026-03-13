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
import json
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

from sdf_physics.physics3d.bodies import SDFBox, SDFSphere, SDFCylinder
from sdf_physics.physics3d.constraints import XConstraint, YConstraint, ZConstraint
from sdf_physics.physics3d.forces import ExternalForce3D
from sdf_physics.physics3d.utils import Defaults3D, Recorder3D, Rx, get_tensor
from sdf_physics.physics3d.world import World3D, run_world

basedir = Path(__file__).resolve().parent.parent.parent.parent
experiment_basedir = basedir.joinpath('experiments', 'inertia_fitting_primitives')
ex = sacred.Experiment(name='inertia_fitting_primitives')
ex.observers.append(sacred.observers.FileStorageObserver(experiment_basedir))
ex.captured_out_filter = apply_backspaces_and_linefeeds


def make_box(dims, custom_mesh=True, custom_inertia=True, col=(255, 255, 255)):
    return SDFBox([0, 0, 0], dims, custom_mesh=custom_mesh, custom_inertia=custom_inertia, col=col)


def make_sphere(rad, custom_mesh=True, custom_inertia=True, col=(255, 255, 255)):
    return SDFSphere([0, 0, 0], rad, custom_mesh=custom_mesh, custom_inertia=custom_inertia, col=col)


def make_cylinder(dims, custom_mesh=True, custom_inertia=True, col=(255, 255, 255)):
    return SDFCylinder([0, 0, 0], rad=dims[0], height=dims[1], custom_mesh=custom_mesh, custom_inertia=custom_inertia,
                       col=col)


def random_uniform(dim, max, min):
    return torch.rand(dim, device=Defaults3D.DEVICE, dtype=Defaults3D.DTYPE) * (max - min) + min


@ex.config
def cfg():
    run_time = 2
    max_iter = 200
    lr = 1e-2
    conv_thresh = 1e-5
    optimizer = 'Adam'

    mesh_freq = 10

    min_dim = 0.5
    max_dim = 2.0

    custom_mesh = False
    custom_inertia = False


@ex.named_config
def box():
    dim = 3
    make_obj = make_box


@ex.named_config
def sphere():
    dim = 1
    make_obj = make_sphere


@ex.named_config
def cylinder():
    dim = 2
    make_obj = make_cylinder


@ex.capture
def make_world(dims, make_obj, torque_dir, custom_mesh=False, custom_inertia=False, obj_col=(255, 255, 255)):
    bodies = []
    joints = []

    def torque(t):
        if t < 0.3:
            force = torch.cat([torque_dir, torque_dir.new_zeros(3)])
        else:
            force = torque_dir.new_zeros(6)
        return force

    obj = make_obj(dims, custom_mesh, custom_inertia, col=obj_col)
    obj.add_force(ExternalForce3D(torque))
    bodies.append(obj)
    joints.append(XConstraint(obj))
    joints.append(YConstraint(obj))
    joints.append(ZConstraint(obj))

    world = World3D(bodies, joints)

    return world, obj


@ex.command(unobserved=True)
def record_results(run_dir, output_dir, run_time):
    with open(os.path.join(run_dir, 'output.pkl'), 'rb') as f:
        data = pickle.load(f)

    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    cam = pyrender.PerspectiveCamera(yfov=math.pi / 3, aspectRatio=4 / 3)
    # cam = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=1500)
    cam_pose = get_tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 3],
                           [0, 0, 0, 1]])
    theta = math.pi / 4
    cam_pose = Rx(-theta) @ cam_pose
    scene.add(cam, pose=cam_pose.cpu())
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5)
    scene.add(light, pose=[[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, -1, 0, 0],
                           [0, 0, 0, 1]])

    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)

    colors = {
        'start': (0, 0, 255),  # blue
        'final': (0, 255, 0),  # green
        'target': (255, 255, 255)  # white
    }

    make_obj = eval(config['make_obj']['py/function'].split('.')[-1])
    for name in ['start', 'final', 'target']:
        world = make_world(data[name + '_dims'], make_obj=make_obj, torque_dir=data['torque_dir'],
                           obj_col=colors[name])[0]
        recorder = Recorder3D(0, scene, os.path.join(output_dir, name), save_to_disk=True)
        run_world(world, fixed_dt=True, scene=scene, run_time=run_time, recorder=recorder, render_torques=True,
                  torque_col=(1., 0., 0., 0.5), torque_scale=1.)


@ex.automain
def inertia_optim(run_time, max_iter, lr, conv_thresh, optimizer, dim, max_dim, min_dim, mesh_freq, _run, seed):
    if _run._id:
        writer = SummaryWriter(os.path.join(experiment_basedir, str(_run._id)))

    torque_dir = torch.zeros(3, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)
    while torque_dir.norm() < 1e-8:
        torque_dir = torch.randn(3, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)
    torque_dir = torque_dir / torque_dir.norm()

    target_dims = random_uniform(dim, max_dim, min_dim)

    start_dims = random_uniform(dim, max_dim, min_dim)

    dims = start_dims.clone()
    dims.requires_grad = True

    if optimizer == 'GD':
        optim = torch.optim.SGD([dims], lr=lr)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam([dims], lr=lr)

    world_target, obj_target = make_world(target_dims, torque_dir=torque_dir)
    run_world(world_target, run_time=run_time)

    if _run._id:
        writer.add_mesh('target_mesh', obj_target.verts.unsqueeze(0), faces=obj_target.faces.unsqueeze(0))

    loss_hist = []
    dims_hist = []
    dist_hist = []
    last_loss = 1e10
    for e in range(max_iter):
        optim.zero_grad()
        world, obj = make_world(dims, torque_dir=torque_dir)
        run_world(world, run_time=run_time)

        dist = chamfer_distance(obj.verts.unsqueeze(0), obj_target.verts.unsqueeze(0))[0]

        loss = ((obj.v - obj_target.v) ** 2).sum()
        loss.backward()

        dist_hist.append(dist.item())
        loss_hist.append(loss.item())
        dims_hist.append(dims.tolist())

        ex.log_scalar('chamfer_dist', dist.item())
        ex.log_scalar('loss', loss.item())

        if _run._id:
            writer.add_scalar('objective', loss, e)
            writer.add_scalar('chamfer_dist', dist, e)
            if dims.nelement() == 1:
                writer.add_scalars('dims_{}'.format(seed),
                                   {'dims': dims.squeeze(), 'target': target_dims.squeeze()}, e)
            else:
                writer.add_scalars('dims_{}'.format(seed),
                                   dict({'dims_{}'.format(i): v for i, v in enumerate(dims.squeeze())},
                                        **{'target_{}'.format(i): v for i, v in enumerate(target_dims.squeeze())}), e)
            if mesh_freq > 0 and e % mesh_freq == 0:
                writer.add_mesh('mesh', obj.verts.unsqueeze(0), faces=obj.faces.unsqueeze(0), global_step=e)

        print('\n', e, '/', max_iter)
        print('chamfer dist: ', dist.item())
        print('loss: ', loss.item())
        print('start dims: ', start_dims.tolist())
        print('dims: ', dims.tolist())
        print('target dims: ', target_dims.tolist())
        print('dim grads: ', dims.grad.tolist())
        print('==================================')

        if abs((last_loss - loss).item()) < conv_thresh:
            break

        optim.step()
        dims.data[dims < min_dim] = min_dim
        dims.data[dims > max_dim] = max_dim
        last_loss = loss

    world, obj = make_world(dims, torque_dir=torque_dir)
    run_world(world, run_time=run_time)
    dist = chamfer_distance(obj.verts.unsqueeze(0), obj_target.verts.unsqueeze(0))[0]
    loss = ((obj.v - obj_target.v) ** 2).sum()
    ex.log_scalar('chamfer_dist', dist.item())
    ex.log_scalar('loss', loss.item())
    dist_hist.append(dist.item())
    loss_hist.append(loss.item())
    print('Final loss:', loss.item())
    print('Final chamfer dist:', dist.item())

    if _run._id:
        if not os.path.exists(os.path.join(experiment_basedir, 'outputs')):
            os.makedirs(os.path.join(experiment_basedir, 'outputs'))
        output_file = os.path.join(experiment_basedir, 'outputs', _run._id + '.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump({'start_dims': start_dims,
                         'final_dims': dims,
                         'target_dims': target_dims,
                         'loss_hist': loss_hist,
                         'dims_hist': dims_hist,
                         'dist_hist': dist_hist,
                         'torque_dir': torque_dir}, f)
        _run.add_artifact(output_file, 'output.pkl')
        writer.close()

    return dist.item()
