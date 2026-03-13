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

from sdf_physics.physics3d.bodies import SDF3D
from sdf_physics.physics3d.constraints import XConstraint, YConstraint, ZConstraint
from sdf_physics.physics3d.forces import ExternalForce3D
from sdf_physics.physics3d.utils import load_igrnet, decode_igr, Defaults3D, get_tensor, Rx, Recorder3D
from sdf_physics.physics3d.world import World3D, run_world

basedir = Path(__file__).resolve().parent.parent.parent.parent
experiment_basedir = basedir.joinpath('experiments', 'inertia_fitting_shapespace')
shapespace_basedir = basedir.joinpath('shapespaces', 'IGR', 'models')
ex = sacred.Experiment(name='inertia_fitting_shapespace')
ex.observers.append(sacred.observers.FileStorageObserver(experiment_basedir))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    run_time = 2
    max_iter = 200
    lr = 1e-3
    conv_thresh = 1e-5
    latent_reg = 1e-4
    optimizer = 'Adam'

    mesh_freq = 10

    init_mode = 'normal'

    name = 'bob_and_spot'
    timestamp = 'latest'
    checkpoint = 'latest'


def random_sample(latents):
    return latents[torch.randint(latents.shape[0], (1,))].detach().clone()


def random_normal(mu, sigma):
    return torch.normal(mu, sigma)


@ex.capture
def make_world(decoder, latent, torque_dir, obj_col=(255, 255, 255)):
    bodies = []
    joints = []

    def torque(t):
        if t < 0.3:
            force = torch.cat([torque_dir, torque_dir.new_zeros(3)])
        else:
            force = torque_dir.new_zeros(6)
        return force

    obj = SDF3D([0, 0, 0], 1., decode_igr(decoder), [latent], col=obj_col)
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

    with open(os.path.join(run_dir, 'run.json')) as f:
        run = json.load(f)
    shapespace_dir = '/'.join(run['resources'][0][0].split('/')[:-2])
    timestamp = run['resources'][0][0].split('/')[-2]
    checkpoint = os.path.splitext(os.path.basename(run['resources'][1][0]))[0]
    decoder, latents = load_igrnet(shapespace_dir, timestamp, checkpoint)

    colors = {
        'start': (0, 0, 255),  # blue
        'final': (0, 255, 0),  # green
        'target': (255, 255, 255)  # white
    }

    for name in ['start', 'final', 'target']:
        world = make_world(decoder, data[name + '_latent'], data['torque_dir'], obj_col=colors[name])[0]
        recorder = Recorder3D(0, scene, os.path.join(output_dir, name), save_to_disk=True)
        run_world(world, fixed_dt=True, scene=scene, run_time=run_time, recorder=recorder, render_torques=True,
                  torque_col=(1., 0., 0., 0.5), torque_scale=1.)


@ex.automain
def inertia_optim(run_time, max_iter, lr, conv_thresh, latent_reg, optimizer, name, timestamp, checkpoint, mesh_freq,
                  init_mode, _run, seed):
    if _run._id:
        writer = SummaryWriter(os.path.join(experiment_basedir, str(_run._id)))

    torque_dir = torch.zeros(3, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)
    while torque_dir.norm() < 1e-8:
        torque_dir = torch.randn(3, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)
    torque_dir = torque_dir / torque_dir.norm()

    decoder, latents = load_igrnet(shapespace_basedir.joinpath(name), timestamp, checkpoint, _run)

    # Select random training latent as target
    target_latent = random_sample(latents)

    if init_mode == 'sample':
        start_latent = random_sample(latents)
    elif init_mode == 'normal':
        mu = latents.mean(dim=0)
        sigma = 0.1 * latents.std(dim=0)
        start_latent = random_normal(mu, sigma)
    else:
        raise ValueError('Unknown init_mode: {}'.format(init_mode))

    latent = start_latent.clone()
    latent.requires_grad = True

    if optimizer == 'GD':
        optim = torch.optim.SGD([latent], lr=lr)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam([latent], lr=lr)

    world_target, obj_target = make_world(decoder, target_latent, torque_dir=torque_dir)
    run_world(world_target, run_time=run_time)

    if _run._id:
        writer.add_mesh('target_mesh', obj_target.verts.unsqueeze(0), faces=obj_target.faces.unsqueeze(0))

    loss_hist = []
    latent_hist = []
    dist_hist = []
    last_loss = 1e10
    for e in range(max_iter):
        optim.zero_grad()
        world, obj = make_world(decoder, latent, torque_dir=torque_dir)
        run_world(world, run_time=run_time)

        dist = chamfer_distance(obj.verts.unsqueeze(0), obj_target.verts.unsqueeze(0))[0]

        loss = ((obj.v - obj_target.v) ** 2).sum() + latent_reg * latent.norm() ** 2
        loss.backward()

        dist_hist.append(dist.item())
        loss_hist.append(loss.item())
        latent_hist.append(latent.tolist())

        ex.log_scalar('chamfer_dist', dist.item())
        ex.log_scalar('loss', loss.item())

        if _run._id:
            writer.add_scalar('objective', loss, e)
            writer.add_scalar('chamfer_dist', dist, e)
            if latent.nelement() == 1:
                writer.add_scalars('latent_code_{}'.format(seed),
                                   {'latent': latent.squeeze(), 'target': target_latent.squeeze()}, e)
            else:
                writer.add_scalars('latent_code_{}'.format(seed),
                                   dict({'latent_{}'.format(i): v for i, v in enumerate(latent.squeeze())},
                                        **{'target_{}'.format(i): v for i, v in enumerate(target_latent.squeeze())}), e)
            if mesh_freq > 0 and e % mesh_freq == 0:
                writer.add_mesh('mesh', obj.verts.unsqueeze(0), faces=obj.faces.unsqueeze(0), global_step=e)

        print('\n', e, '/', max_iter)
        print('chamfer dist: ', dist.item())
        print('loss: ', loss.item())
        print('start latent: ', start_latent.tolist())
        print('latent: ', latent.tolist())
        print('target latent: ', target_latent.tolist())
        print('latent grads: ', latent.grad.tolist())
        print('==================================')

        if abs((last_loss - loss).item()) < conv_thresh:
            break

        optim.step()
        last_loss = loss

    world, obj = make_world(decoder, latent, torque_dir=torque_dir)
    run_world(world, run_time=run_time)
    dist = chamfer_distance(obj.verts.unsqueeze(0), obj_target.verts.unsqueeze(0))[0]
    loss = ((obj.v - obj_target.v) ** 2).sum() + latent_reg * latent.norm() ** 2
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
            pickle.dump({'start_latent': start_latent,
                         'final_latent': latent,
                         'target_latent': target_latent,
                         'loss_hist': loss_hist,
                         'latent_hist': latent_hist,
                         'dist_hist': dist_hist,
                         'torque_dir': torque_dir}, f)
        _run.add_artifact(output_file, 'output.pkl')
        writer.close()

    return dist.item()
