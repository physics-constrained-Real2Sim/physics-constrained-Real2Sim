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
import time
from pathlib import Path

import pyrender
import sacred
import torch
from sacred.utils import apply_backspaces_and_linefeeds
from torch.utils.tensorboard import SummaryWriter

from sdf_physics.physics3d.bodies import SDF3D, SDFBox
from sdf_physics.physics3d.constraints import TotalConstraint3D
from sdf_physics.physics3d.forces import ExternalForce3D, Gravity3D
from sdf_physics.physics3d.utils import load_igrnet, decode_igr, Defaults3D, get_tensor, Rx, Recorder3D
from sdf_physics.physics3d.world import World3D, run_world

basedir = Path(__file__).resolve().parent.parent.parent.parent
experiment_basedir = basedir.joinpath('experiments', 'system_identification')
shapespace_basedir = basedir.joinpath('shapespaces', 'IGR', 'models')
ex = sacred.Experiment(name='system_identification')
ex.observers.append(sacred.observers.FileStorageObserver(experiment_basedir))
ex.captured_out_filter = apply_backspaces_and_linefeeds


def random_sample_latent(latents):
    return latents[torch.randint(latents.shape[0], (1,))].detach().clone()


def random_sample_uni(dim, min, max):
    return torch.rand(dim, device=Defaults3D.DEVICE, dtype=Defaults3D.DTYPE) * (max - min) + min


@ex.config
def cfg():
    goal = 'mass'
    run_time = 1
    max_iter = 100
    lr = 1e-2
    conv_thresh = 1e-5
    optimizer = 'GD'

    restitution = 0.

    min_force = 2.0
    max_force = 5.0

    min_mass = 0.9
    max_mass = 1.1

    min_fric = 0.01
    max_fric = 0.25

    mesh_freq = 10

    name = 'bob_and_spot'
    timestamp = '2021_07_21_17_59_08'
    checkpoint = 'latest'

    use_toc_diff = True

    strict_no_penetration = False

    fric_dirs = 8


@ex.named_config
def mass():
    goal = 'mass'
    lr = 1e-2


@ex.named_config
def friction():
    goal = 'friction'
    lr = 1e-3


@ex.named_config
def force():
    goal = 'force'
    lr = 1e-1


@ex.capture
def make_world(decoder, latent, force, mass, fric_coeff, restitution, use_toc_diff, strict_no_penetration, fric_dirs,
               obj_col=(255, 255, 255)):
    bodies = []
    joints = []

    def force_func(t):
        force_vec = get_tensor([0, 0, 0, 0, 0, 0])
        # if t < 0.1:
        force_vec[[-3, -1]] = force
        return force_vec

    floor = SDFBox([0, -.5, 0], [20, 1, 20], fric_coeff=fric_coeff, col=(255, 255, 255), restitution=restitution)
    bodies.append(floor)
    joints.append(TotalConstraint3D(floor))

    obj = SDF3D([0, 0, 0], scale=1, sdf_func=decode_igr(decoder), params=[latent], mass=mass,
                fric_coeff=fric_coeff, col=obj_col, restitution=restitution)
    obj_pos = get_tensor([0, 0, 0])
    obj_pos[1] = -obj.verts.min(dim=0)[0][1] + 2 * Defaults3D.EPSILON
    obj.set_p(torch.cat([obj_pos.new_ones(1), obj_pos.new_zeros(3), obj_pos]))
    obj.add_force(Gravity3D())
    obj.add_force(ExternalForce3D(force_func))
    bodies.append(obj)

    world = World3D(bodies, joints, time_of_contact_diff=use_toc_diff, strict_no_penetration=strict_no_penetration,
                    fric_dirs=fric_dirs)
    return world, obj


@ex.command(unobserved=True)
def record_results(run_dir, output_dir, run_time):
    with open(os.path.join(run_dir, 'output.pkl'), 'rb') as f:
        data = pickle.load(f)

    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    cam = pyrender.PerspectiveCamera(yfov=math.pi / 3, aspectRatio=4 / 3)
    # cam = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=1500)
    cam_pose = get_tensor([[1, 0, 0, 2],
                           [0, 1, 0, 0],
                           [0, 0, 1, 5],
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

    arrow_cols = {
        'start': (1., 0, 0, 0.5),
        'final': (1., 0, 0, 0.5),
        'target': (1., 1., 1., 0.5)
    }

    for name in ['start', 'final', 'target']:
        world = make_world(decoder, data['latent'], data[name + '_force'], data[name + '_mass'],
                           data[name + '_fric'], obj_col=colors[name])[0]
        recorder = Recorder3D(0, scene, os.path.join(output_dir, name), save_to_disk=True)
        run_world(world, fixed_dt=True, scene=scene, run_time=run_time, recorder=recorder, render_forces=True,
                  force_col=arrow_cols[name], force_scale=0.5)


@ex.automain
def pushing_optim(goal, run_time, max_iter, lr, conv_thresh, optimizer, name, timestamp, checkpoint,
                  min_force, max_force, min_mass, max_mass, min_fric, max_fric, _run, seed):
    if _run._id:
        writer = SummaryWriter(os.path.join(experiment_basedir, str(_run._id)))

    decoder, latents = load_igrnet(shapespace_basedir.joinpath(name), timestamp, checkpoint, _run)

    # Select random training latent as target
    latent = random_sample_latent(latents).clone()

    target_force = random_sample_uni(2, min_force, max_force)

    target_mass = random_sample_uni(1, min_mass, max_mass)

    target_fric_coeff = random_sample_uni(1, min_fric, max_fric)

    optim_params = []
    if goal == 'mass':
        force = start_force = target_force.clone()
        fric_coeff = start_fric_coeff = target_fric_coeff.clone()
        start_mass = random_sample_uni(1, min_mass, max_mass)
        mass = start_mass.clone()
        mass.requires_grad = True
        optim_params.append(mass)
    elif goal == 'force':
        mass = start_mass = target_mass.clone()
        fric_coeff = start_fric_coeff = target_fric_coeff.clone()
        start_force = random_sample_uni(2, min_force, max_force)
        force = start_force.clone()
        force.requires_grad = True
        optim_params.append(force)
    elif goal == 'friction':
        force = start_force = target_force.clone()
        mass = start_mass = target_mass.clone()
        start_fric_coeff = random_sample_uni(1, min_fric, max_fric)
        fric_coeff = start_fric_coeff.clone()
        fric_coeff.requires_grad = True
        optim_params.append(fric_coeff)
    else:
        raise ValueError('Invalid goal: {}'.format(goal))

    if optimizer == 'GD':
        optim = torch.optim.SGD(optim_params, lr=lr)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(optim_params, lr=lr)
    else:
        raise ValueError('Invalid optimizer: {}'.format(optimizer))

    world_target, obj_target = make_world(decoder, latent, target_force, target_mass, target_fric_coeff)
    target_poses = []
    start = time.time()
    while world_target.t < run_time:
        world_target.step(fixed_dt=True)
        target_poses.append(obj_target.pos)
        print('\r', world_target.t, '/', time.time() - start, end='')

    loss_hist = []
    dist_hist = []
    force_hist = []
    fric_hist = []
    mass_hist = []
    last_loss = 1e10
    for e in range(max_iter):
        optim.zero_grad()
        world, obj = make_world(decoder, latent, force, mass, fric_coeff)

        loss = 0.
        start = time.time()
        for target_pose in target_poses:
            world.step(fixed_dt=True)
            loss += ((target_pose - obj.pos) ** 2).sum()
            print('\r', world.t, '/', time.time() - start, end='')

        loss.backward()

        if force.requires_grad:
            force_hist.append(force.tolist())
            dist = (force - target_force).norm()
        if fric_coeff.requires_grad:
            fric_hist.append(fric_coeff.item())
            dist = (fric_coeff - target_fric_coeff).abs()
        if mass.requires_grad:
            mass_hist.append(mass.item())
            dist = (mass - target_mass).abs()

        loss_hist.append(loss.item())
        dist_hist.append(dist.item())

        ex.log_scalar('loss', loss.item())
        ex.log_scalar('dist', dist.item())

        if _run._id:
            writer.add_scalar('objective', loss, e)
            writer.add_scalar('dist', dist, e)
            if force.requires_grad:
                writer.add_scalars('force_{}'.format(seed),
                                   {'force_0': force[0], 'force_1': force[1],
                                    'target_force_0': target_force[0], 'target_force_1': target_force[1]}, e)
            if mass.requires_grad:
                writer.add_scalars('mass_{}'.format(seed),
                                   {'mass': mass,
                                    'target_mass': target_mass}, e)
            if fric_coeff.requires_grad:
                writer.add_scalars('fric_coeff_{}'.format(seed),
                                   {'fric_coeff': fric_coeff,
                                    'target_fric_coeff': target_fric_coeff}, e)

        print('\n', e, '/', max_iter)
        print('loss: ', loss.item())
        if force.requires_grad:
            print('start force: ', start_force.tolist())
            print('force: ', force.tolist())
            print('target force: ', target_force.tolist())
            print('force grads: ', force.grad.tolist())
        if mass.requires_grad:
            print('start mass: ', start_mass.item())
            print('mass: ', mass.item())
            print('target mass: ', target_mass.item())
            print('mass grads: ', mass.grad.item())
        if fric_coeff.requires_grad:
            print('start fric_coeff: ', start_fric_coeff.item())
            print('fric_coeff: ', fric_coeff.item())
            print('target fric_coeff: ', target_fric_coeff.item())
            print('fric_coeff grads: ', fric_coeff.grad.item())
        print('==================================')

        if abs((last_loss - loss).item()) < conv_thresh:
            break

        optim.step()
        last_loss = loss

    world, obj = make_world(decoder, latent, force, mass, fric_coeff)
    loss = 0.
    start = time.time()
    for target_pose in target_poses:
        world.step(fixed_dt=True)
        loss += ((target_pose - obj.pos) ** 2).sum()
        print('\r', world.t, '/', time.time() - start, end='')

    if force.requires_grad:
        dist = (force - target_force).norm()
    if fric_coeff.requires_grad:
        dist = (fric_coeff - target_fric_coeff).abs()
    if mass.requires_grad:
        dist = (mass - target_mass).abs()

    ex.log_scalar('loss', loss.item())
    ex.log_scalar('dist', dist.item())
    loss_hist.append(loss.item())
    dist_hist.append(dist.item())
    print('Final loss:', loss.item())
    print('Final dist:', dist.item())

    if _run._id:
        if not os.path.exists(os.path.join(experiment_basedir, 'outputs')):
            os.makedirs(os.path.join(experiment_basedir, 'outputs'))
        output_file = os.path.join(experiment_basedir, 'outputs', _run._id + '.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump({'start_force': start_force,
                         'final_force': force,
                         'target_force': target_force,
                         'start_mass': start_mass,
                         'final_mass': mass,
                         'target_mass': target_mass,
                         'start_fric': start_fric_coeff,
                         'final_fric': fric_coeff,
                         'target_fric': target_fric_coeff,
                         'loss_hist': loss_hist,
                         'force_hist': force_hist,
                         'mass_hist': mass_hist,
                         'fric_hist': fric_hist,
                         'dist_hist': dist_hist,
                         'final_loss': loss,
                         'final_dist': dist,
                         'latent': latent}, f)
        _run.add_artifact(output_file, 'output.pkl')
        writer.close()

    return dist.item()
