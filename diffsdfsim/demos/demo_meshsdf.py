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
import logging
import math
import os
import pickle
import sys
from pathlib import Path

import pyrender
import torch
from matplotlib import pyplot as plt

from sdf_physics.physics3d.bodies import SDFBox, SDFCylinder, SDF3D
from sdf_physics.physics3d.constraints import TotalConstraint3D
from sdf_physics.physics3d.forces import Gravity3D
from sdf_physics.physics3d.utils import get_tensor, Rx, Ry, Recorder3D, Defaults3D, load_igrnet, decode_igr
from sdf_physics.physics3d.world import World3D, run_world

os.environ['PYOPENGL_PLATFORM'] = 'egl'

basedir = Path(__file__).resolve().parent.parent.parent
shapespace_basedir = basedir.joinpath('shapespaces', 'IGR', 'models')
experiment_basedir = basedir.joinpath('experiments', 'bob_spot_pole')

TIME = 1.1

LEGACY = False


def mesh_sdf_demo(scene, output_dir=experiment_basedir):
    decoder, latent = load_igrnet(os.path.join(shapespace_basedir, 'bob_and_spot'))

    latent_init = latent[1].clone()
    latent_init = latent_init.to(Defaults3D.DTYPE)

    world = make_world(decoder, latent_init)[0]
    recorder = Recorder3D(dt=Defaults3D.DT, scene=scene, path=os.path.join(output_dir, 'init'), save_to_disk=True)
    run_world(world, scene=scene, run_time=TIME, recorder=recorder)

    latent_init.requires_grad = True

    if LEGACY:
        lr = 1e-2
    else:
        lr = 5e-3
    regl2 = 0.05
    decreased_by = 1.5
    adjust_lr_every = 500
    if LEGACY:
        optimizer = torch.optim.Adam([latent_init], lr=lr)
    else:
        optimizer = torch.optim.SGD([latent_init], lr=lr)
    max_iter = 200

    logging.info("Starting optimization:")

    best_loss = None
    sigma = None

    trajectory_hist = []
    latent_hist = [latent_init.detach().cpu()]
    loss_hist = []

    for e in range(max_iter):
        optimizer.zero_grad()

        world, sdf_obj = make_world(decoder, latent_init)
        recorder = Recorder3D(dt=Defaults3D.DT, scene=scene,
                              path=os.path.join(output_dir, str(e)), save_to_disk=True) if e % 1 == 0 and e != 0 else None
        scene_rec = scene if e % 1 == 0 and e != 0 else None
        run_world(world, scene=scene_rec, run_time=TIME, recorder=recorder)
        loss = (sdf_obj.pos - get_tensor([0, 0.64, 0])).norm()**2 + regl2 * latent_init.norm()**2
        loss.backward()

        if not LEGACY:
            torch.nn.utils.clip_grad_norm_([latent_init], max_norm=1e1)

        latent_hist.append(latent_init.detach().cpu())
        loss_hist.append(loss.detach().cpu())

        print('iteration: {} / {}'.format(e, max_iter))
        print('loss: {}'.format(loss.item()))
        print('initial latent: ', latent[1].tolist())
        print('target latent: ', latent[0].tolist())
        print('current latent: ', latent_init.tolist())
        print('latent gradient: ', latent_init.grad.tolist())
        print('=======================================')

        optimizer.step()
        if best_loss is not None and (loss - best_loss).abs() < 1e-5:
            break
        best_loss = loss

    world = make_world(decoder, latent_init)[0]
    recorder = Recorder3D(dt=Defaults3D.DT, scene=scene, path=os.path.join(output_dir, 'result'), save_to_disk=True)
    run_world(world, scene=scene, run_time=TIME, recorder=recorder)

    output_data = {'latent_hist': latent_hist,
                   'loss_hist': loss_hist,
                   'trajectory_hist': trajectory_hist}
    with open(os.path.join(output_dir, 'output_data.bin'), 'wb') as f:
        pickle.dump(output_data, f)


def make_world(decoder, latent):
    bodies = []
    joints = []
    friction_coeff = 0.15

    floor = SDFBox([0, -0.5, 0], [50, 1, 50], col=(255, 255, 255), fric_coeff=friction_coeff, restitution=0)
    bodies.append(floor)
    joints.append(TotalConstraint3D(floor))

    b = SDFCylinder([math.pi / 2, 0, 0, 0.35, 1, 0], 0.2, 2, col=(0, 255, 0), fric_coeff=friction_coeff)
    # b = SDFBox([0, 1, 0], [1, 2, 1])
    b.add_no_contact(floor)
    # b.add_force(Gravity3D())
    bodies.append(b)
    joints.append(TotalConstraint3D(b))

    m = SDF3D(pos=[0, 6, 0], scale=2, sdf_func=decode_igr(decoder), params=[latent], col=(0, 0, 255),
              fric_coeff=friction_coeff, restitution=0)
    m.add_force(Gravity3D())
    bodies.append(m)

    return World3D(bodies, joints), m


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-nd':
        # Run without displaying
        scene = None
    else:
        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
        cam = pyrender.PerspectiveCamera(yfov=math.pi / 3, aspectRatio=4 / 3)
        # cam = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=1500)
        cam_pose = get_tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 13],
                               [0, 0, 0, 1]])
        theta = math.pi / 4
        cam_pose = Ry(theta) @ Rx(-theta) @ cam_pose
        scene.add(cam, pose=cam_pose.cpu())
        light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
        light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
        scene.add(light1, pose=(Rx(-theta)).cpu())
        scene.add(light2, pose=(Ry(theta*2) @ Rx(-theta)).cpu())

    mesh_sdf_demo(scene)
