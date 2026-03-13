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

import numpy as np
import pyrender
import sacred
import torch
from matplotlib import pyplot as plt, image as mpimg
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_quaternion
from sacred.utils import apply_backspaces_and_linefeeds
from torch.utils.tensorboard import SummaryWriter

from sdf_physics.physics3d.bodies import SDFBox, SDFSphere, SDFBoxRounded
from sdf_physics.physics3d.constraints import TotalConstraint3D
from sdf_physics.physics3d.forces import Gravity3D
from sdf_physics.physics3d.utils import get_tensor, Defaults3D, Recorder3D, Rx
from sdf_physics.physics3d.world import World3D

os.environ['PYOPENGL_PLATFORM'] = 'egl'

basedir = Path(__file__).resolve().parent.parent.parent.parent
experiment_basedir = basedir.joinpath('experiments', 'trajectory_fitting_pointcloud_real')
ex = sacred.Experiment(name='trajectory_fitting_pointcloud_real')
ex.observers.append(sacred.observers.FileStorageObserver(experiment_basedir))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    run_time = 1.5
    max_iter = 200
    lr = 1e-1
    conv_thresh = 1e-7
    conv_thresh_shape = 1e-5
    optimizer = 'GD'

    mesh_freq = 10

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

    fit_first_frame = True
    fit_trajectory = True

    optimize_init_pose = True

    traj_optimize_pos = True
    traj_optimize_rot = True

    shape = 'sphere'  # 'cube' or 'sphere'

    strict_no_penetration = True

    file_path = 'real_world_data.pkl'


def random_uniform(dim, min, max):
    return torch.rand(dim, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE) * (max - min) + min


def random_normal(dim, mean, std):
    return torch.randn(dim, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE) * std + mean


@ex.capture
def make_world(rad, planes, fric_coeff, restitution, use_toc_diff, use_gravity, custom_mesh, custom_inertia,
               strict_no_penetration, sphere_col=(0, 255, 0), sphere_pose=(1.0, 0, 0, 0, 0, 5, 0),
               sphere_vel=(0, 0, 0, 5, 0, 0), g=10.0, dt=Defaults3D.DT, shape='undefined'):
    bodies = []
    joints = []

    plane_dims = [1.5, 1, 1.5]

    for plane_params in planes:
        normal = plane_params[:3]
        up_dir = get_tensor([0, 1., 0])
        angle = torch.arccos(torch.dot(normal, up_dir) / torch.norm(normal))

        axis = torch.cross(normal, up_dir)
        axis /= axis.norm()

        rot = axis_angle_to_quaternion(-axis * angle)

        pos = -plane_params[3].sign() * normal * (plane_params[3].abs() + plane_dims[1] / 2)

        plane_pose = torch.cat([rot, pos])

        plane = SDFBox(plane_pose, plane_dims, restitution=restitution, fric_coeff=fric_coeff, col=(255, 255, 255),
                       custom_mesh=custom_mesh, custom_inertia=custom_inertia)
        bodies.append(plane)
        joints.append(TotalConstraint3D(plane))

        if len(bodies) > 1:
            for body in bodies[:-1]:
                body.add_no_contact(plane)

    if shape == 'undefined':
        print('shape undefined')
        exit(-1)

    if shape == 'sphere':
        sphere = SDFSphere(sphere_pose, rad, vel=sphere_vel, restitution=restitution, fric_coeff=fric_coeff, col=sphere_col,
                           custom_mesh=custom_mesh, custom_inertia=custom_inertia, mass=0.058, is_transparent=True)

    if shape == 'cube':
        dims = 2*torch.cat( (rad.unsqueeze(0),rad.unsqueeze(0),rad.unsqueeze(0)) )
        sphere = SDFBoxRounded(sphere_pose, dims, r=0.2, vel=sphere_vel, col=sphere_col,
                               restitution=restitution, fric_coeff=fric_coeff)

    if use_gravity:
        sphere.add_force(Gravity3D(g=g))
    bodies.append(sphere)

    world = World3D(bodies, joints, time_of_contact_diff=use_toc_diff, dt=dt,
                    strict_no_penetration=strict_no_penetration)
    return world, sphere


def match_pointcloud(pointcloud, segmentation_mask, sphere, sphere_pos, sphere_rot):
    # determine point cloud loss for sphere segment in SDF shape and pose estimate
    sphere_segmentation_mask = segmentation_mask == 4

    sphere_pts = pointcloud[sphere_segmentation_mask[:]]

    # transform to body frame of estimated sphere
    sphere_pts = ((sphere_rot.T).unsqueeze(0) @ (sphere_pts - sphere_pos).unsqueeze(-1)).squeeze(-1)

    # compare observation with sdf of estimated sphere
    sdf_values, overlap_mask = sphere.query_sdfs(sphere_pts, return_grads=False, return_overlapmask=True)
    sdf_values[overlap_mask == False] = 0

    loss_pc_t = torch.sum(sdf_values ** 2)
    loss_pc_t_samples = torch.sum(overlap_mask == True)

    return loss_pc_t, loss_pc_t_samples


def trajectory_loss(trajectory, sphere, obs):

    loss_pos = 0.
    loss_rot = 0.
    loss_pc = 0.
    loss_pc_samples = 0.

    for i, s in enumerate(trajectory):
        s_pos = s[-3:]
        s_rot = quaternion_to_matrix(s[-7:-3])

        obs_pointcloud = obs['pcs'][i]
        obs_segmentation_mask = obs['segs'][i]

        loss_pc_t, loss_pc_t_samples = match_pointcloud(obs_pointcloud, obs_segmentation_mask,
                                                        sphere, s_pos, s_rot)

        loss_pc += loss_pc_t
        loss_pc_samples += loss_pc_t_samples

    loss = loss_pc
    loss /= torch.maximum(get_tensor(1.0), loss_pc_samples)

    loss_pos /= len(trajectory)
    loss_rot /= len(trajectory)

    return loss, loss_pc, loss_pos, loss_rot


def run_world_fixed_dt(world, pcs, detach_2nd_bounce=False):
    num_contact_steps = 0
    trajectory = [world.bodies[-1].p]

    frame_id = 1
    while frame_id < pcs.shape[0] - 1:
        had_contacts = world.step(fixed_dt=True)
        print('\r', world.t, end='')
        if detach_2nd_bounce and had_contacts:
            num_contact_steps += 1

        if detach_2nd_bounce and had_contacts and num_contact_steps > 1:
            world.undo_step()
            frame_id -= 1
            trajectory.pop()
            world.v = world.v.detach().clone()
            world.set_v(world.v)
            world.set_p(torch.cat([b.p.detach().clone() for b in world.bodies]))
            num_contact_steps = 0

        frame_id += 1

        trajectory.append(world.bodies[-1].p)
    print('\n')
    return trajectory


@ex.command(unobserved=True)
def record_results(run_dir, output_dir, shape, file_path):
    with open(os.path.join(run_dir, 'output_first_frame.pkl'), 'rb') as f:
        data_first_frame = pickle.load(f)

    with open(os.path.join(run_dir, 'output_trajectory.pkl'), 'rb') as f:
        data_trajectory = pickle.load(f)

    with open(basedir / file_path, 'rb') as f:
        obs = pickle.load(f)
    for k, v in obs.items():
        if isinstance(v, list) and len(v) > 0:
            obs[k] = torch.from_numpy(np.stack(v)).to(dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)

    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    cam = pyrender.PerspectiveCamera(yfov=math.pi / 3)
    cam_pose = get_tensor([[1, 0, 0, 0.3],
                           [0, 1, 0, 0.1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    gamma = math.pi / 4
    theta = math.pi / 5
    cam_pose = Rx(-theta) @ cam_pose
    scene.add(cam, pose=cam_pose.cpu())
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
    scene.add(light)

    def record_world(world, recorder):
        for node in scene.mesh_nodes.copy():
            scene.remove_node(node)

        for body in world.bodies:
            body.render(scene)

        pc = pyrender.Mesh.from_points(obs['pcs'][0][obs['segs'][0] == 4].cpu(), colors=[1., 0, 0.])
        scene.add(pc)
        recorder.record(world.t)

        for t in range(obs['pcs'].shape[0] - 1):
            world.step(fixed_dt=True)
            for node in scene.mesh_nodes.copy():
                scene.remove_node(node)

            for body in world.bodies:
                body.render(scene)

            pc = pyrender.Mesh.from_points(obs['pcs'][t + 1][obs['segs'][t+1] == 4].cpu(), colors=[1, 0, 0.])
            scene.add(pc)

            recorder.record(world.t)

    planes = obs['planes'].mean(dim=0)
    g = obs['grav_dirs'].mean(dim=0).norm()

    fric_coeff = get_tensor(0.15)
    restitution = get_tensor(0.7)

    init_pos = data_first_frame['final_pose'][-3:]
    init_pos1 = data_first_frame['init_pose1'][-3:]
    init_vel = ((init_pos1 - init_pos) / Defaults3D.DT).detach().clone()
    init_vel += get_tensor([0, 1, 0]) * g * Defaults3D.DT

    world_frame_fit = make_world(data_first_frame['final_rad'], planes, fric_coeff, restitution,
                                 sphere_pose=data_first_frame['final_pose'], sphere_vel=init_vel,
                                 shape=shape, sphere_col=(0, 0, 255), g=g)[0]
    recorder = Recorder3D(0, scene, os.path.join(output_dir, 'first_frame_fit'), save_to_disk=True)
    record_world(world_frame_fit, recorder)

    world_final = make_world(data_trajectory['final_rad'], planes, data_trajectory['friction'],
                             data_trajectory['restitution'], sphere_pose=data_trajectory['final_pose'],
                             sphere_vel=data_trajectory['init_vel'], shape=shape, sphere_col=(0, 255, 0))[0]
    recorder = Recorder3D(0, scene, os.path.join(output_dir, 'trajectory_fit'), save_to_disk=True)
    record_world(world_final, recorder)

    for i, frame in enumerate(obs['rgbs'][:10]):
        os.makedirs(os.path.join(output_dir, 'rgbs'), exist_ok=True)
        mpimg.imsave(os.path.join(output_dir, 'rgbs', 'color_{}.png'.format(i)), frame.cpu().numpy().astype(np.uint8))


@ex.automain
def optim_sphere_trajectory(lr, max_iter, optimizer, detach_2nd_bounce, _run, seed, optimize_init_pose,
                            traj_optimize_rot, traj_optimize_pos, fit_first_frame, fit_trajectory, conv_thresh,
                            conv_thresh_shape, shape, file_path):
    if _run._id:
        writer = SummaryWriter(os.path.join(experiment_basedir, str(_run._id)))
    
    with open(basedir / file_path, 'rb') as f:
        obs = pickle.load(f)
    _run.add_resource(basedir / file_path)

    for k, v in obs.items():
        if isinstance(v, list) and len(v) > 0:
            obs[k] = torch.from_numpy(np.stack(v)).to(dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)

    run_time = obs['rgbs'].shape[0] * Defaults3D.DT
    g = obs['grav_dirs'].mean(dim=0).norm()

    planes = obs['planes'].mean(dim=0)

    diam0 = (obs['pcs'][0][obs['segs'][0] == 4].max(dim=0)[0] - obs['pcs'][0][obs['segs'][0] == 4].min(dim=0)[0]).max()

    start_pos0 = obs['pcs'][0][obs['segs'][0] == 4].mean(dim=0)
    start_pos0 += start_pos0 / start_pos0.norm() * diam0 / 2
    start_pos1 = obs['pcs'][1][obs['segs'][1] == 4].mean(dim=0)
    start_pos1 += start_pos1 / start_pos1.norm() * diam0 / 2

    fric_coeff = get_tensor(0.15)
    restitution = get_tensor(0.7)

    start_rad = diam0 / 2
    rad = start_rad.clone()
    optimized_variables = [rad]
    rad.requires_grad = True

    if optimize_init_pose:
        init_rot = torch.tensor([1.0, 0, 0, 0], device=Defaults3D.DEVICE, dtype=Defaults3D.DTYPE)
        init_rot = init_rot / torch.linalg.norm(init_rot)
        init_pos = start_pos0.detach().clone()

        init_rot1 = torch.tensor([1.0, 0, 0, 0], device=Defaults3D.DEVICE, dtype=Defaults3D.DTYPE)
        init_rot1 = init_rot1 / torch.linalg.norm(init_rot1)
        init_pos1 = start_pos1.detach().clone()

        if traj_optimize_rot:
            optimized_variables.append(init_rot)
            optimized_variables.append(init_rot1)
            init_rot.requires_grad = True
            init_rot1.requires_grad = True
        if traj_optimize_pos:
            optimized_variables.append(init_pos)
            optimized_variables.append(init_pos1)
            init_pos.requires_grad = True
            init_pos1.requires_grad = True

        init_pose = torch.cat((init_rot, init_pos))
        init_pose1 = torch.cat((init_rot1, init_pos1))


    # fit first frame
    if fit_first_frame:
        loss_hist = []
        dist_hist = []
        rad_hist = []
        init_pose_hist = []
        pos_err_hist = []
        rot_err_hist = []
        print('fit first frame')

        if optimizer == 'GD':
            optim = torch.optim.SGD(optimized_variables, lr=lr)
        elif optimizer == 'Adam':
            optim = torch.optim.Adam(optimized_variables, lr=lr)

        last_loss = 1e10
        last_rad = 1e10
        for e in range(max_iter):
            print(e, '/', max_iter)
            optim.zero_grad()
            init_pose = torch.cat((init_rot, init_pos))
            init_pose1 = torch.cat((init_rot1, init_pos1))
            world, sphere = make_world(rad, planes, fric_coeff, restitution, sphere_pose=init_pose, shape=shape)
            world1, sphere1 = make_world(rad, planes, fric_coeff, restitution, sphere_pose=init_pose1, shape=shape)

            loss_pc = 0.
            loss_pc_samples = 0.

            s_pos = sphere.p[-3:]
            s_rot = quaternion_to_matrix(sphere.p[-7:-3])

            obs_pointcloud = obs['pcs'][0]
            obs_segmentation_mask = obs['segs'][0]

            loss_pc_t, loss_pc_t_samples = match_pointcloud(obs_pointcloud, obs_segmentation_mask, sphere, s_pos, s_rot)

            loss_pc += loss_pc_t
            loss_pc_samples += loss_pc_t_samples

            # Fit to second frame also for init vel
            s_pos1 = sphere1.p[-3:]
            s_rot1 = quaternion_to_matrix(sphere1.p[-7:-3])

            obs_pointcloud1 = obs['pcs'][1]
            obs_segmentation_mask1 = obs['segs'][1]

            loss_pc_t1, loss_pc_t_samples1 = match_pointcloud(obs_pointcloud1, obs_segmentation_mask1, sphere1, s_pos1, s_rot1)

            loss_pc += loss_pc_t1
            loss_pc_samples += loss_pc_t_samples1

            loss = loss_pc
            loss /= torch.maximum(get_tensor(1.0), loss_pc_samples)

            loss.backward()

            loss_hist.append(loss.item())
            rad_hist.append(rad.item())
            init_pose_hist.append(init_pose.tolist())

            ex.log_scalar('frame_fit_loss', loss.item())

            if _run._id:
                writer.add_scalar('frame_fit_objective', loss, e)

            print('loss:', loss.item())
            print('start rad: ', start_rad.item())
            print('rad: ', rad.item())
            print('rad grad: ', rad.grad.item())
            print('init_pose: ', init_pose)
            print('init pos grad: ', init_pos.grad)
            print('init rot grad: ', init_rot.grad)
            print('==================================')

            if abs((last_loss - loss).item()) < conv_thresh and abs((last_rad - rad).item()) < conv_thresh_shape:
                break

            last_rad = rad.clone().detach()
            optim.step()
            init_rot.data = init_rot.data / torch.linalg.norm(init_rot.data)
            last_loss = loss

        if _run._id:
            output_dir = os.path.join(experiment_basedir, 'outputs')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, str(_run._id) + '_first_frame.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump({'start_rad': start_rad,
                             'final_rad': rad,
                             'final_pose': init_pose,
                             'init_pose1': init_pose1,
                             'loss_hist': loss_hist,
                             'rad_hist': rad_hist,
                             'dist_hist': dist_hist,
                             'pos_err_hist': pos_err_hist,
                             'rot_err_hist': rot_err_hist,
                             'init_pose_hist': init_pose_hist,
                             'final_loss': loss}, f)
            _run.add_artifact(output_file, 'output_first_frame.pkl')
            writer.close()

    # fit trajectory
    if fit_trajectory:
        print('trajectory fitting started')

        init_rot1.requires_grad = False
        init_pos1.requires_grad = False

        init_vel = ((init_pos1 - init_pos) / Defaults3D.DT).detach().clone()
        init_vel += get_tensor([0, 1, 0]) * g * Defaults3D.DT

        init_vel = torch.cat([init_vel.new_zeros(3), init_vel])
        init_vel.requires_grad = True

        fric_coeff.requires_grad = True
        restitution.requires_grad = True

        optimized_variables = [rad, init_rot, init_pos, init_vel, fric_coeff, restitution]

        loss_hist = []
        dist_hist = []
        rad_hist = []
        pos_rmse_hist = []
        init_pos_err_hist = []
        rot_rmse_hist = []
        init_rot_err_hist = []
        init_pose_hist = []

        if optimizer == 'GD':
            optim = torch.optim.SGD(optimized_variables, lr=lr)
        elif optimizer == 'Adam':
            optim = torch.optim.Adam(optimized_variables, lr=lr)

        pc_centrs = torch.stack([pc[seg == 4].mean(dim=0) for pc, seg in zip(obs['pcs'], obs['segs'])])
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(pc_centrs[:, 0].cpu())
        ax[1].plot(pc_centrs[:, 1].cpu())
        ax[2].plot(pc_centrs[:, 2].cpu())

        last_loss = 1e10
        last_rad = 1e10
        for e in range(max_iter):
            print(e, '/', max_iter)
            optim.zero_grad()
            init_pose = torch.cat((init_rot, init_pos))
            world, sphere = make_world(rad, planes, fric_coeff, restitution, g=g, sphere_vel=init_vel,
                                       sphere_pose=init_pose, shape=shape)

            traj = run_world_fixed_dt(world, obs['pcs'], detach_2nd_bounce=detach_2nd_bounce)

            poss = torch.stack(traj)[:, -3:]
            ax[0].plot(poss[:, 0].detach().cpu())
            ax[1].plot(poss[:, 1].detach().cpu())
            ax[2].plot(poss[:, 2].detach().cpu())

            loss, loss_pc, loss_pos_tr, loss_rot_tr = trajectory_loss(traj, sphere, obs)

            loss.backward()

            loss_hist.append(loss.item())
            rad_hist.append(rad.item())

            init_pose_hist.append(init_pose.tolist())

            ex.log_scalar('trajectory_fit_loss', loss.item())

            if _run._id:
                writer.add_scalar('trajectory_objective', loss, e)

            print('loss:', loss.item())
            print('start rad: ', start_rad.item())
            print('rad: ', rad.item())
            print('rad grad: ', rad.grad.item())
            print('init_pose: ', init_pose)
            print('init pos grad: ', init_pos.grad)
            print('init rot grad: ', init_rot.grad)
            print('init_vel:', init_vel)
            print('init_vel grad:', init_vel.grad)
            print('friction:', fric_coeff)
            print('friction grad:', fric_coeff.grad)
            print('restitution:', restitution)
            print('restitution grad:', restitution.grad)
            print('==================================')

            if abs((last_loss - loss).item()) < conv_thresh and abs((last_rad - rad).item()) < conv_thresh_shape:
                break

            last_rad = rad.clone().detach()
            optim.step()
            init_rot.data = init_rot.data / torch.linalg.norm(init_rot.data)
            last_loss = loss

        plt.show()

        if _run._id:
            output_dir = os.path.join(experiment_basedir, 'outputs')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, str(_run._id) + '_trajectory.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump({'start_rad': start_rad,
                             'final_rad': rad,
                             'final_pose': init_pose,
                             'init_vel': init_vel,
                             'friction': fric_coeff,
                             'restitution': restitution,
                             'loss_hist': loss_hist,
                             'rad_hist': rad_hist,
                             'dist_hist': dist_hist,
                             'pos_rmse_hist': pos_rmse_hist,
                             'rot_rmse_hist': rot_rmse_hist,
                             'init_pos_err_hist': init_pos_err_hist,
                             'init_rot_err_hist': init_rot_err_hist,
                             'init_pose_hist': init_pose_hist,
                             'final_loss': loss}, f)
            _run.add_artifact(output_file, 'output_trajectory.pkl')
            writer.close()
