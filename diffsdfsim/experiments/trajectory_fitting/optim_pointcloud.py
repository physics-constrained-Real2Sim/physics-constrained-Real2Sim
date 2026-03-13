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
import os
import pickle
from pathlib import Path

import torch
from pytorch3d.loss import chamfer_distance
import sacred
from sacred.utils import apply_backspaces_and_linefeeds
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.transforms import quaternion_to_matrix, so3_relative_angle

from sdf_physics.physics3d.bodies import SDFBox, SDFSphere, SDFBoxRounded
from sdf_physics.physics3d.constraints import TotalConstraint3D
from sdf_physics.physics3d.forces import Gravity3D
from sdf_physics.physics3d.utils import get_tensor, Defaults3D, Recorder3D, Rx, Ry
from sdf_physics.physics3d.world import World3D, run_world

import pyrender

import math
import numpy as np
import scipy as sp

os.environ['PYOPENGL_PLATFORM'] = 'egl'

basedir = Path(__file__).resolve().parent.parent.parent.parent
experiment_basedir = basedir.joinpath('experiments', 'trajectory_fitting_pointcloud')
ex = sacred.Experiment(name='trajectory_fitting_pointcloud')
ex.observers.append(sacred.observers.FileStorageObserver(experiment_basedir))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    run_time = 1.5
    max_iter = 200
    lr = 1e-1
    conv_thresh = 1e-5
    conv_thresh_shape = 1e-3
    optimizer = 'GD'

    mesh_freq = 10

    min_dim = 0.5
    max_dim = 2.5

    min_target_dim = 0.5
    max_target_dim = 1.5

    min_diff = 0.
    max_diff = 1.0

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

    camera_fx = 515
    camera_fy = 515
    camera_cx = 319.5
    camera_cy = 239.5

    fit_first_frame = True
    fit_trajectory = True

    optimize_init_pose = True

    traj_optimize_pos = True
    traj_optimize_rot = True

    init_pos_std = 0.1
    init_rot_std = 0.1 # for axis angle representation

    depth_noise_factor = 0.0001 # 0.0001

    shape = 'cube'  # 'cube' or 'sphere'

    strict_no_penetration = True


def random_uniform(dim, min, max):
    return torch.rand(dim, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE) * (max - min) + min


def random_normal(dim, mean, std):
    return torch.randn(dim, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE) * std + mean


@ex.capture
def make_world(rad, use_toc_diff, use_friction, use_wall, use_floor, use_gravity, custom_mesh, custom_inertia,
               strict_no_penetration, sphere_col=(0, 255, 0), sphere_pose=(1.0, 0, 0, 0, 0, 5, 0),
               sphere_vel=(0, 0, 0, 5, 0, 0), dt=Defaults3D.DT, shape='undefined'):
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

    if shape == 'undefined':
        print('shape undefined')
        exit(-1)

    if shape == 'sphere':
        sphere = SDFSphere(sphere_pose, rad, vel=sphere_vel, restitution=restitution, fric_coeff=fric_coeff, col=sphere_col,
                           custom_mesh=custom_mesh, custom_inertia=custom_inertia)

    if shape == 'cube':
        dims = 2*torch.cat( (rad.unsqueeze(0),rad.unsqueeze(0),rad.unsqueeze(0)) )
        sphere = SDFBoxRounded(sphere_pose, dims, r=0.2, vel=sphere_vel, col=sphere_col,
                               restitution=restitution, fric_coeff=fric_coeff)

    if use_gravity:
        sphere.add_force(Gravity3D())
    bodies.append(sphere)

    world = World3D(bodies, joints, time_of_contact_diff=use_toc_diff, dt=dt,
                    strict_no_penetration=strict_no_penetration)
    return world, sphere


def match_pointcloud(pointcloud, segmentation_mask, cam_pose, sphere, sphere_pos, sphere_rot):
    # determine point cloud loss for sphere segment in SDF shape and pose estimate
    sphere_pts = np.where(
        (segmentation_mask[:, :, 0] == sphere.col[0]) & (segmentation_mask[:, :, 1] == sphere.col[1]) & (
                segmentation_mask[:, :, 2] == sphere.col[2]))
    sphere_segmentation_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1]), dtype=bool)
    sphere_segmentation_mask[sphere_pts[0], sphere_pts[1]] = True

    # erode segmentation mask by one pixel
    sphere_segmentation_mask = sp.ndimage.binary_erosion(sphere_segmentation_mask)

    sphere_pts = np.where(sphere_segmentation_mask[:, :] == True)

    sphere_pts = pointcloud[sphere_pts[0], sphere_pts[1], :]
    sphere_pts = sphere_pts[np.linalg.norm(sphere_pts, axis=1) > 0]
    sphere_pts = np.concatenate((sphere_pts, np.ones((sphere_pts.shape[0], 1), dtype=sphere_pts.dtype)), axis=1)

    # transform to world frame
    # OpenGL camera frame convention!
    sphere_pts[:, 1] = -sphere_pts[:, 1]
    sphere_pts[:, 2] = -sphere_pts[:, 2]
    sphere_pts = np.matmul(np.expand_dims(cam_pose, axis=0), np.expand_dims(sphere_pts, axis=-1)).squeeze(-1)

    sphere_pts = torch.tensor(sphere_pts[:, :3], device=sphere_pos.device, dtype=sphere_pos.dtype)

    # transform to body frame of estimated sphere
    sphere_pts = ((sphere_rot.T).unsqueeze(0) @ (sphere_pts - sphere_pos).unsqueeze(-1)).squeeze(-1)

    # compare observation with sdf of estimated sphere
    sdf_values, overlap_mask = sphere.query_sdfs(sphere_pts, return_grads=False, return_overlapmask=True)
    sdf_values[overlap_mask == False] = 0

    loss_pc_t = torch.sum(sdf_values ** 2)
    loss_pc_t_samples = torch.sum(overlap_mask == True)

    return loss_pc_t, loss_pc_t_samples


def trajectory_loss(world, sphere, world_target, world_target_rec):

    loss_x = 0.
    loss_y = 0.
    loss_z = 0.
    loss_v = 0.
    loss_w = 0.
    loss_pos = 0.
    loss_rot = 0.
    loss_pc = 0.
    loss_pc_samples = 0.

    last_j = 0
    last_k = 0
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

        min_diff = 1e100
        last_diff = 1e100
        min_obs = None
        new_k = 0
        for k, obs in enumerate(world_target_rec.observations[last_k:]):

            diff = abs(s[0] - obs[0])
            if diff <= min_diff and diff <= 1e-5:
                min_diff = diff
                min_obs = obs
                new_k = last_k + k

            if diff > last_diff:
                break  # ascending time order
            last_diff = diff

        s_pos = s[1][-3:]
        s_rot = quaternion_to_matrix(s[1][-7:-3])
        s_vel = s[2]

        s_target_pos = min_s_target[1][-3:]
        s_target_rot = quaternion_to_matrix(min_s_target[1][-7:-3])
        s_target_vel = min_s_target[2]

        loss_x += ((s_pos[0] - s_target_pos[0]) ** 2).sum()
        loss_y += ((s_pos[1] - s_target_pos[1]) ** 2).sum()
        loss_z += ((s_pos[2] - s_target_pos[2]) ** 2).sum()
        loss_v += ((s_vel[3:] - s_target_vel[3:]) ** 2).sum()
        loss_w += ((s_vel[0:3] - s_target_vel[0:3]) ** 2).sum()

        loss_pos += ((s_pos - s_target_pos) ** 2).sum()
        loss_rot += so3_relative_angle( s_rot.unsqueeze(0), s_target_rot.unsqueeze(0) ) ** 2

        if min_obs:
            obs_pointcloud = min_obs[3]
            obs_segmentation_mask = min_obs[4]
            cam_pose = min_obs[5][0]

            loss_pc_t, loss_pc_t_samples = match_pointcloud(obs_pointcloud, obs_segmentation_mask,
                                                            cam_pose, sphere, s_pos, s_rot)

            loss_pc += loss_pc_t
            loss_pc_samples += loss_pc_t_samples

        last_j = new_j
        last_k = new_k

    loss = loss_pc
    loss /= torch.maximum(get_tensor(1.0), loss_pc_samples)

    loss_pos /= len(world.trajectory)
    loss_rot /= len(world.trajectory)

    return loss, loss_pc, loss_pos, loss_rot


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
def record_results(run_dir, output_dir, run_time, camera_fx, camera_fy, camera_cx, camera_cy, shape, depth_noise_factor):
    with open(os.path.join(run_dir, 'output_first_frame.pkl'), 'rb') as f:
        data_first_frame = pickle.load(f)

    with open(os.path.join(run_dir, 'output_trajectory.pkl'), 'rb') as f:
        data_trajectory = pickle.load(f)

    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    # cam = pyrender.PerspectiveCamera(yfov=math.pi / 3, aspectRatio=10 / 6)
    # cam = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=1500)
    cam = pyrender.IntrinsicsCamera(fx=camera_fx, fy=camera_fy, cx=camera_cx, cy=camera_cy, znear=0.1, zfar=100.0)
    cam_pose = get_tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 20],
                           [0, 0, 0, 1]])

    gamma = math.pi / 4
    theta = math.pi / 4
    cam_pose = Ry(gamma) @ Rx(-theta) @ cam_pose
    scene.add(cam, pose=cam_pose.cpu())
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
    scene.add(light, pose=(Ry(math.pi / 2) @ Rx(-math.pi / 4)).cpu())

    world_target_rec = make_world(data_first_frame['target_rad'], sphere_pose=data_first_frame['target_init_pose'],
                                  shape=shape, sphere_col=(0, 0, 255))[0]
    recorder = Recorder3D(0, scene, os.path.join(output_dir, 'observations'), record_points=True, record_seg=True,
                          noise_factor=depth_noise_factor, save_to_disk=True)
    run_world(world_target_rec, fixed_dt=True, scene=scene, run_time=run_time, recorder=recorder)

    world_init = make_world(data_first_frame['start_rad'], sphere_pose=data_first_frame['init_pose_hist'][0],
                            shape=shape, sphere_col=(255, 0, 0))[0]
    recorder = Recorder3D(0, scene, os.path.join(output_dir, 'init'), save_to_disk=True)
    run_world(world_init, fixed_dt=True, scene=scene, run_time=run_time, recorder=recorder)

    world_frame_fit = make_world(data_first_frame['final_rad'], sphere_pose=data_first_frame['final_pose'],
                            shape=shape, sphere_col=(255, 255, 255))[0]
    recorder = Recorder3D(0, scene, os.path.join(output_dir, 'first_frame_fit'), save_to_disk=True)
    run_world(world_frame_fit, fixed_dt=True, scene=scene, run_time=run_time, recorder=recorder)

    world_final = make_world(data_trajectory['final_rad'], sphere_pose=data_trajectory['final_pose'],
                             shape=shape, sphere_col=(0, 255, 0))[0]
    recorder = Recorder3D(0, scene, os.path.join(output_dir, 'trajectory_fit'), save_to_disk=True)
    run_world(world_final, fixed_dt=True, scene=scene, run_time=run_time, recorder=recorder)


@ex.automain
def optim_sphere_trajectory(min_dim, max_dim, run_time, lr, max_iter, optimizer, detach_2nd_bounce, target_dt, _run,
                            seed, optimize_init_pose, camera_fx, camera_fy, camera_cx, camera_cy, depth_noise_factor,
                            fit_first_frame, fit_trajectory, traj_optimize_rot, traj_optimize_pos,
                            init_rot_std, init_pos_std, conv_thresh, conv_thresh_shape, shape, min_diff, max_diff,
                            min_target_dim, max_target_dim):
    if _run._id:
        writer = SummaryWriter(os.path.join(experiment_basedir, str(_run._id)))
    
    target_rad = random_uniform(1, min_target_dim, max_target_dim).squeeze()

    target_init_rot = torch.tensor([1.0, 0, 0, 0], device=Defaults3D.DEVICE, dtype=Defaults3D.DTYPE)
    target_init_rot = target_init_rot + random_normal(4,0,init_rot_std)
    target_init_rot = target_init_rot / torch.linalg.norm(target_init_rot)
    target_init_pos = torch.tensor([0, 5.0, 0], device=Defaults3D.DEVICE, dtype=Defaults3D.DTYPE)
    target_init_pos = target_init_pos + random_normal(3,0,init_pos_std)
    target_init_pose = torch.cat((target_init_rot, target_init_pos))

    start_rad = target_rad + random_uniform(1, min_diff, max_diff).squeeze()
    #start_rad = get_tensor(1.0)
    rad = start_rad.clone()
    optimized_variables = [rad]
    rad.requires_grad = True

    if optimize_init_pose:
        init_rot = target_init_rot.detach().clone()
        if traj_optimize_rot:
            init_rot = init_rot + random_normal(4, 0, init_rot_std)
        init_rot = init_rot / torch.linalg.norm(init_rot)
        init_pos = target_init_pos.detach().clone()
        if traj_optimize_pos:
            init_pos = init_pos + random_normal(3, 0, init_pos_std)

        if traj_optimize_rot:
            optimized_variables.append(init_rot)
            init_rot.requires_grad = True
        if traj_optimize_pos:
            optimized_variables.append(init_pos)
            init_pos.requires_grad = True


        init_pose = torch.cat((init_rot, init_pos))

        start_init_pose = init_pose.clone()
    else:
        init_pose = target_init_pose.clone().detach()
        start_init_pose = init_pose.clone()


    # record target scene observations
    scene = pyrender.Scene(ambient_light=[1., 1., 1.], )
    cam = pyrender.IntrinsicsCamera( fx=camera_fx, fy=camera_fy, cx=camera_cx, cy=camera_cy, znear=0.1, zfar=100.0 )
    cam_pose = get_tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 20],
                           [0, 0, 0, 1]])

    gamma = math.pi / 4
    theta = math.pi / 4
    cam_pose = Ry(gamma) @ Rx(-theta) @ cam_pose
    scene.add(cam, pose=cam_pose.cpu())
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1e1)
    scene.add(light, pose=(Ry(math.pi / 2) @ Rx(-math.pi / 4)).cpu())

    world_target_rec, sphere_target_rec = make_world(target_rad, sphere_pose=target_init_pose, shape=shape)
    recorder = Recorder3D(0, scene, '/tmp/test', record_points=True, record_seg=True, noise_factor=depth_noise_factor,
                          save_to_disk=False)
    run_world(world_target_rec, fixed_dt=True, scene=scene, run_time=run_time, recorder=recorder)

    world_target, sphere_target = make_world(target_rad, dt=target_dt, sphere_pose=target_init_pose, shape=shape)
    run_world_fixed_dt(world_target, run_time)

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
            world, sphere = make_world(rad, sphere_pose=init_pose, shape=shape)

            loss_pc = 0.
            loss_pc_samples = 0.

            obs = world_target_rec.observations[0]

            s_pos = sphere.p[-3:]
            s_rot = quaternion_to_matrix(sphere.p[-7:-3])

            obs_pointcloud = obs[3]
            obs_segmentation_mask = obs[4]
            cam_pose = obs[5][0]

            loss_pc_t, loss_pc_t_samples = match_pointcloud(obs_pointcloud, obs_segmentation_mask,
                                                            cam_pose, sphere, s_pos, s_rot)

            loss_pc += loss_pc_t
            loss_pc_samples += loss_pc_t_samples

            loss = loss_pc
            # loss /= len(world.trajectory)
            loss /= torch.maximum(get_tensor(1.0), loss_pc_samples)

            loss_pos = ((init_pos - target_init_pos) ** 2).sum()
            loss_rot = so3_relative_angle(quaternion_to_matrix(init_rot).unsqueeze(0),
                                          quaternion_to_matrix(target_init_rot).unsqueeze(0)) ** 2
            dist = chamfer_distance(sphere.verts.unsqueeze(0), sphere_target.verts.unsqueeze(0))[0]
            #loss = dist

            loss.backward()

            loss_hist.append(loss.item())
            dist_hist.append(dist.item())
            rad_hist.append(rad.item())
            init_pose_hist.append(init_pose.tolist())
            pos_err_hist.append(loss_pos.item())
            rot_err_hist.append(loss_rot.item())

            ex.log_scalar('frame_fit_chamfer_dist', dist.item())
            ex.log_scalar('frame_fit_loss', loss.item())
            ex.log_scalar('frame_fit_pos_err', loss_pos.item())
            ex.log_scalar('frame_fit_rot_err', loss_rot.item())

            if _run._id:
                writer.add_scalar('frame_fit_objective', loss, e)
                writer.add_scalar('frame_fit_chamfer_dist', dist, e)
                writer.add_scalar('frame_fit_pos_err', loss_pos, e)
                writer.add_scalar('frame_fit_rot_err', loss_rot, e)
                writer.add_scalars('frame_fit_rad_{}'.format(seed),
                                   {'rad': rad, 'target': target_rad}, e)

            print('chamfer dist:', dist.item())
            print('loss:', loss.item())
            print('start rad: ', start_rad.item())
            print('rad: ', rad.item())
            print('target rad: ', target_rad.item())
            print('rad grad: ', rad.grad.item())
            print('start init_pose: ', start_init_pose)
            print('init_pose: ', init_pose)
            print('target init_pose: ', target_init_pose)
            print('init pos grad: ', init_pos.grad)
            print('init rot grad: ', init_rot.grad)
            print('==================================')

            if abs((last_loss - loss).item()) < conv_thresh and abs((last_rad - rad).item()) < conv_thresh_shape:
                break

            last_rad = rad.clone().detach()
            optim.step()
            rad.data[rad < min_dim] = min_dim
            rad.data[rad > max_dim] = max_dim
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
                             'target_rad': target_rad,
                             'final_pose': init_pose,
                             'target_init_pose': target_init_pose,
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

        last_loss = 1e10
        last_rad = 1e10
        for e in range(max_iter):
            print(e, '/', max_iter)
            optim.zero_grad()
            init_pose = torch.cat((init_rot, init_pos))
            world, sphere = make_world(rad, sphere_pose=init_pose, shape=shape)

            run_world_fixed_dt(world, run_time, detach_2nd_bounce=detach_2nd_bounce)

            loss, loss_pc, loss_pos_tr, loss_rot_tr = trajectory_loss(world, sphere, world_target, world_target_rec)
            loss_pos_tr = loss_pos_tr.sqrt()
            loss_rot_tr = loss_rot_tr.sqrt()

            loss_pos = ((init_pos - target_init_pos) ** 2).sum()
            loss_rot = so3_relative_angle(quaternion_to_matrix(init_rot).unsqueeze(0),
                                          quaternion_to_matrix(target_init_rot).unsqueeze(0)) ** 2

            dist = chamfer_distance(sphere.verts.unsqueeze(0), sphere_target.verts.unsqueeze(0))[0]
            #loss = dist

            loss.backward()

            loss_hist.append(loss.item())
            dist_hist.append(dist.item())
            rad_hist.append(rad.item())
            pos_rmse_hist.append(loss_pos_tr.item())
            rot_rmse_hist.append(loss_rot_tr.item())
            init_pos_err_hist.append(loss_pos.item())
            init_rot_err_hist.append(loss_rot.item())

            init_pose_hist.append(init_pose.tolist())

            ex.log_scalar('trajectory_fit_chamfer_dist', dist.item())
            ex.log_scalar('trajectory_fit_loss', loss.item())
            ex.log_scalar('trajectory_fit_pos_rmse', loss_pos_tr.item())
            ex.log_scalar('trajectory_fit_rot_rmse', loss_rot_tr.item())
            ex.log_scalar('trajectory_fit_pos_err', loss_pos.item())
            ex.log_scalar('trajectory_fit_rot_err', loss_rot.item())

            if _run._id:
                writer.add_scalar('trajectory_objective', loss, e)
                writer.add_scalar('trajectory_chamfer_dist', dist, e)
                writer.add_scalar('trajectory_fit_pos_rmse', loss_pos_tr, e)
                writer.add_scalar('trajectory_fit_rot_rmse', loss_rot_tr, e)
                writer.add_scalar('trajectory_fit_pos_err', loss_pos, e)
                writer.add_scalar('trajectory_fit_rot_err', loss_rot, e)
                writer.add_scalars('trajectory_rad_{}'.format(seed),
                                   {'rad': rad, 'target': target_rad}, e)

            print('chamfer dist:', dist.item())
            print('loss:', loss.item())
            print('start rad: ', start_rad.item())
            print('rad: ', rad.item())
            print('target rad: ', target_rad.item())
            print('rad grad: ', rad.grad.item())
            print('start init_pose: ', start_init_pose)
            print('init_pose: ', init_pose)
            print('target init_pose: ', target_init_pose)
            print('init pos grad: ', init_pos.grad)
            print('init rot grad: ', init_rot.grad)
            print('==================================')

            if abs((last_loss - loss).item()) < conv_thresh and abs((last_rad - rad).item()) < conv_thresh_shape:
                break

            last_rad = rad.clone().detach()
            optim.step()
            rad.data[rad < min_dim] = min_dim
            rad.data[rad > max_dim] = max_dim
            init_rot.data = init_rot.data / torch.linalg.norm(init_rot.data)
            last_loss = loss

        if _run._id:
            output_dir = os.path.join(experiment_basedir, 'outputs')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, str(_run._id) + '_trajectory.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump({'start_rad': start_rad,
                             'final_rad': rad,
                             'target_rad': target_rad,
                             'final_pose': init_pose,
                             'target_init_pose': target_init_pose,
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

    init_pose = torch.cat((init_rot, init_pos))
    world, sphere = make_world(rad, sphere_pose=init_pose, shape=shape)
    dist = chamfer_distance(sphere.verts.unsqueeze(0), sphere_target.verts.unsqueeze(0))[0]
    run_world_fixed_dt(world, run_time, detach_2nd_bounce=detach_2nd_bounce)
    loss, loss_pc, loss_pos, loss_rot = trajectory_loss(world, sphere, world_target, world_target_rec)
    ex.log_scalar('chamfer_dist', dist)
    ex.log_scalar('loss', loss)
    dist_hist.append(dist.item())
    loss_hist.append(loss.item())
    print('Final loss:', loss.item())
    print('Final chamfer dist:', dist.item())
