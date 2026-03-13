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
import importlib.util
import math
import os
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pyrender
import torch
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyhocon import ConfigFactory
from pytorch3d.transforms import axis_angle_to_matrix

from diffsdfsim.lcp_physics.physics import Defaults
from diffsdfsim.lcp_physics.physics.utils import Recorder, Indices

# if not 'IGR_PATH' in os.environ:
#     os.environ['IGR_PATH'] = (Path(__file__).parent.parent.parent.parent / 'IGR').as_posix()
# assert Path(os.environ['IGR_PATH']).exists(), "Could not find IGR repository"


class Defaults3D(Defaults):
    # Dimensions
    DIM = 3

    # Contact detection parameter
    EPSILON = 0.001

    # Penetration tolerance parameter
    TOL = 2e-3

    # Default simulation parameters
    FRIC_DIRS = 8

    CONTACT = "FWContactHandler"

    # Whether to use custom meshes/inertia tensors for analytic shapes
    CUSTOM_MESH = False
    CUSTOM_INERTIA = False

    # DTYPE = torch.float
    DEVICE = torch.device('cuda:0')
    #DEVICE = torch.device('cpu')


class Recorder3D(Recorder):
    def __init__(self, dt, scene, path=os.path.join('videos', 'frames'), resolution=(640, 480), rotate=False,
                 rotate_rate=math.pi / 3.0, rotate_axis=[0, 0, 1], record_points=False, record_seg=False,
                 noise_factor=0.0, save_to_disk=False):
        super().__init__(dt, None, path)
        self.scene = scene
        self.rotate = rotate
        self.rotate_rate = rotate_rate
        self.rotate_axis = rotate_axis
        self.resolution = resolution
        self.record_points = record_points
        self.record_seg = record_seg
        self.noise_factor = noise_factor
        self.save_to_disk = save_to_disk


    def get_pointcloud( self, depth_img ):

        cam = next(iter(self.scene.camera_nodes))._camera

        # mask = np.where(depth_img > 0)
        # x = mask[1]
        # y = mask[0]

        x = np.linspace( 0, self.resolution[0]-1, self.resolution[0], dtype=np.int32 )
        y = np.linspace( 0, self.resolution[1]-1, self.resolution[1], dtype=np.int32 )
        x, y = np.meshgrid( x, y )

        normalized_x = (x.astype(np.float32) + 0.5 - cam.cx) / cam.fx
        normalized_y = (y.astype(np.float32) + 0.5 - cam.cy) / cam.fy

        # std: noise_factor * d^2
        depth_noise = np.random.randn( depth_img.shape[0], depth_img.shape[1] ) * self.noise_factor * (depth_img ** 2)
        depth_img = depth_img + depth_noise

        #depth_img = depth_img/1.04
        world_x = normalized_x * depth_img[y, x]
        world_y = normalized_y * depth_img[y, x]
        world_z = depth_img[y, x]

        return np.stack( (world_x, world_y, world_z), axis=-1 )

    def record(self, t, seg_node_map={}):
        color_img, depth_img, segmentation_mask, pc = None, None, None, None
        camera_poses = []
        if t - self.prev_t >= self.dt:
            renderer = pyrender.OffscreenRenderer(self.resolution[0], self.resolution[1])
            if self.rotate:
                angle = self.dt * self.rotate_rate
                rot = torch.eye(4)
                rot[:3, :3] = axis_angle_to_matrix(get_tensor(self.rotate_axis) * angle)
                for node in self.scene.camera_nodes:
                    camera_pose_matrix = rot.cpu().numpy() @ node.matrix
                    self.scene.set_pose(node, camera_pose_matrix)

            for node in self.scene.camera_nodes:
                camera_poses.append(self.scene.get_pose(node))

            color_img, depth_img = renderer.render(self.scene,
                                  flags=pyrender.RenderFlags.SHADOWS_ALL
                                  )
            if self.save_to_disk:
                mpimg.imsave(os.path.join(self.path, 'color_{}.png'.format(self.frame)), color_img)
                mpimg.imsave(os.path.join(self.path, 'depth_{}.png'.format(self.frame)), depth_img)

            # print('min depth: ', np.min( np.array(depth_img) ), " max depth: ", np.max( np.array(depth_img) ) )

            if self.record_seg:
                segmentation_mask = renderer.render(self.scene,
                                      flags=pyrender.RenderFlags.SEG,
                                      seg_node_map=seg_node_map,
                                      )[0]
                if self.save_to_disk:
                    mpimg.imsave(os.path.join(self.path, 'seg_{}.png'.format(self.frame)), segmentation_mask)
            # mppyplot.imshow(segmentation_mask)

            if self.record_points:
                # create point cloud from depth map
                pc = self.get_pointcloud( depth_img )
                pcd = trimesh.PointCloud(pc.reshape(-1, 3))
                if self.save_to_disk:
                    pcd.export(os.path.join(self.path, 'pointcloud_{}.ply'.format(self.frame)))

                    np.save( os.path.join(self.path, 'pointcloud_{}.npy'.format(self.frame)), pc )

            self.frame += 1
            self.prev_t += self.dt
            renderer.delete()

        return color_img, depth_img, pc, segmentation_mask, camera_poses



def get_colormap():

    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    return colormap

def Rx(theta):
    theta = get_tensor(theta)
    return get_tensor([[1, 0, 0, 0],
                       [0, torch.cos(theta), -torch.sin(theta), 0],
                       [0, torch.sin(theta), torch.cos(theta), 0],
                       [0, 0, 0, 1]])


def Ry(theta):
    theta = get_tensor(theta)
    return get_tensor([[torch.cos(theta), 0, -torch.sin(theta), 0],
                       [0, 1, 0, 0],
                       [torch.sin(theta), 0, torch.cos(theta), 0],
                       [0, 0, 0, 1]])


def Rz(theta):
    theta = get_tensor(theta)
    return get_tensor([[torch.cos(theta), -torch.sin(theta), 0, 0],
                       [torch.sin(theta), torch.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])


def quat(vec, style='wxyz'):
    ''' Returns a quaternion, and takes euler angles as inputs'''
    [phi, the, psi] = [0.5 * v for v in vec]

    w = math.cos(phi) * math.cos(the) * math.cos(psi) + math.sin(phi) * math.sin(the) * math.sin(psi)
    x = math.sin(phi) * math.cos(the) * math.cos(psi) - math.cos(phi) * math.sin(the) * math.sin(psi)
    y = math.cos(phi) * math.sin(the) * math.cos(psi) + math.sin(phi) * math.cos(the) * math.sin(psi)
    z = math.cos(phi) * math.cos(the) * math.sin(psi) - math.sin(phi) * math.sin(the) * math.cos(psi)

    if style == 'xyzw':
        return vec.new_tensor([x, y, z, w])
    elif style == 'wxyz':
        return vec.new_tensor([w, x, y, z])
    else:
        raise ValueError("Invalid quaternion representation")


def cart_to_spherical(cart_vec, positive=True):
    r = cart_vec.norm()
    phi = torch.atan2(cart_vec[Indices.Y], cart_vec[Indices.X])
    theta = torch.atan2(cart_vec[Indices.Z], cart_vec[[Indices.X, Indices.Y]].norm())

    if theta < 0 and positive:
        theta = theta + 2 * math.pi

    if phi < 0 and positive:
        phi = phi + 2 * math.pi

    return r, theta, phi


def spherical_to_cart(r, theta, phi):
    rcos_theta = r * torch.cos(theta)
    x = rcos_theta * torch.cos(phi)
    y = rcos_theta * torch.sin(phi)
    z = r * torch.sin(theta)
    ret = torch.stack([x, y, z])
    return ret


def orthogonal(v):
    """Get any orthogonal vector in 3D space
    """
    coord_dirs = torch.eye(v.shape[0], dtype=v.dtype, device=v.device)

    corr_dirs = coord_dirs @ v

    dir_ind = corr_dirs.abs().argmin()

    return torch.cross(coord_dirs[dir_ind], v)


def skew_symmetric_mat(v):
    assert v.shape[0] == 3
    mat = v.new_zeros(3, 3)
    mat[0, 1] = -v[2]
    mat[0, 2] = v[1]
    mat[1, 2] = -v[0]

    mat = mat - mat.t()
    return mat


def get_tensor(x, base_tensor=None, **kwargs):
    """Wrap array or scalar in torch Tensor, if not already.
    """
    if isinstance(x, torch.Tensor):
        return x
    elif base_tensor is not None:
        return base_tensor.new_tensor(x, **kwargs)
    else:
        return torch.tensor(x,
                            dtype=Defaults3D.DTYPE,
                            device=Defaults3D.DEVICE,
                            # layout=Params.DEFAULT_LAYOUT,
                            **kwargs,
                            )


def load_igrnet(experiment_dir, timestamp='latest', checkpoint='latest', run=None):
    if timestamp == 'latest':
        timestamps = os.listdir(experiment_dir)
        if len(timestamps) > 0:
            timestamp = sorted(timestamps)[-1]
        else:
            raise FileNotFoundError('No timestamp directories found in {}'.format(experiment_dir))

    conf_filename = os.path.join(experiment_dir, timestamp, 'exp.conf')
    conf = ConfigFactory.parse_file(conf_filename)

    latent_size = conf.get_int('train.latent_size')
    d_in = conf.get_int('train.d_in')

    kls = conf.get_string('train.network_class')
    parts = kls.split('.')
    module = "/".join(parts[:-1])
    spec = importlib.util.spec_from_file_location('network',
                                                  os.path.abspath(os.environ['IGR_PATH'] + '/code/' + module + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m = getattr(m, parts[-1])
    network = m(d_in=(d_in+latent_size), **conf.get_config('network.inputs'))

    old_checkpnts_dir = os.path.join(experiment_dir, timestamp, 'checkpoints')

    latent_filename = os.path.join(old_checkpnts_dir, 'LatentCodes', str(checkpoint) + '.pth')
    data = torch.load(latent_filename)
    lat_vecs = data["latent_codes"].detach().to(dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)

    model_params_filename = os.path.join(old_checkpnts_dir, 'ModelParameters', str(checkpoint) + ".pth")
    saved_model_state = torch.load(model_params_filename)
    network.load_state_dict(saved_model_state["model_state_dict"])
    network = network.to(dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)
    network.eval()

    if run is not None:
        run.add_resource(conf_filename)
        run.add_resource(latent_filename)
        run.add_resource(model_params_filename)

    return network, lat_vecs


def decode_igr(network):
    def _decode_igr(samples, latent):
        num_samples = samples.shape[0]
        latent_rep = latent.expand(num_samples, -1)
        sdf = network(torch.cat([latent_rep, samples], 1))
        return sdf

    def sdf(pts, latent, max_batch=32 ** 3):
        num_samples = pts.shape[0]

        sdfs = pts.new_zeros(num_samples)

        head = 0
        while head < num_samples:
            sample_subset = pts[head:min(head + max_batch, num_samples), :]
            sdfs[head:min(head + max_batch, num_samples)] = _decode_igr(sample_subset, latent).squeeze()
            head += max_batch

        return sdfs

    return sdf


def plot_sdf_slices(res, n_slices, sdf_func, sdf_params, slice_dims=[0, 1, 2], plot_contours=[]):
    samp_space = torch.linspace(-1., 1., res, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)
    samples = torch.stack(torch.meshgrid(samp_space, samp_space, samp_space), dim=3).reshape(-1, 3)

    with torch.no_grad():
        sdfs = sdf_func(samples, *sdf_params)
        sdfs = sdfs.reshape(res, res, res)

    fig, axs = plt.subplots(nrows=n_slices, ncols=len(slice_dims))
    for i in range(n_slices):
        for j in slice_dims:
            index = torch.tensor([int(i / (n_slices - 1) * (sdfs.shape[j] - 1))
                                  if n_slices > 1 else int(sdfs.shape[j] / 2)])
            sdf_slice = torch.index_select(sdfs.cpu(), j, index).squeeze()
            if j > 0:
                sdf_slice = sdf_slice.t()
            sdf_slice = sdf_slice.flip(0)
            if n_slices > 1 and len(slice_dims) > 1:
                ax = axs[i, j]
            elif len(slice_dims) > 1:
                ax = axs[j]
            elif n_slices > 1:
                ax = axs[i]
            else:
                ax = axs
            pos = ax.imshow(sdf_slice, cmap='seismic', vmin=-1, vmax=1)
            if plot_contours:
                cs = ax.contour(sdf_slice, levels=plot_contours, colors='black')
                ax.clabel(cs, inline=1, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel( '' )
            ax.set_ylabel( '' )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = fig.colorbar(pos, cax=cax)
            cb.set_label('SDF value')

    return fig
