import trimesh
import skimage
from icecream import ic
import numpy as np
from trimesh import Trimesh
from typing import Tuple
import os

import os
from pathlib import Path

os.environ['IGR_PATH'] = 'IGR'

import logging
import math
import os
import pickle
import sys
from pathlib import Path

import pyrender
import torch
from matplotlib import pyplot as plt

from .sdf_physics.physics3d.bodies import SDFBox, SDFCylinder, SDF3D, Mesh3D, SDFGrid3D
from .sdf_physics.physics3d.constraints import TotalConstraint3D
from .sdf_physics.physics3d.forces import Gravity3D
from .sdf_physics.physics3d.utils import get_tensor, Rx, Ry, Recorder3D, Defaults3D, load_igrnet, decode_igr
from .sdf_physics.physics3d.world import World3D, run_world

from icecream import ic
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

from .kal_sdf import kal_mesh_to_voxel

from geometry.object_3D import object_3D

class node_physics_optim():
    def __init__(self, node_object: object_3D, fixed_objects: list, voxel_resolution = 124):

        self.node_object = node_object

        self.fixed_objects = fixed_objects

        ## Convert node object to SDF voxel grid
        sdf_field, translation, scale = kal_mesh_to_voxel(mesh_path= None, 
                                                        voxel_resolution= voxel_resolution,
                                                        custom_mesh= True,
                                                        mesh = node_object.trimesh_obj)
        print(f"[mesh to sdf]: Done mesh to SDF by kaolin to mesh")

        self.voxels = sdf_field.squeeze(0).squeeze(0).permute(0, 2, 1).detach().to(dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)
        self.voxels = torch.flip(self.voxels, dims=[0])

        self.scale = 1/scale
        ## translation is that applied to mesh before converting
        self.node_origin_translation = -1 * get_tensor(translation)

        ## TODO do subdivide to Ground

        ## Merge fixed bodies and convert to Diffsdfsim coordinates
        fixed_body = trimesh.util.concatenate([obj.trimesh_obj for obj in fixed_objects])

        parent_obj_verts = np.asarray(fixed_body.vertices)

        parent_converted_vertices = np.column_stack((
            -parent_obj_verts[:, 0],   # X -> -X
            parent_obj_verts[:, 2],   # Z -> Y
            parent_obj_verts[:, 1],   # Y -> Z
        ))

        self.parent_vertices = np.array(parent_converted_vertices)
        self.parent_faces    = np.array(fixed_body.faces)

    @staticmethod
    def convert_poses(translation: torch.Tensor) -> torch.Tensor:
        return torch.stack([-1* translation[0],
                            translation[2],
                            translation[1]])

    @staticmethod
    def convert_COM(COM: torch.Tensor) -> torch.Tensor:
        return torch.stack([-1* COM[0],
                                COM[2],
                                COM[1]])
        

    def make_pair_diff_world(self, node_pos, node_COM, node_mass, node_friction, debug_mesh = True):

        bodies = []
        joints = []

        poses_converted = self.convert_poses(self.node_origin_translation + node_pos)
        COM_converted   = self.convert_COM(node_COM)
        
        child_body = SDFGrid3D(
            pos=poses_converted,
            sdf=self.voxels,
            scale = self.scale,
            vel=(0, 0, 0),
            COM=COM_converted,
            mass= node_mass,
            thickness=0.0,
            fric_coeff = node_friction,
            )
        child_body.add_force(Gravity3D())
        bodies.append(child_body)

        parent_body = Mesh3D(
            pos = (0.0,0.0,0.0),
            verts= torch.tensor(self.parent_vertices, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE),
            faces= torch.tensor(self.parent_faces, dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE).long(),
            thickness= 0.0,
        )

        bodies.append(parent_body)
        joints.append(TotalConstraint3D(parent_body))

        world = World3D(bodies, 
                        joints, 
                        strict_no_penetration= True, 
                        time_of_contact_diff= True, 
                        stop_contact_grad= False, 
                        stop_friction_grad= False) 

        return world


    def make_scene(self):
        
        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
        cam = pyrender.PerspectiveCamera(yfov=math.pi / 3, aspectRatio=4 / 3)
        # cam = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=1500)
        cam_pose = get_tensor([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 1],
                                [0, 0, 0, 1]])
        theta = math.pi / 4
        cam_pose = Ry(theta) @ Rx(-theta) @ cam_pose
        scene.add(cam, pose=cam_pose.cpu())
        light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
        light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
        scene.add(light1, pose=(Rx(-theta)).cpu())
        scene.add(light2, pose=(Ry(theta*2) @ Rx(-theta)).cpu())

        return scene
    
    def make_recorder(self, scene, path):
        
        recorder = Recorder3D(dt=Defaults3D.DT, scene=scene, path= path, save_to_disk=True)

        return recorder
    
    def collect_vel(self, world):
        vs = []
        for b in world.bodies:
            if isinstance(b, SDFGrid3D):
                vs.append(b.v.reshape(-1))   
        return torch.cat(vs, dim=0) if vs else torch.zeros((), device=world.device)

    # call back function for physics optimization, return the loss at each step
    def on_step(self, world):
        v = self.collect_vel(world)               
        loss_t = (v * v).sum() * float(world.dt)

        return loss_t                        

    def run_world(self, world, scene, recorder, TIME = 0.3):
        total_loss = run_world(world, fixed_dt= True, 
                               scene=scene, run_time=TIME, 
                               recorder=recorder, on_step= self.on_step)

        return total_loss