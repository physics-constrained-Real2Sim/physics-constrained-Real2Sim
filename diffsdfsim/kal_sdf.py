import kaolin as kal
import torch
import numpy as np
import trimesh
import skimage
from icecream import ic

def scale_to_unit_cube(mesh):

    ## translation and scale applied to mesh before converting SDF
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    translation = - mesh.bounding_box.centroid
    scale =  2 / np.max(mesh.bounding_box.extents)

    vertices = mesh.vertices + translation
    vertices *= scale

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces), translation, scale

def mesh_to_sdf(mesh, grid_size=64, batch_size=1):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        raise RuntimeError("CUDA is not available. Kernel error will occur without CUDA. Exiting.")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = torch.as_tensor(mesh.vertices, device=device, dtype=torch.float32)  # (V, 3)
    faces    = torch.as_tensor(mesh.faces,    device=device, dtype=torch.int64)    # (F, 3) indices


    axis = torch.linspace(-1, 1, grid_size) # range of x, y, z axis
    x, y, z = torch.meshgrid(axis, axis, axis)
    grid_points = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    grid_points_batch = grid_points.unsqueeze(0).repeat(batch_size, 1, 1)
    vertices_batch = vertices.unsqueeze(0).repeat(batch_size, 1, 1)

    face_vertices = kal.ops.mesh.index_vertices_by_faces(vertices_batch, faces)

    squared_distance, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(grid_points_batch, face_vertices)
    distance = torch.sqrt(squared_distance)

    sign = kal.ops.mesh.check_sign(vertices_batch, faces, grid_points_batch)
    sign_num = torch.where(sign, torch.tensor(-1.0).to(device), torch.tensor(1.0).to(device))

    sdf = distance * sign_num

    sdf_field = sdf.reshape(batch_size, 1, grid_size, grid_size, grid_size)

    return sdf_field

def kal_mesh_to_voxel(mesh_path, voxel_resolution = 64, custom_mesh = False, mesh = None):
    if custom_mesh:
        mesh_raw = mesh
    else:
        mesh_raw = trimesh.load(mesh_path)
    mesh, translation, scale = scale_to_unit_cube(mesh_raw)
    sdf_field                = mesh_to_sdf(mesh, grid_size= voxel_resolution)
    return sdf_field, translation, scale


if __name__ == "__main__":
    mesh_raw = trimesh.load('obj2_registered.obj')

    sdf_field, translation, scale = kal_mesh_to_voxel(mesh_path= 'obj2_registered.obj')
    print(sdf_field)
    print(sdf_field.shape)

    voxels = sdf_field.squeeze(0).squeeze(0).cpu().detach().numpy()
    ic(voxels.shape)
    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
    N = voxels.shape[0]
    # (0..N-1)--> [-1, 1]
    verts_norm = (vertices / (N - 1)) * 2.0 - 1.0

    mesh = trimesh.Trimesh(vertices=verts_norm, faces=faces, vertex_normals=normals)
    mesh.apply_scale(float(1/scale))
    mesh.apply_translation(-translation)

    mesh.export("obj2_reconstructed.obj")

    print(mesh_raw.bounding_box.extents)
    print(mesh.bounding_box.extents)