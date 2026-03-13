import argparse
import os
import numpy as np
import random
from icecream import ic
import trimesh
from ultility import *
from tqdm import tqdm
from data_reader import read_scene_zarr

def save_transformed_mesh_obj(mesh_path: str, T_total: np.ndarray, out_path: str):
    """
    read mesh from mesh_path, apply transformation T_total, and save as OBJ to out_path.
    """
    if not isinstance(T_total, np.ndarray) or T_total.shape != (4, 4):
        raise ValueError("T_total must be (4,4) numpy array")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mesh = trimesh.load_mesh(mesh_path)
    mesh.apply_transform(T_total.astype(np.float64))
    mesh.export(out_path, file_type='obj')

    print(f"[Mesh export]: Mesh is exported to {out_path}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    o3d.utility.random.seed(0)
    seed_everything(42)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="demo_google7",
        required=True,
        help="Choose dataset"
    )
    args = parser.parse_args()

    dataset_dir          = f"./dataset/{args.dataset}"
    zarr_path            = f"{dataset_dir}/scene.zarr" 
    SAM3D_recon_path     = f"{dataset_dir}/SAM3D_recon"
    pose_refinement_path = f"{dataset_dir}/ICP_refinement"
    
    if not os.path.exists(pose_refinement_path):
        os.makedirs(pose_refinement_path, exist_ok=True)

    ## Load data from zarr
    data = read_scene_zarr(zarr_path)
    width, height, cam_position, fov_y_deg, target_look_at, cam_up_direction, near, far = data.get("metadata_camera", (None,)*7)
    rgb, depth, seg_mask, intrinsic, extrinsic_world_to_cam, extrinsic_cam_to_world = data.get("camera", (None,)*6)
    partial_points, partial_colors, partial_mask = data.get("partial", (None,)*3)
    complete_points, complete_mask = data.get("complete", (None,)*2)
    GT_objs = data.get("objects", [])

    all_meshes = []
    for obj_idx, obj in enumerate(tqdm(GT_objs)):
        body_id = obj["bid"]
        object_mask = (seg_mask == body_id)

        ## source mesh (reconstrcuted)
        SAM3D_reconstructed_mesh = f"{SAM3D_recon_path}/obj{body_id}/obj{body_id}.obj"
        ## GT point cloud 
        GT_point_cloud = partial_points[partial_mask==body_id]
        ## registered mesh path to be saved
        registered_mesh_save_path = f"{pose_refinement_path}/obj{body_id}/obj{body_id}.obj"
        os.makedirs(os.path.dirname(registered_mesh_save_path), exist_ok=True)

        if not os.path.exists(SAM3D_reconstructed_mesh):
            print("[Warning]: Missing reconstructed mesh for object ", body_id)
            break

        ## Prepare source point cloud from mesh
        o3d.utility.random.seed(0)
        mesh = o3d.io.read_triangle_mesh(SAM3D_reconstructed_mesh)
        source_pcd = mesh.sample_points_poisson_disk(number_of_points=5000) 
        source_point_cloud = np.asarray(source_pcd.points)

        o3d.utility.random.seed(0)
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(GT_point_cloud)
        pcd_gt_down = pcd_gt.farthest_point_down_sample(1000)
        GT_point_cloud = np.asarray(pcd_gt_down.points)
        # draw_registration_result(source_point_cloud, GT_point_cloud, np.eye(4))

        # draw_registration_result(source_point_cloud, GT_point_cloud, init_guess)

        # _, _, _, init_guess = ransac_warp(source_point_cloud,
        #                                 GT_point_cloud, 
        #                                 voxel_size= 0.05, 
        #                                 if_scale= False)
        # draw_registration_result(source_point_cloud, GT_point_cloud, init_guess)

        _, _, _, icp_obj_transformation = ICP_wrap(source_point_cloud,
                                                GT_point_cloud, 
                                                threshold = 0.01,
                                                if_scale= False, 
                                                trans_init =np.eye(4))
        # draw_registration_result(source_point_cloud, GT_point_cloud, icp_obj_transformation)

        mesh = trimesh.load_mesh(SAM3D_reconstructed_mesh)
        mesh.apply_transform(icp_obj_transformation.astype(np.float64))
        mesh.export(registered_mesh_save_path, file_type='obj')
        print(f"[Mesh export]: Mesh is exported to {registered_mesh_save_path}")
        all_meshes.append(mesh)

    scene = trimesh.Scene()
    for m in all_meshes:
        scene.add_geometry(m)
    scene.export(f"{pose_refinement_path}/scene.glb")
    print("ICP refined scene:", f"{pose_refinement_path}/scene.glb")