import os
import random
import zarr
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt

import open3d as o3d
import numpy as np
import torch
import copy
import cv2

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def offline_draw_registration_result_matplotlib(source, target, save_path=None):
    """
    Visualize point cloud registration results using Matplotlib (three-view layout)
    
    Parameters:
    source -- Source point cloud (N,3) or (N,6)
    source -- should use reality    pcd
    target -- should use simulated  pcd
    target -- Target point cloud (N,3) or (N,6)
    save_path -- Image save path (optional)
    """
    # Validate input shape
    if source.shape[1] not in [3, 6] or target.shape[1] not in [3, 6]:
        raise ValueError("Input data must be of shape (N, 3) or (N, 6)")
    
    # Extract coordinates
    source_pts = source[:, :3]
    target_pts = target[:, :3]
    
    # Combine point clouds and compute coordinate range
    all_pts = np.vstack((source_pts, target_pts))
    max_range = np.ptp(all_pts, axis=0).max() / 2.0
    mid_x = (np.max(all_pts[:, 0]) + np.min(all_pts[:, 0])) * 0.5
    mid_y = (np.max(all_pts[:, 1]) + np.min(all_pts[:, 1])) * 0.5
    mid_z = (np.max(all_pts[:, 2]) + np.min(all_pts[:, 2])) * 0.5

    # Create a Figure with three subplots (1 row, 3 columns)
    fig = plt.figure(figsize=(18, 6))  # Widen canvas for three views
    fig.suptitle('Point Cloud Registration (Blue: Reality; Yellow: Sim)', fontsize=20)
    
    # ===================== Subplot 1: YZ Plane View (elev=0°, azim=0°) =====================
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2], 
                s=2, c='yellow', alpha=0.7, label='Transformed recon pcds')
    ax1.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], 
                s=2, c='blue', alpha=0.5, label='GT partial pcds')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('YZ Plane View (elev=0°, azim=0°)')
    ax1.view_init(elev=0, azim=0)  # YZ plane view
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # ===================== Subplot 2: XY Plane View (elev=90°, azim=-90°) =====================
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2], 
                s=2, c='yellow', alpha=0.7)
    ax2.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], 
                s=2, c='blue', alpha=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('XY Plane View (elev=90°, azim=-90°)')
    ax2.view_init(elev=90, azim=-90)  # XY plane view (top view)
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # ===================== Subplot 3: Main View (elev=60°, azim=65°) =====================
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2], 
                s=2, c='yellow', alpha=0.7)
    ax3.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], 
                s=2, c='blue', alpha=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Main View (elev=60°, azim=65°)')
    ax3.view_init(elev=60, azim=65)  # Default isometric view
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add global legend (to avoid duplication)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    
    # Adjust subplot spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for main title
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Registered figure is saved at: {save_path}")
    else:
        plt.show()
    
    plt.close()

def registration_mesh_to_pointcloud(reconstructed_mesh_path, GT_point_cloud, iterations = 2):

    ## Prepare source point cloud from mesh
    o3d.utility.random.seed(0)
    mesh = o3d.io.read_triangle_mesh(reconstructed_mesh_path)
    source_pcd = mesh.sample_points_poisson_disk(number_of_points=2000) 
    source_point_cloud = np.asarray(source_pcd.points)

    ## Prepare target GT point cloud
    ## Prepare source point cloud from mesh
    o3d.utility.random.seed(0)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(GT_point_cloud)
    pcd_gt_down = pcd_gt.farthest_point_down_sample(2000)
    GT_point_cloud = np.asarray(pcd_gt_down.points)

    o3d.utility.random.seed(0)
    ## Scale guess with bounding box
    s_guess = robust_bbox_scale(source_point_cloud, GT_point_cloud,
                            p=0.02,
                            use_volume_cuberoot=False,
                            do_denoise=True,
                            do_pca=True)


    print("[Scale Guess]:", s_guess)
    src_scaled = scale_around_centroid(source_point_cloud, s_guess, extract_centroid_mesh(reconstructed_mesh_path))

    #draw_registration_result(src_scaled, GT_point_cloud, np.eye(4))

    src_scaled = src_scaled     

    o3d.utility.random.seed(0)
    transformed_source, T, fittness = staged_registration(src_scaled, GT_point_cloud, iterations = iterations)

    return transformed_source, s_guess, T, fittness




def staged_registration(source_point_cloud, GT_point_cloud, iterations = 3, final_icp = False):

    ## first stage: RANSAC rough alignment with scaling; ICP without scaling

    ## Next stage: RANSAC fine alignment with scaling; ICP with scaling

    T = np.eye(4)

    for i in range(iterations):
        o3d.utility.random.seed(0)
        _, obj_fitness, _, init_guess  = ransac_warp(source_point_cloud,
                                                    GT_point_cloud, 
                                                    voxel_size= 0.01, 
                                                    if_scale= True)

        #draw_registration_result(source_point_cloud, GT_point_cloud, init_guess)

        o3d.utility.random.seed(0)
        _, obj_fitness, _, icp_obj_transformation  = ICP_wrap(source_point_cloud,
                                                            GT_point_cloud, 
                                                            threshold = 0.01,
                                                            if_scale= False, 
                                                            trans_init =init_guess)
        source_point_cloud = transform_pcd(source_point_cloud, icp_obj_transformation)
        T = icp_obj_transformation @ T
        #draw_registration_result(source_point_cloud, GT_point_cloud, np.eye(4))
    o3d.utility.random.seed(0)
    _, obj_fitness, _, init_guess  = ransac_warp(source_point_cloud,
                                                GT_point_cloud, 
                                                voxel_size= 0.0001, 
                                                if_scale= True)

    #draw_registration_result(source_point_cloud, GT_point_cloud, init_guess)

    o3d.utility.random.seed(0)
    _, obj_fitness, _, icp_obj_transformation  = ICP_wrap(source_point_cloud,
                                                        GT_point_cloud, 
                                                        threshold = 0.001,
                                                        if_scale= False, 
                                                        trans_init =init_guess)
    source_point_cloud = transform_pcd(source_point_cloud, icp_obj_transformation)
    T = icp_obj_transformation @ T



    transformed_source = source_point_cloud
    return transformed_source, T, obj_fitness



def transform_pcd(pcd, transformation):
        # Validate input shapes
    if pcd.shape[1] not in [3, 6] :
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    source_o3d.transform(transformation)

    return  np.array(source_o3d.points)


def extract_centroid_mesh(mesh):
    mesh = trimesh.load(mesh, force='mesh')   

    return np.asarray(mesh.centroid)

def denoise_and_decimate(pts, nb_neighbors=20, std_ratio=2.0, min_cluster=50):
    """轻度去噪并去掉小簇，保持几何主体"""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # 统计滤波
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    # 半径聚类（参数需根据尺度调整）
    labels = np.array(pcd.cluster_dbscan(eps=np.linalg.norm(np.ptp(np.asarray(pcd.points), axis=0))/100.0,
                                         min_points=10, print_progress=False))
    if labels.size > 0 and labels.max() >= 0:
        counts = np.bincount(labels[labels>=0])
        keep_label = np.argmax(counts)  # 保留最大簇
        mask = labels == keep_label
        pcd = pcd.select_by_index(np.where(mask)[0])
    return np.asarray(pcd.points)

def pca_align(pts):
    c = pts.mean(axis=0)
    X = pts - c
    # PCA
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt  
    Xp = X @ R.T
    return Xp + c, R

def percentile_aabb_diag(pts, p=0.02, use_volume_cuberoot=False):
    """Quantiles of the AABB diagonal or volume^(1/3) as scale representatives."""
    qmin = np.quantile(pts, p, axis=0)
    qmax = np.quantile(pts, 1-p, axis=0)
    ext = np.maximum(qmax - qmin, 1e-12)
    if use_volume_cuberoot:
        return float(np.cbrt(ext[0] * ext[1] * ext[2]))
    else:
        return float(np.linalg.norm(qmax - qmin))

def robust_bbox_scale(src_pts, tgt_pts, p=0.02, use_volume_cuberoot=False,
                      do_denoise=True, do_pca=True):
    """Robust BB scale ratio estimation: optional denoising + PCA + quantile AABB"""
    S = src_pts.copy(); T = tgt_pts.copy()
    if do_denoise:
        S = denoise_and_decimate(S)
        T = denoise_and_decimate(T)
    if do_pca:
        S, _ = pca_align(S)
        T, _ = pca_align(T)
    dS = percentile_aabb_diag(S, p=p, use_volume_cuberoot=use_volume_cuberoot)
    dT = percentile_aabb_diag(T, p=p, use_volume_cuberoot=use_volume_cuberoot)
    return dT / dS if dS > 1e-12 else 1.0

def scale_around_centroid(pcd, scale, centroid):
    # Validate input shapes
    if pcd.shape[1] not in [3, 6] or pcd.shape[1] not in [3, 6]:
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")
    
    # Create source point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])

    pcd_o3d.scale(scale, center=centroid)

    return np.asarray(pcd_o3d.points)



def offline_draw_registration_result(source, target, transformation, save_path=None):
    # Convert tensors to numpy arrays if needed
    if isinstance(source, torch.Tensor):
        source = source.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Validate input shapes
    if source.shape[1] not in [3, 6] or target.shape[1] not in [3, 6]:
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")
    
    # Create source point cloud
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source[:, :3])
    if source.shape[1] == 6:
        source_o3d.colors = o3d.utility.Vector3dVector(source[:, 3:])
    
    # Create target point cloud
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target[:, :3])
    if target.shape[1] == 6:
        target_o3d.colors = o3d.utility.Vector3dVector(target[:, 3:])
    
    # Apply colors and transformation
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue
    source_temp.transform(transformation)
    
    # Create visualizer for offline rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Offline mode, window not displayed
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    
    # Set default view parameters (similar to draw_geometries)
    # view_ctl = vis.get_view_control()
    # view_ctl.set_front([0, 0, -1])  # Set default camera orientation
    # view_ctl.set_up([0, 1, 0])      # Vertical up vector
    
    # Automatic zoom to fit all geometry
    # vis.get_render_option().point_size = 5
    vis.update_renderer()
    
    # Capture rendered image
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image)
    
    # Convert to 8-bit color and save
    from matplotlib import pyplot as plt
    plt.imsave(save_path, image)
    
    # Clean up
    vis.destroy_window()

def visualize_point_cloud(data):
    """
    Visualizes a point cloud using Open3D. Supports N*3 and N*6 point clouds,
    and accepts both NumPy arrays and PyTorch tensors.

    :param data: A NumPy array or PyTorch tensor of shape (N, 3) or (N, 6).
                 For (N, 3), it represents the (x, y, z) coordinates of the points.
                 For (N, 6), it represents the (x, y, z, r, g, b) coordinates and colors of the points.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.shape[1] not in [3, 6]:
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:, :3])

    if data.shape[1] == 6:
        point_cloud.colors = o3d.utility.Vector3dVector(data[:, 3:])
    
    scale = np.linalg.norm(data[:, :3].max(axis=0) - data[:, :3].min(axis=0))

        
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([point_cloud,world_frame])



def visualize_point_cloud_offscreen(data, out_path,
                                     width=800, height=600,
                                     lookat=None, up=None, front=None):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if data.ndim != 2 or data.shape[1] not in (3, 6):
        raise ValueError("data must be N×3 or N×6 array/tensor")
    if data.shape[0] == 0:
        print(f"[OFFSCREEN] No pcd points, skip {out_path}")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    if data.shape[1] == 6:
        cols = data[:, 3:6].astype(np.float64)
        if cols.max() > 1.0:
            cols /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False) 
    
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  
    
    view_ctl = vis.get_view_control()
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    
    if lookat is None:
        lookat = center
    if up is None:
        up = [0, 0, 1]  
    if front is None:
        front = [0, 1, 0]  
    
    view_ctl.set_lookat(lookat)
    view_ctl.set_up(up)
    view_ctl.set_front(front)
    view_ctl.set_zoom(0.7) 
    
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path)
    
    vis.destroy_window()
    print(f"[OFFSCREEN] Saved {out_path}")


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, if_scale = False):
    distance_threshold = voxel_size * 1.0
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling= if_scale),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(5000000, 0.999))
    return result

def ransac_warp(source_pcd, target_pcd, voxel_size = 0.005, if_scale = False):
    if isinstance(source_pcd, torch.Tensor):
        source_pcd = source_pcd.cpu().numpy()
    
    if isinstance(target_pcd, torch.Tensor):
        target_pcd = target_pcd.cpu().numpy()

    if source_pcd.shape[1] not in [3, 6] or target_pcd.shape[1] not in [3, 6]:
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")

    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source_pcd[:, :3])

    if source_pcd.shape[1] == 6:
        source_o3d.colors = o3d.utility.Vector3dVector(source_pcd[:, 3:])
    
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target_pcd[:, :3])

    if target_pcd.shape[1] == 6:
        target_o3d.colors = o3d.utility.Vector3dVector(target_pcd[:, 3:])

    source_down, source_fpfh = preprocess_point_cloud(source_o3d, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_o3d, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, if_scale= if_scale)
    
    return result_ransac.correspondence_set, result_ransac.fitness, result_ransac.inlier_rmse, result_ransac.transformation


def ICP_wrap(source_pcd, target_pcd, threshold = 0.01, trans_init = np.eye(4), if_scale = False, if_visualize = False):
    
    if isinstance(source_pcd, torch.Tensor):
        source_pcd = source_pcd.cpu().numpy()
    
    if isinstance(target_pcd, torch.Tensor):
        target_pcd = target_pcd.cpu().numpy()

    if source_pcd.shape[1] not in [3, 6] or target_pcd.shape[1] not in [3, 6]:
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")

    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source_pcd[:, :3])

    if source_pcd.shape[1] == 6:
        source_o3d.colors = o3d.utility.Vector3dVector(source_pcd[:, 3:])
    
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target_pcd[:, :3])

    if target_pcd.shape[1] == 6:
        target_o3d.colors = o3d.utility.Vector3dVector(target_pcd[:, 3:])
        
    result_ICP_p2p = o3d.pipelines.registration.registration_icp(
        source_o3d, target_o3d, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling= if_scale),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50000)
        )
    
    print(result_ICP_p2p)
    if if_visualize:
        # print("Transformation is:")

        # print(result_ICP_p2p.transformation)
        draw_registration_result(np.asarray(source_o3d.points), np.asarray(target_o3d.points), result_ICP_p2p.transformation)

    return result_ICP_p2p.correspondence_set, result_ICP_p2p.fitness, result_ICP_p2p.inlier_rmse, result_ICP_p2p.transformation

def draw_registration_result(source, target, transformation):
    if isinstance(source, torch.Tensor):
        source = source.cpu().numpy()
    
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    if source.shape[1] not in [3, 6] or target.shape[1] not in [3, 6]:
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")

    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source[:, :3])

    if source.shape[1] == 6:
        source_o3d.colors = o3d.utility.Vector3dVector(source[:, 3:])
    
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target[:, :3])

    if target.shape[1] == 6:
        target_o3d.colors = o3d.utility.Vector3dVector(target[:, 3:])

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, world_frame])



def save_mesh(mesh_result, filename):
    vertices = mesh_result.vertices.cpu().numpy() if hasattr(mesh_result.vertices, 'cpu') else mesh_result.vertices
    faces = mesh_result.faces.cpu().numpy() if hasattr(mesh_result.faces, 'cpu') else mesh_result.faces
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.cpu().numpy() if hasattr(mesh_result.vertex_attrs, 'cpu') else mesh_result.vertex_attrs
        mesh.visual.vertex_colors = attrs
    
    mesh.export(filename)

def convert_visible_mask(visible_mask, obj_id):
    mask = (visible_mask == obj_id)

    h, w = visible_mask.shape
    out = np.ones((h, w, 3), dtype=np.uint8) * 255  

    out[mask] = [188, 188, 188]

    return out
