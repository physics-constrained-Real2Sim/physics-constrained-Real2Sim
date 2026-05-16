import os
import json
import trimesh
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import random
import argparse
from icecream import ic
from logging import warning
from scipy.spatial.transform import Rotation as R

# data loader
from ultility import *
from data_reader import read_scene_zarr

# evaluation ultility
from evaluation.ultility_eval_physics import evaluate_stable
from evaluation.ultility_eval_geom import GeometryEvaluator
from evaluation.ultility_eval_rgb import ImageEvaluator

def load_exported_objects_from_json(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # list[dict]


def load_models(visual_mesh_file, 
                vhacd_mesh_file, 
                desired_mass = 1.0,
                position = [0.0,0.0,0.0],
                baseOrientation = [0,0,0,1],
                center_of_mass = [0.0,0.0,0.0],
                lateral_friction = 0.6,
                spinning_friction = 0.003,
                ):
    """
    Load a body into the simulation, given the visual mesh file and the vhacd mesh file.
    Position should be zero if the mesh's origin is at world origin.
    Center of mass very important, should be calculated before loading.
    Returns the body id of the loaded body.
    """
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=vhacd_mesh_file,
                                                meshScale=[1, 1, 1])
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                          fileName=visual_mesh_file,
                                          meshScale=[1, 1, 1])
    
    body_id = p.createMultiBody(baseMass= desired_mass, 
                                baseCollisionShapeIndex=collision_shape_id,
                                baseVisualShapeIndex=visual_shape_id,
                                basePosition=position,
                                baseInertialFramePosition=center_of_mass, 
                                baseOrientation=baseOrientation)

    p.changeDynamics(body_id, -1, lateralFriction = lateral_friction, 
                     spinningFriction = spinning_friction,
                     restitution = 0.5,
                     rollingFriction = 0.005)

    return body_id


def get_com_plus_predict(mesh_file, com_predict):
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Input file '{mesh_file}' does not exist.")
    return trimesh.load(mesh_file).bounding_box.centroid + np.array(com_predict)


def get_com(mesh_file):
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Input file '{mesh_file}' does not exist.")
    return trimesh.load(mesh_file).bounding_box.centroid


def capture_image(width, height, cam_position, fov_y_deg, target_look_at, cam_up_direction, near, far):

    W, H = int(width), int(height)
    cam_pos = cam_position
    target = target_look_at
    fov_y = fov_y_deg
    cam_up = cam_up_direction
    view = p.computeViewMatrix(cam_pos, target, cam_up)
    proj = p.computeProjectionMatrixFOV(fov_y, aspect=float(W)/H, nearVal=near, farVal=far)

    width, height, RGB, depth, mask = p.getCameraImage(
        width=W,
        height=H,
        viewMatrix=view, 
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,  
        lightDirection=[0, -1, 0.8],               
        lightColor=[1, 1, 1],                   
        lightDistance=3.0,                      
        shadow=1,                               
    )

    BGR = RGB[:, :, [2, 1, 0]]
    
    simulated_depth = far * near / (far - (far - near) * depth)

    return width, height, RGB, depth, mask, simulated_depth, BGR


def generate_video_from_rgb(rgb_frames, sim_dir, file_name, fps=30):
    """
    Generate a video from a list of RGB image frames.
    
    Parameters:
      rgb_frames: list of NumPy arrays, each array is an image in RGB format.
      output_path: string, the file path to save the video (e.g., 'simulation_video.mp4').
      fps: frames per second for the video.
    """
    if not rgb_frames:
        raise ValueError("The list of RGB frames is empty!")
    
    height, width, channels = rgb_frames[0].shape
    if channels != 3:
        raise ValueError("Each frame should have 3 channels (RGB).")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(sim_dir,f"{file_name}.avi"), fourcc, fps, (width, height))
    
    for frame in rgb_frames:
        video_writer.write(frame)
    
    video_writer.release()
    print("Video saved to", os.path.join(sim_dir,f"{file_name}.avi"))


def get_true_PositionAndOrientation(body_id):

    pos, quat = p.getBasePositionAndOrientation(body_id)
    dyn = p.getDynamicsInfo(body_id, -1)
    local_inertial_pos = dyn[3]      # baseInertialFramePosition
    local_inertial_orn = dyn[4]      # baseInertialFrameOrientation

    com_world_pos, com_world_orn = p.getBasePositionAndOrientation(body_id)

    # T^W_base = T^W_com * (T^base_inertial)^(-1)
    inv_pos, inv_orn = p.invertTransform(local_inertial_pos, local_inertial_orn)
    mesh_world_pos, mesh_world_orn = p.multiplyTransforms(com_world_pos, com_world_orn,
                                                        inv_pos, inv_orn)
    
    return mesh_world_pos, mesh_world_orn

def coacd_convex_decomposition(obj_filename,threshold = 0.03, preprocess_resolution = 100):
    """
    COACD convex decomposition
    """
    import coacd
    output_file = os.path.splitext(obj_filename)[0] + "_coacd.obj"

    if os.path.exists(output_file):
        print(f"[COACD]: Convex decomposition mesh {output_file} exists")
        return output_file
    
    mesh = trimesh.load(obj_filename, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    #result = coacd.run_coacd(mesh, threshold= 0.02, mcts_max_depth= 5, mcts_nodes= 30, preprocess_mode= "False") # a list of convex hulls.
    result = coacd.run_coacd(mesh, threshold = threshold, preprocess_resolution =preprocess_resolution) # a list of convex hulls.

    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    np.random.seed(0)
    for p in mesh_parts:
        p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)
    scene.export(output_file)

    return output_file


def check_stability(body_ids, lin_th = 0.05, ang_th = 0.05):
    for b in body_ids:
        lin, ang = p.getBaseVelocity(b)
        if np.linalg.norm(lin) > lin_th or np.linalg.norm(ang) > ang_th:
            return False
    return True

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="demo_google7",
        required=True,
        help="Choose dataset"
    )
    parser.add_argument(
        "--result",
        type=str,
        default="geom_optim",
        choices=["physics_optim", "initial_guess", "geom_optim"],
        help="Choose result type"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize rollout"
    )

    args = parser.parse_args()
    root_path   = f"./dataset/{args.dataset}"
    zarr_path   = f"./dataset/{args.dataset}/scene.zarr"
    optim_path  = f"./dataset/{args.dataset}/{args.result}"     # physics_optim | initial_guess | geom_optim
    meta_result = f"./dataset/{args.dataset}/{args.result}/result.json"

    # evaluation outputs
    output_dir = optim_path
    os.makedirs(output_dir, exist_ok=True)
    result_meta_json    = f"{output_dir}/simulated_meta_result.json"
    simulated_objs_json = f"{output_dir}/simulated_objs_result.json"
    initial_image       = f"{output_dir}/pybullet_initial_image.png"
    final_image         = f"{output_dir}/pybullet_final_image.png"
    pybullet_video      = "rollout_video"

    # rollout video
    get_video      = args.visualize

    # read GT data
    data = read_scene_zarr(zarr_path)
    width, height, cam_position, fov_y_deg, target_look_at, cam_up_direction, near, far = data.get("metadata_camera", (None,)*8)
    rgb, depth, seg_mask, intrinsic, extrinsic_world_to_cam, extrinsic_cam_to_world = data.get("camera", (None,)*6)
    partial_points, partial_colors, partial_mask = data.get("partial", (None,)*3)
    complete_points, complete_mask = data.get("complete", (None,)*2)
    GT_objs = data.get("objects", [])

    ## Load optimized experiment meta result
    predict_data = load_exported_objects_from_json(meta_result)

    # physics init
    physicsClient = p.connect(p.DIRECT) #or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0,0,-9.8)
    p.setTimeStep( 1./240.)

    # -------- ground only --------
    planeId = p.loadURDF("./dataset/plane/plane.urdf")

    body_ids = []
    bid2idx = {}
    for i in range(len(predict_data)):
        object = predict_data[i]
        if object["type"] == "ground":
            continue

        object_mesh_path        = object.get("mesh_path")
        post_process_mesh_path  = object.get("post_processed_mesh_path")
        ic(post_process_mesh_path)
        convex_mesh_path        = coacd_convex_decomposition(object_mesh_path)
        visual_mesh_path        = post_process_mesh_path if post_process_mesh_path is not None else object_mesh_path

        transformation          = np.array(object["transformation_accumulation"])
        mass_result             = object.get("mass_result")
        friction_result         = object.get("friction_result")
        com_result              = object.get("com_result")    
        post_process_mesh       = object.get("post_processed_mesh_path")

        mass = mass_result if mass_result is not None else 0.2
        friction = friction_result if friction_result is not None else 0.2

        com  = get_com_plus_predict(object_mesh_path,com_result) if com_result is not None else get_com(object_mesh_path)
        object["com_world"] = com.tolist()

        # ic(transformation[:3, :3])
        r = R.from_matrix(transformation[:3, :3])
        init_quaternion = r.as_quat()
        init_position   = transformation[:3, 3]

        object["init_position"] = init_position.tolist()
        object["init_quaternion"] = init_quaternion.tolist()
        object_id = load_models(
            visual_mesh_file = visual_mesh_path,
            vhacd_mesh_file  = convex_mesh_path,
            desired_mass     = np.array(mass),
            position         = init_position,
            baseOrientation  = init_quaternion,
            center_of_mass   = np.array(com),
            lateral_friction = np.array(friction),
        )
        body_ids.append(object_id)
        bid2idx[object_id] = i
        print(f"[simulation] loaded object {i}, body_id={object_id}")


    frames = []
    width, height, RGB_init, depth, mask_init, simulated_depth, BGR = capture_image(width, height, cam_position, fov_y_deg, 
                                                                          target_look_at, cam_up_direction, near, far)
    cv2.imwrite(initial_image, BGR)

    """ ----------------Simulation -------------------"""
    settling_time      = 0
    settling_frames   = 0
    for counter in range(1000):
        if counter % 10 == 0 and get_video:
            width, height, RGB, depth, mask, simulated_depth, BGR = capture_image(width, height, cam_position, fov_y_deg,
                                                                                  target_look_at, cam_up_direction, near, far)
            frames.append(BGR)
        ## Here we also recorde the object velocity accumulation under the simulated time steps
        for bid in body_ids:
            lin_vel, ang_vel = p.getBaseVelocity(bid)
            obj = predict_data[bid2idx[bid]]

            obj.setdefault("lin", []).append(lin_vel)
            obj.setdefault("ang", []).append(ang_vel)
        if not check_stability(body_ids):
            settling_frames += 1
        
        p.stepSimulation()
        print(f"[simulation] stepping simulation: {counter}/1000", end="\r")

    settling_time = settling_frames * 1./240.
    ic(settling_frames)
    ic(settling_time)
    width, height, RGB_sim, depth, mask_sim, simulated_depth, BGR = capture_image(width, height, cam_position, fov_y_deg, 
                                                                          target_look_at, cam_up_direction, near, far)
    cv2.imwrite(final_image, BGR)

    if get_video:
        generate_video_from_rgb(frames, output_dir , pybullet_video, fps= 240/10)


    """ ----------------Physics evaluation -------------------"""
    non_ground_counter = 0
    for i, obj in enumerate(predict_data):
        if obj["type"] == "ground":
            continue  # ground
        final_pos, final_quat = get_true_PositionAndOrientation(body_ids[non_ground_counter])
        
        sim_T = np.eye(4)
        sim_T[:3, :3] = R.from_quat(final_quat).as_matrix()
        sim_T[:3, 3]  = final_pos
        obj["sim_transform"] = sim_T.tolist()

        init_position = np.array(obj["init_position"], dtype=float)
        init_quaternion = np.array(obj["init_quaternion"], dtype=float)
        is_stable, pos_diff, ang_diff_deg = evaluate_stable(pose = final_pos, quaternion= final_quat,
                                            pose_init = init_position, quaternion_init= init_quaternion,
                                            pos_thresh= 0.05, ang_thresh_deg= 5.0)
        
        print(f"[physics evaluation] object {i}: stable={is_stable}, pos_diff={pos_diff}, ang_diff_deg={ang_diff_deg}")
        obj["is stable"] = bool(is_stable)
        obj["position_difference"] = float(pos_diff)
        obj["orientation_difference_deg"] = float(ang_diff_deg)

        translation_vel = np.linalg.norm( np.mean( np.array(obj["lin"]), axis=0) )
        rotation_vel    = np.linalg.norm( np.mean( np.array(obj["ang"]), axis=0) )
        obj["translation_velocity"] = float(translation_vel)
        obj["rotation_velocity"]    = float(rotation_vel)
        print(f"[physics evaluation] Object {i}: translation velocity={translation_vel}, rotation velocity={rotation_vel}")

        non_ground_counter += 1


    average_stability = np.mean([1.0 if obj["is stable"] else 0.0 for obj in predict_data if obj["type"] != "ground"])
    average_translation_vel = np.mean([obj["translation_velocity"] for obj in predict_data if obj["type"] != "ground"])
    average_rotation_vel    = np.mean([obj["rotation_velocity"] for obj in predict_data if obj["type"] != "ground"])
    
    print(f"    [physics evaluation] average stability: {average_stability}")
    print(f"    [physics evaluation] average translation velocity: {average_translation_vel}")
    print(f"    [physics evaluation] average rotation velocity: {average_rotation_vel} \n")

   
    """ ----------------Geometry Evaluation -------------------"""
    ## Second one: Geometry metrics
    Geometry_evaluator = GeometryEvaluator()
    Image_evaluator = ImageEvaluator()  

    gt_centers_world = np.asarray(
        [np.asarray(g["position"], dtype=float) for g in GT_objs],
        dtype=float
    )  # (N, 3)

    if len(predict_data) -1 != len(GT_objs):
        warning("The number of predicted objects and ground-truth objects should be the same.")
    
    ## Find correspondence between predicted objects and GT objects
    for i, obj in enumerate(predict_data):
        if obj["type"] == "ground":
            continue  # ground

        obj_mesh = trimesh.load(obj["mesh_path"], force="mesh")
        obj_mesh.apply_transform(np.array(obj["transformation_accumulation"]))

        ## find corresponding ground-truth object in the GT point clouds
        ## we use centroid distance to find the closest one as correspondence
        pred_center = obj_mesh.bounding_box.centroid  # (3,)

        if 'remaining_idx' not in locals():
            remaining_idx = list(range(len(gt_centers_world)))

        if not remaining_idx:
            warning("No remaining GT objects to match, but there are still predicted objects.")
        else:
            cand = gt_centers_world[remaining_idx]              # (M,3)
            d = np.linalg.norm(cand - pred_center[None, :], axis=1)
            j = int(np.argmin(d))
            gt_idx = remaining_idx[j]

            obj["gt_index"] = int(gt_idx)                       # g_objs
            obj["gt_center_of_mass"] = gt_centers_world[gt_idx].tolist()
            obj["gt_match_dist_center"] = float(d[j])

            remaining_idx.pop(j)

            print(f"Object {i}: matched GT object {gt_idx} with center distance {d[j]:.4f} m.")
            ## pose erro is gt_match_dist_center

            ## evaluate friction and COM error
            obj["friction_error"] = np.abs(GT_objs[gt_idx]["friction"] - 0.1)
            obj["com_error"] = np.linalg.norm( np.array(GT_objs[gt_idx]["position"]) - np.array(obj.get("com_world", [0.0,0.0,0.0])) )
            print(f"[physics evaluation] Object {i}: friction error={obj['friction_error']}, COM error={obj['com_error']}")
            ## prepare predict and GT point clouds for geometry metrics calculation
            GT_matched_points = complete_points[complete_mask == GT_objs[gt_idx]["bid"] - 1]  

            pred_points = obj_mesh.sample(3000)
            result = Geometry_evaluator(pred_points, GT_matched_points)

            cd_dist, fscore, iou = result
            obj["geometry_metrics"] = {
                "chamfer_distance": float(cd_dist),
                "fscore": float(fscore),
                "iou": float(iou),
            }

            print(f"[geometry evaluation] object {i}: CD={cd_dist}, F-score={fscore}, IoU={iou}")

            ## Object level visual metrics  
            gt_object_mask = (seg_mask == GT_objs[gt_idx]["bid"] )[:, :, None] 
            gt_object_image = np.where(gt_object_mask, rgb, 0)
            # cv2.imwrite(f"gt_object_{gt_idx}.png", cv2.cvtColor(gt_object_image, cv2.COLOR_RGB2BGR))

            pre_object_mask = (mask_init == i)[:, :, None]      # H x W bool
            pre_object_image = np.where(pre_object_mask, RGB_init[:, :, :3], 0)  # H x W x 3
            # cv2.imwrite(f"pre_object_{gt_idx}.png", cv2.cvtColor(pre_object_image, cv2.COLOR_RGB2BGR))
            obj_psnr, obj_ssim, obj_lpips_val = Image_evaluator(pred=pre_object_image/255.0, gt=gt_object_image/255.0)

            obj["psnr"]  = float(obj_psnr)
            obj["ssim"]  = float(obj_ssim)
            obj["lpips"] = float(obj_lpips_val)
            print(f"[visual evaluation]: Object PSNR={obj_psnr}, SSIM={obj_ssim}, LPIPS={obj_lpips_val} \n")

    ## Average physics metrics over all non-ground objects
    avg_friction_error = np.mean([obj["friction_error"] for obj in predict_data if obj["type"] != "ground"])
    avg_com_error      = np.mean([obj["com_error"] for obj in predict_data if obj["type"] != "ground"])
    print(f"  [physics evaluation] Average physics metrics: friction_error={avg_friction_error}, COM_error={avg_com_error} \n")    

    ## get the average metrics over all non-ground objects
    avg_cd_dist = np.mean([obj["geometry_metrics"]["chamfer_distance"] for obj in predict_data if obj["type"] != "ground"])
    avg_fscore  = np.mean([obj["geometry_metrics"]["fscore"] for obj in predict_data if obj["type"] != "ground"])
    avg_iou     = np.mean([obj["geometry_metrics"]["iou"] for obj in predict_data if obj["type"] != "ground"])
    print(f"  [geometry evaluation] Average geometry metrics: CD={avg_cd_dist}, F-score={avg_fscore}, IoU={avg_iou} \n")


    aveg_psnr = np.mean([obj.get("psnr", 0.0) for obj in predict_data if obj["type"] != "ground"])
    aveg_ssim = np.mean([obj.get("ssim", 0.0) for obj in predict_data if obj["type"] != "ground"])
    aveg_lpips = np.mean([obj.get("lpips", 0.0) for obj in predict_data if obj["type"] != "ground"])
    print(f" [visual evaluation] [Object-Level] Visual metrics: PSNR={aveg_psnr}, SSIM={aveg_ssim}, LPIPS={aveg_lpips} \n")
    
    
    """ ----------------Scene-level Visual Evaluation -------------------"""
    ## Scene-level visual metrics
    ## we only evaluate non-ground objects
    non_ground_mask = (seg_mask != 0)[:, :, None]      # H x W bool
    GT_rgb_obj = np.where(non_ground_mask, rgb, 0)  # H x W x 3
    #cv2.imwrite(f"gt_non_ground.png", cv2.cvtColor(GT_rgb_obj, cv2.COLOR_RGB2BGR))

    sim_non_ground_mask = (mask_init != 0)[:, :, None]      # H x W bool
    pre_rgb_obj = np.where(sim_non_ground_mask, RGB_init[:, :, :3], 0)  # H x W x 3
    #cv2.imwrite(f"pre_non_ground.png", cv2.cvtColor(pre_rgb_obj, cv2.COLOR_RGB2BGR))

    psnr, ssim, lpips_val = Image_evaluator(pred=pre_rgb_obj/255.0, gt=GT_rgb_obj/255.0)
    print(f"\n  [visual evaluation] [Scene-Level] PSNR={psnr}, SSIM={ssim}, LPIPS={lpips_val} \n")

    result_meta_data = {
        "[physics]stability": float(average_stability),
        "[physics]settling_time": float(settling_time),
        "[physics]translation_velocity": float(average_translation_vel),
        "[physics]rotation_velocity": float(average_rotation_vel),
        "[geometry]chamfer_distance": float(avg_cd_dist),
        "[geometry]fscore": float(avg_fscore),
        "[geometry]iou": float(avg_iou),
        "[visual-scene]psnr": float(psnr),
        "[visual-scene]ssim": float(ssim),
        "[visual-scene]lpips": float(lpips_val),
        "[visual-object]psnr": float(aveg_psnr),
        "[visual-object]ssim": float(aveg_ssim),
        "[visual-object]lpips": float(aveg_lpips),
        "[physics]friction_error": float(avg_friction_error),
        "[physics]com_error": float(avg_com_error),}
    
    ## write result meta json
    with open(result_meta_json, "w", encoding="utf-8") as f:
        json.dump(result_meta_data, f, indent=4)
    print(f"[OK] Saved simulation evaluation result meta to {result_meta_json}")
    print("all result: \n")
    print(json.dumps(result_meta_data, indent=4, ensure_ascii=False))

    ## write result meta json with per-object detailed info
    with open(simulated_objs_json, "w", encoding="utf-8") as f:
        json.dump(predict_data, f, indent=2, ensure_ascii=False)



