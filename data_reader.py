import os
import zarr
import numpy as np
import cv2
def read_scene_zarr(zarr_path):
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")

    root = zarr.open(zarr_path, mode="r")

    print("===== Scene Metadata =====")
    scene_meta = dict(root.attrs)
    for k, v in root.attrs.items():
        print(f"{k}: {v}")

    # scene camera meta

    print(scene_meta["fov_y_deg"])

    width = scene_meta["width"]
    height = scene_meta["height"]
    cam_position = scene_meta["cam_position"]
    fov_y_deg    = scene_meta["fov_y_deg"]
    target_look_at = scene_meta["target_look_at"]
    cam_up_direction = scene_meta["cam_up_direction"]
    near = scene_meta["near"]
    far = scene_meta["far"]
    print("\nCamera Metadata:")
    print(" width:", width)
    print(" height:", height)
    print(" cam_position:", cam_position)
    print(" fov_y_deg:", fov_y_deg)
    print(" target_look_at:", target_look_at)
    print(" cam_up_direction:", cam_up_direction)
    print(" near:", near)
    print(" far:", far)


    # ---- Camera ----
    g_cam = root["camera"]
    rgb = g_cam["rgb"][:]           # (H,W,3) uint8
    depth = g_cam["depth_m"][:]     # (H,W) float32
    seg_mask = g_cam["seg_mask"][:] # (H,W) int32
    intrinsic = g_cam["intrinsic"][:]
    extrinsic_cam_to_world = g_cam["Extrinsic_cam_to_world"][:]
    extrinsic_world_to_cam = g_cam["Extrinsic_world_to_cam"][:]
    print("\nCamera:")
    print(" rgb:", rgb.shape, rgb.dtype)
    print(" depth:", depth.shape, depth.dtype)
    print(" seg_mask:", seg_mask.shape, seg_mask.dtype)
    print(" intrinsic:\n", intrinsic)
    print(" extrinsic_world_to_cam:\n", extrinsic_world_to_cam)

    # ---- Partial cloud ----
    g_part = root["partial"]
    partial_points = g_part["points_world"][:]  # (N,3) float32
    partial_colors = g_part["colors_rgb"][:]    # (N,3) uint8
    partial_mask = g_part["instance_mask"][:]   # (N,) int32
    print("\nPartial cloud:")
    print(" points:", partial_points.shape, partial_points.dtype)
    print(" colors:", partial_colors.shape, partial_colors.dtype)
    print(" mask:", partial_mask.shape, partial_mask.dtype)

    # ---- Complete cloud ----
    g_comp = root["complete"]
    complete_points = g_comp["points_world"][:]    # (M,3)
    complete_mask = g_comp["instance_mask"][:]
    print("\nComplete cloud:")
    print(" points:", complete_points.shape, complete_points.dtype)
    print(" mask:", complete_mask.shape, complete_mask.dtype)

    # ---- Objects ----
    g_objs = root["objects"]
    body_ids = g_objs["body_ids"][:]
    print("\nObjects:")
    print(" body_ids:", body_ids)

    for obj_name, g in g_objs.items():
        if not obj_name.isdigit():
            continue  # skip "body_ids"
        print(f" Object {obj_name}:")
        for k, v in g.attrs.items():
            print(f"   {k}: {v}")
        print("   center_of_mass:", g["center_of_mass"][:])
        print("   position:", g["position"][:])
        print("   orientation_xyzw:", g["orientation_xyzw"][:])
        print("   position_mesh_origin:", g["position_mesh_origin"][:])
        print("   orientation_xyzw_mesh_origin:", g["orientation_xyzw_mesh_origin"][:])


    return {
        "metadata_camera": (width, height, cam_position, fov_y_deg, target_look_at, cam_up_direction, near, far), 
        "camera": (rgb, depth, seg_mask, intrinsic, extrinsic_world_to_cam, extrinsic_cam_to_world),
        "partial": (partial_points, partial_colors, partial_mask),
        "complete": (complete_points, complete_mask),
        "objects": objs_summary(g_objs)
    }


def objs_summary(g_objs):
    objs = []
    for obj_name, g in g_objs.items():
        if not obj_name.isdigit():
            continue
        objs.append({
            "bid": g.attrs["bid"],
            "mass": g.attrs["mass"],
            "friction": g.attrs["friction"],
            "render_mesh": g.attrs["render_mesh"],
            "collision_mesh": g.attrs["collision_mesh"],
            "position": g["position"][:],
            "orientation": g["orientation_xyzw"][:],
            "position_mesh_origin": g["position_mesh_origin"][:],
            "orientation_xyzw_mesh_origin": g["orientation_xyzw_mesh_origin"][:],
        })
    return objs

if __name__ == "__main__":
    zarr_path = "/mnt/slurmfs-4090node1/homes/txiang031/phy_recon/dataset/easy_case1/scene.zarr"  
    data = read_scene_zarr(zarr_path)

    rgb, depth, seg_mask, intrinsic, extrinsic_world_to_cam, extrinsic_cam_to_world = data.get("camera", (None,)*6)
    camera_data = data.get("camera", (None,)*6)
        
    meta_data =  data.get("metadata_camera", (None,)*7)
    print(meta_data)
    # cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()