import argparse
import os
import json
import trimesh
from geometry import object_3D
from geometry import scene_graph_3D
from geometry import geometry_optim
from hierarchical_physics import diff_hierarchical_physics
from ultility import seed_everything


def export_optim_result(objects: object_3D, save_path = "optimized_path"):
    """
    Export objects to a folder. 
    Include a glb file for visualization, and a json to store data
    """
    json_dir  = f"{save_path}/result.json"
    scene_glb = f"{save_path}/scene.glb"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    scene = trimesh.Scene()
    for obj in objects:
        scene.add_geometry(obj.trimesh_obj)
    scene.export(scene_glb)
    print(f"[OK] Visualized scene path: {os.path.abspath(scene_glb)}")

    data = [obj.to_export_dict() for obj in objects]
    with open(json_dir, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Export {len(objects)} instances to {os.path.abspath(json_dir)}")



if __name__ == "__main__":

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
        "--debug",
        action="store_true",
        help="Whether to debug the scene"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize the diffsim optimization"
    )

    args = parser.parse_args()
    ## Input data paths
    dataset_path = f"./dataset/{args.dataset}"
    mesh_dataset_path = f"{dataset_path}/ICP_refinement"
    ## Output data paths
    initial_guess_result_path   = f"{dataset_path}/initial_guess"
    geom_optim_result_path      = f"{dataset_path}/geom_optim"
    physics_optim_result_path   = f"{dataset_path}/physics_optim"

    ## Debug setting
    debug_scene = args.debug
    ## visualize setting
    debug_sim = args.visualize

    recorder_path = f"{dataset_path}/diff_worlds"
    
    """    
    Loading reconstructed meshes
    """    
    objects_path = []
    objects_list = []
    ## Create a box as ground
    size = (1.5, 1.5, 0.5)
    ground = trimesh.creation.box(extents=size)
    ground.apply_translation([0, 0, -0.25])
    ground_obj = object_3D(mesh_path= None, 
                           type="ground", obj_ID= 0, 
                           custom_trimesh=True, trimesh_obj=ground,
                           compute_convex_decomposition= False)
    objects_list.append(ground_obj)
    # ground_obj.visualize_faces_opposite_to_gravity()

    for i in range(1,100):
        if not os.path.exists(f"{mesh_dataset_path}/obj{i}/obj{i}.obj"):
            break
        print(f"Loading obj{i}.obj from: {mesh_dataset_path}/obj{i}/obj{i}.obj")
        transformed_reconstructed_mesh_path = f"{mesh_dataset_path}/obj{i}/obj{i}.obj"
        objects_path.append(transformed_reconstructed_mesh_path)

        ## The convex decomposition is only needed for evaluation
        ## Our optimization invole no convex decomposition.
        obj = object_3D(transformed_reconstructed_mesh_path, 
                        type="object", obj_ID= i,
                        compute_convex_decomposition= False)
        objects_list.append(obj)    
        # obj.visualize_faces_opposite_to_gravity()

    """
    Initial guess: build scene graph via SAM3D+ICP
    """ 
    ## Build scene graph
    scene_graph = scene_graph_3D(objects_list)
    parent_map = scene_graph.get_scene_graph()
    if debug_scene:
        scene_graph.visualize_scene_graph(show_meshes= True)
        scene_graph.visualize()
    print(f"parent_map: {parent_map}\n")

    ## Output initial guess scene of amodal 3D and shape registration
    export_optim_result(objects_list, initial_guess_result_path)
    print(f"\n [Result export]: Initial guess scene is exported to {initial_guess_result_path}")

    """
    Stage 1: Geometry optimization only
    """ 
    geometry_optimizer = geometry_optim(objects_list, parent_map= parent_map)
    result = geometry_optimizer.constrain_objects()
    result_quaternions, result_poses, loss = result
    print(f"[geometry constrain]: Loss = {loss}")    

    export_optim_result(objects_list, geom_optim_result_path)
    print(f"\n [Result export]: Stage 1 geomtry optimization result is exported to {geom_optim_result_path}")
    if debug_scene:
        scene_graph.visualize()

    ## Do subdivide to ground
    objects_list[0].subdivide_object()

    """
    Stage 2: Hierarchical physics-constrained optimization + geometry optimization
    """ 
    ## TODO: Markov chain optimization
    _diff_hierarchical_physics = diff_hierarchical_physics(object_3D_list= objects_list, parent_map= parent_map)
    schedule = _diff_hierarchical_physics.optimize_world(debug_sim= debug_sim, recorder_path = recorder_path)

    export_optim_result(objects_list, physics_optim_result_path)
    print(f"\n [Result export]: Stage 2 physics constrained optimization result is exported to {physics_optim_result_path}")

    ## TODO: Somehow trimesh viewer is disabled after using DIFFSDFSIM recorder
    if debug_scene:
        scene_graph.visualize()



