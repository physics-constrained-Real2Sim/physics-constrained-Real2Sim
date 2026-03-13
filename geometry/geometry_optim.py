import torch
import yaml
import numpy as np
from pathlib import Path
from icecream import ic
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix

from .object_3D import object_3D
from .scene_graph import scene_graph_3D

class Defaults:
    ## learning parameters
    yaml_path = f"./configs/geometry_optim.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        args_configs = yaml.safe_load(f) or {}
    
    ## learning rate and iteration
    lr_quaternion = float(args_configs["lr_quaternion"])
    lr_pose     = float(args_configs["lr_pose"])
    iteration   = int(args_configs["iteration"])
    lambda_xy   = float(args_configs["lambda_xy"])
    lambda_z    = float(args_configs["lambda_z"])
    ## penetration & contact margin
    penetration_margin  = float(args_configs["penetration_margin"]) # safe margin, this will set a margin around objects.
    contact_margin      = float(args_configs["contact_margin"]) # safe margin for contact detection, this will set a margin around objects
    penetration_weight  = int(args_configs["penetration_weight"])
    contact_weight      = int(args_configs["contact_weight"])


class geometry_optim:
    def __init__(self, object_3D_list: object_3D, parent_map = {}):
        self.object_3D_list = object_3D_list
        # self.scene_graph = scene_graph_3D(object_3D_list)
        self.parent_map = parent_map

        self.debug = True

        self.result_quats = None 
        self.result_poses = None
        self.best_loss  = None

        assert all(isinstance(o, object_3D) for o in self.object_3D_list)

    def heuristic_initialization(self, margin = 2e-3):
        ## For all objs, heuristic remove all inter-penetration with ground by lifting up their z-axis 
        for i in range(len(self.object_3D_list)):
            if self.object_3D_list[i].type == "ground":
                continue
            else:
                obj = self.object_3D_list[i]
                lift_z = max(-obj.vertices[:,2].min(),0)
                obj.trimesh_apply_transform(quaternion = [1.0, 0.0, 0.0, 0.0], pose = [0.0, 0.0, lift_z])

    
    def constrain_objects(self, lr_quaternion = Defaults.lr_quaternion,  lr_pose = Defaults.lr_pose, iteration = Defaults.iteration):
        ## use geometry constraints to objects based on scene graph
        ## return poses of objects by the constrain

        ## heuristic initialization before optimization, this can help to speed up convergence and avoid bad local minima
        print(f"[Heuristic initialization]: Start")
        self.heuristic_initialization()
        print(f"[Heuristic initialization]: Done\n")

        ## SDF-based geometry optimization
        print(f"[Geometry optimization]: Start")
        optim_poses         = []
        optim_quaternions   = []
        for i in range(len(self.object_3D_list)):
            if self.object_3D_list[i].type == "ground":
                continue
            else:
                optim_quaternion = torch.tensor(np.array([1.0, 0.0, 0.0, 0.0]), device= "cuda", requires_grad= True)
                optim_pose = torch.tensor(np.array([0.0, 0.0, 0.0]), device= "cuda", requires_grad= True)

                optim_quaternions.append(optim_quaternion)
                optim_poses.append(optim_pose)
        quaternion_optimizer = torch.optim.Adam(optim_quaternions, lr = lr_quaternion)
        poses_optimizer = torch.optim.Adam(optim_poses, lr= lr_pose)

        best_loss = float("inf")
        best_quats = None
        best_poses = None

        prev_loss = None
        for it in range(iteration):
            poses_optimizer.zero_grad()
            quaternion_optimizer.zero_grad()
            loss = self.run_geometry_world(quaternions= optim_quaternions, poses= optim_poses)
            loss.backward()
            poses_optimizer.step()
            quaternion_optimizer.step()
            # poses_scheduler.step()
            loss_val = float(loss.item())

            # Normalize all quaternions after update
            with torch.no_grad():
                for q in optim_quaternions:
                    q /= q.norm() + 1e-8

            if self.debug and (it % 1 == 0):
                print(f"[it {it}] loss={loss_val}")\
                
            if loss_val < best_loss:
                best_loss = loss_val
                best_quats = [q.detach().clone() for q in optim_quaternions]
                best_poses = [p.detach().clone() for p in optim_poses]


        # Return CPU numpy arrays
        self.result_quats = [q.cpu().numpy() for q in best_quats]
        self.result_poses = [p.cpu().numpy() for p in best_poses]
        self.best_loss  = float(best_loss)

        ## apply the best result to the objects
        self.apply_geometry_optim_result(self.result_quats, self.result_poses)

        return self.result_quats, self.result_poses, self.best_loss
    
    def apply_geometry_optim_result(self, quaternions, poses):
        ## TODO: Move apply result into geometry class
        non_ground_idx = 0
        for idx, obj in enumerate(self.object_3D_list):
            if obj.type == "ground":
                # For ground object, use original mesh without transformation
                continue
            else:
                ## apply geoemtry constrain before diff simulation
                obj.trimesh_apply_transform(quaternion = quaternions[non_ground_idx], pose = poses[non_ground_idx] )
                non_ground_idx += 1
        print(f"[Apply geometry optimization result]: Geometry optimization result is applied to objects in the scene")


    def run_geometry_world(self, quaternions, poses, 
                           lambda_xy = Defaults.lambda_xy, 
                           lambda_z = Defaults.lambda_z,
                           penetration_margin = Defaults.penetration_margin, 
                           contact_margin = Defaults.contact_margin ,
                           penetration_weight = Defaults.penetration_weight, 
                           contact_weight = Defaults.contact_weight
                           ):
        ## make geometry world
        total_loss = 0.0
        ## create ground
        self.object_3D_list[0].create_kal_mesh()
        ## create objects
        for idx in range(len(poses)):
            self.object_3D_list[idx + 1].create_kal_mesh(quaternions[idx], poses[idx])

        id2obj = {o.obj_ID: o for o in self.object_3D_list}
        parent_of = self.parent_map  # key: child_id -> parent_id
        # 2)broad phase 
        neighbor_map = {o.obj_ID: set() for o in self.object_3D_list}
        visited_pairs = set()  # frozenset({idA, idB})

        for obj in self.object_3D_list:
            cand = self.broad_phase_collision_detection(obj)  
            for other in cand:
                if other is obj:
                    continue
                a, b = sorted([obj.obj_ID, other.obj_ID])
                pair = frozenset((a, b))
                if pair in visited_pairs:
                    # 已经登记过这对，跳过
                    continue
                visited_pairs.add(pair)
                neighbor_map[a].add(b)
                neighbor_map[b].add(a)

        # 3) Force to invovle parent object into candidates     
        for child_id, parent_id in parent_of.items():
            if parent_id in id2obj:
                neighbor_map.setdefault(child_id, set()).add(parent_id)
                neighbor_map.setdefault(parent_id, set()).add(child_id)
                visited_pairs.add(frozenset((min(child_id, parent_id), max(child_id, parent_id))))

        # 4) Traverse all pair： penetration loss + contact loss(if share parent-object relation)
        for pair in visited_pairs:
            idA, idB = tuple(pair)
            A, B = id2obj[idA], id2obj[idB]

            # Penetration loss (A is B's parent) or (B is A's parent)
            if parent_of.get(idA) == idB or parent_of.get(idB) == idA:
                contact_loss = self.penetration_and_contact_loss_pair(A, B, safe_margin= penetration_margin, 
                                                                      contact_margin= contact_margin,
                                                                      penetration_weight= penetration_weight,
                                                                      contact_weight= contact_weight)
                total_loss += contact_loss
            else:
                # penetration loss (symmetric, only count once)
                penetration_loss = self.penetration_loss_pair(A, B, safe_margin= penetration_margin,
                                                              penetration_weight= penetration_weight)
                total_loss += penetration_loss

        if lambda_xy > 0:
            xy_reg = 0.0
            for p in poses:
                xy_reg = xy_reg + (p[:2]**2).mean()
            total_loss = total_loss + lambda_xy * xy_reg
        if lambda_z > 0:
            z_reg = 0.0
            for p in poses:
                z_reg = z_reg + (p[2]**2).mean()
            total_loss = total_loss + lambda_z * z_reg

        return total_loss


    @staticmethod
    def penetration_loss_pair(obj_A : object_3D, obj_B : object_3D, safe_margin = 0.0, penetration_weight = 4):
        ## compute the penetration loss to all objects in the scene, use broad phase first 
        surface_pointes = obj_A.sample_surface_points(num_samples= 2000)
        distance = obj_B.compute_sdf_from_points(points= surface_pointes)
        # print("[penetration only:]")

        # ic(distance.min())
        ## penetration loss
        penetration_violation = F.relu(-(distance - safe_margin))
        viols = penetration_violation
        num_viol = (viols > 0).sum()
        if num_viol > 0:
            loss_penetration = penetration_weight * viols[viols > 0].mean() 
        else:
            loss_penetration = distance.new_tensor(0.0)
        # ic(loss_penetration)
        return loss_penetration
    
    
    @staticmethod
    def penetration_and_contact_loss_pair(obj_A : object_3D, obj_B : object_3D, safe_margin = 0.0, contact_margin= 8e-3,
                                          penetration_weight =4.0, contact_weight = 1.0, topk_k = 20):
        ## compute penetration and contact loss
        surface_pointes = obj_A.sample_surface_points(num_samples= 2000)
        distance = obj_B.compute_sdf_from_points(points= surface_pointes)
        # print("[penetration and contact:]")
        # ic(distance.shape)
        # ic(distance.min())
        ## penetration loss
        penetration_violation = F.relu(-(distance - safe_margin))
        viols = penetration_violation
        num_viol = (viols > 0).sum()
        if num_viol > 0:
            loss_penetration = viols[viols > 0].mean()
        else:
            loss_penetration = distance.new_tensor(0.0)
        # print(f"penetration loss between pair: {obj_A.obj_ID} and {obj_B.obj_ID} ", )
        # ic(loss_penetration)
        # ic(penetration_violation.max())
        # ---- Top-K contact loss ----
        # use the average of the K smallest SDFs; require it <= contact_margin
        N = distance.numel()
        k = max(1, min(topk_k, N))
        # largest=False -> take smallest distances
        topk_vals, _ = torch.topk(distance, k, largest=False)
        mean_topk = topk_vals.mean()
        loss_contact = F.relu(mean_topk - contact_margin)
        # ic(loss_contact)
        loss = penetration_weight * loss_penetration + contact_weight * loss_contact 
        return loss

    
    def broad_phase_collision_detection(self, object_3D_instance: object_3D, margin: float = 0.02):
        """
        Broad-phase collision detection using Axis-Aligned Bounding Boxes (AABB).
        Expands each AABB slightly by `margin` to make detection more tolerant.

        Args:
            object_3D_instance (object_3D): The target object to test collisions for.
            margin (float): The amount to expand each AABB in every direction.

        Returns:
            List[object_3D]: List of candidate objects that may collide.
        """
        candidates = []

        # Update or get the AABB of the reference object
        aabb1_min, aabb1_max = object_3D_instance.get_kaolin_mesh_AABB()
        # Expand by margin
        aabb1_min = aabb1_min - margin
        aabb1_max = aabb1_max + margin

        for other in self.object_3D_list:
            if other is object_3D_instance:
                continue

            aabb2_min, aabb2_max = other.get_kaolin_mesh_AABB()
            aabb2_min = aabb2_min - margin
            aabb2_max = aabb2_max + margin

            overlap = True
            for i in range(3):
                if aabb1_max[i] < aabb2_min[i] or aabb2_max[i] < aabb1_min[i]:
                    overlap = False
                    break

            if overlap:
                candidates.append(other)

        return candidates


class node_geometry_optim():
    def __init__(self, node_object: object_3D, fixed_objects: list, parent_map : dict):
        
        self.node_object = node_object
        self.parent_map = parent_map
        self.parent_ID = parent_map[node_object.obj_ID]

        self.fixed_objects = fixed_objects
            
    def run_geometry_world(self, quaternion, pose, 
                           lambda_xy = Defaults.lambda_xy, 
                           lambda_z = Defaults.lambda_z,
                           penetration_margin = Defaults.penetration_margin, 
                           contact_margin = Defaults.contact_margin ,
                           penetration_weight = Defaults.penetration_weight, 
                           contact_weight = Defaults.contact_weight):
        ## make geometry world
        total_loss = 0.0
        ## create all fixed objects without grad
        for idx in range(len(self.fixed_objects)):
            self.fixed_objects[idx].create_kal_mesh()
        
        ## create node object with input poses

        self.node_object.create_kal_mesh(quaternion, pose)

        id2obj = {o.obj_ID: o for o in self.fixed_objects}
        # If the moving node itself is in fixed_objects, ensure it's present in lookup too
        id2obj[self.node_object.obj_ID] = self.node_object


        candidates = self.broad_phase_collision_detection(self.node_object)  

        # --- Force-insert parent if present but not in broad-phase results
        parent_obj = id2obj.get(self.parent_ID, None) if self.parent_ID is not None else None
        if parent_obj is not None and parent_obj not in candidates:
            candidates.append(parent_obj)

        for cand in candidates:
            if cand is self.node_object:
                continue

            if self.parent_map.get(self.node_object.obj_ID, None) == cand.obj_ID:
                contact_loss = self.penetration_and_contact_loss_pair(self.node_object, cand, safe_margin= penetration_margin, 
                                                                      contact_margin= contact_margin,
                                                                      penetration_weight= penetration_weight,
                                                                      contact_weight= contact_weight)
                total_loss += contact_loss
            else:
                penetration_loss = self.penetration_loss_pair(self.node_object, cand, safe_margin= penetration_margin,
                                                              penetration_weight= penetration_weight)
                total_loss += penetration_loss

        if lambda_xy > 0:
            xy_reg = 0.0
            xy_reg = xy_reg + (pose[:2]**2).mean()
            total_loss = total_loss + lambda_xy * xy_reg
        
        if lambda_z > 0:
            z_reg = 0.0
            z_reg = z_reg + (pose[2]**2).mean()
            total_loss = total_loss + lambda_z * z_reg

        return total_loss


    @staticmethod
    def penetration_loss_pair(obj_A : object_3D, obj_B : object_3D, safe_margin = 0.0, penetration_weight = 4):
        ## compute the penetration loss to all objects in the scene, use broad phase first 
        surface_pointes = obj_A.sample_surface_points(num_samples= 2000)
        distance = obj_B.compute_sdf_from_points(points= surface_pointes)

        ## penetration loss
        penetration_violation = F.relu(-(distance - safe_margin))
        viols = penetration_violation
        num_viol = (viols > 0).sum()
        if num_viol > 0:
            loss_penetration = penetration_weight * viols[viols > 0].mean()
        else:
            loss_penetration = distance.new_tensor(0.0)
        return loss_penetration
    

    @staticmethod
    def penetration_and_contact_loss_pair(obj_A : object_3D, obj_B : object_3D, safe_margin = 0.0, contact_margin= 8e-3,
                                          penetration_weight =4.0, contact_weight = 1.0, topk_k = 20):
        ## compute penetration and contact loss
        surface_pointes = obj_A.sample_surface_points(num_samples= 2000)
        distance = obj_B.compute_sdf_from_points(points= surface_pointes)
        # print("[penetration and contact:]")
        # ic(distance.shape)
        # ic(distance.min())
        ## penetration loss
        penetration_violation = F.relu(-(distance - safe_margin))
        viols = penetration_violation
        num_viol = (viols > 0).sum()
        if num_viol > 0:
            loss_penetration = viols[viols > 0].mean()
        else:
            loss_penetration = distance.new_tensor(0.0)
        # print(f"penetration loss between pair: {obj_A.obj_ID} and {obj_B.obj_ID} ", )
        # ic(loss_penetration)
        # ic(penetration_violation.max())
        # ---- Top-K contact loss ----
        # use the average of the K smallest SDFs; require it <= contact_margin
        N = distance.numel()
        k = max(1, min(topk_k, N))
        # largest=False -> take smallest distances
        topk_vals, _ = torch.topk(distance, k, largest=False)
        mean_topk = topk_vals.mean()
        loss_contact = F.relu(mean_topk - contact_margin)
        # ic(loss_contact)
        loss = penetration_weight * loss_penetration + contact_weight * loss_contact 
        return loss

    def broad_phase_collision_detection(self, object_3D_instance: object_3D, margin: float = 0.02):
        """
        Broad-phase collision detection using Axis-Aligned Bounding Boxes (AABB).
        Expands each AABB slightly by `margin` to make detection more tolerant.

        Args:
            object_3D_instance (object_3D): The target object to test collisions for.
            margin (float): The amount to expand each AABB in every direction.

        Returns:
            List[object_3D]: List of candidate objects that may collide.
        """
        candidates = []

        # Update or get the AABB of the reference object
        aabb1_min, aabb1_max = object_3D_instance.get_kaolin_mesh_AABB()
        # Expand by margin
        aabb1_min = aabb1_min - margin
        aabb1_max = aabb1_max + margin

        for other in self.fixed_objects:
            if other is object_3D_instance:
                continue

            aabb2_min, aabb2_max = other.get_kaolin_mesh_AABB()
            aabb2_min = aabb2_min - margin
            aabb2_max = aabb2_max + margin

            overlap = True
            for i in range(3):
                if aabb1_max[i] < aabb2_min[i] or aabb2_max[i] < aabb1_min[i]:
                    overlap = False
                    break

            if overlap:
                candidates.append(other)

        return candidates




