import os
import yaml
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from icecream import ic
from pathlib import Path

from geometry import object_3D
from geometry import scene_graph_3D
from geometry import geometry_optim, node_geometry_optim
from diffsdfsim import node_physics_optim, Defaults3D

class diff_hierarchical_physics():
    def __init__(self, object_3D_list: list, parent_map: dict, parameter_path: str = "./configs/physics_optim.yaml"):
        """
        object_3D_list: list[object_3D]
        parent_map: {child_id: parent_id}, with ground node == 0
        parameter_path: path to YAML file containing optimization parameters
        """
        self.object_3D_list = object_3D_list
        assert all(isinstance(o, object_3D) for o in self.object_3D_list)

        # Build id <-> object mapping.
        # Assumes each object has a unique .id attribute.
        # If your data uses list indices as ids, swap to enumerate().
        self.id_to_obj = {}
        for o in self.object_3D_list:
            if not hasattr(o, "obj_ID"):
                raise ValueError("Each object_3D must have an 'id' attribute.")
            self.id_to_obj[o.obj_ID] = o

        self.parent_map = dict(parent_map)  # copy
        if 0 not in self.id_to_obj:
            raise ValueError("Ground object with id 0 must exist in object_3D_list.")
        if any(cid not in self.id_to_obj for cid in self.parent_map.keys()):
            missing = [cid for cid in self.parent_map if cid not in self.id_to_obj]
            raise ValueError(f"Parent map references unknown child ids: {missing}")
        if any(pid not in self.id_to_obj for pid in self.parent_map.values()):
            missing = [pid for pid in self.parent_map.values() if pid not in self.id_to_obj]
            raise ValueError(f"Parent map references unknown parent ids: {missing}")

        # Build graph structures
        self.children_map = self._build_children_map(self.parent_map)
        self.levels = self._compute_levels_bfs(root_id=0, children_map=self.children_map)

        # Sanity: detect cycles / unreachable nodes
        self._validate_dag_and_reachability()
        self._initialize_parameters(parameter_path = parameter_path)

    def _build_children_map(self, parent_map: dict) -> dict:
        """parent_map: {child: parent} -> children_map: {parent: [child,...]}"""
        children_map = {nid: [] for nid in self.id_to_obj.keys()}
        for child, parent in parent_map.items():
            children_map[parent].append(child)
        # stable deterministic order
        for k in children_map:
            children_map[k].sort()
        return children_map

    def _compute_levels_bfs(self, root_id: int, children_map: dict) -> dict:
        """Return dict node_id -> level (root at level 0)."""
        from collections import deque
        level = {root_id: 0}
        q = deque([root_id])
        while q:
            u = q.popleft()
            for v in children_map.get(u, []):
                if v in level:
                    # already assigned -> potential multiple parents or back-edge
                    continue
                level[v] = level[u] + 1
                q.append(v)
        return level

    def _validate_dag_and_reachability(self):
        """Light checks: every mapped child must have parent chain to 0; no cycles via DFS."""
        # Reachability: any node in parent_map but not in levels wasn't reached from 0
        unreachable = [n for n in self.parent_map.keys() if n not in self.levels]
        if unreachable:
            # Not fatal—just warn; we'll skip these during optimization.
            ic(f"[warn] Unreachable from ground (skipped): {unreachable}")

        # Cycle detection via DFS on parent pointers
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self.id_to_obj}
        def dfs(u):
            color[u] = GRAY
            for v in self.children_map.get(u, []):
                if color[v] == GRAY:
                    raise ValueError(f"Cycle detected involving node {v}")
                if color[v] == WHITE:
                    dfs(v)
            color[u] = BLACK
        dfs(0)

    def get_level_order(self) -> list[tuple[int, list[int]]]:
        """
        Returns a list of (level, [node_ids]) sorted by level ascending and ids ascending.
        Level 0 is just the ground (0).
        """
        from collections import defaultdict
        buckets = defaultdict(list)
        for nid, lvl in self.levels.items():
            buckets[lvl].append(nid)
        order = []
        for lvl in sorted(buckets.keys()):
            order.append((lvl, sorted(buckets[lvl])))
        return order

    def get_optimization_schedule(self) -> list[int]:
        """
        Flattened schedule of node ids in the order they'll be optimized.
        - Level 0 (ground) is considered fixed and NOT optimized here.
        - Within each level L>0, order is ascending by id for determinism.
        """
        schedule = []
        for lvl, nodes in self.get_level_order():
            if lvl == 0:
                continue  # ground is fixed
            schedule.extend(nodes)
        return schedule

    # ---------- optimization driver (logic only) ----------

    def optimize_world(self, debug_sim = False, recorder_path = "./"):
        """
        Logic:
        - Ground (0) is fixed from the start.
        - Proceed level by level (BFS from ground).
        - When optimizing a node at some level, FIX the set:
            {ground} ∪ {all nodes already optimized at previous levels} ∪ {earlier siblings already optimized at this same level}
        - Call a placeholder `self._optimize_node(node_id, fixed_ids)` that you can implement.
        """
        ground_id = 0
        optimized = set([ground_id])   # ground is considered fixed/optimized
        schedule_by_level = self.get_level_order()

        ic(schedule_by_level)
        for lvl, nodes in tqdm(schedule_by_level, desc= "Level"):
            if lvl == 0:
                continue  # nothing to do for ground

            # deterministic order within level (already sorted)
            for nid in tqdm(nodes, desc= f"Nodes in level {lvl}"):
                fixed_ids = set(optimized)  # ground + all previously optimized
                # NOTE: siblings earlier in this loop are already in `optimized`,
                # so they are fixed; later siblings are not.
                self._optimize_node(nid, fixed_ids=fixed_ids,
                                    debug_sim= debug_sim, recorder_path= recorder_path)
                optimized.add(nid)

        # Optionally return the final order for inspection/logging
        return self.get_optimization_schedule()

    def _initialize_parameters(self, parameter_path = "./configs/physics_optim.yaml"):
        ## TODO: move learning rate to config files
        with open(parameter_path,"r") as stream:
            self.args_configs = yaml.safe_load(stream)
        
        print(self.args_configs)
        print(self.args_configs["lr_quaternion"])
        # learning rates
        self.lr_quaternion  = float(self.args_configs["lr_quaternion"])
        self.lr_pose_warmup = float(self.args_configs["lr_pose_warmup"])
        self.lr_pose        = float(self.args_configs["lr_pose"])
        self.lr_com         = float(self.args_configs["lr_com"])
        self.lr_physics     = float(self.args_configs["lr_physics"])
        ## voxel resolution for contact detection in physics optimization, 
        # this will affect the smoothness of the loss landscape and the optimization result.
        self.voxel_resolution = self.args_configs["voxel_resolution"]
        ## warmup settings
        self.penetration_weight_warmup = self.args_configs["penetration_weight_warmup"]
        self.contact_weight_warmup     = self.args_configs["contact_weight_warmup"]
        #diffsim parameters
        self.penetration_weight  = self.args_configs["penetration_weight"]
        self.contact_weight      = self.args_configs["contact_weight"]
        self.lambda_xy_warmup    = self.args_configs["lambda_xy_warmup"]
        self.lambda_z_warmup     = self.args_configs["lambda_z_warmup"]
        self.lambda_xy           = self.args_configs["lambda_xy"]
        self.lambda_z            = self.args_configs["lambda_z"]
        self.rollout_time        = self.args_configs["rollout_time"]


    def _optimize_node(self, node_id: int, fixed_ids: set[int], 
                       debug_sim = False, recorder_path = "./"):
        """
        Placeholder for your actual optimization.
        Here you would:
          - gather the variable object (self.id_to_obj[node_id])
          - gather fixed objects from `fixed_ids`
          - call your optimizer (e.g., geometry_optim / diff_base_world)
          - write back the optimized state into self.id_to_obj[node_id]
        """
        print(f" \n [Markov chain]: Optimizing node {node_id} with fixed {sorted(fixed_ids)} \n")
        ## At each node:
        ## Fix the fixed ids objects, 
        # loss = stable loss  + touching loss

        ## store the best result
        best = {
            "loss": float("inf"),
            "stable_loss": float("inf"),
            "touching_loss": float("inf"),
            "it": -1,
            "quaternion": None,
            "pose": None,
            "com": None,
            "mass": None,
            "friction": None,
        }

        # Object to be optimized
        var_obj = self.id_to_obj[node_id]
        fixed_objs = [self.id_to_obj[i] for i in sorted(fixed_ids)]

        optim_quaternion = torch.tensor(np.array([1.0, 0.0, 0.0, 0.0]),
                                        device= "cuda", 
                                        requires_grad= True)
        optim_pose  = torch.tensor(np.array([0.0, 0.0, 0.0]), 
                                    device= "cuda", 
                                    requires_grad= True)
        com         = torch.tensor(np.array([0.0, 0.0, 0.0]), 
                                    dtype=Defaults3D.DTYPE, 
                                    device=Defaults3D.DEVICE,
                                    requires_grad= True)        
        mass        = torch.tensor(np.array(var_obj.mass),
                                    dtype=Defaults3D.DTYPE, 
                                    device=Defaults3D.DEVICE,
                                    requires_grad= True)
        friction    = torch.tensor(np.array(var_obj.friction),                                 
                                    dtype=Defaults3D.DTYPE, 
                                    device=Defaults3D.DEVICE,                                
                                    requires_grad= True)
        quaternion_optimizer    = torch.optim.Adam([optim_quaternion], lr = self.lr_quaternion)
        pose_optimizer          = torch.optim.Adam([optim_pose], lr= self.lr_pose_warmup)
        com_optimizer           = torch.optim.Adam([com], lr = self.lr_com)
        physics_optimizer       = torch.optim.Adam([friction, mass], lr = self.lr_physics)

        ## create node geometry optimizer
        _node_geom_optimizer = node_geometry_optim(node_object= var_obj, 
                            fixed_objects= fixed_objs, parent_map= self.parent_map)
        ## create node physics optimizer
        _node_physics_optimizer = node_physics_optim(node_object= var_obj, 
                            fixed_objects= fixed_objs, voxel_resolution= self.voxel_resolution)
        
        ## TODO: Warm start with geometry optimization to solve penetration only with high regularization
        penetration_loss = _node_geom_optimizer.run_geometry_world(quaternion = optim_quaternion,
                                                                    pose  = optim_pose,
                                                                    lambda_xy = self.lambda_xy_warmup,
                                                                    lambda_z = self.lambda_z_warmup,
                                                                    penetration_margin = 3e-3,
                                                                    contact_margin = 3e-3,
                                                                    penetration_weight = self.penetration_weight_warmup,
                                                                    contact_weight = self.contact_weight_warmup)
        ic(f"penetration loss at beginning: {penetration_loss.item()}")
        it = 0
        while penetration_loss.item() > 0 and it < 200:
            print("[Warning]: penetration found at initial simulation")   
            print("[Warm-up] penetration present, iter", it)
            quaternion_optimizer.zero_grad()
            pose_optimizer.zero_grad()
            penetration_loss = _node_geom_optimizer.run_geometry_world(quaternion = optim_quaternion,
                                                                    pose  = optim_pose,
                                                                    lambda_xy = self.lambda_xy_warmup,
                                                                    lambda_z = self.lambda_z_warmup,
                                                                    penetration_margin = 3e-3,
                                                                    contact_margin = 3e-3,
                                                                    penetration_weight = self.penetration_weight_warmup,
                                                                    contact_weight = self.contact_weight_warmup)
            penetration_loss.backward()
            quaternion_optimizer.step()
            pose_optimizer.step()

            with torch.no_grad():
                optim_quaternion /= (optim_quaternion.norm() + 1e-12)
            it += 1

        it = 0
        pose_optimizer = torch.optim.Adam([optim_pose], lr= self.lr_pose)
        for it in tqdm(range(15), desc= f"Node {node_id} optimization"):
            # zero gradient
            quaternion_optimizer.zero_grad()
            pose_optimizer.zero_grad()
            com_optimizer.zero_grad()
            physics_optimizer.zero_grad()
            # geometry world
            touching_loss = _node_geom_optimizer.run_geometry_world(quaternion= optim_quaternion,
                                                                    pose  = optim_pose,
                                                                    lambda_xy= self.lambda_xy,
                                                                    lambda_z= self.lambda_z,
                                                                    penetration_margin= 1e-3,
                                                                    contact_margin= 1e-3,
                                                                    penetration_weight= self.penetration_weight,
                                                                    contact_weight= self.contact_weight)
            ## TODO: Need to hook the gradient of pose of z that come from physics optimizer
            pose_for_physics = optim_pose.clone()
            pose_for_physics[2] = optim_pose[2].detach()

            world = _node_physics_optimizer.make_pair_diff_world(node_pos= pose_for_physics,
                                                                 node_COM= com,
                                                                 node_mass = mass,
                                                                 node_friction= friction,
                                                                 debug_mesh= True)
            ## If need DIFFSDFSIM recorder
            debug_path = f"{recorder_path}/node{node_id}/it{it}"
            scene    = _node_physics_optimizer.make_scene() if debug_sim else None
            recorder = _node_physics_optimizer.make_recorder(scene = scene, path = debug_path) if debug_sim else None

            stable_loss = _node_physics_optimizer.run_world(world= world, 
                                                            scene = scene, 
                                                            recorder= recorder, 
                                                            TIME = self.rollout_time) ##TODO: Do we need 0.2s, 25 ts?
            loss = stable_loss + 5 * touching_loss

            # ---- record best (use the loss from *this* forward pass) ----
            loss_val = float(loss.item())
            if loss_val < best["loss"]:
                best.update(
                    {
                        "loss": loss_val,
                        "stable_loss": float(stable_loss.detach().item()),
                        "touching_loss": float(touching_loss.detach().item()),
                        "it": it,
                        "quaternion": optim_quaternion.detach().clone(),
                        "pose": optim_pose.detach().clone(),
                        "com": com.detach().clone(),
                        "mass": mass.detach().clone(),
                        "friction": friction.detach().clone(),
                    }
                )


            loss.backward()
            quaternion_optimizer.step()
            pose_optimizer.step()
            com_optimizer.step()
            physics_optimizer.step()

            ## Implement Mass COM friction constrain
            with torch.no_grad():
                com.data.clamp_(-var_obj.body_half_extents * 0.3,
                                 var_obj.body_half_extents * 0.3)
                mass.data.clamp_(var_obj.mass_initial_guess * 0.5,
                                 var_obj.mass_initial_guess * 2)          
        
            print(
                f"\n[it {it}] "
                f"stable={stable_loss.item():.6f} | touching={touching_loss.item():.6f} | "
                f"loss={loss.item():.6f}"
            )

            ## TODO: Implement an early stop at here
            if stable_loss.item() < 2e-3:
                print(f"\n [Markov chain]: Early stop at it={it} (stable_loss={stable_loss.item():.6f})")
                break
        
        with torch.no_grad():
            result_quat     = best["quaternion"].detach().cpu().numpy()
            result_pose     = best["pose"].detach().cpu().numpy()
            result_com      = best["com"].detach().cpu().numpy()
            result_mass     = float(best["mass"].detach().cpu().item())
            result_friction = float(best["friction"].detach().cpu().item())

            ic(f"[Markov chain]: Node {node_id} best loss {loss.item()} \n ")
            ic(f"[Markov chain]: Node {node_id} result_quat {result_quat}\n")
            ic(f"[Markov chain]: Node {node_id} result_pose {result_pose}\n")
            ic(f"[Markov chain]: Node {node_id} result_com {result_com}\n")
            ic(f"[Markov chain]: Node {node_id} result_mass {result_mass}\n")
            ic(f"[Markov chain]: Node {node_id} result_friction {result_friction}\n")

        with torch.no_grad():
            optim_quaternion /= (optim_quaternion.norm() + 1e-12)
        var_obj.update_all_result(quaternion = result_quat, 
                                  pose =  result_pose,
                                  com  =  result_com,
                                  mass = result_mass,
                                  friction = result_friction)

