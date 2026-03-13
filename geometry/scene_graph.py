import numpy as np
import trimesh
from icecream import ic
from .object_3D import object_3D

class scene_graph_3D:
    def __init__(self, objects_list):
        self.objects_list = objects_list  # list of object_3D instances
        assert all(isinstance(o, object_3D) for o in self.objects_list)
        self.parent_map = {}

    @staticmethod
    def _minmax_norm(vals, eps=1e-12):
        """
        min–max normalization on the values ​​within the same candidate set to [0,1].
        """
        v = np.asarray(vals, dtype=float)
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if vmax - vmin < eps:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)
    
    def support_gap(self, child: object_3D, parent: object_3D, quantile: float = 90.0) -> float:
        """
        Estimate the support surface height using the quantiles of z-axis of the parent object's top surface 
        (the triangle facing the top normal), 
        then calculate the gap from the bottom of the child object to that height 
        (negative values ​​are clamped with 0 to avoid "reward" interlacing)
        """
        if parent.V_faces_z.size == 0:
            parent_support_z = parent.bounds[1, 2]
        else:
            parent_support_z = float(np.percentile(parent.V_faces_z.reshape(-1), quantile))

        child_bottom_z = float(child.centroid[2] - 0.5 * child.bounding_box_extent[2])
        gap = child_bottom_z - parent_support_z
        return max(0.0, gap)  #

    def get_scene_graph(self):    
        for child in self.objects_list:
            if child.type == "object":
                parent = self.find_parent(child)
                parent_ID = parent.obj_ID if parent is not None else None
                self.parent_map[child.obj_ID] = parent_ID
                ic(f"Child object {child.obj_ID} parent object {parent_ID}")
        return self.parent_map
    

    def com_projection_excess(self, child: object_3D, parent: object_3D, pad: float = 0.0) -> float:
        """
        Project the child's COM and the parent's OBB together along gravity (z-axis) onto the world XY plane.

        Calculate the minimum distance from the child's COM to the projected polygon of the OBB; 
        return 0 if it is inside the polygon.

        pad: A "buffer" radius in absolute units such as meters; returns max(0, dist - pad).
        """
        def _convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
            """Andrew's monotone chai"""
            pts = np.asarray(points_xy, dtype=float)
            pts = np.unique(pts, axis=0)
            if len(pts) <= 2:
                return pts

            pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

            lower = []
            for p in pts:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(tuple(p))

            upper = []
            for p in reversed(pts):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(tuple(p))

            hull = np.array(lower[:-1] + upper[:-1], dtype=float)
            return hull

        def _point_in_convex_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
            """
            The test is performed on points within a convex polygon (including the boundary), 
            requiring the poly to be a CCW (Conformal Cross-Product). 
            The consistent cross-product notation method is used.
            """
            if len(poly) == 0:
                return False
            if len(poly) == 1:
                return np.allclose(p, poly[0])
            if len(poly) == 2:
                a, b = poly[0], poly[1]
                ap, ab = p - a, b - a
                t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
                t = max(0.0, min(1.0, t))
                proj = a + t * ab
                return np.linalg.norm(p - proj) <= 1e-12

            for i in range(len(poly)):
                a = poly[i]
                b = poly[(i + 1) % len(poly)]
                edge = b - a
                vp = p - a
                cross = edge[0] * vp[1] - edge[1] * vp[0]
                if cross < -1e-15:
                    return False
            return True

        def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            """Shortest distance from a point to a line segment"""
            ab = b - a
            denom = float(np.dot(ab, ab)) + 1e-15
            t = float(np.dot(p - a, ab)) / denom
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            return float(np.linalg.norm(p - proj))

        def _point_polygon_distance(p: np.ndarray, poly: np.ndarray) -> float:
            """Shortest distance from a point to a convex polygon (0 if inside)."""
            if _point_in_convex_polygon(p, poly):
                return 0.0
            dmin = np.inf
            for i in range(len(poly)):
                a = poly[i]
                b = poly[(i + 1) % len(poly)]
                d = _point_segment_distance(p, a, b)
                if d < dmin:
                    dmin = d
            return float(dmin)


        verts3d = np.asarray(parent.trimesh_obj.vertices, dtype=float)  # (N,3)
        verts_xy = verts3d[:, :2]  

        poly_xy = _convex_hull_2d(verts_xy)  # (M,2), CCW

        p_xy = np.asarray(child.center_obb[:2], dtype=float)

        dist = _point_polygon_distance(p_xy, poly_xy)

        return float(max(0.0, dist - float(pad)))

    
    
    def aabb_xy_overlap(self, a: object_3D, b: object_3D, pad: float = 0.0) -> bool:
        a_min = a.bounds[0]; a_max = a.bounds[1]
        b_min = b.bounds[0]; b_max = b.bounds[1]
        # XY
        return not (a_max[0] + pad < b_min[0] or b_max[0] + pad < a_min[0] or
                    a_max[1] + pad < b_min[1] or b_max[1] + pad < a_min[1])
    
    @staticmethod
    def obb_xy_corners(obj) -> np.ndarray:
        """
        Returns the OBB of obj projected onto the 2D vertex (N, 2) in the XY plane in world coordinates.
        """
        obb = obj.bounding_box_oriented              # trimesh OBB
        T = obb.primitive.transform                  # 4x4 
        extents = obb.primitive.extents / 2.0        # (ex, ey, ez)

        ex, ey, ez = extents
        corners_local = np.array([
            [ ex,  ey,  ez, 1],
            [ ex,  ey, -ez, 1],
            [ ex, -ey,  ez, 1],
            [ ex, -ey, -ez, 1],
            [-ex,  ey,  ez, 1],
            [-ex,  ey, -ez, 1],
            [-ex, -ey,  ez, 1],
            [-ex, -ey, -ez, 1],
        ])

        corners_world = (T @ corners_local.T).T      # (8, 4)
        return corners_world[:, :2]                  # (x, y)

    @staticmethod
    def poly_overlap_xy(p: np.ndarray, q: np.ndarray, pad: float = 0.0) -> bool:
        """
        Use the 2D separating axis theorem to determine if two convex polygons p and q intersect.

        p, q: Vertices (N, 2)

        pad: The "expansion" quantity, the same as your original logic.
        """
        def projections(vertices, axis):
            proj = vertices @ axis
            return proj.min(), proj.max()

        # Generate a normal line for each of the two sides as the separation axis.
        for poly in (p, q):
            n = len(poly)
            for i in range(n):
                edge = poly[(i + 1) % n] - poly[i]
                axis = np.array([-edge[1], edge[0]])    # perpendicular to the edge
                norm = np.linalg.norm(axis)
                if norm == 0:
                    continue
                axis /= norm

                min_p, max_p = projections(p, axis)
                min_q, max_q = projections(q, axis)

                if max_p + pad < min_q or max_q + pad < min_p:
                    return False   # Find the separating axis -> non-intersecting

        return True  


    def obb_xy_overlap(self, a: object_3D, b: object_3D, pad: float = 0.0) -> bool:
        """
        Change original aabb_xy_overlap to the OBB version.
        """
        a_xy = self.obb_xy_corners(a)
        b_xy = self.obb_xy_corners(b)
        return self.poly_overlap_xy(a_xy, b_xy, pad=pad)
        
    def broad_phase(self, child: object_3D, parent: object_3D,
                    z_offset_thresh: float = 0.0,
                    max_gap: float = np.inf) -> bool:
        if child is parent:
            return False
        if parent.type not in ("object", "ground"):
            return False

        cz = float(child.centroid[2])
        pz = float(parent.centroid[2])
        if not (pz + z_offset_thresh < cz):
            return False

        if not self.obb_xy_overlap(a = child, b = parent, pad= 0.0):
            return False
        return True
    
    def find_parent(self, child: object_3D,                     
                    w_dist: float = 1.0,
                    w_com: float = 1.0,
                    tau: float = 5.0,):
        """
        Narrow phase:
        - Filter by broad_phase
        - Score by distance (smaller is better) + COM projection containment (bonus)
        - Return the best parent (or None)
        """
        candidates = []
        for cand in self.objects_list:
            if cand is child:
                continue
            if not self.broad_phase(child, cand):
                continue
            ## TODO： use COM projection score instead of hard constraint
            com_excess = self.com_projection_excess(child, cand, pad=0.0)

            ## TODO: compute distance score and COM projection score
            ## TODO: use oriented bounding box instead of axis-aligned bounding box
            d = self.distance(child, cand)
            # com_score = self.COM_projection_score(child, cand)
            # total = self.score(distance=d, com_score=com_score)

            candidates.append({"dist" :d, "com_excess": com_excess, "cand": cand})

        if not candidates:
            return None
        
        ## get the most likely hood parent object
        ## Do normalization to distance and com_excess
        ## select the best score parent object
        ## TODO: weight for distance and com_excess

        d_list = [c["dist"] for c in candidates]
        c_list = [c["com_excess"] for c in candidates]

        d_norm = self._minmax_norm(d_list)          # [0,1]
        c_norm = self._minmax_norm(c_list)          # [0,1]

        costs = w_dist * d_norm + w_com * c_norm
        # softmax(-tau * cost)
        logits = -tau * costs
        logits -= np.max(logits)
        expv = np.exp(logits)
        probs = expv / np.sum(expv)

        best_idx = int(np.argmax(probs))
        best = candidates[best_idx]["cand"]

        # ic()
        debug_rows = []
        for i, c in enumerate(candidates):
            debug_rows.append({
                "child_id": child.obj_ID,
                "parent_id": c["cand"].obj_ID,
                "dist": c["dist"],
                "dist_norm": float(d_norm[i]),
                "com_excess": c["com_excess"],
                "com_norm": float(c_norm[i]),
                "cost": float(costs[i]),
                "prob": float(probs[i]),
            })

        # ic(debug_rows)
        return best  


    def distance(self, obj1: object_3D, obj2: object_3D):
        
        ## distance between the childrent object obj1 and potential parent object obj2
        ## distance from child bottom bounding box to the faces_opposite_to_gravity of parent object

        ## V is the upper surface of potential parent object
        ## how to get the vertices of V?  

        ## take maximum z of each faces
        # ic(obj2.obj_ID)
        # ic(obj2.type)
        # ic(obj2.V_faces_z.shape)
        dist =  obj1.centroid[2] - (obj2.V_faces_z + obj1.bounding_box_extent[2]/2)
        # ic(dist.shape)
        # ic(dist)

        # take the minimum distance 
        dist = np.min(dist)
        # ic(dist)

        return dist
    def visualize_scene_graph(self,
                            show_meshes: bool = False,
                            node_radius: float = 0.02,
                            draw_ground: bool = True,
                            edge_radius: float = 0.005):
        """
        Visualize the scene graph in 3D using Open3D with thicker, colorful edges.

        - Nodes: small spheres at each object's OBB center
        * level 0 (ground): one color
        * level 1: one color
        * level 2: one color
        * ...
        - Edges: arrows from parent -> child  (用你写的箭头)
        - Optionally overlay triangle meshes (slower)
        """

        import open3d as o3d
        import numpy as np
        from collections import deque, defaultdict

        # --- Gather nodes (centers) ---
        objs = self.objects_list if draw_ground else [o for o in self.objects_list if o.type != "ground"]
        if len(objs) == 0:
            print("[visualize_scene_graph] Nothing to draw.")
            return

        id_to_idx = {}
        points = []
        for i, o in enumerate(objs):
            id_to_idx[o.obj_ID] = i
            points.append(np.asarray(o.center_obb, dtype=float))
        points = np.array(points)

        geoms = []

        # =====================================================
        # 1) Each node's level（ground = 0, child = parent+1）
        # =====================================================
        # parent_map: child -> parent
        children_map = defaultdict(list)
        for child_id, parent_id in self.parent_map.items():
            if parent_id is not None:
                children_map[parent_id].append(child_id)

        levels = {}
        q = deque()

        # ground is level 0 
        for o in objs:
            if o.type == "ground":
                levels[o.obj_ID] = 0
                q.append(o.obj_ID)

        # BFS level
        while q:
            pid = q.popleft()
            parent_level = levels[pid]
            for cid in children_map.get(pid, []):
                if cid not in levels:
                    levels[cid] = parent_level + 1
                    q.append(cid)

        # Any un-connected ground nodes, give a default level 1
        for o in objs:
            if o.obj_ID not in levels:
                if o.type == "ground":
                    levels[o.obj_ID] = 0
                else:
                    levels[o.obj_ID] = 1

        # visualize level color
        level_palette = [
            np.array([0.2, 0.8, 0.2]),  # level 0: ground green
            np.array([0.2, 0.4, 0.9]),  # level 1: blue
            np.array([0.9, 0.6, 0.2]),  # level 2: orange
            np.array([0.9, 0.4, 0.4]),  # level 3: red
            np.array([0.6, 0.3, 0.8]),  # level 4: prurple
        ]

        def get_level_color(level: int):
            if level < len(level_palette):
                return level_palette[level]
            return level_palette[-1]

        for o in objs:
            c = np.asarray(o.center_obb, dtype=float)
            level = levels.get(o.obj_ID, 0 if o.type == "ground" else 1)
            color = get_level_color(level)

            sph = o3d.geometry.TriangleMesh.create_sphere(radius=node_radius)
            sph.compute_vertex_normals()
            sph.translate(c)
            sph.paint_uniform_color(color)
            geoms.append(sph)

        for child_id, parent_id in self.parent_map.items():
            if parent_id is None:
                continue
            if not draw_ground:
                child_o = next((x for x in objs if x.obj_ID == child_id), None)
                parent_o = next((x for x in objs if x.obj_ID == parent_id), None)
                if (child_o is None) or (parent_o is None):
                    continue
            if child_id not in id_to_idx or parent_id not in id_to_idx:
                continue

            p1 = points[id_to_idx[child_id]]   # child
            p2 = points[id_to_idx[parent_id]]  # parent

            # ---- Create Arrow from parent -> child ----
            arrow_color = [1.0, 0.2, 0.2]

            vec = p1 - p2                   # parent -> child
            length = np.linalg.norm(vec)
            if length < 1e-6:
                continue
            direction = vec / length

            base_cone_len = edge_radius * 8.0   

            if length > base_cone_len:
                cone_len = base_cone_len
                cyl_len = length - cone_len - 0.5 * base_cone_len
            else:
                cone_len = length * 0.6
                cyl_len = length - cone_len

            cyl_len = max(cyl_len, 1e-3)
            cone_len = max(cone_len, 1e-3)

            cyl_rad = edge_radius
            cone_rad = edge_radius * 2.5

            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cyl_rad,
                cone_radius=cone_rad,
                cylinder_height=cyl_len,
                cone_height=cone_len
            )
            arrow.compute_vertex_normals()
            arrow.paint_uniform_color(arrow_color)
            z_axis = np.array([0, 0, 1], dtype=float)

            v = np.cross(z_axis, direction)
            c = np.dot(z_axis, direction)
            if np.linalg.norm(v) < 1e-6:
                R = np.eye(3)
            else:
                vx = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

            arrow.rotate(R, center=np.zeros(3))
            arrow.translate(p2)

            geoms.append(arrow)

        # --- Optionally overlay the actual meshes ---
        if show_meshes:
            for o in objs:
                m = o3d.geometry.TriangleMesh()
                m.vertices = o3d.utility.Vector3dVector(np.asarray(o.vertices, dtype=float))
                m.triangles = o3d.utility.Vector3iVector(np.asarray(o.faces, dtype=np.int32))
                m.compute_vertex_normals()
                m.paint_uniform_color([0.8, 0.8, 0.8])
                geoms.append(m)

        # --- Print adjacency info ---
        print("\n[Scene Graph]")
        print("  Node color: level 0=ground, level 1/2/... have different colors")
        print("  Parent map (child -> parent, level in brackets):")
        for child_id, parent_id in self.parent_map.items():
            lvl = levels.get(child_id, "?")
            print(f"    {child_id} (L{lvl}) -> {parent_id}")

        # --- Display ---
        o3d.visualization.draw_geometries(geoms)

    def visualize(self):
        """
        Visualize all 3D objects in the scene using trimesh's built-in viewer.
        """
        # Collect all trimesh geometries from the object list
        geometries = []
        for obj in self.objects_list:
            if hasattr(obj, 'trimesh_obj') and obj.trimesh_obj is not None:
                geometries.append(obj.trimesh_obj)
            else:
                print(f"Warning: Object {obj.obj_ID} has no valid mesh.")
        
        # Create a single scene from all meshes
        scene = trimesh.Scene(geometries)

        # Show the scene in an interactive window
        scene.show()