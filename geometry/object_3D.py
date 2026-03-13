import os
import kaolin
import torch
import coacd
import trimesh
import numpy as np

from icecream import ic
from pytorch3d.transforms import quaternion_to_matrix
from trimesh.remesh import subdivide_to_size
from .ultility import get_tensor, easy_quaternion_to_matrix

class object_3D:
    def __init__(self, mesh_path, type = "object", obj_ID = -1, 
                 custom_trimesh = False, trimesh_obj = None, 
                 density_real = 2000.0, compute_convex_decomposition = False,
                 ):

        if custom_trimesh is False:
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Input file '{mesh_path}' does not exist.")
            self.mesh_path = mesh_path
            self.trimesh_obj = trimesh.load(mesh_path)
        else:
            self.trimesh_obj = trimesh_obj
            trimesh_obj.export('ground.obj')
            self.mesh_path = 'ground.obj'
            print(self.mesh_path)

        self.type = type  # objects / ground

        self.obj_ID = obj_ID  # unique ID for each object

        self.kaolin_mesh = None

        ## physical properties remain after geometry changed
        self.mass = self.trimesh_obj.volume * density_real
        self.friction = 0.5

        ## store the optimized parameters 
        self.transformation_accumulation = np.eye(4)
        self.mass_result        = None
        self.friction_result    = None
        ## The result com from diffsdfsim is relative
        ## to object mesh.bounding_box.centroid
        self.COM_result = None 

        self.compute_geometry_properties()
        self.compute_upward_facing_faces()

        ## If compute convex decomposition
        if compute_convex_decomposition:
            self.convex_mesh_path = self.coacd_convex_decomposition()


    def update_all_result(self, quaternion, pose, com, mass, friction):

        self.trimesh_apply_transform(quaternion= quaternion, 
                                     pose= pose)
        self.COM_result  = com
        self.mass_result = mass
        self.friction_result    = friction

    def trimesh_apply_transform(self, quaternion = [1, 0, 0, 0], pose = [0, 0, 0]):
        """
        Apply a 4x4 transformation matrix to the trimesh object.
        """
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] =  easy_quaternion_to_matrix(quaternion)
        transform_matrix[:3, 3] = pose

        print("trimesh apply transform:\n", transform_matrix)

        self.transformation_accumulation  = transform_matrix @ self.transformation_accumulation 
        self.trimesh_obj.apply_transform(transform_matrix)
        # After transformation, recompute geometry properties
        self.compute_geometry_properties()
        self.compute_upward_facing_faces()

    def subdivide_object(self):
        # subdivide_to_size
        v_new, f_new = subdivide_to_size(
            vertices=self.trimesh_obj.vertices,
            faces=self.trimesh_obj.faces,
            max_edge=0.05,  
            max_iter= 200
        )

        self.trimesh_obj = trimesh.Trimesh(vertices=v_new, faces=f_new, process=True)

        print("\nAfter subdivision:")
        print("Vertices:", len(self.trimesh_obj.vertices))
        print("Faces:", len(self.trimesh_obj.faces))
        self.compute_geometry_properties()
        self.compute_upward_facing_faces()


    def coacd_convex_decomposition(self, threshold = 0.03, 
                                   preprocess_resolution = 100):
        """
        perform convex decomposition using coacd.
        If the output file already exists, it will be skipped.
        Returns the path to the output .obj file.
        """
        output_file = os.path.splitext(self.mesh_path)[0] + "_coacd.obj"

        if os.path.exists(output_file):
            return output_file
        
        convex_decomposed_mesh = coacd.Mesh(self.vertices, self.faces)
        #result = coacd.run_coacd(mesh, threshold= 0.02, mcts_max_depth= 5, mcts_nodes= 30, preprocess_mode= "False") # a list of convex hulls.
        result = coacd.run_coacd(convex_decomposed_mesh, threshold = threshold, preprocess_resolution =preprocess_resolution) # a list of convex hulls.

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


    def compute_geometry_properties(self):
        """
        compute geometry properties of the object
        """
        self.vertices = np.array(self.trimesh_obj.vertices)
        self.faces = np.array(self.trimesh_obj.faces)

        ## bounds
        self.bounds = self.trimesh_obj.bounds
        # ic(self.bounds) # 2 x 3

        ## take center of oriented bounding box as guess of center of scene graph
        self.centroid = self.trimesh_obj.centroid

        self.center_obb = self.trimesh_obj.bounding_box_oriented.primitive.transform[:3,3]
    
        ## oriented bounding box
        self.bounding_box_oriented = self.trimesh_obj.bounding_box_oriented
        self.bounding_box_oriented_extent = self.trimesh_obj.bounding_box_oriented.extents
        self.bounding_box_oriented_transform = self.trimesh_obj.bounding_box_oriented.primitive.transform
        # ic(self.bounding_box_oriented_extent)
        # ic(self.bounding_box_oriented_transform)

        ##  axis-aligned bounding box
        self.bounding_box_extent = self.trimesh_obj.bounding_box.extents
        self.bounding_box_vertices = self.trimesh_obj.bounding_box.vertices
        # ic(self.bounding_box_extent)

        minv, maxv = self.trimesh_obj.bounds
        extents_world = maxv - minv
        body_half_extents = 0.5 * extents_world
        
        self.body_half_extents = torch.tensor(body_half_extents,
                                              device= 'cuda:0',
                                              requires_grad= False)

        self.mass_initial_guess = torch.tensor(self.mass,
                                               device= 'cuda:0',
                                               requires_grad= False)
        
    
    def compute_upward_facing_faces(self):
        """
        compute upward facing faces opposite to gravity direction
        """
        self.faces_normals = self.trimesh_obj.face_normals
        self.gravity = np.array([0, 0, -1])
        # Facet-level filtering
        self.mask, cosines = self.filter_faces_opposite_to_gravity(
            threshold=0.95,
        )
        # ic(self.mask)
        # ic(self.mask.shape)
        self.V = self.faces[self.mask]
        # ic(self.V.shape)

        self.up_tri_vertex_indices = self.V.reshape(-1)               # (K*3,)
        self.up_vertices = self.vertices[self.up_tri_vertex_indices, :]    # (K*3, 3)
        self.V_faces_z = np.mean(self.vertices[self.V, 2], axis=1).reshape(-1, 1)

    def create_kal_mesh(self, quaternion = [1, 0, 0, 0], pose = [0, 0, 0]):
        """
        Kaolin mesh attributes
        """
        vertices = torch.tensor(self.vertices, dtype=torch.float32, device="cuda")
        faces = torch.tensor(self.faces).long()
        quaternion = get_tensor(quaternion)
        pose = get_tensor(pose)
        
        matrix = quaternion_to_matrix(quaternion) ## w x y z
        
        matrix_T = matrix.T.to(vertices.dtype)
        pose_transformed = pose.to(vertices.dtype)

        vertices = vertices @ matrix_T + pose_transformed

        self.kaolin_mesh = kaolin.rep.SurfaceMesh(vertices.float(), faces.long())
        ## move to GPU
        self.kaolin_mesh = self.kaolin_mesh.cuda(attributes=["vertices"])
        self.kaolin_mesh = self.kaolin_mesh.cuda(attributes=["faces"])
        self.kaolin_mesh.allow_auto_compute=True

        return self.kaolin_mesh

    def sample_surface_points(self, num_samples=1000):
        # Sample points on the surface of the mesh
        pointclouds, _ = kaolin.ops.mesh.sample_points(vertices=self.kaolin_mesh.vertices[None], 
                                                       faces=self.kaolin_mesh.faces, 
                                                       num_samples=num_samples)
        return pointclouds
    
    def compute_sdf_from_points(self, points):
        # Compute the signed distance from points to the mesh surface
        index_vertices_by_faces = kaolin.ops.mesh.index_vertices_by_faces(vertices_features=self.kaolin_mesh.vertices[None],
                                                                          faces=self.kaolin_mesh.faces)
        squared_distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(points, index_vertices_by_faces)
        distance = torch.sqrt(squared_distance)

        sign = kaolin.ops.mesh.check_sign(verts=self.kaolin_mesh.vertices[None], faces=self.kaolin_mesh.faces, points=points)
        sign_num = torch.where(sign, torch.tensor(-1.0).to("cuda"), torch.tensor(1.0).to("cuda"))

        sdf = distance * sign_num
        return sdf
    
    def get_kaolin_mesh_AABB(self):
        """
        Compute and cache the axis-aligned bounding box [min, max] from the current
        kaolin mesh vertices. This does not require gradients and should be called
        whenever the object moves / is reposed.

        Returns:
            (aabb_min, aabb_max): tuple of 1D tensors with shape [3], same device/dtype
                                as the mesh vertices.
        """
        if not hasattr(self, "kaolin_mesh") or self.kaolin_mesh is None:
            raise RuntimeError("kaolin_mesh not initialized. Call create_kal_mesh(...) first.")

        with torch.no_grad():
            verts = self.kaolin_mesh.vertices  # [V,3] or [1,V,3]
            # If a batch dimension sneaks in, squeeze it out.
            if verts.dim() == 3 and verts.size(0) == 1:
                verts = verts[0]

            if verts.numel() == 0:
                raise RuntimeError("Mesh has no vertices; cannot compute AABB.")

            aabb_min = verts.min(dim=0).values
            aabb_max = verts.max(dim=0).values

            # Cache for broad-phase collision queries
            self.bounds = (aabb_min, aabb_max)

        return self.bounds

    def visualize_faces_opposite_to_gravity(self):
        m = self.trimesh_obj.copy()
        m.visual = trimesh.visual.ColorVisuals(mesh=m)

        m.visual.face_colors[:] = [200, 200, 200, 255]
        m.visual.face_colors[self.mask] = [255, 80, 80, 255]
        m.show()

    def filter_faces_opposite_to_gravity(
        self,
        threshold: float = 0.9,
    ):
        """
        Select faces (or facets) whose normal is opposite to gravity.

        We keep items where dot(n̂, ĝ) <= -threshold  (e.g., <= -0.9).
        With g = [0,0,-1], this keeps upward-pointing surfaces (n ≈ +Z).

        Args:
            gravity: 3-vector for gravity direction (any magnitude).
            threshold: cosine threshold in [0, 1]. Larger = stricter (closer to exactly opposite).
            return_indices: if True, also return indices of matching faces / facets.

        Returns:
            mask:      Boolean mask over faces (or facets).
            cosines:   dot(n̂, ĝ) for each face (or facet).
            indices:   (optional) np.ndarray of selected indices (faces or facets).
            faces_in_selected_facets: (optional) if use_facets=True, a 1D array of all face indices belonging to the kept facets.
        """
        # Normalize gravity
        g_hat = np.asarray(self.gravity, dtype=float)

        n_hat = np.asarray(self.faces_normals)  # shape (faces, 3)
        # Cosine with gravity

        # ic(n_hat.shape)
        # ic(g_hat.shape)
        cosines = n_hat @ g_hat  # shape (N,)

        # Opposite to gravity means cos <= -threshold
        thr = float(np.clip(threshold, 0.0, 1.0))
        mask = cosines <= -thr
        return mask, cosines


    def visualize_with_normals(self, normal_length=0.05):
        mesh = self.trimesh_obj.copy()
        start_points = mesh.vertices
        end_points = mesh.vertices + mesh.vertex_normals * normal_length

        normal_lines = trimesh.load_path(np.hstack((start_points, end_points)).reshape(-1, 2, 3))

        scene = trimesh.Scene([mesh, normal_lines])
        scene.show()

    def to_export_dict(self):
        def to_abs(p):
            return os.path.abspath(p) if isinstance(p, str) else None

        mesh_path = getattr(self, "mesh_path", None)
        convex_mesh_path = getattr(self, "convex_mesh_path", None)
        T = getattr(self, "transformation_accumulation", None)
        mass_result = getattr(self, "mass_result", None)
        friction_result = getattr(self, "friction_result", None)
        com_result = getattr(self, "COM_result", None)
        obj_id = getattr(self, "obj_ID", None)
        type_ = getattr(self, "type", None)

        def to_list(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x.tolist()
            try:
                return np.asarray(x).tolist()
            except Exception:
                return x

        export = {
            "obj_ID": obj_id,
            "type": type_,
            "mesh_path": to_abs(mesh_path),
            "convex_mesh_path": to_abs(convex_mesh_path),
            "transformation_accumulation": to_list(T),
            "mass_result": mass_result,          # scalar
            "friction_result": friction_result,  # scalar
            "com_result": to_list(com_result),   # vector
        }
        return export