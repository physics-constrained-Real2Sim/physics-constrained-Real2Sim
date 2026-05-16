from kaolin.metrics.pointcloud import chamfer_distance
from kaolin.metrics.pointcloud import f_score
import torch
import numpy as np
import trimesh

class GeometryEvaluator():

    def __init__(self, device='cuda', threshold=0.01):
        self.device = torch.device(device)
        self.threshold = threshold
    
    def _to_tensor(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device).float()
        return torch.from_numpy(data).to(device).float()

    def __call__(self, pred_point, gt_point):
        
        pred_p = self._to_tensor(pred_point, self.device)
        gt_p = self._to_tensor(gt_point, self.device)


        cd_dist, fscore, iou = self.evaluate(pred_p, gt_p)

        return cd_dist, fscore, iou
    
    def calculate_iou(self, pred_points, gt_points, voxel_size=0.05):
        """
        pred_points, gt_points: torch.Tensor (N, 3), already on device
        voxel_size: float, size of each voxel cube
        
        Returns:
            iou: float
        """
        # Shift both point clouds so the minimum is at the origin
        min_coord = torch.min(torch.cat([pred_points, gt_points], dim=0), dim=0)[0]

        pred_vox = torch.floor((pred_points - min_coord) / voxel_size).long()
        gt_vox   = torch.floor((gt_points - min_coord) / voxel_size).long()

        # Convert voxel coords to sets of tuples (hashable for set operations)
        pred_set = set(map(tuple, pred_vox.cpu().numpy()))
        gt_set   = set(map(tuple, gt_vox.cpu().numpy()))

        if len(pred_set) == 0 and len(gt_set) == 0:
            return 1.0   # both empty → perfect IoU

        intersection = len(pred_set & gt_set)
        union = len(pred_set | gt_set)

        if union == 0:
            return 0.0

        return intersection / union
    
    def evaluate(self, pred, gt):
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).to(self.device).float()
        else:
            pred = pred.to(self.device).float()
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).to(self.device).float()
        else:
            gt = gt.to(self.device).float()

        cd_dist = chamfer_distance(pred.unsqueeze(0), gt.unsqueeze(0))
        cd_dist = cd_dist.item()

        fscore = f_score(pred.unsqueeze(0), gt.unsqueeze(0))
        fscore = fscore.item()

        iou = self.calculate_iou(pred, gt)
        ## we also return the Intersection over Union (IoU) of two point clouds
        return cd_dist, fscore, iou
