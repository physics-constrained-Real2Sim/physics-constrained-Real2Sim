import torch
import numpy as np

def get_tensor(x, base_tensor=None, **kwargs):
    """Wrap array or scalar in torch Tensor, if not already.
    """
    if isinstance(x, torch.Tensor):
        return x
    elif base_tensor is not None:
        return base_tensor.new_tensor(x, **kwargs)
    else:
        return torch.tensor(x, device= "cuda")
    

def easy_quaternion_to_matrix(q):
    """
    Convert a quaternion ( w, x, y, z,) into a 3x3 rotation matrix.

    Parameters:
    q : array-like of shape (4,)
        Quaternion in the form [ w ,x, y, z,]

    Returns:
    R : ndarray of shape (3, 3)
        Rotation matrix
    """
    w, x, y, z,  = q

    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    
    return R