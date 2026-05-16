import numpy as np
from icecream import ic
def evaluate_stable(pose, quaternion,
                    pose_init, quaternion_init,
                    pos_thresh=0.01,
                    ang_thresh_deg=5.0):
    pose = np.asarray(pose, dtype=np.float32).reshape(3,)
    pose_init = np.asarray(pose_init, dtype=np.float32).reshape(3,)
    q = np.asarray(quaternion, dtype=np.float32).reshape(4,)
    q_init = np.asarray(quaternion_init, dtype=np.float32).reshape(4,)

    pos_diff = np.linalg.norm(pose - pose_init)

    def normalize_quat(q):
        n = np.linalg.norm(q)
        if n < 1e-8:
            return q
        return q / n

    q = normalize_quat(q)
    q_init = normalize_quat(q_init)

    dot = float(np.clip(np.abs(np.dot(q, q_init)), -1.0, 1.0))
    ang_diff_rad = 2.0 * np.arccos(dot)
    ang_diff_deg = np.degrees(ang_diff_rad)

    is_stable = (pos_diff <= pos_thresh) and (ang_diff_deg <= ang_thresh_deg)

    return is_stable, pos_diff, ang_diff_deg
