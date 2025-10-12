from utils.motion_process import recover_from_ric
import torch

def get_head_traj(data):
    joint_data = data
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
    joint = joint.reshape(-1, 22, 3)
    MINS = joint.min(axis=0).min(axis=0)
    
    height_offset = MINS[1]
    joint[:, :, 1] -= height_offset
    
    head_traj = joint[:, 15]
    
    return head_traj