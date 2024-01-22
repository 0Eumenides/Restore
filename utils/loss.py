import torch
import numpy as np
#  NOTE: all the loss function are implemented in pytorch

# mse损失
# outputs: (32, 20, 22, 3)
# targets: (32, 20, 22, 3)
def mse_error_3d(outputs, targets):
    return torch.mean(torch.norm(outputs - targets, dim=3))

# 速度损失
# outputs: (32, 20, 22, 3)
# targets: (32, 20, 22, 3)
def vel_error_3d(outputs, targets):
    return torch.mean(torch.norm(outputs[:, 1:] - outputs[:, :-1] -
                                 (targets[:, 1:] - targets[:, :-1]), dim=3))

# 骨长损失
# outputs: (32, 20, 22, 3)
# targets: (32, 20, 22, 3)
def bone_len_error_3d(outputs, targets):
    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1

    outputs_p3d = outputs.clone()  # 32,20,32,3

    targ_3d = targets.clone()  # 32,20,32,3
    # Calculate bone length loss
    pred_bone_lengths = torch.norm(outputs_p3d[:, :, I, :] - outputs_p3d[:, :, J, :], p=2, dim=-1)  # 32,20,16
    targ_bone_lengths = torch.norm(targ_3d[:, :, I, :] - targ_3d[:, :, J, :], p=2, dim=-1)  # 32,20,16

    loss = torch.mean(torch.abs(pred_bone_lengths - targ_bone_lengths))

    return loss