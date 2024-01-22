import torch
import numpy as np


def lr_decay_mine(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def orth_project(cam, pts):
    """

    :param cam: b*[s,tx,ty]
    :param pts: b*k*3
    :return:
    """
    s = cam[:, 0:1].unsqueeze(1).repeat(1, pts.shape[1], 2)
    T = cam[:, 1:].unsqueeze(1).repeat(1, pts.shape[1], 1)

    return torch.mul(s, pts[:, :, :2] + T)


def opt_cam(x, x_target):
    """
    :param x: N K 3 or  N K 2
    :param x_target: N K 3 or  N K 2
    :return:
    """
    if x_target.shape[2] == 2:
        vis = torch.ones_like(x_target[:, :, :1])
    else:
        vis = (x_target[:, :, :1] > 0).float()
    vis[:, :2] = 0
    xxt = x_target[:, :, :2]
    xx = x[:, :, :2]
    x_vis = vis * xx
    xt_vis = vis * xxt
    num_vis = torch.sum(vis, dim=1, keepdim=True)
    mu1 = torch.sum(x_vis, dim=1, keepdim=True) / num_vis
    mu2 = torch.sum(xt_vis, dim=1, keepdim=True) / num_vis
    xmu = vis * (xx - mu1)
    xtmu = vis * (xxt - mu2)

    eps = 1e-6 * torch.eye(2).float().cuda()
    Ainv = torch.inverse(torch.matmul(xmu.transpose(1, 2), xmu) + eps.unsqueeze(0))
    B = torch.matmul(xmu.transpose(1, 2), xtmu)
    tmp_s = torch.matmul(Ainv, B)
    scale = ((tmp_s[:, 0, 0] + tmp_s[:, 1, 1]) / 2.0).unsqueeze(1)

    scale = torch.clamp(scale, 0.7, 10)
    trans = mu2.squeeze(1) / scale - mu1.squeeze(1)
    opt_cam = torch.cat([scale, trans], dim=1)
    return opt_cam


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def chunkponosig(x):
    a, b = torch.chunk(x, 2, dim=1)
    a, _, __ = pono(a)
    y = a * torch.sigmoid(b)
    return y


def pono(x, epsilon=1e-5):
    """Positional normalization"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output, mean, std


def batch_euler2matyzx(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ zmat @ ymat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual


def batch_euler2matzyx(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual


def batch_euler2matzxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = ymat @ xmat @ zmat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual


def batch_roteulerSMPL(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 72], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)

    rotMat_root, rotMat_root_individual = batch_euler2matyzx(angle[:, :3].reshape(-1, 3))
    rotMat_s, rotMat_s_individual = batch_euler2matzyx(angle[:, 3:48].reshape(-1, 3))
    rotMat_shoulder, rotMat_shoulder_individual = batch_euler2matzxy(angle[:, 48:54].reshape(-1, 3))
    rotMat_elbow, rotMat_elbow_individual = batch_euler2matyzx(angle[:, 54:60].reshape(-1, 3))
    rotMat_e, rotMat_e_individual = batch_euler2matzyx(angle[:, 60:].reshape(-1, 3))

    rotMat_root = rotMat_root.reshape(B, 1, 3, 3)
    rotMat_s = rotMat_s.reshape(B, 15, 3, 3)
    rotMat_shoulder = rotMat_shoulder.reshape(B, 2, 3, 3)
    rotMat_elbow = rotMat_elbow.reshape(B, 2, 3, 3)
    rotMat_e = rotMat_e.reshape(B, 4, 3, 3)
    rotMat = torch.cat((rotMat_root, rotMat_s, rotMat_shoulder, rotMat_elbow, rotMat_e), dim=1)

    rotMat_root_individual = rotMat_root_individual.reshape(B, 1, 3, 3, 3)
    rotMat_s_individual = rotMat_s_individual.reshape(B, 15, 3, 3, 3)
    rotMat_shoulder_individual = rotMat_shoulder_individual.reshape(B, 2, 3, 3, 3)
    rotMat_elbow_individual = rotMat_elbow_individual.reshape(B, 2, 3, 3, 3)
    rotMat_e_individual = rotMat_e_individual.reshape(B, 4, 3, 3, 3)
    rotMat_individual = torch.cat((rotMat_root_individual,
                                   rotMat_s_individual,
                                   rotMat_shoulder_individual,
                                   rotMat_elbow_individual,
                                   rotMat_e_individual),
                                  dim=1).reshape(B, -1, 3, 3)
    return rotMat, rotMat_individual


def forward_kinematics(smplModel, pose, shape, process_size,vertices=False, joints_smpl=False):
    rotmat, rotMat_individual = batch_roteulerSMPL(pose)

    output_smpl = smplModel.forward(betas=shape, rotmat=rotmat)

    if vertices:
        vertices = output_smpl.vertices.float().cuda()
    else:
        vertices = None

    joints = output_smpl.joints[:, :17].float().cuda()

    if joints_smpl == True:
        joints_smpl = output_smpl.joints_smpl[:, :24].float().cuda()
    else:
        joints_smpl = None


    return vertices, joints, joints_smpl
