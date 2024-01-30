from utils import smpl3d as datasets
from model.PhysMoP import PhysMoP
from model.HumanModel import SMPL
from utils.opt import Options
from utils import util
from utils import log
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
import os
from utils.util import forward_kinematics, remove_singlular_batch
from main_h36m_3d import compute_errors

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(opt):
    test_dataset = datasets.Datasets(opt, split=2)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    net_pred = PhysMoP(hist_length=10,
                       physics=False,
                       data=True,
                       fusion=False
                       ).cuda()

    if opt.checkpoint_path is not None:
        print('>>> loading pretrained model')
        checkpoint = torch.load(opt.checkpoint_path)
        net_pred.load_state_dict(checkpoint['state_dict'])

    smplModel = SMPL(device='cuda')

    n = 0
    m_p3d_h36 = np.zeros([opt.output_n])
    titles = np.array(range(opt.output_n)) + 1

    for i, (pose, shape, trans) in enumerate(test_loader):
        batch_size, seq_n, D = pose.shape
        process_size = batch_size * seq_n

        n += batch_size
        bt = time.time()

        gt_pose = pose.view(process_size, 72).float().cuda()
        gt_shape = shape.view(process_size, 10).float().cuda()

        gt_joints, gt_joints_smpl = forward_kinematics(smplModel, gt_pose, gt_shape,
                                                       joints_smpl=True)
        motion_pred_data, motion_pred_physics_gt, motion_pred_physics_pred, motion_pred_fusion, weight_t = net_pred(
            gt_pose)

        pred_pose_data = motion_pred_data.view(process_size, 72)
        pred_joints_data, pred_joints_smpl_data = forward_kinematics(smplModel, pred_pose_data, gt_shape,
                                                                     joints_smpl=True)

        gt_J = gt_joints_smpl
        pred_J_data = pred_joints_smpl_data

        gt_J = gt_joints_smpl.detach().cpu().numpy()
        pred_J_data = pred_joints_smpl_data.detach().cpu().numpy()
        _, mpjpe_p3d_h36 = compute_errors(gt_J, pred_J_data, 0)
        error_test_data = np.array(mpjpe_p3d_h36).reshape([-1, seq_n])
        error_test_data = error_test_data[:, 10:]
        m_p3d_h36 += error_test_data.sum(axis=0)

    ret = {}
    m_p3d_h36 = m_p3d_h36 / n
    for j in range(opt.output_n):
        ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    ret_log = np.array([])
    head = np.array([])
    for k in ret.keys():
        ret_log = np.append(ret_log, [ret[k]])
        head = np.append(head, ['test_' + k])
    log.save_csv_log(opt, head, ret_log, is_create=True)

if __name__ == '__main__':
    opt = Options().parse()
    main(opt)
