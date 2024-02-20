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
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def align_by_pelvis(joints, root):
    """
    Root alignments, i.e. subtracts the root.
    Args:
        joints: is N x 3
        roots: index of root joints
    """
    hip_id = 0
    pelvis = joints[hip_id, :]

    return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds, root):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 17 common joints.
    Inputs:
      - gt3ds: N x J x 3
      - preds: N x J x 3
      - root: root index for alignment
    """
    perjerrors, errors, perjerrors_pa, errors_pa = [], [], [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d, root)
        pred3d = align_by_pelvis(pred, root)

        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        errors.append(np.mean(joint_error) * 1000)
        perjerrors.append(joint_error * 1000)

    return perjerrors, errors


def visual(joints, ID):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    joints = joints.cpu().numpy()
    # 提取 x, y, z 坐标
    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2]

    # 绘制点
    ax.scatter(x, y, z)
    # for i in range(len(joints)):
    #     ax.text(x[i], y[i], z[i], f'{i}', color='blue', fontsize=10)

    # connections = [
    #     (0, 1), (1, 4), (4, 7), (7, 10),  # 右侧腿部
    #     (0, 2), (2, 5), (5, 8), (8, 11),  # 左侧腿部
    #     (0, 3), (3, 6), (6, 9),  # 脊柱
    #     (9, 12), (12, 15),  # 颈部和头部
    #     (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),  # 右侧手臂
    #     (9, 14), (14, 17), (17, 19), (19, 21), (21, 23)  # 左侧手臂
    # ]
    #
    # # 绘制连接线
    # for (i, j) in connections:
    #     ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'blue')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()
    fig.savefig('joints_3d_plot' + str(ID) + '.png')


def posePreprocess(pose, trans):
    # 检查pose的形状
    if pose.shape[2] != 72:
        raise ValueError("Each pose array must have 72 elements.")

    # 检查trans的形状
    if trans.shape[2] != 3:
        raise ValueError("Each trans array must have 3 elements.")

    # 定义要删除的关节索引
    indices_to_remove = [10, 11, 22, 23]
    # 计算要删除的索引，每个关节由3个值表示
    indices = [i for j in indices_to_remove for i in range(j * 3, j * 3 + 3)]

    # 删除指定的关节
    processed_pose = np.delete(pose, indices, axis=2)

    # 将姿态数据与平移数据连接
    processed_pose = np.concatenate([trans, processed_pose], axis=2)

    return processed_pose


def keypoint_3d_loss(criterion, pred_keypoints_3d, gt_keypoints_3d):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, 0:1, :]
    pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, 0:1, :]

    return criterion(pred_keypoints_3d * 100, gt_keypoints_3d * 100).mean()


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')

    net_pred = PhysMoP(hist_length=10,
                       physics=False,
                       data=True,
                       fusion=False
                       )

    smplModel = SMPL(device='cuda')

    net_pred.cuda()

    if opt.checkpoint_path is not None:
        print('>>> loading pretrained model')
        checkpoint = torch.load(opt.checkpoint_path)
        net_pred.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.Datasets(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, smplModel, optimizer, is_train=0, data_loader=data_loader, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            # ret_valid = run_model(net_pred, smplModel, is_train=1, data_loader=valid_loader, opt=opt)
            # print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model(net_pred, smplModel, is_train=3, data_loader=test_loader, opt=opt)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            # for k in ret_valid.keys():
            #     ret_log = np.append(ret_log, [ret_valid[k]])
            #     head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_test['#10'] < err_best:
                err_best = ret_test['#10']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_test['#10'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_pred, smplModel, optimizer=None, is_train=0, data_loader=None, opt=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    criterion_mae = nn.L1Loss().cuda()
    l_p3d = 0
    l_retore = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n

    # 去除7，9，13，14关节
    G2Hpose_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26,
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                   54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71
                   ]
    # joints at same loc

    st = time.time()
    for i, (pose, shape, trans) in enumerate(data_loader):
        # for i, (angle) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, D = pose.shape
        process_size = batch_size * seq_n

        if batch_size == 1 and is_train == 0:
            continue

        n += batch_size
        bt = time.time()

        gt_shape = shape.view(process_size, 10).float().cuda()

        gt_pose = torch.zeros([batch_size, seq_n, 72])
        gt_pose[:, :, G2Hpose_idx] = pose[:, :, G2Hpose_idx].float()

        in_pose = torch.cat((trans, gt_pose[:, :, G2Hpose_idx]), dim=2).cuda()

        gt_pose = gt_pose.view(process_size, 72).float().cuda()

        # pose = pose.view(process_size, 72).float().cuda()
        # joints, joints_smpl = forward_kinematics(smplModel, pose, gt_shape,
        #                                          joints_smpl=True)
        gt_joints, gt_joints_smpl = forward_kinematics(smplModel, gt_pose, gt_shape,
                                                       joints_smpl=True)
        motion_pred_data = net_pred(
            in_pose.float())

        pred_pose_data = torch.zeros([batch_size, seq_n, 72]).float().cuda()
        pred_pose_data[:, :, G2Hpose_idx] = motion_pred_data[:, :, 3:]
        pred_pose_data = pred_pose_data.view(process_size, 72)

        pred_joints_data, pred_joints_smpl_data = forward_kinematics(smplModel, pred_pose_data, gt_shape,
                                                                     joints_smpl=True)

        # joints_to_remove = [1, 2, 3, 13, 14]
        # mask = torch.ones(24, dtype=torch.bool)
        # mask[joints_to_remove] = False
        # gt_joints_smpl = gt_joints_smpl[:, mask, :]
        # pred_joints_smpl_data = pred_joints_smpl_data[:, mask, :]

        # 2d joint loss:
        # grad_norm = 0
        if is_train == 0:
            loss_keypoints_data = keypoint_3d_loss(criterion_mae, pred_joints_data, gt_joints)
            loss_pose_data = criterion_mae(pred_pose_data, gt_pose).mean()
            # l_p3d += loss_p3d.cpu().data.numpy() * batch_size
            loss_all = 5000 * (loss_keypoints_data + loss_pose_data)
            # loss_all = 5000 * loss_keypoints_data
            optimizer.zero_grad()
            loss_all.backward()
            # nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = keypoint_3d_loss(criterion_mae, pred_joints_smpl_data[:, in_n:in_n + out_n],
                                             gt_joints_smpl[:, in_n:in_n + out_n]) * 5000
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            gt_J = gt_joints.detach().cpu().numpy()
            pred_J_data = pred_joints_data.detach().cpu().numpy()
            _, mpjpe_p3d_h36 = compute_errors(gt_J, pred_J_data, 0)
            error_test_data = np.array(mpjpe_p3d_h36).reshape([-1, seq_n])
            error_test_data = error_test_data[:, 10:]
            m_p3d_h36 += error_test_data.sum(axis=0)
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|'.format(i + 1, len(data_loader), time.time() - bt,
                                                       time.time() - st))
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n
        ret["l_retore"] = l_retore / n
    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
