import pickle

from utils import h36motion as datasets, data_utils
from model import AttModel, Restoration
from utils.opt import Options
from utils import util
from utils import log
from utils import loss
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim
import os

import utils.forward_kinematics as fk
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
smooth_sigma = 6
smooth_sigma_va = 8

def save_dataset(dataset, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(file_name):
    try:
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
        print("Dataset loaded successfully.")
        return dataset
    except FileNotFoundError:
        print("No saved dataset file found.")
        return None

def getMask(bs, input_n, mask_ratio=0.2):
    joint_indices = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
    masked_joints_indices = [18, 19, 21, 22, 2, 3, 4, 5]
    # 将全局关节编号转换为局部索引
    masked_joint_local_indices = [joint_indices.index(joint) for joint in masked_joints_indices]

    # 创建遮掩数组，初始全部设置为 1
    mask = torch.ones((bs, input_n, len(joint_indices)))

    # 计算每个关节需要遮掩的数量
    num_to_mask_per_joint = int(input_n * mask_ratio)

    # 遍历需要遮掩的关节索引
    for joint in masked_joint_local_indices:
        # 对每个批次中的每个关节进行遮掩
        for i in range(bs):
            # 随机选择需要遮掩的时间步
            time_steps_to_mask = torch.randperm(input_n)[:num_to_mask_per_joint]
            mask[i, time_steps_to_mask, joint] = 0

    return mask.cuda()

def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 66
    d_model = opt.d_model
    net_pred = AttModel.AttModel(in_features=in_features, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_restore = Restoration.Restoration(input_n=10, d_model=64, num_stage=12,eta=6)

    net_restore.cuda()
    net_pred.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
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
        dataset = load_dataset('train.pkl')
        if dataset is None:
            dataset = datasets.Datasets(opt, split=0)
            save_dataset(dataset, 'train.pkl')
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        valid_dataset = load_dataset('valid.pkl')
        if valid_dataset is None:
            valid_dataset = datasets.Datasets(opt, split=1)
            save_dataset(valid_dataset, 'valid.pkl')

        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

    test_dataset = load_dataset('test.pkl')
    if test_dataset is None:
        test_dataset = datasets.Datasets(opt, split=2)
        save_dataset(test_dataset, 'test.pkl')
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred,net_restore, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_pred, net_restore,is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model(net_pred,net_restore, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_pred, net_restore,optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.input_n
    # joints at same loc

    itera = 1
    st = time.time()

    dim_used_p3d = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

    dim_used_angle=np.array([6,  7,   8, 9,  10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23,
                         24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41,
                         42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                         60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])-3  # 60

    for i, (p3d_h36) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()

        euler_smooth = p3d_h36.float().cuda()


        euler_smooth = euler_smooth.view(-1,96)

        with torch.no_grad():
            eulerToRot = util.eulerToRot(euler_smooth)
            gt_pose = data_utils.rot2xyz_torch(eulerToRot)
            gt_pose = gt_pose.view(batch_size, in_n + out_n, 32, 3)

        euler_smooth = euler_smooth.view(batch_size, in_n + out_n, 96)

        p3d_src = euler_smooth.clone()[:, :, dim_used_angle]
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n)
        p3d_out_all = p3d_out_all[:,:,0]

        p3d_out = euler_smooth.clone()
        p3d_out[:, :, dim_used_angle] = p3d_out_all

        p3d_out = p3d_out.reshape(-1, 96)
        with torch.no_grad():
            eulerToRot = util.eulerToRot(p3d_out)
            pre_pose = data_utils.rot2xyz_torch(eulerToRot)
            pre_pose = pre_pose.view(batch_size, in_n + out_n, 32, 3)

        p3d_sup = euler_smooth.clone()[:, :,dim_used_angle]

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            # 3d坐标
            loss_p3d = torch.mean(torch.norm(pre_pose - gt_pose, dim=3))

            # 旋转角
            # loss_ang = torch.mean(torch.norm(p3d_out_all - p3d_sup, dim=2))
            loss_ang = torch.mean(torch.sum(torch.abs(p3d_out_all - p3d_sup), dim=2))
            # loss_all = loss_p3d + restore_loss
            loss_all = loss_ang + loss_p3d

            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(pre_pose[:,in_n:] - gt_pose[:,in_n:], dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(pre_pose[:,in_n:] - gt_pose[:,in_n:], dim=3), dim=2), dim=0)
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(pre_pose[:,in_n:] - gt_pose[:,in_n:], dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))

    ret = {}
    if is_train == 0:
        ret["m_p3d_h36"] = m_p3d_h36 / n
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
