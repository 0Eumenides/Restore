from utils import h36motion3d as datasets
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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def getMask(bs, input_n, mask_ratio=0.2):
    joint_indices = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
    # masked_joints_indices = [18, 19, 21, 22, 2, 3, 4, 5]
    masked_joints_indices = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
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

    net_restore = Restoration.Restoration(input_n=opt.input_n, d_model=64, num_stage=12,eta=6)

    net_restore.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_restore.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_restore.parameters()) / 1000000.0))



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

    # evaluation
    if opt.is_eval:
        restore = run_model(net_restore, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        ret_log = np.append(ret_log,[restore])
        head = np.append(head, ["restore"])
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
            ret_train = run_model(net_restore, optimizer, is_train=0, data_loader=data_loader, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_restore,is_train=1, data_loader=valid_loader, opt=opt)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model(net_restore, is_train=3, data_loader=test_loader, opt=opt)
            print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            ret_log = np.append(ret_log, [ret_train['m_p3d_h36']])
            head = np.append(head, ['train'])
            ret_log = np.append(ret_log, [ret_valid['m_p3d_h36']])
            head = np.append(head, ['valid'])
            ret_log = np.append(ret_log, [ret_test['m_p3d_h36']])
            head = np.append(head, ['test'])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid,
                           'state_dict': net_restore.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_restore,optimizer=None, is_train=0, data_loader=None, opt=None):
    if is_train == 0:
        net_restore.train()

    l_retore = 0

    m_p3d_h36 = 0

    n = 0
    in_n = opt.input_n

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])  # 66


    st = time.time()
    for i, (p3d_h36) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()

        p3d_h36 = p3d_h36.float().cuda()

        p3d_src = p3d_h36.clone()[:, :, dim_used]

        with torch.no_grad():
            mask = getMask(batch_size, in_n, 0.4)
            src = p3d_src[:,:in_n].view(batch_size, in_n, 22, 3)
            start = src[:, in_n - 1:in_n]
            # x = src * mask[:, :, :, None] + \
            #     (1 - mask[:, :, :, None]) * self.defaultValue(label)
            x = src * mask[:, :, :, None]

        input_gcn, restore_loss = net_restore(x,src,mask,start)


        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            l_retore += restore_loss
            loss_all = restore_loss
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_restore.parameters()), max_norm=1)
            optimizer.step()
            # update log values

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            m_p3d_h36 += restore_loss.cpu().data.numpy() * batch_size
        else:
            m_p3d_h36 += restore_loss.cpu().data.numpy() * batch_size
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))
    ret = {}
    if is_train == 0:
        ret["m_p3d_h36"] = l_retore / n
    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
