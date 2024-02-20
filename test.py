from torch.utils.data import Dataset
import numpy as np
import torch
import os
from utils import ang2joint
import matplotlib.pyplot as plt
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
    for i in range(len(joints)):
        ax.text(x[i], y[i], z[i], f'{i}', color='blue', fontsize=10)

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
    fig.savefig('joints_amass_plot' + str(ID) + '.png')

amass_splits = [
    ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD'],
    ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    ['BioMotionLab_NTroje'],
]
path_to_data = "/data2/dth/dataset/data_raw/AMASS/"
split = 0
skel = np.load('./body_models/smpl_skeleton.npz')
p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()
parents = skel['parents']
parent = {}
for i in range(len(parents)):
    parent[i] = parents[i]
n = 0
for ds in amass_splits[split]:
    if not os.path.isdir(path_to_data + ds):
        print(ds)
        continue
    print('>>> loading {}'.format(ds))
    for sub in os.listdir(path_to_data + ds):
        if not os.path.isdir(path_to_data + ds + '/' + sub):
            continue
        for act in os.listdir(path_to_data + ds + '/' + sub):
            if not act.endswith('.npz'):
                continue
            # if not ('walk' in act or 'jog' in act or 'run' in act or 'treadmill' in act):
            #     continue
            pose_all = np.load(path_to_data + ds + '/' + sub + '/' + act)
            try:
                poses = pose_all['poses']
            except:
                print('no poses at {}_{}_{}'.format(ds, sub, act))
                continue
            frame_rate = pose_all['mocap_framerate']
            # gender = pose_all['gender']
            # dmpls = pose_all['dmpls']
            # betas = pose_all['betas']
            # trans = pose_all['trans']
            fn = poses.shape[0]
            sample_rate = int(frame_rate // 25)
            fidxs = range(0, fn, sample_rate)
            fn = len(fidxs)
            poses = poses[fidxs]
            poses = torch.from_numpy(poses).float().cuda()
            poses = poses.reshape([fn, -1, 3])
            # remove global rotation
            poses[:, 0] = 0
            p3d0_tmp = p3d0.repeat([fn, 1, 1])
            p3d = ang2joint.ang2joint(p3d0_tmp, poses, parent)
            p3d = p3d[:,:22,:]
            visual(p3d[5],1)
            print(1)