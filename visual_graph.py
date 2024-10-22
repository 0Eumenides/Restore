import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import random
import numpy as np
import torch
def rotate_z(p3d, angle):
    """ 旋转关节数据围绕z轴 """
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # 构造旋转矩阵
    rotation_matrix = torch.tensor([
        [cos_angle, 0, sin_angle],  # 对应x轴和z轴的旋转
        [0, 1, 0],  # 保持z轴不变
        [-sin_angle, 0, cos_angle]  # 对应x轴和z轴的旋转
    ], dtype=torch.float32, device=p3d.device)

    # 应用旋转矩阵
    return torch.matmul(p3d, rotation_matrix)

def create_pose(ax, plots, vals, pred=True, update=False):
    connect = [
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10),
        (0, 1), (0, 6),
        (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
        (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (24, 25), (24, 17),
        (24, 14), (14, 15)
    ]

    LR = [
        False, True, True, True, True,
        True, False, False, False, False,
        False, True, True, True, True,
        True, True, False, False, False,
        False, False, False, False, True,
        False, True, True, True, True,
        True, True
    ]

    # Start and endpoints of our representation
    I = np.array([touple[0] for touple in connect])
    J = np.array([touple[1] for touple in connect])
    # Left / right indicator
    LR = np.array([LR[a] or LR[b] for a, b in connect])
    if not pred:
        lcolor = "#9b59b6"
        rcolor = "#9b59b6"
        # rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#8e8e8e"
        # rcolor = "#383838"

    for i in np.arange(len(I)):
        x = np.array([vals[I[i], 0], vals[J[i], 0]])
        z = np.array([vals[I[i], 1], vals[J[i], 1]])
        y = np.array([vals[I[i], 2], vals[J[i], 2]])
        if not update:

            if i == 0:
                plots.append(ax.plot(x, y, z, lw=2, linestyle='--', c=lcolor if LR[i] else rcolor,
                                     label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2, linestyle='--', c=lcolor if LR[i] else rcolor))

        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)

    return plots

def update(num, data_gt, plots_gt, fig, ax):
    gt_vals = data_gt[num]
    plots_gt = create_pose(ax, plots_gt, gt_vals, pred=False, update=True)

    r = 0.75
    xroot, zroot, yroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
    ax.set_xlim3d([-r + xroot, r + xroot])
    ax.set_ylim3d([-r + yroot, r + yroot])
    ax.set_zlim3d([-r + zroot, r + zroot])
    # ax.set_title('pose at time frame: '+str(num))
    # ax.set_aspect('equal')

    return plots_gt

def visualize(data_gt):

    out_n = 1


    data_gt = data_gt.cpu().detach().numpy()


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=20, azim=-40)
    vals = np.zeros((32, 3))  # or joints_to_consider
    gt_plots = []
    pred_plots = []

    gt_plots = create_pose(ax, gt_plots, vals, pred=False, update=False)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc='lower left')

    # ax.set_xlim3d([0, 0.5])
    ax.set_xlim3d([-1, 1.5])
    ax.set_xlabel('X')

    # ax.set_ylim3d([0, 0.5])
    ax.set_ylim3d([-1, 1.5])
    ax.set_ylabel('Y')

    # ax.set_zlim3d([0, 0.5])
    ax.set_zlim3d([0.0, 1.5])
    ax.set_zlabel('Z')

    line_anim = animation.FuncAnimation(fig, update, out_n, fargs=(data_gt, gt_plots,
                                                                   fig, ax), interval=70, blit=False)
    plt.show()

    save_folder = './visualizations/pred10'
    os.makedirs(save_folder, exist_ok=True)

    animation_file_path = os.path.join(save_folder, 'H36M_'+str(random.randint(1,1000))+'.png')
    line_anim.save(animation_file_path, writer='pillow')


from torch.utils.data import DataLoader
# def load_dataset(file_name):
#     try:
#         with open(file_name, 'rb') as f:
#             dataset = pickle.load(f)
#         print("Dataset loaded successfully.")
#         return dataset
#     except FileNotFoundError:
#         print("No saved dataset file found.")
#         return None
#
# dataset = load_dataset('train.pkl')
from utils import h36motion3d as datasets
from argparse import Namespace

opt = Namespace(input_n=10, output_n=10,test=True,skip_rate=2)

dataset = datasets.Datasets(opt, split=0)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
for i, (p3d_h36) in enumerate(data_loader):
    p3d_h36 = p3d_h36.view(32, 20, 32, 3)
    input = p3d_h36[3, 10:11,:,:]/1000
    visualize(input)

    p3d_flipped = input.clone()
    p3d_flipped[:, :, 0] = -p3d_flipped[:, :, 0]  # 翻转x坐标
    p3d_flipped[:, :, 2] = -p3d_flipped[:, :, 2]
    visualize(p3d_flipped)

    p3d_rotated = rotate_z(input, 30)
    visualize(p3d_rotated)
    break
