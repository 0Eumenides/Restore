from torch.utils.data import Dataset
import numpy as np
import json


class Datasets(Dataset):

    def __init__(self, opt, actions=None, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        # 选择训练集、测试集、验证集
        self.split = split
        self.in_n = opt.input_n
        self.out_n = opt.output_n
        self.sample_rate = 2
        self.p3d = {}
        self.shape = {}
        self.trans  = {}
        # 用来保存动作标签
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        if opt.test:
            acts = ["walking"]
            subs = np.array([[1], [11], [5]])
        else:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
            subs = [[1, 6, 7, 8, 9], [11], [5]]
        if actions is not None:
            acts = actions

        subs = subs[split]
        key = 0
        for subj in subs:
            # with open('/home/eniac/data/CodeTest/dth/Restore/SMPL/Human36M_subject' + str(subj) + '_SMPL_NeuralAnnot.json',
            with open('/code/dth/Restore/SMPL/Human36M_subject' + str(subj) + '_SMPL_NeuralAnnot.json',
                      'r') as f:
                smpl_params = json.load(f)
            for action_idx in np.arange(2,len(acts)+2):
                # subj5需要单独处理£
                for subact in [1, 2]:  # subactions
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action_idx, subact))
                    the_sequence = smpl_params[str(action_idx)][str(subact)]
                    poses = np.array([frame_data['pose'] for frame_data in the_sequence.values()])
                    shapes = np.array([frame_data['shape'] for frame_data in the_sequence.values()])
                    trans = np.array([frame_data['trans'] for frame_data in the_sequence.values()])
                    n= len(the_sequence)
                    even_list = range(0, n, self.sample_rate)
                    num_frames = len(even_list)
                    poses = np.array(poses[even_list])
                    shapes = np.array(shapes[even_list])
                    trans = np.array(trans[even_list])

                    self.p3d[key] = poses
                    self.shape[key] = shapes
                    self.trans[key] = trans

                    # 记录的是开始帧的索引
                    valid_frames = np.arange(0, num_frames - seq_len + 1, opt.skip_rate)

                    # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 1

        # ignore constant joints and joints at same position with other joints

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs], self.shape[key][fs],self.trans[key][fs]
