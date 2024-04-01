from torch.nn import Module
from torch import nn
import torch
from model.GCNII import GC_Block, GraphConvolution_O
from model.Layers import PositionalEncoder, DefaultValue
from model.attention import TemporalAdditiveAttention


# 固定节点遮掩
# def getMask(bs, input_n):
#     joint_indices = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
#     masked_joints = [18, 19, 21, 22,2, 3, 4, 5]  # 需要遮掩的关节编号
#
#     # 创建形状为 (32, 50, 66) 的遮掩数组，初始全部设置为 1
#     mask = torch.ones((bs, input_n, 66))
#
#     # 遍历需要遮掩的关节编号
#     for joint in masked_joints:
#         if joint in joint_indices:
#             idx = joint_indices.index(joint)  # 找到关节编号在列表中的索引
#             start_idx = idx * 3
#             end_idx = start_idx + 3
#             mask[:, :, start_idx:end_idx] = 0  # 将遮掩关节的对应位置设置为 0
#
#     return mask.cuda()

# 随机mask
# def getMask(bs, node_n, input_n, mask_ratio=0.2):
#     # 创建形状为 (32, 50, 66) 的遮掩数组，初始全部设置为 1
#     mask = torch.ones((bs, input_n * node_n))
#     shuffle_indices = torch.rand((bs, node_n * input_n)).argsort().cuda()
#     mask_indices = shuffle_indices[:, :int(node_n * input_n * mask_ratio)]
#     batch_ind1 = torch.arange(bs)[:, None].cuda()
#     mask[batch_ind1, mask_indices] = 0
#     mask = mask.view(bs, input_n, -1)
#     return mask.cuda()

# 遮掩的关节为0
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


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask[:, :, :, None]) / (torch.sum(mask) + 1e-9)


def invert_mask(mask):
    """
    Invert a mask, changing 1s to 0s and 0s to 1s.

    Parameters:
    mask (torch.Tensor): The mask to invert.

    Returns:
    torch.Tensor: The inverted mask.
    """
    # 确保 mask 是一个 torch.Tensor
    if not isinstance(mask, torch.Tensor):
        raise TypeError("Mask needs to be a torch.Tensor")

    # 取反掩码
    inverted_mask = 1 - mask

    return inverted_mask


class GCNAttentionBlock(Module):
    def __init__(self, d_model, node_n, stage, input_n):
        super(GCNAttentionBlock, self).__init__()
        self.GCN = GC_Block(d_model, p_dropout=0.3, node_n=node_n, l=stage + 1, input_n=input_n)
        self.self_attention = TemporalAdditiveAttention(
            input_size=d_model,
            output_size=d_model,
            msg_size=d_model,
            msg_layers=1,
            reweight='softmax',
            dropout=0.0,
            root_weight=False,
            norm=False
        )
        self.self_attention.reset_parameters()

    def forward(self, x, h0, mask=None):
        y = self.GCN(x, h0, mask)
        if mask is not None:
            mask = mask.unsqueeze(-1)
        # mask = mask.permute(0, 2, 1)
        # batch_size, num_joints, num_timesteps = mask.shape
        # mask = mask.unsqueeze(3).expand(-1, -1, -1, num_timesteps).contiguous().view(batch_size,num_joints,num_timesteps*num_timesteps)
        y = self.self_attention(y, mask=mask)
        return y + x


class Restoration(Module):

    def __init__(self, input_n=10, d_model=512, num_stage=2, eta=0):
        super(Restoration, self).__init__()
        self.input_n = input_n
        self.num_stage = num_stage
        self.eta = eta
        self.embedding = PositionalEncoder(in_channels=3, out_channels=d_model, n_layers=2, n_nodes=22)
        self.defaultValue = DefaultValue(label_num=15, node_n=22, d_model=d_model)
        self.encoder = []
        for i in range(num_stage):
            self.encoder.append(GCNAttentionBlock(d_model=d_model, node_n=22, stage=i, input_n=input_n))

        self.encoder = nn.ModuleList(self.encoder)

        self.start_gc = GraphConvolution_O(d_model, d_model, node_n=22)
        self.end_gc = GraphConvolution_O(d_model, 3, node_n=22)
        self.bn1 = nn.BatchNorm1d(d_model * 22 * 10)
        self.act_f = nn.Tanh()
        self.do = nn.Dropout(0.3)
        self.endLinear = nn.Linear(d_model, 3)

    def forward(self, x, src, mask,start):
        bs = src.shape[0]

        # mask = getMask(bs, self.input_n, 0.4)
        # src = src.view(bs, 10, 22, 3)
        # start = src[:, self.input_n - 1:self.input_n]
        # x = src * mask[:, :, :, None] + \
        #     (1 - mask[:, :, :, None]) * self.defaultValue(label)
        # x = src * mask[:, :, :, None]

        y = self.embedding(x)

        h0 = y

        for l in range(self.num_stage):
            if l < self.eta:
                y = self.encoder[l](y, h0, mask)
            else:
                y = self.encoder[l](y, h0)

        y = self.endLinear(y)

        y = (y + start) * (1 - mask[:, :, :, None]) + x * mask[:, :, :, None]

        # reconstruction_loss = masked_mae_cal(y, src, mask)
        imputation_MAE = masked_mae_cal(y, src, invert_mask(mask))

        return y.view(bs, self.input_n, -1), imputation_MAE
