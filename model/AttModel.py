from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import utils.util as util
import numpy as np
import torch.nn.functional as F
from model.Restoration import Restoration


class AttModel(Module):

    def __init__(self, in_features=48, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)

        self.gcn = GCN.GCN(input_feature=dct_n, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)
    # def forward(self, src, label, output_n=10, input_n=10):
    def forward(self, src, output_n=10, input_n=10):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n  # 20
        src = src[:, :input_n]  # [bs,in_n,dim]torch.Size([32, 50, 48])
        src_tmp = src.cuda().clone()

        dct_m, idct_m = util.get_dct_matrix(input_n + output_n)  # (20, 20)

        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        idx = list(range(-input_n, 0, 1)) + [-1] * output_n  # 20
        outputs = []


        input_gcn = src_tmp[:, idx]
        dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)  # torch.Size([32, 48, 20])
        dct_out_tmp = self.gcn(dct_in_tmp)  # torch.Size([32, 48, 40])
        out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                               dct_out_tmp[:, :, :dct_n].transpose(1, 2))  # torch.Size([32, 20, 48])

        outputs.append((out_gcn).unsqueeze(2))  # torch.Size([32, 20, 1, 48])

        outputs = torch.cat(outputs, dim=2)  # torch.Size([32, 20, 1, 48])
        return outputs
