#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn import init
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
import numpy as np


class GraphConvolution_O(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution_O, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # 40,40
        self.att = Parameter(torch.FloatTensor(node_n, node_n))  # 48,48
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphConvolution_Old(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48, residual=False, variant=False, l=1):
        super(GraphConvolution, self).__init__()
        self.variant = True  # variant#
        if self.variant:  #
            self.in_features = 2 * in_features  #
        else:  #
            self.in_features = in_features  #
        self.residual = residual  #
        self.l = l  #
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, out_features))  # 40,40
        self.att = Parameter(torch.FloatTensor(node_n, node_n))  # 48,48
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, h0):
        lamda = 1.4  # 0.5
        alpha = 0.5

        theta = math.log(lamda / self.l + 1)

        hi = torch.matmul(self.att, input)  # torch.Size([32, 60, 256])

        if self.variant:
            support = torch.cat([hi, h0], 2)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.matmul(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input

        # support = torch.matmul(input, self.weight)
        # output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def get_adjacency_matrix():
    # 自连接
    self_link = [(i, i) for i in range(22)]
    # 骨骼连接
    bone_link = [(8, 0), (0, 1), (1, 2), (2, 3),
                 (8, 4), (4, 5), (5, 6), (6, 7),
                 (8, 9), (9, 10), (10, 11),
                 (9, 12), (12, 13), (13, 14), (14, 15), (15, 16),
                 (9, 17), (17, 18), (18, 19), (19, 20), (20, 21)]
    # 语义连接
    sem_link = [(13, 18), (12, 17), (0, 4), (1, 5)]

    # # 自连接
    # self_link = [(i, i) for i in range(23)]
    # # 骨骼连接
    # bone_link = [(0, 3), (3, 6), (6, 9),
    #              (1, 4), (4, 7), (7, 10),
    #              (2, 5), (5, 8), (8, 11), (11, 14),
    #              (8, 12), (12, 15), (15, 17), (17, 19), (19, 21),
    #              (8, 13), (13, 16), (16, 18), (18, 20), (20, 22)]
    # # 语义连接
    # sem_link = [(3, 4), (6, 7), (9, 10),
    #             (17, 18), (19, 20), (21, 22)]

    # 所有连接
    all_link = self_link + bone_link + sem_link

    # 创建邻接矩阵
    adjacency_matrix = {}
    for joint1, joint2 in all_link:
        adjacency_matrix[(joint1, joint2)] = 1
        adjacency_matrix[(joint2, joint1)] = 1

    # 创建一个列表来表示邻接矩阵
    adjacency_matrix_list = []
    # 将邻接矩阵的元素按行放入列表中
    for i in range(22):
        row = []
        for j in range(22):
            # 如果 (i, j) 是邻接矩阵中的元素，添加值；否则添加0
            row.append(adjacency_matrix.get((i, j), 0))
        adjacency_matrix_list.append(row)

    # 返回二维列表表示的邻接矩阵
    return adjacency_matrix_list


class AttLayer(nn.Module):
    def __init__(self, out_channels, use_bias=False, reduction=16):
        super(AttLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, 1, bias=False),
            nn.Hardsigmoid()
        )

    def reset_parameters(self):
        init.normal_(self.fc[0].weight, 0, 0.01)
        init.normal_(self.fc[2].weight, 0, 0.01)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1)
        return x * y.expand_as(x)


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48, residual=False, variant=False, l=1):
        super(GraphConvolution, self).__init__()
        self.node_n = node_n
        self.variant = True  # variant#
        if self.variant:  #
            self.in_features = 2 * in_features  #
        else:  #
            self.in_features = in_features  #
        self.residual = residual  #
        self.l = l  #
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, out_features))  # 40,40
        self.att = Parameter(torch.FloatTensor(node_n, node_n))  # 48,48

        self.M = Parameter(torch.FloatTensor(node_n, node_n))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias_explicit = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_explicit', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, h0, mask=None):
        lamda = 1.4  # 0.5
        alpha = 0.5

        theta = math.log(lamda / self.l + 1)
        att_explicit = get_adjacency_matrix()
        att_explicit = torch.from_numpy(np.array(att_explicit)).float().cuda()
        Adj = self.att + att_explicit
        Adj = torch.where(torch.isnan(Adj), torch.full_like(Adj, 0), Adj)
        Adj_W = torch.mul(Adj, self.M)

        if mask is not None:
            mask = mask.unsqueeze(-2)  # 现在形状是[批大小, 时间步, 1, 关节数]
            mask = mask.repeat(1, 1, self.node_n, 1)  # 重复以匹配Adj_W的形状
            diag_values = Adj_W.diagonal()
            Adj_W = Adj_W * mask
            Adj_W.diagonal(dim1=-2, dim2=-1).copy_(diag_values)

        hi = torch.matmul(Adj_W, input)  # torch.Size([32, 60, 256])

        if self.variant:
            support = torch.cat([hi, h0], 3)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.matmul(support, self.weight) + (1 - theta) * r

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, input_n, bias=True, node_n=48, l=1):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution_O(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(in_features * node_n * input_n)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, l=l)
        self.bn2 = nn.BatchNorm1d(in_features * node_n * input_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, h0, mask=None):
        y = self.gc2(x, h0, mask)
        b, t, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, t, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y, h0, mask)
        b, t, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, t, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution_O(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(self.num_stage):  # 12
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n, l=i + 1))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution_O(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.act_fn = nn.ReLU()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(input_feature, hidden_feature))
        self.fcs.append(nn.Linear(hidden_feature, input_feature))

    def forward(self, x):  # torch.Size([32, 48, 40])
        # h0=F.dropout(x, 0.5)
        # h0=self.act_fn(self.fcs[0](h0))#torch.Size([32, 60,256])
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        h0 = y

        for i in range(self.num_stage):
            y = self.gcbs[i](y, h0)

        y = self.gc7(y)  # torch.Size([32, 48, 40])
        # y =self.fcs[-1](y)

        y = y + x

        return y
