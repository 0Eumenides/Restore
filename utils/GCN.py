#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from utils.util import chunkponosig


# 最基本的图卷积操作
class Graph(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(Graph, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
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


class NatureGraph(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(NatureGraph, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # 学习隐藏连接
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        # 自然连接
        self.adjacency_matrix = torch.tensor(self.get_adjacency_matrix(), dtype=torch.float, requires_grad=False).cuda()
        # 控制连接强度
        self.M = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def get_adjacency_matrix(self):
        self_link = [(i, i) for i in range(self.in_features // 3)]
        bone_link = [(8, 0), (0, 1), (1, 2), (2, 3),
                     (8, 4), (4, 5), (5, 6), (6, 7),
                     (8, 9), (9, 10), (10, 11),
                     (9, 12), (12, 13), (13, 14), (14, 15), (15, 16),
                     (9, 17), (17, 18), (18, 19), (19, 20), (20, 21)]
        sem_link = [(13, 18), (12, 17), (0, 4), (1, 5)]
        all_link = self_link + bone_link + sem_link
        adjacency_matrix = {}
        # 建立连接关节之间的xyz坐标的邻接矩阵
        for joint1, joint2 in all_link:
            for dim1 in range(3):
                for dim2 in range(3):
                    adjacency_matrix[(joint1 * 3 + dim1, joint2 * 3 + dim2)] = 1
                    adjacency_matrix[(joint2 * 3 + dim2, joint1 * 3 + dim1)] = 1

        # 创建一个列表来表示邻接矩阵
        adjacency_matrix_list = []
        # 将邻接矩阵的元素按行放入列表中
        for i in range(66):
            row = []
            for j in range(66):
                # 如果 (i, j) 是邻接矩阵中的元素，添加值；否则添加0
                if (i, j) in adjacency_matrix:
                    row.append(adjacency_matrix[(i, j)])
                else:
                    row.append(0)
            adjacency_matrix_list.append(row)

        # 现在 adjacency_matrix_list 是一个二维列表表示的邻接矩阵
        return adjacency_matrix_list

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        self.M.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        Adj = self.att + self.adjacency_matrix
        # 将NAN值替换为0
        Adj = torch.where(torch.isnan(Adj), torch.full_like(Adj, 0), Adj)
        Adj_W = torch.mul(Adj, self.M)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(Adj_W, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# 图卷积块，由两个图卷积层组成
class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = Graph(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = Graph(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# Pono图卷积块
class PonoGC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = Graph(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = Graph(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        out1 = torch.cat((y, x), dim=2)
        out1 = chunkponosig(out1)
        return out1

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# GCN模型，由多个图卷积块组成
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

        self.gc1 = Graph(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = Graph(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, is_out_resi=True):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        if is_out_resi:
            y = y + x
        return y


# 跳跃连接GCN
class SkipGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(SkipGCN, self).__init__()
        self.num_stage = num_stage
        self.skip_convs = nn.ModuleList()
        self.gc1 = Graph(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))
            if i < num_stage - 1:
                self.skip_convs.append(Graph(hidden_feature, hidden_feature, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = Graph(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.skipE = Graph(hidden_feature, input_feature, node_n=node_n)

    def forward(self, x, is_out_resi=True):
        skip = 0
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)
            if i < self.num_stage - 1:
                s = y
                skip = skip + self.skip_convs[i](s)

        skip = self.skipE(skip)
        y = self.gc7(y)
        if is_out_resi:
            y = y + x + skip
        return y


# 对称残差连接，12个GCN块
class SymResGCN(nn.Module):
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

        self.gc1 = Graph(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = Graph(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, is_out_resi=True):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        # 对称残差连接
        retain = [y.detach()]
        for i in range(self.num_stage):
            y = self.gcbs[i](y)
            if i < 5:
                retain.append(y.detach())
            if i > 5:
                y = y + retain[11 - i]

        y = self.gc7(y)
        if is_out_resi:
            y = y + x
        return y


# 多层残差图结构
class MultiResGraph(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48, residual=False, variant=False, l=1):
        super(MultiResGraph, self).__init__()
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


class MultiResGC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48, l=1):
        """
        Define a residual block of GCN
        """
        super(MultiResGC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = Graph(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = MultiResGraph(in_features, in_features, node_n=node_n, bias=bias, l=l)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, h0):
        y = self.gc2(x, h0)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y, h0)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class MultiResGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(MultiResGCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = Graph(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(self.num_stage):  # 12
            self.gcbs.append(MultiResGC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n, l=i + 1))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = Graph(hidden_feature, input_feature, node_n=node_n)

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
