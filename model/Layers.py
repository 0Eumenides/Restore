import torch
import torch.nn as nn
from einops import rearrange
import math
from torch import Tensor
from typing import Optional, Union, List, Tuple


class PositionalEncoder(nn.Module):

    def __init__(self, in_channels, out_channels,
                 n_layers: int = 1,
                 n_nodes: Optional[int] = None):
        super(PositionalEncoder, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.mlp = MLP(out_channels, out_channels, out_channels,
                       n_layers=n_layers, activation='relu')
        self.positional = PositionalEncoding(out_channels)
        if n_nodes is not None:
            self.node_emb = StaticGraphEmbedding(n_nodes, out_channels)
        else:
            self.register_parameter('node_emb', None)

    def forward(self, x, node_emb=None, node_index=None):
        if node_emb is None:
            node_emb = self.node_emb(node_index=node_index)
        # # x: [b s c], node_emb: [n c] -> [b s n c]
        # x = self.lin(x)
        # x = self.activation(x + node_emb)
        # out = self.mlp(x)
        # out = self.positional(out)

        x = self.lin(x)
        x = x + node_emb
        out = self.positional(x)
        return out


class DefaultValue(nn.Module):
    def __init__(self, label_num, input_n, node_n, d_model, ):
        super(DefaultValue, self).__init__()
        self.value = nn.Parameter(torch.randn(label_num, input_n, node_n, d_model), requires_grad=True)
        self.mlp = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def forward(self):
        """"""
        # center = self.compute_pose_center(src)
        return self.mlp(self.value)
        # center = self.compute_pose_center(src)  # 形状: (批大小, 时间步, 3)
        #
        # # 获取每个标签对应的默认节点值
        # default_value = self.value[label]  # 形状: (批大小, 节点数, d_model)
        #
        # # 处理默认值
        # default_value = self.mlp(default_value)  # 形状: (批大小, 节点数, 3)
        #
        # # 扩展中心到和节点数相同的维度
        # center = center.unsqueeze(2)  # 形状: (批大小, 时间步, 1, 3)
        # center = center.repeat(1, 1, default_value.size(1), 1)  # 形状: (批大小, 时间步, 节点数, 3)
        #
        # # 计算偏移
        # offset = default_value.unsqueeze(1) + center
        #
        # # 返回结果
        # return offset

    def reset_emb(self):
        with torch.no_grad():
            bound = 1.0 / math.sqrt(self.value.size(-1))
            self.value.data.uniform_(-bound, bound)

    def reset_parameters(self):
        self.reset_emb()

    def compute_pose_center(self, src):
        """
        计算每个时间步的姿态中心。

        参数:
            src (torch.Tensor): 形状为 (批大小, 时间步, 节点数, xyz坐标) 的张量，代表人体动作序列。

        返回:
            pose_centers (torch.Tensor): 形状为 (批大小, 时间步, xyz坐标) 的张量，代表每个时间步的姿态中心。
        """
        # 沿着节点数这一维度（dim=2）计算平均值
        pose_centers = src.mean(dim=2)
        return pose_centers


class PositionalEncoding(nn.Module):
    """The positional encoding from the paper `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ (Vaswani et al., NeurIPS 2017)."""

    def __init__(self,
                 d_model: int,
                 dropout: float = 0.,
                 max_len: int = 5000,
                 affinity: bool = False,
                 batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if affinity:
            self.affinity = nn.Linear(d_model, d_model)
        else:
            self.affinity = None
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x: Tensor):
        """"""
        if self.affinity is not None:
            x = self.affinity(x)
        if self.batch_first:
            pe = self.pe[:x.size(1), :]
        else:
            pe = self.pe[:x.size(0), :]
        x = x + pe
        return self.dropout(x)


class StaticGraphEmbedding(nn.Module):
    r"""Creates a table of node embeddings with the specified size.

    Args:
        n_nodes (int): Number of elements for which to store an embedding.
        emb_size (int): Size of the embedding.
        initializer (str or Tensor): Initialization methods.
            (default :obj:`'uniform'`)
        requires_grad (bool): Whether to compute gradients for the embeddings.
            (default :obj:`True`)
    """

    def __init__(self,
                 n_nodes: int,
                 emb_size: int,
                 initializer: Union[str, Tensor] = 'uniform',
                 requires_grad: bool = True):
        super(StaticGraphEmbedding, self).__init__()
        self.n_nodes = int(n_nodes)
        self.emb_size = int(emb_size)

        if isinstance(initializer, Tensor):
            self.initializer = "from_values"
            self.register_buffer('_default_values', initializer.float())
        else:
            self.initializer = initializer
            self.register_buffer('_default_values', None)

        self.emb = nn.Parameter(Tensor(self.n_nodes, self.emb_size),
                                requires_grad=requires_grad)

        self.reset_emb()

    def __repr__(self) -> str:
        return "{}(n_nodes={}, embedding_size={})".format(
            self.__class__.__name__, self.n_nodes, self.emb_size)

    def reset_emb(self):
        with torch.no_grad():
            if self.initializer == 'uniform' or self.initializer is None:
                bound = 1.0 / math.sqrt(self.emb.size(-1))
                self.emb.data.uniform_(-bound, bound)
            elif self.initializer == 'from_values':
                self.emb.data.copy_(self._default_values)
            else:
                raise RuntimeError(
                    f"Embedding initializer '{self.initializer}'"
                    " is not supported.")

    def reset_parameters(self):
        self.reset_emb()

    def get_emb(self):
        return self.emb

    def forward(self,
                expand: Optional[List] = None,
                node_index: Optional[torch.Tensor] = None,
                nodes_first: bool = True):
        """"""
        emb = self.get_emb()
        if node_index is not None:
            emb = emb[node_index]
        if not nodes_first:
            emb = emb.T
        if expand is None:
            return emb
        shape = [*emb.size()]
        view = [
            1 if d > 0 else shape.pop(0 if nodes_first else -1) for d in expand
        ]
        return emb.view(*view).expand(*expand)


class MLP(nn.Module):
    """Simple Multi-layer Perceptron encoder with optional linear readout.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int, optional): Size of the optional readout.
        exog_size (int, optional): Size of the optional exogenous variables.
        n_layers (int, optional): Number of hidden layers. (default: 1)
        activation (str, optional): Activation function. (default: `relu`)
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 exog_size=None,
                 n_layers=1,
                 activation='relu',
                 dropout=0.):
        super(MLP, self).__init__()

        if exog_size is not None:
            input_size += exog_size
        layers = [
            Dense(input_size=input_size if i == 0 else hidden_size,
                  output_size=hidden_size,
                  activation=activation,
                  dropout=dropout) for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def reset_parameters(self) -> None:
        """"""
        for module in self.mlp._modules.values():
            module.reset_parameters()
        if self.readout is not None:
            self.readout.reset_parameters()

    def forward(self, x, u=None):
        """"""
        x = maybe_cat_exog(x, u)
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out


def maybe_cat_exog(x, u, dim=-1):
    r"""
    Concatenate `x` and `u` if `u` is not `None`.

    We assume `x` to be a 4-dimensional tensor, if `u` has only 3 dimensions we
    assume it to be a global exog variable.

    Args:
        x: Input 4-d tensor.
        u: Optional exogenous variable.
        dim (int): Concatenation dimension.

    Returns:
        Concatenated `x` and `u`.
    """
    if u is not None:
        if u.dim() == 3:
            u = rearrange(u, 'b s f -> b s 1 f')
        x = expand_then_cat([x, u], dim)
    return x


def expand_then_cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]],
                    dim: int = -1) -> Tensor:
    """Match the dimensions of tensors in the input list and then concatenate.

    Args:
        tensors (list): Tensors to concatenate.
        dim (int): Dimension along which to concatenate.
            (default: -1)
    """
    shapes = [t.shape for t in tensors]
    expand_dims = torch.max(torch.tensor(shapes), 0).values
    expand_dims[dim] = -1
    tensors = [t.expand(*expand_dims) for t in tensors]
    return torch.cat(tensors, dim=dim)


class Dense(nn.Module):
    r"""A simple fully-connected layer implementing

    .. math::

        \mathbf{x}^{\prime} = \sigma\left(\boldsymbol{\Theta}\mathbf{x} +
        \mathbf{b}\right)

    where :math:`\mathbf{x} \in \mathbb{R}^{d_{in}}, \mathbf{x}^{\prime} \in
    \mathbb{R}^{d_{out}}` are the input and output features, respectively,
    :math:`\boldsymbol{\Theta} \in \mathbb{R}^{d_{out} \times d_{in}} \mathbf{b}
    \in \mathbb{R}^{d_{out}}` are trainable parameters, and :math:`\sigma` is
    an activation function.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        activation (str, optional): Activation function to be used.
            (default: :obj:`'relu'`)
        dropout (float, optional): The dropout rate.
            (default: :obj:`0`)
        bias (bool, optional): If :obj:`True`, then the bias vector is used.
            (default: :obj:`True`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str = 'relu',
                 dropout: float = 0.,
                 bias: bool = True):
        super(Dense, self).__init__()
        self.affinity = nn.Linear(input_size, output_size, bias=bias)
        self.activation = get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def reset_parameters(self) -> None:
        """"""
        self.affinity.reset_parameters()

    def forward(self, x):
        """"""
        out = self.activation(self.affinity(x))
        return self.dropout(out)


def get_layer_activation(activation: Optional[str] = None):
    _torch_activations_dict = {
        'elu': 'ELU',
        'leaky_relu': 'LeakyReLU',
        'prelu': 'PReLU',
        'relu': 'ReLU',
        'rrelu': 'RReLU',
        'selu': 'SELU',
        'celu': 'CELU',
        'gelu': 'GELU',
        'glu': 'GLU',
        'mish': 'Mish',
        'sigmoid': 'Sigmoid',
        'softplus': 'Softplus',
        'tanh': 'Tanh',
        'silu': 'SiLU',
        'swish': 'SiLU',
        'linear': 'Identity'
    }
    if activation is None:
        return nn.Identity
    activation = activation.lower()
    if activation in _torch_activations_dict:
        return getattr(nn, _torch_activations_dict[activation])
    raise ValueError(f"Activation '{activation}' not valid.")
