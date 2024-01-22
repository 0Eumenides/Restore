import torch
from typing import Optional, Tuple, Union
from torch import nn
from torch.nn import LayerNorm, functional as F
from torch import Tensor
from model.Layers import MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter, gather_csr, segment_csr
from torch_scatter.utils import broadcast
from torch_geometric.utils.num_nodes import maybe_num_nodes


class AdditiveAttention(MessagePassing):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 norm: bool = True,
                 dropout: float = 0.0,
                 dim: int = -2,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=dim, **kwargs)

        self.output_size = output_size
        if isinstance(input_size, int):
            self.src_size = self.tgt_size = input_size
        else:
            self.src_size, self.tgt_size = input_size

        self.msg_size = msg_size or self.output_size
        self.msg_layers = msg_layers

        assert reweight in ['softmax', 'l1', None]
        self.reweight = reweight

        self.root_weight = root_weight
        self.dropout = dropout

        # key bias is discarded in softmax
        self.lin_src = Linear(self.src_size, self.output_size,
                              weight_initializer='glorot',
                              bias_initializer='zeros')
        self.lin_tgt = Linear(self.tgt_size, self.output_size,
                              weight_initializer='glorot', bias=False)

        if self.root_weight:
            self.lin_skip = Linear(self.tgt_size, self.output_size,
                                   bias=False)
        else:
            self.register_parameter('lin_skip', None)

        self.msg_nn = nn.Sequential(
            nn.PReLU(init=0.2),
            MLP(self.output_size, self.msg_size, self.output_size,
                n_layers=self.msg_layers, dropout=self.dropout,
                activation='prelu')
        )

        if self.reweight == 'softmax':
            self.msg_gate = nn.Linear(self.output_size, 1, bias=False)
        else:
            self.msg_gate = nn.Sequential(nn.Linear(self.output_size, 1),
                                          nn.Sigmoid())

        if norm:
            self.norm = LayerNorm(self.output_size)
        else:
            self.register_parameter('norm', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_tgt.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()

    def forward(self, x: PairTensor, edge_index: Adj, mask: OptTensor = None):
        # if query/key not provided, defaults to x (e.g., for self-attention)
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        N_src, N_tgt = x_src.size(self.node_dim), x_tgt.size(self.node_dim)

        msg_src = self.lin_src(x_src)
        msg_tgt = self.lin_tgt(x_tgt)

        msg = (msg_src, msg_tgt)

        # propagate_type: (msg: PairTensor, mask: OptTensor)
        out = self.propagate(edge_index, msg=msg, mask=mask,
                             size=(N_src, N_tgt))

        # skip connection
        if self.root_weight:
            out = out + self.lin_skip(x_tgt)

        if self.norm is not None:
            out = self.norm(out)

        return out

    def normalize_weights(self, weights, index, num_nodes, mask=None):
        # mask weights
        if mask is not None:
            fill_value = float("-inf") if self.reweight == 'softmax' else 0.
            weights = weights.masked_fill(torch.logical_not(mask), fill_value)
        # eventually reweight
        if self.reweight == 'l1':
            expanded_index = broadcast(index, weights, self.node_dim)
            weights_sum = scatter(weights, expanded_index, self.node_dim,
                                  dim_size=num_nodes, reduce='sum')
            weights_sum = weights_sum.index_select(self.node_dim, index)
            weights = weights / (weights_sum + 1e-5)
        elif self.reweight == 'softmax':
            weights = sparse_softmax(weights, index, num_nodes=num_nodes,
                                     dim=self.node_dim)
        return weights

    def message(self, msg_j: Tensor, msg_i: Tensor, index, size_i,
                mask_j: OptTensor = None) -> Tensor:
        msg = self.msg_nn(msg_j + msg_i)
        gate = self.msg_gate(msg)
        alpha = self.normalize_weights(gate, index, size_i, mask_j)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = alpha * msg
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.output_size}, '
                f'dim={self.node_dim}, '
                f'root_weight={self.root_weight})')


class TemporalAdditiveAttention(AdditiveAttention):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 norm: bool = True,
                 dropout: float = 0.0,
                 **kwargs):
        kwargs.setdefault('dim', 1)
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         msg_size=msg_size,
                         msg_layers=msg_layers,
                         root_weight=root_weight,
                         reweight=reweight,
                         dropout=dropout,
                         norm=norm,
                         **kwargs)

    def forward(self, x: PairTensor, mask: OptTensor = None,
                temporal_mask: OptTensor = None,
                causal_lag: Optional[int] = None):
        # x: [b s * c]    query: [b l * c]    key: [b s * c]
        # mask: [b s * c]    temporal_mask: [l s]
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        l, s = x_tgt.size(self.node_dim), x_src.size(self.node_dim)
        i = torch.arange(l, dtype=torch.long, device=x_src.device)
        j = torch.arange(s, dtype=torch.long, device=x_src.device)

        # compute temporal index, from j to i
        if temporal_mask is None and isinstance(causal_lag, int):
            temporal_mask = tuple(torch.tril_indices(l, l, offset=-causal_lag,
                                                     device=x_src.device))
        if temporal_mask is not None:
            assert temporal_mask.size() == (l, s)
            i, j = torch.meshgrid(i, j)
            edge_index = torch.stack((j[temporal_mask], i[temporal_mask]))
        else:
            edge_index = torch.cartesian_prod(j, i).T

        return super(TemporalAdditiveAttention, self).forward(x, edge_index,
                                                              mask=mask)


def sparse_softmax(src: Tensor,
                   index: Optional[Tensor] = None,
                   ptr: Optional[Tensor] = None,
                   num_nodes: Optional[int] = None,
                   dim: int = -2) -> Tensor:
    r"""Extension of :func:`~torch_geometric.softmax` with index broadcasting
    to compute a sparsely evaluated softmax over multiple broadcast dimensions.

    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (Tensor, optional): The indices of elements for applying the
            softmax.
            (default: :obj:`None`)
        ptr (Tensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, i.e.,
            :obj:`max_val + 1` of :attr:`index`.
            (default: :obj:`None`)
        dim (int): The dimension on which to normalize, i.e., the edge
            dimension.
            (default: :obj:`-2`)
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        expanded_index = broadcast(index, src, dim)
        src_max = scatter(src, expanded_index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, expanded_index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + 5e-8)
