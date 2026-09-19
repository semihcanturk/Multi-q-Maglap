import torch
from torch import Tensor
from torch.nn import Linear, Sequential, Module, ReLU, Dropout
from torch_geometric.nn import MessagePassing


class _DirEdgeGatedAggregator(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, aggr="mean", scalar_gate=True):
        super().__init__(aggr=aggr)
        self.lin_msg = Linear(in_channels, out_channels, bias=False)
        self.lin_gate = Linear(edge_dim, 1 if scalar_gate else out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg = self.lin_msg(x_j)
        gate = torch.sigmoid(self.lin_gate(edge_attr))
        return gate * msg


class  DirEdgeGatedEncoder(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        aggr="mean",
        dropout=0.0,
        residual=True,
        scalar_gate=True,
    ):
        super().__init__()
        self.residual = residual and (in_channels == out_channels)

        self.in_agg = _DirEdgeGatedAggregator(
            in_channels, out_channels, edge_dim, aggr=aggr, scalar_gate=scalar_gate
        )
        self.out_agg = _DirEdgeGatedAggregator(
            in_channels, out_channels, edge_dim, aggr=aggr, scalar_gate=scalar_gate
        )

        self.combine = Sequential(
            Linear(in_channels + 2 * out_channels, out_channels),
            ReLU(),
            Dropout(dropout),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        m_in = self.in_agg(x, edge_index, edge_attr)
        m_out = self.out_agg(x, rev_edge_index, edge_attr)

        out = self.combine(torch.cat([x, m_in, m_out], dim=-1))

        if self.residual:
            out = out + x

        return out
