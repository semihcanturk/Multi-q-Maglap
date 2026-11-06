"""Absolute directed RWPE encoder, with concatenation to node attributes.

Adjust the two imports below if your existing encoder modules live elsewhere.
"""
from typing import Optional

import torch
from torch import nn


class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, *args, **kwargs):
        return self.encoder(x)


class BatchNorm1dNode(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(channels, eps=1e-5, momentum=0.1)

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(channels, eps=1e-5, momentum=0.1)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


class FeatureEncoder(torch.nn.Module):
    r"""Encodes node and edge features, given the specified input dimension and
    the underlying configuration in :obj:`cfg`.

    Args:
        dim_in (int): The input feature dimension.
    """
    def __init__(self, dim_in: int, dim_hid: int,
                 node_encoder: Optional[torch.nn.Module] = None, node_encoder_bn: bool = False,
                 edge_encoder: Optional[torch.nn.Module] = None, edge_encoder_bn: bool = False):
        super().__init__()
        self.dim_in = dim_in
        if node_encoder:
            # Encode integer node features via `torch.nn.Embedding`:
            self.node_encoder = node_encoder
            if node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(dim_hid)
            # Update `dim_in` to reflect the new dimension fo the node features
            self.dim_in = dim_hid
        if edge_encoder:
            # Encode integer edge features via `torch.nn.Embedding`:
            self.edge_encoder = edge_encoder
            if edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(dim_hid)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class DirRWPENodeEncoder(nn.Module):
    """Pairwise Linear -> ReLU -> source-sum -> Linear -> ReLU -> concat.

    Input data.dir_rw_pe: [sum_g N_g**2, d_rw], row-major for each graph.
    Input data.x: [sum_g N_g, dim_node].
    """
    def __init__(self, dim_pe: int, dim_emb: int,
                 attr_name: str = 'dir_rw_pe') -> None:
        super().__init__()
        self.linear = LinearNodeEncoder(dim_pe, dim_pe)
        self.activation = nn.ReLU()
        self.mlp = LinearNodeEncoder(dim_pe, dim_emb)
        self.attr_name = attr_name

    def forward(self, batch):
        raw = getattr(batch, self.attr_name, None)
        if raw is None:
            raise ValueError(f'Missing pairwise positional encoding: {self.attr_name}')
        ptr = getattr(batch, 'ptr', None)
        if ptr is None:
            sizes = [batch.num_nodes]
        else:
            sizes = (ptr[1:] - ptr[:-1]).tolist()
        embeddings = []
        cursor = 0
        for n in sizes:
            n = int(n)
            pair_pe = raw[cursor:cursor + n * n].reshape(n, n, -1)
            cursor += n * n
            # R[u,v,:]; sum over sources u, retain one vector per target v.
            encoded = self.activation(self.linear(pair_pe)).sum(dim=0)
            embeddings.append(self.activation(self.mlp(encoded)))
        if cursor != raw.size(0):
            raise ValueError(f'RWPE pair count mismatch: consumed {cursor}, '
                             f'found {raw.size(0)}')
        node_pe = torch.cat(embeddings, dim=0)
        batch.x = torch.cat([batch.x, node_pe], dim=-1)
        return batch


class DirRWFeatureEncoder(FeatureEncoder):
    """Fit the existing GNN.encoder interface (reports output dim_in)."""
    def __init__(self, dim_in: int, dim_emb: int, dim_pe: int,
                 dim_hid: int = 128, attr_name: str = 'dir_rw_pe') -> None:
        super().__init__(dim_in=dim_in + dim_emb, dim_hid=dim_hid)
        self.dir_rw_encoder = DirRWPENodeEncoder(
            dim_pe=dim_pe, dim_emb=dim_emb, attr_name=attr_name)
