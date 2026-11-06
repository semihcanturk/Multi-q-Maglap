"""PyG transform for the raw pairwise directed random walk encoding."""
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def dense_random_walk_matrix(edge_index: Tensor, num_nodes: int,
                             reverse: bool = False) -> Tensor:
    if reverse:
        edge_index = edge_index.flip(0)
    source, target = edge_index
    P = torch.zeros((num_nodes, num_nodes), dtype=torch.float32,
                    device=edge_index.device)
    degree = torch.bincount(source, minlength=num_nodes).to(P.dtype)
    if source.numel():
        values = degree[source].reciprocal()
        P.index_put_((source, target), values, accumulate=True)
    dangling = (degree == 0).nonzero(as_tuple=True)[0]
    P[dangling, dangling] = 1.0
    return P


def k_step_random_walk(P: Tensor, walk_length: int,
                       ppr_restart_p: Optional[float] = 0.2) -> Tensor:
    """Return [N, N, k] or [N, N, k+1] with PPR in the last channel."""
    if walk_length < 1:
        raise ValueError('walk_length must be >= 1')
    powers = []
    current = P
    for _ in range(walk_length):
        powers.append(current)
        current = current @ P
    if ppr_restart_p is not None:
        alpha = float(ppr_restart_p)
        if not 0 < alpha <= 1:
            raise ValueError('ppr_restart_p must lie in (0, 1]')
        I = torch.eye(P.size(0), dtype=P.dtype, device=P.device)
        ppr = alpha * torch.linalg.solve(I - (1.0 - alpha) * P, I)
        powers.append(ppr)
    return torch.stack(powers, dim=-1)


def directed_random_walk_pe(edge_index: Tensor, num_nodes: int,
                            walk_length: int = 3, reverse: bool = True,
                            ppr_restart_p: Optional[float] = 0.2) -> Tensor:
    """Return pairwise [N, N, d] RWPE, forward channels followed by reverse."""
    P = dense_random_walk_matrix(edge_index, num_nodes)
    channels = [k_step_random_walk(P, walk_length, ppr_restart_p)]
    if reverse:
        P_reverse = dense_random_walk_matrix(edge_index, num_nodes, reverse=True)
        channels.append(k_step_random_walk(P_reverse, walk_length, ppr_restart_p))
    return torch.cat(channels, dim=-1)


def dir_rwpe_dim(walk_length: int, reverse: bool = True,
                 ppr_restart_p: Optional[float] = 0.2) -> int:
    """Channels d of directed_random_walk_pe: k walk steps (+1 PPR) per direction."""
    return (walk_length + (ppr_restart_p is not None)) * (2 if reverse else 1)


class AddDirectedRandomWalkPE(BaseTransform):
    """Store raw relative PE, intended for learned absolute aggregation later.

    For batching, [N,N,d] is flattened to [N*N,d] in row-major order.
    A PyG Batch concatenates these rows; the encoder reconstructs each graph
    using batch.ptr (node offsets).  Do not use attr_name=None here: it is
    not yet a node-level PE.
    """
    def __init__(self, walk_length: int = 3, reverse: bool = True,
                 ppr_restart_p: Optional[float] = 0.2,
                 attr_name: str = 'dir_rw_pe') -> None:
        self.walk_length = walk_length
        self.reverse = reverse
        self.ppr_restart_p = ppr_restart_p
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        if data.edge_index is None or data.num_nodes is None:
            raise ValueError('data requires edge_index and num_nodes')
        with torch.no_grad():
            pe = directed_random_walk_pe(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                walk_length=self.walk_length,
                reverse=self.reverse,
                ppr_restart_p=self.ppr_restart_p,
            )
        data[self.attr_name] = pe.reshape(-1, pe.size(-1))
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(walk_length={self.walk_length}, '
                f'reverse={self.reverse}, ppr_restart_p={self.ppr_restart_p}, '
                f'attr_name={self.attr_name!r})')
