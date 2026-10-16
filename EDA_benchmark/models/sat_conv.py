"""One layer of the Structure-Aware Transformer (SAT, Chen et al., ICML 2022).

Port of the k-subtree SAT layer in the repo-root ``model/sat/layers.py``
(``TransformerEncoderLayer`` with ``se='gnn'``, ``batch_norm=True``), following the
dense-batch attention path that implementation takes whenever ``batch`` is given:

    x_struct = StructureExtractor(x)                      # k GNN hops, concat, BN, linear
    x = BN(x + Dropout(MHA(query=x_struct, key=x_struct, value=x)))
    x = BN(x + Dropout(FFN(x)))

Differences from the root copy:

* the structure extractor's GNN can be bidirected (separate weights for the forward and
  the reverse edge direction, averaged -- the BIGIN/BIGINE scheme of base_model.py),
  which the root ``get_simple_gnn_layer`` has no option for;
* there are no per-layer PE encoders: as for every other EDA backbone, MiddleModel
  concatenates node PEs to ``x`` and adds edge PEs to ``edge_attr`` before the first layer.

It is a port rather than an import because the root module imports ``utils.activation``
and ``utils.handle_complex``, which EDA_benchmark's own top-level ``utils.py`` shadows (see
maglap/get_path_lap.py).
"""
import torch
from torch import nn
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.utils import to_dense_batch

from models.gps_conv import MultiheadAttention

# structure-extractor kernels: *E variants consume edge_attr, BI* variants are bidirected
INNER_GNNS = ('GIN', 'GINE', 'BIGIN', 'BIGINE')


def gin_layer(embed_dim, use_edge_attr):
    # same kernel as 'gin' / 'gine' in the root model/sat/gnn_layers.py
    mlp = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(True),
        nn.Linear(embed_dim, embed_dim),
    )
    return GINEConv(mlp, train_eps=True) if use_edge_attr else GINConv(mlp, train_eps=True)


class StructureExtractor(nn.Module):
    """k-subtree structure extractor: ``num_layers`` GNN hops, the input and every hop's
    output concatenated, then batch-normed and projected back to ``embed_dim``."""

    def __init__(self, embed_dim, inner_gnn, num_layers):
        super().__init__()
        assert inner_gnn in INNER_GNNS, f'SAT inner_gnn must be one of {INNER_GNNS}, got {inner_gnn}'
        self.use_edge_attr = inner_gnn.endswith('E')
        self.bidirected = inner_gnn.startswith('BI')
        self.num_layers = num_layers
        self.convs_forward = nn.ModuleList([gin_layer(embed_dim, self.use_edge_attr) for _ in range(num_layers)])
        if self.bidirected:
            self.convs_backward = nn.ModuleList([gin_layer(embed_dim, self.use_edge_attr) for _ in range(num_layers)])
        self.relu = nn.ReLU()
        inner_dim = (num_layers + 1) * embed_dim
        self.bn = nn.BatchNorm1d(inner_dim)
        self.out_proj = nn.Linear(inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if self.use_edge_attr and edge_attr is None:
            # AMP has no raw edge features: without an edge-level PE use inner_gnn GIN/BIGIN
            raise ValueError('SAT inner_gnn GINE/BIGINE requires edge_attr')
        edge_args = (edge_attr,) if self.use_edge_attr else ()
        reverse_edge_index = edge_index[[1, 0], :]
        x_cat = [x]
        for i in range(self.num_layers):
            h = self.convs_forward[i](x, edge_index, *edge_args)
            if self.bidirected:
                h = (h + self.convs_backward[i](x, reverse_edge_index, *edge_args)) / 2
            x = self.relu(h)
            x_cat.append(x)
        x = torch.cat(x_cat, dim=-1)
        if self.num_layers > 0:
            x = self.bn(x)
        return self.out_proj(x)


class SATConv(nn.Module):
    def __init__(self, d_model, inner_gnn, k_hop, num_heads, dropout):
        super().__init__()
        # SAT's own training scripts use dim_feedforward = 2 * d_model
        dim_feedforward = 2 * d_model
        self.structure_extractor = StructureExtractor(d_model, inner_gnn, k_hop)
        self.attn = MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # queries/keys come from the structure-aware embeddings, values from x itself
        x_struct = self.structure_extractor(x, edge_index, edge_attr)
        x_struct, mask = to_dense_batch(x_struct, batch)
        value, _ = to_dense_batch(x, batch)
        # need_weights=True: this MultiheadAttention copy only implements that branch
        h, _ = self.attn(x_struct, x_struct, value, key_padding_mask=~mask, need_weights=True)
        x = self.norm1(x + self.dropout1(h[mask]))
        h = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout2(h))
