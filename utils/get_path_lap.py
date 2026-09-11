# from html.parser import endtagfind
from typing import Any, Optional

import networkx
from torch import Tensor

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.transforms import BaseTransform, ToUndirected
from torch_geometric.utils import to_networkx
#from pathhomology.pathComplex import BasicPathComplex
from scipy.sparse import csr_array, issparse, csr_matrix, issparse
from typing import Optional

from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from pathhomology.PathCpx import PathComplex

import logging

log = logging.getLogger(__name__)


CUSOLVER_EIGH_MAX_N = 26733


def solve_smallest_eigs(
    L,
    k: int,
    method: str = 'auto',
    sigma: float = -1e-5,
    dtype: str = 'float64',
    gpu_dense_max_nodes: int = CUSOLVER_EIGH_MAX_N,
    **eigsh_kwargs: Any,
):
    r"""Smallest-``k`` eigenpairs of a real-symmetric / complex-Hermitian sparse ``L``.

    Shared fast solver behind the ``Fast`` PE transforms. Replaces the slow
    regular-mode ``eigsh(which='SA'/'SM')`` and full dense ``np.linalg.eigh``
    paths with:

    - ``"shift_invert"``: ``eigsh(L, sigma, which='LM')`` (real sym & complex
      Hermitian) — one factorization, then fast convergence near ``sigma``.
    - ``"gpu_dense"``: ``torch.linalg.eigh`` on GPU (real or complex Hermitian).
    - ``"dense"``: ``scipy.linalg.eigh(subset_by_index=[0, k-1])`` on CPU.
    - ``"auto"``: ``gpu_dense`` if CUDA is available and ``n <=
      gpu_dense_max_nodes`` (the cuSOLVER ceiling), else ``shift_invert``.

    Note that ``auto`` decides on ``n`` alone, which is a poor predictor of which
    backend is faster. The real predictor is separability (represented by sparse-LU fill-in):
    ``shift_invert`` is preferable on graphs with good separators (roman-empire, a 22k chain,
    has 1.7x fill-in: 0.6 s vs 54 s for ``gpu_dense``) and loses on expander-like graphs
    where fill-in explodes (random 3-regular n=20000, 738x fill-in: 93 s vs 44 s).
    Prefer ``shift_invert`` unless your graphs are genuinely expander-like *and* below
    the cuSOLVER ceiling.

    Returns ``(eig_vals, eig_vecs, k_eff)``: real ascending ``eig_vals`` and
    ``eig_vecs`` in the input's native dtype (complex for a Hermitian input — the
    caller takes the real part where appropriate). ``k_eff`` may be ``< k`` for
    tiny matrices; callers pad as needed.
    """
    n = L.shape[0]

    resolved = method
    if method == 'auto':
        resolved = 'shift_invert'
        try:
            if torch.cuda.is_available() and n <= gpu_dense_max_nodes:
                resolved = 'gpu_dense'
        except Exception:
            resolved = 'shift_invert'

    def _dense():
        from scipy.linalg import eigh
        A = np.asarray(L.todense())
        if not np.iscomplexobj(A):
            A = A.astype(dtype, copy=False)
        kk = min(k, n)
        return eigh(A, subset_by_index=[0, kk - 1])

    def _gpu_dense():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        A_np = np.asarray(L.todense())
        if np.iscomplexobj(A_np):
            tdtype = torch.complex128 if dtype == 'float64' else torch.complex64
        else:
            tdtype = torch.float64 if dtype == 'float64' else torch.float32
        A = torch.from_numpy(A_np).to(device=device, dtype=tdtype)
        evals, evecs = torch.linalg.eigh(A)  # ascending, real eigenvalues
        kk = min(k, n)
        return evals[:kk].cpu().numpy(), evecs[:, :kk].cpu().numpy()

    def _shift_invert():
        from scipy.sparse.linalg import eigsh
        kk = min(k, n - 1)
        if kk < 1:
            raise ValueError("matrix too small for shift-invert eigsh")
        return eigsh(L, k=kk, sigma=sigma, which='LM',
                     return_eigenvectors=True, **eigsh_kwargs)

    if resolved not in ('dense', 'gpu_dense', 'shift_invert'):
        raise ValueError(
            f"Unknown method '{resolved}'; expected 'auto', 'shift_invert', "
            "'gpu_dense', or 'dense'."
        )

    if resolved == 'gpu_dense':
        try:
            vals, vecs = _gpu_dense()
        except Exception as e:
            # cuSOLVER rejects the syevd workspace query for large n on some
            # GPUs (CUSOLVER_STATUS_INVALID_VALUE), and the dense n x n copy can
            # OOM. Never let this escape: callers that treat an exception as
            # "this graph failed" would silently store a dummy PE.
            log.warning("gpu_dense eigh failed (%s); falling back to shift_invert.", e)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            resolved = 'shift_invert'

    if resolved == 'shift_invert':
        try:
            vals, vecs = _shift_invert()
        except Exception as e:  # complex-unsupported / no-convergence / singular factor
            log.warning("shift_invert eigsh failed (%s); falling back to dense.", e)
            vals, vecs = _dense()
    elif resolved == 'dense':
        vals, vecs = _dense()

    vals = np.real(np.asarray(vals))
    order = np.argsort(vals)
    return vals[order], np.asarray(vecs)[:, order], len(vals)



def _eigsh_partial(L, k, **kwargs):
    """Wrapper around eigsh that returns partial results on convergence failure.

    When ARPACK fails to converge, the exception carries the converged
    eigenvalues/eigenvectors. We extract them and let the caller's
    existing zero-padding logic handle the shortfall.

    Returns ``(eig_vals, eig_vecs, k_converged)`` where ``k_converged``
    is the number of actually converged eigenpairs (== ``k`` on success).
    """
    # Generate v0 from numpy's global RNG (seeded by L.seed_everything)
    # so that ARPACK convergence is deterministic and controlled by the
    # config seed, rather than ARPACK's unseeded internal Fortran RNG.
    if 'v0' not in kwargs:
        kwargs['v0'] = np.random.random(L.shape[0])
    try:
        eig_vals, eig_vecs = eigsh(L, k=k, **kwargs)
        return eig_vals, eig_vecs, k
    except ArpackNoConvergence as e:
        eig_vals = e.eigenvalues
        eig_vecs = e.eigenvectors
        k_conv = len(eig_vals)
        log.warning(
            "ARPACK did not converge: %d/%d eigenvectors converged. "
            "Using partial result.",
            k_conv, k,
        )
        return eig_vals, eig_vecs, k_conv


def add_node_attr(
        data: Data,
        value: Any,
        attr_name: Optional[str] = None,
) -> Data:
    # Move to `BaseTransform`.
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


def add_edge_attr(data: Data, edge_att: torch.Tensor, as_weight=False) -> Data:
    if as_weight:
        assert edge_att.ndim == 1
        data.edge_weight = edge_att
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_att = edge_att.unsqueeze(-1) if edge_att.ndim == 1 else edge_att
        edge_att = torch.cat([data.edge_attr, edge_att], dim=-1)
        data.edge_attr = edge_att
    elif hasattr(data, 'edge_weight') and data.edge_weight is not None:
        edge_att = edge_att.unsqueeze(-1) if edge_att.ndim == 1 else edge_att
        edge_att = torch.cat([data.edge_weight.unsqueeze(-1), edge_att], dim=-1)
        data.edge_attr = edge_att
    else:
        edge_att = edge_att.unsqueeze(-1) if edge_att.ndim == 1 else edge_att
        data.edge_attr = edge_att
    return data


def _reindex_pe_edge(pe_edge, edge_index, num_nodes):
    """Map pe_edge from PathComplex's deduplicated lex order to edge_index column order.

    PathComplex produces one PE row per *unique* directed edge in lexicographic
    (source, target) order, excluding self-loops (which PathComplex strips before
    building its path list). But ``edge_index`` may have duplicate (src, dst)
    pairs and self-loops. This function builds a lookup so that:
      - each edge in edge_index gets the PE of its unique (src, dst) pair
      - duplicate edges get identical PE copies
      - self-loops and edges absent from the path complex get zero PE
    """
    n = num_nodes
    E_unique = pe_edge.shape[0]
    E_total = edge_index.size(1)
    k = pe_edge.shape[1]

    keys_all = edge_index[0] * n + edge_index[1]
    non_self_loop = edge_index[0] != edge_index[1]
    unique_keys, _ = torch.sort(torch.unique(keys_all[non_self_loop]))

    assert unique_keys.shape[0] == E_unique, (
        f"pe_edge has {E_unique} rows but edge_index has "
        f"{unique_keys.shape[0]} unique non-self-loop edges"
    )

    key_to_idx = torch.full((n * n,), -1, dtype=torch.long)
    key_to_idx[unique_keys] = torch.arange(E_unique, dtype=torch.long)

    pe_indices = key_to_idx[keys_all]
    mask = pe_indices >= 0

    mapped = torch.zeros(E_total, k, dtype=pe_edge.dtype)
    mapped[mask] = pe_edge[pe_indices[mask]]
    return mapped


def build_two_path_index(edge_index: Tensor, num_nodes: int, remove_self_loops: bool = True) -> Tensor:
    """Return all directed 2-paths of a (batched) graph as pairs of edge ids.

    A pair ``(e, f)`` is emitted whenever ``head(e) == tail(f)``, i.e. the edges
    ``e = (u, v)`` and ``f = (v, w)`` form the 2-path ``u -> v -> w`` (``w == u`` is
    allowed). Works on PyG batches directly because node ids are unique across the
    batch. Self-loops carry no path-complex PE and are skipped by default.

    Args:
        edge_index: ``[2, E]`` edge index.
        num_nodes: Number of nodes (of the whole batch).
        remove_self_loops: Exclude self-loop edges from both roles.
    Returns:
        ``[2, P]`` long tensor of edge-id pairs ``(e, f)``, grouped by ``e``.
    """
    src, dst = edge_index[0], edge_index[1]
    device = src.device
    if remove_self_loops:
        valid = torch.nonzero(src != dst, as_tuple=True)[0]
    else:
        valid = torch.arange(src.numel(), device=device)

    # Successor candidates: valid edges grouped (CSR-style) by their tail node.
    out_src = src[valid]
    perm = torch.argsort(out_src, stable=True)
    out_ids = valid[perm]
    out_deg = torch.bincount(out_src, minlength=num_nodes)
    out_ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    out_ptr[1:] = torch.cumsum(out_deg, dim=0)

    # Every valid edge e pairs with all out-edges of its head node dst[e].
    heads = dst[valid]
    counts = out_deg[heads]
    e_rep = torch.repeat_interleave(valid, counts)
    starts = torch.repeat_interleave(out_ptr[heads], counts)
    group_start = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
    within = torch.arange(e_rep.numel(), device=device) - group_start
    f = out_ids[starts + within]
    return torch.stack([e_rep, f], dim=0)


"""
Transform that adds unit edge features to Data object
"""
class AddConstantEdgeFeatures(BaseTransform):
    def forward(self, data):
        data.edge_attr = torch.ones(
            (data.num_edges, 1),
            dtype=torch.float,
        )
        return data


"""
Transform takes Data object with node features and returns Data where edges are oriented using sign(diff(node1-node2)))
"""


class AddPathLaplacianEigenvectorPE4(BaseTransform):
    """
    Minimal PE writer for the common pmax=2 case.

    Assumes p=0 and p=1 are ALWAYS requested and present.

    It writes:
      - p=0 eigenvectors to node attributes
      - p=1 eigenvectors to edge attributes

    Supports different numbers of eigenvalues/eigenvectors for nodes and edges.

    The per-Laplacian eigensolve uses the shared fast backend
    ``solve_smallest_eigs``. The ``which`` argument is retained for API
    compatibility but is not used by the fast shift-invert solver.
    """

    def __init__(
        self,
        k: int,
        k_edge: Optional[int] = None,
        tol: float = 1e-8,
        which: str = "SM",
        node_attr_name: str = "node_path_eigen_pe3",
        edge_attr_name: str = "edge_path_eigen_pe3",
        normalize: bool = True,
        eigsh_tol: float = 1e-6,
        method: str = "auto",
        sigma: float = -1e-5,
        dtype: str = "float64",
        gpu_dense_max_nodes: int = CUSOLVER_EIGH_MAX_N,
        **kwargs
    ):
        self.k = int(k)
        self.k_edge = self.k if k_edge is None else int(k_edge)
        self.tol = float(tol)
        self.which = which
        self.node_attr_name = node_attr_name
        self.edge_attr_name = edge_attr_name
        self.normalize = bool(normalize)
        self.eigsh_tol = float(eigsh_tol)

        self.method = method
        self.sigma = sigma
        self.dtype = dtype
        self.gpu_dense_max_nodes = gpu_dense_max_nodes

        self.kwargs = kwargs

    def forward(self, data: Data) -> Data:
        # Always compute p=0 and p=1.
        p_list = [0, 1]

        # Computing L_1 requires paths up to dimension 2.
        G = to_networkx(data, to_undirected=False)
        pmax = 2
        cpx = PathComplex(G, pmax=pmax)

        if self.normalize:
            Lap0, Lap1 = cpx.get_normalized_hodge_laplacian(
                p_list,
                tol=1e-12,
                fmt="csr",
            )
        else:
            Lap0, Lap1 = cpx.get_hodge_laplacian(
                p_list,
                tol=1e-12,
                fmt="csr",
            )

        # Helper to compute and pad the requested number of eigenvectors.
        def k_eigs(L, k_target):
            L = L.tocsr() if issparse(L) else csr_matrix(L)
            n = L.shape[0]

            if n == 0:
                vals = np.zeros((0,), dtype=float)
                vecs = np.zeros((0, 0), dtype=float)
                k_eff = 0

            elif n == 1:
                vals = np.array([L[0, 0]], dtype=float)
                vecs = np.array([[1.0]], dtype=float)
                k_eff = 1

            else:
                # Sparse eigensolvers require k < n.
                k_eff = min(k_target, n - 1)

                # Give ARPACK a larger Krylov subspace.
                kwargs = dict(self.kwargs)

                if "ncv" not in kwargs:
                    kwargs["ncv"] = min(
                        n,
                        max(4 * k_eff + 1, 20),
                    )

                vals, vecs, k_eff = solve_smallest_eigs(
                    L,
                    k_eff,
                    method=self.method,
                    sigma=self.sigma,
                    dtype=self.dtype,
                    gpu_dense_max_nodes=self.gpu_dense_max_nodes,
                    tol=self.eigsh_tol,
                    **kwargs,
                )

                # Hodge Laplacians are real symmetric.
                vals = np.real(vals)
                vecs = np.real(vecs)

                # Sort eigenpairs consistently by increasing eigenvalue.
                idx = np.argsort(vals)
                vals = vals[idx]
                vecs = vecs[:, idx]

            # Pad small Laplacians to the requested PE dimension.
            if k_eff < k_target:
                vals = np.pad(
                    vals,
                    (0, k_target - k_eff),
                )

                vecs = np.pad(
                    vecs,
                    ((0, 0), (0, k_target - k_eff)),
                )

            # Remove very small numerical values.
            if self.tol is not None and self.tol > 0:
                vals[np.abs(vals) < self.tol] = 0.0
                vecs[np.abs(vecs) < self.tol] = 0.0

            return vals, vecs

        # p=0: node spectrum
        lam0, vec0 = k_eigs(
            Lap0,
            self.k,
        )

        # p=1: edge spectrum
        lam1, vec1 = k_eigs(
            Lap1,
            self.k_edge,
        )

        # Keep numerical eigensolves in the requested precision, but always
        # provide float32 tensors to the neural-network pipeline.
        pe0 = torch.from_numpy(
            vec0
        ).to(dtype=torch.float32)

        pe1 = torch.from_numpy(
            vec1
        ).to(dtype=torch.float32)

        # p=0 -> node PE
        data = add_node_attr(
            data,
            pe0,
            attr_name=self.node_attr_name,
        )

        # p=1 -> edge PE
        if self.edge_attr_name:
            data[self.edge_attr_name] = pe1
        else:
            data = add_edge_attr(
                data,
                pe1,
            )

        # Store node and edge eigenvalues separately since k and k_edge
        # may be different.
        data["Lambda_node"] = (
            torch.from_numpy(lam0)
            .to(dtype=torch.float32)
            .unsqueeze(0)
        )

        data["Lambda_edge"] = (
            torch.from_numpy(lam1)
            .to(dtype=torch.float32)
            .unsqueeze(0)
        )

        return data


@functional_transform("add_laplacian_eigenvector_pe3_PathCpx")
class AddPathLaplacianEigenvectorPE3(BaseTransform):
    """Path-Hodge-Laplacian eigenvector PE (default, fast implementation).

    Writes p=0 (node) and p=1 (edge) path-Hodge-Laplacian eigenvector PE for the
    common pmax=2 case. The per-Laplacian eigensolve uses the shared fast backend
    (:func:`src.transforms.laplacian.solve_smallest_eigs`) rather than the slow
    regular-mode ``eigsh(which='SM')``. The ``which`` argument is retained for API
    compatibility but ignored (shift-invert targets the smallest eigenvalues
    directly). For the original reference implementation see
    :class:`AddLegacyPathLaplacianEigenvectorPE3`.
    """

    def __init__(
            self,
            k: int,
            tol: float = 1e-8,
            which: str = "SM",
            node_attr_name: str = "node_path_eigen_pe3",
            edge_attr_name: str = "edge_path_eigen_pe3",
            normalize: bool = True,
            eigsh_tol: float = 1e-6,
            method: str = "auto",
            sigma: float = -1e-5,
            dtype: str = "float64",
            gpu_dense_max_nodes: int = CUSOLVER_EIGH_MAX_N,
            **kwargs
    ):
        self.k = int(k)
        self.tol = float(tol)
        self.which = which
        self.node_attr_name = node_attr_name
        self.edge_attr_name = edge_attr_name
        self.normalize = bool(normalize)
        self.eigsh_tol = float(eigsh_tol)
        self.method = method
        self.sigma = sigma
        self.dtype = dtype
        self.gpu_dense_max_nodes = gpu_dense_max_nodes
        self.kwargs = kwargs

    def forward(self, data: Data) -> Data:
        # Always compute p=0 and p=1.
        p_list = [0, 1]

        # Computing L_1 requires paths up to dimension 2.
        G = to_networkx(data, to_undirected=False)
        pmax = 2
        cpx = PathComplex(G, pmax=pmax)

        if self.normalize:
            Lap0, Lap1 = cpx.get_normalized_hodge_laplacian(
                p_list,
                tol=1e-12,
                fmt="csr",
            )
        else:
            Lap0, Lap1 = cpx.get_hodge_laplacian(
                p_list,
                tol=1e-12,
                fmt="csr",
            )

        # Helper to compute k eigenvectors and pad when necessary.
        def k_eigs(L):
            L = L.tocsr() if issparse(L) else csr_matrix(L)
            n = L.shape[0]

            if n == 0:
                vals = np.zeros((0,), dtype=float)
                vecs = np.zeros((0, 0), dtype=float)
                k_eff = 0

            elif n == 1:
                vals = np.array([L[0, 0]], dtype=float)
                vecs = np.array([[1.0]], dtype=float)
                k_eff = 1

            else:
                # Sparse eigensolvers require k < n.
                k_eff = min(self.k, n - 1)

                # Give ARPACK a larger Krylov subspace.
                kwargs = dict(self.kwargs)

                if "ncv" not in kwargs:
                    kwargs["ncv"] = min(
                        n,
                        max(4 * k_eff + 1, 20),
                    )

                vals, vecs, k_eff = solve_smallest_eigs(
                    L,
                    k_eff,
                    method=self.method,
                    sigma=self.sigma,
                    dtype=self.dtype,
                    gpu_dense_max_nodes=self.gpu_dense_max_nodes,
                    tol=self.eigsh_tol,
                    **kwargs,
                )

                # Hodge Laplacians are real symmetric.
                vals = np.real(vals)
                vecs = np.real(vecs)

                # Sort eigenpairs consistently by increasing eigenvalue.
                idx = np.argsort(vals)
                vals = vals[idx]
                vecs = vecs[:, idx]

            # Pad small Laplacians to the requested PE dimension.
            if k_eff < self.k:
                vals = np.pad(
                    vals,
                    (0, self.k - k_eff),
                )

                vecs = np.pad(
                    vecs,
                    ((0, 0), (0, self.k - k_eff)),
                )

            # Remove very small numerical values.
            if self.tol is not None and self.tol > 0:
                vals[np.abs(vals) < self.tol] = 0.0
                vecs[np.abs(vecs) < self.tol] = 0.0

            return vals, vecs

        # p=0: node spectrum
        lam0, vec0 = k_eigs(Lap0)

        # p=1: edge spectrum
        lam1, vec1 = k_eigs(Lap1)

        # Keep numerical eigensolves in the requested precision, but always
        # provide float32 tensors to the neural-network pipeline.
        pe0 = torch.from_numpy(
            vec0
        ).to(dtype=torch.float32)

        pe1 = torch.from_numpy(
            vec1
        ).to(dtype=torch.float32)

        # p=0 -> node PE
        data = add_node_attr(
            data,
            pe0,
            attr_name=self.node_attr_name,
        )

        # p=1 -> edge PE, realigned from PathComplex's internal (lex, self-loop-free)
        # edge order to data.edge_index's actual column order.
        pe1 = _reindex_pe_edge(pe1, data.edge_index, data.num_nodes)

        if self.edge_attr_name:
            data[self.edge_attr_name] = pe1
        else:
            data = add_edge_attr(
                data,
                pe1,
            )

        # Since node and edge spectra both have length k, they can be stacked.
        data["Lambda"] = (
            torch.from_numpy(
                np.stack([lam0, lam1], axis=0)
            )
            .to(dtype=torch.float32)
            .unsqueeze(0)
        )

        return data
