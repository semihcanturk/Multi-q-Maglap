import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix, eye
from pathhomology.pathCpxUtils import *


class PathComplex:
    def __init__(self, G: nx.DiGraph, pmax: int):
        if not isinstance(G, nx.DiGraph):
            raise TypeError("G must be a NetworkX DiGraph.")

        self.G = G.copy()
        self.pmax = int(pmax)

        # Remove self-loops once
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        # Stable node order + size
        self.nodelist = sorted(self.G.nodes())
        self.n = len(self.nodelist)
        self.node_to_idx = {u: i for i, u in enumerate(self.nodelist)}

        # Build adjacency ONCE (CSR for fast row-neighbor access)
        # weight=None means treat as unweighted 0/1 adjacency
        
        # nx.to_scipy_sparse_matrix was removed in networkx 3.0; to_scipy_sparse_array
        # returns a csr_array, so wrap it with a csr_matrix to preserve downstream self.A behavior.
        self.A = csr_matrix(nx.to_scipy_sparse_array(
            self.G,
            nodelist=self.nodelist,
            weight=None,
            dtype=np.int8,
            format="csr",
            ))

        # Precompute paths + lex indices ONCE
        self.all_paths, self.indA = all_paths_csr(self.A, self.pmax)

        # Cache sorted indA once (critical for speed in boundary_split_fast)
        self.indA_sorted = sort_indA(self.indA)

    def get_adjacency(self) -> csr_matrix:
        return self.A

    def get_num_paths(self, p):
        A = self.A.tocsc()
        I = eye(self.n, format="csc", dtype=np.int64)
        numPaths = I.copy()
        cur = I.copy()
        for _ in range(p):
            cur = cur @ A
            numPaths = numPaths + cur
        return numPaths

    def get_all_paths(self):
        return self.all_paths, self.indA

    def get_boundaries(self, p: int):
        # p must be >=1 to have nontrivial boundary; but boundary_split_fast handles p=0.
        if p < 0 or p > self.pmax:
            raise ValueError(f"p must be in [0, {self.pmax}]")
        return boundary_split_fast(self.all_paths, self.indA_sorted, p=p, n=self.n)

    def get_hodge_laplacian(self, p_list, tol: float = 1e-12, fmt: str = "csr"):
        """
        Compute L_p for p in p_list.

        Important: L_p needs bdry[p+1], so you must have built paths up to max(p_list)+1.
        """
        if not p_list:
            return []

        p_max_needed = max(p_list) + 1
        if self.pmax < p_max_needed:
            raise ValueError(
                f"PathComplex built with pmax={self.pmax}, but computing L_p up to p={max(p_list)} "
                f"requires paths up to pmax={p_max_needed}."
            )

        Lap_list = hodge_laplacians_from_precomputed(
            allPaths=self.all_paths,
            indA_sorted=self.indA_sorted,
            n=self.n,
            p_list=p_list,
            tol=tol,
            fmt=fmt,
            info=False
        )
        return Lap_list

    def get_normalized_hodge_laplacian(self, p_list, tol: float = 1e-12, fmt: str = "csr"):
        Lap_list = self.get_hodge_laplacian(p_list, tol=tol, fmt=fmt)
        normalized_list = [normalize_hodge_laplacian(L).asformat(fmt) if fmt else normalize_hodge_laplacian(L)
                           for L in Lap_list]
        return normalized_list
    
    # TODO: Add methods for computing Betti numbers, harmonic representatives, etc., as needed.
    # test/Debug the following
    # Note that the functions below output lists
    # depending on the applications, we might have to adjust that
    
    def get_all_omega(self, symbolic=False, reps=False):
        all_bdryA = self.get_all_bdryA()
        all_omega, all_bdry = compute_omega(all_bdryA, self.indA, self.pmax, self.n, symbolic=symbolic, reps=reps)
        return all_omega, all_bdry

    def get_betti(self, p):
        _, all_bdry = self.get_all_omega(symbolic=False, reps=False)
        return compute_betti(all_bdry,p)

    def get_homology(self):
        all_omega, all_bdry = self.get_all_omega(symbolic=False, reps=False)
        all_hom, all_betti = compute_homology(all_bdry, all_omega, self.pmax, symbolic=False)
        return all_hom, all_betti

