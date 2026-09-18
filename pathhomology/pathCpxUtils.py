"""
New implementations for faster computation of the boundary matrices
and the path array lists. This includes a new implementation of the
path complex Hodge Laplacian and an approximation of the space Omega
"""
#from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.linalg import null_space
from scipy.sparse.linalg import svds, splu
from scipy.sparse import (csc_matrix, csr_matrix, issparse, coo_matrix, spmatrix,
                          spdiags,csr_matrix, csr_array)


# ---------------------------
# 1) Paths up to pmax (fast)
# ---------------------------
def all_paths_csr(A: sp.spmatrix | np.ndarray, pmax: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build all directed p-paths for p=0..pmax using CSR adjacency lists.

    Returns
    -------
    allPaths : list, allPaths[p] shape (#paths_p, p+1)
    indA     : list, indA[p]    shape (#paths_p,), lex indices in base-n
    """
    if pmax < 0:
        raise ValueError("pmax must be non-negative")

    A_csr = A.tocsr() if sp.issparse(A) else sp.csr_matrix(A)
    n = A_csr.shape[0]
    indptr, indices = A_csr.indptr, A_csr.indices

    def extend(prev_paths: np.ndarray) -> np.ndarray:
        if prev_paths.size == 0:
            return np.zeros((0, prev_paths.shape[1] + 1), dtype=np.int64)

        last = prev_paths[:, -1]
        counts = np.empty(prev_paths.shape[0], dtype=np.int64)
        nbrs_list = []

        # Loop over paths, but only touch real neighbors (fast CSR slicing)
        for i, lv in enumerate(last):
            nbrs = indices[indptr[lv]:indptr[lv + 1]]
            nbrs_list.append(nbrs)
            counts[i] = nbrs.size

        total = int(counts.sum())
        if total == 0:
            return np.zeros((0, prev_paths.shape[1] + 1), dtype=np.int64)

        reps = np.repeat(np.arange(prev_paths.shape[0], dtype=np.int64), counts)
        base = prev_paths[reps]
        new_last = np.concatenate(nbrs_list).astype(np.int64, copy=False)
        return np.column_stack([base, new_last])

    allPaths: list[np.ndarray] = [None] * (pmax + 1)
    allPaths[0] = np.arange(n, dtype=np.int64).reshape(-1, 1)
    for p in range(1, pmax + 1):
        allPaths[p] = extend(allPaths[p - 1])

    # lex indices
    indA: list[np.ndarray] = [None] * (pmax + 1)
    for p in range(pmax + 1):
        P = allPaths[p]
        if P.size == 0:
            indA[p] = np.array([], dtype=np.int64)
        else:
            weights = n ** np.arange(p, -1, -1, dtype=np.int64)
            indA[p] = (P @ weights).astype(np.int64, copy=False)

    return allPaths, indA

def sort_indA(indA: list[np.ndarray]) -> list[np.ndarray]:
    """Sort each indA[p] once for fast membership/mapping."""
    return [np.sort(np.asarray(x, dtype=np.int64)) for x in indA]

# --------------------------------------------
# 2) Split boundary: allowed vs disallowed faces
# --------------------------------------------

def _face_lex_indices(P: np.ndarray, p: int, n: int) -> list[np.ndarray]:
    """
    P: (#paths, p+1) vertex indices
    Returns list of length (p+1):
      lxs[i] = lex index (base-n) of face deleting vertex i, shape (#paths,)
    Computed without allocating face arrays.
    """
    # face length is p => weights n^(p-1)...n^0
    pow_face = n ** np.arange(p - 1, -1, -1, dtype=np.int64)

    lxs = []
    for i in range(p + 1):
        left = 0 if i == 0 else (P[:, :i] @ pow_face[:i])
        right = 0 if i == p else (P[:, i + 1:] @ pow_face[i:])
        lxs.append((left + right).astype(np.int64, copy=False))
    return lxs


def boundary_split_fast(allPaths, indA_sorted, p, n):
    """
    Build split boundary for allowed p-paths:
      Bd_allowed    : (#allowed_prev x #paths_p)
      Bd_disallowed : (#disallowed_seen x #paths_p)

    Fixes:
      1) Safe membership test for searchsorted (avoids allowed_prev[pos] OOB).
      2) Handles n_allowed == 0 case (your earlier fix).
    """
    if p == 0:
        m0 = allPaths[0].shape[0]
        return sp.csc_matrix((0, m0)), sp.csc_matrix((0, m0))

    P = np.asarray(allPaths[p], dtype=np.int64)
    m = P.shape[0]

    allowed_prev = np.asarray(indA_sorted[p - 1], dtype=np.int64)  # must be sorted
    n_allowed = allowed_prev.size

    if m == 0:
        return sp.csc_matrix((n_allowed, 0)), sp.csc_matrix((0, 0))

    # face lex weights for length-p faces
    pow_face = n ** np.arange(p - 1, -1, -1, dtype=np.int64)

    # lxs[i] = lex indices for face deleting vertex i
    lxs = []
    for i in range(p + 1):
        left = 0 if i == 0 else (P[:, :i] @ pow_face[:i])
        right = 0 if i == p else (P[:, i + 1:] @ pow_face[i:])
        lxs.append((left + right).astype(np.int64, copy=False))

    cols = np.arange(m, dtype=np.int64)

    # -------------------------------
    # Case: no allowed (p-1)-paths
    # -------------------------------
    if n_allowed == 0:
        Bd_allowed = sp.csc_matrix((0, m))

        dis_lx = np.concatenate(lxs) if lxs else np.array([], dtype=np.int64)
        if dis_lx.size == 0:
            return Bd_allowed, sp.csc_matrix((0, m))

        uniq, inv = np.unique(dis_lx, return_inverse=True)
        rowsD = inv
        colsD = np.tile(cols, p + 1)
        dataD = np.concatenate([
            np.full(m, (-1.0 if (i & 1) else 1.0), dtype=np.float64)
            for i in range(p + 1)
        ])

        Bd_disallowed = sp.coo_matrix((dataD, (rowsD, colsD)),
                                      shape=(uniq.size, m)).tocsc()
        return Bd_allowed, Bd_disallowed

    # -------------------------------
    # Normal case: split allowed/disallowed
    # -------------------------------
    rowsA_list, colsA_list, dataA_list = [], [], []
    disallowed_all = []

    # First pass: allowed COO + collect disallowed lex indices
    for i2, lx in enumerate(lxs):
        sgn = -1.0 if (i2 & 1) else 1.0

        pos = np.searchsorted(allowed_prev, lx)

        ok = (pos < n_allowed)          # safe mask first
        ok_idx = np.where(ok)[0]
        if ok_idx.size:
            ok[ok_idx] = (allowed_prev[pos[ok_idx]] == lx[ok_idx])

        if np.any(ok):
            rowsA_list.append(pos[ok])
            colsA_list.append(cols[ok])
            dataA_list.append(np.full(int(ok.sum()), sgn, dtype=np.float64))

        if np.any(~ok):
            disallowed_all.append(lx[~ok])

    # Allowed block
    if rowsA_list:
        rowsA = np.concatenate(rowsA_list)
        colsA = np.concatenate(colsA_list)
        dataA = np.concatenate(dataA_list)
        Bd_allowed = sp.coo_matrix((dataA, (rowsA, colsA)),
                                   shape=(n_allowed, m)).tocsc()
    else:
        Bd_allowed = sp.csc_matrix((n_allowed, m))

    # Disallowed block
    if not disallowed_all:
        return Bd_allowed, sp.csc_matrix((0, m))

    dis_lx = np.concatenate(disallowed_all)
    uniq, inv = np.unique(dis_lx, return_inverse=True)

    # Second pass: rebuild disallowed entries
    rowsD_list, colsD_list, dataD_list = [], [], []
    offset = 0

    for i2, lx in enumerate(lxs):
        sgn = -1.0 if (i2 & 1) else 1.0

        pos = np.searchsorted(allowed_prev, lx)

        ok = (pos < n_allowed)
        ok_idx = np.where(ok)[0]
        if ok_idx.size:
            ok[ok_idx] = (allowed_prev[pos[ok_idx]] == lx[ok_idx])

        bad = ~ok
        bad_count = int(bad.sum())
        if bad_count:
            rowsD_list.append(inv[offset:offset + bad_count])
            colsD_list.append(cols[bad])
            dataD_list.append(np.full(bad_count, sgn, dtype=np.float64))
            offset += bad_count

    rowsD = np.concatenate(rowsD_list) if rowsD_list else np.array([], dtype=np.int64)
    colsD = np.concatenate(colsD_list) if colsD_list else np.array([], dtype=np.int64)
    dataD = np.concatenate(dataD_list) if dataD_list else np.array([], dtype=np.float64)

    Bd_disallowed = sp.coo_matrix((dataD, (rowsD, colsD)),
                                  shape=(uniq.size, m)).tocsc()
    return Bd_allowed, Bd_disallowed



# ---------------------------
# 3) Omega + bdry construction
# ---------------------------

from scipy.sparse.linalg import ArpackError


def nullspace_numeric(
    B: sp.spmatrix,
    tol: float = 1e-12,
    max_dense: int = 2000,
) -> np.ndarray:
    """
    Compute an orthonormal basis of ker(B).

    Notes
    -----
    For large sparse matrices this routine uses ``svds`` only in cases where
    the compact SVD can contain the full right nullspace.  In particular, a
    large *wide* matrix (n_columns > n_rows) has at least n_columns-n_rows
    right-null directions that are not returned by the compact SVD.  The old
    implementation silently missed those directions.  We now fail loudly in
    that case instead of returning an incorrect kernel.

    Path-complex computations with q=2 do not use this routine: their kernel
    is constructed exactly by :func:`omega2_basis_exact` below.
    """
    B = B.tocsr()
    m, n = B.shape

    # Trivial domain.
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # Zero map: every vector lies in the kernel.
    if m == 0 or B.nnz == 0:
        return np.eye(n, dtype=float)

    # Dense computation is reliable for moderate sizes.
    if m <= max_dense and n <= max_dense:
        return null_space(B.toarray(), rcond=tol)

    # For a large wide matrix, the compact SVD omits the n-m right-null
    # directions.  Returning vt[s <= tol].T is therefore mathematically wrong.
    if n > m:
        raise RuntimeError(
            "nullspace_numeric cannot recover the full right nullspace of a "
            f"large wide sparse matrix with shape {B.shape} using scipy.svds. "
            "Use a structure-aware kernel construction instead."
        )

    min_dim = min(m, n)

    # svds requires k < min(B.shape).  Handle the one-column case directly.
    if min_dim == 1:
        col_norm = np.sqrt(B.multiply(B).sum())
        if col_norm <= tol:
            return np.eye(n, dtype=float)
        return np.zeros((n, 0), dtype=float)

    k_try = min(min_dim - 1, 50)
    ncv = min(min_dim - 1, max(4 * k_try + 1, 20))

    try:
        if ncv > k_try:
            _, s, vt = svds(B, k=k_try, which="SM", ncv=ncv)
        else:
            _, s, vt = svds(B, k=k_try, which="SM")
    except ArpackError:
        print(
            "\nARPACK failure in nullspace_numeric:"
            f"\n  B.shape = {B.shape}"
            f"\n  B.nnz   = {B.nnz}"
            f"\n  k_try   = {k_try}"
            f"\n  ncv     = {ncv}",
            flush=True,
        )
        raise

    keep = s <= tol

    if not np.any(keep):
        return np.zeros((n, 0), dtype=float)

    # If every requested singular triplet is numerically zero, there may be
    # still more null directions outside the k_try vectors we requested.
    # Returning a partial basis would silently corrupt the path Laplacian.
    if np.all(keep):
        raise RuntimeError(
            "nullspace_numeric found at least "
            f"{k_try} null directions for B.shape={B.shape}, but cannot certify "
            "that this is the complete nullspace with the current svds call."
        )

    return vt[keep].T


def omega2_basis_exact(
    Bd_disallowed: sp.spmatrix,
    *,
    check_structure: bool = True,
) -> csr_matrix:
    """
    Construct an exact sparse orthonormal basis of Omega_2.

    For an allowed directed 2-path (v0,v1,v2), the first and last faces
    (v1,v2) and (v0,v1) are automatically allowed.  Hence the only possibly
    disallowed face is the middle face (v0,v2), with sign -1.  Therefore each
    column of the disallowed boundary has at most one nonzero entry.

    The kernel then decomposes into independent blocks:
      * every zero column contributes one standard basis vector;
      * a row supported on r columns contributes an (r-1)-dimensional
        zero-sum subspace.

    We use the orthonormal Helmert basis for each row block.  When r=2 this is
    simply (e_i - e_j)/sqrt(2).

    Parameters
    ----------
    Bd_disallowed : sparse matrix, shape (n_disallowed_faces, n_allowed_2paths)
        Disallowed part of the p=2 boundary.
    check_structure : bool
        If True, verify the p=2 structural assumptions before constructing the
        basis.

    Returns
    -------
    Omega2 : csr_matrix, shape (n_allowed_2paths, dim(Omega_2))
        Sparse matrix whose columns are an orthonormal basis of ker(Bd_disallowed).
    """
    D = Bd_disallowed.tocsr()
    m, ncols = D.shape

    if ncols == 0:
        return csr_matrix((0, 0), dtype=float)

    if m == 0:
        return sp.identity(ncols, dtype=float, format="csr")

    D_csc = D.tocsc()
    col_nnz = np.diff(D_csc.indptr)

    if check_structure and np.any(col_nnz > 1):
        bad = int(np.count_nonzero(col_nnz > 1))
        raise ValueError(
            "omega2_basis_exact expects the p=2 disallowed boundary, where "
            f"each column has at most one nonzero; found {bad} columns with >1."
        )

    zero_cols = np.flatnonzero(col_nnz == 0)
    row_nnz = np.diff(D.indptr)
    omega_dim = int(zero_cols.size + np.maximum(row_nnz - 1, 0).sum())

    rows = []
    cols = []
    vals = []
    basis_col = 0

    # Zero columns of D are themselves kernel basis vectors.
    if zero_cols.size:
        rows.extend(zero_cols.tolist())
        cols.extend(range(basis_col, basis_col + zero_cols.size))
        vals.extend([1.0] * zero_cols.size)
        basis_col += int(zero_cols.size)

    # Each nonempty row has support disjoint from every other row support.
    # On a block of size r, construct an orthonormal basis of the zero-sum
    # hyperplane using Helmert vectors.
    for r in np.flatnonzero(row_nnz >= 2):
        start, end = D.indptr[r], D.indptr[r + 1]
        js = D.indices[start:end]
        ds = D.data[start:end]
        block_size = js.size

        if check_structure:
            # At p=2 all disallowed entries arise from deleting the middle
            # vertex, so they must carry the same coefficient (-1).
            if not np.allclose(ds, ds[0], rtol=0.0, atol=1e-14):
                raise ValueError(
                    "omega2_basis_exact expected equal coefficients within "
                    f"row {r}, got {ds}."
                )

        # Helmert basis: for j=1,...,r-1, put
        #   1/sqrt(j(j+1)) on the first j entries and
        #   -j/sqrt(j(j+1)) on entry j+1.
        for j in range(1, block_size):
            scale = 1.0 / np.sqrt(j * (j + 1.0))

            rows.extend(js[:j].tolist())
            cols.extend([basis_col] * j)
            vals.extend([scale] * j)

            rows.append(int(js[j]))
            cols.append(basis_col)
            vals.append(-j * scale)

            basis_col += 1

    if basis_col != omega_dim:
        raise RuntimeError(
            f"Internal Omega_2 dimension mismatch: built {basis_col}, "
            f"expected {omega_dim}."
        )

    return sp.coo_matrix(
        (np.asarray(vals, dtype=float),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(ncols, omega_dim),
    ).tocsr()


def omega_and_bdry(
    allPaths: list[np.ndarray],
    indA_sorted: list[np.ndarray],
    qmax: int,
    n: int,
    tol: float = 1e-12,
):
    """
    Compute omega[p] and boundary operators bdry[p] for p=0..qmax.

    ``bdry[p]`` has shape (dim Omega_{p-1}, dim Omega_p) and is stored as
    sparse CSR.

    For q=2 we exploit the special directed-path structure and build Omega_2
    exactly, avoiding a numerical sparse SVD altogether.

    Notes
    -----
    To compute L_p one needs bdry[p] and bdry[p+1], so choose
    qmax = max(p_list) + 1.
    """
    if qmax < 0:
        raise ValueError("qmax must be >= 0")
    if qmax >= len(allPaths):
        raise ValueError(f"Need allPaths up to {qmax}, but len(allPaths)={len(allPaths)}")
    if qmax >= len(indA_sorted):
        raise ValueError(
            f"Need indA_sorted up to {qmax}, but len(indA_sorted)={len(indA_sorted)}"
        )

    omega: list = [None] * (qmax + 1)
    bdry: list[csr_matrix] = [None] * (qmax + 1)

    # q=0: every vertex is invariant.  Keep the identity sparse.
    m0 = allPaths[0].shape[0]
    omega[0] = sp.identity(m0, dtype=float, format="csr")
    bdry[0] = csr_matrix((0, m0))

    for q in range(1, qmax + 1):
        Bd_allowed, Bd_disallowed = boundary_split_fast(
            allPaths, indA_sorted, q, n
        )

        if Bd_disallowed.shape[0] == 0:
            # No disallowed faces: Omega_q is the full allowed path space.
            omega_q = sp.identity(
                Bd_allowed.shape[1], dtype=float, format="csr"
            )
        elif q == 2:
            # Exact and sparse; no SVD/ARPACK.
            omega_q = omega2_basis_exact(Bd_disallowed)
        else:
            # Generic fallback.  It now fails loudly rather than silently
            # returning an incomplete nullspace in unsupported sparse cases.
            omega_q = nullspace_numeric(Bd_disallowed, tol=tol)

        omega[q] = omega_q

        Btilde = Bd_allowed @ omega_q

        omega_prev = omega[q - 1]
        if omega_prev.shape[1] == 0:
            bdry[q] = csr_matrix((0, Btilde.shape[1]))
        else:
            bdry[q] = csr_matrix(omega_prev.T @ Btilde)

    return omega, bdry

# ---------------------------
# 4) Hodge Laplacians
# ---------------------------

def hodge_laplacian_path_complex(
    A,
    pmax,                 # kept for API, but not used to compute beyond what we need
    p_list,
    tol: float = 1e-12,
    fmt: str = "csr",
    info: bool = False
):
    if not p_list:
        return [], {}

    if min(p_list) < 0:
        raise ValueError("p must be >= 0")

    # Need bdry[p+1] for L_p
    qmax_needed = max(p_list) + 1

    # If the caller's pmax is too small, fail early with a helpful message
    if pmax < qmax_needed:
        raise ValueError(
            f"Built PathComplex with pmax={pmax}, but computing L_p up to p={max(p_list)} "
            f"requires paths up to pmax={qmax_needed}."
        )

    # Build only what is needed (0..qmax_needed)
    allPaths, indA = all_paths_csr(A, pmax=qmax_needed)
    n = A.shape[0]

    indA_sorted = sort_indA(indA)
    omega, bdry = omega_and_bdry(allPaths, indA_sorted, qmax=qmax_needed, n=n, tol=tol)

    Lap_list = []
    for p in p_list:
        Bp  = bdry[p]
        Bp1 = bdry[p + 1]
        L = (Bp1 @ Bp1.T) + (Bp.T @ Bp if p > 0 else 0)
        Lap_list.append(L.asformat(fmt) if fmt else L)

    info_dict = {}
    if info:
        info_dict = {
            "n": n,
            "pmax_input": pmax,
            "qmax_needed": qmax_needed,
            "num_paths": {q: int(allPaths[q].shape[0]) for q in range(qmax_needed + 1)},
            "omega_dims": {q: int(omega[q].shape[1]) for q in range(qmax_needed + 1)},
            "bdry_shapes": {q: tuple(bdry[q].shape) for q in range(qmax_needed + 1)},
        }

    return (Lap_list, info_dict) if info else Lap_list


def hodge_laplacians_from_precomputed(
    allPaths,
    indA_sorted,
    n: int,
    p_list,
    tol: float = 1e-12,
    fmt: str = "csr",
    info: bool = False
):
    """
    Compute L_p for p in p_list using precomputed allPaths + indA_sorted.
    Avoids rebuilding paths and sorting.

    Needs paths up to qmax = max(p_list)+1.
    """
    if not p_list:
        return [], {}

    p_list = list(p_list)
    qmax = max(p_list) + 1
    if qmax >= len(allPaths):
        raise ValueError(f"Need allPaths up to {qmax}, but len(allPaths)={len(allPaths)}")
    if qmax >= len(indA_sorted):
        raise ValueError(f"Need indA_sorted up to {qmax}, but len(indA_sorted)={len(indA_sorted)}")

    omega, bdry = omega_and_bdry(allPaths, indA_sorted, qmax=qmax, n=n, tol=tol)

    Lap_list = []
    for p in p_list:
        Bp  = bdry[p]
        Bp1 = bdry[p + 1]
        L = (Bp1 @ Bp1.T) + (Bp.T @ Bp if p > 0 else 0)
        Lap_list.append(L.asformat(fmt) if fmt else L)

    info_dict = {}
    if info:
        info_dict = {
            "n": n,
            "qmax": qmax,
            "num_paths": {q: int(allPaths[q].shape[0]) for q in range(qmax + 1)},
            "omega_dims": {q: int(omega[q].shape[1]) for q in range(qmax + 1)},
            "bdry_shapes": {q: tuple(bdry[q].shape) for q in range(qmax + 1)},
        }

    return (Lap_list, info_dict) if info else Lap_list


def normalize_hodge_laplacian(L: spmatrix) -> csr_matrix:
    """
    :param L: sparse matrix -> hodge laplacian
    :return: normalized laplacian
    """
    L = L.tocsr()
    m, n = L.shape
    if m != n:
        raise ValueError(f"Expected square Laplacian, got shape {L.shape}")

    # "degree" choice: row-sum of absolute values of L
    d = np.asarray(np.abs(L).sum(axis=1)).ravel()

    with np.errstate(divide="ignore"):
        inv_sqrt = 1.0 / np.sqrt(d)
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0

    D_inv_sqrt = sp.diags(inv_sqrt, offsets=0, shape=(m, m), format="csr")
    return (D_inv_sqrt @ L @ D_inv_sqrt).tocsr()
