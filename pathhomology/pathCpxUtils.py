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
from scipy.sparse.linalg import svds, splu, ArpackError
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

def has_one_nnz_per_col(B: sp.spmatrix) -> bool:
    """True if every column of ``B`` carries at most one nonzero.

    Always holds for the disallowed boundary block at p=2, which is the only
    block ``omega_and_bdry`` builds when ``pmax=2``. A 2-path (v0,v1,v2) has
    faces (v1,v2), (v0,v2), (v0,v1); ``all_paths_csr`` builds it by extending
    the 1-path (v0,v1) along an edge v1->v2, so the outer two faces are edges by
    construction and only the interior face (v0,v2) can be disallowed.
    """
    Bc = B.tocsc(copy=True)
    Bc.eliminate_zeros()
    counts = np.diff(Bc.indptr)
    return bool(counts.size == 0 or counts.max() <= 1)


def nullspace_one_nnz_per_col(B: sp.spmatrix) -> csr_matrix:
    """Exact orthonormal kernel of a matrix with at most one nonzero per column.

    Such a matrix decouples completely. Column j is either zero -- so ``e_j``
    spans a kernel direction -- or hits exactly one row r. The columns sharing
    row r, with coefficients c, are constrained only by ``c . x = 0``; that
    sum-zero subspace has dimension ``|group| - 1`` and an orthonormal basis
    given by the trailing Householder columns of ``c/||c||``. A group of size 1
    pins its column to zero and contributes nothing. Distinct groups occupy
    disjoint coordinates, so cross-group orthogonality is automatic.

    O(nnz) time and a *sparse* result, against the dense SVD's O(m n min(m,n))
    and O(max(m,n)^2) memory -- on directed_roman_empire the same subspace is
    14k nonzeros instead of a 6.2 GB dense array, which is also what puts it
    out of reach of ``scipy.linalg.null_space`` (LAPACK indexes with int32 and
    would need a 67133 x 67133 factor).

    Raises:
        ValueError: if some column has more than one nonzero.
    """
    Bc = B.tocsc(copy=True)
    Bc.eliminate_zeros()
    m, n = Bc.shape
    counts = np.diff(Bc.indptr)
    if counts.size and counts.max() > 1:
        raise ValueError(
            "nullspace_one_nnz_per_col requires at most one nonzero per column; "
            f"found a column with {counts.max()}."
        )

    nz_cols = np.nonzero(counts == 1)[0]
    zero_cols = np.nonzero(counts == 0)[0]

    # Zero columns contribute e_j, already orthonormal.
    data = [np.ones(zero_cols.size)]
    rows = [zero_cols]
    cols = [np.arange(zero_cols.size)]
    next_col = zero_cols.size

    if nz_cols.size:
        order = np.argsort(Bc.indices, kind="stable")
        r_sorted = Bc.indices[order]
        c_sorted = nz_cols[order]
        v_sorted = Bc.data[order]
        starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
        sizes = np.diff(np.r_[starts, r_sorted.size])
        # Only groups of size >= 2 contribute. On sparse graphs nearly every
        # group is a singleton, so filtering first keeps this off a Python-level
        # loop over essentially every row of the matrix.
        for s, g in zip(starts[sizes >= 2], sizes[sizes >= 2]):
            grp = c_sorted[s:s + g]
            u = v_sorted[s:s + g].astype(float)
            u /= np.linalg.norm(u)
            w = u.copy()
            w[0] += 1.0 if u[0] >= 0 else -1.0
            w /= np.linalg.norm(w)
            # H = I - 2 w w^T is orthogonal with H e_0 proportional to u, so its
            # remaining columns are an orthonormal basis of u-perp.
            block = -2.0 * np.outer(w, w[1:])
            block[np.arange(1, g), np.arange(g - 1)] += 1.0
            data.append(block.ravel(order="F"))
            rows.append(np.tile(grp, g - 1))
            cols.append(np.repeat(np.arange(next_col, next_col + g - 1), g))
            next_col += g - 1

    return csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, next_col),
    )

def nullspace_numeric(
    B: sp.spmatrix,
    tol: float = 1e-12,
    max_dense: int = 2000
) -> np.ndarray:
    m, n = B.shape

    # No columns: trivial domain.
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # No rows: every vector is in the kernel.
    if m == 0:
        return np.eye(n, dtype=float)

    # Exact closed form; always applies at pmax=2. Preferred at every size: it
    # is O(nnz) rather than O(n^3) and returns a sparse basis.
    if has_one_nnz_per_col(B):
       return nullspace_one_nnz_per_col(B)

    # Dense computation for moderate-sized matrices.
    if m <= max_dense and n <= max_dense:
        return null_space(B.toarray(), rcond=tol)

    min_dim = min(m, n)
    k_try = min(max(1, min_dim - 1), 50)

    # Give ARPACK a larger Krylov subspace whenever possible.
    ncv = min(
        min_dim - 1,
        max(4 * k_try + 1, 20),
    )

    try:
        if ncv > k_try:
            _, s, vt = svds(
                B,
                k=k_try,
                which="SM",
                ncv=ncv,
            )
        else:
            _, s, vt = svds(
                B,
                k=k_try,
                which="SM",
            )

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

    return vt[keep].T


def omega_and_bdry(
    allPaths: list[np.ndarray],
    indA_sorted: list[np.ndarray],
    qmax: int,
    n: int,
    tol: float = 1e-12
):
    """
    Compute omega[p] and boundary operators bdry[p] in omega-coordinates or p = 0..qmax only.
    bdry[p] has shape (dim omega[p-1], dim omega[p]) and is stored as sparse CSR.

    Notes:
    - To compute Laplacian L_p you need bdry[p] and bdry[p+1],
      so choose qmax = max(p_list) + 1.
    """
    if qmax < 0:
        raise ValueError("qmax must be >= 0")
    if qmax >= len(allPaths):
        raise ValueError(f"Need allPaths up to {qmax}, but len(allPaths)={len(allPaths)}")
    if qmax >= len(indA_sorted):
        raise ValueError(f"Need indA_sorted up to {qmax}, but len(indA_sorted)={len(indA_sorted)}")

    omega: list = [None] * (qmax + 1)
    bdry:  list[csr_matrix] = [None] * (qmax + 1)

    # q=0: every 0-path is invariant, so omega[0] is the identity. Keep it
    # sparse — a dense eye(#nodes) costs O(n^2) memory and turns the products
    # below into huge no-op GEMMs (e.g. 3.8 GiB / 40 TFLOP for n=22662).
    m0 = allPaths[0].shape[0]
    omega[0] = sp.identity(m0, dtype=float, format="csr")
    bdry[0] = csr_matrix((0, m0))

    for q in range(1, qmax + 1):
        Bd_allowed, Bd_disallowed = boundary_split_fast(allPaths, indA_sorted, q, n)

        if Bd_disallowed.shape[0] == 0:
            # No disallowed faces => the kernel is everything. Same result as
            # nullspace_numeric (which returns eye(n) here), but sparse and free.
            omega_q = sp.identity(Bd_allowed.shape[1], dtype=float, format="csr")
        else:
            omega_q = nullspace_numeric(Bd_disallowed, tol=tol)
        omega[q] = omega_q

        # sparse @ sparse -> sparse; sparse @ dense -> dense (#allowed_{q-1} x k_q)
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
