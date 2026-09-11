import numpy as np
from numpy.linalg import matrix_rank
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from reashortestpaths import *


def path_homology(D, pmax):
    """
    Python translation of the MATLAB code 'pathhomology(D, pmax)'.

    :param D: A NetworkX DiGraph
    :param pmax: maximum path length
    :return: Dictionary 'y' analogous to the MATLAB struct.
    """
    # --- Preliminary setup ---
    # Ensure we have a DiGraph
    if not isinstance(D, nx.DiGraph):
        raise TypeError("D must be a NetworkX DiGraph.")

    # Number of vertices
    n = D.number_of_nodes()

    # If the graph has no explicit node labels or 'Name', we create them.
    # We'll treat D's nodes as strings if possible, but we also want a stable ordering.
    original_nodes = sorted(list(D.nodes()))

    # Create a map node_label -> integer index (0-based)
    node_to_idx = {node: i for i, node in enumerate(original_nodes)}

    # Remove any self-loops
    self_loops = list(nx.selfloop_edges(D))
    if len(self_loops) > 0:
        print("Warning: removing self-loops!")
        D.remove_edges_from(self_loops)

    # Force all edge weights to 1 (mirroring the MATLAB code)
    # In MATLAB, we had D.Edges.Weight = ones(...). In Python, we can store it as an attribute.
    for u, v, data in D.edges(data=True):
        data["weight"] = 1

    # --- Build adjacency matrix A (n x n) ---
    # We'll keep it in a NumPy array or SciPy sparse matrix.
    A = lil_matrix((n, n), dtype=np.int64)
    for u, v in D.edges():
        A[node_to_idx[u], node_to_idx[v]] = 1
    A = A.tocsc()  # convert to CSC for faster multiplication

    # --- Compute numPaths = I + A + A^2 + ... + A^pmax ---
    # We'll do repeated multiplication
    I = csc_matrix(np.eye(n, dtype=np.int64))
    numPaths = I.copy()
    cur_power = I.copy()

    for _ in range(1, pmax + 1):
        cur_power = cur_power @ A
        numPaths = numPaths + cur_power

    # --- Collect all (allowed) paths of length <= pmax ---
    # We'll replicate the logic: for each pair (s, t) with numPaths[s,t] != 0,
    # we explicitly enumerate the paths from s to t.
    # Then we bucket them by path length.

    allPaths = [[] for _ in range(pmax + 1)]  # index 0 => length-0 paths, 1 => length-1, etc.

    # We extract the (s,t) pairs where numPaths[s,t] != 0
    s_t_indices = numPaths.nonzero()  # returns (row_indices, col_indices)
    S = s_t_indices[0]
    T = s_t_indices[1]

    for i in range(len(S)):
        s = S[i]
        t = T[i]
        count_st = numPaths[s, t]
        # If there are 'count_st' paths from s to t in total, we now enumerate them:
        # We'll do that with our BFS-based enumerator or a custom function (like reashortestpaths).
        paths_st = reashortestpaths(D, original_nodes[s], original_nodes[t], count_st)
        # Bucket each path by length
        for pth in paths_st:
            length_of_path = len(pth) - 1  # #edges = #vertices-1
            if length_of_path <= pmax:
                # Convert pth's node labels to integer indices
                idx_path = [node_to_idx[x] for x in pth]
                allPaths[length_of_path].append(idx_path)

    # --- Sort each list of paths in lex order ---
    # The code sorts row-wise in MATLAB; we can replicate that with a Python sort.
    for p in range(pmax + 1):
        allPaths[p].sort()

    # --- Construct the chain complex ---
    # We'll create the "modified boundary map" bdryA, the "space of invariant p-paths" omega,
    # and the "invariant boundary map" bdry for each p.

    # Helper to compute the lex index: 1 + (p-1)*n^(p-index)
    def lex_index(path, n):
        """
        Replicate the base-n indexing:
            idx = 1 + sum_{k=0..p} [(path[k]) * n^(p-k)],
        but keep in mind MATLAB was 1-based. We'll replicate it carefully.
        Here 'path' is a list of 0-based node indices, so we do (path[k]) not (path[k]-1).
        """
        p = len(path) - 1  # if path length is p+1, dimension is p
        idx = 1
        for i, node_idx in enumerate(path):
            exp = p - i
            # In MATLAB: 1 + (path[i]-1)*n^exp.  Here path[i] is 0-based, so:
            #   contribution = (node_idx) * n^exp
            idx += node_idx * (n ** exp)
        return idx

    indA = [None] * (pmax + 1)  # indices of allowed p-paths in "elementary" p-paths
    bdryA = [None] * (pmax + 1)  # modified boundary map
    omega = [None] * (pmax + 1)  # space of invariant p-paths
    bdry = [None] * (pmax + 1)  # invariant boundary map

    for p in range(pmax + 1):
        # allPaths[p] is a list of p-paths (which have p+1 vertices)
        num_p_paths = len(allPaths[p])
        if num_p_paths > 0:
            # indA{p+1} = 1 + (allPaths{p+1}-1)*n.^(p:-1:0)'
            # We'll replicate that logic for each path
            # Construct array of "lex" indices (1-based in MATLAB style)
            indA_p = []
            for path in allPaths[p]:
                indA_p.append(lex_index(path, n))
            indA[p] = np.array(indA_p, dtype=np.int64)

            # Construct bdryA[p]: an n^p by (# of allowed p-paths) sparse matrix
            # Actually, in MATLAB: bdryA{p+1} is n^p x num_p_paths
            # We'll store it as a sparse matrix.  n^p can be huge, so be careful.
            # This direct approach can become large quickly, but we match the logic.
            dim_row = n ** p if p > 0 else 1  # n^0 = 1
            bdryA_p = lil_matrix((dim_row, num_p_paths), dtype=np.float64)

            # Populate columns of the modified boundary map
            # For each p-path, we remove one vertex at a time
            for col_idx, path in enumerate(allPaths[p]):
                # path has p+1 vertices
                for i2 in range(p + 1):
                    # remove the i2-th vertex
                    tmp_path = path[:i2] + path[i2 + 1:]
                    # row index in the "elementary" p-1 path space
                    row_idx = lex_index(tmp_path, n) - 1  # minus 1 to get 0-based
                    bdryA_p[row_idx, col_idx] = ((-1) ** i2)

            bdryA[p] = bdryA_p.tocsc()

            # Next, remove rows corresponding to *allowed* (p-1)-paths and get the kernel
            if p > 0 and indA[p - 1] is not None:
                # setdiff(1:n^p, indA{p}) in 1-based.  We do set difference in 0-based
                all_indices = set(range(n ** p))
                # but note indA[p-1] is 1-based, so shift it to 0-based
                allowed_minus_1 = set((indA[p - 1] - 1).tolist())
                rows_to_keep = list(all_indices.difference(allowed_minus_1))
                rows_to_keep.sort()
                bdryTemp = bdryA[p][rows_to_keep, :]
            else:
                # dimension p=0 or no p-1
                bdryTemp = csc_matrix((1, bdryA[p].shape[1]), dtype=np.float64)

            # Remove zero rows first if we like, or just rely on matrix_rank of the full
            # But for kernel, we typically do null space.  We can do a direct numeric approach:
            # Convert to dense for rank-based kernel:  not super scalable, but simple.
            # We'll identify the rows that are entirely zero:
            nonzero_row_mask = np.array(bdryTemp.sum(axis=1)).ravel() != 0
            bdryTemp_nz = bdryTemp[nonzero_row_mask, :].toarray()

            # We want the null space of bdryTemp_nz. If it is (R^{#cols} -> R^{#rows}), the kernel
            # is dimension #cols - rank.  But we might also want an explicit basis.
            # Here we only store the dimension or the basis? The MATLAB code does:
            #    omega{p+1} = null(full(bdryTemp(any(bdryTemp,2),:)));
            # We'll replicate by computing an orthonormal basis for the kernel with np.linalg.svd
            # or a simpler approach:
            U, s, Vt = np.linalg.svd(bdryTemp_nz, full_matrices=True)
            rank_temp = np.sum(s > 1e-14)
            # dimension of kernel:
            dim_ker_temp = bdryTemp_nz.shape[1] - rank_temp
            if dim_ker_temp <= 0:
                # no kernel
                omega_p = np.zeros((bdryTemp_nz.shape[1], 0))
            else:
                # columns of V corresponding to zero singular values form a basis for kernel
                V = Vt.T
                omega_p = V[:, rank_temp:]  # everything after the "rank_temp" columns

            omega[p] = omega_p

            # Construct the invariant boundary map: bdry[p] = bdryA[p]*omega[p], restricted to
            # the "allowed" p-1 paths if p>0.
            temp = bdryA[p].dot(omega_p)
            if p > 0 and indA[p - 1] is not None:
                # only keep rows indA[p-1]
                keep_rows = (indA[p - 1] - 1).tolist()
                # we must extract those from 'temp'
                bdry[p] = temp[keep_rows, :]
            else:
                bdry[p] = temp
        else:
            # No p-paths
            bdryA[p] = csc_matrix((n ** max(p, 1), 0), dtype=np.float64)
            omega[p] = np.array([0])  # match the MATLAB code's zero
            bdry[p] = csc_matrix((n ** max(p, 1), 0), dtype=np.float64)

    # --- Compute dimensions of boundary map images and kernels ---
    dim_im = np.zeros(pmax + 1, dtype=int)
    dim_ker_array = np.zeros(pmax + 1, dtype=int)

    for p in range(pmax + 1):
        # bdry[p] is (some_rows) x (# of columns = dimension of the p-chain space in the "invariant" sense)
        # remove zero rows:
        if bdry[p].shape[1] == 0:
            # no columns => rank=0
            dim_im[p] = 0
            dim_ker_array[p] = 0
        else:
            # Convert to dense
            mat = bdry[p]#.toarray()
            nonzero_row_mask = np.any(mat != 0, axis=1)
            mat_nz = mat[nonzero_row_mask, :]
            if mat_nz.size == 0:
                # nothing inside => rank 0
                rnk = 0
            else:
                rnk = matrix_rank(mat_nz)
            dim_im[p] = rnk
            dim_ker_array[p] = bdry[p].shape[1] - rnk

    # --- Compute Betti numbers ---
    # betti = dim_ker(1:pmax) - dim_im(2:end) in MATLAB indexing => dimension p to dimension p+1
    betti = []
    for p in range(pmax):
        betti.append(dim_ker_array[p] - dim_im[p + 1])
    betti = np.array(betti, dtype=int)

    # --- Prepare output ---
    y = {
        "allPaths": allPaths,  # list of lists-of-paths
        "bdryA": bdryA,  # modified boundary maps
        "omega": omega,  # space of invariant p-paths
        "bdry": bdry,  # invariant boundary maps
        "betti": betti  # Betti numbers in dimensions 0..(pmax-1)
    }

    return y
