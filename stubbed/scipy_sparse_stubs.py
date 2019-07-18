from .networkx_stubs import TrackedList as TList
from scipy.sparse import _sparsetools


_n_bytes = 4


def csr_matmat_pass1(fn, *args, **kwargs):
    # We are assuming that this pass would keep all the necessary data in
    # and can fuse with csr_matmat_pass2...
    # We are also assuming that the W is pipelined with R.

    # Pseudo code
    # for i in range(n_row):
    #     row_nnz = 0
    #     for jj in range(Ap[i], Ap[i+1]):
    #         j = Aj[jj]
    #         for kk in range(Bp[j], Bp[j+1]):
    #             k = Bj[kk]
    #             if not mask[k] == i:
    #                 mask[k] = i
    #                 row_nnz += 1
    #
    #     nnz = nnz + row_nnz
    #     Cp[i+1] = nnz

    n_row, n_col, *tail = args
    Ap, Aj, Bp, Bj, Cp = [TList(x, is_host_init=True) for x in tail]

    Ap_last_ptr = Ap.head
    jj_idx_dict = {}
    for i in range(n_row):
        Ap_curr_ptr = Ap[i+1]
        for jj in range(Ap_last_ptr, Ap_curr_ptr):
            # Check for jj
            if jj in jj_idx_dict.keys():
                j = jj_idx_dict[jj]
            else:
                j = Aj[jj]
                jj_idx_dict[jj] = j

            # This is a dense load...
            kk_start = Bp[j]
            kk_end = Bp[j+1]
            _ = Bj[kk_start:kk_end]

        Ap_last_ptr = Ap_curr_ptr

        # Emulate a write
        Cp[i+1] = 0


    return fn(*args, **kwargs)


def csr_matmat_pass2(fn, *args, **kwargs):
    """
    For now, we are not profiling this function.
    It seems that with compiler fusion, we should be able to
    execute this with pass 1 in the same space.
    """
    # Pseudo code:
    # next = [-1] * n_col
    # sums = [0] * n_col
    # nnz = 0
    # Cp[0] = 0
    #
    # for i in range(n_row):
    #     head = -2
    #     length = 0
    #     jj_start = Ap[i]
    #     jj_end = Ap[i+1]
    #     for jj in range(jj_start, jj_end):
    #         j = Aj[jj]
    #         v = Ax[jj]
    #         kk_start = Bp[j]
    #         kk_end = Bp[j+1]
    #         for kk in range(kk_start, kk_end):
    #             k = Bj[kk]
    #             sums[k] += v * Bx[kk]
    #
    #             if next[k] == -1:
    #                 next[k] = head
    #                 head = k
    #                 length += 1
    #
    #     for jj in range(length):
    #         if not sums[head] == 0:
    #             Cj[nnz] = head
    #             Cx[nnz] = sums[head]
    #             nnz += 1
    #
    #         temp = head
    #         head = next[head]
    #         next[temp] = -1
    #         sums[temp] = 0
    #
    #         Cp[i+1] = nnz

    return fn(*args, **kwargs)


def csr_matvec(fn, *args, **kwargs):
    print("in csr_matvec")
    n_row, n_col, *tail = args
    Ap, Aj, Ax, Xx, Yx = [TList(x, is_host_init=True) for x in tail]

    Yx_n_row = Yx[0:n_row]
    Ap_last_ptr = Ap.head
    for i, e in enumerate(Yx_n_row):
        Ap_next_ptr = Ap[i+1]
        _ = Ax[Ap_last_ptr:Ap_next_ptr]
        Aj_sub = Aj[Ap_last_ptr:Ap_next_ptr]
        for j_Aj in Aj_sub:
            _ = Xx[j_Aj]
        Ap_last_ptr = Ap_next_ptr

    return fn(*args, **kwargs)


def csr_matvecs(fn, *args, **kwargs):
    n_row, _, n_vecs, *tail = args
    Ap, Aj, Ax, Xx, Yx = [
        TList(x, is_host_init=True) for x in tail
    ]
    Ap_last_ptr = Ap.head

    for i in range(n_row):
        y_ptr = n_vecs * i
        _ = Yx[y_ptr:y_ptr+n_vecs]

        Ap_new_ptr = Ap[i+1]
        Aj_jj_sublist = Aj[Ap_last_ptr:Ap_new_ptr]
        Ax_jj_sublist = Ax[Ap_last_ptr:Ap_new_ptr]

        for j, a in zip(Aj_jj_sublist, Ax_jj_sublist):
            x_ptr = n_vecs * j
            _ = Xx[x_ptr:x_ptr+n_vecs]

        Ap_last_ptr = Ap_new_ptr

    return fn(*args, **kwargs)


def csc_matvec(fn, *args, **kwargs):
    print("WARNING: csc matvec is not yet supported; please use csr matvec.")
    return fn(*args, **kwargs)


def csc_matvecs(fn, *args, **kwargs):
    print("WARNING: csc matvecs is not yet supported; please use csc matvecs.")
    return fn(*args, **kwargs)


def stub():
    _sparsetools.csr_matmat_pass1 = partial(
        csr_matmat_pass1, _sparsetools.csr_matmat_pass1
    )
    _sparsetools.csr_matmat_pass2 = partial(
        csr_matmat_pass2, _sparsetools.csr_matmat_pass2
    )
    _sparsetools.csr_matvec = partial(
        csr_matvec, _sparsetools.csr_matvec
    )
    _sparsetools.csr_matvecs = partial(
        csr_matvecs, _sparsetools.csr_matvecs
    )
    _sparsetools.csc_matvec = partial(
        csc_matvec, _sparsetools.csc_matvec
    )
    _sparsetools.csc_matvecs = partial(
        csc_matvecs, _sparsetools.csc_matvecs
    )
