from .networkx_stubs import TrackedList as TList
from scipy.sparse import _sparsetools
from functools import partial

from torch import sparse
# For debugging
from .networkx_stubs import TraceRegistry as tr
from .networkx_stubs import TraceDepRegistry as tdr
from .networkx_stubs import get_last_trace_id, add_to_ops_accum

_n_bytes = 4


def csr_matmat_pass1(fn, *args, **kwargs):
    """
    We are assuming that this pass would keep all the necessary data in
    and can fuse with csr_matmat_pass2...
    We are also assuming that the W is pipelined with R.

    Pseudo code
    for i in range(n_row):
        row_nnz = 0
        for jj in range(Ap[i], Ap[i+1]): // sparse read
            j = Aj[jj] // sparse read
            for kk in range(Bp[j], Bp[j+1]): // sparse read
                k = Bj[kk] // dense
                if not mask[k] == i:
                    mask[k] = i
                    row_nnz += 1

        nnz = nnz + row_nnz
        Cp[i+1] = nnz

    :param fn:
    :param args:
    :param kwargs:
    :return:
    """

    print("in csr_matmat_pass1")
    n_row, n_col, *tail = args
    _, Aj_raw, Bp_raw, Bj_raw, _ = tail
    n = len(Bp_raw)
    P = float(len(Aj_raw)) / float(n_row * n)
    Q = float(len(Bj_raw)) / float(n_col * n)
    # We are using the assumption that the probability of nnzs are independent.
    add_to_ops_accum(
        n_row * n_col * n * P * Q * 2
    )

    Ap, Aj, Bp, Bj, Cp = [TList(x, is_host_init=True) for x in tail]

    Ap_last_ptr, Ap_last_ptr_trace_id = Ap.__getitem__(
        0, deps=[get_last_trace_id()]
    )

    for i in range(n_row):
        Ap_curr_ptr, Ap_curr_ptr_trace_id = Ap.__getitem__(
            i+1, deps=[get_last_trace_id()]
        )
        js, js_trace_id = Aj.__getitem__(
            slice(Ap_last_ptr, Ap_curr_ptr),
            deps=[Ap_last_ptr_trace_id, Ap_curr_ptr_trace_id]
        )
        for j in js:
            kk_start, kk_start_trace_id = Bp.__getitem__(
                j, deps=[js_trace_id]
            )
            kk_end, kk_end_trace_id = Bp.__getitem__(
                j+1, deps=[js_trace_id]
            )
            _ = Bj.__getitem__(
                slice(kk_start, kk_end),
                deps=[kk_start_trace_id, kk_end_trace_id]
            )

        Ap_last_ptr = Ap_curr_ptr
        Ap_last_ptr_trace_id = Ap_curr_ptr_trace_id

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
    """
    Pseudo code
    for i in range(n_row):
        sum = Yx[i]
        for jj in range(Ap[i], Ap[i+1]):
            sum += Ax[jj] * Xx[Aj[jj]]
        Yx[i] = sum
    """
    print("in csr_matvec")

    n_row, n_col, *tail = args
    _, _, Ax_raw, _, _ = tail
    nnz = len(Ax_raw)
    add_to_ops_accum(nnz * 2)

    Ap, Aj, Ax, Xx, Yx = [TList(x, is_host_init=True) for x in tail]

    Yx_n_row, Yx_n_trace_id = Yx[0:n_row]
    Ap_last_ptr, Ap_last_ptr_trace_id = Ap.__getitem__(
        0, deps=[get_last_trace_id()]
    )
    for i, e in enumerate(Yx_n_row):
        Ap_next_ptr, Ap_next_ptr_trace_id = Ap.__getitem__(
            i+1, deps=[get_last_trace_id()]
        )
        Ax.__getitem__(
            slice(Ap_last_ptr, Ap_next_ptr),
            deps=[Ap_last_ptr_trace_id, Ap_next_ptr_trace_id]
        )
        # _ = Ax[Ap_last_ptr:Ap_next_ptr]
        Aj_sub, Aj_sub_trace_id = Aj.__getitem__(
            slice(Ap_last_ptr, Ap_next_ptr),
            deps=[Ap_last_ptr_trace_id, Ap_next_ptr_trace_id]
        )
        # Aj_sub = Aj[Ap_last_ptr:Ap_next_ptr]
        for j_Aj in Aj_sub:
            # _ = Xx[j_Aj]
            Xx.__getitem__(
                j_Aj, deps=[Aj_sub_trace_id]
            )
        Ap_last_ptr = Ap_next_ptr
        Ap_last_ptr_trace_id = Ap_next_ptr_trace_id

    return fn(*args, **kwargs)


def csr_matvecs(fn, *args, **kwargs):
    """
    Pseudo code
    for i in range(n_row):
        Y_ptr = Yx + n_vecs * i             # 2 * n_row ops
        for jj in range(Ap[i], Ap[i+1):     # nnz
            j = Aj[jj]
            a = Ax[jj]
            x = Xx + n_vecs * j             # 2 * nnz ops
            axpy(n_vecs, a, x, y)           # n_vecs * 2 * nnz ops
    :param fn:
    :param args:
    :param kwargs:
    :return:
    """
    print("in csr_matvecs")
    n_row, _, n_vecs, *tail = args
    Ap_raw = tail[0]
    nnz = len(Ap_raw)
    add_to_ops_accum(
        n_row * 2 + nnz * (2 + n_vecs * 2)
    )
    Ap, Aj, Ax, Xx, Yx = [
        TList(x, is_host_init=True) for x in tail
    ]
    Ap_last_ptr, Ap_last_ptr_trace_id = Ap.__getitem__(
        0, deps=[get_last_trace_id()]
    )

    for i in range(n_row):
        y_ptr = n_vecs * i
        _, Yx_trace_id = Yx.__getitem__(
            slice(y_ptr, y_ptr+n_vecs),
            deps=[get_last_trace_id()]
        )
        Ap_new_ptr, Ap_new_ptr_trace_id = Ap.__getitem__(
            i+1, deps=[get_last_trace_id()]
        )

        # Aj_jj_sublist = [Ap_last_ptr:Ap_new_ptr]
        # Ax_jj_sublist = Ax[Ap_last_ptr:Ap_new_ptr]

        Aj_jj_sublist, Aj_jj_sublist_trace_id = Aj.__getitem__(
            slice(Ap_last_ptr, Ap_new_ptr),
            deps=[Ap_last_ptr_trace_id, Ap_new_ptr_trace_id]
        )
        Ax_jj_sublist, Ax_jj_sublist_trace_id = Ax.__getitem__(
            slice(Ap_last_ptr, Ap_new_ptr),
            deps=[Ap_last_ptr_trace_id, Ap_new_ptr_trace_id]
        )

        for j, a in zip(Aj_jj_sublist, Ax_jj_sublist):
            x_ptr = n_vecs * j
            # _ = Xx[x_ptr:x_ptr+n_vecs]
            _ = Xx.__getitem__(
                slice(x_ptr, x_ptr+n_vecs),
                deps=[Aj_jj_sublist_trace_id, Ax_jj_sublist_trace_id]
            )

            # We assume that axpy is DRAM bound.
            # Hence once the data is ready, we don't need to simulate the
            # compute.

        Ap_last_ptr = Ap_new_ptr
        Ap_last_ptr_trace_id = Ap_new_ptr_trace_id

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
