from .networkx_stubs import TrackedList as TList
from functools import partial
from scipy.sparse import _sparsetools


def csr_matmat_pass1(fn, *args, **kwargs):
    print("In csr matmat pass1, kwargs = ", kwargs)
    return fn(*args, **kwargs)


def csr_matmat_pass2(fn, *args, **kwargs):
    print("In csr matmat pass2, kwargs = ", kwargs)
    return fn(*args, **kwargs)


def csr_matvec(fn, *args, **kwargs):
    n_row, n_col, Ap, Aj, Ax, Xx, Yx = args
    Ap = TList(Ap)
    Aj = TList(Aj)
    Ax = TList(Ax)
    Xx = TList(Xx)
    Yx = TList(Yx)

    for i in range(n_row):
        sum = Yx[i]
        for jj in range(Ap[i], Ap[i+1]):
            sum += Ax[jj] * Xx[Aj[jj]]
        Yx[i] = sum

    return fn(*args, **kwargs)


def _axpy(n, a, x_ptr, Xx, y_ptr, Yx):
    for i in range(n):
        Yx[y_ptr + i] += Xx[x_ptr + i] * a


def csr_matvecs(fn, *args, **kwargs):
    # TODO: update this with auto instance type conversion
    n_row, n_col, n_vecs, Ap, Aj, Ax, Xx, Yx = args
    Ap = TList(Ap)
    Aj = TList(Aj)
    Ax = TList(Ax)
    Xx = TList(Xx)
    Yx = TList(Yx)
    for i in range(n_row):
        y_ptr = n_vecs * i
        for jj in range(Ap[i], Ap[i+1]):
            j = Aj[jj]
            a = Ax[jj]
            x_ptr = n_vecs * j
            _axpy(n_vecs, a, x_ptr, Xx, y_ptr, Yx)

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
