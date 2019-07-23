import numpy
from numpy import array_equal
from .networkx_stubs import TrackedList as TList


class LArray(list):

    def __contains__(self, item):
        for x in self:
            if array_equal(x, item):
                return True
        return False


ProfiledArrayList = LArray()


class TArray(numpy.ndarray):
    """
    We don't need an initializer here; this class is a thin wrapper
    to hijack the runtime for @. Otherwise monkey patching would fail
    due to @ being a builtin.
    """

    def __matmul__(self, *args, **kwargs):
        global ProfiledArrayList
        other = args[0]
        if isinstance(other, (TArray, numpy.ndarray)):
            for k in [self, other]:
                if k not in ProfiledArrayList:
                    n_rows, n_cols = k.shape
                    tmp = TList(
                        [0] * n_rows * n_cols, is_host_init=True
                    )
                    _ = tmp[:]

            for k in [self, self.T, other, other.T]:
                if k not in ProfiledArrayList:
                    ProfiledArrayList.append(k)

        return super(TArray, self).__matmul__(*args, **kwargs)


def array(fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    return result.view(TArray)


