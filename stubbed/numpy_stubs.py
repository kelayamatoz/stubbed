import numpy as np
import typing
from .networkx_stubs import TrackedList as TList
from .networkx_stubs import get_last_trace_id
from .networkx_stubs import TraceDepRegistry as tdr
from .networkx_stubs import TraceRegistry as tr


class NpSet(set):
    """
    Numpy ndarray is not hashable. However a tuple of a flattened numpy
    array is.
    This function wraps a numpy ndarray with a tuple wrapper.
    """

    def add(self, element: np.ndarray) -> None:
        super().add(tuple(element.flatten()))

    def __contains__(self, item):
        if isinstance(item, TArray):
            item = item.view(np.ndarray)
        return super(NpSet, self).__contains__(tuple(item.flatten()))


ProfiledArraySet: typing.Set = NpSet()


def _profiled_check(items: typing.List):
    for k in items:
        if k not in ProfiledArraySet:
            ProfiledArraySet.add(k)


class TArray(np.ndarray):
    """
    We don't need an initializer here; this class is a thin wrapper
    to hijack the runtime for @. Otherwise monkey patching would fail
    due to @ being a builtin.

    TODO: there might be a better way to call subroutines defined in the
    super class. Should be a way to just decorate all of these.
    """

    def _init_book_keeping(self) -> None:
        """
        This function initializes the memory allocation since TArray doesn't
        initialize itself via __init__.
        self._tlist stores the tlist structure.
        self._tdict is a dictionary to store WA_ dependencies.
        :return:
        """
        if not self._tlist:
            self._tlist = TList([0] * self.size)
        if not self._tdict:
            self._tdict = {}

    def _load_tlist(self) -> None:
        """
        Simulate a load of all the elements stored in the current TArray.
        :return:
        """
        _ = self._tlist.__getitem__(slice(None, None, None),
                                    deps=[get_last_trace_id()])

    def _blas_3_stub(self, *args):
        global ProfiledArraySet
        other = args[0]
        self._init_book_keeping()

        if isinstance(other, (TArray, np.ndarray)) and \
                other not in ProfiledArraySet:
            tmp = TList(
                [0] * other.size, is_host_init=True
            )
            _ = tmp.__getitem__(
                slice(None, None, None),
                deps=[get_last_trace_id()]
            )

            _profiled_check([self.T, other.T])

    def _blas_1_stub(self, *args):
        global ProfiledArraySet
        self._init_book_keeping()
        if args:
            other = args[0]
            if isinstance(other, (TArray, np.ndarray)) and \
                    other not in ProfiledArraySet:
                tmp = TList(
                    [0] * other.size, is_host_init=True
                )
                _ = tmp.__getitem__(
                    slice(None, None, None),
                    deps=[get_last_trace_id()]
                )

                _profiled_check([self.T, other.T])

    def __setitem__(self, *args, **kwargs):
        """
        This function is to capture dependencies. For a WA_, it would take
        the form of:
        A = foo.view(TArray)
        A[idx, idx_trace_id_0] = c, c_trace_id # Write
        _, _ = A[idx, idx_trace_id_0] # Read
        or
        A[idx, idx_trace_id_0] = _, _ # Write
        :param args:
        :param kwargs:
        :return:
        """
        # TODO: replace the first line with a decorator function...
        self._init_book_keeping()
        key = args[0]
        value = args[1]
        dep_trace_id = self._tdict[key] if key in self._tdict else \
           get_last_trace_id()
        _, trace_id = self._tlist.__setitem__(key, value, deps=[dep_trace_id])
        self._tdict[key] = trace_id
        super().__setitem__(key, value)

    def __getitem__(self, *args, **kwargs):
        self._init_book_keeping()
        key = args[0]
        dep_trace_id = self._tdict[key] if key in self._dict else \
            get_last_trace_id()
        item, item_trace_id = self._tlist.__getitem__(
            key, deps=[dep_trace_id]
        )
        self._tdict[item] = item_trace_id
        return super().__getitem__(*args, **kwargs)

    def dot(self, *args, **kwargs):
        print("__dot__")
        self._blas_1_stub(*args)
        return super(TArray, self).dot(*args, **kwargs)

    def __add__(self, *args, **kwargs):
        print("add")
        self._blas_1_stub(*args)
        return super(TArray, self).__add__(*args, **kwargs)

    def __iadd__(self, *args, **kwargs):
        print("iadd")
        self._blas_1_stub(*args)
        return super(TArray, self).__iadd__(*args, **kwargs)

    def __radd__(self, *args, **kwargs):
        print("radd")
        self._blas_1_stub(*args)
        return super(TArray, self).__radd__(*args, **kwargs)

    def sum(self, *args, **kwargs):
        print("sum")
        self._blas_1_stub(*args)
        return super(TArray, self).sum(*args, **kwargs)

    def __matmul__(self, *args, **kwargs):
        self._blas_3_stub(*args)
        return super(TArray, self).__matmul__(*args, **kwargs)

    def __rmatmul__(self, *args, **kwargs):
        print("__rmatmul__")
        self._blas_1_stub(*args)
        return super(TArray, self).__rmatmul__(*args, **kwargs)

    def __mul__(self, *args, **kwargs):
        print("__mul__")
        self._blas_1_stub(*args)
        return super(TArray, self).__mul__(*args, **kwargs)

    def __rmul__(self, *args, **kwargs):
        print("__rmul__")
        self._blas_1_stub(*args)
        return super(TArray, self).__rmul__(*args, **kwargs)

    def __imul__(self, *args, **kwargs):
        print("__imul__")
        self._blas_1_stub(*args)
        return super(TArray, self).__imul__(*args, **kwargs)


def array(fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    return result.view(TArray)


