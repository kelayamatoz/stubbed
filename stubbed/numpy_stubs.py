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

    def _blas_3_stub(self, *args):
        global ProfiledArraySet
        other = args[0]
        if isinstance(other, (TArray, np.ndarray)):
            for k in [self, other]:
                if k not in ProfiledArraySet:
                    tmp = TList(
                        [0] * k.size, is_host_init=True
                    )
                    _ = tmp.__getitem__(
                        slice(None, None, None),
                        deps=[get_last_trace_id()]
                    )

            _profiled_check([self.T, other.T])

    def _blas_1_stub(self, *args):
        global ProfiledArraySet
        if args:
            other = args[0]
            if isinstance(other, (TArray, np.ndarray)):
                for k in [self, other]:
                    if k not in ProfiledArraySet:
                        tmp = TList(
                            [0] * k.size, is_host_init=True
                        )
                        _ = tmp.__getitem__(
                            slice(None, None, None),
                            deps=[get_last_trace_id()]
                        )

                _profiled_check([self.T, other.T])
        else:
           # a postfix function without args.
            if self in ProfiledArraySet:
                tmp = TList(
                    [0] * self.size, is_host_init=True
                )
                _ = tmp.__getitem__(
                    slice(None, None, None),
                    deps=[get_last_trace_id()]
                )

                _profiled_check([self.T])

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


