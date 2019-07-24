import numpy as np
import typing
from .networkx_stubs import TrackedList as TList
from .networkx_stubs import get_last_trace_id
from .networkx_stubs import TraceDepRegistry as tdr
from .networkx_stubs import TraceRegistry as tr


class TArrayTraceElement(typing.NamedTuple, int, float, np.ndarray):
    value: object
    trace_id: int

    def __mul__(self, other):
        return super(TArrayTraceElement, self).__mul__(self.value, other)

    def __rmul__(self, other):
        # return super(TArrayTraceElement, self).__rmul__(self.value, other)
        return TArrayTraceElement(
            value=other*self.value,
            trace_id=self.trace_id
        )

    def __imul__(self, other):
        return super(TArrayTraceElement, self).__imul__(self.value, other)

    def __add__(self, other):
        return super(TArrayTraceElement, self).__add__(self.value, other)

    def __radd__(self, other):
        return super(TArrayTraceElement, self).__radd__(self.value, other)

    def __iadd__(self, other):
        return super(TArrayTraceElement, self).__iadd__(self.value, other)

    def __truediv__(self, other):
        return super(TArrayTraceElement, self).__truediv__(self.value, other)

    def __itruediv__(self, other):
        return super(TArrayTraceElement, self).__itruediv__(self.value, other)

    def __rtruediv__(self, other):
        return super(TArrayTraceElement, self).__rtruediv__(self.value, other)


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
        if '_tlist' not in vars(self):
            self._tlist = TList([0] * self.size, is_host_init=True)
        # if not self._tdict:
        #     self._tdict = {}

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
        print("in tarray setitem, ", *args)
        super().__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        # Seems that when calling the view function, the *args are in fact
        # tuples. We only want to init getitem and setitem when the *args
        # are either int or slices.
        v = args[0]
        if not isinstance(v, tuple):
            self._init_book_keeping()
            if isinstance(v, (int, np.int, np.int64, slice)):
                print("getting single element, element = {}".format(v))
                _, trace_id = self._tlist.__getitem__(
                    v, deps=[get_last_trace_id()]
                )
                result = TArrayTraceElement(
                    super(TArray, self).__getitem__(v),
                    trace_id=trace_id
                )
            else:
                print("Wrong format {}...".format(v))
                result = super(TArray, self).__getitem__(*args, **kwargs)
        elif isinstance(v, tuple):
            if isinstance(v, TArrayTraceElement):
                self._init_book_keeping()
                print("getting a TATElement from TArray, value = {}".format(v))
                _, trace_id = self._tlist.__getitem__(
                    v.value, deps=[v.trace_id]
                )
                result = TArrayTraceElement(
                    super(TArray, self).__getitem__(v.value),
                    trace_id=trace_id
                )
            else:
                print(
                    "initing in the view function..., value = {}".format(v)
                )
                result = super(TArray, self).__getitem__(*args, **kwargs)
        else:
            print("Wrong format {}...".format(v))
            result = super(TArray, self).__getitem__(*args, **kwargs)

        # key = args[0]
        # if isinstance(key, tuple):
        #     self._init_book_keeping()
        #     dep_trace_id = self._tdict[key] if key in self._tdict else \
        #         get_last_trace_id()
        #     item, item_trace_id = self._tlist.__getitem__(
        #         key, deps=[dep_trace_id]
        #     )
        #     self._tdict[item] = item_trace_id

        return result
        
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

