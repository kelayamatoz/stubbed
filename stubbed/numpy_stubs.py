from typing import Iterable, Tuple, Any

import numpy as np
import typing
from .networkx_stubs import TrackedList as TList
from .networkx_stubs import get_last_trace_id
from .networkx_stubs import TraceDepRegistry as tdr
from .networkx_stubs import TraceRegistry as tr


class TArrayTraceElement:
    """
    A TArray element for book keeping.
    The trace_id_list contains all the trace ids this element depends on.
    TODO: need to find a way to dispatch these functions in one line. One
    proposal is that I could use the inspect library to look at the function
    vars, and then annotate them with my own subroutine.
    """

    def _merge_trace_id_lists(self, b):
        default_trace_list = self.trace_id_list
        if isinstance(b, TArrayTraceElement):
            default_trace_list += b.trace_id_list
        return default_trace_list

    def _merge_value(self, b, fn: typing.Callable):
        default_value = self.value
        if isinstance(b, TArrayTraceElement):
            default_value = fn(self.value, b)
        return default_value

    def __init__(self, value, trace_id_list: typing.List) -> None:
        self.value = value
        self.trace_id_list = trace_id_list

    def __iter__(self):
        return self.value.__iter__()
        
    def __mul__(self, other):
        return TArrayTraceElement(
            value=self.value * other.value if isinstance(
                other, TArrayTraceElement
            ) else self.value * other,
            trace_id_list=self._merge_trace_id_lists(other)
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return TArrayTraceElement(
            value=self.value + other.value if isinstance(
                other, TArrayTraceElement
            ) else self.value + other,
            trace_id_list=self._merge_trace_id_lists(other)
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        return TArrayTraceElement(
            value=self.value / other.value if isinstance(
                other, TArrayTraceElement
            ) else self.value / other,
            trace_id_list=self._merge_trace_id_lists(other)
        )

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __rtruediv__(self, other):
        return TArrayTraceElement(
            value=other.value / self.value if isinstance(
                other, TArrayTraceElement
            ) else other / self.value,
            trace_id_list=self._merge_trace_id_lists(other)
        )

    def __sub__(self, other):
        return TArrayTraceElement(
            value=self.value - other.value if isinstance(
                other, TArrayTraceElement
            ) else self.value - other,
            trace_id_list=self._merge_trace_id_lists(other)
        )

    def __isub__(self, other):
        return self.__sub__(other)

    def __rsub__(self, other):
        return TArrayTraceElement(
            value=other.value - self.value if isinstance(
                other, TArrayTraceElement
            ) else other - self.value,
            trace_id_list=self._merge_trace_id_lists(other)
        )


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
        TODO: remove the is_host_init. Do it at init time for TList.
        :return:
        """
        if '_tlist' not in vars(self):
            self._tlist = TList([0] * self.size, is_host_init=True)
            self._tlist._is_host_init = False
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

    @staticmethod
    def _extract_val_trace_id_list(a):
        if isinstance(a, TArrayTraceElement):
            return a.value, a.trace_id_list
        else:
            return a, []

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

        print("in tarray setitem, ", *args)
        self._init_book_keeping()
        k, v = args

        # TODO: replace the first line with a decorator function...
        key, key_trace_list = self._extract_val_trace_id_list(k)
        val, val_trace_list = self._extract_val_trace_id_list(v)

        # TODO: for now we assume that writes don't generate dependency.
        # This is definitely wrong but seems to be good enough for many of
        # our apps.
        # TODO: in order to get the write through, I'm flipping the
        # host_init attribute of self. Not a neat way to do and I will need
        # to fix it in the future release. The current patch adds a barrier
        # around writes.
        self._tlist.__setitem__(
            key, val, deps=key_trace_list + val_trace_list
        )
        super().__setitem__(key, val)

    def __getitem__(self, *args, **kwargs):
        # Seems that when calling the view function, the *args are in fact
        # tuples. We only want to init getitem and setitem when the *args
        # are either int or slices.
        # TODO: the instances checks are here to guard against weird
        # accesses thrown by the view function. Right now I'm patching it
        # with these guards. Later we need to understand how these view
        # functions work... The debugger also cannot catch these issues either.
        v = args[0]
        if not isinstance(v, tuple):
            self._init_book_keeping()
            if isinstance(
                    v, (int, np.int, np.int64)
            ) and v >= 0:
                print("getting single element, element = {}".format(v))
                _, trace_id = self._tlist.__getitem__(
                    v, deps=[]
                )
                result = TArrayTraceElement(
                    super(TArray, self).__getitem__(v),
                    trace_id_list=[trace_id]
                )
            elif isinstance(v, slice):
                print("getting slice, slice = {}".format(v))
                slice_start_val, slice_start_trace_id_list = \
                    self._extract_val_trace_id_list(v.start)
                slice_stop_val, slice_stop_trace_id_list = \
                    self._extract_val_trace_id_list(v.stop)
                v = slice(slice_start_val, slice_stop_val)
                _, trace_id = self._tlist.__getitem__(
                    v,
                    deps=slice_start_trace_id_list + slice_stop_trace_id_list
                )
                result = TArrayTraceElement(
                    super(TArray, self).__getitem__(v),
                    trace_id_list=[trace_id]
                )

            elif isinstance(v, TArrayTraceElement):
                print("getting a TATElement from TArray, value = {}".format(v))
                _, trace_id = self._tlist.__getitem__(
                    v.value, deps=v.trace_id_list
                )
                result = TArrayTraceElement(
                    super(TArray, self).__getitem__(v.value),
                    trace_id_list=[trace_id]
                )
            else:
                print("Wrong format {}...".format(v))
                result = super(TArray, self).__getitem__(*args, **kwargs)
        elif isinstance(v, tuple):
                print(
                    "initing in the view function..., value = {}".format(v)
                )
                result = super(TArray, self).__getitem__(*args, **kwargs)
        else:
            print("Wrong format {}...".format(v))
            result = super(TArray, self).__getitem__(*args, **kwargs)

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

