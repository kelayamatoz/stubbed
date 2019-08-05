import numpy as np
import typing
from math import ceil, floor
from .networkx_stubs import TrackedList as TList
from .networkx_stubs import get_last_trace_id, add_to_ops_accum
from .networkx_stubs import TraceDepRegistry as tdr
from .networkx_stubs import TraceRegistry as tr

# Tile size is 4MB, 1M ints
TILE_SIZE = int(2 * 1024 * 1024 / 4)


class TArrayTraceElement:
    """
    A TArray element for book keeping.
    The trace_id_list contains all the trace ids this element depends on.
    TODO: need to find a way to dispatch these functions in one line. One
    proposal is that I could use the inspect library to look at the function
    vars, and then annotate them with my own subroutine.
    """

    def _update_nops_accum(self):
        add_to_ops_accum(
            len(self.value) if isinstance(
                self.value.size, (np.ndarray, TArray)
            ) else 1
        )

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
        """
        If we are iterating on an TArrayTraceElement, we can bulk-load
        the actual array.
        :return:
        """
        return self.value.__iter__()

    def __le__(self, other):
        return self.value <= other

    def __eq__(self, other):
        return self.value == other

    def __gt__(self, other):
        return self.value > other

    def __lt__(self, other):
        return self.value < other

    def __ge__(self, other):
        return self.value >= other

    def __mul__(self, other):
        self._update_nops_accum()
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
        self._update_nops_accum()
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
        self._update_nops_accum()
        return TArrayTraceElement(
            value=self.value / other.value if isinstance(
                other, TArrayTraceElement
            ) else self.value / other,
            trace_id_list=self._merge_trace_id_lists(other)
        )

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __rtruediv__(self, other):
        self._update_nops_accum()
        return TArrayTraceElement(
            value=other.value / self.value if isinstance(
                other, TArrayTraceElement
            ) else other / self.value,
            trace_id_list=self._merge_trace_id_lists(other)
        )

    def __sub__(self, other):
        self._update_nops_accum()
        return TArrayTraceElement(
            value=self.value - other.value if isinstance(
                other, TArrayTraceElement
            ) else self.value - other,
            trace_id_list=self._merge_trace_id_lists(other)
        )

    def __isub__(self, other):
        return self.__sub__(other)

    def __rsub__(self, other):
        self._update_nops_accum()
        return TArrayTraceElement(
            value=other.value - self.value if isinstance(
                other, TArrayTraceElement
            ) else other - self.value,
            trace_id_list=self._merge_trace_id_lists(other)
        )

    def sum(self):
        self._update_nops_accum()
        return TArrayTraceElement(
            value=self.value.sum(),
            trace_id_list=self.trace_id_list
        )

    def dot(self, other):
        self._update_nops_accum()
        return TArrayTraceElement(
            value=self.value.dot(other),
            trace_id_list=self.trace_id_list
        )


class NpSet(set):
    """
    Numpy ndarray is not hashable. However a tuple of a flattened numpy
    array is.
    This function wraps a numpy ndarray with a tuple wrapper.
    """

    def add(self, element) -> None:
        if isinstance(element, TArrayTraceElement):
            element = element.value
        elif isinstance(element, TArray):
            element = TArray.view(np.ndarray)
        super().add(tuple(element.flatten()))

    def __contains__(self, item):
        if isinstance(item, TArray):
            item = item.view(np.ndarray)
        return super(NpSet, self).__contains__(tuple(item.flatten()))


ProfiledArraySet: typing.Set = NpSet()


def _profiled_check(items: typing.List):
    """
    Disable profiled check for now as this seems to incur unnecessary tests...
    :param items:
    :return:
    """
    # for k in items:
    #     if k not in ProfiledArraySet:
    #         ProfiledArraySet.add(k)


class TArray(np.ndarray):
    """
    We don't need an initializer here; this class is a thin wrapper
    to hijack the runtime for @. Otherwise monkey patching would fail
    due to @ being a builtin.

    TODO: there might be a better way to call subroutines defined in the
    super class. Should be a way to just decorate all of these.

    TODO: we may need to also subclass TList. For now I'm doing the
    bookkeeping within the _init_book_keeping function...
    """
    def _init_book_keeping(self) -> None:
        """
        TODO: remove the is_host_init. Do it at init time for TList.
        :return:
        """
        if '_tlist' not in vars(self):
            self._tlist = TList([0] * self.size, is_host_init=True)
            self._tlist._is_host_init = False
            self._is_iter = False
            self._iter_id = -1

    def _update_blas_3_ops(self, a, b):
        if isinstance(
            a, (TArray, np.ndarray)
        ) and isinstance(
            b, (TArray, np.ndarray)
        ):
            a_n_rows, a_n_cols = a.shape
            _, b_n_cols = b.shape
            add_to_ops_accum(
                a_n_rows * a_n_cols * b_n_cols * 2
            )

    def _update_blas_1_map_ops(self, other):
        if isinstance(other, (TArray, np.ndarray)):
            add_to_ops_accum(len(other))

    def _update_blas_1_map_reduce_ops(self, other):
        if isinstance(other, (TArray, np.ndarray)):
            add_to_ops_accum(len(other) * 2)

    def _blas_3_stub(self, *args):
        global ProfiledArraySet
        other = args[0]
        self._init_book_keeping()
        self._update_blas_3_ops(self, other)

        if isinstance(other, (TArray, np.ndarray)) and \
                other not in ProfiledArraySet:
            if other.size > TILE_SIZE or self.size > TILE_SIZE:
                # We need to do tiling here for both reads and writes
                def _get_row_col_steps(a_shape: typing.Tuple,
                                       b_shape: typing.Tuple):
                    """
                    Once getting a shape, this function should return the
                    number of steps required to step through both dimensions
                    based on the given tile size.
                    :param shape:
                    :return: m_steps, k_steps, n_steps, tile_size on the
                    reduce dim
                    """
                    m, _k = a_shape
                    __k, n = b_shape
                    if _k != __k:
                        print("Warning: inner product dimensions don't match!")

                    k = _k

                    _step_size = ceil(TILE_SIZE / float(k))
                    _m_steps = ceil(m / _step_size)
                    _n_steps = ceil(n / _step_size)
                    _k_steps = ceil(
                        float(k) / TILE_SIZE
                    ) if _step_size == 1 else 1
                    _m_residual_step_size = m % _step_size
                    return _m_steps, _k_steps, _n_steps, _step_size, \
                        m % _step_size, n % _step_size

                m_steps, k_steps, n_steps, step_size, \
                m_residual_step_size, n_residual_step_size \
                    = _get_row_col_steps(self.shape, other.shape)

                # Each tiled multiplication is within a map scope; hence
                # there's no dependency here.

                par_dep = get_last_trace_id()
                for i_m in range(m_steps):
                    for i_n in range(n_steps):

                        # Issue all the reads for doing this inner matmul
                        for i_k in range(k_steps):
                            # Load two read tiles of size TILE_SIZE
                            def _get_read_trace_id():
                                _tmp = TList(
                                    [0] * TILE_SIZE, is_host_init=True
                                )
                                _, trace_id = _tmp.__getitem__(
                                    slice(None, None, None),
                                    deps=[par_dep]
                                )

                                return trace_id

                            read_deps = [
                                _get_read_trace_id() for i in range(2)
                            ]

                        # Issue tiled writes
                        m_curr_step_size = step_size \
                            if i_m < m_steps - 1 else m_residual_step_size
                        n_curr_step_size = step_size \
                            if i_n < n_steps - 1 else n_residual_step_size
                        write_size = m_curr_step_size * n_curr_step_size
                        write_steps = ceil(write_size / float(TILE_SIZE))

                        for i_w in range(write_steps):
                            _tmp = TList(
                                [0] * TILE_SIZE,
                                is_host_init=True
                            )
                            _tmp.__setitem__(
                                slice(None, None, None),
                                slice(None, None, None),
                                deps=read_deps
                            )

            else:
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

        add_to_ops_accum(self.size)

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

    def __iter__(self, *args, **kwargs):
        self._init_book_keeping()
        if not self._is_iter:
            self._is_iter = True
            _, self._iter_id = self._tlist.__getitem__(
                slice(None, None, None),
                deps=[get_last_trace_id()]
            )
        return super().__iter__(*args, **kwargs)

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

        self._init_book_keeping()
        k, v = args

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
                if self._is_iter:
                    # print("getting element from iter, element = {}".format(v))
                    trace_id = self._iter_id
                else:
                    # print("getting single element, element = {}".format(v))
                    _, trace_id = self._tlist.__getitem__(v, deps=[])
                result = TArrayTraceElement(
                    super(TArray, self).__getitem__(v),
                    trace_id_list=[trace_id]
                )
            elif isinstance(v, slice):
                # print("getting slice, slice = {}".format(v))
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
                # print("getting a TATElement from TArray, value = {}".format(v))
                if self._is_iter:
                    trace_id = self._iter_id
                else:
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
                # print(
                    # "initing in the view function..., value = {}".format(v)
                # )
                result = super(TArray, self).__getitem__(*args, **kwargs)
        else:
            print("Wrong format {}...".format(v))
            result = super(TArray, self).__getitem__(*args, **kwargs)

        return result
        
    def dot(self, *args, **kwargs):
        # print("__dot__")
        self._blas_1_stub(*args)
        self._update_blas_1_map_ops(self) # Add one more here for reduction
        return super(TArray, self).dot(*args, **kwargs)

    def __add__(self, *args, **kwargs):
        # print("add")
        self._blas_1_stub(*args)
        return super(TArray, self).__add__(*args, **kwargs)

    def __iadd__(self, *args, **kwargs):
        # print("iadd")
        self._blas_1_stub(*args)
        return super(TArray, self).__iadd__(*args, **kwargs)

    def __radd__(self, *args, **kwargs):
        # print("radd")
        self._blas_1_stub(*args)
        return super(TArray, self).__radd__(*args, **kwargs)

    def sum(self, *args, **kwargs):
        # print("sum")
        # stubbing is done already...
        # self._blas_1_stub(*args)

        # ops counting is not
        self._update_blas_1_map_ops(self)
        return super(TArray, self).sum(*args, **kwargs)

    def __matmul__(self, *args, **kwargs):
        self._blas_3_stub(*args)
        return super(TArray, self).__matmul__(*args, **kwargs)

    def __rmatmul__(self, *args, **kwargs):
        # print("__rmatmul__")
        self._blas_1_stub(*args)
        return super(TArray, self).__rmatmul__(*args, **kwargs)

    def __mul__(self, *args, **kwargs):
        # print("__mul__")
        self._blas_1_stub(*args)
        return super(TArray, self).__mul__(*args, **kwargs)

    def __rmul__(self, *args, **kwargs):
        # print("__rmul__")
        self._blas_1_stub(*args)
        return super(TArray, self).__rmul__(*args, **kwargs)

    def __imul__(self, *args, **kwargs):
        # print("__imul__")
        self._blas_1_stub(*args)
        return super(TArray, self).__imul__(*args, **kwargs)


def array(fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    return result.view(TArray)

