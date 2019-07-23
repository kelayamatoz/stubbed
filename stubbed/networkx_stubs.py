import networkx
import itertools
import typing
import random
import sys
import weakref


INVALID_ID: int = -1


class TraceListElement(typing.NamedTuple):
    list_element: object
    trace_id: int = 0


class TraceElement(typing.NamedTuple):
    memory_space: int
    offset: int
    size: int
    type: str
    iterator_id: int = -1


class TraceElementDeps(typing.NamedTuple):
    """
    New Format (per Alex's request):
    [Deps on], Number, addr, size, [out edges]
    where:
        Deps on are accesses that the access depends on
        Out edges are accesses that depend on this access
        Number is a unique number per access
    """
    deps_on: typing.List[int] = []
    out_edges: typing.List[int] = []


class TraceElementRange(range):

    def __init__(self) -> None:
        super().__init__()


TraceElement.READ = "R"
TraceElement.WRITE = "W"


class MemorySpace(typing.NamedTuple):
    memory_space: int
    size: int # in number of bytes
    element_size: int # in number of bytes


MemorySpaceCounter = itertools.count()
IteratorCounter = itertools.count()
TraceCounter = itertools.count()

TraceDepsRegistry: typing.List[TraceElementDeps] = []


class RegistryList(list):

    def append(self, item, deps_ons: typing.List = []) -> None:
        """
        Besides regular update, this function also updates the out_edges
        of previous accesses that this access depends on.
        :param item:
        :param deps_ons:
        :return:
        """
        curr_id = len(TraceDepsRegistry)
        TraceDepsRegistry.append(TraceElementDeps(
            deps_on=deps_ons
        ))
        super().append(item)

        for trace_id in deps_ons:
            TraceDepsRegistry[trace_id].out_edges.append(curr_id)


TraceRegistry = RegistryList()
MemorySpaceRegistry: typing.List[typing.Callable[[], MemorySpace]] = []


def dumptrace(tracefile=sys.stdout):
    print("Traces:", len(TraceRegistry))
    for trace in TraceRegistry[:]:
        try:
            print(*trace(), file=tracefile, sep="\t")
        except TypeError:
            # Networkx has a bad habit of catching TypeErrors
            pass


def dumpmem(spacefile=sys.stdout):
    print("Spaces:", len(MemorySpaceRegistry))
    for space in MemorySpaceRegistry[:]:
        print(*space(), file=spacefile, sep="\t")


def reset():
    TraceRegistry.clear()


class TrackedIterator(typing.Iterator):
    def __init__(
        self, inner_iter, container: 'TrackedContainer', trace_size=1
    ):
        self.container = container
        self.inner_iter = enumerate(inner_iter)
        self.trace_size = trace_size
        self.iterator_id = next(IteratorCounter)

    def __next__(self):
        count, value = next(self.inner_iter)
        if count % self.trace_size == 0:
            TraceRegistry.append(lambda: self.container.getbase()._replace(
                offset=count * self.container.element_size,
                size=max(
                    self.trace_size * self.container.element_size,
                    self.container.getsize()
                ),
                type=TraceElement.READ,
                iterator_id=self.iterator_id
            ))
        return value


class TrackedContainer:
    def __init_subclass__(cls, is_sparse=False):
        cls.is_sparse = is_sparse

    parent: 'TrackedContainer'
    key: typing.Hashable
    max_len: int
    max_element_size_registry: typing.List[
        typing.Union[typing.Callable[[], int]]
    ]

    # Overhead
    base_size: int = 4

    # feature size
    feature_size: int = 4

    key_to_loc_map: typing.MutableMapping[typing.Hashable, int]

    def __init__(self, *args, is_host_init=False, max_size=None, **kwargs):
        """

        :param args:
        :param is_host_init: shows if this allocation is an init allocation.
        :param max_size:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.parent = None
        self.key = None
        self._memory_space_id = None
        self._is_host_init = is_host_init
        self.max_len = len(self)
        self.max_element_size_registry = []
        self._max_size = max_size
        self.key_to_loc_map = {}

    def __setitem__(self, key, value):
        if isinstance(value, TrackedContainer):
            value.parent = self
            value.key = key
            self.max_element_size_registry.append(
                weakref.ref(value, self.max_size_callback)
            )
        else:
            self.max_element_size_registry.append(
                self.feature_size
            )

        # If is not a host init, we should assume that the data is prepared
        # in the DRAM and the accelerator can just read from it.
        if not self._is_host_init:
            TraceRegistry.append(
                lambda: self.getloc(key)._replace(type=TraceElement.WRITE)
            )

        super().__setitem__(key, value)

        self.max_len = max(self.max_len, len(self))

    def __getitem__(self, key):
        val = super().__getitem__(key)

        if isinstance(key, slice):
            def get_slice_dep(_x: slice):
                """
                Check for types of start and stop.
                If is native type of python, we can just go ahead and issue
                the dense loads without dependencies.
                Otherwise, if is TraceListElement type, it means that the
                range indices were from previous reads and this read
                depends on the previous indices. In this case, we need
                to update both the registered trace element and the current one
                with dependency information.
                :param _x: a slice that may contain either python builtins
                or trace list element (i.e. start and stop indices
                depend on previous reads)
                :return:
                """
                def extract_field(
                        slice_field, default_val: int = 0
                ) -> (int, int, int, int) :
                    if isinstance(slice_field, TraceListElement):
                        raw_field = slice_field.list_element
                        field = raw_field if raw_field else default_val
                        dep_id = slice_field.trace_id
                    else:
                        field = slice_field if slice_field else default_val
                        dep_id = INVALID_ID

                    return field, dep_id

                st, st_dep_id = extract_field(_x, 0)
                ed, ed_dep_id = extract_field(_x, len(self))
                dep_ids = list(
                    filter(
                        lambda x: x != INVALID_ID, [st_dep_id, ed_dep_id])
                )

                return st, ed, dep_ids

            start, stop, dep_ids = get_slice_dep(key)
            length = stop - start

            TraceRegistry.append(
                lambda: self.getloc(start)._replace(
                    type=TraceElement.READ,
                    size=self.feature_size * length
                ),
                deps_ons=dep_ids
            )

        elif not isinstance(val, TrackedContainer):
            # Reading into another container can be
            # bypassed via smart address calculation
            TraceRegistry.append(
                lambda: self.getloc(key)._replace(type=TraceElement.READ)
            )

        return val

    def __delitem__(self, key):
        TraceRegistry.append(
            lambda: self.getloc(key)._replace(type="DELETE")
        )
        super().__delitem__(self, key)

    def __iter__(self):
        return TrackedIterator(super(TrackedContainer, self).__iter__(), self)

    @property
    def memory_space_id(self):
        if self.parent:
            return self.parent.memory_space_id
        if self._memory_space_id is None:
            self._memory_space_id = next(MemorySpaceCounter)
            MemorySpaceRegistry.append(
                lambda: MemorySpace(
                    self.memory_space_id, self.getsize(), self.element_size
                )
            )
        return self._memory_space_id

    def getsize(self) -> int:
        return self.element_size * self.max_len + self.base_size

    @property
    def element_size(self) -> int:
        if self._max_size is not None:
            return self._max_size
        max_size = 0
        for element in self.max_element_size_registry:
            if isinstance(element, int):
                max_size = max(element, max_size)
            if isinstance(element, typing.Callable):
                fetched = element()
                if fetched is not None:
                    max_size = max(fetched.getsize(), max_size)

        self._max_size = max_size
        return self._max_size

    def max_size_callback(self, subcontainer):
        self.max_element_size_registry.append(subcontainer.getsize())

    def getbase(self) -> TraceElement:
        if self.parent:
            return self.parent.getloc(self.key)
        return TraceElement(self.memory_space_id, 0, self.getsize(), "")

    def getloc(self, key) -> TraceElement:
        trace = self.getbase()
        if self.is_sparse:
            if key not in self.key_to_loc_map:
                used_offsets = set(self.key_to_loc_map.values())
                remaining_choices = set(range(self.max_len)) - used_offsets
                self.key_to_loc_map[key] = \
                    random.choice(tuple(remaining_choices))
            offset = self.key_to_loc_map[key]
        else:
            if key >= self.max_len:
                raise Exception(
                    "This should probably have been a sparse array."
                )
            offset = key * self.element_size

        return trace._replace(
            offset=trace.offset + offset, size=self.element_size
        )


class TrackedList(TrackedContainer, list):
    def __init__(self, *args, is_host_init=False, **kwargs):
        super().__init__(*args, is_host_init=is_host_init, **kwargs)
        for index, value in enumerate(list.__iter__(self)):
            # register all values.
                self[index] = value
        self._head = self[0]

    def __iter__(self):
        return TrackedIterator(list.__iter__(self), self)

    def __getitem__(self, item):
        return TraceListElement(
            list_element=super(TrackedList, self).__getitem__(item),
            trace_id=len(TraceDepsRegistry) - 1
        )

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            # For now, we only support burst reads.
            # Burst reads are more common for the sparse apps.
            raise NotImplementedError
        else:
            return super(TrackedList, self).__setitem__(key, value)

    @property
    def head(self):
        return self._head


class TrackedDict(TrackedContainer, dict, is_sparse=True):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in super().items():
            # register all values.
            self[key] = value

    def update(self, *args, **kwargs):
        for other in args:
            for k, v in other.items():
                self[k] = v

        for k, v in kwargs:
            self[k] = v

    def keys(self):
        return TrackedIterator(super().keys(), self)

    def values(self):
        return TrackedIterator(super().values(), self)

    def items(self):
        return TrackedIterator(super().items(), self)

    def __contains__(self, item):
        TraceRegistry.append(
            lambda: self.getloc(item)._replace(type=TraceElement.READ)
        )
        return super().__contains__(item)


def stub():
    networkx.Graph.node_dict_factory = TrackedDict
    networkx.Graph.node_attr_dict_factory = TrackedDict
    networkx.Graph.adjlist_inner_dict_factory = TrackedDict
    networkx.Graph.adjlist_outer_dict_factory = TrackedDict
    networkx.Graph.edge_attr_dict_factory = TrackedDict
