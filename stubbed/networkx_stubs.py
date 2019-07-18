import networkx
import itertools
import typing
import random
import sys
import weakref


class TraceElement(typing.NamedTuple):
    memory_space: int
    offset: int
    size: int
    type: str
    iterator_id: int = -1

TraceElement.READ = "R"
TraceElement.WRITE = "W"


class MemorySpace(typing.NamedTuple):
    memory_space: int
    size: int # in number of bytes
    element_size: int # in number of bytes


MemorySpaceCounter = itertools.count()
IteratorCounter = itertools.count()

TraceRegistry: typing.List[typing.Callable[[], TraceElement]] = []
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
    def __init__(self, inner_iter, container: 'TrackedContainer', trace_size=1):
        self.container = container
        self.inner_iter = enumerate(inner_iter)
        self.trace_size = trace_size
        self.iterator_id = next(IteratorCounter)

    def __next__(self):
        count, value = next(self.inner_iter)
        if count % self.trace_size == 0:
            TraceRegistry.append(lambda: self.container.getbase()._replace(
                offset=count*self.container.element_size,
                size=max(self.trace_size*self.container.element_size, self.container.getsize()),
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
    max_element_size_registry: typing.List[typing.Union[typing.Callable[[], int]]]

    # Overhead
    base_size: int = 4

    # feature size
    feature_size: int = 4

    key_to_loc_map: typing.MutableMapping[typing.Hashable, int]

    def __init__(self, *args, is_host_init=False, max_size=None, **kwargs):
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
            self.max_element_size_registry.append(weakref.ref(value, self.max_size_callback))
        else:
            self.max_element_size_registry.append(self.feature_size)

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
            # Issue a dense read for slices
            start = key.start if key.start else 0
            stop = key.stop if key.stop else len(self)
            length = stop - start
            TraceRegistry.append(
                lambda: self.getloc(start)._replace(
                    type=TraceElement.READ,
                    size=self.feature_size * length
                )
            )
        elif not isinstance(val, TrackedContainer):
            # Reading into another container can be bypassed via smart address calculation
            TraceRegistry.append(lambda: self.getloc(key)._replace(type=TraceElement.READ))
        return val

    def __delitem__(self, key):
        TraceRegistry.append(lambda: self.getloc(key)._replace(type="DELETE"))
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
                lambda: MemorySpace(self.memory_space_id, self.getsize(), self.element_size)
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
                self.key_to_loc_map[key] = random.choice(tuple(remaining_choices))
            offset = self.key_to_loc_map[key]
        else:
            if key >= self.max_len:
                raise Exception("This should probably have been a sparse array.")
            offset = key * self.element_size

        return trace._replace(offset=trace.offset + offset, size=self.element_size)


class TrackedList(TrackedContainer, list):
    def __init__(self, *args, is_host_init=False, **kwargs):
        super().__init__(*args, is_host_init=is_host_init, **kwargs)
        for index, value in enumerate(list.__iter__(self)):
            # register all values.
                self[index] = value
        self._head = self[0]

    def __iter__(self):
        return TrackedIterator(list.__iter__(self), self)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
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
        TraceRegistry.append(lambda: self.getloc(item)._replace(type=TraceElement.READ))
        return super().__contains__(item)


def stub():
    networkx.Graph.node_dict_factory = TrackedDict
    networkx.Graph.node_attr_dict_factory = TrackedDict
    networkx.Graph.adjlist_inner_dict_factory = TrackedDict
    networkx.Graph.adjlist_outer_dict_factory = TrackedDict
    networkx.Graph.edge_attr_dict_factory = TrackedDict
