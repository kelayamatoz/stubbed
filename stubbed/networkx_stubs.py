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


class TraceDepElement(typing.NamedTuple):
    dep_on: typing.List[int]
    edges_out: typing.List[int]


TraceElement.READ = "R"
TraceElement.WRITE = "W"


class MemorySpace(typing.NamedTuple):
    memory_space: int
    size: int  # in number of bytes
    element_size: int  # in number of bytes


TraceDepRegistry: typing.List[TraceDepElement] = []


class RegistryList(list):

    def append(
            self, object: typing.Callable[[], TraceElement],
            deps: typing.List[int] = None,
            out_edges: typing.List[int] = None
    ) -> typing.Optional[int]:
        curr_id = len(TraceRegistry)
        deps = list(filter(lambda x: x >= 0, deps)) if deps else []
        out_edges = out_edges if out_edges else []
        TraceDepRegistry.append(TraceDepElement(deps, out_edges))
        for dep_id in deps:
            if dep_id >= 0: # Guard against the first access
                TraceDepRegistry[dep_id].edges_out.append(curr_id)
        super().append(object)

        return curr_id


MemorySpaceCounter = itertools.count()
IteratorCounter = itertools.count()

# TraceRegistry: typing.List[typing.Callable[[], TraceElement]] = []
TraceRegistry = RegistryList()
MemorySpaceRegistry: typing.List[typing.Callable[[], MemorySpace]] = []
OpsAccum: int = 0


def add_to_ops_accum(nops: int):
    global OpsAccum
    OpsAccum += nops

def get_ops_accum():
    global OpsAccum
    return OpsAccum


def get_last_trace_id():
    """
    This function is used for getting the last trace id.
    Generally speaking, we need to obtain the dependencies between traces.
    If the program exits a stubbed function, the next function,
    when starting off, needs to depend the first trace on the last trace
    recorded.
    :return:
    """
    return len(TraceRegistry) - 1


def dumptrace(tracefile=sys.stdout):
    global OpsAccum
    print("Number of Ops: {}".format(int(OpsAccum)))
    print("Traces:", len(TraceRegistry))
    for trace_id, (trace, trace_dep) in enumerate(
            zip(TraceRegistry[:], TraceDepRegistry[:])):
        try:
            print(*trace(), file=tracefile, sep="\t")
            # print(trace_id, file=tracefile, sep="\t", end="\t")
            # print(*trace(), file=tracefile, sep="\t", end="\t")
            # print(*trace_dep, file=tracefile, sep="\t")

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
    def __init__(self, inner_iter, container: 'TrackedContainer',
                 trace_size=1):
        self.container = container
        self.inner_iter = enumerate(inner_iter)
        self.trace_size = trace_size
        self.iterator_id = next(IteratorCounter)

    def __next__(self):
        count, value = next(self.inner_iter)
        if count % self.trace_size == 0:
            TraceRegistry.append(lambda: self.container.getbase()._replace(
                offset=count * self.container.element_size,
                size=max(self.trace_size * self.container.element_size,
                         self.container.getsize()),
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
        typing.Union[typing.Callable[[], int]]]

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

    def _parse_slice(self, key: slice):
        start = key.start if key.start else 0
        stop = key.stop if key.stop else len(self)
        length = stop - start
        return start, stop, length

    def __setitem__(self, key, value, deps: typing.List = None):
        if isinstance(value, TrackedContainer):
            value.parent = self
            value.key = key
            self.max_element_size_registry.append(
                weakref.ref(value, self.max_size_callback))
        else:
            self.max_element_size_registry.append(self.feature_size)

        self.max_len = max(self.max_len, len(self))

        # If is not a host init, we should assume that the data is prepared
        # in the DRAM and the accelerator can just read from it.
        trace_id = get_last_trace_id()

        if isinstance(value, slice):
            start, _, length = self._parse_slice(value)
            trace_id = TraceRegistry.append(
                lambda: self.getloc(start)._replace(
                    type=TraceElement.WRITE,
                    size=self.feature_size * length
                ),
                deps=deps
            )
        elif not self._is_host_init:
            trace_id = TraceRegistry.append(
                lambda: self.getloc(key)._replace(type=TraceElement.WRITE),
                deps=deps
            )

            super().__setitem__(key, value)

        return trace_id

    def __getitem__(
            self, key, deps: typing.List[int] = [],
            out_edges: typing.List[int] = []
    ):
        val = super().__getitem__(key)

        if isinstance(key, slice):
            # Issue a dense read for slices
            start, _, length = self._parse_slice(key)
            trace_id = TraceRegistry.append(
                lambda: self.getloc(start)._replace(
                    type=TraceElement.READ,
                    size=self.feature_size * length
                ),
                deps=deps,
                out_edges=out_edges
            )
        elif not isinstance(val, TrackedContainer):
            # Reading into another container can be bypassed via smart address calculation
            trace_id = TraceRegistry.append(
                lambda: self.getloc(key)._replace(
                    type=TraceElement.READ
                ),
                deps=deps,
                out_edges=out_edges
            )
        return val, trace_id

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
                lambda: MemorySpace(self.memory_space_id, self.getsize(),
                                    self.element_size)
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
                self.key_to_loc_map[key] = random.choice(
                    tuple(remaining_choices))
            offset = self.key_to_loc_map[key]
        else:
            if key >= self.max_len:
                raise Exception(
                    "This should probably have been a sparse array.")
            offset = key * self.element_size

        return trace._replace(offset=trace.offset + offset,
                              size=self.element_size)


class TrackedList(TrackedContainer, list):
    def __init__(self, *args, is_host_init=False, **kwargs):
        super().__init__(*args, is_host_init=is_host_init, **kwargs)
        # Avoid writes as these could be expensive...
        # for index, value in enumerate(list.__iter__(self)):
        #     # register all values.
        #     self[index] = value
        # self._head = self[0]

    def __iter__(self):
        return TrackedIterator(list.__iter__(self), self)

    def __setitem__(self, key, value, deps: typing.List[int] = None):
        return super(TrackedList, self).__setitem__(key, value, deps)


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
            lambda: self.getloc(item)._replace(type=TraceElement.READ))
        return super().__contains__(item)


def stub():
    networkx.Graph.node_dict_factory = TrackedDict
    networkx.Graph.node_attr_dict_factory = TrackedDict
    networkx.Graph.adjlist_inner_dict_factory = TrackedDict
    networkx.Graph.adjlist_outer_dict_factory = TrackedDict
    networkx.Graph.edge_attr_dict_factory = TrackedDict
