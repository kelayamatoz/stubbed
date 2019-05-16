from typing import Iterator

import networkx
import collections
import itertools
import typing
import sys
import abc
import atexit
import random

import weakref


class TraceElement(typing.NamedTuple):
    memory_space: int
    offset: int
    size: int
    type: str


class MemorySpace(typing.NamedTuple):
    memory_space: int
    size: int
    element_size: int


MemorySpaceCounter = itertools.count()

TraceRegistry: typing.List[typing.Callable[[], TraceElement]] = []
MemorySpaceRegistry: typing.List[typing.Callable[[], MemorySpace]] = []


def dump():
    for trace in TraceRegistry[:]:
        print(trace())

    for space in MemorySpaceRegistry[:]:
        print(space())


class TrackedContainer:
    def __init_subclass__(cls, is_sparse=False):
        cls.is_sparse = is_sparse

    parent: 'TrackedContainer'
    key: typing.Hashable
    max_len: int
    max_element_size_registry: typing.List[typing.Union[typing.Callable[[], int]]]

    key_to_loc_map: typing.MutableMapping[typing.Hashable, int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = None
        self.key = None
        self._memory_space_id = None
        self.max_len = len(self)
        self.max_element_size_registry = []

        self._max_size = None

        self.key_to_loc_map = {}

    def __setitem__(self, key, value):
        if isinstance(value, TrackedContainer):
            value.parent = self
            value.key = key
            self.max_element_size_registry.append(weakref.ref(value, self.max_size_callback))
        else:
            self.max_element_size_registry.append(8)

        # setting always requires a write
        TraceRegistry.append(lambda: self.getloc(key)._replace(type="WRITE"))

        super().__setitem__(key, value)

        self.max_len = max(self.max_len, len(self))

    def __getitem__(self, key):
        val = super().__getitem__(key)
        if not isinstance(val, TrackedContainer):
            # Reading into another container can be bypassed via smart address calculation
            TraceRegistry.append(lambda: self.getloc(key)._replace(type="READ"))
        return val

    def __delitem__(self, key):
        TraceRegistry.append(lambda: self.getloc(key)._replace(type="DELETE"))
        super().__delitem__(self, key)

    def __superiter__(self):
        return super().__iter__()

    def __iter__(self):
        return super().__iter__()

    @property
    def memory_space_id(self):
        if self.parent:
            return self.parent.memory_space_id
        if self._memory_space_id is None:
            self._memory_space_id = next(MemorySpaceCounter)
            MemorySpaceRegistry.append(lambda: MemorySpace(self.memory_space_id, self.getsize(), self.element_size))
        return self._memory_space_id

    def getsize(self) -> int:
        return self.element_size * self.max_len

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
                if fetched:
                    max_size = max(fetched.getsize(), max_size)

        self._max_size = max_size
        return self._max_size

    def max_size_callback(self, subcontainer):
        self.max_element_size_registry.append(subcontainer.getsize())

    def getloc(self, key) -> TraceElement:
        if self.parent:
            trace = self.parent.getloc(self.key)
        else:
            trace = TraceElement(self.memory_space_id, 0, self.getsize(), "")

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for index, value in enumerate(self.__superiter__()):
            # register all values.
            self[index] = value


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

    def __contains__(self, item):
        TraceRegistry.append(lambda: self.getloc(item)._replace(type="READ"))
        return super().__contains__(item)


def stub():
    networkx.Graph.node_dict_factory = TrackedDict
    networkx.Graph.node_attr_dict_factory = TrackedDict
    networkx.Graph.adjlist_inner_dict_factory = TrackedDict
    networkx.Graph.adjlist_outer_dict_factory = TrackedDict
    networkx.Graph.edge_attr_dict_factory = TrackedDict
