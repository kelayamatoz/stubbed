from typing import Iterator

import networkx
import collections
import itertools
import typing
import atexit


class TraceElement(typing.NamedTuple):
    memory_space: int
    offset: int
    size: int
    type: str


class MemorySpace(typing.NamedTuple):
    memory_space: int
    size: int


MemorySpaceCounter = itertools.count()

TraceRegistry: typing.List[typing.Callable[[], TraceElement]] = []


# Stores Node Attributes, keyed by name.
class NodeAttrDict(typing.MutableMapping[str, typing.Any]):
    parent: 'NodeDict'
    node_id: int

    def __init__(self):
        self._inner = {}

    def __setitem__(self, k: str, v: typing.Any) -> None:
        self._inner[k] = v

        def materialize():
            return TraceElement(
                self.parent.memory_space_id,
                self.parent.get_loc(self.node_id, k),
                self.parent.node_attr_size,
                "WRITE"
            )

        TraceRegistry.append(materialize)

    def __delitem__(self, k: str) -> None:
        del self._inner[k]

        def materialize():
            return TraceElement(
                self.parent.memory_space_id,
                self.parent.get_loc(self.node_id, k),
                self.parent.node_attr_size,
                "WRITE"
            )

        TraceRegistry.append(materialize)

    def __getitem__(self, k: str) -> typing.Any:
        def materialize():
            return TraceElement(
                self.parent.memory_space_id,
                self.parent.get_loc(self.node_id, k),
                self.parent.node_attr_size,
                "READ"
            )

        TraceRegistry.append(materialize)
        return self._inner[k]

    def __len__(self) -> int:
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)



# Stores NodeAttrDicts
class NodeDict(typing.MutableMapping[typing.Hashable, NodeAttrDict]):
    _inner: typing.MutableMapping[typing.Hashable, NodeAttrDict]
    node_attr_size: int = 8

    def __init__(self):
        self.memory_space_id = next(MemorySpaceCounter)

        self._inner = {}
        attr_counter = itertools.count()
        self.attr_locations = collections.defaultdict(lambda: next(attr_counter))

        node_counter = itertools.count()
        self.node_locations = collections.defaultdict(lambda: next(node_counter))

    def __setitem__(self, k: typing.Hashable, v: NodeAttrDict) -> None:
        self._inner[k] = v
        v.parent = self

        # generates a location for the node.
        self.node_locations[k]

    def __delitem__(self, v: typing.Hashable) -> None:
        del self._inner[v]

    def __getitem__(self, k: typing.Hashable) -> NodeAttrDict:
        return self._inner[k]

    def __len__(self) -> int:
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)

    def get_loc(self, node_id, attr_name):

        # guarantee that the attribute location is registered
        self.attr_locations[attr_name]
        return (node_id * len(self.attr_locations) + self.attr_locations[attr_name]) * self.node_attr_size



def stub():
    networkx.Graph.node_dict_factory = NodeDict
    networkx.Graph.node_attr_dict_factory = NodeAttrDict
