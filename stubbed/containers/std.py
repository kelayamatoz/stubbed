import abc
import dataclasses
import functools
import itertools
import typing
from typing import ItemsView, ValuesView, KeysView, Tuple

from stubbed import lazy_value as slv
from stubbed.sizing import get_size
import stubbed.trace as st

import stubbed.containers.core as scc

K = typing.TypeVar("K")
V = typing.TypeVar("V")

T = typing.TypeVar("T")


class TrackedView(scc.Tracked, typing.Generic[T]):
    def __init__(self, view, parent: 'TrackedDict', elements_per_read: int = 1):
        super().__init__()
        self.view = view
        self.elements_per_read = elements_per_read
        self.parent = parent
        self.move_space(parent.memory.space, parent.memory.offset)

    @property
    def element_size(self):
        return self.parent.element_size

    @abc.abstractmethod
    def __contains__(self, item: T) -> bool:
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass


class TrackedKeysView(TrackedView):
    def __contains__(self, key: T) -> bool:
        ret = key in self.view
        loc = self.parent._get_key_location(key)
        scc.Tracked.register_event(st.TraceElement(loc, st.MemoryAccessType.READ))
        return ret

    def __iter__(self):
        for i, result in enumerate(self.view):
            if i % self.elements_per_read == 0:
                loc = self.parent._get_key_location(result)
                loc = dataclasses.replace(loc, num_elements=slv.Value(self.elements_per_read))
                scc.Tracked.register_event(st.TraceElement(loc, st.MemoryAccessType.READ))
            yield result


class TrackedValuesView(TrackedView):
    def __contains__(self, value: T) -> bool:
        # no choice but to iterate through.
        for v in iter(self):
            if v == value:
                return True
        return False

    def __iter__(self):
        for i, result in enumerate(self.view):
            if i % self.elements_per_read == 0:
                loc = self.parent._get_value_location(i)
                loc = dataclasses.replace(loc, num_elements=slv.Value(self.elements_per_read))
                scc.Tracked.register_event(st.TraceElement(loc, st.MemoryAccessType.READ))
            yield result


class TrackedItemsView(TrackedView):
    def __contains__(self, item):
        ret = item in self.view

        return ret


class TrackedDict(scc.TrackedIterable[K], typing.Dict[K, V]):
    __max_size: slv.Evaluable[int]

    def _get_size(self, v) -> slv.Evaluable[int]:
        if isinstance(v, scc.Tracked):
            return v.size
        else:
            return slv.WeakRefValue(self, lambda s: get_size(s.memory.evaluate().space.evaluate().type, v))

    def __init__(self, mapping_or_iterable, **kwargs):
        super().__init__(mapping_or_iterable, **kwargs)
        self._key_sizes = []
        self._value_sizes = []
        for k, v in typing.Dict.items(self):
            self._key_sizes.append(self._get_size(k))
            self._value_sizes.append(self._get_size(v))

        self.key_size = slv.Operation(functools.partial(max, default=0), self._key_sizes)
        self.value_size = slv.Operation(functools.partial(max, default=0), self._value_sizes)
        self._element_size = slv.Operation(lambda l: l[0] * l[1], [self.key_size, self.value_size])

    @property
    def element_size(self) -> slv.Evaluable[int]:
        return self._element_size

    def popitem(self) -> Tuple[K, V]:
        return super().popitem()

    def update(self, m: typing.Union[typing.Mapping[K, V], typing.Iterable[typing.Tuple[K, V]]], **kwargs: V) -> None:
        iterator = itertools.chain(
            (m.items() if hasattr(m, "items") else m),
            kwargs.items()
        )
        for k, v in iterator:
            self[k] = v

    def _get_key_location(self, key: K) -> st.MemoryBlock:
        pass

    def _get_value_location(self, value_index: int) -> st.MemoryBlock:
        pass

    def _get_value_offset(self) -> slv.Evaluable[int]:
        pass

    def keys(self) -> KeysView[K]:
        return super().keys()

    def values(self) -> ValuesView[V]:
        return super().values()

    def items(self) -> ItemsView[K, V]:
        return super().items()
