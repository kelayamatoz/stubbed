import abc
import dataclasses
import itertools
import typing

import stubbed.trace as st

import stubbed.lazy_value as slv


T_co = typing.TypeVar("T_co", covariant=True)
T = typing.TypeVar("T")


class Tracked:
    _offset: slv.Evaluable[int]
    _memory: st.MemoryBlock

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memoryspace = slv.Value(st.MemorySpace())
        self._memory = self.compute_memory()
        self._offset = slv.Value(0)
        self._moved = False

    @property
    @abc.abstractmethod
    def element_size(self) -> slv.Evaluable[int]:
        pass

    @property
    def num_elements(self) -> slv.Evaluable[int]:
        return slv.Value(1)

    @property
    def size(self):
        return slv.Operation(lambda l: l[0] * l[1], [self.element_size, self.num_elements])

    def compute_memory(self) -> st.MemoryBlock:
        return st.MemoryBlock(self._memoryspace, self._offset, self.element_size, self.num_elements)

    @property
    def memory(self) -> st.MemoryBlock:
        return self._memory

    @staticmethod
    def register_event(event):
        return st.TraceContext.active_trace().register_event(event)

    def move_space(self, space: slv.Evaluable[st.MemorySpace], offset: slv.Evaluable[int]):
        assert not self._moved
        self._moved = True
        self._memoryspace = space
        self._offset = offset


class TrackedIterator(Tracked, typing.Iterator[T_co]):
    counter: typing.ClassVar[itertools.count] = itertools.count()

    def __init__(self, sub_iter: typing.Iterator[T_co], obj: Tracked, trace_size=1):
        super().__init__()
        self.sub_iter = sub_iter
        self.trace_size = trace_size
        self.obj = obj
        self.index = 0
        self.id = next(self.counter)

        self.move_space(self.obj._memoryspace, self.obj._offset)

    def create_event(self) -> st.TraceElement:
        read_mem = dataclasses.replace(
            self.memory.adjust_offset(
                slv.Operation(lambda x: x[0] * x[1], [slv.Value(self.index), self.element_size])),
            num_elements=self.trace_size)
        return st.TraceElement(read_mem, st.MemoryAccessType.READ, self.id)

    def __next__(self) -> T_co:
        val = next(self.sub_iter)
        # Necessary in case next raises a StopIteration.
        if self.index % self.trace_size == 0:
            # Emit a trace.
            self.register_event(self.create_event())
        self.index += 1
        return val

    @property
    def element_size(self):
        return self.obj.element_size


class TrackedIterable(Tracked, typing.Iterable[T_co], abc.ABC):
    def __iter__(self) -> TrackedIterator[T_co]:
        return TrackedIterator(super().__iter__(), self)
