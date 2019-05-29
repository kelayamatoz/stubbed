import atexit
import contextlib
import enum
import itertools
import sys
import typing
import dataclasses

import stubbed.lazy_value as slv


class MemoryAccessType(enum.Enum):
    READ = "R"
    WRITE = "W"
    DELETE = "D"
    UNKNOWN = "U"


class TraceElement(typing.NamedTuple):
    memory: 'MemoryBlock'
    type: MemoryAccessType = MemoryAccessType.UNKNOWN
    iterator_id: int = -1


class MemorySpaceType(enum.Enum):
    CPU = "CPU"
    FPGA = "FPGA"
    GPU = "GPU"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass
class MemorySpace:
    type: MemorySpaceType = dataclasses.field(default=MemorySpaceType.UNKNOWN)
    id: int = -1


@dataclasses.dataclass
class MemoryBlock:
    space: slv.Evaluable[MemorySpace]
    offset: slv.Evaluable[int]  # In bytes
    element_size: slv.Evaluable[int]
    num_elements: slv.Evaluable[int]

    def adjust_offset(self, shift: slv.Evaluable[int]):
        """
        :param shift: Shift amount in terms of bytes
        :return: shifted block (offset + shift)
        """
        new_offset = slv.Operation(lambda x: x[0] + x[1], [self.offset, shift])
        return MemoryBlock(
            self.space, new_offset, self.element_size, self.num_elements
        )


class TraceContext(contextlib.AbstractContextManager):
    context_stack: typing.ClassVar[typing.List['TraceContext']] = []
    context_counter: typing.ClassVar[itertools.count] = itertools.count()
    enabled: bool = True

    event_list: typing.List[slv.Evaluable[TraceElement]]

    def __init__(self, name: str = "unnamed"):
        self.id = next(self.context_counter)
        self.name = name
        self.event_list = []

    def __enter__(self):
        self.context_stack.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        assert self is self.context_stack.pop()

    @classmethod
    def active_trace(cls) -> 'TraceContext':
        return cls.context_stack[-1]

    def __del__(self):
        # Since we can't guarantee that there aren't references (specifically iterators and structures)
        # to an object created within this context, we instead dump when this object is deleted.
        # As a result, they should hold a reference to this context until they have resolved.
        pass

    def dump(self, output: typing.TextIO = sys.stdout):
        is_enabled = self.enabled
        self.enabled = False
        for event in self.event_list:
            print(*event.evaluate(), file=output)
        self.enabled = is_enabled

    def register_event(self, event: slv.Evaluable[TraceElement]) -> bool:
        if self.enabled:
            self.event_list.append(event)
        return self.enabled


# Create a global trace context.
TraceContext.context_stack.append(TraceContext(name="base"))
atexit.register(TraceContext.active_trace().dump)
