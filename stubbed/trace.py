import atexit
import contextlib
import enum
import itertools
import sys
import typing
import dataclasses

from stubbed.lazy_value import Evaluable


class MemoryAccessType(enum.Enum):
    READ = "R"
    WRITE = "W"
    DELETE = "D"
    UNKNOWN = "U"


class TraceElement(typing.NamedTuple):
    memory: 'Memory'
    offset: Evaluable[int]
    size: Evaluable[int]
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
class Memory:
    space: MemorySpace = dataclasses.field(default_factory=MemorySpace)
    id: int = -1
    size: int = Evaluable[int]
    element_size: int = Evaluable[int]


class TraceContext(contextlib.AbstractContextManager):
    context_stack: typing.ClassVar[typing.List['TraceContext']] = []
    context_counter: typing.ClassVar[itertools.count] = itertools.count()
    enabled: bool = True

    event_list: typing.List[typing.Callable[[], TraceElement]]

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
            print(*event(), file=output)
        self.enabled = is_enabled

    def register_event(self, event: typing.Callable[[], TraceElement]) -> bool:
        if self.enabled:
            self.event_list.append(event)
        return self.enabled


# Create a global trace context.
TraceContext.context_stack.append(TraceContext(name="base"))
atexit.register(TraceContext.active_trace().dump)
