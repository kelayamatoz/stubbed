import itertools
import typing
import weakref

import dataclasses

import stubbed.trace as st


@dataclasses.dataclass
class TrackedInternal:
    # This is so that on destruction, a handle to the "key" values can be maintained while destructing
    # the rest of the object.
    memory: st.Memory = dataclasses.field(default_factory=st.Memory)


class Tracked:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.internals = TrackedInternal()

    def _replace_internals(self, internals: TrackedInternal):
        self.internals = internals

    def create_event(self, replacements):
        pass


class TrackedIterator(Tracked, typing.Iterator):
    counter: typing.ClassVar[itertools.count] = itertools.count()

    def __init__(self, sub_iter: typing.Iterator, internals: TrackedInternal, trace_size=1):
        super().__init__()
        self.sub_iter = sub_iter
        self._replace_internals(internals)
        self.trace_size = trace_size

    def __next__(self):
        return next(self.sub_iter)


class TrackedIterable(Tracked, typing.Iterable):
    def __iter__(self) -> typing.Iterator:
        return TrackedIterator(super().__iter__(), self.internals)


class TrackedContainer(Tracked, typing.Container):
    def __contains__(self, item):
        pass
