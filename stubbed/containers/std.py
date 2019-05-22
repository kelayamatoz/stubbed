from stubbed.trace import TraceContext, MemoryAccessType

from stubbed.containers.core import TrackedContainer, TrackedIterator


class TrackedList(TrackedContainer, list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for index, value in enumerate(list.__iter__(self)):
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
            if hasattr(other, "items"):
                for k, v in other.items():
                    self[k] = v
            else:
                for k, v in other:
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
        TraceContext.active_trace().register_event(lambda: self.getloc(item)._replace(type=MemoryAccessType.READ))
        return super().__contains__(item)
