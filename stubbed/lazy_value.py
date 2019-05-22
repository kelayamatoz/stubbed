import abc

import dataclasses
import typing
import weakref
import sys

T = typing.TypeVar("T")


class Evaluable(typing.Generic[T]):
    @abc.abstractmethod
    def evaluate(self) -> T:
        pass


@dataclasses.dataclass
class Value(Evaluable[T]):
    value: T

    def evaluate(self) -> T:
        return self.value


U = typing.TypeVar("U")


@dataclasses.dataclass
class WeakRefValue(Evaluable[T]):
    ref: weakref.ref
    func: typing.Callable[[U], T]
    value: T = None

    def __init__(self, obj: U, func: typing.Callable[[U], T]):
        self.ref = weakref.ref(obj)
        self.func = func

        # hook onto obj's del method
        # since objects use their classes' del methods, we need to hook onto that.
        if not hasattr(obj, "del_callbacks"):
            obj.del_callbacks = []

        if not hasattr(type(obj), "has_del_callback"):
            type(obj).has_del_callback = True

            if hasattr(type(obj), "__del__"):
                old_del = type(obj).__del__
            else:
                old_del = lambda sself: None

            def __del__(sself):
                if hasattr(sself, "del_callbacks"):
                    for callback in sself.del_callbacks:
                        callback(sself)
                return old_del(sself)
            type(obj).__del__ = __del__

        self_wr = weakref.ref(self)

        def callback(o):
            val = self_wr()
            if val is not None:
                val.value = func(o)

        obj.del_callbacks.append(callback)

    def evaluate(self) -> T:
        referred = self.ref()
        if referred is None:
            return self.value
        return self.func(referred)


@dataclasses.dataclass
class Operation(Evaluable[T]):
    func: typing.Callable[[typing.List[typing.Any]], T]
    values: typing.List[typing.Any]

    def evaluate(self) -> T:
        # first check if any evaluable children only have one reference left: this object.
        # In that case, we evaluate it.
        for i in range(len(self.values)):
            # Check for refcount is 2 because it includes counting the argument as a temporary ref.
            if isinstance(self.values[i], Evaluable) and sys.getrefcount(self.values[i]) == 2:
                self.values[i] = self.values[i].evaluate()
        return self.func([(val.evaluate() if isinstance(val, Evaluable) else val) for val in self.values])
