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

    @property
    def finalized(self) -> bool:
        return False


@dataclasses.dataclass
class Value(Evaluable[T]):
    _value: T

    def evaluate(self) -> T:
        return self._value

    @property
    def finalized(self) -> bool:
        # Only reference to self._value is through self, the extra one is temporary in the call.
        return sys.getrefcount(self._value) == 2


U = typing.TypeVar("U")


@dataclasses.dataclass
class WeakRefValue(Evaluable[T]):
    _ref: weakref.ref
    _func: typing.Callable[[U], T]
    _value: T = None

    def __init__(self, obj: U, func: typing.Callable[[U], T]):
        self._ref = weakref.ref(obj)
        self._func = func

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
                val._value = func(o)

        obj.del_callbacks.append(callback)

    def evaluate(self) -> T:
        referred = self._ref()
        if referred is None:
            return self._value
        return self._func(referred)

    @property
    def finalized(self) -> bool:
        return self._ref() is None


@dataclasses.dataclass
class Operation(Evaluable[T]):
    _func: typing.Callable[[typing.List[typing.Any]], T]
    _values: typing.List[typing.Any]

    def evaluate(self) -> T:
        # first check if any evaluable children only have one reference left: this object.
        # In that case, we evaluate it.
        for i in range(len(self._values)):
            # Check for refcount is 2 because it includes counting the argument as a temporary ref.
            if isinstance(self._values[i], Evaluable) and sys.getrefcount(self._values[i]) == 2:
                self._values[i] = self._values[i].evaluate()
        return self._func([(val.evaluate() if isinstance(val, Evaluable) else val) for val in self._values])

    @property
    def finalized(self):
        # check if func is finalized
        if sys.getrefcount(self._func) > 2:
            return False

        # check if values list is finalized
        if sys.getrefcount(self._values) > 2:
            return False

        return all(value.finalized for value in self._values)
