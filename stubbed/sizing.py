import typing
from stubbed.trace import MemorySpaceType


__all__ = ["get_size"]


size_registry: typing.Mapping[MemorySpaceType, typing.Mapping[typing.Type, typing.Callable[[typing.Any], int]]] = {
    MemorySpaceType.CPU: {
        int: lambda _: 4,
        float: lambda _: 4,
        str: lambda s: len(s)
    },
    MemorySpaceType.FPGA: {
        int: lambda _: 4,
        float: lambda _: 4,
        str: lambda s: len(s)
    }
}


def get_size(spacetype: MemorySpaceType, v):
    return size_registry[spacetype][type(v)](v)

