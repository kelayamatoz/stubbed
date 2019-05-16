
class FunctionWrapper:
    def __init__(self, inner_func):
        self.inner_func = inner_func

    def before(self, *args, **kwargs):
        pass

    def after(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.before(*args, **kwargs)
        val = self.inner_func(*args, **kwargs)
        self.after(val, *args, **kwargs)
        return val

    @classmethod
    def method_hook(cls, target, name):
        target_func = getattr(target, name)
        wrapped = cls(target_func)

        def func(self, *args, **kwargs):
            return wrapped(self, *args, **kwargs)
        setattr(target, name, func)
        return wrapped

    @classmethod
    def function_hook(cls, func):
        return cls(func)

