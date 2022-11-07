import functools
import warnings


class ExperimentalWarning(UserWarning):
    pass


def warn_experimental(msg):
    warnings.warn(msg, category=ExperimentalWarning, stacklevel=2)


def mark_experimental(f, msg=None):
    def inner(f):
        message = msg
        if message is None:
            full_name = f"{f.__module__}.{f.__class__.__name__}"
            message = f"Function {full_name} is experimental and may cause any kind of troubles"

        warn_experimental(message)
        return f

    return inner


__all__ = (
    "ExperimentalWarning",
    "warn_experimental",
    "mark_experimental",
)
