import sys

if sys.version_info >= (3, 10):
    from dataclasses import field as dataclass_field
else:
    from dataclasses import field as _field

    def dataclass_field(*, kw_only, **kwargs):
        return _field(**kwargs)


__all__ = ["dataclass_field"]
