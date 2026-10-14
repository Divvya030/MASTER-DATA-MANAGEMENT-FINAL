"""Stub for pyhacrf.adjacent (see package __init__ for rationale)."""


def forward_predict(*args, **kwargs):  # pragma: no cover - defensive placeholder
    from . import _PyhacrfUnavailable

    raise _PyhacrfUnavailable("forward_predict")


__all__ = ["forward_predict"]
