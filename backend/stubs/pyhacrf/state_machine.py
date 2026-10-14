"""Stub for pyhacrf.state_machine (see package __init__ for rationale)."""


class DefaultStateMachine:
    """Inert stand-in; only construction is supported."""

    def __init__(self, classes=None):
        self.classes = list(classes) if classes else ["match", "non-match"]

    def _unavailable(self):
        from . import _PyhacrfUnavailable

        raise _PyhacrfUnavailable("DefaultStateMachine")


__all__ = ["DefaultStateMachine"]
