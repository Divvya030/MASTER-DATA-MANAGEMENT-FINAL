"""Minimal pure-Python stand-in for pyhacrf 0.1.2 (vendored copy).

WHY THIS EXISTS
---------------
pyhacrf is a C-extension dependency of `highered`, which `dedupe` imports
unconditionally via `dedupe.variables.string` (and even instantiates
`CRFEditDistance()` at import time). pyhacrf ships no prebuilt Windows wheels
for Python 3.11/3.12, and building it needs a C compiler + Cython, which this
deployment machine does not have.

This application configures dedupe with `String` variables ONLY
(see app/modules/dedupe_service/field_definitions.py), so `CRFEditDistance`
is imported but never used for prediction or training. The classes below are
inert, constructible placeholders that let the import chain succeed; any
attempt to actually use them raises a clear error.

INSTALL (idempotent): copy this directory into the venv site-packages, e.g.

    backend/.venv/Scripts/python.exe -c "import sysconfig, shutil; \
shutil.copytree('stubs/pyhacrf', sysconfig.get_paths()['purelib'] + '/pyhacrf', dirs_exist_ok=True)"

Or on a machine with a working toolchain, install the real pyhacrf instead:

    uv pip install --python backend/.venv/Scripts/python.exe pyhacrf==0.1.2
"""
__all__ = ["Hacrf", "StringPairFeatureExtractor", "PairFeatureExtractor"]
__version__ = "0.1.2-stub"


class _PyhacrfUnavailable(RuntimeError):
    def __init__(self, cls_name: str):
        super().__init__(
            f"{cls_name} requires the real pyhacrf C extension, which is not "
            f"installed on this platform. This app does not use "
            f"CRFEditDistance (ShortString) variables, so this call is "
            f"unexpected."
        )


class Hacrf:
    """Inert stand-in; only construction and attribute assignment supported."""

    def __init__(self, *args, **kwargs):
        self.parameters = None
        self.classes = None

    def _unavailable(self):
        raise _PyhacrfUnavailable("Hacrf")

    def fit(self, *args, **kwargs):
        self._unavailable()

    def predict_proba(self, *args, **kwargs):
        self._unavailable()


class StringPairFeatureExtractor:
    """Inert stand-in; only construction is supported."""

    def __init__(self, *args, **kwargs):
        self.match = kwargs.get("match", True)
        self.numeric = kwargs.get("numeric", False)

    def _unavailable(self):
        raise _PyhacrfUnavailable("StringPairFeatureExtractor")

    def fit_transform(self, *args, **kwargs):
        self._unavailable()

    def _extract_features(self, *args, **kwargs):
        self._unavailable()


class PairFeatureExtractor:  # pragma: no cover - defensive placeholder
    def __init__(self, *args, **kwargs):
        self._unavailable()

    def _unavailable(self):
        raise _PyhacrfUnavailable("PairFeatureExtractor")
