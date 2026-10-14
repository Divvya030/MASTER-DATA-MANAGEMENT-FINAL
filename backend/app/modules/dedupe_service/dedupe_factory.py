"""
The ONE place Dedupe/StaticDedupe objects are constructed.

dedupe==3.0.3 defaults ``num_cores`` to ``multiprocessing.cpu_count()``, which
spawns a pool of worker *processes* on every ``score()`` call. On Windows
(spawn start method) this intermittently raises:
    RuntimeError: You need to either turn off multiprocessing or protect
    the calls to the Dedupe methods with `if __name__ == '__main__'` ...
because the worker re-imports the hosting process's entry module. Forcing
num_cores=1 makes dedupe use its in-process `multiprocessing.dummy` (thread)
implementation instead -- same scoring code, same results, no child process
spawn, no flakiness. Overridable via MDM_DEDUPE_CORES for environments where
real multiprocessing is desired and known to work.
"""
from __future__ import annotations
import io
import os


def dedupe_cores() -> int:
    try:
        return max(1, int(os.environ.get("MDM_DEDUPE_CORES", "1")))
    except ValueError:
        return 1


def new_dedupe(variables):
    import dedupe
    return dedupe.Dedupe(variables, num_cores=dedupe_cores())


def load_static(settings_bytes: bytes):
    import dedupe
    return dedupe.StaticDedupe(io.BytesIO(settings_bytes), num_cores=dedupe_cores())
