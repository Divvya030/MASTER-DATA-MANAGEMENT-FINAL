"""Memory guard for Dedupe's blocking-predicate learner.

Dedupe's `prepare_training()` draws a hard-coded sample of up to 5,000 records
(labeler.py: N_SAMPLED_RECORDS) and, for EVERY candidate blocking predicate,
materialises every record pair that predicate puts in the same block. For a
low-cardinality field (country, postal prefix, ...) one predicate can cover
~n^2/2 pairs, so with 5+ comparison fields memory grows into the multi-GB range
even on small data. `bounded_sample_records(n)` caps that internal sample for the
duration of a training call. It changes only HOW MANY records the predicate learner
inspects -- never the classifier, the labels, or the matching engine.
"""
from __future__ import annotations
from contextlib import contextmanager

DEFAULT_MAX_SAMPLED_RECORDS = 500


@contextmanager
def bounded_sample_records(max_records: int = DEFAULT_MAX_SAMPLED_RECORDS):
    from dedupe import labeler
    original = labeler.sample_records

    def capped(data, sample_size):
        # labeler calls sample_records twice: the INDEX data (50,000 -- left intact so
        # every labelled record stays indexable) and the predicate-cover sample (5,000
        # -- the memory hog). Only the latter is capped.
        if sample_size == 5000:
            sample_size = min(sample_size, max_records)
        return original(data, sample_size)

    labeler.sample_records = capped
    try:
        yield
    finally:
        labeler.sample_records = original
