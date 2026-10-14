"""
Training orchestration: ACTIVE ER config -> Dedupe variable definitions ->
entity-level split -> training pairs -> real dedupe==3.0.3 training ->
held-out validation scoring -> persisted model artifact.

This module is intentionally free of DB/FastAPI imports (testable in
isolation); app/main.py wires it to persistence and the API.
"""
from __future__ import annotations
import io
from dataclasses import dataclass, field
from typing import Optional

from app.modules.dedupe_service.field_definitions import (
    ERFieldRoles, build_dedupe_variables, InsufficientERConfigError,
)
from app.modules.dedupe_service.training_data import prepare_training_data, SplitDatasets
from app.modules.dedupe_service.field_definitions import coerce_record
from app.modules.dedupe_service.real_labels import (
    split_real_labels, snapshot_pairs, node_key, confusion_metrics, best_f1_threshold, score_labeled_pairs,
)
import random

# Verified empirically against the actual installed dedupe==3.0.3 + sklearn:
# train() uses 5-fold stratified cross-validation internally, so fewer than
# 5 examples in either class makes it raise a raw sklearn ValueError. This
# constant lets us fail with a clear, actionable status INSTEAD of that.
MIN_PAIRS_PER_CLASS = 5


class InsufficientTrainingDataError(ValueError):
    def __init__(self, message: str, pair_counts: dict):
        super().__init__(message)
        self.pair_counts = pair_counts


@dataclass
class TrainingResult:
    status: str  # TRAINED | FAILED | INSUFFICIENT_TRAINING_DATA
    match_fields: list
    split_config: dict
    pair_counts: dict                       # SYNTHETIC pair counts per split (zeros if no augmentation)
    settings_bytes: Optional[bytes] = None
    training_bytes: Optional[bytes] = None
    evaluation_metrics: Optional[dict] = None   # SYNTHETIC validation only (labelled as such)
    error: Optional[str] = None
    # --- real-label evidence (label_source = REAL_STEWARD) ---
    label_counts: Optional[dict] = None         # {"real_steward": {...}, "synthetic": {...}}
    real_evaluation: Optional[dict] = None      # real steward-labelled held-out evaluation
    threshold: Optional[float] = None
    review_threshold: Optional[float] = None
    threshold_calibration: Optional[str] = None
    label_splits: Optional[dict] = None         # {"train": [pair_id..], "validation": [...], "test": [...]}


def _records_to_dedupe_data(records: dict, match_fields: list) -> dict:
    """Restrict each record to ONLY the active MATCH fields and coerce for
    Dedupe. Missing values stay None (Dedupe's has_missing handling) and are
    never turned into "nan"/"None"/"" strings -- see coerce_for_dedupe."""
    return {rid: coerce_record(rec, match_fields) for rid, rec in records.items()}


MIN_REAL_PER_CLASS = 10          # AUTO mode augments with synthetic pairs below this many REAL train labels per class
MIN_CALIBRATION_PER_CLASS = 10   # real validation labels per class needed to CALIBRATE (not just default) the threshold
DEFAULT_THRESHOLD = 0.5
DEFAULT_REVIEW_MARGIN = 0.15


def run_training(
    base_records: dict,
    er_roles: ERFieldRoles,
    seed: int = 42,
    clones_per_entity: int = 2,
    max_entities: int = 400,
    real_labels: Optional[list] = None,
    synthetic_mode: str = "AUTO",          # AUTO | ALWAYS | NEVER
    threshold: Optional[float] = None,
    review_margin: float = DEFAULT_REVIEW_MARGIN,
    base_dataset_id: Optional[str] = None,
) -> TrainingResult:
    """
    base_records: STANDARDIZED records {row_id: {field: value}} (already the
                  output of the existing standardization layer -- nothing is
                  re-standardized here).
    er_roles:     the ACTIVE, steward-confirmed ER roles.
    real_labels:  steward-confirmed labelled pairs (dicts from the LabeledPair
                  repository: pair_id, dataset_id, record_id_a/b, values_a/b,
                  label). PREFERRED evidence. UNSURE / pending are excluded.
    synthetic_mode: AUTO augments with controlled synthetic pairs only when
                  REAL train labels are below MIN_REAL_PER_CLASS per class;
                  ALWAYS forces augmentation; NEVER trains on real labels only.

    Evidence is kept separate end to end: label counts, evaluation and
    provenance distinguish REAL_STEWARD from SYNTHETIC; the two evaluations
    are never merged into one number.
    """
    try:
        variables = build_dedupe_variables(er_roles)
    except InsufficientERConfigError as e:
        return TrainingResult(status="FAILED", match_fields=er_roles.match_fields,
                              split_config={}, pair_counts={}, error=str(e))
    if synthetic_mode not in ("AUTO", "ALWAYS", "NEVER"):
        return TrainingResult(status="FAILED", match_fields=er_roles.match_fields, split_config={},
                              pair_counts={}, error=f"synthetic_mode must be AUTO|ALWAYS|NEVER, got {synthetic_mode!r}")

    match_fields = [v.field for v in variables]
    restricted = {rid: {f: rec.get(f) for f in match_fields} for rid, rec in base_records.items()}
    real_labels = [l for l in (real_labels or []) if l.get("label") in ("MATCH", "DISTINCT")]

    # ---- REAL labels: leak-free split (connected components; see real_labels.py)
    real_split = split_real_labels(real_labels, seed=seed)
    rt_match, rt_distinct = snapshot_pairs(real_split["train"], match_fields)
    real_counts = {
        "real_steward": {
            "match": sum(1 for l in real_labels if l["label"] == "MATCH"),
            "distinct": sum(1 for l in real_labels if l["label"] == "DISTINCT"),
            "train": {"match": len(rt_match), "distinct": len(rt_distinct)},
            "validation": {s_: sum(1 for l in real_split["validation"] if l["label"] == s_.upper()) for s_ in ("match", "distinct")},
            "test": {s_: sum(1 for l in real_split["test"] if l["label"] == s_.upper()) for s_ in ("match", "distinct")},
        },
    }

    # ---- decide about SYNTHETIC augmentation
    real_sufficient = len(rt_match) >= MIN_REAL_PER_CLASS and len(rt_distinct) >= MIN_REAL_PER_CLASS
    use_synthetic = synthetic_mode == "ALWAYS" or (synthetic_mode == "AUTO" and not real_sufficient)

    splits = None
    syn_match, syn_distinct = [], []
    pre_prune_dropped = {"match": 0, "distinct": 0}
    synthetic_pair_counts = {s_: {"match": 0, "distinct": 0} for s_ in ("train", "validation", "test")}
    if use_synthetic:
        splits = prepare_training_data(restricted, match_fields=match_fields, seed=seed,
                                       clones_per_entity=clones_per_entity, max_entities=max_entities)
        synthetic_pair_counts = splits.pair_counts

        # dedupe's index-based predicates cannot index a missing field, and
        # mark_pairs() rejects labelled records its internal sample did not
        # cover (confirmed on ~10k-row real data). For SYNTHETIC pairs only
        # (which are cheap to regenerate) pairs with incomplete match fields
        # are skipped up front; REAL labels keep their missing values and rely
        # on the bounded prune/retry below, so real missing-value evidence is
        # not thrown away.
        def _complete(rec: dict) -> bool:
            return all(rec.get(f) not in (None, "") for f in match_fields)

        syn_match = [(a, b) for a, b in splits.train_match_pairs if _complete(a) and _complete(b)]
        syn_distinct = [(a, b) for a, b in splits.train_distinct_pairs if _complete(a) and _complete(b)]
        pre_prune_dropped = {"match": len(splits.train_match_pairs) - len(syn_match),
                             "distinct": len(splits.train_distinct_pairs) - len(syn_distinct)}
    real_counts["synthetic"] = {
        "used": use_synthetic, "mode": synthetic_mode,
        "train": {"match": len(syn_match), "distinct": len(syn_distinct)},
        "note": "SYNTHETIC pairs are controlled augmentation, not real-world ground truth.",
    }

    train_match = list(rt_match) + syn_match
    train_distinct = list(rt_distinct) + syn_distinct
    split_config = {
        "seed": seed, "clones_per_entity": clones_per_entity, "max_entities": max_entities,
        "synthetic_mode": synthetic_mode, "synthetic_used": use_synthetic,
        "real_split_method": "connected components over all labelled pairs (MATCH and DISTINCT); "
                             "each component assigned wholesale to train/validation/test by hash -- no record in two splits",
        "synthetic_split_method": "entity-level: each base record + its synthetic clones form one entity, assigned wholesale to one split",
    }

    if len(train_match) < MIN_PAIRS_PER_CLASS or len(train_distinct) < MIN_PAIRS_PER_CLASS:
        return TrainingResult(
            status="INSUFFICIENT_TRAINING_DATA", match_fields=match_fields, split_config=split_config,
            pair_counts=synthetic_pair_counts, label_counts=real_counts,
            error=(
                f"Not enough labelled training pairs: {len(train_match)} MATCH / {len(train_distinct)} DISTINCT "
                f"({len(rt_match)}/{len(rt_distinct)} REAL_STEWARD + {len(syn_match)}/{len(syn_distinct)} SYNTHETIC) "
                f"in the training split; Dedupe's internal cross-validation needs at least {MIN_PAIRS_PER_CLASS} of each. "
                f"Label more candidate pairs (real steward labels are preferred), enable synthetic augmentation, "
                f"or configure more MATCH fields."
            ),
        )

    import dedupe

    # ---- data handed to prepare_training(): synthetic train records, the real
    # labelled snapshot records (train split only), and a sample of the real
    # dataset for realistic blocking. Real validation/test records are EXCLUDED
    # so nothing held out influences the learned blocking/classifier.
    holdout_keys = {node_key(l.get("dataset_id"), l[k]) for l in real_split["validation"] + real_split["test"]
                    for k in ("record_id_a", "record_id_b")}
    train_data: dict = {}
    if splits is not None:
        for rid, rec in _records_to_dedupe_data(splits.train, match_fields).items():
            train_data[f"S:{rid}"] = rec
    for i, l in enumerate(real_split["train"]):
        train_data[f"L:{i}:a"] = coerce_record(l["values_a"], match_fields)
        train_data[f"L:{i}:b"] = coerce_record(l["values_b"], match_fields)
    if real_labels:
        rng = random.Random(seed)
        ids = sorted(restricted.keys())
        rng.shuffle(ids)
        for rid in ids[:1500]:
            if base_dataset_id is not None and node_key(base_dataset_id, rid) in holdout_keys:
                continue
            train_data[f"B:{rid}"] = coerce_record(restricted[rid], match_fields)
    sample_size = max(1500, len(train_data) * 3)

    try:
        deduper, train_match, train_distinct, pruned_dropped = _mark_pairs_with_pruning(
            dedupe, variables, train_data, sample_size, train_match, train_distinct, match_fields,
        )
        if len(train_match) < MIN_PAIRS_PER_CLASS or len(train_distinct) < MIN_PAIRS_PER_CLASS:
            return TrainingResult(
                status="INSUFFICIENT_TRAINING_DATA", match_fields=match_fields, split_config=split_config,
                pair_counts=synthetic_pair_counts, label_counts=real_counts,
                error=(f"After pruning labelled pairs dedupe's active learner could not index ({pruned_dropped} dropped), "
                       f"only {len(train_match)} MATCH / {len(train_distinct)} DISTINCT remained, below {MIN_PAIRS_PER_CLASS}."),
            )
        deduper.train(recall=1.0, index_predicates=True)
    except Exception as e:  # noqa: BLE001 -- must never silently report success
        return TrainingResult(status="FAILED", match_fields=match_fields, split_config=split_config,
                              pair_counts=synthetic_pair_counts, label_counts=real_counts,
                              error=f"Dedupe training failed: {type(e).__name__}: {e}")

    settings_buf = io.BytesIO()
    deduper.write_settings(settings_buf)
    training_buf = io.StringIO()
    deduper.write_training(training_buf)

    # ---- (A) SYNTHETIC validation -- only verifies the pipeline + augmentation
    evaluation_metrics = None
    if splits is not None:
        evaluation_metrics = _evaluate_on_held_out(
            deduper, splits.validation_match_pairs, splits.validation_distinct_pairs, match_fields)
        evaluation_metrics["kind"] = "SYNTHETIC_VALIDATION"
        evaluation_metrics["note"] = ("Synthetic validation: measures the training pipeline on controlled corruptions "
                                      "of real records. It is NOT a measurement of real-world accuracy.")

    # ---- (B) REAL steward-labelled evaluation on HELD-OUT real labels
    def _eval_real(lbls):
        pl = [(l["values_a"], l["values_b"], 1 if l["label"] == "MATCH" else 0) for l in lbls]
        return score_labeled_pairs(deduper, pl, match_fields)

    val_scores, val_y = _eval_real(real_split["validation"])
    test_scores, test_y = _eval_real(real_split["test"])

    # ---- threshold: calibrate on REAL validation labels only when enough exist
    calibrated = (sum(val_y) >= MIN_CALIBRATION_PER_CLASS and (len(val_y) - sum(val_y)) >= MIN_CALIBRATION_PER_CLASS)
    if threshold is not None:
        final_threshold, calibration = float(threshold), "MANUAL"
    elif calibrated:
        final_threshold, _ = best_f1_threshold(val_scores, val_y)
        calibration = "CALIBRATED_ON_REAL_VALIDATION_LABELS"
    else:
        final_threshold, calibration = DEFAULT_THRESHOLD, "PROVISIONAL"
    review_threshold = round(max(0.0, final_threshold - review_margin), 4)

    real_evaluation = {
        "kind": "REAL_STEWARD_LABELLED_EVALUATION",
        "label_source": "REAL_STEWARD",
        "validation": {"pair_count": len(val_y), "match_count": sum(val_y), "distinct_count": len(val_y) - sum(val_y),
                       **(confusion_metrics(val_scores, val_y, final_threshold) if val_y else {})},
        "test": {"pair_count": len(test_y), "match_count": sum(test_y), "distinct_count": len(test_y) - sum(test_y),
                 **(confusion_metrics(test_scores, test_y, final_threshold) if test_y else {})},
        "threshold": final_threshold, "threshold_calibration": calibration,
        "note": ("Held-out REAL steward labels only (never trained on). "
                 + ("Threshold calibrated on the real validation split."
                    if calibration.startswith("CALIBRATED") else
                    "Too few real validation labels to calibrate: threshold is PROVISIONAL and these numbers are "
                    "indicative only." if calibration == "PROVISIONAL" else "Threshold set manually.")),
    } if real_labels else None

    split_config.update({"dedupe_sample_size": sample_size,
                         "train_pairs_dropped_for_missing_data": pre_prune_dropped,
                         "train_pairs_dropped_unindexable": pruned_dropped})
    return TrainingResult(
        status="TRAINED", match_fields=match_fields, split_config=split_config,
        pair_counts=synthetic_pair_counts,
        settings_bytes=settings_buf.getvalue(), training_bytes=training_buf.getvalue().encode("utf-8"),
        evaluation_metrics=evaluation_metrics, label_counts=real_counts, real_evaluation=real_evaluation,
        threshold=final_threshold, review_threshold=review_threshold, threshold_calibration=calibration,
        label_splits={k: [l["pair_id"] for l in v] for k, v in real_split.items()},
    )


def _mark_pairs_with_pruning(dedupe_module, variables, train_data: dict, sample_size: int,
                              train_match: list, train_distinct: list, match_fields: list,
                              max_attempts: int = 25):
    """
    dedupe==3.0.3's mark_pairs() (bulk pre-labeled training, as opposed to
    its interactive uncertain_pairs() loop) requires every record in a
    labeled pair to be resolvable through the index-based predicates its
    active learner built from an internal SAMPLE of `train_data` -- and
    that sample is not guaranteed to cover every record, independent of
    whether the record has missing fields (confirmed empirically: this
    reproduces even for a fully-populated single-field record on real
    ~8,000-row customer data, after the missing-data filtering above had
    already fixed a *different* real dataset's failure). No sample_size
    scaling we tested reliably eliminates this, short of an impractically
    large one that would make training far slower.

    So: catch the failure, parse out exactly which record dedupe's active
    learner rejected (message embeds its repr), drop every labeled pair
    that references that specific record, and retry -- on a FRESH Dedupe
    instance, since a failed mark_pairs() call still appends to
    self.training_pairs internally before raising, so the old instance's
    state can't be trusted. Bounded by max_attempts so this can never loop
    forever; if pruning can't converge within that budget, the caller
    treats it as an ordinary training failure.
    """
    import ast

    match_pairs, distinct_pairs = list(train_match), list(train_distinct)
    dropped = 0

    from app.modules.dedupe_service.dedupe_factory import dedupe_cores
    for _ in range(max_attempts):
        deduper = dedupe_module.Dedupe(variables, num_cores=dedupe_cores())
        deduper.prepare_training(train_data, sample_size=sample_size)
        try:
            deduper.mark_pairs({
                "match": [(_coerce(a, match_fields), _coerce(b, match_fields)) for a, b in match_pairs],
                "distinct": [(_coerce(a, match_fields), _coerce(b, match_fields)) for a, b in distinct_pairs],
            })
            return deduper, match_pairs, distinct_pairs, dropped
        except UserWarning as e:
            msg = str(e)
            start, end = msg.find("The record\n"), msg.find("\nis not known")
            if start == -1 or end == -1:
                raise  # not the specific failure mode we know how to prune -- surface it as-is
            try:
                failing_record = ast.literal_eval(msg[start + len("The record\n"):end])
            except (ValueError, SyntaxError):
                raise
            before = len(match_pairs) + len(distinct_pairs)
            match_pairs = [(a, b) for a, b in match_pairs
                           if _coerce(a, match_fields) != failing_record and _coerce(b, match_fields) != failing_record]
            distinct_pairs = [(a, b) for a, b in distinct_pairs
                               if _coerce(a, match_fields) != failing_record and _coerce(b, match_fields) != failing_record]
            after = len(match_pairs) + len(distinct_pairs)
            if after == before:
                raise  # couldn't identify which pair(s) to drop -- don't loop forever on no progress
            dropped += (before - after)

    raise RuntimeError(f"Could not train after pruning {dropped} unindexable labeled pairs "
                        f"within {max_attempts} attempts")


def _coerce(record: dict, match_fields: list) -> dict:
    return coerce_record(record, match_fields)


def _evaluate_on_held_out(deduper, match_pairs_with_ids, distinct_pairs_with_ids, match_fields, threshold: float = 0.5) -> dict:
    """
    Lightweight VALIDATE step scoped to Module 8 itself: scores held-out
    labeled pairs (never used in training) with the just-trained classifier
    and reports pair-level precision/recall/F1. This is NOT the full
    Module 14 evaluation subsystem (no operational/monitoring stats, no
    cluster-level metrics) -- just confirms the trained classifier actually
    discriminates on data it has never seen, using dedupe's own score().
    """
    pairs = []
    labels = []
    for (rid_a, rec_a), (rid_b, rec_b) in match_pairs_with_ids:
        pairs.append(((rid_a, _coerce(rec_a, match_fields)), (rid_b, _coerce(rec_b, match_fields))))
        labels.append(1)
    for (rid_a, rec_a), (rid_b, rec_b) in distinct_pairs_with_ids:
        pairs.append(((rid_a, _coerce(rec_a, match_fields)), (rid_b, _coerce(rec_b, match_fields))))
        labels.append(0)

    if not pairs:
        return {
            "validation_pair_count": 0,
            "note": "No held-out validation pairs were available (dataset too small); "
                    "evaluation metrics could not be computed.",
        }

    scores = deduper.score(pairs)
    predictions = [1 if s >= threshold else 0 for s in scores["score"]]

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "validation_pair_count": len(pairs),
        "validation_match_count": sum(labels),
        "validation_distinct_count": len(labels) - sum(labels),
        "threshold": threshold,
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
    }
