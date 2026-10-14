"""Build + evaluate Supplier V1 on Supplier_1_.csv (explicit `entity_id` per record -> real
pair labels). CAVEAT (repeated in the report): the file is a generated benchmark (all
e-mail/web domains use the reserved .example TLD); numbers measure the training pipeline
against known entity membership, not accuracy on a production supplier master."""
from __future__ import annotations
import itertools, random, time
from collections import defaultdict
from . import common as C
from .specs import SUPPLIER_V1, make_variables

norm = lambda s: (s or "").strip().lower()


def run(csv_path, scratch, seed=42, log=print):
    spec = SUPPLIER_V1; t0 = time.time()
    raw, recs = C.onboard("supplier", csv_path, spec["training_mapping"], scratch)
    entity_of = {i: str(raw["entity_id"].iloc[i]) for i in range(len(raw))}
    ents = defaultdict(list)
    for i, e in entity_of.items():
        ents[e].append(i)
    split_of = {e: C.bucket(e, seed) for e in ents}
    ids_by = defaultdict(list)
    for e, ids in ents.items():
        ids_by[split_of[e]].extend(ids)
    log(f"[supplier] {len(recs)} records via app pipeline; entities={len(ents)}; split records "
        f"{ {k: len(v) for k, v in ids_by.items()} }")

    # ---- training pairs (train split only)
    rng = random.Random(f"{seed}:train")
    ids = ids_by["train"]
    pos = [p for e, m in ents.items() if split_of[e] == "train" and len(m) > 1 for p in itertools.combinations(sorted(m), 2)]
    pos = rng.sample(pos, min(1000, len(pos)))
    by_name, by_ct = defaultdict(list), defaultdict(list)
    for i in ids:
        r = recs[i]; by_name[norm(r.get("supplier_name"))].append(i)
        by_ct[(norm(r.get("city")), norm(r.get("supplier_name")).split(" ")[0])].append(i)
    neg, want = set(), 3 * len(pos)
    def add(groups, k):
        gs = [g for g in groups.values() if len({entity_of[i] for i in g}) > 1]; tries = 0
        while k > 0 and gs and tries < k * 50:
            tries += 1; a, b = rng.sample(rng.choice(gs), 2)
            if entity_of[a] != entity_of[b] and (min(a, b), max(a, b)) not in neg:
                neg.add((min(a, b), max(a, b))); k -= 1
    add(by_name, int(want * .4)); add(by_ct, int(want * .3))
    while len(neg) < want:
        a, b = rng.sample(ids, 2)
        if entity_of[a] != entity_of[b]: neg.add((min(a, b), max(a, b)))
    # original + a masked copy (identifiers blanked: SYNTHETIC augmentation) so the model also
    # decides on name/contact/address alone when uploads carry no identifiers
    idf, arng = spec["identifier_fields"], random.Random(f"aug:{seed}")
    addr = ["address", "city", "postal_code"]
    match, distinct = [], []
    tags = {"original": 0, "identifiers_masked": 0, "address_masked": 0, "name_contact_only": 0}
    for target, pairs in ((match, pos), (distinct, sorted(neg))):
        for i, j in pairs:
            a, b = recs[i], recs[j]
            target.append((a, b)); tags["original"] += 1
            # SYNTHETIC augmentation 1: identifiers blanked, so the model can decide on
            # name/contact/address alone when an upload carries no identifiers
            if arng.random() < .5:
                target.append((C.mask_fields(a, idf, arng, .5), C.mask_fields(b, idf, arng, .5)))
            else:
                target.append((C.mask_fields(a, idf, arng, None), C.mask_fields(b, idf, arng, None)))
            tags["identifiers_masked"] += 1
            # SYNTHETIC augmentation 2: address block blanked, so the learned BLOCKING also works
            # for files with no address columns (e.g. supplier.xlsx) -- names/identifiers/contact only
            if arng.random() < .6:
                target.append((C.mask_fields(a, addr, arng, None), C.mask_fields(b, addr, arng, None)))
                tags["address_masked"] += 1
            # SYNTHETIC augmentation 3: address block AND identifiers blanked (name + e-mail + phone
            # only) -- the typical thin supplier extract; forces the learned blocking to include a
            # name/contact key instead of relying on address or identifiers
            if arng.random() < .6:
                target.append((C.mask_fields(a, addr + idf, arng, None), C.mask_fields(b, addr + idf, arng, None)))
                tags["name_contact_only"] += 1
    log(f"[supplier] training pairs {len(match)} match / {len(distinct)} distinct {tags}")
    bg = {i: recs[i] for i in random.Random(seed).sample(ids, 800)}
    t = time.time()
    dd = C.train_dedupe(make_variables(spec), match, distinct, bg, seed, recall=0.99)
    train_s = round(time.time() - t, 1); settings = C.settings_bytes_of(dd)
    log(f"[supplier] trained {train_s}s predicates={[str(p) for p in dd.predicates]}")

    grid = C.threshold_grid()
    scen = {"full_record": [], "identifiers_hidden": idf, "identifiers_and_contact_hidden": idf + ["email", "phone"],
            "address_hidden": ["address", "city", "postal_code"],
            "address_and_identifiers_hidden": ["address", "city", "postal_code"] + idf}
    results = {"scenarios": {}}
    cache = {}
    def ev(split, blank):
        sub = {i: dict(recs[i]) for i in ids_by[split]}
        for r in sub.values():
            for f in blank: r[f] = None
        truth = {i: entity_of[i] for i in sub}
        res, tp, cand = C.pairwise_end_to_end(settings, sub, truth)
        table, ct = C.score_grid_metrics(tp, cand, grid)
        n = len(sub)
        return ({"records": n, "entities": len(set(truth.values())), "true_pairs": len(tp), "candidate_pairs": len(cand),
                 "possible_pairs": n * (n - 1) // 2, "blocking_recall": round(ct / len(tp), 4),
                 "blocking_reduction_pct": round(100 * (1 - len(cand) / (n * (n - 1) // 2)), 3),
                 "threshold_table": {str(k): v for k, v in table.items()}}, sub, truth, res, tp, cand)
    for name, blank in scen.items():
        results["scenarios"][name] = {}
        for split in ("validation", "test"):
            info, sub, truth, res, tp, cand = ev(split, blank)
            results["scenarios"][name][split] = info
            if name == "identifiers_hidden" and split == "test": cache = (sub, truth, res, tp, cand)
        log(f"[supplier] scenario {name} done")
    # threshold = argmax of the MEAN validation F1 over the three realistic thin-data scenarios
    # (identifiers hidden / identifiers+contact hidden / address+identifiers hidden); ties go to the
    # HIGHER threshold because a false merge costs more than a missed duplicate. Validation only.
    sc = ("identifiers_hidden", "identifiers_and_contact_hidden", "address_and_identifiers_hidden")
    tabs = [results["scenarios"][n]["validation"]["threshold_table"] for n in sc]
    mean_f1 = {k: sum(t[k]["f1"] for t in tabs) / len(tabs) for k in tabs[0]}
    thr = max(((v, float(k)) for k, v in mean_f1.items()))[1]; rthr = round(max(0.0, thr - .15), 4)
    log(f"[supplier] threshold {thr} (mean validation F1 over thin-data scenarios = {mean_f1[str(thr)]:.4f})")
    head = {}
    for name in scen:
        head[name] = {}
        for split in ("validation", "test"):
            info = results["scenarios"][name][split]; row = dict(info["threshold_table"][str(thr)])
            row["precision_ci95"] = C.wilson(row["true_positives"], row["true_positives"] + row["false_positives"])
            row["recall_ci95"] = C.wilson(row["true_positives"], row["true_positives"] + row["false_negatives"])
            row.update(true_pairs=info["true_pairs"], blocking_recall=info["blocking_recall"],
                       blocking_reduction_pct=info["blocking_reduction_pct"])
            head[name][split] = row
    results["headline"] = head
    sub, truth, res, tp, cand = cache
    results["cluster_test"] = C.cluster_metrics(res, sub, truth, thr)
    fps = [k for k, s in cand.items() if s >= thr and k not in tp]
    fns = [k for k in tp if cand.get(k, -1) < thr]
    show = lambda k: {"score": cand.get(k), **{w: {f: recs[k[n]].get(f) for f in ("supplier_name", "city", "email", "phone", "address")} for n, w in enumerate("ab")}}
    r2 = random.Random(seed)
    results["error_analysis"] = {"false_positives": len(fps), "false_negatives": len(fns),
        "missed_by_blocking": sum(1 for k in tp if k not in cand), "scored_below_threshold": len(fns) - sum(1 for k in tp if k not in cand),
        "sample_false_positives": [show(k) for k in r2.sample(fps, min(5, len(fps)))],
        "sample_false_negatives": [show(k) for k in r2.sample(fns, min(5, len(fns)))]}
    results["meta"] = {"threshold": thr, "review_threshold": rthr, "threshold_calibration": "CALIBRATED_ON_VALIDATION_SPLIT",
        "train_seconds": train_s, "training_pairs": {"match": len(match), "distinct": len(distinct), **tags},
        "learned_predicates": [str(p) for p in dd.predicates], "records_total": len(recs), "entities_total": len(ents),
        "split_records": {k: len(v) for k, v in ids_by.items()},
        "split_entities": {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "validation", "test")},
        "total_seconds": round(time.time() - t0, 1)}
    return {"results": results, "settings": settings, "training": C.training_bytes_of(dd)}
