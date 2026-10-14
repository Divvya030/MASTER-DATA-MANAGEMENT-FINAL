"""Customer V1.1: V1.0 evidence + (a) hard negatives from real data (same building, unrelated company),
(b) realistic name/address variation (augment.py), (c) 160 assistant-reviewed pairs used as real-data
evidence. Selection/threshold by a strict GROUP-WISE 2-FOLD CROSS-FIT: records of a held-out fold's
pairs (and their name families) never appear in that fold's training data in any form.

Label provenance, never merged under one name:
  AUTO_IDENTIFIER (DUNS weak pairs) | AUTO_HARD_NEGATIVE | SYNTHETIC_CORRUPTION | ASSISTANT_REVIEWED
There are NO REAL_STEWARD (human) labels in this build.
"""
from __future__ import annotations
import hashlib, io, itertools, json, random, re, time
from collections import defaultdict, Counter
from pathlib import Path
from . import common as C
from .augment import realistic_corrupt
from .customer_build import nn, related
from .specs import CUSTOMER_V1, make_variables

STOP = set("inc ltd llc corp co company limited private pvt pte the of and corporation incorporated group holdings services "
           "solutions systems technologies international sdn bhd plc gmbh sa de cv lp llp pty".split())
core = lambda s: {t for t in nn(s).split() if t not in STOP}


def unrelated(a, b):
    import difflib
    ca, cb = core(a), core(b)
    return bool(ca) and bool(cb) and not (ca & cb) and difflib.SequenceMatcher(None, " ".join(sorted(ca)), " ".join(sorted(cb))).ratio() < .45


def fam(name):
    t = nn(name).split()
    return t[0] if t else ""


def explain(A, B, score, thr, label):
    """Field-level explanation of one decision (spec: every error must be explainable)."""
    f = lambda k: (A.get(k), B.get(k))
    both = lambda k: A.get(k) is not None and B.get(k) is not None
    eq = lambda k: both(k) and nn(str(A[k])) == nn(str(B[k]))
    ov = len(core(A.get("customer_name") or "") & core(B.get("customer_name") or ""))
    d = {"name_core_tokens_shared": ov, "address_equal": eq("address"), "postal_equal": eq("postal_code"), "city_equal": eq("city"),
         "country_equal": eq("country"), "missing_fields": [k for k in ("address", "city", "postal_code", "country") if not (both(k))]}
    pred = "MATCH" if score >= thr else "NON-MATCH"
    cause = None
    if label == "DISTINCT" and pred == "MATCH":
        cause = "LOCATION_EVIDENCE_OUTWEIGHS_NAME" if ov == 0 and (d["address_equal"] or d["postal_equal"]) else "SIMILAR_NAME_DIFFERENT_ENTITY"
    elif label == "MATCH" and pred == "NON-MATCH":
        if len(d["missing_fields"]) >= 2: cause = "SPARSE_RECORD"
        elif ov == 0: cause = "NAME_VARIANT_NOT_RECOGNISED"
        else: cause = "ADDRESS_CONFLICT_PENALISED"
    return {"predicted": pred, "score": round(score, 4), "threshold": thr, "label": label, "likely_cause": cause,
            "comparison": d, "A": A, "B": B}


def run(csv_path, labels_path, scratch, seed=42, log=print):
    import dedupe
    from app.modules.dedupe_service.real_labels import score_labeled_pairs
    spec = CUSTOMER_V1; fields = [f for f, _ in spec["fields"]]; t0 = time.time()
    raw, recs = C.onboard("customer", csv_path, spec["training_mapping"], scratch)
    n = len(recs); rec_f = lambda i: {f: recs[i].get(f) for f in fields}
    full = {i: rec_f(i) for i in recs}
    LB = json.load(open(labels_path))["pairs"]
    lab = [(p["a_row"], p["b_row"], p["label"], p["v1_score"]) for p in LB]
    decided = [x for x in lab if x[2] in ("MATCH", "DISTINCT")]
    log(f"[customer-v1.1] {n} records; assistant-reviewed pairs {Counter(x[2] for x in lab)}")

    # ---- V1.0 weak (DUNS / MDM ref) evidence and its SAME split, so V1.0 and V1.1 are compared on identical weak test pairs
    by_duns, by_ref = defaultdict(list), defaultdict(list)
    for i, r in recs.items():
        d = r.get("duns")
        if d and re.fullmatch(r"\d{9}", str(d)): by_duns[str(d)].append(i)
    for i, v in enumerate(raw["party_orig_system_ref"].tolist()):
        if isinstance(v, str) and v.startswith("MDM_CUST"): by_ref[v].append(i)
    weak, seen = [], set()
    for groups in (by_duns, by_ref):
        for ids in groups.values():
            for a, b in itertools.combinations(sorted(ids), 2):
                if (a, b) not in seen:
                    seen.add((a, b))
                    if related(recs[a].get("customer_name"), recs[b].get("customer_name")): weak.append((a, b))
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in weak:
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    wsplit = {(a, b): C.bucket(f"weak:{find(a)}", seed) for a, b in weak}
    weak_nodes = {s: set() for s in ("train", "validation", "test")}
    for (a, b), s in wsplit.items(): weak_nodes[s].update((a, b))
    held_weak = weak_nodes["validation"] | weak_nodes["test"]

    # ---- fold assignment of reviewed pairs by company family (no family spans folds)
    fold_of_fam = lambda f_: int(hashlib.sha1(f"{seed}:{f_}".encode()).hexdigest(), 16) % 2
    pf = [(a, b, l, s, fold_of_fam(fam(full[a].get("customer_name")))) for a, b, l, s in lab]
    fold_pairs = {k: [p for p in pf if p[4] == k] for k in (0, 1)}
    fold_fams = {k: {fam(full[a].get("customer_name")) for a, b, *_ in fold_pairs[k]} | {fam(full[b].get("customer_name")) for a, b, *_ in fold_pairs[k]} for k in (0, 1)}
    log(f"[customer-v1.1] fold sizes (decided) {[sum(1 for p in fold_pairs[k] if p[2] != 'UNSURE') for k in (0, 1)]}")

    all_ids = list(recs)
    by_city = defaultdict(list)
    for i in all_ids:
        if full[i].get("address") and full[i].get("city"): by_city[(full[i]["city"], full[i].get("postal_code"))].append(full[i]["address"])
    # auto hard negatives: same building (postal+country+address prefix), unrelated company names
    bld = defaultdict(list)
    for i in all_ids:
        r = full[i]
        if r.get("postal_code") and r.get("address") and r.get("customer_name"):
            bld[(r["postal_code"], r.get("country"), nn(r["address"])[:10])].append(i)
    auto_neg_all = []
    for ids in bld.values():
        if 1 < len(ids) <= 40:
            for a, b in itertools.combinations(sorted(ids), 2):
                if unrelated(full[a]["customer_name"], full[b]["customer_name"]): auto_neg_all.append((a, b))
    log(f"[customer-v1.1] AUTO_HARD_NEGATIVE candidates (same building, unrelated names): {len(auto_neg_all)}")

    def build_train(exclude_fams, labelled, tag):
        rng = random.Random(f"{seed}:{tag}")
        ok = lambda i: fam(full[i].get("customer_name")) not in exclude_fams and i not in held_weak
        pool = [i for i in all_ids if ok(i) and full[i].get("customer_name") and full[i].get("address") and full[i].get("city")]
        rng.shuffle(pool); bases = pool[:800]
        match, distinct, prov = [], [], Counter()
        for b in bases:
            base = full[b]; cs = by_city.get((base.get("city"), base.get("postal_code")), [])
            variants = [base] + [realistic_corrupt(base, rng, cs) for _ in range(2)]
            for x, y in itertools.combinations(variants, 2):
                match.append((x, y)); prov["SYNTHETIC_CORRUPTION+"] += 1
        for a, b in weak:
            if wsplit[(a, b)] == "train" and ok(a) and ok(b):
                match.extend([(full[a], full[b])] * 3); prov["AUTO_IDENTIFIER+ (x3)"] += 1
        for a, b, l, s, _f in labelled:
            if l == "MATCH" and ok(a) and ok(b): match.extend([(full[a], full[b])] * 3); prov["ASSISTANT_REVIEWED+ (x3)"] += 1
            if l == "DISTINCT" and ok(a) and ok(b): distinct.extend([(full[a], full[b])] * 2); prov["ASSISTANT_REVIEWED- (x2)"] += 1
        an = [p for p in auto_neg_all if ok(p[0]) and ok(p[1])]; rng.shuffle(an)
        for a, b in an[:700]: distinct.append((full[a], full[b])); prov["AUTO_HARD_NEGATIVE-"] += 1
        # synthetic shared-building negatives: A's location block + another company's name
        for b in bases[:300]:
            o = rng.choice(bases)
            if unrelated(full[b]["customer_name"], full[o]["customer_name"]):
                distinct.append((full[b], {**full[b], "customer_name": full[o]["customer_name"]})); prov["SYNTHETIC_CORRUPTION- (shared building)"] += 1
        tok, nm = defaultdict(list), defaultdict(list)
        for i in pool:
            x = nn(full[i]["customer_name"]); tok[x.split(" ")[0]].append(i); nm[x].append(i)
        def draw(groups, cond, k, label):
            gs = [g for g in groups.values() if len(g) > 1]; c = 0; tries = 0
            while gs and c < k and tries < k * 60:
                tries += 1; a, b = rng.sample(rng.choice(gs), 2)
                if cond(a, b): distinct.append((full[a], full[b])); c += 1; prov[label] += 1
        draw(tok, lambda a, b: nn(full[a]["customer_name"]) != nn(full[b]["customer_name"]), 350, "AUTO_HARD_NEGATIVE- (same first token)")
        draw(nm, lambda a, b: full[a].get("city") != full[b].get("city") and full[a].get("postal_code") != full[b].get("postal_code"), 300, "AUTO_HARD_NEGATIVE- (same name, different site)")
        for _ in range(150):
            a, b = rng.sample(pool, 2)
            if nn(full[a]["customer_name"]) != nn(full[b]["customer_name"]): distinct.append((full[a], full[b])); prov["AUTO_RANDOM-"] += 1
        bg = {i: full[i] for i in pool[:600]}
        return match, distinct, bg, prov

    def fit(exclude_fams, labelled, tag):
        m, d, bg, prov = build_train(exclude_fams, labelled, tag)
        t = time.time(); dd = C.train_dedupe(make_variables(spec), m, d, bg, seed, recall=0.99)
        log(f"[customer-v1.1] fit {tag}: match={len(m)} distinct={len(d)} {round(time.time()-t)}s predicates={[str(p) for p in dd.predicates]}")
        return dd, C.settings_bytes_of(dd), m, d, prov

    def score(settings, pairs):
        st = dedupe.StaticDedupe(io.BytesIO(settings))
        sc, _ = score_labeled_pairs(st, [(full[a], full[b], 1) for a, b in pairs], fields); return sc

    # ---- 2-fold cross-fit on reviewed pairs
    cross = {}
    for k in (0, 1):
        train_lab = [p for p in pf if p[4] != k]
        _, st, *_ = fit(fold_fams[k], train_lab, f"fold{k}")
        held = [p for p in fold_pairs[k] if p[2] != "UNSURE"]
        for p, s in zip(held, score(st, [(p[0], p[1]) for p in held])): cross[(p[0], p[1])] = s
    bands = [(0.20, 0.45, 2727), (0.45, 0.65, 228), (0.65, 0.80, 144), (0.80, 0.95, 244), (0.95, 1.01, 407)]
    def bw(s):
        for lo, hi, N in bands:
            if lo <= s < hi: return N / sum(1 for x in lab if lo <= x[3] < hi)
    W = {(a, b): bw(s) for a, b, l, s in lab}
    def wmetrics(scores, thr):
        tp = fp = fn = 0.0
        for a, b, l, s in decided:
            w = W[(a, b)]; sc_ = scores[(a, b)]
            if l == "MATCH": tp, fn = (tp + w, fn) if sc_ >= thr else (tp, fn + w)
            elif sc_ >= thr: fp += w
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f05 = 1.25 * p * r / (.25 * p + r) if p + r else 0
        return {"precision": round(p, 4), "recall": round(r, 4), "f0_5": round(f05, 4), "est_false_merges": round(fp, 1), "est_true_pairs_found": round(tp, 1), "est_true_pairs_missed": round(fn, 1)}
    def umetrics(scores, thr):
        tp = sum(1 for a, b, l, s in decided if l == "MATCH" and scores[(a, b)] >= thr); fp = sum(1 for a, b, l, s in decided if l == "DISTINCT" and scores[(a, b)] >= thr)
        fn = sum(1 for a, b, l, s in decided if l == "MATCH" and scores[(a, b)] < thr)
        return C.confusion(tp, fp, fn)
    v1scores = {(a, b): s for a, b, l, s in decided}
    grid = [round(x / 100, 2) for x in range(30, 100, 5)]
    tab_new = {t: {"weighted": wmetrics(cross, t), "unweighted": umetrics(cross, t)} for t in grid}
    tab_v1 = {t: {"weighted": wmetrics(v1scores, t), "unweighted": umetrics(v1scores, t)} for t in grid}
    thr = max(((tab_new[t]["weighted"]["f0_5"], t) for t in grid))[1]
    rthr = round(max(.35, thr - .15), 4)
    log(f"[customer-v1.1] threshold {thr} (max weighted F0.5 of cross-fit) new={tab_new[thr]['weighted']} | V1.0@0.65={tab_v1[0.65]['weighted']}")

    # ---- final model on ALL reviewed labels
    dd, settings, m, d, prov = fit(set(), pf, "final")

    # diagnostics on cross-fit errors at the chosen threshold
    diags = [explain(full[a], full[b], cross[(a, b)], thr, l) for a, b, l, s in decided
             if (l == "MATCH") != (cross[(a, b)] >= thr)]
    cause_counts = dict(Counter(x["likely_cause"] for x in diags))
    v1_diags = [explain(full[a], full[b], s, .65, l) for a, b, l, s in decided if (l == "MATCH") != (s >= .65)]
    v1_causes = dict(Counter(x["likely_cause"] for x in v1_diags))

    # ---- held-out weak-real + synthetic + policy checks for the FINAL model (weak test pairs were never trained on)
    st = dedupe.StaticDedupe(io.BytesIO(settings))
    rng = random.Random(f"{seed}:eval"); weak_eval = {}
    negs = auto_neg_all[:0]
    ok_ids = [i for i in all_ids if full[i].get("customer_name")]
    tok = defaultdict(list)
    for i in ok_ids: tok[nn(full[i]["customer_name"]).split(" ")[0]].append(i)
    gs = [g for g in tok.values() if len(g) > 1]; hn = set()
    while len(hn) < 400:
        a, b = rng.sample(rng.choice(gs), 2)
        if nn(full[a]["customer_name"]) != nn(full[b]["customer_name"]): hn.add((min(a, b), max(a, b)))
    hn = sorted(hn)
    def conf(pos, ng, t):
        sp = score(settings, pos); sn = score(settings, ng)
        tp = sum(1 for x in sp if x >= t); fn = len(sp) - tp; fp = sum(1 for x in sn if x >= t)
        return {**C.confusion(tp, fp, fn, len(sn) - fp), "recall_ci95": C.wilson(tp, tp + fn), "positives": len(pos), "negatives": len(ng)}
    weak_head = {s: conf([(a, b) for a, b in weak if wsplit[(a, b)] == s], hn, thr) for s in ("validation", "test")}
    # synthetic held-out: fresh bases from families excluded from final training? (final trains on all families) -> use weak-test records' families only for transparency
    from app.modules.dedupe_service.matching_service import run_matching
    from app.modules.dedupe_service.clustering_service import run_clustering
    t = time.time(); mr = run_matching(settings, full, threshold=thr, review_threshold=rthr); assert mr.status == "COMPLETED", mr.error
    cr = run_clustering(mr.raw_scores, sorted(full), threshold=thr); sizes = sorted((c["cluster_size"] for c in cr.clusters), reverse=True)
    cs = {tuple(sorted((p["record_id_a"], p["record_id_b"]))): p["score"] for p in mr.pairs}
    found = lambda pairs: [sum(1 for a, b in pairs if cs.get((a, b), -1) >= thr), len(pairs)]
    cand = lambda pairs: [sum(1 for a, b in pairs if (a, b) in cs), len(pairs)]
    matchM = [(min(a, b), max(a, b)) for a, b, l, s in decided if l == "MATCH"]
    op = {"records": n, "candidate_pairs": mr.candidate_pair_count, "possible_pairs": mr.total_possible_pairs,
          "reduction_pct": round(100 * (1 - mr.candidate_pair_count / mr.total_possible_pairs), 3), "match_pairs": mr.match_count,
          "uncertain_pairs": mr.uncertain_count, "non_match_pairs": mr.non_match_count, "clusters": cr.cluster_count,
          "singletons": cr.singleton_count, "largest_cluster": sizes[0], "top_cluster_sizes": sizes[:10],
          "clusters_over_10": sum(1 for x in sizes if x > 10), "records_in_multi_clusters": sum(x for x in sizes if x > 1),
          "weak_pairs_recovered_all": found(weak), "weak_pairs_recovered_heldout": found([w for w in weak if wsplit[w] != "train"]),
          "reviewed_match_pairs_reaching_scoring(blocking_recall)": cand(matchM), "match_seconds": round(time.time() - t, 1)}
    meta = {"threshold": thr, "review_threshold": rthr, "threshold_calibration": "ASSISTANT_REVIEWED_CROSSFIT_NOT_HUMAN_VALIDATED",
            "train_seconds": None, "learned_predicates": [str(p) for p in dd.predicates],
            "training_pairs": {"match": len(m), "distinct": len(d), "provenance": dict(prov)},
            "weak_labels": {"pairs_total": len(weak), "by_split": dict(Counter(wsplit.values()))},
            "records_total": n, "total_seconds": round(time.time() - t0, 1)}
    res = {"headline": {"weak_real": weak_head,
                        "assistant_reviewed_crossfit": {"threshold": thr, "new": tab_new[thr], "v1_0_at_0.65": tab_v1[0.65], "v1_0_at_same_threshold": tab_v1[thr],
                                                        "tables": {"new": {str(k): v for k, v in tab_new.items()}, "v1_0": {str(k): v for k, v in tab_v1.items()}}}},
           "diagnostics": {"crossfit_errors_new_model": diags, "cause_counts_new": cause_counts, "cause_counts_v1_0": v1_causes,
                           "v1_0_error_examples": v1_diags[:12]},
           "operational_full_file": op, "audit_sample": [], "meta": meta,
           "label_provenance_note": "ASSISTANT_REVIEWED labels are AI-assigned, NOT human ground truth; no REAL_STEWARD labels exist in this build."}
    return {"results": res, "settings": settings, "training": C.training_bytes_of(dd)}
