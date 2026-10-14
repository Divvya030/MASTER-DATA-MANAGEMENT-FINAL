"""Build + evaluate Customer V1. customer.csv has NO ground-truth entity ids. Evidence used:
 * WEAK POSITIVES  - records sharing a valid 9-digit D-U-N-S, or an MDM_CUST* source reference.
                     Weak labels, never proven truth; pairs whose names share nothing are
                     excluded as 'conflicts'.
 * SYNTHETIC       - controlled corruptions of real records (project's `corrupt_record`).
 * POLICY NEGATIVES- same name at clearly different locations (different city AND postal code)
                     are treated as DISTINCT sites. A business-policy assumption a v2 may change.
"""
from __future__ import annotations
import difflib, io, itertools, random, re, time
from collections import defaultdict
from . import common as C
from .specs import CUSTOMER_V1, make_variables

nn = lambda s: re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()
def related(a, b):
    return bool(set(nn(a).split()) & set(nn(b).split())) or difflib.SequenceMatcher(None, nn(a), nn(b)).ratio() >= .5


def run(csv_path, scratch, seed=42, log=print):
    from app.modules.dedupe_service.training_data import corrupt_record
    import dedupe
    spec = CUSTOMER_V1; fields = [f for f, _ in spec["fields"]]; t0 = time.time()
    raw, recs = C.onboard("customer", csv_path, spec["training_mapping"], scratch)
    n = len(recs); log(f"[customer] {n} records via app pipeline")
    rec_f = lambda i: {f: recs[i].get(f) for f in fields}

    by_duns, by_ref = defaultdict(list), defaultdict(list)
    for i, r in recs.items():
        d = r.get("duns")
        if d and re.fullmatch(r"\d{9}", str(d)): by_duns[str(d)].append(i)
    for i, v in enumerate(raw["party_orig_system_ref"].tolist()):
        if isinstance(v, str) and v.startswith("MDM_CUST"): by_ref[v].append(i)
    weak, conflicts, seen = [], [], set()
    for kind, groups in (("DUNS", by_duns), ("MDM_REF", by_ref)):
        for ids in groups.values():
            for a, b in itertools.combinations(sorted(ids), 2):
                if (a, b) in seen: continue
                seen.add((a, b))
                if related(recs[a].get("customer_name"), recs[b].get("customer_name")): weak.append((a, b, kind))
                else: conflicts.append({"kind": kind, "a": recs[a].get("customer_name"), "b": recs[b].get("customer_name")})
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b, _ in weak:
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    wsplit = {(a, b): C.bucket(f"weak:{find(a)}", seed) for a, b, _ in weak}
    wnodes = {s: set() for s in ("train", "validation", "test")}
    for (a, b), s in wsplit.items(): wnodes[s].update((a, b))
    held = wnodes["validation"] | wnodes["test"]
    log(f"[customer] weak positive pairs {len(weak)} (conflicts excluded {len(conflicts)})")

    def split_of(i):
        for s in ("validation", "test"):
            if i in wnodes[s]: return s
        return C.bucket("name:" + nn(recs[i].get("customer_name")), seed)
    pool = defaultdict(list)
    for i in recs: pool[split_of(i)].append(i)
    complete = lambda r: bool(r.get("customer_name") and r.get("city") and r.get("address"))

    rng = random.Random(seed)
    NB = {"train": 700, "validation": 300, "test": 300}
    synth = {}
    for s in NB:
        cand = [i for i in pool[s] if complete(recs[i]) and (s != "train" or i not in held)]
        rng.shuffle(cand); records, truth = {}, {}
        nid = n + 1 + {"train": 0, "validation": 100000, "test": 200000}[s]
        for b in sorted(cand[:NB[s]]):
            base = rec_f(b); records[b] = base; truth[b] = f"E{b}"
            for _ in range(2): records[nid] = corrupt_record(base, rng); truth[nid] = f"E{b}"; nid += 1
        synth[s] = (records, truth)

    def negatives(s, kh, ks, kr, r):
        ids = pool[s]; tok, nm = defaultdict(list), defaultdict(list)
        for i in ids:
            x = nn(recs[i].get("customer_name"))
            if x: tok[x.split(" ")[0]].append(i); nm[x].append(i)
        out = {"hard_first_token": set(), "same_name_diff_site": set(), "random": set()}
        def draw(groups, key, k, cond):
            gs = [g for g in groups.values() if len(g) > 1]; tries = 0
            while len(out[key]) < k and gs and tries < k * 60:
                tries += 1; a, b = r.sample(r.choice(gs), 2)
                if cond(a, b): out[key].add((min(a, b), max(a, b)))
        dn = lambda a, b: nn(recs[a].get("customer_name")) != nn(recs[b].get("customer_name"))
        def ds(a, b):
            x, y = recs[a], recs[b]
            return bool(x.get("city") and y.get("city") and x["city"] != y["city"] and x.get("postal_code") and y.get("postal_code") and x["postal_code"] != y["postal_code"])
        draw(tok, "hard_first_token", kh, dn); draw(nm, "same_name_diff_site", ks, ds)
        while len(out["random"]) < kr:
            a, b = r.sample(ids, 2)
            if dn(a, b): out["random"].add((min(a, b), max(a, b)))
        return {k: sorted(v) for k, v in out.items()}
    neg = {s: negatives(s, *k, random.Random(f"{seed}{s}")) for s, k in
           (("train", (400, 400, 350)), ("validation", (150, 120, 120)), ("test", (150, 120, 120)))}
    log("[customer] negatives " + str({s: {k: len(v) for k, v in d.items()} for s, d in neg.items()}))

    sr, st = synth["train"]; byent = defaultdict(list)
    for r_, e in st.items(): byent[e].append(r_)
    match = [(sr[a], sr[b]) for ids in byent.values() for a, b in itertools.combinations(ids, 2)]
    n_syn = len(match)
    wtrain = [(a, b) for a, b, _ in weak if wsplit[(a, b)] == "train"]
    for a, b in wtrain: match.extend([(rec_f(a), rec_f(b))] * 3)   # the only REAL positive signal: up-weighted x3
    distinct = [(rec_f(a), rec_f(b)) for ps in neg["train"].values() for a, b in ps]
    log(f"[customer] training pairs match={len(match)} (synthetic {n_syn} + weak-real {len(wtrain)}x3) distinct={len(distinct)}")
    bgids = [i for i in pool["train"] if i not in held]; random.Random(seed).shuffle(bgids)
    t = time.time()
    dd = C.train_dedupe(make_variables(spec), match, distinct, {i: rec_f(i) for i in bgids[:600]}, seed, recall=0.99)
    train_s = round(time.time() - t, 1); settings = C.settings_bytes_of(dd)
    log(f"[customer] trained {train_s}s predicates={[str(p) for p in dd.predicates]}")

    grid = C.threshold_grid(); syn_tab, cache = {}, {}
    for s in ("validation", "test"):
        rs, ts = synth[s]; r_, tp, cand = C.pairwise_end_to_end(settings, rs, ts)
        tab, ct = C.score_grid_metrics(tp, cand, grid)
        syn_tab[s] = {"records": len(rs), "entities": len(set(ts.values())), "true_pairs": len(tp), "candidate_pairs": len(cand),
                      "blocking_recall": round(ct / len(tp), 4), "threshold_table": {str(k): v for k, v in tab.items()}}
        cache[s] = (r_, rs, ts)
    from app.modules.dedupe_service.real_labels import score_labeled_pairs
    static = dedupe.StaticDedupe(io.BytesIO(settings))
    def sc(pos, ng):
        return score_labeled_pairs(static, [(rec_f(a), rec_f(b), 1) for a, b in pos] + [(rec_f(a), rec_f(b), 0) for a, b in ng], fields)
    wr, pn = {}, {}
    for s in ("validation", "test"):
        pos = [(a, b) for a, b, _ in weak if wsplit[(a, b)] == s]; negs = [p for ps in neg[s].values() for p in ps]
        wr[s] = (sc(pos, negs), len(pos), len(negs)); site = neg[s]["same_name_diff_site"]; pn[s] = (sc([], site)[0], len(site))
    # threshold = argmax over the grid of the mean of (synthetic validation F1, weak-real validation F1),
    # ties toward the HIGHER threshold. Chosen on VALIDATION only; the weak-real half keeps the choice
    # honest about real data instead of letting saturated synthetic F1 push the threshold up.
    vt = syn_tab["validation"]["threshold_table"]
    (vsc, vys), _, _ = wr["validation"]
    def wf1(t):
        tp = sum(1 for a, y in zip(vsc, vys) if y and a >= t); fp = sum(1 for a, y in zip(vsc, vys) if not y and a >= t)
        fn = sum(1 for a, y in zip(vsc, vys) if y and a < t)
        return C.confusion(tp, fp, fn)["f1"]
    combo = {k: (v["f1"] + wf1(float(k))) / 2 for k, v in vt.items()}
    thr = max(.60, max(((c, float(k)) for k, c in combo.items()))[1]); rthr = round(max(0.0, thr - .15), 4)
    bestf = combo[str(thr)]
    log(f"[customer] threshold {thr} (mean of synthetic+weak-real validation F1 = {bestf:.4f})")
    def conf(scores, ys):
        tp = sum(1 for a, y in zip(scores, ys) if y and a >= thr); fp = sum(1 for a, y in zip(scores, ys) if not y and a >= thr)
        fn = sum(1 for a, y in zip(scores, ys) if y and a < thr); tn = sum(1 for a, y in zip(scores, ys) if not y and a < thr)
        return {**C.confusion(tp, fp, fn, tn), "recall_ci95": C.wilson(tp, tp + fn)}
    head = {"synthetic": {}, "weak_real": {}, "policy_negatives": {}}
    for s in ("validation", "test"):
        row = dict(syn_tab[s]["threshold_table"][str(thr)])
        row["precision_ci95"] = C.wilson(row["true_positives"], row["true_positives"] + row["false_positives"])
        row["recall_ci95"] = C.wilson(row["true_positives"], row["true_positives"] + row["false_negatives"])
        row["blocking_recall"] = syn_tab[s]["blocking_recall"]; head["synthetic"][s] = row
        (scores, ys), npos, nneg = wr[s]; head["weak_real"][s] = {**conf(scores, ys), "positives": npos, "negatives": nneg}
        ps, k = pn[s]; fm = sum(1 for x in ps if x >= thr)
        head["policy_negatives"][s] = {"pairs": k, "false_merges": fm, "false_merge_rate": round(fm / k, 4) if k else None, "false_merge_rate_ci95": C.wilson(fm, k)}
    r_, rs, ts = cache["test"]; clm = C.cluster_metrics(r_, rs, ts, thr)

    from app.modules.dedupe_service.matching_service import run_matching
    from app.modules.dedupe_service.clustering_service import run_clustering
    full = {i: rec_f(i) for i in recs}; t = time.time()
    mr = run_matching(settings, full, threshold=thr, review_threshold=rthr); assert mr.status == "COMPLETED", mr.error
    cr = run_clustering(mr.raw_scores, sorted(full), threshold=thr); sizes = sorted((c["cluster_size"] for c in cr.clusters), reverse=True)
    cs = {tuple(sorted((p["record_id_a"], p["record_id_b"]))): p["score"] for p in mr.pairs}
    rec_of = lambda sel: [sum(1 for a, b, _ in sel if cs.get((a, b), -1) >= thr), len(sel)]
    matched = [p for p in mr.pairs if p["decision"] == "MATCH"]; r2 = random.Random(seed)
    audit = [{"score": p["score"], **{f"a_{f}": recs[p["record_id_a"]].get(f) for f in fields}, **{f"b_{f}": recs[p["record_id_b"]].get(f) for f in fields}}
             for p in r2.sample(matched, min(40, len(matched)))]
    op = {"records": n, "candidate_pairs": mr.candidate_pair_count, "possible_pairs": mr.total_possible_pairs,
          "reduction_pct": round(100 * (1 - mr.candidate_pair_count / mr.total_possible_pairs), 3), "match_pairs": mr.match_count,
          "uncertain_pairs": mr.uncertain_count, "non_match_pairs": mr.non_match_count, "clusters": cr.cluster_count,
          "singletons": cr.singleton_count, "largest_cluster": sizes[0], "top_cluster_sizes": sizes[:10],
          "clusters_over_10": sum(1 for x in sizes if x > 10), "records_in_multi_clusters": sum(x for x in sizes if x > 1),
          "weak_pairs_recovered_all": rec_of(weak), "weak_pairs_recovered_heldout": rec_of([w for w in weak if wsplit[(w[0], w[1])] != "train"]),
          "match_seconds": round(time.time() - t, 1)}
    meta = {"threshold": thr, "review_threshold": rthr, "threshold_calibration": "PROVISIONAL_SYNTHETIC_AND_WEAK_LABELS",
            "train_seconds": train_s, "learned_predicates": [str(p) for p in dd.predicates],
            "training_pairs": {"match": len(match), "synthetic_match": n_syn, "weak_real_match_x3": len(wtrain), "distinct": len(distinct),
                               "distinct_by_kind": {k: len(v) for k, v in neg["train"].items()}},
            "weak_labels": {"pairs_total": len(weak), "by_split": {s: sum(1 for v in wsplit.values() if v == s) for s in ("train", "validation", "test")},
                            "duns_groups": sum(1 for v in by_duns.values() if len(v) > 1), "conflicts_excluded": len(conflicts)},
            "records_total": n, "synthetic_bases": {s: len(v[0]) for s, v in synth.items()}, "total_seconds": round(time.time() - t0, 1)}
    return {"results": {"headline": head, "synthetic_tables": syn_tab, "cluster_test": clm, "operational_full_file": op,
                        "audit_sample": audit, "duns_conflicts": conflicts[:30], "meta": meta},
            "settings": settings, "training": C.training_bytes_of(dd)}
