"""Re-calibrate the Customer V1 match threshold on the assistant-reviewed pairs (model unchanged).

Method: band-weighted precision/recall/F0.5 (the reviewed sample is stratified by V1 score band, so each
pair stands for pop_band/sample_band candidate pairs), threshold = argmax weighted F0.5 (precision-leaning:
a false merge contaminates a golden record), with a bootstrap over the reviewed pairs to show how stable
the choice is. Also writes explainable diagnostics for every error and refreshes whole-file statistics.
Labels are ASSISTANT_REVIEWED (AI-assigned), NOT human ground truth.
"""
import json, random, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from scripts.v1_build import common as C                    # noqa: E402
from scripts.v1_build.customer_v11_build import explain      # noqa: E402
from scripts.v1_build.specs import CUSTOMER_V1               # noqa: E402

BANDS = [(0.20, 0.45, 2727), (0.45, 0.65, 228), (0.65, 0.80, 144), (0.80, 0.95, 244), (0.95, 1.01, 407)]  # population = candidate pairs per band, from the sampling run
GRID = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def main(csv_path, log=print):
    d = HERE / "models_bundled" / "customer" / "v1"
    L = json.load(open(d / "assistant_reviewed_labels.json"))["pairs"]
    cnt = {lo: sum(1 for x in L if lo <= x["v1_score"] < hi) for lo, hi, _ in BANDS}
    def w(s):
        for lo, hi, N in BANDS:
            if lo <= s < hi: return N / cnt[lo]
    dec = [x for x in L if x["label"] != "UNSURE"]
    def met(rows, t):
        tp = sum(w(x["v1_score"]) for x in rows if x["label"] == "MATCH" and x["v1_score"] >= t)
        fn = sum(w(x["v1_score"]) for x in rows if x["label"] == "MATCH" and x["v1_score"] < t)
        fp = sum(w(x["v1_score"]) for x in rows if x["label"] == "DISTINCT" and x["v1_score"] >= t)
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        return {"precision": round(p, 4), "recall": round(r, 4), "f0_5": round(1.25 * p * r / (.25 * p + r), 4) if p + r else 0,
                "est_false_merges": round(fp, 1), "est_true_pairs_found": round(tp, 1), "est_true_pairs_missed": round(fn, 1)}
    table = {str(t): met(dec, t) for t in GRID}
    thr = max(GRID, key=lambda t: (table[str(t)]["f0_5"], t))
    rng = random.Random(1); boot = Counter()
    for _ in range(500):
        s = [rng.choice(dec) for _ in dec]
        boot[max(GRID, key=lambda t: (met(s, t)["f0_5"], t))] += 1
    rthr = round(thr - .15, 4)
    log(f"threshold {thr}  weighted: {table[str(thr)]}  vs 0.65: {table['0.65']}  bootstrap argmax {dict(sorted(boot.items()))}")

    diags = [explain(x["A"], x["B"], x["v1_score"], thr, x["label"]) for x in dec if (x["label"] == "MATCH") != (x["v1_score"] >= thr)]
    causes = Counter(x["likely_cause"] for x in diags)
    for x in diags: x["rows"] = None

    # whole-file statistics at the new threshold (production matching + clustering code)
    spec = CUSTOMER_V1; fields = [f for f, _ in spec["fields"]]
    raw, recs = C.onboard("customer", csv_path, spec["training_mapping"], Path("/tmp/v1_recal_scratch"))
    full = {i: {f: recs[i].get(f) for f in fields} for i in recs}
    from app.modules.dedupe_service.matching_service import run_matching
    from app.modules.dedupe_service.clustering_service import run_clustering
    settings = (d / "model.settings").read_bytes()
    mr = run_matching(settings, full, threshold=thr, review_threshold=rthr); assert mr.status == "COMPLETED", mr.error
    cr = run_clustering(mr.raw_scores, sorted(full), threshold=thr); sizes = sorted((c["cluster_size"] for c in cr.clusters), reverse=True)
    op = {"records": len(full), "candidate_pairs": mr.candidate_pair_count, "match_pairs": mr.match_count, "uncertain_pairs": mr.uncertain_count,
          "non_match_pairs": mr.non_match_count, "clusters": cr.cluster_count, "singletons": cr.singleton_count, "largest_cluster": sizes[0],
          "top_cluster_sizes": sizes[:10], "clusters_over_10": sum(1 for s in sizes if s > 10)}

    ev = json.load(open(d / "evaluation.json"))
    ev["assistant_reviewed_calibration"] = {
        "label_source": "ASSISTANT_REVIEWED (AI-assigned, NOT human ground truth)", "pairs_total": len(L), "decided": len(dec),
        "counts": dict(Counter(x["label"] for x in L)), "weighting": "stratified by V1 score band; each pair weighted pop_band/sample_band",
        "chosen_threshold": thr, "review_threshold": rthr, "previous_threshold": 0.65, "table": table,
        "bootstrap_argmax_threshold_frequency": {str(k): v for k, v in sorted(boot.items())},
        "errors_at_chosen_threshold": len(diags), "error_causes": dict(causes), "error_examples": diags[:16],
        "operational_at_new_threshold": op,
        "caveats": ["Small sample (111 decided pairs); the interval around every figure is wide.",
                    "Labels were assigned by the AI assistant, not by a human steward; a human audit is required before treating them as ground truth.",
                    "Recall is measured only among pairs the blocking step already proposed with score >= 0.20."]}
    (d / "evaluation.json").write_text(json.dumps(ev, indent=2, default=str))
    md = json.load(open(d / "metadata.json"))
    md.update({"threshold": thr, "review_threshold": rthr, "threshold_calibration": "ASSISTANT_REVIEWED_WEIGHTED_F0_5_NOT_HUMAN_VALIDATED"})
    md["training_summary"]["threshold"] = thr; md["training_summary"]["review_threshold"] = rthr
    md["training_summary"]["threshold_calibration"] = md["threshold_calibration"]
    (d / "metadata.json").write_text(json.dumps(md, indent=2, default=str))
    log("operational at new threshold: " + json.dumps(op))


if __name__ == "__main__":
    main(sys.argv[1])
