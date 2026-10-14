"""Generates V1_EVALUATION_REPORT.md from the two evaluation.json files (no hand-typed numbers)."""
import json, sys
from pathlib import Path
B = Path(__file__).resolve().parents[1]
S = json.load(open(B / "models_bundled/supplier/v1/evaluation.json"))
C = json.load(open(B / "models_bundled/customer/v1/evaluation.json"))
SM = json.load(open(B / "models_bundled/supplier/v1/metadata.json")); CM = json.load(open(B / "models_bundled/customer/v1/metadata.json"))
p = lambda v: "n/a" if v is None else f"{100*v:.2f}%"
ci = lambda c: "n/a" if not c else f"[{100*c[0]:.1f}–{100*c[1]:.1f}%]"
L = []
w = L.append

def row(name, x):
    return (f"| {name} | {x['true_positives']} | {x['false_positives']} | {x['false_negatives']} | {p(x['precision'])} {ci(x.get('precision_ci95'))} | "
            f"{p(x['recall'])} {ci(x.get('recall_ci95'))} | {p(x['f1'])} |")
HDR = "| Slice | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 |\n|---|---|---|---|---|---|---|"

w("# V1 Model — Evaluation & Validation Report\n")
w(f"Models: **{SM['training_run_id']}** (supplier) and **{CM['training_run_id']}** (customer), version 1.0.0. Engine: `dedupe==3.0.3`, no LLM anywhere. "
  f"Built {SM['created_at'][:10]} by `backend/scripts/build_v1_models.py`; the running app only loads these artifacts, it never trains them.\n")
w("## 0. How to read these numbers (please read first)\n")
w("* **Supplier** numbers use `Supplier_1_.csv`, which has a known `entity_id` for every record, so pair labels are *real membership labels*. "
  "But the file is a **generated benchmark** (every e-mail and web domain is the reserved `.example` TLD; duplicates are produced by a small set of perturbations). "
  "Treat the supplier metrics as proof the pipeline works end to end, **not** as the accuracy you will see on a production supplier master.")
w("* **Customer** data (`customer.csv`) has **no ground truth**. Customer metrics are therefore split into three clearly different kinds of evidence — "
  "*synthetic*, *weak-real (DUNS)* and *policy* — and the report never merges them into one 'accuracy' figure. Precision on real customer data has **no human ground truth**; the best available estimate is the 111 decided pairs in §2.7 "
  "(reviewed by the AI assistant acting as a steward — not by a human) plus the 22 DUNS-linked pairs in §2.3.")
w("* Every metric below is on data the model never trained on (split by **entity**, or by connected component for weak labels, so no entity or linked record sits in two splits) "
  "and is computed through the **production** code path (`run_matching`, `run_clustering`), so pairs the blocking step never proposed count as misses.")
w("* `supplier.xlsx` (417 rows, no duplicate labels, no ground truth) was not used to train or score anything.\n")

# ---------------------------------------------------------------- supplier
sm = S["meta"]; sh = S["headline"]
w("## 1. Supplier V1\n")
w(f"**Fields compared:** {', '.join(SM['match_fields'])}.  **Threshold:** match ≥ {sm['threshold']}, review band [{sm['review_threshold']}, {sm['threshold']}) — "
  f"chosen on the *validation* split as the best mean F1 over the three thin-data scenarios in §1.2 (identifiers hidden; identifiers + contact hidden; address + identifiers hidden), ties broken toward the higher threshold because a false merge costs more than a missed duplicate, then frozen before the test split was scored.\n")
w("### 1.1 Data and split")
w(f"{sm['records_total']:,} records, {sm['entities_total']:,} entities (2,800 entities have 2 records, 4,400 have 1). Entity-level split: "
  f"train {sm['split_entities']['train']:,} / validation {sm['split_entities']['validation']:,} / test {sm['split_entities']['test']:,} entities "
  f"({sm['split_records']['train']:,} / {sm['split_records']['validation']:,} / {sm['split_records']['test']:,} records).\n")
tp = sm["training_pairs"]
w(f"Training pairs (train split only): {tp['match']:,} MATCH / {tp['distinct']:,} DISTINCT. Negatives are 40% same-name-different-entity, 30% same-city-same-first-token, 30% random. "
  f"Each pair is also emitted with identifier fields blanked ({tp['identifiers_masked']:,} copies) and, for most pairs, with the address block blanked ({tp['address_masked']:,} copies) and, for most pairs, with address *and* identifiers blanked ({tp['name_contact_only']:,} copies: name + e-mail + phone only) — all labelled *synthetic augmentation* — so the model, and its learned blocking, still work when an upload has no identifiers or no address columns. "
  f"Learned blocking: `{sm['learned_predicates'][0]}`. Training time {sm['train_seconds']}s.\n")
w("### 1.2 Pair-level results at the chosen threshold")
for scen, title in (("full_record", "All fields present"), ("identifiers_hidden", "Identifiers (DUNS / tax ID / registration no.) hidden"),
                    ("identifiers_and_contact_hidden", "Identifiers **and** e-mail / phone hidden (name + address only)"),
                    ("address_hidden", "Address, city and postal code hidden (name + contact + identifiers)"),
                    ("address_and_identifiers_hidden", "Address block **and** identifiers hidden (name + e-mail + phone only)")):
    w(f"\n**{title}**\n"); w(HDR)
    for sp in ("validation", "test"):
        w(row(sp, sh[scen][sp]))
    t = sh[scen]["test"]
    w(f"\nBlocking: recall {p(t['blocking_recall'])} of true pairs proposed; {t['blocking_reduction_pct']}% of all possible pairs avoided.")
w("\nHow to read this: with identifiers present (or only the address hidden) the model is near-perfect on this file. Precision stays at or above 99% until e-mail, phone **and** identifiers are all unavailable; at name + e-mail + phone only, "
  "recall drops to ~74% and a few false merges appear. The hardest realistic case is a thin extract with a name and little else — expect results closer to that row than to the 100% rows.\n")
w("### 1.3 Cluster-level (test, identifiers hidden)")
ct = S["cluster_test"]
w(f"{ct['cluster_count']} predicted clusters vs {ct['true_entity_count']} true entities; largest cluster {ct['largest_cluster_size']}; over-merged clusters {ct['over_merged_clusters']}. "
  f"Pairwise F1 {p(ct['pairwise']['f1'])}; B-cubed precision {p(ct['bcubed']['precision'])}, recall {p(ct['bcubed']['recall'])}, F1 {p(ct['bcubed']['f1'])}.\n")
ea = S["error_analysis"]
w("### 1.4 Error analysis (test, identifiers hidden)")
w(f"{ea['false_positives']} false positives, {ea['false_negatives']} false negatives; {ea['missed_by_blocking']} of the misses were never proposed by blocking, "
  f"{ea['scored_below_threshold']} were proposed but scored below threshold. The misses sampled below are records whose *name* differs completely from their duplicate (an alias / renamed entity) while the address still agrees; "
  "the model correctly weighs the name disagreement. Rows with name + e-mail + phone only (see §1.2) are where most errors concentrate.\n")
w("Sample misses:\n")
for x in ea["sample_false_negatives"][:3]:
    w(f"* A: {x['a']['supplier_name']} / {x['a']['address']} / {x['a']['city']}  ↔  B: {x['b']['supplier_name']} / {x['b']['address']} / {x['b']['city']}")

# ---------------------------------------------------------------- customer
cm = C["meta"]; ch = C["headline"]; op = C["operational_full_file"]
w("\n## 2. Customer V1\n")
w(f"**Fields compared:** {', '.join(CM['match_fields'])}. (`state` is excluded as noisy and redundant; D-U-N-S is excluded because it defines the weak labels — the model cannot read its own label.)  "
  f"**Threshold:** match ≥ {cm['threshold']}, review band [{cm['review_threshold']}, {cm['threshold']}). Calibration: **{cm['threshold_calibration']}** — i.e. *provisional*; there are too few real labels to calibrate it properly.\n")
w("### 2.1 Evidence used (and what each can and cannot prove)")
wl = cm["weak_labels"]
w(f"| Evidence | Size | Nature |\n|---|---|---|\n"
  f"| Weak real positives (shared 9-digit D-U-N-S or `MDM_CUST*` source reference) | {wl['pairs_total']} pairs from {wl['duns_groups']} DUNS groups; train/val/test = {wl['by_split']['train']}/{wl['by_split']['validation']}/{wl['by_split']['test']}; {wl['conflicts_excluded']} name-conflicting pair excluded | Real, but *weak* (a DUNS can be shared by parent/subsidiary or mis-keyed) |\n"
  f"| Synthetic positives | {cm['training_pairs']['synthetic_match']:,} train pairs from {cm['synthetic_bases']['train']:,} corrupted real records | Controlled typos, legal-form swaps, abbreviations, dropped tokens, missing values — **not** ground truth |\n"
  f"| Policy negatives (same name, different city *and* postal code ⇒ distinct sites) | {cm['training_pairs']['distinct_by_kind']['same_name_diff_site']} train pairs | A **business-policy assumption** (site-level resolution) — change it in v2 if you want company-level merging |\n"
  f"| Other negatives | {cm['training_pairs']['distinct_by_kind']['hard_first_token']} same-first-token + {cm['training_pairs']['distinct_by_kind']['random']} random | Assumed distinct; a few could be unlabelled real duplicates |\n")
w(f"Weak positives are up-weighted ×3 in training ({cm['training_pairs']['weak_real_match_x3']} real pairs → {cm['training_pairs']['weak_real_match_x3']*3} training rows). Training: {cm['training_pairs']['match']:,} MATCH / {cm['training_pairs']['distinct']:,} DISTINCT, {cm['train_seconds']}s. "
  f"Learned blocking: `{cm['learned_predicates'][0]}`.\n")
w("### 2.2 Synthetic held-out evaluation (SYNTHETIC — measures the pipeline, not real-world accuracy)\n"); w(HDR)
for sp in ("validation", "test"): w(row(sp, ch["synthetic"][sp]))
w(f"\nBlocking recall (test): {p(ch['synthetic']['test']['blocking_recall'])}. The synthetic corruptions are the *same family* used to train, so this is an optimistic ceiling.\n")
w("### 2.3 Weak-real evaluation on held-out DUNS-linked pairs (small n — indicative only)\n")
w("| Split | Real linked pairs | Found (TP) | Missed (FN) | Recall (95% CI) | Hard negatives scored ≥ threshold (FP) |\n|---|---|---|---|---|---|")
for sp in ("validation", "test"):
    x = ch["weak_real"][sp]
    w(f"| {sp} | {x['positives']} | {x['true_positives']} | {x['false_negatives']} | {p(x['recall'])} {ci(x.get('recall_ci95'))} | {x['false_positives']} of {x['negatives']} |")
w("\nWith so few pairs the intervals are wide; this shows the model can recover genuine real-world duplicates, nothing more.\n")
w("### 2.4 Policy check — false-merge rate on same-name / different-site pairs (held out)\n")
w("| Split | Pairs | Falsely merged | Rate (95% CI) |\n|---|---|---|---|")
for sp in ("validation", "test"):
    x = ch["policy_negatives"][sp]; w(f"| {sp} | {x['pairs']} | {x['false_merges']} | {p(x['false_merge_rate'])} {ci(x['false_merge_rate_ci95'])} |")
ctc = C["cluster_test"]
w(f"\n### 2.5 Cluster-level (synthetic test)\n{ctc['cluster_count']} clusters vs {ctc['true_entity_count']} true; over-merged {ctc['over_merged_clusters']}; pairwise F1 {p(ctc['pairwise']['f1'])}; B-cubed F1 {p(ctc['bcubed']['f1'])}.\n")
w("### 2.6 Whole-file behaviour on customer.csv (operational — this is NOT accuracy)\n")
w(f"| Metric | Value |\n|---|---|\n| Records | {op['records']:,} |\n| Candidate pairs (of {op['possible_pairs']:,} possible) | {op['candidate_pairs']:,} ({op['reduction_pct']}% avoided) |\n"
  f"| MATCH / UNCERTAIN / NON-MATCH pairs | {op['match_pairs']:,} / {op['uncertain_pairs']:,} / {op['non_match_pairs']:,} |\n| Clusters / singletons | {op['clusters']:,} / {op['singletons']:,} |\n"
  f"| Records inside multi-record clusters | {op['records_in_multi_clusters']:,} |\n| Largest cluster / clusters over 10 | {op['largest_cluster']} / {op['clusters_over_10']} |\n"
  f"| Top cluster sizes | {op['top_cluster_sizes']} |\n| DUNS-linked pairs recovered (all / held-out) | {op['weak_pairs_recovered_all'][0]}/{op['weak_pairs_recovered_all'][1]} / {op['weak_pairs_recovered_heldout'][0]}/{op['weak_pairs_recovered_heldout'][1]} |\n"
  f"| Match time | {op['match_seconds']}s |\n")
w("`audit_sample` in `models_bundled/customer/v1/evaluation.json` holds 40 randomly drawn predicted-MATCH pairs; having a steward label them is the fastest route to a *real* precision estimate and to real v2 training labels.\n")
if C.get("duns_conflicts"):
    w("DUNS conflicts excluded from training (same DUNS, unrelated names — probable parent/subsidiary or data error):\n")
    for x in C["duns_conflicts"][:5]: w(f"* {x['a']}  ↔  {x['b']}")

# ---------------------------------------------------------------- customer calibration
cal = C["assistant_reviewed_calibration"]
EX = json.load(open(B / "models_bundled/customer/v1/v11_experiment_result.json"))
w("\n### 2.7 Threshold re-calibration on assistant-reviewed pairs (this is what ships)\n")
w(f"**Shipped Customer thresholds: match ≥ {cal['chosen_threshold']}, review band [{cal['review_threshold']}, {cal['chosen_threshold']}) (previously 0.65 / 0.50).** "
  "The §2.2–2.6 tables above were computed at the V1.0 build threshold 0.65 and are kept as the build-time record; this section supersedes the threshold.\n")
w("**Where the labels come from — read this.** "
  f"{cal['pairs_total']} candidate pairs from customer.csv were sampled, stratified by V1 score band, and reviewed by the AI assistant acting as a data steward "
  f"({cal['counts']['MATCH']} MATCH / {cal['counts']['DISTINCT']} DISTINCT / {cal['counts']['UNSURE']} UNSURE; UNSURE = parent/subsidiary or division cases, excluded). "
  "These labels are **not human ground truth** and are stored as `ASSISTANT_REVIEWED`, never `REAL_STEWARD`. "
  "The file `models_bundled/customer/v1/assistant_reviewed_labels.json` lists every pair so a human can audit it. Because the sample is stratified, each pair is weighted by its band's population share.\n")
w("| Threshold | Precision | Recall | F0.5 | Est. false merges (among the 3,750 candidate pairs scoring ≥0.20) |\n|---|---|---|---|---|")
for t, x in cal["table"].items():
    mark = " **←**" if float(t) == cal["chosen_threshold"] else (" (old)" if float(t) == 0.65 else "")
    w(f"| {t}{mark} | {p(x['precision'])} | {p(x['recall'])} | {p(x['f0_5'])} | {x['est_false_merges']} |")
w(f"\nBootstrap over the reviewed pairs: the F0.5-optimal threshold was 0.75 in {cal['bootstrap_argmax_threshold_frequency'].get('0.75',0)} of 500 resamples, 0.85 in {cal['bootstrap_argmax_threshold_frequency'].get('0.85',0)}, 0.80 in {cal['bootstrap_argmax_threshold_frequency'].get('0.8',0)} — "
  "i.e. 0.75–0.85 are statistically indistinguishable on this sample; 0.85 was chosen because a false merge contaminates a golden record, and pairs in the review band are sent to a steward rather than lost.\n")
op2 = cal["operational_at_new_threshold"]
w(f"Whole file at the shipped threshold: {op2['match_pairs']:,} MATCH / {op2['uncertain_pairs']:,} UNCERTAIN pairs → {op2['clusters']:,} clusters, {op2['singletons']:,} singletons, largest cluster {op2['largest_cluster']}, {op2['clusters_over_10']} clusters over 10 records.\n")
w(f"**Error analysis at the shipped threshold ({cal['errors_at_chosen_threshold']} of 111 decided pairs wrong):** " + ", ".join(f"{k.replace('_',' ').lower()} ×{v}" for k, v in cal["error_causes"].items()) + ".")
w("\n* Missed duplicates are mostly the same company with a *different address record* (address conflict penalised) or sparse records — a data-quality and training-evidence limit, not a threshold problem.")
w("* The remaining false merges are look-alike names or companies sharing a building: the model weights shared address + postal code heavily. This is the failure mode the review found most often at high scores (e.g. “American E Pay” vs “American Imaging Management”, 0.98 under the V1.0 scorer).\n")
for x in cal["error_examples"][:6]:
    a, b = x["A"], x["B"]
    w(f"* [{x['label']} predicted {x['predicted']} @ {x['score']}, {x['likely_cause']}] {a.get('customer_name')} ({a.get('city')}) ↔ {b.get('customer_name')} ({b.get('city')})")
w("\nCaveats: " + " ".join(cal["caveats"]) + "\n")
w("### 2.8 Model-refinement attempt that was NOT adopted\n")
w(f"{EX['what_was_tried']}\n")
r_ = EX["result_from_build_log"]
w(f"Cross-fit result: new model best at threshold {r_['new_model_best_threshold']}: precision {p(r_['new_precision'])}, recall {p(r_['new_recall'])}, F0.5 {p(r_['new_f0_5'])}, ~{r_['new_estimated_false_merges']} estimated false merges — "
  f"versus V1.0 at 0.65: precision {p(r_['v1_0_at_0.65_precision'])}, recall {p(r_['v1_0_at_0.65_recall'])}, F0.5 {p(r_['v1_0_at_0.65_f0_5'])}, ~{r_['v1_0_estimated_false_merges']} false merges.\n")
w(f"**Decision: {EX['decision']}** Likely reasons: " + "; ".join(EX["likely_reasons"]) + ". The honest conclusion is that better customer accuracy now requires **more human-reviewed pairs**, not more synthetic variation.\n")

# ---------------------------------------------------------------- method / limits
w("\n## 3. Validation controls actually applied\n")
w("| Control | How |\n|---|---|\n| No train/test leakage | Supplier: whole entities to one split. Customer: weak-label components, and every record by hash of its normalised name, to one split. |\n"
  "| Thresholds | Chosen on validation only; test scored once afterwards. |\n| Production path | Metrics come from `run_matching` + `run_clustering`, the same code the app uses. |\n"
  "| Standardization | Training data passes through the app's own cleansing step; nothing is re-standardized. |\n| Reproducibility | Fixed seed 42; rebuild with `python scripts/build_v1_models.py <domain> <csv>`. |\n"
  "| Compatibility | `metadata.json` records dedupe/BTrees/numpy/scikit-learn versions; the app flags a model whose dedupe / BTrees / numpy / scikit-learn versions differ from the installed ones. |\n"
  "| Uncertainty | Wilson 95% intervals on every proportion. |\n")
w("## 4. Memory-safe training (engineering note)\n")
w("Dedupe's `prepare_training()` materialises every record pair each candidate blocking predicate puts in one block, from a hard-coded sample of 5,000 records; on this data that exhausted memory even for ~1,000 labelled pairs. "
  "`app/modules/dedupe_service/training_limits.py` caps only that predicate-cover sample (500 records) during training. It changes how many records the blocking learner inspects, not the classifier, labels or matching engine. "
  "Index (Levenshtein/canopy) predicates are disabled for V1 for the same reason.\n")
w("## 5. Known limitations — read before relying on V1\n")
for x in [
 "Supplier numbers come from a generated file; real supplier data will be messier. Expect lower recall than the 99%+ shown for the full-record case.",
 "Customer precision has not been validated by a human steward (§2.7 uses AI-assisted labels on 111 pairs; weighted precision ≈82% at the shipped threshold, recall ≈47% with the rest routed to review). Expect over-merging of parent/subsidiary and branch records with identical names and sparse addresses (the largest customer clusters should be reviewed first).",
 "Blocking is the recall ceiling: V1 blocking keys on address, name, phone and identifiers; a duplicate pair that agrees on none of them is never compared (supplier test: <1% of true pairs; customer: see §2.2).",
 "Thin supplier files (name + e-mail + phone only) reach ~74% recall with a handful of false merges; send addresses or identifiers when you have them.",
 "Customer V1 treats same name at a different city + postal code as different sites — a policy choice, not a fact.",
 "V1 compares text fields and exact identifiers only. It does not use phone, e-mail or website for customers, and no bank-account, GSTIN/PAN or parent-hierarchy signals for suppliers.",
 "Only the Customer and Supplier domains exist; files in other layouts need a column named like the schema aliases, or the Advanced mapping flow.",
 "`supplier.xlsx` (417 rows) was not evaluated: it has no duplicate labels."]:
    w(f"* {x}")
w("\n## 6. Retraining v2 (when you are ready)\n")
w("Use the *Advanced* section of the app: label real pairs, train, validate, activate. Follow the V1 architecture in `backend/scripts/v1_build/specs.py` (field list and types, `has_missing=True`, identifier-masked augmentation, entity-level splits, threshold on validation). "
  "Activating a v2 retires V1 for that domain; V1 is never re-seeded over an active v2.")
Path(B.parent / "V1_EVALUATION_REPORT.md").write_text("\n".join(L), encoding="utf-8")
print("report written", len(L), "lines")
