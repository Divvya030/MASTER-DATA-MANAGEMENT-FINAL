# V1 Model — Evaluation & Validation Report

Models: **TR-V1-SUPPLIER** (supplier) and **TR-V1-CUSTOMER** (customer), version 1.0.0. Engine: `dedupe==3.0.3`, no LLM anywhere. Built 2026-10-04 by `backend/scripts/build_v1_models.py`; the running app only loads these artifacts, it never trains them.

## 0. How to read these numbers (please read first)

* **Supplier** numbers use `Supplier_1_.csv`, which has a known `entity_id` for every record, so pair labels are *real membership labels*. But the file is a **generated benchmark** (every e-mail and web domain is the reserved `.example` TLD; duplicates are produced by a small set of perturbations). Treat the supplier metrics as proof the pipeline works end to end, **not** as the accuracy you will see on a production supplier master.
* **Customer** data (`customer.csv`) has **no ground truth**. Customer metrics are therefore split into three clearly different kinds of evidence — *synthetic*, *weak-real (DUNS)* and *policy* — and the report never merges them into one 'accuracy' figure. Precision on real customer data has **no human ground truth**; the best available estimate is the 111 decided pairs in §2.7 (reviewed by the AI assistant acting as a steward — not by a human) plus the 22 DUNS-linked pairs in §2.3.
* Every metric below is on data the model never trained on (split by **entity**, or by connected component for weak labels, so no entity or linked record sits in two splits) and is computed through the **production** code path (`run_matching`, `run_clustering`), so pairs the blocking step never proposed count as misses.
* `supplier.xlsx` (417 rows, no duplicate labels, no ground truth) was not used to train or score anything.

## 1. Supplier V1

**Fields compared:** supplier_name, legal_name, email, phone, address, city, postal_code, country, duns, tax_identifier, registration_number.  **Threshold:** match ≥ 0.75, review band [0.6, 0.75) — chosen on the *validation* split as the best mean F1 over the three thin-data scenarios in §1.2 (identifiers hidden; identifiers + contact hidden; address + identifiers hidden), ties broken toward the higher threshold because a false merge costs more than a missed duplicate, then frozen before the test split was scored.

### 1.1 Data and split
10,000 records, 7,200 entities (2,800 entities have 2 records, 4,400 have 1). Entity-level split: train 4,367 / validation 1,432 / test 1,401 entities (6,023 / 2,012 / 1,965 records).

Training pairs (train split only): 3,207 MATCH / 9,640 DISTINCT. Negatives are 40% same-name-different-entity, 30% same-city-same-first-token, 30% random. Each pair is also emitted with identifier fields blanked (4,000 copies) and, for most pairs, with the address block blanked (2,440 copies) and, for most pairs, with address *and* identifiers blanked (2,407 copies: name + e-mail + phone only) — all labelled *synthetic augmentation* — so the model, and its learned blocking, still work when an upload has no identifiers or no address columns. Learned blocking: `(SimplePredicate: (suffixArray, phone), ExistsPredicate: (Exists, tax_identifier), ExistsPredicate: (Exists, city))`. Training time 136.1s.

### 1.2 Pair-level results at the chosen threshold

**All fields present**

| Slice | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|---|---|
| validation | 580 | 0 | 0 | 100.00% [99.3–100.0%] | 100.00% [99.3–100.0%] | 100.00% |
| test | 564 | 0 | 0 | 100.00% [99.3–100.0%] | 100.00% [99.3–100.0%] | 100.00% |

Blocking: recall 100.00% of true pairs proposed; 99.585% of all possible pairs avoided.

**Identifiers (DUNS / tax ID / registration no.) hidden**

| Slice | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|---|---|
| validation | 577 | 0 | 3 | 100.00% [99.3–100.0%] | 99.48% [98.5–99.8%] | 99.74% |
| test | 560 | 2 | 4 | 99.64% [98.7–99.9%] | 99.29% [98.2–99.7%] | 99.47% |

Blocking: recall 100.00% of true pairs proposed; 99.585% of all possible pairs avoided.

**Identifiers **and** e-mail / phone hidden (name + address only)**

| Slice | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|---|---|
| validation | 540 | 7 | 40 | 98.72% [97.4–99.4%] | 93.10% [90.7–94.9%] | 95.83% |
| test | 521 | 20 | 43 | 96.30% [94.4–97.6%] | 92.38% [89.9–94.3%] | 94.30% |

Blocking: recall 99.82% of true pairs proposed; 99.586% of all possible pairs avoided.

**Address, city and postal code hidden (name + contact + identifiers)**

| Slice | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|---|---|
| validation | 579 | 0 | 1 | 100.00% [99.3–100.0%] | 99.83% [99.0–100.0%] | 99.91% |
| test | 561 | 0 | 3 | 100.00% [99.3–100.0%] | 99.47% [98.5–99.8%] | 99.73% |

Blocking: recall 99.47% of true pairs proposed; 99.587% of all possible pairs avoided.

**Address block **and** identifiers hidden (name + e-mail + phone only)**

| Slice | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|---|---|
| validation | 436 | 3 | 144 | 99.32% [98.0–99.8%] | 75.17% [71.5–78.5%] | 85.57% |
| test | 417 | 3 | 147 | 99.29% [97.9–99.8%] | 73.94% [70.2–77.4%] | 84.76% |

Blocking: recall 99.47% of true pairs proposed; 99.587% of all possible pairs avoided.

How to read this: with identifiers present (or only the address hidden) the model is near-perfect on this file. Precision stays at or above 99% until e-mail, phone **and** identifiers are all unavailable; at name + e-mail + phone only, recall drops to ~74% and a few false merges appear. The hardest realistic case is a thin extract with a name and little else — expect results closer to that row than to the 100% rows.

### 1.3 Cluster-level (test, identifiers hidden)
1405 predicted clusters vs 1401 true entities; largest cluster 2; over-merged clusters 0. Pairwise F1 99.64%; B-cubed precision 100.00%, recall 99.80%, F1 99.90%.

### 1.4 Error analysis (test, identifiers hidden)
2 false positives, 4 false negatives; 0 of the misses were never proposed by blocking, 4 were proposed but scored below threshold. The misses sampled below are records whose *name* differs completely from their duplicate (an alias / renamed entity) while the address still agrees; the model correctly weighs the name disagreement. Rows with name + e-mail + phone only (see §1.2) are where most errors concentrate.

Sample misses:

* A: Horizon Components LLP / 1932 Park Avenue / Ahmedabad  ↔  B: Oakridge Retail Pvt Ltd / 1932 Park Avenue / Ahmedabad
* A: Delta Industrial Limited / 6176 Industrial Estate Road / Noida  ↔  B: Lotus Enterprises Limited / 6176 Industrial Estate Road / Noida
* A: Meridian Services Private Limited / 9198 Market Road / Bengaluru  ↔  B: Metro Facilities Industries Pvt Ltd / 9198 Market Road / Bengaluru

## 2. Customer V1

**Fields compared:** customer_name, address, city, postal_code, country. (`state` is excluded as noisy and redundant; D-U-N-S is excluded because it defines the weak labels — the model cannot read its own label.)  **Threshold:** match ≥ 0.65, review band [0.5, 0.65). Calibration: **PROVISIONAL_SYNTHETIC_AND_WEAK_LABELS** — i.e. *provisional*; there are too few real labels to calibrate it properly.

### 2.1 Evidence used (and what each can and cannot prove)
| Evidence | Size | Nature |
|---|---|---|
| Weak real positives (shared 9-digit D-U-N-S or `MDM_CUST*` source reference) | 66 pairs from 63 DUNS groups; train/val/test = 30/14/22; 1 name-conflicting pair excluded | Real, but *weak* (a DUNS can be shared by parent/subsidiary or mis-keyed) |
| Synthetic positives | 2,100 train pairs from 2,100 corrupted real records | Controlled typos, legal-form swaps, abbreviations, dropped tokens, missing values — **not** ground truth |
| Policy negatives (same name, different city *and* postal code ⇒ distinct sites) | 332 train pairs | A **business-policy assumption** (site-level resolution) — change it in v2 if you want company-level merging |
| Other negatives | 400 same-first-token + 350 random | Assumed distinct; a few could be unlabelled real duplicates |

Weak positives are up-weighted ×3 in training (30 real pairs → 90 training rows). Training: 2,190 MATCH / 1,082 DISTINCT, 81.0s. Learned blocking: `SimplePredicate: (sameSevenCharStartPredicate, customer_name)`.

### 2.2 Synthetic held-out evaluation (SYNTHETIC — measures the pipeline, not real-world accuracy)

| Slice | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|---|---|---|
| validation | 891 | 24 | 9 | 97.38% [96.1–98.2%] | 99.00% [98.1–99.5%] | 98.18% |
| test | 888 | 34 | 12 | 96.31% [94.9–97.4%] | 98.67% [97.7–99.2%] | 97.48% |

Blocking recall (test): 98.78%. The synthetic corruptions are the *same family* used to train, so this is an optimistic ceiling.

### 2.3 Weak-real evaluation on held-out DUNS-linked pairs (small n — indicative only)

| Split | Real linked pairs | Found (TP) | Missed (FN) | Recall (95% CI) | Hard negatives scored ≥ threshold (FP) |
|---|---|---|---|---|---|
| validation | 14 | 10 | 4 | 71.43% [45.4–88.3%] | 4 of 390 |
| test | 22 | 13 | 9 | 59.09% [38.7–76.7%] | 7 of 353 |

With so few pairs the intervals are wide; this shows the model can recover genuine real-world duplicates, nothing more.

### 2.4 Policy check — false-merge rate on same-name / different-site pairs (held out)

| Split | Pairs | Falsely merged | Rate (95% CI) |
|---|---|---|---|
| validation | 120 | 0 | 0.00% [0.0–3.1%] |
| test | 83 | 1 | 1.20% [0.2–6.5%] |

### 2.5 Cluster-level (synthetic test)
309 clusters vs 300 true; over-merged 1; pairwise F1 97.29%; B-cubed F1 98.85%.

### 2.6 Whole-file behaviour on customer.csv (operational — this is NOT accuracy)

| Metric | Value |
|---|---|
| Records | 9,996 |
| Candidate pairs (of 49,955,010 possible) | 258,029 (99.483% avoided) |
| MATCH / UNCERTAIN / NON-MATCH pairs | 795 / 153 / 257,081 |
| Clusters / singletons | 9,436 / 8,976 |
| Records inside multi-record clusters | 1,020 |
| Largest cluster / clusters over 10 | 7 / 0 |
| Top cluster sizes | [7, 7, 5, 5, 5, 5, 5, 5, 5, 4] |
| DUNS-linked pairs recovered (all / held-out) | 37/66 / 21/36 |
| Match time | 8.3s |

`audit_sample` in `models_bundled/customer/v1/evaluation.json` holds 40 randomly drawn predicted-MATCH pairs; having a steward label them is the fastest route to a *real* precision estimate and to real v2 training labels.

DUNS conflicts excluded from training (same DUNS, unrelated names — probable parent/subsidiary or data error):

* Alrajhi Bank  ↔  Al Rajhi Banking And Investment Corp Malaysia Bhd

### 2.7 Threshold re-calibration on assistant-reviewed pairs (this is what ships)

**Shipped Customer thresholds: match ≥ 0.85, review band [0.7, 0.85) (previously 0.65 / 0.50).** The §2.2–2.6 tables above were computed at the V1.0 build threshold 0.65 and are kept as the build-time record; this section supersedes the threshold.

**Where the labels come from — read this.** 160 candidate pairs from customer.csv were sampled, stratified by V1 score band, and reviewed by the AI assistant acting as a data steward (57 MATCH / 54 DISTINCT / 49 UNSURE; UNSURE = parent/subsidiary or division cases, excluded). These labels are **not human ground truth** and are stored as `ASSISTANT_REVIEWED`, never `REAL_STEWARD`. The file `models_bundled/customer/v1/assistant_reviewed_labels.json` lists every pair so a human can audit it. Because the sample is stratified, each pair is weighted by its band's population share.

| Threshold | Precision | Recall | F0.5 | Est. false merges (among the 3,750 candidate pairs scoring ≥0.20) |
|---|---|---|---|---|
| 0.5 | 67.73% | 59.02% | 65.79% | 233.9 |
| 0.55 | 69.22% | 57.25% | 66.44% | 211.8 |
| 0.6 | 69.39% | 53.71% | 65.56% | 197.1 |
| 0.65 (old) | 73.27% | 52.83% | 68.01% | 160.3 |
| 0.7 | 75.94% | 52.29% | 69.64% | 137.8 |
| 0.75 | 78.87% | 51.75% | 71.39% | 115.3 |
| 0.8 | 80.57% | 48.50% | 71.16% | 97.3 |
| 0.85 **←** | 82.49% | 46.72% | 71.54% | 82.5 |
| 0.9 | 84.22% | 38.72% | 68.20% | 60.3 |
| 0.95 | 88.46% | 35.17% | 67.89% | 38.2 |

Bootstrap over the reviewed pairs: the F0.5-optimal threshold was 0.75 in 164 of 500 resamples, 0.85 in 182, 0.80 in 74 — i.e. 0.75–0.85 are statistically indistinguishable on this sample; 0.85 was chosen because a false merge contaminates a golden record, and pairs in the review band are sent to a steward rather than lost.

Whole file at the shipped threshold: 597 MATCH / 153 UNCERTAIN pairs → 9,554 clusters, 9,173 singletons, largest cluster 7, 0 clusters over 10 records.

**Error analysis at the shipped threshold (30 of 111 decided pairs wrong):** location evidence outweighs name ×4, address conflict penalised ×15, sparse record ×5, similar name different entity ×5, name variant not recognised ×1.

* Missed duplicates are mostly the same company with a *different address record* (address conflict penalised) or sparse records — a data-quality and training-evidence limit, not a threshold problem.
* The remaining false merges are look-alike names or companies sharing a building: the model weights shared address + postal code heavily. This is the failure mode the review found most often at high scores (e.g. “American E Pay” vs “American Imaging Management”, 0.98 under the V1.0 scorer).

* [DISTINCT predicted MATCH @ 0.989, LOCATION_EVIDENCE_OUTWEIGHS_NAME] Aliments Ultima Yoplait (Longueuil) ↔ Alizã‰ (Longueuil)
* [DISTINCT predicted MATCH @ 0.924, LOCATION_EVIDENCE_OUTWEIGHS_NAME] Aeropuerto Internacional De La Ciudad De Mexico Sa De Cv (Ciudad De Mã‰xico) ↔ Aeropuertos Y Servicios Auxiliares (Ciudad De Mã‰xico)
* [MATCH predicted NON-MATCH @ 0.553, ADDRESS_CONFLICT_PENALISED] American Family Insurance (Madison) ↔ American Family Insurance (Madison)
* [MATCH predicted NON-MATCH @ 0.587, ADDRESS_CONFLICT_PENALISED] Agilent Technologies Inc (Haryana) ↔ Agilent Technologies India Private Limited (Gurgaon)
* [MATCH predicted NON-MATCH @ 0.326, SPARSE_RECORD] Agfirst (Columbia) ↔ Agfirst Farm Credit Bank (None)
* [MATCH predicted NON-MATCH @ 0.768, ADDRESS_CONFLICT_PENALISED] Aldine Independent School District (Houston) ↔ Aldine Isd (Houston)

Caveats: Small sample (111 decided pairs); the interval around every figure is wide. Labels were assigned by the AI assistant, not by a human steward; a human audit is required before treating them as ground truth. Recall is measured only among pairs the blocking step already proposed with score >= 0.20.

### 2.8 Model-refinement attempt that was NOT adopted

Customer retrain adding (a) real-data hard negatives (same building, unrelated company names; only 114 such pairs exist), (b) realistic name/address variation using the public cleanco legal-form list, (c) the 111 decided assistant-reviewed pairs as real-data evidence (matches x3, distincts x2).

Cross-fit result: new model best at threshold 0.9: precision 64.43%, recall 83.94%, F0.5 67.57%, ~385.5 estimated false merges — versus V1.0 at 0.65: precision 73.27%, recall 52.83%, F0.5 68.01%, ~160.3 false merges.

**Decision: Not adopted: F0.5 did not improve, estimated false merges were ~2.4x higher, and the learned blocking (name 3-character-start predicates; every name in this file starts with 'A') exhausted memory on the full-file matching step. Higher recall alone is not an improvement for golden-record safety.** Likely reasons: only ~55 real labelled pairs per fold: too few to move a Dedupe classifier; only 114 real same-building hard negatives; assistant-reviewed labels are not human ground truth. The honest conclusion is that better customer accuracy now requires **more human-reviewed pairs**, not more synthetic variation.


## 3. Validation controls actually applied

| Control | How |
|---|---|
| No train/test leakage | Supplier: whole entities to one split. Customer: weak-label components, and every record by hash of its normalised name, to one split. |
| Thresholds | Chosen on validation only; test scored once afterwards. |
| Production path | Metrics come from `run_matching` + `run_clustering`, the same code the app uses. |
| Standardization | Training data passes through the app's own cleansing step; nothing is re-standardized. |
| Reproducibility | Fixed seed 42; rebuild with `python scripts/build_v1_models.py <domain> <csv>`. |
| Compatibility | `metadata.json` records dedupe/BTrees/numpy/scikit-learn versions; the app flags a model whose dedupe / BTrees / numpy / scikit-learn versions differ from the installed ones. |
| Uncertainty | Wilson 95% intervals on every proportion. |

## 4. Memory-safe training (engineering note)

Dedupe's `prepare_training()` materialises every record pair each candidate blocking predicate puts in one block, from a hard-coded sample of 5,000 records; on this data that exhausted memory even for ~1,000 labelled pairs. `app/modules/dedupe_service/training_limits.py` caps only that predicate-cover sample (500 records) during training. It changes how many records the blocking learner inspects, not the classifier, labels or matching engine. Index (Levenshtein/canopy) predicates are disabled for V1 for the same reason.

## 5. Known limitations — read before relying on V1

* Supplier numbers come from a generated file; real supplier data will be messier. Expect lower recall than the 99%+ shown for the full-record case.
* Customer precision has not been validated by a human steward (§2.7 uses AI-assisted labels on 111 pairs; weighted precision ≈82% at the shipped threshold, recall ≈47% with the rest routed to review). Expect over-merging of parent/subsidiary and branch records with identical names and sparse addresses (the largest customer clusters should be reviewed first).
* Blocking is the recall ceiling: V1 blocking keys on address, name, phone and identifiers; a duplicate pair that agrees on none of them is never compared (supplier test: <1% of true pairs; customer: see §2.2).
* Thin supplier files (name + e-mail + phone only) reach ~74% recall with a handful of false merges; send addresses or identifiers when you have them.
* Customer V1 treats same name at a different city + postal code as different sites — a policy choice, not a fact.
* V1 compares text fields and exact identifiers only. It does not use phone, e-mail or website for customers, and no bank-account, GSTIN/PAN or parent-hierarchy signals for suppliers.
* Only the Customer and Supplier domains exist; files in other layouts need a column named like the schema aliases, or the Advanced mapping flow.
* `supplier.xlsx` (417 rows) was not evaluated: it has no duplicate labels.

## 6. Retraining v2 (when you are ready)

Use the *Advanced* section of the app: label real pairs, train, validate, activate. Follow the V1 architecture in `backend/scripts/v1_build/specs.py` (field list and types, `has_missing=True`, identifier-masked augmentation, entity-level splits, threshold on validation). Activating a v2 retires V1 for that domain; V1 is never re-seeded over an active v2.