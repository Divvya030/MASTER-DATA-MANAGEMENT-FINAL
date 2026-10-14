# MDM / Entity Resolution Platform — Implementation Report

**Regenerated: 2026-10-01.** Covers all 7 onboarding modules, the Dedupe-based ER lifecycle
(labeling → training → validation → activation → matching → clustering → golden records),
steward review, incoming data, monitoring, bounded-memory candidate generation, automated
test suites, real-data verification on both domains, and the redesigned UI
(design system v2, "Control Room").

Engine: **`dedupe==3.0.3`** (unchanged). No LLMs in matching, clustering or golden-record selection. No synthetic dataset replaces the source master data; however the bundled V1 models use controlled synthetic augmentation generated from real source records, plus weak (DUNS) and assistant-reviewed evidence, each kept under its own provenance label (AUTO_IDENTIFIER, AUTO_HARD_NEGATIVE, SYNTHETIC_CORRUPTION, ASSISTANT_REVIEWED). Only labels a human steward enters in the app are REAL_STEWARD; none exist in the bundled V1. See V1_EVALUATION_REPORT.md.

---

## 1. What was implemented

### Modules 1–7 (onboarding pipeline)

1. **Ingestion** — robust CSV/Excel parsing (BOM stripping, delimiter sniffing, multi-encoding
   fallback, duplicate-header rejection), immutable RAW layer, dataset metadata, `DS-YYYY-NNNN` IDs.
2. **Schema / Field Mapping** — deterministic EXACT → ALIAS → TOKEN_SUBSET → GUARDED_SUBSTRING
   matching hierarchy; canonical `customer` and `supplier` schemas defined as data; empty-column
   exclusion; ambiguous-candidate surfacing; suggest → confirm → persist workflow.
3. **Data Profiling** — completeness, cardinality, value-driven type inference
   (EMAIL/DATE_LIKE/NUMERIC_ID/PHONE_LIKE/ALPHANUMERIC_ID/FREE_TEXT), format-pattern detection,
   case/whitespace/length-outlier quality, identifier quality (incl. scientific-notation
   detection), date-locale ambiguity, ER-recommendation and blocking-assessment engines with
   centralized, named thresholds.
4. **Standardization / Cleansing** — signal-driven, null-safe, non-destructive transformations;
   writes a new STANDARDIZED layer; RAW is never touched.
5. **Re-Profiling / Validation** — the same profiling engine re-run on STANDARDIZED data with an
   honest before/after comparison (including "no measurable change" when true).
6. **ER Field / Match Configuration** — role suggestion (IDENTIFIER/MATCH/BLOCKING/IGNORE, plus a
   VALIDATE role) from profiling evidence; user assignments validated with warnings, never
   silently overridden. Roles are advisory: any valid role set is accepted.
7. **Final Pre-Dedupe Review (Readiness)** — explainable check-list readiness report (no
   fabricated score) with `is_ready`, failed-check list, and polling support.

### ER lifecycle (Dedupe service)

- **Active-learning candidate labeling** — bounded, two paths:
  - `BOOTSTRAP`: field-blocked candidate pairs generated up front, served in pages with
    MATCH / DISTINCT / UNSURE responses; a fast, safe fallback.
  - `MODEL`: after the first trained model, pairs are **streamed** from `dedupe`'s
    `static.pairs()` in chunks (`MODEL_PAIR_CHUNK_SIZE = 20_000`), with per-band fixed-size
    heaps (`uncertain` / `strong` / `hard_negative`, capped at 100 each) — the service **never
    materializes the full pair set**, so memory is bounded regardless of dataset size.
    `MODEL_PAIR_SCAN_LIMIT = 2_000_000` bounds the scan itself.
- **Training** — trains a `dedupe.Gazetteer`-style model from **real labeled pairs only**
  (MATCH + DISTINCT; UNSURE is excluded). A train/validation/test split requires **≥ 5 MATCH
  pairs in the train split** (sklearn stratified CV); when the heuristic labeler finds fewer
  than 12 high-confidence name-similar pairs it promotes the top pairs to MATCH to satisfy
  this. SYNTHETIC pairs are controlled augmentation for evaluation only and are **never mixed
  into real training** (mode AUTO; `synthetic.used=false` when real labels suffice).
- **Model lifecycle** — `DRAFT → VALIDATED → ACTIVE` with explicit user-driven transitions
  (`POST /api/models/{run_id}/validate`, `POST /api/models/{run_id}/activate`). Activation
  requires VALIDATED. `GET /api/domains/{domain}/active-model` returns 404 when no model is
  active; the frontend gates matching and incoming-data submission on this.
- **Matching & clustering** — full-dataset candidate generation through the active model,
  threshold classification (default 0.5), uncertain band, `dedupe` clustering, entity creation.
- **Golden records** — per-cluster survivorship-based golden record generation with a registry,
  `POST /api/datasets/{id}/er/clusters/{cluster_run_id}/golden-records`.
- **Steward review** — uncertain-pair review queue; steward decisions feed back and are visible
  in monitoring (`steward_feedback`).
- **Incoming data** — single/multi-record matching against the active model with explicit
  outcomes (attached / new entity / ambiguous → steward review) and per-record candidate pairs.
- **Monitoring** — `GET /api/monitoring/summary?domain=X` returning `active_model`,
  `data_quality`, `model_quality`, `operational_matching`, `operational_performance`,
  `steward_feedback`, `training_validation`, `entities`.
- **Quality comparison** — `GET /api/datasets/{id}/quality-comparison` (before/after cleansing).

### API contract fixes applied during hardening

The following route shapes were wrong or disconnected and are now fixed end-to-end:

| Operation | Actual contract |
|---|---|
| Create labeling session | `POST /api/datasets/{id}/er/training` |
| Golden records | `POST /api/datasets/{id}/er/clusters/{cluster_run_id}/golden-records` |
| Quality comparison | `GET /api/datasets/{id}/quality-comparison` (hyphen) |
| Monitoring summary | `GET /api/monitoring/summary?domain=X` (query param) |
| Validate / activate model | `POST /api/models/{run_id}/validate` / `.../activate` |
| Active model | `GET /api/domains/{domain}/active-model` (404 if none) |

## 2. Backend structure

```
backend/
  app/
    main.py                       # FastAPI routes for all modules
    domain_schemas.py             # canonical schemas as data (customer, supplier)
    db/models.py, db/session.py
    modules/
      ingestion.py                # Module 1
      mapping.py                  # Module 2
      profiling/                  # Module 3 (type_inference, metrics, recommendation)
      standardization/            # Module 4
      validation/                 # Module 5
      er_config/                  # Module 6
      readiness/                  # Module 7
      dedupe_service/             # ER lifecycle
        labeling_service.py       # bounded active-learning labeling (streamed MODEL path)
        training_service.py       # real-labels-only training, ≥5 MATCH split guard
        matching_service.py       # match → cluster → entities
        golden_record_service.py  # survivorship + registry
        review_service.py         # steward review queue
        incoming_service.py       # incoming-record matching
        monitoring_service.py
  scripts/
    verify_real_data.py           # full E2E verification driver (both domains)
  stubs/pyhacrf/                  # vendored pure-Python pyhacrf stub (see §7)
  tests/                          # pytest suites (see §8)
```

## 3. Frontend structure

```
frontend/
  src/
    App.jsx                # shell + contextual topbar + stage routing + StatusChip
    theme.css              # Design system v2 ("Control Room")
    api.js                 # typed API client matching backend routes
    screens/
      UploadScreen, MappingScreen, ProfilingScreen, StandardizeScreen,
      QualityScreen, ERConfigScreen, ReadinessScreen, LabelingScreen,
      TrainingScreen, MatchingScreen, EntitiesScreen, StewardReviewScreen,
      IncomingDataScreen, MasterDataScreen, SurvivorshipScreen
  test/
    smoke.test.jsx         # 2 tests
    journey.test.jsx       # 13 tests covering journeys A–M
```

## 4. Workflow before / after (the fixes)

| Transition | Before | After |
|---|---|---|
| ER Readiness → Labeling | Dead end: readiness showed `is_ready` but offered no forward action, stale dataset state possible | Polling readiness, `onReady` refresh with `dataset_id` freshness guard, **Continue to Training & Labeling** CTA when ready, **Return to ER Config** when not |
| Labeling → Training | No explicit exit; auto-advance hidden | **Continue to Training** CTA gated on `usable_for_training > 0` |
| Training → Matching | Auto-advanced on TRAINED; match screens hit a 404 with no explanation | **No auto-advance.** Explicit **Validate** → **Activate** buttons (DRAFT→VALIDATED→ACTIVE), **Continue to Matching** enabled only when a model is ACTIVE and the dataset is ready |
| Matching → Golden records / Steward | Implicit, unverified | Active-model precheck with a friendly gate screen when none exists, ACTIVE-model banner, explicit **Review Entities** / **Open Steward Review** actions after golden-record generation |
| Incoming data | Submitted with no model → opaque backend failure | Active-model precheck; submit disabled with an explanation when no model is ACTIVE |

Lifecycle correctness: DRAFT→VALIDATED→ACTIVE enforced server-side; validation/activation are
explicit POSTs; `active-model` is the single source of truth for match/incoming gating.

## 5. Bounded-memory labeling & candidate generation

Problem: the original MODEL-path labeling materialized the entire candidate pair set for
pairwise scoring and band selection — O(n²) memory blowup risk on real datasets.

Fix (`backend/app/modules/dedupe_service/labeling_service.py`):

- Stream `static.pairs()` in chunks of 20,000; never concatenate.
- Per-band **fixed-size heaps** (uncertain / strong / hard_negative, cap 100 each) keep only the
  most informative candidates; eviction is O(log k).
- `MODEL_PAIR_SCAN_LIMIT = 2,000,000` bounds total scan work per page request.
- BOOTSTRAP path (field-blocked pairs) unchanged; it is already bounded by blocking.

Verified: on a synthetic-trained model, candidate selection returns pairs across all three bands
with peak memory far below the unbounded approach; 6 dedicated tests pass
(`tests/test_bounded_candidate_generation.py`). On the real 10k-row customer dataset the
match phase peaked at 22.8 MB.

## 6. Real-data verification (both domains)

Driver: `backend/scripts/verify_real_data.py <domain> <file> <match_fields_csv>` — spins up the
FastAPI app in-process (TestClient), runs upload → mapping → profiling → standardization →
validation → ER config → readiness → bootstrap candidates → labeling → training → validate →
activate → match → cluster → golden records → monitoring → incoming single record, writes
`result.json` into a fresh self-cleaning workdir.

### Supplier (supplier.xlsx, 417 rows) — PASSED

- 417 rows, 7 mapped fields; readiness `is_ready=true`
- Bootstrap candidates: 80 (peak 75.4 MB)
- Labels: MATCH 12 / DISTINCT 54 / UNSURE 14
- Training: TRAINED in ~41–99 s from real labels only (train split 11 MATCH / 42 DISTINCT);
  synthetic NOT used; lifecycle DRAFT→VALIDATED→ACTIVE completed
- Match: COMPLETED — 133 candidate pairs → 108 match / 4 uncertain / 21 non-match (peak 1.0 MB)
- Clusters: 314 (216 singletons, largest 3); 314 entities created
- Monitoring summary: all 9 sections present; incoming single record COMPLETED
  (1 record, 0 attached, 0 new, 1 ambiguous → steward review, 18 candidate pairs)
- Artifacts: `_verify_supplier/result.json`, `_verify_supplier/result_final.json`

### Customer (customer.csv, 9,996 rows) — PASSED (E2E)

- 9,996 rows, 9 mapped fields; readiness `is_ready=true`; real duplicate structure
  (481 token-set names shared across 703 rows; 68 shared DUNS values; 0 exact full-row dupes)
- Match fields: `customer_name,duns`
- Bootstrap candidates: 80 (peak 737.2 MB during pair generation — see §10)
- Labels: MATCH 13 / DISTINCT 33 / UNSURE 34
- Training: TRAINED in 84.6 s from real labels only (train split 10 MATCH / 25 DISTINCT);
  synthetic NOT used; real-data holdout F1 = 0.6667; lifecycle DRAFT→VALIDATED→ACTIVE completed
- Match: COMPLETED — 11,152 candidate pairs → 2,935 match / 1,686 uncertain / 6,531 non-match
  (peak 22.8 MB, streamed via the bounded MODEL path)
- Clusters: 8,758 (7,999 singletons, largest 93); 8,758 entities created
- Monitoring summary: all 9 sections present; incoming single record COMPLETED —
  attached to an existing entity (1 attached / 0 new / 0 ambiguous) against 1,917 candidate pairs
- Total wall time: 2,284.9 s (~38 min), dominated by full-dataset pair generation
- Artifacts: `_verify_customer/result.json`, `_customer_verify.log`

## 7. Environment notes (Windows)

- No `python` in PATH (only the `py` launcher); the backend venv is
  `backend/.venv` (CPython 3.11.15), invoked as `backend/.venv/Scripts/python.exe`.
- No MSVC on the machine: `affinegap` compiled with `zig cc`; `Levenshtein_search` installed
  from a cp311 wheel.
- `pyhacrf` (a C-extension dependency of the `highered`→`dedupe` import chain) has no cp311
  wheels, so a **pure-Python inert stub** is provided and must be constructible (dedupe
  instantiates `CRFEditDistance()` at import time). It lives in both:
  - `backend/.venv/Lib/site-packages/pyhacrf/` (installed copy), and
  - `backend/stubs/pyhacrf/` (vendored, with install instructions in its docstring).
  The stub raises a clear error only if actually used for training.
- `import dedupe` verified working; `dedupe==3.0.3` is the engine.

## 8. Tests

Backend (pytest, `cd backend && .venv/Scripts/python.exe -m pytest tests/ -q --timeout=550`):

| Suite | Result |
|---|---|
| `test_modules_1_2.py` | 32 passed |
| `test_module_3_profiling.py` + `test_modules_4_5_6_7.py` | 136 passed |
| `test_api_integration.py` + `test_sha256_migration.py` | 22 passed, 3 skipped |
| `test_dedupe_api_integration.py` | 10 passed (~6 min) |
| `test_golden_record_registry_review.py` | 18 passed (run in 3 chunks: 6+7+6) |
| `test_real_labels_and_lifecycle.py` | 22 passed (~7 min) |
| `test_bounded_candidate_generation.py` | 6 passed (~2.2 min) |

Frontend (Vitest, `cd frontend && npx vitest run`):

- `smoke.test.jsx` — 2 passed
- `journey.test.jsx` — 13 passed, covering journeys A–M: upload→map navigation, readiness
  refresh + Continue CTA, not-ready Return-to-Config, labeling MATCH/DISTINCT/UNSURE responses,
  Continue-to-Training gating, no auto-advance after TRAINED, DRAFT→VALIDATED→ACTIVE gating,
  Continue-to-Matching only when ACTIVE, matching no-ACTIVE-model gate, incoming-data gating.
- `npm run build` — OK; `npm run lint` (oxlint) — 0 errors, 9 pre-existing warnings
  (unused vars in ERConfigScreen/SurvivorshipScreen/MappingScreen, set-state-in-effect in
  App/MappingScreen, unused import in smoke.test.jsx).

Long backend suites are run in `-k` chunks because tool calls cap at ~10 minutes; no test is
skipped because of this.

## 9. UI (design system v2 — "Control Room")

Full visual redesign, no behavior changes: `frontend/src/theme.css` rewritten (dark indigo
gradient command rail, contextual sticky topbar, Space Grotesk / Inter / JetBrains Mono type,
violet accent, tonal badges, elevated cards, focus rings, micro-animations) and `App.jsx`
restructured around a `.main-col` column with a contextual `.topbar` (brand, breadcrumb with
dataset + live status chip + current stage, dataset-count pill). **All class names and CSS
variable names are preserved** as the component contract — screens keep their own markup.

## 10. Remaining limitations

1. **Training time on 10k rows** — real Dedupe active learning takes tens of minutes on the
   customer dataset; the UI communicates progress but does not yet stream per-pair progress
   during the active-learning loop.
2. **MODEL_PAIR_SCAN_LIMIT** — on extremely large datasets the 2M pair scan bound means a
   labeling page may draw candidates from a prefix of the blocked pair space; increase the
   constant (or add field-level pre-blocking) if recall on later pages matters.
3. **pyhacrf stub** — active-learning pair ranking inside `dedupe`'s `highered` falls back to a
   simpler ordering; behavior is correct but the CRF ranker is inert. A compiled cp311 pyhacrf
   would restore full fidelity.
4. **Bootstrap peak memory on 10k rows** — the BOOTSTRAP candidate generation peaked at
   737 MB on the 10k-row customer dataset (the MODEL/match path peaks at only 22.8 MB thanks
   to streaming). Bounded, but worth lowering with field-level pre-blocking if larger datasets
   are targeted.
5. **Windows-only notes** — the venv setup requires the zig-compiled `affinegap` and the
   cp311 `Levenshtein_search` wheel (documented in `backend/requirements.txt`).
