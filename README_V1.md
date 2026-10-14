# MDM Entity Resolution — V1 ready-to-use build

Open the app and go straight to **Find duplicates**: the V1 Customer and Supplier models are
bundled, registered and ACTIVE on first start-up. Nothing is trained when you use them.

## Run (Windows, from this folder)
```
cd backend
py -3.11 -m venv .venv                      # or reuse your existing backend\.venv
.venv\Scripts\python -m pip install -r requirements.txt
# Windows without a C compiler: follow the pyhacrf/affinegap notes at the bottom of requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --port 8420

cd ..\frontend
npm install
npm run dev                                  # http://localhost:5173
```
Run the backend with the pinned versions in `requirements.txt` (dedupe 3.0.3, BTrees 6.4,
numpy 2.4.4, scikit-learn 1.8.0): the V1 model files were built under exactly those, and the app
flags any mismatch. Check http://localhost:8420/api/domains/customer/active-model — it should
return `TR-V1-CUSTOMER` (supplier: `TR-V1-SUPPLIER`).

## Use
The app opens on your original pipeline, with every layer intact:
**Ingest -> Map -> Profile -> Cleanse -> Validate -> Configure ER -> Readiness -> Match & Cluster**
(then Entities & Golden Records, Steward Review, Final Master Data, Incoming Data, Survivorship Rules in the
left rail). Upload a file with *+ New dataset*, walk the layers, and at **Readiness** click
*Match now with the ACTIVE V1 model*; Label / Train (v2) are optional and only for retraining.

**Match & Cluster** is one action: click *Match* and the existing matching run is followed automatically by
the existing clustering run, then the actual clustered records are shown (25 clusters per page, expandable).
Review them, then click *Generate Golden Records* (never automatic). On the 10k datasets golden-record
generation takes a few minutes; the button shows *Generating…* until it finishes.

Shortcut: *Find duplicates in one step (V1)* runs Map -> Profile -> Cleanse -> Match -> Cluster for you
with the active model and gives an annotated CSV. Its dataset also appears under *Datasets*, and
*Inspect this dataset's layers* opens it in the stages above so you can review the mapping,
profiling and cleansing it used.
Columns are matched to the model's fields by name; fields your file lacks are treated as *missing*,
never as a mismatch. A file needs a customer/supplier name column.
The Datasets list starts empty on a fresh install (I did not ship your old datasets or database).

## What is where
| Path | What |
|---|---|
| `V1_EVALUATION_REPORT.md` | Full evaluation + validation report (read sections 0 and 5 first) |
| `backend/models_bundled/<domain>/v1/` | model.settings, training.json, metadata.json, evaluation.json |
| `backend/scripts/build_v1_models.py`, `scripts/v1_build/` | How V1 was built (offline). Rebuild: `python scripts/build_v1_models.py supplier <csv>` |
| `backend/scripts/v1_build/specs.py` | The V1 architecture (fields, types) — follow it for v2 |
| `backend/app/modules/dedupe_service/bundled_models.py` | Start-up registration/activation (never overrides a later v2) |
| `backend/app/api/find_duplicates.py` | The one-step upload -> duplicates endpoint and the paged duplicate-groups endpoint used by Match & Cluster (no UI page of its own) |
| `backend/app/modules/dedupe_service/training_limits.py` | Memory cap for Dedupe's blocking learner |

## Retraining v2 (yours to do)
Pipeline stages *Label (v2, optional)* and *Train (v2, optional)* — label, train, validate, activate.
Activating a v2 retires V1 for that domain; V1 is not re-seeded over it.

## Changes to existing code (all small)
* `main.py`: registers the bundled models at start-up; the model-field mapping check now only requires
  a name column (a missing optional field is scored as missing, which Dedupe supports).
* `standardization.py`: form placeholders seen in customer.csv ("Please Select", "123 MAIN ST.",
  "111111", "TBC", ...) are treated as missing instead of as values that make records agree.
* `domain_schemas.py`: "tax identification number" added as a tax-ID alias (it was mapping to a 72%-empty column).
* Frontend: the original pipeline is still the landing page; added the *Find duplicates in one step* shortcut and a *Match now with the ACTIVE V1 model* button on Readiness.

Removed: old trained models and uploaded datasets, the heuristic-label verification runs
(`_verify_*`, `_probe`), logs, tool folders (`.freebuff`), caches, `node_modules`, `dist`, `.git`, `.venv`.

## Model evidence and thresholds (read before trusting V1)
* Customer V1 thresholds were re-calibrated on 160 reviewed pairs (`models_bundled/customer/v1/assistant_reviewed_labels.json`):
  match >= 0.85, review band 0.70-0.85 (was 0.65 / 0.50). Those labels were assigned by the AI assistant acting as a
  steward; they are stored as `ASSISTANT_REVIEWED`, are **not human ground truth**, and should be audited by a person.
* A Customer retrain with extra hard negatives, realistic augmentation and those labels was tried and **not adopted**
  (more false merges, memory-heavy blocking) - `V1_EVALUATION_REPORT.md` section 2.8. Better accuracy needs more human-reviewed pairs.
* Cluster view: a Standardized / Raw toggle shows the values the model compared vs the uploaded values; "match score" is the
  weakest pair score in the cluster and is a model score, not a probability.
* `GET /api/domains/{domain}/retraining/eligibility` reports whether enough REAL_STEWARD labels exist (constants in
  `dedupe_service/retraining.py`); `compare_models()` there is the V1-vs-candidate gate. Nothing retrains automatically.
* **Not implemented from the latest specification:** fixed Customer/Supplier golden-record policies replacing the survivorship UI,
  the VALIDATING/REJECTED lifecycle states, UNSURE-driven cluster-context review changes, automatic V2 training/comparison pipeline,
  and a memory-bounded bootstrap path. The existing configurable survivorship, steward review (CONFIRM / REJECT / UNSURE) and lifecycle remain as they were.

## Final Master Data, Steward Review and the incoming-record form

- **Final Master Data** contains one current Golden Record per active entity that is *final*: a steward
  approved it, or it is a **single-record (distinct) entity with no conflicts** -- unmatched records and
  steward-confirmed DISTINCT pairs appear automatically, since there is nothing to reconcile. Merged
  entities (2+ records) still need an explicit approval, and entities with unresolved conflicts stay in
  Steward Review. `GET /api/domains/{domain}/master-data?limit=&offset=` is paged and returns
  `distinct_record_count`, `merged_record_count` and `awaiting_review_count`.
- **Steward Review** is paged (`limit`/`offset`, default 25, max 200) on `/api/reviews/pending` and
  `/api/reviews/entities-needing-review`; the UI loads the first page and fetches more on demand.
- **Incoming Data -> single record** renders every canonical field configured for the selected domain,
  from `GET /api/domains/{domain}/schema` (add a field to `domain_schemas.py` and it appears in the form).
- The Monitoring and Quick run pages were removed from the UI. The backend endpoints behind them
  (`/api/monitoring/summary`, `/api/domains/{domain}/find-duplicates`) are unchanged.
