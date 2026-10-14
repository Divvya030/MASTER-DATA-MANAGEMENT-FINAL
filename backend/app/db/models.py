from __future__ import annotations
import datetime as dt
from sqlalchemy import (Column, String, Integer, Float, Boolean, DateTime, ForeignKey,
                         JSON, create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

DATASET_STATUSES = [
    "UPLOADED", "MAPPED", "PROFILED", "STANDARDIZED",
    "VALIDATED", "ER_CONFIGURED", "READY_FOR_DEDUPE",
]


class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id = Column(String, primary_key=True)
    original_filename = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    columns = Column(JSON, nullable=False)  # list[str]
    status = Column(String, nullable=False, default="UPLOADED")
    raw_path = Column(String, nullable=False)
    header_warnings = Column(JSON, nullable=True)
    # SHA-256 of the ORIGINAL uploaded file bytes, before any parsing/transformation.
    # Used ONLY for file-level exact-duplicate-upload detection -- unrelated to,
    # and never consumed by, record/entity-level Dedupe.
    sha256 = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    mappings = relationship("FieldMapping", back_populates="dataset", cascade="all, delete-orphan")
    profiles = relationship("Profile", back_populates="dataset", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="dataset", cascade="all, delete-orphan")
    role_configs = relationship("ERFieldConfig", back_populates="dataset", cascade="all, delete-orphan")


class FieldMapping(Base):
    __tablename__ = "field_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False)
    canonical_field = Column(String, nullable=False)
    source_column = Column(String, nullable=False)

    dataset = relationship("Dataset", back_populates="mappings")


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False)
    canonical_field = Column(String, nullable=False)
    stage = Column(String, nullable=False)  # RAW | STANDARDIZED
    data_quality = Column(JSON, nullable=False)
    cleansing_signals = Column(JSON, nullable=False)
    er_readiness = Column(String, nullable=False)
    er_recommendation = Column(String, nullable=False)
    # Legacy column: the application-level BLOCKING assessment was removed from
    # Profiling (matching uses the trained model's learned blocking predicates).
    # Kept as a nullable-ish always-empty column so old databases keep working.
    blocking_assessment = Column(String, nullable=False, default="")
    # One short, human-readable sentence for stewards, rendered from the same
    # underlying statistics as er_recommendation (which stays in `reasons`).
    steward_explanation = Column(String, nullable=True)
    reasons = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    dataset = relationship("Dataset", back_populates="profiles")


class ERFieldConfig(Base):
    __tablename__ = "er_field_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False)
    canonical_field = Column(String, nullable=False)
    role = Column(String, nullable=False)  # IDENTIFIER | MATCH | BLOCKING | IGNORE
    source = Column(String, nullable=False, default="USER_OVERRIDE")

    dataset = relationship("Dataset", back_populates="role_configs")


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False)
    job_type = Column(String, nullable=False)  # PROFILE | STANDARDIZE | REPROFILE
    status = Column(String, nullable=False, default="PENDING")  # PENDING|RUNNING|DONE|ERROR
    progress = Column(Float, nullable=False, default=0.0)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    dataset = relationship("Dataset", back_populates="jobs")


# ---------------------------------------------------------------------------
# Entity Resolution pipeline: Training -> Matching -> Clustering.
#
# Run IDs use a UUID scheme (not a sequential "TR-YYYY-NNNN" counter like
# Dataset.dataset_id) -- this deliberately avoids repeating the earlier
# in-memory-counter-vs-database desync bug (UNIQUE constraint failures from a
# counter disconnected from actual DB state). UUIDs are collision-safe by
# construction with no shared mutable state to desync.
# ---------------------------------------------------------------------------

class TrainingRun(Base):
    __tablename__ = "training_runs"

    training_run_id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False)
    domain = Column(String, nullable=False)
    # A content hash of the ACTIVE ER configuration at the time training was
    # requested, so a result can always be tied back to exactly which
    # steward-confirmed field roles produced it (traceability requirement).
    er_config_version = Column(String, nullable=False)
    match_fields = Column(JSON, nullable=False)
    status = Column(String, nullable=False, default="CREATED")
    # CREATED -> DATA_PREPARED -> PAIRS_GENERATED -> TRAINING -> TRAINED -> EVALUATED | FAILED
    split_config = Column(JSON, nullable=True)
    pair_counts = Column(JSON, nullable=True)
    model_path = Column(String, nullable=True)
    evaluation_metrics = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # --- Model lifecycle (separate from the above TRAINING EXECUTION status,
    # which only ever describes whether the train() call itself succeeded).
    # lifecycle_status: DRAFT -> EVALUATED -> ACTIVE -> RETIRED.
    # A model only reaches DRAFT once status == TRAINED (a FAILED/
    # INSUFFICIENT_TRAINING_DATA run never gets a lifecycle_status at all --
    # there's nothing to activate). Promotion to ACTIVE is always an explicit
    # separate call (see ActiveModel below); training itself never activates.
    lifecycle_status = Column(String, nullable=True)
    activated_at = Column(DateTime, nullable=True)
    retired_at = Column(DateTime, nullable=True)

    # --- Training evidence + decision bands (see training_service.py).
    # label_counts: {"real_steward": {"match":n,"distinct":n}, "synthetic": {...}}
    # threshold/review_threshold: the model's decision bands. threshold_calibration
    # is PROVISIONAL unless enough REAL steward labels were held out to pick it.
    label_counts = Column(JSON, nullable=True)
    threshold = Column(Float, nullable=True)
    review_threshold = Column(Float, nullable=True)
    threshold_calibration = Column(String, nullable=True)
    real_evaluation = Column(JSON, nullable=True)      # real steward-labelled held-out evaluation
    training_options = Column(JSON, nullable=True)

    dataset = relationship("Dataset")


class ActiveModel(Base):
    """One row per domain: which training_run is the ACTIVE model for that
    domain right now. domain is the PRIMARY KEY, which is what mechanically
    enforces "at most one ACTIVE model per domain" -- activating a new model
    for a domain is an upsert on this table (retiring whichever run was
    previously active), not a scan-and-check over TrainingRun rows.

    This table -- not TrainingRun.dataset_id -- is what the matching service
    uses to resolve "domain -> model to use", which is what makes a model
    usable against a NEW dataset the model was never trained on, and usable
    even after the original training dataset has been deleted."""
    __tablename__ = "active_models"

    domain = Column(String, primary_key=True)
    training_run_id = Column(String, ForeignKey("training_runs.training_run_id"), nullable=False)
    activated_at = Column(DateTime, default=dt.datetime.utcnow)

    training_run = relationship("TrainingRun")


class MatchRun(Base):
    __tablename__ = "match_runs"

    match_run_id = Column(String, primary_key=True)
    training_run_id = Column(String, ForeignKey("training_runs.training_run_id"), nullable=False)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False)
    # Stored directly (not just derivable via a join to Dataset) so
    # operational stats survive the source dataset being deleted --
    # matching the same "provenance must not break on dataset deletion"
    # requirement already honored by model_store.py for trained models.
    domain = Column(String, nullable=True)
    status = Column(String, nullable=False, default="CREATED")  # CREATED -> MATCHING -> COMPLETED | FAILED
    threshold = Column(Float, nullable=False)
    review_threshold = Column(Float, nullable=True)   # lower edge of the UNCERTAIN band (None = no band)
    uncertain_count = Column(Integer, nullable=True)
    blocking_evaluation = Column(JSON, nullable=True) # candidate reduction + candidate recall (if labels exist)
    total_records = Column(Integer, nullable=True)
    total_possible_pairs = Column(Integer, nullable=True)
    candidate_pair_count = Column(Integer, nullable=True)
    match_count = Column(Integer, nullable=True)
    non_match_count = Column(Integer, nullable=True)
    results_path = Column(String, nullable=True)   # JSON: [{record_id_a, record_id_b, score, decision}, ...]
    scores_path = Column(String, nullable=True)     # .npy: raw structured Scores array, for later clustering
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    training_run = relationship("TrainingRun")
    dataset = relationship("Dataset")


class ClusterRun(Base):
    __tablename__ = "cluster_runs"

    cluster_run_id = Column(String, primary_key=True)
    match_run_id = Column(String, ForeignKey("match_runs.match_run_id"), nullable=False)
    training_run_id = Column(String, nullable=False)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False)
    domain = Column(String, nullable=True)  # see MatchRun.domain -- same reasoning
    status = Column(String, nullable=False, default="CREATED")  # CREATED -> CLUSTERING -> COMPLETED | FAILED
    threshold = Column(Float, nullable=False)
    cluster_count = Column(Integer, nullable=True)
    singleton_count = Column(Integer, nullable=True)
    largest_cluster_size = Column(Integer, nullable=True)
    clusters_path = Column(String, nullable=True)  # JSON: [{entity_id, record_ids, scores, cluster_size}, ...]
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    match_run = relationship("MatchRun")
    dataset = relationship("Dataset")


# ---------------------------------------------------------------------------
# Golden Record: Entity (the mastered real-world thing) + versioned
# GoldenRecordVersion snapshots (append-only -- see Module 14 requirement:
# "do not overwrite historical golden records blindly") + EntityMember (the
# source records currently attached to an entity, across datasets/batches).
# ---------------------------------------------------------------------------

class Entity(Base):
    __tablename__ = "entities"

    entity_id = Column(String, primary_key=True)
    domain = Column(String, nullable=False)
    status = Column(String, nullable=False, default="ACTIVE")  # ACTIVE | MERGED | RETIRED
    # The run that most recently produced/updated this entity's membership --
    # provenance, not a hard requirement for the entity to keep existing.
    cluster_run_id = Column(String, nullable=True)
    training_run_id = Column(String, nullable=True)
    er_config_version = Column(String, nullable=True)
    # If this entity was MERGED into another, record the survivor here
    # instead of deleting rows -- keeps history/provenance intact.
    merged_into_entity_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    members = relationship("EntityMember", back_populates="entity", cascade="all, delete-orphan")
    golden_versions = relationship("GoldenRecordVersion", back_populates="entity", cascade="all, delete-orphan")


class EntityMember(Base):
    """One source record currently (or formerly, if removed_at is set)
    attached to an entity. Never deleted on removal -- soft-removed instead,
    so lineage/history questions ('why was this record ever here') stay
    answerable."""
    __tablename__ = "entity_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False)
    record_id = Column(String, nullable=False)  # row index/id within that dataset's standardized layer
    source = Column(String, nullable=False, default="BATCH")  # BATCH | INCOMING | STEWARD_MANUAL
    added_at = Column(DateTime, default=dt.datetime.utcnow)
    removed_at = Column(DateTime, nullable=True)
    # Snapshot of the record's STANDARDIZED mapped values at attach time. The
    # mastered entity store must not depend on the source dataset still
    # existing: golden records are regenerated from these snapshots, so
    # deleting a source dataset never degrades an entity's history.
    values = Column(JSON, nullable=True)

    entity = relationship("Entity", back_populates="members")


class GoldenRecordVersion(Base):
    """Append-only. The CURRENT golden record for an entity is simply the
    row with the highest version_number for that entity_id -- never mutated
    or overwritten in place."""
    __tablename__ = "golden_record_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    golden_values = Column(JSON, nullable=False)     # {field: value}
    lineage = Column(JSON, nullable=False)            # {field: {value, dataset_id, record_id, reason}}
    conflicts = Column(JSON, nullable=False)          # {field: [{value, dataset_id, record_id}, ...]}
    field_confidence = Column(JSON, nullable=False)   # {field: 0..1}
    reason = Column(String, nullable=False)  # INITIAL_GENERATION | INCOMING_BATCH_UPDATE | STEWARD_CORRECTION
    survivorship_rule_version = Column(Integer, nullable=True)
    steward_status = Column(String, nullable=False, default="PROPOSED")  # PROPOSED | NEEDS_REVIEW | APPROVED | CORRECTED
    needs_review_fields = Column(JSON, nullable=True)   # [{field, reason, candidates}]
    updated_at = Column(DateTime, default=dt.datetime.utcnow)
    training_run_id = Column(String, nullable=True)
    er_config_version = Column(String, nullable=True)
    created_by = Column(String, nullable=False, default="system")
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    entity = relationship("Entity", back_populates="golden_versions")


# ---------------------------------------------------------------------------
# Steward Review
# ---------------------------------------------------------------------------

class StewardReview(Base):
    __tablename__ = "steward_reviews"

    review_id = Column(String, primary_key=True)
    entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
    action = Column(String, nullable=False)  # APPROVE | REJECT | CORRECT | SPLIT | MERGE
    before_state = Column(JSON, nullable=True)
    after_state = Column(JSON, nullable=True)
    reviewer = Column(String, nullable=False)
    reason = Column(String, nullable=True)
    training_run_id = Column(String, nullable=True)
    er_config_version = Column(String, nullable=True)
    # MODEL RESULT + STEWARD DECISION + FINAL MASTERED RESULT: model_result is
    # what the model/system originally proposed (never overwritten); before/after
    # states are the mastered result around the human decision.
    model_result = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    entity = relationship("Entity")


class PendingReview(Base):
    """A queued item awaiting steward attention -- distinct from
    StewardReview (which is the immutable LOG of a decision once made).
    Created for ambiguous incoming-batch matches ('possible match' band)
    that the system deliberately declines to auto-decide."""
    __tablename__ = "pending_reviews"

    pending_review_id = Column(String, primary_key=True)
    domain = Column(String, nullable=False)
    kind = Column(String, nullable=False)  # AMBIGUOUS_MATCH | LOW_CONFIDENCE_GOLDEN_RECORD
    entity_id = Column(String, nullable=True)          # candidate existing entity, if any
    candidate_dataset_id = Column(String, nullable=False)
    candidate_record_id = Column(String, nullable=False)
    candidate_other_record_id = Column(String, nullable=True)   # when the ambiguity is with ANOTHER NEW record in the batch
    candidate_values = Column(JSON, nullable=True)              # snapshot of the incoming record's STANDARDIZED values
    score = Column(Float, nullable=True)
    training_run_id = Column(String, nullable=True)
    batch_run_id = Column(String, nullable=True)
    match_threshold = Column(Float, nullable=True)
    review_threshold = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="PENDING")  # PENDING | RESOLVED
    resolved_review_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


# ---------------------------------------------------------------------------
# Continuous incoming data: batches processed against the domain's ACTIVE
# model. A batch's dataset_id is an ordinary Dataset row that went through
# the same generic ingestion/mapping/standardization modules as any other
# upload -- there is no separate "incoming data" ingestion pipeline.
# ---------------------------------------------------------------------------

class IncomingBatchRun(Base):
    __tablename__ = "incoming_batch_runs"

    batch_run_id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False)
    domain = Column(String, nullable=False)
    training_run_id = Column(String, nullable=False)  # the ACTIVE model used
    threshold = Column(Float, nullable=False)
    status = Column(String, nullable=False, default="CREATED")  # CREATED -> MATCHING -> COMPLETED | FAILED
    record_count = Column(Integer, nullable=True)
    attached_to_existing_count = Column(Integer, nullable=True)
    new_entity_count = Column(Integer, nullable=True)
    ambiguous_count = Column(Integer, nullable=True)
    review_threshold = Column(Float, nullable=True)
    candidate_pair_count = Column(Integer, nullable=True)
    processing_seconds = Column(Float, nullable=True)
    field_missingness = Column(JSON, nullable=True)   # {field: fraction_missing} for drift monitoring
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# Real steward-labelled pair repository (+ synthetic provenance). Deliberately
# has NO foreign keys to datasets/entities: labels are evidence that must
# outlive the dataset they were collected on.
# ---------------------------------------------------------------------------

class LabeledPair(Base):
    __tablename__ = "labeled_pairs"

    pair_id = Column(String, primary_key=True)
    domain = Column(String, nullable=False)
    dataset_id = Column(String, nullable=True)
    record_id_a = Column(String, nullable=False)   # row id, or "ENTITY::<entity_id>" for entity-vs-record labels
    record_id_b = Column(String, nullable=False)
    values_a = Column(JSON, nullable=False)        # snapshot of STANDARDIZED values (match+identifier fields)
    values_b = Column(JSON, nullable=False)
    display_a = Column(JSON, nullable=True)        # richer snapshot shown to the steward (all mapped fields)
    display_b = Column(JSON, nullable=True)
    strategy = Column(String, nullable=True)       # DEDUPE_ACTIVE_LEARNING | MODEL_UNCERTAIN | MODEL_STRONG | MODEL_HARD_NEGATIVE | REVIEW_DECISION
    evidence = Column(JSON, nullable=True)         # e.g. identifier agreement hints (never used to auto-label)
    model_score = Column(Float, nullable=True)
    label = Column(String, nullable=True)          # NULL (pending) | MATCH | DISTINCT | UNSURE
    label_source = Column(String, nullable=False, default="REAL_STEWARD")   # REAL_STEWARD | SYNTHETIC
    labeler = Column(String, nullable=True)
    reason = Column(String, nullable=True)
    er_config_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    labeled_at = Column(DateTime, nullable=True)


class TrainingRunLabel(Base):
    """Which labels a given training run actually used, and in which split --
    needed so a model is only ever evaluated on labels it did NOT train on."""
    __tablename__ = "training_run_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_run_id = Column(String, nullable=False)
    pair_id = Column(String, nullable=False)
    split = Column(String, nullable=False)   # train | validation | test


class ProcessingJob(Base):
    """Generic job record (training, matching, clustering, golden records,
    incoming batches). No FK to datasets so job history survives dataset
    deletion. A worker queue can later replace job_service.run_job() without
    changing this table or the API."""
    __tablename__ = "processing_jobs"

    job_id = Column(String, primary_key=True)
    job_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="PENDING")   # PENDING | RUNNING | DONE | FAILED
    progress = Column(Float, nullable=False, default=0.0)
    domain = Column(String, nullable=True)
    dataset_id = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    result = Column(JSON, nullable=True)
    artifact_ref = Column(String, nullable=True)   # e.g. training_run_id / match_run_id / batch_run_id
    error = Column(String, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class SurvivorshipConfig(Base):
    __tablename__ = "survivorship_configs"

    domain = Column(String, primary_key=True)
    config = Column(JSON, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    # Lifecycle: DRAFT -> STEWARD_CONFIRMED -> ACTIVE. Only an ACTIVE
    # configuration generates production Golden Records. Legacy rows created
    # before this column existed are treated as ACTIVE (lifecycle_status NULL).
    lifecycle_status = Column(String, nullable=True)  # DRAFT | STEWARD_CONFIRMED | ACTIVE | NULL(legacy=ACTIVE)
    updated_by = Column(String, nullable=True)
    updated_at = Column(DateTime, default=dt.datetime.utcnow)


class IncomingRecordAudit(Base):
    """End-to-end lineage for every incoming record."""
    __tablename__ = "incoming_record_audits"

    audit_id = Column(String, primary_key=True)
    batch_run_id = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    source_dataset_id = Column(String, nullable=False)
    source_record_id = Column(String, nullable=False)
    ingested_at = Column(DateTime, default=dt.datetime.utcnow)
    standardized_values = Column(JSON, nullable=True)
    training_run_id = Column(String, nullable=False)          # active model version used
    match_threshold = Column(Float, nullable=True)
    review_threshold = Column(Float, nullable=True)
    candidate_count = Column(Integer, nullable=True)          # candidate entities considered for this record
    selected_entity_id = Column(String, nullable=True)        # best candidate entity (may be pending review)
    match_score = Column(Float, nullable=True)
    decision = Column(String, nullable=False)                 # MATCH | AMBIGUOUS | NEW_ENTITY
    resulting_entity_id = Column(String, nullable=True)
    golden_record_version = Column(Integer, nullable=True)
    pending_review_id = Column(String, nullable=True)
    steward_decision = Column(String, nullable=True)
    steward_decided_at = Column(DateTime, nullable=True)


def make_engine(db_url: str):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args)


def make_session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(engine):
    Base.metadata.create_all(bind=engine)
    _ensure_sha256_column(engine)
    _ensure_model_lifecycle_columns(engine)
    _ensure_run_domain_columns(engine)
    _ensure_columns(engine, "training_runs", {"label_counts": "JSON", "threshold": "FLOAT",
        "review_threshold": "FLOAT", "threshold_calibration": "VARCHAR", "real_evaluation": "JSON",
        "training_options": "JSON"})
    _ensure_columns(engine, "match_runs", {"review_threshold": "FLOAT", "uncertain_count": "INTEGER",
        "blocking_evaluation": "JSON"})
    _ensure_columns(engine, "golden_record_versions", {"survivorship_rule_version": "INTEGER",
        "steward_status": "VARCHAR DEFAULT 'PROPOSED'", "needs_review_fields": "JSON", "updated_at": "DATETIME"})
    _ensure_columns(engine, "steward_reviews", {"model_result": "JSON"})
    _ensure_columns(engine, "entity_members", {"values": "JSON"})
    _ensure_columns(engine, "pending_reviews", {"candidate_other_record_id": "VARCHAR", "training_run_id": "VARCHAR",
        "batch_run_id": "VARCHAR", "match_threshold": "FLOAT", "review_threshold": "FLOAT",
        "candidate_values": "JSON"})
    _ensure_columns(engine, "profiles", {"steward_explanation": "VARCHAR"})
    _ensure_columns(engine, "survivorship_configs", {"lifecycle_status": "VARCHAR"})
    _ensure_review_indexes(engine)


def _ensure_review_indexes(engine):
    """Indexes behind the Steward Review and Final Master Data lookups. They
    are created with IF NOT EXISTS so existing databases get them too (create_all
    never touches existing tables)."""
    stmts = [
        "CREATE INDEX IF NOT EXISTS ix_pending_reviews_domain_status ON pending_reviews (domain, status, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_pending_reviews_entity ON pending_reviews (entity_id)",
        "CREATE INDEX IF NOT EXISTS ix_golden_versions_entity_version ON golden_record_versions (entity_id, version_number)",
        "CREATE INDEX IF NOT EXISTS ix_entity_members_entity ON entity_members (entity_id, removed_at)",
        "CREATE INDEX IF NOT EXISTS ix_entity_members_record ON entity_members (dataset_id, record_id)",
        "CREATE INDEX IF NOT EXISTS ix_entities_domain_status ON entities (domain, status, created_at)",
    ]
    with engine.connect() as conn:
        for sql in stmts:
            conn.exec_driver_sql(sql)
        conn.commit()


def _ensure_sha256_column(engine):
    """SQLAlchemy's create_all() only creates NEW tables -- it never ALTERs an
    existing one. A `datasets` table created before the sha256 column was added
    to this model would silently be missing it forever without this step. This
    is a real (if minimal) migration, not a hidden workaround: it's an explicit,
    idempotent ALTER TABLE, logged here rather than buried in application code."""
    with engine.connect() as conn:
        cols = conn.exec_driver_sql("PRAGMA table_info(datasets)").fetchall()
        col_names = {c[1] for c in cols}
        if "sha256" not in col_names:
            conn.exec_driver_sql("ALTER TABLE datasets ADD COLUMN sha256 VARCHAR")
            conn.commit()


def _ensure_model_lifecycle_columns(engine):
    """Same reasoning as _ensure_sha256_column: `training_runs` may already
    exist (created before lifecycle_status/activated_at/retired_at were
    added to the model), and create_all() never ALTERs existing tables."""
    with engine.connect() as conn:
        cols = conn.exec_driver_sql("PRAGMA table_info(training_runs)").fetchall()
        col_names = {c[1] for c in cols}
        for col, ddl_type in [("lifecycle_status", "VARCHAR"),
                               ("activated_at", "DATETIME"),
                               ("retired_at", "DATETIME")]:
            if col not in col_names:
                conn.exec_driver_sql(f"ALTER TABLE training_runs ADD COLUMN {col} {ddl_type}")
        conn.commit()


def _ensure_run_domain_columns(engine):
    """Same reasoning again: match_runs/cluster_runs may pre-date the
    `domain` column (added so operational stats survive source-dataset
    deletion -- see MatchRun.domain)."""
    with engine.connect() as conn:
        for table in ("match_runs", "cluster_runs"):
            cols = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            col_names = {c[1] for c in cols}
            if "domain" not in col_names:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN domain VARCHAR")
        conn.commit()


def _ensure_columns(engine, table: str, columns: dict):
    """Generic idempotent ALTER TABLE ADD COLUMN (create_all never alters
    existing tables). No-op if the table doesn't exist yet."""
    with engine.connect() as conn:
        cols = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
        if not cols:
            return
        existing = {c[1] for c in cols}
        for name, ddl in columns.items():
            if name not in existing:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
        conn.commit()
