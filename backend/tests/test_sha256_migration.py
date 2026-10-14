"""
Regression test for the sha256 column migration (Item #1).

SQLAlchemy's Base.metadata.create_all() only creates NEW tables -- it never
ALTERs an existing one. Since `datasets` existed before the sha256 column was
added to the model, upgrading an existing deployment requires an explicit
migration step. This test proves that step actually works against a
pre-existing SQLite file, rather than just asserting it in prose.
"""
import sqlite3
import os


def test_sha256_column_migrates_onto_pre_existing_database(tmp_path):
    db_path = str(tmp_path / "old.db")

    # Simulate a database created BEFORE the sha256 column existed.
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE datasets (
        dataset_id VARCHAR PRIMARY KEY, original_filename VARCHAR, domain VARCHAR,
        row_count INTEGER, column_count INTEGER, columns TEXT, status VARCHAR,
        raw_path VARCHAR, header_warnings TEXT, created_at DATETIME
    )""")
    conn.execute(
        "INSERT INTO datasets VALUES ('DS-OLD-0001','x.csv','customer',10,2,'[]','UPLOADED','/tmp/x','[]',NULL)")
    conn.commit()
    conn.close()

    pre_cols = {r[1] for r in sqlite3.connect(db_path).execute("PRAGMA table_info(datasets)")}
    assert "sha256" not in pre_cols

    from app.db.models import make_engine, init_db
    engine = make_engine(f"sqlite:///{db_path}")
    init_db(engine)

    post_cols = {r[1] for r in sqlite3.connect(db_path).execute("PRAGMA table_info(datasets)")}
    assert "sha256" in post_cols

    # Existing data must survive the migration untouched.
    row = sqlite3.connect(db_path).execute(
        "SELECT dataset_id, original_filename, sha256 FROM datasets").fetchone()
    assert row == ("DS-OLD-0001", "x.csv", None)


def test_migration_is_idempotent_safe_to_run_twice(tmp_path):
    db_path = str(tmp_path / "fresh.db")
    from app.db.models import make_engine, init_db
    engine = make_engine(f"sqlite:///{db_path}")
    init_db(engine)  # first run: creates table + column
    init_db(engine)  # second run: must not raise (column already present)
    cols = {r[1] for r in sqlite3.connect(db_path).execute("PRAGMA table_info(datasets)")}
    assert "sha256" in cols
