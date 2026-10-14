from __future__ import annotations
import os
from app.db.models import make_engine, make_session_factory, init_db

DB_URL = os.environ.get("MDM_DB_URL", "sqlite:///" + os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mdm.db"))

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        os.makedirs(os.path.dirname(DB_URL.replace("sqlite:///", "")), exist_ok=True)
        _engine = make_engine(DB_URL)
        init_db(_engine)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = make_session_factory(get_engine())
    return _SessionLocal


def get_db():
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
