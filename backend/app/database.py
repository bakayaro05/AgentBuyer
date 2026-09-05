"""
SQLite-backed storage for the simulated merchant catalog.

Using SQLite (via SQLAlchemy) instead of Postgres for this phase - zero
setup on your machine, no server to install/run. The models use plain
SQLAlchemy Core/ORM, so migrating to Postgres later is a connection-string
change plus swapping the JSON column type, not a rewrite.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "agent_buyer.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_session() -> Session:
    return SessionLocal()
