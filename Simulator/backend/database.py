"""Database connection and session management."""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, init_db

# Default database path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "qca.db")


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable WAL mode for better concurrent access."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Database:
    """Database manager with connection pooling."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.engine = None
        self.SessionLocal = None
        
    def connect(self):
        """Initialize connection."""
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)

        # Lightweight migration for new run parameters
        self._ensure_run_columns()
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        return self

    def _ensure_run_columns(self) -> None:
        """Add missing columns to runs table (SQLite) without full migrations."""
        required = {
            "lambda_min": "REAL",
            "lambda_max": "REAL",
            "num_points": "INTEGER",
            "avg_frustration": "REAL",
            "safety_factor": "REAL",
            "target_frustration": "REAL",
            "max_multiplier": "REAL",
            "frustration_time": "REAL",
            "use_hotspot_search": "INTEGER",
            "hotspot_tolerance": "REAL",
            "hotspot_max_iterations": "INTEGER",
            "J0": "REAL",
        }
        with self.engine.connect() as conn:
            cols = conn.execute(text("PRAGMA table_info(runs)")).fetchall()
            existing = {c[1] for c in cols}
            for name, col_type in required.items():
                if name not in existing:
                    conn.execute(
                        text(f"ALTER TABLE runs ADD COLUMN {name} {col_type}")
                    )
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session as context manager."""
        if self.SessionLocal is None:
            self.connect()
            
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Get a session (caller must manage commit/close)."""
        if self.SessionLocal is None:
            self.connect()
        return self.SessionLocal()


# Global database instance
db = Database()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    with db.session() as session:
        yield session
