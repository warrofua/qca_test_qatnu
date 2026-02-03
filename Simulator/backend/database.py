"""Database connection and session management."""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
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
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        return self
    
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
