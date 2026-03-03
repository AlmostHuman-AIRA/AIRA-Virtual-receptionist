from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Store the SQLite DB alongside this module for consistent behavior
_db_path = Path(__file__).resolve().parent / "office.db"
DATABASE_URL = f"sqlite:///{_db_path.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
