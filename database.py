from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Check for DATABASE_URL environment variable (Railway provides this)
# If not provided, use SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_interviewer.db")

# If DATABASE_URL is provided but it's a MySQL URL, we need to handle it properly
# For Railway, we'll use SQLite as the fallback since it doesn't require external database
if DATABASE_URL and DATABASE_URL.startswith("mysql"):
    # For Railway deployment, use SQLite instead of MySQL
    SQLALCHEMY_DATABASE_URL = "sqlite:///./ai_interviewer.db"
else:
    SQLALCHEMY_DATABASE_URL = DATABASE_URL

# Create engine with appropriate settings
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False}  # Required for SQLite
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
