from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()

# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def init_db():
    """Initialize database with pgvector extension and create tables."""
    try:
        with engine.connect() as conn:
            # Try to enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("✓ PgVector extension is available")
    except Exception as e:
        print(f"\n⚠ Warning: Could not create PgVector extension: {str(e)}")
        print("⚠ The application will start, but vector operations will fail.")
        print("⚠ Please ask your PostgreSQL admin to install PgVector extension.")
        print("\n" + "="*70)
        print("PGVECTOR INSTALLATION INSTRUCTIONS FOR POSTGRESQL ADMIN:")
        print("="*70)
        print("\nFor PostgreSQL 10+ on RHEL/CentOS:")
        print("  sudo yum install postgresql10-devel")
        print("  git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git")
        print("  cd pgvector")
        print("  make")
        print("  sudo make install")
        print("\nThen connect to PostgreSQL and run:")
        print("  CREATE EXTENSION vector;")
        print("="*70 + "\n")

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
