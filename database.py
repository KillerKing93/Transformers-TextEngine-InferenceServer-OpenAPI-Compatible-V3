#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database connection and session management for AI Marketplace Platform

Usage:
    from database import get_db, init_db

    # Initialize database
    init_db()

    # Use in FastAPI endpoints
    @app.get("/example")
    def example(db: Session = Depends(get_db)):
        ...
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from models import Base

# Database configuration from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./marketplace.db")

# Create engine
# For SQLite: use StaticPool and check_same_thread=False for FastAPI compatibility
# For PostgreSQL/MySQL: remove connect_args
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    Initialize database by creating all tables.
    Should be called at application startup.
    """
    Base.metadata.create_all(bind=engine)
    print(f"[Database] Tables created/verified. Database: {DATABASE_URL}")


def get_db() -> Session:
    """
    Dependency for FastAPI endpoints to get database session.

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            items = db.query(Product).all()
            return items
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
