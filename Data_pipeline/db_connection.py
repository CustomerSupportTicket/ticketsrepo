"""
Database Connection Module
--------------------------
This module creates and returns a PostgreSQL SQLAlchemy engine.
"""

from sqlalchemy import create_engine


# ==============================
# PostgreSQL Configuration
# ==============================

DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "customer_support_db",
    "user": "postgres",
    "password": "your_password"
}


def get_engine():
    """
    Creates and returns a SQLAlchemy engine.
    """

    connection_url = (
        f"postgresql+psycopg2://"
        f"{DB_CONFIG['user']}:"
        f"{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:"
        f"{DB_CONFIG['port']}/"
        f"{DB_CONFIG['database']}"
    )

    engine = create_engine(connection_url)

    return engine