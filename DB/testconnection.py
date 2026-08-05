from sqlalchemy import text
from config import engine

TABLE_NAME = "customer_support_tickets"   # Change if your table name is different

try:
    with engine.connect() as conn:

        # PostgreSQL Version
        version = conn.execute(text("SELECT version();")).scalar()

        print("=" * 70)
        print("✅ Connected Successfully to Amazon RDS PostgreSQL")
        print("=" * 70)
        print(version)

        # ----------------------------------------------------
        # List all tables
        # ----------------------------------------------------
        print("\nTables in Database")
        print("-" * 70)

        tables = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
            ORDER BY table_name;
        """))

        for table in tables:
            print(table[0])

        # ----------------------------------------------------
        # Column Names & Data Types
        # ----------------------------------------------------
        print("\nColumn Names & Data Types")
        print("-" * 70)

        columns = conn.execute(text(f"""
            SELECT
                column_name,
                data_type
            FROM information_schema.columns
            WHERE table_name='{TABLE_NAME}'
            ORDER BY ordinal_position;
        """))

        for col in columns:
            print(f"{col[0]:35} {col[1]}")

        # ----------------------------------------------------
        # First 5 Rows
        # ----------------------------------------------------
        print("\nFirst 5 Rows")
        print("-" * 70)

        rows = conn.execute(text(f"""
            SELECT *
            FROM {TABLE_NAME}
            LIMIT 5;
        """))

        for row in rows:
            print(row)

except Exception as e:
    print("=" * 70)
    print("❌ Connection Failed")
    print("=" * 70)
    print(e)