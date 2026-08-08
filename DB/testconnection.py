from sqlalchemy import text
from config import engine

try:
    with engine.connect() as conn:

        # -----------------------------------------
        # Count total tables
        # -----------------------------------------
        table_count = conn.execute(text("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)).scalar()

        print("=" * 60)
        print("Connected Successfully to Amazon RDS PostgreSQL")
        print("=" * 60)

        print(f"\nTotal Tables: {table_count}")

        # -----------------------------------------
        # Get table names
        # -----------------------------------------
        tables = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)).fetchall()

        # -----------------------------------------
        # Count records in each table
        # -----------------------------------------
        print("\nRecords in Each Table")
        print("-" * 60)

        for table in tables:
            table_name = table[0]

            count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}";')
            ).scalar()

            print(f"{table_name:35} {count} records")

except Exception as e:
    print("=" * 60)
    print("Connection Failed")
    print("=" * 60)
    print(e)