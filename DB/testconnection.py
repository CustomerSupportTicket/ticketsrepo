from sqlalchemy import text
from config import engine

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]

        print("=" * 60)
        print("✅ Connected Successfully to Amazon RDS PostgreSQL")
        print("=" * 60)
        print(version)

except Exception as e:
    print("=" * 60)
    print("❌ Connection Failed")
    print("=" * 60)   
    print(e) 
    