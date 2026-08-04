import pandas as pd
from sqlalchemy import text
from config import engine

# ==========================================================
# Configuration
# ==========================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "Dataset",
        "CleanData",
        "customer_support_tickets_cleaned.csv"
    )
)

print("CSV Path:", CSV_PATH)
print("Exists :", os.path.exists(CSV_PATH))
TABLE_NAME = "customer_support_tickets"

# ==========================================================
# Load CSV
# ==========================================================

print("=" * 60)
print("Reading CSV...")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

# ==========================================================
# Convert Data Types
# ==========================================================

# Integer Columns
int_columns = [
    "customer_age",
    "customer_tenure_months",
    "previous_tickets",
    "customer_satisfaction_score",
    "issue_complexity_score"
]

for col in int_columns:
    df[col] = df[col].astype("Int64")

# Decimal Columns
decimal_columns = [
    "first_response_time_hours",
    "resolution_time_hours"
]

for col in decimal_columns:
    df[col] = df[col].astype(float)

# Date Columns
date_columns = [
    "ticket_created_date",
    "ticket_resolved_date"
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# Boolean Columns
df["escalated"] = df["escalated"].map({
    "Yes": True,
    "No": False
})

df["sla_breached"] = df["sla_breached"].map({
    "Yes": True,
    "No": False
})

# ==========================================================
# Create Table
# ==========================================================

create_table_query = f"""

DROP TABLE IF EXISTS {TABLE_NAME};

CREATE TABLE {TABLE_NAME} (

ticket_id BIGINT PRIMARY KEY,

customer_name TEXT,
customer_email TEXT,
product TEXT,
category TEXT,
issue_description TEXT,
resolution_notes TEXT,
priority TEXT,
status TEXT,
channel TEXT,
region TEXT,

customer_age INTEGER,
customer_gender TEXT,
subscription_type TEXT,

customer_tenure_months INTEGER,
previous_tickets INTEGER,
customer_satisfaction_score INTEGER,

first_response_time_hours DECIMAL(10,2),
resolution_time_hours DECIMAL(10,2),

ticket_created_date DATE,
ticket_resolved_date DATE,

escalated BOOLEAN,
sla_breached BOOLEAN,

operating_system TEXT,
browser TEXT,
payment_method TEXT,
language TEXT,
preferred_contact_time TEXT,

issue_complexity_score INTEGER,
customer_segment TEXT

);

"""

with engine.begin() as conn:
    conn.execute(text(create_table_query))

print("Table Created Successfully")

# ==========================================================
# Upload Data
# ==========================================================

df.to_sql(
    TABLE_NAME,
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=1000
)

print("\nDataset Uploaded Successfully")

# ==========================================================
# Verify
# ==========================================================

with engine.connect() as conn:

    count = conn.execute(
        text(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    ).scalar()

print(f"\nRows Inserted : {count:,}")