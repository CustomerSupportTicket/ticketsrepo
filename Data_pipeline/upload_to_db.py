"""
Upload Clean CSV to PostgreSQL
"""

from pathlib import Path

import pandas as pd

from db_connection import get_engine


# =====================================
# Locate Project Directory
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = (
    PROJECT_ROOT
    / "Dataset"
    / "CleanData"
    / "customer_support_tickets_clean.csv"
)

TABLE_NAME = "customer_support_tickets"


def upload_csv_to_database():

    print("Reading cleaned CSV...")

    df = pd.read_csv(CSV_FILE)

    print(f"Records Found : {len(df)}")
    print(f"Columns       : {len(df.columns)}")

    engine = get_engine()

    print("Uploading data to PostgreSQL...")

    df.to_sql(
        name=TABLE_NAME,
        con=engine,
        if_exists="replace",      # replace existing table
        index=False
    )

    print("\nUpload Completed Successfully.")
    print(f"Table Name : {TABLE_NAME}")
    print(f"Rows Uploaded : {len(df)}")


if __name__ == "__main__":
    upload_csv_to_database()