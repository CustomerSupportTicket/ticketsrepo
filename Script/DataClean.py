"""
Customer Support Tickets — Data Cleaning & ML Preprocessing Pipeline
======================================================================
Takes the raw 200k customer-support-tickets export and turns it into
a clean, leak-free, model-ready dataset (+ fitted preprocessing
artifacts) that can be dropped straight into an sklearn Pipeline.

Design notes
------------
* Every cleaning decision is a separate, named, testable function so you
  can audit / swap steps without touching the rest of the pipeline.
* Nothing is fit on the full dataset before the train/test split — all
  encoders/scalers are fit on train only and applied to test, to avoid
  data leakage.
* PII (name/email) is dropped for modeling but a hashed customer_id is
  kept so you can still group by customer if needed.
* resolution_time_hours, ticket_resolved_date, status, and
  customer_satisfaction_score are only known AFTER a ticket closes.
  They are excluded from the default feature set unless your target
  itself is post-resolution.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = "C:/Users/HP/Desktop/pml project/Dataset/customer_support_tickets_200k.csv"
OUT_DIR = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Column groups (edit these to match your actual modeling target)
# ----------------------------------------------------------------------
ID_COLS = ["ticket_id", "customer_id"]
PII_COLS = ["customer_name", "customer_email"]
TEXT_COLS = ["issue_description", "resolution_notes"]
DATE_COLS = ["ticket_created_date", "ticket_resolved_date"]

NUMERIC_COLS = [
    "customer_age",
    "customer_tenure_months",
    "previous_tickets",
    "first_response_time_hours",
    "issue_complexity_score",
]
# Known only after resolution -> exclude from features unless that IS the target
LEAKAGE_COLS = ["resolution_time_hours", "ticket_resolved_date", "status", "customer_satisfaction_score"]

CATEGORICAL_COLS = [
    "product",
    "category",
    "priority",
    "channel",
    "region",
    "customer_gender",
    "subscription_type",
    "escalated",
    "sla_breached",
    "operating_system",
    "browser",
    "payment_method",
    "language",
    "preferred_contact_time",
    "customer_segment",
]

BINARY_YES_NO_COLS = ["escalated", "sla_breached"]


# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    logger.info("Loading raw CSV from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded shape: %s", df.shape)
    return df


# ----------------------------------------------------------------------
# 2. Structural cleaning
# ----------------------------------------------------------------------
def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are entirely (or almost entirely) null trailer rows."""
    before = len(df)
    df = df.dropna(subset=["ticket_id", "customer_name", "customer_email"]).copy()
    logger.info("Dropped %d fully-empty trailer rows", before - len(df))
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["ticket_id"]).copy()
    df = df.drop_duplicates().copy()
    logger.info("Dropped %d duplicate rows", before - len(df))
    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their correct, memory-efficient dtypes."""
    df["ticket_id"] = df["ticket_id"].astype("int64")

    int_like_cols = ["customer_age", "customer_tenure_months", "previous_tickets"]
    for col in int_like_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    float_cols = ["customer_satisfaction_score", "first_response_time_hours",
                  "resolution_time_hours", "issue_complexity_score"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in TEXT_COLS + PII_COLS:
        df[col] = df[col].astype("string").str.strip()

    return df


def validate_emails(df: pd.DataFrame) -> pd.DataFrame:
    """Flag / null-out malformed emails instead of silently trusting them."""
    email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    valid_mask = df["customer_email"].apply(lambda x: bool(email_re.match(x)) if pd.notna(x) else False)
    n_bad = (~valid_mask).sum()
    if n_bad:
        logger.warning("%d rows have malformed emails -> set to NaN", n_bad)
        df.loc[~valid_mask, "customer_email"] = pd.NA
    return df


def fix_date_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Resolved date can never precede created date; fix or null bad rows."""
    bad = df["ticket_resolved_date"] < df["ticket_created_date"]
    n_bad = bad.sum()
    if n_bad:
        logger.warning("%d rows have resolved_date < created_date -> set resolved_date to NaT", n_bad)
        df.loc[bad, "ticket_resolved_date"] = pd.NaT
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-sane clipping for numeric columns (based on observed valid ranges)."""
    bounds = {
        "customer_age": (18, 100),
        "customer_tenure_months": (0, 600),
        "previous_tickets": (0, 500),
        "customer_satisfaction_score": (1, 5),
        "issue_complexity_score": (1, 10),
        "first_response_time_hours": (0, 720),
        "resolution_time_hours": (0, 4320),
    }
    for col, (lo, hi) in bounds.items():
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_clipped:
            logger.warning("%s: clipping %d out-of-range values to [%s, %s]", col, n_clipped, lo, hi)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and fix casing drift so 'chat'/'Chat '/'CHAT' don't fragment."""
    for col in CATEGORICAL_COLS:
        cleaned = df[col].astype("string").str.strip().str.title()
        df[col] = cleaned.astype("category")
    return df


def encode_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    for col in BINARY_YES_NO_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype("Int64")
    return df


def add_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Replace PII email with a stable, anonymized customer hash for grouping."""
    df["customer_id"] = df["customer_email"].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16] if pd.notna(x) else pd.NA
    )
    return df


# ----------------------------------------------------------------------
# 3. Missing value imputation (simple, transparent, pre-split-safe values)
# ----------------------------------------------------------------------
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # browser has real missingness (~20%) -> explicit "Unknown" category,
    # since imputing a fake browser would be misleading, not a data-quality bug.
    df["browser"] = df["browser"].cat.add_categories(["Unknown"]).fillna("Unknown")

    # any leftover single-row nulls in categoricals -> mode-fill is safe at n=1
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True).iloc[0]
            df[col] = df[col].fillna(mode)

    # numeric leftovers -> median (robust to skew); done here for the raw
    # export, but for the ML matrix we redo this via SimpleImputer fit on
    # train only (see build_preprocessor) to avoid leakage.
    for col in NUMERIC_COLS + ["resolution_time_hours", "customer_satisfaction_score"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ----------------------------------------------------------------------
# 4. Feature engineering
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["created_year"] = df["ticket_created_date"].dt.year.astype("Int64")
    df["created_month"] = df["ticket_created_date"].dt.month.astype("Int64")
    df["created_dayofweek"] = df["ticket_created_date"].dt.dayofweek.astype("Int64")
    df["created_is_weekend"] = df["created_dayofweek"].isin([5, 6]).astype("Int64")

    df["response_to_resolution_ratio"] = (
        df["first_response_time_hours"] / df["resolution_time_hours"].replace(0, np.nan)
    ).astype("float32")

    df["is_repeat_customer"] = (df["previous_tickets"] > 0).astype("Int64")
    df["tenure_years"] = (df["customer_tenure_months"] / 12).astype("float32")

    df["issue_description_len"] = df["issue_description"].str.len().fillna(0).astype("int32")
    df["resolution_notes_len"] = df["resolution_notes"].str.len().fillna(0).astype("int32")

    return df


# ----------------------------------------------------------------------
# 5. Orchestration
# ----------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.pipe(drop_empty_rows)
          .pipe(drop_duplicates)
          .pipe(enforce_dtypes)
          .pipe(validate_emails)
          .pipe(fix_date_logic)
          .pipe(clip_outliers)
          .pipe(normalize_categoricals)
          .pipe(encode_binary_flags)
          .pipe(add_customer_id)
          .pipe(impute_missing)
          .pipe(engineer_features)
    )
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 6. ML-ready preprocessing (fit on train only -> no leakage)
# ----------------------------------------------------------------------
def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def main():
    df