"""
DataClean.py
================================================================================
Customer Support Tickets - Data Cleaning & Data Quality Assurance (DQA) Pipeline
================================================================================

Reads the raw dataset from   Dataset/RawData/customer_support_tickets_200k.csv
Writes the cleaned dataset to Dataset/CleanData/customer_support_tickets_cleaned.csv

Features
--------
1. Chunk-based processing so the pipeline scales to very large CSV files
   without loading the whole file into memory.
2. A DataQualityAssurance (DQA) engine that checks the data BEFORE, DURING
   and AFTER cleaning:
       - Missing values
       - Duplicate records (by primary key)
       - Invalid email formats (via the `email-validator` library)
       - Incorrect date relationships (resolved date before created date, etc.)
       - Out-of-range numeric values (clipped using IQR bounds)
       - Invalid categorical values (mapped against controlled vocabularies)
       - Data type mismatches (coerced + logged)
       - Null primary keys (rows dropped)
       - Inconsistent boolean values (Yes/No/Y/N/True/False/1/0 -> Yes/No)
       - Invalid IDs (non-positive, non-numeric, or duplicate ticket_id)
3. A final Data Quality Report printed to the terminal AND written to a log
   file (logs/data_quality_report_<timestamp>.txt).
4. Robust exception handling at every stage (missing files, corrupt CSVs,
   unsupported dtypes, malformed rows) with meaningful error messages
   instead of an unhandled crash.

Usage
-----
    python DataClean.py
    python DataClean.py --input custom_raw.csv --output custom_clean.csv --chunksize 20000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    from email_validator import validate_email, EmailNotValidError
    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATOR_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "Dataset", "RawData", "customer_support_tickets_200k.csv")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "Dataset", "CleanData", "customer_support_tickets_cleaned.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DEFAULT_CHUNKSIZE = 20_000 

PRIMARY_KEY = "ticket_id"
DATE_COLUMNS = ["ticket_created_date", "ticket_resolved_date"]
EMAIL_COLUMN = "customer_email"

# Columns that behave as booleans but arrive in inconsistent formats.
BOOLEAN_COLUMNS = ["escalated", "sla_breached"]

# Numeric columns that should be checked for out-of-range / outlier values.
NUMERIC_RANGE_RULES = {
    "customer_age": (0, 120),
    "customer_tenure_months": (0, 720),
    "previous_tickets": (0, 500),
    "customer_satisfaction_score": (1, 5),
    "first_response_time_hours": (0, 500),
    "resolution_time_hours": (0, 2000),
    "issue_complexity_score": (1, 10),
}

# Controlled vocabularies for free-text categorical columns. Any value not
# found in the mapping (after normalisation) is flagged as invalid.
CATEGORICAL_STANDARDIZATION = {
    "escalated": {
        "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "1.0": "Yes",
        "no": "No", "n": "No", "false": "No", "0": "No", "0.0": "No",
    },
    "sla_breached": {
        "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "1.0": "Yes",
        "no": "No", "n": "No", "false": "No", "0": "No", "0.0": "No",
    },
    "operating_system": {
        "ios": "iOS", "macos": "MacOS", "mac os": "MacOS", "windows": "Windows",
        "android": "Android", "linux": "Linux",
    },
    "priority": {
        "low": "Low", "medium": "Medium", "med": "Medium", "high": "High", "urgent": "Urgent",
    },
    "status": {
        "open": "Open", "closed": "Closed", "in progress": "In Progress",
        "inprogress": "In Progress", "pending customer": "Pending Customer",
        "resolved": "Resolved",
    },
    "customer_gender": {
        "male": "Male", "m": "Male", "female": "Female", "f": "Female", "other": "Other",
    },
    "channel": {
        "chat": "Chat", "email": "Email", "phone": "Phone",
        "social media": "Social Media", "web form": "Web Form",
    },
}

# Columns that MUST match one of the allowed values (case-insensitive match
# is attempted first via CATEGORICAL_STANDARDIZATION; anything left over is
# validated against this allow-list and flagged if not present).
CATEGORICAL_ALLOWED_VALUES = {
    "priority": {"Low", "Medium", "High", "Urgent"},
    "status": {"Open", "Closed", "In Progress", "Pending Customer", "Resolved"},
    "channel": {"Chat", "Email", "Phone", "Social Media", "Web Form"},
    "region": {"Africa", "Asia", "Australia", "Europe", "North America", "South America"},
    "customer_gender": {"Male", "Female", "Other"},
    "subscription_type": {"Basic", "Enterprise", "Free", "Premium"},
    "operating_system": {"Android", "Linux", "MacOS", "Windows", "iOS"},
    "browser": {"Chrome", "Edge", "Firefox", "Safari"},
    "payment_method": {"Bank Transfer", "Credit Card", "Crypto", "Debit Card", "PayPal"},
    "customer_segment": {"Corporate", "Individual", "Small Business"},
}

EXPECTED_DTYPES = {
    "ticket_id": "int64",
    "customer_age": "float64",
    "customer_tenure_months": "float64",
    "previous_tickets": "float64",
    "customer_satisfaction_score": "float64",
    "first_response_time_hours": "float64",
    "resolution_time_hours": "float64",
    "issue_complexity_score": "float64",
}


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class DataCleaningError(Exception):
    """Base exception for the cleaning pipeline."""


class FileAccessError(DataCleaningError):
    """Raised when the input file cannot be found or read."""


class SchemaValidationError(DataCleaningError):
    """Raised when the CSV structure does not match what the pipeline expects."""


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"pipeline_run_{timestamp}.log")

    logger = logging.getLogger("DataClean")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Logging to: {log_path}")
    return logger


# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

@dataclass
class DataQualityReport:
    total_records_raw: int = 0
    total_columns: int = 0

    missing_values_by_column: dict = field(default_factory=lambda: defaultdict(int))
    duplicate_records_removed: int = 0
    null_primary_keys_removed: int = 0
    invalid_ids_removed: int = 0

    invalid_emails_detected: int = 0
    emails_validated: int = 0

    invalid_dates_detected: int = 0
    dates_corrected: int = 0

    outliers_clipped: int = 0
    outliers_clipped_by_column: dict = field(default_factory=lambda: defaultdict(int))

    invalid_categorical_detected: int = 0
    invalid_categorical_by_column: dict = field(default_factory=lambda: defaultdict(int))

    inconsistent_booleans_standardized: int = 0

    dtype_conversions: dict = field(default_factory=lambda: defaultdict(int))
    dtype_mismatch_errors: int = 0

    rows_removed_total: int = 0
    rows_retained_total: int = 0
    chunks_processed: int = 0
    corrupt_rows_skipped: int = 0

    def missing_value_percentage(self) -> float:
        total_cells = self.total_records_raw * self.total_columns
        total_missing = sum(self.missing_values_by_column.values())
        if total_cells == 0:
            return 0.0
        return round((total_missing / total_cells) * 100, 3)

    def render(self) -> str:
        lines = []
        lines.append("=" * 78)
        lines.append("DATA QUALITY REPORT".center(78))
        lines.append("=" * 78)
        lines.append(f"{'Total records read (raw)':45}: {self.total_records_raw:,}")
        lines.append(f"{'Total columns':45}: {self.total_columns}")
        lines.append(f"{'Chunks processed':45}: {self.chunks_processed}")
        lines.append(f"{'Corrupt / unparsable rows skipped':45}: {self.corrupt_rows_skipped:,}")
        lines.append("-" * 78)
        lines.append(f"{'Overall missing value %':45}: {self.missing_value_percentage()}%")
        top_missing = sorted(self.missing_values_by_column.items(), key=lambda x: -x[1])[:8]
        if top_missing:
            lines.append("  Top columns with missing values:")
            for col, cnt in top_missing:
                if cnt:
                    lines.append(f"    - {col:35}: {cnt:,}")
        lines.append("-" * 78)
        lines.append(f"{'Null primary keys removed':45}: {self.null_primary_keys_removed:,}")
        lines.append(f"{'Invalid IDs removed':45}: {self.invalid_ids_removed:,}")
        lines.append(f"{'Duplicate records removed':45}: {self.duplicate_records_removed:,}")
        lines.append("-" * 78)
        lines.append(f"{'Emails validated':45}: {self.emails_validated:,}")
        lines.append(f"{'Invalid emails detected':45}: {self.invalid_emails_detected:,}")
        lines.append("-" * 78)
        lines.append(f"{'Invalid date relationships detected':45}: {self.invalid_dates_detected:,}")
        lines.append(f"{'Dates corrected/nulled':45}: {self.dates_corrected:,}")
        lines.append("-" * 78)
        lines.append(f"{'Outliers clipped (numeric, IQR-based)':45}: {self.outliers_clipped:,}")
        for col, cnt in self.outliers_clipped_by_column.items():
            if cnt:
                lines.append(f"    - {col:35}: {cnt:,}")
        lines.append("-" * 78)
        lines.append(f"{'Invalid categorical values standardized':45}: {self.invalid_categorical_detected:,}")
        for col, cnt in self.invalid_categorical_by_column.items():
            if cnt:
                lines.append(f"    - {col:35}: {cnt:,}")
        lines.append("-" * 78)
        lines.append(f"{'Inconsistent booleans standardized':45}: {self.inconsistent_booleans_standardized:,}")
        lines.append("-" * 78)
        lines.append(f"{'Data type mismatches coerced':45}: {sum(self.dtype_conversions.values()):,}")
        for col, cnt in self.dtype_conversions.items():
            if cnt:
                lines.append(f"    - {col:35}: {cnt:,}")
        lines.append("-" * 78)
        lines.append(f"{'Rows removed (total)':45}: {self.rows_removed_total:,}")
        lines.append(f"{'Rows retained (final, cleaned)':45}: {self.rows_retained_total:,}")
        if self.total_records_raw:
            retention_pct = round((self.rows_retained_total / self.total_records_raw) * 100, 2)
            lines.append(f"{'Retention rate':45}: {retention_pct}%")
        lines.append("=" * 78)
        return "\n".join(lines)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def is_valid_email(value: str) -> bool:
    """Validate an email address. Uses the `email-validator` library when
    available (syntax-only check, no DNS lookup, so it is fast and works
    offline); falls back to a conservative regex otherwise."""
    if not isinstance(value, str) or not value.strip():
        return False
    if EMAIL_VALIDATOR_AVAILABLE:
        try:
            validate_email(value, check_deliverability=False)
            return True
        except EmailNotValidError:
            return False
    # Fallback regex (only used if email-validator isn't installed)
    import re
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, value.strip()) is not None


def standardize_categorical(series: pd.Series, column: str, report: DataQualityReport) -> pd.Series:
    """Standardize a categorical column using controlled mappings; anything
    that cannot be confidently mapped is title-cased as a fallback, and
    flagged in the DQA report if it falls outside the allowed vocabulary."""
    mapping = CATEGORICAL_STANDARDIZATION.get(column, {})
    allowed = CATEGORICAL_ALLOWED_VALUES.get(column)

    def _clean(val):
        if pd.isna(val):
            return val
        raw = str(val).strip()
        key = raw.lower()
        if key in mapping:
            return mapping[key]
        # Fallback: normalise casing (Title Case) for anything not in the map
        cleaned = raw.title()
        if allowed is not None and cleaned not in allowed:
            report.invalid_categorical_detected += 1
            report.invalid_categorical_by_column[column] += 1
        return cleaned

    return series.apply(_clean)


def standardize_boolean(series: pd.Series, column: str, report: DataQualityReport) -> pd.Series:
    mapping = CATEGORICAL_STANDARDIZATION.get(column, {})

    def _clean(val):
        if pd.isna(val):
            return val
        raw = str(val).strip().lower()
        if raw in mapping:
            standardized = mapping[raw]
            if raw not in ("yes", "no"):
                report.inconsistent_booleans_standardized += 1
            return standardized
        # Unrecognised boolean-like value -> treat as missing rather than guess
        report.inconsistent_booleans_standardized += 1
        return np.nan

    return series.apply(_clean)


def clip_outliers_iqr(series: pd.Series, column: str, report: DataQualityReport) -> pd.Series:
    """Clip numeric outliers using IQR bounds, further constrained by any
    domain-specific min/max range defined in NUMERIC_RANGE_RULES."""
    numeric = pd.to_numeric(series, errors="coerce")

    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    if column in NUMERIC_RANGE_RULES:
        domain_min, domain_max = NUMERIC_RANGE_RULES[column]
        lower_bound = max(lower_bound, domain_min)
        upper_bound = min(upper_bound, domain_max)

    out_of_bounds = ((numeric < lower_bound) | (numeric > upper_bound)) & numeric.notna()
    n_clipped = int(out_of_bounds.sum())
    if n_clipped:
        report.outliers_clipped += n_clipped
        report.outliers_clipped_by_column[column] += n_clipped

    return numeric.clip(lower=lower_bound, upper=upper_bound)


def validate_and_fix_dates(df: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
    created = pd.to_datetime(df["ticket_created_date"], errors="coerce")
    resolved = pd.to_datetime(df["ticket_resolved_date"], errors="coerce")

    # Unparsable date strings
    bad_created = created.isna() & df["ticket_created_date"].notna()
    bad_resolved = resolved.isna() & df["ticket_resolved_date"].notna()
    report.invalid_dates_detected += int(bad_created.sum() + bad_resolved.sum())

    # Logical inconsistency: resolved before created
    inconsistent = (resolved < created) & created.notna() & resolved.notna()
    n_inconsistent = int(inconsistent.sum())
    if n_inconsistent:
        report.invalid_dates_detected += n_inconsistent
        # Correction strategy: null out the resolved date since we cannot
        # know the true value, rather than silently keeping bad data.
        resolved = resolved.mask(inconsistent, pd.NaT)
        report.dates_corrected += n_inconsistent

    df["ticket_created_date"] = created
    df["ticket_resolved_date"] = resolved
    return df


def coerce_dtypes(df: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        before = df[col].copy()
        try:
            coerced = pd.to_numeric(df[col], errors="coerce")
            mismatches = int(((coerced.isna()) & (before.notna())).sum())
            if mismatches:
                report.dtype_conversions[col] += mismatches
                report.dtype_mismatch_errors += mismatches
            df[col] = coerced
        except (ValueError, TypeError) as e:
            report.dtype_mismatch_errors += 1
            logging.getLogger("DataClean").warning(f"Dtype coercion failed for column '{col}': {e}")
    return df


# ============================================================================
# CORE CHUNK CLEANING LOGIC
# ============================================================================

def clean_chunk(df: pd.DataFrame, report: DataQualityReport, seen_ids: set, logger: logging.Logger) -> pd.DataFrame:
    initial_rows = len(df)

    # --- Missing value audit (BEFORE cleaning) ---
    for col in df.columns:
        report.missing_values_by_column[col] += int(df[col].isna().sum())

    # --- Null / invalid primary keys ---
    if PRIMARY_KEY not in df.columns:
        raise SchemaValidationError(f"Expected primary key column '{PRIMARY_KEY}' not found in CSV.")

    null_pk_mask = df[PRIMARY_KEY].isna()
    report.null_primary_keys_removed += int(null_pk_mask.sum())
    df = df[~null_pk_mask]

    pk_numeric = pd.to_numeric(df[PRIMARY_KEY], errors="coerce")
    invalid_id_mask = pk_numeric.isna() | (pk_numeric <= 0)
    report.invalid_ids_removed += int(invalid_id_mask.sum())
    df = df[~invalid_id_mask]
    df[PRIMARY_KEY] = pk_numeric[~invalid_id_mask].astype("int64")

    # --- Duplicate detection across chunks (by primary key) ---
    dup_mask = df[PRIMARY_KEY].isin(seen_ids)
    within_chunk_dup_mask = df[PRIMARY_KEY].duplicated(keep="first")
    full_dup_mask = dup_mask | within_chunk_dup_mask
    report.duplicate_records_removed += int(full_dup_mask.sum())
    df = df[~full_dup_mask]
    seen_ids.update(df[PRIMARY_KEY].tolist())

    # --- Drop fully-duplicate rows (all columns identical) ---
    exact_dupes = df.duplicated(keep="first")
    if exact_dupes.any():
        report.duplicate_records_removed += int(exact_dupes.sum())
        df = df[~exact_dupes]

    # --- Data type coercion ---
    df = coerce_dtypes(df, report)

    # --- Date validation / correction ---
    if all(c in df.columns for c in DATE_COLUMNS):
        df = validate_and_fix_dates(df, report)

    # --- Email validation ---
    if EMAIL_COLUMN in df.columns:
        valid_mask = df[EMAIL_COLUMN].apply(is_valid_email)
        report.emails_validated += len(df)
        n_invalid = int((~valid_mask).sum())
        report.invalid_emails_detected += n_invalid
        # Invalid emails are nulled rather than dropped, to preserve the ticket record
        df.loc[~valid_mask, EMAIL_COLUMN] = np.nan

    # --- Boolean standardization ---
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = standardize_boolean(df[col], col, report)

    # --- Categorical standardization ---
    for col in CATEGORICAL_STANDARDIZATION:
        if col in df.columns and col not in BOOLEAN_COLUMNS:
            df[col] = standardize_categorical(df[col], col, report)

    # Any remaining plain categorical/text columns not in the mapping: trim whitespace
    text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    for col in text_cols:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)

    # --- Outlier detection & clipping for numeric columns ---
    for col in NUMERIC_RANGE_RULES:
        if col in df.columns:
            df[col] = clip_outliers_iqr(df[col], col, report)

    report.rows_removed_total += initial_rows - len(df)
    return df


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

def run_pipeline(input_path: str, output_path: str, chunksize: int, logger: logging.Logger) -> DataQualityReport:
    report = DataQualityReport()
    seen_ids: set = set()

    # --- File existence / access checks ---
    if not os.path.exists(input_path):
        raise FileAccessError(f"Input file not found: {input_path}")
    if not os.access(input_path, os.R_OK):
        raise FileAccessError(f"Input file is not readable (check permissions): {input_path}")
    if os.path.getsize(input_path) == 0:
        raise FileAccessError(f"Input file is empty: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Remove any pre-existing output so we always start from a clean file
    # (chunks are appended below).
    if os.path.exists(output_path):
        os.remove(output_path)

    first_chunk = True
    try:
        reader = pd.read_csv(
            input_path,
            chunksize=chunksize,
            on_bad_lines="warn",
            engine="python",
            encoding="utf-8",
        )
    except UnicodeDecodeError as e:
        raise FileAccessError(f"Could not decode file as UTF-8: {e}") from e
    except pd.errors.EmptyDataError as e:
        raise FileAccessError(f"CSV file has no columns / is empty: {e}") from e
    except pd.errors.ParserError as e:
        raise FileAccessError(f"CSV file appears to be corrupted or malformed: {e}") from e
    except Exception as e:
        raise FileAccessError(f"Unexpected error opening file '{input_path}': {e}") from e

    try:
        for i, raw_chunk in enumerate(reader, start=1):
            try:
                report.total_records_raw += len(raw_chunk)
                report.total_columns = max(report.total_columns, raw_chunk.shape[1])

                cleaned_chunk = clean_chunk(raw_chunk, report, seen_ids, logger)

                cleaned_chunk.to_csv(
                    output_path,
                    mode="a",
                    header=first_chunk,
                    index=False,
                )
                first_chunk = False
                report.chunks_processed += 1
                report.rows_retained_total += len(cleaned_chunk)

                logger.info(
                    f"Chunk {i}: read {len(raw_chunk):,} rows -> "
                    f"retained {len(cleaned_chunk):,} rows after cleaning"
                )

            except SchemaValidationError:
                raise  # fatal — stop the whole pipeline
            except Exception as e:
                # A problem in a single chunk shouldn't crash the whole run;
                # log it, count the rows as skipped, and continue.
                logger.error(f"Error while processing chunk {i}: {e}")
                logger.debug(traceback.format_exc())
                report.corrupt_rows_skipped += len(raw_chunk)
                report.rows_removed_total += len(raw_chunk)
                continue

    except SchemaValidationError as e:
        raise
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        raise
    except Exception as e:
        raise DataCleaningError(f"Unexpected failure during chunk processing: {e}") from e

    return report


def save_report_to_file(report: DataQualityReport) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(LOG_DIR, f"data_quality_report_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.render())
    return report_path


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Customer Support Tickets - Data Cleaning Pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to raw input CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to write cleaned CSV")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help="Rows per chunk")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    if not EMAIL_VALIDATOR_AVAILABLE:
        logger.warning(
            "Package 'email-validator' not found — falling back to regex email validation. "
            "Install it with: pip install email-validator"
        )

    logger.info("Starting Data Cleaning & DQA Pipeline")
    logger.info(f"Input : {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Chunk size: {args.chunksize:,} rows")

    try:
        report = run_pipeline(args.input, args.output, args.chunksize, logger)
    except FileAccessError as e:
        logger.error(f"FILE ERROR: {e}")
        sys.exit(1)
    except SchemaValidationError as e:
        logger.error(f"SCHEMA ERROR: {e}")
        sys.exit(1)
    except DataCleaningError as e:
        logger.error(f"PIPELINE ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    report_text = report.render()
    print("\n" + report_text + "\n")
    logger.info("Pipeline completed successfully.")

    report_path = save_report_to_file(report)
    logger.info(f"Data quality report saved to: {report_path}")
    logger.info(f"Cleaned dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
