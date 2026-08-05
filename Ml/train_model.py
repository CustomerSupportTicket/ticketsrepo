"""
train_model.py
---------------
Trains XGBoost classifiers to predict ticket CATEGORY and PRIORITY
from the issue_description text, using TF-IDF features.

Expected project layout (this script lives in ML/):
    pmlProject/
        Dataset/CleanData/customer_support_tickets_cleaned.csv   <- input
        models/                                                   <- output
        ML/train_model.py                                         <- this file
        app.py

Exports (into the models/ folder at project root):
    models/category_model.pkl
    models/priority_model.pkl
    models/tfidf_vectorizer.pkl
    models/label_encoders.pkl

Usage (run from anywhere, paths are resolved relative to this file):
    python ML/train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier

# ============================================================
# CONFIG
# ============================================================

# This script lives in ML/, so paths are relative to the project root (one level up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "CleanData", "customer_support_tickets_cleaned.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")     # pkl files go here, matching app.py's MODEL_DIR
TEXT_COL = "issue_description"
CATEGORY_COL = "category"
PRIORITY_COL = "priority"

TFIDF_MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Drop rows with missing text or labels
df = df.dropna(subset=[TEXT_COL, CATEGORY_COL, PRIORITY_COL]).reset_index(drop=True)
print(f"Rows after dropping missing values: {len(df)}")

# ============================================================
# STEP 2 — TF-IDF VECTORIZATION (fit once, shared by both models)
# ============================================================
print("Fitting TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
)
X = tfidf_vectorizer.fit_transform(df[TEXT_COL])

# ============================================================
# STEP 3 — LABEL ENCODING
# ============================================================
print("Encoding labels...")
category_encoder = LabelEncoder()
priority_encoder = LabelEncoder()

y_category = category_encoder.fit_transform(df[CATEGORY_COL])
y_priority = priority_encoder.fit_transform(df[PRIORITY_COL])

label_encoders = {
    "category": category_encoder,
    "priority": priority_encoder,
}

# ============================================================
# STEP 4 — TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
    X, y_category, y_priority,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_category,
)

# ============================================================
# STEP 5 — TRAIN CATEGORY MODEL
# ============================================================
print("\nTraining category model (XGBoost)...")
category_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=len(category_encoder.classes_),
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
category_model.fit(X_train, y_cat_train)

cat_preds = category_model.predict(X_test)
print(f"Category Accuracy: {accuracy_score(y_cat_test, cat_preds):.4f}")
print(f"Category Macro F1: {f1_score(y_cat_test, cat_preds, average='macro'):.4f}")
print(classification_report(y_cat_test, cat_preds, target_names=category_encoder.classes_))

# ============================================================
# STEP 6 — TRAIN PRIORITY MODEL
# ============================================================
print("\nTraining priority model (XGBoost)...")
priority_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=len(priority_encoder.classes_),
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
priority_model.fit(X_train, y_pri_train)

pri_preds = priority_model.predict(X_test)
print(f"Priority Accuracy: {accuracy_score(y_pri_test, pri_preds):.4f}")
print(f"Priority Macro F1: {f1_score(y_pri_test, pri_preds, average='macro'):.4f}")
print(classification_report(y_pri_test, pri_preds, target_names=priority_encoder.classes_))

# ============================================================
# STEP 7 — EXPORT ARTIFACTS
# ============================================================
print("\nSaving model artifacts to:", OUTPUT_DIR)
joblib.dump(category_model, os.path.join(OUTPUT_DIR, "category_model.pkl"))
joblib.dump(priority_model, os.path.join(OUTPUT_DIR, "priority_model.pkl"))
joblib.dump(tfidf_vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, "label_encoders.pkl"))

print("\nDone! Files saved:")
for f in ["category_model.pkl", "priority_model.pkl", "tfidf_vectorizer.pkl", "label_encoders.pkl"]:
    print(" -", os.path.join(OUTPUT_DIR, f))