"""
Model comparison: Logistic Regression vs LightGBM
---------------------------------------------------
Predicts BOTH category and priority, using the cleaned dataset from
Dataset/CleanData/. Trains + evaluates both models on both targets,
prints a side-by-side comparison table.

Same structure/preprocessing as the SVM vs KNN comparison, so results
are directly comparable across both scripts (same train/test split logic,
same feature engineering). This is COMPARISON code only — nothing gets
saved to models/ yet. Only once the team picks a final winning model
(across everyone's models) do you export that one to a .pkl file.
"""

import re
import glob
import time
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# 1. Load the cleaned dataset from Dataset/CleanData/
# ============================================================
csv_files = glob.glob("Dataset/CleanData/*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV found in Dataset/CleanData/. Check the path/filename.")
DATA_PATH = csv_files[0]
print("Using dataset:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ============================================================
# 1b. Drop leakage columns — known only AFTER a ticket is resolved,
#     so they must never be used to predict category/priority.
# ============================================================
leakage_cols = ["resolution_notes", "status", "resolution_time_hours",
                 "ticket_resolved_date", "escalated", "sla_breached",
                 "customer_satisfaction_score", "first_response_time_hours"]
present_leakage = [c for c in leakage_cols if c in df.columns]
print("Dropping leakage columns:", present_leakage)
df = df.drop(columns=present_leakage)

# Drop identifier columns too (unique per row, not predictive)
id_cols = ["customer_name", "customer_email", "ticket_id", "ticket_created_date"]
df = df.drop(columns=[c for c in id_cols if c in df.columns])

# ============================================================
# 2. Auto-detect the important columns
# ============================================================
possible_text_cols = ["clean_text", "issue_description", "ticket_description",
                       "description", "text", "ticket_text"]
possible_category_cols = ["category", "ticket_type", "Category"]
possible_priority_cols = ["priority", "ticket_priority", "Priority"]

text_col = next((c for c in possible_text_cols if c in df.columns), None)
category_col = next((c for c in possible_category_cols if c in df.columns), None)
priority_col = next((c for c in possible_priority_cols if c in df.columns), None)

if text_col is None or category_col is None or priority_col is None:
    raise ValueError(
        f"Could not auto-detect required columns. Found -> "
        f"text: {text_col}, category: {category_col}, priority: {priority_col}. "
        f"Your columns are: {df.columns.tolist()}"
    )
print(f"Using text_col='{text_col}', category_col='{category_col}', priority_col='{priority_col}'")

df = df.dropna(subset=[text_col, category_col, priority_col]).reset_index(drop=True)

# ============================================================
# 3. Clean text (skip if already cleaned by teammate, this is safe either way)
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["_clean_text"] = df[text_col].apply(clean_text)

# ============================================================
# 4. Encode targets
# ============================================================
le_category = LabelEncoder()
le_priority = LabelEncoder()
df["category_encoded"] = le_category.fit_transform(df[category_col])
df["priority_encoded"] = le_priority.fit_transform(df[priority_col])

# ============================================================
# 5. One-hot encode remaining metadata columns
# ============================================================
exclude_cols = [text_col, "_clean_text", category_col, priority_col,
                 "category_encoded", "priority_encoded"]
cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in exclude_cols]

# drop high-cardinality identifier-like columns
cat_cols = [c for c in cat_cols if df[c].nunique() < 100]

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

meta_cols = [c for c in df_encoded.columns
             if c not in exclude_cols and df_encoded[c].dtype in ["bool", "int64", "float64"]]

# ============================================================
# 6. Train/test split (ONE split, shared, stratified on category)
# ============================================================
train_df, test_df = train_test_split(
    df_encoded, test_size=0.2, random_state=42, stratify=df_encoded["category_encoded"]
)

# ============================================================
# 7. TF-IDF on text (fit on train only)
# ============================================================
tfidf = TfidfVectorizer(max_features=3000)
X_text_train = tfidf.fit_transform(train_df["_clean_text"])
X_text_test = tfidf.transform(test_df["_clean_text"])

X_meta_train = csr_matrix(train_df[meta_cols].astype(float).values)
X_meta_test = csr_matrix(test_df[meta_cols].astype(float).values)

X_train = hstack([X_text_train, X_meta_train]).tocsr()
X_test = hstack([X_text_test, X_meta_test]).tocsr()

y_cat_train, y_cat_test = train_df["category_encoded"].values, test_df["category_encoded"].values
y_pri_train, y_pri_test = train_df["priority_encoded"].values, test_df["priority_encoded"].values

print("\nX_train shape:", X_train.shape)

# ============================================================
# 8. Train + evaluate: Logistic Regression and LightGBM, for BOTH targets
# ============================================================
results = []

def run_model(model_name, model, X_train, y_train, X_test, y_test, target_name):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    results.append({
        "Target": target_name,
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Macro F1": round(f1, 4),
        "Train Time (s)": round(train_time, 2),
    })
    return model, preds

print("\nTraining Logistic Regression (category)...")
logreg_cat, logreg_cat_preds = run_model(
    "Logistic Regression", LogisticRegression(max_iter=2000),
    X_train, y_cat_train, X_test, y_cat_test, "Category"
)

print("Training LightGBM (category)...")
lgbm_cat, lgbm_cat_preds = run_model(
    "LightGBM", LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    X_train, y_cat_train, X_test, y_cat_test, "Category"
)

print("Training Logistic Regression (priority)...")
logreg_pri, logreg_pri_preds = run_model(
    "Logistic Regression", LogisticRegression(max_iter=2000),
    X_train, y_pri_train, X_test, y_pri_test, "Priority"
)

print("Training LightGBM (priority)...")
lgbm_pri, lgbm_pri_preds = run_model(
    "LightGBM", LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    X_train, y_pri_train, X_test, y_pri_test, "Priority"
)

# ============================================================
# 9. Show comparison table
# ============================================================
results_df = pd.DataFrame(results)
print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))

# ============================================================
# 10. Detailed report for whichever you want to inspect further, e.g.:
# ============================================================
# print(classification_report(y_cat_test, lgbm_cat_preds, target_names=le_category.classes_))