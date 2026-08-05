import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("Dataset/CleanData/customer_support_tickets_cleaned.csv")

# Remove rows having missing targets
df = df.dropna(subset=["category", "priority"])

# ---------------------------------------------------
# Drop unwanted columns
# ---------------------------------------------------

drop_cols = [
    "ticket_id",
    "customer_name",
    "customer_email",
    "resolution_notes",
    "ticket_created_date",
    "ticket_resolved_date"
]

df = df.drop(columns=drop_cols)

# ---------------------------------------------------
# Features
# ---------------------------------------------------

X = df.drop(columns=["category", "priority"])

# Detect categorical and numerical columns

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        cat_cols
    ),
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        num_cols
    )
])

# ===================================================
# CATEGORY MODEL
# ===================================================

y_category = df["category"]

le_category = LabelEncoder()
y_category = le_category.fit_transform(y_category)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_category,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)

category_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ))
])

category_model.fit(X_train, y_train)

pred = category_model.predict(X_test)

print("\nCATEGORY MODEL")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# ===================================================
# PRIORITY MODEL
# ===================================================

y_priority = df["priority"]

le_priority = LabelEncoder()
y_priority = le_priority.fit_transform(y_priority)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_priority
)

priority_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ))
])

priority_model.fit(X_train2, y_train2)

pred2 = priority_model.predict(X_test2)

print("\nPRIORITY MODEL")
print("Accuracy:", accuracy_score(y_test2, pred2))
print(classification_report(y_test2, pred2))

# ---------------------------------------------------
# Save Models
# ---------------------------------------------------

os.makedirs("ML/Models", exist_ok=True)

joblib.dump(category_model, "ML/Models/category_model.pkl")
joblib.dump(priority_model, "ML/Models/priority_model.pkl")

joblib.dump(le_category, "ML/Models/category_label_encoder.pkl")
joblib.dump(le_priority, "ML/Models/priority_label_encoder.pkl")

print("\nModels saved successfully.")