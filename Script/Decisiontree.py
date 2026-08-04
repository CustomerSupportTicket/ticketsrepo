import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the cleaned dataset
import os
import pandas as pd

# Automatically resolve paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "Dataset", "CleanData", "customer_support_tickets_cleaned.csv")

df = pd.read_csv(csv_path)

# 2. Define features to use for prediction (excluding IDs, raw text, and timestamps)
features = [
    'product', 'status', 'channel', 'region', 'customer_age', 
    'customer_gender', 'subscription_type', 'customer_tenure_months', 
    'previous_tickets', 'customer_satisfaction_score', 'first_response_time_hours', 
    'resolution_time_hours', 'escalated', 'sla_breached', 'operating_system', 
    'browser', 'payment_method', 'language', 'preferred_contact_time', 
    'issue_complexity_score', 'customer_segment'
]

numeric_features = [
    'customer_age', 'customer_tenure_months', 'previous_tickets', 
    'customer_satisfaction_score', 'first_response_time_hours', 
    'resolution_time_hours', 'issue_complexity_score'
]

categorical_features = [
    f for f in features if f not in numeric_features
]

# Build preprocessing transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

print("--- TRAINING MODEL 1: PREDICTING CATEGORY ---")
X_cat = df[features].copy()
y_cat = df['category'].copy()

# Handle any missing target rows
mask_cat = y_cat.notna()
X_cat, y_cat = X_cat[mask_cat], y_cat[mask_cat]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cat, y_cat, test_size=0.2, random_state=42
)

cat_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=15))
])

cat_pipeline.fit(X_train_c, y_train_c)
y_pred_c = cat_pipeline.predict(X_test_c)

print(f"Category Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
print("\nClassification Report (Category):")
print(classification_report(y_test_c, y_pred_c))


print("\n--- TRAINING MODEL 2: PREDICTING PRIORITY ---")
X_pri = df[features].copy()
y_pri = df['priority'].copy()

mask_pri = y_pri.notna()
X_pri, y_pri = X_pri[mask_pri], y_pri[mask_pri]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_pri, y_pri, test_size=0.2, random_state=42
)

pri_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=15))
])

pri_pipeline.fit(X_train_p, y_train_p)
y_pred_p = pri_pipeline.predict(X_test_p)

print(f"Priority Accuracy: {accuracy_score(y_test_p, y_pred_p):.4f}")
print("\nClassification Report (Priority):")
print(classification_report(y_test_p, y_pred_p))