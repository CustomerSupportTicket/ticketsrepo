import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_xgboost(data_path, target_column="customer_satisfaction_score"):
    print(f"Loading dataset from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find file at '{data_path}'. Check the path and try again."
        )

    df = pd.read_csv(data_path)
    print(f"Initial shape: {df.shape}")

    # 1. Drop identifier, text description, and date columns (prevents leakage/errors)
    drop_cols = [
        "ticket_id",
        "customer_name",
        "customer_email",
        "issue_description",
        "resolution_notes",
        "ticket_created_date",
        "ticket_resolved_date",
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 2. Check and separate Features (X) and Target (y)
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Drop any missing target values if present
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    # 3. Convert all text/object columns to category dtype for XGBoost native support
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    # Convert binary booleans to integer flags (0 or 1)
    bool_cols = X.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # 4. Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples")

    # 5. Initialize and Train XGBoost Regressor
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,  # Enables native categorical feature processing
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)

    # 6. Evaluate Model Predictions
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n================ MODEL EVALUATION ================")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"R² Score:                       {r2:.4f}")

    # 7. Print Feature Importances
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_10 = importances.sort_values(ascending=False).head(10)

    print("\n================ TOP 10 FEATURES ================")
    print(top_10.to_string())

    return model


if __name__ == "__main__":
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check for cleaned CSV in root project folder
    data_path = os.path.join(
        script_dir, "..", "customer_support_tickets_cleaned.csv"
    )

    # If cleaned dataset hasn't been saved yet, use raw dataset in Dataset folder
    if not os.path.exists(data_path):
        data_path = os.path.join(
            script_dir, "..", "Dataset", "customer_support_tickets_200k.csv"
        )

    # Execute training pipeline
    trained_model = train_xgboost(
        data_path=data_path, target_column="customer_satisfaction_score"
    )