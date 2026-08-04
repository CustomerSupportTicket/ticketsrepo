import os
import numpy as np
import pandas as pd

# ==========================================================
# Paths
# ==========================================================
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("Dataset/CleanData/customer_support_tickets_cleaned.csv")

# ...

# Save engineered dataset

# ==========================================================
# Load Dataset
# ==========================================================

print("Loading cleaned dataset...")



print(f"Rows : {len(df)}")
print(f"Columns : {len(df.columns)}")

# ==========================================================
# Convert Dates
# ==========================================================

df["ticket_created_date"] = pd.to_datetime(
    df["ticket_created_date"],
    errors="coerce"
)

df["ticket_resolved_date"] = pd.to_datetime(
    df["ticket_resolved_date"],
    errors="coerce"
)

# ==========================================================
# Feature 1 : Resolution Days
# ==========================================================

df["resolution_days"] = (
    df["ticket_resolved_date"] -
    df["ticket_created_date"]
).dt.days

# ==========================================================
# Feature 2 : Resolution Hours (Calculated)
# ==========================================================

df["resolution_hours_calculated"] = (
    df["ticket_resolved_date"] -
    df["ticket_created_date"]
).dt.total_seconds() / 3600

# ==========================================================
# Feature 3 : Ticket Month
# ==========================================================

df["created_month"] = df["ticket_created_date"].dt.month_name()

# ==========================================================
# Feature 4 : Ticket Year
# ==========================================================

df["created_year"] = df["ticket_created_date"].dt.year

# ==========================================================
# Feature 5 : Day of Week
# ==========================================================

df["created_day"] = df["ticket_created_date"].dt.day_name()

# ==========================================================
# Feature 6 : Weekend Ticket
# ==========================================================

df["weekend_ticket"] = np.where(
    df["ticket_created_date"].dt.weekday >= 5,
    "Yes",
    "No"
)

# ==========================================================
# Feature 7 : Customer Age Group
# ==========================================================

df["customer_age_group"] = pd.cut(

    df["customer_age"],

    bins=[0,18,25,35,45,60,120],

    labels=[
        "Below 18",
        "18-25",
        "26-35",
        "36-45",
        "46-60",
        "60+"
    ]
)

# ==========================================================
# Feature 8 : Customer Tenure Group
# ==========================================================

df["tenure_group"] = pd.cut(

    df["customer_tenure_months"],

    bins=[0,12,36,60,120,720],

    labels=[
        "New",
        "Regular",
        "Experienced",
        "Loyal",
        "Very Loyal"
    ]
)

# ==========================================================
# Feature 9 : Previous Ticket Group
# ==========================================================

df["previous_ticket_group"] = pd.cut(

    df["previous_tickets"],

    bins=[-1,2,5,10,500],

    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

# ==========================================================
# Feature 10 : Satisfaction Category
# ==========================================================

df["satisfaction_category"] = pd.cut(

    df["customer_satisfaction_score"],

    bins=[0,2,3,4,5],

    labels=[
        "Poor",
        "Average",
        "Good",
        "Excellent"
    ],

    include_lowest=True
)

# ==========================================================
# Feature 11 : Resolution Speed
# ==========================================================

df["resolution_speed"] = pd.cut(

    df["resolution_time_hours"],

    bins=[0,24,72,10000],

    labels=[
        "Fast",
        "Medium",
        "Slow"
    ]
)

# ==========================================================
# Feature 12 : Response Speed
# ==========================================================

df["response_speed"] = pd.cut(

    df["first_response_time_hours"],

    bins=[0,2,8,500],

    labels=[
        "Fast",
        "Medium",
        "Slow"
    ]
)

# ==========================================================
# Feature 13 : Complexity Category
# ==========================================================

df["complexity_category"] = pd.cut(

    df["issue_complexity_score"],

    bins=[0,3,7,10],

    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

# ==========================================================
# Feature 14 : High Priority Flag
# ==========================================================

df["high_priority"] = np.where(

    df["priority"].isin(["High","Urgent"]),

    1,

    0
)

# ==========================================================
# Feature 15 : SLA Status
# ==========================================================

df["sla_status"] = np.where(

    df["sla_breached"]=="Yes",

    "Breached",

    "Within SLA"
)

# ==========================================================
# Feature 16 : Ticket Status
# ==========================================================

df["ticket_status"] = np.where(

    df["status"].isin(["Closed","Resolved"]),

    "Resolved",

    "Pending"
)

# ==========================================================
# Feature 17 : Resolution Efficiency
# ==========================================================

df["resolution_efficiency"] = np.where(

    df["customer_satisfaction_score"] >= 4,

    "Good",

    "Needs Improvement"
)

# ==========================================================
# Save Dataset
# ==========================================================



# Save engineered dataset
df.to_csv(
    "Dataset/CleanData/customer_support_tickets_feature_engineered.csv",
    index=False
)



print("\n===================================")
print("Feature Engineering Completed")
print("===================================")
print("Rows :", len(df))
print("Columns :", len(df.columns))
print("Saved To :")
