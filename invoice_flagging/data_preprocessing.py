"""Data loading, labeling and feature engineering for invoice flagging."""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DB_PATH = r"C:\Users\DELL\Downloads\inventory.db"

# Business rules that define the label
AMOUNT_TOLERANCE = 5     # $ mismatch between invoice total and item-level total
DELAY_THRESHOLD = 10     # avg days from PO to receiving

# Columns used to build the label. They must NOT be model features (leakage).
LABEL_COLUMNS = ["invoice_dollars", "total_item_dollars", "avg_receiving_delay"]

# days_to_pay is excluded: it is only known after payment, so unusable at flag time.
FEATURES = [
    "invoice_quantity",
    "Freight",
    "days_po_to_invoice",
    "total_brands",
    "total_item_quantity",
    "freight_per_unit",
    "quantity_gap",
]

QUERY = """
WITH purchase_agg AS (
    SELECT
        p.PONumber,
        COUNT(DISTINCT p.Brand) AS total_brands,
        SUM(p.Quantity) AS total_item_quantity,
        SUM(p.Dollars) AS total_item_dollars,
        AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
    FROM purchases p
    GROUP BY p.PONumber
)
SELECT
    vi.PONumber,
    vi.Quantity AS invoice_quantity,
    vi.Dollars AS invoice_dollars,
    vi.Freight,
    julianday(vi.InvoiceDate) - julianday(vi.PODate) AS days_po_to_invoice,
    julianday(vi.PayDate) - julianday(vi.InvoiceDate) AS days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay
FROM vendor_invoice vi
LEFT JOIN purchase_agg pa
    ON vi.PONumber = pa.PONumber
"""


def load_data(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(QUERY, conn)
    finally:
        conn.close()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rows with no matching purchases cannot be labeled
    missing = df["total_item_dollars"].isna().sum()
    if missing:
        print(f"Dropping {missing} invoices with no matching purchase records")
    df = df.dropna(subset=["total_item_dollars", "avg_receiving_delay"])

    # One row per PO expected; warn if vendor_invoice has duplicates
    dupes = df["PONumber"].duplicated().sum()
    if dupes:
        print(f"Warning: {dupes} duplicate PONumber rows (multiple invoices per PO)")

    return df.reset_index(drop=True)


def create_label(df: pd.DataFrame) -> pd.Series:
    amount_mismatch = (df["invoice_dollars"] - df["total_item_dollars"]).abs() > AMOUNT_TOLERANCE
    high_delay = df["avg_receiving_delay"] > DELAY_THRESHOLD
    return (amount_mismatch | high_delay).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    qty = df["invoice_quantity"].replace(0, np.nan)
    df["freight_per_unit"] = (df["Freight"] / qty).fillna(0)
    df["quantity_gap"] = df["invoice_quantity"] - df["total_item_quantity"]
    return df


def prepare_dataset(db_path: str = DB_PATH):
    """Returns the full frame (with label) for inspection."""
    df = clean_data(load_data(db_path))
    df["flag_invoice"] = create_label(df)
    df = engineer_features(df)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURES]
    y = df["flag_invoice"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    data = prepare_dataset()
    print(data.shape)
    print(data["flag_invoice"].value_counts(normalize=True))