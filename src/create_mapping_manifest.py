
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

REPORT_PATH = ROOT / "reports" / "invoice_mapping_report.csv"
OUTPUT_PATH = ROOT / "reports" / "image_to_ground_truth_mapping.csv"

df = pd.read_csv(REPORT_PATH, dtype=str)

required_columns = [
    "image_file",
    "ocr_invoice_number",
    "mapping_found",
    "matched_label_image",
    "label_invoice_number",
    "invoice_number_match",
    "vendor_match",
    "invoice_date_match",
    "item_match",
    "quantity_match",
    "unit_price_match",
    "subtotal_match",
    "tax_match",
    "grand_total_match",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in audit report: {missing}")

if len(df) != 30:
    raise ValueError(f"Expected 30 invoices, found {len(df)}")

if df["image_file"].duplicated().any():
    raise ValueError("Duplicate image filenames found.")

if df["ocr_invoice_number"].duplicated().any():
    raise ValueError("Duplicate OCR invoice numbers found.")

if df["label_invoice_number"].duplicated().any():
    raise ValueError("Duplicate mapped invoice numbers found.")

boolean_columns = required_columns[2:3] + required_columns[5:]

for column in boolean_columns:
    invalid = ~df[column].str.strip().str.lower().isin(["true", "false"])
    if invalid.any():
        raise ValueError(f"Invalid boolean values in {column}")

    df[column] = df[column].str.strip().str.lower().eq("true")

if not df["mapping_found"].all():
    raise ValueError("Some invoices have no ground-truth mapping.")

if not df["invoice_number_match"].all():
    raise ValueError("Some invoice numbers do not match.")

field_match_columns = [
    "vendor_match",
    "invoice_date_match",
    "item_match",
    "quantity_match",
    "unit_price_match",
    "subtotal_match",
    "tax_match",
    "grand_total_match",
]

if not df[field_match_columns].all().all():
    raise ValueError("At least one field mismatch exists.")

manifest = df[
    [
        "image_file",
        "ocr_invoice_number",
        "matched_label_image",
        "label_invoice_number",
    ]
].rename(
    columns={
        "ocr_invoice_number": "invoice_number",
        "matched_label_image": "ground_truth_image_file",
        "label_invoice_number": "ground_truth_invoice_number",
    }
)

manifest.to_csv(OUTPUT_PATH, index=False)

print("MAPPING MANIFEST CREATED")
print(f"Images mapped: {len(manifest)}/30")
print(f"Unique invoice numbers: {manifest['invoice_number'].nunique()}")
print(f"Output: {OUTPUT_PATH}")
print("\nFirst five mappings:")
print(manifest.head().to_string(index=False))
