
import pandas as pd

report_path = "reports/invoice_mapping_report.csv"
df = pd.read_csv(report_path)

fields = [
    "invoice_number",
    "vendor",
    "invoice_date",
    "item",
    "quantity",
    "unit_price",
    "subtotal",
    "tax",
    "grand_total",
]

print(f"Records: {len(df)}")
print(f"Mapped records: {int(df['mapping_found'].fillna(False).sum())}")

print("\nField match results:")
for field in fields:
    column = f"{field}_match"

    if column not in df.columns:
        print(f"{field}: match column unavailable")
        continue

    matches = df[column].fillna(False).astype(bool).sum()
    total = len(df)
    percentage = matches / total * 100 if total else 0

    print(f"{field}: {matches}/{total} ({percentage:.2f}%)")

print("\nUnmapped records:")
unmapped = df.loc[
    ~df["mapping_found"].fillna(False).astype(bool),
    ["image_file", "ocr_invoice_number"],
]

if unmapped.empty:
    print("None")
else:
    print(unmapped.to_string(index=False))
