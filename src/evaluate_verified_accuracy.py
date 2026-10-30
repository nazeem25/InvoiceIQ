
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

ALIGNMENT = REPORTS / "human_verified_alignment.csv"
LABELS = ROOT / "data" / "ground_truth" / "invoice_labels.csv"
EXTRACTIONS = ROOT / "outputs" / "extracted" / "invoice_extraction.csv"
IMAGES = ROOT / "data" / "invoices"

FIELDS = [
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

NUMERIC_FIELDS = [
    "quantity",
    "unit_price",
    "subtotal",
    "tax",
    "grand_total",
]


def normalize_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def values_match(field, actual, expected):
    if pd.isna(actual) or pd.isna(expected):
        return False

    if field in NUMERIC_FIELDS:
        try:
            return abs(float(actual) - float(expected)) <= 0.01
        except (TypeError, ValueError):
            return False

    return normalize_text(actual) == normalize_text(expected)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


alignment = pd.read_csv(ALIGNMENT, dtype=str).fillna("")
labels = pd.read_csv(LABELS)
extractions = pd.read_csv(EXTRACTIONS)

required_alignment = {
    "image_file",
    "human_verified_invoice_number",
}
if not required_alignment.issubset(alignment.columns):
    raise ValueError("Alignment CSV is missing required columns.")

if alignment["human_verified_invoice_number"].str.strip().eq("").any():
    raise ValueError(
        "Some invoice numbers are unverified. Complete the CSV first."
    )

if labels["invoice_number"].duplicated().any():
    raise ValueError("Ground-truth invoice numbers are not unique.")

if extractions["image_file"].duplicated().any():
    raise ValueError("Duplicate filenames in extraction results.")

if alignment["image_file"].duplicated().any():
    raise ValueError("Duplicate filenames in alignment CSV.")

label_lookup = labels.set_index("invoice_number", drop=False).to_dict("index")
extraction_lookup = extractions.set_index("image_file").to_dict("index")

details = []
unmatched = []
seen_hashes = set()

for _, verified in alignment.iterrows():
    image_name = verified["image_file"].strip()
    verified_number = verified["human_verified_invoice_number"].strip()

    if image_name not in extraction_lookup:
        unmatched.append({
            "image_file": image_name,
            "reason": "No extraction result for image",
        })
        continue

    if verified_number not in label_lookup:
        unmatched.append({
            "image_file": image_name,
            "reason": "Verified invoice number not found in ground truth",
        })
        continue

    image_path = IMAGES / image_name
    if not image_path.is_file():
        unmatched.append({
            "image_file": image_name,
            "reason": "Original image file not found",
        })
        continue

    label = label_lookup[verified_number]
    extracted = extraction_lookup[image_name]
    image_hash = sha256(image_path)
    is_duplicate = image_hash in seen_hashes
    seen_hashes.add(image_hash)

    row = {
        "image_file": image_name,
        "human_verified_invoice_number": verified_number,
        "matched_label_image": label["image_file"],
        "image_sha256": image_hash,
        "duplicate_image_content": is_duplicate,
    }

    for field in FIELDS:
        row[f"{field}_match"] = values_match(
            field,
            extracted.get(field),
            label.get(field),
        )

    row["all_nine_fields_match"] = all(
        row[f"{field}_match"] for field in FIELDS
    )
    details.append(row)

detail_df = pd.DataFrame(details)
unmatched_df = pd.DataFrame(unmatched)

if detail_df.empty:
    raise ValueError("No records could be evaluated.")

detail_path = REPORTS / "verified_field_comparison.csv"
detail_df.to_csv(detail_path, index=False)

unmatched_path = REPORTS / "verified_unmatched_records.csv"
unmatched_df.to_csv(unmatched_path, index=False)

summary = []

for field in FIELDS:
    column = f"{field}_match"

    for subset_name, subset in [
        ("all_matched_records", detail_df),
        (
            "unique_image_contents",
            detail_df.loc[~detail_df["duplicate_image_content"]],
        ),
    ]:
        total = len(subset)
        correct = int(subset[column].sum()) if total else 0

        summary.append({
            "metric": f"{field}_exact_match_accuracy",
            "evaluation_subset": subset_name,
            "correct": correct,
            "evaluated": total,
            "accuracy_percent": round(100 * correct / total, 2)
            if total else None,
        })

summary.append({
    "metric": "all_nine_fields_correct",
    "evaluation_subset": "all_matched_records",
    "correct": int(detail_df["all_nine_fields_match"].sum()),
    "evaluated": len(detail_df),
    "accuracy_percent": round(
        100 * detail_df["all_nine_fields_match"].mean(), 2
    ),
})

summary.append({
    "metric": "records_unmatched",
    "evaluation_subset": "all_input_records",
    "correct": 0,
    "evaluated": len(alignment),
    "accuracy_percent": None,
})

summary_path = REPORTS / "verified_field_accuracy.csv"
pd.DataFrame(summary).to_csv(summary_path, index=False)

print(f"Input alignment records: {len(alignment)}")
print(f"Successfully evaluated: {len(detail_df)}")
print(f"Unmatched records: {len(unmatched_df)}")
print(
    "Duplicate image records:",
    int(detail_df["duplicate_image_content"].sum()),
)
print("\nField accuracy on all matched records:")
for field in FIELDS:
    accuracy = detail_df[f"{field}_match"].mean() * 100
    print(f"  {field}: {accuracy:.2f}%")

print("\nReports created:")
print(f"  {summary_path}")
print(f"  {detail_path}")
print(f"  {unmatched_path}")
print("\nAccuracy is valid only if invoice numbers were independently")
print("verified against the original images.")

