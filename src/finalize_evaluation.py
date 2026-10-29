
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

summary_path = REPORTS / "evaluation_summary.csv"
mapping_path = REPORTS / "invoice_mapping_report.csv"
labels_path = REPORTS / "label_alignment_report.csv"
extraction_path = ROOT / "outputs" / "extracted" / "invoice_extraction.csv"
ocr_path = ROOT / "outputs" / "ocr" / "ocr_summary.csv"
review_path = ROOT / "outputs" / "review" / "human_review_queue.csv"

summary = pd.read_csv(summary_path)
mapping = pd.read_csv(mapping_path)
labels = pd.read_csv(labels_path)
extraction = pd.read_csv(extraction_path)

def read_optional(path):
    return pd.read_csv(path) if path.exists() else None

ocr = read_optional(ocr_path)
review = read_optional(review_path)

rows = []

def add(metric, value, note):
    rows.append({
        "metric": metric,
        "value": value,
        "interpretation": note,
    })

add(
    "dataset_images",
    len(extraction),
    "Number of extracted invoice image records.",
)
add(
    "ocr_invoice_number_mapping_rate",
    f"{mapping['mapping_found'].fillna(False).astype(bool).mean() * 100:.2f}%",
    "Diagnostic mapping rate; not independent accuracy.",
)

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

for field in fields:
    column = f"{field}_match"
    if column in mapping.columns:
        matches = mapping[column].fillna(False).astype(bool).sum()
        add(
            f"diagnostic_{field}_match_rate",
            f"{matches}/{len(mapping)} ({matches / len(mapping) * 100:.2f}%)",
            "Matched after OCR invoice-number-based label selection; "
            "not an independent accuracy estimate.",
        )

for _, row in summary.iterrows():
    metric = str(row.iloc[0])
    value = row.iloc[1] if len(row) > 1 else ""
    add(
        f"operational_{metric}",
        value,
        "Operational metric from the existing evaluation summary.",
    )

add(
    "independent_field_accuracy",
    "NOT ESTABLISHED",
    "Filename-to-label alignment is inconsistent. "
    "OCR-number-based matching is diagnostic only.",
)
add(
    "dataset_alignment",
    "REQUIRES REVIEW",
    "Inspect the label alignment report and duplicate-image issue. "
    "Original images and labels were not modified.",
)

output = REPORTS / "final_evaluation_report.csv"
pd.DataFrame(rows).to_csv(output, index=False)

print(f"Saved: {output}")
print(f"Dataset records: {len(extraction)}")
print(f"Diagnostic mappings: {int(mapping['mapping_found'].fillna(False).astype(bool).sum())}")
print("Independent accuracy: NOT ESTABLISHED")
print("Original dataset files were not modified.")
