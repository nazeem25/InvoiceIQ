
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

extraction_path = ROOT / "outputs" / "extracted" / "invoice_extraction.csv"
labels_path = ROOT / "data" / "ground_truth" / "invoice_labels.csv"
output_path = ROOT / "reports" / "human_verified_alignment.csv"

extractions = pd.read_csv(extraction_path)
labels = pd.read_csv(labels_path)

if "image_file" not in extractions.columns:
    raise ValueError("Extraction CSV is missing image_file.")

if "invoice_number" not in extractions.columns:
    raise ValueError("Extraction CSV is missing invoice_number.")

if "image_file" not in labels.columns:
    raise ValueError("Ground-truth CSV is missing image_file.")

if "invoice_number" not in labels.columns:
    raise ValueError("Ground-truth CSV is missing invoice_number.")

if extractions["image_file"].duplicated().any():
    raise ValueError("Duplicate image filenames in extraction CSV.")

template = extractions[["image_file", "invoice_number"]].rename(
    columns={"invoice_number": "ocr_candidate_invoice_number"}
)

# This column must be filled by visually inspecting each original image.
template["human_verified_invoice_number"] = ""

# These columns will be populated after verification.
template["matched_label_image"] = ""
template["verification_notes"] = ""

output_path.parent.mkdir(parents=True, exist_ok=True)
template.to_csv(output_path, index=False)

print(f"Created: {output_path}")
print(f"Images requiring visual verification: {len(template)}")
print("\nIMPORTANT:")
print("- OCR candidates are suggestions, not verified answers.")
print("- Open each original image and read its printed invoice number.")
print("- Enter that number in human_verified_invoice_number.")
print("- Do not edit the original images or ground-truth CSV.")
