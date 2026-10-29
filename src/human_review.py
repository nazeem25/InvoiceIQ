
from pathlib import Path
from decimal import Decimal, InvalidOperation
import csv
import json
import re

ROOT = Path(__file__).resolve().parent.parent

OCR_PATH = ROOT / "outputs" / "ocr" / "ocr_results.jsonl"
EXTRACTION_PATH = ROOT / "outputs" / "extracted" / "invoice_extraction.jsonl"
VALIDATION_PATH = ROOT / "outputs" / "extracted" / "validation_report.csv"

OUTPUT_DIR = ROOT / "outputs" / "review"
DETAIL_JSON = OUTPUT_DIR / "field_confidence.jsonl"
QUEUE_CSV = OUTPUT_DIR / "human_review_queue.csv"

# Fields below this OCR confidence are flagged for review.
CONFIDENCE_THRESHOLD = 0.98

FIELD_LABELS = {
    "invoice_number": r"Invoice\s+No\s*:",
    "vendor": r"Vendor\s*:",
    "invoice_date": r"Invoice\s+Date\s*:",
    "item": r"Item\s*:",
    "quantity": r"Quantity\s*:",
    "unit_price": r"Unit\s+Price\s*:",
    "subtotal": r"Subtotal\s*:",
    "tax": r"Tax(?:\s*\([^)]*\))?\s*:",
    "grand_total": r"Grand\s+Total\s*:",
}

def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_validation():
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Run src/validate_extraction.py first: {VALIDATION_PATH}"
        )

    with VALIDATION_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return {
            row["image_file"]: row
            for row in csv.DictReader(file)
        }


def find_line_confidence(field, texts, scores):
    """Return confidence for the OCR line containing this field label."""
    pattern = FIELD_LABELS[field]

    for index, text in enumerate(texts):
        if re.search(pattern, text, flags=re.IGNORECASE):
            if index < len(scores):
                try:
                    return float(scores[index])
                except (TypeError, ValueError):
                    return None
            return None

    return None


def is_missing(value):
    return value is None or str(value).strip() == ""


def field_review_reasons(value, confidence, threshold):
    """Return reasons a field should be reviewed."""
    reasons = []

    if is_missing(value):
        reasons.append("missing_field")

    if confidence is None:
        reasons.append("confidence_unavailable")
    elif confidence < threshold:
        reasons.append("low_ocr_confidence")

    return reasons


def invoice_needs_review(field_results, validation_valid, ocr_status):
    """Flag an invoice if OCR, validation, or field checks require review."""
    if ocr_status != "success":
        return True

    if validation_valid is not True:
        return True

    return any(
        result["needs_review"]
        for result in field_results.values()
    )



def main():
    ocr_records = load_jsonl(OCR_PATH)
    extracted_records = load_jsonl(EXTRACTION_PATH)
    validation_records = load_validation()

    ocr_by_image = {
        row["image_file"]: row for row in ocr_records
    }
    extracted_by_image = {
        row["image_file"]: row for row in extracted_records
    }

    if len(ocr_by_image) != len(ocr_records):
        raise ValueError("Duplicate image filenames in OCR records.")

    if len(extracted_by_image) != len(extracted_records):
        raise ValueError("Duplicate image filenames in extraction records.")

    if set(ocr_by_image) != set(extracted_by_image):
        raise ValueError(
            "OCR and extraction files do not cover the same images."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    details = []
    queue = []

    for image_file in sorted(ocr_by_image):
        ocr = ocr_by_image[image_file]
        extracted = extracted_by_image[image_file]
        validation = validation_records.get(image_file)

        field_results = {}
        reasons = []

        if ocr.get("status") != "success":
            reasons.append("ocr_failed")

        if validation is None:
            reasons.append("validation_result_missing")
        elif validation.get("valid", "").strip().lower() != "true":
            reasons.append("validation_failed")

        texts = ocr.get("texts") or []
        scores = ocr.get("confidence_scores") or []

        for field in FIELD_LABELS:
            value = extracted.get(field)
            confidence = find_line_confidence(field, texts, scores)
            field_reasons = []

            field_reasons = field_review_reasons(
    value, confidence, CONFIDENCE_THRESHOLD
)

            field_results[field] = {
                "value": value,
                "ocr_line_confidence": confidence,
                "needs_review": bool(field_reasons),
                "review_reasons": field_reasons,
            }

            if field_reasons:
                reasons.extend(
                    f"{field}:{reason}" for reason in field_reasons
                )

        needs_review = invoice_needs_review(
    field_results,
    validation.get("valid", "").strip().lower() == "true"
    if validation else None,
    ocr.get("status"),
)

        detail = {
            "image_file": image_file,
            "invoice_number": extracted.get("invoice_number"),
            "fields": field_results,
            "validation_valid": (
                validation.get("valid", "").strip().lower() == "true"
                if validation else None
            ),
            "needs_human_review": needs_review,
            "review_reasons": reasons,
        }
        details.append(detail)

        if needs_review:
            queue.append({
                "image_file": image_file,
                "invoice_number": extracted.get("invoice_number"),
                "review_reasons": ";".join(reasons),
                "fields_needing_review": ";".join(
                    field for field, result in field_results.items()
                    if result["needs_review"]
                ),
            })

    with DETAIL_JSON.open("w", encoding="utf-8") as file:
        for row in details:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with QUEUE_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        columns = [
            "image_file",
            "invoice_number",
            "review_reasons",
            "fields_needing_review",
        ]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(queue)

    print("HUMAN REVIEW QUEUE GENERATED")
    print(f"Records checked: {len(details)}")
    print(f"Records flagged for review: {len(queue)}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"Field-level details: {DETAIL_JSON}")
    print(f"Review queue: {QUEUE_CSV}")


if __name__ == "__main__":
    main()
