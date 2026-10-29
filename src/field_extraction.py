
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = ROOT / "outputs" / "ocr" / "text"
OUTPUT_DIR = ROOT / "outputs" / "extracted"

JSON_OUTPUT = OUTPUT_DIR / "invoice_extraction.jsonl"
CSV_OUTPUT = OUTPUT_DIR / "invoice_extraction.csv"

FIELD_PATTERNS = {
    "invoice_number": r"Invoice\s+No\s*:\s*(.+)",
    "vendor": r"Vendor\s*:\s*(.+)",
    "invoice_date": r"Invoice\s+Date\s*:\s*(\d{2}-\d{2}-\d{4})",
    "item": r"Item\s*:\s*(.+)",
    "quantity": r"Quantity\s*:\s*([\d,]+)",
    "unit_price": r"Unit\s+Price\s*:\s*\$?([\d,]+(?:\.\d{1,2})?)",
    "subtotal": r"Subtotal\s*:\s*\$?([\d,]+(?:\.\d{1,2})?)",
    "tax": r"Tax(?:\s*\([^)]*\))?\s*:\s*\$?([\d,]+(?:\.\d{1,2})?)",
    "grand_total": r"Grand\s+Total\s*:\s*\$?([\d,]+(?:\.\d{1,2})?)",
}

FIELDS = list(FIELD_PATTERNS.keys())


def extract_fields(text):
    """Extract invoice fields from OCR text using labeled patterns."""
    result = {}

    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        result[field] = match.group(1).strip() if match else None

    # Normalize numeric fields by removing thousands separators.
    for field in [
        "quantity",
        "unit_price",
        "subtotal",
        "tax",
        "grand_total",
    ]:
        if result[field] is not None:
            result[field] = result[field].replace(",", "")

    return result


def main():
    if not OCR_DIR.exists():
        raise FileNotFoundError(f"OCR text directory not found: {OCR_DIR}")

    text_files = sorted(OCR_DIR.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(f"No OCR text files found in {OCR_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for text_path in text_files:
        text = text_path.read_text(encoding="utf-8")
        fields = extract_fields(text)

        missing_fields = [
            field for field in FIELDS if fields[field] is None
        ]

        record = {
            "image_file": text_path.stem + ".png",
            **fields,
            "missing_fields": missing_fields,
            "extraction_complete": len(missing_fields) == 0,
        }

        records.append(record)

    with JSON_OUTPUT.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_columns = [
        "image_file",
        *FIELDS,
        "missing_fields",
        "extraction_complete",
    ]

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()

        for record in records:
            csv_record = record.copy()
            csv_record["missing_fields"] = ";".join(
                csv_record["missing_fields"]
            )
            writer.writerow(csv_record)

    complete = sum(record["extraction_complete"] for record in records)

    print("FIELD EXTRACTION COMPLETED")
    print(f"OCR files processed: {len(records)}")
    print(f"Complete records: {complete}/{len(records)}")
    print(f"Incomplete records: {len(records) - complete}")
    print(f"JSON output: {JSON_OUTPUT}")
    print(f"CSV output: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
