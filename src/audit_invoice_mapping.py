
from pathlib import Path
import csv
import json
import re
from decimal import Decimal, InvalidOperation

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "ground_truth" / "invoice_labels.csv"
OCR_RESULTS = ROOT / "outputs" / "ocr" / "ocr_results.jsonl"
REPORT = ROOT / "reports" / "invoice_mapping_report.csv"

FIELDS = [
    "invoice_number", "vendor", "invoice_date", "item",
    "quantity", "unit_price", "subtotal", "tax", "grand_total"
]

PATTERNS = {
    "invoice_number": r"Invoice\s*(?:No\.?|Number|#)\s*:\s*(.+)",
    "vendor": r"Vendor\s*:\s*(.+)",
    "invoice_date": r"Invoice\s*Date\s*:\s*(.+)",
    "item": r"Item\s*:\s*(.+)",
    "quantity": r"Quantity\s*:\s*(.+)",
    "unit_price": r"Unit\s*Price\s*:\s*(.+)",
    "subtotal": r"Subtotal\s*:\s*(.+)",
    "tax": r"Tax(?:\s*\([^)]*\))?\s*:\s*(.+)",
    "grand_total": r"Grand\s*Total\s*:\s*(.+)",
}

NUMERIC_FIELDS = {
    "quantity", "unit_price", "subtotal", "tax", "grand_total"
}


def parse_text(text):
    extracted = {}

    for field, pattern in PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        extracted[field] = match.group(1).strip() if match else ""

    return extracted


def normalize(field, value):
    value = str(value).strip()

    if field == "invoice_number":
        return value.upper()

    if field == "vendor":
        return " ".join(value.casefold().split())

    if field in NUMERIC_FIELDS:
        try:
            return Decimal(value.replace(",", "").replace("$", ""))
        except InvalidOperation:
            return None

    return value


def main():
    labels = pd.read_csv(LABELS, dtype=str).fillna("")

    if labels["invoice_number"].duplicated().any():
        raise ValueError("Ground-truth invoice numbers are not unique.")

    label_by_number = {
        row["invoice_number"].strip().upper(): row.to_dict()
        for _, row in labels.iterrows()
    }

    records = []
    with OCR_RESULTS.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    report_rows = []

    for record in records:
        text = "\n".join(record.get("texts", []))
        predicted = parse_text(text)
        invoice_number = predicted["invoice_number"].upper()
        label = label_by_number.get(invoice_number)

        row = {
            "image_file": record["image_file"],
            "ocr_invoice_number": invoice_number,
            "mapping_found": label is not None,
            "matched_label_image": (
                label["image_file"] if label else ""
            ),
        }

        for field in FIELDS:
            expected = label[field] if label else ""
            actual = predicted[field]

            if not label or not actual:
                matches = False
            else:
                left = normalize(field, actual)
                right = normalize(field, expected)
                matches = left is not None and right is not None and left == right

            row[f"ocr_{field}"] = actual
            row[f"label_{field}"] = expected
            row[f"{field}_match"] = matches

        report_rows.append(row)

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nINVOICE-NUMBER MAPPING AUDIT")
    print(f"OCR records: {len(records)}")
    print(f"Ground-truth records: {len(labels)}")
    print(
        "Records mapped by invoice number: "
        f"{sum(row['mapping_found'] for row in report_rows)}"
        f"/{len(report_rows)}"
    )

    print("\nField matches after invoice-number mapping:")
    for field in FIELDS:
        count = sum(row[f"{field}_match"] for row in report_rows)
        print(f"{field}: {count}/{len(report_rows)}")

    print(f"\nDetailed report: {REPORT}")


if __name__ == "__main__":
    main()
