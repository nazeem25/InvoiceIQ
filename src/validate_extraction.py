
from pathlib import Path
from decimal import Decimal, InvalidOperation
from datetime import datetime
import json
import csv

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "outputs" / "extracted" / "invoice_extraction.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "extracted" / "validation_report.csv"

NUMERIC_FIELDS = [
    "quantity",
    "unit_price",
    "subtotal",
    "tax",
    "grand_total",
]

REQUIRED_FIELDS = [
    "invoice_number",
    "vendor",
    "invoice_date",
    "item",
    *NUMERIC_FIELDS,
]


def to_decimal(value):
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def validate_record(record):
    issues = []

    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if value is None or str(value).strip() == "":
            issues.append(f"missing_{field}")

    date_value = record.get("invoice_date")
    if date_value:
        try:
            datetime.strptime(date_value, "%d-%m-%Y")
        except ValueError:
            issues.append("invalid_invoice_date")

    numbers = {}
    for field in NUMERIC_FIELDS:
        value = record.get(field)
        number = to_decimal(value) if value is not None else None

        if number is None:
            if value is not None:
                issues.append(f"invalid_{field}")
        else:
            numbers[field] = number

    if numbers.get("quantity", Decimal("0")) <= 0:
        if "quantity" in numbers:
            issues.append("nonpositive_quantity")

    for field in ["unit_price", "subtotal", "tax", "grand_total"]:
        if field in numbers and numbers[field] < 0:
            issues.append(f"negative_{field}")

    # Arithmetic checks use a one-cent tolerance for rounding.
    if all(k in numbers for k in ["quantity", "unit_price", "subtotal"]):
        calculated_subtotal = (
            numbers["quantity"] * numbers["unit_price"]
        ).quantize(Decimal("0.01"))

        if abs(calculated_subtotal - numbers["subtotal"]) > Decimal("0.01"):
            issues.append("subtotal_arithmetic_mismatch")

    if all(k in numbers for k in ["subtotal", "tax", "grand_total"]):
        calculated_total = (
            numbers["subtotal"] + numbers["tax"]
        ).quantize(Decimal("0.01"))

        if abs(calculated_total - numbers["grand_total"]) > Decimal("0.01"):
            issues.append("grand_total_arithmetic_mismatch")

    return issues


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Extraction output not found: {INPUT_PATH}"
        )

    records = []
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("No extracted records found.")

    report = []
    for record in records:
        issues = validate_record(record)
        report.append({
            "image_file": record.get("image_file"),
            "invoice_number": record.get("invoice_number"),
            "valid": len(issues) == 0,
            "issue_count": len(issues),
            "issues": ";".join(issues),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_file",
                "invoice_number",
                "valid",
                "issue_count",
                "issues",
            ],
        )
        writer.writeheader()
        writer.writerows(report)

    valid_count = sum(row["valid"] for row in report)

    print("EXTRACTION VALIDATION COMPLETED")
    print(f"Records checked: {len(report)}")
    print(f"Valid records: {valid_count}/{len(report)}")
    print(f"Records with issues: {len(report) - valid_count}")
    print(f"Report: {OUTPUT_PATH}")

    for row in report:
        if not row["valid"]:
            print(
                f"{row['image_file']} ({row['invoice_number']}): "
                f"{row['issues']}"
            )


if __name__ == "__main__":
    main()
