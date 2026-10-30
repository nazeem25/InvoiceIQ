
import json
import csv
import re
from pathlib import Path
from decimal import Decimal, InvalidOperation
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_FILE = (
    ROOT / "data" / "external_ground_truth"
    / "external_invoice_labels.csv"
)

PREDICTION_FILE = (
    ROOT / "outputs" / "external_test" / "llm"
    / "external_llm_extraction.json"
)

REPORT_DIR = ROOT / "reports" / "external_test"

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

NUMERIC_FIELDS = {
    "quantity",
    "unit_price",
    "subtotal",
    "tax",
    "grand_total",
}

DATE_FORMATS = [
    "%d-%b-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
]


def read_ground_truth():
    with GROUND_TRUTH_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def read_predictions():
    with PREDICTION_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Prediction JSON must contain a list.")

    return data


def clean_text(value):
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_text(value):
    return clean_text(value).casefold()


def normalize_number(value):
    try:
        cleaned = str(value).replace(",", "").strip()
        number = Decimal(cleaned)

        if not number.is_finite():
            return None

        return number.normalize()

    except (InvalidOperation, ValueError, TypeError):
        return None


def normalize_date(value):
    cleaned = clean_text(value)

    for date_format in DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, date_format)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return normalize_text(cleaned)


def normalize_field(field, value):
    if field in NUMERIC_FIELDS:
        return normalize_number(value)

    if field == "invoice_date":
        return normalize_date(value)

    return normalize_text(value)


def strict_field_match(field, expected, predicted):
    if expected is None or predicted is None:
        return False

    # Strict comparison ignores only leading/trailing whitespace.
    expected = str(expected).strip()
    predicted = str(predicted).strip()

    if field in NUMERIC_FIELDS:
        expected_number = normalize_number(expected)
        predicted_number = normalize_number(predicted)

        return (
            expected_number is not None
            and predicted_number is not None
            and expected_number == predicted_number
        )

    return expected == predicted


def normalized_field_match(field, expected, predicted):
    if expected is None or predicted is None:
        return False

    expected_value = normalize_field(field, expected)
    predicted_value = normalize_field(field, predicted)

    return (
        expected_value is not None
        and predicted_value is not None
        and expected_value == predicted_value
    )


def validate_arithmetic(record):
    try:
        quantity = normalize_number(record.get("quantity"))
        unit_price = normalize_number(record.get("unit_price"))
        subtotal = normalize_number(record.get("subtotal"))
        tax = normalize_number(record.get("tax"))
        grand_total = normalize_number(record.get("grand_total"))

        values = [
            quantity,
            unit_price,
            subtotal,
            tax,
            grand_total,
        ]

        if any(value is None for value in values):
            return False

        subtotal_ok = (
            abs(quantity * unit_price - subtotal)
            <= Decimal("0.02")
        )

        total_ok = (
            abs(subtotal + tax - grand_total)
            <= Decimal("0.02")
        )

        return subtotal_ok and total_ok

    except (TypeError, InvalidOperation):
        return False


def main():
    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(GROUND_TRUTH_FILE)

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(PREDICTION_FILE)

    ground_truth = read_ground_truth()
    predictions = read_predictions()

    truth_map = {
        row["record_id"]: row
        for row in ground_truth
    }

    prediction_map = {
        row["record_id"]: row
        for row in predictions
        if row.get("record_id")
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    field_stats = {
        field: {
            "strict_matches": 0,
            "normalized_matches": 0,
            "total": len(ground_truth),
        }
        for field in FIELDS
    }

    comparison_rows = []
    invoice_results = []

    for record_id, expected_record in truth_map.items():
        predicted_record = prediction_map.get(record_id, {})

        api_success = (
            predicted_record.get("llm_status") == "success"
        )

        strict_matches = 0
        normalized_matches = 0
        missing_fields = []
        field_details = {}

        for field in FIELDS:
            expected = expected_record.get(field)
            predicted = predicted_record.get(field)

            strict_match = strict_field_match(
                field, expected, predicted
            )

            normalized_match = normalized_field_match(
                field, expected, predicted
            )

            if strict_match:
                strict_matches += 1
                field_stats[field]["strict_matches"] += 1

            if normalized_match:
                normalized_matches += 1
                field_stats[field]["normalized_matches"] += 1

            if predicted is None or str(predicted).strip() == "":
                missing_fields.append(field)

            field_details[field] = {
                "expected": expected,
                "predicted": predicted,
                "strict_match": strict_match,
                "normalized_match": normalized_match,
            }

            comparison_rows.append({
                "record_id": record_id,
                "field": field,
                "expected": expected,
                "predicted": predicted,
                "strict_match": strict_match,
                "normalized_match": normalized_match,
            })

        arithmetic_valid = (
            api_success
            and validate_arithmetic(predicted_record)
        )

        invoice_results.append({
            "record_id": record_id,
            "api_success": api_success,
            "strict_matches": strict_matches,
            "normalized_matches": normalized_matches,
            "total_fields": len(FIELDS),
            "strict_invoice_match": strict_matches == len(FIELDS),
            "normalized_invoice_match": (
                normalized_matches == len(FIELDS)
            ),
            "missing_fields": missing_fields,
            "arithmetic_valid": arithmetic_valid,
            "field_details": field_details,
        })

    total_fields = len(ground_truth) * len(FIELDS)

    strict_total = sum(
        item["strict_matches"]
        for item in invoice_results
    )

    normalized_total = sum(
        item["normalized_matches"]
        for item in invoice_results
    )

    successful_api = sum(
        item["api_success"]
        for item in invoice_results
    )

    arithmetic_passed = sum(
        item["arithmetic_valid"]
        for item in invoice_results
    )

    strict_invoice_matches = sum(
        item["strict_invoice_match"]
        for item in invoice_results
    )

    normalized_invoice_matches = sum(
        item["normalized_invoice_match"]
        for item in invoice_results
    )

    summary = {
        "dataset": "External synthetic invoice test",
        "ground_truth_records": len(ground_truth),
        "prediction_records": len(predictions),
        "fields_per_invoice": len(FIELDS),
        "total_expected_fields": total_fields,
        "successful_api_records": successful_api,
        "strict_field_accuracy_percent": round(
            strict_total / total_fields * 100, 2
        ) if total_fields else 0,
        "normalized_field_accuracy_percent": round(
            normalized_total / total_fields * 100, 2
        ) if total_fields else 0,
        "strict_invoice_exact_match_percent": round(
            strict_invoice_matches / len(ground_truth) * 100, 2
        ) if ground_truth else 0,
        "normalized_invoice_exact_match_percent": round(
            normalized_invoice_matches / len(ground_truth) * 100, 2
        ) if ground_truth else 0,
        "arithmetic_validation_passed": arithmetic_passed,
        "arithmetic_validation_percent": round(
            arithmetic_passed / len(ground_truth) * 100, 2
        ) if ground_truth else 0,
        "field_accuracy": {},
        "note": (
            "Results apply only to this four-invoice synthetic test set. "
            "Normalized accuracy ignores text case and whitespace, "
            "normalizes numeric formatting, and recognizes supported "
            "date formats. It does not establish real-world accuracy."
        ),
    }

    for field, stats in field_stats.items():
        total = stats["total"]

        summary["field_accuracy"][field] = {
            "strict_matches": stats["strict_matches"],
            "normalized_matches": stats["normalized_matches"],
            "total": total,
            "strict_accuracy_percent": round(
                stats["strict_matches"] / total * 100, 2
            ) if total else 0,
            "normalized_accuracy_percent": round(
                stats["normalized_matches"] / total * 100, 2
            ) if total else 0,
        }

    # Save field-by-field comparison.
    comparison_file = (
        REPORT_DIR / "external_field_comparison.csv"
    )

    with comparison_file.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "record_id",
                "field",
                "expected",
                "predicted",
                "strict_match",
                "normalized_match",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Save invoice-level results.
    invoice_file = REPORT_DIR / "external_invoice_results.json"

    with invoice_file.open("w", encoding="utf-8") as file:
        json.dump(invoice_results, file, indent=4, ensure_ascii=False)

    # Save overall summary.
    summary_file = REPORT_DIR / "external_evaluation_summary.json"

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    # Save field accuracy table.
    field_file = REPORT_DIR / "external_field_accuracy.csv"

    with field_file.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "field",
                "strict_matches",
                "normalized_matches",
                "total",
                "strict_accuracy_percent",
                "normalized_accuracy_percent",
            ],
        )

        writer.writeheader()

        for field, stats in summary["field_accuracy"].items():
            writer.writerow({
                "field": field,
                **stats,
            })

    print("\n========== EXTERNAL INVOICE EVALUATION ==========")
    print(f"Ground-truth invoices: {len(ground_truth)}")
    print(f"Successful Gemini records: {successful_api}")
    print(f"Total expected fields: {total_fields}")

    print(
        "Strict field accuracy: "
        f"{summary['strict_field_accuracy_percent']}%"
    )

    print(
        "Normalized field accuracy: "
        f"{summary['normalized_field_accuracy_percent']}%"
    )

    print(
        "Strict invoice exact match: "
        f"{summary['strict_invoice_exact_match_percent']}%"
    )

    print(
        "Normalized invoice exact match: "
        f"{summary['normalized_invoice_exact_match_percent']}%"
    )

    print(
        "Arithmetic validation: "
        f"{arithmetic_passed}/{len(ground_truth)}"
    )

    print("\nField-level accuracy:")

    for field, stats in summary["field_accuracy"].items():
        print(
            f"{field}: "
            f"{stats['normalized_accuracy_percent']}%"
        )

    print("\nReports saved:")
    print(comparison_file)
    print(invoice_file)
    print(summary_file)
    print(field_file)


if __name__ == "__main__":
    main()
