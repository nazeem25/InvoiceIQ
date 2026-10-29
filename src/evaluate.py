
from pathlib import Path
import csv
import json
import statistics

ROOT = Path(__file__).resolve().parent.parent

OCR_PATH = ROOT / "outputs" / "ocr" / "ocr_results.jsonl"
REVIEW_PATH = ROOT / "outputs" / "review" / "field_confidence.jsonl"
MAPPING_PATH = ROOT / "reports" / "invoice_mapping_report.csv"
OUTPUT_PATH = ROOT / "reports" / "evaluation_summary.csv"

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


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    ocr_records = load_jsonl(OCR_PATH)
    review_records = load_jsonl(REVIEW_PATH)

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            "Run src/audit_invoice_mapping.py first."
        )

    with MAPPING_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        mapping_records = list(csv.DictReader(file))

    if not ocr_records:
        raise ValueError("No OCR records available.")

    if not review_records:
        raise ValueError("No review records available.")

    if len({r["image_file"] for r in ocr_records}) != len(ocr_records):
        raise ValueError("Duplicate image filenames in OCR records.")

    if len({r["image_file"] for r in review_records}) != len(review_records):
        raise ValueError("Duplicate image filenames in review records.")

    successful = [
        r for r in ocr_records if r.get("status") == "success"
    ]
    failed_count = len(ocr_records) - len(successful)

    timings = [
        float(r["processing_time_seconds"])
        for r in successful
        if r.get("processing_time_seconds") is not None
    ]

    review_by_image = {
        r["image_file"]: r for r in review_records
    }

    if set(review_by_image) != {r["image_file"] for r in ocr_records}:
        raise ValueError("OCR and review records do not align by filename.")

    flagged_invoices = sum(
        r.get("needs_human_review", False)
        for r in review_records
    )

    total_fields = sum(
        len(r.get("fields", {})) for r in review_records
    )

    flagged_fields = sum(
        field.get("needs_review", False)
        for r in review_records
        for field in r.get("fields", {}).values()
    )

    metrics = [
        ("ocr_images_processed", len(ocr_records), "images"),
        ("ocr_successful", len(successful), "images"),
        ("ocr_failed", failed_count, "images"),
        (
            "ocr_success_rate_percent",
            round(100 * len(successful) / len(ocr_records), 2),
            "percent",
        ),
        (
            "ocr_latency_mean_seconds",
            round(statistics.mean(timings), 3) if timings else "",
            "seconds per successful image",
        ),
        (
            "ocr_latency_median_seconds",
            round(statistics.median(timings), 3) if timings else "",
            "seconds per successful image",
        ),
        (
            "human_review_invoice_count",
            flagged_invoices,
            "invoices",
        ),
        (
            "human_review_invoice_rate_percent",
            round(100 * flagged_invoices / len(review_records), 2),
            "percent",
        ),
        ("total_fields_reviewed", total_fields, "fields"),
        ("fields_flagged_for_review", flagged_fields, "fields"),
        (
            "field_review_rate_percent",
            round(100 * flagged_fields / total_fields, 2)
            if total_fields else "",
            "percent",
        ),
    ]

    # Diagnostic only: labels are selected by OCR-recognized invoice number.
    if mapping_records:
        for field in FIELDS:
            key = f"{field}_match"
            matches = sum(
                row.get(key, "").strip().lower() == "true"
                for row in mapping_records
            )
            metrics.append((
                f"diagnostic_{field}_matches",
                f"{matches}/{len(mapping_records)}",
                "invoice-number-mapped comparisons; not independent accuracy",
            ))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value", "unit_or_interpretation"])
        writer.writerows(metrics)

    print("EVALUATION SUMMARY")
    print(f"OCR images: {len(ocr_records)}")
    print(f"OCR successful: {len(successful)}")
    print(f"OCR failed: {failed_count}")
    print(
        f"Mean OCR latency: "
        f"{statistics.mean(timings):.2f} seconds"
        if timings else "Mean OCR latency: unavailable"
    )
    print(
        f"Median OCR latency: "
        f"{statistics.median(timings):.2f} seconds"
        if timings else "Median OCR latency: unavailable"
    )
    print(
        f"Invoices flagged for review: "
        f"{flagged_invoices}/{len(review_records)}"
    )
    print(
        f"Fields flagged for review: {flagged_fields}/{total_fields}"
    )
    print("Mapped field matches are diagnostic, not independent accuracy.")
    print(f"Report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
