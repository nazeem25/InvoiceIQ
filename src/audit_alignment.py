
from pathlib import Path
import csv
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LABELS_FILE = ROOT / "data" / "ground_truth" / "invoice_labels.csv"
OCR_FILE = ROOT / "outputs" / "ocr" / "ocr_results.jsonl"
REPORT_FILE = ROOT / "reports" / "label_alignment_report.csv"


def extract_invoice_number(text):
    match = re.search(
        r"Invoice\s*(?:No\.?|Number|#)\s*:\s*([A-Z0-9-]+)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def extract_vendor(text):
    match = re.search(
        r"^\s*Vendor\s*:\s*(.+?)\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def normalize(value):
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def main():
    if not LABELS_FILE.is_file():
        raise FileNotFoundError(f"CSV not found: {LABELS_FILE}")

    if not OCR_FILE.is_file():
        raise FileNotFoundError(f"OCR results not found: {OCR_FILE}")

    labels = pd.read_csv(LABELS_FILE, dtype=str)

    required = {"image_file", "invoice_number", "vendor"}
    if not required.issubset(labels.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    if labels["image_file"].duplicated().any():
        raise ValueError("Duplicate image filenames in ground truth.")

    ocr_by_image = {}

    with OCR_FILE.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            filename = record["image_file"]

            if filename in ocr_by_image:
                raise ValueError(f"Duplicate OCR record: {filename}")

            ocr_by_image[filename] = record

    rows = []

    for _, label in labels.iterrows():
        filename = label["image_file"]
        record = ocr_by_image.get(filename)

        if record is None:
            rows.append({
                "image_file": filename,
                "status": "missing_ocr_record",
                "ocr_invoice_number": "",
                "label_invoice_number": label["invoice_number"],
                "invoice_number_match": False,
                "ocr_vendor": "",
                "label_vendor": label["vendor"],
                "vendor_match": False,
            })
            continue

        text = "\n".join(record.get("texts", []))

        predicted_number = extract_invoice_number(text)
        predicted_vendor = extract_vendor(text)

        number_match = (
            bool(predicted_number)
            and normalize(predicted_number)
            == normalize(label["invoice_number"])
        )

        vendor_match = (
            bool(predicted_vendor)
            and normalize(predicted_vendor)
            == normalize(label["vendor"])
        )

        if record.get("status") != "success":
            status = "ocr_failed"
        elif not predicted_number or not predicted_vendor:
            status = "fields_not_found"
        elif number_match and vendor_match:
            status = "both_match"
        else:
            status = "mismatch"

        rows.append({
            "image_file": filename,
            "status": status,
            "ocr_invoice_number": predicted_number,
            "label_invoice_number": label["invoice_number"],
            "invoice_number_match": number_match,
            "ocr_vendor": predicted_vendor,
            "label_vendor": label["vendor"],
            "vendor_match": vendor_match,
        })

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    number_matches = sum(r["invoice_number_match"] for r in rows)
    vendor_matches = sum(r["vendor_match"] for r in rows)
    both_match = sum(r["status"] == "both_match" for r in rows)

    print("\nLABEL ALIGNMENT AUDIT")
    print(f"Ground-truth records: {total}")
    print(f"Invoice number matches: {number_matches}/{total}")
    print(f"Vendor matches: {vendor_matches}/{total}")
    print(f"Both fields match: {both_match}/{total}")
    print(f"Report saved: {REPORT_FILE}")

    print("\nRecords requiring inspection:")
    for row in rows:
        if row["status"] != "both_match":
            print(
                f"{row['image_file']}: {row['status']} | "
                f"OCR number={row['ocr_invoice_number']!r}, "
                f"label number={row['label_invoice_number']!r} | "
                f"OCR vendor={row['ocr_vendor']!r}, "
                f"label vendor={row['label_vendor']!r}"
            )


if __name__ == "__main__":
    main()
