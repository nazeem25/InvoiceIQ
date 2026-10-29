
from pathlib import Path
import sys

import pandas as pd


# Resolve paths from the project root, not the terminal's current folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "invoices"
LABELS_FILE = (
    PROJECT_ROOT / "data" / "ground_truth" / "invoice_labels.csv"
)

REQUIRED_COLUMNS = [
    "image_file",
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def audit_dataset():
    errors = []
    warnings = []

    print("=" * 55)
    print("InvoiceIQ - Dataset Audit")
    print("=" * 55)

    # 1. Check whether the expected files and folders exist.
    if not LABELS_FILE.is_file():
        print(f"ERROR: Labels CSV not found: {LABELS_FILE}")
        return False

    if not IMAGE_DIR.is_dir():
        print(f"ERROR: Image folder not found: {IMAGE_DIR}")
        return False

    # 2. Load the ground-truth CSV.
    try:
        df = pd.read_csv(LABELS_FILE)
    except Exception as exc:
        print(f"ERROR: Could not read CSV: {exc}")
        return False

    print(f"\nCSV rows: {len(df)}")
    print(f"CSV columns: {len(df.columns)}")

    # 3. Check required columns.
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    else:
        print("PASS: All required columns are present.")

    # Stop checks that depend on missing columns.
    if missing_columns:
        for error in errors:
            print(f"ERROR: {error}")
        return False

    # 4. Check empty values.
    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    null_counts = null_counts[null_counts > 0]

    if not null_counts.empty:
        errors.append(f"Missing values:\n{null_counts.to_string()}")
    else:
        print("PASS: No missing values in required columns.")

    # Also detect blank strings, not just CSV null values.
    text_columns = [
        "image_file",
        "invoice_number",
        "vendor",
        "invoice_date",
        "item",
    ]

    for column in text_columns:
        blank_mask = (
            df[column].astype("string").str.strip().eq("")
        ).fillna(False)

        if blank_mask.any():
            errors.append(
                f"Blank values in '{column}': "
                f"{int(blank_mask.sum())} row(s)"
            )

    # 5. Check duplicate image filenames.
    if df["image_file"].duplicated().any():
        duplicates = df.loc[
            df["image_file"].duplicated(keep=False), "image_file"
        ].tolist()
        errors.append(f"Duplicate image_file entries: {duplicates}")
    else:
        print("PASS: No duplicate image_file entries.")

    # 6. Check that image filenames are safe relative filenames.
    filenames = df["image_file"].dropna().astype(str).str.strip()

    unsafe_names = filenames[
        filenames.map(
            lambda name: (
                not name
                or Path(name).name != name
                or "/" in name
                or "\\" in name
                or Path(name).suffix.lower() not in IMAGE_EXTENSIONS
            )
        )
    ]

    if not unsafe_names.empty:
        errors.append(
            "Invalid image filenames or extensions: "
            f"{unsafe_names.tolist()}"
        )

    # 7. Verify every CSV filename exists on disk.
    expected_names = set(filenames)
    actual_names = {
        path.name
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    missing_images = sorted(expected_names - actual_names)
    extra_images = sorted(actual_names - expected_names)

    if missing_images:
        errors.append(f"Images missing from folder: {missing_images}")
    else:
        print("PASS: Every CSV image filename exists.")

    if extra_images:
        warnings.append(
            f"Images not referenced by CSV: {extra_images}"
        )

    # 8. Check numeric fields.
    numeric_columns = [
        "quantity",
        "unit_price",
        "subtotal",
        "tax",
        "grand_total",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")

        if values.isna().any():
            errors.append(
                f"Non-numeric or missing values in '{column}'."
            )
        elif (values < 0).any():
            warnings.append(
                f"Negative values found in '{column}'."
            )

    # 9. Print summary.
    print("\n" + "-" * 55)
    print(f"Images referenced by CSV: {len(expected_names)}")
    print(f"Invoice images found: {len(actual_names)}")

    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        print("\nAUDIT FAILED")
        for error in errors:
            print(f"ERROR: {error}")
        return False

    print("\nAUDIT PASSED")
    print("The dataset passed the implemented structural checks.")
    print("This does not yet verify OCR accuracy or invoice arithmetic.")
    return True


if __name__ == "__main__":
    success = audit_dataset()
    sys.exit(0 if success else 1)
