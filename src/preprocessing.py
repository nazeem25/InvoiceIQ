
from pathlib import Path
import csv

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "invoices"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessed"
REPORT_PATH = PROJECT_ROOT / "reports" / "preprocessing_report.csv"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def preprocess_image(image_path: Path):
    """Create grayscale and thresholded versions of an invoice."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path.name}")

    height, width = image.shape[:2]

    if height == 0 or width == 0:
        raise ValueError(f"Invalid image dimensions: {image_path.name}")

    # Convert the original image to grayscale.
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce small amounts of image noise.
    denoised = cv2.medianBlur(grayscale, 3)

    # Create a binary image for OCR experiments.
    thresholded = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    return image, grayscale, denoised, thresholded


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Invoice directory not found: {IMAGE_DIR}"
        )

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported invoice images found in {IMAGE_DIR}"
        )

    report_rows = []
    failures = []

    for image_path in image_paths:
        try:
            original, grayscale, denoised, thresholded = (
                preprocess_image(image_path)
            )

            variants = {
                "grayscale": grayscale,
                "denoised": denoised,
                "thresholded": thresholded,
            }

            saved_files = []

            for variant_name, variant_image in variants.items():
                output_path = OUTPUT_DIR / (
                    f"{image_path.stem}_{variant_name}.png"
                )

                success = cv2.imwrite(
                    str(output_path), variant_image
                )

                if not success:
                    raise OSError(
                        f"Could not save {output_path.name}"
                    )

                saved_files.append(output_path.name)

            height, width = original.shape[:2]

            report_rows.append({
                "image_file": image_path.name,
                "status": "success",
                "width": width,
                "height": height,
                "channels": original.shape[2],
                "processed_files": ";".join(saved_files),
                "error": "",
            })

            print(
                f"PASS: {image_path.name} "
                f"({width} x {height})"
            )

        except Exception as exc:
            failures.append(image_path.name)

            report_rows.append({
                "image_file": image_path.name,
                "status": "failed",
                "width": "",
                "height": "",
                "channels": "",
                "processed_files": "",
                "error": str(exc),
            })

            print(f"FAIL: {image_path.name}: {exc}")

    fieldnames = [
        "image_file",
        "status",
        "width",
        "height",
        "channels",
        "processed_files",
        "error",
    ]

    with REPORT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as report_file:
        writer = csv.DictWriter(
            report_file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n--- Preprocessing Summary ---")
    print(f"Images discovered: {len(image_paths)}")
    print(f"Successful: {len(image_paths) - len(failures)}")
    print(f"Failed: {len(failures)}")
    print(f"Report: {REPORT_PATH}")
    print(f"Processed images: {OUTPUT_DIR}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
