
from pathlib import Path
import csv
import json
import time

from paddleocr import PaddleOCR


# Resolve paths from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "invoices"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ocr"
TEXT_DIR = OUTPUT_DIR / "text"
SUMMARY_FILE = OUTPUT_DIR / "ocr_summary.csv"
JSONL_FILE = OUTPUT_DIR / "ocr_results.jsonl"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def extract_result_fields(result_item):
    """Extract recognized text and confidence scores from PaddleOCR 3.x."""

    data = getattr(result_item, "json", result_item)

    # Some PaddleOCR result objects expose their JSON as a string.
    if isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        raise ValueError("Unexpected PaddleOCR result format.")

    # PaddleOCR 3.x may wrap recognition data inside a 'res' key.
    if isinstance(data.get("res"), dict):
        data = data["res"]

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])

    if texts is None:
        texts = []
    if scores is None:
        scores = []

    texts = [str(text) for text in texts]
    scores = [float(score) for score in scores]

    if len(texts) != len(scores):
        raise ValueError(
            "Mismatch between recognized texts and confidence scores."
        )

    return texts, scores


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Invoice image directory not found: {IMAGE_DIR}"
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing PaddleOCR on CPU...")

    # PaddleOCR 3.x API.
    ocr = PaddleOCR(
        lang="en",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    summary_rows = []
    raw_results = []
    failures = []

    for index, image_path in enumerate(image_paths, start=1):
        print(
            f"\n[{index}/{len(image_paths)}] "
            f"Processing {image_path.name}"
        )

        start_time = time.perf_counter()

        try:
            # Use original images for the initial OCR baseline.
            results = ocr.predict(input=str(image_path))

            texts = []
            scores = []

            for result_item in results:
                item_texts, item_scores = extract_result_fields(
                    result_item
                )
                texts.extend(item_texts)
                scores.extend(item_scores)

            elapsed_seconds = time.perf_counter() - start_time
            full_text = "\n".join(texts)

            # Save the raw recognized text separately for each invoice.
            text_path = TEXT_DIR / f"{image_path.stem}.txt"
            text_path.write_text(
                full_text + ("\n" if full_text else ""),
                encoding="utf-8",
            )

            mean_confidence = (
                sum(scores) / len(scores) if scores else None
            )

            summary_rows.append({
                "image_file": image_path.name,
                "status": "success",
                "recognized_segments": len(texts),
                "mean_confidence": mean_confidence,
                "processing_time_seconds": round(
                    elapsed_seconds, 4
                ),
                "text_file": str(text_path.relative_to(PROJECT_ROOT)),
                "error": "",
            })

            raw_results.append({
                "image_file": image_path.name,
                "status": "success",
                "texts": texts,
                "confidence_scores": scores,
                "processing_time_seconds": round(
                    elapsed_seconds, 4
                ),
                "error": None,
            })

            print(f"Recognized segments: {len(texts)}")
            print(f"Processing time: {elapsed_seconds:.2f} seconds")
            print(f"Mean OCR confidence: {mean_confidence}")

            if not texts:
                print("WARNING: OCR returned no recognized text.")

        except Exception as exc:
            elapsed_seconds = time.perf_counter() - start_time
            error_message = str(exc)
            failures.append(image_path.name)

            summary_rows.append({
                "image_file": image_path.name,
                "status": "failed",
                "recognized_segments": 0,
                "mean_confidence": None,
                "processing_time_seconds": round(
                    elapsed_seconds, 4
                ),
                "text_file": "",
                "error": error_message,
            })

            raw_results.append({
                "image_file": image_path.name,
                "status": "failed",
                "texts": [],
                "confidence_scores": [],
                "processing_time_seconds": round(
                    elapsed_seconds, 4
                ),
                "error": error_message,
            })

            print(f"ERROR: {error_message}")

    # Save the summary CSV.
    summary_fields = [
        "image_file",
        "status",
        "recognized_segments",
        "mean_confidence",
        "processing_time_seconds",
        "text_file",
        "error",
    ]

    with SUMMARY_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=summary_fields
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save complete text and confidence data in JSON Lines format.
    with JSONL_FILE.open("w", encoding="utf-8") as file:
        for record in raw_results:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    successful = len(image_paths) - len(failures)

    print("\n" + "=" * 50)
    print("OCR SUMMARY")
    print("=" * 50)
    print(f"Images discovered: {len(image_paths)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failures)}")
    print(f"Summary CSV: {SUMMARY_FILE}")
    print(f"Detailed results: {JSONL_FILE}")
    print(f"Text files: {TEXT_DIR}")

    if failures:
        print("\nFailed images:")
        for filename in failures:
            print(f"- {filename}")
        raise SystemExit(1)

    print("\nOCR batch completed without processing failures.")


if __name__ == "__main__":
    main()
