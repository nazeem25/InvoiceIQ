
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from doclayout_yolo import YOLOv10


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "data" / "invoices"
OUTPUT_DIR = ROOT / "outputs" / "cnn_layout"

MODEL_PATH = (
    ROOT / "models" /
    "doclayout_yolo_docstructbench_imgsz1024.pt"
)

MODEL_ID = "juliozhao/DocLayout-YOLO-DocStructBench"


# --------------------------------------------------
# IMAGE DISCOVERY
# --------------------------------------------------

def get_images(input_path):

    extensions = {".png", ".jpg", ".jpeg"}

    if input_path.is_file():

        if input_path.suffix.lower() not in extensions:
            raise ValueError("Unsupported image format.")

        return [input_path]

    return sorted(
        file for file in input_path.iterdir()
        if file.suffix.lower() in extensions
    )


# --------------------------------------------------
# DOCUMENT LAYOUT DETECTION
# --------------------------------------------------

def detect_layout(model, image_path):

    with Image.open(image_path) as image:
        width, height = image.size

    predictions = model.predict(
        source=str(image_path),
        imgsz=1024,
        conf=0.20,
        device="cpu"
    )

    result = predictions[0]

    boxes = result.boxes
    names = result.names

    detections = []

    if boxes is not None:

        for box in boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            coordinates = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]

            class_name = (
                names.get(class_id, str(class_id))
                if isinstance(names, dict)
                else names[class_id]
            )

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": coordinates
            })

    return result, detections, width, height


# --------------------------------------------------
# SAVE ANNOTATED IMAGE
# --------------------------------------------------

def save_annotated_image(result, image_path):

    annotated = result.plot()

    # Convert model output into a NumPy array.
    annotated = np.asarray(annotated).astype("uint8")

    # Convert NumPy array into a PIL image.
    annotated_image = Image.fromarray(annotated)

    annotated_path = (
        OUTPUT_DIR / f"{image_path.stem}_layout.jpg"
    )

    annotated_image.save(annotated_path, quality=95)

    return annotated_path


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="InvoiceIQ CNN document layout detection"
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Invoice image or directory containing images"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images to process; 0 means all"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained model weights not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = get_images(input_path)

    if args.limit > 0:
        image_files = image_files[:args.limit]

    if not image_files:
        print("No supported invoice images found.")
        return

    # ----------------------------------------------
    # LOAD PRETRAINED MODEL
    # ----------------------------------------------

    print("Loading pretrained DocLayout-YOLO model...")

    model = YOLOv10(str(MODEL_PATH))

    all_results = []

    # ----------------------------------------------
    # PROCESS INVOICE IMAGES
    # ----------------------------------------------

    for index, image_path in enumerate(image_files, start=1):

        print(
            f"\n[{index}/{len(image_files)}] "
            f"Processing: {image_path.name}"
        )

        try:

            result, detections, width, height = detect_layout(
                model,
                image_path
            )

            annotated_path = save_annotated_image(
                result,
                image_path
            )

            record = {
                "image_file": image_path.name,
                "image_width": width,
                "image_height": height,
                "model": MODEL_ID,
                "status": "success",
                "detection_count": len(detections),
                "detections": detections,
                "annotated_image": annotated_path.name
            }

            all_results.append(record)

            print(f"Detected regions: {len(detections)}")
            print(f"Annotated image: {annotated_path}")

        except Exception as error:

            print(f"Failed: {error}")

            all_results.append({
                "image_file": image_path.name,
                "status": "failed",
                "error": str(error)
            })

    # ----------------------------------------------
    # SAVE JSON RESULTS
    # ----------------------------------------------

    json_path = OUTPUT_DIR / "layout_results.json"

    with json_path.open("w", encoding="utf-8") as file:

        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    successful = sum(
        1 for record in all_results
        if record["status"] == "success"
    )

    failed = len(all_results) - successful

    # ----------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------

    print("\n" + "=" * 50)
    print("INVOICEIQ CNN LAYOUT DETECTION SUMMARY")
    print("=" * 50)

    print(f"Total images: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    print(f"\nResults directory: {OUTPUT_DIR}")
    print(f"JSON report: {json_path}")

    print("=" * 50)


if __name__ == "__main__":
    main()
