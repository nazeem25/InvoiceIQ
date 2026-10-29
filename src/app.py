
from pathlib import Path
import sys
import time
import tempfile

from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR

# Resolve the project paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import existing InvoiceIQ modules.
from ocr_engine import extract_result_fields
from field_extraction import extract_fields
from validate_extraction import validate_record
from human_review import (
    FIELD_LABELS,
    find_line_confidence,
    field_review_reasons,
    invoice_needs_review,
    CONFIDENCE_THRESHOLD,
)

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

# Upload security limits.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "uploads"

# Initialize the OCR model lazily, then reuse it.
ocr_engine = None


def get_ocr_engine():
    global ocr_engine

    if ocr_engine is None:
        app.logger.info("Initializing PaddleOCR on CPU...")

        ocr_engine = PaddleOCR(
            lang="en",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    return ocr_engine


@app.get("/")
def home():
    """Serve the InvoiceIQ dashboard."""
    return render_template("index.html")


@app.get("/api/health")
def health():
    """Check whether the API is responding."""
    return jsonify({
        "status": "ok",
        "service": "InvoiceIQ API",
    })


@app.post("/api/extract")
def extract_invoice():
    """Process one uploaded invoice image."""
    if "file" not in request.files:
        return jsonify({
            "error": "Please upload an invoice image."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "No file selected."
        }), 400

    filename = secure_filename(uploaded_file.filename)

    if not filename:
        return jsonify({
            "error": "Invalid filename."
        }), 400

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Unsupported file type. Upload PNG, JPG, or JPEG."
        }), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_path = None

    try:
        # Use a unique temporary filename to avoid overwriting uploads.
        with tempfile.NamedTemporaryFile(
            dir=UPLOAD_DIR,
            suffix=extension,
            delete=False,
        ) as temp_file:
            image_path = Path(temp_file.name)
            uploaded_file.save(image_path)

        start_time = time.perf_counter()

        # Run OCR only on the uploaded image.
        results = get_ocr_engine().predict(input=str(image_path))

        texts = []
        scores = []

        for result_item in results:
            item_texts, item_scores = extract_result_fields(result_item)
            texts.extend(item_texts)
            scores.extend(item_scores)

        elapsed_seconds = time.perf_counter() - start_time

        if not texts:
            return jsonify({
                "error": "No text could be recognized in this image.",
                "ocr_status": "success",
                "recognized_segments": 0,
                "needs_human_review": True,
            }), 422

        # Extract structured invoice fields.
        full_text = "\n".join(texts)
        fields = extract_fields(full_text)

        # Validate required fields, dates, numeric values, and totals.
        issues = validate_record(fields)
        validation_valid = len(issues) == 0

        # Calculate field-level OCR confidence and review reasons.
        field_results = {}

        for field in FIELD_LABELS:
            value = fields.get(field)
            confidence = find_line_confidence(field, texts, scores)

            reasons = field_review_reasons(
                value,
                confidence,
                CONFIDENCE_THRESHOLD,
            )

            field_results[field] = {
                "value": value,
                "ocr_line_confidence": confidence,
                "confidence_percent": (
                    round(confidence * 100, 2)
                    if confidence is not None
                    else None
                ),
                "needs_review": bool(reasons),
                "review_reasons": reasons,
            }

        # Decide whether this invoice needs manual review.
        needs_review = invoice_needs_review(
            field_results,
            validation_valid=validation_valid,
            ocr_status="success",
        )

        mean_confidence = (
            sum(scores) / len(scores)
            if scores
            else None
        )

        return jsonify({
            "ocr_status": "success",
            "recognized_segments": len(texts),
            "mean_ocr_confidence": (
                round(mean_confidence, 4)
                if mean_confidence is not None
                else None
            ),
            "processing_time_seconds": round(
                elapsed_seconds, 2
            ),
            "extracted_fields": fields,
            "field_results": field_results,
            "validation_valid": validation_valid,
            "validation_issues": issues,
            "needs_human_review": needs_review,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "recognized_text": full_text,
        })

    except Exception:
        app.logger.exception("Invoice processing failed.")

        return jsonify({
            "error": (
                "Invoice processing failed. "
                "Check the Flask server terminal for details."
            )
        }), 500

    finally:
        # Remove the temporary upload after processing.
        if image_path is not None:
            image_path.unlink(missing_ok=True)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "File exceeds the 10 MB upload limit."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
