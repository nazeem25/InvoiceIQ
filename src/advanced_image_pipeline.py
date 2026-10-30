"""
InvoiceIQ Advanced Image Pipeline
---------------------------------
Standalone add-on for image auditing, preprocessing experiments, swappable OCR,
basic invoice-field extraction, validation, and CSV/JSON reporting.

Run from the repository root:
    python src/advanced_image_pipeline.py --input data/invoices --output outputs/advanced_pipeline --engine paddle

Optional PDF support:
    pip install pymupdf

Optional OCR engines:
    PaddleOCR: pip install paddleocr
    EasyOCR: pip install easyocr
    Tesseract: pip install pytesseract  (also install the Tesseract application)

This module writes to a separate output directory and does not modify source images,
ground-truth labels, or existing pipeline outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

LOG = logging.getLogger("invoiceiq.advanced")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTENSION = ".pdf"

FIELD_NAMES = [
    "invoice_number", "vendor", "invoice_date", "item", "quantity",
    "unit_price", "subtotal", "tax", "grand_total",
]


@dataclass
class OCRResult:
    text: str
    mean_confidence: float | None
    segments: list[dict[str, Any]]
    elapsed_seconds: float
    engine: str


class OCRAdapter(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OCRResult: ...


class PaddleAdapter:
    name = "paddle"

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install it in your active .venv."
            ) from exc

        # PaddleOCR 3.x supports these switches to avoid expensive optional
        # document orientation/unwarping stages. Fall back for PaddleOCR 2.x.
        errors: list[str] = []
        for kwargs in (
            {
                "lang": "en",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            },
            {"lang": "en", "use_angle_cls": False},
            {"lang": "en"},
        ):
            try:
                self.ocr = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError(
                "Could not initialize PaddleOCR. Constructor errors: "
                + " | ".join(errors)
            )

    @staticmethod
    def _as_bgr(image: np.ndarray) -> np.ndarray:
        """PaddleOCR's pipeline can expect 3-channel input; normalize safely."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def recognize(self, image: np.ndarray) -> OCRResult:
        start = time.perf_counter()
        image = self._as_bgr(image)
        segments: list[dict[str, Any]] = []
        raw: Any = None

        # PaddleOCR 3.x exposes predict() and typically returns result objects
        # whose JSON payload contains rec_texts and rec_scores.
        predict = getattr(self.ocr, "predict", None)
        if callable(predict):
            try:
                raw = predict(input=image)
                if raw is not None and not isinstance(raw, (list, tuple)):
                    try:
                        raw = list(raw)
                    except TypeError:
                        raw = [raw]
            except (TypeError, AttributeError, RuntimeError) as exc:
                LOG.debug("PaddleOCR predict() fallback: %s", exc)
                raw = None

        def parse_v3(node: Any) -> None:
            if node is None:
                return
            if isinstance(node, dict):
                # Some PaddleOCR versions wrap the prediction under "res".
                texts = node.get("rec_texts")
                scores = node.get("rec_scores", [])
                if isinstance(texts, (list, tuple, np.ndarray)):
                    for idx, txt in enumerate(texts):
                        score = (
                            scores[idx]
                            if isinstance(scores, (list, tuple, np.ndarray))
                            and idx < len(scores)
                            else None
                        )
                        try:
                            confidence = float(score) if score is not None else None
                        except (TypeError, ValueError):
                            confidence = None
                        if isinstance(txt, str) and txt.strip():
                            segments.append(
                                {"text": txt.strip(), "confidence": confidence}
                            )
                    return
                for value in node.values():
                    parse_v3(value)
                return
            if isinstance(node, (list, tuple)):
                for child in node:
                    parse_v3(child)
                return

            # PaddleX result objects often expose a JSON-compatible payload.
            for attr in ("json", "to_dict"):
                value = getattr(node, attr, None)
                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        continue
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                if isinstance(value, dict):
                    parse_v3(value)
                    return

        if raw is not None:
            parse_v3(raw)

        # PaddleOCR 2.x fallback, or if a 3.x result did not expose text.
        if not segments:
            old_ocr = getattr(self.ocr, "ocr", None)
            if callable(old_ocr):
                try:
                    raw_old = old_ocr(image, cls=False)
                except TypeError:
                    raw_old = old_ocr(image)

                def parse_v2(node: Any) -> None:
                    if not isinstance(node, (list, tuple)):
                        return
                    # Typical PaddleOCR 2.x line: [box, (text, confidence)]
                    if (
                        len(node) >= 2
                        and isinstance(node[1], (list, tuple))
                        and len(node[1]) >= 2
                        and isinstance(node[1][0], str)
                    ):
                        txt, conf = node[1][0], node[1][1]
                        try:
                            confidence = float(conf)
                        except (TypeError, ValueError):
                            confidence = None
                        if txt.strip():
                            segments.append(
                                {"text": txt.strip(), "confidence": confidence}
                            )
                        return
                    for child in node:
                        parse_v2(child)

                parse_v2(raw_old)

        confidences = [
            s["confidence"]
            for s in segments
            if s.get("confidence") is not None
        ]
        return OCRResult(
            text="\n".join(s["text"] for s in segments),
            mean_confidence=statistics.mean(confidences) if confidences else None,
            segments=segments,
            elapsed_seconds=time.perf_counter() - start,
            engine=self.name,
        )


class EasyOCRAdapter:
    name = "easyocr"

    def __init__(self) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is not installed. Install it with: pip install easyocr"
            ) from exc
        self.reader = easyocr.Reader(["en"], gpu=False)

    def recognize(self, image: np.ndarray) -> OCRResult:
        start = time.perf_counter()
        if image.ndim == 2:
            prepared = image
        elif image.ndim == 3 and image.shape[2] == 4:
            prepared = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            prepared = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows = self.reader.readtext(prepared)
        segments = []
        for row in rows:
            if len(row) >= 3:
                try:
                    confidence = float(row[2])
                except (TypeError, ValueError):
                    confidence = None
                segments.append({"text": str(row[1]), "confidence": confidence})
        confidences = [
            s["confidence"] for s in segments if s["confidence"] is not None
        ]
        return OCRResult(
            text="\n".join(s["text"] for s in segments),
            mean_confidence=statistics.mean(confidences) if confidences else None,
            segments=segments,
            elapsed_seconds=time.perf_counter() - start,
            engine=self.name,
        )


class TesseractAdapter:
    name = "tesseract"

    def __init__(self) -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is not installed. Run: pip install pytesseract"
            ) from exc
        if shutil.which("tesseract") is None:
            raise RuntimeError(
                "Tesseract application was not found on PATH. Install Tesseract OCR "
                "and reopen the terminal."
            )
        self.pytesseract = pytesseract

    def recognize(self, image: np.ndarray) -> OCRResult:
        start = time.perf_counter()
        data = self.pytesseract.image_to_data(
            image,
            output_type=self.pytesseract.Output.DICT,
            config="--psm 6",
        )
        segments = []
        for txt, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            txt = str(txt).strip()
            try:
                conf = float(raw_conf) / 100.0
            except (ValueError, TypeError):
                conf = -1.0
            if txt and conf >= 0:
                segments.append({"text": txt, "confidence": conf})
        confidences = [s["confidence"] for s in segments]
        return OCRResult(
            text=" ".join(s["text"] for s in segments),
            mean_confidence=statistics.mean(confidences) if confidences else None,
            segments=segments,
            elapsed_seconds=time.perf_counter() - start,
            engine=self.name,
        )


def create_ocr_adapter(engine: str) -> OCRAdapter:
    adapters = {
        "paddle": PaddleAdapter,
        "easyocr": EasyOCRAdapter,
        "tesseract": TesseractAdapter,
    }
    try:
        return adapters[engine.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported OCR engine: {engine}") from exc


def load_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an image safely and report dimensions and quality indicators."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return image, {
        "width": width,
        "height": height,
        "channels": image.shape[2] if image.ndim == 3 else 1,
        "file_bytes": path.stat().st_size,
        "aspect_ratio": round(width / height, 4) if height else None,
        "orientation": "landscape" if width > height else (
            "portrait" if height > width else "square"
        ),
        "brightness_mean": round(float(np.mean(gray)), 2),
        "contrast_stddev": round(contrast, 2),
        "blur_laplacian_variance": round(blur, 2),
        "low_resolution_warning": bool(width < 800 or height < 600),
        "low_contrast_warning": bool(contrast < 25),
        "possible_blur_warning": bool(blur < 50),
    }


def estimate_skew_angle(gray: np.ndarray) -> float | None:
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(
            edges, 1, np.pi / 180, threshold=max(80, gray.shape[1] // 3)
        )
        if lines is None:
            return None
        angles = []
        for line in lines[:100]:
            _, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -15 <= angle <= 15:
                angles.append(float(angle))
        return float(np.median(angles)) if angles else None
    except cv2.error:
        return None


def preprocess_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    """Create preprocessing variants in memory; never alter the source image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    variants: dict[str, np.ndarray] = {
        "original": image.copy(),
        "grayscale": gray,
        "threshold_otsu": cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
        "threshold_adaptive": cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        ),
        "denoised": cv2.fastNlMeansDenoising(gray, None, 10, 7, 21),
        "contrast_clahe": cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(gray),
    }

    if max(gray.shape[:2]) < 1800:
        scale = min(2.0, 1800 / max(gray.shape[:2]))
        variants["resized_2x"] = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    else:
        variants["resized_2x"] = image.copy()

    angle = estimate_skew_angle(gray)
    if angle is not None and abs(angle) >= 0.3:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        variants["deskewed"] = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
    else:
        variants["deskewed"] = image.copy()
    return variants


def pdf_to_images(pdf_path: Path, dpi: int = 180) -> list[tuple[str, np.ndarray]]:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    pages = []
    with pymupdf.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            pages.append((f"{pdf_path.stem}_page_{idx + 1:03d}", arr))
    return pages


def discover_inputs(input_path: Path, include_pdf: bool) -> list[Path]:
    allowed = IMAGE_EXTENSIONS | ({PDF_EXTENSION} if include_pdf else set())
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in allowed else []
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed
    )


def normalize_text(value: str) -> str:
    # OCR outputs from older runs may contain literal backslash-n sequences.
    value = value.replace("\\n", "\n")
    return re.sub(r"[ \t]+", " ", value).strip()


def find_first(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return normalize_text(match.group(1)).replace("\n", " ").strip()
    return None


def extract_fields(text: str) -> dict[str, Any]:
    """Extract labelled invoice fields; values are candidates and need verification."""
    text = text.replace("\\n", "\n")
    lines = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
    joined = "\n".join(lines)
    money = r"[$€£]?\s*([0-9][0-9,]*(?:\.[0-9]{2})?)"

    fields: dict[str, Any] = {
        "invoice_number": find_first([
            r"(?:invoice\s*(?:number|no\.?|#|id))\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-\/]+)"
        ], joined),
        "vendor": find_first([
            r"^\s*vendor\s*:\s*(.+?)\s*$",
            r"^\s*supplier\s*:\s*(.+?)\s*$",
        ], joined),
        "invoice_date": find_first([
            r"(?:invoice\s*date|date)\s*[:#\-]?\s*(\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4})",
            r"(?:invoice\s*date|date)\s*[:#\-]?\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
        ], joined),
        "item": find_first([
            r"^\s*item\s*:\s*(.+?)\s*$",
            r"^\s*(?:description|product|service)\s*:\s*(.+?)\s*$",
        ], joined),
        "quantity": find_first([
            r"^\s*(?:quantity|qty)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$"
        ], joined),
        "unit_price": find_first([
            rf"^\s*unit\s*price\s*:\s*{money}\s*$",
            rf"^\s*unitprice\s*:\s*{money}\s*$",
            rf"^\s*price\s*per\s*unit\s*:\s*{money}\s*$",
        ], joined),
        "subtotal": find_first([
            rf"^\s*sub\s*total\s*:\s*{money}\s*$",
            rf"^\s*subtotal\s*:\s*{money}\s*$",
        ], joined),
        "tax": find_first([
            rf"^\s*(?:tax|vat|gst)(?:\s*\([^)]*\))?\s*:\s*{money}\s*$"
        ], joined),
        "grand_total": find_first([
            rf"^\s*(?:grand\s*total|total\s*due|amount\s*due)\s*:\s*{money}\s*$",
            rf"^\s*total\s*:\s*{money}\s*$",
        ], joined),
    }
    return fields


def validate_fields(fields: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    missing = [name for name in FIELD_NAMES if fields.get(name) in (None, "")]
    if missing:
        issues.append("Missing fields: " + ", ".join(missing))

    def amount(key: str) -> float | None:
        raw = fields.get(key)
        if raw in (None, ""):
            return None
        try:
            return float(
                str(raw).replace(",", "").replace("$", "")
                .replace("€", "").replace("£", "").strip()
            )
        except ValueError:
            issues.append(f"{key} is not a parseable numeric amount")
            return None

    subtotal, tax, total = amount("subtotal"), amount("tax"), amount("grand_total")
    if subtotal is not None and tax is not None and total is not None:
        if abs((subtotal + tax) - total) > 0.02:
            issues.append(
                f"Arithmetic mismatch: subtotal + tax = {subtotal + tax:.2f}, "
                f"grand_total = {total:.2f}"
            )
    return {"valid": not issues, "issues": issues, "missing_fields": missing}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="InvoiceIQ image audit, preprocessing, OCR comparison, and baseline extraction."
    )
    parser.add_argument(
        "--input", default="data/invoices",
        help="Image file or directory; PDFs included with --include-pdf."
    )
    parser.add_argument("--output", default="outputs/advanced_pipeline",
                        help="Separate output directory.")
    parser.add_argument("--engine", choices=["paddle", "easyocr", "tesseract"],
                        default="paddle")
    parser.add_argument("--include-pdf", action="store_true",
                        help="Rasterize PDF pages with PyMuPDF.")
    parser.add_argument("--compare-preprocessing", action="store_true",
                        help="Run OCR on each preprocessing variant; slower.")
    parser.add_argument("--pdf-dpi", type=int, default=180)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    input_path = Path(args.input).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        LOG.error("Input path does not exist: %s", input_path)
        return 2

    try:
        ocr = create_ocr_adapter(args.engine)
    except Exception as exc:
        LOG.error("%s", exc)
        return 2

    audit_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    paths = discover_inputs(input_path, args.include_pdf)
    if not paths:
        LOG.error("No supported input files found under %s", input_path)
        return 2

    total_pages = 0
    for source in paths:
        try:
            if source.suffix.lower() == PDF_EXTENSION:
                if not args.include_pdf:
                    continue
                pages = pdf_to_images(source, args.pdf_dpi)
                page_items = [
                    (name, image, source, idx + 1)
                    for idx, (name, image) in enumerate(pages)
                ]
            else:
                image, _ = load_image(source)
                page_items = [(source.stem, image, source, None)]

            for item_name, image, source_path, page_number in page_items:
                total_pages += 1
                height, width = image.shape[:2]
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                contrast = float(np.std(gray))
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                audit_rows.append({
                    "record_id": item_name,
                    "source_path": str(source_path),
                    "source_page": page_number,
                    "source_extension": source_path.suffix.lower(),
                    "width": width,
                    "height": height,
                    "file_bytes": source_path.stat().st_size,
                    "orientation": "landscape" if width > height else (
                        "portrait" if height > width else "square"
                    ),
                    "aspect_ratio": round(width / height, 4) if height else None,
                    "brightness_mean": round(float(np.mean(gray)), 2),
                    "contrast_stddev": round(contrast, 2),
                    "blur_laplacian_variance": round(blur, 2),
                    "low_resolution_warning": width < 800 or height < 600,
                    "low_contrast_warning": contrast < 25,
                    "possible_blur_warning": blur < 50,
                    "status": "ok",
                    "error": "",
                })

                variants = (
                    preprocess_variants(image)
                    if args.compare_preprocessing
                    else {"original": image}
                )
                base_result: OCRResult | None = None
                best_text = ""

                for variant_name, variant in variants.items():
                    try:
                        result = ocr.recognize(variant)
                        raw_rows.append({
                            "record_id": item_name,
                            "source_path": str(source_path),
                            "source_page": page_number,
                            "engine": args.engine,
                            "preprocessing": variant_name,
                            "elapsed_seconds": round(result.elapsed_seconds, 4),
                            "mean_ocr_confidence": (
                                round(result.mean_confidence, 4)
                                if result.mean_confidence is not None else None
                            ),
                            "recognized_text": result.text,
                        })
                        if variant_name == "original":
                            base_result = result
                            best_text = result.text

                        if args.compare_preprocessing:
                            comparison_rows.append({
                                "record_id": item_name,
                                "source_path": str(source_path),
                                "engine": args.engine,
                                "preprocessing": variant_name,
                                "elapsed_seconds": round(result.elapsed_seconds, 4),
                                "mean_ocr_confidence": (
                                    round(result.mean_confidence, 4)
                                    if result.mean_confidence is not None else None
                                ),
                                "characters_recognized": len(result.text),
                                "words_recognized": len(result.text.split()),
                                "relative_character_change_vs_original": None,
                                "error": "",
                            })
                    except Exception as exc:
                        LOG.warning(
                            "OCR failed for %s (%s): %s",
                            item_name, variant_name, exc
                        )
                        if args.compare_preprocessing:
                            comparison_rows.append({
                                "record_id": item_name,
                                "source_path": str(source_path),
                                "engine": args.engine,
                                "preprocessing": variant_name,
                                "error": str(exc),
                            })

                fields = extract_fields(best_text)
                validation = validate_fields(fields)
                extraction_rows.append({
                    "record_id": item_name,
                    "source_path": str(source_path),
                    "source_page": page_number,
                    "engine": args.engine,
                    "ocr_mean_confidence": (
                        round(base_result.mean_confidence, 4)
                        if base_result and base_result.mean_confidence is not None
                        else None
                    ),
                    "ocr_elapsed_seconds": (
                        round(base_result.elapsed_seconds, 4) if base_result else None
                    ),
                    **fields,
                    "validation_valid": validation["valid"],
                    "validation_issues": " | ".join(validation["issues"]),
                    "needs_human_review": (
                        (not validation["valid"])
                        or (base_result is None)
                        or (
                            base_result.mean_confidence is not None
                            and base_result.mean_confidence < 0.98
                        )
                    ),
                })
        except Exception as exc:
            audit_rows.append({
                "record_id": source.name,
                "source_path": str(source),
                "status": "error",
                "error": str(exc),
            })
            LOG.exception("Could not process %s", source)

    # Character counts are diagnostics only; they do not establish accuracy.
    if args.compare_preprocessing:
        original_counts = {}
        for row in comparison_rows:
            if row.get("preprocessing") == "original" and "characters_recognized" in row:
                original_counts[row["record_id"]] = row["characters_recognized"]
        for row in comparison_rows:
            base = original_counts.get(row.get("record_id"))
            count = row.get("characters_recognized")
            if base and count is not None:
                row["relative_character_change_vs_original"] = round(
                    (count - base) / base, 4
                )

    write_csv(out / "image_audit.csv", audit_rows)
    write_csv(out / "field_extraction.csv", extraction_rows)
    write_csv(out / "preprocessing_comparison.csv", comparison_rows)
    write_csv(out / "ocr_results.csv", raw_rows)
    (out / "field_extraction.json").write_text(
        json.dumps(extraction_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "input_path": str(input_path),
        "output_path": str(out),
        "engine": args.engine,
        "include_pdf": args.include_pdf,
        "compare_preprocessing": args.compare_preprocessing,
        "source_files_discovered": len(paths),
        "pages_or_images_processed": total_pages,
        "audit_errors": sum(1 for row in audit_rows if row.get("status") == "error"),
        "extraction_records": len(extraction_rows),
        "valid_records": sum(1 for row in extraction_rows if row.get("validation_valid")),
        "records_needing_human_review": sum(
            1 for row in extraction_rows if row.get("needs_human_review")
        ),
        "note": (
            "Preprocessing character counts/confidence are diagnostic only. "
            "Accuracy requires independently verified ground-truth labels."
        ),
    }
    (out / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOG.info("Processed %d images/pages. Outputs: %s", total_pages, out)
    LOG.info("Summary: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
