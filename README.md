# InvoiceIQ — Intelligent Invoice Data Extraction

InvoiceIQ is a Python-based invoice processing application that uses Optical Character Recognition (OCR), structured field extraction, validation rules, and confidence-based human review to convert invoice images into structured data.

The application provides a web dashboard where users can upload invoice images, inspect extracted fields, review validation results, and export the results as CSV or JSON.

## Features

- **Invoice Image Upload:** Upload PNG, JPG, and JPEG invoice images.
- **Image Preprocessing:** Improve invoice image readability using OpenCV.
- **OCR Processing:** Extract text using PaddleOCR with CPU-based inference.
- **Structured Data Extraction:** Identify invoice fields using rule-based extraction.
- **Data Validation:** Validate required fields, dates, numeric values, and invoice calculations.
- **Confidence-Based Review:** Flag fields that have low OCR confidence or validation issues.
- **Interactive Dashboard:** View extracted information, confidence scores, and review warnings.
- **CSV Export:** Download extracted fields and associated review information.
- **JSON Export:** Download the complete extraction response.
- **Evaluation Metrics:** Measure OCR success rate, processing time, and human-review workload.
- **Automated Testing:** Test extraction, validation, review logic, and evaluation calculations.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| OCR Engine | PaddleOCR |
| Image Processing | OpenCV |
| Data Processing | Pandas |
| Frontend | HTML, CSS, JavaScript |
| Data Formats | CSV, JSON, JSONL |
| Testing | pytest |
| Version Control | Git and GitHub |
| Inference | CPU-based |

## System Architecture

```text
                  InvoiceIQ
                      |
                      v
              Invoice Image Upload
                      |
                      v
              Image Preprocessing
                 (OpenCV)
                      |
                      v
                PaddleOCR
              Text Recognition
                      |
                      v
            Structured Extraction
              (Rule-Based Logic)
                      |
                      v
              Data Validation
                      |
                      v
          Confidence-Based Review
                      |
                      v
             Flask API Response
                      |
                      v
              Web Dashboard
                 /       \
                v         v
          CSV Export   JSON Export
```

## Invoice Fields

InvoiceIQ extracts the following nine fields:

1. Invoice Number
2. Vendor
3. Invoice Date
4. Line Item
5. Quantity
6. Unit Price
7. Subtotal
8. Tax
9. Grand Total

The extracted values are checked using validation rules, including required-field checks, date parsing, numeric validation, and arithmetic consistency checks.

## Project Structure

```text
InvoiceIQ/
├── data/
│   ├── invoices/
│   └── ground_truth/
│       └── invoice_labels.csv
├── outputs/
│   ├── extracted/
│   ├── ocr/
│   ├── preprocessed/
│   ├── review/
│   └── uploads/
├── reports/
├── src/
│   ├── app.py
│   ├── audit_dataset.py
│   ├── audit_alignment.py
│   ├── audit_invoice_mapping.py
│   ├── create_mapping_manifest.py
│   ├── evaluate.py
│   ├── field_extraction.py
│   ├── human_review.py
│   ├── ocr_engine.py
│   ├── preprocessing.py
│   ├── validate_extraction.py
│   ├── test_evaluate.py
│   ├── test_field_extraction.py
│   └── test_human_review.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

Generated output files and reports may be created when the processing and evaluation scripts run.

## Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Visual Studio Code or another Python-compatible editor
- Internet access for initial dependency installation

### 1. Clone the repository

```bash
git clone https://github.com/nazeem25/InvoiceIQ.git
cd InvoiceIQ
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks virtual environment activation, use an appropriate local execution-policy setting or activate the environment from a supported terminal.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

PaddleOCR and PaddlePaddle may take some time to install. Their compatibility depends on the Python version, operating system, and available package builds.

### 4. Run the application

From the project root:

```powershell
python src/app.py
```

Open the dashboard in your browser:

http://127.0.0.1:5000/

Keep the terminal open while using the application. Press `Ctrl + C` to stop the development server.

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /api/health`

Checks whether the Flask application is responding.

Example:

```bash
curl http://127.0.0.1:5000/api/health
```

A successful response indicates that the application is available.

### 2. Invoice Extraction

**Endpoint:** `POST /api/extract`

Accepts an invoice image and returns OCR output, extracted fields, validation results, confidence information, and human-review status.

Supported image formats:

- PNG
- JPG
- JPEG

Maximum upload size: 10 MB.

Example using Python:

```python
import requests

url = "http://127.0.0.1:5000/api/extract"

with open("invoice.png", "rb") as image:
    response = requests.post(
        url,
        files={"file": ("invoice.png", image, "image/png")}
    )

print("Status:", response.status_code)
print(response.json())
```

The request field name and file handling should match the implementation in `src/app.py`.

The response includes information such as:

- `extracted_fields`
- `field_results`
- `validation_valid`
- `validation_issues`
- `needs_human_review`
- `confidence_threshold`
- `recognized_text`
- `mean_ocr_confidence`
- Processing time and recognized segment count

Refer to `src/app.py` for the exact response schema and error handling.

## Human Review and Validation

InvoiceIQ uses confidence-based rules to identify records that may need manual inspection.

The current configured confidence threshold is **98%**.

A field or invoice may be flagged when:

- A required value is missing.
- OCR confidence is unavailable.
- OCR confidence falls below the configured threshold.
- OCR processing fails.
- Validation identifies an issue.

A passing validation result does not guarantee that every extracted value is correct. Human review remains important for financial data.

The 98% threshold is an initial configuration, not a threshold proven optimal through a separate calibration study.

## Testing

Run the automated test suite from the project root:

```powershell
python -m pytest -v
```

The current test suite contains 24 tests covering:

- Field extraction and missing values
- Numeric normalization
- Date and arithmetic validation
- Confidence threshold handling
- Missing confidence and required values
- OCR and validation failure review logic
- Evaluation metric calculations

**Recorded test result:** 24 passed.

Run the tests again after making code changes to confirm the current state of the application.

## Evaluation

The recorded dataset processing run produced the following results:

| Metric | Recorded Result |
|---|---:|
| Invoice images processed | 30 |
| Successful OCR runs | 30 |
| OCR failures | 0 |
| Mean processing time | Approximately 10.48 seconds per image |
| Median processing time | Approximately 10.57 seconds per image |
| Invoices flagged for review | 23 of 30 |
| Fields flagged for review | 37 of 270 |

These results describe the recorded run and may vary across machines and environments.

### Evaluation Limitation

The dataset contains an image-to-label alignment issue, including two files with identical image contents. Therefore, the recorded processing results must not be interpreted as proof of independently verified field-level accuracy.

The source invoice images and ground-truth CSV are preserved. Independent accuracy evaluation requires resolving or documenting the correspondence between the original images and their ground-truth labels without altering the source data.

## Current Limitations

- The initial dataset contains only 30 invoice images.
- Extraction relies on rule-based logic and may need adaptation for new invoice layouts.
- OCR confidence is not equivalent to the probability that an extracted field is correct.
- The human-review threshold has not been empirically optimized.
- Independent extraction accuracy has not yet been established.
- Processing time depends on hardware and image complexity.
- The application is intended as a prototype and should not be treated as a production financial-processing system without additional security, reliability, and deployment work.

## Future Enhancements

- Expand the dataset with diverse invoice layouts.
- Establish verified image-to-label correspondence.
- Calculate field-level precision, recall, and exact-match accuracy using independently verified labels.
- Improve extraction for multiple line items and varied date and currency formats.
- Add persistent storage and invoice history.
- Improve field-level correction and reviewer feedback workflows.
- Add authentication, access control, and audit logging.
- Deploy the application with production-grade security and monitoring.

## GitHub Repository

Source code and project updates:

https://github.com/nazeem25/InvoiceIQ

## License

No license has been specified yet. Add a license file before distributing the project under an open-source license.
