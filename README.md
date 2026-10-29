
# InvoiceIQ

InvoiceIQ is an invoice OCR and structured data extraction project.
It converts invoice images into structured fields, validates extracted
values, and flags uncertain results for human review.

## Dataset

- 30 synthetic invoice images
- Ground-truth labels in `data/ground_truth/invoice_labels.csv`
- All values are fictional

## Project Status

Module 1: Project setup and dataset audit.

## Requirements

- Python 3.10 or later
- Git
- VS Code
- Pandas

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Dataset Structure

```text
data/
├── invoices/
└── ground_truth/
    └── invoice_labels.csv
```

## Run Dataset Audit

From the project root:

```powershell
python src/audit_dataset.py
```

## Planned Modules

1. Project setup and dataset audit
2. Image intake and preprocessing
3. OCR text extraction
4. Structured field extraction
5. Validation and confidence scoring
6. Evaluation and error analysis
7. API and human review dashboard
8. Testing, integration and final delivery

## Limitations

The starter dataset is small and synthetic. Results on these invoices
alone do not establish production performance or generalization to
unseen invoice formats.


## Invoice Field Extraction and Validation

### Extract fields from OCR text

```powershell
python src/field_extraction.py
```

Extracts nine fields:
- Invoice number
- Vendor
- Invoice date
- Item
- Quantity
- Unit price
- Subtotal
- Tax
- Grand total

Outputs:
- `outputs/extracted/invoice_extraction.jsonl`
- `outputs/extracted/invoice_extraction.csv`

### Validate extracted records

```powershell
python src/validate_extraction.py
```

Checks required fields, date format, numeric values, and basic subtotal and total arithmetic.

Output:
- `outputs/extracted/validation_report.csv`

### Run unit tests

```powershell
python -m unittest src/test_field_extraction.py -v
```

### Current limitations

The dataset contains 30 image files but only 29 unique image contents in its original state. The duplicate-image and ground-truth assignment issue is documented in `reports/dataset_integrity_notes.md`.

The current validation results measure field completeness, formatting, and arithmetic consistency. They do not establish extraction accuracy against the original images.
