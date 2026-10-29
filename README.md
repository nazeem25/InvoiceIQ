
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
