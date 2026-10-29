# InvoiceIQ — Dataset Integrity Notes

## Issue Identified

The dataset contains 30 PNG files but only 29 unique image contents.

The files `invoice_005.png` and `invoice_015.png` have identical SHA-256 hashes:

`58cc6ff62b43b7c4a553626a6fe8c2729b43e95c993753ed6b9ebbd302151b6b`

## Ground-Truth Conflict

The ground-truth CSV assigns `INV-2026-0005` to `invoice_005.png` and `INV-2026-0015` to `invoice_015.png`. However, the shared image contains invoice number `INV-2026-0015`.

## Handling Decision

The original image files and ground-truth CSV are preserved. No source-data correction has been made.

The duplicate must be considered when evaluating invoice-level extraction performance. The current dataset should not be described as containing 30 unique invoices until the original source data is verified.

## Temporary Evaluation Limitation

Mapping OCR results by invoice number can help diagnose field extraction, but it does not resolve the incorrect image-to-label assignment. Evaluation results must disclose this limitation.
