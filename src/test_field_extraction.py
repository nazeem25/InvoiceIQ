
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from field_extraction import extract_fields
from validate_extraction import validate_record


VALID_TEXT = """SAMPLE PURCHASE INVOICE
Vendor: Sunrise Traders
Invoice No: INV-2026-0005
Invoice Date: 25-08-2026
Item: Printer paper
Quantity: 8
Unit Price: 651.47
Subtotal: 5211.76
Tax (18%): 938.12
Grand Total: 6149.88
TRAINING DATA - NOT A REAL INVOICE
"""


class TestFieldExtraction(unittest.TestCase):

    def test_extracts_all_nine_fields(self):
        result = extract_fields(VALID_TEXT)

        self.assertEqual(result["invoice_number"], "INV-2026-0005")
        self.assertEqual(result["vendor"], "Sunrise Traders")
        self.assertEqual(result["invoice_date"], "25-08-2026")
        self.assertEqual(result["item"], "Printer paper")
        self.assertEqual(result["quantity"], "8")
        self.assertEqual(result["unit_price"], "651.47")
        self.assertEqual(result["subtotal"], "5211.76")
        self.assertEqual(result["tax"], "938.12")
        self.assertEqual(result["grand_total"], "6149.88")

    def test_missing_fields_return_none(self):
        result = extract_fields("Vendor: Sunrise Traders")

        self.assertEqual(result["vendor"], "Sunrise Traders")
        self.assertIsNone(result["invoice_number"])
        self.assertIsNone(result["grand_total"])

    def test_numeric_commas_are_normalized(self):
        text = VALID_TEXT.replace("Unit Price: 651.47", "Unit Price: 1,651.47")
        result = extract_fields(text)

        self.assertEqual(result["unit_price"], "1651.47")

    def test_valid_record_passes_validation(self):
        record = {
            "invoice_number": "INV-2026-0005",
            "vendor": "Sunrise Traders",
            "invoice_date": "25-08-2026",
            "item": "Printer paper",
            "quantity": "8",
            "unit_price": "651.47",
            "subtotal": "5211.76",
            "tax": "938.12",
            "grand_total": "6149.88",
        }

        self.assertEqual(validate_record(record), [])

    def test_invalid_date_is_flagged(self):
        record = {
            "invoice_number": "INV-2026-0005",
            "vendor": "Sunrise Traders",
            "invoice_date": "35-15-2026",
            "item": "Printer paper",
            "quantity": "8",
            "unit_price": "651.47",
            "subtotal": "5211.76",
            "tax": "938.12",
            "grand_total": "6149.88",
        }

        self.assertIn("invalid_invoice_date", validate_record(record))

    def test_arithmetic_mismatch_is_flagged(self):
        record = {
            "invoice_number": "INV-2026-0005",
            "vendor": "Sunrise Traders",
            "invoice_date": "25-08-2026",
            "item": "Printer paper",
            "quantity": "8",
            "unit_price": "651.47",
            "subtotal": "5000.00",
            "tax": "938.12",
            "grand_total": "5938.12",
        }

        self.assertIn(
            "subtotal_arithmetic_mismatch",
            validate_record(record),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
