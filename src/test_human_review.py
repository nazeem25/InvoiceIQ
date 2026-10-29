
import unittest

from human_review import is_missing


class TestHumanReview(unittest.TestCase):

    def test_missing_none(self):
        self.assertTrue(is_missing(None))

    def test_missing_empty_string(self):
        self.assertTrue(is_missing(""))

    def test_missing_whitespace(self):
        self.assertTrue(is_missing("   "))

    def test_present_value(self):
        self.assertFalse(is_missing("INV-2026-0001"))

    def test_zero_is_not_missing(self):
        self.assertFalse(is_missing("0"))

    
    def test_low_confidence_is_flagged(self):
        from human_review import field_review_reasons
        self.assertIn(
            "low_ocr_confidence",
            field_review_reasons("INV-123", 0.75, 0.98),
        )

    def test_high_confidence_is_not_flagged(self):
        from human_review import field_review_reasons
        self.assertEqual(
            field_review_reasons("INV-123", 0.995, 0.98),
            [],
        )

    def test_missing_confidence_is_flagged(self):
        from human_review import field_review_reasons
        self.assertIn(
            "confidence_unavailable",
            field_review_reasons("INV-123", None, 0.98),
        )

    def test_missing_value_is_flagged(self):
        from human_review import field_review_reasons
        self.assertIn(
            "missing_field",
            field_review_reasons(None, 0.995, 0.98),
        )

    def test_confidence_exactly_at_threshold_passes(self):
        from human_review import field_review_reasons
        self.assertEqual(
            field_review_reasons("INV-123", 0.98, 0.98),
            [],
        )
    
    def test_validation_failure_requires_review(self):
        from human_review import invoice_needs_review

        fields = {
            "invoice_number": {
                "needs_review": False
            }
        }

        self.assertTrue(
            invoice_needs_review(fields, False, "success")
        )

    def test_ocr_failure_requires_review(self):
        from human_review import invoice_needs_review

        fields = {
            "invoice_number": {
                "needs_review": False
            }
        }

        self.assertTrue(
            invoice_needs_review(fields, True, "error")
        )

if __name__ == "__main__":
    unittest.main()
