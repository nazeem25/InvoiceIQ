
import unittest
from evaluate import statistics


class TestEvaluationMetrics(unittest.TestCase):

    def test_mean_latency(self):
        timings = [10.0, 12.0, 14.0]
        self.assertEqual(statistics.mean(timings), 12.0)

    def test_median_latency_odd_count(self):
        timings = [10.0, 14.0, 12.0]
        self.assertEqual(statistics.median(timings), 12.0)

    def test_median_latency_even_count(self):
        timings = [10.0, 12.0, 14.0, 16.0]
        self.assertEqual(statistics.median(timings), 13.0)

    def test_review_rate_calculation(self):
        flagged = 23
        total = 30
        rate = round(100 * flagged / total, 2)
        self.assertEqual(rate, 76.67)

    def test_field_review_rate_calculation(self):
        flagged = 37
        total = 270
        rate = round(100 * flagged / total, 2)
        self.assertEqual(rate, 13.7)

    def test_zero_failures_gives_full_success_rate(self):
        successful = 30
        total = 30
        rate = round(100 * successful / total, 2)
        self.assertEqual(rate, 100.0)


if __name__ == "__main__":
    unittest.main()
