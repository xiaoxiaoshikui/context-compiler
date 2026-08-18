"""Tests for benchmarks/learn_weights.py's pure-Python logistic regression."""

import sys
import unittest
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

import learn_weights  # noqa: E402


class LogisticRegressionTests(unittest.TestCase):
    def test_learns_separating_feature(self):
        # Feature 0 (relevance) is perfectly correlated with the label; the
        # rest are constant noise. The fitted weight on relevance should end
        # up clearly larger than every other feature's weight.
        rows = []
        for i in range(40):
            label = i % 2
            relevance = 1.0 if label == 1 else 0.0
            noise = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            rows.append(([relevance, *noise], label, f"item-{i}"))
        coefs = learn_weights.fit_logistic(rows, epochs=1000)
        relevance_weight = coefs[1]
        for other in coefs[2:]:
            self.assertGreater(relevance_weight, other)

    def test_predict_direction_matches_coefficient_sign(self):
        coefs = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # intercept + 7 features
        high = learn_weights.predict(coefs, [1.0, 0, 0, 0, 0, 0, 0])
        low = learn_weights.predict(coefs, [0.0, 0, 0, 0, 0, 0, 0])
        self.assertGreater(high, low)

    def test_coefficients_to_weights_is_non_negative_and_matches_default_scale(self):
        coefs = [0.1, -0.5, 0.2, 0.3, 0.0, 0.1, 0.4, 0.05]
        weights = learn_weights.coefficients_to_weights(coefs)
        for f in learn_weights.FEATURES:
            self.assertGreaterEqual(getattr(weights, f), 0.0)
        total = sum(getattr(weights, f) for f in learn_weights.FEATURES)
        self.assertAlmostEqual(total, learn_weights.DEFAULT_WEIGHT_SUM, places=6)

    def test_untestable_features_keep_their_default_value(self):
        from context_compiler.scoring import ScoringWeights

        coefs = [0.1, 5.0, 0.2, 0.3, -9.0, 0.1, 0.4, -9.0]  # recency, pin_bonus coefs are garbage
        weights = learn_weights.coefficients_to_weights(
            coefs, untestable_features=frozenset({"recency", "pin_bonus"})
        )
        default = ScoringWeights()
        self.assertEqual(weights.recency, default.recency)
        self.assertEqual(weights.pin_bonus, default.pin_bonus)
        total = sum(getattr(weights, f) for f in learn_weights.FEATURES)
        self.assertAlmostEqual(total, learn_weights.DEFAULT_WEIGHT_SUM, places=6)


if __name__ == "__main__":
    unittest.main()
