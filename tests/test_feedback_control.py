"""Model-free numerical and invariance tests for feedback experiments."""

import unittest

import numpy as np

from experiments.feedback.controller import bound_excursion, damped_correction
from experiments.feedback.jacobian import central_difference, matrix_diagnostics
from experiments.feedback.perturbation import apply_perturbation, gaussian_window


class NumericalFeedbackTests(unittest.TestCase):
    def test_linear_jacobian_and_scale_conversion(self):
        rng = np.random.default_rng(11)
        matrix = rng.normal(size=(4, 12))
        offset = rng.normal(size=4)
        point = rng.normal(size=12)
        estimated, _, _ = central_difference(lambda x: matrix @ x + offset, point, .01)
        np.testing.assert_allclose(estimated, matrix, atol=1e-12)
        sd, scale = np.arange(1, 13), np.array([150, 500, 1500, 2500])
        normalized, _, _ = central_difference(lambda z: (matrix @ (sd * z) + offset) / scale, point, .02)
        np.testing.assert_allclose(normalized, matrix * sd[None, :] / scale[:, None], atol=1e-12)

    def test_central_difference_has_second_order_accuracy(self):
        point = np.array([.4, -.2])
        derivative = np.diag(3 * point ** 2)
        coarse, _, _ = central_difference(lambda x: x ** 3, point, .02)
        fine, _, _ = central_difference(lambda x: x ** 3, point, .01)
        self.assertAlmostEqual(np.linalg.norm(coarse - derivative) / np.linalg.norm(fine - derivative), 4, places=8)

    def test_damped_inverse_is_stable_and_reduces_linear_error(self):
        matrix = np.array([[1., 0, 1], [0, 2, 1]])
        error = np.array([.1, -.2])
        step = damped_correction(matrix, error, .0001, 1, 1)
        self.assertEqual(step.shape, (3,))
        self.assertLess(np.linalg.norm(error - matrix @ step), np.linalg.norm(error) * .001)
        singular = np.ones((2, 12))
        self.assertTrue(np.isfinite(damped_correction(singular, error, .01, 1, .1)).all())
        self.assertLessEqual(np.linalg.norm(damped_correction(matrix, error * 100, .01, 1, .1)), .1 + 1e-12)
        self.assertLessEqual(np.linalg.norm(bound_excursion(np.ones(12), .5)), .5 + 1e-12)

    def test_smooth_perturbation_keeps_all_other_conditions_and_input(self):
        code = {"ema": np.zeros((50, 12), dtype=np.float32), "pitch": np.full((50, 1), 125.),
                "loudness": np.ones((50, 1)), "periodicity": np.ones((50, 1)),
                "pitch_stats": np.array([125., 1]), "spk_emb": np.ones(64), "ft_len": np.int64(50)}
        window = gaussian_window(50, 50, .5, .06, .18)
        self.assertEqual(window.max(), 1)
        self.assertEqual(window[0], 0)
        self.assertEqual(window[-1], 0)
        delta = np.zeros(12); delta[3] = .05
        modified = apply_perturbation(code, window, np.arange(1, 13), delta)
        np.testing.assert_allclose(modified["ema"][:, 3], window * .2, atol=1e-8)
        for key in code:
            if key != "ema":
                np.testing.assert_array_equal(code[key], modified[key])
        self.assertTrue(np.all(code["ema"] == 0))
        self.assertTrue(np.all(modified["ema"][:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]] == 0))

    def test_invalid_numerics_fail_and_missing_gradients_remain_invalid(self):
        matrix, _, _ = central_difference(lambda x: np.array([x[0], np.nan]), np.zeros(12), .01)
        self.assertTrue(np.isnan(matrix[1]).all())
        self.assertFalse(matrix_diagnostics(matrix)["valid"])
        with self.assertRaises(ValueError):
            central_difference(lambda x: x, np.zeros(12), 0)
        with self.assertRaises(ValueError):
            damped_correction(np.zeros((2, 12)), np.ones(2), 0, 1, .1)


if __name__ == "__main__":
    unittest.main()
