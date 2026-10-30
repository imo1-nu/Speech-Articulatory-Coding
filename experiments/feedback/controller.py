"""Damped least-squares corrections with explicit step and excursion limits."""

import numpy as np


def damped_correction(jacobian, error, regularization, gain, max_step_norm):
    jacobian, error = np.asarray(jacobian, dtype=float), np.asarray(error, dtype=float)
    if (jacobian.ndim != 2 or error.shape != (jacobian.shape[0],)
            or not np.isfinite(jacobian).all() or not np.isfinite(error).all()
            or regularization <= 0 or gain <= 0 or max_step_norm <= 0):
        raise ValueError("Expected finite compatible inputs and positive regularization/gain/limit")
    step = gain * jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + regularization * np.eye(len(error)), error)
    norm = np.linalg.norm(step)
    if norm > max_step_norm:
        step *= max_step_norm / norm
    return step


def bound_excursion(point, max_norm):
    point = np.asarray(point, dtype=float)
    if max_norm <= 0 or not np.isfinite(point).all():
        raise ValueError("Expected a finite point and positive excursion limit")
    norm = np.linalg.norm(point)
    return point * min(1, max_norm / norm) if norm else point.copy()
