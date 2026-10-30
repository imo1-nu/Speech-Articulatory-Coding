"""Central differences for a fixed temporal perturbation, without model imports."""

import numpy as np


def central_difference(evaluate, point, epsilon):
    point = np.asarray(point, dtype=float)
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    plus, minus = [], []
    for j in range(len(point)):
        step = np.zeros_like(point)
        step[j] = epsilon
        plus.append(np.asarray(evaluate(point + step), dtype=float))
        minus.append(np.asarray(evaluate(point - step), dtype=float))
    plus, minus = np.stack(plus, axis=1), np.stack(minus, axis=1)
    jacobian = (plus - minus) / (2 * epsilon)
    return jacobian, plus, minus


def matrix_diagnostics(matrix):
    if not np.isfinite(matrix).all():
        return {"valid": False, "rank": None, "singular_values": None, "condition_number": None}
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    return {"valid": True, "rank": rank, "singular_values": singular.tolist(),
            "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None}


def prediction_metrics(predicted, actual):
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        return {"valid": False, "relative_error": None, "cosine": None, "error_norm": None}
    p, a = np.linalg.norm(predicted), np.linalg.norm(actual)
    error = float(np.linalg.norm(predicted - actual))
    return {"valid": True, "error_norm": error,
            "relative_error": error / a if a > 1e-12 else None,
            "cosine": float(np.dot(predicted, actual) / (p * a)) if p * a > 1e-12 else None}
