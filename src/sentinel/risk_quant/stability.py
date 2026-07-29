"""Population Stability Index (PSI) — a credit-risk drift staple."""
from __future__ import annotations

import numpy as np


def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between a reference and a live sample.

    PSI < 0.10 : no significant shift
    0.10-0.25  : moderate shift, monitor
    > 0.25     : material shift, investigate/revalidate
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e_perc = np.histogram(expected, bins=quantiles)[0] / max(len(expected), 1)
    a_perc = np.histogram(actual, bins=quantiles)[0] / max(len(actual), 1)
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))
