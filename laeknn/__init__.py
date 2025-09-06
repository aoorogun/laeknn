"""Top-level package for LAE‑KNN.

This module exposes the :class:`~laeknn.laeknn.LAEKNN` class implementing the
Locally Adaptive Evidential K‑Nearest Neighbours algorithm.  Import
`LAEKNN` from here for convenience.
"""

from .laeknn import LAEKNN

__all__ = ["LAEKNN"]