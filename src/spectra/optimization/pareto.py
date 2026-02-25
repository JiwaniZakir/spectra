"""Pareto frontier analysis for multi-objective pipeline selection.

Identifies the set of Pareto-optimal pipeline configurations -- those
where no other configuration dominates on all objectives simultaneously.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class ParetoPoint(BaseModel):
    """A single point in multi-objective space."""

    config: dict[str, Any] = Field(default_factory=dict)
    objectives: dict[str, float] = Field(default_factory=dict)
    is_pareto_optimal: bool = False
    dominated_by: int = 0  # number of points that dominate this one
    crowding_distance: float = 0.0


class ParetoFrontier(BaseModel):
    """Result of a Pareto frontier analysis."""

    frontier_points: list[ParetoPoint] = Field(default_factory=list)
    dominated_points: list[ParetoPoint] = Field(default_factory=list)
    objective_names: list[str] = Field(default_factory=list)
    num_total: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParetoAnalyzer:
    """Multi-objective Pareto frontier analyzer.

    Given a set of pipeline evaluation results with multiple objective
    values, identifies the Pareto-optimal configurations and computes
    crowding distances for diversity-aware selection.

    Parameters
    ----------
    directions:
        Mapping from objective name to "maximize" or "minimize".
        Defaults to "maximize" for all objectives.

    Example
    -------
    >>> analyzer = ParetoAnalyzer({"quality": "maximize", "latency": "minimize"})
    >>> points = [
    ...     ParetoPoint(config={"name": "A"}, objectives={"quality": 0.9, "latency": 0.2}),
    ...     ParetoPoint(config={"name": "B"}, objectives={"quality": 0.8, "latency": 0.1}),
    ... ]
    >>> frontier = analyzer.analyze(points)
    """

    def __init__(
        self,
        directions: dict[str, str] | None = None,
    ) -> None:
        self._directions = directions or {}

    def analyze(self, points: Sequence[ParetoPoint]) -> ParetoFrontier:
        """Compute the Pareto frontier from a set of evaluated configurations.

        Parameters
        ----------
        points:
            Evaluated pipeline configurations with objective scores.

        Returns
        -------
        ParetoFrontier
            Frontier and dominated points with crowding distances.
        """
        if not points:
            return ParetoFrontier()

        points = list(points)
        obj_names = sorted(points[0].objectives.keys())

        # Build the objective matrix, flipping sign for "minimize" objectives
        matrix = np.zeros((len(points), len(obj_names)))
        for i, pt in enumerate(points):
            for j, name in enumerate(obj_names):
                val = pt.objectives.get(name, 0.0)
                direction = self._directions.get(name, "maximize")
                # We want to maximize everything internally
                matrix[i, j] = val if direction == "maximize" else -val

        # Find Pareto-optimal indices
        pareto_mask = self._find_pareto_optimal(matrix)

        # Compute domination counts
        domination_counts = self._count_dominations(matrix)

        # Compute crowding distances for frontier points
        frontier_indices = np.where(pareto_mask)[0]
        crowding = self._crowding_distance(matrix[frontier_indices])

        # Build result
        frontier_points: list[ParetoPoint] = []
        dominated_points: list[ParetoPoint] = []

        frontier_i = 0
        for i, pt in enumerate(points):
            new_pt = pt.model_copy(
                update={
                    "is_pareto_optimal": bool(pareto_mask[i]),
                    "dominated_by": int(domination_counts[i]),
                }
            )
            if pareto_mask[i]:
                new_pt = new_pt.model_copy(
                    update={"crowding_distance": float(crowding[frontier_i])}
                )
                frontier_points.append(new_pt)
                frontier_i += 1
            else:
                dominated_points.append(new_pt)

        # Sort frontier by crowding distance (higher is more diverse)
        frontier_points.sort(key=lambda p: p.crowding_distance, reverse=True)

        return ParetoFrontier(
            frontier_points=frontier_points,
            dominated_points=dominated_points,
            objective_names=obj_names,
            num_total=len(points),
            metadata={"directions": self._directions},
        )

    def suggest_best(
        self,
        frontier: ParetoFrontier,
        preference: dict[str, float] | None = None,
    ) -> ParetoPoint | None:
        """Suggest the best configuration from the Pareto frontier.

        Parameters
        ----------
        frontier:
            A computed Pareto frontier.
        preference:
            Optional weights for each objective (higher = more important).
            If None, selects the point with the highest crowding distance
            (most balanced).

        Returns
        -------
        ParetoPoint or None
            The suggested configuration.
        """
        if not frontier.frontier_points:
            return None

        if preference is None:
            # Return the most diverse (highest crowding distance) point
            return frontier.frontier_points[0]

        # Weighted score selection
        best_pt: ParetoPoint | None = None
        best_score = -float("inf")

        for pt in frontier.frontier_points:
            score = sum(
                pt.objectives.get(name, 0.0) * weight
                for name, weight in preference.items()
            )
            if score > best_score:
                best_score = score
                best_pt = pt

        return best_pt

    # ------------------------------------------------------------------
    # Pareto computation
    # ------------------------------------------------------------------

    @staticmethod
    def _find_pareto_optimal(matrix: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Find Pareto-optimal points (non-dominated).

        A point *p* is Pareto-optimal if no other point *q* has
        ``q[j] >= p[j]`` for all objectives *j* with at least one strict
        inequality.

        Parameters
        ----------
        matrix:
            Shape ``(n_points, n_objectives)``.  All objectives are to be
            maximized.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask of Pareto-optimal points.
        """
        n = matrix.shape[0]
        is_optimal = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_optimal[i]:
                continue
            for j in range(n):
                if i == j or not is_optimal[j]:
                    continue
                # Check if j dominates i
                if np.all(matrix[j] >= matrix[i]) and np.any(matrix[j] > matrix[i]):
                    is_optimal[i] = False
                    break

        return is_optimal

    @staticmethod
    def _count_dominations(matrix: NDArray[np.float64]) -> NDArray[np.int_]:
        """Count how many points dominate each point."""
        n = matrix.shape[0]
        counts = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(matrix[j] >= matrix[i]) and np.any(matrix[j] > matrix[i]):
                    counts[i] += 1

        return counts

    @staticmethod
    def _crowding_distance(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute NSGA-II crowding distances for frontier points.

        Crowding distance measures how isolated a point is along the
        frontier, with boundary points receiving infinite distance.

        Parameters
        ----------
        matrix:
            Shape ``(n_frontier_points, n_objectives)``.

        Returns
        -------
        NDArray[np.float64]
            Crowding distance for each frontier point.
        """
        n, m = matrix.shape
        if n <= 2:
            return np.full(n, float("inf"))

        distances = np.zeros(n)

        for obj in range(m):
            sorted_idx = np.argsort(matrix[:, obj])
            obj_range = matrix[sorted_idx[-1], obj] - matrix[sorted_idx[0], obj]

            # Boundary points get infinity
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")

            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        matrix[sorted_idx[i + 1], obj] - matrix[sorted_idx[i - 1], obj]
                    ) / obj_range

        return distances
