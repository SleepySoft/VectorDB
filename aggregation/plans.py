# VectorDB/aggregation/plans.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class AggregationPlan:
    """
    A plan defines *what* to aggregate and *how* to run it.

    Notes:
    - This plan is purely configuration: it does NOT assume the collection stores "summary" or anything.
    - It only binds to collection_name and other constraints/params.
    """
    plan_id: str
    collection_name: str

    # Data scope
    time_window_sec: int = 24 * 3600         # rolling window size (e.g., 24h)
    run_every_sec: int = 3600                # schedule period (e.g., 1h)
    filter_criteria: Dict[str, Any] = field(default_factory=dict)
    limit: int = 50000                       # max rows fetched from DB
    max_points: int = 50000                  # hard cap for clustering input points (after preprocessing)

    # Offline clustering method
    method: str = "hdbscan"                  # "hdbscan" | "dbscan" | "agglomerative_threshold"
    params: Dict[str, Any] = field(default_factory=dict)

    # Feature configuration
    semantic_only: bool = True
    includes_metas: Optional[list] = None
    weights: Optional[Dict[str, float]] = None

    # Online mode
    enable_online: bool = True
    # For online microcluster decisions; your online manager can ignore if not needed.
    online_params: Dict[str, Any] = field(default_factory=dict)

    # Persistence & safety
    persist: bool = True
    # If True, registry refuses creating more than one plan for same collection by default.
    exclusive_collection: bool = True

    # Optional: override plan time field (metadata key)
    time_field: str = "timestamp"

    def validate(self) -> None:
        if not self.plan_id or not isinstance(self.plan_id, str):
            raise ValueError("plan_id must be a non-empty string")
        if not self.collection_name or not isinstance(self.collection_name, str):
            raise ValueError("collection_name must be a non-empty string")
        if self.time_window_sec <= 0:
            raise ValueError("time_window_sec must be positive")
        if self.run_every_sec <= 0:
            raise ValueError("run_every_sec must be positive")
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.max_points <= 0:
            raise ValueError("max_points must be positive")
        if self.method not in ("hdbscan", "dbscan", "agglomerative_threshold"):
            raise ValueError(f"Unsupported method: {self.method}")
