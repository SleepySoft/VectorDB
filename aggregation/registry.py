# VectorDB/aggregation/registry.py
from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .plans import AggregationPlan


class AggregationRegistry:
    """
    Registry manages plans and enforces constraints:
    - max_plans total
    - optional max plans per collection
    - unique plan_id
    - optional exclusivity per collection (exclusive_collection flag in plan)
    """

    def __init__(
        self,
        max_plans: int = 3,
        max_plans_per_collection: int = 1,
    ):
        self._max_plans = int(max_plans)
        self._max_plans_per_collection = int(max_plans_per_collection)

        self._lock = threading.RLock()
        self._plans: Dict[str, AggregationPlan] = {}   # plan_id -> plan

    def list_plans(self) -> List[AggregationPlan]:
        with self._lock:
            return list(self._plans.values())

    def get_plan(self, plan_id: str) -> Optional[AggregationPlan]:
        with self._lock:
            return self._plans.get(plan_id)

    def add_plan(self, plan: AggregationPlan, overwrite: bool = False) -> AggregationPlan:
        plan.validate()

        with self._lock:
            if plan.plan_id in self._plans and not overwrite:
                raise ValueError(f"Plan already exists: {plan.plan_id}")

            # Capacity check
            if plan.plan_id not in self._plans and len(self._plans) >= self._max_plans:
                raise ValueError(f"Too many plans (max={self._max_plans})")

            # Per-collection constraint
            if plan.exclusive_collection:
                # Enforce max_plans_per_collection for this collection
                existing_for_collection = [
                    p for p in self._plans.values()
                    if p.collection_name == plan.collection_name
                    and p.plan_id != plan.plan_id
                ]
                if len(existing_for_collection) >= self._max_plans_per_collection:
                    raise ValueError(
                        f"Collection '{plan.collection_name}' already has "
                        f"{len(existing_for_collection)} plans (max={self._max_plans_per_collection})"
                    )

            self._plans[plan.plan_id] = plan
            return plan

    def remove_plan(self, plan_id: str) -> bool:
        with self._lock:
            return self._plans.pop(plan_id, None) is not None

    def update_plan(self, plan: AggregationPlan) -> AggregationPlan:
        """
        Update is equivalent to overwrite add; it still validates constraints.
        """
        return self.add_plan(plan, overwrite=True)

    def find_by_collection(self, collection_name: str) -> List[AggregationPlan]:
        with self._lock:
            return [p for p in self._plans.values() if p.collection_name == collection_name]

    def clear(self) -> None:
        with self._lock:
            self._plans.clear()
