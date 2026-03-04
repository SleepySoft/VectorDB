# VectorDB/aggregation/cluster_manager.py
from __future__ import annotations

import time
import uuid
import logging
import threading
from dataclasses import dataclass, replace, field
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

from .registry import AggregationRegistry
from .plans import AggregationPlan

logger = logging.getLogger(__name__)


# ----------------------------
# Interfaces (pluggable parts)
# ----------------------------

class OfflineRunner(Protocol):
    """
    Offline runner executes one full offline aggregation for a plan.

    Expected to:
    - fetch data from engine/repo
    - cluster using plan.method/plan.params
    - persist results if plan.persist
    - return a version identifier and summary
    """
    def run(self, plan: AggregationPlan, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...


class OnlineHandler(Protocol):
    """
    Online handler consumes real-time events for a plan.
    It MUST be non-blocking at entry point (ideally enqueue internally).
    """
    def on_event(self, event: Dict[str, Any]) -> None:
        ...

    def reconcile(self, plan: AggregationPlan, offline_result: Dict[str, Any]) -> None:
        ...


# ----------------------------
# Job tracking
# ----------------------------

@dataclass
class PlanRuntimeState:
    plan_id: str
    last_run_at: float = 0.0
    last_status: str = "never"     # never | running | ok | failed
    last_error: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None
    last_version: Optional[str] = None
    running_lock: threading.Lock = field(default_factory=threading.Lock)


# ----------------------------
# ClusterManager
# ----------------------------

class ClusterManager:
    """
    ClusterManager orchestrates offline clustering + online micro-clustering per plan.

    - It manages plan lifecycle (via registry)
    - It routes engine upsert events to correct online handler by plan.collection_name
    - It runs offline jobs with per-plan locks to avoid concurrent runs
    - It triggers reconcile on online handler after offline completes

    NOTE:
    - This manager does NOT assume what a collection stores.
      It only binds by plan.collection_name.
    """

    def __init__(
        self,
        engine: Any,  # VectorStorageEngine (typed Any to avoid import loops)
        registry: AggregationRegistry,
        offline_runner_factory: Callable[[Any], OfflineRunner],
        online_handler_factory: Optional[Callable[[Any, AggregationPlan], OnlineHandler]] = None,
        max_workers: int = 2,
    ):
        self._engine = engine
        self._registry = registry

        self._offline_runner = offline_runner_factory(engine)
        self._online_handler_factory = online_handler_factory

        self._lock = threading.RLock()
        self._states: Dict[str, PlanRuntimeState] = {}  # plan_id -> state
        self._online_handlers: Dict[str, OnlineHandler] = {}  # plan_id -> online handler

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ClusterMgr")
        self._jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> job metadata

        # Subscribe to engine event bus if available
        if hasattr(self._engine, "register_upsert_listener"):
            self._engine.register_upsert_listener(self._handle_engine_event)
            logger.info("ClusterManager subscribed to engine upsert events.")
        else:
            logger.warning("Engine has no event bus; online clustering will not receive real-time events.")

    # ----------------------------
    # Plan lifecycle
    # ----------------------------

    def register_plan(self, plan: AggregationPlan, overwrite: bool = False) -> AggregationPlan:
        """
        Register plan and (optionally) create online handler for it.
        """
        p = self._registry.add_plan(plan, overwrite=overwrite)
        with self._lock:
            self._states.setdefault(p.plan_id, PlanRuntimeState(plan_id=p.plan_id))

            if p.enable_online and self._online_handler_factory:
                if p.plan_id not in self._online_handlers:
                    self._online_handlers[p.plan_id] = self._online_handler_factory(self._engine, p)

        return p

    def unregister_plan(self, plan_id: str) -> bool:
        """
        Remove plan and stop routing online events to it.
        """
        removed = self._registry.remove_plan(plan_id)
        with self._lock:
            self._online_handlers.pop(plan_id, None)
            self._states.pop(plan_id, None)
        return removed

    def list_plans(self):
        return self._registry.list_plans()

    def get_plan(self, plan_id: str) -> Optional[AggregationPlan]:
        return self._registry.get_plan(plan_id)

    # ----------------------------
    # Offline run APIs
    # ----------------------------

    def run_offline(self, plan_id: str, async_run: bool = True, overrides: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        plan = self._registry.get_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        # Apply overrides by cloning plan (plan is frozen dataclass)
        plan2 = plan
        if overrides:
            safe = {}
            # Only allow a safe subset to avoid abuse
            for k in ("time_window_sec", "filter_criteria", "limit", "max_points", "method", "params", "semantic_only",
                      "time_field"):
                if k in overrides:
                    safe[k] = overrides[k]
            plan2 = replace(plan, **safe)

        state = self._get_or_create_state(plan_id)
        if not state.running_lock.acquire(blocking=False):
            raise RuntimeError(f"Plan {plan_id} is already running")

        job_id = str(uuid.uuid4())
        created_at = time.time()

        job = {
            "job_id": job_id,
            "plan_id": plan_id,
            "collection_name": plan.collection_name,
            "status": "pending",
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "overrides": overrides or {},
        }

        with self._lock:
            self._jobs[job_id] = job
            state.last_status = "running"
            state.last_error = None

        if async_run:
            fut = self._executor.submit(self._run_offline_job, job_id, plan2, state)
            job["future"] = fut
            return {"status": "accepted", "job_id": job_id}
        else:
            try:
                result = self._run_offline_job(job_id, plan2, state)
                return {"status": "completed", "job_id": job_id, "result": result}
            finally:
                pass

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            # Hide Future object if present (not JSON serializable)
            out = dict(job)
            out.pop("future", None)
            return out

    def cleanup_jobs(self, ttl_sec: int = 3600) -> int:
        """
        Remove old finished jobs from memory.
        Returns number of removed jobs.
        """
        now = time.time()
        removed = 0
        with self._lock:
            to_del = []
            for jid, j in self._jobs.items():
                finished_at = j.get("finished_at") or 0
                if finished_at and (now - finished_at > ttl_sec):
                    to_del.append(jid)
            for jid in to_del:
                del self._jobs[jid]
                removed += 1
        return removed

    def get_plan_state(self, plan_id: str) -> Dict[str, Any]:
        state = self._get_or_create_state(plan_id)
        return {
            "plan_id": plan_id,
            "last_run_at": state.last_run_at,
            "last_status": state.last_status,
            "last_error": state.last_error,
            "last_version": state.last_version,
        }

    # ----------------------------
    # Engine event routing (online)
    # ----------------------------

    def _handle_engine_event(self, event: Dict[str, Any]):
        """
        Engine emits events like:
          - type="doc_embeddings_ready" (from Repo hook)
          - type="doc_upsert_done" (optional)

        We route only those events that match a plan's collection_name.
        """
        ev_type = event.get("type")
        if ev_type not in ("doc_embeddings_ready", "doc_upsert_done"):
            return

        collection_name = event.get("collection_name")
        if not collection_name:
            return

        # Route by collection_name (not by semantics)
        plans = self._registry.find_by_collection(collection_name)
        if not plans:
            return

        # Deliver to online handler of each plan (usually 1)
        for p in plans:
            if not p.enable_online:
                continue
            handler = self._online_handlers.get(p.plan_id)
            if handler:
                try:
                    handler.on_event(event)  # must be non-blocking inside handler
                except Exception as e:
                    logger.warning(f"Online handler error (plan={p.plan_id}): {e}")

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _get_or_create_state(self, plan_id: str) -> PlanRuntimeState:
        with self._lock:
            if plan_id not in self._states:
                self._states[plan_id] = PlanRuntimeState(plan_id=plan_id)
            return self._states[plan_id]

    def _run_offline_job(self, job_id: str, plan: AggregationPlan, state: PlanRuntimeState) -> Dict[str, Any]:
        """
        The actual offline job execution with state updates + reconcile.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = "processing"
                job["started_at"] = time.time()

        try:
            # Execute offline runner
            overrides = job.get("overrides") or {}
            result = self._offline_runner.run(plan, overrides=overrides)  # must return dict

            version = result.get("version")
            with self._lock:
                if job:
                    job["status"] = "completed"
                    job["finished_at"] = time.time()
                    job["result"] = result

                state.last_run_at = time.time()
                state.last_status = "ok"
                state.last_error = None
                state.last_result = result
                state.last_version = version

            # Reconcile online handler if present
            handler = self._online_handlers.get(plan.plan_id)
            if handler:
                try:
                    handler.reconcile(plan, result)
                except Exception as e:
                    logger.warning(f"Reconcile failed (plan={plan.plan_id}): {e}")

            return result

        except Exception as e:
            err = str(e)
            logger.error(f"Offline job failed (plan={plan.plan_id}): {err}", exc_info=True)
            with self._lock:
                if job:
                    job["status"] = "failed"
                    job["finished_at"] = time.time()
                    job["error"] = err
                state.last_run_at = time.time()
                state.last_status = "failed"
                state.last_error = err
            raise
        finally:
            # Always release per-plan lock
            try:
                state.running_lock.release()
            except RuntimeError:
                pass


# Just for quick test.

# class OfflineRunnerStub:
#     def __init__(self, engine):
#         self.engine = engine
#
#     def run(self, plan: AggregationPlan) -> Dict[str, Any]:
#         # TODO: 以后替换为真实实现：
#         # 1) repo.prepare_article_matrix(...)
#         # 2) run hdbscan/dbscan/agglomerative_threshold
#         # 3) persist mapping/meta/centroids
#         version = time.strftime("%Y%m%d_%H%M%S")
#         return {
#             "plan_id": plan.plan_id,
#             "collection_name": plan.collection_name,
#             "version": version,
#             "summary": {
#                 "method": plan.method,
#                 "params": plan.params,
#                 "note": "stub result, no actual clustering executed"
#             }
#         }
#
# def offline_runner_factory(engine):
#     return OfflineRunnerStub(engine)
