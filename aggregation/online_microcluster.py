# VectorDB/aggregation/online_microcluster.py
from __future__ import annotations

import time
import uuid
import queue
import threading
import numpy as np
from typing import Any, Dict, Optional

from .plans import AggregationPlan
from .cluster_manager import OnlineHandler
from .persistence import InMemoryAggregationStore


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # assumes normalized
    return float(np.dot(a, b))


class OnlineMicroClusterManager(OnlineHandler):
    """
    In-memory online micro-clustering for one plan.

    - Receives engine events (doc_embeddings_ready / doc_upsert_done)
    - Maintains online state: clusters + centroids + doc_to_cluster
    - Provides reconcile() to reset baseline from offline result
    """

    def __init__(self, engine: Any, plan: AggregationPlan, store: InMemoryAggregationStore):
        self.engine = engine
        self.plan = plan
        self.store = store

        self._q = queue.Queue(maxsize=2000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"OnlineMicro-{plan.plan_id}", daemon=True)
        self._thread.start()

        # initialize online state (baseline from latest offline if available)
        latest = self.store.get_latest_offline(plan.plan_id)
        if latest:
            self.reconcile(plan, latest)
        else:
            self._init_empty_state()

    def _init_empty_state(self):
        state = {
            "plan_id": self.plan.plan_id,
            "collection_name": self.plan.collection_name,
            "base_version": None,
            "updated_at": time.time(),
            "clusters": {},          # cluster_id -> {..., centroid:[...]}
            "doc_to_cluster": {},
        }
        self.store.set_online_state(self.plan.plan_id, state)

    # ---------- Protocol methods ----------
    def on_event(self, event: Dict[str, Any]) -> None:
        """
        Called by ClusterManager routing. Must be fast.
        We only enqueue.
        """
        # Route safety: only handle matching collection
        if event.get("collection_name") != self.plan.collection_name:
            return

        ev_type = event.get("type")
        if ev_type not in ("doc_embeddings_ready", "doc_upsert_done"):
            return

        try:
            self._q.put_nowait(event)
        except queue.Full:
            # Drop under pressure; offline reconcile will correct later
            return

    def reconcile(self, plan: AggregationPlan, offline_result: Dict[str, Any]) -> None:
        """
        Reset baseline from offline result. For simplicity, we overwrite current state.
        Future improvement: keep online-only clusters and merge.
        """
        clusters = offline_result.get("clusters", {}) or {}
        doc_to_cluster = offline_result.get("doc_to_cluster", {}) or {}
        base_version = offline_result.get("version")

        # Ensure centroid arrays are present and normalized lists
        # offline already outputs centroid list[float]
        state = {
            "plan_id": plan.plan_id,
            "collection_name": plan.collection_name,
            "base_version": base_version,
            "updated_at": time.time(),
            "clusters": clusters,
            "doc_to_cluster": doc_to_cluster,
        }
        self.store.set_online_state(plan.plan_id, state)

    # ---------- Worker loop ----------
    def stop(self, wait: bool = True, timeout: float = 5.0):
        self._stop.set()
        if wait:
            self._thread.join(timeout=timeout)

    def _loop(self):
        while not self._stop.is_set():
            try:
                ev = self._q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._process_event(ev)
            finally:
                self._q.task_done()

    # ---------- Core logic ----------
    def _process_event(self, ev: Dict[str, Any]):
        # We prefer embeddings_ready (contains embeddings)
        if ev.get("type") != "doc_embeddings_ready":
            return

        doc_id = ev.get("doc_id")
        emb = ev.get("embeddings")  # np.ndarray
        if doc_id is None or emb is None:
            return

        # Build doc vector by averaging chunk vectors
        try:
            if isinstance(emb, np.ndarray):
                v = emb.mean(axis=0)
            else:
                # in case it got serialized somehow
                v = np.mean(np.array(emb, dtype=np.float32), axis=0)
        except Exception:
            return

        v = _normalize(np.array(v, dtype=np.float32))

        # Decide thresholds
        op = self.plan.online_params or {}
        T_event = float(op.get("T_event", 0.85))
        T_dup = float(op.get("T_dup", 0.95))

        state = self.store.get_online_state(self.plan.plan_id)
        if not state:
            self._init_empty_state()
            state = self.store.get_online_state(self.plan.plan_id)

        clusters = state["clusters"]
        doc_to_cluster = state["doc_to_cluster"]

        # Prevent duplicate processing
        allow_update = bool((self.plan.online_params or {}).get("allow_update", False))
        if doc_id in doc_to_cluster and not allow_update:
            return

        # Find best cluster by cosine similarity to centroids
        best_id = None
        best_sim = -1.0

        # For small-scale test: linear scan
        for cid, cobj in clusters.items():
            cen_list = cobj.get("centroid")
            if not cen_list:
                continue
            cen = _normalize(np.array(cen_list, dtype=np.float32))
            sim = _cosine_sim(v, cen)
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        # Decide assign vs new
        if best_id is not None and best_sim >= T_event:
            # assign to best_id
            self._assign_to_cluster(state, doc_id, v, best_id, sim=best_sim, is_dup=(best_sim >= T_dup))
        else:
            # create new cluster
            new_id = f"online_{uuid.uuid4().hex[:12]}"
            self._create_cluster(state, doc_id, v, new_id)

        state["updated_at"] = time.time()
        self.store.set_online_state(self.plan.plan_id, state)

    def _create_cluster(self, state: Dict[str, Any], doc_id: str, v: np.ndarray, cluster_id: str):
        state["clusters"][cluster_id] = {
            "size": 1,
            "members": [doc_id],
            "repr_doc_id": doc_id,
            "repr_preview": "",
            "centroid": v.astype(float).tolist(),
            "last_seen": time.time(),
        }
        state["doc_to_cluster"][doc_id] = cluster_id

    def _assign_to_cluster(self, state: Dict[str, Any], doc_id: str, v: np.ndarray, cluster_id: str, sim: float, is_dup: bool):
        c = state["clusters"].get(cluster_id)
        if not c:
            self._create_cluster(state, doc_id, v, cluster_id)
            return

        # update centroid with incremental mean
        n = int(c.get("size", 1))
        cen = np.array(c.get("centroid"), dtype=np.float32)
        cen = _normalize(cen)

        new_cen = _normalize((cen * n + v) / (n + 1))
        c["centroid"] = new_cen.astype(float).tolist()
        c["size"] = n + 1
        c["members"].append(doc_id)
        c["last_seen"] = time.time()

        # choose representative if better similarity (optional)
        if sim > 0.99:  # arbitrary: if almost identical, keep old repr
            pass
        state["doc_to_cluster"][doc_id] = cluster_id

    # ---------- Read APIs (for web/UI) ----------
    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Return current online state snapshot (JSON-serializable).
        """
        st = self.store.get_online_state(self.plan.plan_id)
        return st or {}
