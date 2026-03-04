# VectorDB/aggregation/persistence.py
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Data formats (schemas)
# ----------------------------
"""
OfflineResult schema (in-memory / persistable dict):

{
  "plan_id": str,
  "collection_name": str,
  "version": str,                 # e.g. "20260304_170501"
  "created_at": float,            # epoch seconds
  "time_range": [start_ts, end_ts] or null,
  "method": "hdbscan" | "dbscan" | "agglomerative_threshold",
  "params": {...},

  "n_points": int,
  "n_clusters": int,
  "n_noise": int,

  "clusters": {
      "cluster_0": {
          "size": int,
          "members": [doc_id, ...],
          "repr_doc_id": str,
          "repr_preview": str,
          "centroid": [float, ...],     # normalized vector
          "last_seen": float            # max timestamp among members if available
      },
      ...
  },
  "noise": {
      "size": int,
      "members": [doc_id, ...]
  },

  "doc_to_cluster": { doc_id: "cluster_0" or "noise" }
}

OnlineState schema:

{
  "plan_id": str,
  "collection_name": str,
  "base_version": str | None,       # offline version used as baseline
  "updated_at": float,

  "clusters": { ... same cluster objects ... },
  "doc_to_cluster": {...},
}

Notes:
- "doc_id" here means logical document id for aggregation, NOT necessarily Chroma record id.
  We derive doc_id as metadata["original_doc_id"] if present, else use record id.
"""


@dataclass
class InMemoryAggregationStore:
    """
    Stores offline versions and online incremental state in memory.

    Persistence hooks:
      - dump() returns a JSON-serializable dict
      - load(snapshot) restores from dict

    You can later implement a SQLiteStore/RedisStore with same methods.
    """
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _offline_versions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # plan_id -> [offline_result...]
    _online_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)           # plan_id -> online_state

    # ---------- Offline ----------
    def save_offline(self, plan_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._offline_versions.setdefault(plan_id, []).append(result)

    def get_latest_offline(self, plan_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            arr = self._offline_versions.get(plan_id) or []
            return arr[-1] if arr else None

    def list_offline_versions(self, plan_id: str) -> List[str]:
        with self._lock:
            arr = self._offline_versions.get(plan_id) or []
            return [r.get("version") for r in arr if r.get("version")]

    # ---------- Online ----------
    def set_online_state(self, plan_id: str, state: Dict[str, Any]) -> None:
        with self._lock:
            self._online_state[plan_id] = state

    def get_online_state(self, plan_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._online_state.get(plan_id)

    # ---------- Persistence hooks ----------
    def dump(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot.
        (Note: vectors are already list[float] in our result/state schema.)
        """
        with self._lock:
            return {
                "created_at": time.time(),
                "offline_versions": self._offline_versions,
                "online_state": self._online_state,
            }

    def load(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore from snapshot dict.
        """
        with self._lock:
            self._offline_versions = snapshot.get("offline_versions", {}) or {}
            self._online_state = snapshot.get("online_state", {}) or {}
