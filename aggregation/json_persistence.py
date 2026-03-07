# VectorDB/aggregation/json_persistence.py
from __future__ import annotations

import os
import json
import gzip
import time
import threading
from typing import Any, Dict, List, Optional

from .persistence import InMemoryAggregationStore


class JsonAggregationStore(InMemoryAggregationStore):
    """
    Persist offline aggregation results to JSON files.

    - Persist ONLY when save_offline() is called (offline job completed).
    - Load ONLY latest on startup (current requirement).
    - Keep manifest.json for future time browsing and external cleanup tooling.

    Layout:
      base_dir/plan_<plan_id>/manifest.json
      base_dir/plan_<plan_id>/latest.json
      base_dir/plan_<plan_id>/YYYY/MM/<version>.summary.json
      base_dir/plan_<plan_id>/YYYY/MM/<version>.full.json.gz   (optional)
    """

    def __init__(self, base_dir: str, write_full_gzip: bool = True, keep_full: bool = True):
        super().__init__()
        self._fs_lock = threading.RLock()
        self.base_dir = os.path.abspath(base_dir)
        self.write_full_gzip = bool(write_full_gzip)
        self.keep_full = bool(keep_full)
        os.makedirs(self.base_dir, exist_ok=True)

    def get_manifest(
            self,
            plan_id: str,
            *,
            since: Optional[float] = None,
            until: Optional[float] = None,
            limit: Optional[int] = None,
            offset: int = 0,
            descending: bool = True
    ) -> Dict[str, Any]:
        """
        Read manifest.json and optionally filter/slice versions for browsing.

        - since/until filter on versions[*].created_at
        - descending sorts by created_at
        - offset/limit for pagination
        """
        with self._fs_lock:
            path = self._manifest_path(plan_id)
            manifest = self._read_json_safe(path) or {}

        versions = manifest.get("versions") or []
        # filter
        if since is not None:
            versions = [v for v in versions if float(v.get("created_at") or 0) >= float(since)]
        if until is not None:
            versions = [v for v in versions if float(v.get("created_at") or 0) <= float(until)]

        # sort
        versions.sort(key=lambda v: float(v.get("created_at") or 0), reverse=bool(descending))

        # slice
        offset = max(0, int(offset or 0))
        if limit is not None:
            limit = max(0, int(limit))
            versions = versions[offset: offset + limit]
        else:
            versions = versions[offset:]

        out = dict(manifest)
        out["versions"] = versions
        out["returned_at"] = time.time()
        out["returned_count"] = len(versions)
        out["descending"] = bool(descending)
        out["since"] = since
        out["until"] = until
        out["offset"] = offset
        out["limit"] = limit
        return out

    # ----------------------------
    # Load APIs
    # ----------------------------

    def load_latest_only(self, plan_id: Optional[str] = None) -> int:
        """
        Load latest.json into memory for one plan or all plans.
        Return number of loaded plans.
        """
        loaded = 0
        with self._fs_lock:
            if plan_id:
                return 1 if self._load_one_latest(plan_id) else 0

            for name in os.listdir(self.base_dir):
                if not name.startswith("plan_"):
                    continue
                pid = name[len("plan_"):]
                if self._load_one_latest(pid):
                    loaded += 1
        return loaded

    # ----------------------------
    # Override: persist offline only
    # ----------------------------

    def save_offline(self, plan_id: str, result: Dict[str, Any]) -> None:
        """
        Save into memory and persist to disk.
        Called only when offline aggregation completed.
        """
        super().save_offline(plan_id, result)
        self._persist_offline(plan_id, result)

    # ----------------------------
    # Internal: persistence
    # ----------------------------

    def _plan_dir(self, plan_id: str) -> str:
        return os.path.join(self.base_dir, f"plan_{plan_id}")

    def _manifest_path(self, plan_id: str) -> str:
        return os.path.join(self._plan_dir(plan_id), "manifest.json")

    def _latest_path(self, plan_id: str) -> str:
        return os.path.join(self._plan_dir(plan_id), "latest.json")

    def _persist_offline(self, plan_id: str, result: Dict[str, Any]) -> None:
        version = str(result.get("version") or time.strftime("%Y%m%d_%H%M%S"))
        created_at = float(result.get("created_at") or time.time())

        yyyy = version[0:4] if len(version) >= 4 else time.strftime("%Y")
        mm = version[4:6] if len(version) >= 6 else time.strftime("%m")

        plan_dir = self._plan_dir(plan_id)
        out_dir = os.path.join(plan_dir, yyyy, mm)
        os.makedirs(out_dir, exist_ok=True)

        summary_file_rel = os.path.join(yyyy, mm, f"{version}.summary.json").replace("\\", "/")
        summary_path = os.path.join(plan_dir, summary_file_rel)

        full_file_rel = os.path.join(yyyy, mm, f"{version}.full.json.gz").replace("\\", "/")
        full_path = os.path.join(plan_dir, full_file_rel)

        # Write files + manifest atomically (best-effort)
        with self._fs_lock:
            os.makedirs(plan_dir, exist_ok=True)

            # 1) summary json (currently full schema; later can slim if needed)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)

            # 2) optional full gzip (same object for now; future can store extended)
            if self.keep_full and self.write_full_gzip:
                with gzip.open(full_path, "wt", encoding="utf-8") as gf:
                    json.dump(result, gf, ensure_ascii=False)

            # 3) manifest update
            manifest = self._read_json_safe(self._manifest_path(plan_id)) or {
                "plan_id": plan_id,
                "collection_name": result.get("collection_name"),
                "updated_at": 0.0,
                "versions": [],
                "latest_version": None,
            }

            versions: List[Dict[str, Any]] = manifest.get("versions") or []
            if not any(v.get("version") == version for v in versions):
                versions.append({
                    "version": version,
                    "created_at": created_at,
                    "time_range": result.get("time_range"),
                    "method": result.get("method"),
                    "params": result.get("params") or {},
                    "n_points": int(result.get("n_points") or 0),
                    "n_clusters": int(result.get("n_clusters") or 0),
                    "n_noise": int(result.get("n_noise") or 0),
                    "summary_file": summary_file_rel,
                    "full_file": (full_file_rel if (self.keep_full and self.write_full_gzip) else None),
                })

            manifest["versions"] = versions
            manifest["latest_version"] = version
            manifest["updated_at"] = time.time()
            if result.get("collection_name"):
                manifest["collection_name"] = result.get("collection_name")

            self._write_json_atomic(self._manifest_path(plan_id), manifest)

            # 4) latest.json overwrite
            # 修改：防御性覆盖 (空聚类不冲掉 latest)
            # 如果新跑出来的聚类结果是空的（比如系统刚启动且恰好无新数据），
            # 我们可以选择不更新 latest.json，保留上一次的有意义结果。
            if int(result.get("n_clusters", 0)) > 0:
                self._write_json_atomic(self._latest_path(plan_id), result)
            else:
                # 即使是空结果，也记录日志，但不冲掉 existing latest
                import logging
                logging.getLogger(__name__).info(
                    f"Aggregation yielded 0 clusters for {plan_id}, skipping latest.json overwrite.")

    def _load_one_latest(self, plan_id: str) -> bool:
        latest_path = self._latest_path(plan_id)
        if not os.path.exists(latest_path):
            return False
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                latest = json.load(f)

            # 修改 1：使用本类定义的 _fs_lock，或者确认父类是否存在 _lock
            # 这里保守起见，先获取锁属性
            lock = getattr(self, '_lock', getattr(self, '_fs_lock', None))
            if lock:
                with lock:
                    self._offline_versions[plan_id] = [latest]
            else:
                self._offline_versions[plan_id] = [latest]

            return True
        except Exception as e:
            # 修改 2：暴露出异常，方便排错
            import logging
            logging.getLogger(__name__).error(f"Failed to load latest.json for {plan_id}: {e}")
            return False

    # ----------------------------
    # JSON helpers
    # ----------------------------

    def _read_json_safe(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_json_atomic(self, path: str, obj: Dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)
