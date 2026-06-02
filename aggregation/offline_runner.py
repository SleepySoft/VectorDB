# VectorDB/aggregation/offline_runner.py
from __future__ import annotations

import time
import uuid
import numpy as np
from typing import Any, Dict, Optional, Tuple, List

from .plans import AggregationPlan
from .cluster_manager import OfflineRunner
from .persistence import InMemoryAggregationStore


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def _safe_preview(text: Any, n: int = 150) -> str:
    if not text:
        return ""
    s = str(text)
    return s[:n]


def _safe_get(results: dict, key: str, default):
    v = results.get(key, None)
    return default if v is None else v


class OfflineClusterRunner(OfflineRunner):
    """
    Real offline clustering runner.
    Stores results in InMemoryAggregationStore (for now).
    """

    def __init__(self, engine: Any, store: InMemoryAggregationStore):
        self.engine = engine
        self.store = store

    def run(self, plan: AggregationPlan, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        overrides supports:
          - time_range: (start, end)  # for testing
          - filter_criteria: dict
          - limit: int
          - max_points: int
          - method / params etc (you can keep it minimal for now)
        """
        overrides = overrides or {}
        collection_name = plan.collection_name

        repo = self.engine.ensure_repository(collection_name)

        # Determine time_range
        time_range = overrides.get("time_range")
        if time_range is None:
            # default: now-window
            end = time.time()
            start = end - plan.time_window_sec
            time_range = (start, end)

        filter_criteria = overrides.get("filter_criteria", plan.filter_criteria or {})
        limit = int(overrides.get("limit", plan.limit))
        max_points = int(overrides.get("max_points", plan.max_points))
        time_field = overrides.get("time_field") or plan.time_field or "timestamp"

        # Fetch chunk-level records
        results = repo.fetch_for_analysis(
            filter_criteria=filter_criteria,
            time_range=time_range,
            limit=limit,
            time_field=time_field,
        )

        ids = _safe_get(results, "ids", [])
        embs = _safe_get(results, "embeddings", [])
        metas = _safe_get(results, "metadatas", [])
        docs = _safe_get(results, "documents", [])

        # ---- SAFE emptiness checks (ids/embs may be numpy arrays) ----
        ids_empty = (ids is None) or (len(ids) == 0)
        embs_empty = (embs is None) or (len(embs) == 0)

        if ids_empty or embs_empty:
            out = self._empty_result(plan, time_range, overrides)
            self.store.save_offline(plan.plan_id, out)
            return out

        # Group to logical doc_id
        groups: Dict[str, Dict[str, Any]] = {}
        for i in range(len(ids)):
            meta = metas[i] or {}
            doc_id = meta.get("original_doc_id") or ids[i]
            g = groups.setdefault(doc_id, {"embs": [], "preview": "", "last_seen": None})
            g["embs"].append(np.array(embs[i], dtype=np.float32))
            # preview: keep first non-empty
            if not g["preview"]:
                g["preview"] = _safe_preview(docs[i])
            # last_seen: if time_field present
            # ts = meta.get(plan.time_field)
            ts = meta.get(time_field)
            try:
                ts = float(ts) if ts is not None else None
            except Exception:
                ts = None
            if ts is not None:
                g["last_seen"] = ts if (g["last_seen"] is None or ts > g["last_seen"]) else g["last_seen"]

        doc_ids = list(groups.keys())
        X = []
        previews = []
        last_seen_list = []
        for d in doc_ids:
            arr = np.stack(groups[d]["embs"], axis=0)

            # 方案 A：时间衰减/位置衰减加权（假设第一个 chunk 是标题/摘要，最重要的）
            # 给 chunk 分配递减的权重，例如：1.0, 0.9, 0.8...
            weights = np.linspace(1.0, 0.5, num=arr.shape[0])
            v = np.average(arr, axis=0, weights=weights)

            # 方案 B：Max Pooling（提取最显著的局部语义特征，防止被稀释）
            # v = arr.max(axis=0)

            X.append(v)
            previews.append(groups[d]["preview"])
            last_seen_list.append(groups[d]["last_seen"])

        X = np.stack(X, axis=0).astype(np.float32)
        # normalize for cosine based methods
        Xn = _normalize_rows(X)

        # Hard cap points to avoid memory blow (simple truncation for now)
        if Xn.shape[0] > max_points:
            Xn = Xn[:max_points]
            doc_ids = doc_ids[:max_points]
            previews = previews[:max_points]
            last_seen_list = last_seen_list[:max_points]

        method = overrides.get("method", plan.method)
        params = overrides.get("params", plan.params or {})

        labels = self._cluster(Xn, method=method, params=params)
        coords = self._reduce_for_visualization(Xn, params=params)

        # Build result dict
        version = time.strftime("%Y%m%d_%H%M%S")
        out = self._build_result(
            plan=plan,
            time_range=time_range,
            method=method,
            params=params,
            doc_ids=doc_ids,
            previews=previews,
            last_seen_list=last_seen_list,
            Xn=Xn,
            labels=labels,
            coords=coords,
            version=version,
            overrides=overrides,
        )

        self.store.save_offline(plan.plan_id, out)
        return out

    def _cluster(self, X: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
        """
        Executes the clustering algorithm on the provided normalized embedding matrix.

        Args:
            X (np.ndarray): The feature matrix, typically L2-normalized semantic embeddings
                            of shape (n_samples, n_features).
            method (str): The clustering algorithm identifier ("hdbscan", "dbscan",
                          or "agglomerative_threshold").
            params (Dict[str, Any]): Hyperparameters specific to the chosen algorithm.

        Returns:
            np.ndarray: An array of cluster labels. Noise points are labeled as -1.

        Supported Algorithms & Parameters:

        1. HDBSCAN ("hdbscan"):
           Hierarchical Density-Based Spatial Clustering of Applications with Noise.
           Best for discovering clusters of varying densities.
           - min_cluster_size (int, default=3): The minimum size of clusters. Smaller values
             yield more, finer clusters.
           - min_samples (int, default=None): The number of samples in a neighborhood for a point
             to be considered a core point. Lower values make the algorithm less conservative
             (less noise). Defaults to `min_cluster_size` if not provided.
           - metric (str, default="euclidean"): The distance metric. Since inputs X are
             L2-normalized, "euclidean" behaves similarly to "cosine" but is optimized in HDBSCAN.
           - cluster_selection_method (str, default="eom"): Determines how flat clusters are
             extracted from the hierarchy.
             "eom" (Excess of Mass) favors large, stable macro-clusters.
             "leaf" extracts the smallest, most homogeneous micro-clusters at the bottom of the tree.
           - cluster_selection_epsilon (float, default=0.0): Ensures clusters below a certain
             distance threshold are not split.

        2. DBSCAN ("dbscan"):
           Density-Based Spatial Clustering of Applications with Noise.
           Good for uniform density clusters; requires careful tuning of `eps`.
           - eps (float, default=0.25): The maximum distance between two samples for one to be
             considered as in the neighborhood of the other.
           - min_samples (int, default=3): The number of samples in a neighborhood for a point
             to be considered a core point.
           - metric (str, default="cosine"): Distance metric. Cosine is standard for text embeddings.

        3. Agglomerative Threshold ("agglomerative_threshold"):
           Bottom-up hierarchical clustering with a strict distance cutoff.
           Best for strict, predictable grouping without relying on density.
           - distance_threshold (float, default=0.25): The linkage distance threshold above which
             clusters will not be merged. Smaller values = more granular clusters.
           - linkage (str, default="average"): Which distance to use between sets of observation.
             "average" or "complete" works well for semantic grouping to avoid chaining.
           - metric (str, default="cosine"): Distance metric.

        4. K-Means ("kmeans"):
           Fixed-count clustering for workflows that need exactly K groups.
           - n_clusters (int, default=10): Target number of clusters.
           - batch_size (int, default=1024): MiniBatchKMeans batch size.
        """
        method = (method or "").lower().strip()

        if method == "hdbscan":
            try:
                import hdbscan
            except ImportError as e:
                raise ImportError("hdbscan is not installed. pip install hdbscan") from e

            min_cluster_size = int(params.get("min_cluster_size", 3))
            min_samples = params.get("min_samples", None)
            metric = params.get("metric", "euclidean")
            cluster_selection_epsilon = float(params.get("cluster_selection_epsilon", 0.0))
            cluster_selection_method = params.get("cluster_selection_method", "eom")

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method
            )
            return clusterer.fit_predict(X)

        if method == "dbscan":
            from sklearn.cluster import DBSCAN
            eps = float(params.get("eps", 0.25))
            min_samples = int(params.get("min_samples", 3))
            metric = params.get("metric", "cosine")
            model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            return model.fit_predict(X)

        if method in ("agglomerative_threshold", "agglomerative", "hierarchical_threshold"):
            from sklearn.cluster import AgglomerativeClustering
            distance_threshold = float(params.get("distance_threshold", 0.25))
            linkage = params.get("linkage", "average")
            metric = params.get("metric", "cosine")
            try:
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    linkage=linkage,
                    metric=metric
                )
            except TypeError:
                # Fallback for older scikit-learn versions
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    linkage=linkage,
                    affinity=metric
                )
            return model.fit_predict(X)

        if method == "kmeans":
            from sklearn.cluster import MiniBatchKMeans
            n_clusters = int(params.get("n_clusters", 10))
            n_clusters = max(1, min(n_clusters, int(X.shape[0])))
            batch_size = int(params.get("batch_size", 1024))
            model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init="auto")
            return model.fit_predict(X)

        raise ValueError(f"Unsupported clustering method: {method}")

    def _reduce_for_visualization(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Return 2D coordinates for UI visualization. Defaults to PCA because it is
        deterministic and already available with scikit-learn.
        """
        if X.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        method = str(params.get("reduce_method", "pca") or "pca").lower().strip()
        reduce_params = params.get("reduce_params") or {}

        if X.shape[0] == 1:
            return np.array([[0.0, 0.0]], dtype=np.float32)

        try:
            if method == "umap":
                import umap
                n_neighbors = int(reduce_params.get("n_neighbors", 15))
                min_dist = float(reduce_params.get("min_dist", 0.1))
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=max(2, min(n_neighbors, X.shape[0] - 1)),
                    min_dist=min_dist,
                    metric=reduce_params.get("metric", "cosine"),
                    random_state=int(reduce_params.get("random_state", 42))
                )
                return reducer.fit_transform(X).astype(np.float32)

            if method == "none":
                coords = X[:, :2]
                if coords.shape[1] == 1:
                    coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)])
                return coords.astype(np.float32)

            from sklearn.decomposition import PCA
            n_components = min(2, X.shape[0], X.shape[1])
            coords = PCA(n_components=n_components).fit_transform(X)
            if coords.shape[1] == 1:
                coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)])
            return coords.astype(np.float32)
        except Exception:
            coords = X[:, :2]
            if coords.shape[1] == 1:
                coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)])
            return coords.astype(np.float32)

    def _empty_result(self, plan: AggregationPlan, time_range, overrides) -> Dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "collection_name": plan.collection_name,
            "version": time.strftime("%Y%m%d_%H%M%S"),
            "created_at": time.time(),
            "time_range": list(time_range) if time_range else None,
            "method": overrides.get("method", plan.method),
            "params": overrides.get("params", plan.params or {}),
            "n_points": 0,
            "n_clusters": 0,
            "n_noise": 0,
            "clusters": {},
            "noise": {"size": 0, "members": []},
            "doc_to_cluster": {},
            "points": [],
        }

    def _build_result(
        self,
        plan: AggregationPlan,
        time_range,
        method: str,
        params: Dict[str, Any],
        doc_ids: List[str],
        previews: List[str],
        last_seen_list: List[Optional[float]],
        Xn: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        version: str,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        # cluster aggregation
        clusters: Dict[str, Dict[str, Any]] = {}
        doc_to_cluster: Dict[str, str] = {}
        points: List[Dict[str, Any]] = []

        # group indices by label
        label_to_idx: Dict[int, List[int]] = {}
        for i, lb in enumerate(labels.tolist()):
            label = int(lb)
            label_to_idx.setdefault(label, []).append(i)
            cluster_id = "noise" if label == -1 else f"cluster_{label}"
            points.append({
                "doc_id": doc_ids[i],
                "cluster": cluster_id,
                "label": label,
                "x": round(float(coords[i][0]), 6),
                "y": round(float(coords[i][1]), 6),
                "preview": previews[i],
                "last_seen": last_seen_list[i],
            })

        for lb, idxs in label_to_idx.items():
            if lb == -1:
                continue
            key = f"cluster_{lb}"
            members = [doc_ids[i] for i in idxs]
            # centroid
            c = Xn[idxs].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-9)

            # choose representative: first one (could be max similarity to centroid later)
            repr_i = idxs[0]
            last_seen = None
            for i in idxs:
                ts = last_seen_list[i]
                if ts is not None:
                    last_seen = ts if (last_seen is None or ts > last_seen) else last_seen

            clusters[key] = {
                "size": len(idxs),
                "members": members,
                "repr_doc_id": doc_ids[repr_i],
                "repr_preview": previews[repr_i],
                "centroid": c.astype(float).tolist(),
                "last_seen": last_seen,
            }
            for d in members:
                doc_to_cluster[d] = key

        noise_members = []
        if -1 in label_to_idx:
            noise_members = [doc_ids[i] for i in label_to_idx[-1]]
            for d in noise_members:
                doc_to_cluster[d] = "noise"

        n_clusters = len([k for k in clusters.keys()])
        n_noise = len(noise_members)

        return {
            "plan_id": plan.plan_id,
            "collection_name": plan.collection_name,
            "version": version,
            "created_at": time.time(),
            "time_range": list(time_range) if time_range else None,
            "method": method,
            "params": params,
            "n_points": len(doc_ids),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "clusters": clusters,
            "noise": {"size": n_noise, "members": noise_members},
            "doc_to_cluster": doc_to_cluster,
            "points": points,
        }
