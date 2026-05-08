import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# --- 1. Configuration Object ---
@dataclass
class AnalysisConfig:
    """
    Snapshot of configuration for a single analysis session.
    """
    # Query Parameters
    filter_criteria: Dict[str, Any] = field(default_factory=dict)
    time_range: Optional[Tuple[float, float]] = None  # (start_timestamp, end_timestamp)
    limit: int = 20000  # Safety limit

    # Algorithm Selection
    reduce_method: str = "pca"  # Options: 'pca', 'umap', 'tsne', 'none'
    cluster_method: str = "birch"  # Options: 'birch', 'kmeans', 'dbscan'

    # Algorithm Hyperparameters
    reduce_params: Dict[str, Any] = field(default_factory=dict)
    cluster_params: Dict[str, Any] = field(default_factory=dict)

    # Feature Engineering
    time_weight: float = 0.1  # Weight of time vs semantic similarity


# --- 2. Strategy Interfaces ---
class AlgoStrategy(ABC):
    pass


class ReductionStrategy(AlgoStrategy):
    @abstractmethod
    def reduce(self, features: np.ndarray, **kwargs) -> np.ndarray: pass


class ClusterStrategy(AlgoStrategy):
    @abstractmethod
    def cluster(self, features: np.ndarray, **kwargs) -> Tuple[List[int], int]: pass


# --- 3. Algorithms Implementation ---
class AlgoFactory:
    """Factory to create algorithm instances from strings."""

    @staticmethod
    def get_reducer(name: str) -> ReductionStrategy:
        name = name.lower().strip()
        if name == "pca":
            from sklearn.decomposition import PCA
            class PCAImpl(ReductionStrategy):
                def reduce(self, X, **kw):
                    # Handle case where n_samples < n_components
                    n_comp = kw.get("n_components", 2)
                    n_comp = min(n_comp, len(X))
                    return PCA(n_components=n_comp).fit_transform(X)

            return PCAImpl()

        elif name == "umap":
            try:
                import umap
                class UMAPImpl(ReductionStrategy):
                    def reduce(self, X, **kw):
                        n_neighbors = kw.get("n_neighbors", 15)
                        min_dist = kw.get("min_dist", 0.1)
                        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine').fit_transform(X)

                return UMAPImpl()
            except ImportError:
                logger.error("UMAP not installed. Fallback to PCA.")
                return AlgoFactory.get_reducer("pca")

        # Default fallback or 'none'
        class IdentityImpl(ReductionStrategy):
            def reduce(self, X, **kw): return X[:, :2]  # Just take first 2 dims if no reduction

        return IdentityImpl()

    @staticmethod
    def get_clusterer(name: str) -> ClusterStrategy:
        name = name.lower().strip()
        if name == "birch":
            from sklearn.cluster import Birch
            class BirchImpl(ClusterStrategy):
                def cluster(self, X, **kw):
                    thresh = kw.get("threshold", 0.5)
                    br_factor = kw.get("branching_factor", 50)
                    model = Birch(threshold=thresh, branching_factor=br_factor, n_clusters=None)
                    labels = model.fit_predict(X)
                    return labels, len(set(labels))

            return BirchImpl()

        elif name == "kmeans":
            from sklearn.cluster import MiniBatchKMeans
            class KMeansImpl(ClusterStrategy):
                def cluster(self, X, **kw):
                    k = kw.get("n_clusters", 10)
                    k = min(k, len(X))
                    if k < 1: k = 1
                    model = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init="auto")
                    labels = model.fit_predict(X)
                    return labels, k

            return KMeansImpl()

        elif name == "dbscan":
            from sklearn.cluster import DBSCAN
            class DBSCANImpl(ClusterStrategy):
                def cluster(self, X, **kw):
                    eps = kw.get("eps", 0.5)
                    min_samples = kw.get("min_samples", 5)
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(X)
                    # DBSCAN labels: -1 is noise
                    unique = set(labels)
                    n_clusters = len(unique) - (1 if -1 in unique else 0)
                    return labels, n_clusters

            return DBSCANImpl()

        raise ValueError(f"Unknown clustering method: {name}")


# --- 4. The Pipeline Class ---
class IntelligenceAnalysisPipeline:
    """
    Disposable pipeline: Query -> Feature -> Reduce -> Cluster -> Result.
    """

    def __init__(self, repo_interface, config: AnalysisConfig):
        self.repo = repo_interface
        self.cfg = config
        self._timings = {}

    def close(self):
        """
        主动断开可能的大对象引用。
        """
        self.repo = None
        self.cfg = None
        self._timings = None

    def execute(self) -> Dict[str, Any]:
        t_start = time.time()

        # 1. Fetch Data
        raw_data = self._fetch_data()
        if not raw_data or not raw_data['ids'] or len(raw_data['ids']) == 0:
            return {"error": "No data found", "meta": {"total_docs": 0}}

        # 2. Prepare Features
        features = self._prepare_features(raw_data)

        # 3. Reduction (Optimization: Skip if features are too few)
        coords = self._run_reduction(features)

        # 4. Clustering
        labels, n_clusters = self._run_clustering(features)

        # 5. Assembly
        result = self._assemble_result(raw_data, coords, labels, n_clusters)

        self._timings['total'] = time.time() - t_start
        result['timings'] = self._timings
        return result

    def _fetch_data(self):
        t0 = time.time()
        # Delegate the actual DB fetch to the repo instance to keep DB logic encapsulated
        # This requires the repo to implement `fetch_for_analysis`
        data = self.repo.fetch_for_analysis(
            filter_criteria=self.cfg.filter_criteria,
            time_range=self.cfg.time_range,
            limit=self.cfg.limit
        )
        self._timings['fetch'] = time.time() - t0
        return data

    def _prepare_features(self, raw_data):
        t0 = time.time()

        embeddings = raw_data["embeddings"]
        X = np.asarray(embeddings, dtype=np.float32)

        if X.ndim == 3 and X.shape[0] == 1:
            X = X[0]

        if X.ndim != 2:
            raise ValueError(f"Invalid embeddings shape: {X.shape}")

        if self.cfg.time_weight > 0:
            metas = raw_data.get("metadatas") or []

            times = np.asarray(
                [m.get("timestamp", 0) for m in metas],
                dtype=np.float32
            ).reshape(-1, 1)

            if len(times) > 0 and times.max() > times.min():
                times = (times - times.min()) / (times.max() - times.min())

            X = np.hstack([
                X,
                times * np.float32(self.cfg.time_weight)
            ])

        self._timings["prep"] = time.time() - t0
        return X

    def _run_reduction(self, features):
        t0 = time.time()
        reducer = AlgoFactory.get_reducer(self.cfg.reduce_method)
        coords = reducer.reduce(features, **self.cfg.reduce_params)

        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)

        if coords.shape[1] == 1:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])

        self._timings['reduction'] = time.time() - t0
        return coords

    def _run_clustering(self, features):
        t0 = time.time()
        clusterer = AlgoFactory.get_clusterer(self.cfg.cluster_method)
        labels, n = clusterer.cluster(features, **self.cfg.cluster_params)
        self._timings['clustering'] = time.time() - t0
        return labels, n

    def _assemble_result(self, raw_data, coords, labels, n_clusters):
        t0 = time.time()

        if raw_data.get("unit") != "article":
            raise ValueError(f"Pipeline expects article-level data, got: {raw_data.get('unit')}")

        ids = raw_data["ids"]
        docs = raw_data["documents"]
        metas = raw_data["metadatas"]

        points = []
        # Groups for summary
        groups = {i: [] for i in range(n_clusters)}

        for i in range(len(ids)):
            c_id = int(labels[i])
            points.append({
                "id": ids[i],
                "x": round(float(coords[i][0]), 4),
                "y": round(float(coords[i][1]), 4),
                "cluster": c_id,
                "preview": docs[i][:100],
                "meta": metas[i]
            })
            if c_id != -1:
                groups[c_id].append(docs[i])

        # Simple Summary Generation
        clusters_info = []
        for cid, texts in groups.items():
            if not texts: continue
            clusters_info.append({
                "cluster_id": cid,
                "count": len(texts),
                "topic_preview": texts[0][:50]  # Simple placeholder
            })

        clusters_info.sort(key=lambda x: x['count'], reverse=True)

        self._timings['assemble'] = time.time() - t0
        return {
            "meta": {
                "total_docs": len(ids),
                "n_clusters": n_clusters,
                "config": {
                    "reduce": self.cfg.reduce_method,
                    "cluster": self.cfg.cluster_method
                }
            },
            "clusters": clusters_info,
            "points": points
        }
