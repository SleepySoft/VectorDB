from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pydantic import BaseModel


class ClusteringResult(BaseModel):
    """Standardized output for any clustering strategy."""
    n_clusters: int
    labels: List[int]
    cluster_info: List[Dict[str, Any]]
    execution_time: float
    method: str


class ClusteringStrategy(ABC):
    """Abstract Base Class for clustering algorithms."""

    @abstractmethod
    def perform_clustering(self,
                           features: np.ndarray,
                           documents: List[str],
                           metadatas: List[Dict[str, Any]],
                           **kwargs) -> ClusteringResult:
        """
        Execute clustering logic.
        :param features: Numerical matrix (n_samples, n_features).
        :param documents: Raw text content.
        :param metadatas: Metadata dictionaries.
        :param kwargs: Algorithm-specific hyperparameters.
        """
        pass


# -------------------------------------------------------------------------------------------------

from sklearn.cluster import MiniBatchKMeans
import time


class KMeansStrategy(ClusteringStrategy):
    """Legacy fixed-K clustering using MiniBatchKMeans for performance."""

    def perform_clustering(self, features, documents, metadatas, **kwargs) -> ClusteringResult:
        start_time = time.time()
        n_clusters = kwargs.get("n_clusters", 10)

        # Ensure we don't request more clusters than samples
        actual_k = min(n_clusters, len(features))
        if actual_k < 1: actual_k = 1

        model = MiniBatchKMeans(n_clusters=actual_k, n_init="auto", batch_size=1024)
        labels = model.fit_predict(features)

        return ClusteringResult(
            n_clusters=actual_k,
            labels=labels.tolist(),
            cluster_info=[],  # Summary logic can be injected here
            execution_time=time.time() - start_time,
            method="kmeans"
        )


class IntelligenceClusteringEngine:
    def __init__(self):
        self._strategies = {
            "fixed_k": KMeansStrategy(),
            "auto_fine_grained": BirchStrategy()
        }

    def _build_feature_matrix(self, embeddings: List[List[float]], metadatas: List[Dict], time_weight: float):
        """Fuses semantic vectors with normalized temporal data."""
        X_semantic = np.array(embeddings)

        if time_weight > 0:
            times = np.array([m.get('timestamp', 0) for m in metadatas]).reshape(-1, 1)
            # Simple Min-Max scaling for the time feature
            if times.max() > times.min():
                times = (times - times.min()) / (times.max() - times.min())
            X_combined = np.hstack([X_semantic, times * time_weight])
            return X_combined
        return X_semantic

    def cluster_data(self, method: str, chroma_data: Dict[str, Any], time_weight: float = 0.1,
                     **kwargs) -> ClusteringResult:
        """
        Main entry point for 'Query-then-Cluster' workflow.
        :param method: 'fixed_k' or 'auto_fine_grained'
        :param chroma_data: The result of a collection.get() or collection.query()
        """
        if method not in self._strategies:
            raise ValueError(f"Strategy {method} not registered.")

        # Extract data from Chroma format
        # Handle cases where Chroma returns nested lists (batch results)
        embeddings = chroma_data['embeddings'][0] if isinstance(chroma_data['embeddings'][0], list) else chroma_data[
            'embeddings']
        documents = chroma_data['documents'][0] if isinstance(chroma_data['documents'][0], list) else chroma_data[
            'documents']
        metadatas = chroma_data['metadatas'][0] if isinstance(chroma_data['metadatas'][0], list) else chroma_data[
            'metadatas']

        # 1. Prepare features
        features = self._build_feature_matrix(embeddings, metadatas, time_weight)

        # 2. Execute selected strategy
        strategy = self._strategies[method]
        return strategy.perform_clustering(features, documents, metadatas, **kwargs)


def main():
    # 1. Query only items from the last 48 hours
    filtered_data = repo._collection.get(
        where={"timestamp": {"$gt": time.time() - 172800}},
        include=['embeddings', 'documents', 'metadatas']
    )

    # 2. Use the engine to cluster these results into thousands of small groups
    engine = IntelligenceClusteringEngine()
    result = engine.cluster_data(
        method="auto_fine_grained",
        chroma_data=filtered_data,
        threshold=0.3,  # Fine-grained similarity threshold
        time_weight=0.15  # Sensitivity to time differences
    )

    print(f"Algorithm: {result.method}")
    print(f"Found {result.n_clusters} clusters in {result.execution_time:.2f}s")
