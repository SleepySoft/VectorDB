# VectorDB/VectorDBBService.py

import gc
import os
import time
import json
import queue
import shutil
import datetime
import logging
import threading
import numpy as np
from enum import Enum
from chromadb import Settings
from typing import List, Dict, Any, Optional, Union, Tuple

from VectorDB.memory_utils import memory_scope, cleanup_memory
from VectorDB.ClusterAnalysisPipeline import IntelligenceAnalysisPipeline, AnalysisConfig


logger = logging.getLogger(__name__)

os.environ['CHROMA_OTEL_ENABLED'] = 'False'


# Note: Heavy imports (chromadb, sentence_transformers) are delayed inside methods
# or imported at module level depending on startup preference.
# Here we keep them lazy-loaded inside the class to speed up module import.

class VectorStorageEngine:

    class Status(str, Enum):
        INIT = "initializing"
        READY = "ready"
        ERROR = "error"

    """
    VectorStorageEngine: The heavy-lifting engine.

    Responsibilities:
    1. Manages the connection to the Vector DB (ChromaDB).
    2. Loads and holds the Embedding Model in memory (SentenceTransformer).
    3. Acts as a factory for VectorCollectionRepo instances.

    This class is thread-safe. You should typically create one instance of this
    per application lifecycle, but multiple instances are allowed (e.g., for different DB paths).
    """

    def __init__(self, db_path: str, model_name: str, worker_enabled: bool = True):
        """
        Initializes the engine. This operation is blocking and heavy.

        Args:
            db_path (str): File system path for the persistent vector database.
            model_name (str): HuggingFace model name for embeddings.
        """
        self._db_path = db_path
        self._model_name = model_name

        self._status = VectorStorageEngine.Status.INIT
        self._error_message = None
        self._ready_event = threading.Event()
        self._lock = threading.RLock()

        # Resources (Initially None)
        self._client = None
        self._model = None
        # Use an LRU-like strategy or simple dict.
        # For now, just keep it, but be aware of memory if collections are infinite.
        self._repos = {}

        self._pending = {}  # {collection_name: set(doc_id)}
        self._failed = {}   # {collection_name: {doc_id: "err"}}
        self._pending_lock = threading.RLock()

        # --- Event Bus (Upsert listeners) ---
        # listeners will receive dict events, must be fast/non-blocking
        self._listeners_lock = threading.RLock()
        self._upsert_listeners = []  # List[Callable[[Dict[str, Any]], None]]

        # --- Async Task Queue Setup ---
        self._queue = queue.Queue(maxsize=100)  # Limit queue to prevent OOM on backlog
        self._worker_thread = None
        self._stop_worker = threading.Event()

        # Start initialization
        threading.Thread(target=self._load_heavy_resources, name="EngineInit", daemon=True).start()

        if worker_enabled:
            self._start_worker()

    def _load_heavy_resources(self):
        """Internal method to load libraries and models."""
        try:
            logger.info("Importing heavy libraries...")
            # Lazy imports
            import torch
            import chromadb
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading ChromaDB from {self._db_path}...")
            self._client = chromadb.PersistentClient(
                path=self._db_path,
                settings=Settings(anonymized_telemetry=False)
            )

            logger.info(f"Loading Model {self._model_name}...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model = SentenceTransformer(self._model_name, device=device)
            logger.info(f"Model loaded on device: {device}")

            # Mark as Ready
            with self._lock:
                self._status = VectorStorageEngine.Status.READY
                self._ready_event.set()

            logger.info("VectorStorageEngine is READY.")

        except Exception as e:
            logger.error(f"FATAL: Engine initialization failed: {e}")
            with self._lock:
                self._status = VectorStorageEngine.Status.ERROR
                self._error_message = str(e)
                # We do NOT set the ready event, so waiters will timeout or handle status manually

    def stop_worker(self, wait: bool = True, timeout: float = 10.0):
        """
        Gracefully stop background worker thread.
        """
        self._stop_worker.set()
        if self._worker_thread and wait:
            self._worker_thread.join(timeout=timeout)

    # --- Worker Logic ---

    def _start_worker(self):
        """Starts the background worker thread for processing heavy write tasks."""
        self._worker_thread = threading.Thread(target=self._worker_loop, name="VectorWorker", daemon=True)
        self._worker_thread.start()
        logger.info("Background worker started.")

    def _worker_loop(self):
        """Consumers tasks from the queue strictly sequentially to manage memory."""
        while not self._stop_worker.is_set():
            try:
                # Wait for a task (blocking with timeout to allow checking stop_event)
                task = self._queue.get(timeout=2)
            except queue.Empty:
                continue

            try:
                task_type = task.get("type")
                logger.info(f"Processing task: {task_type}")

                if task_type == "upsert":
                    self._handle_upsert_task(task)
                elif task_type == "batch_upsert":
                    for item in task['items']:
                        self._handle_upsert_task(item)

                # Add other async tasks here (e.g., delete, batch_import)

            except Exception as e:
                logger.error(f"Worker failed processing task: {e}")
            finally:
                self._queue.task_done()
                # Optional: Force GC after heavy tasks if memory is tight
                # gc.collect()

    # ----------------------------
    # Event Bus APIs
    # ----------------------------

    def register_upsert_listener(self, fn):
        """
        Register a listener callback: fn(event_dict) -> None
        Listener MUST be non-blocking. If heavy work is needed, enqueue internally.
        """
        if fn is None:
            return
        with self._listeners_lock:
            if fn not in self._upsert_listeners:
                self._upsert_listeners.append(fn)

    def unregister_upsert_listener(self, fn):
        """Unregister a listener callback."""
        if fn is None:
            return
        with self._listeners_lock:
            try:
                self._upsert_listeners.remove(fn)
            except ValueError:
                pass

    def _emit_event(self, event: Dict[str, Any]):
        """Internal: broadcast event to all listeners."""
        with self._listeners_lock:
            listeners = list(self._upsert_listeners)

        for fn in listeners:
            try:
                fn(event)
            except Exception as e:
                logger.warning(f"Upsert listener error: {e}")

    def _repo_embeddings_hook(self, **payload):
        """
        Internal hook passed to VectorCollectionRepo.upsert_document().
        This converts payload to a single event dict and emits it.
        """
        event = {
            "type": "doc_embeddings_ready",
            "ts": time.time(),
            **payload
        }
        # Emit event synchronously (listeners must be fast).
        # If you want absolute isolation, you can enqueue to a small internal queue here.
        self._emit_event(event)

    def _handle_upsert_task(self, task: Dict):
        """Process the upsert logic inside the worker thread."""
        collection_name = task["collection_name"]
        doc_id = task["doc_id"]
        text = task["text"]
        metadata = task["metadata"]

        # Ensure repo exists (thread-safe)
        repo = self.ensure_repository(collection_name)

        try:
            # Perform the heavy lifting (pass optional embeddings hook)
            repo.upsert_document(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
                on_embeddings=self._repo_embeddings_hook
            )

            # On success -> remove from pending
            with self._pending_lock:
                s = self._pending.get(collection_name)
                if s:
                    s.discard(doc_id)

            self._emit_event({
                "type": "doc_upsert_done",
                "ts": time.time(),
                "collection_name": collection_name,
                "doc_id": doc_id,
            })

        except Exception as e:
            # On failure -> remove pending + record failed (optional)
            with self._pending_lock:
                s = self._pending.get(collection_name)
                if s:
                    s.discard(doc_id)
                self._failed.setdefault(collection_name, {})[doc_id] = str(e)

            raise

    def submit_upsert(self, collection_name: str, doc_id: str, text: str, metadata: Dict = None) -> bool:
        """
        Public API to submit a task. Non-blocking.
        Returns True if queued, False if queue is full.
        """
        if not self.is_ready():
            raise RuntimeError("Engine not ready")

        task = {
            "type": "upsert",
            "collection_name": collection_name,
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata or {}
        }

        try:
            self._queue.put(task, block=False)

            # Only mark pending after actually queued
            with self._pending_lock:
                self._pending.setdefault(collection_name, set()).add(doc_id)
                # Optional: clear previous failed mark
                if collection_name in self._failed:
                    self._failed[collection_name].pop(doc_id, None)

            return True
        except queue.Full:
            logger.warning("Task queue is full! Dropping request.")
            return False

    def submit_upsert_batch(self, tasks: List[Dict]) -> bool:
        if not self.is_ready():
            return False

        batch_task = {
            "type": "batch_upsert",
            "items": tasks  # [{collection, doc_id, text, metadata}, ...]
        }

        try:
            self._queue.put(batch_task, block=True, timeout=5)

            with self._pending_lock:
                for item in tasks:
                    c = item["collection_name"]
                    d = item["doc_id"]
                    self._pending.setdefault(c, set()).add(d)
                    if c in self._failed:
                        self._failed[c].pop(d, None)

            return True
        except queue.Full:
            return False

    def exists_batch(self, collection_name: str, doc_ids: List[str], include_pending: bool = False) -> Dict[str, bool]:
        status = self.exists_batch_status(collection_name, doc_ids)
        if include_pending:
            return {d: (s in ("present", "pending")) for d, s in status.items()}
        else:
            return {d: (s == "present") for d, s in status.items()}

    def exists_batch_status(self, collection_name: str, doc_ids: List[str]) -> Dict[str, str]:
        """
        Returns tri-state: 'present' | 'pending' | 'missing'
        - present: persisted in DB
        - pending: queued/in-flight in memory
        - missing: neither persisted nor pending
        """
        if not self.is_ready():
            raise RuntimeError("Engine not ready")

        repo = self.ensure_repository(collection_name)

        # 1) persisted check (DB)
        persisted = repo.exists_batch(doc_ids)  # bool map via chunk_id

        # 2) pending check (in-memory)
        with self._pending_lock:
            pending_set = set(self._pending.get(collection_name, set()))

        # 3) merge into tri-state
        out = {}
        for d in doc_ids:
            if persisted.get(d, False):
                out[d] = "present"
            elif d in pending_set:
                out[d] = "pending"
            else:
                out[d] = "missing"
        return out

    def get_queue_status(self):
        running = self._worker_thread is not None and self._worker_thread.is_alive()
        return {
            "qsize": self._queue.qsize(),
            "status": running
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns the current lifecycle status."""
        return {
            "status": self._status,
            "error": self._error_message,
            "db_path": self._db_path,
            "model": self._model_name
        }

    def is_ready(self) -> bool:
        return self._status == VectorStorageEngine.Status.READY

    def wait_until_ready(self, timeout: float = None) -> bool:
        """
        Blocks until the engine is ready.
        Returns True if ready, False if timed out or errored.
        """
        if self._status == VectorStorageEngine.Status.READY:
            return True
        if self._status == VectorStorageEngine.Status.ERROR:
            return False
        ok = self._ready_event.wait(timeout=timeout)
        return ok and (self._status == VectorStorageEngine.Status.READY)

    def ensure_repository(self, collection_name: str, chunk_size: int = 512,
                          chunk_overlap: int = 50) -> "VectorCollectionRepo":
        """
        Factory method: Creates a repo if not exists, OR updates the config of an existing one.
        This is Thread-Safe.
        """
        if not self.is_ready():
            raise RuntimeError("Engine not ready")

        with self._lock:
            # 1. 如果已存在于内存缓存中
            if collection_name in self._repos:
                repo = self._repos[collection_name]
                # 更新 split 配置，以便后续写入使用新参数
                # (注意：这不会改变已存入数据库的数据，只影响新数据)
                repo.update_config(chunk_size, chunk_overlap)
                return repo

            # 2. 如果内存没有，但 Chroma 物理文件可能存在，或者完全新建
            # VectorCollectionRepo 的初始化逻辑会处理 get_or_create
            repo = VectorCollectionRepo(
                client=self._client,
                model=self._model,
                collection_name=collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                db_path=self._db_path
            )
            self._repos[collection_name] = repo
            return repo

    def get_repository(self, collection_name: str) -> Optional["VectorCollectionRepo"]:
        """
        Strictly retrieves an existing repository handle.
        Returns None if not found in cache (and ideally checks DB presence).
        """
        if not self.is_ready():
            raise RuntimeError("Engine not ready")

        with self._lock:
            if collection_name in self._repos:
                return self._repos[collection_name]

            # 检查 Chroma 中是否真的存在该 Collection
            # 只有存在时，才以默认(或推测)配置加载它
            try:
                self._client.get_collection(collection_name)
                # 存在，但内存没加载。我们必须加载它。
                # 缺点：我们不知道上次用的 chunk_size 是多少，只能用默认值。
                # 生产环境通常会将每个 Collection 的配置存入 SQLite 或 metadata 中，这里简化处理。
                return self.ensure_repository(collection_name)  # Load with defaults
            except Exception:
                # 不存在
                return None

    def list_collections(self) -> List[str]:
        """Returns a list of all available collection names."""
        if not self.is_ready():
            return []
        with self._lock:
            # ChromaDB client has a list_collections method
            colls = self._client.list_collections()
            return [c.name for c in colls]

    def create_backup(self, backup_dir: str) -> str:
        """
        Creates a hot backup of the database.

        Mechanism:
        1. Acquire global lock (blocks new writes).
        2. Create a zip archive of the database directory.
        3. Release lock.

        Args:
            backup_dir (str): Directory to store the zip file.

        Returns:
            str: Path to the generated zip file.
        """
        if not self.is_ready():
            raise RuntimeError("Engine not ready")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vectordb_backup_{timestamp}"
        archive_path = os.path.join(backup_dir, filename)

        # CRITICAL: Hold the lock to prevent modification during copy
        with self._lock:
            logger.info(f"Starting backup... Locking DB at {self._db_path}")
            try:
                # shutil.make_archive creates a zip file.
                # Note: This reads files. If Chroma holds exclusive locks (Windows),
                # this might fail. Usually SQLite allows read-sharing.
                zip_file = shutil.make_archive(
                    base_name=archive_path,
                    format='zip',
                    root_dir=self._db_path
                )
                logger.info(f"Backup created at: {zip_file}")
                return zip_file
            except Exception as e:
                logger.error(f"Backup failed: {e}")
                raise e
            finally:
                logger.info("Backup finished. Unlocking DB.")

    def restore_backup(self, zip_file_path: str):
        """
        Restores the database from a zip file and performs a HOT RELOAD.

        Mechanism:
        1. Acquire lock.
        2. Dereference and unload the Chroma Client (attempt to release file handles).
        3. Wipe the current DB directory.
        4. Unzip the backup into the DB directory.
        5. Re-initialize the Chroma Client.
        """
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError("Backup file not found")

        with self._lock:
            logger.info("Starting Restore... Service locked.")
            try:
                # 0. Mark status as INIT during restore
                self._status = VectorStorageEngine.Status.INIT
                self._error_message = None
                self._ready_event.clear()

                # 1. Unload resources to release file locks
                self._repos.clear()  # Clear repo cache
                del self._client  # Remove reference
                self._client = None
                gc.collect()  # Force Garbage Collection

                logger.info("Client unloaded. Replacing files...")

                # 2. Wipe current directory
                # Warning: If this fails (e.g., file locked by OS), we are in trouble.
                # In production, you might rename current to .bak before deleting.
                if os.path.exists(self._db_path):
                    shutil.rmtree(self._db_path)

                os.makedirs(self._db_path, exist_ok=True)

                # 3. Unzip
                shutil.unpack_archive(zip_file_path, self._db_path)
                logger.info("Files unpacked.")

                # 4. Reload Client
                import chromadb
                self._client = chromadb.PersistentClient(path=self._db_path)

                # Restore success -> READY
                self._status = VectorStorageEngine.Status.READY
                self._error_message = None
                self._ready_event.set()

                logger.info("Client re-initialized. Restore Complete.")

            except Exception as e:
                # If restore fails, the DB might be in a corrupted state.
                self._status = VectorStorageEngine.Status.ERROR
                self._error_message = f"Restore failed: {e}"
                logger.error(f"FATAL: Restore failed: {e}")
                raise e


class VectorCollectionRepo:
    """
    VectorCollectionRepo: Manages a specific collection of documents.

    Responsibilities:
    1. Text chunking and splitting.
    2. CRUD operations for documents (Add, Search, Delete).
    3. Managing the relationship between `doc_id` (User concept) and `chunk_id` (DB concept).
    """

    def __init__(
            self,
            client: Any,  # Typed as Any to avoid strict dependency on top-level import
            model: Any,
            collection_name: str,
            chunk_size: int,
            chunk_overlap: int,
            db_path: str,
    ):
        """
        Initialized by VectorStorageEngine. Do not instantiate directly.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._client = client
        self._model = model
        self._collection_name = collection_name
        self._db_path = db_path
        self._current_config = {}
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None

        # Get or create the actual Chroma collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.update_config(chunk_size, chunk_overlap)

    def update_config(self, chunk_size: int, chunk_overlap: int):
        """Updates the text splitter configuration for future operations."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
        )
        self._current_config = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

    def get_config(self):
        return self._current_config

    def _vectorize(self, texts: List[str]) -> Any:
        """
        MEMORY FIX: Use batch_size to prevent OOM with large inputs.
        """
        # batch_size=32 is a safe default for CPUs and small GPUs.
        # If texts list is huge (e.g. 10k chunks), this processes them 32 at a time.
        return self._model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

    def upsert_document(
            self,
            doc_id: str,
            text: str,
            metadata: Dict[str, Any] = None,
            on_embeddings=None
    ) -> List[str]:
        """
        Upserts a document: fully replaces any existing document with the same doc_id.

        CRITICAL: This method performs a "Delete-then-Insert" strategy.
        It first deletes ALL existing chunks associated with `doc_id` to ensure
        no stale chunks remain.

        Args:
            doc_id (str): Unique identifier for the document.
            text (str): The full text content.
            metadata (Dict): Searchable metadata (e.g., {"timestamp": 123}).
            on_embeddings (Callable): optional hook called after embeddings computed.
                Signature: on_embeddings(**payload) or on_embeddings(payload_dict)

        Returns:
            List[str]: The list of generated chunk IDs.
        """
        chunks = None
        chunk_ids = None
        chunk_metadatas = None
        embeddings_np = None

        try:
            if not text:
                return []
            if metadata is None:
                metadata = {}

            try:
                self._collection.delete(where={"original_doc_id": doc_id})
            except Exception:
                pass

            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (list, dict)):
                    clean_metadata[k] = json.dumps(v, ensure_ascii=False)
                elif isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)

            chunks = self._text_splitter.split_text(text)
            if not chunks:
                return []

            chunk_ids = [f"{doc_id}#chunk_{i}" for i in range(len(chunks))]

            chunk_metadatas = []
            total_chunks = len(chunks)
            for i in range(total_chunks):
                meta = {
                    "original_doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                }
                meta.update(clean_metadata)
                chunk_metadatas.append(meta)

            embeddings_np = self._vectorize(chunks).astype(np.float32, copy=False)

            if on_embeddings is not None:
                try:
                    payload = {
                        "collection_name": self._collection_name,
                        "doc_id": doc_id,
                        "chunk_ids": chunk_ids,
                        "embedding_shape": tuple(embeddings_np.shape),
                        "embedding_dtype": str(embeddings_np.dtype),
                        "metadata": clean_metadata,
                        "chunks_count": len(chunks),
                    }
                    on_embeddings(**payload)
                except Exception as e:
                    logger.warning(f"[VectorRepo] on_embeddings hook failed: {e}")

            MAX_BATCH = 256

            for i in range(0, len(chunk_ids), MAX_BATCH):
                end = i + MAX_BATCH
                self._collection.upsert(
                    ids=chunk_ids[i:end],
                    documents=chunks[i:end],
                    embeddings=embeddings_np[i:end].tolist(),
                    metadatas=chunk_metadatas[i:end]
                )

            return chunk_ids

        finally:
            embeddings_np = None
            chunks = None
            chunk_metadatas = None

            cleanup_memory(
                tag=f"upsert_document:{self._collection_name}:{doc_id}",
                aggressive=False
            )

    def exists(self, doc_id: str) -> bool:
        """Checks if a document (any of its chunks) exists in the DB."""
        try:
            # Minimal query to check existence
            result = self._collection.get(
                where={"original_doc_id": doc_id},
                limit=1,
                include=[]  # We don't need data, just the check
            )
            return len(result["ids"]) > 0
        except Exception:
            return False

    def exists_batch(self, doc_ids: Union[str, List[str]]) -> Dict[str, bool]:
        """
        Check existence of one or multiple documents in the collection.

        This method performs a batch query to check which document IDs exist
        in the collection. It's more efficient than individual queries when
        checking multiple IDs.

        Args:
            doc_ids: Single document ID as string, or list of document IDs.

        Returns:
            Dictionary mapping document IDs to boolean existence status.
            Example: {"doc1": True, "doc2": False}

        Raises:
            ValueError: If doc_ids is empty or not a valid type.
        """
        # Normalize input
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        elif not isinstance(doc_ids, list) or not doc_ids:
            raise ValueError("doc_ids must be a non-empty string or list of strings")

        # Existence check via chunk_0 ids
        chunk0_ids = [f"{doc_id}#chunk_0" for doc_id in doc_ids]

        try:
            # Only need ids, so include can be empty
            # ids are always returned by Chroma get()
            #   [1](https://docs.trychroma.com/reference/python/collection)
            result = self._collection.get(ids=chunk0_ids, include=[])
            found_ids = set(result.get("ids") or [])
        except Exception as e:
            logger.error(f"Batch existence check failed (chunk-id strategy): {e}")
            return {doc_id: False for doc_id in doc_ids}

        # Map back to doc_id
        return {doc_id: (f"{doc_id}#chunk_0" in found_ids) for doc_id in doc_ids}

    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes all chunks associated with the given doc_id.
        """
        try:
            # Delete based on metadata filter
            self._collection.delete(where={"original_doc_id": doc_id})
            return True
        except Exception as e:
            print(f"[VectorRepo] Error deleting document {doc_id}: {e}")
            return False

    def search(
            self,
            query_text: str,
            top_n: int = 5,
            score_threshold: float = 0.0,
            filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with metadata filtering and deduplication.

        Args:
            query_text (str): The search query.
            top_n (int): Number of unique documents to return.
            score_threshold (float): Minimum similarity score (0 to 1).
            filter_criteria (Dict): MongoDB-style filter (e.g., {"category": "news"}).

        Returns:
            List[Dict]: List of result objects containing doc_id, score, text, metadata.
        """

        query_vector = self._vectorize([query_text])[0].tolist()

        # Fetch more chunks because multiple chunks may belong to the same article.
        fetch_k = max(top_n * 5, 20)

        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                where=filter_criteria,
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            print(f"[VectorRepo] Search failed: {e}")
            return []

        if not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        return self._aggregate_search_chunks_to_articles(
            ids=ids,
            distances=distances,
            metadatas=metadatas,
            documents=documents,
            top_n=top_n,
            score_threshold=score_threshold,
        )

    def clear(self):
        """WARNING: Deletes all data in this collection."""
        try:
            self._client.delete_collection(self._collection_name)
            # Re-init handle
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"[VectorRepo] Error clearing collection: {e}")

    def count(self) -> int:
        """Returns total chunk count."""
        return self._collection.count()

    def list_documents(self, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        Returns a paginated list of documents (without embeddings).
        Useful for browsing data.
        """
        # ChromaDB .get() supports limit and offset
        results = self._collection.get(
            limit=limit,
            offset=offset,
            include=["metadatas", "documents"]
        )

        # Format into a cleaner list of dicts
        items = []
        if results['ids']:
            for i in range(len(results['ids'])):
                items.append({
                    "chunk_id": results['ids'][i],
                    "doc_id": results['metadatas'][i].get("original_doc_id", "unknown"),
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })

        return {
            "items": items,
            "total": self.count(),
            "limit": limit,
            "offset": offset
        }

    def timestamp_stats(self, time_field: str = "timestamp", scan_limit: int = 20000, offset: int = 0) -> Dict[
        str, Any]:
        """
        Scan metadatas to compute min/max timestamp.
        This is for testing & window selection. It does NOT assume semantics.
        """
        try:
            results = self._collection.get(
                limit=scan_limit,
                offset=offset,
                include=["metadatas"]
            )
            metas = results.get("metadatas") or []
            min_ts, max_ts = None, None
            cnt = 0
            for m in metas:
                if not m:
                    continue
                ts = m.get(time_field)
                if ts is None:
                    continue
                try:
                    ts = float(ts)
                except Exception:
                    continue
                cnt += 1
                min_ts = ts if (min_ts is None or ts < min_ts) else min_ts
                max_ts = ts if (max_ts is None or ts > max_ts) else max_ts

            return {
                "time_field": time_field,
                "min_ts": min_ts,
                "max_ts": max_ts,
                "count_scanned": len(metas),
                "count_with_ts": cnt,
                "offset": offset,
                "scan_limit": scan_limit
            }
        except Exception as e:
            logger.error(f"timestamp_stats failed: {e}")
            return {"error": str(e), "time_field": time_field}

    def _fetch_chunks_for_analysis(
            self,
            filter_criteria: Optional[Dict[str, Any]],
            time_range: Optional[Tuple[float, float]],
            limit: int,
            time_field: str = "timestamp",
            include_documents: bool = True,
            offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Internal chunk-level fetch for analysis.

        This method talks directly to Chroma and returns raw chunk records.
        It is intentionally private because chunk is an internal storage detail.

        Public APIs should return article/document-level records instead.

        Args:
            filter_criteria:
                Chroma metadata filter.
                Example:
                    {"category": "news"}
                    {"source": {"$eq": "xxx"}}
                    {"$and": [{"category": "news"}, {"lang": "zh"}]}

            time_range:
                Optional (start_ts, end_ts). If provided, it will be combined
                with filter_criteria using "$and".

            limit:
                Chunk scan limit. This is NOT article limit.

            time_field:
                Metadata field used for time filtering.

            include_documents:
                Whether to fetch chunk text from Chroma.
                For pure vector analysis, this can be False to save memory.

            offset:
                Chroma pagination offset.

        Returns:
            {
                "unit": "chunk",
                "ids": [...],
                "embeddings": [...],
                "metadatas": [...],
                "documents": [...],      # present, maybe empty strings if not requested
            }
        """
        if limit is None or limit <= 0:
            return {
                "unit": "chunk",
                "ids": [],
                "embeddings": [],
                "metadatas": [],
                "documents": [],
            }

        criteria = dict(filter_criteria or {})

        where_clauses = []

        # 1. Add time range clauses.
        if time_range is not None:
            start, end = time_range

            if start is not None:
                where_clauses.append({
                    time_field: {"$gte": float(start)}
                })

            if end is not None:
                where_clauses.append({
                    time_field: {"$lte": float(end)}
                })

        # 2. Add user filter criteria.
        #
        # Important:
        # Chroma where syntax is sensitive when combining conditions.
        # If we have both time clauses and user criteria, wrap them under "$and".
        if criteria:
            where_clauses.append(criteria)

        # 3. Build final Chroma where expression.
        if not where_clauses:
            where = None
        elif len(where_clauses) == 1:
            where = where_clauses[0]
        else:
            where = {"$and": where_clauses}

        # 4. Build include fields.
        include = ["embeddings", "metadatas"]
        if include_documents:
            include.append("documents")

        try:
            results = self._collection.get(
                where=where,
                limit=limit,
                offset=offset,
                include=include,
            )

            ids = results.get("ids") or []

            embeddings = results.get("embeddings", None)
            if embeddings is None:
                embeddings = []

            metadatas = results.get("metadatas") or []

            if include_documents:
                documents = results.get("documents") or []
            else:
                documents = [""] * len(ids)

            # Normalize lengths defensively.
            if len(metadatas) < len(ids):
                metadatas = metadatas + [{} for _ in range(len(ids) - len(metadatas))]

            if len(documents) < len(ids):
                documents = documents + ["" for _ in range(len(ids) - len(documents))]

            return {
                "unit": "chunk",
                "ids": ids,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "documents": documents,
            }

        except Exception as e:
            logger.error(f"DB Fetch chunks failed: {e}")
            return {
                "unit": "chunk",
                "ids": [],
                "embeddings": [],
                "metadatas": [],
                "documents": [],
                "error": str(e),
            }

    def _group_chunks_by_article(
            self,
            ids: List[str],
            metadatas: List[Dict[str, Any]],
            documents: Optional[List[str]] = None,
            embeddings: Optional[Any] = None,
            distances: Optional[List[float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group chunk-level records by original_doc_id.

        This is a shared internal utility used by both:
        - analysis aggregation
        - search aggregation

        It does not decide aggregation policy.
        """
        if documents is None:
            documents = [""] * len(ids)

        embeddings_np = None
        if embeddings is not None:
            embeddings_np = np.asarray(embeddings, dtype=np.float32)

        groups = {}

        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

            doc_id = meta.get("original_doc_id")
            if not doc_id:
                doc_id = str(chunk_id).split("#chunk_")[0]

            chunk_index = meta.get("chunk_index", 0)
            try:
                chunk_index = int(chunk_index)
            except Exception:
                chunk_index = 0

            if doc_id not in groups:
                groups[doc_id] = {
                    "doc_id": doc_id,
                    "chunk_ids": [],
                    "chunk_indexes": [],
                    "metadatas": [],
                    "documents": [],
                    "embeddings": [],
                    "distances": [],
                }

            g = groups[doc_id]
            g["chunk_ids"].append(chunk_id)
            g["chunk_indexes"].append(chunk_index)
            g["metadatas"].append(meta)
            g["documents"].append(documents[i] if i < len(documents) else "")

            if embeddings_np is not None:
                g["embeddings"].append(embeddings_np[i])

            if distances is not None:
                g["distances"].append(float(distances[i]))

        return groups

    def _aggregate_search_chunks_to_articles(
            self,
            ids: List[str],
            distances: List[float],
            metadatas: List[Dict[str, Any]],
            documents: List[str],
            top_n: int,
            score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate chunk search results into article-level search results.

        Search policy:
        - article score = max chunk similarity
        - representative content = best matching chunk
        - representative metadata = best matching chunk metadata
        """
        groups = self._group_chunks_by_article(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
            distances=distances,
        )

        article_results = []

        for doc_id, g in groups.items():
            if not g["distances"]:
                continue

            # cosine distance -> similarity
            scores = [1.0 - d for d in g["distances"]]
            best_pos = int(np.argmax(scores))
            best_score = float(scores[best_pos])

            if best_score < score_threshold:
                continue

            best_meta = dict(g["metadatas"][best_pos]) if g["metadatas"] else {}
            best_content = g["documents"][best_pos] if g["documents"] else ""
            best_chunk_id = g["chunk_ids"][best_pos]

            best_meta["original_doc_id"] = doc_id

            article_results.append({
                "doc_id": doc_id,
                "chunk_id": best_chunk_id,
                "score": best_score,
                "content": best_content,
                "metadata": best_meta,
                "matched_chunk_count": len(g["chunk_ids"]),
            })

        article_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return article_results[:top_n]

    def _aggregate_chunks_to_articles(
            self,
            chunk_data: Dict[str, Any],
            time_field: str = "timestamp",
    ) -> Dict[str, Any]:
        ids = chunk_data.get("ids") or []
        embeddings = chunk_data.get("embeddings")
        metadatas = chunk_data.get("metadatas") or []
        documents = chunk_data.get("documents") or [""] * len(ids)

        if not ids or embeddings is None:
            return {
                "unit": "article",
                "ids": [],
                "embeddings": np.empty((0, 0), dtype=np.float32),
                "metadatas": [],
                "documents": [],
                "chunk_counts": [],
                "source_chunk_ids": [],
            }

        groups = self._group_chunks_by_article(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
            embeddings=embeddings,
        )

        article_ids = []
        article_embeddings = []
        article_metadatas = []
        article_documents = []
        chunk_counts = []

        for doc_id, g in groups.items():
            chunk_arr = np.asarray(g["embeddings"], dtype=np.float32)

            article_emb = chunk_arr.mean(axis=0)

            norm = np.linalg.norm(article_emb)
            if norm > 0:
                article_emb = article_emb / norm

            indexes = np.asarray(g["chunk_indexes"], dtype=np.int32)
            first_pos = int(np.argmin(indexes)) if len(indexes) > 0 else 0

            base_meta = dict(g["metadatas"][first_pos]) if g["metadatas"] else {}
            base_meta["original_doc_id"] = doc_id
            base_meta["chunk_count"] = len(g["chunk_ids"])
            base_meta["embedding_aggregation"] = "mean_normalized"

            preview = g["documents"][first_pos] if g["documents"] else ""

            article_ids.append(doc_id)
            article_embeddings.append(article_emb.astype(np.float32, copy=False))
            article_metadatas.append(base_meta)
            article_documents.append(preview)
            chunk_counts.append(len(g["chunk_ids"]))

        article_embeddings_np = (
            np.asarray(article_embeddings, dtype=np.float32)
            if article_embeddings
            else np.empty((0, 0), dtype=np.float32)
        )

        return {
            "unit": "article",
            "ids": article_ids,
            "embeddings": article_embeddings_np,
            "metadatas": article_metadatas,
            "documents": article_documents,
            "chunk_counts": chunk_counts,
        }

    def fetch_articles_for_analysis(
            self,
            filter_criteria: Dict[str, Any],
            time_range: Optional[Tuple[float, float]],
            chunk_scan_limit: int,
            time_field: str = "timestamp",
            include_documents: bool = True,
    ) -> Dict[str, Any]:
        """
        Public analysis fetch API.

        Returns article-level records for downstream analysis.

        Important:
        - Chroma stores chunks internally.
        - Repo exposes articles/documents externally.
        - `chunk_scan_limit` currently means chunk scan limit, not final article limit.
          This keeps compatibility with the old fetch_for_analysis behavior.
        """
        chunk_data = self._fetch_chunks_for_analysis(
            filter_criteria=filter_criteria,
            time_range=time_range,
            limit=chunk_scan_limit,
            time_field=time_field,
            include_documents=include_documents,
        )

        if not chunk_data:
            return {
                "unit": "article",
                "ids": [],
                "embeddings": np.empty((0, 0), dtype=np.float32),
                "metadatas": [],
                "documents": [],
                "chunk_counts": [],
                "source_chunk_ids": [],
            }

        if chunk_data.get("error"):
            return {
                "unit": "article",
                "ids": [],
                "embeddings": np.empty((0, 0), dtype=np.float32),
                "metadatas": [],
                "documents": [],
                "chunk_counts": [],
                "source_chunk_ids": [],
                "error": chunk_data.get("error"),
            }

        return self._aggregate_chunks_to_articles(
            chunk_data=chunk_data,
            time_field=time_field,
        )

    def fetch_for_analysis(
            self,
            filter_criteria: Dict[str, Any],
            time_range: Optional[Tuple[float, float]],
            limit: int,
            time_field: str = "timestamp",
    ) -> Dict[str, Any]:
        """
        Public analysis API.

        Returns article-level records.

        Chunks are an internal storage detail and should not leak to Pipeline.
        """
        return self.fetch_articles_for_analysis(
            filter_criteria=filter_criteria,
            time_range=time_range,
            chunk_scan_limit=limit,
            time_field=time_field,
            include_documents=True,
        )

    def run_analysis(self, config: AnalysisConfig) -> Dict[str, Any]:
        with memory_scope(
                f"run_analysis:{self._collection_name}",
                aggressive_cleanup=True
        ):
            pipeline = None
            try:
                pipeline = IntelligenceAnalysisPipeline(repo_interface=self, config=config)
                result = pipeline.execute()

                try:
                    self._save_analysis_report(result)
                except Exception as e:
                    logger.error(f"Failed to save analysis report text: {e}")

                return result

            finally:
                if pipeline is not None:
                    try:
                        pipeline.close()
                    except Exception:
                        pass
                    del pipeline

                cleanup_memory(
                    tag=f"run_analysis finally:{self._collection_name}",
                    aggressive=True
                )

    def _save_analysis_report(self, result: Dict[str, Any]):
        """
        将聚类分析结果保存为文本文件，放置在 DB 目录同级。
        格式体现层次结构，仅包含标题。
        """
        if not result:
            return

        # 1. 确定保存路径：DB目录的同级目录
        # 如果 self._db_path 是 ".../VectorDB/storage"，我们希望存在 ".../VectorDB/" 下
        base_dir = os.path.dirname(self._db_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Clustering_Report_{self._collection_name}_{timestamp}.txt"
        file_path = os.path.join(base_dir, filename)

        # 2. 递归写入辅助函数
        def write_node(f, node, level=0):
            indent = "    " * level

            # 获取节点信息 (根据 Pipeline 常见的输出结构适配)
            # 假设结构包含 'label', 'summary', 'children' 或 'items'
            label = node.get("label") or node.get("title") or f"Cluster {node.get('id', '?')}"

            # 写入当前节点标题
            f.write(f"{indent}- {label}\n")

            # 处理子节点 (Sub-clusters)
            children = node.get("children", [])
            for child in children:
                write_node(f, child, level + 1)

            # 处理叶子节点文档 (Documents/Items)
            items = node.get("items", []) or node.get("documents", [])
            for item in items:
                # 尝试从 metadata 获取标题，如果没有则使用 ID 或截断的内容
                meta = item.get("metadata", {})
                title = meta.get("title") or meta.get("source") or item.get("doc_id") or "Untitled Document"
                f.write(f"{indent}    * [DOC] {title}\n")

        # 3. 执行写入
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Analysis Report for Collection: {self._collection_name}\n")
                f.write(f"Time: {datetime.datetime.now()}\n")
                f.write("=" * 50 + "\n\n")

                # 假设 result['clusters'] 是顶层列表
                clusters = result.get("clusters", [])
                if isinstance(clusters, list):
                    for cluster in clusters:
                        write_node(f, cluster, level=0)
                else:
                    # 如果 result 本身就是单个根节点
                    write_node(f, result, level=0)

            logger.info(f"Analysis report saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error writing analysis report file: {e}")
            raise e
