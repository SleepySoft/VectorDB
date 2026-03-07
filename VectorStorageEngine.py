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
        return {
            "qsize": self._queue.qsize(),
            "status": "running" if self._worker_thread.is_alive() else "stopped"
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
                chunk_overlap=chunk_overlap
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
            chunk_overlap: int
    ):
        """
        Initialized by VectorStorageEngine. Do not instantiate directly.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._client = client
        self._model = model
        self._collection_name = collection_name
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
        if not text:
            return []
        if metadata is None:
            metadata = {}

        # 1) Delete old
        try:
            self._collection.delete(where={"original_doc_id": doc_id})
        except Exception:
            pass

        # 2) Clean metadata (keep your original behavior)
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (list, dict)):
                clean_metadata[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)

        # 3) Split
        chunks = self._text_splitter.split_text(text)
        if not chunks:
            return []

        # 4) Prepare IDs & metadatas
        chunk_ids = [f"{doc_id}#chunk_{i}" for i in range(len(chunks))]
        chunk_metadatas = []
        for i in range(len(chunks)):
            meta = {
                "original_doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            meta.update(clean_metadata)
            chunk_metadatas.append(meta)

        # 5) Vectorize (np.ndarray)
        embeddings_np = self._vectorize(chunks)  # convert_to_numpy=True in _vectorize
        # ---- NEW: emit hook BEFORE converting to list ----
        if on_embeddings is not None:
            try:
                # Keep payload compact: do NOT always pass chunks (may be large)
                payload = {
                    "collection_name": self._collection_name,
                    "doc_id": doc_id,
                    "chunk_ids": chunk_ids,
                    "embeddings": embeddings_np,  # np.ndarray
                    "metadata": clean_metadata,
                    "chunks_count": len(chunks),
                }
                # Support both fn(**kwargs) and fn(dict)
                try:
                    on_embeddings(**payload)
                except TypeError:
                    on_embeddings(payload)
            except Exception as e:
                logger.warning(f"[VectorRepo] on_embeddings hook failed: {e}")

        # 6) Insert (Chroma expects list)
        embeddings = embeddings_np.tolist()

        MAX_BATCH = 5000
        for i in range(0, len(chunk_ids), MAX_BATCH):
            end = i + MAX_BATCH
            self._collection.upsert(
                ids=chunk_ids[i:end],
                documents=chunks[i:end],
                embeddings=embeddings[i:end],
                metadatas=chunk_metadatas[i:end]
            )

        return chunk_ids

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

        # Ensure _vectorize receives List[str], and we extract the first vector
        query_vector = self._vectorize([query_text])[0].tolist()

        # Request more chunks than top_n because multiple chunks might come from same doc
        fetch_k = top_n * 3

        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                where=filter_criteria,  # Apply metadata filtering at DB level
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            print(f"[VectorRepo] Search failed: {e}")
            return []

        # Parse results
        # Chroma returns lists of lists (batch format), we usually query one at a time.
        if not results['ids'] or not results['ids'][0]:
            return []

        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]

        # Standardize results into a list of dicts
        raw_candidates = []
        for i in range(len(ids)):
            score = 1.0 - distances[i]  # Convert distance to similarity

            if score < score_threshold:
                continue

            raw_candidates.append({
                "doc_id": metadatas[i].get("original_doc_id", "unknown"),
                "chunk_id": ids[i],
                "score": score,
                "content": documents[i],
                "metadata": metadatas[i]
            })

        # Deduplicate: Keep only the highest scoring chunk per original_doc_id
        unique_docs_map = {}
        for candidate in raw_candidates:
            d_id = candidate["doc_id"]
            if d_id not in unique_docs_map:
                unique_docs_map[d_id] = candidate
            else:
                # If this chunk has a higher score than the one we already have, replace it
                if candidate["score"] > unique_docs_map[d_id]["score"]:
                    unique_docs_map[d_id] = candidate

        # Sort by score descending and take top N
        final_results = sorted(
            unique_docs_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return final_results[:top_n]

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

    def _extract_features(self,
                          metadatas: List[Dict],
                          weights: Dict[str, float],
                          time_field: str = "timestamp",
                          includes_metas: Optional[List[str]] = None
                          ) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler

        # 1. Temporal Feature (保持不变，如果没有 timestamp 默认为 0)
        times = np.array([m.get(time_field, 0) for m in metadatas]).reshape(-1, 1)

        # 2. Generic Density Feature (通用密度特征)
        # 逻辑：把 includes_metas 里提到的所有字段的信息量加起来
        densities = []

        target_keys = includes_metas if includes_metas else []

        for m in metadatas:
            score = 0
            for key in target_keys:
                val = m.get(key)

                # 此时 val 已经被 _smart_parse_value 处理过，可能是 List, Dict, Str, Int
                if isinstance(val, list):
                    score += len(val)  # 列表贡献长度权重
                elif isinstance(val, dict):
                    score += len(val)  # 字典贡献键值对数量
                elif isinstance(val, (int, float)):
                    score += float(val)  # 数值直接贡献大小
                elif val:
                    score += 1  # 非空字符串贡献 1 分

            densities.append(score)

        ents = np.array(densities).reshape(-1, 1)

        # Scale to [0, 1]
        scaler = MinMaxScaler()
        norm_times = scaler.fit_transform(times) * weights.get("time", 0.2)
        norm_ents = scaler.fit_transform(ents) * weights.get("entities", 0.5)

        return np.hstack([norm_times, norm_ents])

    def _aggregate_to_articles(self,
                               results: Dict,
                               weights: Dict[str, float],
                               time_field: str = "timestamp",
                               includes_metas: Optional[List[str]] = None
                               ):
        doc_groups = {}

        for i in range(len(results['ids'])):
            # --- 1. 智能清洗 Metadata ---
            raw_meta = results['metadatas'][i]
            clean_meta = {}
            for k, v in raw_meta.items():
                # 只有当字段在 includes_metas 里，或者没传 includes_metas (全都要) 时，才解析
                if includes_metas is None or k in includes_metas or k == time_field or k == "original_doc_id":
                    clean_meta[k] = self._smart_parse_value(v)
                else:
                    clean_meta[k] = v
                    # ---------------------------

            d_id = clean_meta.get("original_doc_id", "unknown")

            if d_id not in doc_groups:
                doc_groups[d_id] = {
                    "embeddings": [],
                    "metadata": clean_meta,  # 使用清洗后的
                    "text": results['documents'][i][:150]
                }
            doc_groups[d_id]["embeddings"].append(results['embeddings'][i])

        article_ids = list(doc_groups.keys())

        semantic_vecs = []
        for d_id in article_ids:
            avg_emb = np.mean(doc_groups[d_id]["embeddings"], axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-9)
            semantic_vecs.append(avg_emb * weights.get("semantic", 1.0))

        # 传递 includes_metas 给特征提取器
        meta_features = self._extract_features(
            metadatas=[doc_groups[d_id]["metadata"] for d_id in article_ids],
            weights=weights,
            time_field=time_field,
            includes_metas=includes_metas
        )

        X = np.hstack([np.array(semantic_vecs), meta_features])
        previews = [doc_groups[d_id]["text"] for d_id in article_ids]

        return X, article_ids, previews

    def _smart_parse_value(self, value: Any) -> Any:
        """
        尝试将字符串还原为 Python 对象（List/Dict）。
        如果失败，或者不是字符串，则原样返回。
        """
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if not (stripped.startswith('[') or stripped.startswith('{')):
            return value

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def fetch_for_analysis(
            self,
            filter_criteria: Dict[str, Any],
            time_range: Optional[Tuple[float, float]],
            limit: int,
            time_field: str = "timestamp",
    ) -> Dict[str, Any]:
        """
        Fetch raw records for analysis (embeddings + metadatas + documents + ids).
        NOTE: Chroma where filter requires that operator dict has exactly ONE operator.
              For a range query (gte & lte), must use $and with two separate clauses.
        """
        criteria = (filter_criteria or {}).copy()
        where = None

        # Build time range expression correctly for Chroma:
        # { "$and": [ {"timestamp":{"$gte":start}}, {"timestamp":{"$lte":end}}, ... ] }
        if time_range:
            start, end = time_range

            time_and = [
                {time_field: {"$gte": float(start)}},
                {time_field: {"$lte": float(end)}},
            ]

            if not criteria:
                where = {"$and": time_and}
            else:
                # If criteria already uses $and/$or, wrap it safely
                # Chroma expects where to have exactly one top-level operator OR field map
                # so we combine everything under $and.
                where = {"$and": time_and + [criteria]}

        else:
            # no time range
            where = criteria if criteria else None

        try:
            results = self._collection.get(
                where=where,
                limit=limit,
                include=["embeddings", "metadatas", "documents"]
            )
            return results
        except Exception as e:
            logger.error(f"DB Fetch failed: {e}")
            return {"error": str(e)}

    def run_analysis(self, config: AnalysisConfig) -> Dict[str, Any]:
        """
        Entry point for running an analysis session.
        Creates a disposable pipeline, executes it, and returns the result.
        """
        # Create pipeline instance (Connecting Repo <-> Pipeline)
        pipeline = IntelligenceAnalysisPipeline(repo_interface=self, config=config)

        # Execute
        try:
            result = pipeline.execute()
            return result
        finally:
            # Explicit cleanup if necessary, though Python GC handles it
            del pipeline
