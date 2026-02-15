import time
import requests
import random
import logging
from typing import List, Dict, Any, Optional
from functools import wraps

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("VectorDBClient")


class RetryableError(RuntimeError):
    """Base class for retryable errors."""
    pass

class NonRetryableError(RuntimeError):
    """Base class for non-retryable errors."""
    pass

class ServerBusyError(RetryableError):
    """Server is busy, can be retried."""
    pass

class ServerInitializingError(RetryableError):
    """Server is initializing, can be retried."""
    pass

class AuthenticationError(NonRetryableError):
    """Authentication failed, should not retry."""
    pass

class InvalidRequestError(NonRetryableError):
    """Invalid request, should not retry."""
    pass


class VectorDBInitializationError(Exception):
    pass


class VectorDBTimeoutError(TimeoutError):
    """Raised when the operation exceeds the maximum retry duration."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_execute(self) -> bool:
        """Check if request can be executed based on current state."""
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            # Reset on successful execution in half-open state
            self.state = "CLOSED"
            self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery

    def on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"  # Back to open if half-open attempt fails

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics for monitoring."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time
        }


def retry_with_timeout(default_timeout: float = 60.0, max_retries: int = -1):
    """
    Decorator that implements an exponential backoff retry strategy with a global time budget.

    It specifically handles `ServerBusyError`, `ServerInitializingError`, and network-level
    exceptions, while letting logical errors (Auth, Bad Request) pass through immediately.

    Args:
        default_timeout (float): The total time budget (in seconds) allowed for the operation
                                 before raising a VectorDBTimeoutError.
        max_retries (int): Maximum number of retry attempts. -1 implies infinite retries
                           within the timeout budget.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Extract 'timeout' from arguments if provided (runtime override)
            #    We pop it so it doesn't get passed to the underlying method if it doesn't expect it.
            total_timeout = kwargs.pop('timeout', default_timeout)

            start_time = time.time()
            retries = 0
            delay = 1.0  # Initial backoff delay
            max_delay = 10.0

            last_error = None

            while (max_retries < 0) or (retries < max_retries):
                # 2. Check Time Budget
                elapsed = time.time() - start_time
                if elapsed > total_timeout:
                    error_msg = f"Operation timed out after {elapsed:.2f}s (Max: {total_timeout}s). Last error: {last_error}"
                    logger.error(error_msg)
                    raise VectorDBTimeoutError(error_msg)

                try:
                    # 3. Attempt the operation
                    return func(*args, **kwargs)

                # 4. Catch Retryable Errors
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    last_error = f"Connection failed: {e}"

                except RuntimeError as e:
                    # Check for server busy signals
                    error_str = str(e).lower()
                    if "busy" in error_str or "queue" in error_str or "initializing" in error_str:
                        last_error = f"Server busy: {e}"
                    else:
                        # Non-retryable logic error (e.g. 401 Auth, 400 Bad Request)
                        raise e

                # 5. Backoff Strategy
                # Calculate remaining time to avoid sleeping past the timeout
                remaining = total_timeout - (time.time() - start_time)
                if remaining <= 0:
                    continue  # Loop back to trigger the timeout check immediately

                # Sleep = min(exponential_backoff, remaining_time)
                sleep_time = min(delay + random.uniform(0, 0.5), remaining)

                logger.warning(
                    f"{last_error} | Retrying in {sleep_time:.2f}s... (Elapsed: {time.time() - start_time:.1f}s)")
                time.sleep(sleep_time)

                # Increase delay for next round
                delay = min(delay * 2.0, max_delay)
                retries += 1

        return wrapper

    return decorator


class VectorDBClient:
    """
    A robust Python client for the standalone VectorDB Service.

    This client manages the connection lifecycle, health checks, and collection creation.
    It includes a Circuit Breaker pattern to prevent cascading failures when the
    service is unavailable.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Args:
            base_url (str): The root URL of the VectorDB service.
        """
        self.base_url = base_url.rstrip("/")
        self._circuit_breaker = CircuitBreaker()  # Instance-level circuit breaker

    def get_status(self) -> Dict[str, Any]:
        """
        Retrieves the raw status JSON from the service.

        Returns:
            Dict[str, Any]: Service status info (e.g., {"status": "ready", "model": "..."}).
                            Returns {"status": "unreachable"} on connection failure.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/status", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"status": "unreachable", "error": str(e)}

    def get_queue_status(self) -> Dict[str, Any]:
        """Check the depth of the async processing queue."""
        try:
            resp = requests.get(f"{self.base_url}/api/status/queue", timeout=2)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"qsize": -1, "status": "unknown"}

    def wait_until_ready(self, timeout: float = 60.0, poll_interval: float = 2.0) -> bool:
        """
        Blocks execution until the VectorDB service reports a 'ready' status.

        This handles the 'initializing' state (e.g., loading heavy ML models) by polling.

        Args:
            timeout (float): Max wait time in seconds.
            poll_interval (float): Seconds between status checks.

        Returns:
            bool: True if service is ready.

        Raises:
            TimeoutError: If the service is not ready within the timeout.
            VectorDBInitializationError: If the service reports an explicit internal error.
        """
        start_time = time.time()
        # print(f"[Client] Waiting for VectorDB at {self.base_url} (Timeout: {timeout}s)...")

        while True:
            if (time.time() - start_time) > timeout:
                raise TimeoutError(f"VectorDB service not ready after {timeout} seconds.")

            try:
                resp = requests.get(f"{self.base_url}/api/status", timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status")

                    if status == "ready":
                        print(f"[Client] VectorDB is READY.")
                        return True
                    elif status == "error":
                        raise VectorDBInitializationError(f"Server failed: {data.get('error')}")
                    # If initializing, loop again
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                print(f"[Client] Warning during poll: {e}")

            time.sleep(poll_interval)

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return self._circuit_breaker.get_metrics()

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker (for testing/recovery)."""
        self._circuit_breaker = CircuitBreaker()
        logger.info("Circuit breaker manually reset")

    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def create_collection(self, name: str, chunk_size: int = 512,
                          chunk_overlap: int = 50, **kwargs) -> "RemoteCollection":
        """
        Creates a new collection or updates the config of an existing one.

        Protected by a Circuit Breaker to fail fast if the service is down.

        Args:
            name (str): Unique collection identifier.
            chunk_size (int): Token limit for text chunks.
            chunk_overlap (int): Overlap between chunks.
            **kwargs: Additional configuration parameters.

        Returns:
            RemoteCollection: A handle to operate on the collection.

        Raises:
            ServerBusyError: If the circuit breaker is open.
        """
        if not self._circuit_breaker.can_execute():
            raise ServerBusyError("Circuit breaker is OPEN, rejecting request")

        try:
            url = f"{self.base_url}/api/collections"
            payload = {
                "name": name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }

            # Short timeout for individual requests
            resp = requests.post(url, json=payload, timeout=5)

            if resp.status_code == 503:
                self._handle_503_response(resp)

            resp.raise_for_status()
            return RemoteCollection(self.base_url, name)

        except Exception as e:
            self._circuit_breaker.on_failure()
            raise

    def get_collection(self, name: str) -> "RemoteCollection":
        """
        Gets a handle to an EXISTING collection.
        Note: This does not verify existence immediately. Operations will fail if it doesn't exist.
        """
        return RemoteCollection(self.base_url, name)

    def list_collections(self) -> List[str]:
        """Lists all available collections."""
        resp = requests.get(f"{self.base_url}/api/collections")
        resp.raise_for_status()
        return resp.json().get("collections", [])

    def _handle_503_response(self, resp: requests.Response):
        """专门处理503响应的辅助方法"""
        try:
            error_data = resp.json()
            error_msg = error_data.get("error", "Unknown")
            error_code = error_data.get("error_code")

            if error_code == "BUSY" or "busy" in error_msg.lower():
                raise ServerBusyError(f"Server busy: {error_msg}")
            elif error_code == "INIT" or "initializing" in error_msg.lower():
                raise ServerInitializingError(f"Server initializing: {error_msg}")
            else:
                raise RetryableError(f"Service Unavailable: {error_msg}")
        except ValueError:
            raise RetryableError(f"Service Unavailable: {resp.text}")


class RemoteCollection:
    """
    Represents a specific collection on the remote VectorDB service.

    Provides methods to index (upsert), search, and manage documents.
    Most operations support automatic retries for transient server errors.
    """

    def __init__(self, base_url: str, name: str):
        self.api_url = f"{base_url}/api/collections/{name}"
        self.name = name

    def _handle_response(self, resp: requests.Response) -> Any:
        """
        Enhanced error classification with specific exception types.
        """
        if resp.status_code == 503:
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", "Unknown")
                error_code = error_data.get("error_code")  # Server should provide error codes
            except:
                error_msg = "Service Unavailable"
                error_code = None

            # Classify based on error code first, then fallback to string matching
            if error_code == "BUSY":
                raise ServerBusyError(f"Server busy: {error_msg}")
            elif error_code == "INIT":
                raise ServerInitializingError(f"Server initializing: {error_msg}")
            elif "initializing" in error_msg.lower():
                raise ServerInitializingError(f"Server initializing: {error_msg}")
            elif "queue" in error_msg.lower() or "busy" in error_msg.lower():
                raise ServerBusyError(f"Server busy: {error_msg}")
            else:
                raise RetryableError(f"Service Unavailable: {error_msg}")

        if resp.status_code == 401:
            raise AuthenticationError("Authentication failed")
        if resp.status_code == 400:
            raise InvalidRequestError(f"Invalid request: {resp.text}")
        if resp.status_code == 404:
            raise InvalidRequestError(f"Collection '{self.name}' not found.")

        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()

    @retry_with_timeout(default_timeout=120.0)  # Default 2 minutes total retry
    def upsert(self, doc_id: str, text: str, metadata: Dict[str, Any] = None, **kwargs) -> Dict:
        """
        Enqueues a single document for indexing.

        Note: This operation is ASYNCHRONOUS. A successful return (HTTP 202) means
        the document has been queued, not necessarily indexed.

        Args:
            doc_id (str): Unique document identifier (e.g., UUID).
            text (str): The raw text content to be embedded.
            metadata (Dict): Associated metadata for filtering.
            timeout (float, optional): Override default retry timeout (default: 120s).

        Returns:
            Dict: Response payload, typically {'status': 'queued', 'doc_id': ...}
        """
        if metadata is None: metadata = {}
        payload = {"doc_id": doc_id, "text": text, "metadata": metadata}

        # KEY: Internal request timeout is small (5s).
        # The 'timeout' arg passed to this function is handled by the decorator loop.
        resp = requests.post(f"{self.api_url}/upsert", json=payload, timeout=10)

        if resp.status_code == 202:
            return resp.json()

        return self._handle_response(resp)

    @retry_with_timeout(default_timeout=120.0)
    def upsert_batch(self, documents: List[Dict], **kwargs) -> Dict:
        """
        Enqueues a batch of documents for indexing.

        More efficient than single upserts due to reduced network overhead.

        Args:
            documents (List[Dict]): List of dicts, each containing {"doc_id", "text", "metadata"}.

        Returns:
            Dict: Response payload indicating queue status.
        """
        resp = requests.post(f"{self.api_url}/upsert_batch", json=documents)
        if resp.status_code == 202:
            return resp.json()
        return self._handle_response(resp)

    def search(
            self,
            query: str,
            top_n: int = 5,
            score_threshold: float = 0.0,
            filter_criteria: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Performs a semantic similarity search.

        Args:
            query (str): The natural language query.
            top_n (int): Max number of results to return.
            score_threshold (float): Minimum similarity score (0.0 to 1.0) to include.
            filter_criteria (Dict, optional): Metadata filters (e.g., {"category": "news"}).

        Returns:
            List[Dict]: A list of results, sorted by relevance score.
        """
        payload = {
            "query": query,
            "top_n": top_n,
            "score_threshold": score_threshold,
            "filter_criteria": filter_criteria
        }
        resp = requests.post(f"{self.api_url}/search", json=payload)
        return self._handle_response(resp)

    def delete(self, doc_id: str) -> bool:
        """Deletes a document by ID."""
        resp = requests.delete(f"{self.api_url}/documents/{doc_id}")
        if resp.status_code == 404:
            return False
        res = self._handle_response(resp)
        return res.get("status") == "success"

    def stats(self) -> Dict:
        """Gets collection stats."""
        resp = requests.get(f"{self.api_url}/stats")
        return self._handle_response(resp)

    def clear(self) -> bool:
        """Clears all data in collection."""
        resp = requests.post(f"{self.api_url}/clear")
        res = self._handle_response(resp)
        return res.get("status") == "cleared"


    def exists(self, doc_id: str, include_pending: bool = False, **kwargs) -> bool:
        """
        - include_pending: if True, treat 'pending' as exists=True
        - still returns bool for backward compatibility
        """
        return self._exists_impl(doc_id, include_pending=include_pending, **kwargs)

    def exists_batch(self, doc_ids: List[str], include_pending: bool = False, **kwargs) -> Dict[str, bool]:
        """
        Checks the existence of multiple documents.

        Args:
            doc_ids (List[str]): List of document IDs to check.
            include_pending (bool): If True, returns True for documents that are currently
                                    in the processing queue but not yet indexed.
                                    Essential for avoiding duplicates during incremental builds.

        Returns:
            Dict[str, bool]: Mapping of doc_id to existence status.
        """
        return self._exists_batch_impl(doc_ids, include_pending=include_pending, **kwargs)

    def exists_state(self, doc_id: str, **kwargs) -> str:
        """
        Returns tri-state from server: 'present' | 'pending' | 'missing'
        (If server doesn't support it, fallback to 'present'/'missing')
        """
        return self._exists_state_impl(doc_id, **kwargs)

    @retry_with_timeout(default_timeout=30.0)
    def _exists_impl(self, doc_id: str, include_pending: bool = False, **kwargs) -> bool:
        """
        - pass include_pending to server via query param
        - interpret response:
            - prefer 'state' if present
            - fallback to 'exists' if older server
        """
        url = f"{self.api_url}/documents/{doc_id}/exists"
        params = {"include_pending": "1" if include_pending else "0"}
        resp = requests.get(url, params=params, timeout=5)

        data = self._handle_response(resp)

        # prefer tri-state
        state = data.get("state")
        if state:
            if include_pending:
                return state in ("present", "pending")
            return state == "present"

        # older server only returns exists bool
        return bool(data.get("exists", False))

    @retry_with_timeout(default_timeout=30.0)
    def _exists_batch_impl(self, doc_ids: List[str], include_pending: bool = False, **kwargs) -> Dict[str, bool]:
        url = f"{self.api_url}/exists"
        payload = {"doc_ids": doc_ids, "include_pending": bool(include_pending)}
        resp = requests.post(url, json=payload, timeout=5)

        data = self._handle_response(resp)

        if "exists_map" in data:
            return data.get("exists_map", {})

        states_map = data.get("states_map", {}) or {}
        if include_pending:
            return {d: (states_map.get(d) in ("present", "pending")) for d in doc_ids}
        return {d: (states_map.get(d) == "present") for d in doc_ids}

    @retry_with_timeout(default_timeout=30.0)
    def _exists_state_impl(self, doc_id: str, **kwargs) -> str:
        """
        Ask server for tri-state. If not available, fallback using exists.
        """
        url = f"{self.api_url}/documents/{doc_id}/exists"
        # tri-state doesn't require include_pending; ask strict by default
        resp = requests.get(url, params={"include_pending": "0"}, timeout=5)

        data = self._handle_response(resp)

        if "state" in data:
            return data["state"]

        # map bool -> state
        return "present" if data.get("exists", False) else "missing"

