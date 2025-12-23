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
    Decorator that retries the function until success or until 'timeout' expires.

    Args:
        default_timeout: The default total time (in seconds) allowed for the operation.
        max_retries: The max retries. -1 means retry forever.
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
    A Python client for the standalone VectorDB Service.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self._circuit_breaker = CircuitBreaker()  # Instance-level circuit breaker

    def get_status(self) -> Dict[str, Any]:
        """Check the raw status of the server."""
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
        Blocks until the VectorDB service is fully initialized and ready.
        """
        start_time = time.time()
        print(f"[Client] Waiting for VectorDB at {self.base_url} (Timeout: {timeout}s)...")

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
        Enhanced create_collection with circuit breaker and retry limits.
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
        Upserts a document.
        Args:
            doc_id:
            text:
            metadata:
            timeout (float): Total duration (in seconds) to keep retrying if server is busy.
                             If not provided, defaults to 120s.
            max_retries:
        Returns:
            Dict: {'status': 'queued', 'message': '...', 'doc_id': '...'}
                  The operation is NOT finished when this returns.
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
        documents: List of {"doc_id": str, "text": str, "metadata": dict}
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
        """Searches the remote DB."""
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
