# VectorDB/VectorDBClient.py

import os
import time
import requests
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
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


class ServiceNotConfiguredError(NonRetryableError):
    """
    Feature not enabled on server side (e.g., ClusterManager/store not configured).
    The service returns 501 in these cases.
    """
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
            if self.last_failure_time is not None and (time.time() - self.last_failure_time > self.recovery_timeout):
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)

    def on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time
        }


def retry_with_timeout(default_timeout: float = 60.0, max_retries: int = -1):
    """
    Decorator implementing exponential backoff retry with a global time budget.

    Handles:
      - requests ConnectionError/Timeout
      - ServerBusyError / ServerInitializingError / RetryableError
    Lets NonRetryableError pass through immediately.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            total_timeout = kwargs.pop('timeout', default_timeout)

            start_time = time.time()
            retries = 0
            delay = 1.0
            max_delay = 10.0
            last_error = None

            while (max_retries < 0) or (retries < max_retries):
                elapsed = time.time() - start_time
                if elapsed > total_timeout:
                    msg = f"Operation timed out after {elapsed:.2f}s (Max: {total_timeout}s). Last error: {last_error}"
                    logger.error(msg)
                    raise VectorDBTimeoutError(msg)

                try:
                    return func(*args, **kwargs)

                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    last_error = f"Connection failed: {e}"

                except RetryableError as e:
                    last_error = str(e)

                except NonRetryableError:
                    raise

                except requests.exceptions.HTTPError as e:
                    # In case caller forgot to classify status codes, treat as non-retryable by default
                    raise NonRetryableError(str(e)) from e

                except Exception as e:
                    # Unknown errors: do not retry by default
                    raise

                remaining = total_timeout - (time.time() - start_time)
                if remaining <= 0:
                    continue

                sleep_time = min(delay + random.uniform(0, 0.5), remaining)
                logger.warning(f"{last_error} | Retrying in {sleep_time:.2f}s... (Elapsed: {time.time() - start_time:.1f}s)")
                time.sleep(sleep_time)

                delay = min(delay * 2.0, max_delay)
                retries += 1

        return wrapper
    return decorator


class VectorDBClient:
    """
    A robust Python client for the standalone VectorDB Service.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self._circuit_breaker = CircuitBreaker()

    # ----------------
    # Core status APIs
    # ----------------
    def get_status(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/api/status", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"status": "unreachable", "error": str(e)}

    def get_queue_status(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/api/status/queue", timeout=2)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"qsize": -1, "status": "unknown"}

    def wait_until_ready(self, timeout: float = 60.0, poll_interval: float = 2.0) -> bool:
        start_time = time.time()
        while True:
            if (time.time() - start_time) > timeout:
                raise TimeoutError(f"VectorDB service not ready after {timeout} seconds.")

            try:
                resp = requests.get(f"{self.base_url}/api/status", timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status")
                    if status == "ready":
                        logger.info("VectorDB is READY.")
                        return True
                    elif status == "error":
                        raise VectorDBInitializationError(f"Server failed: {data.get('error')}")
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                logger.warning(f"Warning during poll: {e}")

            time.sleep(poll_interval)

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        return self._circuit_breaker.get_metrics()

    def reset_circuit_breaker(self):
        self._circuit_breaker = CircuitBreaker()
        logger.info("Circuit breaker manually reset")

    def get_job(self, job_id: str) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/jobs/{job_id}", timeout=10)
        return RemoteCollection._static_handle_response(resp)

    def get_job_result(self, job_id: str, request_timeout: int = 30) -> Any:
        resp = requests.get(f"{self.base_url}/api/jobs/{job_id}/result", timeout=request_timeout)
        return RemoteCollection._static_handle_response(resp)

    def wait_job_result(
            self,
            job_id: str,
            timeout: float = 120.0,
            poll_interval: float = 1.0,
            result_request_timeout: int = 30
    ) -> Any:
        return RemoteCollection._wait_job_result(
            self.base_url,
            job_id,
            timeout=timeout,
            poll_interval=poll_interval,
            result_request_timeout=result_request_timeout
        )

    def backup_async(self) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/api/admin/backup-jobs", timeout=10)
        return RemoteCollection._static_handle_response(resp)

    def download_backup(
            self,
            destination_path: Optional[str] = None,
            timeout: float = 300.0,
            poll_interval: float = 2.0
    ) -> str:
        job = self.backup_async()
        job_id = job["job_id"]
        self._wait_job_succeeded(job_id, timeout=timeout, poll_interval=poll_interval)

        resp = requests.get(f"{self.base_url}/api/jobs/{job_id}/result", stream=True, timeout=60)
        resp.raise_for_status()

        if destination_path is None:
            destination_path = self._filename_from_response(resp) or "vectordb_backup.zip"

        with open(destination_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return os.path.abspath(destination_path)

    def restore_backup_async(self, zip_file_path: str) -> Dict[str, Any]:
        with open(zip_file_path, "rb") as f:
            files = {"file": (os.path.basename(zip_file_path), f, "application/zip")}
            resp = requests.post(f"{self.base_url}/api/admin/restore-jobs", files=files, timeout=30)
        return RemoteCollection._static_handle_response(resp)

    def restore_backup(
            self,
            zip_file_path: str,
            timeout: float = 300.0,
            poll_interval: float = 2.0
    ) -> Dict[str, Any]:
        job = self.restore_backup_async(zip_file_path)
        return self.wait_job_result(job["job_id"], timeout=timeout, poll_interval=poll_interval)

    def _wait_job_succeeded(self, job_id: str, timeout: float, poll_interval: float) -> Dict[str, Any]:
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise VectorDBTimeoutError(f"Job {job_id} timed out after {elapsed:.2f}s")

            job = self.get_job(job_id)
            status = job.get("status")
            if status == "succeeded":
                return job
            if status == "failed":
                raise RuntimeError(f"Job {job_id} failed: {job.get('error')}")
            if status == "canceled":
                raise RuntimeError(f"Job {job_id} was canceled")

            remaining = timeout - (time.time() - start_time)
            time.sleep(min(poll_interval, max(0.0, remaining)))

    @staticmethod
    def _filename_from_response(resp: requests.Response) -> Optional[str]:
        content_disposition = resp.headers.get("Content-Disposition", "")
        marker = "filename="
        if marker not in content_disposition:
            return None
        filename = content_disposition.split(marker, 1)[1].split(";", 1)[0].strip()
        return filename.strip('"') or None

    # -------------------
    # Collection management
    # -------------------
    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def create_collection(self, name: str, chunk_size: int = 512, chunk_overlap: int = 50, **kwargs) -> "RemoteCollection":
        if not self._circuit_breaker.can_execute():
            raise ServerBusyError("Circuit breaker is OPEN, rejecting request")

        try:
            url = f"{self.base_url}/api/collections"
            payload = {"name": name, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            resp = requests.post(url, json=payload, timeout=5)

            if resp.status_code == 503:
                self._handle_503_response(resp)

            resp.raise_for_status()
            self._circuit_breaker.on_success()
            return RemoteCollection(self.base_url, name)

        except Exception:
            self._circuit_breaker.on_failure()
            raise

    def get_collection(self, name: str) -> "RemoteCollection":
        return RemoteCollection(self.base_url, name)

    def list_collections(self) -> List[str]:
        resp = requests.get(f"{self.base_url}/api/collections", timeout=10)
        resp.raise_for_status()
        return resp.json().get("collections", [])

    def _handle_503_response(self, resp: requests.Response):
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

    # -------------------
    # Analysis (new in service)
    # -------------------
    @retry_with_timeout(default_timeout=120.0, max_retries=10)
    def get_analysis_job(self, job_id: str, **kwargs) -> Dict[str, Any]:
        """
        GET /api/analysis/<job_id>
        """
        resp = requests.get(f"{self.base_url}/api/analysis/{job_id}", timeout=10)
        return RemoteCollection._static_handle_response(resp)

    # -------------------
    # Aggregation (new in service)
    # -------------------
    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def list_aggregation_plans(self, **kwargs) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/aggregation/plans", timeout=10)
        return RemoteCollection._static_handle_response(resp)

    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def register_aggregation_plan(self, plan: Dict[str, Any], overwrite: bool = False, **kwargs) -> Dict[str, Any]:
        """
        POST /api/aggregation/plans
        plan: dict containing required keys: plan_id, collection_name, ...
        """
        payload = dict(plan)
        payload["overwrite"] = bool(overwrite)
        resp = requests.post(f"{self.base_url}/api/aggregation/plans", json=payload, timeout=10)
        return RemoteCollection._static_handle_response(resp)

    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def delete_aggregation_plan(self, plan_id: str, **kwargs) -> Dict[str, Any]:
        resp = requests.delete(f"{self.base_url}/api/aggregation/plans/{plan_id}", timeout=10)
        return RemoteCollection._static_handle_response(resp)

    @retry_with_timeout(default_timeout=120.0, max_retries=10)
    def run_aggregation_plan(
        self,
        plan_id: str,
        overrides: Optional[Dict[str, Any]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        POST /api/aggregation/plans/<plan_id>/run
        Body:
          {
            "time_range": [start, end],   # optional
            "overrides": {...}            # optional
          }
        """
        payload: Dict[str, Any] = {"overrides": overrides or {}}
        if time_range is not None:
            payload["time_range"] = [float(time_range[0]), float(time_range[1])]
        resp = requests.post(f"{self.base_url}/api/aggregation/plans/{plan_id}/run", json=payload, timeout=10)
        return RemoteCollection._static_handle_response(resp)

    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def get_aggregation_job(self, job_id: str, **kwargs) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/aggregation/jobs/{job_id}", timeout=10)
        return RemoteCollection._static_handle_response(resp)

    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def get_aggregation_offline_latest(self, plan_id: str, **kwargs) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/aggregation/plans/{plan_id}/offline/latest", timeout=10)
        return RemoteCollection._static_handle_response(resp)

    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def get_aggregation_online_state(self, plan_id: str, **kwargs) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/aggregation/plans/{plan_id}/online/state", timeout=10)
        return RemoteCollection._static_handle_response(resp)

    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def get_aggregation_offline_cluster_items(self, plan_id: str, cluster_id: str, limit: int = 100, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/api/aggregation/plans/{plan_id}/offline/cluster/{cluster_id}/items"
        resp = requests.get(url, params={"limit": int(limit)}, timeout=20)
        return RemoteCollection._static_handle_response(resp)


class RemoteCollection:
    """
    Represents a specific collection on the remote VectorDB service.
    """

    def __init__(self, base_url: str, name: str):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/collections/{name}"
        self.name = name

    # -------------
    # Unified response handler (instance)
    # -------------
    def _handle_response(self, resp: requests.Response) -> Any:
        return self._static_handle_response(resp, collection_name=self.name)

    # -------------
    # Unified response handler (static; used by VectorDBClient too)
    # -------------
    @staticmethod
    def _static_handle_response(resp: requests.Response, collection_name: Optional[str] = None) -> Any:
        # 503 structured errors
        if resp.status_code == 503:
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", "Unknown")
                error_code = error_data.get("error_code")
            except Exception:
                error_msg = "Service Unavailable"
                error_code = None

            if error_code == "BUSY":
                raise ServerBusyError(f"Server busy: {error_msg}")
            elif error_code == "INIT":
                raise ServerInitializingError(f"Server initializing: {error_msg}")
            elif "initializing" in str(error_msg).lower():
                raise ServerInitializingError(f"Server initializing: {error_msg}")
            elif "queue" in str(error_msg).lower() or "busy" in str(error_msg).lower():
                raise ServerBusyError(f"Server busy: {error_msg}")
            else:
                raise RetryableError(f"Service Unavailable: {error_msg}")

        # 501 not configured / not enabled
        if resp.status_code == 501:
            try:
                j = resp.json()
                msg = j.get("error") or resp.text
            except Exception:
                msg = resp.text
            raise ServiceNotConfiguredError(f"Service feature not configured: {msg}")

        # 拦截 500 级别错误，把服务端的报错本体 (resp.text) 打印出来
        if resp.status_code >= 500 and resp.status_code not in (501, 503):
            error_body = resp.text
            raise NonRetryableError(f"Server Internal Error ({resp.status_code}): {error_body}")

        # Auth & request validation
        if resp.status_code == 401:
            raise AuthenticationError("Authentication failed")
        if resp.status_code == 400:
            raise InvalidRequestError(f"Invalid request: {resp.text}")
        if resp.status_code == 404:
            if collection_name:
                raise InvalidRequestError(f"Collection '{collection_name}' not found.")
            raise InvalidRequestError(f"Not found: {resp.text}")

        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()

    @staticmethod
    def _wait_job_result(
            base_url: str,
            job_id: str,
            timeout: float = 120.0,
            poll_interval: float = 1.0,
            result_request_timeout: int = 30
    ) -> Any:
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise VectorDBTimeoutError(f"Job {job_id} timed out after {elapsed:.2f}s")

            resp = requests.get(f"{base_url}/api/jobs/{job_id}", timeout=10)
            job = RemoteCollection._static_handle_response(resp)
            status = job.get("status")

            if status == "succeeded":
                result_resp = requests.get(
                    f"{base_url}/api/jobs/{job_id}/result",
                    timeout=result_request_timeout
                )
                return RemoteCollection._static_handle_response(result_resp)
            if status == "failed":
                raise RuntimeError(f"Job {job_id} failed: {job.get('error')}")
            if status == "canceled":
                raise RuntimeError(f"Job {job_id} was canceled")

            remaining = timeout - (time.time() - start_time)
            time.sleep(min(poll_interval, max(0.0, remaining)))

    # ----------------
    # Existing APIs
    # ----------------
    @retry_with_timeout(default_timeout=120.0)
    def upsert(self, doc_id: str, text: str, metadata: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if metadata is None:
            metadata = {}
        payload = {"doc_id": doc_id, "text": text, "metadata": metadata}
        resp = requests.post(f"{self.api_url}/upsert", json=payload, timeout=10)
        if resp.status_code == 202:
            return resp.json()
        return self._handle_response(resp)

    @retry_with_timeout(default_timeout=120.0)
    def upsert_batch(self, documents: List[Dict], **kwargs) -> Dict[str, Any]:
        resp = requests.post(f"{self.api_url}/upsert_batch", json=documents, timeout=20)
        if resp.status_code == 202:
            return resp.json()
        return self._handle_response(resp)

    def search(
            self,
            query: str,
            top_n: int = 5,
            score_threshold: float = 0.0,
            filter_criteria: Optional[Dict] = None,
            timeout: int = 30,
            wait: bool = True,
            poll_interval: float = 1.0,
            force_db_filter: bool = False,
            post_filter_multiplier: int = 10,
    ) -> List[Dict]:
        payload = {
            "query": query,
            "top_n": top_n,
            "score_threshold": score_threshold,
            "filter_criteria": filter_criteria,
            "force_db_filter": force_db_filter,
            "post_filter_multiplier": post_filter_multiplier,
        }
        resp = requests.post(f"{self.api_url}/search-jobs", json=payload, timeout=min(10, timeout))
        if resp.status_code == 202:
            job = resp.json()
            if not wait:
                return job
            return self._wait_job_result(
                self.base_url,
                job["job_id"],
                timeout=timeout,
                poll_interval=poll_interval,
                result_request_timeout=timeout
            )

        # Backward compatibility when talking to an older service instance.
        if resp.status_code == 404 and "search-jobs" in resp.request.url:
            resp = requests.post(f"{self.api_url}/search", json=payload, timeout=timeout)
        return self._handle_response(resp)

    def delete(self, doc_id: str) -> bool:
        resp = requests.delete(f"{self.api_url}/documents/{doc_id}", timeout=10)
        if resp.status_code == 404:
            return False
        res = self._handle_response(resp)
        return res.get("status") == "success"

    def stats(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.api_url}/stats", timeout=10)
        return self._handle_response(resp)

    def clear(self, timeout: int = 120, wait: bool = True, poll_interval: float = 1.0) -> bool:
        resp = requests.post(f"{self.api_url}/clear-jobs", timeout=min(10, timeout))
        if resp.status_code == 202:
            job = resp.json()
            if not wait:
                return job
            res = self._wait_job_result(
                self.base_url,
                job["job_id"],
                timeout=timeout,
                poll_interval=poll_interval,
                result_request_timeout=30
            )
            return res.get("status") == "cleared"

        # Backward compatibility when talking to an older service instance.
        if resp.status_code == 404 and "clear-jobs" in resp.request.url:
            resp = requests.post(f"{self.api_url}/clear", timeout=30)
        res = self._handle_response(resp)
        return res.get("status") == "cleared"

    def exists(self, doc_id: str, include_pending: bool = False, **kwargs) -> bool:
        return self._exists_impl(doc_id, include_pending=include_pending, **kwargs)

    def exists_batch(self, doc_ids: List[str], include_pending: bool = False, **kwargs) -> Dict[str, bool]:
        return self._exists_batch_impl(doc_ids, include_pending=include_pending, **kwargs)

    def exists_state(self, doc_id: str, **kwargs) -> str:
        return self._exists_state_impl(doc_id, **kwargs)

    @retry_with_timeout(default_timeout=30.0)
    def _exists_impl(self, doc_id: str, include_pending: bool = False, **kwargs) -> bool:
        url = f"{self.api_url}/documents/{doc_id}/exists"
        params = {"include_pending": "1" if include_pending else "0"}
        resp = requests.get(url, params=params, timeout=10)
        data = self._handle_response(resp)

        state = data.get("state")
        if state:
            return (state in ("present", "pending")) if include_pending else (state == "present")

        return bool(data.get("exists", False))

    @retry_with_timeout(default_timeout=30.0)
    def _exists_batch_impl(self, doc_ids: List[str], include_pending: bool = False, **kwargs) -> Dict[str, bool]:
        url = f"{self.api_url}/exists"
        payload = {"doc_ids": doc_ids, "include_pending": bool(include_pending)}
        resp = requests.post(url, json=payload, timeout=20)
        data = self._handle_response(resp)

        if "exists_map" in data:
            return data.get("exists_map", {}) or {}

        states_map = data.get("states_map", {}) or {}
        if include_pending:
            return {d: (states_map.get(d) in ("present", "pending")) for d in doc_ids}
        return {d: (states_map.get(d) == "present") for d in doc_ids}

    @retry_with_timeout(default_timeout=30.0)
    def _exists_state_impl(self, doc_id: str, **kwargs) -> str:
        url = f"{self.api_url}/documents/{doc_id}/exists"
        resp = requests.get(url, params={"include_pending": "0"}, timeout=10)
        data = self._handle_response(resp)

        if "state" in data:
            return data["state"]
        return "present" if data.get("exists", False) else "missing"

    # ----------------
    # NEW: timestamp_stats
    # ----------------
    def timestamp_stats(
            self,
            time_field: str = "timestamp",
            scan_limit: int = 20000,
            timeout: int = 120,
            wait: bool = True,
            poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """
        GET /api/collections/<name>/timestamp_stats?time_field=...&scan_limit=...
        """
        payload = {"time_field": time_field, "scan_limit": int(scan_limit)}
        resp = requests.post(f"{self.api_url}/timestamp_stats-jobs", json=payload, timeout=min(10, timeout))
        if resp.status_code == 202:
            job = resp.json()
            if not wait:
                return job
            return self._wait_job_result(
                self.base_url,
                job["job_id"],
                timeout=timeout,
                poll_interval=poll_interval,
                result_request_timeout=30
            )

        # Backward compatibility when talking to an older service instance.
        if resp.status_code == 404 and "timestamp_stats-jobs" in resp.request.url:
            params = {"time_field": time_field, "scan_limit": int(scan_limit)}
            resp = requests.get(f"{self.api_url}/timestamp_stats", params=params, timeout=20)
        return self._handle_response(resp)

    # ----------------
    # NEW: analysis trigger (per collection)
    # ----------------
    @retry_with_timeout(default_timeout=60.0, max_retries=10)
    def trigger_analysis(self, config_payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        POST /api/collections/<name>/analysis
        Returns: {status:'accepted', job_id:'...'}
        """
        resp = requests.post(f"{self.api_url}/analysis", json=config_payload, timeout=20)
        return self._handle_response(resp)
