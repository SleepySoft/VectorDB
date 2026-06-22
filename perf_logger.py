# VectorDB/perf_logger.py
"""
VectorDB 服务自包含的性能日志模块。

不依赖项目其他目录（如 Tools），可独立运行。
默认把结构化 JSON 性能日志写入文件，同时在控制台打印简化行。
"""

import os
import sys
import time
import logging
import threading
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


DEFAULT_LOGGER_NAME = "VectorDB.Performance"


def _default_log_file() -> str:
    """默认日志文件：优先使用环境变量 VECTOR_PERF_LOG，否则放在包根目录的 logs/ 下。"""
    env_path = os.getenv("VECTOR_PERF_LOG")
    if env_path:
        return env_path
    # 包根目录 = VectorDB 的父目录
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(package_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "vector_perf.log")


def _snapshot_system() -> Dict[str, Any]:
    """轻量级进程指标快照。"""
    out = {
        "cpu_percent": None,
        "rss_mb": None,
        "vms_mb": None,
        "thread_count": threading.active_count(),
        "source": None,
    }

    try:
        import psutil

        proc = psutil.Process(os.getpid())
        out["cpu_percent"] = round(proc.cpu_percent(interval=None), 2)
        mem = proc.memory_info()
        out["rss_mb"] = round(mem.rss / 1024 / 1024, 2)
        out["vms_mb"] = round(mem.vms / 1024 / 1024, 2)
        out["source"] = "psutil"
        return out
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(ProcessMemoryCountersEx)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                out["rss_mb"] = round(counters.WorkingSetSize / 1024 / 1024, 2)
                out["vms_mb"] = round(counters.PrivateUsage / 1024 / 1024, 2)
                out["source"] = "windows_psapi"
        except Exception:
            pass

    return out


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


class _ConsoleFormatter(logging.Formatter):
    """把 dict 性能日志简化成一行控制台输出。"""

    def format(self, record):
        payload = record.msg if isinstance(record.msg, dict) else {"message": record.getMessage()}
        ts = self.formatTime(record, datefmt="%H:%M:%S")
        op = payload.get("operation", "-")
        status = payload.get("status", "-")
        elapsed = payload.get("elapsed_ms")
        extra_parts = []
        for k in ("collection", "top_n", "doc_id", "queue_size", "result_count", "error"):
            v = payload.get(k)
            if v is not None:
                extra_parts.append(f"{k}={v}")
        extra = " ".join(extra_parts)
        if elapsed is not None:
            return f"[VPERF] {ts} {op} {elapsed:.2f}ms [{status}] {extra}"
        return f"[VPERF] {ts} {op} [{status}] {extra}"


_setup_done = False
_setup_lock = threading.Lock()


def setup_vector_perf_logging(
    log_file: Optional[str] = None,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 3,
    level: int = logging.INFO,
) -> logging.Logger:
    """初始化 VectorDB 性能日志 logger。幂等。"""
    global _setup_done

    log_file = log_file or _default_log_file()
    log_file = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    with _setup_lock:
        if _setup_done:
            return logger

        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        try:
            from pythonjsonlogger import jsonlogger

            file_formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        except Exception:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_ConsoleFormatter())
        logger.addHandler(console_handler)

        _setup_done = True

    return logger


class VectorPerformanceLogger:
    """VectorDB 性能日志记录器。"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(DEFAULT_LOGGER_NAME)

    def record(
        self,
        operation: str,
        *,
        status: str = "ok",
        elapsed_ms: Optional[float] = None,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        payload = {"operation": operation, "status": status}
        if elapsed_ms is not None:
            payload["elapsed_ms"] = round(elapsed_ms, 2)
        if error is not None:
            payload["error"] = str(error)
        if extra:
            for k, v in extra.items():
                payload[k] = _safe_json_value(v)

        payload.update(_snapshot_system())

        if status == "error" or error is not None:
            self.logger.warning(payload)
        else:
            self.logger.info(payload)

    def timed(self, operation: str, **ctx):
        return _TimedContext(self, operation, ctx)


class _TimedContext:
    def __init__(self, perf: VectorPerformanceLogger, operation: str, ctx: Dict[str, Any]):
        self.perf = perf
        self.operation = operation
        self.ctx = ctx
        self.start = None
        self.status = "ok"
        self.error = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start) * 1000 if self.start else None
        if exc_val is not None:
            self.status = "error"
            self.error = f"{exc_type.__name__}: {exc_val}"
        self.perf.record(
            self.operation,
            status=self.status,
            elapsed_ms=elapsed_ms,
            error=self.error,
            extra=self.ctx,
        )
        return False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def get_vector_performance_logger() -> VectorPerformanceLogger:
    return VectorPerformanceLogger()
