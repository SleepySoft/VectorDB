# VectorDB/memory_utils.py

import os
import gc
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_rss_mb() -> float:
    try:
        import psutil
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / 1024 / 1024
    except Exception:
        return -1.0


def cleanup_memory(tag: str = "", aggressive: bool = False):
    """
    清理 Python / PyTorch / sklearn / numpy 相关临时内存。

    注意：
    1. gc.collect() 只能回收没有引用的 Python 对象。
    2. PyTorch CPU allocator / Python allocator 不一定立刻把 RSS 还给 OS。
    3. CUDA 显存需要 torch.cuda.empty_cache()。
    """
    before = get_rss_mb()

    # 1. Python GC
    gc.collect()

    # 2. PyTorch cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    # 3. malloc trim: Linux 下尝试把 free heap 还给 OS
    if aggressive and os.name == "posix":
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

    after = get_rss_mb()
    logger.info(
        f"[MEM_CLEANUP] {tag} RSS before={before:.1f}MB after={after:.1f}MB aggressive={aggressive}"
    )


@contextmanager
def memory_scope(name: str, aggressive_cleanup: bool = False):
    before = get_rss_mb()
    t0 = time.time()
    logger.info(f"[MEM_SCOPE] enter {name}, RSS={before:.1f}MB")

    try:
        yield
    finally:
        cleanup_memory(tag=name, aggressive=aggressive_cleanup)
        after = get_rss_mb()
        logger.info(
            f"[MEM_SCOPE] exit {name}, RSS={after:.1f}MB, elapsed={time.time() - t0:.2f}s"
        )
