# test_aggregation.py
import time
import json
import requests

BASE = "http://127.0.0.1:8001"
PLAN_ID = "agg_intelligence_summary_24h"
COLLECTION = "intelligence_summary"
TIME_FIELD = "timestamp"


def wait_ready(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE}/api/status", timeout=3)
            j = r.json()
            if j.get("status") == "ready":
                print("[OK] Engine READY")
                return True
            print("[WAIT]", j)
        except Exception as e:
            print("[WAIT] status error:", e)
        time.sleep(2)
    return False


def ensure_collection():
    payload = {"name": COLLECTION, "chunk_size": 256, "chunk_overlap": 30}
    r = requests.post(f"{BASE}/api/collections", json=payload, timeout=10)
    print("[COLLECTION]", r.status_code, r.text)


def get_ts_stats():
    r = requests.get(
        f"{BASE}/api/collections/{COLLECTION}/timestamp_stats",
        params={"time_field": TIME_FIELD, "scan_limit": 20000},
        timeout=10
    )
    j = r.json()
    print("[TS_STATS]", j)
    return j


def wait_queue_empty(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{BASE}/api/status/queue", timeout=5).json()
        if r.get("qsize", 0) == 0:
            return True
        time.sleep(1)
    return False


# def get_ts_stats():
#     url = f"{BASE}/api/collections/{COLLECTION}/timestamp_stats"
#     r = requests.get(url, params={"time_field": TIME_FIELD, "scan_limit": 20000}, timeout=10)
#
#     print("[TS_STATS_RAW]", r.status_code, r.headers.get("content-type"))
#     print(r.text[:500])  # 打印前 500 字符，看看是不是 404/HTML/异常栈
#
#     # 如果不是 2xx，直接抛错
#     r.raise_for_status()
#
#     # 再尝试解析 JSON
#     return r.json()


def insert_dummy_docs(n=6):
    now = time.time()
    docs = []
    for i in range(n):
        docs.append({
            "doc_id": f"test_news_{int(now)}_{i}",
            "text": f"测试新闻 {i}：上海今日天气晴朗。事件{i}相关细节...",
            "metadata": {TIME_FIELD: now - i * 60, "source": "unit_test"}
        })
    r = requests.post(f"{BASE}/api/collections/{COLLECTION}/upsert_batch", json=docs, timeout=10)
    print("[UPSERT_BATCH]", r.status_code, r.text)

    # 等待异步写入完成（简单 sleep；你也可以增加队列监控）
    time.sleep(5)


def trigger_run(time_range):
    payload = {
        "time_range": [time_range[0], time_range[1]],
        "overrides": {
            # 这里还可覆盖 method/params/max_points 等
            "time_field": TIME_FIELD
        }
    }
    r = requests.post(f"{BASE}/api/aggregation/plans/{PLAN_ID}/run", json=payload, timeout=10)
    print("[RUN]", r.status_code, r.text)
    r.raise_for_status()
    return r.json()["job_id"]


def poll_job(job_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{BASE}/api/aggregation/jobs/{job_id}", timeout=10)
        j = r.json()
        status = j.get("status")
        print("[JOB]", status)
        if status in ("completed", "failed"):
            return j
        time.sleep(2)
    raise TimeoutError("Job polling timeout")


def main():
    if not wait_ready():
        raise RuntimeError("Service not ready")

    ensure_collection()

    stats = get_ts_stats()
    max_ts = stats.get("max_ts")
    if max_ts is None:
        print("[INFO] No timestamp found, inserting dummy docs...")
        insert_dummy_docs()
        wait_queue_empty()
        stats = get_ts_stats()
        max_ts = stats.get("max_ts")

    if max_ts is None:
        # still none -> fallback to now window
        end = time.time()
        start = end - 24 * 3600
    else:
        end = float(max_ts)
        start = end - 24 * 3600

    print("[TIME_RANGE]", start, end)
    job_id = trigger_run((start, end))
    result = poll_job(job_id)

    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()