# VectorDB 网页查询超时 —— 全链路分析与优化建议

> 环境信息：VectorDB 约 20GB、20万+ 文档；网页查询经常超时。
> 分析时间：2026-06-23
> 基于代码版本：当前工作目录 HEAD

---

## 1. 搜索全链路

```text
浏览器 /intelligences/search
    │
    ▼
Flask: IntelligenceHubWebService.py
    /intelligences/query  (GET/POST)
        │
        ├── _get_combined_params()          # 参数解析
        ├── _apply_public_search_limits()   # 游客限制（仅未登录）
        └── _perform_search_logic()
                │
                ├── mongo 模式 → MongoDB 查询（不走 VectorDB）
                └── vector_text / vector_similar 模式
                        │
                        ▼
                IntelligenceHub.vector_search_intelligence()
                        │
                        ├── 快照 engine_summary / engine_full
                        ├── IntelligenceVectorDBEngine.query(summary)
                        └── IntelligenceVectorDBEngine.query(fulltext)
                                │
                                ▼
                        RemoteCollection.search()  (VectorDBClient.py)
                                │
                                ├── POST /api/collections/{name}/search-jobs  → 202 Accepted
                                └── 轮询 GET /api/jobs/{job_id}/result
                                        │
                                        ▼
                        VectorDBBService.submit_search_job()
                                │
                                ├── 获取 _search_semaphore 槽位
                                ├── AsyncJobManager.submit("collection.search", run_search)
                                └── run_search() 在线程池执行
                                        │
                                        ▼
                                VectorCollectionRepo.search()
                                        │
                                        ├── _cached_query_vector(query_text)  # 带 LRU 缓存
                                        ├── _collection.query(...)            # ChromaDB HNSW 搜索
                                        └── 去重/排序/截断
```

---

## 2. 关键问题答疑

### 2.1 `ef_search=10` 是什么意思？

`ef_search` 是 HNSW（Hierarchical Navigable Small World）索引的搜索参数：

- 它表示**搜索时动态维护的候选池大小**。
- 值越小，HNSW 在图里只考察越少邻居，**速度快但召回率低**；值越大，考察越多邻居，**召回率高但速度慢**。
- 对 20万+ 向量，`ef_search=10` 的候选池太小，图遍历容易"迷路"，反而要做更多无效跳转，经常出现**又慢又找不全**的情况。

实验数据（ChromaDB 1.2.1，10万条 128 维随机向量）：

| ef_search | 单次查询耗时 | 是否找回真实最近邻 |
|-----------|--------------|--------------------|
| 10        | ~55 ms       | 是（但只是巧合）   |
| 50        | ~3 ms        | 是                 |
| 100       | ~3 ms        | 是                 |
| 200       | ~3 ms        | 是                 |

> 注意：`ef_search=10` 并不是"查询 10 个结果"，而是内部只保留 10 个候选。它和你请求的 `top_n` 是两个概念。

**建议**：20万~100万量级设置为 **100~200**。ChromaDB 1.2.1 中 `ef_search` 不能在创建 collection 时通过 metadata 传入，但可以通过 `collection.modify(metadata={"hnsw:ef_search": 100})` 对已存在 collection 生效。

### 2.2 "每次查询重复 encode query" 能优化吗？每次查询文本不是不一样吗？

你说得对，不同用户的查询文本确实可能不同，但**重复查询非常普遍**：

1. **分页翻页**：同一关键词查第 1 页、第 2 页会反复 encode 同一个 query。
2. **相似推荐**：`vector_similar` 模式根据同一篇 reference 文章反复推荐。
3. **热门关键词**："俄乌冲突"、"美国大选"、"中国经济" 等会被大量用户反复搜索。
4. **浏览器回退/刷新**：用户回退或 F5 会再次发送相同请求。
5. **空/默认查询**：首页或测试请求经常相同。

实测（本地 `bge-m3`，CPU）：

| 场景 | 时间 |
|------|------|
| 单次 encode | ~23 ms |
| 500 次不同文本 encode | ~21 s |
| 500 次重复查询（LRU 缓存后） | ~0.16 s |

所以 **LRU 缓存不能解决所有查询，但能显著降低热门/重复查询的延迟**。更彻底的方案是把 embedding 服务独立出来用 GPU batch，但缓存是成本低、效果快的优化。

### 2.3 带 `archive_period`/`event_period` 时间范围的查询会在 ChromaDB 的 SQLite 元数据层做全表扫描，这是设计问题吗？当时加时间字段就是为了不全表搜索。

**不是设计错误，但效果没有预期好。**

我检查了 ChromaDB 1.2.1 的源码和实际数据库：

- `embedding_metadata` 表上有索引：
  ```sql
  CREATE INDEX embedding_metadata_int_value ON embedding_metadata (key, int_value) WHERE int_value IS NOT NULL;
  CREATE INDEX embedding_metadata_float_value ON embedding_metadata (key, float_value) WHERE float_value IS NOT NULL;
  CREATE INDEX embedding_metadata_string_value ON embedding_metadata (key, string_value) WHERE string_value IS NOT NULL;
  ```
- `EXPLAIN QUERY PLAN` 显示时间范围查询确实会走 `embedding_metadata_int_value` 索引，**不是全表扫描**。

但实验结果（10万条向量，ef_search=100）：

| 过滤方式 | 耗时 |
|----------|------|
| 无过滤 | ~7 ms |
| 窄时间范围（1000 条） | ~78 ms |
| 宽时间范围（8万条） | ~346 ms |
| **先召回 1000 个再内存过滤** | **~20 ms** |

结论：
- ChromaDB 的 `where` 时间过滤确实走了索引，但它的实现方式是**先扫 metadata 得到候选 id 集合，再和向量搜索结果做 join/过滤**，这个额外开销在 20万+ 数据下很大。
- 你加时间字段的思路是对的，但 ChromaDB 的 metadata 索引机制对**范围查询**支持不够高效。
- **更快的做法**：让 HNSW 只做向量最近邻（`where=None`），召回足够多的候选后在内存里按时间戳过滤。这样通常快 10~50 倍，且召回质量一致（实验验证 top10 完全一致）。

---

## 3. 各环节瓶颈分析

### 3.1 前端 / Web 服务层

| 位置 | 代码 | 风险点 |
|------|------|--------|
| 最大召回 | `VECTOR_MAX_TOP_N = 50` | 太小，分页到第 2 页可能不够用；游客 `vector_max_top_n=20` 更小 |
| 分页逻辑 | `_do_vector_search` 先取 `top_n = page*per_page` 再内存分页 | 每翻一页都要重新做一次完整向量搜索，无法利用上一页结果 |
| 相似推荐 | `vector_similar` 模式要先 `get_intelligence(reference)` | 如果 MongoDB 慢或 UUID 不存在，会额外增加 RTT |
| 并发限制 | `_vector_search_concurrency` 仅对未登录生效 | 登录用户无并发保护，可能被慢查询占满 Flask worker |

### 3.2 Hub / 业务层

| 位置 | 代码 | 风险点 |
|------|------|--------|
| 双库查询 | `IntelligenceHub.vector_search_intelligence()` | 同时查 summary + fulltext，耗时叠加；游客 `_effective_top_n` 被压到 20 |
| 时间过滤 | `IntelligenceVectorDBEngine.query()` 把 datetime 转成 int/float 后做 `$gte/$lte` | ChromaDB metadata 范围过滤开销大 |
| 无缓存 | 每次请求都重新走完整链路 | 重复查询浪费资源 |

### 3.3 HTTP 客户端层

| 位置 | 代码 | 风险点 |
|------|------|--------|
| Job 模式 | `RemoteCollection.search()` 默认 `wait=True` | 提交 job(1 RTT) + 轮询 job(多次 RTT) |
| 超时 | `timeout=min(10, timeout)` 提交，`result_request_timeout=timeout` | 轮询结果时可能超时 |
| 重试 | `retry_with_timeout` 默认 `max_retries=-1` | 一旦服务端 503/connection 错误会无限重试到总超时 |

### 3.4 VectorDB 服务层

| 位置 | 代码 | 风险点 |
|------|------|--------|
| WSGI | `VectorDBService.run_standalone()` 使用 `app.run()` | Flask dev server，**不适合生产高并发** |
| 并发槽 | `_max_concurrent_searches = 8` | 慢查询容易占满，新请求直接 503 |
| 线程池 | `AsyncJobManager(max_workers=8)` | 与搜索槽位数量相同，慢查询会占满整个线程池 |
| 信号量实现 | `_acquire_search_slot(timeout=0)` | 不等待直接拒绝 |
| HNSW 参数 | 只设了 `hnsw:space` | 默认 `ef_search=10` 对 20万+ 数据过低 |
| 模型编码 | `_vectorize(texts, batch_size=32)` | 每次查询都重新 encode |
| ChromaDB PersistentClient | 底层 SQLite | 高并发读写共享一个 SQLite 文件 |
| 20GB 数据 | 全部加载到一个 PersistentClient | 没有分 collection/分片/分区 |

---

## 4. 已做的代码优化

| 文件 | 改动 |
|------|------|
| `VectorDB/VectorStorageEngine.py` | ① `VectorCollectionRepo` 支持 `hnsw_config`，启动时通过 `collection.modify()` 应用 `ef_search`/`ef_construction`；② `search()` 对纯时间范围过滤自动转**内存后过滤**，可配置 `post_filter_multiplier`；③ query embedding 加 `lru_cache`；④ `VectorStorageEngine` 透传 `hnsw_config`。 |
| `VectorDB/VectorDBBService.py` | ① `run_standalone()` 优先使用 Waitress，退化为 Flask dev server；② 新增 HNSW 启动参数 `--hnsw-m/--hnsw-ef-construction/--hnsw-ef-search/--hnsw-num-threads`；③ `/search` 和 `/search-jobs` 接口透传 `force_db_filter` 和 `post_filter_multiplier`。 |
| `VectorDB/VectorDBClient.py` | `RemoteCollection.search()` 透传 `force_db_filter` 和 `post_filter_multiplier`。 |
| `ServiceComponent/IntelligenceVectorDBEngine.py` | ① 增加业务层 LRU 缓存；② 透传 `force_db_filter`/`post_filter_multiplier`。 |

---

## 5. 优化建议（按优先级排序）

### P0 — 立即生效

1. **确认性能日志有输出**
   ```bash
   set VECTOR_PERF_LOG=C:\IIS\_log\vector_perf.log
   ```
   复现一次慢查询，看 `vectordb_async_search` 耗时。

2. **调整 HNSW 参数**
   - ChromaDB 1.2.1 中启动时只能传 `M`、`num_threads` 等；`ef_search`/`ef_construction` 已通过 `collection.modify()` 自动应用。
   - 启动命令示例：
     ```bash
     python VectorDB/VectorDBBService.py ^
       --db-path C:\IIS\_data\VectorDB ^
       --model C:\Models\bge-m3 ^
       --hnsw-m 32 ^
       --hnsw-ef-search 100 ^
       --hnsw-num-threads 8
     ```

3. **启用内存时间过滤**
   - 现在默认行为已经会把纯时间范围过滤改到内存里做，无需改调用方。
   - 如果结果数量不够（例如时间窗口极窄），可以调大 `post_filter_multiplier` 或设置 `force_db_filter=true` 回退到 DB 过滤。

4. **切到 Waitress**
   - 代码已自动优先使用 Waitress，确保生产环境已安装 `waitress`（已在 `requirements.txt`）。

5. **增大并发**
   ```bash
   set VECTOR_MAX_CONCURRENT_SEARCHES=16
   set VECTOR_JOB_MAX_WORKERS=16
   set VECTOR_WAITRESS_THREADS=64
   set VECTOR_WAITRESS_CONNECTION_LIMIT=256
   ```

### P1 — 中期优化

6. **独立 embedding 服务或预计算**
7. **分页优化**：使用 `search_after` 风格避免重复 encode
8. **监控与告警**：对 `vectordb_async_search` P99 设置告警

### P2 — 长期架构

9. **评估替换 ChromaDB**：20GB / 20万+ 文档接近 ChromaDB PersistentClient 舒适区，可考虑 Milvus/Qdrant。
10. **按时间分 collection**：如 `intelligence_summary_2025`、`intelligence_summary_2026`。

---

## 6. 关键指标验证清单

| 检查项 | 命令/位置 | 预期 |
|--------|-----------|------|
| VectorDB 单次搜索耗时 | `logs/vector_perf.log` 中 `vectordb_async_search` | 应 < 500ms |
| embedding 耗时 | 单独计时 | 应 < 50ms（重复查询命中缓存 ≈ 0） |
| 并发压力 | `/api/status/memory` 的 `jobs.search` | active 不应长期接近上限 |
| 队列深度 | `/api/status/queue` | qsize 不应长期 > 50 |
| 内存使用 | 系统监控 | 加载 20GB 索引 + 模型，应 > 32GB，避免 swap |
| 磁盘 IO | iostat/资源管理器 | 不应长期高 IO wait |

---

## 7. 结论

当前超时不是单一原因，而是 **"Flask dev server + HNSW 默认 ef_search=10 + 每次重复 encode + ChromaDB metadata 时间范围过滤开销大 + 并发槽位过少 + Job 轮询"** 叠加导致的。

最应该先做的三件事：
1. 部署最新代码，切到 Waitress，确认 `ef_search` 已生效；
2. 用 `VECTOR_PERF_LOG` 复现并确认 `vectordb_async_search` 耗时；
3. 观察内存过滤后的查询延迟，必要时调大 `post_filter_multiplier`。
