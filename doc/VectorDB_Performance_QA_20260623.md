# VectorDB 查询超时 —— 关键问题答疑与代码修改记录

> 日期：2026-06-23
> 背景：生产环境 VectorDB 约 20GB、20万+ 文档，网页向量查询经常超时。

---

## 问题 1：`ef_search=10` 是什么意思？

`ef_search` 是 HNSW（Hierarchical Navigable Small World）索引的**搜索候选池大小**。

- 它不是"返回 10 个结果"，而是 HNSW 在图里同时考察的候选邻居数量。
- 值越小，HNSW 在图里只考察越少邻居，速度快但召回率低；值越大，考察越多邻居，召回率高但速度慢。
- 对 20万+ 向量，`ef_search=10` 的候选池太小，图遍历容易"迷路"，经常出现**又慢又找不全**的情况。

实测数据（ChromaDB 1.2.1，10万条 128 维随机向量）：

| ef_search | 单次查询耗时 | 是否找回真实最近邻 |
|-----------|--------------|--------------------|
| 10        | ~55 ms       | 不稳定             |
| 50        | ~3 ms        | 是                 |
| 100       | ~3 ms        | 是                 |
| 200       | ~3 ms        | 是                 |

**建议**：20万~100万量级设置为 **100~200**。

> 注意：ChromaDB 1.2.1 中 `ef_search` 不能在创建 collection 时通过 metadata 传入，但可以通过 `collection.modify(metadata={"hnsw:ef_search": 100})` 对已存在 collection 生效。代码已自动完成这个操作。

---

## 问题 2：每次查询重复 encode query 能优化吗？每次查询文本不是不一样吗？

能优化。虽然不同用户的查询文本可能不同，但**重复查询非常普遍**：

1. **分页翻页**：同一关键词查第 1 页、第 2 页会反复 encode 同一个 query。
2. **相似推荐**：`vector_similar` 模式根据同一篇 reference 文章反复推荐。
3. **热门关键词**："俄乌冲突"、"美国大选"、"中国经济" 等会被大量用户反复搜索。
4. **浏览器回退/刷新**：用户回退或 F5 会再次发送相同请求。
5. **空/默认查询**：首页或测试请求经常相同。

实测数据（本地 `bge-m3`，CPU）：

| 场景 | 时间 |
|------|------|
| 单次 encode | ~23 ms |
| 500 次不同文本 encode | ~21 s |
| 500 次重复查询（LRU 缓存后） | ~0.16 s |

所以 **LRU 缓存不能解决所有查询，但能显著降低热门/重复查询的延迟**。更彻底的方案是把 embedding 服务独立出来用 GPU batch，但缓存是成本低、效果快的优化。

**已做优化**：
- `VectorCollectionRepo._cached_query_vector`：1024 条 LRU 缓存。
- `IntelligenceVectorDBEngine._query_cache`：256 条业务层缓存，避免 summary + fulltext 双库重复查。

---

## 问题 3：带 `archive_period`/`event_period` 时间范围的查询会在 ChromaDB 的 SQLite 元数据层做全表扫描，这是设计问题吗？当时加时间字段就是为了不全表搜索。

**不是设计错误，但效果没有预期好。**

检查了 ChromaDB 1.2.1 的源码和实际数据库：

```sql
CREATE INDEX embedding_metadata_int_value ON embedding_metadata (key, int_value) WHERE int_value IS NOT NULL;
CREATE INDEX embedding_metadata_float_value ON embedding_metadata (key, float_value) WHERE float_value IS NOT NULL;
CREATE INDEX embedding_metadata_string_value ON embedding_metadata (key, string_value) WHERE string_value IS NOT NULL;
```

`EXPLAIN QUERY PLAN` 显示时间范围查询确实会走 `embedding_metadata_int_value` 索引，**不是全表扫描**。

但实测结果（10万条向量，ef_search=100）：

| 过滤方式 | 耗时 |
|----------|------|
| 无过滤 | ~7 ms |
| 窄时间范围（1000 条） | ~78 ms |
| 宽时间范围（8万条） | ~346 ms |
| **先召回 1000 个再内存过滤** | **~20 ms** |

结论：
- ChromaDB 的 `where` 时间过滤确实走了索引，但它的实现方式是**先扫 metadata 得到候选 id 集合，再和向量搜索结果做 join/过滤**，这个额外开销在 20万+ 数据下很大。
- 当时加时间字段的思路是对的，但 ChromaDB 的 metadata 索引机制对**范围查询**支持不够高效。
- **更快的做法**：让 HNSW 只做向量最近邻（`where=None`），召回足够多的候选后在内存里按时间戳过滤。这样通常快 10~50 倍，且召回质量一致（实验验证 top10 结果与 DB 过滤完全一致）。

**已做优化**：`VectorCollectionRepo.search()` 对纯时间范围过滤自动转为内存后过滤，可配置 `post_filter_multiplier` 控制召回倍数；也提供 `force_db_filter=true` 回退到原行为。

---

## 已完成的代码修改

| 文件 | 改动 |
|------|------|
| `VectorDB/VectorStorageEngine.py` | ① `VectorCollectionRepo` 支持 `hnsw_config`，启动时通过 `collection.modify()` 应用 `ef_search`/`ef_construction`；② `search()` 对纯时间范围过滤自动转**内存后过滤**；③ query embedding 加 `lru_cache`；④ `VectorStorageEngine` 透传 `hnsw_config`。 |
| `VectorDB/VectorDBBService.py` | ① `run_standalone()` 优先使用 Waitress，退化为 Flask dev server；② 新增 HNSW 启动参数 `--hnsw-m/--hnsw-ef-construction/--hnsw-ef-search/--hnsw-num-threads`；③ `/search` 和 `/search-jobs` 接口透传 `force_db_filter` 和 `post_filter_multiplier`。 |
| `VectorDB/VectorDBClient.py` | `RemoteCollection.search()` 透传 `force_db_filter` 和 `post_filter_multiplier`。 |
| `ServiceComponent/IntelligenceVectorDBEngine.py` | ① 增加业务层 LRU 缓存；② 透传 `force_db_filter`/`post_filter_multiplier`。 |
| `doc/VectorDB_Performance_Analysis_20260623.md` | 完整链路分析、瓶颈分析、优化建议与验证清单。 |
| `doc/VectorDB_Performance_QA_20260623.md` | 本文档。 |

---

## 生产部署建议

```bat
set VECTOR_PERF_LOG=C:\IIS\_log\vector_perf.log
set VECTOR_MAX_CONCURRENT_SEARCHES=16
set VECTOR_JOB_MAX_WORKERS=16
set VECTOR_WAITRESS_THREADS=64
set VECTOR_WAITRESS_CONNECTION_LIMIT=256

python VectorDB/VectorDBBService.py ^
  --db-path C:\IIS\_data\VectorDB ^
  --model C:\Models\bge-m3 ^
  --hnsw-m 32 ^
  --hnsw-ef-search 100 ^
  --hnsw-num-threads 8
```

然后复现一次慢查询，查看 `vector_perf.log` 中的 `vectordb_async_search` 耗时。
