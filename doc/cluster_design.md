***

# 设计文档：基于 Collection 的新闻聚合（离线重聚合 + 在线微簇）

## 1. 背景与目标

系统当前提供：

*   `VectorStorageEngine`：管理 ChromaDB PersistentClient、SentenceTransformer 模型、异步 upsert 队列与 Repo 工厂。
*   `VectorCollectionRepo`：对指定 collection 做文档切分、向量化（encode）、CRUD、向量检索。
*   `VectorDBService`：对外提供 Flask API，含异步 analysis pipeline（已有指定簇数量的聚类能力）。

新增目标：

1.  **离线聚合（Offline clustering）**：对指定 collection 在指定时间窗口内做“自动聚类”（无需指定簇数量 K），产出“事件簇/去重列表”。
2.  **在线聚合（Online micro-clustering）**：实时消费新入库数据，在“最近离线版本”基础上持续增量归类，提供“实时不重复列表”。
3.  **计划化注册**：聚合不是对所有 collection 默认开启，而是先注册/配置“聚合计划”（Plan），并在程序内做限制、防重、资源控制。
4.  **最小侵入**：在线聚合尽可能复用 upsert 时已经计算出的 embeddings，避免重复 encode；同时不引入反向耦合（engine 不 import 在线模块）。

> 约束说明：Chroma 单机集合默认使用 HNSW 近邻索引，collection 的 `space` 参数定义距离度量（`l2/cosine/ip`），其中 cosine 距离定义为 `1 - cosine_similarity`。这些事实决定了我们统一的阈值/距离语义与检索方式。 [\[petkir.at\]](https://www.petkir.at/blog/semantic-kernel/02_sindex_02_vector-search-basics), [\[arxiv.org\]](https://arxiv.org/abs/2407.08623), [\[linkedin.com\]](https://www.linkedin.com/pulse/rethinking-similarity-how-diem-outperforms-cosine-high-berkowitz-oklwe)

***

## 2. 需求（Requirements）

### 2.1 功能需求（Functional）

**R1. Plan 注册与管理**

*   必须能注册聚合计划：至少包含 `plan_id`、`collection_name`、`time_window`、`run_every`、`method`、`params`、`max_points`、`enable_online`、`persist`。
*   必须防重：同 `plan_id` 不可重复注册；可限制同一 `collection_name` 的 plan 数量（默认 1）。
*   必须可查询：列出已注册 plan、查看 plan 的最新运行版本、状态。

**R2. 离线聚合（自动簇数）**

*   支持方法：
    *   HDBSCAN / DBSCAN（密度聚类，自动簇数、支持噪声点 label=-1）
    *   Agglomerative（层次聚类） + 距离阈值（`n_clusters=None, distance_threshold=...`）
*   输入：来自指定 collection 的数据（按 where/time\_range/limit）。
*   输出：cluster\_id 列表、每簇成员 doc\_id、代表项（可选）、噪声集合。
*   结果持久化：写入 `cluster_meta`、`doc_cluster_map`、（可选）`cluster_centroids`。

**R3. 在线微簇**

*   必须能实时消费“某个 plan 对应 collection 的 upsert 成功事件”。
*   优先复用已计算 embeddings，避免重复向量化。
*   在线归类使用最近离线版本作为基准，并支持离线重跑后的对齐（reconcile）。
*   结果可被 API 查询（实时事件列表/簇详情）。

**R4. 资源控制与稳定性**

*   必须限制每次离线聚合的最大点数 `max_points`，以及并发执行数（避免内存爆）。
*   必须提供运行锁：同一个 plan 同一时刻只跑一个离线任务。
*   在线处理不能阻塞入库 worker（必须异步/解耦）。
*   支持过期/窗口滑动：聚合窗口不会无限增大。

### 2.2 非功能需求（Non-functional）

*   **低侵入**：对现有 Engine/Repo/Service 改动应可控（新增可选 hook/事件机制），不破坏已有接口。
*   **低耦合**：Engine 不反向依赖 OnlineMicroClusterManager，避免循环 import；使用 listener/回调方式。
*   **可观测**：提供运行状态（job 状态、簇数、噪声数、耗时、资源用量估计）。
*   **可扩展**：未来可对多个 collection 开 plan，但默认限制数量，避免“无限实例化”的平台化复杂度。

***

## 3. 关键事实与设计影响

1.  \*\*Chroma collection 单机默认使用 HNSW 作为 ANN 索引，`space` 定义距离函数，并支持 `cosine`。\*\*这允许我们将“簇中心”也存为一个 collection，用同一套 ANN 查询做在线归类候选检索。 [\[petkir.at\]](https://www.petkir.at/blog/semantic-kernel/02_sindex_02_vector-search-basics), [\[arxiv.org\]](https://arxiv.org/abs/2407.08623)
2.  \*\*Chroma 的 cosine 距离是 `1 - cosine_similarity`。\*\*因此在线/离线若使用阈值 `T_event`（相似度），对应距离阈值为 `1 - T_event`，语义完全一致。 [\[petkir.at\]](https://www.petkir.at/blog/semantic-kernel/02_sindex_02_vector-search-basics), [\[linkedin.com\]](https://www.linkedin.com/pulse/rethinking-similarity-how-diem-outperforms-cosine-high-berkowitz-oklwe)
3.  \*\*流式聚类典型采用“在线维护微簇摘要 + 离线阶段重聚合纠偏”的两阶段范式。\*\*你的“每小时跑 24h 全量 + 在线微簇”与这一范式一致。 [\[faiss.ai\]](https://faiss.ai/cpp_api/struct/structfaiss_1_1OPQMatrix.html), [\[github.com\]](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization)

***

## 4. 方案选择与权衡（Options & Trade-offs）

### 4.1 在线聚合如何拿到“实时数据 + embedding”？

**Option A：在线重复 encode（无侵入）**

*   优点：无需改引擎。
*   缺点：重复算 embedding；吞吐受影响。

**Option B：upsert 后从 DB 读回 embeddings（无侵入但多一次 DB I/O）**

*   优点：不重复算。
*   缺点：增加 DB 读；可能影响延迟；需要一致性处理。

**Option C：在 upsert 流程中加入“可选 embedding hook + engine 事件广播”（小侵入，推荐）**

*   优点：不重复算，不额外读 DB；Engine 不反向依赖在线模块（listener 注册即可）；在线处理可异步队列化，不阻塞 worker。
*   缺点：需要对 `VectorCollectionRepo.upsert_document` 与 `VectorStorageEngine._handle_upsert_task` 做小幅修改。

**结论：选择 Option C**，因为它最符合“实时 + 不重复运算 + 小侵入 + 无反向引用”的综合目标。

***

### 4.2 离线聚类结果存储：只返回还是持久化？

**Option 1：只返回（类似现有 analysis jobs 内存存储）**

*   优点：实现快。
*   缺点：重启丢；无法给在线微簇提供稳定基准；多实例无法共享。

**Option 2：持久化（推荐）**

*   优点：稳定可查询；离线版本可追溯；在线可对齐版本；易于扩展。
*   缺点：需要设计存储（SQLite/Redis/其它）。

**结论：选择持久化（Option 2）**。在线微簇与“24h 不重复列表”属于业务核心产物，不应仅存在内存。

***

### 4.3 多 collection / 多 plan 的通用性 vs 必要性

**Option 1：服务只支持一个聚合目标（硬编码某 collection）**

*   优点：简单。
*   缺点：扩展困难。

**Option 2：任意多个聚合实例（无限 plan）**

*   优点：最通用。
*   缺点：资源/权限/配额/状态管理复杂，与你当前需求不匹配。

**Option 3：Plan 注册 + 数量限制（推荐）**

*   优点：足够通用（可覆盖多个 collection）；同时通过 registry 做限制与防重，避免不必要复杂度。
*   缺点：需要 plan registry 与调度。

**结论：选择 Option 3**。

***

## 5. 最终设计（Final Design）

### 5.1 总体架构

*   `VectorStorageEngine`（基础设施层）
    *   负责 DB/模型/Repo/异步写入
    *   新增：**UpsertEventBus（listener 注册与事件广播）**
    *   新增：**可选 embedding hook**（把 upsert 时算出的 embeddings 推送给事件总线）

*   `AggregationRegistry`（计划注册与限制）
    *   管理 `AggregationPlan`
    *   做 plan 防重、collection 限制、最大计划数限制

*   `ClusterManager`（聚合编排层 / Orchestrator）
    *   管理离线任务调度、运行锁、版本
    *   持有多个 `OfflineClusterRunner`（按 plan）
    *   持有多个 `OnlineMicroClusterManager`（按 plan，若 enable\_online）
    *   负责离线完成后对在线做 reconcile

*   `OfflineClusterRunner`（离线执行层）
    *   从 engine/repo 拉取窗口数据
    *   运行 HDBSCAN/DBSCAN 或 Agglomerative+threshold
    *   产出聚合结果并持久化
    *   产出簇中心（centroid）写入“cluster-centroids collection”（可选但强烈建议）

*   `OnlineMicroClusterManager`（在线执行层）
    *   订阅 engine 的 upsert 事件
    *   对属于本 plan 的 collection 的事件入队处理
    *   基于最近离线版本的簇中心集合做增量归类
    *   更新在线簇中心与映射（持久化 + 可选写回 centroids collection）
    *   提供对齐接口：`reconcile(version)`

*   `VectorDBService`（API 层）
    *   新增 plan 管理与聚合 API
    *   复用现有 executor/jobs 机制运行离线任务

> 说明：簇中心存为 Chroma collection 的原因是：Chroma 支持 HNSW ANN 与 cosine 等 distance space，可以直接做“簇中心近邻检索”以支持在线归类候选召回。 [\[petkir.at\]](https://www.petkir.at/blog/semantic-kernel/02_sindex_02_vector-search-basics), [\[arxiv.org\]](https://arxiv.org/abs/2407.08623)

***

### 5.2 数据模型（持久化）

建议至少两张表（SQLite 或其它 KV/DB）：

1.  **doc\_cluster\_map**

*   `plan_id`
*   `collection_name`
*   `doc_id`
*   `cluster_id`
*   `version`（离线版本号；在线增量可标记 `version=latest+online` 或额外字段）
*   `ts`（文档时间戳）
*   `source_meta`（可选）

2.  **cluster\_meta**

*   `plan_id`
*   `collection_name`
*   `cluster_id`
*   `version`
*   `size`
*   `last_seen`
*   `repr_doc_id`（代表文档）
*   `preview`（可选）
*   `stats`（可选：radius/dispersion 等）

可选第三种持久化：
3\) **cluster\_centroids collection（Chroma）**

*   collection 名可为：`clusters__{plan_id}`
*   每条记录：`id = cluster_id`
*   `embedding = centroid`
*   `metadata = {plan_id, version, size, last_seen, ...}`

> 这允许在线模块用 ANN 查询快速找到最相近簇（HNSW）。 [\[petkir.at\]](https://www.petkir.at/blog/semantic-kernel/02_sindex_02_vector-search-basics), [\[arxiv.org\]](https://arxiv.org/abs/2407.08623)

***

### 5.3 Upsert 事件与“无反向引用”在线消费（小侵入关键点）

#### 5.3.1 Engine 新增：事件总线（listener）

*   `engine.register_upsert_listener(fn)`
*   `engine._emit_upsert_event(event)`

Engine 本身不 import 在线类，只广播 dict 事件，实现低耦合。

#### 5.3.2 Repo 新增：可选 embedding hook

在 `VectorCollectionRepo.upsert_document` 中，计算 `embeddings_np = self._vectorize(chunks)` 后触发 hook（若存在），再写入 Chroma。hook 传递：

*   `collection_name`
*   `doc_id`
*   `chunk_ids`
*   `chunks`（可选）
*   `embeddings_np`（np.ndarray）
*   `metadata`（清洗后）

> 这使在线聚合可复用 embeddings，避免重复 encode。  
> 同时因为 hook 是可选参数，不会影响现有调用路径（小侵入、向后兼容）。

#### 5.3.3 Worker 不阻塞：在线模块内部队列化

在线模块 `on_event` 只 enqueue，内部线程顺序处理，避免阻塞 engine 的写入 worker（你现有写入队列设计已经体现了“重任务异步化”的思路，可复用同样理念）。

***

### 5.4 离线-在线配对与版本对齐（Reconcile）

**每个 plan 配对一组：**

*   OfflineRunner：每小时跑 `window=24h`（或 plan 配置），产出 `version=YYYYMMDD_HH00`。
*   OnlineManager：使用 `version` 的簇中心集合做增量归类。

**reconcile 策略：**

*   离线任务完成后：
    1.  写 `doc_cluster_map` / `cluster_meta` / `clusters__{plan_id}`（centroids）
    2.  调用 OnlineManager：`reconcile(plan_id, version)`
        *   切换在线基准版本
        *   清理过期簇
        *   可选择“重建在线缓存”或“增量更新”

这与典型数据流聚类“在线摘要 + 离线重聚合纠偏”一致。 [\[faiss.ai\]](https://faiss.ai/cpp_api/struct/structfaiss_1_1OPQMatrix.html), [\[github.com\]](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization)

***

### 5.5 ClusterManager 的职责（你提出的关键点确认）

你最后提出：“ClusterManager 应该管理映射和资源，需要增加 offline cluster”。最终设计中：

*   `ClusterManager` = **计划与生命周期管理 + 资源约束 + 映射与版本编排**
    *   持有 `AggregationRegistry`
    *   持有 per-plan `OfflineClusterRunner` / `OnlineMicroClusterManager`
    *   提供 run/schedule/reconcile API
    *   强制 plan 数量上限、max\_points、并发限制与运行锁
*   `OfflineClusterRunner` = **离线聚合执行器（自动聚类算法实现位置）**

> 这样职责清晰：算法在 runner，编排与资源在 manager，事件流在 online manager。

***

## 6. 软件结构（Software Structure）

建议新增模块（文件）：

    VectorDB/
      VectorStorageEngine.py
      VectorDBService.py
      ClusterAnalysisPipeline.py  (已有)
      aggregation/
        plans.py                 # AggregationPlan dataclass
        registry.py              # AggregationRegistry（防重、限制）
        cluster_manager.py       # ClusterManager（编排、调度、reconcile）
        offline_runner.py        # OfflineClusterRunner（HDBSCAN/DBSCAN/AggloThreshold）
        online_microcluster.py   # OnlineMicroClusterManager（事件消费、增量归类）
        persistence.py           # Storage adapter（SQLite/Redis/抽象接口）

Engine 侧最小改动：

*   增加 event bus（listener 列表）
*   Repo 的 upsert\_document 增加可选 hook 参数（默认 None）

Service 侧新增 endpoints：

*   `/api/aggregation/plans`（注册/查询/删除 plan）
*   `/api/aggregation/plans/<plan_id>/run`（触发离线聚合）
*   `/api/aggregation/plans/<plan_id>/events?version=latest`（取事件列表）
*   （可选）`/api/aggregation/plans/<plan_id>/status`（在线状态）

***

## 7. 备选方案（Alternatives）

### Alternative 1：完全无侵入（在线重复 encode）

*   不改 engine/repo；在线 manager 在收到“新文档通知”时自行 encode。
*   适合：吞吐较低、模型较小、或只对非常短文本做在线聚合。

### Alternative 2：无侵入但多一次 DB I/O（upsert 后读回 embeddings）

*   在线模块通过 `collection.get(include=['embeddings'])` 拉回 chunk embeddings，再聚合为文章向量。
*   优点：不重复算。
*   缺点：额外 DB 读；在高写入量下可能成为瓶颈。

### Alternative 3：不使用簇中心库（在线直接查原始文档向量库）

*   在线归类时对“原始 doc 向量集合”做 top-k 检索，再映射到 cluster\_id。
*   缺点：映射层复杂且候选更多；簇中心库更直接、效率更好（尤其 plan 多时）。
*   仍可作为简化版 MVP。

### Alternative 4：在线仅做 near-duplicate（LSH/MinHash），事件聚合只离线做

*   在线阶段仅去重，不做事件簇；事件聚合全靠每小时离线。
*   对“实时事件列表”要求不高时非常省事；对 typo/轻改写鲁棒的 MinHash/LSH 在大规模近重复检测中常用。 [\[qdrant.tech\]](https://qdrant.tech/course/essentials/day-2/what-is-hnsw/), [\[learn.microsoft.com\]](https://learn.microsoft.com/en-us/azure/search/vector-search-ranking)

***

## 8. 风险与缓解（Risks & Mitigations）

1.  **阈值与算法参数敏感**

*   缓解：离线结果作为基准版本；在线只做增量并受离线纠偏；为 plan 提供参数调优与 A/B。 [\[faiss.ai\]](https://faiss.ai/cpp_api/struct/structfaiss_1_1OPQMatrix.html), [\[github.com\]](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization)

2.  **在线事件处理阻塞写入**

*   缓解：listener 只 enqueue；在线内部线程处理；队列满时降级（丢弃/延迟）。

3.  **plan 过多导致资源失控**

*   缓解：registry 限制总 plan 数、同 collection plan 数、每次 max\_points、并发数与运行锁。

4.  **多实例一致性**

*   缓解：核心结果持久化；在线缓存可重建；离线版本化；必要时用 Redis/DB 作为共享状态。

***

## 9. 最终决策摘要（Decisions）

*   使用 **Plan 注册 + 限制**（防重、资源控制）作为聚合入口。
*   离线聚合实现 **HDBSCAN/DBSCAN + AgglomerativeThreshold** 的自动聚类。
*   在线微簇采用 **小侵入 hook + event bus** 获取实时数据与 embeddings（避免重复运算、无反向引用）。
*   `ClusterManager` 作为编排与资源管理中枢，按 plan 配对 `OfflineClusterRunner` 与 `OnlineMicroClusterManager`，离线完成后 reconcile 在线状态。
*   聚合结果**持久化**，并可选将簇中心写入 Chroma collection 以利用 HNSW 近邻检索能力。 [\[petkir.at\]](https://www.petkir.at/blog/semantic-kernel/02_sindex_02_vector-search-basics), [\[arxiv.org\]](https://arxiv.org/abs/2407.08623), [\[linkedin.com\]](https://www.linkedin.com/pulse/rethinking-similarity-how-diem-outperforms-cosine-high-berkowitz-oklwe)

***

# 下一步（落地执行清单）

如果你认可该设计，我建议按以下顺序实现（每步都可独立验收）：

1.  **实现 Plan/Registry（不动 engine）**：先把“注册、限制、防重、查询”跑通。
2.  **实现 OfflineClusterRunner（读 repo → 自动聚类 → 持久化）**：先把离线 24h 聚合跑通，产出 version。
3.  **实现 Engine 事件总线 + Repo hook（小侵入）**：仅加 hook 与 listener，不引入在线类。
4.  **实现 OnlineMicroClusterManager（订阅事件 → 入队 → 归类 → 写映射/centroid）**
5.  **实现 reconcile（离线完成后刷新在线基准）**
6.  **在 VectorDBService 增加 endpoints**：plan 管理、触发离线聚合、查询事件列表。

***