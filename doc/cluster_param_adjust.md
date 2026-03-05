可以，从“聚合的距离/精度”这个问法来看，你其实关心两件事：

1.  **哪些文档会被聚到同一个事件簇**（相当于“距离阈值/相似度阈值”）
2.  **聚合结果的质量**（误聚合少=Precision高；漏聚合少=Recall高）

而你的系统里又分两条链路：

*   **离线聚合（OfflineClusterRunner）**：HDBSCAN / DBSCAN / Agglomerative+threshold
*   **在线微簇（OnlineMicroClusterManager）**：阈值归类（T\_event/T\_dup 等）

下面我按“你现在的实现”给你一个**可直接落地的调参指南**，并告诉你各参数分别控制什么、怎么调、如何验证。

***

## 0) 先统一“距离/相似度”的定义（避免阈值用错）

你现在 Chroma collection 配的是 `cosine`，Chroma 返回的是 **cosine distance**（而不是 cosine similarity），并且 cosine distance 的定义是 `1 - cosine_similarity`。 [\[github.com\]](https://github.com/chroma-core/chroma/issues/1136), [\[github.com\]](https://github.com/chroma-core/chroma/issues/3307)

因此如果你在代码里用：

*   `similarity = 1 - distance` ✅  
    那么阈值最好都用“相似度阈值”来描述更直观（例如 `T_event=0.85`）。

> 小提醒：离线聚类你目前是直接从 DB 拿 embeddings，然后在本地算法里算距离；在线微簇是你自己算 cosine（dot）做阈值归类。最好统一用**归一化向量 + cosine相似度**这套语义（你代码里也基本是这么做的）。

***

## 1) 离线聚合：怎么调“距离/精度”？

### A) HDBSCAN（推荐用于新闻事件聚合）

HDBSCAN 的优势是：**不需要指定 K**，还能把噪声点标为 `-1`，对“热点密集 + 长尾稀疏”很友好。 [\[pypi.org\]](https://pypi.org/project/chronicle-events/), [\[mdpi.com\]](https://www.mdpi.com/2078-2489/17/3/233)

你可以把它理解为：你主要在调 **“什么规模才算一个事件”** 和 **“对噪声有多严格”**。

#### 关键参数（从重要到次要）

1.  **`min_cluster_size`（最重要）**

> “一个事件至少要有多少条报道才算事件？”

*   调大：簇更少、更“干净”（Precision ↑，Recall ↓）
*   调小：簇更多、更容易把长尾也聚起来（Recall ↑，Precision ↓）  
    HDBSCAN 文档也把它作为最直观的主参数。 [\[pypi.org\]](https://pypi.org/project/chronicle-events/), [\[mdpi.com\]](https://www.mdpi.com/2078-2489/17/3/233)

2.  **`min_samples`（严格程度 / 噪声阈值）**

> “多大密度才算簇内点？”

*   调大：更严格，更多点变噪声（Precision ↑，Recall ↓）
*   调小：更宽松，更容易把点吸入簇（Recall ↑，Precision ↓） [\[pypi.org\]](https://pypi.org/project/chronicle-events/), [\[mdpi.com\]](https://www.mdpi.com/2078-2489/17/3/233)

3.  **`cluster_selection_epsilon`（合并相近簇的距离阈值，慎用）**

> 用于把距离很近的簇合并；一般先不动，等你发现簇碎片化严重再考虑。 [\[pypi.org\]](https://pypi.org/project/chronicle-events/), [\[mdpi.com\]](https://www.mdpi.com/2078-2489/17/3/233)

4.  **`metric`**

*   你现在对向量做了归一化，用 `euclidean` 跑 HDBSCAN通常没问题（归一化后欧式距离与 cosine 距离单调相关）。
*   如果你想严格用 cosine，可以试 `metric="cosine"`，但实际效果看数据分布。

#### 推荐调参流程（很实用）

*   固定 `metric`（先用你现在的），先扫 `min_cluster_size`：例如 2/3/5/8/10
*   每个点跑一次，记录：
    *   `n_clusters`
    *   `n_noise`
    *   最大簇大小/平均簇大小
*   你会看到一个“合适区间”：噪声不至于过多，簇也不至于碎到爆。

***

### B) DBSCAN（也可用，但 eps 很敏感）

DBSCAN 的核心是 **`eps`**（邻域半径）和 `min_samples`。  
你可以把 `eps` 直接理解成“最大允许距离”。

*   `eps` 小：只聚非常近的（Precision ↑，Recall ↓）
*   `eps` 大：聚得更松（Recall ↑，Precision ↓）

> 如果你的向量是 L2 归一化的，并且你想用 cosine 相似度阈值 `T_event`，那么一个常见的直觉换算是：  
> **cosine\_distance = 1 - T\_event**  
> 所以 DBSCAN 的 `eps` 可以从 `1 - T_event` 附近试起。

***

### C) Agglomerative + distance\_threshold（你最想要的“阈值式聚合”）

这个方式非常直观：

*   你只需要一个 `distance_threshold`（或等价的 `similarity_threshold`）
*   相似度高于阈值就会被聚在一起
*   簇数量自然产生

实践建议：

*   如果你在离线里用的是 cosine 距离：
    *   `distance_threshold = 1 - T_event`
*   一般做事件聚合，`T_event` 先从 **0.80\~0.90** 扫描，然后再用小样本标注校准（下一节讲）

***

## 2) 在线微簇：怎么调“距离/精度”？

在线微簇的“距离/精度”几乎完全由你定义的阈值控制：

*   **`T_event`**：达到这个相似度就认为是“同事件”
*   **`T_dup`**：达到这个相似度就认为是“近重复/转载”（可以做更严格处理）

你现在 OnlineMicroClusterManager 就是这么做的（cosine 相似度 + 阈值）。

调参规律（非常清晰）：

*   `T_event` ↑ → 更不容易加入已有簇 → 簇更碎、Precision ↑、Recall ↓
*   `T_event` ↓ → 更容易加入已有簇 → 簇更大、Recall ↑、Precision ↓
*   `T_dup` 一般设置比 `T_event` 高很多（例如 0.95+），用于识别几乎相同内容（转载/重复）。

> 你也可以给 online 加一个 “半径/松散度”闸门（你之前讨论过）：  
> 簇内部方差太大就拒绝继续吸纳边界点，从而提升 Precision，避免簇漂移。

***

## 3) “精度”怎么量化？（否则只是在拍脑袋调阈值）

无论离线还是在线，想把“距离/精度”调到靠谱，你需要一个最小评估闭环：

### 最小评估集（建议 200\~500 对）

人工标注一些文本对（同事件 / 不同事件 / 近重复），然后看：

*   **Precision（误聚合率低）**
*   **Recall（漏聚合率低）**

### 快速可行的替代（无标注也能做）

在一个时间窗口里，对每篇文档取 top-1/ top-5 的相似度分布（你可以用现有向量检索实现），看看相似度有没有明显双峰：

*   高峰：转载/同事件
*   低峰：新事件/无关  
    据此挑一个分割点作为 `T_event` 初值，再用离线结果纠偏。

***

## 4) 一个经常被忽略但影响巨大的因素：你聚合的“粒度”

你现在离线 runner 是按 `original_doc_id` 归并 chunk embedding 做 doc 级向量，这一步对精度影响很大：

*   如果 doc 很长且 chunk 很多，平均向量可能会被“主题稀释”
*   你可以尝试：
    *   只用前 N 个 chunk
    *   或者用“最相似 chunk”代表 doc（对事件聚合可能更敏感）
    *   或者用 summary collection（你现在也有）做聚合更稳定

***

## 5) 在你的系统里，具体怎么落地调参（建议默认值）

### 离线（HDBSCAN）

*   `min_cluster_size`: 3（你现在就是）
*   `min_samples`: 2（你现在就是）
*   如果发现簇太碎：优先 **调大 min\_cluster\_size**
*   如果发现噪声太多：**调小 min\_samples** 或调小 `min_cluster_size`

### 在线（阈值）

*   `T_event`: 0.85（起点不错）
*   `T_dup`: 0.95（转载近重复）
*   如果误聚合多：提高 `T_event`（例如 0.88/0.90）
*   如果漏聚合多：降低 `T_event`（例如 0.82/0.80）

***

## 6) 最后一个关键：检索精度 vs 聚类精度是两码事

*   **聚类精度**：由 HDBSCAN/阈值控制
*   **检索精度（ANN）**：由 HNSW/向量库索引参数控制（例如 ef\_search 等） [\[github.com\]](https://github.com/chroma-core/chroma/issues/1136)

你现在离线聚合是从 DB 拉数据到内存聚类，跟 HNSW 参数关系不大；  
但在线微簇未来如果改成“簇中心也放 Chroma 里做 ANN 检索”，那就要开始调 HNSW 的 ef\_search（更高 recall，延迟更大）。 [\[github.com\]](https://github.com/chroma-core/chroma/issues/1136)

***

# 给你一个“下一步最有效”的问题（能让我给出更精确的建议）

你现在最在意的是哪一种错误？

1.  **误聚合（不同事件被合到一起）太多**（Precision 不够）
2.  **漏聚合（同一事件被分成多个簇）太多**（Recall 不够）
3.  **簇太碎/噪声太多**（HDBSCAN 输出大量 -1）

你回答后，我可以给你一个更具体的调参建议（比如把 `min_cluster_size/min_samples` 的组合、或者把在线 `T_event/T_dup` 的区间缩到很窄，并告诉你该如何用你现有 UI/接口快速验证）。
