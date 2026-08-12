# DeepView 研究汇总与技术方案

> 状态：主体冻结，决策表 #1-13、#16-17、#20-22 待拍板；已定项：#14-15、#18-19、#23-24
> 日期：2026-08-12

## 1. 项目概述

**目标**：Tauri 桌面应用，把本地原始的动漫/电影文件刮削成媒体知识（LightRAG 知识库），全程可视化 DeepAgents（PyPI 包名 `deepagents`）多子代理执行过程。

**本质**：输入是"原始文件"（目录名/文件名），输出是"结构化知识"（媒体图谱 + 元数据）。中间靠 LLM 世界知识 + TMDb/Bangumi 双源 API 检索，用户交互（候选确认、合并确认）是流程一等公民。

**使用场景**：个人自用，几十到几百部媒体，数据量小，SQLite 足够。

## 2. 技术调研结论

### 2.1 DeepAgents（langchain-ai/deepagents）

- Python 库（PyPI 包名 `deepagents`），构建于 LangGraph 之上，官方 README 确认
- 核心能力：
  - **Sub-agents**：通过 `task` 工具委派子代理，隔离上下文
  - **异步子代理**（`AsyncSubAgentMiddleware`，v0.5+）：在远端 Agent Protocol 服务器上跑后台任务
  - **TodoListMiddleware**：todo 在 state 的 `todos` 字段（来自 langchain）
  - **FilesystemMiddleware**：文件读写（StateBackend 线程内 / StoreBackend 跨线程）
  - **Human-in-the-loop**：`interrupt_on` 参数 + `interrupt()` 原语（见第 7 节）
  - **Persistent memory**：checkpointer（短时）+ store（长时）

### 2.2 LangGraph 持久化（关键结论：状态在数据库里）

- **Checkpointer**：每次 superstep（一次 LLM 调用/一次工具调用后）把 graph state 快照写入数据库。`SqliteSaver`/`AsyncSqliteSaver` 是官方 SQLite 实现。
- **Store**：跨线程长时记忆（`SqliteStore`）。
- **重要推论**：异步子代理的任务状态（`async_tasks` 字段）、todo（`todos` 字段）、消息历史全部随 state 落进 SQLite。
- **结论**：**读数据库 = 拿任务状态**。不需要翻译 LangGraph 事件流，不需要事件落盘回放。这是协议设计的基础。

### 2.3 async_tasks 数据结构（读源码确认）

```python
class AsyncTask(TypedDict):
    task_id: str        # = thread_id（子代理线程）
    agent_name: str
    thread_id: str
    run_id: str
    status: str         # running / success / error / cancelled / timeout / interrupted
    created_at / last_checked_at / last_updated_at: str
```

- 存于 agent state 的 `async_tasks` 字段（reducer 合并更新）
- 工具：`start_async_task` / `check_async_task` / `update_async_task` / `cancel_async_task` / `list_async_tasks`
- **扁平 dict，无父子层级**——任务树层级需从消息流推导（后续增强）

### 2.4 工具链现状验证

- **sqlite-vec**：活跃维护（asg017/sqlite-vec，Mozilla Builders 项目）；**已不再需要**——图谱/向量由 LightRAG 承载（见 3.1）
- **LightRAG**（HKUDS/LightRAG）：选为图谱 + RAG 后端（见 2.5、3.1）
- **CodeGraph**（colbymchenry/codegraph）：图 + 检索参考实现——SQLite 邻接表 + FTS5 + 拆词表 + RWR，零向量（见 5.6，背景调研）
- **Bangumi v0 API**：现代 API（/v0/search/subjects、/v0/subjects/{id}、characters、persons），需 User-Agent + Access Token
- **run-deck 项目**（用户已有 Tauri 项目）：React 19 + Base UI + shadcn，借鉴思路不复制文件

### 2.5 LightRAG 源码调研（2026-08-11，选为图谱 + RAG 后端）

**项目**：HKUDS/LightRAG（EMNLP 2025，38.7k stars，PyPI `lightrag-hku`）。轻量图 RAG 框架，微软 GraphRAG 的高性价比替代。

**核心机制**：
- 文档 → 分块 → LLM 抽取实体/关系 → 图 + 向量双索引；增量更新 + 选择性删除（删文档自动重建 KG）
- **双级检索**：local（局部实体精确）+ global（全局关系链宏观主题）+ hybrid + naive（纯向量）+ mix（全合并）；默认 mix；支持 rerank
- **4 角色 LLM**（EXTRACT/QUERY/KEYWORD/VLM）可独立配置；抽取阶段推荐非 thinking 模型
- 用法：SDK（`rag.insert_custom_kg(entities, relationships, chunks)` 可直接喂结构化数据**绕过 LLM 抽取**）或 REST server（`lightrag-server`，自带 WebUI，官方推荐集成方式）

**存储事实（源码验证，决定架构）**：
- **无 SQLite 后端**。默认：图=NetworkX 落 GraphML 文件、向量=NanoVectorDB 落 JSON、KV=JSON、文档状态=JSON；生产用 PostgreSQL/MongoDB/OpenSearch 全家桶或 Neo4j/Milvus 混搭
- 默认存储是**单写者约束** + 跨进程全量文件重载（源码 docstring 明确 invariant）→ **多进程直连不可行，独立 server 模式解决**（server 内部自洽单写者，外部全走 REST）
- 存储后端可插拔（BaseGraphStorage/BaseVectorStorage/BaseKVStorage 抽象）
- embedding 维度锁死：换模型必须清库重建（rebuild_vdb 工具存在）

**对我们适用性结论**：它擅"文档进、图谱出"；我们"结构化数据进、图谱出"——靠 `insert_custom_kg` 适配，LLM 抽取能力用不上但无碍。**本地运行零运维**（默认文件存储，免 Neo4j/Milvus/PG 等重组件），个人规模即开即用；引擎承担知识存储（全文/向量/图谱）+ 检索 + 问答后端，不替代编排/隔离（见 3.1）。

### 2.5.1 2026 替代方案调研（2026-08-12 增补；3.1 决策维持不变）

**动机**：重新审视 ① LightRAG（2024 出品）在 2026 年是否有更好替代 ② LLM 抽取"跨窗口断连"（同一实体在不同 chunk 重复出现/跨 chunk 关系连不上，如"Vue 是前端框架"跨越两个窗口时丢失连接）③ 是否改用 Wikidata/DBpedia 这类预抽取图谱。

**2026 图 RAG 格局**（TypeGraph 2026 评测 + 源码核查）：

| 方案 | 2026 状态 | 断连问题 | 与 DeepView 匹配度 |
|---|---|---|---|
| **LightRAG** | 活跃（38.7k stars，持续迭代：pipeline 重构、GHSA 修复、新 LLM 后端适配） | ❌ 抽取范式固有 | ✅ cost-first 生产级；我们不用其抽取管线 |
| Microsoft GraphRAG | accuracy-first，索引成本 10-40x | ❌ | ❌ 重组件 + 贵 |
| LazyGraphRAG（微软 2025） | 懒抽取：索引成本降 99.9%，查询降 700x，本地友好 | ❌ 仍是抽取范式 | ⚠️ 无增量图维护价值 |
| Graphiti（Zep 时序图谱） | **增量边维护 + 实体解析**，专治跨事件断连；混合检索（向量+FTS+图） | ✅ 唯一对症 | ❌ 依赖 Neo4j 服务器，破坏零运维 |
| Cognee | 模块化 memory 引擎，多后端（Kuzu/Neo4j/FalkorDB） | ⚠️ | ❌ 组装成本高，定位偏 agent 记忆 |
| TypeGraph | TS + Postgres，早期 | - | ❌ 场景不符 |

**Kuzu 弃坑证据链（2026-08-12 核实）**：Kuzu（嵌入式属性图库，pip 即装、Cypher）——GitHub API 实锤 `archived: true`、最后提交 2025-10-10、329 个 open issue 无人处理；原因=母公司 **Kùzu Inc 被 Apple 收购**，团队撤走，归档公告仅留一句 "Kuzu is working on something new"；The Register 2025-10-14 报道。**嵌入式图库现状**：Kuzu 本尊死了，但社区 fork 继任者活着——**LadybugDB**（最主流，2025 年启动，自称 KuzuDB 继任者，v0.18，嵌入式列式图库，内置 FTS + vector index + MCP server）、RyuGraph（Predictable Labs）、bighorn（Kineviz）；但均**成立不足一年、无生产验证**。结论修正：不能说"嵌入式图库方向不成立"——而是**没有"新且久经考验"的选项**（老牌 Neo4j Embedded 是 JVM 库，违背零运维），自建路线（Kuzu 系 + 自研检索管线）= 重写 LightRAG 且押注未验证项目，无收益。

**结论**：
1. **LightRAG 保留**：仍是 2026 年 cost-first 生产级选项，其"文档/实体/chunk 三层"匹配"剧情内容 + 实体关系 + 检索"；我们经 `insert_custom_kg` 绕过其抽取管线，断连缺陷已免疫。
2. **断连痛点的正解不是换引擎**：要么 Graphiti 式增量边维护（要 Neo4j，不选）；要么（我们已走的）结构化数据直喂，不依赖 LLM 抽取。**Wikidata 预抽取图谱同样免疫此问题**（关系是人类维护的，非窗口抽取）。

### 2.5.2 Wikidata 预抽取图谱补全层（2026-08-12 新增，建议采纳，决策表 #21）

**可行性证据**：
- **WikiProject Anime and Manga 活跃**：anime(Q1107)/manga(Q8274)/seiyū 声优(Q622807) 概念成熟，作品/角色/声优/系列关系人类维护，日本动画覆盖极佳
- **ID 锚点现成（确定性实体链接，无需模糊匹配）**：Wikidata 属性 P345(IMDb)、P4983(TMDb TV)、P1207(MyAnimeList)、P1114(AniDB)——刮削得到的 TMDb/Bangumi ID 反查 wikidata item 即完成链接
- **工具链成熟**：SPARQL endpoint（按需查询）/ `wikidata-dl`（按 SPARQL 查询下载子集）/ `wikibase-dump-filter`（87GB 全量 dump 流式过滤出媒体子集，本地化可选）

**定位**：**关系补全层，不替代 LightRAG、不替代 media.db**。补全刮削 API（TMDb/Bangumi）缺失的关系——声优-角色-作品、系列-作品、制作公司、题材——关系来自人类维护图谱，无断连/错连。

**数据流**（挂在现有 merge 流程后）：
1. 刮削确认 → media.db 已存 TMDb/Bangumi ID
2. 实体链接：ID property 反查 wikidata item（wbsearchentities）
3. SPARQL 拉一跳子图：作品/角色/声优/系列/类型/制作公司
4. 适配器转 `insert_custom_kg`（entity/relation，source_id = wikidata item id），与刮削 KG 同批写入 LightRAG
5. 原始关系 JSON 缓存 media.db（防重拉 + SPARQL 限流缓解）

**约束**：依赖网络（刮削本身联网，可接受）；SPARQL 失败**静默降级**（补全失败不影响主刮削/入库流程）；不引入本地 wikidata 全量 dump（个人规模无必要）。

**关系本体映射表**（wikidata property → DeepView relation 类型）：

| Wikidata Property | 含义 | 映射 relation | 备注 |
|---|---|---|---|
| P50/P57 | 导演 | `directed_by` | 电影/剧集 |
| P161 | 主演 | `cast_in` | 演员-作品 |
| P175 | 出演者 | `performed_by` | 角色-演员（作品→角色） |
| P674 | 角色（作品→角色） | `appears_in` | 动漫核心 |
| P725 | 配音演员（角色→声优） | `voiced_by` | **动漫核心** |
| P179 | 系列归属 | `part_of_series` | 系列-作品 |
| P31 | 实例类型 | `instance_of` | 消歧（TV series/film/OVA） |
| P136 | 题材 | `has_genre` | 分类导航 |
| P495/P364 | 国家/原始语言 | 属性（media.db） | 不进图谱 |

### 2.5.3 检索方案对比（2026-08-12；LightRAG vs SQLite 自研检索，待拍板，决策表 #22）

**动机**：用户质疑 LightRAG 复杂度（管理界面/嵌入模型/重排模型/抽取 LLM 全家桶 + 生态动荡：Kuzu 被收购、领域年轻），倾向"更成熟的组件拼装"（SQLite/FTS5/向量算法）；同时"描述找片"（记得剧情不记得片名，如"讲高中生玩音乐剧的番"）场景确认需要**语义召回**能力。

**四个候选方案**：

- **A：LightRAG 图 RAG**（3.1 现状）——图+向量+全文+RAG 全家桶，`insert_custom_kg` 喂结构化数据绕过抽取；独立进程 :8300 黑盒隔离
- **B：CodeGraph 式纯确定性**——SQLite 邻接表 + FTS5（trigram 中文）+ 拆词表 + networkx RWR，**零向量零模型**（源码原话 `deterministic, no embeddings`）；对"描述找片"无力
- **C：CodeGraph + 向量层**（用户倾向）——B 的骨架 + embedding 语义召回，混合检索（FTS ∪ 向量 RRF 融合）；全在 SQLite 一个文件
- **D：纯向量**（Dify/Coze 主流模式）——向量库 + 混合检索（向量+BM25），无图结构、无多跳

**对比**：

| 维度 | A LightRAG | B CodeGraph 式 | C CodeGraph+向量 | D 纯向量 |
|---|---|---|---|---|
| 架构 | 独立进程 :8300 | SQLite 一个文件 | SQLite 一个文件 | 向量库一文件 |
| 依赖 | lightrag-hku + 4 角色 LLM + embedding + rerank | 无 | 一个 embedding 模型 | 向量库 + embedding |
| 复杂度 | 高（抽取/图/检索全家桶） | 低 | 低-中 | 低 |
| 精确检索（片名/别名） | ✅ | ✅ FTS5 | ✅ | ⚠️ 弱 |
| 语义检索（描述找片） | ✅ | ❌ | ✅ | ✅ |
| 关系多跳（谁关联谁） | ✅ | ✅ RWR/CTE | ✅ | ❌ |
| RAG 问答 | ✅ 开箱 | ❌（agent 直查 SQL） | ❌（agent 直查 SQL） | ⚠️ 组装 |
| 黑盒/生态风险 | 高（存储格式演进、2024 新领域） | 无（教科书技术） | 无 | 低 |
| 技术年龄 | 2024 | 1990s-2015 | 同左 + 2017+ | 2017+ |
| 自研工作量 | 适配器 + 检索调优（M3 已排） | 检索管线自研 ~200-400 行 | + 向量层 ~50-100 行 | 检索管线自研 |
| 数据可迁移 | GraphML/JSON 可导出 | 原生 SQLite | 同左 | 可 |

**向量层嵌入方式（方案 C 细节）**：
- `sqlite-vec`（vec0 虚拟表，Mozilla Builders 维护，Windows 预编译）——与 FTS5 同姿势：`WHERE embedding MATCH ? ORDER BY distance`；或 numpy 暴力召回（BLOB 存向量，几千条简介毫秒级，5.6.3 历史方案）
- 融合：FTS5 结果 ∪ 向量结果 → RRF 融合（`score = Σ 1/(rank+60)`）→ 喂 agent——业界标准，~10 行
- embedding 模型三选：本地 bge-m3（~2GB 模型，离线）/ 本地小模型 bge-small-zh（~100MB）/ 云端 API（硅基流动 bge 系，联网）

**能力边界（B/C 共同）**：无 RAG 问答开箱（agent 直查 SQL + LLM 世界知识兜底——DeepView 知识多为公共知识，LLM 已知，此损失可接受）；语义质量受简介文本质量限制。

**结论（记录为待拍板，不推翻 3.1）**：C 是"更成熟 + 能力全覆盖"的候选；B→C 意味着自研检索管线，工作量与风险转移到自研代码（比依赖黑盒可控）。若拍板 C：删除 ④ LightRAG server（四服务→三服务），M3 重定义为 media.db 图/检索层，决策 #14-16 作废，5.6 从"背景参考"转为实施依据。

#### 2.5.3.1 A vs C 能力细化对比（2026-08-12 补充）

**结论先行：能力上没有大损失，工程属性上有大变化（变好）。**

**能力逐项对比（用户可感知维度）**：

| 能力 | A LightRAG | C SQLite 邻接表+向量 | 差异 |
|---|---|---|---|
| 片名/别名精确检索 | ✅ 内部 BM25 | ✅ FTS5 trigram | 对等 |
| 描述找片（语义） | ✅ query 接口开箱 | ✅ 自研向量层（vec0/numpy） | 对等，C 更可控 |
| 关系多跳（谁关联谁） | ✅ graph_lookup（黑盒） | ✅ 递归 CTE + RWR（透明） | 对等，C 可自调深度/权重 |
| 增量更新/删除 | 文档级（内部 KV 状态机） | **SQL 事务级** | C 更精细 |
| 实体去重合并 | insert_custom_kg 内部合并 | UNIQUE 约束 + source_id | C 更确定性 |
| 可观测/调试 | 黑盒文件 + WebUI | **直接查表，SQL 全透明** | C 明显强 |
| 数据迁移 | GraphML/JSON 导出 | 原生 SQLite | C 更强 |
| 全局主题问答 | ✅ local/global 双级检索 | ❌ 无社区摘要 | **A 独有** |
| RAG 问答开箱 | ✅ query 直接出答案 | ❌ agent 自行拼检索上下文 | 工作量差异 |

**唯二"能力下降"项分析（影响均不大）**：
- **全局主题问答**：LightRAG 论文核心卖点，但为**大规模文档库**设计（几十万 chunk）；DeepView 几百部片，用户问的是具体作品/人物，宏观主题问题 LLM 世界知识 + 类型字段 SQL 聚合即可兜底——该能力在本场景基本用不上
- **RAG 问答从"调一个接口"变"agent 自己拼"**：最大实现差异（+200-400 行自研检索管线），但 DeepView 本有 agent 编排层，只是把 LightRAG 内部逻辑拿到手里重写；能力上限反而更高（组合多路召回 + 追问 + 多轮澄清）

**工程属性变化（全部变好）**：风险从外部移到内部（黑盒演进 → 自研成熟积木）；四服务→三服务；零新依赖（SQLite 基础组件 + bge-m3 已锁定）；决策 #14-16 作废、5.6 复活为实施依据。

### 2.6 Yuxi（语析）源码调研（2026-08-11，架构模式参考）

**项目**：xerrors/Yuxi。多租户企业知识库平台：RAG + 知识图谱 + LangGraph 多智能体。**直接引入 DeepAgents**（与我们同源）；早期参考 LightRAG，后自研 Milvus 图谱链路替代（README 明说"降低兼容性问题"——印证自研图谱耗时且难做，也印证 LightRAG 引擎适合做后端）。

**技术栈**：Vue 3 + FastAPI + LangGraph + ARQ worker + PostgreSQL + Redis + MinIO + Milvus（向量）+ Neo4j（图谱）+ Docker Compose 全家桶——企业多租户，**组件不可复用**（违背零运维原则）。

**图谱实现（MilvusGraphService 源码）**：
- 图本体在 **Neo4j**：Chunk/Entity 节点（entity_id = hash(kb_id, 规范化名, label)）+ MENTIONS（Chunk→Entity）/ RELATION（Entity→Entity，triple_id 幂等 MERGE）
- 实体与三元组**同时向量化进 Milvus**（MilvusGraphVectorStore）
- 构建管线：chunk → LLM 抽取（worker 池 + 3 次重试 + 进度上报）→ 写 Neo4j + PG 落账 → 向量批量索引（lease + 失败重试 + 状态机 pending/indexed/failed）
- **检索：`query_and_rank_chunks_by_ppr` = 向量召回种子实体 → 2 跳子图（`[*1..2]` Cypher）→ networkx Personalized PageRank（种子权重作 personalization，damping 0.85）→ chunk 打分排序**——与 CodeGraph RWR 同思路的现代实现样板

**可借鉴（写进我们实现计划）**：
1. PPR 检索排序算法（~30 行，M3/M5 查询层直接照搬思路）
2. `query_seed_subgraph`（两跳子图 Cypher）——任务子图"种子提取"样板
3. 构建状态机（chunk 级 + 向量 lease/重试）——任务子图构建状态设计参考
4. DeepAgents 集成 + 审批 middleware（LangGraph checkpoint resume 实现审批）——跨进程 HITL 恢复路径的官方实践参考

### 2.7 Yuxi 源码深度探索（本地完整源码 D:\code\github\Yuxi-main，2026-08-11）

技术基线：`deepagents>=0.6.8`（风险清单锁定值）+ `langchain>=1.3.9`（与我们同源）。本节省略重复内容，只记 2.6 未覆盖的深度发现。

#### 2.7.1 目录结构

```
backend/                FastAPI + uv；server(HTTP层) + package/yuxi(业务域)
├── server/             main.py / worker_main.py(ARQ入口) / routers(22个，全挂/api) / utils/lifespan.py
├── package/yuxi/
│   ├── agents/         BaseAgent/BaseContext/审批/Skill门控/toolkits/backends/skills/mcp
│   ├── services/       Run生命周期/请求排队/SubAgent编排/SSE事件/task_service(进程内Tasker)
│   ├── knowledge/      知识库(implementations/、graphs/、chunking/、parser/、eval/)
│   ├── repositories/   SQLAlchemy仓储层（14个）
│   └── storage/        postgres(双连接池)/redis/minio/neo4j
├── web/                Vue3 + Vite + Pinia + ant-design-vue
├── packages/yuxi-cli/  CLI
└── docker-compose.yml  全家桶
```

#### 2.7.2 最有价值的 5 个借鉴点

**① 审批 HITL 全链路**（`backend/package/yuxi/agents/tool_approval.py:10-26`，仅 26 行）：
```python
SENSITIVE_BACKEND_TOOLS = frozenset({"write_file", "edit_file", "execute"})
TOOL_APPROVAL_INTERRUPT_ON = {t: {"allowed_decisions": ["approve", "reject"]} for t in SENSITIVE_BACKEND_TOOLS}
def create_tool_approval_middleware(mode):
    if mode == "always_trust": return None
    return HumanInTheLoopMiddleware(interrupt_on=TOOL_APPROVAL_INTERRUPT_ON)
```
- 子智能体默认隐藏敏感工具（subagent/graph.py:31-48），防绕过主线程逐项审批
- resume **不走普通消息队列**，直接建 run：`Command(resume=resume_input)` → `agent.stream_resume_with_state(...)`（chat_service.py:1155,1197）
- 恢复结束扫 checkpoint 兜底：`state.tasks[].interrupts` / `__interrupt__` 提取载荷（chat_service.py:713-736），转 `human_approval_required`{action_requests, review_configs} 或 `ask_user_question_required`{questions}

**② SubAgent 调用链 + 确定性线程 ID**（`middlewares/subagent_task.py:409-448`）：
```python
child_thread_id = subagent_child_thread_id(creator_run.conversation_thread_id, agent_item.slug, tool_call_id)
```
- `child_thread_id = hash(父线程, slug, tool_call_id)`——断线恢复后子线程归属稳定（我们 async_tasks 扁平结构正需要）
- 异步五件套 `subagent_start/status/events/cancel/await` 返回结构化 JSON ToolMessage（含 run_id/thread_id/events_url）

**③ 两阶段事件模型**（Request SSE → Run SSE）：
- Request SSE（agent_request_queue_service.py:608）：DB 轮询 + COUNT 查队列位置（O(1)），`queued`(含 position) → `run_created`(含 run_id) 
- Run SSE（run_queue_service.py:155）：Redis Stream `xadd run:events:{run_id}`，envelope 带 schema_version/run_id/thread_id/event/payload/created_at；游标续传 `xrange(min=f"({after_seq}")`；事件 2h 过期
- 取消：key `run:cancel:{run_id}`(TTL 1800s) + pubsub 双通道；worker 侧 `asyncio.Event` + `asyncio.wait` 竞速优雅终止（run_worker.py:289-305）
- **DeepView 平移**：WS 代替 SSE/Redis Stream，事件语义（排队→运行分离、游标续传、取消竞速）直接套用

**④ 图谱 PPR 检索**（`knowledge/implementations/milvus.py:1163-1199` + `graphs/milvus_graph_service.py:917-970`）：
- 种子权重：向量命中实体 1.0 / 关系命中两端 0.8 / 向量命中 chunk 的 ent_ids 0.3
- `networkx.pagerank(graph, alpha=damping, personalization=personalization)`——Chunk 与 Entity 同图，Chunk reset=0 只做排名对象；personalization 必须覆盖全部节点（0 值节点抛错）
- 纯 Python 逻辑，**无需 Neo4j**——LightRAG 提供实体/chunk 关系后可直接平移

**⑤ Skill 门控**（`middlewares/skills.py:263-300` + `:400-483`）：
- 未激活 Skill 的工具对模型不可见：`model_tools = [t for t in request.tools if t.name not in gated_tool_names]`
- 模型 read_file(SKILL.md) → 动态激活 → `Command(update={"activated_skills": [slug]})` 经 state reducer 累积；依赖闭包 DFS 展开
- **DeepView 用法**：FastMCP 刮削/知识库工具照抄此门控——模型默认拿不到全部敏感工具，按需激活防 prompt 注入

#### 2.7.3 架构不变量（值得抄的工程纪律）

- HTTP 路由保持薄；用例在 services，持久化在 repositories
- **intake-then-dispatch**：先提交 DB 事实，再投队列（幂等/崩溃恢复根基）
- 同一线程普通请求 FIFO 串行派发；排队与运行用不同状态模型
- DB 保存事实状态，Redis 只承担投递/事件/取消/缓存（我们对应：SQLite 事实 + WS 增量）

#### 2.7.4 两个重要发现

- **SQLite checkpoint 姿势**（base.py:397-412）：Yuxi 的 LangGraph checkpoint 默认就是 SQLite（`AsyncSqliteSaver`，WAL + `busy_timeout=5000` + `synchronous=NORMAL`，为 api/worker 多进程并发写同库专门配）——**我们 M0 直接抄这三条 PRAGMA**；`is_alive()` 缺失需打补丁
- **LITE_MODE**（server/routers/__init__.py:24,51）：官方显式支持"不注册 knowledge/graph 路由、跳过重依赖初始化"，认证/聊天/Skills/MCP 照常——与我们"轻量化可裁剪"定位同构

#### 2.7.5 轻量化删减清单（DeepView 不需要的）

| 删减项 | 替代 |
|---|---|
| 多租户/RBAC/部门/oidc 权限体系 | 单用户，删 role 过滤（BaseContext metadata 机制保留，忽略 auth） |
| PostgreSQL（业务池 + checkpoint 池，42KB ORM 模型） | SQLite（其 checkpoint SQLite 代码即范本） |
| Redis/ARQ + 独立 worker 进程 | 进程内执行 + 内存队列 / task_service 式 Tasker（轻任务官方就是这么干的）；事件用 WS 直推 |
| MinIO 对象存储 | 本地文件系统 |
| Milvus/etcd/Neo4j | LightRAG 内置向量+图；PPR 思路平移 |
| 沙盒 provisioner（docker） | 本机执行 + 工具审批兜底 |
| MinerU/PaddleOCR 解析全家桶 | 无企业文档量级 |
| 评估体系（eval/、benchmark、langfuse 上传） | 后期按需 |
| 图谱构建三队列 worker + 图谱管理 UI（GraphCanvas） | LightRAG 图构建内建 |
| Dashboard/用户管理/部门管理/CLI 授权 | 桌面端不适用 |
| docker-compose 全家桶/nginx/Makefile | 桌面应用无部署编排 |

#### 2.7.6 方案对比结论：DeepView vs Yuxi（2026-08-11 讨论定稿）

**结论：DeepView 轻量方案方向正确——Yuxi 的复杂度是为"多租户 + 并发 + 企业可观测"付费的，个人自用一个都不欠；砍掉 80% 组件，保留全部关键模式，轻而不弱。**

| 维度 | DeepView | Yuxi | 判断 |
|---|---|---|---|
| 服务数量 | 4 进程一键起 | 11 容器（Neo4j 内存 1GB+、Milvus/etcd 更多） | DeepView 胜 |
| 存储 | SQLite × 3 + LightRAG 文件 | PG(双池)+Redis+MinIO+Milvus+Neo4j | DeepView 胜——零运维可复制迁移 |
| 执行模型 | 主服务进程内 asyncio（2026-08-12 更新：统一 anyio API，asyncio 后端，见决策 #24） | API/worker 分离 + ARQ 重试 + Redis Stream | **Yuxi 胜**——崩溃恢复/重试健壮；我们进程挂任务丢（重跑可接受） |
| 事件推送 | WS 增量 + /state 全量兜底 | SSE 双通道 + Stream 2h 留存续传 | 平手——/state 对齐更简单可靠 |
| 工具安全 | MCP + 审批 HITL + Skill 门控 | MCP + 审批 + docker 沙盒 | 稍弱——本地直执行，靠审批兜底；工具面窄（只读 API + 受限写）风险低 |
| 并发/多租户 | 单用户无争抢 | FIFO 队列 + 权限体系 | 不需要，砍得对 |
| 知识图谱 | LightRAG 成熟引擎零维护 | 自研 Neo4j+Milvus+PPR（作者都后悔的坑） | DeepView 胜 |

**砍掉项判断**：
- worker 分离 + 重试：桌面单用户低并发，agent 跑 asyncio task 即可；**前提**：FastAPI 长任务全走 async（LLM/HTTP/DB 都 await，禁同步 SQLite 阻塞 loop，M1 测）
- 沙盒：工具面窄（只读 API + 受限写 + 候选确认），**前提**：不向 agent 暴露通用文件写/执行工具
- Postgres checkpoint：Yuxi 因双进程并发写才上 PG；我们单写者，SQLite + PRAGMA 姿势（2.7.4）足够

**三个警惕点（写进实现计划）**：
1. 主服务进程内跑一切（编排+WS+watcher+agent）：崩溃=运行中任务丢（重跑可接受）；防 event loop 阻塞（M1 测试项）
2. LightRAG 黑盒依赖：锁版本 + REST 薄封装隔离在 media 层
3. merge 写 LightRAG 失败要有重试/标记（media.db 记 sync 状态），避免"用户确认了但知识没进库"

### 2.8 上下文压缩策略（2026-08-11 调研定稿）

**调研对象**：主流 Code Agent（Codex CLI / Cline / Aider / Claude Code / Continue / Cursor）+ 方法论（Anthropic context engineering、MemGPT、LLMLingua）+ Yuxi 本地源码。**结论先行：没有一家用朴素滑动窗口**。共性公式 = **触发阈值 + 最近消息逐字保留 + 其余摘要化 + 初始上下文重注入**，并按消息类型分层加权（用户指令 > 工具结果 > assistant 消息）。Yuxi 已有基于 deepagents 的完整两级压缩实现（与我们同源），**直接继承**。

#### 2.8.1 主流 Code Agent 调研

| 项目 | 触发机制 | 保留策略 | 权重差异 |
|---|---|---|---|
| OpenAI Codex CLI | token 阈值自动 + 手动 /compact | 最近用户消息逐字保留（20K tokens 预算）、工具输出按策略截断、system prompt/AGENTS.md 重注入 | 用户 > 工具结果 > assistant（摘要化） |
| Cline | 接近窗口限制自动触发 | 全量 LLM 摘要替换历史 + checkpoint 可回滚到压缩前 | 技术决策/代码模式/文件变更优先保留 |
| Aider | `--max-chat-history-tokens` 软阈值 | 弱模型（haiku 档）做历史摘要；Repo Map 图排名注入（`--map-tokens` 预算） | 可编辑文件 > 只读文件 > Repo Map |
| Claude Code | 接近窗口自动触发 | compaction 摘要块 + 系统提示重注入；**工具定义可能导致压缩失败**（需明确指令防工具调用） | 最近消息原始保留，工具结果可摘要化 |

共性参数维度：token 阈值 / 保留窗口大小 / 工具输出截断比例 / 摘要模型强弱 / 初始上下文重注入。

#### 2.8.2 方法论调研

- **Anthropic 上下文工程**：context rot（token 越多注意力越稀释，质量 > 长度）；**Tool Result Clearing**（工具结果一旦被消费即可清除原始输出）；**Structured Note-taking**（笔记持久化到窗口外按需拉回）；**Sub-agent**（子代理返回 1000-2000 token 精简摘要而非完整过程）
- **MemGPT / Letta**：Main Context（RAM）/ External Context（磁盘）分层记忆；Core Memory 始终在上下文不可压缩；LLM 自编辑记忆（自主决定读写外部存储）
- **LLMLingua**（微软）：小模型 token 级裁剪，20x 压缩性能损失极小；LongLLMLingua 缓解 lost-in-the-middle
- **权重参考**：系统提示 1.0 > 用户指令 0.95 > 工具结果摘要 0.8 > 中间对话 0.5 > 旧工具输出 0.2；双窗口 = 近期窗口完整 + 长期窗口摘要

#### 2.8.3 Yuxi 两级压缩（源码验证，DeepView 直接继承）

关键源码：`backend/package/yuxi/agents/context.py:14-19`（默认参数）、`:20-50`（摘要 prompt）、`:63-104`（工作区文件）、`middlewares/summary.py:296`（`YuxiSummarizationMiddleware(SummarizationMiddleware)`）、`:359-387`（L1）、`:416-495`（L2）、`:610-652`（增量语义）、`backends/composite.py:95-116`（常驻 eviction）、设计文档 `docs/agents/middleware.md:68-90`。

| 机制 | 默认值 | 要点 |
|---|---|---|
| 触发 | 100K tokens（近似计数，刻意不用 provider usage） | 历史全量进上下文，触发后压缩；keep=最近 10 条 |
| **L1 结构精简**（零成本，不调 LLM） | tool args >2000 字符截断；ToolMessage >300 tokens 卸载 | 卸载到 `large_tool_results/{tool}-{sha256[:16]}.txt`，上下文替换为路径 + 近似 token 数 + ≤300 token 预览；**L1 只在模型调用视图上做，不改 state**——checkpoint 永远全量历史，可回溯 |
| **L2 LLM 摘要** | L1 后仍 > 阈值 × 0.4 才触发 | 结构化摘要 prompt：SESSION INTENT / USER REQUIREMENTS / PROGRESS AND DECISIONS / ARTIFACTS AND REFERENCES / NEXT STEPS（保留路径与标识符、跟随用户语言）；**增量语义**：只摘要新增长度，`[summary] + messages[cutoff:]` 重建；摘要调用带 TAG_NOSTREAM 防串流 |
| 常驻 eviction | 工具结果 >3K tokens 就卸载（不止压缩时） | `read_file`/`open_kb_document` 豁免；与压缩时 300 双阈值各司其职 |
| 摘要事件 | `yuxi.context_compression` 流事件 | started/completed/failed 推前端——可平移成我们 WS 事件 |
| 装配位置 | 主 agent 与 SubAgent 各自装配（graph.py:33-84 / 66-103），阈值独立可配 | 参数透出 admin 可配（`summary_threshold` 等全到 UI） |

**Yuxi 缺的（DeepView 需自补）**：跨线程/跨会话长期记忆（`enable_memory` 是未接线的死代码）、加载期窗口（触发阈值以下完全不清理）、agent 主动压缩工具（deepagents `compact_conversation` 未启用）。

#### 2.8.4 DeepView 方案（继承 + 两处补强）

1. **压缩器落在子 agent 服务**（langgraph-api 进程内，跟 Yuxi 同位置），每个 agent 独立阈值；主服务只管编排不感知压缩
2. **参数照抄 Yuxi 已验证默认值**：触发 100K / keep 10 条 / L1 卸载 300 tokens / 常驻 eviction 3K / L2 ratio 0.4；做成可配置，前端设置页透出（Yuxi 同款）
3. **摘要模型**：复用 4 角色 LLM 的轻量档（KEYWORD 档），不新增 LLM 角色（待拍板，见决策表 #17）
4. **补强一：加载期窗口**——续跑/重启时只加载最近 N 条 + 已存 summary 到模型上下文，避免触发阈值前的悬崖式膨胀（Yuxi 是历史全量进）
5. **补强二：压缩事件走 WS 推前端**——`yuxi.context_compression` 语义平移为 `context.compressed` 事件，前端显示"上下文已压缩"状态块（与 4.4 协议同表）
6. **卸载文件**落子服务工作区（working_dir），路径可检索；`_TOOL_RESULT_SAVED_MARKER` 标记去重避免 L2 二次处理
7. 引入时机：M2 后随长会话自然出现，M3 验证参数

### 2.9 Yuxi 深挖补遗（8 路并行调研，2026-08-11）

8 个调研 agent 对 Yuxi 本地源码（D:\code\github\Yuxi-main）分领域深挖，补齐 2.7/2.8 未覆盖的设计。本节省略重复内容，只记新发现。**采纳优先级见 2.9.9**。

#### 2.9.1 server 层与配置体系（路由端点 + 配置 + 横切）

**路由端点清单**（backend/server/routers/，22 个路由）：标注 DeepView 没想到的功能。

| 路由文件 | 没想到的端点 | 价值 |
|---|---|---|
| system_router | GET /system/discovery（能力发现）、GET /system/logs（日志尾部查看）、config/options 动态表单配置 + PUT 单键保存 | 中——配置热更新 + 前端动态表单 |
| auth_router | check-first-run + initialize（**首次运行引导**）、cli/sessions（device flow 授权） | 低——单用户但仍需首次引导 |
| agent_router | thread/{id}/requests 队列快照、requests/continue、requests/{id}/cancel、requests/{id}/steer（**队列优先接替**）、thread/{id}/active_run | 高——多任务队列交互 |
| chat_router | threads/search（**对话全文搜索**，snippet）、thread/{id}/files 线程文件管理、artifacts/save（交付物持久化）、message/{id}/feedback（like/dislike）、image/upload 多模态 | 高——搜索 + 反馈闭环 |
| dashboard_router | stats/tools 工具调用统计（成功率/错误分布/趋势）、stats/agents、feedbacks | 中——可观测性，后置 |
| skill_router | import/export zip、remote 远程 Skill 市场、dependency-options 依赖声明、share-config | 低——Skill 生态后置 |
| 其他 | mcp-servers CRUD+test+refresh、model-providers CRUD+remote-models+status、viewer/filesystem 文件树、mention/search @文件搜索、workspace 树、agent-invocation/call（**OpenAI 兼容同步 API**）+channel（/state /approve slash 命令） | 中——OpenAI 兼容 API 可暴露本地能力 |

**配置体系**：
- 应用配置：pydantic BaseModel（非 pydantic-settings）+ TOML 持久化 + Redis 快照 5s 同步（多进程一致性，我们不需要 Redis，用进程内信号）
- **ConfigOptions 体系**（config/options.py:1-254）：代码定义配置项（key/name/params schema/value）+ 启动幂等同步 DB + **敏感字段脱敏展示** + 环境变量回退——直接可作 DeepView 的模型 API Key 管理
- Model provider：`provider_id:model_id` + 能力声明（chat/embedding/rerank）

**横切**：
- 日志：basicConfig 简单格式 + AccessLogMiddleware（IP/路径/耗时）
- 登录限流：内存滑动窗口 10 次/60s + 429 Retry-After（`main.py:95-128`）
- **lifespan 启动顺序**（lifespan.py:1-133）：DB schema 确保 → 内置 MCP/Skills/默认 Agent 幂等同步 → 配置同步 → 预热 → checkpoint 表创建 → Tasker 启动
- worker：ARQ `max_tries=2, job_timeout=3600s, keep_result=60`
- CLI（yuxi-cli）：remote/login/whoami/status/chat/kb upload|list|files|query|open|find/agent eval

#### 2.9.2 agents 核心与中间件链（BaseAgent/BaseContext/12 中间件/prompt）

**BaseAgent**（base.py）：
- **capabilities 声明**（base.py:106-114）：`capabilities: list[str]`（如 file_upload/files），前端按能力渲染操作按钮——刮削 Agent 可声明 `media_scrape`
- **get_info()**（base.py:133-164）：返回 `{id, name, description, metadata, configurable_items, capabilities}`；`configurable_items` 由 context_schema 动态生成 + 资源选项（tools/kbs/mcps/skills/subagents）从 DB 拉取——**前端设置页免硬编码**
- 三种流式接口：`stream_values` / `stream_messages` / `_stream_input_with_state`（生产主路径：astream_events v3 + 并行 task 收集 subagent routes，事件按 method 分发，Command 内 ToolMessage 归一化暴露）
- checkpointer 三级降级：PG → SQLite → InMemory；`reload_graph()` 热重建

**BaseContext 字段**（context.py:132-328）：除已知 summary 5 参数外，新发现：
- `max_execution_steps=300`（防死循环）、`model_retry_times=2`（可配置重试）
- `system_prompt`（kind="prompt"）、`model`（kind="llm"，每 agent 独立选模型）、`mcps`/`knowledges`/`skills`（动态挂载）
- **metadata 机制**（context.py:330-358）：字段级 `hide` / `auth`（admin/superadmin）/ `configurable` → `get_configurable_items(user_role)` 自动生成配置 schema；`options` 支持 callable 延迟求值
- **workspace agent context**（context.py:63-85）：工作区 `agents/{filename}` 文件（64KB 上限）注入 system_prompt——刮削任务工作区放 SCRAPER.md 注入任务指令

**完整中间件链**（chatbot/graph.py:57-84，顺序即策略）：

| # | 中间件 | 职责 | DeepView 采纳 |
|---|---|---|---|
| 1 | Steer | 队列驱动 run 中断：模型调用前后检查 `should_end_run_for_steer` → `jump_to: end`；仅在无待执行 tool_calls 时跳转（安全边界） | **高**——新任务打断旧任务 |
| 2 | Filesystem（Yuxi 版） | CompositeBackend 挂载 + 大工具结果 evict（已知） | 已知 |
| 3 | Attachment | state.uploads → 注入 system prompt"用户上传了：{name}:{path}，请优先 read_file"；marker 防重复注入 | 中——拖入文件提示 agent |
| 4 | Skills | 已知 | 已知 |
| 5 | SubAgentTask | 已知 | 已知 |
| 6 | Summary | 已知 | 已知 |
| 7 | TodoList | 注入 write_todos + 指导 prompt（任务名 ≤20 汉字） | 高——todo 可视化 |
| 8 | PatchToolCalls | 修复 tool call 格式（deepagents 库内） | 参考 |
| 9 | ModelRetry | 模型调用失败重试（context 读 max_retries，默认 2） | 高 |
| 10 | ImageInputCompatibility | ①OpenAI 兼容：ToolMessage 图片块→HumanMessage 桥接 ②**OCR fallback**：模型拒绝图片（400/415/422+关键词判断）自动发起 ocr_parse_file | 中——海报识别降级 |
| 11 | TokenUsage | model call 后 26 字段 token 快照写 state：llm_input_tokens/context_usage_ratio/remaining/summary_active | 高——token 可视化 |
| 12 | ToolApproval | 已知 | 已知 |

**SubAgent 链差异**（subagent/graph.py:89-103）：无 Steer/ToolApproval；**新增 _SubAgentToolFilterMiddleware**（subagent/graph.py:31-63）：default 模式隐藏 `present_artifacts, ask_user_question, install_skill` + 敏感工具（write_file/edit_file/execute）——子代理无"问用户/装插件"能力；**子 agent 模型继承**（graph.py:450-460）：未配置时沿用父模型

**Prompt 工程**（chatbot/prompt.py）：
- 结构 = `当前日期（上海时区 shanghai_now()）` + 基础 PROMPT + 自定义 system_prompt（prompt.py:53-56）
- 基础 PROMPT 关键段：**`<| 内部执行约束:重要 |>`**——"内部实现细节（工作区路径/工具调用方式）不向用户说明"；文件系统约束（outputs/uploads/workspace 目录规范）；风格规范（专业严谨少 emoji）
- TODO_MID_PROMPT：按复杂程度用 write_todos 记录规划

**MCP 集成**（agents/mcp/service.py:1-672）：
- 工具缓存 `_mcp_tools_cache` key = `server_slug:config_hash`——**配置变化自动失效**
- 工具命名 `mcp__{server_cc}__{tool_cc}`；`handle_tool_error = True` 防 ToolException 击穿
- 单工具级 enable/disable（disabled_tools 列表）；**用户 MCP 禁用 stdio 传输**（安全：不启动本地进程）
- 内置 MCP：mcp-server-chart（@antv 图表生成）
- 启动时代码↔DB 幂等同步（ensure_builtin_mcp_servers_in_db）

**状态设计**（state.py）：`BaseState(AgentState) + artifacts`（merge reducer 保序去重）；**AgentStatePayload**（state.py:25-32）：前端统一 `{todos, files, artifacts, subagent_runs, token_usage}` 序列化；SubAgentRunState：`{id, run_id, subagent_slug, child_thread_id, status(pending/running/completed/failed/cancel_requested/cancelled/interrupted), artifacts, events_url}` + 按 run_id 增量合并 reducer

**模型加载**（agents/models.py:11-101）：三级解析（请求值→fallback→系统默认）；按 provider_type 分发（ChatAnthropic/ChatGoogleGenerativeAI/ChatOpenAI）；**_ToolCallChunkFixChatOpenAI**：流式 tool_call 续片空串 name/id 归一化（规避硅基流动/阿里云百炼 v3 流式缺陷）——**接国内 DeepSeek 时注意**

#### 2.9.3 工具（toolkits）与 backends

**内置工具完整清单**（toolkits/buildin/tools.py）：

| 工具 | 参数 | 用途 | DeepView |
|---|---|---|---|
| web_search | query/count/time_range/sites/block_hosts/content_format | 网络搜索（豆包/Tavily） | **需补充**——刮削世界知识检索 |
| present_artifacts | filepaths | 展示生成文件给用户 | 中——交付物展示 |
| ocr_parse_file | file_path/ocr_engine | OCR 解析 PDF/Office/图片→MD | 低（不引 OCR） |
| ask_user_question | questions | 向用户提问等待回答 | **需补充**——通用 HITL |
| install_skill | source/skill_names | 安装 Skill | 不需要 |

**知识库工具**（toolkits/kbs/tools.py）：list_kbs / get_mindmap / query_kb(kb_id, query_text, file_name) / open_kb_document(kb_id, file_id, **window_size=1800**, line/offset) / find_kb_document(kb_id, file_id, patterns, use_regex, case_sensitive, **max_windows=5, window_size=80**) / search_file / download_kb_file——参数模式：窗口+位置定位防大输出

**backends 设计**：
- 四类：StateBackend（线程内）/ CompositeBackend（跨线程组合）/ SelectedSkillsReadonlyBackend（**只读隔离**，所有写返回 permission_denied）/ ProvisionerSandboxBackend（沙盒）
- CompositeBackend（backends/composite.py:33-93）：**路由前缀映射**（/skills/ → 只读后端）+ 虚拟路径→物理路径映射 + 路径穿越防护
- YuxiFilesystemMiddleware（composite.py:95-116）：wrap_tool_call 拦截大工具结果，豁免 `read_file`/`open_kb_document`
- 工具结果格式约定：`{query, results, response_time}` 或 `{error}` 字符串；工具内 try-except 防异常传播——**建议统一 {status, data, error}**
- 工具加载管线（toolkits/service.py:97-144）：三阶段——builtin 工具 → MCP 工具（并行）→ Skill 依赖工具（门控注册进 ToolNode）

#### 2.9.4 知识域（chunking/parser/检索管线/文档生命周期）

**分块**（knowledge/chunking/）：
- 6 种预设：general（分隔符+naive_merge+硬切保护 1.5x）/ qa（自动识别问答对）/ book（层级标题+hierarchical_merge depth=5）/ laws（法条归一化+tree_merge depth=3）/ semantic（embedding+AgglomerativeClustering，无需预设 k）/ separator
- 参数：chunk_token_num=512、overlapped_percent=0；硬切保护 `GENERAL_HARD_LIMIT_RATIO=1.5`
- **中文标题层级检测**（nlp.py:7-44）：5 组 BULLET_PATTERN（中文数字编/章/节/条、阿拉伯、序号、英文 Chapter、Markdown #）
- chunk 元数据：**start_char_pos/end_char_pos** 字符级定位（chunk→原文回溯，NFO 处理可参考）

**解析**（parser/）：
- 类型路由（unified.py:334-409）：txt/md 直读、docx/xlsx/pptx→**Docling**（MinerU 轻量替代，无 OCR 依赖）→python-docx 兜底、html→**markdownify**、csv→pandas→MD 表格、json 原样包代码块、zip 解包、图片→OCR
- PDF 预检（pdf_utils.py:37-87）：pypdfium2 加密/空 PDF/页槽校验——**入库前文件完整性预检**
- OCR 注册表 7 引擎（rapid/mineru/pp-structure/deepseek_ocr/paddleocr 系列）
- 近似 token 计数（nlp.py:49-55）：CJK 字符 + 英文单词，不调 tokenizer

**检索管线参数**（implementations/milvus.py，LightRAG 侧参考）：

| 参数 | Yuxi 值 | DeepView 参考 |
|---|---|---|
| recall_top_k / final_top_k | 50 / 10 | **召回与终返分离**——LightRAG 召回→本地 rerank→截断 |
| vector_weight / bm25_weight | 0.7 / 0.3 | LightRAG 内置 |
| similarity_threshold | 0.2 | 按需 |
| rrf_k（多路融合） | 60.0 | 标准值 |
| use_reranker 失败 | 回退原始排序 | graceful fallback |

- 管线：召回 50 →（可选图检索 RRF 融合）→ rerank → 截断 10；**来源注释批量查 DB 防 N+1**（_hydrate_chunk_sources）
- RRF：`score = weight / (rrf_k + rank)`

**文档生命周期状态机**（knowledge/base.py:31-38, 256-367）：
```
UPLOADED → PARSING → PARSED → INDEXING → INDEXED
             ↘ ERROR_PARSING（可重试）  ↘ ERROR_INDEXING（可重试）
```
- **乐观锁状态转换** `update_fields_if_status()`（CAS：`UPDATE...WHERE status IN (...)`）防并发
- 重新索引先 `delete_file_chunks_only()` 再重建；CancelledError 处理先 uncancel 再标记
- **统计修复**（base.py:1051-1136）：定期校验 chunk_count/token_count/file_size 一致性（媒体入库校验 LightRAG 与本地一致可参考）

**eval**：precision@k/recall@k/f1@k + LLM Judge 事实一致性 + recall@10 兜底；基准生成（vector/graph_enhanced 两模式）；JSONL 数据集

**问答引用**：`文档 N:` 前缀 + max_docs=5 限制 + "信息不足，无法回答"指令——检索结果带 file_id+source 可点击跳转

#### 2.9.5 services 运行层（Tasker/Run 状态机/队列/SSE）

**Tasker 进程内任务系统**（services/task_service.py:1-516，DeepView 轻任务照搬语义，原语换 anyio 等价物——见 5.7.1 决策 #24）：
- 纯 asyncio（Yuxi 原版）：`asyncio.create_task(_worker_loop())` + worker_count=2 从同一 asyncio.Queue 竞争消费（DeepView 对应：`TaskGroup` + `MemoryObjectStream`）；Task dataclass 12 字段（含 progress/message/payload/result/cancel_requested）
- TaskContext 5 API：set_progress/set_message/set_result/is_cancel_requested/raise_if_cancelled
- **进度节流**（:438-447）：增量 <2% 只改内存不写 DB
- **去重入队** `enqueue_unique_by_payload`（:194-232）：按 task_type+payload 命中复用
- **协作式取消**（:265-276）：cancel_requested 标志 + 任务自身检查点
- **任务级超时**（:354-379）：asyncio.wait + 默认 6h
- **崩溃恢复**（:473-492）：重启后非终态统一标记 failed（"服务重启时任务中断"）
- 终态裁剪 MAX_TERMINAL_TASKS=200；模块级单例

**Run 生命周期状态机**：
```
pending → running → cancel_requested → {completed | failed | cancelled | interrupted}
```
- **cancel_requested 两阶段取消**（关键设计）：HTTP 写 DB + 发布取消信号 → worker 后台监听（0.2s 轮询）→ 流消费 asyncio.wait 竞速（下一条事件 vs 取消信号）→ CancelledError 路径 → 终态
- 终态集合 AGENT_RUN_TERMINAL_STATUSES = (completed, failed, cancelled, interrupted)
- DB 写入时机：运行中消息**不写 run 表**（写事件流），仅终态写 run；`FOR UPDATE` 行锁（SQLite 用 WAL+事务模拟）；**终态幂等检查**（已终态直接返回，不二次变更）
- **部分唯一索引** `uq_agent_runs_one_active_per_thread`：同线程仅一个活跃 run（WHERE status NOT IN 终态）
- 崩溃恢复两层：Tasker 标记 failed + `recover_pending_dispatches()` 重启后重新派发 queued
- LLM 重试（Yuxi 原版）：ARQ max_tries=2；可重试异常（OperationalError/ConnectionError/TimeoutError）vs 不可重试（NonRetryableRunError）
- **run 完成后 finally 自动拉起下一个排队请求**（FIFO 闭环）

**请求队列**（agent_request_queue_service.py:1-891）：
- 状态：queued → dispatched → cancelled/rejected/failed；三策略：**enqueue（FIFO）/ reject（不能立即执行就 409）/ steer（优先接替，永远排第一）**
- FIFO 排序：`(queue_policy != 'steer') asc, created_at asc, id asc`
- 队列位置 O(1) COUNT；排队中取消先 SELECT FOR UPDATE；dispatched 拒绝取消（409 + run_id）
- Request SSE：1s 轮询、位置变化才发事件、15s 心跳、30min 上限

**SSE 事件完整类型**：metadata / messages（loading 流式+批量 items）/ end / error（error_type+retryable）/ interrupt（reason+questions/approval）/ custom.yuxi.agent_state（{todos, files, artifacts, subagent_runs}）/ custom.yuxi.warning / custom.yuxi.context_compression / custom.yuxi.{status}
- **ChunkedEventWriter**（run_worker.py:104-141）：按 thread 分 buffer，**100ms 或 512 字符批量 flush**；tool_call 立即 flush；loading token 合并为 {items} 批量
- **终端事件补发兜底**（agent_run_service.py:976-996）：事件流过期时按 DB 状态补发 end——防止前端等不到结束

**会话管理**（conversation_service.py:1-1017）：软删除（status=deleted）、is_pinned、会话级 tool_approval_mode（extra_metadata）、**消息全文搜索**（snippet 半径 72/长度 180/每线程 2 条）、**附件→LangGraph state 双向同步**（graph.aupdate_state() 写 uploads）、模型快照在 run 创建时固化（覆盖>agent 配置>默认）

**SubAgent 编排**（subagent_run_service.py）：**防无限递归**（subagent 不能建 subagent）；child_thread_id 确定性哈希；runtime payload `{tool_call_id, subagent_name, parent_thread_id, file_thread_id, skills_thread_id}`；**取消级联**（先逐个取消子 run 再父 run，并发发布）

#### 2.9.6 数据模型与仓储层（15 仓储 + 表 schema）

**核心表设计**（models_business.py，可直接翻译 SQLite DDL）：

**agent_runs**（:831-917）：id TEXT PK / conversation_thread_id / agent_slug / uid / status / **request_id UNIQUE 幂等键** / source / channel / external_id / created_by_run_id / run_type（chat/resume/subagent）/ input_message_id / output_message_id / **last_event_id**（事件游标）/ input_payload JSON / **error_type + error_message 分离** / started_at / finished_at

**messages**（:356-403）：**完整存储不截断**（content TEXT）+ role / **message_type（text/tool_call/tool_result）** / token_count / extra_metadata JSON / image_content（Base64）/ run_id / request_id / **delivery_status**（独立于业务状态）

**tool_calls 独立表**（:405-434）：message_id / **langgraph_tool_call_id 幂等** / tool_name / tool_input JSON / tool_output / status（pending/success/error）

**conversations**（:279-316）：id INTEGER PK + **thread_id UNIQUE 双 ID** / title / status（active/archived/deleted 软删）/ is_pinned / extra_metadata JSON

**tasks**（:704-744）：progress REAL + message TEXT + **cancel_requested 标志位**（信号非状态）+ payload/result JSON 分离 + started_at/completed_at

**agent_run_requests**（:920-997）：自增 id 仅 FIFO 排序 + request_id UNIQUE / queue_policy / status / input_payload JSON

**config_options**（:687-701）：key UNIQUE / **params（schema 定义）+ value（当前值）JSON 分离**——全局配置模板
**agent_envs**（:140-159）：JSON 列存字典配置（API key 类）——**注意**：Yuxi 的 env 也存 DB，DeepView 建议 OS keyring，见 2.9.8

**并发模式**（SQLite 对应物）：
- `SELECT FOR UPDATE` 行锁 → SQLite WAL + 事务内先读后写
- `SKIP LOCKED` 乐观领取 → **租约锁**：`locked_until + lock_token + attempt_count`（UPDATE...WHERE status='pending' AND (locked_until IS NULL OR < now)），重试退避：attempt1→5s, 2→30s, ≥3→failed
- **终态幂等检查**（返回是否真正变更 bool）；CAS `UPDATE...WHERE status IN (...)`；**flush() 不 commit**（service 层决定事务边界）
- **Schema 演进用 ALTER TABLE 运行时迁移**（无 Alembic，manager.py:486-900）
- 关键索引：部分唯一索引防并发 run、FIFO 队列复合索引（uid, agent_slug, thread, status, created_at, id）

#### 2.9.7 前端 web 架构（Vue3，跨框架提取）

**页面**：/agent 主聊天三栏（侧栏+聊天+右状态面板）/ agent-manage / workspace / extensions（知识库+技能+MCP）/ knowledgebase 详情 / dashboard / login

**store 划分原则**（DeepView zustand 参考）：
- 数据 vs UI 分离（chatThreads vs chatUI）；**运行时状态放组件本地**（threadStates[threadId] = isStreaming/activeRunId/pendingInterrupt/queueSnapshot/queuedRequests，不进全局 store，按 conversationId 分片）
- persist 精确控制（只持久化 selectedAgentId、sidebarCollapsed）

**审批交互 UI**（★DeepView HITL 核心参考）：
- **审批卡片 slide-up 内嵌输入区上方**（非居中弹窗，chat 不阻断）、输入区 inert
- 工具审批逐步圆点（tool-progress-step 逐个 approve/reject）+ 参数可折叠（摘要+展开）
- **pendingInterrupt 持久化**：刷新后可恢复弹出
- **ToolApprovalModeSelector**：输入框旁下拉切换「请求审批」/「完全信任」，持久化到 thread metadata

**排队 UI**：queued-request-panel 显示排队列表（位置/取消/steer 引导/继续队列）；queueSnapshot 状态机 `idle → running → paused → interrupted`

**useStreamSmoother**（useStreamSmoother.js:1-458）：rAF 自适应排速流式渲染，维护 content/reasoning/toolCallArg buffers，按 avgChunkChars/avgIntervalMs 算排放速率，overflow 加速——**避免 token 涌入卡顿**

**GraphCanvas**（@antv/g6，React Flow 可映射）：节点大小按 degree 缩放（`Math.min(15+deg*5, 50)`）、颜色按类型 hash、**聚焦模式（点击只看邻居子图）**、关键词高亮数组、实体/关系类型统计面板（颜色+计数）、二次曲线边+箭头、暗色主题重渲染

**其他**：推理过程折叠（Thinking 动画→可展开）、工具调用分组折叠（活跃自动展开+状态标签+图标映射表 30+）、消息与工具调用交替渲染（displayItems）、TaskCenterDrawer 任务中心（segmented 筛选+状态指示器+取消按钮、hasActiveTasks 驱动自动轮询）、线程搜索模态框、子 agent 弹窗、文件面板拖拽调宽、token usage 堆叠条形图面板

#### 2.9.8 横切能力与杂项

- **图片处理管线**（utils/image_processor.py）：EXIF 方向修正→透明像素白底合成→缩略图→智能压缩（质量递减+缩放）；图片进 LLM = base64+mime_type
- **消息反馈系统**：like/dislike + 原因 → 聚合面板——**刮削质量改进闭环，建议采纳**
- **API key 安全管理**（utils/auth_utils.py）：`secrets.token_hex(24)` + `yxkey_` 前缀 + **SHA-256 hash 存储**（DB 只存 hash）+ 前端只显前缀；JWT HS256 7 天 + issuer/audience；生产强制非默认密钥——DeepView 存 API key 用 OS keyring（Tauri）或 hash 化
- **URL 安全抓取**（knowledge/utils/url_fetcher.py）：SSRF 防护（私有 IP/loopback 检测）+ **每跳重定向验证** + 内容类型白名单 + 10MB 限制 + 手动重定向
- **Langfuse 可观测性**：run 级 trace 关联 user/session/request + 反馈同步评分——后置（详见 2.9.11 调研）
- **Dashboard 统计**：用户活跃/工具调用（成功率/错误分布/趋势）/知识库/反馈四维
- 运行时配置多进程同步（Redis 5s 拉取，我们不需要）
- **docs/ 未读文档**（30 个中 2 个已读）：tools-system、subagents-management、skills-management、sandbox-architecture、mcp-integration、agents-config、design、configuration、api-key-integration、roadmap、model-config、agent-evaluation 等——需要时可定向精读
- 测试三层：unit → integration → e2e（真实 agent 运行 + SSE 消费）
- 未深入（低价值）：通知系统、对话导出、cron 调度（不存在）、agent 模板市场

#### 2.9.9 采纳优先级汇总（DeepView 权衡参考）

| 优先级 | 发现 | 落点 |
|---|---|---|
| ★ 立即采纳 | Run 状态机 + 两阶段取消 + 部分唯一索引（2.9.5/2.9.6） | 5.7 main.db schema |
| ★ 立即采纳 | Tasker 四件套：进度节流/去重入队/协作取消/超时（2.9.5） | 5.7 tasks 表 + 主服务实现 |
| ★ 立即采纳 | 表 schema：agent_runs/messages/tool_calls/config_options（2.9.6） | 5.7 已落实 |
| ★ 立即采纳 | 审批卡片内嵌输入区 + pendingInterrupt + 模式选择器（2.9.7） | 7.6 + 9.4 |
| ★ 立即采纳 | 日期注入 + 内部执行约束段 prompt + max_execution_steps + ModelRetry（2.9.2） | 子 agent 装配（实现计划） |
| ☆ 高价值 | Steer 中断、TokenUsage 快照、AgentStatePayload、tool_calls 独立表 | M2/M4 |
| ☆ 高价值 | 检索参数参考（recall/final/rrf_k=60）+ 文档状态机 + 统计修复（2.9.4） | M3/M5 |
| ☆ 中价值 | 消息反馈闭环、对话全文搜索、useStreamSmoother、GraphCanvas 交互模式 | M5/M6 |
| ◇ 后置 | Langfuse/可观测性（评估结论见 2.9.11，路径待拍板 #20）、Dashboard 统计、web_search 工具、OpenAI 兼容 API、URL 抓取 | 打磨期 |
| ★ 立即采纳 | 子任务三层感知：流识别即时上屏 + 生命周期卡片 + 子线程详情抽屉（2.9.10） | 9.6 + M2 |

#### 2.9.10 子任务前端感知机制（Yuxi 源码确认，2026-08-11）

**核心结论**：Yuxi 的子任务绝不黑盒——三层感知 + 一个前端复刻哈希算法。后端 `subagent_runs` 只在 task 完成时写入 agent_state，**"运行中"感知全靠前端从主线程流里主动识别**。

**三层感知**：

| 层 | 实现 | 机制 |
|---|---|---|
| ① 聊天流内生命周期卡片 | SubagentLifecycleTool.vue（398 行） | `subagent_start/status/events/cancel/await` 5 工具各有专用卡片：中文标题 + 状态徽章（等待中/运行中/已完成/失败/取消中，彩色）+ 展开显示 meta 网格（run_id/thread_id/last_seq/事件条数）+ **progress 消息列表**（子线程实时进展：assistant_message 消息/assistant_reasoning 思考/tool_call 工具）+ 事件列表（seq+事件名）+ 最终结果 markdown |
| ② 右侧状态面板 subagents 区块 | AgentChatComponent.vue:529-592, 1603-1687 | **runningSubagentRunsFromStream**：检测主线程流中的 subagent_start 工具调用 → `makeChildThreadId` 前端复刻 sha256 推算 child_thread_id → 即时上屏"运行中"条目（不等后端事件）；`displaySubagentRuns` 合并后端已完成条目 + 前端运行中条目；点击条目 → openSubagentThread 开弹窗 |
| ③ 子线程详情弹窗 | SubagentThreadModal.vue（503 行） | 打开时：run 已终态 → `getAgentHistory` 拉持久化消息；**运行中** → `getAgentState(childThreadId, {includeMessages: true})` 拉 LangGraph checkpoint 当前消息（tool 结果嵌入 AI 消息）+ `streamAgentRunEvents(runId, '0-0')` **从 0 重放该 run 事件流** → 复用同一个 handleStreamChunk + streamSmoother 渲染完整子线程聊天视图（消息/工具/推理/审批）→ 增量实时续流；事件按 thread_id 路由过滤（routeChunkThreadId，vue:327-336） |

**前端复刻哈希**（utils/subagentThread.js）：`child_thread_id = "subagent_" + sha256("{parent}:{slug}:{tool_call_id}")[:55]`——纯 JS 复刻（crypto.subtle + 兜底实现），因为子 agent 由 graph.ainvoke 独立调用、流式事件不带 tool_call_id，前端必须自算才能把子任务关联到父工具调用。

**干预能力**（Yuxi 的局限）：`subagent_cancel` 工具（父 agent 驱动）+ run 取消级联（`request_cancel(cascade_children=True)` 先子后父）；**弹窗和面板无直接取消按钮**——用户要干预得发消息让父 agent 调 cancel 工具。DeepView 需补齐直接干预入口（见 9.6）。

#### 2.9.11 可观测性调研：Langfuse / Phoenix / LangSmith（2026-08-12）

**Langfuse 是什么**：开源 LLM 工程平台（MIT 协议，YC W23，2026-01 加入 ClickHouse 家族，16k stars）——对标 LangSmith 的开源替代，Open WebUI/Langflow/LibreChat/LangChain-Chatchat 等大量项目使用。

**六大功能**：

| 功能 | 用途 | 对 DeepView 价值 |
|---|---|---|
| Tracing | LLM 调用链追踪（prompt/响应/token/耗时），可下钻 retrieval/embedding/agent 动作 | 中——已有内建可视化（/state + 三层感知 + token_usage） |
| Prompt Management | 集中管理 + 版本化 prompt，客户端缓存无延迟 | 低——单人项目 |
| **Evaluations** | **LLM-as-judge / 代码评估器 / 用户反馈 / 手动标注** | **高——最缺的能力**（自研 eval 已砍，见 2.7.5） |
| **Datasets** | 测试集 + 基准，支持 LangChain 集成 | 高（配合 eval） |
| LLM Playground | 坏 trace 一键跳转试 prompt | 中 |
| 完整 API | OpenAPI + Python/JS SDK | 中 |

**部署形态（决定性问题）**：
- **Langfuse Cloud**：免费额度，SDK 上报，零本地运维——代价是 trace 数据出网（个人媒体数据可接受）
- **Self-Host**：docker compose 5 分钟起——但底层是 **ClickHouse + Postgres + Redis + S3** 全家桶——**与 DeepView 零运维原则直接冲突**（同 2.7.5 弃 Milvus/Neo4j 的理由），单机桌面应用为一个 eval 功能背 4 个存储系统不划算

**集成成本**：`pip install langfuse` + LangChain `CallbackHandler` 一行接入（我们即 langchain>=1.3.9）——代码层面几乎零成本。

**Yuxi 用法**（见 2.9.8）：run 级 trace 关联 user/session/request + like/dislike 反馈同步 boolean score + 异步取 trace URL + CLI eval 跑 Langfuse 数据集。前提是 Yuxi 本就用 docker 全家桶，加 Langfuse 无额外运维成本——**此前提我们不成立**。

**三方案对比**：

| 方案 | 部署 | 零运维 | 数据 | 关键能力 |
|---|---|---|---|---|
| Langfuse Cloud | 免部署（SDK 上报） | ✅ | 出网 | 全功能（tracing/prompts/evals/datasets/playground） |
| **Arize Phoenix** | `pip install` 本地进程 | ✅ 最强 | 完全本地 | tracing + evals，OpenInference 语义，LangChain 集成良好 |
| LangSmith | SaaS | ✅ | 出网 | LangChain 官方集成最深，但闭源、免费额度小 |
| DeepView 内建 | 无 | ✅ | 本地 | 实时可视化已有（/state + 面板），**缺 eval/历史对比** |

**评估结论**：
1. **现在不引入**（2.9.9 维持 ◇ 后置）——自托管违背零运维；内建可观测性已覆盖 90% 调试需求；单人项目 console + LangGraph 事件够用
2. **引入路径二选一**（决策表 #20 待拍板）：
   - **首选 Phoenix**：pip 本地进程、完全离线零运维、LangChain 集成 + LLM-as-judge——与零运维原则最契合
   - **备选 Langfuse Cloud**：功能最全（datasets + evals + playground），零本地运维，代价是 trace 出网
3. **最有价值的引入时机**：M3 检索质量调优 / M5 反馈闭环时——**要的是 eval 能力（LLM-as-judge 评估刮削/检索质量），不是 tracing**

## 3. 总体架构：四层服务 + 前端壳

```
│ 前端 (Tauri 2 + React 19 + TS + Vite)                 │
│  只跟主 agent 服务通信：POST 命令 + WS 订阅 + /state    │
└───────────────┬──────────────────────────────────────┘
                │ :8000
┌───────────────▼──────────────────────────────────────┐
│ ① 主 agent 服务  (FastAPI：REST + WS + /state)        │
│   create_deep_agent + AsyncSubAgentMiddleware         │
│   工具暴露：只有 start/check/update/cancel/list 5 个   │
│   → 只做编排，不知道 TMDb/Bangumi 是什么              │
│   数据：main.db（sessions + checkpoints + async_tasks）│
└───────────────┬──────────────────────────────────────┘
                │ LangGraph SDK (Agent Protocol)
┌───────────────▼──────────────────────────────────────┐
│ ② 子 agent 服务  (langgraph-api create_app(), :8100)  │
│   create_deep_agent + mcp_servers=[media-mcp]         │
│   工具暴露：只有 MCP 刮削工具                         │
│   数据：worker.db（task threads checkpoints）          │
└───────────────┬──────────────────────────────────────┘
                │ streamable-http MCP
┌───────────────▼──────────────────────────────────────┐
│ ③ MCP server  (FastMCP, :8200)                        │
│   工具：tmdb_search/detail、bangumi_search/detail、   │
│   characters、media_upsert、media_meta、graph_query、 │
│   graph_lookup_global（图工具封装 LightRAG）           │
│   数据：media.db（刮削工作流 + 精确索引 + 缓存）        │
└───────────────┬──────────────────────────────────────┘
                │ REST（图谱读写 + 检索 + RAG 问答）
┌───────────────▼──────────────────────────────────────┐
│ ④ LightRAG server（lightrag-server, :8300）           │
│   知识图谱（entity/relation）+ 向量 + 双级检索 + 问答  │
│   数据：working_dir 文件存储（GraphML + JSON，单写者） │
└──────────────────────────────────────────────────────┘
```

**分层原则（工具可见性隔离）**：每层只暴露该暴露的工具——主 agent 看不到刮削工具，子 agent 看不到图库实现，TMDb/Bangumi 藏在 MCP 后面，图谱/检索/问答藏在 LightRAG 后面。

**各服务独立存储**，避免多进程 SQLite 锁竞争；LightRAG server 是图谱唯一写者，其余服务只经 REST 访问。

### 3.1 图谱 + RAG 后端：LightRAG（2026-08-11 用户决策）

**决策**：不自研图谱存储/检索层，采用 LightRAG（HKUDS/LightRAG）作为**媒体知识库唯一后端**——知识存储（文档/实体/关系/向量）+ 全文/语义检索 + RAG 问答全部由它承担。理由：自研图谱 + 混合检索耗时且效果不可控（Yuxi 自研图谱链路的经验也印证）；**LightRAG 本地运行零运维**——默认文件存储（GraphML + JSON）即够，个人规模不必部署 Neo4j/Milvus/PostgreSQL 这类重组件，这是它优于 Yuxi 全家桶方案的关键；且 `insert_custom_kg` 可喂结构化数据绕过 LLM 抽取。

**分工边界（LightRAG 不是"整个后端"，不替代编排层；但知识域全部归它）**：

| 模块 | 归属 | 说明 |
|---|---|---|
| 会话 / 编排 / 任务状态 / HITL | 自研（主服务 + 子 agent） | DeepAgents + checkpointer + WS，LightRAG 无此概念 |
| 任务子图隔离 + merge | 自研（**写入时机控制** + media.db 记录归属） | 隔离 = 未确认内容不写入 LightRAG；merge = 用户确认后写入 |
| 媒体知识（详情/角色/声优/关系/剧集/简介/标签） | **LightRAG** | 文档 + `insert_custom_kg` 双形态，唯一知识存储 |
| 图谱 + 向量 + 双级检索 + RAG 问答 | **LightRAG** | `query`（local/global/hybrid/mix/naive） |
| 媒体精确索引（按 id 查重/取回） | media.db 薄索引表 | 适配约定：LightRAG entity_id = hash(media_id)，薄表做 id ↔ 双源 id 映射 |
| 文件事实 / 任务状态 / 候选确认 / API 缓存 | media.db | 刮削工作流数据，非知识 |

**数据管道（单写，无双写问题）**：LightRAG 是知识唯一写入端。刮削确认（HITL）→ 适配器转 `insert_custom_kg`（entity/relation/chunks，source_id = media_id）或文档形态喂入 → 同时 media.db 记任务归属与状态。删/改媒体 = 按 source_id 文档单位删除 + 重建（LightRAG 原生支持）。

**关键技术点**：
- 部署用**独立 lightrag-server 进程**（:8300）：默认文件存储对个人规模足够（几千媒体 → 几万实体，全量内存无压力）；百万级才需切 PG/Milvus，届时再议；锁版本 `lightrag-hku`
- 4 角色 LLM 配置一次：QUERY=DeepSeek（问答）、KEYWORD=轻量模型、EXTRACT 占位（不走文档抽取）、embedding=本地 bge-m3；rerank=bge-reranker-v2-m3（可选）
- 隔离由"写入时机"保证：任务进行中数据只存在于主服务/子 agent 会话，LightRAG 仅含已确认知识
- 图谱浏览（前端子图视图）用 `query` 图检索结果 + media.db 薄索引拼装

**M0 待验证项**：①部署形态独立进程（已定）②隔离分工按上表（已定）③media.db 保留工作流 + 薄索引（已定，见 5.2）④lightrag-server 启动 + insert_custom_kg 通路（M0 spike 验证）

## 4. 协议设计

### 4.1 传输层决策：方案 C（2026-08-11 用户拍板）

**决策**：REST 全部经 `tauri-plugin-http` v2（Rust 侧代理执行）；WS 前端原生 WebSocket 直连；**不引入 tauri-plugin-websocket**。

**调研事实**（官方文档/仓库确认）：
- `tauri-plugin-http` v2 官方维护（plugins-workspace v2 分支）：前端 `fetch` 实际由 Rust 侧执行 → **无 Origin/CORS 限制，直接消除 CORS 高风险项**（风险清单降级）
- `tauri-plugin-websocket` 有 v2 官方版本（2025-11 更新）但功能薄：仅 connect/send/addListener/disconnect——重连/心跳/订阅路由/seq 续传都要自己外层实现
- 官方文档警告（calling-frontend）：**事件系统不适用于低延迟/高吞吐**（emit 底层 eval JS，payload 恒为 JSON 字符串）；**Channel 才是有序高速通道**（专为 WebSocket 消息设计）；而 plugin-websocket 的 addListener 转发底层用的是事件系统——高频 token 流经它有性能风险

**对比结论**（A 前端直连 / B Rust 持连接+Channel 转发 / C 混合）：

| 维度 | A | B | C（采纳） |
|---|---|---|---|
| REST | 直连（要 CORS 白名单） | 直连或 plugin-http | **tauri-plugin-http**（零 CORS） |
| 高频 token 流 | 零中间跳 | 事件转发性能风险/需自包 Channel | 零中间跳 |
| 协议状态机 | TS，与 UI 状态同居 | Rust tokio + 另一套 TS 同步（心智成本高） | TS |
| HITL 双向 | 最短路径 | send 两跳 | 最短路径 |

**落地要点**：
- REST：前端统一封装 `fetch` 于 `@tauri-apps/plugin-http`（抽象层可注入，dev 模式退化为直连）
- WS：`new WebSocket('ws://localhost:8000/ws')`，协议状态机（subscribe 路由/30s 心跳/seq 续传/指数退避重连）在 TS 通信层实现，与 zustand/streamSmoother/pendingInterrupt 同居
- M1 验证项：生产 CSP `connect-src` 放行 `ws://localhost:8000`；plugin-http JSON 响应行为；dev 模式（Vite:1420）后端 CORS 白名单

### 4.2 核心原则

- **前端不存最终状态，后端不推全量大快照**：前端消费增量与全量小块，权威状态在数据库（"不推快照"仅指不推全量大快照，state.snapshot 全量小块属增量范畴，见 4.4/4.5）
- **DB 是权威，WS 只管增量，永远自洽**：断线重连后拉 `/state` 全量对齐
- 事件序列号用 WS 帧内的递增 id，前端去重

### 4.3 端点清单

```
POST   /api/sessions                      # 创建会话 → thread_id
POST   /api/sessions/{id}/runs            # 发消息，agent 后台执行；HITL 恢复复用本端点：带 resume 载荷 + created_by_run_id（Yuxi 模式，见 7.6）
GET    /api/sessions/{id}/state           # 权威状态: {messages, todos, async_tasks, files}
GET    /api/subagents/{child_thread_id}/state   # 子线程详情：checkpoint 消息（含 tool 结果）+ 状态（Yuxi 模式，见 2.9.10）
DELETE /api/runs/{run_id}                 # 取消执行（cascade_children=True 级联取消子 run，见 2.9.5）
POST   /api/runs/{id}/response            # HITL 恢复（已废弃——HITL 恢复统一走 runs 端点带 resume 载荷，见 7.6）
POST   /api/tasks/{id}/merge              # 任务合并：insert_custom_kg 写入 LightRAG（用户确认）
GET    /api/tasks/{id}/media              # 任务媒体归属查询（media.db task_media）

# 队列管理（Yuxi agent_router 平移，见 2.9.1）
GET    /api/sessions/{id}/requests        # 排队快照（位置/策略/状态）
POST   /api/requests/{id}/cancel          # 取消排队请求（dispatched 拒绝 409）
POST   /api/requests/{id}/steer           # 排队请求优先接替（steer 永远排第一）
POST   /api/sessions/{id}/queue/continue  # 继续被暂停的队列
GET    /api/sessions/{id}/active_run      # 当前活跃 run（前端恢复订阅用）

WS     ws://localhost:8000/ws             # 全局单连接，subscribe 路由多会话（协议见 4.4）
```

### 4.4 WebSocket 协议 v2（Yuxi 事件模型平移，2026-08-11 定稿）

**选择 WS 而非 SSE 的理由**（不变）：①多会话路由（单连接 + subscribe 路由）②HITL 双向交互 ③FastAPI 原生支持

**① 统一信封**：所有帧 `{seq, type, session_id, payload, ts}`——seq 每连接递增；服务端内存环形缓冲（上限 N 条 + TTL 2h）；重连后 `subscribe {session_id, last_seq}` 从 last_seq+1 补发（Yuxi Last-Event-ID 续传语义，WS 取代 Redis Stream）

**② 事件类型全集**（服务器→客户端）：

```
run.queued          {position}                    # 排队位置（变化才推，O(1) COUNT）
run.started         {run_id, request_id}
run.cancel_requested                              # 两阶段取消：已受理，agent 安全点协作终止
message.delta       {chunk} | {items:[...]}       # token 打字机；批量合并帧（100ms/512 字符，tool.call 立即推，Yuxi ChunkedEventWriter 语义）
tool.call / tool.result                           # 工具生命周期（驱动前端生命周期卡片，见 9.6）
state.snapshot      {todos, files, async_tasks, subagent_runs, token_usage}   # 全量小块覆盖（Yuxi agent_state 模式，见 4.5）；注：DeepView 原生字段是 deepagents 的 async_tasks，subagent_runs 为其 UI 视图（由主服务转换，前端只消费 subagent_runs）
interrupt           {kind: tool_approval|question, action_requests[], questions[], reason}
run.done            {status, error_type?, error_message?, retryable?}
context.compressed  {status: started/completed/failed}
warning             {message}
pong
```

（客户端→服务器）：

```
subscribe {session_id, last_seq?} | unsubscribe | ping
cancel_run {run_id, cascade?: bool}               # 取消（级联先子后父）
cancel_queued {request_id} | steer {request_id} | continue_queue
```

**③ 心跳与重连**：客户端每 30s ping，后端 3 次未收到判定死连接；重连指数退避 → 重新 `subscribe(last_seq)` 补发 → 拉 `/state` 全量对齐 → 恢复前端 pendingInterrupt（7.6 模式）

**④ 取消两阶段**（Yuxi 2.9.5 平移）：`cancel_run` → 后端写 cancel_requested + 推 `run.cancel_requested` → agent 在安全点协作终止（Steer 语义）→ `run.done {status: cancelled}`；级联 = 先取消子 run 再父 run

**⑤ 子线程订阅**（9.6 三层感知）：抽屉打开 = `GET /api/subagents/{child}/state` 拉 checkpoint（无需 0 重放）+ WS `subscribe {session_id: "child:{child_thread_id}"}` 增量订阅

**⑥ 审批**：interrupt 载荷 = 候选列表（questions 型，对应 ask_user_question）/ 门控（tool_approval 型，action_requests+review_configs）；用户提交复用 runs + resume + created_by_run_id（7.6）；已恢复 run 的旧审批提交 → 409 run_interrupted 前端识别（toolApproval.js:87-88）

### 4.5 实时机制：watcher 检测 + 全量小块快照（不做事件翻译层）

- 后端起 per-session watcher：每个 superstep 后对比 checkpoint 里 `todos`/`async_tasks` 差异（检测用 diff，推送用全量）
- **推送载荷 = 全量小块**（todos/files/async_tasks/subagent_runs/token_usage 均为小结构，其中 subagent_runs 是 async_tasks 的 UI 视图转换，见 4.4）——比 v1 的"变更片段"更简单：天然幂等、前端直接覆盖、重连无需对账（决策 2026-08-11，Yuxi agent_state 模式确认）
- token 打字机：`astream_events` 的 `on_chat_model_stream` 推 `message.delta`（丢了由 /state 兜底）

## 5. 数据库设计：刮削工作流库（media.db）+ LightRAG 知识库

### 5.1 设计思想

**知识域与工作流域分离**：媒体知识（实体/关系/简介/向量）全部存 LightRAG（唯一知识后端，本地零运维）；media.db 只存刮削工作流数据（文件、任务、确认状态、缓存、精确索引）。

**git 分支模型（隔离靠写入时机）**：总库 = LightRAG（main 分支，只含已确认知识）；任务进行中的数据 = feature 分支（存在主服务/子 agent 会话与 media.db 任务状态里，**未确认绝不写入 LightRAG**）。

```
任务启动 → 从 LightRAG 检索已有知识（种子信息，graph_lookup_global）
              │
              ▼
      子代理刮削（media.db 记任务归属，LightRAG 不动）
              │
              ▼
    任务完成 → 用户确认 → 写入 LightRAG（去重/冲突解决）→ 标记任务 merged
```

### 5.2 表设计（media.db，MCP server 持有；知识本体在 LightRAG）

```sql
-- 刮削工作流
task_runs(task_id, title, status, created_at, merged_at)
task_media(task_id, media_id, role)          -- seed(检索引用) | scraped(新建) | linked(引用)
scrape_candidates(task_id, file_path, status, payload JSON, confirmed_at)
  -- HITL 候选确认状态（pending/confirmed/rejected）
media_files(media_id, path, size, mtime)     -- 文件事实（一媒体多文件）

-- 精确索引（薄表：id ↔ 双源 id ↔ 标题，供查重与取回；entity_id = hash(media_id) 对齐 LightRAG）
media_index(id, type, source, source_id, title, name_cn, year)
  -- 媒体唯一：UNIQUE(type, source, source_id)

-- 缓存
media_cache(provider, query_key, response JSON, ttl)  -- API 响应缓存
```

> 修订说明：原自研 nodes/edges 图谱、vec_nodes 向量、media_relations 关系表、FTS5 全文索引**全部移除**——知识域（图谱/向量/全文/关系/检索）由 LightRAG 承载（见 3.1）。精确匹配用普通 B-tree 索引即可，不需要 FTS。5.6 节保留作背景调研。

### 5.3 图谱域（LightRAG 的 entity/relation 映射，`insert_custom_kg` 输入）

| 节点类型 | 属性 |
|---|---|
| anime / movie / tv / manga | 原名、中文名、年份、评分、简介、海报 URL、source_id(tmdb/bgm) |
| season / episode | 季号/集号、标题 |
| character | 角色名、简介 |
| person | 声优/演员/导演/编剧/音乐 |
| studio | 制作公司 |
| series | 系列/原作 |
| tag | 标签/类型 |

| 边类型 | 语义 |
|---|---|
| part_of | 剧集→媒体、季→媒体 |
| has_character + voiced_by | 媒体→角色→人物（Bangumi 图谱核心价值） |
| produced_by / directed_by / written_by | 媒体→工作室/人物 |
| belongs_to | 媒体→系列 |
| has_tag | 媒体→标签 |
| related_to | 媒体↔媒体（Bangumi subject_relations） |

映射约定：entity_name = 规范化媒体名、entity_id = hash(media_id)（与 media.db 薄索引对齐）、description = 原名 + 中文名 + 类型 + 简介融合文本（决定检索质量，M3 验证）。

### 5.4 图查询 API（MCP server 进程内的 REST 路由 + MCP 工具，全部走 LightRAG）

**进程归属（2026-08-11 澄清）**：以下 REST 端点挂在 **MCP server 进程（:8200）**——FastMCP 基于 Starlette，可同时挂载普通 REST 路由，与 MCP 协议共存于一个进程。**前端不直连**：保持"前端只跟主 agent 服务通信"原则，主服务加 3 个薄代理端点（`/api/graph/*` 透传到 :8200），图浏览数据通路 = 前端 → 主服务代理 → MCP server → LightRAG。

```
# MCP server 进程内（:8200），主服务 /api/graph/* 透传
POST /api/graph/search {q, type}          # LightRAG query（naive 全文 / local 实体 / hybrid）
POST /api/graph/semantic-search {text}    # LightRAG query（local/global/mix，语义 + RAG 问答）
POST /api/graph/lookup-global {text}      # LightRAG 全库检索（graph_lookup_global 工具）
GET  /api/media/{id}                      # media.db 薄索引精确取回（查重/详情拼装）
```

- 子 agent 工具 `graph_query` = 上述 LightRAG 检索封装（经 MCP 协议调用，非 REST）；`media_meta` = media.db 薄索引
- REST 路由与 MCP 工具职责对应：同一套 LightRAG 调用，两个通道（子 agent 走 MCP / 前端图谱视图走 REST 代理）
- 无自研图遍历/向量/FTS——知识域检索全部交给 LightRAG

### 5.5 Agent 隔离策略

- **隔离 = 写入时机**：任务进行中的刮削数据不写入 LightRAG（只在主服务/子 agent 会话与 media.db 任务状态中），已确认知识才入库——agent 天然看不到未确认内容
- `graph_query` / `graph_lookup_global` 都是查 LightRAG（已确认知识）；任务内专属中间数据经 `media_meta`（media.db）获取
- 合并流程：任务完成 → 用户 UI 确认（HITL）→ 适配器 `insert_custom_kg` 写入 LightRAG（source_id 幂等去重）→ media.db 标记任务 merged
- 失败/取消任务：LightRAG 毫发无伤，media.db 任务状态可重试或清理
- **工具门控**（借鉴 Yuxi Skill 门控，见 2.7.2⑤）：刮削/知识工具默认对模型不可见，按任务阶段动态激活——防 prompt 注入

### 5.6 图 + 向量存储设计（CodeGraph 源码调研结论）

> **修订标注（2026-08-11）**：本节结论已被 3.1 决策取代——图谱 + 向量 + 全文 + 检索全部由 LightRAG 承载。保留本节作为背景调研：确定性内容不上向量、向量索引为百万级设计、ORM 决策（用 Tortoise，见 5.6.6）等原则仍有效；numpy 暴力召回/sqlite-vec/自研 FTS 索引/递归 CTE 图遍历均不再需要（media.db 精确匹配用普通 B-tree 索引）。
> **修订标注（2026-08-12）**：ORM 决策已翻转——用 Tortoise（5.6.6）；"不再需要"清单随检索后端选型（决策表 #22）可能再次翻转。

**调研对象**：colbymchenry/codegraph（用户本地安装的 CLI 工具）——通过 GitHub 源码确认其存储与检索实现。

**核心结论：CodeGraph 没用向量数据库，它的"图 + AI 检索"= SQLite 图邻接表 + FTS5 + 图上概率传播（RWR），一个 embedding 都没有。**

#### 5.6.1 CodeGraph 实际存储方案（源码验证）

- **图**：`nodes` + `edges` 两张普通表（邻接表），复合索引 `(source, kind)` / `(target, kind)`，边身份 UNIQUE(source, target, kind, line, col)
- **全文**：`nodes_fts` FTS5 虚拟表，`content='nodes'` 外置表 + 3 个触发器同步（INSERT/DELETE/UPDATE）
- **拆词**：`name_segment_vocab`（camelCase 标识符拆词表，让自然语言词命中符号名）——FTS 默认分词器把 `GraphTraverser` 当单个 token，拆词表解决
- **排序**：FTS 召回种子后，在图上做随机游走重启（RWR）传播概率排序
- 源码注释原话：`deterministic, no embeddings`；`nodes_fts` 检索策略同样声明"deterministic, no embeddings"
- 全部 9 个 migration 无任何向量/embedding 列

**启示**：对符号化、确定性内容（代码符号、媒体标题/imdb_id/路径/类型标签），**确定性检索（FTS5 + 精确匹配 + 图上传播）比向量召回更精准**——不丢大小写、不改名、零模型依赖。向量只留给"模糊语义"（简介相似、自然语言查询）。

#### 5.6.2 概念辨析：图数据库 vs 向量数据库 vs SQLite 邻接表

| | 图数据库（Neo4j/Memgraph） | 向量数据库（Milvus/Qdrant/pgvector） | CodeGraph 式（SQLite 邻接表） |
|---|---|---|---|
| 数据模型 | 节点/边，Cypher 查询 | 高维向量 + ANN 索引 | 图结构 + 全文/向量索引同库 |
| 擅长 | 大型动态图遍历 | 千万级模糊语义召回 | 单机零运维，图 + 检索同事务 |
| 部署 | 独立服务要运维 | 独立服务要运维 | 一个文件 |
| ACID | 强 | 弱 | SQLite 强 |

不存在"图向量数据库"这个品类。个人单机规模 + 零部署需求 → SQLite 邻接表 + FTS5 + 可选向量索引是正确组合，为 10 万符号上 Neo4j 是杀鸡用牛刀。

#### 5.6.3 向量存储方案决策（已废弃，背景参考）

> 本节为原自研方案，已被 3.1 LightRAG 决策取代——所有"MVP 首选/后置/验证"表述均为历史语境，不作为实施依据。

| 方案 | 说明 | 结论 |
|---|---|---|
| **Python 暴力 top-k（原 MVP 首选）** | embedding 存 BLOB（512 维 float32），查询时 numpy 内积排序 | 已废弃——向量由 LightRAG 承载 |
| sqlite-vec（vec0 虚拟表） | Mozilla Builders 维护，活跃；SQL 内召回 `WHERE v MATCH ? ORDER BY distance`；Windows 有预编译 | 已废弃 |
| pgvector / LanceDB | 前者要 Postgres 服务，后者多一个组件 | 排除（违背零运维） |

**关键认知**：向量索引（HNSW/vec0）为百万级设计；几千条数据暴力召回又快又简单。

#### 5.6.4 我们要向量化什么（确定性内容不上向量）（已废弃，背景参考）

> 本节为原自研方案的历史记录（注意：原 bge-small-zh 512 维与现行 bge-m3 1024 维方案无关，现行方案见 3.1）。向量化已由 LightRAG 承担。

| 内容 | 方案（历史） |
|---|---|
| 标题 / imdb_id / TMDb·Bangumi ID / 路径 / 类型标签 | FTS5 + 精确匹配（不用向量） |
| 剧情简介/概要（中文语义） | 向量（融合 title + overview 一段文本） |
| 角色/声优名 + 简介 | 向量（可选，M3 后） |
| 用户查询（"赛博朋克题材剧场版"） | 查询时现算向量，检索上面两类 |
| 用途 | 相似推荐 + 语义搜索 |

原 MVP 范围：一张 `vec_nodes` 表（media_id + embedding BLOB + model/dim 元信息），bge-small-zh 512 维本地算，Python 暴力召回。（已废弃）

#### 5.6.5 实现坑点清单（已废弃，背景参考）

1. **FTS5 中文不分词**：默认 tokenizer 对中文按整串/字符处理，需 trigram tokenizer 或预分词
2. **递归图遍历**：复杂多跳/环的递归 CTE 要小心，性能用 EXPLAIN 调
3. **SQLite 写锁**：主服务 + MCP server 双进程连 media.db，WAL + busy_timeout 必须配（CodeGraph 为此专门写了 wal-valve.ts，真实痛点）——**此条仍有效**（media.db 双进程场景保留）
4. **向量维度锁死**：embedding 模型定下维度后换模型要全量重算（LightRAG 内仍适用）
5. **无图查询语言**：关系查询全手写 SQL（递归 CTE），比 Cypher 啰嗦
6. **sqlite-vec × aiosqlite**（若后置启用）：扩展需每个连接 load_extension（→ **升级为 M0 验证项，见 5.6.6**）

#### 5.6.6 ORM 决策：Tortoise（2026-08-12 修订，原"不用 ORM"决策被用户推翻）

- **决策（2026-08-12 用户拍板）**：用 **Tortoise ORM**——原"aiosqlite 裸连接 + repository 薄层"与"备选 SQLAlchemy 2.0 async"结论作废（见决策表 #23）
- Tortoise 能管：media.db/main.db/worker.db 业务表 CRUD + **Aerich** 迁移；原生 async（aiosqlite 驱动，`await` 不阻塞事件循环）
- 帮不上的核心部分（事实不变）：FTS5/vec0 虚拟表、递归 CTE 图遍历全要原生 SQL——走 `Model.raw()` / `Tortoise.get_connection().execute_query()` 通道；**混合模式**（ORM 管常规 CRUD，原生 SQL 管特殊检索，FTS/向量/CTE 约 80% 在图谱层）
- **连接管理**：`Tortoise.init()` 进程级初始化（FastAPI lifespan 里 init / `close_connections()`），每次查询自动从连接池取连接——天然 per-request / per-task session 语义，无需手管连接生命周期
- **已知坑（列入 M0 spike 验证项）**：
  1. WAL + busy_timeout=5000 + synchronous=NORMAL（2.7.4 PRAGMA 姿势）与 sqlite-vec `load_extension` 都需在**每个惰性连接**上 hook——Tortoise 连接懒创建，init 后统一执行 PRAGMA + 扩展加载的姿势待 spike 验证
  2. FTS 触发器走迁移脚本 DDL，与 ORM 无关（不变）

### 5.7 主服务数据库（main.db）表设计（2026-08-11，Yuxi 2.9.6 平移）

主 agent 服务持有 main.db：会话 + run 状态机 + 消息 + 轻任务（Tasker）。表结构按 Yuxi 2.9.6 翻译为 SQLite DDL，关键模式：幂等键、部分唯一索引、CAS 状态更新、终态幂等。

```sql
-- 会话/线程（2.9.6 conversations 平移）
sessions(id INTEGER PK, thread_id TEXT UNIQUE, title TEXT, status DEFAULT 'active',  -- active/deleted 软删
         is_pinned BOOLEAN DEFAULT 0, extra_metadata JSON, created_at, updated_at)

-- 运行（2.9.6 agent_runs 平移）
runs(id TEXT PK, thread_id TEXT, status TEXT DEFAULT 'pending',
     request_id TEXT UNIQUE,             -- 幂等键：主服务→子服务调用重试不重复执行
     created_by_run_id TEXT,             -- 父 run（resume/子任务归属）
     run_type TEXT DEFAULT 'chat',       -- chat/resume/subagent
     input_payload JSON, error_type TEXT, error_message TEXT,
     last_seq INTEGER,                   -- 事件游标（WS 续传）
     started_at, finished_at, created_at, updated_at)
CREATE UNIQUE INDEX uq_runs_one_active_per_thread
    ON runs(thread_id) WHERE status NOT IN ('completed','failed','cancelled','interrupted');

-- 消息（2.9.6 messages 平移：完整存储不截断）
messages(id INTEGER PK, thread_id TEXT, role TEXT, content TEXT,
         message_type TEXT DEFAULT 'text',  -- text/tool_call/tool_result
         run_id TEXT, request_id TEXT, token_count INTEGER, extra_metadata JSON)

-- 工具调用独立表（2.9.6 tool_calls 平移）
tool_calls(id INTEGER PK, message_id INTEGER, langgraph_tool_call_id TEXT UNIQUE,
           tool_name TEXT, tool_input JSON, tool_output TEXT, status TEXT DEFAULT 'pending')

-- 轻任务（2.9.5 Tasker 平移：进度 + 取消标志 + payload/result 分离）
tasks(id TEXT PK, name TEXT, type TEXT, status TEXT DEFAULT 'pending',
      progress REAL DEFAULT 0, message TEXT, payload JSON, result JSON,
      cancel_requested INTEGER DEFAULT 0, started_at, completed_at, created_at, updated_at)

-- 排队请求（2.9.6 agent_run_requests 平移：FIFO 自增 id + 策略）
run_requests(id INTEGER PK AUTOINCREMENT, request_id TEXT UNIQUE, thread_id TEXT,
             queue_policy TEXT DEFAULT 'enqueue',  -- enqueue/reject/steer
             status TEXT DEFAULT 'queued', input_payload JSON, created_at)

-- 全局配置（2.9.6 config_options 平移：params schema + value 分离）
config_options(key TEXT UNIQUE, name TEXT, description TEXT, params JSON, value JSON)

-- 并发模式（SQLite 对应物，见 2.9.6）：WAL + busy_timeout=5000 + synchronous=NORMAL（2.7.4）；
-- 行锁 = 事务内先读后写；终态幂等检查（返回是否真正变更）；CAS UPDATE...WHERE status IN (...)
```

**与媒体库的分工**：main.db = 编排域（会话/run/消息/任务/配置）；worker.db = 子线程 checkpoint；media.db = 刮削工作流 + 精确索引；LightRAG = 知识域。

#### 5.7.1 任务队列管理（决策 #24，2026-08-12）

**结论：不用 Redis/ARQ**（Yuxi 用 Redis + ARQ worker 做跨进程分发，桌面单机场景砍掉）。队列 = **SQLite 持久队列 + anyio 调度**双层模型，连接统一经 Tortoise（5.6.6），零中间件。

**第一层：持久队列 = SQLite 表（队列真相，跨进程可靠）**
- 主服务 `run_requests` 表（5.7 上表）：用户可见排队层，FIFO 排序 `(queue_policy != 'steer') asc, created_at asc, id asc`，队列位置 O(1) COUNT（2.9.5 平移）
- 子服务 `runs` 表（`run_type='subagent'`）：子任务落库即排队（status=queued），`request_id` UNIQUE 幂等（主服务→子服务重试不重复执行）、intake-then-dispatch（先 DB 事实再投递）、崩溃恢复 `recover_pending_dispatches`（重启后 queued 任务重派，Yuxi 先例 2.9.5/line 523）
- 策略：enqueue（默认）/ reject / steer（2.9.5 平移）——**reject 仅用于无效请求（如线程已有活跃 run 且请求非法），不再用于并发限流**；并发满 3 时新任务排队（queued）等水位，不 reject

**第二层：执行调度 = anyio（纯调度，不做持久化）**
- `TaskGroup` + `MemoryObjectStream` 替代 Tasker 的 `asyncio.Queue`/`create_task`（2.9.5 四件套语义平移：进度节流 / 去重入队 / 协作取消 / 超时，取消与超时用 `cancel_scope` + `move_on_after` 实现）
- **`CapacityLimiter(3)`：子服务 run 并发上限 3**，同一时刻最多 3 个 run 并行执行
- 阻塞调用（若有同步 sqlite3 场景）走 `to_thread.run_sync`；连接统一经 Tortoise `init()` 管理，不手管裸连接
- **底层事件循环仍是 asyncio**（anyio 默认后端，兼容 langgraph `AsyncSqliteSaver`/FastAPI）；项目自身代码统一写 anyio API，不直接碰 asyncio 原语

**投递/事件/取消**（不变）：主服务 → HTTP `POST /runs {request_id, input_payload}` → 子服务落库 → 调度执行；事件经 WS 协议 v2 回传；取消两阶段级联（先子后父）。

**M0 spike 验证项**：Tortoise 惰性连接上 PRAGMA（WAL + busy_timeout=5000 + synchronous=NORMAL）+ sqlite-vec `load_extension` 的 hook 姿势（5.6.6 第 1 条）；CapacityLimiter(3) 满水位排队行为。

## 6. 刮削链路（完整流程）

```
① 媒体目录扫描 ── 用户选根目录 → 规则解析出条目列表
② 文件解析 ── 确定性规则：文件名 → {标题, 季, 集数范围, 年份?, 压制组/分辨率剥离}
③ LLM 推断 ── 世界知识：清理后标题 → 作品身份（中文名/日文名/类型/年份）
④ 双源检索 ── TMDb + Bangumi 搜索 → 候选列表（置信度、年份、类型、海报）
⑤ 融合 ── imdb_id/标题匹配，双源合并去重
⑥ 用户确认 ── HITL interrupt：候选推给用户 → 选中/纠错/拒绝（低置信度必问）
⑦ 详情拉取 ── 确认后拉详情+角色图谱 → 暂存任务数据（不入 LightRAG）
⑧ 合并 ── 任务完成 → 用户确认 → insert_custom_kg 写入 LightRAG
```

- 用户交互批量模式：多个条目攒着一次确认；低置信度标红
- 任务做一半暂停（checkpoint 等待），醒来接着确认——天然支持

## 7. HITL 人类交互（官方验证）

DeepAgents 原生完整支持（docs.langchain.com/oss/python/deepagents/human-in-the-loop）：

### 7.1 interrupt_on 参数

```python
agent = create_deep_agent(
    tools=[media_upsert, ...],
    interrupt_on={
        "media_upsert": {
            "allowed_decisions": ["approve", "reject", "edit"],
            "when": only_low_confidence,   # 条件中断：高置信度自动放行
        },
    },
    checkpointer=checkpointer,  # 必须
)
```

### 7.2 4 种人类决策

| 决策 | 含义 |
|---|---|
| approve | 批准执行 |
| edit | 修改参数后执行 |
| reject | 拒绝 + 反馈给 agent |
| respond | 人类直接答话（ask_user 型工具） |

### 7.3 两个刮削场景用法

- **候选确认**：子代理工具内调 `interrupt({候选列表})` → 前端展示 → `Command(resume=...)` 恢复 → 工具拿用户选择
- **低置信度门控**：`interrupt_on` + `when` 谓词，高置信度自动入库

### 7.4 注意点

- 恢复：`result.interrupts` → `Command(resume={"decisions":[...]})` 同 config
- 多工具调用合并成一个 interrupt batch
- 每个子 agent 可独立配置 interrupt_on（适配三层架构）
- **跨进程 interrupt 需要 spike 验证**：子 agent 服务暂停 → 主服务经 SDK 获取/resume

### 7.5 落地参照：Yuxi 审批实现（见 2.7.2①，已源码验证）

- `HumanInTheLoopMiddleware(interrupt_on={...})` 26 行即可（tool_approval.py）
- **子 agent 隐藏敏感工具**（subagent/graph.py:31-48），防绕过主线程审批
- resume 不走普通消息队列，直接建 run：`Command(resume=resume_input)` → `stream_resume_with_state`
- 恢复后扫 checkpoint 兜底提取 interrupt 载荷（`state.tasks[].interrupts` / `__interrupt__`）
- 前端交互载荷形态：`human_approval_required`{action_requests, review_configs} 与 `ask_user_question_required`{questions}——与我们的候选确认/低置信度门控一一对应

### 7.6 Yuxi 用户交互端到端链路（2026-08-11 前端源码确认，HITL 完整闭环）

**闭环链路**（后端 → SSE → 前端 → 恢复）：

```
① agent 调用敏感工具/ask_user_question → LangGraph interrupt（run → interrupted）
② 事件流推 status=human_approval_required / ask_user_question_required
   + 载荷 action_requests[] + review_configs[]（一一对应）或 questions[]
③ useAgentStreamHandler.js:181-190 检测 status → flushThread（排空流式缓冲）→
   停 loading → processApprovalInStream
④ useApproval.js:87-106 解析载荷 → 构造 pendingInterrupt {kind, actionRequests/
   questions, status, threadId, interruptedRunId} → 写入 threadState.pendingInterrupt
⑤ HumanApprovalModal 弹出（slide-up 内嵌输入区上方，非居中弹窗，chat 不阻断）
   ├─ tool_approval 型：多工具逐步圆点审批（tool-progress-step）+ 参数摘要折叠
   │   （getToolApprovalSummary：execute 提取 command、write_file 提取 file_path，
   │   其余压空格）+ 拒绝带 message"用户拒绝执行该操作"
   └─ question 型：多题 Tab 逐个作答（单选/多选/other 自定义/推荐标记）
⑥ 提交 → handleApprovalWithStream（AgentChatComponent.vue:2944-2997）
   = 复用 create run 接口：POST /agent/runs {resume: answer, created_by_run_id: 旧run}
   → 后端 Command(resume=...) → stream_resume_with_state 直接续跑
   → 拿新 run_id → startRunStream 继续流式
```

**五个关键设计（DeepView 直接采纳）**：

1. **resume 复用 create run 接口**：同一端点，多带 `resume: answer` + `created_by_run_id`（父 run 引用）；**取消 = `resume('reject')`**（vue:3003-3005）——审批与取消共用一条通道，断线恢复路径统一
2. **pendingInterrupt 前端持久化**（useApproval.js:108-118）：挂 threadState，刷新后 `restoreInterruptFromThreadState` 重新弹出；提交失败回滚恢复（vue:2989-2992）——刷新/失败三层兜底
3. **审批前先 flushThread**：流式缓冲排空再弹卡片，消息完整渲染，卡片不阻断聊天（输入区 inert）
4. **双模式审批**：工具审批（逐项 approve/reject）vs 问答审批（多题 Tab）；`extractToolApprovalPayload` 要求 action_requests 与 review_configs 长度一致否则丢弃（useApproval.js:22-29）
5. **409 run_interrupted 冲突**（toolApproval.js:87-88）：run 已恢复时旧审批提交 409，前端识别处理

**模式选择器**（ToolApprovalModeSelector + toolApproval.js）：`default / always_trust` 两档；解析优先级：线程级模式 > localStorage > agent 默认（resolveToolApprovalMode:6-9）——对应我们"低置信度才问"的门控策略

**映射表**：

| Yuxi | DeepView |
|---|---|
| ask_user_question 型审批卡片 | CandidatePicker 候选确认（照搬 UI 模式：内嵌输入区 + pendingInterrupt + 失败回滚） |
| tool_approval 型（action_requests） | 低置信度门控（interrupt_on + when） |
| `resume: answer + created_by_run_id` 复用 create run | **协议采纳**：resume 复用 `POST /api/sessions/{id}/runs` + resume 载荷 + created_by_run_id（见 4.3），跨进程形态统一 |
| 审批模式选择器 | 会话级"严格/宽松"审批模式 |
| pendingInterrupt 持久化 | 接 checkpoint 恢复，M0 spike 一起验证 |

**Yuxi 没有、DeepView 保留**：候选确认的 **edit 决策**（用户修改参数后执行）——Yuxi 只有 approve/reject，媒体刮削需要改参（如纠错标题/年份）

## 8. TMDb / Bangumi 适配层

### 8.1 双源分工

| | TMDb | Bangumi (bgm.tv) |
|---|---|---|
| 擅长 | 电影、真人剧、全球内容 | 动画、漫画、游戏；角色/声优图谱 |
| 中文 | language=zh-CN | 中文社区原生 |
| 鉴权 | API Key | Access Token + User-Agent 头 |
| 核心端点 | /search/movie、/search/tv、/movie/{id}、/tv/{id}、/configuration | /v0/search/subjects、/v0/subjects/{id}、characters、persons |

### 8.2 统一接口（backend/app/providers/）

```
search(query, media_type) → [Candidate]         # 双源各自搜索
resolve(candidate) → MediaMeta                   # 拉详情
fetch_graph_extras(media_id) → [CharacterRef...] # 仅 Bangumi（角色/声优图谱）
```

- 融合：动漫 → Bangumi 优先，TMDb 补海报/评分/imdb_id；电影/真人剧 → TMDb 优先
- 交叉匹配：imdb_id（TMDb 有，Bangumi infobox 常带）→ 标题模糊 → LLM 判断
- 所有节点带 source + source_id 溯源
- 缓存：media_cache 表 TTL 30 天；令牌桶限流器

## 9. 前端组件设计

### 9.1 组件策略（三层）

1. **借鉴 run-deck 的组织模式**（不复制文件）：run-deck 的 ui/ 全套是标准 shadcn 模式（cva + cn），我们按同模式用 shadcn CLI 装标准版即可
2. **直接引入成熟库**：assistant-ui（聊天）、React Flow（图谱）、虚拟列表、toast
3. **自研 DeepView 特有组件**：任务面板、候选选择器、媒体库卡片、图谱视图

### 9.2 从 run-deck 借鉴的（模式与基础设施）

| run-deck 模式 | DeepView 用法 |
|---|---|
| `schemas/*.ts` zod 守所有 IPC/SSE 边界 | WS 帧 + `/state` 全部过 zod 校验 |
| `store/` zustand + immer 分模块 | 拆 `chatStore` / `taskStore` / `wsStore` |
| `ts-pattern` `match(...).exhaustive()` 状态映射 | 任务状态徽章、WS 事件分发 |
| `constants/labels.ts` 文案集中 | 中文文案统一管理 |
| 自研 `TitleBar`（窗口控制 + 动态标签页 + 主题切换） | 直接复刻思路（同为 Tauri 自定义标题栏） |
| `TaskList` 行设计（状态徽章 + 进度 + 行内操作） | 任务面板视觉骨架（数据源换成 WS 流） |
| `McpServerSidebar`（服务状态徽章映射） | 任务/服务状态展示同款思路 |
| 日志面板（LogSettings + LogViewer） | 结构化日志面板 |

### 9.3 成熟组件库（关键选型）

**assistant-ui**（重点评估，LangChain 官方在使用，MIT + YC 背书）：

- 现成 ChatGPT 级聊天全套：Thread/Message/Composer、流式渲染、自动滚动、markdown、代码高亮（react-shiki）、键盘快捷键、无障碍
- **Generative UI**：工具调用渲染成 React 组件 + 内联人类审批——正好承接 HITL 候选确认，interrupt 在聊天流内渲染成候选卡片
- 对接：默认 runtime 是 Vercel AI SDK / LangGraph SDK，我们是自研 WS 协议，需写 **ExternalStore adapter**（文档化自定义 runtime 模式）把 WS 流 + `/state` 对齐喂进消息模型
- 风险：runtime 抽象较复杂、版本迭代快；对策：只当渲染层用（外部数据驱动）
- 备选方案：自建聊天（react-markdown + remark-gfm + react-shiki + 自研消息流，+300 行代码，零依赖风险）

其他成熟件：

| 库 | 用途 |
|---|---|
| `@tanstack/react-virtual` | 会话/消息长列表虚拟化 |
| `@xyflow/react`（React Flow） | 图谱/关系视图（LightRAG 检索结果渲染） |
| `sonner` | toast 通知 |
| `@tauri-apps/plugin-dialog` / `plugin-notification` | 选媒体目录、任务完成系统通知 |
| `react-markdown + remark-gfm + react-shiki` | 聊天 markdown/代码高亮（assistant-ui 内部已含，备选自建时用） |

### 9.4 自研组件清单（DeepView 特有）

- **TaskPanel**：任务条目列表（文件路径 + 识别状态 + 候选确认入口 + 子代理任务状态）——子代理条目照抄 Yuxi 三层感知（见 9.6）：运行中即时上屏 + 状态徽章 + 点击展开子线程
- **CandidatePicker**：候选对比卡片（海报 + 双源标记 + 置信度 + 选择/纠错/拒绝）——HITL 核心交互；UI 模式照抄 Yuxi 审批卡片：slide-up 内嵌输入区上方 + pendingInterrupt 持久化 + 提交失败回滚 + 多候选逐步审批（见 7.6）
- **MediaCard / MediaGrid / MediaDetailDrawer**：海报网格 + 详情抽屉（关系导航）
- **SessionSidebar**：会话列表 + 全局运行状态指示灯
- **布局骨架**：TitleBar + 左侧会话栏 + 主区（聊天/任务）+ 右侧图谱抽屉

### 9.5 技术栈与页面

- **技术栈**：React 19 + TS + Vite + Tailwind + shadcn（CLI 全新安装，Radix 底座）
- **通信层**（方案 C，见 4.1）：REST 统一封装 `@tauri-apps/plugin-http`（dev 模式可退化为直连）；WS 自研通信模块（原生 WebSocket + subscribe 路由 + 心跳 + seq 续传，协议 v2 见 4.4），消费端接 streamSmoother/审批/任务面板
- **状态**：Zustand（实时流）+ TanStack Query（会话/历史）
- **页面/面板**：
  - 聊天面板（流式打字机 + 工具调用/审批内联渲染）
  - 任务面板（条目列表 + 状态 + 候选确认卡片）
  - 媒体库（海报网格 + 详情抽屉 + 关系导航）
  - 图谱视图（LightRAG 检索结果，React Flow 增强）
  - Todo / 文件 / 日志面板

### 9.6 子任务可视化：三层感知 + 直接干预（2026-08-11 定稿，Yuxi 2.9.10 平移）

**设计原则**：子任务绝不黑盒——"运行中"必须即时可见、过程可审视、干预可直达。Yuxi 三层感知俱全但**无直接干预入口**（见 2.9.10）；DeepView 平移三层 + 补齐直接干预。

**三层感知**：

| 层 | DeepView 落法 | 数据源 |
|---|---|---|
| ① 聊天流内生命周期卡片 | 消息流里 `start_async_task` 工具调用渲染成卡片：子 agent 名称 + 状态徽章（等待中/运行中/已完成/失败/取消中）+ 展开显示 run_id/thread_id/progress 消息列表/最终结果 | WS message.delta + tool_call 事件 |
| ② 任务面板运行中条目 | **流识别即时上屏**：检测 `start_async_task` 工具调用 → 立即插入"运行中"条目（不等 state.snapshot）——async_tasks 扁平结构需要靠消息流推导层级（见 2.3）；后端 `state.snapshot` 全量小块推送与前端流识别条目合并（去重键 = child_thread_id） | WS 工具调用事件 + state.snapshot |
| ③ 子线程详情抽屉 | 点条目 → 拉 `GET /api/subagents/{child_thread_id}/state`（worker.db checkpoint 消息，tool 结果嵌入 AI 消息）+ WS 订阅该线程增量 → 复用主聊天渲染（消息/工具卡片/推理折叠/todo/审批）渲染完整子线程视图 | REST + WS 增量 |

**直接干预**（超越 Yuxi，桌面端"看得见就能管"）：

- 子线程抽屉内：**取消按钮**（`DELETE /api/runs/{run_id}`，cascade_children=True 级联）；运行中步骤停止
- **对子任务发消息/回复**：子线程处于 interrupt 时，抽屉内直接出审批卡片（复用 7.6 的 resume 通道，resume 载荷带 created_by_run_id）
- 主线程侧：任务面板条目提供"取消子任务"行内操作

**子线程 ID 关联**：子线程 id 由后端 async_tasks 直接给出（无需 Yuxi 式前端复刻哈希——我们 WS 全双工，tool_call_id ↔ child_thread_id 映射由主服务在 start_async_task 工具返回时下发）。

**M2 验收**：提交刮削任务 → 子任务卡片即时出现"运行中" → 点开抽屉见子线程完整过程（消息/工具/推理）实时滚动 → 可取消 → 断线重连后状态对齐。

## 10. 决策确认表（未决项待拍板；已定项已标注）

| # | 决策 | 我的建议 |
|---|---|---|
| 1 | 会话与任务的关系 | 任务独立顶层实体，会话只是触发它的对话上下文 |
| 2 | 本地文件路径是否入图 | 建议不入图（仅 media.db media_files 记录，与 5.3 schema 一致）；若坚持入图需 5.3 节点属性补 path 字段 |
| 3 | 多媒体根目录 / 扫描策略 | 手动 + 增量扫描，不监听（此"监听"指文件系统监听，非 4.5 checkpoint watcher） |
| 4 | LLM 模型选型 | DeepSeek（中文强、便宜、tool-calling 好）（已定，见 3.1 QUERY 角色） |
| 5 | embedding 方案 | 本地 bge-m3（在 LightRAG 内配置，维度锁死后勿换模型）（已定，见 3.1） |
| 6 | NFO + 海报写入 | 待定（Kodi/Jellyfin 兼容需要则做） |
| 7 | 图浏览 MVP 形态 | 海报网格 + 详情抽屉 + 关系导航（力导向图后置） |
| 8 | API keys 管理 | 主服务 .env + 前端设置页（2.9.8 建议 OS keyring/Tauri 安全存储，两案待选） |
| 9 | 打包范围 | 先只做一键启动脚本（uv），打包 M6 再议 |
| 10 | 合并时机 | HITL 用户确认入库（可回滚） |
| 11 | assistant-ui 是否引入 | 引入，只做聊天 Thread + Generative UI（HITL 渲染），ExternalStore 接自研协议；备选自建聊天 |
| 12 | 组件底座 | 标准 shadcn（Radix）——文档生态最全，与 assistant-ui 默认主题同系 |
| 13 | 布局形态 | 三栏：会话列表 | 聊天 + 任务详情 | 图谱抽屉（聊天/任务并排，适配 agent 场景） |
| 14 | LightRAG 部署形态 | 独立 lightrag-server 进程（:8300），锁版本 lightrag-hku（已定，见 3.1） |
| 15 | 图谱/检索归属 | LightRAG 承接全部知识域（存储/全文/向量/检索/RAG）；media.db 只留刮削工作流 + 精确索引（已定，见 3.1、5.2） |
| 16 | LightRAG 4 角色 LLM | QUERY=DeepSeek、KEYWORD=轻量模型、EXTRACT 占位、embedding=bge-m3 本地（rerank 可选）（已定，见 3.1） |
| 17 | 上下文压缩 | 两级压缩（L1 结构精简 + L2 摘要），参数照抄 Yuxi 默认（100K/keep 10/300/3K/ratio 0.4）；**摘要模型建议=轻量档 LLM（KEYWORD 档）**，见 2.8 |
| 18 | Yuxi 深挖补遗采纳（见 2.9.9） | ★ 立即采纳 6 项：Run 状态机+两阶段取消、Tasker 四件套、表 schema（5.7 已落实）、审批卡片 UI、prompt 工程、子任务三层感知（2.9.10/9.6）；☆ 高价值 4 项按里程碑排期；◇ 后置项打磨期再议 |
| 19 | 前后端传输层（4.1） | **方案 C（已定）**：REST 走 tauri-plugin-http v2（消 CORS 高风险项）；WS 前端原生直连（协议 v2 见 4.4）；不引入 tauri-plugin-websocket |
| 20 | 可观测性引入（2.9.11） | 现在不引入（◇ 后置）；需要时二选一：**首选 Arize Phoenix**（pip 本地零运维）/ 备选 Langfuse Cloud（功能全但 trace 出网）；最佳引入时机 M3/M5（eval 能力） |
| 21 | Wikidata 关系补全层（2.5.2） | 建议采纳（◇ 后置）：挂在 merge 流程后，用刮削 ID 实体链接 → SPARQL 拉一跳子图 → insert_custom_kg 补全声优-角色-系列关系；失败静默降级；结果缓存 media.db |
| 22 | 检索后端选型（2.5.3） | **待拍板**：A LightRAG（3.1 现状）vs C CodeGraph 式 + 向量层（SQLite 邻接表 + FTS5 + RWR + vec0，全成熟技术，工作量自研转移）；B/D 不满足"描述找片"或"关系多跳"。倾向 C（用户），拍板后 M3 重排、决策 #14-16 作废 |
| 23 | ORM 选型（5.6.6） | **Tortoise（已定，用户拍板 2026-08-12）**：业务表 CRUD + Aerich 迁移；FTS5/vec0/递归 CTE 走 `Model.raw()` 原生通道（混合模式）；连接由 `Tortoise.init()` 进程级管理（lifespan init/close）；PRAGMA/sqlite-vec 每惰性连接 hook 为 M0 spike 验证项；原"不用 ORM / 备选 SQLAlchemy"作废 |
| 24 | 任务队列管理（5.7.1） | **anyio + SQLite 双层模型（已定，用户拍板 2026-08-12）**：不用 Redis/ARQ（Yuxi 有，砍掉）；持久队列 = SQLite 表（主服务 run_requests + 子服务 runs[Yuxi 原名 agent_runs]，request_id 幂等 + intake-then-dispatch + 崩溃恢复 recover_pending_dispatches）；执行调度 = anyio（TaskGroup + MemoryObjectStream 替代 asyncio 原语，CapacityLimiter(3) 并发上限，并发满时排队不 reject）；连接统一经 Tortoise（5.6.6）；底层 asyncio 后端、项目代码统一 anyio API |

## 11. 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | Spike：三件套跨进程（langgraph-api 子服务 + FastMCP + AsyncSubAgentMiddleware）+ AsyncSqliteSaver 落库（Yuxi PRAGMA 姿势：WAL + busy_timeout=5000 + synchronous=NORMAL，见 2.7.4）+ **跨服务 HITL 中断/恢复** + lightrag-server 启动与 insert_custom_kg 通路 | 一条命令起四服务，主 agent spawn 子 agent，状态可见，interrupt/resume 跑通，结构化 KG 喂入 LightRAG 可查询 |
| M1 | 骨架：Tauri+React+shadcn；主服务 REST/WS；聊天打字机；**传输层验证**（4.1）：tauri-plugin-http 联通 + CSP connect-src 放行 ws://localhost:8000 + dev 模式 CORS 白名单 + WS 重连/seq 续传 | 聊天面板流式出字；断线重连自动续传不丢事件 |
| M2 | 状态面板：/state + watcher 全量小块推送 + 任务卡片 + todo + 通知 + **子任务三层感知**（流识别即时上屏 + 生命周期卡片 + 子线程详情抽屉，见 9.6）+ **Steer 中断** + **TokenUsage 快照** + **AgentStatePayload 序列化** + tool_calls 独立表（2.9.9） | 子任务状态实时（含运行中条目即时出现）、点开抽屉可见子线程完整过程、可取消、断线重连对齐 |
| M3 | LightRAG 集成：4 角色 LLM 配置 + media.db 工作流表/薄索引 + insert_custom_kg 适配器（custom_kg 与文档双形态）+ **上下文压缩参数验证**（2.8.4）+ **检索参数调优**（recall/final/rrf_k=60，2.9.4）+ **文档状态机 + 统计修复**（2.9.4） | 手动入库一部片，media.db 可查重，LightRAG 语义检索/问答可用；长会话压缩触发正常 |
| M4 | 刮削链路：扫描→解析→推断→双源检索→HITL→任务数据（不入 LightRAG） | 提交一批文件，候选确认后任务状态齐全 |
| M5 | 合并 + 媒体库 UI：merge（写入 LightRAG）+ 海报网格 + 详情抽屉 + 图谱视图（LightRAG 检索结果）+ **消息反馈闭环** + **对话全文搜索** + **useStreamSmoother 流式平滑** + **GraphCanvas 交互模式**（2.9.9） | 端到端：文件→LightRAG 可问答，全程可视化 |
| M6 | 打磨：批量确认、自动合并策略、打包 | — |

## 12. 风险清单

| 风险 | 等级 | 缓解 |
|---|---|---|
| langgraph-api create_app() 具体形态未验证 | 高 | M0 spike 首项验证 |
| 前端 → :8000 通道（tauri://localhost origin） | ~~高~~ → 中（已缓解） | **方案 C（4.1）**：REST 走 tauri-plugin-http（Rust 侧执行，无 CORS）；WS 前端直连仅需 CSP connect-src 放行 ws://localhost:8000，M1 验证 |
| 跨进程 HITL interrupt/resume | 高 | M0 spike 验证项 |
| lightrag-hku API 演进 | 中 | 锁版本（PyPI 固定）；REST 接口薄封装隔离 |
| insert_custom_kg 映射质量（媒体 schema → entity/relation 描述） | 中 | 描述文本用"原名+中文名+类型+简介"融合，M3 验证检索效果 |
| 默认文件存储全量内存（GraphML/JSON 全量载入） | 低 | 个人规模（几万实体）无压力；百万级再评估切 PG/Milvus |
| deepagents API 演进 | 中 | 锁定版本（>=0.6.8，filesystem permission interrupts 需要） |
| 主服务单进程崩溃丢失运行中任务 | 低 | 重跑可接受；防 event loop 阻塞（禁同步 SQLite/CPU 密集，M1 测试项） |
| merge 写 LightRAG 失败（用户确认了但知识没进库） | 低 | media.db 记 sync 状态 + 重试/标记 |
| 压缩摘要信息丢失（L2 LLM 摘要失真） | 中 | 结构化摘要 prompt（2.8.3）；L1 不动 state，checkpoint 保留全量历史可回溯 |
| Wikidata 补全层依赖网络 / SPARQL 限流超时（2.5.2） | 低 | 结果缓存 media.db（复用刮削缓存机制）+ 批量拉取 + 失败静默降级（不影响主刮削/入库） |
| 四进程编排 | 低 | justfile / uv 脚本一键启动 |

## 13. 参考资料

- https://github.com/langchain-ai/deepagents（官方仓库，libs/ARCHITECTURE.md、middleware/async_subagents.py 源码）
- https://docs.langchain.com/oss/python/deepagents/overview
- https://docs.langchain.com/oss/python/deepagents/human-in-the-loop（HITL 已验证）
- https://docs.langchain.com/oss/python/langgraph/persistence（checkpointer/store）
- https://bangumi.github.io/api/（Bangumi API）
- https://deepwiki.com/langchain-ai/deepagents-quickstarts/6.3-deepagents-ui-integration（官方 UI 事件机制参考）
- https://github.com/asg017/sqlite-vec
- https://github.com/colbymchenry/codegraph（图 + 检索参考实现：邻接表 + FTS5 + 拆词 + RWR，src/db/schema.sql、migrations.ts 源码）
- https://github.com/assistant-ui/assistant-ui（聊天组件库：Thread/Generative UI/custom runtime）
- https://react-shiki.vercel.app/（流式代码高亮）
- https://github.com/HKUDS/LightRAG（图谱 + RAG 后端，EMNLP 2025；examples/insert_custom_kg.py、lightrag/kg/networkx_impl.py、nano_vector_db_impl.py 源码）
- https://github.com/xerrors/Yuxi（架构模式参考：DeepAgents 集成 + Neo4j/Milvus 图谱 + PPR 检索，backend/package/yuxi/knowledge/graphs/milvus_graph_service.py 源码）
- https://xerrors.github.io/Yuxi/（Yuxi 项目文档站）
- D:\code\github\Yuxi-main（Yuxi 完整源码本地副本，深度探索见 2.7、补遗见 2.9；docs/ 30 个文档中 tools-system/subagents-management/skills-management/sandbox-architecture/mcp-integration/agents-config/design/configuration/api-key-integration/roadmap/model-config/agent-evaluation 未精读，按需查阅）
- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents（Anthropic 上下文工程）
- https://docs.claude.com/en/docs/agents-and-tools/compaction（Claude Code 压缩）
- https://github.com/openai/codex（Codex CLI 压缩：codex-rs/core/src/compact.rs）
- https://typegraph.ai/blog/best-open-source-graph-rag-tools（2026 图 RAG 工具对比：GraphRAG/Graphiti/LightRAG/Cognee/TypeGraph）
- https://github.com/getzep/graphiti + https://arxiv.org/abs/2501.13956（Graphiti 时序知识图谱：增量边维护 + 混合检索）
- http://kuzudb.github.io/ + https://api.github.com/repos/kuzudb/kuzu（Kuzu 弃坑证据：2025-10-10 归档、archived:true、Apple 收购；The Register 报道 https://www.theregister.com/software/2025/10/14/kuzudb-graph-database-abandoned-community-mulls-options/1142229 ）
- https://github.com/LadybugDB/ladybug + https://ladybugdb.com/（Kuzu 社区 fork 继任者：嵌入式列式图库，FTS + vector index，v0.18）
- https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/（LazyGraphRAG：懒抽取，索引成本降 99.9%）
- https://www.wikidata.org/wiki/Wikidata:WikiProject_Anime_and_Manga（Wikidata 动漫媒体项目）
- https://www.wikidata.org/wiki/Property:P345（IMDb ID）、https://www.wikidata.org/wiki/Property:P4983（TMDb TV ID）、https://www.wikidata.org/wiki/Property:P1207（MyAnimeList ID）、https://www.wikidata.org/wiki/Property:P1114（AniDB ID）
- https://github.com/yaph/wikidata-dl（SPARQL 查询下载 wikidata 子集工具）
- https://github.com/asg017/sqlite-vec（vec0 向量虚拟表，Mozilla Builders 维护；FTS5 同姿势的向量检索）
- https://sqlite.org/fts5.html（FTS5 全文检索；trigram tokenizer 中文子串匹配）
- https://docs.cline.bot/features/auto-compact（Cline 自动压缩）
- https://aider.chat/docs/repomap.html（Aider Repo Map + 历史软阈值）
- https://arxiv.org/abs/2310.08560（MemGPT 分层记忆）+ https://github.com/letta-ai/letta
- https://github.com/microsoft/LLMLingua（token 级 prompt 压缩）
- https://github.com/langfuse/langfuse（可观测性：LLM 工程平台，六大功能见 2.9.11；LangChain CallbackHandler 一行接入）
- https://github.com/Arize-ai/phoenix（可观测性备选：pip 本地零运维，OpenInference 语义，LangChain 集成）
