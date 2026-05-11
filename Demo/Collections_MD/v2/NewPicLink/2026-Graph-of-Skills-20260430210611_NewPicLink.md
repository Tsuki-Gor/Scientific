# Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills
# 技能图谱：面向海量智能体技能的依赖感知结构化检索


Dawei Liu ${}^{1 \dagger  }$ Zongxia Li ${}^{2 \dagger  }$ Hongyang Du ${}^{3}$ Xiyang Wu ${}^{2}$ Shihang Gui ${}^{3}$ Yongbei Kuang ${}^{4}$ Lichao Sun ${}^{5}$
Dawei Liu ${}^{1 \dagger  }$ Zongxia Li ${}^{2 \dagger  }$ Hongyang Du ${}^{3}$ Xiyang Wu ${}^{2}$ Shihang Gui ${}^{3}$ Yongbei Kuang ${}^{4}$ Lichao Sun ${}^{5}$


${}^{1}$ University of Pennelvenia ${}^{2}$ University of Maryland ${}^{3}$ Brown University
${}^{1}$ 宾夕法尼亚大学 ${}^{2}$ 马里兰大学 ${}^{3}$ 布朗大学


${}^{4}$ Carnegie Melon University ${}^{5}$ Lehigh University
${}^{4}$ 卡内基梅隆大学 ${}^{5}$ 利哈伊大学


liudawei@seas.upenn.edu zli12321@umd.edu lis221@lehigh.edu
liudawei@seas.upenn.edu zli12321@umd.edu lis221@lehigh.edu


O Github: https://github.com/davidliuk/graph-of-skills
O Github: https://github.com/davidliuk/graph-of-skills


## Abstract
## 摘要


Skill usage has become a core component of modern agent systems and can substantially improve agents' ability to complete complex tasks. In real-world settings, where agents must monitor and interact with numerous personal applications, web browsers, and other environment interfaces, skill libraries can scale to thousands of reusable skills. Scaling to larger skill sets introduces two key challenges. First, loading the full skill set saturates the context window, driving up token costs, hallucination, and latency. In this paper, we present Graph of Skills (GoS), an inference-time structural retrieval layer for large skill libraries. GoS constructs an executable skill graph offline from skill packages, then at inference time retrieves a bounded, dependency-aware skill bundle through hybrid semantic-lexical seeding, reverse-weighted Personalized PageRank, and context-budgeted hydration. On SkillsBench and ALFWorld, GoS improves average reward by 43.6% over the vanilla full skill-loading baseline while reducing input tokens by ${37.8}\%$ , and generalizes across three model families: Claude Sonnet, GPT-5.2 Codex, and MiniMax. Additional ablation studies across skill libraries ranging from 200 to 2,000 skills further demonstrate that GoS consistently outperforms both vanilla skills loading and simple vector retrieval in balancing reward, token efficiency, and runtime.
技能使用已成为现代智能体系统的核心组件，能显著提升智能体完成复杂任务的能力。在现实场景中，智能体需监控并交互于众多个人应用、网页浏览器及其他环境接口，技能库规模可扩展至数千种可复用技能。扩展至大规模技能集带来了两大关键挑战。首先，加载全量技能集会填满上下文窗口，导致 Token 成本增加、幻觉增多及延迟上升。本文提出了“技能图谱”（Graph of Skills, GoS），这是一种面向大规模技能库的推理时结构化检索层。GoS 通过技能包离线构建可执行的技能图，并在推理时通过混合语义-词法种子、反向加权个性化 PageRank 以及上下文预算约束下的填充，检索出有界的、具备依赖感知的技能包。在 SkillsBench 和 ALFWorld 上的测试表明，相比传统的全量技能加载基线，GoS 将平均奖励提升了 43.6%，同时减少了 ${37.8}\%$ 的输入 Token，并可在 Claude Sonnet、GPT-5.2 Codex 和 MiniMax 三个模型系列中泛化。针对 200 到 2,000 种技能规模的消融实验进一步证明，GoS 在平衡奖励、Token 效率和运行时间方面始终优于传统的技能加载和简单的向量检索。


## 1 Introduction
## 1 引言


Large Language Model (LLM) agents solve complex technical tasks by invoking external tools, APIs, and reusable skills (Schick et al., 2023; Mialon et al., 2023). As these tools and skills grow from dozens of tools to thousands or even tens of thousands of candidates (Patil et al., 2023; Li et al., 2023; Xu et al., 2023; Qin et al., 2024), the core challenge shifts from deciding whether to use a skill to retrieving the most relevant set of skills that is sufficient for a task. Shi et al. (2025) already shows that skill retrieval itself is now a major bottleneck in realistic tool ecosystems.
大语言模型（LLM）智能体通过调用外部工具、API 和可复用技能来解决复杂的技术任务（Schick 等人，2023；Mialon 等人，2023）。随着这些工具和技能从几十个增长到数千甚至数万个候选（Patil 等人，2023；Li 等人，2023；Xu 等人，2023；Qin 等人，2024），核心挑战已从“是否使用技能”转变为“检索出足以完成任务的最相关技能集”。Shi 等人（2025）的研究已表明，技能检索本身已成为现实工具生态系统中的主要瓶颈。


Two common strategies are widely used for handling large skill libraries. Vanilla Skills (Agent Skills, 2026) prepends the entire skill set to the context window. This can work for small toolsets, but it scales poorly: token cost grows linearly with library size, and critical domain constraints become easy for the model to overlook inside an overloaded context (Liu et al., 2024a). An alternative is vector-based retrieval (Lewis et al., 2020; Wang et al., 2023), which improves efficiency by retrieving semantically similar skills. However, semantic proximity does not imply executable sufficiency. In many engineering tasks, the top semantic match is a high-level solver, while the actual solution also requires a lower-level parser, converter, setup utility, or domain-specific preprocessor that is semantically weak but functionally necessary (Qin et al., 2024; Patel et al., 2025; Patil et al., 2023) (Figure 1).
目前处理大规模技能库主要有两种常用策略。传统技能加载（Agent Skills, 2026）将整个技能集预置于上下文窗口中。这对于小型工具集有效，但扩展性较差：Token 成本随库规模线性增长，且关键的领域约束容易在过载的上下文中被模型忽略（Liu 等人，2024a）。另一种方案是基于向量的检索（Lewis 等人，2020；Wang 等人，2023），通过检索语义相似的技能来提高效率。然而，语义接近并不等同于执行上的充分性。在许多工程任务中，语义匹配度最高的往往是高层求解器，而实际解决方案还需要低层的解析器、转换器、设置工具或领域特定的预处理器，这些组件虽然语义相关性弱，但在功能上却是必需的（Qin 等人，2024；Patel 等人，2025；Patil 等人，2023）（图 1）。


---



${}^{ \dagger  }$ Core Contribution.
${}^{ \dagger  }$ 核心贡献。


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_21_bd87de.jpg"/>



Figure 1: Conceptual comparison between flat skill loading, vector retrieval, and Graph of Skills (GoS). Vanilla Skills prepends the full skill library to the prompt, so relevant constraints and prerequisite skills become buried in an overloaded context. Vector Skills improves efficiency by returning semantically similar skills, but it can still miss a functionally required prerequisite outside the retrieved set, creating the prerequisite gap. Graph of Skills starts from hybrid semantic-lexical seeds and then performs structure-aware retrieval to recover prerequisite skills and assemble a compact execution bundle.
图 1：扁平化技能加载、向量检索与技能图谱（GoS）的概念对比。传统技能加载将完整技能库预置于提示词中，导致相关约束和前置技能被淹没在过载的上下文中。向量检索通过返回语义相似的技能提高了效率，但仍可能遗漏检索集之外的功能性前置条件，从而产生“前置条件缺口”。技能图谱从混合语义-词法种子出发，通过结构感知检索来恢复前置技能，并组装成紧凑的执行包。


We present Graph of Skills (GoS), an inference-time structural retrieval layer for large local skill libraries to combat limitations of the previous two approaches. GoS constructs a directed multi-relational graph over local skill packages, where nodes are executable skills and edges encode prerequisite and workflow structure. At query time, GoS uses semantic and lexical signals only to identify a small seed set, then applies reverse-weighted Personalized PageRank (PPR) (Haveliwala, 2002; Yang et al., 2024) to recover additional skills that are structurally important for execution. The result is a bounded skill bundle that is both relevant and closer to dependency-complete than isolated top- $k$ retrieval. This problem setting is complementary to repository-scale skill infrastructures such as SkillNet and AgentSkillOS (Liang et al., 2026; Li et al., 2026a).
我们提出了技能图（GoS），这是一种用于大型本地技能库的推理时结构化检索层，旨在克服前两种方法的局限性。GoS 在本地技能包之上构建了一个有向多关系图，其中节点为可执行技能，边则编码了先决条件和工作流结构。在查询时，GoS 仅利用语义和词法信号识别出一组小型种子，随后应用反向加权个性化 PageRank（PPR）（Haveliwala, 2002; Yang et al., 2024）来恢复对执行具有结构重要性的额外技能。其结果是一个有界的技能包，它不仅具有相关性，而且比孤立的 $k$ 检索更接近依赖完整性。这一问题设定是对 SkillNet 和 AgentSkillOS 等仓库级技能基础设施的补充（Liang et al., 2026; Li et al., 2026a）。


Those works focus on creating, organizing, evaluating, and orchestrating large skill ecosystems. GoS targets a critical downstream question: if a large local skill library already exists, how should an agent retrieve the smallest and most relevant executable subset that is sufficient for the current task? Rather than exposing only keyword or vector search over repository metadata, GoS parses each skill specification into executable fields such as I/O schemas, tooling, script entrypoints, and stable source paths, constructs typed dependency and workflow edges, and retrieves a bounded bundle through graph diffusion plus reranking.
这些工作侧重于创建、组织、评估和编排大型技能生态系统。GoS 针对一个关键的下游问题：如果已经存在一个大型本地技能库，智能体应如何检索出足以完成当前任务的最小且最相关的可执行子集？GoS 不仅仅提供针对仓库元数据的关键词或向量搜索，而是将每个技能规范解析为可执行字段（如 I/O 模式、工具、脚本入口点和稳定源路径），构建类型化的依赖和工作流边，并通过图扩散加重排序来检索有界技能包。


Our contributions are as follows: (1) We introduce GoS, an agentic skill usage pipeline that combines offline graph construction with inference-time structural retrieval to improve skill selection accuracy while reducing input token consumption. (2) We evaluate GoS on the 1,000-skill SkillsBench setting and find that, compared to the full skill-loading baseline, GoS improves average reward by ${43.6}\%$ and reduces input tokens by ${37.8}\%$ across two benchmarks and three model families. Additional ablation studies confirm that this pattern holds consistently across skill library sizes ranging from 200 to 2,000 skills.
我们的贡献如下：（1）我们引入了 GoS，这是一种智能体技能使用流水线，结合了离线图构建与推理时结构化检索，在提高技能选择准确性的同时减少了输入 Token 的消耗。（2）我们在 1,000 个技能的 SkillsBench 环境中评估了 GoS，发现与全量技能加载基线相比，GoS 在两个基准测试和三个模型系列中将平均奖励提高了 ${43.6}\%$，并将输入 Token 减少了 ${37.8}\%$。额外的消融研究证实，这种模式在 200 到 2,000 个技能的库规模范围内均保持一致。


## 2 Related Work
## 2 相关工作


Tool Use, Tool Discovery, and Tool Retrieval for Agents. Early research on tool-augmented language models focuses on relatively small, fixed toolsets, where the primary challenge is deciding when to invoke a tool and formatting the call correctly (Schick et al., 2023; Mialon et al., 2023). As tool sets grow from dozens of tools to thousands (Patil et al., 2023; Li et al., 2023; Xu et al., 2023; Qin et al., 2024) and context windows continue to expand (Singh et al., 2026; Li et al., 2026c; Comanici et al., 2025), the problem shifts toward tool discovery and tool retrieval. Systems and benchmarks such as Gorilla Patil et al. (2023), API-Bank Li et al. (2023), ToolBench-style evaluations Xu et al. (2023), and ToolLLM Qin et al. (2024) show that large tool universes require scalable retrieval over API descriptions and tool documentation. ToolNet (Liu et al., 2024b) introduces graph structure into large-scale tool access, with the objective to connect models to broad tool ecosystems rather than recover dependency-complete local executable bundles. However, Shi et al. (2025) shows that tool retrieval is itself a difficult modeling problem and that generic dense retrievers are often poorly aligned with real tool-use needs (Shi et al., 2025).
智能体的工具使用、工具发现与工具检索。早期关于工具增强语言模型的研究集中在相对较小且固定的工具集上，其主要挑战在于决定何时调用工具以及如何正确格式化调用（Schick et al., 2023; Mialon et al., 2023）。随着工具集从几十个增长到数千个（Patil et al., 2023; Li et al., 2023; Xu et al., 2023; Qin et al., 2024），且上下文窗口持续扩大（Singh et al., 2026; Li et al., 2026c; Comanici et al., 2025），问题转向了工具发现与工具检索。Gorilla（Patil et al., 2023）、API-Bank（Li et al., 2023）、ToolBench 式评估（Xu et al., 2023）以及 ToolLLM（Qin et al., 2024）等系统和基准测试表明，庞大的工具库需要针对 API 描述和工具文档进行可扩展的检索。ToolNet（Liu et al., 2024b）将图结构引入大规模工具访问中，旨在将模型连接到广泛的工具生态系统，而非恢复依赖完整的本地可执行包。然而，Shi 等人（2025）的研究表明，工具检索本身是一个困难的建模问题，通用的密集检索器往往与实际的工具使用需求匹配度较差（Shi et al., 2025）。


Skill Repositories, Ecosystems, and Benchmarks. Recent systems increasingly treat agent skills as reusable assets rather than ad hoc prompts (Agent Skills, 2026; Liang et al., 2026; Li et al., 2026a). SkillNet (Liang et al., 2026) is a repository-scale skill library in this space, supporting skill creation from heterogeneous sources, multi-dimensional evaluation, ontology construction, and relational analysis over large skill collections. AgentSkillOS (Li et al., 2026a) similarly advocates for an ecosystem-level approach, emphasizing that massive skill libraries must be systematically categorized for efficient retrieval and dynamically chained together to execute complex, multi-step tasks. SkillsBench (Li et al., 2026b) shows that curated external skills can improve agent performance, while also exposing that simply having many skills available does not guarantee reliable and safe use. Other systems and registries such as SkillsMP, ClawHub, and LangSkills similarly support packaging, discovery, and search over large skill collections (SkillsMP, 2026; OpenClaw, 2026; Li et al., 2025; LabRAI, 2026). These skill repository platforms lower the cost of packaging, publishing, browsing, and searching large skill collections, but their primary interface remains entry-level search or distribution over individual skills or bundles.
技能仓库、生态系统与基准测试。近期的系统越来越多地将智能体技能视为可重用资产，而非临时提示（Agent Skills, 2026; Liang et al., 2026; Li et al., 2026a）。SkillNet（Liang et al., 2026）是该领域中仓库级的技能库，支持从异构来源创建技能、多维度评估、本体构建以及对大型技能集合的关系分析。AgentSkillOS（Li et al., 2026a）同样提倡生态系统级的方法，强调必须对海量技能库进行系统化分类以实现高效检索，并动态链接以执行复杂的多步任务。SkillsBench（Li et al., 2026b）表明，精选的外部技能可以提高智能体性能，同时也揭示了仅仅拥有大量可用技能并不能保证可靠且安全的使用。SkillsMP、ClawHub 和 LangSkills 等其他系统和注册中心同样支持对大型技能集合的打包、发现和搜索（SkillsMP, 2026; OpenClaw, 2026; Li et al., 2025; LabRAI, 2026）。这些技能仓库平台降低了打包、发布、浏览和搜索大型技能集合的成本，但其主要接口仍然停留在入门级的搜索，或针对单个技能或包的分布检索。


Graph-Based Retrieval and Relational Memory. Graph-structured retrieval has recently improved knowledge access in document, memory, and tool-use settings, but its role differs substantially across these regimes. GraphRAG (Edge et al., 2024) uses graph structure to support query-focused synthesis over document collections, HippoRAG (Jiménez Gutiérrez et al., 2024) models long-term memory as an associative graph for improved retrieval, and adjacent agent systems such as ControlLLM (Liu et al., 2023) and ToolNet (Liu et al., 2024b) incorporate graph structure over tools rather than treating tools as a flat list. However, these lines of work do not directly study retrieval over large local skill repositories. GraphRAG-style systems target knowledge synthesis, memory access, or relational QA; tool-graph methods focus primarily on graph-guided tool planning and navigation during reasoning. By contrast, our setting requires an upstream retrieval layer that selects a small executable bundle before generation begins. To our knowledge, prior work has not focused on graph-based retrieval for agent skills under this objective: recovering a dependency-complete executable bundle under a tight context budget, rather than merely retrieving one relevant item.
基于图的检索与关系记忆。图结构检索近期提升了文档、记忆及工具使用场景下的知识获取能力，但在不同领域中其作用存在显著差异。GraphRAG (Edge et al., 2024) 利用图结构支持针对文档集合的查询聚焦式综合；HippoRAG (Jiménez Gutiérrez et al., 2024) 将长期记忆建模为关联图以改进检索；而 ControlLLM (Liu et al., 2023) 和 ToolNet (Liu et al., 2024b) 等相关智能体系统则在工具之上引入了图结构，而非将其视为扁平列表。然而，这些研究并未直接探讨针对大规模本地技能库的检索。GraphRAG 类系统侧重于知识综合、记忆访问或关系问答；工具图方法主要关注推理过程中的图引导工具规划与导航。相比之下，我们的场景需要一个上游检索层，在生成开始前选择一个小的可执行包。据我们所知，现有工作尚未针对此目标研究基于图的智能体技能检索：即在严格的上下文预算下恢复一个依赖完整的可执行包，而非仅仅检索单个相关项。


## 3 Methodology
## 3 方法论


GoS is an inference-time retrieval layer for large local skill libraries. It constructs a typed graph offline from local skill packages and, at query time, returns a compact execution bundle that is relevant to the task and more likely than flat retrieval to include the prerequisites required for successful execution.
GoS 是一个针对大规模本地技能库的推理时检索层。它在离线状态下从本地技能包构建一个类型化图，并在查询时返回一个与任务相关且比扁平检索更有可能包含成功执行所需先决条件的紧凑执行包。


### 3.1 Problem Setup
### 3.1 问题设置


Let $\mathcal{C} = \left\{  {{d}_{1},\ldots ,{d}_{m}}\right\}$ denote a local corpus of skill packages. Each package contains a primary specification document together with optional scripts, references, and auxiliary assets. GoS converts $\mathcal{C}$ into a typed directed graph
令 $\mathcal{C} = \left\{  {{d}_{1},\ldots ,{d}_{m}}\right\}$ 表示一个本地技能包语料库。每个包包含一份主要规范文档以及可选的脚本、引用和辅助资产。GoS 将 $\mathcal{C}$ 转换为一个类型化有向图


$$
G = \left( {V,E,w,\phi }\right) ,
$$



where each node $v \in  V$ is a normalized executable skill record,each edge $e \in  E$ connects two skills, $w\left( e\right)  > 0$ is an edge weight,and $\phi \left( e\right)  \in  \mathcal{R}$ assigns an edge type from the relation set
其中每个节点 $v \in  V$ 是一个归一化的可执行技能记录，每条边 $e \in  E$ 连接两个技能，$w\left( e\right)  > 0$ 是边权重，$\phi \left( e\right)  \in  \mathcal{R}$ 从关系集中指定一个边类型


$$
\mathcal{R} = \{ \mathrm{{dep}},\mathrm{{wf}},\mathrm{{sem}},\mathrm{{alt}}\} .
$$



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_21_fdf83d.jpg"/>



Figure 2: Overview of Graph of Skills (GoS). Left: offline indexing converts local skill packages into normalized skill records and typed edges. Dependency edges are induced from I/O compatibility, while workflow, semantic, and alternative relations are added through sparse validation. Center: the typed directed skill graph is the retrieval substrate; edge labels denote dependency, workflow, semantic, and alternative relations. Right: online retrieval maps a task query to a compact query schema, forms merged seeds from semantic and lexical retrieval, applies reverse-aware Personalized PageRank, and returns a budgeted execution bundle after reranking and hydration.
图 2：技能图 (GoS) 概览。左侧：离线索引将本地技能包转换为归一化的技能记录和类型化边。依赖边由输入/输出兼容性诱导得出，而工作流、语义和替代关系则通过稀疏验证添加。中间：类型化有向技能图是检索基底；边标签表示依赖、工作流、语义和替代关系。右侧：在线检索将任务查询映射为紧凑的查询模式，通过语义和词法检索形成合并种子，应用反向感知个性化 PageRank，并在重排序和填充后返回一个受预算限制的执行包。


Given a task query $q$ and a context budget $\tau$ ,the retrieval problem is to return a bundle $B\left( q\right)  \subseteq  V$ that is simultaneously relevant,execution-complete when possible,and compact. We view this as a budgeted selection problem,
给定任务查询 $q$ 和上下文预算 $\tau$，检索问题在于返回一个同时具备相关性、尽可能执行完整且紧凑的包 $B\left( q\right)  \subseteq  V$。我们将此视为一个带预算的选择问题，


$$
\mathop{\max }\limits_{{B \subseteq  V}}\mathop{\sum }\limits_{{v \in  B}}\operatorname{rel}\left( {v,q}\right)  + \beta \mathop{\sum }\limits_{{\left( {u,v}\right)  \in  {E}_{\mathrm{{dep}}}}}\mathbb{I}\left\lbrack  {u \in  B \land  v \in  B}\right\rbrack  \text{ s.t. }\;\operatorname{cost}\left( B\right)  \leq  \tau , \tag{1}
$$



where the first term favors query relevance, the second rewards dependency-complete bundles, and $\operatorname{cost}\left( B\right)$ measures the prompt budget consumed by the hydrated bundle. Equation (1) is not solved exactly. Instead, GoS approximates this objective through three stages: hybrid seed retrieval, reverse-aware graph diffusion, and budgeted reranking plus hydration.
其中第一项偏向查询相关性，第二项奖励依赖完整的包，$\operatorname{cost}\left( B\right)$ 衡量填充后的包所消耗的提示词预算。公式 (1) 并非精确求解。相反，GoS 通过三个阶段近似实现该目标：混合种子检索、反向感知图扩散，以及带预算的重排序与填充。


### 3.2 Offline Graph Construction
### 3.2 离线图构建


Skill Normalization. Each skill package is parsed into a normalized skill record containing a canonical name, capability summary, I/O fields, domain tags, tooling, entrypoints, compatibility notes, and a stable local source path. This normalization step is primarily deterministic: the system extracts executable fields whenever possible and uses a lightweight LLM pass only to recover retrieval-critical semantic fields when package documentation is incomplete. The goal is to convert each local skill into a retrieval unit that an agent can directly consume at inference time.
技能归一化。每个技能包被解析为归一化的技能记录，包含规范名称、能力摘要、输入/输出字段、领域标签、工具、入口点、兼容性说明以及稳定的本地源路径。此归一化步骤主要是确定性的：系统尽可能提取可执行字段，仅在包文档不完整时使用轻量级 LLM 过程来恢复检索关键的语义字段。其目标是将每个本地技能转换为智能体在推理时可直接使用的检索单元。


Typed Relation Induction. GoS uses four edge types. Dependency edges represent executable prerequisites and form the primary structural relation in the graph. A dependency edge ${v}_{i} \rightarrow  {v}_{j}$ is added when the outputs produced by ${v}_{i}$ are compatible with inputs required by ${v}_{j}$ ,so that ${v}_{i}$ can plausibly provide an artifact consumed by ${v}_{j}$ . Workflow edges capture common multi-step pipelines, semantic edges connect near-duplicate or topically adjacent skills, and alternative edges link interchangeable strategies for the same subproblem.
类型化关系诱导。GoS 使用四种边类型。依赖边代表可执行的先决条件，是图中主要的结构关系。当 ${v}_{i}$ 产生的输出与 ${v}_{j}$ 所需的输入兼容时，添加一条依赖边 ${v}_{i} \rightarrow  {v}_{j}$，使得 ${v}_{i}$ 可以合理地提供 ${v}_{j}$ 所消耗的工件。工作流边捕获常见的多步流水线，语义边连接近义或主题相邻的技能，替代边则链接针对同一子问题的可互换策略。


Rather than performing unconstrained all-pairs relation inference, GoS constructs nondependency edges through sparse validation. For each node, it first forms a small candidate pool using lexical similarity, semantic neighbors, and I/O-based expansion. Relation validation is then applied only inside this pool. This keeps graph construction tractable while ensuring that the resulting graph remains anchored in executable structure rather than metadata proximity alone.
GoS 并非执行无约束的全对关系推断，而是通过稀疏验证构建非依赖边。对于每个节点，它首先利用词法相似度、语义邻居和基于输入/输出的扩展形成一个小规模候选池，随后仅在该池内进行关系验证。这既保证了图构建的可行性，又确保了所得图结构锚定于可执行结构，而非仅仅依赖元数据邻近性。


### 3.3 Online Structural Retrieval
### 3.3 在线结构化检索


Query Representation and Hybrid Seeding. Dense retrieval (Karpukhin et al., 2020) is often effective at finding the visible top-level skill but weak at recovering semantically subtle prerequisites. Lexical retrieval in the probabilistic ranking tradition (Robertson et al., 2009) is robust for concrete artifacts and filenames, but brittle under paraphrase. GoS therefore combines the two signals at the seeding stage.
查询表示与混合种子生成。稠密检索（Karpukhin et al., 2020）在查找显式顶层技能时通常很有效，但在恢复语义微妙的先决条件时表现较弱。基于概率排序传统的词法检索（Robertson et al., 2009）对于具体工件和文件名非常稳健，但在处理释义时较为脆弱。因此，GoS 在种子生成阶段结合了这两种信号。


At retrieval time, the raw query is mapped to a lightweight retrieval schema containing the task goal, salient operations, referenced artifacts, and normalized keywords. This schema can be produced by an optional LLM rewrite, following the general intuition of rewrite-then-retrieve pipelines (Ma et al., 2023); when rewriting is unavailable or disabled, GoS falls back to deterministic lexical normalization. GoS then computes a semantic seed score ${s}_{i}^{\text{ sem }}\left( q\right)$ and a lexical seed score ${s}_{i}^{\text{ lex }}\left( q\right)$ for each candidate skill ${v}_{i}$ ,and merges them as
在检索时，原始查询被映射到一个轻量级检索模式，其中包含任务目标、显著操作、引用的工件以及归一化关键词。该模式可由可选的 LLM 重写生成，遵循“先重写后检索”流水线的一般直觉（Ma et al., 2023）；当重写不可用或被禁用时，GoS 会回退到确定性的词法归一化。随后，GoS 为每个候选技能 ${v}_{i}$ 计算语义种子分数 ${s}_{i}^{\text{ sem }}\left( q\right)$ 和词法种子分数 ${s}_{i}^{\text{ lex }}\left( q\right)$，并将它们合并为


$$
{z}_{i}\left( q\right)  = \eta {s}_{i}^{\text{ sem }}\left( q\right)  + \left( {1 - \eta }\right) {s}_{i}^{\text{ lex }}\left( q\right) , \tag{2}
$$



where $\eta  \in  \left\lbrack  {0,1}\right\rbrack$ controls the semantic-lexical tradeoff. The initial seed distribution is obtained by normalizing the merged scores over the candidate pool,
其中 $\eta  \in  \left\lbrack  {0,1}\right\rbrack$ 控制语义与词法之间的权衡。初始种子分布通过对候选池中的合并分数进行归一化获得，


$$
{\mathbf{p}}_{i} = \frac{{z}_{i}\left( q\right) }{\mathop{\sum }\limits_{j}{z}_{j}\left( q\right) }.
$$



Reverse-Aware Typed Diffusion. Let ${A}_{r}$ denote the weighted adjacency matrix for relation type $r \in  \mathcal{R}$ . To let retrieval move from a matched high-level skill toward likely prerequisites, GoS uses both forward and reverse transitions. For each relation type, we define a row-normalized forward operator ${T}_{r}^{ \rightarrow  }$ and a row-normalized reverse operator ${T}_{r}^{ \leftarrow  }$ obtained from ${A}_{r}$ and ${A}_{r}^{\top }$ ,respectively. GoS then forms the unified transition operator
反向感知类型扩散。设 ${A}_{r}$ 表示关系类型 $r \in  \mathcal{R}$ 的加权邻接矩阵。为了使检索能够从匹配的高层技能向可能的先决条件移动，GoS 同时使用了前向和反向转换。对于每种关系类型，我们定义了分别由 ${A}_{r}$ 和 ${A}_{r}^{\top }$ 导出的行归一化前向算子 ${T}_{r}^{ \rightarrow  }$ 和行归一化反向算子 ${T}_{r}^{ \leftarrow  }$。随后，GoS 形成统一的转换算子


$$
T = \operatorname{RowNorm}\left( {\mathop{\sum }\limits_{{r \in  \mathcal{R}}}{\lambda }_{r}\left( {{T}_{r}^{ \rightarrow  } + {\gamma }_{r}{T}_{r}^{ \leftarrow  }}\right) }\right) , \tag{3}
$$



where ${\lambda }_{r} \geq  0$ and $\mathop{\sum }\limits_{r}{\lambda }_{r} = 1$ weight relation types,and ${\gamma }_{r} \geq  0$ controls how strongly reverse traversal is allowed for each type.
其中 ${\lambda }_{r} \geq  0$ 和 $\mathop{\sum }\limits_{r}{\lambda }_{r} = 1$ 对关系类型进行加权，${\gamma }_{r} \geq  0$ 控制每种类型允许反向遍历的强度。


The core retrieval step is a reverse-aware Personalized PageRank-style diffusion over this operator (Page et al., 1999; Jeh & Widom, 2003; Yang et al., 2024):
核心检索步骤是基于该算子的反向感知个性化 PageRank 式扩散（Page et al., 1999; Jeh & Widom, 2003; Yang et al., 2024）：


$$
{\mathbf{s}}^{\left( \ell  + 1\right) } = \alpha \mathbf{p} + \left( {1 - \alpha }\right) {T}^{\top }{\mathbf{s}}^{\left( \ell \right) }, \tag{4}
$$



where $\alpha  \in  \left( {0,1}\right)$ is the restart parameter. Relative to flat top- $k$ retrieval,relevance is not assigned only to individually matched skills; it is propagated across a local executable neighborhood. In particular, once a high-level solver is retrieved as a seed, upstream parser, setup, or preprocessing skills can still accumulate score through reverse dependency paths even when they are not themselves strong semantic matches to the original query.
其中 $\alpha  \in  \left( {0,1}\right)$ 是重启参数。相对于扁平化的 Top-$k$ 检索，相关性不仅被分配给单独匹配的技能，还会传播到局部可执行邻域中。特别是，一旦某个高层求解器被检索为种子，上游的解析器、设置或预处理技能即使自身与原始查询的语义匹配度不高，也能通过反向依赖路径积累分数。


Budgeted Reranking and Hydration. The diffusion score alone is insufficient, because the final output must be compact and directly usable by an agent. GoS therefore reranks candidate skills by combining graph score with field-level query evidence:
预算重排序与填充。仅靠扩散分数是不够的，因为最终输出必须紧凑且可直接供智能体使用。因此，GoS 通过结合图分数与字段级查询证据对候选技能进行重排序：


$$
{\rho }_{i}\left( q\right)  = {\mathbf{s}}_{i}^{ \star  } + \mu {m}_{i}\left( q\right) , \tag{5}
$$



where ${\mathbf{s}}_{i}^{ \star  }$ is the converged diffusion score, ${m}_{i}\left( q\right)$ aggregates direct matches between the query and skill fields such as name,capability summary,artifacts,and entrypoints,and $\mu$ controls how much local grounding is preserved after graph expansion.
其中 ${\mathbf{s}}_{i}^{ \star  }$ 是收敛后的扩散分数，${m}_{i}\left( q\right)$ 聚合了查询与技能字段（如名称、能力摘要、工件和入口点）之间的直接匹配项，$\mu$ 控制图扩展后保留局部基础信息的程度。


Candidates are then hydrated in descending order of ${\rho }_{i}\left( q\right)$ under both per-skill and global context budgets. Here, hydration denotes materializing a selected skill into an agent-consumable payload that includes a stable source path together with concise capability text and the most relevant execution notes. The final output is therefore a bounded execution bundle designed to maximize executable coverage within the prompt budget.
随后，候选技能按照 ${\rho }_{i}\left( q\right)$ 的降序在单技能和全局上下文预算下进行填充。此处，“填充”指将选定的技能具体化为智能体可消费的有效载荷，其中包含稳定的源路径、简洁的能力文本以及最相关的执行说明。因此，最终输出是一个有界的执行包，旨在最大化提示词预算内的可执行覆盖率。


## 4 Experiments
## 4 实验


We evaluate whether graph-structured retrieval improves agent performance and efficiency relative to flat full-library access and non-graph semantic retrieval.
我们评估了图结构化检索相对于扁平化全库访问和非图语义检索在提升智能体性能与效率方面的表现。


### 4.1 Experimental Setup
### 4.1 实验设置


We evaluate GoS on two benchmarks using the full released task sets. SkillsBench (Li et al., 2026b) contains a diverse set of real-world technical tasks across 11 domains, paired with curated Skills: structured packages of procedural knowledge (instructions, code templates, resources) that augment LLM agents at inference time. The task domains span complex technical work such as macroeconomic detrending, power-grid feasibility analysis, 3D scan analysis, financial modeling, and seismic phase picking. ALFWorld (Shridhar et al., 2020b) is an interactive simulator that aligns text descriptions and commands with a physically embodied robotic environment, built by combining TextWorld (Côté et al., 2018), an engine for interactive text-based games and the ALFRED dataset (Shridhar et al., 2020a). Its tasks involve multi-step household activities such as navigating rooms, finding objects, and manipulating them. In the LLM agent literature, ALFWorld is widely used in its text-only mode as a benchmark for sequential decision making, where an agent receives textual room descriptions and must issue a chain of commands to accomplish a goal (Yao et al., 2023; Shinn et al., 2023). We evaluate on the full 140-episode split.
我们在两个基准测试上使用完整发布的任务集对 GoS 进行了评估。SkillsBench (Li et al., 2026b) 包含 11 个领域中多样化的真实技术任务，并配有精选的“技能”：即在推理时增强 LLM 代理的程序化知识结构化包（指令、代码模板、资源）。任务领域涵盖了宏观经济去趋势分析、电网可行性分析、3D 扫描分析、金融建模和地震震相拾取等复杂技术工作。ALFWorld (Shridhar et al., 2020b) 是一个交互式模拟器，它将文本描述和指令与物理具身机器人环境对齐，由交互式文本游戏引擎 TextWorld (Côté et al., 2018) 和 ALFRED 数据集 (Shridhar et al., 2020a) 结合而成。其任务涉及多步骤的家庭活动，如房间导航、寻找物体及操作物体。在 LLM 代理文献中，ALFWorld 的纯文本模式被广泛用作序列决策的基准，代理接收文本房间描述并必须发布一系列指令来达成目标 (Yao et al., 2023; Shinn et al., 2023)。我们对完整的 140 个片段进行了评估。


Baselines. We compare GoS against two baselines. Vanilla Skills exposes the entire skill library directly to the agent, maximizing recall but providing no retrieval-time compression. On ALFWorld, this follows the official Agent Skills format and reference repository (Agent Skills, 2026). Vector Skills retrieves a bounded set of skills using semantic similarity over the same embedding model used by GoS, namely openai/text-embedding-3-large (OpenAI, 2024) (3072 dimensions). It isolates the effect of graph structure from the general benefit of retrieval-time compression. GoS uses the same base embedding model as Vector Skills but replaces flat nearest-neighbor retrieval with structure-aware retrieval over the skill graph. In the experiments reported here, we disable the optional query-rewrite module and use the raw task instruction as the retrieval query. The critical comparison is therefore between flat semantic retrieval and dependency-aware structural retrieval under the same backbone and embedding setup.
基线。我们将 GoS 与两个基线进行了比较。Vanilla Skills 将整个技能库直接暴露给代理，虽然最大化了召回率，但未提供任何检索时的压缩。在 ALFWorld 上，这遵循官方的 Agent Skills 格式和参考库 (Agent Skills, 2026)。Vector Skills 使用与 GoS 相同的嵌入模型，即 openai/text-embedding-3-large (OpenAI, 2024)（3072 维），通过语义相似度检索一组有限的技能。它将图结构的影响与检索时压缩带来的普遍收益分离开来。GoS 使用与 Vector Skills 相同的基准嵌入模型，但将扁平的最近邻检索替换为基于技能图的结构感知检索。在本文报告的实验中，我们禁用了可选的查询重写模块，并使用原始任务指令作为检索查询。因此，关键的比较在于相同骨干网络和嵌入设置下，扁平语义检索与依赖感知结构检索之间的差异。


Models and Evaluation. All experiments are conducted with Claude Sonnet 4.5 (Anthropic, 2025), MiniMax M2.7 (MiniMax, 2026), and GPT-5.2 Codex (OpenAI, 2025). Each model-method setting is run twice, and we report the mean across runs. We report average reward across tasks as the primary evaluation metric. For ALFWorld, rewards are binary, so average reward is equivalent to success rate. We additionally report average total token usage and agent-only runtime; runtime is measured from agent start to agent finish and excludes environment setup.
模型与评估。所有实验均使用 Claude Sonnet 4.5 (Anthropic, 2025)、MiniMax M2.7 (MiniMax, 2026) 和 GPT-5.2 Codex (OpenAI, 2025) 进行。每种模型-方法设置均运行两次，我们报告多次运行的平均值。我们将任务的平均奖励作为主要评估指标。对于 ALFWorld，奖励是二元的，因此平均奖励等同于成功率。此外，我们还报告了平均总 Token 使用量和仅代理运行时间；运行时间从代理开始到结束进行测量，不包括环境设置时间。


Evaluation Protocol. We use the full benchmark task sets and apply the same retry policy across the main and sensitivity experiments. If environment construction fails, we rebuild and rerun the task up to two additional times; tasks that still fail after these retries are excluded as unresolved infrastructure failures rather than counted as model failures. For agent timeouts, we distinguish between substantive execution failures and startup failures: if the agent has already been executing for a long time and then times out, we record reward 0 and keep the run in the aggregate; if the timeout occurs before a meaningful run is established, we rerun the trial. This protocol is applied to the full SkillsBench evaluation, the full 140-episode ALFWorld evaluation, and the library-size sensitivity study.
评估协议。我们使用完整的基准任务集，并在主要实验和敏感性实验中应用相同的重试策略。如果环境构建失败，我们会重建并重新运行任务最多两次；在这些重试后仍然失败的任务被排除为无法解决的基础设施故障，而不计为模型故障。对于代理超时，我们区分了实质性执行失败和启动失败：如果代理已经执行了很长时间后超时，我们记录奖励为 0 并将该运行保留在汇总中；如果超时发生在建立有意义的运行之前，我们则重新运行该试验。此协议适用于完整的 SkillsBench 评估、完整的 140 个片段的 ALFWorld 评估以及库大小敏感性研究。


Table 1: $\mathbf{R}$ denotes average reward (%), $\mathbf{T}$ denotes tokens,and $\mathbf{S}$ denotes runtime (s) (↑ indicates larger values are better, and $\downarrow$ denotes smaller values are better). Results are means over two runs per setting. For ALFWorld, average reward equals success rate. The top-performing results are highlighted in bold, and the second-best are underlined.
表 1：$\mathbf{R}$ 表示平均奖励 (%)，$\mathbf{T}$ 表示 Token 数，$\mathbf{S}$ 表示运行时间 (s)（↑ 表示数值越大越好，$\downarrow$ 表示数值越小越好）。结果为每种设置下两次运行的平均值。对于 ALFWorld，平均奖励等于成功率。表现最好的结果以粗体突出显示，次优结果加下划线。


<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">SkillsBench</td><td colspan="3">ALFWorld</td></tr><tr><td>R↑</td><td>T ↓</td><td>S↓</td><td>R↑</td><td>T $\downarrow$</td><td>S↓</td></tr><tr><td rowspan="3">Claude Sonnet 4.5</td><td>Vanilla Skills</td><td>25.0</td><td>967,791</td><td>465.8</td><td>89.3</td><td>1,524,401</td><td>53.2</td></tr><tr><td>Vector Skills</td><td>19.3</td><td>894,640</td><td>357.3</td><td>93.6</td><td>28,407</td><td>37.8</td></tr><tr><td>+ GoS</td><td>31.0</td><td>860,315</td><td>364.9</td><td>97.9</td><td>27,215</td><td>49.2</td></tr><tr><td rowspan="3">MiniMax M2.7</td><td>Vanilla Skills</td><td>17.2</td><td>942,113</td><td>580.7</td><td>47.1</td><td>2,184,823</td><td>88.6</td></tr><tr><td>Vector Skills</td><td>10.4</td><td>852,881</td><td>552.9</td><td>50.7</td><td>66,109</td><td>73.4</td></tr><tr><td>+ GoS</td><td>18.7</td><td>867,452</td><td>502.5</td><td>54.3</td><td>65,227</td><td>68.8</td></tr><tr><td rowspan="3">GPT-5.2 Codex</td><td>Vanilla Skills</td><td>27.4</td><td>3,187,749</td><td>686.8</td><td>89.3</td><td>1,435,614</td><td>83.3</td></tr><tr><td>Vector Skills</td><td>21.5</td><td>1,243,648</td><td>773.0</td><td>92.9</td><td>34,436</td><td>57.0</td></tr><tr><td>+ GoS</td><td>34.4</td><td>1,379,773</td><td>715.6</td><td>93.6</td><td>46,462</td><td>64.7</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">方法</td><td colspan="3">SkillsBench</td><td colspan="3">ALFWorld</td></tr><tr><td>R↑</td><td>T ↓</td><td>S↓</td><td>R↑</td><td>T $\downarrow$</td><td>S↓</td></tr><tr><td rowspan="3">Claude Sonnet 4.5</td><td>原生技能</td><td>25.0</td><td>967,791</td><td>465.8</td><td>89.3</td><td>1,524,401</td><td>53.2</td></tr><tr><td>向量技能</td><td>19.3</td><td>894,640</td><td>357.3</td><td>93.6</td><td>28,407</td><td>37.8</td></tr><tr><td>+ GoS</td><td>31.0</td><td>860,315</td><td>364.9</td><td>97.9</td><td>27,215</td><td>49.2</td></tr><tr><td rowspan="3">MiniMax M2.7</td><td>原生技能</td><td>17.2</td><td>942,113</td><td>580.7</td><td>47.1</td><td>2,184,823</td><td>88.6</td></tr><tr><td>向量技能</td><td>10.4</td><td>852,881</td><td>552.9</td><td>50.7</td><td>66,109</td><td>73.4</td></tr><tr><td>+ GoS</td><td>18.7</td><td>867,452</td><td>502.5</td><td>54.3</td><td>65,227</td><td>68.8</td></tr><tr><td rowspan="3">GPT-5.2 Codex</td><td>原生技能</td><td>27.4</td><td>3,187,749</td><td>686.8</td><td>89.3</td><td>1,435,614</td><td>83.3</td></tr><tr><td>向量技能</td><td>21.5</td><td>1,243,648</td><td>773.0</td><td>92.9</td><td>34,436</td><td>57.0</td></tr><tr><td>+ GoS</td><td>34.4</td><td>1,379,773</td><td>715.6</td><td>93.6</td><td>46,462</td><td>64.7</td></tr></tbody></table>


### 4.2 Main Results
### 4.2 主要结果


We present the main results in Table 1. Across all six model-benchmark blocks, GoS attains the highest average reward. Relative to Vanilla Skills, it reduces average token usage in all six blocks and reduces agent runtime in five of the six. Relative to Vector Skills, it improves reward in every block while keeping the token budget in the same compressed regime.
我们在表1中展示了主要结果。在全部六个模型-基准测试模块中，GoS均获得了最高的平均奖励。与Vanilla Skills相比，它在所有六个模块中均降低了平均Token使用量，并在其中五个模块中缩短了智能体运行时间。与Vector Skills相比，它在每个模块中都提升了奖励，同时将Token预算保持在同样的压缩水平。


Naive semantic retrieval struggles on long-horizon tasks. Many tasks in SkillsBench are long-horizon and require combining relevant skills with prerequisite utilities, such as environment setup, data preprocessing, or output formatting. These skills may not be lexically salient in the task description. Vector Skills, which retrieves based solely on embedding similarity to the query, often misses these indirect but essential dependencies, leading to incomplete skill sets and lower task completion rates. This pattern is most visible on SkillsBench. Under Claude Sonnet 4.5, Vector Skills drops from 25.0 to 19.3 average reward relative to Vanilla Skills; under MiniMax M2.7, it drops from 17.2 to 10.4; and under GPT-5.2 Codex, it drops from 27.4 to 21.5. In contrast, GoS improves over both baselines in all three SkillsBench blocks while still using substantially fewer tokens than flat prompting. These results are consistent with the hypothesis that long-horizon tasks are sensitive not only to topical relevance, but also to whether the retrieved bundle contains the prerequisite helpers needed to complete the full execution path.
朴素的语义检索在长程任务中表现不佳。SkillsBench中的许多任务属于长程任务，需要将相关技能与环境设置、数据预处理或输出格式化等前置工具相结合。这些技能在任务描述中可能并不具备词汇显著性。仅基于查询嵌入相似度进行检索的Vector Skills，往往会遗漏这些间接但必要的依赖关系，导致技能集不完整，进而降低任务完成率。这种模式在SkillsBench上最为明显。在Claude Sonnet 4.5下，Vector Skills的平均奖励相对于Vanilla Skills从25.0下降至19.3；在MiniMax M2.7下，从17.2下降至10.4；在GPT-5.2 Codex下，从27.4下降至21.5。相比之下，GoS在所有三个SkillsBench模块中均优于这两个基线，且使用的Token远少于扁平化提示（flat prompting）。这些结果印证了以下假设：长程任务不仅对主题相关性敏感，还取决于检索到的集合是否包含完成完整执行路径所需的前置辅助工具。


GoS achieves the strongest overall tradeoff on ALFWorld. The ALFWorld results show that the same advantage transfers to a sequential embodied environment. Under Claude Sonnet 4.5, GoS reaches 97.9% average success, compared with 93.6% for Vector Skills and 89.3% for Vanilla Skills, while reducing average total tokens from 1,524,401 to 27,215 relative to flat prompting. Under MiniMax M2.7, GoS again gives the strongest overall tradeoff, improving reward from 47.1% under Vanilla Skills and 50.7% under Vector Skills to 54.3%, while also achieving the lowest token usage and runtime in that block. Under GPT-5.2 Codex, GoS and Vector Skills are close on reward (93.6% vs. 92.9%), but GoS still remains clearly more efficient than Vanilla Skills. Taken together, these results suggest that the benefit of structure-aware retrieval is not limited to technical code-execution tasks.
GoS在ALFWorld上实现了最优的整体权衡。ALFWorld的结果表明，同样的优势可迁移至序列化具身环境。在Claude Sonnet 4.5下，GoS达到了97.9%的平均成功率，而Vector Skills为93.6%，Vanilla Skills为89.3%，同时将平均总Token数从1,524,401大幅降低至27,215（相对于扁平化提示）。在MiniMax M2.7下，GoS再次提供了最优的整体权衡，将奖励从Vanilla Skills的47.1%和Vector Skills的50.7%提升至54.3%，同时在该模块中实现了最低的Token使用量和运行时间。在GPT-5.2 Codex下，GoS与Vector Skills的奖励相近（93.6%对92.9%），但GoS在效率上仍明显优于Vanilla Skills。综上所述，这些结果表明结构感知检索的优势并不局限于技术性代码执行任务。


GoS offers the best efficiency-performance tradeoff. Vanilla Skills preserves maximal recall, but its cost grows rapidly with library size and leaves the agent to search an unstructured skill set at inference time. Vector Skills reduces token cost, but its retrieved set is often incomplete, because semantically nearby skills are not always jointly sufficient. GoS improves reward over Vector Skills by 10.97 points on SkillsBench and 2.87 points on ALFWorld while remaining far more efficient than Vanilla Skills (Table 1). The results are averaged over two runs per setting. We interpret them as a consistent empirical pattern rather than a formal significance claim.
GoS提供了最佳的效率-性能权衡。Vanilla Skills保留了最大的召回率，但其成本随库规模的增加而迅速增长，且要求智能体在推理时搜索非结构化的技能集。Vector Skills降低了Token成本，但其检索到的集合往往不完整，因为语义相近的技能并不总是联合充分的。GoS在SkillsBench上比Vector Skills提升了10.97个奖励点，在ALFWorld上提升了2.87个奖励点，同时保持了远高于Vanilla Skills的效率（表1）。上述结果为每种设置下两次运行的平均值。我们将这些结果视为一种一致的经验模式，而非正式的显著性声明。


Figure 3: Sensitivity to library size on full SkillsBench under GPT-5.2 Codex. Left: compact summary table for Vanilla Skills, Vector Skills, and GoS. Right: reward and input-token trends as the skill repository grows from 200 to 2,000 skills. GoS preserves the strongest reward once the library becomes moderately large, while both retrieval-based methods substantially weaken the growth of prompt cost relative to flat exposure.
图3：在GPT-5.2 Codex下对完整SkillsBench库规模的敏感性分析。左图：Vanilla Skills、Vector Skills和GoS的紧凑汇总表。右图：随着技能库从200个增加到2,000个，奖励和输入Token的趋势。一旦库规模达到中等水平，GoS便能保持最强的奖励，而两种基于检索的方法在抑制提示成本增长方面均显著优于扁平化展示。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_09_21_0a7028.jpg"/>



<table><tr><td>$\mathrm{N}$</td><td>Method</td><td>R↑</td><td>T $\downarrow$</td><td>S↓</td></tr><tr><td rowspan="3">200</td><td>Vanilla Skills</td><td>32.5</td><td>1.85</td><td>701.6</td></tr><tr><td>Vector Skills</td><td>21.2</td><td>1.06</td><td>833.8</td></tr><tr><td>+ GoS</td><td>32.1</td><td>1.36</td><td>731.2</td></tr><tr><td rowspan="3">500</td><td>Vanilla Skills</td><td>26.0</td><td>1.93</td><td>756.8</td></tr><tr><td>Vector Skills</td><td>20.7</td><td>1.10</td><td>849.5</td></tr><tr><td>+ GoS</td><td>31.4</td><td>1.16</td><td>890.3</td></tr><tr><td rowspan="3">1000</td><td>Vanilla Skills</td><td>27.4</td><td>3.19</td><td>686.8</td></tr><tr><td>Vector Skills</td><td>21.5</td><td>1.24</td><td>773.0</td></tr><tr><td>+ GoS</td><td>34.4</td><td>1.38</td><td>715.6</td></tr><tr><td rowspan="3">2000</td><td>Vanilla Skills</td><td>26.7</td><td>5.84</td><td>733.5</td></tr><tr><td>Vector Skills</td><td>23.8</td><td>1.11</td><td>799.8</td></tr><tr><td>+ GoS</td><td>31.3</td><td>1.14</td><td>788.0</td></tr></table>
<table><tbody><tr><td>$\mathrm{N}$</td><td>方法</td><td>R↑</td><td>T $\downarrow$</td><td>S↓</td></tr><tr><td rowspan="3">200</td><td>基础技能</td><td>32.5</td><td>1.85</td><td>701.6</td></tr><tr><td>向量技能</td><td>21.2</td><td>1.06</td><td>833.8</td></tr><tr><td>+ GoS</td><td>32.1</td><td>1.36</td><td>731.2</td></tr><tr><td rowspan="3">500</td><td>基础技能</td><td>26.0</td><td>1.93</td><td>756.8</td></tr><tr><td>向量技能</td><td>20.7</td><td>1.10</td><td>849.5</td></tr><tr><td>+ GoS</td><td>31.4</td><td>1.16</td><td>890.3</td></tr><tr><td rowspan="3">1000</td><td>基础技能</td><td>27.4</td><td>3.19</td><td>686.8</td></tr><tr><td>向量技能</td><td>21.5</td><td>1.24</td><td>773.0</td></tr><tr><td>+ GoS</td><td>34.4</td><td>1.38</td><td>715.6</td></tr><tr><td rowspan="3">2000</td><td>基础技能</td><td>26.7</td><td>5.84</td><td>733.5</td></tr><tr><td>向量技能</td><td>23.8</td><td>1.11</td><td>799.8</td></tr><tr><td>+ GoS</td><td>31.3</td><td>1.14</td><td>788.0</td></tr></tbody></table>


N: library size, R: reward, T: input tokens (M), S: runtime (s).
N：库规模，R：奖励，T：输入 Token 数（百万），S：运行时间（秒）。


### 4.3 Qualitative Analysis
### 4.3 定性分析


We inspect trajectory-level evidence to study how retrieval quality changes the downstream execution path, not just the final score. Appendix F provides a broader case set; here we focus on one representative example that isolates the main mechanism behind GoS.
我们检查了轨迹层面的证据，以研究检索质量如何改变下游执行路径，而不仅仅是最终得分。附录 F 提供了更广泛的案例集；此处我们重点分析一个具有代表性的示例，该示例隔离了 GoS 背后的主要机制。


Pedestrian Traffic Counting. pedestrian-traffic-counting requires a short but complete visual pipeline: extracting frames, counting pedestrians reliably, and formatting the output in the expected structure. In this case, GoS retrieved a compact bundle centered on gemini-count-in-video, video-frame-extraction, and openai-vision, and achieved the highest score (0.417). Vanilla Skills eventually opened related helpers, including gemini-count-in-video, video-frame-extraction, and object_counter, but reached only 0.267 , suggesting that broad library access can recover relevant tools while still leaving the agent with a noisier search problem. Vector Skills scored only 0.041: it retrieved some relevant context, but not a bundle that the agent could convert into a workable end-to-end plan. The qualitative lesson is that GoS does not help merely by retrieving topically similar skills. It helps by exposing a bundle that is already close to the executable decomposition of the task, so the agent can commit earlier to a verifier-aligned plan rather than spending budget on additional search and assembly.
行人流量统计。行人流量统计（pedestrian-traffic-counting）需要一个简短但完整的视觉流水线：提取帧、可靠地统计行人，并按预期结构格式化输出。在此案例中，GoS 检索到了一个以 gemini-count-in-video、video-frame-extraction 和 openai-vision 为核心的紧凑包，并获得了最高分（0.417）。Vanilla Skills 最终调用了相关辅助工具，包括 gemini-count-in-video、video-frame-extraction 和 object_counter，但仅达到 0.267，这表明广泛的库访问虽然能找回相关工具，却也给智能体留下了更嘈杂的搜索问题。Vector Skills 仅得 0.041：它检索到了一些相关上下文，但未能形成智能体可转化为可行端到端计划的工具包。定性结论是，GoS 的帮助不仅在于检索主题相似的技能，更在于它提供了一个已接近任务可执行分解的工具包，使智能体能更早地确定符合验证器要求的计划，而非将预算浪费在额外的搜索与组装上。


## 5 Ablation Study
## 5 消融研究


We conduct an additional ablation study to evaluate the impact of the skill library size on GoS, Vanilla Skills, and Vector Skills, alongside the effects of the lexical merge and reranker components on GoS performance.
我们进行了额外的消融研究，以评估技能库规模对 GoS、Vanilla Skills 和 Vector Skills 的影响，以及词法合并和重排序组件对 GoS 性能的作用。


### 5.1 Sensitivity to Skill Library Size
### 5.1 对技能库规模的敏感性


We run an additional full-SkillsBench study with GPT-5.2 Codex while varying the library size from 200 to 500, 1,000, and 2,000 skills. We report average reward, average input tokens, and agent-only runtime under the same retry and exclusion rules used in the main experiments in Table 2.
我们使用 GPT-5.2 Codex 运行了额外的完整 SkillsBench 研究，并将库规模分别设定为 200、500、1,000 和 2,000 个技能。我们报告了在表 2 主要实验所用的相同重试和排除规则下的平均奖励、平均输入 Token 数以及仅智能体运行时间。


Prompt cost grows rapidly for all skill exposure. The strongest trend is on input tokens. As the library grows from 500 to 2,000 skills, Vanilla Skills rises from 1.93M to
对于所有技能暴露方式，提示词成本均迅速增长。最显著的趋势体现在输入 Token 数上。随着库规模从 500 增加到 2,000 个技能，Vanilla Skills 从 1.93M 增加到


5.84M average input tokens,roughly a $3 \times$ increase. Over the same range,Vector Skills stays near 1.10M-1.24M tokens and GoS stays near 1.14M-1.38M tokens. This result shows that simple retrieval substantially weakens the coupling between repository size and prompt size, while GoS does so without giving up reward.
5.84M 平均输入 Token，大约增加了 $3 \times$。在相同范围内，Vector Skills 保持在 1.10M-1.24M Token 左右，GoS 保持在 1.14M-1.38M Token 左右。该结果表明，简单的检索显著削弱了存储库规模与提示词规模之间的耦合，而 GoS 在实现这一点的同时并未牺牲奖励。


GoS maintains a reward advantage at all tested scales. At 200 skills, Vanilla Skills is still slightly ahead of GoS (32.5 vs. 32.1). Once the library becomes moderately large, GoS outperforms both baselines at every tested scale: 31.4 vs. 26.0 / 20.7 at 500 skills, 34.4 vs. 27.4 / 21.5 at 1,000 skills, and 31.3 vs. 26.7 / 23.8 at 2,000 skills (GoS / Vanilla Skills / Vector Skills). The margin is largest at 1,000 skills and remains substantial at 2,000, indicating that increasing library size does not weaken the benefit of dependency-aware retrieval.
GoS 在所有测试规模下均保持奖励优势。在 200 个技能时，Vanilla Skills 仍略领先于 GoS（32.5 对 32.1）。一旦库规模变得适中，GoS 在每个测试规模下均优于两个基线：500 个技能时为 31.4 对 26.0 / 20.7，1,000 个技能时为 34.4 对 27.4 / 21.5，2,000 个技能时为 31.3 对 26.7 / 23.8（GoS / Vanilla Skills / Vector Skills）。差距在 1,000 个技能时最大，且在 2,000 个技能时依然显著，这表明增加库规模并不会削弱依赖感知检索的优势。


The extra retrieval step adds modest runtime for GPT-5.2 Codex, but does not change the main scaling conclusion. Both retrieval-based methods are slower than Vanilla Skills in agent-only runtime for GPT at most scales, reflecting the overhead of searching before execution. However, this reduced runtime is unique to GPT-5.2 Codex, likely due to caching mechanisms for fixed skill libraries within the black-box model, where Claude and MiniMax have longer runtime when using Vanilla Skills than GoS and Vector Skills (Table 1). In contrast, the Claude model lacks this optimization, making the Vanilla Skills approach significantly slower than retrieval methods. Furthermore, the results suggest that the primary system bottleneck is not graph traversal or vector search, but rather the overhead of exposing an increasingly large, flat library directly to the model.
额外的检索步骤为 GPT-5.2 Codex 增加了少量运行时间，但并未改变主要的扩展结论。在大多数规模下，两种基于检索的方法在 GPT 的仅智能体运行时间上均慢于 Vanilla Skills，这反映了执行前搜索的开销。然而，这种运行时间的减少仅针对 GPT-5.2 Codex，这可能是由于黑盒模型内部针对固定技能库的缓存机制所致，而 Claude 和 MiniMax 在使用 Vanilla Skills 时比使用 GoS 和 Vector Skills 运行时间更长（表 1）。相比之下，Claude 模型缺乏这种优化，使得 Vanilla Skills 方法显著慢于检索方法。此外，结果表明系统主要的瓶颈并非图遍历或向量搜索，而是将日益庞大的扁平库直接暴露给模型所带来的开销。


### 5.2 Component Analysis of the GoS Retrieval Pipeline
### 5.2 GoS 检索流水线的组件分析


We next evaluate the contribution of two key retrieval components in the GoS pipeline: graph propagation and lexical reranking. These ablations were run under the main SkillsBench configuration - GPT-5.2 Codex across skill libraries of increasing size (200, 500, 1,000, and 2,000 skills). In the first ablation, we remove graph propagation, disabling the system's ability to expand beyond seed skills to structurally related prerequisites. In the second, we remove lexical retrieval and reranking, forcing the system to rely solely on the semantic retriever before graph expansion.
接下来，我们评估 GoS 流水线中两个关键检索组件的贡献：图传播和词法重排序。这些消融实验是在主要的 SkillsBench 配置下运行的——即在规模递增（200、500、1,000 和 2,000 个技能）的技能库上使用 GPT-5.2 Codex。在第一次消融中，我们移除了图传播，禁用了系统在种子技能之外扩展到结构相关先决条件的能力。在第二次消融中，我们移除了词法检索和重排序，强制系统在图扩展前仅依赖语义检索器。


<table><tr><td>Method</td><td>R↑</td><td>T $\downarrow$</td><td>S↓</td></tr><tr><td>Full GoS</td><td>34.4</td><td>1.38</td><td>715.6</td></tr><tr><td>w/o graph propagation</td><td>29.3</td><td>0.89</td><td>766.2</td></tr><tr><td>w/o lexical + rerank</td><td>26.7</td><td>1.01</td><td>747.7</td></tr></table>
<table><tbody><tr><td>方法</td><td>R↑</td><td>T $\downarrow$</td><td>S↓</td></tr><tr><td>完整 GoS</td><td>34.4</td><td>1.38</td><td>715.6</td></tr><tr><td>无图传播</td><td>29.3</td><td>0.89</td><td>766.2</td></tr><tr><td>无词法+重排序</td><td>26.7</td><td>1.01</td><td>747.7</td></tr></tbody></table>


Table 2: Component ablation on full Skills-Bench with GPT-5.2 Codex and the 1,000-skill library. R: average reward (%), T: average total tokens (M), S: agent-only runtime (s).
表2：在GPT-5.2 Codex和1,000个技能库下，Skills-Bench全量数据的组件消融实验。R：平均奖励（%），T：平均总Token数（百万），S：仅代理运行时间（秒）。


Graph propagation and lexical reranking are important components for GoS' success. Removing graph propagation reduces average token usage from 1.38M to 0.89M, but it also lowers average reward from 34.4 to 29.3 ( $\downarrow  {5.1}$ ). Removing lexical retrieval and reranking lowers average token usage from 1.38M to 1.01M. It lowers average reward from 34.4 to 26.7 ( $\downarrow  {7.7}$ ). The larger degradation in the second ablation suggests that better seed quality is especially important on SkillsBench: if the initial retrieved skills are weak, graph expansion has less useful structure from which to recover missing prerequisites. These results show that hybrid semantic-lexical retrieval improves entry-point quality, and graph propagation then converts those stronger seeds into a more execution-complete bundle.
图传播和词法重排序是GoS成功的关键组件。移除图传播会将平均Token消耗从1.38M降至0.89M，但平均奖励也从34.4降至29.3（ $\downarrow  {5.1}$ ）。移除词法检索和重排序会将平均Token消耗从1.38M降至1.01M，并将平均奖励从34.4降至26.7（ $\downarrow  {7.7}$ ）。第二次消融实验中更大的性能下降表明，种子质量在SkillsBench上尤为重要：如果初始检索到的技能较弱，图扩展将缺乏足够的有效结构来恢复缺失的先决条件。这些结果表明，混合语义-词法检索提升了入口质量，而图传播则将这些更强的种子转化为执行更完整的集合。


## 6 Conclusion
## 6 结论


Skill retrieval is a critical bottleneck for agents operating over massive skill libraries. Unlike approaches that retrieve only semantically relevant skills, GoS recovers a small, jointly sufficient set of skills by capturing not just the target skill but also the parsers, preprocessors, and dependencies needed for successful execution. GoS is complementary to, but distinct from, broader skill management systems such as SkillNet and AgentSkillOS (Liang et al., 2026; Li et al., 2026a). Our results demonstrate that GoS consistently outperforms both vanilla skills loading and simple vector retrieval, improving execution reward while reducing token consumption. This advantage holds across two benchmarks, three model families, and skill libraries of varying sizes.
技能检索是代理在海量技能库中运行的关键瓶颈。与仅检索语义相关技能的方法不同，GoS不仅捕获目标技能，还捕获成功执行所需的解析器、预处理器和依赖项，从而恢复出一组规模较小且联合充分的技能。GoS是对SkillNet和AgentSkillOS（Liang et al., 2026; Li et al., 2026a）等更广泛技能管理系统的补充，但与之有所不同。我们的结果表明，GoS始终优于原生技能加载和简单的向量检索，在提高执行奖励的同时降低了Token消耗。这一优势在两个基准测试、三个模型系列以及不同规模的技能库中均成立。


Limitations and Future Work. GoS still depends on the quality of its offline graph. Poorly documented skills, ambiguous I/O schemas, or missing execution metadata can degrade edge quality and downstream retrieval. In addition, the current graph system is mostly static: it does not yet refine graph structure from repeated execution traces, verifier outcomes, or user feedback. Future work can focus on including online edge-weight adaptation, graph updates from successful trajectories, stronger reranking over candidate bundles, and broader evaluation on multimodal and interactive agent settings.
局限性与未来工作。GoS仍依赖于其离线图的质量。文档记录不佳的技能、模糊的I/O模式或缺失的执行元数据可能会降低边质量及后续检索效果。此外，当前的图系统大多是静态的：尚未根据重复的执行轨迹、验证器结果或用户反馈来优化图结构。未来的工作可聚焦于引入在线边权重自适应、基于成功轨迹的图更新、针对候选集合的更强重排序，以及在多模态和交互式代理设置中进行更广泛的评估。


## Reproducibility Statement
## 可复现性声明


The paper and appendix specify the core components needed to reproduce the proposed method: parser-first skill normalization, optional LLM-based semantic field completion, typed edge construction, hybrid semantic-lexical seeding, reverse-aware graph diffusion, reranking, and budgeted hydration. We also document the evaluation protocol, including benchmark settings, run structure, reward definitions, token accounting, agent-only runtime measurement, and the treatment of unresolved infrastructure failures.
本文及附录明确了复现所提方法所需的核心组件：解析器优先的技能归一化、可选的基于LLM的语义字段补全、类型化边构建、混合语义-词法种子生成、反向感知图扩散、重排序以及预算化水合。我们还记录了评估协议，包括基准测试设置、运行结构、奖励定义、Token统计、仅代理运行时间测量以及对未解决基础设施故障的处理。


Because the submission package is size-constrained, we do not include the full codebase and experiment assets with the anonymous submission. We plan to release the implementation, experiment configurations, prompt templates, agent-environment instructions, and result-processing scripts in the camera-ready release. The public release will also include the benchmark configuration files and task-generation assets needed to reproduce the SkillsBench and ALFWorld experiments reported in the paper.
由于提交包的大小限制，我们在匿名提交中未包含完整的代码库和实验资产。我们计划在最终版本中发布实现代码、实验配置、提示词模板、代理-环境指令以及结果处理脚本。公开版本还将包含复现文中报告的SkillsBench和ALFWorld实验所需的基准测试配置文件和任务生成资产。


Exact numerical replication may still vary because the experiments depend on black-box LLM APIs and external inference providers whose behavior can change over time. To reduce this variance, we use fixed prompts and configurations, repeated runs under the same protocol, and a clear separation between infrastructure failures and substantive model failures. We therefore expect the main comparative trends and qualitative conclusions to be reproducible even when exact per-run metrics vary modestly.
精确的数值复现可能仍存在差异，因为实验依赖于黑盒LLM API和外部推理提供商，其行为随时间可能发生变化。为减少这种差异，我们使用了固定的提示词和配置、相同协议下的重复运行，并明确区分了基础设施故障与实质性模型故障。因此，我们预期即使单次运行的指标有细微波动，主要的比较趋势和定性结论仍是可复现的。


## References
## 参考文献


Agent Skills. Agent skills, 2026. URL https://github.com/agentskills/agentskills.Specification and documentation repository, accessed 2026-04-01.
Agent Skills. Agent skills, 2026. URL https://github.com/agentskills/agentskills. 规范与文档仓库，访问于2026-04-01。


Anthropic. Claude sonnet 4.5, 2025. URL https://www.anthropic.com/news/ claude-sonnet-4-5. Official release page, accessed 2026-04-01.
Anthropic. Claude sonnet 4.5, 2025. URL https://www.anthropic.com/news/ claude-sonnet-4-5. 官方发布页面，访问于2026-04-01。


Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.


Marc-Alexandre Côté, Akos Kádár, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, et al. Textworld: A learning environment for text-based games. In Workshop on Computer Games, pp. 41-75. Springer, 2018.
Marc-Alexandre Côté, Akos Kádár, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, et al. Textworld: A learning environment for text-based games. In Workshop on Computer Games, pp. 41-75. Springer, 2018.


Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. 从局部到全局：一种用于查询聚焦摘要的图 RAG 方法。arXiv 预印本 arXiv:2404.16130, 2024。


Taher H. Haveliwala. Topic-sensitive pagerank. In Proceedings of the 11th International Conference on World Wide Web, WWW '02, pp. 517-526, New York, NY, USA, 2002. Association for Computing Machinery. ISBN 1581134495. doi: 10.1145/511446.511513. URL https://doi.org/10.1145/511446.511513.
Taher H. Haveliwala. 主题敏感的 PageRank。第 11 届国际万维网会议论文集 (WWW '02), 第 517-526 页, 美国纽约州纽约市, 2002。计算机协会。ISBN 1581134495。doi: 10.1145/511446.511513。URL https://doi.org/10.1145/511446.511513。


Glen Jeh and Jennifer Widom. Scaling personalized web search. In Proceedings of the 12th international conference on World Wide Web, pp. 271-279, 2003.
Glen Jeh and Jennifer Widom. 扩展个性化网络搜索。第 12 届国际万维网会议论文集, 第 271-279 页, 2003。


Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. arXiv preprint arXiv:2405.14831, 2024.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. HippoRAG：受神经生物学启发的用于大语言模型的长期记忆。arXiv 预印本 arXiv:2405.14831, 2024。


Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769-6781, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.550. URL https://aclanthology.org/2020.emnlp-main.550/.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 用于开放域问答的密集段落检索。2020 年自然语言处理经验方法会议 (EMNLP) 论文集, 第 6769-6781 页, 在线, 2020。计算语言学协会。doi: 10.18653/v1/2020.emnlp-main.550。URL https://aclanthology.org/2020.emnlp-main.550/。


LabRAI. Langskills, 2026. URL https://github.com/LabRAI/LangSkills.GitHub repository, accessed 2026-04-01.
LabRAI. LangSkills, 2026。URL https://github.com/LabRAI/LangSkills。GitHub 仓库，访问日期 2026-04-01。


Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pp. 9459-9474, 2020.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, 等。用于知识密集型 NLP 任务的检索增强生成。神经信息处理系统进展, 第 33 卷, 第 9459-9474 页, 2020。


Hao Li, Chunjiang Mu, Jianhao Chen, Siyue Ren, Zhiyao Cui, Yiqun Zhang, Lei Bai, and Shuyue Hu. Organizing, orchestrating, and benchmarking agent skills at ecosystem scale. arXiv preprint arXiv:2603.02176, 2026a.
Hao Li, Chunjiang Mu, Jianhao Chen, Siyue Ren, Zhiyao Cui, Yiqun Zhang, Lei Bai, and Shuyue Hu. 生态系统规模下智能体技能的组织、编排与基准测试。arXiv 预印本 arXiv:2603.02176, 2026a。


Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. Api-bank: A comprehensive benchmark for tool-augmented llms. arXiv preprint arXiv:2304.08244, 2023.
Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. API-Bank：一个用于工具增强型大语言模型的综合基准测试。arXiv 预印本 arXiv:2304.08244, 2023。


Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, and Han-chung Lee. Skillsbench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670, 2026b. doi: 10.48550/arXiv.2602.12670. URL https://arxiv.org/abs/2602.12670.
Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, and Han-chung Lee. SkillsBench：跨多样化任务评估智能体技能表现的基准测试。arXiv 预印本 arXiv:2602.12670, 2026b。doi: 10.48550/arXiv.2602.12670。URL https://arxiv.org/abs/2602.12670。


Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding vision-language model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025.
Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, 等。通过推理分解实现视觉语言模型的自我奖励。arXiv 预印本 arXiv:2508.19652, 2025。


Zongxia Li, Hongyang Du, Chengsong Huang, Xiyang Wu, Lantao Yu, Yicheng He, Jing Xie, Xiaomin Wu, Zhichao Liu, Jiarui Zhang, et al. Mm-zero: Self-evolving multi-model vision language models from zero data. arXiv preprint arXiv:2603.09206, 2026c.
Zongxia Li, Hongyang Du, Chengsong Huang, Xiyang Wu, Lantao Yu, Yicheng He, Jing Xie, Xiaomin Wu, Zhichao Liu, Jiarui Zhang 等。Mm-zero: 基于零数据自演化的多模态视觉语言模型。arXiv 预印本 arXiv:2603.09206, 2026c。


Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Xin Xu, Tongtong Wu, Kun Wang, Yang Liu, Zhen Bi, Jungang Lou, Yuchen Eleanor Jiang, Hangcheng Zhu, Gang Yu, Haiwen Hong, Longtao Huang, Hui Xue, Chenxi Wang, Yijun Wang, Zifei Shan, Xi Chen, Zhaopeng Tu, Feiyu Xiong, Xin Xie, Peng Zhang, Zhengke Gui, Lei Liang, Jun Zhou, Chiyu Wu, Jin Shang, Yu Gong, Junyu Lin, Changliang Xu, Hongjie Deng, Wen Zhang, Keyan Ding, Qiang Zhang, Fei Huang, Ningyu Zhang, Jeff Z. Pan, Guilin Qi, Haofen Wang, and Huajun Chen. Skillnet: Create, evaluate, and connect ai skills. arXiv preprint arXiv:2603.04448, 2026.
Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Xin Xu, Tongtong Wu, Kun Wang, Yang Liu, Zhen Bi, Jungang Lou, Yuchen Eleanor Jiang, Hangcheng Zhu, Gang Yu, Haiwen Hong, Longtao Huang, Hui Xue, Chenxi Wang, Yijun Wang, Zifei Shan, Xi Chen, Zhaopeng Tu, Feiyu Xiong, Xin Xie, Peng Zhang, Zhengke Gui, Lei Liang, Jun Zhou, Chiyu Wu, Jin Shang, Yu Gong, Junyu Lin, Changliang Xu, Hongjie Deng, Wen Zhang, Keyan Ding, Qiang Zhang, Fei Huang, Ningyu Zhang, Jeff Z. Pan, Guilin Qi, Haofen Wang, 和 Huajun Chen。Skillnet: 创建、评估与连接 AI 技能。arXiv 预印本 arXiv:2603.04448, 2026。


Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157-173, 2024a. doi: 10.1162/tacl_a_00638. URL https://aclanthology.org/2024.tacl-1.9/.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, 和 Percy Liang。迷失在中间：语言模型如何利用长上下文。计算语言学协会汇刊, 12:157-173, 2024a。doi: 10.1162/tacl_a_00638。URL https://aclanthology.org/2024.tacl-1.9/。


Xukun Liu, Zhiyuan Peng, Xiaoyuan Yi, Xing Xie, Lirong Xiang, Yuchen Liu, and Dongkuan Xu. Toolnet: Connecting large language models with massive tools via tool graph. arXiv preprint arXiv:2403.00839, 2024b.
Xukun Liu, Zhiyuan Peng, Xiaoyuan Yi, Xing Xie, Lirong Xiang, Yuchen Liu, 和 Dongkuan Xu。Toolnet: 通过工具图将大语言模型与海量工具连接。arXiv 预印本 arXiv:2403.00839, 2024b。


Zhaoyang Liu, Zeqiang Lai, Zhangwei Gao, Erfei Cui, Ziheng Li, Xizhou Zhu, Lewei Lu, Qifeng Chen, Yu Qiao, Jifeng Dai, and Wenhai Wang. Controlllm: Augment language models with tools by searching on graphs. arXiv preprint arXiv:2310.17796, 2023.
Zhaoyang Liu, Zeqiang Lai, Zhangwei Gao, Erfei Cui, Ziheng Li, Xizhou Zhu, Lewei Lu, Qifeng Chen, Yu Qiao, Jifeng Dai, 和 Wenhai Wang。Controlllm: 通过图搜索为语言模型增强工具能力。arXiv 预印本 arXiv:2310.17796, 2023。


Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting in retrieval-augmented large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 5303-5315, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.322. URL https://aclanthology.org/2023.emnlp-main.322/.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, 和 Nan Duan。检索增强大语言模型中的查询重写。收录于 2023 年经验方法自然语言处理会议论文集，第 5303-5315 页，新加坡，2023。计算语言学协会。doi: 10.18653/v1/2023.emnlp-main.322。URL https://aclanthology.org/2023.emnlp-main.322/。


Grégoire Mialon, Roberto Dessi, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, et al. Augmented language models: a survey. Transactions on Machine Learning Research, 2023.
Grégoire Mialon, Roberto Dessi, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz 等。增强型语言模型：综述。机器学习研究汇刊, 2023。


MiniMax. Minimax m2.7, 2026. URL https://www.minimax.io/news/minimax-m27-en.Official release page, accessed 2026-04-01.
MiniMax。Minimax m2.7, 2026。URL https://www.minimax.io/news/minimax-m27-en。官方发布页面，访问日期 2026-04-01。


OpenAI. text-embedding-3-large, 2024. URL https://developers.openai.com/api/docs/ models/text-embedding-3-large. Official model documentation, accessed 2026-04-01.
OpenAI。text-embedding-3-large, 2024。URL https://developers.openai.com/api/docs/ models/text-embedding-3-large。官方模型文档，访问日期 2026-04-01。


OpenAI. Gpt-5.2-codex, 2025. URL https://developers.openai.com/api/docs/models/ gpt-5.2-codex. Official model documentation, accessed 2026-04-01.
OpenAI。Gpt-5.2-codex, 2025。URL https://developers.openai.com/api/docs/models/ gpt-5.2-codex。官方模型文档，访问日期 2026-04-01。


OpenClaw. Clawhub registry, 2026. URL https://openclawdoc.com/docs/skills/ clawhub/. Documentation page, accessed 2026-04-01.
OpenClaw。Clawhub 注册中心, 2026。URL https://openclawdoc.com/docs/skills/ clawhub/。文档页面，访问日期 2026-04-01。


Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. The pagerank citation ranking: Bringing order to the web. Technical report, Stanford InfoLab, 1999.
Lawrence Page, Sergey Brin, Rajeev Motwani, 和 Terry Winograd。PageRank 引文排名：为网络带来秩序。技术报告，斯坦福 InfoLab, 1999。


Bhrij Patel, Davide Belli, Amir Jalalirad, Maximilian Arnold, Aleksandr Ermovol, and Bence Major. Dynamic tool dependency retrieval for efficient function calling. arXiv preprint arXiv:2512.17052, 2025.
Bhrij Patel, Davide Belli, Amir Jalalirad, Maximilian Arnold, Aleksandr Ermovol, 和 Bence Major。用于高效函数调用的动态工具依赖检索。arXiv 预印本 arXiv:2512.17052, 2025。


Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334, 2023.
Shishir G. Patil, Tianjun Zhang, Xin Wang, 和 Joseph E. Gonzalez。Gorilla：连接海量 API 的大语言模型。arXiv 预印本 arXiv:2305.15334, 2023。


Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. Toolllm: Facilitating large language models to master 16000+ real-world apis. In International Conference on Learning Representations, 2024.
Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, 和 Maosong Sun。Toolllm：助力大语言模型掌握 16000+ 现实世界 API。载于《国际学习表征会议》，2024。


Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333-389, 2009.
Stephen Robertson, Hugo Zaragoza 等。概率相关性框架：BM25 及后续研究。《信息检索基础与趋势®》，3(4):333-389, 2009。


Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2023.
Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, 和 Thomas Scialom。Toolformer：语言模型可以自学使用工具。《神经信息处理系统进展》，36, 2023。


Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, and Zhaochun Ren. Retrieval models aren't tool-savvy: Benchmarking tool retrieval for large language models. arXiv preprint arXiv:2503.01763, 2025.
Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, 和 Zhaochun Ren。检索模型并不精通工具：大语言模型工具检索基准测试。arXiv 预印本 arXiv:2503.01763, 2025。


Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. arXiv preprint arXiv:2303.11366, 2023.
Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, 和 Shunyu Yao。Reflexion：具备语言强化学习的语言智能体。arXiv 预印本 arXiv:2303.11366, 2023。


Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10740-10749, 2020a.
Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, 和 Dieter Fox。ALFRED：日常任务基础指令解释基准测试。载于《IEEE/CVF 计算机视觉与模式识别会议》，第 10740-10749 页，2020a。


Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020b.
Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, 和 Matthew Hausknecht。ALFWorld：对齐文本与具身环境以进行交互式学习。arXiv 预印本 arXiv:2010.03768, 2020b。


Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2026.
Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram 等。OpenAI GPT-5 系统卡。arXiv 预印本 arXiv:2601.03267, 2026。


SkillsMP. Skillsmp, 2026. URL https://skillsmp.com/.Agent Skills Marketplace, accessed 2026-04-01.
SkillsMP。Skillsmp, 2026。网址 https://skillsmp.com/。智能体技能市场，访问日期 2026-04-01。


Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.
Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, 和 Anima Anandkumar。Voyager：基于大语言模型的开放式具身智能体。arXiv 预印本 arXiv:2305.16291, 2023。


Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, and Jian Zhang. On the tool manipulation capability of open-source large language models. arXiv preprint arXiv:2305.16504, 2023.
Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, 和 Jian Zhang。论开源大语言模型的工具操作能力。arXiv 预印本 arXiv:2305.16504, 2023。


Mingji Yang, Hanzhi Wang, Zhewei Wei, Sibo Wang, and Ji-Rong Wen. Efficient algorithms for personalized pagerank computation: A survey. IEEE Transactions on Knowledge and Data Engineering, 36(9):4582-4602, 2024. doi: 10.1109/TKDE.2024.3376000.
Mingji Yang, Hanzhi Wang, Zhewei Wei, Sibo Wang, 和 Ji-Rong Wen。个性化 PageRank 计算的高效算法：综述。《IEEE 知识与数据工程汇刊》，36(9):4582-4602, 2024。doi: 10.1109/TKDE.2024.3376000。


Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, 和 Yuan Cao。ReAct：协同语言模型中的推理与行动。载于《国际学习表征会议》，2023。


## A Appendix Overview
## A 附录概览


To make the supplementary material easier to navigate, we briefly summarize the organization of the appendix before presenting the detailed evidence. The appendix is designed to complement the main paper along four axes: implementation fidelity, prompt/interface design, retrieval mechanics, and trajectory-grounded empirical analysis.
为方便查阅补充材料，我们在展示详细证据前先简要概述附录的组织结构。附录旨在从实现保真度、提示/界面设计、检索机制以及基于轨迹的实证分析四个维度对正文进行补充。


Table 3: Appendix roadmap. The table summarizes the role of each supplementary section and how it complements the main paper.
表3：附录路线图。该表总结了各补充章节的作用及其对正文的补充方式。


<table><tr><td>Section</td><td>Purpose</td></tr><tr><td>Implementation Details</td><td>Documents how GoS is instantiated in code, including parsing, graph construction, hybrid seeding, diffusion, reranking, and hydration.</td></tr><tr><td>Prompt and Interface Examples</td><td>Shows representative internal prompts and agent-facing interface rules used during graph construction and online retrieval.</td></tr><tr><td>Core Retrieval Pseudocode</td><td>Gives compact pseudocode for the offline indexing pipeline and the online graph-based retrieval pipeline.</td></tr><tr><td>Error Analysis</td><td>Separates retrieval misses, partial retrieval, execution drift, and infrastructure failures.</td></tr><tr><td>Qualitative Analysis</td><td>Provides trajectory-grounded case studies that compare the actual skill context exposed under different retrieval conditions.</td></tr></table>
<table><tbody><tr><td>章节</td><td>目的</td></tr><tr><td>实现细节</td><td>记录 GoS 如何在代码中实例化，包括解析、图构建、混合种子生成、扩散、重排序和填充。</td></tr><tr><td>提示词与界面示例</td><td>展示图构建和在线检索过程中使用的典型内部提示词及面向智能体的界面规则。</td></tr><tr><td>核心检索伪代码</td><td>提供离线索引流水线和在线图检索流水线的精简伪代码。</td></tr><tr><td>错误分析</td><td>区分检索遗漏、部分检索、执行偏差和基础设施故障。</td></tr><tr><td>定性分析</td><td>提供基于轨迹的案例研究，对比在不同检索条件下实际暴露的技能上下文。</td></tr></tbody></table>


## B Implementation Details
## B 实现细节


This appendix summarizes the implementation decisions behind GoS and clarifies how the abstract pipeline in the main text is instantiated in code. The purpose is not to enumerate every engineering detail, but to expose the concrete design choices that determine graph quality, retrieval behavior, and the final agent-facing bundle.
本附录总结了 GoS 背后的实现决策，并阐明了正文中的抽象流程如何在代码中实例化。其目的并非罗列每一项工程细节，而是揭示决定图质量、检索行为以及最终面向智能体程序包的具体设计选择。


Table 4: GoS implementation summary. The table highlights the main design choices that determine graph construction, retrieval, and agent-facing hydration.
表 4：GoS 实现摘要。该表重点列出了决定图构建、检索以及面向智能体填充的主要设计选择。


<table><tr><td>Component</td><td>Implementation choice</td></tr><tr><td>Node construction</td><td>Parser-first normalization from SKILL.md plus optional LLM completion of retrieval-critical semantic fields.</td></tr><tr><td>Dependency edges</td><td>Directed edges induced by bidirectional output-input compatibility checks.</td></tr><tr><td>Higher-order edges</td><td>Sparse LLM validation over a bounded candidate pool for workflow, semantic, and alternative relations.</td></tr><tr><td>Seed retrieval</td><td>Hybrid semantic and lexical seeding over normalized node fields.</td></tr><tr><td>Graph scoring</td><td>Reverse-aware typed Personalized PageRank with relation-specific reverse transitions.</td></tr><tr><td>Final output</td><td>Reranked, budgeted hydration into agent-usable skill payloads with stable Source: paths.</td></tr></table>
<table><tbody><tr><td>组件</td><td>实现选择</td></tr><tr><td>节点构建</td><td>基于 SKILL.md 的解析器优先归一化，辅以检索关键语义字段的 LLM 补全。</td></tr><tr><td>依赖边</td><td>由双向输出-输入兼容性检查导出的有向边。</td></tr><tr><td>高阶边</td><td>在有限候选池内进行稀疏 LLM 验证，以确定工作流、语义及替代关系。</td></tr><tr><td>种子检索</td><td>基于归一化节点字段的混合语义与词法种子检索。</td></tr><tr><td>图评分</td><td>具备反向感知能力的类型化个性化 PageRank，包含关系特定的反向转移。</td></tr><tr><td>最终输出</td><td>经重排序、预算限制填充至智能体可用技能负载，并保留稳定的 Source: 路径。</td></tr></tbody></table>


Table 5: Normalized node fields used at retrieval time. The table lists the fields retained in each skill node and their retrieval role in GoS.
表5：检索时使用的归一化节点字段。该表列出了每个技能节点中保留的字段及其在GoS中的检索作用。


<table><tr><td>Field</td><td>Primary role</td><td>Why it matters</td></tr><tr><td>name, description</td><td>canonical identity and coarse semantic match</td><td>Provide the most stable high-level skill signature during lexical and semantic seeding.</td></tr><tr><td>one_line_capability</td><td>concise capability abstraction</td><td>Helps retrieval align a task to what the skill actually does, rather than to document wording alone.</td></tr><tr><td>inputs, outputs</td><td>executable interface schema</td><td>Support deterministic dependency induction and help retrieval recover prerequisite producers and consumers.</td></tr><tr><td>domain_tags, tooling</td><td>technical context</td><td>Improve matching on domain-specific libraries, APIs, and workflows when task language is underspecified.</td></tr><tr><td>example_tasks</td><td>usage priors</td><td>Improve recall for tasks described by objective or scenario rather than by direct tool names.</td></tr><tr><td>script_entrypoints</td><td>implementation affor-dances</td><td>Help the agent discover reusable scripts instead of re-implementing logic from scratch.</td></tr><tr><td>compatibility, allowed_tools</td><td>execution constraints</td><td>Preserve operational restrictions that are important for verifier-aligned use.</td></tr><tr><td>source_path, rendered_snippet</td><td>hydration and agent consumption</td><td>Make retrieved skills directly inspectable inside the execution environment and keep the bundle compact.</td></tr></table>
<table><tbody><tr><td>领域</td><td>主要作用</td><td>重要性</td></tr><tr><td>名称、描述</td><td>规范化标识与粗粒度语义匹配</td><td>在词法和语义种子生成阶段提供最稳定的高层技能签名。</td></tr><tr><td>单行能力描述</td><td>简洁的能力抽象</td><td>帮助检索过程将任务与技能的实际功能对齐，而非仅依赖文档措辞。</td></tr><tr><td>输入、输出</td><td>可执行接口模式</td><td>支持确定性的依赖归纳，并帮助检索过程恢复前置的生产者与消费者。</td></tr><tr><td>领域标签、工具</td><td>技术上下文</td><td>在任务描述不明确时，改善对特定领域库、API及工作流的匹配效果。</td></tr><tr><td>示例任务</td><td>使用先验</td><td>针对通过目标或场景而非直接工具名称描述的任务，提高召回率。</td></tr><tr><td>脚本入口点</td><td>实现功能支持</td><td>帮助智能体发现可复用脚本，避免从零开始重新实现逻辑。</td></tr><tr><td>兼容性、允许使用的工具</td><td>执行约束</td><td>保留对验证器对齐使用至关重要的操作限制。</td></tr><tr><td>源路径、渲染代码片段</td><td>实例化与智能体调用</td><td>使检索到的技能可在执行环境中直接查看，并保持包体紧凑。</td></tr></tbody></table>


Table 6: Relation types and reverse weights in reverse-aware graph diffusion. The table shows how each edge type contributes to backward structural propagation during retrieval.
表6：逆向感知图扩散中的关系类型与反向权重。该表展示了在检索过程中，每种边类型如何对反向结构传播产生影响。


<table><tr><td>Relation type</td><td>Weight</td><td>Meaning</td><td>Retrieval consequence</td></tr><tr><td>Dependency</td><td>1.0</td><td>Skill $u$ produces an artifact consumed by skill $v$</td><td>Strongest backward propagation, since recovering prerequisites is the main purpose of GoS.</td></tr><tr><td>Workflow</td><td>0.5</td><td>Two skills are commonly chained in a concrete multi-step pipeline</td><td>Allows moderate backward expansion toward adjacent pipeline stages without dominating dependency evidence.</td></tr><tr><td>Semantic</td><td>0.2</td><td>Two skills belong to the same narrow capability cluster</td><td>Provides weak smoothing across near-neighbor skills while limiting topical drift.</td></tr><tr><td>Alternative</td><td>0.1</td><td>Two skills solve the same subproblem via different implementations</td><td>Provides minimal backward mass, mainly to keep interchangeable options reachable.</td></tr></table>
<table><tbody><tr><td>关系类型</td><td>权重</td><td>含义</td><td>检索结果</td></tr><tr><td>依赖</td><td>1.0</td><td>技能 $u$ 产生的产物被技能 $v$ 所消耗</td><td>最强的反向传播，因为恢复先决条件是 GoS 的主要目的。</td></tr><tr><td>工作流</td><td>0.5</td><td>两个技能通常在具体的多步流水线中串联</td><td>允许向相邻流水线阶段进行适度的反向扩展，且不会主导依赖证据。</td></tr><tr><td>语义</td><td>0.2</td><td>两个技能属于同一个狭义能力集群</td><td>在近邻技能间提供弱平滑，同时限制主题偏移。</td></tr><tr><td>替代</td><td>0.1</td><td>两个技能通过不同的实现方式解决同一个子问题</td><td>提供最小的反向权重，主要用于保持可互换选项的可达性。</td></tr></tbody></table>


Pipeline Summary. GoS proceeds in two phases. Offline, it parses local skill packages into normalized nodes, adds dependency edges by I/O matching, and augments the graph with sparse workflow, semantic, and alternative relations. Online, it forms a hybrid semantic-lexical seed set, applies reverse-aware graph diffusion, and returns a reranked, budgeted bundle of execution-ready skills.
流水线摘要。GoS 分为两个阶段。离线阶段，它将本地技能包解析为标准化节点，通过输入/输出匹配添加依赖边，并利用稀疏工作流、语义和替代关系增强图结构。在线阶段，它形成混合语义-词法种子集，应用反向感知图扩散，并返回一个经过重排序且符合预算的、可直接执行的技能包。


Implementation Substrate. GoS is implemented on top of a graph-backed retrieval substrate for workspace management, vector indexing, and graph storage, while replacing document-centric assumptions with skill-specific parsing, relation induction, and agent-oriented hydration. Concretely, the system maintains an HNSW vector index over skill representations together with a typed directed graph whose vertices are normalized skills and whose edges carry relation labels, directional semantics, and scalar weights. This yields a retrieval substrate in which semantic proximity and structural connectivity can be combined inside a single inference-time pipeline rather than treated as disjoint retrieval regimes.
实现基础。GoS 构建于一个支持工作空间管理、向量索引和图存储的图检索基础之上，同时摒弃了以文档为中心的假设，转而采用技能特定的解析、关系归纳和面向代理的填充。具体而言，系统在技能表示上维护一个 HNSW 向量索引，并结合一个类型化有向图，其中顶点为标准化技能，边则携带关系标签、方向语义和标量权重。这产生了一个检索基础，使得语义邻近性和结构连通性可以在单一推理流水线中结合，而非作为分离的检索机制处理。


Parser-First Skill Normalization. Each local skill package is first parsed deterministically from its primary SKILL.md file and nearby package structure. The parser extracts the canonical name and description from YAML frontmatter, collects explicit input and output fields, recovers domain tags, tooling, example tasks, compatibility notes, and allowed tools from both frontmatter and markdown sections, and resolves script entrypoints by scanning the local scripts/ directory when present. It also materializes a stable local source path and a rendered snippet used later for retrieval and hydration. This parser-first design keeps node construction anchored in executable package structure rather than relying entirely on free-form semantic extraction.
解析优先的技能标准化。每个本地技能包首先根据其主要的 SKILL.md 文件及周边包结构进行确定性解析。解析器从 YAML 前言中提取规范名称和描述，收集显式的输入和输出字段，恢复领域标签、工具、示例任务、兼容性说明及允许使用的工具，并在存在 scripts/ 目录时通过扫描该目录解析脚本入口点。它还固化了一个稳定的本地源路径以及稍后用于检索和填充的渲染片段。这种解析优先的设计使节点构建锚定在可执行的包结构上，而非完全依赖自由形式的语义提取。


LLM-Assisted Semantic Completion. When package documentation is incomplete, GoS optionally performs a lightweight LLM pass over the full markdown body to recover retrieval-critical semantic fields, including capability summaries, inputs, outputs, domain tags, tooling, and example tasks. Importantly, this stage is constrained to normalize a single skill node and is not used to emit graph relations directly. In other words, the LLM here serves as a high-precision semantic completion module for node attributes rather than as an unconstrained graph-construction oracle. The inferred fields are then merged back with the deterministic parse, with the implementation favoring completed semantic fields only when they improve the retrieval representation.
LLM 辅助的语义补全。当包文档不完整时，GoS 可选择对完整的 Markdown 正文进行轻量级 LLM 处理，以恢复检索关键的语义字段，包括能力摘要、输入、输出、领域标签、工具和示例任务。重要的是，此阶段仅限于标准化单个技能节点，不用于直接生成图关系。换言之，此处的 LLM 充当节点属性的高精度语义补全模块，而非不受约束的图构建预言机。推断出的字段随后与确定性解析结果合并，实现时仅在补全的语义字段能改善检索表示时才予以采纳。


Typed Node Representation. After normalization, each skill is serialized into a node record that stores both structured lists and compact textual views. Besides canonical descriptive fields, the node retains raw skill content, rendered snippets, script entrypoints, and a stable Source: path. This dual representation is operationally important: the graph and vector index operate over normalized fields, whereas the final agent-facing bundle requires concise but directly usable payloads that can be opened inside the execution environment without path reconstruction.
类型化节点表示。标准化后，每个技能被序列化为一个节点记录，存储结构化列表和紧凑的文本视图。除规范描述字段外，节点还保留原始技能内容、渲染片段、脚本入口点和稳定的 Source: 路径。这种双重表示在操作上至关重要：图和向量索引基于标准化字段运行，而最终面向代理的包需要简洁且可直接使用的载荷，以便在执行环境中无需路径重构即可打开。


Directed Typed Relation Induction. The GoS graph is a typed directed graph rather than an undirected similarity graph. Dependency edges are induced deterministically by matching producer outputs against consumer inputs in both directions for each candidate pair. An edge $u \rightarrow  v$ therefore has explicit executable semantics: $u$ can plausibly provide an artifact consumed by $v$ . Because I/O compatibility is asymmetric in general,this dependency structure cannot be reduced to undirected similarity without losing the notion of prerequisite direction.
有向类型化关系归纳。GoS 图是一个类型化有向图，而非无向相似度图。依赖边通过匹配每个候选对中生产者输出与消费者输入的方向性来确定性地归纳。因此，边 $u \rightarrow  v$ 具有明确的可执行语义：$u$ 可以合理地提供 $v$ 所消耗的工件。由于输入/输出兼容性在一般情况下是不对称的，如果不保留先决条件的方向性，这种依赖结构就无法简化为无向相似度。


Non-dependency relations, namely workflow, semantic, and alternative, are added through sparse LLM validation rather than dense all-pairs inference. For each node, GoS first forms a bounded candidate pool by combining lexical overlap, semantic neighbors from the vector index, and I/O-based candidate expansion. The LLM is then asked only to validate high-confidence relations inside this restricted pool. This two-stage design keeps graph construction tractable and biases the resulting graph toward precision rather than density.
非依赖关系（即工作流、语义和替代关系）通过稀疏的 LLM 验证而非密集的两两推理来添加。对于每个节点，GoS 首先通过结合词法重叠、向量索引中的语义邻居以及基于输入/输出的候选扩展来形成一个有界的候选池。随后，仅要求 LLM 验证该受限池内高置信度的关系。这种两阶段设计使图构建保持可控，并使生成的图偏向于精确性而非密度。


Hybrid Seeding at Query Time. At retrieval time, GoS does not rely on vector search alone. The system first optionally rewrites the raw task request into a compact query schema containing the goal, operations, artifacts, constraints, and high-value keywords. Semantic seeding is then obtained from nearest-neighbor search in embedding space, while lexical seeding is computed from token overlap over normalized node fields such as name, capability, I/O descriptors, tooling, example tasks, entrypoints, and snippets. These candidate pools are merged and reranked before graph diffusion, so the graph is seeded by a hybrid entry set rather than by a single retriever. In practice, this detail matters because the quality of the initial seeds strongly influences whether later graph expansion can recover the correct prerequisite chain.
查询时的混合种子生成。在检索时，GoS 不仅依赖向量搜索。系统首先可选择将原始任务请求重写为包含目标、操作、工件、约束和高价值关键词的紧凑查询模式。语义种子通过嵌入空间中的最近邻搜索获得，而词法种子则通过名称、能力、输入/输出描述符、工具、示例任务、入口点和片段等标准化节点字段上的词元重叠计算得出。这些候选池在图扩散前进行合并和重排序，因此图是由混合入口集而非单一检索器引导的。在实践中，这一细节至关重要，因为初始种子的质量强烈影响后续图扩展能否恢复正确的先决条件链。


Reverse-Aware Structural Diffusion. Retrieval over the graph uses a Personalized PageRank-style diffusion operator constructed from the directed typed edges. The implementation first inserts forward transition mass along each stored edge, and then injects type-specific reverse transitions so that relevance can flow back from a matched high-level skill toward likely prerequisites. The reverse coefficients are largest for dependency edges and smaller for workflow, semantic, and alternative links, reflecting the fact that reverse traversal is most justified when recovering executable prerequisites. Operationally, this means the graph remains directed, but retrieval is explicitly reverse-aware. GoS therefore does not collapse the graph into an undirected graph; instead, it performs controlled backward propagation during scoring.
逆向感知结构扩散。图检索使用一种基于有向类型边的个性化 PageRank 式扩散算子。实现时，首先沿每条存储的边注入前向转移权重，随后注入特定类型的逆向转移，使相关性能够从匹配的高级技能回溯至可能的先决条件。逆向系数在依赖边上最大，在工作流、语义及替代链接上较小，这反映了逆向遍历在恢复可执行先决条件时最为合理。在操作层面，这意味着图保持有向性，但检索过程显式具备逆向感知能力。因此，GoS 不会将图折叠为无向图，而是在评分过程中执行受控的反向传播。


Reranking and Budgeted Hydration. The stationary graph score is not exposed to the agent directly. After diffusion, GoS reranks candidate skills by combining graph relevance with field-level query evidence, then hydrates only the top skills into an agent-facing bundle under both per-skill and global context budgets. Each hydrated payload includes a concise skill rendering, relevant execution notes, and the original local source path. The retrieval output therefore functions as a bounded execution context rather than a generic search-result list. This final budgeted hydration step is essential for preserving the efficiency advantage over flat all-skills loading while still presenting enough structure for downstream execution.
重排序与预算化填充。平稳图分数不会直接暴露给智能体。扩散后，GoS 通过结合图相关性与字段级查询证据对候选技能进行重排序，随后在单项技能和全局上下文预算限制下，仅将顶级技能填充至面向智能体的包中。每个填充的有效载荷包含简洁的技能呈现、相关的执行说明及原始本地源路径。因此，检索输出充当的是有界执行上下文，而非通用的搜索结果列表。这一最终的预算化填充步骤对于保持优于扁平化全量加载的效率优势至关重要，同时能为下游执行提供足够的结构信息。


Section Summary. From an implementation perspective, GoS is best understood as a hybrid graph-construction and retrieval system: deterministic parsing and I/O matching provide a reliable executable backbone; optional LLM semantic completion improves node quality when documentation is incomplete; sparse LLM relation validation adds higher-order inter-skill structure; and reverse-aware graph diffusion converts a small hybrid seed set into a compact, more execution-complete bundle. These implementation choices are what instantiate the central claim of the paper that structural retrieval should recover not only relevant skills, but also the prerequisite context needed to use them effectively.
本节总结。从实现角度看，GoS 最好被理解为一个混合图构建与检索系统：确定性解析与 I/O 匹配提供了可靠的可执行骨干；可选的 LLM 语义补全在文档不完整时提升了节点质量；稀疏的 LLM 关系验证增加了高阶的技能间结构；而逆向感知图扩散将一小部分混合种子集转化为紧凑且执行完备的包。这些实现选择印证了本文的核心主张，即结构化检索不仅应恢复相关技能，还应恢复有效使用这些技能所需的先决条件上下文。


## C Prompt and Interface Examples
## C 提示词与接口示例


Layered Prompt Design. GoS uses two prompt layers with deliberately separated responsibilities. The first layer operates inside the indexing and retrieval stack, where LLMs are used only for constrained normalization, optional query rewriting, and sparse relation validation. The second layer operates at the agent interface, where the environment prompt tells the downstream agent when to call retrieval, how to interpret the returned bundle, and how strongly to prefer reuse over open-ended exploration. This separation is methodologically important. Graph-side prompts determine what semantic structure enters the retrieval substrate, whereas agent-side prompts determine whether that retrieved structure is converted into a verifier-aligned execution plan.
分层提示词设计。GoS 使用两层职责明确分离的提示词。第一层在索引与检索栈内运行，LLM 仅用于受限的归一化、可选的查询重写及稀疏关系验证。第二层在智能体接口处运行，环境提示词告知下游智能体何时调用检索、如何解读返回的包，以及在重用与开放式探索之间应如何权衡。这种分离在方法论上很重要。图侧提示词决定了何种语义结构进入检索基底，而智能体侧提示词则决定了检索到的结构是否被转化为符合验证器要求的执行计划。


Presentation Goal. This section is not intended to enumerate full prompt templates. Instead, it exposes the narrow prompt fragments and interface rules that are most important for understanding the method. From a reviewer perspective, the key point is that GoS does not rely on unconstrained prompt engineering. The internal prompts are used to normalize or validate bounded objects, and the external interface is used to constrain downstream behavior once retrieval has occurred. Together, these prompts form an interface contract between offline graph construction and online execution.
展示目标。本节并非旨在列举完整的提示词模板，而是展示对理解该方法至关重要的狭义提示词片段与接口规则。从评审角度看，关键点在于 GoS 不依赖无约束的提示词工程。内部提示词用于归一化或验证有界对象，外部接口用于在检索发生后约束下游行为。这些提示词共同构成了离线图构建与在线执行之间的接口契约。


Internal Prompt A: Skill Semantic Completion. The semantic-completion prompt is intentionally narrow. It asks the model to normalize exactly one skill document and extract only retrieval-critical fields. In the implementation, the prompt explicitly preserves the canonical name and description when present, emphasizes high precision, and requires the returned edges list to be empty. This design reflects a conservative use of LLMs: the
内部提示词 A：技能语义补全。语义补全提示词被刻意限制在狭窄范围内。它要求模型仅归一化一个技能文档，并仅提取检索关键字段。在实现中，该提示词明确保留了规范名称与描述（若存在），强调高精度，并要求返回的边列表为空。这种设计反映了对 LLM 的保守使用：模型被允许填补节点属性中的语义空白，但不得凭空创造图结构。


Table 7: Representative prompt and interface components in GoS. The table highlights the small set of prompt contracts that shape graph construction and downstream agent behavior. model is allowed to fill semantic gaps in node attributes, but not to invent graph structure. Operationally, this improves the quality of node representations used for indexing while avoiding a common failure mode in LLM-built graphs, namely relation over-generation. The prompt is therefore best understood as a constrained semantic completion module, not as a latent graph generator.
表 7：GoS 中具有代表性的提示词与接口组件。该表强调了塑造图构建与下游智能体行为的一小部分提示词契约。模型被允许填补节点属性中的语义空白，但不得凭空创造图结构。在操作层面，这提升了用于索引的节点表示质量，同时避免了 LLM 构建图时常见的失效模式，即关系过度生成。因此，该提示词最好被理解为受限的语义补全模块，而非潜在的图生成器。


<table><tr><td>Component</td><td>Role</td><td>Key constraint</td><td>Intended effect</td></tr><tr><td colspan="4">Internal Prompt A</td></tr><tr><td>Skill semantic completion</td><td>Normalize skill document into retrieval-critical fields.</td><td>Preserve canonical name/description; f fill only node-local semantic fields; return an empty edges list.</td><td>Improve node quality when SKILL.md is incomplete while preventing relation hallucination.</td></tr><tr><td>Internal Prompt B</td><td></td><td></td><td></td></tr><tr><td>Relation validation</td><td>Verify whether a bounded candidate pair should receive a typed edge.</td><td>Restrict outputs to \{dependency, workflow, semantic, alternative\}; preserve exact source/target names; emit nothing when uncertain.</td><td>Keep the graph sparse and precise instead of generating dense all-pairs links.</td></tr><tr><td colspan="4">Internal Prompt C</td></tr><tr><td>Query rewrite</td><td>Rewrite a raw request into a compact retrieval schema.</td><td>Extract goal, operations, artifacts, constraints, and keywords without redefining the task.</td><td>Improve seed-stage lexical and semantic coverage while preserving task intent.</td></tr><tr><td colspan="4">Agent Interface</td></tr><tr><td>Retrieval usage contract</td><td>Tell the downstream agent when retrieval must be called and how the returned bundle should be used.</td><td>Run graphskills-query first; read Retrieval Status; Source: paths; prefer adapting retrieved scripts; prioritize verifier-minimal behavior.</td><td>Make retrieval operational immediately, so the bundle narrows search instead of serving as optional background context.</td></tr></table>
<table><tbody><tr><td>组件</td><td>角色</td><td>关键约束</td><td>预期效果</td></tr><tr><td colspan="4">内部提示词 A</td></tr><tr><td>技能语义补全</td><td>将技能文档规范化为检索关键字段。</td><td>保留规范名称/描述；仅填充节点局部语义字段；返回空边列表。</td><td>在 SKILL.md 不完整时提升节点质量，同时防止关系幻觉。</td></tr><tr><td>内部提示词 B</td><td></td><td></td><td></td></tr><tr><td>关系验证</td><td>验证受限候选对是否应建立类型边。</td><td>输出限制为 {dependency, workflow, semantic, alternative}；保留精确的源/目标名称；不确定时不输出。</td><td>保持图的稀疏性和精确性，避免生成密集的任意两点连接。</td></tr><tr><td colspan="4">内部提示词 C</td></tr><tr><td>查询重写</td><td>将原始请求重写为紧凑的检索模式。</td><td>提取目标、操作、工件、约束和关键词，且不重新定义任务。</td><td>在保留任务意图的同时，提升种子阶段的词汇和语义覆盖率。</td></tr><tr><td colspan="4">智能体接口</td></tr><tr><td>检索使用契约</td><td>告知下游智能体何时必须调用检索，以及如何使用返回的包。</td><td>优先运行 graphskills-query；读取检索状态；来源：路径；优先适配检索到的脚本；优先采取验证器最小化行为。</td><td>使检索即刻生效，确保检索包缩小搜索范围，而非仅作为可选的背景上下文。</td></tr></tbody></table>


Internal Prompt B: Relation Validation. The relation-validation prompt is invoked only after GoS has formed a small candidate pool using lexical overlap, semantic neighbors, and I/O-based expansion. The prompt defines four edge types: dependency, workflow, semantic, and alternative. It also explicitly instructs the model to prefer sparse, high-precision edges, to emit nothing when uncertain, and to preserve exact skill names in the source and target fields. This makes the prompt function more like a relation verifier than a free-form graph generator. In practice, this design is important because it limits graph density and preserves the typed semantics later used during reverse-aware diffusion.
内部提示词 B：关系验证。关系验证提示词仅在 GoS 通过词汇重叠、语义邻居和基于 I/O 的扩展形成小型候选池后才被调用。该提示词定义了四种边类型：依赖、工作流、语义和替代。它还明确指示模型优先选择稀疏、高精度的边，在不确定时保持沉默，并保留源字段和目标字段中的精确技能名称。这使得该提示词的功能更像是一个关系验证器，而非自由形式的图生成器。在实践中，这种设计至关重要，因为它限制了图的密度，并保留了后续在反向感知扩散（reverse-aware diffusion）中使用的类型化语义。


## Prompt excerpt: skill semantic completion
## 提示词摘录：技能语义补全


---



1. Extract exactly one skill node from the document.
1. 从文档中精确提取一个技能节点。


3. Infer only retrieval-critical fields: capability, inputs,
3. 仅推断检索关键字段：能力、输入、


&nbsp;&nbsp;&nbsp;&nbsp;outputs, domain_tags, tooling, example_tasks.
&nbsp;&nbsp;&nbsp;&nbsp;输出、领域标签、工具、示例任务。


6. Use high precision. If uncertain, leave a field empty.
6. 保持高精度。如果不确定，请将字段留空。


7. Do not invent relationships. Return an empty 'edges' list.
7. 不要虚构关系。返回一个空的“edges”列表。


---



This excerpt illustrates the central design principle of the internal extraction prompt: GoS uses the LLM as a constrained semantic normalizer, not as an unconstrained graph author. For the appendix, the important point is not merely that an LLM appears in the pipeline, but that the allowable output space is sharply restricted to node-local semantic completion.
此摘录阐明了内部提取提示词的核心设计原则：GoS 将大语言模型用作受限的语义归一化器，而非不受限的图构建者。对于附录而言，重点不仅在于流水线中使用了大语言模型，而在于其允许的输出空间被严格限制在节点局部的语义补全上。


Internal Prompt C: Query Rewrite. The optional query-rewrite prompt maps a raw task request to a compact retrieval schema with fields such as goal, operations, artifacts, constraints, and keywords. The prompt explicitly instructs the model not to invent benchmark-specific labels and to leave unclear fields empty. This is consistent with the retrieval objective in GoS: rewriting is used only to expose retrieval-critical technical terms such as file formats, APIs, protocols, and concrete operations. It is not intended to change the task itself. When rewriting is disabled or unavailable, the system falls back to deterministic lexical normalization, so query rewriting is a retrieval enhancement rather than a mandatory dependency. In other words, the prompt improves lexical and semantic coverage at the seed stage, but it is not allowed to redefine the problem.
内部提示词 C：查询重写。可选的查询重写提示词将原始任务请求映射为紧凑的检索模式，包含目标、操作、工件、约束和关键词等字段。该提示词明确指示模型不要虚构特定于基准测试的标签，并将不明确的字段留空。这与 GoS 的检索目标一致：重写仅用于暴露文件格式、API、协议和具体操作等检索关键技术术语。它无意改变任务本身。当重写功能被禁用或不可用时，系统会回退到确定性的词汇归一化，因此查询重写是一种检索增强手段，而非强制性依赖。换言之，该提示词在种子阶段提高了词汇和语义覆盖率，但严禁重新定义问题。


Agent Interface Prompt. In the SkillsBench GoS environment, the agent is instructed to begin with a targeted retrieval query built from the task goal, artifact or format, operation or API, and verifier-critical constraints. The interface then forces the agent to read the retrieval status before continuing. A NO_SKILL_HIT response means the agent must explicitly acknowledge that no relevant skill was found and proceed without claiming skill use. A SKILL_HIT response means the returned bundle should be treated as a narrowing device: the agent is told to use the returned local source paths, inspect scripts before implementing from scratch, and prioritize the shortest path to verifier pass. This interface design matters because the main quality difference is often not whether some relevant skill exists somewhere in the library, but whether the agent receives a compact, execution-ready bundle early enough to affect the trajectory. In that sense, the interface prompt is part of the method rather than a presentation detail.
智能体接口提示词。在 SkillsBench GoS 环境中，智能体被指示从一个有针对性的检索查询开始，该查询由任务目标、工件或格式、操作或 API 以及验证器关键约束构建而成。随后，接口强制智能体在继续操作前读取检索状态。NO_SKILL_HIT 响应意味着智能体必须明确承认未找到相关技能，并在不声称使用技能的情况下继续执行。SKILL_HIT 响应意味着返回的包应被视为一种缩小范围的工具：智能体被告知使用返回的本地源路径，在从头实现前检查脚本，并优先选择通往验证器通过的最短路径。这种接口设计至关重要，因为主要的质量差异往往不在于库中是否存在相关技能，而在于智能体是否能足够早地收到紧凑、可执行的包以影响执行轨迹。从这个意义上讲，接口提示词是该方法的一部分，而非展示细节。


## Prompt excerpt: agent-facing retrieval interface
## 提示词摘录：面向智能体的检索接口


---



Before writing any code, run:
在编写任何代码之前，请运行：


&nbsp;&nbsp;&nbsp;&nbsp;graphskills-query "goal + artifact/format + operation/API +
&nbsp;&nbsp;&nbsp;&nbsp;graphskills-query "目标 + 工件/格式 + 操作/API +


&nbsp;&nbsp;&nbsp;&nbsp;verifier-critical constraint"
&nbsp;&nbsp;&nbsp;&nbsp;验证器关键约束"


- If Retrieval Status: NO_SKILL_HIT, proceed without claiming skill use.
- 若检索状态为：NO_SKILL_HIT，请在不声称使用技能的情况下继续。


- If Retrieval Status: SKILL_HIT, use retrieved skills only as constraints.
- 若检索状态为：SKILL_HIT，请仅将检索到的技能作为约束使用。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Use the exact Source path already returned.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- - 使用已返回的精确源路径。


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Prefer adapting retrieved scripts over broader re-implementation.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- - 优先适配检索到的脚本，而非进行更广泛的重新实现。


---



This second excerpt shows that the agent-facing interface is itself part of the method. It does not merely expose a search command. It constrains when retrieval is called, how the returned bundle is interpreted, and how aggressively the downstream agent is allowed to branch away from authoritative local implementations. The resulting effect is to make retrieval operational rather than decorative: the bundle is meant to narrow the search space immediately, not merely provide optional background context.
这段摘录表明，面向智能体的接口本身就是该方法的一部分。它不仅是提供一个搜索命令，还约束了何时调用检索、如何解读返回的包，以及允许下游智能体偏离权威本地实现的激进程度。其最终效果是使检索具有操作性而非装饰性：该包旨在立即缩小搜索空间，而非仅仅提供可选的背景信息。


Section Summary. Taken together, these examples show that prompt design in GoS is not generic scaffolding. The internal prompts constrain how semantic structure enters the graph; the external interface constrains how retrieved structure enters the agent's working context. The two layers therefore form an interface contract between offline graph construction and online agent execution. This contract is especially important in our setting because many
本节总结。综上所述，这些示例表明 GoS 中的提示词设计并非通用的脚手架。内部提示词约束了语义结构如何进入图谱；外部接口约束了检索到的结构如何进入智能体的工作上下文。因此，这两层结构在离线图谱构建与在线智能体执行之间形成了一种接口契约。这种契约在我们的设置中尤为重要，因为许多


---



Input: Local skill documents $\mathcal{C}$ ,optional LLM services,linking budget $k$
输入：本地技能文档 $\mathcal{C}$，可选 LLM 服务，链接预算 $k$


Output: Typed directed graph $G = \left( {V,E}\right)$ and vector index over normalized skills
输出：类型化有向图 $G = \left( {V,E}\right)$ 以及基于归一化技能的向量索引


Initialize empty node set $V$ and edge set $E$ .
初始化空节点集 $V$ 和边集 $E$。


For each skill document $d \in  \mathcal{C}$ :
对于每个技能文档 $d \in  \mathcal{C}$：


&nbsp;&nbsp;&nbsp;&nbsp;Parse YAML frontmatter and markdown structure from SKILL .md.
&nbsp;&nbsp;&nbsp;&nbsp;解析 SKILL .md 中的 YAML 前置元数据和 Markdown 结构。


&nbsp;&nbsp;&nbsp;&nbsp;Extract deterministic fields: name, description, inputs, outputs, tags, tooling, Source:
&nbsp;&nbsp;&nbsp;&nbsp;提取确定性字段：名称、描述、输入、输出、标签、工具、源：


path, and script entrypoints.
路径和脚本入口点。


&nbsp;&nbsp;&nbsp;&nbsp;If retrieval-critical semantic fields are incomplete, run constrained semantic completion to
&nbsp;&nbsp;&nbsp;&nbsp;如果检索关键的语义字段不完整，运行受限语义补全以


fill capability, inputs, outputs, and example tasks.
填充能力、输入、输出和示例任务。


&nbsp;&nbsp;&nbsp;&nbsp;Serialize the result as a normalized skill node $v$ and add $v$ to $V$ .
&nbsp;&nbsp;&nbsp;&nbsp;将结果序列化为归一化技能节点 $v$ 并将 $v$ 添加到 $V$ 中。


For each ordered pair of nodes $\left( {u,v}\right)$ in a bounded candidate pool:
对于有限候选池中的每一对有序节点 $\left( {u,v}\right)$：


&nbsp;&nbsp;&nbsp;&nbsp;Compute producer-consumer overlap between outputs of $u$ and inputs of $v$ .
&nbsp;&nbsp;&nbsp;&nbsp;计算 $u$ 的输出与 $v$ 的输入之间的生产者-消费者重叠度。


&nbsp;&nbsp;&nbsp;&nbsp;If overlap exceeds threshold,add typed dependency edge $u \rightarrow  v$ .
&nbsp;&nbsp;&nbsp;&nbsp;若重叠度超过阈值，则添加类型化依赖边 $u \rightarrow  v$ 。


For each node $u$ :
对于每个节点 $u$ ：


&nbsp;&nbsp;&nbsp;&nbsp;Form a sparse candidate set using lexical similarity, semantic neighbors, and I/O-based
&nbsp;&nbsp;&nbsp;&nbsp;利用词法相似度、语义邻居及基于输入/输出的


expansion.
扩展，构建稀疏候选集。


&nbsp;&nbsp;&nbsp;&nbsp;Run constrained relation validation on this candidate set.
&nbsp;&nbsp;&nbsp;&nbsp;对该候选集执行约束关系验证。


&nbsp;&nbsp;&nbsp;&nbsp;Add validated workflow,semantic,and alternative edges to $E$ .
&nbsp;&nbsp;&nbsp;&nbsp;将验证后的工作流边、语义边及替代边添加至 $E$ 。


Build an embedding index over the normalized node representations.
基于归一化后的节点表示构建嵌入索引。


Persist the typed graph, vector index, and retrieval metadata to the GoS workspace.
将类型化图、向量索引及检索元数据持久化至 GoS 工作空间。


---



Algorithm 1: Offline graph construction for GoS
算法 1：GoS 离线图构建


failures are not pure retrieval misses, but retrieval-hit trajectories in which the agent still drifts unless the interface strongly biases it toward verifier-minimal reuse. The appendix evidence should therefore be read as support for a broader methodological claim: in graph-augmented agent systems, retrieval quality depends not only on what is indexed, but also on how the retrieved structure is exposed and behaviorally enforced downstream.
失败并非单纯的检索缺失，而是指在检索命中的轨迹中，除非接口强烈引导智能体进行验证器最小化复用，否则智能体仍会发生偏移。因此，附录中的证据应被视为对更广泛方法论主张的支持：在图增强智能体系统中，检索质量不仅取决于索引内容，还取决于检索到的结构如何被呈现以及在下游如何进行行为强制。


## D Core Retrieval Pseudocode
## D 核心检索伪代码


Presentation Goal. For completeness, we provide pseudocode for the two main algorithmic stages of GoS: offline graph construction and online structural retrieval. The presentation is intentionally close to the implementation, but abstracted enough to foreground the method rather than the surrounding engineering details.
展示目标。为保持完整性，我们提供了 GoS 两个主要算法阶段的伪代码：离线图构建与在线结构化检索。该展示在逻辑上尽可能贴近实现，但在抽象层面进行了处理，旨在突出方法本身而非周边的工程细节。


Implementation Correspondence. Algorithm 1 corresponds to the parser-first normalization and relation-induction logic in the GoS implementation. Algorithm 2 corresponds to the retrieval path and the reverse-aware Personalized PageRank utilities. In practice, these stages additionally include engineering details such as workspace bootstrapping, embedding-dimension checks, source-path rewriting for containerized environments, and context clipping under hard character budgets. We omit those details from the pseudocode because they are not conceptually central, but they are nevertheless important for stable end-to-end deployment.
实现对应关系。算法 1 对应 GoS 实现中的解析器优先归一化与关系归纳逻辑。算法 2 对应检索路径及反向感知个性化 PageRank 工具。在实践中，这些阶段还包含诸如工作空间引导、嵌入维度检查、容器化环境下的源路径重写以及硬字符限制下的上下文截断等工程细节。我们从伪代码中省略了这些细节，因为它们在概念上并非核心，但对于稳定的端到端部署而言依然重要。


## E Error Analysis
## E 错误分析


Error Taxonomy. We distinguish retrieval-side errors from downstream execution failures, since these correspond to different limits of the overall system. A retrieval method can fail because it never surfaces the correct skill, because it retrieves an incomplete bundle that omits critical prerequisites, or because it produces a broadly adequate bundle that is subsequently misused by the downstream agent. Treating these failure modes separately is important for attributing gains correctly: GoS is designed to improve retrieval completeness, but it cannot by itself eliminate planning or execution failures once a bundle has already been provided. Across our trajectories, several of the most informative failures are not simple
错误分类。我们将检索侧错误与下游执行失败区分开来，因为它们对应于整个系统不同的局限性。检索方法可能因以下原因失败：未能呈现正确的技能、检索到的不完整包缺失关键先决条件，或生成的包虽大体适宜但被下游智能体误用。将这些失败模式分开处理对于正确归因收益至关重要：GoS 旨在提高检索完整性，但一旦提供了技能包，它本身无法消除规划或执行层面的失败。在我们的轨迹分析中，一些最具信息量的失败并非简单的


Input: Query $q$ ,prebuilt GoS workspace,retrieval budget $\tau$
输入：查询 $q$ ，预构建的 GoS 工作空间，检索预算 $\tau$


Output: Bounded execution bundle $B\left( q\right)$
输出：有界执行包 $B\left( q\right)$


Optionally rewrite $q$ into a compact schema containing goal,operations,artifacts,constraints, and keywords.
可选：将 $q$ 重写为包含目标、操作、工件、约束和关键字的紧凑模式。


Retrieve semantic seed candidates from the vector index.
从向量索引中检索语义种子候选。


Retrieve lexical seed candidates from normalized node fields.
从归一化节点字段中检索词法种子候选。


Merge the candidate pools and construct a seed distribution $\mathbf{p}$ .
合并候选池并构建种子分布 $\mathbf{p}$ 。


Build a typed transition matrix over the graph.
在图上构建类型化转移矩阵。


Add forward transition mass for each stored edge.
为每个存储的边添加前向转移权重。


Add relation-specific reverse mass, with the largest reverse coefficient on dependency edges.
添加特定于关系的逆向权重，其中依赖边具有最大的逆向系数。


Run reverse-aware Personalized PageRank until convergence to obtain graph scores ${\mathbf{s}}^{ \star  }$ .
运行感知逆向的个性化 PageRank 直至收敛，以获得图分数 ${\mathbf{s}}^{ \star  }$ 。


Rerank candidate skills using graph score plus direct field-level evidence from the query.
结合图分数和查询的直接字段级证据，对候选技能进行重排序。


Hydrate the ranked skills into agent-facing payloads with exact Source: paths and concise execution notes.
将排序后的技能填充为面向代理的有效载荷，包含精确的 Source: 路径和简洁的执行说明。


Truncate the hydrated bundle under per-skill and global context budgets.
在单项技能和全局上下文预算限制下截断填充后的包。


Return retrieval status, ranked bundle summary, bounded agent-facing context, and graph evidence among selected skills when it fits the budget.
返回检索状态、排序后的包摘要、有界的面向代理上下文，以及在预算允许时返回所选技能间的图证据。


Algorithm 2: Online graph-based skill retrieval
算法 2：在线基于图的技能检索


Table 8: Primary error modes in GoS-style retrieval experiments. The table separates retrieval failures, downstream execution failures, and infrastructure issues. retrieval misses, but long-horizon search failures in which the correct general tool family is present and the agent still does not converge to a verifier-passing implementation.
表 8：GoS 式检索实验中的主要错误模式。该表区分了检索失败、下游执行失败和基础设施问题。检索缺失，以及长程搜索失败（即存在正确的通用工具族，但代理仍无法收敛到通过验证器的实现）。


<table><tr><td>Error type</td><td>Typical symptom</td><td>Whether GoS helps</td></tr><tr><td>Retrieval miss</td><td>The correct skill exists in the library but is never surfaced, so the agent falls back to a from-scratch path.</td><td>Yes; better seed quality and graph completion can reduce this failure.</td></tr><tr><td>Partial retrieval</td><td>A topically relevant skill is retrieved, but a parser, setup routine, converter or constraint-carrying prerequisite is absent.</td><td>Yes; this is the main failure mode GoS is designed to reduce.</td></tr><tr><td>Good retrieval, bad execution</td><td>The retrieved bundle is broadly adequate, but the agent still drifts, over-engineers, or mismatches the verifier.</td><td>Only indirectly; this is primarily a backbone or planning issue.</td></tr><tr><td>Infrastructure failure</td><td>Build, environment, or startup failures prevent a meaningful episode from occurring.</td><td>No; these are excluded from model-quality comparisons.</td></tr></table>
<table><tbody><tr><td>错误类型</td><td>典型症状</td><td>GoS 是否有帮助</td></tr><tr><td>检索缺失</td><td>库中存在正确技能但未被检索到，导致智能体退回到从零开始的路径。</td><td>是；提高种子质量和完善图结构可减少此类故障。</td></tr><tr><td>部分检索</td><td>检索到了主题相关的技能，但缺少解析器、设置例程、转换器或带有约束的前置条件。</td><td>是；这是 GoS 旨在减少的主要故障模式。</td></tr><tr><td>检索良好，执行失败</td><td>检索到的包基本适用，但智能体仍出现偏差、过度设计或与验证器不匹配。</td><td>仅间接相关；这主要是骨干模型或规划层面的问题。</td></tr><tr><td>基础设施故障</td><td>构建、环境或启动失败，导致无法进行有效的任务片段。</td><td>否；此类故障不计入模型质量对比。</td></tr></tbody></table>


Retrieval Misses. A retrieval miss occurs when the correct skill exists in the repository but is not surfaced at all. In this regime, the agent is forced onto a generic from-scratch path, so any downstream failure should be attributed primarily to the retriever rather than to execution drift. Misses typically arise when the query language does not overlap strongly with the skill description, when the task is phrased around a downstream objective but the critical skill is an upstream parser or setup utility, or when semantic retrieval overweights topical similarity relative to executable relevance. A representative example is dapt-intrusion-detection. In the failed GoS-style trajectory, the agent issued graphskills-query but did not recover pcap-analysis, instead receiving an irrelevant bundle that included items such as dc-power-flow and dialogue-graph. By contrast, the stronger baseline trajectory opened pcap-analysis and reused its tested helper code. The resulting difference in behavior is characteristic of a true retrieval miss: once the relevant analysis skill is absent, the task degrades into from-scratch implementation and fails the verifier.
检索缺失。当正确的技能存在于库中但完全未被检索到时，即发生检索缺失。在这种情况下，智能体被迫从零开始执行，因此任何后续失败应主要归咎于检索器而非执行偏差。缺失通常源于查询语言与技能描述的语义重叠不足、任务描述侧重于下游目标而关键技能属于上游解析器或设置工具，或者语义检索过度权衡了主题相似性而忽视了可执行相关性。dapt-intrusion-detection 是一个典型案例。在失败的 GoS 轨迹中，智能体发出了 graphskills-query 但未能获取 pcap-analysis，反而收到了包含 dc-power-flow 和 dialogue-graph 等无关项的集合。相比之下，更强的基线轨迹调用了 pcap-analysis 并复用了其经过测试的辅助代码。这种行为差异是典型的检索缺失：一旦缺乏相关分析技能，任务就会退化为从零实现，从而无法通过验证器。


Partial Retrieval. Partial retrieval is more subtle and often more important. Here, the retrieved bundle contains an obviously relevant high-level skill, but omits one or more prerequisite helpers needed for successful completion. In our setting, these missing items are often parsers, format converters, preprocessing utilities, setup routines, or constraint-carrying reference skills. This is precisely the regime in which graph-aware retrieval is intended to help: once a high-level skill is matched as a seed, reverse-aware propagation can recover supporting skills that are weak semantic matches to the raw query but strong structural neighbors in the skill graph. earthquake-phase-association illustrates the boundary of this idea. In the stronger all-skills trajectory, the agent assembled a coherent seismic stack including gamma-phase-associator, obspy-data-api, seisbench-model-api, and seismic-picker-selection, and the task passed. In the corresponding GoS case, the graph retrieval did bring in a partially relevant seismic bundle, but the resulting context was still less complete, and the task failed with reward 0.0 . This suggests that structural retrieval helps only when the recovered neighborhood is sufficiently complete to support the downstream pipeline, not merely when one or two domain-relevant skills are present.
部分检索。部分检索更为隐蔽，且往往更为关键。在这种情况下，检索到的集合包含明显相关的高级技能，但遗漏了一个或多个成功完成任务所需的先决辅助技能。在我们的设置中，这些缺失项通常是解析器、格式转换器、预处理工具、设置例程或带有约束的参考技能。这正是图感知检索旨在解决的问题：一旦高级技能被匹配为种子，反向感知传播就能找回那些与原始查询语义匹配度弱、但在技能图中具有强结构关联的支撑技能。earthquake-phase-association 展示了这一理念的边界。在更强的全技能轨迹中，智能体组合了一个连贯的地震分析栈，包括 gamma-phase-associator、obspy-data-api、seisbench-model-api 和 seismic-picker-selection，任务顺利通过。而在对应的 GoS 案例中，图检索确实引入了一个部分相关的地震集合，但所得上下文仍不够完整，导致任务以 0.0 的奖励失败。这表明，结构化检索只有在恢复的邻域足以支撑下游流水线时才有效，而非仅仅检索到一两个领域相关技能即可。


Good Retrieval, Bad Execution. Some failures occur even when the retrieved bundle is broadly adequate. In these cases, the agent may still over-generalize the task, ignore a retrieved authoritative interface, implement unnecessary functionality, or fail to align with the verifier. These episodes matter because they bound what can be credited to retrieval alone. They also motivate the conservative agent-facing instructions used in our environments, which emphasize verifier-minimal solutions, direct use of returned local source paths, and avoidance of unnecessary branching. energy-market-pricing is a representative example: both all-skills and GoS had access to the relevant economic-dispatch / power-flow skill family and both eventually passed, but the all-skills trajectory required substantially more agent time before converging. This is not a retrieval miss; it is a difference in how efficiently a broadly adequate bundle is converted into a verifier-passing plan. Conversely, adaptive-cruise-control shows the opposite failure mode: the retrieved bundle included clearly relevant control skills such as pid-controller, mpc-horizon-tuning, vehicle-dynamics, and simulation-metrics, yet the run still finished with reward 0.0 . In that case the failure is better described as long-horizon execution drift or verifier mismatch rather than poor retrieval.
检索良好，执行不佳。即使检索到的集合大体适用，仍可能出现失败。在这种情况下，智能体可能仍会过度泛化任务、忽略检索到的权威接口、实现不必要的功能，或未能与验证器对齐。这些片段之所以重要，是因为它们界定了仅凭检索本身能达到的上限。它们也促使我们在环境中使用保守的智能体指令，强调验证器最小化方案、直接使用返回的本地源路径以及避免不必要的分支。energy-market-pricing 是一个代表性例子：全技能和 GoS 都能访问相关的 economic-dispatch / power-flow 技能族，且最终都通过了任务，但全技能轨迹在收敛前消耗了更多的智能体时间。这并非检索缺失，而是将大体适用的集合转化为通过验证器计划的效率差异。相反，adaptive-cruise-control 展示了相反的失败模式：检索到的集合包含了 pid-controller、mpc-horizon-tuning、vehicle-dynamics 和 simulation-metrics 等明显相关的控制技能，但运行结果仍为 0.0 奖励。在这种情况下，失败更应归因于长程执行偏差或验证器不匹配，而非检索质量不佳。


A related example is dialogue-parser. Multiple conditions had access to relevant task structure, yet only the strongest GoS trajectory converted that context into a full pass, while other conditions remained at partial reward. This again indicates that the dominant bottleneck was not the absence of any relevant skill at all, but how effectively the agent translated the available skill context into the exact output expected by the verifier.
一个相关的例子是 dialogue-parser。多种条件下都能访问相关的任务结构，但只有最强的 GoS 轨迹将其转化为完全通过，而其他条件仅获得部分奖励。这再次表明，主要的瓶颈不在于缺乏相关技能，而在于智能体将可用的技能上下文转化为验证器所期望的精确输出的有效性。


Infrastructure Failures. Finally, a subset of observed failures are not retrieval failures at all, but infrastructure failures such as environment build issues, setup crashes, or startup timeouts before a substantive episode begins. These cases are methodologically important but conceptually distinct: they should be tracked for experiment hygiene and rerun logic, yet they should not be interpreted as evidence against the quality of the retrieval method or the underlying model. Representative examples include Docker / BuildKit failures such as layer does not exist on dapt-intrusion-detection, missing compiler toolchains for obspy-dependent tasks, and logging failures that leave some trials with incomplete session traces or null token fields. These episodes matter operationally because they require reruns and infrastructure fixes, but they are not evidence about the retrieval quality of GoS, vector retrieval, or all-skills. For this reason, we treat them as experiment-hygiene issues and exclude them from method-quality interpretation whenever possible.
基础设施故障。最后，观察到的一部分失败并非检索失败，而是基础设施故障，例如环境构建问题、设置崩溃或在实质性片段开始前的启动超时。这些案例在方法论上很重要，但在概念上是独立的：它们应被记录以用于实验清理和重试逻辑，但不应被解读为反对检索方法或底层模型质量的证据。代表性例子包括 Docker / BuildKit 故障（如 dapt-intrusion-detection 上的层不存在）、obspy 相关任务缺少编译器工具链，以及导致部分试验会话跟踪不完整或标记字段为空的日志记录故障。这些片段在操作上很重要，因为它们需要重试和基础设施修复，但它们不能证明 GoS、向量检索或全技能的检索质量。因此，我们将其视为实验清理问题，并尽可能将其从方法质量的解读中排除。


## F Qualitative Analysis
## F 定性分析


Section Framing. We next examine a set of trajectory-grounded qualitative cases and compare the concrete skill evidence that actually entered the agent's working context in each condition. Table 9 reports the skills that materially shaped each run: for GoS and Vector Skills, these are the skills surfaced by the retrieval call and then used downstream, while for Vanilla Skills they are the skills the agent explicitly opened from the mounted library. This keeps the comparison grounded in executed trajectories rather than hypothetical retrieval quality.
章节框架。接下来，我们考察一组基于轨迹的定性案例，并比较在每种条件下实际进入智能体工作上下文的具体技能证据。表9列出了对每次运行产生实质性影响的技能：对于GoS和向量技能（Vector Skills），这些是检索调用所呈现并随后在下游使用的技能；而对于原生技能（Vanilla Skills），则是智能体从挂载库中显式打开的技能。这使得比较基于已执行的轨迹，而非假设的检索质量。


Across the cases below, we focus on a single question: does the method expose a compact, execution-ready bundle early enough to change the trajectory? The main qualitative difference is often not whether a relevant skill exists somewhere in the repository, but whether the agent receives the right subset in a form that can be operationalized under the task budget.
在下述案例中，我们聚焦于一个核心问题：该方法是否足够早地暴露了一个紧凑且可立即执行的工具包，从而改变轨迹？主要的定性差异往往不在于存储库中是否存在相关技能，而在于智能体是否以一种能在任务预算内被操作的形式接收到了正确的子集。


Case Study 1: Pedestrian Traffic Counting. The clearest intermediate case in our trajectories is pedestrian-traffic-counting. The task requires frame extraction, reliable pedestrian counting, and structured output generation. GoS surfaced a compact visual pipeline centered on gemini-count-in-video, video-frame-extraction, and openai-vision, and achieved the strongest outcome among the three conditions (0.417). The Vanilla Skills baseline did eventually open relevant helpers, including gemini-count-in-video, video-frame-extraction, and object_counter, but reached only a partial score (0.267). The Vector Skills run performed worst (0.041): although it issued the retrieval call, the retrieved context was not converted into a workable plan. This example is useful because it is not a pure pass/fail contrast. Vanilla Skills does locate relevant functionality, but GoS exposes a smaller and more coherent bundle that appears easier to operationalize within the available task budget.
案例研究1：行人交通计数。在我们轨迹中最清晰的中间案例是行人交通计数（pedestrian-traffic-counting）。该任务需要帧提取、可靠的行人计数以及结构化输出生成。GoS呈现了一个以gemini-count-in-video、video-frame-extraction和openai-vision为核心的紧凑视觉流水线，并在三种条件下取得了最强结果（0.417）。原生技能基线最终确实打开了相关辅助工具，包括gemini-count-in-video、video-frame-extraction和object_counter，但仅获得了部分分数（0.267）。向量技能运行表现最差（0.041）：尽管它发出了检索调用，但检索到的上下文并未转化为可行的计划。此例很有参考价值，因为它并非单纯的成功/失败对比。原生技能确实定位到了相关功能，但GoS暴露出的工具包更小、更连贯，在可用任务预算内似乎更容易操作。


Case Study 2: Flood Risk Analysis. The flood-risk-analysis task illustrates a different regime: both GoS and Vanilla Skills succeed, but GoS exposes the required chain with much less search friction. In this task the correct workflow is not generic time-series analysis; it is specifically the combination of usgs-data-download for measurements, nws-flood-thresholds for stage cutoffs, and flood-detection for aggregation and comparison. GoS surfaced exactly this bundle and passed with reward 1.0. The Vanilla Skills baseline also passed with reward 1.0, but only after the agent explicitly searched through the large mounted library and opened the same family of skills. Vector Skills, by contrast, issued the retrieval command but never translated retrieval into a usable flood-analysis bundle, and the run failed with reward 0.0 . This case is useful because it does not primarily show a reward gap between GoS and Vanilla Skills; instead, it shows that when the right execution chain exists in the repository, GoS mainly helps by making that chain explicit earlier in the trajectory.
案例研究2：洪水风险分析。洪水风险分析（flood-risk-analysis）任务展示了另一种情况：GoS和原生技能均获得成功，但GoS以更少的搜索摩擦暴露了所需的链条。在此任务中，正确的工作流并非通用的时间序列分析，而是usgs-data-download（用于测量）、nws-flood-thresholds（用于阶段截止值）和flood-detection（用于聚合与比较）的特定组合。GoS准确呈现了这一组合并以1.0的奖励通过。原生技能基线也以1.0的奖励通过，但前提是智能体必须显式搜索庞大的挂载库并打开同一系列的技能。相比之下，向量技能发出了检索命令，却未能将检索结果转化为可用的洪水分析组合，最终以0.0的奖励失败。此例很有意义，因为它主要展示的不是GoS与原生技能之间的奖励差距，而是当存储库中存在正确的执行链时，GoS主要通过在轨迹早期明确该链条来提供帮助。


Case Study 3: Travel Planning. The travel-planning task is informative precisely because all three conditions surfaced clearly relevant travel skills. In the GoS run, the retrieved context centered on search-cities, search-accommodations, search-attractions, search-driving-distance, and search-restaurants, which is very close to the intended tool chain for the task. The Vanilla Skills baseline likewise opened essentially the same family of skills after searching through the library. Vector Skills also surfaced and used this same cluster of search-* skills, and that run passed the verifier with reward 1.0. This example sharpens the qualitative claim of the paper. The advantage of GoS is not that flat semantic retrieval can never recover the correct skill family; rather, it is that GoS more reliably exposes a compact and coherent bundle early in the episode. When Vector Skills does succeed, as it does here, its behavior becomes qualitatively much closer to GoS than to a clean retrieval miss.
案例研究3：旅行规划。旅行规划（travel-planning）任务之所以具有启发性，恰恰是因为所有三种条件都清晰地呈现了相关的旅行技能。在GoS运行中，检索到的上下文集中在search-cities、search-accommodations、search-attractions、search-driving-distance和search-restaurants，这非常接近该任务预期的工具链。原生技能基线在搜索库后同样打开了基本相同的技能系列。向量技能也呈现并使用了这一相同的search-*技能集群，且该运行以1.0的奖励通过了验证。此例强化了本文的定性主张。GoS的优势不在于扁平语义检索永远无法恢复正确的技能系列，而在于GoS能更可靠地在片段早期暴露一个紧凑且连贯的组合。当向量技能确实成功时（如本例），其行为在定性上会比单纯的检索失败更接近GoS。


Case Study 4: Network Intrusion Detection. A clean GoS-positive example is dapt-intrusion-detection. In this case, GoS surfaced pcap-analysis together with adjacent analysis helpers such as pcap-triage-tshark and threat-detection, and the task passed. By contrast, the corresponding vector condition retrieved unrelated automation-oriented skills rather than a usable PCAP analysis bundle, while the all-skills condition still failed despite full library access. This case is a useful counterpart to the retrieval-miss pattern discussed in the Error Analysis section: once retrieval surfaces the right analysis bundle, the task becomes a reuse problem rather than a from-scratch reverse-engineering problem.
案例研究4：网络入侵检测。一个清晰的GoS正面案例是dapt-intrusion-detection。在此案例中，GoS呈现了pcap-analysis以及诸如pcap-triage-tshark和threat-detection等相邻的分析辅助工具，任务顺利通过。相比之下，相应的向量条件检索到的是无关的自动化导向技能，而非可用的PCAP分析组合；而全技能条件（all-skills）尽管拥有完整的库访问权限，依然失败了。此案例是错误分析章节中所讨论的“检索缺失”模式的有效对应：一旦检索呈现了正确的分析组合，任务就变成了重用问题，而非从零开始的逆向工程问题。


Case Study 5: Dialogue Parsing. The dialogue-parser examples show a strong gradient across methods. GoS converted the task into a full pass while exposing a compact bundle centered on dialogue_graph, together with structural helpers such as obj-exporter, browser-testing, and parser-oriented support. Vanilla Skills eventually improved once it explicitly opened dialogue_graph, and Vector Skills also reached a substantial partial score, but neither condition showed the same level of structured completeness as the strongest GoS trajectory. This case illustrates a pattern that appears repeatedly: once the right latent-representation skill is surfaced early, the rest of the pipeline becomes much easier for the agent to operationalize.
案例研究 5：对话解析。dialogue-parser 示例展示了不同方法间显著的性能梯度。GoS 将任务转化为一次完整执行，同时提供了一个以 dialogue_graph 为核心的紧凑工具包，并辅以 obj-exporter、browser-testing 和 parser-oriented 等结构化辅助工具。Vanilla Skills 在显式调用 dialogue_graph 后最终有所改进，Vector Skills 也获得了相当高的部分得分，但两者均未展现出与最强 GoS 轨迹同等水平的结构完整性。该案例揭示了一个反复出现的模式：一旦尽早呈现出正确的潜在表征技能，代理后续的执行流程就会变得容易得多。


Case Study 6: Earthquake Phase Association. earthquake-phase-association is a useful negative case for GoS because it shows that structural retrieval does not help automatically when the recovered neighborhood is still incomplete. In the strongest all-skills trajectory, the agent assembled a seismic processing stack including gamma-phase-associator, obspy-data-api, obspy-datacenter-client, seisbench-model-api, and seismic-picker-selection, and the task passed. The corresponding GoS case surfaced only a weaker subset, centered on gamma-phase-associator, seisbench-model-api, and seismic-picker-selection, with an irrelevant distraction skill mixed into the bundle, and the task failed. This is exactly the kind of case that is easy to miss if one looks only at whether some domain-relevant skill was retrieved. The qualitative difference is that the all-skills trajectory assembled a more execution-complete stack, while the GoS trajectory remained one step short of the required pipeline.
案例研究 6：地震震相关联。earthquake-phase-association 是 GoS 的一个有益负面案例，它表明当检索到的邻域仍不完整时，结构化检索无法自动提供帮助。在最强的“全技能”轨迹中，代理组装了一个包含 gamma-phase-associator、obspy-data-api、obspy-datacenter-client、seisbench-model-api 和 seismic-picker-selection 的地震处理栈，任务顺利通过。而对应的 GoS 案例仅呈现了一个较弱的子集，以 gamma-phase-associator、seisbench-model-api 和 seismic-picker-selection 为核心，且混入了一个无关的干扰技能，导致任务失败。这正是那种仅关注是否检索到领域相关技能就容易忽略的案例。其定性差异在于，“全技能”轨迹组装了一个执行完备的栈，而 GoS 轨迹则比所需流程少了一步。


Case Study 7: Energy Market Pricing. A final useful case is energy-market-pricing, where all-skills and GoS both passed but with very different trajectory quality. The all-skills condition explicitly used dc-power-flow and economic-dispatch, while GoS surfaced a broader but still coherent optimization bundle including dc-power-flow, power-flow-data, locational-marginal-prices, and casadi-ipopt-nlp. Both runs eventually passed the verifier, but the trajectory quality differed sharply: GoS converted the retrieved bundle into a solution with substantially less agent-side search. This is one of the clearest examples in which the main value of GoS is not higher reward, but a shorter path from retrieval to execution.
案例研究 7：能源市场定价。最后一个有用的案例是 energy-market-pricing，其中“全技能”和 GoS 均通过了任务，但轨迹质量迥异。“全技能”条件显式使用了 dc-power-flow 和 economic-dispatch，而 GoS 呈现了一个更广泛但逻辑连贯的优化工具包，包括 dc-power-flow、power-flow-data、locational-marginal-prices 和 casadi-ipopt-nlp。两次运行最终都通过了验证，但轨迹质量差异明显：GoS 将检索到的工具包转化为解决方案时，代理侧的搜索量显著减少。这是最清晰的案例之一，证明了 GoS 的核心价值不在于更高的奖励，而在于从检索到执行的路径更短。


Case Study 8: 3D Scan Calculation. The 3d-scan-calc task serves as a useful control because all three conditions can succeed when they recover the same latent geometry bottleneck. GoS exposed mesh-analysis together with adjacent geometric helpers, directly matching the preprocessing structure of the task. Vanilla Skills also reached a passing solution once the agent opened mesh-analysis, but the surrounding library context was notably noisier. Vector Skills likewise passed when it surfaced mesh-analysis together with geometry-oriented companions such as obj-exporter, pymatgen, and threejs. The qualitative lesson is therefore not that GoS is uniquely capable of solving the task; rather, when all methods recover a geometry-centered bundle, all can succeed, and the remaining difference is how directly that bundle is exposed.
案例研究 8：3D 扫描计算。3d-scan-calc 任务是一个有用的对照组，因为当三种方法都能恢复相同的潜在几何瓶颈时，它们都能成功。GoS 呈现了 mesh-analysis 及其相邻的几何辅助工具，直接匹配了任务的预处理结构。Vanilla Skills 在代理调用 mesh-analysis 后也找到了通过方案，但周围的库上下文明显更嘈杂。Vector Skills 在呈现 mesh-analysis 以及 obj-exporter、pymatgen 和 threejs 等几何相关配套工具时也同样通过了任务。因此，定性结论并非 GoS 在解决该任务上具有唯一能力；相反，当所有方法都能恢复以几何为中心的工具包时，它们都能成功，剩下的区别仅在于该工具包呈现的直接程度。


Case Study 9: Adaptive Cruise Control. adaptive-cruise-control is a useful failure case because all three conditions surfaced highly plausible control-related skills and still failed. GoS exposed imc-tuning-rules, pid-controller, safety-interlocks, and vehicle-dynamics, while Vector Skills surfaced an even more explicit control bundle including pid-controller, mpc-horizon-tuning, integral-action-design, simulation-metrics, and vehicle-dynamics. The Vanilla Skills condition also had access to a broad control-oriented context, yet none of the three settings converged to a passing solution. This case is important precisely because it is not a retrieval miss. It shows that once the task requires a long control-design and verifier-alignment chain, even a qualitatively good skill bundle may not be enough; the dominant bottleneck shifts from retrieval to execution discipline.
案例研究 9：自适应巡航控制。adaptive-cruise-control 是一个有用的失败案例，因为三种方法都呈现了高度合理的控制相关技能，但最终均告失败。GoS 呈现了 imc-tuning-rules、pid-controller、safety-interlocks 和 vehicle-dynamics，而 Vector Skills 呈现了一个更明确的控制工具包，包括 pid-controller、mpc-horizon-tuning、integral-action-design、simulation-metrics 和 vehicle-dynamics。Vanilla Skills 条件下也拥有广泛的控制导向上下文，然而三种设置均未收敛到通过方案。该案例之所以重要，恰恰因为它并非检索失误。它表明，一旦任务需要长链条的控制设计和验证器对齐，即使是定性上优秀的技能包也可能不足以解决问题；主要的瓶颈已从检索转移到了执行规范性上。


Case Study 10: Economic Detrending and Correlation. The econ-detrending-correlation task offers a complementary success case. GoS surfaced timeseries-detrending and converted the task into a full pass, while the all-skills condition failed to assemble a comparably coherent detrending-centered bundle. Vector Skills also reached a full pass, but with a noisier retrieved context whose skills were only weakly connected to the intended econometric workflow. This case is useful in two ways. First, it shows another task where surfacing the right latent preprocessing step, here detrending rather than raw correlation, materially changes the result. Second, it reinforces the lesson from travel-planning: vector retrieval can still succeed, but its successful episodes do not always arise from a bundle that is as semantically crisp or structurally interpretable as the one surfaced by GoS.
案例研究 10：经济去趋势与相关性。econ-detrending-correlation 任务提供了一个互补的成功案例。GoS 呈现了 timeseries-detrending 并将任务转化为完全通过，而“全技能”条件未能组装出同样连贯的、以去趋势为中心的工具包。Vector Skills 也实现了完全通过，但检索到的上下文较为嘈杂，且其中的技能与预期的计量经济学工作流关联较弱。该案例在两方面很有价值。首先，它展示了另一个任务，即呈现正确的潜在预处理步骤（此处为去趋势而非原始相关性）会实质性改变结果。其次，它强化了从旅行规划中得出的教训：向量检索虽然仍能成功，但其成功案例并不总是源于像 GoS 那样语义清晰或结构可解释的工具包。


Takeaway. Across all ten qualitative cases in this appendix, the main pattern is not simply that GoS retrieves skills with better topical overlap. Rather, GoS more often exposes a bundle that is already close to the executable decomposition of the task. The pedestrian-traffic-counting example shows a genuine middle case in which Vanilla Skills finds relevant tools but still underperforms the tighter GoS bundle. The flood-risk-analysis example shows that when the correct chain is available to multiple methods, GoS mainly reduces search friction and makes the intended execution path explicit earlier. The travel-planning, 3d-scan-calc, and econ-detrending-correlation examples show that Vector Skills can also succeed when it recovers the right family of skills, but these successes are most convincing when the retrieved bundle becomes qualitatively similar to what GoS surfaces directly. The dialogue-parser and dapt-intrusion-detection cases show how GoS can convert that structural advantage into clearer downstream wins. By contrast, earthquake-phase-association shows a real boundary condition in which GoS still falls short of an execution-complete bundle, while energy-market-pricing and adaptive-cruise-control show that even with broadly adequate retrieval, trajectory efficiency and verifier alignment remain separate bottlenecks. Taken together, these cases support the core claim of the paper: structural retrieval helps not only by improving relevance, but by presenting agents with a more execution-ready context.
结论。在本附录的所有十个定性案例中，主要规律并非仅仅是 GoS 检索到的技能具有更好的主题重合度。相反，GoS 更常提供一个已接近任务可执行分解的技能包。行人流量统计示例展示了一个典型的中间情况：Vanilla Skills 虽然找到了相关工具，但表现仍不及更紧凑的 GoS 技能包。洪水风险分析示例表明，当多种方法都能获取正确链条时，GoS 主要通过降低搜索阻力，更早地明确了预期的执行路径。旅行规划、3D 扫描计算和经济去趋势相关性示例显示，Vector Skills 在检索到正确的技能族时也能成功，但这些成功在检索到的技能包与 GoS 直接呈现的内容在定性上相似时才最具说服力。对话解析器和 DAPT 入侵检测案例展示了 GoS 如何将这种结构优势转化为更显著的下游成果。相比之下，地震震相关联展示了一个真正的边界条件，即 GoS 仍未能提供完整的执行包；而能源市场定价和自适应巡航控制则表明，即使检索结果大致充足，轨迹效率和验证器对齐仍是独立的瓶颈。综上所述，这些案例支持了本文的核心主张：结构化检索不仅通过提高相关性发挥作用，更通过为智能体提供更具执行准备的上下文来提升性能。


Table 9: Trajectory-grounded skill evidence from executed qualitative cases. USEFUL denotes skills that were clearly operationalized downstream; NOISY denotes retrieved or opened items that were tangential or not visibly used.
表 9：来自已执行定性案例的轨迹基础技能证据。USEFUL 表示在下游被明确操作的技能；NOISY 表示检索到或打开但属于无关或未见使用的项目。


<table><tr><td>Task</td><td>GoS Bundle</td><td>Vanilla Bundle</td><td>Vector Bundle</td></tr><tr><td>pedestrian-traffic-counting</td><td>Useful gemini-count-in-video; multimodal-fusion; openai-vision; video-frame-extraction NOISY threat-detection</td><td>Useful gemini-count-in-video; object_counter; openai-vision; video-frame-extraction Noisy alfworld-heat-object-with-appliance; alfworld-object-locator; broader noisy context</td><td>USEFUL none <br> Noisy <br> google-classroom-automation; rdkit; <br> salesforce-service-cloud-automation; <br> segmetrics-automation</td></tr><tr><td>flood-risk-analysis</td><td>USEFUL flood-detection; nws-flood-thresholds; usgs-data-download Noisy <br> time_series_anomaly_detection; <br> -21risk-automation</td><td>USEFUL flood-detection; nws-flood-thresholds; usgs-data-download</td><td>USEFUL none <br> NOISY leverly-automation; scienceworld-room-navigator; text-to-speech; broader noisy context</td></tr><tr><td>travel-planning</td><td>Useful <br> search-accommodations; search-attractions; search-cities; search-driving-distance; search-restaurants <br> Noisy search-flights; <br> fjsp-baseline-repair-with-downtime-and-policy</td><td>Useful <br> search-accommodations; search-attractions; search-cities; search-restaurants</td><td>Useful <br> search-accommodations; search-attractions; search-cities; <br> search-driving-distance; search-restaurants <br> Noisy additional noisy items</td></tr><tr><td>dapt-intrusion-detection</td><td>Useful pcap-analysis; pcap-triage-tshark <br> Noisy dc-power-flow; power-flow-data;</td><td>Useful no clearly reused core skill</td><td>Useful none <br> NOISY codacy-automation; jakarta-namespace; rootly-automation; broader noisy context</td></tr><tr><td>dialogue-parser</td><td>-21risk-automation Useful dialogue_graph; webshop-query-parser NOISY browser-testing; obj-exporter; alfworld-goal-interpreter Useful</td><td>Useful dialogue_graph</td><td>Useful none <br> NOISY docnify-automation; scienceworld-task-focuser: temporal-python-testing; broader noisy context Useful none</td></tr><tr><td>earthquake-phase-association</td><td>gamma-phase-associator; seisbench-model-api; seismic-picker-selection <br> NOISY flood-detection; <br> -21risk-automation</td><td>Useful <br> gamma-phase-associator; obspy-data-api; obspy-datacenter-client; seisbench-model-api; seismic-picker-selection <br> NOISY gamma-automation; seismic-automation</td><td>NOISY fixer-automation; maven-build-lifecycle; segmetrics-automation; broader noisy context</td></tr><tr><td>energy-market-pricing</td><td>Useful dc-power-flow; power-flow-data; locational-marginal-prices; casadi-ipopt-nlp NOISY -21risk-automation</td><td>Useful dc-power-flow; economic-dispatch</td><td>USEFUL power-flow-data NOISY aryn-automation; moxie-automation; mural-automation; broader noisy context</td></tr><tr><td>3d-scan-calc</td><td>Useful mesh-analysis; dyn-object-masks Noisy <br> scienceworld-circuit-builder; scienceworld-circuit-connector; <br> scienceworld-conductivity-tester</td><td>Useful mesh-analysis Noisy broader noisy context</td><td>Useful mesh-analysis; obj-exporter; pymatgen; threejs <br> Noisy <br> reflow-profile-compliance-toolkit</td></tr><tr><td>adaptive-cruise-control</td><td>Useful imc-tuning-rules; pid-controller; safety-interlocks; vehicle-dynamics Noisy -21risk-automation Useful</td><td>USEFUL no clearly reused core skill</td><td>Useful pid-controller; mpc-horizon-tuning; integral-action-design; simulation-metrics; vehicle-dynamics</td></tr><tr><td>econ-detrending-correlation</td><td>timeseries-detrending NOISY artifacts-builder; dyn-object-masks; mesh-analysis; <br> -21risk-automation</td><td>USEFUL no clearly reused <br> core skill</td><td>USEFUL none <br> NOISY breezy-hr-automation; scienceworld-object-classifier; <br> webshop-purchase-initiator; broader noisy context</td></tr></table>
<table><tbody><tr><td>任务</td><td>GoS 集合</td><td>原生集合</td><td>向量集合</td></tr><tr><td>行人流量统计</td><td>有用 gemini-count-in-video; multimodal-fusion; openai-vision; video-frame-extraction 噪声 threat-detection</td><td>有用 gemini-count-in-video; object_counter; openai-vision; video-frame-extraction 噪声 alfworld-heat-object-with-appliance; alfworld-object-locator; 更广泛的噪声上下文</td><td>有用 无 <br/> 噪声 <br/> google-classroom-automation; rdkit; <br/> salesforce-service-cloud-automation; <br/> segmetrics-automation</td></tr><tr><td>洪水风险分析</td><td>有用 flood-detection; nws-flood-thresholds; usgs-data-download 噪声 <br/> time_series_anomaly_detection; <br/> -21risk-automation</td><td>有用 flood-detection; nws-flood-thresholds; usgs-data-download</td><td>有用 无 <br/> 噪声 leverly-automation; scienceworld-room-navigator; text-to-speech; 更广泛的噪声上下文</td></tr><tr><td>旅行规划</td><td>有用 <br/> search-accommodations; search-attractions; search-cities; search-driving-distance; search-restaurants <br/> 噪声 search-flights; <br/> fjsp-baseline-repair-with-downtime-and-policy</td><td>有用 <br/> search-accommodations; search-attractions; search-cities; search-restaurants</td><td>有用 <br/> search-accommodations; search-attractions; search-cities; <br/> search-driving-distance; search-restaurants <br/> 噪声 其他噪声项</td></tr><tr><td>dapt-入侵检测</td><td>有用 pcap-analysis; pcap-triage-tshark <br/> 噪声 dc-power-flow; power-flow-data;</td><td>有用 无明确复用的核心技能</td><td>有用 无 <br/> 噪声 codacy-automation; jakarta-namespace; rootly-automation; 更广泛的噪声上下文</td></tr><tr><td>对话解析器</td><td>-21risk-automation 有用 dialogue_graph; webshop-query-parser 噪声 browser-testing; obj-exporter; alfworld-goal-interpreter 有用</td><td>有用 dialogue_graph</td><td>有用 无 <br/> 噪声 docnify-automation; scienceworld-task-focuser: temporal-python-testing; 更广泛的噪声上下文 有用 无</td></tr><tr><td>地震震相关联</td><td>gamma-phase-associator; seisbench-model-api; seismic-picker-selection <br/> 噪声 flood-detection; <br/> -21risk-automation</td><td>有用 <br/> gamma-phase-associator; obspy-data-api; obspy-datacenter-client; seisbench-model-api; seismic-picker-selection <br/> 噪声 gamma-automation; seismic-automation</td><td>噪声 fixer-automation; maven-build-lifecycle; segmetrics-automation; 更广泛的噪声上下文</td></tr><tr><td>能源市场定价</td><td>有用 dc-power-flow; power-flow-data; locational-marginal-prices; casadi-ipopt-nlp 噪声 -21risk-automation</td><td>有用 dc-power-flow; economic-dispatch</td><td>有用 power-flow-data 噪声 aryn-automation; moxie-automation; mural-automation; 更广泛的噪声上下文</td></tr><tr><td>3d扫描计算</td><td>有用 mesh-analysis; dyn-object-masks 噪声 <br/> scienceworld-circuit-builder; scienceworld-circuit-connector; <br/> scienceworld-conductivity-tester</td><td>有用 mesh-analysis 噪声 更广泛的噪声上下文</td><td>有用 mesh-analysis; obj-exporter; pymatgen; threejs <br/> 噪声 <br/> reflow-profile-compliance-toolkit</td></tr><tr><td>自适应巡航控制</td><td>有用 imc-tuning-rules; pid-controller; safety-interlocks; vehicle-dynamics 噪声 -21risk-automation 有用</td><td>有用 无明确复用的核心技能</td><td>有用 pid-controller; mpc-horizon-tuning; integral-action-design; simulation-metrics; vehicle-dynamics</td></tr><tr><td>经济去趋势相关性</td><td>timeseries-detrending 噪声 artifacts-builder; dyn-object-masks; mesh-analysis; <br/> -21risk-automation</td><td>有用 无明确复用的<br/>核心技能</td><td>有用 无 <br/> 噪声 breezy-hr-automation; scienceworld-object-classifier; <br/> webshop-purchase-initiator; 更广泛的噪声上下文</td></tr></tbody></table>