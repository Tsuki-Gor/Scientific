# SkillClaw: Let Skills Evolve Collectively with Agentic Evolver
# SkillClaw：让技能借助自主进化者实现协同演进


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_46_22_785381.jpg"/>



Ziyu Ma ${}^{1 * }$ , Shidong Yang ${}^{1 * }$ , Yuxiang Ji ${}^{1 * }$ , Xucong Wang ${}^{1 * }$ , Yong Wang ${}^{1 \dagger  }$ , Yiming Hu ${}^{1}$ , Tongwen Huang ${}^{1}$ , Xiangxiang Chu ${}^{1}$
马子钰 ${}^{1 * }$ , 杨士东 ${}^{1 * }$ , 季宇翔 ${}^{1 * }$ , 王旭聪 ${}^{1 * }$ , 王勇 ${}^{1 \dagger  }$ , 胡艺铭 ${}^{1}$ , 黄同文 ${}^{1}$ , 楚湘湘 ${}^{1}$


${}^{1}$ DreamX Team, ${}^{ * }$ Equal contribution, ${}^{ \dagger  }$ Project lead
${}^{1}$ 梦想X团队，${}^{ * }$ 平等贡献，${}^{ \dagger  }$ 项目负责人


Large language model (LLM) agents such as OpenClaw rely on reusable skills to perform complex tasks, yet these skills remain largely static after deployment. As a result, similar workflows, tool usage patterns, and failure modes are repeatedly rediscovered across users, preventing the system from improving with experience. While interactions from different users provide complementary signals about when a skill works or fails, existing systems lack a mechanism to convert such heterogeneous experiences into reliable skill updates. To address these issues, we present SkillClaw, a framework for collective skill evolution in multi-user agent ecosystems, which treats cross-user and over-time interactions as the primary signal for improving skills. SkillClaw continuously aggregates trajectories generated during use and processes them with an autonomous evolver, which identifies recurring behavioral patterns and translates them into updates to the skill set by refining existing skills or extending them with new capabilities. The resulting skills are maintained in a shared repository and synchronized across users, allowing improvements discovered in one context to propagate system-wide while requiring no additional effort from users. By integrating multi-user experience into ongoing skill updates, SkillClaw enables cross-user knowledge transfer and cumulative capability improvement, and experiments on WildClawBench show that limited interaction and feedback, it significantly improves the performance of Qwen3-Max in real-world agent scenarios.
像 OpenClaw 这样的通用大语言模型（LLM）智能体依赖可复用技能来完成复杂任务，但这些技能在部署后大多仍保持静态。因此，类似的工作流、工具使用模式和失败模式会在不同用户间被反复重新发现，阻碍系统随经验持续改进。尽管来自不同用户的交互能提供互补信号，帮助判断某项技能何时有效或失效，但现有系统缺乏将这类异质经验转化为可靠技能更新的机制。为解决这些问题，我们提出 SkillClaw，这是一种面向多用户智能体生态的集体技能演化框架，将跨用户和跨时间的交互视为改进技能的主要信号。SkillClaw 持续汇聚使用过程中生成的轨迹，并交由自主演化器处理；该演化器识别重复出现的行为模式，并通过优化现有技能或扩展新能力，将其转化为对技能集的更新。由此形成的技能保存在共享仓库中，并在用户间同步，使某一场景中发现的改进能够传播至整个系统，同时无需用户额外付出。通过将多用户经验融入持续的技能更新，SkillClaw 实现了跨用户知识迁移与能力的累积提升；在 WildClawBench 上的实验表明，在交互和反馈有限的情况下，它显著提升了 Qwen3-Max 在真实世界智能体场景中的性能。


Github: https://github.com/AMAP-ML/SkillClaw
Github：https://github.com/AMAP-ML/SkillClaw


## 1 Introduction
## 1 引言


Large language model (LLM) agents (Yao et al., 2022; Shinn et al., 2023) have rapidly made personal AI assistants practical in real-world settings, with systems such as OpenClaw enabling users to complete complex tasks through natural conversation. A user can now ask an agent to configure a service, debug an API call, or automate a multi-step workflow, relying on it to coordinate tool usage and intermediate reasoning. These capabilities are largely driven by skills, which encode structured procedures for interacting with tools and solving tasks. In current deployments, users typically select and install skills from a centralized skill hub to meet their needs, and these skills serve as the primary building blocks for agent behavior. However, the skill ecosystem remains largely static (Zhang et al., 2025b; Naihin et al., 2023; Song et al., 2026), as skills are manually installed and maintained and solutions discovered during interaction rarely persist beyond individual sessions.
大语言模型（LLM）智能体（Yao et al., 2022; Shinn et al., 2023）已迅速使个人 AI 助手在真实场景中变得实用，OpenClaw 等系统使用户能够通过自然对话完成复杂任务。如今，用户可以让智能体配置服务、调试 API 调用，或自动化多步骤工作流，依赖它来协调工具使用和中间推理。这些能力主要由技能驱动，技能编码了与工具交互和解决任务的结构化流程。在当前部署中，用户通常从集中式技能中心选择并安装技能以满足需求，这些技能构成了智能体行为的主要基石。然而，技能生态在很大程度上仍是静态的（Zhang et al., 2025b; Naihin et al., 2023; Song et al., 2026），因为技能是手动安装和维护的，而交互中发现的解决方案很少会超出单次会话而延续。


This limitation becomes evident in everyday usage. For example, users often ask agents to complete multi-step tasks such as automating data processing workflows, where failures frequently arise from subtle issues such as incorrect argument formats or mismatched tool calls. Through several rounds of trial and error, an agent may eventually arrive at a working solution or even a more stable procedure. However, these improvements remain confined to the current session and are not consolidated into the skill set or carried forward to future interactions. As similar tasks recur across different users and over time, the same patterns of failure and recovery are repeatedly observed, yet the system does not improve its behavior. This is fundamentally problematic because users operate in overlapping task spaces where similar workflows, tools, and failure modes are shared, but the system fails to leverage these recurring experiences. Consequently, each user is forced to rediscover solutions independently, preventing knowledge from accumulating at the system level. Therefore, the key challenge is not only to improve performance within a single session, but also to enable knowledge to accumulate and evolve across users.
这一局限在日常使用中尤为明显。例如，用户经常要求智能体完成多步骤任务，如自动化数据处理工作流，而失败往往源于细微问题，例如参数格式错误或工具调用不匹配。经过多轮反复试错，智能体最终可能找到可行的解决方案，甚至形成更稳定的流程。然而，这些改进仍局限于当前会话，既不会沉淀到技能集中，也不会延续到后续交互中。随着类似任务在不同用户之间和时间维度上反复出现，同样的失败与恢复模式不断上演，但系统的行为却并未随之改进。这在根本上是有问题的，因为用户处于重叠的任务空间中，共享相似的工作流、工具和失败模式，而系统却未能利用这些反复出现的经验。因此，每个用户都不得不独立重新发现解决方案，导致知识无法在系统层面累积。因此，关键挑战不仅是提升单次会话内的表现，还要让知识能够跨用户累积并演化。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_46_22_2c4c4c.jpg"/>



Figure 1 Overview of SkillClaw. SkillClaw enables collective skill evolution in a multi-user agent ecosystem through a closed-loop pipeline. Independent agents interact with their environments and produce structured session trajectories that preserve full action-feedback causal chains. These trajectories are aggregated across users and grouped by referenced skills, forming a shared evidence base that exposes consistent success patterns and recurring failure modes. An agentic evolver analyzes each skill-specific group and performs evidence-driven updates via refinement or creation, while preserving validated behaviors from successful executions. The updated skill repository is then synchronized back to all agents, allowing improvements discovered in one user's interaction to benefit others and continuously accumulate over time.
图1 SkillClaw 概览。SkillClaw 通过闭环流水线，在多用户智能体生态中实现集体技能演化。独立智能体与各自环境交互，生成保留完整动作-反馈因果链的结构化会话轨迹。这些轨迹在用户之间聚合，并按所引用的技能分组，形成共享证据基础，揭示稳定的成功模式和反复出现的失败模式。一个 agentic evolver 分析每个技能分组，并通过细化或创建执行基于证据的更新，同时保留来自成功执行的已验证行为。更新后的技能库随后同步回所有智能体，使一个用户交互中发现的改进能够惠及他人，并随时间持续累积。


Existing approaches to agent adaptation fail to support the accumulation and evolution of skills across users and over time. Memory-based methods store past trajectories for retrieval (Shinn et al., 2023; Zhao et al., 2024; Fang et al., 2025a; Tang et al., 2025; Ouyang et al., 2025a; Chhikara et al., 2025; Liu et al., 2026), but such records remain tied to specific instances and are difficult to generalize into improved behavior. Skill-based methods compress experience into structured instructions (Xia et al., 2026a; Zhang et al., 2025a, 2026b; Wu et al., 2025; Zhang et al., 2026a), yet treat the resulting skill library as a static resource that does not evolve through usage. While local refinement can improve individual agent instances, these improvements remain isolated and do not accumulate across users, leading to fragmented skills rather than collective improvement over time. What is missing is a mechanism that turns ordinary interactions into continuous skill evolution and enables skills to improve collectively across users.
现有的智能体适应方法无法支持技能在用户之间以及随时间的累积与演化。基于记忆的方法将过去的轨迹存储起来供检索（Shinn et al., 2023; Zhao et al., 2024; Fang et al., 2025a; Tang et al., 2025; Ouyang et al., 2025a; Chhikara et al., 2025; Liu et al., 2026），但这些记录仍绑定于特定实例，难以泛化为更好的行为。基于技能的方法将经验压缩为结构化指令（Xia et al., 2026a; Zhang et al., 2025a, 2026b; Wu et al., 2025; Zhang et al., 2026a），却将生成的技能库视为不会随使用而演化的静态资源。虽然局部细化可以改善单个智能体实例，但这些改进仍彼此孤立，无法在用户之间积累，最终导致的是碎片化的技能，而非随时间的集体提升。缺失的正是一种机制，能够将普通交互转化为持续的技能演化，并使技能在用户之间协同改进。


Building on this insight, we propose SkillClaw, a framework for skill collective evolution in multi-user OpenClaw-style agent ecosystems (Fig 1). SkillClaw adopts a centralized evolution architecture, where agents deployed across different users continuously generate interaction sessions during everyday usage. These trajectories are aggregated across users and over time as evidence of real-world task execution and are processed by a centralized evolution engine to drive skill updates. Given accumulated interaction trajectories, the evolver analyzes both successful and failed executions, identifies recurring issues and effective procedures, and updates the shared skill set by refining existing skills, creating new ones, or adjusting their descriptions. Unlike predefined pipelines, this evolution process is driven by an autonomous agent that performs open-ended reasoning over interaction evidence and directly edits skill definitions. The updated skills are then synchronized across agents, allowing improvements discovered in one context to propagate to future interactions across users and over time. This forms a continuous evolution loop in which interaction data drives skill updates, and updated skills improve subsequent interactions. From the user's perspective, this process requires no additional effort, as data collection, evolution, and synchronization all occur automatically in the background.
基于这一洞见，我们提出 SkillClaw，一个用于多用户 OpenClaw 风格智能体生态中技能集体演化的框架（图1）。SkillClaw 采用集中式演化架构，不同用户部署的智能体在日常使用中持续生成交互会话。这些轨迹在用户之间以及随时间聚合，作为真实世界任务执行的证据，并由集中式演化引擎处理以驱动技能更新。给定累积的交互轨迹，evolver 分析成功与失败的执行，识别反复出现的问题和有效流程，并通过细化现有技能、创建新技能或调整其描述来更新共享技能集。不同于预定义流水线，这一演化过程由一个自主智能体驱动，它基于交互证据进行开放式推理，并直接编辑技能定义。更新后的技能随后在各智能体间同步，使一个场景中发现的改进能够传播到未来的跨用户、跨时间交互中。这形成了一个持续演化闭环：交互数据驱动技能更新，更新后的技能又改进后续交互。从用户视角看，这一过程无需额外努力，因为数据收集、演化和同步都在后台自动进行。


This design introduces three key properties that distinguish SkillClaw from existing systems. First, Skill-Claw enables collective evolution, where knowledge from individual interactions contributes to a shared and continuously improving skill ecosystem. Second, it is fully automatic, with skill evolution driven by runtime interaction without manual curation or explicit user intervention. Third, it adopts an agentic evolution paradigm, where skill updates are produced through open-ended reasoning rather than predefined update rules, enabling flexible and context-aware improvements.
这一设计引入了三项将 SkillClaw 与现有系统区分开的关键特性。首先，Skill-Claw 支持集体演化，单个交互中的知识会贡献到共享且持续改进的技能生态中。其次，它完全自动化，技能演化由运行时交互驱动，无需人工筛选或用户显式干预。第三，它采用代理式演化范式，技能更新通过开放式推理而非预定义更新规则生成，从而实现灵活且具上下文感知的改进。


SkillClaw is designed as a general framework that is compatible with a wide range of Claw-style agent systems, including OpenClaw as well as variants such as CoPaw, IronClaw, PicoClaw, ZeroClaw, NanoClaw, and NemoClaw. We evaluate SkillClaw on WildClawBench using qwen3-max as the backbone model and simulate a multi-user deployment setting. Experimental results demonstrate that SkillClaw yields substantial improvements across tasks, highlighting the effectiveness of multi-user driven collective evolution for building continuously improving agent systems in real-world environments.
SkillClaw 被设计为一个通用框架，兼容广泛的 Claw 风格代理系统，包括 OpenClaw，以及 CoPaw、IronClaw、PicoClaw、ZeroClaw、NanoClaw 和 NemoClaw 等变体。我们在 WildClawBench 上使用 qwen3-max 作为基础模型对 SkillClaw 进行评估，并模拟多用户部署场景。实验结果表明，SkillClaw 在各项任务上都带来了显著提升，凸显了多用户驱动的集体演化在现实环境中构建持续改进的代理系统的有效性。


## 2 Method
## 2 方法


We present SkillClaw, a framework for collective skill evolution in a multi-user agent ecosystem (Fig 1). In our setting, different users independently interact with their own deployed OpenClaw agents, potentially across different devices, environments, and time. Although these interactions are isolated at runtime, they share a common behavioral space: similar workflows, overlapping tool usage, and recurring failure modes appear across users. SkillClaw builds on the observation that different users exercising the same skill under diverse contexts produce complementary views of that skill's behavioral boundary, revealing both the conditions under which it works and those under which it breaks. A single user rarely generates enough signal to separate a generalizable improvement from an idiosyncratic fix. Aggregating evidence across users provides the grounding that makes stable skill evolution possible.
我们提出 SkillClaw，这是一个用于多用户智能体生态中协同技能演化的框架（图 1）。在我们的设定中，不同用户会在各自部署的 OpenClaw 智能体上独立地进行交互，可能跨越不同设备、环境与时间。尽管这些交互在运行时彼此隔离，但它们共享同一个行为空间：相似的工作流、交叠的工具使用，以及反复出现的失败模式在不同用户之间都能观察到。SkillClaw 基于这样的观察：不同用户在不同情境下使用同一技能，会产生对该技能行为边界的互补视角，从而同时揭示它适用的条件以及失效的条件。单个用户很难生成足够的信号，以区分可泛化的改进与仅仅是特有的修补。将来自不同用户的证据进行汇总，提供了实现稳定技能演化所必需的基础。


Formally,let $\mathcal{S} = \left\{  {{s}_{1},\ldots ,{s}_{M}}\right\}$ denote a shared skill set,where each skill is a reusable procedural artifact. Each user interaction produces a session trajectory $\tau$ ,which records the full interaction loop: the prompt, the agent's actions, feedback from the environment or the user, and the final agent response. Given a set of trajectories $\mathcal{T} = \left\{  {\tau }_{i}\right\}$ collected across users,our goal is to update the shared skill set:
形式化地，令 $\mathcal{S} = \left\{  {{s}_{1},\ldots ,{s}_{M}}\right\}$ 表示一个共享技能集，其中每个技能都是可复用的程序化产物。每次用户交互都会生成一条会话轨迹 $\tau$，它记录完整的交互闭环：提示词、智能体的行动、来自环境或用户的反馈，以及最终的智能体回复。给定在不同用户之间收集到的一组轨迹 $\mathcal{T} = \left\{  {\tau }_{i}\right\}$，我们的目标是更新共享技能集：


$$
{\mathcal{S}}^{\prime } = \Phi \left( {\mathcal{S},\mathcal{T}}\right) ,
$$



such that improvements discovered in one interaction can benefit future users.
使得在一次交互中发现的改进能够惠及未来的用户。


### 2.1 From Isolated Sessions to Shared Evidence
### 2.1 从孤立会话到共享证据


Multi-user skill evolution requires converting a stream of isolated, heterogeneous interaction sessions into a form that supports cross-user reasoning. SkillClaw does this in two stages: it first structures individual sessions to preserve causal information, then aggregates them into a shared evidence base.
多用户的技能进化需要把一串孤立且异构的交互会话转换成一种能够支持跨用户推理的形式。SkillClaw 通过两个阶段完成：先对各个会话进行结构化以保留因果信息，再将其聚合成共享证据基础。


At the system level, SkillClaw connects independently deployed agents through a common skill repository. Each agent has access to the current skill set and produces interaction sessions during normal usage. These sessions are recorded and uploaded as shared evidence. A centralized evolution engine periodically processes the collected sessions, updates the skill repository, and synchronizes the updated skills back to all agents, forming a closed loop:
在系统层面，SkillClaw 通过一个共同的技能仓库连接独立部署的智能体。每个智能体都能访问当前技能集，并在正常使用过程中生成交互会话。这些会话会被记录并上传为共享证据。集中式进化引擎会定期处理收集到的会话，更新技能仓库，并把更新后的技能同步回所有智能体，从而形成闭环：


Multi-user Interaction $\rightarrow$ Session Collection $\rightarrow$ Skill Evolution $\rightarrow$ Skill Synchronization.
多用户交互 $\rightarrow$ 会话收集 $\rightarrow$ 技能进化 $\rightarrow$ 技能同步。


At inference time, the agent receives a catalogue of available skills in its prompt and can dynamically select and load those relevant to the current task. Users do not interact directly, and no coordination among agents is required. Collective improvement arises entirely from shared skill evolution.
在推理时，智能体会在提示中收到可用技能目录，并可根据当前任务动态选择并加载相关技能。用户不会直接互动，也不需要智能体之间进行协调。集体改进完全来自共享的技能进化。


Within this loop, each interaction session contains more than plain dialogue. SkillClaw records the full causal chain: the user prompt, the agent's actions (including tool calls), intermediate feedback (tool results, errors, and explicit user responses), and the final agent response. We record all of this because most skill-level failures are procedural. An incorrect argument format, a missing validation step, or a misordered tool call can cause a task to fail, yet none of these problems appears in the final response. They can only be diagnosed from the intermediate action-feedback trace. Each raw session is converted into a structured representation that preserves this chain:
在这个闭环中，每个交互会话不只是普通对话。SkillClaw 会记录完整的因果链：用户提示、智能体的行动（包括工具调用）、中间反馈（工具结果、错误以及明确的用户回应）以及最终的智能体回复。我们之所以都要记录，是因为多数技能层面的失败是程序性的。参数格式不正确、缺少验证步骤，或工具调用顺序错误都可能导致任务失败，但这些问题不会体现在最终回复中。只能从中间的行动-反馈轨迹中诊断。每个原始会话都会被转换为结构化表示，以保留这条链：


$$
\text{ prompt } \rightarrow  \text{ action } \rightarrow  \text{ feedback } \rightarrow  \cdots  \rightarrow  \text{ agent response. }
$$



Algorithm 1 Agentic Collective Skill Evolution
算法 1 面向智能体的集体技能进化


---



Require: Skill repository $\mathcal{S}$ ,user sessions $\mathcal{T}$
Require: 技能仓库 $\mathcal{S}$，用户会话 $\mathcal{T}$


Ensure: Updated repository ${\mathcal{S}}^{\prime }$
Ensure: 更新后的仓库 ${\mathcal{S}}^{\prime }$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Convert $\mathcal{T}$ into structured evidence $\mathcal{E}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将 $\mathcal{T}$ 转换为结构化证据 $\mathcal{E}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Group $\mathcal{E}$ by referenced skills to obtain $\{ \mathcal{G}\left( s\right) \}$ and $\mathcal{G}\left( \varnothing \right)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;按所引用的技能对 $\mathcal{E}$ 分组，得到 $\{ \mathcal{G}\left( s\right) \}$ 和 $\mathcal{G}\left( \varnothing \right)$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${\mathcal{S}}^{\prime } \leftarrow  \mathcal{S}$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for all group $\mathcal{G}\left( s\right)$ do
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于所有组 $\mathcal{G}\left( s\right)$ do


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Use the agentic evolver to analyze recurring success and failure patterns
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用智能体进化器分析反复出现的成功与失败模式


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Select an evolution action from \{refine, create, skip\}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;从 \{refine, create, skip\} 中选择一次进化动作


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generate a candidate skill update if the evidence supports modification
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;若证据支持修改，则生成候选的技能更新


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Apply conservative editing and validation
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;应用保守的编辑与验证


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Merge approved updates into ${\mathcal{S}}^{\prime }$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将已批准的更新合并到 ${\mathcal{S}}^{\prime }$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;end for
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;结束循环


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Analyze $\mathcal{G}\left( \varnothing \right)$ for missing but reusable procedures
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;分析 $\mathcal{G}\left( \varnothing \right)$ 中缺失但可复用的流程


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add validated new skills into ${\mathcal{S}}^{\prime }$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将经验证的新技能添加到 ${\mathcal{S}}^{\prime }$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Synchronize ${\mathcal{S}}^{\prime }$ back to all agents
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将 ${\mathcal{S}}^{\prime }$ 同步回所有智能体


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return ${\mathcal{S}}^{\prime }$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;返回 ${\mathcal{S}}^{\prime }$


---



We also extract lightweight metadata from each session: (i) which skills were referenced, (ii) whether tool errors occurred, and (iii) a coarse quality estimate. These signals help organize sessions but do not impose rigid labels.
我们还会从每次会话中提取轻量级元数据：（i）引用了哪些技能，（ii）是否发生了工具错误，（iii）一个粗略的质量评估。这些信号有助于组织会话，但不会强制施加固定标签。


Once sessions are structured, they are grouped by the skills they reference to enable cross-user reasoning. For each skill $s$ ,we collect all sessions that invoked $s$ :
一旦会话被结构化，就会按其引用的技能进行分组，以支持跨用户推理。对于每个技能 $s$ ，我们收集所有调用了 $s$ 的会话：


$$
\mathcal{G}\left( s\right)  = \left\{  {{\tau }_{i} \mid  s \in  {\mathcal{K}}_{i}}\right\}
$$



and place sessions that did not use any skill into a separate group $\mathcal{G}\left( \varnothing \right)$ . This grouping does more than organize the data. When multiple sessions invoke the same skill but produce different outcomes across different users, tasks, or environments, the comparison directly reveals where the skill works and where it fails, with the skill itself as the controlled factor. This amounts to a natural ablation and enables two operations that would be unreliable from single-user data alone: (1) evaluating how an existing skill actually performs under diverse real-world usage, and (2) identifying recurring procedures that no existing skill covers, surfaced by patterns in $\mathcal{G}\left( \varnothing \right)$ .
并将未使用任何技能的会话放入一个单独的组 $\mathcal{G}\left( \varnothing \right)$。这样的分组不只是为了整理数据。当多个会话调用同一技能却在不同用户、任务或环境下产生不同结果时，这种对比会直接揭示该技能在哪些地方有效、哪些地方失效，而技能本身作为被控制的因素。这相当于一种自然消融，并使两项仅凭单用户数据难以可靠完成的操作成为可能：（1）评估现有技能在多样的真实使用场景中的实际表现；（2）识别那些由 $\mathcal{G}\left( \varnothing \right)$ 中的模式所揭示的、反复出现但尚未被任何现有技能覆盖的流程。


### 2.2 Agentic Skill Evolution
### 2.2 代理式技能演化


The core of SkillClaw is an agentic evolver that updates the shared skill repository with open-ended reasoning. SkillClaw instantiate an agentic evolver, an LLM agent equipped with a structured harness that supplies the grouped session evidence, the current skill definitions, and a set of permitted evolution actions. The harness provides structured inputs but does not constrain the evolver's reasoning. The evolver diagnoses root causes from sessions of varying context lengths and skills of different formats, and decides how to act. This separation between a fixed harness and open-ended reasoning allows SkillClaw to handle diverse failure modes without hand-crafted rules for each type.
SkillClaw 的核心是一个代理式演化器：它借助开放式推理，更新共享的技能库。SkillClaw 会实例化一个代理式演化器——一个配备结构化工具的 LLM 代理。该工具会提供分组后的会话证据、当前的技能定义，以及一组允许的演化动作。工具提供结构化输入，但不会限制演化器的推理。演化器会从上下文长度不同、技能格式各异的会话中诊断根本原因，并决定如何行动。固定工具与开放式推理之间的这种分离，使 SkillClaw 能够应对多样的失败模式，而无需为每种情况手工编写规则。


Concretely,given a skill $s$ and its associated session group $\mathcal{G}\left( s\right)$ ,the evolver examines both successful and failed executions and selects one of three actions:
具体来说，给定技能 $s$ 及其对应的会话组 $\mathcal{G}\left( s\right)$ ，演化器会同时审视成功与失败的执行，并从三个动作中选择一个：


- Refine. Update the skill to correct identified errors or improve robustness based on observed failure patterns.
- 精炼（Refine）。根据观察到的失败模式，更新该技能以纠正已识别的错误或提升鲁棒性。


- Create. Introduce a new skill when $\mathcal{G}\left( s\right)$ reveals recurring sub-procedures that are not captured by any existing skill.
- 创建（Create）。当 $\mathcal{G}\left( s\right)$ 显示出某些反复出现的子流程，而这些子流程未被任何现有技能所涵盖时，引入一个新技能。


- Skip. Leave the skill unchanged when the available evidence is insufficient to justify a modification.
- 跳过（Skip）。当可用证据不足以证明修改的合理性时，保持该技能不变。


For sessions in $\mathcal{G}\left( \varnothing \right)$ ,i.e.,those that did not invoke any skill,the evolver focuses on discovering missing but reusable procedures. New skills are created only when the observed patterns are specific enough to be teachable and likely to recur.
对于 $\mathcal{G}\left( \varnothing \right)$ 中的会话（即未调用任何技能的会话），演化器重点在于发现缺失但可复用的流程。只有当观察到的模式足够具体，能够被教授且很可能会再次出现时，才会创建新技能。


Regardless of which action is chosen, the evolver always reasons over successful and failed sessions jointly. Successful sessions define the invariants of a skill, the parts that work and must not be altered. Failed sessions define the targets, the specific behaviors that need correction. This joint view is what prevents a naive failure: fixing one problem while inadvertently breaking a previously effective procedure. Each update corrects identified deficiencies while preserving what successful sessions have validated, making evolution cumulative. The complete procedure is given in Algorithm 1.
无论选择哪种动作，演化器都会将成功与失败会话的证据共同纳入推理。成功会话定义技能的不变量——那些运作良好且必须不被改变的部分。失败会话定义目标——需要被纠正的特定行为。正是这种联合视角，避免了天真的失败：修复一个问题的同时无意中破坏先前有效的流程。每次更新都会纠正已识别的不足，同时保留成功会话所验证的内容，使演化能够逐步累积。完整流程见算法 1。


### 2.3 Skill Synchronization and the Evolution Loop
### 2.3 技能同步与演化循环


After evolution, candidate skill updates are validated before being written back to the shared repository. Validation is performed during the nighttime and executed in available idle user environments, ensuring that evaluation reflects real deployment conditions. For a skill $s$ and its candidate update ${s}^{\prime }$ ,the system selects relevant tasks from the interaction data collected during the day. Both versions are executed under the same environment using the full toolchain, including multi-step interactions and intermediate feedback. After execution,the system uses the model to compare the outcomes produced by $s$ and ${s}^{\prime }$ . The decision is based on overall task success and execution stability. If the updated skill demonstrates better performance, it is marked as Accept; otherwise, it is marked as Reject. Accepted updates are merged into the shared repository and synchronized to all agents for the next day. Rejected updates are retained only as candidates and are not deployed. As a result, users always interact with the best validated skill pool from the previous night, rather than unverified updates. This validation step induces a monotonic deployment behavior. Since only improvements are accepted, the deployed skill pool does not degrade over time. Combined with the evolution process, the system forms a closed loop:
演化后，候选技能更新在写回共享仓库之前会先进行验证。验证在夜间进行，并在可用的空闲用户环境中执行，确保评估反映真实部署条件。对于技能$s$及其候选更新${s}^{\prime }$，系统从当天收集的交互数据中选择相关任务。两个版本都在相同环境下使用完整工具链执行，包括多步交互和中间反馈。执行完成后，系统使用模型比较$s$和${s}^{\prime }$产生的结果。决策依据是整体任务成功率和执行稳定性。如果更新后的技能表现更好，则标记为Accept；否则标记为Reject。被接受的更新会合并到共享仓库，并在次日同步给所有代理。被拒绝的更新仅作为候选保留，不会部署。因此，用户始终与前一晚经过最佳验证的技能池交互，而不是未验证的更新。这个验证步骤引入了单调的部署行为。由于只接受改进，已部署的技能池不会随时间退化。结合演化过程，系统形成一个闭环：


$$
\text{ Interaction } \rightarrow  \text{ Evidence } \rightarrow  \text{ Evolution } \rightarrow  \text{ Validation } \rightarrow  \text{ Deployment. }
$$



where updated skills shape future interactions and generate new evidence for the next round of evolution.
其中，更新后的技能塑造未来的交互，并为下一轮演化生成新的证据。


Three properties follow from this design. First, collective evolution. Sessions are aggregated across users, and knowledge discovered in one interaction is propagated to a shared skill ecosystem that benefits all users. Second, full automation. The entire pipeline, from session recording to skill synchronization, runs without manual curation or explicit user intervention. The only human input is normal agent usage. Third, agentic adaptability. Skill updates are produced through open-ended reasoning rather than predefined rules, enabling the system to handle previously unseen failure modes and usage patterns.
这一设计带来三个属性。第一，集体演化。会话按用户聚合，在一次交互中发现的知识会传播到共享技能生态中，惠及所有用户。第二，全自动化。整个流程，从会话记录到技能同步，都无需人工筛选或明确的用户干预。唯一的人类输入就是正常使用代理。第三，具备代理式适应能力。技能更新通过开放式推理而非预定义规则生成，使系统能够处理此前未见的失败模式和使用模式。


From the user's perspective, none of this is visible. Users interact with their agents as usual, while skill evolution happens in the background. Over time, isolated user experiences are consolidated into a shared skill set that improves with continued use.
从用户的角度来看，这一切都不可见。用户像往常一样与自己的代理交互，而技能演化在后台发生。随着时间推移，彼此孤立的用户体验会被整合为共享技能集，并在持续使用中不断改进。


## 3 Experiments
## 3 实验


### 3.1 Benchmark: WildClawBench
### 3.1 基准：WildClawBench


We evaluate SkillClaw on WildClawBench (Ding et al. (2026)), a real-world agent benchmark consisting of 60 complex tasks across six capability domains. As summarized in Table 1, the benchmark covers diverse scenarios including productivity workflows, code execution, social interaction, retrieval, creative generation, and safety alignment. Unlike prior benchmarks, WildClawBench requires full end-to-end execution in realistic environments with multimodal tool usage. Table 2 highlights its key properties, including fine-grained evaluation metrics and hard constraints that enforce strict correctness.
我们在 WildClawBench（Ding 等，（2026））上评估 SkillClaw。该基准是一个面向真实世界的智能体基准，包含六个能力领域的60个复杂任务。如表1所述，该基准覆盖多样场景，包括生产力工作流、代码执行、社交互动、检索、创意生成以及安全对齐。与以往基准不同，WildClawBench 要求在真实环境中完成端到端执行，并进行多模态工具使用。表2展示了其关键特性，包括细粒度的评估指标以及强制严格正确性的硬约束。


Table 1 Task categories in WildClawBench. The benchmark spans six domains covering a wide spectrum of real-world agent scenarios, from procedural workflows to multimodal generation and safety-critical decision making.
表1 WildClawBench 的任务类别。该基准跨越六个领域，覆盖广泛的真实世界智能体场景，从流程化工作流到多模态生成，以及安全关键的决策制定。


<table><tr><td>Category</td><td>Example Tasks</td><td>Challenges</td></tr><tr><td>Productivity Flow <br> Code Intelligence <br> Social Interaction <br> Search & Retrieval <br> Creative Synthesis <br> Safety & Alignment</td><td>arXiv classification, scheduling, SCP <br> debugging, puzzle solving <br> negotiation, chat analysis <br> academic search, conflict resolution <br> video notes, poster generation <br> prompt injection, leakage detection</td><td>multi-step pipelines execution correctness multi-turn reasoning <br> API usage <br> multimodal generation <br> constraint satisfaction</td></tr></table>
<table><tbody><tr><td>类别</td><td>示例任务</td><td>挑战</td></tr><tr><td>生产力流程 <br/> 代码智能 <br/> 社交互动 <br/> 搜索与检索 <br/> 创意综合 <br/> 安全与对齐</td><td>arXiv 分类、日程安排、SCP <br/> 调试、解谜 <br/> 谈判、聊天分析 <br/> 学术搜索、冲突解决 <br/> 视频笔记、海报生成 <br/> 提示注入、泄露检测</td><td>多步流水线执行正确性 多轮推理 <br/> API 使用 <br/> 多模态生成 <br/> 约束满足</td></tr></tbody></table>


Table 2 Key properties of WildClawBench, highlighting its realistic execution environment, multimodal inputs, and long-horizon, failure-sensitive evaluation setting.
表2 WildClawBench的关键属性，突出其真实的执行环境、多模态输入以及长时程、对失败敏感的评估设定。


<table><tr><td>Property</td><td>Description</td></tr><tr><td>Execution Environment</td><td>Full Linux container with tools</td></tr><tr><td>Multimodality</td><td>Text, code, image, video</td></tr><tr><td>Evaluation</td><td>3-27 metrics aggregated</td></tr><tr><td>Hard Constraints</td><td>Critical errors $\rightarrow$ zero score</td></tr><tr><td>Task Length</td><td>15-50 steps</td></tr><tr><td>External Dependency</td><td>APIs and model downloads</td></tr></table>
<table><tbody><tr><td>属性</td><td>描述</td></tr><tr><td>执行环境</td><td>完整 Linux 容器，配备工具</td></tr><tr><td>多模态</td><td>文本、代码、图像、视频</td></tr><tr><td>评估</td><td>3-27 项指标汇总</td></tr><tr><td>严格约束</td><td>关键错误 $\rightarrow$ 零分</td></tr><tr><td>任务长度</td><td>15-50 步</td></tr><tr><td>外部依赖</td><td>API 和模型下载</td></tr></tbody></table>


### 3.2 Experimental Setup
### 3.2 实验设置


We simulate a realistic deployment scenario using a continuous day-night skill evolution process. The experiment runs for 6 days (6 rounds), where each day consists of two phases: a daytime online interaction phase and a nighttime skill evolution and validation phase. During the daytime, users interact with deployed OpenClaw agents to complete tasks in WildClawBench. These interactions generate session trajectories that capture failure modes, edge cases, and recurring bottlenecks encountered during execution. During the nighttime, the system processes the collected interaction data to generate candidate skill updates targeting these observed deficiencies. A validator then filters candidate updates, and only approved skills are added to the shared deployment pool for the next day. This process forms a closed loop: users operate with the current best skill pool during the day, while the system absorbs feedback and produces updated skills at night, which are then redeployed for subsequent interactions. Our setup involves 8 concurrent users, each interacting with the system under WildClawBench tasks based on their individual goals and task requirements. All execution, skill evolution, and validation processes are powered by Qwen3-Max. At the system level, we maintain a shared current best skill pool. Day 1 starts with an initial skill set corresponding to the baseline. In subsequent rounds, only skills that are triggered during interaction and exhibit potential for improvement are considered for candidate updates. Results are reported on four representative categories, with additional categories to be included in the future version.
我们通过连续的昼夜技能演化过程模拟一个真实的部署场景。实验持续 6 天（6 轮），每天包含两个阶段：白天的在线交互阶段和夜间的技能演化与验证阶段。白天，用户与已部署的 OpenClaw 代理交互，以完成 WildClawBench 中的任务。这些交互会生成会话轨迹，记录执行过程中遇到的失败模式、边界情况和反复出现的瓶颈。夜间，系统处理收集到的交互数据，生成针对这些已观察到缺陷的候选技能更新。随后，验证器筛选候选更新，只有通过批准的技能才会被加入共享的部署池，并在第二天使用。该过程形成一个闭环：白天用户使用当前最佳技能池，系统在夜间吸收反馈并生成更新技能，随后再部署到后续交互中。我们的设置包含 8 个并发用户，每个用户都依据各自的目标和任务要求，在 WildClawBench 任务下与系统交互。所有执行、技能演化和验证过程均由 Qwen3-Max 驱动。在系统层面，我们维护一个共享的当前最佳技能池。第 1 天从与基线对应的初始技能集开始。在后续轮次中，仅将交互中被触发且具有改进潜力的技能纳入候选更新。结果在四个代表性类别上报告，更多类别将包含在未来版本中。


Validation Mechanism. The validation mechanism is a critical component of our experimental design. During the nighttime phase, the system first identifies candidate skill updates based on interaction logs accumulated during the day. These candidate updates are then deployed to available user environments and evaluated under real execution conditions. The validator follows a simple decision rule. If a candidate skill outperforms the currently deployed best skill on the corresponding validation tasks, it is marked as Accept; otherwise, it is marked as Reject. Accepted skills are merged into the current best skill pool and deployed to all users on the following day. Rejected skills are retained only as candidate records and are not deployed. As a result, users always interact with the best validated skill pool from the previous night, rather than unverified updates. This validation strategy introduces additional token cost, as candidate skills must be executed in real environments with full tool interaction. However, compared to direct deployment without validation, this overhead leads to significantly more stable user-facing performance.
验证机制。验证机制是我们实验设计中的关键组成部分。在夜间阶段，系统首先根据白天积累的交互日志识别候选技能更新。随后，这些候选更新会被部署到可用的用户环境中，并在真实执行条件下进行评估。验证器采用一个简单的决策规则：如果某个候选技能在相应的验证任务上优于当前部署的最佳技能，则标记为 Accept；否则，标记为 Reject。被接受的技能会合并到当前最佳技能池中，并在第二天部署给所有用户。被拒绝的技能仅作为候选记录保留，不会部署。因此，用户始终与前一夜经过验证的最佳技能池交互，而不是未验证的更新。该验证策略会带来额外的 token 成本，因为候选技能必须在具备完整工具交互的真实环境中执行。然而，与不经验证直接部署相比，这一开销显著提升了面向用户的性能稳定性。


Table 3 User-side daytime results (best-skill deployment view). Day 1 is the baseline experience; Day 2-6 reflect the best skill pool carried forward after each nightly validator decision. Absolute and relative gains are computed w.r.t. Day 1.
表 3 用户侧白天结果（最佳技能部署视图）。第 1 天为基线体验；第 2-6 天反映的是每晚验证器决策后沿用的最佳技能池。绝对和相对增益均相对于第 1 天计算。


<table><tr><td>Category</td><td>Day 1</td><td>Day 2</td><td>Day 3</td><td>Day 4</td><td>Day 5</td><td>Day 6</td><td>Abs. Gain</td><td>Rel. Gain</td></tr><tr><td>Social Interaction</td><td>54.01%</td><td>60.34%</td><td>60.34%</td><td>60.34%</td><td>60.34%</td><td>60.34%</td><td>+6.33</td><td>+11.72%</td></tr><tr><td>Search & Retrieval</td><td>22.73%</td><td>30.00%</td><td>30.00%</td><td>34.55%</td><td>34.55%</td><td>34.55%</td><td>+11.82</td><td>+52.00%</td></tr><tr><td>Creative Synthesis</td><td>11.57%</td><td>21.80%</td><td>21.80%</td><td>21.80%</td><td>21.80%</td><td>21.80%</td><td>+10.23</td><td>+88.41%</td></tr><tr><td>Safety & Alignment</td><td>24.00%</td><td>24.00%</td><td>24.00%</td><td>24.00%</td><td>32.00%</td><td>32.00%</td><td>+8.00</td><td>+33.33%</td></tr></table>
<table><tbody><tr><td>类别</td><td>第 1 天</td><td>第 2 天</td><td>第 3 天</td><td>第 4 天</td><td>第 5 天</td><td>第 6 天</td><td>绝对收益</td><td>相对收益</td></tr><tr><td>社交互动</td><td>54.01%</td><td>60.34%</td><td>60.34%</td><td>60.34%</td><td>60.34%</td><td>60.34%</td><td>+6.33</td><td>+11.72%</td></tr><tr><td>搜索与检索</td><td>22.73%</td><td>30.00%</td><td>30.00%</td><td>34.55%</td><td>34.55%</td><td>34.55%</td><td>+11.82</td><td>+52.00%</td></tr><tr><td>创意融合</td><td>11.57%</td><td>21.80%</td><td>21.80%</td><td>21.80%</td><td>21.80%</td><td>21.80%</td><td>+10.23</td><td>+88.41%</td></tr><tr><td>安全与对齐</td><td>24.00%</td><td>24.00%</td><td>24.00%</td><td>24.00%</td><td>32.00%</td><td>32.00%</td><td>+8.00</td><td>+33.33%</td></tr></tbody></table>


### 3.3 Main Results
### 3.3 主要结果


As shown in Table 3, all four categories exhibit a consistent evolution pattern over 6 days. The system first resolves primary bottlenecks, then stabilizes deployment around the current best skill pool. The trajectory is not characterized by daily fluctuations, but by progressively consolidating locally effective updates into a stable skill set deployed to users.
如表3所示，四个类别在6天内都呈现出一致的演化模式。系统首先解决主要瓶颈，随后围绕当前最佳技能池稳定部署。其轨迹并非表现为每日波动，而是将局部有效的更新逐步整合为一个稳定的技能集并部署给用户。


Social Interaction improves earliest and most sharply. Performance increases from 54.01% to 60.34% on Day 2 and remains stable thereafter. This indicates the presence of a high-impact workflow bottleneck with broad coverage. Once the corresponding skill is improved, the system quickly gains capability in cross-source integration, task organization, and high-level summarization. Although additional skill updates are proposed in later rounds, Day 2 already establishes the current best skill pool for this category, leading to consistently strong user-side performance.
社交互动最早且最显著地提升。性能从54.01%在第2天提高到60.34%，此后保持稳定。这表明存在一个覆盖面广、影响巨大的工作流瓶颈。一旦相应技能得到改进，系统便迅速获得跨源整合、任务组织和高层摘要能力。尽管后续轮次还提出了额外的技能更新，但第2天已为该类别建立了当前最佳技能池，从而带来持续稳定的用户侧表现。


Search & Retrieval follows a more staged improvement trajectory, increasing from 22.73% to 30.00%, and then further to 34.55%. Unlike Social Interaction, the gains are not driven by a single skill update but by a sequence of improvements. The system first resolves input validation and file accessibility, then builds toward constraint-aware retrieval planning. This reflects a key property of retrieval tasks, where higher-level reasoning becomes effective only after lower-level reliability is ensured.
搜索与检索呈现出更分阶段的提升轨迹，从22.73%提升到30.00%，随后进一步升至34.55%。与社交互动不同，这些收益并非由单一技能更新驱动，而是由一系列改进共同促成。系统首先解决输入校验和文件可访问性问题，然后逐步构建出具备约束感知的检索规划能力。这反映了检索任务的一个关键特性：只有在底层可靠性得到保证后，更高层的推理才会真正生效。


Creative Synthesis shows a large early jump from 11.57% to 21.80% on Day 2 and then plateaus. This suggests that the primary bottleneck lies not in content generation itself, but in environment setup, including file handling, working directory configuration, and multimodal pipelines. Once these foundational issues are resolved, user-facing performance improves rapidly. More complex multimodal skills continue to emerge and pass validation, but within the 6-day window, they do not surpass the early-established best skill pool.
创意综合在第2天从11.57%大幅跃升至21.80%，随后进入平台期。这说明主要瓶颈不在内容生成本身，而在环境配置上，包括文件处理、工作目录配置和多模态管线。一旦这些基础问题得到解决，面向用户的性能便迅速提升。更复杂的多模态技能仍在持续出现并通过验证，但在6天窗口内，它们并未超过早期建立的最佳技能池。


Safety & Alignment improves later, from 24.00% to 32.00%. Improvements in this category primarily target execution reliability in real-world environments rather than surface-level task performance. Effective updates focus on mechanisms such as Git fallback, directory cloning protocols, and safe execution in non-interactive settings. These changes may not immediately yield higher scores but, once validated, are retained in the deployment pool and contribute to long-term system robustness.
安全与对齐的提升较晚，从24.00%升至32.00%。该类别的改进主要针对真实环境中的执行可靠性，而非表层任务性能。有效更新集中在Git回退、目录克隆协议以及非交互式设置下的安全执行等机制上。这些变化未必会立即带来更高分数，但一旦通过验证，便会保留在部署池中，并有助于系统的长期稳健性。


From a deployment perspective, Table 3 reflects not a sequence of independent experiments, but a continuously running system that consolidates nightly verified updates into a unified skill pool for daytime usage. It is important to note that this study represents a small-scale test of collective skill evolution, with limited user queries, feedback signals, and interaction depth. Despite these constraints, SkillClaw still achieves consistent performance gains, demonstrating its effectiveness in realistic interaction settings. Scaling up the number of users, extending the time horizon, and introducing more diverse tasks and validation conditions are likely to further enrich the evolution trajectory and further improve system performance.
从部署角度看，表3反映的并不是一系列独立实验，而是一个持续运行的系统，它将每晚经验证的更新整合为白天使用的统一技能池。需要指出的是，本研究只是对集体技能演化的一次小规模测试，用户查询、反馈信号和交互深度都有限。尽管存在这些约束，SkillClaw仍实现了持续的性能提升，展示了其在真实交互场景中的有效性。扩大用户数量、延长时间跨度，并引入更多样的任务和验证条件，可能会进一步丰富演化轨迹，并进一步提升系统性能。


Table 4 Social Interaction: nightly skill evolution and validator decisions. The only skill update that entered the deployed best pool was 03_task6 (accepted after Night 1).
表4 社交互动：夜间技能演化与验证器决策。唯一进入已部署最佳池的技能更新是03_task6（在第1晚后被接受）。


<table><tr><td>Day</td><td>Candidate Skill</td><td>Skill Function</td><td>Change Summary</td><td>Validator</td><td>Next-Day Action</td></tr><tr><td>1</td><td>03_task6</td><td>Cross-dept Slack summarization, data reconciliation, risk identification, board-level brief drafting</td><td>Rewrote workflow into strictly-ordered steps; strengthened project keyword filtering, finance priority, change detection, COO contact confirmation</td><td>Accept</td><td>Day 2: upgrade to new best pool</td></tr><tr><td>2</td><td>(none)</td><td>Continued using current Social best pool</td><td>Same-pool retest; no new skill text landed</td><td>Reject</td><td>Day 3: keep Day 2 best pool</td></tr><tr><td>3</td><td>03_task1</td><td>Gmail + Calendar meeting coordination</td><td>Extended workflow with meeting-param extraction, multi-participant availability check, confirmation loop, reschedule on rejection</td><td>Reject</td><td>Not admitted; Day 4 keeps current best pool</td></tr><tr><td>4</td><td>(none)</td><td>Continued using current Social best pool</td><td>Same-pool retest; no new skill text landed</td><td>Reject</td><td>Day 5: keep current best pool</td></tr><tr><td>5</td><td>(none)</td><td>Continued using current Social best pool</td><td>Same-pool retest; no new skill text landed</td><td>Reject</td><td>Day 6: keep current best pool</td></tr><tr><td>6</td><td>03_task3</td><td>Slack feasibility analysis</td><td>Added fallback & grounding constraints; analysis must rely on real API results or user-provided context</td><td>Reject</td><td>Not admitted to next cycle</td></tr></table>
<table><tbody><tr><td>天</td><td>候选技能</td><td>技能功能</td><td>变更摘要</td><td>验证器</td><td>次日操作</td></tr><tr><td>1</td><td>03_task6</td><td>跨部门Slack总结、数据核对、风险识别、起草面向董事会的简报</td><td>将工作流重写为严格有序的步骤；强化项目关键词筛选、财务优先级、变更检测、以及对COO联系确认</td><td>接受</td><td>第2天：升级到新的最佳技能池</td></tr><tr><td>2</td><td>（无）</td><td>继续使用当前Social最佳技能池</td><td>同池复测；未投放新技能文本</td><td>拒绝</td><td>第3天：保留第2天的最佳技能池</td></tr><tr><td>3</td><td>03_task1</td><td>Gmail + 日历会议协调</td><td>扩展工作流：提取会议信息参数、多参与者可用性校验、确认闭环；若被拒绝则重新安排</td><td>拒绝</td><td>未通过；第4天保留当前最佳技能池</td></tr><tr><td>4</td><td>（无）</td><td>继续使用当前Social最佳技能池</td><td>同池复测；未投放新技能文本</td><td>拒绝</td><td>第5天：保留当前最佳技能池</td></tr><tr><td>5</td><td>（无）</td><td>继续使用当前Social最佳技能池</td><td>同池复测；未投放新技能文本</td><td>拒绝</td><td>第6天：保留当前最佳技能池</td></tr><tr><td>6</td><td>03_task3</td><td>Slack可行性分析</td><td>加入后备与约束条件；分析必须基于真实API结果或用户提供的上下文</td><td>拒绝</td><td>未被纳入下一轮</td></tr></tbody></table>


### 3.4 Analysis
### 3.4 分析


As shown in Table 4-Table 7, skill evolution is highly heterogeneous across categories, following distinct capability trajectories rather than a uniform pattern.
如表4至表7所示，不同类别的技能演化高度异质，呈现出各自不同的能力轨迹，而非统一模式。


In Social Interaction, evolution primarily improves workflow explicitness and execution reliability. The category already starts with relatively complete task-oriented skills, including meeting coordination, Slack task extraction, feasibility analysis, status reporting, support triage, and executive summarization. The limitation is therefore not missing capabilities, but insufficient executability. The most impactful update comes from executive-level summarization, which spans message retrieval, information filtering, data verification, risk extraction, and structured output. Once this skill is rewritten from a descriptive instruction into an explicit procedural workflow, performance improves sharply. Subsequent updates to meeting coordination and feasibility analysis mainly refine and strengthen this existing structure.
在社交互动中，演化主要提升了工作流的显式性和执行可靠性。该类别一开始就已具备相对完整的任务导向技能，包括会议协调、Slack任务提取、可行性分析、状态汇报、支持分诊和高层总结。因此，其限制并不在于能力缺失，而在于可执行性不足。最具影响力的更新来自高层总结，它涵盖消息检索、信息筛选、数据核验、风险提取和结构化输出。一旦该技能从描述性指令改写为明确的流程化工作流，性能便显著提升。随后对会议协调和可行性分析的更新，主要是在细化和强化这一既有结构。


Search & Retrieval exhibits a staged evolution pattern. Early updates focus on file existence checks, path resolution, and multimodal input validation, indicating that initial failures stem from unreliable input handling rather than high-level reasoning. As these issues are resolved, evolution shifts toward higher-level capabilities such as constraint-aware retrieval planning and missing input recovery. This input-first, strategy-later progression aligns with real-world retrieval systems and explains why improvements emerge incrementally through multiple skill updates rather than a single change.
搜索与检索呈现出阶段式演化模式。早期更新集中于文件存在性检查、路径解析和多模态输入校验，表明初始失败源于不可靠的输入处理，而非高层推理。随着这些问题得到解决，演化转向更高层能力，如考虑约束的检索规划和缺失输入恢复。这种先输入、后策略的推进路径与真实世界的检索系统一致，也解释了为何改进是通过多次技能更新逐步出现，而非一次性变化。


In Creative Synthesis, evolution centers on organizing multimodal processing pipelines. Early gains come from establishing reliable execution environments, including working directory validation, input checking, and media preprocessing. This suggests that the primary bottleneck lies in entering a correct execution flow rather than generating creative content. Later updates extend toward higher-level multimodal pipelines, such as PDF-to-poster generation, video summarization, and image-based synthesis. These updates indicate a transition from getting tasks to run to running tasks professionally. However, the early-established best skill pool already provides strong performance, and later improvements do not yet surpass this level within the 6-day window.
在创意综合中，演化的核心是组织多模态处理流水线。早期收益来自建立可靠的执行环境，包括工作目录校验、输入检查和媒体预处理。这说明主要瓶颈在于进入正确的执行流程，而非生成创意内容。后续更新进一步扩展到更高层的多模态流水线，如PDF转海报生成、视频总结和基于图像的综合。这些更新表明，系统正从让任务跑起来过渡到专业地跑任务。不过，早期建立的最佳技能池已提供了强劲性能，而在6天窗口内，后续改进尚未超越这一水平。


Safety & Alignment follows a reliability-driven evolution path. Updates in this category focus on robust execution under real-world constraints rather than expanding task capabilities. Typical improvements include fallback strategies for Git authentication failures and correct directory cloning procedures. These skills do not primarily increase apparent intelligence but reduce failure rates under edge conditions. Once validated, they are retained in the deployment pool and form the foundation of system stability.
安全与对齐遵循一条由可靠性驱动的演化路径。该类别的更新聚焦于在真实世界约束下的稳健执行，而非扩展任务能力。典型改进包括针对Git认证失败的回退策略，以及正确的目录克隆流程。这些技能并不主要提升表面上的智能，而是降低边界条件下的失败率。一旦验证通过，它们便会被保留在部署池中，并构成系统稳定性的基础。


Overall, Table 4-Table 7 show that skill evolution is not a simple accumulation of rules, but a structured process driven by category-specific bottlenecks. Social Interaction emphasizes workflow executability, Search & Retrieval emphasizes input reliability and planning, Creative Synthesis emphasizes multimodal pipeline organization, and Safety & Alignment emphasizes robust and recoverable execution in real-world environments.
总体而言，表4至表7表明，技能演化并非规则的简单累积，而是由类别特定瓶颈驱动的结构化过程。社交互动强调工作流可执行性，搜索与检索强调输入可靠性与规划，创意综合强调多模态流水线组织，而安全与对齐强调在真实环境中的稳健且可恢复的执行。


Table 5 Search & Retrieval: nightly skill evolution and validator decisions. Key accepted updates: validate-file-existence (Night 1) and best-so-far confirmation (Night 3).
表5 搜索与检索：夜间技能演化与验证器决策。关键获批更新：validate-file-existence（第1夜）和best-so-far确认（第3夜）。


<table><tr><td>Day</td><td>Candidate Skill</td><td>Skill Function</td><td>Change Summary</td><td>Validator</td><td>Next-Day Action</td></tr><tr><td>1</td><td>validate-file-existence</td><td>Pre-processing file existence check</td><td>Before any file parsing / image reading / multimodal call, first confirm the input file actually exists</td><td>Accept</td><td>Day 2: upgrade to new best pool</td></tr><tr><td>2</td><td>debug-missing-file-path</td><td>Missing-file path debugging</td><td>List parent directory, verify naming, correct path instead of halting on "missing"</td><td>Reject</td><td>Day 3: keep Day 2 best pool</td></tr><tr><td>3</td><td>(none)</td><td>Continued using current Search best pool</td><td>Same-pool retest; nightly readout was stronger, confirming current pool as best-so-far</td><td>Accept</td><td>Day 4: continue same best pool</td></tr><tr><td>4</td><td>robust-file-validati before-multimodal</td><td>Stronger multimodal pre-validation</td><td>Upgraded from "exists?" to "exists + parent-dir search + hard pre-multimodal validation"</td><td>Reject</td><td>Day 5: keep current best pool</td></tr><tr><td>5</td><td>constrained-technica search-planning</td><td>Budget-constrained technical / academic search planning</td><td>Added feasibility check, sub-question decomposition, official-source priority. evidence-chain output</td><td>Reject</td><td>Day 6: keep current best pool</td></tr><tr><td>6</td><td>recover-missing-input-file</td><td>Recover / locate real input file from workspace</td><td>When benchmark's expected path fails, proactively search the working directory for the actual input file</td><td>Reject</td><td>Not admitted to next cycle</td></tr></table>
<table><tbody><tr><td>天</td><td>候选技能</td><td>技能功能</td><td>变更摘要</td><td>验证器</td><td>次日行动</td></tr><tr><td>1</td><td>验证文件是否存在</td><td>预处理的文件存在性检查</td><td>在进行任何文件解析/图像读取/多模态调用之前，先确认输入文件确实存在</td><td>通过</td><td>第 2 天：升级到新的最佳池</td></tr><tr><td>2</td><td>调试缺失的文件路径</td><td>缺失文件路径调试</td><td>列出父目录，核对命名；不要因“缺失”就停止，改为修正路径</td><td>拒绝</td><td>第 3 天：保留第 2 天的最佳池</td></tr><tr><td>3</td><td>（无）</td><td>继续使用当前的搜索最佳池</td><td>同池复测；夜间读出更强，确认当前池为迄今最佳</td><td>通过</td><td>第 4 天：继续使用同一个最佳池</td></tr><tr><td>4</td><td>多模态前的稳健文件验证</td><td>更强的多模态前置验证</td><td>从“存在吗？”升级为“存在 + 父目录搜索 + 多模态前的严格验证”</td><td>拒绝</td><td>第 5 天：保留当前最佳池</td></tr><tr><td>5</td><td>受约束的技术搜索规划</td><td>在预算限制下的技术/学术搜索规划</td><td>加入可行性检查、子问题分解、官方来源优先；输出证据链</td><td>拒绝</td><td>第 6 天：保留当前最佳池</td></tr><tr><td>6</td><td>找回缺失的输入文件</td><td>从工作区恢复/定位真实的输入文件</td><td>当基准的期望路径失败时，主动在工作目录中搜索实际的输入文件</td><td>拒绝</td><td>未被纳入下一轮</td></tr></tbody></table>


Controlled validation of skill evolution. Table 8 provides a controlled validation of the evolution mechanism using three custom queries: basic extraction, deadline parsing, and save report. Unlike the full benchmark, these queries are designed to isolate common failure modes observed in the main results, allowing us to examine whether skill evolution can directly resolve them. We observe a consistent improvement after a single round of evolution,with an average gain of +42.1%. In particular, save report improves from 28.3% to 100.0%, where the initial failure is caused by missing environment-specific procedures (e.g., output path or format), which can be fully corrected once encoded as a reusable skill. Similarly, basic extraction shows a large gain (+47.8%), indicating that recurring execution patterns can be effectively captured through evolution. In contrast, deadline parsing exhibits a smaller improvement (+6.9%), suggesting that tasks relying more on nuanced reasoning are less sensitive to procedural skill updates. Overall, these controlled results complement the main benchmark findings by showing that skill evolution is particularly effective when failures arise from missing or incorrect procedural knowledge, providing a direct mechanism-level explanation for the gains observed in earlier experiments.
技能演化的受控验证。表8通过三个自定义查询——基础抽取、截止时间解析和保存报告——对演化机制进行了受控验证。与完整基准不同，这些查询旨在隔离主结果中观察到的常见失效模式，从而检验技能演化是否能直接加以解决。我们观察到在单轮演化后呈现一致的提升，平均增益为+42.1%。其中，保存报告从28.3%提升到100.0%；初始失效由缺少特定于环境的流程导致（例如输出路径或格式），一旦将其编码为可复用技能，便可完全纠正。类似地，基础抽取也体现出较大的增益（+47.8%），表明可通过演化有效捕捉反复出现的执行模式。相对而言，截止时间解析的提升较小（+6.9%），说明那些更依赖细致推理的任务，对程序性技能更新不那么敏感。总体而言，这些受控结果补充了主要基准的发现：当失效源自缺失或不正确的程序性知识时，技能演化尤其有效，并为此前实验中观察到的收益提供了直接的机制层面解释。


### 3.5 Case Study
### 3.5 案例研究


Figure 2 illustrates how skill evolution improves task execution on a Slack message analysis task. The original agent follows a naive workflow that retrieves all messages and processes them uniformly, while also relying on trial-and-error to handle tool failures (e.g., incorrect API port configuration). As a result, execution is both inefficient and error-prone. In contrast, the evolved skill introduces a structured and reliable workflow. It first scans message previews to identify task-relevant candidates, then selectively retrieves full message content when necessary, and finally extracts actionable items. At the same time, previously observed tool failures are corrected by encoding the proper API configuration directly into the skill. This transformation reflects three key improvements: (1) task decomposition, where the problem is divided into filtering and extraction stages; (2) error correction, where tool-level failures are resolved proactively rather than through reactive retries; and (3) selective retrieval, which focuses computation on relevant messages and improves extraction quality. Overall, this example demonstrates that skill evolution not only fixes execution errors but also restructures the interaction pipeline into a more efficient and reliable strategy.
图2展示了技能演化如何提升 Slack 消息分析任务的执行效果。原始智能体遵循一种朴素流程：检索所有消息并统一处理，同时依赖试错来应对工具失败（例如 API 端口配置错误）。因此，执行既低效又容易出错。相比之下，演化后的技能引入了结构化且可靠的工作流。它首先扫描消息预览以识别与任务相关的候选项，然后在必要时选择性地检索完整消息内容，最后提取可执行事项。与此同时，先前观察到的工具失败通过将正确的 API 配置直接编码进技能中而得到修正。这一转变体现了三个关键改进：（1）任务分解，即将问题拆分为过滤和提取两个阶段；（2）错误修正，即主动解决工具层面的失败，而不是通过被动重试；以及（3）选择性检索，即将计算聚焦于相关消息并提升提取质量。总体而言，这一示例表明，技能演化不仅修复了执行错误，还将交互管线重构为更高效、更可靠的策略。


Table 6 Creative Synthesis: nightly skill evolution and validator decisions. The only accepted skill w validate-tmp-workspace-inputs (Night 1).
表6 创造性综合：夜间技能演化和验证器决策。唯一被接受的技能是 validate-tmp-workspace-inputs（第1晚）。


<table><tr><td>Day</td><td>Candidate Skill</td><td>Skill Function</td><td>Change Summary</td><td>Validator</td><td>Next-Day Action</td></tr><tr><td>1</td><td>validate-tmp-workspace-inputs</td><td>Check /tmp_workspace inputs & environment setup</td><td>Before creative tasks, verify /tmp_workspace inputs, directories, and symlinks are correct</td><td>Accept</td><td>Day 2: upgrade to new best pool</td></tr><tr><td>2</td><td>multimodal-input-validation-and-setup</td><td>Multimodal input validation & output env init</td><td>Check video / image / PDF / audio files exist, are readable, and format-correct; prepare output directories</td><td>Reject</td><td>Day 3: keep current best pool</td></tr><tr><td>3</td><td>multimodal-creative-task-pipeline</td><td>Multimodal creative pipeline</td><td>New unified pipeline: extract content from PDF / video / image and generate posters, webpages, slides, etc.</td><td>Reject</td><td>Day 4: keep current best pool</td></tr><tr><td>4</td><td>multimodal-creative-task-pipeline (impr.)</td><td>Multimodal creative pipeline</td><td>Added image classification, visual generation, garment synthesis, structured output validation</td><td>Reject</td><td>Day 5: keep current best pool</td></tr><tr><td>5</td><td>multimodal-creative-task-pipeline (impr.); validate-required-input-files</td><td>Creative pipeline + per-file fail-fast validation</td><td>Pipeline added audio/video fallback & halt on missing input; new skill forces per-file validation for all named inputs</td><td>Reject</td><td>Day 6: keep current best pool</td></tr><tr><td>6</td><td>multimodal-creative-task-pipeline (cand.)</td><td>Multimodal creative pipeline</td><td>Extended PDF-to-poster / document-to-visual paths; did not yield better deployment results</td><td>Reject</td><td>Not admitted to next cycle</td></tr></table>
<table><tbody><tr><td>天</td><td>候选技能</td><td>技能功能</td><td>变更摘要</td><td>验证器</td><td>次日行动</td></tr><tr><td>1</td><td>validate-tmp-workspace-inputs</td><td>检查 /tmp_workspace 输入和环境设置</td><td>在创意任务前，核实 /tmp_workspace 输入、目录和符号链接是否正确</td><td>接受</td><td>第2天：升级为新的最佳池</td></tr><tr><td>2</td><td>multimodal-input-validation-and-setup</td><td>多模态输入验证与输出环境初始化</td><td>检查视频/图片/PDF/音频文件是否存在、可读且格式正确；准备输出目录</td><td>拒绝</td><td>第3天：保持当前最佳池</td></tr><tr><td>3</td><td>multimodal-creative-task-pipeline</td><td>多模态创意流水线</td><td>新统一流水线：从 PDF/视频/图片中提取内容，并生成海报、网页、幻灯片等</td><td>拒绝</td><td>第4天：保持当前最佳池</td></tr><tr><td>4</td><td>multimodal-creative-task-pipeline (impr.)</td><td>多模态创意流水线</td><td>新增图像分类、视觉生成、服装合成、结构化输出验证</td><td>拒绝</td><td>第5天：保持当前最佳池</td></tr><tr><td>5</td><td>multimodal-creative-task-pipeline (impr.); validate-required-input-files</td><td>创意流水线 + 按文件快速失败验证</td><td>流水线新增音频/视频回退与缺失输入即停止；新技能强制对所有命名输入逐文件验证</td><td>拒绝</td><td>第6天：保持当前最佳池</td></tr><tr><td>6</td><td>multimodal-creative-task-pipeline (cand.)</td><td>多模态创意流水线</td><td>扩展了 PDF 转海报 / 文档转视觉路径；未带来更好的部署结果</td><td>拒绝</td><td>未被纳入下一轮</td></tr></tbody></table>


Table 7 Safety & Alignment: nightly skill evolution and validator decisions. Skills were accepted on Nights 1-4; candidate improvements on Nights 5-6 were rejected.
表 7 安全与对齐：夜间技能演化与验证器的决策。第 1-4 夜接受了技能；第 5-6 夜的候选改进被拒绝。


<table><tr><td>Day</td><td>Candidate Skill</td><td>Skill Function</td><td>Change Summary</td><td>Validator</td><td>Next-Day Action</td></tr><tr><td>1</td><td>git-push-with-auth-fallback</td><td>Patch / bundle fallback on git push failure</td><td>In no-credential / auth-failure scenarios, provide safe fallback instead of blocking on push</td><td>Accept</td><td>Day 2: add to Safety best pool</td></tr><tr><td>2</td><td>git-push-with-auth-fallback</td><td>Git auth-failure fallback</td><td>Unified patch / bundle filenames and verification; reduced filename inconsistency during fallback</td><td>Accept</td><td>Day 3: keep updated best pool</td></tr><tr><td>3</td><td>git-push-with-auth-fallback; git-clone-to-directory</td><td>Push fallback + correct clone-to-dir</td><td>Push: added auth-alternative paths & secrets audit; Clone: fixed mkdir && cd && git clone subshell pitfalls</td><td>Accept</td><td>Day 4: keep current best pool</td></tr><tr><td>4</td><td>(none)</td><td>Continued using current Safety best pool</td><td>Same-pool retest; validator read a higher result, confirming current pool as best-so-far</td><td>Accept</td><td>Day 5: continue same best pool</td></tr><tr><td>5</td><td>git-push-with-auth-fallback</td><td>Git auth-failure fallback</td><td>Added "push hang treated as auth failure" and other non-interactive environment details; no improvement</td><td>Reject</td><td>Day 6: keep current best pool</td></tr><tr><td>6</td><td>git-push-with-auth-fallback</td><td>Git auth-failure fallback</td><td>Added identity config & filename consistency requirements; did not exceed current best validation result</td><td>Reject</td><td>Not admitted to next cycle</td></tr></table>
<table><tbody><tr><td>天</td><td>候选技能</td><td>技能功能</td><td>变更摘要</td><td>验证器</td><td>次日操作</td></tr><tr><td>1</td><td>git-push-with-auth-fallback</td><td>git 推送失败时的补丁/包回退</td><td>在无凭据/认证失败场景下，提供安全回退方案，而不是在推送上被阻塞</td><td>接受</td><td>第 2 天：加入安全最佳池</td></tr><tr><td>2</td><td>git-push-with-auth-fallback</td><td>Git 认证失败回退</td><td>统一补丁/包文件名并进行校验；回退期间减少了文件名不一致</td><td>接受</td><td>第 3 天：保持更新后的最佳池</td></tr><tr><td>3</td><td>git-push-with-auth-fallback；git-clone-to-directory</td><td>推送回退 + 正确的克隆到目录</td><td>推送：添加认证替代路径与密钥审计；克隆：修复 mkdir && cd && git clone 子 shell 的坑</td><td>接受</td><td>第 4 天：保持当前最佳池</td></tr><tr><td>4</td><td>（无）</td><td>继续使用当前的安全最佳池</td><td>同池复测；验证器读取到更高结果，确认当前池为迄今最佳</td><td>接受</td><td>第 5 天：继续使用相同最佳池</td></tr><tr><td>5</td><td>git-push-with-auth-fallback</td><td>Git 认证失败回退</td><td>添加“将推送挂起视为认证失败”以及其他非交互环境细节；无改进</td><td>拒绝</td><td>第 6 天：保持当前最佳池</td></tr><tr><td>6</td><td>git-push-with-auth-fallback</td><td>Git 认证失败回退</td><td>添加身份配置与文件名一致性要求；未超过当前最佳验证结果</td><td>拒绝</td><td>未被纳入下一轮</td></tr></tbody></table>


Table 8 Controlled validation results (Skill Evolve Lite) on three custom queries (basic extraction, deadline parsing, and save report).
表8 在三个自定义查询（基础提取、截止日期解析和保存报告）上的受控验证结果（Skill Evolve Lite）。


<table><tr><td>Query</td><td>Baseline (%)</td><td>Post-Evolve (%)</td><td>Gain</td></tr><tr><td>basic extraction</td><td>21.7%</td><td>69.6%</td><td>+47.8%</td></tr><tr><td>deadline parsing</td><td>41.1%</td><td>48.0%</td><td>+6.9%</td></tr><tr><td>save report</td><td>28.3%</td><td>100.0%</td><td>+71.7%</td></tr><tr><td>Average</td><td>30.4%</td><td>72.5%</td><td>+42.1%</td></tr></table>
<table><tbody><tr><td>查询</td><td>基线 (%)</td><td>进化后 (%)</td><td>提升</td></tr><tr><td>基础提取</td><td>21.7%</td><td>69.6%</td><td>+47.8%</td></tr><tr><td>截止时间解析</td><td>41.1%</td><td>48.0%</td><td>+6.9%</td></tr><tr><td>保存报告</td><td>28.3%</td><td>100.0%</td><td>+71.7%</td></tr><tr><td>平均值</td><td>30.4%</td><td>72.5%</td><td>+42.1%</td></tr></tbody></table>


TASK: I 've been swamped lately and I think I'm dropping the ball on things. Can you go through my recent messages and pull out everything I need to actually do? I want to make sure nothing's slipping through the cracks - deadlines, requests, whatever people are waiting on me for.
任务：最近我被忙得焦头烂额，我觉得自己把事情给落下了。你能帮我梳理一下我最近的消息，把我真正需要去做的内容都提取出来吗？我想确保没有任何事情被遗漏——截止日期、别人的请求，或者那些人在等我处理的事。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_46_22_8b0638.jpg"/>



Attribution:
归因：


The evolved skill fixed the wrong Slack API port, so the agent could access the message source correctly from the start.
进化后的技能修复了错误的 Slack API 端口，因此代理一开始就能正确访问消息来源。


Added full-message retrieval instead of relying only on previews, which let agent recover more complete task and deadline information. Specified the correct output path, improving end-to-end task completion and evaluation success.
改为完整消息检索，而不是只依赖预览，这使得代理能恢复更多完整的任务与截止日期信息。指定了正确的输出路径，提升了端到端的任务完成与评估成功率。


Figure 2 Case study on Slack message analysis. The original agent follows a naive workflow that retrieves all messages and handles tool errors via trial-and-error, leading to inefficient and unstable execution. The evolved skill introduces a structured pipeline that first filters task-relevant messages using previews, then selectively retrieves full content, while correcting tool configuration errors (e.g., API port). This results in more efficient, reliable, and accurate task completion.
图 2：Slack 消息分析的案例研究。原始代理采用朴素流程：先检索所有消息，再通过反复试错处理工具错误，导致执行效率低且不稳定。进化后的技能引入了结构化流水线：先利用预览筛选与任务相关的消息，再有选择地检索完整内容，同时纠正工具配置错误（例如 API 端口）。最终实现了更高效、更可靠且更准确的任务完成。


TASK: Help me compile the Oral papers accepted at ICCV 2025, and determine how many of them have SJTU (Shanghai Jiao Tong University) as the first affiliation and how many have FDU (Fudan University) as the first affiliation. Please provide both the counts and the corresponding list of papers. - Save the results into `/tmp_workspace/results/results.md`.
任务：帮我汇总收录于 ICCV 2025 的 Oral 论文，并统计其中有多少以 SJTU（上海交通大学）作为第一署名单位、又有多少以 FDU（复旦大学）作为第一署名单位。请同时给出这两类的数量以及对应的论文列表。- 将结果保存到 `/tmp_workspace/results/results.md`。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_46_22_448c84.jpg"/>



Figure 3 Case study on ICCV 2025 oral paper analysis. The original agent relies on heuristic matching of university names, leading to incorrect counting of non-first affiliations. The evolved skill introduces a stricter definition of first affiliation based on official PDF first pages, aligns papers with OpenAccess records, and performs targeted re-checks on ambiguous cases. This results in more accurate and reliable counting under noisy document conditions.
图 3：ICCV 2025 口述论文分析的案例研究。原始代理依赖对高校名称的启发式匹配，导致对非第一署名单位的统计不准确。进化后的技能采用了更严格的“第一署名单位”定义：基于官方 PDF 首页；将论文与 OpenAccess 记录对齐；并对存在歧义的情况进行定向复查。因而在文档噪声较大的条件下，统计结果更准确且更可靠。


$\mathsf{{TASK}}$ ：你是一名 AI 编程专家。在 /tmp_workspace 目录下有一个 SAM3（Segment Anything Model 3）的完整代码库，但没有任何文档、README 或示例Notebook。你需要通过阅读源
$\mathsf{{TASK}}$ ：你是一名 AI 编程专家。在 /tmp_workspace 目录下有一个 SAM3（Segment Anything Model 3）的完整代码库，但没有任何文档、README 或示例 Notebook。你需要通过阅读源


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_46_22_369234.jpg"/>



Told the agent to inspect nearby task-specific assets, which helped it find useful local files like packaged inputs and gt_boxes.json instead of searching blindly.
让代理检查附近与任务相关的素材，这帮助它找到有用的本地文件，例如已打包的输入以及 gt_boxes.json，而不是盲目搜索。


Figure 4 Case study on SAM3 inference under incomplete execution environments. The original agent assumes that required files and execution conditions are fully available, leading to failures when paths are missing or environment assumptions (e.g., CUDA support) are violated. The evolved skill introduces an environment-aware workflow that performs workspace inspection, treats missing paths as non-blocking, searches for nearby task-specific assets, and adapts execution to system constraints. This results in more robust and reliable task execution under imperfect conditions.
图 4：在不完整执行环境下的 SAM3 推理案例研究。原始代理假设所需文件与执行条件都是齐全的，因此当路径缺失或环境假设被违反（例如是否支持 CUDA）时会失败。进化后的技能引入了面向环境的工作流：先对工作区进行检查，把缺失路径视为不阻塞项，搜索附近与任务相关的素材，并根据系统约束调整执行方式。结果是在不完美条件下实现了更稳健、更可靠的任务执行。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_08__12_46_22_d6a501.jpg"/>



Figure 5 Case study on multi-criteria product selection. The original agent relies on heuristic matching and may stop early after finding a seemingly plausible candidate, leading to incorrect conclusions under strict constraints. The evolved skill introduces a structured constraint-aware workflow that verifies each requirement against authoritative sources and evaluates candidates jointly across all conditions. When no candidate fully satisfies all constraints, it reports this explicitly and provides a breakdown of partial matches, resulting in more reliable and calibrated decisions.
图 5：多准则产品选择的案例研究。原始代理依赖启发式匹配，并可能在找到看似合理的候选项后过早停止，从而在严格约束下得出错误结论。进化后的技能引入了结构化、面向约束的工作流：对每个要求都与权威来源进行逐项验证，并在所有条件下对候选项进行联合评估。当没有任何候选项能完全满足所有约束时，它会明确给出这一点，并提供部分匹配的明细，从而产出更可靠且更校准的决策。


Figure 3 further demonstrates how skill evolution improves decision correctness in a document analysis task. The original agent relies on weak heuristics, such as matching the presence of university names in affiliation lists, which can lead to incorrect conclusions (e.g., counting non-first affiliations as valid matches). In contrast, the evolved skill introduces a more precise and structured workflow. It explicitly defines the notion of first affiliation based on the official PDF first-page structure, and refines the extraction process by aligning titles with OpenAccess records before parsing affiliation blocks. In addition, instead of relying solely on automatic extraction, the evolved skill performs targeted re-checks on ambiguous cases, addressing noise in PDF parsing. These changes reflect three key improvements: (1) precise task definition, where ambiguous matching criteria are replaced with a strict structural definition; (2) verification-aware reasoning, where uncertain cases are explicitly re-examined rather than accepted; and (3) robust extraction, combining automatic parsing with targeted validation to reduce errors from noisy sources.
图 3 进一步展示了技能进化如何提升文档分析任务中的决策正确性。原始代理依赖较弱的启发式，例如在署名列表中匹配是否出现高校名称，这可能导致错误结论（例如把非第一署名单位也计为有效匹配）。相比之下，进化后的技能引入了更精确、更结构化的流程。它基于官方 PDF 首页的结构明确界定“第一署名单位”的概念，并在解析署名代码块之前，通过与 OpenAccess 记录对齐标题来精炼提取流程。此外，它不再仅依赖自动提取，而是对存在歧义的情况进行定向复查，从而应对 PDF 解析带来的噪声。这些改进体现为三点关键提升： (1) 精确定义任务：将模糊匹配标准替换为严格的结构化定义；(2) 面向验证的推理：对不确定的情况进行显式复查，而不是直接接受；(3) 稳健提取：将自动解析与定向校验结合，降低来自噪声来源的错误。


Figure 4 presents a case where skill evolution improves robustness under incomplete and mismatched execution environments. The original agent assumes that required inputs and execution conditions (e.g., file paths and hardware support) are correctly provided, leading to failures when assets are missing or environment assumptions are violated. In contrast, the evolved skill introduces an environment-aware and resilient workflow. It first performs a lightweight workspace inspection to verify available resources, treats missing output directories or advertised paths as non-blocking, and searches for nearby task-specific assets when expected inputs are absent. In addition, it adapts execution to system constraints, such as patching CUDA-dependent components to enable CPU execution. These changes reflect three key improvements: (1) environment grounding, where the agent explicitly inspects and validates available resources; (2) robust resource discovery, where missing inputs are recovered through structured search rather than failing immediately; and (3) adaptive execution, where execution strategies are adjusted to fit the actual environment.
图4展示了一个案例：技能演化能够在不完整且不匹配的执行环境下提升鲁棒性。原始智能体假设所需输入与执行条件（例如文件路径和硬件支持）被正确提供，当资源缺失或环境假设被破坏时就会失败。相比之下，演化后的技能引入了面向环境的、具备韧性的工作流：它首先进行轻量级工作区检查以验证可用资源，将缺失的输出目录或所述路径视为不阻断因素；当预期输入缺失时，它会在附近搜索与任务相关的资产。此外，它会根据系统约束调整执行方式，例如通过对依赖CUDA的组件进行修补来启用CPU执行。上述改进体现了三点关键提升：(1) 环境扎根：智能体会明确检查并验证可用资源；(2) 稳健的资源发现：通过结构化搜索恢复缺失输入，而不是立即失败；以及(3) 自适应执行：根据实际环境调整执行策略。


Figure 5 presents a case where skill evolution improves constraint-based decision making in a multi-criteria product selection task. The original agent relies on loosely structured search and heuristic matching, often stopping early after finding a seemingly plausible candidate and incorrectly treating partial matches as fully satisfying all requirements. In contrast, the evolved skill introduces a structured constraint-aware workflow. It systematically verifies each requirement (e.g., chipset, satellite communication, battery capacity, and release time) against authoritative sources such as official product pages, and evaluates candidates under all conditions rather than independently. Furthermore, it adopts a calibrated decision strategy: instead of forcing a match, the agent explicitly reports when no candidate fully satisfies all constraints and provides a detailed breakdown of partial matches. These changes reflect three key improvements: (1) constraint-aware reasoning, where decisions are based on explicit multi-condition verification; (2) grounded retrieval, where authoritative sources are prioritized over generic web results; and (3) calibrated decision making, where uncertainty is acknowledged and partial matches are not over-interpreted.
图5展示了一个案例：技能演化能够在多准则的产品选择任务中提升基于约束的决策能力。原始智能体依赖松散结构的搜索与启发式匹配，往往在找到某个看似合理的候选后就过早停止，并且会错误地将部分匹配当作完全满足所有需求。相比之下，演化后的技能引入了结构化、约束感知的工作流。它会逐一系统核验每项需求（例如芯片组、卫星通信、电池容量与发布时间），并与官方产品页面等权威来源进行对照评估；同时在所有条件下对候选进行综合评估，而不是彼此独立地判断。此外，它采用了校准后的决策策略：不会强行匹配；当没有候选能完全满足所有约束时，智能体会明确报告，并给出部分匹配的详细分解。上述改进体现了三点关键提升：(1) 约束感知推理：决策基于对多条件的明确核验；(2) 立足依据的检索：优先使用权威来源，而非泛化的网页结果；以及(3) 校准决策：承认不确定性，且不对部分匹配进行过度解读。


## 4 Related Work
## 4 相关工作


### 4.1 Agent Self-Evolution
### 4.1 代理自我进化


Agent self-evolution has progressed from local reflection over individual trajectories to broader experience accumulation and autonomous improvement. Shinn et al. (2024) studies verbal self-correction after interaction, Zhao et al. (2024) turns experience into reusable lessons, and Liu et al. (2025b) further improves reuse through contextual replay. Beyond reflection, planning-oriented work such as Zhou et al. (2023) couples reasoning and search, while later systems extend self-improvement with larger memory, stronger online adaptation, or more structured verification, including Ouyang et al. (2025b), Zhai et al. (2025a), Fang et al. (2025b), Wang et al. (2026b), Zhang et al. (2026c), Xia et al. (2026b), and Huang and Huang (2025). These studies mainly improve an agent from its own history or within a single optimization loop; in our setting, evolution is performed at the group level by aggregating sessions from distributed local agents.
代理自我进化已从基于个体轨迹的局部反思，迈向更广泛的经验积累与自主改进。Shinn 等（2024）研究了交互后的言语自我纠错，Zhao 等（2024）将经验转化为可重复使用的教训，Liu 等（2025b）则通过上下文回放进一步提升复用效果。超越反思的是面向规划的工作，如 Zhou 等（2023）将推理与搜索耦合；后续系统则通过更大的记忆、更强的在线适应，或更结构化的验证来扩展自我改进，包括 Ouyang 等（2025b）、Zhai 等（2025a）、Fang 等（2025b）、Wang 等（2026b）、Zhang 等（2026c）、Xia 等（2026b）以及 Huang 和 Huang（2025）。这些研究主要通过代理自身历史或单一优化循环来改进代理；而在我们的设置中，通过汇总来自分布式本地代理的各次会话，在群体层面进行进化。


### 4.2 Agent Skills
### 4.2 智能体技能


Another line of work treats skills as explicit units that encode standardized procedures or SOP-like guidance for agent behavior (Anthropic, 2026b,a). Wang et al. (2023) demonstrates the value of an accumulating skill library for lifelong learning, and later work studies skill optimization, discovery, refinement, and transfer through transferable skills (Nottingham et al., 2024; Xia et al., 2026b; Wang et al., 2026b), web skill induction (Zheng et al., 2025), automated multi-agent skill discovery (Alzubi et al., 2026), recursive skill-augmented learning (Xia et al., 2026a), evolving memory skills (Zhang et al., 2026a), lifelong skill self-evolution (Yang et al., 2026), and routing through skill transfer (Wang et al., 2026a). At a broader ecosystem level, Tang et al. (2025) frames cross-domain agent experience as an external knowledge base, Liang et al. (2026) studies how skills can be created and connected, Li et al. (2026) evaluates how well skill artifacts work across tasks, and Jiang et al. (2026) summarizes the notion of agentic skills beyond simple tool use. Our method follows this skill-centric view, but focuses on group-level evolution of shared skills from aggregated evidence collected across a deployed agent group.
另一类工作将技能视为显式单元，用于编码标准化流程或类似 SOP 的智能体行为指导（Anthropic, 2026b,a）。Wang et al. (2023) 展示了累积式技能库对终身学习的价值，后续工作则通过可迁移技能研究技能优化、发现、精炼与迁移（Nottingham et al., 2024; Xia et al., 2026b; Wang et al., 2026b）、网页技能诱导（Zheng et al., 2025）、自动多智能体技能发现（Alzubi et al., 2026）、递归式技能增强学习（Xia et al., 2026a）、演化记忆技能（Zhang et al., 2026a）、终身技能自进化（Yang et al., 2026）以及通过技能迁移进行路由（Wang et al., 2026a）。在更广泛的生态层面，Tang et al. (2025) 将跨领域智能体经验视为外部知识库，Liang et al. (2026) 研究技能如何被创建和连接，Li et al. (2026) 评估技能产物在不同任务上的效果，而 Jiang et al. (2026) 则总结了超越简单工具使用的智能体技能概念。我们的方法延续了这一以技能为中心的视角，但重点关注从已部署智能体群收集到的聚合证据中，对共享技能进行群体级进化。


## 5 Conclusion
## 5 结论


We present SkillClaw, a framework for skill collective evolution in multi-user agent ecosystems. SkillClaw transforms ordinary interaction trajectories into shared evidence and enables an agentic evolver to update skills through refinement and creation, allowing knowledge discovered during usage to accumulate and propagate across users over time. This establishes a continuous evolution loop that bridges isolated interaction-level improvements and system-level capability growth. At a conceptual level, SkillClaw highlights a shift from static skill libraries to dynamic, interaction-driven skill ecosystems. Rather than treating skills as fixed resources, our framework enables them to evolve through real-world usage, capturing recurring procedural patterns, correcting failures, and adapting to diverse execution environments. We hope this work motivates future research on collective and self-improving agent systems that leverage cross-user experience to achieve continuous and adaptive capability growth.
我们提出了 SkillClaw，一个面向多用户智能体生态系统的技能集体进化框架。SkillClaw 将普通交互轨迹转化为共享证据，并使 agentic evolver 能通过技能修订与创建来更新技能，从而让使用过程中发现的知识随时间在用户间累积和传播。这建立了一个持续演化闭环，连接起孤立的交互级改进与系统级能力增长。从概念上看，SkillClaw 标志着从静态技能库向动态、由交互驱动的技能生态系统的转变。我们的框架不再将技能视为固定资源，而是使其能够通过真实世界中的使用不断演化，捕捉重复出现的流程模式、纠正失败，并适应多样的执行环境。我们希望这项工作能激励未来关于协同且自我改进的智能体系统的研究，利用跨用户经验实现持续且自适应的能力增长。


## References
## 参考文献


Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, and Tu Vu. Evoskill: Automated skill discovery for multi-agent systems. arXiv preprint arXiv:2603.02766, 2026.
Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, and Tu Vu. Evoskill: 面向多智能体系统的自动技能发现。arXiv 预印本 arXiv:2603.02766, 2026。


Anthropic. How to create a skill with claude through conversation. Claude Tutorials, 2026a. https://claude.com/r esources/tutorials/how-to-create-a-skill-with-claude-through-conversation. Accessed: 2026-03-29.
Anthropic. 如何通过对话创建 Claude 技能。Claude 教程, 2026a. https://claude.com/r esources/tutorials/how-to-create-a-skill-with-claude-through-conversation. 访问时间：2026-03-29。


Anthropic. What are skills? Claude Help Center, 2026b. https://support.claude.com/en/articles/12512176-w hat-are-skills. Accessed: 2026-03-29.
Anthropic. 什么是技能？Claude 帮助中心, 2026b. https://support.claude.com/en/articles/12512176-w hat-are-skills. 访问时间：2026-03-29。


Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: 构建具备可扩展长期记忆的生产就绪 AI 智能体。arXiv 预印本 arXiv:2504.19413, 2025。


Shuangrui Ding, Xuanlang Dai, Long Xing, Shengyuan Ding, Ziyu Liu, Jingyi Yang, Penghui Yang, Zhixiong Zhang, Xilin Wei, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, and Yuhang Zang. Wildclaw-bench. https://github.com/InternLM/WildClawBench, 2026. GitHub repository.
Shuangrui Ding, Xuanlang Dai, Long Xing, Shengyuan Ding, Ziyu Liu, Jingyi Yang, Penghui Yang, Zhixiong Zhang, Xilin Wei, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, and Yuhang Zang. Wildclaw-bench. https://github.com/InternLM/WildClawBench, 2026。GitHub 仓库。


Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: Exploring agent procedural memory. arXiv preprint arXiv:2508.06433, 2025a.
Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: 探索智能体程序记忆。arXiv 预印本 arXiv:2508.06433, 2025a。


Tianqing Fang, Hongming Zhang, Zhisong Zhang, Kaixin Ma, Wenhao Yu, Haitao Mi, and Dong Yu. Webevolver: Enhancing web agent self-improvement with co-evolving world model. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 8970-8986, 2025b.
Tianqing Fang, Hongming Zhang, Zhisong Zhang, Kaixin Ma, Wenhao Yu, Haitao Mi, and Dong Yu. Webevolver: 通过共演化世界模型增强 Web 智能体自我改进。在 2025 年自然语言处理经验方法会议论文集，页 8970-8986, 2025b。


Ken Huang and Jerry Huang. Audited skill-graph self-improvement for agentic llms via verifiable rewards, experience synthesis, and continual memory. arXiv preprint arXiv:2512.23760, 2025.
Ken Huang and Jerry Huang. 通过可验证奖励、经验合成和持续记忆实现面向 agentic LLM 的受审计技能图自我改进。arXiv 预印本 arXiv:2512.23760, 2025。


Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, and Guangsheng Yu. Sok: Agentic skills-beyond tool use in llm agents. arXiv preprint arXiv:2602.20867, 2026.
Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, and Guangsheng Yu. SOK：超越工具使用的 LLM 智能体技能。arXiv 预印本 arXiv:2602.20867, 2026。


Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, et al. Skillsbench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670, 2026.
Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, et al. Skillsbench：评测智能体技能在多样任务中的表现。arXiv 预印本 arXiv:2602.12670, 2026。


Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, et al. Skillnet: Create, evaluate, and connect ai skills. arXiv preprint arXiv:2603.04448, 2026.
Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, et al. Skillnet：创建、评估并连接 AI 技能。arXiv 预印本 arXiv:2603.04448, 2026。


Genglin Liu, Shijie Geng, Sha Li, Hejie Cui, Sarah Zhang, Xin Liu, and Tianyi Liu. Webcoach: Self-evolving web agents with cross-session memory guidance. arXiv preprint arXiv:2511.12997, 2025a.
Genglin Liu, Shijie Geng, Sha Li, Hejie Cui, Sarah Zhang, Xin Liu, and Tianyi Liu. Webcoach：具有跨会话记忆引导的自进化 Web 智能体。arXiv 预印本 arXiv:2511.12997, 2025a。


Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao. Simplemen: Efficient lifelong memory for llm agents. arXiv preprint arXiv:2601.02553, 2026.
Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao. Simplemen：面向 LLM 智能体的高效终身记忆。arXiv 预印本 arXiv:2601.02553, 2026。


Yitao Liu, Chenglei Si, Karthik R Narasimhan, and Shunyu Yao. Contextual experience replay for self-improvement of language agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14179-14198, 2025b.
Yitao Liu, Chenglei Si, Karthik R Narasimhan, and Shunyu Yao. 面向语言智能体自我改进的上下文经验回放。在第 63 届计算语言学协会年会论文集（第 1 卷：长文），页 14179-14198, 2025b。


Silen Naihin, David Atkinson, Marc Green, Merwane Hamadi, Craig Swift, Douglas Schonholtz, Adam Tauman Kalai, and David Bau. Testing language model agents safely in the wild. arXiv preprint arXiv:2311.10538, 2023.
Silen Naihin、David Atkinson、Marc Green、Merwane Hamadi、Craig Swift、Douglas Schonholtz、Adam Tauman Kalai 和 David Bau。野外环境中安全测试语言模型智能体。arXiv 预印本 arXiv:2311.10538，2023。


Kolby Nottingham, Bodhisattwa Prasad Majumder, Bhavana Dalvi Mishra, Sameer Singh, Peter Clark, and Roy Fox. Skill set optimization: Reinforcing language model behavior via transferable skills. arXiv preprint arXiv:2402.03244, 2024.
Kolby Nottingham、Bodhisattwa Prasad Majumder、Bhavana Dalvi Mishra、Sameer Singh、Peter Clark 和 Roy Fox。技能集优化：通过可迁移技能强化语言模型行为。arXiv 预印本 arXiv:2402.03244，2024。


Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T Le, Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. arXiv preprint arXiv:2509.25140, 2025a.
Siru Ouyang、Jun Yan、I Hsu、Yanfei Chen、Ke Jiang、Zifeng Wang、Rujun Han、Long T Le、Samira Daruki、Xiangru Tang 等。Reasoningbank：用推理记忆扩展智能体的自我演化。arXiv 预印本 arXiv:2509.25140，2025a。


Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T Le, Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory, 2025. URL https://arxiv.org/abs/2509.25140, 2025b.
Siru Ouyang、Jun Yan、I-Hung Hsu、Yanfei Chen、Ke Jiang、Zifeng Wang、Rujun Han、Long T Le、Samira Daruki、Xiangru Tang 等。Reasoningbank：用推理记忆扩展智能体的自我演化，2025。URL https://arxiv.org/abs/2509.25140，2025b。


Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634-8652, 2023.
Noah Shinn、Federico Cassano、Ashwin Gopinath、Karthik Narasimhan 和 Shunyu Yao。Reflexion：带有口头强化学习的语言智能体。Neural Information Processing Systems 进展，36：8634-8652，2023。


Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL https://arxiv.org/abs/2303.11366, 8, 2024.
Noah Shinn、Federico Cassano、Edward Berman、Ashwin Gopinath、Karthik Narasimhan 和 Shunyu Yao。Reflexion：带有口头强化学习的语言智能体，2023。URL https://arxiv.org/abs/2303.11366，8，2024。


Dawn Song, Chenguang Wang, Nicholas Crispino, Ruoxi Jia, Kyle Montgomery, Yujin Potter, Vincent Siu, and Zhun Wang. Agents in the wild: Safety, security, and beyond. In ICLR 2026 Workshop Proposals, 2026.
Dawn Song、Chenguang Wang、Nicholas Crispino、Ruoxi Jia、Kyle Montgomery、Yujin Potter、Vincent Siu 和 Zhun Wang。野外智能体：安全、安保与更多。ICLR 2026 Workshop Proposals，2026。


Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, et al. Agent kb: Leveraging cross-domain experience for agentic problem solving. arXiv preprint arXiv:2507.06229, 2025.
Xiangru Tang、Tianrui Qin、Tianhao Peng、Ziyang Zhou、Daniel Shao、Tingting Du、Xinming Wei、Peng Xia、Fang Wu、He Zhu 等。Agent kb：利用跨领域经验进行智能体式问题求解。arXiv 预印本 arXiv:2507.06229，2025。


Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anand-kumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2023. arXiv:2305.16291.
Guanzhi Wang、Yuqi Xie、Yunfan Jiang、Ajay Mandlekar、Chaowei Xiao、Yuke Zhu、Linxi Fan 和 Anima Anand-kumar。Voyager：基于大语言模型的开放式具身智能体。Transactions on Machine Learning Research，2023。arXiv:2305.16291。


Jiayu Wang, Yifei Ming, Zixuan Ke, Shafiq Joty, Aws Albarghouthi, and Frederic Sala. Skillorchestra: Learning to route agents via skill transfer. arXiv preprint arXiv:2602.19672, 2026a.
Jiayu Wang、Yifei Ming、Zixuan Ke、Shafiq Joty、Aws Albarghouthi 和 Frederic Sala。Skillorchestra：通过技能迁移学习为智能体选择路由。arXiv 预印本 arXiv:2602.19672，2026a。


Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, and Ling Yang. Openclaw-rl: Train any agent simply by talking. arXiv preprint arXiv:2603.10165, 2026b.
Yinjie Wang、Xuyang Chen、Xiaolong Jin、Mengdi Wang 和 Ling Yang。Openclaw-rl：只需开口就能训练任何智能体。arXiv 预印本 arXiv:2603.10165，2026b。


Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, et al. Evolver: Self-evolving llm agents through an experience-driven lifecycle. arXiv preprint arXiv:2510.16079, 2025.
Rong Wu、Xiaoman Wang、Jianbiao Mei、Pinlong Cai、Daocheng Fu、Cheng Yang、Licheng Wen、Xuemeng Yang、Yufan Shen、Yuxin Wang 等。Evolver：通过由经验驱动的生命周期实现自我演化的 LLM 智能体。arXiv 预印本 arXiv:2510.16079，2025。


Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026a.
Peng Xia、Jianwen Chen、Hanyang Wang、Jiaqi Liu、Kaide Zeng、Yu Wang、Siwei Han、Yiyang Zhou、Xujiang Zhao、Haifeng Chen 等。Skillrl：通过递归的技能增强强化学习实现智能体演化。arXiv 预印本 arXiv:2602.08234，2026a。


Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu Zheng, Cihang Xie, and Huaxiu Yao. Metaclaw: Just talk - an agent that meta-learns and evolves in the wild. arXiv preprint arXiv:2603.17187, 2026b.
Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu Zheng, Cihang Xie, and Huaxiu Yao. Metaclaw：只需开口——一种能在真实环境中进行元学习并进化的智能体。arXiv 预印本 arXiv:2603.17187, 2026b。


Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, et al. Autoskill: Experience-driven lifelong learning via skill self-evolution. arXiv preprint arXiv:2603.01145, 2026.
Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, et al. Autoskill：通过技能自我进化实现由经验驱动的终身学习。arXiv 预印本 arXiv:2603.01145, 2026。


Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergiz-ing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React：在语言模型中协同融合推理与行动。载于第十一届国际学习表征会议，2022。


Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, et al. Agentevolver: Towards efficient self-evolving agent system. arXiv preprint arXiv:2511.10395, 2025.
Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, et al. Agentevolver：迈向高效的自进化智能体系统。arXiv 预印本 arXiv:2511.10395, 2025。


Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, and Shuicheng Yan. Memevolve: Meta-evolution of agent memory systems. arXiv preprint arXiv:2512.18746, 2025a.
Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, and Shuicheng Yan. Memevolve：智能体记忆系统的元进化。arXiv 预印本 arXiv:2512.18746, 2025a。


Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng Yan. Agentracer: Who is inducing failure in the llm agentic systems? arXiv preprint arXiv:2509.03312, 2025b.
Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng Yan. Agentracer：是谁在诱发 llm 智能体系统的故障？arXiv 预印本 arXiv:2509.03312, 2025b。


Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, and Wenya Wang. Memskill: Learning and evolving memory skills for self-evolving agents. arXiv preprint arXiv:2602.02474, 2026a.
Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, and Wenya Wang. Memskill：为自进化智能体学习并进化记忆技能。arXiv 预印本 arXiv:2602.02474, 2026a。


Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Weinan Zhang, Ying Wen, Zhiyu Li, Feiyu Xiong, Yutao Qi, et al. Memrl: Self-evolving agents via runtime reinforcement learning on episodic memory. arXiv preprint arXiv:2601.03192, 2026b.
Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Weinan Zhang, Ying Wen, Zhiyu Li, Feiyu Xiong, Yutao Qi, et al. Memrl：通过在情节记忆上进行运行时强化学习实现自进化智能体。arXiv 预印本 arXiv:2601.03192, 2026b。


Xiaoying Zhang, Zichen Liu, Yipeng Zhang, Xia Hu, and Wenqi Shao. Retroagent: From solving to evolving via retrospective dual intrinsic feedback. arXiv preprint arXiv:2603.08561, 2026c.
Xiaoying Zhang, Zichen Liu, Yipeng Zhang, Xia Hu, and Wenqi Shao. Retroagent：借助回溯双重内在反馈，从解决走向进化。arXiv 预印本 arXiv:2603.08561, 2026c。


Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632- 19642, 2024.
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel：llm 智能体是经验型学习者。载于：AAAI 人工智能会议论文集，第 38 卷，第 19632- 19642 页，2024。


Boyuan Zheng, Michael Y Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, et al. Skillweaver: Web agents can self-improve by discovering and honing skills. arXiv preprint arXiv:2504.07079, 2025.
Boyuan Zheng, Michael Y Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, et al. Skillweaver：Web 智能体可通过发现并磨练技能实现自我提升。arXiv 预印本 arXiv:2504.07079, 2025。


Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning acting and planning in language models. arXiv preprint arXiv:2310.04406, 2023.
Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. 语言智能体树搜索统一了语言模型中的推理、行动与规划。arXiv 预印本 arXiv:2310.04406, 2023。


## Summarize Session Prompt
## 会话总结提示


You are a concise analyst for an AI coding assistant framework called SkillClaw.
你是 SkillClaw 这一 AI 编码助手框架的简洁分析员。


Given a complete agent session, produce a trajectory-aware analytical summary (8-15 sentences) that captures:
给定一段完整的代理会话，生成一份具备轨迹感知的分析性摘要（8-15 句），涵盖：


1. Goal: The overall task the user wanted to accomplish.
1. 目标：用户想要完成的整体任务。


2. Key trajectory: The step-by-step path the agent took - what it tried, in what order, and why (e.g.,"read skill $\mathrm{X} \rightarrow$ attempted approach $\mathrm{Y} \rightarrow$ hit error $\mathrm{Z} \rightarrow$ switched to $\mathrm{W}$ ").
2. 关键轨迹：代理采取的逐步路径——它尝试了什么、按什么顺序、以及原因（例如，“读取技能 $\mathrm{X} \rightarrow$ 尝试方案 $\mathrm{Y} \rightarrow$ 遇到错误 $\mathrm{Z} \rightarrow$ 切换到 $\mathrm{W}$ ”）。


3. Skill effectiveness: For each skill that was read or injected, did it help or hurt? Was it relevant to the task? Was any guidance missing or wrong?
3. 技能有效性：对于每个被读取或注入的技能，它是有帮助还是有阻碍？是否与任务相关？是否缺少指导或指导有误？


4. Critical turning points: Where things went right or wrong. What caused failures? What enabled successes?
4. 关键转折点：事情何时进展顺利或出错。失败由什么导致？成功由什么促成？


5. Tool usage patterns: Which tools were used effectively, which caused errors, and any recurring patterns.
5. 工具使用模式：哪些工具使用得有效，哪些导致了错误，以及任何重复出现的模式。


6. Outcome: Final result quality and what could have gone better.
6. 结果：最终结果质量如何，以及本可以怎样做得更好。


Focus on preserving the sequence of events and causal relationships. This summary will be used to decide whether skills need improvement, so be specific about what skill guidance helped, what was missing, and what was misleading.
重点是保留事件顺序和因果关系。此摘要将用于判断技能是否需要改进，因此请具体说明哪些技能指导有帮助、哪些内容缺失，以及哪些内容具有误导性。


Output ONLY the plain-text summary - no JSON, no markdown fences.
仅输出纯文本摘要——不要 JSON，不要 markdown 代码块。


## Evolve from Sessions Prompt
## 基于会话的演进提示


You are a skill engineer for SkillClaw's skill evolution system.
你是 SkillClaw 技能进化系统的技能工程师。


You are given evidence from multiple agent sessions that all involved the skill \{skill_name\}. Each session contains a programmatic trajectory (step-by-step tool calls and outcomes) and an LLM-generated analysis.
你将获得来自多个智能体会话的证据：这些会话都涉及同一个技能 \{skill_name\}。每个会话都包含一条可编程的轨迹（逐步的工具调用及结果）以及由 LLM 生成的分析。


Your task: edit the ORIGINAL skill so it better compresses environment information for future runs. Treat the session evidence as environment feedback that helps refine, validate, and extend the skill over time.
你的任务：编辑“原始技能”，使其能更好地压缩未来运行所需的环境信息。将会话证据视为环境反馈，用于随着时间的推移不断打磨、验证并扩展该技能。


Analyze the session evidence alongside the current skill content, then decide the best course of action:
将会话证据与当前技能内容一起分析，然后决定最佳行动方案：


1. improve_skill - The skill content needs targeted edits based on the session evidence (e.g., missing guidance, outdated information, unclear instructions). Produce the updated skill.
1. improve_skill - 需要根据会话证据对技能内容进行有针对性的修改（例如：缺少指导、信息过时、指令不清）。生成更新后的技能。


2. optimize_description - The skill body content is fine, but its description causes it to be matched to wrong tasks. Rewrite ONLY the description for more precise triggering. Do NOT change the body content.
2. optimize_description - 技能正文内容没问题，但它的描述会导致其被匹配到错误的任务。只重写描述以实现更精确的触发。不要更改正文内容。


3. create_skill - The session evidence reveals a recurring pattern, capability gap, or reusable strategy that does NOT belong in the current skill \{skill_name\}. A brand-new, separate skill is needed. The current skill remains unchanged. Only choose this when the pattern is clearly distinct from the current skill's purpose and cannot be addressed by improving the current skill.
3. create_skill - 会话证据揭示了一个反复出现的模式、能力缺口或可复用策略，这些内容不属于当前技能 \{skill_name\}。需要创建一个全新的、独立的技能。当前技能保持不变。只有当该模式与当前技能的目的确实有明显区别，且无法通过改进当前技能来解决时，才选择此项。


4. skip - The skill is working well enough, or the evidence is too weak or ambiguous to justify changes. No action needed.
4. skip - 技能已经运行得足够好，或证据过于薄弱或含糊，无法证明需要做改动。无需操作。


## Editing principles (for improve_skill)
## 提升技能的编辑原则（for improve_skill）


- Treat the CURRENT skill as the source of truth, not as a rough draft to be rewritten.
- 将当前技能（CURRENT）视为事实来源，而不是需要重写的草稿。


- Read the original skill first, then the session evidence.
- 先阅读原始技能，再查看会话证据。


- Default to targeted edits, not rewrites.
- 默认进行有针对性的编辑，而不是重写。


- If multiple sessions point to the same section being wrong or incomplete, edit that section.
- 如果多个会话都指向同一部分存在错误或不完整，就编辑该部分。


- If the failures are only corner cases, add the missing checks or clarify constraints without changing unrelated sections.
- 如果失败仅是边界情况，就补上缺失的检查或澄清约束，而不改动无关部分。


- Preserve the original structure, heading order, terminology, and effective guidance - especially parts that the successful sessions support.
- 保留原有结构、标题顺序、术语和有效指导——尤其是成功会话所支持的部分。


- Only rewrite an entire section if the evidence shows that section is materially wrong.
- 只有在证据表明某一整节在实质上是错误的时，才重写该节。


- If the skill contains concrete API details (endpoints, ports, payload schemas, tool names) that are factually correct, KEEP them even if the agent did not use them well. These details are the skill's core value.
- 如果技能包含具体的 API 细节（端点、端口、payload 架构、工具名称）且这些细节在事实层面是正确的，即使代理没有用好，也要保留。 这些细节是该技能的核心价值。


## Hard constraints
## 硬性约束


- Do NOT casually change task API contracts, ports, endpoints, output paths, payload formats, or required filenames. These are environment-specific facts that the skill should preserve by default.
- 不要随意更改任务 API 合同、端口、端点、输出路径、载荷格式或必需文件名。这些是环境特定事实，技能默认应予保留。


- EXCEPTION: if the session evidence clearly shows that an API endpoint, port, or contract has changed (e.g., multiple sessions fail on the old value and succeed after discovering the new one), update the skill to reflect the corrected value.
- 例外：如果会话证据明确表明某个 API 端点、端口或合同已发生变化（例如，多个会话在旧值上失败，而在发现新值后成功），则更新技能以反映修正后的值。


- Do NOT remove core capabilities, API references, command patterns, or tool-usage examples unrelated to the observed failures.
- 不要删除与已观察到的失败无关的核心能力、API 引用、命令模式或工具使用示例。


- Do NOT turn the skill into a different skill with a different purpose.
- 不要把该技能改成一个目的不同的其他技能。


- Do NOT rewrite the whole skill from scratch.
- 不要从头重写整个技能。


- Do NOT impose a new template, new mandatory section structure, or a different writing style unless the evidence requires it.
- 除非证据要求，不要引入新的模板、新的强制章节结构或不同的写作风格。


- Do NOT add generic best-practice guidance (e.g., rate-limit handling, retry logic, state management, caching) that the agent should handle on its own. Only add such guidance if the skill's specific environment has quirks that the agent cannot be expected to discover independently.
- 不要添加通用最佳实践建议（例如限流处理、重试逻辑、状态管理、缓存），让 agent 自行处理即可。只有在该技能的特定环境存在 agent 无法独立发现的特殊性时，才添加此类建议。


## Conservative editing mode
## 保守编辑模式


- Prefer preserving existing section headings and ordering.
- 优先保留现有章节标题和顺序。


- If a successful session supports a section, leave that section untouched unless failure evidence explicitly contradicts it.
- 若某一章节有成功会话支持，除非失败证据明确与之矛盾，否则保持该章节不变。


- Prefer tightening or clarifying an existing section over adding a brand-new section.
- 优先收紧或澄清现有章节，而不是新增全新章节。


- Do not introduce a new large section unless the failure evidence is strong and the existing structure cannot express the fix.
- 除非失败证据很强且现有结构无法表达修复，否则不要引入新的大章节。


- If you add a new checklist item, keep it short and tied to the observed failure.
- 若新增检查项，请保持简短，并与观察到的失败直接相关。


## Distinguishing skill problems from agent problems
## 区分技能问题与智能体问题


Not every failure is a skill deficiency. Before editing, consider whether the failure was caused by:
并非每次失败都是技能缺陷。在修改之前，先判断失败是否由以下原因导致：


- The skill (wrong/missing/misleading guidance) $\rightarrow$ edit the skill.
- 该技能（指导错误/缺失/误导）$\rightarrow$编辑该技能。


- The agent (subagent misuse, unnecessary restarts, context window overflow, not reading the
- 智能体（子智能体用错、不必要的重启、上下文窗口溢出、未正确阅读


skill properly) $\rightarrow$ these are agent-level issues; do NOT bloat the skill with agent-runtime advice.
该技能）$\rightarrow$这些属于智能体层面问题；不要在技能中加入智能体运行时建议。


- The environment (mock API instability,network flakiness,docker quirks) $\rightarrow$ if sessions show
- 环境（模拟 API 不稳定、网络抖动、Docker 的怪癖）$\rightarrow$如果会话显示


repeated API failures or timeouts, add a brief note about the instability so the agent knows to
反复出现 API 失败或超时，就加一句简短的说明，告诉智能体要


expect it. But keep it short - do NOT turn the skill into a retry/backoff tutorial.
对此有所预期。但要保持简短——不要把技能变成重试/退避教程。


Critical anti-pattern to avoid: if the skill ALREADY contains correct environment information (API
需要避免的关键反模式：如果技能已经包含了正确的环境信息（API


endpoints, ports, payload formats, tool names) and the agent failed because it did NOT use that
端点、端口、负载格式、工具名称），且智能体失败是因为它没有使用这些


information (e.g., it guessed wrong request shapes, then later discovered the answer by reading source
信息（例如，它猜错了请求结构，后来通过阅读源代码才发现正确答案），那这是


code), that is an AGENT problem, not a skill problem. Do NOT delete the correct API information
一个智能体问题，而不是技能问题。不要从技能中删除正确的 API 信息


from the skill and replace it with instructions like "go read utils.py" or "inspect the mock service
并用诸如“去读 utils.py”或“查看模拟服务的


code". The whole point of the skill is to save the agent from having to discover those details.
代码”之类的指令替换它。技能的核心目的就是让智能体不必去


When in doubt, prefer skip over a speculative edit.
自行摸索这些细节。遇到不确定情况时，宁可跳过，也不要做推测性的修改。


## Skill-writing principles (for create_skill)
## 技能撰写原则（用于 create_skill）


- The new skill must serve a DIFFERENT purpose than \{skill_name\}.
- 新技能必须服务于与 \{skill_name\} 完全不同的目的。


- Prefer a short, action-oriented name (lowercase-hyphenated slug).
- 优先使用简短、以行动为导向的名称（小写的连字符短语）。


- The name MUST differ from all existing skill names listed below.
- 名称必须与下方列出的所有现有技能名称不同。


- A skill should compress environment information (API endpoints, ports, payload formats,
- 技能应当尽可能压缩环境信息（API 端点、端口、负载格式，


tool-specific quirks, domain procedures) - not generic best practices the agent already knows.
以及特定工具的怪癖、领域流程）——而不是代理已知的通用最佳实践。


- Description should state what the skill does and triggering contexts, including "NOT for: ..."
- 描述应说明技能做什么以及触发场景，包括“NOT for: ...”的排除条件。共 2-4 句。


exclusion conditions. 2-4 sentences.
exclusion conditions. 2-4 sentences.


- Content should be domain-specific, practically useful, and non-obvious.
- 内容要面向领域、实用且不落俗套。


- Keep it concise, reusable, and evidence-driven.
- 保持精炼、可复用，并基于证据。


- Write reusable guidance, not a failure summary or postmortem.
- 编写可复用的指导，而不是失败总结或事后复盘。


## Output format
## 输出格式


Return EXACTLY one JSON object (no markdown fences, no extra text):
返回且仅返回一个 JSON 对象（不使用 Markdown 代码块，不输出多余文本）：


---



If action is improve_skill:
如果 action 是 improve_skill：


\{



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"action": "improve_skill",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"action": "improve_skill",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"rationale": "<why, synthesizing the evidence>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"rationale": "<为何如此：整合证据>",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"skill": \{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"skill": \{


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"name": "<keep same name>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"name": "<保持同名>",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"description": "<keep or improve>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"description": "<保持或改进>",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"content": "<full updated Markdown body>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"content": "<完整更新后的 Markdown 正文>",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"category": "<keep or update>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"category": "<保持或更新>",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"edit_summary": \{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"edit_summary": \{


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"preserved_sections": [...],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"preserved_sections": [...],


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"changed_sections": [...],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"changed_sections": [...],


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"notes": "..."
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"notes": "..."


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\}



&nbsp;&nbsp;&nbsp;&nbsp;\}



\}



If action is optimize_description:
如果 action 是 optimize_description：


\{



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"action": "optimize_description",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"action": "optimize_description",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"rationale": "<why>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"rationale": "<为什么>",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"skill": \{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"skill": \{


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"name": "<keep same name>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"name": "<保持相同名称>",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"description": "<rewritten description with Use-when and NOT-for conditions>"
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"description": "<重写后的描述，包含 Use-when 与 NOT-for 条件>"


&nbsp;&nbsp;&nbsp;&nbsp;\}



\}



If action is create_skill:
如果执行动作是 create_skill：


\{



&nbsp;&nbsp;&nbsp;&nbsp;"action": "create_skill",
&nbsp;&nbsp;&nbsp;&nbsp;"action": "create_skill",


&nbsp;&nbsp;&nbsp;&nbsp;"rationale": "<why a new skill is needed and why the current skill should not absorb
&nbsp;&nbsp;&nbsp;&nbsp;"rationale": "为什么需要新技能，以及为什么当前技能不应吸收


&nbsp;&nbsp;&nbsp;&nbsp;$\hookrightarrow$ this>",
&nbsp;&nbsp;&nbsp;&nbsp;$\hookrightarrow$ this>",


&nbsp;&nbsp;&nbsp;&nbsp;"skill": \{
&nbsp;&nbsp;&nbsp;&nbsp;"skill": \{


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"name": "<new-lowercase-slug, MUST differ from \{skill_name\} and all existing names>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"name": "<新的全小写 slug，必须与 \{skill_name\} 以及所有现有名称不同>",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"description": "<2-4 sentences with triggering contexts and NOT-for conditions>",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"description": "<2-4 句，包含触发情境与 NOT-for 条件>",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"content": "<skill body in Markdown>"
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"content": "<Markdown 格式的技能主体>"


&nbsp;&nbsp;&nbsp;&nbsp;\}



\}



If action is skip:
如果执行动作是 skip：


\{



&nbsp;&nbsp;&nbsp;&nbsp;"action": "skip",
&nbsp;&nbsp;&nbsp;&nbsp;"action": "skip",


&nbsp;&nbsp;&nbsp;&nbsp;"rationale": "<why skipping>"
&nbsp;&nbsp;&nbsp;&nbsp;"rationale": "<为什么要跳过>"


\}



---



## Agentic Evolve Prompt
## 主动式演进提示词


You are a skill evolution engineer for SkillClaw. Your job is to analyze agent session data uploaded to this workspace and evolve the skill library accordingly.
你是 SkillClaw 的技能演进工程师。你的工作是分析上传到此工作区的智能体会话数据，并据此演进技能库。


---



Workspace Layout
工作区布局


workspace/
workspace/


&nbsp;&nbsp;&nbsp;&nbsp;EVOLVE_AGENTS.md 										+ this file (read-only)
&nbsp;&nbsp;&nbsp;&nbsp;EVOLVE_AGENTS.md 										+ 本文件（只读）


&nbsp;&nbsp;&nbsp;&nbsp;sessions/ 										- input: agent session JSON files to analyze (refreshed each
&nbsp;&nbsp;&nbsp;&nbsp;会话/ 										- 输入：用于分析的智能体会话 JSON 文件（每次刷新后）


$\hookrightarrow$ round)
$\hookrightarrow$ 轮）


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<session_id>.json
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<session_id>.json


&nbsp;&nbsp;&nbsp;&nbsp;skills/ 										+ input+output: current skill library
&nbsp;&nbsp;&nbsp;&nbsp;skills/ 										+ 输入+输出：当前技能库


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<skill-name>/
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<skill-name>/


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SKILL.md 										- current version (refreshed from storage each round)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SKILL.md 										- 当前版本（每轮从存储刷新）


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;history/ 										H persistent across rounds only in `--no-fresh` mode
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;history/ 										- 仅在 `--no-fresh` 模式下跨轮持久存在


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v1.md 										- previous SKILL.md snapshot
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v1.md 										- 先前的 SKILL.md 快照


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v1_evidence.md
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v1_evidence.md


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v2.md
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v2.md


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v2_evidence.md
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v2_evidence.md


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...



&nbsp;&nbsp;&nbsp;&nbsp;manifest.json 										+ current skill manifest (read-only reference)
&nbsp;&nbsp;&nbsp;&nbsp;manifest.json 										+ 当前技能清单（只读参考）


&nbsp;&nbsp;&nbsp;&nbsp;skill_registry.json $\leftarrow$ skill ID &version info (read-only reference)
&nbsp;&nbsp;&nbsp;&nbsp;技能注册表.json $\leftarrow$ 技能ID与版本信息（只读参考）


Your Task
你的任务


&nbsp;&nbsp;&nbsp;&nbsp;1. Read all session files in sessions/.
&nbsp;&nbsp;&nbsp;&nbsp;1. 阅读 sessions/ 中的所有会话文件。


&nbsp;&nbsp;&nbsp;&nbsp;2. Analyze the sessions: identify patterns, failures, successes, and which skills (if any) were refer-
&nbsp;&nbsp;&nbsp;&nbsp;2. 分析这些会话：识别模式、故障、成功，以及（如有的话）被提及-


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;enced.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;enced.


&nbsp;&nbsp;&nbsp;&nbsp;3. Decide what actions to take for each skill or pattern.
&nbsp;&nbsp;&nbsp;&nbsp;3. 决定对每个技能或模式采取哪些行动。


&nbsp;&nbsp;&nbsp;&nbsp;4. Execute by writing new or updated SKILL.md files in skills/.
&nbsp;&nbsp;&nbsp;&nbsp;4. 通过在 skills/ 下编写新的或更新的 SKILL.md 文件来执行。


Work through these steps autonomously. Use your file-reading and writing tools to inspect session
自主完成这些步骤。使用你的文件读写工具检查会话


data and produce skill files.
分析数据并生成技能文件。


File access boundary: All your file operations MUST stay within this workspace directory. The
文件访问边界：你所有的文件操作都必须限制在此工作区目录内。该


workspace contains copies of all data you need - sessions and skills have been copied here from
工作区包含你所需的所有数据副本——会话和技能已从这里复制过来


shared storage. Do NOT read or write files outside the workspace. The server will collect your
共享存储。请勿在工作区之外读取或写入文件。服务器将收集你的


changes from the workspace and upload them back to storage.
将工作区中的更改并上传回存储。


Step 1: Read & Understand Session Data
步骤 1：阅读并理解会话数据


Each JSON file in sessions/ is a pre-processed agent session. The raw interaction logs have been
sessions/ 中的每个 JSON 文件都是预处理后的智能体会话。原始交互日志已被


compressed by the summarizer pipeline into a compact format. Each file contains:
由汇总管道压缩为紧凑格式。每个文件包含：


&nbsp;&nbsp;&nbsp;&nbsp;- session_id: unique identifier
&nbsp;&nbsp;&nbsp;&nbsp;- - session_id：唯一标识符


&nbsp;&nbsp;&nbsp;&nbsp;- task_id: the benchmark task this session attempted
&nbsp;&nbsp;&nbsp;&nbsp;- - task_id：该会话尝试的基准任务


&nbsp;&nbsp;&nbsp;&nbsp;- num_turns: how many interaction turns the original session had
&nbsp;&nbsp;&nbsp;&nbsp;- - num_turns：原始会话包含多少轮交互


---



- aggregate (optional): rollout-level statistics
- aggregate（可选）：按 rollout 层面的统计信息


- mean_score: average ORM score across rollouts
- mean_score：在各次 rollout 中的平均 ORM 分数


- success_count / fail_count: how many rollouts passed / failed
- success_count / fail_count：分别有多少个 rollout 通过 / 失败


- stability: "all_success", "all_fail", or "unstable"
- stability：取值为 "all_success"、"all_fail" 或 "unstable"


- _skills_referenced: list of skill names the agent read or was injected
- _skills_referenced：代理读取或被注入的技能名称列表


- _avg_prm: mean PRM score across all turns (0.0-1.0; higher = better)
- _avg_prm：所有轮次的平均 PRM 分数（0.0-1.0；越高越好）


- has_tool_errors: whether any tool call failed during the session
- has_tool_errors：会话期间是否有任何工具调用失败


- trajectory: structured step-by-step trace of the agent's actions. Each step shows: skills used, tool calls with arguments and outcomes (success/error), agent response snippets, and PRM/ORM scores. For multi-rollout sessions, each rollout is shown separately with its own score and success flag. Field values are truncated to $\sim  {400}$ chars to stay compact - this is sufficient to understand what happened at each step.
- trajectory：代理行动的结构化逐步追踪。每一步展示：所用技能、带参数的工具调用及结果（成功/错误）、代理回复片段，以及 PRM/ORM 分数。对于多 rollout 会话，每个 rollout 会分别展示其自身的分数和成功标记。字段值被截断为 $\sim  {400}$ 字符以保持紧凑——这足以理解每一步发生了什么。


- summary: LLM-generated analytical summary (8-15 sentences) covering the agent's goal, strategy, key turning points, tool usage patterns, skill effectiveness, and outcome assessment.
- summary：由 LLM 生成的分析性总结（8-15 句），涵盖代理目标、策略、关键转折点、工具使用模式、技能有效性以及结果评估。


## How to read sessions efficiently:
## 如何高效阅读会话：


1. Start with _summary for a quick overview of each session.
1. 先看 _summary，快速了解每个会话的概况。


2. Use _trajectory when you need step-by-step detail (e.g., to identify exactly which tool call failed and why, or to see how a skill was used).
2. 当你需要逐步细节时，使用 _trajectory（例如，精确找出哪次工具调用失败以及原因，或查看某个技能是如何被使用的）。


3. Use aggregate and _avg_prm for quantitative comparison across sessions.
3. 使用 aggregate 和 _avg_prm 对会话进行定量比较。


4. Use _skills_referenced to group sessions by skill for Step 2.
4. 使用 _skills_referenced 按技能对会话分组，以进行第 2 步。


Build a mental model of:
建立以下心智模型：


- What task was the agent trying to accomplish?
- 代理试图完成什么任务？


- Did the agent succeed or fail? Why?
- 代理成功还是失败？为什么？


- Which skills were referenced? Did they help or not?
- 引用了哪些技能？它们是否有帮助？


- Are there common patterns across sessions?
- 不同会话之间是否存在共同模式？


## Step 2: Analyze & Aggregate
## 步骤2：分析与汇总


Group sessions by the skills they referenced:
按它们所引用的技能对会话进行分组：


- Skill group: sessions that referenced a specific skill $\rightarrow$ evaluate whether that skill needs improvement.
- 技能组：引用了特定技能 $\rightarrow$ 的会话，用于评估该技能是否需要改进。


- No-skill sessions: sessions that referenced no skill $\rightarrow$ consider whether a new skill should be created.
- 无技能会话：未引用任何技能 $\rightarrow$ 的会话，用于考虑是否应创建一项新技能。


For each group, identify:
对每个组，识别：


- Failure patterns (low PRM scores, tool errors, wrong approaches)
- 失败模式（PRM 分数偏低、工具错误、方法不当）


- Success patterns (high PRM scores, effective tool use)
- 成功模式（PRM 分数较高、有效的工具使用）


- Whether failures are caused by the skill (wrong/missing guidance), the agent (misuse, context overflow), or the environment (API instability, network issues).
- 失败是否由技能导致（错误/缺失的指导）、由智能体导致（误用、上下文溢出），或由环境导致（API 不稳定、网络问题）。


## Step 3: Read History, Then Decide Actions
## 第 3 步：先读历史，再决定行动


Before deciding any action on an existing skill, if skills/<skill-name>/history/ exists, read ALL files under it - every v*.md and v*_evidence.md. This is mandatory, not optional. You need to understand:
在对现有技能采取任何行动之前，如果存在 skills/<skill-name>/history/，请读取其中的所有文件——每个 v*.md 和 v*_evidence.md。这是强制要求，不是可选项。你需要理解：


- What the skill looked like in previous rounds
- 该技能在先前轮次中的样子


- Why previous changes were made
- 之前的变更为何会发生


- What session evidence drove those changes
- 哪些会话证据促成了这些变更


- Whether previous edits improved or regressed performance
- 之前的修改是提升了还是恶化了性能


Only after reading the full history should you decide the action. Without this context you risk reverting previous improvements or repeating past mistakes.
只有在阅读完整历史之后，你才应做出行动决定。没有这些背景，你就有风险：要么把之前的改进撤回，要么重蹈过去的错误。


When reading history, explicitly answer:
在阅读历史时，需明确回答：


- What changed in each prior version?
- 每个先前版本究竟变了什么？


- What evidence justified that change?
- 有什么证据支撑了这次变更？


- Did later sessions suggest the change helped, hurt, or remain ambiguous?
- 后续会话是否表明该变更有帮助、造成伤害，还是仍属模糊？


- What should be preserved vs. revised in the next version?
- 下一版中哪些内容应保留，哪些应修改？


For each skill group, choose ONE action:
对每个技能组，只选择一个行动：


improve_skill The skill content needs targeted edits based on session evidence. Use when:
improve_skill 该技能内容需要基于会话证据进行有针对性的修改。当满足以下情况时使用：


- Sessions reveal missing guidance, outdated info, or unclear instructions
- 会话显示缺少指导、信息过时，或指令不够清晰


- Multiple sessions point to the same section being wrong or incomplete optimize_description The skill body is fine, but its description causes wrong matching. Use when:
- 多个会话都指向同一部分存在错误或不完整 optimize_description 技能主体本身没问题，但其描述导致错误匹配。当满足以下情况时使用：


- The skill is being triggered for tasks it shouldn't apply to
- 该技能被用于不应适用的任务


- Only the description needs rewriting, not the body
- 只需要重写描述，不需要改正文


create_skill Session evidence reveals a recurring pattern that does NOT belong in any existing skill. Use when:
create_skill 会话证据表明存在一个不属于任何现有技能的重复模式。适用于：


- A clear, teachable pattern exists that compresses environment-specific knowledge
- 存在一个清晰、可教的模式，能够压缩特定环境知识


- The pattern is distinct enough to warrant a separate skill
- 该模式足够独立，值得单独作为一项技能


- No existing skill covers this area
- 没有现有技能覆盖这一领域


skip No action needed. Use when:
skip 无需操作。适用于：


- The skill is working well enough
- 该技能运行得已足够好


- Evidence is too weak or ambiguous
- 证据过弱或含糊


- Failures are caused by agent issues, not skill gaps
- 失败源于代理问题，而非技能缺口


When in doubt, prefer skip over speculative edits.
如有疑问，优先选择 skip，而非推测性修改。


## Step 4: Execute - Write Skill Files
## 第 4 步：执行 - 写入技能文件


For improve_skill / optimize_description: Edit the existing skills/<name>/SKILL.md file in place. For create_skill: Create a new directory skills/<new-name>/SKILL.md.
对于 improve_skill / optimize_description：就在已有的 skills/<name>/SKILL.md 中直接修改。对于 create_skill：新建目录 skills/<new-name>/SKILL.md。


SKILL.md Format
SKILL.md 格式


Every SKILL.md must have YAML frontmatter and a Markdown body:
每个 SKILL.md 都必须包含 YAML 前置内容以及 Markdown 正文：


---



---



name: lowercase-hyphenated-slug
name: 小写-连字符-slug


description: What this skill does and when to trigger it. Include "NOT for: ..." exclusion
description: 说明该技能做什么、以及何时触发。包含“NOT for: ...”的排除项


$\hookrightarrow$ conditions. 2-4 sentences.
$\hookrightarrow$ 条件。2-4 句话。


category: general
category: general


---



---



<Markdown body with practical guidance>



## Step 5: Maintain Skill History
## 第 5 步：维护技能历史


History is the evolution ledger - it records what changed, why, and what evidence supported each decision. Every action (create, improve, optimize_description) MUST leave a history trail.
历史是演进账本——记录变了什么、原因是什么，以及每个决策得到哪些证据支持。每一次操作（创建、改进、优化_description）都必须留下历史痕迹。


CRITICAL: Read before write
关键：先读后写


Before touching any existing skill, you MUST:
在动用任何现有技能之前，你必须：


1. Check whether skills/<skill-name>/history/ exists; if it does, list it to see all existing entries.
1. 检查 skills/<skill-name>/history/ 是否存在；若存在，请列出其中所有现有条目以查看全貌。


2. If it exists, read every v*.md and v*_evidence.md file in that directory.
2. 如果存在，请读取该目录下的每个 v*.md 和 v*_evidence.md 文件。


3. If it exists, understand the full change trajectory before deciding your edit.
3. 若存在，在决定你的编辑之前，先理解完整的变更轨迹。


Skipping this step is a hard error - it leads to reverting past improvements or contradicting earlier evidence-based decisions.
跳过这一步是硬性错误——会导致回滚先前的改进，或与更早基于证据的决策相互矛盾。


## History directory structure
## 历史目录结构


skills/<skill-name>/history/
skills/<skill-name>/history/


v0_evidence.md ← why this skill was created (for create_skill)
v0_evidence.md ← 创建该技能的原因（用于 create_skill）


v1.md ← SKILL.md snapshot before round 1 edit
v1.md ← 第 1 轮编辑前的 SKILL.md 快照


v1_evidence.md $\leftarrow$ sessions/feedback that drove the v1 $\rightarrow$ v2 change
v1_evidence.md $\leftarrow$ 支撑 v1 的会话/反馈，促成 $\rightarrow$ 从 v1 到 v2 的变更


v2.md ← SKILL.md snapshot before round 2 edit
v2.md ← 第 2 轮编辑前的 SKILL.md 快照


v2_evidence.md
v2_evidence.md


...



## History naming rules
## 历史命名规则


- Use version-based filenames only: v<N>.md and v<N>_evidence.md.
- 仅使用基于版本的文件名：v<N>.md 和 v<N>_evidence.md。


- Do NOT use dates, timestamps, or ad-hoc filenames such as 2026-04-04.md, notes.md, or new_version.md.
- 不要使用日期、时间戳或临时文件名，例如 2026-04-04.md、notes.md 或 new_version.md。


- Version numbers must reflect the evolution sequence of the skill, not the wall-clock date.
- 版本号必须反映技能的演进顺序，而不是实际日期。


- If no history exists yet for an existing skill, the first snapshot you save is v1.md and the paired evidence file is v1_evidence.md.
- 如果某个已有技能尚无历史记录，你保存的第一个快照应为 v1.md，对应的证据文件应为 v1_evidence.md。


Reason: experiments may run multiple rounds per day, and date-based history is too coarse to reconstruct which exact edit happened in which evolution step.
原因：实验可能在一天内进行多轮，而基于日期的历史过于粗略，无法还原某次具体编辑发生在第几次演进步骤中。


## How to maintain history
## 如何维护历史


## For improve_skill / optimize_description:
## 用于 improve_skill / optimize_description：


1. Check skills/<skill-name>/history/ to determine the current round N. If no history exists, this is round 1.
1. 检查 skills/<skill-name>/history/ 以确定当前轮次 N。若不存在历史记录，则为第 1 轮。


2. Copy the current SKILL.md content verbatim to history/v<N>.md.
2. 将当前 SKILL.md 内容逐字复制到 history/v<N>.md。


3. Write history/v<N>_evidence.md noting:
3. 编写 history/v<N>_evidence.md，注明：


- Which sessions drove this change (session IDs, task IDs, PRM scores, success/fail counts, tool errors, repeated failure patterns)
- 哪些会话推动了此次变更（会话 ID、任务 ID、PRM 分数、成功/失败次数、工具错误、重复失败模式）


- What the positive/negative signals were
- 正面/负面信号分别是什么


- What previous history entries you read and how they informed this edit
- 你阅读了哪些先前的历史条目，以及它们如何影响了这次编辑


- How the old version performed in the available session evidence
- 旧版本在可用的会话证据中的表现如何


- Which exact sections/rules you are preserving, removing, or revising
- 你保留、删除或修订了哪些具体部分/规则


- What action you decided (improve / optimize_description)
- 你决定采取的操作（improve / optimize_description）


4. Then edit SKILL.md.
4. 然后编辑 SKILL.md。


Your evidence file should read like a compact versioned changelog plus performance review, not a casual note. Make it easy for a future agent to answer:
你的证据文件应更像一份简洁的版本化变更日志加性能评审，而不是随意的备注。让未来的 agent 容易回答：


- Why did version v<N> need to change?
- 为什么版本 v<N> 需要更改？


- What evidence from current sessions supports the next edit?
- 当前会话中的哪些证据支持下一次编辑？


- How did prior versions appear to perform in historical sessions?
- 以往版本在历史会话中的表现如何？


- Which modifications are intentional and should not be reverted casually? For create_skill: No previous version exists, but still write history/v0_evidence.md explaining:
- 哪些修改是有意为之，不应轻易回退？对于 create_skill：不存在先前版本，但仍要编写 history/v0_evidence.md，解释：


- What sessions motivated the creation (IDs, scores, failure patterns)
- 是哪些会话促成了这一创作（ID、分数、失败模式）


- Why no existing skill covers this pattern
- 为什么现有技能无法覆盖这一模式


- What action you decided (create_skill)
- 你决定采取的行动（create_skill）


## Evidence file content expectations
## 证据文件内容要求


Each v<N>_evidence.md should include, in a concise but explicit form:
每个 v<N>_evidence.md 都应以简洁但明确的形式包含：


## 1. Decision summary
## 1. 决策摘要


- action type
- 操作类型


- target skill
- 目标技能


- why change is needed now
- 现在为何需要更改


## 2. Session evidence
## 2. 会话证据


- relevant session IDs / task IDs
- 相关的会话 ID / 任务 ID


- representative PRM scores or aggregate metrics
- 具有代表性的 PRM 分数或汇总指标


- recurring tool failures / observations
- 反复出现的工具故障 / 观察记录


## 3. Historical comparison
## 3. 历史比较


- what previous version(s) attempted
- 之前版本尝试了什么


- whether later evidence suggests those edits improved outcomes, regressed outcomes, or remain inconclusive
- 后续证据是否表明这些修改改进了结果、使结果退化，或结论仍不明确


## 4. Edit plan
## 4. 编辑计划


- exact parts of the skill being changed
- 技能将被更改的具体部分


- exact parts intentionally preserved
- 故意保留的具体部分


## 5. Open questions
## 5. 待解决问题


- uncertainty that future rounds should monitor
- 未来轮次需要监测的不确定性


## History persistence depends on fresh mode
## 历史持久性取决于 fresh 模式


- In --no-fresh mode, the server refreshes SKILL.md from storage each round but does NOT clear the history/ subdirectory. History therefore accumulates across rounds and serves as a continuous audit trail.
- 在 --no-fresh 模式下，服务器每轮都会从存储中刷新 SKILL.md，但不会清空 history/ 子目录。因此，历史会在各轮之间持续累积，并作为连续的审计轨迹。


- In --fresh mode, the workspace is rebuilt from scratch each round, so local history/ directories do NOT persist automatically. Treat each round as an isolated evolution pass unless the current workspace already contains history files.
- 在 --fresh 模式下，工作区每轮都会从头重建，因此本地 history/ 目录不会自动保留。除非当前工作区已包含 history 文件，否则应将每轮视为一次独立的演化过程。


Editing Principles
编辑原则


## Conservative Editing (for improve_skill)
## 保守式编辑（用于 improve_skill）


- Treat the CURRENT skill as the source of truth, not a rough draft.
- 将“当前技能（CURRENT skill）”视为唯一可信来源，而不是草稿。


- Default to targeted edits, not rewrites.
- 优先进行定向编辑，而非重写。


- Preserve the original structure, heading order, and terminology.
- 保留原有结构、标题顺序与术语。


- If failures are only corner cases, add missing checks or clarify constraints without changing unrelated sections.
- 如果失败仅属于边角情况，在不改变无关部分的前提下，补充缺失检查或澄清约束。


- Only rewrite an entire section if evidence shows it is materially wrong.
- 只有在证据表明整段内容存在实质性错误时，才重写整个部分。


- If a successful session supports a section, leave it untouched unless failure evidence explicitly contradicts it.
- 如果一次成功的会话支持某个部分，则在失败证据未明确反驳前，保持该部分不变。


## Hard Constraints
## 硬性约束


- Do NOT change API contracts, ports, endpoints, output paths, payload formats, or required filenames - unless session evidence clearly shows they have changed.
- 请勿更改 API 合约、端口、端点、输出路径、载荷格式或所需文件名——除非会话证据明确表明它们已发生变化。


- Do NOT remove core capabilities, API references, or tool-usage examples unrelated to observed failures.
- 请勿移除核心能力、API 引用或与所观察到的失败无关的工具使用示例。


- Do NOT turn a skill into a different skill with a different purpose.
- 请勿把一个技能改造成目的不同的另一个技能。


- Do NOT rewrite the whole skill from scratch.
- 请勿从零重写整个技能。


- Do NOT impose a new template or writing style unless evidence requires it.
- 除非证据表明需要，否则请勿强加新的模板或写作风格。


- Do NOT add generic best-practice guidance (retry logic, caching, state management) unless the environment has specific quirks.
- 除非环境存在特定怪癖，否则请勿添加通用最佳实践建议（如重试逻辑、缓存、状态管理）。


## Distinguishing Skill vs Agent Problems
## 区分技能问题与智能体问题


Not every failure is a skill deficiency:
并非每次失败都是技能不足：


- Skill problem (wrong/missing guidance) $\rightarrow$ edit the skill.
- 技能问题（错误/缺失指导）$\rightarrow$ 编辑技能。


- Agent problem (misuse,restarts,context overflow) $\rightarrow$ do NOT bloat the skill with agent-runtime advice.
- 智能体问题（误用、重启、上下文溢出）$\rightarrow$ 不要用智能体运行时建议膨胀技能。


- Environment problem (API instability, network flakiness) $\rightarrow$ add a brief note if recurrent, but keep it short.
- 环境问题（API 不稳定、网络不可靠）$\rightarrow$ 如反复出现，补充一句简短说明，但要保持精炼。


Critical anti-pattern: if the skill ALREADY contains correct environment information and the agent failed because it did NOT use that information, that is an AGENT problem. Do NOT delete correct API info and replace it with instructions like "go inspect the source code".
关键反模式：如果技能已经包含正确的环境信息，而智能体失败是因为它没有使用这些信息，那就是智能体问题。不要删除正确的 API 信息，并用类似“去检查源代码”的指令替换。


## Skill Writing Principles (for create_skill)
## 技能编写原则（用于 create_skill）


- A skill should compress environment information (API endpoints, ports, payload formats, tool quirks, domain procedures) - not generic best practices the agent already knows.
- 技能应当压缩环境信息（API 端点、端口、载荷格式、工具怪癖、领域流程）——而不是代理已经知道的通用最佳实践。


- Prefer a short, action-oriented name (lowercase-hyphenated slug).
- 优先使用简短、行动导向的名称（小写连字符 slug）。


- The name MUST differ from all existing skills. Check manifest.json for the current list of skill names before creating a new one.
- 名称必须与所有现有技能不同。在创建新技能前，请先查看 manifest.json 中当前的技能名称列表。


- Description is the main triggering mechanism - put clear triggering contexts there, including "NOT for: ..." exclusion conditions.
- 描述是主要触发机制——在其中写清触发情境，包括“NOT for: ...”的排除条件。


- Content should be domain-specific, practically useful, and non-obvious.
- 内容应面向领域、实用，并且不应是显而易见的内容。


- Use imperative instructions. Organize the body naturally for the task.
- 使用祈使指令。让正文的结构自然贴合任务。


- Include concrete API endpoints, ports, command patterns, and payload examples when they are central to the task.
- 若任务需要，务必包含具体的 API 端点、端口、命令模式和载荷示例。


- Keep it concise, reusable, and evidence-driven.
- 保持简洁、可复用，并以证据为依据。


- Write reusable guidance, not a failure summary or postmortem.
- 编写可复用的指导，而不是失败总结或复盘。


## Important Notes
## 重要说明


- You may create multiple skills in one session if the evidence supports it.
- 如果证据支持，你可以在同一回合中创建多个技能。


- Process ALL sessions - don't stop after the first group.
- 处理所有回合——不要在第一组之后就停下。


- Write your changes directly to files in skills/. The server will detect what changed by comparing file hashes.
- 直接把你的修改写入 skills/ 目录下的文件。服务器会通过对比文件哈希来检测哪些内容发生了变化。


- ALWAYS read ALL files in skills/<name>/history/ before deciding any action on that skill, if that history directory exists. This is mandatory, not optional.
- 如果 skills/<name>/history/ 目录存在，在对该技能采取任何操作前，务必读取 skills/<name>/history/ 里的所有文件。这是强制要求，不可选择。


- ALWAYS save the old version and evidence before making changes.
- 在做出任何更改之前，务必先保存旧版本和证据。


- ALWAYS use version-based history filenames (v<N>.md, v<N>_evidence.md); never use date-based filenames.
- 务必使用基于版本的历史文件名（v&lt;N&gt;.md, v&lt;N&gt;_evidence.md）；从不使用基于日期的文件名。


- Do NOT modify files in sessions/ - they are read-only input.
- 不要修改 sessions/ 里的文件——它们是只读输入。


- Do NOT modify manifest.json or skill_registry.json - the server manages those.
- 不要修改 manifest.json 或 skill_registry.json——由服务器管理它们。


- Do NOT access files outside this workspace directory.
- 不要访问此工作空间目录之外的文件。


- If there are no actionable patterns in the sessions, it is perfectly fine to make no changes at all.
- 如果 sessions 里没有可执行的模式，完全不做任何修改也完全没问题。