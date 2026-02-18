# Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation
# 训练具有执行反馈强化学习的高层调度器以应对长时程 GUI 自动化


Zehao Deng ${}^{1}$ , Tianjie Ju ${}^{2}$ , Zheng Wu ${}^{2}$ , Zhuosheng Zhang ${}^{2 * }$ , Gongshen Liu ${}^{2}$
Zehao Deng ${}^{1}$ , Tianjie Ju ${}^{2}$ , Zheng Wu ${}^{2}$ , Zhuosheng Zhang ${}^{2 * }$ , Gongshen Liu ${}^{2}$


${}^{1}$ School of Computer Science and Technology,Soochow University ${}^{2}$ School of Computer Science,Shanghai Jiao Tong University
${}^{1}$ 计算机科学与技术学院，苏州大学 ${}^{2}$ 上海交通大学 计算机科学学院


2327406010@stu.suda.edu.cn, \{jometeorie, wzh815918208, zhangzs, lgshen\}@sjtu.edu.cn
2327406010@stu.suda.edu.cn, \{jometeorie, wzh815918208, zhangzs, lgshen\}@sjtu.edu.cn


## Abstract
## 摘要


The rapid development of large vision-language model (VLM) has greatly promoted the research of GUI agent. However, GUI agents still face significant challenges in handling long-horizon tasks. First, single-agent models struggle to balance high-level capabilities and low-level execution capability, facing prevalent issues of responsibility coupling and capability conflicts. Second, agents lack awareness of the task state, leading to progress loss in long-horizon tasks. To address these challenges, we propose a staged execution-feedback reinforcement learning algorithm. Unlike training a unified policy model, we focus on training high-level scheduling models. Specifically, we propose and train two agents: a Coordinator, responsible for the strategic planning and task decomposition; and a State Tracker, responsible for context compression and information management to maintain the task's state and coherence. Based on this, we built the Coordinator-Executor-State Tracker (CES) multi-agent framework, which can be integrated with any low-level Executor model, assisting the Executor in solving long-horizon tasks through task scheduling and state management. Experiments on long-horizon task benchmarks demonstrate that CES significantly enhances the system's planning and state management capabilities. Furthermore, analysis confirms that our trained high-level scheduling module is a generalizable, plug-and-play module that significantly enhances the long-horizon capabilities of various Executors. Code can be available at https://github.com/hehehahi4/CES
大型视觉-语言模型（VLM）的迅速发展极大推动了 GUI 代理的研究。然而，GUI 代理在处理长时程任务方面仍面临显著挑战。首先，单代理模型难以在高层能力与低层执行能力之间取得平衡，存在职责耦合与能力冲突的普遍问题。其次，代理缺乏任务状态感知，导致长时程任务中的进展丢失。为应对这些挑战，我们提出了一种分阶段的执行反馈强化学习算法。与训练统一策略模型不同，我们专注于训练高层调度模型。具体地，我们提出并训练了两个代理：协调器（Coordinator），负责战略规划与任务分解；状态跟踪器（State Tracker），负责上下文压缩与信息管理，以维持任务的状态与连贯性。在此基础上，我们构建了协调-执行-状态跟踪器（CES）多代理框架，该框架可与任一低级 Executor 模型集成，通过任务调度和状态管理帮助 Executor 解决长时程任务。对长时程任务基准的实验表明，CES 能显著提升系统的规划与状态管理能力。此外，分析证实我们训练的高层调度模块具有可泛化、即插即用的特性，能显著提升多种 Executor 的长时程能力。代码可在 https://github.com/hehehahi4/CES 获取


## 1. Introduction
## 1. 引言


Graphical User Interface (GUI) agents play a crucial role in automating complex tasks $\left\lbrack  {1,4,{14},{25},{28},{31},{37},{40}}\right\rbrack$ . Traditional training paradigm for GUI agents primarily relies on Supervised Fine-Tuning (SFT), where the model learns the mapping from environmental states to actions through imitation learning [24, 35, 48]. However, SFT heavily depends on large-scale, costly, and meticulously annotated high-quality trajectory data, and often exhibits poor generalization capability in unseen environments [12, 21, 23, 33].
图形用户界面（GUI）代理在自动化复杂任务方面发挥着关键作用 $\left\lbrack  {1,4,{14},{25},{28},{31},{37},{40}}\right\rbrack$。 GUI 代理的传统训练范式主要依赖监督微调（SFT），模型通过模仿学习学习环境状态到动作的映射 [24, 35, 48]。然而，SFT 高度依赖大规模、昂贵且经过精心标注的高质量轨迹数据，且在未见环境中通常表现出较差的泛化能力 [12, 21, 23, 33]。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_929d48.jpg"/>



Figure 1. (a) Difference between simple tasks and long-horizon tasks. A simple task only involves a single action on one screen driven by atomic instruction, whereas a long-horizon task requires a complex trajectory driven by ambiguous, high-level user instruction. (b) Comparison of how existing methods and our method address long-horizon challenges. Left (Responsibility Coupling and Capability Conflict): A single agent is overloaded by coupling high-level capability and low-level execution. Our method resolves this by decoupling these roles into high-level and low-level components. Right (Lack of Task State Awareness): A single agent loses context on ambiguous screens like the Home screen. Our State Tracker provides high-semantic memory, enabling correct, context-aware decisions.
图1。(a) 简单任务与长时程任务的区别。简单任务仅在一个屏幕上执行单一动作，由原子指令驱动；而长时程任务需要由模糊的高层用户指令驱动的复杂轨迹。 (b) 现有方法与我们的方法在应对长时程挑战方面的比较。左侧（职责耦合与能力冲突）：单一代理被耦合的高层能力与低层执行能力所压垮。我们的方法通过将这两个角色解耦为高层与低层组件来解决。右侧（缺乏任务状态感知）：单一代理对像首页等模糊屏幕缺乏上下文。我们的状态跟踪器提供高语义记忆，使决策具有正确的上下文感知能力。


Recent studies [16, 23, 34, 47] have largely focused on using rule-based Reinforcement Learning (RL) to train agents. They utilizes verifiable reward functions, supplanting the need for expensive manual annotations and human feedback. RL has been demonstrated as an efficient alternative to SFT in GUI tasks, as it requires only a relatively small number of high-quality training data to achieve significant performance gains [23]. While existing methods have achieved commendable results in simple tasks [18, 21, 23], they still face two fundamental problems when confronted with long-horizon tasks, illustrated in Figure 1:
近年的研究[16, 23, 34, 47] 主要聚焦于使用基于规则的强化学习（RL）来训练代理。它们使用可验证的奖励函数，降低对昂贵人工标注和人为反馈的依赖。RL 已被证明是 GUI 任务中对 SFT 的高效替代，因为它仅需要相对较少的高质量训练数据即可实现显著的性能提升 [23]。尽管现有方法在简单任务上取得了可观的结果 [18, 21, 23]，但在面对长时程任务时仍存在两个根本性问题，如图1所示：


---



*Corresponding author
通讯作者


---



(i) Responsibility Coupling and Capability Conflict in Single-Agent Architectures. Current mainstream end-to-end model attempts to couple heterogeneous capabilities, such as long-term task planning, multi-step reasoning, GUI element grounding, and precise action execution, within a unified policy network. This design in optimization presents fundamental difficulties: a model with finite parameters struggles to simultaneously master high-level abilities, such as decomposing complex instructions, tracking task progress, alongside low-level abilities like grounding and execution. As task complexity increases, this coupling may lead to a catastrophic collapse of the model's diverse capabilities.
(i) 单一代理体系中的职责耦合与能力冲突。当前主流的端到端模型尝试将长时任务规划、多步推理、GUI 元素定位和精确执行等异构能力耦合在一个统一策略网络中。这一设计在优化上存在根本性困难：参数有限的模型很难同时掌握高层能力（如分解复杂指令、跟踪任务进度）与低层能力（如定位与执行）。随着任务复杂性的提升，这种耦合可能导致模型的多种能力发生灾难性崩溃。


(ii) Lack of Task State Awareness. In long-horizon tasks, an accurate awareness of the current progress is crucial for making correct decisions. Most methods rely on historical action sequences (e.g., Click (x, y)), but these low-level actions provide almost no task state information or semantic context. Therefore, the agent has to primarily rely on the visual information from screenshots to infer its progress. However, screenshots are an insufficient and unreliable representation of task progress. This limitation makes it difficult for the agent to determine its current position in a long task, leading to errors and progress loss.
(ii) 缺乏任务状态感知。在长时任务中，对当前进展的准确感知对做出正确决策至关重要。大多数方法依赖历史动作序列（如 Click (x, y)），但这些低级动作几乎不提供任何任务状态信息或语义上下文。因此，代理必须主要依赖截图中的视觉信息来推断进展。然而，截图是对任务进度的不足且不可靠的表示。此局限使代理难以确定自己在长任务中的当前位置，导致出错和进度丢失。


To address the challenges above, we propose a staged execution-feedback RL algorithm, where the insight is to resolve responsibility coupling and conflicting optimization objectives. To support this algorithmic paradigm, we built a Coordinator-Executor-State Tracker (CES) framework, which structurally decouples the complex automation process into three specialized agents. Unlike training a unified and overloaded policy model, our algorithm is designed to optimize specific high-level scheduling models. Specifically, we treat the low-level Executor as a frozen, plug-and-play component to provide verifiable feedback, and focus our training exclusively on the Coordinator and State Tracker. Through our staged optimization strategy, the Coordinator is trained to handle strategic planning and task decomposition, effectively decoupling high-level reasoning from low-level execution; meanwhile, the State Tracker is trained to act as dynamic memory, directly solving the lack of task state awareness by maintaining a high-semantic state summary in natural language. In summary, our main contributions are as follows:
为应对上述挑战，我们提出一个分阶段执行-反馈 RL 算法，其核心在于解决职责耦合与目标优化冲突。为支持该算法范式，我们构建了 Coordinat or-Executor-State Tracker（CES）框架，从结构上将复杂的自动化过程解耦为三个专业化的代理。与训练一个统一且负荷过重的策略模型不同，该算法旨在优化特定的高层调度模型。具体而言，我们将低层执行器视为冻结的、即插即用的组件以提供可验证的反馈，培训仅聚焦于协调器和状态追踪器。通过分阶段的优化策略，协调器被训练以处理战略规划和任务分解，有效将高层推理与低层执行解耦；同时，状态追踪器被训练为动态记忆，通过在自然语言中维护高语义的状态摘要，直接解决缺乏任务状态感知的问题。总之，我们的主要贡献如下：


- We build CES multi-agent framework, featuring general-purpose, plug-and-play high-level components (Coordinator and State Tracker) that can integrate with various Executors and enhance their abilities.
- 构建 CES 多代理框架，具备通用、即插即用的高层组件（协调器和状态追踪器），可与各种执行器集成并提升其能力。


- We introduce a State Tracker, whose core task is dynamic context compression and state summarization, effectively resolving the state unawareness problem and maintaining the agent's logical coherence in long-horizon tasks.
- 引入状态追踪器，其核心任务是动态上下文压缩与状态摘要化， effectively 解决状态感知不足问题并在长时任务中维持代理的逻辑连贯性。


- We propose a staged execution-feedback RL strategy. The core of this algorithm is to decouple high-level capabilities from low-level execution: it freezes a pre-trained Executor and uses the reward signals it generates to exclusively train the high-level Coordinator and State Tracker.
- 提出分阶段执行-反馈 RL 策略。该算法的核心在于将高层能力与低层执行解耦：冻结一个预训练的执行器，并利用它产生的奖励信号专门训练高层协调器和状态追踪器。


- Extensive experiments demonstrate that our method significantly enhances the long-horizon scheduling and state management capabilities of various Executor models and surpasses existing baselines.
- 大量实验表明，我们的方法显著提升了多种执行器模型的长时任务调度与状态管理能力，并超过现有基线。


## 2. Related Work
## 2. 相关工作


#### 2.1.GUI Agent
#### 2.1.GUI Agent


Traditional mobile intelligent assistants primarily relied on rule-based or intent-driven API calls or structured text representations $\left\lbrack  {6,{13},{44}}\right\rbrack$ . However,these methods are often platform-specific [32], difficult to generalize [23]. With the development of VLMs, research has shifted towards pure-vision settings [13, 40, 42]. In this pure-vision paradigm, the agent receives only screenshots and task descriptions as input, generating coordinate-based actions directly in pixel space $\left\lbrack  {4,8,{10},{44}}\right\rbrack$ .
传统的移动智能助手主要依赖基于规则或面向意图的 API 调用或结构化文本表示 $\left\lbrack  {6,{13},{44}}\right\rbrack$ 。然而，这些方法通常是平台特定的 [32]，难以泛化 [23]。随着 Visual-Language Models（VLMs）的发展，研究已转向纯视觉设定 [13, 40, 42]。在这种纯视觉范式中，代理只接收截图和任务描述作为输入，直接在像素空间生成基于坐标的动作 $\left\lbrack  {4,8,{10},{44}}\right\rbrack$ 。


Early methods [4, 25, 28, 35] commomly relied on the SFT paradigm, training models via imitation learning. However, the SFT paradigm has two major limitations: dependency on large-scale annotated data and poor generalization $\left\lbrack  {{23},{27},{29},{33},{47}}\right\rbrack$ .
早期方法 [4, 25, 28, 35] 常依赖 SFT 范式，通过模仿学习进行模型训练。然而，SFT 范式存在两大局限：对大规模标注数据的依赖和泛化能力有限 $\left\lbrack  {{23},{27},{29},{33},{47}}\right\rbrack$ 。


### 2.2. RL-Based GUI Agent
### 2.2. 基于 RL 的 GUI Agent


To enable agents to learn more strategic decision-making, recent research has gradually shifted to RL, which is a key paradigm for enhancing the planning and decision-making capabilities of GUI agents in difficult task scenarios [13, 21, 27]. RL involves fine-tuning pretrained VLMs using relatively small-scale, high-quality interaction trajectories, thereby enhancing their specialized reasoning and decision-making skills without compromising their original capabilities [14, 39, 43]. Inspired by the success of DeepSeek-R1 [5], the Group Relative Policy Optimization (GRPO) [26] algorithm has been widely adopted in the GUI agent domain. GUI-R1 [23] is a representative model that applies R1-style reinforcement learning to enhance the capabilities of VLMs in high-level, real-world GUI tasks.
为使代理能够学习更具策略性决策，最近的研究逐渐转向 RL，这是提升 GUI 代理在复杂任务场景中的规划与决策能力的关键范式 [13, 21, 27]。RL 通过相对较小规模的高质量交互轨迹对预训练的 VLM 进行微调，从而在不损害原有能力的前提下增强其专业推理和决策能力 [14, 39, 43]。受到 DeepSeek-R1 [5] 成功的启发，Group Relative Policy Optimization (GRPO) [26] 算法在 GUI 代理领域被广泛采用。GUI-R1 [23] 是一个代表性模型，应用 R1 风格的强化学习来提升 VLM 在高层、真实世界 GUI 任务中的能力。


However, these RL-based single-agent methods [18, 21- 23, 27, 29, 30], while improving the decision-making algorithm, fail to address a fundamental problem: the capability overload of a single policy network. We argue that whether using SFT or RL, coupling heterogeneous abilities like high-level strategic planning with low-level visual perception and precise execution into the same model inevitably leads to optimization conflicts and role confusion. Therefore, we innovate through architecture, thoroughly decoupling different responsibilities to enhance all facets of the agent's capabilities via specialized division of labor and collaboration.
然而，这些基于 RL 的单智能体方法 [18, 21-23, 27, 29, 30] 在提升决策算法的同时，未能解决一个根本问题：单个策略网络的能力过载。我们认为，无论使用 SFT 还是 RL，将高层策略规划等异质能力与低层视觉感知和精确执行耦合到同一模型，势必导致优化冲突与角色混淆。因此，我们通过架构创新，彻底解耦不同职责，通过专业分工与协作来提升代理的各方面能力。


### 2.3. Multi-agent Framework
### 2.3. 多智能体框架


Multi-agent systems represent a key trend in the development of LLM-driven agents [45], overcoming the inherent limitations of single agents in handling complex, dynamic, and long-horizon tasks through specialized division of labor and intelligent collaboration [15, 38]. For example, Mobile-Agent-v3 [43] coordinates four agents to achieve robust, adaptive task automation with knowledge evolution capabilities. MobiAgent [45] adopts a Planner-Decider-Grounder multi-agent architecture, combined with an acceleration framework to improve execution accuracy.
多智能体系统是以LLM驱动代理发展的关键趋势之一[45]，通过专业分工与智能协作[15, 38]克服单代理在处理复杂、动态和长时间任务方面的固有局限性。例如，Mobile-Agent-v3 [43] 协调四个代理实现具有知识演化能力的鲁棒、可自适应任务自动化。MobiAgent [45] 采用计划-决策-落地的多代理架构，并结合加速框架以提升执行准确性。


However, existing works [9, 17, 43] often use general VLMs, employing prompt engineering to make models play different roles. This collaboration lacks deep optimization for each role, making it difficult to achieve optimal efficiency and specialized capability. Furthermore, existing research often overlooks the importance of task state management and information compression in long-horizon tasks, merely using historical actions [20, 27] or passively storing large amounts of redundant information [7, 11]. Our CES framework, by introducing a dedicated State Tracker and an execution-feedback RL strategy, provides specialized and efficient optimization paths for each multi-agent role.
然而，现有工作[9, 17, 43] 常使用通用的VLM，通过提示工程使模型扮演不同角色。这种协作缺乏对每个角色的深度优化，使其难以实现最优的效率和专业化能力。此外，现有研究往往忽视在长时间任务中对任务状态管理和信息压缩的重要性，仅使用历史动作[20, 27]或被动地存储大量冗余信息[7, 11]。我们的CES框架通过引入专门的状态跟踪器和执行反馈强化学习策略，为每个多代理角色提供专门且高效的优化路径。


## 3. Preliminary Experiment
## 3. 初步实验


In the Introduction, we posited a key hypothesis: GUI agents fail in long-horizon tasks due to a lack of task state awareness, arguing that screenshots are an insufficient and unreliable representation of task progress.
在引言中，我们提出了一个关键假设：GUI代理在长时间任务中失败，是因为缺乏任务状态感知；屏幕截图不足以可靠地表示任务进展。


To empirically validate this specific hypothesis, we designed a temporal reasoning experiment. This experiment is built upon a key logic: if a screenshot were a sufficient representation of state, an agent should be able to reliably determine its relative position in a task's timeline. Our experiment directly tests this ability by tasking three powerful GUI agents to determine the correct temporal order of two screenshots sampled from the same task trajectory.
为实证验证这个具体假设，我们设计了一个时序推理实验。该实验建立在一个关键逻辑之上：如果屏幕截图是状态的充分表示，代理应能够可靠地确定其在任务时间线中的相对位置。我们的实验直接通过让三位强大的GUI代理来判断来自同一任务轨迹的两张截图的正确时间顺序来测试这一能力。


Results in Figure 2, reveal a clear dichotomy. The model's accuracy is high for adjacent steps, but drops dramatically as the step interval between the screenshots increases. This failure occurs for two primary reasons: First, a long-horizon task often involves repeating screenshots, such as the Home screen or going back to an interface, leading to confusion about its true progress. Second, tasks frequently navigate into Out-of-Distribution (OOD) interfaces [36], such as deep, specialized menus that the model has not encountered during training. When faced with an unfamiliar OOD screen, the model has no prior knowledge or context to determine its state in the overall task.
图2中的结果揭示了清晰的二分性。模型在相邻步骤中的准确性较高，但随着截图之间的步长增加，准确性显著下降。这一失败主要有两方面原因：第一，长时间任务往往涉及重复的截图，例如主页界面或返回到某个界面，导致对真实进度的混淆。第二，任务常常进入分布外（OOD）界面[36]，如模型在训练中未遇到的深入、专业菜单等。当遇到不熟悉的OOD屏幕时，模型没有先验知识或上下文来确定其在整个任务中的状态。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_82863d.jpg"/>



Figure 2. Temporal Judgement Accuracy. While accuracy is high for adjacent steps, it drops dramatically as the step interval increases. This result empirically demonstrates that screenshots fail to represent task state sufficiently and we need a mechanism to record progress for long-horizon tasks.
图2. 时序判断准确性。尽管在相邻步骤的准确性较高，但步长增加时会显著下降。这一结果经验性地证明了屏幕截图不足以充分表示任务状态，我们需要一种记录长时间任务进展的机制。


This failure highlights the critical need for a mechanism to explicitly maintain high-level semantic state. To address both this critical state awareness bottleneck and the challenge of responsibility coupling outlined in our introduction, we then introduce our CES framework and the staged execution-feedback RL algorithm in the subsequent section.
这一失败凸显了显式维持高层语义状态的关键需求。为解决此关键的状态感知瓶颈以及引言中所述的职责耦合挑战，我们在随后的章节中引入了CES框架及分阶段执行反馈RL算法。


## 4. Methodology
## 4. 方法学


### 4.1. Preliminary
### 4.1. 初步


We define the GUI task as a Markov Decision Process (MDP),characterized by the tuple $\langle \mathcal{S},\mathcal{A},\mathcal{F},R\rangle$ ,where $\mathcal{S}$ is the state space containing all possible GUI environment states; $\mathcal{A}$ is the action space representing all executable atomic actions; $\mathcal{F} : \mathcal{S} \times  \mathcal{A} \rightarrow  \mathcal{S}$ is the state transition function; and $R : \mathcal{S} \times  \mathcal{A} \rightarrow  \left\lbrack  {0,1}\right\rbrack$ is the reward function. At each timestep $t$ ,the agent’s policy $\pi$ generates a chain of thought $t{h}^{t}$ and an action ${a}^{t} \in  \mathcal{A}$ based on the user-specified natural language instruction $q$ ,the current screen state ${s}^{t} \in  \mathcal{S}$ ,and the historical action information ${H}^{t}$ ,that is $\left( {t{h}^{t},{a}^{t}}\right)  \sim  \pi \left( {\cdot  \mid  q,{s}^{t},{H}^{t}}\right)$ . Our goal is to learn an optimal policy ${\pi }^{ * }$ that maximizes the cumulative reward. However,directly learning an end-to-end policy $\pi$ faces significant challenges. Therefore, we decompose this decision-making process into a structured multi-agent collaborative loop, transforming the original MDP problem into a series of more manageable sub-problems.
我们将GUI任务定义为一个马尔可夫决策过程（MDP），其由元组 $\langle \mathcal{S},\mathcal{A},\mathcal{F},R\rangle$ 表征，其中 $\mathcal{S}$ 是包含所有可能GUI环境状态的状态空间；$\mathcal{A}$ 是表示所有可执行原子动作的行动空间；$\mathcal{F} : \mathcal{S} \times  \mathcal{A} \rightarrow  \mathcal{S}$ 是状态转移函数；$R : \mathcal{S} \times  \mathcal{A} \rightarrow  \left\lbrack  {0,1}\right\rbrack$ 是奖励函数。在每个时间步 $t$，代理的策略 $\pi$ 基于用户指定的自然语言指令 $q$、当前屏幕状态 ${s}^{t} \in  \mathcal{S}$、以及历史动作信息 ${H}^{t}$，即 $\left( {t{h}^{t},{a}^{t}}\right)  \sim  \pi \left( {\cdot  \mid  q,{s}^{t},{H}^{t}}\right)$，生成一连串推理过程 $t{h}^{t}$ 和一个动作 ${a}^{t} \in  \mathcal{A}$。我们的目标是学习一个使累计奖励最大化的最优策略 ${\pi }^{ * }$。然而，直接学习端到端策略 $\pi$ 面临显著挑战。因此，我们将这一决策过程分解为一个结构化的多代理协作循环，将原始MDP问题转化为一系列更易处理的子问题。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_cf6752.jpg"/>



Figure 3. The CES multi-agent loop framework. CES executes complex long-horizon tasks through the collaboration of three specialized agents. The Coordinator, as the task scheduling and decision-making core, combines the user's high-level instruction and the current task state (provided by the State Tracker) to decompose the task into a clear atomic instruction. The Executor, acting as the tool, precisely executes this atomic instruction and interacts with the GUI environment. Finally, the State Tracker, as the memory, observes the Executor's output and updates it into a high-semantic task state summary, which is then fed back to the Coordinator for the next step of decision-making.
Figure 3. CES 多代理循环框架。CES 通过三种专业代理的协作来执行复杂的长周期任务。Coordinator 作为任务调度与决策核心，结合用户的高层指令与当前任务状态（由 State Tracker 提供）将任务分解为清晰的原子指令。Executor 作为工具，精准执行该原子指令并与 GUI 环境交互。最后，State Tracker 作为记忆，观察 Executor 的输出并将其更新为高语义的任务状态摘要，再反馈给 Coordinator 进入下一步决策。


#### 4.2.CES Collaborative Loop
#### 4.2.CES 协作循环


As shown in Figure 3, our CES framework operates as a loop information flow: the Coordinator plans based on the state from the State Tracker, the Executor executes the instruction, the State Tracker updates the state based on the execution result, and the Coordinator receives the new state for the next planning step. This division of labor is inspired by a modern operating system, where the Coordinator acts as the "CPU" (planning), the Executor as the "I/O Device" (action), and the State Tracker as the "Dynamic Memory" (state management). Through this OS-inspired design, we completely separate high-level strategic planning from low-level precise execution, allowing each agent to focus on its core capabilities and jointly accomplish complex long-horizon tasks through efficient collaboration.
如图 3 所示，我们的 CES 框架以循环信息流的方式运作：Coordinator 根据 State Tracker 的状态进行计划，Executor 执行指令，State Tracker 基于执行结果更新状态，Coordinator 接收新状态用于下一步规划。这一分工理念源自现代操作系统，其中 Coordinator 相当于“CPU”（计划）、Executor 相当于“输入/输出设备”（行动）、State Tracker 相当于“动态内存”（状态管理）。通过这一受操作系统启发的设计，我们完全将高层战略计划与低层精确执行分离，使每个代理都能专注于自身核心能力，并通过高效协作共同完成复杂的长周期任务。


#### 4.2.1. Coordinator
#### 4.2.1. Coordinator


The Coordinator is the strategic decision-making core of the task. Its primary responsibility is to translate the user's high-level, often ambiguous, natural language instructions into a series of clear, executable steps. To make decisions that align with the long-term strategy yet adapt to the current situation, the Coordinator fuses three types of information at each timestep $t$ : the user’s high-level instruction $q$ ,which serves as the global macro-objective,,the compressed high-semantic state summary ${m}^{t - 1}$ provided by the State Tracker,and the current screenshot ${s}^{t}$ . Formally,
Coordinator 是任务的战略决策核心。其主要职责是将用户的高层、常常含糊的自然语言指令转化为一系列清晰、可执行的步骤。为了做出符合长期策略又能适应当前情境的决策，Coordinator 在每个时间步融合三类信息 $t$ ：用户的高层指令 $q$，作为全局宏目标；由 State Tracker 提供的压缩后高语义状态摘要 ${m}^{t - 1}$；以及当前屏幕截图 ${s}^{t}$。形式化地，


$$
{l}^{t} = {\pi }_{c}\left( {q,{m}^{t - 1},{s}^{t}}\right) , \tag{1}
$$



where ${\pi }_{c}$ denotes the Coordinator agent’s policy,and ${l}^{t}$ is the generated atomic instruction, which focuses on guiding the Executor on the specific action to perform in the current environment. Furthermore, when exceptions or errors occur, the Coordinator can reflect and re-plan the task, guiding the Executor to take corrective actions and return to the correct path.
其中 ${\pi }_{c}$ 表示 Coordinator 代理的策略，${l}^{t}$ 是生成的原子指令，聚焦于在当前环境中引导 Executor 执行特定操作。此外，当发生异常或错误时，Coordinator 可以进行反思并重新规划任务，指导 Executor 采取纠正措施并回到正确路径。


#### 4.2.2. Executor
#### 4.2.2. Executor


The Executor is the action endpoint in the CES framework. Its role is strictly limited to translating the atomic instruction ${l}^{t}$ from the Coordinator into a physical operation on the interface. The core advantage of this design is the complete separation of cognitive load: the Executor does not need to understand the user's long-term intent or maintain complex task context. Its sole task is to find the target element described by ${l}^{t}$ on the current screen ${s}^{t}$ and perform the corresponding action. Formally,
执行者是 CES 框架中的动作端点。它的角色仅限于将协调器中的原子指令 ${l}^{t}$ 转化为界面上的物理操作。该设计的核心优势在于实现认知负载的完全分离：执行者无需理解用户的长期意图或维护复杂的任务上下文。它唯一的任务是找到当前屏幕上由 ${l}^{t}$ 描述的目标元素 ${s}^{t}$，并执行相应的操作。正式地，


$$
{u}^{t} = \left( {t{h}^{t},{a}^{t}}\right)  = {\pi }_{e}\left( {{l}^{t},{s}^{t}}\right) , \tag{2}
$$



where ${u}^{t}$ is the Executor’s output,including the chain of thought $t{h}^{t}$ and the standardized action ${a}^{t}$ ,and ${\pi }_{e}$ is the Executor's policy.
其中 ${u}^{t}$ 是 Executor 的输出，包含推理链 $t{h}^{t}$ 与标准化行动 ${a}^{t}$，${\pi }_{e}$ 是 Executor 的策略。


#### 4.2.3. State Tracker
#### 4.2.3. State Tracker


The State Tracker is key to solving the long-horizon challenges for GUI agents. Its core function is dynamic context compression and state updating. It acts as the framework's dynamic memory unit, shifting the agent's state understanding task from processing high-dimensional, redundant historical screenshots to a low-dimensional, high-semantic natural language space. It is a language model that does not directly perceive the GUI environment but rather infers and generates the new state summary ${m}^{t}$ by understanding the Executor’s output ${u}^{t}$ ,combined with the user intent $q$ and the previous task state ${m}^{t - 1}$ :
State Tracker 是解决 GUI 代理长期任务挑战的关键。其核心功能是动态上下文压缩与状态更新。它充当框架的动态内存单元，将代理的状态理解任务从处理高维、冗余的历史截图，转换为低维度、高语义的自然语言空间。它是一个不会直接感知 GUI 环境的语言模型，而是通过理解 Executor 的输出 ${u}^{t}$，结合用户意图 $q$ 和先前的任务状态 ${m}^{t - 1}$ 来推断并生成新的状态摘要 ${m}^{t}$：


$$
{m}^{t} = {\pi }_{s}\left( {q,{m}^{t - 1},{u}^{t}}\right) , \tag{3}
$$



where ${\pi }_{s}$ is the State Tracker’s policy. This natural language-based state evolution not only compresses information and filters visual noise but also provides an exceptionally clear and coherent basis for the Coordinator's decisions, effectively solving context loss in long-horizon tasks.
其中 ${\pi }_{s}$ 是 State Tracker 的策略。这一基于自然语言的状态演化不仅压缩信息、滤除视觉噪声，还为 Coordinator 的决策提供了极为清晰且连贯的基础，有效解决了长期任务中的上下文丢失问题。


### 4.3. Staged Execution-Feedback Reinforcement Learning
### 4.3. 分阶段执行-反馈强化学习


We post-train the Coordinator on multimodal large language model and post-train the State Tracker on large language model. Notably, in our framework, the Executor is designed as a frozen, swappable component. We do not train it, instead, we directly use a fixed, powerful pre-trained GUI model. This design allows us to focus our research on optimizing the Coordinator's planning and the State Tracker's state understanding, avoiding interference from low-level execution details. This also demonstrates the generality and plug-and-play nature of the CES framework, which can be combined with any powerful execution model as its brain to enhance long-horizon task capabilities.
我们在多模态大语言模型上对协调器进行后训练，并在大语言模型上对状态跟踪器进行后训练。值得注意的是，在我们的框架中，执行器被设计为一个冻结、可替换的组件。我们不对其进行训练，而是直接使用一个固定、强大的预训练 GUI 模型。该设计使我们能够将研究焦点放在优化协调器的规划和状态跟踪器的状态理解，避免低层执行细节的干扰。此举也展示了 CES 框架的通用性和即插即用特性；它可以与任何强大的执行模型结合，作为大脑来提升长时程任务能力。


#### 4.3.1. Warm-up SFT
#### 4.3.1. 预热 SFT


Firstly, we perform a SFT warm-up for the Coordinator and State Tracker. This aims to let each agent learn its basic role, responsibilities, and strict output format. We use trajectories from existing datasets and through automated scripts , we construct high-quality (input, output) data pairs for the preliminary fine-tuning of both agents.
首先，我们对协调器和状态跟踪器进行 SFT 预热训练。此举旨在让每个智能体了解其基本角色、职责和严格的输出格式。我们使用来自现有数据集的轨迹，并通过自动化脚本构建高质量的（输入，输出）数据对，以对两者进行初步微调。


#### 4.3.2. Rule-based RL
#### 4.3.2. 基于规则的强化学习


GRPO. To further enhance the model's generalization, we use RL to optimize the models, learning a planning policy that maximizes long-term task success. We use the GRPO algorithm [26]:
GRPO。为进一步提升模型的泛化能力，我们使用强化学习来优化模型，学习最大化长期任务成功的规划策略。我们采用 GRPO 算法 [26]：


$$
\mathcal{J}\left( \theta \right)  = \mathbb{E}\left\lbrack  {\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}\min \left( {{\rho }_{i}\left( \theta \right) {\widehat{A}}_{i},\operatorname{clip}\left( {{\rho }_{i}\left( \theta \right) ,1 \pm  \epsilon }\right) {\widehat{A}}_{i}}\right) }\right\rbrack
$$



$$
- \beta {D}_{KL}\left( {{\pi }_{\theta }\parallel {\pi }_{\text{ ref }}}\right) ,
$$



(4)
where ${\rho }_{i}\left( \theta \right)  = \frac{{\pi }_{\theta }\left( {o}_{i}\right) }{{\pi }_{{\theta }_{\text{ old }}}\left( {o}_{i}\right) }$ is the importance sampling ratio, ${\pi }_{\theta }$ and ${\pi }_{{\theta }_{\text{ old }}}$ denote the current and previous policies,and ${o}_{i}$ is the candidate output. $\epsilon$ is the clipping hyperparameter, and $\beta$ controls the KL penalty against a reference policy ${\pi }_{ref}$ .
其中 ${\rho }_{i}\left( \theta \right)  = \frac{{\pi }_{\theta }\left( {o}_{i}\right) }{{\pi }_{{\theta }_{\text{ old }}}\left( {o}_{i}\right) }$ 为重要性采样比值，${\pi }_{\theta }$ 与 ${\pi }_{{\theta }_{\text{ old }}}$ 分别表示当前策略与先前策略，${o}_{i}$ 为候选输出。$\epsilon$ 为裁剪超参数，$\beta$ 控制相对于参考策略 ${\pi }_{ref}$ 的 KL 损失。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_9fa3c8.jpg"/>



Figure 4. Our proposed staged execution-feedback RL strategy. This strategy utilizes the Execution-Feedback Reward from a fixed Executor to sequentially optimize the Coordinator (Stage 1) and State Tracker (Stage 2) in two independent training phases.
图 4. 我们提出的阶段性执行-反馈强化学习策略。该策略利用固定执行器的执行-反馈奖励，在两个独立的训练阶段（阶段 1 的协调器、阶段 2 的状态跟踪器）中顺序优化。


For each input,the model generates $N$ candidate outputs $\mathcal{O} = \left\{  {{o}_{1},{o}_{2},\ldots ,{o}_{N}}\right\}$ ,and each output is scored by a rule-based reward function to get $\mathcal{R} = \left\{  {{r}_{1},{r}_{2},\ldots ,{r}_{N}}\right\}$ . The estimated relative advantage ${\widehat{A}}_{i}$ is then calculated as:
对于每个输入，模型生成 $N$ 个候选输出 $\mathcal{O} = \left\{  {{o}_{1},{o}_{2},\ldots ,{o}_{N}}\right\}$，每个输出由基于规则的奖励函数进行评分，得到 $\mathcal{R} = \left\{  {{r}_{1},{r}_{2},\ldots ,{r}_{N}}\right\}$。随后计算估计的相对优势 ${\widehat{A}}_{i}$，公式为：


$$
{\widehat{A}}_{i} = \frac{{r}_{i} - \operatorname{mean}\left( \left\{  {{r}_{1},{r}_{2},\ldots ,{r}_{N}}\right\}  \right) }{\operatorname{std}\left( \left\{  {{r}_{1},{r}_{2},\ldots ,{r}_{N}}\right\}  \right) }. \tag{5}
$$



Execution-Feedback Reward. Assigning rewards for agents responsible for abstract tasks (like planning) is difficult. Directly evaluating the quality of the Coordinator's atomic instructions or the State Tracker's state summaries lacks an objective standard and cannot guarantee their contribution to the final task success. To address this challenge, we do not directly evaluate the intermediate outputs. Instead, these outputs are passed through the CES loop to the Executor. The action output by the Executor is then objectively scored using a verifiable, rule-based reward function. This reward signal, originating from the final execution, is back-propagated to optimize the policies of the Coordinator and State Tracker.
执行-反馈奖励。为负责抽象任务（如规划）的智能体分配奖励具有一定难度。直接评估协调器的原子指令质量或状态跟踪器的状态摘要缺乏客观标准，且不能保证它们对最终任务成功的贡献。为应对这一挑战，我们不直接评估中间输出。相反，这些输出会通过 CES 循环传递给执行器。执行器输出的动作随后通过一个可验证的、基于规则的奖励函数进行客观打分。该奖励信号来自最终执行，并被反向传播以优化协调器和状态跟踪器的策略。


Specifically, we design a rule-based reward function, termed the Execution-Feedback Reward:
具体来说，我们设计了一种基于规则的奖励函数，称为执行-反馈奖励：


$$
R = {\alpha }_{1}{R}_{\text{ format }} + {\alpha }_{2}{R}_{\text{ executor }}, \tag{6}
$$



where ${R}_{\text{ format }}$ is the format reward,rewarding the model for outputting in the <think> and <answer> tags, ${R}_{\text{ executor }}$ is the executor reward,and ${\alpha }_{1},{\alpha }_{2}$ are coefficients. By passing the Coordinator's output and the current screenshot ${s}^{t}$ to the executor,we derive the executor reward from its action:
其中 ${R}_{\text{ format }}$ 为格式奖励，奖励模型在 <think> 与 <answer> 标签中的输出，${R}_{\text{ executor }}$ 为执行者奖励，${\alpha }_{1},{\alpha }_{2}$ 为系数。通过将协调器的输出和当前截图 ${s}^{t}$ 送入执行器，我们从其行动中推导执行器奖励：


$$
{R}_{\text{ executor }} = {\gamma }_{1}{R}_{\text{ type }} + {\gamma }_{2}{R}_{\text{ param }}, \tag{7}
$$



Table 1. High-level task performance comparison on three benchmarks. Bold indicates the best, underline indicates the second best. * indicates that the data of the benchmark is from GUI-R1's original paper.
表 1. 三个基准上的高层任务性能比较。Bold 表示最佳，Underline 表示第二名。星号表示该基准的数据来自 GUI-R1 原论文。


<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">AITZ</td><td colspan="3">AMEX</td><td colspan="3">GUI-Odyssey</td></tr><tr><td>Type</td><td>GR</td><td>SR</td><td>Type</td><td>GR</td><td>SR</td><td>Type</td><td>GR</td><td>SR</td></tr><tr><td>GPT-40</td><td>Zero Shot</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>37.50</td><td>14.17</td><td>5.36*</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Zero Shot</td><td>34.23</td><td>55.27</td><td>18.11</td><td>59.52</td><td>48.24</td><td>35.10</td><td>55.60</td><td>37.78</td><td>34.37*</td></tr><tr><td>OS-Atlas-7B</td><td>SFT</td><td>38.52</td><td>44.14</td><td>25.97</td><td>55.11</td><td>40.30</td><td>33.89</td><td>60.42</td><td>39.74</td><td>26.96*</td></tr><tr><td>UI-R1-3B</td><td>RL</td><td>41.63</td><td>49.27</td><td>24.55</td><td>60.23</td><td>41.78</td><td>35.81</td><td>52.16</td><td>34.46</td><td>32.49*</td></tr><tr><td>GUI-Owl-7B</td><td>RL</td><td>53.86</td><td>52.08</td><td>32.70</td><td>61.56</td><td>48.38</td><td>40.48</td><td>60.60</td><td>45.96</td><td>35.82</td></tr><tr><td>SWIRL</td><td>Multi-Agent</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>74.87</td><td>66.39</td><td>51.65</td></tr><tr><td>GUI-R1-7B</td><td>RL</td><td>52.73</td><td>54.92</td><td>30.59</td><td>67.26</td><td>57.12</td><td>43.69</td><td>65.49</td><td>43.64</td><td>38.79*</td></tr><tr><td>+ GPT-5</td><td>Multi-Agent</td><td>62.50</td><td>59.10</td><td>40.55</td><td>72.80</td><td>52.15</td><td>35.80</td><td>70.37</td><td>49.92</td><td>42.47</td></tr><tr><td>+ CES (Ours)</td><td>Multi-Agent</td><td>64.44</td><td>64.58</td><td>43.05</td><td>77.57</td><td>61.64</td><td>48.48</td><td>79.24</td><td>63.82</td><td>53.69</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">方法</td><td colspan="3">AITZ</td><td colspan="3">AMEX</td><td colspan="3">GUI-Odyssey</td></tr><tr><td>类型</td><td>GR</td><td>SR</td><td>类型</td><td>GR</td><td>SR</td><td>类型</td><td>GR</td><td>SR</td></tr><tr><td>GPT-40</td><td>零样本</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>37.50</td><td>14.17</td><td>5.36*</td></tr><tr><td>Qwen2.5-VL-7B</td><td>零样本</td><td>34.23</td><td>55.27</td><td>18.11</td><td>59.52</td><td>48.24</td><td>35.10</td><td>55.60</td><td>37.78</td><td>34.37*</td></tr><tr><td>OS- Atlas-7B</td><td>SFT</td><td>38.52</td><td>44.14</td><td>25.97</td><td>55.11</td><td>40.30</td><td>33.89</td><td>60.42</td><td>39.74</td><td>26.96*</td></tr><tr><td>UI-R1-3B</td><td>RL</td><td>41.63</td><td>49.27</td><td>24.55</td><td>60.23</td><td>41.78</td><td>35.81</td><td>52.16</td><td>34.46</td><td>32.49*</td></tr><tr><td>GUI-Owl-7B</td><td>RL</td><td>53.86</td><td>52.08</td><td>32.70</td><td>61.56</td><td>48.38</td><td>40.48</td><td>60.60</td><td>45.96</td><td>35.82</td></tr><tr><td>SWIRL</td><td>多智能体</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>74.87</td><td>66.39</td><td>51.65</td></tr><tr><td> GUI-R1-7B</td><td>RL</td><td>52.73</td><td>54.92</td><td>30.59</td><td>67.26</td><td>57.12</td><td>43.69</td><td>65.49</td><td>43.64</td><td>38.79*</td></tr><tr><td>+ GPT-5</td><td>多智能体</td><td>62.50</td><td>59.10</td><td>40.55</td><td>72.80</td><td>52.15</td><td>35.80</td><td>70.37</td><td>49.92</td><td>42.47</td></tr><tr><td>+ CES（我们的视频）</td><td>多智能体</td><td>64.44</td><td>64.58</td><td>43.05</td><td>77.57</td><td>61.64</td><td>48.48</td><td>79.24</td><td>63.82</td><td>53.69</td></tr></tbody></table>


Table 2. Main results evaluating CES framework's Effectiveness and Generalization. Baseline: The model is used directly. CES-P: The same base model acts as all three roles within the CES framework via Prompting. CES: Our full framework, using our specialized Coordinator and State Tracker, with the base model serving as the Executor. Bold indicates the best.
表 2. 评估 CES 框架的有效性与泛化性的主要结果。基线：直接使用模型。CES-P：同一基础模型通过提示在 CES 框架中承担三种角色。CES：我们的完整框架，使用专用的协调器和状态跟踪器，基础模型作为执行者。粗体表示最佳。


<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td colspan="3">AMEX</td><td colspan="3">GUI-Odyssey</td></tr><tr><td>Type</td><td>GR</td><td>SR</td><td>Type</td><td>GR</td><td>SR</td></tr><tr><td rowspan="3">UI-R1-3B</td><td>Baseline</td><td>60.23</td><td>41.78</td><td>35.81</td><td>52.16</td><td>34.46</td><td>32.49</td></tr><tr><td>CES-P</td><td>42.52 (-17.71)</td><td>52.15 (+10.37)</td><td>29.12 (-6.69)</td><td>25.81 (-26.35)</td><td>53.05 (+18.59)</td><td>14.44 (-18.05)</td></tr><tr><td>CES</td><td>70.39 (+10.16)</td><td>66.28 (+24.50)</td><td>43.38 (+7.57)</td><td>66.37 (+14.21)</td><td>64.29 (+29.83)</td><td>38.04 (+5.55)</td></tr><tr><td rowspan="3">GUI-Owl-7B</td><td>Baseline</td><td>61.56</td><td>48.38</td><td>40.48</td><td>60.60</td><td>45.96</td><td>35.82</td></tr><tr><td>CES-P</td><td>69.18 (+7.62)</td><td>53.05 (+4.67)</td><td>44.91 (+4.43)</td><td>66.03 (+5.43)</td><td>52.15 (+6.19)</td><td>37.53 (+1.71)</td></tr><tr><td>CES</td><td>75.72 (+14.16)</td><td>61.19 (+12.81)</td><td>47.24 (+6.76)</td><td>74.87 (+14.27)</td><td>61.39 (+15.43)</td><td>46.65 (+10.83)</td></tr><tr><td rowspan="3">GUI-Owl-32B</td><td>Baseline</td><td>69.13</td><td>53.24</td><td>43.16</td><td>67.15</td><td>47.33</td><td>39.60</td></tr><tr><td>CES-P</td><td>81.02 (+11.89)</td><td>63.80 (+10.56)</td><td>56.34 (+13.18)</td><td>76.90 (+9.75)</td><td>65.10 (+17.77)</td><td>53.88 (+14.28)</td></tr><tr><td>CES</td><td>78.55 (+9.42)</td><td>63.11 (+9.87)</td><td>52.05 (+8.89)</td><td>79.58 (+12.43)</td><td>65.42 (+18.09)</td><td>56.75 (+17.15)</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">设置</td><td colspan="3">美联储卡</td><td colspan="3">GUI-奥德赛</td></tr><tr><td>类型</td><td>GR</td><td>SR</td><td>类型</td><td>GR</td><td>SR</td></tr><tr><td rowspan="3">UI-R1-3B</td><td>基线</td><td>60.23</td><td>41.78</td><td>35.81</td><td>52.16</td><td>34.46</td><td>32.49</td></tr><tr><td>CES-P</td><td>42.52 (-17.71)</td><td>52.15 (+10.37)</td><td>29.12 (-6.69)</td><td>25.81 (-26.35)</td><td>53.05 (+18.59)</td><td>14.44 (-18.05)</td></tr><tr><td>CES</td><td>70.39 (+10.16)</td><td>66.28 (+24.50)</td><td>43.38 (+7.57)</td><td>66.37 (+14.21)</td><td>64.29 (+29.83)</td><td>38.04 (+5.55)</td></tr><tr><td rowspan="3">GUI-猫头鹰-7B</td><td>基线</td><td>61.56</td><td>48.38</td><td>40.48</td><td>60.60</td><td>45.96</td><td>35.82</td></tr><tr><td>CES-P</td><td>69.18 (+7.62)</td><td>53.05 (+4.67)</td><td>44.91 (+4.43)</td><td>66.03 (+5.43)</td><td>52.15 (+6.19)</td><td>37.53 (+1.71)</td></tr><tr><td>CES</td><td>75.72 (+14.16)</td><td>61.19 (+12.81)</td><td>47.24 (+6.76)</td><td>74.87 (+14.27)</td><td>61.39 (+15.43)</td><td>46.65 (+10.83)</td></tr><tr><td rowspan="3">GUI-猫头鹰-32B</td><td>基线</td><td>69.13</td><td>53.24</td><td>43.16</td><td>67.15</td><td>47.33</td><td>39.60</td></tr><tr><td>CES-P</td><td>81.02 (+11.89)</td><td>63.80 (+10.56)</td><td>56.34 (+13.18)</td><td>76.90 (+9.75)</td><td>65.10 (+17.77)</td><td>53.88 (+14.28)</td></tr><tr><td>CES</td><td>78.55 (+9.42)</td><td>63.11 (+9.87)</td><td>52.05 (+8.89)</td><td>79.58 (+12.43)</td><td>65.42 (+18.09)</td><td>56.75 (+17.15)</td></tr></tbody></table>


where ${\gamma }_{1},{\gamma }_{2}$ are coefficients, ${R}_{\text{ type }}$ rewards the correct action type,and ${R}_{\text{ param }}$ rewards the correct action parameters, more details can be found at Appendix A.
其中 ${\gamma }_{1},{\gamma }_{2}$ 是系数，${R}_{\text{ type }}$ 对正确的行为类型给出奖励，${R}_{\text{ param }}$ 对正确的行为参数给出奖励，更多细节参见附录 A。


Staged Optimization. Based on the GRPO algorithm and the Execution-Feedback Reward, we optimize the Coordinator and State Tracker in two distinct stages, as shown in Figure 4. This strategy ensures that high-level planning and state tracking are always optimized towards goals that are verifiably effective and easily understood by the Executor.
分阶段优化。基于 GRPO 算法与执行-反馈奖励，我们在两个不同阶段对 Coordinator 与 State Tracker 进行优化，如图 4 所示。这一策略确保高层规划与状态跟踪始终朝向可被执行者验证有效且易于理解的目标进行优化。


Stage 1: Optimizing the Coordinator's Planning Capability. In this stage, we use a frozen Executor model and calculate the reward based on its execution results to update the Coordinator's policy network. Since we do not yet have a trained State Tracker,the required task state ${m}^{t - 1}$ is sourced directly from the ground-truth annotated states in our preprocessed dataset. This ensures the Coordinator focuses on learning the mapping from a "perfect" state to the optimal atomic instruction.
阶段 1：优化 Coordinator 的规划能力。在此阶段，我们使用冻结的执行器模型，并基于其执行结果来计算奖励以更新 Coordinator 的策略网络。由于我们尚未训练 State Tracker，因此所需的任务状态 ${m}^{t - 1}$ 直接来自我们预处理数据集中经 ground-truth 注释的状态。这确保了 Coordinator 专注于学习从“完美”状态到最优原子指令的映射。


Stage 2: Optimizing the State Tracker's State Evolution Capability. After the Coordinator is trained, we freeze its parameters. In this stage, the State Tracker's capability is the focus of optimization. The task state ${m}^{t - 1}$ it generates is passed through the fixed Coordinator and Executor, influencing the entire decision chain. The final Execution-Feedback Reward from the Executor is used exclusively to optimize the State Tracker, teaching it to generate the most valuable state information that helps the Coordinator make optimal decisions. In this way, the State Tracker is explicitly trained to produce state summaries that are maximally useful to the fixed Coordinator's policy, effectively learning to generate what the Coordinator understands best.
阶段 2：优化 State Tracker 的状态演化能力。在 Coordinator 训练完成后，我们冻结其参数。在此阶段，聚焦于优化 State Tracker 的能力。它生成的任务状态 ${m}^{t - 1}$ 将通过固定的 Coordinator 与 Executor 进行传递，影响整个决策链。最终来自 Executor 的执行-反馈奖励仅用于优化 State Tracker，教会它生成对 Coordinator 的策略最有价值的状态信息，帮助 Coordinator 做出最优决策。通过这种方式，明确训练 State Tracker 以产出对固定 Coordinator 的策略最有用的状态摘要， effectively 学会生成 Coordinator 研究得最透彻的内容。


## 5. Experiments
## 5. 实验


### 5.1. Settings
### 5.1 设置


Training Details. We choose Qwen2.5-VL-7B [2] as the base model for the Coordinator and Qwen3-4B [41] for the State Tracker. For the warm-up SFT, we use the LLaMA Factory framework and trained for 1 epoch with a learning rate of 5e-5 to prevent overfitting. For RL, we use the Verl framework to train the Coordinator for 10 epochs with a learning rate of 1e-6 and State Tracker for 5 epochs. We use GUI-R1-7B [23] as the Executor to calculate the reward function when training. For the reward coefficients, we set ${\alpha }_{1},{\alpha }_{2}$ to 0.1 and 0.9,respectively,and ${\gamma }_{1},{\gamma }_{2}$ to 0.2,and 0.8,respectively. All experiments are conducted on $8 \times$ 80G GPUs.
训练细节。我们选择 Qwen2.5-VL-7B [2] 作为 Coordinator 的基础模型，Qwen3-4B [41] 作为 State Tracker。对于 warm-up SFT，我们使用 LLaMA Factory 框架，训练 1 轮，学习率 5e-5，以防止过拟合。对于 RL，使用 Verl 框架训练 Coordinator 10 轮，学习率 1e-6，State Tracker 5 轮。训练时我们使用 GUI-R1-7B [23] 作为 Executor 来计算奖励函数。对于奖励系数，我们将 ${\alpha }_{1},{\alpha }_{2}$ 设为 0.1 与 0.9，分别以及 ${\gamma }_{1},{\gamma }_{2}$ 设为 0.2 与 0.8，分别。所有实验在 $8 \times$ 80G 的 GPU 上进行。


Benchmarks. To thoroughly evaluate our framework's ability to solve long-horizon tasks, we leverage three benchmarks for long-horizon, complex tasks, including AITZ [46], AMEX [3] and GUI-Odyssey [19], with an average steps of 7.5, 12.8 and 15.3 respectively. We only use the high-level instructions provided by the datasets, not the low-level ones.
基准测试。为了全面评估我们框架解决长时程任务的能力，我们采用三个长时程、复杂任务基准：AITZ [46]、AMEX [3] 与 GUI-Odyssey [19]，平均步骤分别为 7.5、12.8 和 15.3。我们仅使用数据集中提供的高层指令，而不使用低层指令。


Evaluation Metrics. Following previous work [20, 23, 35], we use three common metrics for GUI agent evaluation: (i) Type, the prediction accuracy of the action type. (ii) GR (Grounding), the click point prediction accuracy, where a prediction is considered correct if the point falls within the ground-truth bounding box. (iii) SR (Success Rate), correct only if both action type and parameters are correct. Parameter correctness includes: the point being in the bounding box, correct scroll direction, or an F1 similarity $> {0.5}$ for input text.
评估指标。遵循既有工作 [20, 23, 35]，我们使用三项常用 GUI 代理评估指标：(i) Type，行动类型的预测准确度；(ii) GR（Grounding），点击点预测准确度，若预测点落在真实边界框内则视为正确；(iii) SR（成功率），只有当行动类型与参数均正确时才算成功。参数正确性包括：点在边界框内、滚动方向正确，或输入文本的 F1 相似度 $> {0.5}$。


### 5.2. Main Results
### 5.2 主要结果


#### 5.2.1. Long-Horizon Task Performance
#### 5.2.1 长时程任务性能


As shown in Table 1, CES achieves compelling performance on complex GUI tasks by implementing atomic instruction and progress state management. On top of the GUI-R1-7B executor baseline, our method improves the Type accuracy by an average of 10.38% across all benchmarks. This significant gain strongly validates the effectiveness of our decoupled design: by having the Coordinator bear the cognitive load of high-level planning and generate clear atomic instructions, the Executor no longer needs to reason about complex task context and can focus on simple perception and localization, thereby greatly enhancing its execution accuracy and stability.
如表 1 所示，CES 通过实现原子指令与进度状态管理，在复杂 GUI 任务上取得了有说服力的性能。在 GUI-R1-7B 执行器基线之上，我们的方法在所有基准上将 Type 的准确率平均提升了 10.38%。这一显著提升强有力地验证了我们解耦设计的有效性：让 Coordinator 承担高层规划的认知负担并生成清晰的原子指令，Executor 就不再需要推理复杂的任务上下文，而是专注于简单的感知与定位，从而大幅提升执行的准确性与稳定性。


Furthermore, when we use GPT-5 via prompting as the Coordinator and State Tracker, the performance only shows a minor improvement exemplified by an average increase of only 4% in Type accuracy, while some metrics even degrade. This indicates that while powerful general-purpose models can offer some improvement through prompt engineering, the effect is unstable. In contrast, our CES framework, through targeted execution-feedback reinforcement learning, enables the Coordinator and State Tracker to learn planning and state understanding strategies that are much better aligned with GUI tasks, thus achieving stable and significant performance superiority.
此外，当我们通过提示将 GPT-5 用作协调者与状态追踪器时，性能仅有小幅提升，Type 准确率平均仅提高 4%，而某些指标甚至下降。这表明，尽管强大的通用模型通过提示工程能带来一定改进，但效果不稳定。相较之下，我们的 CES 框架通过定向执行反馈强化学习，令协调者与状态追踪器学会更贴合 GUI 任务的规划与状态理解策略，从而实现稳定且显著的性能优势。


#### 5.2.2. Effectiveness and Generalization
#### 5.2.2. 效能与泛化


To validate the efficiency and generality of our CES framework, we conducted a further experiment using three models of varying sizes. For each model, we tested three distinct configurations: (i) Baseline: The model is used directly. (ii) CES-P: The same base model acts as all three roles within the CES framework via prompting. (iii) CES: Our full framework, using our specialized Coordinator and State Tracker, with the base model serving as the Executor. We conducted extensive evaluations on two AMEX and GUI-Odyssey with long average task steps, with the results shown in Table 2.
为验证我们的 CES 框架的效率与通用性，我们进行了另一项规模不同的三模型实验。对每个模型，测试了三种配置：(i) 基线：直接使用该模型。 (ii) CES-P：同一基础模型在 CES 框架中通过提示扮演三种角色。 (iii) CES：完整框架，使用我们专门的协调者与状态追踪器，基础模型作为执行者。我们在两个 AMEX 与 GUI-Odyssey（长期平均任务步骤）上进行了广泛评估，结果如表 2 所示。


From these results, we derive several key insights:
从这些结果中，我们得到若干关键见解：


(i) Architectural Superiority: Solving State Awareness and Cognitive Load. A consistent finding across all model sizes is that the CES-P setup significantly outperforms the baseline. The baseline model fails in long-horizon tasks because it lacks a mechanism to manage task state, forcing it to infer progress from ambiguous screen-shots. The CES framework solves this by introducing a State Tracker, which provides explicit, high-semantic memory, and a Coordinator, which decouples the cognitive load of planning from the immediate burden of execution.
(i) 架构优势：解决状态感知与认知负荷。所有模型规模下的共性发现是，CES-P 设置显著优于基线。基线模型在长时程任务中失败，因为缺乏管理任务状态的机制，必须从模糊的屏幕截图中推断进度。CES 框架通过引入状态追踪器，提供明确的高语义记忆，以及通过协调者将规划认知负荷与执行的即时负担解耦，从而解决此问题。


(ii) Validating the Capability Conflict Hypothesis. While CES-P helps all models, the magnitude of the improvement reveals evidence capability conflict while training. For the small UI-R1-3B, the CES-P setup actually resulted in a performance degradation, with a significant drop of 18.05% in SR on the GUI-Odyssey. In contrast, the large GUI-Owl-32B sees a massive improvement of 13.18% on AMEX. This suggests that the 3B model, due to its limited parameter capacity, struggled to learn these distinct of planning and execution capabilities. The training-time conflict was too severe for it to overcome. Conversely, the 32B model learned planning, execution and state understanding abilities during training, even though these abilities remained coupled. The CES-P setup reveals the latent high-level skills that the 32B model had already acquired tentatively. This validates our hypothesis that it is extremely difficult to train these conflicting capabilities within a single policy network, and this challenge is magnified in models with smaller parameter counts.
(ii) 验证能力冲突假说。虽然 CES-P 对所有模型有帮助，但提升幅度的差异揭示了训练过程中的能力冲突。对于小型 UI-R1-3B，CES-P 设置实际导致性能下降，在 GUI-Odyssey 上的 SR 下降显著，降至 18.05%。相比之下，较大型号 GUI-Owl-32B 在 AMEX 上获得了 massive 提升，提升了 13.18%。这表明 3B 模型因参数容量有限，难以同时学习这些不同的规划与执行能力，训练时的冲突过于严重难以克服。相反，32B 模型在训练中就学习了规划、执行与状态理解能力，尽管这些能力仍然耦合。CES-P 设置揭示了 32B 模型已初步具备的潜在高级技能。这验证了我们的假设：在单一策略网络中训练这些冲突能力极为困难，而在参数量较小的模型中这一挑战更为放大。


(iii) The Efficacy of Specialized, Trained Components. Comparing the CES-P to CES highlights the significant value of our training strategy. For instance, on GUI-Odyssey, our specialized training boosts the Gui-Owl-32B's SR from 39.60% to 56.75%, and the GUI-Owl-7B's SR from 37.53% to 46.65%. This demonstrates that our Coordinator and State Tracker via staged execution-feedback RL are highly effective and vastly superior to simply prompting a non-specialized model for these roles. Even, we can get the comparable effect as the Gui-Owl-32B model just by using 7B Coordinator and 4B State Tracker. This also proves the effectiveness of decoupling training.
(iii) 专门化、经过训练的组件的有效性。将 CES-P 与 CES 进行对比，凸显了我们训练策略的显著价值。例如，在 GUI-Odyssey 上，我们的专门化训练将 Gui-Owl-32B 的 SR 从 39.60% 提升至 56.75%，将 GUI-Owl-7B 的 SR 从 37.53% 提升至 46.65%。这表明通过分阶段执行反馈的强化学习得到的协调者与状态追踪器，远比仅通过提示一个非专门化模型担任这些角色要高效且显著优越。甚至，我们只用 7B 的协调者与 4B 的状态追踪器就能达到与 Gui-Owl-32B 相近的效果。这也证明了训练解耦的有效性。


Table 3. Ablation study on components and training stages.
表 3. 组件与训练阶段的消融研究。


<table><tr><td rowspan="2">Model</td><td colspan="3">AMEX</td><td colspan="3">GUI-Odyssey</td></tr><tr><td>Type</td><td>GR</td><td>SR</td><td>Type</td><td>GR</td><td>SR</td></tr><tr><td>CES</td><td>77.57</td><td>61.64</td><td>48.48</td><td>79.24</td><td>63.82</td><td>53.69</td></tr><tr><td>w/o Coordinator</td><td>62.63</td><td>50.57</td><td>33.27</td><td>70.11</td><td>48.03</td><td>39.15</td></tr><tr><td>w/o State Tracker</td><td>70.62</td><td>55.70</td><td>42.08</td><td>73.34</td><td>45.10</td><td>42.52</td></tr><tr><td>w/o RL (SFT only)</td><td>72.37</td><td>53.47</td><td>36.54</td><td>72.18</td><td>51.33</td><td>42.89</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td colspan="3">AMEX</td><td colspan="3">GUI-Odyssey</td></tr><tr><td>类型</td><td>GR</td><td>SR</td><td>类型</td><td>GR</td><td>SR</td></tr><tr><td>CES</td><td>77.57</td><td>61.64</td><td>48.48</td><td>79.24</td><td>63.82</td><td>53.69</td></tr><tr><td>无协调者</td><td>62.63</td><td>50.57</td><td>33.27</td><td>70.11</td><td>48.03</td><td>39.15</td></tr><tr><td>无状态跟踪器</td><td>70.62</td><td>55.70</td><td>42.08</td><td>73.34</td><td>45.10</td><td>42.52</td></tr><tr><td>无 RL（仅 SFT）</td><td>72.37</td><td>53.47</td><td>36.54</td><td>72.18</td><td>51.33</td><td>42.89</td></tr></tbody></table>


(iv) Generality of the CES Framework. Finally, our full framework, CES, provides a substantial and consistent performance improvement over the baseline for every executor model tested. This confirms that CES is a robust and generalizable plug-and-play solution that can effectively enhance any underlying executor, dramatically improving its long-horizon task automation capabilities.
(iv) CES 框架的普适性。最终，完整框架 CES 在所有测试的执行器模型上均对基线实现了显著且一致的性能提升。这证明 CES 是一个鲁棒且可广泛泛化的即插即用解决方案，能够有效提升任何底层执行器，显著改进其长时程任务自动化能力。


### 5.3. Analysis
### 5.3. 分析


Component and Training Ablation. We removed the Coordinator, State Tracker, and the RL stage from our framework to observe the effects. As shown in Table 3, when we remove the Coordinator and feed the user's high-level instruction directly to the Executor, performance drops significantly, such as SR drops by 12.77% on GUI-Odyssey. This indicates that the Executor still struggles to understand high-level instructions and make current decisions, proving the necessity of decoupling planning from execution. When the State Tracker is removed, where we instead record the last four action histories as most method and input them to the Coordinator, performance also drops significantly. This shows the critical role of information compression and state management in long-horizon tasks; without the concise, high-semantic summary from the State Tracker, the Coordinator cannot make correct plans. Finally, when we test the model using only SFT warm-up, its performance is far below the final RL-optimized model. This demonstrates that while SFT allows agents to learn basic roles and output formats, execution-feedback RL is an indispensable step to acquire a generalizable, optimal planning policy.
组件与训练消融。我们从框架中移除了协调者（Coordinator）、状态跟踪器（State Tracker）和强化学习阶段，以观察效果。如表 3 所示，当移除协调者并将用户的高层指令直接输入给执行器时，性能显著下降，例如 GUI-Odyssey 的 SR 降低了 12.77%。这表明执行器仍难以理解高层指令并做出当前决策，证明将规划与执行解耦的必要性。当移除状态跟踪器时，我们改为将最近四次行动历史记录作为多数方法的输入给协调者，性能也显著下降。这显示了信息压缩与状态管理在长时程任务中的关键作用；若缺乏状态跟踪器所提供的简明、高语义摘要，协调者就无法做出正确的计划。最后，当仅使用 SFT 预热来测试模型时，其性能远低于最终的 RL 优化模型。这表明虽然 SFT 使代理人能够学习基本角色和输出格式，但执行反馈的 RL 是获得通用、最优规划策略的不可或缺步骤。


Failure Case Analysis. To understand the source of our CES framework's advantages, we conducted a detailed failure case analysis, with results shown in Figure 5. Specifically, the CES framework almost completely eliminated State Loss errors, reducing the count from 14% to 2% and significantly reduced Planning Error from 12% to 4%. This result demonstrates that our framework's performance gain stems precisely from our specially trained Coordinator and State Tracker, which successfully solve the core challenges of planning and state management. In contrast, the low-level errors attributable to the frozen Executor, namely Perception Error and Generalization Failure, remained largely unchanged. The performance bottleneck has now effectively shifted to the inherent perceptual limitations of the Executor itself. More case study can be found in Appendix D.
故障案例分析。为理解我们的 CES 框架优势的来源，我们进行了详细的故障案例分析，结果如图 5 所示。具体而言，CES 框架几乎完全消除了状态丢失错误，将数量从 14% 降至 2%，并显著降低了规划错误从 12% 降至 4%。这一结果表明，我们框架的性能提升恰恰来源于我们特别训练的协调者和状态跟踪器，成功解决了规划与状态管理的核心挑战。相比之下，归因于冻结执行器的低层错误，即感知错误（Perception Error）与泛化失败（Generalization Failure），基本保持不变。性能瓶颈现已实际转移到执行器本身固有的感知能力限制。更多案例研究请参见附录 D。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_a1af13.jpg"/>



Figure 5. Failure Case Analysis. Compared to the baseline, our CES framework almost completely eliminates cognitive errors like State Loss and Planning Error.
图 5。故障案例分析。与基线相比，我们的 CES 框架几乎完全消除了诸如状态丢失和规划错误等认知性错误。


## 6. Conclusion
## 6. 结论


In this paper, we propose a staged execution-feedback RL strategy to address the core challenges of long-horizon GUI automation. Our algorithm confronts the capability overload and lack of task progress awareness by shifting the training strategy. Instead of training a unified policy, our method thoroughly decouples high-level strategic planning from low-level precise execution. It leverages verifiable results to efficiently optimize high-level agents responsible for scheduling. Critically, our algorithm trains a dedicated State Tracker that resolves the agent's difficulties in long-horizon tasks via dynamic context compression and high-semantic state summarization. This algorithmic approach is instantiated in our CES framework. Extensive experiments on challenging long-horizon GUI benchmarks demonstrate that our trained high-level modules are a generalizable, plug-and-play solution that significantly enhances the long-horizon planning and state management capabilities of various Executor models. In the future, synergetic evolution and joint training of multi-agent system for GUI tasks may be a promising direction.
本文提出一种分阶段执行反馈强化学习策略，以解决长期 GUI 自动化的核心挑战。通过转变训练策略，我们的算法应对能力超载与任务进展感知缺失的问题。我们的方法并非训练一个统一策略，而是彻底将高层战略规划与低层精确执行解耦。它利用可验证的结果高效优化负责调度的高层代理。关键在于，我们的算法训练出一个专门的状态跟踪器，通过动态上下文压缩和高语义状态摘要来解决代理在长期任务中的困难。这一算法思想在我们的 CES 框架中得到实现。在面向具有挑战性的长期 GUI 基准测试中的广泛实验表明，我们训练的高层模块是一个可泛化、即插即用的解决方案，显著增强了各执行器模型在长期规划与状态管理方面的能力。未来，GUI 任务中多智能体系统的协同进化与联合训练可能是一个有前景的方向。


## References
## 参考文献


[1] Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents, 2025. 1
[1] Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents, 2025. 1


[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhao-hai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report, 2025. 7
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhao-hai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report, 2025. 7


[3] Yuxiang Chai, Siyuan Huang, Yazhe Niu, Han Xiao, Liang Liu, Dingyu Zhang, Shuai Ren, and Hongsheng Li. AMEX: Android Multi-annotation Expo Dataset for Mobile GUI Agents. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2138-2156, 2025. 7
[3] Yuxiang Chai, Siyuan Huang, Yazhe Niu, Han Xiao, Liang Liu, Dingyu Zhang, Shuai Ren, and Hongsheng Li. AMEX: Android Multi-annotation Expo Dataset for Mobile GUI Agents. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2138-2156, 2025. 7


[4] Pengzhou Cheng, Zheng Wu, Zongru Wu, Aston Zhang, Zhuosheng Zhang, and Gongshen Liu. OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents, 2025. 1, 2
[4] Pengzhou Cheng, Zheng Wu, Zongru Wu, Aston Zhang, Zhuosheng Zhang, and Gongshen Liu. OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents, 2025. 1, 2


[5] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-rong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Ming-ming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wan-jia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xi-aosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yi-fan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yun-fan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 2025. 2
[5] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-rong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Ming-ming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wan-jia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xi-aosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yi-fan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yun-fan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 2025. 2


[6] Jingru Fan, Yufan Dang, Jingyao Wu, Huatao Li, Runde Yang, Xiyuan Yang, Yuheng Wang, Zhong Zhang, Yaxi Lu, Yankai Lin, et al. Appcopilot: Toward general, accurate, long-horizon, and efficient mobile agent. arXiv preprint arXiv:2509.02444, 2025. 2
[6] Jingru Fan, Yufan Dang, Jingyao Wu, Huatao Li, Runde Yang, Xiyuan Yang, Yuheng Wang, Zhong Zhang, Yaxi Lu, Yankai Lin, et al. Appcopilot: Toward general, accurate, long-horizon, and efficient mobile agent. arXiv preprint arXiv:2509.02444, 2025. 2


[7] Xinzge Gao, Chuanrui Hu, Bin Chen, and Teng Li. Chain-of-Memory: Enhancing GUI Agents for Cross-Application Navigation, 2025. 3
[7] Xinzge Gao, Chuanrui Hu, Bin Chen, and Teng Li. Chain-of-Memory: Enhancing GUI Agents for Cross-Application Navigation, 2025. 3


[8] Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, Yue Wen, Jingya Dou, Fei Tang, Jinzhen Lin, Yulin Liu, Zhenlin Guo, Yichen Gong, Heng Jia, Changlong Gao, Yuan Guo, Yong Deng, Zhenyu Guo, Liang Chen, and Weiqiang Wang. UI-Venus Technical Report: Building High-performance UI Agents with RFT, 2025. 2
[8] Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, Yue Wen, Jingya Dou, Fei Tang, Jinzhen Lin, Yulin Liu, Zhenlin Guo, Yichen Gong, Heng Jia, Changlong Gao, Yuan Guo, Yong Deng, Zhenyu Guo, Liang Chen, and Weiqiang Wang. UI-Venus Technical Report: Building High-performance UI Agents with RFT, 2025. 2


[9] Yuan Guo, Tingjia Miao, Zheng Wu, Pengzhou Cheng, Ming Zhou, and Zhuosheng Zhang. Atomic-to-Compositional Generalization for Mobile Agents with A New Benchmark and Scheduling System, 2025. 3
[9] Yuan Guo, Tingjia Miao, Zheng Wu, Pengzhou Cheng, Ming Zhou, and Zhuosheng Zhang. Atomic-to-Compositional Generalization for Mobile Agents with A New Benchmark and Scheduling System, 2025. 3


[10] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. CogAgent: A Visual Language Model for GUI Agents, 2024. 2
[10] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. CogAgent: A Visual Language Model for GUI Agents, 2024. 2


[11] Yi Kong, Dianxi Shi, Guoli Yang, Zhang ke-di, Chen-lin Huang, Xiaopeng Li, and Songchang Jin. MapAgent: Trajectory-Constructed Memory-Augmented Planning for Mobile Task Automation, 2025. 3
[11] Yi Kong, Dianxi Shi, Guoli Yang, Zhang ke-di, Chen-lin Huang, Xiaopeng Li, and Songchang Jin. MapAgent: Trajectory-Constructed Memory-Augmented Planning for Mobile Task Automation, 2025. 3


[12] Hanyu Lai, Xiao Liu, Yanxiao Zhao, Han Xu, Hanchen Zhang, Bohao Jing, Yanyu Ren, Shuntian Yao, Yuxiao Dong, and Jie Tang. ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents, 2025. 1
[12] Hanyu Lai, Xiao Liu, Yanxiao Zhao, Han Xu, Hanchen Zhang, Bohao Jing, Yanyu Ren, Shuntian Yao, Yuxiao Dong, and Jie Tang. ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents, 2025. 1


[13] Jiahao Li and Kaer Huang. A Survey on GUI Agents with Foundation Models Enhanced by Reinforcement Learning, 2025. 2
[13] Jiahao Li and Kaer Huang. A Survey on GUI Agents with Foundation Models Enhanced by Reinforcement Learning, 2025. 2


[14] Ning Li, Qiqiang Lin, Zheng Wu, Xiaoyun Mo, Weiming Zhang, Yin Zhao, Xiangmou Qu, Jiamu Zhou, Jun Wang, Congmin Zheng, Yuanyi Song, Hongjiang Chen, Heyuan Huang, Jihong Wang, Jiaxin Yin, Jingwei Yu, Junwei Liao, Qiuying Peng, Xingyu Lou, Jun Wang, Weiwen Liu, Zhu-osheng Zhang, and Weinan Zhang. ColorAgent: Building A Robust, Personalized, and Interactive OS Agent, 2025. 1, 2
[14] Ning Li, Qiqiang Lin, Zheng Wu, Xiaoyun Mo, Weiming Zhang, Yin Zhao, Xiangmou Qu, Jiamu Zhou, Jun Wang, Congmin Zheng, Yuanyi Song, Hongjiang Chen, Heyuan Huang, Jihong Wang, Jiaxin Yin, Jingwei Yu, Junwei Liao, Qiuying Peng, Xingyu Lou, Jun Wang, Weiwen Liu, Zhu-osheng Zhang, and Weinan Zhang. ColorAgent: Building A Robust, Personalized, and Interactive OS Agent, 2025. 1, 2


[15] Ning Li, Xiangmou Qu, Jiamu Zhou, Jun Wang, Muning Wen, Kounianhua Du, Xingyu Lou, Qiuying Peng, Jun Wang, and Weinan Zhang. MobileUse: A GUI Agent with Hierarchical Reflection for Autonomous Mobile Operation, 2025. 3
[15] Ning Li, Xiangmou Qu, Jiamu Zhou, Jun Wang, Muning Wen, Kounianhua Du, Xingyu Lou, Qiuying Peng, Jun Wang, and Weinan Zhang. MobileUse: A GUI Agent with Hierarchical Reflection for Autonomous Mobile Operation, 2025. 3


[16] Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao, Xin-peng Liu, Jiayu Zhang, Zhenqiang Huang, Qianben Chen, Weichen Sun, Qiexiang Wang, Hongxuan Lu, Tianrui Qin, Chenghao Zhu, Yi Yao, Shuying Fan, Xiaowan Li, Tian-nan Wang, Pai Liu, King Zhu, He Zhu, Dingfeng Shi, Piaohong Wang, Yeyi Guan, Xiangru Tang, Minghao Liu, Yuchen Eleanor Jiang, Jian Yang, Jiaheng Liu, Ge Zhang, and Wangchunshu Zhou. Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL, 2025. 1
[16] Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao, Xin-peng Liu, Jiayu Zhang, Zhenqiang Huang, Qianben Chen, Weichen Sun, Qiexiang Wang, Hongxuan Lu, Tianrui Qin, Chenghao Zhu, Yi Yao, Shuying Fan, Xiaowan Li, Tian-nan Wang, Pai Liu, King Zhu, He Zhu, Dingfeng Shi, Piaohong Wang, Yeyi Guan, Xiangru Tang, Minghao Liu, Yuchen Eleanor Jiang, Jian Yang, Jiaheng Liu, Ge Zhang, and Wangchunshu Zhou. Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL, 2025. 1


[17] Yanda Li, Chi Zhang, Wenjia Jiang, Wanqi Yang, Bin Fu, Pei Cheng, Xin Chen, Ling Chen, and Yunchao Wei. AppAgent v2: Advanced Agent for Flexible Mobile Interactions, 2025. 3
[17] Yanda Li, Chi Zhang, Wenjia Jiang, Wanqi Yang, Bin Fu, Pei Cheng, Xin Chen, Ling Chen, and Yunchao Wei. AppAgent v2: Advanced Agent for Flexible Mobile Interactions, 2025. 3


[18] Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xi-aotian Han, Shengyu Zhang, Hongxia Yang, and Fei Wu. InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners, 2025. 2
[18] 刘宇航, 李朋翔, 谢聪恺, 胡翔科, 韩喜奥天, 张盛宇, 杨宏霞, 吴飞。InfiGUI-R1: 以反应 actor 向深思型推理者推进的多模态 GUI 代理, 2025. 2


[19] Quanfeng Lu, Wenqi Shao, Zitao Liu, Lingxiao Du, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, and Ping Luo. GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices, 2024. 7
<text>[19] 陆全风, 邵文琪, 刘子涛, 杜灵潇, 孟范青, 李博轩, 陈博通, 黄思源, 张凯鹏, 罗平. GUIOdyssey: 面向移动设备跨应用 GUI 导航的综合数据集, 2024. 7</text>


[20] Quanfeng Lu, Zhantao Ma, Shuai Zhong, Jin Wang, Dahai Yu, Michael K. Ng, and Ping Luo. SWIRL: A Staged Work-flow for Interleaved Reinforcement Learning in Mobile GUI Control, 2025. 3, 7
<text>[20] 陆全风, 马战涛, 钟帅, 王晋, 于大海, Michael K. Ng, 罗平. SWIRL: 移动 GUI 控制中交替强化学习的分阶段工作流, 2025. 3, 7</text>


[21] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Guanjing Xiong, and Hongsheng Li. UI-R1: Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning, 2025. 1, 2
<text>[21] 陆政西, 蔡宇翔, 郭亚璇, 尹溪, 劉良, 王浩, 肖瀚, 任帅, 熊冠景, 李宏生. UI-R1: 通过强化学习提升 GUI 代理的高效行动预测, 2025. 1, 2</text>


[22] Zhengxi Lu, Jiabo Ye, Fei Tang, Yongliang Shen, Haiyang Xu, Ziwei Zheng, Weiming Lu, Ming Yan, Fei Huang, Jun Xiao, et al. Ui-s1: Advancing gui automation via semi-online reinforcement learning. arXiv preprint arXiv:2509.11543, 2025.
<text>[22] 陆政西, 叶吉宝, 唐非, 沈永亮, 徐海洋, 郑子玮, 陆伟名, 闫明, 黄飞, 萧军, 等. Ui-s1: 通过半在线强化学习推进 GUI 自动化. arXiv 预印本 arXiv:2509.11543, 2025.</text>


[23] Run Luo, Lu Wang, Wanwei He, and Xiaobo Xia. GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents, 2025. 1, 2, 3, 7
<text>[23] 罗润, 王璐, 贺万伟, 夏晓博. GUI-R1: 一款通用的 R1 风格视觉-语言行动模型，面向 GUI 代理, 2025. 1, 2, 3, 7</text>


[24] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shi-jue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhao-jian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. UI-TARS: Pioneering Automated GUI Interaction with Native Agents, 2025. 1
<text>[24] 秦玉嘉, 葉一宁, 方俊杰, 王浩明, 梁诗豪, 田志若, 张君达, 李嘉豪, 李云欣, 黄诗觉, 钟万君, 李宽业, 杨佳乐, 廖宇, 林武勇, 刘兴翔, 江旭, 马谦礼, 李婧, 小周, 唐凯, 李创业, 郑炳怀, 何海, 杨海花, 刘海福, 林枫, 彭涛, 刘鑫, 石广. UI-TARS: 与本地代理共同推动自动化 GUI 交互, 2025. 1</text>


[25] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lilli-crap, and Oriana Riva. AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents, 2025. 1, 2
<text>[25] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lilli-crap, Oriana Riva. AndroidWorld: 面向自主代理的动态基准测试环境, 2025. 1, 2</text>


[26] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, 2024. 2, 5
<text>[26] 赵红烧, 王培毅, 朱琪浩, 徐润欣, 宋俊孝, 毕晓, 张浩伟, 张明川, 李耀康, 李雯, 郭大亚. DeepSeekMath: 推动开源语言模型在数学推理方面的极限, 2024. 2, 5</text>


[27] Yucheng Shi, Wenhao Yu, Zaitang Li, Yonglin Wang, Hong-ming Zhang, Ninghao Liu, Haitao Mi, and Dong Yu. MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment, 2025. 2, 3
<text>[27] 石宇成, 于文浩, 李在堂, 王永林, 张红明, 刘宁浩, 米海涛, 于东. MobileGUI-RL: 通过在线环境的强化学习推进移动 GUI 代理, 2025. 2, 3</text>


[28] Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Li-heng Chen, Zhoumianze Liu, et al. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5555-5579, 2025. 1, 2
<text>[28] 孙秋士, 程看志, 丁紫臣, 金川洋, 王宜安, 徐方志, 吴振宇, 贾成友, 陈力衡, 刘周民择, 等. Os-genesis: 通过逆任务综合自动化 GUI 代理轨迹构建. 计算语言学协会第63届年会论文集（卷1：长篇论文）, 页码 5555-5579, 2025. 1, 2</text>


[29] Zeyi Sun, Ziyu Liu, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Tong Wu, Dahua Lin, and Jiaqi Wang. SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience, 2025. 2, 3
<text>[29] 孙泽伊, 刘紫瑜, 伽阳枫, 曹宇航, 董晓逸, 吴彤, 林大华, 王嘉琪. SEAgent: 具备从经验中自主学习的自进化计算机使用代理, 2025. 2, 3</text>


[30] Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, et al. Gui-g': Gaussian reward modeling for gui grounding. arXiv preprint arXiv:2507.15846, 2025. 3
[30] 费Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, 等人。Gui-g': GUI  grounding 的高斯奖励建模。arXiv 预印本 arXiv:2507.15846，2025。 3


[31] Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, and Yueting Zhuang. A survey on (m)llm-based gui agents, 2025. 1
[31] 费 Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, 及 Yueting Zhuang。关于基于(m)llm 的 GUI 代理的综述，2025。 1


[32] Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, Wanjun Zhong, Yining Ye, Yujia Qin, Yuwen Xiong, Yuxin Song, Zhiyong Wu, Aoyan Li, Bo Li, Chen Dun, Chong Liu, Daoguang Zan, Fuxing Leng, Hanbin Wang, Hao Yu, Haobin Chen, Hongyi Guo, Jing Su, Jingjia Huang, Kai Shen, Kaiyu Shi, Lin Yan, Peiyao Zhao, Pengfei Liu, Qinghao Ye, Renjie Zheng, Shulin Xin, Wayne Xin Zhao, Wen Heng, Wenhao Huang, Wenqian Wang, Xiaobo Qin, Yi Lin, Youbin Wu, Zehui Chen, Zihao Wang, Baoquan Zhong, Xinchun Zhang, Xujing Li, Yuanfan Li, Zhongkai Zhao, Chengquan Jiang, Faming Wu, Haotian Zhou, Jinlin Pang, Li Han, Qi Liu, Qianli Ma, Siyao Liu, Songhua Cai, Wenqi Fu, Xin Liu, Yaohui Wang, Zhi Zhang, Bo Zhou, Guoliang Li, Jiajun Shi, Jiale Yang, Jie Tang, Li Li, Qi-hua Han, Taoran Lu, Woyu Lin, Xiaokang Tong, Xinyao Li, Yichi Zhang, Yu Miao, Zhengxuan Jiang, Zili Li, Ziyuan Zhao, Chenxin Li, Dehua Ma, Feng Lin, Ge Zhang, Haihua Yang, Hangyu Guo, Hongda Zhu, Jiaheng Liu, Junda Du, Kai Cai, Kuanye Li, Lichen Yuan, Meilan Han, Minchao Wang, Shuyue Guo, Tianhao Cheng, Xiaobo Ma, Xiaojun Xiao, Xiaolong Huang, Xinjie Chen, Yidi Du, Yilin Chen, Yiwen Wang, Zhaojian Li, Zhenzhu Yang, Zhiyuan Zeng, Chaolin Jin, Chen Li, Hao Chen, Haoli Chen, Jian Chen, Qinghao Zhao, and Guang Shi. UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning, 2025. 2
[32] Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, Wanjun Zhong, Yining Ye, Yujia Qin, Yuwen Xiong, Yuxin Song, Zhiyong Wu, Aoyan Li, Bo Li, Chen Dun, Chong Liu, Daoguang Zan, Fuxing Leng, Hanbin Wang, Hao Yu, Haobin Chen, Hongyi Guo, Jing Su, Jingjia Huang, Kai Shen, Kaiyu Shi, Lin Yan, Peiyao Zhao, Pengfei Liu, Qinghao Ye, Renjie Zheng, Shulin Xin, Wayne Xin Zhao, Wen Heng, Wenhao Huang, Wenqian Wang, Xiaobo Qin, Yi Lin, Youbin Wu, Zehui Chen, Zihao Wang, Baoquan Zhong, Xinchun Zhang, Xujing Li, Yuanfan Li, Zhongkai Zhao, Chengquan Jiang, Faming Wu, Haotian Zhou, Jinlin Pang, Li Han, Qi Liu, Qianli Ma, Siyao Liu, Songhua Cai, Wenqi Fu, Xin Liu, Yaohui Wang, Zhi Zhang, Bo Zhou, Guoliang Li, Jiajun Shi, Jiale Yang, Jie Tang, Li Li, Qi-hua Han, Taoran Lu, Woyu Lin, Xiaokang Tong, Xinyao Li, Yichi Zhang, Yu Miao, Zhengxuan Jiang, Zili Li, Ziyuan Zhao, Chenxin Li, Dehua Ma, Feng Lin, Ge Zhang, Haihua Yang, Hangyu Guo, Hongda Zhu, Jiaheng Liu, Junda Du, Kai Cai, Kuanye Li, Lichen Yuan, Meilan Han, Minchao Wang, Shuyue Guo, Tianhao Cheng, Xiaobo Ma, Xiaojun Xiao, Xiaolong Huang, Xinjie Chen, Yidi Du, Yilin Chen, Yiwen Wang, Zhaojian Li, Zhenzhu Yang, Zhiyuan Zeng, Chaolin Jin, Chen Li, Hao Chen, Haoli Chen, Jian Chen, Qinghao Zhao, 及 Guang Shi。UI-TARS-2 技术报告：以多轮强化学习推进 GUI 代理，2025。 2


[33] Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, Hyokun Yun, and Lihong Li. WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning, 2025. 1, 2
[33] Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, Hyokun Yun, 及 Lihong Li。WebAgent-R1：通过端到端多轮强化学习训练网络代理，2025。 1, 2


[34] Qinzhuo Wu, Pengzhi Gao, Wei Liu, and Jian Luan. Back-trackAgent: Enhancing GUI Agent with Error Detection and Backtracking Mechanism, 2025. 1
[34] Qinzhuo Wu, Pengzhi Gao, Wei Liu, 及 Jian Luan。Back-trackAgent：通过错误检测与回溯机制提升 GUI 代理，2025。 1


[35] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and Yu Qiao. OS-ATLAS: A Foundation Action Model for Generalist GUI Agents, 2024. 1, 2, 7
[35] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, 及 Yu Qiao。OS-ATLAS：面向通用 GUI 代理的基础动作模型，2024。 1, 2, 7


[36] Zheng Wu, Pengzhou Cheng, Zongru Wu, Lingzhong Dong, and Zhuosheng Zhang. Gem: Gaussian embedding modeling for out-of-distribution detection in gui agents. arXiv preprint arXiv:2505.12842, 2025. 3
[36] Zheng Wu, Pengzhou Cheng, Zongru Wu, Lingzhong Dong, 及 Zhuosheng Zhang。Gem：用于 GUI 代理的分布外检测的高斯嵌入建模。arXiv 预印本 arXiv:2505.12842，2025。 3


[37] Zheng Wu, Heyuan Huang, Xingyu Lou, Xiangmou Qu, Pengzhou Cheng, Zongru Wu, Weiwen Liu, Weinan Zhang, Jun Wang, Zhaoxiang Wang, and Zhuosheng Zhang. VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents, 2025. 1
[37] Zheng Wu, Heyuan Huang, Xingyu Lou, Xiangmou Qu, Pengzhou Cheng, Zongru Wu, Weiwen Liu, Weinan Zhang, Jun Wang, Zhaoxiang Wang, and Zhuosheng Zhang. VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents, 2025. 1


[38] Zhe Wu, Hongjin Lu, Junliang Xing, Changhao Zhang, Yin Zhu, Yuhao Yang, Yuheng Jing, Kai Li, Kun Shao, Jianye Hao, Jun Wang, and Yuanchun Shi. Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control, 2025. 3
[38] Zhe Wu, Hongjin Lu, Junliang Xing, Changhao Zhang, Yin Zhu, Yuhao Yang, Yuheng Jing, Kai Li, Kun Shao, Jianye Hao, Jun Wang, and Yuanchun Shi. Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control, 2025. 3


[39] Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, and Yuxiao Dong. MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents, 2025. 2
[39] Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, and Yuxiao Dong. MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents, 2025. 2


[40] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tian-bao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction, 2025. 1, 2
[40] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tian-bao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction, 2025. 1, 2


[41] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jia-long Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tian-hao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 Technical Report, 2025. 7
[41] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jia-long Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tian-hao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 Technical Report, 2025. 7


[42] Chenyu Yang, Shiqian Su, Shi Liu, Xuan Dong, Yue Yu, Weijie Su, Xuehui Wang, Zhaoyang Liu, Jinguo Zhu, Hao Li, Wenhai Wang, Yu Qiao, Xizhou Zhu, and Jifeng Dai. ZeroGUI: Automating Online GUI Learning at Zero Human Cost, 2025. 2
[42] Chenyu Yang, Shiqian Su, Shi Liu, Xuan Dong, Yue Yu, Weijie Su, Xuehui Wang, Zhaoyang Liu, Jinguo Zhu, Hao Li, Wenhai Wang, Yu Qiao, Xizhou Zhu, and Jifeng Dai. ZeroGUI: Automating Online GUI Learning at Zero Human Cost, 2025. 2


[43] Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. Mobile-agent-v3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025. 2, 3
[43] Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. Mobile-agent-v3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025. 2, 3


[44] Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, et al. Large language model-brained gui agents: A survey. arXiv preprint arXiv:2411.18279, 2024. 2
[44] Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, et al. Large language model-brained gui agents: A survey. arXiv preprint arXiv:2411.18279, 2024. 2


[45] Cheng Zhang, Erhu Feng, Xi Zhao, Yisheng Zhao, Wangbo Gong, Jiahui Sun, Dong Du, Zhichao Hua, Yubin Xia, and Haibo Chen. MobiAgent: A Systematic Framework for Customizable Mobile Agents, 2025. 3
[45] Cheng Zhang, Erhu Feng, Xi Zhao, Yisheng Zhao, Wangbo Gong, Jiahui Sun, Dong Du, Zhichao Hua, Yubin Xia, and Haibo Chen. MobiAgent: A Systematic Framework for Customizable Mobile Agents, 2025. 3


[46] Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the Zoo: Chain-of-Action-Thought for GUI Agents, 2024. 7
[46] Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the Zoo: Chain-of-Action-Thought for GUI Agents, 2024. 7


[47] Zijing Zhang, Ziyang Chen, Mingxiao Li, Zhaopeng Tu, and Xiaolong Li. RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents, 2025. 1, 2
[47] Zijing Zhang, Ziyang Chen, Mingxiao Li, Zhaopeng Tu, and Xiaolong Li. RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents, 2025. 1, 2


[48] Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, Jie Xie, Wei Zhou, Wang Xu, Yuanheng Zhang, Zhou Su, Zhongwu Zhai, Xiaoming Liu, Yudong Mei, Jianming Xu, Hongyan Tian, Chongyi Wang, Chi Chen, Yuan Yao, Zhiyuan Liu, and Maosong Sun. AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning, 2025. 1
[48] Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, Jie Xie, Wei Zhou, Wang Xu, Yuanheng Zhang, Zhou Su, Zhongwu Zhai, Xiaoming Liu, Yudong Mei, Jianming Xu, Hongyan Tian, Chongyi Wang, Chi Chen, Yuan Yao, Zhiyuan Liu, and Maosong Sun. AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning, 2025. 1


## A. Details of Staged Execution-Feedback Reinforcement Learning
## A. 详细信息：分阶段执行反馈增强学习


### A.1. Staged Optimization
### A.1. 分阶段优化


The core idea of our algorithm is to completely decouple the training of the Coordinator,denoted as ${\pi }_{c}$ ,and the State Tracker, denoted as ${\pi }_{s}$ . This is achieved through two independent optimization stages,each with a specifically defined objective function. We use parameters ${\theta }_{c}$ and ${\theta }_{s}$ to denote the trainable parameters of the Coordinator and State Tracker,respectively.
我们算法的核心思想是将协调器（记为 ${\pi }_{c}$）的训练与状态追踪器（记为 ${\pi }_{s}$）的训练完全解耦。通过两个独立的优化阶段实现，每个阶段都具有明确定义的目标函数。我们用参数 ${\theta }_{c}$ 与 ${\theta }_{s}$ 分别表示协调器和状态追踪器的可训练参数。


## Stage 1: Optimizing the Coordinator
## 第1阶段：优化协调器


The Coordinator’s optimization objective, $\mathcal{J}\left( {\theta }_{c}\right)$ ,is to maximize the following expectation:
协调器的优化目标 $\mathcal{J}\left( {\theta }_{c}\right)$，是使以下期望最大化：


$$
\mathcal{J}\left( {\theta }_{c}\right)  = {\mathbb{E}}_{\left( q,{s}^{t},{m}_{{q}_{t}}^{t - 1},{a}_{{q}_{t}}^{t}\right) } \sim  \mathcal{D}\left\lbrack  {\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}\min \left( {{\rho }_{i}^{t,\left( c\right) }\left( {\theta }_{c}\right) {\widehat{A}}_{i}^{t,\left( c\right) },\operatorname{clip}\left( {{\rho }_{i}^{t,\left( c\right) },1 \pm  \epsilon }\right) ,{\widehat{A}}_{i}^{t,\left( c\right) }}\right) }\right\rbrack   - \beta {D}_{KL}\left( {{\pi }_{{\theta }_{c}}\parallel {\pi }_{c,{ref}}}\right) . \tag{8}
$$



We firstly sample $N$ candidate atomic instructions $\mathcal{L} = \left\{  {{l}_{1}^{t},\ldots ,{l}_{N}^{t}}\right\}$ using the old Coordinator policy ${\pi }_{{\theta }_{c,\text{ old }}}$ :
我们首先使用旧的协调器策略 $N$ 对候选原子指令 $\mathcal{L} = \left\{  {{l}_{1}^{t},\ldots ,{l}_{N}^{t}}\right\}$ 进行采样 ${\pi }_{{\theta }_{c,\text{ old }}}$：


$$
{l}_{i}^{t} \sim  {\pi }_{{\theta }_{c,\text{ old }}}\left( {\cdot  \mid  q,{m}_{gt}^{t - 1},{s}^{t}}\right) , \tag{9}
$$



where $q,{m}_{gt}^{t - 1},{s}^{t}$ are sampled from the dataset $\mathcal{D}$ . For each candidate instruction ${l}_{i}^{t}$ ,obtain the executed action ${a}_{i}^{t}$ via the frozen Executor ${\pi }_{e}$ :
其中 $q,{m}_{gt}^{t - 1},{s}^{t}$ 从数据集 $\mathcal{D}$ 采样。对于每个候选指令 ${l}_{i}^{t}$，通过冻结的执行器 ${\pi }_{e}$ 获取执行动作 ${a}_{i}^{t}$：


$$
{a}_{i}^{t} = {\pi }_{e}\left( {{l}_{i}^{t},{s}^{t}}\right) . \tag{10}
$$



The ${a}_{i}^{t}$ are used to calculate reward as:
用于计算奖励的 ${a}_{i}^{t}$ 记为：


$$
{r}_{i}^{t,\left( c\right) } = {\alpha }_{1}{R}_{\text{ format }}\left( {l}_{i}^{t}\right)  + {\alpha }_{2}{R}_{\text{ executor }}\left( {{a}_{i}^{t},{a}_{gt}^{t}}\right) . \tag{11}
$$



Based on the rewards of the $N$ candidates $\left\{  {{r}_{1}^{\left( c\right) },\ldots ,{r}_{N}^{\left( c\right) }}\right\}$ ,calculate the relative advantage:
基于候选 $N$ 的奖励 $\left\{  {{r}_{1}^{\left( c\right) },\ldots ,{r}_{N}^{\left( c\right) }}\right\}$，计算相对优势：


$$
{\widehat{A}}_{i}^{t,\left( c\right) } = \frac{{r}_{i}^{t,\left( c\right) } - \operatorname{mean}\left( \left\{  {{r}_{1}^{t,\left( c\right) },{r}_{2}^{t,\left( c\right) },\ldots ,{r}_{N}^{t,\left( c\right) }}\right\}  \right) }{\operatorname{std}\left( \left\{  {{r}_{1}^{t,\left( c\right) },{r}_{2}^{t,\left( c\right) },\ldots ,{r}_{N}^{t,\left( c\right) }}\right\}  \right) }. \tag{12}
$$



At the same time,we calculate the probability ratio between the new Coordinator policy ${\pi }_{{\theta }_{c}}$ and the old policy ${\pi }_{{\theta }_{c,\text{ old }}}$ :
同时，我们计算新协调器策略 ${\pi }_{{\theta }_{c}}$ 与旧策略 ${\pi }_{{\theta }_{c,\text{ old }}}$ 之间的概率比：


$$
{\rho }_{i}^{t,\left( c\right) }\left( {\theta }_{c}\right)  = \frac{{\pi }_{{\theta }_{c}}\left( {{l}_{i}^{t} \mid  q,{m}_{gt}^{t - 1},{s}^{t}}\right) }{{\pi }_{{\theta }_{c,{old}}}\left( {{l}_{i}^{t} \mid  q,{m}_{gt}^{t - 1},{s}^{t}}\right) }. \tag{13}
$$



After calculating the above items,we can calculate the gradient of $\mathcal{J}\left( {\theta }_{c}\right)$ and update the strategy ${\theta }_{c}$ .
在计算上述各项后，我们可以计算 $\mathcal{J}\left( {\theta }_{c}\right)$ 的梯度并更新策略 ${\theta }_{c}$。


## Stage 2: Optimizing the State Tracker
## 第2阶段：优化状态追踪器


With ${\pi }_{c}$ and ${\pi }_{e}$ frozen,train the State Tracker ${\pi }_{s}$ to generate the state ${m}^{t}$ that guides the fixed ${\pi }_{c}$ to make an optimal decision
在 ${\pi }_{c}$ 与 ${\pi }_{e}$ 固定的情况下，训练状态追踪器 ${\pi }_{s}$ 以生成引导固定 ${\pi }_{c}$ 做出最优决策的状态 ${m}^{t}$


The State Tracker’s optimization objective, $\mathcal{J}\left( {\theta }_{s}\right)$ ,is to maximize the following expectation :
状态追踪器的优化目标 $\mathcal{J}\left( {\theta }_{s}\right)$，是使以下期望最大化：


$$
\mathcal{J}\left( {\theta }_{s}\right)  = {\mathbb{E}}_{\left( q,{s}^{t},{m}_{gt}^{t - 1},{m}_{gt}^{t - 1},{a}_{gt}^{t})\right. } \sim  \mathcal{D}\left\lbrack  {\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}\min \left( {{\rho }_{i}^{t,\left( s\right) }\left( {\theta }_{s}\right) {\widehat{A}}_{i}^{t,\left( s\right) },{\operatorname{clip}}_{i}^{t,\left( s\right) },\operatorname{clip}\left( {{\rho }_{i}^{t,\left( s\right) }\left( {\theta }_{s}\right) ,1 \pm  \epsilon }\right) {\widehat{A}}_{i}^{t,\left( s\right) }}\right) }\right\rbrack   - \beta {D}_{KL}\left( {{\pi }_{{\theta }_{s}}\parallel {\pi }_{s,{ref}}}\right) . \tag{14}
$$



We firstly sample $N$ candidate state summaries $\mathcal{M} = \left\{  {{m}_{1}^{t},\ldots ,{m}_{N}^{t}}\right\}$ using the old State Tracker policy ${\pi }_{{\theta }_{s,\text{ old }}}$ :
我们首先使用旧的状态跟踪策略 ${\pi }_{{\theta }_{s,\text{ old }}}$ 对 $\mathcal{M} = \left\{  {{m}_{1}^{t},\ldots ,{m}_{N}^{t}}\right\}$ 候选状态摘要 $N$ 进行采样：


$$
{m}_{i}^{t} \sim  {\pi }_{{\theta }_{s,{otd}}}\left( {\cdot  \mid  q,{m}_{gt}^{t - 1},{u}_{gt}^{t - 1}}\right) \tag{15}
$$



This begins a chained evaluation process. For each candidate state ${m}_{i}^{t}$ ,we first obtain the instruction ${l}_{i}^{t}$ via the frozen Coordinator ${\pi }_{c}$ :
这将开始一个链式评估过程。对于每个候选状态 ${m}_{i}^{t}$，我们先通过冻结的协调器 ${\pi }_{c}$ 获取指令 ${l}_{i}^{t}$：


$$
{l}_{i}^{t} = {\pi }_{c}\left( {q,{m}_{i}^{t},{s}^{t}}\right) . \tag{16}
$$



The instruction ${l}_{i}^{t}$ is passed to the frozen Executor ${\pi }_{e}$ to obtain the action ${a}_{i}^{t}$ :
指令 ${l}_{i}^{t}$ 被传递给冻结的执行者 ${\pi }_{e}$ 以获得动作 ${a}_{i}^{t}$：


$$
{a}_{i}^{t} = {\pi }_{e}\left( {{l}_{i}^{t},{s}^{t}}\right) . \tag{17}
$$



Finally, ${a}_{i}^{t}$ is used to calculate the reward ${r}_{i}^{t,\left( s\right) }$ for the state ${m}_{i}^{t}$ ,which includes a format reward for the state and the final executor reward :
最后，使用 ${a}_{i}^{t}$ 计算状态 ${m}_{i}^{t}$ 的奖励 ${r}_{i}^{t,\left( s\right) }$，其中包括对状态的格式化奖励以及最终执行者奖励：


$$
{r}_{i}^{t,\left( s\right) } = {\alpha }_{1}{R}_{\text{ format }}\left( {m}_{i}^{t}\right)  + {\alpha }_{2}{R}_{\text{ executor }}\left( {{a}_{i}^{t},{a}_{gt}^{t}}\right) . \tag{18}
$$



Based on the rewards of the $\mathrm{N}$ candidates $\left\{  {{r}_{1}^{t,\left( s\right) },\ldots ,{r}_{N}^{t,\left( s\right) }}\right\}$ ,we calculate the relative advantage :
基于候选项 $\left\{  {{r}_{1}^{t,\left( s\right) },\ldots ,{r}_{N}^{t,\left( s\right) }}\right\}$ 的奖励，$\mathrm{N}$，我们计算相对优势：


$$
{\widehat{A}}_{i}^{t,\left( s\right) } = \frac{{r}_{i}^{t,\left( s\right) } - \operatorname{mean}\left( \left\{  {{r}_{1}^{t,\left( s\right) },{r}_{2}^{t,\left( s\right) },\ldots ,{r}_{N}^{t,\left( s\right) }}\right\}  \right) }{\operatorname{std}\left( \left\{  {{r}_{1}^{t,\left( s\right) },{r}_{2}^{t,\left( s\right) },\ldots ,{r}_{N}^{t,\left( s\right) }}\right\}  \right) }. \tag{19}
$$



At the same time,we calculate the probability ratio between the new State Tracker policy ${\pi }_{{\theta }_{s}}$ and the old policy ${\pi }_{{\theta }_{s,\text{ old }}}$ :
同时，我们计算新状态跟踪策略 ${\pi }_{{\theta }_{s}}$ 与旧策略 ${\pi }_{{\theta }_{s,\text{ old }}}$ 之间的概率比：


$$
{\rho }_{i}^{t,\left( s\right) }\left( {\theta }_{s}\right)  = \frac{{\pi }_{{\theta }_{s}}\left( {{m}_{i}^{t} \mid  q,{m}_{gt}^{t - 1},{u}_{gt}^{t - 1}}\right) }{{\pi }_{{\theta }_{s,{old}}}\left( {{m}_{i}^{t} \mid  q,{m}_{gt}^{t - 1},{u}_{gt}^{t - 1}}\right) }. \tag{20}
$$



### A.2. Reward Function
### A.2. 奖励函数


The total reward $R$ is defined as a weighted sum of the format reward and the executor reward:
总奖励 $R$ 定义为格式化奖励与执行者奖励的加权和：


$$
R = {\alpha }_{1}{R}_{\text{ format }} + {\alpha }_{2}{R}_{\text{ executor }}. \tag{21}
$$



The ${R}_{\text{ format }}$ component is a binary reward that encourages the model to generate outputs in the specified format.
${R}_{\text{ format }}$ 成分是一种二元奖励，鼓励模型输出符合指定格式。


$$
{R}_{\text{ format }}\left( o\right)  = \mathbb{I}\left( {\text{ CheckFormat }\left( o\right) }\right) , \tag{22}
$$



where CheckFormat $\left( \cdot \right)$ is a function that returns 1 if the output $o$ strictly adheres to the ⟨think⟩ and ⟨answer⟩ tag format,and 0 otherwise.
其中 CheckFormat $\left( \cdot \right)$ 是一个函数，当输出 $o$ 严格符合 ⟨think⟩ 与 ⟨answer⟩ 标签格式时返回 1，否则返回 0。


${R}_{\text{ executor }}$ evaluates the contribution of the high-level agent's decision to the downstream Executor's actual execution result,based on the $i$ -th candidate’s action ${a}_{i}^{t}$ and the ground-truth action ${a}_{gt}^{t}$ . It is defined as a weighted sum of a type reward and a parameter reward:
${R}_{\text{ executor }}$ 评估高级代理决策对下游执行者实际执行结果的贡献，基于第 $i$ 候选的动作 ${a}_{i}^{t}$ 与真实动作 ${a}_{gt}^{t}$。它被定义为类型奖励与参数奖励的加权和：


$$
{R}_{\text{ executor }}\left( {{a}_{i}^{t},{a}_{gt}^{t}}\right)  = {\gamma }_{1}{R}_{\text{ type }}\left( {{a}_{i}^{t},{a}_{gt}^{t}}\right)  + {\gamma }_{2}{R}_{\text{ param }}\left( {{a}_{i}^{t},{a}_{gt}^{t}}\right) , \tag{23}
$$



where ${R}_{type}$ evaluates if the Executor’s predicted action type matches the ground-truth. Let ${a}_{i}^{t,{type}}$ be the predicted action type and ${a}_{gt}^{t,\text{ type }}$ be the ground-truth action type:
其中 ${R}_{type}$ 评估执行者预测的动作类型是否与真实动作类型匹配。设 ${a}_{i}^{t,{type}}$ 为预测的动作类型，${a}_{gt}^{t,\text{ type }}$ 为真实的动作类型：


$$
{R}_{\text{ type }}\left( {{a}_{i}^{t},{a}_{gt}^{t}}\right)  = \mathbb{I}\left( {{a}_{i}^{t,\text{ type }} = {a}_{gt}^{t,\text{ type }}}\right)  = \left\{  \begin{array}{ll} 1 & \text{ if }{a}_{i}^{t,\text{ type }} = {a}_{gt}^{t,\text{ type }} \\  0 & \text{ otherwise } \end{array}\right. \tag{24}
$$



where $\mathbb{I}\left( \cdot \right)$ is the indicator function.
其中 $\mathbb{I}\left( \cdot \right)$ 是指示函数。


Parameter Reward ${R}_{\text{ param }}$ evaluates the correctness of the predicted action parameters,consistent with the Success Rate (SR) metric. Its definition depends on the predicted action type ${a}_{i}^{t,\text{ type }}$ :
参数奖励 ${R}_{\text{ param }}$ 评估预测动作参数的正确性，与成功率（SR）指标一致。其定义取决于预测的动作类型 ${a}_{i}^{t,\text{ type }}$：


$$
{R}_{\text{ param }}\left( {{a}_{i}^{t},{a}_{gt}^{t}}\right)  = \left\{  \begin{array}{ll} \mathbb{I}\left( {{p}_{i}^{t,\text{ param }} \in  {p}_{gt}^{t,\text{ bbox }}}\right) & \text{ if }{a}_{i}^{t,\text{ type }} \in  \left\{  {\text{ ‘click’, }\text{ ‘long\_press’ }}\right\}  \\  \mathbb{I}\left( {{F1}\left( {{p}_{i}^{t,\text{ param }},{p}_{gt}^{t,\text{ text }}}\right)  > {0.5}}\right) & \text{ if }{a}_{i}^{t,\text{ type }} = \text{ ‘type’ } \\  \mathbb{I}\left( {{p}_{i}^{t,\text{ Param }} = {p}_{gt}^{t,\text{ dir }}}\right) & \text{ if }{a}_{i}^{t,\text{ type }} = \text{ ‘scroll’ } \\  \mathbb{I}\left( {{a}_{i}^{t,\text{ type }} = {a}_{gt}^{t,\text{ type }}}\right) & \text{ otherwise } \end{array}\right. \tag{25}
$$



where ${p}_{i}^{t,\text{ param }}$ is the parameters of the action, ${p}_{gt}^{t,\text{ bbox }}$ is the ground-truth bounding box, ${F1}\left( \cdot \right)$ is the F1 similarity function for text. For actions without parameters (e.g., 'press_home'), the reward is based on type correctness.
其中 ${p}_{i}^{t,\text{ param }}$ 是动作的参数，${p}_{gt}^{t,\text{ bbox }}$ 是真实框，${F1}\left( \cdot \right)$ 是文本的 F1 相似性函数。对于没有参数的动作（例如 'press_home'），奖励基于类型正确性。


## B. Experiment Details
## B. 实验细节


### B.1. Data Collection
### B.1. 数据收集


SFT For the warm-up SFT stage, we randomly selected 1K samples from the GUI-Odyssey training set due to its rich semantic annotations. This dataset includes: description: A description of the current screen; intention: The intent of the action to be taken, similar to a chain of thought; low_level_instruction: The specific low-level instruction for the step; context: A summary of actions taken before the current step. For the Coordinator, we constructed the ground truth as: <think>description.intention</think><answer>low_level_instruction</answer>. For the State Tracker, we used the context of the next step as its ground truth.
SFT 对于暖启动阶段的 SFT，我们从 GUI-Odyssey 训练集中随机选取 1K 条样本，因其具备丰富的语义标注。该数据集包括：description: 当前屏幕的描述；intention: 即将执行动作的意图，类似于思维过程的链条；low_level_instruction: 该步骤的具体低级指令；context: 对当前步骤前所执行动作的摘要。对于协调器，我们将地真实值构造为：<think>description.intention</think><answer>low_level_instruction</answer>。对于状态跟踪器，我们将下一步的上下文作为其地真实值。


Staged Execution-Feedback Reinforcement Learning For the RL stage, rich semantic annotations are no longer required; only an execution result for the reward signal is needed. We randomly selected 3K samples from the GUI-Odyssey training set to serve as the task pool. The ground truth was the fixed action and parameters for reward calculation.
Staged Execution-Feedback Reinforcement Learning 对于 RL 阶段，不再需要丰富的语义标注；只需要一个执行结果作为奖励信号。我们从 GUI-Odyssey 训练集中随机选取 3K 条样本作为任务池。地真实值为用于奖励计算的固定动作及参数。


### B.2. Experiment Settings
### B.2. 实验设置


SFT We trained the Coordinator (Qwen2.5-VL-7B) and State Tracker (Qwen3-4B) using the LLaMA-Factory framework. Both models were trained for one epoch with a learning rate of 5e-5. We used LoRA for fine-tuning, with a rank of 8 and alpha of 16.
SFT 我们使用 LLaMA-Factory 框架对协调器（Qwen2.5-VL-7B）和状态跟踪器（Qwen3-4B）进行训练。两者训练一个时期，学习率为 5e-5。我们采用 LoRA 进行微调，秩为 8，α 为 16。


Staged Execution-Feedback Reinforcement Learning We used a total of $8 \times  {80}\mathrm{G}$ GPUs. In Stage 1,4 GPUs were used to deploy the fixed Executor via vLLM for reward calculation. In Stage 2, 2 GPUs were used to deploy the Coordinator and Executor respectively. For the reward coefficients,we set ${\alpha }_{1},{\alpha }_{2}$ to 0.1 and 0.9,respectively,and ${\gamma }_{1},{\gamma }_{2}$ to 0.2,and 0.8, respectively. Detailed hyperparameters are provided in Table 4.
Staged Execution-Feedback Reinforcement Learning 我们总共使用 $8 \times  {80}\mathrm{G}$ 台 GPU。在阶段1，使用 4 张 GPU 通过 vLLM 部署固定执行器以进行奖励计算。在阶段2，分别使用 2 张 GPU 部署协调器与执行器。关于奖励系数，${\alpha }_{1},{\alpha }_{2}$ 分别设为 0.1 和 0.9，${\gamma }_{1},{\gamma }_{2}$ 分别设为 0.2 和 0.8。详细超参数见表 4。


Table 4. Hyperparameters for Staged RL Training
Table 4. Hyperparameters for Staged RL Training


<table><tr><td>HyperParameter</td><td>Coordinator</td><td>State Tracker</td></tr><tr><td>lr</td><td>1e-6</td><td>1e-6</td></tr><tr><td>epochs</td><td>10</td><td>5</td></tr><tr><td>optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>train_batch_size</td><td>32</td><td>32</td></tr><tr><td>clip_ratio</td><td>0.2</td><td>0.2</td></tr><tr><td>rollout_n</td><td>4</td><td>4</td></tr><tr><td>max_prompt_length</td><td>8192</td><td>8192</td></tr><tr><td>max_response_length</td><td>256</td><td>512</td></tr></table>
<table><tbody><tr><td>超参数</td><td>协调器</td><td>状态跟踪器</td></tr><tr><td>学习率</td><td>1e-6</td><td>1e-6</td></tr><tr><td>训练轮数</td><td>10</td><td>5</td></tr><tr><td>优化器</td><td>AdamW</td><td>AdamW</td></tr><tr><td>训练批大小</td><td>32</td><td>32</td></tr><tr><td>裁剪比</td><td>0.2</td><td>0.2</td></tr><tr><td> rollout_n</td><td>4</td><td>4</td></tr><tr><td>最大提示长度</td><td>8192</td><td>8192</td></tr><tr><td>最大响应长度</td><td>256</td><td>512</td></tr></tbody></table>


### B.3. Benchmarks
### B.3. 基准测试


AMEX is a large-scale, multi-level annotated Android GUI data set, designed to provide support for general mobile GUI control agents. It is committed to providing multi-level understanding of mobile GUI, including more than 104k high-resolution screenshots and about 3000 unique complex instructions, with an average of 12.8 steps per instruction. AITZ (Andriod in the Zoo) is a refined data set built for the Android GUI navigation field. It is the first time to connect the perception of screen layout/ui elements with the cognition of action decision-making. It contains 2504 unique instructions and 18643 screen action pairs, covering more than 70 Android Applications. GUI-Odyssey is a comprehensive data set that focuses on cross application GUI navigation on mobile devices. This data set aims to solve the complexity of cross application workflow. Tasks usually need to integrate multiple applications and transfer context and data between applications. It contains 8334 task tracks, with an average of 15.3 steps per track, which is the longest average step in the mobile GUI navigation dataset.
AMEX 是一个大规模、分层注释的 Android GUI 数据集，旨在为通用移动 GUI 控制代理提供支持。它致力于提供对移动 GUI 的多层次理解，包括超过 104k 张高分辨率截图和约 3000 条唯一的复杂指令，平均每条指令 12.8 步。AITZ（在动物园里的 Android）是为 Android GUI 导航领域打造的精炼数据集。它首次将屏幕布局/界面元素的感知与动作决策的认知连接起来。它包含 2504 条唯一指令和 18643 对屏幕动作，覆盖超过 70 个 Android 应用。GUI-Odyssey 是一个专注于移动设备跨应用 GUI 导航的综合数据集。该数据集旨在解决跨应用工作流的复杂性。任务通常需要集成多个应用并在应用之间传递上下文和数据。它包含 8334 条任务轨迹，平均每条轨迹 15.3 步，是移动 GUI 导航数据集中步数的最长平均值。


### B.4. Visualization
### B.4. 可视化


Figure 6 illustrates the progression of various variables throughout the training process.
图 6 展示了在训练过程中各种变量的演变。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_2e636f.jpg"/>



Figure 6. Coordinator (left) and State Tracker (right) training process.
图 6。协调器（左）与状态追踪器（右）的训练过程。


### B.5. Model Scaling Analysis
### B.5. 模型规模分析


To explore how model capacity influences the CES framework, we conducted a scaling analysis by retraining the Coordinator and State Tracker with varying parameter sizes.
为了探究模型容量如何影响 CES 框架，我们通过对协调器与状态追踪器在不同参数规模下重新训练，进行了规模分析。


For Coordinator, we investigated the trade-off between model size and planning capability by replacing the Qwen-2.5- VL-7B backbone with the compact Qwen-2.5-VL-3B. The results demonstrate a significant performance degradation across all metrics with the 3B model. This decline suggests that the smaller model lacks the capacity for complex instruction decomposition and fine-grained visual understanding required for the Coordinator role. Moreover, a weaker Coordinator creates a bottleneck that propagates to the second training stage, hindering the State Tracker's ability to learn effective state representations.
对于协调器，我们通过替换 Qwen-2.5-VL-7B 主干为紧凑型 Qwen-2.5-VL-3B，来研究模型大小与规划能力之间的权衡。结果在所有指标上都显示 3B 模型显著的性能下降。这一下降表明较小的模型缺乏完成协调器所需的复杂指令分解和细粒度视觉理解的容量。此外，较弱的协调器会在第二阶段训练中形成瓶颈，阻碍状态追踪器学习有效的状态表征。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_6033f7.jpg"/>



Figure 7. Model scaling analysis.
图 7。模型规模分析。


For State Tracker, we performed Qwen3 models with varying sizes (0.6B, 1.7B, 4B, and 8B) to determine the optimal configuration. As illustrated in the Figure 7, performance improves markedly as the model scales from 0.6B to 4B. However, increasing the size further to 8B yields only marginal gains. Therefore, we identify Qwen3-4B as the optimal choice for the State Tracker, offering a favorable balance between performance and computational efficiency.
对于状态追踪器，我们对 Qwen3 模型进行了 0.6B、1.7B、4B 与 8B 的不同尺寸尝试，以确定最佳配置。如图 7 所示，性能随着模型从 0.6B 增长到 4B 而显著提升。然而，进一步将尺寸扩大到 8B 时收益仅有边际。因此，我们将 Qwen3-4B 识别为状态追踪器的最佳选择，在性能与计算效率之间取得有利的平衡。


## C. Prompt
## C. 提示


## Coordinator
## 协调器


You are a GUI task coordinator Agent. Your role is to actively collaborate with the Executor Agent to complete complex GUI navigation tasks. Given a high-level task description and the current state of the task, your goal is to provide a clear and precise fine-grained instruction for the Executor Agent to help accomplish the task.
你是一名 GUI 任务协调代理。你的角色是积极与执行者代理协作，完成复杂的 GUI 导航任务。给定一个较高层次的任务描述和当前任务状态，你的目标是为执行者代理提供清晰而精确的细粒度指令，以帮助完成任务。


Screenshot: <image>
截图： <image>


High-level task: \{high_level_instruction\}
高级任务：\{high_level_instruction\}


Current_state: \{current_state\}
当前状态：\{current_state\}


First, think step-by-step. Put your reasoning within <think> tags. After your reasoning, provide the instruction within <answer> tags.
首先，按步就班地思考。在 <think> 标签中放入你的推理。推理完成后，在 <answer> 标签中给出指令。


## Executor
## 执行者


You are GUI executor Agent, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to execute the command '\{instruction\}'. Please provide the action to perform (enumerate from [complete, close, press home, click, press back, type, select, scroll, enter]), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.
你是 GUI 执行代理，一位具备推理能力的 GUI 助手。在这个 UI 截图 <image> 中，我希望你执行命令 '\{instruction\}'。请提供要执行的动作（从 [complete, close, press home, click, press back, type, select, scroll, enter] 枚举），若执行点击操作则提供鼠标光标移动到的位置（整数），以及完成该动作所需的任何输入文本。


Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows: <think>...</think> <answer>['action': enum[complete, close, press home, click, press back, type, select, scroll, enter], 'point': [x, y], 'input_text': 'no input text [default]']</answer>
以 <think> </think> 标签输出思考过程，最终答案以 <answer> </answer> 标签输出，格式如下：<think>...</think> <answer>['action': enum[complete, close, press home, click, press back, type, select, scroll, enter], 'point': [x, y], 'input_text': 'no input text [default]']</answer>


## State Tracker
## 状态追踪器


You are a GUI task State Tracker Agent. Your core function is dynamic context compression and state updating. You will receive the high-level user instruction, the previous task state (a summary of progress up to the last step), and the latest output of executor agent. Your task is to generate the new task state. This should be a high-semantic natural language summary that updates the previous state based on the latest action, maintaining a coherent record of the task's progress.
你是一名 GUI 任务状态追踪代理。你的核心功能是动态上下文压缩和状态更新。你将收到高级用户指令、上一次任务状态（到上一步的进度摘要）以及执行者代理的最新输出。你的任务是生成新的任务状态。这应当是一种高度语义化的自然语言摘要，在保持连贯记录的前提下，根据最新动作更新先前状态。


High-level user instruction: \{high_level_instruction\}
高级用户指令：\{high_level_instruction\}


Latest output of executor agent: \{executor_output\}
执行者代理的最新输出：\{executor_output\}


Previous Task State: \{current_state\}
先前的任务状态：\{current_state\}


These prompt are also used in +GPT-5 setting (in Table 1) and CES-P setting (in Table 2).
这些提示也在 +GPT-5 设置（表 1）和 CES-P 设置（表 2）中使用。


## D. Case Study
## D. 案例研究


Figure 8, 9, 10, 11, 12 show a scenario where our CES framework solves a complete long-horizon task.
图 8、9、10、11、12 展示了我们的 CES 框架解决完整长程任务的场景。


### D.1. Overall Analysis
### D.1. 总体分析


The trajectory demonstrates the CES framework's ability of solving long-horizon task, driven primarily by the State Tracker's pivotal role in bridging cross-application context gaps. For instance, after the Executor copied the meeting link in Zoom at Step 9, the agent transitioned to a completely different environment, Tumblr, in Step 12. Crucially, the State Tracker maintained a high-semantic summary: "The meeting information was copied ... to facilitate sharing", explicitly carrying this hidden state to Step 18 to guide the message sending, effectively preventing the context loss often seen in single-agent baselines during app switching. Complementing this memory retention, the Coordinator provided the necessary strategic backbone by decomposing the dense, high-level user instruction into a strictly ordered sequence: finalizing the meeting setup first, handling the information transfer second, and finally configuring the Clock in Step 28, ensuring the system maintained both logical continuity and procedural order throughout the complex workflow.
该轨迹展示了 CES 框架解决长程任务的能力，主要由状态追踪器在跨应用上下文缺口之间桥接所发挥的关键作用驱动。例如，在步骤 9 执行者复制 Zoom 的会议链接后，代理在步骤 12 转移到完全不同的环境 Tumblr。关键在于，状态追踪器维持了高度语义的摘要：“已复制会议信息以便共享”，并显式地将这一隐藏状态带至步骤 18 指导发送消息，有效避免了在应用切换时单代理基线常见的上下文丢失。配合这种记忆保留，协调者通过将高层用户指令分解为严格有序的序列提供了必要的战略支撑：先完成会议设置，再处理信息转移，最后在步骤 28 配置时钟，确保系统在复杂工作流中维持逻辑连续性与程序顺序。


### D.2. Failure Case Analysis
### D.2. 失败案例分析


Step 3 in Figure 8 shows a failure case. The true action should be TYPE: Business, but predicted action is SCROLL : DOWN. While the user's high-level instruction explicitly requested to "Organize a business meeting", the Coordinator failed to translate the adjective "business" into the specific atomic action of renaming the meeting topic. As observed in the Coordinator's output, the agent prioritized procedural configurations, such as "setting the meeting time" and "enabling security features", instead of noticing topic, which leads to faulty atomic instruction.
图 8 的步骤 3 展示了一个失败案例。真实动作应为 TYPE: Business，但预测动作为 SCROLL: DOWN。虽然用户的高层指令明确要求“组织一次商务会议”，协调者未能将形容词 “商务” 转换为将会议主题重命名为具体原子动作所需的操作。正如协调者输出所示，代理偏向于程序配置，例如“设定会议时间”和“启用安全特性”，而未注意到主题，这导致了错误的原子指令。


Step 12 in Figure 9 shows another failure case. In this step, the Coordinator correctly issued the atomic instruction "Open the messaging section". However, the Executor failed to ground this instruction to the correct pixel coordinates. A closer look at Executor's output shows that, it even correctly said "The messages icon is typically represented by a bubble with an smiling face in the bottom of the screen", but when he answer it, he chose another bubble without smiling face in the middle of screen.
步骤12在图9中展示了另一种失败情况。在此步骤，协调器正确发出原子指令“打开消息部分”。然而，执行器未能将该指令定位到正确的像素坐标。仔细观察执行器的输出，会发现它甚至正确地说“消息图标通常表示为屏幕底部带笑脸的气泡”，但在回答时，他选择了屏幕中间的另一个没有笑脸的气泡。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_9039f3.jpg"/>



Figure 8. An example of long-horizon task (Part 1 of 5).
图8。长时任务的一个示例（第1部分，共5部分）。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_87eb67.jpg"/>



Figure 9. An example of long-horizon task (Part 2 of 5).
图9。长时任务的一个示例（第2部分，共5部分）。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_bdcf19.jpg"/>



Figure 10. An example of long-horizon task (Part 3 of 5).
图10。长时任务的一个示例（第3部分，共5部分）。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_a9c50b.jpg"/>



Figure 11. An example of long-horizon task (Part 4 of 5).
图11。长时任务的一个示例（第4部分，共5部分）。


## User Instruction
## 用户指令


Organize a business meeting with saltyfunsweets using ZOOM Cloud Meetings, send out the meeting invitations via Tumblr, and set an alarm clock for the meeting time using the Clock app.
使用 ZOOM Cloud Meetings 组织一次与 saltyfunsweets 的商务会议，通过 Tumblr 发送会议邀请，并在会议时间使用 Clock 应用设置闹钟。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_26_f7507c.jpg"/>



Figure 12. An example of long-horizon task (Part 5 of 5).
图12。长时任务的一个示例（第5部分，共5部分）。