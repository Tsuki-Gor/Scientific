# Autonomous Integration and Improvement of Robotic Assembly using Skill Graph Representations
# 基于技能图表示的机器人装配自主集成与改进


Peiqi Yu ${}^{1, * }$ ,Philip Huang ${}^{2, * }$ ,Chaitanya Chawla ${}^{2, * }$ , Guanya Shi ${}^{2}$ , Jiaoyang Li ${}^{2}$ , and Changliu Liu ${}^{2}$
Peiqi Yu ${}^{1, * }$ ,Philip Huang ${}^{2, * }$ ,Chaitanya Chawla ${}^{2, * }$ , Guanya Shi ${}^{2}$ , Jiaoyang Li ${}^{2}$ , and Changliu Liu ${}^{2}$


Abstract-Robotic assembly systems traditionally require substantial manual engineering effort to integrate new tasks, adapt to new environments, and improve performance over time. This paper presents a framework for autonomous integration and continuous improvement of robotic assembly systems based on Skill Graph representations. A Skill Graph organizes robot capabilities as verb-based skills, explicitly linking semantic descriptions (verbs and nouns) with executable policies, pre-conditions, post-conditions, and evaluators. We show how Skill Graphs enable rapid system integration by supporting semantic-level planning over skills, while simultaneously grounding execution through well-defined interfaces to robot controllers and perception modules. After initial deployment, the same Skill Graph structure supports systematic data collection and closed-loop performance improvement, enabling iterative refinement of skills and their composition. We demonstrate how this approach unifies system configuration, execution, evaluation, and learning within a single representation, providing a scalable pathway toward adaptive and reusable robotic assembly systems. The code is at https://github.com/intelligent-control-lab/AIDF.
摘要——传统的机器人装配系统在集成新任务、适应新环境以及随时间提升性能方面，通常需要大量的人工工程投入。本文提出了一种基于技能图（Skill Graph）表示的机器人装配系统自主集成与持续改进框架。技能图将机器人能力组织为基于动词的技能，明确地将语义描述（动词和名词）与可执行策略、前置条件、后置条件及评估器关联起来。我们展示了技能图如何通过支持技能层面的语义规划，同时通过与机器人控制器和感知模块的定义良好的接口实现执行落地，从而实现快速的系统集成。部署后，相同的技能图结构支持系统化的数据收集和闭环性能改进，从而实现技能及其组合的迭代优化。我们证明了该方法如何在单一表示中统一了系统配置、执行、评估和学习，为实现自适应和可重用的机器人装配系统提供了一条可扩展的途径。代码地址：https://github.com/intelligent-control-lab/AIDF。


## I. INTRODUCTION
## I. 引言


Robotic assembly remains a central challenge in industrial automation due to task variability, environmental uncertainty, and the high cost of system integration. Even for well-structured assembly tasks, deploying a robotic system typically involves extensive manual configuration, custom scripting, and iterative tuning by domain experts. As a result, adapting an existing system to new products or improving performance over time is often slow and labor-intensive.
由于任务的多变性、环境的不确定性以及系统集成的高昂成本，机器人装配仍然是工业自动化中的核心挑战。即使对于结构良好的装配任务，部署机器人系统通常也涉及大量的人工配置、定制脚本编写以及领域专家的迭代调试。因此，将现有系统适配到新产品或随时间提升性能往往缓慢且耗费人力。


Recent advances in learning-based control and foundation models have shown promise in improving robot autonomy, particularly in perception, manipulation, and decision-making. However, their integration into real-world robotic assembly systems remains challenging for several fundamental reasons. First, many learning-based approaches struggle to reliably generalize to long-horizon, multi-stage assembly tasks, where small errors can accumulate over time and lead to task failure. Second, these methods often rely heavily on large-scale human demonstration data, which may be difficult to obtain for complex industrial setups and, more importantly, may be insufficient for high-precision manipulation tasks that require accuracy beyond human demonstration fidelity.
近年来，基于学习的控制和基础模型在提升机器人自主性方面展现出前景，特别是在感知、操作和决策领域。然而，由于几个根本原因，将其集成到现实世界的机器人装配系统中仍然具有挑战性。首先，许多基于学习的方法难以可靠地泛化到长时程、多阶段的装配任务中，微小的误差会随时间累积并导致任务失败。其次，这些方法通常严重依赖大规模的人类演示数据，这对于复杂的工业环境可能难以获取，更重要的是，对于需要超越人类演示精度的精密操作任务，这些数据可能不足。


On the other hand, for conventional non-learning-based systems, there remains a significant gap between high-level task semantics (e.g., "pick", "insert", "fasten") and low-level executable controllers. Task logic is frequently encoded in ad-hoc scripts that tightly couple symbolic intent with platform-specific control implementations, making it difficult to generalize to new tasks, products, or robot platforms. Finally, across both learning-based and classical approaches, there is a lack of explicit structure for systematically evaluating deployed skills, diagnosing failure modes, and driving principled performance improvement over time. As a result, system integration, adaptation, and optimization are often treated as separate and manual processes, rather than as part of a unified autonomous framework.
另一方面，对于传统的非学习型系统，高层任务语义（如“抓取”、“插入”、“紧固”）与底层可执行控制器之间仍存在巨大鸿沟。任务逻辑通常编码在临时脚本中，将符号意图与特定平台的控制实现紧密耦合，导致难以泛化到新任务、新产品或新机器人平台。最后，无论是基于学习的方法还是经典方法，都缺乏明确的结构来系统地评估已部署的技能、诊断故障模式并推动原则性的性能改进。因此，系统集成、适配和优化往往被视为独立且人工的过程，而非统一自主框架的一部分。


In this paper, we propose Skill Graph representations as a unifying abstraction for autonomous integration and improvement of robotic assembly systems. The core idea is to represent robot capabilities as a graph of skills, where each skill is defined by:
在本文中，我们提出将技能图表示作为机器人装配系统自主集成与改进的统一抽象。其核心思想是将机器人能力表示为技能图，其中每个技能由以下部分定义：


(1) a semantic description in terms of verbs and associated objects,
(1) 基于动词和相关对象的语义描述，


(2) executable implementations,
(2) 可执行的实现，


(3) pre-conditions and post-conditions, and
(3) 前置条件和后置条件，以及


(4) evaluation criteria.
(4) 评估准则。


As illustrated in Fig. 1, we show how this representation enables rapid system integration by allowing assembly tasks to be specified and planned at the semantic level, while preserving precise execution semantics. Furthermore, once deployed, the Skill Graph provides a natural scaffold for data collection and performance-driven improvement, enabling the system to iteratively refine individual skills and their compositions.
如图1所示，我们展示了该表示如何通过允许在语义层面指定和规划装配任务，同时保留精确的执行语义，从而实现快速的系统集成。此外，一旦部署，技能图为数据收集和性能驱动的改进提供了自然的支架，使系统能够迭代地优化单个技能及其组合。


## II. RELATED WORK
## II. 相关工作


## A. Task and Skill Representations in Robotics
## A. 机器人中的任务与技能表示


Robotic task execution has traditionally been structured using behavior trees, finite-state machines, hierarchical task networks (HTNs) [1], and options in reinforcement learning [2]-[4]. Classical symbolic planners such as PDDL [5] and Answer Set Programming [6] model tasks as discrete operators with logical preconditions and effects, enabling formal reasoning. Extensions including FastDownward [7] and HDDL [8] improve hierarchical expressivity and search efficiency.
传统的机器人任务执行通常使用行为树、有限状态机、分层任务网络（HTN）[1]以及强化学习中的选项（options）[2]-[4]来构建。经典的符号规划器如PDDL [5]和答案集编程（Answer Set Programming）[6]将任务建模为具有逻辑前置条件和效果的离散算子，从而实现形式化推理。包括FastDownward [7]和HDDL [8]在内的扩展进一步提高了分层表达能力和搜索效率。


However, purely symbolic approaches struggle in dynamic environments and in bridging symbolic plans with continuous geometric reasoning and control [9], [10]. While recent work incorporates online action instantiation, concurrency reasoning, and replanning [11], [12], the gap between symbolic specification and executable control remains a central challenge.
然而，纯符号化方法在动态环境中，以及在将符号化规划与连续几何推理和控制相结合时往往力不从心 [9], [10]。尽管近期研究已引入在线动作实例化、并发推理和重规划 [11], [12]，但符号化规范与可执行控制之间的鸿沟依然是一个核心挑战。


---



* Equal Contribution
* 同等贡献


${}^{1}$ Peiqi Yu is with the Department of Electrical and Computer Engineering, Carnegie Mellon University, peiqiy@andrew.cmu.edu
${}^{1}$ Peiqi Yu 任职于卡内基梅隆大学电气与计算机工程系，邮箱：peiqiy@andrew.cmu.edu


2 Philip Huang, Chaitanya Chawla, Guanya Shi, Jiaoyang Li, and Changliu Liu is with the Robotics Institute, Carnegie Mellon University, \{yizhouhu, cchawla, guanyas, jiaoyanl, cliu6\}@andrew.cmu.edu
2 Philip Huang, Chaitanya Chawla, Guanya Shi, Jiaoyang Li, 和 Changliu Liu 任职于卡内基梅隆大学机器人研究所，邮箱：\{yizhouhu, cchawla, guanyas, jiaoyanl, cliu6\}@andrew.cmu.edu


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_06_30_bbf8bd.jpg"/>



Fig. 1: Overview of the Skill Graph representation and its integration with a bimanual robotic LEGO assembly task.
图 1：技能图（Skill Graph）表示及其与双臂机器人乐高组装任务集成的概览。


## B. Learning-Based Manipulation and Assembly
## B. 基于学习的操纵与组装


Imitation learning and reinforcement learning have achieved strong performance in contact-rich manipulation and assembly [13]-[15]. Although robust within specific domains, these methods are often task- or platform-specific and lack standardized interfaces for reuse and composition.
模仿学习和强化学习在接触丰富的操纵与组装任务中已取得显著表现 [13]-[15]。尽管这些方法在特定领域内表现稳健，但通常针对特定任务或平台，缺乏用于复用和组合的标准化接口。


Vision-Language-Action (VLA) models [16]-[18] integrate perception and language for end-to-end action prediction. Yet many treat complex tasks as monolithic skills, limiting interpretability and compositional generalization when new tasks require recombining learned primitives [19], [20]. These limitations motivate structured and modular skill representations.
视觉-语言-动作（VLA）模型 [16]-[18] 将感知与语言集成，用于端到端的动作预测。然而，许多模型将复杂任务视为单一技能，限制了可解释性，且在需要重组已学习原语以完成新任务时，其组合泛化能力受限 [19], [20]。这些局限性促使人们寻求结构化和模块化的技能表示。


## C. Skill-Centric and Neuro-Symbolic Frameworks
## C. 以技能为中心及神经符号框架


Skill-centric and neuro-symbolic approaches decompose tasks into reusable executable components [21]-[23]. Control-driven abstractions ground skills in executable controllers rather than purely symbolic operators [24], and frameworks such as SCALE [25] demonstrate improved data efficiency and compositionality through semantically meaningful skills.
以技能为中心及神经符号方法将任务分解为可复用的可执行组件 [21]-[23]。控制驱动的抽象将技能扎根于可执行控制器而非纯符号算子 [24]，而诸如 SCALE [25] 等框架通过语义明确的技能展示了更高的数据效率和组合性。


Multi-agent extensions such as MA-PDDL [26] incorporate coordination and spatial constraints but often require manual integration between planning and low-level control, leaving semantic-execution consistency unresolved.
诸如 MA-PDDL [26] 等多智能体扩展引入了协调和空间约束，但通常需要在规划与底层控制之间进行人工集成，导致语义与执行的一致性问题尚未得到解决。


Skill Graph builds upon these works by defining atomic and meta skills as parameterized executors with grounded preconditions, postconditions, and evaluators. Unlike purely symbolic planners, it maintains direct links to controllers; unlike monolithic learning approaches, it supports modular composition and reuse. By integrating semantic structure with execution-aware interfaces, Skill Graph provides an ontology-driven architecture [27] that bridges symbolic reasoning and physical grounding for scalable multi-robot manipulation.
技能图（Skill Graph）在这些工作的基础上，将原子技能和元技能定义为带有扎根前置条件、后置条件和评估器的参数化执行器。与纯符号规划器不同，它保持了与控制器的直接链接；与单一的学习方法不同，它支持模块化组合与复用。通过将语义结构与执行感知接口相结合，技能图提供了一种本体驱动的架构 [27]，为可扩展的多机器人操纵架起了连接符号推理与物理扎根的桥梁。


## III. Skill Graph Representation
## III. 技能图表示


## A. Skill Definition
## A. 技能定义


A skill is defined as a tuple
技能被定义为一个元组


$$
\mathcal{S} = \left( {v,\mathcal{N},\pi ,{\mathcal{P}}_{\text{ pre }},{\mathcal{P}}_{\text{ post }},\mathcal{E}}\right) , \tag{1}
$$



where $v$ is a verb representing the action type (e.g.,pick, place, insert), $\mathcal{N}$ denotes applicable nouns including objects, robots,and environments, $\pi$ is an executable policy or controller, ${\mathcal{P}}_{\text{ pre }}$ and ${\mathcal{P}}_{\text{ post }}$ are pre-conditions and post-conditions, and $\mathcal{E}$ is a skill evaluator. In each skill,we characterize its dependencies on the applicable objects, robots, and environments, in order to use these as parameters to condition the policy, the pre- and post- conditions, and the evaluator. For example, a "pick" skill with a parallel gripper will have a different execution policy (which usually grasps on the side of the object) than a "pick" skill with a suction gripper (which usually sucks on top). But the semantic meaning of these low-level actions remains the same, as the objects are being attached to the robot end-effector, hence would be treated similarly in task description. The type of "pick" to choose will be decided by the planner using the skill evaluator $\mathcal{E}$ in the task setting; for big objects that are wider than the gripper width, the skill evaluator will favor the suction gripper.
其中 $v$ 是表示动作类型的动词（例如：抓取、放置、插入），$\mathcal{N}$ 表示适用的名词，包括物体、机器人和环境，$\pi$ 是可执行策略或控制器，${\mathcal{P}}_{\text{ pre }}$ 和 ${\mathcal{P}}_{\text{ post }}$ 分别是前置条件和后置条件，$\mathcal{E}$ 是技能评估器。在每个技能中，我们刻画了其对适用物体、机器人和环境的依赖关系，以便将这些作为参数来调节策略、前置/后置条件以及评估器。例如，使用平行夹爪的“抓取”技能与使用吸盘夹爪的“抓取”技能，其执行策略（前者通常抓取物体侧面，后者通常吸附顶部）会有所不同。但这些底层动作的语义含义保持不变，因为物体最终都附着在机器人末端执行器上，因此在任务描述中会被同等对待。选择哪种“抓取”方式将由规划器在任务设置中使用技能评估器 $\mathcal{E}$ 来决定；对于比夹爪宽度更大的物体，技能评估器将倾向于选择吸盘夹爪。


Before instantiating the nouns, the skill is still in its abstract form. Once the nouns are determined, e.g., which robot to use, which object to manipulate, in which environment, the skill is concretized and executable. The concretization happens at the task configuration stage, and then a skill graph can be formed to ease planning.
在实例化名词之前，技能仍处于抽象形式。一旦确定了名词，例如使用哪台机器人、操作哪个物体、在何种环境下，技能便会具体化并可执行。具体化过程发生在任务配置阶段，随后即可构建技能图以简化规划。


## B. Skill Graph Structure
## B. 技能图结构


In a given task setup,for a given set of nouns $\mathcal{N}$ ,e.g., two robot arms with customized end-effectors, LEGO bricks (as objects to be manipulated), a workspace with LEGO baseboard, we could extract an organized skill graph. The skill graph contains a directed graph $\mathcal{G} = \left( {\mathcal{V},\mathcal{L}}\right)$ ,where each node $s \in  \mathcal{V}$ corresponds to a skill defined in Sec. III-A,and each directed edge $\left( {{s}_{i} \rightarrow  {s}_{j}}\right)  \in  \mathcal{L}$ encodes feasible transitions between two skills. Transitions in the Skill Graph are governed by the pre-conditions and post-conditions associated with each skill. All conditions are defined over a shared state space $\mathcal{Z}$ as the states of the items in $\mathcal{N}$ . We denote the current state as $z \in  \mathcal{Z}$ ,where $z$ can be measured. A directed edge $\left( {{s}_{i} \rightarrow  {s}_{j}}\right)$ is feasible if ${\mathcal{P}}_{\text{ post }}\left( {{s}_{i},z}\right)  \Rightarrow  {\mathcal{P}}_{\text{ pre }}\left( {{s}_{j},z}\right)$ ,meaning the current execution of skill ${s}_{i}$ establishes the conditions required to execute skill ${s}_{j}$ . Under this formulation,planning over the Skill Graph corresponds to searching for a path of skills, while the execution of the task corresponds to chaining the executables of the skills and ensuring the transitions across skills meet the pre- and post- condition requirements. These aspects will be discussed in the following section.
在给定的任务设置中，对于一组给定的名词 $\mathcal{N}$，例如带有定制末端执行器的两只机械臂、乐高积木（作为被操作对象）、带有乐高底板的工作区，我们可以提取出一个有组织的技能图。该技能图包含一个有向图 $\mathcal{G} = \left( {\mathcal{V},\mathcal{L}}\right)$，其中每个节点 $s \in  \mathcal{V}$ 对应于第 III-A 节中定义的技能，每条有向边 $\left( {{s}_{i} \rightarrow  {s}_{j}}\right)  \in  \mathcal{L}$ 编码了两个技能之间可行的转换。技能图中的转换由与每个技能相关联的前置条件和后置条件控制。所有条件均定义在共享状态空间 $\mathcal{Z}$ 上，即 $\mathcal{N}$ 中各项的状态。我们将当前状态表示为 $z \in  \mathcal{Z}$，其中 $z$ 是可测量的。如果 ${\mathcal{P}}_{\text{ post }}\left( {{s}_{i},z}\right)  \Rightarrow  {\mathcal{P}}_{\text{ pre }}\left( {{s}_{j},z}\right)$ 成立，则有向边 $\left( {{s}_{i} \rightarrow  {s}_{j}}\right)$ 是可行的，这意味着技能 ${s}_{i}$ 的当前执行建立了执行技能 ${s}_{j}$ 所需的条件。根据这一表述，在技能图上进行规划相当于搜索一条技能路径，而任务的执行则对应于将技能的可执行程序串联起来，并确保跨技能的转换满足前置和后置条件要求。这些方面将在下一节中讨论。


## C. Atomic and Meta Skills
## C. 原子技能与元技能


Skills are composable. For commonly used skill composition, we can group them together to form a new skill. Here we introduce two concepts, atomic skills and meta skills.
技能是可组合的。对于常用的技能组合，我们可以将其归类以形成新的技能。在此，我们引入两个概念：原子技能和元技能。


Atomic Skills correspond to low-level, directly executable actions, such as "Pick", "Place", "Transit", or "Detect". We define a skill as a parameterized function that either changes the world state or updates the system's belief about the world state; as a result, both motion skills and perception skills are naturally included in this formulation. In the proposed framework, atomic skills are obtained by instantiating the skill definition $s \in  \mathcal{S}$ with concrete executable policies $\pi$ and evaluators $\mathcal{E}$ . These policies may be implemented using parameterized motion primitives, wrappers around existing robot controllers, task-specific procedural control logic, or learned black-box policies.
原子技能对应于底层的、可直接执行的动作，例如“抓取”、“放置”、“移动”或“检测”。我们将技能定义为一个参数化函数，它要么改变世界状态，要么更新系统对世界状态的认知；因此，运动技能和感知技能都被自然地包含在此表述中。在所提出的框架中，原子技能是通过将技能定义 $s \in  \mathcal{S}$ 与具体的可执行策略 $\pi$ 和评估器 $\mathcal{E}$ 实例化而获得的。这些策略可以通过参数化运动基元、现有机器人控制器的封装、特定任务的程序控制逻辑或学习到的黑盒策略来实现。


Meta Skills represent higher-level capabilities formed by composing multiple skills into a single reusable abstraction. A meta skill is defined by grouping a sequence or subgraph of skills within the Skill Graph and associating the composition with its own pre-conditions and post-conditions. Internally, a meta skill expands into its constituent atomic skills during execution, while externally it behaves as a single skill node within the Skill Graph. This uniform abstraction allows meta skills to be planned, executed, and evaluated in the same manner as atomic skills. During execution, each atomic skill within a meta skill is instantiated with task-specific parameters.
元技能代表了通过将多个技能组合成单一可重用抽象而形成的高级能力。元技能的定义方式是：将技能图中的一系列技能或子图进行分组，并为该组合关联其自身的前置和后置条件。在内部，元技能在执行时会展开为其组成的原子技能，而在外部，它表现为技能图中的单个技能节点。这种统一的抽象使得元技能能够以与原子技能相同的方式进行规划、执行和评估。在执行过程中，元技能内的每个原子技能都会使用特定任务的参数进行实例化。


## IV. Autonomous System Integration Using SKILL GRAPHS
## IV. 使用技能图的自主系统集成


The Skill Graph serves as a central abstraction for integrating planning, execution, and task specification in robotic assembly. By separating semantic definitions from platform-specific implementations, it enables semantic-level planning (Sec. IV-A), execution grounding (Sec. IV-B) and intuitive task specification (Sec. IV-C).
技能图作为一种核心抽象，用于集成机器人装配中的规划、执行和任务规范。通过将语义定义与平台特定的实现分离开来，它实现了语义级规划（第 IV-A 节）、执行落地（第 IV-B 节）和直观的任务规范（第 IV-C 节）。


## A. Semantic-Level Planning
## A. 语义级规划


For robotic assembly, we use a planner to bridge the Skill Graph representation to safe and efficient execution on real robot systems. The key idea is to chain available skills according to their pre- and post-conditions and ground them to specific robots and objects based on task specifications and available resources.
对于机器人装配，我们使用规划器将技能图表示与真实机器人系统上的安全高效执行连接起来。其核心思想是根据技能的前置和后置条件将可用技能串联起来，并根据任务规范和可用资源将其落实到具体的机器人和物体上。


We assume the assembly task is specified as a sequence $\mathcal{A} = \left\{  {{a}_{1},{a}_{2},\ldots ,{a}_{A}}\right\}$ ,and the environment contains $n$ robots and a set of objects $\mathcal{O} = \left\{  {{o}_{1},{o}_{2},\ldots ,{o}_{o}}\right\}$ . Each task ${a}_{i}$ corresponds semantically to a meta skill ${s}_{\text{ meta }}$ designed to complete that step (e.g., "assemble" a brick, which then corresponds to a sequence of atomic skills: "detect", "pick", "transit", "place"). The planning goal is to ground the assembly sequence to a set of meta skills ${s}_{1},{s}_{2},\ldots ,{s}_{A}$ by assigning a subject (robot) and an object (to manipulate) to each skill.
我们假设装配任务被指定为一个序列 $\mathcal{A} = \left\{  {{a}_{1},{a}_{2},\ldots ,{a}_{A}}\right\}$，且环境包含 $n$ 台机器人和一组对象 $\mathcal{O} = \left\{  {{o}_{1},{o}_{2},\ldots ,{o}_{o}}\right\}$。每个任务 ${a}_{i}$ 在语义上对应一个旨在完成该步骤的元技能 ${s}_{\text{ meta }}$（例如，“组装”一块积木，这对应于一系列原子技能：“检测”、“抓取”、“移动”、“放置”）。规划的目标是通过为每个技能分配主体（机器人）和客体（被操作对象），将装配序列映射到一组元技能 ${s}_{1},{s}_{2},\ldots ,{s}_{A}$ 上。


Assignments are constrained by the pre-conditions of each meta skill and its associated atomic skills. Skill preconditions may be semantic (e.g., gripper compatibility) or geometric (e.g., kinematic feasibility and reachability). For example, one pre-condition for a pick skill is that there is no object on the robot end-effector and the object to pick is within the reach of the robot. During planning, the state used to evaluate pre- and post- conditions is obtained from simulation.
分配过程受到每个元技能及其关联原子技能的前置条件约束。技能前置条件可以是语义上的（例如，夹爪兼容性）或几何上的（例如，运动学可行性和可达性）。例如，抓取技能的一个前置条件是机器人末端执行器上没有物体，且待抓取物体在机器人的可达范围内。在规划过程中，用于评估前置和后置条件的状态是从仿真中获取的。


We employ a best-first search planner. Each search node represents a partial grounded skill sequence with the evaluator $\mathcal{E}$ assigning cost from simulated execution time. At each step, we pop the top node, enumerate feasible skills that satisfy preconditions for the next assembly step, and insert the resulting nodes back into the queue. The process continues until a fully feasible skill sequence is obtained.
我们采用最佳优先搜索规划器。每个搜索节点代表一个部分已映射的技能序列，评估器 $\mathcal{E}$ 根据仿真执行时间分配代价。在每一步中，我们弹出最优节点，枚举满足下一步装配步骤前置条件的可行技能，并将生成的节点插回队列。该过程持续进行，直到获得一个完全可行的技能序列。


## B. Execution-Level Grounding
## B. 执行层面的映射


Executing a planned skill sequence on real robots is challenging due to uncertainties such as controller stochasticity, sensor noise, and environmental variation. During execution, each skill must satisfy its pre- and post-conditions ${\mathcal{P}}_{\text{ pre }}$ and ${\mathcal{P}}_{\text{ post }}$ with respect to the true system states while executing policy $\pi$ . Robot-specific parameters (e.g.,speed or force limits) can be adjusted to ensure safe and efficient behavior.
由于控制器随机性、传感器噪声和环境变化等不确定性，在真实机器人上执行规划好的技能序列具有挑战性。在执行过程中，每个技能必须在执行策略 $\pi$ 的同时，针对真实系统状态满足其前置条件 ${\mathcal{P}}_{\text{ pre }}$ 和后置条件 ${\mathcal{P}}_{\text{ post }}$。机器人特定的参数（如速度或力限制）可以进行调整，以确保安全高效的行为。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_06_30_053d90.jpg"/>



Fig. 2: Intuitive Task Specification via Video Extraction. The pipeline transforms a raw human demonstration video into a structured Skill Graph.
图 2：通过视频提取实现直观的任务规范。该流水线将原始人类演示视频转换为结构化的技能图。


To support multi-robot assembly, we provide two execution modes: sequential execution and asynchronous execution based on APEX-MR [28].
为了支持多机器人装配，我们提供了两种执行模式：顺序执行和基于 APEX-MR [28] 的异步执行。


In the sequential execution mode, each meta skill is expanded into atomic skills and executed sequentially, with only one robot active at a time. Intermediate states are propagated via post-conditions to subsequent skills. This mode is safe and simple but inefficient, as no skills are executed concurrently.
在顺序执行模式下，每个元技能被展开为原子技能并按顺序执行，同一时间仅有一台机器人处于活动状态。中间状态通过后置条件传递给后续技能。这种模式安全且简单，但效率低下，因为没有任何技能是并发执行的。


In the asynchronous execution (APEX-MR) mode, a sequential task and motion plan is postprocessed into a multi-modal Temporal Plan Graph (TPG) [29] for parallel multi-robot execution. A TPG (e.g., Fig. 1 Part III.) is a directed acyclic graph where nodes represent robot actions and edges encode precedence constraints. Conflicting actions are ordered explicitly to avoid collisions, and a node can execute only after all incoming dependencies are satisfied. In practice, the policy for each robot skill is parameterized as a trajectory, which may be split into smaller segments corresponding to graph nodes. Intra-robot edges connect consecutive segments, while inter-robot edges enforce collision and task constraints. During execution, a central server dispatches executable nodes to each robot's action queue and updates dependencies upon completion. Because precedence constraints are encoded in the TPG, concurrent execution remains safe and significantly more efficient than sequential execution.
在异步执行（APEX-MR）模式下，顺序任务和运动规划被后处理为用于并行多机器人执行的多模态时间规划图（TPG）[29]。TPG（例如图 1 第三部分）是一个有向无环图，其中节点代表机器人动作，边编码优先级约束。冲突动作被显式排序以避免碰撞，且节点仅在所有输入依赖项满足后才能执行。在实践中，每个机器人技能的策略被参数化为轨迹，该轨迹可拆分为对应于图节点的较小片段。机器人内部的边连接连续片段，而机器人之间的边强制执行碰撞和任务约束。执行期间，中央服务器将可执行节点分发到每个机器人的动作队列，并在完成后更新依赖关系。由于优先级约束被编码在 TPG 中，并发执行既保持了安全性，又比顺序执行高效得多。


## C. Intuitive Task Specification via Video Extraction
## C. 通过视频提取实现直观的任务规范


To reduce manual engineering effort in defining task sequences in $\mathcal{A}$ ,we introduce a video-based pipeline for zero-shot skill sequence extraction from human demonstrations, shown in Fig. 2 Using a Vision-Language Model (Google Gemini API [30]), raw demonstration videos are parsed according to Skill Graph representation $\mathcal{G}$ in Sec.
为了减少在 $\mathcal{A}$ 中定义任务序列的人工工程工作量，我们引入了一种基于视频的流水线，用于从人类演示中进行零样本技能序列提取，如图 2 所示。利用视觉-语言模型（Google Gemini API [30]），原始演示视频将根据第 $\mathcal{G}$ 节中的技能图表示进行解析。


III The pipeline consists of the following stages. 1. Video Preprocessing and Token Optimization: The video is temporally downsampled to ${10}\mathrm{\;{Hz}}$ and cropped to the relevant workspace to reduce token usage and remove visual or audio noise while preserving atomic actions. 2. Prompting with Relative Spatial Reasoning: To improve reliability and reduce hallucinations, we employ a structured prompting strategy. A one-shot multimodal example (image-JSON pair) calibrates the mapping between visual cues and semantic action labels. The initial inventory is injected into the prompt to constrain predictions to physically present objects. Since VLMs cannot reliably regress 3D coordinates, spatial goals are expressed as relative semantic constraints (e.g., "Aligned Center" vs. "Shifted Left"), which are later grounded into physical poses during execution. 3. Scene Initialization and State Grounding: The initial video frames are used to estimate the resource inventory and workspace configuration, defining the initial state ${z}_{0}$ . The extracted plan is validated against this state to prevent hallucinated actions and ensure resource consistency. 4. Schema-Constrained Generation: The VLM output is restricted to a typed schema defining valid meta skills (e.g., PickPlace, PickPlacewSupport) and object categories according to the skill graph. Structured JSON output ensures direct mapping to the task sequence $\mathcal{A}$ . Results: Fig. 4 shows the results of our pipeline on a human video, feeding the output assembly task to the task-planner, and demonstrating the output trajectory on the real hardware. Note how our pipeline accurately detects the alignment shift in the top two bricks of the structure. For more results, please refer to our supplementary material.
III 该流程包含以下阶段：1. 视频预处理与 Token 优化：将视频在时间维度下采样至 ${10}\mathrm{\;{Hz}}$，并裁剪至相关工作空间，以减少 Token 使用量并去除视觉或音频噪声，同时保留原子动作。2. 基于相对空间推理的提示：为提高可靠性并减少幻觉，我们采用结构化提示策略。通过单样本多模态示例（图像-JSON 对）校准视觉线索与语义动作标签之间的映射。将初始清单注入提示中，以限制预测范围为物理存在的物体。由于 VLM 无法可靠地回归 3D 坐标，空间目标被表示为相对语义约束（例如“中心对齐”与“向左偏移”），这些约束在执行过程中会被映射为物理位姿。3. 场景初始化与状态接地：利用初始视频帧估计资源清单与工作空间配置，定义初始状态 ${z}_{0}$。提取的计划将针对该状态进行验证，以防止产生幻觉动作并确保资源一致性。4. 模式约束生成：VLM 的输出被限制在定义的类型模式内，该模式根据技能图定义了有效的元技能（如 PickPlace、PickPlacewSupport）和物体类别。结构化的 JSON 输出确保了与任务序列 $\mathcal{A}$ 的直接映射。结果：图 4 展示了我们流程在人类视频上的结果，将输出的装配任务输入任务规划器，并在真实硬件上演示了输出轨迹。请注意我们的流程如何准确检测到结构顶部两个积木的对齐偏移。更多结果请参考我们的补充材料。


## V. Data Collection and Performance Evaluation
## V. 数据收集与性能评估


The Skill Graph not only enables structured execution, but also provides a foundation for systematic data collection and iterative improvement. Each execution generates structured logs that can be reused for analysis, evaluation, and refinement.
技能图不仅支持结构化执行，还为系统化数据收集与迭代改进奠定了基础。每次执行都会生成可用于分析、评估和优化的结构化日志。


## A. Skill-Level Logging
## A. 技能级日志记录


We pair the Skill Graph with a digital data backbone that records both time-series signals and structured metadata. During execution, each skill invocation produces robot state trajectories, sensor observations, controller inputs, and evaluations of pre- and post-conditions. In addition to streaming time-series data, we store contextual metadata as JSON objects, including skill names, associated entities (robots, objects, tools), policy parameters, and task specifications. This indexed and structured representation enables full reconstruction of skill executions in a digital twin and supports offline analysis at the skill level (Fig. 3). More importantly, these naturally generate structured and labeled data at scale. Such data can be readily leveraged to train end-to-end vision-language-action policies, which we leave for future work.
我们将技能图与数字数据主干相结合，记录时间序列信号和结构化元数据。在执行过程中，每次技能调用都会产生机器人状态轨迹、传感器观测值、控制器输入以及前置和后置条件的评估结果。除了流式传输时间序列数据外，我们还将上下文元数据存储为 JSON 对象，包括技能名称、关联实体（机器人、物体、工具）、策略参数和任务规范。这种索引化的结构化表示支持在数字孪生中完整重构技能执行，并支持技能级的离线分析（图 3）。更重要的是，这些数据能自然地大规模生成结构化标注数据。此类数据可直接用于训练端到端视觉-语言-动作策略，我们将此留待未来工作。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_06_30_5881c8.jpg"/>



Fig. 3: Skill-level trajectory visualization. The upper panels present representative execution snapshots for different skills (Pick, Place Up, Place Down, Support Top/Bottom, Handover, and Transit) performed by the two Yaskawa GP4 robots (named as "DESTROYER" and "ARCHITECT"). The lower plots illustrate the temporal evolution of two robots' joint position and force over the full task duration separated by different skill labels, illustrating motion dynamics throughout the entire manipulation sequence with skill labels.
图 3：技能级轨迹可视化。上图展示了由两台安川 GP4 机器人（分别命名为“DESTROYER”和“ARCHITECT”）执行不同技能（抓取、向上放置、向下放置、顶部/底部支撑、移交和转运）的代表性执行快照。下图展示了在整个任务期间，两台机器人的关节位置和力随时间的变化，并按不同技能标签进行了区分，体现了整个操作序列中的运动动力学。


## B. Evaluators and Failure Modes
## B. 评估器与故障模式


The structured logs produced by the Skill Graph provide labeled execution outcomes for each atomic skill. Using these logs,we update skill evaluators $\mathcal{E}$ that estimate the likelihood of successful execution under specific environmental conditions. Common failure modes could range from failure to grasp or place a part, an unintended collision between robots and their environment, or a catastrophic structural failure due to accumulated errors from suboptimal assembly steps. We train new perception skills using real execution data and later integrate them into the Skill Graph atomic skills or pre/post-conditions to guide skill selection during planning and to monitor execution outcomes online. In this way, the system uses experience collected during deployment to continuously improve the reliability of its skills.
技能图生成的结构化日志为每个原子技能提供了标注好的执行结果。利用这些日志，我们更新了技能评估器 $\mathcal{E}$，用于估计在特定环境条件下成功执行的可能性。常见的故障模式包括抓取或放置零件失败、机器人与环境之间的意外碰撞，或由于次优装配步骤累积误差导致的结构性灾难故障。我们利用真实执行数据训练新的感知技能，并将其集成到技能图的原子技能或前置/后置条件中，以指导规划过程中的技能选择并在线监控执行结果。通过这种方式，系统利用部署期间收集的经验不断提高其技能的可靠性。


## VI. EXPERIMENTS ON ROBOTIC LEGO ASSEMBLY
## VI. 机器人乐高装配实验


We illustrate the proposed framework on a challenging LEGO assembly task that requires substantial configuration and tuning effort by domain experts. Starting from a minimal Skill Graph, the system is rapidly integrated and deployed. Through repeated execution, the system collects skill-level data and improves performance, demonstrating increased success rates and reduced execution time.
我们在一个具有挑战性的乐高装配任务上展示了所提框架，该任务需要领域专家进行大量的配置和调试。从一个最小化的技能图开始，系统得以快速集成并部署。通过重复执行，系统收集了技能级数据并提升了性能，表现出更高的成功率和更短的执行时间。


## A. System Setup and Skill Instantiation
## A. 系统设置与技能实例化


We initialize the Skill Graph with seven (Transit, Pick, Place up, Place Down, Support up, Support Down, Han-dover) manipulation policies described in [28] as atomic skills. The executables of these skills are implemented on two Yaskawa GP4 industrial robots equipped with force-torque sensors, as shown in Fig. 3. We then form three meta skills - PickPlace, PickPlacewSupport, and PickHandoverPlace from combining these atomic skills. Each of these represents a reusable higher-level capability that can directly manipulate a single LEGO assembly step.
我们初始化了技能图，包含 [28] 中描述的七种操作策略（转运、抓取、向上放置、向下放置、向上支撑、向下支撑、移交）作为原子技能。这些技能的可执行程序在两台配备力矩传感器的安川 GP4 工业机器人上实现，如图 3 所示。随后，我们将这些原子技能组合成三种元技能：PickPlace、PickPlacewSupport 和 PickHandoverPlace。每一项都代表了一种可重用的高级能力，能够直接处理单个乐高装配步骤。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_06_30_a5a772.jpg"/>



Fig. 4: Intuitive Task Specification Results: Transfer from Human demonstration to Robot Execution.
图 4：直观任务规范结果：从人类演示到机器人执行的迁移。


## B. Task Specification
## B. 任务规范


We define the tasks as a sequence of assembly steps, where each step,denoted as ${a}_{i}$ ,represents a LEGO brick of a specific type (e.g., 2x2 or 2x4) and a discretized position $\left( {x,y,z}\right)$ on a calibrated LEGO assembly plate. This assembly sequence can be directly specified by the user or inferred intuitively from a human demonstration video as shown in Fig. 4 In addition, we assume that the current initial state of each LEGO bricks and type are known.
我们将任务定义为一系列装配步骤，其中每个步骤（记为 ${a}_{i}$）代表特定类型的乐高积木（例如 2x2 或 2x4）以及校准后的乐高底板上的离散位置 $\left( {x,y,z}\right)$。该装配序列可由用户直接指定，或如图 4 所示，从人类演示视频中直观推断得出。此外，我们假设每块乐高积木的当前初始状态和类型均已知。


## C. Semantic Skill Planning and Execution
## C. 语义技能规划与执行


We then plan a grounded skill sequence as described in Sec. IV-A For the search-based skill planning, each assembly step is assigned a meta-skill, the corresponding robot arm, and an available LEGO brick. We use the LEGO-specific stability estimator [31], forward kinematic feasibility, and collision constraints to determine the feasibility of a meta-skill in its pre-condition. The transit trajectories are planned using RRT-Connect [32], and we follow Sec. IV-B to generate the asynchronous execution graph.
随后，我们按照第 IV-A 节所述规划一个落地的技能序列。对于基于搜索的技能规划，每个装配步骤都被分配一个元技能、相应的机械臂以及一块可用的乐高积木。我们利用乐高专用稳定性估计器 [31]、正向运动学可行性及碰撞约束来确定元技能在其前置条件下的可行性。过渡轨迹使用 RRT-Connect [32] 进行规划，并遵循第 IV-B 节生成异步执行图。


Fig. 3 shows an example skill-level trajectory recorded with our data backbone when building a LEGO 'Facuet', rendered in Fig. 6 (a). We visualize the temporal evolution of joint states and force readings over the full horizon, as well as video snapshots of all atomic manipulation skills.
图 3 展示了在构建图 6 (a) 所示的乐高“水龙头”时，利用我们的数据主干记录的技能级轨迹示例。我们可视化了全时域内关节状态和力传感读数的时间演变，以及所有原子操作技能的视频快照。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_06_30_41531b.jpg"/>



Fig. 5: Using planning and execution data to craft new vision-based perception in Skill Graph. In the top, we present the sources of data captured during skill-based planning and execution of a LEGO assembly sequence. The bottom half shows a pick/place post-condition evaluator from an in-hand camera and anomaly detection skill from side-view cameras.
图 5：利用规划与执行数据在技能图中构建新的视觉感知。上方展示了在乐高装配序列的基于技能的规划与执行过程中捕获的数据源。下半部分展示了来自手持相机的抓取/放置后置条件评估器，以及来自侧视相机的异常检测技能。


TABLE I: Bimanual Construction Comparison. Success Rate: number of trials attempted to have the system successfully build the brick design once without restarting. Survival Length: number of bricks (averaged over the attempts) the system assembled without restarting.
表 I：双臂构建对比。成功率：系统在无需重启的情况下成功完成一次积木设计构建的尝试次数。存活长度：系统在无需重启的情况下组装的积木数量（取尝试次数的平均值）。


<table><tr><td>Method</td><td>Design</td><td>Success Rate</td><td>Survival Length</td></tr><tr><td rowspan="4">Before Skill Improvement</td><td>Faucet</td><td>1/5</td><td>9.2</td></tr><tr><td>Fish</td><td>0/5</td><td>7.8</td></tr><tr><td>Vessel</td><td>1/3</td><td>33.7</td></tr><tr><td>Guitar</td><td>1/1</td><td>24</td></tr><tr><td rowspan="4">After Skill Improvement</td><td>Faucet</td><td>1/1</td><td>14</td></tr><tr><td>Fish</td><td>1/1</td><td>29</td></tr><tr><td>Vessel</td><td>1/1</td><td>36</td></tr><tr><td>Guitar</td><td>1/1</td><td>24</td></tr></table>
<table><tbody><tr><td>方法</td><td>设计</td><td>成功率</td><td>存续长度</td></tr><tr><td rowspan="4">技能改进前</td><td>水龙头</td><td>1/5</td><td>9.2</td></tr><tr><td>鱼</td><td>0/5</td><td>7.8</td></tr><tr><td>容器</td><td>1/3</td><td>33.7</td></tr><tr><td>吉他</td><td>1/1</td><td>24</td></tr><tr><td rowspan="4">技能改进后</td><td>水龙头</td><td>1/1</td><td>14</td></tr><tr><td>鱼</td><td>1/1</td><td>29</td></tr><tr><td>容器</td><td>1/1</td><td>36</td></tr><tr><td>吉他</td><td>1/1</td><td>24</td></tr></tbody></table>


## D. Failure Modes and Vision-Based Skill Evaluators
## D. 故障模式与基于视觉的技能评估器


Although the manipulation-only Skill Graph can assemble LEGO structures on real robots, execution may fail due to a variety of real-world uncertainties. These issues become more pronounced in long-horizon dual-arm assembly tasks. To improve reliability, we analyze the structured execution logs generated during deployment. Based on the observed failure modes, we introduce several data-driven improvements to the Skill Graph, including new pre-conditions, post-conditions, and perception skills. These are achieved using additional sensory inputs from an Eye-in-Finger (EIF) camera [33] and multiple third-view cameras (Fig. 5).
尽管仅基于操作的技能图（Skill Graph）可以在真实机器人上组装乐高结构，但由于现实世界中存在各种不确定性，执行过程仍可能失败。在长程双臂组装任务中，这些问题尤为突出。为了提高可靠性，我们分析了部署过程中生成的结构化执行日志。基于观察到的故障模式，我们对技能图进行了多项数据驱动的改进，包括引入新的前置条件、后置条件和感知技能。这些改进利用了来自指尖相机（EIF）[33]和多个第三视角相机的额外传感器输入（图5）。


## 1) Enhancing Post-Condition Checks for Pick/Place
## 1) 增强抓取/放置的后置条件检查


A major source of failure arises from undetected manipulation errors, such as a brick not being grasped successfully or not being released after placement. To address this issue, we use the collected data to train post-condition checkers that verify the outcome of pick and place operations using the EiF camera. A binary classifier built on DINOv2 visual features [34] determines whether the brick is securely in hand after a pick operation or successfully released after a place operation. These checkers allow the system to detect manipulation failures immediately and prevent error propagation to later stages of the task.
故障的主要来源是未被检测到的操作错误，例如积木抓取失败或放置后未成功释放。为解决此问题，我们利用收集到的数据训练后置条件检查器，通过指尖相机验证抓取和放置操作的结果。基于DINOv2视觉特征[34]构建的二分类器可判断积木在抓取后是否稳固，或在放置后是否成功释放。这些检查器使系统能够立即检测到操作故障，并防止错误传播到任务的后续阶段。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_06_30_ca4c0d.jpg"/>



Fig. 6: Brick assembly designs for comparing bimanual assembly construction. The number in each figure indicates the number of bricks required.
图6：用于比较双臂组装构建的积木设计。图中数字表示所需的积木数量。


## 2) Enhancing Pre-Condition Checks for Pick
## 2) 增强抓取的前置条件检查


Another common failure mode arises from millimeter-level pose misalignment caused by calibration drift or perception noise. Such errors can lead to failed grasps or unstable placements if left uncorrected. To address this issue, a pre-condition checker is introduced to estimate fine-grained pose offsets using the EiF camera before executing a pick operation. When misalignment is detected, the grasp pose is corrected prior to execution, improving the robustness of the pick skill.
另一种常见的故障模式是由校准漂移或感知噪声引起的毫米级位姿偏差。如果未加纠正，此类误差可能导致抓取失败或放置不稳定。为解决此问题，我们引入了一个前置条件检查器，在执行抓取操作前利用指尖相机估计细粒度的位姿偏移。当检测到偏差时，系统会在执行前修正抓取位姿，从而提高抓取技能的鲁棒性。


## 3) New Perception Skill for Structural Anomaly Detection
## 3) 用于结构异常检测的新感知技能


Long-horizon assembly may also fail due to structural anomalies such as loosened connections, accumulated placement errors, or fragile connections in the LEGO design itself, such as the Fish in Fig. 6 (b). To detect these situations, a new perception skill is introduced to monitor the partially assembled structure using a third-view camera. The observed structure is compared against the corresponding simulated structure in Gazebo [35], and a geometric discrepancy measure identifies abnormal assembly states. When anomalies are detected, the system pauses execution and requests corrective intervention from humans.
长程组装也可能因结构异常而失败，例如连接松动、累积的放置误差，或如图6(b)中鱼类模型本身存在的脆弱连接。为检测这些情况，我们引入了一种新的感知技能，通过第三视角相机监控部分组装完成的结构。将观察到的结构与Gazebo [35]中的相应模拟结构进行对比，通过几何差异度量识别异常的组装状态。当检测到异常时，系统会暂停执行并请求人工干预以进行修正。


Together, these improvements extend the manipulation-only Skill Graph into a perception-aware and data-informed framework capable of monitoring manipulation outcomes, correcting geometric errors, detecting structural anomalies, and adaptively allocating verification skills during planning. As shown in Fig. 1 incorporating these improvements significantly increases both task success rate and execution robustness in long-horizon assembly tasks.
总之，这些改进将仅基于操作的技能图扩展为一个具备感知能力且数据驱动的框架，能够监控操作结果、纠正几何误差、检测结构异常，并在规划过程中自适应地分配验证技能。如图1所示，整合这些改进显著提高了长程组装任务的任务成功率和执行鲁棒性。


## VII. DISCUSSION
## VII. 讨论


In addition to what is shown in the results, we discuss a few ways to further "close the loop" that utilize the data generated to improve the skill execution, adapt the task plan, and automate the failure detection and improvement loop.
除了结果中展示的内容外，我们还讨论了几种进一步“闭环”的方法，利用生成的数据来改进技能执行、调整任务规划，并实现故障检测与改进循环的自动化。


## A. Improving Skill Execution
## A. 改进技能执行


Collected data could improve execution policies for individual skills through two feedback loops: parametric adaptation for physical consistency and policy adaptation for algorithmic efficiency. 1. Parametric Adaptation: Real-world deployments introduce systematic biases due to calibration drift and mechanical wear. To resolve this, we introduce a Parametric Corrector that subscribes to logged error signals (e.g., offsets) and maintains an estimate of accumulated bias. Before a skill is executed, it queries this corrector to adjust the target parameters dynamically. This compensates for drift without requiring manual recalibration or high-level replanning. 2. Policy Adaptation: The skill policy supports multiple algorithm implementations (e.g., RRTConnect and BITStar). Algorithm selection is modeled as a contextual multi-armed bandit. The evaluator could be updated to evaluate the performance of different algorithm implementations using execution feedback (e.g., success rate and runtime). Over time, the planner could learn to select strategies appropriate to task context, such as compliant control for contact-rich tasks and stiff control for free-space motion.
收集到的数据可以通过两个反馈循环改进单个技能的执行策略：用于物理一致性的参数自适应和用于算法效率的策略自适应。1. 参数自适应：现实世界的部署会因校准漂移和机械磨损引入系统性偏差。为解决此问题，我们引入了一个参数校正器，它订阅记录的误差信号（如偏移量）并维护累积偏差的估计值。在执行技能前，它会查询该校正器以动态调整目标参数。这无需手动重新校准或进行高层重规划即可补偿漂移。2. 策略自适应：技能策略支持多种算法实现（如RRTConnect和BITStar）。算法选择被建模为上下文多臂老虎机问题。评估器可更新以利用执行反馈（如成功率和运行时间）来评估不同算法实现的性能。随着时间推移，规划器可以学习选择适合任务上下文的策略，例如在接触密集型任务中使用柔顺控制，在自由空间运动中使用刚性控制。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_06_30_ec75a7.jpg"/>



Fig. 7: Illustration of graph adaptation. (Top) The planner relies on static costs and selects the high-risk brick b4 (red). (Bottom) With adaptation, the planner updates the cost tensor with failure history. The solver autonomously reallocates the task to the lower-cost, redundant brick b2 (green), bypassing the failure mode.
图7：图自适应示意图。（上图）规划器依赖静态成本并选择了高风险积木b4（红色）。（下图）通过自适应，规划器利用故障历史更新了成本张量。求解器自主将任务重新分配给成本更低、具有冗余性的积木b2（绿色），从而绕过了故障模式。


## B. Failure-Probability-Aware Planning
## B. 具备故障概率感知能力的规划


Beyond improving individual skills, execution logs also enable planning-level adaptation. From collected execution traces, the empirical failure probability associated with each skill transition in the Skill Graph is estimated and used to update the corresponding skill evaluators $\mathcal{E}$ . These updated evaluators provide failure-probability-aware cost estimates to the planner. During planning, the system could selectively insert perception skills at steps with high estimated failure probability to detect anomalies, while low-risk steps can be executed without additional monitoring. In addition, the planner could also adapt the task allocation to improve the overall success rate, for example by selecting an alternative brick that exhibits less wear (statistically estimated from the failure rate). The adapted graph with an alternative brick is shown in Fig. 7. Although these adaptations are currently done manually, they are easy to automate.
除了改进单个技能外，执行日志还能实现规划层面的自适应。通过收集执行轨迹，可以估算技能图中每个技能转换所关联的经验故障概率，并据此更新相应的技能评估器 $\mathcal{E}$。这些更新后的评估器为规划器提供了感知故障概率的成本估算。在规划过程中，系统可以选择性地在故障概率较高的步骤中插入感知技能以检测异常，而低风险步骤则无需额外监控。此外，规划器还可以调整任务分配以提高整体成功率，例如选择磨损较小的积木（根据故障率进行统计估算）。图 7 展示了使用替代积木后的调整图。尽管这些调整目前是手动完成的，但它们很容易实现自动化。


## C. Autonomous Failure Discovery with Foundation Models
## C. 基于基础模型的主动故障发现


The Skill Graph representation also enables autonomous improvement through semi-automatic discovery of new failure modes using vision-language models (VLMs). For each execution, structured context is constructed from semantic task descriptions, skill specifications, and execution logs. Logged data, including trajectories, force signals, control commands, and visual observations, are summarized by analysis agents. VLMs extract key visual events from image streams, while signal-level agents compute interpretable temporal features such as force anomalies or trajectory deviations. These summaries could be fused with the execution context and analyzed by a language model to infer likely failure causes. As similar patterns recur across executions, the system can identify consistent failure modes and automatically propose new skill adaptations, evaluators, pre- or post-condition checks that capture these conditions.
技能图表示法还通过利用视觉语言模型（VLM）半自动发现新故障模式，实现了自主改进。对于每次执行，系统都会根据语义任务描述、技能规范和执行日志构建结构化上下文。分析代理会对包括轨迹、力信号、控制指令和视觉观测在内的日志数据进行汇总。VLM 从图像流中提取关键视觉事件，而信号级代理则计算可解释的时间特征，如力异常或轨迹偏差。这些汇总信息可与执行上下文融合，并由语言模型进行分析，从而推断出可能的故障原因。随着类似模式在多次执行中重复出现，系统能够识别出一致的故障模式，并自动提出新的技能调整、评估器或前/后置条件检查，以捕捉这些状况。


## VIII. CONCLUSION
## VIII. 结论


This paper presents a Skill Graph-based framework for autonomous integration and continuous improvement of robotic assembly systems. By structuring robot capabilities around semantic skills with explicit execution and evaluation interfaces, Skill Graphs enable rapid deployment, systematic data collection, and iterative performance improvement. We believe this approach offers a scalable foundation for adaptive, reusable, and safety-aware robotic systems.
本文提出了一种基于技能图的框架，用于机器人装配系统的自主集成与持续改进。通过围绕具有明确执行和评估接口的语义技能来构建机器人能力，技能图实现了快速部署、系统化数据收集和迭代性能提升。我们相信，该方法为自适应、可重用且具备安全意识的机器人系统提供了一个可扩展的基础。


## ACKNOWLEDGMENTS
## 致谢


This project is supported by The ARM Institute National Artificial Intelligence Data Foundry for Robotics and the Manufacturing Futures Institute at Carnegie Mellon University.
本项目由 ARM 研究所国家人工智能机器人数据代工厂及卡内基梅隆大学制造未来研究所资助。


## REFERENCES
## 参考文献


[1] D. Nau, Y. Cao, A. Lotem, and H. Muñoz-Avila, "Shop: Simple hierarchical ordered planner," in Proceedings of the 16th International Joint Conference on Artificial Intelligence (IJCAI), pp. 968-973, 1999.
[1] D. Nau, Y. Cao, A. Lotem, and H. Muñoz-Avila, "Shop: Simple hierarchical ordered planner," in Proceedings of the 16th International Joint Conference on Artificial Intelligence (IJCAI), pp. 968-973, 1999.


[2] M. Colledanchise and P. Ögren, "Behavior trees in robotics and ai," July 2018.
[2] M. Colledanchise and P. Ögren, "Behavior trees in robotics and ai," July 2018.


[3] L. P. Kaelbling and T. Lozano-Perez, "Integrated task and motion planning in belief space," International Journal of Robotics Research, vol. 32, no. 9, 2013.
[3] L. P. Kaelbling and T. Lozano-Perez, "Integrated task and motion planning in belief space," International Journal of Robotics Research, vol. 32, no. 9, 2013.


[4] R. S. Sutton, D. Precup, and S. Singh, "Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning," Artificial Intelligence, vol. 112, no. 1-2, pp. 181-211, 1999.
[4] R. S. Sutton, D. Precup, and S. Singh, "Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning," Artificial Intelligence, vol. 112, no. 1-2, pp. 181-211, 1999.


[5] D. McDermott, M. Ghallab, A. Howe, C. Knoblock, A. Ram, M. Veloso, D. Weld, D. Wilkins, and the AIPS-98 Planning Competition Committee, "Pddl - the planning domain definition language," technical report cvc tr-98-003 / dcs tr-1165, Yale Center for Computational Vision and Control / Yale University Department of Computer Science, October 1998. Planning Domain Definition Language manual for the AIPS-98 competition.
[5] D. McDermott, M. Ghallab, A. Howe, C. Knoblock, A. Ram, M. Veloso, D. Weld, D. Wilkins, and the AIPS-98 Planning Competition Committee, "Pddl - the planning domain definition language," technical report cvc tr-98-003 / dcs tr-1165, Yale Center for Computational Vision and Control / Yale University Department of Computer Science, October 1998. Planning Domain Definition Language manual for the AIPS-98 competition.


[6] E. Erdem and V. Patoglu, "Answer set programming in robotics," Theory and Practice of Logic Programming, vol. 12, no. 4-5, pp. 433- 466, 2012.
[6] E. Erdem and V. Patoglu, "Answer set programming in robotics," Theory and Practice of Logic Programming, vol. 12, no. 4-5, pp. 433- 466, 2012.


[7] M. Helmert, "The fast downward planning system," Journal of Artificial Intelligence Research, vol. 26, p. 191-246, July 2006.
[7] M. Helmert, "The fast downward planning system," Journal of Artificial Intelligence Research, vol. 26, p. 191-246, July 2006.


[8] D. Höller, G. Behnke, P. Bercher, S. Biundo, H. Fiorino, D. Pellier, and R. Alford, "Hddl: An extension to pddl for expressing hierarchical planning problems," Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 9883-9891, Apr. 2020.
[8] D. Höller, G. Behnke, P. Bercher, S. Biundo, H. Fiorino, D. Pellier, and R. Alford, "Hddl: An extension to pddl for expressing hierarchical planning problems," Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 9883-9891, Apr. 2020.


[9] Z. Kootbally, C. Schlenoff, C. Lawler, T. Kramer, and S. Gupta, "Towards robust assembly with knowledge representation for the planning domain definition language (pddl)," Robotics and Computer-Integrated Manufacturing, vol. 33, pp. 42-55, 2015. Special Issue on Knowledge Driven Robotics and Manufacturing.
[9] Z. Kootbally, C. Schlenoff, C. Lawler, T. Kramer, 和 S. Gupta，“迈向基于规划领域定义语言（PDDL）知识表示的鲁棒装配”，《机器人与计算机集成制造》，第33卷，第42-55页，2015年。知识驱动机器人与制造特刊。


[10] Y. Jiang, S. Zhang, P. Khandelwal, et al., "Task planning in robotics: an empirical comparison of pddl- and asp-based systems," Frontiers of Information Technology & Electronic Engineering, vol. 20, no. 3, pp. 363-373, 2019.
[10] Y. Jiang, S. Zhang, P. Khandelwal 等，“机器人任务规划：基于PDDL和ASP系统的实证比较”，《信息技术与电子工程前沿》，第20卷，第3期，第363-373页，2019年。


[11] J. Buehler and M. Pagnucco, "A framework for task planning in heterogeneous multi robot systems based on robot capabilities," Proceedings of the AAAI Conference on Artificial Intelligence, vol. 28, Jun. 2014.
[11] J. Buehler 和 M. Pagnucco，“一种基于机器人能力的异构多机器人系统任务规划框架”，《AAAI人工智能会议论文集》，第28卷，2014年6月。


[12] I. T. Michailidis, P. Michailidis, E. Kosmatopoulos, and N. Bassiliades, "Open-source online mission-planning in emergent environments with pddl for multi-robot applications," in Artificial Intelligence Applications and Innovations. AIAI 2024 IFIP WG 12.5 International Workshops (I. Maglogiannis, L. Iliadis, I. Karydis, A. Papaleonidas, and I. Chochliouros, eds.), (Cham), pp. 433-446, Springer Nature Switzerland, 2024.
[12] I. T. Michailidis, P. Michailidis, E. Kosmatopoulos, 和 N. Bassiliades，“在多机器人应用中利用PDDL进行突发环境下的开源在线任务规划”，载于《人工智能应用与创新》，AIAI 2024 IFIP WG 12.5 国际研讨会（I. Maglogiannis, L. Iliadis, I. Karydis, A. Papaleonidas, 和 I. Chochliouros 编辑），（Cham），第433-446页，Springer Nature Switzerland，2024年。


[13] S. Levine, C. Finn, T. Darrell, and P. Abbeel, "End-to-end training of deep visuomotor policies," The Journal of Machine Learning Research, vol. 17, no. 1, pp. 1334-1373, 2016.
[13] S. Levine, C. Finn, T. Darrell, 和 P. Abbeel，“深度视觉运动策略的端到端训练”，《机器学习研究杂志》，第17卷，第1期，第1334-1373页，2016年。


[14] R.-Z. Qiu, S. Yang, X. Cheng, C. Chawla, J. Li, T. He, G. Yan, D. J. Yoon, R. Hoque, L. Paulsen, et al., "Humanoid policy" human policy," arXiv preprint arXiv:2503.13441, 2025.
[14] R.-Z. Qiu, S. Yang, X. Cheng, C. Chawla, J. Li, T. He, G. Yan, D. J. Yoon, R. Hoque, L. Paulsen 等，“人形策略”人类策略”，arXiv 预印本 arXiv:2503.13441，2025年。


[15] H. Zhu, A. Gupta, A. Rajeswaran, V. Kumar, and S. L. Kumar, "Dexterous manipulation with deep reinforcement learning: Efficient, general, and low-cost," IEEE International Conference on Robotics and Automation (ICRA), pp. 3651-3657, 2019.
[15] H. Zhu, A. Gupta, A. Rajeswaran, V. Kumar, 和 S. L. Kumar，“基于深度强化学习的灵巧操作：高效、通用且低成本”，《IEEE国际机器人与自动化会议 (ICRA)》，第3651-3657页，2019年。


[16] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al., "Rt-1: Robotics transformer for real-world control at scale," arXiv preprint arXiv:2212.06817, 2022.
[16] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu 等，“RT-1：用于大规模现实世界控制的机器人 Transformer”，arXiv 预印本 arXiv:2212.06817，2022年。


[17] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choro-manski, T. Ding, D. Driess, A. Dubey, C. Finn, et al., "Rt-2: Vision-language-action models transfer web knowledge to robotic control," arXiv preprint arXiv:2307.15818, 2023.
[17] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choro-manski, T. Ding, D. Driess, A. Dubey, C. Finn 等，“RT-2：将网络知识迁移至机器人控制的视觉-语言-动作模型”，arXiv 预印本 arXiv:2307.15818，2023年。


[18] O. M. Team, D. Ghosh, H. Homer, et al., "Octo: An open-source generalist robot policy," arXiv preprint arXiv:2405.12213, 2024.
[18] O. M. Team, D. Ghosh, H. Homer 等，“Octo：一种开源通用机器人策略”，arXiv 预印本 arXiv:2405.12213，2024年。


[19] W. Mao, W. Zhong, Z. Jiang, D. Fang, Z. Zhang, Z. Lan, H. Li, F. Jia, T. Wang, H. Fan, and O. Yoshie, "Robomatrix: A skill-centric hierarchical framework for scalable robot task planning and execution in open-world," 2025.
[19] W. Mao, W. Zhong, Z. Jiang, D. Fang, Z. Zhang, Z. Lan, H. Li, F. Jia, T. Wang, H. Fan, 和 O. Yoshie，“Robomatrix：一种用于开放世界中可扩展机器人任务规划与执行的技能中心化分层框架”，2025年。


[20] J. Zhou, K. Ye, J. Liu, T. Ma, Z. Wang, R. Qiu, K.-Y. Lin, Z. Zhao, and J. Liang, "Exploring the limits of vision-language-action manipulations in cross-task generalization," 2025.
[20] J. Zhou, K. Ye, J. Liu, T. Ma, Z. Wang, R. Qiu, K.-Y. Lin, Z. Zhao, 和 J. Liang，“探索跨任务泛化中视觉-语言-动作操作的极限”，2025年。


[21] O. Kroemer, S. Niekum, and G. D. Konidaris, "A review of robot learning for manipulation: Challenges, representations, and algorithms," CoRR, vol. abs/1907.03146, 2019.
[21] O. Kroemer, S. Niekum, 和 G. D. Konidaris，“机器人操作学习综述：挑战、表示与算法”，CoRR，第 abs/1907.03146 卷，2019年。


[22] T. Shankar, C. Chawla, A. Hassan, and J. Oh, "Translating agent-environment interactions from humans to robots," in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 8952-8959, IEEE, 2024.
[22] T. Shankar, C. Chawla, A. Hassan, 和 J. Oh, “将人机交互转化为机器人行为,” 载于 2024 IEEE/RSJ 国际机器人与系统会议 (IROS), 第 8952-8959 页, IEEE, 2024.


[23] I. Mishani, Y. Shaoul, and M. Likhachev, "Mosaic: A skill-centric algorithmic framework for long-horizon manipulation planning," arXiv preprint arXiv:2504.16738, 2025.
[23] I. Mishani, Y. Shaoul, 和 M. Likhachev, “Mosaic: 一种用于长程操作规划的以技能为中心的算法框架,” arXiv 预印本 arXiv:2504.16738, 2025.


[24] A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal, "Dynamical movement primitives: learning attractor models for motor behaviors," Neural Computation, vol. 25, no. 2, pp. 328-373, 2013.
[24] A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, 和 S. Schaal, “动态运动基元：学习运动行为的吸引子模型,” 神经计算, 第 25 卷, 第 2 期, 第 328-373 页, 2013.


[25] T. E. Lee, S. Vats, S. Girdhar, and O. Kroemer, "Scale: Causal learning and discovery of robot manipulation skills using simulation," in Proceedings of The 7th Conference on Robot Learning (J. Tan, M. Toussaint, and K. Darvish, eds.), vol. 229 of Proceedings of Machine Learning Research, pp. 2229-2256, PMLR, 06-09 Nov 2023.
[25] T. E. Lee, S. Vats, S. Girdhar, 和 O. Kroemer, “Scale: 使用仿真进行机器人操作技能的因果学习与发现,” 载于第 7 届机器人学习会议论文集 (J. Tan, M. Toussaint, 和 K. Darvish 编), 机器学习研究论文集第 229 卷, 第 2229-2256 页, PMLR, 2023 年 11 月 6-9 日.


[26] M. Diehl, I. Zappa, A. M. Zanchettin, and K. Ramirez-Amaro, "Learning robot skills from demonstration for multi-agent planning*," in 2024 IEEE 20th International Conference on Automation Science and Engineering (CASE), pp. 2348-2355, 2024.
[26] M. Diehl, I. Zappa, A. M. Zanchettin, 和 K. Ramirez-Amaro, “从演示中学习多智能体规划的机器人技能*,” 载于 2024 IEEE 第 20 届自动化科学与工程国际会议 (CASE), 第 2348-2355 页, 2024.


[27] M. Beetz, D. Beßler, A. Haidu, M. Pomarlan, A. K. Bozcuoglu, and G. Bartels, "Knowrob 2.0-a 2nd generation knowledge processing framework for cognition-enabled robotic agents," International Conference on Robotics and Automation (ICRA), pp. 512-519, 2018.
[27] M. Beetz, D. Beßler, A. Haidu, M. Pomarlan, A. K. Bozcuoglu, 和 G. Bartels, “Knowrob 2.0——一种用于认知机器人智能体的第二代知识处理框架,” 国际机器人与自动化会议 (ICRA), 第 512-519 页, 2018.


[28] P. Huang, R. Liu, S. Aggarwal, C. Liu, and J. Li, "Apex-mr: Multi-robot asynchronous planning and execution for cooperative assembly," in Robotics: Science and Systems, 2025.
[28] P. Huang, R. Liu, S. Aggarwal, C. Liu, 和 J. Li, “Apex-mr: 用于协作装配的多机器人异步规划与执行,” 载于机器人：科学与系统, 2025.


[29] W. Hoenig, T. K. Kumar, L. Cohen, H. Ma, H. Xu, N. Ayanian, and S. Koenig, "Multi-agent path finding with kinematic constraints," in Proceedings of the International Conference on Automated Planning and Scheduling (ICAPS), vol. 26, pp. 477-485, Mar. 2016.
[29] W. Hoenig, T. K. Kumar, L. Cohen, H. Ma, H. Xu, N. Ayanian, 和 S. Koenig, “具有运动学约束的多智能体路径规划,” 载于国际自动化规划与调度会议论文集 (ICAPS), 第 26 卷, 第 477-485 页, 2016 年 3 月.


[30] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al., "Gemini: a family of highly capable multimodal models," arXiv preprint arXiv:2312.11805, 2023.
[30] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, 等, “Gemini: 一个功能强大的多模态模型家族,” arXiv 预印本 arXiv:2312.11805, 2023.


[31] R. Liu, K. Deng, Z. Wang, and C. Liu, "Stablelego: Stability analysis of block stacking assembly," IEEE Robotics and Automation Letters, vol. 9, no. 11, pp. 9383-9390, 2024.
[31] R. Liu, K. Deng, Z. Wang, 和 C. Liu, “Stablelego: 积木堆叠装配的稳定性分析,” IEEE 机器人与自动化快报, 第 9 卷, 第 11 期, 第 9383-9390 页, 2024.


[32] J. J. Kuffner and S. M. LaValle, "Rrt-connect: An efficient approach to single-query path planning," in Proceedings 2000 ICRA. Millennium conference. IEEE international conference on robotics and automation. Symposia proceedings (Cat. No. 00CH37065), vol. 2, pp. 995- 1001, IEEE, 2000.
[32] J. J. Kuffner 和 S. M. LaValle, “Rrt-connect: 一种高效的单查询路径规划方法,” 载于 2000 年 ICRA 论文集. 千禧年会议. IEEE 国际机器人与自动化会议. 研讨会论文集 (目录号 00CH37065), 第 2 卷, 第 995-1001 页, IEEE, 2000.


[33] Z. Tang, R. Liu, and C. Liu, "Eye-in-finger: Smart fingers for delicate assembly and disassembly of lego," in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 20990- 20996, IEEE, 2025.
[33] Z. Tang, R. Liu, 和 C. Liu, “Eye-in-finger: 用于乐高精密装配与拆卸的智能手指,” 载于 2025 IEEE/RSJ 国际机器人与系统会议 (IROS), 第 20990-20996 页, IEEE, 2025.


[34] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khali-dov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., "Dinov2: Learning robust visual features without supervision," arXiv preprint arXiv:2304.07193, 2023.
[34] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khali-dov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, 等, “Dinov2: 在无监督情况下学习鲁棒视觉特征,” arXiv 预印本 arXiv:2304.07193, 2023.


[35] N. Koenig and A. Howard, "Design and use paradigms for gazebo, an open-source multi-robot simulator," in 2004 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), vol. 3, pp. 2149-2154, IEEE, 2004.
[35] N. Koenig 和 A. Howard，“Gazebo：一种开源多机器人模拟器的设计与使用范式”，载于 2004 年 IEEE/RSJ 国际智能机器人与系统会议 (IROS)，第 3 卷，第 2149-2154 页，IEEE，2004 年。