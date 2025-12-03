# Long-Horizon Planning for Multi-Agent Robots in Partially Observable Environments
长期视野规划：在部分可观测环境中的多智能体机器人


Siddharth Nayak ${}^{1 * }$ Adelmo Morrison Orozco ${}^{1 * }$ Marina Ten Have ${}^{1}$ Vittal Thirumalai ${}^{1}$ Jackson Zhang ${}^{1}$ Darren Chen ${}^{1}$ Aditya Kapoor ${}^{2}$ Eric Robinson ${}^{3}$ Karthik Gopalakrishnan ${}^{4}$ James Harrison ${}^{5} \; {\text{ on }}^{5}$ Brian Ichter ${}^{ \dagger  }{}^{5}$ Anuj Mahajan ${}^{ \ddagger  }{}^{6}$ Hamsa Balakrishnan ${}^{1}$
Siddharth Nayak ${}^{1 * }$ Adelmo Morrison Orozco ${}^{1 * }$ Marina Ten Have ${}^{1}$ Vittal Thirumalai ${}^{1}$ Jackson Zhang ${}^{1}$ Darren Chen ${}^{1}$ Aditya Kapoor ${}^{2}$ Eric Robinson ${}^{3}$ Karthik Gopalakrishnan ${}^{4}$ James Harrison ${}^{5} \; {\text{ on }}^{5}$ Brian Ichter ${}^{ \dagger  }{}^{5}$ Anuj Mahajan ${}^{ \ddagger  }{}^{6}$ Hamsa Balakrishnan ${}^{1}$


${}^{1}$ MIT ${}^{2}$ TCS ${}^{3}$ USAF-MIT AI Accelerator
${}^{1}$ MIT ${}^{2}$ TCS ${}^{3}$ USAF-MIT AI Accelerator


${}^{4}$ Stanford ${}^{5}$ Google DeepMind ${}^{6}$ Apple
${}^{4}$ Stanford ${}^{5}$ Google DeepMind ${}^{6}$ Apple


## Abstract
## 摘要


The ability of Language Models (LMs) to understand natural language makes them a powerful tool for parsing human instructions into task plans for autonomous robots. Unlike traditional planning methods that rely on domain-specific knowledge and handcrafted rules, LMs generalize from diverse data and adapt to various tasks with minimal tuning, acting as a compressed knowledge base. However, LMs in their standard form face challenges with long-horizon tasks, particularly in partially observable multi-agent settings. We propose an LM-based Long-Horizon Planner for Multi-Agent Robotics (LLaMAR), a cognitive architecture for planning that achieves state-of-the-art results in long-horizon tasks within partially observable environments. LLaMAR employs a plan-act-correct-verify framework, allowing self-correction from action execution feedback without relying on oracles or simulators. Additionally, we present MAP-THOR, a comprehensive test suite encompassing household tasks of varying complexity within the AI2-THOR environment. Experiments show that LLaMAR achieves a 30% higher success rate than other state-of-the-art LM-based multi-agent planners in MAP-THOR and Search & Rescue tasks. Code can be found at https://github.com/nsidn98/LLaMAR
语言模型（LM）理解自然语言的能力使其成为将人类指令解析为自主机器人任务计划的强大工具。不同于依赖领域特定知识和手工规则的传统规划方法，LM 可以从多样化数据中泛化，并通过最少的调优适应各种任务，充当压缩的知识库。然而，标准形式的 LM 在长期视野任务中面临挑战，尤其是在部分可观测的多智能体环境中。我们提出了基于 LM 的多智能体机器人长期规划器 LLaMAR，一种实现长期任务规划的认知架构，在部分可观测环境中取得了最先进的结果。LLaMAR 采用 plan-act-correct-verify 框架，允许在不依赖甲骨文或模拟器的情况下根据执行反馈进行自我纠正。此外，我们提出了 MAP-THOR，这是一个在 AI2-THOR 环境中涵盖不同复杂度家务任务的综合测试套件。实验表明，在 MAP-THOR 和搜救任务中，LLaMAR 比其他最先进的基于 LM 的多智能体规划器成功率高 30%。代码见 https://github.com/nsidn98/LLaMAR


## 1 Introduction
## 1 引言


Creating embodied agents that assist humans in real-life scenarios is a significant challenge when humans communicate their intentions in natural language. Certain tasks like moving furniture [1, 2], search-and-rescue [3], environmental monitoring [4], etc., require coordination among multiple agents to solve the tasks efficiently as compared to single-agent scenarios. This challenge of understanding these natural language inputs and effectively coordinating these agents to solve the task is exacerbated in the multi-agent scenario. Recent works [5, 6, 7, 8, 9, 10, 11, 12, 13] have shown that Language Models (LMs) ${}^{1}$ can effectively use language instructions to develop plans for robots. However,most studies focus on single-agent long-horizon task planning. Naïve extensions of single-agent planning algorithms to multi-agent settings often fail due to environment non-stationarity, where the policies of other agents—modeled as a part of the environment—are continuously changing [14, 15]. Such failures lead to suboptimal performance as agents struggle to anticipate and adapt to the actions of others. We therefore formulate a centralized process in which decisions are made simultaneously for all agents based on their (partial) observations, similar to the centralized multi-agent system framework (CMAS) proposed in [16]. Leveraging the ability of pre-trained LMs to generalize across diverse tasks, we aim to use LMs for long-horizon embodied multi-agent task planning.
在现实场景中创建能够用自然语言理解人类意图并协助人的具身智能体是一项重大挑战。像搬动家具 [1, 2]、搜救 [3]、环境监测 [4] 等任务相比单智能体场景更需要多智能体之间的协同以提高效率。理解这些自然语言输入并有效协调多智能体以完成任务在多智能体场景下更为艰巨。最近的工作 [5, 6, 7, 8, 9, 10, 11, 12, 13] 表明语言模型（LMs） ${}^{1}$ 能够有效利用语言指令为机器人制定计划。然而，大多数研究聚焦于单智能体的长期任务规划。简单地将单智能体规划算法扩展到多智能体设置常常因环境非平稳性而失败，即其他智能体的策略——被视为环境的一部分——在不断变化 [14, 15]。这种失败导致性能次优，因为智能体难以预测并适应他者行为。因此，我们提出一个集中式过程，根据各智能体的（部分）观测同时为所有智能体做出决策，类似于 [16] 提出的集中式多智能体系统框架（CMAS）。利用预训练 LM 在多样任务间泛化的能力，我们旨在将 LM 用于长期视野的具身多智能体任务规划。


The key insight of our work is that integrating a plan-act-correct-verify framework with LMs enables a robust and adaptive approach to multi-agent task planning in dynamic, partially observable environments that allows agents to: (1) plan subtasks required to complete the task, (2) select high-level actions for each agent to complete the proposed subtasks, (3) identify and correct failures after high-level action execution, and (4) self-verify subtask completion based on high-level action execution. Unlike existing methods, our approach uses real-time execution feedback, observations, and agent histories to iteratively refine action planning and execution. This allows agents to adjust strategies based on reasoned insights on action execution, effectively addressing failures without relying on perfect environmental knowledge or oracle feedback. The correction and verification process in our cognitive architecture [17] is grounded in the environment's reality, which sets it apart from LM self-verification methods that lack such grounding [18]. This framework enhances agents' ability to complete complex, long-horizon tasks, yielding substantial improvement over current state-of-the-art methods.
我们工作的关键见解是，将 plan-act-correct-verify 框架与 LM 集成，能在动态、部分可观测环境中提供一种稳健且自适应的多智能体任务规划方法，使智能体能：（1）规划完成任务所需的子任务，（2）为每个智能体选择完成子任务的高级动作，（3）在执行高级动作后识别并纠正失败，以及（4）根据高级动作执行自我验证子任务完成情况。与现有方法不同，我们的方法使用实时执行反馈、观测和智能体历史来迭代地改进行动规划与执行。这使智能体能基于对动作执行的推理性见解调整策略，有效解决失败而无需依赖完美环境知识或甲骨文反馈。我们的认知架构 [17] 中的纠正与验证过程以环境的现实为基础，这使其区别于缺乏此类着地的 LM 自我验证方法 [18]。该框架增强了智能体完成复杂长期任务的能力，在现有最先进方法上带来显著改进。


---



* Equal Contribution. ${}^{ \dagger  }$ Now at Physical Intelligence. ${}^{ \ddagger  }$ Work done outside Apple. ${}^{1}$ We denote Large Language Models as LLMs, Vision Language Models as VLMs
* 同等贡献。 ${}^{ \dagger  }$ 现任 Physical Intelligence。 ${}^{ \ddagger  }$ 工作在 Apple 之外完成。 ${}^{1}$ 我们将大型语言模型称为 LLM，将视觉语言模型称为 VLM


---



Similar to our approach, recent works [19, 20, 21, 22, 23, 24, 16] utilize LMs for multi-agent planning, often adopting a hierarchical decision-making structure. The LMs are used for high-level planning to determine subtasks, sometimes in conjunction with planning domain definition language (PDDL) that together with the LM planner, functions as a feasibility solver. Specific actions are executed using low-level policies pre-trained through reinforcement learning, behavior cloning, or heuristic approaches. While these methods effectively use LMs as high-level planners, they assume perfect low-level primitive action policies and simulator or oracle-provided environmental information. By contrast, LLaMAR does not assume perfect knowledge of the environment, does not rely on oracle feedback, and does not assume perfect execution of low-level primitive policies. This approach moves us closer to enabling real-world robots that operate independently of privileged knowledge.
与我们的方法类似，近期工作 [19, 20, 21, 22, 23, 24, 16] 使用语言模型进行多智能体规划，常采用分层决策结构。语言模型用于高层规划以确定子任务，有时结合规划域定义语言 (PDDL)，与语言模型规划器一起作为可行性求解器。具体动作由通过强化学习、行为克隆或启发式方法预训练的低层策略执行。尽管这些方法有效地将语言模型作为高层规划器，但它们假设低层原始动作策略完美且模拟器或预言机提供环境信息。相比之下，LLaMAR 不假设对环境的完美知识，不依赖预言机反馈，也不假设低层原始策略能完美执行。该方法使我们更接近实现无需特权知识即可独立运行的真实机器人。


To avoid ambiguity, we use the following conventions. We refer to the objectives within the environments as "tasks" or "goals" and "subtasks" to describe the breakdown of tasks or goals. "High-level actions" are defined as skills the agent can perform, while "low-level actions" or "primitive actions" refer to existing policies—either learned or predefined using heuristics—that execute a sequence of actions to accomplish a high-level action. More details and examples can be found in Appendix A
为避免歧义，我们使用以下约定。我们将环境内的目标称为“任务”或“目标”，将任务或目标的分解称为“子任务”。“高层动作”定义为智能体可执行的技能，而“低层动作”或“原始动作”指现有的策略——通过学习得到或用启发式预定义——这些策略执行一系列动作以完成一个高层动作。更多细节和示例见附录 A


The main contributions of this paper are:
本文的主要贡献有：


- LLaMAR: An LM-based Long-Horizon Planner for Multi-Agent Robotics, designed for iterative planning of long-horizon, multi-objective tasks in partially observable environments, with the following key features:
- LLaMAR：一种基于语言模型的多智能体长程规划器，旨在对部分可观测环境中的长程、多目标任务进行迭代规划，具有以下关键特性：


- It operates without prior knowledge of the environment, allowing agents to explore and make decisions based on new observations.
- 无需事先了解环境，允许智能体基于新观测进行探索和决策。


- It evaluates outcomes through direct observation of images, rather than relying on oracles for feedback, enabling independent identification and correction of action failures.
- 通过直接观测图像评估结果，而非依赖预言机反馈，从而能够独立识别并纠正动作失败。


- MAP-THOR (Multi-Agent Planning in THOR): a benchmark suite of tasks within the AI2- THOR simulator under partial observability to standardize methodologies and metrics for evaluating multi-agent planning effectiveness and robustness.
- MAP-THOR（THOR 中的多智能体规划）：在 AI2-THOR 模拟器下的部分可观测任务基准套件，用以规范评估多智能体规划有效性和鲁棒性的方法与度量。


## 2 Related Work
## 2 相关工作


Reinforcement Learning (RL) for Long-Horizon Planning: While RL algorithms have shown promise in many applications, they still struggle with long-horizon tasks. Hierarchical reinforcement learning (HRL) has been used to address these challenges in both single-agent [25, 26, 27, 28] and multi-agent settings [29, 30, 31]. However, these approaches are typically applied to single-task, stationary environments, such as games, where agents solve for one goal in a fixed environment. Consequently, these methods do not generalize well across multiple environments or tasks. Multi-task RL has been explored as a potential solution, requiring sophisticated task planning to handle diverse objectives [32, 33]. This often involves decomposing tasks into manageable subtasks, a process well-suited for hierarchical frameworks. However, subtasks are known apriori in multi-task RL formulations. Real-world long-horizon RL necessitates robust task planning, and LMs have emerged as a promising approach for this purpose.
用于长程规划的强化学习（RL）：尽管 RL 算法在许多应用中展现潜力，但在长程任务上仍存在困难。分层强化学习（HRL）已被用于解决单智能体 [25, 26, 27, 28] 和多智能体 [29, 30, 31] 环境中的这些挑战。然而，这些方法通常应用于单任务、静态环境（如游戏），智能体在固定环境中解决单一目标。因此，这些方法在多个环境或任务间泛化性较差。多任务 RL 被作为一种潜在解决方案进行探索，但需要复杂的任务规划以处理多样目标 [32, 33]，通常涉及将任务分解为可管理的子任务，这一过程适合分层框架。但在多任务 RL 的设定中，子任务是先验已知的。真实世界的长程 RL 需要鲁棒的任务规划，语言模型已成为这一目的的有希望方法。


LMs for Embodied Single-Agent Planning: Recent studies have demonstrated the effectiveness of LMs in generating and executing plans in embodied single-agent environments [34, 35, 36, 37, 38, 39 40] and creating plans in single-agent embodied robotic environments [41, 42, 43, 44, 45, 46, 47] 48 49 50, 51 52]. Works like SayCan [5] and Grounded Decoding [6] use a combination of value functions and LLM predictions for long-horizon tasks. ProgPrompt [8] and Zero-Shot Language Planner [13] generate static plans executed in the environment, which may fail in partially observable and dynamic settings. To mitigate this, LLM-planner [53] updates plans based on new observations, similar to our approach.
用于具身单智能体规划的语言模型：近期研究展示了语言模型在具身单智能体环境中生成和执行计划的有效性 [34, 35, 36, 37, 38, 39 40]，以及在具身单智能体机器人环境中创建计划的能力 [41, 42, 43, 44, 45, 46, 47] 48 49 50, 51 52]。像 SayCan [5] 和 Grounded Decoding [6] 这样的工作结合了价值函数和 LLM 预测用于长程任务。ProgPrompt [8] 和 Zero-Shot Language Planner [13] 生成静态计划在环境中执行，这在部分可观测和动态环境中可能失败。为缓解此问题，LLM-planner [53] 会根据新观测更新计划，类似于我们的方法。


LMs for Multi-Agent Planning: Xu et al. [54] use LLMs in multi-agent games, while CoNavGPT [23] creates global plans for two robots in an embodied environment. RoCo [24] and CoELA [22] assign separate LMs to each agent for decentralized action prediction, allowing natural language communication between agents. However, RoCo and CoNavGPT require detailed environment information for planning, and CoELA's action space is filtered by an oracle. Relying on privileged information from an oracle is impractical in real-world applications. By contrast, our work focuses on free-form action generation and handles tasks with more ambiguous descriptions. Prior work [16] compare centralized (CMAS) and decentralized (DMAS) planning frameworks, showing that centralized planners perform better, though their experiments are in simple, known environments with limited number of agents. Two-Step [21] decomposes goals for main and helper agents, using PDDL planners for high-level actions. SmartLLM [19] uses multiple LLM modules for subtask decomposition, multi-robot group formation and task allocation but assumes robots have complete knowledge of the environment, making plans prone to errors in unknown settings. S-ATLAS [20] use LLMs with conformal prediction for safe multi-agent planning, but the action choices are limited to a small set of objects. Table 1 presents a comparison of the characteristics of different LM-based approaches to multi-agent planning with our work.
LMs 用于多智能体规划：Xu 等人 [54] 在多智能体游戏中使用 LLMs，而 CoNavGPT [23] 在具身环境中为两台机器人生成全局计划。RoCo [24] 和 CoELA [22] 为每个智能体分配独立的 LM 以进行去中心化的动作预测，允许智能体之间用自然语言通信。然而，RoCo 和 CoNavGPT 在规划时需要详细的环境信息，CoELA 的动作空间被一个神谕过滤。依赖来自神谕的特权信息在真实世界应用中不可行。相比之下，我们的工作侧重于自由形式动作生成，并处理描述更模糊的任务。先前工作 [16] 比较了中心化（CMAS）和去中心化（DMAS）规划框架，显示中心化规划器表现更好，但他们的实验是在简单、已知环境且智能体数量有限的条件下进行的。Two-Step [21] 为主智能体和辅助智能体分解目标，使用 PDDL 规划器生成高层动作。SmartLLM [19] 使用多个 LLM 模块进行子任务分解、多机器人组建和任务分配，但假设机器人对环境有完整认识，在未知环境中容易产生错误。S-ATLAS [20] 将 LLM 与一致性预测结合用于安全的多智能体规划，但动作选择仅限于少量对象。表 1 比较了不同基于 LM 的多智能体规划方法与我们工作的特点。


LMs can interpret high-level instructions and break them down into feasible subtasks, making them ideal for long-horizon, multi-task scenarios. Our work leverages LMs to enable long-horizon planning across a variety of tasks and environments, building on these advances to address the limitations of traditional RL and HRL methods. By integrating LMs into our planning framework, we enhance the ability to generalize across diverse tasks and scenarios, making significant strides toward practical, real-world applications of RL in dynamic, multi-agent settings.
LMs 能解读高层指令并将其拆解为可行的子任务，使其非常适合长时域、多任务场景。我们的工作利用 LMs 实现跨多种任务和环境的长时域规划，在这些进展基础上解决传统强化学习和分层强化学习方法的局限。通过将 LMs 集成到我们的规划框架中，我们增强了在多样任务和场景间的泛化能力，向在动态多智能体环境中实现实际可用的强化学习迈出重要一步。


<table><tr><td>Method</td><td>Dynamic Planning</td><td>Local Information</td><td>Failure Correction</td><td>Self Verification</td></tr><tr><td>Two-Step [21]</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>Smart LLM [19]</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>S-ATLAS [20]</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>CoELA [22]</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>LLaMAR (this paper)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>
<table><tbody><tr><td>方法</td><td>动态规划</td><td>局部信息</td><td>错误修正</td><td>自我验证</td></tr><tr><td>两步法 [21]</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>智能大模型 [19]</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>S-ATLAS [20]</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>CoELA [22]</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>LLaMAR（本文）</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></tbody></table>


Table 1: The proposed model, LLaMAR: 1) performs dynamic planning, avoiding the open-loop plan-and-execute paradigm; 2) operates without privileged simulator information (e.g., access to all objects in the environment); 3) re-plans when low-level actions fail, not assuming perfect execution; and 4) self-verifies subtask completion without relying on the simulator.
表 1：所提模型 LLaMAR：1）进行动态规划，避免开环的“计划并执行”范式；2）在没有特权模拟器信息（例如无法访问环境中所有对象）的情况下运行；3）在低级动作失败时重新规划，而非假定执行完美；4）无需依赖模拟器即可自我验证子任务完成情况。


## 3 Background
## 3 背景


Problem Setting: We consider a setting where multiple robots perform a series of tasks (a) such as cleaning a room or putting groceries in the fridge, in a home-like environment, and (b) rescuing missing personnel and putting out forest fires in a search & rescue environment (SAR). These tasks typically require long-horizon planning, involving around 100 low-level actions to reach the goal. Our objective is to compute plans for a team of robots to execute high-level language instructions, $I$ . We formalize these tasks as partially observable Markov decision processes (POMDP) [55, 56], denoted as $\left\langle  {N,\mathcal{I},\mathcal{S},\left\{  {\mathcal{O}}_{i}\right\}  ,\left\{  {\mathcal{A}}_{i}\right\}  ,\mathcal{P},\mathcal{G},T}\right\rangle  .N$ is the number of agents and $\mathcal{I}$ is the high-level language instruction set. Here, $s \in  \mathcal{S}$ represents the joint state of all agents,and $o \in  \mathcal{O}$ denotes the observation set for all agents. Particularly, ${o}_{i} \in  {\mathcal{O}}_{i}$ is the observation set of agent $i$ ,that captures incomplete environment state information. $a \in  \mathcal{A} = {\mathcal{A}}_{1} \times  {\mathcal{A}}_{2}\cdots {\mathcal{A}}_{N}$ represents the joint action space. The joint action space comprises of different categories of high-level actions $\mathcal{A} = {\mathcal{A}}_{NAV} \cup  {\mathcal{A}}_{INT} \cup  {\mathcal{A}}_{EXP}$ , where ${\mathcal{A}}_{NAV}$ is the joint navigation action set, ${\mathcal{A}}_{INT}$ is the joint interaction actions which allow the agents to interact with objects,and ${\mathcal{A}}_{EXP}$ are the joint exploration actions which allow the agents to explore the environment. Examples of the high-level actions include PickUp(object) $\in  {\mathcal{A}}_{INT}$ and NavigateTo(location) $\in  {\mathcal{A}}_{NAV}$ . Each high-level action is associated with a low-level primitive action (pre-trained RL, behavior cloned, or heuristic-based policy). These actions are executed synchronously by all agents at every high-level decision step. $\mathcal{P}\left( {{s}^{\prime } \mid  s,a}\right)$ is the joint transition probability function that defines the probability of arriving at ${s}^{\prime } \in  \mathcal{S}$ after taking joint action $a \in  \mathcal{A}$ in $s \in  \mathcal{S}.\mathcal{G} = \left\{  {{g}_{1},\cdots ,{g}_{k}}\right\}$ defines the subtasks that the agents need to perform to accomplish the language instruction task. $T$ is the length of the planning horizon.
问题设置：我们考虑一种场景，多机器人在类家庭环境中执行一系列任务（a）例如打扫房间或将杂货放进冰箱，和（b）在搜索与救援（SAR）环境中救援失踪人员与扑灭森林火灾。这些任务通常需要长时程规划，涉及约 100 步低级动作以达成目标。我们的目标是为机器人团队计算计划以执行高级语言指令，$I$。我们将这些任务形式化为部分可观测马尔可夫决策过程（POMDP）[55, 56]，记作 $\left\langle  {N,\mathcal{I},\mathcal{S},\left\{  {\mathcal{O}}_{i}\right\}  ,\left\{  {\mathcal{A}}_{i}\right\}  ,\mathcal{P},\mathcal{G},T}\right\rangle  .N$ 是智能体数量且 $\mathcal{I}$ 是高级语言指令集。此处，$s \in  \mathcal{S}$ 表示所有智能体的联合状态，$o \in  \mathcal{O}$ 表示所有智能体的观测集。特别地，${o}_{i} \in  {\mathcal{O}}_{i}$ 是智能体 $i$ 的观测集，捕捉不完全的环境状态信息。$a \in  \mathcal{A} = {\mathcal{A}}_{1} \times  {\mathcal{A}}_{2}\cdots {\mathcal{A}}_{N}$ 表示联合动作空间。联合动作空间包括不同类别的高级动作 $\mathcal{A} = {\mathcal{A}}_{NAV} \cup  {\mathcal{A}}_{INT} \cup  {\mathcal{A}}_{EXP}$，其中 ${\mathcal{A}}_{NAV}$ 是联合导航动作集，${\mathcal{A}}_{INT}$ 是允许智能体与物体交互的联合交互动作，${\mathcal{A}}_{EXP}$ 是允许智能体探索环境的联合探索动作。高级动作示例包括 PickUp(object) $\in  {\mathcal{A}}_{INT}$ 和 NavigateTo(location) $\in  {\mathcal{A}}_{NAV}$。每个高级动作都关联一个低级原语动作（预训练 RL、行为克隆或基于启发式的策略）。这些动作在每个高级决策步由所有智能体同步执行。$\mathcal{P}\left( {{s}^{\prime } \mid  s,a}\right)$ 是联合转移概率函数，定义在采取联合动作 $a \in  \mathcal{A}$ 于 $s \in  \mathcal{S}.\mathcal{G} = \left\{  {{g}_{1},\cdots ,{g}_{k}}\right\}$ 后到达 ${s}^{\prime } \in  \mathcal{S}$ 的概率；$s \in  \mathcal{S}.\mathcal{G} = \left\{  {{g}_{1},\cdots ,{g}_{k}}\right\}$ 定义了智能体为完成语言指令任务需要执行的子任务。$T$ 是规划时域的长度。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_16_58_1d6e65.jpg"/>



Figure 1: An overview of LLaMAR's modular cognitive architecture. LLaMAR leverages LMs within four key modules: Planner, Actor, Corrector, and Verifier, each with specific roles. The Planner breaks down the high-level language instruction into feasible subtasks to achieve the environment goal. The Actor determines the high-level actions each agent should perform. These actions trigger low-level policies that generate and execute a sequence of primitive actions in sync across all agents. Based on execution feedback, the Corrector suggests corrections for high-level actions and the Verifier Module validates completion of subtasks.
图 1：LLaMAR 模块化认知架构概览。LLaMAR 在四个关键模块中利用语言模型：Planner、Actor、Corrector 与 Verifier，各自承担特定角色。Planner 将高级语言指令分解为可行子任务以达成环境目标。Actor 决定每个智能体应执行的高级动作。这些动作触发低级策略，在所有智能体间同步生成并执行原语动作序列。基于执行反馈，Corrector 对高级动作提出修正建议，Verifier 模块验证子任务的完成情况。


Environment: To simulate open-ended, long-horizon tasks that resemble everyday activities in a home-like environment, we use the AI2Thor simulator [41], which supports a diverse set of interactions and photorealistic rendering. Since our approach does not require any parametric training, it can potentially translate to other similar embodied environments like VirtualHome [57], Habitat [42, 58, 59], and ThreeDWorld [60], possibly extending beyond household domains. Similarly, to simulate the search & rescue scenario, we create a custom Search & Rescue environment (SAR). More information about the MAP-THOR and SAR environments can be found in Appendix B and D respectively. We instantiate the problem with $N$ agents cooperating to accomplish a long-horizon rearrangement task [61] in an indoor environment. The agents do not know the objects present in the environment a priori and are encouraged to explore the environment to gather more information to complete the task. Unlike previous solutions in similar settings [22, 19, 20], we do not rely on an oracle/simulator to verify subtask completion. In prior works, a predefined conditional satisfiability check is used as subtask completion feedback.
环境：为模拟类似家居环境中开放式、长时任务，我们使用 AI2Thor 模拟器 [41]，其支持多样交互和光真实渲染。由于我们的方法不依赖任何参数化训练，它有望迁移到其他类似的具身环境，如 VirtualHome [57]、Habitat [42, 58, 59] 和 ThreeDWorld [60]，并可能扩展到家庭领域之外。同样，为模拟搜救情景，我们创建了自定义的搜救环境 (SAR)。有关 MAP-THOR 和 SAR 环境的更多信息分别见附录 B 和 D。我们用 $N$ 个智能体合作，在室内环境中实例化一个长时重排任务 [61]。智能体事先未知环境中存在的物体，需被鼓励探索环境以收集更多信息来完成任务。与类似设定中的先前方法 [22, 19, 20] 不同，我们不依赖神谕/模拟器来验证子任务完成情况。先前工作中，使用预定义的条件可满足性检查作为子任务完成反馈。


## 4 Approach
## 4 方法


We describe our approach in this section. Figure 1 illustrates LLaMAR's architecture comprising four modules: Planner, Actor, Corrector, and Verifier, each an LM with a distinct role. Prior work [62] shows that splitting roles across different LMs improves performance in sequential decision-making. Our initial experiments confirm that LMs tasked with reasoning about multiple inputs and providing long outputs perform poorly. We iterate through these four modules at every high-level decision step. The pseudocode for our approach is in Appendix E We define some key notation below:
本节描述我们的方法。图1 展示了 LLaMAR 的架构，包含四个模块：规划器、执行器、纠正器和验证器，均为具有不同角色的语言模型。先前工作 [62] 表明将不同角色分配给不同语言模型可提升序列决策性能。我们的初步实验也证实，需要同时处理多输入并生成长输出的语言模型表现较差。我们在每个高层决策步骤迭代执行这四个模块。方法伪代码见附录 E。下面定义一些关键符号：


- Memory $\mathcal{M}$ : A textual description of the joint memory of all agents,summarizing past observations, high-level actions, plausible reasons for action failures, and specific subtasks that each agent is attempting to solve.
- 记忆 $\mathcal{M}$ ：对所有智能体联合记忆的文本描述，概述过去观察、高层动作、动作失败的可能原因，以及每个智能体正在尝试解决的具体子任务。


- Open Subtasks ${\mathcal{G}}_{O} \subset  \mathcal{G}$ : Feasible subtasks proposed by the Planner LM to achieve the environment task that are yet to be accomplished by the agents.
- 未完成子任务 ${\mathcal{G}}_{O} \subset  \mathcal{G}$ ：规划器语言模型提出的为完成环境任务而尚未被智能体完成的可行子任务。


- Completed Subtasks ${\mathcal{G}}_{S} \subset  \mathcal{G}$ : Subtasks completed by the agents.
- 已完成子任务 ${\mathcal{G}}_{S} \subset  \mathcal{G}$ ：由智能体完成的子任务。


- Corrective Actions ${a}_{c}$ : Corrective actions for each agent based on failure information from the previous step.
- 纠正动作 ${a}_{c}$ ：基于上一步的失败信息为每个智能体提供的纠正动作。


At the start of each episode,Memory $\mathcal{M}$ ,Open Subtasks ${\mathcal{G}}_{O}$ ,Completed Subtasks ${\mathcal{G}}_{S}$ ,Actions $a$ , Corrective Actions ${a}_{c}$ ,and Failure Information $\mathcal{F}$ are initialized as empty sets.
在每次试验开始时，记忆 $\mathcal{M}$、未完成子任务 ${\mathcal{G}}_{O}$、已完成子任务 ${\mathcal{G}}_{S}$、动作 $a$、纠正动作 ${a}_{c}$ 和失败信息 $\mathcal{F}$ 均初始化为空集。


Consider an example of a kitchen with groceries, a fridge, and a counter. Two agents are tasked with "Fetch the groceries and place them in the fridge". This example will help illustrate the utility of each module. All LMs receive a language task instruction $\mathcal{I}$ ,joint observations from all agents, and information about open and completed subtasks and memory unless stated otherwise. We next discuss the various components in our architecture in detail:
考虑一个有杂货、冰箱和操作台的厨房示例。两个智能体的任务是“取走杂货并放入冰箱”。此示例有助于说明各模块的作用。除非另有说明，所有语言模型均接收语言任务指令 $\mathcal{I}$、来自所有智能体的联合观察，以及关于未完成和已完成子任务与记忆的信息。下面详细讨论我们架构的各个组成部分：


Planner Module The Planner LM module suggests feasible subtasks to ensure the completion of the environment task. This method, similar to SmartLLM's [19] zero-shot planning, uses only observations in the agents' field of view. The Planner suggests subtasks related to objects seen in the current observation or memory of all the agents. For the example considered, it decomposes the task into subtasks like "transport the tomato to the fridge" and "transport the lettuce to the fridge", which are added to ${\mathcal{G}}_{O}$ .
规划器模块 规划器语言模型提出可行子任务以确保环境任务的完成。该方法类似于 SmartLLM [19] 的零样本规划，仅使用智能体视野内的观察。规划器建议与当前观察或所有智能体记忆中看到的物体相关的子任务。对于该示例，它将任务分解为诸如“把番茄运到冰箱”和“把生菜运到冰箱”的子任务，并将其加入到 ${\mathcal{G}}_{O}$ 中。


Actor Module The Actor LM additionally uses corrective actions suggested by the Corrector module in the previous time step to predict high-level actions for the current step. These actions are then executed in the environment to progress a subset of subtasks in the open subtask set ${\mathcal{G}}_{O}$ and accordingly updates the joint memory. For instance, the Actor module might suggest actions such as $a =$ [Pickup(Tomato),NavigateTo(Lettuce)],updating memory with "We saw a tomato on the counter-top, Alice is picking up the tomato, and Bob is navigating to the lettuce".
执行器模块 执行器语言模型还利用纠正器模块在上一步建议的纠正动作来预测当前步骤的高层动作。然后在环境中执行这些动作以推进未完成子任务集 ${\mathcal{G}}_{O}$ 中的一部分子任务，并相应更新联合记忆。例如，执行器模块可能建议诸如 $a =$ [Pickup(Tomato),NavigateTo(Lettuce)] 的动作，并将记忆更新为“我们在操作台上看到一个番茄，Alice 正在拿起番茄，Bob 正在前往生菜处”。


Corrector Module The Corrector LM self-corrects high-level actions suggested by the Actor LM after controller failures in the previous step's execution ${}^{2}$ It suggests corrective high-level actions and provides reasons for failures and chosen corrections. For example, it might suggest "The action of picking up the tomato failed because it is too far away. Alice first needs to navigate closer to the tomato."; ${a}_{c} =$ [NavigateTo(Tomato),None].
更正器模块 Corrector LM 在上一步执行因控制器失败后会对 Actor LM 建议的高层行动进行自我修正 ${}^{2}$ 它提出纠正性的高层行动并说明失败原因及所选纠正措施。例如，它可能建议“拾取番茄的动作失败，因为番茄太远。Alice 需要先靠近番茄再操作。”； ${a}_{c} =$ [NavigateTo(Tomato),None].


Verifier Module After executing high-level actions, the Verifier LM assesses whether these actions have completed any subtasks in the open subtask set. Successful subtasks are moved to the completed subtask set. Without the Verifier LM, the method would need to rely on the simulator/oracle for success or failure information. The Verifier LM along with other information uses the successfully executed high-level actions proposed by the Actor LM to predict subtask completion. For example, after transporting the lettuce to the fridge, the Verifier updates the completed subtasks with "transport lettuce to the fridge".
验证器模块 在执行高层行动后，Verifier LM 评估这些行动是否完成了开放子任务集中的任何子任务。完成的子任务被移至已完成子任务集。若无 Verifier LM，此方法将需要依赖模拟器/预言机来获取成功或失败信息。Verifier LM 结合其他信息使用 Actor LM 成功执行的高层行动来预测子任务完成情况。例如，在将生菜运到冰箱后，Verifier 会将“将生菜运到冰箱”加入已完成子任务。


Admissible Action parsing with Semantic Translation When LMs generate action plans, natural language outputs often fail to translate to executable high-level actions. This happens when the output does not match the predefined format or refers to unrecognized contextually similar objects. We use a cosine similarity method from [13], fine-tuning a pre-trained sentence-BERT [63] to transform the free-form text into admissible high-level actions. Hyperparameters and additional details of the sentence transformer fine-tuning are provided in Appendix I
可接受动作解析与语义翻译 当语言模型生成行动计划时，自然语言输出常无法转化为可执行的高层动作。此类情况发生在输出不符合预定义格式或引用了未识别的语境相似对象时。我们使用文献[13]中的余弦相似度方法，微调预训练的 sentence-BERT [63]，将自由文本转换为可接受的高层动作。sentence transformer 微调的超参数及更多细节见附录 I


Exploration Strategy In unexplored environments, agents need to search for task-relevant objects. If agents cannot find the required objects, the language model can choose an 'exploration' action ${a}_{exp}$ . We use a semantically-guided heuristic to determine the choice of region to be explored. The agent rotates to four cardinal directions $d \in$ North,South,East,West,capturing image observations ${o}_{n,d}$ . These images are processed through a pre-trained CLIP image encoder [64] to obtain embeddings ${I}_{d}$ . The list of open subtasks ${\mathcal{G}}_{O}$ is processed through the corresponding CLIP text encoder to get text embeddings ${g}_{O,i}$ . The exploration score ${\mathcal{E}}_{d}$ in direction $d$ is defined as ${\mathcal{E}}_{d} = \mathop{\sum }\limits_{{i = 1}}^{\left| {\mathcal{G}}_{O}\right| }\frac{{g}_{O,i} \cdot  {I}_{d}}{\begin{Vmatrix}{g}_{O,i}\end{Vmatrix}\begin{Vmatrix}{I}_{d}\end{Vmatrix}}$ . The direction with the highest score ${d}^{ * } = \arg \mathop{\max }\limits_{d}{\mathcal{E}}_{d}$ is chosen. Summing the scores helps select the best direction to explore in expectation. The agent rotates towards ${d}^{ * }$ and moves $J = 2$ steps,repeating this process $K = 3$ times in one explore action. This approach ensures that images relevant to identifying potential subtasks are prioritized. For example,if ${\mathcal{G}}_{O}$ includes "locate a computer", it is more likely to find a computer on a table than on a sofa, resulting in a higher cosine similarity score between the subtask CLIP text embedding and table CLIP image embedding. Refer to Appendix B. 2 for more details about the exploration heuristic.
探索策略 在未探索的环境中，智能体需要搜索与任务相关的物体。如果智能体找不到所需物体，语言模型可选择“探索”动作 ${a}_{exp}$ 。我们使用语义引导的启发式方法来确定要探索的区域。智能体朝四个基准方向旋转 $d \in$ 北、南、东、西，捕获图像观测 ${o}_{n,d}$ 。这些图像经预训练的 CLIP 图像编码器 [64] 处理以获得嵌入向量 ${I}_{d}$ 。开放子任务列表 ${\mathcal{G}}_{O}$ 通过相应的 CLIP 文本编码器处理以获取文本嵌入 ${g}_{O,i}$ 。方向 $d$ 的探索得分 ${\mathcal{E}}_{d}$ 定义为 ${\mathcal{E}}_{d} = \mathop{\sum }\limits_{{i = 1}}^{\left| {\mathcal{G}}_{O}\right| }\frac{{g}_{O,i} \cdot  {I}_{d}}{\begin{Vmatrix}{g}_{O,i}\end{Vmatrix}\begin{Vmatrix}{I}_{d}\end{Vmatrix}}$ 。选择得分最高的方向 ${d}^{ * } = \arg \mathop{\max }\limits_{d}{\mathcal{E}}_{d}$ 。对得分求和有助于期望下选择最佳探索方向。智能体朝 ${d}^{ * }$ 旋转并移动 $J = 2$ 步，重复该过程 $K = 3$ 次作为一次探索动作。这种方法确保优先处理有助于识别潜在子任务的图像。例如，若 ${\mathcal{G}}_{O}$ 包含“定位一台电脑”，则在桌子上比在沙发上更有可能找到电脑，从而使子任务的 CLIP 文本嵌入与桌子 CLIP 图像嵌入之间的余弦相似度更高。更多关于探索启发式的细节见附录 B.2。


---



${}^{2}$ We use the simulator just to provide a boolean value about the success of high-level action execution.
${}^{2}$ 我们仅使用模拟器来提供关于高层动作执行成功与否的布尔值。


---



Motivation for Proposed Framework The specific order of the modules in LLaMAR is due to natural causal relationships in which environment feedback is received. We use the Planner as the first module because it allows LLaMAR to come up with an initial list of open subtasks that could be completed based on the current observation and past memory to satisfy the task. This list serves as a rough high-level plan. The actor then uses this information to suggest the necessary actions. The Corrector is used after the Actor module to identify reasons for failures in the execution of the actions suggested by the Actor . Note that the failure module is inert and only suggests corrective actions. Only the Actor module decides the final actions to be executed. This role distinction allows for clear reasoning on failures when they occur and lets the actor module focus on choosing actions. The Verifier is used after the action is executed to update the list of closed subtasks so that LLaMAR can be current with the progress toward the completion of the environment task. This allows the planner to update the list of open subtasks in the next step. In essence, the Planner and the Verifier ensure that the progress of the agents is tracked and the actor and the corrector ensure that the actions are executed successfully to advance towards completion of the task.
提出框架的动机 LLaMAR 中模块的特定顺序来源于接收环境反馈的自然因果关系。我们将 Planner 放在首位，因为它能基于当前观察和过去记忆提出可完成的初步未完成子任务列表，作为粗略的高层计划。Actor 随后利用这些信息提出必要的动作。Corrector 在 Actor 之后用于识别 Actor 建议的动作执行失败的原因。注意，失败模块是惰性的，仅建议纠正措施，只有 Actor 决定最终执行的动作。这样的角色划分使得在发生失败时能进行清晰推理，并让 Actor 专注于选择动作。Verifier 在动作执行后使用，用于更新已完成子任务列表，使 LLaMAR 能反映环境任务的进展，从而在下一步让 Planner 更新未完成子任务列表。本质上，Planner 和 Verifier 跟踪代理的进度，Actor 和 Corrector 确保动作被成功执行以推进任务完成。


Multi-Agent features in LLaMAR While our method can be easily adapted to a single-agent setting, our design choice for the architecture was motivated to include the following multi-agent features:
LLaMAR 的多智能体特性 虽然我们的方法可以很容易地适配到单智能体场景，但我们在架构上的设计选择旨在包含以下多智能体特性：


- Coordination through communication: Agents share their state information with the centralized LLaMAR modules to predict actions, enabling them to coordinate and avoid conflicts. This information sharing allows for the agents to cooperate and achieve the collective goal.
- 通过通信进行协调：智能体与集中式 LLaMAR 模块共享状态信息以预测动作，使其能够协调并避免冲突。此信息共享使智能体能够合作以实现集体目标。


- Dynamic Role Assignment: Agents are dynamically assigned roles based on the current task requirements and their capabilities. This flexibility allows LLaMAR to adapt to changing environments and task demands.
- 动态角色分配：根据当前任务需求和各自能力动态分配角色。该灵活性使 LLaMAR 能适应变化的环境和任务要求。


- Hierarchical Task Decomposition: To handle the complexity of multi-agent planning, LLaMAR decomposes the action space by creating specific subgoals/subtasks available for any agent to assign itself (done by the actor module) based on the observation and current context. This decomposition reduces the overall search space and improves planning efficiency.
- 分层任务分解：为处理多智能体规划的复杂性，LLaMAR 通过创建特定子目标/子任务来分解动作空间，任何智能体可根据观察和当前上下文（由 Actor 完成）将其分配给自己。该分解减少了整体搜索空间并提高了规划效率。


## 5 Experiments
## 5 实验


MAP-THOR: To evaluate the performance of LLaMAR and benchmark other baseline methods, we create a benchmark dataset of tasks which we call MAP-THOR (Multi-Agent Planning tasks in AI2-THOR). While Smart-LLM [19] introduces a dataset of 36 tasks within AI2-Thor [41] classified by complexity, their tasks are limited to single floor plans. This limitation hinders testing the robustness of planners across different room layouts. Additionally, some tasks in their dataset cannot be performed by multiple agents, regardless of task division, such as Pick up the pillow, Open the laptop to turn it on, and Turn off the lamp.
MAP-THOR：为评估 LLaMAR 的性能并对比其他基线方法，我们创建了一个任务基准数据集，称为 MAP-THOR（AI2-THOR 中的多智能体规划任务）。尽管 Smart-LLM [19] 在 AI2-Thor [41] 中引入了按复杂度分类的 36 个任务数据集，但其任务被限制在单层平面上。这一限制阻碍了对不同房间布局下规划器鲁棒性的测试。此外，他们的数据集中有些任务无论如何划分任务都无法由多名智能体完成，例如捡起枕头、打开笔记本电脑并开机、以及关灯。


By contrast, MAP-THOR includes tasks solvable by both single and multiple agents. We classify the tasks into four categories based on the ambiguity of the language instructions. To test the planner robustness, we provide five different floor plans for each task. We also include automatic checker modules to verify subtask completion and evaluate plan quality. Our dataset comprises 45 tasks, each defined for five distinct floor plans, ensuring comprehensive testing and evaluation.
相比之下，MAP-THOR 包含可由单智能体或多智能体解决的任务。我们根据语言指令的歧义性将任务分为四类。为测试规划器的鲁棒性，我们为每个任务提供五种不同的户型图。我们还包括自动检查模块以验证子任务完成并评估规划质量。我们的数据集包含 45 个任务，每个任务为五个不同户型定义，确保全面的测试与评估。


We conduct experiments with tasks of varying difficulty levels, where an increase in difficulty of the tasks corresponds to an increased ambiguity in the language instructions. The complete task list of each category can be found in the Appendix C
我们进行了不同难度等级的任务实验，任务难度增加对应于语言指令歧义性的增加。每类的完整任务列表见附录 C


- Explicit item type, quantity, and target location: Agents are explicitly instructed to transport specific items to specific target locations. For example, put bread, lettuce, and a tomato in the fridge clearly defines the objects (tomato, lettuce, bread) and the target (fridge).
- 明确的物品类型、数量和目标位置：明确指示智能体将特定物品运送到特定目标位置。例如，put bread, lettuce, and a tomato in the fridge 明确了对象（tomato, lettuce, bread）和目标（fridge）。


- Explicit item type and target location but implicit item quantity: The object type is explicitly described, but its quantity is not disclosed. For example, Put all the apples in the fridge. Agents must explore the environment to locate all specified items and also predict when to stop.
- 明确物品类型和目标位置但隐含数量：物品类型被明确描述，但数量未披露。例如，Put all the apples in the fridge。智能体必须探索环境以找到所有指定物品并判断何时停止。


- Explicit target location but implicit item types and quantity: The target location is explicitly defined but the item types and their quantities are concealed. For example, Put all groceries in the fridge.
- 明确目标位置但隐含物品类型和数量：目标位置被明确，但物品类型及其数量被隐去。例如，Put all groceries in the fridge。


- Implicit target location, item type, and quantity: Item types and their quantities along with the target location are implicitly defined. For example, Clear the floor by placing the items at their appropriate positions. The agent is expected to place items like pens, books, and laptops on the study table, and litter in the trash can.
- 隐含目标位置、物品类型和数量：物品类型和数量以及目标位置均以隐含方式给出。例如，Clear the floor by placing the items at their appropriate positions。期望智能体将笔、书和笔记本等物品放在书桌上，将垃圾放入垃圾桶。


Search & Rescue Environment (SAR): To showcase the effectiveness of LLaMAR with respect to explicit coordination in multi-agent settings, we evaluate LLaMAR in a partially observable search & rescue and fire relief environment in a grid world. Depending on the scene, there is a mix of missing people to be found, and wildfires to be stopped before they spread geographically. More details about the environment can be found in Appendix D.
搜救与救火环境（SAR）：为展示 LLaMAR 在多智能体场景下进行明确协调的有效性，我们在一个部分可观测的网格世界中评估 LLaMAR 的搜救与灭火救援环境。根据场景，有待发现的失踪人员，以及需要在蔓延前扑灭的野火。关于该环境的更多细节见附录 D。


- Fire Extinguishing: Fires consist of expansive flammable regions with a fixed set of sources that propagate over time. The rate of fire spread is proportional to its intensity; higher intensities result in faster spread. Fires are categorized as either Class A or Class B, which are extinguished using water or sand, respectively. These extinguishing resources are sourced from reservoirs distributed across the environment.
- 灭火：火灾由可扩展的可燃区域和随时间扩散的一组固定火源组成。火势蔓延速度与其强度成正比；强度越高，蔓延越快。火灾分为 A 类或 B 类，分别用水或沙子扑灭。灭火资源来自分布在环境中的水源/沙坑。


- Human Rescue: Each individual is initially located at an unknown position within the environment. The objective is to locate, rescue, and transport the individuals to a designated drop-off location, which is known beforehand. Transporting a person requires the coordinated effort of two agents simultaneously, who must carry them to the specified drop-off point.
- 人员救援：每名个体最初位于环境中未知位置。目标是定位、救助并将其运送到预先已知的指定投放点。搬运一人需要两名智能体同时协调合作，将其运至指定投放点。


## Metrics
## 指标


We evaluate the algorithms using the following metrics to compare their performances on the tasks:
我们使用以下指标评估算法，以比较它们在各任务上的表现：


- Success Rate (SR): The fraction of episodes in which all subtasks are completed. Success equals 1 if all subtasks are successfully executed in an episode, otherwise it is 0.
- 成功率 (SR)：在所有子任务均完成的回合中所占的比例。若一回合中所有子任务均成功执行，则成功为 1，否则为 0。


- Transport Rate (TR): The fraction of subtasks completed within an episode, provides a finer granularity of task completion.
- 运输率 (TR)：一回合内完成的子任务比例，提供更细粒度的任务完成度度量。


- Coverage (C): The fraction of successful interactions with target objects. It is useful to verify if the LMs can infer the objects to interact with, in scenarios where the tasks have objects that are specified implicitly.
- 覆盖率 (C)：与目标对象成功交互的比例。当任务中对象是隐式指定时，该指标有助于验证语言模型能否推断出应交互的对象。


- Balance (B): The ratio between the minimum and maximum number of successful high-level actions executed by any agent that contributed towards task completion. We only check for a subset of high-level actions that must be executed to accomplish critical subtasks that lead to the successful completion of the language instruction task. If each agent $i$ out of $n$ agents completes ${s}_{i}$ successful tasks,the balance is defined as: $B \mathrel{\text{ := }} \frac{\min \left\{  {{s}_{1},\cdots ,{s}_{n}}\right\}  }{\max \left\{  {{s}_{1},\cdots ,{s}_{n}}\right\}   + \epsilon }$ . This measures how evenly the work is distributed among agents. A balance of zero indicates at least one agent performed no successful high-level actions, while a balance of one indicates all agents performed the same number of successful high-level actions. Here $\epsilon  = {1e} - 4$ is a small number to avoid division by zero.
- 平衡度 (B)：对为完成任务做出贡献的任一智能体执行的成功高层次动作数量的最小值与最大值之比。我们仅检查必须执行以完成关键子任务并导致语言指令任务成功完成的部分高层次动作。如果每个智能体 $i$ 出自 $n$ 名智能体完成了 ${s}_{i}$ 个成功任务，则平衡度定义为： $B \mathrel{\text{ := }} \frac{\min \left\{  {{s}_{1},\cdots ,{s}_{n}}\right\}  }{\max \left\{  {{s}_{1},\cdots ,{s}_{n}}\right\}   + \epsilon }$ 。该指标衡量工作在智能体间的分配均匀性。平衡度为零表示至少有一名智能体没有执行成功的高层次动作，而平衡度为一表示所有智能体执行的成功高层次动作数量相同。此处 $\epsilon  = {1e} - 4$ 是为避免除以零而设的小数值。


- Average steps (L): The number of high-level actions taken by the team to complete the task, capped at $L = {30}$ in our experiments. If the task is not completed within $L$ steps,the episode is deemed a failure. Note that the metric $L$ is presented in the table providing the complete results, located in Appendix F.
- 平均步数 (L)：团队完成任务所采取的高层次动作数，在我们的实验中以 $L = {30}$ 为上限。如果任务未在 $L$ 步内完成，则回合被视为失败。注意表格中呈现了指标 $L$ 的完整结果，位于附录 F。


For all the metrics, we report the means along with the 95% confidence interval across all the tasks (refer Appendix F for complete results). Since SR is a binomial metric, we report the Clopper-Pearson Interval as the confidence interval.
对于所有指标，我们报告各任务上的均值及 95% 置信区间（完整结果见附录 F）。由于 SR 为二项指标，我们使用 Clopper-Pearson 区间作为置信区间。


## Baselines
## 基线方法


For a fair comparison with our method, we make modifications to the baselines to make them work in partially observable settings with limited reliance on the simulator. More details about implementations can be found in Appendix H
为与我们的方法公平比较，我们对基线方法进行了修改，使其能在部分可观测的设置中工作，并减少对模拟器的依赖。实现细节见附录 H。


- Act: We query the LLM with the task and the observations to suggest a high-level action.
- Act：我们向大模型提供任务和观测以建议一个高层次动作。


- Chain-of-Thought [65]: We modify the Act prompt with a chain-of-thought style addendum to let the LM reason about the possible implications while selecting a high-level action.
- 连锁思维（Chain-of-Thought）[65]：我们在 Act 提示中加入连锁思维风格的附注，允许模型在选择高层次动作时对可能的含义进行推理。


- ReAct [66]: We use a ReAct-style prompting to let the LMs reason after suggesting high-level actions and possibly suggest ways to correct any failures.
- ReAct [66]：我们使用 ReAct 风格的提示，使模型在建议高层次动作后进行推理，并可能建议纠正失败的方法。


- SmartLLM [19]: We modify the official codebase to only include information from the local observations of the agents instead of assuming full observability.
- SmartLLM [19]：我们修改官方代码库，仅包含来自智能体局部观测的信息，而不假设全局可观测性。


- CoELA [22]: We modify the list of available high-level actions to include all possible valid combinations of actions with interactable objects in the agent's local observation. As the scene becomes more cluttered, this list and the prompt becomes combinatorially longer. In the original implementation, the list of available actions is filtered based on the feasibility of the actions as suggested by a conditional checker.
- CoELA [22]：我们修改了可用高层动作列表，使其包含与智能体局部观察中可交互对象的所有可能有效动作组合。随着场景变得更杂乱，该列表和提示将组合性地变长。在原始实现中，可用动作列表基于条件检查器建议的动作可行性进行过滤。


It should be noted that Act, Chain-of-Thought, ReAct, and SmartLLM are all CMAS frameworks where CoELA follows the DMAS framework.
应当指出，Act、Chain-of-Thought、ReAct 和 SmartLLM 都属于 CMAS 框架，而 CoELA 遵循 DMAS 框架。


## 6 Results and Discussion
## 6 结果与讨论


Choice of the underlying LM: To understand the impact of the underlying LM's quality on decision-making, we initially experimented with different LMs on MAP-THOR. Specifically, we utilize both the language-only and vision-language models of GPT-4 [67], IDEFICS-2 [68, 69], LLaVA [70, 71], and CoGVLM [72]. Among these, GPT-4, when used solely with text inputs, exhibits the poorest performance. This is attributed to the agents' inability to reason about visual observations, which is particularly detrimental for the Corrector module. Substituting GPT-4V with other vision-language models results in a decline in performance (refer Table 2) and hence we use GPT-4V as the underlying VLM while comparing to the baselines.
底层语言模型的选择：为了解底层语言模型质量对决策的影响，我们最初在 MAP-THOR 上对不同模型进行了试验。具体而言，我们使用了 GPT-4 [67]、IDEFICS-2 [68, 69]、LLaVA [70, 71] 和 CoGVLM [72] 的纯语言与视觉-语言版本。其中，仅用文本输入的 GPT-4 表现最差，这是因为代理无法对视觉观测进行推理，这对 Corrector 模块尤其不利。将 GPT-4V 替换为其他视觉-语言模型会导致性能下降（参见表 2），因此在与基线比较时我们使用 GPT-4V 作为底层 VLM。


Baseline Comparisons: Table 2 compares our method, LLaMAR, with other baselines in a 2-agent scenario using GPT-4V as the underlying VLM. Act and ReAct show similar performance, with Act struggling due to its lack of strategic planning or correction, and ReAct performing slightly better by dynamically adjusting actions based on reasoning on immediate feedback. CoT's performance declines with longer planning horizons due to its inability to maintain coherence over extended planning sequences, consistent with findings in [73], showing its effectiveness only with highly specific prompts. SmartLLM, operating in a plan-and-execute paradigm, generates impractical plans with issues like infinite loops and failure to handle low-level action failures, leading to lower success rates and poor transport metrics. It also tends to hallucinate objects. CoELA, using a decentralized multi-agent system (DMAS), performs poorly due to large input prompts and struggles to select the correct action from numerous choices. Its decentralized decision-making is less efficient than the centralized multi-agent system (CMAS) used by LLaMAR. Previous research [16] confirms CMAS frameworks are more effective than DMAS frameworks. Overall, our method, LLaMAR, benefits from its modular cognitive architecture, which integrates planning, acting, correcting, and verifying through distinct LLM roles, resulting in superior performance across various evaluation metrics. By avoiding reliance on privileged information and incorporating a robust exploration strategy that allows it to scout for objects that are not initially visible, LLaMAR ensures higher success rates and balanced task execution among agents.
基线比较：表 2 比较了在 2 代理场景中以 GPT-4V 为底层 VLM 时我们的方法 LLaMAR 与其他基线的表现。Act 与 ReAct 表现相近，Act 因缺乏战略性规划或纠正而困难重重，ReAct 通过基于对即时反馈的推理动态调整动作略有更好。CoT 随着规划视野变长表现下降，因其无法在长序列规划中保持连贯性，这与 [73] 的发现一致，CoT 仅在高度具体的提示下有效。SmartLLM 采用“计划-执行”范式，生成不切实际的计划，存在无限循环和无法处理低级动作失败等问题，导致成功率低且传输指标差，同时还倾向于产生对象幻觉。CoELA 使用去中心化多智能体系统（DMAS），由于输入提示过大且难以从众多选项中选择正确动作而表现不佳。其去中心化决策效率低于 LLaMAR 使用的集中式多智能体系统（CMAS）。先前研究 [16] 证实 CMAS 框架比 DMAS 更有效。总体而言，我们的方法 LLaMAR 受益于其模块化认知架构，通过不同的 LLM 角色整合规划、执行、纠正与验证，在各项评估指标上表现优异。通过避免依赖特权信息并引入允许搜寻初始不可见对象的强健探索策略，LLaMAR 确保了更高的成功率和代理间更均衡的任务执行。


Roles of different modules in LLaMAR: To demonstrate the effectiveness of the various modules in our cognitive architecture, we performed ablation studies by evaluating performance metrics with each module removed individually. The results are summarized in Table 3 Using only the Actor module corresponds to the "Act" baseline, which demonstrates its fundamental capabilities in isolation but shows limited effectiveness without planning and correction due to relatively lower success and transport rates. Adding the Planner and Verifier modules improves performance, benefiting from better task planning and validation, increasing the overall SR and TR, and ensuring more effective task completion and even work distribution, as indicated by the increase in balance (B). However, in scenarios where the suggested action fails, the actor suggests the same action in the next decision step since it is not able to reason why the action failed until the end of the planning horizon. Incorporating the Corrector module, with access to privileged information from an environment oracle, significantly boosts performance, enhancing the SR, TR, C, and further improving B, consistent with the findings in [17]. This highlights the Corrector module's importance in adjusting actions based on controller feedback, resulting in higher task success and more efficient task completion, albeit with reliance on oracle knowledge. Finally, the complete LLaMAR system, without privileged information, achieves SR, TR, C, and B values close to those of the oracle setup. This demonstrates the system's robustness and effectiveness in a realistic setting. The Corrector module plays a crucial role in enabling agents to learn from past failures and avoid repeating actions, preventing task failures due to timeout. Despite lacking oracle knowledge, LLaMAR performs nearly as well as the oracle-enhanced setup. These results highlight the importance of each module in our cognitive architecture. Removing any module diminishes effectiveness.
LLaMAR 中不同模块的作用：为展示我们认知架构中各模块的有效性，我们通过逐一移除各模块进行消融实验并评估性能指标。结果汇总于表 3。仅使用 Actor 模块对应于“Act”基线，展示了其独立时的基本能力，但由于缺乏规划与纠正，成功率和传输率相对较低。加入 Planner 与 Verifier 模块提升了性能，受益于更好的任务规划与验证，整体 SR 与 TR 增加，并通过平衡性（B）的提升实现更有效的任务完成与更均匀的工作分配。然而在建议动作失败的场景中，actor 在下一个决策步仍会建议相同动作，因为其无法在规划视野结束前推理出动作失败的原因。引入具有来自环境神谕特权信息访问的 Corrector 模块显著提升了性能，增强了 SR、TR、C 并进一步改善 B，这与 [17] 的发现一致。这凸显了 Corrector 模块在基于控制器反馈调整动作方面的重要性，从而提高任务成功率并更高效地完成任务，尽管依赖于神谕知识。最后，完整的 LLaMAR 系统在没有特权信息的情况下取得了接近具神谕设置的 SR、TR、C 和 B 值，这证明了系统在现实环境中的鲁棒性与有效性。Corrector 模块在使代理从过去失败中学习并避免重复动作方面起到关键作用，防止因超时导致的任务失败。尽管缺乏神谕知识，LLaMAR 的表现几乎可与增强神谕的设置相媲美。这些结果强调了我们认知架构中每个模块的重要性，移除任何模块都会降低效能。


Increasing the number of agents Increasing the number of agents in the environment shows distinct trends in our method's performance metrics for both MAP-THOR and SAR environments (refer Table 4). In MAP-THOR, with two agents, we establish a solid baseline for success rate (SR) and transport rate (TR), which is similarly reflected in the SAR environment. Adding a third agent improves both SR and TR in both environments, indicating enhanced task completion and transportation efficiency. Coverage (C) also increases, suggesting better exploration and interaction with objects across both environments. There is a slight decrease in SR and TR when the number of agents increases from 3 to 5 in the MAP-THOR environment. The decrease in these metrics can be attributed to the rooms in MAP-THOR becoming crowded with 4 and 5 agents hence blocking the agents from navigating without colliding with other agents. But this phenomenon is not seen in the SAR environment which is comparatively more spacious and navigable. However, balance (B), which measures the even distribution of tasks among agents, decreases with more agents. This drop highlights the challenge of ensuring equal contributions from all agents in a larger multi-agent system. While SR remains high, the balance metric drops significantly from 2 to 5 agents, indicating some agents do more work than others. In summary, adding more agents improves task performance and efficiency but introduces challenges in maintaining balanced contributions. Addressing this imbalance is crucial for refining multi-agent planning algorithms.
增加智能体数量 在环境中增加智能体数量，在我们的评估指标上对 MAP-THOR 和 SAR 环境都表现出明显趋势（参见表 4）。在 MAP-THOR 中，两个智能体时我们建立了成功率（SR）和运输率（TR）的稳固基线，SAR 环境中有类似表现。增加第三个智能体后，两个环境中的 SR 和 TR 都有所提升，表明任务完成度和运输效率提高。覆盖率（C）也增加，说明在两个环境中对物体的探索和交互更好。当 MAP-THOR 中智能体数量从 3 增加到 5 时，SR 和 TR 略有下降。这些指标的下降可归因于在 4 或 5 个智能体时房间变得拥挤，导致智能体在不与其他智能体碰撞的情况下难以导航。但在相对更宽敞、可导航的 SAR 环境中未出现该现象。然而，衡量任务在智能体间均匀分配的平衡度（B）随着智能体增多而下降。该下降凸显了在更大的多智能体系统中确保均等贡献的挑战。尽管 SR 仍然较高，但平衡度从 2 到 5 个智能体时显著下降，表明部分智能体承担了更多工作。总之，增加智能体可提升任务性能和效率，但会带来维持贡献平衡的挑战。解决该不平衡对于改进多智能体规划算法至关重要。


<table><tr><td>Algorithm</td><td>LM</td><td>SR↑</td><td>TR↑</td><td>C↑</td><td>B↑</td></tr><tr><td>Act</td><td>GPT-4V</td><td>0.33</td><td>0.67</td><td>0.91</td><td>0.59</td></tr><tr><td>ReAct</td><td>GPT-4V</td><td>0.34</td><td>0.72</td><td>0.92</td><td>0.67</td></tr><tr><td>CoT</td><td>GPT-4V</td><td>0.14</td><td>0.59</td><td>0.87</td><td>0.62</td></tr><tr><td>SmartLLM</td><td>GPT-4V</td><td>0.11</td><td>0.23</td><td>0.91</td><td>0.45</td></tr><tr><td>CoELA</td><td>GPT-4V</td><td>0.25</td><td>0.46</td><td>0.76</td><td>0.73</td></tr><tr><td>LLaMAR</td><td>GPT-4</td><td>0.51</td><td>0.85</td><td>0.95</td><td>0.83</td></tr><tr><td>LLaMAR</td><td>LLaVA</td><td>0.54</td><td>0.84</td><td>0.91</td><td>0.75</td></tr><tr><td>LLaMAR</td><td>IDEFICS-2</td><td>0.57</td><td>0.86</td><td>0.94</td><td>0.78</td></tr><tr><td>LLaMAR</td><td>CogVLM</td><td>0.61</td><td>0.89</td><td>0.95</td><td>0.80</td></tr><tr><td>LLaMAR (w/o expl)</td><td>GPT-4V</td><td>0.62</td><td>0.87</td><td>0.95</td><td>0.82</td></tr><tr><td>LLaMAR (w/ expl)</td><td>GPT-4V</td><td>0.66</td><td>0.91</td><td>0.97</td><td>0.82</td></tr></table>
<table><tbody><tr><td>算法</td><td>语言模型</td><td>SR↑</td><td>TR↑</td><td>C↑</td><td>B↑</td></tr><tr><td>动作</td><td>GPT-4V</td><td>0.33</td><td>0.67</td><td>0.91</td><td>0.59</td></tr><tr><td>ReAct</td><td>GPT-4V</td><td>0.34</td><td>0.72</td><td>0.92</td><td>0.67</td></tr><tr><td>链式思维</td><td>GPT-4V</td><td>0.14</td><td>0.59</td><td>0.87</td><td>0.62</td></tr><tr><td>SmartLLM</td><td>GPT-4V</td><td>0.11</td><td>0.23</td><td>0.91</td><td>0.45</td></tr><tr><td>CoELA</td><td>GPT-4V</td><td>0.25</td><td>0.46</td><td>0.76</td><td>0.73</td></tr><tr><td>LLaMAR</td><td>GPT-4</td><td>0.51</td><td>0.85</td><td>0.95</td><td>0.83</td></tr><tr><td>LLaMAR</td><td>LLaVA</td><td>0.54</td><td>0.84</td><td>0.91</td><td>0.75</td></tr><tr><td>LLaMAR</td><td>IDEFICS-2</td><td>0.57</td><td>0.86</td><td>0.94</td><td>0.78</td></tr><tr><td>LLaMAR</td><td>CogVLM</td><td>0.61</td><td>0.89</td><td>0.95</td><td>0.80</td></tr><tr><td>LLaMAR（无解释）</td><td>GPT-4V</td><td>0.62</td><td>0.87</td><td>0.95</td><td>0.82</td></tr><tr><td>LLaMAR（含解释）</td><td>GPT-4V</td><td>0.66</td><td>0.91</td><td>0.97</td><td>0.82</td></tr></tbody></table>


Table 2: Comparison of evaluation metrics against baselines averaged across all tasks for the 2-agent MAP-THOR scenarios. The complete table with confidence intervals can be found in Appendix F More details about peculiar behaviors for the baselines can be found in Appendix H
表 2：与基线相比的评估指标，在所有任务上对 2-agent MAP-THOR 场景取平均。完整带置信区间的表见附录 F，有关基线异常行为的更多细节见附录 H


<table><tr><td>Modules Used</td><td>SR $\uparrow$</td><td>TR↑</td><td>C↑</td><td>B↑</td></tr><tr><td>Actor</td><td>0.33</td><td>0.67</td><td>0.91</td><td>0.59</td></tr><tr><td>Planner + Actor + Verifier</td><td>0.45</td><td>0.78</td><td>0.92</td><td>0.69</td></tr><tr><td>Planner + Actor + Corrector‡</td><td>0.67</td><td>0.91</td><td>0.97</td><td>0.84</td></tr><tr><td>Planner + Actor + Corrector + Verifier +</td><td>0.66</td><td>0.91</td><td>0.97</td><td>0.82</td></tr></table>
<table><tbody><tr><td>使用的模块</td><td>SR $\uparrow$</td><td>TR↑</td><td>C↑</td><td>B↑</td></tr><tr><td>执行器</td><td>0.33</td><td>0.67</td><td>0.91</td><td>0.59</td></tr><tr><td>规划器 + 执行器 + 验证器</td><td>0.45</td><td>0.78</td><td>0.92</td><td>0.69</td></tr><tr><td>规划器 + 执行器 + 校正器‡</td><td>0.67</td><td>0.91</td><td>0.97</td><td>0.84</td></tr><tr><td>规划器 + 执行器 + 校正器 + 验证器 +</td><td>0.66</td><td>0.91</td><td>0.97</td><td>0.82</td></tr></tbody></table>


Table 3: Performance in the 2-agent scenarios in MAP-THOR obtained by ablating different modules in LLaMAR with GPT-4V as the underlying LM
表 3：在 MAP-THOR 的两智能体场景中通过剔除 LLaMAR 的不同模块（以 GPT-4V 为底层语言模型）得到的性能


<table><tr><td rowspan="2">#of agents</td><td colspan="4">MAP-THOR</td><td colspan="4">SAR</td></tr><tr><td>SR↑</td><td>TR↑</td><td>C↑</td><td>B↑</td><td>SR↑</td><td>TR↑</td><td>C↑</td><td>B↑</td></tr><tr><td>1</td><td>0.37</td><td>0.67</td><td>0.87</td><td>1.00</td><td>0.28</td><td>0.75</td><td>0.86</td><td>1.00</td></tr><tr><td>2</td><td>0.62</td><td>0.87</td><td>0.95</td><td>0.82</td><td>0.44</td><td>0.86</td><td>0.94</td><td>0.91</td></tr><tr><td>3</td><td>0.70</td><td>0.91</td><td>0.98</td><td>0.66</td><td>0.68</td><td>0.92</td><td>0.96</td><td>0.80</td></tr><tr><td>4</td><td>0.68</td><td>0.90</td><td>0.99</td><td>0.62</td><td>0.72</td><td>0.94</td><td>0.98</td><td>0.78</td></tr><tr><td>5</td><td>0.62</td><td>0.90</td><td>0.99</td><td>0.54</td><td>0.74</td><td>0.96</td><td>1.00</td><td>0.73</td></tr></table>
<table><tbody><tr><td rowspan="2">代理数量</td><td colspan="4">MAP-THOR</td><td colspan="4">SAR</td></tr><tr><td>SR↑</td><td>TR↑</td><td>C↑</td><td>B↑</td><td>SR↑</td><td>TR↑</td><td>C↑</td><td>B↑</td></tr><tr><td>1</td><td>0.37</td><td>0.67</td><td>0.87</td><td>1.00</td><td>0.28</td><td>0.75</td><td>0.86</td><td>1.00</td></tr><tr><td>2</td><td>0.62</td><td>0.87</td><td>0.95</td><td>0.82</td><td>0.44</td><td>0.86</td><td>0.94</td><td>0.91</td></tr><tr><td>3</td><td>0.70</td><td>0.91</td><td>0.98</td><td>0.66</td><td>0.68</td><td>0.92</td><td>0.96</td><td>0.80</td></tr><tr><td>4</td><td>0.68</td><td>0.90</td><td>0.99</td><td>0.62</td><td>0.72</td><td>0.94</td><td>0.98</td><td>0.78</td></tr><tr><td>5</td><td>0.62</td><td>0.90</td><td>0.99</td><td>0.54</td><td>0.74</td><td>0.96</td><td>1.00</td><td>0.73</td></tr></tbody></table>


Table 4: LLaMAR with various number of agents in the scenario in both MAP-THOR and SAR environments
表 4：LLaMAR 在 MAP-THOR 和 SAR 环境中随场景中代理数量变化的表现


Correcting Failures: In numerous instances, the actions proposed by the Actor module, such as pick up <object>, are unsuccessful due to the agent's insufficient proximity to the target object. In such situations, the Corrector module uses visual feedback to learn from these failures and recommends appropriate corrective actions, such as navigate to <object> to facilitate closer proximity. Figure 2 shows examples where the Corrector module interprets low-level action failures and suggests remedies, highlighting its importance.
纠正失败：在许多情况下，Actor 模块提出的动作（例如 pick up <object>）因代理与目标物体距离过远而未能成功。在这种情况下，Corrector 模块利用视觉反馈从这些失败中学习并推荐适当的纠正动作，例如 navigate to <object> 以便靠近。图 2 展示了 Corrector 模块如何解释底层动作失败并提出补救措施，突显其重要性。


## 7 Limitations and Future Work
## 7 限制与未来工作


Higher number queries to the LM: Since each high-level decision step requires querying 4 different LM-based modules, the cost and the compute times are higher than other baselines, especially compared to the plan-and-execute baselines like SmartLLM. An interesting future direction to
对语言模型的更高查询次数：由于每个高层决策步骤需要查询 4 个不同的基于 LM 的模块，成本和计算时间比其他基线更高，尤其是与诸如 SmartLLM 的“计划并执行”基线相比。一个有趣的未来方向是


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_16_58_ee7b4a.jpg"/>



Figure 2: A few examples of the Corrector module mitigate failures in predicted actions by the Actor module. (a) the Corrector suggests getting closer to the agent before attempting to pick it up, (b) the Corrector recommends opening the fridge because the previous action of placing the plate failed, (c) the Corrector advises rotating right so that it can access the table to place the tissue box on it when the low-level navigation policy failed to find a path to the table
图 2：几个示例展示了 Corrector 模块如何缓解 Actor 模块预测动作的失败。(a) Corrector 建议在尝试拾取前先靠近代理，(b) Corrector 建议打开冰箱，因为之前放置盘子的动作失败，(c) 当低层导航策略无法找到通往桌子的路径时，Corrector 建议向右旋转以便能接近桌子并将纸巾盒放上去


improve this would be to fine-tune smaller LMs with trajectories collected in the simulator (eg: ALFRED [46]) as done in [22]. Another potential direction worth exploring is using different sizes of LMs for each module based on their specific utility.
改进这一点的办法是用在模拟器中收集的轨迹对更小的语言模型进行微调（例如 ALFRED [46]），如 [22] 所做的那样。另一个值得探索的方向是根据各模块的具体作用使用不同规模的语言模型。


Limited spatial reasoning: Although we use both textual descriptions and visual features to guide the language model's actions, it still lacks the ability to reason about the spatial features of the environment. Spatial reasoning is crucial in scenarios such as navigating around obstacles to reach an object, or determining the shortest path to collect multiple items scattered across different locations. One way to address this limitation is to inject information about the 3D world into the LM, as done in [74], which is an interesting direction for future work.
有限的空间推理：尽管我们使用文本描述和视觉特征来引导语言模型的动作，但它仍然缺乏对环境空间特征的推理能力。空间推理在诸如绕过障碍物到达物体，或确定收集分散在不同位置的多个物品的最短路径等场景中至关重要。解决该限制的一种方法是向 LM 注入关于三维世界的信息，如 [74] 所示，这是未来工作的一个有趣方向。


Performance limited by the underlying VLM: Although LMs make correct reasoning most of the time, they still occasionally make mistakes, including misunderstanding the environment rules specified in the prompt. For example, the agent assumes that the cleaning task requires putting soap, drying, and putting it in the sink when all it needs is the action "CleanObject", and can't figure out the appropriate level of abstraction. The performance of the algorithm is limited by the instruction following and reasoning capability of the underlying LM [75, 76]; this calls for developing LMs that are fine-tuned to instruction-image pairs relevant to the environment (as done in [22]).
性能受限于底层视觉语言模型：尽管语言模型大部分时间能做出正确推理，但仍会偶尔出错，包括误解提示中指定的环境规则。例如，代理认为清洁任务需要放肥皂、擦干并放入水槽，而实际上所需只是“CleanObject”这一动作，因此无法确定合适的抽象级别。算法的性能受限于底层 LM 的指令遵循和推理能力 [75, 76]；这需要开发针对与环境相关的指令-图像对进行微调的 LM（如 [22] 所做）。


## 8 Conclusion
## 8 结论


We address long-horizon planning in dynamic, partially observable multi-agent environments with LLaMAR, an LM-based planner using four specialized modules: Planner, Actor, Corrector, and Verifier. This framework iteratively refines action planning, adapts to failures, and verifies subtask completion using real-time observations and action feedback, without privileged information. We also introduce a heuristic-based exploration strategy to guide agents to semantically relevant regions. Additionally, we present MAP-THOR, a benchmark dataset for multi-agent tasks in the AI2Thor simulator. Empirical results show LLaMAR outperforms existing LM-based approaches, achieving a 30% higher success rate on MAP-THOR.
我们提出 LLaMAR，一种用于动态、部分可观测多代理环境的基于 LM 的规划器，包含四个专用模块：Planner、Actor、Corrector 和 Verifier。该框架迭代精炼动作规划、适应失败并利用实时观测与动作反馈验证子任务完成情况，无需特权信息。我们还引入了一种启发式探索策略，引导代理到语义相关区域。此外，我们提出了 MAP-THOR，这是在 AI2Thor 模拟器中用于多代理任务的基准数据集。实验结果表明 LLaMAR 优于现有基于 LM 的方法，在 MAP-THOR 上取得了 30% 的成功率提升。


## Acknowledgements
## 致谢


We would like to thank Keerthana Gopalakrishnan, Sydney Dolan, Jasmine Aloor, and Victor Qin for helpful discussions and feedback. OpenAI credits for GPT-4 access was provided through OpenAI's Researcher Access Program. The research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
我们感谢 Keerthana Gopalakrishnan、Sydney Dolan、Jasmine Aloor 和 Victor Qin 的有益讨论与反馈。通过 OpenAI 的研究者访问计划提供了 GPT-4 的访问信用。本研究由空军人工智能加速器资助，并在合作协议编号 FA8750-19-2-1000 下完成。本文所载观点与结论仅为作者个人观点，不应被解读为代表空军部或美国政府的官方政策（无论明示或暗示）。尽管本文含有任何版权说明，美国政府有权为政府用途复制和分发再版材料。


## References
## 参考文献


[1] Unnat Jain, Luca Weihs, Eric Kolve, Ali Farhadi, Svetlana Lazebnik, Aniruddha Kembhavi, and Alexander G. Schwing. A Cordial Sync: Going Beyond Marginal Policies For Multi-Agent Embodied Tasks. In ECCV, 2020. first two authors contributed equally. 1
[1] Unnat Jain, Luca Weihs, Eric Kolve, Ali Farhadi, Svetlana Lazebnik, Aniruddha Kembhavi, 和 Alexander G. Schwing。《A Cordial Sync：超越边际策略的多代理具身任务》。发表于 ECCV, 2020。前两位作者贡献相同。1


[2] Unnat Jain, Luca Weihs, Eric Kolve, Mohammad Rastegari, Svetlana Lazebnik, Ali Farhadi, Alexander G. Schwing, and Aniruddha Kembhavi. Two Body Problem: Collaborative Visual Task Completion. In CVPR, 2019. first two authors contributed equally. 1
[2] Unnat Jain, Luca Weihs, Eric Kolve, Mohammad Rastegari, Svetlana Lazebnik, Ali Farhadi, Alexander G. Schwing, 和 Aniruddha Kembhavi。《Two Body Problem：协作视觉任务完成》。发表于 CVPR, 2019。前两位作者贡献相同。1


[3] Vijay Kumar, D. Rus, and Sanjiv Singh. Robot and sensor networks for first responders. IEEE Pervasive Computing, 3(4):24-33, 2004. 1
[3] Vijay Kumar, D. Rus, and Sanjiv Singh. 面向第一反应者的机器人与传感器网络。IEEE Pervasive Computing，3(4):24-33，2004。1


[4] Matthew Dunbabin and Lino Marques. Robots for Environmental Monitoring: Significant Advancements and Applications. IEEE Robotics & Automation Magazine, 19(1):24-39, 2012. 1
[4] Matthew Dunbabin and Lino Marques. 用于环境监测的机器人：重要进展与应用。IEEE Robotics & Automation Magazine，19(1):24-39，2012。1


[5] Brian Ichter, Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, Dmitry Kalashnikov, Sergey Levine, Yao Lu, Carolina Parada, Kanishka Rao, Pierre Sermanet, Alexander T Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Mengyuan Yan, Noah Brown, Michael Ahn, Omar Cortes, Nicolas Sievers, Clayton Tan, Sichun Xu, Diego Reyes, Jarek Rettinghouse, Jornell Quiambao, Peter Pastor, Linda Luu, Kuang-Huei Lee, Yuheng Kuang, Sally Jesmonth, Nikhil J. Joshi, Kyle Jeffrey, Rosario Jauano, Jasmine Hsu, Keerthana Gopalakrishnan, Byron David, Andy Zeng, and Chuyuan Kelly Fu. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. In Karen Liu, Dana Kulic, and Jeff Ichnowski, editors, Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 287-318. PMLR, 14-18 Dec 2023. 1.2
[5] Brian Ichter, Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, Dmitry Kalashnikov, Sergey Levine, Yao Lu, Carolina Parada, Kanishka Rao, Pierre Sermanet, Alexander T Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Mengyuan Yan, Noah Brown, Michael Ahn, Omar Cortes, Nicolas Sievers, Clayton Tan, Sichun Xu, Diego Reyes, Jarek Rettinghouse, Jornell Quiambao, Peter Pastor, Linda Luu, Kuang-Huei Lee, Yuheng Kuang, Sally Jesmonth, Nikhil J. Joshi, Kyle Jeffrey, Rosario Jauano, Jasmine Hsu, Keerthana Gopalakrishnan, Byron David, Andy Zeng, and Chuyuan Kelly Fu. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances。收录于 Karen Liu、Dana Kulic 和 Jeff Ichnowski 主编，《第6届机器人学习大会论文集》，作为 Proceedings of Machine Learning Research 第205卷，页287-318。PMLR，2023年12月14-18日。1.2


[6] Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor Mordatch, Sergey Levine, Karol Hausman, and Brian Ichter. Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 59636-59661. Curran Associates, Inc., 2023. 12
[6] Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor Mordatch, Sergey Levine, Karol Hausman, and Brian Ichter. Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents。收录于 A. Oh、T. Naumann、A. Globerson、K. Saenko、M. Hardt 和 S. Levine 主编，《神经信息处理系统进展》（Advances in Neural Information Processing Systems），第36卷，页59636-59661。Curran Associates, Inc., 2023。12


[7] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Tomas Jackson, Noah Brown, Linda Luu, Sergey Levine, Karol Hausman, and Brian Ichter. Inner Monologue: Embodied Reasoning through Planning with Language Models. In Karen Liu, Dana Kulic, and Jeff Ichnowski, editors, Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 1769-1782. PMLR, 14-18 Dec 2023. 1
[7] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Tomas Jackson, Noah Brown, Linda Luu, Sergey Levine, Karol Hausman, and Brian Ichter. Inner Monologue: Embodied Reasoning through Planning with Language Models。收录于 Karen Liu、Dana Kulic 和 Jeff Ichnowski 主编，《第6届机器人学习大会论文集》，作为 Proceedings of Machine Learning Research 第205卷，页1769-1782。PMLR，2023年12月14-18日。1


[8] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 11523-11530, 2023. 12
[8] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models。收录于 2023 IEEE 国际机器人与自动化会议（ICRA），页11523-11530，2023。12


[9] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as Policies: Language Model Programs for Embodied Control. In arXiv preprint arXiv:2209.07753, 2022. [1]
[9] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as Policies: Language Model Programs for Embodied Control。收录于 arXiv 预印本 arXiv:2209.07753，2022。[1]


[10] Kevin Lin, Christopher Agia, Toki Migimatsu, Marco Pavone, and Jeannette Bohg. Text2Motion: from natural language instructions to feasible plans. Autonomous Robots, 47(8):1345-1365, November 2023. 1
[10] Kevin Lin, Christopher Agia, Toki Migimatsu, Marco Pavone, and Jeannette Bohg. Text2Motion: from natural language instructions to feasible plans。Autonomous Robots，47(8):1345-1365，2023年11月。1


[11] Dhruv Shah, Blazej Osinski, Brian Ichter, and Sergey Levine. LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action. In 6th Annual Conference on Robot Learning, 2022. 1
[11] Dhruv Shah, Blazej Osinski, Brian Ichter, and Sergey Levine. LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action。收录于第6届年度机器人学习大会，2022。1


[12] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual Language Maps for Robot Navigation. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), London, UK, 2023. 1
[12] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual Language Maps for Robot Navigation。收录于 IEEE 国际机器人与自动化会议（ICRA）论文集，伦敦，英国，2023。1


[13] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 9118-9147. PMLR, 17-23 Jul 2022. 12 25
[13] 黄文龙, Pieter Abbeel, Deepak Pathak, 和 Igor Mordatch. 将语言模型作为零样本规划器：为具身代理提取可执行知识. 收录于 Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, 和 Sivan Sabato 编, 第39届国际机器学习会议论文集, Proceedings of Machine Learning Research 第162卷, 页9118-9147. PMLR, 2022年7月17-23日. 12 25


[14] Ming Tan. Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents. In In Proceedings of the Tenth International Conference on Machine Learning, pages 330-337. Morgan Kaufmann, 1993. 1
[14] 谭明. 多智能体强化学习：独立代理 vs 合作代理. 收录于 第十届国际机器学习会议论文集, 页330-337. Morgan Kaufmann, 1993. 1


[15] Ardi Tampuu, Tambet Matiisen, Dorian Kodelja, Ilya Kuzovkin, Kristjan Korjus, Juhan Aru, Jaan Aru, and Raul Vicente. Multiagent Cooperation and Competition with Deep Reinforcement Learning. CoRR, abs/1511.08779, 2015. 1
[15] Ardi Tampuu, Tambet Matiisen, Dorian Kodelja, Ilya Kuzovkin, Kristjan Korjus, Juhan Aru, Jaan Aru, 和 Raul Vicente. 使用深度强化学习的多智能体合作与竞争. CoRR, abs/1511.08779, 2015. 1


[16] Yongchao Chen, Jacob Arkin, Yang Zhang, Nicholas Roy, and Chuchu Fan. Scalable Multi-Robot Collaboration with Large Language Models: Centralized or Decentralized Systems? arXiv preprint arXiv:2309.15943, 2023. 1233.
[16] 陈永超, Jacob Arkin, 张洋, Nicholas Roy, 和 茹楚楚. 使用大型语言模型进行可扩展多机器人协作：集中式还是去中心化系统？ arXiv 预印本 arXiv:2309.15943, 2023. 1233.


[17] Daman Arora and Subbarao Kambhampati. Learning and Leveraging Verifiers to Improve Planning Capabilities of Pre-trained Language Models. arXiv preprint arXiv:2305.17077, 2023. 2. B
[17] Daman Arora 和 Subbarao Kambhampati. 学习并利用验证器以提高预训练语言模型的规划能力. arXiv 预印本 arXiv:2305.17077, 2023. 2. B


[18] Karthik Valmeekam, Matthew Marquez, and Subbarao Kambhampati. Can Large Language Models Really Improve by Self-critiquing Their Own Plans? arXiv preprint arXiv:2310.08118, 2023. 2
[18] Karthik Valmeekam, Matthew Marquez, 和 Subbarao Kambhampati. 大型语言模型真的能通过自我批评其自身计划而提升吗？ arXiv 预印本 arXiv:2310.08118, 2023. 2


[19] Shyam Sundar Kannan, Vishnunandan LN Venkatesh, and Byung-Cheol Min. SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models. arXiv preprint arXiv:2309.10062, 2023. 234567
[19] Shyam Sundar Kannan, Vishnunandan LN Venkatesh, 和 Byung-Cheol Min. SMART-LLM：使用大型语言模型的智能多智能体机器人任务规划. arXiv 预印本 arXiv:2309.10062, 2023. 234567


[20] Jun Wang, Guocheng He, and Yiannis Kantaros. Safe Task Planning for Language-Instructed Multi-Robot Systems using Conformal Prediction. arXiv preprint arXiv:2402.15368, 2024. 2. 3. 4. 24
[20] 王俊, 何国成, 和 Yiannis Kantaros. 使用保形预测的语言指令多机器人系统的安全任务规划. arXiv 预印本 arXiv:2402.15368, 2024. 2. 3. 4. 24


[21] Ishika Singh, David Traum, and Jesse Thomason. TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models. arXiv preprint arXiv:2403.17246, 2024. 2 3 24
[21] Ishika Singh, David Traum, 和 Jesse Thomason. TwoStep：使用经典规划器和大型语言模型的多智能体任务规划. arXiv 预印本 arXiv:2403.17246, 2024. 2 3 24


[22] Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B Tenenbaum, Tianmin Shu, and Chuang Gan. Building Cooperative Embodied Agents Modularly with Large Language Models. ICLR, 2024.204810
[22] 张宏昕, 杜卫华, 单家明, 周庆宏, 杜一伦, Joshua B Tenenbaum, 舒天民, 和 甘闯. 模块化构建合作具身代理与大型语言模型. ICLR, 2024.204810


[23] Bangguo Yu, Hamidreza Kasaei, and Ming Cao. Co-NavGPT: Multi-Robot Cooperative Visual Semantic Navigation using Large Language Models. ArXiv, abs/2310.07937, 2023. 2 3
[23] 于邦国, Hamidreza Kasaei, 和 曹明. Co-NavGPT：使用大型语言模型的多机器人协作视觉语义导航. ArXiv, abs/2310.07937, 2023. 2 3


[24] Zhao Mandi, Shreeya Jain, and Shuran Song. RoCo: Dialectic multi-robot collaboration with large language models. arXiv preprint arXiv:2307.04738, 2023. 2 3
[24] 赵曼迪, Shreeya Jain, 和 宋曙然. RoCo：与大型语言模型的辩证式多机器人协作. arXiv 预印本 arXiv:2307.04738, 2023. 2 3


[25] Andrew G. Barto and Sridhar Mahadevan. Recent Advances in Hierarchical Reinforcement Learning. Discrete Event Dynamic Systems, 13:41-77, 2003. 2
[25] Andrew G. Barto 和 Sridhar Mahadevan. 分层强化学习的最新进展. Discrete Event Dynamic Systems, 13:41-77, 2003. 2


[26] Ofir Nachum, Shixiang (Shane) Gu, Honglak Lee, and Sergey Levine. Data-Efficient Hierarchical Reinforcement Learning. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. 2
[26] Ofir Nachum, Shixiang (Shane) Gu, Honglak Lee, 和 Sergey Levine. 数据高效的分层强化学习. 收录于 S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, 和 R. Garnett 编, Advances in Neural Information Processing Systems, 第31卷. Curran Associates, Inc., 2018. 2


[27] Shubham Pateria, Budhitama Subagdja, Ah-hwee Tan, and Chai Quek. Hierarchical Reinforcement Learning: A Comprehensive Survey. ACM Comput. Surv., 54(5), jun 2021. 2
[27] Shubham Pateria, Budhitama Subagdja, Ah-hwee Tan, 和 Chai Quek. 分层强化学习：全面综述. ACM Comput. Surv., 54(5), 2021年6月. 2


[28] Jan Wohlke, Felix Schmitt, and Herke van Hoof. Hierarchies of Planning and Reinforcement Learning for Robot Navigation. In 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, May 2021. 2
[28] Jan Wohlke、Felix Schmitt 和 Herke van Hoof。用于机器人导航的规划与强化学习层级结构。载于 2021 年 IEEE 国际机器人与自动化会议（ICRA）。IEEE，2021 年 5 月。2


[29] Hongyao Tang, Jianye Hao, Tangjie Lv, Yingfeng Chen, Zongzhang Zhang, Hangtian Jia, Chunxu Ren, Yan Zheng, Zhaopeng Meng, Changjie Fan, and Li Wang. Hierarchical Deep Multiagent Reinforcement Learning with Temporal Abstraction, 2019. 2
[29] Hongyao Tang、Jianye Hao、Tangjie Lv、Yingfeng Chen、Zongzhang Zhang、Hangtian Jia、Chunxu Ren、Yan Zheng、Zhaopeng Meng、Changjie Fan 和 Li Wang。具有时间抽象的分层深度多智能体强化学习，2019 年。2


[30] Jiachen Yang, Igor Borovikov, and Hongyuan Zha. Hierarchical Cooperative Multi-Agent Reinforcement Learning with Skill Discovery. CoRR, abs/1912.03558, 2019. 2
[30] Jiachen Yang、Igor Borovikov 和 Hongyuan Zha。具有技能发现的分层协作多智能体强化学习。CoRR，abs/1912.03558，2019 年。2


[31] Jesse Meyer, Ryan Praeuner, and Luke Vanderbeek. Hierarchical Multi-Agent Reinforcement Learning. https://cse.unl.edu/~lksoh/Classes/CSCE475_875_Fall11/seminars/Seminar_JRL pdf 2020.2
[31] Jesse Meyer、Ryan Praeuner 和 Luke Vanderbeek。分层多智能体强化学习。https://cse.unl.edu/~lksoh/Classes/CSCE475_875_Fall11/seminars/Seminar_JRL pdf 2020.2


[32] Adam Stooke, Kimin Lee, Pieter Abbeel, and Michael Laskin. Decoupling representation learning from reinforcement learning. In International Conference on Machine Learning, pages 9870-9879. PMLR, 2021. 2
[32] Adam Stooke、Kimin Lee、Pieter Abbeel 和 Michael Laskin。将表示学习与强化学习解耦。在国际机器学习会议（ICML），页码 9870-9879。PMLR，2021 年。2


[33] Ruihan Yang, Huazhe Xu, Yi Wu, and Xiaolong Wang. Multi-task reinforcement learning with soft modularization. Advances in Neural Information Processing Systems, 33:4767-4777, 2020. 2
[33] Ruihan Yang、Huazhe Xu、Yi Wu 和 Xiaolong Wang。带有软模块化的多任务强化学习。NeurIPS，33：4767-4777，2020 年。2


[34] Sherry Yang, Ofir Nachum, Yilun Du, Jason Wei, Pieter Abbeel, and Dale Schuurmans. Foundation models for decision making: Problems, methods, and opportunities. arXiv preprint arXiv:2303.04129, 2023. 2
[34] Sherry Yang、Ofir Nachum、Yilun Du、Jason Wei、Pieter Abbeel 和 Dale Schuurmans。用于决策的基础模型：问题、方法与机遇。arXiv 预印本 arXiv:2303.04129，2023 年。2


[35] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A Survey on Large Language Model based Autonomous Agents, 2023. 2
[35] Lei Wang、Chen Ma、Xueyang Feng、Zeyu Zhang、Hao Yang、Jingsen Zhang、Zhiyuan Chen、Jiakai Tang、Xu Chen、Yankai Lin、Wayne Xin Zhao、Zhewei Wei 和 Ji-Rong Wen。基于大语言模型的自治代理综述，2023 年。2


[36] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, and Tao Gui. The Rise and Potential of Large Language Model Based Agents: A Survey, 2023. 2
[36] Zhiheng Xi、Wenxiang Chen、Xin Guo、Wei He、Yiwen Ding、Boyang Hong、Ming Zhang、Junzhe Wang、Senjie Jin、Enyu Zhou、Rui Zheng、Xiaoran Fan、Xiao Wang、Limao Xiong、Yuhao Zhou、Weiran Wang、Changhao Jiang、Yicheng Zou、Xiangyang Liu、Zhangyue Yin、Shihan Dou、Rongxiang Weng、Wensen Cheng、Qi Zhang、Wenjuan Qin、Yongyan Zheng、Xipeng Qiu、Xuanjing Huang 和 Tao Gui。基于大语言模型的代理的兴起与潜力：一项综述，2023 年。2


[37] Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Griffiths. Cognitive Architectures for Language Agents, 2023. 2
[37] Theodore R. Sumers、Shunyu Yao、Karthik Narasimhan 和 Thomas L. Griffiths。面向语言代理的认知架构，2023 年。2


[38] Pratyusha Sharma, Antonio Torralba, and Jacob Andreas. Skill Induction and Planning with Latent Language. CoRR, abs/2110.01517, 2021. 2
[38] Pratyusha Sharma、Antonio Torralba 和 Jacob Andreas。用潜在语言进行技能归纳与规划。CoRR，abs/2110.01517，2021 年。2


[39] Shreyas Sundara Raman, Vanya Cohen, Eric Rosen, Ifrah Idrees, David Paulius, and Stefanie Tellex. Planning with Large Language Models via Corrective Re-prompting. January 2022. 2
[39] Shreyas Sundara Raman、Vanya Cohen、Eric Rosen、Ifrah Idrees、David Paulius 和 Stefanie Tellex。通过纠正性重提示用大语言模型进行规划。2022 年 1 月。2


[40] Maitrey Gramopadhye and Daniel Szafir. Generating Executable Action Plans with Environmentally-Aware Language Models, 2023. 2
[40] Maitrey Gramopadhye 和 Daniel Szafir。用具备环境感知能力的语言模型生成可执行行动计划，2023 年。2


[41] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Daniel Gordon, Yuke Zhu, Abhinav Gupta, and Ali Farhadi. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv, 2017. 246
[41] Eric Kolve、Roozbeh Mottaghi、Winson Han、Eli VanderBilt、Luca Weihs、Alvaro Herrasti、Daniel Gordon、Yuke Zhu、Abhinav Gupta 和 Ali Farhadi。AI2-THOR：用于视觉 AI 的交互式 3D 环境。arXiv，2017 年。246


[42] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A Platform for Embodied AI Research. CoRR, abs/1904.01201, 2019. 2 4
[42] Manolis Savva、Abhishek Kadian、Oleksandr Maksymets、Yili Zhao、Erik Wijmans、Bhavana Jain、Julian Straub、Jia Liu、Vladlen Koltun、Jitendra Malik、Devi Parikh 和 Dhruv Batra。Habitat：面向体现式 AI 研究的平台。CoRR，abs/1904.01201，2019 年。2 4


[43] Fei Xia, Amir R. Zamir, Zhi-Yang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson Env: Real-World Perception for Embodied Agents. CoRR, abs/1808.10654, 2018. 2
[43] Fei Xia, Amir R. Zamir, Zhi-Yang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson Env: 面向具身智能体的真实世界感知. CoRR, abs/1808.10654, 2018. 2


[44] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio Savarese, Hyowon Gweon, Karen Liu, Jiajun Wu, and Li Fei-Fei. BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities and Realistic Simulation. In Karen Liu, Dana Kulic, and Jeff Ichnowski, editors, Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 80-93. PMLR, 14-18 Dec 2023. 2
[44] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio Savarese, Hyowon Gweon, Karen Liu, Jiajun Wu, and Li Fei-Fei. BEHAVIOR-1K: 面向具身人工智能的基准，涵盖1000项日常活动与逼真仿真. In Karen Liu, Dana Kulic, and Jeff Ichnowski, editors, Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 80-93. PMLR, 14-18 Dec 2023. 2


[45] Aishwarya Padmakumar, Jesse Thomason, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gokhan Tur, and Dilek Hakkani-Tur. TEACh: Task-driven Embodied Agents that Chat, 2021. 2
[45] Aishwarya Padmakumar, Jesse Thomason, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gokhan Tur, and Dilek Hakkani-Tur. TEACh: 以任务为导向且会对话的具身智能体, 2021. 2


[46] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. CoRR, abs/1912.01734, 2019. 2 10
[46] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. ALFRED: 用于解释日常任务的有地面化指令的基准. CoRR, abs/1912.01734, 2019. 2 10


[47] Dipendra Kumar Misra, Andrew Bennett, Valts Blukis, Eyvind Niklasson, Max Shatkhin, and Yoav Artzi. Mapping Instructions to Actions in 3D Environments with Visual Goal Prediction. CoRR, abs/1809.00786, 2018.2
[47] Dipendra Kumar Misra, Andrew Bennett, Valts Blukis, Eyvind Niklasson, Max Shatkhin, and Yoav Artzi. 在带视觉目标预测的3D环境中将指令映射为动作. CoRR, abs/1809.00786, 2018.2


[48] Yuke Zhu, Daniel Gordon, Eric Kolve, Dieter Fox, Li Fei-Fei, Abhinav Gupta, Roozbeh Mottaghi, and Ali Farhadi. Visual semantic planning using deep successor representations. In Proceedings of the IEEE international conference on computer vision, pages 483-492, 2017. 2
[48] Yuke Zhu, Daniel Gordon, Eric Kolve, Dieter Fox, Li Fei-Fei, Abhinav Gupta, Roozbeh Mottaghi, and Ali Farhadi. 使用深度继任者表示的视觉语义规划. In Proceedings of the IEEE international conference on computer vision, pages 483-492, 2017. 2


[49] Simon Brodeur, Ethan Perez, Ankesh Anand, Florian Golemo, Luca Celotti, Florian Strub, Jean Rouat, Hugo Larochelle, and Aaron C. Courville. HoME: a Household Multimodal Environment. CoRR, abs/1711.11017, 2017. 2
[49] Simon Brodeur, Ethan Perez, Ankesh Anand, Florian Golemo, Luca Celotti, Florian Strub, Jean Rouat, Hugo Larochelle, and Aaron C. Courville. HoME: 一种家庭多模态环境. CoRR, abs/1711.11017, 2017. 2


[50] Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan, He Wang, Li Yi, Angel X. Chang, Leonidas J. Guibas, and Hao Su. SAPIEN: A Simulated Part-based Interactive ENvironment. CoRR, abs/2003.08515, 2020. 2
[50] Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan, He Wang, Li Yi, Angel X. Chang, Leonidas J. Guibas, and Hao Su. SAPIEN: 一种基于部件的交互式仿真环境. CoRR, abs/2003.08515, 2020. 2


[51] Unnat Jain, Luca Weihs, Eric Kolve, Mohammad Rastegari, Svetlana Lazebnik, Ali Farhadi, Alexander G. Schwing, and Aniruddha Kembhavi. Two Body Problem: Collaborative Visual Task Completion. CoRR, abs/1904.05879, 2019. 2
[51] Unnat Jain, Luca Weihs, Eric Kolve, Mohammad Rastegari, Svetlana Lazebnik, Ali Farhadi, Alexander G. Schwing, and Aniruddha Kembhavi. Two Body Problem: 协作视觉任务完成. CoRR, abs/1904.05879, 2019. 2


[52] Unnat Jain, Luca Weihs, Eric Kolve, Ali Farhadi, Svetlana Lazebnik, Aniruddha Kembhavi, and Alexander Schwing. A cordial sync: Going beyond marginal policies for multi-agent embodied tasks. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part V 16, pages 471-490. Springer, 2020. 2
[52] Unnat Jain, Luca Weihs, Eric Kolve, Ali Farhadi, Svetlana Lazebnik, Aniruddha Kembhavi, and Alexander Schwing. A cordial sync: 在多智能体具身任务中超越边缘策略. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part V 16, pages 471-490. Springer, 2020. 2


[53] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M. Sadler, Wei-Lun Chao, and Yu Su. LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models, 2023. 3
[53] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M. Sadler, Wei-Lun Chao, and Yu Su. LLM-Planner: 用大语言模型进行少样本有根规划以支持具身智能体, 2023. 3


[54] Lin Xu, Zhiyuan Hu, Daquan Zhou, Hongyu Ren, Zhen Dong, Kurt Keutzer, See Kiong Ng, and Jiashi Feng. MAgIC: Benchmarking Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration. arXiv preprint arXiv:2311.08562, 2023. 3
[54] 林旭, 胡志远, 周大权, 任鸿宇, 董振, Kurt Keutzer, 吴思炯, 及 冯家石. MAgIC：基于大语言模型的多智能体在认知、适应性、理性与协作方面的基准测试. arXiv 预印本 arXiv:2311.08562, 2023. 3


[55] Martin L. Puterman. Markov Decision Processes. Wiley, 1994. 3
[55] Martin L. Puterman. 马尔可夫决策过程. Wiley, 1994. 3


[56] Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1):99-134, 1998. 3
[56] Leslie Pack Kaelbling, Michael L. Littman, 及 Anthony R. Cassandra. 在部分可观测随机领域中的规划与行动. Artificial Intelligence, 101(1):99-134, 1998. 3


[57] Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. VirtualHome: Simulating Household Activities via Programs, 2018. 4
[57] Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, 及 Antonio Torralba. VirtualHome：通过程序模拟家庭活动, 2018. 4


[58] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondruš, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, and Dhruv Batra. Habitat 2.0: Training Home Assistants to Rearrange their Habitat. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 4
[58] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondruš, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, 及 Dhruv Batra. Habitat 2.0：训练家居助手重新布置其栖息环境. 载于 Advances in Neural Information Processing Systems (NeurIPS), 2021. 4


[59] Xavi Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Ruslan Partsey, Jimmy Yang, Ruta Desai, Alexander William Clegg, Michal Hlavac, Tiffany Min, Theo Gervet, Vladimír Vondruš, Vincent-Pierre Berges, John Turner, Oleksandr Maksymets, Zsolt Kira, Mrinal Kalakrishnan, Jitendra Malik, Devendra Singh Chaplot, Unnat Jain, Dhruv Batra, Akshara Rai, and Roozbeh Mottaghi. Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots, 2023. 4
[59] Xavi Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Ruslan Partsey, Jimmy Yang, Ruta Desai, Alexander William Clegg, Michal Hlavac, Tiffany Min, Theo Gervet, Vladimír Vondruš, Vincent-Pierre Berges, John Turner, Oleksandr Maksymets, Zsolt Kira, Mrinal Kalakrishnan, Jitendra Malik, Devendra Singh Chaplot, Unnat Jain, Dhruv Batra, Akshara Rai, 及 Roozbeh Mottaghi. Habitat 3.0：人类、虚拟身影与机器人的共栖环境, 2023. 4


[60] Chuang Gan, Siyuan Zhou, Jeremy Schwartz, Seth Alter, Abhishek Bhandwaldar, Dan Gutfreund, Daniel L. K. Yamins, James J DiCarlo, Josh McDermott, Antonio Torralba, and Joshua B. Tenenbaum. The Three-DWorld Transport Challenge: A Visually Guided Task-and-Motion Planning Benchmark for Physically Realistic Embodied AI, 2021. 4
[60] Chuang Gan, Siyuan Zhou, Jeremy Schwartz, Seth Alter, Abhishek Bhandwaldar, Dan Gutfreund, Daniel L. K. Yamins, James J DiCarlo, Josh McDermott, Antonio Torralba, 及 Joshua B. Tenenbaum. 三维世界运输挑战：面向物理真实感具身 AI 的视觉引导任务与运动规划基准, 2021. 4


[61] Dhruv Batra, Angel X. Chang, Sonia Chernova, Andrew J. Davison, Jia Deng, Vladlen Koltun, Sergey Levine, Jitendra Malik, Igor Mordatch, Roozbeh Mottaghi, Manolis Savva, and Hao Su. Rearrangement: A Challenge for Embodied AI. CoRR, abs/2011.01975, 2020. 4
[61] Dhruv Batra, Angel X. Chang, Sonia Chernova, Andrew J. Davison, Jia Deng, Vladlen Koltun, Sergey Levine, Jitendra Malik, Igor Mordatch, Roozbeh Mottaghi, Manolis Savva, 及 Hao Su. 重排：具身 AI 的一项挑战. CoRR, abs/2011.01975, 2020. 4


[62] Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, and Tushar Khot. ADaPT: As-Needed Decomposition and Planning with Language Models, 2023. 4
[62] Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, 及 Tushar Khot. ADaPT：基于语言模型的按需分解与规划, 2023. 4


[63] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, 2019. 5
[63] Nils Reimers 及 Iryna Gurevych. Sentence-BERT：使用连体 BERT 网络的句子向量, 2019. 5


[64] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. CoRR, abs/2103.00020, 2021. 5
[64] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, 及 Ilya Sutskever. 从自然语言监督中学习可迁移的视觉模型. CoRR, abs/2103.00020, 2021. 5


[65] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. Chain of Thought Prompting Elicits Reasoning in Large Language Models. CoRR, abs/2201.11903, 2022. 7
[65] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, 及 Denny Zhou. 连锁思路提示在大语言模型中激发推理. CoRR, abs/2201.11903, 2022. 7


[66] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing Reasoning and Acting in Language Models, 2023. 7
[66] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, 及 Yuan Cao. ReAct：在语言模型中协同推理与行动, 2023. 7


[67] OpenAI, :, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mo Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. GPT-4 Technical Report, 2023. 8
[67] OpenAI, :, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mo Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph。GPT-4 技术报告，2023。8


[68] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents, 2023. 8 32
[68] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents, 2023. 8 32


[69] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024. 8 32
[69] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024. 8 32


[70] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In NeurIPS, 2023. 8 32
[70] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In NeurIPS, 2023. 8 32


[71] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning, 2023. 8
[71] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning, 2023. 8


[72] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. CogAgent: A Visual Language Model for GUI Agents, 2023. 8 32
[72] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. CogAgent: A Visual Language Model for GUI Agents, 2023. 8 32


[73] Kaya Stechly, Karthik Valmeekan, and Subbarao Kambhampati. Chain of Thoughtlessness: An Analysis of CoT in Planning. arXiv preprint arXiv:2405.04776, 2024. 8
[73] Kaya Stechly, Karthik Valmeekan, and Subbarao Kambhampati. Chain of Thoughtlessness: An Analysis of CoT in Planning. arXiv preprint arXiv:2405.04776, 2024. 8


[74] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3D World into Large Language Models, 2023. 10
[74] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3D World into Large Language Models, 2023. 10


[75] Subbarao Kambhampati. Can large language models reason and plan? Annals of the New York Academy of Sciences, 1534(1):15-18, 2024. 10
[75] Subbarao Kambhampati. Can large language models reason and plan? Annals of the New York Academy of Sciences, 1534(1):15-18, 2024. 10


[76] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models-a critical investigation. Advances in Neural Information Processing Systems, 36:75993-76005, 2023. 10
[76] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models-a critical investigation. Advances in Neural Information Processing Systems, 36:75993-76005, 2023. 10


## A Terminology
## 术语


We differentiate between the terms subtasks and high-level actions in this section. In essence, multiple high-level actions are needed to be carried out in a sequence to complete a subtask. Multiple subtasks need to be satisfied to complete the high-level language instruction.
本节我们区分“子任务”和“高层动作”二词。本质上，完成一个子任务需要按顺序执行多个高层动作；完成高层语言指令又需满足多个子任务。


- Subtasks: A task is split up into multiple subtasks. For example, if a task is "Fetch all the groceries and put them in the fridge", then the initial subtasks could include: "Locate the groceries", "transport the groceries", "Locate the fridge". These subtasks could get updated with new observations. For example, while locating the groceries, the agents come across a tomato and a lettuce. Then the subtasks "transport the tomato to the fridge" and "transport the lettuce to the fridge" gets updated in the subtasks list. This basically splits up the high-level instruction $\mathcal{I}$ into multiple mid-level subtasks
- 子任务：一个任务被拆分为多个子任务。例如，若任务是“把购物全部拿来并放进冰箱”，初始子任务可包括：“找到购物物品”、“搬运购物物品”、“找到冰箱”。这些子任务可随新观测更新。例如，在寻找购物物品时，代理发现了番茄和生菜，则子任务列表中会更新为“把番茄搬到冰箱”和“把生菜搬到冰箱”。这基本上将高层指令 $\mathcal{I}$ 拆分为多个中层子任务


- High-level actions: These are the set of actions required to complete the subtasks. For example, to complete the "transport the lettuce in the fridge", we would require: the following set of actions:
- 高层动作：完成子任务所需的一系列动作。例如，为了完成“把生菜搬进冰箱”，需要以下动作集合：


- Navigate to lettuce
- 导航至生菜


- Pickup lettuce
- 拾取生菜


- Navigate to the fridge
- 导航至冰箱


- Open fridge
- 打开冰箱


- Put lettuce in the fridge
- 将生菜放入冰箱


- Close fridge
- 关上冰箱


Note that different agents have to complete different high-level actions that progress the subtasks efficiently whilst avoiding conflicts.
注意不同的代理必须完成不同的高层动作，以高效推进子任务并避免冲突。


- Conflicts can arise in the following ways:
- 冲突可能以以下方式出现：


- Same actions: Agents performing the same action at the same time. Example: "Open the fridge".
- 相同动作：代理在同一时间执行相同动作。例如：“打开冰箱”。


- Blocking: Agent 1 is blocking Agent 2 and not allowing it to complete its high-level action. Example: Agent 1 is attempting to execute "PlaceObject(Tomato)" in front of the fridge to place the tomato in its hand in the fridge and Agent 2 is attempting to execute "OpenFreezer()" needs to interact with the fridge. Would require some form of conflict resolution in the state cell domain. Agent 1 should move away to allow fridge access to Agent 2. In LLaMAR, the Corrector module helps in figuring out these conflicts and suggest different corrective high-level actions.
- 阻挡：代理1阻挡代理2，阻止其完成高层动作。例如：代理1试图在冰箱前执行“PlaceObject(Tomato)”将番茄放入手中并放入冰箱，而代理2试图执行“OpenFreezer()”与冰箱交互。在状态单元域中需要某种冲突解决机制。代理1应后退以允许代理2访问冰箱。在LLaMAR中，Corrector模块有助于识别这些冲突并建议不同的纠正高层动作。


## B MAP-THOR Environment
## B MAP-THOR 环境


The environment is based on the AI2Thor simulator with a multi-agent setup. All the experiments were performed in the single-room floor plans. When more than 3 agents are added to some of the floor plans (especially the kitchen floor plans), the environment gets crowded and does not allow for a lot of free space to navigate to different objects (the number of reachable paths reduces).
该环境基于 AI2Thor 模拟器并采用多代理设置。所有实验均在单房间平面图中进行。当某些平面图（尤其是厨房平面图）中加入超过3个代理时，环境会变得拥挤，不允许太多自由空间用于前往不同物体（可达路径数量减少）。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_16_58_02aea7.jpg"/>



Figure 3: Photorealistic rendering of household scenarios in the AI2Thor simulator enables the usage of multiple autonomous robots to carry out daily tasks.
图3：AI2Thor 模拟器中家居场景的光照片真实渲染使多台自主机器人能够执行日常任务。


### B.1 Observation Space
### B.1 观测空间


The observations for each robot include an image of size resolution ${1000} \times  {1000} \times  3$ . The textual observation for each agent in the prompt is the list of objects visible in this image and the agents' current location and rotation. The field of view is 90 degrees. The agents can interact with the objects only if it is within its visibility range of ${1.5}\mathrm{\;m}$ .
每个机器人的观测包括分辨率为 ${1000} \times  {1000} \times  3$ 的图像。提示中每个代理的文本观测是该图像中可见物体的列表以及代理当前的位置和朝向。视场为90度。只有当物体在其可见范围 ${1.5}\mathrm{\;m}$ 内时，代理才能与之交互。


### B.2 Action Space
### B.2 动作空间


The actions space $\mathcal{A}$ consists of navigation actions ${\mathcal{A}}_{NAV}$ ,interaction actions ${\mathcal{A}}_{INT}$ ,exploration action ${\mathcal{A}}_{EXP}$ .
动作空间 $\mathcal{A}$ 包含导航动作 ${\mathcal{A}}_{NAV}$ 、交互动作 ${\mathcal{A}}_{INT}$ 、探索动作 ${\mathcal{A}}_{EXP}$ 。


Navigation actions ${\mathcal{A}}_{NAV}$ consists of the following actions:
导航动作 ${\mathcal{A}}_{NAV}$ 包含以下动作：


- Move(<direction>): Moves the robot by ${0.25}\mathrm{\;m}$ towards the specified direction where <direction> can be one of (Ahead, Back, Right, Left)
- Move(&lt;direction&gt;): 使机器人以 ${0.25}\mathrm{\;m}$ 向指定方向移动，&lt;direction&gt; 可以是 (Ahead, Back, Right, Left) 之一


- Rotate(<direction>): Rotates the robot by 90 degrees towards the specified direction where, <direction> can be one of (Right, Left)
- Rotate(<direction>): 使机器人朝指定方向旋转90度，<direction> 可为 (Right, Left)


- LookUp(<angle>) rotates the pitch of the robot camera upwards by the specified angle.
- LookUp(<angle>) 将机器人摄像头的俯仰角向上旋转指定角度。


- LookDown<angle> rotates the pitch of the robot camera downwards by the specified angle.
- LookDown<angle> 将机器人摄像头的俯仰角向下旋转指定角度。


- NavigateTo(<object_id>) makes the robot navigate to the specified object. The path is found using the ${A}^{ * }$ -shortest path algorithm. Note that the robot is only able to find a path to the specified object in the environment only if it has encountered that object previously during the episode. Otherwise, the NavigateTo(.) action will be unsuccessful and the agent will have to explore.
- NavigateTo(<object_id>) 使机器人导航到指定对象。路径使用 ${A}^{ * }$ 最短路径算法寻找。注意，只有在本回合中机器人先前遇到过该对象时，才可能在环境中找到到该对象的路径。否则，NavigateTo(.) 操作会失败，代理需进行探索。


Interaction actions ${\mathcal{A}}_{INT}$ consists of the following actions:
交互动作 ${\mathcal{A}}_{INT}$ 包括以下动作：


---



- Pickup(<object_id>): Picks up the object
- Pickup(<object_id>): 拾取对象


- Put(<receptacle_id>): Puts the object in the robots hand on the receptacle
- Put(<receptacle_id>): 将机器人手中的对象放到容器上


	- Open(<object_id>): Opens the object
	- - Open(<object_id>): 打开对象


	- Close(<object_id>): Closes the open object
	- - Close(<object_id>): 关闭已打开的对象


	- Slice(<object_id>): Slices the object
	- - Slice(<object_id>): 切割对象


	- Clean(<object_id>): Cleans the object
	- - Clean(<object_id>): 清洁对象


- ToggleOn(<object_id>): Toggles the object on
- ToggleOn(<object_id>): 将对象打开


- ToggleOff (<object_id>): Toggles the object off
- ToggleOff (<object_id>): 将对象关闭


---



Explore action ${\mathcal{A}}_{EXP}$ : The explore action is carried out by the heuristic explained in Algorithm 1 and Figure 4 We use the clip-vit-large-patch14-336 model for the CLIP weights which we download from https://huggingface.co/openai/clip-vit-large-patch14-336
探索动作 ${\mathcal{A}}_{EXP}$：探索动作由算法1和图4中解释的启发式方法执行。我们使用 clip-vit-large-patch14-336 模型作为 CLIP 权重，从 https://huggingface.co/openai/clip-vit-large-patch14-336 下载。


The action space consists of several executable actions, including Move(<direction>), Rotate(<direction>), LookUp(<angle>), LookDown(<angle>), NavigateTo(<object_id>), Pickup(<object_id>), Put(<receptacle_id>), and others. These actions can be combined in numerous ways, as the <object_id>, <directions>, and <angles> can vary significantly, resulting in a combinatorially large action space. To address this complexity, we do not impose constraints on the language model (LM) modules to select actions in a predefined format from this set. Instead, we allow for free-form language outputs, offering greater expressivity. The SentenceBERT module then maps these free-form natural language outputs to executable actions within the environment.
动作空间包含若干可执行动作，包括 Move(<direction>)、Rotate(<direction>)、LookUp(<angle>)、LookDown(<angle>)、NavigateTo(<object_id>)、Pickup(<object_id>)、Put(<receptacle_id>) 等。这些动作可以多种组合，因 <object_id>、<directions> 和 <angles> 可大幅变化，导致组合式巨大的动作空间。为应对该复杂性，我们不强制语言模型（LM）模块以预定义格式从该集合中选择动作，而允许自由形式语言输出，以获得更强表达力。然后 SentenceBERT 模块将这些自由形式自然语言输出映射为环境中的可执行动作。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_16_58_d65856.jpg"/>



Figure 4: Choice of direction for the exploration heuristic: The agent (Alice) rotates towards 4 cardinal directions to get observations. The cosine similarity between the CLIP embeddings ${I}_{d}$ for these 4 images are calculated with the CLIP embeddings for each subtask in the open subtasks set ${\mathcal{G}}_{O}$ to get the exploration score ${\mathcal{E}}_{d}$ for each direction. The direction with the highest ${\mathcal{E}}_{d}$ is chosen to explore and the agent moves $J = 2$ steps in that direction.
图4：用于探索启发式的方向选择：代理（Alice）向4个主方向旋转以获取观测。计算这4张图像的 CLIP 嵌入 ${I}_{d}$ 与开放子任务集合中每个子任务的 CLIP 嵌入 ${\mathcal{G}}_{O}$ 的余弦相似度，以得到每个方向的探索得分 ${\mathcal{E}}_{d}$。选择具有最高 ${\mathcal{E}}_{d}$ 的方向进行探索，代理朝该方向移动 $J = 2$ 步。


Algorithm 1 AI2Thor Exploration Heuristic
算法 1 AI2Thor 探索启发式


---



Input: Agent ID $n$ ,Environment env,number of exploration steps $K$ ,number of move steps $J$
输入: 代理 ID $n$ , 环境 env, 探索步数 $K$ , 移动步数 $J$


	${g}_{O} = {\operatorname{CLIP}}_{\text{ text }}\left( {\mathcal{G}}_{O}\right)$



	while $k < K$ do
	while $k < K$ do


		Exploration Score $\mathcal{E} \in  {\mathbb{R}}^{4} \leftarrow  \mathbf{0}$
		探索得分 $\mathcal{E} \in  {\mathbb{R}}^{4} \leftarrow  \mathbf{0}$


		for $d \in  \{$ North,South,East,West $\}$ do
		for $d \in  \{$ 北、南、东、西 $\}$ do


			${o}_{n,d} =$ env.step(Rotate(Right,n))
			${o}_{n,d} =$ env.step(Rotate(Right,n))


			${I}_{d} = {\operatorname{CLIP}}_{\text{ img }}\left( {o}_{n,d}\right)$



			${\mathcal{E}}_{d} = \frac{{I}_{d} \cdot  {g}_{O}}{\begin{Vmatrix}{I}_{d}\end{Vmatrix}\begin{Vmatrix}{g}_{O}\end{Vmatrix}}$



		end for
		end for


		${d}^{ * } = \arg \mathop{\max }\limits_{d}\mathcal{E}$



		while $j < J$ do
		while $j < J$ do


			${o}_{i} = \operatorname{env.step}\left( {\operatorname{Move}\left( {{d}^{ * },n}\right) }\right)$



			$j \leftarrow  j + 1$



		end while
		end while


		$k \leftarrow  k + 1$



	end while
	end while


---



## C MAP-THOR Task Types
## C MAP-THOR 任务类型


The complete list of task for each task type:
每种任务类型的完整任务列表：


- Explicit object type, explicit quantity and target:
- 明确对象类型、明确数量与目标：


- put bread, lettuce, and a tomato in the fridge
- 将面包、生菜和一个番茄放入冰箱


- Put the pots and pans on the stove burners
- 将锅碗放在炉灶的炉眼上


- Slice the bread and tomato and crack the egg
- 把面包和番茄切片并把鸡蛋敲开


- Put the butter knife, bowl, and mug in the sink
- 把黄油刀、碗和杯子放进水槽里


- Turn off the faucet and light if either is on
- 如果水龙头或灯开着，就关掉它们


- Put the tissue box, keys, and plate in the box
- 把纸巾盒、钥匙和盘子放进盒子里


- Put the computer, book, and pen on the couch
- 把电脑、书和笔放到沙发上


- Put the bowl and tissue box on the table
- 把碗和纸巾盒放在桌子上


- Put apple in fridge and switch off the light
- 把苹果放进冰箱并关灯


- Put the watch and Keychain inside the drawer
- 把手表和钥匙圈放进抽屉里


- Wash the bowl, mug, pot, and pan
- 洗碗、杯子、锅和煎锅


- Put the Box on the sofa and the bowl in the box
- 把盒子放在沙发上，把碗放进盒子里


- Explicit object type and explicit target: Here we explicitly describe the object type but keep the quantity of the objects ambiguous. E.g. Put all the apples in the fridge. For this, the agents have to explore the environment to ensure that they find all of them.
- 明确物体类型和明确目标：这里我们明确描述物体类型但保持数量不明。例如，把所有苹果都放进冰箱。为此，执行者必须探索环境以确保找到所有苹果。


- Open all the drawers
- 打开所有抽屉


- Open all the cabinets
- 打开所有橱柜


- Turn on all the stove knobs
- 打开所有炉灶旋钮


- Put all the vases on the table
- 把所有花瓶放在桌子上


- Put all the potatoes in the bowl
- 把所有土豆放进碗里


- Put all pencils and pens in the box
- 把所有铅笔和钢笔放进盒子里


- Move all lamps next to the door
- 将所有门旁的灯移动到门旁


- Turn off all light switches
- 关闭所有灯开关


- Turn on all light switches
- 打开所有灯开关


- Explicit target, implicit object types: The object types are implicitly defined whereas the target is explicitly defined. E.g. Put all groceries in the fridge. This tests whether the model can identify objects of certain categories.
- 明确目标，隐式对象类型：对象类型是隐式定义的，而目标是明确的。例如：把所有杂货放进冰箱。这测试模型是否能识别某些类别的物体。


- Put all groceries in the fridge (should identify the tomato, bread, apple, potato, and lettuce)
- 把所有杂货放进冰箱（应识别番茄、面包、苹果、土豆和生菜）


- Put all shakers in the closest drawer (should identify the salt shaker and pepper shaker)
- 把所有调味罐放进最近的抽屉（应识别盐罐和胡椒罐）


- Put all tableware on the countertop (should identify the bowl, plate, mug)
- 把所有餐具放在台面上（应识别碗、盘子、杯子）


- Put all food on the countertop (should identify the tomato, bread, apple, potato, and lettuce)
- 把所有食物放在台面上（应识别番茄、面包、苹果、土豆和生菜）


- Put all school supplies on the couch (should identify the pencil, computer, and book)
- 把所有文具放在沙发上（应识别铅笔、电脑和书）


- Put all kitchenware in the cardboard box (should move the bowl and plate)
- 把所有厨具放进纸箱（应移动碗和盘子）


- Put all silverware in the sink
- 把所有餐具放进水槽


- Move everything on the table to the desk (should move the laptop, pencil, pen, plate, credit card, book, and newspaper)
- 把桌上的所有东西移到书桌上（应移动笔记本、铅笔、圆珠笔、盘子、信用卡、书和报纸）


- Slice the lettuce, trash the mug and switch off the light
- 切生菜，扔掉杯子并关灯


- Put all electronics on the couch
- 把所有电子设备放到沙发上


- Make a dish by microwaving eggs and tomato
- 用微波炉把鸡蛋和番茄做成一道菜


- Put all readable objects on the sofa
- 把所有可阅读的物品放在沙发上


- Wash all fruits
- 清洗所有水果


- Implicit target and object types: Here both the object type and the target are implicitly defined. E.g. Clear the floor by placing the items at their appropriate positions. Here the model is expected to keep items like pens, book, laptop on the study table, litter in the trash can, etc.
- 隐含的目标和对象类型：此处对象类型和目标均为隐含定义。例如，通过将物品放到合适位置来清理地面。模型预期将像钢笔、书、笔记本放到书桌上，垃圾放入垃圾桶等。


- Clear the floor by placing items at their appropriate positions (depending on what's on the floor)
- 通过将物品放到合适位置来清理地面（取决于地面上有什么）


- Clear the table by placing the items in their appropriate positions (depends on the floorplan, e.g. bread, apple, tomato, knife, bowl, book)
- 通过将物品放到合适位置来清理桌面（取决于平面布局，例如面包、苹果、西红柿、刀、碗、书）


- Clear the countertop by placing items in their appropriate positions (should move the lettuce, mug, and paper towel roll)
- 通过将物品放到合适位置来清理台面（应移动生菜、杯子和纸巾卷）


- Clear the desk by placing the items in other appropriate positions (should move the statue, watch, and remote control)
- 通过将物品放到其他合适位置来清理书桌（应移动雕像、手表和遥控器）


- Clear the table by placing the items in other appropriate positions (should move the book, credit card, laptop, plate, newspaper, pen, and pencil)
- 通过将物品放到其他合适位置来清理桌子（应移动书、信用卡、笔记本、盘子、报纸、钢笔和铅笔）


- Clear the couch by placing the items in other appropriate positions (should move the pillow) - Make the living room dark
- 通过将物品放到其他合适位置来清理沙发（应移动枕头） - 把客厅弄暗


- Make a mug of coffee and toast the bread
- 做一杯咖啡并把面包烤成吐司


- Trash all groceries
- 把所有杂货扔掉


- Slice all sliceable objects
- 切所有可切的物体


## D Search & Rescue Environment (SAR)
## D 搜救环境（SAR）


The Search & Rescue environment consists of multiple agents in an unknown environment that has multiple wildfires and missing personnel in the environment. The agents are tasked to extinguish all the fires before they spread and rescue the missing humans. Here, each fire is composed of a large flammable region with a fixed set of sources that spread through time. The higher the intensity, the faster the fire will spread. The fires can be of class A or B, extinguished through the use of water and sand respectively. Both these resources can be collected through resource reservoirs spread geographically. Each person is initially stranded in an unknown location. The goal is to rescue and transport each person to a drop-off location (known apriori). The person must have two agents simultaneously carrying them to be transported to the drop-off location.
搜救环境由多个智能体组成，位于一个未知环境中，该环境存在多处野火和失踪人员。智能体的任务是在火势蔓延前扑灭所有火源并救出失踪人员。这里，每处火情由一个大面积易燃区域和一组随时间扩散的固定火源组成。强度越大，火势扩散越快。火分为 A 类或 B 类，分别用水和沙子扑灭。这两种资源可从地理分布的资源水池中收集。每名人员最初被困在未知位置。目标是救出并将每名人员运送到事先已知的放置点。运输时必须有两名智能体同时抬着该人员才能将其运送到放置点。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_16_58_66a715.jpg"/>



Figure 5: The search & rescue environment consists of multiple drones in an unknown environment with missing people, fires of different types, and water and sand reservoirs.
图 5：搜救环境由多架无人机组成，位于一个未知环境中，含有失踪人员、不同类型的火灾以及水和沙的储备。


We provide a justification to why this task is relevant to long-term planning and multi-agent collaboration below.
我们在下面给出该任务为何与长期规划和多智能体协作相关的理由。


### D.1 Multi-agent Nature
### D.1 多智能体特性


- Scalability - Unlike the AI2THOR environment, the search & rescue environment is more spacious and hence prevents congestion amongst agents. With the addition of multiple parallel emergency disaster scenarios at once, the environment scales comfortably to more agents. Additionally, the complexity slowly increases as the model has to coordinate between more actors and competing emergencies.
- 可扩展性 - 与 AI2THOR 环境不同，搜救环境空间更大，从而避免了智能体之间的拥堵。通过同时加入多个并行的紧急灾难场景，环境可以轻松扩展以容纳更多智能体。此外，随着模型需要在更多参与者和相互竞争的紧急事件之间协调，复杂性会逐步增加。


- Tasks requires explicit cooperation -
- 任务需要明确合作 -


- To move a group of humans, at least 2 agents are required. Thus, explicit multi-agent collaboration is required.
- 要移动一组人，至少需要 2 个智能体。因此，需要明确的多智能体协作。


- Due to the time pressure, in order to successfully stop the fire, agents must all effectively collaborate in collecting and strategically using resources in high-intensity regions.
- 由于时间压力，为了成功扑灭火灾，智能体必须在高强度区域中有效协作，收集并战略性地使用资源。


- Exploration & Task Assignment - There is an explicit tradeoff between allocating agents towards exploring to find lost people and fighting the current fires.
- 探索与任务分配 - 在将智能体分配去探索以寻找失踪人员与去扑灭当前火灾之间存在明确的权衡。


### D.2 Long-term Planning
### D.2 长期规划


- Dependency Chain: To resolve a fire it requires that the source is identified, the type of fire is identified, and the appropriate resources to stop it are acquired and then used.
- 依赖链：要解决一起火灾，需要先确定起火源、识别火灾类型、获取合适的资源，然后再使用这些资源。


- Uncertainty: Inherent uncertainty as to where the lost people can be found, and at what point in the timeline they will.
- 不确定性：关于失踪人员可能在哪里以及他们会在时间线上的何时出现存在固有不确定性。


- Irreversibility and forced consequences - With a fire that spreads, present actions have irreversible future consequences. A balance needs to be struck between fighting the fire's source (to stop it from continuing) versus a periphery (to prevent geographic spread).
- 不可逆性与被迫后果 - 对于会蔓延的火灾，当前的行动会带来不可逆的未来后果。需要在扑灭火源（以阻止其继续蔓延）与处理周边（以防止地理扩散）之间找到平衡。


### D.3 Description of Scenes
### D.3 场景描述


We evaluate LLaMAR in five different scenarios in the search & rescue environment. Here each scene is evaluated on 5 different random seeds.
我们在搜救环境中对 LLaMAR 在五种不同场景中进行了评估。每个场景均在 5 个不同随机种子上评估。


- Scene 1: This consists of 1 Type A fire with 2 initial sources, 1 Type B fire with 1 initial source, and 1 lost person at a random location in the environment.
- 场景 1：包括 1 个 A 型火灾，具有 2 个初始起火源，1 个 B 型火灾具有 1 个初始起火源，以及 1 名位于环境中随机位置的失踪人员。


- Scene 2: This consists of 1 Type A fire with 1 initial source, 1 Type B fire with 2 initial sources, and 1 lost person at a random location in the environment.
- 场景 2：包括 1 个 A 型火灾具有 1 个初始起火源，1 个 B 型火灾具有 2 个初始起火源，以及 1 名位于环境中随机位置的失踪人员。


- Scene 3: This consists of 2 Type A fires each with 1 initial source, 1 Type B fire with 1 initial source, and 1 lost person at a random location in the environment.
- 场景 3：包括 2 个各有 1 个初始起火源的 A 型火灾，1 个 B 型火灾具有 1 个初始起火源，以及 1 名位于环境中随机位置的失踪人员。


- Scene 4: This consists of 1 Type A fire with 3 initial sources, and 1 lost person at a random location in the environment.
- 场景 4：包括 1 个 A 型火灾具有 3 个初始起火源，以及 1 名位于环境中随机位置的失踪人员。


- Scene 5: This consists of 1 Type A fire with 1 initial source, 1 Type B fire with 1 initial source, and 2 lost persons at random locations in the environment.
- 场景 5：包括 1 个 A 型火灾具有 1 个初始起火源，1 个 B 型火灾具有 1 个初始起火源，以及 2 名位于环境中随机位置的失踪人员。


### D.4 Observation Space
### D.4 观测空间


The agent’s observation $\mathcal{O} = {\mathcal{O}}_{G} \cup  {\mathcal{O}}_{L} \cup  {\mathcal{O}}_{L}$ is a union of the following observations.
智能体的观测 $\mathcal{O} = {\mathcal{O}}_{G} \cup  {\mathcal{O}}_{L} \cup  {\mathcal{O}}_{L}$ 是以下观测的并集。


Global Observations ${\mathcal{O}}_{G}$ consists of all the objects that are globally visible by the agent:
全局观测 ${\mathcal{O}}_{G}$ 包含智能体全局可见的所有对象：


- Fires: If visible, fire name, type, and average intensity.
- 火灾：若可见，火灾名称、类型和平均强度。


- Fire Regions: If fire visible, fire region name, type, and average intensity.
- 火区：若火可见，火区名称、类型和平均强度。


- Reservoir: If visible, reservoir name, and reservoir resource type.
- 水库：若可见，水库名称和水库资源类型。


- Deposit: If visible, deposit name, inventory of all the resources (water, sand) and persons in deposit.
- 储备点：若可见，储备点名称、所有资源的库存（如水、沙）以及储备点中的人员。


- Person: If visible, person name, carried or not status, dropped-off or not status.
- 人员：若可见，人员名称、是否被携带状态、是否已放下状态。


- Agent inventory: List of all the resources (water, sand) and person (being carried) in the agent's inventory.
- 智能体背包：智能体背包中所有资源（如水、沙）和被携带人员的列表。


Local Observations ${\mathcal{O}}_{L}$ consists of all the objects that are visible in grid cells adjacent to the agent:
局部观测 ${\mathcal{O}}_{L}$ 包含智能体相邻网格单元中可见的所有对象：


- For all the directions <direction> in (Up, Down, Left, Right, and Center), output of the following.
- 对于所有方向 <direction>（上、下、左、右和中心），输出以下内容。


- <direction>: At direction <direction>, either 'Empty' if there is no object, 'Flammable' along with corresponding intensity & fire name if object is a part of a fire, or 'Obstacle' for any other object.
- <direction>：在方向 <direction> 上，若无对象则为 'Empty'，若对象属于火的一部分则为 'Flammable' 连同相应强度和火名，其他对象则为 'Obstacle'。


Names ${\mathcal{O}}_{N}$ consists of a list of the name of all visible and interactable objects.
名称 ${\mathcal{O}}_{N}$ 包含所有可见且可交互对象的名称列表。


### D.5 Action Space
### D.5 动作空间


The action space $\mathcal{A}$ consists of navigation actions ${\mathcal{A}}_{NAV}$ ,interaction actions ${\mathcal{A}}_{INT}$ ,exploration action ${\mathcal{A}}_{EXP}$ .
动作空间 $\mathcal{A}$ 包含导航动作 ${\mathcal{A}}_{NAV}$、交互动作 ${\mathcal{A}}_{INT}$、探索动作 ${\mathcal{A}}_{EXP}$。


Navigation actions ${\mathcal{A}}_{NAV}$ consists of the following actions:
导航动作 ${\mathcal{A}}_{NAV}$ 包含以下动作：


- Move (<direction>): Moves the agent to the neighboring grid cell in the specified direction where <direction> can be one of (Up, Down, Left, Right, and Center)
- Move (<direction>): 将智能体移动到指定方向的相邻网格单元，<direction> 可为 (Up, Down, Left, Right, Center)


- NavigateTo(<targetID>): Moves the agent next to the location of the object <targetID> if <targetID> is visible.
- NavigateTo(<targetID>): 如果 <targetID> 可见，则将智能体移动到对象 <targetID> 所在位置的附近。


Interaction actions ${\mathcal{A}}_{INT}$ consists of the following actions:
Interaction actions ${\mathcal{A}}_{INT}$ consists of the following actions:


- Carry (<person>): Makes agent carry a person <person> if <person> is visible and inter-actable. The person is successfully 'group' carried, if at least the required number of agents successfully does the 'Carry (<person>)' action. If carry action is successful, all other resources in agent's inventory are dropped.
- Carry (<person>): 使智能体携带可见且可交互的人员 <person>。当至少所需数量的智能体成功执行 'Carry (<person>)' 行为时，该人员视为被“群体”携带。若携带成功，智能体背包中所有其他资源将被丢弃。


- DropOff (<person>, <deposit>): Drops off person <person> at location <deposit>. This action is only successful if the person has been 'group' carried, all the agents carrying the person have the deposit be visible and interactable, and after all the agents do this action.
- DropOff (<person>, <deposit>): 在位置 <deposit> 放下人员 <person>。只有当该人员已被“群体”携带、所有携带该人员的智能体都能看到且能交互该存放点，并且所有这些智能体都执行此动作时，该动作才成功。


- StoreSupply(<deposit>): Stores all the resources from the agent's current inventory in the deposit <deposit>.
- StoreSupply(<deposit>): 将智能体当前背包中的所有资源存入存放点 <deposit>。


- UseSupply(<fire>, <supply-type>): Uses all the supplies of type <supply-type> on the fire <fire> at the location the agent is at.
- UseSupply(<fire>, <supply-type>): 在智能体所在位置对火源 <fire> 使用所有类型为 <supply-type> 的补给品。


- GetSupply(<deposit>, <supply-type>): Fills the remaining of the agent's inventory space with the available supply of type <supply-type> in <deposit>.
- GetSupply(<deposit>, <supply-type>): 用 <deposit> 中可用的 <supply-type> 补给填满智能体背包剩余空间。


- GetSupply(<reservoir>): Collects 1 unit of supply from reservoir <reservoir> and stores it in the agent's inventory.
- GetSupply(<reservoir>): 从水源/补给源 <reservoir> 收集1单位补给并存入智能体背包。


Exploration action ${\mathcal{A}}_{EXP}$ consists of the following actions:
Exploration action ${\mathcal{A}}_{EXP}$ consists of the following actions:


- Explore(): Takes the agent in a direction of exploration as described by the heuristic exploration function described in algorithm 2.
- Explore(): 按算法2中描述的启发式探索函数指示的探索方向移动智能体。


Algorithm 2 SAR Exploration Heuristic
Algorithm 2 SAR Exploration Heuristic


---



Input: Agent A, Environment env, Previous Direction D, Current Position P, Max steps M
Input: Agent A, Environment env, Previous Direction D, Current Position P, Max steps M


Initialize: New direction ${ND} \leftarrow  \varnothing$
Initialize: New direction ${ND} \leftarrow  \varnothing$


	${ND} \leftarrow$ randomly choose an angle from $\lbrack 0,{2\pi })$ at $\pi /4$ intervals
	${ND} \leftarrow$ randomly choose an angle from $\lbrack 0,{2\pi })$ at $\pi /4$ intervals


	while ${ND} = D$ or ${ND} \equiv  D + \pi {\;\operatorname{mod}\;2}\pi$
	while ${ND} = D$ or ${ND} \equiv  D + \pi {\;\operatorname{mod}\;2}\pi$


			or a barrier exists at most $\frac{3}{4}M$ steps in direction ${ND}$ from $P$ do
			或在从 $P$ 朝 ${ND}$ 方向最多 $\frac{3}{4}M$ 步处存在障碍


		${ND} \leftarrow$ randomly choose an angle from $\lbrack 0,{2\pi })$ at $\pi /4$ intervals
		${ND} \leftarrow$ 在 $\pi /4$ 间隔内从 $\lbrack 0,{2\pi })$ 随机选择一个角度


	end while
	结束循环


	Move agent $A,M$ steps in direction ${ND}$ in environment env
	在环境 env 中将智能体朝 ${ND}$ 方向移动 $A,M$ 步


---



## E Pseudocode for LLaMAR
## LLaMAR 的伪代码


Algorithm 3 LLaMAR
算法 3 LLaMAR


---



Input: $N$ agents,Task instruction $\mathcal{I}$ ,Environment env
输入：$N$ 个智能体，任务指令 $\mathcal{I}$，环境 env


Initialize: Memory $\mathcal{M} \leftarrow  \varnothing$ ; Open Subtasks ${\mathcal{G}}_{O} \leftarrow  \varnothing$ ;
初始化：记忆 $\mathcal{M} \leftarrow  \varnothing$；未完成子任务 ${\mathcal{G}}_{O} \leftarrow  \varnothing$；


Completed Subtasks ${\mathcal{G}}_{C} \leftarrow  \varnothing$ ; Actions $a \leftarrow  \varnothing$ ;
已完成子任务 ${\mathcal{G}}_{C} \leftarrow  \varnothing$；动作 $a \leftarrow  \varnothing$；


Corrective Actions ${a}_{c} \leftarrow  \varnothing$
纠正动作 ${a}_{c} \leftarrow  \varnothing$


Actions Executed $d \leftarrow  \varnothing$
已执行动作 $d \leftarrow  \varnothing$


	$o = \left( {{o}_{1},\cdots ,{o}_{N}}\right)  =$ env.reset $\left( \right)$
	$o = \left( {{o}_{1},\cdots ,{o}_{N}}\right)  =$ env.reset $\left( \right)$


	while $t < T$ do
	当 $t < T$ 时


		${\mathcal{G}}_{O} \leftarrow  \operatorname{Planner}\left( {\mathcal{I},o,{\mathcal{G}}_{O},{\mathcal{G}}_{C},\mathcal{M}}\right)$



		$a,\mathcal{M} \leftarrow  \operatorname{Actor}\left( {\mathcal{I},o,{a}_{c},{\mathcal{G}}_{O},{\mathcal{G}}_{C},\mathcal{M}}\right)$



		$o = \left( {{o}_{1},\cdots ,{o}_{N}}\right) ,d = \left( {{d}_{1},\cdots ,{d}_{N}}\right)  =$ env.step $\left( a\right)$
		 $o = \left( {{o}_{1},\cdots ,{o}_{N}}\right) ,d = \left( {{d}_{1},\cdots ,{d}_{N}}\right)  =$ env.step $\left( a\right)$


		${a}_{c} \leftarrow  \operatorname{Corrector}\left( {\mathcal{I},o,a,d,{\mathcal{G}}_{O},{\mathcal{G}}_{C},\mathcal{M}}\right)$



		${\mathcal{G}}_{C} \leftarrow  \operatorname{Verifier}\left( {\mathcal{I},o,a,d,{\mathcal{G}}_{O},{\mathcal{G}}_{C},\mathcal{M}}\right)$



		if ${\mathcal{G}}_{O} = \varnothing$ then
		 如果 ${\mathcal{G}}_{O} = \varnothing$ 则


			break
			  退出


		end if
		结束 if


		$t \leftarrow  t + 1$



	end while
		结束 while


---



## F Full Results
## F 完整结果


<table><tr><td>Algorithm</td><td>LM</td><td>Success Rate</td><td>Transport Rate</td><td>Coverage</td><td>Balance</td><td>Steps</td></tr><tr><td>Act</td><td>GPT-4V</td><td>0.33 (0.19, 0.49)</td><td>0.67 (0.59, 0.76)</td><td>0.91 (0.86,0.95)</td><td>0.59 (0.52, 0.66)</td><td>24.92 (22.12,27.73)</td></tr><tr><td>ReAct</td><td>GPT-4V</td><td>0.34 (0.20, 0.49)</td><td>0.72 (0.63,0.80)</td><td>0.92 (0.86, 0.97)</td><td>0.67 (0.61, 0.73)</td><td>24.08 (21.27, 26.89)</td></tr><tr><td>CoT</td><td>GPT-4V</td><td>0.14 (0.06, 0.28)</td><td>0.59 (0.51, 0.67)</td><td>0.87 (0.81, 0.92)</td><td>0.62 (0.56,0.69)</td><td>28.40 (26.91, 29.97)</td></tr><tr><td>SmartLLM</td><td>GPT-4V</td><td>0.11 (0.05, 0.23)</td><td>0.23 (0.13, 0.31)</td><td>0.91 (0.80, 0.96)</td><td>0.45 (0.37, 0.52)</td><td>29.87 (26.20, 30.00)</td></tr><tr><td>CoELA</td><td>GPT-4V</td><td>0.25 (0.10, 0.36)</td><td>0.46 (0.35, 0.56)</td><td>0.76 (0.67, 0.85)</td><td>0.73 (0.67, 0.80)</td><td>28.93 (27.77,30.00)</td></tr><tr><td>LLaMAR</td><td>GPT-4</td><td>0.51 (0.36, 0.66)</td><td>0.85 (0.80, 0.91)</td><td>0.95 (0.91, 0.98)</td><td>0.83 (0.78, 0.86)</td><td>25.80 (23.72,27.88)</td></tr><tr><td>LLaMAR</td><td>LLaVA</td><td>0.54 (0.41, 0.65)</td><td>0.84 (0.71, 0.90)</td><td>0.91 (0.87, 0.98)</td><td>0.75 (0.64, 0.83)</td><td>26.21 (21.56, 28.97)</td></tr><tr><td>LLaMAR</td><td>IDEFICS-2</td><td>0.57 (0.43, 0.67)</td><td>0.86 (0.74, 0.91)</td><td>0.94 $\left( {{0.89},{0.98}}\right)$</td><td>0.78 (0.65, 0.84)</td><td>25.27 (20.14, 28.37)</td></tr><tr><td>LLaMAR</td><td>CogVLM</td><td>0.61 (0.47, 0.68)</td><td>0.89 (0.73, 0.95)</td><td>0.95 (0.89, 0.99)</td><td>0.80 (0.73, 0.86)</td><td>23.21 (20.57, 26.82)</td></tr><tr><td>LLaMAR (w/o exploration)</td><td>GPT-4V</td><td>0.62 (0.46, 0.76)</td><td>0.87 (0.80, 0.93)</td><td>0.95 (0.91, 0.98)</td><td>0.82 (0.77, 0.87)</td><td>23.44 (20.88, 26.00)</td></tr><tr><td>LLaMAR (w/ exploration)</td><td>GPT-4V</td><td>0.66 (0.50, 0.78)</td><td>0.91 (0.81,0.96)</td><td>0.97 (0.93,0.99)</td><td>0.82 (0.75,0.87)</td><td>21.87 (18.76,24.23)</td></tr></table>
<table><tbody><tr><td>算法</td><td>语言模型</td><td>成功率</td><td>运输率</td><td>覆盖率</td><td>平衡度</td><td>步骤数</td></tr><tr><td>行动</td><td>GPT-4V</td><td>0.33 (0.19, 0.49)</td><td>0.67 (0.59, 0.76)</td><td>0.91 (0.86,0.95)</td><td>0.59 (0.52, 0.66)</td><td>24.92 (22.12,27.73)</td></tr><tr><td>ReAct</td><td>GPT-4V</td><td>0.34 (0.20, 0.49)</td><td>0.72 (0.63,0.80)</td><td>0.92 (0.86, 0.97)</td><td>0.67 (0.61, 0.73)</td><td>24.08 (21.27, 26.89)</td></tr><tr><td>CoT</td><td>GPT-4V</td><td>0.14 (0.06, 0.28)</td><td>0.59 (0.51, 0.67)</td><td>0.87 (0.81, 0.92)</td><td>0.62 (0.56,0.69)</td><td>28.40 (26.91, 29.97)</td></tr><tr><td>SmartLLM</td><td>GPT-4V</td><td>0.11 (0.05, 0.23)</td><td>0.23 (0.13, 0.31)</td><td>0.91 (0.80, 0.96)</td><td>0.45 (0.37, 0.52)</td><td>29.87 (26.20, 30.00)</td></tr><tr><td>CoELA</td><td>GPT-4V</td><td>0.25 (0.10, 0.36)</td><td>0.46 (0.35, 0.56)</td><td>0.76 (0.67, 0.85)</td><td>0.73 (0.67, 0.80)</td><td>28.93 (27.77,30.00)</td></tr><tr><td>LLaMAR</td><td>GPT-4</td><td>0.51 (0.36, 0.66)</td><td>0.85 (0.80, 0.91)</td><td>0.95 (0.91, 0.98)</td><td>0.83 (0.78, 0.86)</td><td>25.80 (23.72,27.88)</td></tr><tr><td>LLaMAR</td><td>LLaVA</td><td>0.54 (0.41, 0.65)</td><td>0.84 (0.71, 0.90)</td><td>0.91 (0.87, 0.98)</td><td>0.75 (0.64, 0.83)</td><td>26.21 (21.56, 28.97)</td></tr><tr><td>LLaMAR</td><td>IDEFICS-2</td><td>0.57 (0.43, 0.67)</td><td>0.86 (0.74, 0.91)</td><td>0.94 $\left( {{0.89},{0.98}}\right)$</td><td>0.78 (0.65, 0.84)</td><td>25.27 (20.14, 28.37)</td></tr><tr><td>LLaMAR</td><td>CogVLM</td><td>0.61 (0.47, 0.68)</td><td>0.89 (0.73, 0.95)</td><td>0.95 (0.89, 0.99)</td><td>0.80 (0.73, 0.86)</td><td>23.21 (20.57, 26.82)</td></tr><tr><td>LLaMAR（无探索）</td><td>GPT-4V</td><td>0.62 (0.46, 0.76)</td><td>0.87 (0.80, 0.93)</td><td>0.95 (0.91, 0.98)</td><td>0.82 (0.77, 0.87)</td><td>23.44 (20.88, 26.00)</td></tr><tr><td>LLaMAR（含探索）</td><td>GPT-4V</td><td>0.66 (0.50, 0.78)</td><td>0.91 (0.81,0.96)</td><td>0.97 (0.93,0.99)</td><td>0.82 (0.75,0.87)</td><td>21.87 (18.76,24.23)</td></tr></tbody></table>


Table 5: Comparison of evaluation metrics against baselines averaged across all tasks for the 2-agent MAP-THOR scenarios.
表5：针对 2 代理 MAP-THOR 场景、相对于基线在所有任务上平均的评估指标对比。


<table><tr><td>Modules Used</td><td>Success Rate</td><td>Transport Rate</td><td>Coverage</td><td>Balance</td><td>Steps</td></tr><tr><td>Actor</td><td>0.33 (0.19, 0.49)</td><td>0.67 (0.59, 0.76)</td><td>0.91 (0.86,0.95)</td><td>0.59 (0.52,0.66)</td><td>24.92 <br> (22.12,27.73)</td></tr><tr><td>Planner + <br> Actor + Verifier</td><td>0.45 <br> (0.29, 0.57)</td><td>0.78 <br> (0.67, 0.84)</td><td>0.92 <br> (0.84, 0.95)</td><td>0.69 <br> (0.61, 0.75)</td><td>24.87 <br> (20.48, 27.95)</td></tr><tr><td>Planner + Actor + Corrector‡</td><td>0.67 <br> (0.51, 0.80)</td><td>0.91 <br> (0.83, 0.96)</td><td>0.97 <br> (0.94, 0.99)</td><td>0.84 <br> (0.79, 0.89)</td><td>22.81 <br> (19.95, 25.76)</td></tr><tr><td>LLaMAR</td><td>0.66 <br> (0.50, 0.76)</td><td>0.91 <br> (0.81, 0.96)</td><td>0.97 <br> (0.93,0.99)</td><td>0.82 <br> (0.75, 0.87)</td><td>21.87 <br> (18.76, 26.43)</td></tr></table>
<table><tbody><tr><td>使用模块</td><td>成功率</td><td>运输率</td><td>覆盖率</td><td>平衡</td><td>步骤</td></tr><tr><td>执行者</td><td>0.33 (0.19, 0.49)</td><td>0.67 (0.59, 0.76)</td><td>0.91 (0.86,0.95)</td><td>0.59 (0.52,0.66)</td><td>24.92 <br/> (22.12,27.73)</td></tr><tr><td>规划器 + <br/> 执行者 + 验证器</td><td>0.45 <br/> (0.29, 0.57)</td><td>0.78 <br/> (0.67, 0.84)</td><td>0.92 <br/> (0.84, 0.95)</td><td>0.69 <br/> (0.61, 0.75)</td><td>24.87 <br/> (20.48, 27.95)</td></tr><tr><td>规划器 + 执行者 + 校正器‡</td><td>0.67 <br/> (0.51, 0.80)</td><td>0.91 <br/> (0.83, 0.96)</td><td>0.97 <br/> (0.94, 0.99)</td><td>0.84 <br/> (0.79, 0.89)</td><td>22.81 <br/> (19.95, 25.76)</td></tr><tr><td>LLaMAR</td><td>0.66 <br/> (0.50, 0.76)</td><td>0.91 <br/> (0.81, 0.96)</td><td>0.97 <br/> (0.93,0.99)</td><td>0.82 <br/> (0.75, 0.87)</td><td>21.87 <br/> (18.76, 26.43)</td></tr></tbody></table>


Table 6: Ablating different modules LLaMAR with GPT-4V as the underlying VLM, 2-agents scenarios.
表 6：在 2 代理场景中，以 GPT-4V 为底层视觉语言模型的 LLaMAR 不同模块消融。


<table><tr><td rowspan="2">#of agents</td><td colspan="5">MAP-THOR</td><td colspan="5">SAR</td></tr><tr><td>Success Rate</td><td>Transport Rate</td><td>Coverage</td><td>Balance</td><td>Steps</td><td>Success Rate</td><td>Transport Rate</td><td>Coverage</td><td>Balance</td><td>Steps</td></tr><tr><td>1</td><td>0.37 <br> (0.21, 0.51)</td><td>0.67 <br> (0.58, 0.74)</td><td>0.87 <br> (0.81, 0.90)</td><td>1.00 (1.00,1.00)</td><td>28.44 <br> (25.23, 30.00)</td><td>0.28</td><td>0.75</td><td>0.86</td><td>1.00</td><td>28</td></tr><tr><td>2</td><td>0.62 <br> (0.46, 0.76)</td><td>0.87 <br> (0.80, 0.93)</td><td>0.95 <br> (0.91, 0.98)</td><td>0.82 <br> (0.77,0.87)</td><td>23.44 <br> (20.88, 26.00)</td><td>0.44 <br> (0.24,0.65)</td><td>0.86 <br> (0.79, 0.94)</td><td>0.94 <br> (0.88, 0.99)</td><td>0.91 <br> (0.88, 0.95)</td><td>27.76 <br> (24.15, 30)</td></tr><tr><td>3</td><td>0.70 <br> (0.55, 0.82)</td><td>0.91 <br> (0.85, 0.95)</td><td>0.98 <br> (0.95, 0.99)</td><td>0.66 <br> (0.61, 0.71)</td><td>21.30 <br> (18.60, 23.99)</td><td>0.68 <br> (0.46, 0.85)</td><td>0.92 <br> (0.86, 0.98)</td><td>0.96 <br> (0.91,1.0)</td><td>0.80 <br> (0.73, 0.86)</td><td>21.88 <br> (17.83, 25.92)</td></tr><tr><td>4</td><td>0.68 <br> (0.52, 0.79)</td><td>0.90 <br> (0.84, 0.94)</td><td>0.99 <br> (0.95, 0.99)</td><td>0.62 <br> (0.57,0.68)</td><td>22.83 <br> (19.63, 25.69)</td><td>0.72 <br> (0.50, 0.85)</td><td>0.94 <br> (0.88,0.98)</td><td>0.98 <br> (0.93,1.00)</td><td>0.78 <br> (0.74, 0.83)</td><td>22.00 <br> (17.96, 26.03)</td></tr><tr><td>5</td><td>0.62 <br> (0.46, 0.75)</td><td>0.90 <br> (0.85, 0.94)</td><td>0.99 <br> (0.97,1.00)</td><td>0.54 <br> (0.48, 0.59)</td><td>22.91 <br> (20.26, 25.57)</td><td>0.74 <br> (0.52, 0.86)</td><td>0.96 <br> (0.94, 0.99)</td><td>1.00 <br> (1.0,1.0)</td><td>0.73 <br> (0.67, 0.79)</td><td>24.52 <br> (20.24,28.79)</td></tr></table>
<table><tbody><tr><td rowspan="2">代理数量</td><td colspan="5">MAP-THOR</td><td colspan="5">SAR</td></tr><tr><td>成功率</td><td>运输率</td><td>覆盖率</td><td>平衡性</td><td>步数</td><td>成功率</td><td>运输率</td><td>覆盖率</td><td>平衡性</td><td>步数</td></tr><tr><td>1</td><td>0.37 <br/> (0.21, 0.51)</td><td>0.67 <br/> (0.58, 0.74)</td><td>0.87 <br/> (0.81, 0.90)</td><td>1.00 (1.00,1.00)</td><td>28.44 <br/> (25.23, 30.00)</td><td>0.28</td><td>0.75</td><td>0.86</td><td>1.00</td><td>28</td></tr><tr><td>2</td><td>0.62 <br/> (0.46, 0.76)</td><td>0.87 <br/> (0.80, 0.93)</td><td>0.95 <br/> (0.91, 0.98)</td><td>0.82 <br/> (0.77,0.87)</td><td>23.44 <br/> (20.88, 26.00)</td><td>0.44 <br/> (0.24,0.65)</td><td>0.86 <br/> (0.79, 0.94)</td><td>0.94 <br/> (0.88, 0.99)</td><td>0.91 <br/> (0.88, 0.95)</td><td>27.76 <br/> (24.15, 30)</td></tr><tr><td>3</td><td>0.70 <br/> (0.55, 0.82)</td><td>0.91 <br/> (0.85, 0.95)</td><td>0.98 <br/> (0.95, 0.99)</td><td>0.66 <br/> (0.61, 0.71)</td><td>21.30 <br/> (18.60, 23.99)</td><td>0.68 <br/> (0.46, 0.85)</td><td>0.92 <br/> (0.86, 0.98)</td><td>0.96 <br/> (0.91,1.0)</td><td>0.80 <br/> (0.73, 0.86)</td><td>21.88 <br/> (17.83, 25.92)</td></tr><tr><td>4</td><td>0.68 <br/> (0.52, 0.79)</td><td>0.90 <br/> (0.84, 0.94)</td><td>0.99 <br/> (0.95, 0.99)</td><td>0.62 <br/> (0.57,0.68)</td><td>22.83 <br/> (19.63, 25.69)</td><td>0.72 <br/> (0.50, 0.85)</td><td>0.94 <br/> (0.88,0.98)</td><td>0.98 <br/> (0.93,1.00)</td><td>0.78 <br/> (0.74, 0.83)</td><td>22.00 <br/> (17.96, 26.03)</td></tr><tr><td>5</td><td>0.62 <br/> (0.46, 0.75)</td><td>0.90 <br/> (0.85, 0.94)</td><td>0.99 <br/> (0.97,1.00)</td><td>0.54 <br/> (0.48, 0.59)</td><td>22.91 <br/> (20.26, 25.57)</td><td>0.74 <br/> (0.52, 0.86)</td><td>0.96 <br/> (0.94, 0.99)</td><td>1.00 <br/> (1.0,1.0)</td><td>0.73 <br/> (0.67, 0.79)</td><td>24.52 <br/> (20.24,28.79)</td></tr></tbody></table>


Table 7: LLaMAR with more agents
表7：更多智能体的 LLaMAR


## G Failure Cases
## G 失败案例


One of the main motivations of our paper was to achieve better success rates in planning for a variety of tasks compared to previous approaches. While our method outperforms other baselines, we acknowledge that the success rate is still under the expectation for real-world deployment. We believe that our approach LLaMAR and the MAP-THOR benchmark would serve as a starting point for future research to improve the success rates in multi-agent embodied robotics tasks. Here we describe a few major types of failures in our experiments:
本文的主要动机之一是比以往方法在多种任务的规划成功率上取得更好表现。尽管我们的方法优于其他基线，但我们承认成功率仍低于实际部署的期望。我们认为 LLaMAR 方法和 MAP-THOR 基准可以作为未来研究的起点，以提高多智能体实体机器人任务的成功率。下面描述了我们实验中的几类主要失败情况：


- Mis-generalization: The agent can sometime fail to properly infer the level of abstraction to perform a task at, even if it should be clear from the action space. For example, for the washing tasks, the LM assumes that it must put the objects in the sink and add some, rather than just using the "CleanObject" action. We observe this error in the direction of performing actions more low-level than necessary, and not the other way around (more high-level than is feasible).
- 错误的泛化：智能体有时无法正确推断执行任务所需的抽象层级，即便从动作空间应当很清楚。例如在清洗任务中，语言模型会假定必须把物体放入水槽并加入某些东西，而不是直接使用“CleanObject”动作。我们观察到的错误表现为执行比必要更低层次的动作，而不是执行比可行更高层次的动作。


- Mutual Interference (limited spatial reasoning): The agents sometimes block each other and thus fail to carry out actions such as placing an object on a receptacle, opening an object, etc. In particular, we see this behavior in the "put objects on the sofa" and the "put objects in a box" tasks, where the LM does not prevent the agents from blocking each other.
- 相互干扰（有限的空间推理）：智能体有时会互相阻挡，从而无法执行诸如将物体放到容器上、打开物体等动作。我们在“把物品放到沙发上”和“把物品放进盒子里”任务中尤其看到这种行为，语言模型没有阻止智能体互相阻挡。


- Improper Object Visibility (un-observability bias): The LM often fails to prioritize exploration for tasks that require objects not yet seen, simply due to it no being able to see it. This bias causes it to improperly assume that further exploration is not necessary, so it fails to find relevant objects.
- 不当的物体可见性（不可观测性偏差）：语言模型经常未能优先进行探索以寻找尚未看见的物体，仅仅因为它当前看不到该物体。这种偏差导致它错误地认为无需进一步探索，从而无法找到相关物体。


- SentenceBERT mismatch: The free-form natural language output from the Actor is incorrectly mapped to the executable action. Based on our experimental results, 96.7% of the time the free-form natural language was correctly mapped to the feasible actions. The 2 main reasons for failures in the few cases it failed were:
- SentenceBERT 不匹配：Actor 的自由形式自然语言输出被错误映射为可执行动作。根据我们的实验结果，96.7% 的情况下自由形式自然语言被正确映射为可行动作。少数失败的两个主要原因是：


- Incorrect mapping of objects: When the agents have not explored the environment enough and the Actor module suggests the agents interact with objects that are not yet available in the agents' combined memory. Example: The free-form output "navigate to the table" was mapped to the action "NavigateTo(ArmChair)".
- 物体映射错误：当智能体尚未充分探索环境且 Actor 模块建议与尚未出现在智能体合并记忆中的物体交互时会发生此问题。例如：自由格式输出“navigate to the table”被映射为动作“NavigateTo(ArmChair)”。


- Wrong object numeration: When there are multiple objects of the same type in the memory, the sentenceBERT maps "Cabinet_3" to "Cabinet_1" which sometimes leads to incorrect actions being executed for the completion of the plan.
- 物体编号错误：当记忆中存在多个相同类型物体时，sentenceBERT 会把“Cabinet_3”映射为“Cabinet_1”，这有时会导致为完成计划执行了错误的动作。


- Limited horizon: While each task is solvable by a single-agent human-controlled agent within our chosen episode length of $L = {30}$ ,this is under the assumption of no failures during control and maximum visibility. For the autonomous agents the visibility distance is restricted to 1.5 meters, so while a human could identify objects from afar (from the rendered images) the agents have to explore the environment under a stricter partial observability constraint.
- 有限的步长：在我们选择的回合长度为 $L = {30}$ 的假设下，每个任务都可由单一人工控制的智能体完成，但这是基于控制过程中没有失败且可见性最大化的前提。对于自主智能体，可见距离被限制为 1.5 米，因此尽管人类可以从远处（渲染图像中）识别物体，智能体必须在更严格的部分可观测约束下探索环境。


We hypothesize that a higher episode length would lead to higher metrics, but the cutoff was chosen for computational budget considerations. As can be observed in figure 6 , the coverage and transportation rate metrics in particular seem to plateau around our chosen episode length $L$ ,so $L = {30}$ is appropriate.
我们假设更长的回合长度会带来更高的指标，但选择该截断是出于计算预算考虑。如图6 所示，覆盖率和运输率等指标在我们选择的回合长度 $L$ 附近似乎趋于平稳，因此 $L = {30}$ 是合适的。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2025/12/2025_12_03__23_16_58_7cf786.jpg"/>



Figure 6: Plots of the average transport rate (a),(d); coverage (b),(e); and success rate (c),(f) metrics vs. maximum horizon steps for the GPT-4V, LLaMAR algorithm on MAP-THOR. Shown for both the 2 agent (a),(b),(c); and 5 agent cases (d),(e),(f), along with an error region of one standard deviation.
图6：在 MAP-THOR 上 GPT-4V、LLaMAR 算法的平均运输率 (a),(d)、覆盖率 (b),(e) 和成功率 (c),(f) 相对于最大步数的曲线。分别显示 2 个智能体 (a),(b),(c) 和 5 个智能体的情况 (d),(e),(f)，并带有一标准差的误差区间。


### G.1 Failure Cases
### G.1 失败案例


We have observed the following categories of failures for the LM on the SAR environment.
我们在 SAR 环境中观察到语言模型的以下失败类别。


- Incorrect Causal Ordering: The LM sometimes fails to execute the steps in the proper causal ordering. For instance, we have observed the agent trying to drop off a lost person before picking them up, or properly getting supplies to stop the fire but not navigating to the fire location before using them.
- 因果顺序错误：语言模型有时未能按正确的因果顺序执行步骤。例如，我们观察到智能体在捡起走失人员之前试图放下他们，或在使用物资灭火之前没有先导航到火源位置。


- Sub-Optimal Action Sequence: Due to the time-constraints of the fire spread, it is not sufficient for the LM to do the sequence of tasks in order. They must also be done in a non-redundant, timely fashion. Otherwise, we have observed that the LM takes the right action sequence, but does not do it efficiently enough to stop the fire spread. For instance, if the agent prioritizes exploration to find the person, then the fire might spread until beyond a critical point where it cannot be extinguished within the horizon.
- 次优的动作序列：由于火势蔓延的时间限制，语言模型不仅要按顺序完成任务，还必须以非冗余且及时的方式执行。否则我们观察到语言模型虽采取了正确的动作序列，但执行效率不足以阻止火势蔓延。例如，如果智能体优先探索以寻找人员，火势可能在临界点之前蔓延至无法在回合内扑灭的程度。


- Catastrophic Failure Misunderstanding: Even after the LM has completed the task, it can misunderstand a later failure as a cause for continuing the task. For instance, we have observed that if the LM has successfully dropped off a person, but then fails to pick them up again (which is expected since they're already dropped), it will think the task is incomplete.
- 灾难性失败的误解：即使语言模型已经完成了任务，它也可能将之后的失败误认为需要继续任务的原因。例如，我们观察到如果模型已经成功送走了某人，但随后未能再次接回该人（这是预期的，因为他们已被送走），它仍会认为任务未完成。


## H Baselines
## H 基线


While there are a lot of impressive LLM-based multi-agent planners as mentioned in Table 1 they vary in the assumptions about access to information about the environment. We were not able to find the official codebase for the Safe Multi-Agent Planning with Conformal Prediction [20] and TwoStep [21]. We describe the prompts used for our model as well as every baseline. Note that we show the prompt for the 2-agent case,but it is easily modified to generalize to the $n$ -agent case. The italics and bolding added for emphasis.
虽然表1中提到有许多令人印象深刻的基于大型语言模型的多智能体规划器，但它们在对环境信息可访问性的假设上各不相同。我们未能找到“带有一致性预测的安全多智能体规划”[20]和 TwoStep [21] 的官方代码库。我们描述了用于我们模型以及每个基线的提示。注意我们展示的是 2 智能体情形的提示，但可轻松修改以推广到 $n$ 智能体情形。斜体和加粗为强调所加。


### H.1 LLaMAR
### H.1 LLaMAR


We describe the prompts used for each of the modules used in LLaMAR:
我们描述了用于 LLaMAR 中各模块的提示：


## Prompt for Planner Module in LLaMAR
## LLaMAR 中规划器模块的提示


You are an excellent planner who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
你是一位出色的规划者，负责帮助两台具身机器人 Alice 和 Bob 完成任务。两台机器人对环境的观察是部分可观测的，因此它们需要在环境中探索以完成任务。


You will get a description of the task robots are supposed to do. You will get an image of the environment from Alice's perspective and Bob's perspective as the observation input. To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
你将收到机器人应执行任务的描述。你还将收到来自 Alice 和 Bob 视角的环境图像作为观测输入。为了帮助你检测图像中的物体，你还会得到每个智能体在环境中能看到的物体列表。这里物体命名为 "<object_name>_<object_id>"。


So, along with the image inputs you will get the following information:
因此，除了图像输入，你还将得到以下信息：


## ###INPUT FORMAT ###
## ### 输入格式 ###


\{Task: description of the task the robots are supposed to do,
\{Task: 对机器人应执行任务的描述,


Alice's observation: list of objects the Alice is observing,
Alice's observation: Alice 观察到的物体列表,


Bob's observation: list of objects the Bob is observing,
Bob's observation: Bob 观察到的物体列表,


Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' open subtasks: 机器人应完成以结束任务的子任务列表。如果尚未创建计划，则为 None.


Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' completed subtasks: 机器人已完成的子任务列表。如果没有子任务已完成，则为 None.


Robots' combined memory: description of robots' combined memory\}
Robots' combined memory: 机器人合并记忆的描述\}


Reason over the robots' task, image inputs, observations, open subtasks, completed subtasks and memory, and then output the following:
根据机器人的任务、图像输入、观察结果、未完成的子任务、已完成的子任务和记忆进行推理，然后输出以下内容：


* Reason: The reason for why new subtasks need to be added.
* 推理：说明为何需要添加新的子任务。


* Subtasks: A list of open subtasks the robots are supposed to take to complete the task. Remember, as you get new information about the environment, you can modify this list. You can keep the same plan if you think it is still valid. Do not include the subtasks that have already been completed.
* 子任务：机器人应执行以完成任务的未完成子任务列表。请记住，随着对环境获得新信息，你可以修改此列表。如果你认为原计划仍然有效，可以保留原计划。不要包含已完成的子任务。


The "Plan" should be in a list format where the subtask are listed sequentially.
“计划”应为列表格式，子任务按顺序列出。


For example:
例如：


["locate the apple", "transport the apple to the fridge", "transport the book to the table"]
["定位苹果", "将苹果运到冰箱", "将书运到桌子上"]


["locate the cup", "go to cup", "clean cup"]
["定位杯子", "前往杯子处", "清洁杯子"]


When possible do not perform additional steps when one is sufficient (e.g. CleanObject is sufficient to clean an object, no other actions need to be done) Your output should be in the form of a python dictionary as shown below.
在可能的情况下，当一步足够时不要执行额外步骤（例如 CleanObject 足以清洁物体，无需其他动作）。你的输出应为如下所示的 python 字典形式。


## Example output:
## 示例输出：


\{"reason": "Since the subtask list is empty, the robots need to transport the apple to the fridge and transport the book to the table.",
\{"reason": "由于子任务列表为空，机器人需要将苹果运到冰箱并将书运到桌子上。",


"plan": ["transport the apple to the fridge", "transport the book to the table"]\}
"plan": ["将苹果运到冰箱", "将书运到桌子上"]\}


Ensure that the subtasks are not generic statements like "do the task". They should be specific to the task at hand.
确保子任务不是像“完成任务”这样泛泛的表述。它们应针对手头的任务具体说明。


Do not assign subtasks to any particular robot. Try not to modify the subtasks that already exist in the open subtasks list. Rather add new subtasks to the list.
不要将子任务分配给任何特定机器人。尽量不要修改开放子任务列表中已存在的子任务，而是向列表中添加新子任务。


* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
* 注意：除指定内容外不要输出任何额外信息


Let's work this out in a step by step way to be sure we have the right answer.
让我们一步步推理，确保得到正确答案。


## Prompt for Verifier Module in LLaMAR
## 提供给 LLaMAR 验证模块的提示


You are an excellent task verifier who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
你是一个出色的任务验证者，负责帮助两台具身机器人 Alice 和 Bob 完成任务。两台机器人对环境的观测是部分可观的，因此它们必须在环境中探索以完成任务。


You will get a description of the task robots are supposed to do. You will get an image of the environment from Alice's perspective and Bob's perspective as the observation input. To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
你会得到机器人应执行任务的描述。你将获得来自 Alice 和 Bob 视角的环境图像作为观察输入。为帮助你在图像中检测物体，你还会得到每个智能体能够在环境中看到的物体列表。这里的物体命名为 "<object_name>_<object_id>"。


So, along with the image inputs you will get the following information:
因此，除了图像输入，你还会得到以下信息：


## ###INPUT FORMAT ###
## ###INPUT FORMAT ###


\{Task: description of the task the robots are supposed to do,
\{Task: 对机器人应执行任务的描述,


Alice's observation: list of objects the Alice is observing,
Alice's observation: Alice 正在观察的物体列表,


Alice's state: description of Alice's state,
Alice's state: Alice 的状态描述,


Alice's previous action: the action Alice took in the previous step and if it was successful,
Alice's previous action: Alice 在上一步采取的动作及其是否成功,


Bob's observation: list of objects the Bob is observing,
Bob's observation: Bob 正在观察的物体列表,


Bob's state: description of Bob's state, Bob's previous action: the action Bob took in the previous step, Robots' open subtasks: list of open subtasks the robots in the previous step. If no plan has been already created, this will be None.
Bob's state: Bob 的状态描述, Bob's previous action: Bob 在上一步采取的动作, Robots' open subtasks: 上一步机器人的未完成子任务列表。如果尚未制定计划，则为 None.


Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' completed subtasks: 机器人已完成的子任务列表。如果没有完成任何子任务，则为 None.


Robots' combined memory: description of robots' combined memory\}
Robots' combined memory: 机器人合并记忆的描述\}


Reason over the robots' task, image inputs, observations, previous actions, open subtasks, completed subtasks and memory, and then output the following:
基于机器人任务、图像输入、观察、先前动作、未完成子任务、已完成子任务和记忆进行推理，然后输出以下内容：


* Reason: The reason for why you think a particular subtask should be moved from the open subtasks list to the completed subtasks list.
* Reason: 说明你认为应该将某个子任务从未完成子任务列表移到已完成子任务列表的原因。


* Completed Subtasks: The list of subtasks that have been completed by the robots. Note that you can add subtasks to this list only if they have been successfully completed and were in the open subtask list. If no subtasks have been completed at the current step, return an empty list.
* Completed Subtasks: 机器人已完成的子任务列表。注意，只有在子任务已成功完成且位于未完成子任务列表中时，你才能将其添加到此列表。如果在当前步骤没有完成任何子任务，则返回空列表。


The "Completed Subtasks" should be in a list format where the completed subtasks are listed.
“Completed Subtasks” 应为列表格式，列出已完成的子任务。


For example: ["locate the apple", "transport the apple to the fridge", "transport the book to the table"]
例如：["定位苹果", "将苹果运到冰箱里", "将书运到桌子上"]


Your output should be in the form of a python dictionary as shown below.
你的输出应为如下所示的 python 字典形式。


## Example output:
## 示例输出：


\{



"reason": "Alice placed the apple in the fridge in the previous step and was successful and Bob picked up the the book from the table. Hence Alice has completed the subtask of transporting the apple to the fridge, Bob has picked up the book, but Bob has still not completed the subtask of transporting the book to the table",
"reason": "在上一步中，Alice 已将苹果放入冰箱并且成功，Bob 从桌子上拿起了那本书。因此 Alice 已完成将苹果运到冰箱的子任务，Bob 已拿起书，但 Bob 仍未完成将书运到桌子的子任务",


"completed subtasks": ["picked up book from the table", "transport the apple to the fridge"]
"completed subtasks": ["从桌子上拿起书", "将苹果运到冰箱"]


\}



* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
* 注意：除指定内容外不要输出任何额外信息


When you output the completed subtasks, make sure to not forget to include the previous ones in addition to the new ones.
当你输出已完成的子任务时，确保在增加新的子任务的同时不要忘记包含之前已完成的那些。


Let's work this out in a step by step way to be sure we have the right answer.
让我们一步一步推理，以确保答案正确。


## Prompt for the Actor Module in LLaMAR
## LLaMAR 中 Actor 模块的提示


You are an excellent planner and robot controller who is tasked with helping 2 embodied robots named Alice, and Bob carry out a task. All 2 robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
你是出色的规划者和机器人控制器，负责帮助两台具身机器人 Alice 和 Bob 完成任务。两台机器人对环境都是部分可观测的，因此它们必须在环境中探索以完成任务。


They can perform the following actions:
它们可以执行以下动作：


["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"]
["导航到物体 <object_id>", "向 <rotation> 方向旋转", "拾起物体 <object_id>", "将物体放到 <receptacle_id> 上", "打开物体 <object_id>", "关闭物体 <object_id>", "切割物体 <object_id>", "将物体 <object_id> 打开", "将物体 <object_id> 关闭", "清洁物体 <object_id>", "按角度向上看 <angle>", "按角度向下看 <angle>", "向 <translation> 方向移动", "保持空闲", "完成"]


Here "Done" is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete.
这里“完成”用于当所有机器人完成主任务时。仅在你认为所有子任务均已完成时使用。


"stay idle" is used when you want the robot to stay idle for a one-time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.
“保持空闲”用于让机器人在一个时间步内保持空闲。这可用于等待另一台机器人完成它的子任务。仅在你认为有必要时使用。


Here <rotation> can be one of ["Right", "Left"].
这里 <rotation> 可以是 ["Right", "Left"] 中的一个。


Here <angle> is the angle in degrees and can only be one of $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ .
这里 <angle> 是角度，单位为度，只能是 $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ 中的一个。


Here <translation> can be one of ["Ahead", "Back", "Left", "Right"].
这里 <translation> 可以是 ["Ahead", "Back", "Left", "Right"].


So, along with the image inputs you will get the following information:
因此，除了图像输入外，你还会得到以下信息：


## ###INPUT FORMAT ###
## ###输入格式 ###


\{Task: description of the task the robots are supposed to do,
\{Task: 机器人应完成任务的描述,


Alice's observation: list of objects the Alice is observing,
Alice's observation: Alice 观察到的对象列表,


Alice's state: description of Alice's state,
Alice's state: Alice 的状态描述,


Alice's previous action: description of what Alice did in the previous time step and whether it was successful,
Alice's previous action: 对 Alice 在上一步所做动作及其是否成功的描述,


Alice's previous failures: if Alice's few previous actions failed,
Alice's previous failures: 如果 Alice 的前几次动作失败,


description of what failed,
描述失败的内容,


Bob's observation: list of objects the Bob is observing,
Bob's observation: Bob 观察到的对象列表,


Bob's state: description of Bob's state,
Bob's state: Bob 的状态描述,


Bob's previous action: description of what Bob did in the previous time step and whether it was successful,
Bob's previous action: 对 Bob 在上一步所做动作及其是否成功的描述,


Bob's previous failures: if Bob's few previous actions failed, description of what failed,
Bob's previous failures: 如果 Bob 的前几次动作失败, 描述失败的内容,


Robots' open subtasks: list of subtasks supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' open subtasks: 应执行以完成任务的子任务列表。如果尚未制定计划，则为 None.


Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' completed subtasks: 机器人已完成的子任务列表。如果没有完成任何子任务，则为 None.


Robots' subtask: description of the subtasks the robots were trying to complete in the previous step, Robots' combined memory: description of robot's combined memory\}
Robots' subtask: 上一步机器人试图完成的子任务描述, Robots' combined memory: 机器人联合记忆的描述\}


## ###OUTPUT FORMAT ###
## ###OUTPUT FORMAT ###


First of all you are supposed to reason over the image inputs, the robots' observations, previous actions, previous failures, previous memory, subtasks and the available actions the robots can perform, and think step by step and then output the following things:
首先你应当基于图像输入、机器人观测、先前动作、先前失败、先前记忆、子任务和机器人可执行的可用动作进行推理，逐步思考然后输出以下内容：


* Failure reason: If any robot's previous action failed, use the previous history, your current knowledge of the room (i.e. what things are where), and your understanding of causality to think and rationalize about why the previous action failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
* Failure reason: 如果任何机器人先前的动作失败，利用先前历史、你对房间的当前认知（即物品各在何处）以及对因果关系的理解来推理并说明先前动作失败的原因。输出失败原因以及如何在下一时刻修复。如果先前动作成功，输出 "None"。


Common failure reasons to lookout for include: one agent blocking another so must move out of the way, agent can't see an object or its destination and must explore (such as move, rotate, or look in a different direction) to find it, agent doing extraneous actions (such as drying objects when cleaning), etc. If the previous action was successful, output "None".
常见的失败原因包括：一个智能体挡住另一个，需要让路；智能体看不到某个物体或其目的地，需要探索（如移动、旋转或朝不同方向看）以找到它；智能体做了多余动作（如清洁时擦干物品）等。如果先前动作成功，输出 "None"。


* Memory: Whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. So, you should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.
* Memory: 将你认为应当记住以备将来使用的场景中重要信息作为记忆。请记住这些记忆将在后续步骤用于完成任务。因此，不应包含与任务无关的信息。如果你认为已有记忆中的信息在未来可能有用，也可以保留。


## (CONTINUED) Prompt for the Actor Module in LLaMAR
## (CONTINUED) Prompt for the Actor Module in LLaMAR


* Reason: The reasoning for what each robot is supposed to do next
* Reason: 每个机器人下一步应做事情的推理


* Subtask: The subtask each robot should currently try to solve, choose this from the list of open subtasks.
* Subtask: 每个机器人当前应尝试解决的子任务，从未完成子任务列表中选择。


* Action: The actions the robots are supposed to take just in the next step such that they make progress towards completing the task. Make sure that these suggested actions make these robots more efficient in completing the task as compared to only one agent solving the task.
* Action: 机器人在下一步应采取的动作，使其向完成任务取得进展。确保这些建议动作使多机器人协作比单个智能体解决任务更高效。


Your output should just be in the form of a python dictionary as shown below.
你的输出应仅为下述形式的 python 字典。


## Examples of output:
## Examples of output:


## Example 1:
## Example 1:


\{ "failure reason": "Bob failed to put the mug in the cabinet earlier because Alice was blocking it when she was putting the knife. To fix this, Alice should close the cabinet and move away, Charlie should move away to a different open area than Alice to avoid congestion, and Bob should wait until the next timestep until Alice can move aside.",
\{ "failure reason": "Bob failed to put the mug in the cabinet earlier because Alice was blocking it when she was putting the knife. To fix this, Alice should close the cabinet and move away, Charlie should move away to a different open area than Alice to avoid congestion, and Bob should wait until the next timestep until Alice can move aside.",


"memory": "Alice finished putting the knife in the cabinet when Alice was at co-ordinates (1, .5) and was facing north. Bob wanted to put the mug in the cabinet when Bob was at co-ordinates (1, 0.25) and was facing north.",
"memory": "Alice finished putting the knife in the cabinet when Alice was at co-ordinates (1, .5) and was facing north. Bob wanted to put the mug in the cabinet when Bob was at co-ordinates (1, 0.25) and was facing north.",


"reason": "Alice can close the cabinet door and then later back out in order help Bob with completing the task. Bob can be idle until the next timestep when Alice moves aside, by then Bob can navigate to the cabinet.",
"reason": "Alice can close the cabinet door and then later back out in order help Bob with completing the task. Bob can be idle until the next timestep when Alice moves aside, by then Bob can navigate to the cabinet.",


"subtask": "Alice is currently closing the cabinet door, Bob is currently waiting to get to navigate to the cabinet",
"subtask": "Alice is currently closing the cabinet door, Bob is currently waiting to get to navigate to the cabinet",


"Alice's action" : "close the Cabinet_1",
"Alice's action" : "关闭 Cabinet_1",


"Bob's action": "stay idle"
"Bob's action": "保持空闲"


\}



## Example 2: \{
## 示例 2: \{


"failure reason": "Bob failed to clean the cup earlier because Bob had not navigated to it, Bob assumed the cup to be in the sink which was erroneous. To fix this, Bob should navigate to the cup and in the next step clean cup.",
"failure reason": "Bob 之前未能清洗杯子，因为 Bob 未前往杯子所在位置，Bob 错误地以为杯子在水槽中。为了解决此问题，Bob 应该先前往杯子所在位置，下一步清洗杯子。",


"memory": "Alice finished navigating to the dish when Alice was at co-ordinates (-.5, .5) and was facing east. Bob was not able to clean the cup in the cabinet when Bob was at co-ordinates (1, .25) and was facing north.",
"memory": "Alice 在坐标 (-.5, .5) 并朝东时完成了前往盘子的导航。Bob 在坐标 (1, .25) 并朝北时未能在柜子里清洗杯子。",


"reason": "Alice can now clean the dish since Alice has navigated to it. Bob should navigate to the cup in order to be close enough to clean the cup.",
"reason": "Alice 现在可以清洗盘子，因为 Alice 已经前往盘子处。Bob 应该前往杯子以便足够靠近来清洗杯子。",


"subtask": "Alice is currently trying to clean the dish, Bob is currently trying to navigate to the cup", "Alice's action" : "clean the dish object",
"subtask": "Alice 目前尝试清洗盘子，Bob 目前尝试前往杯子", "Alice's action" : "清洗盘子对象",


"Bob's action": "navigate to the cup"
"Bob's action": "前往杯子"


Note that the output should just be a dictionary similar to the example outputs.
请注意输出应仅为类似示例输出的字典。


## ###Important Notes ###
## ###重要说明 ###


* The robots can hold only one object at a time.
* 机器人一次只能持有一个物体。


For example: If Alice is holding an apple, she cannot pick up another object until she puts the apple down.
例如：如果 Alice 正在拿着一个苹果，她在放下苹果之前不能再拿起另一个物体。


* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigate closer to the objects they want to interact with.
* 即使机器人能看到物体，如果距离太远也可能无法与之交互。因此你需要让机器人靠近它们想要交互的物体。


For example: An action like "pick up <object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to navigate closer to it before you can pick it up.
例如：像 "pick up <object_id>" 这样的动作只有在机器人能看到该物体并且足够靠近时才可行。因此你需要先靠近它才能拾取。


* In some scenarios, the agents might not see the objects that they want to interact with. In such cases, you will have to make the robot explore the environment to find the object. In such scenarios you can use actions to rotate in place or look up / down or navigate to explore the environment.
* 在某些情形下，代理可能看不到它们想要交互的物体。在这种情况下，你需要让机器人探索环境以找到该物体。在此类情形中，你可以使用原地旋转、上下看或导航来探索环境。


* If you open an object, please ensure that you close it before you navigate to a different place.
* 如果你打开了某个物体，请确保在导航到其他地方之前将其关闭。


* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* 打开抽屉、橱柜、冰箱等物体可能会阻挡机器人的通行路径。因此只有在确有必要时才打开物体。


* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
* 注意：除非按要求，否则不要输出任何额外内容


### H.2 Act
### H.2 行动


We describe the prompt used for the Act baseline:
我们描述用于 Act 基线的提示：


## Prompt for the Act Baseline
## Act 基线的提示


You are an excellent planner and robot controller who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
你是一名出色的规划者和机器人控制者，负责帮助两台具身机器人 Alice 和 Bob 完成任务。两台机器人对环境都是部分可观测的，因此必须在环境中探索才能完成任务。


They can perform the following actions:
它们可以执行以下动作：


["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"]
["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"]


Here "Done" is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete.
这里当所有机器人完成主要任务时使用 "Done"。仅在你认为所有子任务完成时使用。


"stay idle" is used when you want the robot to stay idle for a one-time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.
"stay idle" 用于让机器人在一个时间步内保持静止。这可用于等待另一台机器人完成其子任务。仅在你认为有必要时使用。


Here <rotation> can be one of ["Right", "Left"].
这里 <rotation> 可以是 ["Right", "Left"] 之一。


Here <angle> is the angle in degrees and can only be one of $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ .
这里 <angle> 为角度，单位为度，只能是 $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ 中的值。


Here <translation> can be one of ["Ahead", "Back", "Left", "Right"].
这里 <translation> 可以是 ["Ahead", "Back", "Left", "Right"] 之一。


You need to suggest the action that each robot should take at the current time step.
你需要建议每台机器人在当前时间步应采取的动作。


## ###Important Notes ###
## ###重要说明 ###


* The robots can hold only one object at a time.
* 机器人一次只能持有一个物体。


For example: If Alice is holding an apple, she cannot pick up another object until she puts the apple down.
例如：如果 Alice 正在拿着一个苹果，在她放下苹果之前不能再去拿其它物体。


* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigate closer to the objects they want to interact with.
* 即使机器人能看到物体，如果物体太远也可能无法与之交互。因此你需要让机器人靠近想要交互的物体。


For example: An action like "pick up <object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to navigate closer to it before you can pick it up.
例如：像“拾起 <object_id>”这样的动作只有在机器人能看到该物体且足够接近时才可行。因此你必须先靠近它才能将其拾起。


* In some scenarios, the agents might not see the objects that they want to interact with. In such cases, you will have to make the robot explore the environment to find the object. In such scenarios you can use actions to rotate in place or look up / down or navigate to explore the environment.
* 在某些场景中，代理可能看不到它们想要交互的物体。在这种情况下，你需要让机器人探索环境以找到该物体。此类场景可使用原地旋转、向上/向下看或导航等动作来探索环境。


* If you open an object, please ensure that you close it before you navigate to a different place.
* 如果你打开了一个物体，请在导航到其它地方之前确保将其关闭。


* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* 打开抽屉、橱柜、冰箱等物体可能会阻挡机器人的通路。因此仅在必要时打开物体。


## ###INPUT FORMAT ###
## ###输入格式 ###


* You will get a description of the task robots are supposed to do.
* 你将得到机器人应执行任务的描述。


* You will get an image of the environment at the current time step from Alice's perspective and Bob's perspective as the observation input. Here the objects are named as "<object_name>_<object_id>".
* 你将获得当前时间步 Alice 视角和 Bob 视角的环境图像作为观察输入。这里物体命名为“<object_name>_<object_id>”。


* You will get a trace of the steps taken by the robots and the actions they took at each time step and whether it was successful or not.
* 你将得到机器人所采取步骤的轨迹以及每个时间步它们执行的动作和是否成功。


## ###OUTPUT FORMAT ###
## ###输出格式 ###


In your output, do not have any extra text or content outside of the python dictionary as below. Do NOT put any text, spaces, or enter keys (i.e. "/n") outside of it.
在你的输出中，除下面的 Python 字典外不要有任何额外文本或内容。不要在其外放置任何文本、空格或换行（即“/n”）。


Your output should ONLY be in the form of a python dictionary, without any reasoning or extra text, as shown below:
你的输出应仅以 Python 字典的形式，不带任何推理或额外文本，如下所示：


\{"Alice": "action to be taken by Alice",
\{"Alice": "Alice 需要执行的动作",


"Bob": "action to be taken by Bob\}
"Bob": "Bob 需要执行的动作\}


For example: If you think Alice should pick up an apple and Bob should navigate to the fridge, you will have to give the output as:
例如：如果你认为 Alice 应该拾起一个苹果而 Bob 应该前往冰箱，你需要按以下格式给出输出：


\{"Alice": "pick up apple",
{"Alice": "捡起苹果",


"Bob": "navigate to fridge"\}
"Bob": "导航到冰箱"}


* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
* 注意：不要输出除指定内容之外的任何内容


### H.3 ReAct
### H.3 ReAct


We describe the prompt used for the ReAct baseline:
我们描述用于 ReAct 基线的提示：


## Prompt for ReAct Baseline
## ReAct 基线的提示


You are an excellent planner who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
你是一名出色的规划者，负责帮助两台具身机器人 Alice 和 Bob 完成任务。两台机器人对环境的观察是部分可观测的，因此它们必须在环境中探索以完成任务。


They can perform the following actions: ["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"
它们可以执行以下动作：["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"


Here "Done" is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete.
这里 "Done" 用于当所有机器人已完成主要任务时。只有在你认为所有子任务已完成时才使用它。


"stay idle" is used when you want the robot to stay idle for a one-time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.
"stay idle" 用于让机器人在一时间步保持空闲。可用于等待另一台机器人完成其子任务。仅在你认为有必要时使用。


Here <rotation> can be one of ["Right,"
这里 <rotation> 可以是以下之一 ["Right,"


Here <angle> is the angle in degrees and can only be one of $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ .
这里 <angle> 是以度为单位的角度，仅能是 $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ 中的值。


Here <translation> can be one of ["Ahead", "Back", "Left", "Right"].
这里 <translation> 可以是 ["Ahead", "Back", "Left", "Right"] 中的一个。


You need to suggest the action that each robot should take at the current time step.
你需要建议每台机器人在当前时间步应采取的动作。


## ###Important Notes ###
## ###重要说明 ###


* The robots can hold only one object at a time.
* 机器人一次只能拿一个物体。


For example: If Alice is holding an apple, she cannot pick up another object until she puts the apple down.
例如：如果爱丽丝手里拿着一个苹果，她在放下苹果之前无法拿起另一个物体。


* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigate closer to the objects they want to interact with.
* 即便机器人能看到物体，如果物体太远也可能无法与之交互。因此你需要让机器人靠近它们想要交互的物体。


For example: An action like "pick up <object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to navigate closer to it before you can pick it up.
例如：“pick up &lt;object_id&gt;” 这样的动作只有在机器人能看到该物体且足够接近时才可行。因此你必须先导航靠近它，然后才能将其拾起。


* In some scenarios, the agents might not see the objects that they want to interact with. In such cases, you will have to make the robot explore the environment to find the object. In such scenarios you can use actions to rotate in place or look up / down or navigate to explore the environment.
* 在某些场景中，代理可能看不到它们想要交互的物体。在这种情况下，你需要让机器人探索环境以找到该物体。你可以使用就地旋转、向上/下看或导航等动作来探索环境。


* If you open an object, please ensure that you close it before you navigate to a different place.
* 如果你打开了一个物体，请确保在导航到其他地方之前把它关上。


* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* 打开抽屉、橱柜、冰箱等物体可能会阻挡机器人的通路。因此仅在你认为必要时才打开这些物体。


## ###INPUT FORMAT ###
## ###INPUT FORMAT ###


* You will get a description of the task robots are supposed to do.
* 你将得到机器人应完成任务的描述。


* You will get an image of the environment at the current time step from Alice's perspective and Bob's perspective as the observation input. Here the objects are named as "<object_name>_<object_id>".
* 你将以观察输入的形式得到当前时间步环境的图像，分别来自 Alice 的视角和 Bob 的视角。这里物体命名为“<object_name>_<object_id>”。


* You will get a trace of the steps taken by the robots and the actions they took at each time step and whether it was successful or not.
* 你将得到机器人采取步骤的轨迹以及每个时间步他们采取的动作及其是否成功。


## ###OUTPUT FORMAT ###
## ###OUTPUT FORMAT ###


You are supposed to think and suggest the action each robot is supposed to take at the current time step. Before suggesting an action you need to think, which requires that you reason over the inputs and logically reflect on the task, observation and course of actions needed to complete the task.
你应当思考并建议每个机器人在当前时间步应采取的动作。在建议动作之前你需要思考，这要求你基于输入对任务、观测和完成任务所需动作的过程进行推理和逻辑反思。


Output Requirements: At each time step you must ONLY output a PYTHON DICTIONARY of the following two elements:
输出要求：在每个时间步你必须仅输出以下两个元素的 PYTHON DICTIONARY：


*First Element: Key = "Think" | Value:(Type: String): A logical reflection of the best action to be taken given the inputs: task at hand, observations, and trace.
*第一元素：键 = "Think" | 值:(类型：字符串)：对在给定输入（任务、观测和轨迹）下应采取的最佳动作的逻辑反思。


*Second Element: Key = "Action" | Value:(Type: Python Dictionary):
*第二元素：键 = "Action" | 值:(类型：Python Dictionary)：


The value should be in the form of a python dictionary as shown below.
其值应为如下所示形式的 python 字典。


\{"Alice": "action to be taken by Alice", "Bob": "action to be taken by Bob"\}
{"Alice": "Alice 应采取的动作", "Bob": "Bob 应采取的动作"}


For example: If you think Alice should pick up an apple and Bob should navigate to the fridge, you will have to give the output as: \{"Alice": "pick up apple", "Bob": "navigate to fridge"\} Here is an example output:
例如：如果你认为 Alice 应拿起一个苹果而 Bob 应前往冰箱，则输出应为：{"Alice": "pick up apple", "Bob": "navigate to fridge"} 下面是一个示例输出：


\{"Think": "To solve the task, I need to find and put the apple. The apple is likely to be on the countertop or table. Then find the fridge.", "Action": \{"Alice": "pick up apple", "Bob": "navigate to fridge"\}\} * NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
{"Think": "为完成任务，我需要找到并放好苹果。苹果很可能在台面或桌子上。然后去找冰箱。", "Action": {"Alice": "pick up apple", "Bob": "navigate to fridge"}} * 注意：除指定内容外不得输出任何额外信息


### H.4 Chain of Thought
### H.4 思路链


We describe the prompt used for the Chain-of-Thought baseline:
我们在此描述用于思路链基线的提示：


## Prompt for Chain of Thought Baseline
## 思路链基线的提示


You are an excellent planner who is tasked with helping 2 embodied robots named Alice and Bob carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
你是一名出色的规划者，负责帮助两台具身机器人 Alice 和 Bob 完成任务。两台机器人对环境的观察是部分可观测的，因此它们需要在环境中探索以完成任务。


They can perform the following actions: ["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"] Here "Done" is used when all the robots have completed the main task. Only use it when you think all the subtasks are complete. "stay idle" is used when you want the robot to stay idle for a one-time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary. Here <rotation> can be one of ["Right", "Left"].
它们可以执行以下动作：["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", "toggle object <object_id> on", "toggle object <object_id> off", "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", "move in <translation> direction", "stay idle", "Done"] 其中 "Done" 在所有机器人完成主要任务时使用，仅当你认为所有子任务已完成时才使用。"stay idle" 用于让机器人在一个时间步内保持空闲，可用于等待另一台机器人完成子任务，仅在你认为必要时使用。此处 <rotation> 可为 ["Right", "Left"] 之一。


Here <angle> is the angle in degrees and can only be one of $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ .
此处 <angle> 为角度（度），只能是 $\left\lbrack  {{30},{60},{90},{120},{150},{180}}\right\rbrack$ 中的值。


Here <translation> can be one of ["Ahead", "Back", "Left", "Right"].
此处 <translation> 可为 ["Ahead", "Back", "Left", "Right"] 之一。


You need to suggest the action that each robot should take at the current time step.
你需要建议每台机器人在当前时间步应采取的动作。


## ###Important Notes ###
## ###重要说明 ###


* The robots can hold only one object at a time. For example: If Alice is holding an apple, she cannot pick up another object until she puts the apple down.
* 机器人一次只能拿一个物体。例如：如果 Alice 正拿着一个苹果，她在放下苹果之前不能再拿起其他物体。


* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot navigate closer to the objects they want to interact with. For example: An action like "pick up $<$ object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to navigate closer to it before you can pick it up.
* 即便机器人能看到物体，如果物体太远也可能无法与之交互。因此你需要让机器人朝物体靠近再进行交互。例如：像 "pick up $<$ object_id>" 这样的动作只有在机器人能看到且足够靠近物体时才可行，所以你必须先靠近它才能拾取。


* In some scenarios, the agents might not see the objects that they want to interact with. In such cases, you will have to make the robot explore the environment to find the object. In such scenarios you can use actions to rotate in place or look up / down or navigate to explore the environment.
* 在某些情形下，智能体可能看不到想要交互的物体。这种情况下，你需要让机器人探索环境以找到物体。可使用就地旋转、向上/向下查看或导航等动作来探索环境。


* If you open an object, please ensure that you close it before you navigate to a different place.
* 如果你打开了某个物体，请确保在导航到其他地点之前将其关闭。


* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* 打开像抽屉、橱柜、冰箱之类的物品可能会挡住机器人的路径。所以只有在你认为有必要时才打开这些物品。


## ###INPUT FORMAT ###
## ###输入 格式 ###


* You will get a description of the task robots are supposed to do.
* 你会得到机器人应完成任务的描述。


* You will get an image of the environment at the current time step from Alice's perspective and Bob's perspective as the observation input. Here the objects are named as "<object_name>_<object_id>".
* 你会收到来自 Alice 视角和 Bob 视角的当前时间步环境图像作为观测输入。这里物体被命名为 "<object_name>_<object_id>"。


* You will get a trace of the steps taken by the robots and the actions they took at each time step and whether it was successful or not.
* 你会得到机器人所采取步骤的轨迹以及它们在每个时间步采取的动作和是否成功。


## ###OUTPUT FORMAT ###
## ###输出 格式 ###


You are supposed to FIRST reason through the situation logically and step by step, then suggest the action each robot is supposed to take at the current time step.
你应当首先逻辑且逐步地推理情境，然后建议每个机器人在当前时间步应采取的动作。


In your output, do not have any extra text or content outside of the python dictionary as below.
在你的输出中，不要在下面的 python 字典之外包含任何额外文本或内容。


Your output should ONLY be in the form of a python dictionary as shown below:
你的输出应仅采用如下 python 字典的形式：


\{"reason": "Reasoning for action plan....", "Alice": "action to be taken by Alice", "Bob": "action to be taken by Bob"\}
{"reason": "Reasoning for action plan....", "Alice": "action to be taken by Alice", "Bob": "action to be taken by Bob"}


Put all of your reasoning inside of the "reason" key of the dictionary. Do NOT put any text, spaces, or enter keys (i.e. "/n") outside of it.
将所有推理放在字典的 "reason" 键中。不要在其外放置任何文本、空格或回车（即 "\n"）。


For example: If you think Alice should pick up an apple and Bob should navigate to the fridge, you will have to give the output as:
例如：如果你认为 Alice 应该拿起一个苹果而 Bob 应该前往冰箱，你必须如下给出输出：


\{"reason": "since the subtask list is empty, the robots need to transport the apple to the fridge", "Alice": "pick up apple", "Bob": "navigate to fridge"\}
{"reason": "since the subtask list is empty, the robots need to transport the apple to the fridge", "Alice": "pick up apple", "Bob": "navigate to fridge"}


Let's think step by step, but make sure to put all of your reasoning inside of the "reason" key of the dictionary!
让我们逐步思考，但务必将所有推理放在字典的 "reason" 键中！


* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
* 注意：不要输出除了已指定内容之外的任何其他内容


### H.5 SmartLLM
### H.5 SmartLLM


We adapt the prompt from the official codebase of SmartLLM (master branch; commit #be42930050f7d4d8f2fad027aff14a699c3300aa) as given here: https://github.com/SMARTlab-Purdue/SMART-LLM/blob/master/scripts/run_llm.py with a slight modification. Instead of letting the agents access all the objects in the environment through the simulator metadata, we just give the list of objects visible from the agents' point-of-view.
我们改编了 SmartLLM 官方代码库（master 分支；提交 #be42930050f7d4d8f2fad027aff14a699c3300aa）中的提示，如此处所示：https://github.com/SMARTlab-Purdue/SMART-LLM/blob/master/scripts/run_llm.py，并作了轻微修改。我们没有让代理通过模拟器元数据访问环境中所有对象，而只是提供从代理视角可见的对象列表。


### H.6 CoELA
### H.6 CoELA


We adapt the prompt from the official codebase of CoELA (master branch: commit #3d34de46dc77f9aaabe438cd2b92ea6c5c04973a) as given here: https://github.com/UMass Foundation-Model/Co-LLM-Agents/tree/master/tdw_mat/LLM. We modify some aspects of the prompt as described: Instead of relying on the simulator/pre-defined conditional logic for generating the list of available action options, we give a list of all possible actions based on the observation. This includes the option to send the communication message, all navigation actions, and all combinations of valid actions with the interactable objects in the current observation.
我们改编了 CoELA 官方代码库（master 分支：提交 #3d34de46dc77f9aaabe438cd2b92ea6c5c04973a）中的提示，如此处所示：https://github.com/UMass Foundation-Model/Co-LLM-Agents/tree/master/tdw_mat/LLM。我们按下述方式修改了提示的某些方面：不依赖模拟器/预定义条件逻辑来生成可用动作选项列表，而是根据观测提供所有可能动作的列表。该列表包括发送通信信息的选项、所有导航动作，以及与当前观测中可交互对象组合的所有有效动作。


## I Open Source VLMs
## I 开源 VLMs


We list the source of the weights we used for the open-source VLMs:
我们列出用于开源 VLMs 的权重来源：


- Idefics 2 [68, 69]: We use the 8B base model fine-tuned on a mixture of supervised and instruction datasets (text-only and multimodal datasets) from HuggingFace. The weights were downloaded from https://huggingface.co/HuggingFaceM4/idefics2-8b with the commit #2c031da2dc71f3ac989f9efa9b8ff476df3842c0. We chose Idefics because it is able to take multiple images as input similar to GPT-4V and reason on them.
- Idefics 2 [68, 69]：我们使用在监督和指令数据集（仅文本和多模态数据集）混合上微调的 8B 基模型。权重从 https://huggingface.co/HuggingFaceM4/idefics2-8b 下载，提交号为 #2c031da2dc71f3ac989f9efa9b8ff476df3842c0。我们选择 Idefics 因其能够像 GPT-4V 一样接受多张图像作为输入并在其上推理。


- LLaVA [70]: We use the 7B model t trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal instruction-following data. The weights were downloaded from https://huggingface.co/llava-hf/lava-1.5-7b-hf with the commit # 05ae2434cbb430be33edcba0c5203e7023f785b7.
- LLaVA [70]：我们使用 7B 模型，该模型通过在 GPT 生成的多模态指令跟随数据上微调 LLaMA/Vicuna 训练得到。权重从 https://huggingface.co/llava-hf/lava-1.5-7b-hf 下载，提交号为 #05ae2434cbb430be33edcba0c5203e7023f785b7。


- CogVLM [72]: We use the 18B model. The weights were downloaded from https://huggingface.co/THUDM/cogagent-chat-hf with the commit # d519da3b191401234f4bd86ce1c287c61bc276a3.
- CogVLM [72]：我们使用 18B 模型。权重从 https://huggingface.co/THUDM/cogagent-chat-hf 下载，提交号为 #d519da3b191401234f4bd86ce1c287c61bc276a3。


## J SentenceBERT fine-tuning
## J SentenceBERT 微调


We finetuned a pre-trained BERT model to function as a semantic mapper between free-form natural language output and the robot's admissible actions in the environment. The pre-trained weights were obtained from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 The model was trained on a dataset consisting of 2800 free-form input, valid action output pairs. It ran on one (1) Apple M1 core for a wall clock time of 5 minutes. Table 8 shows the hyper-parameters used for the pre-training of the BERT model.
我们对一个预训练的 BERT 模型进行了微调，使其充当自由形式自然语言输出与机器人在环境中可接受动作之间的语义映射器。预训练权重来自 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2。该模型在包含 2800 对自由形式输入与有效动作输出的数据集上训练。训练在一（1）核 Apple M1 上运行，总耗时 5 分钟。表 8 显示了用于 BERT 模型预训练的超参数。


<table><tr><td>Epochs</td><td>10</td></tr><tr><td>Max gradient norm</td><td>1</td></tr><tr><td>Learning rate</td><td>$2 \times  {10}^{-5}$</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Encoding dimension</td><td>384</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Scheduler</td><td>Warm-up linear</td></tr><tr><td>Warm-up steps</td><td>45</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Loss scale</td><td>20</td></tr><tr><td>Loss type</td><td>Multiple negatives ranking loss</td></tr><tr><td>Similarity function</td><td>Cosine similarity</td></tr></table>
<table><tbody><tr><td>训练轮数</td><td>10</td></tr><tr><td>最大梯度范数</td><td>1</td></tr><tr><td>学习率</td><td>$2 \times  {10}^{-5}$</td></tr><tr><td>批次大小</td><td>64</td></tr><tr><td>编码维度</td><td>384</td></tr><tr><td>优化器</td><td>AdamW</td></tr><tr><td>调度器</td><td>线性预热</td></tr><tr><td>预热步数</td><td>45</td></tr><tr><td>权重衰减</td><td>0.01</td></tr><tr><td>损失缩放</td><td>20</td></tr><tr><td>损失类型</td><td>多重负样本排序损失</td></tr><tr><td>相似度函数</td><td>余弦相似度</td></tr></tbody></table>


Table 8: Hyper-parameters for the model fine-tuning including the loss.
表8：包括损失在内的模型微调超参数。


## NeurIPS Paper Checklist
## NeurIPS 论文检查清单


## 1. Claims
## 1. 论断


Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
问题：摘要和引言中的主要论断是否准确反映了论文的贡献与范围？


## Answer: [Yes]
## 答复：[是]


Justification: Our claims in the abstract are reflected in the experimental results shown in the paper.
理由：我们在摘要中的论断在论文展示的实验结果中得到了体现。


Guidelines:
指导：


- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- 答案为 NA 表示摘要和引言未包含论文中提出的论断。


- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- 摘要和/或引言应清楚陈述所提出的论断，包括论文的贡献以及重要假设和局限。对此问题回答“No”或“NA”可能会被审稿人不看好。


- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- 所提出的论断应与理论和实验结果一致，并反映这些结果在其他情形下可推广的程度。


- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
- 将抱负性目标作为动机是可以的，只要明确这些目标并非由论文达到。


## 2. Limitations
## 2. 局限


Question: Does the paper discuss the limitations of the work performed by the authors?
问题：论文是否讨论了作者所做工作的局限性？


## Answer: [Yes]
## 答复：[是]


Justification: we mention the limitations of our paper in a separate section in the paper.
理由：我们在论文的独立章节中提到了论文的局限性。


Guidelines:
指导：


- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- 答案 NA 表示论文无局限，而答案 No 表示论文存在局限，但论文中未讨论这些局限。


- The authors are encouraged to create a separate "Limitations" section in their paper.
- 建议作者在论文中单独设立“局限性”一节。


- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- 论文应指出任何强假设以及结果对这些假设被违反时的稳健性（例如独立性假设、无噪设置、模型充分设定、渐近近似仅在局部成立等）。作者应反思这些假设在实践中可能如何被违反及其影响。


- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- 作者应反思所作结论的适用范围，例如方法是否仅在少量数据集或少次运行上测试过。一般而言，经验结果往往依赖隐含假设，应予以阐明。


- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- 作者应反思影响方法性能的因素。例如，当图像分辨率低或光线不足时，人脸识别算法可能表现不佳；或语音转文字系统可能无法可靠为在线讲座提供字幕，因为它无法处理专业术语。


- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- 作者应讨论所提算法的计算效率及其随数据集规模的扩展性。


- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- 如适用，作者应讨论其方法在解决隐私和公平问题方面可能存在的局限性。


- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
- 虽然作者可能担心诚实陈述局限会被审稿人作为拒稿理由，但更糟的结果是审稿人发现论文中未承认的局限。作者应以最佳判断行事，并认识到促进透明性的个人行为有助于建立维护社区诚信的规范。审稿人将被明确指示不要因如实陈述局限而予以惩罚。


## 3. Theory Assumptions and Proofs
## 3. 理论假设与证明


Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
问题：对于每一项理论结果，论文是否提供了完整的假设集合和完整（且正确）的证明？


Answer: [NA]
答复： [NA]


Justification: We do not include any theoretical results in the paper and hence we do not report the theoretical assumptions. Although we note the assumptions in the AI2Thor environment in which we run all our experiments.
理由：我们在论文中未包含任何理论结果，因此未报告理论假设。但我们确实在运行所有实验的 AI2Thor 环境中注明了假设。


Guidelines:
指导：


- The answer NA means that the paper does not include theoretical results.
- 答案 NA 表示论文不包含理论结果。


- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- 论文中的所有定理、公式和证明应编号并相互引用。


- All assumptions should be clearly stated or referenced in the statement of any theorems.
- 在任何定理的陈述中，所有假设应明确说明或引用。


- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- 证明可以出现在正文或补充材料中，但若放在补充材料中，鼓励作者提供简短的证明草图以提供直观理解。


- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- 反之，论文正文中任何非正式的证明应以附录或补充材料中的正式证明予以补充。


- Theorems and Lemmas that the proof relies upon should be properly referenced.
- 证明所依赖的定理和引理应被适当引用。


## 4. Experimental Result Reproducibility
## 4. 实验结果可复现性


Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
问题：论文是否充分披露了复现实验中主要结果所需的全部信息，以至于这些信息会影响论文的主要主张和/或结论（无论是否提供代码和数据）？


## Answer: [Yes]
## 答案：[Yes]


Justification: We provide as much detail as we can for the reproducibility of the experiments. Although there will be some variations to the underlying LLMs/VLMs we note the specific LLM/VLM model we used for our experiments in the Appendix.
理由：我们为实验的可复现性提供了尽可能多的细节。尽管底层的 LLMs/VLMs 可能存在一些差异，我们在附录中注明了用于实验的具体 LLM/VLM 模型。


Guidelines:
指南：


- The answer NA means that the paper does not include experiments.
- 答案 NA 表示论文不包含实验。


- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- 若论文包含实验，对此问题回答 No 将不会被审稿人认可：使论文可复现很重要，无论是否提供代码和数据。


- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- 如果贡献是数据集和/或模型，作者应描述为使其结果可复现或可验证而采取的步骤。


- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- 根据贡献的不同，可复现性可通过多种方式实现。例如，如果贡献是新架构，完整描述该架构可能就足够；或者如果贡献是特定模型和实证评估，可能需要让他人能够使用相同数据集复现该模型，或提供模型访问。通常，发布代码和数据是实现这一点的常用方法，但也可通过详细的复现说明、提供托管模型的访问（例如大型语言模型）、发布模型检查点或其他适合所做研究的方式实现可复现性。


- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- 虽然 NeurIPS 不要求发布代码，但会议要求所有投稿为可复现性提供某种合理途径，这可能取决于贡献的性质。例如


(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
(a) 若贡献主要是新算法，论文应明确说明如何复现该算法。


(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
(b) 若贡献主要是新模型架构，论文应清晰且完整地描述该架构。


(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
(c) 若贡献是新模型（例如大型语言模型），则应提供访问该模型以复现结果的方式，或提供复现该模型的方法（例如使用开源数据集或构建数据集的说明）。


(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
(d) 我们认识到在某些情况下可复现性可能很棘手，在这种情况下欢迎作者描述他们为实现可复现性所采取的具体方式。对于闭源模型，模型的访问可能以某种方式受到限制（例如，仅限注册用户），但其他研究者应当有某种途径来复现或验证结果。


## 5. Open access to data and code
## 5. 数据与代码的开放获取


Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
问题：论文是否提供对数据和代码的开放访问，并附有足够的说明以忠实复现主要实验结果，如补充材料所述？


## Answer: [Yes]
## 回答：[是]


Justification: We provide a link to the GitHub repository in the abstract.
理由：我们在摘要中提供了指向 GitHub 仓库的链接。


Guidelines:
指南：


- The answer NA means that paper does not include experiments requiring code.
- 答案 NA 意味着论文不包含需要代码的实验。


- Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.
- 更多细节请参见 NeurIPS 的代码与数据提交指南 (https://nips.cc/ public/guides/CodeSubmissionPolicy)。


- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- 虽然我们鼓励发布代码和数据，但我们理解这可能不可行，因此 “否” 是可接受的答案。仅因未包含代码而拒稿是不可取的，除非代码对贡献至关重要（例如用于新的开源基准）。


- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- 指示应包含为复现结果所需的精确命令和运行环境。更多细节请参见 NeurIPS 的代码与数据提交指南 (https: //nips.cc/public/guides/CodeSubmissionPolicy)。


- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- 作者应提供有关数据访问和准备的说明，包括如何访问原始数据、预处理数据、中间数据和生成的数据等。


- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- 作者应提供脚本以复现新方法和基线的所有实验结果。如果只有部分实验可复现，应说明哪些实验未包含在脚本中及其原因。


- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- 在提交时，为保持匿名性，作者应发布匿名版本（如适用）。


- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
- 建议在补充材料（附在论文后）中提供尽可能多的信息，但允许包含指向数据和代码的 URL。


## 6. Experimental Setting/Details
## 6. 实验设置/细节


Question: Does the paper specify all the training and test details (e.g., data splits, hyper-parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
问题：论文是否说明了所有训练和测试细节（例如数据划分、超参数、如何选择、优化器类型等）以理解结果？


## Answer: [Yes]
## 答案: [是]


Justification: We specify as many details of our experiments as we can in the appendix.
理由: 我们在附录中尽可能详细地说明了实验的各项细节。


Guidelines:
指南:


- The answer NA means that the paper does not include experiments.
- 答案为 NA 表示论文不包含实验。


- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- 实验设置应在论文正文中以足够详尽的程度呈现，以便理解结果并使其有意义。


- The full details can be provided either with the code, in appendix, or as supplemental material.
- 完整细节可随代码提供，或在附录中，或作为补充材料。


## 7. Experiment Statistical Significance
## 7. 实验统计显著性


Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
问题: 论文是否适当地并正确定义并报告了误差条或其他关于实验统计显著性的适当信息？


## Answer: [Yes]
## 答案: [是]


Justification: Yes, we report means and the 95% confidence intervals.
理由: 是的，我们报告了均值和 95% 置信区间。


Guidelines:
指南:


- The answer NA means that the paper does not include experiments.
- 答案为 NA 表示论文不包含实验。


- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- 如果结果附有误差条、置信区间或统计显著性检验，至少针对支持论文主要结论的实验，作者应答“是”。


- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- 应明确说明误差条所反映的变异因素（例如，训练/测试划分、初始化、某些参数的随机抽取，或在给定实验条件下的整体运行）。


- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- 应解释计算误差条的方法（解析公式、调用库函数、自助法等）。


- The assumptions made should be given (e.g., Normally distributed errors).
- 应给出所作的假设（例如，误差服从正态分布）。


- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- 应明示误差线代表的是标准差还是均值的标准误。


- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a ${96}\% \mathrm{{CI}}$ ,if the hypothesis of Normality of errors is not verified.
- 可以报告 1-σ 误差线，但应声明如此。若误差服从正态分布的假设未被验证，作者最好报告 2-σ 误差线，而不是仅说明他们有一个 ${96}\% \mathrm{{CI}}$。


- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- 对于不对称分布，作者应注意不要在表格或图中展示对称误差线，从而产生超出取值范围（例如负误差率）的结果。


- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
- 若在表格或图中报告误差线，作者应在正文中解释其计算方法并在文中引用相应的图表。


## 8. Experiments Compute Resources
## 8. 实验计算资源


Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
问题：对于每个实验，论文是否提供了足够的关于重现实验所需计算资源（计算节点类型、内存、执行时间）的信息？


## Answer: [Yes]
## 回答：[是]


Justification: Yes, we denote all these details in the appendix.
理由：是的，我们在附录中注明了所有这些细节。


Guidelines:
指导原则：


- The answer NA means that the paper does not include experiments.
- 回答 NA 意味着论文不包含实验。


- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- 论文应说明计算节点类型（CPU 或 GPU）、内部集群或云提供商，并包括相关内存和存储信息。


- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- 论文应提供每次单独实验运行所需的计算量，并估算总计算量。


- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
- 论文应披露整个研究项目是否比论文中报告的实验需要更多计算资源（例如未入文的初步或失败实验）。


## 9. Code Of Ethics
## 9. 伦理守则


Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
问题：论文中开展的研究在各方面是否遵守 NeurIPS 伦理守则 https://neurips.cc/public/EthicsGuidelines？


## Answer: [Yes]
## 回答：[是]


Justification: We have read the code of ethics on the website.
理由：我们已在网站上阅读了伦理守则。


Guidelines:
指南：


- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- 回答 NA 表示作者未审阅 NeurIPS 伦理守则。


- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- 如果作者回答 No，应解释要求偏离伦理守则的特殊情况。


- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
- 作者应确保保持匿名性（例如，若因其司法辖区的法律或法规存在特殊考虑）。


## 10. Broader Impacts
## 10. 更广泛影响


Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
问题：论文是否讨论了所做工作的潜在正面社会影响和负面社会影响？


## Answer: [NA]
## 答案： [NA]


Justification: In our opinion, our work does not have any significant societal impact other than being able to progress towards enabling embodied multi-agent robots for household planning.
理由：我们认为，除能推动实现用于家庭规划的具身多智能体机器人外，我们的工作没有其他重大社会影响。


Guidelines:
指南：


- The answer NA means that there is no societal impact of the work performed.
- 回答 NA 表示该工作没有社会影响。


- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- 如果作者回答 NA 或 No，应解释为何其工作没有社会影响或为何论文未讨论社会影响。


- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- 负面社会影响的例子包括潜在的恶意或意外用途（例如，散布错误信息、生成虚假资料、监控）、公平性考量（例如，部署可能不公平影响特定群体的技术）、隐私和安全考虑。


- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- 会议预计许多论文为基础研究，与具体应用或部署无关。然而若存在直接通向任何负面应用的路径，作者应指出。例如，指出生成模型质量的提升可能被用于制作误导性深度伪造是合理的。另一方面，不需要指出通用的神经网络优化算法可能使人们更快训练出生成深度伪造的模型。


- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- 作者应考虑技术在按预期正常使用时可能产生的危害、按预期使用但产生错误结果时的危害，以及技术被（有意或无意）滥用时的后果。


- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
- 若存在负面社会影响，作者也可讨论可能的缓解策略（例如，模型分级发布、在攻击之外提供防御、监控滥用的机制、监测系统随时间从反馈中学习的机制、提高 ML 的效率和可及性）。


## 11. Safeguards
## 11. 保障措施


Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
问题：论文是否描述了为负责发布具有高滥用风险的数据或模型（例如，预训练语言模型、图像生成器或抓取的数据集）而采取的保障措施？


Answer: [NA]
回答：[NA]


Justification: No high risks in the paper.
理由：论文中不存在高风险。


Guidelines:
指南：


- The answer NA means that the paper poses no such risks.
- 回答 NA 表示论文不构成此类风险。


- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- 对于具有高滥用或双重用途风险的发布模型，应采用必要的保障措施以便受控使用，例如要求用户遵守使用指南或限制访问模型，或实施安全过滤。


- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- 从互联网抓取的数据集可能带来安全风险。作者应说明如何避免发布不安全的图像。


- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
- 我们认识到提供有效保障具有挑战性，且许多论文不需要，但我们鼓励作者考虑并尽最大努力采取良信措施。


## 12. Licenses for existing assets
## 12. 现有资源的许可


Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
问题：论文中使用的资源（例如代码、数据、模型）的创建者或原始所有者是否得到了适当致谢，许可证与使用条款是否被明确提及并得到妥善遵守？


## Answer: [Yes]
## 回答：[Yes]


Justification: We have cited all the authors of the works we base our work on especially the AI2Thor simulator. We have also provided links to the weights we used for the open-source VLMs in the appendix with the commit # number on HuggingFace.
理由：我们已引用所有作为本工作基础的作者，特别是 AI2Thor 模拟器。我们还在附录中提供了用于开源 VLM 的权重链接，并在 HuggingFace 上标注了 commit # 编号。


Guidelines:
指南：


- The answer NA means that the paper does not use existing assets.
- 回答 NA 表示论文未使用现有资源。


- The authors should cite the original paper that produced the code package or dataset.
- 作者应引用产生代码包或数据集的原始论文。


- The authors should state which version of the asset is used and, if possible, include a URL.
- 作者应说明使用了资产的哪个版本，并在可能的情况下提供 URL。


- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- 应为每个资产标明许可证名称（例如 CC-BY 4.0）。


- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- 对于从特定来源（例如网站）抓取的数据，应提供该来源的版权和服务条款。


- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- 若发布资产，应在包内提供许可证、版权信息和使用条款。对于常见数据集，paperswithcode.com/datasets 对部分数据集整理了许可信息，其许可指南可帮助确定数据集的许可证。


- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- 对于重新打包的现有数据集，应提供原始许可证以及派生资产的许可证（若有变更）。


- If this information is not available online, the authors are encouraged to reach out to the asset's creators.
- 若这些信息在网上不可获得，建议作者联系资产的创建者。


### 13.New Assets
### 13.New Assets


Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
问题：论文中引入的新资产是否有充分文档，并且文档是否随资产一起提供？


## Answer: [Yes]
## 答： [Yes]


Justification: We provide documentation for the initialization of the multi-agent setting in the AI2Thor simulator in the supplementary material.
理由：我们在补充材料中提供了在 AI2Thor 模拟器中初始化多智能体设置的文档。


Guidelines:
指南：


- The answer NA means that the paper does not release new assets.
- 答 NA 表示论文未发布新资产。


- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- 研究者应通过结构化模板在提交中说明数据集/代码/模型的细节，包括训练、许可证、限制等信息。


- The paper should discuss whether and how consent was obtained from people whose asset is used.
- 论文应讨论是否以及如何征得被使用资产相关人员的同意。


- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
- 在提交时，记得对你的资产进行匿名（如适用）。你可以创建一个匿名 URL 或包含一个匿名的压缩文件。


## 14. Crowdsourcing and Research with Human Subjects
## 14. Crowdsourcing and Research with Human Subjects


Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
问题：对于众包实验和涉及人的研究，论文是否包含提供给参与者的完整说明文字和截图（如适用），以及有关报酬（如有）的详细信息？


## Answer: [NA]
## 答复： [NA]


Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:
理由：论文不涉及众包或以人为对象的研究。指南：


- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- 回答 NA 意味着论文不涉及众包或以人为对象的研究。


- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- 在补充材料中包含此信息是可以的，但如果论文的主要贡献涉及人类受试者，则应在主文中尽可能详尽地包含这些细节。


- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
- 根据 NeurIPS 伦理守则，参与数据收集、整理或其他劳务的工作人员应至少按数据收集者所在国家的最低工资支付报酬。


## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects
## 15. 针对涉及人类受试者研究的机构审查委员会（IRB）批准或同等审批


Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
问题：论文是否描述了研究参与者可能遭受的风险、是否已向受试者披露这些风险，以及是否已获得机构审查委员会（IRB）批准（或基于贵国或机构要求的同等审批/审查）？


## Answer: [NA]
## 答复： [NA]


Justification: The paper does not involve crowdsourcing or research with human subjects.
理由：论文不涉及众包或以人为对象的研究。


Guidelines:
指南：


- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- 回答 NA 意味着论文不涉及众包或以人为对象的研究。


- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- 根据研究开展所在国家的规定，任何涉及人的研究可能需要 IRB 批准（或同等审批）。如果您已获得 IRB 批准，应在论文中明确说明。


- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- 我们认识到此类程序在不同机构和地区可能有显著差异，期望作者遵守 NeurIPS 伦理守则及其所属机构的指南。


- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
- 在初稿提交时，不要包含会破坏匿名性的任何信息（如适用），例如进行审查的机构名称。