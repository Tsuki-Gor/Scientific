# Language-Grounded Dynamic Scene Graphs for Interactive Object Search with Mobile Manipulation
# 面向移动操作的交互式目标搜索：语言引导的动态场景图


Daniel Honerkamp ${}^{1 * }$ , Martin Büchner ${}^{1 * }$ , Fabien Despinoy ${}^{2}$ , Tim Welschehold ${}^{1}$ , Abhinav Valada ${}^{1}$
Daniel Honerkamp ${}^{1 * }$ ，Martin Büchner ${}^{1 * }$ ，Fabien Despinoy ${}^{2}$ ，Tim Welschehold ${}^{1}$ ，Abhinav Valada ${}^{1}$


Abstract-To fully leverage the capabilities of mobile manipulation robots, it is imperative that they are able to autonomously execute long-horizon tasks in large unexplored environments. While large language models (LLMs) have shown emergent reasoning skills on arbitrary tasks, existing work primarily concentrates on explored environments, typically focusing on either navigation or manipulation tasks in isolation. In this work, we propose MoMa-LLM, a novel approach that grounds language models within structured representations derived from open-vocabulary scene graphs, dynamically updated as the environment is explored. We tightly interleave these representations with an object-centric action space. Given object detections, the resulting approach is zero-shot, open-vocabulary, and readily extendable to a spectrum of mobile manipulation and household robotic tasks. We demonstrate the effectiveness of MoMa-LLM in a novel semantic interactive search task in large realistic indoor environments. In extensive experiments in both simulation and the real world, we show substantially improved search efficiency compared to conventional baselines and state-of-the-art approaches, as well as its applicability to more abstract tasks. We make the code publicly available at http://moma-llm.cs.uni-freiburg.de.
摘要-要充分发挥移动操作机器人能力，关键在于它们能够在大型、尚未探索的环境中自主执行长时域任务。尽管大型语言模型（LLMs）已在任意任务上展现出涌现式推理能力，但现有工作主要集中在已探索环境中，通常仅分别关注导航或操作任务。本研究提出 MoMa-LLM，这是一种新方法：将语言模型嵌入到从开放词汇场景图得到的结构化表示中，并在探索环境过程中动态更新。我们将这些表示与以对象为中心的动作空间紧密耦合。基于目标检测，该方法实现了零样本、开放词汇，并可轻松扩展到多种移动操作与家用机器人任务。我们在大型真实室内环境中的一项新颖语义交互式搜索任务中验证了 MoMa-LLM 的有效性。在仿真与真实世界的大规模实验中，我们发现其在搜索效率上相较于传统基线和最先进方法均有显著提升，并且适用于更抽象的任务。代码已公开发布：http://moma-llm.cs.uni-freiburg.de。


Index Terms-Scene graphs, decision making, object search.
关键词-场景图，决策制定，目标搜索。


## I. INTRODUCTION
## I. 引言


INTERACTIVE embodied AI tasks in large, unexplored, human-centered environments require reasoning over long horizons and a multitude of objects. In many cases, the considered environments are a priori unknown or continuously rearranged. Recent advancements have demonstrated the potential of large language models (LLMs) in generating high-level plans [1]-[4]. However, these efforts have predominantly focused on fully observed environments such as table-top manipulation, or a priori explored scenes, struggling to generate executable and grounded plans suitable for real-world robotic execution. This problem is strongly exacerbated in large scenes with numerous objects and long time horizons. In turn, this increases the risk of generating impractical sequences or hallucinations [5], [6]. Furthermore, the presence of interactive scenes and articulated objects introduces a multitude of potential states and failure cases. To address these challenges, we propose grounding LLMs in dynamically built scene graphs. Our approach incorporates a scene understanding module that, given object detections, constructs open-vocabulary scene graphs from dense maps and Voronoi graphs. These diverse representations are then tightly interweaved with an object-centric action space. Leveraging the current scene representation, we extract structured and compact textual representations of the scene to facilitate efficient planning with pre-trained LLMs.
在大规模、未充分探索、以人为中心的环境中开展的交互式具身智能任务，需要在长时间跨度内进行推理，并处理大量物体。在许多情况下，这些环境先验未知，或会被持续重新排列。近期的进展表明，大型语言模型（LLM）有潜力生成高层计划[1]-[4]。然而，这些工作大多侧重于完全可观测的环境，如桌面操作，或先验已探索的场景，难以生成可执行且可落地的计划，来适配真实世界的机器人执行。在包含众多物体且时间跨度很长的大场景中，这一问题被显著放大。进而，这会增加生成不切实际序列或幻觉的风险[5]，[6]。此外，交互式场景与可关节物体的存在，会引入大量潜在状态与失败情形。为应对这些挑战，我们提出将LLM与动态构建的场景图进行扎根对齐。我们的方法包含一个场景理解模块：基于物体检测结果，从密集图与Voronoi图中构建开放词汇的场景图。随后，这些多样化表示与面向物体的动作空间紧密耦合。利用当前的场景表示，我们提取结构化且紧凑的文本化场景表达，以便借助预训练LLM进行高效规划。


<img src="https://cdn.noedgeai.com/bo_d8lcn0s91nqc738unag0_0.jpg?x=966&y=433&w=642&h=745&r=0"/>



Fig. 1. MoMa-LLM performs long-horizon interactive object search in household environments from language queries using dynamically built scene graphs.
图1. MoMa-LLM基于动态构建的场景图，利用语言查询在家庭环境中执行长时域的交互式物体搜索。


To evaluate the efficacy of our approach, we formulate an interactive semantic search task, extending previous nonsemantic interactive tasks [7] to more complex scenarios. In this task, an agent has to find a target object within an indoor environment, encapsulating real-world challenges such as opening doors to navigate through the environment, and searching inside cabinets and drawers to find the desired object. This task is challenging as it requires reasoning about manipulation and navigation skills, operating in unexplored environments, spanning large apartments with numerous rooms and objects. Consequently, it is representative of more complex mobile manipulation tasks while retaining the specificity required for thorough evaluations and comparisons against conventional methods. Furthermore, we introduce a novel evaluation paradigm for object search tasks, employing full efficiency curves to remove the dependency on arbitrary time budgets inherent in existing methods. Additionally, we propose the AUC- $E$ metric to distill these curves into a single metric for coherent evaluation. We perform extensive experimental evaluations in both simulation and the real-world, and demonstrate that given appropriately structured representations, LLMs can leverage their accumulated knowledge about the human world to achieve exceptional results, outperforming state-of-the-art approaches across diverse fields. Our approach is zero-shot, with open-vocabulary reasoning, and inherently scalable to various mobile manipulation and household robotic tasks, as we demonstrate on a set of abstract search tasks.
为评估我们方法的有效性，我们提出一项交互式语义搜索任务，将以往的非语义交互任务[7]扩展到更复杂的场景。在该任务中，智能体需要在室内环境中找到目标物体，并包含真实世界的挑战：例如为了穿行环境而打开门，以及搜索橱柜和抽屉内部以找到所需物体。该任务具有挑战性，因为它要求同时具备操控与导航的推理能力，在未探索的环境中运行，覆盖拥有众多房间和物体的大型公寓。因此，它在保留与传统方法进行充分评估与对比所需的特定性的同时，也体现了更复杂的移动操作任务。此外，我们为物体搜索任务引入了新的评估范式，采用完整效率曲线来消除现有方法中固有的任意时间预算依赖。进一步地，我们提出AUC-$E$指标，将这些曲线蒸馏为单一度量以实现连贯评估。我们在仿真与真实世界中开展了广泛的实验评估，并表明：在经过恰当结构化表示的前提下，LLM能够利用其积累的关于人类世界的知识取得卓越结果，在多个领域超过最先进的方法。我们的方案为零样本、支持开放词汇推理，并且能够天然扩展到各类移动操作与家庭机器人任务，正如我们在一组抽象搜索任务上所展示的那样。


---



* Equal contribution.
* 等贡献。


1 Department of Computer Science, University of Freiburg, Germany.
1 计算机科学系，德国弗赖堡大学。


2 Toyota Motor Europe (TME).
2 丰田汽车欧洲（TME）。


This work was funded by Toyota Motor Europe (TME) and an academic grant from NVIDIA. We thank Aron Distelzweig for his help in implementing the HIMOS model for this task.
本研究由丰田汽车欧洲（TME）以及NVIDIA提供的学术资助资助。我们感谢Aron Distelzweig在为该任务实现HIMOS模型方面提供的帮助。


© 2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.
© 2024 IEEE。允许个人使用本材料。除当前或未来的所有其他用途外，必须从IEEE处获得许可，包括在所有媒体中进行转载/再出版，以用于广告或宣传目的，创作新的集体作品，用于转售或向服务器或列表分发，或在其他作品中复用本工作中任何受版权保护的组成部分。


---



To summarize, our main contributions are
概括而言，我们的主要贡献是：


- A scalable scene representation centered around a dynamic scene graph with open-vocabulary room clustering and classification.
- 围绕动态场景图构建的可扩展场景表示：包含开放词汇的房间聚类与分类。


- Structured compact knowledge extraction to ground LLMs in scene graphs for large unexplored environments.
- 结构化、紧凑的知识提取：将LLM扎根到场景图中，以适配大规模未探索环境。


- Semantic interactive search task for large scenes with numerous objects and receptacles.
- 面向大场景的语义交互式搜索任务：处理大量物体与容器。


- Novel evaluation paradigm for object search tasks through full efficiency curves, instead of a single time budget.
- 通过完整效率曲线为物体搜索任务提出新的评估范式，而不是依赖单一时间预算。


- We release the code at http://moma-llm.cs.uni-freiburg.de.
- 我们将代码发布在http://moma-llm.cs.uni-freiburg.de。


## II. RELATED WORK
## II. 相关工作


3D Scene Graphs serve as sparse environment representations that abstract from dense semantic maps [8]. The disassembly of large scenes into objects, regions, etc., and their representation as nodes thus provides hierarchical and object-centric representations. In addition, nodes and edges may contain semantic attributes [9]-[12]. Despite the lower geometric fidelity compared to dense maps, scene graphs prove particularly successful in the realm of high-level reasoning and planning, while providing a powerful interface with mapping [5], [11]. Orthogonally, Hydra [9] focuses on representing dynamically changing scenes. Multiple works have explored the use of scene graphs for reasoning in mobile robotics. Most build a hierarchical scene graph of the form (building, floors, rooms, objects). Different variants add edges among objects [13], add a Voronoi graph [14] for storing observations, or separate the set of objects into static and potentially moving objects [15]. While ConceptGraphs [13] and VoroNav [14] investigate the use of zero-shot perception inputs for task planning, others such as SayPlan [5], SayNav [15], and Taskography [6] focus on the reasoning task itself by utilizing ground truth semantic scene graphs [16], [17].
3D场景图可作为稀疏的环境表示，从稠密语义地图中抽象而来[8]。将大型场景拆解为对象、区域等，并将其表示为节点，因此提供了分层且以对象为中心的表示。此外，节点和边还可包含语义属性[9]-[12]。尽管与稠密地图相比几何保真度较低，场景图在高层推理与规划方面尤其成功，同时为建图提供了强大的接口[5]，[11]。此外，Hydra[9]侧重于表示动态变化的场景。已有多项工作探索了在移动机器人中使用场景图进行推理。大多数构建了（建筑、楼层、房间、对象）形式的分层场景图。不同变体会在对象之间添加边[13]，添加用于存储观测的Voronoi图[14]，或将对象集合分为静态对象和可能移动的对象[15]。虽然ConceptGraphs[13]和VoroNav[14]研究了使用零样本感知输入进行任务规划，但SayPlan[5]、SayNav[15]和Taskography[6]等则通过利用真实语义场景图[16]，[17]，将重点放在推理任务本身。


Realizing object navigation using both dynamic and interactive scene graphs has not been tackled thus far in the aforementioned works.
迄今为止，尚无上述工作实现同时利用动态与交互式场景图的物体导航。


Language Models for Planning: Several recent works have investigated language models' abilities to generate task plans for robotic manipulation. These largely focus either on static table-top scenes of limited size and a limited number of objects [1], [3], [4], or a fully observable scene. On the other side of the spectrum, vision-language-navigation investigates pure navigation tasks in large apartments to either navigate along a described path or towards a specific instance of an object [18], [19]. A smaller number of works have investigated apartment-wide mobile manipulation tasks. LLM-Planner [19] uses information retrieval of the closest matching known task. Chalvatzaki et al. [17] finetune an LLM to encode object-object relations extracted from a scene graph. SayCan [20] combines affordance values with language scores. However, all of these methods focus on tasks restricted to single rooms. SayPlan [5] focuses on identifying relevant subgraphs in large known scene graphs by iteratively extracting or collapsing nodes. Ni et al. [16] learn a transformer-based model on top of a frozen LLM to predict subtasks from fully known scene graphs. In contrast, we focus on interactive search in large fully unexplored environments. As a result, simple prompting strategies, such as lists of observed objects [13], [16], [19] or raw JSON input [5] of a full scene graph to a language model becomes insufficient, as we demonstrate in our experiments.
用于规划的语言模型：近年来，多项工作研究了语言模型为机器人操作生成任务计划的能力。这些工作大多聚焦于规模有限、对象数量有限的静态桌面场景[1]，[3]，[4]，或完全可观测场景。另一方面，视觉-语言导航研究的是大公寓中的纯导航任务，要么沿描述路径导航，要么前往某一特定物体实例[18]，[19]。也有少数工作研究了整套公寓范围的移动操作任务。LLM-Planner[19]通过信息检索找到最相近的已知任务。Chalvatzaki等[17]对LLM进行微调，以编码从场景图中提取的物体-物体关系。SayCan[20]将可供性值与语言分数结合。然而，这些方法都只关注限制在单个房间内的任务。SayPlan[5]通过迭代提取或折叠节点，着重于在大型已知场景图中识别相关子图。Ni等[16]在冻结的LLM之上训练了一个基于Transformer的模型，用于从完全已知的场景图中预测子任务。相比之下，我们关注在大型、完全未探索环境中的交互式搜索。因此，简单的提示策略，例如观测到的对象列表[13]，[16]，[19]，或将完整场景图以原始JSON输入[5]给语言模型，都会变得不足；我们的实验也证明了这一点。


Object Search has been tackled via a wide range of methods, including classical methods such as frontier exploration [21], vision-based reinforcement learning [22], or auditory signals [23]. Graph Neural networks (GNNs) have been used with scene graphs to find specific object instances with hierarchical and relational constraints [24] or in frequently changing, dynamic scenes [25], [26]. Schmalstieg et al. [7] introduced the interactive search task, in which an agent has to open doors and search through cabinets and drawers. While they focus on random target placements and a restricted number of objects and receptacles, we introduce a semantic single-object search variation of this task, which uses all objects in the scene and keeps the semantic co-occurrences in the scene intact.
物体搜索已通过多种方法解决，包括前沿探索[21]、基于视觉的强化学习[22]或听觉信号[23]等经典方法。图神经网络（GNN）已与场景图结合，用于在具有分层与关系约束的情况下寻找特定物体实例[24]，或在频繁变化的动态场景中使用[25]，[26]。Schmalstieg等[7]提出了交互式搜索任务，在该任务中，智能体必须打开门，并搜索柜子和抽屉。尽管他们关注的是随机目标放置以及受限数量的对象和容器，我们提出了这一任务的语义单物体搜索变体，它使用场景中的所有对象，并保留场景中的语义共现关系。


Non-interactive semantic search has been previously tackled. Most recent methods used language models to extract similarities or co-occurrences with the target object to score frontiers [27]-[29] or predict potential functions towards a target object with supervised learning [30]. While these works focus on pairwise score calculations, we treat it as a planning problem in which the full scene is encoded jointly. In contrast to these works, we consider objects that are not freely accessible and require interaction with the environment and thereby reasoning over multiple steps such as opening doors and receptacles instead of pure directional reasoning.
非交互式语义搜索此前已被研究。最近的方法多使用语言模型提取与目标物体的相似性或共现关系，以对前沿打分[27]-[29]，或通过监督学习预测指向目标物体的潜在函数[30]。这些工作聚焦于成对分数计算，而我们将其视为一个规划问题，在该问题中完整场景被联合编码。与这些工作不同，我们考虑的是无法自由接近、需要与环境交互的物体，因此需要多步推理，例如打开门和容器，而非纯粹的方向性推理。


Lastly, given object detections, our representation and reasoning is fully open-vocabulary - both in terms of room and object categories. Conceptually most similar to our work, SayNav [15] utilizes a scene graph together with an LLM. However, it focuses on non-interactive search, restricting the LLM's access to a room subgraph, assumes restrictions such as knowledge about scene graph edges, and relies on a hardcoded heuristic of when to go to the next (already open) door.
最后，在给定物体检测的情况下，我们的表示与推理在词汇上是完全开放的——无论是房间类别还是物体类别。就概念而言，与我们的工作最相似的是，SayNav[15]将场景图与LLM结合使用。然而，它专注于非交互式搜索，将LLM的访问限制在房间子图内，假设已知场景图边等限制，并依赖一个硬编码启发式规则来决定何时前往下一个（已打开的）门。


## III. PROBLEM STATEMENT: EMBODIED REASONING
## III. 问题陈述：具身推理


In our setting, an embodied, robotic agent is situated in a large, unexplored environment and has to complete a given task,described by a language goal $g$ . The agent is acting in a Partially Observable Markov Decision Process (POMDP) $\mathcal{M} = \; \left( {\mathcal{S},\mathcal{A},\mathcal{O},T\left( {{s}^{\prime } \mid  s,a}\right) ,P\left( {o \mid  s}\right) ,r\left( {s,a}\right) }\right)$ where $\mathcal{S},\mathcal{A}$ and $\mathcal{O}$ are the state,action and observation spaces, $T$ and $P$ describe the transition and observation probabilities, $s,{s}^{\prime }$ are the underlying current and next state, $o$ is the agent’s current observation consisting of posed RGB-D frame ${I}_{t},a$ is the current action and $r$ is the reward. To succeed in these tasks,the agent has to perceive the environment and create a representation while reasoning about how to complete the tasks through exploration and interaction with the environment.
在我们的设定中，一个具身机器人智能体处于一个大型、未探索的环境中，并必须完成一个给定任务，该任务由语言目标 $g$ 描述。该智能体在部分可观测马尔可夫决策过程（POMDP）$\mathcal{M} = \; \left( {\mathcal{S},\mathcal{A},\mathcal{O},T\left( {{s}^{\prime } \mid  s,a}\right) ,P\left( {o \mid  s}\right) ,r\left( {s,a}\right) }\right)$ 中行动，其中 $\mathcal{S},\mathcal{A}$ 和 $\mathcal{O}$ 分别表示状态、动作和观测空间，$T$ 和 $P$ 描述转移概率和观测概率，$s,{s}^{\prime }$ 是底层的当前状态和下一状态，$o$ 是智能体当前的观测，由带位姿的 RGB-D 帧组成，${I}_{t},a$ 是当前动作，$r$ 是奖励。为完成这些任务，智能体必须感知环境并构建表征，同时通过探索和与环境交互来推理如何完成任务。


We introduce the task of semantic interactive object search. In contrast to most existing works [22], [27], [28], [31], interactive object search requires manipulation of the environment to navigate and explore it. As in realistic, human-centric environments, doors may block pathways and objects are not openly visible but may be stored away in receptacles like drawers or cabinets. We extend the interactive task introduced in [7] to a much larger number of objects and receptacles and a prior distribution of realistic room-object and object-object relations. As a result, other objects in the scene can provide valuable information about the position of the target. While existing tasks such as the Habitat challenge and Robothor use semantic placements, they do not support any physical interactions or objects placed within receptacles.
我们提出语义交互式物体搜索任务。与大多数现有工作 [22]、[27]、[28]、[31] 不同，交互式物体搜索要求通过操控环境来导航和探索。在真实的、以人为中心的环境中，门可能会阻挡通道，物体也不会完全可见，而可能被存放在抽屉或橱柜等容器中。我们将 [7] 中提出的交互任务扩展到更多的物体和容器，以及更符合现实的房间-物体和物体-物体关系先验分布。因此，场景中的其他物体可以为目标位置提供有价值的信息。虽然 Habitat challenge 和 Robothor 等现有任务使用语义放置，但它们不支持任何物理交互，也不支持放置在容器中的物体。


<img src="https://cdn.noedgeai.com/bo_d8lcn0s91nqc738unag0_2.jpg?x=137&y=153&w=1527&h=473&r=0"/>



Fig. 2. MoMa-LLM: From posed RGB-D images and semantics, we construct a semantic 3D map from which we extract a various occupancy maps in the BEV space and construct a navigational Voronoi graph. Through room clustering and room-object assigments we then build up a hierarchical scene graph. From this scalable scene representation, we extract the task-relevant knowledge and encode it into a structured language representation. A large language model then produces high-level commands which are executed by low-level subpolicies. These in turn draw on and update the scene representations.
图 2. MoMa-LLM：从带位姿的 RGB-D 图像和语义信息出发，我们构建语义 3D 地图，并从中提取 BEV 空间中的多种占用地图，进而构建导航 Voronoi 图。通过房间聚类和房间-物体分配，我们再构建层次化场景图。从这一可扩展的场景表征中，我们提取任务相关知识，并将其编码为结构化语言表示。随后，大语言模型生成高层命令，由低层子策略执行。这些子策略反过来会利用并更新场景表征。


We implement the task in the iGibson scenes [32], consisting of 15 interactive apartments based on scans of real houses. At the beginning of an episode, all doors are closed and the agent is given a task description in natural language. The task is deemed successful if the agent has observed an instance of the target category and calls done().
我们在 iGibson 场景中实现该任务 [32]，这些场景由基于真实住宅扫描构建的 15 个交互式公寓组成。在一个 episode 开始时，所有门都关闭，智能体会获得一个自然语言任务描述。若智能体已经观察到目标类别的一个实例并调用 done()，则任务视为成功。


The iGibson scenes contain realistic furniture and room distributions, but few other objects are placed in relation to this. We enrich the scenes with realistic object placements, both within receptacles and on top of furniture, by extending and matching previously introduced prior distributions ${P}^{\text{ prior }}$ over room and object relations [25], by aligning room names manually and matching object names via SBERT cosine similarities. We then assume that all objects that can be found on top of an object and that fit in size, can also be found inside it and vice-versa. Given a valid scene instantiation, we then draw a target category $g \sim  U$ (scene) from all categories in the scene. This results in the procedural generation of a wide range of tasks over 84 possible target classes. Further details can be found in the Supplementary Sec. S.1.
iGibson 场景包含逼真的家具和房间分布，但与之相关的其他物体较少。我们通过扩展并匹配先前引入的关于房间与物体关系的先验分布 ${P}^{\text{ prior }}$ [25]，并通过手动对齐房间名称、利用 SBERT 余弦相似度匹配物体名称，来丰富这些场景中的现实物体摆放，既包括放置在容器内，也包括放置在家具上。随后，我们假设，所有可以放在某个物体上且尺寸适配的物体，也可以被放入其中，反之亦然。在给定一个有效的场景实例后，我们再从场景中的所有类别中抽取一个目标类别 $g \sim  U$（场景）。这使得能够围绕 84 个可能的目标类别程序化生成大量任务。更多细节见补充材料第 S.1 节。


## IV. MOMA-LLM
## IV. MOMA-LLM


To address the challenges of interactive open-vocabulary household tasks, we propose MoMa-LLM, which intertwines high-level reasoning with scalable dynamic scene representations. We ground large-language models in hierarchical 3D scene graphs ${\mathcal{G}}_{S}$ that hold object- and room-level entities as well as a more fine-grained Voronoi graph for navigation. The LLM provides high-level actions that are executed through low-level skills as shown in Fig. 2. In general, we assume access to ground truth perception for semantic masks, depth, localization and handle detection as the focus of this work is on the reasoning aspect.
为应对交互式开放词汇家居任务的挑战，我们提出 MoMa-LLM，它将高层推理与可扩展的动态场景表征相融合。我们将大型语言模型建立在分层三维场景图 ${\mathcal{G}}_{S}$ 之上，该图同时包含物体级与房间级实体，并为导航提供更细粒度的 Voronoi 图。LLM 给出高层动作，并通过低层技能执行，如图 2 所示。总体而言，我们假设能够获得语义掩码、深度、定位以及手部检测的真实感知，因为本工作的重点在于推理方面。


## A. Hierarchical 3D Scene Graph
## A. 分层3D场景图


To provide an LLM with structured input, we craft a hierarchical scene graph that includes a navigational Voronoi graph.
为向LLM提供结构化输入，我们构建了一个包含导航Voronoi图的分层场景图。


1) Dynamic RGB-D Mapping: The agent perceives posed RGB-D frames $\left\{  {{I}_{0},\ldots ,{I}_{t}}\right\}$ including semantics from the environment. The contained points are transformed into the global coordinate frame and arranged on a 3D voxel grid ${\mathcal{M}}_{t}$ . As we tackle an interactive problem, our map is dynamically updated based on novel explored areas or the occurrence of object dynamics in the scene. To infer obstacle positions, walls and explored free space, we first obtain the highest occupied entry per stixel in ${\mathcal{M}}_{t}$ . These entries are then turned into a two-dimensional bird’s-eye-view (BEV) occupancy map ${\mathcal{B}}_{t}$ by inferring all occupied positions except for those classified as free space ${\mathcal{F}}_{t}$ . The latter in turn represents the navigable area that is used for robot exploration.
1) 动态RGB-D建图：智能体感知带位姿的RGB-D帧$\left\{  {{I}_{0},\ldots ,{I}_{t}}\right\}$，其中包含来自环境的语义信息。所含点被变换到全局坐标系中，并排列在3D体素网格${\mathcal{M}}_{t}$上。由于我们处理的是交互式问题，地图会根据新探索区域或场景中物体动态的发生而动态更新。为推断障碍物位置、墙体和已探索自由空间，我们首先获取${\mathcal{M}}_{t}$中每个stixel的最高占据项。随后，这些项被转化为二维鸟瞰图（BEV）占据图${\mathcal{B}}_{t}$，方法是推断所有占据位置，但将被归类为自由空间${\mathcal{F}}_{t}$的部分除外。后者进一步表示用于机器人探索的可导航区域。


2) Voronoi Graph: Similar to Hydra [9], we abstract from the created dense maps by computing a navigational graph ${\mathcal{G}}_{\mathcal{V}}$ . We first inflate ${\mathcal{B}}_{t}$ using an Euclidean signed distance field (ESDF) formulation for robustness, but overwrite free space coordinates as given in ${\mathcal{F}}_{t}$ as zero. Based on this,we compute a Generalized Voronoi Diagram (GVD) that holds a set of points $\mathcal{V}$ with the same clearance to the closest obstacles drawn from ${\mathcal{B}}_{t}$ .
2) Voronoi图：与Hydra [9]类似，我们通过计算导航图${\mathcal{G}}_{\mathcal{V}}$从所生成的稠密地图中抽象出结果。为增强鲁棒性，我们首先使用欧几里得有符号距离场（ESDF）对${\mathcal{B}}_{t}$进行膨胀处理，但将${\mathcal{F}}_{t}$中给出的自由空间坐标覆盖为零。基于此，我们计算广义Voronoi图（GVD），其中包含一组点$\mathcal{V}$，它们与从${\mathcal{B}}_{t}$中提取的最近障碍物保持相同的净空距离。


We exclude all nodes that lie in the immediate vicinity of obstacles or do not reside within ${\mathcal{B}}_{t}$ . Given the GVD boundaries, we construct edges $\mathcal{E}$ among $\mathcal{V}$ and obtain our navigational Voronoi graph ${\mathcal{G}}_{\mathcal{V}} = \left( {\mathcal{V},\mathcal{E}}\right)$ . Throughout our experiments,we found that extracting the largest connected component of the graph provides the robot-centric Voronoi graph while other components commonly lie outside the explored area. Lastly, we sparsify ${\mathcal{G}}_{\mathcal{V}}$ to obtain fewer navigational nodes.
我们排除所有位于障碍物紧邻区域内或不在${\mathcal{B}}_{t}$内的节点。根据GVD边界，我们在$\mathcal{E}$与$\mathcal{V}$之间构建边，并得到我们的导航Voronoi图${\mathcal{G}}_{\mathcal{V}} = \left( {\mathcal{V},\mathcal{E}}\right)$。在整个实验中，我们发现提取图中最大的连通分量可得到以机器人为中心的Voronoi图，而其他分量通常位于已探索区域之外。最后，我们对${\mathcal{G}}_{\mathcal{V}}$进行稀疏化，以获得更少的导航节点。


3) 3D Scene Graph: The MoMa-LLM-policy operates on an attributed 3D scene graph ${\mathcal{G}}_{S}$ that holds different abstraction levels, namely rooms and objects. We first separate the global Voronoi graph ${\mathcal{G}}_{\mathcal{V}}$ into multiple regions. To do so,we eliminate edges and nodes of ${G}_{\mathcal{V}}$ near doors instead of separating graphs at geometrical constrictions [9]. Using a mixture of Gaussians, we generate a two-dimensional probability distribution over all observed door positions in the environment:
3) 3D场景图：MoMa-LLM策略运行于一个带属性的3D场景图${\mathcal{G}}_{S}$上，该图包含不同抽象层级，即房间和物体。我们首先将全局Voronoi图${\mathcal{G}}_{\mathcal{V}}$划分为多个区域。为此，我们移除${G}_{\mathcal{V}}$中靠近门口的边和节点，而不是像文献[9]那样在几何收缩处分割图。利用高斯混合模型，我们在环境中所有已观测到的门位置上生成一个二维概率分布：


$$
{\rho }_{\mathcal{N}}\left( {\mathbf{x},\mathbf{H}}\right)  = \frac{1}{{N}_{D}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{D}}{K}_{\mathbf{H}}\left( {\mathbf{x} - {\mathbf{x}}_{\mathbf{i}}}\right) , \tag{1}
$$



<img src="https://cdn.noedgeai.com/bo_d8lcn0s91nqc738unag0_3.jpg?x=143&y=166&w=734&h=320&r=0"/>



Fig. 3. Room Classification Prompt: based on the objects and room clusters of the scene graph, an LLM performs open-vocabulary classification.
图3. 房间分类提示：基于场景图中的物体和房间聚类，LLM执行开放词汇分类。


where ${\mathbf{x}}_{i} = \left( {{x}_{i},{y}_{i}}\right)$ are the door center coordinates, ${K}_{\mathbf{H}}$ is the scaled Gaussian kernel of observed doors and $\mathbf{H}$ the bandwidth matrix, which we set to 2.0 based on manual tuning on the training scenes. Edges that fall into high-probability regions and exceed an empirically tuned probability threshold are disregarded along with isolated nodes. Following this principle, we obtain the separated Voronoi graph ${\mathcal{G}}_{\mathcal{V}}^{R}$ covering distinct rooms. In the next step, we infer the high-level connectivity among rooms by calculating the shortest paths between nodes of ${\mathcal{G}}_{\mathcal{V}}$ that belong to disjoint components of ${\mathcal{G}}_{\mathcal{V}}^{R}$ . Whenever a path traverses just two distinct rooms as given by ${\mathcal{G}}_{\mathcal{V}}^{R}$ ,the two rooms count as immediate neighbors. Finally, we map objects to rooms. For each object $o \in  {\mathcal{G}}_{S}$ ,we identify the node that minimizes the distance ${d}_{vo}$ to the closest viewpoint ${v}_{p}$ from which the object was seen. To this end, we calculate the shortest path from the object $o$ to this viewpoint. It consists of the path on the Voronoi graph ${\mathcal{G}}_{\mathcal{V}}$ ,and the Euclidean distances $d$ from the Voronoi nodes ${n}_{o}$ and ${n}_{{v}_{p}}$ to the object $o$ and viewpoint ${v}_{p}$ ,respectively. By weighting the distance to the object with an exponent of $\lambda  = {1.3}$ ,we ensure to prefer nodes close to the object. Objects are then assigned to the room label $R$ of the node ${n}_{o}$ that minimizes Eq. (2). This prohibits the erroneous assignments of objects to a neighboring room through walls. Doors may be connected to multiple rooms.
其中${\mathbf{x}}_{i} = \left( {{x}_{i},{y}_{i}}\right)$为门中心坐标，${K}_{\mathbf{H}}$为观测到的门的缩放高斯核，$\mathbf{H}$为带宽矩阵，我们根据训练场景上的人工调参将其设为2.0。落入高概率区域并超过经验调优概率阈值的边会与孤立节点一并被忽略。按照这一原则，我们得到覆盖不同房间的分离 Voronoi 图${\mathcal{G}}_{\mathcal{V}}^{R}$。下一步，我们通过计算属于${\mathcal{G}}_{\mathcal{V}}$且位于${\mathcal{G}}_{\mathcal{V}}^{R}$不连通分量中的节点之间的最短路径，推断房间之间的高层连通性。每当一条路径如${\mathcal{G}}_{\mathcal{V}}^{R}$所示仅穿过两个不同房间时，这两个房间就被视为直接邻居。最后，我们将物体映射到房间。对于每个物体$o \in  {\mathcal{G}}_{S}$，我们找到使距离${d}_{vo}$最小的节点，即从该物体被观察到的最近视点${v}_{p}$。为此，我们计算从物体$o$到该视点的最短路径。它由 Voronoi 图${\mathcal{G}}_{\mathcal{V}}$上的路径，以及从 Voronoi 节点${n}_{o}$和${n}_{{v}_{p}}$到物体$o$和视点${v}_{p}$的欧氏距离$d$组成。通过对物体距离施加指数$\lambda  = {1.3}$的权重，我们确保优先选择靠近物体的节点。随后，将物体分配给使式(2)最小的节点${n}_{o}$的房间标签$R$。这可避免将物体错误地穿墙分配到相邻房间。门可能连接多个房间。


$$
{d}_{vo} = \mathop{\min }\limits_{{{n}_{o},{n}_{{v}_{p}} \in  {\mathcal{G}}_{\mathcal{V}}^{R}}}\operatorname{path}\left( {{n}_{o},{n}_{{v}_{p}}}\right)  + d{\left( o,{n}_{o}\right) }^{\lambda } + d\left( {{v}_{p},{n}_{{v}_{p}}}\right) \tag{2}
$$



4) Room Classification: Similar to Chen et al. [33], we perform room classification by providing an LLM with the set of object categories contained in each room. We perform this as open-set classification, in which we let the LLM freely pick the room categories deemed most appropriate.
4）房间分类：与 Chen et al. [33] 类似，我们通过向 LLM 提供每个房间中包含的物体类别集合来进行房间分类。我们将其作为开放集分类来执行，即让 LLM 自由选择其认为最合适的房间类别。


The resulting LLM prompts are detailed in Fig. 3. Room classification is performed in each high-level policy step, as the explored scene and scene graph evolve. We provide a concise overview of all scene graph layers in Tab. S.1.
生成的 LLM 提示词详见图3。由于所探索的场景和场景图会不断演化，房间分类会在每个高层策略步骤中进行。我们在表S.1中简要概述了所有场景图层。


## B. High-Level Action Space
## B. 高层动作空间


We design an object-centric action space, which is tightly intertwined with the different granularities of the scene representation. It consists of the following high-level actions:
我们设计一种以物体为中心的动作空间，它与场景表征的不同粒度紧密耦合。它包含以下高层动作：


navigate(room_name, object_name): Navigation to an object in a room via an ${\mathrm{A}}^{ * }$ planner in the explored BEV-map ${\mathcal{B}}_{t}$ ,inflated by ${0.1}\mathrm{\;m}$ . It first navigates to the Voronoi node associated with the object, then to the most central, free point on an arc around the object. This enables robust navigation to objects in partially explored space and ensures navigation to the correct room through the Voronoi assignment detailed in Sec. IV-A3. Navigation is considered successful if the agent reaches within ${1.5}\mathrm{\;m}$ of the object.
navigate(room_name, object_name)：借助在已探索的 BEV 地图 ${\mathcal{B}}_{t}$ 中的 ${\mathrm{A}}^{ * }$ 规划器进行导航，并通过 ${0.1}\mathrm{\;m}$ 进行膨胀。它首先导航到与该物体相关联的 Voronoi 节点，然后再导航到围绕物体的弧线上最中心的、空闲的点。这使得在部分探索空间中对物体的导航更稳健，并确保通过第 IV-A3 节中所述的 Voronoi 分配导航到正确的房间。若智能体到达物体的距离在 ${1.5}\mathrm{\;m}$ 之内，则视为导航成功。


go_to_and_open(room_name, object_name): Navigate to a specific object, then open it. For doors, continue to navigate into the opened door frame.
go_to_and_open(room_name, object_name)：导航到特定物体，然后打开它。对于门，则继续导航进入已打开的门框。


close(room_name, object_name): Equivalent to opening. explore(room_name): Move to an unexplored frontier within this room. Deemed successful if within ${0.5}\mathrm{\;m}$ of the frontier. done(): Terminate the episode and evaluate if the target object has been found.
close(room_name, object_name)：等价于打开。explore(room_name)：移动到本房间内尚未探索的前沿。若在该前沿 ${0.5}\mathrm{\;m}$ 之内，则视为成功。done()：终止该回合，并评估是否已找到目标物体。


Ambiguities of multiple instances of the specified class in a room are resolved by selecting the closest instance. The subpolicies then generate actions in the low-level action space and return once they succeed or encounter a failure. Throughout their execution, they continuously update the scene representations. Refer to the Supplementary Sec. S. 1 for details.
对于房间内指定类别的多个实例所引入的歧义，通过选择最近的实例来解决。随后子策略将在低层动作空间中生成动作，并在其成功或遇到失败后返回。在执行过程中，它们会持续更新场景表征。细节请参见补充第 S.1 节。


## C. Grounded High-Level Planning
## C. 基于场景的高层规划


We encode the accumulated knowledge of the scene graph into natural language by extracting the relevant components and embedding them in a problem-specific structured manner. Our method fulfills three properties: (i) grounding - guiding the LLM to adhere to the physical realities of the scene, (ii) specificity - avoiding long or irrelevant context queries that increase hallucinations and the difficulty of the planning problem [5], [6], and (iii) open-set - our reasoning is open-vocabulary and performs in a zero-shot manner, enabling direct deployment with unknown semantics and perception models. The resulting prompt for the language model is shown in Fig. 4. In the following, we describe the main components of structured encoding. We demonstrate the importance of this structure in Sec. V.
我们通过提取相关组成部分，并以面向特定问题的结构化方式将其嵌入，自然语言中编码场景图所累积的知识。我们的方法满足三个性质：（i）落地（grounding）——引导LLM遵守场景的物理现实；（ii）特异性（specificity）——避免冗长或不相关的上下文查询，从而减少幻觉并降低规划问题的难度[5]，[6]；以及（iii）开放集（open-set）——我们的推理是开放词汇的，并以零样本方式执行，使其能够直接部署到未知语义与感知模型中。语言模型的结果提示词如图4所示。下面，我们描述结构化编码的主要组成。我们将在第V节展示这种结构的重要性。


1) Scene Structure: We encode the main room-object structure from the scene graph into a structured list of rooms and their containing objects and encode path distances (based on an ${\mathrm{A}}^{ * }$ -planner) by binning them and mapping them to adjectives [17], as detailed in Supplementary Sec. S.4. We then employ the following filtering to allow for compact text encodings: we summarize matching nodes within a room with a counter, we filter out open doors that provide no new connectivity, and we encode object states directly within the object name, e.g. as "opened" or "closed [object-name]".
1）场景结构：我们从场景图中编码主要的房间-物体结构，将其表示为由房间及其包含物体构成的结构化列表；然后基于${\mathrm{A}}^{ * }$-planner对路径距离进行分箱，并将其映射为形容词[17]，如补充第S.4节所述。接着，我们使用以下过滤以获得紧凑的文本编码：在房间内对匹配节点进行计数汇总，过滤掉不会带来新连通性的敞开门，并将物体状态直接编码在物体名称中，例如写作“opened”或“closed [object-name]”。


2) Partial Observability: As the environment is initially unknown, it requires explicit reasoning about exploration-exploitation trade-offs. We identify frontiers to explorable areas [21], then leverage the scene graph to provide them with semantic meaning. Firstly, we associate each frontier with a room through matching with ${G}_{\mathcal{V}}^{R}$ . Secondly,we apply hole-filling to the BEV map to differentiate whether a frontier is an encapsulated area within a room, such as occluded space behind furniture, or whether the frontier is leading out to new areas. Correspondingly, we then represent them as "unexplored area" within a room, while frontiers that lead to other areas are listed separately, see Fig. 4. The second type of unexplored space is receptacles that may contain target objects. Together with the encoded object states, we find that the language model is capable of inferring affordances from the object descriptions, removing the need to explicitly encode them. If trying to open objects that cannot be opened, the according subpolicy will fail and the LLM has to reason about an appropriate response.
2）部分可观测性：由于环境一开始未知，因此需要对探索-利用权衡进行显式推理。我们识别可探索区域的前沿[21]，随后利用场景图为其赋予语义含义。首先，我们通过与${G}_{\mathcal{V}}^{R}$的匹配，将每个前沿关联到一个房间。其次，我们对BEV地图进行填洞（hole-filling），以区分前沿是房间内部的封闭区域——例如家具遮挡后的隐蔽空间——还是通向全新区域的通路。相应地，我们将前沿表述为房间内的“未探索区域”，而会导向其他区域的前沿则单独列出，见图4。第二类未探索空间是可能包含目标物体的容器。结合编码的物体状态，我们发现语言模型能够从物体描述中推断可操作性（affordances），从而无需显式编码。如果尝试打开无法打开的物体，对应的子策略将失败，LLM必须推理出合适的回应。


<img src="https://cdn.noedgeai.com/bo_d8lcn0s91nqc738unag0_4.jpg?x=216&y=159&w=1369&h=672&r=0"/>



Fig. 4. High-level Reasoning Prompt: We encode the extracted scene representation to natural language, providing structured information to a language model.
图4. 高层推理提示词：我们将提取到的场景表示编码为自然语言，为语言模型提供结构化信息。


3) History in Dynamic Scenes: Given the size of the scenes, the conversation history quickly grows too large to provide to a language model directly. Instead, we aim to find the most compact representation of previous actions to fulfill the Markov property. For each high-level decision, we encode the latest scene representation and start a new query to the LLM. As the scene representation is dynamically updated, this automatically encodes all newly acquired knowledge. To account for previous interactions, we provide the LLM with a history of the last $h$ actions. But as the scene graph changes dynamically,the previous room- and object-centric function calls may no longer match the current scene. Instead, we keep track of interaction positions, and then re-align the previous actions by matching the positions to their closest Voronoi nodes and associated room labels. We then provide the LLM with a list of the re-aligned function calls, as shown in Fig. 4. E.g., the agent executes explore(living room). But revealing a fridge, later classifies the same room as kitchen. The realigned history will then correctly reflect this action as explore(kitchen).
3）动态场景中的历史：鉴于场景规模较大，对话历史很快会膨胀到无法直接提供给语言模型。相反，我们希望找到对先前动作最紧凑的表示，以满足马尔可夫性质。对于每个高层决策，我们编码最新的场景表示，并向LLM发起新的查询。由于场景表示会被动态更新，它会自动编码所有新获得的知识。为考虑先前交互，我们为LLM提供最近$h$次动作的历史。但由于场景图会动态变化，先前基于房间与物体的功能调用可能不再与当前场景匹配。为此，我们记录交互位置，然后通过将位置匹配到最近的Voronoi节点及其对应的房间标签，对先前动作进行重新对齐。随后，我们为LLM提供重新对齐后的功能调用列表，如图4所示。例如，智能体执行explore(living room)。但在随后暴露出冰箱之后，同一个房间会被重新分类为kitchen。此时，重新对齐后的历史将正确反映该动作为explore(kitchen)。


4) Re-trial and Re-planning: Extracting meaningful feedback for failure reasons for robots in the real world remains an open problem [2], as the number of possible failure reasons is almost unlimited. Instead, we provide very limited feedback about subpolicy success, which can be readily generated in the real world. We rely on a simple success state to the action history, stating "success", "failure", or "invalid argument" in case the output of the LLM could not be matched to the scene graph. We differentiate two cases of replanning: if the agent attempted interactions or commands that cannot be parsed or are deemed infeasible without attempting execution, we have not gained any new information about the scene, and we continue the conversation with the message "The last action <function-call> failed. Please try another command.". In case of more than five failures without state change, we terminate the episode as unsuccessful. If a subpolicy attempted execution but failed to complete its task, we re-encode the latest scene, update the action history, and let the LLM make a normal next decision with the updated state.
4）重试与重新规划：为真实世界中的机器人提取失败原因所对应的有意义反馈仍是一个开放问题[2]，因为可能的失败原因几乎不受限制。相反，我们只提供非常有限的关于子策略是否成功的反馈，这些反馈在真实世界中能够容易地产生。我们依赖一个简单的成功状态来记录动作历史：若LLM的输出无法与场景图匹配，则在历史中标记“success”“failure”，或“invalid argument”。我们区分两种重新规划情况：如果智能体尝试了无法解析的交互或命令，或在不执行的情况下被认为不可行，那么我们并未获得关于场景的任何新信息；此时我们继续对话，并发送消息“The last action <function-call> failed. Please try another command.”。如果在状态没有变化的前提下发生超过五次失败，我们将把该回合终止为不成功。若子策略确实尝试执行但未能完成其任务，我们就对最新场景重新编码，更新动作历史，并让LLM在更新后的状态下做出常规的下一步决策。


## V. EXPERIMENTS
## V. 实验


As language models, we use gpt-4-1106-preview for the high-level reasoning and gpt-3.5-turbo-1106 for the room classification task [34]. For simplicity, we recompute the scene graph each time step. More advanced implementations would reduce costs through incremental updates.
作为语言模型，我们使用 gpt-4-1106-preview 进行高层推理，使用 gpt-3.5-turbo-1106 完成房间分类任务 [34]。为简化起见，我们在每个时间步都会重新计算场景图。更先进的实现可通过增量更新来降低成本。


Baselines: We compare our approach against heuristic-based, recent learning-based, and language-based methods. We provide all baselines except Unstructured LLM with a ground truth done() decision when the object has been observed.
基线：我们将方法与基于启发式、近期学习型以及基于语言的方法进行比较。除非结构化 LLM 外，我们为所有基线提供当物体被观测到时的 ground truth done() 决策。


Random: uniform random choice among all available actions (detected frontiers and closed objects).
随机：在所有可用动作中进行均匀随机选择（检测到的前沿与已关闭的物体）。


Greedy: greedily triggers the closest available action based on the shortest path calculated by an ${\mathrm{A}}^{ * }$ -planner.
贪心：基于 ${\mathrm{A}}^{ * }$ -planner 计算得到的最短路径，贪婪地触发最近的可用动作。


ESC-Interactive: ESC is a recent approach for semantic object search [27] which scores frontiers based on object-object and object-room co-occurrences as well as their distance. We extend the approach to interactive search by using the same rules to score openable objects and then select the action with the highest value. Co-occurrences are based on similarities of a finetuned Deberta-v3 language model [35], following the authors' instructions. To isolate the impact of the decision making, we use the same scene graph and low-level policies as for our method.
ESC-交互：ESC 是一种用于语义物体搜索的近期方法 [27]，它根据物体-物体与物体-房间的共现情况以及距离对前沿进行评分。我们通过对可开启物体使用相同规则来扩展该方法以支持交互式搜索，然后选择得分最高的动作。共现基于一个微调的 Deberta-v3 语言模型 [35] 的相似性，并遵循作者的指示。为隔离决策过程的影响，我们使用与我们方法相同的场景图与低层策略。


HIMOS: A hierarchical reinforcement learning approach [7] which learns to combine frozen low-level policies for interactive object search, based on a semantic map memory. We adapt it by giving it the same subpolicies as our approach and scale it to the much larger number of objects in our scenes by restricting the instance navigation to target and articulated objects.
HIMOS：一种层次化强化学习方法 [7]，基于语义地图记忆学习如何结合用于交互式物体搜索的冻结低层策略。我们通过为其提供与我们方法相同的子策略来进行改编，并通过将实例导航限制为目标与关节化物体，将其扩展到我们场景中数量更大的物体集合。


Unstructured LLM: This baseline provides the scene graph in a JSON format without any additional structure to the language model. The prompt structure is derived from SayPlan [5], adapted to the instructions and scene graph of our method. See Supplementary Sec. S. 5 for a prompt example.
非结构化 LLM：该基线以 JSON 格式提供场景图，并且不对其额外添加任何结构供语言模型使用。提示结构来源于 SayPlan [5]，并根据我们方法的指令与场景图进行调整。提示示例见补充材料 Sec. S. 5。


MoMa-LLM w/ Hydra: We incorporate the room segmentation approach introduced by Hydra [9] into our scene graph construction pipeline to measure the impact of our proposed door-wise room separation mechanism.
MoMa-LLM w/ Hydra：我们将 Hydra [9] 提出 的房间分割方法纳入场景图构建流程，用于衡量我们提出的按门划分房间机制的影响。


Metrics: We use three types of metrics to evaluate methods.
指标：我们使用三类指标来评估各方法。


Success rate (SR): the share of episodes in which the agent finds the target object. We terminate an episode if the agent reaches 50 high-level steps, indicating being stuck.
成功率（SR）：代理找到目标物体的所有 episode 中的占比。如果代理达到 50 个高层步骤则终止该 episode，表示陷入卡住状态。


Success weighted by Path Length (SPL) [36] calculates the fraction of distance traveled to the shortest possible path and weights it by whether the episode was successful. This metric does not take into account the costs of object interactions.
按路径长度加权的成功率（SPL）[36] 计算实际行进距离占最短可行路径距离的比例，并根据 episode 是否成功对其进行加权。该指标不考虑物体交互带来的成本。


Search efficiency curve and AUC-E: While the commonly used success and SPL metrics allow for reducing the evaluation to a single number, they rely on an arbitrarily set maximum allowed time budget or number of environment steps. As a result, these metrics do not differentiate between methods that search thoroughly but on average take more steps versus methods that search large areas very quickly, but that might fail to search every corner for small objects. Instead, we reason that the desired time budget depends heavily on the use case and propose a new metric that evaluates the full efficiency curve. For each possible budget (number of steps), we calculate the share of episodes that succeeded with this or fewer number of steps. This results in an efficiency curve, in which the best policies are located in the top left corner, enabling the comparison of success rates for arbitrary budgets. We can still reduce this to a single number by calculating the area under the efficiency curve (AUC-E). A perfect (but unachievable) policy, that finds all objects in a single step will have a value of one, a policy that does not find any objects will have a value of zero. We calculate the integral up to 5,000 low-level steps, at which points almost all methods make no further progress.
搜索效率曲线与 AUC-E：尽管常用的成功率与 SPL 指标允许将评估简化为单一数值，但它们依赖于人为设定的最大允许时间预算或环境步数。因此，这些指标无法区分“搜索得很彻底但平均用更多步骤”的方法与“在较大区域内很快搜索但可能找不到每个角落小物体”的方法。我们认为期望的时间预算高度依赖具体使用场景，并提出一种评估完整效率曲线的新指标。对于每个可能预算（步数），我们计算在该预算或更少步数内成功的 episode 占比。由此得到效率曲线，其中最优策略位于左上角，从而可对任意预算下的成功率进行比较。我们仍可通过计算效率曲线下的面积（AUC-E）将其压缩为单一数值。理想的（但无法实现的）策略若能在一步找到所有物体，其取值为 1；不找到任何物体的策略取值为 0。我们在最多 5,000 个低层步骤范围内计算积分，此时几乎所有方法都不再取得进一步进展。


## A. Simulation Experiments
## A. 模拟实验


We instantiate the task in the iGibson simulator [32] with a Fetch robot. In contrast to previous LLM-based works [5], [15], we evaluate all approaches in completely unseen apartments, following the data split of the iGibson challenge into eight training scenes for the development of all modules and prompt engineering and seven test scenes. For each scene, we evaluate the agent over 25 procedurally generated episodes with randomized start poses, target objects, and object distributions.
我们在 iGibson 模拟器 [32] 中使用一台 Fetch 机器人来构建任务。与以往基于 LLM 的工作 [5], [15] 不同，我们在完全未见过的公寓中评估所有方法，依据 iGibson 挑战的数据划分：用八个训练场景用于所有模块与提示工程的开发，并用七个测试场景进行评估。对于每个场景，我们在 25 个通过程序生成的回合中对智能体进行评估；每个回合的起始姿态、目标物体以及物体分布均随机化。


TABLE I
表 I


INTERACTIVE OBJECT SEARCH RESULTS IN SIMULATION
模拟中的交互式物体搜索结果


<table><tr><td>Model</td><td>SR</td><td>SPL</td><td>AUC-E</td><td>Object Interactions</td><td>Distance Traveled</td><td>Infeasible Actions</td></tr><tr><td>Random</td><td>93.1</td><td>50.2</td><td>77.0</td><td>5.7</td><td>32.9</td><td>-</td></tr><tr><td>Greedy</td><td>85.7</td><td>50.9</td><td>72.9</td><td>8.1</td><td>22.3</td><td>-</td></tr><tr><td>ESC-Interactive</td><td>95.4</td><td>62.7</td><td>84.5</td><td>4.1</td><td>19.6</td><td>-</td></tr><tr><td>HIMOS</td><td>93.7</td><td>48.5</td><td>77.4</td><td>4.8</td><td>35.9</td><td>-</td></tr><tr><td>Unstructured LLM</td><td>86.3</td><td>59.4</td><td>77.6</td><td>3.6</td><td>18.5</td><td>0.41</td></tr><tr><td>MoMa-LLM w/ Hydra</td><td>92.0</td><td>61.9</td><td>84.3</td><td>2.7</td><td>12.9</td><td>0.06</td></tr><tr><td>MoMa-LLM (ours)</td><td>97.7</td><td>63.6</td><td>87.2</td><td>3.9</td><td>18.2</td><td>0.19</td></tr><tr><td>Ours w/o frontiers</td><td>79.4</td><td>55.0</td><td>72.2</td><td>4.3</td><td>15.6</td><td>0.91</td></tr><tr><td>Ours w/o history</td><td>94.9</td><td>63.0</td><td>84.1</td><td>3.6</td><td>17.1</td><td>0.26</td></tr><tr><td>Ours w/ room-history</td><td>97.1</td><td>63.0</td><td>86.6</td><td>3.8</td><td>17.8</td><td>0.28</td></tr><tr><td>Ours w/o distances</td><td>97.1</td><td>61.5</td><td>86.4</td><td>3.8</td><td>18.9</td><td>0.24</td></tr></table>
<table><tbody><tr><td>模型</td><td>SR</td><td>SPL</td><td>AUC-E</td><td>物体交互</td><td>行进距离</td><td>不可行动作</td></tr><tr><td>随机</td><td>93.1</td><td>50.2</td><td>77.0</td><td>5.7</td><td>32.9</td><td>-</td></tr><tr><td>贪心</td><td>85.7</td><td>50.9</td><td>72.9</td><td>8.1</td><td>22.3</td><td>-</td></tr><tr><td>ESC-交互</td><td>95.4</td><td>62.7</td><td>84.5</td><td>4.1</td><td>19.6</td><td>-</td></tr><tr><td>HIMOS</td><td>93.7</td><td>48.5</td><td>77.4</td><td>4.8</td><td>35.9</td><td>-</td></tr><tr><td>非结构化LLM</td><td>86.3</td><td>59.4</td><td>77.6</td><td>3.6</td><td>18.5</td><td>0.41</td></tr><tr><td>MoMa-LLM（带Hydra）</td><td>92.0</td><td>61.9</td><td>84.3</td><td>2.7</td><td>12.9</td><td>0.06</td></tr><tr><td>MoMa-LLM（我们的）</td><td>97.7</td><td>63.6</td><td>87.2</td><td>3.9</td><td>18.2</td><td>0.19</td></tr><tr><td>我们的方法（不含前沿）</td><td>79.4</td><td>55.0</td><td>72.2</td><td>4.3</td><td>15.6</td><td>0.91</td></tr><tr><td>我们的方法（不含历史）</td><td>94.9</td><td>63.0</td><td>84.1</td><td>3.6</td><td>17.1</td><td>0.26</td></tr><tr><td>我们的方法（含房间历史）</td><td>97.1</td><td>63.0</td><td>86.6</td><td>3.8</td><td>17.8</td><td>0.28</td></tr><tr><td>我们的方法（不含距离）</td><td>97.1</td><td>61.5</td><td>86.4</td><td>3.8</td><td>18.9</td><td>0.24</td></tr></tbody></table>


Top two in bold and underline. Object interactions, distance travelled and infeasible actions averaged over all episodes, including early terminated failures. Infeasible: avg. number of steps the LLM produced an action that could not be executed, resulting in re-planning with continued conversation (cf. Sec. IV-C.4).
前两名用粗体并加下划线。对象交互、行进距离和不可执行动作均为所有回合的平均值，包括提前终止的失败回合。不可执行：LLM 产生了无法执行的动作的平均步数，随后会重新规划并继续对话（参见第 IV-C.4 节）。


<img src="https://cdn.noedgeai.com/bo_d8lcn0s91nqc738unag0_5.jpg?x=929&y=763&w=706&h=390&r=0"/>



Fig. 5. Interactive search efficiency curve in simulation. Each point depicts the success rate for a given maximum time budget (x-axis).
图 5. 仿真中的交互式搜索效率曲线。每个点表示给定最大时间预算（x 轴）下的成功率。


Scene Understanding: We compare our door-based room separation algorithm against Hydra [9], which separates a Voronoi graph of places based on dilating obstacles. We evaluate the room segmentation precision and recall as defined by Hughes et al. [9] and the separated Voronoi graph purity at all high-level policy steps to account for robustness throughout exploration. As depicted in Tab. II, we observe greater average precision and recall of MoMa-LLM in terms of dense region segmentation as well as a lower variance across time. The sparsely evaluated graph purity shows that the separated Voronoi graphs of MoMa-LLM cover fewer ground-truth rooms per predicted room than graphs produced by Hydra, which may produce inferior results when facing non-apparent constrictions or object clutter. We found that real-world scenes contained in iGibson regularly feature constant-diameter corridors and narrow passages due to furniture placements, which impede detecting rooms based on geometrical constrictions. This demonstrates that room separation algorithms benefit from semantic cues such as detected doors, door frames, archways, or changing floor materials at room boundaries. We found our policy to be robust to under-segmented rooms even though objects from multiple rooms were, e.g., considered part of a single room. By relying on the camera pose from which an object is observed we reduce the number of wrong object-room assignments through walls. Following the door-wise separation of rooms, our approach however is prone to open room concepts such as combined kitchen and living rooms. For more information and graph depictions, refer to the Supplementary Sec. S.1.
场景理解：我们将基于门的房间分离算法与 Hydra [9] 进行比较，后者通过对障碍物进行膨胀来分离地点的 Voronoi 图。我们按照 Hughes 等人 [9] 的定义，评估房间分割精度和召回率，以及在所有高层策略步骤上的分离 Voronoi 图纯度，以反映探索全过程中的鲁棒性。如表 II 所示，我们观察到 MoMa-LLM 在密集区域分割方面的平均精度和召回率更高，且随时间的方差更低。稀疏评估的图纯度表明，MoMa-LLM 的分离 Voronoi 图中，每个预测房间覆盖的真实房间数少于 Hydra 生成的图；当面对不明显的狭窄处或物体杂乱时，这可能导致较差结果。我们发现，iGibson 中的真实场景常因家具摆放而具有恒定直径的走廊和狭窄通道，这会妨碍基于几何狭窄处来检测房间。这表明，房间分离算法受益于语义线索，例如检测到的门、门框、拱门，或房间边界处地面材料的变化。我们发现，尽管来自多个房间的物体，例如会被视为同一房间的一部分，但我们的策略对房间分割不足仍具有鲁棒性。通过依赖观察某个物体时的相机位姿，我们减少了穿墙误将物体分配到错误房间的情况。然而，在按门分割房间后，我们的方法也容易把厨房和客厅等开放式空间概念合并处理。更多信息和图示请参见补充材料第 S.1 节。


TABLE II
表 II


ENVIRONMENT PARTITIONING THROUGHOUT EXPLORATION
探索过程中的环境划分


<table><tr><td rowspan="2">Approach</td><td colspan="2">Precision</td><td colspan="2">Recall</td><td rowspan="2">Purity $\uparrow$</td></tr><tr><td>$\mu  \uparrow$</td><td>$\sigma  \downarrow$</td><td>$\mu  \uparrow$</td><td>$\sigma  \downarrow$</td></tr><tr><td>Hydra</td><td>0.621</td><td>0.081</td><td>0.943</td><td>0.044</td><td>0.562</td></tr><tr><td>MoMa-LLM</td><td>0.666</td><td>0.064</td><td>0.948</td><td>0.032</td><td>0.615</td></tr></table>
<table><tbody><tr><td rowspan="2">方法</td><td colspan="2">精度</td><td colspan="2">召回率</td><td rowspan="2">纯度 $\uparrow$</td></tr><tr><td>$\mu  \uparrow$</td><td>$\sigma  \downarrow$</td><td>$\mu  \uparrow$</td><td>$\sigma  \downarrow$</td></tr><tr><td>Hydra</td><td>0.621</td><td>0.081</td><td>0.943</td><td>0.044</td><td>0.562</td></tr><tr><td>MoMa-LLM</td><td>0.666</td><td>0.064</td><td>0.948</td><td>0.032</td><td>0.615</td></tr></tbody></table>


Dense room segmentation precision and recall as defined in Hughes et al. [9] in terms of mean and standard deviation throughout exploration. The purity (Supplementary Sec. S.6.1.A) measures the number of ground-truth rooms erroneously captured per predicted room given sparse Voronoi graphs. Evaluated across 10 episodes and all test scenes with $2\mathrm{D}$ grid resolution of ${0.05}\mathrm{\;m}$ to account for thin walls. Best values are written in bold.
稠密房间分割的精确率与召回率（Hughes 等人[9]的定义）在整个探索过程中以均值和标准差表示。纯度（补充材料 Sec. S.6.1.A）衡量：在给定稀疏的 Voronoi 图时，每个预测房间中被错误捕获的真实房间数量。共评估 10 个回合以及所有测试场景，并采用 $2\mathrm{D}$ 的网格分辨率 ${0.05}\mathrm{\;m}$ 以考虑薄墙。最佳值以加粗表示。


Policies: The results and efficiency curves for the search task are shown in Tab. I and Fig. 5. We find that, given appropriate subpolicies, heuristics can complete a significant share of episodes. However, they are not sufficient for an efficient search strategy, resulting in low SPL and AUC-E. Similarly, while HIMOS achieves a high success rate, it is unable to explore efficiently. We found that the RL agent struggled with the much larger action space that resulted from the many more interactable instances in our scenes. ESC in contrast, is able to exploit the co-occurrences to improve over the other baselines. However, given its pair-wise comparisons, it is unable to optimize over longer action sequences. In contrast, MoMa-LLM achieves similar success rates as HIMOS with a much higher search efficiency, both in terms of SPL and AUC-E. We find that the structured prompt representation is essential for this, with the Unstructured LLM performing much worse. We then perform a number of ablations of the language encodings. We find that encoding the frontiers is very important. Removing the history also leads to a, although smaller, drop in performance. Even a coarser representation of the history, consisting of only a list of visited rooms similar to [5], is already beneficial, but slightly worse than the full action history. Lastly, we evaluate the impact of not encoding distances nor nearby objects and also find a small drop in performance.
策略：搜索任务的结果及效率曲线分别见表 I 和图 5。我们发现，在给定合适的子策略时，启发式方法能够完成相当一部分回合。然而，它们不足以形成高效的搜索策略，导致 SPL 和 AUC-E 较低。类似地，尽管 HIMOS 能达到较高的成功率，但仍无法高效探索。我们发现，RL 智能体在我们的场景中由于可交互实例更多而产生了更大的动作空间，从而难以适应。相比之下，ESC 能利用共现关系，从而优于其他基线。但由于其仅进行成对比较，它无法优化更长的动作序列。相比之下，MoMa-LLM 在成功率上与 HIMOS 相当，但搜索效率更高——无论是 SPL 还是 AUC-E。我们发现结构化的提示表示至关重要，而非结构化 LLM 表现明显更差。随后，我们对语言编码进行多项消融。我们发现，对“前沿（frontiers）”的编码非常重要。移除历史也会导致性能下降，尽管幅度较小。即使只采用更粗略的历史表示——仅包含类似于 [5] 的已访问房间列表——也已经有益，但仍略差于完整的动作历史。最后，我们评估不编码距离以及附近物体的影响，同样观察到性能小幅下降。


This picture is fortified by the full efficiency curves in Fig. 5, which show that the MoMa-LLM-based approaches achieve the highest performance for all given time budgets, with only MoMa-LLM w/ Hydra being more efficient for some of the small budgets, but not overall. In contrast, random heuristics achieve very high coverage, resulting in good success rates, but often take very long to find specific objects. Further examining the different models, we find that MoMa-LLM both travel much shorter distances and open fewer objects on average, indicating efficient and target-driven behavior. In contrast, Unstructured LLM produces almost 50% more invalid actions. Qualitatively, we find that MoMa-LLM is robust to various room layouts, such as "combined kitchen and living rooms" that result in large room clusterings and can handle the open-vocabulary room classification well. In contrast, Hydra tends to predict a larger number of small rooms. For reasoning examples, refer to Supplementary Sec. S.6.
图 5 中完整的效率曲线进一步印证了这一点：MoMa-LLM 的方法在所有给定时间预算下都取得了最高表现。仅在部分较小的预算中，带 Hydra 的 MoMa-LLM 更高效，但总体仍不如前者。相反，随机启发式虽然覆盖率很高，因此成功率也不错，但往往需要很长时间才能找到特定物体。进一步分析不同模型，我们发现 MoMa-LLM 平均而言行进距离更短、打开的物体更少，表明其行为高效且以目标为导向。相对而言，非结构化 LLM 的无效动作几乎多 50%。从定性结果来看，我们发现 MoMa-LLM 对各种房间布局都很鲁棒，例如会导致大房间簇的“组合式厨房与客厅”，并能很好地处理开放词汇的房间分类。相对地，Hydra 往往预测更多数量的小房间。推理示例请参见补充材料 Sec. S.6。


TABLE III
表 III


INTERACTIVE OBJECT SEARCH RESULTS IN THE REAL WORLD
真实世界中的交互式物体搜索结果


<table><tr><td>Model</td><td>Success Rate</td><td>Navig Fails</td><td>Manip Fails</td><td>Distance Traveled</td><td>Object Interact.</td></tr><tr><td>ESC-Inter.</td><td>80%</td><td>2</td><td>0</td><td>33.9</td><td>3.5</td></tr><tr><td>MoMa-LLM</td><td>80%</td><td>1</td><td>1</td><td>17.9</td><td>2.2</td></tr></table>
<table><tbody><tr><td>模型</td><td>成功率</td><td>导航失败</td><td>操作失败</td><td>行驶距离</td><td>物体交互</td></tr><tr><td>ESC-交互</td><td>80%</td><td>2</td><td>0</td><td>33.9</td><td>3.5</td></tr><tr><td>MoMa-LLM</td><td>80%</td><td>1</td><td>1</td><td>17.9</td><td>2.2</td></tr></tbody></table>


Dist. travelled is the average distance travelled per episode in meters. Object interactions are the average number of object interactions per episode.
行驶距离是每个回合平均行驶的米数。物体交互是每个回合平均的物体交互次数。


## B. Real-World Experiments
## B. 真实世界实验


We create a real-world apartment, consisting of four rooms: a combined kitchen and dining room, a living room, a long h allway, and a bathroom. We use a Toyota HSR robot, equipped with an RGB-D camera and a ${270}^{ \circ  }$ LiDAR. We replace the navigation policies with the ROS Nav Stack and the manipulation actions with the ${\mathrm{N}}^{2}{\mathrm{M}}^{2}$ manipulation policies [37]. We rely on the same assumptions as in simulation and assume access to localization, accurate semantic perception, and handle detection. We implement this by pre-recording a map with the robot's LiDAR and annotating it with semantic labels. At test time, we create an occupancy map from the robot's RGBD camera and reveal the corresponding part of the semantic map to the agent. The pre-recorded map is also used for localization. To detect handles, we use AR-Markers placed on each object. For details refer to the Supplementary Sec. S.2.
我们构建了一个真实世界公寓，由四个房间组成：一个厨房兼餐厅、一个客厅、一条长走廊和一个浴室。我们使用一台配备 RGB-D 相机和${270}^{ \circ  }$激光雷达的 Toyota HSR 机器人。我们将导航策略替换为 ROS Nav Stack，将操作动作替换为${\mathrm{N}}^{2}{\mathrm{M}}^{2}$操作策略[37]。我们依赖与仿真中相同的假设，并假定可获取定位、准确的语义感知和把手检测。我们通过用机器人的激光雷达预先录制地图，并为其标注语义标签来实现这一点。在测试时，我们从机器人的 RGBD 相机生成占据地图，并向智能体展示语义地图中对应的部分。预录制的地图也用于定位。为检测把手，我们在每个物体上放置 AR 标记。详情参见补充材料第 S.2 节。


We evaluate both MoMa-LLM and the most efficient baseline, ESC, on identical start positions and targets. The results are shown in Tab. III, Fig. 6, and the video. Both methods succeeded in 8/10 episodes, demonstrating the successful transfer of the system to the real world. We find that the Voronoi- and scene graph construction transfer directly to the quite different, unseen layout. Similarly, the system directly transfers to the change in subpolicies, where the mobile manipulation policies ensure a the transition between all subpolicies. The two failures stemmed from irrecoverable failures of the subpolicies, in particular, collisions of the base during navigation or of the arm while opening the door. Comparing the methods, we find confirmation of the simulation results, with MoMa-LLM moving and opening objects more target-driven and efficiently. Furthermore, the agent was able to react to the (unseen) subpolicy failures, such as re-trying to open a drawer when the gripper slipped off the handle.
我们在相同的起始位置和目标上评估了 MoMa-LLM 和最高效的基线 ESC。结果如表 III、图 6 和视频所示。两种方法都在 8/10 个回合中成功，证明该系统已成功迁移到真实世界。我们发现，Voronoi 和场景图的构建可直接迁移到这一截然不同、未见过的布局。同样地，系统也可直接迁移到子策略的变化，其中移动操作策略确保了所有子策略之间的过渡。两次失败都源于子策略无法恢复的故障，尤其是在导航时底座碰撞，或在开门时机械臂碰撞。比较这些方法后，我们确认了仿真结果，即 MoMa-LLM 在移动和打开物体时更具目标导向，也更高效。此外，该智能体还能对（未见过的）子策略失败作出反应，例如在夹爪从把手上滑脱后重试打开抽屉。


## C. Towards General Household Tasks
## C. 朝向通用家务任务


As we move to more abstract and complex tasks, it becomes increasingly difficult to define problem-specific rules or heuristics. In contrast, our approach is readily expandable to a wide range of household and mobile manipulation tasks. Representative of this, we introduce a fuzzy search task. In this task, the robot does not receive a specific object class to find, but rather a fuzzy description, such as "I am hungry. Find me something for breakfast". The full set of queries are shown in Tab. S.4. We find that the agent is capable of finding objects that satisfy respective queries, and correctly reasoning about task completion by calling done(). We further test this capability with three tasks that cannot be solved with the given subpolicies (bottom part of Tab. S.4). For these cases, the agent terminated the episode after finding the relevant objects, reasoning that these objects would now be sufficient for further completion of the tasks. This demonstrates the flexibility of our approach. We leave the extension to arbitrary tasks to future work.
随着我们将任务推进到更抽象、更复杂的场景，问题特定的规则或启发式变得愈发难以定义。相较之下，我们的方法可以轻松扩展到各类家务与移动操作任务。为体现这一点，我们引入了一项模糊搜索任务。在该任务中，机器人并不会接收到要寻找的特定物体类别，而是收到诸如“我饿了。给我找点早餐吃的东西”这种模糊描述。完整的查询集合见表 S.4。我们发现，该智能体能够找到满足各自查询的物体，并通过调用 done() 来对任务完成情况进行正确推理。我们还进一步用三个无法通过给定子策略（表 S.4 下半部分）解决的任务来测试这一能力。在这些情况下，智能体在找到相关物体后终止了该回合，并推理认为这些物体现在已足以完成任务的后续步骤。这表明了我们方法的灵活性。关于扩展到任意任务，我们将留待未来工作。


<img src="https://cdn.noedgeai.com/bo_d8lcn0s91nqc738unag0_7.jpg?x=248&y=165&w=522&h=542&r=0"/>



Fig. 6. We construct a real-world apartment covering four rooms and 54 objects and transfer the model to a Toyota HSR robot.
图 6. 我们搭建了一个覆盖四个房间、包含 54 个物体的真实公寓，并将模型迁移到丰田 HSR 机器人上。


## VI. CONCLUSION
## 第六章 结论


We developed a method to ground language models for high-level reasoning with scalable, dynamic scene graphs and efficient low-level policies for interactive tasks that require combined reasoning about manipulation, navigation and exploration. We demonstrated the importance of extracting structured knowledge for large and unexplored scenes to enable LLMs to reason about efficient search strategies, outperforming fully learned or co-occurrence-based methods. We then transferred our method to a real-world apartment, achieving consistent performance over many episodes. Lastly, we demonstrate the extendability of our approach to abstract tasks, opening the door towards general household tasks.
我们提出了一种方法，用于将语言模型扎根于高层推理：通过可扩展、动态的场景图，以及用于交互任务的高效低层策略，来完成需要同时进行操作、导航和探索的综合推理。我们证明了：从大型且尚未充分探索的场景中提取结构化知识，对于让大语言模型能够推理高效的搜索策略至关重要，其表现优于完全学习或基于共现的方法。随后，我们将该方法迁移到真实的公寓环境中，在多次回合中实现了稳定的性能。最后，我们展示了该方法对抽象任务的可扩展性，为走向通用的家居任务打开了大门。


## REFERENCES
## 参考文献


[1] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, "Voxposer: Composable 3d value maps for robotic manipulation with language models," in Proc. of the Conf. on Rob. Learning, 2023.
[1] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, “Voxposer：用于语言模型驱动的机器人操作的可组合3d值映射”，载于：Rob. Learning会议论文集，2023。


[2] Z. Liu, A. Bahety, and S. Song, "REFLECT: Summarizing robot experiences for failure explanation and correction," in Proc. of the Conf. on Rob. Learning, 2023.
[2] Z. Liu, A. Bahety, and S. Song, “REFLECT：用于故障解释与纠正的机器人经验摘要”，载于：Rob. Learning会议论文集，2023。


[3] B. Li, P. Wu, P. Abbeel, and J. Malik, "Interactive task planning with language models," in 2nd Workshop on Lang. and Robot Learning: Language as Grounding (CoRL), 2023.
[3] B. Li, P. Wu, P. Abbeel, and J. Malik, “带语言模型的交互式任务规划”，载于：语言与机器人学习第二届研讨会：作为基础的语言（CoRL），2023。


[4] K. Lin, C. Agia, T. Migimatsu, M. Pavone, and J. Bohg, "Text2motion: from natural language instructions to feasible plans," Autonomous Robots, vol. 47, no. 8, pp. 1345-1365, 2023.
[4] K. Lin, C. Agia, T. Migimatsu, M. Pavone, and J. Bohg, “Text2motion：从自然语言指令到可行计划”，Autonomous Robots，第47卷，第8期，第1345-1365页，2023。


[5] K. Rana, J. Haviland, S. Garg, J. Abou-Chakra et al., "Sayplan: Grounding large language models using $3\mathrm{\;d}$ scene graphs for scalable task planning," Proc. of the Conf. on Rob. Learning, 2023.
[5] K. Rana, J. Haviland, S. Garg, J. Abou-Chakra 等人，“Sayplan：使用 $3\mathrm{\;d}$ 场景图对大型语言模型进行落地，以实现可扩展的任务规划”，载于：Rob. Learning会议论文集，2023。


[6] C. Agia, K. Jatavallabhula, M. Khodeir, O. Miksik, V. Vineet et al., "Taskography: Evaluating robot task planning over large 3d scene graphs," in Proc. of the Conf. on Rob. Learning, 2022, pp. 46-58.
[6] C. Agia, K. Jatavallabhula, M. Khodeir, O. Miksik, V. Vineet 等人，“Taskography：在大规模3d场景图上评估机器人任务规划”，载于：Rob. Learning会议论文集，2022，第46-58页。


[7] F. Schmalstieg, D. Honerkamp, T. Welschehold, and A. Valada, "Learning hierarchical interactive multi-object search for mobile manipulation," IEEE Robotics and Automation Letters, 2023.
[7] F. Schmalstieg, D. Honerkamp, T. Welschehold, and A. Valada, “面向移动操作的层次化交互式多目标搜索学习”，IEEE Robotics and Automation Letters，2023。


[8] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, "Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation," Robotics: Science and Systems, 2024.
[8] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, “用于语言引导机器人导航的层次化开放词汇3d场景图”，Robotics: Science and Systems，2024。


[9] N. Hughes, Y. Chang, and L. Carlone, "Hydra: A real-time spatial perception system for 3D scene graph construction and optimization," in Robotics: Science and Systems, 2022.
[9] N. Hughes, Y. Chang, and L. Carlone, “Hydra：用于3D场景图构建与优化的实时空间感知系统”，载于：Robotics: Science and Systems，2022。


[10] I. Armeni, Z.-Y. He, A. Zamir, J. Gwak, J. Malik, M. Fischer, and S. Savarese, "3D scene graph: A structure for unified semantics, 3D space, and camera," in Int. Conf. on Computer Vision, 2019.
[10] I. Armeni, Z.-Y. He, A. Zamir, J. Gwak, J. Malik, M. Fischer, and S. Savarese, “3D场景图：统一语义、3D空间与相机的结构”，载于：国际计算机视觉会议（Int. Conf. on Computer Vision），2019。


[11] E. Greve, M. Büchner, N. Vödisch, W. Burgard, and A. Valada, "Collaborative dynamic 3d scene graphs for automated driving," Int. Conf. on Robotics & Automation, 2024.
[11] E. Greve, M. Büchner, N. Vödisch, W. Burgard, and A. Valada, “用于自动驾驶的协作动态3d场景图”，载于：机器人与自动化国际会议（Int. Conf. on Robotics & Automation），2024。


[12] S.-C. Wu, J. Wald, K. Tateno, N. Navab, and F. Tombari, "SceneGraph-Fusion: Incremental 3D scene graph prediction from RGB-D sequences," in Proc. of the IEEE Conf. on Comp. Vision and Pattern Recognition, 2021, pp. 7515-7525.
[12] S.-C. Wu, J. Wald, K. Tateno, N. Navab, and F. Tombari, “SceneGraph-Fusion：基于RGB-D序列的增量3D场景图预测”，载于：IEEE计算机视觉与模式识别会议论文集，2021，第7515-7525页。


[13] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen et al., "Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning," Int. Conf. on Robotics & Automation, 2024.
[13] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen 等人，“Conceptgraphs：用于感知与规划的开放词汇3d场景图”，载于：机器人与自动化国际会议（Int. Conf. on Robotics & Automation），2024。


[14] P. Wu, Y. Mu, B. Wu, Y. Hou, J. Ma, S. Zhang, and C. Liu, "Voronav: Voronoi-based zero-shot object navigation with large language model," arXiv preprint arXiv:2401.02695, 2024.
[14] P. Wu, Y. Mu, B. Wu, Y. Hou, J. Ma, S. Zhang, and C. Liu, “Voronav：基于Voronoi的零样本大语言模型目标导航”，arXiv预印本，arXiv:2401.02695，2024。


[15] A. Rajvanshi, K. Sikka, X. Lin, B. Lee, H. pang Chiu, and A. Velasquez, "Saynav: Grounding large language models for dynamic planning to navigation in new environments," in Int. Conf. on Automated Planning and Scheduling, 2024.
[15] A. Rajvanshi, K. Sikka, X. Lin, B. Lee, H. pang Chiu, and A. Velasquez, “Saynav：用于动态规划到新环境导航的基于大型语言模型的落地”，载于：自动规划与调度国际会议（Int. Conf. on Automated Planning and Scheduling），2024。


[16] Z. Ni, X.-X. Deng, C. Tai, X.-Y. Zhu, X. Wu, Y.-J. Liu, and L. Zeng, "Grid: Scene-graph-based instruction-driven robotic task planning," arXiv preprint arXiv:2309.07726, 2023.
[16] Z. Ni, X.-X. Deng, C. Tai, X.-Y. Zhu, X. Wu, Y.-J. Liu, 和 L. Zeng, “Grid：基于场景图的指令驱动机器人任务规划，”arXiv 预印本 arXiv:2309.07726，2023年。


[17] G. Chalvatzaki, A. Younes, D. Nandha, A. T. Le, L. F. Ribeiro, and I. Gurevych, "Learning to reason over scene graphs: a case study of finetuning gpt-2 into a robot language model for grounded task planning," Frontiers in Robotics and AI, vol. 10, 2023.
[17] G. Chalvatzaki, A. Younes, D. Nandha, A. T. Le, L. F. Ribeiro, 和 I. Gurevych, “学会在场景图上推理：通过微调 gpt-2 为机器人语言模型来实现具身任务规划的案例研究，”《机器人与人工智能前沿》，第10卷，2023年。


[18] J. Gu, E. Stefani, Q. Wu, J. Thomason, and X. Wang, "Vision-and-language navigation: A survey of tasks, methods, and future directions," in Proc. of the Assoc. for Computational Linguistics, 2022, pp. 7606-7623.
[18] J. Gu, E. Stefani, Q. Wu, J. Thomason, 和 X. Wang, “视觉与语言导航：任务、方法与未来方向综述，”载于《计算语言学协会会议论文集》，2022年，第7606-7623页。


[19] C. H. Song, J. Wu, C. Washington, B. M. Sadler, W.-L. Chao, and Y. Su, "Llm-planner: Few-shot grounded planning for embodied agents with large language models," in Int. Conf. on Computer Vision, 2023, pp. 2998-3009.
[19] C. H. Song, J. Wu, C. Washington, B. M. Sadler, W.-L. Chao, 和 Y. Su, “Llm-planner：面向具身智能体的大语言模型的少样本有依据规划，”载于《国际计算机视觉会议》，2023年，第2998-3009页。


[20] B. Ichter, A. Brohan, Y. Chebotar, C. Finn, K. Hausman et al., "Do as i can, not as i say: Grounding language in robotic affordances," in Proc. of the Conf. on Rob. Learning, 2022.
[20] B. Ichter, A. Brohan, Y. Chebotar, C. Finn, K. Hausman 等, “做你能做到的，而不是你说的：在机器人可操作性中为语言落地，”载于《机器人学习会议论文集》，2022年。


[21] B. Yamauchi, "A frontier-based approach for autonomous exploration," in Proc. of the IEEE Int. Symp. on Comput. Intell. in Rob. and Aut., 1997.
[21] B. Yamauchi, “一种基于前沿的自主探索方法，”载于《IEEE计算智能机器人与自动化国际研讨会会议论文集》，1997年。


[22] D. S. Chaplot, D. Gandhi, S. Gupta, A. Gupta, and R. Salakhutdinov, "Learning to explore using active neural slam," in Int. Conf. on Learn. Repr., 2020.
[22] D. S. Chaplot, D. Gandhi, S. Gupta, A. Gupta, 和 R. Salakhutdinov, “通过主动神经 slam 学习探索，”载于《学习表征国际会议》，2020年。


[23] A. Younes, D. Honerkamp, T. Welschehold, and A. Valada, "Catch me if you hear me: Audio-visual navigation in complex unmapped environments with moving sounds," IEEE Robotics and Automation Letters, vol. 8, no. 2, pp. 928-935, 2023.
[23] A. Younes, D. Honerkamp, T. Welschehold, 和 A. Valada, “如果你能听见我就抓住我：在复杂的未建图环境中利用移动声源进行音视导航，”《IEEE 机器人与自动化快报》，第8卷，第2期，第928-935页，2023年。


[24] M. Lingelbach, C. Li, M. Hwang, A. Kurenkov, A. Lou, R. Martín-Martín, R. Zhang, L. Fei-Fei, and J. Wu, "Task-driven graph attention for hierarchical relational object navigation," in Int. Conf. on Robotics & Automation. IEEE, 2023, pp. 886-893.
[24] M. Lingelbach, C. Li, M. Hwang, A. Kurenkov, A. Lou, R. Martín-Martín, R. Zhang, L. Fei-Fei, 和 J. Wu, “面向任务的图注意力用于分层关系对象导航，”载于《机器人与自动化国际会议（IEEE）》，2023年，第886-893页。


[25] A. Kurenkov, M. Lingelbach, T. Agarwal, E. Jin, C. Li, R. Zhang et al., "Modeling dynamic environments with scene graph memory," in Int. Conf. on Mach. Learning, 2023, pp. 17976-17993.
[25] A. Kurenkov, M. Lingelbach, T. Agarwal, E. Jin, C. Li, R. Zhang 等, “用场景图记忆对动态环境建模，”载于《机器学习国际会议》，2023年，第17976-17993页。


[26] Z. Ying, X. Yuan, B. Yang, Y. Song, Q. Xu, F. Zhou, and W. Sheng, "Rp-sg: Relation prediction in 3d scene graphs for unobserved objects localization," IEEE Robotics and Automation Letters, 2023.
[26] Z. Ying, X. Yuan, B. Yang, Y. Song, Q. Xu, F. Zhou, 和 W. Sheng, “Rp-sg：面向未观测物体定位的三维场景图关系预测，”《IEEE机器人与自动化快报》，2023年。


[27] K. Zhou, K. Zheng, C. Pryor, Y. Shen, H. Jin, L. Getoor, and X. E. Wang, "Esc: Exploration with soft commonsense constraints for zero-shot object navigation," Int. Conf. on Mach. Learning, 2023.
[27] K. Zhou, K. Zheng, C. Pryor, Y. Shen, H. Jin, L. Getoor, 和 X. E. Wang, “Esc：用于零样本对象导航的软常识约束探索，”载于《机器学习国际会议》，2023年。


[28] J. Chen, G. Li, S. Kumar, B. Ghanem, and F. Yu, "How to not train your dragon: Training-free embodied object goal navigation with semantic frontiers," Robotics: Science and Systems, 2023.
[28] J. Chen, G. Li, S. Kumar, B. Ghanem, 和 F. Yu, “如何不训练你的龙：使用语义前沿进行免训练的具身对象目标导航，”《机器人学：科学与系统》，2023年。


[29] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, "Vlfm: Vision-language frontier maps for zero-shot semantic navigation," Int. Conf. on Robotics & Automation, 2024.
[29] N. Yokoyama, S. Ha, D. Batra, J. Wang, 和 B. Bucher, “Vlfm：面向零样本语义导航的视觉-语言前沿地图，”载于《机器人与自动化国际会议》，2024年。


[30] S. K. Ramakrishnan, D. S. Chaplot, Z. Al-Halah, J. Malik, and K. Grauman, "Poni: Potential functions for objectgoal navigation with interaction-free learning," in Proc. of the IEEE Conf. on Comp. Vision and Pattern Recognition, 2022, pp. 18890-18900.
[30] S. K. Ramakrishnan, D. S. Chaplot, Z. Al-Halah, J. Malik, 和 K. Grauman, “Poni：用于无交互学习的对象目标导航的潜力函数，”载于《IEEE计算机视觉与模式识别会议论文集》，2022年，第18890-18900页。


[31] F. Schmalstieg, D. Honerkamp, T. Welschehold, and A. Valada, "Learning long-horizon robot exploration strategies for multi-object search in continuous action spaces," in Robotics Research, 2022, pp. 52-66.
[31] F. Schmalstieg, D. Honerkamp, T. Welschehold, and A. Valada, “学习面向长时间范围的机器人探索策略：在连续动作空间中进行多目标搜索”，载于《Robotics Research》，2022年，第52-66页。


[32] C. Li, F. Xia, R. Martín-Martín et al., "igibson 2.0: Object-centric simulation for robot learning of everyday household tasks," in Proc. of the Conf. on Rob. Learning, vol. 164, 08-11 Nov 2022, pp. 455-465.
[32] C. Li, F. Xia, R. Martín-Martín 等， “igibson 2.0：面向日常家务任务的机器人学习的以对象为中心的仿真”，载于机器人学习会议论文集，第164卷，2022年11月8-11日，第455-465页。


[33] W. Chen, S. Hu, R. Talak, and L. Carlone, "Leveraging large language models for robot 3d scene understanding," arXiv preprint arXiv:2209.05629, 2022.
[33] W. Chen, S. Hu, R. Talak, and L. Carlone, “利用大型语言模型进行机器人三维场景理解”，arXiv预印本 arXiv:2209.05629，2022年。


[34] OpenAI, "Gpt-4 technical report," arXiv, pp. 2303-08774, 2023.
[34] OpenAI, “Gpt-4 技术报告”，arXiv，第2303-08774页，2023年。


[35] P. He, J. Gao, and W. Chen, "Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing," in Int. Conf. on Learn. Repr., 2022.
[35] P. He, J. Gao, and W. Chen, “Debertav3：通过采用类似 electra 的预训练，并使用梯度解耦的嵌入共享来改进 deberta”，载于国际学习表征会议，2022年。


[36] P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik et al., "On evaluation of embodied navigation agents," arXiv preprint arXiv:1807.06757, 2018.
[36] P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik 等， “关于具身导航智能体的评估”，arXiv预印本 arXiv:1807.06757，2018年。


[37] D. Honerkamp, T. Welschehold, and A. Valada, "N2m2: Learning navigation for arbitrary mobile manipulation motions in unseen and dynamic environments," IEEE Transactions on Robotics, 2023.
[37] D. Honerkamp, T. Welschehold, and A. Valada, “N2m2：在未见且动态的环境中学习任意移动操作运动的导航”，IEEE Robotics Transactions，2023年。


# Language-Grounded Dynamic Scene Graphs for Interactive Object Search with Mobile Manipulation
#用于移动操作的交互式目标搜索的语言引导动态场景图


- Supplementary Material -
-补充材料-


Daniel Honerkamp ${}^{1 * }$ , Martin Büchner ${}^{1 * }$ , Fabien Despinoy ${}^{2}$ , Tim Welschehold ${}^{1}$ , Abhinav Valada ${}^{1}$
Daniel Honerkamp ${}^{1 * }$ , Martin Büchner ${}^{1 * }$ , Fabien Despinoy ${}^{2}$ , Tim Welschehold ${}^{1}$ , Abhinav Valada ${}^{1}$


In this supplementary material, we provide additional details on the simulation and real-world environments, the subpolicies, and the baselines. Moreover, we provide additional results from the experiments. Further examples of real-world experiments are also demonstrated in the video material.
在这份补充材料中，我们提供了关于仿真和真实环境、子策略以及基线的更多细节。此外，我们还提供了实验的更多结果。视频材料中也展示了真实世界实验的更多示例。


### S.1. SIMULATION ENVIRONMENT
### S.1. 仿真环境


## A. Modifications
## A. 修改


We make the following modifications to the iGibson scenes:
我们对 iGibson 场景做以下修改：


- We close all exterior doors and filter them out of the scene graph as they lead out to empty space.
- 我们关闭所有室外门，并将其从场景图中筛除，因为它们会通向空旷空间。


- We scale the size of the Fetch robot by a factor of 0.85 as otherwise it is too large to navigate a significant number of areas.
- 我们将 Fetch 机器人的尺寸按 0.85 的比例缩放，否则它太大，无法有效穿行足够多的区域。


- We remove six doors that block the robot's pathway when opening (door_52 in Pomaria_0_int, door_75 in scene Beechwood_1_int and door_106 in Ihlen_1_int, door_90 in Merom_1_int, door_138 and door_139 in Wainscott_0_int).
- 我们移除六扇在打开时会阻塞机器人的路径的门（Pomaria_0_int 中的 door_52、场景 Beechwood_1_int 中的 door_75、Ihlen_1_int 中的 door_106、Merom_1_int 中的 door_90、Wainscott_0_int 中的 door_138 和 door_139）。


- We rename four object categories, with misleading asset names, such as renaming breakfast_table (which can be found e.g. in bedrooms) to table.
- 我们将四类物体类别重命名为更合理的名称，并修正带有误导性的资源命名，例如将 breakfast_table（例如可在卧室中找到）重命名为 table。


## B. Task Generation
## B. 任务生成


We first align the rooms and objects with the prior distribution. We manually match room labels and use cosine similarities of object name embeddings to match objects. We embed the object names from the scene and from the prior distribution with SBERT [1]. We then define a match as a cosine similarity $\leq  {0.7}$ and being in the top 50 matched categories. As ${P}^{\text{ prior }}$ only contains two "inside" relationships, we enrich these relationships by assuming that all objects that can be found on top of an object and that fit in size, can also be found inside it and vice-versa. This results in an extended distribution ${P}^{\text{ prior,ext }}$ . We then procedurally enrich each episode by drawing additional objects from the set of admissible room-object-relations for all existing furniture in the iGibson scenes. We keep drawing until the physical placement of a relation succeeds or the distribution is exhausted. Given a valid scene instantiation, we draw a target category $g \sim  U$ (scene) from all categories in the scene. We reject targets as infeasible if no target instance is reachable from the agent's random start position. We also reject all episodes in which the target object is immediately visible.
我们先让房间与物体与先验分布对齐。我们手动匹配房间标签，并使用物体名称嵌入的余弦相似度来匹配物体。我们用 SBERT [1] 将场景中的物体名称与先验分布中的物体名称嵌入。然后，我们将余弦相似度 $\leq  {0.7}$ 且位于前 50 个匹配类别中的项定义为匹配。由于 ${P}^{\text{ prior }}$ 只包含两个“inside”关系，我们通过如下方式扩充这些关系：假设所有能够在某个物体上方找到、且在尺寸上适配的物体也能在其内部找到，反之亦然。这会得到一个扩展分布 ${P}^{\text{ prior,ext }}$。随后，我们通过从 iGibson 场景中已存在的所有家具所对应的允许房间-物体-关系集合中，程序性地为每个 episode 追加抽取物体来丰富内容。我们持续抽取，直到某个关系的物理摆放成功，或先验分布被抽尽。对于有效的场景实例化，我们从场景中的所有类别里抽取一个目标类别 $g \sim  U$（场景）。如果无法从智能体的随机起始位置到达任何目标实例，我们就判定目标不可行并予以拒绝。我们也会拒绝那些目标物体一开始就立即可见的所有 episode。


## C. Perception
## C. 感知


The robot in simulation is equipped with a differential drive and an RGB-D camera with a resolution of ${256} \times  {256}$ pixels and a vertical field of view of ${120}^{ \circ  }$ . As the focus of this work is on decision making, we abstract from imperfect perception and assume access to ground truth instances and semantic segmentation from the simulator. For a realistic detection range, we restrict all sensors (depth, semantics) to a maximum range of $5\mathrm{\;m}$ and set a threshold of 50 segmented pixels before an object is considered detected - except for objects with a volume below ${0.01}{\mathrm{\;m}}^{3}$ . We furthermore assume accurate detection of whether an articulated object is open or closed and assume that objects within receptacles are detected after opening the object. We construct all maps at a resolution of ${0.075}\mathrm{\;m}$ and detect the floor and carpet categories as free space.
仿真中的机器人配备差速驱动和一台分辨率为${256} \times  {256}$像素、垂直视场角为${120}^{ \circ  }$的RGB-D相机。由于本文重点在于决策，我们对不完美感知进行了抽象处理，假设可直接从模拟器获得真实标注实例和语义分割。为获得更真实的检测范围，我们将所有传感器（深度、语义）的最大范围限制为$5\mathrm{\;m}$，并设置50个分割像素的阈值，只有超过该阈值时才将物体视为已检测到——体积低于${0.01}{\mathrm{\;m}}^{3}$的物体除外。我们还假设能够准确检测一个可动对象是打开还是关闭，并假设容器中的物体会在打开该对象后被检测到。我们以${0.075}\mathrm{\;m}$的分辨率构建所有地图，并将地板和地毯类别视为可通行空间。


## D. Execution
## D. 执行


All methods start by fully turning around in-place to initialize the scene graph.
所有方法都从在原地完全转身开始，以初始化场景图。


Low-level action space: The subpolicies act in a low-level action space consisting of the following actions:
低层动作空间：子策略在由以下动作组成的低层动作空间中执行：


- move forward by ${7.5}\mathrm{\;{cm}}$
- 前进 ${7.5}\mathrm{\;{cm}}$


- turn-left by up to 35 rad
- 左转，最大 35 弧度


- turn-right by up to 35 rad
- 右转，最大 35 弧度


- open articulated object
- 打开可关节物体


- close articulated object
- 关闭可关节物体


- done: end the episode and evaluate the success
- 完成：结束该回合并评估成功与否


Navigation: To reduce run-time, navigation actions are executed without physical simulation, but rather by directly setting the state of the robot sequentially to each waypoint the resulting path from the ${\mathrm{A}}^{ * }$ -planner (matching the map resolution of 0.075 m), collecting the observations along the whole path. If no complete path in free space exists, the navigation policy will consider unexplored areas as traversable and replan with newly revealed free space. It will return 'failed' if it cannot find a path or reaches too many replanning attempts.
导航：为降低运行时开销，导航动作不进行物理仿真，而是通过将机器人状态依次直接设置为${\mathrm{A}}^{ * }$-planner得到的每个航路点（匹配地图分辨率 0.075 m），在整个路径上收集观测。如果在自由空间中不存在完整路径，导航策略会将尚未探索的区域视为可通行，并使用新揭示的自由空间进行重新规划。若找不到路径或重新规划尝试次数过多，则返回“failed”。


Object interactions: We follow previous work [2], [7] and execute articulated object manipulations as "magic actions". These actions directly set the joint values of the object to their minimum or maximum. In the calculation of the efficiency curves and AUC-E, we weigh these actions by a time factor of 30 to make their time cost comparable to an execution duration of roughly 30 seconds.
物体交互：我们遵循先前的工作[2]、[7]，将可关节物体操控作为“魔法动作”。这些动作会直接将物体关节值设置为其最小值或最大值。在效率曲线和 AUC-E 的计算中，我们用时间因子 30 对这些动作加权，以使其时间成本与大约 30 秒的执行时长保持可比。


---



*These authors contributed equally.
*这些作者贡献相同。


${}^{1}$ Department of Computer Science,University of Freiburg,Germany.
${}^{1}$ 计算机科学系，弗赖堡大学，德国。


${}^{2}$ Toyota Motor Europe (TME)
${}^{2}$ 丰田汽车欧洲（TME）


Project page: http://moma-llm.cs.uni-freiburg.de
项目主页：http://moma-llm.cs.uni-freiburg.de


---



<img src="https://cdn.noedgeai.com/bo_d8lcn0s91nqc738unag0_10.jpg?x=142&y=158&w=743&h=757&r=0"/>



Fig. S.1. Map of the real-world environment. Different intensities represent different object instances in the environment. Room annotations are for illustration and not used by our model. Object labels denote approximate object positions.
图 S.1. 真实环境的地图。不同的强度表示环境中的不同物体实例。房间标注仅用于说明，并未被我们的模型使用。物体标签表示物体的大致位置。


The go_to_and_open() action is implemented as follows:
go_to_and_open() 动作按如下方式实现：


1) Navigate to the Voronoi node closest to the object
1）导航到离目标物体最近的 Voronoi 节点


2) Navigate to the most central free space in front of the object and turn toward it
2）导航到目标物体前方最居中的空闲空间，并朝其转向


3) Trigger the magic open action
3）触发“magic open（神奇开启）”动作


### S.2. REAL-WORLD ENVIRONMENT
### S.2. 真实世界环境


## A. Map
## A. 地图


Fig. S. 1 shows the map of the real-world environment. Different object instances are indicated in different intensities. The environment covers five different room types and 54 different object categories, including furniture such as tables, chairs, and a coat hanger, as well as small and less common objects such as soap, gamepad, or scotch tape. We set start positions for the robot in the kitchen, living room, and hallway. We ensure the same initial positions and targets for all methods. The target categories given to the agents were shoes, milk (inside fridge), knife (inside kitchen drawer), book (inside living room cabinet), toilet paper, toaster, tea (inside kitchen cabinet), pencil (inside living room drawer), soap and lemon.
图 S.1 展示了真实环境的地图。不同的目标实例以不同的强度标示。该环境包含五种不同的房间类型和 54 类不同的物体类别，包括桌子、椅子和衣帽架等家具，以及肥皂、游戏手柄或透明胶带等小型且较少见的物体。我们为机器人在厨房、客厅和走廊设置了起始位置。我们确保所有方法使用相同的初始位置和目标。提供给代理的目标类别包括：鞋子、牛奶（在冰箱内）、刀具（在厨房抽屉内）、书籍（在客厅柜内）、厕纸、烤面包机、茶（在厨房柜内）、铅笔（在客厅抽屉内）、肥皂和柠檬。


## B. Execution
## B. 执行


Navigation: We use the default manufacturer versions of the ROS NavStack as developed by Toyota for navigation. It uses the robot's LiDAR and depth cameras for dynamic obstacle detection and navigates in a map inflated by ${0.25}\mathrm{\;m}$ .
导航：我们使用由丰田开发的 ROS NavStack 默认厂商版本进行导航。它利用机器人的 LiDAR 和深度相机进行动态障碍物检测，并在由${0.25}\mathrm{\;m}$膨胀的地图中导航。


Mobile manipulation: Articulated object interactions are executed with pretrained ${\mathrm{N}}^{2}{\mathrm{M}}^{2}$ manipulation policies [37]. The policy receives the handle pose, detected through AR-Markers, and uses a local occupancy map constructed from the LiDAR of the robot for obstacle avoidance. For each articulated object, we collect a single demonstration of opening the object. This demonstration consists of a set of poses of the robot's wrist link during the opening motion. These poses are then interpolated with splines to generate an end-effector motion. This agent's aim is to follow this end-effector motion to complete the object interaction. We evaluate whether the motion was successful based on whether the marker on the object changed its position after the execution of the subpolicy. For doors, we do not lock the spring-loaded door latch, as the robot is not strong enough to press the handle down.
移动操作：通过预训练的${\mathrm{N}}^{2}{\mathrm{M}}^{2}$操作策略[37]执行具身物体交互。该策略接收通过 AR-Marker 检测到的把手位姿，并使用由机器人 LiDAR 构建的局部占据地图进行避障。对于每个可操作物体，我们采集一次打开该物体的示范。该示范由机器人腕部连杆在开启动作中的一组位姿组成。随后用样条对这些位姿进行插值，生成末端执行器运动。该智能体的目标是跟随这一路径完成物体交互。我们根据子策略执行后物体上的标记是否改变位置来评估该动作是否成功。对于门，我们不锁住弹簧锁门闩，因为机器人没有足够的力将把手压下。


The only exception to this is the door between the kitchen and living room. We found that the robot was unable to localize itself during the opening motion, as the moving door occupies the overwhelming majority of the robot's LiDAR measurements. As a result, it was not possible to follow the opening demonstration whenever the marker moved out of the robot's view. Instead, we use a simpler pushing motion from the inside and abstain from opening it in the other direction.
唯一的例外是厨房和客厅之间的门。我们发现机器人在开启动作中无法定位自身，因为移动的门占据了机器人 LiDAR 测量的绝大部分。因此，一旦标记移出机器人的视野，就无法继续跟随开门示范。为此，我们改用从室内进行更简单的推门动作，并放弃从另一侧打开它。


### S.3. HIERACHICAL SCENE GRAPH STRUCTURE
### S.3. 分层场景图结构


In order to provide a concise overview a complete picture of the employed scene graph hierarchy we provide a detailed overview of it in Tab. S.1.
为提供简明概览，我们在表 S.1 中对所采用的场景图层级做了详细说明。


In addition, we outline the construction of the navigational Voronoi graph used for navigation and associating objects to regions in the following. The Generalized Voronoi Diagram (GVD) [5] comprises two-equidistant faces that each represent the set of points equidistant to two obstacles ${C}_{i}$ and ${C}_{j}$ . Each point in this set is closer to ${C}_{i}$ and ${C}_{j}$ than any other obstacle:
此外，我们将在下文概述用于导航的导航 Voronoi 图构建方法，并说明如何将物体关联到各个区域。广义 Voronoi 图（GVD）[5] 由两个等距的面构成，每个面都表示一组到两个障碍物 ${C}_{i}$ 和 ${C}_{j}$ 距离相等的点。该集合中的任意一点都比任何其他障碍物更接近 ${C}_{i}$ 和 ${C}_{j}$：


$$
{\mathcal{F}}_{ij} = \left\{  {x \in  {\mathbb{R}}^{m} : 0 \leq  {d}_{i}\left( x\right)  = {d}_{j}\left( x\right) \forall k \neq  i,j,\nabla {d}_{i}\left( x\right)  \neq  {d}_{j}\left( x\right) }\right\}
$$



(1)



The union of all two-equidistant faces generated by the obstacle positions defined by ${\mathcal{B}}_{t}$ is called the 2-Voronoi set ${\mathcal{F}}^{2}$ or the two-dimensional GVD of the space of obstacles ${C}_{i} \in  {\mathcal{B}}_{t}$ :
由 ${\mathcal{B}}_{t}$ 处障碍物位置所生成的所有“两等距面”的并集称为 2-Voronoi 集 ${\mathcal{F}}^{2}$，即障碍物空间 ${C}_{i} \in  {\mathcal{B}}_{t}$ 的二维 GVD：


$$
{\mathcal{F}}^{2} = \mathop{\bigcup }\limits_{{i = 1}}^{{n - 1}}\mathop{\bigcup }\limits_{{j = i + 1}}^{n}{\mathcal{F}}_{ij} \tag{2}
$$



The set of points contained in ${\mathcal{F}}^{2}$ constitutes the initial set of edges of the generalized Voronoi graph (GVG). We extract the corresponding nodes by computing the 3-Voronoi sets, which constitute the joints of the GVD:
包含在 ${\mathcal{F}}^{2}$ 中的点集构成广义 Voronoi 图（GVG）的初始边集。我们通过计算 3-Voronoi 集来提取对应节点，而这些 3-Voronoi 集构成 GVD 的关节：


$$
{\mathcal{F}}^{3} = \underset{i = 1}{\overset{n - 2}{ \cup  }}\underset{j = i + 1}{\overset{n - 1}{ \cup  }}\underset{k = j + 1}{\overset{n}{ \cup  }}{\mathcal{F}}_{ijk} \tag{3}
$$



Given this,the generalized Voronoi graph GVG $= \left\{  {{\mathcal{F}}^{2},{\mathcal{F}}^{3}}\right\}$ undergoes sparsification by eliminating edges of degree 2 to form ${\mathcal{G}}_{\mathcal{V}}$ .
因此，广义 Voronoi 图 GVG $= \left\{  {{\mathcal{F}}^{2},{\mathcal{F}}^{3}}\right\}$ 通过移除度为 2 的边实现稀疏化，以形成 ${\mathcal{G}}_{\mathcal{V}}$。


TABLE S. 1
表 S.1


SCENE GRAPH STRUCTURE
场景图结构


<table><tr><td>Component</td><td>Description</td></tr><tr><td>${\mathcal{G}}_{\mathcal{V}} = \left( {\mathcal{V},\mathcal{E}}\right)$</td><td>Navigational Voronoi graph that undergoes sparsification and covers traversable map areas</td></tr><tr><td>↓</td><td></td></tr><tr><td>${\mathcal{G}}_{\mathcal{V}}^{\mathcal{R}} = \left( {{\mathcal{V}}^{\prime },{\mathcal{E}}^{\prime }}\right)$ with ${\mathcal{V}}^{\prime } = \left\{  {{\mathcal{V}}_{1},{\mathcal{V}}_{2},\ldots ,{\mathcal{V}}_{k}}\right\}$ where ${\mathcal{V}}_{i} \cap  {\mathcal{V}}_{j} = \varnothing$ for all $i,j \in  \{ 1,\ldots ,k\}$ with $i \neq  j$ <br> ↓</td><td>Deletion of edges with sufficient edge integrals computed over ${\rho }_{\mathcal{N}}$ (Eq. 2) results in pair-wise edge-disjoint sets ${\mathcal{V}}_{i}$ with each set covering a room ${r}_{i}$ .</td></tr><tr><td>${\mathcal{G}}_{S} = \left( {{\mathcal{V}}_{O} \cup  {\mathcal{V}}_{\mathcal{R}},{\mathcal{E}}_{OR} \cup  {\mathcal{E}}_{RR}}\right)$</td><td>We define the actual scene graph ${\mathcal{G}}_{S}$ over all object nodes ${\mathcal{V}}_{O}$ and room supernodes ${\mathcal{V}}_{R}$ . Each room ${r}_{i} \in  {\mathcal{V}}_{R}$ holds edges its corresponding Voronoi nodes ${\mathcal{V}}_{i} \in  {\mathcal{V}}^{\prime }$ . Each room ${r}_{i} \in  {\mathcal{V}}_{R}$ (and its Voronoi nodes ${\mathcal{V}}_{i}$ ) undergoes semantic classification (Fig. 3). The edges of ${\mathcal{E}}_{OR}$ connect objects $o \in  {\mathcal{G}}_{S}$ with their associated room ${r}_{i} \in  {\mathcal{V}}_{R}$ via Eq. 3. ${\mathcal{E}}_{RR}$ are the edges connecting neighboring rooms ${r}_{i} \in  {\mathcal{V}}_{R}$ to one another.</td></tr></table>
<table><tbody><tr><td>组件</td><td>描述</td></tr><tr><td>${\mathcal{G}}_{\mathcal{V}} = \left( {\mathcal{V},\mathcal{E}}\right)$</td><td>经历稀疏化并覆盖可通行地图区域的导航 Voronoi 图</td></tr><tr><td>↓</td><td></td></tr><tr><td>在 ${\mathcal{V}}_{i} \cap  {\mathcal{V}}_{j} = \varnothing$ 中对所有 $i,j \in  \{ 1,\ldots ,k\}$ 且 $i \neq  j$ 的 ${\mathcal{G}}_{\mathcal{V}}^{\mathcal{R}} = \left( {{\mathcal{V}}^{\prime },{\mathcal{E}}^{\prime }}\right)$ 与 ${\mathcal{V}}^{\prime } = \left\{  {{\mathcal{V}}_{1},{\mathcal{V}}_{2},\ldots ,{\mathcal{V}}_{k}}\right\}$ <br/> ↓</td><td>对 ${\rho }_{\mathcal{N}}$ 上计算得到的足够边积分（式 2）对应边的删除，会产生两两边不相交的集合 ${\mathcal{V}}_{i}$，且每个集合覆盖一个房间 ${r}_{i}$。</td></tr><tr><td>${\mathcal{G}}_{S} = \left( {{\mathcal{V}}_{O} \cup  {\mathcal{V}}_{\mathcal{R}},{\mathcal{E}}_{OR} \cup  {\mathcal{E}}_{RR}}\right)$</td><td>我们在所有目标节点 ${\mathcal{V}}_{O}$ 与房间超节点 ${\mathcal{V}}_{R}$ 上定义实际场景图 ${\mathcal{G}}_{S}$。每个房间 ${r}_{i} \in  {\mathcal{V}}_{R}$ 包含其对应的 Voronoi 节点 ${\mathcal{V}}_{i} \in  {\mathcal{V}}^{\prime }$ 的边。每个房间 ${r}_{i} \in  {\mathcal{V}}_{R}$（以及其 Voronoi 节点 ${\mathcal{V}}_{i}$）都进行语义分类（图 3）。${\mathcal{E}}_{OR}$ 的边通过式 3 将目标 $o \in  {\mathcal{G}}_{S}$ 与其对应的房间 ${r}_{i} \in  {\mathcal{V}}_{R}$ 相连。${\mathcal{E}}_{RR}$ 是连接相邻房间 ${r}_{i} \in  {\mathcal{V}}_{R}$ 的边。</td></tr></tbody></table>


TABLE S.2
表 S.2


MAPPING OF DISTANCES TO NATURAL LANGUAGE.
距离到自然语言的映射。


<table><tr><td>≤ Distance</td><td>Encoding</td></tr><tr><td>3.0</td><td>very close</td></tr><tr><td>10.0</td><td>near</td></tr><tr><td>20.0</td><td>far</td></tr><tr><td>$\infty$</td><td>distant</td></tr></table>
<table><tbody><tr><td>≤ 距离</td><td>编码</td></tr><tr><td>3.0</td><td>非常近</td></tr><tr><td>10.0</td><td>近</td></tr><tr><td>20.0</td><td>远</td></tr><tr><td>$\infty$</td><td>遥远</td></tr></tbody></table>


### S.4. LANGUAGE ENCODING
### S.4. 语言编码


We encode distance to natural language based on a discrete mapping, following the principle of [17]. We bin the distance to the object, then apply the mapping reported in Tab. S.2. This results in a consistent relative language encoding.
我们基于离散映射将距离编码为自然语言，遵循文献[17]的原则。我们先将到目标物体的距离分箱，再应用表S.2所示的映射。由此得到一致的相对语言编码。


### S.5. BASELINES
### S.5. 基线方法


The Unstructured LLM baseline receives the same instructions and "remember" notes as our approach. The full JSON-formatted prompt of this baseline is depicted in Fig. S.2. We find that the much less structured and longer prompt leads to more frequent invalid actions or hallucinations (cf. Tab. I), such as trying to open objects that do not exist or are already opened. If stuck for repeated steps, this can result in failed episodes. A second source of failures are wrong terminations, in which the LLM calls done() while it has not found the correct object of interest.
非结构化 LLM 基线会接收与我们方法相同的指令和“记忆”笔记。该基线的完整、JSON 格式提示词如图 S.2 所示。我们发现，提示词结构更少且更长会导致无效动作或幻觉更频繁（参见表 I），例如尝试打开并不存在或已被打开的物体。如果在多步操作中反复卡住，这可能会导致失败的回合。第二类失败来源是错误终止：当 LLM 还未找到目标物体时，它却调用了 done()。


### S.6. ADDITIONAL RESULTS
### S.6. 其他结果


## A. Hierarchical Scene Graph
## A. 层次场景图


1) Metrics: In the following, we list the metrics used for evaluating the accuracy of the scene graph.
1) 指标：下面列出用于评估场景图准确性的指标。


Room Segmentation Precision / Recall: In order to compare our method of room segmentation against the approach used by Hydra [9] we make use of the same metrics they evaluated:
房间分割精确率 / 召回率：为将我们的方法与 Hydra [9] 使用的房间分割方法进行比较，我们采用了他们评估时使用的相同指标：


$$
{P}_{t} = \frac{1}{\left| {R}_{e}\right| }\mathop{\sum }\limits_{{{r}_{e} \in  {R}_{e}}}\mathop{\max }\limits_{{{r}_{g} \in  {R}_{g}}}\frac{\left| {r}_{g} \cap  {r}_{e}\right| }{\left| {r}_{e}\right| }, \tag{4}
$$



$$
{R}_{t} = \frac{1}{\left| {R}_{g}\right| }\mathop{\sum }\limits_{{{r}_{g} \in  {R}_{g}}}\mathop{\max }\limits_{{{r}_{e} \in  {R}_{e}}}\frac{\left| {r}_{e} \cap  {r}_{g}\right| }{\left| {r}_{g}\right| }, \tag{5}
$$



where ${R}_{e}$ is the set of estimated rooms and ${R}_{g}$ is the set of ground-truth rooms. The cardinality of a set is given by $\left| \cdot \right|$ . Each room ${r}_{e}$ or ${r}_{g}$ is defined by its set of covered pixels on a 2D grid. In order to reflect instabilities throughout exploring the environment we report the means and standard deviations of the precision and recall, respectively:
其中 ${R}_{e}$ 是估计房间集合，${R}_{g}$ 是真实房间集合。集合的基数记为 $\left| \cdot \right|$。每个房间 ${r}_{e}$ 或 ${r}_{g}$ 由其在二维网格上覆盖的像素集合定义。为反映探索环境过程中的不稳定性，我们分别报告精确率和召回率的均值与标准差：


$$
\bar{P} = \frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{P}_{t}\;{\sigma }_{P} = \sqrt{\frac{\mathop{\sum }\limits_{{t = 1}}^{T}{\left( {P}_{t} - \bar{P}\right) }^{2}}{T}} \tag{6}
$$



$$
\bar{R} = \frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{R}_{t}\;{\sigma }_{R} = \sqrt{\frac{\mathop{\sum }\limits_{{t = 1}}^{T}{\left( {R}_{t} - \bar{R}\right) }^{2}}{T}} \tag{7}
$$



The metrics are evaluated on a dense 2D grid with a resolution of ${0.05}\mathrm{\;m}$ ,which is increased compared to the normal resolution of ${0.075}\mathrm{\;m}$ used in all other evaluations. This is done to account for thin walls contained in iGibson. In order to evaluate the separated Voronoi graphs covering distinct rooms on a dense manifold we employ room-specific wavefronts initialized at each node of the separated Voronoi graph bounded by the extracted obstacles.
这些指标在分辨率为 ${0.05}\mathrm{\;m}$ 的稠密二维网格上评估，相比所有其他评估中使用的正常分辨率 ${0.075}\mathrm{\;m}$ 更高。这样做是为了考虑 iGibson 中包含的薄墙。为了在稠密流形上评估覆盖不同房间的分离 Voronoi 图，我们采用以分离 Voronoi 图中每个节点初始化、并受提取出的障碍物限制的房间特定波前。


Room Segmentation Purity: In addition to the dense evaluation outlined above we also evaluate the purity of each of the generated components of the separated Voronoi graphs. Being a criterion used for measuring clustering quality [8] it penalizes the effect of covering multiple ground truth rooms per classified room.
房间分割纯度：除上述稠密评估外，我们还评估分离 Voronoi 图各生成组件的纯度。作为衡量聚类质量的准则 [8]，它会对每个分类房间覆盖多个真实房间的情况进行惩罚。


$$
\operatorname{purity}\left( {\Omega ,\mathbb{C}}\right)  = \frac{1}{N}\mathop{\sum }\limits_{k}\mathop{\max }\limits_{j}\left| {{\omega }_{k} \cap  {c}_{j}}\right| \tag{8}
$$



where $\Omega  = \left\{  {{\omega }_{1},{\omega }_{2},\ldots ,{\omega }_{K}}\right\}$ is the set of components of ${\mathcal{G}}_{V}^{R}$ and $\mathbb{C} = \left\{  {{c}_{1},{c}_{2},\ldots ,{c}_{J}}\right\}$ is the set of ground-truth rooms. Each ${\omega }_{k}$ holds the predicted room types of all nodes of the respective component. In our case, the purity measures the extent to which a set of Voronoi nodes covers a room that contains a single class. Thus, the graph purity describes the degree of room under-segmentation apparent in the scene and thus measures how well the door-wise Voronoi graph separation performs. Similar to the room segmentation precision and recall we average the purity throughout exploring the environment.
其中 $\Omega  = \left\{  {{\omega }_{1},{\omega }_{2},\ldots ,{\omega }_{K}}\right\}$ 是 ${\mathcal{G}}_{V}^{R}$ 的组件集合，$\mathbb{C} = \left\{  {{c}_{1},{c}_{2},\ldots ,{c}_{J}}\right\}$ 是真实房间集合。每个 ${\omega }_{k}$ 包含该组件内所有节点预测的房间类型。在我们的情形中，纯度衡量一组 Voronoi 节点覆盖仅包含单一类别的房间的程度。因此，图纯度描述了场景中房间欠分割的程度，也即衡量按门分隔的 Voronoi 图划分效果如何。与房间分割精确率和召回率类似，我们在探索环境过程中对纯度取平均。


system: You are a robot in an unexplored house. Your task is to find an oven.
系统：你是一名在一栋未探索房屋中的机器人。你的任务是找到一个烤箱。


1. navigate(room name, object name): navigate to this object in this room.
1. navigate(room name, object name): 导航到该房间中的该物体。


2. go to and open(room name, object name): go to this articulated object, door or container and open it.
2. go to and open(room name, object name): 前往该可动对象、门或容器并将其打开。


3. close(room name, object_name): close this articulated object, door or container.
3. close(room name, object_name): 关闭该可动对象、门或容器。


5. done(): call when the task is completed or if you are unable to take any further actions.
5. done(): 在任务完成时或在无法采取进一步行动时调用。


Output Response Format:
输出响应格式：


Analysis: describe where you could find the objects of interest and what actions you need to execute to get there.
Analysis: 描述你可能在哪里找到感兴趣的物体，以及需要执行哪些动作才能到达那里。


Reasoning: justify why the next action is important to solve the task.
Reasoning: 说明为什么下一步行动对解决任务很重要。


Command: function call
Command: function call


console_table_25, bedroom - window_125, bathroom-1 - mirror_61, living room - pencil box_16020, living room - window_121, bathroom-3 - paper_towel_16045, bedroom - alarm_10007, living
控制台桌_25，卧室 - 窗户_125，浴室-1 - 镜子_61，客厅 - 铅笔盒_16020，客厅 - 窗户_121，浴室-3 - 纸巾_16045，卧室 - 闹钟_10007，客厅


user; Scene Graph; Trades; Iffrontier_points; ((-3.974999999999999, 7.725), CFRONTIER CLASSIFICATION.LEADNG.OUT: 1>), ((2.85, 6.825), <FRONTIER_CLASSIFICATION.LEADNG.OUT: 1>), ((-4.125, 5.325), CFRONTER_CLASSIFICATION.LEADNG_OUT: 1>), ((-4.125, 5.325), <FRONTIER_CLASSIFICATION.LEDATNG_UT: 1>), ((-3.3, 7.74999999999999), <FRONTER_CLASSIFICATION.WITHIN: 6->), ( (0.375, 9.15), <FRONTER_CLASSIFICATION.LEDING_UTT-1>), ((-0.76, 5.1), <FRONTERCLASSTFICATION.WITHIN: 6->), ((-1.5, 6.75), <FRONTTER_CLASSTFICATION.WITHIN: 6->), closed_doors: set(), id: living room, node_type: room\}, \{id: room-4\}, \{frontier_point set(), closed_cost_set(), id: before, non_up_text version (id: room-8), (id: room-2), (id: room-2), (id: room-3), (frontier_points: set(), closed_boors: set(), id: bathroom-1, node_type: room\}, (fictance promise: set(1, closed_bours: set(1), id: other from, node_to_ve; ion), (frontier_points: set(), closed_bours: set(), id: backtronca, node_type: room, tit, rooms+), fromtr-points: set, constants II.(-2.243699999997, 1.95). CPONTER CLASSTFICTION.LEADSIVECTIVEL.(-LADSIVELON.LEADSIVEL.) Closed doors: set(), id: bietherom-3, node tweet; non-1, node users, non-1, node users), from other nodes. strikit their a prom. bedcom node type: sheet. file: console table 10, room before, mode type: sheet. (i): istheir 1011, their better, node type: object, (ii): the last mid table 1916, room, becodon, node type: object), (id: table 1917, room; before, node type: id: tighter lane 18, room; before, before, object, (id: took plant 16, room; bedroom, room; bedroom, room; bedroom, node_type: object); \{ id: console_table_26, room: bedrops, node_type: object; (id: table_1021, room: living room, node_type: object; / id: notebook_1004, room: bedropound. armchair 23 room. Vilvia froom, node -types: object; it did so ra2, room living room, node -types object; (ii) sizes-10246 rooms batterion-2 node-tweets object; its console table 25, room living room, node type, (id: batch combined 2.6 . Iron: Living robot, state: based, node tweets object). (id: floor Jamp2, room: living room, node type, object). (id: coffee). room, node-type: object), (id: bad_49, room: bottom, node-type: object), (it) bottom-cabinst_54, room: bedroom, state: closed, node-time object), (it) cost to object colsed, node two choices, in the code of a node become, node two choices, it is sorted to node before, node two objects, node two choices of node in the code and node to be nodes. room: bathroom-1, hode type: object), (it open 10000, room: bedroom, node type: object), (it coillect0, room: bathroom-1, node type: object), (it) mirror-1, room: both room-1, node type: object bathroom-2, node type: object) \{id: tionite 66, room: bathroom-3, node type: object), \{id: tight2, room: bathroom-3, node type: object), \{id: nodera bottom-2, noderate obriect, (id: panel revel 10045, roman bathrooms., node type: object), (id: mirror_68, room: barhoon-3, node-type: object), (id: picture 69, room: batherrom-3, node-type: object), (id: Trandatter_code. Proc. 2011. The common mode type: object), (id: mirror_1, room. Living noise, proposited, (ii) pattern2, room Living room, noise type: object), (iii: Translations on node (iv): shifted), it can read 7, non-living non node-two object), (ii) picture 7, non-living now, node-type: object), (iii:) mixture now, node-type: object), (iii: node two: objective, this window in room: before, state: closed, node true: object, it divisions in a direct objective, it divisions in a direct closed node. The closed, node type cities, (id: window li21, room living room, state: closed, node trees: direct); (id: window li21, room: bathroom-1, state: closed, node tree object), field: window li23, room: bedroom, state: closed, node-type: object); (ii): window_125, room: bedroom, sate: closed, node-type: object); (iii) window_127, room: itvan row: rate: closed, node-type: object); (ii) alarm 10007, room: bedroom, node type: object\}, \{id: floor_lam_10008, room: bedroom, node_type: object\}, \{id: pencil box_10014, room: bedroom, node_type: object\}, \{id: light_bulb-10016, room: bedroom, node type: object, \{id: plate 10817, room: bedroom, node type: object\}, fid: pencil box 18018, room: 11ving room, node type: object\}, \{id: pencil box 16028, room: 11ving room, node type object) \{id: pencil box 10921, room: living room, node type: object). \{id: document 10024, room: livina room, node type: object\}, \{location: living room, id: robot\}], links: [bathroom-1 shower 62, bathroom-3 - sink 67, bedroom - pot plant 19, living room - coffee table 28, living room - picture 72, bedroom - swivel chair 13, bathroom-2 - soap 16046, bedroom - table 1017, living room - mirror 71, room-1 - bathroom-1, living room - grandfather clock 78, room-4 - bathroom-3, living room - picture 73, bathroom-2 - room-5, bedroom - alarm 18037, bedroom - floor lamp 18008. bathroom-2 - towel rack 65, bedroom - straight chair 9, living room - picture 76, bedroom - room-9, bedroom - carpet 56, living room - carpet 74, bedroom - sofa 57, bedroom - table lamp 14, battroneous mirrors Ga, bedroom table 1105, bedconom window 1171, living room - penalty bus 1001 before - bottom cabinet 55, iving room - robot, bedroom - pencil box before, penetrating cost). Withing room - radius, beforem light bubble logic, bearing that the 2 bi-butterms - a vicible of the observe - intered the bi-bottom - intered light, living non - richieve 8, buttone - not debook 100 horizon - action - beforeom - table lane 1.8, livino room - table 1621, before - - bottom cosine 54, living room - bottom cabinet 26, bathroom - picture 63, bedroom - table 1616, living room - armchair 23, living room - document 10624, beform - window_111, bathroom-1 - visited_ob, bathroom-1 - sink_111, rigorom - room-4, bedroom - table-122, living room - nomi.plug 2, living room - nominal-121, living room - nominal-102, rises. room-1, bedroom - room-2, living room-1 for lamp_27, bathroom-1 window_122, bedroom - masses/ living-History: go to and open(living room, room-1) - invalid argument, go to and open(living room, room-2) - invalid argument, go to and open(living room, room-3) - invalid argument, . go to and open(living room, room-0) - invalid argument, go to and open(other room-2, door) - invalid argument, go to and open(bathroom-2, door) - invalid argument, explore(bedroom) - success explore(living room) - success, explore(living room) - success. bedroom - window_116, bedroom - console_table_10, living room - stor_24, living room - towel_rack_64, bedroom - window_118, battromo-1 - powder_bottle_100-43, room-1 - other room,
用户；场景图；交易；Iffrontier_points；(( -3.974999999999999, 7.725), CFRONTIER CLASSIFICATION.LEADNG.OUT: 1>), ((2.85, 6.825), <FRONTIER_CLASSIFICATION.LEADNG.OUT: 1>), ((-4.125, 5.325), <CFRONTER_CLASSIFICATION.LEADNG_OUT: 1>), ((-4.125, 5.325), <FRONTIER_CLASSIFICATION.LEDATNG_UT: 1>), ((-3.3, 7.74999999999999), <FRONTER_CLASSIFICATION.WITHIN: 6->), ( (0.375, 9.15), <FRONTER_CLASSIFICATION.LEDING_UTT-1>), ((-0.76, 5.1), <FRONTERCLASSTFICATION.WITHIN: 6->), ((-1.5, 6.75), <FRONTTER_CLASSTFICATION.WITHIN: 6->), 关闭的门：set()，id：客厅，node_type：room\}, \{id: room-4\}, \{frontier_point set(), closed_cost_set(), id: 之前，non_up_text version (id: room-8), (id: room-2), (id: room-2), (id: room-3), (frontier_points: set(), closed_boors: set(), id: 浴室-1, node_type: room\}, (fictance promise: set(1, closed_bours: set(1), id: other from, node_to_ve; ion), (frontier_points: set(), closed_bours: set(), id: backtronca, node_type: room, tit, rooms+), fromtr-points: set, constants II.(-2.243699999997, 1.95). CPONTER CLASSTFICTION.LEADSIVECTIVEL.(-LADSIVELON.LEADSIVEL.) 关闭的门：set(), id: bietherom-3, node tweet; non-1, node users, non-1, node users), 来自其他节点。strikit 他们 a prom。bedcom 节点类型：sheet。文件：console table 10, room before, mode type: sheet. (i): istheir 1011, 他们 better, node type: object, (ii): 最后中间表 1916, room, becodon, node type: object), (id: table 1917, room; before, node type: id: tighter lane 18, room; before, before, object, (id: took plant 16, room; bedroom, room; bedroom, room; bedroom, node_type: object); \{ id: console_table_26, room: bedrops, node_type: object; (id: table_1021, room: 客厅, node_type: object; / id: notebook_1004, room: bedropound. armchair 23 room. Vilvia froom, node -types: object; it did so ra2, room living room, node -types object; (ii) sizes-10246 rooms batterion-2 node-tweets object; its console table 25, room living room, node type, (id: batch combined 2.6 . Iron: Living robot, state: based, node tweets object). (id: floor Jamp2, room: living room, node type, object). (id: coffee). room, node-type: object), (id: bad_49, room: bottom, node-type: object), (it) bottom-cabinst_54, room: bedroom, state: closed, node-time object), (it) cost to object colsed, node two choices, in the code of a node become, node two choices, it is sorted to node before, node two objects, node two choices of node in the code and node to be nodes. room: bathroom-1, hode type: object), (it open 10000, room: bedroom, node type: object), (it coillect0, room: bathroom-1, node type: object), (it) mirror-1, room: both room-1, node type: object bathroom-2, node type: object) \{id: tionite 66, room: bathroom-3, node type: object), \{id: tight2, room: bathroom-3, node type: object), \{id: nodera bottom-2, noderate obriect, (id: panel revel 10045, roman bathrooms., node type: object), (id: mirror_68, room: barhoon-3, node-type: object), (id: picture 69, room: batherrom-3, node-type: object), (id: Trandatter_code. Proc. 2011. 通用模式类型：object), (id: mirror_1, room. Living noise, proposited, (ii) pattern2, room Living room, noise type: object), (iii: Node 上的翻译 (iv): shifted), it can read 7, non-living non node-two object), (ii) picture 7, non-living now, node-type: object), (iii:) 混合 now, node-type: object), (iii: node two: objective, this window in room: before, state: closed, node true: object, it divisions in a direct objective, it divisions in a direct closed node. The closed, node type cities, (id: window li21, room living room, state: closed, node trees: direct); (id: window li21, room: bathroom-1, state: closed, node tree object), field: window li23, room: bedroom, state: closed, node-type: object); (ii): window_125, room: bedroom, sate: closed, node-type: object); (iii) window_127, room: itvan row: rate: closed, node-type: object); (ii) alarm 10007, room: bedroom, node type: object\}, \{id: floor_lam_10008, room: bedroom, node_type: object\}, \{id: pencil box_10014, room: bedroom, node_type: object\}, \{id: light_bulb-10016, room: bedroom, node type: object, \{id: plate 10817, room: bedroom, node type: object\}, fid: pencil box 18018, room: 11ving room, node type: object\}, \{id: pencil box 16028, room: 11ving room, node type object) \{id: pencil box 10921, room: living room, node type: object). \{id: document 10024, room: livina room, node type: object\}, \{location: living room, id: robot\}], links: [bathroom-1 shower 62, bathroom-3 - sink 67, bedroom - pot plant 19, living room - coffee table 28, living room - picture 72, bedroom - swivel chair 13, bathroom-2 - soap 16046, bedroom - table 1017, living room - mirror 71, room-1 - bathroom-1, living room - grandfather clock 78, room-4 - bathroom-3, living room - picture 73, bathroom-2 - room-5, bedroom - alarm 18037, bedroom - floor lamp 18008. bathroom-2 - towel rack 65, bedroom - straight chair 9, living room - picture 76, bedroom - room-9, bedroom - carpet 56, living room - carpet 74, bedroom - sofa 57, bedroom - table lamp 14, battroneous mirrors Ga, bedroom table 1105, bedconom window 1171, living room - penalty bus 1001 before - bottom cabinet 55, iving room - robot, bedroom - pencil box before, penetrating cost). Withing room - radius, beforem light bubble logic, bearing that the 2 bi-butterms - a vicible of the observe - intered the bi-bottom - intered light, living non - richieve 8, buttone - not debook 100 horizon - action - beforeom - table lane 1.8, livino room - table 1621, before - - bottom cosine 54, living room - bottom cabinet 26, bathroom - picture 63, bedroom - table 1616, living room - armchair 23, living room - document 10624, beform - window_111, bathroom-1 - visited_ob, bathroom-1 - sink_111, rigorom - room-4, bedroom - table-122, living room - nomi.plug 2, living room - nominal-121, living room - nominal-102, rises. room-1, bedroom - room-2, living room-1 for lamp_27, bathroom-1 window_122, bedroom - masses/ 历史：前往并打开(living room, room-1) - 无效参数，前往并打开(living room, room-2) - 无效参数，前往并打开(living room, room-3) - 无效参数，. 前往并打开(living room, room-0) - 无效参数，前往并打开(other room-2, door) - 无效参数，前往并打开(bathroom-2, door) - 无效参数，探索(bedroom) - 成功 探索(living room) - 成功，探索(living room) - 成功。bedroom - window_116, bedroom - console_table_10, living room - stor_24, living room - towel_rack_64, bedroom - window_118, battromo-1 - powder_bottle_100-43, room-1 - other room,


What is the best next action to complete the task as efficiently as possible? I you don't think that the object can be found in a known room, prioritize opening doors over exploring a room.
为了尽可能高效地完成任务，接下来最好的行动是什么？如果你认为在已知房间里找不到该物体，请优先打开门，而不是探索房间。


Remember:
记住：


1. Respond with a function call
1. 通过函数调用作答


2. You can only use the objects and rooms that you have already found. Object names have to match the description exactly.
2. 只能使用你已经找到的物体和房间。物体名称必须与描述完全一致。


3. You can only explore rooms that are listed as having unexplored space
3. 只能探索那些被列为仍有未探索空间的房间


4. If you have found the object you are looking for, directly call done(). You do not need to navigate to it or interact with it.
4. 如果你已经找到要找的物体，直接调用 done()。你不需要导航到它或与它交互。


5. If some actions failed repeatedly, they may not be possible.
5. 如果某些行动反复失败，可能就无法实现。


assistant: Analysis:
assistant: 分析：


The oven is twoflocally located in the kitchen, which is not explicitly listed in the scene arrach, However, there are rooms that have not been explored yet, as indicated by the results of the liquid and the liquid of the growth. presence of frontier points. Since the oven has not been found in the known rooms and the priority is to one doors over exploring rooms when the object cannot be found in a known room, the next step should be to open any unopened doors that may lead to new rooms, potentially the kitchen.
烤箱在厨房中被两处定位，但该厨房并未在场景图中明确列出。然而，如结果所示液体与液体的生长存在前沿点，说明还有尚未探索的房间。由于烤箱未在已知房间中找到，而当物体在已知房间中找不到时应优先打开门而不是探索房间，因此下一步应打开任何可能通往新房间的未打开的门，可能是厨房。


The scene graph does not list any closed doors in the known rooms, but there are frontier points in the living room and bathroom-3 that suggest there are areas leading out or not fully exolored. Since the living room has already been explored multiple times without finding an oven, and the bathrom is an unlikely location for an oven. the next ”logical step is to explore the frontier point in bathroom-3, which may lead to a new room.
场景图并未在已知房间中列出任何关闭的门，但起居室和浴室-3中都有前沿点，表明可能存在通往外部的区域，或尚未完全探索。由于起居室已经被多次探索但仍未找到烤箱，而浴室对烤箱来说是不太可能的位置，那么下一步“合乎逻辑的操作”是探索浴室-3中的前沿点，它可能会通往一個新房间。


Reasoning:
推理：


Exploring the frontier point in bathroom-3 could reveal a new room, possibly the kitchen where the oven is likely to be found. Since the oven has not been located in any of the
探索浴室-3中的前沿点，可能会发现一个新房间，那里可能会找到烤箱。由于烤箱尚未在任何已知的


currently known
房间中定位到


rooms and there are no closed doors to open, exploring the frontier point is the next best action to potentially discover the kitchen. In
，且没有关闭的门需要打开，探索前沿点就是下一步最好的行动，以便有机会发现厨房。


Command:
指令：


explore(bathroom-3)
explore(bathroom-3)


Fig. S.2. Prompt of the Unstructured-LLM baseline. It receives the same instructions and notes as our approach, but instead of a structured knowledge encoding, it receives a pure JSON encoding of the scene-graph.
图 S.2. 非结构化 LLM 基线的提示。它接收与我们方法相同的指令和备注，但不是结构化的知识编码，而是对场景图的纯 JSON 编码。


2) Room Segmentation and Classification: Our proposed room separation scheme relies on separating Voronoi graphs at door positions. Thus, it is prone to under-segmentation whenever faced with open room layouts or, e.g., missing doors to hallways. We visualize the scene graphs produced by Hydra as well as MoMa-LLM in Fig. S.3. Regarding the approach of Hydra, we observe that maps containing a many corridors with similar diameters produce either a very large number or a very small number of graph components when applying a range of various obstacle dilation values. As Hydra selects the final environment partitioning based on the median of the number of graph components obtained through various obstacle dilation values, it is faced with a bi-modal distribution. This ultimately renders a median-based selection of the segmentation difficult.
2）房间分割与分类：我们提出的房间划分方案依赖于在门的位置分离 Voronoi 图。因此，当面对开放式的房间布局，或例如缺少通往走廊的门时，它容易发生欠分割。我们在图 S.3 中可视化了 Hydra 生成的场景图以及 MoMa-LLM 的结果。就 Hydra 的方法而言，我们观察到：当地图中存在许多直径相似的走廊时，在对不同障碍物膨胀取值进行一系列处理时，图组件的数量要么会变得非常多，要么非常少。由于 Hydra 会基于通过各种障碍物膨胀取值得到的图组件数量的中位数来选择最终的环境划分，因此它会遇到双峰分布。最终，这使得基于中位数的分割选择变得困难。


TABLE S.3 CLOSED-SET ROOM CATEGORY EVALUATION
表 S.3 封闭集房间类别评估


<table><tr><td>Environment</td><td>Open-Set</td><td>Purity ${G}_{\mathcal{V}}^{R}$</td><td>Room Category Accuracy</td></tr><tr><td>iGibson</td><td>✘</td><td>0.615</td><td>0.276</td></tr><tr><td>Real-World</td><td>✓</td><td>-</td><td>0.901</td></tr></table>
<table><tbody><tr><td>环境</td><td>开放集</td><td>纯度 ${G}_{\mathcal{V}}^{R}$</td><td>房间类别准确率</td></tr><tr><td>iGibson</td><td>✘</td><td>0.615</td><td>0.276</td></tr><tr><td>真实世界</td><td>✓</td><td>-</td><td>0.901</td></tr></tbody></table>


In addition to evaluation in Tab. II, we observe an average purity for MoMa-LLM of 0.615 throughout the exploration over 10 episodes across all of the iGibson test scenes. Compared with that, the room segmentation approach introduced by Hydra [9] reaches a purity of 0.562 as listed in Tab. II. While both Hydra and MoMa-LLM tend to under-segment the given room layout, Hydra is specifically affected by narrow constrictions induced by obstacle placements and non-varying door widths, which creates a significant number of isolated graphs covering small corridors that do not represent full ground-truth regions. In general, we infer that long and narrow corridors as well as cluttered scenes are challenging to segment using classical morphological segmentation algorithms (see Fig. S.3).
除了表 II 中的评估外，我们观察到，在 iGibson 所有测试场景中，MoMa-LLM 在 10 个 episode 的整个探索过程中平均纯度为 0.615。相比之下，Hydra [9] 提出的房间分割方法达到 0.562 的纯度，如表 II 所示。尽管 Hydra 和 MoMa-LLM 都倾向于对给定房间布局进行欠分割，但 Hydra 尤其受到障碍物摆放和门宽不变所引起的狭窄收缩的影响，这会产生大量覆盖狭窄走廊的孤立图，而这些并不代表完整的真实区域。总体而言，我们推断，长而狭窄的走廊以及杂乱场景对传统形态学分割算法来说都具有挑战性（见图 S.3）。


<img src="https://cdn.noedgeai.com/bo_d8lcn0s91nqc738unag0_13.jpg?x=142&y=161&w=1506&h=1115&r=0"/>



Fig. S.3. Visualization of the scene graphs produced by Hydra compared to MoMa-LLM. The left column represents the abstracted scene graphs ${\mathcal{G}}_{S}$ while the right column depicts the separated Voronoi graphs ${\mathcal{G}}_{V}^{\mathcal{R}}$ as defined in Tab. S.1. The red crosses represent extracted door positions of MoMa-LLM. The depicted scene is Merom_1_int contained in the iGibson test scenes.
图 S.3. Hydra 与 MoMa-LLM 生成的场景图可视化。左列表示抽象化的场景图 ${\mathcal{G}}_{S}$，右列表示按表 S.1 定义分离出的 Voronoi 图 ${\mathcal{G}}_{V}^{\mathcal{R}}$。红色叉号表示 MoMa-LLM 提取的门位置。所示场景为 iGibson 测试场景中的 Merom_1_int。


We found our downstream policy to be robust to under-segmented rooms even though objects from multiple rooms were, e.g., considered part of a single room. By relying on the camera pose from which an object is observed we reduce the number of false object-room assignments (through walls) to a minimum. We show multiple resulting Voronoi graphs in Fig. S.4.
我们发现，尽管来自多个房间的物体会被例如视为同一房间的一部分，我们的下游策略对房间欠分割仍然具有鲁棒性。通过依赖观察到某个物体时的相机位姿，我们将错误的物体-房间分配（穿墙）降到最低。我们在图 S.4 中展示了多个生成的 Voronoi 图。


In addition to the time-wise averaged room segmentation results reported in Tab. II, we evaluate the semantic room categories predicted by GPT-3.5. in Tab. S.3. Even though MoMa-LLM normally uses open-set room categories, we evaluate the performance on a closed-set of room categories to report reproducible results. To do so, we provided GPT-3.5 with all room categories contained in the iGibson dataset with the task to pick the most suitable given the objects assigned to each particular Voronoi component representing a room. Similar to the segmentation evaluation, we report numbers that are averaged over 10 episodes per scene as well as across all high-level policy steps per episode. We compare the predicted room category of each Voronoi node with the underlying ground-truth room layout maps. Following this, we arrive at an average predicted room category accuracy of 27.6% This number is largely affected by open room layouts as mentioned above.
除了表 II 中按时间平均的房间分割结果外，我们还在表 S.3 中评估了 GPT-3.5 预测的语义房间类别。尽管 MoMa-LLM 通常使用开放集房间类别，我们仍在封闭集房间类别上评估性能，以报告可复现的结果。为此，我们向 GPT-3.5 提供了 iGibson 数据集中的全部房间类别，并要求其根据分配给每个代表房间的特定 Voronoi 组件的物体，选出最合适的类别。与分割评估类似，我们报告的是每个场景中 10 个 episode 的平均值，以及每个 episode 内所有高层策略步骤上的平均值。我们将每个 Voronoi 节点预测的房间类别与其对应的真实房间布局图进行比较。由此，我们得到平均预测房间类别准确率为 27.6%。该数值在很大程度上受到如上所述开放式房间布局的影响。


In addition to the closed-set evaluation, we also evaluated the predicted room categories in an open-set manner on the real-world map shown in Fig. S.1. Across the 10 trials executed in the real world as given in Tab. III, we follow the same evaluation protocol and obtain an average room category accuracy of 90.1% as listed in Tab. S.3. Human-level assessment allows evaluating errors such as entryway instead of hallway positively, which drastically increases the metrics. Nonetheless, the real-world map is less complex in terms of its topology and object distribution compared to the iGibson environments, which feature, e.g., rooms with no objects contained.
除了封闭集评估外，我们还在图 S.1 所示的真实世界地图上以开放集方式评估了预测的房间类别。根据表 III 中在真实世界执行的 10 次试验，我们遵循相同的评估协议，得到表 S.3 所示的平均房间类别准确率为 90.1%。与人工级评估相结合，可以对如 entryway 而非 hallway 之类的错误进行正向评价，这会大幅提高指标。尽管如此，与 iGibson 环境相比，真实世界地图在拓扑和物体分布方面都更简单，例如其中存在没有物体的房间。


<img src="https://cdn.noedgeai.com/bo_d8lcn0s91nqc738unag0_14.jpg?x=140&y=156&w=1522&h=996&r=0"/>



Fig. S.4. Examples of the reasoning of Voronoi graph and identified frontiers (left), scene graph (middle), MoMa-LLM (left). The green environment feedback is not provided to the LLM. Black-white: agent trace, red crosses: closed doors, red rectangles: undiscovered target objects, green stars: next selected navigation / interaction points. The bottom figure illustrates a subpotimal room clustering, which results in a very large room. We find the model to be robust against these clusterings.
图 S.4. Voronoi 图推理与识别到的前沿（左）、场景图（中）、MoMa-LLM（左）的示例。绿色环境反馈不会提供给 LLM。黑白：智能体轨迹，红色叉号：关闭的门，红色矩形：未发现的目标物体，绿色星号：下一步选定的导航/交互点。底部图示展示了次优的房间聚类，其结果是形成了一个非常大的房间。我们发现该模型对这些聚类具有鲁棒性。


## B. Reasoning
## B. 推理


Tab. S. 4 show the full set of fuzzy search queries (top) and infeasible queries (bottom) that were evaluated in Sec. V, together with the language model's reasoning in response to these tasks.
表 S.4 展示了在第 V 节中评估过的全部模糊搜索查询（上）与不可行查询（下），以及语言模型对这些任务的推理回应。


Fig. S. 4 shows additional examples of the scene representations and the model reasoning. It depicts the Voronoi graph and frontiers to unexplored areas (left), the BEV-map together with the constructed scene graph (middle), and the input prompt and answers of the LLM (right). Additional video material with full prompt reasoning is shown on the project website.
图 S.4 给出了更多场景表示示例与模型推理。它展示了用于未探索区域的 Voronoi 图与前沿（左），BEV 地图以及构建得到的场景图（中），以及 LLM 的输入提示与回答（右）。项目网站上还提供了包含完整提示推理的额外视频材料。


## C. Deployment with Full Perception Pipeline
## C. 搭载完整感知流水线的部署


While we focus on evaluation with ground-truth perception in the main work to be able to focus on the representation and decision making components, this section provides details on the requirements of full deployment and guidance to facilitate the deployment.
虽然我们在主文中侧重使用真实值感知进行评估，以便专注于表示和决策组件，但本节提供完整部署的要求细节以及有助于部署的指导。


Our approach requires (i) an RGB-D sensor (ii) localization and mapping (iii) semantic segmentation and (iv) grasp pose detection. Modern RGB-D SLAM approaches such as RTAB-Map [9] can provide (i) and (ii). As our approach supports open-vocabulary representation and reasoning, it enables deployment with any semantic segmentation model, irrespective of its supported classes. This can be closed vocabulary methods [10], [11] such as Mask R-CNN [12] or newer, transformer-based methods [13], [14]. A further range of methods can provide object detections and bounding boxes for a given list of open-vocabulary query categories [15]-[18]. The best model should be based on the use-case, available compute, required object categories, and accuracy. Chen et al. [28] provide a possible reference implementation: they deploy an RTAB-node for localization and mapping. This can be extended to semantic labels through an additional RTAB-node that listens to the semantic masks, to then fuse the resulting point clouds. Finally (iv), for handle detection and grasp-pose detection, [19] achieve accurate results with a retrained YOLO model on a public handle-specific dataset.
我们的方法需要：(i)RGB-D传感器，(ii)定位与建图，(iii)语义分割，以及(iv)抓取位姿检测。像RTAB-Map [9] 这样的现代RGB-D SLAM方法可以提供(i)和(ii)。由于我们的方法支持开放词汇表示和推理，因此可使用任意语义分割模型进行部署，而不受其支持类别限制。这可以是闭集词汇方法 [10]、[11]，如Mask R-CNN [12]，或更新的基于Transformer的方法 [13]、[14]。此外，还有一些方法可以针对给定的开放词汇查询类别列表提供目标检测和边界框 [15]-[18]。最佳模型应基于具体应用场景、可用算力、所需目标类别和精度来选择。Chen等人 [28] 提供了一个可能的参考实现：他们部署了一个RTAB-node用于定位和建图。通过另一个监听语义掩码的RTAB-node，这一实现可扩展到语义标签，随后融合得到的点云。最后，对于(iv)手柄检测和抓取位姿检测，[19]在公开的手柄专项数据集上通过重新训练的YOLO模型取得了准确结果。


TABLE S.4
表S.4


RESULTS ON THE FUZZY SEARCH TASKS IN THE REAL WORLD.
现实世界中的模糊搜索任务结果。


<table><tr><td>Task</td><td>Success</td><td>Reasoning</td></tr><tr><td>I am hungry. Find me something for breakfast.</td><td>✓</td><td>Found cereals, toast, coffee. Then opened fridge to look for milk or eggs. Found milk in the fridge and decided these are sufficient items.</td></tr><tr><td>Find me something to wash my hands.</td><td>✓</td><td>Searched kitchen for a sink. When not finding one, searches for storage room or bathroom until it finds the s in the bathroom.</td></tr><tr><td>I feel sleepy. Find me something to wake up.</td><td>✓</td><td>The teapot is associated with tea that contains caffeine and can help someone wake up.</td></tr><tr><td>Find things to set the kitchen table.</td><td>✓</td><td>Explored kitchen, opening cabinets. Found knife, glasses, plates. Continued to look for forks or spoons. When not finding them, called done. (No forks or spoons existed).</td></tr><tr><td>Find me the book in the living room.</td><td>✓</td><td>Explored until finding living room, then opened cabinet looking for book, found it inside.</td></tr><tr><td>Pour me a glass of milk.</td><td>✓</td><td>Finds milk in fridge, glass on table. Navigates between the two, assuming to transport the last object. Then terminates, reasoning that it has found and navigated to both.</td></tr><tr><td>Turn on the oven.</td><td>✓</td><td>Finds the oven and calls done(): "Turning on oven is implied as completion of the task".</td></tr><tr><td>What's the time?</td><td>✓</td><td>Finds the clock and calls done(): "The clock is the object that will provide the time".</td></tr></table>
<table><tbody><tr><td>任务</td><td>成功</td><td>推理</td></tr><tr><td>我饿了。给我找点早餐吃的。</td><td>✓</td><td>找到了麦片、吐司和咖啡。然后打开冰箱找牛奶或鸡蛋。冰箱里找到了牛奶，于是决定这些就足够了。</td></tr><tr><td>给我找点东西洗手。</td><td>✓</td><td>在厨房里找水槽。找不到时，就继续找储物间或浴室，直到找到浴室里的水槽。</td></tr><tr><td>我有点困。给我找点能提神的东西。</td><td>✓</td><td>这个茶壶对应含有咖啡因的茶，能帮助人清醒。</td></tr><tr><td>给我找点东西摆上餐桌。</td><td>✓</td><td>在厨房里四处查看，打开柜子。找到了刀、玻璃杯和盘子。继续找叉子或勺子。找不到时，就调用 done。因为（根本不存在叉子或勺子）。</td></tr><tr><td>把客厅里的那本书给我找出来。</td><td>✓</td><td>一直探索，找到客厅后，打开柜子找书，最后发现就在里面。</td></tr><tr><td>给我倒一杯牛奶。</td><td>✓</td><td>找到冰箱里的牛奶，看到桌上的杯子。它在两者之间移动，假设是在搬运最后一个物体。然后终止，推理认为它已经找到了并到达了这两个目标。</td></tr><tr><td>把烤箱打开。</td><td>✓</td><td>找到了烤箱并调用 done()：“打开烤箱被视为任务完成。”</td></tr><tr><td>现在几点了？</td><td>✓</td><td>找到了时钟并调用 done()：“时钟这个物体会提供时间。”</td></tr></tbody></table>


Notes: Top: fuzzy search queries. Bottom: infeasible task queries. Success in these tasks is evaluated by human judgment as a reasonable response. The reasoning has been qualitatively paraphrased for brevity.
注：上方为模糊搜索查询。下方为不可行的任务查询。在这些任务中，是否成功由人工判断其回应是否合理。为简洁起见，相关推理已做了定性改写。


TABLE S.5
表 S.5


EVALUATION OF THE AVERAGE RUNTIME PER COMPONENT.
各组件平均运行时间的评估。


<table><tr><td></td><td>Component</td><td>Time per step (s)</td><td>Total time (s)</td><td>Share (%)</td></tr><tr><td rowspan="5">Real-World</td><td>Total</td><td>24.1</td><td>457</td><td>100</td></tr><tr><td>Low-level Execution</td><td>13.2</td><td>250</td><td>55</td></tr><tr><td>Navigation</td><td>7.0</td><td>139</td><td>30</td></tr><tr><td>Manipulation</td><td>13.9</td><td>111</td><td>24</td></tr><tr><td>High-level Reasoning</td><td>10.9</td><td>207</td><td>45</td></tr><tr><td rowspan="4">Simulation</td><td>High-level Reasoning</td><td>11.1</td><td>91.1</td><td>-</td></tr><tr><td>Scene Graph Construction</td><td>3.0</td><td>31.5</td><td>-</td></tr><tr><td>Room Classification</td><td>0.6</td><td>5.0</td><td>-</td></tr><tr><td>LLM Reasoning</td><td>7.5</td><td>64.6</td><td>-</td></tr></table>
<table><tbody><tr><td></td><td>组件</td><td>每步时间（秒）</td><td>总时间（秒）</td><td>占比（%）</td></tr><tr><td rowspan="5">真实世界</td><td>总计</td><td>24.1</td><td>457</td><td>100</td></tr><tr><td>底层执行</td><td>13.2</td><td>250</td><td>55</td></tr><tr><td>导航</td><td>7.0</td><td>139</td><td>30</td></tr><tr><td>操作</td><td>13.9</td><td>111</td><td>24</td></tr><tr><td>高层推理</td><td>10.9</td><td>207</td><td>45</td></tr><tr><td rowspan="4">仿真</td><td>高层推理</td><td>11.1</td><td>91.1</td><td>-</td></tr><tr><td>场景图构建</td><td>3.0</td><td>31.5</td><td>-</td></tr><tr><td>房间分类</td><td>0.6</td><td>5.0</td><td>-</td></tr><tr><td>LLM推理</td><td>7.5</td><td>64.6</td><td>-</td></tr></tbody></table>


Notes: Time per step is the average time for one high-level step or subpolicy call. Total time is the average total time per episode. Real-world numbers are averaged over three episodes of the fuzzy-search experiments. Simulation numbers are averaged over 175 episodes in the iGibson simulator.
备注：每步耗时是一次高级步骤或子策略调用的平均时间。总耗时是每个回合的平均总时间。真实环境数据是在模糊搜索实验的三个回合上取平均。仿真数据是在 iGibson 仿真器的 175 个回合上取平均。


## D. Runtime Analysis
## D. 运行时分析


Tab. S. 5 decomposes the runtime of the system into individual components. While the agent executes more navigation subpolicy calls (this includes driving to objects to open), each manipulation takes longer on average, resulting in similar total times spent in each. We also find that the high-level reasoning takes up a significant fraction of time. We extend this with data from the simulation experiments to be able to break it down into components. In particular the LLM queries for high-level reasoning take up the majority of this time. This demonstrates the importance of current work for compact and fast inference, which is currently receiving a lot of attention [20], as well as the importance of open-source models [21] that can be run locally instead of purely through an API.
表 S. 5 将系统的运行时间分解为各个组件。尽管智能体执行了更多的导航子策略调用（包括为打开物体而进行的移动），但每次操作的平均耗时更长，因而在各部分上花费的总时间相近。我们还发现，高层推理占用了相当大的一部分时间。我们进一步结合来自仿真实验的数据，对其进行组件级分解。特别是，用于高层推理的 LLM 查询占据了这段时间的大部分。这凸显了当前工作对紧凑且快速推理的重要性，该方向目前正受到广泛关注 [20]，以及开源模型的重要性 [21]：它们可以在本地运行，而非仅通过 API。


While the current implementation of the scene graph is not optimized for speed, and the graph is fully recomputed at each time step instead of only updating areas that received new observations, we find its overall time impact to be reasonable, as it is only required at high-level reasoning steps.
尽管当前场景图实现并未针对速度进行优化，并且每个时间步都会从头完整重算，而不是仅更新接收到新观测的区域，我们仍发现其整体时间影响是合理的，因为它只在高层推理步骤中才需要使用。


## E. Extended Future Work Discussion
## E. 扩展未来工作讨论


In this work, we introduce scene graphs as an efficient and scalable representation for high-level, language model based reasoning, by encoding the scene graphs in a structured language representation and the incorporation of knowledge about distances and unexplored areas.
在这项工作中，我们引入场景图作为一种高效且可扩展的表示，用于基于语言模型的高层推理；方法是将场景图编码为结构化的语言表示，并纳入关于距离和未探索区域的知识。


In future work, we aim to relax the assumptions about accurate perception, fully constructing scene graphs from noisy sensor inputs, as e.g. done in Hydra [9] or the direct incorporation of open-vocabulary representations [8]. While we currently encoding distances and spatial arrangements as adjectives and room-object relations, full maps provide much more dense spatial and geometric information. Research for more direct incorporation of such information, e.g. through vision-language models is very promising. Furthermore, more holistic approaches to incorporate spatial and semantic details in room clustering and classifications will be important to address non-standard layouts and designs. Lastly, methods to incorporate more detailed visual feedback for the identification of object states and failure reasons are an important are to increase robustness and success over long tasks.
在未来工作中，我们希望放宽对准确感知的假设，直接从有噪声的传感器输入中完整构建场景图，例如如 Hydra [9] 所做，或直接纳入开放词汇表示 [8]。虽然我们目前将距离和空间布局编码为形容词以及房间-物体关系，但完整地图提供了更为密集的空间与几何信息。研究如何更直接地纳入这类信息，例如通过视觉-语言模型，将非常有前景。此外，将更整体的方法用于在房间聚类和分类中纳入空间与语义细节，对于应对非标准布局和设计也将十分重要。最后，将更细致的视觉反馈纳入物体状态和失败原因识别的方法，是提高长期任务鲁棒性和成功率的重要方向。


## REFERENCES
## 参考文献


[1] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. of the Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Nov. 2019, pp. 3982-3992.
[1] N. Reimers 和 I. Gurevych，"Sentence-BERT：使用 Siamese BERT 网络的句子嵌入，" 见自然语言处理经验方法会议与自然语言处理国际联合会议（EMNLP-IJCNLP）论文集，2019 年 11 月，页 3982-3992。


[2] M. Shridhar, J. Thomason, D. Gordon, Y. Bisk, W. Han, R. Mottaghi, L. Zettlemoyer, and D. Fox, "Alfred: A benchmark for interpreting grounded instructions for everyday tasks," in Proc. of the IEEE Conf. on Comp. Vision and Pattern Recognition, 2020, pp. 10740-10749.
[2] M. Shridhar，J. Thomason，D. Gordon，Y. Bisk，W. Han，R. Mottaghi，L. Zettlemoyer，和 D. Fox，"Alfred：用于解释日常任务中落地指令的基准，" 见 IEEE 计算机视觉与模式识别会议论文集，2020 年，页 10740-10749。


[3] F. Schmalstieg, D. Honerkamp, T. Welschehold, and A. Valada, "Learning hierarchical interactive multi-object search for mobile manipulation," IEEE Robotics and Automation Letters, 2023.
[3] F. Schmalstieg，D. Honerkamp，T. Welschehold，和 A. Valada，"面向移动操作的分层交互式多目标搜索学习，" IEEE Robotics and Automation Letters，2023。


[4] D. Honerkamp, T. Welschehold, and A. Valada, "N2m2: Learning navigation for arbitrary mobile manipulation motions in unseen and dynamic environments," IEEE Transactions on Robotics, 2023.
[4] D. Honerkamp，T. Welschehold，和 A. Valada，"N2m2：在未见和动态环境中学习任意移动操作动作的导航，" IEEE Transactions on Robotics，2023。


[5] H. Choset and J. Burdick, "Sensor based planning. i. the generalized voronoi graph," in Proceedings of 1995 IEEE international conference on robotics and automation, vol. 2. IEEE, 1995, pp. 1649-1655.
[5] H. Choset 和 J. Burdick，"基于传感器的规划。I. 广义 Voronoi 图，" 见 1995 年 IEEE 国际机器人与自动化会议论文集，第 2 卷。IEEE，1995，页 1649-1655。


[6] G. Chalvatzaki, A. Younes, D. Nandha, A. T. Le, L. F. Ribeiro, and I. Gurevych, "Learning to reason over scene graphs: a case study of finetuning gpt-2 into a robot language model for grounded task planning," Frontiers in Robotics and AI, vol. 10, 2023.
[6] G. Chalvatzaki，A. Younes，D. Nandha，A. T. Le，L. F. Ribeiro，和 I. Gurevych，"在场景图上进行推理学习：将 GPT-2 微调为面向落地任务规划的机器人语言模型的案例研究，" Frontiers in Robotics and AI，第 10 卷，2023。


[7] N. Hughes, Y. Chang, and L. Carlone, "Hydra: A real-time spatial perception system for 3D scene graph construction and optimization," in Robotics: Science and Systems, 2022.
[7] N. Hughes，Y. Chang，和 L. Carlone，"Hydra：用于 3D 场景图构建与优化的实时空间感知系统，" 见 Robotics: Science and Systems，2022。


[8] C. D. Manning, P. Raghavan, and H. Schütze, Introduction to information retrieval. Cambridge university press, 2008.
[8] C. D. Manning，P. Raghavan，和 H. Schütze，信息检索导论。剑桥大学出版社，2008。


[9] M. Labbé and F. Michaud, "Rtab-map as an open-source lidar and visual simultaneous localization and mapping library for large-scale and long-term online operation," Journal of field robotics, vol. 36, no. 2, pp. 416-446, 2019.
[9] M. Labbé 和 F. Michaud，"Rtab-map：用于大规模和长期在线运行的开源激光雷达与视觉同步定位与建图库，" Journal of Field Robotics，第 36 卷，第 2 期，页 416-446，2019。


[10] N. Vödisch, K. Petek, W. Burgard, and A. Valada, "Codeps: Online continual learning for depth estimation and panoptic segmentation," in Robotics: Science and Systems, 2023.
[10] N. Vödisch，K. Petek，W. Burgard，和 A. Valada，"Codeps：用于深度估计和全景分割的在线持续学习，" 见 Robotics: Science and Systems，2023。


[11] N. Gosala, K. Petek, P. L. Drews-Jr, W. Burgard, and A. Valada, "Skyeye: Self-supervised bird's-eye-view semantic mapping using monocular frontal view images," in Proc. of the IEEE Conf. on Comp. Vision and Pattern Recognition, 2023, pp. 14901-14910.
[11] N. Gosala，K. Petek，P. L. Drews-Jr，W. Burgard，和 A. Valada，"Skyeye：使用单目前视图图像的自监督鸟瞰语义建图，" 见 IEEE 计算机视觉与模式识别会议论文集，2023，页 14901-14910。


[12] K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask r-cnn," in Proc. of the IEEE Conf. on Comp. Vision and Pattern Recognition, 2017, pp. 2961-2969.
[12] K. He，G. Gkioxari，P. Dollár，和 R. Girshick，"Mask r-cnn，" 见 IEEE 计算机视觉与模式识别会议论文集，2017，页 2961-2969。


[13] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, "Masked-attention mask transformer for universal image segmentation," Proc. of the IEEE Conf. on Comp. Vision and Pattern Recognition, 2022.
[13] B. Cheng，I. Misra，A. G. Schwing，A. Kirillov，和 R. Girdhar，"用于通用图像分割的掩码注意力掩码变换器，" IEEE 计算机视觉与模式识别会议论文集，2022。


[14] M. Käppeler, K. Petek, N. Vödisch, W. Burgard, and A. Valada, "Few-shot panoptic segmentation with foundation models," Int. Conf. on Robotics & Automation, 2023.
[14] M. Käppeler，K. Petek，N. Vödisch，W. Burgard，和 A. Valada，"利用基础模型进行小样本全景分割，" Int. Conf. on Robotics & Automation，2023。


[15] X. Gu, T.-Y. Lin, W. Kuo, and Y. Cui, "Open-vocabulary object detection via vision and language knowledge distillation," in Int. Conf. on Learn. Repr., 2022.
[15] X. Gu，T.-Y. Lin，W. Kuo，和 Y. Cui，"通过视觉与语言知识蒸馏实现开放词汇目标检测，" 见 Int. Conf. on Learn. Repr.，2022。


[16] M. Minderer, A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn, A. Dosovitskiy, A. Mahendran, A. Arnab, M. Dehghani, Z. Shen et al., "Simple open-vocabulary object detection," in Proc. of the Europ. Conf. on Computer Vision. Springer, 2022, pp. 728-755.
[16] M. Minderer, A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn, A. Dosovitskiy, A. Mahendran, A. Arnab, M. Dehghani, Z. Shen et al., "简单的开放词汇目标检测," 载于欧洲计算机视觉会议论文集。Springer, 2022, pp. 728-755.


[17] M. Minderer, A. Gritsenko, and N. Houlsby, "Scaling open-vocabulary object detection," Proc. of the Conf. on Neural Information Processing Systems, vol. 36, 2024.
[17] M. Minderer, A. Gritsenko, and N. Houlsby, "开放词汇目标检测的扩展," 神经信息处理系统会议论文集, vol. 36, 2024.


[18] W. Kuo, Y. Cui, X. Gu, A. Piergiovanni, and A. Angelova, "Open-vocabulary object detection upon frozen vision and language models," in Int. Conf. on Learn. Repr., 2023.
[18] W. Kuo, Y. Cui, X. Gu, A. Piergiovanni, and A. Angelova, "基于冻结视觉和语言模型的开放词汇目标检测," in Int. Conf. on Learn. Repr., 2023.


[19] M. Arduengo, C. Torras, and L. Sentis, "Robust and adaptive door operation with a mobile robot," Intelligent Service Robotics, vol. 14, no. 3, pp. 409-425, 2021.
[19] M. Arduengo, C. Torras, and L. Sentis, "面向移动机器人的稳健自适应开门操作," Intelligent Service Robotics, vol. 14, no. 3, pp. 409-425, 2021.


[20] A. Chavan, R. Magazine, S. Kushwaha, M. Debbah, and D. Gupta, "Faster and lighter llms: A survey on current challenges and way forward," Int. Conf. on Artificial Intelligence, 2024.
[20] A. Chavan, R. Magazine, S. Kushwaha, M. Debbah, and D. Gupta, "更快更轻的LLM：当前挑战与未来方向综述," Int. Conf. on Artificial Intelligence, 2024.


[21] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., "Llama 2: Open foundation and fine-tuned chat models," arXiv preprint arXiv:2307.09288, 2023.
[21] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., "Llama 2：开源基础模型与微调聊天模型," arXiv preprint arXiv:2307.09288, 2023.


[22] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, "Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation," Robotics: Science and Systems, 2024.
[22] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, "面向语言引导机器人导航的分层开放词汇3D场景图," Robotics: Science and Systems, 2024.