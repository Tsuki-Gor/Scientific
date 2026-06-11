# MOMAGRAPH GE: STATE-AWARE UNIFIED SCENE GRAPHS WITH VISION-LANGUAGE MODEL FOR EM- BODIED TASK PLANNING
# MOMAGRAPH GE：用于具身任务规划的具备状态感知的统一场景图与视觉语言模型


Yuanchen Ju*1, Yongyuan Liang*2, Yen-Jen Wang*1, Nandiraju Gireesh, Yuanliang Ju ${}^{3}$ Seungjae Lee ${}^{2}$ , Qiao Gu ${}^{3}$ , Elvis Hsieh ${}^{1}$ , Furong Huang ${}^{7}$ , Koushil Sreenath ${}^{ \dagger  }{}^{1}$
Ju Yuanchen*1, Liang Yongyuan*2, Wang Yen-Jen*1, Nandiraju Gireesh, Ju Yuanliang ${}^{3}$ Lee Seungjae ${}^{2}$，Gu Qiao ${}^{3}$，Hsieh Elvis ${}^{1}$，Huang Furong ${}^{7}$，Sreenath Koushil ${}^{ \dagger  }{}^{1}$


${}^{1}$ University of California,Berkeley ${}^{2}$ University of Maryland,College Park
${}^{1}$ 加州大学伯克利分校 ${}^{2}$ 马里兰大学帕克分校


${}^{3}$ University of Toronto
${}^{3}$ 多伦多大学


Project website: https://HybridRobotics.github.io/MomaGraph/
项目网站：https://HybridRobotics.github.io/MomaGraph/


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_a774eb.jpg"/>



Figure 1: Overview of the MomaGraph. Given a task instruction, MomaGraph constructs a task-specific scene graph that highlights relevant objects and parts along with their spatial-functional relationships, enabling the robot to perform spatial understanding and task planning.
图 1：MomaGraph 的概览。给定任务指令，MomaGraph 构建面向任务的场景图，突出相关的物体及部件，以及它们与空间—功能关系，从而使机器人能够进行空间理解与任务规划。


## ABSTRACT
## 摘要


Mobile manipulators in households must both navigate and manipulate. This requires a compact, semantically rich scene representation that captures where objects are, how they function, and which parts are actionable. Scene graphs are a natural choice, yet prior work often separates spatial and functional relations, treats scenes as static snapshots without object states or temporal updates, and overlooks information most relevant for accomplishing the current task. To address these limitations, we introduce MomaGraph, a unified scene representation for embodied agents that integrates spatial-functional relationships and part-level interactive elements. However, advancing such a representation requires both suitable data and rigorous evaluation, which have been largely missing. We thus contribute MomaGraph-Scenes, the first large-scale dataset of richly annotated, task-driven scene graphs in household environments, along with MomaGraph-Bench, a systematic evaluation suite spanning six reasoning capabilities from high-level planning to fine-grained scene understanding. Built upon this foundation, we further develop MomaGraph-R1, a 7B vision-language model trained with reinforcement learning on MomaGraph-Scenes. MomaGraph-R1 predicts task-oriented scene graphs and serves as a zero-shot task planner under a Graph-then-Plan framework. Extensive experiments demonstrate that our model achieves state-of-the-art results among open-source models, reaching 71.6% accuracy on the benchmark (+11.4% over the best baseline), while generalizing across public benchmarks and transferring effectively to real-robot experiments.
家庭中的移动操作机器人既要导航，又要操作。这需要一种紧凑且语义丰富的场景表示，能够刻画物体的位置、功能以及哪些部分可交互。场景图是自然的选择，但以往工作往往将空间关系与功能关系分开，且把场景视为没有对象状态或时间更新的静态快照，同时忽视了对完成当前任务最相关的信息。为解决这些局限，我们提出 MomaGraph，一种面向具身智能体的统一场景表示，融合空间-功能关系与部件级交互元素。然而，推动此类表示的发展既需要合适的数据，也需要严格评测，而这两者长期匮乏。因此，我们贡献了 MomaGraph-Scenes，这是首个大规模、带丰富标注的、面向任务驱动的家庭环境场景图数据集；同时还提出了 MomaGraph-Bench，一个系统化评测套件，覆盖从高层规划到细粒度场景理解的六种推理能力。在此基础上，我们进一步开发了 MomaGraph-R1，一种在 MomaGraph-Scenes 上通过强化学习训练的 7B 视觉语言模型。MomaGraph-R1 预测面向任务的场景图，并在 Graph-then-Plan 框架下充当零样本任务规划器。大量实验表明，我们的模型在开源模型中取得了最先进的结果，在基准测试上达到 71.6% 的准确率（比最佳基线高 11.4%），同时还能泛化到公开基准，并有效迁移到真实机器人实验中。


---



* Equal Contribution, ${}^{ \dagger  }$ Equal Advising
* 共同贡献，${}^{ \dagger  }$ 共同指导


---



## 1 INTRODUCTION
## 1 引言


When mobile manipulators (Qiu et al., 2024; Honerkamp et al., 2024a; Wu et al., 2023; Zhang et al., 2024a) enter household environments, they face the fundamental challenge of understanding how the environment works, which objects are interactive, and how they can be used. In other words, such robots must not only be capable of navigating through the home, but also manipulate objects within it. While navigation requires modeling the overall spatial layout, manipulation demands capturing more fine-grained object affordances (Ju et al., 2024; Zhu et al., 2025). This naturally raises a central question: What is the most effective, compact, and semantically rich representation of an indoor scene? An intuitive answer is the scene graph, which organizes objects and their relationships in a scene through a graph structure (Armeni et al., 2019; Koch et al., 2024a;b) and has shown great potential in various downstream robotic applications (Rana et al., 2023; Werby et al., 2024; Ekpo et al., 2024).
当移动式机械臂（Qiu et al., 2024; Honerkamp et al., 2024a; Wu et al., 2023; Zhang et al., 2024a）进入家庭环境时，它们面临一个根本性挑战：理解环境如何运作、哪些物体具有交互性，以及如何使用这些物体。换言之，这类机器人不仅需要能在室内导航，还必须能够在其中操纵物体。导航需要对整体空间布局进行建模，而操纵则要求捕捉更细粒度的物体可操作性（Ju et al., 2024; Zhu et al., 2025）。这自然引出一个核心问题：室内场景最有效、最紧凑、且语义最丰富的表示是什么？一个直观答案是场景图：它通过图结构组织场景中的物体及其关系（Armeni et al., 2019; Koch et al., 2024a;b），并在多种下游机器人应用中展现出巨大潜力（Rana et al., 2023; Werby et al., 2024; Ekpo et al., 2024）。


However, existing scene graphs suffer from notable limitations. (1) Their edges typically encode only a single type of relationship, either spatial (Jatavallabhula et al., 2023; Gu et al., 2024a; Loo et al., 2025; Hughes et al., 2024; Rosinol et al., 2021) or functional (Zhang et al., 2025; Dong et al., 2021)(e.g., a remote controlling a TV, a knob adjusting parameters). Relying solely on spatial relationships captures geometric layout but overlooks operability, while relying solely on functional relationships ignores spatial constraints, leading to incomplete and less actionable structures. (2) Most existing methods (Wu et al., 2021; Takmaz et al., 2025; Zhang et al., 2021) are limited to static scenes and struggle to adapt to dynamic environments where object positions change or object states change. (3) They lack task relevance, as they fail to emphasize information directly tied to task execution, thereby reducing efficiency and effectiveness. In contrast, cognitive science research (Uithol et al., 2021; Kondyli et al., 2020; Castanheira et al., 2025) shows that human perception in new environments is both dynamic and task-oriented. Humans do not process all information equally; instead, they flexibly adjust their attention according to the current task. This process is similar to browsing a map on an iPad: people first take a broad view to roughly locate the area of interest, and then zoom in to focus on the specific details needed for the task.
然而，现有场景图存在显著局限。（1）它们的边通常只编码单一类型的关系：要么是空间关系（Jatavallabhula et al., 2023; Gu et al., 2024a; Loo et al., 2025; Hughes et al., 2024; Rosinol et al., 2021），要么是功能关系（Zhang et al., 2025; Dong et al., 2021）（例如，用遥控器控制电视、旋钮用于调节参数）。仅依赖空间关系能刻画几何布局，却忽略可操作性；仅依赖功能关系则忽视空间约束，导致结构不完整且难以用于实际执行。（2）多数现有方法（Wu et al., 2021; Takmaz et al., 2025; Zhang et al., 2021）只能处理静态场景，难以适应动态环境：物体位置会改变，物体状态也会变化。（3）它们缺乏任务相关性：未能突出与任务执行直接相关的信息，从而降低效率与有效性。相比之下，认知科学研究（Uithol et al., 2021; Kondyli et al., 2020; Castanheira et al., 2025）表明，人类在新环境中的感知既是动态的，也是以任务为导向的。人类并不会以同等方式处理所有信息，而是会根据当前任务灵活调整注意力。这个过程类似于在 iPad 上浏览地图：人们先以整体视角大致定位感兴趣区域，然后再放大聚焦任务所需的具体细节。


Motivated by these insights, we emphasize that an ideal scene graph should integrate both spatial and functional relationships, include fine-grained object parts as nodes, making the representation compact, adaptive to dynamic changes, and highly aligned with task instructions, thus providing a more concrete guidance for embodied perception and task planning.
基于这些洞见，我们强调：理想的场景图应当同时整合空间与功能关系，将细粒度的物体部件作为节点，使表示紧凑、能适应动态变化，并与任务指令高度对齐，从而为具身感知与任务规划提供更具体的指导。


To achieve this goal, we present MomaGraph, a novel scene representation specifically designed for embodied agents. It is the first to unify spatial and functional relationships while introducing part-level interactive nodes, providing a more fine-grained, compact, and task-relevant structured representation than existing approaches. To support this representation, we build MomaGraph-Scenes, the first dataset that jointly models spatial and functional relationships with part-level annotations, encompassing multi-view observations, executed actions, and their interactive object parts, and task-aligned scene graph annotations.
为实现这一目标，我们提出 MomaGraph，这是一种专为具身代理设计的新型场景表示。它是首个在引入部件级交互节点的同时统一空间与功能关系的方法，相比现有方案，提供了更细粒度、更紧凑且更具任务相关性的结构化表示。为支撑该表示，我们构建了 MomaGraph-Scenes，这是首个联合建模空间与功能关系并包含部件级标注的数据集，涵盖多视角观测、已执行动作，以及它们所涉及的交互物体部件，并提供与任务对齐的场景图标注。


Building on this foundation, we propose MomaGraph-R1, a 7B vision-language model (VLM) trained with the DAPO (Yu et al., 2025) reinforcement learning algorithm on MomaGraph-Scenes. We design a graph-alignment reward function to guide the model toward constructing accurate, task-oriented scene graphs. MomaGraph-R1 not only predicts scene graphs but also serves as a zero-shot task planner within a Graph-then-Plan framework: the model first generates a structured scene graph as an intermediate representation and then performs task planning based on this graph, significantly improving reasoning effectiveness and interpretability.
在此基础上，我们提出 MomaGraph-R1：一个 7B 视觉-语言模型（VLM），在 MomaGraph-Scenes 上使用 DAPO（Yu et al., 2025）强化学习算法进行训练。我们设计了图对齐奖励函数，引导模型构建准确且面向任务的场景图。MomaGraph-R1 不仅能够预测场景图，还能在 Graph-then-Plan 框架中作为零样本任务规划器使用：模型先生成结构化的场景图作为中间表示，然后基于该图进行任务规划，从而显著提升推理有效性与可解释性。


Despite progress in task-graph planning (Agia et al., 2022), the community still lacks a unified benchmark to systematically evaluate whether and how task-oriented scene graphs improve planning performance. To address this gap, we introduce MomaGraph-Bench, a comprehensive evaluation suite that systematically assesses six key reasoning capabilities, spanning from high-level task planning to fine-grained scene understanding.
尽管在任务图规划方面已有进展（Agia et al., 2022），但社区仍缺少一个统一的基准，用于系统评估任务导向的场景图是否以及如何提升规划性能。为填补这一空白，我们提出 MomaGraph-Bench：一个全面的评测套件，系统评估六项关键推理能力，从高层任务规划到细粒度场景理解，覆盖面广且评估有序。


In summary, our work makes the following key contributions:
总之，我们的工作带来以下关键贡献：


- We propose MomaGraph, the first scene graph representation that jointly models spatial and functional relationships while incorporating part-level interactive nodes, providing a compact, dynamic, and task-aligned knowledge structure for embodied intelligence.
- 我们提出 MomaGraph，这是首个同时联合建模空间与功能关系、并引入部件级交互节点的场景图表示，为具身智能提供紧凑、动态且与任务对齐的知识结构。


- We construct MomaGraph-Scenes, the first large-scale dataset of richly annotated, task-driven scene graphs in household environments, and build MomaGraph-Bench, a unified evaluation suite that systematically measures the impact of scene graph representations on task planning across six core reasoning capabilities.
- 我们构建 MomaGraph-Scenes，这是首个面向家庭环境、具有丰富标注且以任务为驱动的大规模场景图数据集，并搭建 MomaGraph-Bench，一个统一的评估套件，用于系统衡量场景图表征在六项核心推理能力上的任务规划影响。


- We develop MomaGraph-R1, a 7B vision-language model that leverages reinforcement learning to optimize spatial-functional reasoning, enabling zero-shot planning in a Graph-then-Plan paradigm.
- 我们开发 MomaGraph-R1，这是一个 7B 视觉-语言模型，通过强化学习优化空间-功能推理，从而在“先图后规（Graph-then-Plan）”范式下实现零样本规划。


- MomaGraph-R1 surpasses all open-source baseline models, delivering substantial gains across public benchmarks and translating these improvements into strong generalization and effectiveness in real-world robotic experiments.
- MomaGraph-R1 超越所有开源基线模型，在公开基准上带来显著提升，并将这些改进转化为在真实机器人实验中的强泛化能力与有效性。


## 2 RELATED WORKS
## 2 相关工作


Scene Graphs for 3D Indoor Scene Understanding. Scene graphs have emerged as a structured and hierarchical representation in autonomous driving (Zhang et al., 2024b; Greve et al., 2024), robot manipulation (Jiang et al., 2024; Wang et al., 2025; Engelbracht et al., 2024; Jiang et al., 2025; Maggio et al., 2024), and spatial intelligence (Yin et al., 2025; Zemskova & Yudin; Liang et al., 2025a;b) community. They function not only as a means of scene representation but also as a critical bridge between spatial understanding (Cao et al., 2024; Yang et al., 2024; Gu et al., 2024b)and action planning. We focus on the household scenes. However, existing works often focus on a single type of scene graphs. For example, ConceptGraphs (Gu et al., 2024a) primarily model spatial layouts, representing object instances and their geometric relations in an open-vocabulary manner. While spatial graphs (Honerkamp et al., 2024b; Yan et al., 2025) provide useful geometric and semantic grounding, they overlook how objects can functionally interact with one another. Conversely, functional graphs (Li et al., 2021; Dong et al., 2021; Zhang et al., 2025) highlight object affordances and control relations but do not capture the overall spatial structure. Relying solely on either spatial or functional graphs leads to incomplete and less actionable representations. This motivates us to build MomaGraph, which unifies spatial and functional relationships, incorporates part-level nodes, and explicitly models state changes, providing a more comprehensive foundation for embodied task planning.
面向三维室内场景理解的场景图。场景图已成为自动驾驶（Zhang 等，2024b；Greve 等，2024）、机器人操作（Jiang 等，2024；Wang 等，2025；Engelbracht 等，2024；Jiang 等，2025；Maggio 等，2024）以及空间智能（Yin 等，2025；Zemskova & Yudin；Liang 等，2025a;b）社区中一种结构化且层次化的表示形式。它们不仅是场景表征手段，也是空间理解（Cao 等，2024；Yang 等，2024；Gu 等，2024b）与动作规划之间的关键桥梁。我们关注家居场景。然而，现有工作往往只聚焦于单一类型的场景图。例如，ConceptGraphs（Gu 等，2024a）主要建模空间布局，以开放词汇方式表示物体实例及其几何关系。尽管空间图（Honerkamp 等，2024b；Yan 等，2025）提供了有用的几何与语义支撑，但它们忽略了物体如何在功能上相互作用。相反，功能图（Li 等，2021；Dong 等，2021；Zhang 等，2025）强调物体能力与控制关系，却未能刻画整体空间结构。仅依赖空间图或功能图都会导致表征不完整、且可操作性较弱。这促使我们构建 MomaGraph：统一空间与功能关系，加入部件级节点，并显式建模状态变化，从而为具身任务规划提供更全面的基础。


Zero-shot Embodied Task Planning with VLMs. VLMs (OpenAI, 2023; Team et al., 2025; Ahn et al., 2022) have gained significant attention in robotic task planning (Niu et al., 2024; Yue et al., 2024; Lu et al., 2023; Liang et al., 2024; Guo et al., 2024) due to their powerful capabilities in processing multimodal inputs, such as images and language instructions. However, when directly used as task planners, VLMs (Huang et al., 2023; 2024; Ahn et al., 2022; Zheng et al., 2025a; Yang et al., 2025) often suffer from sensitivity to visual noise and shallow semantic grounding; more fundamentally, their lack of structured object-relationship representations necessitates extracting or constructing more effective representations from the same visual inputs to support accurate and reliable high-level planning. Prior approaches such as SayPlan (Ahn et al., 2022) assume access to a reliable 3D scene graph, which is often unrealistic in practice. To overcome this gap, we propose the Graph-then-Plan strategy, which first generates task-specific scene graphs as an intermediate structured representation before high-level planning. By explicitly modeling objects and their relations, this approach significantly improves the accuracy and robustness of task planning. Unlike prior graph-then-plan methods (Dai et al., 2024; Ekpo et al., 2024) that either assume reliable scene graphs or treat graph construction and planning as separate modules, our approach enables a single VLM to jointly generate structured, task-oriented scene graphs and perform high-level planning.
借助 VLM 的零样本具身任务规划。由于其在处理多模态输入（如图像与语言指令）方面的强大能力，VLM（OpenAI，2023；Team 等，2025；Ahn 等，2022）在机器人任务规划中受到了广泛关注（Niu 等，2024；Yue 等，2024；Lu 等，2023；Liang 等，2024；Guo 等，2024）。然而，当将其直接用作任务规划器时，VLM（Huang 等，2023；2024；Ahn 等，2022；Zheng 等，2025a；Yang 等，2025）往往会对视觉噪声敏感，且语义支撑较浅；更根本的是，它们缺少结构化的对象—关系表示，这就需要从相同的视觉输入中提取或构建更有效的表示，以支持准确且可靠的高层规划。现有方法如 SayPlan（Ahn 等，2022）假设能够获得可靠的三维场景图，而在实践中这往往不现实。为弥补这一差距，我们提出“先图后规”（Graph-then-Plan）策略：在高层规划之前，先生成面向任务的场景图，作为中间的结构化表示。通过显式建模物体及其关系，该方法显著提升任务规划的准确性与鲁棒性。与以往的先图后规方法（Dai 等，2024；Ekpo 等，2024）不同，这些方法要么假设可靠的场景图，要么将图构建与规划视为独立模块；而我们的方案使单个 VLM 能够共同生成结构化、面向任务的场景图，并执行高层规划。


## 3 PRELIMINARY FINDINGS AND MOTIVATION EXPERIMENTS
## 3 初步发现与动机实验


To ground our analysis, before the full evaluations we perform two motivating experiments on the MomaGraph-Bench. These comparisons are designed to validate our motivation and design principles, and to reveal why our proposed model is essential for embodied task planning. In this section, we aim to answer the following questions.
为支撑我们的分析，在进行完整评估之前，我们先在 MomaGraph-Bench 上开展两个用于说明动机的实验。这些对比旨在验证我们的动机与设计原则，并揭示为何我们提出的模型对于具身任务规划至关重要。在本节中，我们力求回答以下问题。


### 3.1 ARE VLMS RELIABLE FOR DIRECT PLANNING WITHOUT SCENE GRAPHS?
### 3.1 在没有场景图的情况下，VLMS 能否可靠进行直接规划？


To examine whether direct planning from visual inputs is reliable even for strong closed-source VLMs, we design controlled evaluations on real-world household tasks such as "Open the window" and "Obtain clean boiled water". In these scenarios, models must reason over functional relationships, spatial constraints, and multi-step dependencies (e.g., plug-in before activation, filtration before boiling). As shown in Fig. 2, despite their scale, closed-source VLMs like GPT-5 produce incorrect or incomplete plans, missing prerequisite steps, or misidentifying interaction types. In contrast, our Graph-then-Plan approach, which first generates a task-specific scene graph and then performs planning, consistently produces correct and complete action sequences aligned with ground-truth logic. This demonstrates that incorporating structured scene representations significantly enhances planning accuracy and robustness beyond what direct planning can achieve.
为检验即使是强大的闭源 VLM，从视觉输入直接规划是否仍然可靠，我们在真实家庭任务上设计了受控评测，例如“打开窗户”和“获取干净的沸水”。在这些场景中，模型必须推理功能关系、空间约束以及多步依赖（例如，启动前先插电、煮沸前先过滤）。如图 2 所示，尽管规模更大，像 GPT-5 这样的闭源 VLM 仍会生成错误或不完整的计划，遗漏前提步骤或误判交互类型。相比之下，我们的 Graph-then-Plan 方法先生成任务特定的场景图，再进行规划，能够始终产生与真实逻辑一致、正确且完整的动作序列。这表明，引入结构化场景表示能显著提升规划准确性与鲁棒性，超越直接规划所能达到的水平。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_eadfa5.jpg"/>



Figure 2: Direct planning often fails even for strong closed-source models like GPT-5, producing wrong actions or missing key steps, while our Graph-then-Plan approach with structured scene graphs enables accurate and complete task sequences aligned with ground truth.
图 2：即使是像 GPT-5 这样的强大闭源模型，直接规划也常常失败，生成错误动作或遗漏关键步骤；而我们结合结构化场景图的 Graph-then-Plan 方法能够生成与真实情况一致、准确且完整的任务序列。


## Preliminary Findings 1
## 初步发现 1


- In contrast to directly relying on vision-language models for task planning from raw scene images, our Graph-then-Plan strategy-which incorporates task-oriented scene graph generation as an intermediate structured representation prior to high-level planning, substantially improves both the accuracy and robustness of task planning.
- 与直接依赖视觉语言模型从原始场景图像进行任务规划不同，我们的Graph-then-Plan策略——将面向任务的场景图生成作为高层规划前的中间结构化表示——显著提升了任务规划的准确性和鲁棒性。


### 3.2 ARE SINGLE-RELATIONSHIP GRAPHS ADEQUATE FOR EMBODIED AGENTS?
### 3.2 单一关系图足以满足具身智能体吗？


To ensure a fair comparison, we retrain our model using the same graph structure as in MomaGraph, but constrain the edge types to encode only a single kind of relation—either spatial or functional. This setup allows us to isolate the effect of relation types while keeping the graph topology consistent, thereby directly examining whether single-relation representations are sufficient for task planning. To ensure this finding generalizes beyond one specific architecture, we evaluate this comparison across different base models using the same dataset and experimental configurations. As demonstrated in Table 1, both MomaGraph-R1(trained from Qwen-2.5-VL-7B) and LLaVA-Onevision consistently show superior performance with unified spatial-functional scene graphs compared to single-relationship variants, supporting our hypothesis that integrated representations are essential for effective embodied task planning. Detailed training methodology is described in the Sec. 4.2.
为确保公平比较，我们使用与 MomaGraph 中相同的图结构重新训练模型，但将边类型限制为仅编码一种关系——空间关系或功能关系。这样可以在保持图拓扑一致的同时，隔离关系类型的影响，从而直接检验单一关系表示是否足以用于任务规划。为确保这一发现不局限于某一特定架构，我们在相同数据集和实验配置下，使用不同基础模型进行比较。如表 1 所示，由 Qwen-2.5-VL-7B 训练得到的 MomaGraph-R1 和 LLaVA-Onevision，相较于单一关系变体，均在统一的空间-功能场景图上持续表现出更优性能，支持我们的假设：集成表示对有效的具身任务规划至关重要。详细训练方法见第 4.2 节。


Table 1: Comparison between MomaGraph-R1and LLaVA variants across task tiers.
表 1：MomaGraph-R1 和 LLaVA 各变体在不同任务层级上的比较。


<table><tr><td>Models</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>Overall</td><td>Models</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>Overall</td></tr><tr><td>& MomaGraph-R1 (Spatial-only)</td><td>69.1</td><td>67.0</td><td>58.4</td><td>45.4</td><td>59.9</td><td>& LLaVA-Onevision (Spatial-only)</td><td>63.4</td><td>56.7</td><td>59.7</td><td>36.3</td><td>54.0</td></tr><tr><td>& MomaGraph-R1 (Functional-only)</td><td>71.4</td><td>65.8</td><td>63.6</td><td>59.0</td><td>64.9</td><td>& LLaVA-Onevision (Functional-only)</td><td>65.1</td><td>61.7</td><td>55.8</td><td>45.4</td><td>57.0</td></tr><tr><td>& MomaGraph-R1 (Unified)</td><td>76.4</td><td>71.9</td><td>70.1</td><td>68.1</td><td>71.6</td><td>& LLaVA-Onevision (Unified)</td><td>68.6</td><td>62.9</td><td>67.5</td><td>56.5</td><td>66.0</td></tr></table>
<table><tbody><tr><td>模型</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>总体</td><td>模型</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>总体</td></tr><tr><td>& MomaGraph-R1（仅空间）</td><td>69.1</td><td>67.0</td><td>58.4</td><td>45.4</td><td>59.9</td><td>& LLaVA-Onevision（仅空间）</td><td>63.4</td><td>56.7</td><td>59.7</td><td>36.3</td><td>54.0</td></tr><tr><td>& MomaGraph-R1（仅功能）</td><td>71.4</td><td>65.8</td><td>63.6</td><td>59.0</td><td>64.9</td><td>& LLaVA-Onevision（仅功能）</td><td>65.1</td><td>61.7</td><td>55.8</td><td>45.4</td><td>57.0</td></tr><tr><td>& MomaGraph-R1（统一）</td><td>76.4</td><td>71.9</td><td>70.1</td><td>68.1</td><td>71.6</td><td>& LLaVA-Onevision（统一）</td><td>68.6</td><td>62.9</td><td>67.5</td><td>56.5</td><td>66.0</td></tr></tbody></table>


## Preliminary Findings 2
## 初步发现 2


- Graph representations that rely solely on spatial relationships or solely on functional relationships are insufficient. For embodied agents, a unified representation that jointly models both spatial and functional relationships provides a more complete and effective foundation for perception and action.
- 仅依赖空间关系或仅依赖功能关系的图表示是不充分的。对于具身智能体，将空间关系与功能关系共同建模的统一表示，能提供更完整、更有效的感知与行动基础。


## 4 METHOD
## 4 方法


### 4.1 MOMAGRAPH DEFINITION
### 4.1 MOMAGRAPH 定义


Given a single indoor room,the agent receives as input a set of multi-view images ${\left\{  {\mathcal{I}}_{i}\right\}  }_{i = 1}^{n}$ and a natural language instruction $\mathcal{T}$ . The objective is to construct an instruction-conditioned,task-oriented scene graph ${\mathcal{G}}_{\mathcal{T}} = \left( {{\mathcal{N}}_{\mathcal{T}},{\mathcal{E}}_{s}^{\mathcal{T}},{\mathcal{E}}_{f}^{\mathcal{T}}}\right)$ . Here, ${\mathcal{N}}_{\mathcal{T}}$ denotes the set of nodes representing objects relevant to task $\mathcal{T}$ . ${\mathcal{E}}_{s}^{\mathcal{T}}$ encodes the spatial relationships among these nodes,and ${\mathcal{E}}_{f}^{\mathcal{T}}$ captures their functional relationships. This task-oriented scene graph provides a minimal yet sufficient structured representation that grounds the instruction $\mathcal{T}$ in the observed scene and facilitates downstream embodied task planning. Both ${\mathcal{E}}_{s}^{\mathcal{T}}$ and ${\mathcal{E}}_{f}^{\mathcal{T}}$ are modeled as directed edges,pointing from the triggering object to the affected object.
给定一个室内房间，智能体将一组多视角图像 ${\left\{  {\mathcal{I}}_{i}\right\}  }_{i = 1}^{n}$ 和一条自然语言指令 $\mathcal{T}$ 作为输入。目标是构建一个以指令为条件、面向任务的场景图 ${\mathcal{G}}_{\mathcal{T}} = \left( {{\mathcal{N}}_{\mathcal{T}},{\mathcal{E}}_{s}^{\mathcal{T}},{\mathcal{E}}_{f}^{\mathcal{T}}}\right)$ 。其中，${\mathcal{N}}_{\mathcal{T}}$ 表示与任务 $\mathcal{T}$ 相关的对象所对应的节点集合。${\mathcal{E}}_{s}^{\mathcal{T}}$ 对这些节点之间的空间关系进行编码，${\mathcal{E}}_{f}^{\mathcal{T}}$ 则刻画它们之间的功能关系。该面向任务的场景图提供了一种最小但足够的结构化表示，用于将指令 $\mathcal{T}$ 扎根于观察到的场景，并支持后续的具身任务规划。${\mathcal{E}}_{s}^{\mathcal{T}}$ 与 ${\mathcal{E}}_{f}^{\mathcal{T}}$ 均建模为有向边，从触发对象指向被影响对象。


### 4.2 VLMs LEARN SCENE GRAPH REPRESENTATIONS WITH REINFORCEMENT LEARNING
### 4.2 VLMs 通过强化学习学习场景图表示


Existing open-source VLMs have demonstrated limited capability in generating accurate task-oriented scene graphs ${\mathcal{G}}_{\mathcal{T}}$ from multi-view observations ${\left\{  {\mathcal{I}}_{i}\right\}  }_{i = 1}^{n}$ and natural language instructions $\mathcal{T}$ . VLMs do not form structured spatial-functional representations or reason effectively about task-relevant object relationships needed for embodied tasks. To go further, we want to know: Can reinforcement learning teach VLMs to build more precise and task-relevant scene graph representations with MomaGraph?
现有开源 VLMs 在从多视角观测 ${\left\{  {\mathcal{I}}_{i}\right\}  }_{i = 1}^{n}$ 和自然语言指令 $\mathcal{T}$ 生成准确的任务导向场景图 ${\mathcal{G}}_{\mathcal{T}}$ 方面能力有限。VLMs 不会形成结构化的空间-功能表示，也无法有效推理具身任务所需的、与任务相关的物体关系。为了进一步推进，我们想知道：强化学习能否借助 MomaGraph，教会 VLMs 构建更精确、更贴合任务需求的场景图表示？


Reinforcement learning offers a more principled approach by encouraging the model to explore, reason, and iteratively refine its representations through outcome-driven feedback. Rather than replicating memorized patterns, RL enables models to discover effective strategies for constructing task-relevant scene graphs through structured thinking and reasoning. We apply the DAPO (Yu et al., 2025). The key innovation lies in our carefully designed graph-based reward function $\mathcal{R}\left( {{\mathcal{G}}_{\mathcal{T}}^{\text{ pred }},{\mathcal{G}}_{\mathcal{T}}^{\text{ gt }}}\right)$ ,where ${\mathcal{G}}_{\mathcal{T}}^{\text{ pred }}$ and ${\mathcal{G}}_{\mathcal{T}}^{\text{ gt }}$ denote the predicted and ground truth task-oriented scene graphs, respectively, which evaluates how well predicted graphs embody these principles through three key components.
强化学习提供了更有原则的方法：通过结果驱动的反馈，引导模型探索、推理，并迭代精炼其表示。RL 并不是简单复制记忆化模式，而是让模型通过结构化思考与推理，发现构建任务相关场景图的有效策略。我们采用 DAPO（Yu et al., 2025）。关键创新在于我们精心设计的基于图的奖励函数 $\mathcal{R}\left( {{\mathcal{G}}_{\mathcal{T}}^{\text{ pred }},{\mathcal{G}}_{\mathcal{T}}^{\text{ gt }}}\right)$ ，其中 ${\mathcal{G}}_{\mathcal{T}}^{\text{ pred }}$ 和 ${\mathcal{G}}_{\mathcal{T}}^{\text{ gt }}$ 分别表示预测的与真实的任务导向场景图，用三个关键组成部分评估预测图如何体现这些原则。


Action type prediction. Given the task instruction $\mathcal{T}$ ,we ensure correct prediction of the required action type through ${R}_{\text{ action }} = \mathbb{I}\left\lbrack  {{a}_{\text{ pred }} = {a}_{\mathrm{{gt}}}}\right\rbrack$ ,where ${a}_{\text{ pred }}$ and ${a}_{\mathrm{{gt}}}$ denote the predicted and ground truth action types, respectively.
动作类型预测。给定任务指令 $\mathcal{T}$ ，我们通过 ${R}_{\text{ action }} = \mathbb{I}\left\lbrack  {{a}_{\text{ pred }} = {a}_{\mathrm{{gt}}}}\right\rbrack$ 确保对所需动作类型的正确预测，其中 ${a}_{\text{ pred }}$ 和 ${a}_{\mathrm{{gt}}}$ 分别表示预测的与真实的动作类型。


Spatial-functional integration on edges. We jointly evaluate both spatial relationships ${\mathcal{E}}_{s}^{\mathcal{T}}$ and functional relationships ${\mathcal{E}}_{f}^{\mathcal{T}}$ within each edge,where ${\mathcal{E}}_{\text{ pred }}^{\mathcal{T}}$ and ${\mathcal{E}}_{\text{ gt }}^{\mathcal{T}}$ represent the predicted and ground truth edge sets:
边上的空间-功能融合。我们在每条边上同时评估空间关系 ${\mathcal{E}}_{s}^{\mathcal{T}}$ 和功能关系 ${\mathcal{E}}_{f}^{\mathcal{T}}$ ，其中 ${\mathcal{E}}_{\text{ pred }}^{\mathcal{T}}$ 和 ${\mathcal{E}}_{\text{ gt }}^{\mathcal{T}}$ 表示预测与真实的边集：


$$
{R}_{\text{ edges }} = \frac{1}{\left| {\mathcal{E}}_{\text{ gt }}^{\mathcal{T}}\right| }\mathop{\sum }\limits_{{{e}_{j} \in  {\mathcal{E}}_{\text{ gt }}^{\mathcal{T}}}}\mathop{\max }\limits_{{{e}_{i} \in  {\mathcal{E}}_{\text{ pred }}^{\mathcal{T}}}}{S}_{\text{ edge }}\left( {{e}_{i},{e}_{j}}\right) \tag{1}
$$



where ${S}_{\text{ edge }}\left( {{e}_{i},{e}_{j}}\right)$ measures semantic similarity between edges ${e}_{i}$ and ${e}_{j}$ based on their spatial and functional relationship labels.
其中 ${S}_{\text{ edge }}\left( {{e}_{i},{e}_{j}}\right)$ 衡量两条边 ${e}_{i}$ 与 ${e}_{j}$ 之间的语义相似度，依据它们的空间与功能关系标签。


Node completeness. We compute intersection-over-union similarity for task-relevant objects in ${\mathcal{N}}_{\mathcal{T}}$ , where ${\mathcal{N}}_{\mathcal{T}}^{\text{ pred }}$ and ${\mathcal{N}}_{\mathcal{T}}^{\text{ gt }}$ denote the predicted and ground truth sets of task-relevant nodes: ${R}_{\text{ nodes }} =$
节点完整性。我们在 ${\mathcal{N}}_{\mathcal{T}}$ 上对与任务相关的物体计算交并比相似度，其中 ${\mathcal{N}}_{\mathcal{T}}^{\text{ pred }}$ 和 ${\mathcal{N}}_{\mathcal{T}}^{\text{ gt }}$ 分别表示与任务相关节点的预测集合与真实集合：${R}_{\text{ nodes }} =$


$\frac{\left| {\mathcal{N}}_{\mathcal{T}}^{\text{ pred }} \cap  {\mathcal{N}}_{\mathcal{T}}^{\text{ gt }}\right| }{\left| {\mathcal{N}}_{\mathcal{T}}^{\text{ pred }} \cup  {\mathcal{N}}_{\mathcal{T}}^{\text{ gt }}\right| }.$



The final reward function integrates these task-oriented design principles with format validation and length control,where ${R}_{\text{ format }}$ ensures valid JSON structure and ${R}_{\text{ length }}$ penalizes overly verbose outputs:
最终的奖励函数将这些面向任务的设计原则与格式校验和长度控制结合起来，其中 ${R}_{\text{ format }}$ 确保有效的 JSON 结构，${R}_{\text{ length }}$ 则惩罚过长、冗余的输出：


$$
\mathcal{R}\left( {{\mathcal{G}}_{\mathcal{T}}^{\text{ pred }},{\mathcal{G}}_{\mathcal{T}}^{\text{ gt }}}\right)  = {w}_{a} \cdot  \left( {{R}_{\text{ action }} + {R}_{\text{ edges }} + {R}_{\text{ nodes }}}\right)  + {w}_{f} \cdot  {R}_{\text{ format }} + {w}_{l} \cdot  {R}_{\text{ length }} \tag{2}
$$



where ${w}_{a},{w}_{f}$ ,and ${w}_{l}$ are hyperparameters controlling the relative importance of each component.
其中 ${w}_{a},{w}_{f}$ 和 ${w}_{l}$ 是超参数，用于控制各组成部分的相对重要性。


This reward design directly implements our core insight: scene graphs must simultaneously capture spatial layout $\left( {\mathcal{E}}_{s}^{\mathcal{T}}\right)$ and functional relationships $\left( {\mathcal{E}}_{f}^{\mathcal{T}}\right)$ while remaining tightly coupled to task requirements $\left( \mathcal{T}\right)$ . With RL training on MomaGraph-Scenes,we develop MomaGraph-R1,a 7B vision-language model built on Qwen2.5-VL-7B-Instruct (Qwen, 2025), which learns to generate compact, task-relevant representations that provide concrete guidance for embodied planning.
这种奖励设计直接落实了我们的核心洞见：场景图必须同时捕捉空间布局 $\left( {\mathcal{E}}_{s}^{\mathcal{T}}\right)$ 和功能关系 $\left( {\mathcal{E}}_{f}^{\mathcal{T}}\right)$ ，同时与任务需求 $\left( \mathcal{T}\right)$ 紧密耦合。通过在 MomaGraph-Scenes 上进行 RL 训练，我们开发了 MomaGraph-R1：一个基于 Qwen2.5-VL-7B-Instruct（Qwen, 2025）的 7B 视觉-语言模型，它学习生成紧凑、与任务相关的表示，为具身规划提供具体指导。


We demonstrate that RL significantly enhances both the effectiveness and generalizability of open-source VLMs for scene graph generation in the following section. This aligns with broader findings that combining structured scene representations with reasoning consistently improves VLM scene understanding. Critically, MomaGraph-R1 achieves robust performance across diverse environments and task configurations, enabling practical deployment in unseen embodied scenarios.
我们在下一节中表明，强化学习显著提升了开源 VLM 在场景图生成任务上的效果与泛化能力。这与更广泛的结论一致：将结构化场景表示与推理结合，能够持续改善 VLM 的场景理解。关键在于，MomaGraph-R1 能在多样环境与任务配置中保持稳健表现，使其能够在未见的具身场景中实现可落地部署。


### 4.3 STATE-AWARE DYNAMIC SCENE GRAPH UPDATE
### 4.3 具备状态感知的动态场景图更新


In realistic environments, multiple objects of the same category may coexist, and their task-related correspondences are often initially uncertain. Take Figure 3 as an example, a kitchen stove may have several knobs, but only one controls the burner required for the current cooking task. Simply relying on visual appearance is insufficient to determine the correct functional relationship. In this work, we do not focus on the agent's interaction policy; instead, our emphasis lies on how to capture and incorporate observed state changes in the environment into the scene graph to resolve such ambiguities.
在真实环境中，同一类别的多个物体可能同时存在，而它们与任务相关的对应关系往往一开始并不明确。以图 3 为例，厨房炉灶可能有多个旋钮，但只有一个控制当前烹饪任务所需的燃烧器。仅依赖视觉外观不足以判断正确的功能关系。在这项工作中，我们并不关注智能体的交互策略；相反，我们的重点在于如何捕捉并将环境中观察到的状态变化纳入场景图，以消除这类歧义。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_227773.jpg"/>



Figure 3: MomaGraph captures state changes in the environment and dynamically updates the task-specific scene graph accordingly, enabling the graph to evolve as interactions occur and reflecting updated spatial-functional relationships.
图3：MomaGraph 捕捉环境中的状态变化，并相应地动态更新面向任务的场景图，使图能够随着交互的发生而演进，同时反映更新后的空间-功能关系。


Formally,at time step $t$ ,the task-oriented scene graph is represented as:
形式上，在时间步 $t$ ，面向任务的场景图表示为：


$$
{\mathcal{G}}_{\mathcal{T}}^{\left( t\right) } = \left( {{\mathcal{N}}_{\mathcal{T}}^{\left( t\right) },{\mathcal{E}}_{s}^{\mathcal{T},\left( t\right) },{\mathcal{E}}_{f}^{\mathcal{T},\left( t\right) }}\right) , \tag{3}
$$



where ${\mathcal{N}}_{\mathcal{T}}^{\left( t\right) }$ denotes the set of task-relevant candidate objects, ${\mathcal{E}}_{s}^{\mathcal{T},\left( t\right) }$ encodes their spatial layout,and ${\mathcal{E}}_{f}^{\mathcal{T},\left( t\right) }$ captures hypothesized functional relationships, which may initially include one-to-many mappings.
其中 ${\mathcal{N}}_{\mathcal{T}}^{\left( t\right) }$ 表示与任务相关的候选对象集合，${\mathcal{E}}_{s}^{\mathcal{T},\left( t\right) }$ 编码它们的空间布局，${\mathcal{E}}_{f}^{\mathcal{T},\left( t\right) }$ 则刻画被假设的功能关系；这些关系一开始可能包含一对多的映射。


After the agent executes an action ${a}_{t}$ and observes the new environment state ${s}_{t + 1}$ ,the scene graph is refined as:
代理执行动作 ${a}_{t}$ 并观察到新的环境状态 ${s}_{t + 1}$ 后，场景图会被如下细化：


$$
{\mathcal{G}}_{\mathcal{T}}^{\left( t + 1\right) } = \mathcal{U}\left( {{\mathcal{G}}_{\mathcal{T}}^{\left( t\right) },{a}_{t},{s}_{t + 1}}\right) , \tag{4}
$$



where the update function $\mathcal{U}\left( \cdot \right)$ removes incon-
其中，更新函数$\mathcal{U}\left( \cdot \right)$移除不连-


sistent hypotheses and strengthens confirmed correspondences based on the observed state transition. As illustrated in Fig. 3, if rotating a specific knob ignites the burner while others have no effect, the functional edge [control] between that knob and the burner is established, while edges from other knobs are pruned. This process enables the scene graph to evolve from ambiguous, one-to-many hypotheses into a compact, state-aware dynamic representation with unique and reliable object-to-object correspondences.
基于观测到的状态转变，消除不一致的假设，并强化已确认的对应关系。如图3所示，如果旋转某个特定旋钮会点燃燃烧器，而其他旋钮没有效果，则会建立该旋钮与燃烧器之间的功能边 [control]，同时剪除其他旋钮的边。该过程使场景图能够从含糊的、一对多的假设演化为紧凑、感知状态的动态表示，具有唯一且可靠的物体间对应关系。


## 5 DATASET AND BENCHMARK
## 5 数据集与基准


### 5.1 MOMAGRAPH-SCENES DATASET
### 5.1 MOMAGRAPH-SCENES 数据集


Existing scene graph datasets for 3D indoor environments are often constrained to a single relationship: some focus exclusively on spatial layouts of objects (Armeni et al., 2019; Koch et al., 2024b), while others emphasize functional interactions (Dong et al., 2021; Zhang et al., 2025). However, these scene graph representations that are restricted to a single relationship type are insufficient for embodied agents, as task execution in household environments requires reasoning about both where objects are and how they can be used. To address these limitations, we introduce MomaGraph-Scenes, the first dataset designed to provide a more comprehensive and task-relevant scene representation. MomaGraph-Scenes jointly encodes spatial relationships and functional relationships, covering 9 spatial relationship types and 6 functional relationship
用于3D室内环境的现有场景图数据集往往只能覆盖单一关系：有些仅关注物体的空间布局（Armeni et al., 2019; Koch et al., 2024b），而另一些则强调功能交互（Dong et al., 2021; Zhang et al., 2025）。然而，受限于单一关系类型的这些场景图表征不足以满足具身智能体的需求，因为在家庭环境中执行任务需要同时推理物体在哪里以及它们如何被使用。为了解决这些局限，我们提出 MomaGraph-Scenes，这是首个旨在提供更全面且与任务相关的场景表征的数据集。MomaGraph-Scenes 共同编码空间关系与功能关系，覆盖 9 种空间关系类型和 6 种功能关系


types, explicitly representing interactive elements such as handles and buttons. Our dataset consists of approximately 1,050 task-oriented subgraphs and 6278 multi-view RGB images, collected from a combination of manually collected real-world data, re-annotated existing datasets (Zhang et al., 2025; Delitzas et al., 2024), and simulated environments built with AI2-THOR (Kolve et al., 2017). These samples span more than 350 diverse household scenes and encompass 93 distinct task instructions. Compared with prior datasets, our annotations are significantly more detailed, and capturing interaction semantics at both the object and part levels. This broad coverage ensures rich variability in scene layouts, object configurations, and interaction types, supporting robust learning and evaluation of embodied reasoning.
类型，并明确表示如把手和按钮之类的交互元素。我们的数据集包含约 1,050 个面向任务的子图以及 6278 张多视角 RGB 图像，数据来源结合了人工采集的真实数据、对已有数据集重新标注（Zhang et al., 2025; Delitzas et al., 2024），以及使用 AI2-THOR 构建的模拟环境（Kolve et al., 2017）。这些样本覆盖了 350 多个多样的家庭场景，并包含 93 条不同的任务指令。与以往数据集相比，我们的标注细节显著更丰富，并且能够在对象层级与部件层级共同捕获交互语义。如此广泛的覆盖范围确保了场景布局、物体配置与交互类型的多样性，从而支持具身推理的稳健学习与评估。


#### 5.1.1 DATASET ANNOTATION
#### 5.1.1 数据集标注


Multi-View Observation. The multi-view images provided for each graph are not constrained to always contain every relevant object within each single view. We also do not impose restrictions on the number of viewpoints or their exact configurations. This flexible setup better reflects realistic perception conditions, where embodied agents must reason across partial and diverse observations to build consistent scene graph representations.
多视角观测。为每个图提供的多视角图像并不被要求在任一单视角中都始终包含所有相关物体。我们也不限制视角的数量或其具体配置。这样的灵活设置更贴近真实的感知条件：具身智能体必须在部分且多样的观测中进行推理，才能构建一致的场景图表示。


Task Instruction. It is worth noting that the task instructions in our dataset do not explicitly mention all the objects required to accomplish the task. Instead, they are expressed in simple and natural forms (e.g., "Fill the bathtub"), where the relevant objects such as the bathtub, faucet, and button must be inferred by the model. This design encourages the model to learn how to ground natural instructions into the appropriate set of objects and relationships, rather than relying on object names being explicitly stated.
任务指令。需要注意的是，我们数据集中的任务指令并不会明确提及完成任务所需的所有物体。相反，它们以简单自然的形式表达（例如“填满浴缸”），其中与之相关的物体如浴缸、水龙头和按钮都需要由模型进行推断。该设计旨在促使模型学习如何将自然语言指令落到合适的物体集合及其关系上，而不是依赖物体名称被直接写明。


Nodes. ${\mathcal{N}}_{\mathcal{T}}$ primarily consists of the objects necessary to accomplish the instruction. When the task execution requires interacting with specific parts, the graph may additionally include part-level interactive elements (e.g., handles, knobs, or buttons). For example, for the instruction "Open the fridge," ${\mathcal{N}}_{\mathcal{T}}$ includes both the fridge and its handle; for the instruction "Turn on the light," ${\mathcal{N}}_{\mathcal{T}}$ consists of the switch and the ceiling light.
节点。${\mathcal{N}}_{\mathcal{T}}$ 主要由完成指令所需的物体构成。当任务执行需要与特定部件交互时，图还可能包含基于部件级别的交互元素（例如把手、旋钮或按钮）。例如，对于指令“打开冰箱”，${\mathcal{N}}_{\mathcal{T}}$ 同时包含冰箱及其把手；对于指令“打开灯光”，${\mathcal{N}}_{\mathcal{T}}$ 由开关和顶灯组成。


Edges. Edges in the task-oriented scene graph capture both functional and spatial relationships between nodes.
边。任务导向的场景图中的边同时刻画节点之间的功能关系和空间关系。


- Functional Relationships. We define a functional relationship as the ability of one object to change the state of another object. In indoor environments, common tasks can be broadly categorized as Parameter Adjustment, Device Control, Open/Close the Cabinet or Door, Water Flow Control, Power Supply, and Assembly. Accordingly, we identify six major types:
- 功能关系。我们将功能关系定义为：一个物体能够改变另一个物体状态的能力。在室内环境中，常见任务可大致归类为 参数调整、设备控制、打开/关闭柜子或门、控制水流、供电，以及装配。因此，我们识别出六种主要类型：


[OPEN OR CLOSE], [ADJUST], [CONTROL], [ACTIVATE], [POWER BY], [PAIR WITH].
[打开或关闭]、[调整]、[控制]、[激活]、[由…供电]、[与…配对]。


Notably, [PAIR WITH] does not alter the internal state of objects but instead modifies their spatial configuration, which is essential for assembly tasks (Qi et al., 2025). Since such tasks are critical for robotic interaction and task planning, we explicitly include [PAIR WITH] as a functional relationship. Through this definition, our dataset extends beyond physical and electronic interactions to encompass fine-grained reasoning about assembly and pairing, enhancing its utility for downstream action execution and planning.
值得注意的是，[与…配对] 不会改变物体的内部状态，而是修改它们的空间配置；这对装配任务至关重要（Qi et al., 2025）。由于这类任务对机器人交互与任务规划至关重要，我们明确将[与…配对]纳入为一种功能关系。通过这一定义，我们的数据集从单纯的物理与电子交互扩展到对装配与配对的细粒度推理，从而提升其在下游动作执行与规划中的效用。


- Spatial Relationships. Capture geometric dependencies between objects and parts. The dataset primarily annotates:
- 空间关系。捕捉物体及部件之间的几何依赖性。数据集主要标注：


Directional: [LEFT OF], [RIGHT OF], [IN FRONT OF], [BEHIND], [HIGHER THAN], [LOWER THAN].
方向性：[在左侧]、[在右侧]、[在前方]、[在后方]、[更高于]、[更低于]。


Distance-based: [CLOSE], [FAR], [TOUCHING].
基于距离：[近]、[远]、[接触]。


These annotations provide the rich context necessary for reasoning about layout, reachability, and interaction feasibility.
这些标注为推理布局、可达性以及交互可行性提供了丰富的上下文。


### 5.2 MOMAGRAPH BENCHMARK AND EVALUATION
### 5.2 MOMAGRAPH 基准与评估


We introduce MomaGraph-Bench, the first benchmark that jointly evaluates fine-grained scene understanding and task planning abilities across diverse levels of difficulty. Our design principle for MomaGraph-Bench is to evaluate whether advances in scene understanding provide tangible improvements in downstream task planning and reasoning. Our evaluation framework examines six essential reasoning capabilities in four tiers of difficulty levels: (1) Action Sequence Reasoning, (2)
我们提出 MomaGraph-Bench，这是首个在不同难度层级上对细粒度场景理解与任务规划能力进行联合评估的基准。MomaGraph-Bench 的设计原则是评估：场景理解方面的进步能否为下游任务规划与推理带来切实提升。我们的评估框架从四个难度层级中考察六项关键推理能力：（1）动作序列推理，（2）


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_6b026a.jpg"/>



Figure 4: Examples of evaluation Multi-Choices VQA tasks in the MomaGraph-Bench. We showcase example questions covering six core reasoning capabilities. Beyond these core capabilities, we further design tasks on Dynamic Verification and Long-horizon Task Decomposition to evaluate temporal reasoning and multi-steps planning.
图4：MomaGraph-Bench 中多选式 VQA 评测任务示例。我们展示涵盖六种核心推理能力的示例问题。在这六种核心能力之外，我们还设计了用于动态验证与长时域任务分解的任务，以评估时间推理与多步骤规划。


Spatial Reasoning, (3) Object Affordance Reasoning, (4) Precondition & Effect Reasoning, (5) Goal Decomposition, and (6) Visual Correspondence (with concrete examples shown in Fig. 4).
空间推理，（3）物体可用性推理，（4）前置条件与效果推理，（5）目标分解，以及（6）视觉对应（并在图4中展示了具体示例）。


- Action Sequence Reasoning: examines whether models understand the order and dependency of actions and can plan efficient sequences.
- 动作序列推理：考察模型是否理解动作的顺序与依赖关系，并能规划高效的序列。


- Spatial Reasoning: focuses on reasoning over spatial relations such as left_of or in_front_of, judging reachability, and selecting the most suitable object among candidates.
- 空间推理：侧重于对诸如 left_of 或 in_front_of 等空间关系进行推理，判断可达性，并在候选对象中选择最合适的目标。


- Object Affordance Reasoning: evaluates whether models can infer the functionality of objects (e.g., knobs can be turned, cabinets can be opened), match objects to task requirements, and reason about indirect tool use.
- 物体可用性推理：评估模型能否推断物体的功能（例如旋钮可被转动、柜子可被打开），将物体与任务需求相匹配，并推理间接的工具使用。


- Precondition & Effect Reasoning: assesses whether models understand the preconditions and effects of actions, such as a door needing to be closed before it can be opened, and can predict possible side effects.
- 前提与效果推理：评估模型是否理解动作的前提与结果，例如门在被打开前需要先关闭，以及是否能预测可能的副作用。


- Goal Decomposition: measures the ability to break down complex tasks into sub-goals, prioritize them, and determine parallel versus sequential execution strategies.
- 目标分解：衡量将复杂任务拆解为子目标、对其优先级进行排序，以及确定并行还是串行执行策略的能力。


- Visual Correspondence (extended capability): tests whether models can maintain object consistency across multiple views and integrate information under viewpoint changes.
- 视觉对应关系（扩展能力）：检验模型能否在多视角下保持对象一致性，并在视角变化时整合信息。


MomaGraph-Bench is formulated as a multi-choice VQA task which comprises 294 diverse indoor scenes with 1,446 multi-view images, featuring 352 task-oriented scene graphs spanning 1,315 instances that range from simple step object manipulation(Tier 1) to complex multi-step replanning (Tier 4) scenarios (detailed breakdown in Appendix A.4). MomaGraph-Bench offers the most comprehensive assessment for embodied agents' capacity to generalize across tasks and scenarios. To ensure that the evaluation truly reflects generalization rather than memorization, all scenarios are drawn from entirely unseen environments.
MomaGraph-Bench 被设计为一项多选 VQA 任务，包含 294 个多样化的室内场景，配有 1,446 幅多视角图像。任务提供 352 个面向任务的场景图，覆盖 1,315 个实例，场景从简单的逐步物体操作（第 1 层级）到复杂的多步骤重规划（第 4 层级）场景（详见附录 A.4）。MomaGraph-Bench 为具身智能体在跨任务与跨场景进行泛化的能力提供了最全面的评估。为确保评估反映的是泛化而非记忆，所有场景均来自完全未见过的环境。


## 6 EXPERIMENTS
## 6 实验


### 6.1 BENCHMARK EVALUATION FOR EMBODIED TASK PLANNING
### 6.1 面向具身任务规划的基准评估


We compare the performance of our MomaGraph-R1 with other models across all task tiers in MomaGraph-Bench to rigorously assess embodied planning, including state-of-the-art closed source models (Claude-4.5-Sonnet, GPT-5, Gemini-2.5-Pro) and leading open source models (In-structBLIP, LLaVA-V1.5, DeepSeek-VL2, InternVL2.5, LLaVA-OneVision, Qwen2.5). We further examine whether Graph-then-Plan brings performance gains by evaluating each model under two controlled settings: (i) Direct Plan (w/o Graph): the model is directly evaluated on task planning in MomaGraph-Bench using multi-view observations and instructions; (ii) Graph-then-Plan (w/ Graph): the model first generates a task-oriented scene graph ${\mathcal{G}}_{\mathcal{T}}$ ,capturing nodes,spatial and functional edges, and action types, and then performs task planning conditioned on the graph.
我们在 MomaGraph-Bench 的所有任务层级上，将我们提出的 MomaGraph-R1 与其他模型的表现进行对比，以严格评估具身规划能力，包括最先进的闭源模型（Claude-4.5-Sonnet、GPT-5、Gemini-2.5-Pro）以及领先的开源模型（In-structBLIP、LLaVA-V1.5、DeepSeek-VL2、InternVL2.5、LLaVA-OneVision、Qwen2.5）。我们进一步检验“图后计划（Graph-then-Plan）”是否能带来性能提升：通过在两种受控设置下分别评估每个模型：（i）直接计划（Direct Plan，无图）：模型在使用多视角观测与指令的前提下，直接在 MomaGraph-Bench 上进行任务规划评估；（ii）图后计划（Graph-then-Plan，有图）：模型先生成面向任务的场景图 ${\mathcal{G}}_{\mathcal{T}}$ ，以刻画节点、空间与功能边，以及动作类型，然后在该图的条件下执行任务规划。


Table 2: Performance comparison on the MomaGraph-Bench. We report accuracy (%) across four tiers (T1-T4) and the overall score, with and without graph-based reasoning.
表 2：MomaGraph-Bench 上的性能对比。我们报告四个层级（T1-T4）的准确率（%）以及总体得分，并区分是否使用基于图的推理。


<table><tr><td rowspan="3"></td><td rowspan="3">ype Models</td><td rowspan="3">Params</td><td colspan="10">MomaGraph Benchmark</td></tr><tr><td colspan="2">Tier 1</td><td colspan="2">Tier 2</td><td colspan="2">Tier 3</td><td colspan="2">Tier 4</td><td colspan="2">Overall</td></tr><tr><td>w/o Graph</td><td>w/ Graph</td><td>w/o Graph</td><td>w/ Graph</td><td>w/o Graph</td><td>w/ Graph</td><td>w/o Graph</td><td>w/ Graph</td><td>w/o Graph</td><td>w/ Graph</td></tr><tr><td rowspan="3">Closed Source</td><td>* Claude-4.5-Sonnet</td><td>-</td><td>77.3</td><td>83.7</td><td>67.0</td><td>70.3</td><td>69.7</td><td>72.3</td><td>65.2</td><td>69.5</td><td>69.8</td><td>73.9</td></tr><tr><td>§ GPT-5</td><td>-</td><td>77.3</td><td>79.8</td><td>63.4</td><td>68.2</td><td>70.8</td><td>75.0</td><td>54.5</td><td>63.6</td><td>66.5</td><td>71.6</td></tr><tr><td>- Gemini-2.5-Pro</td><td>-</td><td>76.6</td><td>79.0</td><td>65.8</td><td>69.5</td><td>67.5</td><td>72.7</td><td>60.8</td><td>65.2</td><td>67.6</td><td>71.6</td></tr><tr><td rowspan="7">Open Source</td><td>- InstructBLIP-7B</td><td>7B</td><td>43.1</td><td>44.1</td><td>42.6</td><td>41.4</td><td>38.6</td><td>36.3</td><td>31.8</td><td>36.3</td><td>39.0</td><td>39.5</td></tr><tr><td>& LLaVA-V1.5-7B</td><td>7B</td><td>51.0</td><td>53.4</td><td>46.3</td><td>48.7</td><td>40.2</td><td>36.3</td><td>38.9</td><td>40.9</td><td>44.1</td><td>44.8</td></tr><tr><td>C DeepSeek-VL2</td><td>4.5B</td><td>54.2</td><td>56.9</td><td>51.2</td><td>53.6</td><td>61.8</td><td>61.3</td><td>40.9</td><td>45.4</td><td>52.0</td><td>54.3</td></tr><tr><td>InternVL2.5-8B</td><td>8B</td><td>53.6</td><td>51.0</td><td>51.2</td><td>53.0</td><td>55.8</td><td>59.7</td><td>33.3</td><td>40.9</td><td>48.4</td><td>51.1</td></tr><tr><td>I  ALLaVA-Onevision-7B</td><td>7B</td><td>60.0</td><td>63.8</td><td>52.4</td><td>56.0</td><td>58.4</td><td>59.2</td><td>43.4</td><td>43.4</td><td>53.5</td><td>55.6</td></tr><tr><td>5 Owen2.5-VL-7B-Instruct</td><td>7B</td><td>62.1</td><td>66.3</td><td>58.5</td><td>58.5</td><td>51.9</td><td>57.1</td><td>56.5</td><td>59.0</td><td>57.2</td><td>60.2</td></tr><tr><td>& MomaGraph-R1(Ours)</td><td>7B</td><td>70.2</td><td>76.4</td><td>65.8</td><td>71.9</td><td>63.6</td><td>70.1</td><td>60.8</td><td>68.1</td><td>65.1</td><td>71.6</td></tr></table>
<table><tbody><tr><td rowspan="3"></td><td rowspan="3">框架模型</td><td rowspan="3">参数</td><td colspan="10">MomaGraph 基准</td></tr><tr><td colspan="2">第 1 档</td><td colspan="2">第 2 档</td><td colspan="2">第 3 档</td><td colspan="2">第 4 档</td><td colspan="2">总体</td></tr><tr><td>无图</td><td>有图</td><td>无图</td><td>有图</td><td>无图</td><td>有图</td><td>无图</td><td>有图</td><td>无图</td><td>有图</td></tr><tr><td rowspan="3">闭源</td><td>* Claude-4.5-Sonnet</td><td>-</td><td>77.3</td><td>83.7</td><td>67.0</td><td>70.3</td><td>69.7</td><td>72.3</td><td>65.2</td><td>69.5</td><td>69.8</td><td>73.9</td></tr><tr><td>§ GPT-5</td><td>-</td><td>77.3</td><td>79.8</td><td>63.4</td><td>68.2</td><td>70.8</td><td>75.0</td><td>54.5</td><td>63.6</td><td>66.5</td><td>71.6</td></tr><tr><td>- Gemini-2.5-Pro</td><td>-</td><td>76.6</td><td>79.0</td><td>65.8</td><td>69.5</td><td>67.5</td><td>72.7</td><td>60.8</td><td>65.2</td><td>67.6</td><td>71.6</td></tr><tr><td rowspan="7">开源</td><td>- InstructBLIP-7B</td><td>7B</td><td>43.1</td><td>44.1</td><td>42.6</td><td>41.4</td><td>38.6</td><td>36.3</td><td>31.8</td><td>36.3</td><td>39.0</td><td>39.5</td></tr><tr><td>& LLaVA-V1.5-7B</td><td>7B</td><td>51.0</td><td>53.4</td><td>46.3</td><td>48.7</td><td>40.2</td><td>36.3</td><td>38.9</td><td>40.9</td><td>44.1</td><td>44.8</td></tr><tr><td>C DeepSeek-VL2</td><td>4.5B</td><td>54.2</td><td>56.9</td><td>51.2</td><td>53.6</td><td>61.8</td><td>61.3</td><td>40.9</td><td>45.4</td><td>52.0</td><td>54.3</td></tr><tr><td>InternVL2.5-8B</td><td>8B</td><td>53.6</td><td>51.0</td><td>51.2</td><td>53.0</td><td>55.8</td><td>59.7</td><td>33.3</td><td>40.9</td><td>48.4</td><td>51.1</td></tr><tr><td>I  ALLaVA-Onevision-7B</td><td>7B</td><td>60.0</td><td>63.8</td><td>52.4</td><td>56.0</td><td>58.4</td><td>59.2</td><td>43.4</td><td>43.4</td><td>53.5</td><td>55.6</td></tr><tr><td>5 Owen2.5-VL-7B-Instruct</td><td>7B</td><td>62.1</td><td>66.3</td><td>58.5</td><td>58.5</td><td>51.9</td><td>57.1</td><td>56.5</td><td>59.0</td><td>57.2</td><td>60.2</td></tr><tr><td>& MomaGraph-R1（我们）</td><td>7B</td><td>70.2</td><td>76.4</td><td>65.8</td><td>71.9</td><td>63.6</td><td>70.1</td><td>60.8</td><td>68.1</td><td>65.1</td><td>71.6</td></tr></tbody></table>


Table 3: Performance comparison on the BLINK and MomaGraph-Bench. By enforcing multiview consistency, our method significantly improves correspondence reasoning across all open-source models.
表 3：在 BLINK 和 MomaGraph-Bench 上的性能对比。通过强制多视角一致性，我们的方法显著提升了所有开源模型在对应关系推理方面的表现。


<table><tr><td rowspan="2">Model</td><td colspan="2">F GPT-5</td><td colspan="2">& LLaVA-Onevision</td><td colspan="2">IP Qwen2.5-VL-7B-Instruct</td><td colspan="2">CoepSeek-VL2</td><td colspan="2">GomoaGraph-R1</td></tr><tr><td>BLINK</td><td>MomaGraph-Bench</td><td>BLINK</td><td>MomaGraph-Bench</td><td>BLINK</td><td>MomaGraph-Bench</td><td>BLINK</td><td>MomaGraph-Bench</td><td>BLINK</td><td>MomaGraph-Bench</td></tr><tr><td>Results</td><td>66.1</td><td>81.2</td><td>59.7</td><td>70.7</td><td>58.7</td><td>72.7</td><td>57.4</td><td>68.4</td><td>63.5</td><td>77.5</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">GPT-5</td><td colspan="2">& LLaVA-Onevision</td><td colspan="2">IP Qwen2.5-VL-7B-Instruct</td><td colspan="2">CoepSeek-VL2</td><td colspan="2">GomoaGraph-R1</td></tr><tr><td>BLINK</td><td>MomaGraph-Bench</td><td>BLINK</td><td>MomaGraph-Bench</td><td>BLINK</td><td>MomaGraph-Bench</td><td>BLINK</td><td>MomaGraph-Bench</td><td>BLINK</td><td>MomaGraph-Bench</td></tr><tr><td>结果</td><td>66.1</td><td>81.2</td><td>59.7</td><td>70.7</td><td>58.7</td><td>72.7</td><td>57.4</td><td>68.4</td><td>63.5</td><td>77.5</td></tr></tbody></table>


#### 6.1.1 RESULT ANALYSIS.
#### 6.1.1 结果分析。


The results in Table 2 yield several key insights:
表2中的结果带来若干关键洞见：


(1) Effectiveness of Graph-then-Plan. Across all models, the w/ Graph setting consistently outperforms the w/o Graph baseline, demonstrating that explicitly structuring task-oriented scene graphs provides a tangible benefit for downstream planning. This validates our central hypothesis that disentangling scene representation from action generation improves reasoning reliability.
(1) 图后再规划的有效性。对所有模型而言，采用Graph设置的版本始终优于不使用Graph的基线，表明对任务导向场景图进行显式建模确实能为下游规划带来切实收益。这验证了我们的核心假设：将场景表征与动作生成解耦能够提升推理可靠性。


(2) Competitiveness of MomaGraph-R1. Our MomaGraph-R1 achieves performance on par with closed-source giants like Claude-4.5-Sonnet and GPT-5, while clearly surpassing all leading open-source VLMs. Notably, MomaGraph-R1 delivers a +11.4% relative improvement over its base model (Qwen2.5-VL-7B) under $w/$ Graph,highlighting the effectiveness of reinforcement learning with graph-based rewards.
(2) MomaGraph-R1的竞争力。MomaGraph-R1的表现与Claude-4.5-Sonnet和GPT-5这类闭源巨头处于同一水平，同时明显超过所有领先的开源VLM。值得注意的是，在 $w/$ Graph 条件下，MomaGraph-R1相对其基础模型（Qwen2.5-VL-7B）实现了+11.4%的相对提升，突显了基于图结构奖励的强化学习的有效性。


(3) Scalability with Task Complexity. As task complexity increases from Tier 1 to Tier 4, the performance of most open-source baselines drops sharply, reflecting their limited ability to generalize to multi-step reasoning. In contrast, MomaGraph-R1 exhibits a much smaller degradation, preserving strong performance in Tier 3 and Tier 4. This indicates superior scalability to long-horizon planning scenarios, a crucial capability for embodied agents.
(3) 随任务复杂度提升的可扩展性。随着任务复杂度从第1层提升到第4层，多数开源基线的性能显著下滑，反映出它们对多步推理的泛化能力受限。相比之下，MomaGraph-R1的退化幅度更小，能够在第3层和第4层仍保持强劲表现。这表明其在长时域规划场景中具备更优的可扩展性，这是具身智能体的一项关键能力。


(4) General Trend Across Communities. Closed-source models still maintain the highest absolute performance, benefiting from larger-scale pretraining and proprietary data. However, the consistent gap reduction achieved by MomaGraph-R1 shows that reinforcement learning with graph-structured intermediate representations can substantially narrow the divide, offering a practical path toward competitive open-source systems.
(4) 跨社区的一般趋势。闭源模型依然保持最高的绝对性能，这得益于更大规模的预训练和专有数据。不过，MomaGraph-R1所实现的稳定差距缩小表明：通过强化学习引入基于图结构的中间表示，可以在很大程度上拉近差距，为构建具竞争力的开源系统提供了可行路径。


### 6.2 BENCHMARK EVALUATION FOR VISUAL CORRESPONDENCE
### 6.2 视觉对应基准评估


As the model learns scene representations from multi-view observations, it exhibits an emergent ability of cross-view consistency, which can reason about the same point across different viewpoints.
随着模型从多视角观测中学习场景表征，它表现出一种涌现出的跨视角一致性能力，能够在不同视角下对同一目标点进行推理。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_3fe203.jpg"/>



Figure 5: Real Robot experiments on the RobotEra Q5 with a D455, demonstrating four household tasks that require spatial, functional, and part-level interactive elements reasoning for task execution.
图5：在 RobotEra Q5 上进行的真实机器人实验，使用 D455，展示了四项家庭任务；这些任务需要对空间、功能以及部件级交互要素进行推理以完成任务执行。


This capability is most evident in visual correspondence tasks. As shown in Table 3, we compare model performance on visual correspondence tasks from public benchmark BLINK Fu et al. (2024) and our MomaGraph-Bench. Scene graph representations enhance performance universally by reducing VLM hallucinations in visual perception. By prompting models to first generate structured scene graphs (w/ Graph) and then answer questions in single-turn interactions, we force them to explicitly reason about spatial and functional relationships between objects before answering. We primarily evaluate perception on multi-view reasoning and visual correspondence tasks from BLINK, as well as multi-view correspondence in MomaGraph-Bench. Our MomaGraph-R1 achieves state-of-the-art performance among open-source VLMs, leading by 3.8% on BLINK and 4.8% on our correspondence benchmark compared to the best competing open-source models. These results confirm that MomaGraph-R1 enables more nuanced and detailed perception of complex indoor scenes, effectively mitigating hallucinations and enabling more reliable scene perception.
这种能力在视觉对应任务中最为明显。如表3所示，我们比较了模型在公共基准 BLINK Fu 等人（2024）和我们的 MomaGraph-Bench 上的视觉对应任务表现。场景图表征通过减少视觉感知中的 VLM 幻觉，使性能得到普遍提升。我们通过提示模型先生成结构化场景图（带 Graph），再在单轮交互中回答问题，迫使它在作答前显式推理物体之间的空间与功能关系。我们主要评估来自 BLINK 的多视角推理与视觉对应任务，以及 MomaGraph-Bench 上的多视角对应。与表现最好的开源竞争模型相比，我们的 MomaGraph-R1 在开源 VLM 中达到最先进水平：在 BLINK 上领先 3.8%，在我们的对应基准上领先 4.8%。这些结果表明，MomaGraph-R1 能够实现对复杂室内场景更细致、更深入的感知，有效缓解幻觉，并实现更可靠的场景感知。


### 6.3 REAL ROBOT DEMONSTRATIONS
### 6.3 真实机器人演示


Setup. To validate the effectiveness of our model in real-world settings, we deploy on the RobotEra Q5, a bimanual humanoid platform with a mobile base. An Intel RealSense D455 camera is mounted to enhance RGB-D perception. Importantly, all evaluation scenes are unseen, ensuring that performance reflects true generalization.
设置。为验证我们模型在真实环境中的有效性，我们部署在 RobotEra Q5 上，这是一款带移动底座的双臂类人平台。我们安装了 Intel RealSense D455 摄像头以增强 RGB-D 感知。重要的是，所有评估场景均为未见过的场景，从而确保性能体现真正的泛化能力。


Tasks. We design four representative tasks (Figure 5), consisting of two local interactions (e.g., opening a cabinet, opening a microwave) and two remote interactions (e.g., turning on the TV, turning off a light).
任务。我们设计了四个具有代表性的任务（图 5），包括两个局部交互（例如，打开柜子、打开微波炉）和两个远程交互（例如，打开电视、关掉灯光）。


Deployment. Prior to execution, the robot performs active perception by adjusting its head pose to acquire multi-view observations. MomaGraph-R1 processes these observations together with the task instruction to generate a task-specific subgraph, which explicitly encodes the relevant objects and their spatial-functional relationships, see more deployment details in B.3. Following the Graph-then-Plan paradigm, MomaGraph-R1 then functions as a task planner, producing a structured action sequence. These specifications are subsequently instantiated as low-level trajectories through a library of parameterized primitive skills. We note that the primitive skills are task-specific and derived from teleoperation data for each scenario; the primary contribution of this work lies in the high-level planning and scene graph generating enabled by MomaGraph-R1.
部署。在执行之前，机器人通过主动感知来调整头部姿态，以获取多视角观测。MomaGraph-R1 将这些观测与任务指令一并处理，生成一个面向任务的子图，该子图会明确编码相关物体以及它们的空间-功能关系；更多部署细节见 B.3。遵循“图到计划”的范式，MomaGraph-R1 随后作为任务规划器，生成结构化的动作序列。随后，这些规格会通过一个参数化原始技能库被具体化为低层轨迹。我们指出：这些原始技能是针对任务的，并由每个场景的遥操作数据获得。本工作的主要贡献在于 MomaGraph-R1 所实现的高层规划与场景图生成。


Summary. Our real-world evaluations show that MomaGraph-R1 delivers robust scene understanding and task planning even in unseen scenarios, while remaining directly compatible with standard mobile humanoid systems. This combination underscores the strength of our model and its practicality for real-world deployment.
总结。我们的真实环境评估表明，即便在未见过的场景中，MomaGraph-R1 也能提供稳健的场景理解与任务规划，同时还能与标准移动类人系统保持直接兼容。该组合凸显了我们模型的优势，并证明其适用于真实场景部署。


### 6.4 QUANTITATIVE REAL-ROBOT EVALUATION
### 6.4 定量真实机器人评估


To provide rigorous quantitative validation of our system's robustness, we conduct a comprehensive evaluation on a complex multi-step long-horizon task. This evaluation includes success rates and failure analysis across different stages to validate overall system performance under realistic, sequential conditions (see Figure 6).
为对系统稳健性提供严格的定量验证，我们在一个复杂的多步骤长时程任务上进行了全面评估。该评估包括不同阶段的成功率和失败分析，以在真实的顺序条件下验证整体系统性能（见图6）。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_66d47a.jpg"/>



Figure 6: Quantitative real-robot evaluation. (a) Environment setup of the real-robot experiment. (b) Failure analysis illustrating success/failure rates across different reasoning stages.
图6：定量真实机器人评估。（a）真实机器人实验的环境设置。（b）失败分析，展示不同推理阶段的成功/失败率。


Task Setup. We evaluate the following natural language instruction that requires sequential reasoning and manipulation: "I need better lighting. Turn on the light closest to the remote so I can find it and turn on the monitor to watch." To assess system robustness, we conducted 10 experimental trials, changing the camera viewpoint in each trial.
任务设置。我们评估以下需要按步骤推理与操作的自然语言指令：“我需要更好的照明。把离遥控器最近的灯打开，这样我能找到它并打开显示器来观看。”为评估系统的鲁棒性，我们进行了10次实验，每次都改变摄像头视角。


This task requires spatial reasoning (finding the switch and the remote), functional understanding (linking switches, lights, remote, monitor), and state-dependent planning (lighting affects perception). Additionally, there's object uncertainty (multiple similar lamps or switches), complex spatial relations between objects, and sequential manipulation under partial observability.
此任务需要空间推理（找到开关和遥控器）、功能理解（将开关、灯、遥控器与监视器对应起来）以及与状态相关的规划（照明会影响感知）。此外，还存在物体不确定性（多个相似的灯或开关）、物体之间复杂的空间关系，以及在部分可观测条件下的序列式操作。


Results. As shown in Figure 6, our system achieves an 80% success rate in graph generation, 87.5% success rate in planning (conditioned on correct graphs), and an overall task success rate of 70% over 10 trials.
结果。如图6所示，我们的系统在图生成上达到80%的成功率，在规划（以图正确为前提）上达到87.5%的成功率，并在10次试验中实现了70%的总体任务成功率。


The main failure modes were: (1)spatial relation errors or missing nodes during graph generation; and (2) action sequencing errors in the planning phase, suggesting that the system sometimes plans the right actions but in a suboptimal order.
主要的故障模式有：(1)在生成图时出现空间关系错误或节点缺失；以及(2)在规划阶段的动作衔接错误，这表明系统有时能规划出正确动作，但顺序不够最优。


These results demonstrate that MomaGraph remains robust across multiple reasoning and execution stages, achieving a 70% overall success rate on a complex multi-step task. This validates the system's reliability under realistic long-horizon conditions where errors can compound across stages.
这些结果表明，MomaGraph 在多个推理与执行阶段仍能保持鲁棒性，针对复杂的多步骤任务实现了 70% 的总体成功率。这验证了在真实的长时域条件下系统的可靠性，因为在各阶段出现的错误可能会相互累积。


## 7 CONCLUSION
## 7 结论


This work addresses to the fundamental limitations of existing scene graphs for embodied agents: reliance on a single type of relationship, inability to adapt to dynamic environments, and lack of task relevance. To overcome these issues, we introduce MomaGraph, a novel scene representation that unifies spatial and functional scene graphs with interactive elements. To learn this representation, we construct a large-scale dataset MomaGraph-Scenes and propose MomaGraph-R1, a 7B VLM trained with reinforcement learning, which predicts task-oriented scene graphs and serves as a zero-shot task planner under a Graph-then-Plan framework. Furthermore, we design the MomaGraph-Bench, a comprehensive benchmark that rigorously evaluates both fine-grained reasoning and high-level planning. Through extensive experiments, we demonstrate that our approach achieves state-of-the-art performance among open source models, remains competitive with closed source systems, and transfers effectively to public benchmarks and real robot experiments. We hope that MomaGraph will serve as a foundation for advancing scene representations, fostering stronger connections between the spatial VLM and robotics communities, and ultimately enabling more intelligent and adaptive embodied agents.
本工作针对具身智能体现有场景图的根本局限：依赖单一关系类型、无法适应动态环境，以及缺乏任务相关性。为解决这些问题，我们提出了 MomaGraph，一种将空间场景图与功能场景图及交互元素统一起来的新型场景表示。为学习这一表示，我们构建了大规模数据集 MomaGraph-Scenes，并提出 MomaGraph-R1，这是一款在强化学习下训练的 7B VLM，可预测面向任务的场景图，并在 Graph-then-Plan 框架下充当零样本任务规划器。此外，我们设计了 MomaGraph-Bench，一个全面的基准，严格评估细粒度推理与高层规划。通过大量实验，我们证明我们的方法在开源模型中达到了最先进性能，与闭源系统相比也具有竞争力，并且能有效迁移到公开基准和真实机器人实验中。我们希望 MomaGraph 能成为推进场景表示发展的基础，促进空间 VLM 与机器人社区更紧密的联系，并最终使具身智能体更加智能、更加自适应。


## 8 ACKNOWLEDGEMENTS
## 8 致谢


We would like to express our heartfelt thanks to Chenyangguang Zhang, Prof. Florian Shkurti, and Prof. Tom Silver for their insightful suggestions and constructive feedback. We also thank Guowei Zhang, Yuman Gao, Bike Zhang, Gechen Qu, Lihan Zha, Yuanhang Zhang, and Yu Qi for their valuable assistance in the collection of benchmark data. We thank Robot Era for providing their Q5 Mobile Manipulator for our experiments.
我们衷心感谢张晨阳光、舒尔基教授和汤·西尔弗教授提出的深刻建议以及建设性的反馈。也感谢张国伟、高清曼、张斌、屈格晨、查丽涵、张元航以及齐宇为基准数据的收集提供了宝贵帮助。感谢 Robot Era 为我们的实验提供其 Q5 移动机械臂。


Liang, Lee and Huang are supported by DARPA Transfer from Imprecise and Abstract Models to Autonomous Technologies (TIAMAT) 80321, DARPA HR001124S0029-AIQ-FP-019, DOD-AFOSR-Air Force Office of Scientific Research under award number FA9550-23-1-0048, National Science Foundation NSF-IIS-2147276 FAI, National Science Foundation NAIRR240045, National Science Foundation TRAILS Institute (2229885). Private support was provided by Peraton and Open Philanthropy.
梁、李和黄得到了 DARPA“从不精确与抽象模型到自主技术”（TIAMAT）80321、DARPA HR001124S0029-AIQ-FP-019、国防部-AFOSR（美国空军科学研究办公室）在奖励编号 FA9550-23-1-0048 下的资助，以及国家科学基金会 NSF-IIS-2147276 FAI、国家科学基金会 NAIRR240045、国家科学基金会 TRAILS 研究所（2229885）的支持。私人支持来自 Peraton 和 Open Philanthropy。


The work by Ju, Wang, and Sreenath was supported by The Robotics and AI Institute.
朱、王和斯里纳斯的工作得到了机器人与人工智能研究所（The Robotics and AI Institute）的支持。


## REFERENCES
## 参考文献


Christopher Agia, Krishna Murthy Jatavallabhula, Mohamed Khodeir, Ondrej Miksik, Vibhav Vi-neet, Mustafa Mukadam, Liam Paull, and Florian Shkurti. Taskography: Evaluating robot task planning over large 3d scene graphs. In Conference on Robot Learning, pp. 46-58. PMLR, 2022.
Christopher Agia, Krishna Murthy Jatavallabhula, Mohamed Khodeir, Ondrej Miksik, Vibhav Vi-neet, Mustafa Mukadam, Liam Paull 和 Florian Shkurti。《Taskography：在大型三维场景图上评估机器人任务规划》。载于机器人学习会议（Conference on Robot Learning），第46-58页。PMLR，2022。


Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.
Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman 等。《照我做的，而不是照我说的：在机器人可操作性中实现语言落地》。arXiv预印本 arXiv:2204.01691，2022。


Iro Armeni, Zhi-Yang He, JunYoung Gwak, Amir R Zamir, Martin Fischer, Jitendra Malik, and Silvio Savarese. 3d scene graph: A structure for unified semantics, 3d space, and camera. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5664-5673, 2019.
Iro Armeni, Zhi-Yang He, JunYoung Gwak, Amir R Zamir, Martin Fischer, Jitendra Malik 和 Silvio Savarese。《3d场景图：统一语义、三维空间与相机的一种结构》。载于IEEE/CVF计算机视觉国际会议论文集，第5664-5673页，2019。


Yang Cao, Yuanliang Jv, and Dan Xu. 3dgs-det: Empower 3d gaussian splatting with boundary guidance and box-focused sampling for 3d object detection. arXiv preprint arXiv:2410.01647, 2024.
Yang Cao, Yuanliang Jv 和 Dan Xu。《3dgs-det：通过边界引导与面向3d目标检测的框聚焦采样来赋能3d高斯样条》。arXiv预印本 arXiv:2410.01647，2024。


Jason da Silva Castanheira, Nicholas Shea, and Stephen M Fleming. How attention simplifies mental representations for planning. arXiv preprint arXiv:2506.09520, 2025.
Jason da Silva Castanheira, Nicholas Shea 和 Stephen M Fleming。《注意力如何让规划的心智表征变得更简单》。arXiv预印本，arXiv:2506.09520，2025。


Zhirui Dai, Arash Asgharivaskasi, Thai Duong, Shusen Lin, Maria-Elizabeth Tzes, George Pappas, and Nikolay Atanasov. Optimal scene graph planning with large language model guidance. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 14062-14069. IEEE, 2024.
Zhirui Dai, Arash Asgharivaskasi, Thai Duong, Shusen Lin, Maria-Elizabeth Tzes, George Pappas 和 Nikolay Atanasov。《利用大语言模型引导的最优场景图规划》。载于2024年IEEE机器人与自动化国际会议（ICRA），第14062-14069页。IEEE，2024。


Alexandros Delitzas, Ayca Takmaz, Federico Tombari, Robert Sumner, Marc Pollefeys, and Francis Engelmann. SceneFun3D: Fine-Grained Functionality and Affordance Understanding in 3D Scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
Alexandros Delitzas, Ayca Takmaz, Federico Tombari, Robert Sumner, Marc Pollefeys 和 Francis Engelmann。《SceneFun3D：三维场景中的精细功能性与可操作性理解》。载于IEEE/CVF计算机视觉与模式识别会议（CVPR），2024。


Ang Dong, Li Feng, Dengcheng Yang, Shuang Wu, Jinshuai Zhao, Jing Wang, and Rongling Wu. Fungraph: A statistical protocol to reconstruct omnigenic multilayer interactome networks for complex traits. Star Protocols, 2(4):100985, 2021.
Ang Dong, Li Feng, Dengcheng Yang, Shuang Wu, Jinshuai Zhao, Jing Wang 和 Rongling Wu。《Fungraph：用于为复杂性状重建全基因组多层相互作用网络的统计协议》。Star Protocols，第2卷（第4期）：100985，2021。


Daniel Ekpo, Mara Levy, Saksham Suri, Chuong Huynh, and Abhinav Shrivastava. Verigraph: Scene graphs for execution verifiable robot planning. arXiv preprint arXiv:2411.10446, 2024.
Daniel Ekpo, Mara Levy, Saksham Suri, Chuong Huynh 和 Abhinav Shrivastava。《Verigraph：用于可验证执行的场景图机器人规划》。arXiv预印本 arXiv:2411.10446，2024。


Tim Engelbracht, René Zurbrügg, Marc Pollefeys, Hermann Blum, and Zuria Bauer. Spotlight: Robotic scene understanding through interaction and affordance detection. arXiv preprint arXiv:2409.11870, 2024.
Tim Engelbracht, René Zurbrügg, Marc Pollefeys, Hermann Blum 和 Zuria Bauer。《Spotlight：通过交互与可操作性检测实现机器人场景理解》。arXiv预印本 arXiv:2409.11870，2024。


Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pp. 148-166. Springer, 2024.
Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma 和 Ranjay Krishna。《Blink：多模态大语言模型看得见，但感知不到》。载于欧洲计算机视觉会议（European Conference on Computer Vision），第148-166页。Springer，2024。


Elias Greve, Martin Büchner, Niclas Vödisch, Wolfram Burgard, and Abhinav Valada. Collaborative dynamic 3d scene graphs for automated driving. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 11118-11124. IEEE, 2024.
Elias Greve, Martin Büchner, Niclas Vödisch, Wolfram Burgard 和 Abhinav Valada。《用于自动驾驶的协作动态三维场景图》。载于2024年IEEE机器人与自动化国际会议（ICRA），第11118-11124页。IEEE，2024。


Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, et al. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 5021-5028. IEEE, 2024a.
Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa 等。《Conceptgraphs：面向感知与规划的开放词汇三维场景图》。载于2024年IEEE机器人与自动化国际会议（ICRA），第5021-5028页。IEEE，2024a。


Qiao Gu, Zhaoyang Lv, Duncan Frost, Simon Green, Julian Straub, and Chris Sweeney. Egolifter: Open-world 3d segmentation for egocentric perception. In European Conference on Computer Vision, pp. 382-400, 2024b.
Qiao Gu, Zhaoyang Lv, Duncan Frost, Simon Green, Julian Straub 和 Chris Sweeney。《Egolifter：面向自我视角感知的开放世界三维分割》。载于欧洲计算机视觉会议（European Conference on Computer Vision），第382-400页，2024b。


Yanjiang Guo, Yen-Jen Wang, Lihan Zha, and Jianyu Chen. Doremi: Grounding language model by detecting and recovering from plan-execution misalignment. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 12124-12131. IEEE, 2024.
Yanjiang Guo，Yen-Jen Wang，Lihan Zha，和 Jianyu Chen。Doremi：通过检测并恢复计划-执行不一致来为语言模型提供基础。在 2024 年 IEEE/RSJ 国际智能机器人与系统会议（IROS）中，第 12124-12131 页。IEEE，2024。


Daniel Honerkamp, Martin Büchner, Fabien Despinoy, Tim Welschehold, and Abhinav Valada. Language-grounded dynamic scene graphs for interactive object search with mobile manipulation. IEEE Robotics and Automation Letters, 2024a.
Daniel Honerkamp，Martin Büchner，Fabien Despinoy，Tim Welschehold，和 Abhinav Valada。用于移动操作的交互式物体搜索的语言引导动态场景图。IEEE 机器人与自动化字母期刊，2024a。


Daniel Honerkamp, Martin Büchner, Fabien Despinoy, Tim Welschehold, and Abhinav Valada. Language-grounded dynamic scene graphs for interactive object search with mobile manipulation. IEEE Robotics and Automation Letters, 2024b.
Daniel Honerkamp，Martin Büchner，Fabien Despinoy，Tim Welschehold，和 Abhinav Valada。用于移动操作的交互式物体搜索的语言引导动态场景图。IEEE 机器人与自动化字母期刊，2024b。


Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023.
Wenlong Huang，Chen Wang，Ruohan Zhang，Yunzhu Li，Jiajun Wu，和 Li Fei-Fei。Voxposer：用于语言模型驱动机器人的可组合 3d 价值图。arXiv 预印本 arXiv:2307.05973，2023。


Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatiotemporal reasoning of relational keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652, 2024.
Wenlong Huang，Chen Wang，Yunzhu Li，Ruohan Zhang，和 Li Fei-Fei。Rekep：用于机器人操作的关系关键点约束的时空推理。arXiv 预印本 arXiv:2409.01652，2024。


Nathan Hughes, Yun Chang, Siyi Hu, Rajat Talak, Rumaia Abdulhai, Jared Strader, and Luca Car-lone. Foundations of spatial perception for robotics: Hierarchical representations and real-time systems. The International Journal of Robotics Research, 43(10):1457-1505, 2024.
Nathan Hughes，Yun Chang，Siyi Hu，Rajat Talak，Rumaia Abdulhai，Jared Strader，和 Luca Car-lone。机器人空间感知的基础：分层表示与实时系统。《国际机器人研究期刊》，43(10)：1457-1505，2024。


Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd. Omama, Ganesh Iyer, Soroush Saryazdi, Tao Chen, Alaa Maalouf, Shuang Li, Nikhil Varma Keetha, Ayush Tewari, Joshua B. Tenenbaum, Celso Miguel de Melo, K. Madhava Krishna, Liam Paull, Florian Shkurti, and Antonio Torralba. Conceptfusion: Open-set multimodal 3d mapping. In Robotics: Science and Systems, 2023.
Krishna Murthy Jatavallabhula，Alihusein Kuwajerwala，Qiao Gu，Mohd. Omama，Ganesh Iyer，Soroush Saryazdi，Tao Chen，Alaa Maalouf，Shuang Li，Nikhil Varma Keetha，Ayush Tewari，Joshua B. Tenenbaum，Celso Miguel de Melo，K. Madhava Krishna，Liam Paull，Florian Shkurti，和 Antonio Torralba。Conceptfusion：开放集多模态 3d 映射。在 Robotics: Science and Systems，2023。


Guangqi Jiang, Yifei Sun, Tao Huang, Huanyu Li, Yongyuan Liang, and Huazhe Xu. Robots pretrain robots: Manipulation-centric robotic representation from large-scale robot datasets. 2025.
Guangqi Jiang，Yifei Sun，Tao Huang，Huanyu Li，Yongyuan Liang，和 Huazhe Xu。让机器人预训练机器人：来自大规模机器人数据集的以操作为中心的机器人表征。2025。


Hanxiao Jiang, Binghao Huang, Ruihai Wu, Zhuoran Li, Shubham Garg, Hooshang Nayyeri, Shen-long Wang, and Yunzhu Li. Roboexp: Action-conditioned scene graph via interactive exploration for robotic manipulation. arXiv preprint arXiv:2402.15487, 2024.
Hanxiao Jiang，Binghao Huang，Ruihai Wu，Zhuoran Li，Shubham Garg，Hooshang Nayyeri，Shen-long Wang，和 Yunzhu Li。Roboexp：通过交互式探索生成动作条件化的场景图，用于机器人操作。arXiv 预印本 arXiv:2402.15487，2024。


Yuanchen Ju, Kaizhe Hu, Guowei Zhang, Gu Zhang, Mingrun Jiang, and Huazhe Xu. Robo-abc: Affordance generalization beyond categories via semantic correspondence for robot manipulation. In European Conference on Computer Vision (ECCV), 2024.
Yuanchen Ju，Kaizhe Hu，Guowei Zhang，Gu Zhang，Mingrun Jiang，和 Huazhe Xu。Robo-abc：借助语义对应实现超越类别的可用性泛化，用于机器人操作。在欧洲计算机视觉会议（ECCV），2024。


Sebastian Koch, Pedro Hermosilla, Narunas Vaskevicius, Mirco Colosi, and Timo Ropinski. Lang3dsg: Language-based contrastive pre-training for 3d scene graph prediction. In 2024 International Conference on 3D Vision (3DV), pp. 1037-1047. IEEE, 2024a.
Sebastian Koch，Pedro Hermosilla，Narunas Vaskevicius，Mirco Colosi，和 Timo Ropinski。Lang3dsg：用于 3d 场景图预测的基于语言的对比式预训练。在 2024 年国际 3D 视觉会议（3DV）中，第 1037-1047 页。IEEE，2024a。


Sebastian Koch, Narunas Vaskevicius, Mirco Colosi, Pedro Hermosilla, and Timo Ropinski. Open3dsg: Open-vocabulary 3d scene graphs from point clouds with queryable objects and open-set relationships. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14183-14193, 2024b.
Sebastian Koch，Narunas Vaskevicius，Mirco Colosi，Pedro Hermosilla，和 Timo Ropinski。Open3dsg：从点云中获得可查询对象与开放集关系的开放词汇 3d 场景图。在 IEEE/CVF 计算机视觉与模式识别会议论文集，第 14183-14193 页，2024b。


Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474, 2017.
Eric Kolve，Roozbeh Mottaghi，Winson Han，Eli VanderBilt，Luca Weihs，Alvaro Herrasti，Matt Deitke，Kiana Ehsani，Daniel Gordon，Yuke Zhu，等。Ai2-thor：面向视觉 AI 的交互式 3d 环境。arXiv 预印本 arXiv:1712.05474，2017。


Vasiliki Kondyli, Mehul Bhatt, and Jakob Suchan. Towards a human-centred cognitive model of visuospatial complexity in everyday driving. arXiv preprint arXiv:2006.00059, 2020.
Vasiliki Kondyli，Mehul Bhatt，和 Jakob Suchan。迈向以人为中心的认知模型：日常驾驶中视觉空间复杂性的刻画。arXiv 预印本 arXiv:2006.00059，2020。


Qi Li, Kaichun Mo, Yanchao Yang, Hang Zhao, and Leonidas Guibas. Ifr-explore: Learning inter-object functional relationships in 3d indoor scenes. arXiv preprint arXiv:2112.05298, 2021.
Qi Li, Kaichun Mo, Yanchao Yang, Hang Zhao, 和 Leonidas Guibas。Ifr-explore：学习 3D 室内场景中的对象间功能关系。arXiv 预印本 arXiv:2112.05298, 2021。


Yongyuan Liang, Tingqiang Xu, Kaizhe Hu, Guangqi Jiang, Furong Huang, and Huazhe Xu. Make-an-agent: A generalizable policy network generator with behavior-prompted diffusion. 2024.
Yongyuan Liang, Tingqiang Xu, Kaizhe Hu, Guangqi Jiang, Furong Huang, 和 Huazhe Xu。Make-an-agent：一种带有行为提示扩散的可泛化策略网络生成器。2024。


Yongyuan Liang, Wei Chow, Feng Li, Ziqiao Ma, Xiyao Wang, Jiageng Mao, Jiuhai Chen, Jiatao Gu, Yue Wang, and Furong Huang. Rover: Benchmarking reciprocal cross-modal reasoning for omnimodal generation. arXiv preprint arXiv:2511.01163, 2025a.
Yongyuan Liang, Wei Chow, Feng Li, Ziqiao Ma, Xiyao Wang, Jiageng Mao, Jiuhai Chen, Jiatao Gu, Yue Wang, 和 Furong Huang。Rover：面向全模态生成的互惠跨模态推理基准。arXiv 预印本 arXiv:2511.01163, 2025a。


Yongyuan Liang, Xiyao Wang, Yuanchen Ju, Jianwei Yang, and Furong Huang. Lemon: A unified and scalable 3d multimodal model for universal spatial understanding. arXiv preprint arXiv:2512.12822, 2025b.
Yongyuan Liang, Xiyao Wang, Yuanchen Ju, Jianwei Yang, 和 Furong Huang。Lemon：一种统一且可扩展的 3D 多模态模型，用于通用空间理解。arXiv 预印本 arXiv:2512.12822, 2025b。


Joel Loo, Zhanxin Wu, and David Hsu. Open scene graphs for open-world object-goal navigation. arXiv preprint arXiv:2508.04678, 2025.
Joel Loo, Zhanxin Wu, 和 David Hsu。面向开放世界物体目标导航的开放场景图。arXiv 预印本 arXiv:2508.04678, 2025。


Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, 和 Jianfeng Gao。Mathvista：在视觉语境中评估基础模型的数学推理能力。arXiv 预印本 arXiv:2310.02255, 2023。


Dominic Maggio, Yun Chang, Nathan Hughes, Matthew Trang, Dan Griffith, Carlyn Dougherty, Eric Cristofalo, Lukas Schmid, and Luca Carlone. Clio: Real-time task-driven open-set 3d scene graphs. IEEE Robotics and Automation Letters, 2024.
Dominic Maggio, Yun Chang, Nathan Hughes, Matthew Trang, Dan Griffith, Carlyn Dougherty, Eric Cristofalo, Lukas Schmid, 和 Luca Carlone。Clio：实时任务驱动的开放集 3D 场景图。IEEE Robotics and Automation Letters, 2024。


Dantong Niu, Yuvan Sharma, Giscard Biamby, Jerome Quenum, Yutong Bai, Baifeng Shi, Trevor Darrell, and Roei Herzig. Llarva: Vision-action instruction tuning enhances robot learning. arXiv preprint arXiv:2406.11815, 2024.
Dantong Niu, Yuvan Sharma, Giscard Biamby, Jerome Quenum, Yutong Bai, Baifeng Shi, Trevor Darrell, 和 Roei Herzig。Llarva：视觉-动作指令微调增强机器人学习。arXiv 预印本 arXiv:2406.11815, 2024。


OpenAI. Gpt-4 technical report. Technical report, OpenAI, 2023. URL https://api.semanticscholar.org/CorpusID:257532815.
OpenAI。Gpt-4 技术报告。技术报告，OpenAI，2023。URL https://api.semanticscholar.org/CorpusID:257532815。


Yu Qi, Yuanchen Ju, Tianming Wei, Chi Chu, Lawson LS Wong, and Huazhe Xu. Two by two: Learning multi-task pairwise objects assembly for generalizable robot manipulation. CVPR 2025, 2025.
Yu Qi, Yuanchen Ju, Tianming Wei, Chi Chu, Lawson LS Wong, 和 Huazhe Xu。Two by two：学习多任务成对物体组装以实现可泛化机器人操作。CVPR 2025, 2025。


Ri-Zhao Qiu, Yafei Hu, Yuchen Song, Ge Yang, Yang Fu, Jianglong Ye, Jiteng Mu, Ruihan Yang, Nikolay Atanasov, Sebastian Scherer, et al. Learning generalizable feature fields for mobile manipulation. arXiv preprint arXiv:2403.07563, 2024.
Ri-Zhao Qiu, Yafei Hu, Yuchen Song, Ge Yang, Yang Fu, Jianglong Ye, Jiteng Mu, Ruihan Yang, Nikolay Atanasov, Sebastian Scherer, 等。面向移动操作的可泛化特征场学习。arXiv 预印本 arXiv:2403.07563, 2024。


Qwen. Qwen2.5-vl, January 2025. URL https://qwenlm.github.io/blog/qwen2.5-vl/.
Qwen。Qwen2.5-vl，2025 年 1 月。URL https://qwenlm.github.io/blog/qwen2.5-vl/。


Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf. Sayplan: Grounding large language models using 3d scene graphs for scalable robot task planning. arXiv preprint arXiv:2307.06135, 2023.
Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, 和 Niko Suenderhauf。Sayplan：使用 3D 场景图对大型语言模型进行落地，以实现可扩展的机器人任务规划。arXiv 预印本 arXiv:2307.06135, 2023。


Antoni Rosinol, Andrew Violette, Marcus Abate, Nathan Hughes, Yun Chang, Jingnan Shi, Arjun Gupta, and Luca Carlone. Kimera: From slam to spatial perception with 3d dynamic scene graphs. The International Journal of Robotics Research, 40(12-14):1510-1546, 2021.
Antoni Rosinol, Andrew Violette, Marcus Abate, Nathan Hughes, Yun Chang, Jingnan Shi, Arjun Gupta, 和 Luca Carlone。Kimera：从 SLAM 到具有 3D 动态场景图的空间感知。The International Journal of Robotics Research, 40(12-14):1510-1546, 2021。


Ayca Takmaz, Alexandros Delitzas, Robert W Sumner, Francis Engelmann, Johanna Wald, and Federico Tombari. Search3d: Hierarchical open-vocabulary 3d segmentation. IEEE Robotics and Automation Letters, 2025.
Ayca Takmaz, Alexandros Delitzas, Robert W Sumner, Francis Engelmann, Johanna Wald, 和 Federico Tombari。Search3d：层次化开放词汇 3D 分割。IEEE Robotics and Automation Letters, 2025。


Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.
Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, 等。Gemini robotics：将 AI 带入物理世界。arXiv 预印本 arXiv:2503.20020, 2025。


Sebo Uithol, Katherine L Bryant, Ivan Toni, and Rogier B Mars. The anticipatory and task-driven nature of visual perception. Cerebral Cortex, 31(12):5354-5362, 2021.
Sebo Uithol、Katherine L Bryant、Ivan Toni 和 Rogier B Mars。视觉感知的前瞻性与任务驱动特性。Cerebral Cortex，31(12)：5354-5362，2021。


Yixuan Wang, Leonor Fermoselle, Tarik Kelestemur, Jiuguang Wang, and Yunzhu Li. Curious-bot: Interactive mobile exploration via actionable 3d relational object graph. arXiv preprint arXiv:2501.13338, 2025.
Yixuan Wang、Leonor Fermoselle、Tarik Kelestemur、Jiuguang Wang 和 Yunzhu Li。Curious-bot：通过可执行的3d关系物体图进行交互式移动探索。arXiv预印本 arXiv:2501.13338，2025。


Abdelrhman Werby, Chenguang Huang, Martin Büchner, Abhinav Valada, and Wolfram Burgard. Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024.
Abdelrhman Werby、Chenguang Huang、Martin Büchner、Abhinav Valada 和 Wolfram Burgard。面向语言引导机器人导航的分层开放词汇3d场景图。在 ICRA 2024 于 ICRA 2024 举办的“导航与操作的视觉-语言模型”第一届研讨会上，2024。


Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, Andy Zeng, Shuran Song, Jeannette Bohg, Szymon Rusinkiewicz, and Thomas Funkhouser. Tidybot: Personalized robot assistance with large language models. Autonomous Robots, 47(8):1087-1102, 2023.
Jimmy Wu、Rika Antonova、Adam Kan、Marion Lepert、Andy Zeng、Shuran Song、Jeannette Bohg、Szymon Rusinkiewicz 和 Thomas Funkhouser。Tidybot：借助大语言模型实现个性化机器人辅助。Autonomous Robots，47(8)：1087-1102，2023。


Shun-Cheng Wu, Johanna Wald, Keisuke Tateno, Nassir Navab, and Federico Tombari. Scene-graphfusion: Incremental 3d scene graph prediction from rgb-d sequences. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7515-7525, 2021.
Shun-Cheng Wu、Johanna Wald、Keisuke Tateno、Nassir Navab 和 Federico Tombari。Scene-graphfusion：从rgb-d序列进行增量式3d场景图预测。在IEEE/CVF计算机视觉与模式识别会议论文集中，第7515-7525页，2021。


Zhijie Yan, Shufei Li, Zuoxu Wang, Lixiu Wu, Han Wang, Jun Zhu, Lijiang Chen, and Jihong Liu. Dynamic open-vocabulary 3d scene graphs for long-term language-guided mobile manipulation. IEEE Robotics and Automation Letters, 2025.
Zhijie Yan、Shufei Li、Zuoxu Wang、Lixiu Wu、Han Wang、Jun Zhu、Lijiang Chen 和 Jihong Liu。面向长期语言引导移动操作的动态开放词汇3d场景图。IEEE Robotics and Automation Letters，2025。


Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 14203-14214, 2025.
Jianwei Yang、Reuben Tan、Qianhui Wu、Ruijie Zheng、Baolin Peng、Yongyuan Liang、Yu Gu、Mu Cai、Seonghyeon Ye、Joel Jang 等。Magma：多模态AI智能体的基础模型。在计算机视觉与模式识别会议论文集中，第14203-14214页，2025。


Timing Yang, Yuanliang Ju, and Li Yi. Imov3d: Learning open vocabulary point clouds 3d object detection from only 2d images. Advances in Neural Information Processing Systems, 37:141261- 141291, 2024.
Timing Yang、Yuanliang Ju 和 Li Yi。Imov3d：仅从2d图像学习开放词汇点云的3d目标检测。Advances in Neural Information Processing Systems，37：141261-141291，2024。


Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. arXiv preprint arXiv:2506.21458, 2025.
Baiqiao Yin、Qineng Wang、Pingyue Zhang、Jianshu Zhang、Kangrui Wang、Zihan Wang、Jieyu Zhang、Keshigeyan Chandrasegaran、Han Liu、Ranjay Krishna 等。基于有限视角的空间心智建模。arXiv预印本 arXiv:2506.21458，2025。


Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
Qiying Yu、Zheng Zhang、Ruofei Zhu、Yufeng Yuan、Xiaochen Zuo、Yu Yue、Weinan Dai、Tiantian Fan、Gaohong Liu、Lingjun Liu 等。Dapo：面向规模化的开源LLM强化学习系统。arXiv预印本 arXiv:2503.14476，2025。


Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556-9567, 2024.
Xiang Yue、Yuansheng Ni、Kai Zhang、Tianyu Zheng、Ruoqi Liu、Ge Zhang、Samuel Stevens、Dongfu Jiang、Weiming Ren、Yuxuan Sun 等。Mmmu：面向专家AGI的海量跨学科多模态理解与推理基准。在IEEE/CVF计算机视觉与模式识别会议论文集中，第9556-9567页，2024。


Tatiana Zemskova and Dmitry Yudin. 3dgraphllm: Combining semantic graphs and large language models for 3d referred object grounding.
Tatiana Zemskova 和 Dmitry Yudin。3dgraphllm：结合语义图与大语言模型进行3d指代物体定位。


Chenyangguang Zhang, Alexandros Delitzas, Fangjinhua Wang, Ruida Zhang, Xiangyang Ji, Marc Pollefeys, and Francis Engelmann. Open-vocabulary functional 3d scene graphs for real-world indoor spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19401-19413, 2025.
Chenyangguang Zhang、Alexandros Delitzas、Fangjinhua Wang、Ruida Zhang、Xiangyang Ji、Marc Pollefeys 和 Francis Engelmann。面向真实室内空间的开放词汇功能性3d场景图。在计算机视觉与模式识别会议论文集中，第19401-19413页，2025。


Shoulong Zhang, Aimin Hao, Hong Qin, et al. Knowledge-inspired 3d scene graph prediction in point cloud. Advances in Neural Information Processing Systems, 34:18620-18632, 2021.
Shoulong Zhang、Aimin Hao、Hong Qin 等。受知识启发的点云3d场景图预测。Advances in Neural Information Processing Systems，34：18620-18632，2021。


Yuanhang Zhang, Tianhai Liang, Zhenyang Chen, Yanjie Ze, and Huazhe Xu. Catch it! learning to catch in flight with mobile dexterous hands. arXiv preprint arXiv:2409.10319, 2024a.
Yuanhang Zhang、Tianhai Liang、Zhenyang Chen、Yanjie Ze 和 Huazhe Xu。抓住它！用移动灵巧手在空中学会接球。arXiv预印本 arXiv:2409.10319，2024a。


Yunpeng Zhang, Deheng Qian, Ding Li, Yifeng Pan, Yong Chen, Zhenbao Liang, Zhiyao Zhang, Shurui Zhang, Hongxu Li, Maolei Fu, et al. Graphad: Interaction scene graph for end-to-end autonomous driving. arXiv preprint arXiv:2403.19098, 2024b.
Yunpeng Zhang，Deheng Qian，Ding Li，Yifeng Pan，Yong Chen，Zhenbao Liang，Zhiyao Zhang，Shurui Zhang，Hongxu Li，Maolei Fu等。Graphad：面向端到端自动驾驶的交互式场景图。arXiv预印本arXiv:2403.19098，2024b。


Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. 2025a.
Ruijie Zheng，Yongyuan Liang，Shuaiyi Huang，Jianfeng Gao，Hal Daumé III，Andrey Kolobov，Furong Huang，以及 Jianwei Yang。Tracevla：视觉轨迹提示增强通用机器人策略的时空感知。2025a。


Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyrl: An efficient, scalable, multi-modality rl training framework. https://github.com/ hiyouga/EasyR1, 2025b.
Yaowei Zheng，Junting Lu，Shenzhi Wang，Zhangchi Feng，Dongdong Kuang，以及 Yuwen Xiong。Easyrl：一种高效、可扩展的多模态RL训练框架。https://github.com/ hiyouga/EasyR1，2025b。


Junzhe Zhu, Yuanchen Ju, Junyi Zhang, Muhan Wang, Zhecheng Yuan, Kaizhe Hu, and Huazhe Xu. Densematcher: Learning 3d semantic correspondence for category-level manipulation from a single demo. International Conference on Learning Representations (ICLR) Spotlight, 2025.
Junzhe Zhu，Yuanchen Ju，Junyi Zhang，Muhan Wang，Zhecheng Yuan，Kaizhe Hu，以及 Huazhe Xu。Densematcher：从单个演示中学习用于类别级操作的三维语义对应关系。国际学习表征会议（ICLR）Spotlight，2025。


## A APPENDIX
## A 附录


### A.1 MOMAGRAPH-SCENES DATASET
### A.1 MOMAGRAPH-场景数据集


#### A.1.1 REAL-WORLD DATASET SOURCE AND COLLECTION.
#### A.1.1 真实世界数据集来源与采集。


Our dataset is built through a synergistic integration of newly curated data and existing public resources. We manually collected a substantial portion of the data in real-world household environments, capturing diverse interaction scenarios under natural conditions. To further enrich the dataset, we incorporated samples from two public benchmarks, OpenFunGraph (Zhang et al., 2025) and SceneFun3D (Delitzas et al., 2024), both of which contain videos depicting human-object interactions in indoor contexts. From these videos, we carefully curated representative keyframes to derive multi-view RGB observations, ensuring comprehensive coverage of interaction dynamics and spatial variability.
我们的数据集由新策划数据与现有公开资源协同整合构建。我们在真实世界的家庭环境中手动收集了数据的相当一部分，在自然条件下捕捉多样的交互场景。为进一步丰富数据集，我们加入了两个公开基准的样本：OpenFunGraph（Zhang et al., 2025）和SceneFun3D（Delitzas et al., 2024），两者均包含室内环境中人体-物体交互的视频。我们从这些视频中精心挑选具有代表性的关键帧，以生成多视角RGB观测，确保对交互动态与空间变化的全面覆盖。


#### A.1.2 Simulation Data Collection
#### A.1.2 仿真数据采集


To complement the real-world data, we additionally generated samples within the AI2-THOR simulation environment Kolve et al. (2017). We strategically positioned the embodied agent at diverse, reachable viewpoints and captured multi-view observations from varying perspectives, as illustrated in Fig. 7. Throughout this process, we applied manual post-filtering to exclude non-interactable elements, thereby ensuring that the curated dataset remains focused on actionable objects and emphasizes functional relevance critical for downstream embodied reasoning tasks.
为补充真实数据，我们还在 AI2-THOR 仿真环境 Kolve et al. (2017) 中额外生成了样本。我们将具身智能体有策略地放置在不同的、可到达的视点，并从多种视角获取多视图观测，如图 7 所示。整个过程中，我们进行了人工后筛选，以剔除不可交互的元素，从而确保整理后的数据集聚焦于可执行目标，并突出对下游具身推理任务至关重要的功能相关性。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_05828e.jpg"/>



Figure 7: Simulated indoor environments in our benchmark. Each row shows three scenes (Floor 15, Floor 224 and Floor 301) with a top-down view of the layout, reachable locations for the robot, and multiview observations from different viewpoints.
图 7：我们基准中的仿真实内环境。每一行展示三个场景（Floor 15、Floor 224 和 Floor 301），包含俯视布局图、机器人可到达位置，以及来自不同视点的多视图观测。


#### A.1.3 DATASET ANNOTATION AND FORMAT.
#### A.1.3 数据集标注与格式。


Annotation and Format. Each task-oriented subgraph in MomaGraph-Scenes is stored in a structured JSON format and linked to its corresponding scene. Annotations include a subgraph identifier, the associated scene identifier, the action type, the functional category, the natural language task instruction, a set of nodes, and a set of edges. Nodes correspond to the objects or part-level interactive elements required to accomplish the task, while edges capture both functional relationships (e.g., control, open or close) and spatial relationships (e.g., close, in_front_of, lower_than).
标注与格式。MomaGraph-Scenes 中每个面向任务的子图都以结构化 JSON 格式存储，并与其对应场景关联。标注包括子图标识、关联场景标识、动作类型、功能类别、自然语言任务指令、一组节点和一组边。节点对应完成任务所需的对象或部件级交互元素，而边同时捕捉功能关系（如控制、打开或关闭）和空间关系（如靠近、在前方、低于）。


This example corresponds to the instruction "Turn on the television", where the relevant nodes are the remote control and the ${TV}$ ,connected by a control functional edge and spatial relations lower_than, in_front_of, and close.
此示例对应指令“打开电视”，其中相关节点是遥控器和${TV}$，它们通过一条控制功能边以及 lower_than、in_front_of 和 close 空间关系相连。


In addition, each subgraph is grounded in multi-view observations. For every scene, we provide synchronized RGB images captured from multiple viewpoints. This multi-view grounding allows the annotated subgraphs to be consistently aligned with visual evidence, supporting both instruction-conditioned graph prediction from perception and multi-view reasoning tasks.
此外，每个子图都基于多视角观测。对于每个场景，我们提供从多个视角拍摄的同步 RGB 图像。这种多视角对齐使标注子图能够与视觉证据一致对齐，支持从感知进行指令条件图预测以及多视角推理任务。


---



&nbsp;&nbsp;&nbsp;&nbsp;"subgraph_id": "da21b9f9-f4fa-4a85-961b-2e2c2e182d3e",
&nbsp;&nbsp;&nbsp;&nbsp;"subgraph_id": "da21b9f9-f4fa-4a85-961b-2e2c2e182d3e",


&nbsp;&nbsp;&nbsp;&nbsp;"scene_id": "466828",
&nbsp;&nbsp;&nbsp;&nbsp;"scene_id": "466828",


&nbsp;&nbsp;&nbsp;&nbsp;"action_type": "press",
&nbsp;&nbsp;&nbsp;&nbsp;"action_type": "press",


&nbsp;&nbsp;&nbsp;&nbsp;"function_type": "device_control",
&nbsp;&nbsp;&nbsp;&nbsp;"function_type": "device_control",


&nbsp;&nbsp;&nbsp;&nbsp;"task_instruction": "Turn on the television.",
&nbsp;&nbsp;&nbsp;&nbsp;"task_instruction": "打开电视。",


&nbsp;&nbsp;&nbsp;&nbsp;"nodes": [
&nbsp;&nbsp;&nbsp;&nbsp;"nodes": [


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"label": "remote control", "id": "f15474de-7b35-4a5e-ac8a-dc02f93960b3"\},
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"label": "遥控器", "id": "f15474de-7b35-4a5e-ac8a-dc02f93960b3"\},


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"label": "tv", "id": "91486017-94ce-4788-aabd-0d07262c9bed"\}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"label": "电视", "id": "91486017-94ce-4788-aabd-0d07262c9bed"\}


&nbsp;&nbsp;&nbsp;&nbsp;],



&nbsp;&nbsp;&nbsp;&nbsp;"edges": [
&nbsp;&nbsp;&nbsp;&nbsp;"edges": [


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"relation_id": "ef3e72fe-ae9f-42e4-9b5a-505b5cb1844a",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"relation_id": "ef3e72fe-ae9f-42e4-9b5a-505b5cb1844a",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"functional_relationship": "control",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"functional_relationship": "control",


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"object1": \{"label": "remote control", "id": "f15474de-7b35-4a5e-ac8a-dc02f93960b3"\},
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"object1": \{"label": "遥控器", "id": "f15474de-7b35-4a5e-ac8a-dc02f93960b3"\},


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"object2": \{"label": "tv", "id": "91486017-94ce-4788-aabd-0d07262c9bed"\},
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"object2": \{"label": "电视", "id": "91486017-94ce-4788-aabd-0d07262c9bed"\},


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"spatial_relations": ["lower_than", "in_front_of", "close"],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"spatial_relations": ["低于", "在前方", "近距离"],


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"is_touching": false
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"is_touching": false


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\}



&nbsp;&nbsp;&nbsp;&nbsp;]



\}



---



Figure 8: Example JSON annotation for the task "Turn on the television."
Figure 8：任务“打开电视”的示例 JSON 标注。


#### A.1.4 MULTI-ASPECT STATISTICS OF THE TRAINING DATASET
#### A.1.4 训练数据集的多方面统计


Our dataset consists of approximately 1,050 subgraphs and 6278 multi-view RGB images, collected across more than 350 diverse household scenes and encompassing 93 distinct task instructions. This broad coverage ensures rich variability in scene layouts, object configurations, and interaction types.
我们的数据集包含约 1,050 个子图和 6278 张多视角 RGB 图像，采集于超过 350 个多样化的家庭场景，并覆盖 93 条不同的任务指令。如此广泛的覆盖确保了场景布局、物体配置和交互类型的丰富多样性。


To provide a comprehensive overview of our training data, we present multi-aspect statistics covering scene context, action diversity, functional relationships, and object distributions. As shown in Fig. 9, the dataset spans four common household room types and captures the correspondence between action types and functional categories, reflecting the diversity and richness of real-world manipulation scenarios. Fig. 10 illustrates the distribution of action types across different room contexts, while Fig. 11 summarizes the prevalence of various functional relationships and Fig. 12 summarizes the frequency of object occurrences. Together, these statistics highlight the diversity and task relevance of our dataset, ensuring broad coverage of spatial-functional interactions essential for embodied planning and reasoning.
为全面概述我们的训练数据，我们给出涵盖场景语境、动作多样性、功能关系以及物体分布的多方面统计。如图 9 所示，数据集跨越四种常见的家庭房间类型，并刻画了动作类型与功能类别之间的对应关系，体现了真实操作场景的多样性与丰富性。图 10 展示了不同房间语境下动作类型的分布；图 11 汇总了各类功能关系的普遍程度；图 12 汇总了物体出现的频率。总体而言，这些统计结果凸显了数据集的多样性与任务相关性，确保覆盖了对具身规划与推理至关重要的空间-功能交互。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_35c01e.jpg"/>



Figure 9: Dataset statistics: (a) Distribution across four room types; (b)Heatmap showing the correspondence between action types and functional types.
图 9：数据集统计：（a）在四种房间类型中的分布；（b）热力图展示动作类型与功能类型之间的对应关系。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_e9eddd.jpg"/>



Figure 10: Task distribution across four room types: kitchen, living room, bedroom, and bathroom.
图 10：四种房间类型下的任务分布：厨房、客厅、卧室和浴室。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_d07a3d.jpg"/>



Figure 11: Distribution of functional relationships across all tasks in the dataset.
图 11：数据集中所有任务的功能关系分布。


### A.2 TRAINING DETAILS
### A.2 训练细节


We train our model using 8x 80GB A100 GPUs for approximately 13 hours based on the EasyR1 (Zheng et al., 2025b) training framework. The complete training configuration for DAPO algorithm is presented in Table 4.
我们基于 EasyR1（Zheng et al., 2025b）训练框架，使用 8 块 80GB A100 GPU 训练模型，耗时约 13 小时。DAPO 算法的完整训练配置见表 4。


### A.3 TRAINING CURVE
### A.3 训练曲线


Figure 13 and 14 shows the training curves during DAPO optimization. The training and validation curves closely align across all metrics, indicating good generalization without significant overfitting. The overall reward converges to $\sim  {0.93}$ ,while accuracy reward stabilizes at $\sim  {0.9}$ . The format reward quickly reaches 1.0 within the first 25 steps, showing the model rapidly learns to produce valid JSON-structured outputs.
图13和14展示了DAPO优化过程中的训练曲线。训练曲线与验证曲线在所有指标上都高度一致，表明模型具有良好的泛化能力，且没有明显过拟合。总体奖励收敛到$\sim  {0.93}$，而准确性奖励稳定在$\sim  {0.9}$。格式奖励在前25步内迅速达到1.0，表明模型很快学会生成有效的JSON结构输出。


### A.4 MOMAGRAPH BENCHMARK
### A.4 MOMAGRAPH 基准


#### A.4.1 BENCHMARK DESIGN
#### A.4.1 基准设计


To rigorously evaluate spatial-functional reasoning and task planning capabilities, we design a comprehensive multi-choice VQA benchmark based on the scenes and tasks in our dataset. Rather than manually crafting all questions, we leverage large vision-language models (VLMs) to generate them in a scalable and diverse manner. Specifically, we provide the model with structured prompts describing the scene images, state-aware scene graph, and task instructions, and instruct it to produce question-answer pairs that probe different reasoning skills, such as spatial relation understanding, affordance inference, precondition reasoning, and goal decomposition. To ensure the reliability and correctness of the benchmark, all generated questions and answers undergo several rounds of manual verification, during which ambiguous or low-quality samples are refined or removed.
为严格评估空间-功能推理与任务规划能力，我们基于数据集中的场景与任务设计了一个全面的多选项VQA基准。我们并不手工逐一编写所有问题，而是利用大型视觉-语言模型（VLM）以可扩展且多样的方式生成问题。具体而言，我们向模型提供结构化提示，描述场景图像、具备状态感知的场景图，以及任务指令，并要求其生成问题-答案对，用于检验不同的推理能力，例如空间关系理解、可操作性推断、前置条件推理以及目标分解。为确保基准的可靠性与正确性，所有生成的问题与答案都要经过多轮人工核验；对存在歧义或质量较低的样本，会进行修订或剔除。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_b2fc9a.jpg"/>



Figure 12: Statistics of object occurrences, highlighting the most frequent objects in tasks.
图12：物体出现次数统计，突出任务中最常见的物体。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_3e024d.jpg"/>



Figure 13: Training reward curves during MomaGraph-R1 training.
图13：在 MomaGraph-R1 训练过程中的训练奖励曲线。


Moreover, since the benchmark is formulated as a multi-choice VQA task with clearly defined correct answers, it does not require complex evaluation metrics. Model performance can be directly measured by simple accuracy - i.e., the proportion of correctly answered questions - which provides an intuitive and reliable indicator of spatial-functional reasoning and planning capabilities. This simplicity enables straightforward comparison across models while ensuring that the evaluation remains rigorous and meaningful.
此外，由于该基准被表述为一个具有明确正确答案的多选项VQA任务，因此不需要复杂的评估指标。模型性能可以直接通过简单准确率来衡量——也就是正确回答问题的比例——这为空间-功能推理与规划能力提供了直观且可靠的指标。由于这种简洁性，可以在保证评估严谨且有意义的同时，便于不同模型之间进行直接对比。


Data Source and Task Scope. We leverage long video sequences from SceneFun3D (Delitzas et al., 2024) that capture human-recorded layouts of entire indoor environments, from which key frames are extracted and manually annotated with task-specific graphs. To enhance diversity and coverage, we additionally collect data from real indoor scenes. Our benchmark spans four representative indoor room categories: bathroom, kitchen, living room, and bedroom. The task scope is organized into four levels of difficulty:
数据来源与任务范围。我们利用来自 SceneFun3D（Delitzas et al., 2024） 的长视频序列，它们记录了完整室内环境的人为布局；从中提取关键帧，并用任务相关的图进行人工标注。为提升多样性与覆盖面，我们还额外收集了真实室内场景的数据。我们的基准覆盖四类具有代表性的室内房间：浴室、厨房、客厅和卧室。任务范围划分为四个难度等级：


T1 Single-step actions: e.g., turning on a light, pulling a drawer, opening a door.
T1 单步动作：例如打开照明、拉动抽屉、打开门。


T2 Two complementary steps: e.g., filling a bathtub by first pressing the drain button and then turning on the faucet.
T2 两步互补操作：例如先按下放水按钮，再打开水龙头，以填满浴缸。


T3 Multi-step or preconditioned tasks: e.g., making coffee (pick up a cup $\rightarrow$ add water $\rightarrow$ start the coffee machine).
T3 多步或带前置条件的任务：例如做咖啡（拿起杯子 $\rightarrow$ 加水 $\rightarrow$ 启动咖啡机）。


T4 Dynamic verification tasks: e.g., when the target object is missing, the system must perform graph-based replanning and identify alternative interactive objects.
T4 动态验证任务：例如当目标物体缺失时，系统必须进行基于图的重新规划，并识别可替代的交互物体。


Table 4: DAPO Training Configuration
表4：DAPO 训练配置


<table><tr><td>Parameter</td><td>Value</td></tr><tr><td colspan="2">Model Configuration</td></tr><tr><td>Base Model</td><td>Qwen2.5-VL-7B-Instruct</td></tr><tr><td>Mixed Precision</td><td>bfloat16</td></tr><tr><td colspan="2">Training Setup</td></tr><tr><td>Total Epochs</td><td>25</td></tr><tr><td>Training Steps</td><td>175</td></tr><tr><td>Actor Global Batch Size</td><td>128</td></tr><tr><td>Critic Global Batch Size</td><td>256</td></tr><tr><td>Micro Batch Size (Actor)</td><td>1</td></tr><tr><td>Micro Batch Size (Critic)</td><td>4</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Learning Rate</td><td>1e-6</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Weight Decay</td><td>0.01</td></tr><tr><td>Beta1, Beta2</td><td>0.9, 0.999</td></tr><tr><td>Gradient Clipping</td><td>1.0</td></tr><tr><td colspan="2">DAPO Algorithm</td></tr><tr><td>KL Coefficient</td><td>0.01</td></tr><tr><td>KL Penalty</td><td>low_var_kl</td></tr><tr><td>Disable KL</td><td>True</td></tr><tr><td>Clip Ratio Low</td><td>0.2</td></tr><tr><td>Clip Ratio High</td><td>0.28</td></tr><tr><td>Clip Ratio Dual</td><td>3.0</td></tr><tr><td colspan="2">Reward Function</td></tr><tr><td>Format Weight</td><td>0.2</td></tr><tr><td>Max Response Length</td><td>2048</td></tr><tr><td>Overlong Penalty Factor</td><td>0.5</td></tr><tr><td colspan="2">Generation Config</td></tr><tr><td>Temperature</td><td>1.0</td></tr><tr><td>Top-p</td><td>1.0</td></tr><tr><td>Rollout Samples</td><td>5</td></tr></table>
<table><tbody><tr><td>参数</td><td>数值</td></tr><tr><td colspan="2">模型配置</td></tr><tr><td>基础模型</td><td>Qwen2.5-VL-7B-Instruct</td></tr><tr><td>混合精度</td><td>bfloat16</td></tr><tr><td colspan="2">训练设置</td></tr><tr><td>总轮数</td><td>25</td></tr><tr><td>训练步数</td><td>175</td></tr><tr><td>Actor 全局批大小</td><td>128</td></tr><tr><td>Critic 全局批大小</td><td>256</td></tr><tr><td>微批大小（Actor）</td><td>1</td></tr><tr><td>微批大小（Critic）</td><td>4</td></tr><tr><td colspan="2">优化</td></tr><tr><td>学习率</td><td>1e-6</td></tr><tr><td>优化器</td><td>AdamW</td></tr><tr><td>权重衰减</td><td>0.01</td></tr><tr><td>Beta1，Beta2</td><td>0.9, 0.999</td></tr><tr><td>梯度裁剪</td><td>1.0</td></tr><tr><td colspan="2">DAPO 算法</td></tr><tr><td>KL 系数</td><td>0.01</td></tr><tr><td>KL 惩罚</td><td>low_var_kl</td></tr><tr><td>禁用 KL</td><td>是</td></tr><tr><td>裁剪比下限</td><td>0.2</td></tr><tr><td>裁剪比上限</td><td>0.28</td></tr><tr><td>裁剪比双侧</td><td>3.0</td></tr><tr><td colspan="2">奖励函数</td></tr><tr><td>格式权重</td><td>0.2</td></tr><tr><td>最大响应长度</td><td>2048</td></tr><tr><td>过长惩罚系数</td><td>0.5</td></tr><tr><td colspan="2">生成配置</td></tr><tr><td>温度</td><td>1.0</td></tr><tr><td>Top-p</td><td>1.0</td></tr><tr><td>采样次数</td><td>5</td></tr></tbody></table>


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_5cd8fb.jpg"/>



Figure 14: Validation reward curves during MomaGraph-R1 training.
图14：MomaGraph-R1训练期间的验证奖励曲线。


## B ADDITIONAL ABLATION STUDIES
## B 额外消融研究


### B.1 COMPARISON WITH SFT AND ICL BASELINES
### B.1 与 SFT 和 ICL 基线的对比


To validate our choice of RL-based training over alternative learning paradigms, we compare our method against two additional baselines:
为验证我们选择基于强化学习（RL）的训练而非其他学习范式的合理性，我们将方法与另外两个基线进行比较：


- SFT baseline: We fine-tune Qwen2.5-VL-7B on MomaGraph-Scenes using supervised learning only (without RL), with the same graph-alignment objectives as our full method.
- SFT 基线：我们仅使用监督学习（不使用 RL）在 MomaGraph-Scenes 上对 Qwen2.5-VL-7B 进行微调，并采用与完整方法相同的图对齐目标。


- ICL baseline: We evaluate the base model with 3-5 in-context graph examples provided in the prompt (same setting as Qwen2.5-VL-7B-Instruct (w/ Graph) in Table 2 and 3 of the main paper).
- ICL 基线：我们在提示中提供 3-5 个上下文图示例来评估基础模型（设置与主论文表 2 和表 3 中的 Qwen2.5-VL-7B-Instruct（带图）相同）。


<table><tr><td>Method</td><td>BLINK</td><td>MomaGraph-Bench (Overall)</td></tr><tr><td>SFT baseline</td><td>60.4</td><td>63.9</td></tr><tr><td>ICL baseline</td><td>58.7</td><td>60.2</td></tr><tr><td>RL w/ Graph (Ours)</td><td>63.5</td><td>71.6</td></tr></table>
<table><tbody><tr><td>方法</td><td>眨眼</td><td>MomaGraph-Bench（整体）</td></tr><tr><td>SFT 基线</td><td>60.4</td><td>63.9</td></tr><tr><td>ICL 基线</td><td>58.7</td><td>60.2</td></tr><tr><td>带图的 RL（我们）</td><td>63.5</td><td>71.6</td></tr></tbody></table>


Table 5: Comparison of our RL-based training with SFT and ICL baselines. Our method achieves substantially better performance on both benchmarks.
表5：我们基于RL的训练与SFT和ICL基线的比较。我们的方法在两个基准上都取得了明显更好的性能。


As shown in Table 5, our RL training method achieves clearly superior performance compared to both the SFT baseline (+3.1 on BLINK, +7.7 on MomaGraph-Bench) and the ICL baseline (+4.8 on BLINK, +11.4 on MomaGraph-Bench). This demonstrates that the RL formulation is crucial for learning high-quality scene graph generation that effectively improves downstream planning performance.
如表5所示，我们的RL训练方法相较于SFT基线（在BLINK上+3.1，在MomaGraph-Bench上+7.7）和ICL基线（在BLINK上+4.8，在MomaGraph-Bench上+11.4）都表现出明显更优的性能。这表明RL形式化对于学习高质量场景图生成至关重要，并能有效提升下游规划性能。


### B.2 REWARD WEIGHT SENSITIVITY STUDY
### B.2 奖励权重敏感性研究


We follow the original DAPO implementation in the EasyR1 framework for default settings of ${w}_{f}$ and ${w}_{l}$ in Eq. 2 of the main paper. We conduct a sensitivity study by varying $\left( {{w}_{a},{w}_{f},{w}_{l}}\right)$ around the default configuration:
我们在 EasyR1 框架中遵循原始 DAPO 实现，对主论文 Eq. 2 中的 ${w}_{f}$ 和 ${w}_{l}$ 使用默认设置。我们通过在默认配置的基础上调整 $\left( {{w}_{a},{w}_{f},{w}_{l}}\right)$ 来进行敏感性研究：


<table><tr><td>Setting ID</td><td>Wa</td><td>Wf</td><td>${\mathrm{w}}_{1}$</td><td>BLINK</td><td>MomaGraph-Bench (Overall)</td></tr><tr><td>A</td><td>0.5</td><td>0.5</td><td>0.5</td><td>61.3</td><td>68.2</td></tr><tr><td>B</td><td>0.7</td><td>0.3</td><td>0.5</td><td>63.1</td><td>70.9</td></tr><tr><td>C</td><td>0.8</td><td>0.2</td><td>0.7</td><td>63.7</td><td>71.2</td></tr><tr><td>Default</td><td>0.8</td><td>0.2</td><td>0.5</td><td>63.5</td><td>71.6</td></tr></table>
<table><tbody><tr><td>设置 ID</td><td>Wa</td><td>Wf</td><td>${\mathrm{w}}_{1}$</td><td>BLINK</td><td>MomaGraph-Bench（总体）</td></tr><tr><td>A</td><td>0.5</td><td>0.5</td><td>0.5</td><td>61.3</td><td>68.2</td></tr><tr><td>B</td><td>0.7</td><td>0.3</td><td>0.5</td><td>63.1</td><td>70.9</td></tr><tr><td>C</td><td>0.8</td><td>0.2</td><td>0.7</td><td>63.7</td><td>71.2</td></tr><tr><td>默认</td><td>0.8</td><td>0.2</td><td>0.5</td><td>63.5</td><td>71.6</td></tr></tbody></table>


Table 6: Sensitivity analysis of reward weights $\left( {{w}_{a},{w}_{f},{w}_{l}}\right)$ in our DAPO training. The model’s performance remains stable across different weight configurations.
表6：我们在DAPO训练中对奖励权重$\left( {{w}_{a},{w}_{f},{w}_{l}}\right)$的敏感性分析。模型在不同权重配置下的性能保持稳定。


As shown in Table 6, the model's performance remains stable across these weight configurations, with variations of less than ${2.4}\%$ on BLINK and 3.4% on MomaGraph-Bench. This indicates low sensitivity to reward-weight choices and demonstrates the robustness of our training approach.
如表6所示，模型在这些权重配置下的性能保持稳定，在BLINK上的变化小于${2.4}\%$，在MomaGraph-Bench上变化为3.4%。这表明模型对奖励权重选择不敏感，也证明了我们训练方法的鲁棒性。


### B.3 DETAILED REAL-WORLD DEMONSTRATIONS.
### B.3 详细的真实世界演示。


To provide a closer look into the behavior of our system, this section presents fine-grained real-world examples. We illustrate how the model processes raw images captured in realistic household environments, transforms them into task-oriented scene graphs, and generates corresponding planner outputs. These case studies highlight the system's ability to capture subtle details, encode them into structured graphs, and reason over them to produce actionable plans.
为更深入了解我们系统的行为，本节展示了细粒度的真实世界示例。我们将说明模型如何处理在真实家庭环境中采集的原始图像，将其转换为面向任务的场景图，并生成相应的规划器输出。这些案例研究凸显了系统捕捉细微细节、将其编码进结构化图中的能力，并能在此基础上进行推理，从而生成可执行的计划。


To validate the effectiveness of our approach in real-world settings, we deploy the system on a mobile manipulator to perform a variety of everyday tasks, as shown in Fig. 15. These tasks span multiple functional categories, such as turning off a light, opening a microwave, turning on a TV, and opening a cabinet. In each case, the robot leverages the predicted spatial-functional scene graph to plan and execute a sequence of actions without task-specific fine-tuning. The successful completion of these tasks demonstrates the system's ability to generalize from structured graph representations to real-world interaction scenarios, highlighting its potential for practical household assistance.
为在真实场景中验证我们方法的有效性，我们在移动机械臂上部署该系统，执行多种日常任务，如图15所示。这些任务覆盖多个功能类别，例如关灯、打开微波炉、打开电视以及打开橱柜。在每种情况下，机器人都会利用预测得到的空间-功能场景图来规划并执行一系列动作，而无需进行特定任务的细调。成功完成这些任务表明：该系统能够从结构化图表示中泛化到真实的交互场景，凸显其在实际家庭助理方面的潜力。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_2a2754.jpg"/>



Figure 15: Real-world robot execution of household tasks.
图15：真实世界中机器人执行家庭任务。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_84f218.jpg"/>



## MomaGraph-R1 Answer:
## MomaGraph-R1 答案：


Scene Graph Analysis:
场景图分析：


First, I understand the task: "Open the cabinet." Then, I identify the key objects needed: two handles and a cabinet. Next, I analyze their spatial relationships: knob1 is in front of and close to the cabinet, while knob2 is in front of and close to the cabinet, requiring a "pull" action for opening/closing operations.
首先，我理解任务：“打开橱柜”。然后，我识别所需的关键物体：两个把手和一个橱柜。接着，我分析它们的空间关系：knob1 位于橱柜前方且距离很近；同时，knob2 也位于橱柜前方且距离很近，因此在执行开/关操作时需要“拉”的动作。


Answer: \{"task_instruction": "Open the cabinet.", "nodes": ["handle1", "handle2", "cabinet"], "edges": [\{"functional_relationship": "openorclose", "object1": "handle1", "handle2": " cabinet", "spatial_relations": ["in_front_of", "close"], "is_touching": true\}, \{"functional_relationship": "openorclose", "object1": "handle1", "handle2": " cabinet", "spatial_relations": ["in_front_of", "close"], "is_touching": true\}], "action_type": "pull", "function_type": "open_close_control"\}
答案：\{"task_instruction": "打开橱柜。", "nodes": ["handle1", "handle2", "cabinet"], "edges": [\{"functional_relationship": "openorclose", "object1": "handle1", "handle2": " cabinet", "spatial_relations": ["in_front_of", "close"], "is_touching": true\}, \{"functional_relationship": "openorclose", "object1": "handle1", "handle2": " cabinet", "spatial_relations": ["in_front_of", "close"], "is_touching": true\}], "action_type": "pull", "function_type": "open_close_control"\}


Action Sequence:
动作序列：


1. Approach the target area and identify the cabinet.
1. 靠近目标区域并识别橱柜。


2. Locate the handle1, which is in front of and close to the cabinet.
2. 找到把手1：它位于橱柜前方且距离很近。


3. Grasp the handle1 and pull it to the right.
3. 握住把手1，并向右拉。


4. If the cabinet does not open, locate the handle2, which is in front of and close to the kitchen cabinet.
4. 如果橱柜没有打开，找到把手2：它位于厨房橱柜前方且距离很近。


5. Grasp the handle2 and pull it to the left.
5. 握住把手2，并向左拉。


6. Verify that the kitchen cabinet is now open.
6. 确认厨房橱柜现在已打开。


Figure 16: Real-world example of MomaGraph-R1 performing the task "Open the Cabinet." From multiview images, the system generates a scene graph capturing spatial-functional relations and outputs the corresponding action plan.
图16：MomaGraph-R1 在执行“打开橱柜”任务时的真实场景示例。通过多视角图像，系统生成场景图以捕捉空间-功能关系，并输出对应的动作计划。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_260937.jpg"/>



## MomaGraph-R1 Answer:
## MomaGraph-R1 答案：


Scene Graph Analysis:
场景图分析：


First, I understand the task: "Turn off the light." Then, I identify the key objects needed: a light switch and a light. Next, I analyze their spatial relationships: the light switch is to the right of and far from the light, requiring a "press" action to turn off the light.
首先，我理解任务：“关掉灯”。然后，我识别所需的关键物体：一个电灯开关和一盏灯。接着，我分析它们的空间关系：电灯开关在灯的右侧且距离较远，需要通过“按下”操作来关掉灯。


Answer: \{"task_instruction": "Turn off the light.", "nodes": ["light switch", "light"], "edges": [\{"functional_relationship": "control", "object1": "light switch", "object2": "light", "spatial_relations": ["right_of", "far"], "is_touching": false\}], "action_type": "press", "function_type": "device_control"\} Action Sequence:
答案： \{"task_instruction": "关掉灯.", "nodes": ["电灯开关", "灯"], "edges": [\{"functional_relationship": "控制", "object1": "电灯开关", "object2": "灯", "spatial_relations": ["在右侧", "远离"], "is_touching": false\}], "action_type": "按下", "function_type": "设备控制"\} 动作序列：


1. Approach the target area: Move towards the light switch, which is located to the right of and far from the light.
1. 靠近目标区域：向电灯开关移动，它位于灯的右侧且距离较远。


2. Identify the key objects mentioned in the scene graph: Locate the light switch and the light.
2. 在场景图中识别提到的关键物体：找到电灯开关和灯。


3. Press the light switch: Use your hand to press the light switch.
3. 按下电灯开关：用手按下电灯开关。


4. Verify the task is completed: Check if the light is off by looking at the light.
4. 验证任务是否完成：通过观察灯来确认它是否已关闭。


5. Confirm the task completion: Ensure that the light is off, indicating that the task "Turn off the light" has been successfully completed.
5. 确认任务完成：确保灯已关闭，表明任务“关掉灯”已成功完成。


Figure 17: Real-world example of MomaGraph-R1 performing the task "Turn off the light." From multiview images, the system generates a scene graph capturing spatial-functional relations and outputs the corresponding action plan.
图 17：MomaGraph-R1 执行“关掉灯”任务的真实场景示例。系统从多视角图像中生成场景图，捕捉空间—功能关系，并输出对应的动作计划。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_cf7fc2.jpg"/>



## MomaGraph-R1 Answer:
## MomaGraph-R1 答案：


Scene Graph Analysis:
场景图分析：


First, I understand the task: "Open the microwave." Then, I identify the key objects needed: a microwave and a microwave button. Next, I analyze their spatial relationships: the microwave button is in front of and close to the microwave, requiring a "press" action for opening/closing operations.
首先，我理解任务：“打开微波炉”。然后，我确定所需的关键物体：微波炉和微波炉按钮。接着，我分析它们的空间关系：微波炉按钮在微波炉前方且距离很近，因此需要执行“按压”动作来进行打开/关闭操作。


Answer: \{"task_instruction": "Open the microwave.", "nodes": ["microwave button", "microwave"], "edges": [\{"functional_relationship": "openorclose", "object1": "microwave button", "object2": "microwave", "spatial_relations": ["in_front_of", "close"], "is_touching": true\}], "action_type": "press", "function_type": "open_close_control"\}
答案：\{"task_instruction": "打开微波炉。", "nodes": ["微波炉按钮", "微波炉"], "edges": [\{"functional_relationship": "openorclose", "object1": "微波炉按钮", "object2": "微波炉", "spatial_relations": ["in_front_of", "close"], "is_touching": true\}], "action_type": "press", "function_type": "open_close_control"\}


Action Sequence:
动作序列：


1. Approach the microwave door, which is in front of the microwave and currently closed.
1. 走向微波炉门，它位于微波炉前方，且目前是关闭的。


2. Identify the microwave button and the microwave as the key objects mentioned in the scene graph.
2. 识别场景图中提到的关键物体：微波炉按钮和微波炉。


3. Press the button of the microwave with your hand.
3. 用手按下微波炉的按钮。


4. Verify that the microwave door is fully open.
4. 确认微波炉门已完全打开。


Figure 18: Real-world example of MomaGraph-R1 performing the task "Open the microwave." From multiview images, the system generates a scene graph capturing spatial-functional relations and outputs the corresponding action plan.
图18：MomaGraph-R1 在执行“打开微波炉”任务时的真实世界示例。基于多视角图像，系统生成包含空间-功能关系的场景图，并输出对应的动作计划。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/06/2026_06_11__23_42_49_12e78f.jpg"/>



## MomaGraph-R1 Answer:
## MomaGraph-R1 答案：


Scene Graph Analysis:
场景图分析：


First, I understand the task: "Turn on the TV." Then, I identify the key objects needed: a remote and a TV. Next, I analyze their spatial relationships: The remote is to the right of and in front of the TV, requiring a "press" action for activation.
首先，我理解任务是：“打开电视。”然后，我确定所需的关键对象：遥控器和电视。接着，我分析它们的空间关系：遥控器在电视的右侧并且前方，需要通过“按下”动作来激活。


Answer: \{"task_instruction": "Turn on the TV.", "nodes": ["remote", "TV"], "edges": [\{"functional_relationship": "control", "object1": "remote", "object2": "TV", "spatial_relations": ["in_front_of", "right_of"], "is_touching": false\}], "action_type": "press", "function_type": "device_control"\}
答案：\{"task_instruction": "打开电视。", "nodes": ["遥控器", "电视"], "edges": [\{"functional_relationship": "控制", "object1": "遥控器", "object2": "电视", "spatial_relations": ["前方", "右侧"], "is_touching": false\}], "action_type": "按下", "function_type": "设备控制"\}


Action Sequence:
动作序列：


1. Approach the target area: Move towards the TV and the remote, ensuring they are within reach.
1. 接近目标区域：朝电视和遥控器移动，确保它们在可达范围内。


2. Identify the key objects mentioned in the scene graph: Locate the remote and the TV.
2. 识别场景图中提到的关键对象：找到遥控器和电视。


3. Hold the remote: Use your hand to grasp the remote, ensuring it is in your control.
3. 拿起遥控器：用手握住遥控器，确保其在你的控制之中。


4. Aim the remote at the TV: Position the remote in front of the TV, making sure it is close enough to the TV's surface for effective control.
4. 将遥控器对准电视：把遥控器放在电视前方，确保它足够接近电视表面，以便有效控制。


5. Press the power button: Locate the power button on the remote and press it firmly. This action should turn on the TV.
5. 按下电源按钮：找到遥控器上的电源按钮并用力按下。这个动作应该会打开电视。


6. Verify the task is completed: Check if the TV is turned on by looking for the power indicator light or by attempting to change channels or adjust the volume.
6. 验证任务已完成：通过查看电源指示灯，或尝试切换频道、调节音量，检查电视是否已打开。


Figure 19: Real-world example of MomaGraph-R1 performing the task "Turn on the TV." From multiview images, the system generates a scene graph capturing spatial-functional relations and outputs the corresponding action plan.
图19：MomaGraph-R1 执行任务“打开电视。”的真实世界示例。系统从多视角图像中生成一个捕捉空间-功能关系的场景图，并输出相应的动作计划。