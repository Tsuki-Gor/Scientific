# LLM-Empowered Embodied Agent for Memory-Augmented Task Planning in Household Robotics
# 面向家用机器人中基于记忆增强任务规划的LLM赋能具身智能体


Marc Glocker ${}^{1,2}$ ,Peter Hönig ${}^{1}$ ,Matthias Hirschmanner ${}^{1}$ ,and Markus Vincze ${}^{1}$
Marc Glocker ${}^{1,2}$ ,Peter Hönig ${}^{1}$ ,Matthias Hirschmanner ${}^{1}$ ,and Markus Vincze ${}^{1}$


Abstract-We present an embodied robotic system with an LLM-driven agent-orchestration architecture for autonomous household object management. The system integrates memory-augmented task planning, enabling robots to execute high-level user commands while tracking past actions. It employs three specialized agents: a routing agent, a task planning agent, and a knowledge base agent, each powered by task-specific LLMs. By leveraging in-context learning, our system avoids the need for explicit model training. RAG enables the system to retrieve context from past interactions, enhancing long-term object tracking. A combination of Grounded SAM and LLaMa3.2- Vision provides robust object detection, facilitating semantic scene understanding for task planning. Evaluation across three household scenarios demonstrates high task planning accuracy and an improvement in memory recall due to RAG. Specifically, Qwen2.5 yields best performance for specialized agents, while LLaMA3.1 excels in routing tasks. The source code is available at: https://github.com/marc1198/chat-hsr
摘要——我们提出了一种具身机器人系统，其采用由LLM驱动的智能体编排架构，实现自主家用物体管理。该系统集成了记忆增强的任务规划，使机器人能够在跟踪过往动作的同时执行高层用户指令。系统使用三种专用智能体：路由智能体、任务规划智能体和知识库智能体，分别由面向任务的LLM提供支持。借助上下文学习，我们的系统无需显式模型训练。RAG使系统能够从过往交互中检索上下文，从而增强长期物体跟踪。Grounded SAM与LLaMa3.2-Vision的组合提供了稳健的物体检测，支持用于任务规划的语义场景理解。在三个家居场景上的评估表明，任务规划准确率较高，且由于RAG，记忆召回能力有所提升。具体而言，Qwen2.5在专用智能体中表现最佳，而LLaMA3.1在路由任务中表现优异。源代码可在以下地址获取：https://github.com/marc1198/chat-hsr


Index Terms-Embodied AI, Task Planning, Memory Retrieval
索引词——具身AI，任务规划，记忆检索


## I. INTRODUCTION
## I. 引言


Despite recent progress in robotics and artificial intelligence, robots still struggle to adapt flexibly to the diverse, dynamic situations of real-world environments, particularly in household settings [24]. While symbolic task planning with languages like the Planning Domain Definition Language (PDDL) [11] is effective in domains with fixed rules and predictable object categories, it lacks the adaptability required for open-ended household environments. In such settings, robots must deal with ambiguous user commands, detect novel or unstructured objects, and respond to constantly changing spatial configurations [24]. These limitations motivate our hypothesis that a modular LLM-driven system can enhance flexibility by leveraging natural language understanding, contextual reasoning, and memory-based adaptation. We provide a proof-of-concept implementation and assess its performance in real-world household tasks.
尽管机器人和人工智能领域近年来取得了进展，机器人仍难以灵活适应真实环境中多样且动态的情况，尤其是在家庭场景中[24]。虽然使用规划域定义语言（Planning Domain Definition Language, PDDL）[11]等符号任务规划在规则固定、物体类别可预测的领域中很有效，但它缺乏开放式家庭环境所需的适应能力。在这种场景下，机器人必须处理含糊的用户指令、识别新颖或非结构化的物体，并应对空间构型不断变化[24]。这些局限促使我们提出假设：一个模块化的基于大语言模型（LLM）的系统可以通过利用自然语言理解、基于上下文的推理以及基于记忆的适应来增强灵活性。我们提供了概念验证实现，并在真实家庭任务中评估其性能。


In this work, we present an embodied robotic system with an LLM-driven agent-orchestration architecture, where specialized software agents collaborate to address long-horizon household tasks. Recent advances in Large Language Models (LLMs) [13], [4], [15], [23], [5] have improved systems real-world understanding, enabling common-sense reasoning in human language and making them accessible to researchers. These advances combined with in-context learning [26] enable flexible embodied task planning by decomposing high-level commands, such as "clear the dining table", into actionable steps based on detected objects [2], [7], [25], [9], [21]. By integrating Grounded Segment Anything Model (Grounded SAM) [17] and LLaMa3.2-Vision [4], our system creates grounded task plans. Unlike most other works, we address long-term operations by maintaining action and environment records, utilizing Retrieval-Augmented Generation (RAG) for efficient memory retrieval. Our approach enables the robot to autonomously organize and retrieve objects, interpret complex tasks, and provide updates on object locations, all while ensuring privacy through the use of offline LLMs and avoiding explicit model training. To illustrate the systems interaction, Fig. 1 shows an example of our system in action.
在这项工作中，我们提出一种具身的机器人系统，其采用基于LLM的智能体编排架构，其中专门的软件智能体协同完成长时域的家庭任务。大型语言模型（LLMs）[13]、[4]、[15]、[23]、[5]的最新进展提升了系统对真实世界的理解能力，使其能够在人的语言中进行常识推理，并让研究者能够使用。这些进展结合上下文学习[26]，通过将诸如“清理餐桌”这样的高层指令分解为基于已检测到物体的可执行步骤[2]、[7]、[25]、[9]、[21]，从而实现灵活的具身任务规划。通过集成Grounded Segment Anything Model（Grounded SAM）[17]和LLaMa3.2-Vision[4]，我们的系统生成具备落点的任务计划。不同于大多数相关工作，我们通过维护动作与环境记录来应对长期运行，并利用检索增强生成（Retrieval-Augmented Generation, RAG）实现高效的记忆检索。我们的方法使机器人能够自主组织并检索物体、理解复杂任务，并更新物体位置，同时通过使用离线LLM来保证隐私，并避免显式的模型训练。为说明系统交互，图1展示了我们系统运行时的一个示例。


<img src="https://cdn.noedgeai.com/bo_d8j3pns91nqc738ucjr0_0.jpg?x=864&y=579&w=690&h=402&r=0"/>



Fig. 1: Our LLM-driven robotic system autonomously plans tasks and retrieves past interactions to improve object handling, illustrated by LLM-enforced task planning and memory-retrieved reasoning in a household setting.
图1：我们的基于LLM的机器人系统能够自主规划任务，并检索过去的交互以改进物体处理能力；该过程通过在家庭场景中由LLM约束的任务规划以及由记忆检索驱动的推理来说明。


In summary, we present the following key contributions:
总结而言，我们提出以下关键贡献：


- A long-horizon task planner for household tasks leveraging in-context learning and offline LLMs.
- 面向家庭任务的长时域任务规划器，利用上下文学习与离线LLM。


- Use of RAG for efficient memory retrieval and object tracking.
- 使用RAG实现高效的记忆检索与物体跟踪。


- A modular agent-orchestration system that improves robustness and modularity.
- 一个模块化的智能体编排系统，用于提升鲁棒性与模块性。


- Evaluation of the system's performance in three real-world household scenarios.
- 在三个真实家庭场景中评估系统性能。


This paper is structured as follows: Section III reviews related work in the areas of task planning and memory mechanisms. Section III details the proposed system architecture. Section IV describes the experimental setup and household scenarios. Section V presents the results. Finally, Section VI concludes the paper and outlines directions for future work.
本文结构如下：第III节回顾任务规划与记忆机制相关工作。第III节详细介绍所提出的系统架构。第IV节描述实验设置与家庭场景。第V节给出结果。最后，第VI节对本文进行总结，并概述未来工作的方向。


---



1 Automation and Control Institute, Faculty of Electrical Engineering, TU Wien, 1040 Vienna, Austria \{hoenig, hirschmanner, vincze $\}$ @acin.ac.tuwien.at
1 自动化与控制研究所，电气工程学院，维也纳工业大学（TU Wien），1040维也纳，奥地利 \{hoenig, hirschmanner, vincze $\}$ @acin.ac.tuwien.at


2 AIT Austrian Institute of Technology GmbH, Center for Vision, Automation and Control, 1210 Vienna, Austria marc.glocker@ait.ac.at
2 AIT奥地利科技有限公司，视觉、自动化与控制中心，1210维也纳，奥地利 marc.glocker@ait.ac.at


---



## II. RELATED WORK
## II. 相关工作


In this section we discuss related work for action and task planning, as well as memory and knowledge base.
本节讨论动作与任务规划，以及记忆和知识库方面的相关工作。


## A. Action and Task Planning
## A. 动作与任务规划


Recent advancements in prompt engineering have improved the problem-solving capabilities of LLMs [26], [28], enabling the generation of structured plans without fine-tuning. Consequently, modern agent architectures leverage LLMs to dynamically react to execution failures [27], [7] and expand their context by retrieval [8] or external tools [19], [18]. However, LLMs lack an inherent understanding of a robot's physical abilities and real-world constraints. SayCan [2] addresses this by integrating value functions of pre-trained robotic skills to ensure feasibility, whereas Huang et al. [6] leverage LLMs to match high-level plans with low-level actions through semantic mapping. Some works treat LLMs as programmers rather than direct decision-makers: Code-as-Policies [9] and ProgPrompt [21] allow LLMs to generate structured code for robotic executions, enhancing flexibility but adding an execution layer.
近年来，提示工程的进展提升了 LLM 的问题解决能力 [26]、[28]，使其无需微调即可生成结构化计划。因此，现代智能体架构利用 LLM 动态应对执行失败 [27]、[7]，并通过检索 [8] 或外部工具 [19]、[18] 扩展上下文。然而，LLM 缺乏对机器人物理能力和现实世界约束的内在理解。SayCan [2] 通过整合预训练机器人技能的价值函数来保证可行性；而 Huang 等 [6] 则利用 LLM 通过语义映射，将高层计划与低层动作进行匹配。部分工作将 LLM 视为程序员而非直接决策者：Code-as-Policies [9] 和 ProgPrompt [21] 使 LLM 能为机器人执行生成结构化代码，增强灵活性，但同时增加了一个执行层。


Pallagani et al. [14] found that LLMs perform better as translators of natural language into structured plans rather than generating plans from scratch. This ensures feasible actions based on predefined world models [20], [10]. These approaches are particularly effective in highly controlled environments, but present challenges when applied to open-ended, dynamic household settings. Our work, instead, embraces flexible, dynamic task planning with in-context learning like shown in [25]. The approaches named, while effective for short-horizon tasks, do not track object positions over time. For long-horizon tasks that involve real-world dynamic conditions, a combination of task planning and a memory mechanism is required.
Pallagani 等人 [14] 发现，LLM 更擅长将自然语言翻译为结构化计划，而不是从零开始生成计划。这样可以基于预先定义的世界模型 [20]、[10] 确保可行动作。上述方法在高度受控的环境中尤其有效，但在开放式、动态的家庭场景中应用时会面临挑战。相较之下，我们的工作采用如 [25] 所示的情境学习来进行灵活、动态的任务规划。尽管这些方法对短视距任务有效，但无法随时间跟踪物体位置。对于涉及现实世界动态条件的长视距任务，则需要任务规划与记忆机制的结合。


## B. Memory and Knowledge Base
## B. 记忆与知识库


Long-horizon tasks require robust memory mechanisms. While LLM context windows keep expanding [23], using excessively large contexts in robotics is computationally inefficient. Instead, long-term memory retrieval, accessed only when needed, is a more viable solution. RAG [8] provides an efficient mechanism for narrowing context by querying a vast dataset and retrieving only relevant information. Additionally, scene graphs, used in approaches like SayPlan [16] and DELTA [10], offer structured memory that improves action verification and contextual reasoning. However, in unstructured and constantly changing environments, maintaining these graphs becomes challenging due to the need for complex automatic mechanisms or manual curation.
长时程任务需要可靠的记忆机制。尽管 LLM 的上下文窗口不断扩展 [23]，但在机器人中使用过大的上下文在计算上效率低下。相较之下，仅在需要时才进行的长期记忆检索是一种更可行的方案。RAG [8] 提供了一种高效机制，通过查询海量数据集并仅检索相关信息来缩小上下文范围。此外，场景图在如 SayPlan [16] 与 DELTA [10] 之类的方法中被使用，它们提供结构化记忆，从而提升动作验证与情境推理。然而，在非结构化且持续变化的环境中，由于需要复杂的自动机制或人工策展，维护这些图会变得困难。


Our work explores the feasibility of a lightweight, fully natural language-driven approach using RAG as a memory mechanism. Inspired by ReMEmbR [1], our system incorporates temporal elements into the retrieval process, ensuring the robot tracks long-term changes in its environment. While using language-based memory retrieval introduces potential for increased errors compared to structured models like scene graphs, we aim to evaluate how well purely language-based memory retrieval performs in practical, dynamic household scenarios. This approach offers flexibility, adaptability, and reduces the need for explicit world modelling, making it more suitable for real-world applications.
我们的工作探讨了采用 RAG 作为记忆机制、实现轻量且完全由自然语言驱动的方法的可行性。受 ReMEmbR [1] 启发，我们的系统将时间要素融入检索过程，确保机器人能够跟踪其环境中的长期变化。虽然基于语言的记忆检索相较于场景图这类结构化模型可能引入更高的错误率，但我们旨在评估在实际、动态的家庭场景中，仅依靠语言的记忆检索表现如何。该方法具有灵活性与适应性，并减少对显式世界建模的需求，使其更适用于真实应用。


## III. METHODOLOGY
## III. 方法论


<img src="https://cdn.noedgeai.com/bo_d8j3pns91nqc738ucjr0_1.jpg?x=853&y=662&w=712&h=413&r=0"/>



Fig. 2: The full pipeline, integrating long-horizon task planning. Newly introduced components are highlighted in blue.
图 2：完整流程，集成了长时程任务规划。新引入的组件以蓝色高亮显示。


Our system, coordinated by an agent-orchestration framework, combines task planning with RAG [8]. This chapter explains the individual components and their interaction.
我们的系统由一个智能体编排框架协调，将任务规划与 RAG [8] 相结合。本章将解释各个组件及其交互方式。


Fig. 2 illustrates the overall pipeline. The focus of this work is the agent-orchestration system, which processes object detection and user requests to create a robot task plan. In the system, each agent uses an LLM with a specialized role. The task planning agent additionally is prompted with a chain-of-thought technique [26].
图 2 展示了整体流程。本工作的重点是智能体编排系统，它处理物体检测和用户请求，以生成机器人任务计划。在该系统中，每个智能体都使用具有特定角色的 LLM。任务规划智能体还额外采用了思维链技术 [26] 提示。


<img src="https://cdn.noedgeai.com/bo_d8j3pns91nqc738ucjr0_1.jpg?x=867&y=1532&w=687&h=433&r=0"/>



Fig. 3: The agent-orchestration architecture
图 3：智能体编排架构


The system architecture of the agent orchestrator, illustrated in Fig. 3 consists of:
图 3 所示的智能体编排器系统架构由以下部分组成：


1) A routing agent, responsible for analyzing incoming user requests.
1) 路由智能体，负责分析传入的用户请求。


2) A task planning agent, handling commands that require the robot to perform actions.
2) 任务规划智能体，处理需要机器人执行动作的命令。


3) A knowledge base agent, processing follow-up questions about previously handled objects.
3) 知识库智能体，处理关于先前处理过的物体的后续问题。


When a user request arrives, the routing agent first analyzes it to determine its nature. The request is then categorized into one of three types:
当用户请求到达时，路由智能体首先对其进行分析以确定其性质。随后，该请求被归类为以下三种类型之一：


1) Action command: If the robot is asked to perform an action, it is forwarded to the task planning agent.
1) 动作命令：如果要求机器人执行某项动作，则将其转发给任务规划智能体。


2) Query about history: If it concerns previously handled objects, it is directed to the knowledge base agent.
2) 历史查询：如果涉及先前处理过的物体，则将其发送给知识库智能体。


3) Unclear request: If the request doesn't fit either category, clarification is requested before proceeding.
3) 请求不明确：如果请求不属于上述任一类别，则在继续之前会要求澄清。


## A. Task Planning Agent
## A. 任务规划代理


The task planning agent receives frequent environmental updates via camera perception, encoded as a list of single objects. Grounded SAM [17] enables text-driven object detection and segmentation for the pipeline, while Vision Language Models (VLMs) generate natural language descriptions of the environment. Although VLMs alone can extract the object list for the LLM, Grounded SAM is essential for precise segmentation, which is critical for grasping tasks. Using the object list, the LLM processes the user request - which can be both expressed in high-level or low-level terms - and formulates tasks that best fulfill the command. The generated answer has to include a JSON string for an action following this structure:
任务规划代理通过相机感知持续接收环境更新，将其编码为单个物体的列表。Grounded SAM [17] 使得该流程能够基于文本进行物体检测与分割，而视觉语言模型（VLMs）则生成对环境的自然语言描述。尽管仅靠 VLMs 也能为 LLM 提取物体列表，但 Grounded SAM 对于精确分割至关重要，而这对于抓取任务的理解是关键。借助物体列表，LLM 会处理用户请求——该请求既可以用高层次也可以用低层次的方式表达——并制定最能满足指令的任务。生成的回答必须包含一个遵循以下结构的动作 JSON 字符串：


1) Objects involved in the task.
1）任务中涉及的物体。


2) The destination for placement tasks.
2）用于放置任务的目标位置。


After the action is determined, the grasping process is initiated. We use the segmentation from Grounded SAM and the camera intrinsics to crop the depth image and project the depth crop to a 3D pointcloud of the respective object. To estimate a grasp approach vector, we feed the cropped object point cloud to Control-GraspNet [22], a pre-trained grasp estimator.
在确定动作后，随即启动抓取过程。我们使用来自 Grounded SAM 的分割结果以及相机内参来裁剪深度图，并将该深度裁剪投影为相应物体的三维点云。为估计抓取接近向量，我们将裁剪后的物体点云输入 Control-GraspNet [22]，它是一个预训练的抓取估计器。


## B. Knowledge Base Agent
## B. 知识库代理


<img src="https://cdn.noedgeai.com/bo_d8j3pns91nqc738ucjr0_2.jpg?x=156&y=1682&w=688&h=292&r=0"/>



Fig. 4: RAG workflow for long-term question answering: Relevant past actions are retrieved from dialogue history, and the LLM generates responses based on the retrieved context.
图 4：用于长期问答的 RAG 工作流：从对话历史中检索相关的过去操作，并基于检索到的上下文由大语言模型生成回答。


The knowledge base agent is used for user inquiries regarding past robot actions, such as object locations or task completion status. These queries require access to long-term memory, for which RAG has proven most effective, as discussed in Section 1 Fig 4 illustrates the RAG workflow, comprising two key steps:
知识库代理用于处理用户关于过去机器人操作的询问，例如物体位置或任务完成状态。此类查询需要访问长期记忆，而 RAG 已被证明最有效。第 1 节讨论了这一点。图 4 说明了 RAG 工作流，包括两个关键步骤：


1) Document Ingestion: Input data, such as conversation history, is preprocessed, split into smaller chunks (each representing a question-answer pair), and converted into high-dimensional vectors using an embedding model. These embeddings are then stored in a vector database for efficient retrieval.
1）文档摄取：对输入数据（如对话历史）进行预处理，将其拆分为更小的片段（每个片段对应一个问答对），并使用嵌入模型将其转换为高维向量。这些嵌入随后存储到向量数据库中，以便高效检索。


2) User Query, Retrieval, and Response Generation: User queries are embedded using the same model and are matched against the stored vectors to retrieve the most relevant context. This context is then provided to the LLM, which generates a response tailored to the user's query.
2）用户查询、检索与响应生成：使用同一模型对用户查询进行嵌入，并与存储的向量进行匹配，以检索最相关的上下文。随后将该上下文提供给大语言模型，使其生成与用户查询相匹配的回答。


To enable chronological reasoning, essential for tracking object movements over time, we augment RAG with a time stamp for each question-answer pair.
为实现按时间顺序的推理（这对于随时间跟踪物体移动至关重要），我们为每个问答对为 RAG 增加了时间戳。


## IV. EXPERIMENTS
## IV. 实验


To evaluate our system, we conduct experiments addressing the three key challenges from Chapter 1 (1) flexible task planning in dynamic household environments, (2) long-term memory usage, and (3) modular agent coordination. Specifically, we assess the system's ability to create grounded task plans, answer questions based on prior interactions, and route tasks to the appropriate agent.
为评估我们的系统，我们开展实验，针对第一章提出的三项关键挑战：（1）在动态家庭环境中进行灵活任务规划，（2）长期记忆的使用，以及（3）模块化智能体协作。具体而言，我们评估系统生成可落地的任务计划、基于既有交互回答问题，以及将任务分派给合适智能体的能力。


## A. Experimental Setup
## A. 实验设置


This study evaluates an agent-orchestration system for symbolic task planning and follow-up questions via a knowledge base. To ensure a thorough evaluation, we consider three distinct phases:
本研究通过知识库评估一种面向符号任务规划及后续问题的智能体编排系统。为确保评估充分，我们考虑三个不同阶段：


1) Task Planning Performance - The symbolic task planning output is assessed independently, measuring accuracy of object assignment to their destinations.
1) 任务规划性能——独立评估符号任务规划输出，衡量对象被分配到其目标位置的准确性。


2) Knowledge Base Reliability - The system's ability to reason about past actions (with and without RAG) is tested by asking about the system's current status, such as locations of previously moved items.
2) 知识库可靠性——通过询问系统的当前状态（例如先前移动物品的位置），测试系统对既往行动的推理能力（有无 RAG 均测试）。


3) Routing Reliability - Measures the accuracy of the routing agent in directing queries to the appropriate agent (Task Planning, History, or itself).
3) 路由可靠性——衡量路由智能体在将查询导向相应智能体（任务规划、历史记录或自身）方面的准确性。


To isolate the performance of the specialized agents, agent handoff is not considered in the evaluation of 1) and 2).
为隔离专用智能体的性能，评估 1) 和 2) 时不考虑智能体交接。


## B. Algorithmic Framework
## B. 算法框架


The frameworks and models used are shown in gray in Fig. 4 To enable efficient collaboration among agents, we use OpenAI Swarm [12], a lightweight framework for agent orchestration and task delegation. We evaluate the performance of Qwen2.5-32b [15], Gemma2-27b [23], and LLaMa3.1-8b [4], selected for their open-source availability and ability to run locally on 16GB GPU RAM. For RAG, we employ ChromaDB [3], a vector database optimized for fast lookups, combined with the embedding model BGE-M3.
所用框架和模型以灰色显示于图4中。为实现代理之间的高效协作，我们采用OpenAI Swarm [12]，这是一个用于代理编排与任务分配的轻量级框架。我们评估了Qwen2.5-32b [15]、Gemma2-27b [23]和LLaMa3.1-8b [4]的性能，这些模型因其开源可用性以及可在16GB GPU RAM上本地运行而被选中。对于RAG，我们采用ChromaDB [3]，一种针对快速检索优化的向量数据库，并结合BGE-M3嵌入模型。


<img src="https://cdn.noedgeai.com/bo_d8j3pns91nqc738ucjr0_3.jpg?x=141&y=350&w=688&h=309&r=0"/>



Fig. 5: The artificial household environment used in the experiment.
图5：实验中使用的人工家庭环境。


The experiment is conducted in an artificial household environment, where objects must be assigned to correct destinations based on high-level commands. To evaluate task planning, we define three scenarios (see Fig. 6) that share five predefined placement locations, while each uses a different cleanup zone. Fig. 5 shows a visual representation of the environment. These locations reflect common-sense knowledge typically understood by LLMs. To ensure clarity, the agent receives explicit definitions for each destination:
实验在一个人工家庭环境中进行，物体必须根据高层指令被分配到正确的目标位置。为评估任务规划，我们定义了三种场景（见图6），它们共享五个预定义放置位置，但各自使用不同的清理区域。图5展示了该环境的可视化表示。这些位置体现了LLM通常能理解的常识知识。为确保清晰，代理会收到每个目标位置的明确定义：


- Sink - For items that need washing.
- 水槽 - 用于需要清洗的物品。


- Trash Can - For disposable or inedible items.
- 垃圾桶 - 用于一次性或不可食用物品。


- Fridge - For perishable food.
- 冰箱 - 用于易腐食物。


- Food Shelf - For non-perishable food items.
- 食品架 - 用于不易腐坏的食物。


- Storage Box - For general storage.
- 储物箱 - 用于一般存放。


Fig. 6 shows the object list extracted from a captured image of each task scenario using LLaMa3.2-Vision along with the user queries and the segmentation results from Grounded SAM.
图6展示了使用LLaMa3.2-Vision从每个任务场景的捕获图像中提取的物体列表，以及用户查询和Grounded SAM的分割结果。


After execution of all scenarios, the knowledge base agent is asked four distinct folow-up questions targeting different aspects of retrieval and reasoning:
在所有场景执行完毕后，系统会向知识库代理提出四个不同的后续问题，针对检索与推理的不同方面：


- Error Detection: "Where is the jacket that was in the living room? I thought you put it in the storage box, but I can't find it there."
- 错误检测："客厅里的那件夹克在哪里？我以为你把它放进了储物箱，但我在那里找不到。"


- Hallucination: "Where did you put the laptop? It's not on the desk anymore."
- 幻觉："你把笔记本电脑放哪了？它不在桌子上了。"


- Food Availability: "I am hungry. Is there any food left from earlier?"
- 食物可用性："我饿了。之前还有食物剩下吗？"


- Trash Status: "How many objects are in the trash can?"
- 垃圾状态："垃圾桶里有多少物体？"


To better reflect real-world applications, we extend the conversation dialogue with additional question-answer pairs containing actions. Furthermore, deliberate errors are introduced into the task plans, where the agent provides the user a different location than the one forwarded to the state machine. This allows us to evaluate how well the knowledge base handles inaccuracies. Beyond evaluating the specialized agents in isolated setups, we assess how effectively the routing agent delegates tasks to the appropriate specialized agent. Specifically, we test:
为更好地贴近实际应用，我们在对话中扩展了包含动作的附加问答对。此外，我们在任务计划中引入了故意错误，即代理向用户提供的地点与其传递给状态机的地点不同。这使我们能够评估知识库对不准确性的处理效果。除了在独立设置下评估专门代理外，我们还考察路由代理将任务分配给合适专门代理的效率。具体而言，我们测试：


- Task Planning Queries: The three high-level commands from the task planning scenarios (see Fig. 6) and an additional low-level request ("Can I have a banana?")
- 任务规划查询：来自任务规划场景的三条高层指令（见图6）以及一条额外的低层请求（“我能要一根香蕉吗？”）


- Knowledge Base Queries: The four follow-up questions from the knowledge base scenario.
- 知识库查询：来自知识库场景的四个后续问题。


## D. Evaluation Methodology
## D. 评估方法


The evaluation of the agent-orchestration system's components is based on the task scenarios and follow-up questions defined in Section IV-C Task planning performance is evaluated by testing each model on the three task scenarios, with each scenario executed five times per model. Accuracy is measured at the object level as the percentage of correctly assigned tasks. A task is deemed correct if it satisfies the following criteria:
代理编排系统各组件的评估基于第 IV-C 节中定义的任务场景和后续问题。任务规划性能通过在三个任务场景上测试每个模型来评估，每个场景对每个模型执行五次。准确率以对象级别衡量，即正确分配任务的百分比。若任务满足以下标准，则视为正确：


- Valid JSON format
- 有效的 JSON 格式


- Correct destination assignment
- 正确的目标分配


- Stationary Object Exclusion (ensuring no task is assigned to items that should remain in place)
- 静止对象排除（确保不会将任务分配给应保持原位的物品）


The final accuracy score represents the percentage of objects for which tasks were correctly assigned, including the implicit "no task" assignment for stationary objects (e.g., table).
最终准确率表示任务被正确分配的对象所占百分比，其中也包括对静止对象（如桌子）隐含的“无任务”分配。


The knowledge base is evaluated using four follow-up questions, each tested five times per model. Unlike the task planning agent, the knowledge base agent does not require a strict output format. It is assessed based on factual correctness, measured as the percentage of correct answers. For queries expecting multiple objects as an answer (e.g., "Which objects are in the trash?"), accuracy is based on the percentage of correctly identified objects.
知识库通过四个后续问题进行评估，每个问题对每个模型测试五次。与任务规划代理不同，知识库代理不要求严格的输出格式。它依据事实正确性进行评估，以正确答案的百分比衡量。对于期望以多个对象作为答案的查询（如“哪些物体在垃圾桶里？”），准确率基于被正确识别对象的百分比。


The routing agent's ability to correctly assign tasks is evaluated by processing queries from the task planning scenarios and history-based questions, along with one additional query, five times per model. The final metric is quantified as the percentage of correctly assigned tasks. Gemma2, which does not support tool calling, is excluded from this test.
路由代理正确分配任务的能力通过处理任务规划场景中的查询、基于历史的问题，以及一个额外查询来评估，每个模型测试五次。最终指标量化为正确分配任务的百分比。Gemma2 不支持工具调用，因此被排除在此测试之外。


## V. RESULTS AND DISCUSSION
## V. 结果与讨论


This section presents the experimental results for task planning, knowledge base and agent routing.
本节介绍任务规划、知识库和智能体路由的实验结果。


## A. Task Planning
## A. 任务规划


We introduce a lenient evaluation metric (cf. Table 1), where reasonable alternative placements based on user preferences are counted as correct. The strictly correct placements, following the intended plan as prompted to the LLM, are presented under the strict metric in Table 1
我们提出一种宽松的评估指标（参见表1）：基于用户偏好所作的合理替代安置被计为正确。严格正确的安置——即按提示给LLM的预期计划执行的安置——在表1的严格指标下给出


Table 1 shows that Qwen consistently outperforms the other models in nearly all scenarios. LLaMA performs notably worse in the living room scenario, with the lowest strict accuracy (40.0%). Gemma2 falls between the two, showing higher accuracy than LLaMA but lower than Qwen.
表1显示，Qwen几乎在所有场景中都始终优于其他模型。LLaMA在客厅场景表现尤为较差，严格准确率最低（40.0%）。Gemma2介于两者之间：其准确率高于LLaMA，但低于Qwen。


## B. Knowledge Base
## B. 知识库


The integration of RAG notably enhances the accuracy of the knowledge base's responses, even in medium-term interactions consisting of 21 question-answer pairs with approximately 4000 tokens. Qwen achieves the highest validity (91.3%) with RAG (cf. Table II), highlighting the potential of retrieval-augmented approaches for maintaining consistency over longer interactions.
RAG 的集成显著提升了知识库回复的准确性。即使在包含 21 组问答、约 4000 tokens 的中期交互中也同样如此。Qwen 在使用 RAG 时达到最高有效性（91.3%）（见表 II），这凸显了基于检索增强的方法在更长交互中维持一致性的潜力。


<img src="https://cdn.noedgeai.com/bo_d8j3pns91nqc738ucjr0_4.jpg?x=144&y=319&w=481&h=364&r=0"/>



(a) Scenario 1: Dining Table Cleanup Object list from VLM: Plate, Fork, Spoon, Salt shaker, Glass, Frying pan, Spatula, Chair, Table top, Pepper grinder. Command: I just finished dinner, please clear the dining table.
（a）场景 1：来自 VLM 的餐桌清理物体列表：盘子、叉子、勺子、盐罐、玻璃杯、煎锅、刮刀、椅子、桌面、胡椒研磨器。指令：我刚吃完饭，请把餐桌收拾一下。


<img src="https://cdn.noedgeai.com/bo_d8j3pns91nqc738ucjr0_4.jpg?x=694&y=279&w=309&h=406&r=0"/>



(b) Scenario 2: Living Room Cleanup Object list from VLM: A table, A couch, A brush, Scissors, Pen, Book, Salt packet, Jacket, Markers. Command: Please hand me the brush and tidy up the rest of the living room.
（b）场景 2：来自 VLM 的客厅清理物体列表：一张桌子、一张沙发、一把刷子、剪刀、笔、书、一包盐、夹克、马克笔。指令：把刷子递给我，并把客厅剩下的地方收拾干净。


<img src="https://cdn.noedgeai.com/bo_d8j3pns91nqc738ucjr0_4.jpg?x=1073&y=319&w=480&h=366&r=0"/>



(c) Scenario 3: Desk Organization
（c）场景 3：桌面整理


Object list from VLM: Desk, Computer Monitor, Laptop, Mouse, Plate, Crumbs, Lemon, Cup, Glass of water, Bag of chips, Piece of paper, Potted plant, Cord, Wooden desk, White wall. Command: Please clear my desk, leaving only the essentials for work.
来自 VLM 的物体列表：桌子、电脑显示器、笔记本、鼠标、盘子、面包屑、柠檬、杯子、一杯水、薯片袋、一张纸、盆栽植物、电线、木质桌子、白色墙面。指令：请把我的桌子清理一下，只留下工作所需的基本物品。


Fig. 6: The three scenarios used for task planning. For each scenario we have extracted an object list using the Vision-Language Model LLaMa3.2-Vision. This list is used as input for Grounded SAM [17] to perform segmentation.
图 6：用于任务规划的三个场景。我们使用视觉-语言模型 LLaMa3.2-Vision 从每个场景中提取物体列表。该列表作为输入交给 Grounded SAM [17] 以执行分割。


<table><tr><td rowspan="2">Model</td><td colspan="2">Dining Table</td><td colspan="2">Living Room</td><td colspan="2">Desk Organization</td><td colspan="2">Total Accuracy (%)</td></tr><tr><td>Strict (%)</td><td>Lenient (%)</td><td>Strict (%)</td><td>Lenient (%)</td><td>Strict (%)</td><td>Lenient (%)</td><td>Strict (%)</td><td>Lenient (%)</td></tr><tr><td>LLaMa3.1-8B</td><td>68.0</td><td>78.0</td><td>40.0</td><td>40.0</td><td>61.3</td><td>65.3</td><td>56.4</td><td>61.1</td></tr><tr><td>Gemma2-27B</td><td>58.0</td><td>68.0</td><td>68.9</td><td>68.9</td><td>68.0</td><td>69.3</td><td>65.0</td><td>68.7</td></tr><tr><td>Qwen2.5-32B</td><td>64.0</td><td>80.0</td><td>88.9</td><td>88.9</td><td>78.7</td><td>84.0</td><td>77.2</td><td>84.3</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">餐桌</td><td colspan="2">客厅</td><td colspan="2">桌面整理</td><td colspan="2">总准确率 (%)</td></tr><tr><td>严格 (%)</td><td>宽松 (%)</td><td>严格 (%)</td><td>宽松 (%)</td><td>严格 (%)</td><td>宽松 (%)</td><td>严格 (%)</td><td>宽松 (%)</td></tr><tr><td>LLaMa3.1-8B</td><td>68.0</td><td>78.0</td><td>40.0</td><td>40.0</td><td>61.3</td><td>65.3</td><td>56.4</td><td>61.1</td></tr><tr><td>Gemma2-27B</td><td>58.0</td><td>68.0</td><td>68.9</td><td>68.9</td><td>68.0</td><td>69.3</td><td>65.0</td><td>68.7</td></tr><tr><td>Qwen2.5-32B</td><td>64.0</td><td>80.0</td><td>88.9</td><td>88.9</td><td>78.7</td><td>84.0</td><td>77.2</td><td>84.3</td></tr></tbody></table>


TABLE I: Task Planning Accuracy Across Different LLMs. Strict (%): Percentage of objects correctly placed according to the intended plan. Lenient (%): Percentage of objects placed differently than expected, but with reasonable alternative placements based on user preferences.
表 I：不同大语言模型的任务规划准确率。严格（%）：根据预期计划正确放置对象的比例。宽松（%）：对象放置方式与预期不同，但基于用户偏好给出合理替代放置的比例。


<table><tr><td rowspan="2">Method</td><td rowspan="2">Model</td><td colspan="4">Response Validity (%)</td><td rowspan="2">Total Validity (%)</td></tr><tr><td>Err. Detection</td><td>Hallucination</td><td>Food Avail.</td><td>Trash Status</td></tr><tr><td rowspan="3">Without RAG (Ablation Study)</td><td>LLaMa3.1-8B</td><td>20.0</td><td>80.0</td><td>70.0</td><td>65.0</td><td>58.8</td></tr><tr><td>Gemma2-27B</td><td>0.0</td><td>80.0</td><td>10.0</td><td>60.0</td><td>37.5</td></tr><tr><td>Qwen2.5-32B</td><td>0.0</td><td>80.0</td><td>60.0</td><td>75.0</td><td>53.75</td></tr><tr><td rowspan="3">With RAG</td><td>LLaMa3.1-8B</td><td>40.0</td><td>100.0</td><td>90.0</td><td>55.0</td><td>71.25</td></tr><tr><td>Gemma2-27B</td><td>80.0</td><td>100.0</td><td>40.0</td><td>60.0</td><td>70.0</td></tr><tr><td>Qwen2.5-32B</td><td>100.0</td><td>100.0</td><td>90.0</td><td>75.0</td><td>91.3</td></tr></table>
<table><tbody><tr><td rowspan="2">方法</td><td rowspan="2">模型</td><td colspan="4">响应有效性（%）</td><td rowspan="2">总有效性（%）</td></tr><tr><td>错误检测</td><td>幻觉</td><td>食物可用性</td><td>垃圾状态</td></tr><tr><td rowspan="3">无 RAG（消融研究）</td><td>LLaMa3.1-8B</td><td>20.0</td><td>80.0</td><td>70.0</td><td>65.0</td><td>58.8</td></tr><tr><td>Gemma2-27B</td><td>0.0</td><td>80.0</td><td>10.0</td><td>60.0</td><td>37.5</td></tr><tr><td>Qwen2.5-32B</td><td>0.0</td><td>80.0</td><td>60.0</td><td>75.0</td><td>53.75</td></tr><tr><td rowspan="3">使用 RAG</td><td>LLaMa3.1-8B</td><td>40.0</td><td>100.0</td><td>90.0</td><td>55.0</td><td>71.25</td></tr><tr><td>Gemma2-27B</td><td>80.0</td><td>100.0</td><td>40.0</td><td>60.0</td><td>70.0</td></tr><tr><td>Qwen2.5-32B</td><td>100.0</td><td>100.0</td><td>90.0</td><td>75.0</td><td>91.3</td></tr></tbody></table>


TABLE II: Knowledge Base Response Accuracy Across Different LLMs. Used Embedding Model for RAG: BGE-M3. No. of question-answer pairs retrieved by RAG: 5
表 II：不同 LLM 的知识库响应准确率。RAG 所用嵌入模型：BGE-M3。RAG 检索的问题-答案对数量：5


## C. Agent Routing
## C. Agent Routing


In task delegation, LLaMA exhibits the highest routing accuracy (92.5%), despite its weaker reasoning abilities (cf. Table III). Its structured approach to tool-calling ensures stable performance. In contrast, Qwen, while superior in contextual understanding, occasionally produces incorrect structured outputs, leading to execution failures.
在任务委派中，LLaMA 尽管推理能力较弱（参见表 III），但表现出最高的路由准确率（92.5%）。其对工具调用的结构化方法确保了稳定的性能。相比之下，Qwen 虽然在上下文理解方面更占优势，但偶尔会生成不正确的结构化输出，从而导致执行失败。


## D. Summary
## D. Summary


Our findings highlight the potential of lightweight, open-source LLMs for memory-augmented long-horizon task planning. A combination of LLaMA (routing) and Qwen (specialized agents) achieves the best balance between structured execution and high-level reasoning.
我们的发现凸显了轻量级、开源 LLM 在记忆增强的长时域任务规划方面的潜力。LLaMA（路由）与 Qwen（专业智能体）的组合，在结构化执行与高层推理之间取得了最佳平衡。


Evaluating task execution remains challenging due to subjective human preferences, emphasizing the need for user studies. Furthermore, integrating Vision-Language Models (VLMs) into the agent orchestrator - rather than only using them for object lists - could enhance robustness. Embedding contextual information into the latent space reduces command dependency and improves autonomy.
由于人类偏好具有主观性，任务执行的评估仍然颇具挑战，这强调了开展用户研究的必要性。此外，将视觉-语言模型（VLM）集成进智能体编排器——而不仅仅用于物体列表——有望提升系统的鲁棒性。将上下文信息嵌入潜在空间可降低对指令的依赖，并提高自主性。


RAG improves factual consistency in knowledge retrieval but struggles with repeated object interactions and long histories, making full-history queries impractical. Scene graphs, as proposed by Liu et al. [10], present a promising alternative for efficient and robust knowledge integration.
RAG 能在知识检索中提升事实一致性，但在重复的物体交互与长序列历史方面表现不佳，使得对全历史的查询并不现实。Liu 等人［10］提出的场景图，为高效且稳健的知识整合提供了一种有前景的替代方案。


<table><tr><td>Model</td><td>Task Planning Queries (%)</td><td>Knowledge Base Queries (%)</td><td>Total Success Rate (%)</td></tr><tr><td>LLaMa3.1-8B</td><td>85.0</td><td>100.0</td><td>92.5</td></tr><tr><td>Qwen2.5-32B</td><td>95.0</td><td>85.0</td><td>90.0</td></tr></table>
<table><tbody><tr><td>模型</td><td>任务规划查询(%)</td><td>知识库查询(%)</td><td>总成功率(%)</td></tr><tr><td>LLaMa3.1-8B</td><td>85.0</td><td>100.0</td><td>92.5</td></tr><tr><td>Qwen2.5-32B</td><td>95.0</td><td>85.0</td><td>90.0</td></tr></tbody></table>


TABLE III: Routing Success Rate Across Different LLMs
表 III：不同 LLM 的路由成功率


While task delegation via the routing agent was mostly successful, certain models occasionally produced invalid structured outputs, leading to execution failures. To increase robustness, future work should explore schema validation and adaptive retry mechanisms that can automatically mitigate such issues.
尽管通过路由代理进行任务分配大多成功，但某些模型偶尔会生成无效的结构化输出，导致执行失败。为提高鲁棒性，未来工作应探索模式验证和自适应重试机制，以自动缓解此类问题。


In summary, open-source LLMs prove viable for long-horizon task planning. However, addressing key challenges - refining evaluation metrics, improving long-term robustness, and integrating multimodal perception - remains essential for achieving reliable household robotics.
总之，开源 LLM 证明了其在长时程任务规划中的可行性。然而，要实现可靠的家庭机器人，仍需解决关键挑战——改进评估指标、提升长期鲁棒性，以及集成多模态感知。


## VI. CONCLUSION
## VI. 结论


This work presents a prototype of an agent-orchestration system for household robots, utilizing local, lightweight open-source LLMs to translate high-level user commands into structured task plans for tidy-up scenarios. Memory-augmented task planning enables follow-up queries about past actions, improving user interaction and assisting in locating misplaced objects. Our evaluation shows strong task planning, routing, and knowledge retrieval. with Qwen2.5 excelling in reasoning-heavy tasks and LLaMA3.1 providing a more efficient routing solution. However, RAG-based retrieval for general tasks remains a challenge, particularly for implicit queries where relevant information is not always found. Addressing these limitations is key to improving long-term reasoning and knowledge access.
本工作提出了一个面向家用机器人的智能体编排系统原型，利用本地、轻量级的开源 LLM，将高层用户指令转换为整理场景中的结构化任务计划。借助记忆增强的任务规划，系统能够就过往动作进行后续查询，从而改善用户交互并辅助定位放错位置的物品。我们的评估表明，该系统在任务规划、路由和知识检索方面表现出色，其中 Qwen2.5 在推理密集型任务上更为突出，而 LLaMA3.1 提供了更高效的路由方案。然而，基于 RAG 的通用任务检索仍然是一个挑战，尤其是在隐式查询中，相关信息并不总能被找到。解决这些局限是提升长期推理与知识获取能力的关键。


Future work will focus on robust storage solutions, improved knowledge representations, broader user studies with structured datasets for evaluating and benchmarking existing approaches. Enhancing communication and tool usage in agent-orchestration will be crucial for greater adaptability and autonomy in household robotics.
未来工作将聚焦于更稳健的存储方案、更好的知识表示，以及面向现有方法评估与基准测试的更大规模、结构化数据集用户研究。提升智能体编排中的通信与工具使用能力，对于家用机器人实现更强的适应性和自主性至关重要。


## ACKNOWLEDGMENT
## 致谢


This research is supported by the EU program EC Horizon 2020 for Research and Innovation under grant agreement No. 101017089, project TraceBot, and the Austrian Science Fund (FWF), under project No. I 6114, iChores.
本研究得到欧盟研究与创新计划“地平线2020”（EC Horizon 2020）的资助，资助协议编号为 101017089，项目为 TraceBot；并得到奥地利科学基金（FWF）的支持，项目编号为 I 6114，iChores。


## REFERENCES
## 参考文献


[1] A. Anwar, J. Welsh, J. Biswas, S. Pouya, and Y. Chang, "Remembr: Building and reasoning over long-horizon spatio-temporal memory for robot navigation," arXiv preprint arXiv:2409.13682, 2024.
[1] A. Anwar, J. Welsh, J. Biswas, S. Pouya 和 Y. Chang，“Remembr：为机器人导航构建并推理长时域时空记忆”，arXiv 预印本 arXiv:2409.13682，2024。


[2] A. Brohan, et al., "Do as i can, not as i say: Grounding language in robotic affordances," in in Proceedings of the Conference on Robot Learning (CoRL), 2023, pp. 287-318.
[2] A. Brohan 等，“按我能说到的做，而不是按我说的做：在机器人可操作性中对语言进行落地”，发表于机器人学习大会（CoRL）论文集，2023 年，第 287-318 页。


[3] Chroma, "Chromadb," open-source vector database for AI applications. [Online]. Available: https://www.trychroma.com/
[3] Chroma，“Chromadb”，用于 AI 应用的开源向量数据库。[Online] 可从：https://www.trychroma.com/ 获取。


[4] A. Dubey, et al., "The llama 3 herd of models," arXiv preprint arXiv:2407.21783, 2024.
[4] A. Dubey 等，“Llama 3 模型群”，arXiv 预印本 arXiv:2407.21783，2024。


[5] D. Guo, et al., "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning," arXiv preprint arXiv:2501.12948, 2025.
[5] D. Guo 等，“Deepseek-r1：通过强化学习来激励大语言模型的推理能力”，arXiv 预印本 arXiv:2501.12948，2025。


[6] W. Huang, P. Abbeel, D. Pathak, and I. Mordatch, "Language models as zero-shot planners: Extracting actionable knowledge for embodied agents," in in Proceedings of the International Conference on Machine Learning (ICML), 2022, pp. 9118-9147.
[6] W. Huang, P. Abbeel, D. Pathak 和 I. Mordatch，“语言模型作为零样本规划器：为具身代理提取可执行知识”，发表于机器学习国际会议（ICML）论文集，2022 年，第 9118-9147 页。


[7] W. Huang, et al., "Inner monologue: Embodied reasoning through planning with language models," arXiv preprint arXiv:2207.05608, 2022.
[7] W. Huang 等，“内心独白：借助语言模型进行规划的具身推理”，arXiv 预印本 arXiv:2207.05608，2022。


[8] P. Lewis, et al., "Retrieval-augmented generation for knowledge-intensive nlp tasks," Advances in neural information processing systems, vol. 33, pp. 9459-9474, 2020.
[8] P. Lewis 等，“面向知识密集型 nlp 任务的检索增强生成”，神经信息处理系统进展，第 33 卷，第 9459-9474 页，2020。


[9] J. Liang, et al., "Code as policies: Language model programs for embodied control," in in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 9493-9500.
[9] J. Liang 等，“代码即策略：用于具身控制的语言模型程序”，发表于 IEEE 国际机器人与自动化会议（ICRA）论文集。IEEE，2023 年，第 9493-9500 页。


[10] Y. Liu, L. Palmieri, S. Koch, I. Georgievski, and M. Aiello, "Delta: Decomposed efficient long-term robot task planning using large language models," arXiv preprint arXiv:2404.03275, 2024.
[10] Y. Liu, L. Palmieri, S. Koch, I. Georgievski 和 M. Aiello，“Delta：利用大型语言模型实现高效的分解式长期机器人任务规划”，arXiv 预印本 arXiv:2404.03275，2024。


[11] D. M. McDermott, "The 1998 AI planning systems competition," AI Magazine, vol. 21, no. 2, p. 35, 2000.
[11] D. M. McDermott，“1998 年 AI 规划系统竞赛”，AI Magazine，第 21 卷，第 2 期，第 35 页，2000。


[12] OpenAI, "Swarm: Educational framework for multi-agent systems." [Online]. Available: https://github.com/openai/swarm
[12] OpenAI，“Swarm：面向多智能体系统的教学框架”。[Online] 可从：https://github.com/openai/swarm 获取。


[13] OpenAI, et al., "GPT-4 technical report," 2024. [Online]. Available: http://arxiv.org/abs/2303.08774
[13] OpenAI 等，“GPT-4 技术报告”，2024。[Online] 可从：http://arxiv.org/abs/2303.08774 获取。


[14] V. Pallagani, et al., "On the prospects of incorporating large language models (llms) in automated planning and scheduling (aps)," in Proceedings of the International Conference on Automated Planning and Scheduling, vol. 34, 2024, pp. 432-444.
[14] V. Pallagani 等，“在自动规划与调度（aps）中引入大型语言模型（llms）的前景”，发表于自动规划与调度国际会议论文集，第 34 卷，2024 年，第 432-444 页。


[15] Qwen, et al., "Qwen2.5 technical report," 2025. [Online]. Available: http://arxiv.org/abs/2412.15115
[15] Qwen 等，“Qwen2.5 技术报告”，2025。[Online] 可从：http://arxiv.org/abs/2412.15115 获取。


[16] K. Rana, et al., "Sayplan: Grounding large language models using 3d scene graphs for scalable robot task planning," arXiv preprint arXiv:2307.06135, 2023.
[16] K. Rana 等，“Sayplan：利用 3D 场景图为可扩展机器人任务规划提供大语言模型落地支持，”arXiv 预印本 arXiv:2307.06135，2023。


[17] T. Ren, et al., "Grounded sam: Assembling open-world models for diverse visual tasks," arXiv preprint arXiv:2401.14159, 2024.
[17] T. Ren 等，“Grounded SAM：为多样化视觉任务组装开放世界模型，”arXiv 预印本 arXiv:2401.14159，2024。


[18] J. Ruan, et al., "Tptu: Task planning and tool usage of large language model-based ai agents," in NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023.
[18] J. Ruan 等，“Tptu：基于大语言模型的 AI 智能体的任务规划与工具使用，”载于 NeurIPS 2023 决策制定基础模型研讨会，2023。


[19] T. Schick, et al., "Toolformer: Language models can teach themselves to use tools," Advances in Neural Information Processing Systems, vol. 36, pp. 68539-68551, 2023.
[19] T. Schick 等，“Toolformer：语言模型可以自学使用工具，”《神经信息处理系统进展》，第 36 卷，pp. 68539-68551，2023。


[20] T. Silver, et al., "Generalized planning in pddl domains with pretrained large language models," in in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, 2024, pp. 20256-20264.
[20] T. Silver 等，“在 PDDL 域中使用预训练大语言模型进行通用规划，”载于《AAAI 人工智能会议论文集》，第 38 卷，2024，pp. 20256-20264。


[21] I. Singh, et al., "Progprompt: Generating situated robot task plans using large language models," in in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 11523-11530.
[21] I. Singh 等，“Progprompt：使用大语言模型生成具身机器人任务规划，”载于 IEEE 国际机器人与自动化会议（ICRA）论文集。IEEE，2023，pp. 11523-11530。


[22] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox, "Contact-graspnet: Efficient 6-dof grasp generation in cluttered scenes," in in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 13438-13444.
[22] M. Sundermeyer、A. Mousavian、R. Triebel 和 D. Fox，“Contact-graspnet：在杂乱场景中高效生成 6 自由度抓取，”载于 IEEE 国际机器人与自动化会议（ICRA）论文集，2021，pp. 13438-13444。


[23] G. Team, et al., "Gemma 2: Improving open language models at a practical size," arXiv preprint arXiv:2408.00118, 2024.
[23] G. Team 等，“Gemma 2：以实用规模改进开放语言模型，”arXiv 预印本 arXiv:2408.00118，2024。


[24] S. Tellex, N. Gopalan, H. Kress-Gazit, and C. Matuszek, "Robots that use language," Annual Review of Control, Robotics, and Autonomous Systems, vol. 3, no. 1, pp. 25-55, 2020.
[24] S. Tellex、N. Gopalan、H. Kress-Gazit 和 C. Matuszek，“使用语言的机器人，”《控制、机器人与自主系统年评》，第 3 卷，第 1 期，pp. 25-55，2020。


[25] S. H. Vemprala, R. Bonatti, A. Bucker, and A. Kapoor, "Chatgpt for robotics: Design principles and model abilities," Ieee Access, 2024.
[25] S. H. Vemprala、R. Bonatti、A. Bucker 和 A. Kapoor，“用于机器人学的 ChatGPT：设计原则与模型能力，”Ieee Access，2024。


[26] J. Wei, et al., "Chain-of-thought prompting elicits reasoning in large language models," Advances in neural information processing systems, vol. 35, pp. 24824-24837, 2022.
[26] J. Wei 等，“思维链提示激发大语言模型的推理能力，”《神经信息处理系统进展》，第 35 卷，pp. 24824-24837，2022。


[27] S. Yao, et al., "React: Synergizing reasoning and acting in language models," in in Proceedings of the International Conference on Learning Representations (ICLR), 2023.
[27] S. Yao 等，“React：在语言模型中协同推理与行动，”载于国际学习表征会议（ICLR）论文集，2023。


[28] D. Zhou, et al., "Least-to-most prompting enables complex reasoning in large language models," arXiv preprint arXiv:2205.10625, 2022.
[28] D. Zhou 等，“由少到多提示使大语言模型能够进行复杂推理，”arXiv 预印本 arXiv:2205.10625，2022。