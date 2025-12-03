# Generalizable Long-Horizon Manipulations with Large Language Models
# 通用化的大跨度操作（Long-Horizon Manipulations）与大型语言模型


Haoyu Zhou ${}^{1}$ , Mingyu Ding ${}^{2 * }$ , Weikun Peng ${}^{1}$ , Masayoshi Tomizuka ${}^{2}$ , Lin Shao ${}^{1 * }$ and Chuang Gan ${}^{3,4}$
周浩宇 ${}^{1}$ , 丁明宇 ${}^{2 * }$ , 彭威坤 ${}^{1}$ , 富塚正義 ${}^{2}$ , 邵琳 ${}^{1 * }$ 和 甘闯 ${}^{3,4}$


Abstract- This work introduces a framework harnessing the capabilities of Large Language Models (LLMs) to generate primitive task conditions for generalizable long-horizon manipulations with novel objects and unseen tasks. These task conditions serve as guides for the generation and adjustment of Dynamic Movement Primitives (DMP) trajectories for long-horizon task execution. We further create a challenging robotic manipulation task suite based on Pybullet for long-horizon task evaluation. Extensive experiments in both simulated and real-world environments demonstrate the effectiveness of our framework on both familiar tasks involving new objects and novel but related tasks, highlighting the potential of LLMs in enhancing robotic system versatility and adaptability. Project website: https://object814.github.io/ Task-Condition-With-LLM/
摘要- 本文提出一个框架，利用大型语言模型（LLMs）生成原始任务条件，以实现对新物体和未见任务的可泛化大跨度操作。这些任务条件作为指导，用于生成和调整用于长序列任务执行的动态运动基元（DMP）轨迹。我们还基于 Pybullet 创建了用于大跨度任务评估的具有挑战性的机器人操控任务集。在仿真和真实环境中的大量实验表明，所提框架在涉及新物体的熟悉任务和新但相关的任务上均有效，突显了 LLM 在增强机器人系统多功能性与适应性方面的潜力。项目网站: https://object814.github.io/ Task-Condition-With-LLM/


## I. INTRODUCTION
## 一、引言


Recent years have witnessed significant achievements in robot manipulations, and the growing demand for household and multifunctional robots capable of handling complex tasks has brought long-horizon manipulations into the spotlight.
近年来，机器人操控取得了显著进展，对能处理复杂任务的家用与多功能机器人的需求增长，使得大跨度操控成为研究热点。


Approaches like Task and Motion Planning [1-3] and hierarchical reinforcement/imitation learning methods [4, 5] propose to decompose long-horizon tasks into hierarchies, comprising high-level primitive tasks or discrete symbolic states alongside low-level manipulation motions. However, the full promise of long-horizon tasks necessitates not only task decomposition with primitive tasks but also a keen focus on environmental conditions. These conditions encompass aspects such as object interactions and spatial relationships, e.g., an object inside another or a gripper grasping an object, playing an essential role in determining the success or failure of primitive tasks. They are crucial for guiding and correcting low-level trajectories and motions during long-horizon execution.
诸如任务与运动规划（Task and Motion Planning）[1-3] 以及分层强化/模仿学习方法[4, 5] 等方法，提出将大跨度任务分解为层级结构，包含高层的原始任务或离散符号状态以及低层的操控动作。然而，要充分实现大跨度任务的潜力，不仅需要将任务分解为原始任务，还必须关注环境条件。这些条件涵盖物体交互与空间关系，例如一个物体在另一个物体内或夹爪握住某物，决定原始任务成败，且对在长跨度执行中指导与修正低层轨迹和动作至关重要。


While such environmental conditions can be obtained through data sampling and point cloud or image processing from human demonstrations, two key limitations arise. Firstly, acquiring such data and demonstrations can be costly; and secondly, it is challenging to generalize to novel scenarios and tasks without additional demonstrations. Recently, Large Language Models (LLMs) have demonstrated strong reasoning abilities with human commonsense and in-context learning capabilities. For example, [6-9] showcase LLMs excel in failure explanation, task decomposition, and plan scoring for robot manipulation tasks.
尽管这类环境条件可通过数据采样以及对人类示范的点云或图像处理获得，但存在两个关键限制。首先，获取此类数据与示范代价高昂；其次，若无额外示范，难以对新场景与新任务泛化。近期，大型语言模型在常识推理与上下文学习方面表现出强大能力。例如，[6-9] 展示了 LLM 在失败解释、任务分解和任务计划评分方面在机器人操控任务中的优越性。


<img src="https://cdn.noedgeai.com/bo_d4nfvdref24c73bbfi60_0.jpg?x=916&y=456&w=729&h=904&r=0"/>



Fig. 1: Overall Framework. We leverage LLMs to generate and generalize primitive task conditions for both familiar tasks with novel objects and novel but related tasks. Subsequently, the high-level task conditions guide the generation and adjustment of low-level trajectories originally learned from demonstrations for long-horizon task execution.
图1：总体框架。我们利用 LLM 生成并泛化原始任务条件，适用于涉及新物体的熟悉任务及新但相关的任务。随后，高层任务条件指导基于示范学得的低层轨迹的生成与调整，以实现长跨度任务执行。


Motivated by the above observations, this work leverages LLMs to create and generalize task conditions for both familiar tasks involving novel objects and new but related tasks. This offers promise in two ways: 1) Task conditions can be derived not only from human demonstrations but also from the inherent commonsense knowledge within LLMs, and 2) The prior knowledge and in-context learning capabilities of LLMs help generalize to novel objects and unseen tasks. Subsequently, the high-level task conditions guide the generation and adjustment of low-level Dynamic Movement Primitives (DMP) trajectories initially learned from demonstrations for long-horizon task execution. These components are seamlessly integrated and rigorously evaluated in our proposed manipulation task suite simulator and real-world environments. We summarize our contributions as follows:
基于以上观察，本工作利用 LLM 为涉及新物体的熟悉任务和新但相关的任务创建并泛化任务条件，这在两方面具有前景：1）任务条件不仅可从人类示范中获得，也可从 LLM 内在常识知识推导；2）LLM 的先验知识与上下文学习能力有助于对新物体与未见任务的泛化。随后，高层任务条件指导最初从示范中学得的低层动态运动基元（DMP）轨迹的生成与调整，以实现长跨度任务执行。这些组件被无缝整合并在我们提出的操控任务套件模拟器与真实环境中严格评估。我们总结的贡献如下：


---



${}^{1}$ National University of Singapore zhouhaoyu01@u.nus.edu, linshao@nus.edu.sg
${}^{1}$ 新加坡国立大学 zhouhaoyu01@u.nus.edu, linshao@nus.edu.sg


${}^{2}$ UC Berkeley,USA \{myding,tomizuka\}@berkeley.edu ${}^{3}$ University of Massachusetts Amherst,USA
${}^{2}$ 加州大学伯克利分校, 美国 \{myding,tomizuka\}@berkeley.edu ${}^{3}$ 马萨诸塞大学阿默斯特分校, 美国


${}^{4}$ MIT-IBM Watson AI Lab,USA ganchuang1990@gmail.com *Corresponding authors
${}^{4}$ MIT-IBM Watson AI Lab, 美国 ganchuang1990@gmail.com *通信作者


---



- An LLM module for generating and generalizing task conditions for both seen and unseen primitive tasks.
- 一个用于为已见与未见原始任务生成与泛化任务条件的 LLM 模块。


- A systematic long-horizon manipulation framework that leverages high-level task conditions to guide the generation of low-level action trajectory based on DMPs.
- 一个系统化的长跨度操控框架，利用高层任务条件指导基于 DMP 的低层动作轨迹生成。


- A challenging manipulation task suite within Pybullet.
- 一个基于 Pybullet 的具有挑战性的操控任务套件。


- Extensive experiments in both simulated and real-world settings to demonstrate the effectiveness of our pipeline.
- 在仿真与真实环境中广泛的实验以验证我们流水线的有效性。


## II. RELATED WORK
## II. 相关工作


## A. Robotic Skill Learning for Long Horizon Tasks
## A. 面向长时序任务的机器人技能学习


A common approach to tackle long-horizon tasks is task-and-motion-planning (TAMP) that decomposes the planning process of a long-horizon task into discrete symbolic states and continuous motion generation [1-3]. However, classical TAMP methods rely on manually specified symbolic rules, thereby requiring known physical states with high dimensional search space in complex tasks. Recent works [10, 11] integrate learning into the TAMP framework to speed up the search of feasible plans [12-14] or directly predict the action sequences from an initial image [15].
一种常见方法是任务与运动规划 (TAMP)，将长时序任务的规划分解为离散的符号状态和连续的运动生成 [1-3]。但传统的 TAMP 方法依赖人工指定的符号规则，因此在复杂任务中需要已知的物理状态并面临高维搜索空间。近来工作 [10, 11] 将学习引入 TAMP 框架以加速可行计划的搜索 [12-14]，或直接从初始图像预测动作序列 [15]。


Another potential solution for solving long-horizon tasks is hierarchical reinforcement/imitation learning [4, 5]. The options framework [16] and FUN (FeUdal Networks) [17], adopt modular architectures where agents learn both high-and low-level policies, with the high-level policy generating abstract sub-tasks and the low-level policy executing primitive actions. However, designing an efficient hierarchical structure and representing reusable domain knowledge across tasks remains a challenging problem. Relay Policy Learning [18] involves an imitation learning stage that produces goal-conditioned hierarchical policies, followed by a reinforcement learning phase that fine-tunes these policies to improve task performance. In this work, we explore using large language models as task condition generators for long-horizon robot manipulation, which are generalizable to both novel objects and novel tasks.
另一种潜在的解决方案是层次化强化/模仿学习 [4, 5]。options 框架 [16] 和 FeUdal Networks (FUN) [17] 采用模块化结构，代理同时学习高层与低层策略，高层策略生成抽象子任务，低层策略执行原子动作。然而，设计高效的层次结构并跨任务表示可复用的领域知识仍然具有挑战性。Relay Policy Learning [18] 包含一个生成目标条件层次策略的模仿学习阶段，随后通过强化学习阶段微调这些策略以提升任务性能。在本工作中，我们探索使用大型语言模型作为长时序机器人操作的任务条件生成器，能够推广到新物体和新任务。


## B. Large Language Models for Robotic Learning
## B. 用于机器人学习的大型语言模型


LLMs, such as GPT-3 [19], PaLM [20, 21], Galactica [22], and LLaMA [23], exhibit strong capacities to understand natural language and solve complex tasks. For robotics, many research endeavors have focused on enabling robots and other agents to comprehend and follow natural language instructions [24, 25], often through the acquisition of language-conditioned policies [26-33]. Additionally, researchers also explores connecting LLMs to robot commands [7, 8, 33- 38], leveraging pre-trained language embeddings [26, 28, 31, 39-45] and pre-trained vision-language models [46-48] in robotic imitation learning. The uniqueness of our work is to leverage GPT-3.5 to generate generalizable task conditions for long-horizon manipulation tasks.
LLM（如 GPT-3 [19]、PaLM [20, 21]、Galactica [22] 和 LLaMA [23]）展示了强大的自然语言理解与复杂任务解决能力。对于机器人领域，许多研究致力于使机器人及其它代理理解并执行自然语言指令 [24, 25]，通常通过学习语言条件策略 [26-33]。此外，研究者还探索将 LLM 与机器人命令相连接 [7, 8, 33-38]，在机器人模仿学习中利用预训练语言嵌入 [26, 28, 31, 39-45] 和预训练视觉-语言模型 [46-48]。我们工作的独特之处在于利用 GPT-3.5 为长时序操作任务生成可泛化的任务条件。


## III. TECHNICAL APPROACH
## III. 技术方法


Our work presents a framework that leverages pretrained large language models (LLMs) to generate and generalize task conditions of primitive tasks, which are subsequently used to guide the generation and refinement of the Dynamic Movement Primitives (DMP) trajectories for long-horizon task execution. The framework is structured into four parts: 1) task condition generation (Sec. III-B), 2) generalization (Sec. III-C), which use LLMs to acquire task conditions for both seen and unseen tasks; 3) trajectory learning (Sec. III-E) for generating DMP trajectories; and 4) trajectory generation & adjustment (Sec. III-E), which incorporates information from task conditions and the environment.
我们提出一个框架，利用预训练大型语言模型 (LLMs) 生成并泛化原子任务的任务条件，随后用于引导动态运动基元 (DMP) 轨迹的生成与细化，从而执行长时序任务。该框架由四部分构成：1) 任务条件生成 (Sec. III-B)，2) 泛化 (Sec. III-C)，使用 LLM 获取已见与未见任务的任务条件；3) 轨迹学习 (Sec. III-E)，用于生成 DMP 轨迹；4) 轨迹生成与调整 (Sec. III-E)，结合任务条件与环境信息。


---



Task name: move bottle into box
任务名: move bottle into box


Relevant objects: bottle, box, gripper
相关物体: bottle, box, gripper


	Pre-conditions:
	前置条件:


												gripper grasping bottle
												gripper grasping bottle


												box open
												box open


	Post-conditions:
	后置条件:


													bottle inside box
													bottle inside box


													gripper inside box
													gripper inside box


													gripper grasping bottle
													夹爪抓取瓶子


													box open
													盒子打开


	Collisions: gripper and bottle
	碰撞：夹爪与瓶子


---



Listing 1: Task condition structure.
清单 1：任务条件结构。


## A. Definitions
## A. 定义


1) Task Condition Definition: In this work, the task condition for primitive tasks is focusing on the following aspects:
1）任务条件定义：在本工作中，原语任务的任务条件侧重于以下方面：


- Task name: a brief description of the primitive task.
- 任务名称：对原语任务的简短描述。


- Relevant objects: potential objects relevant to the primitive task.
- 相关对象：与原语任务可能相关的对象。


- Pre-conditions: environmental conditions that must be satisfied for the initiation of a primitive task, otherwise lead to task failure.
- 前置条件：启动原语任务必须满足的环境条件，否则导致任务失败。


- Post-conditions: environmental conditions that mark the success of a primitive task.
- 后置条件：标志原语任务成功的环境条件。


- Collisions: permissible collisions during the task execution.
- 碰撞：任务执行期间允许的碰撞。


A typical example of task condition is shown in Lst. [1] the pre/post-conditions are divided into two categories:
任务条件的典型示例如清单[1]所示，前/后置条件分为两类：


- Spatial relations: shown in orange, representing spatial relations between relevant objects. The spatial relations we will focus on include: inside, above, below, in front of
- 空间关系：用橙色表示，代表相关对象之间的空间关系。我们关注的空间关系包括：inside、above、below、in front of


- Object states: shown in magenta, representing the state of the relevant objects. The object states we will focus on include: gripper: grasping/not grasping, and container: open/close
- 对象状态：用品红色表示，代表相关对象的状态。我们关注的对象状态包括：夹爪：grasping/not grasping，容器：open/close


2) Environment Information Acquiring: We enable the framework with a certain perception ability. The raw data we are getting from the environment is semantic point clouds generated from RGBD cameras. The following methods are then used to get different environmental information:
2）环境信息获取：我们为框架赋予一定的感知能力。来自环境的原始数据是由 RGBD 相机生成的语义点云。然后使用以下方法获取不同的环境信息：


- Object center position/bounding box: we get the ground truth object center position and bounding box directly from Pybullet if it is visible in the semantic point cloud. A Gaussian noise is then added to each coordinate of the center position and bounding box, in order to simulate error in the real scenario.
- 对象中心位置/边界框：如果在语义点云中可见，我们直接从 Pybullet 获取真实的对象中心位置和边界框。然后在中心位置和边界框的每个坐标上加入高斯噪声，以模拟真实场景中的误差。


- Spatial relations: the four spatial relations between objects mentioned in Sec. III-A.1 is generated based on center positions and bounding boxes of the objects, as in [6].
- 空间关系：文献第 III-A.1 节提到的四种物体间空间关系基于物体的中心位置和边界框生成，方法同 [6]。


- Object states: the gripper state is judged through its spatial relations with other objects. If another is inside the gripper, we consider it to be grasped; the open or closed state of containers is judged by getting their moving joint position.
- 物体状态：通过夹爪与其他物体的空间关系判断夹爪状态。如果有物体在夹爪内，则视为已抓取；容器的开合状态通过读取其活动关节位置判断。


- Collisions: Collision between environment objects is detected using an implemented function in Pybullet. The position of the collision is generated at the same time.
- 碰撞：环境物体之间的碰撞使用 Pybullet 中实现的函数检测。碰撞位置同时生成。


<img src="https://cdn.noedgeai.com/bo_d4nfvdref24c73bbfi60_2.jpg?x=159&y=136&w=1480&h=592&r=0"/>



Fig. 2: Task Condition Generation and DMP Learning During Demonstration. This figure shows the outline of our framework during demonstrations. Task condition is generated by LLM with prompts and examples (green part), or from environment information (blue part) as comparison. The trajectory is encoded by DMP (orange part). Task condition and trajectory make up the demo data.
图 2：演示过程中任务条件生成与 DMP 学习。该图展示了我们在演示阶段框架的概要。任务条件由带有提示和示例的 LLM 生成（绿色部分），或从环境信息获取（蓝色部分）以作对比。轨迹由 DMP 编码（橙色部分）。任务条件与轨迹构成演示数据。


---



Imagine you are a spatial relation & collision judgment machine
假设你是一个空间关系与碰撞判定机


Here are the spatial relations you can choose from:
以下是你可以选择的空间关系：


	above, below, inside, in front of
	上方、下方、内部、前方


Here are some examples:
以下是一些示例：


[Task condition examples with only spatial relation & collision]
[仅含空间关系与碰撞的任务条件示例]


Remember to strictly follow the examples' format.
请严格遵循示例的格式。


Q: Task name: [Generation Task Name]
问：任务名：[Generation Task Name]


---



Listing 2: Prompt for spatial relation & collision generation.
列表 2：空间关系与碰撞生成的提示。


---



Imagine you are an object state judgment machine
假设你是一个物体状态判定机


Here are the object states you can choose from:
以下是你可以选择的物体状态：


	gripper grasping/not grasping, container open/closed
	夹爪 抓取/未抓取，容器 开/闭


Here are some examples:
以下是一些示例：


[Examples of task condition with only object states]
[仅含对象状态的任务条件示例]


Remember to strictly follow the examples' format.
请严格遵循示例的格式。


Q: Task name: [Generation Task Name]
问：任务名称： [Generation Task Name]


---



Listing 3: Prompt for object state generation.
图 3：用于生成物体状态的提示。


3) Trajectory and Manipulator Control: In our framework, we sample and generate manipulator trajectories using end-effector pose in Cartesian coordinates and Euler angles. For manipulator control, we convert between Cartesian and joint space using Pybullet's built-in inverse kinematics. We employ position control, which, while straightforward, proves to be stable for the defined primitive tasks.
3) 轨迹与机械臂控制：在我们的框架中，使用末端执行器在笛卡尔坐标和欧拉角下的位姿来采样并生成机械臂轨迹。对于机械臂控制，我们使用 Pybullet 内置的逆运动学在笛卡尔空间与关节空间之间转换。我们采用位置控制，虽然方法简单，但对于所定义的基元任务表现出稳定性。


## B. Task Condition Generation with LLMs
## B. 使用大模型生成任务条件


In order to generate task conditions of the same task with novel objects using LLM, we must overcome the wellknown shortcoming of it being random and inconsistent. We managed to achieve this goal by first dividing the task condition generation problem into two parts: spatial relations & collisions generation problem, and object states generation problem. Two LLM chats are prompted differently to do these two parts separately. Also, the prompts we design emphasize the importance of remembering the examples given to it, and generating answers in the same format.
为了使用 LLM 生成含新物体的相同任务条件，我们必须克服其随机且不一致的众所周知的缺点。我们通过先将任务条件生成问题划分为两部分来实现这一目标：空间关系与碰撞生成问题，以及物体状态生成问题。分别以不同提示词驱动两个 LLM 对话来完成这两部分。此外，我们设计的提示词强调必须记住给出的示例，并以相同格式生成答案。


The prompt for spatial relations and collision generation is shown in Lst. 2 It has four components overall, including:
生成空间关系和碰撞的提示如清单 2 所示。它总体包含四个部分，包括：


- an overall description of the job for LLM, indicated in black;
- 对 LLM 的整体岗位描述，用黑色标注；


- spatial relations or object states to choose from, indicated in blue;
- 可供选择的空间关系或对象状态，用蓝色表示；


---



Imagine you are a spatial relation & collision judgment machine
想象你是一台空间关系与碰撞判断机器


I will give you a task name describing a manipulator task, the
我将给你一个描述操纵器任务的任务名，


end effector of the manipulator is a gripper.
该机械臂的末端执行器是一个夹持器。


First, you should determine what are the relevant objects in
首先，你应确定相关对象有哪些


this task.
这个任务。


Then, you should present what spatial relations these objects
然后，你应说明这些物体之间存在的空间关系


should have before (pre-conditions) and after (post-conditions)
应该在处理任务之前（前置条件）和之后（后置条件）具备


the task is processed.
该任务被处理。


Here are the spatial relations you can choose from:
以下是可供选择的空间关系：


	above, below, inside, in front of
	上方、下方、内部、前方


At last, you should present what collisions there might occur
最后，你应说明在完成任务过程中可能发生的碰撞


during the completion of the task. You must only generate
你必须只生成


collisions that include the relevant objects.
包含相关对象的碰撞。


Here are some examples:
以下是一些示例：


[Task condition examples with only spatial relation & collisionl
[仅含空间关系和碰撞的任务条件示例


Remember to strictly follow the examples' format
记住严格遵循示例格式


Do not generate any pre/post conditions except spatial relations
不要生成除我提供给你选择的空间关系之外的任何前/后条件。


I gave you to choose from.
我给你的选择。


Q: Task name: [Generalization Task Name]
问：任务名称：[泛化任务名称]


---



Listing 4: Chain-of-thoughts prompt for condition generalization.
列表 4：用于条件泛化的链式思考提示。


- examples in the form of Lst. 1 with only spatial relations included, indicated in orange;
- 以表 1 形式的示例，仅包含以橙色标注的空间关系；


- answer format indicated in pink.
- 以粉色标注的答案格式。


Finally, we present a task name to LLM for task condition generation indicated by teal. To be mentioned, object names will be represented by letters (A, B, etc.) instead of specific object names. The prompt for object state generation remains consistent in structure and composition, except that changes in descriptions and examples are made for object state generation. Detailed prompt is shown in Lst. 3
最后，我们向 LLM 提供一个以青绿色标示的任务名用于任务条件生成。需要说明的是，物体名称将用字母（A、B 等）表示，而不是具体物体名。物体状态生成的提示模板在结构和组成上保持一致，仅在描述和示例上作出调整以适应状态生成。详细提示见 Lst. 3


## C. Task Condition Generalization with LLMs
## C. 使用 LLM 的任务条件泛化


Task condition generalization with LLM follows the prompting concept we mentioned in Sec. III-B, but should be able to generalize novel but similar task conditions. Therefore, changes in prompts are made in order to let the LLM loosen up a little in order to leverage its reasoning and learning ability from examples, meanwhile making sure the answer is in the same format as examples. We add the chain of thoughts prompting into the previous prompt in Lst. 2 telling the LLM how to reason step by step, indicated in purple in Lst. 4 The prompt for object states is similar and not presented here due to space constraints.
LLM 的任务条件泛化遵循我们在 Sec. III-B 中提到的提示概念，但应能泛化出新颖但相似的任务条件。因此，对提示作出修改，使 LLM 在利用示例进行推理和学习时稍微放开限制，同时确保答案格式与示例一致。我们在 Lst. 2 的前一提示中加入了链式思维提示，告诉 LLM 如何逐步推理，在 Lst. 4 中以紫色标示。物体状态的提示类似，因篇幅限制此处不再展示。


<img src="https://cdn.noedgeai.com/bo_d4nfvdref24c73bbfi60_3.jpg?x=154&y=144&w=726&h=464&r=0"/>



Fig. 3: Framework Flow Chat During Task Execution. This figure starts with the top-left primitive task we wish to execute, and the task condition of it is generalized based on generated task conditions. Further, information in the task condition and from the environment is used to help DMP generate and adjust end-effector trajectory, achieving execution of the given primitive task.
图 3：任务执行期间的框架流程图。该图从左上角我们希望执行的原始原语任务开始，基于生成的任务条件对其任务条件进行泛化。此外，任务条件中的信息与环境信息被用于帮助 DMP 生成并调整末端执行器轨迹，从而实现给定原语任务的执行。


## D. Trajectory Learning with DMP
## D. 使用 DMP 的轨迹学习


Dynamic Movement Primitives (DMP) is a trajectory imitation learning method with high nonlinear characteristics and real-time performance ability. Meanwhile, it is capable of generating similar shaping trajectories to different goal positions. A large amount of work has been done based on the original formulation raised by [49, 50], and we choose to stick with the original discrete DMP, considering it is aimed to solve trajectory optimization problems in Cartesian space, and shows simplicity yet efficiency in our framework.
动态运动基元（DMP）是一种具有强非线性特性和实时性能的轨迹模仿学习方法。同时，它能够生成到不同目标位置的相似形状轨迹。基于 [49, 50] 提出之原始公式已有大量工作，我们选择沿用原始离散 DMP，考虑到其旨在解决笛卡尔空间的轨迹优化问题，并在我们的框架中表现出简洁且高效。


The basic formula of the discrete DMP is described by Eq. 1 $y$ is the current system status (e.g. position) and $\dot{y}$ , $\ddot{y}$ being its first and second derivatives. $g$ is the goal status. The first term on the right is a PD controller with ${\alpha }_{y}$ and ${\beta }_{y}$ representing the $\mathrm{P}$ parameter and $\mathrm{D}$ parameter and $\tau$ controlling the speed of convergence.
离散 DMP 的基本公式如 Eq. 1 所示，$y$ 为当前系统状态（例如位置），$\dot{y}$、$\ddot{y}$ 为其一阶和二阶导数。$g$ 为目标状态。右侧第一项为带有 ${\alpha }_{y}$ 和 ${\beta }_{y}$ 的 PD 控制器，分别表示 $\mathrm{P}$ 参数和 $\mathrm{D}$ 参数，$\tau$ 控制收敛速度。


$$
{\tau }^{2}\ddot{y} = {\alpha }_{y}\left( {{\beta }_{y}\left( {g - y}\right)  - \tau \dot{y}}\right)  + f \tag{1}
$$



The nonlinear term $f$ is implemented by the normalized weighting of multiple nonlinear basis functions to control the process of convergence. The variable $x$ in it satisfies a first-order system, making the nonlinear term time-independent. Eventually,it can be described by Eq. 2 N and ${w}_{i}$ are the number and weight of the basis functions ${\Psi }_{i}$ respectively, which we choose to be the Gaussian basis function described by Eq. 3, $\sigma$ and $c$ are its width and center position.
非线性项 $f$ 通过多个非线性基函数的归一化加权来实现，以控制收敛过程。其中变量 $x$ 满足一阶系统，使非线性项与时间无关。最终可由 Eq. 2 描述，N 和 ${w}_{i}$ 分别为基函数的数量和权重 ${\Psi }_{i}$，我们选择由 Eq. 3 描述的高斯基函数，$\sigma$ 和 $c$ 为其宽度与中心位置。


$$
f\left( {x,g}\right)  = \frac{\mathop{\sum }\limits_{{i = 1}}^{N}{\Psi }_{i}\left( x\right) {w}_{i}}{\mathop{\sum }\limits_{{i = 1}}^{N}{\Psi }_{i}\left( x\right) }x\left( {g - {y}_{0}}\right) ,\;\tau \dot{x} =  - {\alpha }_{x}x \tag{2}
$$



$$
\Psi \left( x\right)  = \exp \left( \frac{-1}{2{\sigma }^{2}{\left( x - c\right) }^{2}}\right) \tag{3}
$$



In our work, we sample the trajectory of the end effector $y,\dot{y}$ and $\ddot{y}$ in each dimension mentioned in Sec. III-A.3,and the nonlinear term can be represented by rearranging Eq. 1 and put $f$ to the left side. Then,with hyperparameters in Eq. 1 and Eq. 3 set as: $N = {100},{\alpha }_{y} = {60},{\beta }_{y} = \frac{{\alpha }_{y}}{4},{\alpha }_{x} = \; 1,{x}_{0} = 1$ ,we conducted the locally weighted regression method in [50] to calculate rest of the variables, and mathematical expressions for calculation will not presented here. The trajectory encoded by DMP will be part of the demo data shown in Fig. 2
在我们的工作中，我们对 Sec. III-A.3 中提及的每个维度对末端执行器轨迹 $y,\dot{y}$ 和 $\ddot{y}$ 进行采样，非线性项可通过重排 Eq. 1 并将 $f$ 移到左侧来表示。然后，在将 Eq. 1 和 Eq. 3 的超参数设为：$N = {100},{\alpha }_{y} = {60},{\beta }_{y} = \frac{{\alpha }_{y}}{4},{\alpha }_{x} = \; 1,{x}_{0} = 1$ 后，我们采用 [50] 中的局部加权回归方法计算其余变量，具体数学表达不在此呈现。由 DMP 编码的轨迹将作为示例数据的一部分，如图 2 所示。


Algorithm 1 Pre-Condition Satisfaction
算法 1 先验条件满足


---



Require:
要求：


	T_Cond - Task Condition
	T_Cond - 任务条件


	function SATISFYPRECOND (T_Cond)
	function SATISFYPRECOND (T_Cond)


		currentConds ← FROMENVPC()
		currentConds ← FROMENVPC()


		preConds ← FROMCOND(T_Cond)
		preConds ← FROMCOND(T_Cond)


		for preCond in preConds do
		for preCond in preConds do


			if preCond not in currentCond then
			if preCond not in currentCond then


				newT_Name ← COND2TASK(preCond)
				newT_Name ← COND2TASK(preCond)


				newT_Cond ← GENCOND(newT_Name)
				newT_Cond ← GENCOND(newT_Name)


				SATISFYPRECOND (newT_Cond)
				SATISFYPRECOND (newT_Cond)


				CONTROLWITHCOND(newT_Cond)
				CONTROLWITHCOND(newT_Cond)


			end if
			end if


			if RUNTIME > 60s then
			if RUNTIME > 60s then


				break
				break


			end if
			end if


		end for
		end for


	end function
	end function


---



Algorithm 2 Trajectory Generation With Condition
算法 2 带条件的轨迹生成


---



Require: ${T}_{ - }$ Cond - Task Condition
Require: ${T}_{ - }$ Cond - 任务条件


	: function CONTROLWITHCOND( ${T}_{ - }$ Cond)
	: function CONTROLWITHCOND( ${T}_{ - }$ Cond)


			taskColl,postConds $\leftarrow$ FromCond( ${T}_{ - }$ Cond)
			taskColl,postConds $\leftarrow$ FromCond( ${T}_{ - }$ Cond)


			targetPos $\leftarrow$ FromEnvPC()
			targetPos $\leftarrow$ FromEnvPC()


			goalPos $\leftarrow$ targetObjPos
			goalPos $\leftarrow$ targetObjPos


			trajDMP[:] ← GenDMP(goalPos)
			trajDMP[:] ← GenDMP(goalPos)


			while RUNTIME $\leq  {60}$ s or end(trajDMP) or postConds satisfied do
			while RUNTIME $\leq  {60}$ s or end(trajDMP) or postConds satisfied do


				ControlManipulator(trajDMP[+1])
				ControlManipulator(trajDMP[+1])


				currentColl ← FromEnvPC()
				currentColl ← FromEnvPC()


				if currentColl not in taskColl then
				if currentColl not in taskColl then


						ControlManipulator(trajDMP[-10])
						ControlManipulator(trajDMP[-10])


						collPos $\leftarrow$ FromEnvPC()
						collPos $\leftarrow$ FromEnvPC()


						goalPos $\leftarrow$ goalPos + (goalPos - collPos)
						goalPos $\leftarrow$ goalPos + (goalPos - collPos)


						trajDMP[:-1] $\leftarrow$ GenDMP(goalPos)
						trajDMP[:-1] $\leftarrow$ GenDMP(goalPos)


				end if
				end if


			end while
			end while


	end function
	end function


---



## E. Trajectory Generation & Adjustment via Task Conditions
## E. 通过任务条件的轨迹生成与调整


The method to generate trajectories using DMP is straightforward with parameters learned in Sec. III-D With new starting position ${y}_{0}$ and goal position $g$ given,we can use them to generate the nonlinear term described by Eq. 2 as well as position at any point by running a numerical simulation of the second-order system described by Eq. 1 In our framework, this method will be guided by corresponding task conditions to help generate and adjust trajectory. Two main functions are used sequentially, which are shown as yellow blocks in Fig. 3
使用 DMP 生成轨迹的方法很直接，参数在第 III-D 节中学习得到。给定新的起始位置 ${y}_{0}$ 和目标位置 $g$，我们可用它们生成由式 (2) 描述的非线性项，并通过对式 (1) 描述的二阶系统进行数值仿真来得到任意时刻的位置。在我们的框架中，该方法将由相应的任务条件引导以帮助生成和调整轨迹。顺序使用两个主要函数，如图 3 中的黄色块所示。


The first function is pre-condition satisfaction. It is conducted before reproducing a primitive task and is aimed to satisfy the pre-conditions of the task before execution. To achieve this, a recursive algorithm is used and its pseudo code is shown as Alg. 1. Given the task condition, the algorithm gets the pre-conditions in it and compares them to the current conditions in the environment acquired from environment information mentioned in Sec. III-A.2 Then, the pre-conditions that are not currently satisfied will raise a primitive task aiming to satisfy it. The mapping between conditions and primitive tasks is done by humans in advance, for example, 'gripper grasping bottle' not satisfied will lead to 'grasp bottle' as a new primitive task, and its pre-condition will be generated and satisfied recursively.
第一个函数是前置条件满足。它在重现一个原始任务之前进行，目的是在执行前满足该任务的前置条件。为此使用了一个递归算法，其伪代码见算法 1。给定任务条件，算法获取其中的前置条件并将其与第 III-A.2 节所述环境信息中获取的当前环境条件进行比较。然后，当前未满足的前置条件将触发一个旨在满足它的原始任务。条件与原始任务之间的映射由人工事先完成，例如“夹持器夹住瓶子”未满足将导致“抓瓶子”作为新的原始任务，且其前置条件将被递归生成并满足。


TABLE I: Task descriptions. Notations: PT - primitive task, LHT - long-horizon task, * - primitive task with novel objects.
表 I：任务描述。符号：PT - 原始任务，LHT - 长时序任务，* - 含新物体的原始任务。


<table><tr><td>PT1</td><td>Grasp Object</td><td>PT6</td><td>Fold Object</td></tr><tr><td>PT2</td><td>Release Object</td><td>PT7</td><td>Move Object A to position</td></tr><tr><td>PT3</td><td>Open Object</td><td>PT8</td><td>Move Object A On Top of Object B</td></tr><tr><td>PT4</td><td>Close Object</td><td>PT9</td><td>Move Object A into Object B</td></tr><tr><td>PT5</td><td>Tilt Object</td><td>PT10</td><td>Move Object A In Front of Object B</td></tr><tr><td>LHT1</td><td colspan="3" rowspan="2">Pick up a bottle, put the bottle into the box, close the box. <br> To put the bowl into the bottom drawer of the cabinet, first close the top drawer, then open the bottom drawer, put the bowl into the bottom drawer and then close the drawer.</td></tr><tr><td>LHT2</td></tr><tr><td>LHT3</td><td colspan="3" rowspan="2">Hold a mug up, place it on the table, and put toothpaste into the mug. <br> Grasp the spatula and place it onto the cloth, then fold the cloth.</td></tr><tr><td>LHT4</td></tr><tr><td>LHT1*</td><td colspan="3" rowspan="2">Pick up a bowl, put the bowl into the box, and close the box. <br> To put the bowl into the upper drawer of the cabinet, first close the bottom drawer, then open the upper drawer, put the bowl into the upper drawer and then close the drawer.</td></tr><tr><td>LHT2*</td></tr></table>
<table><tbody><tr><td>PT1</td><td>抓取物体</td><td>PT6</td><td>折叠物体</td></tr><tr><td>PT2</td><td>释放物体</td><td>PT7</td><td>将物体A移动到位置</td></tr><tr><td>PT3</td><td>打开物体</td><td>PT8</td><td>将物体A放置在物体B上方</td></tr><tr><td>PT4</td><td>关闭物体</td><td>PT9</td><td>将物体A放入物体B内</td></tr><tr><td>PT5</td><td>倾斜物体</td><td>PT10</td><td>将物体A移到物体B前方</td></tr><tr><td>LHT1</td><td colspan="3" rowspan="2">拿起一个瓶子，把瓶子放进箱子，并关上箱子。<br/>要把碗放进橱柜的下层抽屉，先关上上层抽屉，然后打开下层抽屉，把碗放进下层抽屉，最后关闭抽屉。</td></tr><tr><td>LHT2</td></tr><tr><td>LHT3</td><td colspan="3" rowspan="2">举起一个杯子，放到桌上，然后把牙膏放进杯子。<br/>抓起抹刀，把它放到布上，然后折叠布。</td></tr><tr><td>LHT4</td></tr><tr><td>LHT1*</td><td colspan="3" rowspan="2">拿起一个碗，把碗放进箱子，并关上箱子。<br/>要把碗放进橱柜的上层抽屉，先关上下层抽屉，然后打开上层抽屉，把碗放进上层抽屉，最后关闭抽屉。</td></tr><tr><td>LHT2*</td></tr></tbody></table>


The second function is trajectory adjustment with task conditions to prevent unintended collisions during execution. The pseudo-code is shown as Alg. 2 First, we generate a trajectory given the target object center from environment information as goal position using DMP, primitive tasks without a target object, such as moving or folding, goal position is defined by the difference between starting and ending positions during demonstration. During the execution of the trajectory, we monitor collisions in the environment. If there is an unwanted collision not in task condition, we stop executing and go backward 10 trajectory points. Then, we update the goal position regarding where the collision happens and adjust the rest of the trajectory. The algorithm terminates when all post-conditions are met or if the execution time exceeds a predefined threshold.
第二个功能是基于任务条件的轨迹调整，以防止执行过程中发生非预期碰撞。伪代码见 Alg. 2。首先，我们使用 DMP 根据环境信息中目标物体中心作为目标位置生成轨迹；对于无目标物体的原始任务（如移动或折叠），目标位置由示范过程中的起止位置差定义。在轨迹执行过程中，我们监测环境中的碰撞。如果出现不属于任务条件的意外碰撞，我们停止执行并回退 10 个轨迹点。然后，根据碰撞发生的位置更新目标位置并调整剩余轨迹。当所有后置条件满足或执行时间超过预设阈值时，算法终止。


## IV. SIMULATION
## IV. 仿真


To evaluate our framework, we design a challenging Robotic Manipulation Task Suite in Pybullet [51]. The environment consists of two 7 Dof robots Franka and Kinova with a kitchen scene including various interactive objects. It contains 10 diverse primitive tasks (37 if considering different objects) and 4 long-horizon tasks in simulation.
为评估我们的框架，我们在 Pybullet [51] 中设计了一个具挑战性的机器人操作任务套件。环境由两台 7 自由度机器人 Franka 和 Kinova 组成，场景为包含多种可交互物体的厨房。仿真中包含 10 个多样的原始任务（若考虑不同物体则为 37 个）和 4 个长时序任务。


These primitive tasks cover a broad spectrum of robotic manipulation skills. Our simulation includes 1) rigid object manipulation such as grasping and moving, 2) articulated object manipulation such as open/close the drawer/box, 3) periodic manipulation such as tilting the mug, 4) soft object manipulation such as grasp the cloth, and 5) dual-arm manipulation tasks, e.g., folding the cloth. The detailed description of primitive tasks and long-horizon tasks is shown in Tab. 1 We visualize our introduced simulator in Fig. 4 (left).
这些原始任务覆盖了广泛的机器人操作技能。我们的仿真包括：1) 刚性物体操作，如抓取和移动；2) 铰接物体操作，如打开/关闭抽屉/盒子；3) 周期性操作，如倾斜杯子；4) 软体物体操作，如抓取布料；5) 双臂操作任务，例如折叠布料。原始任务和长时序任务的详细描述见表 1。我们在图 4（左）中展示了所引入的模拟器。


## V. EXPERIMENTS
## V. 实验


## A. Task Condition Generation & Generalization Experiment
## A. 任务条件生成与泛化实验


First, we evaluate the ability of our framework to generate and generalize task conditions on all 10 primitive tasks. The LLM (GPT-3.5) is provided with condition examples.
首先，我们在所有 10 个原始任务上评估框架生成与泛化任务条件的能力。LLM（GPT-3.5）提供了条件示例。


TABLE II: Task condition generation and generalization.
表 II：任务条件生成与泛化。


<table><tr><td rowspan="2">PT Name</td><td colspan="2">Generation</td><td colspan="2">Generalization</td></tr><tr><td>LLM</td><td>FromEnv</td><td>LLM</td><td>FromEnv</td></tr><tr><td>Grasp</td><td>100%</td><td>100%</td><td>24%</td><td>-</td></tr><tr><td>Release</td><td>100%</td><td>100%</td><td>36%</td><td>-</td></tr><tr><td>Open</td><td>90%</td><td>100%</td><td>18%</td><td>-</td></tr><tr><td>Close</td><td>100%</td><td>100%</td><td>25%</td><td>-</td></tr><tr><td>Tilt</td><td>100%</td><td>100%</td><td>27%</td><td>-</td></tr><tr><td>Fold</td><td>100%</td><td>95%</td><td>30%</td><td>-</td></tr><tr><td>Move</td><td>81%</td><td>100%</td><td>40%</td><td>-</td></tr><tr><td>MoveInTo</td><td>90%</td><td>91.7%</td><td>32%</td><td>-</td></tr><tr><td>MoveOnTop</td><td>90%</td><td>86.7%</td><td>36%</td><td>-</td></tr><tr><td>MoveInFront</td><td>100%</td><td>100%</td><td>27%</td><td>-</td></tr></table>
<table><tbody><tr><td rowspan="2">PT 名称</td><td colspan="2">世代</td><td colspan="2">泛化</td></tr><tr><td>大模型</td><td>来自环境</td><td>大模型</td><td>来自环境</td></tr><tr><td>抓取</td><td>100%</td><td>100%</td><td>24%</td><td>-</td></tr><tr><td>释放</td><td>100%</td><td>100%</td><td>36%</td><td>-</td></tr><tr><td>打开</td><td>90%</td><td>100%</td><td>18%</td><td>-</td></tr><tr><td>关闭</td><td>100%</td><td>100%</td><td>25%</td><td>-</td></tr><tr><td>倾斜</td><td>100%</td><td>100%</td><td>27%</td><td>-</td></tr><tr><td>折叠</td><td>100%</td><td>95%</td><td>30%</td><td>-</td></tr><tr><td>移动</td><td>81%</td><td>100%</td><td>40%</td><td>-</td></tr><tr><td>移入</td><td>90%</td><td>91.7%</td><td>32%</td><td>-</td></tr><tr><td>移到上方</td><td>90%</td><td>86.7%</td><td>36%</td><td>-</td></tr><tr><td>移到前方</td><td>100%</td><td>100%</td><td>27%</td><td>-</td></tr></tbody></table>


TABLE III: Primitive task execution success rate.
表 III：原语任务执行成功率。


<table><tr><td>PT Name</td><td>Baseline</td><td>w/o Cond</td><td>w/ Cond</td></tr><tr><td>Grasp</td><td>8.3%</td><td>87.5%</td><td>93.1%</td></tr><tr><td>Release</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Open</td><td>6.7%</td><td>90%</td><td>90%</td></tr><tr><td>Close</td><td>31.7%</td><td>89.6%</td><td>89.6%</td></tr><tr><td>Tilt</td><td>86.7%</td><td>75%</td><td>85%</td></tr><tr><td>Fold</td><td>0%</td><td>80.2%</td><td>85%</td></tr><tr><td>Move</td><td>16.65%</td><td>90.9%</td><td>94.1%</td></tr><tr><td>MoveInTo</td><td>68.8%</td><td>90.9%</td><td>94.4%</td></tr><tr><td>MoveOnTop</td><td>45.6%</td><td>81.9%</td><td>91.1%</td></tr><tr><td>MoveInFront</td><td>13.3%</td><td>100%</td><td>100%</td></tr></table>
<table><tbody><tr><td>PT 名称</td><td>基线</td><td>无条件</td><td>有条件</td></tr><tr><td>抓取</td><td>8.3%</td><td>87.5%</td><td>93.1%</td></tr><tr><td>释放</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>打开</td><td>6.7%</td><td>90%</td><td>90%</td></tr><tr><td>关闭</td><td>31.7%</td><td>89.6%</td><td>89.6%</td></tr><tr><td>倾斜</td><td>86.7%</td><td>75%</td><td>85%</td></tr><tr><td>折叠</td><td>0%</td><td>80.2%</td><td>85%</td></tr><tr><td>移动</td><td>16.65%</td><td>90.9%</td><td>94.1%</td></tr><tr><td>移入</td><td>68.8%</td><td>90.9%</td><td>94.4%</td></tr><tr><td>移至顶部</td><td>45.6%</td><td>81.9%</td><td>91.1%</td></tr><tr><td>移到前面</td><td>13.3%</td><td>100%</td><td>100%</td></tr></tbody></table>


Comparison is made with task conditions generated from environments. A successfully generated task condition should contain accurate and enough information to guide the execution of the primitive task. The result is shown in Tab. II.
与从环境生成的任务条件进行比较。成功生成的任务条件应包含准确且充分的信息以指导原子任务的执行。结果见表 II。


When we evaluate the result of the experiment, the high success rate in task generation shows our properly prompted LLM has consistent abilities to accurately determine relevant objects and generate correct task conditions in an expected format. It even outperformed the success rate of generation using environmental information in some cases. When it comes to task condition generalization, the LLM still has a chance to figure out proper task condition for every primitive task, with the highest success rate being 40%. This shows our prompting method is feasible and gives the LLM task condition generalization ability. In contrast, another method clearly does not have the ability to generalize since no demonstration of the corresponding primitive task was given.
在评估实验结果时，任务生成的高成功率表明我们适当提示的 LLM 能稳定地准确识别相关对象并以期望格式生成正确的任务条件。在某些情况下，甚至超过了使用环境信息生成的成功率。关于任务条件的泛化，LLM 仍有机会为每个原子任务找出合适的任务条件，最高成功率为 40%。这表明我们的提示方法是可行的，赋予了 LLM 任务条件泛化能力。相反，另一种方法显然不具备泛化能力，因为没有给出对应原子任务的示例。


## B. Primitive Task Execution Experiment
## B. 原子任务执行实验


Secondly, we evaluate the ability of our framework to do imitation learning with DMP, and leverage task conditions to generate and adjust DMP trajectories. We run the execution of all 10 primitive tasks using three different methods:
其次，我们评估框架使用 DMP 进行模仿学习并利用任务条件生成和调整 DMP 轨迹的能力。我们用三种不同方法运行所有 10 个原子任务的执行：


- the image-based imitation learning baseline;
- 基于图像的模仿学习基线；


- with only DMP learning and generating trajectories;
- 仅进行 DMP 学习并生成轨迹；


- with correct task conditions generating and adjusting trajectories.
- 使用正确的任务条件生成并调整轨迹。


From Tab. III, the image-based baseline achieves much worse performance in all but PT5 execution, with ups and downs from 0 to 100, since it highly depends on good quality images with certain angles. In contrast, trajectories generated by DMP alone show better robustness with a consistent over 80% success rate in most primitive task execution. Then, when we compare methods [w/o task condition], we can see the success rate on every primitive task is better if not the same when using our framework's method to generate and adjust trajectories. Though the improvement it makes seems not exceptional, it will make an impact when the success rate is to be multiplied during a long-horizon task.
从表 III 可见，基于图像的基线在除 PT5 外的所有任务上表现远差，结果起伏在 0 到 100 之间，因为它高度依赖于具有特定角度的高质量图像。相比之下，仅由 DMP 生成的轨迹在大多数原子任务执行中表现出更好的鲁棒性，成功率稳定在 80% 以上。然后，当我们比较 [w/o task condition] 方法时，可以看到使用我们框架的方法生成和调整轨迹后，每个原子任务的成功率要么更好要么相同。虽然其带来的改进似乎并不显著，但在长时序任务中成功率需要相乘时，这种影响会显现出来。


TABLE IV: Evaluation of long-horizon tasks. '-' for infeasible.
表 IV：长时序任务评估。 '-' 表示不可行。


<table><tr><td>Task Name</td><td>Baseline</td><td>w/o Cond</td><td>w/ Cond FromEnv</td><td>w/ Cond from LLM</td></tr><tr><td colspan="5">Same as demonstration:</td></tr><tr><td>LHT1</td><td>0.8%</td><td>50%</td><td>59.63%</td><td>60.75%</td></tr><tr><td>LHT2</td><td>0%</td><td>35%</td><td>43.72%</td><td>32.47%</td></tr><tr><td>LHT3</td><td>0%</td><td>20%</td><td>31.80%</td><td>21.25%</td></tr><tr><td>LHT4</td><td>0%</td><td>40%</td><td>45.30%</td><td>49.50%</td></tr><tr><td colspan="5">Generalize to novel objects:</td></tr><tr><td>LHT1*</td><td>-</td><td>-</td><td>-</td><td>48.6%</td></tr><tr><td>LHT2*</td><td>-</td><td>-</td><td>-</td><td>28.8%</td></tr><tr><td colspan="5">Generalize to novel primitive tasks:</td></tr><tr><td>LHT1</td><td>-</td><td>-</td><td>-</td><td>19.5%</td></tr><tr><td>LHT2</td><td>-</td><td>-</td><td>-</td><td>10.1%</td></tr><tr><td>LHT3</td><td>-</td><td>-</td><td>-</td><td>7.7%</td></tr><tr><td>LHT4</td><td>-</td><td>-</td><td>-</td><td>15.5%</td></tr></table>
<table><tbody><tr><td>任务名称</td><td>基线</td><td>无条件</td><td>带环境条件</td><td>来自LLM的条件</td></tr><tr><td colspan="5">与示例相同：</td></tr><tr><td>LHT1</td><td>0.8%</td><td>50%</td><td>59.63%</td><td>60.75%</td></tr><tr><td>LHT2</td><td>0%</td><td>35%</td><td>43.72%</td><td>32.47%</td></tr><tr><td>LHT3</td><td>0%</td><td>20%</td><td>31.80%</td><td>21.25%</td></tr><tr><td>LHT4</td><td>0%</td><td>40%</td><td>45.30%</td><td>49.50%</td></tr><tr><td colspan="5">推广到新物体：</td></tr><tr><td>LHT1*</td><td>-</td><td>-</td><td>-</td><td>48.6%</td></tr><tr><td>LHT2*</td><td>-</td><td>-</td><td>-</td><td>28.8%</td></tr><tr><td colspan="5">推广到新原始任务：</td></tr><tr><td>LHT1</td><td>-</td><td>-</td><td>-</td><td>19.5%</td></tr><tr><td>LHT2</td><td>-</td><td>-</td><td>-</td><td>10.1%</td></tr><tr><td>LHT3</td><td>-</td><td>-</td><td>-</td><td>7.7%</td></tr><tr><td>LHT4</td><td>-</td><td>-</td><td>-</td><td>15.5%</td></tr></tbody></table>


## C. Long-horizon Task Execution Experiment
## C. 长期任务执行实验


Finally, we combine previous experiments to evaluate the overall ability of our framework to generate, generalize task conditions, then use it to generate and adjust DMP trajectory.
最后，我们结合之前的实验评估框架生成、泛化任务条件的整体能力，然后用其生成并调整 DMP 轨迹。


1) Experiment Setting: In order to improve the integrity of our execution experiment, and to maximize evaluation of different modules in our framework, we conduct the experiments under three cases. Before experiments, demo data including task condition and encoded trajectory of each primitive task is gained from demonstrations of long-horizon tasks (LHT1 to LHT4). And this experiment will take the success rate in Exp. V-A into consideration as well. The setting of the three cases for execution is as follows:
1) 实验设置：为提高执行实验的完整性并最大化评估框架中不同模块，我们在三种情况下进行实验。实验前，从长期任务（LHT1 到 LHT4）的示范中获取示例数据，包括每个原始任务的任务条件和编码轨迹。同时本实验也将考虑 Exp. V-A 的成功率。三种执行情况设置如下：


- same as demonstration: LHT1 to LHT4 using the same objects as in the demonstrations, with all example task conditions provided to the LLM.
- 与示范相同：LHT1 到 LHT4 使用与示范中相同的物体，并向 LLM 提供所有示例任务条件。


- generalize to novel objects: LHT1* and LHT2* with novel objects, with all example task conditions from LHT1 and LHT2.
- 泛化到新物体：LHT1* 和 LHT2* 使用新物体，示例任务条件来自 LHT1 和 LHT2。


- generalize to novel primitive tasks: LHT1 to LHT4 with a randomly selected unseen (novel) primitive task, the example conditions excluding this primitive task are provided to LLM.
- 泛化到新原始任务：LHT1 到 LHT4 中随机选取一个未见（新）原始任务，向 LLM 提供不包含该原始任务的示例条件。


Similar to Exp. V-B] we run execution and evaluation under each case using four different methods:
与 Exp. V-B 类似，我们在每种情况下使用四种不同方法运行执行与评估：


- the image-based imitation learning baseline;
- 基于图像的模仿学习基线；


- with only DMP learning and generating trajectories;
- 仅有 DMP 学习与轨迹生成；


- with task conditions from the environment to guide trajectories;
- 使用来自环境的任务条件来引导轨迹；


- with task conditions from LLM to guide trajectories.
- 使用来自 LLM 的任务条件来引导轨迹。


2) Experiment Results: We first evaluate the result under same as demonstration case in Tab. IV. The performances of the baseline and with only DMP meet our expectations regarding Exp. V-B, since the success rate of a long-horizon task is approximately the product of that the primitive tasks have. The baseline performs even worse due to more inconsistent images during long-horizon tasks. As for methods [w/o task condition], the improvement in each primitive task adds up, leading to around ${10}\%$ better with task condition generating and adjusting the trajectories, even considering the chance for failure in task condition generating. The difference between using environment information and LLM to generate task conditions depends on the result in Exp. V-A with LLM still showing great competitiveness, outperforming in two long-horizon executions.
2) 实验结果：我们先在表 IV 中评估“与示范相同”情形的结果。基线与仅使用 DMP 的表现符合我们对 Exp. V-B 的预期，因为长期任务的成功率大致等于各原始任务成功率的乘积。基线表现更差是由于长期任务中图像更不一致。对于 [w/o task condition] 方法，各原始任务的改进累加，导致在生成并调整轨迹的任务条件下平均提升约 ${10}\%$，即使考虑到任务条件生成可能失败的情况。使用环境信息与使用 LLM 生成任务条件的差异取决于 Exp. V-A 的结果，LLM 仍显示出强竞争力，在两次长期执行中表现更好。


<img src="https://cdn.noedgeai.com/bo_d4nfvdref24c73bbfi60_5.jpg?x=916&y=148&w=726&h=248&r=0"/>



Fig. 4: Our simulator vs. real-world experimental setup.
图 4：我们的仿真器与真实世界实验装置对比。


When it comes to other two cases, any slight backwardness of our framework before does not seem so important, as it is the only feasible method to do generalization to novel objects and primitive tasks. Only an average $8\%$ decrease is noticed when generalizing the same task to novel objects. While for novel primitive tasks, because of the success rate shown in Exp. V-A experiencing a massive drop, there is an average decrease of 27.5%, leaving only around 13% success rate. But considering its difficulty, this is already a breakthrough compared to our other methods.
对于另外两种情形，框架之前的任何轻微落后似乎并不重要，因为它是唯一可行的方法来实现对新物体和原始任务的泛化。将相同任务泛化到新物体时仅注意到平均 $8\%$ 的下降。而对于新原始任务，由于 Exp. V-A 中的成功率大幅下降，平均下降为 27.5%，仅剩约 13% 的成功率。但考虑到其难度，相较于我们的其他方法这已是突破。


3) Advantage: We highlight an interesting advantage of our framework in executing long-horizon tasks. It has the ability to generate the needed trajectory on its own if an over far-sighted primitive task is given. For example, if we ask the manipulator to move a bottle before giving the instruction on grasping it, our framework can generate a grasping trajectory autonomously. Not until grasping is successfully finished will the manipulator begin moving it. This shows a certain intelligence our framework has and from certain aspects, can mean a higher success rate in long-horizon task execution.
3) 优势：我们强调框架在执行长期任务时的一个有趣优势。当给定一个过于超前的原始任务时，它能自主生成所需轨迹。例如，如果我们要求机械臂在给出抓取指令前先移动一个瓶子，框架可以自主生成抓取轨迹。机械臂只有在抓取成功完成后才开始移动它。这表明框架具有一定的智能性，并在某些方面能提升长期任务执行的成功率。


## D. Real-world Experiments
## D. 真实世界实验


To demonstrate the practicality of our to perform long-horizon tasks in real scenarios, we set up an environment shown in Fig. 4 (right). It shows objects such as a bowl, a bottle that the manipulator can grasp, and a microwave that the manipulator can open and close. The environment features a dual-arm MOVO [52] robot which has two 7 DoF manipulators and a Kinect RGB-D camera overhead. We use Segment Anything [53] to obtain segmented point clouds of the surroundings and objects. Position control is performed as we use rangeIK [54] for solving inverse kinematics. Due to space constraints, we present our real-world experiments in the supplemental video and the project website.
为了展示我们在真实场景中执行长时序任务的实用性，我们搭建了如图4（右）所示的环境。该环境展示了碗、可被机械臂抓取的瓶子以及可被机械臂开关的微波炉等物体。环境采用双臂 MOVO [52] 机器人，配备两根 7 自由度操作臂和头顶的 Kinect RGB-D 相机。我们使用 Segment Anything [53] 获取周围环境和物体的分割点云。由于我们使用 rangeIK [54] 求解逆运动学，因此采用位置控制。由于篇幅限制，我们将在补充视频和项目网站中展示真实世界的实验。


## VI. CONCLUSION
## VI. 结论


This work explores the potential of LLMs on primitive task condition generalization for generalizable long-horizon manipulations with novel objects and unseen tasks. The generated conditions are then utilized to steer the generation of low-level manipulation trajectories using DMP. A robotic manipulation task suite based on Pybullet is also introduced. We conduct experiments in both our simulation and real-world scenarios, demonstrating the effectiveness of the proposed framework for long-horizon manipulation tasks and its ability to generalize to tasks involving novel objects or unseen scenarios. While our framework shows promise, there is room for improvement and a more versatile trajectory generator could complement our framework.
本工作探索了大型语言模型在原子任务条件泛化方面的潜力，以实现对新奇物体和未见任务具有泛化能力的可推广长时序操作。生成的条件随后用于引导以 DMP 生成的低级操作轨迹。我们还引入了基于 Pybullet 的机器人操作任务套件。我们在仿真和真实场景中进行了实验，证明了所提框架在长时序操作任务上的有效性及其对包含新奇物体或未见场景任务的泛化能力。尽管框架展现了前景，仍有改进空间，更通用的轨迹生成器可以补强本框架。


## REFERENCES
## 参考文献


[1] L. P. Kaelbling and T. Lozano-Pérez, "Hierarchical task and motion planning in the now," in 2011 IEEE International Conference on Robotics and Automation. IEEE, 2011, pp. 1470-1477.
[1] L. P. Kaelbling 和 T. Lozano-Pérez, "Hierarchical task and motion planning in the now," 载于 2011 IEEE International Conference on Robotics and Automation. IEEE, 2011, 页 1470-1477。


[2] T. Migimatsu and J. Bohg, "Object-centric task and motion planning in dynamic environments," IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 844-851, 2020.
[2] T. Migimatsu 和 J. Bohg, "Object-centric task and motion planning in dynamic environments," IEEE Robotics and Automation Letters, 卷 5, 期 2, 页 844-851, 2020。


[3] M. Toussaint, "Logic-geometric programming: an optimization-based approach to combined task and motion planning," in Proceedings of the 24th International Conference on Artificial Intelligence, 2015, pp. 1930-1936.
[3] M. Toussaint, "Logic-geometric programming: an optimization-based approach to combined task and motion planning," 载于第24届国际人工智能大会论文集, 2015, 页 1930-1936。


[4] A. G. Barto and S. Mahadevan, "Recent advances in hierarchical reinforcement learning," Discrete event dynamic systems, vol. 13, no. 1, pp. 41-77, 2003.
[4] A. G. Barto 和 S. Mahadevan, "Recent advances in hierarchical reinforcement learning," Discrete event dynamic systems, 卷 13, 期 1, 页 41-77, 2003。


[5] T. D. Kulkarni, K. R. Narasimhan, A. Saeedi, and J. B. Tenenbaum, "Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation," Advances in Neural Information Processing Systems, pp. 3682-3690, 2016.
[5] T. D. Kulkarni, K. R. Narasimhan, A. Saeedi, 和 J. B. Tenenbaum, "Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation," Advances in Neural Information Processing Systems, 页 3682-3690, 2016。


[6] Z. Liu, A. Bahety, and S. Song, "Reflect: Summarizing robot experiences for failure explanation and correction," arXiv preprint arXiv:2306.15724, 2023.
[6] Z. Liu, A. Bahety, 和 S. Song, "Reflect: Summarizing robot experiences for failure explanation and correction," arXiv 预印本 arXiv:2306.15724, 2023。


[7] I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Trem-blay, D. Fox, J. Thomason, and A. Garg, "Progprompt: Generating situated robot task plans using large language models," arXiv preprint arXiv:2209.11302, 2022.
[7] I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Trem-blay, D. Fox, J. Thomason, 和 A. Garg, "Progprompt: Generating situated robot task plans using large language models," arXiv 预印本 arXiv:2209.11302, 2022。


[8] W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, et al., "Inner monologue: Embodied reasoning through planning with language models," arXiv preprint arXiv:2207.05608, 2022.
[8] W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, 等, "Inner monologue: Embodied reasoning through planning with language models," arXiv 预印本 arXiv:2207.05608, 2022。


[9] W. Huang, F. Xia, D. Shah, D. Driess, A. Zeng, Y. Lu, P. Florence, I. Mordatch, S. Levine, K. Hausman, et al., "Grounded decoding: Guiding text generation with grounded models for robot control," arXiv preprint arXiv:2303.00855, 2023.
[9] W. Huang, F. Xia, D. Shah, D. Driess, A. Zeng, Y. Lu, P. Florence, I. Mordatch, S. Levine, K. Hausman, 等, "Grounded decoding: Guiding text generation with grounded models for robot control," arXiv 预印本 arXiv:2303.00855, 2023。


[10] S. Nair and C. Finn, "Hierarchical foresight: Self-supervised learning of long-horizon tasks via visual subgoal generation," in International Conference on Learning Representations, 2020. [Online]. Available: https://openreview.net/forum?id= H1gzR2VKDH
[10] S. Nair 和 C. Finn, "Hierarchical foresight: Self-supervised learning of long-horizon tasks via visual subgoal generation," 载于 International Conference on Learning Representations, 2020. [在线]. 可用: https://openreview.net/forum?id= H1gzR2VKDH


[11] K. Pertsch, O. Rybkin, F. Ebert, C. Finn, D. Jayaraman, and S. Levine, "Long-horizon visual planning with goal-conditioned hierarchical predictors," 2020.
[11] K. Pertsch, O. Rybkin, F. Ebert, C. Finn, D. Jayaraman, 和 S. Levine, "Long-horizon visual planning with goal-conditioned hierarchical predictors," 2020。


[12] R. Chitnis, D. Hadfield-Menell, A. Gupta, S. Srivastava, E. Groshev, C. Lin, and P. Abbeel, "Guided search for task and motion plans using learned heuristics," in 2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016, pp. 447-454.
[12] R. Chitnis, D. Hadfield-Menell, A. Gupta, S. Srivastava, E. Groshev, C. Lin, and P. Abbeel, “使用学习启发式进行任务与运动规划的引导搜索，”发表于 2016 年 IEEE 国际机器人与自动化会议 (ICRA)。IEEE, 2016, 第 447-454 页。


[13] B. Kim, Z. Wang, L. P. Kaelbling, and T. Lozano-Pérez, "Learning to guide task and motion planning using score-space representation," The International Journal of Robotics Research, vol. 38, no. 7, pp. 793-812, 2019.
[13] B. Kim, Z. Wang, L. P. Kaelbling, and T. Lozano-Pérez, “使用分数空间表示学习引导任务与运动规划，”《国际机器人研究杂志》，第 38 卷，第 7 期，页 793-812，2019 年。


[14] Z. Wang, C. R. Garrett, L. P. Kaelbling, and T. Lozano-Pérez, "Active model learning and diverse action sampling for task and motion planning," in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 4107-4114.
[14] Z. Wang, C. R. Garrett, L. P. Kaelbling, and T. Lozano-Pérez, “用于任务与运动规划的主动模型学习与多样化动作采样，”发表于 2018 年 IEEE/RSJ 国际智能机器人与系统会议 (IROS)。IEEE, 2018, 第 4107-4114 页。


[15] D. Driess, J.-S. Ha, and M. Toussaint, "Deep visual reasoning: Learning to predict action sequences for task and motion planning from an initial scene image," in Robotics: Science and Systems 2020 (RSS 2020). RSS Foundation, 2020.
[15] D. Driess, J.-S. Ha, and M. Toussaint, “深度视觉推理：从初始场景图像学习预测任务与运动规划的动作序列，”发表于 2020 年机器人学：科学与系统会议 (RSS 2020)。RSS 基金会, 2020 年。


[16] R. S. Sutton, D. Precup, and S. Singh, "Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning," Artificial intelligence, vol. 112, no. 1-2, pp. 181-211, 1999.
[16] R. S. Sutton, D. Precup, and S. Singh, “介于 MDP 与半 MDP 之间：强化学习中时间抽象的框架，”《人工智能》，第 112 卷，第 1-2 期，页 181-211，1999 年。


[17] A. S. Vezhnevets, S. Osindero, T. Schaul, N. Heess, M. Jader-berg, D. Silver, and K. Kavukcuoglu, "Feudal networks for hierarchical reinforcement learning," in International Conference on Machine Learning. PMLR, 2017, pp. 3540-3549.
[17] A. S. Vezhnevets, S. Osindero, T. Schaul, N. Heess, M. Jader-berg, D. Silver, and K. Kavukcuoglu, “用于分层强化学习的封建网络，”发表于国际机器学习大会。PMLR, 2017, 第 3540-3549 页。


[18] A. Gupta, V. Kumar, C. Lynch, S. Levine, and K. Hausman, "Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning," in Conference on Robot Learning. PMLR, 2020, pp. 1025-1037.
[18] A. Gupta, V. Kumar, C. Lynch, S. Levine, and K. Hausman, “接力策略学习：通过模仿与强化学习解决长时域任务，”发表于机器人学习会议。PMLR, 2020, 第 1025-1037 页。


[19] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, "Language models are few-shot learners," in Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., 2020.
[19] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, “语言模型是少样本学习者，”发表于《神经信息处理系统进展 33：神经信息处理系统年度会议 2020，NeurIPS 2020》，2020 年 12 月 6-12 日，线上，H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, 和 H. Lin 主编，2020 年。


[20] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omer-nick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel, "Palm: Scaling language modeling with pathways," CoRR, vol. abs/2204.02311, 2022.
[20] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omer-nick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel, "Palm: Scaling language modeling with pathways," CoRR, vol. abs/2204.02311, 2022.


[21] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al., "Palm 2 technical report," arXiv preprint arXiv:2305.10403, 2023.
[21] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al., "Palm 2 technical report," arXiv preprint arXiv:2305.10403, 2023.


[22] R. Taylor, M. Kardas, G. Cucurull, T. Scialom, A. Hartshorn, E. Saravia, A. Poulton, V. Kerkez, and R. Stojnic, "Galac-tica: A large language model for science," CoRR, vol. abs/2211.09085, 2022.
[22] R. Taylor, M. Kardas, G. Cucurull, T. Scialom, A. Hartshorn, E. Saravia, A. Poulton, V. Kerkez, and R. Stojnic, "Galac-tica: A large language model for science," CoRR, vol. abs/2211.09085, 2022.


[23] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, "Llama: Open and efficient foundation language models," CoRR, 2023.
[23] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, "Llama: Open and efficient foundation language models," CoRR, 2023.


[24] J. Luketina, N. Nardelli, G. Farquhar, J. N. Foerster, J. Andreas, E. Grefenstette, S. Whiteson, and T. Rocktäschel, "A survey of reinforcement learning informed by natural language," in IJCAI, 2019.
[24] J. Luketina, N. Nardelli, G. Farquhar, J. N. Foerster, J. Andreas, E. Grefenstette, S. Whiteson, and T. Rocktäschel, "A survey of reinforcement learning informed by natural language," in IJCAI, 2019.


[25] M. Ding, Y. Xu, Z. Chen, D. D. Cox, P. Luo, J. B. Tenenbaum, and C. Gan, "Embodied concept learner: Self-supervised learning of concepts and mapping through instruction following," in Conference on Robot Learning. PMLR, 2023, pp. 1743-1754.
[25] M. Ding, Y. Xu, Z. Chen, D. D. Cox, P. Luo, J. B. Tenenbaum, and C. Gan, "Embodied concept learner: Self-supervised learning of concepts and mapping through instruction following," in Conference on Robot Learning. PMLR, 2023, pp. 1743-1754.


[26] L. Shao, T. Migimatsu, Q. Zhang, K. Yang, and J. Bohg, "Concept2Robot: Learning manipulation concepts from instructions and human demonstrations," in Proceedings of Robotics: Science and Systems (RSS), 2020.
[26] L. Shao, T. Migimatsu, Q. Zhang, K. Yang, and J. Bohg, "Concept2Robot: Learning manipulation concepts from instructions and human demonstrations," in Proceedings of Robotics: Science and Systems (RSS), 2020.


[27] S. Stepputtis, J. Campbell, M. Phielipp, S. Lee, C. Baral, and H. Ben Amor, "Language-conditioned imitation learning for robot manipulation tasks," Advances in Neural Information Processing Systems, vol. 33, pp. 13139-13150, 2020.
[27] S. Stepputtis, J. Campbell, M. Phielipp, S. Lee, C. Baral, and H. Ben Amor, "Language-conditioned imitation learning for robot manipulation tasks," Advances in Neural Information Processing Systems, vol. 33, pp. 13139-13150, 2020.


[28] S. Nair, E. Mitchell, K. Chen, S. Savarese, C. Finn, et al., "Learning language-conditioned robot behavior from offline data and crowd-sourced annotation," in Conference on Robot Learning. PMLR, 2022, pp. 1303-1315.
[28] S. Nair, E. Mitchell, K. Chen, S. Savarese, C. Finn, et al., "Learning language-conditioned robot behavior from offline data and crowd-sourced annotation," in Conference on Robot Learning. PMLR, 2022, pp. 1303-1315.


[29] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard, "CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks," IEEE Robotics and Automation Letters, 2022.
[29] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard, "CALVIN：用于语言条件策略学习的长时程机器人操作任务基准," IEEE Robotics and Automation Letters, 2022.


[30] O. Mees, L. Hermann, and W. Burgard, "What matters in language conditioned robotic imitation learning over unstructured data," IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 11205-11212, 2022.
[30] O. Mees, L. Hermann, and W. Burgard, "在非结构化数据上进行语言条件机器人模仿学习时什么最重要," IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 11205-11212, 2022.


[31] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, "BC-Z: Zero-shot task generalization with robotic imitation learning," in Conference on Robot Learning (CoRL), 2021, pp. 991-1002.
[31] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, "BC-Z：通过机器人模仿学习实现零样本任务泛化," Conference on Robot Learning (CoRL), 2021, pp. 991-1002.


[32] M. Shridhar, L. Manuelli, and D. Fox, "Perceiver-actor: A multi-task transformer for robotic manipulation," Conference on Robot Learning (CoRL), 2022.
[32] M. Shridhar, L. Manuelli, and D. Fox, "Perceiver-actor：用于机器人操作的多任务变换器," Conference on Robot Learning (CoRL), 2022.


[33] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al., "RT-1: Robotics transformer for real-world control at scale," Robotics: Science and Systems (RSS), 2023.
[33] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al., "RT-1：面向规模化现实控制的机器人变换器," Robotics: Science and Systems (RSS), 2023.


[34] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, et al., "Do as i can, not as i say: Grounding language in robotic affordances," Conference on Robot Learning (CoRL), 2022.
[34] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, et al., "按我所能，而非我所言：将语言落地于机器人可供性," Conference on Robot Learning (CoRL), 2022.


[35] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng, "Code as policies: Language model programs for embodied control," arXiv preprint arXiv:2209.07753, 2022.
[35] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng, "将代码视为策略：用于具身控制的语言模型程序," arXiv preprint arXiv:2209.07753, 2022.


[36] S. Vemprala, R. Bonatti, A. Bucker, and A. Kapoor, "ChatGPT for robotics: Design principles and model abilities," Microsoft Auton. Syst. Robot. Res, vol. 2, p. 20, 2023.
[36] S. Vemprala, R. Bonatti, A. Bucker, and A. Kapoor, "ChatGPT 用于机器人：设计原则与模型能力," Microsoft Auton. Syst. Robot. Res, vol. 2, p. 20, 2023.


[37] K. Lin, C. Agia, T. Migimatsu, M. Pavone, and J. Bohg, "Text2motion: From natural language instructions to feasible plans," arXiv preprint arXiv:2303.12153, 2023.
[37] K. Lin, C. Agia, T. Migimatsu, M. Pavone, and J. Bohg, "Text2motion：从自然语言指令到可行计划," arXiv preprint arXiv:2303.12153, 2023.


[38] A. Bucker, L. Figueredo, S. Haddadin, A. Kapoor, S. Ma, and R. Bonatti, "Latte: Language trajectory transformer," arXiv preprint arXiv:2208.02918, 2022.
[38] A. Bucker, L. Figueredo, S. Haddadin, A. Kapoor, S. Ma, and R. Bonatti, "Latte：语言轨迹变换器," arXiv preprint arXiv:2208.02918, 2022.


[39] F. Hill, S. Mokra, N. Wong, and T. Harley, "Human instruction-following with deep reinforcement learning via transfer-learning from text," arXiv preprint arXiv:2005.09382, 2020.
[39] F. Hill, S. Mokra, N. Wong, and T. Harley, "通过从文本迁移学习实现的深度强化学习进行人类指令跟随," arXiv preprint arXiv:2005.09382, 2020.


[40] C. Lynch and P. Sermanet, "Grounding language in play," Robotics: Science and Systems (RSS), 2021.
[40] C. Lynch and P. Sermanet, "在游戏中将语言落地," Robotics: Science and Systems (RSS), 2021.


[41] Y. Jiang, A. Gupta, Z. Zhang, G. Wang, Y. Dou, Y. Chen, L. Fei-Fei, A. Anandkumar, Y. Zhu, and L. Fan, "VIMA: General robot manipulation with multimodal prompts," International Conference on Machine Learning (ICML), 2023.
[41] Y. Jiang, A. Gupta, Z. Zhang, G. Wang, Y. Dou, Y. Chen, L. Fei-Fei, A. Anandkumar, Y. Zhu, and L. Fan, "VIMA：基于多模态提示的一般机器人操作," International Conference on Machine Learning (ICML), 2023.


[42] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, "Voxposer: Composable 3d value maps for robotic manipulation with language models," arXiv preprint arXiv:2307.05973, 2023.
[42] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, "Voxposer：用于结合语言模型的机器人操作的可组合三维价值图," arXiv preprint arXiv:2307.05973, 2023.


[43] M. Shridhar, L. Manuelli, and D. Fox, "Cliport: What and where pathways for robotic manipulation," in Conference on Robot Learning. PMLR, 2022, pp. 894-906.
[43] M. Shridhar, L. Manuelli, and D. Fox, "Cliport：用于机器人操作的何物与何处路径," in Conference on Robot Learning. PMLR, 2022, pp. 894-906.


[44] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, "R3m: A universal visual representation for robot manipulation," arXiv preprint arXiv:2203.12601, 2022.
[44] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, "R3M：用于机器人操作的通用视觉表征," arXiv preprint arXiv:2203.12601, 2022.


[45] Y. Mu, S. Yao, M. Ding, P. Luo, and C. Gan, "Ec2: Emergent communication for embodied control," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6704-6714.
[45] Y. Mu, S. Yao, M. Ding, P. Luo, and C. Gan, "Ec2: 面向具身控制的新兴通信," 收录于 IEEE/CVF 计算机视觉与模式识别会议论文集, 2023, 页 6704-6714.


[46] A. Stone, T. Xiao, Y. Lu, K. Gopalakrishnan, K.-H. Lee, Q. Vuong, P. Wohlhart, B. Zitkovich, F. Xia, C. Finn, et al., "Open-world object manipulation using pre-trained vision-language models," arXiv preprint arXiv:2303.00905, 2023.
[46] A. Stone, T. Xiao, Y. Lu, K. Gopalakrishnan, K.-H. Lee, Q. Vuong, P. Wohlhart, B. Zitkovich, F. Xia, C. Finn, et al., "使用预训练视觉-语言模型进行开放世界物体操作," arXiv 预印本 arXiv:2303.00905, 2023.


[47] Y. Mu, Q. Zhang, M. Hu, W. Wang, M. Ding, J. Jin, B. Wang, J. Dai, Y. Qiao, and P. Luo, "Embodiedgpt: Vision-language pre-training via embodied chain of thought," arXiv preprint arXiv:2305.15021, 2023.
[47] Y. Mu, Q. Zhang, M. Hu, W. Wang, M. Ding, J. Jin, B. Wang, J. Dai, Y. Qiao, and P. Luo, "Embodiedgpt: 通过具身思维链进行视觉-语言预训练," arXiv 预印本 arXiv:2305.15021, 2023.


[48] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al., "RT-2: Vision-language-action models transfer web knowledge to robotic control," arXiv preprint arXiv:2307.15818, 2023.
[48] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al., "RT-2: 视觉-语言-动作模型将网络知识转移至机器人控制," arXiv 预印本 arXiv:2307.15818, 2023.


[49] S. Schaal, "Dynamic movement primitives-a framework for motor control in humans and humanoid robotics," in Adaptive motion of animals and machines. Springer, 2006, pp. 261- 280.
[49] S. Schaal, "动态运动基元——人类与类人机器人运动控制的框架," 收录于动物与机器的自适应运动. Springer, 2006, 页 261-280.


[50] A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal, "Dynamical movement primitives: learning attractor models for motor behaviors," Neural computation, vol. 25, no. 2, pp. 328-373, 2013.
[50] A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal, "动力学运动基元：为运动行为学习吸引子模型," Neural computation, 卷 25, 期 2, 页 328-373, 2013.


[51] E. Coumans and Y. Bai, "Pybullet, a python module for physics simulation for games, robotics and machine learning," http://pybullet.org.2016-2021.
[51] E. Coumans and Y. Bai, "Pybullet，一个用于游戏、机器人与机器学习的物理仿真 Python 模块," http://pybullet.org.2016-2021.


[52] Kinova, "Kinova-movo." [Online]. Available: https://docs.kinovarobotics.com/kinova-movo/Concepts/ c_movo_hardware_overview.html
[52] Kinova, "Kinova-movo." [在线]. 可用: https://docs.kinovarobotics.com/kinova-movo/Concepts/ c_movo_hardware_overview.html


[53] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., "Segment anything," arXiv preprint arXiv:2304.02643, 2023.
[53] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., "Segment anything," arXiv 预印本 arXiv:2304.02643, 2023.


[54] Y. Wang, P. Praveena, D. Rakita, and M. Gleicher, "Rangedik: An optimization-based robot motion generation method for ranged-goal tasks," arXiv preprint arXiv:2302.13935, 2023.
[54] Y. Wang, P. Praveena, D. Rakita, and M. Gleicher, "Rangedik: 一种面向范围目标任务的基于优化的机器人运动生成方法," arXiv 预印本 arXiv:2302.13935, 2023.