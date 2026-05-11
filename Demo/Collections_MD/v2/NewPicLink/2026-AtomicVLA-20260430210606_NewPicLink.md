# AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots
# AtomicVLA：释放机器人原子技能学习的潜力


Likui Zhang ${}^{1}$ Tao Tang ${}^{1}$ Zhihao Zhan ${}^{1}$ Xiuwei Chen ${}^{1}$ Zisheng Chen ${}^{1}$ Jianhua Han ${}^{3}$ Jiangtong Zhu ${}^{3}$ Pei ${\mathrm{{Xu}}}^{3}$ Hang ${\mathrm{{Xu}}}^{3}$ Hefeng ${\mathrm{{Wu}}}^{1}$ Liang ${\mathrm{{Lin}}}^{1 \dagger  }$ Xiaodan Liang ${}^{1,2 \dagger  } \; {}^{1}$ Sun Yat-sen University ${}^{2}$ Peng Cheng Laboratory ${}^{3}$ Yinwang Intelligent Technology Co. Ltd. zhanglk9@mail2.sysu.edu.cn
Likui Zhang ${}^{1}$ Tao Tang ${}^{1}$ Zhihao Zhan ${}^{1}$ Xiuwei Chen ${}^{1}$ Zisheng Chen ${}^{1}$ Jianhua Han ${}^{3}$ Jiangtong Zhu ${}^{3}$ Pei ${\mathrm{{Xu}}}^{3}$ Hang ${\mathrm{{Xu}}}^{3}$ Hefeng ${\mathrm{{Wu}}}^{1}$ Liang ${\mathrm{{Lin}}}^{1 \dagger  }$ Xiaodan Liang ${}^{1,2 \dagger  } \; {}^{1}$ 中山大学 ${}^{2}$ 鹏城实验室 ${}^{3}$ 银王智能科技有限公司 zhanglk9@mail2.sysu.edu.cn


## Abstract
## 摘要


Recent advances in Visual-Language-Action (VLA) models have shown promising potential for robotic manipulation tasks. However, real-world robotic tasks often involve long-horizon, multi-step problem-solving and require generalization for continual skill acquisition, extending beyond single actions or skills. These challenges present significant barriers for existing VLA models, which use monolithic action decoders trained on aggregated data, resulting in poor scalability. To address these challenges, we propose Atom-icVLA, a unified planning-and-execution framework that jointly generates task-level plans, atomic skill abstractions, and fine-grained actions. AtomicVLA constructs a scalable atomic skill library through a Skill-Guided Mixture-of-Experts (SG-MoE), where each expert specializes in mastering generic yet precise atomic skills. Furthermore, we introduce a flexible routing encoder that automatically assigns dedicated atomic experts to new skills, enabling continual learning. We validate our approach through extensive experiments. In simulation, AtomicVLA outperforms ${\pi }_{0}$ by 2.4% on LIBERO,10% on LIBERO-LONG,and outperforms ${\pi }_{0}$ and ${\pi }_{0.5}$ by 0.22 and 0.25 in average task length on CALVIN. Additionally, our AtomicVLA consistently surpasses baselines by 18.3% and 21% in real-world long-horizon tasks and continual learning. These results highlight the effectiveness of atomic skill abstraction and dynamic expert composition for long-horizon and lifelong robotic tasks. The project page is here.
视觉-语言-动作（VLA）模型的最新进展在机器人操作任务中展现出巨大潜力。然而，现实世界的机器人任务通常涉及长程、多步骤的问题求解，并需要具备持续获取技能的泛化能力，而不仅仅是单一动作或技能。这些挑战为现有 VLA 模型带来了显著障碍，因为它们使用在聚合数据上训练的单一动作解码器，导致可扩展性较差。为解决这些挑战，我们提出了 AtomicVLA，这是一个统一的规划与执行框架，能够联合生成任务级规划、原子技能抽象和细粒度动作。AtomicVLA 通过“技能引导专家混合模型”（SG-MoE）构建了一个可扩展的原子技能库，其中每个专家都专注于掌握通用且精确的原子技能。此外，我们引入了一种灵活的路由编码器，可自动为新技能分配专门的原子专家，从而实现持续学习。我们通过大量实验验证了该方法。在仿真中，AtomicVLA 在 LIBERO 上比 ${\pi }_{0}$ 提升了 2.4%，在 LIBERO-LONG 上提升了 10%，在 CALVIN 的平均任务长度上比 ${\pi }_{0}$ 和 ${\pi }_{0.5}$ 分别提升了 0.22 和 0.25。此外，我们的 AtomicVLA 在现实世界的长程任务和持续学习中，性能始终优于基线模型 18.3% 和 21%。这些结果凸显了原子技能抽象和动态专家组合在长程及终身机器人任务中的有效性。项目页面见此处。


## 1. Introduction
## 1. 引言


Building on powerful Vision-Language Models [2, 6, 9, 21, 38, 50], Vision-Language-Action (VLA) models [3, 4, 22, 30] unify visual perception, language understanding, and action generation into a single framework, achieving significant advances in robotic manipulation tasks. Despite this progress, current VLA models still face challenges in real-world deployments for complex long-horizon tasks and the continual acquisition of new skills.
基于强大的视觉-语言模型 [2, 6, 9, 21, 38, 50]，视觉-语言-动作（VLA）模型 [3, 4, 22, 30] 将视觉感知、语言理解和动作生成统一到一个框架中，在机器人操作任务中取得了显著进展。尽管取得了这些进步，但当前的 VLA 模型在处理复杂的长程任务和持续获取新技能的实际部署中仍面临挑战。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_85e9b1.jpg"/>



Figure 1. Overview of AtomicVLA. Unlike previous VLA models with a single action head, which suffer from limited scalabil-ity and severe interference among mixed skills, AtomicVLA employs a SG-MoE architecture to build a scalable skill expert library. By unifying task planning and action execution within this framework, it achieves strong performance on long-horizon and continual learning tasks in both simulation and real-world settings.
图 1. AtomicVLA 概览。与以往具有单一动作头、可扩展性有限且混合技能间干扰严重的 VLA 模型不同，AtomicVLA 采用 SG-MoE 架构来构建可扩展的技能专家库。通过在该框架内统一任务规划与动作执行，它在仿真和现实环境的长程及持续学习任务中均实现了卓越性能。


To overcome these challenges, a robotic model must support both high-level reasoning and fine-grained action generation, while enabling scalable continual learning. To support high-level reasoning and task planning, some existing approaches employ a two-stage architecture [1, 13, 20, 35, 41, 46], where a pretrained vision-language model (VLM) serves as a high-level planner to generate subtask instructions, while a separate VLA-based controller translates these instructions into executable actions. However, recent studies [26, 59, 60] suggest that modular decoupling leads to a lack of mutual awareness between the planner and controller, causing suboptimal task coordination. Moreover, in real-world applications, this can result in the generation of outdated or irrelevant instructions due to system delays. In addition, most existing VLA models rely on a single action-decoding module, limiting their scalabil-ity. Incrementally learning new skills requires fine-tuning existing models, which demands substantial computational resources and large datasets. Given the current scarcity of robot data, fully leveraging well-pretrained VLA model weights is essential during the scaling process. Moreover, when learning new skills incrementally, these models often interfere with previously acquired skills, leading to catastrophic forgetting and thereby hindering the lifelong learning capabilities.
为了克服这些挑战，机器人模型必须同时支持高层推理和细粒度动作生成，并实现可扩展的持续学习。为支持高层推理和任务规划，一些现有方法采用了两阶段架构 [1, 13, 20, 35, 41, 46]，其中预训练的视觉-语言模型（VLM）作为高层规划器生成子任务指令，而独立的基于 VLA 的控制器将这些指令转化为可执行动作。然而，近期研究 [26, 59, 60] 表明，模块解耦会导致规划器与控制器之间缺乏相互感知，从而导致任务协调欠佳。此外，在实际应用中，由于系统延迟，这可能导致生成过时或不相关的指令。此外，大多数现有 VLA 模型依赖于单一的动作解码模块，限制了其可扩展性。增量学习新技能需要对现有模型进行微调，这需要大量的计算资源和大规模数据集。鉴于当前机器人数据的稀缺性，在扩展过程中充分利用预训练良好的 VLA 模型权重至关重要。而且，在增量学习新技能时，这些模型往往会干扰先前习得的技能，导致灾难性遗忘，从而阻碍了终身学习能力。


---



${}^{ \dagger  }$ Co-corresponding author
${}^{ \dagger  }$ 共同通讯作者


---



To this end, we propose AtomicVLA, as illustrated in Fig. 1, an end-to-end framework that unifies task planning and action execution by adaptively generating either natural language instructions or latent actions. AtomicVLA first infers the current execution state from the input observations and dynamically activates either its thinking module or its acting module. At task initialization or during transitions between sub-skills, the model triggers thinking to produce a task chain, create a task chain plan based on the current state, and outputs atomic skill abstractions. In the acting execution phase, it dynamically selects the corresponding skill-specific expert based on the most recent skill abstraction to generate precise robot control signals. Furthermore, to enable AtomicVLA with continual learning capability, we introduce a Skill-Guided Mixture-of-Experts (SG-MoE) architecture that constructs a scalable library of atomic skills. This library comprises a shared expert and multiple dedicated skill experts, each focusing on mastering a specific atomic skill. Through a well-designed skill encoding mechanism and an extensible routing encoder, each atomic skill abstraction is mapped to a fixed embedding vector, allowing the routing module to rapidly adapt to new skills even as the skill library grows. When a new skill is introduced, only the corresponding expert and associated routing parameters need to be trained, leaving existing experts unchanged. This effectively prevents catastrophic forgetting, ensuring efficient and stable lifelong skill growth.
为此，我们提出了 AtomicVLA（如图 1 所示），这是一个端到端框架，通过自适应生成自然语言指令或潜在动作，实现了任务规划与动作执行的统一。AtomicVLA 首先根据输入观测推断当前执行状态，并动态激活其“思考”模块或“执行”模块。在任务初始化或子技能转换期间，模型触发思考过程以生成任务链，根据当前状态制定任务链计划，并输出原子技能抽象。在执行阶段，它根据最新的技能抽象动态选择相应的技能专家，以生成精确的机器人控制信号。此外，为使 AtomicVLA 具备持续学习能力，我们引入了技能引导专家混合（SG-MoE）架构，构建了一个可扩展的原子技能库。该库包含一个共享专家和多个专用技能专家，每个专家专注于掌握特定的原子技能。通过精心设计的技能编码机制和可扩展的路由编码器，每个原子技能抽象都被映射为一个固定的嵌入向量，使路由模块即使在技能库不断增长的情况下也能快速适应新技能。当引入新技能时，只需训练相应的专家和关联的路由参数，而无需改动现有专家。这有效防止了灾难性遗忘，确保了高效且稳定的终身技能增长。


We conducted extensive experiments to validate the effectiveness of AtomicVLA both in simulation platforms and real-world robots. In the LIBERO [28] benchmark, Atom-icVLA achieved an average performance improvement of 2.4% over baseline models, with a notable 10% improvement on the LIBERO-LONG. On the CALVIN [33] benchmark, specifically on the task ABC-D training set, our method increased the average successful execution length by 0.22 and 0.25 . Furthermore, we performed long-horizon task execution and continual learning experiments on a real-world Franka robot, where we observed performance improvements of 18.3% and 21%, respectively. These results further validate the potential of AtomicVLA's proposed atomic skill dynamic combination mechanism in supporting long-term task completion and lifelong skill accumulation. Overall, our contributions are as follows:
我们进行了广泛的实验，以验证AtomicVLA在模拟平台和现实世界机器人中的有效性。在LIBERO [28]基准测试中，AtomicVLA相较于基线模型平均性能提升了2.4%，在LIBERO - LONG上显著提升了10%。在CALVIN [33]基准测试中，特别是在任务ABC - D训练集上，我们的方法使平均成功执行长度分别增加了0.22和0.25。此外，我们在现实世界的Franka机器人上进行了长视野任务执行和持续学习实验，观察到性能分别提升了18.3%和21%。这些结果进一步验证了AtomicVLA提出的原子技能动态组合机制在支持长期任务完成和终身技能积累方面的潜力。总体而言，我们的贡献如下：


- We introduce AtomicVLA, an end-to-end framework that unifies task planning and action execution for long-horizon tasks and continual skill expansion.
- 我们引入了 AtomicVLA，这是一个端到端框架，统一了长程任务的任务规划与动作执行，并支持持续的技能扩展。


- We propose a Skill-Guided Mixture-of-Experts (SG-MoE) architecture and a scalable skill router for building a library of atomic skills.
- 我们提出了技能引导专家混合（SG-MoE）架构和一个可扩展的技能路由器，用于构建原子技能库。


- We validate the effectiveness of AtomicVLA through extensive experiments conducted in both simulated environments and real-world robots.
- 我们通过在仿真环境和真实机器人上进行的广泛实验，验证了 AtomicVLA 的有效性。


## 2. Related Work
## 2. 相关工作


### 2.1. Vision-Language-Action Models
### 2.1. 视觉-语言-动作模型


Vision-Language Action Models (VLAs) have emerged as a dominant paradigm in general-purpose robotic learning by leveraging the rich semantic priors and strong cross-modal generalization of large-scale Vision-Language Models (VLMs) pretrained on internet-scale data. Recent works $\left\lbrack  {3,4,{17},{22},{24},{36},{55},{66}}\right\rbrack$ fine-tune VLMs $\left\lbrack  {2,9,{21},{50}}\right\rbrack$ on diverse robotic datasets to directly map visual and linguistic inputs to motor actions, demonstrating impressive generalization to novel environments and tasks.
视觉-语言-动作模型（VLAs）利用大规模视觉-语言模型（VLMs）在互联网规模数据上预训练所获得的丰富语义先验和强大的跨模态泛化能力，已成为通用机器人学习的主流范式。近期研究 $\left\lbrack  {3,4,{17},{22},{24},{36},{55},{66}}\right\rbrack$ 通过在多样化的机器人数据集上微调 VLMs $\left\lbrack  {2,9,{21},{50}}\right\rbrack$，将视觉和语言输入直接映射为电机动作，在面对新环境和新任务时展现出了令人印象深刻的泛化能力。


However, constrained to some extent by the VLM's inherent hierarchical planning capability, most current VLAs exhibit limitations in structured task decomposition and long-horizon task planning. Several approaches introduce external high-level planners [10, 14, 41, 46, 62, 67] that decompose long-horizon tasks into subgoals, which are then executed by a separate low-level policy. However, modular approaches often fail to unify action with vision and language in a shared latent space, resulting in misaligned decisions that compound loss. To address this problem, recent work $\left\lbrack  {5,{11},{26},{29},{59},{60}}\right\rbrack$ propose integrated frameworks that jointly perform hierarchical reasoning and action generation within a unified model. Our work aligns with this direction: we adopt a Think-Act unified architecture, where a VLM simultaneously performs high-level task planning and atomic action abstraction, thereby directly guiding a specialized action expert to produce executable, temporally coherent action sequences.
然而，受限于 VLM 本身的层次化规划能力，目前大多数 VLA 在结构化任务分解和长程任务规划方面仍存在局限。一些方法引入了外部高层规划器 [10, 14, 41, 46, 62, 67]，将长程任务分解为子目标，再由独立的底层策略执行。但模块化方法往往难以在共享潜在空间中统一动作与视觉、语言，导致决策失准并累积误差。为解决这一问题，近期研究 $\left\lbrack  {5,{11},{26},{29},{59},{60}}\right\rbrack$ 提出了集成框架，在统一模型内联合执行层次化推理和动作生成。我们的工作与这一方向一致：我们采用“思考-执行”统一架构，使 VLM 同时进行高层任务规划和原子动作抽象，从而直接引导专门的动作专家生成可执行且时间连贯的动作序列。


### 2.2. Multimodal Mixture-of-Experts
### 2.2. 多模态专家混合


The Sparse Mixture-of-Experts (MoE) architecture has become a mainstream approach for scaling large language models (LLMs). By replacing the standard feed-forward layers with expert modules [8, 19], MoE improves task specialization and representation capability through conditional computation, while maintaining inference efficiency. In the field of autonomous driving, models such as [57, 61] design specialized MoE architectures tailored to multi-view observations and action skills, improving both trajectory prediction accuracy and inference efficiency. Similarly, in robotics, some works [15, 39, 51, 58, 64, 68] employ MoE to tackle task heterogeneity and long-tailed data distributions. While these approaches demonstrate the utility of MoE for representation learning, they largely treat experts as interchangeable components within fixed architectural slots, without explicitly modeling structured, composable behaviors. In contrast, we reinterpret the MoE paradigm through the lens of skill modularity: we construct a dynamically scalable atomic skill library, where each expert corresponds to a semantically meaningful, reusable action primitive. Integrated with a pre-trained VLM that encodes atomic action abstractions, our approach enables a universal VLA model capable of both fine-grained skill decomposition and coherent long-horizon task composition.
稀疏混合专家（MoE）架构已成为扩展大型语言模型（LLM）的主流方法。通过用专家模块 [8, 19] 替换标准的各种前馈层，MoE 在保持推理效率的同时，通过条件计算提升了任务专业化程度和表征能力。在自动驾驶领域，[57, 61] 等模型设计了针对多视图观测和动作技能的专用 MoE 架构，提高了轨迹预测精度和推理效率。同样，在机器人领域，一些研究 [15, 39, 51, 58, 64, 68] 采用 MoE 来处理任务异构性和长尾数据分布。尽管这些方法证明了 MoE 在表征学习中的效用，但它们大多将专家视为固定架构槽位中的可互换组件，而未明确建模结构化、可组合的行为。相比之下，我们从技能模块化的视角重新诠释了 MoE 范式：我们构建了一个动态可扩展的原子技能库，其中每个专家对应一个语义明确、可复用的动作基元。结合编码了原子动作抽象的预训练视觉语言模型（VLM），我们的方法实现了一个通用的 VLA 模型，能够同时进行细粒度的技能分解和连贯的长程任务组合。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_6f6f2e.jpg"/>



Figure 2. (a) AtomicVLA Pipline. AtomicVLA is a framework that unifies task planning and action execution. The VLM adaptively predicts atomic skill abstraction and latent action. Action Decoder in the SG-MoE architecture receives both the latent action and the newly inferred atomic skill abstraction, and generates fine grained motor actions. (b) Skill-Guided Mixture of Experts. SG-MoE includes a skill router, a shared expert, and multiple atomic-skill experts. The router selects the top skill expert based on the atomic skill, and the action token is processed by both the activated skill expert and the shared expert. (c) Continual Learning with Skill Expansion. New skills are added by training only the new expert and extending the router. (d) Task Planning Embodied Data Generation. High-quality embodied reasoning data are generated using principal-axis analysis with InternVideo2.5 [52] model.
图 2. (a) AtomicVLA 流水线。AtomicVLA 是一个统一任务规划与动作执行的框架。VLM 自适应地预测原子技能抽象和潜在动作。SG-MoE 架构中的动作解码器接收潜在动作和新推断出的原子技能抽象，并生成细粒度的电机动作。(b) 技能引导的混合专家。SG-MoE 包含一个技能路由、一个共享专家和多个原子技能专家。路由根据原子技能选择顶层技能专家，动作标记由激活的技能专家和共享专家共同处理。(c) 具备技能扩展的持续学习。通过仅训练新专家并扩展路由来添加新技能。(d) 任务规划具身数据生成。利用 InternVideo2.5 [52] 模型通过主轴分析生成高质量的具身推理数据。


### 2.3. Continual Learning with Skill Abstractions
### 2.3. 基于技能抽象的持续学习


To adapt to new tasks that emerge in dynamic environments, continual learning has become essential for developing general-purpose intelligent agents. Prior studies $\left\lbrack  {7,{12},{32},{42},{43},{49}}\right\rbrack$ have leveraged unsupervised learning and hierarchical imitation learning to enable autonomous skill discovery from continuous data, which allows an agent to expand its skill set over time. Furthermore, to learn from streaming data without suffering from catastrophic forgetting, several approaches [23, 34, 40, 63] introduce latent action representations that abstract different skills and preserve previously acquired capabilities without relying on experience replay. Current VLA Models primarily focus on learning generalizable skills from broad pretraining, while dedicated investigations into continual learning remain limited. Although many VLAs [53, 54] have explored various motion decoding methods, such as diffusion models [30], flow matching [27], and discrete encoding [34, 48], they all use a single decoder. Their core focus is on the model's accuracy on the current task rather than its scalability. We construct an expandable library of skill experts by using atomic units of robotic behavior together with a specialized routing module, which enhances the scal-ability of such models in skill acquisition.
为了适应动态环境中出现的新任务，持续学习对于开发通用智能体至关重要。先前的研究 $\left\lbrack  {7,{12},{32},{42},{43},{49}}\right\rbrack$ 利用无监督学习和分层模仿学习，实现了从连续数据中自主发现技能，使智能体能够随时间扩展其技能集。此外，为了在不依赖经验回放的情况下从流式数据中学习且避免灾难性遗忘，一些方法 [23, 34, 40, 63] 引入了抽象不同技能的潜在动作表征，并保留了先前习得的能力。当前的 VLA 模型主要侧重于从广泛的预训练中学习可泛化技能，而针对持续学习的专门研究仍然有限。尽管许多 VLA [53, 54] 探索了各种运动解码方法，如扩散模型 [30]、流匹配 [27] 和离散编码 [34, 48]，但它们都使用单一解码器。它们的核心关注点是模型在当前任务上的准确性，而非其可扩展性。我们通过使用机器人行为的原子单元以及专门的路由模块，构建了一个可扩展的技能专家库，从而增强了此类模型在技能获取方面的可扩展性。


## 3. Method
## 3. 方法


### 3.1. Overview
### 3.1. 概述


As illustrated in Fig. 2, AtomicVLA integrates the thinking modality for task planning and the acting modality for action execution within a unified framework (Sec. 3.2). Building upon this architecture, we develop a skill-guided library of atomic action experts (Sec. 3.3) based on pi0 and introduce an extensible skill router that facilitates continual learning of new skills (Sec. 3.4) in real-world environments. To further ensure the generation of high-quality task planning data, we introduce an embodiment data generation pipeline (Sec. 3.5) grounded in principal axis analysis, which provides structured and consistent data to support effective task planning and execution.
如图 2 所示，AtomicVLA 在统一框架内集成了用于任务规划的思维模态和用于动作执行的行动模态（第 3.2 节）。基于此架构，我们开发了一个基于 pi0 的原子动作专家技能引导库（第 3.3 节），并引入了一个可扩展的技能路由，以促进在真实环境中持续学习新技能（第 3.4 节）。为了进一步确保生成高质量的任务规划数据，我们引入了一个基于主轴分析的具身数据生成流水线（第 3.5 节），该流水线提供了结构化且一致的数据，以支持有效的任务规划与执行。


Algorithm 1 Inference Pipeline of AtomicVLA
算法 1 AtomicVLA 推理流水线


---



Require: VLA model ${\pi }_{\theta }$ ,language instruction $\ell$
要求：VLA 模型 ${\pi }_{\theta }$，语言指令 $\ell$


&nbsp;&nbsp;&nbsp;&nbsp;$t \leftarrow  0,{O}_{\mathrm{t}}^{1 : n} \leftarrow$ initial image,Atomic $\leftarrow$ none
&nbsp;&nbsp;&nbsp;&nbsp;$t \leftarrow  0,{O}_{\mathrm{t}}^{1 : n} \leftarrow$ 初始图像，原子 $\leftarrow$ 无


&nbsp;&nbsp;&nbsp;&nbsp;while "task not done" do
&nbsp;&nbsp;&nbsp;&nbsp;当“任务未完成”时


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$M \sim  {\pi }_{\theta } \cdot  \operatorname{PREDICT}\left( {\cdot  \mid  {O}_{t}^{1 : n},\ell }\right)$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if $M = \left\lbrack  \text{ think }\right\rbrack$ then
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果 $M = \left\lbrack  \text{ think }\right\rbrack$ 则


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\left\lbrack  {{C}_{0 - k},{C}_{t},\sigma }\right\rbrack   \sim  {\pi }_{\theta }$ . THINKING $\left( {\cdot  \mid  {O}_{t}^{1 : n},\ell }\right)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\left\lbrack  {{C}_{0 - k},{C}_{t},\sigma }\right\rbrack   \sim  {\pi }_{\theta }$ . 思考 $\left( {\cdot  \mid  {O}_{t}^{1 : n},\ell }\right)$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Atomic $\leftarrow  \sigma$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;原子 $\leftarrow  \sigma$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else if $M = \left\lbrack  \text{ act }\right\rbrack$ then
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;否则如果 $M = \left\lbrack  \text{ act }\right\rbrack$ 则


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${w}_{k} \sim  \operatorname{Router}\left( {\operatorname{embeded}\left( \text{ Atomic }\right) }\right)$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${A}_{t} \sim  {\pi }_{\theta } \cdot  \operatorname{ACTING}\left( {\cdot  \mid  {O}_{t}^{1 : n},\ell ,{s}_{t},{w}_{k}}\right)$



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Execute ${A}_{t}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;执行 ${A}_{t}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;end if
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;结束如果


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$t \leftarrow  t + 1$



&nbsp;&nbsp;&nbsp;&nbsp;end while
&nbsp;&nbsp;&nbsp;&nbsp;结束循环


---



### 3.2. Unified Task Planning and Action Execution
### 3.2. 统一任务规划与动作执行


Problem formulation. The central problem addressed in this section is to design a robot policy that simultaneously possesses task planning (thinking) and action execution (acting) capabilities, and can autonomously decide its output modality based on the current states. Specifically, in thinking mode, the policy takes multiple cameras observations ${O}_{t}^{1 : n}$ and a language instruction $\ell$ as input and outputs a high-level task plan $\left\lbrack  {{C}_{0 - k},{C}_{t},\sigma }\right\rbrack$ in textual form. In contrast, in acting mode, the policy generates a concrete action command conditioned on the robot's proprioceptive state ${S}_{t}$ and the most recent planning output $\sigma$ .
问题表述。本节旨在设计一种机器人策略，使其同时具备任务规划（思考）和动作执行（行动）能力，并能根据当前状态自主决定输出模态。具体而言，在思考模式下，策略以多视角相机观测 ${O}_{t}^{1 : n}$ 和语言指令 $\ell$ 作为输入，输出文本形式的高层任务计划 $\left\lbrack  {{C}_{0 - k},{C}_{t},\sigma }\right\rbrack$。相反，在行动模式下，策略根据机器人的本体感知状态 ${S}_{t}$ 和最新的规划输出 $\sigma$ 生成具体的动作指令。


Adaptive thinking and acting. To enable seamless switching between the two output modalities, we introduce two special output tokens: [think] and [act]. As illustrated in Algorithm 1,given the current visual observations ${O}_{t}^{1 : n}$ and task instruction $\ell$ ,the model first predicts identifier either [think] or [act]. When the model outputs [think], it enters the thinking mode,in which it generates a task chain ${C}_{0 - k}$ that outlines the high-level plan, tracks the current execution progress ${C}_{t}$ ,and specifies the atomic skill abstraction $\sigma$ to be performed. Typically,this mode is activated only at key time steps, such as task initiation or during the transition between sub-skills. Conversely, when [act] is predicted, the model switches to acting mode, where it produces a low-level action chunk ${A}_{t}$ based on the atomic skill abstract $\sigma$ obtained in the most recent [think] step and the current proprioceptive state.
自适应思考与行动。为实现两种输出模态间的无缝切换，我们引入了两个特殊输出标记：[think] 和 [act]。如算法 1 所示，给定当前视觉观测 ${O}_{t}^{1 : n}$ 和任务指令 $\ell$，模型首先预测标识符 [think] 或 [act]。当模型输出 [think] 时，进入思考模式，生成概述高层计划的任务链 ${C}_{0 - k}$，跟踪当前执行进度 ${C}_{t}$，并指定待执行的原子技能抽象 $\sigma$。通常，该模式仅在关键时间步（如任务启动或子技能转换期间）激活。反之，当预测为 [act] 时，模型切换至行动模式，根据最近一次 [think] 步骤获得的原子技能抽象 $\sigma$ 及当前本体感知状态，生成低层动作块 ${A}_{t}$。


### 3.3. Skill-guided Mixture of Experts Architecture
### 3.3. 技能引导的专家混合架构


Atomic skill abstract embedding. To enhance the representational distinctiveness among atomic skills, we adopt an encoding strategy inspired by noise scheduling in diffusion-based denoising models. Specifically, each atomic skill abstract is mapped to a scalar noise level $\sigma  \in  \left\lbrack  {0,{100}}\right\rbrack$ ,which is then embedded into a high-dimensional vector. This continuous and structured embedding space facilitates semantic separation across skills and enables robust routing to the corresponding skill-specific experts.
原子技能抽象嵌入。为增强原子技能间的表征区分度，我们采用了一种受扩散模型中噪声调度启发的编码策略。具体地，每个原子技能抽象被映射为一个标量噪声水平 $\sigma  \in  \left\lbrack  {0,{100}}\right\rbrack$，随后被嵌入到高维向量中。这种连续且结构化的嵌入空间促进了技能间的语义分离，并实现了向相应技能专家的高效路由。


$$
{Z}_{\sigma } = E\left( {\operatorname{norm}\left( {\log \left( \sigma \right) }\right) }\right) , \tag{1}
$$



where $\sigma$ denotes the assigned noise level for the skill,and $E\left( \cdot \right)$ is a embedding function that maps the normalized scalar to a high-dimensional embedding vector ${Z}_{\sigma }$ .
其中 $\sigma$ 表示分配给该技能的噪声水平，$E\left( \cdot \right)$ 是将归一化标量映射为高维嵌入向量 ${Z}_{\sigma }$ 的嵌入函数。


Skill-Guided dynamic routing. We build upon the ${\pi }_{0}$ vision-language-action (VLA) foundation model, a generalist robotic policy pretrained on large-scale multimodal data, and extend it with an atomic action abstraction-guided Mixture-of-Experts (MoE) architecture to construct a scalable atomic skill library. As illustrated in Fig. 2(b), our skill library consists of three key components: (1) a skill router, (2) a shared expert that maintains the pre-trained action generation capabilities of ${\pi }_{0}$ ,and (3) multiple atomic skill experts, each specialized in executing a distinct atomic skill.
技能引导的动态路由。我们基于 ${\pi }_{0}$ 视觉-语言-动作 (VLA) 基础模型（一种在大规模多模态数据上预训练的通用机器人策略），通过引入原子动作抽象引导的专家混合 (MoE) 架构进行扩展，构建了一个可扩展的原子技能库。如图 2(b) 所示，我们的技能库包含三个关键组件：(1) 技能路由器，(2) 保持 ${\pi }_{0}$ 预训练动作生成能力的共享专家，以及 (3) 多个原子技能专家，每个专家专门负责执行一种特定的原子技能。


To maintain the specialized skills of individual atomic experts, we first derive an atomic action abstraction from the high-level task instruction and environmental observation via thinking pipeline. This abstraction is deterministically mapped to a fixed high-dimensional embedding ${Z}_{\sigma } \in  {\mathbb{R}}^{d}$ , which serves as the conditioning signal for the skill router. The router computes a probability distribution over experts as:
为保持各原子专家的专业技能，我们首先通过思考流水线从高层任务指令和环境观测中推导出原子动作抽象。该抽象被确定性地映射为一个固定的高维嵌入 ${Z}_{\sigma } \in  {\mathbb{R}}^{d}$，作为技能路由器的条件信号。路由器计算专家概率分布如下：


$$
{w}_{k} = \operatorname{Router}\left( {Z}_{\sigma }\right) ,\;k \in  \{ 1,2,\ldots ,K\} , \tag{2}
$$



where $K$ denotes the number of atomic skill experts. We adopt a sparse activation strategy: only the top-scoring expert is selected for action generation. Let $k$ be the index of the activated expert,and let its raw score be ${w}_{k}$ . The final action chunk ${A}_{t}$ is computed as a weighted combination of the shared expert and the selected atomic expert:
其中 $K$ 表示原子技能专家的数量。我们采用稀疏激活策略：仅选择得分最高的专家进行动作生成。设 $k$ 为激活专家的索引，其原始得分为 ${w}_{k}$。最终动作块 ${A}_{t}$ 由共享专家和所选原子专家的加权组合计算得出：


$$
{F}_{\text{ out }} = \left( {1 - {w}_{k}}\right)  \cdot  {F}_{\text{ share }}\left( {x}_{t}\right)  + {w}_{k} \cdot  {F}_{k}\left( {x}_{t}\right) , \tag{3}
$$



where ${\mathbf{x}}_{t}$ denotes the current multimodal input $\left\lbrack  {{O}_{t}^{1 : n},\ell ,{s}_{t}}\right\rbrack$ . This architecture enables the system to retain the strong generalization capability of ${\pi }_{0}$ while achieving high-fidelity execution of specific skills through dedicated experts.
其中 ${\mathbf{x}}_{t}$ 表示当前的各种模态输入 $\left\lbrack  {{O}_{t}^{1 : n},\ell ,{s}_{t}}\right\rbrack$。该架构使系统既能保留 ${\pi }_{0}$ 强大的泛化能力，又能通过专用专家模型实现特定技能的高保真执行。


### 3.4. Continual Learning with Skill Expansion
### 3.4. 基于技能扩展的持续学习


In real-world deployments, robots inevitably encounter new tasks that require atomic skills not previously observed during training. Directly incorporating these novel skills into the existing skill library and retraining the entire model often leads to catastrophic forgetting, significantly impairing the performance of previously learned skills.
在实际部署中，机器人不可避免地会遇到训练阶段未曾见过的原子技能相关新任务。直接将这些新技能纳入现有技能库并对整个模型进行重训练，往往会导致灾难性遗忘，从而显著损害先前习得技能的表现。


AtomicVLA adopts a modular skill-expert mechanism, which enables continual scalability of the skill library. Specifically, as introduced in Sec. 3.3, each atomic skill is mapped to a fixed high-dimensional embedding vector ${Z}_{\sigma }$ , providing an explicit semantic abstraction of the skill. This design inherently enables incremental learning in lifelong settings: when a new atomic skill is introduced, it is sufficient to add a corresponding expert module to the existing architecture and extend the routing network.
AtomicVLA 采用模块化的技能-专家机制，实现了技能库的持续可扩展性。具体而言，如第 3.3 节所述，每个原子技能都被映射为一个固定的高维嵌入向量 ${Z}_{\sigma }$，从而提供该技能明确的语义抽象。这种设计从本质上支持了终身学习环境下的增量学习：当引入新的原子技能时，只需在现有架构中添加相应的专家模块并扩展路由网络即可。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_019ed0.jpg"/>



Figure 3. Inference Example of AtomicVLA. We visualize two tasks from LIBERO-LONG. For each task, the top row shows the task progression, and the bottom row shows AtomicVLA's inferred outputs. Gray blocks denote Thinking, while colored blocks indicate Acting, with colors corresponding to the activated skill experts. The left row shows the initial task state (top) and the skill-expert activation during inference (bottom).
图 3. AtomicVLA 推理示例。我们可视化了来自 LIBERO-LONG 的两个任务。对于每个任务，顶行显示任务进度，底行显示 AtomicVLA 的推理输出。灰色块表示“思考”（Thinking），彩色块表示“执行”（Acting），颜色对应于被激活的技能专家。左侧一列显示了初始任务状态（上）和推理过程中的技能专家激活情况（下）。


To ensure smooth integration, the expanded router is initialized by copying weights from the original router, while the new routing branch is initialized with small random values. This initialization strategy allows the model to adapt to the enlarged skill set with minimal fine-tuning, while preserving the performance of previously acquired skills. Consequently, AtomicVLA achieves efficient and interference-free expansion of its atomic skill library, a crucial requirement for scalable lifelong robotic learning.
为确保平滑集成，扩展后的路由器通过复制原始路由器的权重进行初始化，而新的路由分支则使用较小的随机值进行初始化。这种初始化策略使模型能够以最小的微调代价适应扩充后的技能集，同时保留先前习得技能的性能。因此，AtomicVLA 实现了原子技能库的高效且无干扰扩展，这是实现可扩展终身机器人学习的关键需求。


### 3.5. Task Planning Embodied Data Generation
### 3.5. 任务规划具身数据生成


To obtain accurate and reliable annotations of atomic actions, we propose a trajectory-based atomic decomposition method grounded in principal-axis analysis. Traditional approaches often rely on Vision-Language Models for video understanding or optical flow-based motion features to segment action sequences. However, these methods are prone to ambiguity and noise, which typically require extensive manual post-processing to correct and refine the results.
为了获得准确可靠的原子动作标注，我们提出了一种基于主轴分析的轨迹原子分解方法。传统方法通常依赖视觉语言模型进行视频理解，或利用基于光流的运动特征来分割动作序列。然而，这些方法容易产生歧义和噪声，通常需要大量人工后处理来修正和优化结果。


In contrast, our method analyzes the key kinematic dimensions of the end-effector trajectory, including translational displacements $\left( {{\Delta x},{\Delta y},{\Delta z}}\right)$ ,rotational changes (Δroll, Δpitch, Δyaw), and binary gripper states, to achieve coarse but semantically meaningful segmentation of atomic actions. Specifically, for each short motion chunk, we identify the dominant mode of motion by comparing the magnitudes of translational and rotational components. Concurrently, gripper state transitions are tracked to infer action semantics and execution progress. For instance, a continuous decrease in the $z$ -coordinate combined with a gripper closing event indicates a "pick" action, whereas limited translational movement accompanied by significant rotation with a closed gripper is classified as a "turn" operation. This physics-informed decomposition produces temporally precise and semantically interpretable boundaries for atomic actions, substantially reducing the reliance on manual refinement.
相比之下，我们的方法通过分析末端执行器轨迹的关键运动维度，包括平移位移 $\left( {{\Delta x},{\Delta y},{\Delta z}}\right)$、旋转变化（Δroll, Δpitch, Δyaw）以及二值化夹爪状态，来实现原子动作粗略但具有语义意义的分割。具体而言，对于每个短运动片段，我们通过比较平移和旋转分量的大小来识别主导运动模式。同时，通过跟踪夹爪状态转换来推断动作语义和执行进度。例如，$z$ 坐标的持续减小结合夹爪闭合事件表示“抓取”动作，而有限的平移运动伴随夹爪闭合状态下的显著旋转则被归类为“旋转”操作。这种物理信息驱动的分解方法产生了时间上精确且语义上可解释的原子动作边界，大幅降低了对人工修正的依赖。


Based on the output of principal-axis analysis, we decompose a full task trajectory into a temporally ordered sequence of atomic action segments. To refine and validate the semantic labels of these segments, we employ the Intern-Video2.5 model [52] to interpret the corresponding video clips, enabling automatic correction and enrichment of the initial atomic action annotations. By aligning these refined labels with the full trajectory, we construct a structured reasoning chain comprising the sequence of executed atomic actions and the associated high-level plan for subsequent steps. This integrated representation not only improves the fidelity of atomic action annotation but also provides interpretable, step-by-step execution guidance that supports robust long-horizon task planning and decision-making.
基于主轴分析的输出，我们将完整的任务轨迹分解为按时间排序的原子动作片段序列。为了优化和验证这些片段的语义标签，我们采用 Intern-Video2.5 模型 [52] 来解读相应的视频片段，从而实现对初始原子动作标注的自动修正与丰富。通过将这些优化后的标签与完整轨迹对齐，我们构建了一个结构化的推理链，包含执行的原子动作序列以及后续步骤的相关高层规划。这种集成表示不仅提高了原子动作标注的保真度，还提供了可解释的、分步执行的指导，从而支持稳健的长程任务规划与决策。


## 4. Experiments
## 4. 实验


### 4.1. Experiments Setup
### 4.1. 实验设置


Benchmarks. We evaluate AtomicVLA and AtomicVLA* on two widely adopted robotic manipulation benchmarks: LIBERO [28] and CALVIN [33]. For the LIBERO benchmark, we assess model performance across all four task suites. To further examine the model's capability in long-horizon planning and compositional generalization, we perform additional experiments on the CALVIN benchmark using the ABC-D split.
基准测试。我们在两个广泛采用的机器人操作基准上评估了 AtomicVLA 和 AtomicVLA*：LIBERO [28] 和 CALVIN [33]。对于 LIBERO 基准，我们评估了模型在所有四个任务套件上的表现。为了进一步检验模型在长程规划和组合泛化方面的能力，我们使用 ABC-D 划分在 CALVIN 基准上进行了额外实验。


Table 1. Comparison of Different Methods on LIBERO Benchmark(%).
表 1. 不同方法在 LIBERO 基准上的对比（%）。


<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td>Octo [47]</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>75.1</td></tr><tr><td>OpenVLA [22]</td><td>84.9</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>SpatialVLA [37]</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>78.1</td></tr><tr><td>CoT-VLA [65]</td><td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td><td>81.1</td></tr><tr><td>${\pi }_{0}$ [3]</td><td>96.4</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td></tr><tr><td>${\pi }_{0.5}\left\lbrack  {17}\right\rbrack$</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>AtomicVLA (Ours)</td><td>96.8</td><td>98.0</td><td>96.4</td><td>95.2</td><td>96.6</td></tr><tr><td>AtomicVLA* (Ours)</td><td>98.8</td><td>98.8</td><td>97.2</td><td>96.2</td><td>97.8</td></tr></table>
<table><tbody><tr><td>方法</td><td>空间</td><td>物体</td><td>目标</td><td>长程</td><td>平均</td></tr><tr><td>Octo [47]</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>75.1</td></tr><tr><td>OpenVLA [22]</td><td>84.9</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>SpatialVLA [37]</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>78.1</td></tr><tr><td>CoT-VLA [65]</td><td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td><td>81.1</td></tr><tr><td>${\pi }_{0}$ [3]</td><td>96.4</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td></tr><tr><td>${\pi }_{0.5}\left\lbrack  {17}\right\rbrack$</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>AtomicVLA (本文)</td><td>96.8</td><td>98.0</td><td>96.4</td><td>95.2</td><td>96.6</td></tr><tr><td>AtomicVLA* (本文)</td><td>98.8</td><td>98.8</td><td>97.2</td><td>96.2</td><td>97.8</td></tr></tbody></table>


Table 2. Long-horizon Robotic Manipulation Evaluation on CALVIN Benchmark(%).
表 2. CALVIN 基准测试中的长程机器人操作评估（%）。


<table><tr><td rowspan="2">Method</td><td rowspan="2">Task</td><td colspan="5">Tasks Completed in a Row</td><td rowspan="2">Avg. Len↑</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>${\pi }_{0}\left\lbrack  3\right\rbrack$</td><td>$\mathrm{{ABC}} \rightarrow  \mathrm{D}$</td><td>94.3</td><td>87.0</td><td>77.9</td><td>68.5</td><td>59.4</td><td>3.87</td></tr><tr><td>${\pi }_{0.5}\left\lbrack  {17}\right\rbrack$</td><td>$\mathrm{{ABC}} \rightarrow  \mathrm{D}$</td><td>91.9</td><td>84.6</td><td>79.4</td><td>75.5</td><td>71.0</td><td>4.02</td></tr><tr><td>AtomicVLA (Ours)</td><td>$\mathrm{{ABC}} \rightarrow  \mathrm{D}$</td><td>95.0</td><td>87.8</td><td>81.9</td><td>75.0</td><td>69.1</td><td>4.09</td></tr><tr><td>AtomicVLA* (Ours)</td><td>$\mathrm{{ABC}} \rightarrow  \mathrm{D}$</td><td>94.1</td><td>88.7</td><td>85.2</td><td>81.7</td><td>77.6</td><td>4.27</td></tr></table>
<table><tbody><tr><td rowspan="2">方法</td><td rowspan="2">任务</td><td colspan="5">连续完成任务数</td><td rowspan="2">平均长度↑</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>${\pi }_{0}\left\lbrack  3\right\rbrack$</td><td>$\mathrm{{ABC}} \rightarrow  \mathrm{D}$</td><td>94.3</td><td>87.0</td><td>77.9</td><td>68.5</td><td>59.4</td><td>3.87</td></tr><tr><td>${\pi }_{0.5}\left\lbrack  {17}\right\rbrack$</td><td>$\mathrm{{ABC}} \rightarrow  \mathrm{D}$</td><td>91.9</td><td>84.6</td><td>79.4</td><td>75.5</td><td>71.0</td><td>4.02</td></tr><tr><td>AtomicVLA（本文方法）</td><td>$\mathrm{{ABC}} \rightarrow  \mathrm{D}$</td><td>95.0</td><td>87.8</td><td>81.9</td><td>75.0</td><td>69.1</td><td>4.09</td></tr><tr><td>AtomicVLA*（本文方法）</td><td>$\mathrm{{ABC}} \rightarrow  \mathrm{D}$</td><td>94.1</td><td>88.7</td><td>85.2</td><td>81.7</td><td>77.6</td><td>4.27</td></tr></tbody></table>


Training setup. We build AtomicVLA and AtomicVLA* upon the pretrained ${\pi }_{0}$ and ${\pi }_{0.5}$ foundation model. The models were trained using robot trajectory data formatted according to the Lerobot standard. We use 5 skill experts for both the LIBERO benchmark suite and real-world robot experiments. For the CALVIN benchmark, we employ 8 skill experts to cover its broader task vocabulary. Further implementation details are provided in the Appendix.
训练设置。我们基于预训练的 ${\pi }_{0}$ 和 ${\pi }_{0.5}$ 基础模型构建了 AtomicVLA 和 AtomicVLA*。模型使用符合 Lerobot 标准的机器人轨迹数据进行训练。我们在 LIBERO 基准套件和真实机器人实验中均使用了 5 个技能专家。对于 CALVIN 基准，我们采用了 8 个技能专家以覆盖其更广泛的任务词汇。更多实现细节请参阅附录。


Real-world robot. We conduct real-world experiments using a Franka robotic arm, which includes three long-horizon tasks and five different types of short tasks. For each short-horizon task, we collect 50 trajectories, while each long-horizon task contains 100 trajectories, resulting in a total of 550 real-world demonstration trajectories. The five short tasks cover different categories of manipulation actions, including Grasp block, Stack blocks, Close microwave, Press button, and Open drawer. The long-horizon tasks include:
真实机器人。我们使用 Franka 机械臂进行真实世界实验，包含三个长程任务和五种不同类型的短程任务。每个短程任务收集 50 条轨迹，每个长程任务包含 100 条轨迹，总计 550 条真实世界演示轨迹。这五个短程任务涵盖了不同类别的操作动作，包括抓取方块、堆叠方块、关闭微波炉、按下按钮和打开抽屉。长程任务包括：


- Objects in plate: place all blocks on the table into a green plate.
- 盘中物体：将桌上所有方块放入绿色盘子中。


- Object into drawer: open the top drawer and place the block inside.
- 物体入抽屉：打开顶层抽屉并将方块放入其中。


- Object into microwave: place the plate into the microwave and close the door.
- 物体入微波炉：将盘子放入微波炉并关上门。


### 4.2. Results on Simulation
### 4.2. 仿真实验结果


Results on LIBERO. As shown in Tab. 1, AtomicVLA achieves an average success rate of 96.6% across the four suites, outperforming the strong baseline by 2.4%. Notably, on the most challenging LIBERO-LONG suite, Atom-icVLA attains a success rate of 95.2%, representing a 10% improvement over the ${\pi }_{0}$ . Furthermore,AtomicVLA* demonstrates even stronger performance, reaching an average success rate of 97.8% and 96.2% on LIBERO-LONG.
LIBERO 上的结果。如表 1 所示，AtomicVLA 在四个套件上的平均成功率达到 96.6%，比强基线模型高出 2.4%。值得注意的是，在最具挑战性的 LIBERO-LONG 套件上，AtomicVLA 达到了 95.2% 的成功率，较 ${\pi }_{0}$ 提升了 10%。此外，AtomicVLA* 表现更为出色，平均成功率达到 97.8%，在 LIBERO-LONG 上达到 96.2%。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_e4c1fb.jpg"/>



Figure 4. Error Recovery Capability Demonstration. When encountering a skill execution failure, AtomicVLA automatically assesses the progress and re-executes the current skill.
图 4. 错误恢复能力演示。当遇到技能执行失败时，AtomicVLA 会自动评估进度并重新执行当前技能。


This superior performance can be attributed to the core mechanism of AtomicVLA, which explicitly decomposes long-horizon tasks into a sequence of atomic skill abstractions and dynamically activates the corresponding skill experts. The "decompose-plan-compose" paradigm naturally aligns with the structure of multi-stage robotic tasks. As illustrated in Fig. 3, at the beginning of each atomic subtask, AtomicVLA generates a precise skill-level action abstraction to guide the selection of the appropriate expert. Importantly, when an execution failure occurs, for example, the butter is grasped but subsequently dropped as illustrated in Fig. 4, AtomicVLA can detect the task anomaly, regenerate a new atomic skill abstraction, and recover from the error to resume task execution.
这种卓越的性能归功于 AtomicVLA 的核心机制，即显式地将长程任务分解为一系列原子技能抽象，并动态激活相应的技能专家。“分解-规划-组合”范式与多阶段机器人任务的结构自然契合。如图 3 所示，在每个原子子任务开始时，AtomicVLA 会生成精确的技能级动作抽象，以指导选择合适的专家。重要的是，当执行失败发生时（例如图 4 所示的黄油被抓取后掉落），AtomicVLA 能够检测到任务异常，重新生成新的原子技能抽象，并从错误中恢复以继续执行任务。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_283d0f.jpg"/>



Figure 5. Demonstrations show the execution process of AtomicVLA* (second row) and baselines ${\pi }_{0.5}$ (first row).
图 5. 演示展示了 AtomicVLA*（第二行）与基线 ${\pi }_{0.5}$（第一行）的执行过程。


Table 3. Long-horizon Multi-task Experiments(%). InP, IntoD, and IntoM stand for Objects in plate, Object into drawer, Object into microwave, respectively.
表 3. 长程多任务实验（%）。InP、IntoD 和 IntoM 分别代表盘中物体、物体入抽屉和物体入微波炉。


<table><tr><td>Method</td><td>InP</td><td>IntoD</td><td>IntoM</td><td>Avg.</td><td>$\Delta$ Avg.</td></tr><tr><td>${\pi }_{0}\left\lbrack  3\right\rbrack$</td><td>45</td><td>55</td><td>10</td><td>36.7</td><td>-</td></tr><tr><td>${\pi }_{0.5}\left\lbrack  {17}\right\rbrack$</td><td>65</td><td>35</td><td>35</td><td>45</td><td>-</td></tr><tr><td>AtomicVLA</td><td>65</td><td>60</td><td>45</td><td>56.7</td><td>+20.0↑</td></tr><tr><td>AtomicVLA*</td><td>75</td><td>60</td><td>55</td><td>63.3</td><td>+18.3 ↑</td></tr></table>
<table><tbody><tr><td>方法</td><td>InP</td><td>IntoD</td><td>IntoM</td><td>平均值</td><td>$\Delta$ 平均值</td></tr><tr><td>${\pi }_{0}\left\lbrack  3\right\rbrack$</td><td>45</td><td>55</td><td>10</td><td>36.7</td><td>-</td></tr><tr><td>${\pi }_{0.5}\left\lbrack  {17}\right\rbrack$</td><td>65</td><td>35</td><td>35</td><td>45</td><td>-</td></tr><tr><td>AtomicVLA</td><td>65</td><td>60</td><td>45</td><td>56.7</td><td>+20.0↑</td></tr><tr><td>AtomicVLA*</td><td>75</td><td>60</td><td>55</td><td>63.3</td><td>+18.3 ↑</td></tr></tbody></table>


Results on Calvin. As shown in Tab. 2, AtomicVLA achieves an average task length of 4.09, outperforming the ${\pi }_{0}$ baseline by 0.22,while AtomicVLA* reaches an average task length of 4.27,outperforming the ${\pi }_{0.5}$ baseline by 0.25. Notably, AtomicVLA* demonstrates superior overall task completion rate with relative improvements of 5.8%, 6.2%, and 6.6% on the last three stages of the evaluation sequence. These results indicate that AtomicVLA is particularly effective in handling temporally extended and sequential manipulation tasks.
Calvin 上的结果。如表 2 所示，AtomicVLA 的平均任务长度达到 4.09，比 ${\pi }_{0}$ 基线高出 0.22；而 AtomicVLA* 的平均任务长度达到 4.27，比 ${\pi }_{0.5}$ 基线高出 0.25。值得注意的是，AtomicVLA* 在评估序列的最后三个阶段表现出更优的整体任务完成率，相对提升分别为 5.8%、6.2% 和 6.6%。这些结果表明，AtomicVLA 在处理时间跨度长且具有顺序性的操作任务时尤为有效。


As illustrated in Fig. 4, we also observe that Atom-icVLA exhibits a capability for error recovery in experiments. However, due to the evaluation constraints of the CALVIN benchmark, successful recoveries after failures are not considered valid completions, which prevents subsequent tasks from being executed. As a result, the reported performance metrics may slightly underestimate the true capability of the model.
如图 4 所示，我们还观察到 AtomicVLA 在实验中展现出了错误恢复能力。然而，由于 CALVIN 基准测试的评估限制，失败后的成功恢复不被视为有效完成，这导致后续任务无法继续执行。因此，报告的性能指标可能略微低估了该模型的真实能力。


### 4.3. Results on Real-world Robot
### 4.3. 真实机器人实验结果


Long-horizon Tasks. We perform mixed training using the collected data from three long-horizon tasks. As shown in Tab. 3, AtomicVLA and AtomicVLA* outperform the baseline model by 20% and 18.3%, respectively. As illustrated in Fig. 5, we present two representative long-horizon tasks. AtomicVLA* reliably completes the experimental configurations that ${\pi }_{0.5}$ fails to accomplish,and this advantage becomes more evident in tasks involving door-closing operations. Building on this observation, AtomicVLA* demonstrates stronger robustness and execution stability across complex manipulation sequences.
长跨度任务。我们使用收集到的三个长跨度任务数据进行混合训练。如表 3 所示，AtomicVLA 和 AtomicVLA* 分别比基线模型高出 20% 和 18.3%。如图 5 所示，我们展示了两个典型的长跨度任务。AtomicVLA* 能可靠地完成 ${\pi }_{0.5}$ 无法实现的实验配置，这种优势在涉及关门操作的任务中表现得更为明显。基于此观察，AtomicVLA* 在复杂的操纵序列中展现出更强的鲁棒性和执行稳定性。


Previous real-world studies on robotic manipulation typically focus on training and evaluating a single specific task, while joint training across multiple heterogeneous tasks has been relatively uncommon. Our observations indicate that combining tasks with large differences can lead to mutual interference, which in turn limits overall performance. This effect becomes particularly pronounced in tasks that involve significant changes in gripper states across different execution stages. For instance, in the "Object into drawer" task, the drawer-opening subtask does not require gripper closure, which can adversely affect the model's behavior on other grasping-related tasks, resulting in unintended gripper opening or closing actions, as illustrated in Fig. 6. By constructing an explicit library of atomic skills, AtomicVLA effectively mitigates such cross-task interference. Each skill precisely activates its corresponding expert to execute the required operation, which substantially alleviates the interference between heterogeneous skills and overcomes the performance bottleneck of mixed multi-task training.
以往关于机器人操作的真实世界研究通常侧重于训练和评估单一特定任务，而跨多个异构任务的联合训练相对较少。我们的观察表明，结合差异巨大的任务会导致相互干扰，进而限制整体性能。这种影响在涉及不同执行阶段夹爪状态显著变化的任务中尤为明显。例如，在“将物体放入抽屉”任务中，开抽屉子任务不需要夹爪闭合，这可能会对模型在其他抓取相关任务上的行为产生负面影响，导致意外的夹爪开合动作，如图 6 所示。通过构建显式的原子技能库，AtomicVLA 有效缓解了这种跨任务干扰。每项技能都能精确激活其对应的专家来执行所需操作，从而大幅减轻了异构技能间的干扰，克服了混合多任务训练的性能瓶颈。


Continual learning skills. To evaluate the effectiveness of our proposed lifelong skill expansion mechanism in real-world scenarios, we conduct training and evaluation on short-horizon task dataset consisting of five diverse manipulation categories. In this experiment, the "open" operation is treated as a new atomic skill, which is introduced as an additional capability during the continual learning phase after the initial training stage. Specifically, we first perform mixed training on four short-horizon tasks and train the "open" skill independently on top of the pretrained model.
持续学习技能。为了评估我们提出的终身技能扩展机制在真实场景中的有效性，我们对包含五个不同操作类别的短跨度任务数据集进行了训练和评估。在本实验中，“打开”操作被视为一项新的原子技能，作为初始训练阶段后持续学习阶段的额外能力引入。具体而言，我们首先对四个短跨度任务进行混合训练，并在预训练模型的基础上独立训练“打开”技能。


Learning a new skill often causes substantial interference with previously acquired abilities in conventional baseline models, leading to noticeable performance degradation. As illustrated in Fig. 6, in a case that was originally expected to succeed, the task could not be completed after continual learning. The gripper failed to close promptly after reaching the target position. As shown in Tab. 4, the average success rate of ${\pi }_{0.5}$ decreases by approximately 15%,with the stack task exhibiting the most severe interference, showing a 20% decrease. In contrast, AtomicVLA* maintains stable performance after continual learning. Owing to its structured skill library management, the previously learned skills remain largely unaffected. Moreover, under the same number of training steps, AtomicVLA* acquires new skills more efficiently and achieves an overall improvement of 21% across all five tasks compared to ${\pi }_{0.5}$ . These findings highlight our effectiveness for continual learning.
在传统的基线模型中，学习新技能往往会对先前习得的能力造成严重干扰，导致性能显著下降。如图 6 所示，在原本预期成功的案例中，持续学习后任务无法完成，夹爪在到达目标位置后未能及时闭合。如表 4 所示，${\pi }_{0.5}$ 的平均成功率下降了约 15%，其中堆叠任务受到的干扰最为严重，下降了 20%。相比之下，AtomicVLA* 在持续学习后保持了稳定的性能。得益于其结构化的技能库管理，先前习得的技能基本未受影响。此外，在相同的训练步数下，AtomicVLA* 获取新技能的效率更高，且与 ${\pi }_{0.5}$ 相比，在所有五个任务上的整体性能提升了 21%。这些发现突显了我们在持续学习方面的有效性。


Table 4. Continual Learning with Skill Expansion(%). ΔAvg. represents the average performance change on the four base tasks after learning new skills compared with their performance before learning.CL is continual learning
表 4. 技能扩展下的持续学习（%）。ΔAvg. 表示学习新技能后，四个基础任务的平均性能变化（与学习前相比）。CL 为持续学习。


<table><tr><td>Method</td><td>Grasp</td><td>Stack</td><td>Close</td><td>Press</td><td>Open (new)</td><td>Avg.</td><td>$\Delta$ Avg.</td></tr><tr><td>${\pi }_{0.5}$ [17]</td><td>85</td><td>65</td><td>70</td><td>90</td><td>-</td><td>77.5</td><td>-</td></tr><tr><td>${\pi }_{0.5}$ [17] (CL)</td><td>70</td><td>45</td><td>60</td><td>75</td><td>55</td><td>61</td><td>-15.0 ↓</td></tr><tr><td>AtomicVLA*</td><td>95</td><td>80</td><td>70</td><td>100</td><td>-</td><td>86.3</td><td>-</td></tr><tr><td>AtomicVLA* (CL)</td><td>90</td><td>80</td><td>80</td><td>100</td><td>70</td><td>82</td><td>$- {1.3} \downarrow$</td></tr></table>
<table><tbody><tr><td>方法</td><td>抓取</td><td>堆叠</td><td>关闭</td><td>按压</td><td>打开（新增）</td><td>平均</td><td>$\Delta$ 平均</td></tr><tr><td>${\pi }_{0.5}$ [17]</td><td>85</td><td>65</td><td>70</td><td>90</td><td>-</td><td>77.5</td><td>-</td></tr><tr><td>${\pi }_{0.5}$ [17] (CL)</td><td>70</td><td>45</td><td>60</td><td>75</td><td>55</td><td>61</td><td>-15.0 ↓</td></tr><tr><td>AtomicVLA*</td><td>95</td><td>80</td><td>70</td><td>100</td><td>-</td><td>86.3</td><td>-</td></tr><tr><td>AtomicVLA* (CL)</td><td>90</td><td>80</td><td>80</td><td>100</td><td>70</td><td>82</td><td>$- {1.3} \downarrow$</td></tr></tbody></table>


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_e761cf.jpg"/>



Figure 6. Mixed-Training Skill Interference and Continual-Learning Degradation. The top two rows illustrate skill interference in long-horizon tasks: the first shows successful single-skill executions, while the second shows failures after mixed training. The bottom two rows show degradation after continual learning: the first row presents the performance of ${\pi }_{0.5}$ before learning new skills, and the second shows its performance afterward. Red and green boxes highlight the key differences.
图6. 混合训练中的技能干扰与持续学习退化。前两行展示了长程任务中的技能干扰：第一行显示了单项技能的成功执行，第二行显示了混合训练后的失败情况。后两行展示了持续学习后的退化现象：第一行呈现了 ${\pi }_{0.5}$ 在学习新技能前的表现，第二行则显示了学习后的表现。红色和绿色框突出了关键差异。


### 4.4. Ablation Study
### 4.4. 消融实验


We conduct ablation experiments on the LIBERO-LONG benchmark to evaluate the effectiveness of our skill-aware routing mechanism. Specifically, we compare AtomicVLA against three baselines: (i) a non-MoE ${\pi }_{0}$ -based baseline, (ii) a standard token-level Mixture-of-Experts (MoE) that selects experts independently for each action token, and (iii) a variant adapted from MoDE [39], which conditions expert selection on the denoising timestep $t$ (i.e.,using $t$ as the routing signal).
我们在 LIBERO-LONG 基准测试上进行了消融实验，以评估我们技能感知路由机制的有效性。具体而言，我们将 AtomicVLA 与三个基线进行了比较：(i) 基于非 MoE ${\pi }_{0}$ 的基线，(ii) 为每个动作 token 独立选择专家的标准 token 级混合专家模型 (MoE)，以及 (iii) 改编自 MoDE [39] 的变体，该变体根据去噪时间步 $t$ 来调节专家选择（即使用 $t$ 作为路由信号）。


As shown in Tab. 5, AtomicVLA achieves a success rate of 95.2%, outperforming the MoE baseline by 6.6% and the timestep-conditioned MoDE variant by 5.7%. The experimental results indicate that the performance gap between the MoE-based and MoDE-based methods is relatively small. This is primarily because both approaches rely on token-level expert routing, where the improvements largely stem from load balancing that distributes tokens across experts. As a result, each expert still learns a mixture of skills without clear specialization. In contrast, SG-MoE employs atomic skill abstractions as the routing criterion, which ensures that all tokens associated with a specific skill stage are consistently processed by the corresponding expert network. Consequently, each expert focuses on a single skill with a similar action distribution, reducing interference among different skills. Moreover, this notable performance gain demonstrates that routing experts based on semantically meaningful atomic skills, rather than on individual action tokens or denoising steps, leads to more coherent and efficient skill execution in long-horizon tasks.
如表5所示，AtomicVLA 的成功率达到了 95.2%，分别比 MoE 基线和基于时间步调节的 MoDE 变体高出 6.6% 和 5.7%。实验结果表明，基于 MoE 和基于 MoDE 的方法之间的性能差距相对较小。这主要是因为这两种方法都依赖于 token 级的专家路由，其性能提升主要源于在专家之间分配 token 的负载均衡。因此，每个专家仍然在学习多种技能的混合，而没有明确的专业化分工。相比之下，SG-MoE 采用原子技能抽象作为路由准则，确保了与特定技能阶段相关的所有 token 都能由相应的专家网络一致地处理。因此，每个专家专注于具有相似动作分布的单一技能，从而减少了不同技能之间的干扰。此外，这一显著的性能提升证明，基于语义明确的原子技能而非单个动作 token 或去噪步骤来路由专家，能在长程任务中实现更连贯、更高效的技能执行。


Table 5. Results on LIBERO Benchmark(%).
表5. LIBERO 基准测试结果(%)。


<table><tr><td>Method</td><td>LIBERO-LONG</td></tr><tr><td>${\pi }_{0}\left\lbrack  3\right\rbrack$</td><td>85.2</td></tr><tr><td>+ MoE</td><td>88.6</td></tr><tr><td>+ MoDE [39]</td><td>89.5</td></tr><tr><td>+ SG-MoE (Ours)</td><td>95.2</td></tr></table>
<table><tbody><tr><td>方法</td><td>LIBERO-LONG</td></tr><tr><td>${\pi }_{0}\left\lbrack  3\right\rbrack$</td><td>85.2</td></tr><tr><td>+ MoE</td><td>88.6</td></tr><tr><td>+ MoDE [39]</td><td>89.5</td></tr><tr><td>+ SG-MoE (本文方法)</td><td>95.2</td></tr></tbody></table>


## 5. Conclusion
## 5. 结论


In this paper, we introduce AtomicVLA, an end-to-end framework that unifies task planning and action execution for long-horizon tasks and continual skill expansion. We design a unified architecture capable of adaptively deciding task plans and generating latent action outputs, and construct an atomic skill-guided expert library based on our proposed SG-MoE architecture and the specialized skill router. AtomicVLA is inherently scalable: when learning new skills, it only requires extending the skill router and adding the corresponding new skill experts to rapidly acquire the novel capabilities. We validate AtomicVLA in both simulated and real-world robotic environments, demonstrating its superior performance in long-horizon tasks and continual learning. Notably, it effectively mitigates skill interference arising from joint training and alleviates knowledge forgetting and performance degradation during continual skill acquisition, highlighting its significant potential for scalable continual learning in vision-language-action models.
本文介绍了 AtomicVLA，这是一个将长程任务的任务规划与动作执行统一起来，并支持持续技能扩展的端到端框架。我们设计了一种能够自适应决策任务规划并生成潜在动作输出的统一架构，并基于所提出的 SG-MoE 架构和专用技能路由构建了原子技能引导的专家库。AtomicVLA 具有内在的可扩展性：在学习新技能时，只需扩展技能路由并添加相应的新技能专家，即可快速获得新能力。我们在仿真和真实机器人环境中验证了 AtomicVLA，证明了其在长程任务和持续学习中的卓越性能。值得注意的是，它有效地缓解了联合训练带来的技能干扰，并减轻了持续技能获取过程中的知识遗忘和性能下降，凸显了其在视觉-语言-动作模型中实现可扩展持续学习的巨大潜力。


## Acknowledgments
## 致谢


This work was supported by the National Key Research and Development Program of China (2024YFE0203100), the Scientific Research Innovation Capability Support Project for Young Faculty (No. ZYGXQNJSKYCXNLZCXM-I28), the National Natural Science Foundation of China (NSFC) under Grants No. 62476293, No. 62372482 and No. 62272494, and in part by the Major Key Project of PCL (Grant No. PCL2025A17) and the General Embodied AI Center of Sun Yat-sen University.
本研究得到了国家重点研发计划（2024YFE0203100）、青年教师科研创新能力支持项目（No. ZYGXQNJSKYCXNLZCXM-I28）、国家自然科学基金（NSFC，项目编号：62476293、62372482 和 62272494）的资助，并部分得到了鹏城实验室重大重点项目（项目编号：PCL2025A17）及中山大学通用具身智能中心的资助。


## References
## 参考文献


[1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Cheb-otar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Ir-pan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Ser-manet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as i can, not as i say: Grounding language in robotic affordances, 2022. 1
[1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Cheb-otar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Ir-pan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Ser-manet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as i can, not as i say: Grounding language in robotic affordances, 2022. 1


[2] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisensch-los, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. Paligemma: A versatile 3b vlm for transfer, 2024. 1, 2
[2] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisensch-los, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. Paligemma: A versatile 3b vlm for transfer, 2024. 1, 2


[3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. ${\pi }_{0}$ : A vision-language-action flow model for general robot control, 2024. 1, 2, 6, 7, 8
[3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. ${\pi }_{0}$ : A vision-language-action flow model for general robot control, 2024. 1, 2, 6, 7, 8


[4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakr-ishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kan-ishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023. 1, 2
[4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakr-ishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kan-ishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: 视觉-语言-动作模型将网络知识迁移至机器人控制，2023. 1, 2


[5] Hao Chen, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Ren-rui Zhang, Xiaoqi Li, Xiao He, Yandong Guo, Chi-Wing Fu, Shanghang Zhang, and Pheng-Ann Heng. Fast-in-slow: A dual-system foundation model unifying fast manipulation within slow reasoning, 2025. 2
[5] Hao Chen, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Ren-rui Zhang, Xiaoqi Li, Xiao He, Yandong Guo, Chi-Wing Fu, Shanghang Zhang, and Pheng-Ann Heng. Fast-in-slow：一种统一慢速推理与快速操作的双系统基础模型，2025. 2


[6] Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, and Radu Sori-cut. Pali-3 vision language models: Smaller, faster, stronger, 2023. 1
[6] Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, and Radu Sori-cut. Pali-3 视觉语言模型：更小、更快、更强，2023. 1


[7] Rohan Chitnis, Shubham Tulsiani, Saurabh Gupta, and Ab-hinav Gupta. Efficient bimanual manipulation using learned task schemas, 2020. 3
[7] Rohan Chitnis, Shubham Tulsiani, Saurabh Gupta, and Ab-hinav Gupta. 基于学习任务模式的高效双臂操作，2020. 3


[8] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseek-moe: Towards ultimate expert specialization in mixture-of-experts language models, 2024. 2
[8] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseek-moe：迈向专家混合语言模型中的极致专家专业化，2024. 2


[9] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tri-pathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models, 2024. 1, 2
[9] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tri-pathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo 和 pixmo：用于最先进视觉语言模型的开放权重与开放数据，2024. 1, 2


[10] Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon, Hiroki Furuta, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. Plan-and-act: Improving planning of agents for long-horizon tasks. arXiv preprint arXiv:2503.09572, 2025. 2
[10] Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon, Hiroki Furuta, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. Plan-and-act：改进长程任务中智能体的规划能力。arXiv 预印本 arXiv:2503.09572，2025. 2


[11] Huang Fang, Mengxi Zhang, Heng Dong, Wei Li, Zixuan Wang, Qifeng Zhang, Xueyun Tian, Yucheng Hu, and Hang Li. Robix: A unified model for robot interaction, reasoning and planning, 2025. 2
[11] Huang Fang, Mengxi Zhang, Heng Dong, Wei Li, Zixuan Wang, Qifeng Zhang, Xueyun Tian, Yucheng Hu, and Hang Li. Robix: 机器人交互、推理与规划的统一模型, 2025. 2


[12] Roy Fox, Richard Shin, William Paul, Yitian Zou, Dawn Song, Ken Goldberg, Pieter Abbeel, and Ion Stoica. Hierarchical variational imitation learning of control programs, 2019. 3
[12] Roy Fox, Richard Shin, William Paul, Yitian Zou, Dawn Song, Ken Goldberg, Pieter Abbeel, and Ion Stoica. 控制程序的分层变分模仿学习, 2019. 3


[13] Yingdong Hu, Fanqi Lin, Tong Zhang, Li Yi, and Yang Gao. Look before you leap: Unveiling the power of gpt-4v in robotic vision-language planning, 2023. 1
[13] Yingdong Hu, Fanqi Lin, Tong Zhang, Li Yi, and Yang Gao. 三思而后行：揭示 GPT-4V 在机器人视觉语言规划中的能力, 2023. 1


[14] Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, and Yang Gao. Copa: General robotic manipulation through spatial constraints of parts with foundation models, 2024. 2
[14] Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, and Yang Gao. Copa: 基于基础模型的部件空间约束通用机器人操作, 2024. 2


[15] Suning Huang, Zheyu Zhang, Tianhai Liang, Yihan Xu, Zhehao Kou, Chenhao Lu, Guowei Xu, Zhengrong Xue, and Huazhe Xu. Mentor: Mixture-of-experts network with task-oriented perturbation for visual reinforcement learning, 2025. 2
[15] Suning Huang, Zheyu Zhang, Tianhai Liang, Yihan Xu, Zhehao Kou, Chenhao Lu, Guowei Xu, Zhengrong Xue, and Huazhe Xu. Mentor: 用于视觉强化学习的面向任务扰动的专家混合网络, 2025. 2


[16] Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, Hunter Hancock, Karol Hausman, Gashon Hussein, Brian Ichter, Szymon Jakubczak, Rowan Jen, Tim Jones, Ben Katz, Liyiming Ke, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Yao Lu, Vishnu Mano, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Charvi Sharma, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachow-icz, Will Stoeckle, Alex Swerdlow, James Tanner, Marcel Torne, Quan Vuong, Anna Walling, Haohuan Wang, Blake Williams, Sukwon Yoo, Lili Yu, Ury Zhilinsky, and Zhiyuan Zhou. ${\pi }_{0.6}^{ * }$ : a vla that learns from experience,2025. 13
[16] Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, Hunter Hancock, Karol Hausman, Gashon Hussein, Brian Ichter, Szymon Jakubczak, Rowan Jen, Tim Jones, Ben Katz, Liyiming Ke, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Yao Lu, Vishnu Mano, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Charvi Sharma, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachow-icz, Will Stoeckle, Alex Swerdlow, James Tanner, Marcel Torne, Quan Vuong, Anna Walling, Haohuan Wang, Blake Williams, Sukwon Yoo, Lili Yu, Ury Zhilinsky, and Zhiyuan Zhou. ${\pi }_{0.6}^{ * }$ : 一个从经验中学习的 VLA 模型, 2025. 13


[17] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Gal-liker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. To.5: a vision-language-action model with open-world generalization, 2025. 2, 6, 7, 8, 13, 15
[17] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Gal-liker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. To.5: 一个具备开放世界泛化能力的视觉-语言-动作模型, 2025. 2, 6, 7, 8, 13, 15


[18] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, Xinda Xue, Qinghang Su, Huaihai Lyu, Xi-aolong Zheng, Jiaming Liu, Zhongyuan Wang, and Shang-hang Zhang. Robobrain: A unified brain model for robotic manipulation from abstract to concrete, 2025. 13
[18] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, Xinda Xue, Qinghang Su, Huaihai Lyu, Xi-aolong Zheng, Jiaming Liu, Zhongyuan Wang, and Shang-hang Zhang. Robobrain: 从抽象到具体的机器人操作统一大脑模型, 2025. 13


[19] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Deven-dra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024. 2
[19] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Deven-dra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts, 2024. 2


[20] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jian-ning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system vla model, 2025. 1
[20] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jian-ning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system vla model, 2025. 1


[21] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models, 2024. 1, 2
[21] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models, 2024. 1, 2


[22] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kol-lar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model, 2024. 1, 2, 6
[22] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kol-lar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model, 2024. 1, 2, 6


[23] Daehee Lee, Minjong Yoo, Woo Kyung Kim, Wonje Choi, and Honguk Woo. Incremental learning of retrievable skills for efficient continual task adaptation, 2025. 3
[23] Daehee Lee, Minjong Yoo, Woo Kyung Kim, Wonje Choi, and Honguk Woo. Incremental learning of retrievable skills for efficient continual task adaptation, 2025. 3


[24] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, and Ranjay Krishna. Molmoact: Action reasoning models that can reason in space, 2025. 2
[24] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, and Ranjay Krishna. Molmoact: Action reasoning models that can reason in space, 2025. 2


[25] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. Simplevla-rl: Scaling vla training via reinforcement learning, 2025. 13
[25] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. Simplevla-rl: Scaling vla training via reinforcement learning, 2025. 13


[26] Fanqi Lin, Ruiqian Nai, Yingdong Hu, Jiacheng You, Jun-ming Zhao, and Yang Gao. Onetwovola: A unified vision-language-action model with adaptive reasoning, 2025. 1, 2
[26] Fanqi Lin, Ruiqian Nai, Yingdong Hu, Jiacheng You, Jun-ming Zhao, and Yang Gao. Onetwovola: A unified vision-language-action model with adaptive reasoning, 2025. 1, 2


[27] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. 3
[27] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. 3


[28] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning, 2023. 2, 5
[28] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning, 2023. 2, 5


[29] Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, Chengkai Hou, Mengdi Zhao, KC alex Zhou, Pheng-Ann Heng, and Shanghang Zhang. Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model, 2025. 2
[29] Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, Chengkai Hou, Mengdi Zhao, KC alex Zhou, Pheng-Ann Heng, and Shanghang Zhang. Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model, 2025. 2


[30] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation, 2025. 1, 3
[30] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation, 2025. 1, 3


[31] Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning, 2025. 13
[31] Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning, 2025. 13


[32] Xiaofeng Mao, Gabriele Giudici, Claudio Coppola, Kaspar Althoefer, Ildar Farkhatdinov, Zhibin Li, and Lorenzo Jamone. Dexskills: Skill segmentation using haptic data for learning autonomous long-horizon robotic manipulation tasks, 2024. 3
[32] Xiaofeng Mao, Gabriele Giudici, Claudio Coppola, Kaspar Althoefer, Ildar Farkhatdinov, Zhibin Li, and Lorenzo Jamone. Dexskills: Skill segmentation using haptic data for learning autonomous long-horizon robotic manipulation tasks, 2024. 3


[33] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks, 2022. 2, 5
[33] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks, 2022. 2, 5


[34] Atharva Mete, Haotian Xue, Albert Wilcox, Yongxin Chen, and Animesh Garg. Quest: Self-supervised skill abstractions for learning continuous control, 2024. 3
[34] Atharva Mete, Haotian Xue, Albert Wilcox, Yongxin Chen, and Animesh Garg. Quest: Self-supervised skill abstractions for learning continuous control, 2024. 3


[35] NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cher-niadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llon-top, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. 1
[35] NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cher-niadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llon-top, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. 1


[36] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models, 2025. 2
[36] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models, 2025. 2


[37] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. Spatialvla: Exploring spatial representations for visual-language-action model, 2025. 6
[37] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. Spatialvla: Exploring spatial representations for visual-language-action model, 2025. 6


[38] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Jun-yang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. 1
[38] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Jun-yang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. 1


[39] Moritz Reuss, Jyothish Pari, Pulkit Agrawal, and Rudolf Li-outikov. Efficient diffusion transformer policies with mixture of expert denoisers for multitask learning, 2024. 2, 8
[39] Moritz Reuss, Jyothish Pari, Pulkit Agrawal, and Rudolf Li-outikov. Efficient diffusion transformer policies with mixture of expert denoisers for multitask learning, 2024. 2, 8


[40] Kaushik Roy, Akila Dissanayake, Brendan Tidd, and Pey-man Moghadam. M2distill: Multi-modal distillation for lifelong imitation learning, 2025. 3
[40] Kaushik Roy, Akila Dissanayake, Brendan Tidd, and Pey-man Moghadam. M2distill: Multi-modal distillation for lifelong imitation learning, 2025. 3


[41] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyim-ing Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, and Chelsea Finn. Hi robot: Open-ended instruction following with hierarchical vision-language-action models, 2025. 1, 2
[41] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyim-ing Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, and Chelsea Finn. Hi robot: Open-ended instruction following with hierarchical vision-language-action models, 2025. 1, 2


[42] Robin Strudel, Alexander Pashevich, Igor Kalevatykh, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Learning to combine primitive skills: A step towards versatile robotic manipulation, 2020. 3
[42] Robin Strudel, Alexander Pashevich, Igor Kalevatykh, Ivan Laptev, Josef Sivic, and Cordelia Schmid. 学习组合原始技能：迈向通用机器人操作的一步，2020. 3


[43] Jiankai Sun, Aidan Curtis, Yang You, Yan Xu, Michael Koehle, Qianzhong Chen, Suning Huang, Leonidas Guibas, Sachin Chitta, Mac Schwager, and Hui Li. Arch: Hierarchical hybrid learning for long-horizon contact-rich robotic assembly, 2025. 3
[43] Jiankai Sun, Aidan Curtis, Yang You, Yan Xu, Michael Koehle, Qianzhong Chen, Suning Huang, Leonidas Guibas, Sachin Chitta, Mac Schwager, and Hui Li. Arch：用于长程接触式机器人装配的分层混合学习，2025. 3


[44] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Xiansheng Chen, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning of vision language models, 2025. 13
[44] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Xiansheng Chen, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft：视觉语言模型视觉推理的强化微调，2025. 13


[45] BAAI RoboBrain Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Xiansheng Chen, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, Yi Han, Yingbo Tang, Xi-angqi Xu, Wei Guo, Yaoxu Lyu, Yijie Xu, Jiayu Shi, Mengfei Du, Cheng Chi, Mengdi Zhao, Xiaoshuai Hao, Junkai Zhao, Xiaojie Zhang, Shanyu Rong, Huaihai Lyu, Zhengliang Cai, Yankai Fu, Ning Chen, Bolun Zhang, Lingfeng Zhang, Shuyi Zhang, Dong Liu, Xi Feng, Songjing Wang, Xiaodan Liu, Yance Jiao, Mengsi Lyu, Zhuo Chen, Chenrui He, Yulong Ao, Xue Sun, Zheqi He, Jingshu Zheng, Xi Yang, Donghai Shi, Kunchang Xie, Bochao Zhang, Shaokai Nie, Chunlei Men, Yonghua Lin, Zhongyuan Wang, Tiejun Huang, and Shanghang Zhang. Robobrain 2.0 technical report, 2025. 13
[45] BAAI RoboBrain Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Xiansheng Chen, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, Yi Han, Yingbo Tang, Xi-angqi Xu, Wei Guo, Yaoxu Lyu, Yijie Xu, Jiayu Shi, Mengfei Du, Cheng Chi, Mengdi Zhao, Xiaoshuai Hao, Junkai Zhao, Xiaojie Zhang, Shanyu Rong, Huaihai Lyu, Zhengliang Cai, Yankai Fu, Ning Chen, Bolun Zhang, Lingfeng Zhang, Shuyi Zhang, Dong Liu, Xi Feng, Songjing Wang, Xiaodan Liu, Yance Jiao, Mengsi Lyu, Zhuo Chen, Chenrui He, Yulong Ao, Xue Sun, Zheqi He, Jingshu Zheng, Xi Yang, Donghai Shi, Kunchang Xie, Bochao Zhang, Shaokai Nie, Chunlei Men, Yonghua Lin, Zhongyuan Wang, Tiejun Huang, and Shanghang Zhang. Robobrain 2.0 技术报告，2025. 13


[46] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bohez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D'Ambrosio, Sudeep Dasari, Todor Davchev, Co-line Devin, Norman Di Palo, Tianli Ding, Adil Dost-mohamed, Danny Driess, Yilun Du, Debidatta Dwibedi, Michael Elabd, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Leonard Hasenclever, Nicolas Heess, Brandon Hernaez, Alexander Herzog, R. Alex Hofer, Jan Hump-lik, Atil Iscen, Mithun George Jacob, Deepali Jain, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Chase Kew, Jerad Kirkland, Sean Kirmani, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Isabel Leal, Alex X. Lee, Tsang-Wei Edward Lee, Jacky Liang, Yixin Lin, Sharath Maddineni, Anirudha Majumdar, Assaf Hurwitz Michaely, Robert Moreno, Michael Neunert, Francesco Nori, Carolina Parada, Emilio Parisotto, Peter Pastor, Acorn Pooley, Kanishka Rao, Krista Reymann, Dorsa Sadigh, Stefano Saliceti, Pannag Sanketi, Pierre Sermanet, Dhruv Shah, Mohit Sharma, Kathryn Shea, Charles Shu, Vikas Sind-hwani, Sumeet Singh, Radu Soricut, Jost Tobias Springen-berg, Rachel Sterneck, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Vincent Vanhoucke, Jake Varley, Grace Vesom, Giulia Vezzani, Oriol Vinyals, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Fei Xia, Ted Xiao, Annie Xie, Jinyu Xie, Peng Xu, Sichun Xu, Ying Xu, Zhuo Xu, Yuxiang Yang, Rui Yao, Sergey Yaroshenko, Wenhao Yu, Wentao Yuan, Jing-wei Zhang, Tingnan Zhang, Allan Zhou, and Yuxiang Zhou. Gemini robotics: Bringing ai into the physical world, 2025. 1,2
[46] Gemini 机器人团队, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bohez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D'Ambrosio, Sudeep Dasari, Todor Davchev, Co-line Devin, Norman Di Palo, Tianli Ding, Adil Dost-mohamed, Danny Driess, Yilun Du, Debidatta Dwibedi, Michael Elabd, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Leonard Hasenclever, Nicolas Heess, Brandon Hernaez, Alexander Herzog, R. Alex Hofer, Jan Hump-lik, Atil Iscen, Mithun George Jacob, Deepali Jain, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Chase Kew, Jerad Kirkland, Sean Kirmani, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Isabel Leal, Alex X. Lee, Tsang-Wei Edward Lee, Jacky Liang, Yixin Lin, Sharath Maddineni, Anirudha Majumdar, Assaf Hurwitz Michaely, Robert Moreno, Michael Neunert, Francesco Nori, Carolina Parada, Emilio Parisotto, Peter Pastor, Acorn Pooley, Kanishka Rao, Krista Reymann, Dorsa Sadigh, Stefano Saliceti, Pannag Sanketi, Pierre Sermanet, Dhruv Shah, Mohit Sharma, Kathryn Shea, Charles Shu, Vikas Sind-hwani, Sumeet Singh, Radu Soricut, Jost Tobias Springen-berg, Rachel Sterneck, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Vincent Vanhoucke, Jake Varley, Grace Vesom, Giulia Vezzani, Oriol Vinyals, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Fei Xia, Ted Xiao, Annie Xie, Jinyu Xie, Peng Xu, Sichun Xu, Ying Xu, Zhuo Xu, Yuxiang Yang, Rui Yao, Sergey Yaroshenko, Wenhao Yu, Wentao Yuan, Jing-wei Zhang, Tingnan Zhang, Allan Zhou, 和 Yuxiang Zhou. Gemini 机器人：将人工智能引入物理世界, 2025. 1,2


[47] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy, 2024. 6
[47] Octo 模型团队, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, 和 Sergey Levine. Octo：一种开源通用机器人策略, 2024. 6


[48] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning, 2018. 3
[48] Aaron van den Oord, Oriol Vinyals, 和 Koray Kavukcuoglu. 神经离散表征学习, 2018. 3


[49] Weikang Wan, Yifeng Zhu, Rutav Shah, and Yuke Zhu. Lotus: Continual imitation learning for robot manipulation through unsupervised skill discovery, 2024. 3
[49] Weikang Wan, Yifeng Zhu, Rutav Shah, 和 Yuke Zhu. Lotus：通过无监督技能发现实现机器人操作的持续模仿学习, 2024. 3


[50] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Jun-yang Lin. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution, 2024. 1, 2
[50] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, 和 Jun-yang Lin. Qwen2-vl：增强视觉语言模型在任意分辨率下对世界的感知能力, 2024. 1, 2


[51] Yixiao Wang, Mingxiao Huo, Zhixuan Liang, Yushi Du, Lingfeng Sun, Haotian Lin, Jinghuan Shang, Chen-sheng Peng, Mohit Bansal, Mingyu Ding, and Masayoshi Tomizuka. Ver: Vision expert transformer for robot learning via foundation distillation and dynamic routing, 2025. 2
[51] Yixiao Wang, Mingxiao Huo, Zhixuan Liang, Yushi Du, Lingfeng Sun, Haotian Lin, Jinghuan Shang, Chen-sheng Peng, Mohit Bansal, Mingyu Ding, 和 Masayoshi Tomizuka. Ver：通过基础蒸馏和动态路由实现机器人学习的视觉专家 Transformer, 2025. 2


[52] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xi-angyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, Min Dou, Kai Chen, Wenhai Wang, Yu Qiao, Yali Wang, and Limin Wang. Internvideo2.5: Empowering video mllms with long and rich context modeling, 2025. 3, 5, 15
[52] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xi-angyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, Min Dou, Kai Chen, Wenhai Wang, Yu Qiao, Yali Wang, and Limin Wang. Internvideo2.5: 通过长且丰富的上下文建模赋能视频多模态大模型, 2025. 3, 5, 15


[53] Yating Wang, Haoyi Zhu, Mingyu Liu, Jiange Yang, Hao-Shu Fang, and Tong He. Vq-vla: Improving vision-language-action models via scaling vector-quantized action tokenizers, 2025. 3
[53] Yating Wang, Haoyi Zhu, Mingyu Liu, Jiange Yang, Hao-Shu Fang, and Tong He. Vq-vla: 通过缩放向量量化动作分词器改进视觉-语言-动作模型, 2025. 3


[54] Junjie Wen, Minjie Zhu, Yichen Zhu, Zhibin Tang, Jinming Li, Zhongyi Zhou, Chengmeng Li, Xiaoyu Liu, Yaxin Peng, Chaomin Shen, and Feifei Feng. Diffusion-vla: Generalizable and interpretable robot foundation model via self-generated reasoning, 2025. 3
[54] Junjie Wen, Minjie Zhu, Yichen Zhu, Zhibin Tang, Jinming Li, Zhongyi Zhou, Chengmeng Li, Xiaoyu Liu, Yaxin Peng, Chaomin Shen, and Feifei Feng. Diffusion-vla: 基于自生成推理的通用且可解释的机器人基础模型, 2025. 3


[55] Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, and Feifei Feng. Dexvla: Vision-language model with plug-in diffusion expert for general robot control, 2025. 2
[55] Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, and Feifei Feng. Dexvla: 带有插件式扩散专家用于通用机器人控制的视觉-语言模型, 2025. 2


[56] Philipp Wu, Yide Shentu, Zhongke Yi, Xingyu Lin, and Pieter Abbeel. Gello: A general, low-cost, and intuitive tele-operation framework for robot manipulators, 2024. 15
[56] Philipp Wu, Yide Shentu, Zhongke Yi, Xingyu Lin, and Pieter Abbeel. Gello: 一种用于机器人操纵器的通用、低成本且直观的遥操作框架, 2024. 15


[57] Lu Xu, Jiaqian Yu, Xiongfeng Peng, Yiwei Chen, Weiming Li, Jaewook Yoo, Sunghyun Chunag, Dongwook Lee, Dae-hyun Ji, and Chao Zhang. Mose: Skill-by-skill mixture-of-experts learning for embodied autonomous machines, 2025. 2
[57] Lu Xu, Jiaqian Yu, Xiongfeng Peng, Yiwei Chen, Weiming Li, Jaewook Yoo, Sunghyun Chunag, Dongwook Lee, Dae-hyun Ji, and Chao Zhang. Mose: 面向具身自主机器的技能级专家混合学习, 2025. 2


[58] Jiange Yang, Haoyi Zhu, Yating Wang, Gangshan Wu, Tong He, and Limin Wang. Tra-moe: Learning trajectory prediction model from multiple domains for adaptive policy conditioning, 2025. 2
[58] Jiange Yang, Haoyi Zhu, Yating Wang, Gangshan Wu, Tong He, and Limin Wang. Tra-moe: 从多领域学习轨迹预测模型以实现自适应策略调节, 2025. 2


[59] Shuai Yang, Hao Li, Yilun Chen, Bin Wang, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, and Jiangmiao Pang. Instructvla: Vision-language-action instruction tuning from understanding to manipulation, 2025. 1, 2
[59] Shuai Yang, Hao Li, Yilun Chen, Bin Wang, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, and Jiangmiao Pang. Instructvla: 从理解到操纵的视觉-语言-动作指令微调, 2025. 1, 2


[60] Yi Yang, Jiaxuan Sun, Siqi Kou, Yihan Wang, and Zhijie Deng. Lohovla: A unified vision-language-action model for long-horizon embodied tasks, 2025. 1, 2
[60] Yi Yang, Jiaxuan Sun, Siqi Kou, Yihan Wang, and Zhijie Deng. Lohovla: 用于长程具身任务的统一视觉-语言-动作模型, 2025. 1, 2


[61] Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, and Junchi Yan. Drivemoe: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving, 2025. 2
[61] Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, and Junchi Yan. Drivemoe: 端到端自动驾驶中视觉-语言-动作模型的专家混合架构, 2025. 2


[62] Zhutian Yang, Caelan Garrett, Dieter Fox, Tomás Lozano-Pérez, and Leslie Pack Kaelbling. Guiding long-horizon task and motion planning with vision language models. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 16847-16853. IEEE, 2025. 2
[62] Zhutian Yang, Caelan Garrett, Dieter Fox, Tomás Lozano-Pérez, and Leslie Pack Kaelbling. 利用视觉语言模型引导长程任务与运动规划. 见：2025 IEEE 国际机器人与自动化会议 (ICRA), 第 16847-16853 页. IEEE, 2025. 2


[63] Yuanqi Yao, Siao Liu, Haoming Song, Delin Qu, Qizhi Chen, Yan Ding, Bin Zhao, Zhigang Wang, Xuelong Li, and Dong Wang. Think small, act big: Primitive prompt learning for lifelong robot manipulation, 2025. 3
[63] Yuanqi Yao, Siao Liu, Haoming Song, Delin Qu, Qizhi Chen, Yan Ding, Bin Zhao, Zhigang Wang, Xuelong Li, and Dong Wang. Think small, act big: 面向终身机器人操纵的基元提示学习, 2025. 3


[64] Jiawen Yu, Hairuo Liu, Qiaojun Yu, Jieji Ren, Ce Hao, Haitong Ding, Guangyu Huang, Guofan Huang, Yan Song, Panpan Cai, Cewu Lu, and Wenqiang Zhang. Forcevla: Enhancing vla models with a force-aware moe for contact-rich manipulation, 2025. 2
[64] Jiawen Yu, Hairuo Liu, Qiaojun Yu, Jieji Ren, Ce Hao, Haitong Ding, Guangyu Huang, Guofan Huang, Yan Song, Panpan Cai, Cewu Lu, and Wenqiang Zhang. Forcevla: 通过力感知专家混合模型增强接触丰富操纵任务中的 VLA 模型, 2025. 2


[65] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xi-ang, Gordon Wetzstein, and Tsung-Yi Lin. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models, 2025. 6
[65] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xi-ang, Gordon Wetzstein, and Tsung-Yi Lin. Cot-vla: 视觉-语言-动作模型的视觉思维链推理, 2025. 6


[66] Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, Ya-Qin Zhang, and Xianyuan Zhan. Universal actions for enhanced embodied foundation models, 2025. 2
[66] Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, Ya-Qin Zhang, and Xianyuan Zhan. Universal actions for enhanced embodied foundation models, 2025. 2


[67] Zhehua Zhou, Jiayang Song, Kunpeng Yao, Zhan Shu, and Lei Ma. Isr-llm: Iterative self-refined large language model for long-horizon sequential task planning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 2081-2088. IEEE, 2024. 2
[67] Zhehua Zhou, Jiayang Song, Kunpeng Yao, Zhan Shu, and Lei Ma. Isr-llm: Iterative self-refined large language model for long-horizon sequential task planning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 2081-2088. IEEE, 2024. 2


[68] Zhongyi Zhou, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu. Chatvla-2: Vision-language-action model with open-world embodied reasoning from pretrained knowledge, 2025. 2
[68] Zhongyi Zhou, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu. Chatvla-2: Vision-language-action model with open-world embodied reasoning from pretrained knowledge, 2025. 2


# AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots
# AtomicVLA：释放机器人原子技能学习的潜力


Supplementary Material
补充材料


## Contents
## 目录


A.1. Video Demonstration 13
A.1. 视频演示 13


A.2. Future Work and Limitations 13
A.2. 未来工作与局限性 13


A.3. Additional Details 13
A.3. 补充细节 13


A.3.1. Training Setup 13
A.3.1. 训练设置 13


A.3.2. Simulations Setting 13
A.3.2. 仿真环境设置 13


A.3.3. Real-world Setting 15
A.3.3. 真实世界设置 15


A.3.4. Continual Learning Setting 15
A.3.4. 持续学习设置 15


A.3.5. Data Generation Setting 15
A.3.5. 数据生成设置 15


A.4. Additional Results 15
A.4. 补充结果 15


A.5. Additional Visualizations 17
A.5. 补充可视化 17


### A.1. Video Demonstration
### A.1. 视频演示


Please refer to the video file in the attachment for a quick overview of AtomicVLA.
请参阅附件中的视频文件，以快速了解 AtomicVLA。


### A.2. Future Work and Limitations
### A.2. 未来工作与局限性


Most current vision-language-action (VLA) models are typically trained and evaluated on individual tasks. In this work, we investigate skill interference arising from multi-skill joint training through controlled experiments and introduce a Skill-Gated Mixture-of-Experts (SG-MoE) framework to construct a scalable atomic skill library, thereby exploring the potential of VLA models in long-horizon tasks and continual learning. Although this paradigm shows clear promise, many advantages remain insufficiently explored.
目前大多数视觉-语言-动作（VLA）模型通常针对单一任务进行训练和评估。本研究通过对照实验探讨了多技能联合训练中产生的技能干扰，并引入了技能门控专家混合（SG-MoE）框架来构建可扩展的原子技能库，从而探索了 VLA 模型在长程任务和持续学习中的潜力。尽管该范式展现出明显的前景，但其许多优势仍有待深入挖掘。


- AtomicVLA relies on a task planning module that produces accurate atomic skill abstractions and on a set of well trained skill experts. The skill router relies on the VLM to produce accurate atomic skill abstractions during task execution, a capability constrained by the VLM's reasoning and planning fidelity. Recent studies such as Embodied Brain [18,44,45] and ${\pi }_{0.5}$ [17] indicate that combining large scale web data with embodied experience can effectively train VLMs that are capable of skill decomposition and task planning while enabling the construction of a high quality expert skill library, which can further enhance the performance of AtomicVLA.
- AtomicVLA 依赖于一个能生成精确原子技能抽象的任务规划模块，以及一组训练有素的技能专家。技能路由器的表现取决于 VLM 在任务执行期间生成精确原子技能抽象的能力，而这一能力受限于 VLM 的推理和规划保真度。近期研究如 Embodied Brain [18,44,45] 和 ${\pi }_{0.5}$ [17] 表明，将大规模网络数据与具身经验相结合，可以有效训练出具备技能分解和任务规划能力的 VLM，同时支持构建高质量的专家技能库，从而进一步提升 AtomicVLA 的性能。


- By decoupling skill learning, AtomicVLA substantially mitigates interference during multi-skill training and demonstrates strong adaptability to new skills. However, acquiring new tasks still requires collecting substantial human demonstration data for imitation learning(IL). Notably,recent works like ${\pi }_{0.6}^{ * }$ [16],SimpleVLA-RL [25] and VLA-RL [31] have shown that reinforcement learning(RL) can effectively train VLA models and achieve strong performance. Integrating a pre-trained skill expert library with reinforcement learning(RL) may empower AtomicVLA to generalize to novel tasks under few-shot or even zero-shot settings.
- 通过解耦技能学习，AtomicVLA 大幅减轻了多技能训练过程中的干扰，并展现出对新技能的强大适应性。然而，获取新任务仍需收集大量人类演示数据以进行模仿学习（IL）。值得注意的是，近期研究如 ${\pi }_{0.6}^{ * }$ [16]、SimpleVLA-RL [25] 和 VLA-RL [31] 表明，强化学习（RL）能够有效训练 VLA 模型并实现优异性能。将预训练的技能专家库与强化学习（RL）相结合，有望使 AtomicVLA 在少样本甚至零样本设置下泛化至新任务。


Table 6. Atomic skill distribution in the LIBERO dataset.
表 6. LIBERO 数据集中的原子技能分布。


<table><tr><td>Atomic Skill</td><td>Count</td></tr><tr><td>Pick</td><td>2462</td></tr><tr><td>Place</td><td>761</td></tr><tr><td>Open</td><td>201</td></tr><tr><td>Close</td><td>152</td></tr><tr><td>Turn</td><td>175</td></tr></table>
<table><tbody><tr><td>原子技能</td><td>计数</td></tr><tr><td>拾取</td><td>2462</td></tr><tr><td>放置</td><td>761</td></tr><tr><td>打开</td><td>201</td></tr><tr><td>关闭</td><td>152</td></tr><tr><td>旋转</td><td>175</td></tr></tbody></table>


### A.3. Additional Details
### A.3. 补充细节


#### A.3.1. Training Setup
#### A.3.1. 训练设置


For all experiments, we construct the skill library using one shared expert together with multiple skill experts. Each skill expert follows the Gemma architecture, where the feedfor-ward module is implemented with an independent SwiGLU activated MLP. All skill experts are randomly initialized at the beginning of training to enable disentangled skill representations and support incremental learning. The model configuration is width $= {2048}$ ,mlp $\dim  = {4096}$ ,depth $= {18}$ , num heads $= 8$ ,and head $\dim  = {256}$ . Building on this configuration, the learning rate follows CosineDecaySchedul with a warm up phase of 1,000 steps, a peak learning rate of ${2.5} \times  {10}^{-5}$ ,and a final learning rate of $5 \times  {10}^{-6}$ . The optimizer is AdamW with a gradient clipping norm = 1.0.To stabilize training, an exponential moving average (decay = 0.999) is used throughout optimization.
在所有实验中，我们通过一个共享专家和多个技能专家构建技能库。每个技能专家均采用 Gemma 架构，其中前馈模块由独立的 SwiGLU 激活 MLP 实现。所有技能专家在训练开始时均进行随机初始化，以实现解耦的技能表征并支持增量学习。模型配置为宽度 $= {2048}$，mlp $\dim  = {4096}$，深度 $= {18}$，注意力头数 $= 8$，以及头维度 $\dim  = {256}$。在此配置基础上，学习率采用 CosineDecaySchedul，包含 1,000 步预热阶段，峰值学习率为 ${2.5} \times  {10}^{-5}$，最终学习率为 $5 \times  {10}^{-6}$。优化器使用 AdamW，梯度裁剪范数为 1.0。为稳定训练，整个优化过程中均使用指数移动平均（衰减率 = 0.999）。


Following this setup, we train the model for 100k iterations on both the LIBERO and Calvin simulation platforms, and for 30k iterations in real world robotic experiments, with a batch size of 64. All training is performed on $8 \times$ H200 GPUs,and inference is conducted on a single NVIDIA RTX RPO6000 GPU.
基于此设置，我们在 LIBERO 和 Calvin 仿真平台上对模型进行了 100k 次迭代训练，在真实机器人实验中进行了 30k 次迭代训练，批次大小均为 64。所有训练均在 $8 \times$ 张 H200 GPU 上完成，推理则在单张 NVIDIA RTX RPO6000 GPU 上进行。


#### A.3.2. Simulations Setting
#### A.3.2. 仿真设置


LIBERO Setting. We use the public dataset provided by LIBERO and convert it into the Lerobot format for all experiments. Following the data processing method introduced in Sec. 3.5, we perform fine grained annotation and organize the collected data into five atomic action abstractions: Pick, Place, Open, Close, and Turn. The data distribution for these action categories is presented in Tab. 6. All skills are trained in a mixed manner, and therefore maintaining balanced data becomes essential. To achieve this, we increase the sampling frequency of the less represented actions, specifically Open, Close, and Turn, in order to equalize the data distribution and prevent insufficient training of the corresponding skill experts. For a fair comparison, AtomicVLA is consistent with the evaluation of the baseline method, testing each task 50 times and reporting the average results.
LIBERO 设置。我们使用 LIBERO 提供的公开数据集，并将其转换为 Lerobot 格式进行所有实验。遵循第 3.5 节介绍的数据处理方法，我们执行细粒度标注，并将收集的数据组织为五种原子动作抽象：抓取（Pick）、放置（Place）、打开（Open）、关闭（Close）和旋转（Turn）。这些动作类别的数据分布如表 6 所示。所有技能均以混合方式训练，因此保持数据平衡至关重要。为此，我们增加了样本较少动作（即打开、关闭和旋转）的采样频率，以均衡数据分布，防止相应技能专家训练不足。为确保公平比较，AtomicVLA 与基准方法的评估保持一致，每个任务测试 50 次并报告平均结果。


Role:
角色：


You are an expert in robotics data analysis. You are analyzing a video clip of a robot performing a task, based on given task instructions. The clip was detected based on the robot's movement patterns and segmented into basic skill segments. Your goal is to determine the task progress of the current segment and identify the specific atomic actions.
你是一位机器人数据分析专家。你需要根据给定的任务指令，分析机器人执行任务的视频片段。该片段是根据机器人的运动模式检测并分割为基础技能片段的。你的目标是确定当前片段的任务进度，并识别具体的原子动作。


Input:
输入：


1. The complete task instructions and coarse labels for the video clip.
1. 完整的任务指令及视频片段的粗略标签。


2. Image frames from a video clip (sampled every three frames)
2. 视频片段的图像帧（每三帧采样一次）。


Instruction: Turn on the stove and put the moka pot on it Coarse label: Turn
指令：打开炉灶并将摩卡壶放上去 粗略标签：旋转


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_f9175d.jpg"/>



Task:
任务：


Your task is to provide a complete task chain based on the task instructions, and analyze the current task progress and corresponding atomic tasks and actions based on the coarse labels and video content.
你的任务是根据任务指令提供完整的任务链，并根据粗略标签和视频内容分析当前任务进度及相应的原子任务和动作。


Thought process and examples :
思考过程与示例：


For task chain, it is a list of multiple atomic tasks.
任务链即多个原子任务的列表。


For each atomic task, the formatting and constraints is:
对于每个原子任务，格式与约束如下：


1. Output one imperative sentence starting with an action verb.
1. 输出一个以动作动词开头的祈使句。


2. Use exactly one verb from this set: [Pick/Place/Turn/Open/Close/Push/Pull/Adjust].
2. 必须且仅能使用以下集合中的一个动词：[Pick/Place/Turn/Open/Close/Push/Pull/Adjust]。


3. Focus on the final positions of the manipulated objects to avoid errors or repeated identification of atomic actions.
3. 聚焦于被操作物体的最终位置，以避免错误或重复识别原子动作。


4. Sentence length limit: no more than 15 words.
4. 句子长度限制：不超过15个单词。


5. Specify the manipulated object and key attributes (color, category, location/support surface/container).
5. 指明被操作物体及其关键属性（颜色、类别、位置/支撑面/容器）。


6. For Place/Move, specify the destination or target container.
6. 对于Place/Move（放置/移动），需指明目的地或目标容器。


7. Describe only one atomic action and the final action (no multi-step sequences, no plans or intentions).
7. 仅描述一个原子动作及最终动作（不包含多步序列，不包含计划或意图）。


8. If this is final step(N/N), considering the completeness of the task, please avoid erroneous judgments such as "open" or "pick."
8. 若为最后一步(N/N)，考虑到任务的完整性，请避免“打开”或“拾取”等错误判断。


For atomic action, it must be one verb from this set: [Pick/Place/Turn/Open/Close/Push/Pull/Adjust].
对于原子动作，必须是以下集合中的一个动词：[Pick/Place/Turn/Open/Close/Push/Pull/Adjust]。


Examples:
示例：


1. The task chain is [pick up the yellow cup, place the yellow cup in microwave, close the microwave], This is step 1/3, pick up the yellow cup, and the atomic abstraction is pick.
1. 任务链为[拾取黄色杯子，将黄色杯子放入微波炉，关闭微波炉]，这是第1/3步，拾取黄色杯子，原子抽象为pick。


2. The task chain is [pick up the butter, place the butter in the basket], This is step 2/2, place the butter in the basket, and the atomic abstraction is place.
2. 任务链为[拾取黄油，将黄油放入篮子]，这是第2/2步，将黄油放入篮子，原子抽象为place。


3. The task chain is [open the top drawer, pick up the block, place the block into the top drawer], This is step 2/3, open the top drawer, and the atomic abstraction is open.
3. 任务链为[打开顶部抽屉，拾取积木，将积木放入顶部抽屉]，这是第2/3步，打开顶部抽屉，原子抽象为open。


Instructions:
指令：


Based on the text and video information above, please provide the task chain for this task, as well as the task progress and atomic actions corresponding to the video clip. Your judgment should be as detailed and accurate as possible, with reasoning supported by the video clip and task instructions. If the coarse label is incorrect, ignore it and provide the correct label.
基于上述文本和视频信息，请提供该任务的任务链，以及与视频片段对应的任务进度和原子动作。你的判断应尽可能详细准确，并由视频片段和任务说明提供推理支持。若粗略标签不正确，请忽略它并提供正确的标签。


Output Format:
输出格式：


The task chain is <the list of atomic tasks>, This is step x/N, <current atomic task>, and the atomic abstraction is <choose one atomic action>.
任务链为 <原子任务列表>，当前为第 x/N 步，<当前原子任务>，原子抽象为 <选择一个原子动作>。


The task chain is [turn on the stone, pick up the moka pot, place the moka pot on the stone], This is step 1/3, turn on the stone, and the atomic abstraction is turn.
任务链为 [打开炉灶，拿起摩卡壶，将摩卡壶放在炉灶上]，当前为第 1/3 步，打开炉灶，原子抽象为旋转。


Figure 7. The prompts and examples of the InternVideo2.5.
图 7. InternVideo2.5 的提示词与示例。


Calvin Setting. We use the task ABC-D public dataset provided by Calvin and divide the data according to the instruction annotations and the corresponding frame intervals. Each trajectory is capped at 64 frames and is converted into the Lerobot format for our experiments. Following the data processing method introduced in Sec.3.5, we perform fine grained annotation and organize the data into eight atomic action abstractions: Rotate, Push, Move, Open&Close, Lift, Place, Turn, and Stack. Based on these categories, we construct a skill expert library consisting of 8 skill experts. Building on this configuration, we ensure fair comparison by keeping the AtomicVLA evaluation protocol consistent with that of the baseline methods. In this setting, the robot executes 1,000 task sequences, and each sequence contains five consecutive tasks. We report the average success rates together with the average length of the completed sequences.
Calvin 设置。我们使用 Calvin 提供的 ABC-D 公共数据集，并根据指令标注及相应的帧间隔对数据进行划分。每条轨迹上限为 64 帧，并转换为 Lerobot 格式以供实验使用。遵循第 3.5 节介绍的数据处理方法，我们进行细粒度标注，并将数据整理为八种原子动作抽象：旋转 (Rotate)、推动 (Push)、移动 (Move)、开合 (Open&Close)、抬起 (Lift)、放置 (Place)、转向 (Turn) 和堆叠 (Stack)。基于这些类别，我们构建了一个包含 8 个技能专家的技能专家库。在此配置基础上，我们通过保持 AtomicVLA 的评估协议与基线方法一致，确保了比较的公平性。在此设置下，机器人执行 1,000 个任务序列，每个序列包含五个连续任务。我们报告平均成功率以及已完成序列的平均长度。


Table 7. List of tasks and prompts used in our real-world experiments.
表 7. 我们真实世界实验中使用的任务与提示词列表。


<table><tr><td>Task Type</td><td>Task Prompt</td></tr><tr><td colspan="2">Long-horizon Tasks</td></tr><tr><td>Objects in plate</td><td>Place all blocks on the table into a green plate.</td></tr><tr><td>Object into drawer</td><td>Open the top drawer and place the block inside.</td></tr><tr><td>Object into microwave</td><td>Place the plate into the microwave and close the door.</td></tr><tr><td colspan="2">Short Tasks</td></tr><tr><td>Grasp</td><td>Grasp the block from the table.</td></tr><tr><td>Stack</td><td>Stack the red block on the orange block.</td></tr><tr><td>Close</td><td>Close the microwave on the table.</td></tr><tr><td>Press</td><td>Press the button on the table.</td></tr><tr><td>Open</td><td>Open the top drawer.</td></tr><tr><td colspan="2">Complex Scenes</td></tr><tr><td>Objects in plate</td><td>Put the pepper and corn into the green plate.</td></tr><tr><td>Objects in plate</td><td>Put the carrot and cucumber into the green plate.</td></tr><tr><td>Objects in plate</td><td>Put the potato and eggplant into the green plate.</td></tr></table>
<table><tbody><tr><td>任务类型</td><td>任务提示</td></tr><tr><td colspan="2">长程任务</td></tr><tr><td>盘中物体</td><td>将桌上所有积木放入绿色盘子中。</td></tr><tr><td>物体放入抽屉</td><td>打开顶层抽屉并将积木放入其中。</td></tr><tr><td>物体放入微波炉</td><td>将盘子放入微波炉并关上门。</td></tr><tr><td colspan="2">短程任务</td></tr><tr><td>抓取</td><td>从桌上抓取积木。</td></tr><tr><td>堆叠</td><td>将红色积木堆叠在橙色积木上。</td></tr><tr><td>关闭</td><td>关闭桌上的微波炉。</td></tr><tr><td>按压</td><td>按下桌上的按钮。</td></tr><tr><td>打开</td><td>打开顶层抽屉。</td></tr><tr><td colspan="2">复杂场景</td></tr><tr><td>盘中物体</td><td>将胡椒和玉米放入绿色盘子中。</td></tr><tr><td>盘中物体</td><td>将胡萝卜和黄瓜放入绿色盘子中。</td></tr><tr><td>盘中物体</td><td>将土豆和茄子放入绿色盘子中。</td></tr></tbody></table>


Table 8. Results on Complex Scenes.
表8. 复杂场景下的实验结果。


<table><tr><td>Method</td><td>Pepper/Corn</td><td>Carrot/Cucumber</td><td>Potatoe/Eggplant</td><td>Avg.</td></tr><tr><td>${\pi }_{0.5}$ [17]</td><td>25</td><td>40</td><td>35</td><td>33.3</td></tr><tr><td>AtomicVLA*</td><td>40</td><td>45</td><td>45</td><td>43.3</td></tr></table>
<table><tbody><tr><td>方法</td><td>青椒/玉米</td><td>胡萝卜/黄瓜</td><td>土豆/茄子</td><td>平均值</td></tr><tr><td>${\pi }_{0.5}$ [17]</td><td>25</td><td>40</td><td>35</td><td>33.3</td></tr><tr><td>AtomicVLA*</td><td>40</td><td>45</td><td>45</td><td>43.3</td></tr></tbody></table>


#### A.3.3. Real-world Setting
#### A.3.3. 真实世界设置


Hardware. Our real-world experimental setup consists of a Franka Research3 robotic arm with two Realsense D435i cameras: one mounted on the wrist to provide a first-person perspective, and the other positioned opposite the robotic arm to offer a third-person view.
硬件。我们的真实世界实验装置包括一台 Franka Research3 机械臂和两台 Realsense D435i 相机：一台安装在手腕上提供第一人称视角，另一台放置在机械臂对面提供第三人称视角。


Evaluation Tasks. In the real world, we collected three long-horizon tasks and five short tasks, and additionally gathered three long-horizon tasks in more complex scenarios to evaluate the performance of AtomicVLA. We employed Gello [56] to control the Franka arm and record demonstration data. We collected 100 trajectories per long-horizon task and 50 per short task. The results reported in this paper were obtained using a multi-task mixed training protocol. Each task was evaluated 20 times with randomized object placements, and the average performance across these trials was reported as the final test result. The full list of tasks is presented in Tab. 7.
评估任务。在真实世界中，我们收集了三个长程任务和五个短程任务，并额外在更复杂的场景中收集了三个长程任务以评估 AtomicVLA 的性能。我们使用 Gello [56] 控制 Franka 机械臂并记录演示数据。每个长程任务我们收集了 100 条轨迹，每个短程任务收集了 50 条。本文报告的结果均采用多任务混合训练协议获得。每个任务在随机放置物体的情况下进行 20 次评估，并以这些试验的平均表现作为最终测试结果。任务完整列表见表 7。


#### A.3.4. Continual Learning Setting
#### A.3.4. 持续学习设置


We conducted experiments on continual learning for short tasks. Specifically, we used four tasks for mixed training, iterating for ${20}\mathrm{k}$ steps. Then,we applied "open the top drawer" as a new skill for continual learning, fine-tuning on the weights learned from the four tasks. We used a learning rate of $5 \times  {10}^{-6}$ and iterated for $7\mathrm{k}$ steps,and reported the results by averaging over 20 validation runs for each of the five tasks.
我们针对短程任务进行了持续学习实验。具体而言，我们使用四个任务进行混合训练，迭代 ${20}\mathrm{k}$ 步。随后，我们将“打开顶部抽屉”作为一项新技能进行持续学习，在四个任务学到的权重基础上进行微调。我们使用 $5 \times  {10}^{-6}$ 的学习率并迭代 $7\mathrm{k}$ 步，通过对五个任务各进行 20 次验证运行取平均值来报告结果。


#### A.3.5. Data Generation Setting
#### A.3.5. 数据生成设置


We use principal component analysis to obtain precise video segmentation and coarse labels. By analyzing the motion changes across five consecutive frames, we determine the dominant motion axis. Specifically, the threshold for the translation axis(Δ $x,$ Δ $y,$ Δ $z$ ) is set to 3 cm,the threshold for the rotation axis(Δroll, Δpitch, Δyaw) is set to 0.05 radians, and the gripper change( ΔGrip) threshold is set to 0.1.
我们使用主成分分析法获取精确的视频分割和粗略标签。通过分析连续五帧的运动变化，我们确定主运动轴。具体来说，平移轴（Δ $x,$ Δ $y,$ Δ $z$ ）的阈值设为 3 厘米，旋转轴（Δroll, Δpitch, Δyaw）的阈值设为 0.05 弧度，夹爪变化（ΔGrip）阈值设为 0.1。


In Fig. 7, we provide detailed prompts and examples for VLM (InternVideo2.5 [52]). The VLM analyzes video clips and generates task chains, task progress, and atomic actions based on the input text instructions.
在图 7 中，我们提供了 VLM（InternVideo2.5 [52]）的详细提示词和示例。VLM 根据输入的文本指令分析视频片段，并生成任务链、任务进度和原子动作。


### A.4. Additional Results
### A.4. 补充结果


Detail Results on Calvin. As shown in Tab. 9, we report the performance of AtomicVLA* on the 34 evaluation tasks of the Calvin ABC-D dataset. The results indicate that the model achieves success rates close to 100 percent on most tasks. However, performance on several Push blocks right tasks is considerably lower, with average success rates only between 20 and 30 percent. Building on this observation, we find that in the training set the relevant blocks are typically placed near the center of the table. In contrast, during evaluation the blocks are often positioned on the right side of the table. This distribution shift leads the model to push the block in the correct direction while failing to push it far enough to satisfy the success criterion, which results in task failure and prevents the execution of subsequent steps.
Calvin 上的详细结果。如表 9 所示，我们报告了 AtomicVLA* 在 Calvin ABC-D 数据集 34 个评估任务上的表现。结果表明，该模型在大多数任务上达到了接近 100% 的成功率。然而，在几个“向右推方块”任务上的表现明显较低，平均成功率仅在 20% 到 30% 之间。基于这一观察，我们发现训练集中相关方块通常放置在桌面中心附近。相比之下，在评估过程中，方块往往位于桌面的右侧。这种分布偏移导致模型虽然能向正确的方向推动方块，但推的距离不足以满足成功标准，从而导致任务失败，并阻碍了后续步骤的执行。


Table 9. Success rates for all evaluated tasks on CALVIN ABC-D dataset.
表 9. CALVIN ABC-D 数据集上所有评估任务的成功率。


<table><tr><td>Task Name</td><td>SR (%)</td><td>Task Name</td><td>SR (%)</td><td>Task Name</td><td>SR (%)</td></tr><tr><td>rotate blue block right</td><td>97.4</td><td>lift red block table</td><td>99.4</td><td>lift blue block table</td><td>99.4</td></tr><tr><td>move slider right</td><td>100.0</td><td>lift pink block table</td><td>94.5</td><td>place in drawer</td><td>100.0</td></tr><tr><td>lift red block slider</td><td>99.3</td><td>move slider left</td><td>100.0</td><td>rotate red block left</td><td>98.5</td></tr><tr><td>place in slider</td><td>98.6</td><td>turn on lightbulb</td><td>100.0</td><td>push pink block left</td><td>93.5</td></tr><tr><td>turn off lightbulb</td><td>100.0</td><td>rotate blue block left</td><td>100.0</td><td>lift blue block slider</td><td>95.6</td></tr><tr><td>turn off led</td><td>98.8</td><td>push blue block left</td><td>94.2</td><td>lift pink block drawer</td><td>100.0</td></tr><tr><td>push into drawer</td><td>86.0</td><td>turn on led</td><td>100.0</td><td>rotate pink block right</td><td>98.6</td></tr><tr><td>lift blue block drawer</td><td>100.0</td><td>stack block</td><td>98.4</td><td>unstack block</td><td>98.6</td></tr><tr><td>close drawer</td><td>100.0</td><td>push pink block right</td><td>33.8</td><td>push blue block right</td><td>22.2</td></tr><tr><td>lift pink block slider</td><td>97.8</td><td>push red block right</td><td>29.2</td><td>rotate pink block left</td><td>100.0</td></tr><tr><td>open drawer</td><td>100.0</td><td>push red block left</td><td>89.9</td><td>lift red block drawer</td><td>100.0</td></tr><tr><td>rotate red block right</td><td>97.3</td><td></td><td></td><td></td><td></td></tr></table>
<table><tbody><tr><td>任务名称</td><td>成功率 (%)</td><td>任务名称</td><td>成功率 (%)</td><td>任务名称</td><td>成功率 (%)</td></tr><tr><td>向右旋转蓝色方块</td><td>97.4</td><td>从桌面提起红色方块</td><td>99.4</td><td>从桌面提起蓝色方块</td><td>99.4</td></tr><tr><td>向右移动滑块</td><td>100.0</td><td>从桌面提起粉色方块</td><td>94.5</td><td>放入抽屉</td><td>100.0</td></tr><tr><td>从滑块提起红色方块</td><td>99.3</td><td>向左移动滑块</td><td>100.0</td><td>向左旋转红色方块</td><td>98.5</td></tr><tr><td>放入滑块</td><td>98.6</td><td>打开灯泡</td><td>100.0</td><td>向左推动粉色方块</td><td>93.5</td></tr><tr><td>关闭灯泡</td><td>100.0</td><td>向左旋转蓝色方块</td><td>100.0</td><td>从滑块提起蓝色方块</td><td>95.6</td></tr><tr><td>关闭LED灯</td><td>98.8</td><td>向左推动蓝色方块</td><td>94.2</td><td>从抽屉提起粉色方块</td><td>100.0</td></tr><tr><td>推入抽屉</td><td>86.0</td><td>打开LED灯</td><td>100.0</td><td>向右旋转粉色方块</td><td>98.6</td></tr><tr><td>从抽屉提起蓝色方块</td><td>100.0</td><td>堆叠方块</td><td>98.4</td><td>拆解方块</td><td>98.6</td></tr><tr><td>关闭抽屉</td><td>100.0</td><td>向右推动粉色方块</td><td>33.8</td><td>向右推动蓝色方块</td><td>22.2</td></tr><tr><td>从滑块提起粉色方块</td><td>97.8</td><td>向右推动红色方块</td><td>29.2</td><td>向左旋转粉色方块</td><td>100.0</td></tr><tr><td>打开抽屉</td><td>100.0</td><td>向左推动红色方块</td><td>89.9</td><td>从抽屉提起红色方块</td><td>100.0</td></tr><tr><td>向右旋转红色方块</td><td>97.3</td><td></td><td></td><td></td><td></td></tr></tbody></table>


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_c41868.jpg"/>



Figure 8. Demonstrations of LIBERO and Calvin experiments.
图 8. LIBERO 和 Calvin 实验演示。


Table 10. Parameter and inference-time.
表 10. 参数与推理时间。


<table><tr><td>Experts</td><td>${\pi }_{0}$</td><td>K=5</td><td>K=8</td><td>K=12</td></tr><tr><td>Params</td><td>3.24B</td><td>4.17B</td><td>4.81B</td><td>5.65B</td></tr><tr><td>Act</td><td>71ms</td><td>92 ms</td><td>126 ms</td><td>160 ms</td></tr><tr><td>Think</td><td>-</td><td>104 ms</td><td>104 ms</td><td>104 ms</td></tr></table>
<table><tbody><tr><td>专家</td><td>${\pi }_{0}$</td><td>K=5</td><td>K=8</td><td>K=12</td></tr><tr><td>参数</td><td>3.24B</td><td>4.17B</td><td>4.81B</td><td>5.65B</td></tr><tr><td>执行</td><td>71毫秒</td><td>92毫秒</td><td>126毫秒</td><td>160毫秒</td></tr><tr><td>思考</td><td>-</td><td>104毫秒</td><td>104毫秒</td><td>104毫秒</td></tr></tbody></table>


Results on Complex Scenes. As shown in Tab. 9, we report the performance of AtomicVLA* and ${\pi }_{0.5}$ on three additional real-world experiments designed to evaluate its ability to handle complex scenes and grasp irregular objects. AtomicVLA* achieved an average accuracy of 43.3%, which is ${10}\%$ higher than the ${\pi }_{0.5}$ average. In addition, when picking corn, due to the color being similar to the background of the table, AtomicVLA* was able to make multiple corrections as it approached the target, resulting in a 15% improvement.
复杂场景下的结果。如表9所示，我们报告了AtomicVLA*与${\pi }_{0.5}$在三个额外真实世界实验中的表现，旨在评估其处理复杂场景及抓取不规则物体的能力。AtomicVLA*实现了43.3%的平均准确率，比${\pi }_{0.5}$平均水平高出${10}\%$。此外，在抓取玉米时，由于其颜色与桌面背景相似，AtomicVLA*在接近目标过程中能够进行多次修正，从而提升了15%的性能。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_750281.jpg"/>



Figure 9. Error recovery cases in real-world experiments.
图9. 真实世界实验中的错误恢复案例。


Parameter and inference-time. Tab. 10 shows the parameter counts and inference-time on a single H20 GPU. Even with 12 experts, the inference latency is only 160 ms, which is fully practical for real-world use.
参数量与推理时间。表10展示了在单张H20 GPU上的参数量与推理时间。即使拥有12个专家模型，推理延迟也仅为160毫秒，完全满足真实世界的使用需求。


### A.5. Additional Visualizations
### A.5. 额外可视化


In Fig. 8, we present a comparison between AtomicVLA and ${\pi }_{0}$ across simulation environments. Representative task cases are selected from both LIBERO and Calvin. As shown, AtomicVLA successfully completes several task instances where pi0 fails, demonstrating its stronger robustness and execution reliability in simulated settings.
在图8中，我们展示了AtomicVLA与${\pi }_{0}$在仿真环境中的对比。我们从LIBERO和Calvin中选取了具有代表性的任务案例。如图所示，AtomicVLA成功完成了多个pi0失败的任务实例，证明了其在仿真设置中更强的鲁棒性和执行可靠性。


In Fig. 9, we further illustrate AtomicVLA's real-world error recovery capability. When a subtask fails during execution, AtomicVLA automatically replans and corrects its behavior to ensure successful completion of the overall task. Specifically, as highlighted in the red box in the figure, when execution errors occur, such as misgrasps due to inaccurate positioning or visual ambiguity between the target object and the background, AtomicVLA can assess the current task state, generate an updated task plan, and reattempt the failed subtask, thereby ensuring robust completion of the overall task.
在图9中，我们进一步展示了AtomicVLA的真实世界错误恢复能力。当子任务在执行过程中失败时，AtomicVLA会自动重新规划并修正其行为，以确保整体任务的成功完成。具体而言，如图中红框所示，当出现执行错误（例如因定位不准或目标物体与背景视觉模糊导致的抓取失败）时，AtomicVLA能够评估当前任务状态，生成更新后的任务计划，并重试失败的子任务，从而确保整体任务的稳健完成。


Additionally, we show more demonstrations of real-world experiments in Fig. 10. These experiments span a wide spectrum of scenarios, from simple to highly complex tasks and from regular to irregular objects. Across all settings, AtomicVLA consistently exhibits strong performance and robust generalization.
此外，我们在图10中展示了更多真实世界实验的演示。这些实验涵盖了从简单到高度复杂、从规则到不规则物体的广泛场景。在所有设置中，AtomicVLA始终表现出强大的性能和稳健的泛化能力。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_05_47_b688b3.jpg"/>



Figure 10. Demonstrations of real-world experiments(Long-horizon tasks).
图10. 真实世界实验演示（长程任务）。