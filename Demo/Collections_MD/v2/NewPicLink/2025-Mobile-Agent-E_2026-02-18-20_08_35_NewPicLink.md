# Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks
# 移动代理 E：用于复杂任务的自进化移动助手


Zhenhailong Wang * 1 Haiyang Xu ${}^{ * }$ Junyang Wang ${}^{2}$ Xi Zhang ${}^{2}$
Zhenhailong Wang * 1 Haiyang Xu ${}^{ * }$ Junyang Wang ${}^{2}$ Xi Zhang ${}^{2}$


Ming Yan ${}^{2}$ Ji Zhang ${}^{2}$ Fei Huang ${}^{2}$ Heng Ji ${}^{ * }$
Ming Yan ${}^{2}$ Ji Zhang ${}^{2}$ Fei Huang ${}^{2}$ Heng Ji ${}^{ * }$


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_9a48e9.jpg"/>



Figure 1. We propose Mobile-Agent-E, a novel hierarchical multi-agent mobile assistant that outperforms previous state-of-the-art approaches (Zhang et al., 2023; Wang et al., 2024b;a) on complex real-world tasks. Mobile-Agent-E disentangles high-level planning and low-level action decision with dedicated agents. Equipped with a newly introduced self-evolution module that learns general Tips and reusable Shortcuts from past experiences, Mobile-Agent-E demonstrates further improvements in both performance and efficiency.
图 1。我们提出 Mobile-Agent-E，一种新型分层多智能体移动助手，在复杂真实场景任务上优于先前的最新方法（Zhang et al., 2023; Wang et al., 2024b;a）。Mobile-Agent-E 将高层规划与低层行动决策解耦为专用智能体。通过新增的自进化模块从 past experiences 学习通用提示与可重复使用的快捷方式，Mobile-Agent-E 在性能与效率方面均显示出进一步提升。


## Abstract
## 摘要


Smartphones have become indispensable in modern life, yet navigating complex, multi-step tasks on mobile devices often remains frustrating and time-consuming. Recent advancements in large multimodal model (LMM)-based mobile agents have demonstrated the ability to perceive and act in mobile environments on behalf of users. However, current approaches face significant limitations: they fall short in addressing real-world human needs, struggle with reasoning-intensive and long-horizon tasks, and lack mechanisms to learn and improve from prior experiences. To overcome
智能手机在现代生活中已变得不可或缺，但在移动设备上完成复杂、需多步的任务往往仍然令人沮丧且耗时。基于大型多模态模型（LMM）的移动代理在感知和代表用户行动方面的能力已经有所体现。然而，现有方法仍面临显著局限：在满足现实世界的人类需求方面不足，难以处理需要推理和长远规划的任务，并且缺乏从以往经验中学习和改进的机制。为克服


these challenges, we introduce Mobile-Agent-E, a hierarchical multi-agent framework capable of self-evolution through past experience. By "hierarchical," we mean an explicit separation of high-level planning and low-level action execution. The framework comprises a Manager, responsible for devising overall plans by breaking down complex tasks into subgoals, and four subordinate agents—Perceptor, Operator, Action Reflector, and Notetaker-which handle fine-grained visual perception, immediate action execution, error verification, and information aggregation, respectively. Mobile-Agent-E also features a novel self-evolution module which maintains a persistent long-term memory comprising Tips and Shortcuts. Tips are general guidance and lessons learned from prior tasks on how to effectively interact with the environment. Shortcuts are reusable, executable sequences of atomic operations tailored for specific subroutines. The inclusion of Tips and Shortcuts facilitates continuous refinement of task performance and efficiency. Alongside this framework, we introduce Mobile-Eval-E, a new benchmark featuring complex real-world mobile tasks requiring long-horizon, multi-app interactions. Empirical results show that Mobile-Agent-E achieves a 22% absolute improvement over previous state-of-the-art approaches across three foundation model backbones. Additionally, we provide a comprehensive analysis of the impact of our self-evolution mechanism and suggest directions for future work. Code and data are publicly available for research purposes at https: //x-plug.github.io/MobileAgent.
在这些挑战中，我们提出 Mobile-Agent-E，这是一个通过 past experience 自我进化的分层多智能体框架。所谓“分层”，指的是高层规划与低层执行之间的明确分离。该框架由一个 Manager 负责通过将复杂任务分解为子目标来制定总体计划，以及四个下属智能体——Perceptor、Operator、Action Reflector 和 Notetaker——分别处理细粒度的视觉感知、即时行动执行、错误验证和信息聚合。Mobile-Agent-E 还具备一个新颖的自我进化模块，维持包含提示与捷径的持久长期记忆。提示是对如何有效与环境交互的通用指导与以往任务中的经验教训。捷径是为特定子程序定制的、可重复使用的原子操作序列。提示与捷径的引入促进了任务性能与效率的持续改进。作为本框架的一部分，我们还引入 Mobile-Eval-E，一个新基准，包含需要长时程、多应用交互的复杂真实世界移动任务。实证结果显示，Mobile-Agent-E 对前一代最先进方法在三种基础模型骨架上实现了 22% 的绝对提升。此外，我们提供对自我进化机制影响的全面分析，并提出未来工作的方向。代码与数据出于研究目的公开可用，网址为 https: //x-plug.github.io/MobileAgent。


---



${}^{1}$ University of Illinois Urbana-Champaign ${}^{2}$ Alibaba Group. *Corresponding authors: Zhenhailong Wang <wangz3@illinois.edu>, Haiyang Xu <shuofeng.xhy@alibaba-inc.com>, Heng Ji <hengji@illinois.edu>.
${}^{1}$ 伊利诺伊大学香槟分校 ${}^{2}$ 阿里巴巴集团。 *通讯作者：王震海龙 <wangz3@illinois.edu>、徐海洋 <shuofeng.xhy@alibaba-inc.com>、季恆 <hengji@illinois.edu>。


---



## 1. Introduction
## 1. Introduction


Smartphones have become integral to our daily lives, transforming the way we connect, work, and find entertainment. Yet, the average 4.5 hours people spend on their phones daily* often includes moments of frustration. Tedious tasks, such as deal hunting across multiple apps or gathering scattered information from various websites, often make us wish for a smart mobile assistant to ease these burdens. Recent advancements in large multimodal models (LMMs) (OpenAI, 2024; Anthropic, 2024; Team et al., 2024) have led to the emergence of LMM-based GUI agents (Wang et al., 2024c; Nguyen et al., 2024) capable of perceiving and acting in the Web, PC, and mobile environments on behalf of human users. Despite these initial successes, current research on mobile agents (Wang et al., 2024b; Zhang et al., 2023; Wang et al., 2024a; Li et al., 2024) has yet to fully address the challenges of real-world mobile tasks. We identify two key limitations below.
智能手机已成为我们日常生活的组成部分，改变了我们的连接、工作与娱乐方式。然而，人们每天在手机上花费的平均约4.5小时* 常常包含挫折时刻。像跨多应用进行特价搜索、从各网站收集散落信息等繁琐任务，常让我们期望有一位智能移动助手来减轻负担。最近在大型多模态模型（LMMs）方面的进展（OpenAI，2024；Anthropic，2024；Team 等，2024）促成了基于LMM的GUI代理的出现（Wang 等，2024c；Nguyen 等，2024），能够在代表人类用户的前提下感知并在网页、PC和移动环境中行动。尽管初步取得了一些成功，当前关于移动代理的研究（Wang 等，2024b；Zhang 等，2023；Wang 等，2024a；Li 等，2024）尚未充分解决现实世界移动任务的挑战。下面我们指出两个关键局限性。


First, we observe a significant gap between the capabilities of current mobile agents and the demands of real-world scenarios. While existing mobile agent tasks are typically short, straightforward, and goal-oriented, such as "Navigate to a nearby gas station" (Wang et al., 2024a), tasks that better reflect actual human needs are far more complex. These tasks often require a combination of (1) intensive reasoning to address multiple constraints, such as balancing various factors or criteria; (2) long-horizon planning, which may involve a lengthy sequence of steps across multiple apps; and (3) exploration, where the instructions can be vague and require active information gathering rather than following a fixed trajectory. For instance, as shown in Figure 1, online shopping often involves navigating across different apps to compare prices and find the best deal. Furthermore, the highly dynamic nature of mobile environments, characterized by pop-up advertisements and frequently changing app layouts, poses additional challenges in tackling these complex real-world tasks.
首先，我们观察到当前移动智能体的能力与现实世界场景的需求之间存在显著差距。虽然后续的移动智能体任务通常较短、直接且以目标为导向，例如“导航到附近的加油站”（Wang et al., 2024a），但更能反映实际人类需求的任务要复杂得多。这些任务往往需要结合以下三方面： (1) 进行密集推理以应对多重约束，例如在各种因素或标准之间取得平衡；(2) 长期规划，可能涉及跨越多个应用的长序列步骤；以及 (3) 探索性执行，其中指令可能模糊，需要主动信息收集而非遵循固定轨迹。因此，如图 1 所示，网购往往需要在不同应用间切换以比较价格并寻求最佳交易。此外，移动环境的高动态性，如弹出广告和应用布局的频繁变化，也给处理这些复杂真实任务带来额外挑战。


Second, unlike humans, who quickly adapt and become proficient with new devices or apps, current mobile agents lack the ability to learn from prior experiences. For example, when a human user first opens an app like Maps, it may take some trial and error to understand the layout and successfully perform a search. However, with each interaction, the user learns, becoming faster and more accurate the next time. In contrast, existing mobile agents treat every task as if it were their first attempt, allocating the same computational resources at each step and repeating the same mistakes, regardless of how many times they perform the same task. This inability to accumulate knowledge and refine actions from past experiences severely limits their ability to handle the aforementioned complex, long-horizon tasks, where subroutines such as searching and creating notes are often shared across different objectives.
其次，与人类在新设备或新应用上迅速适应并变得熟练不同，当前移动智能体缺乏从以往经验中学习的能力。例如，当人类用户首次打开 Maps 等应用时，可能需要经过一些试错来理解布局并成功进行搜索。然而，随着每次交互，用户都会学习，下一次会更快也更准确。相反，现有移动智能体将每个任务都视为第一次尝试，在每一步分配相同的计算资源并重复同样的错误，不论他们已经执行同一任务多少次。这种无法从过去经验中积累知识并优化行动的能力严重限制了它们处理上述复杂、长期的任务的能力——这些任务通常需要跨不同目标共享的子程序，如搜索和记笔记等。


To address these limitations, we propose Mobile-Agent-E, a hierarchical multi-agent framework capable of self-evolution through past experiences. Mobile-Agent-E explicitly disentangles high-level planning-such as decomposing a task into smaller subgoals-from low-level actions, which involves determining specific actions and their parameters (e.g., tap (x, y)). The framework is structured with a Manager, responsible for creating overall plans, and four subordinate agents—Perceptor, Operator, Action Reflector, and Notetaker-that handle fine-grained visual perception, action decision, outcome verification, and information aggregation, respectively. This hierarchical design significantly enhances long-term planning and improves error recovery in complex tasks. Figure 1 shows an overview of Mobile-Agent-E on a challenging online shopping task requiring multi-step reasoning and interaction across three different apps.
为解决这些局限，我们提出 Mobile-Agent-E，这是一种通过 past experiences 自我进化的分层多智能体框架。Mobile-Agent-E 明确将高层次规划（如将任务分解为更小的子目标）与低层次行动分离开来，后者涉及确定具体行动及其参数（如点击 (x, y)）。该框架由一个 Manager（负责制定整体计划）和四个下属智能体组成——Perceptor、Operator、Action Reflector 和 Notetaker——分别处理细粒度的视觉感知、行动决策、结果验证和信息汇聚。这一分层设计显著提升了长期规划能力并改进了复杂任务中的错误恢复能力。图 1 显示了 Mobile-Agent-E 在一个需要跨三种不同应用、进行多步推理与交互的具有挑战性的网购任务中的概览。


Mobile-Agent-E also features a self-evolution module, which includes a persistent long-term memory and two Experience Reflectors. We define two types of critical knowledge that are continuously updated in the long-term memory across tasks: Tips-general guidance on effective interactions and lessons learned from previous trail-and-error experiences—and Shortcuts—reusable, executable functions that contains sequences of atomic operations tailored to efficiently complete recurring subroutines under specific preconditions. After completing each task, the Experience Reflectors are triggered to update the Tips and propose new Shortcuts based on the interaction history. These are then fed to the Manager and Operator, enabling improved planning and action decision-making in future tasks. This design draws inspiration from human cognitive science, where Tips are akin to the lessons encoded in episodic memory (Tul-ving, 2002), which involves recalling specific past experiences and using them to inform future decisions, while Shortcuts resemble procedural knowledge that facilitates the efficient and often subconscious execution of well-practiced tasks (Squire & Zola, 1996; Anderson, 1982). An example of Shortcuts and Tips is provided in Figure 1.
Mobile-Agent-E 还具备自我进化模块，其中包含持久的长期记忆和两个 Experience Reflectors。我们在跨任务的长期记忆中持续更新两种关键信息：Tips——关于有效交互的通用指引与以往试错经验教训，以及 Shortcuts——可重复利用、可执行的函数，包含适配特定前提条件的原子操作序列，以高效完成经常性子程序。在完成每个任务后，Experience Reflectors 会被触发来更新 Tips，并基于交互历史提出新的 Shortcuts。随后将这些输入给 Manager 与 Operator，从而在未来任务中提升规划与行动决策能力。这一设计汲取了人类认知科学的启发，其中 Tips 类似于在情景记忆（Tulving，2002）中编码的教训，涉及回忆具体的过去经历并将其用于指导未来决策；而 Shortcuts 类似于程序性知识，促成高效且常常在无意识中完成的熟练任务执行（Squire & Zola, 1996；Anderson, 1982）。图 1 中给出 Shortcuts 与 Tips 的一个示例。


---



*https://explodingtopics.com/blog/ smartphone-usage-stats
*https://explodingtopics.com/blog/ smartphone-usage-stats


---



To address the limitation of existing mobile benchmarks, which mainly include short-horizon and straightforward tasks with already saturated performance, we introduce a new benchmark, Mobile-Eval-E, designed to evaluate complex, real-world tasks. Mobile-Eval-E features more than twice the number of expected operations per task compared to previous benchmarks (Wang et al., 2024b; Zhang et al., 2023; Wang et al., 2024a) and incorporating a significantly higher proportion of tasks requiring multi-app interactions. Accompanying the benchmark, we introduce a new evaluation metric called the Satisfaction Score to address the challenge posed by real-world tasks that often lack a binary success flag or a ground truth trajectory. This metric is computed based on human-written rubrics that account for both milestone completion, such as "opened Maps," and exploratory behaviors, such as "viewed more than one review." This approach offers a reliable measure of agent performance aligned with human preferences. We further propose a Satisfaction Score vs Steps (SSS) curve to better evaluate and visualize the efficiency of mobile agents. Mobile-Eval-E sets a high standard of difficulty, with prior state-of-the-art methods achieving only about 50-70% of human satisfaction.
为解决现有移动基准测试的局限性——这些测试主要包含短期且相对简单、性能已经饱和的任务，我们推出新的基准 Mobile-Eval-E，旨在评估复杂的现实世界任务。Mobile-Eval-E 在每个任务中所需的操作数量比先前基准高出两倍以上（Wang et al., 2024b; Zhang et al., 2023; Wang et al., 2024a），并在任务中包含跨应用交互的比例显著提高。随基准测试，我们还引入一个新的评估指标“满意度分数”（Satisfaction Score），以应对现实世界任务往往缺乏二元成功标志或真实轨迹的问题。该指标基于人工撰写的评分标准，兼顾里程碑完成情况（如“打开 Maps”）与探索行为（如“查看超过一个评价”）。这一方法为评估智能体性能提供了与人类偏好一致的可靠度量。我们进一步提出 Satisfaction Score 与 Steps 的曲线（SSS）以更好地评估和展示移动智能体的效率。Mobile-Eval-E 设定了高难度标准，此前的最先进方法仅约达到人类满意度的 50-70%。


Empirical results show that Mobile-Agent-E achieves an average absolute gain of 22.1% over previous state-of-the-art approaches across three different foundation model backbones. Mobile-Agent-E also demonstrates promising self-evolution behavior in both performance and efficiency, resulting in a 6.5% absolute improvement compared to no evolution. The incorporation of Shortcuts further reduces the computational overhead, achieving speeds comparable to prior models while delivering significantly better performance. Additionally, we provide a comprehensive analysis of various aspects of self-evolution's impact and outline directions for future work.
实证结果显示，在三种不同基础模型骨干上，Mobile-Agent-E 相对于此前的最先进方法平均绝对提升为 22.1%。Mobile-Agent-E 还在性能与效率上展示出有希望的自我进化行为，与不进化相比实现了 6.5% 的绝对提升。引入快捷方式进一步降低计算开销，在速度接近早期模型的同时带来显著更好的性能。此外，我们对自我进化的各方面影响进行了全面分析，并勾勒出未来工作的方向。


## 2. Mobile-Agent-E
## 2. Mobile-Agent-E


Figure 2 provides an overview of Mobile-Agent-E. A summary of the notation definitions is presented in Table 1. We detail the hierarchical multi-agent framework (§2.1) and the self-evolution module (§2.2) in Mobile-Agent-E below.
图2给出了 Mobile-Agent-E 的概览。表1给出符号定义的摘要。下文我们在 Mobile-Agent-E 中详细阐述分层多智能体框架（§2.1）和自我进化模块（§2.2）。


Table 1. Notation definitions.
表1. 符号定义。


<table><tr><td></td><td>Notation Description</td></tr><tr><td colspan="2">Environment</td></tr><tr><td>I</td><td>Input task query</td></tr><tr><td>${a}^{t}$</td><td>Action† at time $t$</td></tr><tr><td>${s}^{t}$</td><td>Phone state (screenshot) at time $t$</td></tr><tr><td colspan="2">Agents</td></tr><tr><td>${\mathcal{A}}_{P}$</td><td>Perceptor</td></tr><tr><td>${\mathcal{A}}_{M}$</td><td>Manager</td></tr><tr><td>${\mathcal{A}}_{O}$</td><td>Operator</td></tr><tr><td>${\mathcal{A}}_{R}$</td><td>Action Reflector</td></tr><tr><td>${\mathcal{A}}_{N}$</td><td>Notetaker</td></tr><tr><td>${\mathcal{A}}_{ES}$</td><td>Experience Reflector for Shortcuts</td></tr><tr><td>${\mathcal{A}}_{ET}$</td><td>Experience Reflector for Tips</td></tr><tr><td colspan="2">Working Memory</td></tr><tr><td>${W}_{V}^{t}$</td><td>Visual perception result at time $t$</td></tr><tr><td>${W}_{P}^{t}$</td><td>Overall plan (decomposed subgoals) at time $t$</td></tr><tr><td>${W}_{S}^{t}$</td><td>Current subgoal at time $t$</td></tr><tr><td>${W}_{G}^{t}$</td><td>Progress status at time $t$</td></tr><tr><td>${W}_{N}^{t}$</td><td>Important notes at time $t$</td></tr><tr><td>${W}_{EF}^{t}$</td><td>Error Escalation Flag at time $t$</td></tr><tr><td>${\mathbf{W}}_{\mathbf{A}}$</td><td>Action history with outcome status</td></tr><tr><td>${\mathbf{W}}_{\mathbf{E}}$</td><td>Error history with feedback</td></tr><tr><td colspan="2">Long-term Memory</td></tr><tr><td>${L}_{S}$</td><td>Shortcuts</td></tr><tr><td>${L}_{T}$</td><td>Tips</td></tr></table>
<table><tbody><tr><td></td><td>符号说明</td></tr><tr><td colspan="2">环境</td></tr><tr><td>I</td><td>输入任务查询</td></tr><tr><td>${a}^{t}$</td><td>在时间 $t$ 的动作†</td></tr><tr><td>${s}^{t}$</td><td>在时间 $t$ 的电话状态（截图）</td></tr><tr><td colspan="2">代理</td></tr><tr><td>${\mathcal{A}}_{P}$</td><td>感知器</td></tr><tr><td>${\mathcal{A}}_{M}$</td><td>管理员</td></tr><tr><td>${\mathcal{A}}_{O}$</td><td>操作员</td></tr><tr><td>${\mathcal{A}}_{R}$</td><td>行动反射器</td></tr><tr><td>${\mathcal{A}}_{N}$</td><td>记笔记者</td></tr><tr><td>${\mathcal{A}}_{ES}$</td><td>快捷方式的体验反射器</td></tr><tr><td>${\mathcal{A}}_{ET}$</td><td>提示的体验反射器</td></tr><tr><td colspan="2">工作记忆</td></tr><tr><td>${W}_{V}^{t}$</td><td>在时间 $t$ 的视觉感知结果</td></tr><tr><td>${W}_{P}^{t}$</td><td>在时间 $t$ 的总体计划（分解的子目标）</td></tr><tr><td>${W}_{S}^{t}$</td><td>在时间 $t$ 的当前子目标</td></tr><tr><td>${W}_{G}^{t}$</td><td>在时间 $t$ 的进度状态</td></tr><tr><td>${W}_{N}^{t}$</td><td>在时间 $t$ 的重要注记</td></tr><tr><td>${W}_{EF}^{t}$</td><td>在时间 $t$ 的错误升级标记</td></tr><tr><td>${\mathbf{W}}_{\mathbf{A}}$</td><td>带结果状态的行动历史</td></tr><tr><td>${\mathbf{W}}_{\mathbf{E}}$</td><td>带反馈的错误历史</td></tr><tr><td colspan="2">长期记忆</td></tr><tr><td>${L}_{S}$</td><td>快捷方式</td></tr><tr><td>${L}_{T}$</td><td>提示</td></tr></tbody></table>


### 2.1. Hierachical Multi-Agent Framework
### 2.1. 层级化多智能体框架


Figure 3 provides a detailed breakdown of the main agent loop with concrete examples. Except for the Perceptor, all reasoning agents are instantiated from a frozen large multimodal model (LMM), such as GPT-40 (OpenAI, 2024). The inputs and outputs of each agent are detailed as follows.
图 3 提供了主智能体循环的详细分解及具体示例。除 Perceptor 外，所有推理智能体均来自冻结的大型多模态模型（LMM），如 GPT-40（OpenAI，2024）。每个智能体的输入输出如下所述。


Manager $\left( {\mathcal{A}}_{M}\right)$ : High-level planning. The Manager focuses on devising high-level plans to achieve the user's requests. At each step, the Manager checks the input query $I$ ,the current screenshot ${s}_{t}$ ,the previous overall plan ${W}_{P}^{t - 1}$ ,the previous subgoal ${W}_{S}^{t - 1}$ ,the progress status ${W}_{G}^{t - 1}$ ,available Shortcuts from long-term memory ${L}_{S}$ ,and any recorded important notes ${W}_{N}^{t - 1}$ to provide an updated overall plan ${W}_{P}^{t}$ and identify the next immediate subgoal ${W}_{S}^{t}$ to achieve. Note that the Manager does not condition on the fine-grained perception results from the Perceptor, as it is not necessary and can add noise to high-level planning.
Manager $\left( {\mathcal{A}}_{M}\right)$：高层次规划。经理专注于制定实现用户请求的高层计划。在每一步，经理会检查输入查询 $I$、当前截图 ${s}_{t}$、之前的总体计划 ${W}_{P}^{t - 1}$、上一目标 ${W}_{S}^{t - 1}$、进度状态 ${W}_{G}^{t - 1}$、来自长期记忆的可用快捷方式 ${L}_{S}$，以及任何记录的重要笔记 ${W}_{N}^{t - 1}$，以提供更新后的总体计划 ${W}_{P}^{t}$，并确定要实现的下一个直接子目标 ${W}_{S}^{t}$。请注意，经理不会以来自 Perceptor 的细粒度感知结果为条件，因为那样会增加对高层规划的噪声，不必要。


$$
{W}_{P}^{t},{W}_{S}^{t} = {\mathcal{A}}_{M}\left( {I,{s}_{t},{W}_{P}^{t - 1},{W}_{S}^{t - 1},{W}_{G}^{t - 1},{W}_{N}^{t - 1},{L}_{S}}\right)
$$



$$
\text{ if }t \geq  0\text{ and }{W}_{EF}^{t - 1} =  = \text{ False } \tag{1}
$$



$$
{W}_{P}^{t},{W}_{S}^{t} = {\mathcal{A}}_{M}\left( {I,{s}_{t},{W}_{P}^{t - 1},{W}_{S}^{t - 1},{W}_{G}^{t - 1},{W}_{N}^{t - 1},{L}_{S},}\right.
$$



$$
\left. {{\mathbf{W}}_{\mathbf{E}}\left\lbrack  {-k : }\right\rbrack  }\right) \text{ if }t \geq  k\text{ and }{W}_{EF}^{t - 1} =  = \text{ True } \tag{2}
$$



Additionally, when the model is potentially stuck in an error loop,that is,observing $k$ consecutive failed actions (e.g., $k = 2$ ) reported by the Action Reflector,a special Error Escalation Flag ${W}_{EF}^{t - 1}$ will be raised to the Manager. In such cases, the Manager will be prompted with additional information about the recent errors ${\mathbf{W}}_{\mathbf{E}}\left\lbrack  {-k : }\right\rbrack$ and asked to determine how to address the error from a higher-level perspective-such as refining the overall plan or adjusting the current subgoal to rectify the issue. In other cases, when an error first occurs, the Operator will attempt to address it before escalating the issue to the Manager. A concrete example of how the error escalation can help recovering from errors can be found in Figure 9.
此外，当模型可能陷入错误循环时，即观察到 $k$ 连续的失败操作（例如 $k = 2$）由 Action Reflector 报告时，将向经理提出一个特殊的错误升级标志 ${W}_{EF}^{t - 1}$。在这种情况下，经理将获得关于最近错误的额外信息 ${\mathbf{W}}_{\mathbf{E}}\left\lbrack  {-k : }\right\rbrack$，并被要求从更高层次的角度来确定如何解决错误——如对总体计划进行细化或调整当前子目标以纠正问题。在其他情况下，当首次发生错误时，操作者将尝试先解决它再将问题升级给经理。关于错误升级如何帮助从错误中恢复的具体示例，请参见图 9。


---



${}^{ \dagger  }{a}_{t}$ can represent either a single atomic operation or a sequence of atomic operations if performing a Shortcut.
${}^{ \dagger  }{a}_{t}$ 既可以表示单个原子操作，也可以表示在执行快捷方式时的一系列原子操作。


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_066342.jpg"/>



Figure 2. An overview of the Mobile-Agent-E framework,where the Manager,Perceptor $\left( {\mathcal{A}}_{P}\right)$ ,Operator,Action Reflector,and Notetaker are involved in the main agent loop for each task, while two Experience Reflectors contribute to updating long-term memory across tasks. Decision-making at each step is disentangled into high-level planning by the Manager and low-level actions by the Operator. The Action Reflector verifies the outcome of each action, tracks progress, and provides error feedback. The Notetaker aggregates important information during navigation. A detailed example illustrating one step in the agent loop and the self-evolution process is presented in Figures 3 and 4.
图 2. Mobile-Agent-E 框架概览，其中经理、Perceptor $\left( {\mathcal{A}}_{P}\right)$、Operator、Action Reflector 与 Notetaker 参与每个任务的主智能体循环，而两个 Experience Reflector 有助于跨任务更新长期记忆。每一步的决策分解为经理的高层计划与操作者的低层动作。Action Reflector 验证每项行动的结果、跟踪进度并提供错误反馈。Notetaker 在导航过程中汇聚重要信息。图 3 和图 4 给出展示智能体循环一个步骤及自我演化过程的详细示例。


Perceptor $\left( {\mathcal{A}}_{P}\right)$ : Fine-grained visual perception. The Perceptor aims to detect and recognize rich information about the current phone state, such as icons and text. We use a purely vision-based perception module that does not rely on the underlying XML file, following (Wang et al., 2024a). The Perceptor consists of three main components: an OCR model, an icon grounding model, and an icon captioning model. Given a screenshot ${s}_{t}$ at time $t$ ,the Perceptor generates a fine-grained list of texts and icons, along with their corresponding coordinates ${W}_{V}^{t}$ . Note that we still provide the original screenshot image to subsequent reasoning agents as a holistic visual context.
Perceptor $\left( {\mathcal{A}}_{P}\right)$：细粒度视觉感知。Perceptor 的目标是检测并识别当前手机状态的丰富信息，如图标和文本。我们使用纯基于视觉的感知模块，不依赖底层 XML 文件，遵循（Wang 等，2024a）。Perceptor 由三大组件组成：光学字符识别（OCR）模型、图标定位模型和图标描述模型。给定时间点 $t$ 的截图 ${s}_{t}$，Perceptor 生成细粒度的文本和图标列表及其相应坐标 ${W}_{V}^{t}$。请注意，我们仍然向后续推理智能体提供原始截图图像作为整体视觉上下文。


$$
{W}_{V}^{t} = {\mathcal{A}}_{P}\left( {s}_{t}\right) \tag{3}
$$



Operator $\left( {\mathcal{A}}_{O}\right)$ : Low-level action decisions. The Operator decides which concrete action to perform based on the input query $I$ ,the overall plan ${W}_{P}^{t}$ and current subgoal ${W}_{S}^{t}$ from the Manager,the previous progress status ${W}_{G}^{t - 1}$ ,the important notes ${W}_{N}^{t - 1}$ ,along with a history of the latest $m$ actions ${\mathbf{W}}_{\mathbf{A}}\left\lbrack  {-m : }\right\rbrack$ and errors ${\mathbf{W}}_{\mathbf{E}}\left\lbrack  {-m : }\right\rbrack  .{}^{ \ddagger  }$ The action history includes both the action and its outcome (success or failure). The Operator is explicitly prompted to rectify errors if it observes unresolved failures in the history. The Operator also considers the Tips as guidance from the long-term memory, which can be self-evolved from past experiences. To enable accurate generation of the action parameters, e.g., the (x,y) coordinates on the screen for tapping, we also provide the Operator with the fine-grained perception results ${W}_{V}^{t}$ from the Perceptor along with the screenshot ${s}_{t}$ .
运维 $\left( {\mathcal{A}}_{O}\right)$ : 低级行动决策。运维基于输入查询 $I$、总体计划 ${W}_{P}^{t}$、来自管理者的当前子目标 ${W}_{S}^{t}$、之前的进度状态 ${W}_{G}^{t - 1}$、重要注记 ${W}_{N}^{t - 1}$，以及最近 $m$ 动作 ${\mathbf{W}}_{\mathbf{A}}\left\lbrack  {-m : }\right\rbrack$ 与错误 ${\mathbf{W}}_{\mathbf{E}}\left\lbrack  {-m : }\right\rbrack  .{}^{ \ddagger  }$ 的历史记录来决定执行的具体行动。动作历史记录同时包含动作及其结果（成功或失败）。若观察到历史中存在未解决的失败，运维将被明确提示纠正错误。运维还将 Tips 视作来自长期记忆的指引，可以基于 past experiences 自我演化。为实现行动参数的准确生成，例如在屏幕上点击的(x,y)坐标，我们还提供来自 Perceptor 的细粒度感知结果 ${W}_{V}^{t}$ 以及截图 ${s}_{t}$ 给运维使用。


$$
{a}_{t} = {\mathcal{A}}_{O}\left( {I,{s}_{t},{W}_{V}^{t},{W}_{P}^{t},{W}_{S}^{t},{W}_{G}^{t},{W}_{N}^{t},}\right.
$$



$$
\left. {{\mathbf{W}}_{\mathbf{A}}\left\lbrack  {-m : }\right\rbrack  ,{\mathbf{W}}_{\mathbf{E}}\left\lbrack  {-m : }\right\rbrack  ,{L}_{S},{L}_{T}}\right) \tag{4}
$$



The output of the Operator is the next action ${a}_{t}$ to perform. The action space is defined to contain not only Atomic Operations but also Shortcuts, which can evolve through tasks. The atomic operations include Open_App, Tap, Swipe, Type, Enter, Switch_App, Back, Home, and Wait. The full descriptions of the atomic operations can be found in Table 8. We detail the definitions and examples of Shortcuts and Tips in §2.2.
运维的输出是要执行的下一个行动 ${a}_{t}$。行动空间不仅包含原子操作，还包含可以在任务中演化的快捷方式（Shortcuts）。原子操作包括 Open_App、Tap、Swipe、Type、Enter、Switch_App、Back、Home 与 Wait。原子操作的完整描述见表 8。我们在 §2.2 详细说明快捷方式（Shortcuts）与提示（Tips）的定义与示例。


---



${}^{ \ddagger  }$ We empirically set $m = 5$ in our experiments.
${}^{ \ddagger  }$ 我们在实验中经验性地设定 $m = 5$。


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_5914d2.jpg"/>



Figure 3. A detailed breakdown of one inference step $t$ with Mobile-Agent-E,showing the inputs and outputs of each agent. Omitted information indicates no change.
图 3. 使用 Mobile-Agent-E 的单步推断的详细拆解 $t$，展示各代理的输入与输出。省略信息表示无变化。


Action Reflector $\left( {\mathcal{A}}_{R}\right)$ : Reflection on the action outcome. The Action Reflector checks the screenshots before $\left( {s}_{t}\right)$ and after $\left( {s}_{t + 1}\right)$ of an action $\left( {a}_{t}\right)$ to verify if the previous action achieves the expected outcome. We define three types of outcomes for an action: A. Successful or partially successful: the result of the last action meets the expectation; ${}^{8}$ B. Failed: the last action results in a wrong page; and C. Failed: the last action produces no changes. After identifying the outcome, if the outcome is A, the Action Reflector updates the action history ${\mathbf{W}}_{\mathbf{A}}\left\lbrack  t\right\rbrack$ as well as the progress status ${W}_{G}^{t}$ . If the outcome is B or C, the Action Reflector additionally provides a description of the error and suggests potential reasons and solutions in ${\mathbf{W}}_{\mathbf{E}}\left\lbrack  t\right\rbrack$ .
行动校验器 $\left( {\mathcal{A}}_{R}\right)$ : 对行动结果的反思。行动校验器在 $\left( {s}_{t}\right)$ 之前与 $\left( {s}_{t + 1}\right)$ 之后对一个行动 $\left( {a}_{t}\right)$ 的屏幕截图进行检查，以验证前一行动是否达到预期结果。我们将一个行动的结果定义为三种类型：A. 成功或部分成功：前一行动的结果符合预期；${}^{8}$ B. 失败：前一行动导致页面错误；C. 失败：前一行动未产生任何变化。在识别结果后，如果结果为 A，行动校验器会更新行动历史 ${\mathbf{W}}_{\mathbf{A}}\left\lbrack  t\right\rbrack$ 以及进度状态 ${W}_{G}^{t}$。如果结果为 B 或 C，行动校验器还会在 ${\mathbf{W}}_{\mathbf{E}}\left\lbrack  t\right\rbrack$ 提供错误描述并给出潜在原因与解决方案。


$$
{W}_{V}^{t + 1} = {\mathcal{A}}_{P}\left( {s}_{t + 1}\right) \;\text{ \#run Perceptor on }{s}_{t + 1} \tag{5}
$$



$$
{\mathbf{W}}_{\mathbf{A}}\left\lbrack  t\right\rbrack  ,{\mathbf{W}}_{\mathbf{E}}\left\lbrack  t\right\rbrack  ,{W}_{G}^{t} = {\mathcal{A}}_{R}\left( {I,{s}_{t},{W}_{V}^{t},{s}_{t + 1},{W}_{V}^{t + 1},}\right.
$$



$$
\left. {{a}_{t},{W}_{S}^{t},{W}_{G}^{t - 1},}\right) \tag{6}
$$



Notetaker $\left( {\mathcal{A}}_{N}\right)$ : Information aggregation. In complex mobile tasks, we often need to keep track of important notes during exploration, such as the price of a product or the phone number of a restaurant. The Notetaker is dedicated to extracting and aggregating task-relevant information ${W}_{N}^{t}$ after each step,based on the input query $I$ ,overall plan ${W}_{P}^{t}$ ,current subgoal ${W}_{S}^{t}$ ,current progress ${W}_{G}^{t}$ ,fine-grained screen perception ${W}_{V}^{t + 1}$ after executing the action, and existing notes ${W}_{N}^{t - 1}$ .
记事者 $\left( {\mathcal{A}}_{N}\right)$ : 信息汇总。在复杂的移动任务中，我们常在探索过程中需要记录重要笔记，如商品价格或餐馆电话。记事者专注于在每一步后基于输入查询 $I$、总体计划 ${W}_{P}^{t}$、当前子目标 ${W}_{S}^{t}$、当前进度 ${W}_{G}^{t}$、执行行动后对屏幕的细粒度感知 ${W}_{V}^{t + 1}$ 以及已有笔记 ${W}_{N}^{t - 1}$，提取并汇聚与任务相关的信息 ${W}_{N}^{t}$。


$$
{W}_{N}^{t} = {\mathcal{A}}_{N}\left( {I,{s}_{t + 1},{W}_{V}^{t + 1},{W}_{P}^{t},{W}_{S}^{t},{W}_{G}^{t},{W}_{N}^{t - 1}}\right) \tag{7}
$$



### 2.2. Self-Evolution Module
### 2.2. 自我进化模块


Inspired by how humans become increasingly effective and efficient in operating smartphones, we maintain a long-term memory that persists across tasks and leverage two dedicated agents to reflect on past experiences. The long-term memory contains two important types of knowledge to evolve upon, Tips and Shortcuts, aiming to improve both the performance and efficiency of the agent. Figure 4 provides a detailed breakdown of one self-evolution step.
受人类在操作智能手机方面变得越发高效的启发，我们维持一个跨任务持续的长期记忆，并使用两位专属代理来回顾过去的经验。长期记忆包含需要演化的两类重要知识——Tips 与 Shortcuts，旨在提升代理的性能与效率。图 4 对单次自我进化步骤给出详细拆解。


Tips $\left( {L}_{T}\right)$ are defined as general guidance on effective interactions and lessons learned from previous trial-and-error experiences. Tips resemble episodic memory (Tulving, 2002), which enables humans to recall past experiences and apply insights to future decisions.
Tips $\left( {L}_{T}\right)$ 被定义为关于有效互动的一般性指导以及以往试错经验教训的总结。Tips 类似于情节记忆（Tulving, 2002），使人类能够回想过去的经历并将洞见应用于未来的决策。


---



${}^{\text{ § }}$ Some actions may need multiple repetitions to fulfill the expectation, for example, swipe up to find reviews. Thus, we include partially successful as meeting the expectation.
${}^{\text{ § }}$ 一些行动可能需要多次重复才能符合预期，例如向上滑动以查看评论。因此，我们将部分成功视为符合预期。


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_e79e65.jpg"/>



Figure 4. Illustration of the inputs and outputs to the Experience Reflectors for a single self-evolution step, including a concrete example of the newly generated Shortcuts and Tips.
图 4. 展示单步自我进化过程的 Experience Reflectors 的输入与输出，以及新生成的 Shortcuts 与 Tips 的具体示例。


Shortcuts $\left( {L}_{S}\right)$ are defined as reusable,executable functions composed of sequences of atomic operations tailored for recurring subroutines. Shortcuts are akin to procedural knowledge, which allows humans to perform well-practiced tasks efficiently and often subconsciously (Squire & Zola, 1996; Anderson, 1982). Due to the highly dynamic nature of the mobile environment, a Shortcut may only be applicable in certain states. For instance, the "Tap_Type_and_Enter" Shortcut is usable only when the current screen has a text input box. To address this, we explicitly include a precondition in the definition of a Shortcut and require the Operator to verify that the current state satisfies the precondition before using the Shortcut. The arguments of a Shortcut have a unique one-to-one mapping to the arguments of its atomic operations.
Shortcuts $\left( {L}_{S}\right)$ 定义为可重复使用的、由一系列原子操作组成、为重复子程序量身定制的可执行函数。Shortcuts 类似于过程性知识，使人类能够高效且常常在无意识中执行长期练习过的任务（Squire & Zola, 1996; Anderson, 1982）。由于移动环境的高度动态性，Shortcut 可能仅在某些状态下适用。例如，“Tap_Type_and_Enter” Shortcut 仅在当前屏幕有文本输入框时可用。为解决这一点，我们在 Shortcut 的定义中明确包含一个前提条件，并要求操作员在使用 Shortcut 之前验证当前状态是否满足该前提。Shortcut 的参数与其原子操作的参数之间具有唯一的一对一映射关系。


When the self-evolution module is enabled, we leverage two Experience Reflectors, ${\mathcal{A}}_{ES}$ and ${\mathcal{A}}_{ET}$ ,to update the Tips and Shortcuts at the end of each task. The Experience Reflectors are also instantiated from frozen large multimodal model such as GPT-40. Let the final time step of a task be $t = \tau$ . The input to the Experience Reflectors includes the input query $I$ ,the final overall plan ${W}_{P}^{\tau }$ ,the final progress status ${W}_{G}^{\tau }$ ,the entire action history ${\mathbf{W}}_{\mathbf{A}}$ and error history ${\mathbf{W}}_{\mathbf{E}}$ ,the existing Shortcuts ${L}_{S}$ and Tips ${L}_{T}$ ,and a list of future tasks ${T}_{F}$ (if provided). The outputs consist of newly generated Shortcuts in a predefined JSON format and updated Tips in natural language. Figures 12 and 13 shows a full list of generated Shortcuts and Tips by Mobile-Agent-E.
当自我进化模块开启时，我们使用两个 Experience Reflectors，${\mathcal{A}}_{ES}$ 与 ${\mathcal{A}}_{ET}$，在每个任务结束时更新 Tips 与 Shortcuts。Experience Reflectors 也可从冻结的大型多模态模型（如 GPT-4o）实例化。将任务的最终时间步记为 $t = \tau$。Experience Reflectors 的输入包括查询输入 $I$、最终总体计划 ${W}_{P}^{\tau }$、最终进度状态 ${W}_{G}^{\tau }$、完整的动作历史 ${\mathbf{W}}_{\mathbf{A}}$、错误历史 ${\mathbf{W}}_{\mathbf{E}}$、现有的 Shortcuts ${L}_{S}$ 与 Tips ${L}_{T}$、以及未来任务清单 ${T}_{F}$（若提供）。输出包括以预定义的 JSON 格式生成的新 Shortcuts 以及以自然语言更新的 Tips。图 12 与图 13 显示了 Mobile-Agent-E 生成的完整 Shortcuts 与 Tips 列表。


$$
{L}_{T} = {\mathcal{A}}_{ET}\left( {I,{W}_{P}^{\tau },{W}_{G}^{\tau },{\mathbf{W}}_{\mathbf{A}},{\mathbf{W}}_{\mathbf{E}},{T}_{F},{L}_{T}}\right) \tag{8}
$$



$$
{L}_{S} = {\mathcal{A}}_{ES}\left( {I,{W}_{P}^{\tau },{W}_{G}^{\tau },{\mathbf{W}}_{\mathbf{A}},{\mathbf{W}}_{\mathbf{E}},{T}_{F},{L}_{S}}\right) \tag{9}
$$



The updated Tips and Shortcuts are then utilized by the Manager and the Operator in the subsequent task, facilitating evolution in both high-level planning and low-level action decisions.
更新后的 Tips 与 Shortcuts 将在后续任务中由管理者与操作员使用，从而在高层规划与低层行动决策上实现演化。


## 3. Experiments
## 3. 实验


We perform a dynamic evaluation, evaluating the models in real-time and on actual devices-following previous work (Wang et al., 2024b; Zhang et al., 2023; Wang et al., 2024a). Specifically, we use the Android Debug Bridge (ADB) to control an Android phone ${}^{ \Uparrow  }$ and perform human evaluation on the recorded screenshots and action histories.
我们进行动态评估，在真实设备上实时评估模型，遵循以往工作（Wang et al., 2024b; Zhang et al., 2023; Wang et al., 2024a）。具体地，我们使用 Android Debug Bridge（ADB）控制一部安卓手机 ${}^{ \Uparrow  }$，并对记录的屏幕截图与动作历史进行人工评估。


#### 3.1.A More Challenging Benchmark: Mobile-Eval-E
#### 3.1.A 更具挑战性的基准：Mobile-Eval-E


Existing dynamic benchmarks (Wang et al., 2024b; Zhang et al., 2023; Wang et al., 2024a) primarily focus on short-horizon, straightforward tasks, where the performance has already saturated. To address this limitation, we propose a new and challenging benchmark, Mobile-Eval-E, which emphasizes reasoning-intensive, long-horizon, multi-app tasks. Mobile-Eval-E comprises 25 manually crafted tasks spanning 5 real-world scenarios: "Restaurant Recommendation", "Information Searching", "Online Shopping", "What's Trending", and "Travel Planning". As shown in Table 2, Mobile-Eval-E significantly surpasses previous benchmarks in complexity,featuring more than $\mathbf{2} \times$ the number of expected operations per task and a greater total number of operations. Most tasks in existing benchmarks can be viewed as specific subgoals in Mobile-Eval-E. Additionally, Mobile-Eval-E encompasses a broader range of Apps, with 76% of the tasks requiring interactions with multiple Apps-compared to less than ${10}\%$ in previous benchmarks. In $\text{ § }3$ , we demonstrate that this benchmark presents a substantial challenge for existing state-of-the-art models. The full set of task queries can be found in Appendix Table 7. Due to the long-horizon nature of the tasks, we keep the number of tasks relatively small to ensure a reasonable human evaluation workload for fine-grained analysis.
现有动态基准（Wang et al., 2024b; Zhang et al., 2023; Wang et al., 2024a）主要关注短期、简单任务，性能已趋于饱和。为解决此局限，我们提出一个新的、更具挑战性的基准 Mobile-Eval-E，强调推理密集、长期、跨多应用的任务。Mobile-Eval-E 由 25 个手工设计的任务组成，覆盖 5 个现实场景：\"餐厅推荐\"、\"信息检索\"、\"在线购物\"、\"热门趋势\"与\"旅行规划\"。如表 2 所示，Mobile-Eval-E 在复杂性方面显著超过以往基准，任务期望操作次数多于 $\mathbf{2} \times$ 倍，总操作次数也更大。现有基准中的大多数任务可以看作是 Mobile-Eval-E 的具体子目标。另外，Mobile-Eval-E 覆盖更广泛的应用范围，约 76% 的任务需要与多个应用交互，而在以往基准中这一比例不足 ${10}\%$。在 $\text{ § }3$，我们证明该基准对现有最先进模型提出了相当大的挑战。完整的任务查询集合见附录表 7。由于任务的长时间跨度，我们将任务数量保持在相对较少，以确保在细粒度分析中的人类评估工作量在合理范围内。


---



${}^{q}$ A Samsung Galaxy A15 is used for all experiments.
${}^{q}$ 一部三星 Galaxy A15 用于所有实验。


---



Table 2. Comparison with existing dynamic evaluation benchmarks on real devices. Mobile-Eval-E emphasizes long-horizon, complex tasks that require significantly more operations and a wider variety of apps.
表 2. 与真实设备上现有动态评估基准的对比。Mobile-Eval-E 强调长时程、需要显著更多操作和更广泛应用程序的复杂任务。


<table><tr><td>Benchmark</td><td>#Tasks</td><td>#Multi-App Tasks</td><td>#Apps</td><td>Avg # Ops</td><td>Total # Ops</td></tr><tr><td>Mobile-Eval</td><td>33</td><td>3</td><td>10</td><td>5.55</td><td>183</td></tr><tr><td>Mobile-Eval-v2</td><td>44</td><td>4</td><td>10</td><td>5.57</td><td>245</td></tr><tr><td>AppAgent</td><td>45</td><td>0</td><td>9</td><td>6.31</td><td>284</td></tr><tr><td>Mobile-Eval-E</td><td>25</td><td>19</td><td>15</td><td>14.56</td><td>364</td></tr></table>
<table><tbody><tr><td>基准</td><td>#任务</td><td>#多应用任务</td><td>#应用</td><td>平均执行数 # Ops</td><td>总执行数 # Ops</td></tr><tr><td>移动端评估</td><td>33</td><td>3</td><td>10</td><td>5.55</td><td>183</td></tr><tr><td>移动端评估-v2</td><td>44</td><td>4</td><td>10</td><td>5.57</td><td>245</td></tr><tr><td>应用代理</td><td>45</td><td>0</td><td>9</td><td>6.31</td><td>284</td></tr><tr><td>移动端评估-E</td><td>25</td><td>19</td><td>15</td><td>14.56</td><td>364</td></tr></tbody></table>


### 3.2. Metrics with Better Human Alignment
### 3.2. 具有更好人类对齐的度量


Previous dynamic evaluation typically employs a binary success rate or a completion rate against a "ground truth" trajectory to evaluate the level of task completeness. However, real-world tasks often do not have a binary success flag or a single ground truth action sequence. For example, some tasks, such as "Plan a one-day itinerary for Palo Alto," may involve exploration and information aggregation, where multiple reasonable solutions might exist. Thus, we seek to measure human satisfaction rather than exact matches with a ground truth trajectory. For each task, we first manually write a list of rubrics (an example shown in Figure 5(a)), containing both milestone steps (e.g., "Opened Tripadvisor") and satisfaction criteria (e.g., "Viewed multiple attractions"). We then define the Satisfaction Score (SS) as the number of fulfilled rubrics divided by the total number of rubrics, as judged by a human evaluator.
以往的动态评估通常采用对“真实轨迹”完成度的二元成功率或完成率来评估任务的完整程度。然而，现实世界的任务往往没有二元的成功标志或单一的真实轨迹。例如，“为帕洛阿尔托计划一天行程”等任务可能涉及探索与信息汇总，存在多种合理的解决方案。因此，我们寻求衡量人类满意度，而非与真实轨迹的严格匹配。对每个任务，我们先人工撰写一份评审标准清单（如图5(a)所示的一个示例），其中包含里程碑步骤（如“打开 Tripadvisor”）以及满意度标准（如“查看了多处景点”）。随后，我们将满足的评审项数除以评审项总数，作为 Satisfaction Score（SS），由人工评估者给出。


We also include Action Accuracy (AA) and Reflection Accuracy (RA) as metrics to evaluate action-level performance. These metrics are also assessed by humans through a review of recorded screenshots and action histories. Finally, we include a Termination Error (TE) rate to reflect the agent's robustness and error recovery capability. There are five ways an agent can exit from performing a task: (1) self-reported success: the agent decides to stop on its own; (2) reaching the maximum number of iterations: we set the maximum iteration count to 40 to prevent infinite loops; (3) reaching the maximum number of consecutive errors: if the agent has an action reflector and it identifies 3 consecutive errors, the agent is exited; (4) reaching the maximum number of repeated actions: if the agent performs the exact same action (excluding Swipe and Back) more than 3 consecutive times; (5) any other errors, such as errors when parsing the raw response into a valid action. If a task exits in one of the ways described in 2-5, it is marked as having a termination error. The TE rate is computed as the ratio of tasks with termination errors to all tasks.
我们还引入 Action Accuracy（AA）与 Reflection Accuracy（RA）作为评估行动层面表现的指标。这些指标也由人工通过对记录的截图和行动历史进行复核来评估。最后，我们加入 Termination Error（TE）比率，以反映代理的鲁棒性与错误恢复能力。代理结束任务的方式有五种：（1）自报成功：代理自行决定停止；（2）达到最大迭代次数：我们将最大迭代次数设为40，以防止无限循环；（3）达到连续错误的最大次数：若代理有行动反思器且识别出连续3次错误，代理将被退出；（4）达到重复动作的最大次数：若代理执行完全相同的动作（不包括 Swipe 与 Back）超过3次；（5）其他错误，例如在将原始响应解析为有效动作时出错。如果任务以2-5中的任一方式退出，则记为存在终止错误。TE 率等于具有终止错误的任务数与总任务数的比率。


### 3.3. Evaluating Self-Evolving Mobile Agents
### 3.3. 自我进化移动代理的评估


To the best of our knowledge, this is the first work exploring evaluation in cross-task evolution settings. We consider two variants of Mobile-Agent-E: with and without the self-evolution module. When self-evolution module is enabled—referred to as Mobile-Agent-E + Evo-the agent performs sequentially across tasks within each scenario from the Mobile-Eval-E benchmark. The five tasks in a scenario share a persistent long-term memory. At the end of the $k$ -th task, the Experience Reflectors are prompted to update the long-term memory based on the interaction history of the current task as well as the queries for the remaining $5 - k$ tasks. This mimics the implicit requirement for an evolving agent to plan ahead, storing relevant knowledge for future interactions. In this setting, tasks performed later in the sequence benefit from a greater accumulation of Tips and Shortcuts, enabling us to analyze the progressive impact of self-evolution over time (detailed in Figure 6).
在我们所知范围内，这是首次在跨任务演化设定中探索评估。我们考虑两种 Mobile-Agent-E 的变体：带自我演化模块与不带自我演化模块。当启用自我演化模块时——记作 Mobile-Agent-E + Evo——代理在 Mobile-Eval-E 基准中的每个场景内按顺序执行一系列任务。一个场景中的五个任务共享一个持续的长期记忆。在第 $k$ 任务结束时，Experience Reflectors 将基于当前任务的交互历史以及对剩余 $5 - k$ 任务的查询，提示更新长期记忆。这模仿了一个进化代理需要做出前瞻性规划、为未来交互存储相关知识的隐性要求。在这种设置下，序列后面执行的任务将受益于更丰富的知识积累，便于我们分析自我演化随时间的渐进影响（详见图6）。


### 3.4. Models
### 3.4. 模型


Baselines. We compare against a wide range of open-sourced mobile agent frameworks, including AppA-gent (Zhang et al., 2023), Mobile-Agent-v1 (Wang et al., 2024b), and Mobile-Agent-v2 (Wang et al., 2024a). To maximize an apple-to-apple comparison with Mobile-Agent-v2, which is the previous state-of-the-art, we apply an identical atomic operation space, perception model, and initial Tips to Mobile-Agent-v2 as Mobile-Agent-E. AppAgent originally requires an additional exploration phase, which does not fit our setting; thus, we add the initial Tips as additional knowledge.
基线。我们对比了一系列开源移动代理框架，包括 AppAgent（Zhang 等，2023）、Mobile-Agent-v1（Wang 等，2024b）与 Mobile-Agent-v2（Wang 等，2024a）。为实现与 Mobile-Agent-v2 的苹果对苹果比较（它是此前的最先进模型），我们将 Mobile-Agent-v2 的原子操作空间、感知模型和初始提示与 Mobile-Agent-E 统一。AppAgent 原本需要额外的探索阶段，不符合我们的设定；因此，我们将初始提示作为额外知识加入。


Backbones. We explore using various large multimodal models (LMM) as backbones for the reasoning agents, including GPT-40 (OpenAI, 2024)", Claude-3.5-Sonnet (Anthropic, 2024)**, and Gemini-1.5-pro (Team et al., 2024) Unless otherwise specified, the default backbone for all models is GPT-40.
Backbones。我们探索将各种大型多模态模型（LMM）作为推理代理的骨架，包括 GPT-40（OpenAI，2024）、Claude-3.5-Sonnet（Anthropic，2024）以及 Gemini-1.5-pro（Team 等，2024）。除非另有说明，所有模型的默认骨架为 GPT-40。


---



"GPT-40 version: gpt-40-2024-11-20
"GPT-40 版本：gpt-40-2024-11-20


---



Table 3. Comparison with state-of-the-art models on the Mobile-Eval-E benchmark, using GPT-40 as the backbone. Mobile-Agent-E outperforms previous SOTA models by a significant margin across all metrics, demonstrating superior long-term planning, decision accuracy, and error recovery. Enabling self-evolution (Mobile-Agent-E + Evo) further enhances performance. Reflection Accuracy for AppAgent and Mobile-Agent-v1 are omitted since they do not have action reflectors.
表3. 在 Mobile-Eval-E 基准上以 GPT-40 为骨架的状态-of-the-art 模型对比。Mobile-Agent-E 在所有指标上均显著优于先前的 SOTA 模型，展现出更出色的长期规划、决策准确性与错误恢复能力。启用自我演化（Mobile-Agent-E + Evo）进一步提升了性能。AppAgent 与 Mobile-Agent-v1 的 Reflection Accuracy 因无行动反射器而略去。


<table><tr><td>Model</td><td>Type</td><td>Satisfaction Score (%) ↑</td><td>Action Accuracy (%) ↑</td><td>Reflection Accuracy (%) ↑</td><td>Termination Error (%)↓</td></tr><tr><td>AppAgent (Zhang et al., 2023)</td><td>Single-Agent</td><td>25.2</td><td>60.7</td><td>-</td><td>96.0</td></tr><tr><td>Mobile-Agent-v1 (Wang et al., 2024b)</td><td>Single-Agent</td><td>45.5</td><td>69.8</td><td>-</td><td>68.0</td></tr><tr><td>Mobile-Agent-v2 (Wang et al., 2024a)</td><td>Multi-Agent</td><td>53.0</td><td>73.2</td><td>96.7</td><td>52.0</td></tr><tr><td>Mobile-Agent-E</td><td>Multi-Agent</td><td>75.1</td><td>85.9</td><td>97.4</td><td>32.0</td></tr><tr><td>Mobile-Agent-E + Evo</td><td>Multi-Agent</td><td>86.9</td><td>90.4</td><td>97.8</td><td>12.0</td></tr></table>
<table><tbody><tr><td>模型</td><td>类型</td><td>满意度分数 (%) ↑</td><td>行动准确性 (%) ↑</td><td>反思准确性 (%) ↑</td><td>终止误差 (%)↓</td></tr><tr><td>AppAgent (Zhang 等, 2023)</td><td>单智能体</td><td>25.2</td><td>60.7</td><td>-</td><td>96.0</td></tr><tr><td>Mobile-Agent-v1 (Wang 等, 2024b)</td><td>单智能体</td><td>45.5</td><td>69.8</td><td>-</td><td>68.0</td></tr><tr><td>Mobile-Agent-v2 (Wang 等, 2024a)</td><td>多智能体</td><td>53.0</td><td>73.2</td><td>96.7</td><td>52.0</td></tr><tr><td>Mobile-Agent-E</td><td>多智能体</td><td>75.1</td><td>85.9</td><td>97.4</td><td>32.0</td></tr><tr><td>Mobile-Agent-E + Evo</td><td>多智能体</td><td>86.9</td><td>90.4</td><td>97.8</td><td>12.0</td></tr></tbody></table>


Table 4. Results on different large multimodal model backbones, including GPT-40, Gemini, and Claude. The metrics SS, AA, RA, and TE represent Satisfaction Score, Action Accuracy, Reflection Accuracy, and Termination Error, respectively, expressed as percentages.
表4. 关于不同大型多模态模型骨干的结果，包括 GPT-40、Gemini 和 Claude。指标 SS、AA、RA 和 TE 分别表示满意度分数、行动准确率、反思准确率和终止错误，均以百分比表示。


<table><tr><td rowspan="2">Model</td><td colspan="4">Gemini-1.5-pro</td><td colspan="4">Claude-3.5-Sonnet</td><td colspan="4">GPT-40</td></tr><tr><td>SS↑</td><td>AA↑</td><td>RA↑</td><td>TE↓</td><td>SS↑</td><td>AA↑</td><td>RA↑</td><td>TE↓</td><td>SS↑</td><td>AA↑</td><td>RA↑</td><td>TE↓</td></tr><tr><td>Mobile-Agent-v2 (Wang et al., 2024a)</td><td>50.8</td><td>63.4</td><td>83.9</td><td>64.0</td><td>70.9</td><td>76.4</td><td>96.9</td><td>32.0</td><td>53.0</td><td>73.2</td><td>96.7</td><td>52.0</td></tr><tr><td>Mobile-Agent-E</td><td>70.9</td><td>74.3</td><td>91.3</td><td>48.0</td><td>75.5</td><td>91.1</td><td>99.1</td><td>12.0</td><td>75.1</td><td>85.9</td><td>97.4</td><td>32.0</td></tr><tr><td>Mobile-Agent-E + Evo</td><td>71.2</td><td>77.4</td><td>89.6</td><td>48.0</td><td>83.0</td><td>91.4</td><td>99.7</td><td>12.0</td><td>86.9</td><td>90.4</td><td>97.8</td><td>12.0</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td colspan="4">Gemini-1.5-pro</td><td colspan="4">Claude-3.5-Sonnet</td><td colspan="4">GPT-40</td></tr><tr><td>SS↑</td><td>AA↑</td><td>RA↑</td><td>TE↓</td><td>SS↑</td><td>AA↑</td><td>RA↑</td><td>TE↓</td><td>SS↑</td><td>AA↑</td><td>RA↑</td><td>TE↓</td></tr><tr><td>移动代理-v2（Wang 等，2024a）</td><td>50.8</td><td>63.4</td><td>83.9</td><td>64.0</td><td>70.9</td><td>76.4</td><td>96.9</td><td>32.0</td><td>53.0</td><td>73.2</td><td>96.7</td><td>52.0</td></tr><tr><td>移动代理-E</td><td>70.9</td><td>74.3</td><td>91.3</td><td>48.0</td><td>75.5</td><td>91.1</td><td>99.1</td><td>12.0</td><td>75.1</td><td>85.9</td><td>97.4</td><td>32.0</td></tr><tr><td>Mobile-Agent-E + Evo</td><td>71.2</td><td>77.4</td><td>89.6</td><td>48.0</td><td>83.0</td><td>91.4</td><td>99.7</td><td>12.0</td><td>86.9</td><td>90.4</td><td>97.8</td><td>12.0</td></tr></tbody></table>


Perceptor Implementation in Mobile-Agent-E. We closely follow Mobile-Agent-v2 (Wang et al., 2024a) to implement the Perceptor with slight modifications. We use DBNet#(Liao et al., 2020) and ConvNextViT-document## from ModelScope for OCR detection and recognition respectively. We use GroundingDINO (Liu et al., 2023) for icon grounding and Qwen-VL-Plus (Bai et al., 2023) for generating captions for each cropped icon.
在移动代理中的 Perceptor 实现。我们严格遵循 Mobile-Agent-v2 (Wang et al., 2024a) 的实现，做出轻微修改。我们分别使用 DBNet#(Liao et al., 2020) 和 ConvNextViT-document## 来自 ModelScope 进行 OCR 检测与识别。我们使用 GroundingDINO (Liu et al., 2023) 进行图标定位，以及 Qwen-VL-Plus (Bai et al., 2023) 为每个裁剪图标生成描述。


## 4. Results
## 4. 结果


### 4.1. Evaluation on Performance
### 4.1. 性能评估


Comparison with state-of-the-art. Table 3 presents the results on Mobile-Eval-E using an identical GPT-40 backbone for all baselines and Mobile-Agent-E. Mobile-Agent-E outperforms the previous multi-agent state-of-the-art (SOTA) model (Wang et al., 2024a) by 22.1% in the Satisfaction Score. This comparison particularly highlights the effectiveness of the hierarchy in our multi-agent framework. Our approach also demonstrates superior robustness and error recovery capabilities, as indicated by a significantly lower termination error rate. Moreover, enabling self-evolution further enhances performance, leading to an improvement of 33.9% against the previous SOTA, underscoring the benefit of learning from experience. In §4.3, we provide further analysis of the impact of the evolution module.
与最先进方法的比较。表3 给出在 Mobile-Eval-E 上使用相同的 GPT-40 主干网络对所有基线和 Mobile-Agent-E 的结果。Mobile-Agent-E 在满意度分数上超过之前的多代理 SOTA 模型（Wang et al., 2024a），提升了 22.1%。此对比特别凸显了我们多代理框架中层次结构的有效性。我们的方法还展示了更强的鲁棒性与错误恢复能力，终止错误率显著较低。此外，启用自我进化进一步提升了性能，相较于前一 SOTA，提升了 33.9%，强调了从经验中学习的好处。在 §4.3 中，我们对进化模块的影响进行了进一步分析。


Varying reasoning backbones. Table 4 shows the comparison with previous SOTA (Wang et al., 2024a) using various backbone LMMs. We observe consistent improvements on all recent LMMs, including GPT-40, Claude-3.5-Sonnet, and Gemini-1.5-pro, with average absolute gains of 22.1% and 15.6% with and without evolution, respectively. Additionally, the benefits of self-evolution appear to be more pronounced in stronger backbones, such as GPT-40 and Claude.
变化推理主干。表4 显示与此前 SOTA（Wang et al., 2024a）在不同主干 LMMs 下的比较。我们在所有较新 LMMs 上均观察到持续改进，包括 GPT-40、Claude-3.5-Sonnet 和 Gemini-1.5-pro，平均绝对增益在有无进化时分别为 22.1% 和 15.6%。此外，自我进化的好处在更强的主干上，如 GPT-40 与 Claude，似乎更为明显。


### 4.2. Evaluation on Efficiency
### 4.2. 评估效率


Evaluating the efficiency of mobile agents on complex, potentially open-ended tasks is not straightforward. Merely counting the number of steps is not optimal, as many tasks require exploration. A smaller number of steps reflects a quick exit but may result in insufficient exploration. Intuitively, if an agent achieves higher satisfaction, i.e., fulfills more rubrics, in a smaller number of steps, it is considered more efficient. Thus, we introduce the Satisfaction Score vs Steps (SSS) curve to compare and visualize the efficiency of different agents. To plot the SSS curve, we manually examine the recorded trajectories and track the satisfaction of rubrics after each step. We then plot a poly-line with the step number as the x-axis and the Satisfaction Score as the y-axis. To enable visualization of trajectories with different lengths on the same graph, we normalize the steps to the range $\left\lbrack  {0,1}\right\rbrack$ . The y-axis of the rightmost point indicates the final satisfaction score. Intuitively, a steeper and higher SSS curve indicates better efficiency and completeness. As shown in Figure 5, we observe that Mobile-Agent-E not only achieves better final performance but also fulfills rubrics faster.
在复杂、潜在开放式任务上评估移动代理的效率并非易事。仅仅统计步骤数并非最优，因为许多任务需要探索。步骤数较少虽然代表快速退出，但可能导致探索不足。直觉上，如果一个代理在更少的步骤内实现更高的满意度，即完成更多评估项，则被认为更高效。因此，我们引入 Satisfaction Score vs Steps (SSS) 曲线来对比并可视化不同代理的效率。绘制 SSS 曲线时，我们逐条检查记录的轨迹，在每一步后跟踪评估项的满意度。然后绘制一个以步骤数为 x 轴、满意度分数为 y 轴的折线图。为了在同一图上实现不同长度轨迹的可视化，我们将步骤数归一化到范围 $\left\lbrack  {0,1}\right\rbrack$。右端点的 y 轴表示最终的满意度分数。直观上，更陡更高的 SSS 曲线表示更高的效率和完整性。图5 如所示，我们观察到 Mobile-Agent-E 不仅在最终性能上更出色，也更快地完成评估项。


---



**Claude-3.5 version: claude-3-5-sonnet-20241022
**Claude-3.5 版本：claude-3-5-sonnet-20241022


††Gemini-1.5 version: gemini-1.5-pro-latest (Dec 2024)
††Gemini-1.5 版本：gemini-1.5-pro-latest (2024 年 12 月)


#https://modelscope.cn/models/iic/cv_ resnet18_ocr-detection-db-line-level_damo
#https://modelscope.cn/models/iic/cv_ resnet18_ocr-detection-db-line-level_damo


\$8 https://modelscope.cn/models/iic/cv_ convnext Tiny_ocr-recognition-document_damo
\$8 https://modelscope.cn/models/iic/cv_ convnext Tiny_ocr-recognition-document_damo


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_8453ba.jpg"/>



Figure 5. Satisfaction Score vs. Steps (SSS) curve for (a) a single task and (b) all tasks. In (a), we also provide a concrete example of the human-written rubrics for the task, which are used to compute the Satisfaction Score during human evaluation. In (b), we additionally include a linear regression line for each model; a steeper and higher line indicates better efficiency for completing the task.
图5。 (a) 单任务和 (b) 所有任务的 Satisfaction Score vs. Steps (SSS) 曲线。 在 (a) 中，我们还提供了任务的人类书写评估项的具体示例，用于在人工评估中计算 Satisfaction Score。 在 (b) 中，我们还为每个模型添加了线性回归线；线越陡、越高表示完成任务的效率越高。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_d19221.jpg"/>



Figure 6. Progressive impact of self-evolution over time. The task index represents the order in which a task is performed in the evolution setting. The results demonstrate that tasks performed later in the sequence show more significant improvements, highlighting the increased benefits from additional iterations of self-evolution.
图6。自我进化随时间的递进影响。任务索引表示在进化设置中执行任务的顺序。结果表明，序列中后执行的任务其改进更显著，凸显了额外迭代自我进化的收益增加。


### 4.3. Further Analysis
### 4.3. 进一步分析


Progressive impact of self-evolution over time. The ideal behavior of self-evolution is to progressively bring more benefits to the agent as knowledge accumulates. To investigate this, we group the results of the tasks by their ordering index in each scenario and compare the performance with and without enabling the evolution module. In Figure 6, the x-axis reflects the task index in the sequence
自我进化随时间的递进影响。自我进化的理想行为是随着知识积累，逐步为代理带来更多收益。为研究这一点，我们按场景中的排序索引对任务结果分组，并比较启用进化模块与否的性能。在图6中，x 轴表示序列中的任务索引


Table 5. Analysis of computational overhead and Shortcut usage. In the inference speed table, the reasoning only section accounts for time spent solely on reasoning agents, while perception + reasoning includes the runtime of the Perceptor on CPU. Shortcut usage statistics are calculated as the ratio of Shortcuts used to the total number of actions performed by the Operator. The use of Shortcuts significantly accelerates inference, achieving comparable times to previous, simpler frameworks.
表 5. 计算开销与 Shortcut 使用的分析。在推理速度表中，reasoning only 部分仅计入用于推理代理的时间，而 perception + reasoning 包含 Perceptor 在 CPU 上的运行时。Shortcut 使用统计量按使用的 Shortcuts 数量与 Operator 执行的总动作数之比计算。使用 Shortcuts 可以显著加速推理，达到与以往更简单框架相当的时间。


<table><tr><td colspan="7">Inference Speed (Seconds per operation)</td></tr><tr><td rowspan="2">Model</td><td colspan="3">Reasoning Only</td><td colspan="3">Perception + Reasoning</td></tr><tr><td>Gemini</td><td>Claude</td><td>GPT</td><td>Gemini</td><td>Claude</td><td>GPT</td></tr><tr><td>Mobile-Agent-v2</td><td>9.8</td><td>21.4</td><td>12.3</td><td>25.6</td><td>38.4</td><td>43.5</td></tr><tr><td>Mobile-Agent-E</td><td>16.5</td><td>25.5</td><td>17.4</td><td>30.8</td><td>41.0</td><td>30.1</td></tr><tr><td>Mobile-Agent-E + Evo</td><td>12.9</td><td>24.8</td><td>14.9</td><td>27.2</td><td>39.6</td><td>27.4</td></tr></table>
<table><tbody><tr><td colspan="7">推理速度（每次操作的秒数）</td></tr><tr><td rowspan="2">模型</td><td colspan="3">仅推理</td><td colspan="3">感知 + 推理</td></tr><tr><td>Gemini</td><td>Claude</td><td>GPT</td><td>Gemini</td><td>Claude</td><td>GPT</td></tr><tr><td>Mobile-Agent-v2</td><td>9.8</td><td>21.4</td><td>12.3</td><td>25.6</td><td>38.4</td><td>43.5</td></tr><tr><td>Mobile-Agent-E</td><td>16.5</td><td>25.5</td><td>17.4</td><td>30.8</td><td>41.0</td><td>30.1</td></tr><tr><td>Mobile-Agent-E + Evo</td><td>12.9</td><td>24.8</td><td>14.9</td><td>27.2</td><td>39.6</td><td>27.4</td></tr></tbody></table>


Shortcut Usage Percentage (%)
快捷键使用百分比 (%)


<table><tr><td>Model</td><td>Gemini</td><td>Claude</td><td>GPT</td></tr><tr><td>Mobile-Agent-E</td><td>11.9</td><td>12.8</td><td>12.4</td></tr><tr><td>Mobile-Agent-E + Evo</td><td>14.8</td><td>13.2</td><td>14.4</td></tr></table>
<table><tbody><tr><td>模型</td><td>Gemini</td><td>Claude</td><td>GPT</td></tr><tr><td>移动代理-E</td><td>11.9</td><td>12.8</td><td>12.4</td></tr><tr><td>移动代理-E + Evo</td><td>14.8</td><td>13.2</td><td>14.4</td></tr></tbody></table>


it is performed, with later tasks having access to Tips and Shortcuts that are updated through more tasks. We observe a generally increasing trend indicating that the gain tends to be more significant in later tasks, demonstrating that the self-evolution module is capable of continuously improving the agent as it experiences more tasks. The gain is not strictly monotonically increasing, as expected, since the difficulty of tasks at different indices varies.
它被执行，后续任务可以访问通过更多任务更新的 Tips 与 Shortcuts。我们观察到总体上呈现上升趋势，表明在后续任务中收益往往更显著，这表明自我进化模块能够在经历更多任务时不断提升代理能力。收益并非严格单调增加，这是有预期的，因为不同索引处任务的难度各异。


Shortcut reduces computational overhead. The hierarchical multi-agent architecture in Mobile-Agent-E significantly improves performance on complex tasks but inevitably increases computational complexity. However, we found that the use of Shortcuts largely mitigates this overhead, enabling Mobile-Agent-E to achieve a speed comparable to that of previous models. In Table 5, we report the seconds per operation averaged across all tasks as well as the usage of Shortcuts. We observe a positive correlation between using more Shortcuts and faster inference speed. This is because a Shortcut enables the execution of multiple operations within a single decision-making iteration. For example, using the Tap_Type_and_Enter Shortcut to perform a search subroutine saves two iterations of perception and reasoning compared to using three atomic actions: Tap, Type, and Enter.
Shortcut 降低了计算开销。Mobile-Agent-E 的分层多智能体结构在处理复杂任务时显著提高性能，但不可避免地增加了计算复杂度。然而，我们发现使用 Shortcut 在很大程度上减轻了这一开销，使 Mobile-Agent-E 的速度达到与前一代模型相当的水平。在表 5 中，我们报告跨所有任务的平均每次操作所需秒数以及 Shortcut 的使用情况。我们观察到使用更多 Shortcuts 与推理速度更快之间存在正相关关系。这是因为 Shortcut 使得在单次决策迭代中可以执行多项操作。例如，使用 Tap_Type_and_Enter Shortcut 执行搜索子程序，相较于使用 Tap、Type 和 Enter 三个原子动作，节省了两次感知与推理的迭代。


Table 6. To investigate the unique impact of Tips, we compute the Satisfaction Score on a subset of instances where no newly generated Shortcuts are used in the trajectory. The results show distinctive benefits from the evolved Tips.
表 6。为研究 Tips 的独特影响，我们在轨迹中未使用新生成的 Shortcuts 的实例子集上计算满意度分数。结果显示进化后 Tips 带来显著的收益。


<table><tr><td></td><td>Gemini</td><td>Claude</td><td>GPT-40</td></tr><tr><td>Mobile-Agent-E</td><td>69.0</td><td>75.6</td><td>79.7</td></tr><tr><td>Mobile-Agent-E + evolved Tips</td><td>72.6</td><td>85.2</td><td>87.5</td></tr></table>
<table><tbody><tr><td></td><td>Gemini</td><td>Claude</td><td>GPT-40</td></tr><tr><td>Mobile-Agent-E</td><td>69.0</td><td>75.6</td><td>79.7</td></tr><tr><td>Mobile-Agent-E + 演化提示</td><td>72.6</td><td>85.2</td><td>87.5</td></tr></tbody></table>


Unique impact from Tips. While the impact from Shortcuts is directly visible in the action history, it is less obvious whether the evolved Tips bring distinctive benefits. To visualize this, we filter out task instances where the same set of unique Shortcuts is used or where only atomic actions are employed, and compare the Satisfaction Score with or without the evolution. Table 6 shows that Tips alone serve as an important aspect of self-evolution.
Tips 带来的独特影响。虽然来自 Shortcuts 的影响在行动历史中直接可见，但进化后 Tips 是否带来显著利益则不太明显。为可视化这一点，我们过滤出使用相同集合的独特 Shortcuts 的任务实例，或仅使用原子操作的实例，并对比有无进化时的 Satisfaction Score。表 6 显示单独的 Tips 也是自我进化的重要方面。


### 4.4. Case Study: A Closed-Loop Self-Evolving Agent
### 4.4. 案例研究：一个闭环自我进化代理


In real-world mobile usage, after running the agent on a large number of tasks in various scenarios, the accumulated Tips and Shortcuts may grow to an amount where it is no longer feasible to include all of them in the decision-making context. Thus, in this case study, we aim to explore closing the self-evolution loop by introducing two additional Experience Retriever agents for Tips ${\mathcal{A}}_{ERT}$ and Shortcuts ${\mathcal{A}}_{ERS}$ . We consider a new task in an unknown scenario,as shown in Figure 7. First, we provide all the updated Tips and Shortcuts—after running Mobile-Agent-E on all 5 scenarios (a total of 25 tasks) in Mobile-Eval-E—to the Experience Retrievers. With GPT-40 as the backbone, the updated long-term memory contains a total of 7 unique Shortcuts and 58 Tips, among which 6 Shortcuts and 55 Tips are newly proposed by Mobile-Agent-E during experience reflection. Then, the Experience Retrievers are prompted to select only the relevant Tips and Shortcuts for the current task. The qualitative example in Figure 7 shows that Mobile-Agent-E effectively retrieves and leverages highly relevant Shortcuts and Tips to successfully complete a challenging unseen task. The full list of Tips and Shortcuts after evolution can be found in Appendices G and F.
在现实世界的移动端使用中，在对大量任务、各种场景运行代理后，累积的 Tips 和 Shortcuts 可能增长到无法在决策上下文中全部包含的程度。因此，在本案例研究中，我们旨在通过引入两个额外的 Experience Retriever 代理来针对 Tips ${\mathcal{A}}_{ERT}$ 与 Shortcuts ${\mathcal{A}}_{ERS}$ ，实现自我进化循环的闭合。我们在一个未知场景中考虑一个新任务，如图 7 所示。首先，我们在 Mobile-Eval-E 的 5 个场景（共 25 个任务）上运行 Mobile-Agent-E 之后，将所有更新的 Tips 与 Shortcuts 交给 Experience Retrievers。以 GPT-40 作为骨干，更新后的长期记忆包含共 7 个独特 Shortcuts 与 58 条 Tips，其中 6 条 Shortcuts 和 55 条 Tips 是在体验反思阶段由 Mobile-Agent-E 新提出的。随后，Experience Retrievers 被提示仅选择与当前任务相关的 Tips 与 Shortcuts。图 7 的定性示例显示，Mobile-Agent-E 能有效检索并利用高度相关的 Shortcuts 与 Tips，成功完成一个具有挑战性的未见任务。进化后的所有 Tips 与 Shortcuts 的完整列表可见于附录 G 与 F。


## 5. Related Work
## 5. 相关工作


#### 5.1.GUI Agents
#### 5.1.GUI 智能体


The advancement of large multimodal models (LMM) has introduced a new area of agentic research focused on LMM-based GUI agents (Wang et al., 2024c). The goal is to develop AI assistants capable of performing tasks in various GUI environments, such as Web (Deng et al., 2023; Zheng et al., 2024; He et al., 2024; Yoran et al., 2024; Reddy et al., 2024), PC (Hong et al., 2023; Zhang et al., 2024; Liu et al., 2024b; Xie et al., 2024; Tan et al., 2024), and mobile devices (Wang et al., 2024b; Zhang et al., 2023; Li et al., 2024; Wang et al., 2024a; Liu et al., 2024a). In the mobile environment, one line of research focuses on improving the perception and reasoning abilities of a single agent through tool usage (Wang et al., 2024b) and an additional exploration phase (Zhang et al., 2023; Li et al., 2024). Recent studies (Rawles et al., 2024; Wang et al., 2024a) show significant promise by incorporating multiple agents for decision-making and reflection. However, current multi-agent frameworks still face challenges such as short-sighted planning and poor error recovery. Specifically, the "planning" module in Mobile-Agent-v2 (Wang et al., 2024a) functions primarily as a progress tracker, while the "decision-making" module continues to handle both high-level planning (e.g., "what to do next") and low-level action decisions (e.g., "where to tap"). A key difference in Mobile-Agent-E is the introduction of a hierarchy among the agents, enabling more effective long-horizon planning and improved low-level action accuracy.
大型多模态模型（LMM）的发展引入了一个新的代理研究领域，聚焦于基于 LMM 的 GUI 智能体（Wang 等，2024c）。目标是开发能够在多种 GUI 环境中执行任务的 AI 助手，例如 Web（Deng 等，2023；Zheng 等，2024；He 等，2024；Yoran 等，2024；Reddy 等，2024）、PC（Hong 等，2023；Zhang 等，2024；Liu 等，2024b；Xie 等，2024；Tan 等，2024）和移动设备（Wang 等，2024b；Zhang 等，2023；Li 等，2024；Wang 等，2024a；Liu 等，2024a）。在移动环境中，一类研究着力于通过工具使用（Wang 等，2024b）和额外的探索阶段（Zhang 等，2023；Li 等，2024）来提升单一代理的感知与推理能力。最近的研究（Rawles 等，2024；Wang 等，2024a）通过引入多代理进行决策和反思，显示出显著的潜力。然而，当前的多代理框架仍面临如短视规划与差的错误恢复等挑战。具体而言，Mobile-Agent-v2（Wang 等，2024a）中的“规划”模块主要功能是进度跟踪，而“决策”模块仍需同时处理高层次的规划（例如“接下来做什么”）与低层次的行动决策（例如“在哪里点击”）。Mobile-Agent-E 的一个关键区别在于引入代理之间的层级结构，使得更有效的长远规划和更高的低层行动准确性成为可能。


### 5.2. Self-Evolution in Foundation Models
### 5.2. 基础模型中的自我进化


Investigating how to make large language models and multimodal models self-improve has long been an active area of research (Tao et al., 2024). One line of work focuses on enhancing the base abilities of foundation models, such as improving reasoning and reducing knowledge hallucination. This includes approaches like iterative refinement (Madaan et al., 2024), self-reflection (Shinn et al., 2024), self-training (Huang et al., 2022), self-improvement (Wang et al., 2024d), and multi-persona collaboration (Wang et al., 2023). Another line of work explores improving task-solving with foundation models through tool learning and tool creation (Cai et al., 2023; Qian et al., 2023; Yuan et al., 2023). In the context of GUI agents, self-evolution is less studied. The skill curation mechanism in Cradle (Tan et al., 2024) shows initial promise in the PC environment; however, no previous work has systematically explored self-evolution in mobile environments. In this work, we demonstrate the importance of self-evolution in both Tips and Shortcuts. Notably, unlike the "skills" in Cradle, which are directly added to the atomic operation space, we explicitly define preconditions for our Shortcuts, as this is critical for decision-making across multiple apps and varying layouts.
研究如何让大语言模型和多模态模型自我提升长期以来一直是活跃的研究领域（Tao 等，2024）。一类工作聚焦于提升基础模型的基本能力，如推理能力提升和知识幻觉减少。这包括迭代精炼（Madaan 等，2024）、自我反思（Shinn 等，2024）、自我训练（Huang 等，2022）、自我提升（Wang 等，2024d）以及多角色协作（Wang 等，2023）等方法。另一类工作通过工具学习与工具创建（Cai 等，2023；Qian 等，2023；Yuan 等，2023）来提升基础模型解决任务的能力。在 GUI 代理的语境中，自我进化研究较少。Cradle 的技能策展机制（Tan 等，2024）在 PC 环境中初现端倪；然而此前未有系统性研究在移动环境中进行自我进化。在本工作中，我们展示了自我进化在 Tips 与 Shortcuts 中的重要性。值得注意的是，与 Cradle 中直接被添加到原子操作空间的“技能”不同，我们为 Shortcuts 明确设定了前提条件，因为这对跨多个应用及不同布局的决策至关重要。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_c44480.jpg"/>



Figure 7. Case study example where relevant Shortcuts and Tips are automatically retrieved from the previously evolved long-term memory and subsequently leveraged to complete an unseen, challenging task. The action trajectory also includes an example where the agent recovers from an error.
图 7。案例研究示例：相关的 Shortcuts 与 Tips 会从先前进化的长期记忆中自动检索并随后用于完成一个未知的、具有挑战性的任务。行动轨迹还包括代理从错误中恢复的示例。


## 6. Conclusion and Future Work
## 6. 结论与未来工作


We introduce Mobile-Agent-E, a novel mobile assistant featuring a hierarchical multi-agent framework and a self-evolution module that significantly enhances long-term planning, error recovery, and efficiency, excelling in a wide variety of complex real-world tasks. Remaining limitations include the incorrect usage of Shortcuts with invalid preconditions and erroneous agent-generated Shortcuts, with detailed examples provided in Appendix C. Future work will focus on developing improved strategies for generating, invoking, and revising Shortcuts, enhancing personalization to better adapt to individual user needs, and strengthening safety precautions to enable more effective human-agent collaboration.
我们提出了 Mobile-Agent-E，一种新颖的移动助手，具备分层多代理框架与自我进化模块，显著提升长期规划、错误恢复和效率，在各种复杂的真实世界任务中表现出色。现阶段的局限包括对具有无效前提条件的 Shortcuts 的错误使用以及代理生成的 Shortcuts 错误，附在附录 C 里有详细示例。未来工作将聚焦于开发改进的 Shortcuts 生成、调用与修订策略，提升个性化以更好地适应个体用户需求，并加强安全预防措施以实现更高效的人机协作。


## Impact Statement
## 影响陈述


This paper aims to advance the field of LMM-based agents by developing a hierarchical multi-agent framework and benchmark to improve the usability and efficiency of smart-phones in complex, multi-step tasks. While the primary goal is to enhance human-device interaction, the proposed system has the potential for broader societal benefits, particularly in improving accessibility for individuals with disabilities or limited mobility. By enabling more intuitive and automated task management on mobile devices, this framework can assist users with physical impairments, cognitive challenges, or conditions that make precise interactions with touchscreens difficult.
本文旨在通过提出一个分层多代理框架和基准，推进基于 LMM 的代理在复杂多步任务中提升智能手机的可用性与效率。尽管主要目标是提升人机交互，但该系统具有更广泛的社会效益潜力，特别是改善残障或行动受限人群的可及性。通过在移动设备上实现更直观、更加自动化的任务管理，该框架可帮助具备身体障碍、认知挑战或与触控屏进行精确交互困难的用户。


While the primary aim is to enhance mobile task efficiency and user accessibility, the development of mobile agents capable of autonomous decision-making introduces potential risks. For example, unauthorized or unintended actions by the agent, such as the misuse of sensitive information including credit card details or private data, could result in serious consequences for users. These risks emphasize the critical need for robust safeguards, error recovery mechanisms, and fail-safe systems to ensure that the agent's actions consistently align with user intentions.
虽然主要目标是提升移动任务效率和用户可及性，但开发具有自主决策能力的移动代理也带来潜在风险。例如，代理的未授权或意外行为，如滥用包含信用卡信息或私人数据等敏感信息，可能对用户造成严重后果。这些风险凸显了需要强健的防护措施、错误恢复机制和容错系统，以确保代理的行为始终符合用户意图。


We are actively pursuing future work that focuses on designing and integrating robust privacy and safety mechanisms. These include explicit user consent workflows for sensitive operations, encryption protocols to protect user data during processing and storage, and automated systems to flag potentially harmful or unauthorized actions. These advancements will be crucial for maximizing the societal benefits of these systems, minimizing potential risks, and building user trust in autonomous mobile agents.
我们正积极推进聚焦于设计与整合稳健隐私和安全机制的未来工作。其中包括对敏感操作的明确用户同意流程、对在处理和存储过程中保护用户数据的加密协议，以及用于标记潜在有害或未授权行为的自动化系统。这些进步对最大化这些系统的社会效益、最小化潜在风险，以及建立用户对自治移动代理的信任将至关重要。


## References
## 参考文献


Anderson, J. R. Acquisition of cognitive skill. Psychological review, 89(4):369, 1982.
Anderson, J. R. Acquisition of cognitive skill. Psychological review, 89(4):369, 1982.


Anthropic. Claude 3.5 Sonnet, 2024. URL https://www.anthropic.com/news/ 3-5-models-and-computer-use.
Anthropic. Claude 3.5 Sonnet, 2024. URL https://www.anthropic.com/news/ 3-5-models-and-computer-use.


Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities. arXiv preprint arXiv:2308.12966, 2023. URL https: //doi.org/10.48550/arXiv.2308.12966.
Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities. arXiv preprint arXiv:2308.12966, 2023. URL https: //doi.org/10.48550/arXiv.2308.12966.


Cai, T., Wang, X., Ma, T., Chen, X., and Zhou, D. Large language models as tool makers. arXiv preprint arXiv:2305.17126, 2023.
Cai, T., Wang, X., Ma, T., Chen, X., and Zhou, D. Large language models as tool makers. arXiv preprint arXiv:2305.17126, 2023.


Deng, X., Gu, Y., Zheng, B., Chen, S., Stevens, S., Wang, B., Sun, H., and Su, Y. Mind2web: Towards a generalist agent for the web. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=kiYqbO3wqw.
Deng, X., Gu, Y., Zheng, B., Chen, S., Stevens, S., Wang, B., Sun, H., and Su, Y. Mind2web: 面向网页的通用代理。 In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=kiYqbO3wqw.


He, H., Yao, W., Ma, K., Yu, W., Dai, Y., Zhang, H., Lan, Z., and Yu, D. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024.
He, H., Yao, W., Ma, K., Yu, W., Dai, Y., Zhang, H., Lan, Z., and Yu, D. Webvoyager: 使用大规模多模态模型构建端到端网页代理。 arXiv 预印本 arXiv:2401.13919, 2024.


Hong, W., Wang, W., Lv, Q., Xu, J., Yu, W., Ji, J., Wang, Y., Wang, Z., Dong, Y., Ding, M., and Tang, J. Cogagent: A visual language model for gui agents, 2023.
Hong, W., Wang, W., Lv, Q., Xu, J., Yu, W., Ji, J., Wang, Y., Wang, Z., Dong, Y., Ding, M., and Tang, J. Cogagent: 面向图形用户界面的可视语言模型，作为代理，2023。


Huang, J., Gu, S. S., Hou, L., Wu, Y., Wang, X., Yu, H., and Han, J. Large language models can self-improve. arXiv preprint arXiv:2210.11610, 2022.
Huang, J., Gu, S. S., Hou, L., Wu, Y., Wang, X., Yu, H., and Han, J. 大语言模型可以自我提升。 arXiv 预印本 arXiv:2210.11610, 2022。


Li, Y., Zhang, C., Yang, W., Fu, B., Cheng, P., Chen, X., Chen, L., and Wei, Y. Appagent v2: Advanced agent for flexible mobile interactions. arXiv preprint arXiv:2408.11824, 2024.
Li, Y., Zhang, C., Yang, W., Fu, B., Cheng, P., Chen, X., Chen, L., and Wei, Y. Appagent v2：灵活移动交互的高级代理。 arXiv 预印本 arXiv:2408.11824, 2024。


Liao, M., Wan, Z., Yao, C., Chen, K., and Bai, X. Real-time scene text detection with differentiable binarization. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 11474-11481, 2020.
Liao, M., Wan, Z., Yao, C., Chen, K., and Bai, X. 基于可微分二值化的实时场景文本检测。 In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34, 页码 11474-11481, 2020。


Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., and Zhang, L. Grounding DINO: marrying DINO with grounded pre-training for open-set object detection. CoRR, abs/2303.05499, 2023. doi: 10.48550/ARXIV.2303.05499. URL https:// doi.org/10.48550/arXiv.2303.05499.
Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., and Zhang, L. Grounding DINO：将 DINO 与具定位预训练相结合用于开集目标检测。 CoRR, abs/2303.05499, 2023. doi: 10.48550/ARXIV.2303.05499. URL https:// doi.org/10.48550/arXiv.2303.05499.


Liu, X., Qin, B., Liang, D., Dong, G., Lai, H., Zhang, H., Zhao, H., Iong, I. L., Sun, J., Wang, J., et al. Autoglm: Autonomous foundation agents for guis. arXiv preprint arXiv:2411.00820, 2024a.
Liu, X., Qin, B., Liang, D., Dong, G., Lai, H., Zhang, H., Zhao, H., Iong, I. L., Sun, J., Wang, J., 等. Autoglm：面向 GUI 的自治基础代理。 arXiv 预印本 arXiv:2411.00820, 2024a。


Liu, X., Zhang, T., Gu, Y., Iong, I. L., Xu, Y., Song, X., Zhang, S., Lai, H., Liu, X., Zhao, H., et al. Visualagent-bench: Towards large multimodal models as visual foundation agents. arXiv preprint arXiv:2408.06327, 2024b.
Liu, X., Zhang, T., Gu, Y., Iong, I. L., Xu, Y., Song, X., Zhang, S., Lai, H., Liu, X., Zhao, H., 等. Visualagent-bench：向以视觉为基础的多模态模型迈进，成为视觉基础代理。 arXiv 预印本 arXiv:2408.06327, 2024b。


Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36, 2024.
Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., 等. Self-refine：带自我反馈的迭代精炼。 Advances in Neural Information Processing Systems, 36, 2024。


Nguyen, D., Chen, J., Wang, Y., Wu, G., Park, N., Hu, Z., Lyu, H., Wu, J., Aponte, R., Xia, Y., et al. Gui agents: A survey. arXiv preprint arXiv:2412.13501, 2024.
Nguyen, D., Chen, J., Wang, Y., Wu, G., Park, N., Hu, Z., Lyu, H., Wu, J., Aponte, R., Xia, Y., 等. Gui agents: 一项综述。 arXiv 预印本 arXiv:2412.13501, 2024。


OpenAI. GPT-40 System Card, 2024. URL https: //cdn.openai.com/gpt-4o-system-card. pdf.
OpenAI. GPT-40 系统卡，2024。URL https: //cdn.openai.com/gpt-4o-system-card. pdf.


Qian, C., Han, C., Fung, Y. R., Qin, Y., Liu, Z., and Ji, H. Creator: Tool creation for disentangling abstract and concrete reasoning of large language models. arXiv preprint arXiv:2305.14318, 2023.
Qian, C., Han, C., Fung, Y. R., Qin, Y., Liu, Z., and Ji, H. Creator：用于将大型语言模型的抽象推理与具体推理解耦的工具创建。 arXiv 预印本 arXiv:2305.14318, 2023。


Rawles, C., Clinckemaillie, S., Chang, Y., Waltz, J., Lau, G., Fair, M., Li, A., Bishop, W., Li, W., Campbell-Ajala, F., et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024.
Rawles, C., Clinckemaillie, S., Chang, Y., Waltz, J., Lau, G., Fair, M., Li, A., Bishop, W., Li, W., Campbell-Ajala, F., 等. Androidworld：面向自治代理的动态基准测试环境。 arXiv 预印本 arXiv:2405.14573, 2024。


Reddy, R. G., Mukherjee, S., Kim, J., Wang, Z., Hakkani-Tur, D., and Ji, H. Infogent: An agent-based framework for web information aggregation. arXiv preprint arXiv:2410.19054, 2024.
Reddy, R. G., Mukherjee, S., Kim, J., Wang, Z., Hakkani-Tur, D., and Ji, H. Infogent: 一个基于代理的网页信息聚合框架。 arXiv 预印本 arXiv:2410.19054, 2024.


Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.
Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: 具有语言强化学习的语言代理。 Advances in Neural Information Processing Systems, 36, 2024.


Squire, L. R. and Zola, S. M. Structure and function of declarative and nondeclarative memory systems. Proceedings of the National Academy of Sciences, 93(24): 13515-13522, 1996.
Squire, L. R. and Zola, S. M. 陈述性记忆与非陈述性记忆系统的结构与功能。 Proceedings of the National Academy of Sciences, 93(24): 13515-13522, 1996.


Tan, W., Ding, Z., Zhang, W., Li, B., Zhou, B., Yue, J., Xia, H., Jiang, J., Zheng, L., Xu, X., et al. Towards general computer control: A multimodal agent for red dead redemption ii as a case study. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024.
Tan, W., Ding, Z., Zhang, W., Li, B., Zhou, B., Yue, J., Xia, H., Jiang, J., Zheng, L., Xu, X., et al. 走向通用计算机控制：以红死救赎 ii 为个案的多模态代理。 In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024.


Tao, Z., Lin, T.-E., Chen, X., Li, H., Wu, Y., Li, Y., Jin, Z., Huang, F., Tao, D., and Zhou, J. A survey on self-evolution of large language models. arXiv preprint arXiv:2404.14387, 2024.
Tao, Z., Lin, T.-E., Chen, X., Li, H., Wu, Y., Li, Y., Jin, Z., Huang, F., Tao, D., and Zhou, J. 关于大型语言模型自我进化的综述。 arXiv 预印本 arXiv:2404.14387, 2024.


Team, G., Georgiev, P., Lei, V. I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
Team, G., Georgiev, P., Lei, V. I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al. Gemini 1.5：在大量上下文令牌中解锁多模态理解。 arXiv 预印本 arXiv:2403.05530, 2024.


Tulving, E. Episodic memory: From mind to brain. Annual review of psychology, 53(1):1-25, 2002.
Tulving, E. Episodic memory: 从心智到大脑。 Annual review of psychology, 53(1):1-25, 2002.


Wang, J., Xu, H., Jia, H., Zhang, X., Yan, M., Shen, W., Zhang, J., Huang, F., and Sang, J. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. arXiv preprint arXiv:2406.01014, 2024a.
Wang, J., Xu, H., Jia, H., Zhang, X., Yan, M., Shen, W., Zhang, J., Huang, F., and Sang, J. Mobile-agent-v2：通过多代理协作实现有效导航的移动设备操作助手。 arXiv 预印本 arXiv:2406.01014, 2024a.


Wang, J., Xu, H., Ye, J., Yan, M., Shen, W., Zhang, J., Huang, F., and Sang, J. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024b.
Wang, J., Xu, H., Ye, J., Yan, M., Shen, W., Zhang, J., Huang, F., and Sang, J. Mobile-agent：具备视觉感知的自主多模态移动设备代理。 arXiv 预印本 arXiv:2401.16158, 2024b.


Wang, S., Liu, W., Chen, J., Gan, W., Zeng, X., Yu, S., Hao, X., Shao, K., Wang, Y., and Tang, R. Gui agents with foundation models: A comprehensive survey. arXiv preprint arXiv:2411.04890, 2024c.
Wang, S., Liu, W., Chen, J., Gan, W., Zeng, X., Yu, S., Hao, X., Shao, K., Wang, Y., and Tang, R. Gui agents with foundation models: 一份全面综述。 arXiv 预印本 arXiv:2411.04890, 2024c.


Wang, Z., Mao, S., Wu, W., Ge, T., Wei, F., and Ji, H. Unleashing the emergent cognitive synergy in large language models: A task-solving agent through multi-persona self-collaboration. arXiv preprint arXiv:2307.05300, 2023.
Wang, Z., Mao, S., Wu, W., Ge, T., Wei, F., and Ji, H. 在大型语言模型中释放涌现认知协同：通过多人设自我协作的任务求解代理。 arXiv 预印本 arXiv:2307.05300, 2023.


Wang, Z., Hou, L., Lu, T., Wu, Y., Li, Y., Yu, H., and Ji, H. Enable lanuguage models to implicitly learn self-improvement from data. In Proc. The Twelfth International Conference on Learning Representations (ICLR2024), 2024d.
Wang, Z., Hou, L., Lu, T., Wu, Y., Li, Y., Yu, H., and Ji, H. 使语言模型能够从数据中隐式学习自我提升。 In Proc. The Twelfth International Conference on Learning Representations (ICLR2024), 2024d.


Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., Cheng, Z., Shin, D., Lei, F., et al. Os-world: Benchmarking multimodal agents for open-ended tasks in real computer environments. arXiv preprint arXiv:2404.07972, 2024.
Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., Cheng, Z., Shin, D., Lei, F., et al. Os-world: 基准测试多模态代理在真实计算环境中的开放式任务。 arXiv 预印本 arXiv:2404.07972, 2024.


Yoran, O., Amouyal, S. J., Malaviya, C., Bogin, B., Press, O., and Berant, J. Assistantbench: Can web agents solve realistic and time-consuming tasks?, 2024. URL https: //arxiv.org/abs/2407.15711.
Yoran, O., Amouyal, S. J., Malaviya, C., Bogin, B., Press, O., and Berant, J. Assistantbench: Web 代理能解决现实且耗时的任务吗？, 2024. URL https: //arxiv.org/abs/2407.15711.


Yuan, L., Chen, Y., Wang, X., Fung, Y. R., Peng, H., and Ji, H. Craft: Customizing llms by creating and retrieving from specialized toolsets. arXiv preprint arXiv:2309.17428, 2023.
Yuan, L., Chen, Y., Wang, X., Fung, Y. R., Peng, H., and Ji, H. Craft: 通过创建与检索专门工具集来定制 llms。 arXiv 预印本 arXiv:2309.17428, 2023.


Zhang, C., Yang, Z., Liu, J., Han, Y., Chen, X., Huang, Z., Fu, B., and Yu, G. Appagent: Multimodal agents as smartphone users, 2023.
Zhang, C., Yang, Z., Liu, J., Han, Y., Chen, X., Huang, Z., Fu, B., and Yu, G. Appagent: 多模态智能体作为智能手机用户，2023。


Zhang, C., Li, L., He, S., Zhang, X., Qiao, B., Qin, S., Ma, M., Kang, Y., Lin, Q., Rajmohan, S., Zhang, D., and Zhang, Q. UFO: A UI-Focused Agent for Windows OS Interaction. arXiv preprint arXiv:2402.07939, 2024.
Zhang, C., Li, L., He, S., Zhang, X., Qiao, B., Qin, S., Ma, M., Kang, Y., Lin, Q., Rajmohan, S., Zhang, D., and Zhang, Q. UFO: 一个面向 Windows OS 交互的 UI 为焦点的智能体。arXiv 预印本 arXiv:2402.07939, 2024。


Zheng, B., Gou, B., Kil, J., Sun, H., and Su, Y. Gpt- 4v(ision) is a generalist web agent, if grounded. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum? id=piecKJ2D1B.
Zheng, B., Gou, B., Kil, J., Sun, H., and Su, Y. Gpt- 4v(ision) 是一个具有普适性的网络智能体，若具备基础。收录于第四十一届国际机器学习大会，2024。URL https://openreview.net/forum? id=piecKJ2D1B。


## A. Full Trajectory Comparison Example with Previous SOTA
## A. 与以往 SOTA 的完整轨迹对比示例


Figure 8 presents the full trajectory of the task shown in Figure 1, comparing the previous state-of-the-art, Mobile-Agent-v2 (Wang et al., 2024a), and our proposed Mobile-Agent-E. Mobile-Agent-v2 suffers from early termination after interacting with two Apps, whereas Mobile-Agent-E fulfills all rubrics and stops at the App offering the best deal.
图 8 展示了图 1 所示任务的完整轨迹，与以往的最先进方法 Mobile-Agent-v2（Wang 等，2024a）以及我们提出的 Mobile-Agent-E 进行对比。Mobile-Agent-v2 在与两个应用交互后就提前终止，而 Mobile-Agent-E 能完成所有评判准则并在提供最佳交易的应用处停止。


## B. Error Recovery with Escalation to Manager
## B. 拓展至管理员的错误恢复


Figure 9 illustrates how the error escalation mechanism in Mobile-Agent-E enhances error recovery ability. A detailed description of the example is provided in the caption.
图 9 展示了 Mobile-Agent-E 的错误升级机制如何提升错误恢复能力。对该示例的详细描述见图题。


## C. Remaining Limitations
## C. 剩余的局限性


### C.1. Misuse of Shortcuts due to Incorrect Perception of Phone State
### C.1. 由于对手机状态的错误感知而导致快捷方式的误用


Although we explicitly require the Operator to verify the current phone state to ensure it fulfills the precondition of a Shortcut before calling it, there are still cases where the model incorrectly perceives the state, resulting in the misuse of Shortcuts in an invalid state. Figure 10 illustrates an example of such error. A detailed description of the example is provided in the caption. This type of error could potentially be mitigated by employing a dedicated agent for verifying preconditions or by enhancing the perception module to better understand phone states.
尽管我们明确要求操作员在调用快捷方式前核对当前手机状态以确保其满足前提条件，但仍存在模型错误感知状态、在无效状态下误用快捷方式的情况。图 10 给出此类错误的示例。对该示例的详细描述见图题。通过引入专门的前提条件验证代理或增强感知模块以更好理解手机状态，可能缓解此类错误。


### C.2. Errors and Imperfections in Self-Evolved Shortcuts
### C.2. 自我演化快捷方式的错误与瑕疵


Although effective in most cases, we still observe errors and imperfections in the agent-generated Shortcuts during self-evolution. These issues can lead to propagated errors when an erroneous Shortcut is used in subsequent tasks. Figure 11 illustrates an example of such erroneous and imperfect Shortcuts. A detailed description of the example is provided in the caption. This highlights the need for future work on approaches to generate higher-quality Shortcuts and equipping the agent with the ability to reflect on and revise generated Shortcuts in subsequent tasks.
尽管在大多数情况下有效，我们仍观察到代理生成的快捷方式在自我演化过程中出现错误与瑕疵。这些问题在后续任务中使用错误的快捷方式时可能会带来传播性错误。图 11 展示了此类错误和不完美的快捷方式示例。对该示例的详细描述见图题。这凸显了未来在生成更高质量快捷方式、并赋予代理对生成的快捷方式进行反思与修订能力方面的研究必要性。


### D.All Tasks in Mobile-Eval-E Benchmark
### D. Mobile-Eval-E 基准中的所有任务


Table 7 presents the input queries, involved App types, and scenarios for all Mobile-Eval-E tasks. The complete list of rubrics and human reference operation sequences is provided in the supplementary material.
表 7 展示了 Mobile-Eval-E 所有任务的输入查询、涉及的应用类型及情景。完整的评判准则及人工参考操作序列见补充材料。


## E. Atomic Operation Space
## E. 原子操作空间


Table 8 presents all atomic operations considered in Mobile-Agent-E.
表 8 介绍了 Mobile-Agent-E 中考虑的所有原子操作。


## F. Full list of Self-Evolved Shortcuts
## F. Self-Evolved Shortcuts 的完整列表


Figure 12 shows a full list of generated Shortcuts by Mobile-Agent-E after self-evolution on all 25 tasks from Mobile-Eval-E benchmark.
Figure 12 显示 Mobile-Agent-E 在 Mobile-Eval-E 基准的全部 25 项任务自进化后生成的 Shortcuts 的完整列表。


## G. Full list of Self-Evolved Tips
## G. Self-Evolved Tips 的完整列表


Figure 13 shows a full list of generated Tips by Mobile-Agent-E after self-evolution on all 25 tasks from Mobile-Eval-E benchmark.
Figure 13 显示 Mobile-Agent-E after 自进化在 Mobile-Eval-E 基准的全部 25 项任务生成的 Tips 的完整列表。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_dcafd3.jpg"/>



Figure 8. Full trajectory comparison between the previous state-of-the-art, Mobile-Agent-v2 (Wang et al., 2024a), and Mobile-Agent-E.
Figure 8. 之前的最优状态 Mobile-Agent-v2 (Wang et al., 2024a) 与 Mobile-Agent-E 之间的完整轨迹对比。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_bd83b3.jpg"/>



Figure 9. Error recovery with escalation. The task requires the agent to search for three different items on Walmart and note their sales information. At the step shown in the figure, the agent has already searched for ribeye steak and intends to search for fresh oranges next. However, the Operator erroneously calls the Shortcut that inputs text into the search bar and performs a search without clearing the previously entered text. Although the Action Reflector raises an error, the subgoal remains unchanged, and the Operator fails to rectify the error on the second attempt. After observing two consecutive errors, the error is escalated to the Manager, which correctly identifies the problem and revises the subgoal with detailed, decomposed steps to address the error. This helps the Operator correctly recover from the previous error by first tapping the " $\times$ " icon to clear the previous search query.
Figure 9. 升级的错误恢复。任务要求代理在 Walmart 对三种不同物品进行搜索并记录其销售信息。在图中所示步骤，代理已搜索肋眼牛排，打算接着搜索新鲜橙子。然而，操作员错误地调用了向搜索栏输入文本的 Shortcut 并在未清除先前文本的情况下执行搜索。尽管 Action Reflector 给出错误，子目标保持不变，操作员未能在第二次尝试中纠正错误。在观察到两次连续错误后，错误被升级到经理，经理正确识别问题并给出详细、分解的步骤以解决错误。这有助于操作员通过先点击 \"$\times$\" 图标清除前一次搜索查询来正确从前一次错误中恢复。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__20_17_53_4dfe02.jpg"/>



Figure 10. Example of misuse of Shortcuts in an invalid state. At the current step, as shown in the figure, the agent intended to switch back to Walmart to search for the final item requested by the user. While it correctly performs the "Switch_App" operation, it then calls a Shortcut for searching without realizing that it is not yet in the App where the search bar is available.
Figure 10. 在无效状态下滥用 Shortcuts 的示例。当前步骤如图所示，代理原本打算切换回 Walmart 以搜索用户请求的最终项。虽然正确执行了 \"Switch_App\" 操作，但随后调用了一个用于搜索的 Shortcut，却没有意识到当前所在的不是具有搜索栏的应用。


---



\{



&nbsp;&nbsp;&nbsp;&nbsp;"name": "Search_Location_in_Maps",
&nbsp;&nbsp;&nbsp;&nbsp;"name": "Search_Location_in_Maps",


&nbsp;&nbsp;&nbsp;&nbsp;"arguments": ["x","y","text"],
&nbsp;&nbsp;&nbsp;&nbsp;"arguments": ["x","y","text"],


&nbsp;&nbsp;&nbsp;&nbsp;"description": "Tap the search bar in Google Maps at position (x, y), type the location text, and select the first search result to display the route options.",
&nbsp;&nbsp;&nbsp;&nbsp;"description": "在 Google 地图中，在位置 (x, y) 处点击搜索栏，输入地点文本，并选择第一个搜索结果以显示路线选项。",


&nbsp;&nbsp;&nbsp;&nbsp;"precondition": "The Google Maps app is open, and the search bar is visible on the screen.",
&nbsp;&nbsp;&nbsp;&nbsp;"precondition": "Google 地图应用已打开，且屏幕上可见搜索栏。",


&nbsp;&nbsp;&nbsp;&nbsp;"atomic_action_sequence": [
&nbsp;&nbsp;&nbsp;&nbsp;"atomic_action_sequence": [


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Tap","arguments_map":\{"x":"x","y":"y"\}\},
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Tap","arguments_map":\{"x":"x","y":"y"\}\},


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Type","arguments_map":\{"text":"text"\}\},
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Type","arguments_map":\{"text":"text"\}\},


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Enter","arguments_map":\{\}\},
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Enter","arguments_map":\{\}\},


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Tap","arguments_map":\{"x":"x","y":"y"\}\}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Tap","arguments_map":\{"x":"x","y":"y"\}\}


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1



\{



&nbsp;&nbsp;&nbsp;&nbsp;"name": "Switch_App_And_Search",
&nbsp;&nbsp;&nbsp;&nbsp;"name": "Switch_App_And_Search",


&nbsp;&nbsp;&nbsp;&nbsp;"arguments": ["app_name","x","y","text"],
&nbsp;&nbsp;&nbsp;&nbsp;"arguments": ["app_name","x","y","text"],


&nbsp;&nbsp;&nbsp;&nbsp;"description": "Switch to a specified app, tap on a search bar at position (x, y), type the given text, and press Enter to perform a search.",
&nbsp;&nbsp;&nbsp;&nbsp;"description": "切换到指定应用，在位置 (x, y) 处点击搜索栏，输入给定文本，并按回车执行搜索。",


&nbsp;&nbsp;&nbsp;&nbsp;"precondition": "The app to switch to is already open in the app switcher, and the search bar is visible on the screen after switching.",
&nbsp;&nbsp;&nbsp;&nbsp;"precondition": "要切换到的应用已在应用切换器中打开，切换后屏幕上可见搜索栏。",


&nbsp;&nbsp;&nbsp;&nbsp;"atomic_action_sequence": [
&nbsp;&nbsp;&nbsp;&nbsp;"atomic_action_sequence": [


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Switch_App","arguments_map":\{\}\}, ___Missing an additional Tap
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Switch_App","arguments_map":\{\}\}, ___Missing an additional Tap


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Tap","arguments_map":\{"x":"x","y":"y"\}\}, - action to get into the App
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Tap","arguments_map":\{"x":"x","y":"y"\}\}, - action to get into the App


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Type","arguments_map":\{"text":"text"\}\},
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Type","arguments_map":\{"text":"text"\}\},


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Enter","arguments_map":\{\}\}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{"name":"Enter","arguments_map":\{\}\}


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]



\}



---



Figure 11. Example of imperfect (above) and erroneous (below) generated Shortcuts. The "Search_Location_in_Maps" Shortcut includes an unnecessary Tap action in the operation sequence, while the "Switch_App_And_Search" Shortcut omits a Tap action needed to first enter the desired App before performing the search.
Figure 11. Example of imperfect (above) and erroneous (below) generated Shortcuts. The "Search_Location_in_Maps" Shortcut includes an unnecessary Tap action in the operation sequence, while the "Switch_App_And_Search" Shortcut omits a Tap action needed to first enter the desired App before performing the search.


Table 7. All task queries in Mobile-Eval-E.
Table 7. All task queries in Mobile-Eval-E.


<table><tr><td>Scenario</td><td>Task ID</td><td>$\mathbf{{APPs}}$</td><td>Input Query</td></tr><tr><td rowspan="5">Restaurant Recommendation</td><td>1_late_night_korean_food</td><td>Maps</td><td>Find the best-rated late-night Korean restaurant in Champaign, IL that opens beyond 9pm on Google Maps.</td></tr><tr><td>1_nearest_bakery</td><td>Maps</td><td>Get directions to the nearest Bakery that has a rating higher than 4.0 on Google Maps. Stop at the screen showing the route.</td></tr><tr><td>1_thai_duck</td><td>Maps, Notes</td><td>Find the best-rated Thai restaurant in Urbana, IL that serves duck cuisine on Google Maps. Review customer comments and compile a summary of positive and negative feedback in Notes.</td></tr><tr><td>1_bakery_birthday_cake</td><td>Maps, Notes</td><td>Find me a Bakery that is within ${10}\mathrm{\;{min}}$ drive near me and does birthday cakes on Google Maps. Find the phone number and create a new note in Notes for that.</td></tr><tr><td>1_chinese_ohare</td><td>Maps, X, Notes</td><td>Find me a popular Chinese restaurant near Chicago O'Hare airport on Google Maps. Check X for recent posts about their signature dishes and write a summary in Notes. Then get directions to that restaurant on Google Maps. Stop at the screen showing the route.</td></tr><tr><td rowspan="5">Information Researching</td><td>2_segment_anything_cited</td><td>Chrome</td><td>Find the most-cited paper that cites the paper 'Segment Anything' on Google Scholar. Stop at the screen showing the paper abstract.</td></tr><tr><td>2_llm_agents_survey</td><td>Chrome, Notes</td><td>Find at least three representative survey papers on LLM agents on Google Scholar, and add their titles to the Notes.</td></tr><tr><td>2_recipes_chinese</td><td>Chrome, YouTube</td><td>I have some onions, beef, and potatoes in my refrigerator. Can you find me a Chinese-style recipe that uses all three ingredients and can be prepared in under an hour? And find me a video tutorial on YouTube for that. Stop at the screen displaying the video.</td></tr><tr><td>2_mcdonalds_deals</td><td>McDonald's, Maps</td><td>Can you check the McDonald's APP to see if there are any Rewards or Deals including Spicy McCrispy. If so, help me add that to Mobile Order (Do not pay yet, I will do it myself). And then check the pickup location and get directions on Google Maps. Stop at the screen showing the route.</td></tr><tr><td>2.headphones_reviews</td><td>Amazon, Notes</td><td>Find three detailed user reviews of the Bose QC45 headphones from Amazon. Summarize the general sentiment in the Notes.</td></tr><tr><td rowspan="5">Online Shopping</td><td>3_oled_tv</td><td>Best Buy</td><td>Find the best deal on a 55-inch 4K OLED TV at Best Buy. Stop at the screen displaying the best deal you find.</td></tr><tr><td>3_laptop_nvidia_gpu</td><td>Amazon Shopping</td><td>Find me a laptop on Amazon that is under \$1000 with an Nvidia GPU and more than 8GB RAM.</td></tr><tr><td>3_ninja_air_fryer</td><td>Amazon Shopping, Walmart</td><td>Compare the price of a Ninja air fryer 8 qt at Walmart and Amazon. Stop at the screen displaying the best deal you find.</td></tr><tr><td>3_walmart_sale_items</td><td>Walmart, Notes</td><td>Check if any of the following items are on sale at Walmart: ribeye steak, fresh oranges, or toilet paper. If any are on sale, add a note in Notes with their prices.</td></tr><tr><td>3_nintendo_switch_joy_con</td><td>Amazon Shopping, Best Buy, Walmart</td><td>I want to buy a brand-new Nintendo Switch Joy-Con. Any color is fine. Please compare the prices on Amazon, Walmart, and Best Buy. Find the cheapest option and stop at the screen where I can add it to the cart.</td></tr><tr><td rowspan="5">What's Trending</td><td>4_x_black_myth_wukong</td><td>X. Notes</td><td>Find the top posts about the game 'Black Myth Wukong' on X and summarize the key highlights in Notes.</td></tr><tr><td>4_x_trending_news</td><td>X, Notes</td><td>Check the top 3 trending news on X. Read a few posts to figure out what's happening. And create a new Note to summarize your findings.</td></tr><tr><td>4_watercolor_painting_tutorial</td><td>Lemon8, Notes</td><td>I want to learn how to paint watercolor. Find me some content creators to follow on Lemon8 tha has highly liked posts about watercolor painting tutorials. List their account names in Notes.</td></tr><tr><td>4_movie_trending</td><td>Fandango, Notes</td><td>Check the top 5 trending movies on Fandango that are currently in theaters. Compare their ratings and create a note in Notes for the highest-rated one, including its name and showtimes.</td></tr><tr><td>4_horror_movie_reviews</td><td>Fandango. Lemon8, Notes</td><td>Find me the latest horror movie currently in theaters on Fandango. Check some reviews on Lemon8 about the movie and create a note in Notes with the general sentiment.</td></tr><tr><td rowspan="5">Travel Planning</td><td>5_cheap_flights_newyork</td><td>Booking</td><td>Find the cheapest round-trip flight from Chicago to New York City in the next month on Booking. Stop at the screen showing the best deal.</td></tr><tr><td>5_things_to_do_la</td><td>Tripadvisor. Notes</td><td>Suggest some interesting things to do in LA. Find the top 3 attractions on Tripadvisor. Save the list in Notes.</td></tr><tr><td>5_palo_alto_tour</td><td>Tripadvisor, Notes</td><td>Plan a one-day itinerary for Palo Alto, CA using Tripadvisor. Choose the attractions and dining recommendations, but keep in mind that I don't like seafood and I love museums. Write the plan in Notes.</td></tr><tr><td>5_local_food_chicago</td><td>Tripadvisor, Notes</td><td>Find a highly recommended local restaurant in Chicago on Tripadvisor. Check the reviews about must-try dishes and summarize in Notes.</td></tr><tr><td>5_hotel_champaign</td><td>Booking, Maps</td><td>Help me find a hotel in Champaign, IL on Booking that is under \$200 for a queen bed. Make sure that the rating is higher than 7.0. Double-check on Google Maps to see if it is close to Green Street. Show me your final choice on Booking.</td></tr></table>
<table><tbody><tr><td>场景</td><td>任务ID</td><td>$\mathbf{{APPs}}$</td><td>输入查询</td></tr><tr><td rowspan="5">餐厅推荐</td><td>1_late_night_korean_food</td><td>地图</td><td>在谷歌地图中查找 Champaign, IL 夜间9点后仍营业、评价最高的深夜韩餐厅。</td></tr><tr><td>1_nearest_bakery</td><td>地图</td><td>获取到最近的面包店的导航，且在 Google 地图上评分高于4.0。停在显示路线的屏幕上。</td></tr><tr><td>1_thai_duck</td><td>地图，笔记</td><td>在谷歌地图中查找 Urbana, IL 评价最好的泰国餐厅，提供鸭肉菜肴。审阅顾客评论并在笔记中汇总积极与消极反馈。</td></tr><tr><td>1_bakery_birthday_cake</td><td>地图，笔记</td><td>在 Google 地图中找一家 ${10}\mathrm{\;{min}}$ 车程内且提供生日蛋糕的面包店。找出电话号码并在 Notes 中为该店创建新笔记。</td></tr><tr><td>1_chinese_ohare</td><td>地图，X，笔记</td><td>在谷歌地图中找一家靠近芝加哥奥黑尔机场的热门中餐馆。查看 X 的关于他们招牌菜的最新帖子并在 Notes 中写摘要。然后在谷歌地图中获取到该餐厅的路线。停在显示路线的屏幕。</td></tr><tr><td rowspan="5">信息检索</td><td>2_segment_anything_cited</td><td>Chrome</td><td>在 Google 学术上查找引用论文“Segment Anything”次数最多的论文。停在显示论文摘要的屏幕。</td></tr><tr><td>2_llm_agents_survey</td><td>Chrome，笔记</td><td>在 Google Scholar 至少找到三篇代表性的关于大语言模型代理的综述论文，并将它们的标题添加到笔记中。</td></tr><tr><td>2_recipes_chinese</td><td>Chrome，YouTube</td><td>我冰箱里有洋葱、牛肉和土豆。你能给我找一个使用这三样食材、且在一小时内能完成的中式食谱吗？并为此找一个 YouTube 视频教程。停在显示视频的屏幕上。</td></tr><tr><td>2_mcdonalds_deals</td><td>McDonald’s，地图</td><td>你能否查看 McDonald’s 应用，看看是否有任何含 Spicy McCrispy 的奖励或优惠？若有，帮我添加到手机订餐中（暂不付款，我自己来下单）。然后再检查自提地点并在谷歌地图上获取路线。停在显示路线的屏幕。</td></tr><tr><td>2.headphones_reviews</td><td>亚马逊，笔记</td><td>在亚马逊上找到三条关于 Bose QC45 耳机的详细用户评价。总结整体情感并写入笔记。</td></tr><tr><td rowspan="5">在线购物</td><td>3_oled_tv</td><td>Best Buy</td><td>在 Best Buy 找到一台 55 英寸 4K OLED 电视的最佳交易。停在显示你找到的最佳交易的屏幕上。</td></tr><tr><td>3_laptop_nvidia_gpu</td><td>亚马逊购物</td><td>在亚马逊找一台价格低于 1000 美元、带 NVIDIA 图形处理单元且内存大于 8GB 的笔记本。</td></tr><tr><td>3_ninja_air_fryer</td><td>亚马逊购物，沃尔玛</td><td>比较沃尔玛与亚马逊上 Ninja 8 qt 空气炸锅的价格。停在显示你找到的最佳交易的屏幕上。</td></tr><tr><td>3_walmart_sale_items</td><td>沃尔玛，笔记</td><td>检查以下商品在沃尔玛是否促销：肋眼牛排、新鲜橙子或卫生纸。如果有促销，请在笔记中记录价格。</td></tr><tr><td>3_nintendo_switch_joy_con</td><td>亚马逊购物、Best Buy、沃尔玛</td><td>我想买一台全新的任天堂 Switch Joy-Con。颜色随意。请比较亚马逊、沃尔玛和 Best Buy 的价格，找出最便宜的选项并停在可加入购物车的屏幕上。</td></tr><tr><td rowspan="5">热门趋势</td><td>4_x_black_myth_wukong</td><td>X。笔记</td><td>在 X 上找到关于游戏“Black Myth Wukong”的热门帖文并在笔记中总结要点。</td></tr><tr><td>4_x_trending_news</td><td>X，笔记</td><td>查看 X 上前 3 条趋势新闻。浏览几篇帖子以了解发生了什么，并创建新笔记总结你的发现。</td></tr><tr><td>4_watercolor_painting_tutorial</td><td>Lemon8，笔记</td><td>我想学习水彩画。请在 Lemon8 上找到一些绘有水彩画教程且受欢迎的内容创作者。把他们的账户名列在笔记中。</td></tr><tr><td>4_movie_trending</td><td>Fandango，笔记</td><td>查看正在上映的前五部 Fandango 热门电影。比较评分，并在笔记中为评分最高的电影创建一条笔记，包含其名称和场次。</td></tr><tr><td>4_horror_movie_reviews</td><td>Fandango。Lemon8，笔记</td><td>在 Fandango 上查找当前在院上映的最新恐怖电影。查看 Lemon8 上关于该电影的一些评价，在 Notes 中创建一个包含总体情感的笔记。</td></tr><tr><td rowspan="5">旅行规划</td><td>5_便宜机票_纽约</td><td>预订</td><td>在未来一个月内，在 Booking 上查找从芝加哥到纽约市的最便宜往返机票。停留在显示最佳交易的屏幕上。</td></tr><tr><td>5 件在 LA 可做的事</td><td>Tripadvisor. 笔记</td><td>在 LA 找一些有趣的活动。找出 Tripadvisor 上的前三大景点。将清单保存在 Notes 中。</td></tr><tr><td>5_palo_alto_tour</td><td>Tripadvisor，Notes</td><td>使用 Tripadvisor 为 Palo Alto, CA 计划一天行程。选择景点和用餐建议，但请记住我不喜欢海鲜且喜爱博物馆。将计划写在 Notes 中。</td></tr><tr><td>5_local_food_chicago</td><td>Tripadvisor，Notes</td><td>在 Tripadvisor 上找到芝加哥的一家高度推荐的本地餐厅。查看必尝菜品的评价并在 Notes 中做摘要。</td></tr><tr><td>5_hotel_champaign</td><td>Booking, 地图</td><td>请帮我在 Booking 上为芝加哥附近的 Champaign, IL 找一间价格低于 \$200 的 queen 床房的酒店。确保评分高于 7.0。再在 Google 地图上核对它是否靠近 Green Street。把最终选择在 Booking 上展示给我。</td></tr></tbody></table>


Table 8. Atomic operations space.
表 8. 原子操作的空间。


<table><tr><td>Operation</td><td>Description</td></tr><tr><td>Open_App(app_name)</td><td>If the current screen is Home or App screen, you can use this action to open the app named "app_name" on the visible on the current screen.</td></tr><tr><td>$\operatorname{Tap}\left( {x,y}\right)$</td><td>Tap the position $\left( {x,y}\right)$ in current screen.</td></tr><tr><td>$\operatorname{Swipe}\left( {{x}_{1},{y}_{1},{x}_{2},{y}_{2}}\right)$</td><td>Swipe from position $\left( {{x}_{1},{y}_{1}}\right)$ to position $\left( {{x}_{2},{y}_{2}}\right)$ . To swipe up or down to review more content, you can adjust the y-coordinate offset based on the desired scroll distance. For example, setting ${x}_{1} = {x}_{2} = {0.5} *$ width, ${y}_{1} = {0.5} *$ height,and ${y}_{2} = {0.1} *$ height will swipe upwards to review additional content below. To swipe left or right in the App switcher screen to choose between open apps, set the x-coordinate offset to at least 0.5 * width.</td></tr><tr><td>Type(text)</td><td>Type the "text" in an input box.</td></tr><tr><td>Enter()</td><td>Press the Enter key after typing (useful for searching).</td></tr><tr><td>Switch_App()</td><td>Show the App switcher for switching between opened apps.</td></tr><tr><td>Back()</td><td>Return to the previous state.</td></tr><tr><td>Home()</td><td>Return to home page.</td></tr><tr><td>Wait()</td><td>Wait for 10 seconds to give more time for a page loading.</td></tr></table>
<table><tbody><tr><td>操作</td><td>描述</td></tr><tr><td>打开应用(app_name)</td><td>若当前屏幕是主页或应用屏幕，可使用此动作打开当前屏幕上可见的名为“app_name”的应用。</td></tr><tr><td>$\operatorname{Tap}\left( {x,y}\right)$</td><td>在当前屏幕上点按位置 $\left( {x,y}\right)$。</td></tr><tr><td>$\operatorname{Swipe}\left( {{x}_{1},{y}_{1},{x}_{2},{y}_{2}}\right)$</td><td>从位置 $\left( {{x}_{1},{y}_{1}}\right)$ 向位置 $\left( {{x}_{2},{y}_{2}}\right)$ 滑动。若要向上或向下滚动以查看更多内容，可根据所需滚动距离调整 y 坐标偏移量。例如，设置 ${x}_{1} = {x}_{2} = {0.5} *$ 宽度、${y}_{1} = {0.5} *$ 高度和 ${y}_{2} = {0.1} *$ 高度将向上滑动以查看下方的更多内容。若要在应用切换器屏幕中向左或向右滑动以在打开的应用之间选择，请将 x 坐标偏移量设为至少 0.5 * 宽度。</td></tr><tr><td>输入文字(types)</td><td>在输入框中输入“text”。</td></tr><tr><td>Enter()</td><td>输入后按 Enter 键（用于搜索）。</td></tr><tr><td>切换应用(Switch_App())</td><td>显示应用切换器以在已打开的应用之间切换。</td></tr><tr><td>返回(Back())</td><td>返回到上一个状态。</td></tr><tr><td>主页(Home())</td><td>返回到主页。</td></tr><tr><td>等待(Wait())</td><td>等待 10 秒，为页面加载提供更多时间。</td></tr></tbody></table>


Inital Shortcuts (User Provided)
初始快捷键（用户提供）


"name": "Tap_Type_and_Enter",
"name": "Tap_Type_and_Enter",


"arguments": ["x","y","text"],
"arguments": ["x","y","text"],


"description": "Tap an input box at position (x, y), Type the \\"text\\", and then perform the Enter operation. Very useful for searching and sending messages!",
"description": "在坐标 (x, y) 处点击输入框，输入 \\"text\\"，然后执行 Enter 操作。对于搜索和发送消息非常有用！",


"precondition": "There is a text input box on the screen with no previously entered content.",
"precondition": "屏幕上有一个文本输入框且未输入内容。",


"atomic_action_sequence": [\{"name":"Tap","arguments_map": \{"x":"x","y":"y"\}], \{"name":"Type","arguments_map": \{"text":"text"\},"rangumets_map":\{\}\}\}
"atomic_action_sequence": [\{"name":"Tap","arguments_map": \{"x":"x","y":"y"\}], \{"name":"Type","arguments_map": \{"text":"text"\},"rangumets_map":\{\}\}\}


\{ Agent Generated Shortcuts
\{ 代理生成的快捷键


"name": "Create_New_Note",
"name": "Create_New_Note",


"arguments": ["text"],
"arguments": ["text"],


"description": "Create a new note in the Notes app and type the provided text into it.",
"description": "在 Notes 应用中创建一个新便签，并将提供的文本输入其中。",


"precondition": "The Notes app is open, and the 'Add' button (orange icon with a pencil) is visible on the screen.",
"precondition": "Notes 应用已打开，屏幕上可见‘添加’按钮（橙色铅笔图标）。",


"atomic_action_sequence": [\{"name""Tap"","arguments_map":\{"x""929","y""2053"\}\},\{"name":"Type","arguments_map":\{"text":"text"\}\}]
"atomic_action_sequence": [\{"name""Tap"","arguments_map":\{"x""929","y""2053"\}\},\{"name":"Type","arguments_map":\{"text":"text"\}\}]


"name": "Search Location in Maps",
"name": "Search Location in Maps",


"arguments": ["x","y","text"],
"arguments": ["x","y","text"],


"description": "Tap the search bar in Google Maps at position (x, y), type the location text, and select the first search result to display the route options.",
"description": "在 Google 地图的搜索栏处于坐标 (x, y) 的位置，输入位置信息文本，并选择第一个搜索结果以显示路线选项。",


"precondition": "The Google Maps app is open, and the search bar is visible on the screen.",
"precondition": "Google 地图应用已打开，屏幕上可见搜索栏。",


"atomic_action_sequence": [\{"name":"Tap","arguments_map":\{"x":"x","y":"y"\},\{"name":"Type","arguments_map": \{"text","treaty"\},\{"name":"Enter","arguments_map": \{\}), \{"name":"Tap","arguments_map":\{"x":"x","y":"y"\}\}]
"atomic_action_sequence": [\{"name":"Tap","arguments_map":\{"x":"x","y":"y"\},\{"name":"Type","arguments_map": \{"text","treaty"\},\{"name":"Enter","arguments_map": \{\}), \{"name":"Tap","arguments_map":\{"x":"x","y":"y"\}\}]


\{



"name": "Swipe_to_Reveal_Content",
"name": "Swipe_to_Reveal_Content",


"arguments": ["xl","yl","x2","y2"],
"arguments": ["xl","yl","x2","y2"],


"description": "Swipe from position (x1,y1) to position (x2,y2) to reveal additional content below or above on the screen.",
"description": "从位置 (x1,y1) 向位置 (x2,y2) 滑动以在屏幕上向下方或上方显示更多内容。",


"atomic_action_sequence": [\{"name":"Swipe","arguments_map":\{"x1":"xl","yl":"yl","x2":"x2","y2":"y2"\}\}]
"atomic_action_sequence": [\{"name":"Swipe","arguments_map":\{"x1":"xl","yl":"yl","x2":"x2","y2":"y2"\}\}]


"name": "Clear_Search_And_Type",
"name": "Clear_Search_And_Type",


"arguments": ["x clear","y clear","text"],
"arguments": ["x clear","y clear","text"],


"description": "Clear the current search term by tapping the 'X' icon and then type the new search term into the search bar.",
"description": "通过点击“X”图标清除当前搜索词，然后在搜索栏中输入新的搜索词。",


"precondition": "The search bar is active, and the 'X' icon to clear the current search term is visible on the screen.",
"precondition": "搜索栏处于活动状态，屏幕上可见用于清除当前搜索词的“X”图标。",


"atomic_action_sequence": [\{"name":"Tap","arguments_map":\{"x":"x_clear","y":"_clear"\}\},\{"name":"Type","arguments_map":\{"text":"text"\}\}\}
"atomic_action_sequence": [\{"name":"Tap","arguments_map":\{"x":"x_clear","y":"_clear"\}\},\{"name":"Type","arguments_map":\{"text":"text"\}\}\}


\{



"name": "Save_Note_As_File",
"name": "Save_Note_As_File",


"arguments": ["folder_x","folder y","done x","done y","save_x","save y"],
"arguments": ["folder_x","folder y","done x","done y","save_x","save y"],


"description": "Save a note as a file in a specified folder by selecting the folder, confirming the selection, and tapping the save button.",
"description": "通过选择文件夹、确认选择并点击保存按钮，将笔记保存为指定文件夹中的文件。",


"atomic_action_sequence": [\{"name":"Tap","arguments_map":\{"x":"folder_x","y":"folder_y"\},\{"name"""Tap","arguments","predone_y"\}
"atomic_action_sequence": [\{"name":"Tap","arguments_map":\{"x":"folder_x","y":"folder_y"\},\{"name"""Tap","arguments","predone_y"\}


\{"name":"Tap","arguments_map":\{"x":"save_x","y":"save_y"\} \}]
\{"name":"Tap","arguments_map":\{"x":"save_x","y":"save_y"\} \}]


"name": "Switch_App_And_Search",
"name": "Switch_App_And_Search",


"arguments": ["app_name","x","y","text"],
"arguments": ["app_name","x","y","text"],


"description": "Switch to a specified app, tap on a search bar at position (x, y), type the given text, and press Enter to perform a search.",
"description": "切换到指定应用，点击位于 (x, y) 的搜索栏，输入给定文本，然后按 Enter 执行搜索。",


"precondition": "The app to switch to is already open in the app switcher, and the search bar is visible on the screen after switching."
"precondition": "要切换的应用已在应用切换器中打开，切换后屏幕上可见搜索栏。"


"atomic action_sequence": [\{"name":"Switch_App","arguments_map":\{\}\},\{"name":"Tap","arguments_map":\{"x":"x","y","sy"]\},\{"name":"Type","arguments_map": \{"text":"text"\}\},\{"name":"Enter","arguments_map": \{ \} \}]
"atomic action_sequence": [\{"name":"Switch_App","arguments_map":\{\}\},\{"name":"Tap","arguments_map":\{"x":"x","y","sy"]\},\{"name":"Type","arguments_map": \{"text":"text"\}\},\{"name":"Enter","arguments_map": \{ \} \}]


Figure 12. Full list of Shortcuts generated by Mobile-Agent-E (with GPT-40) after self-evolution.
Figure 12. 全部快捷方式列表，由 Mobile-Agent-E（带 GPT-40）自我进化后生成。


## ** Initial Tips (User Provided) **
## ** 初始提示（用户提供） **


0. Do not add any payment information. If you are asked to sign in, ignore it or sign in as a guest if possible. Close any pop-up windows when opening an app. 1. By default, no apps are opened in the background. 2. Screenshots may show partial text in text boxes from your previous input; this does not count as an error. 3. When creating new Notes, you do not need to enter a title unless the user specifically requests it.
0. 不要添加任何支付信息。如果被要求登录，请忽略或在可能的情况下以访客身份登录。打开应用时关闭任何弹出窗口。1. 默认情况下，后台不会打开应用。2. 截图可能显示你先前输入的文本框中的部分文本；这不算错误。3. 创建新笔记时，除非用户明确要求，否则不需要输入标题。


## ** Agent Generated Tips (Scenario 1) **
## ** 代理生成的提示（情景1） **


4. When searching for restaurants or businesses, ensure the query includes specific details like location, type of cuisine, and operational hours to narrow down results effectively. 5. Always verify the operational hours of businesses to ensure they meet the user's requirements, especially for late-night or time-sensitive searches. 6. When filtering search results (e.g., by rating or distance), ensure the filter criteria are applied correctly to avoid irrelevant results. 7. Double-check the selected location or business to ensure it matches the user's requirements thoroughly. 10. If an action does not return to the expected screen, use alternative navigation methods (e.g., tapping "X" or returning to the home screen) to correct the workflow. 11. When summarizing customer feedback, include both positive and negative aspects to provide a balanced overview. 12. When retrieving contact information, ensure the details (e.g., phone number or address) are accurate and match the selected business before saving them in Nots. 13. If a task involves multiple apps (e.g., Google Maps and Notes), ensure smooth transitions between apps and verify that the required information is correctly transferred. 14. If an app fails to open or respond in the app switcher, return to the home screen and reopen the app directly to avoid delays.
4. 在搜索餐厅或商家时，确保查询包含具体信息，如位置、菜系类型和营业时间，以更有效地缩小结果范围。5. 始终核实商家的营业时间，确保符合用户的要求，尤其是深夜或时效性搜索。6. 过滤搜索结果（如按评分或距离）时，确保过滤条件正确应用，以避免无关结果。7. 反复核对所选地点或商家，确保其与用户需求完全匹配。10. 如果某个操作未返回到期望的屏幕，使用替代导航方法（例如点击“X”或返回主页）来纠正工作流程。11. 在总结客户反馈时，包含正面和负面方面，以提供平衡的概览。12. 在检索联系信息时，确保细节（如电话号码或地址）准确并与所选商家匹配，再保存到 Nots。13. 若任务涉及多个应用（如 Google 地图和笔记），确保应用之间的平滑过渡，并验证所需信息正确传输。14. 如果应用在应用切换器中无法打开或响应，请返回主页并直接重新打开应用以避免延误。


## ** Agent Generated Tips (Scenario 2) **
## ** 代理生成的提示（情景2） **


4. When identifying the most-cited paper or similar tasks, ensure to sort the results by citation count if the option is available. This minimizes manual scanning and reduces errors. 5. If a search action fails, verify the input text and ensure the correct search bar is targeted before retrying. Adjust the tap location if necessary. 6. When recording information from search results, ensure the details are accurate and clearly formatted to avoid confusion. 7. If a task involves multiple setps across different apps, always confirm the completion of one step Retry the action with slight adjustments if recessary. 9. When selecting a viteo or item from a list, ensure the title matches the intended choice to avoid selecting the orange option. 10. If a button or option does not respond to a tap, ensure it is fully visible on the screen. Use a swipe or scroll action to adjust the view of necessary before retrying. 11. When switching between apps, ensure the correct app is selected from the app switcher to avoid unnecessary navigation errors. 12. Always stop at the final screen requested by the user, ensuring the task is fully completed before ending the interaction.
4. 在识别最常被引用的论文或类似任务时，如有选项，请确保按引用次数排序结果。这可最小化人工扫描并减少错误。5. 如果搜索操作失败，核实输入文本并确保定位正确的搜索栏后重试。如有必要，调整点击位置。6. 记录搜索结果信息时，确保细节准确且格式清晰，避免混淆。7. 若任务涉及跨多个应用的多步骤操作，始终确认一个步骤完成后再重试，必要时略微调整后重试。9. 选取视频或列表项时，确保标题与目标选择匹配，避免选中错误项。10. 如果按钮或选项对点击无反应，确保其在屏幕上完全可见。可使用滑动或滚动调整视图后再重试。11. 在切换应用时，确保从应用切换器中选中正确的应用，以避免不必要的导航错误。12. 始终在用户请求的最终屏幕停止，确保任务完全完成后再结束交互。


## ** Agent Generated Tips (Scenario 3) **
## ** 代理生成的提示（情景3） **


4. When identifying the best deal prioritize both price and features, and ensure any discounts or promotions are clearly noted. 5. Always confirm that the displayed product matches the search criteria (e.g., size, specifications) to avoid selecting an incorrect item. 6. If the task requires stopping at a specific screen, ensure the screen is fully loaded and all relevant details are visible before soppings. 7. If a filter does not apply correctly, try adjusting it again by swiping or rapping alternative areas of the screen to reveal insiden options. 8. When using siliers for filters (e.g., price ange), swirping is often more effective than tapping to adjust the values. 9. If a filter unexpectedly resets or renders itself, reapply the varity the results ensure that the product model and specifications (e.g., size, features) are identical to avoid inaccurate comparisons. 12. If swipping to reveal content, ensure the swipe is smooth and covers enough distance to load all relevant details on the screen. 13. If an app fails to open or navigate correctly, return to the home screen and retry the action. This often resolves navigation issues. 14. If a tap action does not work as expected, consider tapping alternative areas of the screen, such as associated buttons or options, to achieve the desired outcome. 15. When switching between apps, ensure the correct app is reopened and verify the screen before proceeding to avoid unnecessary repetition.
在识别最佳交易时，优先考虑价格和功能，并确保清晰标注任何折扣或促销信息。始终确认所显示的产品与搜索条件（例如尺寸、规格）一致，以避免选择错误的商品。若任务需要在特定屏幕停止，请确保该屏幕已完全加载，所有相关细节均可见后再进行操作。若筛选条件未正确应用，请通过滑动或点击屏幕其他区域来调整，以显示隐藏的选项。使用筛选器时（如价格区间），滑动通常比点击更能有效地调整数值。若筛选条件意外重置或自行改变，请重新应用以确保结果的多样性，确保产品型号与规格（如尺寸、特征）一致，避免产生不准确的对比。若需要滑动以显示内容，请确保滑动平滑且距离足够以加载屏幕上的所有相关细节。若应用程序无法正确打开或导航，请返回主页再重试此操作，常能解决导航问题。若某次点击未按预期工作，请尝试点击屏幕上的其他区域，如相关按钮或选项，以达到期望结果。切换应用时，请确保重新打开正确的应用并在继续前核对屏幕，以避免不必要的重复。


## ** Agent Generated Tips (Scenario 4) **
## ** Agent Generated Tips (Scenario 4) **


4. When navigating apps, ensure that the correct icon is tapped by carefully identifying its position and function to avoid misalignment or unintended actions. 5. If a search filter is applied uninventionally, clear it by tapping the "X" icon in the search bar before proceeding with a new search. 6. A laws serity the context of the search results to ensure they align with the intended query before summarizing or proceeding to the next step. 7. When recording information in Notes, ensure the formatting is clear and consistent for easy readability. 8. Double-check the accuracy of the recorded information (e.g., account names, titles) before saving the note to avoid errors. 9. If redirected to an unintended page (e.g., "My Orders"), navigate back to the main interface or intended section before proceeding. 10. When comparing multiple items (e.g., movie ratings), keep track of all relevant data to ensure accurate the correct app to avoid confusion. 12. When entering search terms, ensure the previous query is cleared completely to prevent appending incorrect text to the new query. 13. If a misaligned tap opens an unintended menu (e.g., Fitters), close it immediately and retry the intended action. 14. Use broader search terms if specific queries fail to yield results, and refine the search gradually based on the context. 15. If an app fails to execute a search or action, consider switching to a browser or alternative apple to complete the task.
在浏览应用时，务必通过仔细确认其位置与功能来确保点击的是正确图标，避免错位或非预期操作。若搜索筛选器应用不当，请在继续新搜索前点击搜索栏中的“X”图标清除筛选。应在总结或进入下一步前，确保搜索结果的上下文与所述查询相符。记笔记时，请确保格式清晰、统一，便于阅读。再保存笔记前，请再次核对所记录的信息的准确性（如账户名、标题）。若跳转到错误页面（如“我的订单”），在继续前返回到主界面或目标栏目。比较多项内容时（如电影评分），请保留所有相关数据，确保应用正确无误，避免混淆。进入搜索词时，请确保完全清除上一次查询，以防把错误文本追加到新查询中。若误导页导致的点击打开了错误菜单（如“筛选器”），请立即关闭并重试正确操作。若具体查询未能返回结果，请使用更广泛的搜索词，并根据上下文逐步缩小范围。若应用无法执行搜索或操作，请考虑切换到浏览器或其他应用完成任务。


## ** Agent Generated Tips (Scenario 5) **
## ** Agent Generated Tips (Scenario 5) **


4. Always confirm that the displayed results match the search criteria (i.e., correct cities, dates, and round-trip selection) before proceeding to the next step. 5. If multiple options are displayed, ensure the cheapest or most relevant option is clearly identified and selected as per the task requirements. 6. If a "Back" button fails to function as expected, consider alternative methods to save or exit, such as using a menu or additional options (e.g., "Save as file"). 7. When saving a note as a file, ensure the correct folder and file format are selected before confirming the save. 8. Double-check that the task is fully completed (e.g., the note is saved in the correct location) before marking it as done. 9. If scrolling through content does not reveal new information, consider alternative methods to locate the required details, such as using a search or filter function within the app. 10. If the end of a section is reached and the required information is not found, reasses the search criteria or explor other sections of the app for relevant details. I. Unken searching for specific items (e.g., dishes, anematties), use keywords of filters to narrow down results and save time. 12. If repetitive actions (e.g., swipping) fail to yield results, pass each evaluate whether the task can be completed using a different approach or if the information is unavailable. 13. When switching between apps, ensure that the context of the task is maintained, and verify that the information gathered in one app aligns with the requirements in the other app. 14. Always confirm the proximity or location details (e.g., using Google Maps) before finalizing a selection, especially when location is a key criterion.
在展示结果前，始终确认显示的结果符合搜索条件（即城市、日期及往返选项正确）。若显示多项选项，应明确识别并选取最便宜或最相关的选项以符合任务要求。若“返回”按钮无法按预期工作，请考虑使用其他方式保存或退出，例如使用菜单或“另存为文件”等选项。保存为文件时，请确保已选择正确的文件夹与文件格式。再次确认任务已完全完成（如笔记已保存到正确位置）后再标记为完成。若滚动内容未显示新信息，请考虑使用其他方法查找所需细节，如应用内的搜索或筛选功能。若到达某部分末尾且未找到所需信息，请重新评估搜索条件或探索应用的其他部分以获取相关细节。I. 针对特定项的搜索（如菜肴、种类），请使用筛选条件的关键词来缩小结果并节省时间。若重复操作（如滑动）未产生结果，请评估任务是否可通过不同方法完成，或信息是否不可用。切换应用时，请确保任务上下文保持一致，并核实在一个应用中获取的信息与另一应用的要求相符。最终敲定选择前，始终确认附近位置或地理信息（如使用谷歌地图）等细节，尤其当位置是关键标准时。