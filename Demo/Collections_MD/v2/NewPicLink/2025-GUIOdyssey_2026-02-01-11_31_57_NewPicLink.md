# GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices
# GUIOdyssey: 面向移动设备跨应用 GUI 导航的综合数据集


Quanfeng ${\mathrm{{Lu}}}^{4,1}$ ,Wenqi Shao ${}^{1, \dagger  }$ ,Zitao Liu ${}^{5}$ ,Lingxiao ${\mathrm{{Du}}}^{3,1}$ ,Fanqing Meng ${}^{4,1}$ , Boxuan ${\mathrm{{Li}}}^{3}$ ,Botong Chen ${}^{5}$ ,Siyuan Huang ${}^{4,1}$ ,Kaipeng Zhang ${}^{1}$ ,Ping Luo ${}^{2, \dagger  } \; {}^{1}$ Shanghai AI Laboratory ${}^{2}$ The University of Hong Kong ${}^{3}$ Nanjing University ${}^{4}$ Shanghai Jiao Tong University ${}^{5}$ Harbin Institute of Technology,Shenzhen
Quanfeng ${\mathrm{{Lu}}}^{4,1}$ ,Wenqi Shao ${}^{1, \dagger  }$ ,Zitao Liu ${}^{5}$ ,Lingxiao ${\mathrm{{Du}}}^{3,1}$ ,Fanqing Meng ${}^{4,1}$ , Boxuan ${\mathrm{{Li}}}^{3}$ ,Botong Chen ${}^{5}$ ,Siyuan Huang ${}^{4,1}$ ,Kaipeng Zhang ${}^{1}$ ,Ping Luo ${}^{2, \dagger  } \; {}^{1}$ 上海人工智能实验室 ${}^{2}$ 香港大学 ${}^{3}$ 南京大学 ${}^{4}$ 上海交通大学 ${}^{5}$ 哈尔滨工业大学（深圳）


shaowenqi@pjlab.org.cn, pluo@cs.hku.hk
shaowenqi@pjlab.org.cn, pluo@cs.hku.hk


https://github.com/OpenGVLab/GUI-Odyssey



## Abstract
## 摘要


Autonomous Graphical User Interface (GUI) navigation agents can enhance user experience in communication, entertainment, and productivity by streamlining workflows and reducing manual intervention. However, prior GUI agents often trained with datasets comprising tasks that can be completed within a single app, leading to poor performance in cross-app navigation. To address this problem, we present GUIOdyssey, a comprehensive dataset for cross-app mobile GUI navigation. GUIOdyssey comprises 8,334 episodes with an average of 15.3 steps per episode, covering 6 mobile devices, 212 distinct apps, and 1,357 app combinations. Each step is enriched with detailed semantic reasoning annotations, which aid the model in building cognitive processes and enhancing its reasoning abilities for complex cross-app tasks. Building on GUIOdyssey, we develop OdysseyAgent, an exploratory multimodal agent for long-step cross-app navigation equipped with a history re-sampler module that efficiently attends to historical screen-shot tokens, balancing performance and inference speed. Extensive experiments conducted in both in-domain and out-of-domain scenarios validate the effectiveness of our approach. Moreover, we demonstrate that historial information involving actions, screenshots and context in our dataset can significantly enhances OdysseyAgent's performance on complex cross-app tasks.
自主图形用户界面（GUI）导航代理可以通过简化工作流、减少人工干预来提升沟通、娱乐和生产力领域的用户体验。然而，现有 GUI 代理往往基于仅能在单一应用内完成的任务数据集进行训练，导致跨应用导航的性能较差。为解决该问题，我们提出 GUIOdyssey——一个面向跨应用的移动 GUI 导航综合数据集。GUIOdyssey 包含 8,334 条 episode，平均每条 15.3 步，覆盖 6 种移动设备、212 个不同应用和 1,357 种应用组合。每一步都附有详细的语义推理注释，帮助模型建立认知过程并提升对复杂跨应用任务的推理能力。基于 GUIOdyssey，我们开发了 OdysseyAgent，一种带有历史重新采样模块的探索性多模态代理，用于长步跨应用导航，能够高效关注历史屏幕截图令牌，在性能和推理速度之间取得平衡。在内域和跨域场景下进行的广泛实验验证了我们方法的有效性。此外，我们还证明在数据集中涉及动作、截图和上下文的历史信息可以显著提升 OdysseyAgent 在复杂跨应用任务上的表现。


## 1. Introduction
## 1. 介绍


Smartphones have become indispensable tools in our daily lives [6]. With a growing number of mobile applications, users frequently navigate across multiple apps to complete tasks, such as sharing content between social media platforms or coordinating schedules between messaging apps and calendars. Introducing a smart assistant to streamline these workflows and reduce manual intervention would be highly beneficial, particularly for individuals with physical disabilities [35]. Nowadays, the rapid advancement of large foundation model $\left\lbrack  {1,2,{12},{18},{50}}\right\rbrack$ has enabled the development of intelligent agents $\left\lbrack  {1,{40},{51},{69}}\right\rbrack$ . These agents process environmental observations, maintain multi-turn context, and execute actions to achieve specific goals, making autonomous GUI navigation increasingly feasible and practical.
智能手机已成为我们日常生活中不可或缺的工具 [6]。随着移动应用数量的不断增加，用户经常跨越多个应用完成任务，例如在社媒平台之间分享内容，或在消息应用和日历之间协调日程。引入一个智能助手来简化这些工作流并减少人工干预将非常有利，尤其是对身体残障人士 [35]。如今，大型基础模型 $\left\lbrack  {1,2,{12},{18},{50}}\right\rbrack$ 的快速进展使开发智能代理 $\left\lbrack  {1,{40},{51},{69}}\right\rbrack$ 成为可能。这些代理处理环境观测、维护多轮上下文并执行行动以实现特定目标，使自动化 GUI 导航变得越来越可行和实用。


While current foundation models are yet fully capable across various domains [62], they can still be effectively leveraged through GUI navigation datasets to build GUI agents that deliver more efficient and user-friendly mobile experiences. For instance, AITW [38] constructs a dataset encompassing various tasks to develop generalist agents for smartphones using large language models (LLMs) [16]. Similarly, AndroidControl [26] introduces a dataset focused on everyday tasks involving Android apps, providing both high-level and low-level instructions for GUI agents. However, these datasets primarily comprise operational actions, such as 'click' and 'scroll'. Furthermore, existing mobile GUI navigation datasets predominantly focus on tasks solvable within a single app, as depicted in Fig. 1(a). However, many real-world tasks require cross-app navigation, involving the transfer of context and data among multiple apps, as shown in Fig. 1(b). These complex work-flows cannot be fully captured by single-app datasets, nor can they be decomposed without losing critical cross-app interactions. While some studies have investigated cross-app tasks, their focus has been limited to evaluation purposes [28, 39, 48, 55]. In particular, evaluations from studies $\left\lbrack  {{54},{55}}\right\rbrack$ reveal that current performance on cross-app tasks remains significantly worse than on single-app tasks. Therefore, it is crucial to develop dedicated datasets to improve the cross-app navigation capabilities of GUI agents.
尽管当前的基础模型在各个领域尚未完全成熟 [62]，但通过 GUI 导航数据集仍可被有效利用来构建提供更高效、用户友好移动体验的 GUI 代理。例如，AITW [38] 构建了一个涵盖多种任务的数据集，以使用大型语言模型（LLMs）在智能手机上开发通用代理 [16]。同样，AndroidControl [26] 引入了一个聚焦于 Android 应用日常任务的数据集，为 GUI 代理提供高层和低层指令。然而，这些数据集主要包含操作性动作，如“点击”和“滚动”。此外，现有移动 GUI 导航数据集大多聚焦于可在单一应用内解决的任务，如图 1(a) 所示。然而，许多现实世界的任务需要跨应用导航，涉及在多个应用之间传递上下文和数据，如图 1(b) 所示。这些复杂的工作流不能被单应用数据集完整捕获，也无法在不丢失关键跨应用交互的情况下进行分解。尽管有些研究已探讨跨应用任务，但其重点通常限于评估 [28, 39, 48, 55]。特别是，来自研究 $\left\lbrack  {{54},{55}}\right\rbrack$ 的评估显示当前跨应用任务的表现仍显著差于单一应用任务。因此，开发专门的数据集以提升 GUI 代理的跨应用导航能力至关重要。


---



${}^{ \dagger  }$ Corresponding author.
${}^{ \dagger  }$ 通信作者。


---



<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_19_980f15.jpg"/>



Figure 1. Illustration of single-app (a) and cross-app (b) GUI navigation. We see that cross-app navigation tasks demand the integration of multiple apps and the transfer of context and data between them, involving more complex workflows than single-app navigation.
图1。这是单应用（a）与跨应用（b）GUI导航的示意。我们看到跨应用导航任务需要整合多个应用并在它们之间传递上下文与数据，涉及比单应用导航更为复杂的工作流。


To address this issue and advance the development of general GUI agents, we introduce GUIOdyssey, the first cross-app GUI navigation dataset for mobile devices, featuring task instructions designed to reflect two levels of granularity. High-level instructions emulate natural human requests to capture real-world needs, whereas low-level instructions correspond to fine-grained tasks, providing precise and unambiguous guidance to eliminate potential misunderstandings. On one hand, we propose high-level [17, 38] cross-app navigation instructions by brainstorming with human participants and GPT-4 [1], and create episode-specific user instructions to enrich task diversity. Independent annotators are then employed to annotate the entire navigation process comprehensively, including screenshots and corresponding actions,using an Android emulator ${}^{1}$ . On the other hand, after collecting the human-annotated data, we use GPT-40 [36] to generate low-level instructions [4, 14] for each step, providing a more fine-grained guide to task completion and facilitating deeper exploration of the GUI agent's potential for cross-app tasks.
为解决这一问题并推动通用GUI智能体的发展，我们推出 GUIOdyssey，这是首个用于移动设备的跨应用GUI导航数据集，包含旨在反映两级粒度的任务指令。高层指令模仿自然的人类请求，以捕捉现实世界需求；低层指令对应细粒度任务，提供精确且明确的指导，消除潜在误解。一方面，我们通过与人类参与者和 GPT-4 [1] 的头脑风暴提出高层次的 [17, 38] 跨应用导航指令，并创建逐集的用户指令以丰富任务多样性。随后由独立标注人员对整个导航过程进行全面标注，包括截图与相应操作，使用 Android 模拟器 ${}^{1}$。另一方面，在收集人类标注数据后，我们使用 GPT-40 [36] 为每一步生成低层指令 [4, 14]，提供更细粒度的任务完成指引，方便对 GUI智能体在跨应用任务中的潜力进行更深入的探索。


To simulate the human approach, we further enrich GUIOdyssey with semantic annotations by breaking down each step into three components: screen comprehension, historical context review, and decision-making reasoning. GPT-40 [36] is employed to generate semantic annotations for these components. Subsequently, GUIOdyssey undergoes a quality check to ensure screenshot integrity, action accuracy, and alignment between GPT-4-generated instructions and the originals. After construction through this rigorous pipeline, designed to enhance task diversity and annotation quality, GUIOdyssey comprises 8, 334 episodes with an average of 15.3 steps, meticulously curated from 6 different mobile devices such as Pixel Pro and Tablet. It features 6 types of cross-app navigation tasks, ranging from general system tool use to media entertainment, involving navigation across 212 different apps and 1,357 app combinations in fields such as video, music, and reading, as depicted in Fig. 2. Table 1 presents a comparison between GUIOdyssey and previous datasets.
为模拟人类方法，我们进一步通过将每一步拆分为屏幕理解、历史上下文回顾和决策推理三个组成部分来丰富 GUIOdyssey 的语义标注。使用 GPT-40 [36] 为这些组成部分生成语义标注。随后，GUIOdyssey 经过质量检查，确保截图完整性、操作准确性，以及 GPT-4 生成的指令与原始指令的一致性。经这一严格流程构建，旨在提升任务多样性和标注质量，GUIOdyssey 现有 8,334 条剧集，平均 15.3 步，来自 Pixel Pro、Tablet 等 6 种不同移动设备的精心筛选。它包含 6 种跨应用导航任务类型，涵盖从一般系统工具使用到媒体娱乐等场景，涉及跨 212 个不同应用和 1,357 对应用组合，覆盖视频、音乐、阅读等领域，如图 2 所示。表 1 对 GUIOdyssey 与以往数据集进行了对比。


Leveraging GUIOdyssey, we develop an exploratory cross-app multimodal agent named OdysseyAgent. Cross-app navigation tasks inherently involve long step sequences, requiring to retain numerous screenshots and actions for informed decision-making. However, processing large numbers of screenshot tokens can significantly slow inference, which is a critical concern for GUI agents frequently interacting with users. To balance performance with speed, OdysseyAgent incorporates a history resampler module that selectively attends to historical screenshot tokens while maintaining high inference throughput, thereby effectively and efficiently tackling complex cross-app tasks. We thoroughly validate our approach on GUIOdyssey in both in-domain and out-of-domain scenarios. OdysseyA-gent achieves highest accuracy among existing methods, including Claude3.5-Sonnet and GPT-4o. Moreover, incorporating semantic annotations leads to further performance gains. We also conduct an in-depth analysis demonstrating that enriching historical information with actions, screenshots, and contextual information significantly improves OdysseyAgent's performance, highlighting the importance of comprehensively modeling historical information for complex cross-app navigation tasks.
在 GUIOdyssey 的基础上，我们开发了一个名为 OdysseyAgent 的探索性跨应用多模态智能体。跨应用导航任务本质上涉及较长的步骤序列，需要保留大量截图与操作以供知情决策。然 而，处理大量截图标记符可能显著降低推理速度，这对经常与用户互动的 GUI 智能体来说是一个关键问题。为在性能与速度之间取得平衡，OdysseyAgent 引入了历史采样模块，选择性关注历史截图标记，同时维持高推理吞吐量，从而有效高效地应对复杂的跨应用任务。我们在 GUIOdyssey 的在域内与跨域场景中对该方法进行了充分验证。OdysseyAgent 在现有方法（包括 Claude3.5-Sonnet 与 GPT-4o）中取得最高准确率。此外，加入语义标注还带来进一步的性能提升。我们还进行了深入分析，证明通过用操作、截图和上下文信息丰富历史信息，显著提升 OdysseyAgent 的性能，强调对复杂跨应用导航任务全面建模历史信息的重要性。


The contributions of this work are three-fold. 1) We introduce GUIOdyssey, a comprehensive dataset for cross-app mobile GUI navigation, comprising 8, 334 episodes with an average length of 15.3 steps. It covers a wide range of apps, tasks, and devices, with each step annotated by rich semantic reasoning to facilitate cognitive processes and enhance reasoning capabilities, thereby boosting performance on complex cross-app tasks. 2) We propose OdysseyAgent, an exploratory multimodal agent equipped with a history resampler module that balances performance and inference speed for cross-app navigation. 3) Through extensive experiments with OdysseyAgent, we demonstrate that comprehensively leveraging historical information substantially enhances performance on cross-app navigation tasks, highlighting the importance of historical information modeling.
本工作的贡献有三方面。1）我们推出 GUIOdyssey，这是一个面向跨应用移动GUI导航的综合数据集，共 8,334 条剧集，平均长度 15.3 步。它覆盖广泛的应用、任务与设备，每一步都带有丰富的语义推理标注，以促进认知过程并增强推理能力，从而提升在复杂跨应用任务中的性能。2）我们提出 OdysseyAgent，是一个具备历史采样模块的探索性多模态智能体，在跨应用导航中实现性能与推理速度的平衡。3）通过对 OdysseyAgent 的广泛实验，我们证明全面利用历史信息能显著提升跨应用导航任务的性能，凸显历史信息建模的重要性。


---



${}^{1}$ https://developer.android.com/studio
${}^{1}$ https://developer.android.com/studio


---



## 2. Related Work
## 2. 相关工作


GUI Navigation Agent. Large foundation models [1, 2, ${12},{30},{46},{50},{56}\rbrack$ have recently demonstrated the capacity to utilize extensive world knowledge to solve complex autonomous tasks [34, 41, 43, 59, 61]. These advancements have paved the way for the development of GUI agents capable of autonomous device control. For instance, works such as $\left\lbrack  {{17},{17},{21},{67}}\right\rbrack$ focus on autonomous agents in the Web domain, while studies like [15, 28, 48, 49, 57] leverage powerful language models, such as GPT-4V [1], to address GUI navigation tasks on mobile devices. Additionally, other research [52, 65] explores the potential applications of OS-specific agents. This line of research often incorporates supplementary inputs, such as accessibility trees (A11y trees), to provide details like UI element coordinates or utilizes the Set-Of-Marks [58] strategy to outline bounding boxes of UI elements, supported by GUI-specific grounding models [20, 33]. An alternative approach, as exemplified by [9, 14, 19, 23, 29, 42, 53, 63, 66], employs a coordinate-based method combined with visual models to develop GUI navigation agents. This approach directly provides positional information for executing actions, without relying on additional information. While coordinate-based navigation can be fragile and may underperform in certain scenarios, it represents the ultimate solution for GUI navigation in the long run [54]. In cases where structured A11y trees are unavailable or impractical [14, 42, 54], coordinate-based navigation offers a natural and straightforward solution that enhances task and device transferability [11]. GUIOdyssey specifically adopts coordinate-based methods, aiming to create versatile, general-purpose GUI agents.
GUI 导航代理。大型基础模型 [1, 2, ${12},{30},{46},{50},{56}\rbrack$ 最近已经展示出利用广泛世界知识来解决复杂自主管理任务的能力 [34, 41, 43, 59, 61]。这些进展为开发能够自主控制设备的 GUI 代理铺平了道路。例如，像 $\left\lbrack  {{17},{17},{21},{67}}\right\rbrack$ 这样的工作专注于 Web 领域的自主管理代理，而像 [15, 28, 49, 57] 这样的研究则利用强大的语言模型，如 GPT-4V [1]，来解决移动设备上的 GUI 导航任务。此外，其他研究 [52, 65] 探索了 OS 专用代理的潜在应用。这一研究方向常常结合辅助输入，如无障碍树（A11y 树），以提供 UI 元素坐标等细节，或利用 Set-Of-Marks [58] 策略来勾勒 UI 元素的边界框，并由 GUI 专用的着陆模型 [20, 33] 支持。另一种做法，如 [9, 14, 19, 23, 29, 42, 53, 63, 66] 所示，采用坐标基方法结合视觉模型来开发 GUI 导航代理。这种方法直接提供执行操作的位置信息，而无需额外信息。尽管基于坐标的导航可能脆弱，在某些场景下表现不佳，但从长远来看它代表了 GUI 导航的终极解决方案 [54]。在结构化的 A11y 树不可用或不可行的情况下 [14, 42, 54]，坐标基导航提供了一种自然且直接的解决方案，提升了任务和设备的可迁移性 [11]。GUIOdyssey 具体采用坐标基方法，旨在创建多功能、通用的 GUI 代理。


Benchmarks and Datasets for GUI Agents. Numerous benchmarks and datasets have been proposed to advance research in GUI navigation. Interactive online environments $\left\lbrack  {{22},{25},{39},{44},{54},{55},{60},{68}}\right\rbrack$ evaluate agents’ GUI navigation capabilities, while other datasets [3, 4, 10, 14, 31, 63] primarily enhance UI perception and comprehension. Recent GUI datasets $\left\lbrack  {8,9,{11},{17},{24},{26},{27},{32},{38},{45},{47},{66}}\right\rbrack$ predominantly involve tasks confined to a single app. However, real-world usage frequently requires navigation across multiple apps, significantly increasing complexity. Cross-app tasks typically require longer action sequences (see Table 1), leading to higher error propagation risks. Additionally, cross-app interactions necessitate managing diverse working memory since key UI elements and contextual information span multiple apps. Furthermore, these tasks demand broader functional knowledge to integrate distinct interaction types like file sharing, email composition, and messaging. Additional examples of cross-app tasks are provided in Appendix Sec. 8.4. To address these challenges, we introduce GUIOdyssey, the first comprehensive cross-app GUI navigation dataset. A detailed comparison between GUIOdyssey and prior datasets is presented in Table 1.
用于 GUI 代理的基准测试与数据集。为推进 GUI 导航研究，提出了大量基准测试和数据集。交互式在线环境 $\left\lbrack  {{22},{25},{39},{44},{54},{55},{60},{68}}\right\rbrack$ 评估代理的 GUI 导航能力，而其他数据集 [3, 4, 10, 14, 31, 63] 主要提升 UI 感知与理解。最近的 GUI 数据集 $\left\lbrack  {8,9,{11},{17},{24},{26},{27},{32},{38},{45},{47},{66}}\right\rbrack$ 主要涉及限定在单一应用中的任务。然而，现实使用常需要跨越多个应用进行导航，显著增加了复杂度。跨应用任务通常需要更长的动作序列（见表 1），导致更高的错误传播风险。此外，跨应用交互需要管理多样的工作记忆，因为关键 UI 元素和上下文信息跨越多个应用。此外，这些任务需要更广泛的功能性知识，以整合诸如文件共享、邮件撰写和消息传递等不同交互类型。附加的跨应用任务示例见附录 Sec. 8.4。为应对这些挑战，我们推出 GUIOdyssey —— 首个全面的跨应用 GUI 导航数据集。表 1 中给出 GUIOdyssey 与之前数据集的详细对比。


## 3. GUIOdyssey Dataset
## 3. GUIOdyssey 数据集


This section introduces the proposed cross-app navigation dataset. We present the metadata definition in Sec. 3.1, details in data collection in Sec. 3.2, and dataset statistics in Sec. 3.3, respectively. The dataset overview is shown in Fig. 2 and the collection process is presented in Fig. 3.
本节介绍提出的跨应用导航数据集。我们在 3.1 节给出元数据定义，在 3.2 节给出数据收集细节，在 3.3 节给出数据集统计信息。数据集概览如图 2 所示，收集过程如图 3 所示。


### 3.1. Metadata Definition
### 3.1. 元数据定义


GUI Episode. A GUI episode is a recorded sequence of interactions capturing the action steps to complete the navigation task from the user's high-level instruction. Formally,given the user's high-level instruction ${I}_{\text{ user }}$ and the screenshot ${X}^{t}$ at the time step $t$ ,the GUI Agent $\mathcal{G}$ will take the action ${A}^{t} = \mathcal{G}\left( {{X}^{t},{I}_{\text{ user }}}\right)$ to complete this instruction. When the task is completed, the episode is defined as the sequence including all screenshots and actions denoted as $E = \left\{  {{\left( {X}^{t},{A}^{t}\right) }_{t = 1}^{T},{I}_{\text{ user }}}\right\}$ where $T$ indicates the total steps. An example of the episode is illustrated in Fig. 3. Note that the total step $T$ of cross-app navigation is much larger than that of single-app navigation as shown in Fig. 1.
GUI Episode. GUI 片段是一系列互动的记录，用以捕捉从用户的高层指令完成导航任务的行动步骤。正式地，给定用户的高层指令 ${I}_{\text{ user }}$ 和时间步 $t$ 的截图 ${X}^{t}$，GUI Agent $\mathcal{G}$ 将采取动作 ${A}^{t} = \mathcal{G}\left( {{X}^{t},{I}_{\text{ user }}}\right)$ 来完成此指令。当任务完成时，片段被定义为包含所有截图和动作的序列，记为 $E = \left\{  {{\left( {X}^{t},{A}^{t}\right) }_{t = 1}^{T},{I}_{\text{ user }}}\right\}$，其中 $T$ 表示总步骤数。图 3 展示了片段的一个例子。注意，与单应用导航相比，跨应用导航的总步骤 $T$ 明显更大，如图 1 所示。


Action Set. The action set of GUIOdyssey comprises 9 kinds of actions: CLICK, SCROLL, LONG PRESS, TYPE, COMPLETE, IMPOSSIBLE, HOME, BACK, and RECENT. The arguments and functionalities of these actions are summarized in Table 5 of Appendix Sec. 8.2.
动作集合。GUIOdyssey 的动作集合包含 9 种动作：CLICK、SCROLL、LONG PRESS、TYPE、COMPLETE、IMPOSSIBLE、HOME、BACK、RECENT。关于这些动作的参数与功能，在附录 Sec. 8.2 的表格 5 中有摘要。


### 3.2. Data Collection
### 3.2. 数据收集


Cross-app Task Proposal. As depicted in Fig. 2, GUIOdyssey comprises six types of cross-app navigation tasks: 1) General Tool, which includes tasks that entail system-wide operations. 2) Information Management. It encompasses the activities of searching for and recording information for future utilization. 3) Web Shopping. Shopping encompass a variety of activities associated with online product purchases. 4) Media Entertainment, which revolves around engaging in activities related to video and music streaming applications. 5) Social Sharing encompasses activities where users share content across various social media platforms, and 6) Multi-Apps, which involve more complex operations across different domains. See Appendix Sec. 8.1 for details on these tasks.
跨应用任务提案。如图 2 所示，GUIOdyssey 包含六种跨应用导航任务：1) 通用工具，包含需要系统范围操作的任务。2) 信息管理，涵盖信息的搜索与记录以便将来使用。3) Web 购物，购物涉及与在线产品购买相关的各种活动。4) 媒体娱乐，围绕视频与音乐流媒体应用的相关活动。5) 社交分享，用户在各种社交平台上分享内容的活动；6) 多应用，涉及跨不同领域的更复杂操作。关于这些任务的细节，请参阅附录 Sec. 8.1。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_19_e58500.jpg"/>



Figure 2. An overview of the proposed GUIOdessey. It encompasses 6 types of cross-app navigation tasks spanning 212 unique apps and 1, 357 combos of multiple apps from 6 different devices.
图 2. 提议的 GUIOdessey 概览。它涵盖 6 种跨应用导航任务，跨越 212 款独立应用及来自 6 个不同设备的 1,357 组多应用组合。


Table 1. GUI navigation dataset comparison. GUIOdyssey is a comprehensive cross-app GUI navigation dataset with over 8k+ episodes, featuring an average of 15.3 steps, which is the longest among mobile GUI datasets, and includes a diverse range of devices such as tablets.
表 1. GUI 导航数据集比较。GUIOdyssey 是一个全面的跨应用 GUI 导航数据集，包含超过 8k 条目，平均 15.3 步，是移动 GUI 数据集中的最长之一，且涵盖包括平板在内的多种设备。


<table><tr><td>Dataset</td><td>#Episodes</td><td>#Unique ${I}_{\text{ user }}$</td><td>#Avg. Steps</td><td>Cross-app?</td><td>Platform</td><td>#Domains</td><td>Instruction Level</td><td>Semantic Annotation</td></tr><tr><td>Mind2Web [17]</td><td>2,350</td><td>2,350</td><td>7.3</td><td>✘</td><td>Web</td><td>137 sites</td><td>high</td><td>✘</td></tr><tr><td>WebLINX [32]</td><td>2,337</td><td>2,337</td><td>43.0</td><td>✘</td><td>Web</td><td>155 sites</td><td>high</td><td>✘</td></tr><tr><td>PixelHelp [27]</td><td>187</td><td>187</td><td>4.2</td><td>✘</td><td>Phone</td><td>4 apps</td><td>high & low</td><td>✘</td></tr><tr><td>MoTIF [8]</td><td>4,707</td><td>276</td><td>4.5</td><td>✘</td><td>Phone</td><td>125 apps</td><td>high & low</td><td>✘</td></tr><tr><td>UGIF [47]</td><td>523</td><td>480</td><td>5.3</td><td>✘</td><td>Phone</td><td>12 apps</td><td>high & low</td><td>✓</td></tr><tr><td>Meta-GUI</td><td>4.684</td><td>1,125</td><td>5.3</td><td>✘</td><td>Phone</td><td>11 apps</td><td>high</td><td>✘</td></tr><tr><td>AITW [66]</td><td>715,142</td><td>30.378</td><td>6.5</td><td>✘</td><td>Phone</td><td>159 apps, 198+ sites</td><td>high</td><td>✘</td></tr><tr><td>AITZ [66]</td><td>2,504</td><td>2.504</td><td>7.5</td><td>✘</td><td>Phone</td><td>70+ apps</td><td>high</td><td>✓</td></tr><tr><td>AndroidControl [26]</td><td>15,283</td><td>14.548</td><td>5.5</td><td>✘</td><td>Phone</td><td>833 apps</td><td>high & low</td><td>✘</td></tr><tr><td>AMEX [9]</td><td>2,946</td><td>2,946</td><td>12.8</td><td>✘</td><td>Phone</td><td>110 apps</td><td>high</td><td>✓</td></tr><tr><td>GUIOdyssey</td><td>8,334</td><td>8,334</td><td>15.3</td><td>✓</td><td>Phone & Tablet</td><td>212 apps, 1,357 app combos</td><td>high & low</td><td>✓</td></tr></table>
<table><tbody><tr><td>数据集</td><td>#剧集</td><td>#唯一 ${I}_{\text{ user }}$</td><td>#平均步骤</td><td>跨应用？</td><td>平台</td><td>#领域</td><td>指令等级</td><td>语义注释</td></tr><tr><td>Mind2Web [17]</td><td>2,350</td><td>2,350</td><td>7.3</td><td>✘</td><td>网络</td><td>137 个站点</td><td>高</td><td>✘</td></tr><tr><td>WebLINX [32]</td><td>2,337</td><td>2,337</td><td>43.0</td><td>✘</td><td>网络</td><td>155 个站点</td><td>高</td><td>✘</td></tr><tr><td>PixelHelp [27]</td><td>187</td><td>187</td><td>4.2</td><td>✘</td><td>手机</td><td>4 个应用</td><td>高与低</td><td>✘</td></tr><tr><td>MoTIF [8]</td><td>4,707</td><td>276</td><td>4.5</td><td>✘</td><td>手机</td><td>125 个应用</td><td>高与低</td><td>✘</td></tr><tr><td>UGIF [47]</td><td>523</td><td>480</td><td>5.3</td><td>✘</td><td>手机</td><td>12 个应用</td><td>高与低</td><td>✓</td></tr><tr><td>Meta-GUI</td><td>4.684</td><td>1,125</td><td>5.3</td><td>✘</td><td>手机</td><td>11 个应用</td><td>高</td><td>✘</td></tr><tr><td>AITW [66]</td><td>715,142</td><td>30.378</td><td>6.5</td><td>✘</td><td>手机</td><td>159 个应用，198+ 个站点</td><td>高</td><td>✘</td></tr><tr><td>AITZ [66]</td><td>2,504</td><td>2.504</td><td>7.5</td><td>✘</td><td>手机</td><td>70+ 个应用</td><td>高</td><td>✓</td></tr><tr><td>AndroidControl [26]</td><td>15,283</td><td>14.548</td><td>5.5</td><td>✘</td><td>手机</td><td>833 个应用</td><td>高与低</td><td>✘</td></tr><tr><td>AMEX [9]</td><td>2,946</td><td>2,946</td><td>12.8</td><td>✘</td><td>手机</td><td>110 个应用</td><td>高</td><td>✓</td></tr><tr><td>GUIOdyssey</td><td>8,334</td><td>8,334</td><td>15.3</td><td>✓</td><td>手机与平板</td><td>212 个应用，1,357 种应用组合</td><td>高与低</td><td>✓</td></tr></tbody></table>


High-Level Task Instruction. For all aforementioned cross-app tasks, we propose a flexible high-level instruction template to construct diverse GUI episodes. The instruction templates are generated by i) human participants and ii) prompting GPT-4 with task descriptions. Ultimately, we collect 91 high-level instruction templates. The diversity of instructions is implemented in three ways. First, the item in each template can be replaced with various candidates. For instance, the item in the instruction "Listen to a podcast episode on \{item: yoga\} for beginners and create a to-do list" can be substituted with "meditation" or "digital marketing" as shown in Fig. 3. Second, the apps used to complete the instruction can be selected from a predefined pool. For example, the podcast app can be Spotify or Google Pod-cast and the scheduling app can be Todoist or Microsoft To Do. Finally, we employ GPT-4 to rewrite the instruction using candidate items and apps with different expressions.
高层任务指令。对于上述所有跨应用任务，我们提出一个灵活的高层指令模板来构造多样的 GUI 场景。指令模板由 i) 人类参与者和 ii) 用任务描述提示 GPT-4 生成。最终，我们收集了 91 份高层指令模板。指令的多样性通过三种方式实现。第一，每个模板中的项可以被多种候选项替换。例如，指令“听一个关于 \{item: yoga\} 的播客集并为初学者创建待办清单”中的项可替换为“冥想”或“数字营销”，如图 3 所示。第二，可以从预定义池中选择完成指令所用的应用。例如，播客应用可以是 Spotify 或 Google Pod-cast，日程安排应用可以是 Todoist 或 Microsoft To Do。最后，我们使用 GPT-4 以不同表达方式对候选项和应用重新撰写指令。


Human Demonstration. With diverse high-level instructions collected, we then engage independent annotators, experienced in using mobile devices and various apps, participate in the annotation of GUI episodes. As mentioned in Sec. 3.1, we use an Android emulator to record GUI episodes on various mobile devices such as Pixel Pro, Tablet, and Fold as shown in Fig. 2. All annotators are required to complete the instructions step-by-step and avoid clicking on anything unrelated to the task while recording their interactions. To improve data quality, annotators are trained to annotate at least twenty episodes before starting annotation. During annotation, annotators are asked to save the screenshot before each action step. As shown in Table 5 of Appendix Sec. 8.2, we use the actions IMPOSSIBLE and COMPLETE to denote the instructions that cannot be completed and those that have been completed, respectively. Specifically, when annotators select IMPOSSIBLE, they are required to record the reason why the task could not be completed. Upon completion of the navigation, our data annotation tools save the episode, including the user's instructions, screenshots, actions taken at each step, the apps used by the annotator, and any additional notes. An example of the annotation process is illustrated in Fig. 3.
人工示范。收集了多样化的高层指令后，我们再让独立标注者参与 GUI 场景的标注，这些标注者具备使用移动设备和各种应用的经验。正如第 3.1 节所述，我们使用 Android 模拟器在 Pixel Pro、Tablet、Fold 等多种设备上记录 GUI 场景，如图 2 所示。所有标注者在记录互动时须按步骤完成指令，并避免在与任务无关的操作。为提高数据质量，标注者在开始标注前需完成至少 twenty 个场景的标注。在标注过程中，标注者需在每个操作步骤前保存截图。如附录 Sec. 8.2 表 Table 5 所示，我们用 IMPOSSIBLE 和 COMPLETE 两个动作来表示无法完成的指令与已完成的指令。具体地，当标注者选择 IMPOSSIBLE 时，需要记录任务无法完成的原因。导航完成后，我们的数据标注工具会保存该场景，包括用户的指令、截图、各步骤所执行的操作、标注者使用的应用及任何附加备注。标注过程的一个示例如图 3 所示。


Fine-grained Episode Annotation. After collecting human-demonstrated GUI episodes, we utilize the state-of-the-art model GPT-40 to generate fine-grained episode annotations, consisting of two main components. The first component is the Low-Level Instruction, which refers to a set of fine-grained instructions that serve as atomic decompositions of high-level instructions, providing detailed steps for executing the next action on the current page. The second component is Semantic Annotation, which includes: (1) Screen Description, offering a detailed depiction of the content displayed in the screenshot; (2) Contextual Information, summarizing the preceding steps that led to the current stage of the task; and (3) Decision Rationale, explaining the reasoning behind the next action based on both historical context and the current screen content. Further details can be found in Appendix Sec. 8.3, while an example of the semantic annotation process is illustrated in Fig. 3.
细粒度场景标注。在收集到人工示范的 GUI 场景后，我们利用最前沿的模型 GPT-40 生成细粒度的场景标注，包含两大部分。第一部分是低级指令，指一组细粒度指令，作为高级指令的原子分解，为在当前页面执行下一步操作提供详细步骤。第二部分是语义标注，包含：(1) 屏幕描述，对截图中显示内容的详细描绘；(2) 上下文信息，总结导致当前任务阶段的前一步骤；(3) 决策理由，基于历史上下文与当前屏幕内容解释下一步操作的推理。更多细节请参见附录 Sec. 8.3，语义标注过程的示例见图 3。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_19_420e82.jpg"/>



Figure 3. Data collection pipeline of GUIOdyssey for cross-app GUI navigation. GUIOdyssey comprises six categories of navigation tasks. For each category, we construct instruction templates with items and apps selected from a predefined pool, resulting in a vast array of unique instructions for annotating GUI episodes. Human demonstrations on an Android emulator capture the annotation of each episode in a comprehensive format. After rigorous quality checks, GUIOdyssey includes 8, 334 validated cross-app GUI navigation episodes.
图 3. GUIOdyssey 的跨应用 GUI 导航数据收集流程。GUIOdyssey 由六大类导航任务组成。对于每一类，我们从预定义池中选择项与应用来构建指令模板，生成大量不同的指令以便标注 GUI 场景。基于 Android 模拟器的人类演示以全面格式记录每个场景的标注。经严格质量检查后，GUIOdyssey 包含 8,334 条经过验证的跨应用 GUI 导航场景。


Data Quality Check. With all episodes collected, we perform a data quality check. The episode is thought to be accurate and complete if it satisfies the following three criteria: i) whether any screenshot in the episode is regularly saved; ii) whether the sequence of screenshots and actions can complete the instruction; iii) whether the instruction rewritten by GPT-4 is equivalent to the original one. After filtering low-quality data, we obtain our cross-app navigation dataset called GUIOdyssey.
数据质量检查。所有场景收集完成后，我们进行数据质量检查。若场景满足以下三项标准，即认为准确完整：i) 场景中的任意截图是否被定期保存；ii) 截图与操作序列是否能完成指令；iii) GPT-4 重写的指令是否等价于原始指令。经筛除低质量数据后，我们得到跨应用导航数据集 GUIOdyssey。


### 3.3. Dataset Statistics
### 3.3. 数据集统计


GUIOdyssey targets cross-app navigation, a more practical scenario than single-app navigation in real-world settings. It comprises 8, 334 episodes with an average of 15.3 steps per episode, making it the mobile GUI navigation dataset with the longest average episode length. Compared to existing datasets, GUIOdyssey encompasses a broader range of navigation tasks and more complex workflows, featuring six types of cross-app tasks that span 212 apps across domains such as video, music, and reading. It also includes six types of electronic devices, including foldable phones and tablets, which were not covered in previous datasets. Visual statistics are presented in Fig. 4, where Fig. 4 (c) highlights its significantly longer episode lengths compared to single-app datasets[26, 38]. Other provide additional insights into app combination and usage frequency (Fig. 4 a, b), episode length distribution across task types (Fig. 4 d), the presence of 25 app categories (Fig. 4 e), and the diversity of device types (Fig. 4 f).
GUIOdyssey 以跨应用导航为目标，优于现实环境中单应用导航的更实际场景。它包含 8,334 条情节，平均每条情节 15.3 步，使其成为平均情节长度最大的移动端 GUI 导航数据集。与现有数据集相比，GUIOdyssey 覆盖更广泛的导航任务和更复杂的工作流，包含跨 212 个应用、跨视频、音乐、阅读等领域的六种跨应用任务类型。它还覆盖六种电子设备类型，包括可折叠手机和平板电脑，这是此前数据集所未覆盖的。可视化统计见图 4，图 4（c）凸显其情节长度显著高于单应用数据集的特征[26, 38]。其他图表则对应用组合和使用频率（图 4 a、b）、任务类型下的情节长度分布（图 4 d）、存在的 25 类应用品类（图 4 e）以及设备类型的多样性（图 4 f）提供了额外洞察。


## 4. Method: OdysseyAgent
## 4. 方法：OdysseyAgent


Building upon GUIOdyssey, we introduce OdysseyAgent, an exploratory framework for cross-app navigation tasks powered by Large Vision-Language Models (LVLMs). A key challenge in cross-app tasks is balancing the need to process numerous historical screenshots and lengthy action sequences with the requirement for fast inference in frequent user interactions. To address these demands, we fine-tune Qwen-VL [5] on GUIOdyssey, incorporating a history replay module to optimize both performance and efficiency.
在 GUIOdyssey 的基础上，我们引入 OdysseyAgent——一个由大型视觉-语言模型（LVLMs）驱动的跨应用导航任务探索框架。跨应用任务的一个关键挑战是，在处理大量历史截图和冗长动作序列的需求与在频繁用户交互中实现快速推理之间取得平衡。为应对这些需求，我们在 GUIOdyssey 上对 Qwen-VL [5] 进行微调，并加入历史回放模块以优化性能与效率。


As illustrated in Fig. 5, OdysseyAgent inherits from Qwen-VL-Chat [5] and comprises a vision encoder, a large language model (LLM), and a vision-language (VL) adapter. Crucially, we introduce a history resampler to compress historical screenshot tokens before they reach the LLM. This design alleviates the overhead of stacking all past screenshots while still leveraging essential contextual information. In Appendix Sec. 10.1, we compare the history resampler with a straightforward multi-image concatenation approach, demonstrating that the history resampler achieves a more favorable balance between performance and inference efficiency.
如图 5 所示，OdysseyAgent 继承自 Qwen-VL-Chat [5]，由一个视觉编码器、一个大语言模型（LLM）和一个视觉-语言（VL）适配器组成。关键在于，我们引入历史重采样器，在将历史截图令牌送入 LLM 之前对其进行压缩。这一设计缓解了堆叠所有历史截图的开销，同时仍能利用关键的上下文信息。在附录第 10.1 节，我们将历史重采样器与简单的多图像拼接方法进行了对比，结果表明历史重采样器在性能与推理效率之间取得了更有利的平衡。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_19_232637.jpg"/>



Figure 4. Statistics for GUIOdyssey, zoom in to view details. (a) App Combinations Frequency. (b) App Frequency. (c) Episode length of AITW, AndroidControl and GUIOdyssey. (d) Episode length distribution. (e) App categories distribution. (f) Device statistics.
图 4。GUIOdyssey 的统计信息，放大查看细节。（a）应用组合频率。（b）应用频率。（c）AITW、AndroidControl 与 GUIOdyssey 的情节长度。（d）情节长度分布。（e）应用类别分布。（f）设备统计。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/02/2026_02_18__00_21_19_b3b5f9.jpg"/>



Figure 5. The architecture of OdysseyAgent. Beyond Qwen-VL's standard components, OdysseyAgent introduces a history resam-pler that enables efficient attention to historical screenshots.
图 5。OdysseyAgent 的架构。除了 Qwen-VL 的标准组件外，OdysseyAgent 还引入一个历史重采样器，使对历史截图的关注更高效。


Specifically, the history resampler is implemented as a single-layer cross-attention module, where learnable em-beddings serve as the query and historical screenshot tokens function as both key and value. After resampling, the compressed historical screenshot tokens are concatenated with the current screen image token, user instruction, and previous actions. This fused representation is then fed into the LLM to predict the next action. Formally,the next-word prediction objective $\mathcal{L}$ is defined as: $\mathcal{L} = \mathop{\sum }\limits_{{i = 1}}^{N}{P}_{\theta }\left( {{A}_{i}^{t} \mid  {X}^{\{ t,t - 1,\cdots ,t - \delta \} },{I}_{\text{ user }},{A}_{ < i}^{t}}\right)$ ,where $N$ is the number of tokens in action ${A}^{t},\delta$ denotes the historical image window,and $\theta$ represents the trainable parameters in OdysseyAgent (namely the VL adapter, history resampler, and LLM as shown in Fig. 5).
具体而言，历史重采样器实现为一个单层跨注意力模块，学习到的嵌入作为查询，历史截图令牌同时充当键和值。经过重采样后，压缩的历史截图令牌与当前屏幕图像令牌、用户指令以及先前的动作连接在一起，形成的融合表示随后输入到 LLM 以预测下一步动作。正式而言，下一字预测目标 $\mathcal{L}$ 的定义为：$\mathcal{L} = \mathop{\sum }\limits_{{i = 1}}^{N}{P}_{\theta }\left( {{A}_{i}^{t} \mid  {X}^{\{ t,t - 1,\cdots ,t - \delta \} },{I}_{\text{ user }},{A}_{ < i}^{t}}\right)$ ，其中 $N$ 为动作中的令牌数，${A}^{t},\delta$ 表示历史图像窗口，$\theta$ 代表 OdysseyAgent 的可训练参数（即 VL 适配器、历史重采样器和如图 5 所示的 LLM）。


## 5. Experiment
## 5. 实验


The experimental setup is detailed in Sec. 5.1. In Sec. 5.2, we evaluate OdysseyAgent's performance under both in-and out-of-domain settings. Sec. 5.3 further explores the role of historical information in cross-app tasks.
实验设置在 Sec. 5.1 中给出详细说明。在 Sec. 5.2 中，我们评估 OdysseyAgent 在同域与跨域设定下的表现。Sec. 5.3 进一步探讨历史信息在跨应用任务中的作用。


### 5.1. Experimental Setup
### 5.1. 实验设置


We leverage the comprehensiveness of GUIOdyssey to evaluate OdysseyAgent's performance in both in- and out-of-domain scenarios. To this end, we divide GUIOdyssey into four distinct setups. The first is an in-domain split: (i) Train-Random & Test-Random. The remaining three are out-of-domain splits: (ii) Train-App & Test-App, (iii) Train-Task & Test-Task, and (iv) Train-Device & Test-Device. These setups are designed to assess the agent's generalizability across different app, task, and device scenarios. A detailed description of the four setups is provided in Appendix Sec. 9.1, while the training details are available in Sec. 9.2.
我们利用 GUIOdyssey 的全面性，在同域与跨域情境下评估 OdysseyAgent 的表现。为此，我们将 GUIOdyssey 分为四个不同的设定。第一种为同域分割：(i) Train-Random & Test-Random。其余三种为跨域分割：(ii) Train-App & Test-App、(iii) Train-Task & Test-Task、(iv) Train-Device & Test-Device。这些设定旨在评估代理在不同应用、任务和设备场景中的泛化能力。四个设定的详细描述可在附录 Sec. 9.1 中获得，训练细节见 Sec. 9.2。


Evaluation Metrics. To ensure reproducibility and efficiency, we adopt an offline evaluation method to benchmark performance. We use the Action Matching Score (AMS) as our metric, inspired by the approaches presented in AITW [38] and AutoUI [64]. An action is considered correct if its action type matches the ground-truth type. Additionally, for CLICK and LONG PRESS actions, we consider them correct if they fall within ${14}\%$ of the screen distance from the reference gesture. Furthermore, we utilize SAM2 [37] to determine the coordinates of the target element, and if the predicted coordinates lie within the region segmented by SAM2, the action is also deemed correct. As for SCROLL actions, we compare whether the direction (i.e., up, down, left, or right) matches the gold gesture's direction. For TYPE actions, we evaluate the Average Normalized Levenshtein Similarity (ANLS) [7] between the predicted and gold gestures. If the ANLS is below a certain threshold (set to 0.5 in our experiments), we consider it correct. We then calculate Success Rate (SR) for the whole episode. A task is considered successful only if all actions are correct. Success Rate (SR) is a rigorous metric. It would be harder to achieve higher SR in tasks with more action steps.
评价指标。为确保可重现性和效率，我们采用离线评估方法来基准性能。我们使用动作匹配分数（AMS）作为评估指标，灵感来自 AITW [38] 和 AutoUI [64] 的方法。若动作类型与真实类别相符，则认为该动作正确。此外，对于 CLICK 和 LONG PRESS 动作，若其距离参考手势的屏幕距离位于 ${14}\%$ 以内，则认为正确。 furthermore，我们利用 SAM2 [37] 来确定目标元素的坐标，若预测坐标位于 SAM2 分割的区域内，该动作也被视为正确。至于 SCROLL 动作，我们比较方向是否与金标准手势的方向一致（即向上、向下、向左或向右）。对于 TYPE 动作，我们评估预测手势与金标准手势之间的平均归一化莱文斯坦相似度（ANLS）[7]。若 ANLS 低于某一阈值（在我们的实验中设为 0.5），则视为正确。随后我们计算整段任务的成功率（SR）。只有当所有动作都正确时，任务才被视为成功。成功率（SR）是一个严格的度量。在具有更多动作步骤的任务中，更难达到较高的 SR。


Table 2. Results of different LVLMs on Test-Random split. The evaluation metric is the action matching score (AMS). 'HL' and 'LL' indicate that the task instruction is high-level and low-level, respectively. * indicates that agent's training also includes semantic annotations.
表 2。不同 LVLM 在 Test-Random 分割上的结果。评估指标为动作匹配分数（AMS）。'HL' 和 'LL' 表示任务指令分别为高层级和低层级。* 表示代理的训练还包含语义注释。


<table><tr><td rowspan="2">Model</td><td colspan="2">Tool</td><td colspan="2">Information</td><td colspan="2">Shopping</td><td colspan="2">Media</td><td colspan="2">Social</td><td colspan="2">Multi-Apps</td><td colspan="2">Overall</td></tr><tr><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td></tr><tr><td colspan="15">zero-shot</td></tr><tr><td>GPT-4V</td><td>14.93</td><td>40.86</td><td>14.69</td><td>38.56</td><td>12.17</td><td>36.04</td><td>10.80</td><td>48.40</td><td>16.79</td><td>43.21</td><td>11.54</td><td>40.61</td><td>13.49</td><td>41.28</td></tr><tr><td>GPT-40</td><td>14.15</td><td>38.11</td><td>13.86</td><td>42.40</td><td>11.69</td><td>40.57</td><td>12.00</td><td>52.40</td><td>18.14</td><td>44.44</td><td>9.28</td><td>38.35</td><td>13.19</td><td>42.71</td></tr><tr><td>Claude3.5-sonnet</td><td>22.99</td><td>40.28</td><td>14.69</td><td>34.56</td><td>12.17</td><td>31.03</td><td>14.00</td><td>37.60</td><td>16.79</td><td>30.62</td><td>14.14</td><td>31.00</td><td>15.80</td><td>34.18</td></tr><tr><td>InternVL2-Pro</td><td>19.45</td><td>49.51</td><td>15.86</td><td>40.23</td><td>17.18</td><td>41.05</td><td>13.20</td><td>51.60</td><td>14.81</td><td>40.00</td><td>15.72</td><td>41.52</td><td>16.04</td><td>43.98</td></tr><tr><td>CogAgent</td><td>18.81</td><td>33.52</td><td>12.35</td><td>29.79</td><td>13.02</td><td>26.89</td><td>12.63</td><td>25.80</td><td>14.72</td><td>33.54</td><td>12.15</td><td>33.09</td><td>13.95</td><td>30.44</td></tr><tr><td>SphAgent</td><td>22.24</td><td>36.81</td><td>17.43</td><td>29.95</td><td>13.60</td><td>25.08</td><td>15.54</td><td>33.03</td><td>14.20</td><td>28.83</td><td>12.89</td><td>26.32</td><td>15.98</td><td>30.00</td></tr><tr><td colspan="15">zero-shot with OmniParser</td></tr><tr><td>GPT-4V</td><td>24.37</td><td>56.03</td><td>22.44</td><td>51.10</td><td>17.16</td><td>46.75</td><td>18.65</td><td>62.18</td><td>32.28</td><td>59.18</td><td>24.21</td><td>54.58</td><td>23.18</td><td>54.97</td></tr><tr><td>GPT-40</td><td>26.63</td><td>55.28</td><td>23.45</td><td>53.91</td><td>18.34</td><td>47.63</td><td>19.17</td><td>63.21</td><td>31.01</td><td>61.08</td><td>23.39</td><td>55.95</td><td>23.67</td><td>56.18</td></tr><tr><td>Claude3.5-sonnet</td><td>39.20</td><td>64.07</td><td>28.46</td><td>61.92</td><td>27.22</td><td>56.51</td><td>28.50</td><td>66.84</td><td>40.82</td><td>68.99</td><td>33.11</td><td>65.12</td><td>32.88</td><td>63.91</td></tr><tr><td>InternVL2-Pro</td><td>16.58</td><td>58.04</td><td>16.03</td><td>51.30</td><td>8.88</td><td>49.70</td><td>16.58</td><td>60.62</td><td>16.14</td><td>53.80</td><td>13.95</td><td>52.39</td><td>14.69</td><td>54.31</td></tr><tr><td colspan="15">fine-tuned</td></tr><tr><td>Qwen-VL</td><td>85.55</td><td>90.79</td><td>68.04</td><td>83.36</td><td>62.28</td><td>80.67</td><td>77.56</td><td>88.15</td><td>80.29</td><td>88.36</td><td>74.27</td><td>86.56</td><td>74.67</td><td>86.32</td></tr><tr><td>Qwen-VL*</td><td>86.35</td><td>90.99</td><td>72.01</td><td>85.77</td><td>67.31</td><td>82.86</td><td>80.33</td><td>89.48</td><td>82.39</td><td>88.01</td><td>77.52</td><td>89.40</td><td>77.65</td><td>87.78</td></tr><tr><td>OdysseyAgent</td><td>86.01</td><td>91.21</td><td>69.83</td><td>83.37</td><td>65.19</td><td>82.63</td><td>77.10</td><td>88.55</td><td>81.47</td><td>87.66</td><td>75.13</td><td>87.84</td><td>75.79</td><td>86.88</td></tr><tr><td>OdysseyAgent*</td><td>86.82</td><td>91.25</td><td>71.79</td><td>86.58</td><td>68.58</td><td>83.74</td><td>80.93</td><td>89.66</td><td>82.88</td><td>88.27</td><td>78.47</td><td>89.39</td><td>78.24</td><td>88.15</td></tr></table>
<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">工具</td><td colspan="2">信息</td><td colspan="2">购物</td><td colspan="2">媒体</td><td colspan="2">社交</td><td colspan="2">多应用</td><td colspan="2">总体</td></tr><tr><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td><td>HL</td><td>LL</td></tr><tr><td colspan="15">零-shot</td></tr><tr><td>GPT-4V</td><td>14.93</td><td>40.86</td><td>14.69</td><td>38.56</td><td>12.17</td><td>36.04</td><td>10.80</td><td>48.40</td><td>16.79</td><td>43.21</td><td>11.54</td><td>40.61</td><td>13.49</td><td>41.28</td></tr><tr><td>GPT-40</td><td>14.15</td><td>38.11</td><td>13.86</td><td>42.40</td><td>11.69</td><td>40.57</td><td>12.00</td><td>52.40</td><td>18.14</td><td>44.44</td><td>9.28</td><td>38.35</td><td>13.19</td><td>42.71</td></tr><tr><td>Claude3.5-sonnet</td><td>22.99</td><td>40.28</td><td>14.69</td><td>34.56</td><td>12.17</td><td>31.03</td><td>14.00</td><td>37.60</td><td>16.79</td><td>30.62</td><td>14.14</td><td>31.00</td><td>15.80</td><td>34.18</td></tr><tr><td>InternVL2-Pro</td><td>19.45</td><td>49.51</td><td>15.86</td><td>40.23</td><td>17.18</td><td>41.05</td><td>13.20</td><td>51.60</td><td>14.81</td><td>40.00</td><td>15.72</td><td>41.52</td><td>16.04</td><td>43.98</td></tr><tr><td>CogAgent</td><td>18.81</td><td>33.52</td><td>12.35</td><td>29.79</td><td>13.02</td><td>26.89</td><td>12.63</td><td>25.80</td><td>14.72</td><td>33.54</td><td>12.15</td><td>33.09</td><td>13.95</td><td>30.44</td></tr><tr><td>SphAgent</td><td>22.24</td><td>36.81</td><td>17.43</td><td>29.95</td><td>13.60</td><td>25.08</td><td>15.54</td><td>33.03</td><td>14.20</td><td>28.83</td><td>12.89</td><td>26.32</td><td>15.98</td><td>30.00</td></tr><tr><td colspan="15">带 OmniParser 的零-shot</td></tr><tr><td>GPT-4V</td><td>24.37</td><td>56.03</td><td>22.44</td><td>51.10</td><td>17.16</td><td>46.75</td><td>18.65</td><td>62.18</td><td>32.28</td><td>59.18</td><td>24.21</td><td>54.58</td><td>23.18</td><td>54.97</td></tr><tr><td>GPT-40</td><td>26.63</td><td>55.28</td><td>23.45</td><td>53.91</td><td>18.34</td><td>47.63</td><td>19.17</td><td>63.21</td><td>31.01</td><td>61.08</td><td>23.39</td><td>55.95</td><td>23.67</td><td>56.18</td></tr><tr><td>Claude3.5-sonnet</td><td>39.20</td><td>64.07</td><td>28.46</td><td>61.92</td><td>27.22</td><td>56.51</td><td>28.50</td><td>66.84</td><td>40.82</td><td>68.99</td><td>33.11</td><td>65.12</td><td>32.88</td><td>63.91</td></tr><tr><td>InternVL2-Pro</td><td>16.58</td><td>58.04</td><td>16.03</td><td>51.30</td><td>8.88</td><td>49.70</td><td>16.58</td><td>60.62</td><td>16.14</td><td>53.80</td><td>13.95</td><td>52.39</td><td>14.69</td><td>54.31</td></tr><tr><td colspan="15">微调</td></tr><tr><td>Qwen-VL</td><td>85.55</td><td>90.79</td><td>68.04</td><td>83.36</td><td>62.28</td><td>80.67</td><td>77.56</td><td>88.15</td><td>80.29</td><td>88.36</td><td>74.27</td><td>86.56</td><td>74.67</td><td>86.32</td></tr><tr><td>Qwen-VL*</td><td>86.35</td><td>90.99</td><td>72.01</td><td>85.77</td><td>67.31</td><td>82.86</td><td>80.33</td><td>89.48</td><td>82.39</td><td>88.01</td><td>77.52</td><td>89.40</td><td>77.65</td><td>87.78</td></tr><tr><td>OdysseyAgent</td><td>86.01</td><td>91.21</td><td>69.83</td><td>83.37</td><td>65.19</td><td>82.63</td><td>77.10</td><td>88.55</td><td>81.47</td><td>87.66</td><td>75.13</td><td>87.84</td><td>75.79</td><td>86.88</td></tr><tr><td>OdysseyAgent*</td><td>86.82</td><td>91.25</td><td>71.79</td><td>86.58</td><td>68.58</td><td>83.74</td><td>80.93</td><td>89.66</td><td>82.88</td><td>88.27</td><td>78.47</td><td>89.39</td><td>78.24</td><td>88.15</td></tr></tbody></table>


Table 3. OdyssseyAgent's performance on out-of-domain tasks. 'Semantic?' indicates whether the model's training includes semantic annotations. 'HL' and 'LL' indicate that the task instruction is high-level and low-level, respectively.
表 3. OdyssseyAgent 在域外任务上的表现。'语义？' 表示模型的训练是否包含语义标注。'HL' 与 'LL' 表示任务指令分别为高层和低层。


<table><tr><td rowspan="2">Semantic?</td><td rowspan="2">Task Level</td><td colspan="2">Test-Task</td><td colspan="2">Test-Device</td><td colspan="2">Test-App</td><td colspan="2">Overall</td></tr><tr><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td></tr><tr><td rowspan="2">✘</td><td>HL</td><td>54.36</td><td>0.09</td><td>61.20</td><td>1.88</td><td>63.03</td><td>7.70</td><td>59.53</td><td>3.22</td></tr><tr><td>LL</td><td>78.97</td><td>2.20</td><td>79.66</td><td>8.47</td><td>84.24</td><td>20.70</td><td>80.96</td><td>10.46</td></tr><tr><td rowspan="2">✓</td><td>HL</td><td>56.19</td><td>0.26</td><td>66.63</td><td>5.07</td><td>65.89</td><td>8.81</td><td>62.90</td><td>4.71</td></tr><tr><td>LL</td><td>80.19</td><td>2.29</td><td>79.93</td><td>11.66</td><td>83.47</td><td>20.02</td><td>81.20</td><td>11.32</td></tr></table>
<table><tbody><tr><td rowspan="2">语义？</td><td rowspan="2">任务级</td><td colspan="2">测试任务</td><td colspan="2">测试设备</td><td colspan="2">测试应用</td><td colspan="2">总体</td></tr><tr><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td></tr><tr><td rowspan="2">✘</td><td>HL</td><td>54.36</td><td>0.09</td><td>61.20</td><td>1.88</td><td>63.03</td><td>7.70</td><td>59.53</td><td>3.22</td></tr><tr><td>LL</td><td>78.97</td><td>2.20</td><td>79.66</td><td>8.47</td><td>84.24</td><td>20.70</td><td>80.96</td><td>10.46</td></tr><tr><td rowspan="2">✓</td><td>HL</td><td>56.19</td><td>0.26</td><td>66.63</td><td>5.07</td><td>65.89</td><td>8.81</td><td>62.90</td><td>4.71</td></tr><tr><td>LL</td><td>80.19</td><td>2.29</td><td>79.93</td><td>11.66</td><td>83.47</td><td>20.02</td><td>81.20</td><td>11.32</td></tr></tbody></table>


### 5.2. Comprehensive evaluation on the GUIOdyssey
### 5.2. GUIOdyssey 的综合评估


We evaluate OdysseyAgent's performance in both in-domain and out-of-domain scenarios. For each step in the dataset, we construct prompts using high-level and low-level instructions separately for training and evaluation. High-level instructions reflect the model's capability to handle real-world GUI navigation tasks, while low-level instructions break down each step of the high-level tasks, assessing the model's ability to follow simpler commands. Naturally, high-level instructions are more challenging than low-level instructions.
我们在同域和跨域场景下评估 OdysseyAgent 的性能。对于数据集中的每一个步骤，我们分别使用高层和低层指令进行训练与评估来构造提示。高层指令体现模型处理真实世界 GUI 导航任务的能力，而低层指令将高层任务的每一步拆解，评估模型执行更简单命令的能力。显然，高层指令比低层指令更具挑战性。


In-domain Performance. We compare OdysseyA-gent against three types of methods on the Test-Random split of GUIOdyssey: (1) LVLMs zero-shot, including closed-source proprietary LVLMs (GPT-4V [1], GPT-40 [36], Claude3.5-Sonnet [2], InternVL2-Pro [13]) and open-source GUI-specific models (SphAgent [9], CogAgent [23]); (2) closed-source LVLMs zero-shot with OmniParser [33]; and (3) fine-tuned LVLMs Qwen-VL[5]. Due to budget constraints, for closed-source models, we sample 200 episodes from the original test set to serve as their evaluation set. Note that Qwen-VL is effectively OdysseyA-gent without the history resampler, meaning it does not incorporate historical screenshots. The result is shown in Table 2. InternVL2-Pro achieves the best overall performance among all coordinated-based models. Despite being trained on other GUI navigation datasets, CogAgent and SphAgent exhibit poor performance on GUIOdyssey, which we attribute to a significant domain gap between cross-app and single-app tasks, resulting in substantial performance disparities. Supported by OmniParser's robust GUI grounding, most closed-source LVLMs substantially improve their cross-app performance, with Claude3.5-Sonnet achieving the best results. In addition, OdysseyAgent surpasses the fine-tuned Qwen-VL, indicating that the proposed history resampler module enhances cross-app navigation. After incorporating semantic annotations during training, OdysseyAgent further improves its performance on all cross-app tasks, achieving 78.24 and 88.15 AMS in high-level and low-level instruction tasks, respectively, thereby demonstrating the effectiveness of our dataset.
在域内性能。我们在 GUIOdyssey 的 Test-Random 片段上，将 OdysseyAgent 与三种方法进行比较：（1）零-shot 的 LVLMs，包括闭源专有 LVLMs（GPT-4V [1]、GPT-40 [36]、Claude3.5-Sonnet [2]、InternVL2-Pro [13]）以及开源 GUI 专用模型（SphAgent [9]、CogAgent [23]）；（2）零-shot 的闭源 LVLMs 结合 OmniParser [33]；以及（3）微调的 LVLMs Qwen-VL[5]。由于预算限制，对于闭源模型，我们从原始测试集中抽取 200 条剧本作为其评估集。注意，Qwen-VL 实质上是没有历史记录重采样器的 OdysseyAgent，意味着它不包含历史截图。结果如表 2 所示。InternVL2-Pro 在所有基于坐标的模型中取得最佳总体表现。尽管 CogAgent 和 SphAgent 在其他 GUI 导航数据集上进行了训练，但在 GUIOdyssey 上表现较差，我们将其归因于跨应用与单应用任务之间存在显著的领域差距，导致性能差异较大。在 OmniParser 的强大 GUI 绑定支持下，大多数闭源 LVLMs 的跨应用性能显著提升，Claude3.5-Sonnet 取得最佳结果。此外，OdysseyAgent 超越了微调的 Qwen-VL，表明所提出的历史重采样模块提升了跨应用导航能力。经过训练阶段引入语义注释后，OdysseyAgent 在所有跨应用任务上的表现进一步提升，在高层和低层指令任务分别达到 78.24 和 88.15AMS，从而证明了我们数据集的有效性。


Out-of-domain Performance. We further assess the OdysseyAgent's generalization capability in unseen scenarios. As shown in Table 3, OdysseyAgent's out-of-domain performance declines by 16.26 and 5.92 for high- and low-level instructions, respectively, compared to in-domain performance without semantic annotations. With semantic annotations, these declines become 15.34 and 6.95. This suggests that high-level instructions are more challenging to generalize in cross-app tasks compared to low-level instructions. Furthermore, incorporating semantic annotations during training improves performance in most scenarios, with especially notable gains on high-level instruction tasks, underscoring the value of semantic annotations for unseen domain. Compared to in-domain performance, the performance gap between high- and low-level instructions is even larger in out-of-domain tasks. This implies that the model currently lacks sufficient reasoning and planning capabilities to effectively handle unseen high-level instruction tasks.
在域外性能。我们进一步评估 OdysseyAgent 在未见场景中的泛化能力。如表 3 所示，与未含语义注释的在域内性能相比，OdysseyAgent 的跨域性能在高层和低层指令上的下降分别为 16.26 和 5.92；在含有语义注释的情况下，这些下降变为 15.34 和 6.95。这表明高层指令在跨应用任务中的泛化比低层指令更具挑战性。此外，在训练阶段加入语义注释在大多数场景下提升了性能，尤其在高层指令任务上提升显著，凸显了语义注释对于未知领域的价值。与在域内性能相比，跨域任务中高层与低层指令之间的性能差距更大。这表明模型目前还缺乏足够的推理与规划能力来有效处理未见的高层指令任务。


Table 4. The impact of different historical components in GUIOdyssey across four splits. High-level instructions are used for both training and evaluation, and performance is measured by AMS and SR.
表 4。GUIOdyssey 在四个拆分中的不同历史组件的影响。高层指令被用于训练和评估，性能以 AMS 和 SR 进行衡量。


<table><tr><td rowspan="2"></td><td colspan="3">Historical Information</td><td colspan="2">Test-Random</td><td colspan="2">Test-Task</td><td colspan="2">Test-Device</td><td colspan="2">Test-App</td><td colspan="2">Overall</td></tr><tr><td>action</td><td>screenshot</td><td>context</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td></tr><tr><td>(1)</td><td>✘</td><td>✘</td><td>✘</td><td>66.13</td><td>1.65</td><td>47.62</td><td>0.00</td><td>54.15</td><td>0.72</td><td>54.49</td><td>3.59</td><td>55.60</td><td>1.49</td></tr><tr><td>(2)</td><td>✓</td><td>✘</td><td>✘</td><td>74.67</td><td>9.70</td><td>55.00</td><td>0.00</td><td>62.03</td><td>2.03</td><td>62.06</td><td>8.98</td><td>63.44</td><td>5.18</td></tr><tr><td>(3)</td><td>✘</td><td>✓</td><td>✘</td><td>71.22</td><td>6.69</td><td>51.69</td><td>0.09</td><td>59.12</td><td>2.24</td><td>59.16</td><td>7.78</td><td>60.30</td><td>4.20</td></tr><tr><td>(4)</td><td>✘</td><td>✘</td><td>✓</td><td>75.25</td><td>9.50</td><td>57.66</td><td>0.62</td><td>62.35</td><td>2.24</td><td>63.82</td><td>7.87</td><td>64.77</td><td>5.06</td></tr><tr><td>(5)</td><td>✓</td><td>✓</td><td>✘</td><td>75.79</td><td>9.38</td><td>54.36</td><td>0.09</td><td>61.20</td><td>1.88</td><td>63.03</td><td>7.70</td><td>63.60</td><td>4.76</td></tr><tr><td>(6)</td><td>✓</td><td>✓</td><td>✓</td><td>77.06</td><td>11.61</td><td>58.83</td><td>0.18</td><td>65.85</td><td>5.00</td><td>65.63</td><td>8.47</td><td>66.84</td><td>6.32</td></tr></table>
<table><tbody><tr><td rowspan="2"></td><td colspan="3">历史信息</td><td colspan="2">测试-随机</td><td colspan="2">测试-任务</td><td colspan="2">测试-设备</td><td colspan="2">测试-应用</td><td colspan="2">总体</td></tr><tr><td>动作</td><td>截图</td><td>上下文</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td><td>AMS</td><td>SR</td></tr><tr><td>(1)</td><td>✘</td><td>✘</td><td>✘</td><td>66.13</td><td>1.65</td><td>47.62</td><td>0.00</td><td>54.15</td><td>0.72</td><td>54.49</td><td>3.59</td><td>55.60</td><td>1.49</td></tr><tr><td>(2)</td><td>✓</td><td>✘</td><td>✘</td><td>74.67</td><td>9.70</td><td>55.00</td><td>0.00</td><td>62.03</td><td>2.03</td><td>62.06</td><td>8.98</td><td>63.44</td><td>5.18</td></tr><tr><td>(3)</td><td>✘</td><td>✓</td><td>✘</td><td>71.22</td><td>6.69</td><td>51.69</td><td>0.09</td><td>59.12</td><td>2.24</td><td>59.16</td><td>7.78</td><td>60.30</td><td>4.20</td></tr><tr><td>(4)</td><td>✘</td><td>✘</td><td>✓</td><td>75.25</td><td>9.50</td><td>57.66</td><td>0.62</td><td>62.35</td><td>2.24</td><td>63.82</td><td>7.87</td><td>64.77</td><td>5.06</td></tr><tr><td>(5)</td><td>✓</td><td>✓</td><td>✘</td><td>75.79</td><td>9.38</td><td>54.36</td><td>0.09</td><td>61.20</td><td>1.88</td><td>63.03</td><td>7.70</td><td>63.60</td><td>4.76</td></tr><tr><td>(6)</td><td>✓</td><td>✓</td><td>✓</td><td>77.06</td><td>11.61</td><td>58.83</td><td>0.18</td><td>65.85</td><td>5.00</td><td>65.63</td><td>8.47</td><td>66.84</td><td>6.32</td></tr></tbody></table>


#### 5.3.The effect of different historical information.
#### 5.3. 不同历史信息的作用。


We now conduct a detailed experiment to deeply explore the role of historical information components. Currently, two main types of historical information are used in GUI agents: historical actions and historical screenshots. Note that the Contextual Information included in the semantic annotations of GUIOdyssey serves as a summary of previous steps, providing a more comprehensive textual representation of historical information. Therefore, we also include it in our experiments. Detailed results are presented in Table 4. Comparing experiments (2)-(4) with the baseline experiment (1), we observe that all three types of historical information significantly improve model performance, with contextual information producing the most substantial enhancement: improving AMS by 9.17 and SR by 240% compared to the baseline. A comparison between experiments (4) and (5) shows that using contextual information alone significantly improves out-of-domain performance compared to employing both actions and screenshots as historical input. This suggests that summarizing and abstracting historical information can better help the model generalize to unseen GUI scenarios. Additionally, experiment (6) shows that incorporating all types of historical information as input further enhances the model performance. We hypothesize that cross-app tasks inherently require more sophisticated memory mechanisms due to the dependencies and interactions between multiple apps. For example, as illustrated in Fig. 6 (Appendix Sec. 8.4), completing a cross-app task-such as identifying properties of triangles from Chrome and subsequently recording them in Google Docs-requires effectively remembering and transferring key information across apps. This example highlights the critical role historical information plays, underscoring the importance of comprehensive historical context modeling for GUI agents in complex cross-app scenarios.
我们现在进行详细实验，深入探究历史信息组成部分的作用。目前，在 GUI 代理中使用的两种主要历史信息类型为历史操作和历史屏幕截图。请注意，GUIOdyssey 的语义注释中包含的上下文信息，作为先前步骤的摘要，提供了对历史信息更全面的文本表示。因此，我们也将其计入实验。详细结果如表 4 所示。将实验（2）-（4）与基线实验（1）进行比较，我们发现三种历史信息类型都显著提升模型性能，其中上下文信息带来最显著的提升：相较于基线，AMS 提升 9.17，SR 提升 240%。对比实验（4）和（5），仅使用上下文信息在域外性能上显著优于同时使用历史操作和屏幕截图作为历史输入。这表明对历史信息进行概括和抽象更有助于模型在未见 GUI 场景中的泛化能力。此外，实验（6）显示将所有类型的历史信息作为输入，可以进一步提升模型性能。我们假设跨应用任务本质上需要更为复杂的记忆机制，因为涉及多应用之间的依赖与交互。例如，如 图 6（附录第 8.4 节）所示，完成跨应用任务（如在 Chrome 中识别三角形的属性并随后记录到 Google Docs）需要在应用之间有效地记住并传递关键信息。此示例强调了历史信息在复杂跨应用场景中对 GUI 代理的关键作用，并强调了对历史上下文进行全面建模的重要性。


More experiments. To deepen our analysis using GUIOdyssey, we conduct additional experiments detailed in Appendix Sec. 10. These include investigations of various strategies for handling historical screenshots, different semantic annotation components, transferability across devices, different instruction granularities, and the relationship between cross-app and single-app tasks.
更多实验。为深化我们对 GUIOdyssey 的分析，我们在附录第 10 节中进行了详细的额外实验。这些实验包括对处理历史屏幕截图的各种策略、不同语义注释组件、跨设备的可迁移性、不同指令粒度，以及跨应用与单应用任务之间关系的研究。


## 6. Conclusion
## 6. 结论


In this work, we address the limitations of existing GUI navigation agents for cross-app tasks by introducing GUIOdyssey, the first comprehensive cross-app mobile GUI navigation dataset enriched with semantic annotations. Leveraging this dataset, we develop OdysseyAgent, a multimodal cross-app navigation agent equipped with a history resampler module that efficiently processes historical image tokens to balance performance and inference speed. We conduct extensive experiments with OdysseyA-gent to evaluate our approach on both in-domain and out-of-domain scenarios. Our results further indicate that richer utilization of historical information can substantially enhance OdysseyAgent's performance. We hope GUIOdyssey and OdysseyAgent can drive the research in the field of general GUI Agents.
在本工作中，我们通过引入 GUIOdyssey — 第一个丰富语义注释的跨应用移动 GUI 导航数据集，来解决现有跨应用任务 GUI 导航代理的局限性。基于该数据集，我们开发了 OdysseyAgent，这是一种具备历史重采样模块、能高效处理历史图像标记以在性能与推理速度之间取得平衡的多模态跨应用导航代理。我们对 OdysseyAgent 进行了广泛的在域内和跨域场景的实验。结果进一步表明，更丰富地利用历史信息能够显著提升 OdysseyAgent 的性能。我们希望 GUIOdyssey 和 OdysseyAgent 能推动通用 GUI 代理领域的研究。


## Acknowledgments and Disclosure of Funding
## 致谢与资金披露


We thank Zhouheng Yao, Zihao Zhao for their help in data collection. This paper is partially supported by the National Key R & D Program of China No.2022ZD0160101 & No.2022ZD0161000.
感谢 Zhouheng Yao、Zihao Zhao 在数据收集方面的帮助。本文得到中国国家重点研发计划（No.2022ZD0160101 & No.2022ZD0161000）部分资助。


## References
## 参考文献


[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1, 2, 3, 7
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, 等. GPT-4 技术报告。arXiv 预印本 arXiv:2303.08774, 2023. 1, 2, 3, 7


[2] Anthropic. Claude, 2023. Accessed: 2023-04-18. 1, 3, 7
[2] Anthropic. Claude, 2023. 获取日期：2023-04-18。 1, 3, 7


[3] Gilles Baechler, Srinivas Sunkara, Maria Wang, Fedir Zubach, Hassan Mansoor, Vincent Etter, Victor Cărbune, Jason Lin, Jindong Chen, and Abhanshu Sharma. Screenai: A vision-language model for ui and infographics understanding. arXiv preprint arXiv:2402.04615, 2024. 3
[3] Gilles Baechler, Srinivas Sunkara, Maria Wang, Fedir Zubach, Hassan Mansoor, Vincent Etter, Victor Cărbune, Jason Lin, Jindong Chen, 和 Abhanshu Sharma. Screenai: 用于 UI 与信息图理解的视觉-语言模型。arXiv 预印本 arXiv:2402.04615, 2024. 3


[4] Chongyang Bai, Xiaoxue Zang, Ying Xu, Srinivas Sunkara, Abhinav Rastogi, Jindong Chen, et al. Uibert: Learning generic multimodal representations for ui understanding. arXiv preprint arXiv:2107.13731, 2021. 2, 3
[4] Chongyang Bai, Xiaoxue Zang, Ying Xu, Srinivas Sunkara, Abhinav Rastogi, Jindong Chen, 等. Uibert: 学习用于 UI 理解的通用多模态表示。arXiv 预印本 arXiv:2107.13731, 2021. 2, 3


[5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 5, 7
[5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, 与 Jingren Zhou. Qwen-vl: 具备多样能力的前沿大型视觉-语言模型。arXiv 预印本 arXiv:2308.12966, 2023. 5, 7


[6] Louise Barkhuus and Valerie E Polichar. Empowerment through seamfulness: smart phones in everyday life. Personal and Ubiquitous Computing, 15:629-639, 2011. 1
[6] Louise Barkhuus 与 Valerie E Polichar. 通过 seamfulness 实现的赋权：智能手机在日常生活中的应用。Personal and Ubiquitous Computing, 15:629-639, 2011. 1


[7] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimos-thenis Karatzas. Scene text visual question answering. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4291-4301, 2019. 7
[7] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimos-thenis Karatzas. 场景文本视觉问答。 发表在 IEEE/CVF 国际计算机视觉会议论文集，页码 4291-4301，2019。 7


[8] Andrea Burns, Deniz Arsan, Sanjna Agrawal, Ranjitha Kumar, Kate Saenko, and Bryan A Plummer. Mobile app tasks with iterative feedback (motif): Addressing task feasibility in interactive visual environments. arXiv preprint arXiv:2104.08560, 2021. 3, 4
[8] Andrea Burns, Deniz Arsan, Sanjna Agrawal, Ranjitha Kumar, Kate Saenko, and Bryan A Plummer. 使用迭代反馈的移动应用任务（motif）：在交互式视觉环境中解决任务可行性。 arXiv 预印本 arXiv:2104.08560，2021。 3, 4


[9] Yuxiang Chai, Siyuan Huang, Yazhe Niu, Han Xiao, Liang Liu, Dingyu Zhang, Peng Gao, Shuai Ren, and Hongsheng Li. Amex: Android multi-annotation expo dataset for mobile gui agents. arXiv preprint arXiv:2407.17490, 2024. 3, 4, 7
[9] Yuxiang Chai, Siyuan Huang, Yazhe Niu, Han Xiao, Liang Liu, Dingyu Zhang, Peng Gao, Shuai Ren, and Hongsheng Li. Amex：面向移动GUI代理的安卓多注释 Expo 数据集。 arXiv 预印本 arXiv:2407.17490，2024。 3, 4, 7


[10] Dongping Chen, Yue Huang, Siyuan Wu, Jingyu Tang, Liuyi Chen, Yilin Bai, Zhigang He, Chenlong Wang, Huichi Zhou, Yiqiang Li, et al. Gui-world: A dataset for gui-oriented multimodal llm-based agents. arXiv preprint arXiv:2406.10819, 2024. 3
[10] Dongping Chen, Yue Huang, Siyuan Wu, Jingyu Tang, Liuyi Chen, Yilin Bai, Zhigang He, Chenlong Wang, Huichi Zhou, Yiqiang Li, 等。Gui-world：一个面向GUI的多模态LLM代理数据集。 arXiv 预印本 arXiv:2406.10819，2024。 3


[11] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language models to versatile gui agents. arXiv preprint arXiv:2406.11317, 2024. 3
[11] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, 等。Guicourse：从通用视觉语言模型到多才多艺的GUI代理。 arXiv 预印本 arXiv:2406.11317，2024。 3


[12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 1, 3
[12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl：扩展视觉基础模型规模并为通用视觉-语言任务对齐。 arXiv 预印本 arXiv:2312.14238，2023。 1, 3


[13] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhang-wei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 7
[13] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhang-wei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, 等。我们离GPT-4V有多远？通过开源套件缩小与商用多模态模型的差距。 arXiv 预印本 arXiv:2404.16821，2024。 7


[14] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yan-tao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024. 2, 3
[14] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yan-tao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick：为高级视觉GUI代理利用GUI定位。 arXiv 预印本 arXiv:2401.10935，2024。 2, 3


[15] Zhang Chi, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771, 2023. 3
[15] Zhang Chi, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent：作为智能手机用户的多模态代理。 arXiv 预印本 arXiv:2312.13771，2023。 3


[16] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240): 1-113, 2023. 1
[16] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, 等。Palm：通过 Pathways 扩展语言模型能力。 Journal of Machine Learning Research, 24(240): 1-113, 2023。 1


[17] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 4
[17] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web：迈向网页的通用代理。 Advances in Neural Information Processing Systems, 36, 2024。 2, 3, 4


[18] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 1
[18] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, 等。The llama 3 herd of models. arXiv 预印本 arXiv:2407.21783，2024。 1


[19] Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Mat-suo, Aleksandra Faust, Shixiang Shane Gu, and Izzeddin Gur. Multimodal web navigation with instruction-finetuned foundation models. arXiv preprint arXiv:2305.11854, 2023. 3
[19] Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Mat-suo, Aleksandra Faust, Shixiang Shane Gu, and Izzeddin Gur. 具备指令微调的基础模型的多模态网页导航。 arXiv 预印本 arXiv:2305.11854, 2023. 3


[20] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024. 3
[20] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. 像人类一样在数字世界导航：面向 gui 代理的通用视觉定位。 arXiv 预印本 arXiv:2410.05243, 2024. 3


[21] Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856, 2023. 3
[21] Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. 具备计划能力、长上下文理解与程序合成的真实世界网页代理。 arXiv 预印本 arXiv:2307.12856, 2023. 3


[22] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. We-bvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024. 3
[22] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. We-bvoyager：基于大型多模态模型构建端到端网页代理。 arXiv 预印本 arXiv:2401.13919, 2024. 3


[23] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914, 2023. 3, 7
[23] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, 等. Cogagent：用于 gui 代理的可视语言模型。 arXiv 预印本 arXiv:2312.08914, 2023. 3, 7


[24] Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem Alshikh, and Ruslan Salakhut-dinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. arXiv preprint arXiv:2402.17553, 2024. 3
[24] Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem Alshikh, and Ruslan Salakhut-dinov. Omniact：一个数据集与基准，用于使桌面与网页的多模态通用自主代理成为可能。 arXiv 预印本 arXiv:2402.17553, 2024. 3


[25] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwe-barena: Evaluating multimodal agents on realistic visual web tasks. arXiv preprint arXiv:2401.13649, 2024. 3
[25] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwe-barena：在真实世界视觉网页任务上评估多模态代理。 arXiv 预印本 arXiv:2401.13649, 2024. 3


[26] Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. On the effects of data scale on computer control agents. arXiv preprint arXiv:2406.03679, 2024. 1, 3, 4, 5
[26] Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. 数据规模对计算机控制代理影响的研究。 arXiv 预印本 arXiv:2406.03679, 2024. 1, 3, 4, 5


[27] Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge. Mapping natural language instructions to mobile ui action sequences. arXiv preprint arXiv:2005.03776, 2020. 3,4
[27] Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge. 将自然语言指令映射到移动 UI 动作序列。 arXiv 预印本 arXiv:2005.03776, 2020. 3,4


[28] Yanda Li, Chi Zhang, Wanqi Yang, Bin Fu, Pei Cheng, Xin Chen, Ling Chen, and Yunchao Wei. Appagent v2: Advanced agent for flexible mobile interactions. arXiv preprint arXiv:2408.11824, 2024. 1, 3
[28] Yanda Li, Chi Zhang, Wanqi Yang, Bin Fu, Pei Cheng, Xin Chen, Ling Chen, and Yunchao Wei. Appagent v2：用于灵活移动交互的高级代理。 arXiv 预印本 arXiv:2408.11824, 2024. 1, 3


[29] Zhangheng Li, Keen You, Haotian Zhang, Di Feng, Harsh Agrawal, Xiujun Li, Mohana Prasad Sathya Moorthy, Jeff Nichols, Yinfei Yang, and Zhe Gan. Ferret-ui 2: Mastering universal user interface understanding across platforms. arXiv preprint arXiv:2410.18967, 2024. 3
[29] Zhangheng Li, Keen You, Haotian Zhang, Di Feng, Harsh Agrawal, Xiujun Li, Mohana Prasad Sathya Moorthy, Jeff Nichols, Yinfei Yang, and Zhe Gan. Ferret-ui 2：跨平台掌握通用用户界面理解。 arXiv 预印本 arXiv:2410.18967, 2024. 3


[30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 3
[30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 视觉指令微调。神经信息处理系统进展，36，2024。 3


[31] Junpeng Liu, Yifan Song, Bill Yuchen Lin, Wai Lam, Graham Neubig, Yuanzhi Li, and Xiang Yue. Visualwebbench: How far have multimodal llms evolved in web page understanding and grounding? arXiv preprint arXiv:2404.05955, 2024. 3
[31] Junpeng Liu, Yifan Song, Bill Yuchen Lin, Wai Lam, Graham Neubig, Yuanzhi Li, and Xiang Yue. Visualwebbench：多模态大模型在网页理解与定位方面的进展如何？ arXiv 预印本 arXiv:2404.05955, 2024. 3


[32] Xing Han Lù, Zdeněk Kasner, and Siva Reddy. We-blinx: Real-world website navigation with multi-turn dialogue. arXiv preprint arXiv:2402.05930, 2024. 3, 4
[32] 邢汉录、Zdeněk Kasner 与 Siva Reddy. We-blinx: 具有多轮对话的现实网页导航。arXiv 预印本 arXiv:2402.05930，2024。 3, 4


[33] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent. arXiv preprint arXiv:2408.00203, 2024. 3, 7
[33] Yadong Lu、Jianwei Yang、Yelong Shen 与 Ahmed Awadallah。Omniparser 用于纯视觉基础 GUI 代理。arXiv 预印本 arXiv:2408.00203，2024。 3, 7


[34] Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pre-training via embodied chain of thought. Advances in Neural Information Processing Systems, 36, 2024. 3
[34] Yao Mu、Qinglong Zhang、Mengkang Hu、Wenhai Wang、Mingyu Ding、Jun Jin、Bin Wang、Jifeng Dai、Yu Qiao 与 Ping Luo。Embodiedgpt：通过具身思维链进行视觉-语言预训练。神经信息处理系统进展，36，2024。 3


[35] Amal Nanavati, Vinitha Ranganeni, and Maya Cakmak. Physically assistive robots: A systematic review of mobile and manipulator robots that physically assist people with disabilities. Annual Review of Control, Robotics, and Autonomous Systems, 7, 2023. 1
[35] Amal Nanavati、Vinitha Ranganeni 与 Maya Cakmak。Physically assistive robots: 面向残障人士的移动与机械臂机器人之系统综述。控制、机器人与自主系统年评，7，2023。 1


[36] OpenAI. Gpt40, 2024. 2, 7
[36] OpenAI。Gpt40，2024。 2, 7


[37] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junt-ing Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feicht-enhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 6, 2
[37] Nikhila Ravi、Valentin Gabeur、Yuan-Ting Hu、Ronghang Hu、Chaitanya Ryali、Tengyu Ma、Haitham Khedr、Roman Rädle、Chloe Rolland、Laura Gustafson、Eric Mintun、Junt-ing Pan、Kalyan Vasudev Alwala、Nicolas Carion、Chao-Yuan Wu、Ross Girshick、Piotr Dollár 与 Christoph Feicht-enhofer。Sam 2：在图像与视频中对任意对象进行分割。arXiv 预印本 arXiv:2408.00714，2024。 6, 2


[38] Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Android in the wild: A large-scale dataset for android device control. arXiv preprint arXiv:2307.10088, 2023. 1, 2, 3, 5, 6
[38] Christopher Rawles、Alice Li、Daniel Rodriguez、Oriana Riva 与 Timothy Lillicrap。野外的安卓：用于安卓设备控制的大规模数据集。arXiv 预印本 arXiv:2307.10088，2023。 1, 2, 3, 5, 6


[39] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024. 1,3
[39] Christopher Rawles、Sarah Clinckemaillie、Yifan Chang、Jonathan Waltz、Gabrielle Lau、Marybeth Fair、Alice Li、William Bishop、Wei Li、Folawiyo Campbell-Ajala 等。Androidworld：用于自治代理的动态基准环境。arXiv 预印本 arXiv:2405.14573，2024。 1,3


[40] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023. 1
[40] Baptiste Roziere、Jonas Gehring、Fabian Gloeckle、Sten Sootla、Itai Gat、Xiaoqing Ellen Tan、Yossi Adi、Jingyu Liu、Tal Remez、Jérémy Rapin 等。Code llama：用于代码的开源基础模型。arXiv 预印本 arXiv:2308.12950，2023。 1


[41] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2024. 3
[41] Timo Schick、Jane Dwivedi-Yu、Roberto Dessì、Roberta Raileanu、Maria Lomeli、Eric Hambro、Luke Zettlemoyer、Nicola Cancedda 与 Thomas Scialom。Toolformer：语言模型能够自行学习使用工具。神经信息处理系统进展，36，2024。 3


[42] Peter Shaw, Mandar Joshi, James Cohan, Jonathan Berant, Panupong Pasupat, Hexiang Hu, Urvashi Khandelwal, Kenton Lee, and Kristina Toutanova. From pixels to ui actions: Learning to follow instructions via graphical user interfaces. In Advances in Neural Information Processing Systems, 2023. 3
[42] Peter Shaw、Mandar Joshi、James Cohan、Jonathan Berant、Panupong Pasupat、Hexiang Hu、Urvashi Khandelwal、Kenton Lee 与 Kristina Toutanova。从像素到用户界面操作：通过图形用户界面学习遵循指令。在神经信息处理系统进展，2023。 3


[43] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36, 2024. 3
[43] Yongliang Shen、Kaitao Song、Xu Tan、Dongsheng Li、Weiming Lu 与 Yueting Zhuang。Hugginggpt：在 hugging face 中用 chatgpt 及其伙伴解决 ai 任务。神经信息处理系统进展，36，2024。 3


[44] Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. World of bits: An open-domain platform for web-based agents. In Proceedings of the 34th International Conference on Machine Learning, pages 3135- 3144. PMLR, 2017. 3
[44] Tianlin Shi、Andrej Karpathy、Linxi Fan、Jonathan Hernandez、Percy Liang。Bits 的世界：一个开放域的基于网络的代理平台。收录于第34届国际机器学习会议论文集，页码 3135-3144。PMLR，2017。 3


[45] Liangtai Sun, Xingyu Chen, Lu Chen, Tianle Dai, Zichen Zhu, and Kai Yu. Meta-gui: towards multi-modal conversational agents on mobile gui. arXiv preprint arXiv:2205.11029, 2022. 3
[45] Liangtai Sun、Xingyu Chen、Lu Chen、Tianle Dai、Zichen Zhu、Kai Yu。Meta-gui：面向移动 GUI 的多模态对话代理。arXiv 预印本 arXiv:2205.11029，2022。 3


[46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
[46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: 开放高效的基础语言模型。 arXiv 预印本 arXiv:2302.13971, 2023. 3


[47] Sagar Gubbi Venkatesh, Partha Talukdar, and Srini Narayanan. Ugif: Ui grounded instruction following. arXiv preprint arXiv:2211.07615, 2022. 3, 4
[47] Sagar Gubbi Venkatesh, Partha Talukdar, and Srini Narayanan. Ugif: Ui grounded instruction following. arXiv 预印本 arXiv:2211.07615, 2022. 3, 4


[48] Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. arXiv preprint arXiv:2406.01014, 2024. 1, 3
[48] Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. arXiv 预印本 arXiv:2406.01014, 2024. 1, 3


[49] Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024. 3
[49] Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv 预印本 arXiv:2401.16158, 2024. 3


[50] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Jun-yang Lin. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 3
[50] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Jun-yang Lin. Qwen2-vl: 提升视觉-语言模型在任意分辨率下对世界的感知。 arXiv 预印本 arXiv:2409.12191, 2024. 1, 3


[51] Zhiping Paul Wang, Priyanka Bhandary, Yizhou Wang, and Jason H Moore. Using gpt-4 to write a scientific review article: a pilot evaluation study. bioRxiv, pages 2024-04, 2024. 1
[51] Zhiping Paul Wang, Priyanka Bhandary, Yizhou Wang, and Jason H Moore. Using gpt-4 to write a scientific review article: a pilot evaluation study. bioRxiv, 2024-04, 2024. 1


[52] Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. Os-copilot: Towards generalist computer agents with self-improvement. arXiv preprint arXiv:2402.07456, 2024. 3
[52] Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. Os-copilot: Towards generalist computer agents with self-improvement. arXiv 预印本 arXiv:2402.07456, 2024. 3


[53] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024. 3
[53] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv 预印本 arXiv:2410.23218, 2024. 3


[54] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. arXiv preprint arXiv:2404.07972, 2024. 1, 3
[54] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. arXiv 预印本 arXiv:2404.07972, 2024. 1, 3


[55] Mingzhe Xing, Rongkai Zhang, Hui Xue, Qi Chen, Fan Yang, and Zhen Xiao. Understanding the weakness of large language model agents within a complex android environment. arXiv preprint arXiv:2402.06596, 2024. 1, 3
[55] Mingzhe Xing, Rongkai Zhang, Hui Xue, Qi Chen, Fan Yang, and Zhen Xiao. Understanding the weakness of large language model agents within a complex android environment. arXiv 预印本 arXiv:2402.06596, 2024. 1, 3


[56] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023. 3
[56] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv 预印本 arXiv:2304.12244, 2023. 3


[57] An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin. Lin-jie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, et al. Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation. arXiv preprint arXiv:2311.07562, 2023. 3
[57] An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin. Lin-jie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, et al. Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation. arXiv 预印本 arXiv:2311.07562, 2023. 3


[58] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 3
[58] 杨建伟、张浩、李锋、邹雪艳、李春元、郜建风。集合提示引发GPT-4V的卓越视觉对齐。arXiv 预印本 arXiv:2310.11441，2023. 3


[59] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023. 3
[59] 杨正远、李麟杰、王建峰、林凯文、艾哈桑·阿扎尔纳萨卜、法伊萨尔·艾哈迈德、刘子诚、Ce Liu、Michael Zeng、王丽娟。Mm-react：提示ChatGPT进行多模态推理与行动。arXiv 预印本 arXiv:2303.11381，2023. 3


[60] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744-20757, 2022. 3
[60] 姚瞬玉、陈浩、杨俊、Narasimhan 卡斯特克。Webshop：通过有据语言代理实现可扩展的现实世界网页互动。神经信息处理系统进展，35:20744-20757，2022. 3


[61] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022. 3
[61] 姚瞬玉、赵杰弗里、余颂、段娜、伊扎克·沙夫兰、Narasimhan 卡斯特、Yuan Cao。React：在语言模型中协同推理与行动。arXiv 预印本 arXiv:2210.03629，2022. 3


[62] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, et al. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. arXiv preprint arXiv:2404.16006, 2024. 1
[62] 应凯宁、孟范庆、王晋、李智乾、林汉、杨越、张浩、张文博、林宇琦、刘朔、等。Mmt-bench：用于评估大型视觉-语言模型向多任务通向通用人工智能的全面多模态基准。arXiv 预印本 arXiv:2404.16006，2024. 1


[63] Keen You, Haotian Zhang, Eldon Schoop, Floris Weers, Amanda Swearngin, Jeffrey Nichols, Yinfei Yang, and Zhe Gan. Ferret-ui: Grounded mobile ui understanding with multimodal llms. In European Conference on Computer Vision, pages 240-255. Springer, 2025. 3
[63] 郭可宁、张浩天、埃尔登·修普、弗洛里斯·韦尔斯、阿曼达·斯瓦恩金、尼科尔斯、杨音菲、贺赐·甘。Ferret-ui：具多模态大模型的移动UI理解的基础。欧洲计算机视觉大会论文集，页码 240-255。Springer，2025. 3


[64] Zhuosheng Zhan and Aston Zhang. You only look at screens: Multimodal chain-of-action agents. arXiv preprint arXiv:2309.11436, 2023. 6
[64] 盧娅·詹和阿斯顿·张。你只看屏幕：多模态行动链代理。arXiv 预印本 arXiv:2309.11436，2023. 6


[65] Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Raj-mohan, et al. Ufo: A ui-focused agent for windows os interaction. arXiv preprint arXiv:2402.07939, 2024. 3
[65] 张超云、李力群、贺思林、张旭、乔博、秦思、马明华、康宇、林庆威、萨拉万·Raj-mohan 等。Ufo：一个面向Windows OS交互的UI聚焦代理。arXiv 预印本 arXiv:2402.07939，2024. 3


[66] Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for gui agents. arXiv preprint arXiv:2403.02713, 2024. 3, 4
[66] 张纪文、吴吉豪、滕一华、廖明辉、许诺、萧熙、魏忠宇、唐都羽。Android in the zoo：GUI代理的行动-思维链。arXiv 预印本 arXiv:2403.02713，2024. 3、4


[67] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v(ision) is a generalist web agent, if grounded. In Forty-first International Conference on Machine Learning, 2024. 3
[67] 郑博远、沟博宇、基尔基赫、孙欢、苏瑜。Gpt-4v(ision)是一个通用网页代理，但要有据。第41届机器学习国际会议，2024。3


[68] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023. 3
[68] 周舒彦、徐芳静、朱昊、周徐熙、罗罗伯特、斯里达·阿比舍克、程显、比斯克·约南坦、弗里德·丹尼尔、阿龙·尤里等。Webarena：用于构建自治代理的现实网页环境。arXiv 预印本 arXiv:2307.13854，2023. 3


[69] Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Wei-jie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiao-gang Wang, et al. Ghost in the minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory. arXiv preprint arXiv:2305.17144, 2023. 1
[69] 朱熙周、陈云涛、田浩、陶晨欣、苏巍杰、杨晨烽、黄高、李本、卢乐威、王小刚、等。Minecraft中的鬼魂：通过大模型的文本知识与记忆实现对开放世界环境的一般能力代理。arXiv 预印本 arXiv:2305.17144，2023. 1