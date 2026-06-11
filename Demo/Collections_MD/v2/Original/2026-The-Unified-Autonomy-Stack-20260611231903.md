# The Unified Autonomy Stack: Toward a Blueprint for Generalizable Robot Autonomy
# 统一自主栈：迈向通用机器人自主性的蓝图


---



Journal Title
期刊标题


XX(X):1-35
XX(X):1-35


©The Author(s) 2026
©作者 2026


Reprints and permission:
转载与许可：


sagepub.co.uk/journalsPermissions.nav
sagepub.co.uk/journalsPermissions.nav


DOI: 10.1177/ToBeAssigned
DOI: 10.1177/ToBeAssigned


www.sagepub.com/
www.sagepub.com/


SAGE



---



Mihir Dharmadhikari*, Nikhil Khedekar*, Mihir Kulkarni*, Morten Nissov*, Martin Jacquet, Angelos Zacharia, Marvin Harms, Albert Gassol Puigjaner, Philipp Weiss, Kostas Alexis
Mihir Dharmadhikari*, Nikhil Khedekar*, Mihir Kulkarni*, Morten Nissov*, Martin Jacquet, Angelos Zacharia, Marvin Harms, Albert Gassol Puigjaner, Philipp Weiss, Kostas Alexis


## Abstract
## 摘要


We introduce and open-source the Unified Autonomy Stack, a system-level solution that enables resilient autonomy across diverse aerial and ground robot morphologies. The architecture centers on three synergistic modules -multi-modal perception, multi-behavior planning, and multi-layered safe navigation- that together deliver comprehensive mission autonomy. The stack fuses data from LiDAR, radar, vision, and inertial sensing, enabling (a) robust localization and mapping through factor graph-based fusion, (b) semantic scene understanding, (c) motion and informative path planning through sampling-based techniques adaptive across spatial scales, as well as (d) multilayered safe navigation both through planning on the online reconstructed map and deep learning-driven exteroceptive policies alongside last-resort safety filters using control barrier functions. The resulting behaviors include safe GNSS-denied navigation into unknown and perceptually-degraded regions, exploration of complex environments, object discovery, and efficient inspection planning. The stack has been field-tested and validated on both aerial (rotorcraft) and ground (legged) robots operating in a host of demanding environments, including self-similar and smoke-filled settings, with complex geometries and high obstacle clutter. These tests demonstrate resilient performance in challenging conditions. To facilitate ease of adoption, we open-source the implementation alongside supporting documentation, validation, and evaluation datasets https://github.com/ntnu-arl/unified_autonomy_stack.A video giving the overview of the paper and the field experiments is available at https://youtu.be/18Su8OXsM-E.
我们推出并开源统一自主栈，这是一种系统级解决方案，可在多样的空中与地面机器人形态间实现稳健自主。该架构以三个协同模块为核心——多模态感知、多行为规划和多层安全导航——共同实现全面的任务自主。该栈融合来自激光雷达、雷达、视觉和惯性传感的数据，实现(a)基于因子图融合的稳健定位与建图，(b)语义场景理解，(c)通过在空间尺度上自适应的采样式技术进行运动与信息路径规划，以及(d)既通过在线重建地图上的规划、又借助深度学习驱动的外感知策略实现的多层安全导航，并辅以使用控制屏障函数的最后一道安全过滤器。由此产生的行为包括：在 GNSS 受限、未知且感知退化区域中的安全导航，复杂环境探索，目标发现，以及高效的巡检规划。该栈已在多种严苛环境中完成实地测试与验证，涵盖空中（旋翼）与地面（足式）机器人，包括自相似和充满烟雾的场景，以及具有复杂几何结构和高障碍物密度的环境。这些测试表明其在挑战性条件下具有稳健性能。为便于采用，我们开源了实现、配套文档、验证与评估数据集 https://github.com/ntnu-arl/unified_autonomy_stack。论文概述与实地实验视频可见于 https://youtu.be/18Su8OXsM-E。


## Keywords
## 关键词


Autonomy, Perception, Planning, Navigation
自主性、感知、规划、导航


## 1 Introduction
## 1 引言


Mobile robots are increasingly deployed to operate in environments where Global Navigation Satellite System (GNSS) is unavailable, perception is degraded, geometry is self-similar, and safe navigation is broadly challenged Tran-zatto et al. (2022); Harlow et al. (2024); Ebadi et al. (2023); Datar et al. (2025); Chung et al. (2023). Although the robotics community has developed mature components for many individual functionalities such as localization, planning and control, existing autonomy stacks (e.g., the works in Fernandez-Cortizas et al. (2023); Sanchez-Lopez et al. (2016); Baca et al. (2021); Mohta et al. (2018); Foehn et al. (2022); Goodin et al. (2024); AirLab (2025); Real et al. (2020)) remain largely specialized to a particular robot morphology, sensor suite, or mission class. This specialization limits reuse across platforms, makes systematic field evaluation difficult, slows the accumulation of shared deployment experience, and hinders both consolidation and broader uptake of autonomy across robot categories.
移动机器人正越来越多地部署到这样一些环境中：全球导航卫星系统（GNSS）不可用、感知退化、几何具有自相似性，而安全导航面临广泛挑战 Tran-zatto 等（2022）；Harlow 等（2024）；Ebadi 等（2023）；Datar 等（2025）；Chung 等（2023）。尽管机器人领域已为许多单项功能（如定位、规划与控制）开发出成熟组件，但现有自主系统（例如 Fernandez-Cortizas 等（2023）；Sanchez-Lopez 等（2016）；Baca 等（2021）；Mohta 等（2018）；Foehn 等（2022）；Goodin 等（2024）；AirLab（2025）；Real 等（2020）的工作）仍在很大程度上专用于特定的机器人形态、传感器配置或任务类别。这种专用化限制了跨平台复用，使系统性的现场评估变得困难，减缓了共享部署经验的积累，并阻碍了跨机器人类别的自主能力整合与更广泛采用。


At the same time, limited attention has been given to feature-rich autonomy stacks readily delivering complex behaviors and functionalities such as resilient GNSS-denied navigation in perceptually-degraded environments, combined with sophisticated informative path planning and assured safety ensuring robust operation across operational environments and conditions. However, recent advances across the "sense-think-act" loop -from perception to planning and deep control policies- point toward the potential for a resilient and, to a significant extent, unified autonomy engine. Although research on universal autonomy is still in its early stages, the benefits of unification and the collective need to advance robot capabilities underscore the need for general autonomy solutions.
与此同时，鲜有关注能够即刻提供丰富特征的自主系统：例如在感知退化的环境中实现具有韧性的 GNSS 受限导航，并结合复杂的信息引导路径规划与确保安全的机制，从而在不同运行环境与条件下实现可靠运转。然而，近期在“感知-思考-行动”闭环——从感知到规划与深度控制策略——方面的进展表明，可能存在一个具有韧性、且在相当程度上统一的自主引擎。尽管关于通用自主的研究仍处于早期阶段，但统一带来的益处以及共同推进机器人能力的需求，凸显了对一般化自主解决方案的必要性。


Motivated by the above, we present the Unified Autonomy stack (UAstack), a comprehensive open-source autonomy stack that can support mission-level operation across diverse aerial and ground robot configurations. The UAstack represents a step towards a common autonomy blueprint across diverse robot types -from multirotors and other rotorcrafts to ground robots such as legged systems-delivering mission-complete capabilities for navigation and complex information sampling behaviors (e.g., exploration, inspection, object discovery) in diverse settings, including in strenuous, high-risk natural and industrial environments. Its design emphasizes resilience in that it presents robustness (e.g., against noisy sensor data and mapping imperfections), resourcefulness (e.g., multiple solutions for safety in navigation), and redundancy (e.g., complementary sensor data to handle single-modality failures), enabling it to retain high performance across environments and conditions, including GNSS-denied, perceptually-degraded, geometrically complex, and potentially adversarial settings that typically challenge safe navigation and mission autonomy.
基于上述动机，我们提出统一自主栈（Unified Autonomy stack，UAstack）：一个全面的开源自主系统，可支持面向任务级别的运行，覆盖多样的空中与地面机器人配置。UAstack 体现了面向不同机器人类型的共同自主蓝图迈出的一步——从多旋翼及其他旋翼飞行器到地面机器人（如腿式系统）——在多种场景中提供任务级完成能力，包括导航与复杂信息采样行为（例如探索、巡检、目标发现）。这些场景包含严苛、高风险的自然与工业环境。其设计强调韧性：它提供鲁棒性（例如对噪声传感数据与建图不完善的抵抗）、机敏性（例如在导航安全方面提供多种方案）以及冗余性（例如互补的传感器数据以应对单一模态失效）。因此，它能够在各种环境与条件下保持高性能，包括 GNSS 受限、感知退化、几何复杂以及可能对抗性的设置；而这些恰恰是通常会挑战安全导航与任务自主性的因素。


---



All authors are affiliated with the Autonomous Robots Lab, Norwegian University of Science and Technology (NTNU), Trondheim, Norway
所有作者均隶属于挪威科技与自然科学大学（NTNU）自主机器人实验室，位于挪威特隆赫姆


## Corresponding author:
## 通讯作者：


Kostas Alexis, Department of Engineering Cybernetics, Norwegian University of Science and Technology, Høgskoleringen 1, Trondheim 7034, Norway. * indicates equal contribution.
Kostas Alexis，工程控制学系，挪威科技大学，Høgskoleringen 1，Trondheim 7034，挪威。* 表示同等贡献。


Email: konstantinos.alexis@ntnu.no
邮箱：konstantinos.alexis@ntnu.no


---



<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_1.jpg?x=140&y=143&w=1369&h=1202&r=0"/>



Figure 1. Indicative subset of the evaluation studies conducted to validate and assess the performance of the Unified Autonomy Stack. The tests involve both aerial and ground robots operating in diverse GNSS-denied and at instances perceptually-degraded environments including (a) snow-covered forests, (b) underground mines, (c) road tunnels, (d) a frozen lake, (e) ship cargo holds, as well as (f) the university campus with subsets of it filled with fog. All core modules of the autonomy stack were evaluated independently, alongside full-stack experiments. In this image "FS" stands for Full-Stack deployments (involving the synergy of the perception module, planning module, and navigation module), "N" for deployments to evaluate the navigation module, "PF" for testing both the localization and mapping as well as object-level reasoning of the perception module, while "PS" marks tests focusing only on localization and mapping. The presented instances are from a subset of the experiments as detailed in the paper. The full datasets of the experiments are also released to support verifiability.
图1. 用于验证并评估统一自主栈（Unified Autonomy Stack）性能的评估研究的示例子集。测试同时包含在多种无GNSS以及在某些情况下感知受损环境中运行的空中与地面机器人，包括（a）积雪覆盖的森林，（b）地下矿井，（c）道路隧道，（d）结冰的湖泊，（e）船舶货舱，以及（f）大学校园，其部分区域充满雾气。所有自主栈的核心模块均被独立评估，同时也进行了全栈实验。在该图中，“FS”表示全栈部署（融合感知模块、规划模块与导航模块的协同）， “N”表示用于评估导航模块的部署，“PF”表示测试感知模块的定位与建图，以及目标级推理；而“PS”表示仅关注定位与建图的测试。所展示的实例来自论文中所述实验的一个子集。为支持可核验性，实验的完整数据集也已发布。


The UAstack builds upon three core modules and associated contributions. The perception module is centered around a novel approach to multi-modal Simultaneous Localization And Mapping (SLAM) based on factor graphs, enabling robust fusion of LiDAR, Frequency Modulated Continuous Wave (FMCW) radar, visual perception, and Inertial Measurement Unit (IMU) cues. This supports resilient performance in GNSS-denied environments with multiple perceptual degradations, including geometric self-similarity, low texture, icy scenes, and dense obscurants (e.g., fog, smoke). Furthermore, the perception module integrates scene reasoning through Vision-Language Models (VLMs) enabling object discovery and visual question & answering (Q&A). The planning module builds upon OmniPlanner Zacharia et al. (2026), facilitating target reaching, exploration, and inspection path planning through sampling-based methods over an online-derived volumetric map of the environment. It provides a versatile framework that abstracts vehicle configuration and supports diverse mission objectives, which the planner optimizes accordingly. The navigation module builds upon the contributions in Jacquet et al. (2025); Harms et al. (2025), alongside introducing a novel approach to exteroceptive reinforcement learning for navigation. Recognizing that localization errors and mapping imperfections may arise, we adopt a redundant, multi-layered safety approach in which depth-based exteroceptive navigation policies and last-resort control barrier function-based safety filters enhance safety by providing direct and reactive collision avoidance. As a result, the UAstack is tailored to environments that challenge localization and mapping, both by leveraging sensor multimodality to maintain perception under degraded conditions, and by implementing reactive safety mechanisms to reduce reliance on perfect scene understanding.
UAstack 基于三个核心模块及其相关贡献构建。感知模块围绕一种面向多模态的创新方法：基于因子图的同步定位与建图（Simultaneous Localization And Mapping，SLAM），从而实现对 LiDAR、频率调制连续波（Frequency Modulated Continuous Wave, FMCW）雷达、视觉感知以及惯性测量单元（IMU）线索的可靠融合。这支持在无GNSS环境中、在多种感知退化条件下仍能保持稳健性能，包括几何自相似、低纹理、结冰场景以及密集遮挡物（例如雾、烟）。此外，感知模块通过视觉-语言模型（Vision-Language Models, VLMs）集成场景推理，实现目标发现与视觉问答（Q&A）。规划模块基于 OmniPlanner Zacharia et al.（2026），借助基于采样的方法，在在线生成的环境体积地图上进行目标到达、探索与检视路径规划，并提供一个灵活的框架：抽象车辆配置，并支持多样的任务目标，规划器会据此进行优化。导航模块基于 Jacquet et al.（2025）；Harms et al.（2025）的相关贡献，同时提出一种用于导航的外部感知强化学习新方法。考虑到定位误差与建图不完善可能出现，我们采用冗余的、多层级安全方案：由基于深度的外部感知导航策略与基于“最后手段”的控制屏障函数安全过滤器共同提升安全性，通过提供直接且具有反应性的避碰能力来增强保障。因此，UAstack 针对那些对定位与建图提出挑战的环境进行了定制：一方面利用传感器多模态在退化条件下维持感知，另一方面引入反应式安全机制以降低对完美场景理解的依赖。


To evaluate the performance and assess its resilience, the UAstack is evaluated onboard multiple robot configurations and within a diverse set of environments, an indicative subset of which is shown in Figure 1. First, a detailed quantitative evaluation of the perception module is presented, covering 2 urban tunnels, a frozen lake and a university campus environment characterized by geometric self-similarity, low visibility, and heavy airborne obscurants. It displays the superior performance of the perception module compared to State-of-the-Art LiDAR-Inertial, LiDAR-Radar-Inertial, LiDAR-Visual-Inertial, or Visual-Inertial SLAM methods. Object-level reasoning is assessed by building 3D scene graphs with object-level annotations, alongside enabling visual Q&A on online camera data. Next, the navigation module is evaluated in two real-world deployments. First, in a forest environment for a waypoint-navigation task requiring maneuvering to avoid trees, while map-based path planning is disabled to isolate the reactive layer. Second, we evaluate safety under map discrepancies by introducing previously unmapped obstacles along the path planned by the planning module and demonstrate that the navigation module layer can handle such unseen obstacles. These experiments highlight the importance of the multi-layered safety approach and the complementary roles of each safety method. Finally, the full UAstack is evaluated on aerial and legged robots performing autonomous exploration and inspection missions guided by the planning module. The aerial robot is deployed in a) a low-visibility, multi-branched underground mine, and b) a forest with thin obstacles and local clutter, performing exploration missions, with navigation module modalities being tested. Additionally, the aerial robot is deployed in the cargo hold of a ship performing an exploration and inspection mission. On the other hand, the legged robot is deployed in a university campus, and inside the same underground mine as the aerial robot. This demonstrates large-scale missions across heterogeneous platforms and environments.
为评估其性能并检验其鲁棒性，UAstack 在多种机器人配置以及多样化环境中进行了评估，其中一个示例子集如图1所示。首先，给出感知模块的详细定量评估，覆盖两个城市隧道、一个结冰湖泊以及一个大学校园环境；该环境具有几何自相似、低能见度与大量悬浮遮挡物等特征。结果表明，感知模块相较于现有技术（State-of-the-Art）的 LiDAR-惯性、LiDAR-雷达-惯性、LiDAR-视觉-惯性或视觉-惯性 SLAM 方法表现更优。通过为带有目标级标注的三维场景图构建来评估目标级推理，同时借助在线相机数据支持视觉问答。接下来，在两次真实场景部署中评估导航模块。首先，在森林环境中进行航点导航任务：需要在绕开树木的同时完成机动；同时禁用基于地图的路径规划，以隔离反应式层。其次，我们通过在规划模块规划的路径上引入先前未建图的障碍物来评估在地图不一致下的安全性，并证明导航模块层能够处理此类未见障碍物。这些实验凸显了多层级安全方案的重要性，以及各类安全方法之间互补的作用。最后，在空中与足式机器人上评估完整的 UAstack：它们执行由规划模块引导的自主探索与检视任务。空中机器人分别部署在（a）低能见度、多分支的地下矿井中，以及（b）具有细小障碍物与局部杂乱的森林中，进行探索任务，同时对导航模块的各类模态进行测试。此外，空中机器人还部署在船舶货舱内，执行探索与检视任务。另一方面，足式机器人部署在大学校园，并且与空中机器人处于同一座地下矿井内部。这表明能够在异构平台与多种环境上完成大规模任务。


Importantly, the Unified Autonomy Stack is open to extension, both from the perspective of the robots it readily supports and the missions it enables. Our team is targeting its full-fledged expansion to a diverse set of robot configurations and the extension of the enabled behaviors. To that end, the full implementation is open-sourced (https://github.com/ntnu-arl/unified_ autonomy_stack), alongside documentation and supporting datasets (https://ntnu-arl.github.io/unified_autonomy_stack/).
重要的是，统一自主栈（Unified Autonomy Stack）可在原有基础上扩展，无论是从它所能直接支持的机器人角度，还是从它所能实现的任务角度。我们的团队正着力将其全面扩展到多样化的机器人配置，并扩展所启用的行为。为此，完整实现已开源（https://github.com/ntnu-arl/unified_ autonomy_stack），同时配套了文档与支持数据集（https://ntnu-arl.github.io/unified_autonomy_stack/）。


The remainder of this paper is structured as follows. Section 2 overviews related work in autonomy stacks. The UAstack and its modules are detailed in Section 3. Evaluation studies are presented in Section 4, followed by conclusions and plans for future work in Section 5.
本文其余部分结构如下。第2节概述自主栈相关工作。UAstack 及其模块将在第3节详细说明。第4节给出评估研究，随后在第5节提出结论与未来工作的计划。


## 2 Related Work
## 2 相关工作


Research in autonomy is a particularly active field. The robotics community has increasingly released reusable autonomy components, while more complete autonomy stacks remain less common. Driven by the major success of open-source autopilot releases, such as PX4 and ArduPilot Meier et al. (2015), these developments aim to accelerate research on robot autonomy and democratize its adoption through open implementations and standards. In this section, we review relevant literature and position how the UAstack contributes to this landscape.
自主领域的研究是一项特别活跃的方向。机器人社区已越来越多地发布可复用的自主组件，而更完整的自主堆栈仍较少。受诸如 PX4 和 ArduPilot Meier et al. (2015) 等开源飞控发布的巨大成功推动，这些进展旨在加速对机器人自主性的研究，并通过开放实现与标准来推动其普及。本节将回顾相关文献，并阐明 UAstack 如何融入这一领域。


Typically, major building blocks of autonomy are often documented and released as separate publications and open-source contributions. Indicative examples span SLAM Xu et al. (2022); Vizzo et al. (2023); Koide et al. (2024); Campos et al. (2021); Geneva et al. (2020a); Qin et al. (2019), motion and path planning Sucan et al. (2012); Wang et al. (2022); Mellinger and Kumar (2011); Likhachev et al. (2003), control Verschueren et al. (2022); Andersson et al. (2018); Giftthaler et al. (2018), robot learning Petrenko et al. (2020); Makoviichuk and Makoviychuk (2021); Schwarke et al. (2025); NVIDIA et al. (2025); Zakka et al. (2025), and navigation frameworks Macenski et al. (2020); Planning (2012). While these works provide an important foundation for robot autonomy, they are not the focus of this section. Instead, this section assesses and compares existing integrated, open-source autonomy stacks for mobile robots, including aerial and ground systems.
通常，自主的主要构建模块往往以独立出版物和开源贡献的形式被记录并发布。典型示例涵盖 SLAM：Xu et al. (2022); Vizzo et al. (2023); Koide et al. (2024); Campos et al. (2021); Geneva et al. (2020a); Qin et al. (2019)，运动与路径规划：Sucan et al. (2012); Wang et al. (2022); Mellinger and Kumar (2011); Likhachev et al. (2003)，控制：Verschueren et al. (2022); Andersson et al. (2018); Giftthaler et al. (2018)，机器人学习：Petrenko et al. (2020); Makoviichuk and Makoviychuk (2021); Schwarke et al. (2025); NVIDIA et al. (2025); Zakka et al. (2025)，以及导航框架：Macenski et al. (2020); Planning (2012)。尽管这些工作为机器人自主奠定了重要基础，但本节并不聚焦于此。相反，本节将评估并对比面向移动机器人的现有集成、开源自主堆栈，包括空中与地面系统。


Autonomous driving is among the most mature fields in terms of integrated autonomy stacks. The Autoware Universe Autoware Foundation (2021); Kato et al. (2018) is a modular ROS2 autonomous driving stack that involves a highly-integrated perception, planning, and control solution. Apollo Auto (2017) is another flagship open autonomous driving project, targeting autonomy in structured urban streets. A comparative analysis of major autonomous driving stacks is provided in Jung et al. (2025), reflecting the broader community interest in open autonomy frameworks. Off-road autonomy is addressed in NATURE Goodin et al. (2024), which introduces an open-source stack, providing a self-contained pipeline with perception, global and local planning, and control components, with support for both ROS1 and ROS2.
自动驾驶是就集成自主堆栈而言最成熟的领域之一。Autoware Universe Autoware Foundation (2021); Kato et al. (2018) 是一个模块化的 ROS2 自动驾驶自主堆栈，包含高度集成的感知、规划与控制解决方案。Apollo Auto (2017) 是另一个旗舰级的开源自动驾驶项目，面向结构化城市街道上的自主能力。主要自动驾驶堆栈的对比分析见 Jung et al. (2025)，反映了社区对开放自主框架的更广泛关注。非道路自主在 NATURE Goodin et al. (2024) 中得到讨论，该工作提出了一个开源堆栈，提供一套自包含的流水线，包含感知、全局与局部规划以及控制组件，并同时支持 ROS1 与 ROS2。


Beyond autonomous driving, aerial robotics has seen some of the most active development of autonomy stacks. The MRS UAV System Baca et al. (2021) is among the most complete open-source autonomy stacks for multirotor aerial robots. It supports LiDAR-based SLAM state estimation, several control methods including SE(3) formulations and model predictive control, trajectory generation, and selected multi-robot capabilities. From a complementary perspective,
除了自动驾驶之外，无人机/空中机器人领域也看到了自主堆栈的极其活跃发展。MRS UAV System Baca et al. (2021) 是面向多旋翼空中机器人的最完整开源自主堆栈之一。它支持基于 LiDAR 的 SLAM 状态估计，提供多种控制方法（包括 SE(3) 表述与模型预测控制）、轨迹生成，以及部分多机器人能力。从互补的视角，


Table 1. Comparison with existing autonomy stacks.
表 1. 与现有自主堆栈的对比。


<table><tr><td rowspan="3">Stack</td><td rowspan="3">Modular</td><td rowspan="3">Modalities1</td><td rowspan="3">Embodiments2</td><td rowspan="3">Verified3</td><td colspan="10">Autonomy Features ${}^{4}$</td></tr><tr><td colspan="3">Perception</td><td colspan="3">Planning</td><td colspan="4">Safety</td></tr><tr><td>SLAM</td><td>OD</td><td>LQ</td><td>EP</td><td>IP</td><td>TP</td><td>M</td><td>T</td><td>Ct</td><td>LR</td></tr><tr><td>AeroStack</td><td>✓</td><td>C</td><td>A</td><td>SL</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>-</td><td>✘</td><td>✘</td></tr><tr><td>AeroStack2</td><td>✓</td><td>C</td><td>A</td><td>SL</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>-</td><td>✘</td><td>✘</td></tr><tr><td>MRS UAV</td><td>✓</td><td>CLG</td><td>A</td><td>SLF</td><td>E</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>-</td><td>✘</td><td>✓</td></tr><tr><td>Nebula</td><td>✓</td><td>CL</td><td>AG</td><td>SLF</td><td>✓</td><td>十</td><td>✘</td><td>+</td><td>✘</td><td>✘</td><td>+</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>Agilicious</td><td>✓</td><td>C</td><td>A</td><td>SLF</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>-</td><td>✘</td><td>✘</td></tr><tr><td>KR Aut. Flight</td><td>✓</td><td>CL</td><td>A</td><td>SLF</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>一</td><td>✘</td><td>✘</td></tr><tr><td>NATURE</td><td>✘</td><td>L</td><td>G</td><td>SLF</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>AirStack</td><td>✓</td><td>CL</td><td>A</td><td>SL</td><td>✓</td><td>+</td><td>✘</td><td>✓</td><td>✘</td><td>✓</td><td>+</td><td>一</td><td>+</td><td>✘</td></tr><tr><td>Nav2</td><td>✓</td><td>L</td><td>G</td><td>SLF</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td></tr><tr><td>Autoware</td><td>✓</td><td>CRLG</td><td>G</td><td>SLF</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td></tr><tr><td>Apollo</td><td>✓</td><td>CRLG</td><td>G</td><td>SLF</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td></tr><tr><td>Ours (UAstack)</td><td>✓</td><td>CRL</td><td>AG</td><td>SLF</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>
<table><tbody><tr><td rowspan="3">栈</td><td rowspan="3">模块化</td><td rowspan="3">模态1</td><td rowspan="3">具身形态2</td><td rowspan="3">已验证3</td><td colspan="10">自主功能 ${}^{4}$</td></tr><tr><td colspan="3">感知</td><td colspan="3">规划</td><td colspan="4">安全</td></tr><tr><td>SLAM</td><td>OD</td><td>LQ</td><td>EP</td><td>IP</td><td>TP</td><td>M</td><td>T</td><td>Ct</td><td>LR</td></tr><tr><td>AeroStack</td><td>✓</td><td>C</td><td>A</td><td>SL</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>-</td><td>✘</td><td>✘</td></tr><tr><td>AeroStack2</td><td>✓</td><td>C</td><td>A</td><td>SL</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>-</td><td>✘</td><td>✘</td></tr><tr><td>MRS 无人机</td><td>✓</td><td>CLG</td><td>A</td><td>SLF</td><td>E</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>-</td><td>✘</td><td>✓</td></tr><tr><td>Nebula</td><td>✓</td><td>CL</td><td>AG</td><td>SLF</td><td>✓</td><td>十</td><td>✘</td><td>+</td><td>✘</td><td>✘</td><td>+</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>Agilicious</td><td>✓</td><td>C</td><td>A</td><td>SLF</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>-</td><td>✘</td><td>✘</td></tr><tr><td>KR 自动飞行</td><td>✓</td><td>CL</td><td>A</td><td>SLF</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>一</td><td>✘</td><td>✘</td></tr><tr><td>NATURE</td><td>✘</td><td>L</td><td>G</td><td>SLF</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>AirStack</td><td>✓</td><td>CL</td><td>A</td><td>SL</td><td>✓</td><td>+</td><td>✘</td><td>✓</td><td>✘</td><td>✓</td><td>+</td><td>一</td><td>+</td><td>✘</td></tr><tr><td>Nav2</td><td>✓</td><td>L</td><td>G</td><td>SLF</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td></tr><tr><td>Autoware</td><td>✓</td><td>CRLG</td><td>G</td><td>SLF</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td></tr><tr><td>Apollo</td><td>✓</td><td>CRLG</td><td>G</td><td>SLF</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td></tr><tr><td>我们（UAstack）</td><td>✓</td><td>CRL</td><td>AG</td><td>SLF</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></tbody></table>


${}^{1}$ Supported perception modalities,where C: camera,R: radar,L: LiDAR,G: GNSS. The underline denotes modalities which are used by SLAM, and breaks indicate non-multi-modal SLAM usage.
${}^{1}$ 支持的感知模态，其中 C：相机，R：雷达，L：LiDAR，G：GNSS。下划线表示被 SLAM 使用的模态，断开表示未使用多模态 SLAM。


${}^{2}$ Supported embodiments,where A: aerial,G: ground.
${}^{2}$ 支持的机器人形态，其中 A：空中，G：地面。


${}^{3}$ How the autonomy stack was tested,where S: simulation,L: lab,F: field
${}^{3}$ 自主栈的测试方式，其中 S：仿真，L：实验室，F：实地


${}^{4}$ Presence of feature,where: $\checkmark$ : feature exists and open-sourced, $\dagger$ : exists and not open-sourced, $\text{ ✗ }$ : does not exist,E: external package integrated as is, - : not applicable
${}^{4}$ 功能是否存在，其中：$\checkmark$：功能存在且开源，$\dagger$：存在但未开源，$\text{ ✗ }$：不存在，E：按原样集成的外部包，-：不适用


OD: Object Detection LQ: Language Query EP: Exploration Planning
OD：目标检测 LQ：语言查询 EP：探索规划


IP: Inspection Planning TP: Planning to Target M: Map-based safety
IP：检查规划 TP：面向目标规划 M：基于地图的安全


T: Traversability checking Ct: Safety at control layer LR: Last resort safety
T：可通行性检查 Ct：控制层安全 LR：最后手段安全


Aerostack2 Fernandez-Cortizas et al. (2023) is a ROS2- based autonomy framework for aerial robots that emphasizes standardization, modularity, and reusability. It provides replaceable and dynamically configurable components, enables behavior-based mission planning, and supports multi-robot coordination. Unlike more integrated systems, it does not provide a comprehensive perception-planning-control pipeline and does not include some components required for autonomy (e.g., a SLAM solution). Aerostack2 builds upon Aerostack Sanchez-Lopez et al. (2016), which was an early effort to standardize aerial robot frameworks. Going further, Aerostack2 provides better modularity, a wider set of controller/estimator options, a richer behavior tree, and is implemented in ROS 2. AirStack AirLab (2025) targets modular aerial autonomy and provides a research-to-deployment pipeline with tight support for testing in both simulation and reality. Released as an alpha version at the time of writing, it is not yet fully available; instead, only a subset of its modules is open at the time of writing (e.g., exploration planning, visual-inertial odometry). Following a different direction, the unmanned aerial vehicle abstraction layer (UAL) Real et al. (2020) contributes a layered design providing a standard API for aerial robot control and abstracts specific autopilot implementations. Through a namespaced design, it facilitates multi-robot support, but does not provide higher-level autonomy components. Agilicious Foehn et al. (2022) differs from these systems by offering a hardware-software stack for agile vision-based quadrotor flight, with an emphasis on tight integration of the perception-action loop. Compared with systems such as the MRS UAV System and Aerostack, it follows a performance-first philosophy, resulting in a tightly coupled but necessarily less modular solution. The kr_autonomous_flight stack Mohta et al. (2018) also provides a complete autonomy stack for GNSS-denied aerial robot missions, integrating vision-based odometry, LiDAR-based SLAM, global planning for target reaching, trajectory generation, while also supporting downstream tasks such as semantic SLAM. Beyond aerial robotics, Nav2 Macenski et al. (2020) provides a navigation framework for ground robots in predominantly indoor environments. Built around behavior trees, it includes LiDAR-based SLAM, global and local planning, and control, offering a ROS2-compatible navigation stack for ground robots.
Aerostack2 Fernandez-Cortizas et al. (2023) 是一个面向空中机器人的基于 ROS 2 的自主框架，强调标准化、模块化和可复用性。它提供可替换且可动态配置的组件，支持基于行为的任务规划，并支持多机器人协同。不同于更集成的系统，它没有提供完整的感知-规划-控制流水线，也不包含自主所需的一些组件（例如 SLAM 方案）。Aerostack2 建立在 Aerostack Sanchez-Lopez et al. (2016) 之上，后者是早期推动空中机器人框架标准化的尝试。进一步而言，Aerostack2 提供了更好的模块化、更广的控制器/估计器选项、更丰富的行为树，并采用 ROS 2 实现。AirStack AirLab (2025) 面向模块化空中自主，提供研究到部署的流水线，并对仿真与现实环境中的测试提供紧密支持。该系统在写作时以 alpha 版本发布，尚未完全开放；目前仅有部分模块开源（例如探索规划、视觉惯性里程计）。沿着不同方向，无人机抽象层（UAL）Real et al. (2020) 采用分层设计，为空中机器人控制提供标准 API，并对特定自动驾驶仪实现进行抽象。通过命名空间设计，它便于支持多机器人，但不提供更高层的自主组件。Agilicious Foehn et al. (2022) 则不同于这些系统，它提供了用于敏捷视觉四旋翼飞行的软硬件栈，强调感知-动作闭环的紧密集成。与 MRS UAV System 和 Aerostack 等系统相比，它遵循性能优先的理念，因此方案更紧耦合，也必然更不模块化。kr_autonomous_flight 栈 Mohta et al. (2018) 也为 GNSS 受限的空中机器人任务提供了完整自主栈，集成了基于视觉的里程计、基于 LiDAR 的 SLAM、面向目标抵达的全局规划以及轨迹生成，同时还支持语义 SLAM 等下游任务。除空中机器人外，Nav2 Macenski et al. (2020) 为主要处于室内环境中的地面机器人提供导航框架。它围绕行为树构建，包含基于 LiDAR 的 SLAM、全局与局部规划以及控制，为地面机器人提供了兼容 ROS 2 的导航栈。


Targeting a more unified approach to autonomy, Nebula NASA CoSTAR (2022); Agha et al. (2021) is team CoSTAR's architecture for their participation in the DARPA Subterranean Challenge. The public release includes components such as the multi-robot SLAM method Chang et al. (2022), while other elements of the perception-action loop and the complete autonomy stack are not open. Beyond autonomy stacks themselves, recent work has also explored how such systems can support higher-level autonomy. For example, Ravichandran et al. (2025) present progress in Large Language Model (LLM)-enabled autonomy in field settings with the LLM orchestrating autonomous operations over several kilometers across aerial and ground systems. This direction aligns with ongoing efforts on Vision-Language-Action (VLA) models for autonomy Zhang et al. (2026); Serpiva et al. (2025); Xu et al. (2026); Wu et al. (2025), building on core modules such as works in SLAM and control.
为追求更统一的自主方案，Nebula NASA CoSTAR (2022); Agha et al. (2021) 是 CoSTAR 团队为参加 DARPA 地下挑战赛而构建的架构。公开发布的部分包括多机器人 SLAM 方法 Chang et al. (2022)，而感知-动作闭环和完整自主栈的其他元素尚未开放。除了自主栈本身，近期工作也开始探索此类系统如何支持更高层级的自主能力。例如，Ravichandran et al. (2025) 展示了在实地场景中由大语言模型（LLM）驱动的自主系统进展，由 LLM 在空中与地面系统间跨越数公里协调自主操作。这一方向与正在推进的用于自主的视觉-语言-动作（VLA）模型 Zhang et al. (2026); Serpiva et al. (2025); Xu et al. (2026); Wu et al. (2025) 相一致，并建立在 SLAM 和控制等核心模块之上。


Table 1 summarizes how various autonomy stacks compare in terms of functionalities and features. Specifically, we compare across four aspects: (a) supported sensing modalities and multi-layered safety mechanisms; (b) breadth of capabilities, including supported mission objectives; (c) design modularity and support for different robot embodiments; and (d) open-source availability and field evaluation.
表 1 总结了不同自主栈在功能与特性上的对比。具体而言，我们从四个方面进行比较：(a) 支持的感知模态与多层安全机制；(b) 能力广度，包括支持的任务目标；(c) 设计模块化程度及对不同机器人形态的支持；以及 (d) 开源可用性与实地评估。


Compared with existing works, the UAstack presents distinct contributions. First, it represents a functionality-rich autonomy stack experimentally validated on heterogeneous platforms, including multirotors and legged robots. It provides complete implementations of all the core modules (in perception, planning, and navigation) and the overarching architecture for high-performance autonomy. This includes SLAM, VLM-based scene reasoning, motion, and informative path planning, and multi-layered safe navigation and control. Second, the UAstack incorporates design choices aimed at enhanced resilience. Aiming to enable resilient operations in strenuous environments, the UAstack builds upon lessons learned from prior works in field autonomy such as Tranzatto et al. (2022); Ebadi et al. (2023); Agha et al. (2021); Kottege et al. (2024); Dharmad-hikari and Alexis (2025) and combines multi-modal sensor fusion (combining the complementary benefits of LiDAR, vision, and radar in order to penetrate through GNSS-denied and perceptually-degraded environments) with multi-layered safety (such that collision-avoidance is assured through three distinct but synergetic methodological pathways). Targeting advanced behaviors, the UAstack supports advanced mission-level behaviors, offering exploration and coverage in complex, large-scale, geometrically complex environments. Finally, the complete implementation of the UAstack is released open-source with supporting datasets and, where applicable, simulation integration, to accelerate testing, verification, adoption, and extension by the community.
与现有工作相比，UAstack 具有显著不同的贡献。首先，它是一个功能丰富的自主性栈，已在多种异构平台上通过实验验证，包括多旋翼和足式机器人。它提供了感知、规划和导航中所有核心模块的完整实现，以及高性能自主性的整体架构。其中包括 SLAM、基于 VLM 的场景推理、运动与信息路径规划，以及多层安全导航与控制。其次，UAstack 融入了旨在增强韧性的设计选择。为使系统能够在严苛环境中实现韧性运行，UAstack 借鉴了以往现场自主领域工作的经验，如 Tranzatto et al. (2022); Ebadi et al. (2023); Agha et al. (2021); Kottege et al. (2024); Dharmad-hikari and Alexis (2025)，并将多模态传感器融合（结合 LiDAR、视觉和雷达的互补优势，以穿透 GNSS 受限和感知退化环境）与多层安全机制相结合（通过三条彼此独立但相互协同的方法路径确保避免碰撞）。面向高级行为，UAstack 支持高级任务级行为，提供在复杂、大规模且几何结构复杂环境中的探索与覆盖。最后，UAstack 的完整实现已开源，并附带支持数据集，且在适用情况下提供仿真集成，以加速社区的测试、验证、采用和扩展。


## 3 Unified Autonomy
## 3 统一自主


This section presents the architecture and key modules of the unified autonomy stack.
本节介绍统一自主栈的架构及关键模块。


### 3.1 Autonomy Architecture
### 3.1 自主架构


The Unified Autonomy Stack is organized around three core modules - perception, planning, and navigation - following the principles of the "sense-think-act" loop, while targeting generalizability across aerial and ground robot configurations, and resilience in demanding environments. Its overall architecture is presented in Figure 2. The UAstack consumes diverse sensor data and outputs low-level commands to standard controllers available in most modern robotic systems, for example, on PX4-based drones Meier et al. (2015) (or any other MAVLink-compatible autopilot Koubâa et al. (2019)) and standard (linear and angular) velocity controllers on ground platforms. Its key features are as follows:
统一自主栈围绕感知、规划和导航三大核心模块组织，遵循“感知-思考-行动”循环原则，同时面向空中与地面机器人配置的通用性，以及在严苛环境中的鲁棒性。其整体架构如图2所示。UAstack接收多样的传感器数据，并向大多数现代机器人系统中可用的标准控制器输出底层指令，例如基于PX4的无人机 Meier et al. (2015)（或任何其他兼容MAVLink的自动驾驶仪 Koubâa et al. (2019)）以及地面平台上的标准（线速度和角速度）控制器。其主要特性如下：


Generalizability: The UAstack applies with few adjustments to a wide range of robot configurations, offering a consistent user experience for navigation and informative path planning tasks across platforms. Currently out-of-the-box supporting multirotors and other rotorcrafts, alongside several ground systems and especially legged robots, it provides a strong foundation for research in unified embodied AI. In this paper, we present experimental verification with multirotors and quadrupeds, while the associated open-source repository includes examples with additional morphologies such as ground rovers and helicopters.
通用性：UAstack只需少量调整即可适用于广泛的机器人配置，为跨平台的导航与信息化路径规划任务提供一致的用户体验。目前开箱即支持多旋翼及其他旋翼机，以及若干地面系统，尤其是足式机器人，为统一具身智能研究提供了坚实基础。本文中，我们通过多旋翼和四足机器人进行了实验验证，而相关开源仓库还包含地面车和直升机等其他形态的示例。


Multi-modality: The UAstack fuses complementary sensor cues, currently including LiDAR, FMCW radar, vision, and IMU, enabling resilience in perceptually-degraded and GNSS-denied conditions Chung et al. (2023), including settings characterized by self-similar geometries, dark or low-texture scenes, icy regions, and obscurants such as smoke and dust. We emphasize the tight fusion of LiDAR and radar data focusing on their complementary role in numerous perceptually-degraded environments as discussed in the evaluation section.
多模态：UAstack融合互补的传感线索，目前包括LiDAR、FMCW雷达、视觉和IMU，使其在感知退化和GNSS拒止条件下具备鲁棒性 Chung et al. (2023)，包括具有自相似几何特征、昏暗或低纹理场景、冰雪区域，以及烟雾和灰尘等遮蔽物的环境。我们强调LiDAR与雷达数据的紧密融合，重点在于其在众多感知退化环境中的互补作用，详见评估部分。


Multi-layer Safety: The UAstack departs from conventional architectures in which safety is ensured by solutions with a single point-of-failure. Most commonly, modern autonomy solutions rely on a cascade of calculations in which collision-free planning takes place only on an online reconstructed map. In practice, it entails that nontrivial localization or mapping errors (e.g., such as those often encountered in perceptually-degraded settings or when encountering thin obstacles) can lead to collisions. The UAstack combines map-based motion planning with deep learning-driven navigation strategies and safety filters that directly consume online exteroceptive depth measurements and, if necessary, adjust the robot's path to re-assert safety.
多层安全：UAstack不同于传统架构，在传统架构中，安全由具有单点故障的方案来保障。最常见的是，现代自主方案依赖一系列级联计算，其中无碰撞规划仅在在线重建的地图上进行。实践中，这意味着不容忽视的定位或建图误差（例如在感知退化场景中或遇到细小障碍物时常见的误差）可能导致碰撞。UAstack将基于地图的运动规划与深度学习驱动的导航策略和安全过滤器结合起来，这些模块直接消费在线外部感知深度测量，并在必要时调整机器人的路径，以重新确保安全。


Methodological Plurality: The UAstack integrates both "conventional" model-based control, estimation, perception, optimization and planning techniques, as well as deep learning-based methods alongside hybrid techniques. Indicative examples include its factor graph-based multi-modal SLAM and its navigation policies offering options for Deep Reinforcement Learning (DRL)-based and Neural Model Predictive Control (NMPC).
方法多元性：UAstack整合了“传统”的基于模型的控制、估计、感知、优化与规划技术，也结合了深度学习方法以及混合技术。典型例子包括其基于因子图的多模态SLAM，以及其导航策略中提供的基于深度强化学习（DRL）和神经模型预测控制（NMPC）的选项。


Subsequently, we outline the key modules of the UAstack and point to prior works as applicable. Furthermore, we discuss the interfaces considered and how UAstack can be extended to new robot configurations. Importantly, all autonomy modules described hereafter operate subject to the information provided to the stack's Robot Abstraction Layer and the Mission Abstraction Layer.
随后，我们概述UAstack的关键模块，并在适用处指向相关既有工作。此外，我们讨论所考虑的接口，以及UAstack如何扩展到新的机器人配置。重要的是，下文所述的所有自主模块都在栈的机器人抽象层和任务抽象层所提供的信息约束下运行。


Robot Abstraction Layer: This layer defines the robot type (e.g., multirotor, quadruped, etc.) and its key motion parameters,alongside its sensor suite $\mathcal{S}$ including the sensor types (e.g., LiDAR, Radar, cameras, IMU) and their configurations (e.g., fields of view, effective ranges, calibration constants). Based on this layer, the modules of the UAstack determine the robot motion constraints ${\mu }_{R}$ to be applied (e.g.,robot size,potentially applicable traversability constraints, kinematic constraints), the sensor data to be fused, sensor intrinsic and extrinsic calibration parameters, time-synchronization information between the sensors, as well as the command interface to be used (e.g., acceleration commands for multirotors, velocity commands to a quadruped legged robot) as detailed in Section 3.5. It also sets if certain modules of the UAstack are enabled such as if multi-layered safety will be employed and if yes, which submodules of it will be engaged.
机器人抽象层：该层定义机器人类型（例如多旋翼、四足机器人等）及其关键运动参数，并定义其传感器套件 $\mathcal{S}$，包括传感器类型（例如LiDAR、雷达、相机、IMU）及其配置（例如视场、有效量程、标定常数）。基于该层，UAstack的各模块确定要应用的机器人运动约束 ${\mu }_{R}$（例如机器人尺寸、潜在适用的可通行性约束、运动学约束）、要融合的传感器数据、传感器内外参标定参数、传感器间时间同步信息，以及所使用的指令接口（例如多旋翼的加速度指令、四足足式机器人的速度指令），详见第3.5节。它还设置UAstack的某些模块是否启用，例如是否采用多层安全，以及若采用，则启用其中哪些子模块。


Mission Abstraction Layer: This layer sets the task for the robot, including if this relates to reaching a target, exploring an unknown area, inspecting a previously explored region, any combination among those, or any other behavior built beyond the existing capabilities of the UAstack. It accordingly defines the objective of the planning module. It also sets if certain modules are active such as the VLM-based scene reasoning.
任务抽象层：该层为机器人设定任务，包括该任务是否涉及到达目标、探索未知区域、检查先前探索过的区域、上述任意组合，或基于UAstack现有能力之外构建的任何其他行为。相应地，它定义规划模块的目标。它还设置某些模块是否激活，例如基于VLM的场景推理。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_5.jpg?x=156&y=165&w=1339&h=686&r=0"/>



Figure 2. The architecture of the UAstack. The stack involves three core modules, on perception, planning, and navigation that operate in a synergistic fashion. Aiming for operational resilience in diverse GNSS-denied, perceptually-degraded environments the UAstack emphasizes multi-modal sensor fusion merging data from LiDAR, radar, and camera sensing, alongside IMU cues. VLM-based reasoning builds upon the geometric reconstruction and supports object discovery and visual question/answering. The stack's planning layer offers diverse behaviors, with ready-made implementations for target-reach, unknown area exploration and inspection, across robot morphologies. Even though map-based collision avoidance and traversability analysis are provided within the planning module, the stack's navigation module further offers a multi-layered approach to safety involving deep exteroceptive navigation strategies, either through neural model predictive contraction. or reinforcement learning, alongside formal last-resort safety based on control barrier functions. In terms of morphologies, the UAstack currently supports aerial and ground robots, especially rotorcrafts (e.g., multirotors), legged robots and ground rovers Experimental validation has taken place on different multirotor systems and quadruped legged robots, while simulation examples in the released code include additional morphologies such as ground rovers and helicopters.
图 2。UAstack 的架构。该系统由三个核心模块组成：感知、规划与导航，它们以协同方式运行。UAstack 面向在多样的 GNSS 缺失且感知退化的环境中的作业韧性，强调多模态传感器融合，将来自 LiDAR、雷达和摄像头的数据与 IMU 信号一并融合。基于 VLM 的推理建立在几何重建之上，支持目标发现以及视觉问答。规划层提供多样化行为，并为目标到达、未知区域探索与巡检等任务提供现成实现，覆盖不同机器人形态。尽管规划模块内已提供基于地图的避碰与可通行性分析，UAstack 的导航模块进一步采用多层安全策略，包括深度外部感知导航方法：可通过神经模型预测收缩实现，或结合强化学习，同时还提供基于控制屏障函数的形式化“最后手段”安全机制。在机器人形态方面，UAstack 目前支持空中与地面机器人，尤其是旋翼平台（如多旋翼）、腿式机器人与地面探测车。实验验证已在不同的多旋翼系统与四足腿式机器人上开展；而在已发布的代码中，仿真示例还包含更多形态，如地面探测车与直升机。


### 3.2 Perception Module
### 3.2 感知模块


The perception module includes our solution for multi-modal SLAM, alongside integration with a VLM-based reasoning step.
感知模块包括我们针对多模态 SLAM 的方案，并集成了基于 VLM 的推理步骤。


3.2.1 Multi-Modal SLAM The proposed novel multimodal SLAM system (dubbed MIMOSA-X, where 'X' denotes the modalities used) uses a factor graph estimator to fuse LiDAR, radar, camera, and IMU measurements using a windowed smoother Dellaert and GTSAM Contributors (2022) for computational efficiency. This architecture, as shown in Figure 2, builds upon ideas proposed in Khedekar et al. (2022); Nissov et al. (2024b), with improvements drawn from further developments in Khedekar and Alexis (2025); Nissov et al. (2024a) for enhanced LiDAR and radar integration, alongside vision integration. Unlike loosely coupled approaches Khedekar et al. (2022); Khattak et al. (2020); Shan et al. (2020), MIMOSA-X fuses LiDAR registration factors, radar Doppler factors, and preintegrated IMU factors in a tightly-coupled manner to avoid degenerate optimizations returning partially observable results. Vision is further optionally fused through between factors.
3.2.1 多模态 SLAM 所提出的新型多模态 SLAM 系统（称为 MIMOSA-X，其中“X”表示所使用的模态）使用因子图估计器融合 LiDAR、雷达、相机和 IMU 测量，并借助滑窗平滑器 Dellaert and GTSAM Contributors (2022) 以提高计算效率。该架构如图 2 所示，建立在 Khedekar et al. (2022); Nissov et al. (2024b) 提出的思想之上，并从 Khedekar and Alexis (2025); Nissov et al. (2024a) 的进一步发展中汲取改进，以增强 LiDAR 和雷达集成，并结合视觉集成。不同于松耦合方法 Khedekar et al. (2022); Khattak et al. (2020); Shan et al. (2020)，MIMOSA-X 以紧耦合方式融合 LiDAR 配准因子、雷达多普勒因子和预积分 IMU 因子，以避免退化优化返回部分可观测结果。视觉还可通过中间因子进一步选择性融合。


The estimator considers a state space comprised of the position ${\mathbf{p}}_{\mathrm{{WB}}}^{\mathrm{W}}$ ,velocity ${\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{W}}$ ,and attitude ${\mathbf{R}}_{\mathrm{B}}^{\mathrm{W}}$ of the body IMU frame \{B\} with respect to an inertial (map) frame \{W\}. Note that $\{ W\}$ is not perfectly gravity aligned,i.e.,gravity in $\{ W\}$ does not point exactly down. Furthermore, calibration states such as the accelerometer ${\mathbf{b}}_{\mathrm{a}}$ and gyroscope ${\mathbf{b}}_{\mathrm{g}}$ biases and gravity direction ${\mathbf{g}}^{\mathrm{W}}$ in the map frame are also included. The online gravity estimation has been shown to improve performance Nemiroff et al. (2023), as initial uncertainty regarding the platform attitude can result in map-errors which compound over large distances. The state space $\mathbf{x}$ is thus decomposed into local ${\mathbf{x}}_{\mathrm{L}}$ and global ${\mathbf{x}}_{\mathrm{G}}$ states such that
估计器考虑的状态空间由机体 IMU 坐标系 \{B\} 相对于惯性（地图）坐标系 \{W\} 的位置 ${\mathbf{p}}_{\mathrm{{WB}}}^{\mathrm{W}}$ 、速度 ${\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{W}}$ 和姿态 ${\mathbf{R}}_{\mathrm{B}}^{\mathrm{W}}$ 组成。注意 $\{ W\}$ 并未与重力完全对齐，即 $\{ W\}$ 中的重力并不严格指向下方。此外，还包括诸如加速度计 ${\mathbf{b}}_{\mathrm{a}}$ 和陀螺仪 ${\mathbf{b}}_{\mathrm{g}}$ 偏置以及地图坐标系中的重力方向 ${\mathbf{g}}^{\mathrm{W}}$ 等标定状态。在线重力估计已被证明可提升性能 Nemiroff et al. (2023)，因为平台姿态的不确定性会导致地图误差，并在长距离上不断累积。因此，状态空间 $\mathbf{x}$ 被分解为局部 ${\mathbf{x}}_{\mathrm{L}}$ 和全局 ${\mathbf{x}}_{\mathrm{G}}$ 状态，即


$$
\mathbf{x} = \left( {\underset{{\mathbf{x}}_{\mathrm{L}}}{\underbrace{{\mathbf{p}}_{\mathrm{{WB}}}^{\mathrm{W}}\;{\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{W}}\;{\mathbf{R}}_{\mathrm{B}}^{\mathrm{W}}\;{\mathbf{b}}_{\mathrm{a}}\;{\mathbf{b}}_{\mathrm{g}}}}\;\underset{{\mathbf{x}}_{\mathrm{G}}}{\underbrace{{\mathbf{g}}^{\mathrm{W}}}}}\right) . \tag{1}
$$



By concatenating the states from times ${t}_{k - l}$ to ${t}_{k}$ ,the windowed set of states ${\mathcal{X}}_{k - l : k}$ is defined as
通过将时刻 ${t}_{k - l}$ 到 ${t}_{k}$ 的状态串联起来，窗口内状态集合 ${\mathcal{X}}_{k - l : k}$ 定义为


$$
{\mathcal{X}}_{k - l : k} = \left\{  \begin{array}{lllll} {\mathbf{x}}_{\mathrm{L},k - l} & {\mathbf{x}}_{\mathrm{L},k - l + 1} & \ldots & {\mathbf{x}}_{\mathrm{L},k} & {\mathbf{x}}_{\mathrm{G}} \end{array}\right\}  . \tag{2}
$$



The states over this temporal window of size $l + 1$ are thus estimated by the iSAM2 Kaess et al. (2011) nonlinear optimizer,where the optimal estimate ${\mathcal{X}}_{k - l : k}^{ * }$ is found by minimizing the covariance-weighted $\left( {\sum }_{ \star  }\right)$ sum of the residuals ${\mathbf{e}}_{ \star  }$ derived from the IMU,LiDAR,radar,and vision sensor measurements included in the temporal window, denoted by $\mathcal{I},\mathcal{L},\mathcal{R}$ ,and $\mathcal{V}$ ,respectively. This minimization problem can thus be written as follows
因此，这一时间窗口大小为 $l + 1$ 内的状态由 iSAM2 Kaess et al. (2011) 非线性优化器估计，其中通过最小化由时间窗口内包含的 IMU、LiDAR、雷达和视觉传感器测量所导出的残差 ${\mathbf{e}}_{ \star  }$ 的协方差加权 $\left( {\sum }_{ \star  }\right)$ 和来求得最优估计 ${\mathcal{X}}_{k - l : k}^{ * }$，这些测量分别记为 $\mathcal{I},\mathcal{L},\mathcal{R}$ 和 $\mathcal{V}$。因此，该最小化问题可写为如下形式


$$
{\mathcal{X}}_{k - l : k}^{ * } = \underset{{\mathcal{X}}_{k - l : k}}{\arg \min }\left\lbrack  {{\begin{Vmatrix}{\mathbf{e}}_{0}\end{Vmatrix}}_{{\sum }_{0}}^{2} + \mathop{\sum }\limits_{{i \in  {\mathcal{F}}_{k - l : k}^{\mathcal{I}}}}{\begin{Vmatrix}{\mathbf{e}}_{{\mathcal{I}}_{i}}\end{Vmatrix}}_{{\sum }_{{\mathcal{I}}_{i}}}^{2}}\right.
$$



$$
+ \mathop{\sum }\limits_{{i \in  {\mathcal{F}}_{k - l : k}^{\mathcal{L}}}}{\begin{Vmatrix}{\mathbf{e}}_{{\mathcal{L}}_{i}}\end{Vmatrix}}_{{\sum }_{\mathcal{L}}}^{2} + \mathop{\sum }\limits_{{i \in  {\mathcal{F}}_{k - l : k}^{\mathcal{R}}}}{\begin{Vmatrix}{\mathbf{e}}_{{\mathcal{R}}_{i}}\end{Vmatrix}}_{{\sum }_{\mathcal{R}}}^{2} \tag{3}
$$



$$
\left. {+\mathop{\sum }\limits_{{i \in  {\mathcal{F}}_{k - l : k}^{\mathcal{V}}}}{\begin{Vmatrix}{\mathbf{e}}_{{\mathcal{V}}_{i}}\end{Vmatrix}}_{{\sum }_{{\mathcal{V}}_{i}}}^{2}}\right\rbrack  \text{ , }
$$



for the marginalization prior ${\mathbf{e}}_{0},{\sum }_{0}$ ,and the IMU,LiDAR, radar,and vision factors ${\mathcal{F}}_{k - l : k}^{ \star  }$ in the window time frame, denoted with the same notation as the residuals and covariance matrices. For each IMU measurement, a prediction is made from the current graph state, resulting in a high-rate odometry output. Upon receiving a new exteroceptive sensor measurement, the graph is updated with the relevant factors, and the optimal state estimate is calculated, alongside maps updated if the measurement was a LiDAR measurement.
其中 ${\mathbf{e}}_{0},{\sum }_{0}$ 为边缘化先验，${\mathcal{F}}_{k - l : k}^{ \star  }$ 为窗口时间框架内的 IMU、LiDAR、雷达和视觉因子，记号与残差和协方差矩阵相同。对于每个 IMU 测量，都会根据当前图状态进行预测，从而输出高频里程计。当接收到新的外部感知传感器测量时，图会更新相关因子，并计算最优状态估计；若该测量为 LiDAR 测量，则还会同步更新地图。


It is not required that all sensors operate at the same frequency, nor that their measurements arrive chronologically by their timestamps. The method does, however, assume that the timestamps of the input measurements are accurate, i.e., some effort from the implementer has been taken to ensure the accuracy of the timestamping of the sensor data in the form of hardware or software-based time synchronization, such that all timestamps are on a common time axis. A representative factor graph constructed by the multi-modal estimator is shown in Figure 3. We first give an overarching view of the operation of the system and later detail how each of the sensor measurements is used.
并不要求所有传感器以相同频率工作，也不要求它们的测量按时间戳顺序到达。不过，该方法假定输入测量的时间戳是准确的，即实现者已通过硬件或软件的时间同步对传感器数据的时间戳精度作出了一定努力，使所有时间戳位于同一时间轴上。多模态估计器构建的一个代表性因子图如图3所示。我们首先概述系统的运行过程，随后详细说明如何使用各传感器测量。


Initialization The method assumes the system is stationary at startup. During this period, IMU measurements are accumulated over a $1\mathrm{\;s}$ window and averaged,yielding the mean accelerometer reading ${\mathbf{\mu }}_{\mathrm{a}}$ and mean gyroscope reading ${\mathbf{\mu }}_{\mathrm{g}}$ . Since the system is at rest,the gyroscope mean is a direct estimate of the gyroscope bias,so we set ${\mathbf{b}}_{\mathrm{g},0} = {\mathbf{\mu }}_{\mathrm{g}}$ . The accelerometer mean satisfies ${\mathbf{\mu }}_{\mathrm{a}} = {\mathbf{b}}_{\mathrm{a},0} + {\left( {\mathbf{R}}_{\mathrm{B},0}^{\mathrm{W}}\right) }^{\top }{\mathbf{g}}_{\mathrm{W}}$ ,under the assumption that ${\mathbf{b}}_{\mathrm{a},0}$ is constant over the initialization window and that measurement noise is negligible. Because the accelerometer bias is not yet known, we approximate ${\mathbf{R}}_{\mathrm{B},0}^{\mathrm{W}}$ by aligning ${\mathbf{\mu }}_{\mathrm{a}}$ with the gravity direction ${\left\lbrack  \begin{array}{lll} 0 & 0 & 1 \end{array}\right\rbrack  }^{\top }$ , which determines the roll and pitch components. The yaw component is unobservable from accelerometry alone and is set to zero. This approximation introduces an attitude error proportional to the magnitude of ${\mathbf{b}}_{\mathrm{a},0}$ ; for instance,a bias of $1\mathrm{\;m}/{\mathrm{s}}^{2}$ yields an initial attitude error of approximately ${5}^{ \circ  }$ ,following Farrell (2008). The gravity direction ${\mathbf{g}}_{\mathrm{W}}$ is estimated, instead of being fixed, to account for this error. The initial position ${\mathbf{p}}_{\mathrm{{WB}},0}^{\mathrm{W}}$ and velocity ${\mathbf{v}}_{\mathrm{{WB}},0}^{\mathrm{W}}$ are both set to zero.
初始化 该方法假定系统在启动时处于静止状态。在此期间，IMU测量会在一个$1\mathrm{\;s}$窗口内累积并取平均，得到平均加速度计读数${\mathbf{\mu }}_{\mathrm{a}}$和平均陀螺仪读数${\mathbf{\mu }}_{\mathrm{g}}$。由于系统处于静止状态，陀螺仪均值可直接作为陀螺仪偏置的估计，因此我们设${\mathbf{b}}_{\mathrm{g},0} = {\mathbf{\mu }}_{\mathrm{g}}$。在${\mathbf{b}}_{\mathrm{a},0}$在初始化窗口内保持恒定且测量噪声可忽略的假设下，加速度计均值满足${\mathbf{\mu }}_{\mathrm{a}} = {\mathbf{b}}_{\mathrm{a},0} + {\left( {\mathbf{R}}_{\mathrm{B},0}^{\mathrm{W}}\right) }^{\top }{\mathbf{g}}_{\mathrm{W}}$。由于加速度计偏置尚未知晓，我们通过将${\mathbf{\mu }}_{\mathrm{a}}$与重力方向${\left\lbrack  \begin{array}{lll} 0 & 0 & 1 \end{array}\right\rbrack  }^{\top }$对齐来近似${\mathbf{R}}_{\mathrm{B},0}^{\mathrm{W}}$，从而确定横滚和俯仰分量。仅凭加速度计无法观测偏航分量，因此将其设为零。该近似会引入与${\mathbf{b}}_{\mathrm{a},0}$幅值成正比的姿态误差；例如，按照Farrell（2008），$1\mathrm{\;m}/{\mathrm{s}}^{2}$的偏置会导致约${5}^{ \circ  }$的初始姿态误差。为考虑这一误差，重力方向${\mathbf{g}}_{\mathrm{W}}$被估计而非固定。初始位置${\mathbf{p}}_{\mathrm{{WB}},0}^{\mathrm{W}}$和速度${\mathbf{v}}_{\mathrm{{WB}},0}^{\mathrm{W}}$均设为零。


Initialization is triggered upon arrival of the first exteroceptive sensor measurement, provided the IMU buffer spans at least 1 s. In the case of LiDAR, this step additionally initializes the global map, as described in the following section.
当第一条外部感知传感器测量到达时即触发初始化，前提是IMU缓冲区至少覆盖1 s。对于LiDAR，此步骤还会初始化全局地图，如下一节所述。


Propagation Upon receiving a new measurement from the IMU, the measurement is added to a buffer for future use in the main thread. In a separate thread, this measurement is used to propagate the latest state in the graph, which is then published to provide high-rate odometry for feedback control.
传播 接收到一条新的IMU测量后，该测量会被加入缓冲区，以供主线程后续使用。在单独的线程中，该测量用于传播图中的最新状态，随后发布以提供用于反馈控制的高频里程计。


Upon receiving a new measurement from an exteroceptive sensor, typically, the method (a) creates a new state at the timestamp derived from the measurement, (b) connects the new state with the remaining graph with a preintegrated IMU factor derived from the IMU buffer, (c) adds a factor derived from the measurement, (d) optimizes the graph, and (e) publishes the new state. However, depending upon the actual sensor stream and system, the method may deviate slightly from this. We now detail these deviations. If the timestamp of the incoming measurement is older than the lag window, it is discarded prioritizing low-latency data. If the timestamp of the new measurement is very close (i.e., there are no IMU measurements in between) to the timestamp of any of the states in the window, then we treat the measurement as having the same timestamp as that state, i.e., we do not create a new state and use the matched state for adding a measurement-derived factor. This has the added advantage that late measurements arriving to the graph out-of-order can seamlessly be integrated, assuming they are within the smoother window. If the timestamp is older than the newest state in the window (i.e., it arrived with high latency), then we identify the correct location that it should have been added as well as the preintegrated IMU factor that currently connects the states straddling this timestamp. The preintegrated IMU factor is then replaced by a new state at the new timestamp, along with the measurement-derived factor and two new preintegrated IMU factors.
在从外部感知传感器接收到一条新测量时，通常该方法会：a）在由该测量得到的时间戳处创建一个新状态；b）通过从 IMU 缓冲区获得的预积分 IMU 因子，将该新状态与图中其余部分连接起来；c）添加由该测量得到的因子；d）优化该图；以及 e）发布该新状态。不过，取决于实际传感器流与系统，该方法可能会有些偏离。下面我们详细说明这些偏离。若传入测量的时间戳早于滞后窗口，则优先考虑低延迟数据而将其丢弃。若新测量的时间戳与窗口内任一状态的时间戳非常接近（即两者之间没有 IMU 测量），则我们将该测量视为与该状态具有相同时间戳——也就是不创建新状态，而是使用匹配状态来添加由测量得到的因子。这样做的额外好处是：当较晚到达的测量在图中乱序插入时，也能在更平滑的窗口假设下无缝集成。若该时间戳早于窗口内最新状态（即具有较高延迟），则我们同时确定它应当被加入的位置，以及当前连接着该时间戳两侧状态的预积分 IMU 因子。随后，用该新时间戳处的一个新状态替换预积分 IMU 因子，并加入由测量得到的因子，同时再引入两个新的预积分 IMU 因子。


The next paragraphs detail the specifics on the handling of each sensor modality. Note, some factor residuals are constructed on a per-point basis, e.g., in the case of the LiDAR and radar factors. For computational savings, these are implemented as a single Hessian factor by summing individual contributions. For brevity, the factor residuals will be defined on a per-point basis, and the summation implied.
接下来的段落将详细说明如何处理每种传感器模态的具体细节。注意，有些因子残差是按点构造的，例如在 LiDAR 和雷达的情况下。为了节省计算开销，这些实现为通过求和各个单独贡献得到的单个 Hessian 因子。为简洁起见，本文将把因子残差定义为按点形式，并默认存在上述求和过程。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_6.jpg?x=847&y=1518&w=655&h=489&r=0"/>



Figure 3. Representative exemplary factor graph constructed by the multi-modal estimator.
图 3：由多模态估计器构建的代表性示例因子图。


Inertial Measurement Unit IMU measurements are stored in a buffer on arrival, for easy use upon receiving measurements from one of the aiding sensors. At that point, the corresponding exteroceptive factors are created and connected to the graph by an IMU preintegration factor, following Forster et al. (2017), with the following residuals that preintegrate the measurements between times $i$ and $j$
惯性测量单元 IMU 测量在到达时被存入缓冲区，以便在收到来自某个辅助传感器的测量时方便使用。此时，会创建对应的外部感知因子，并通过 IMU 预积分因子将其连接到图中，遵循 Forster et al.（2017）。该预积分的残差为：对 $i$ 和 $j$ 之间的测量进行预积分


$$
{\mathbf{e}}_{\mathcal{I}} = {\left\lbrack  \begin{array}{lll} {\mathbf{e}}_{\mathcal{I},\mathbf{R}}^{\top } & {\mathbf{e}}_{\mathcal{I},\mathbf{p}}^{\top } & {\mathbf{e}}_{\mathcal{I},\mathbf{v}}^{\top } \end{array}\right\rbrack  }^{\top }, \tag{4}
$$



including attitude ${\mathbf{e}}_{\mathcal{I},\mathbf{R}}$ ,position ${\mathbf{e}}_{\mathcal{I},\mathbf{p}}$ ,and velocity ${\mathbf{e}}_{\mathcal{I},\mathbf{v}}$ error terms. The reader is referred to Forster et al. (2017); Dellaert and GTSAM Contributors (2022) for a detailed description of each term.
其中包含姿态 ${\mathbf{e}}_{\mathcal{I},\mathbf{R}}$、位置 ${\mathbf{e}}_{\mathcal{I},\mathbf{p}}$ 和速度 ${\mathbf{e}}_{\mathcal{I},\mathbf{v}}$ 的误差项。读者可参考 Forster et al.（2017）以及 Dellaert 和 GTSAM Contributors（2022）获取对每一项的详细说明。


LiDAR The point cloud measurement obtained from a LiDAR typically contains several thousand points that were sampled at various instants between $\left\lbrack  {t,t + {t}_{s}}\right\rbrack$ where $t$ is the timestamp of the measurement and ${t}_{s}$ is the duration of the sweep. Since the LiDAR is likely to have moved during this time, it is necessary to deskew the point cloud to account for this motion. As the IMU measurements for this duration are available, we use them to propagate the latest state in the graph up to $t + {t}_{s}$ ,storing the intermediate pose ${\widetilde{\mathbf{T}}}_{\mathrm{B},\mathrm{t} + {\mathrm{t}}_{\mathrm{k}}}^{\mathrm{W}}$ for every unique timestamp $t + {t}_{k}$ in the point cloud. Using these intermediate poses,we iterate over the points ${\mathbf{r}}^{\mathrm{L},\mathrm{t} + {\mathrm{t}}_{\mathrm{k}}}$ at each unique timestamp and transform them to the body frame at the timestamp of the last point $t + {t}_{s}$ as given by
LiDAR 由 LiDAR 获得的点云测量通常包含数千个点，这些点是在 $\left\lbrack  {t,t + {t}_{s}}\right\rbrack$ 与 $t$ 之间的不同时间采样的，其中 $t$ 是测量时间戳，而 ${t}_{s}$ 是扫描持续时间。由于在这段时间里 LiDAR 很可能发生了运动，因此需要对点云进行去畸变以补偿该运动。由于该时段的 IMU 测量是可用的，我们利用它们将图中的最新状态传播到 $t + {t}_{s}$，并为点云中每个唯一时间戳 $t + {t}_{k}$ 存储中间位姿 ${\widetilde{\mathbf{T}}}_{\mathrm{B},\mathrm{t} + {\mathrm{t}}_{\mathrm{k}}}^{\mathrm{W}}$。随后，我们对每个唯一时间戳 ${\mathbf{r}}^{\mathrm{L},\mathrm{t} + {\mathrm{t}}_{\mathrm{k}}}$ 的点进行遍历，并将其变换到最后一个点 $t + {t}_{s}$ 时刻的机体系下，变换由下式给出


$$
{\mathbf{r}}^{\mathrm{B},\mathrm{t} + {\mathrm{t}}_{\mathrm{s}}} = {\widetilde{\mathbf{T}}}_{\mathrm{W}}^{\mathrm{B},\mathrm{t} + {\mathrm{t}}_{\mathrm{s}}}{\widetilde{\mathbf{T}}}_{\mathrm{B},\mathrm{t} + {\mathrm{t}}_{\mathrm{k}}}^{\mathrm{W}}{\mathbf{T}}_{\mathrm{L}}^{\mathrm{B}} \circ  {\mathbf{r}}^{\mathrm{L},\mathrm{t} + {\mathrm{t}}_{\mathrm{k}}}, \tag{5}
$$



where $\circ$ denotes the homogeneous transformation action on a vector in ${\mathbb{R}}^{3}$ .
其中 $\circ$ 表示对 ${\mathbb{R}}^{3}$ 中向量执行的齐次变换操作。


For further processing, assuming that the IMU propagation was correct, the point cloud is considered to have been sampled instantaneously at $t + {t}_{s}$ . Afterwards,the point cloud is downsampled, for computational efficiency, first by removing three out of four points, and second by organizing the point cloud into a voxel grid and subsampling, ensuring a maximum of ${n}_{p}$ points per voxel and a minimum distance of ${\eta }_{p}$ between any two points in a voxel. Afterwards,the correspondences are found by relating points in the current cloud with planes fit in the map; these correspondences are added to the graph in the form of point-to-plane residuals. This per-point residual ${\epsilon }_{\mathcal{L}}$ is calculated as follows
为进一步处理，在假设 IMU 传播正确的前提下，点云被认为在 $t + {t}_{s}$ 时刻被瞬时采样。之后，为了计算效率，点云首先通过移除四个点中的三个进行下采样；其次，将点云组织成体素网格并进行子采样，确保每个体素内最多有 ${n}_{p}$ 个点，且同一体素内任意两点之间的最小距离为 ${\eta }_{p}$。随后，通过将当前点云中的点与地图中拟合的平面对应来找到对应关系；这些对应关系以点到平面的残差形式加入图中。这个逐点残差 ${\epsilon }_{\mathcal{L}}$ 的计算如下


$$
{\epsilon }_{\mathcal{L}} = {\mathbf{n}}^{\mathrm{W}} \cdot  \left( {{\mathbf{R}}_{\mathrm{B}}^{\mathrm{W}}{\widetilde{\mathbf{r}}}_{\mathcal{L}}^{\mathrm{B}} + {\mathbf{p}}_{\mathrm{{WB}}}^{\mathrm{W}} - {\mathbf{r}}_{0}^{\mathrm{W}}}\right) , \tag{6}
$$



for a plane defined by the normal ${\mathbf{n}}^{\mathtt{W}}$ and point ${\mathbf{r}}_{0}^{\mathtt{W}}$ and the corresponding transformed point ${\widetilde{\mathbf{r}}}_{\mathcal{L}}^{\mathrm{B}}$ from the downsampled LiDAR point cloud. The per-point residuals are whitened using the point noise covariance ${\sigma }_{{\epsilon }_{\mathcal{L}}}^{2}$ and assembled into a single dense hessian factor. For outlier rejection, these residuals are augmented with Huber M-estimators Huber (1964). Post-optimization, the pose is compared with previous key frames, and if a significant difference in position or attitude is detected, a new key frame is created, and the current point cloud is added to maintain a monolithic map.
对于由法向 ${\mathbf{n}}^{\mathtt{W}}$ 和点 ${\mathbf{r}}_{0}^{\mathtt{W}}$ 定义的平面，以及来自下采样 LiDAR 点云的对应变换点 ${\widetilde{\mathbf{r}}}_{\mathcal{L}}^{\mathrm{B}}$。使用点噪声协方差 ${\sigma }_{{\epsilon }_{\mathcal{L}}}^{2}$ 对逐点残差进行白化，并汇总为一个单一的稠密 Hessian 因子。为剔除离群值，这些残差会引入 Huber M-估计器 Huber (1964)。优化完成后，将位姿与先前的关键帧进行比较；如果检测到位置或姿态存在显著差异，就创建一个新的关键帧，并将当前点云加入以维持一个单一的整体地图。


Radar In the context of this method, FMCW radars are assumed to return point cloud measurements, where each point is defined by its 3D position ${\mathbf{r}}_{\mathcal{R}}$ and radial speed ${v}_{r}$ . Unlike the estimator proposed in Nissov et al. (2024b), here the RANdom SAmple Consensus (RANSAC) least-squares calculation of linear velocity from the radar point cloud is omitted. Instead, the individual points from the radar point cloud are integrated into the graph. By directly integrating the radial speed measurements, we avoid the potential limitations associated with first estimating linear velocity independently. Namely, these are the minimum number and diversity of points required for fully resolving the 3 axes of linear velocity. Even with sufficient points, low point cloud sizes can still result in poor estimation of either the velocity or the covariance matrix, leading to degraded performance.
在该方法的语境下，假设 FMCW 雷达返回点云测量，其中每个点由其 3D 位置 ${\mathbf{r}}_{\mathcal{R}}$ 和径向速度 ${v}_{r}$ 定义。与 Nissov 等人 (2024b) 提出的估计器不同，这里省略了基于雷达点云使用 RANdom SAmple Consensus（RANSAC）进行线速度的最小二乘计算。相反，将雷达点云中的各个点直接集成到图中。通过直接整合径向速度测量，我们避免了先分别估计线速度所可能带来的潜在限制。也就是说，为了能够完全解析线速度的三个坐标轴所需的最少点数与点的多样性。即便点数足够，较小的点云规模仍可能导致对速度或协方差矩阵的估计较差，从而使性能下降。


Thus,the per-point residual ${\epsilon }_{\mathcal{R}}$ for the radar Doppler factor is
因此，雷达多普勒因子的逐点残差 ${\epsilon }_{\mathcal{R}}$ 为


$$
{\epsilon }_{\mathcal{R}} =  - {\widehat{\mathbf{v}}}_{\mathrm{{WR}}}^{\mathrm{R}} \cdot  \frac{{\widetilde{\mathbf{r}}}_{\mathcal{R}}}{\begin{Vmatrix}{\widetilde{\mathbf{r}}}_{\mathcal{R}}\end{Vmatrix}} - {\widetilde{v}}_{r}, \tag{7}
$$



where ${\widehat{\mathbf{v}}}_{\mathrm{{WR}}}^{\mathrm{R}}$ is the radar-frame velocity estimate and $\widetilde{\mathbf{r}},{\widetilde{v}}_{r}$ are the radar point position and radial speed measurements. The radar-frame velocity is composed from the state estimates as
其中，${\widehat{\mathbf{v}}}_{\mathrm{{WR}}}^{\mathrm{R}}$ 是雷达坐标系下的速度估计，$\widetilde{\mathbf{r}},{\widetilde{v}}_{r}$ 是雷达点的位置以及径向速度测量。雷达坐标系下的速度由状态估计构成为


$$
{\widehat{\mathbf{v}}}_{\mathrm{{WR}}}^{\mathrm{R}} = {\mathbf{R}}_{\mathrm{B}}^{\mathrm{R}}\left( {{\left( {\widehat{\mathbf{R}}}_{\mathrm{B}}^{\mathrm{W}}\right) }^{\top }{\widehat{\mathbf{v}}}_{\mathrm{{WB}}}^{\mathrm{W}} + \left( {{\overline{\mathbf{\omega }}}_{\mathrm{{WB}}}^{\mathrm{B}} \times  {\mathbf{p}}_{\mathrm{{BR}}}^{\mathrm{B}}}\right) }\right) , \tag{8}
$$



assuming the extrinsic translation ${\mathbf{p}}_{\mathrm{{BR}}}^{\mathrm{B}}$ and rotation ${\mathbf{R}}_{\mathrm{B}}^{\mathrm{R}}$ between $\{ B\}$ and $\{ R\}$ is known a priori,and that the angular rate during a given radar chirp period ${\overline{\omega }}_{\mathrm{{WB}}}^{\mathrm{B}}$ can be accurately estimated by averaging the IMU gyroscope measurements. As the radar sensor is known to generate spurious points Harlow et al. (2024), the residual is augmented with a Cauchy M-estimator for outlier rejection. This has the added benefit of improved resilience against dynamic objects by suppressing the influence of such outliers. The choice of Cauchy is motivated by the desire for an M-estimator that more rapidly nullifies the impact of significantly large outliers.
假设 $\{ B\}$ 与 $\{ R\}$ 之间的外参平移 ${\mathbf{p}}_{\mathrm{{BR}}}^{\mathrm{B}}$ 和旋转 ${\mathbf{R}}_{\mathrm{B}}^{\mathrm{R}}$ 已先验已知，并且可通过对 IMU 陀螺仪测量值取平均，准确估计在给定雷达啁啾周期 ${\overline{\omega }}_{\mathrm{{WB}}}^{\mathrm{B}}$ 期间的角速率。由于已知雷达传感器会生成伪点（Harlow 等，2024），因此将残差引入带 Cauchy M-估计器的机制以进行离群值剔除。这样做还能通过抑制这类离群值的影响来提高对动态目标的鲁棒性。选择 Cauchy 的动机在于希望这种 M-估计器能更快速地抵消显著较大的离群值所带来的影响。


Vision Vision factors are added in a loosely-coupled manner, taking advantage of the wealth of capable estimators that exist in the vision community. Specifically, a visual-inertial estimator based on Bloesch et al. (2017) processes the camera and IMU measurements, creating odometry estimates as a result. The pose estimates ${\widetilde{\mathbf{T}}}_{\mathrm{B}}^{\mathrm{W}}$ from this external method are stored in a buffer. This information is incorporated, if it passes a D-Optimality pose quality check Carrillo et al. (2012), into the factor graph with a relative transform factor, which compares the relative transform between pose estimates ${\widehat{\mathbf{T}}}_{\mathrm{B}}^{\mathrm{W}}$ and the aforementioned measurements across the same time interval. Assuming vision pose measurements are available for times ${t}_{i}$ and ${t}_{j}$ ,the factor residual can be calculated as
以松耦合的方式加入视觉因子，从而利用视觉领域中现有的大量高能力估计器。具体而言，基于 Bloesch et al.（2017）的视觉-惯性估计器处理相机与 IMU 测量，进而得到里程计估计。由该外部方法得到的位姿估计 ${\widetilde{\mathbf{T}}}_{\mathrm{B}}^{\mathrm{W}}$ 会被存入缓冲区。若其通过 D-Optimality 的位姿质量检验 Carrillo et al.（2012），则将其以相对变换因子的形式并入因子图。该因子会在同一时间区间内，对位姿估计 ${\widehat{\mathbf{T}}}_{\mathrm{B}}^{\mathrm{W}}$ 与上述测量之间的相对变换进行比较。假设视觉位姿测量在时刻 ${t}_{i}$ 与 ${t}_{j}$ 可用，则可计算因子残差为


$$
{\mathbf{e}}_{\mathcal{V}} = \operatorname{Log}\left( {{\left( {\left( {\widetilde{\mathbf{T}}}_{\mathrm{B},i}^{\mathrm{W}}\right) }^{-1}{\widetilde{\mathbf{T}}}_{\mathrm{B},j}^{\mathrm{W}}\right) }^{-1}\left( {{\left( {\widehat{\mathbf{T}}}_{\mathrm{B},i}^{\mathrm{W}}\right) }^{-1}{\widehat{\mathbf{T}}}_{\mathrm{B},j}^{\mathrm{W}}}\right) }\right) , \tag{9}
$$



where Log denotes the logarithmic map from the ${SE}\left( 3\right)$ manifold to its Lie algebra $\mathfrak{{se}}\left( 3\right)$ . This approach draws inspiration from Khattak et al. (2020); Khedekar et al. (2022).
其中，Log 表示从 ${SE}\left( 3\right)$ 流形到其李代数 $\mathfrak{{se}}\left( 3\right)$ 的对数映射。该方法受 Khattak et al.（2020）和 Khedekar et al.（2022）启发。


3.2.2 Vision-Language Reasoning Our semantic reasoning system integrates two complementary vision-language model (VLM) capabilities: (i) open-vocabulary object perception with semantic 3D mapping, and (ii) binary visual question-answering (Yes/No) for high-level scene reasoning. Together, these capabilities collectively enable semantic scene understanding and contextual judgment and decision making from online visual data. The VLM-based functionality is illustrated in Figure 4. Even though our implementation employs YOLOe Wang et al. (2025) and GPT-5, the proposed semantic reasoning system is designed to be compatible with other open-source and proprietary VLMs that support open-vocabulary object detection and visual question-answering tasks.
3.2.2 视觉-语言推理 我们的语义推理系统整合了两种互补的视觉-语言模型（VLM）能力：（i）基于语义三维映射的开放词汇目标感知；以及（ii）用于高层场景推理的二值视觉问答（是/否）。这两种能力共同使得系统能够从在线视觉数据中实现语义场景理解、语境判断与决策。图 4 展示了基于 VLM 的功能。尽管我们的实现采用 YOLOe Wang et al.（2025）和 GPT-5，但所提出的语义推理系统旨在与其他支持开放词汇目标检测与视觉问答任务的开源与专有 VLM 兼容。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_8.jpg?x=133&y=503&w=685&h=384&r=0"/>



Figure 4. Illustration of the current VLM-based functionality of the UAstack for semantic scene mapping and visual Q&A.
图 4. UAstack 用于语义场景建图与视觉问答的当前基于 VLM 功能示意。


Open-Vocabulary Object Detection and Semantic 3D Mapping 3D object detection is formulated as a semantic mapping problem. Objects are detected on the camera image using an open-vocabulary detector (YOLOe) or a VLM-based detector (GPT-5) initialized with a set of labels. These models produce labeled 2D bounding boxes and associated detection confidences. In parallel, a 3D voxel grid is maintained using LiDAR measurements and pose estimates from our SLAM solution.
开放词汇目标检测与语义三维映射 三维目标检测被表述为语义映射问题。使用开放词汇检测器（YOLOe）或基于 VLM 的检测器（GPT-5），并以一组标签进行初始化，可在相机图像上检测目标。这些模型输出带标签的二维边界框及对应的检测置信度。与此同时，使用来自 LiDAR 的测量以及我们 SLAM 解算得到的位姿估计，维护一个三维体素网格。


To integrate semantic detections into the 3D representation, voxel grid points are projected into the camera frame using the camera extrinsics and the current odometry estimate. For each 2D detection, the subset of projected points that fall within the corresponding bounding box is extracted and clustered to remove outliers. The resulting points are used to update the voxels' semantic values via Bayesian fusion, which uses the object detector's confidences.
为将语义检测融入三维表示，将体素网格点利用相机外参与当前里程计估计投影到相机坐标系中。对每个二维检测，提取落在对应边界框内的投影点子集，并进行聚类以剔除离群点。所得点通过贝叶斯融合更新体素的语义值，其中融合使用目标检测器的置信度。


Finally, Euclidean clustering is performed for each semantic class present in the voxel grid to extract 3D object instances. These objects are represented as 3D bounding boxes, enabling semantic and spatial reasoning.
最后，对体素网格中每个存在的语义类别执行欧氏聚类，以提取三维目标实例。这些目标以三维边界框形式表示，从而实现语义与空间推理。


Binary Visual Question-Answering For high-level semantic assessment, a VLM (GPT-5) processes the front-camera image together with a binary "Yes/No" question. These visual question-answering tasks typically focus on safety- or navigation-related properties of the scene (e.g., "is an object blocking a door?"), which can be challenging to infer from geometric information alone. The model produces a binary answer, alongside its response confidence (ranging from 0 to 1) and a brief explanation of its reasoning.
二值视觉问答 为进行高层语义评估，一个 VLM（GPT-5）将前置相机图像与一个二值“是/否”问题一起进行处理。这类视觉问答任务通常关注场景的安全性或导航相关性质（例如：“有物体会阻挡门吗？”），仅凭几何信息往往难以推断。模型输出二值答案，同时给出其响应置信度（范围为 0 到 1）以及对其推理的简要解释。


### 3.3 Planning Module
### 3.3 规划模块


The planning module in the UAstack is facilitated through OmniPlanner Zacharia et al. (2026), a graph-based planner designed to work across diverse aerial, ground, and underwater robot morphologies. The planner is currently applicable to systems for which graph-based planning is a viable option (e.g., various thrust-controlled aerial and underwater systems such as multirotors and ROVs, legged robots, and differential drive ground rovers). At the core, the planner utilizes a unified planning kernel that is agnostic to robot morphology, environment type, and mission objective and provides both target navigation as well as informative planning behaviors, such as exploration and inspection, based on the task objective $\mathcal{J}\left( {\mathcal{J}}_{TP}\right.$ : Planning to a target, ${\mathcal{J}}_{EP}$ : Exploration Planning, ${\mathcal{J}}_{IP}$ : Inspection Planning) set by the Mission Abstraction Layer. Algorithm 1 gives a high-level overview of the planning module, while Figure 5 illustrates the different parts of the module and their interactions. The reader is referred to Zacharia et al. (2026) for the algorithmic details regarding this module.
UAstack 中的规划模块由 OmniPlanner Zacharia et al. (2026) 提供支持，它是一种基于图的规划器，旨在适用于多种空中、地面和水下机器人形态。该规划器目前适用于图规划可行的系统（例如，各种推力控制的空中和水下系统，如多旋翼和 ROV、足式机器人，以及差速驱动地面巡航车）。其核心是一个统一的规划内核，对机器人形态、环境类型和任务目标均不敏感，并可根据任务目标 $\mathcal{J}\left( {\mathcal{J}}_{TP}\right.$ : 面向目标的规划、${\mathcal{J}}_{EP}$ : 探索规划、${\mathcal{J}}_{IP}$ : 检查规划）提供目标导航以及信息性规划行为，例如探索和检查，由任务抽象层设定。算法 1 概述了规划模块的高级流程，而图 5 展示了该模块的不同部分及其交互。关于该模块的算法细节，读者可参见 Zacharia et al. (2026)。


Algorithm 1 planning module
算法 1 规划模块


---



${\mu }_{R} \leftarrow$ GetRobotConstraints() 													D Robot
${\mu }_{R} \leftarrow$ 获取机器人约束() 													D 机器人


Abstraction Layer
抽象层


$\{ \mathcal{D},\mathcal{C}\}  \leftarrow$ GetSensorParams() $\; \vartriangleright$ Robot
$\{ \mathcal{D},\mathcal{C}\}  \leftarrow$ 获取传感器参数() $\; \vartriangleright$ 机器人


Abstraction Layer
抽象层


$\mathcal{J} \leftarrow$ GetMissionObjective() $\vartriangleright$ Mission
$\mathcal{J} \leftarrow$ 获取任务目标() $\vartriangleright$ 任务


Abstraction Layer
抽象层


repeat
重复


&nbsp;&nbsp;&nbsp;&nbsp;${\mathbb{G}}_{L} \leftarrow$ BuildLocalGraph $\left( {{\mu }_{R},\mathcal{M},\mathcal{H}}\right) \; \vartriangleright$ Planning
&nbsp;&nbsp;&nbsp;&nbsp;${\mathbb{G}}_{L} \leftarrow$ 构建局部图 $\left( {{\mu }_{R},\mathcal{M},\mathcal{H}}\right) \; \vartriangleright$ 规划


Kernel
内核


&nbsp;&nbsp;&nbsp;&nbsp;${\mathbb{G}}_{G} \leftarrow$ UpdateGlobalGraph $\left( {{\mu }_{R},{\mathbb{G}}_{L}}\right) \; \vartriangleright$ Planning
&nbsp;&nbsp;&nbsp;&nbsp;${\mathbb{G}}_{G} \leftarrow$ 更新全局图 $\left( {{\mu }_{R},{\mathbb{G}}_{L}}\right) \; \vartriangleright$ 规划


Kernel
内核


&nbsp;&nbsp;&nbsp;&nbsp;if $\mathcal{J} = {\mathcal{J}}_{TP}$ then
&nbsp;&nbsp;&nbsp;&nbsp;如果 $\mathcal{J} = {\mathcal{J}}_{TP}$ 则


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PlanningToTarget $\left( {{\mu }_{R},{\mathbf{p}}_{t},\mathcal{D},{\mathbb{G}}_{L},{\mathbb{G}}_{G}}\right)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;面向目标规划 $\left( {{\mu }_{R},{\mathbf{p}}_{t},\mathcal{D},{\mathbb{G}}_{L},{\mathbb{G}}_{G}}\right)$


&nbsp;&nbsp;&nbsp;&nbsp;else if $\mathcal{J} = {\mathcal{J}}_{EP}$ then
&nbsp;&nbsp;&nbsp;&nbsp;else if $\mathcal{J} = {\mathcal{J}}_{EP}$ then


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ExplorationPlanning $\left( {{\mu }_{R},\mathcal{D},{\mathbb{G}}_{L},{\mathbb{G}}_{G},\mathcal{M}}\right)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;探索规划 $\left( {{\mu }_{R},\mathcal{D},{\mathbb{G}}_{L},{\mathbb{G}}_{G},\mathcal{M}}\right)$


&nbsp;&nbsp;&nbsp;&nbsp;else if $\mathcal{J} = {\mathcal{J}}_{IP}$ then
&nbsp;&nbsp;&nbsp;&nbsp;else if $\mathcal{J} = {\mathcal{J}}_{IP}$ then


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;InspectionPlanning $\left( {{\mu }_{R},\mathcal{D},\mathcal{C},{\mathbb{G}}_{L},\mathcal{M}}\right)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;检查规划 $\left( {{\mu }_{R},\mathcal{D},\mathcal{C},{\mathbb{G}}_{L},\mathcal{M}}\right)$


&nbsp;&nbsp;&nbsp;&nbsp;end if
&nbsp;&nbsp;&nbsp;&nbsp;end if


until Robot Endurance Critical
直到机器人耐久度临界


ReturnToHome $\left( {{\mathbb{G}}_{L},{\mathbb{G}}_{G},{\mu }_{R}}\right) \; \vartriangleright$ Planning Kernel
返回原点 $\left( {{\mathbb{G}}_{L},{\mathbb{G}}_{G},{\mu }_{R}}\right) \; \vartriangleright$ 规划内核


---



3.3.1 Planning Kernel The Planning Kernel of OmniPlanner serves as the backbone of the methodology, providing the data structures and functions for searching the robot's admissible configuration space to enable the desired behaviors. All operations take place on a dual environment representation consisting of a) a Volumetric Map $\mathcal{M}$ (in this work Voxblox Oleynikova et al. (2017)), with voxel size ${v}_{m}$ ,and,when applicable,b) an elevation map $\mathcal{H}$ for ground robots (utilizing the work from Fankhauser et al. (2018,2014)),with grid size ${v}_{h}$ . The kernel employs a bifurcated local/global planning architecture. The Local Planning Submodule operates in a local volume ${\mathbf{b}}_{L}$ and constructs a bounded,sampling-based,dense graph ${\mathbb{G}}_{L}$ , with its vertex and edge sets ${\mathcal{V}}_{L},{\mathcal{E}}_{L}$ inside ${\mathbf{b}}_{L}$ spanning the locally reachable configuration space. All vertices and edges are sampled such that they lie entirely in collision-free space and respect the robot motion constraints ${\mu }_{R}$ defined by the Robot Abstraction Layer (e.g., traversability, robot size ${\mathbf{b}}_{R}$ ,etc.). The graph ${\mathbb{G}}_{L}$ is used by the planning module for local information gathering and collision-free navigation as described in the subsequent subsections. The Global Planning Submodule maintains a sparse global graph ${\mathbb{G}}_{G} = \left\{  {{\mathcal{V}}_{G},{\mathcal{E}}_{G}}\right\}$ built by aggregating the sparsified local planning graphs across the mission. This graph is used to represent the entire known space, providing fast global planning functionality. As ${\mathbb{G}}_{G}$ is built from ${\mathbb{G}}_{L}$ , all vertices and edges in ${\mathcal{V}}_{G},{\mathcal{E}}_{G}$ are in free space and satisfy the robot motion constraints. The Global Planning Submodule also keeps track of the robot's remaining endurance (or otherwise-defined remaining mission time). At each planning iteration, the Planning Kernel checks if the remaining time is sufficient to execute the path given by the specific behavior and return to the start location. If yes, that path is executed, else, the Kernel triggers a homing manuever to guide the robot back to the starting location.
3.3.1 规划内核 OmniPlanner 的规划内核是该方法的支柱，提供用于搜索机器人可接受配置空间的数据结构和函数，以实现所需行为。所有操作都在双环境表示上进行，该表示由 a) 体素地图 $\mathcal{M}$（本文使用 Voxblox Oleynikova et al. (2017)），体素大小为 ${v}_{m}$，以及在适用时 b) 面向地面机器人的高程地图 $\mathcal{H}$（采用 Fankhauser et al. (2018,2014) 的工作），网格大小为 ${v}_{h}$。该内核采用分叉的局部/全局规划架构。局部规划子模块在局部体积 ${\mathbf{b}}_{L}$ 中运行，构建一个有界的、基于采样的、稠密图 ${\mathbb{G}}_{L}$，其顶点集和边集 ${\mathcal{V}}_{L},{\mathcal{E}}_{L}$ 位于 ${\mathbf{b}}_{L}$ 内，覆盖局部可达的配置空间。所有顶点和边的采样都确保其完全位于无碰撞空间中，并满足机器人抽象层定义的机器人运动约束 ${\mu }_{R}$（例如，可通行性、机器人尺寸 ${\mathbf{b}}_{R}$ 等）。该图 ${\mathbb{G}}_{L}$ 由规划模块用于局部信息获取和无碰撞导航，如后续小节所述。全局规划子模块维护一个稀疏的全局图 ${\mathbb{G}}_{G} = \left\{  {{\mathcal{V}}_{G},{\mathcal{E}}_{G}}\right\}$，该图通过在整个任务过程中聚合稀疏化的局部规划图构建而成。该图用于表示整个已知空间，提供快速的全局规划功能。由于 ${\mathbb{G}}_{G}$ 由 ${\mathbb{G}}_{L}$ 构建，${\mathcal{V}}_{G},{\mathcal{E}}_{G}$ 中所有顶点和边都位于自由空间，并满足机器人运动约束。全局规划子模块还跟踪机器人剩余续航时间（或其他定义的剩余任务时间）。在每次规划迭代中，规划内核会检查剩余时间是否足以执行由特定行为给出的路径并返回起始位置。若足够，则执行该路径；否则，内核触发返航机动，引导机器人返回起始位置。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_9.jpg?x=129&y=138&w=1387&h=534&r=0"/>



Figure 5. Planning module architecture. The planning module is facilitated through OmniPlanner designed to work universally across aerial, ground, and underwater robot morphologies. At the core, the planner integrates a domain- and morphology-age planning kernel that utilizes the dual map representation to build local and global graphs, both satisfying the robot motion constraints set by the Robot Abstraction Layer. The local graph is used for local information gathering and collision-free navigation, whereas the global graph is used for fast but coarse planning in the entire known space. Utilizing these, the appropriate behavior (Exploration, Inspection, Planning to Target) is executed as per the task set by Mission Abstraction Layer.
图 5. 规划模块架构。规划模块通过为 OmniPlanner 设计的方式实现，可通用于空中、地面和水下机器人的不同形态。其核心是一个与领域和形态无关的规划内核，利用双地图表示构建局部和全局图，两者都满足机器人抽象层设定的机器人运动约束。局部图用于局部信息获取和无碰撞导航，而全局图用于在整个已知空间中进行快速但粗略的规划。借助这些，系统会按照任务抽象层设定的任务执行相应行为（探索、检查、前往目标规划）。


3.3.2 Planning to a Target UAstack facilitates planning to a desired waypoint ${\mathbf{p}}_{t}$ both within the already explored space, as well as in the unknown, as long as this is iteratively found to be possible. In each planning iteration, first, a guiding path ${\sigma }_{t}$ is calculated. If ${\mathbf{p}}_{t}$ lies in the known (explored) space, ${\sigma }_{t}$ is simply calculated as the shortest path along ${\mathbb{G}}_{G}$ to the vertex in ${\mathcal{V}}_{G}$ closest to ${\mathbf{p}}_{t}$ . Otherwise, ${\sigma }_{t}$ is calculated as the path towards the vertex that is on the frontier of the explored space and is closest to ${\mathbf{p}}_{t}$ . Next, ${\mathbb{G}}_{L}$ is used to calculate a local path ${\sigma }_{TP}$ guiding the robot along ${\sigma }_{t}$ which is then commanded to the subsequent modules to track. When the robot reaches within a distance ${d}_{\text{ path }}$ from the last point in ${\sigma }_{TP}$ ,the next iteration of the planner is triggered. This entire process is repeated until the robot reaches ${\mathbf{p}}_{t}$ or no progress can be made towards it,at which point the planner declares that the waypoint is unreachable. It is highlighted that the homing maneuver mentioned in Section 3.3.1 uses the Planning to Target behavior with the starting location as ${\mathbf{p}}_{t}$ .
3.3.2 规划到目标 UAstack 能够在已探索空间内以及未知空间中，将规划引导到期望的航点 ${\mathbf{p}}_{t}$，只要能够通过迭代确认这一点在可行范围内。在每次规划迭代中，首先会计算一条引导路径 ${\sigma }_{t}$。如果 ${\mathbf{p}}_{t}$ 位于已知（已探索）空间中，${\sigma }_{t}$ 只需沿 ${\mathbb{G}}_{G}$ 计算从 ${\mathbb{G}}_{G}$ 到 ${\mathcal{V}}_{G}$ 中最接近 ${\mathbf{p}}_{t}$ 的顶点的最短路径即可。否则，${\sigma }_{t}$ 则计算为指向位于已探索空间前沿、且最接近 ${\mathbf{p}}_{t}$ 的那个顶点的路径。接着，使用 ${\mathbb{G}}_{L}$ 计算一条局部路径 ${\sigma }_{TP}$，引导机器人沿 ${\sigma }_{t}$ 行进，然后将其下发给后续模块以进行跟踪。当机器人距离 ${\sigma }_{TP}$ 中最后一个点的距离 ${d}_{\text{ path }}$ 内时，会触发规划器的下一次迭代。整个过程重复进行，直到机器人到达 ${\mathbf{p}}_{t}$，或无法再朝其取得进展；此时规划器会声明该航点不可达。需要强调的是，3.3.1 节中提到的回航动作使用的是“规划到目标”行为，其起始位置设置为 ${\mathbf{p}}_{t}$ 。


3.3.3 Exploration Planning The first informative planning behavior supported by OmniPlanner is the Volumetric Exploration (VE), where the robot is tasked to iteratively uncover the unknown volume using an Field of View (FoV)- and range-constrained depth sensor $\mathcal{D}$ (whose parameters are given by the Robot Abstraction Layer). In each planning iteration, ${\mathbb{G}}_{L}$ is used to find the path that leads to uncovering the largest amount of unknown space. First, shortest paths from the current robot location to each vertex in ${\mathcal{V}}_{L}$ are calculated. An information gain,called Volume Gain, related to the amount of unknown volume mapped by $\mathcal{D}$ from a robot configuration,is calculated for each vertex in ${\mathcal{V}}_{L}$ . The path ${\sigma }_{EP}$ with the highest aggregated Volume Gain is selected as the next exploration path. When the robot reaches within a distance ${d}_{\text{ path }}$ from the last point in ${\sigma }_{EP}$ , the next iteration of the planner is triggered and the process is repeated. When no informative path is found in ${\mathbb{G}}_{L}$ ,the ${\mathbb{G}}_{G}$ is utilized to reposition the robot to a frontier of the explored space. The planner tracks vertices in ${\mathbb{G}}_{G}$ having high Volume Gain (called frontier vertices), and repositions the robot to the frontier vertex having the highest gain using the Planning to Target behavior described in Section 3.3.2. Upon reaching the frontier, local exploration continues.
3.3.3 探索规划 OmniPlanner 支持的首个信息性规划行为是体积探索（Volumetric Exploration, VE）。在该行为中，机器人被要求使用视场（FoV）与距离约束的深度传感器 $\mathcal{D}$（其参数由机器人抽象层给定）迭代地揭示未知体积。在每次规划迭代中，使用 ${\mathbb{G}}_{L}$ 来寻找能够揭示最大未知空间量的路径。首先，计算机器人当前位置到 ${\mathcal{V}}_{L}$ 中每个顶点的最短路径。对 ${\mathcal{V}}_{L}$ 中每个顶点，计算与由机器人位姿配置下的 $\mathcal{D}$ 所映射的未知体积量相关的信息增益（称为体积增益 Volume Gain）。选择聚合后的体积增益最高的路径 ${\sigma }_{EP}$ 作为下一条探索路径。当机器人距离 ${\sigma }_{EP}$ 的最后一个点在 ${d}_{\text{ path }}$ 之内时，会触发规划器的下一次迭代并重复该过程。当在 ${\mathbb{G}}_{L}$ 中找不到信息性路径时，使用 ${\mathbb{G}}_{G}$ 将机器人重新定位到已探索空间的前沿。规划器会跟踪具有高体积增益的顶点（称为前沿顶点），并通过 3.3.2 节所述的“规划到目标”行为，将机器人重新定位到增益最高的前沿顶点。到达前沿后，继续进行局部探索。


3.3.4 Inspection Planning In the Visual Inspection (VI) behavior, the planner is tasked to inspect a subset of the occupied surface in the mapped volume using an FoV-and range-constrained camera sensor $\mathcal{C}$ (whose parameters are given by the Robot Abstraction Layer) at the desired viewing distance ${d}_{\text{ view }}$ (given by the Mission Abstraction Layer). A set $\mathbb{V}$ of viewpoints,at a distance ${d}_{\text{ view }}$ from the occupied surface,is built. A graph ${\mathbb{G}}_{VI}$ is built using the Local Planning Submodule to connect the viewpoints in $\mathbb{V}$ . The minimal viewpoint set ${\mathbb{V}}_{\text{ best }}$ viewing the entire surface is selected,and the order to visit them is calculated by solving the Traveling Salesman Problem (TSP) problem. The shortest paths along ${\mathbb{G}}_{VI}$ connecting the subsequent viewpoint in the tour are concatenated to form the inspection path.
3.3.4 检测规划 在视觉检测（Visual Inspection, VI）行为中，规划器被要求使用视场（FoV）与距离约束的相机传感器 $\mathcal{C}$（其参数由机器人抽象层给定），在期望观察距离 ${d}_{\text{ view }}$（由任务抽象层给定）下，检查映射体积中占据表面的一部分。构建一组视点集合 $\mathbb{V}$，其与占据表面保持 ${d}_{\text{ view }}$ 的距离。使用局部规划子模块构建图 ${\mathbb{G}}_{VI}$，以连接 $\mathbb{V}$ 中的各个视点。选择能够覆盖整个表面的最小视点集合 ${\mathbb{V}}_{\text{ best }}$，并通过求解旅行商问题（TSP）来计算访问它们的顺序。沿 ${\mathbb{G}}_{VI}$ 连接巡回中相邻视点的最短路径会被拼接起来，形成检测路径。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_10.jpg?x=144&y=144&w=1371&h=790&r=0"/>



Figure 6. Navigation module architecture. Two swappable local navigation modalities are offered: (top) SDF-NMPC, which encodes depth images into a latent SDF representation online and embeds it as a constraint in an NMPC controller; and (bottom) ExRL, which combines inverted range images with proprioceptive state through a policy trained using PPO in simulation to directly output acceleration commands for waypoint-directed collision-free navigation. In both cases, the resulting acceleration control reference is passed through the C-CBF safety filter, which composes range measurements into a composite barrier function to reactively modify the reference acceleration and guarantee collision-free operation as a last resort, before forwarding to the robot-specific low-level controller.
图6. 导航模块架构。提供两种可互换的本地导航模式：（上）SDF-NMPC：将深度图像在线编码为潜在SDF表示，并将其作为约束嵌入NMPC控制器；以及（下）ExRL：将反向距离图像与本体感知状态结合，并通过在仿真中用PPO训练得到的策略，直接输出用于航点定向、无碰撞导航的加速度指令。在这两种情况下，得到的加速度控制参考都会先经过C-CBF安全滤波器：将距离测量整合为复合屏障函数，以便在最后手段时对参考加速度进行反应式修正，从而保证无碰撞运行，然后再转交给面向机器人特定的低层控制器。


### 3.4 Navigation Module
### 3.4 导航模块


The UAstack takes a multi-layered approach to safety, illustrated in Figure 6. Conventional modern safe navigation and collision avoidance are based on the planning of paths subject to map constraints, which are then blindly followed by an onboard controller. Despite the major success of this paradigm in many environments and mission profiles, as discussed in Jacquet et al. (2025); Kulkarni and Alexis (2024) and demonstrated in field experience Ebadi et al. (2023) it represents a single point of failure which can lead robots to collisions due to odometry errors or erroneous/incomplete mapping. Although such errors are not common, experience shows that they do manifest and are often catastrophic, at least regarding a robot's ability to continue its mission. Furthermore, odometry and mapping challenges manifest more frequently in perceptually-degraded and geometrically complex environments Ebadi et al. (2023). While the UAstack does perform map-based avoidance through volumetric mapping for collisions and traversability analysis for ground systems, it further adds two redundant layers of safety: depth sensor-based trajectory tracking and a last-resort safety filtering based on Control Barrier Functions (CBFs). With respect to depth sensor-driven navigation policies, two approaches are offered within the stack, owing to their distinct benefits in certain conditions: (i) a Signed Distance Function (SDF)-based neural NMPC Jacquet et al. (2025) or (ii) a novel DRL-based policy trained for safe navigation and smooth collision avoidance. These methods enable local deviations from the reference trajectory if and when necessary, to ensure collision avoidance. Both methods provide swappable local navigation policies with distinct performance characteristics that will be discussed in Section 3.4.4. To further assert safety with formal guarantees, a composite CBF-based formulation building upon Harms et al. (2025) is introduced as a last-resort safety filter allowing to modify the control references in the unlikely situation that all other collision-avoidance methods in the stack fail. Importantly, this additional safety layer within the UAstack is currently implemented for obstacle avoidance and not for traversability analysis, predominantly targeting flying robots. This architecture extends upon recent ideas and developments in the research community, such as the perceptive locomotion in Miki et al. (2022), which indeed may also be used as an alternative to the exteroceptive depth navigation strategies discussed below when it comes to quadruped robots.
UAstack 采用多层次的安全策略，如图 6 所示。传统的现代安全导航与避碰通常基于受地图约束的路径规划，然后由机载控制器盲目跟踪。尽管这种范式在许多环境和任务场景中都取得了巨大成功，如 Jacquet et al. (2025)；Kulkarni and Alexis (2024) 所述，并由 Ebadi et al. (2023) 的实地经验所证明，但它也是单点故障源，可能因里程计误差或错误/不完整建图而导致机器人碰撞。虽然这类错误并不常见，但经验表明它们确实会发生，并且往往是灾难性的，至少会影响机器人继续执行任务的能力。此外，里程计和建图方面的挑战在感知退化且几何复杂的环境中更为常见 Ebadi et al. (2023)。尽管 UAstack 通过体素建图进行碰撞规避，并通过可通行性分析服务于地面系统，但它进一步增加了两层冗余安全机制：基于深度传感器的轨迹跟踪，以及基于控制屏障函数（CBFs）的最后安全过滤。对于深度传感器驱动的导航策略，系统中提供了两种方法，鉴于它们在特定条件下各有优势：(i) 基于符号距离函数（SDF）的神经 NMPC Jacquet et al. (2025)，或 (ii) 一种新颖的基于 DRL 的策略，经过训练可实现安全导航和平滑避碰。这些方法在必要时允许相对参考轨迹做局部偏离，以确保避碰。两种方法都提供可互换的局部导航策略，具有不同的性能特征，将在 3.4.4 节讨论。为进一步以形式化保证强化安全性，系统引入了一个基于 CBF 的复合形式，建立在 Harms et al. (2025) 的基础上，作为最后的安全过滤器；在极不可能的情况下，当系统中所有其他避碰方法都失效时，它可修改控制参考。需要强调的是，UAstack 中这一额外安全层目前仅针对障碍物规避实现，而非针对可通行性分析，主要面向飞行机器人。该架构延伸了研究社区中的近期思想与进展，例如 Miki et al. (2022) 的感知运动控制；对于四足机器人而言，它实际上也可作为下文讨论的外部感知深度导航策略的替代方案。


3.4.1 Neural SDF-NMPC Detailed in Jacquet et al. (2025), the SDF-NMPC enables collision-free navigation in unknown environments relying only on depth sensing and (possibly drifting) odometry. In its current implementation, it emphasizes rotorcraft navigation.
3.4.1 神经 SDF-NMPC 如 Jacquet et al. (2025) 所详述，SDF-NMPC 仅依赖深度感知和（可能漂移的）里程计，即可在未知环境中实现无碰撞导航。在当前实现中，它侧重于旋翼飞行器导航。


The SDF-NMPC represents the visible environment as a Euclidean SDF, defined to be positive in visible free space and negative in occluded regions (i.e., behind obstacles). For efficient computation and to provide a representation compatible with gradient-based NMPC, this SDF is approximated by a neural network, online, from the latest depth measurement. As a result, the environment is described purely locally, improving robustness to potentially drifting odometry. To keep the representation compatible with a compact neural model, the SDF is saturated beyond a threshold ${T}_{\mathrm{{SDF}}}$ . This SDF is constructed online from the latest depth measurement. As a result, the environment is described purely locally, improving robustness to potentially drifting odometry.
SDF-NMPC 将可见环境表示为欧氏 SDF，其在可见自由空间中为正，在遮挡区域（即障碍物后方）中为负。为提高计算效率并提供与基于梯度的 NMPC 兼容的表示，该 SDF 通过神经网络在线地由最新深度测量近似得到。因此，环境仅以局部方式描述，提高了对潜在里程计漂移的鲁棒性。为使表示与紧凑的神经模型兼容，SDF 在超过阈值 ${T}_{\mathrm{{SDF}}}$ 后会被截断。该 SDF 由最新深度测量在线构建。因此，环境仅以局部方式描述，提高了对潜在里程计漂移的鲁棒性。


A two-stage neural network is used. First, the input depth image is clamped at a distance ${d}_{\max }$ ,since only short-range surroundings are relevant for short-horizon collision avoidance. Compression into a low-dimensional latent space $\mathbf{z}$ is achieved via a convolutional encoder,trained jointly with a decoder to reconstruct the input,ensuring that $\mathbf{z}$ captures a reliable latent representation of the depth data. Notably, the encoding is biased to place greater emphasis on obstacles close to the robot, encouraging accurate encoding of nearby geometry. The decoder is used only during training, while the encoder provides input to a downstream Multi-Layer Perceptron (MLP) network that reconstructs the SDF.
采用两阶段神经网络。首先，将输入深度图像在距离 ${d}_{\max }$ 处进行截断，因为对于短时域避碰而言，只有近距离环境才相关。随后通过卷积编码器压缩到低维潜空间 $\mathbf{z}$，并与解码器联合训练以重建输入，确保 $\mathbf{z}$ 捕获深度数据的可靠潜表示。值得注意的是，该编码更强调机器人附近的障碍物，从而鼓励对近处几何结构进行准确编码。解码器仅在训练期间使用，而编码器则为下游的多层感知机（MLP）网络提供输入，由其重建 SDF。


Specifically, this MLP takes as input the latent vector and a 3D position, and approximates the corresponding SDF value evaluated in that point. The regression task is trained in a supervised manner, including losses that enforce consistency of the SDF gradient. In this way, the trained MLP represents the following parametric function:
具体而言，该 MLP 以潜向量和三维位置作为输入，近似该点处对应的 SDF 值。该回归任务以监督方式训练，并包括约束 SDF 梯度一致性的损失。通过这种方式，训练后的 MLP 表示如下参数化函数：


(10)

$$
{\mathbb{R}}^{3} \rightarrow  \mathbb{R}
$$

$$
\mathbf{p} \mapsto  {\operatorname{SDF}}_{\mathbf{\theta },\mathbf{z}}\left( \mathbf{p}\right) ,
$$



where $\mathbf{\theta }$ are the neural network weights,and $\mathbf{z}$ is the latent code corresponding to the depth measurement.
其中 $\mathbf{\theta }$ 为神经网络权重，$\mathbf{z}$ 为对应深度测量的潜代码。


Finally, the neural SDF is embedded into the nonlinear NMPC controller as an explicit position constraint. The following constraint is enforced over the receding horizon:
最后，神经 SDF 以显式位置约束的形式嵌入非线性 NMPC 控制器中。以下约束在滚动时域内被施加：


$$
{\mathrm{{SDF}}}_{\mathbf{\theta },\mathbf{z}}\left( {\mathbf{p}}_{\mathrm{S}}^{\mathrm{B}}\right)  \geq  r \tag{11}
$$



where $r$ is a user-defined threshold accounting for the robot radius and a possible safety margin,and ${\mathbf{p}}_{\mathrm{S}}^{\mathrm{B}}$ denotes the position of the robot expressed in the frame $\{ \mathrm{S}\}$ in which the depth measurement was captured. Note that additional constraints further ensure that the robot remains within the sensor frustum, i.e., within the region that is currently observable and where the neural SDF is defined. This follows the intuitive principle of "look where you move", effectively restricting motion to visible free space.
其中$r$是用户自定义的阈值，用于考虑机器人半径以及可能的安全裕度；${\mathbf{p}}_{\mathrm{S}}^{\mathrm{B}}$表示机器人在坐标系$\{ \mathrm{S}\}$中的位置，而$\{ \mathrm{S}\}$正是捕获深度测量的坐标系。注意，额外的约束还进一步保证机器人保持在传感器视锥内，即在当前可观测区域内，并且神经SDF在该区域中有定义。这遵循直观的“朝着你能看到的地方移动”原则，从而将运动限制在可见的自由空间。


Critically, the method enforces feasibility and stability-type criteria (under fixed sensor observations), with a terminal condition ensuring that the terminal state allows a collision-free braking maneuver to hover. This is assessed by computing the minimum braking distance given the input bounds, and evaluating the SDF at the predicted hovering position. Under this condition, recursive feasibility is ensured, and with a suitable quadratic terminal cost, the optimal value is shown to be non-increasing over time. The NMPC generates acceleration commands from a velocity reference trajectory, since velocity tracking prevents the accumulation of position errors when collision constraints prevent accurate tracking of a nominal trajectory. Accordingly, the interface with the planning module provides such velocity references derived from planned paths.
关键在于，该方法在固定传感器观测下强制满足可行性与稳定性类准则，并通过终端条件确保终端状态允许无碰撞制动以悬停。评估时会在输入界限内计算最小制动距离，并在预测的悬停位置处计算SDF。在该条件下，递归可行性得到保证；同时，在合适的二次型终端代价下，最优值被证明随时间不增。NMPC会从速度参考轨迹生成加速度指令，因为当碰撞约束阻止对名义轨迹进行精确跟踪时，速度跟踪能够避免位置误差的累积。因此，与规划模块的接口会提供由规划路径生成的这类速度参考。


3.4.2 Exteroceptive Deep Reinforcement Learning The UAstack further offers exteroceptive DRL-based navigation (dubbed ExRL). The proposed novel Exteroceptive DRL (ExRL) approach is formulated as a waypoint navigation problem and considers as input the vector to goal location, the robot orientation, velocity, and angular rates alongside the instantaneous depth image from an exteroceptive sensor (including stereo or RGB-D cameras, Time-of-Flight camera sensors, or LiDARs). We employ end-to-end learning to train a navigation policy to generate commands directly from the robot's current state and range measurement. The policy is trained using the Aerial Gym Simulator Kulkarni et al. (2025b) to command acceleration and yaw-rate set-point commands (as commonly provided by most autopilots such as PX4 Meier et al. (2015) and ArduPilot ArduPi-lot Dev Team (2024)). Relevant open-source examples for training are provided in Aerial Gym, while the policy can be trained on any compatible simulation tool. Similar to the SDF-NMPC, ExRL ensures safe collision-free navigation without a map and thus contributes to multi-layered safety. This work is distinct from prior work of the authors in Kulkarni et al. (2023); Kulkarni and Alexis (2023), by a) departing from a modularized two-step approach and introducing an end-to-end methodology, b) introducing a novel reward function incorporating Time-to-Collision (TTC), and c) commanding acceleration and yaw-rate setpoints instead of velocity references.
3.4.2 外感知深度强化学习 UAstack进一步提供基于外感知DRL的导航（称为ExRL）。所提出的新型外感知DRL（ExRL）方法被表述为一个航点导航问题，其输入包括指向目标位置的向量、机器人的姿态、速度与角速度，以及来自外感知传感器的瞬时深度图像（包括双目或RGB-D相机、ToF相机传感器或LiDAR）。我们采用端到端学习来训练导航策略，使其直接依据机器人的当前状态和距离测量生成指令。该策略使用Aerial Gym Simulator Kulkarni等（2025b）进行训练，以输出加速度与偏航角速率设定点指令（通常由诸如PX4 Meier等（2015）和ArduPilot ArduPi-lot Dev Team（2024）等多数飞控系统提供）。训练的相关开源示例在Aerial Gym中给出，而该策略也可在任何兼容的仿真工具中进行训练。与SDF-NMPC类似，ExRL无需地图即可保证安全无碰撞导航，从而促进多层次安全。该工作区别于Kulkarni等（2023）及Kulkarni和Alexis（2023）的既有研究：a）从模块化的两步方法转向，并引入端到端方案；b）提出了包含碰撞时间（TTC）的新奖励函数；c）下发的是加速度与偏航角速率设定点，而不是速度参考。


Observations The observation vector for training the policy consists of both proprioceptive and exteroceptive components. The proprioceptive components are expressed in a yaw-aligned, roll and pitch stabilized coordinate frame \{V\} sharing the same origin as the robot body IMU frame \{B\}. Let ${\mathbf{p}}^{\mathrm{w}},{\mathbf{p}}_{ * }^{\mathrm{w}} \in  {\mathbb{R}}^{3}$ be the robot and target positions, $\psi ,{\psi }^{ * }$ be the robot and target yaw respectively, the yaw error be ${\psi }_{e} = \left( {\left( {{\psi }^{ * } - \psi  + \pi }\right) {\;\operatorname{mod}\;2}\pi }\right)  - \pi$ ,and ${\delta }^{w} = {\mathbf{p}}_{ * }^{w} - {\mathbf{p}}^{w}$ be the position error with $\delta  = \parallel \mathbf{\delta }\parallel$ the vector to the goal. During training,the goal direction ${\mathbf{\delta }}^{\mathrm{V}}/\delta$ ,roll $\phi$ and pitch $\theta$ measurements are perturbed by adding noise sampled from a uniform distribution before populating the observation tensor. The linear ${\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{B}}$ and angular velocities ${\mathbf{\omega }}_{\mathrm{{WB}}}^{\mathrm{B}}$ are not perturbed. The range measurements from the exteroceptive sensor are min-pooled to ${16} \times  {20}$ pixels and inverted as a 2D inverse range-image ${\mathbf{I}}_{\mathbf{r}}$ . Actions The DRL policy outputs a normalized action command ${\mathbf{a}}_{t} = \left\{  {{a}_{x},{a}_{y},{a}_{z},{a}_{\dot{\psi }}}\right\}   \in  {\left\lbrack  -1,1\right\rbrack  }^{4}$ that is scaled and mapped to linear acceleration and yaw-rate setpoint commands at time $t$ as ${\mathbf{u}}_{t}^{acc} = \left( {{u}_{n}^{x}{a}_{x},{u}_{n}^{y}{a}_{y},{u}_{n}^{z}{a}_{z},{u}_{n}^{\dot{\psi }}{a}_{\dot{\psi }}}\right)$ , where ${u}_{n}^{x},{u}_{n}^{y},{u}_{n}^{z},{u}_{n}^{\dot{\psi }}$ are tunable scaling parameters,with their values described in Table 6.
观察训练策略的观测向量由本体感知与外部感知两部分组成。本体感知部分在偏航对齐、滚转与俯仰稳定的坐标系 \{V\} 中表示，并与机器人机身 IMU 坐标系 \{B\} 共用同一原点。令 ${\mathbf{p}}^{\mathrm{w}},{\mathbf{p}}_{ * }^{\mathrm{w}} \in  {\mathbb{R}}^{3}$ 为机器人与目标的位置，$\psi ,{\psi }^{ * }$ 为机器人与目标的偏航角；偏航误差为 ${\psi }_{e} = \left( {\left( {{\psi }^{ * } - \psi  + \pi }\right) {\;\operatorname{mod}\;2}\pi }\right)  - \pi$，位置误差为 ${\delta }^{w} = {\mathbf{p}}_{ * }^{w} - {\mathbf{p}}^{w}$，其中 $\delta  = \parallel \mathbf{\delta }\parallel$ 为指向目标的向量。在训练过程中，目标方向 ${\mathbf{\delta }}^{\mathrm{V}}/\delta$、滚转 $\phi$ 与俯仰 $\theta$ 的测量会先加入从均匀分布中采样得到的噪声，然后再填充到观测张量中。线速度 ${\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{B}}$ 与角速度 ${\mathbf{\omega }}_{\mathrm{{WB}}}^{\mathrm{B}}$ 不做扰动。外部传感器的距离测量先在最小池化后得到 ${16} \times  {20}$ 个像素，并被反转为二维的反距离图像 ${\mathbf{I}}_{\mathbf{r}}$ 。动作 DRL 策略输出一个归一化的动作指令 ${\mathbf{a}}_{t} = \left\{  {{a}_{x},{a}_{y},{a}_{z},{a}_{\dot{\psi }}}\right\}   \in  {\left\lbrack  -1,1\right\rbrack  }^{4}$，在时间 $t$ 处按 ${\mathbf{u}}_{t}^{acc} = \left( {{u}_{n}^{x}{a}_{x},{u}_{n}^{y}{a}_{y},{u}_{n}^{z}{a}_{z},{u}_{n}^{\dot{\psi }}{a}_{\dot{\psi }}}\right)$ 进行缩放并映射为线性加速度与偏航角速率设定值命令，其中 ${u}_{n}^{x},{u}_{n}^{y},{u}_{n}^{z},{u}_{n}^{\dot{\psi }}$ 为可调的缩放参数，它们的取值见表 6。


Table 2. Observation vector ${\mathbf{o}}_{t} \in  {\mathbb{R}}^{337}$ .
表 2. 观测向量 ${\mathbf{o}}_{t} \in  {\mathbb{R}}^{337}$ 。


<table><tr><td>Index</td><td>Symbol</td><td>Description</td></tr><tr><td>0:2</td><td>${\delta }^{\mathrm{V}}/\delta$</td><td>Goal direction in $\{ V\}$</td></tr><tr><td>3</td><td>$\delta$</td><td>Distance to goal (m)</td></tr><tr><td>4</td><td>$\phi$</td><td>Roll (rad)</td></tr><tr><td>5</td><td>$\theta$</td><td>Pitch (rad)</td></tr><tr><td>6</td><td>${\psi }_{e}$</td><td>Yaw error (rad)</td></tr><tr><td>7:9</td><td>${v}_{\mathrm{{WB}}}^{\mathrm{B}}$</td><td>Linear velocity in \{B\} (m/s)</td></tr><tr><td>10:12</td><td>${\omega }_{\mathrm{{WB}}}^{\mathrm{B}}$</td><td>Angular velocity in $\{ B\}$ (rad/s)</td></tr><tr><td>13:16</td><td>${\mathbf{u}}_{t - 1}^{acc}$</td><td>Previous setpoint $\left( {\mathrm{m}/{\mathrm{s}}^{2},\mathrm{{rad}}/\mathrm{s}}\right)$</td></tr><tr><td>17:336</td><td>${\mathbf{I}}_{\mathbf{r}}$</td><td>Inverse-range image $\left( {\mathrm{m}}^{-1}\right)$</td></tr></table>
<table><tbody><tr><td>索引</td><td>符号</td><td>描述</td></tr><tr><td>0:2</td><td>${\delta }^{\mathrm{V}}/\delta$</td><td>在$\{ V\}$中的目标方向</td></tr><tr><td>3</td><td>$\delta$</td><td>到目标的距离（m）</td></tr><tr><td>4</td><td>$\phi$</td><td>横滚（rad）</td></tr><tr><td>5</td><td>$\theta$</td><td>俯仰（rad）</td></tr><tr><td>6</td><td>${\psi }_{e}$</td><td>偏航误差（rad）</td></tr><tr><td>7:9</td><td>${v}_{\mathrm{{WB}}}^{\mathrm{B}}$</td><td>\{B\}中的线速度（m/s）</td></tr><tr><td>10:12</td><td>${\omega }_{\mathrm{{WB}}}^{\mathrm{B}}$</td><td>在$\{ B\}$中的角速度（rad/s）</td></tr><tr><td>13:16</td><td>${\mathbf{u}}_{t - 1}^{acc}$</td><td>前一个设定点$\left( {\mathrm{m}/{\mathrm{s}}^{2},\mathrm{{rad}}/\mathrm{s}}\right)$</td></tr><tr><td>17:336</td><td>${\mathbf{I}}_{\mathbf{r}}$</td><td>$\left( {\mathrm{m}}^{-1}\right)$中的逆距离图像</td></tr></tbody></table>


Time-to-Collision (TTC) We propose the usage of an expected "time-to-collision" metric at each timestep that provides a dense reward signal to the robot. This metric is distinct compared to approaches that directly reward based on raw range data Kulkarni and Alexis (2024), thus prioritizing the relationship between velocity and distance to measured obstacles, only penalizing rapid approaches to obstacles. This metric is calculated only in simulation and is treated as privileged information that is not available to the policy,yet influences the reward signal. For each point $i$ in the full-resolution point cloud expressed in a sensor frame $\{ \mathrm{S}\}$ ,the vector from the sensor is calculated as ${\mathbf{r}}^{\mathrm{S}}{}_{i}$ . The linear component of velocity of the robot along the direction to each point is computed as the projection of ${\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{S}}$ onto the unit direction vector of ${\mathbf{r}}^{\mathrm{S}}{}_{i}$ . This is subsequently used to calculate the expected time to collision ${\tau }_{i}$ using the distance of the point from the robot:
碰撞时间（TTC）我们提出在每个时间步使用期望的“碰撞时间”指标，为机器人提供稠密奖励信号。与直接基于原始距离数据进行奖励的方法 Kulkarni and Alexis (2024) 不同，该指标强调速度与测得障碍物距离之间的关系，仅对快速接近障碍物进行惩罚。该指标仅在仿真中计算，并被视为策略不可见的特权信息，但会影响奖励信号。对于在传感器坐标系 $\{ \mathrm{S}\}$ 中表示的完整分辨率点云中的每个点 $i$，计算从传感器指向该点的向量为 ${\mathbf{r}}^{\mathrm{S}}{}_{i}$。机器人沿每个点方向的速度线性分量通过将 ${\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{S}}$ 投影到 ${\mathbf{r}}^{\mathrm{S}}{}_{i}$ 的单位方向向量上来计算。随后结合该点相对于机器人的距离，用于计算期望碰撞时间 ${\tau }_{i}$：


$$
{v}_{i}^{ \bot  } = {\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{S}} \cdot  \frac{{\mathbf{r}}^{\mathrm{S}}}{\begin{Vmatrix}{\mathbf{r}}^{\mathrm{S}}{}_{i}\end{Vmatrix}},\;{\tau }_{i} = \left\{  \begin{array}{ll} \begin{Vmatrix}{{\mathbf{r}}^{\mathrm{S}}{}_{i}}\end{Vmatrix}/{v}_{i}^{ \bot  } & {v}_{i}^{ \bot  } > 0, \\  {10}\mathrm{\;s} & {v}_{i}^{ \bot  } \leq  0. \end{array}\right. \tag{12}
$$



Positive time-to-collision values are clamped between $0\mathrm{\;s}$ and ${10}\mathrm{\;s}$ . Negative values indicate that the robot is moving away from an obstacle,hence they are set to ${10}\mathrm{\;s}$ to make their effect negligible. The minimum time-to-collision ${\tau }_{\min } = \mathop{\min }\limits_{i}{\tau }_{i}$ across all points is considered and used to penalize the robot, emphasizing imminent collisions while largely ignoring well-separated obstacles.
正的碰撞时间值会被截断到 $0\mathrm{\;s}$ 和 ${10}\mathrm{\;s}$ 之间。负值表示机器人正在远离障碍物，因此将其设为 ${10}\mathrm{\;s}$，使其影响可忽略不计。随后取所有点中的最小碰撞时间 ${\tau }_{\min } = \mathop{\min }\limits_{i}{\tau }_{i}$ 作为惩罚依据，突出迫近碰撞，同时基本忽略距离较远的障碍物。


Policy Architecture The observation vector ${\mathbf{o}}_{t}$ is partitioned into a proprioceptive and an exteroceptive component, as detailed in Table 2. The latter is treated as a single-channel 2D image and processed by a Convolutional Neural Network (CNN)-based encoder ${f}_{\mathrm{{CNN}}}$ with three convolutional blocks. Each block consists of a $3 \times  3$ convolution with padding of width 1, an Exponential Linear Unit (ELU) activation, and a $3 \times  3$ max-pooling layer. The resulting feature map is flattened into a 128-dimensional range embedding, which is concatenated with the proprioceptive state vector. This combined representation is then passed through an MLP with hidden layer sizes $\left\lbrack  {{256},{128},{64}}\right\rbrack$ and ELU activations, followed by a 128-dimensional Gated Recurrent Unit (GRU) layer. The policy is trained using Proximal Policy Optimization (PPO) Schulman et al. (2017) implementation provided by Sample Factory Petrenko et al. (2020). The training time is about ${60}\mathrm{\;{min}}$ on a consumer grade laptop with an NVIDIA RTX 3080 Ti GPU.
策略架构 观测向量 ${\mathbf{o}}_{t}$ 被划分为本体感知和外部感知两部分，如表 2 所示。后者被视为单通道二维图像，并由基于卷积神经网络（CNN）的编码器 ${f}_{\mathrm{{CNN}}}$ 处理，该编码器包含三个卷积块。每个块由一个宽度为 1 的填充 $3 \times  3$ 卷积、一个指数线性单元（ELU）激活和一个 $3 \times  3$ 最大池化层组成。得到的特征图被展平为 128 维的距离嵌入，并与本体感知状态向量拼接。随后将这一组合表示输入一个隐藏层大小为 $\left\lbrack  {{256},{128},{64}}\right\rbrack$ 且带 ELU 激活的 MLP，再接一个 128 维门控循环单元（GRU）层。该策略使用 Sample Factory Petrenko et al. (2020) 提供的近端策略优化（PPO） Schulman et al. (2017) 实现进行训练。在配备 NVIDIA RTX 3080 Ti GPU 的消费级笔记本电脑上，训练时间约为 ${60}\mathrm{\;{min}}$。


Environment Setup and Curriculum Each simulated environment is composed of a rectangular room with dimensions ranging from ${10} \times  {10} \times  6$ to ${15} \times  {15} \times  {10}\mathrm{\;m}$ . A simulated robot is initialized on one side of the environment while the goal position is sampled on the opposite side with an arbitrary yaw setpoint. The acceleration and yaw-rate setpoint is tracked using a controller derived from the work in Lee et al. (2010), whose parameters are randomized at each episode to increase robustness and improve sim2real performance. Within each environment, 25 to 70 cuboidal obstacles of various sizes are sampled. An episode is marked as a success if the robot reaches within $1\mathrm{\;m}$ of the goal after a predefined number of time steps. If the robot remains collision-free but does not reach the goal, the episode is marked as a timeout. Collisions with obstacles are detected by the physics engine, terminate the episode, and are recorded as crashes. To encourage stable learning across various environment complexities,a curriculum adjusts the number of obstacles ${n}_{\text{ obs }}$ in each environment. The number is increased or decreased when the average success rate ${\zeta }_{s}$ over 2048 episodes crosses the upper or lower thresholds, ${\zeta }_{s}^{ + } = {0.70}$ and ${\zeta }_{s}^{ - } = {0.60}$ , respectively:
环境设置与课程学习 每个仿真环境由一个矩形房间组成，尺寸范围为 ${10} \times  {10} \times  6$ 到 ${15} \times  {15} \times  {10}\mathrm{\;m}$。仿真机器人初始化在环境一侧，而目标位置在另一侧随机采样，并设置任意偏航设定点。加速度和偏航角速度设定点由一个源自 Lee et al. (2010) 的控制器跟踪，其参数在每个回合随机化，以提高鲁棒性并改善仿真到现实的性能。在每个环境中，会采样 25 到 70 个不同尺寸的长方体障碍物。如果机器人在预定义时间步数后到达目标点 $1\mathrm{\;m}$ 范围内，则该回合记为成功；如果机器人始终无碰撞但未到达目标，则记为超时。与障碍物的碰撞由物理引擎检测，会终止回合，并记录为撞击。为促进在不同环境复杂度下的稳定学习，课程学习会调整每个环境中的障碍物数量 ${n}_{\text{ obs }}$。当 2048 个回合上的平均成功率 ${\zeta }_{s}$ 超过上阈值或低于下阈值时，该数量会相应增加或减少，阈值分别为 ${\zeta }_{s}^{ + } = {0.70}$ 和 ${\zeta }_{s}^{ - } = {0.60}$：


$$
{n}_{\mathrm{{obs}}} \leftarrow  \left\{  \begin{array}{ll} \min \left( {{n}_{\mathrm{{obs}}} + 2,{70}}\right) & {r}_{s} > {\zeta }_{s}^{ + }, \\  \max \left( {{n}_{\mathrm{{obs}}} - 1,{25}}\right) & {r}_{s} < {\zeta }_{s}^{ - }, \\  {n}_{\mathrm{{obs}}} & \text{ otherwise. } \end{array}\right. \tag{13}
$$



The normalized progress fraction $\wp  = \left( {{n}_{\text{ obs }} - {25}}\right) /({70} -$ 25) is calculated and used to scale all non-terminal reward terms by $K\left( \wp \right)  = 1 + 2\wp  \in  \left\lbrack  {1,3}\right\rbrack$ ,amplifying training signal as task difficulty increases.
归一化进展分数 $\wp  = \left( {{n}_{\text{ obs }} - {25}}\right) /({70} -$ 25) 被计算并用于按 $K\left( \wp \right)  = 1 + 2\wp  \in  \left\lbrack  {1,3}\right\rbrack$ 缩放所有非终止奖励项，随着任务难度增加而放大训练信号。


Reward Function The reward function is designed to encourage the robot to navigate efficiently to the goal while maintaining safe separation from obstacles and smooth control behavior. It contains three groups of terms: (i) goal-directed terms that reward proximity and velocity alignment toward the goal, (ii) stabilization terms that encourage low velocity, correct heading, and low angular rate when in the vicinity of the goal, and (iii) penalty terms that discourage excessive speed, large control increments, and proximity to obstacles as captured by the TTC metric.
奖励函数奖励函数旨在鼓励机器人高效导航至目标，同时与障碍物保持安全间距，并实现平滑的控制行为。它包含三组项：（i）面向目标的项，用于奖励朝向目标的接近程度与速度一致性；（ii）稳定化项：在靠近目标时，鼓励低速度、正确朝向以及低角速度；（iii）惩罚项：根据 TTC 指标，抑制过高速度、过大的控制增量以及靠近障碍物。


Let the speed $v = \begin{Vmatrix}{\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{B}}\end{Vmatrix},\;{\mathbf{u}}_{t - 1}^{acc} \in  {\mathbb{R}}^{4}$ the previous command; ${\tau }_{\min } \in  \left\lbrack  {0,{10}}\right\rbrack$ s the minimum time-to-collision across all rays; and $\wp  \in  \left\lbrack  {0,1}\right\rbrack$ the curriculum progress fraction. Two kernel functions compose all reward and penalty terms:
令速度 $v = \begin{Vmatrix}{\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{B}}\end{Vmatrix},\;{\mathbf{u}}_{t - 1}^{acc} \in  {\mathbb{R}}^{4}$ 为上一条指令；${\tau }_{\min } \in  \left\lbrack  {0,{10}}\right\rbrack$ 为所有射线上的最小碰撞时间；$\wp  \in  \left\lbrack  {0,1}\right\rbrack$ 为课程进展分数。所有奖励与惩罚项由两个核函数组成：


$$
\mathfrak{R}\left( {\mathfrak{m},\mathfrak{a},\mathfrak{v}}\right)  = \mathfrak{m}\exp \left( {-\mathfrak{a}{\mathfrak{v}}^{2}}\right) , \tag{14}
$$



$$
\mathfrak{P}\left( {\mathfrak{m},\mathfrak{a},\mathfrak{v}}\right)  = \mathfrak{m}\left( {\exp \left( {-\mathfrak{a}{\mathfrak{v}}^{2}}\right)  - 1}\right) . \tag{15}
$$



The following intermediate quantities are defined for compactness and summarized in Table 3, while all reward terms are presented in Table 4.
为简洁起见，以下中间量被定义，并在表 3 中汇总，而所有奖励项在表 4 中给出。


$$
{r}_{t} = \left\{  \begin{matrix}  - {10} & \text{ collision, } \\  K\left( \wp \right) \left( {{r}_{\text{ pos }} + {r}_{\text{ prox }} + {r}_{\text{ vel }}}\right. & \\   + {r}_{\text{ lin }} + {r}_{\text{ stab }} + {P}_{\text{ spd }} & \text{ otherwise. } \\  \left. {+{P}_{+x} + {P}_{\text{ ctrl }} + {P}_{\text{ TTC }}}\right) &  \end{matrix}\right. \tag{16}
$$



Table 3. Intermediate quantities used in the reward function.
表 3. 奖励函数中使用的中间量。


<table><tr><td>Symbol</td><td>Definition</td><td>Description</td></tr><tr><td>$w\left( {\psi }_{e}\right)$</td><td>$\mathfrak{R}\left( {1,2,{\psi }_{e}}\right)$</td><td>Heading gate</td></tr><tr><td>$S$</td><td>$\mathfrak{R}\left( {2,2,v - 2}\right)$</td><td>Speed-saturation gate</td></tr><tr><td>$\xi$</td><td>${\mathbf{v}}_{\mathrm{{WV}}}^{\mathrm{V}} \cdot  {\mathbf{\delta }}^{\mathrm{V}}/\left( {v \cdot  \delta }\right)$</td><td>Cosine alignment</td></tr><tr><td>$w\left( \delta \right)$</td><td>$1 - \mathfrak{R}\left( {1,2,\delta }\right)$</td><td>$\approx  0$ near, $\approx  1\mathrm{{far}}$</td></tr><tr><td>$\Delta \mathbf{u}$</td><td>${\mathbf{u}}_{t}^{acc} - {\mathbf{u}}_{t - 1}^{acc}$</td><td>Control increment</td></tr></table>
<table><tbody><tr><td>符号</td><td>定义</td><td>描述</td></tr><tr><td>$w\left( {\psi }_{e}\right)$</td><td>$\mathfrak{R}\left( {1,2,{\psi }_{e}}\right)$</td><td>航向门</td></tr><tr><td>$S$</td><td>$\mathfrak{R}\left( {2,2,v - 2}\right)$</td><td>速度饱和门</td></tr><tr><td>$\xi$</td><td>${\mathbf{v}}_{\mathrm{{WV}}}^{\mathrm{V}} \cdot  {\mathbf{\delta }}^{\mathrm{V}}/\left( {v \cdot  \delta }\right)$</td><td>余弦对齐</td></tr><tr><td>$w\left( \delta \right)$</td><td>$1 - \mathfrak{R}\left( {1,2,\delta }\right)$</td><td>$\approx  0$ 近，$\approx  1\mathrm{{far}}$</td></tr><tr><td>$\Delta \mathbf{u}$</td><td>${\mathbf{u}}_{t}^{acc} - {\mathbf{u}}_{t - 1}^{acc}$</td><td>控制增量</td></tr></tbody></table>


The total reward at each timestep takes the form:
每个时间步的总奖励形式为：


Table 4. Reward terms.
表 4. 奖励项。


<table><tr><td>Term</td><td>Formula</td></tr><tr><td>${r}_{\mathrm{{pos}}}$</td><td>$\mathfrak{R}\left( {3,1,\delta }\right)$</td></tr><tr><td>${r}_{\mathrm{{prox}}}$</td><td>$\mathfrak{R}\left( {5,8,\delta }\right)  \cdot  w\left( {\psi }_{e}\right)$</td></tr><tr><td>${r}_{\mathrm{{lin}}}$</td><td>$\left( {{20} - \delta }\right) /{20}$</td></tr><tr><td>${r}_{\mathrm{{vel}}}$</td><td>$\left\lbrack  {{\xi s}\text{ if }\xi  > 0\text{ ,else } - {0.2}}\right\rbrack   \cdot  \min \left( {\delta /3,1}\right)$</td></tr><tr><td>${r}_{\mathrm{{spd}}}$</td><td>$\Re \left( {{1.5},{10},v}\right)  + \Re \left( {{1.5},{0.5},v}\right)$</td></tr><tr><td>${r}_{\mathrm{{hdg}}}$</td><td>$\Re \left( {2,{0.2},{\psi }_{e}}\right)  + \Re \left( {4,{15},{\psi }_{e}}\right)$</td></tr><tr><td>${r}_{\omega }$</td><td>$\Re \left( {{1.5},5,{\mathbf{\omega }}_{\mathrm{{WB}}z}^{\mathrm{B}}}\right)  \cdot  w\left( {\psi }_{e}\right)$</td></tr><tr><td>${r}_{\text{ stab }}$</td><td>$\left( {{r}_{\mathrm{{spd}}} + {r}_{\mathrm{{hdg}}} + {r}_{\omega }}\right)  \cdot  {\mathbf{1}}_{\delta  < 1\mathrm{\;m}}$</td></tr><tr><td>${P}_{\mathrm{{spd}}}$</td><td>$\mathfrak{P}\left( {2,2,\max \left( {v - 3,0}\right) }\right)$</td></tr><tr><td>${P}_{+x}$</td><td>$\mathfrak{P}\left( {2,8,\max \left( {{\mathbf{v}}_{\mathrm{{WV}}x}^{\mathrm{V}},0}\right) }\right)  \cdot  w\left( \delta \right)$</td></tr><tr><td>${P}_{\Delta \mathbf{u}}$</td><td>$\mathop{\sum }\limits_{{i \in  \{ x,y,z,\dot{\psi }\} }}\mathfrak{P}\left( {{0.3},5,\Delta {u}_{n}^{i}}\right)$</td></tr><tr><td>${P}_{\left| \mathbf{u}\right| }$</td><td>$\mathfrak{P}\left( {{0.1},{0.3},{u}_{n}^{x}}\right)  + \mathfrak{P}\left( {{0.1},{0.3},{u}_{n}^{y}}\right)$ <br> $+ \mathfrak{P}\left( {{0.15},1,{u}_{n}^{z}}\right)  + \mathfrak{P}\left( {{0.15},2,{u}_{n}^{\psi }}\right)$</td></tr><tr><td>${P}_{\text{ ctrl }}$</td><td>${P}_{\Delta \mathbf{u}} + {P}_{\left| \mathbf{u}\right| }$</td></tr><tr><td>${P}_{\mathrm{{TTC}}}$</td><td>$\mathfrak{R}\left( {-3,2,{\tau }_{min}^{2}}\right)$</td></tr></table>
<table><tbody><tr><td>术语</td><td>公式</td></tr><tr><td>${r}_{\mathrm{{pos}}}$</td><td>$\mathfrak{R}\left( {3,1,\delta }\right)$</td></tr><tr><td>${r}_{\mathrm{{prox}}}$</td><td>$\mathfrak{R}\left( {5,8,\delta }\right)  \cdot  w\left( {\psi }_{e}\right)$</td></tr><tr><td>${r}_{\mathrm{{lin}}}$</td><td>$\left( {{20} - \delta }\right) /{20}$</td></tr><tr><td>${r}_{\mathrm{{vel}}}$</td><td>$\left\lbrack  {{\xi s}\text{ if }\xi  > 0\text{ ,else } - {0.2}}\right\rbrack   \cdot  \min \left( {\delta /3,1}\right)$</td></tr><tr><td>${r}_{\mathrm{{spd}}}$</td><td>$\Re \left( {{1.5},{10},v}\right)  + \Re \left( {{1.5},{0.5},v}\right)$</td></tr><tr><td>${r}_{\mathrm{{hdg}}}$</td><td>$\Re \left( {2,{0.2},{\psi }_{e}}\right)  + \Re \left( {4,{15},{\psi }_{e}}\right)$</td></tr><tr><td>${r}_{\omega }$</td><td>$\Re \left( {{1.5},5,{\mathbf{\omega }}_{\mathrm{{WB}}z}^{\mathrm{B}}}\right)  \cdot  w\left( {\psi }_{e}\right)$</td></tr><tr><td>${r}_{\text{ stab }}$</td><td>$\left( {{r}_{\mathrm{{spd}}} + {r}_{\mathrm{{hdg}}} + {r}_{\omega }}\right)  \cdot  {\mathbf{1}}_{\delta  < 1\mathrm{\;m}}$</td></tr><tr><td>${P}_{\mathrm{{spd}}}$</td><td>$\mathfrak{P}\left( {2,2,\max \left( {v - 3,0}\right) }\right)$</td></tr><tr><td>${P}_{+x}$</td><td>$\mathfrak{P}\left( {2,8,\max \left( {{\mathbf{v}}_{\mathrm{{WV}}x}^{\mathrm{V}},0}\right) }\right)  \cdot  w\left( \delta \right)$</td></tr><tr><td>${P}_{\Delta \mathbf{u}}$</td><td>$\mathop{\sum }\limits_{{i \in  \{ x,y,z,\dot{\psi }\} }}\mathfrak{P}\left( {{0.3},5,\Delta {u}_{n}^{i}}\right)$</td></tr><tr><td>${P}_{\left| \mathbf{u}\right| }$</td><td>$\mathfrak{P}\left( {{0.1},{0.3},{u}_{n}^{x}}\right)  + \mathfrak{P}\left( {{0.1},{0.3},{u}_{n}^{y}}\right)$ <br/> $+ \mathfrak{P}\left( {{0.15},1,{u}_{n}^{z}}\right)  + \mathfrak{P}\left( {{0.15},2,{u}_{n}^{\psi }}\right)$</td></tr><tr><td>${P}_{\text{ ctrl }}$</td><td>${P}_{\Delta \mathbf{u}} + {P}_{\left| \mathbf{u}\right| }$</td></tr><tr><td>${P}_{\mathrm{{TTC}}}$</td><td>$\mathfrak{R}\left( {-3,2,{\tau }_{min}^{2}}\right)$</td></tr></tbody></table>


3.4.3 Composite CBF-based Safety Filter Beyond the aforementioned navigation approaches -which combine map-based safety of the planning module with reactive collision avoidance control- the UAstack further provides a last-resort safety filter. The rationale for adding this final layer is twofold. On the one hand, both the Neural NMPC through Signed Distance Field Encoding for Collision Avoidance (SDF-NMPC) and the DRL navigation strategies involve deep neural network processing, which, despite training to consider noise and other imperfections, is treated as a source of possible (albeit unlikely) error. On the other hand, using fundamentally different collision-checking at different spatiotemporal scales -spanning map-based planning, the navigation strategies, and the safety filter-offers resourcefulness. It thus reflects a conservative but meaningful choice to safeguard the robot from a collision which represents one of the most problematic events during a mission.
3.4.3 基于复合CBF的安全过滤器：超越上述导航方法——这些方法将规划模块的基于地图的安全性与反应式避碰控制相结合——UAstack 进一步提供了最后手段的安全过滤器。引入这一最终层的理由有两方面。一方面，借助用于避碰的符号距离场编码（SDF-NMPC）的神经 NMPC，以及 DRL 导航策略，都会涉及深度神经网络处理；尽管训练时会考虑噪声及其他不完美因素，它仍被视为可能（尽管不太可能）产生错误的来源。另一方面，在不同时空尺度上采用本质不同的碰撞检测——覆盖基于地图的规划、导航策略以及安全过滤器——体现出这种方法的灵活性。因此，这是一种保守但有意义的选择，用于保护机器人免于碰撞，而碰撞是任务过程中最棘手的事件之一。


Based on Composite Control Barrier Functions (C-CBFs) formalism for safe navigation Harms et al. (2025), we compose a C-CBF directly from recent range measurements as in Misyats et al. (2025) to modify the acceleration setpoint when an unexpected impending collision is detected. It is a key module to fully and formally safeguard autonomous robots. The C-CBF is described hereafter considering the case of a flying robot. First, the system model is approximated to be of degree 2 and takes the form:
基于用于安全导航的复合控制屏障函数（C-CBF）形式（Harms et al., 2025），我们像 Misyats et al.（2025）那样，直接从近期的距离测量中构建一个 C-CBF；一旦检测到意外迫近的碰撞，就用它来修正加速度设定值。这是为全面且形式化地保障自主机器人安全的关键模块。下文在考虑飞行机器人的情况下对 C-CBF 进行描述。首先，系统模型被近似为二阶，并写为：


$$
\mathbf{x} = \left\lbrack  \begin{array}{l} {\mathbf{p}}_{\mathrm{{WB}}}^{\mathrm{W}} \\  {\mathbf{v}}_{\mathrm{{WB}}}^{\mathrm{W}} \end{array}\right\rbrack  ,\;\mathbf{u} = \left\lbrack  {\mathbf{a}}^{\mathrm{W}}\right\rbrack  . \tag{17}
$$



For a rotorcraft such as a multirotor aerial robot, we here use the simplified linear system model
对于如多旋翼无人机这类旋翼飞行器，我们在此采用简化的线性系统模型


$$
\dot{\mathbf{x}} = \underset{f\left( \mathbf{x}\right) }{\underbrace{\left\lbrack  \begin{array}{ll} {\mathbf{0}}_{3} & {\mathbf{I}}_{3} \\  {\mathbf{0}}_{3} & {\mathbf{0}}_{3} \end{array}\right\rbrack  }}\mathbf{x} + \underset{g\left( \mathbf{x}\right) }{\underbrace{\left\lbrack  \begin{array}{l} {\mathbf{0}}_{3} \\  {\mathbf{I}}_{3} \end{array}\right\rbrack  }}\mathbf{u}, \tag{18}
$$



<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_13.jpg?x=833&y=162&w=671&h=410&r=0"/>



Figure 7. Visualization of the used kappa function with nominal values $\lambda  = 1,p = {0.5},\sigma  = 1$ .
图 7：在标称值 $\lambda  = 1,p = {0.5},\sigma  = 1$ 下，所用 kappa 函数的可视化。


where ${\mathbf{a}}_{\mathrm{{WB}}}^{\mathrm{W}}$ is the linear acceleration. Here $f\left( \mathbf{x}\right)$ and $g\left( \mathbf{x}\right) \mathbf{u}$ denote the state system dynamics vectors.
其中 ${\mathbf{a}}_{\mathrm{{WB}}}^{\mathrm{W}}$ 是线性加速度。此处 $f\left( \mathbf{x}\right)$ 与 $g\left( \mathbf{x}\right) \mathbf{u}$ 分别表示状态系统动力学向量。


Considering the above, the method then represents the "free-space" set as ${\mathcal{C}}_{\mathbf{x}} = \mathop{\bigcap }\limits_{{i = 1}}^{N}\left\{  {\mathbf{x} : {\begin{Vmatrix}{\mathbf{p}}_{\mathrm{{WB}}}^{\mathrm{w}} - {\mathbf{p}}_{{\mathrm{{WB}}}_{i}}^{\mathrm{w}}\end{Vmatrix}}^{2} \geq  \varepsilon }\right\}$ with $\varepsilon  > 0$ representing safety radius around each obstacle point ${\mathbf{p}}_{\mathrm{W}{0}_{i}}^{\mathrm{W}}$ . A condensed set of $N$ points is obtained by means of the most recent, sparsified depth observation. This is then associated with equivalent scalar "distance-squared" functions ${\nu }_{i,0}\left( \mathbf{x}\right)  \mathrel{\text{ := }} {\begin{Vmatrix}{\mathbf{p}}_{\mathrm{{WB}}}^{\mathrm{W}} - {\mathbf{p}}_{{\mathrm{{WI}}}_{i}}^{\mathrm{W}}\end{Vmatrix}}^{2} - {\varepsilon }^{2}$ with the safety for obstacle $i$ being ${\nu }_{i,0} \geq  0$ . Note that there is no dependence on any consistent world frame as only the relative distances are used in the construction. Subsequently, we define higher-order CBF (HO-CBF) functions as:
考虑上述内容，该方法将“自由空间”集合表示为 ${\mathcal{C}}_{\mathbf{x}} = \mathop{\bigcap }\limits_{{i = 1}}^{N}\left\{  {\mathbf{x} : {\begin{Vmatrix}{\mathbf{p}}_{\mathrm{{WB}}}^{\mathrm{w}} - {\mathbf{p}}_{{\mathrm{{WB}}}_{i}}^{\mathrm{w}}\end{Vmatrix}}^{2} \geq  \varepsilon }\right\}$，其中 $\varepsilon  > 0$ 表示围绕每个障碍点 ${\mathbf{p}}_{\mathrm{W}{0}_{i}}^{\mathrm{W}}$ 的安全半径。通过最近一次的、经过稀疏化的深度观测，可以得到一个由 $N$ 点组成的压缩集合。随后，将其关联到等价的标量“距离平方”函数 ${\nu }_{i,0}\left( \mathbf{x}\right)  \mathrel{\text{ := }} {\begin{Vmatrix}{\mathbf{p}}_{\mathrm{{WB}}}^{\mathrm{W}} - {\mathbf{p}}_{{\mathrm{{WI}}}_{i}}^{\mathrm{W}}\end{Vmatrix}}^{2} - {\varepsilon }^{2}$，其中对障碍物 $i$ 的安全性为 ${\nu }_{i,0} \geq  0$ 。注意，在构建中仅使用相对距离，因此不存在对任何一致世界坐标系的依赖。接着，我们定义更高阶的 CBF（HO-CBF）函数为：


$$
{\nu }_{i} \mathrel{\text{ := }} {\mathfrak{L}}_{f}{\nu }_{i,0} - \varsigma \left( {\nu }_{i,0}\right) , \tag{19}
$$



where ${\mathfrak{L}}_{f}$ is the Lie derivative along the drift dynamics and $\varsigma$ is a tunable class ${\mathcal{K}}_{\infty }$ function of the form
其中 ${\mathfrak{L}}_{f}$ 是沿漂移动力学的李导数，$\varsigma$ 是一个可调的函数类别 ${\mathcal{K}}_{\infty }$，其形式为


$$
\varsigma \left( h\right)  = \lambda  \cdot  h{\left( {h}^{2} + {\sigma }^{2}\right) }^{\left( {p - 1}\right) /2}. \tag{20}
$$



The used function $\varsigma$ with parameters $\lambda  > 0,\sigma  > 0,p > 0$ is displayed in Fig. 7. For $p < 1$ the tuneable parameters allow for a more rapid convergence behavior than a for $p = 1$ .
所使用的函数 $\varsigma$（参数为 $\lambda  > 0,\sigma  > 0,p > 0$）在图 7 中展示。对于 $p < 1$，可调参数使得其收敛速度比 $p = 1$ 更快。


We formulate the C-CBF as
我们将 C-CBF 表示为


$$
h\left( \mathbf{x}\right)  \mathrel{\text{ := }}  - \frac{\gamma }{\kappa }\log \left( {\mathop{\sum }\limits_{{i = 1}}^{N}{e}^{-\kappa \tanh \left( {{\nu }_{i}\left( \mathbf{x}\right) /\gamma }\right) }}\right) , \tag{21}
$$



with saturation parameter $\gamma$ and temperature parameter $\kappa$ .
其中饱和参数为 $\gamma$，温度参数为 $\kappa$ 。


To enforce invariance of the safe set ${\mathcal{X}}_{\text{ safe }} = \{ \mathbf{x} \mid  h\left( \mathbf{x}\right)  \geq$ 0 \}, the condition for a Robust Control Barrier Function (R-CBF) Nanayakkara et al. (2025) reads
为保证安全集合 ${\mathcal{X}}_{\text{ safe }} = \{ \mathbf{x} \mid  h\left( \mathbf{x}\right)  \geq$ 0 \} 的不变性，根据鲁棒控制屏障函数（R-CBF）（Nanayakkara et al., 2025），条件为


$$
{\mathfrak{L}}_{g}h\left( \mathbf{x}\right) u \geq  \vartheta \left( \mathbf{x}\right) , \tag{22}
$$



$$
\vartheta \left( \mathbf{x}\right)  \mathrel{\text{ := }}  - {\mathfrak{L}}_{f}h\left( \mathbf{x}\right)  - {\alpha h}\left( \mathbf{x}\right)  + \rho \left( {\begin{Vmatrix}{{\mathfrak{L}}_{g}h\left( \mathbf{x}\right) }\end{Vmatrix})}\right) \tag{23}
$$



for a scalar $\alpha  \in  {\mathbb{R}}^{ + }$ and a robustness function $\rho \left( y\right)  = {\rho }_{1}y + \; {\rho }_{2}{y}^{2}$ . For a valid C-CBF $h$ ,the satisfaction of condition (22) is sufficient to ensure forward invariance of the safe set ${\mathcal{X}}_{\text{ safe }}$ , even for inputs with unknown,additive disturbances $\mathbf{d}$ added to the undisturbed input ${\mathbf{u}}_{\mathrm{n}}$ ,e.g. $\mathbf{u} = {\mathbf{u}}_{\mathrm{n}} + \mathbf{d}$ ,accounting for tracking errors satisfying $\parallel \mathbf{d}\parallel  \leq  \mathop{\inf }\limits_{{y \in  \mathbb{R} \geq  0}}\frac{\rho \left( y\right) }{y}$ in the low-level controllers. This scheme can thus be used to certify collision-free navigation in spite of imperfect tracking. We enforce the condition (22) for a nominal input ${\mathbf{u}}_{\mathrm{{sp}}}$ by means of a reactive safety filter Quadratic Programming (QP) of the form
针对标量 $\alpha  \in  {\mathbb{R}}^{ + }$ 和鲁棒性函数 $\rho \left( y\right)  = {\rho }_{1}y + \; {\rho }_{2}{y}^{2}$。对于有效的 C-CBF $h$，条件 (22) 的满足足以保证安全集 ${\mathcal{X}}_{\text{ safe }}$ 的前向不变性，即使输入中加入了未知的加性扰动 $\mathbf{d}$，该扰动叠加于无扰动输入 ${\mathbf{u}}_{\mathrm{n}}$ 上，例如 $\mathbf{u} = {\mathbf{u}}_{\mathrm{n}} + \mathbf{d}$，并考虑到低层控制器中满足 $\parallel \mathbf{d}\parallel  \leq  \mathop{\inf }\limits_{{y \in  \mathbb{R} \geq  0}}\frac{\rho \left( y\right) }{y}$ 的跟踪误差。因此，该方案可用于在跟踪不完美的情况下保证无碰撞导航。我们通过如下形式的反应式安全滤波二次规划（QP）对名义输入 ${\mathbf{u}}_{\mathrm{{sp}}}$ 强制满足条件 (22)


$$
{\mathbf{u}}^{ * } = \underset{\mathbf{u} \in  {\mathbb{R}}^{3}}{\arg \min }{\begin{Vmatrix}\mathbf{u} - {\mathbf{u}}_{\mathrm{{sp}}}\end{Vmatrix}}^{2} \tag{24}
$$



$$
\text{ s.t. }{\mathfrak{L}}_{g}h\left( \mathbf{x}\right) \mathbf{u} \geq  \vartheta \left( \mathbf{x}\right) \text{ . }
$$



For the case of one single constraint, an analytical solution to (24) can be computed Alan et al. (2023) as
对于单一约束的情况，方程 (24) 的解析解可由 Alan et al. (2023) 计算为


$$
{\mathbf{u}}^{ * } = {\mathbf{u}}_{\mathrm{{sp}}} + \max \left( {0,\eta \left( \mathbf{x}\right) }\right) {\mathfrak{L}}_{g}h{\left( \mathbf{x}\right) }^{\top }, \tag{25}
$$



where
其中


$$
\eta \left( \mathbf{x}\right)  = \left\{  \begin{array}{ll}  - \frac{{\mathfrak{L}}_{g}h\left( \mathbf{x}\right) {\mathbf{u}}_{\mathrm{{sp}}} - \vartheta \left( \mathbf{x}\right) }{{\begin{Vmatrix}{\mathfrak{L}}_{g}h\left( \mathbf{x}\right) \end{Vmatrix}}^{2}} & \text{ if }{\mathfrak{L}}_{g}h\left( \mathbf{x}\right)  \neq  \mathbf{0} \\  0 & \text{ if }{\mathfrak{L}}_{g}h\left( \mathbf{x}\right)  = \mathbf{0}. \end{array}\right. \tag{26}
$$



This computationally cheap and flexible scheme to reactively enforce collision avoidance of any higher level control policy. To further reduce chattering of the safety filter, we further apply Exponential Moving Average (EMA) filtering to the output.
这种计算代价低且灵活的方案可用于对任意更高层控制策略反应式地强制避免碰撞。为进一步减少安全滤波器的抖振，我们还对输出应用指数移动平均（EMA）滤波。


3.4.4 Ablation Studies A set of evaluation studies are conducted using a simulated quadrotor with mass ${2.10}\mathrm{\;{kg}}$ and dimensions ${0.3} \times  {0.3} \times  {0.1}\mathrm{\;m}$ in Gazebo Koenig and Howard (2004). The robot is equipped with an acceleration and yaw-rate tracking controller and a hemispherical dome LiDAR sensor modeled after the RoboSense Airy (FoV ${180}^{ \circ  } \times  {90}^{ \circ  }$ ). The performance of the ExRL and the SDF-NMPC methods is compared both with and without the C-CBF-based safety filter, across environments of varying density, and under different levels of command mismatch induced by an independent first-order low-pass filter with time constant ${\tau }_{d}$ on each acceleration and yaw-rate command dimension. The slower dynamics introduced by the low-pass filter serve to practically emulate the effects of the closed-loop attitude response of the system, which presents non-instantaneous reference tracking and at times imperfect disturbance rejection. This is of particular interest especially as the actual time constant of the attitude subsystem differs between robots and delayed response poses a challenge to navigation in tight spaces. This allows us to characterize the failure modes of the navigation policies induced by increasingly disturbed system behavior.
3.4.4 消融研究 使用 Gazebo Koenig and Howard (2004) 中一架质量为 ${2.10}\mathrm{\;{kg}}$、尺寸为 ${0.3} \times  {0.3} \times  {0.1}\mathrm{\;m}$ 的仿真四旋翼开展了一组评估实验。机器人配备了加速度和偏航角速度跟踪控制器，以及一个按照 RoboSense Airy（FoV ${180}^{ \circ  } \times  {90}^{ \circ  }$）建模的半球形穹顶 LiDAR 传感器。比较了 ExRL 和 SDF-NMPC 方法在有无基于 C-CBF 的安全滤波器、不同环境密度以及不同指令失配水平下的性能；这种失配由作用于每个加速度和偏航角速度指令维度、时间常数为 ${\tau }_{d}$ 的独立一阶低通滤波器引入。低通滤波器带来的较慢动力学在实践中模拟了系统闭环姿态响应的影响，其表现为非即时参考跟踪以及有时不完美的扰动抑制。这一点尤为重要，因为不同机器人的姿态子系统实际时间常数不同，而延迟响应会给狭窄空间中的导航带来挑战。这使我们能够刻画由日益受扰的系统行为所诱发的导航策略失效模式。


Test environments, illustrated in Figure 8a, are procedurally generated as rectangular corridors of width and height $8\mathrm{\;m}$ each,populated with spherical obstacles of $1\mathrm{\;m}$ diameter placed using Poisson-disc sampling, guaranteeing a parametric minimum separation ${r}_{\text{ sep }} \in  \{ {1.5},{1.8},{2.0},{2.5},{3.0}\} \mathrm{m}$ between obstacle centers, i.e., surface-to-surface gaps of $\{ {0.5},{0.8},{1.0},{1.5},{2.0}\} \mathrm{m}$ . The corridor is bounded above, below and on the sides by walls, leaving the longitudinal direction open for traversal. The robot is initialized at the start of the corridor and tasked with following a path through the corridor to the other side.
测试环境如图 8a 所示，按程序生成：由宽度和高度均为 $8\mathrm{\;m}$ 的矩形走廊构成，内部填充使用泊松圆盘采样放置的直径为 $1\mathrm{\;m}$ 的球形障碍物，从而保证障碍物中心之间具有参数化的最小间距 ${r}_{\text{ sep }} \in  \{ {1.5},{1.8},{2.0},{2.5},{3.0}\} \mathrm{m}$，即表面到表面的间隙为 $\{ {0.5},{0.8},{1.0},{1.5},{2.0}\} \mathrm{m}$。走廊上方、下方及两侧均由墙体围住，纵向方向则保持开放供通行。机器人在走廊起点初始化，并被要求沿着走廊路径前往另一端。


For each environment, and for each value of the low-pass time constant ${\tau }_{d},{20}$ independent runs are performed with each of the four configurations and the outcomes are visualized in Figure 8b. Every run terminates in one of three mutually exclusive states: success, in which the robot reaches the goal; stagnation, in which the robot halts before the goal but remains collision-free; or crash, in which a collision with the environment is detected. The figure reports the rate of each outcome as a function of ${r}_{\text{ sep }}$ ,with rows corresponding to the three outcome categories and columns to the four values of ${\tau }_{d}$ . Smaller ${r}_{\text{ sep }}$ values correspond to denser,more constrained environments,while larger ${\tau }_{d}$ values correspond to a more severe command mismatch. Solid lines indicate the C-CBF-filtered configurations and dashed lines the baselines without this last-resort filter, with colour distinguishing the upstream policy (ExRL in blue, SDF-NMPC in green). Each marker aggregates the 20 runs for the corresponding combination of parameters.
对每个环境，以及每个低通时间常数 ${\tau }_{d},{20}$ 的取值，都会针对四种配置各进行独立运行，并将结果可视化于图 8b 中。每次运行都会以三种互斥状态之一结束：成功，即机器人到达目标；停滞，即机器人在到达目标前停止但未发生碰撞；或崩溃，即检测到与环境发生碰撞。该图给出了各结果的发生率随 ${r}_{\text{ sep }}$ 的变化情况，其中行对应三类结果，列对应 ${\tau }_{d}$ 的四个取值。较小的 ${r}_{\text{ sep }}$ 值对应更稠密、约束更强的环境，而较大的 ${\tau }_{d}$ 值对应更严重的命令失配。实线表示经过 C-CBF 过滤的配置，虚线表示未经过这一最后保险过滤器的基线，颜色区分上游策略（蓝色为 ExRL，绿色为 SDF-NMPC）。每个标记汇总了对应参数组合下的 20 次运行。


Two sets of comparisons are performed. Among the values of ${\tau }_{d}$ tested, ${\tau }_{d} \leq  {0.10}\mathrm{\;s}$ corresponds to the realistic operating regime that the studies are intended to characterize; ${\tau }_{d} = {0.25}\mathrm{\;s}$ is included as a deliberate stress test,beyond realistic deployment conditions, in order to probe the limits of each configuration and expose the regime in which the safety ceases to hold. Because both success and stagnation outcomes correspond to collision-free behaviour, the crash rate is treated as the primary safety metric throughout, while the success rate is reported as a secondary but important measure of task completion.
进行了两组比较。在测试的 ${\tau }_{d}$ 取值中，${\tau }_{d} \leq  {0.10}\mathrm{\;s}$ 对应于研究旨在表征的真实运行区间；${\tau }_{d} = {0.25}\mathrm{\;s}$ 则被有意作为压力测试，超出真实部署条件，以探查各配置的极限并暴露安全失效的区间。由于成功和停滞两种结果都对应无碰撞行为，因此在全文中将崩溃率作为主要安全指标，而成功率则作为次要但重要的任务完成度指标予以报告。


The first comparison contrasts the two unfiltered baselines, ExRL only and SDF-NMPC only, which differ markedly in how they degrade under increasing ${\tau }_{d}$ . At ${\tau }_{d} = 0$ both reach the goal on 80-100% of runs across all densities, with the ExRL policy crashing on $5\%$ of the densest layouts and SDF-NMPC on ${10}\%$ . As ${\tau }_{d}$ grows,the two baselines diverge sharply. SDF-NMPC degrades gracefully and primarily along the density axis ${r}_{\text{ sep }}$ . At ${\tau }_{d} = {0.25}\mathrm{\;s}$ it still succeeds on ${100}\%$ of runs at ${r}_{\text{ sep }} = {3.0}\mathrm{\;m}$ and on ${95}\%$ at ${r}_{\text{ sep }} = {2.5}\mathrm{\;m}$ ,with crashes concentrated in the dense environment setup. ExRL, in contrast, degrades globally: at the same ${\tau }_{d}$ it crashes on 85-100% of runs across the three densest settings and even at ${r}_{\text{ sep }} = {3.0}\mathrm{\;m}$ retains only 65% success. This asymmetry may reflect SDF-NMPC's robustness against the induced mismatch, while the learned policy's reliance on a specific dynamics model employed during training and the subsequent degradation wherever it departs from that model.
第一组比较对比了两个未过滤的基线，即仅 ExRL 和仅 SDF-NMPC，它们在随 ${\tau }_{d}$ 增大时的退化方式明显不同。在 ${\tau }_{d} = 0$ 时，两者在所有密度下的到达目标率均为 80-100%，其中 ExRL 策略在最稠密布局中有 $5\%$ 的运行发生崩溃，SDF-NMPC 则为 ${10}\%$。随着 ${\tau }_{d}$ 增大，两条基线迅速分化。SDF-NMPC 退化平缓，且主要沿密度轴 ${r}_{\text{ sep }}$ 变化。在 ${\tau }_{d} = {0.25}\mathrm{\;s}$ 时，它在 ${100}\%$ 的运行中于 ${r}_{\text{ sep }} = {3.0}\mathrm{\;m}$ 仍能成功，在 ${95}\%$ 时为 ${r}_{\text{ sep }} = {2.5}\mathrm{\;m}$，崩溃主要集中在稠密环境设置中。相比之下，ExRL 的退化是整体性的：在相同的 ${\tau }_{d}$ 下，它在最稠密的三个设置中有 85-100% 的运行发生崩溃，甚至在 ${r}_{\text{ sep }} = {3.0}\mathrm{\;m}$ 时成功率也仅剩 65%。这种不对称性可能反映出 SDF-NMPC 对所引入失配的鲁棒性，而学习策略则依赖于训练时采用的特定动力学模型，并在偏离该模型的情况下迅速退化。


The second comparison contrasts each baseline against its C-CBF-augmented counterpart. At ${\tau }_{d} = 0$ ,the safety filter eliminates collisions entirely and converts results into stagnations rather than goal completions, preserving safety at the cost of task completion. The filter's protective effect persists across the realistic regime. At ${\tau }_{d} = {0.10}\mathrm{\;s}$ , SDF-NMPC+C-CBF crashes on at most ${20}\%$ of runs in any environment, always lower than SDF-NMPC alone, and ExRL+C-CBF crashes on 10-85% in the dense band, again lower than ExRL alone. This indicates a substantial reduction in crash rate with the inclusion of the C-CBF. Under the unrealistic ${\tau }_{d} = {0.25}\mathrm{\;s}$ conditions,the filter loses authority in the densest environments and both augmented pipelines crash on ${85} - {100}\%$ of runs at ${r}_{\text{ sep }} = {1.5}\mathrm{\;m}$ , locating the failure boundary that the test is designed to expose. The augmented versions inherit the characteristics of their upstream policy: ExRL+C-CBF crashes earlier and across a wider density range than SDF-NMPC+C-CBF at every ${\tau }_{d} > 0$ ,indicating that the C-CBF compensates for command distortion but not for the upstream policy's own degradation under that distortion. This analysis shows that imperfect control can occur in demanding environments and conditions, highlighting the need for navigation strategies with multiple layers of safety checking and avoidance. The UAstack offers a configurable methodological plurality to support demanding deployments.
第二组对比将各基线与其加入 C-CBF 后的对应版本进行比较。在 ${\tau }_{d} = 0$ 时，安全过滤器将碰撞完全消除，并把结果从完成目标转为停滞，在牺牲任务完成度的同时保住安全。该过滤器的保护作用在现实区间内持续存在。在 ${\tau }_{d} = {0.10}\mathrm{\;s}$ 时，SDF-NMPC+C-CBF 在任何环境下的失事率最多为 ${20}\%$ ，始终低于仅用 SDF-NMPC；而 ExRL+C-CBF 在稠密区间的失事率为 10-85%，同样低于仅用 ExRL。这表明引入 C-CBF 可显著降低失事率。在不现实的 ${\tau }_{d} = {0.25}\mathrm{\;s}$ 条件下，过滤器在最稠密环境中失去约束力，两个增强管线在 ${85} - {100}\%$ 的运行中于 ${r}_{\text{ sep }} = {1.5}\mathrm{\;m}$ 发生失事，定位出该测试旨在揭示的失效边界。增强版本继承其上游策略的特性：在每个 ${\tau }_{d} > 0$ 下，ExRL+C-CBF 都比 SDF-NMPC+C-CBF 更早失事，且覆盖更宽的密度范围，这表明 C-CBF 能补偿命令失真，但不能补偿上游策略在该失真下自身的退化。此分析表明，在高要求环境和条件下可能出现不完善控制，凸显了需要具有多层安全检查与规避的导航策略。UAstack 提供了可配置的方法多样性，以支持高要求部署。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_15.jpg?x=178&y=155&w=1279&h=460&r=0"/>



(a) Procedurally generated test environments used in the ablation studies regarding the navigation module. Each environment is a rectangular corridor bounded above, below, and on the sides by walls, with the longitudinal direction left open for traversal. Spherical obstacles of 1 m diameter are placed via Poisson-disc sampling with a parametric minimum separation ${r}_{\text{ sep }}$ between obstacle centers,with lower values representing denser environments. The robot dimensions are visualized in each environment.
(a) 导航模块消融研究中使用的程序生成测试环境。每个环境都是一个矩形走廊，上下及两侧由墙围成，纵向方向留空供通行。直径 1 m 的球形障碍物通过 Poisson-disc 采样放置，障碍物中心之间的参数化最小间距为 ${r}_{\text{ sep }}$ ，较低数值表示更稠密的环境。机器人尺寸在每个环境中均有可视化。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_15.jpg?x=179&y=840&w=1289&h=789&r=0"/>



(b) Outcome rates across density (horizontal axis, ${r}_{\text{ sep }}$ ) and command mismatch (columns, ${\tau }_{d}$ ). Rows show success,crash,and stagnation rates over 20 runs for each case. Solid lines: C-CBF-filtered; dashed: unfiltered. Blue: ExRL; green: SDF-NMPC.
(b) 随密度（横轴，${r}_{\text{ sep }}$ ）和命令失配（列，${\tau }_{d}$ ）变化的结果率。各行显示 20 次运行中每种情形下的成功、失事和停滞率。实线：C-CBF 过滤；虚线：未过滤。蓝色：ExRL；绿色：SDF-NMPC。


Figure 8. Ablation study comparing ExRL and SDF-NMPC navigation policies, with and without the C-CBF last-resort safety filter, across obstacle densities (8a) and a sweep of the time constant ${\tau }_{d}$ (8b). ${\tau }_{d} \leq  {0.10}$ s approximately corresponds to the realistic operating regime; ${\tau }_{d} = {0.25}\mathrm{\;s}$ is unrealistic. Since success and stagnation both correspond to collision-free behaviour,the crash rate is the primary safety metric.
图 8. 比较 ExRL 和 SDF-NMPC 导航策略在不同障碍密度（8a）及时间常数 ${\tau }_{d}$ 扫描（8b）下、加入与未加入 C-CBF 最后安全过滤器的消融研究。${\tau }_{d} \leq  {0.10}$ s 大致对应现实运行区间；${\tau }_{d} = {0.25}\mathrm{\;s}$ 则不现实。由于成功和停滞都对应无碰撞行为，失事率是主要安全指标。


3.4.5 Exteroceptive Overwrite When the map-based safety of planning module or the specific exteroceptive navigation methods are not desirable for, or compatible with, a given robot, the UAstack allows the use of a conventional state-feedback controller. Examples include the Linear MPC methods in Greeff and Schoellig (2018); Kamel et al. (2017), the underlying MPC in SDF-NMPC with disabled collision constraints, or a solution for any other particular robot or through existing autopilot (e.g., PX4).
3.4.5 外感知覆盖 当基于地图的规划模块安全机制或特定的外感知导航方法对某一机器人并不理想，或与其不兼容时，UAstack 允许使用传统的状态反馈控制器。示例包括 Greeff and Schoellig (2018); Kamel et al. (2017) 中的线性 MPC 方法、在禁用碰撞约束时的 SDF-NMPC 底层 MPC，或针对其他特定机器人的方案，亦或通过现有自动驾驶仪（例如 PX4）实现的方案。


### 3.5 Low-level Interfaces
### 3.5 底层接口


The UAstack interfaces diverse aerial and ground robots as follows. Although extendable in principle, the currently provided open-source interfaces are as follows:
UAstack 以如下方式为多样的空中与地面机器人提供接口。原则上可扩展，但当前提供的开源接口如下：


Aerial Robots - Full Stack: When the full stack with multi-layered safety is considered for flying systems, we command acceleration setpoints directly to compatible autopilots and low-level controllers such as PX4- and ArduPilot-based flight controllers.
空中机器人 - 完整栈：当针对飞行系统考虑带多层安全的完整栈时，我们将加速度设定值直接下发给兼容的飞控与底层控制器，例如基于 PX4 和 ArduPilot 的飞行控制器。


Aerial Robots - Without Multi-layered Safety: If the multi-layered safe navigation is not necessary for flying systems, we provide options for commanding waypoints or 3D accelerations to existing controllers. Waypoints are straightforward for all systems that offer such control. However, this does not deliver the full stack functionality and collision avoidance is ensured only at the map/planning level.
空中机器人 - 无多层安全：如果飞行系统不需要多层安全导航，我们提供对现有控制器下发航点或三维加速度的选项。只要系统具备这种控制能力，航点就很直接。然而，这并未提供完整栈功能，碰撞规避仅在地图/规划层面得到保证。


Ground Robots: For legged systems and other ground robots, we interface the planning module with a custom PID tracking controller to output velocity commands to follow the path given by the planning module. Velocity-setpoint commands are provided to the platforms for them to track. Those velocity references may be passed through other safety-mechanisms in case those are provided onboard the platforms.
地面机器人：对于腿式系统及其他地面机器人，我们将规划模块与自定义 PID 跟踪控制器对接，以输出速度指令，来跟随规划模块给定的路径。将速度设定值下发给平台以便其跟踪。这些速度参考在平台板载提供相应安全机制时，可再通过其他安全机制传递。


Direct support for widely-adopted low-level control or autopilot interfaces is available out-of-the-box through MAVROS, or more broadly through the ROS framework. As the stack is structured in a Dockerized format, we support both ROS 1 and ROS 2-based robots out of the box.
通过 MAVROS 可开箱即用地直接支持广泛采用的底层控制或飞控接口；更广义地，也可通过 ROS 框架支持。由于该栈以 Docker 化方式组织，我们开箱即用地同时支持基于 ROS 1 与 ROS 2 的机器人。


## 4 Evaluation Studies
## 4 评估研究


To validate the UAstack, a comprehensive set of evaluation studies was conducted including (a) evaluation of the performance, accuracy and overall resilience of the perception module in diverse environments and conditions, (b) evaluation of the performance and safety-inducing behaviors of the navigation module, as well as (c) full-stack results requiring the orchestrated operation of the perception, planning and navigation modules. Studies were conducted using both aerial and ground robots, alongside some handheld experiments (as part of the evaluation regime of the perception module). An overview of these experiments is shown in Table 5 and key parameters are listed in Table 6. All presented field experiments are included in the video files of the submission and can be found at https://youtube.com/playlist?list= PLu70ME0whad9KCs5PHx-35-qpyK14yZYi&si= E-QJCTTyfhfFUqFo.
为验证 UAstack，开展了一套全面的评估研究，包括（a）在多样环境与条件下，对感知模块的性能、准确性及整体鲁棒性进行评估，（b）对导航模块的性能及引发安全的行为进行评估，以及（c）需要感知、规划和导航模块协同运行的全栈结果。研究同时使用了空中与地面机器人，并配合了一些手持实验（作为感知模块评估方案的一部分）。这些实验的概览见表 5，关键参数列于表 6。所展示的所有实地实验均包含在提交的影像文件中，可在 https://youtube.com/playlist?list= PLu70ME0whad9KCs5PHx-35-qpyK14yZYi&si= E-QJCTTyfhfFUqFo 查看。


### 4.1 Verified Robot Morphologies
### 4.1 已验证的机器人形态


In this paper, we evaluate the UAstack on two multirotor robot configurations and a legged robot. In all robot missions, the UAstack runs fully onboard in real-time, with module configurations as detailed in Table 5. The released open-source code involves simulation examples with additional morphologies such as ground rovers and helicopters. Verifying on diverse robots - while further expanding the morphologies we support - is aligned with our strategic goal to provide a generalist autonomy stack across broad morphological categories.
本文中，我们在两种多旋翼机器人配置和一种足式机器人上评估了 UAstack。在所有机器人任务中，UAstack 均完全在机载端实时运行，模块配置详见表 5。发布的开源代码还包含仿真示例，涵盖地面车和直升机等其他形态。在多样化机器人上进行验证——同时进一步扩展我们支持的形态——与我们的战略目标一致，即为更广泛的形态类别提供通用型自主栈。


#### 4.1.1 Multirotors
#### 4.1.1 多旋翼


Aerial Robot 1 (AR-1): The first aerial robot, referred to as AR-1, is an improved version of the RMF-Owl Petris et al. (2022). AR-1 integrates a Khadas Vim4 computer and a sensing suite comprising of an Ouster OS0-128 LiDAR (10Hz), a Flir Blackfly S 0.4 MP color camera (20 Hz normally, 25 Hz in Fyllingsdal), a Texas Instruments mmWave IWR6843AOP radar sensor (10 Hz normally, ${25}\mathrm{\;{Hz}}$ in Fyllingsdal),and a VectorNav VN-100 IMU $\left( {{200}\mathrm{\;{Hz}}}\right)$ . The sensors and the onboard computer are time synchronized using a separate microcontroller as described in Nissov et al. (2025).
空中机器人1（AR-1）：第一台空中机器人，称为AR-1，是RMF-Owl Petris等（2022）的改进版本。AR-1集成了Khadas Vim4计算机，以及传感套件：Ouster OS0-128激光雷达（10 Hz）、Flir Blackfly S 0.4 MP彩色相机（通常20 Hz，在Fyllingsdal为25 Hz）、德州仪器mmWave IWR6843AOP雷达传感器（通常10 Hz，在Fyllingsdal为${25}\mathrm{\;{Hz}}$），以及VectorNav VN-100 IMU $\left( {{200}\mathrm{\;{Hz}}}\right)$。传感器与板载计算机通过一个独立的微控制器实现时间同步，如Nissov等（2025）所述。


Aerial Robot 2 (AR-2): AR-2 is a collision-tolerant quadrotor designed for autonomous operation in GNSS-denied confined environments. AR-2 features a lightweight protective frame made from carbon-foam sandwich measuring approximately ${0.52} \times  {0.52} \times  {0.24}\mathrm{\;m}\left( {\mathrm{\;L} \times  \mathrm{W} \times  \mathrm{H}}\right)$ , weighing 2.3 kg. The robot carries the UniPilot Kulkarni et al. (2025a) sensing and computing payload on which the entire UAstack runs onboard. The module features an NVIDIA Jetson Orin NX as the compute module and a multi-modal sensing suite including a RoboSense Airy dome LiDAR (10 Hz), 3×MIPI Vision Components IMX296 color cameras (20 Hz), a D3 Embedded RS-6843AOPU FMCW radar (10 Hz), and a VectorNav VN-100 IMU (200 Hz). The onboard sensors are time-synchronized following the work in Kulkarni et al. (2025a) AR-2 further integrates a Pixracer Pro PX4-based flight control to track the acceleration commands given by the UAstack.
空中机器人2（AR-2）：AR-2是一款对碰撞更具容错能力的四旋翼，用于在无GNSS的封闭环境中进行自主运行。AR-2采用由碳-泡沫夹层制成的轻量化防护框架，尺寸约为${0.52} \times  {0.52} \times  {0.24}\mathrm{\;m}\left( {\mathrm{\;L} \times  \mathrm{W} \times  \mathrm{H}}\right)$，重量为2.3 kg。机器人搭载UniPilot Kulkarni等（2025a）的感知与计算载荷，UAstack的全部运行都在其板载完成。该模块以NVIDIA Jetson Orin NX作为计算模块，并配备多模态感知套件：RoboSense Airy头部激光雷达（10 Hz）、3×MIPI Vision Components IMX296彩色相机（20 Hz）、D3 Embedded RS-6843AOPU FMCW雷达（10 Hz）以及VectorNav VN-100 IMU（200 Hz）。板载传感器按照Kulkarni等（2025a）的工作进行时间同步。AR-2进一步集成了一款基于Pixracer Pro的PX4飞行控制器，用于跟踪UAstack给出的加速度指令。


#### 4.1.2 Legged Robots
#### 4.1.2 行走机器人


Ground Robot 1 (GR-1) On the ground, the UAstack is evaluated using the ANYmal D legged robot - hereafter referred to as GR-1 - with dimensions of ${0.93} \times  {0.53} \times  {0.80}\mathrm{\;m}\left( {\mathrm{\;L} \times  \mathrm{W} \times  \mathrm{H}}\right)$ ,and a mass of ${50}\mathrm{\;{kg}}$ . The stock standard robot features a Velodyne VLP-16 LiDAR and $6 \times$ depth cameras for sensing and two 8th-generation Intel Core i7 CPUs (6 cores each) for compute. However, GR-1 is also equipped with the UniPilot module which runs the UAstack for all evaluations. This UniPilot carries the same sensing payload as in the AR-2 platform and is interfaced with the onboard Intel computers which then run the locomotion and velocity tracking controllers. All the stack runs on the onboard UniPilot, considering the data of this module.
地面机器人 1（GR-1）：在地面条件下，使用 ANYmal D 行走机器人对 UAstack 进行评估——下文简称为 GR-1——其尺寸为 ${0.93} \times  {0.53} \times  {0.80}\mathrm{\;m}\left( {\mathrm{\;L} \times  \mathrm{W} \times  \mathrm{H}}\right)$ ，质量为 ${50}\mathrm{\;{kg}}$ 。该原装机器人配备 Velodyne VLP-16 LiDAR 以及 $6 \times$ 深度相机用于感知，并使用两颗第 8 代英特尔 Core i7 CPU（每颗 6 核）用于计算。不过，GR-1 还集成了 UniPilot 模块，在所有评估中运行 UAstack。该 UniPilot 携带与 AR-2 平台相同的感知载荷，并与机载英特尔计算机相连，随后由这些计算机运行步态与速度跟踪控制器。所有模块均在机载 UniPilot 上运行，并考虑该模块的数据。


### 4.2 Perception Module Evaluation
### 4.2 感知模块评估


We first evaluate the performance and resilience of the stack's perception module, focused around its multimodal SLAM functionalities. We subsequently evaluate downstream functionalities for VLM-based scene reasoning. 4.2.1 Multi-Modal SLAM To evaluate the multi-modal SLAM solution, we assess its accuracy and robustness in perceptually-degraded conditions. Considering a set of diverse environments, specifically a bicycle tunnel (Fyllingsdal), a road tunnel (Runehamar), a frozen lake, and the NTNU main campus, we present a modality-wise ablation as well as comparisons against state-of-the-art methods. The goal of this evaluation is to demonstrate the flexibility of the proposed SLAM system, as well as the comparative advantages of the multi-modal fusion. These experiments are collected with ground truth, such that the estimation performance can be evaluated quantitatively. The same SLAM system runs online to support the autonomous missions conducted in all the other experiments. For these autonomous missions, the SLAM module is operating in the Ours - LRI (fusing LiDAR, radar, IMU) configuration, except when noted otherwise (see Table 5), as this has been found to be most robust across diverse environments and conditions.
我们首先评估堆栈的感知模块的性能与鲁棒性，重点围绕其多模态 SLAM 功能。随后，我们评估基于 VLM 的场景推理等下游功能。4.2.1 多模态 SLAM 为了评估多模态 SLAM 方案，我们在感知退化的条件下考察其准确性与鲁棒性。考虑一组多样化的环境，包括自行车隧道（Fyllingsdal）、道路隧道（Runehamar）、结冰的湖面以及 NTNU 主校区，我们给出按模态划分的消融结果，并与最先进的方法进行对比。本次评估的目标在于展示所提出 SLAM 系统的灵活性，以及多模态融合的相对优势。这些实验都收集了真实标注数据，因此可以对估计性能进行定量评估。同一个 SLAM 系统在线运行，以支持其余所有实验中开展的自主任务。在这些自主任务中，SLAM 模块运行在 Ours - LRI（融合 LiDAR、雷达、IMU）的配置下，除非另有说明（见表 5），因为已发现这种配置在不同环境与条件下最为稳健。


Table 5. Overview of the different experiments composing the evaluation.
表 5. 构成本次评估的不同实验概览。


<table><tr><td>Purpose</td><td>Environment</td><td>Robot</td><td>UAstack Module(s) ${}^{1}$</td><td>Perceptual Challenge(s)</td><td>Geometric Challenge(s)</td></tr><tr><td rowspan="4">SLAM Validation</td><td>Fyllingsdal</td><td>AR-1</td><td>S</td><td>Self-similarity</td><td>-</td></tr><tr><td>Runehamar</td><td>AR-1</td><td>S</td><td>Low-light</td><td>-</td></tr><tr><td>Frozen Lake</td><td>AR-1</td><td>S</td><td>Self-Similarity</td><td>-</td></tr><tr><td>Campus</td><td>Handheld</td><td>S+V</td><td>Visual obscurants</td><td>-</td></tr><tr><td rowspan="5">Safety Validation</td><td rowspan="3">Forest</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + \mathrm{R} + \mathrm{C}$</td><td rowspan="3">Thin features</td><td rowspan="3">Thin obstacles and cluttered</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{N}}_{\mathrm{S}} + \mathrm{C}$</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{N}}_{\mathrm{U}} + \mathrm{C}$</td></tr><tr><td rowspan="2">Campus</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + \mathrm{R} + \mathrm{C}$</td><td rowspan="2">- <br> -</td><td rowspan="2">Moving obstacles</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + {\mathrm{N}}_{\mathrm{S}} + \mathrm{C}$</td></tr><tr><td rowspan="7">Full-Stack Validation</td><td rowspan="2">Forest</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + \mathrm{R} + \mathrm{C}$</td><td rowspan="2">Thin features</td><td rowspan="2">Thin obstacles and cluttered</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + {\mathrm{N}}_{\mathrm{S}} + \mathrm{C}$</td></tr><tr><td rowspan="3">Mine</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + \mathrm{R} + \mathrm{C}$</td><td rowspan="3">Low-light</td><td rowspan="3">Narrow and multiple branches</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + {\mathrm{N}}_{\mathrm{S}} + \mathrm{C}$</td></tr><tr><td>GR-1</td><td>${\mathrm{S}}_{\mathrm{{LI}}} + {\mathrm{P}}_{\mathrm{E}}$</td></tr><tr><td>Ship</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LI}}} + {\mathrm{P}}_{\mathrm{E}} + {\mathrm{P}}_{\mathrm{I}}$</td><td>Low-light</td><td>-</td></tr><tr><td>Campus</td><td>GR-1</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + \mathrm{V}$</td><td>-</td><td>Narrow, varying size, and multiple branches</td></tr></table>
<table><tbody><tr><td>目的</td><td>环境</td><td>机器人</td><td>UAstack 模块 ${}^{1}$</td><td>感知挑战</td><td>几何挑战</td></tr><tr><td rowspan="4">SLAM 验证</td><td>Fyllingsdal</td><td>AR-1</td><td>S</td><td>自相似性</td><td>-</td></tr><tr><td>Runehamar</td><td>AR-1</td><td>S</td><td>低光照</td><td>-</td></tr><tr><td>Frozen Lake</td><td>AR-1</td><td>S</td><td>自相似性</td><td>-</td></tr><tr><td>校园</td><td>手持</td><td>S+V</td><td>视觉遮挡物</td><td>-</td></tr><tr><td rowspan="5">安全验证</td><td rowspan="3">森林</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + \mathrm{R} + \mathrm{C}$</td><td rowspan="3">细薄特征</td><td rowspan="3">薄障碍物及杂乱</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{N}}_{\mathrm{S}} + \mathrm{C}$</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{N}}_{\mathrm{U}} + \mathrm{C}$</td></tr><tr><td rowspan="2">校园</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + \mathrm{R} + \mathrm{C}$</td><td rowspan="2">- <br/> -</td><td rowspan="2">移动障碍物</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + {\mathrm{N}}_{\mathrm{S}} + \mathrm{C}$</td></tr><tr><td rowspan="7">全栈验证</td><td rowspan="2">森林</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + \mathrm{R} + \mathrm{C}$</td><td rowspan="2">细薄特征</td><td rowspan="2">薄障碍物及杂乱</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + {\mathrm{N}}_{\mathrm{S}} + \mathrm{C}$</td></tr><tr><td rowspan="3">矿井</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + \mathrm{R} + \mathrm{C}$</td><td rowspan="3">低光照</td><td rowspan="3">狭窄且多分支</td></tr><tr><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + {\mathrm{N}}_{\mathrm{S}} + \mathrm{C}$</td></tr><tr><td>GR-1</td><td>${\mathrm{S}}_{\mathrm{{LI}}} + {\mathrm{P}}_{\mathrm{E}}$</td></tr><tr><td>船舶</td><td>AR-2</td><td>${\mathrm{S}}_{\mathrm{{LI}}} + {\mathrm{P}}_{\mathrm{E}} + {\mathrm{P}}_{\mathrm{I}}$</td><td>低光照</td><td>-</td></tr><tr><td>校园</td><td>GR-1</td><td>${\mathrm{S}}_{\mathrm{{LRI}}} + {\mathrm{P}}_{\mathrm{E}} + \mathrm{V}$</td><td>-</td><td>狭窄、尺寸变化且多分支</td></tr></tbody></table>


${}^{1}$ Which modules of the UAstack were used,including SLAM (denoted by S,with a subscript denoting the configuration used in autonomous missions, where L: LiDAR, R: Radar, I: IMU), VLM (denoted by V), Planning (denoted by P,while ${\mathrm{P}}_{\mathrm{E}}$ refers to exploration and ${\mathrm{P}}_{\mathrm{I}}$ to inspection),ExRL (denoted by R),SDF-NMPC (denoted by ${\mathrm{N}}_{\mathrm{S}}$ ),SDF-NMPC without SDF constraints (denoted by ${\mathrm{N}}_{\mathrm{U}}$ ),and C-CBF (denoted by C).
${}^{1}$ 使用了UAstack中的哪些模块，包括SLAM（以S表示，且下标表示用于自主任务的配置：L代表LiDAR，R代表雷达，I代表IMU）、VLM（以V表示）、规划（以P表示，其中${\mathrm{P}}_{\mathrm{E}}$表示探索、${\mathrm{P}}_{\mathrm{I}}$表示巡检）、ExRL（以R表示）、SDF-NMPC（以${\mathrm{N}}_{\mathrm{S}}$表示）、无SDF约束的SDF-NMPC（以${\mathrm{N}}_{\mathrm{U}}$表示）以及C-CBF（以C表示）。


The results of the evaluation are presented in Table 7. The table compares and ablates our multi-modal solution against state-of-the-art LiDAR-Inertial (FAST-LIO2 Xu et al. (2022)), Visual-Inertial (ROVIO Bloesch et al. (2017) and OpenVINS Geneva et al. (2020b)), LiDAR-Visual-Inertial (FAST-LIVO2 Zheng et al. (2025)), and LiDAR-Radar-Inertial (GaRLIO Noh et al. (2025) and AF-RLIO Qian et al. (2025)) works. GaRLIO and AF-RLIO are selected as a representative set of state-of-the-art methods for LiDAR-radar-inertial fusion which are both: likely to work with the small form-factor sensors considered in this work and with open-source implementations. The different permutations of our MIMOSA-X multi-modal SLAM (LiDAR (L), Radar (R), Vision (V), and Inertial (I)) are noted in the table as Ours - XXXI. As the IMU is an integral component of our method, permutations without IMU are omitted. The key parameters used in the evaluation of the SLAM module are reported in Table 6. Note that all parameters are fixed across all tests, with the sole exception of the LiDAR point standard deviation (which is increased in the frozen lake environment), the IMU bias noise densities (decreased in the frozen lake environment), and the D-Optimality threshold for the vision fusion (which is different between environments with and without fog). In turn, the latter is one of the reasons that the robot experiments presented subsequently predominantly rely on the LRI solution. The table reports the Absolute Trajectory Error (ATE) (in m) and Relative Trajectory Error $\left( {\mathrm{{RTE}}}_{10}\right)$ (in %),with ${10}\mathrm{\;m}$ segment length, following Grupp (2017), calculated against the ground truth estimates. The ground truth for the tunnel trajectories (Fyllingsdal and Runehamar) is generated by fusing the tracking of a Leica GRZ101 mini-prism mounted on AR-1 by a Leica MS60 MultiStation with the onboard IMU in an offline Levenberg-Marquardt (LM) optimization. In the campus and frozen lake datasets, where GNSS is available, ground truth is created using Pix4DMatic for a GNSS-augmented visual bundle adjustment optimization.
评估结果见表7。该表将我们的多模态方案与最先进的LiDAR-惯性（FAST-LIO2 Xu等（2022））、视觉-惯性（ROVIO Bloesch等（2017）与OpenVINS Geneva等（2020b））、LiDAR-视觉-惯性（FAST-LIVO2 Zheng等（2025））以及LiDAR-雷达-惯性（GaRLIO Noh等（2025）与AF-RLIO Qian等（2025））工作进行对比与消融分析。选取GaRLIO和AF-RLIO作为LiDAR-雷达-惯性融合的代表性最先进方法，因为它们都可能适用于本文考虑的小型传感器形式，并且提供开源实现。表中将我们的MIMOSA-X多模态SLAM（LiDAR（L）、雷达（R）、视觉（V）与惯性（I））的不同组合记为“我们的-XXXI”。由于IMU是我们方法的组成部分，省略IMU的组合被排除。SLAM模块评估使用的关键参数见表6。注意，除以下情况外，所有参数在所有测试中保持不变：仅改变LiDAR点的标准差（在结冰湖环境中增大）、IMU偏置噪声密度（在结冰湖环境中减小）以及用于视觉融合的D-Optimality阈值（在有雾与无雾环境之间不同）。因此，后续展示的机器人实验主要依赖LRI方案。表中给出了绝对轨迹误差（ATE）（以m计）与相对轨迹误差$\left( {\mathrm{{RTE}}}_{10}\right)$（以%计），并按Grupp（2017）在采用${10}\mathrm{\;m}$分段长度的情况下，相对真实值进行计算。隧道轨迹（Fyllingsdal与Runehamar）的真实值通过离线Levenberg-Marquardt（LM）优化获得：将安装在AR-1上的Leica GRZ101微棱镜的跟踪结果，与Leica MS60 MultiStation的测量以及机载IMU进行融合。对于校园与结冰湖数据集（GNSS可用），真实值使用Pix4DMatic生成：进行带GNSS增强的视觉束调整优化。


The SLAM datasets were collected with (a) AR-1 aerial robot and (b) a helmet-mounted modified version of the UniPilot Kulkarni et al. (2025a) module integrating a Hesai JT-128 LiDAR (10 Hz), replacing the Robosense Airy, and a uRAD Industrial radar $\left( {{10}\mathrm{\;{Hz}}}\right)$ ,replacing the $3\mathrm{D}$ Embedded RS-6843AOPU radar (both of which integrate the Texas Instruments IWR6843AOP chip). The full datasets including raw data and ground-truth are released to facilitate comparison and reproducibility.
SLAM数据集通过以下方式采集：（a）AR-1空中机器人；（b）在头盔上安装的UniPilot Kulkarni等（2025a）的改进版本模块：集成Hesai JT-128 LiDAR（10 Hz），替换Robosense Airy，并加入一台uRAD Industrial雷达$\left( {{10}\mathrm{\;{Hz}}}\right)$，替换$3\mathrm{D}$ Embedded RS-6843AOPU雷达（两者均集成德州仪器IWR6843AOP芯片）。完整数据集（包含原始数据与真实值）已发布，以便进行对比与复现。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_18.jpg?x=134&y=141&w=1381&h=1163&r=0"/>



Figure 9. Overview of the SLAM performance in the two tunnel environments. The Fyllingsdal tunnel presents an environment with geometric self-similarity, thus causing divergence in LiDAR-geometry-based methods. Multi-modal fusion demonstrates increased performance, by replacing the missing observability of the LiDAR measurements with information from either radar or vision. The Runehamar tunnel contains sections of near geometric self-similarity, too short to cause widespread divergence. This environment also has sections with low-lighting and poor radar returns, complicating state estimation. Again the multi-modal fusion demonstrates robust performance.
图9展示了两个隧道环境中的SLAM性能概览。Fyllingsdal隧道具有几何自相似性，从而导致基于LiDAR几何的方法出现发散。多模态融合通过用雷达或视觉信息替代LiDAR测量缺失的可观测性，从而提升性能。Runehamar隧道同样包含几段接近几何自相似的区域，但长度不足以引发广泛发散。该环境里还存在低照度以及雷达回波较差的部分，使状态估计更为复杂。再一次，多模态融合表现出稳健的性能。


Fyllingsdal Tunnel The AR-1 aerial robot was manually piloted at an average speed of ${6.82}\mathrm{\;m}/\mathrm{s}$ (max speed ${9.2}\mathrm{\;m}/\mathrm{s}$ ) in a $\sim  {650}\mathrm{\;m}$ section of the Fyllingsdal bicycle tunnel. The tunnel is composed primarily of long geometrically self-similar sections with sparse geometrically dissimilar rest areas. Ground truth for this environment is generated using the Leica-IMU fusion described above. Notably, the chirp configuration for the radar in this experiment is changed for better performance at high speeds. This results in a greater maximum Doppler as well as an increased measurement rate of ${25}\mathrm{\;{Hz}}$ . GaRLIO could not be evaluated on this sequence as the implementation requires the radar and LiDAR to be at the same rate. Due to the geometric self-similarity affecting the LiDAR optimization, FAST-LIO2 as well as Ours - LI failed. AF-RLIO fails similarly, due to the method relying on radar-based scan registration during periods of LiDAR degeneracy, which the small form-factor radar sensor used on the aerial platform is not well-suited for. The radar-inertial ablation (Ours - RI), while able to function, has a significant vertical and yaw drift due to the nature of the sensor. Furthermore, the high speeds created difficulties for vision-based methods, resulting in significant performance deterioration for ROVIO. As a result, Ours - VI fails, and Ours - RVI does not see large performance improvements over Ours - RI. OpenVINS performed better than ROVIO, Ours - VI, Ours - RVI, and Ours - LVI, however still not as well as the nominal multi-modal configuration Ours - LRI and Ours - LRVI. The proposed method, by fusing the multiple exteroceptive modalities, demonstrates performance robust both to the challenging conditions, but also to the asynchronous measurements, leading to the improved results of Ours - LRI, and Ours - LRVI. Similarly, FAST-LIVO2 is able to leverage the vision information to outperform most of the other methods, achieving performance similar to, but slightly worse than, Ours - LRI and Ours - LRVI. The estimated trajectories, ground truth and the accumulated LiDAR point cloud map from the Ours - LRI configuration is visualized in Figure 9.
Fyllingsdal 隧道 在该实验的 Fyllingsdal 自行车隧道的 $\sim  {650}\mathrm{\;m}$ 路段中，AR-1 无人机手动操控，平均速度为 ${6.82}\mathrm{\;m}/\mathrm{s}$（最高速度 ${9.2}\mathrm{\;m}/\mathrm{s}$）。该隧道主要由长距离、几何自相似性强的路段构成，夹杂少量几何差异明显的休息区域。该环境的真实值通过前述的 Leica-IMU 融合生成。值得注意的是，为了在高速下获得更好的性能，本实验中雷达的 chirp 配置进行了调整。其结果是最大多普勒更高，并且测量频率提高到 ${25}\mathrm{\;{Hz}}$。由于实现需要雷达与激光雷达保持相同的采样速率，因此 GaRLIO 无法在该序列上进行评估。由于几何自相似性会影响激光雷达的优化，FAST-LIO2 以及我们的方法 - LI 失败。AF-RLIO 也以类似方式失败，因为该方法依赖雷达的基于扫描配准来应对激光雷达退化阶段，而在无人机平台上使用的小型雷达传感器并不适合这种情况。雷达-惯性消融（我们的方法 - RI）虽然能够运行，但由于传感器特性，会出现显著的垂直漂移和偏航漂移。此外，高速环境也给基于视觉的方法带来困难，导致 ROVIO 的性能出现明显下降。因而，我们的方法 - VI 失败，而我们的方法 - RVI 相比我们的方法 - RI 没有看到大的性能提升。OpenVINS 的表现优于 ROVIO、我们的方法 - VI、我们的方法 - RVI 以及我们的方法 - LVI，但仍不如名义的多模态配置我们的方法 - LRI 和我们的方法 - LRVI。所提出的方法通过融合多种外部感知模态，表明其在这些具有挑战性的条件下具有稳健性，同时也能应对异步测量，从而带来我们的方法 - LRI 和我们的方法 - LRVI 的改进结果。同样，FAST-LIVO2 能够利用视觉信息，超越大多数其他方法，其性能与我们的方法 - LRI 和我们的方法 - LRVI 相近，但略差。图 9 展示了我们的方法 - LRI 配置下估计轨迹、真实值以及累积的激光雷达点云地图。


Table 6. Key parameters used in the evaluations.
表 6. 评估中使用的关键参数。


<table><tr><td>Module</td><td>Parameter</td><td>Value</td><td>Unit ${}^{1}$</td></tr><tr><td rowspan="9">SLAM</td><td>${\eta }_{p}$</td><td>0.2</td><td>m</td></tr><tr><td>${n}_{p}$</td><td>20</td><td>-</td></tr><tr><td>${\sigma }_{{\epsilon }_{C}}$ (nom.)</td><td>0.07</td><td>m</td></tr><tr><td>${\sigma }_{{\epsilon }_{C}}$ (lake)</td><td>0.18</td><td>m</td></tr><tr><td>Radar FoV</td><td>${120} \times  {120}$</td><td>0</td></tr><tr><td>Map update</td><td>1</td><td>m</td></tr><tr><td>thresholds</td><td>10</td><td>0</td></tr><tr><td>D-Opt. (nom.)</td><td>0.02</td><td>-</td></tr><tr><td>D-Opt. (fog)</td><td>0.001</td><td>-</td></tr><tr><td rowspan="10">Planner</td><td>${v}_{m}$</td><td>0.2</td><td>m</td></tr><tr><td>${v}_{h}$</td><td>0.2</td><td>m</td></tr><tr><td>${\mathbf{b}}_{R}$ (AR-2)</td><td>${0.8} \times  {0.8} \times  {0.8}$</td><td>m</td></tr><tr><td>${\mathbf{b}}_{R}$ (GR-1)</td><td>${1.0} \times  {1.0} \times  {0.4}$</td><td>m</td></tr><tr><td>${\mathbf{b}}_{L}$ (AR-2)</td><td>${30.0} \times  {30.0} \times  {4.0}$</td><td>m</td></tr><tr><td>${\mathbf{b}}_{L}$ (GR-1)</td><td>${40.0} \times  {40.0} \times  {4.0}$</td><td>m</td></tr><tr><td>FoV of $\mathcal{D}$</td><td>${180} \times  {90}$</td><td>0</td></tr><tr><td>FoV of $\mathcal{C}$</td><td>118×94</td><td>0</td></tr><tr><td>${d}_{view}$</td><td>1.25</td><td>m</td></tr><tr><td>${d}_{path}$</td><td>2.0</td><td>m</td></tr><tr><td rowspan="3">SDF-NMPC</td><td>${T}_{SDF}$</td><td>1.0</td><td>m</td></tr><tr><td>${d}_{\max }$</td><td>5.0</td><td>m</td></tr><tr><td>$r$</td><td>0.5</td><td>m</td></tr><tr><td rowspan="4">ExRL</td><td>${u}_{n}^{x}$</td><td>2.0</td><td>m/s2</td></tr><tr><td>${u}_{n}^{y}$</td><td>2.0</td><td>m/s2</td></tr><tr><td>${u}_{n}^{z}$</td><td>1.5</td><td>m/s2</td></tr><tr><td>${u}_{n}^{\psi }$</td><td>$\frac{\pi }{3}$</td><td>rad/s</td></tr><tr><td rowspan="10">C-CBF</td><td>$\lambda$</td><td>3.0</td><td>-</td></tr><tr><td>$\sigma$</td><td>0.3</td><td>-</td></tr><tr><td>$p$</td><td>0.8</td><td>-</td></tr><tr><td>${\rho }_{1}$</td><td>0.5</td><td>-</td></tr><tr><td>${\rho }_{2}$</td><td>0.5</td><td>-</td></tr><tr><td>$\alpha$</td><td>4.0</td><td>-</td></tr><tr><td>$\kappa$</td><td>80.0</td><td>-</td></tr><tr><td>$\gamma$</td><td>40.0</td><td>-</td></tr><tr><td>$\varepsilon$</td><td>0.3</td><td>m</td></tr><tr><td>$N$</td><td>256</td><td>points</td></tr></table>
<table><tbody><tr><td>模块</td><td>参数</td><td>值</td><td>单位 ${}^{1}$</td></tr><tr><td rowspan="9">SLAM</td><td>${\eta }_{p}$</td><td>0.2</td><td>m</td></tr><tr><td>${n}_{p}$</td><td>20</td><td>-</td></tr><tr><td>${\sigma }_{{\epsilon }_{C}}$（标称）</td><td>0.07</td><td>m</td></tr><tr><td>${\sigma }_{{\epsilon }_{C}}$（湖面）</td><td>0.18</td><td>m</td></tr><tr><td>雷达视场角</td><td>${120} \times  {120}$</td><td>0</td></tr><tr><td>地图更新</td><td>1</td><td>m</td></tr><tr><td>阈值</td><td>10</td><td>0</td></tr><tr><td>D-Opt.（标称）</td><td>0.02</td><td>-</td></tr><tr><td>D-Opt.（雾）</td><td>0.001</td><td>-</td></tr><tr><td rowspan="10">规划器</td><td>${v}_{m}$</td><td>0.2</td><td>m</td></tr><tr><td>${v}_{h}$</td><td>0.2</td><td>m</td></tr><tr><td>${\mathbf{b}}_{R}$（AR-2）</td><td>${0.8} \times  {0.8} \times  {0.8}$</td><td>m</td></tr><tr><td>${\mathbf{b}}_{R}$（GR-1）</td><td>${1.0} \times  {1.0} \times  {0.4}$</td><td>m</td></tr><tr><td>${\mathbf{b}}_{L}$（AR-2）</td><td>${30.0} \times  {30.0} \times  {4.0}$</td><td>m</td></tr><tr><td>${\mathbf{b}}_{L}$（GR-1）</td><td>${40.0} \times  {40.0} \times  {4.0}$</td><td>m</td></tr><tr><td>$\mathcal{D}$的视场角</td><td>${180} \times  {90}$</td><td>0</td></tr><tr><td>$\mathcal{C}$的视场角</td><td>118×94</td><td>0</td></tr><tr><td>${d}_{view}$</td><td>1.25</td><td>m</td></tr><tr><td>${d}_{path}$</td><td>2.0</td><td>m</td></tr><tr><td rowspan="3">SDF-NMPC</td><td>${T}_{SDF}$</td><td>1.0</td><td>m</td></tr><tr><td>${d}_{\max }$</td><td>5.0</td><td>m</td></tr><tr><td>$r$</td><td>0.5</td><td>m</td></tr><tr><td rowspan="4">ExRL</td><td>${u}_{n}^{x}$</td><td>2.0</td><td>米/秒2</td></tr><tr><td>${u}_{n}^{y}$</td><td>2.0</td><td>米/秒2</td></tr><tr><td>${u}_{n}^{z}$</td><td>1.5</td><td>米/秒2</td></tr><tr><td>${u}_{n}^{\psi }$</td><td>$\frac{\pi }{3}$</td><td>弧度/秒</td></tr><tr><td rowspan="10">C-CBF</td><td>$\lambda$</td><td>3.0</td><td>-</td></tr><tr><td>$\sigma$</td><td>0.3</td><td>-</td></tr><tr><td>$p$</td><td>0.8</td><td>-</td></tr><tr><td>${\rho }_{1}$</td><td>0.5</td><td>-</td></tr><tr><td>${\rho }_{2}$</td><td>0.5</td><td>-</td></tr><tr><td>$\alpha$</td><td>4.0</td><td>-</td></tr><tr><td>$\kappa$</td><td>80.0</td><td>-</td></tr><tr><td>$\gamma$</td><td>40.0</td><td>-</td></tr><tr><td>$\varepsilon$</td><td>0.3</td><td>m</td></tr><tr><td>$N$</td><td>256</td><td>点</td></tr></tbody></table>


${}^{1}$ Unitless quantities denoted by - .
${}^{1}$ 用 - 表示无量纲量。


Runehamar Tunnel The AR-1 aerial robot was manually flown through a section of the Runehamar tunnel, for a total trajectory length of $\sim  {1.4}\mathrm{\;{km}}$ . Ground truth for this environment is generated using the Leica-IMU method described above. The tunnel has a rough interior, which allows for LiDAR-based methods (FAST-LIO2 and Ours - LI) to function well despite the otherwise minor geometric self-similarity. However, regions of the tunnel are not illuminated. Despite the aerial platform carrying onboard lighting, the scale of the environment is such that the images captured by the camera are dark and hence challenging for visual-based methods, leading to very poor performance from ROVIO (and hence Ours - VI). OpenVINS, by doing a histogram equalization of the image is able to better extract features from the darkest regions and as a result, demonstrates improved performance over ROVIO, however, still worse than the LiDAR-based fusions. Impacted by the poor visual measurement quality, FAST-LIVO2 performs slightly worse than FAST-LIO2. Radar measurement quality in this environment was also quite poor, in part due to the speed of the trajectory and in part due to low number of reflections. As a result Ours - RI, Ours - RVI, and the baseline radar-fusion methods (GaRLIO and AF-RLIO) do not perform well either. However, in combination with the LiDAR (in Ours - LVI, Ours - LRI, or Ours - LRVI) the proposed multi-modal fusion demonstrates robust performance. Some of the aforementioned results are visualized in Figure 9, alongside the previous tunnel experiment. Note here the sparsity of the instantaneous radar point cloud in this particular environment.
Runehamar 隧道 AR-1 无人机被手动飞行穿过 Runehamar 隧道的一段区域，总轨迹长度为 $\sim  {1.4}\mathrm{\;{km}}$ 。该环境的真值采用上文所述的 Leica-IMU 方法生成。隧道内部较为粗糙，使得基于 LiDAR 的方法（FAST-LIO2 和 Ours - LI）即便在原本几何自相似性较弱的情况下也能良好运行。然而，隧道的部分区域没有被照亮。尽管机载照明存在，但由于环境尺度的原因，相机采集的图像较暗，因此基于视觉的方法难以开展，导致 ROVIO（以及因此的 Ours - VI）表现极差。通过对图像进行直方图均衡化，OpenVINS 能更好地从最暗区域提取特征，从而相较 ROVIO 获得了提升，但仍不如基于 LiDAR 的融合。受较差的视觉测量质量影响，FAST-LIVO2 的表现略差于 FAST-LIO2。该环境中的雷达测量质量也相当差，部分是因为轨迹速度较快，部分是因为反射次数较少。因此 Ours - RI、Ours - RVI 以及基线雷达融合方法（GaRLIO 和 AF-RLIO）也表现不佳。然而，与 LiDAR 结合后（在 Ours - LVI、Ours - LRI 或 Ours - LRVI 中），所提出的多模态融合依然展现出稳健的性能。上述部分结果在图 9 中与先前的隧道实验一并可视化。注意：在该环境中，瞬时雷达点云具有较强的稀疏性。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_19.jpg?x=837&y=144&w=678&h=817&r=0"/>



Figure 10. Overview of the SLAM performance in the Frozen Lake environment, comparing different uni-modal approaches with the multi-modal LRI configuration. This environment presents difficulty for methods relying only on LiDAR or radar. For the former, the planar geometry of the environment can result in lacking observability in lateral position and yaw. For the radar, the limited number of returns results in increased drift.
图 10. 冻结湖（Frozen Lake）环境下的 SLAM 性能概览，比较不同单模态方案与多模态 LRI 配置。该环境对仅依赖 LiDAR 或雷达的方法具有难度。对于前者，环境的平面几何可能导致横向位置和航向角缺乏可观性。对于雷达，回波次数有限会导致漂移增加。


Frozen Lake The AR-1 aerial robot was manually piloted on top of a frozen lake, starting from close to a bank, flying out into the middle, and returning close to the starting location. The ground truth was generated using the Pix4DMatic bundle adjustment result. Once the robot has traveled further than the range of the LiDAR away from any bank of the lake, the point cloud is geometrically self-similar and resembles a large plane. Notably, the LiDAR point cloud is also affected by the ice such that rays with a large incidence angle on the ice return invalid points and valid points have increased noise standard deviation. The radar after takeoff and before landing mostly returns less than three points per point cloud with frequently empty point clouds. The returns are primarily from the surface directly below the robot as it is flying.
冻结湖（Frozen Lake） AR-1 无人机在结冰湖面上进行手动操控：从靠近岸边的位置出发，飞向湖中央，然后返回到接近起始位置。真值使用 Pix4DMatic 的位束调整结果生成。当机器人行进的距离超过其 LiDAR 相对任意岸边的量程后，点云在几何上呈自相似，并类似于一个大平面。值得注意的是，LiDAR 点云也会受到冰的影响：入射角较大的射线会返回无效点，而有效点的噪声标准差会增加。起飞后、降落前，雷达每个点云中大多返回少于三个点，且经常出现空点云。回波主要来自机器人飞行正下方的表面。


Table 7. Metrics for the multi-modal SLAM evaluation,including ATE [m] and ${\mathrm{{RTE}}}_{10}\left\lbrack  \% \right\rbrack$ .
表 7. 多模态 SLAM 评估指标，包括 ATE [m] 和 ${\mathrm{{RTE}}}_{10}\left\lbrack  \% \right\rbrack$ 。


<table><tr><td></td><td></td><td>Fyllingsdal Tunnel</td><td>Runehamar Tunnel</td><td>Frozen Lake</td><td>Campus Fog</td></tr><tr><td></td><td>Length [m]</td><td>1275.378</td><td>1444.151</td><td>826.054</td><td>669.609</td></tr><tr><td rowspan="13">ATE [m] / RTE10 [%]</td><td>FAST-LIO2</td><td>✘</td><td>6.578 / 1.891</td><td>✘</td><td>✘</td></tr><tr><td>FAST-LIVO2</td><td>4.930 / 2.852</td><td>7.881 / 1.805</td><td>✘</td><td>✘</td></tr><tr><td>GaRLIO</td><td>-</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>AF-RLIO</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>ROVIO</td><td>42.851 / 20.295</td><td>62.375 / 29.806</td><td>3.112/7.177</td><td>✘</td></tr><tr><td>OpenVINS</td><td>13.710 / 6.271</td><td>13.337 / 5.443</td><td>1.985 / 3.700</td><td>✘</td></tr><tr><td>Ours - LI</td><td>✘</td><td>7.183 / 1.762</td><td>18.661 / 7.840</td><td>✘</td></tr><tr><td>Ours - RI</td><td>21.489 / 6.245</td><td>✘</td><td>10.189 / 30.868</td><td>14.075 / 6.189</td></tr><tr><td>Ours - VI</td><td>✘</td><td>✘</td><td>7.728 / 7.989</td><td>✘</td></tr><tr><td>Ours - LVI</td><td>✘</td><td>7.012 / 1.763</td><td>10.598 / 4.633</td><td>✘</td></tr><tr><td>Ours - RVI</td><td>20.123 / 5.906</td><td>✘</td><td>9.141 / 7.589</td><td>17.737 / 6.351</td></tr><tr><td>Ours - LRI</td><td>3.899 / 1.649</td><td>7.094 / 1.787</td><td>10.562 / 6.900</td><td>7.324 / 4.240</td></tr><tr><td>Ours - LRVI</td><td>3.872 / 1.642</td><td>7.048 / 1.789</td><td>10.030 / 8.609</td><td>8.345 / 4.231</td></tr></table>
<table><tbody><tr><td></td><td></td><td>Fyllingsdal隧道</td><td>Runehamar隧道</td><td>冻湖</td><td>校园雾霾</td></tr><tr><td></td><td>长度 [m]</td><td>1275.378</td><td>1444.151</td><td>826.054</td><td>669.609</td></tr><tr><td rowspan="13">ATE [m] / RTE10 [%]</td><td>FAST-LIO2</td><td>✘</td><td>6.578 / 1.891</td><td>✘</td><td>✘</td></tr><tr><td>FAST-LIVO2</td><td>4.930 / 2.852</td><td>7.881 / 1.805</td><td>✘</td><td>✘</td></tr><tr><td>GaRLIO</td><td>-</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>AF-RLIO</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>ROVIO</td><td>42.851 / 20.295</td><td>62.375 / 29.806</td><td>3.112/7.177</td><td>✘</td></tr><tr><td>OpenVINS</td><td>13.710 / 6.271</td><td>13.337 / 5.443</td><td>1.985 / 3.700</td><td>✘</td></tr><tr><td>我们 - LI</td><td>✘</td><td>7.183 / 1.762</td><td>18.661 / 7.840</td><td>✘</td></tr><tr><td>我们 - RI</td><td>21.489 / 6.245</td><td>✘</td><td>10.189 / 30.868</td><td>14.075 / 6.189</td></tr><tr><td>我们 - VI</td><td>✘</td><td>✘</td><td>7.728 / 7.989</td><td>✘</td></tr><tr><td>我们 - LVI</td><td>✘</td><td>7.012 / 1.763</td><td>10.598 / 4.633</td><td>✘</td></tr><tr><td>我们 - RVI</td><td>20.123 / 5.906</td><td>✘</td><td>9.141 / 7.589</td><td>17.737 / 6.351</td></tr><tr><td>我们 - LRI</td><td>3.899 / 1.649</td><td>7.094 / 1.787</td><td>10.562 / 6.900</td><td>7.324 / 4.240</td></tr><tr><td>我们 - LRVI</td><td>3.872 / 1.642</td><td>7.048 / 1.789</td><td>10.030 / 8.609</td><td>8.345 / 4.231</td></tr></tbody></table>


Method failure due to ATE $> 5\%$ is indicated by $\times$ and due to inability to generalize to sensor configuration (radar being at ${25Hz}$ ) is indicated by - .
方法因ATE $> 5\%$ 失败由$\times$表示，而因无法泛化到传感器配置（雷达位于${25Hz}$）则以-表示。


For verifiability and reproducibility, the full implementation and the dataset involved in these studies are openly released.
为确保可验证性与可复现性，这些研究所用的完整实现和数据集均已公开发布。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_20.jpg?x=133&y=923&w=1380&h=766&r=0"/>



Figure 11. Overview of the SLAM performance of the experiment in the campus environment with the handheld module. The experiment has the trajectory passing through a room filled with dense fog, causing large increase of noise present in the LiDAR and vision measurements, whereas the radar remains largely unaffected. The multi-modal fusion retains the accuracy of LiDAR-based methods in nominal conditions alongside the robustness of radar through degraded environments.
图11. 便携式模块在校园环境中实验的SLAM性能概览。轨迹穿过一个充满浓雾的房间，导致LiDAR和视觉测量中的噪声大幅增加，而雷达基本不受影响。多模态融合在退化环境下保留了基于LiDAR方法在正常条件下的精度，并兼具雷达的鲁棒性。


GaRLIO fails in this dataset as it calculates a least-squares estimate of velocity from the radar point cloud for outlier rejection, which requires a minimum of three points. AF-RLIO while functioning initially, quickly breaks once far enough away from the bank due to the sparse radar point clouds. Both FAST-LIO2 and FAST-LIVO2 initially function well, however, their accuracy deteriorates rapidly when the system performs an aggressive yaw maneuver and accumulates significant error. Despite also based on LiDAR-inertial fusion, Ours - LI does not fail in this environment due to parametric differences in Ours - LI, that were not replicable in FAST-LIO2, which results in more robust performance. Due to the visually feature-full environment and good outdoor lighting conditions, OpenVINS provides the best baseline performance (followed by ROVIO). Our solution retains performance similar to these baselines across most vision-involving configurations (i.e., Ours - VI, Ours - LVI, Ours - RVI, and Ours - LRVI), while it is notable that
GaRLIO在该数据集中失败，因为它从雷达点云中计算速度的最小二乘估计以进行异常值剔除，而这需要至少三个点。AF-RLIO虽起初可运行，但由于雷达点云稀疏，一旦离河岸足够远便迅速失效。FAST-LIO2和FAST-LIVO2最初都运行良好，然而当系统执行激进的偏航机动并累积显著误差时，其精度迅速退化。尽管同样基于LiDAR-惯性融合，Ours - LI在该环境中并未失效，这是由于Ours - LI中的参数差异，而这些差异无法在FAST-LIO2中复现，从而带来了更鲁棒的性能。由于视觉特征丰富且户外光照条件良好，OpenVINS提供了最佳的基线性能（其次是ROVIO）。我们的方案在大多数涉及视觉的配置下（即Ours - VI、Ours - LVI、Ours - RVI和Ours - LRVI）都保持与这些基线相近的性能，同时值得注意的是


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_21.jpg?x=202&y=143&w=1251&h=1310&r=0"/>



Figure 12. Qualitative results of the proposed VLM reasoning system. Left: semantic 3D mapping with open-vocabulary object detections fused into a voxel grid, showing labeled objects and the robot trajectory over time. Right: binary visual question-answering examples, where the model provides "Yes/No" answers with confidence scores and explanations for high-level scene understanding tasks. The figure demonstrates the integration of spatial semantic mapping with high-level reasoning.
图12. 所提VLM推理系统的定性结果。左：将开放词汇目标检测融合进体素网格的语义三维建图，显示已标注物体和随时间变化的机器人轨迹。右：二元视觉问答示例，模型针对高层场景理解任务给出带置信分数和解释的“是/否”答案。该图展示了空间语义建图与高层推理的结合。


Ours - LI, Ours - RI, and Ours - LRI remain functional. The performance of the proposed method's ablation is shown in Figure 10, where the geometric self-similarity of the frozen lake is clear.
Ours - LI、Ours - RI和Ours - LRI保持可用。所提方法消融实验的性能如图10所示，其中冰封湖面的几何自相似性清晰可见。


Campus Fog A handheld UniPilot module is carried through a typical university campus environment, for a total trajectory length of ${670}\mathrm{\;m}$ . The trajectory starts outdoors but proceeds indoors into a fog-filled room before returning outdoors. As a result, the experiment features large variation in the local environment scale as well as dense visual obscurants, which are known to cause problems for LiDAR- and vision-based estimation. The ground truth for this experiment is created using the Pix4DMatic bundle adjustment-based method.
校园雾 A手持UniPilot模块穿过典型大学校园环境，总轨迹长度为${670}\mathrm{\;m}$。轨迹起始于室外，但随后进入一间充满雾气的房间，之后又返回室外。因此，该实验在局部环境尺度上变化很大，并伴有浓重的视觉遮挡，这些都已知会对基于LiDAR和视觉的估计造成问题。该实验的真值由基于Pix4DMatic束调整的方法生成。


As expected, both the vision- and LiDAR-based methods (i.e., FAST-LIO2, FAST-LIVO2, ROVIO, OpenVINS, Ours - LI, Ours - VI, and Ours - LVI) perform well until the room with visual obscurants (fog), wherein the aforementioned methods accumulate significant error resulting from the extended duration of unusable measurements. The Ours - RI ablation functions regardless as the radar is unaffected by such phenomena, however still accumulates drift due to the long mission duration. GaRLIO and AF-RLIO, despite including the radar, are challenged here as well. For the former, the sparseness of the radar point clouds result in challenges with respect to RANSAC-based outlier rejection, and as a result the method fails. For the latter, when LiDAR point cloud degeneracy is detected, the method switches to radar-based registration. This is generally difficult with small form-factor radar sensors, again leading to failure. This also indicates the strengths and inherent robustness of our radar Doppler factor formulation. The proposed multi-modal ablations which include the radar (Ours - RI, Ours - RVI, Ours - LRI, and LRVI) perform robustly, as the radar's invariance to visual obscurants is able to be complementarily fused with the accuracy associated with vision- and LiDAR-based methods in suitable conditions. As expected, the ablations which include the LiDAR together with the radar perform best. The results for this experiment are visualized in Figure 11, note in particular the challenge posed on LiDAR-and vision-based sensing by the visual obscurants in the fog-filled room. Here, the LiDAR returns nearly no points and the camera is completely blinded.
正如预期，基于视觉和LiDAR的方法（即FAST-LIO2、FAST-LIVO2、ROVIO、OpenVINS、Ours - LI、Ours - VI和Ours - LVI）在进入充满视觉遮挡（雾）的房间前都表现良好，而在该房间中，上述方法由于长时间测量不可用而累积了显著误差。Ours - RI消融项则始终可用，因为雷达不受此类现象影响，但由于任务持续时间长仍会累积漂移。GaRLIO和AF-RLIO尽管包含雷达，在这里也面临挑战。对于前者，雷达点云稀疏使基于RANSAC的异常值剔除变得困难，因此该方法失败。对于后者，当检测到LiDAR点云退化时，方法切换到基于雷达的配准。对于这种小型雷达传感器而言，这通常较难，再次导致失败。这也表明了我们雷达多普勒因子形式的优势及其固有鲁棒性。所提包含雷达的多模态消融项（Ours - RI、Ours - RVI、Ours - LRI和LRVI）表现稳健，因为雷达对视觉遮挡的不变性可以与适用条件下基于视觉和LiDAR方法的精度互补融合。正如预期，结合LiDAR与雷达的消融项表现最佳。该实验结果如图11所示，尤其值得注意的是雾房间中的视觉遮挡给基于LiDAR和视觉的传感带来的挑战。此时，LiDAR几乎没有返回点，摄像头则完全失明。


 Robot's Trajectory<br>Robot's Trajectory<br>A Robot's Trajectory<br>${40}\mathrm{m}$<br>Navigation Instances<br>Navigation Instances<br>Navigation Instances<br>(1.2)<br>1.2<br>2.1<br>C-CBF Intervening<br>2<br>C-CBF Intervening<br>2.1)<br>2.3<br>Avoiding thin obstacles<br>Robotstuck<br>C-CBF Filtered Acceleration Command<br>to Goal<br>SDF-NMPC<br>Rollout<br>Direction to Goal<br>C-CBF Filtered Acceleration Command<br>Direction to Goal<br>Unsafe Acceleration Command<br>SDF-NMPC Acceleration Command<br>C-CBF Filtered Acceleration Command<br>ExRL Acceleration Command -->



<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_22.jpg?x=135&y=144&w=1383&h=945&r=0"/>



Figure 13. Navigation module evaluation is performed in the forest using AR-2 with SDF-NMPC + C-CBF, ExRL + C-CBF, and C-CBF paired with an unsafe policy (SDF-NMPC but with its collision-avoidance constraints disabled) respectively. The robot was tasked to navigate to a waypoint ${38}\mathrm{\;m}$ in front of it,with a reference path going through trees and branches. Individual experiments are shown, with specific instances highlighted. Both the SDF-NMPC + C-CBF and the ExRL + C-CBF methods successfully navigated to the goal location, avoiding obstacles in the path. The C-CBF paired with an unsafe policy does not reach the goal and remains stuck, but remaining safe at all times. The full map is shown with specific instances highlighted.
图13。在森林中使用AR-2对导航模块进行评估，分别采用SDF-NMPC + C-CBF、ExRL + C-CBF，以及与不安全策略配对的C-CBF（即关闭其避碰约束的SDF-NMPC）。机器人被要求导航到前方的一个航点${38}\mathrm{\;m}$，参考路径穿过树木和树枝。展示了单次实验，并突出标出了具体实例。SDF-NMPC + C-CBF和ExRL + C-CBF两种方法都成功到达了目标位置，并避开了路径中的障碍物。与不安全策略配对的C-CBF未能到达目标并卡住，但始终保持安全。图中展示了完整地图并标出了具体实例。


4.2.2 Scene Reasoning Provided the multi-modal SLAM capabilities of the UAstack, we further assess downstream functionality for scene reasoning. Figure 12 presents two real-world examples of the proposed VLM reasoning system. As a straightforward extension to the core perception module capabilities, our open-vocabulary semantic mapping system and 3D object detection are shown on the left side of the figure, where the reconstructed point cloud, together with the bounding boxes of the detected objects, is illustrated. Our semantic mapping pipeline, based on the open-vocabulary object detector, operates at 1 Hz. This result demonstrates the ability of the perception module to consistently maintain semantic understanding over time.
4.2.2 场景推理 鉴于UAstack具备多模态SLAM能力，我们进一步评估其在场景推理中的下游功能。图12展示了所提出的VLM推理系统的两个真实世界示例。作为核心感知模块能力的直接扩展，我们的开放词汇语义建图系统和3D目标检测结果显示在图的左侧，其中展示了重建点云及检测到的物体边界框。我们的语义建图流程基于开放词汇目标检测器，以1 Hz运行。该结果表明，感知模块能够随时间持续保持语义理解。


On the right side of Figure 12, we show instances of the visual question/answering module, which provides contextual reasoning beyond geometric perception. As shown in the examples, the system correctly identifies potentially unsafe conditions in fog with high confidence, as well as determines whether there are exits or entrances in the current view. In our current implementation, GPT-5 is queried through the OpenAI API every ${50}\mathrm{\;s}$ . The end-to-end inference latency across our evaluations was ${5.66} \pm  {1.55}$ s, including both the API latency and the model inference time. Overall, these results demonstrate that the combination of semantic 3D mapping and binary visual Q&A can enable more robust scene understanding and reasoning.
在图12右侧，我们展示了视觉问答模块的实例，它提供超越几何感知的上下文推理。如示例所示，系统能够以高置信度正确识别雾中的潜在不安全状况，并判断当前视野中是否存在出口或入口。在我们当前的实现中，GPT-5通过OpenAI API每${50}\mathrm{\;s}$被查询一次。我们所有评估中的端到端推理延迟为${5.66} \pm  {1.55}$秒，包括API延迟和模型推理时间。总体而言，这些结果表明，语义3D建图与二元视觉问答的结合能够实现更稳健的场景理解与推理。


### 4.3 Navigation Module Evaluation
### 4.3 导航模块评估


We conduct experiments to thoroughly evaluate the navigation module. Specifically, two studies are conducted. The first, evaluates the ability of the navigation module to navigate to a waypoint without the presence of a guiding path from the planning module. In the second, the ability of the navigation module to handle the sudden appearance of unmapped obstacles in the planned path is studied. The aim of these experiments is to a) evaluate the performance of the navigation module in the SDF-NMPC + C-CBF, ExRL + C-CBF, and unsafe controller + C-CBF (where applicable) configurations, b) contrast the behaviors of these collision avoidance methods, and c) evaluate the benefits of the multi-layered safety approach. In all these missions, the AR-2 platform was used, with the SLAM module running the graph optimization online after receiving each exteroceptive measurement, considering radar measurements at ${10}\mathrm{\;{Hz}}$ and LiDAR measurements at ${10}\mathrm{\;{Hz}}$ . In each experiment (including the evaluations in Section 4.4) the SDF-NMPC runs at ${40}\mathrm{\;{Hz}}$ ,ExRL at ${30}\mathrm{\;{Hz}}$ ,and C-CBF runs at ${50}\mathrm{\;{Hz}}$ . The last received exteroceptive sensor measurements and state estimates are used to populate the state and inputs for each method while it executes at the desired frequency.
我们通过实验对导航模块进行全面评估。具体开展了两项研究。第一项评估导航模块在没有规划模块提供引导路径的情况下到达航点的能力。第二项研究导航模块应对已规划路径中突然出现的未建图障碍物的能力。这些实验旨在 a) 评估 SDF-NMPC + C-CBF、ExRL + C-CBF，以及不安全控制器 + C-CBF（适用时）配置下导航模块的性能，b) 对比这些避碰方法的行为，c) 评估多层安全方法的优势。在所有任务中，均使用 AR-2 平台，SLAM 模块在接收每次外部感知测量后在线运行图优化，考虑的雷达测量频率为${10}\mathrm{\;{Hz}}$，LiDAR 测量频率为${10}\mathrm{\;{Hz}}$。在每次实验中（包括第 4.4 节中的评估），SDF-NMPC 以${40}\mathrm{\;{Hz}}$运行，ExRL 以${30}\mathrm{\;{Hz}}$运行，C-CBF 以${50}\mathrm{\;{Hz}}$运行。每种方法在以目标频率执行时，使用最近一次接收的外部感知传感器测量和状态估计来填充其状态与输入。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_23.jpg?x=135&y=143&w=1380&h=769&r=0"/>



Figure 14. Navigation module evaluation is performed in the campus using AR-2 with SDF-NMPC + C-CBF. The robot was tasked to explore the basement autonomously and return home. At two instances, an obstacle was moved into the robot's planned path during the exploration and the return phases respectively, forcing the SDF-NMPC + C-CBF to avoid these previously-absent obstacles in the robot's path. The two specific instances in the mission are highlighted to demonstrate reactive collision-avoidance. The full map is shown alongside the specific instances, with the regions of the specific obstacles highlighted.
图14。在校园中使用 AR-2 搭配 SDF-NMPC + C-CBF 进行导航模块评估。机器人被要求自主探索地下室并返回起点。在探索和返程阶段，各有一次障碍物被移入机器人规划路径中，迫使 SDF-NMPC + C-CBF 避开这些原本不存在于路径中的障碍物。任务中的这两个具体时刻被标出，以展示反应式避碰。左侧给出完整地图，右侧标出具体时刻，并高亮对应障碍物区域。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_23.jpg?x=136&y=1087&w=1377&h=782&r=0"/>



Figure 15. Navigation module evaluation is performed in the campus using AR-2 with ExRL + C-CBF. The robot was tasked to explore the basement autonomously and return home. At two instances, an obstacle was moved into the robot's planned path during the exploration and the return phases respectively, forcing the ExRL + C-CBF to avoid these previously-absent obstacles in the robot's path. The two specific instances in the mission are highlighted to demonstrated reactive collision-avoidance. The full map is shown alongside the specific instances, with the regions of the specific obstacles highlighted.
图15。在校园中使用 AR-2 搭配 ExRL + C-CBF 进行导航模块评估。机器人被要求自主探索地下室并返回起点。在探索和返程阶段，各有一次障碍物被移入机器人规划路径中，迫使 ExRL + C-CBF 避开这些原本不存在于路径中的障碍物。任务中的这两个具体时刻被标出，以展示反应式避碰。左侧给出完整地图，右侧标出具体时刻，并高亮对应障碍物区域。


Navigation to Waypoint The first evaluation studies the ability of the navigation module to navigate to a waypoint without a guiding path from the map-based planning module. The study is conducted in a forest environment. Three experiments are conducted in which the collision avoidance methods used are SDF-NMPC + C-CBF, ExRL + C-CBF, and an unsafe policy (SDF-NMPC but with its collision-avoidance constraints disabled) + C-CBF. In each experiment, the robot starts from the same location, and the same waypoint is given to the navigation module. The results of the study are shown in Figure 13. Both configurations, SDF-NMPC + C-CBF and ExRL + C-CBF, are able to reach the waypoint, avoiding the obstacles. The collision avoidance is predominantly carried out by the SDF-NMPC or ExRL, with the C-CBF intervening only in a few instances (Figures 13.1.3, 13.2.1, and 13.2.2). Engagement of the C-CBF is not a proof that SDF-NMPC or ExRL would necessarily lead to a collision but indicates that the tuning of this last-resort method was such that it triggers it to adjust the reference commands. In turn, close evaluation of the few instances when C-CBF was engaged in these experiments indicates that the robot centroid was on average ${0.65}\mathrm{\;m}$ from the obstacles (which given the robot dimensions, entails less than ${0.5}\mathrm{\;m}$ clearance). On the other hand,the combination of the unsafe version of SDF-NMPC with the C-CBF (which then becomes the only obstacle-avoidance mechanism) is not able to reach the goal and gets stuck, but remains safe at all times. As the C-CBF only aims to remain in the safe set, navigating to the goal is not per se an objective for this method. It is noted that the third configuration (unsafe controller + C-CBF) is not a recommended configuration of the UAstack and is only evaluated to present the different roles of the navigation layers.
通往航点的导航 第一项评估研究导航模块在没有地图规划模块提供引导路径的情况下到达航点的能力。该研究在森林环境中进行。共开展三组实验，所用避碰方法分别为 SDF-NMPC + C-CBF、ExRL + C-CBF，以及不安全策略（SDF-NMPC 但禁用其避碰约束）+ C-CBF。每组实验中，机器人从相同位置出发，并向导航模块提供相同的航点。研究结果如图13所示。SDF-NMPC + C-CBF 和 ExRL + C-CBF 两种配置都能够到达航点，同时避开障碍物。避碰主要由 SDF-NMPC 或 ExRL 完成，C-CBF 仅在少数情况下介入（图13.1.3、13.2.1 和 13.2.2）。C-CBF 的介入并不意味着 SDF-NMPC 或 ExRL 必然会导致碰撞，但表明这一最后防线方法的调参使其会在必要时触发并调整参考指令。进一步对这些实验中 C-CBF 被介入的少数情况进行仔细评估表明，机器人质心与障碍物的平均距离为${0.65}\mathrm{\;m}$（考虑到机器人尺寸，这意味着净空小于${0.5}\mathrm{\;m}$）。另一方面，禁用版 SDF-NMPC 与 C-CBF 的组合（此时 C-CBF 成为唯一的避障机制）无法到达目标并会卡住，但始终保持安全。由于 C-CBF 只旨在保持在安全集合内，导航到目标本身并非该方法的目标。需要指出的是，第三种配置（不安全控制器 + C-CBF）并非 UAstack 推荐配置，仅用于展示导航各层的不同作用。


Qualitatively, this study further shows the clear difference in the robot's behavior when using SDF-NMPC vs ExRL. The SDF-NMPC follows the straight line from the start to the goal more closely, while the ExRL policy deviates significantly. On the other hand, the ExRL achieves higher speeds throughout the trajectory than SDF-NMPC. Due to this, the resulting mission time for both is comparable. Hence, the selection between SDF-NMPC and ExRL depends on the requirements of the task. The different behaviors manifested between SDF-NMPC and ExRL is among the reasons why the release of the UAstack contains both methods.
定性结果进一步表明，机器人在使用 SDF-NMPC 与 ExRL 时的行为差异十分明显。SDF-NMPC 会更紧密地沿着从起点到目标的直线前进，而 ExRL 策略则发生了显著偏离。另一方面，在整个轨迹过程中，ExRL 的速度整体高于 SDF-NMPC。因此，两者的任务完成时间相当。由此可见，在 SDF-NMPC 与 ExRL 之间的选择取决于任务需求。SDF-NMPC 与 ExRL 所体现出的不同行为，也是 UAstack 发布中同时包含这两种方法的原因之一。


Moving Obstacles The second study conducted to evaluate the navigation module aims to evaluate its performance in the presence of obstacles appearing in the planned path. Specifically, the following scenario was constructed. The robot was tasked to explore a section of a university building at NTNU. In two separate instances, after the planning module plans a path based on the online map, an obstacle is placed to block this path. The planner is not re-triggered and thus the reference path shall be in collision. The ability of the navigation module to handle this scenario is tested. Two experiments are conducted with the configurations SDF-NMPC + C-CBF and ExRL + C-CBF. Figures 14 and 15 show the result of the respective experiments. As can be seen, both policies are able to successfully avoid the unseen obstacle. The figures also show that the SDF-NMPC tends to avoid the obstacle with less, yet sufficiently safe, clearance as compared to ExRL as its formulation requires it to have minimal deviation from the reference path. ExRL does not have this constraint and only aims to reach the end of the planned path safely.
移动障碍 第二项研究用于评估导航模块，其目标是在计划路径中出现障碍物时检验其性能。具体而言，构造了如下场景。机器人被要求在 NTNU 的一处大学建筑区域内进行探索。在两个独立的实例中，规划模块基于在线地图生成路径后，会放置一个障碍物以阻断该路径。不会重新触发规划器，因此参考路径将与障碍物发生碰撞。测试导航模块对这一场景的处理能力。分别在配置 SDF-NMPC + C-CBF 与 ExRL + C-CBF 下进行两项实验。图 14 和图 15 展示了各自实验的结果。如图所示，两种策略都能够成功避开未被预见的障碍物。图中还表明，由于其表述要求在参考路径上尽量偏离最小，SDF-NMPC 往往以比 ExRL 更少但足够安全的净空来回避障碍。ExRL 不具备这一约束，只是力求安全到达计划路径的终点。


### 4.4 Evaluation of the Full Stack
### 4.4 全栈评估


The full UAstack is evaluated using both aerial and ground robots. When evaluating the full stack, we examine the result of the coordinated interaction between the perception, planning and navigation modules. The perception module is the foundation of the demonstrated autonomy, the planning module drives the behaviors manifested by the robots, while the navigation module provides control and reinforces safety for the autonomous systems. Specifically, the Exploration and Inspection objectives are tested with the Planning to Target being implicitly evaluated through these. Additional insights regarding the Planning to Target behavior can be found in Zacharia et al. (2026). Similarly to Section 4.3 the SLAM module graph optimization is calculated online, with update rate matching what was previously described.
使用空中和地面机器人对完整的UAstack进行评估。在评估全栈时，我们考察感知、规划和导航模块之间协同交互的结果。感知模块是所展示自主性的基础，规划模块驱动机器人表现出的行为，而导航模块为自主系统提供控制并增强安全性。具体而言，通过这些测试探索和巡检目标，而面向目标规划则在此过程中被隐式评估。有关面向目标规划行为的更多见解，可参见Zacharia et al. (2026)。与第4.3节类似，SLAM模块的图优化在线计算，更新速率与前文所述一致。


#### 4.4.1 Evaluations with an Aerial Robot
#### 4.4.1 使用空中机器人进行评估


We evaluate the UAstack on the AR-2 in three distinct environments namely a) an underground mine, b) a forest, and c) a ship cargo hold.
我们在三个不同环境中对AR-2上的UAstack进行评估：a) 地下矿井，b) 森林，c) 船舶货舱。


Underground Mine The first experiment is conducted in a section of the Løkken mine in Norway. We demonstrate results both when otherwise using the SDF-NMPC and ExRL. The selected section of the mine is a 3-way intersection with the robot starting at one end of the narrowest branch ( ${1.5}\mathrm{\;m}$ wide). The mine has low lighting conditions, and due to the dome FoV, the LiDAR data can quickly become degenerate in narrow branches if the sensor is facing a wall. In both missions, the robot explored the first branch it started in, continued to one of the other branches, repositioned to the next upon exploring it, before finally returning to the start location when the allotted area was fully explored. The robot successfully explored the environment while remaining safe at all times. Figures 16 and 17 show the maps and planning instances in the missions corresponding to SDF-NMPC and ExRL respectively. As the environment topology does not allow much deviation from the planned path, the total path length in both missions is comparable, however, the ExRL policy finishes faster reaching higher speeds. It was observed that at some instances in the narrowest part near the starting area (marked as Start/End in the figures), the SDF-NMPC/ExRL and the C-CBF objectives are competing resulting in transient oscillations. The SDF-NMPC/ExRL are tasked to both make progress along the planned path and maintain safety, while the C-CBF only aims for robot safety thus leading to competing actions in situations that have tight safety margins as here. Nevertheless, the system was always able to continue and this behavior was short-lived.
地下矿井 第一项实验在挪威Løkken矿井的一处区域进行。我们在分别采用SDF-NMPC和ExRL（或其他情况下）时均展示结果。所选矿区为三向交叉路口，机器人从最窄分支的一端（宽度 ${1.5}\mathrm{\;m}$）起步。矿井光照较弱，由于穹顶视场FoV，当传感器正对墙壁时，LiDAR数据在狭窄分支中可能很快变得退化。在两次任务中，机器人都先探索它起步时所在的第一条分支，随后继续到其他分支之一；在探索完后再调整位置到下一条分支，最后在分配区域完全被探索后返回起始位置。机器人始终安全地成功探索了环境。图16和图17分别展示了对应SDF-NMPC与ExRL的任务地图和规划实例。由于环境拓扑不允许偏离太多，两个任务中的总路径长度相近；然而，ExRL策略能更快完成并达到更高速度。观察到，在起始区域附近最窄位置的一些时刻（图中标为Start/End），SDF-NMPC/ExRL与C-CBF目标发生竞争，导致短暂振荡。SDF-NMPC/ExRL需要同时沿规划路径取得进展并保持安全，而C-CBF只追求机器人安全，因此在如本例这种安全裕度紧张的情况下会产生竞争行为。尽管如此，系统始终能够继续运行，这种行为也只是短暂发生。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_25.jpg?x=136&y=147&w=1377&h=588&r=0"/>



Figure 16. Full-stack evaluation in a multi-branched section of an underground mine using AR-2 with SDF-NMPC as the navigation method. The robot started in one branch and explored all three branches, repositioning when needed. The SDF-NMPC tracks the planned path closely, resulting in next to zero interventions from the SDF-NMPC or C-CBF. The figure shows the full map of the environment along with key instances in the mission. Additionally, the mission statistics are shown at the top.
图16. 使用AR-2、以SDF-NMPC作为导航方法，在地下矿井的多分支区域进行全栈评估。机器人从一条分支出发并探索全部三条分支，必要时会重新定位。SDF-NMPC紧密跟踪规划路径，导致来自SDF-NMPC或C-CBF的干预几乎为零。该图展示了环境的完整地图以及任务中的关键实例。此外，任务统计信息显示在图顶。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_25.jpg?x=136&y=886&w=1377&h=587&r=0"/>



Figure 17. Full-stack evaluation in a multi-branched section of an underground mine using AR-2 with ExRL as the navigation method. The robot started in one branch and explored all three branches, repositioning when needed. The ExRL policy only aims to reach the end of each planned path, presenting larger local deviations, but remaining safe at all times. The figure shows the full map of the environment along with key instances in the mission. Additionally, the mission statistics are shown at the top.
图17. 使用AR-2、以ExRL作为导航方法，在地下矿井的多分支区域进行全栈评估。机器人从一条分支出发并探索全部三条分支，必要时会重新定位。ExRL策略只旨在到达每条规划路径的终点，因此会产生更大的局部偏离，但始终保持安全。该图展示了环境的完整地图以及任务中的关键实例。此外，任务统计信息显示在图顶。


Forest The AR-2 was deployed in a forest in Trondheim, Norway to explore an area of ${120} \times  {80}\mathrm{\;m}$ with height limited to ${2.5}\mathrm{\;m}$ . The forest area contains trees at varying densities with thin branches and foliage in some places. The ground was covered in snow at the time of testing. In this test, two missions were conducted, one each with the SDF-NMPC and the ExRL as the core navigation policy feeding into the C-CBF. Figures 18 and 19 show the results for the respective missions. In both missions, the robot started at the same location with identical mission and robot parameters. The robot first performed exploration of the given space and returned back to the start location. As can be seen from Figures 18 and 19, the SDF-NMPC is designed to follow the path given by the planning module more accurately than ExRL. Hence, the robot can take longer trajectories to reach the end of the same path when ExRL is used as compared to SDF-NMPC. However, as shown by the average and max speed, the ExRL policy generates smoother and faster trajectories than SDF-NMPC (partially as a result of the non-smooth paths given by the planning module), thus resulting in similar mission times. It is thus a decision point for the user of the UAstack to select among these two core navigation policies with the SDF-NMPC being a very reasonable choice when following the planner plans closely is desired, while ExRL is particularly relevant when a more loose tracking of these references combined with agile maneuvering is preferred. Figures 18 and 19 show the full map and planning instances from the respective missions, along with the robot in the environment. In both missions, the robot was successfully able to explore the allotted area, avoiding collisions even in the presence of thin obstacles due to the multi-layered safety. Figures 18.2.1-18.2.4 show one such instance where the SDF-NMPC deviates from the path planned by the planning module. Similarly, the ExRL policy successfully guides the robot to the end of the path. At one instance in the mission, shown in Figure 19.1.2, the C-CBF can be seen intervening and correcting the command of the ExRL policy as the robot passes through a narrow opening, thus highlighting the importance of the multi-layered safety approach. As when the navigation module was evaluated separately, it is worth mentioning that when the C-CBF was engaged the distance of the robot centroid from the obstacles was ${0.68}\mathrm{\;m}$ and the component of the velocity towards the obstacle was ${1.10}\mathrm{\;m}/\mathrm{s}$ . Although the engagement of the C-CBF does not strictly imply that the core method would lead to a collision, it indicates the role of such a last-resort safety method to assure what proximity and maneuvering towards the obstacles is considered as acceptable.
森林 AR-2 被部署在挪威特隆赫姆的一片森林中，以探索一个 ${120} \times  {80}\mathrm{\;m}$ 的区域，且高度受限为 ${2.5}\mathrm{\;m}$ 。该森林区域的树木密度不一，在部分位置还有细枝和茂密的枝叶。测试时地面覆盖着积雪。本次测试开展了两项任务，分别以 SDF-NMPC 和 ExRL 作为核心导航策略，并将其输入到 C-CBF。图18和图19分别展示了各任务的结果。在两项任务中，机器人都从相同位置出发，且任务与机器人参数完全一致。机器人先对给定空间进行探索，然后返回到起始位置。如图18和图19所示，SDF-NMPC 的设计使其比 ExRL 更能准确沿着规划模块给出的路径运行。因此，在使用 ExRL 时，为到达同一路径的终点，机器人可能需要更长的轨迹。与此同时，从平均速度和最大速度可以看出，ExRL 策略生成的轨迹更平滑且更快（部分原因在于规划模块给出的路径并不平滑），从而导致任务时间相近。因此，UAstack 的用户需要在这两种核心导航策略之间做出选择：当希望机器人紧密跟随规划器的计划时，SDF-NMPC 是一个非常合理的选择；而当更偏好对这些参考进行更松的跟踪，并结合敏捷机动时，ExRL 则尤为相关。图18和图19展示了各任务对应的完整地图与规划实例，以及机器人在环境中的状态。在两项任务中，机器人都能成功探索分配的区域，即使面对细小障碍也能通过多层次安全机制避免碰撞。图18.2.1-18.2.4 展示了一个实例：SDF-NMPC 偏离了规划模块规划的路径。类似地，ExRL 策略也能将机器人引导到路径终点。在任务的一个实例中（见图19.1.2），可以看到当机器人通过狭窄开口时，C-CBF 会介入并纠正 ExRL 策略的指令，从而凸显多层次安全方法的重要性。与导航模块被单独评估时类似，还值得一提的是：当 C-CBF 启用时，机器人质心到障碍物的距离为 ${0.68}\mathrm{\;m}$ ，而速度中朝向障碍物的分量为 ${1.10}\mathrm{\;m}/\mathrm{s}$ 。尽管 C-CBF 的介入并不严格意味着核心方法一定会导致碰撞，但它表明这种最后手段的安全机制在确保“到障碍物的接近程度以及朝向障碍物的机动被认为是可接受的”方面所发挥的作用。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_26.jpg?x=148&y=147&w=1356&h=1416&r=0"/>



Figure 18. Full-stack evaluation in the forest using AR-2 with SDF-NMPC as the navigation method. The robot was tasked to explore a given area autonomously and return home. The figure shows the full map and planning instances in the mission. As the SDF-NMPC follows the planned path closely, it needs to intervene infrequently.
图18：使用 AR-2 并以 SDF-NMPC 作为导航方法，在森林中进行全栈评估。机器人被要求在自主探索给定区域并返回家。该图展示了任务中的完整地图与规划实例。由于 SDF-NMPC 会紧密跟随规划路径，因此它需要的介入次数很少。


Ship Cargo Hold In the third experiment, the AR-2 was deployed in a cargo hold of an oil tanker ship. In contrast to the previous missions, here the inspection behavior of the planning module is engaged. The dimensions of the cargo hold were ${16} \times  {13} \times  {15}\mathrm{\;m}$ ,however,the mission height was limited to $3\mathrm{\;m}$ for safety considerations. The robot was tasked to explore the cargo hold and inspect the mapped surfaces. The robot started inside the cargo hold with no prior knowledge, performed exploration, and upon completion switched to the inspection behavior. The complete map and instances of the mission along with an image of the robot in the environment is shown in Figure 20. As an exception, it is noted that in this experiment the safety policies in the navigation module were disabled as (a) the environment is not demanding in terms of collision avoidance (one large room with no obstacles inside), and (b) this allows the robot to more flexibly travel outside the depth sensor's FoV in the inspection phase.
船舶货舱 在第三次实验中，AR-2 被部署在一艘油轮的货舱内。与前两次任务不同，这里启用了规划模块的检查行为。货舱的尺寸为 ${16} \times  {13} \times  {15}\mathrm{\;m}$ ，但出于安全考虑，任务高度被限制为 $3\mathrm{\;m}$ 。机器人被要求探索货舱并检查已建图的表面。机器人在没有任何先验知识的情况下从货舱内部开始，先进行探索，完成后切换到检查行为。完整地图以及任务实例，连同机器人在环境中的图像，一并展示于图20。需要特别指出的是，该实验中禁用了导航模块中的安全策略： (a) 环境在碰撞规避方面并不苛刻（舱内只有一个大空间，内部没有障碍物），以及 (b) 这使得机器人在检查阶段可以更灵活地穿行于超出深度传感器 FoV 的区域。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_27.jpg?x=146&y=146&w=1359&h=1418&r=0"/>



Figure 19. Full-stack evaluation in the forest using AR-2 with ExRL as the navigation method. The robot was tasked to explore a given area autonomously and return home. The figure shows the full map and planning instances in the mission. As can be seen, the ExRL is not formulated to follow the planned path strictly, but successfully navigates towards the end of the path, avoiding obstacles.
图19：使用 AR-2 并以 ExRL 作为导航方法，在森林中进行全栈评估。机器人被要求在自主探索给定区域并返回家。该图展示了任务中的完整地图与规划实例。可以看出，ExRL 并非被设计为严格跟随规划路径，但它仍能成功导航到路径终点，并在过程中避开障碍物。


Through these experiments, we demonstrate the importance and the role of the three layers of safety namely a) map-based collision-avoidance, b) depth-driven SDF-NMPC or ExRL policies, and c) the last resort C-CBF. The map-based safety allows longer horizon planning enabling more complex behavior, and is the safety layer doing the majority of the collision-avoidance throughout the missions. The depth-driven navigation policies add an additional safeguard against challenges to map-based safety as documented in this work. Finally, the C-CBF provides formal safety guarantees ensuring that the robot remains safe at all times.
通过这些实验，我们证明并阐明三层安全机制的重要性与作用：a）基于地图的避碰，b）由深度驱动的 SDF-NMPC 或 ExRL 策略，以及 c）最后手段 C-CBF。基于地图的安全机制使得更长视野的规划成为可能，从而支持更复杂的行为；在整个任务过程中，它承担了大部分避碰工作。本文所述的深度驱动导航策略则为基于地图安全应对各类挑战提供了额外保障。最后，C-CBF 提供形式化的安全保证，确保机器人始终保持安全。


#### 4.4.2 Legged
#### 4.4.2 具腿


To demonstrate the performance of the UAstack on ground robots, we deployed GR-1 in two distinct settings a) in an underground mine, and b) inside a university building at NTNU.
为展示 UAstack 在地面机器人上的性能，我们在两个不同场景中部署了 GR-1：a）地下矿井；b）在 NTNU 校内一栋大学楼内。


Underground Mine In the first mission, GR-1 was deployed in another section of the Løkken mine. This section consisted of one mine shaft having narrow passages and areas with gaps on the side, requiring careful planning and locomotion. Figure 21 shows the map and planning instances of the mission. Using the dual map representation (volumetric and elevation map), the UAstack is able to successfully complete the mission. It is noted that here the additional safety layers of the navigation module are not utilized as the commercial ANYmal robot already provides the partially analogous feature of "perceptive locomotion" fusing short-range depth from its all-around depth cameras for traversability-aware near-term navigation.
地下矿井 在第一项任务中，GR-1 被部署在 Løkken 矿的另一处区域。该区域包含一座矿井竖井，通道狭窄，且两侧有空隙区域，因而需要精心规划与运动。图 21 展示了该任务的地图以及规划示例。借助双重地图表示（体积图与高程图），UAstack 能够成功完成任务。需要指出的是，这里未使用导航模块的额外安全层，因为商业 ANYmal 机器人已提供了部分类似的功能“感知式行走”，它会融合来自全向深度摄像头的短距离深度信息，以实现具备可通行性认知的近程导航。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_28.jpg?x=134&y=146&w=1383&h=660&r=0"/>



Figure 20. Deployment of AR-2 in the ship cargo hold. The robot started with no prior information of the environment. It first performed exploration to map the tank. Upon completion, it switched to the inspection behavior, where it viewed the mapped surfaces with the camera sensor at the desired viewing distance.
图 20. AR-2 在船舶货舱中的部署。机器人在没有任何环境先验信息的情况下开始工作。它首先进行探索以绘制油罐地图。完成后，它切换到检查行为，在期望的观察距离下使用相机传感器查看已绘制的表面。


University Building The second mission was conducted inside a building at NTNU. The building consists of two sections a) a large open hall with side offshoots, and b) a section with a network of narrow (width $< {1.5}\mathrm{\;m}$ ) corridors, as can be seen in Figure 22. The robot started in the open hall, and explored it along with the offshoots. Upon completion, the robot repositioned towards the narrow corridor section, explored those, and returned to the start location. This environment presents several challenges to the entire stack, including a) large scale, b) branching corridors, and c) varying environment size. However, the multi-modal SLAM solution provided resilient odometry and consistent maps throughout the mission, as well as the bifurcated architecture of the planning module with traversability-aware planning (exploiting both the volumetric and the elevation maps) lead to successful mission completion.
大学楼 第二项任务在 NTNU 的一栋建筑内进行。该建筑由两个区域组成：a）一处大型开放大厅，带有侧向分支；b）一处由狭窄（宽度 $< {1.5}\mathrm{\;m}$ ）走廊构成的网络区域，如图 22 所示。机器人从开放大厅开始，并连同这些分支进行探索。完成后，机器人重新定位到狭窄走廊区域，探索完毕后返回到起始位置。该环境给整个系统带来多项挑战，包括 a）规模较大，b）走廊分叉，c）环境尺寸不一。然而，所提供的多模态 SLAM 方案在整个任务过程中提供了稳健的里程计与一致的地图；规划模块采用的分支式架构，并通过面向可通行性的规划（同时利用体积图与高程图）实现成功的任务完成。


## 5 Conclusion & Future Work
## 5 结论与未来工作


The UAstack is openly released with the aim of serving as a foundation for a common autonomy blueprint across diverse robot configurations operating in the air, on land, and at sea. Currently, the UAstack supports a wide variety of aerial and ground robot morphologies and enables resilient GNSS-denied, perceptually-degraded localization, mapping, and scene reasoning within target reach, exploration and inspection missions with a key focus on assured safety through multi-layered navigation. Extensive field evaluation results, alongside openly released datasets, allow for its comprehensive evaluation.
UAstack 以开放方式发布，旨在为各类机器人构型在空中、陆地和海上运行提供一套共同的自主蓝图基础。目前，UAstack 支持多种多样的空中与地面机器人形态，并可在目标任务范围内实现稳健的无 GNSS 定位、感知退化条件下的定位、建图与场景推理，重点是通过多层次导航来确保安全。结合广泛的实地评估结果以及公开发布的数据集，可对其进行全面评测。


We seek to collaborate with the research community towards enhancing the reliability and resilient performance of the stack, alongside its extension to different robot morphologies and the incorporation of new behaviors. Future development plans specifically include (a) the support of further morphologies, including highly non-holonomic platforms such as fixed-wing uncrewed aerial vehicles, (b) increased emphasis on navigation within dynamic environments, (c) development of further object-centric behaviors, especially guided by natural language, (d) the fusion of additional modalities and specifically infrared vision, (e) improving the vision fusion into MIMOSA-X to be more tightly coupled analogous to what is already done for LiDAR and radar, (f) enhancing the ExRL toward improved long-horizon capabilities, and (g) extend multilayered safety with traversability-aware reactive modules for ground systems.
我们希望与科研社区合作，提升该堆栈的可靠性与稳健性能，同时扩展到不同的机器人形态，并引入新的行为。未来开发计划具体包括：(a) 支持更多形态，包括高度非完整约束平台，如固定翼无人机；(b) 加强对动态环境中的导航能力的重视；(c) 开发更多以目标为中心的行为，尤其是由自然语言引导的行为；(d) 融合更多模态，特别是红外视觉；(e) 改进将视觉融合到 MIMOSA-X，使其与 LiDAR 和雷达已有的做法一样，更紧密地耦合；(f) 提升 ExRL，以获得更强的长时域能力；以及 (g) 面向地面系统，以可通行性认知的反应模块扩展多层级安全。


Last but not least, we aim to document how lessons learned from the deployment of the stack can lead to certain improvements and adaptations. This currently includes investigations for (a) how to best handle the tradeoff between C-CBF and the exteroceptive SDF-NMPC and ExRL methods, (b) computationally-efficient ways to directly fuse vision features in the SLAM solution of the UAstack, alongside (c) refining the rewards of ExRL to better balance between the ability of the method to negotiate complex environments, and how energetically-efficient the trajectories are.
最后但同样重要的是，我们旨在记录从该堆栈部署中获得的经验如何带来某些改进与适配。目前，这包括对 (a) 如何在 C-CBF 与外观感知方法 SDF-NMPC 和 ExRL 之间实现最佳权衡的研究；(b) 以计算高效的方式在 UAstack 的 SLAM 解算中直接融合视觉特征；以及 (c) 优化 ExRL 的奖励，使该方法在应对复杂环境的能力与轨迹的能效之间取得更好的平衡。


## Acknowledgements
## 致谢


We would like to acknowledge Statens Vegvesen for enabling us to perform tests in Runehamar, Vestland Fylkeskommune for allowing us to perform experiments in the Fyllingsdal sykkeltunnel, Leica Geosystems for providing the Robot Operating System (ROS) compatible MS60 and AP20 setup for collection of ground truth, as well as Orkla Industrimuseum for facilitating the tests in the Løkken Mine.
我们谨向 Statens Vegvesen 致谢，感谢其使我们能够在 Runehamar 进行测试；向 Vestland Fylkeskommune 致谢，允许我们在 Fyllingsdal 自行车隧道开展实验；向 Leica Geosystems 致谢，提供与机器人操作系统（ROS）兼容的 MS60 和 AP20 配置以采集地面真实数据；同时也感谢 Orkla Industrimuseum 促成了在 Løkken 采矿场的测试。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_29.jpg?x=140&y=147&w=1369&h=406&r=0"/>



Figure 21. Full-stack evaluation in the Lokken underground mine using the GR-1 legged robot. The mission was conducted in one of the mine shafts, presenting narrow cross-section at times, and gaps on the side. Due to the dual map representation (volumetric and elevation maps), the robot was successfully able to handle these challenges completing the mission.
图 21. 在 Løkken 地下矿井中使用 GR-1 轮式/腿式机器人进行全栈评估。任务在矿井竖井之一中开展，部分时段横截面较窄，且两侧存在空隙。由于采用双重地图表示（体积图与高度图），机器人成功应对了这些挑战，并完成了任务。


<img src="https://cdn.noedgeai.com/bo_d8lcm4c91nqc738unaa0_29.jpg?x=134&y=674&w=1384&h=1002&r=0"/>



Figure 22. Full-stack evaluation was conducted in a university building using the GR-1 legged robot. The robot was tasked with exploring the entire ground floor, which featured both open spaces and narrow corridors. The figure shows the complete map along with the planning instances from the mission.
图 22. 使用 GR-1 轮式/腿式机器人在大学建筑中开展全栈评估。机器人被要求探索整个一层楼，该楼既有开阔空间，也有狭窄走廊。图中展示了完整地图以及任务过程中的规划实例。


## Author contributions
## 作者贡献


- Mihir Dharmadhikari: Contributed in the formulation of the idea, the UAstack architecture, and all the evaluations. Furthermore, he is the core developer of the planning module.
- Mihir Dharmadhikari：参与了这一想法、UAstack 架构及所有评测的构思。此外，他还是规划模块的核心开发者。


- Nikhil Khedekar: Contributed in the formulation of the idea, the UAstack architecture, and all the evaluations. Furthermore, he is a core developer of the multi-modal SLAM, specifically the LiDAR and Vision modalities.
- Nikhil Khedekar：参与了这一想法、UAstack 架构及所有评测的构思。此外，他还是多模态 SLAM 的核心开发者，具体负责 LiDAR 和视觉模态。


- Mihir Kulkarni: Contributed in the formulation of the idea, the UAstack architecture, and all the evaluations. Furthermore, he is the core developer of Exteroceptive Deep RL navigation policy.
- Mihir Kulkarni：参与了这一想法、UAstack 架构及所有评测的构思。此外，他还是外感知深度强化学习导航策略的核心开发者。


- Morten Nissov: Contributed in the formulation of the idea, the UAstack architecture, and all the evaluations. Furthermore, he is a core developer of the multi-modal SLAM, specifically the Radar modality.
- Morten Nissov：参与了这一想法、UAstack 架构及所有评测的构思。此外，他还是多模态 SLAM 的核心开发者，具体负责雷达模态。


- Martin Jacquet: He is the core developer of the Neural SDF-NMPC and co-developer of the Composite CBF-based Safety Filter, and contributed towards their integration in the UAstack.
- Martin Jacquet：他是 Neural SDF-NMPC 的核心开发者和基于复合 CBF 的安全过滤器的共同开发者，并为它们在 UAstack 中的集成做出了贡献。


- Angelos Zacharia: He is the co-developer of the planning module. He contributed towards its integration in the UAstack and the legged robot experiments.
- Angelos Zacharia：他是规划模块的共同开发者。他为其在 UAstack 中的集成以及足式机器人实验做出了贡献。


- Marvin Harms: He is the core developer of the Composite CBF-based Safety Filter. He contributed towards its integration in the UAstack and the evaluations of the safety policies.
- Marvin Harms：他是基于复合 CBF 的安全过滤器的核心开发者。他为其在 UAstack 中的集成以及安全策略的评测做出了贡献。


- Albert Gassol Puigjaner: He is the core developer of the VLM-based reasoning part of the UAstack and contributed towards its integration in the UAstack.
- Albert Gassol Puigjaner：他是 UAstack 中基于 VLM 的推理部分的核心开发者，并为其在 UAstack 中的集成做出了贡献。


- Philipp Weiss: Contributed towards the development of the hardware setup and conducting the field experiments.
- Philipp Weiss：他为硬件搭建和现场实验的开展做出了贡献。


- Kostas Alexis: Contributed in the formulation of the idea, the UAstack architecture, and planning for all evaluations. Furthermore, he contributed to the planning, problem formulation, and algorithmic approach of each module in the UAstack.
- Kostas Alexis：参与了这一想法、UAstack 架构及所有评测规划的构思。此外，他还为 UAstack 中各模块的规划、问题定义和算法方法做出了贡献。


All authors contributed to the writing of this manuscript.
所有作者都为本文的写作做出了贡献。


## Statements and declarations
## 声明与陈述


## Ethical considerations
## 伦理考虑


This article does not contain any studies with human or animal participants.
本文不包含任何涉及人类或动物参与者的研究。


## Consent to participate
## 参与同意


Not applicable.
不适用。


## Consent for publication
## 出版同意


Not applicable.
不适用。


## Declaration of conflicting interests
## 利益冲突声明


The author(s) declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.
作者（们）声明，就本研究、署名和/或本文发表而言，不存在任何潜在的利益冲突。


## Funding
## 资助


The author(s) disclosed receipt of the following financial support for the research, authorship, and/or publication of this article: This work was supported by European Commission Horizon Europe grant agreements a) SPEAR (EC 101119774), b) DIGIFOREST (EC 101070405), c) SYNERGISE (EC 101121321), and d) AUTOASSESS (EC 101120732).
作者已披露其在本研究、署名和/或本文发表过程中获得以下财政支持：本工作得到欧洲委员会“地平线欧洲”资助协议的支持：a) SPEAR（EC 101119774），b) DIGIFOREST（EC 101070405），c) SYNERGISE（EC 101121321），以及 d) AUTOASSESS（EC 101120732）。


## References
## 参考文献


Agha A, Otsu K, Morrell B, Fan DD, Thakker R, Santamaria-Navarro A, Kim SK, Bouman A, Lei X, Edlund J et al. (2021) Nebula: Quest for robotic autonomy in challenging environments; team costar at the darpa subterranean challenge. arXiv preprint arXiv:2103.11470 .
Agha A, Otsu K, Morrell B, Fan DD, Thakker R, Santamaria-Navarro A, Kim SK, Bouman A, Lei X, Edlund J 等 (2021) Nebula：在具有挑战的环境中追求机器人自主；team costar在DARPA地下挑战赛中的表现。arXiv预印本 arXiv:2103.11470 .


AirLab C (2025) Airstack. https://github.com/ castacks/AirStack. Accessed: 2025-02-04.
AirLab C（2025）Airstack. https://github.com/ castacks/AirStack. 访问日期：2025-02-04.


Alan A, Taylor AJ, He CR, Ames AD and Orosz G (2023) Control barrier functions and input-to-state safety with application to automated vehicles. IEEE Transactions on Control Systems Technology 31(6): 2744-2759. DOI:10.1109/TCST.2023. 3286090.
Alan A, Taylor AJ, He CR, Ames AD 和 Orosz G（2023）控制屏障函数以及输入到状态的安全性，并应用于自动驾驶车辆。IEEE控制系统技术汇刊 31(6)：2744-2759. DOI:10.1109/TCST.2023. 3286090.


Andersson J, Gillis J, Horn G, Rawlings J and Diehl M (2018) Casadi-a software framework for nonlinear optimization and optimal control. Mathematical Programming Computation 11(1): 1-36.
Andersson J, Gillis J, Horn G, Rawlings J 和 Diehl M（2018）Casadi—用于非线性优化与最优控制的软件框架。数学规划计算 11(1)：1-36.


ArduPilot Dev Team (2024) Ardupilot documentation. URL https://ardupilot.org/ardupilot/.Accessed: 2026-03-26.
ArduPilot开发团队（2024）Ardupilot文档。网址 https://ardupilot.org/ardupilot/.访问日期：2026-03-26.


Auto A (2017) apollo. URL https://github.com/ ApolloAuto/apollo. Accessed 2026-04-09.
Auto A（2017）apollo。网址 https://github.com/ ApolloAuto/apollo. 访问日期：2026-04-09.


Autoware Foundation T (2021) autoware_universe. URL https://github.com/autowarefoundation/ autoware_universe. Accessed 2026-04-09.
Autoware基金会T（2021）autoware_universe。网址 https://github.com/autowarefoundation/ autoware_universe. 访问日期：2026-04-09.


Baca T, Petrlik M, Vrba M, Spurny V, Penicka R, Hert D and Saska M (2021) The mrs uav system: Pushing the frontiers of reproducible research, real-world deployment, and education with autonomous unmanned aerial vehicles. Journal of Intelligent & Robotic Systems 102(1): 26.
Baca T, Petrlik M, Vrba M, Spurny V, Penicka R, Hert D 和 Saska M（2021）mrs uav系统：通过自主无人机推动可重复研究、真实部署与教育的前沿。智能与机器人系统期刊 102(1)：26.


Bloesch M, Burri M, Omari S, Hutter M and Siegwart R (2017) Iterated extended kalman filter based visual-inertial odometry using direct photometric feedback. The International Journal of Robotics Research 36(10): 1053-1072. DOI:10.1177/ 0278364917728574.
Bloesch M, Burri M, Omari S, Hutter M 和 Siegwart R（2017）基于直接光度反馈的迭代扩展卡尔曼滤波视觉-惯性里程计。国际机器人研究期刊 36(10)：1053-1072. DOI:10.1177/ 0278364917728574.


Campos C, Elvira R, Rodríguez JJG, M Montiel JM and D Tardós J (2021) Orb-slam3: An accurate open-source library for visual, visual-inertial, and multimap slam. IEEE Transactions on Robotics 37(6): 1874-1890. DOI:10.1109/TRO.2021. 3075644.
Campos C, Elvira R, Rodríguez JJG, M Montiel JM 和 D Tardós J（2021）Orb-slam3：用于视觉、视觉-惯性以及多地图SLAM的准确开源库。IEEE机器人与自动化汇刊 37(6)：1874-1890. DOI:10.1109/TRO.2021. 3075644.


Carrillo H, Reid I and Castellanos JA (2012) On the comparison of uncertainty criteria for active slam. In: 2012 IEEE International Conference on Robotics and Automation. pp. 2080-2087. DOI:10.1109/ICRA.2012.6224890.
Carrillo H, Reid I 和 Castellanos JA（2012）关于主动SLAM中不确定性准则的比较。载于：2012 IEEE国际机器人与自动化会议。第2080-2087页。DOI:10.1109/ICRA.2012.6224890.


Chang Y, Ebadi K, Denniston CE, Ginting MF, Rosinol A, Reinke A, Palieri M, Shi J, Chatterjee A, Morrell B et al. (2022) Lamp 2.0: A robust multi-robot slam system for operation in challenging large-scale underground environments. IEEE Robotics and Automation Letters 7(4): 9175-9182.
Chang Y, Ebadi K, Denniston CE, Ginting MF, Rosinol A, Reinke A, Palieri M, Shi J, Chatterjee A, Morrell B 等（2022）Lamp 2.0：面向在具有挑战的大规模地下环境中运行的鲁棒多机器人SLAM系统。IEEE机器人与自动化快报 7(4)：9175-9182.


Chung TH, Orekhov V and Maio A (2023) Into the robotic depths: Analysis and insights from the darpa subterranean challenge. Annual Review of Control, Robotics, and Autonomous Systems 6(1): 477-502.
Chung TH, Orekhov V 和 Maio A（2023）走进机器人的深处：来自DARPA地下挑战赛的分析与洞见。控制、机器人与自主系统年度综述 6(1)：477-502.


Datar A, Pokhrel A, Nazeri M, Rao MB, Rangwala H, Pan C, Zhang Y, Harrison A, Wigness M, Osteen PR et al. (2025) M2p2: A multi-modal passive perception dataset for off-road mobility in extreme low-light conditions. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, pp. 13690-13696.
Datar A, Pokhrel A, Nazeri M, Rao MB, Rangwala H, Pan C, Zhang Y, Harrison A, Wigness M, Osteen PR 等（2025）M2p2：用于极低照度条件下越野移动的多模态被动感知数据集。载于：2025 IEEE/RSJ智能机器人与系统国际会议（IROS）。IEEE，第13690-13696页.


Dellaert F and GTSAM Contributors (2022) borglab/gtsam. DOI: 10.5281/zenodo.5794541. URL https://github.com/ borglab/gtsam.
Dellaert F 和GTSAM贡献者（2022）borglab/gtsam。DOI: 10.5281/zenodo.5794541. 网址 https://github.com/ borglab/gtsam.


Dharmadhikari M and Alexis K (2025) Semantics-aware predictive inspection path planning. IEEE Transactions on Field Robotics
Dharmadhikari M 和 Alexis K（2025）语义感知的预测性巡检路径规划。IEEE《现场机器人》汇刊


Ebadi K, Bernreiter L, Biggie H, Catt G, Chang Y, Chatterjee A, Denniston CE, Deschênes SP, Harlow K, Khattak S et al. (2023) Present and future of slam in extreme environments: The darpa subt challenge. IEEE Transactions on Robotics 40: 936-959.
Ebadi K、Bernreiter L、Biggie H、Catt G、Chang Y、Chatterjee A、Denniston CE、Deschênes SP、Harlow K、Khattak S 等（2023）极端环境中 SLAM 的现在与未来：DARPA SubT 挑战。IEEE《机器人学汇刊》40：936-959。


Fankhauser P, Bloesch M, Gehring C, Hutter M and Siegwart R (2014) Robot-centric elevation mapping with uncertainty estimates. In: International Conference on Climbing and Walking Robots (CLAWAR).
Fankhauser P、Bloesch M、Gehring C、Hutter M 和 Siegwart R（2014）带不确定性估计的机器人中心高程地图。在：爬行与步行机器人国际会议（CLAWAR）。


Fankhauser P, Bloesch M and Hutter M (2018) Probabilistic terrain mapping for mobile robots with uncertain localization. IEEE Robotics and Automation Letters 3(4): 3019-3026. DOI: 10.1109/LRA.2018.2849506.
Fankhauser P、Bloesch M 和 Hutter M（2018）面向具有不确定定位的移动机器人的概率地形地图。IEEE《机器人与自动化快报》3(4)：3019-3026。DOI：10.1109/LRA.2018.2849506。


Farrell J (2008) Aided Navigation: GPS with High Rate Sensors. 1 edition. USA: McGraw-Hill, Inc. ISBN 0071493298.
Farrell J（2008）辅助导航：带高频传感器的 GPS。第 1 版。美国：McGraw-Hill, Inc. ISBN 0071493298。


Fernandez-Cortizas M, Molina M, Arias-Perez P, Perez-Segui R, Perez-Saura D and Campoy P (2023) Aerostack2: A software framework for developing multi-robot aerial systems. arXiv preprint arXiv:2303.18237 .
Fernandez-Cortizas M、Molina M、Arias-Perez P、Perez-Segui R、Perez-Saura D 和 Campoy P（2023）Aerostack2：用于开发多机器人空中系统的软件框架。arXiv 预印本 arXiv:2303.18237。


Foehn P, Kaufmann E, Romero A, Penicka R, Sun S, Bauersfeld L, Laengle T, Cioffi G, Song Y, Loquercio A et al. (2022) Agilicious: Open-source and open-hardware agile quadrotor for vision-based flight. Science robotics 7(67): eabl6259.
Foehn P、Kaufmann E、Romero A、Penicka R、Sun S、Bauersfeld L、Laengle T、Cioffi G、Song Y、Loquercio A 等（2022）Agilicious：面向基于视觉飞行的开源开硬件敏捷四旋翼。Science Robotics 7(67)：eabl6259。


Forster C, Carlone L, Dellaert F and Scaramuzza D (2017) On-manifold preintegration for real-time visual-inertial odometry. IEEE Transactions on Robotics 33(1): 1-21. DOI:10.1109/ TRO.2016.2597321.
Forster C、Carlone L、Dellaert F 和 Scaramuzza D（2017）流形上预积分用于实时视觉惯性里程计。IEEE《机器人学汇刊》33(1)：1-21。DOI:10.1109/ TRO.2016.2597321。


Geneva P, Eckenhoff K, Lee W, Yang Y and Huang G (2020a) OpenVINS: A research platform for visual-inertial estimation. In: Proc. of the IEEE International Conference on Robotics and Automation. Paris, France.
Geneva P、Eckenhoff K、Lee W、Yang Y 和 Huang G（2020a）OpenVINS：用于视觉惯性估计的研究平台。在：IEEE 机器人与自动化国际会议论文集。法国巴黎。


Geneva P, Eckenhoff K, Lee W, Yang Y and Huang G (2020b) Openvins: A research platform for visual-inertial estimation. In: 2020 IEEE International Conference on Robotics and Automation (ICRA). pp. 4666-4672. DOI:10. 1109/ICRA40945.2020.9196524.
Geneva P、Eckenhoff K、Lee W、Yang Y 和 Huang G（2020b）Openvins：用于视觉惯性估计的研究平台。在：2020 IEEE 机器人与自动化国际会议（ICRA）。第 4666-4672 页。DOI:10. 1109/ICRA40945.2020.9196524。


Giftthaler M, Neunert M, Stäuble M and Buchli J (2018) The control toolbox-an open-source c++ library for robotics, optimal and model predictive control. In: 2018 IEEE international conference on simulation, modeling, and programming for autonomous robots (SIMPAR). IEEE, pp. 123-129.
Giftthaler M、Neunert M、Stäuble M 和 Buchli J（2018）控制工具箱——一个面向机器人、最优控制和模型预测控制的开源 C++ 库。在：2018 IEEE 自主机器人仿真、建模与编程国际会议（SIMPAR）。IEEE，第 123-129 页。


Goodin C, Moore MN, Carruth DW, Hudson CR, Cagle LD, Wapnick S and Jayakumar P (2024) The nature autonomy stack: an open-source stack for off-road navigation. In: Unmanned Systems Technology XXVI, volume 13055. SPIE, pp. 8-17.
Goodin C、Moore MN、Carruth DW、Hudson CR、Cagle LD、Wapnick S 和 Jayakumar P（2024）Nature Autonomy Stack：一个用于越野导航的开源栈。在：无人系统技术 XXVI，第 13055 卷。SPIE，第 8-17 页。


Greeff M and Schoellig AP (2018) Flatness-based model predictive control for quadrotor trajectory tracking. In: 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 6740-6745. DOI:10.1109/IROS.2018.8594012.
Greeff M 和 Schoellig AP（2018）基于平坦性的四旋翼轨迹跟踪模型预测控制。在：2018 IEEE/RSJ 智能机器人与系统国际会议（IROS）。第 6740-6745 页。DOI:10.1109/IROS.2018.8594012。


Grupp M (2017) evo: Python package for the evaluation of odometry and slam. https://github.com/MichaelGrupp/ evo.
Grupp M（2017）evo：用于里程计和 SLAM 评估的 Python 包。https://github.com/MichaelGrupp/evo。


Harlow K, Jang H, Barfoot TD, Kim A and Heckman C (2024) A new wave in robotics: Survey on recent mmwave radar applications in robotics. IEEE Transactions on Robotics 40: 4544-4560. DOI:10.1109/TRO.2024.3463504.
Harlow K、Jang H、Barfoot TD、Kim A 和 Heckman C（2024）机器人学中新一波浪潮：近期毫米波雷达在机器人中的应用综述。IEEE《机器人学汇刊》40：4544-4560。DOI:10.1109/TRO.2024.3463504。


Harms M, Jacquet M and Alexis K (2025) Safe Quadrotor Navigation Using Composite Control Barrier Functions. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 6343-6349. DOI:10.1109/ICRA55743.2025. 11127368. URL https://ieeexplore.ieee.org/ document/11127368.
Harms M、Jacquet M 和 Alexis K（2025）使用复合控制屏障函数的安全四旋翼导航。在：2025 IEEE 机器人与自动化国际会议（ICRA）。第 6343-6349 页。DOI:10.1109/ICRA55743.2025. 11127368。URL https://ieeexplore.ieee.org/document/11127368。


Huber PJ (1964) Robust estimation of a location parameter. The Annals of Mathematical Statistics 35(1): 73-101. URL http: //www.jstor.org/stable/2238020.
Huber PJ（1964）位置参数的稳健估计。数理统计年鉴 35（1）：73-101。URL http: //www.jstor.org/stable/2238020。


Jacquet M, Harms M and Alexis K (2025) Neural NMPC through Signed Distance Field Encoding for Collision Avoidance. DOI: 10.1177/02783649251401223.URL http://arxiv.org/ abs/2511.21312. ArXiv:2511.21312 [cs].
Jacquet M、Harms M 和 Alexis K（2025）通过符号距离场编码实现神经 NMPC，用于避碰。DOI：10.1177/02783649251401223.URL http://arxiv.org/ abs/2511.21312。ArXiv：2511.21312 [cs]。


Jung HY, Paek DH and Kong SH (2025) Open-source autonomous driving software platforms: Comparison of autoware and apollo. arXiv preprint arXiv:2501.18942 .
Jung HY、Paek DH 和 Kong SH（2025）开源自动驾驶软件平台：autoware 与 apollo 的对比。arXiv 预印本 arXiv：2501.18942 。


Kaess M, Johannsson H, Roberts R, Ila V, Leonard J and Dellaert F (2011) iSAM2: Incremental smoothing and mapping with fluid relinearization and incremental variable reordering. In: 2011 IEEE International Conference on Robotics and Automation. IEEE, pp. 3281-3288. DOI:10.1109/icra.2011.5979641.
Kaess M、Johannsson H、Roberts R、Ila V、Leonard J 和 Dellaert F（2011）iSAM2：增量平滑与建图，采用流畅的重线性化和增量变量重排序。见：2011 年 IEEE 国际机器人与自动化会议。IEEE，第 3281-3288 页。DOI：10.1109/icra.2011.5979641。


Kamel M, Stastny T, Alexis K and Siegwart R (2017) Model Predictive Control for Trajectory Tracking of Unmanned Aerial Vehicles Using Robot Operating System. In: Koubaa A (ed.) Robot Operating System (ROS), volume 707. Cham: Springer International Publishing. ISBN 978-3-319-54926-2 978-3- 319-54927-9, pp. 3-39. DOI:10.1007/978-3-319-54927-9_ 1. URL http://link.springer.com/10.1007/ 978-3-319-54927-9_1.
Kamel M、Stastny T、Alexis K 和 Siegwart R（2017）基于机器人操作系统（ROS）的无人机轨迹跟踪模型预测控制。见：Koubaa A（编）机器人操作系统（ROS），第 707 卷。Cham：Springer International Publishing。ISBN 978-3-319-54926-2 978-3- 319-54927-9，第 3-39 页。DOI：10.1007/978-3-319-54927-9_ 1。URL http://link.springer.com/10.1007/ 978-3-319-54927-9_1。


Kato S, Tokunaga S, Maruyama Y, Maeda S, Hirabayashi M, Kitsukawa Y, Monrroy A, Ando T, Fujii Y and Azumi T (2018) Autoware on board: Enabling autonomous vehicles with embedded systems. In: 2018 ACM/IEEE 9th International Conference on Cyber-Physical Systems (ICCPS). IEEE, pp. 287-296.
Kato S、Tokunaga S、Maruyama Y、Maeda S、Hirabayashi M、Kitsukawa Y、Monrroy A、Ando T、Fujii Y 和 Azumi T（2018）车载 Autoware：借助嵌入式系统实现自动驾驶。见：2018 年 ACM/IEEE 第 9 届网络物理系统国际会议（ICCPS）。IEEE，第 287-296 页。


Khattak S, Nguyen H, Mascarich F, Dang T and Alexis K (2020) Complementary Multi-Modal Sensor Fusion for Resilient Robot Pose Estimation in Subterranean Environments. In: 2020 International Conference on Unmanned Aircraft Systems (ICUAS). Athens, Greece: IEEE. ISBN 978-1-72814- 278-4, pp. 1024-1029. DOI:10.1109/ICUAS48674.2020. 9213865. URL https://ieeexplore.ieee.org/ document/9213865/.
Khattak S、Nguyen H、Mascarich F、Dang T 和 Alexis K（2020）用于地下环境中稳健机器人位姿估计的互补多模态传感器融合。见：2020 年国际无人驾驶飞行器系统会议（ICUAS）。希腊雅典：IEEE。ISBN 978-1-72814- 278-4，第 1024-1029 页。DOI：10.1109/ICUAS48674.2020. 9213865。URL https://ieeexplore.ieee.org/ document/9213865/。


Khedekar N and Alexis K (2025) PG-LIO: Photometric-Geometric fusion for Robust LiDAR-Inertial Odometry. DOI:10. 48550/arXiv.2506.18583. URL http://arxiv.org/ abs/2506.18583. ArXiv:2506.18583 [cs].
Khedekar N 和 Alexis K（2025）PG-LIO：用于稳健激光雷达-惯性里程计的光度-几何融合。DOI：10. 48550/arXiv.2506.18583。URL http://arxiv.org/ abs/2506.18583。ArXiv：2506.18583 [cs]。


Khedekar N, Kulkarni M and Alexis K (2022) MIMOSA: A Multi-Modal SLAM Framework for Resilient Autonomy against Sensor Degradation. In: 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Kyoto, Japan: IEEE. ISBN 978-1-66547-927-1, pp. 7153-7159. DOI:10.1109/IROS47612.2022.9981108. URL https:// ieeexplore.ieee.org/document/9981108/.
Khedekar N、Kulkarni M 和 Alexis K（2022）MIMOSA：面向传感器退化的、实现稳健自主的多模态 SLAM 框架。见：2022 年 IEEE/RSJ 智能机器人与系统国际会议（IROS）。日本京都：IEEE。ISBN 978-1-66547-927-1，第 7153-7159 页。DOI：10.1109/IROS47612.2022.9981108。URL https:// ieeexplore.ieee.org/document/9981108/。


Koenig N and Howard A (2004) Design and use paradigms for gazebo, an open-source multi-robot simulator. In: IEEE/RSJ International Conference on Intelligent Robots and Systems. Sendai, Japan, pp. 2149-2154.
Koenig N 和 Howard A（2004）为 gazebo 设计与使用范式：一种开源多机器人仿真器。见：IEEE/RSJ 智能机器人与系统国际会议。日本仙台，第 2149-2154 页。


Koide K, Yokozuka M, Oishi S and Banno A (2024) Glim: 3d range-inertial localization and mapping with gpu-accelerated scan matching factors. Robotics and Autonomous Systems 179: 104750. DOI:https://doi.org/10.1016/j.robot.2024.104750.URL https://www.sciencedirect.com/science/ article/pii/S0921889024001349.
Koide K、Yokozuka M、Oishi S 和 Banno A（2024）Glim：基于 GPU 加速扫描匹配因子的三维距离-惯性定位与建图。机器人与自主系统 179：104750。DOI：https://doi.org/10.1016/j.robot.2024.104750.URL https://www.sciencedirect.com/science/ article/pii/S0921889024001349。


Kottege N, Williams J, Tidd B, Talbot F, Steindl R, Cox M, Frousheger D, Hines T, Pitt A, Tam B et al. (2024) Heterogeneous robot teams with unified perception and autonomy: How team csiro data61 tied for the top score at the darpa subterranean challenge. Field Robotics 4: 313-359.
Kottege N, Williams J, Tidd B, Talbot F, Steindl R, Cox M, Frousheger D, Hines T, Pitt A, Tam B 等（2024）统一感知与自主的异构机器人团队：Team CSIRO Data61 如何在 DARPA 地下挑战赛中并列最高分。Field Robotics 4: 313-359。


Koubâa A, Allouch A, Alajlan M, Javed Y, Belghith A and Khalgui M (2019) Micro air vehicle link (mavlink) in a nutshell: A survey. IEEE Access 7: 87658-87680.
Koubâa A, Allouch A, Alajlan M, Javed Y, Belghith A 和 Khalgui M（2019）简明微型飞行器链路（MAVLink）：综述。IEEE Access 7: 87658-87680。


Kulkarni M and Alexis K (2023) Task-Driven Compression for Collision Encoding Based on Depth Images. In: Bebis G, Ghiasi G, Fang Y, Sharf A, Dong Y, Weaver C, Leo Z, LaViola Jr JJ and Kohli L (eds.) Advances in Visual Computing. Cham: Springer Nature Switzerland. ISBN 978-3-031-47966- 3, pp. 259-273. DOI:10.1007/978-3-031-47966-3_20.
Kulkarni M 和 Alexis K（2023）基于深度图像的碰撞编码任务驱动压缩。见：Bebis G, Ghiasi G, Fang Y, Sharf A, Dong Y, Weaver C, Leo Z, LaViola Jr JJ 和 Kohli L（编）《视觉计算进展》。Cham：Springer Nature Switzerland。ISBN 978-3-031-47966-3，pp. 259-273。DOI:10.1007/978-3-031-47966-3_20。


Kulkarni M and Alexis K (2024) Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). pp. 15781-15788. DOI:10.1109/ICRA57147. 2024.10610287. URL https://ieeexplore.ieee.org/document/10610287.
Kulkarni M 和 Alexis K（2024）利用深度碰撞编码实现无碰撞飞行的强化学习。见：2024 IEEE 国际机器人与自动化会议（ICRA）。pp. 15781-15788。DOI:10.1109/ICRA57147.2024.10610287。URL https://ieeexplore.ieee.org/document/10610287。


Kulkarni M, Dharmadhikari M, Khedekar N, Nissov M, Singh M, Weiss P and Alexis K (2025a) Unipilot: Enabling gps-denied autonomy across embodiments. arXiv preprint arXiv:2509.11793 .
Kulkarni M, Dharmadhikari M, Khedekar N, Nissov M, Singh M, Weiss P 和 Alexis K（2025a）Unipilot：实现跨载体的无 GPS 自主。arXiv 预印本 arXiv:2509.11793。


Kulkarni M, Nguyen H and Alexis K (2023) Semantically-enhanced deep collision prediction for autonomous navigation using aerial robots. In: 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, pp. 3056-3063.
Kulkarni M, Nguyen H 和 Alexis K（2023）使用空中机器人进行语义增强的深度碰撞预测，用于自主导航。见：2023 IEEE/RSJ 国际智能机器人与系统会议（IROS）。IEEE，pp. 3056-3063。


Kulkarni M, Rehberg W and Alexis K (2025b) Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots. IEEE Robotics and Automation Letters 10(4): 4093-4100. DOI:10.1109/LRA.2025.3548507. URL https: //ieeexplore.ieee.org/document/10910148/.
Kulkarni M, Rehberg W 和 Alexis K（2025b）Aerial Gym Simulator：用于高度并行化空中机器人仿真的框架。IEEE Robotics and Automation Letters 10(4): 4093-4100。DOI:10.1109/LRA.2025.3548507。URL https://ieeexplore.ieee.org/document/10910148/。


Lee T, Leok M and McClamroch NH (2010) Geometric tracking control of a quadrotor UAV on SE(3). In: 49th IEEE Conference on Decision and Control (CDC). Atlanta, GA: IEEE. ISBN 978-1-4244-7745-6 978-1-4244-7746-3, pp. 5420-5425. DOI:10.1109/CDC.2010.5717652. URL http: //ieeexplore.ieee.org/document/5717652/.
Lee T, Leok M 和 McClamroch NH（2010）SE(3) 上四旋翼无人机的几何跟踪控制。见：第 49 届 IEEE 决策与控制会议（CDC）。Atlanta, GA：IEEE。ISBN 978-1-4244-7745-6 978-1-4244-7746-3，pp. 5420-5425。DOI:10.1109/CDC.2010.5717652。URL http://ieeexplore.ieee.org/document/5717652/。


Likhachev M, Gordon GJ and Thrun S (2003) Ara*: Anytime a* with provable bounds on sub-optimality. Advances in neural information processing systems 16.
Likhachev M, Gordon GJ 和 Thrun S（2003）Ara*：带有可证明次优界的 Anytime A*。神经信息处理系统进展 16。


Macenski S, Martín F, White R and Ginés Clavero J (2020) The marathon 2: A navigation system. In: 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). URL https://github.com/ros-planning/ navigation2.
Macenski S, Martín F, White R 和 Ginés Clavero J（2020）Marathon 2：一种导航系统。见：2020 IEEE/RSJ 国际智能机器人与系统会议（IROS）。URL https://github.com/ros-planning/navigation2。


Makoviichuk D and Makoviychuk V (2021) rl-games: A high-performance framework for reinforcement learning. https: //github.com/Denys88/rl_games.
Makoviichuk D 和 Makoviychuk V（2021）rl-games：一个高性能强化学习框架。https://github.com/Denys88/rl_games。


Meier L, Honegger D and Pollefeys M (2015) Px4: A node-based multithreaded open source robotics framework for deeply embedded platforms. In: 2015 IEEE international conference on robotics and automation (ICRA). IEEE, pp. 6235-6240.
Meier L, Honegger D 和 Pollefeys M（2015）Px4：面向深度嵌入式平台的基于节点的多线程开源机器人框架。见：2015 IEEE 国际机器人与自动化会议（ICRA）。IEEE，pp. 6235-6240。


Mellinger D and Kumar V (2011) Minimum snap trajectory generation and control for quadrotors. In: 2011 IEEE international conference on robotics and automation. Ieee, pp. 2520-2525.
Mellinger D 和 Kumar V（2011）四旋翼最小跃度轨迹生成与控制。见：2011 IEEE 国际机器人与自动化会议。Ieee，pp. 2520-2525。


Miki T, Lee J, Hwangbo J, Wellhausen L, Koltun V and Hutter M (2022) Learning robust perceptive locomotion for quadrupedal robots in the wild. Science Robotics 7(62): eabk2822. DOI:10. 1126/scirobotics.abk2822. URL https://www.science.org/doi/abs/10.1126/scirobotics.abk2822.
Miki T，Lee J，Hwangbo J，Wellhausen L，Koltun V 和 Hutter M（2022）在野外环境中为四足机器人学习稳健的感知式行走。Science Robotics 7(62)：eabk2822。DOI：10. 1126/scirobotics.abk2822。URL https://www.science.org/doi/abs/10.1126/scirobotics.abk2822。


Misyats N, Harms M, Nissov M, Jacquet M and Alexis K (2025) Embedded safe reactive navigation for multirotors systems using control barrier functions. In: 2025 International Conference on Unmanned Aircraft Systems (ICUAS). IEEE, pp. 697-704.
Misyats N，Harms M，Nissov M，Jacquet M 和 Alexis K（2025）使用控制障碍函数为多旋翼系统实现嵌入式安全的反应式导航。载于：2025 国际无人机系统会议（ICUAS）。IEEE，第697-704页。


Mohta K, Watterson M, Mulgaonkar Y, Liu S, Qu C, Makineni A, Saulnier K, Sun K, Zhu A, Delmerico J, Thakur D, Karydis K, Atanasov N, Loianno G, Scaramuzza D, Daniilidis K, Taylor CJ and Kumar V (2018) Fast, autonomous flight in gps-denied and cluttered environments. Journal of Field Robotics 35(1): 101-120.
Mohta K，Watterson M，Mulgaonkar Y，Liu S，Qu C，Makineni A，Saulnier K，Sun K，Zhu A，Delmerico J，Thakur D，Karydis K，Atanasov N，Loianno G，Scaramuzza D，Daniilidis K，Taylor CJ 和 Kumar V（2018）在无GPS且复杂环境中的快速自主飞行。Journal of Field Robotics 35(1)：101-120。


Nanayakkara R, Ames AD and Tabuada P (2025) Safety under state uncertainty: Robustifying control barrier functions. URL https://arxiv.org/abs/2508.17226.
Nanayakkara R，Ames AD 和 Tabuada P（2025）在状态不确定性下的安全性：强化控制障碍函数。URL https://arxiv.org/abs/2508.17226。


NASA CoSTAR (2022) NeBula Autonomy — github.com. https://github.com/NeBula-Autonomy.[Accessed 09-04-2026].
NASA CoSTAR（2022）NeBula Autonomy — github.com. https://github.com/NeBula-Autonomy.[Accessed 09-04-2026]。


Nemiroff R, Chen K and Lopez BT (2023) Joint on-manifold gravity and accelerometer intrinsics estimation for inertially aligned mapping. In: 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 1388-1394. DOI:10.1109/IROS55552.2023.10342424.
Nemiroff R，Chen K 和 Lopez BT（2023）用于惯性对齐建图的基于流形的联合重力与加速度计内参标定。载于：2023 IEEE/RSJ 智能机器人与系统国际会议（IROS）。第1388-1394页。DOI：10.1109/IROS55552.2023.10342424。


Nissov M, Edlund JA, Spieler P, Padgett C, Alexis K and Khattak S (2024a) Robust high-speed state estimation for off-road navigation using radar velocity factors. IEEE Robotics and Automation Letters 9(12): 11146-11153. DOI:10.1109/lra. 2024.3486189.
Nissov M，Edlund JA，Spieler P，Padgett C，Alexis K 和 Khattak S（2024a）使用雷达速度因子实现越野导航的稳健高速状态估计。IEEE Robotics and Automation Letters 9(12)：11146-11153。DOI：10.1109/lra. 2024.3486189。


Nissov M, Khedekar N and Alexis K (2024b) Degradation Resilient LiDAR-Radar-Inertial Odometry. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). pp. 8587-8594. DOI:10.1109/ICRA57147.2024.10611444. URL https://ieeexplore.ieee.org/document/ 10611444.
Nissov M，Khedekar N 和 Alexis K（2024b）退化鲁棒的 LiDAR-雷达-惯性里程计。载于：2024 IEEE 机器人与自动化国际会议（ICRA）。第8587-8594页。DOI：10.1109/ICRA57147.2024.10611444。URL https://ieeexplore.ieee.org/document/ 10611444。


Nissov M, Khedekar N and Alexis K (2025) Simultaneous triggering and synchronization of sensors and onboard computers. URL https://arxiv.org/abs/2507.05717.
Nissov M，Khedekar N 和 Alexis K（2025）传感器与板载计算机的同时触发与同步。URL https://arxiv.org/abs/2507.05717。


Noh C, Yang W, Jung M, Jung S and Kim A (2025) Garlio: Gravity enhanced radar-lidar-inertial odometry. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 9869-9875. DOI:10.1109/ICRA55743.2025.11128334.
Noh C，Yang W，Jung M，Jung S 和 Kim A（2025）Garlio：重力增强的雷达-激光雷达-惯性里程计。载于：2025 IEEE 机器人与自动化国际会议（ICRA）。第9869-9875页。DOI：10.1109/ICRA55743.2025.11128334。


NVIDIA, :, Mittal M, Roth P, Tigue J, Richard A, Zhang O, Du P, Serrano-Muñoz A, Yao X, Zurbrügg R, Rudin N, Wawrzyniak L, Rakhsha M, Denzler A, Heiden E, Borovicka A, Ahmed O, Akinola I, Anwar A, Carlson MT, Feng JY, Garg A, Gasoto R, Gulich L, Guo Y, Gussert M, Hansen A, Kulkarni M, Li C, Liu W, Makoviychuk V, Malczyk G, Mazhar H, Moghani M, Murali A, Noseworthy M, Poddubny A, Ratliff N, Rehberg W, Schwarke C, Singh R, Smith JL, Tang B, Thaker R, Trepte M, Wyk KV, Yu F, Millane A, Ramasamy V, Steiner R, Subramanian S, Volk C, Chen C, Jawale N, Kuruttukulam AV, Lin MA, Mandlekar A, Patzwaldt K, Welsh J, Zhao H, Anes F, Lafleche JF, Moënne-Loccoz N, Park S, Stepinski R, Gelder DV, Amevor C, Carius J, Chang J, Chen AH, de Heras Ciechomski P, Daviet G, Mohajerani M, von Muralt J, Reutskyy V, Sauter M, Schirm S, Shi EL, Terdiman P, Vilella K, Widmer T, Yeoman G, Chen T, Grizan S, Li C, Li L, Smith C, Wiltz R, Alexis K, Chang Y, Chu D, Fan LJ, Farshidian F, Handa A, Huang S, Hutter M, Narang Y, Pouya S, Sheng S, Zhu Y, Macklin M, Moravanszky A, Reist P, Guo Y, Hoeller D and State G (2025) Isaac lab: A gpu-accelerated simulation framework for multi-modal robot learning. URL https://arxiv.org/abs/2511.04831.
NVIDIA，：，Mittal M，Roth P，Tigue J，Richard A，Zhang O，Du P，Serrano-Muñoz A，Yao X，Zurbrügg R，Rudin N，Wawrzyniak L，Rakhsha M，Denzler A，Heiden E，Borovicka A，Ahmed O，Akinola I，Anwar A，Carlson MT，Feng JY，Garg A，Gasoto R，Gulich L，Guo Y，Gussert M，Hansen A，Kulkarni M，Li C，Liu W，Makoviychuk V，Malczyk G，Mazhar H，Moghani M，Murali A，Noseworthy M，Podbubny A，Ratliff N，Rehberg W，Schwarke C，Singh R，Smith JL，Tang B，Thaker R，Trepte M，Wyk KV，Yu F，Millane A，Ramasamy V，Steiner R，Subramanian S，Volk C，Chen C，Jawale N，Kuruttukulam AV，Lin MA，Mandlekar A，Patzwaldt K，Welsh J，Zhao H，Anes F，Lafleche JF，Moënne-Loccoz N，Park S，Stepinski R，Gelder DV，Amevor C，Carius J，Chang J，Chen AH，de Heras Ciechomski P，Daviet G，Mohajerani M，von Muralt J，Reutskyy V，Sauter M，Schirm S，Shi EL，Terdiman P，Vilella K，Widmer T，Yeoman G，Chen T，Grizan S，Li C，Li L，Smith C，Wiltz R，Alexis K，Chang Y，Chu D，Fan LJ，Farshidian F，Handa A，Huang S，Hutter M，Narang Y，Pouya S，Sheng S，Zhu Y，Macklin M，Moravanszky A，Reist P，Guo Y，Hoeller D 和 State G（2025）Isaac lab：一种面向多模态机器人学习的、GPU 加速的仿真框架。URL https://arxiv.org/abs/2511.04831。


Oleynikova H, Taylor Z, Fehr M, Siegwart R and Nieto J (2017) Voxblox: Incremental 3d euclidean signed distance fields for on-board mav planning. In: 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, pp. 1366-1373.
Oleynikova H，Taylor Z，Fehr M，Siegwart R 和 Nieto J（2017）Voxblox：用于车载 mav 规划的增量式三维欧几里得有符号距离场。在：2017 年 IEEE/RSJ 智能机器人与系统国际会议（IROS）。IEEE，第 1366-1373 页。


Petrenko A, Huang Z, Kumar T, Sukhatme G and Koltun V (2020) Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning. Proceedings of the 37 th International Conference on Machine Learning, Online, PMLR 119, 2020 .
Petrenko A，Huang Z，Kumar T，Sukhatme G 和 Koltun V（2020）Sample Factory：以 100000 FPS 从像素获得自我中心式三维控制，并配合异步强化学习。发表于：第 37 届机器学习国际会议，线上，PMLR 119，2020。


Petris PD, Nguyen H, Dharmadhikari M, Kulkarni M, Khedekar N, Mascarich F and Alexis K (2022) RMF-Owl: A Collision-Tolerant Flying Robot for Autonomous Subterranean Exploration. In: 2022 International Conference on Unmanned Aircraft Systems (ICUAS). Dubrovnik, Croatia: IEEE. ISBN 978-1-66540-593-5, pp. 536-543. DOI:10.1109/ICUAS54217. 2022.9836115. URL https://ieeexplore.ieee.org/document/9836115/.
Petris PD，Nguyen H，Dharmadhikari M，Kulkarni M，Khedekar N，Mascarich F 和 Alexis K（2022）RMF-Owl：一种面向自主地下探索的、耐碰撞的飞行机器人。在：2022 年国际无人驾驶飞行器系统会议（ICUAS）。克罗地亚杜布罗夫尼克：IEEE。ISBN 978-1-66540-593-5，第 536-543 页。DOI：10.1109/ICUAS54217.2022.9836115。URL https://ieeexplore.ieee.org/document/9836115/。


Planning R (2012) navigation. URL https://github.com/ ros-planning/navigation. Accessed 2026-04-09.
Planning R（2012）navigation。URL https://github.com/ros-planning/navigation。访问日期：2026-04-09。


Qian C, Xu Y, Shi X, Chen J and Li L (2025) Af-rlio: Adaptive fusion of radar-lidar-inertial information for robust odometry in challenging environments. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 1-7. DOI: 10.1109/ICRA55743.2025.11128046.
Qian C，Xu Y，Shi X，Chen J 和 Li L（2025）Af-rlio：在复杂环境中实现鲁棒里程计的雷达-激光雷达-惯性信息自适应融合。在：2025 年 IEEE 国际机器人与自动化会议（ICRA），第 1-7 页。DOI：10.1109/ICRA55743.2025.11128046。


Qin T, Pan J, Cao S and Shen S (2019) A general optimization-based framework for local odometry estimation with multiple sensors. URL https://arxiv.org/abs/1901.03638.
Qin T，Pan J，Cao S 和 Shen S（2019）一种基于通用优化的多传感器局部里程计估计框架。URL https://arxiv.org/abs/1901.03638。


Ravichandran Z, Cladera F, Hughes J, Murali V, Hsieh MA, Pappas GJ, Taylor CJ and Kumar V (2025) Deploying foundation model-enabled air and ground robots in the field: Challenges and opportunities. arXiv preprint arXiv:2505.09477 .
Ravichandran Z，Cladera F，Hughes J，Murali V，Hsieh MA，Pappas GJ，Taylor CJ 和 Kumar V（2025）在实地部署由基础模型赋能的空中与地面机器人：挑战与机遇。arXiv 预印本 arXiv:2505.09477。


Real F, Torres-González A, Soria PR, Capitán J and Ollero A (2020) Unmanned aerial vehicle abstraction layer: An abstraction layer to operate unmanned aerial vehicles. International Journal of Advanced Robotic Systems 17(4): 1-13. DOI:10. 1177/1729881420925011. URL https://doi.org/10.1177/1729881420925011.
Real F，Torres-González A，Soria PR，Capitán J 和 Ollero A（2020）无人机抽象层：用于操作无人机的抽象层。国际先进机器人系统期刊 17(4)：1-13。DOI：10. 1177/1729881420925011。URL：https://doi.org/10.1177/1729881420925011。


Sanchez-Lopez JL, Fernández RAS, Bavle H, Sampedro C, Molina M, Pestana J and Campoy P (2016) Aerostack: An architecture and open-source software framework for aerial robotics. In: 2016 International Conference on Unmanned Aircraft Systems (ICUAS). IEEE, pp. 332-341.
Sanchez-Lopez JL，Fernández RAS，Bavle H，Sampedro C，Molina M，Pestana J 和 Campoy P（2016）Aerostack：面向航空机器人的架构与开源软件框架。载于：2016 年国际无人机系统会议（ICUAS）。IEEE，第 332-341 页。


Schulman J, Wolski F, Dhariwal P, Radford A and Klimov O (2017) Proximal Policy Optimization Algorithms. URL http:// arxiv.org/abs/1707.06347.
Schulman J，Wolski F，Dhariwal P，Radford A 和 Klimov O（2017）近端策略优化算法。URL：http:// arxiv.org/abs/1707.06347。


Schwarke C, Mittal M, Rudin N, Hoeller D and Hutter M (2025) Rsl-rl: A learning library for robotics research. arXiv preprint arXiv:2509.10771 .
Schwarke C，Mittal M，Rudin N，Hoeller D 和 Hutter M（2025）Rsl-rl：面向机器人研究的学习库。arXiv 预印本 arXiv：2509.10771 。


Serpiva V, Lykov A, Myshlyaev A, Khan MH, Abdulkarim AA, Sautenkov O and Tsetserukou D (2025) Racevla: Vla-based racing drone navigation with human-like behaviour. arXiv preprint arXiv:2503.02572 .
Serpiva V，Lykov A，Myshlyaev A，Khan MH，Abdulkarim AA，Sautenkov O 和 Tsetserukou D（2025）Racevla：基于 Vla 的类人行为竞速无人机导航。arXiv 预印本 arXiv：2503.02572 。


Shan T, Englot B, Meyers D, Wang W, Ratti C and Daniela R (2020) Lio-sam: Tightly-coupled lidar inertial odometry via smoothing and mapping. In: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, pp. 5135-5142.
Shan T，Englot B，Meyers D，Wang W，Ratti C 和 Daniela R（2020）Lio-sam：通过平滑与建图实现紧耦合激光雷达惯性里程计。载于：IEEE/RSJ 智能机器人与系统国际会议（IROS）。IEEE，第 5135-5142 页。


Sucan IA, Moll M and Kavraki LE (2012) The open motion planning library. IEEE Robotics & Automation Magazine 19(4): 72-82.
Sucan IA，Moll M 和 Kavraki LE（2012）开放式运动规划库。IEEE 机器人与自动化杂志 19(4)：72-82。


Tranzatto M, Miki T, Dharmadhikari M, Bernreiter L, Kulkarni M, Mascarich F, Andersson O, Khattak S, Hutter M, Siegwart R et al. (2022) Cerberus in the darpa subterranean challenge. Science Robotics 7(66): eabp9742.
Tranzatto M，Miki T，Dharmadhikari M，Bernreiter L，Kulkarni M，Mascarich F，Andersson O，Khattak S，Hutter M，Siegwart R 等（2022）达尔帕地下挑战中的 Cerberus。Science Robotics 7(66)：eabp9742。


Verschueren R, Frison G, Kouzoupis D, Frey J, Duijkeren Nv, Zanelli A, Novoselnik B, Albin T, Quirynen R and Diehl M (2022) acados-a modular open-source framework for fast embedded optimal control. Mathematical Programming Computation 14(1): 147-183.
Verschueren R，Frison G，Kouzoupis D，Frey J，Duijkeren Nv，Zanelli A，Novoselnik B，Albin T，Quirynen R 和 Diehl M（2022）acados：面向快速嵌入式最优控制的模块化开源框架。数学规划计算 14(1)：147-183。


Vizzo I, Guadagnino T, Mersch B, Wiesmann L, Behley J and Stachniss C (2023) KISS-ICP: In Defense of Point-to-Point ICP - Simple, Accurate, and Robust Registration If Done the Right Way. IEEE Robotics and Automation Letters (RA-L) 8(2): 1029-1036. DOI:10.1109/LRA.2023.3236571.
Vizzo I，Guadagnino T，Mersch B，Wiesmann L，Behley J 和 Stachniss C（2023）KISS-ICP：为点到点 ICP 辩护——只要用对方式，就能简单、准确且鲁棒。IEEE 机器人与自动化快报（RA-L）8(2)：1029-1036。DOI：10.1109/LRA.2023.3236571。


Wang A, Liu L, Chen H, Lin Z, Han J and Ding G (2025) Yoloe: Real-time seeing anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 24591-24602.
Wang A，Liu L，Chen H，Lin Z，Han J 和 Ding G（2025）Yoloe：实时看见任何东西。载于：IEEE/CVF 国际计算机视觉会议论文集（ICCV）。第 24591-24602 页。


Wang Z, Zhou X, Xu C and Gao F (2022) Geometrically constrained trajectory optimization for multicopters. IEEE Transactions on Robotics 38(5): 3259-3278.
Wang Z，Zhou X，Xu C 和 Gao F（2022）用于多旋翼的几何约束轨迹优化。IEEE 机器人学汇刊 38(5)：3259-3278。


Wu Y, Zhu M, Li X, Du Y, Fan Y, Li W, Han Z, Zhou X and Gao F (2025) Vla-an: An efficient and onboard vision-language-action framework for aerial navigation in complex environments. arXiv preprint arXiv:2512.15258 .
Wu Y，Zhu M，Li X，Du Y，Fan Y，Li W，Han Z，Zhou X 和 Gao F（2025）Vla-an：面向复杂环境航空导航的高效端上视觉-语言-行动框架。arXiv 预印本 arXiv：2512.15258 。


Xu P, Deng Z, Deng J, Gu Z and Wan S (2026) Aerialvla: A vision-language-action model for uav navigation via minimalist end-to-end control. arXiv preprint arXiv:2603.14363 .
Xu P，Deng Z，Deng J，Gu Z 和 Wan S（2026）Aerialvla：通过极简端到端控制实现的用于 UAV 导航的视觉-语言-行动模型。arXiv 预印本 arXiv：2603.14363 。


Xu W, Cai Y, He D, Lin J and Zhang F (2022) Fast-lio2: Fast direct lidar-inertial odometry. IEEE Transactions on Robotics 38(4): 2053-2073. DOI:10.1109/TRO.2022.3141876.
Xu W，Cai Y，He D，Lin J 和 Zhang F（2022）Fast-lio2：快速直接激光雷达惯性里程计。IEEE 机器人学汇刊 38(4)：2053-2073。DOI：10.1109/TRO.2022.3141876。


Zacharia A, Dharmadhikari M, Singh M and Alexis K (2026) Omniplanner: Universal exploration and inspection path planning across robot morphologies. URL https:// arxiv.org/abs/2603.04284.
Zacharia A, Dharmadhikari M, Singh M 和 Alexis K（2026）Omniplanner：面向多种机器人形态的通用探索与检视路径规划。URL https:// arxiv.org/abs/2603.04284。


Zakka K, Tabanpour B, Liao Q, Haiderbhai M, Holt S, Luo JY, Allshire A, Frey E, Sreenath K, Kahrs LA, Sferrazza C, Tassa Y and Abbeel P (2025) Mujoco playground: An open-source framework for gpu-accelerated robot learning and sim-to-real transfer. URL https://github.com/ google-deepmind/mujoco_playground.
Zakka K, Tabanpour B, Liao Q, Haiderbhai M, Holt S, Luo JY, Allshire A, Frey E, Sreenath K, Kahrs LA, Sferrazza C, Tassa Y 和 Abbeel P（2025）Mujoco playground：用于GPU加速机器人学习与仿真到现实迁移的开源框架。URL https://github.com/ google-deepmind/mujoco_playground。


Zhang Q, Zheng S, Sun J, Li C, Wu X, Song Z, Cui Z, Lv Y and Tian Y (2026) Uav-track vla: Embodied aerial tracking via vision-language-action models. arXiv preprint arXiv:2604.02241 .
Zhang Q, Zheng S, Sun J, Li C, Wu X, Song Z, Cui Z, Lv Y 和 Tian Y（2026）Uav-track vla：借助视觉-语言-行动模型实现具身空中追踪。arXiv预印本 arXiv:2604.02241 。


Zheng C, Xu W, Zou Z, Hua T, Yuan C, He D, Zhou B, Liu Z, Lin J, Zhu F, Ren Y, Wang R, Meng F and Zhang F (2025) Fast-livo2: Fast, direct lidar-inertial-visual odometry. IEEE Transactions on Robotics 41: 326-346. DOI:10.1109/TRO.2024.3502198.
Zheng C, Xu W, Zou Z, Hua T, Yuan C, He D, Zhou B, Liu Z, Lin J, Zhu F, Ren Y, Wang R, Meng F 和 Zhang F（2025）Fast-livo2：快速、直接的激光雷达-惯性-视觉里程计。IEEE Robotics Transactions 41：326-346。DOI:10.1109/TRO.2024.3502198。


## A Index to multimedia Extensions
## 多媒体扩展目录


<table><tr><td>Ext.</td><td>Media type</td><td>Description</td></tr><tr><td>1</td><td>Video</td><td>Filename: 00_unified_autonomy_stack_main_video.mp4. This video provides an overview of this paper including a) the UAstack architecture, b) the details of each core modules, perception, planning, and navigation, and c) videos of indicative field deployments presented in the paper.</td></tr><tr><td>2</td><td>Video</td><td>Filename: 01_full_stack_evaluation_aerial_robot.mp4. This video shows the <br> recording and visualizations of the field experiments evaluating the full stack on the aerial robot <br> as presented in Section 4.4.1. The robot is deployed in an underground mine, a forest, and a ship <br> cargo hold, evaluating the exploration and inspection behaviors.</td></tr><tr><td>3</td><td>Video</td><td>Filename: 02_full_stack_evaluation_ground_robot.mp4. This video shows the <br> recording and visualizations of the field experiments evaluating the full stack on the ground robot <br> as presented in Section 4.4.2. The robot is deployed in an underground mine and a university <br> building, evaluating the exploration behavior in narrow and large scale environments.</td></tr><tr><td>4</td><td>Video</td><td>Filename: 03_navigation_evaluation.mp4. This video shows the recording and <br> visualizations of the field experiments evaluating the navigation module as presented in <br> Section 4.3. The navigation module is tested in a waypoint navigation task and in the presence <br> of moving obstacles demonstrating the usefulness of the multi-layered safety approach.</td></tr><tr><td>5</td><td>Video</td><td>Filename: 04_slam_evaluation.mp4. This video shows the recording and visualizations <br> of the field experiments evaluating the multi-modal SLAM component of the perception <br> module as presented in Section 4.2.1. The multi-modal SLAM is evaluated in datasets <br> including a) flying in self-similar and dimly lit tunnels, b) flying over a self-similar frozen lake, <br> and c) handheld dataset in a university campus containing area filled with fog.</td></tr><tr><td>6</td><td>Video</td><td>Filename: 05_reasoning_evaluation.mp4. This video shows the recording and <br> visualizations of the field experiments evaluating the VLM-based reasoning component of the <br> perception module as presented in Section 4.2.2. The module is tested on two datasets a) <br> the campus dataset from the SLAM evaluation, and b) the ground robot mission in the university <br> building. The video shows the open vocabulary object detection and binary Q&A capabilities of <br> the module.</td></tr></table>
<table><tbody><tr><td>扩展。</td><td>媒体类型</td><td>描述</td></tr><tr><td>1</td><td>视频</td><td>文件名：00_unified_autonomy_stack_main_video.mp4。该视频概述本文，包括：a）UAstack架构，b）各核心模块的细节（感知、规划和导航），以及c）本文中展示的典型实地部署视频。</td></tr><tr><td>2</td><td>视频</td><td>文件名：01_full_stack_evaluation_aerial_robot.mp4。该视频展示了在第4.4.1节中所述，评估空中机器人上的完整堆栈的实地实验的<br/>录制和可视化。机器人部署在地下矿井、森林以及船舱货<br/>舱中，评估探索与巡检行为。</td></tr><tr><td>3</td><td>视频</td><td>文件名：02_full_stack_evaluation_ground_robot.mp4。该视频展示了在第4.4.2节中所述，评估地面机器人上的完整堆栈的实地实验的<br/>录制和可视化。机器人部署在地下矿井以及一所大学<br/>建筑中，在狭窄与大规模环境下评估探索行为。</td></tr><tr><td>4</td><td>视频</td><td>文件名：03_navigation_evaluation.mp4。该视频展示了在第<br/>4.3节中所述，对导航模块开展的实地实验的录制与<br/>可视化。导航模块在航点导航任务中测试，并在<br/>存在移动障碍物的情况下测试，体现了多层安全方案的有效性。</td></tr><tr><td>5</td><td>视频</td><td>文件名：04_slam_evaluation.mp4。该视频展示了在第4.2.1节中所述，<br/>对感知模块的多模态SLAM组件开展的实地实验的录制与<br/>可视化。多模态SLAM在数据集上进行评估，包括：a）在自相似且昏暗照明的隧道中飞行，b）在自相似的<br/>结冰湖上飞行，<br/>以及c）大学校园中包含雾气填充区域的手持数据集。</td></tr><tr><td>6</td><td>视频</td><td>文件名：05_reasoning_evaluation.mp4。该视频展示了在第4.2.2节中所述，<br/>对基于VLM的推理组件（属于感知模块）的实地实验录制与<br/>可视化。该模块在两个数据集上进行测试：a）来自SLAM评估的校园数据集；b）大学<br/>建筑中的地面机器人任务。视频展示了该模块的开放词汇目标检测能力以及<br/>二元问答能力。</td></tr></tbody></table>


Table 8. Index of multimedia extensions.
多媒体扩展索引表 8。