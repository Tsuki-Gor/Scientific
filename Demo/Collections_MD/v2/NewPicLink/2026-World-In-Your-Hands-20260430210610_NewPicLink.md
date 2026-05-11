# World In Your Hands: A Large-Scale and Open-Source Ecosystem for Learning Human-Centric Manipulation in the Wild
# World In Your Hands：一个用于学习真实世界以人为中心操作的大规模开源生态系统


Yupeng Zheng*, Jichao Peng*, Weize Li*, Yuhang Zheng, Xiang Li, Yujie Jin, Julong Wei, Guanhua Zhang, Ruiling Zheng, Ming Cao, Songen Gu, Zhenhong Zou, Kaige Li, Ke Wu, Mingmin Yang, Jiahao Liu, Pengfei Li, Hengjie Si, Feiyu Zhu, Wang Fu, Likun Wang, Ruiwen Yao, Jieru Zhao, Yilun Chen, Wenchao Ding†
Yupeng Zheng*, Jichao Peng*, Weize Li*, Yuhang Zheng, Xiang Li, Yujie Jin, Julong Wei, Guanhua Zhang, Ruiling Zheng, Ming Cao, Songen Gu, Zhenhong Zou, Kaige Li, Ke Wu, Mingmin Yang, Jiahao Liu, Pengfei Li, Hengjie Si, Feiyu Zhu, Wang Fu, Likun Wang, Ruiwen Yao, Jieru Zhao, Yilun Chen, Wenchao Ding†


Project Page: https://wiyh.tars-ai.com
项目主页：https://wiyh.tars-ai.com


Code: https://github.com/tars-robotics/World-In-Your-Hands
代码仓库：https://github.com/tars-robotics/World-In-Your-Hands


TARS Robotics
TARS Robotics


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_283866.jpg"/>



Fig. 1: Our World In Your Hands (WIYH) ecosystem is built on top of a large-scale self-collected dataset in real-world human environments. It captures human subjects performing over 40 distinct tasks across a wide variety of scenarios. The dataset provides rich RGB video streams, along with comprehensive motion ground truth encompassing more than 100 human skills. In addition to sensor and action data, it includes diverse annotations. This rich multimodal collection makes WIYH an excellent resource for research in spatial intelligence and embodied foundation model training.
图 1：我们的 World In Your Hands (WIYH) 生态系统建立在真实人类环境中大规模自采集数据集的基础上。它捕捉了人类在各种场景下执行的 40 多种不同任务。该数据集提供了丰富的 RGB 视频流，以及涵盖 100 多种人类技能的全面运动真值。除了传感器和动作数据外，它还包含多样化的标注。这种丰富的多模态集合使 WIYH 成为空间智能和具身基础模型训练研究的绝佳资源。


Abstract. We introduce World In Your Hands (WIYH), a large-scale open-source ecosystem comprising over 1,000 hours of human manipulation data collected in-the-wild with millimeter-scale motion accuracy. Specifically, WIYH includes (1) the Oracle Suite, a wearable data collection kit with an auto-labeling
摘要。我们介绍了 World In Your Hands (WIYH)，这是一个大规模开源生态系统，包含超过 1,000 小时在真实世界中采集的、具有毫米级运动精度的人类操作数据。具体而言，WIYH 包括 (1) Oracle Suite，一套带有自动标注


pipeline for accurate motion capture; (2) the WIYH Dataset, featuring over 1,000 hours of multimodal manipulation data across hundreds of skills in diverse real-world scenarios; and (3) extensive annotations and benchmarks supporting tasks from perception to action. Furthermore, experiments based on the WIYH ecosystem show that integrating WIYH's human-centric data improves robotic manipulation success rates from 8% to 60% in cluttered scenes. World In Your Hands provides a foundation for advancing human-centric data collection and cross-embodiment policy learning. All data and hardware design will be open-source.
流水线的可穿戴数据采集套件，用于精确的运动捕捉；(2) WIYH 数据集，包含在多样化真实场景中、跨越数百种技能的 1,000 多小时多模态操作数据；以及 (3) 支持从感知到动作等任务的广泛标注和基准测试。此外，基于 WIYH 生态系统的实验表明，整合 WIYH 的以人为中心的数据，可将机器人操作在复杂场景下的成功率从 8% 提升至 60%。World In Your Hands 为推进以人为中心的数据采集和跨具身策略学习奠定了基础。所有数据和硬件设计都将开源。


Keywords: Robotic Manipulation - Ego-centric Learning - System
关键词：机器人操作 - 自我中心学习 - 系统


## 1 Introduction
## 1 引言


Large-scale pre-training has been established as a cornerstone for achieving generalization in Large Language Models [1, 8, 12, 44, 55], Vision-Language Models [2, 29-31, 38] and Vision-Language-Action (VLA) Models [6, 7, 26, 51, 56, 59]. However, datasets for dexterous hand manipulation remain far smaller and less diverse than language datasets, limiting progress in learning robust manipulation. To address the problem of data scarcity, human-centric data collection and learning have attracted growing interest 21, 24, 39, 57, 58. Some approaches 5, 11,28 capture human hand data during object manipulation using devices such as VR systems and smart glasses, while others 16, 18, 46, 53 leverage foundation models to extract human actions from unlabeled videos of human manipulation available on the internet.
大规模预训练已成为实现大语言模型 [1, 8, 12, 44, 55]、视觉-语言模型 [2, 29-31, 38] 以及视觉-语言-动作 (VLA) 模型 [6, 7, 26, 51, 56, 59] 泛化能力的基石。然而，灵巧手操作数据集的规模和多样性仍远不及语言数据集，限制了学习稳健操作的进展。为解决数据匮乏问题，以人为中心的数据采集与学习引起了越来越多的关注 [21, 24, 39, 57, 58]。一些方法 [5, 11, 28] 利用 VR 系统和智能眼镜等设备在物体操作过程中捕捉人类手部数据，而另一些方法 [16, 18, 46, 53] 则利用基础模型从互联网上无标注的人类操作视频中提取人类动作。


Despite these advances, current human manipulation datasets still face three major limitations, as shown in Table 1 that hinder their usefulness for dexterous manipulation research: (1) Limited Scenario Diversity: Many datasets are collected in constrained laboratory environments, lacking the diversity and complexity of real-world settings. (2) Inadequate Alignment: Existing collections often lack one or more key modalities, such as the absence of fine-grained language instructions, precise 3D action, or object poses, resulting in misalignment across vision, language, and action. (3) Insufficient Benchmarking: Most datasets do not support comprehensive evaluation benchmarks that span the entire pipeline of dexterous manipulation, including scene perception, task planning, and action.
尽管取得了这些进展，但如表 1 所示，当前的人类操作数据集仍面临三个主要局限，阻碍了它们在灵巧操作研究中的应用：(1) 场景多样性有限：许多数据集是在受限的实验室环境中采集的，缺乏真实世界场景的多样性和复杂性。(2) 对齐不足：现有数据集往往缺少一个或多个关键模态，例如缺乏细粒度的语言指令、精确的 3D 动作或物体姿态，导致视觉、语言和动作之间无法对齐。(3) 基准测试不足：大多数数据集不支持涵盖灵巧操作全流程（包括场景感知、任务规划和动作执行）的综合评估基准。


Table 1: Comparison to existing datasets. WIYH is characterized by two key distinctions: (1) Its entire 1000 hours of data were collected in the wild, with the tasks performed by skilled practitioners, thereby capturing the most authentic decision-making processes and actions; (2) Despite the significant challenge of annotating such a diverse, real-world collection, WIYH provides comprehensive annotations, including motion ground truth, depth, masks, task/sub-task labels, and chain-of-thought (CoT) reasoning, etc.
表 1：与现有数据集的对比。WIYH 具有两个关键区别：(1) 其全部 1,000 小时数据均在真实世界中采集，且任务由熟练从业者执行，从而捕捉到了最真实的人类决策过程和动作；(2) 尽管对如此多样化的真实世界数据集进行标注极具挑战，WIYH 仍提供了全面的标注，包括运动真值、深度、掩码、任务/子任务标签以及思维链 (CoT) 推理等。


<table><tr><td>Dataset</td><td>Hours</td><td>Clips</td><td>Wild</td><td>Cam. Calib</td><td>RGB</td><td>Tactile</td><td>Action</td><td>Depth</td><td>Mask</td><td>Instruct.</td><td>VLM</td></tr><tr><td>Ego4D 161</td><td>3,670</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>2D</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>HOI4D 34</td><td>44.4</td><td>4k</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>2D</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>HoloAssist [49]</td><td>166</td><td>2.22k</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>-</td><td>✓</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>Ego-Exo4D 17</td><td>1,286</td><td>5.04k</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>CaptainCook4D 36</td><td>94.5</td><td>384</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>HOT3D [5]</td><td>13.9</td><td>425</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>3D</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>HO-Cap [45]</td><td>-</td><td>64</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>3D</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>Ego-ViD-5M 4</td><td>-</td><td>5M</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>EgoDex 20</td><td>829</td><td>338k</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>3D</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td></tr><tr><td>WIYH (Ours)</td><td>1045</td><td>125.4k</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>3D</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>
<table><tbody><tr><td>数据集</td><td>时长（小时）</td><td>片段数</td><td>野外环境</td><td>相机标定</td><td>RGB</td><td>触觉</td><td>动作</td><td>深度</td><td>掩码</td><td>指令</td><td>视觉语言模型</td></tr><tr><td>Ego4D 161</td><td>3,670</td><td>✓</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>2D</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>HOI4D 34</td><td>44.4</td><td>4k</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>2D</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>HoloAssist [49]</td><td>166</td><td>2.22k</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>-</td><td>✓</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>Ego-Exo4D 17</td><td>1,286</td><td>5.04k</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>CaptainCook4D 36</td><td>94.5</td><td>384</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td><td>✓</td><td>✓</td></tr><tr><td>HOT3D [5]</td><td>13.9</td><td>425</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>3D</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>HO-Cap [45]</td><td>-</td><td>64</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>3D</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>Ego-ViD-5M 4</td><td>-</td><td>5M</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>✘</td><td>✓</td><td>✘</td></tr><tr><td>EgoDex 20</td><td>829</td><td>338k</td><td>✓</td><td>✘</td><td>✓</td><td>✘</td><td>3D</td><td>✘</td><td>✓</td><td>✓</td><td>✘</td></tr><tr><td>WIYH (本文)</td><td>1045</td><td>125.4k</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>3D</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></tbody></table>


To bridge this gap and investigate in-depth how human-centric data can advance dexterous manipulation, we introduce World In Your Hands (WIYH), a large-scale open-source ecosystem for learning human-centric manipulation in the wild, as shown in Figure 1. Specifically, the WIYH ecosystem comprises three core components:
为了弥补这一差距并深入研究以人为中心的数据如何推动灵巧操作的发展，我们推出了“World In Your Hands”（WIYH），这是一个用于学习野外以人为中心操作的大规模开源生态系统，如图1所示。具体而言，WIYH生态系统包含三个核心组件：


- The Oracle Suite. We introduce a lightweight, flexible wearable data-collection kit that captures 3D action data with millimeter-scale translational accuracy (mean error below $5\mathrm{\;{mm}}$ ) without external third-person tracking.
- Oracle Suite。我们引入了一套轻量级、灵活的可穿戴数据采集套件，能够在无需外部第三人称追踪的情况下，以毫米级的平移精度（平均误差低于 $5\mathrm{\;{mm}}$ ）捕获3D动作数据。


- The WIYH Dataset. We present a large-scale multi-modal resource comprising over 1,000 hours of manipulation demonstrations across 100 skills, captured in 10 diverse scenarios. The dataset includes rich sensory streams: multi-view images, camera calibration parameters, hand poses, and wrist trajectories.
- WIYH数据集。我们呈现了一个大规模多模态资源，包含在10种不同场景下采集的超过1000小时的操作演示，涵盖100种技能。该数据集包含丰富的感官流：多视角图像、相机标定参数、手部姿态和手腕轨迹。


- Diverse Annotations. Building upon the WIYH dataset, we provide extensive annotations, including 400 hours of atomic action instructions and 100k instances of vision-language data, supporting a comprehensive usage from scene perception to robotic manipulation.
- 多样化标注。基于WIYH数据集，我们提供了广泛的标注，包括400小时的原子动作指令和10万条视觉-语言数据实例，支持从场景感知到机器人操作的全面应用。


Based on the WIYH ecosystem, we study how human-centric data empowers dexterous manipulation learning in robotic manipulation tasks. We conduct two sets of experiments: (1) cross-embodiment pretraining and (2) retargeting co-training, and evaluate the task success rate across varied environments and object layouts. Our experiments demonstrate that human-centric data substantially improves the generalization and robustness of robotic manipulation, whether as large-scale cross-embodiment pre-training data or retargeted action data.
基于WIYH生态系统，我们研究了以人为中心的数据如何赋能机器人操作任务中的灵巧操作学习。我们进行了两组实验：（1）跨形态预训练和（2）重定向协同训练，并评估了在不同环境和物体布局下的任务成功率。我们的实验表明，无论是作为大规模跨形态预训练数据还是重定向动作数据，以人为中心的数据都能显著提高机器人操作的泛化能力和鲁棒性。


Limitation. To increase accessibility and reproducibility, we are developing high-fidelity simulation environments and plan to release them in a future version of the dataset and benchmarks.
局限性。为了提高可访问性和可复现性，我们正在开发高保真仿真环境，并计划在数据集和基准测试的未来版本中发布它们。


## 2 World In Your Hands Ecosystem
## 2 World In Your Hands 生态系统


### 2.1 Oracle Suite: Human-centric Data Collection Suite
### 2.1 Oracle Suite：以人为中心的数据采集套件


We develop Oracle Suite, a low-cost wearable data collection system designed for human-centric multimodal data collection. Specifically, as illustrated in Figure 2. Oracle Suite comprises a hardware system that captures multi-view RGB images, IMU-based localization, and tactile information and an autolabeling module that achieves high-precision motion capture by fusing IR, RGB, and IMU localization.
我们开发了Oracle Suite，这是一种专为以人为中心的多模态数据采集而设计的低成本可穿戴数据采集系统。具体如图2所示，Oracle Suite包含一个捕获多视角RGB图像、基于IMU的定位和触觉信息的硬件系统，以及一个通过融合红外、RGB和IMU定位来实现高精度运动捕捉的自动标注模块。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_905f6f.jpg"/>



Fig. 2: Oracle Suite: Human-centric Data Collection Suite. It is primarily composed of three integrated components: (1) H-FPVHive: A first-person perception suite equipped with multiple cameras of different modalities to comprehensively record the operator's environmental context. (2) H-Glove: A hand motion capture and tactile perception module. It integrates motion capture gloves, tactile sensors, and visual trackers. The H-Glove is synchronized with the H-FPVHive, enabling precise action localization and capture in unstructured, real-world settings. (3) H-Backpack: A power supply and data storage unit.
图2：Oracle Suite：以人为中心的数据采集套件。它主要由三个集成组件构成：（1）H-FPVHive：一套第一人称感知套件，配备多种模态的摄像头，用于全面记录操作者的环境背景。（2）H-Glove：手部运动捕捉与触觉感知模块。它集成了运动捕捉手套、触觉传感器和视觉追踪器。H-Glove与H-FPVHive同步，能够在非结构化的现实环境中实现精确的动作定位与捕获。（3）H-Backpack：电源供应与数据存储单元。


Hardware System Overview. Oracle Suite adopts a modular architecture integrating three primary hardware components: H-FPVHive, H-Gloves, and H-Backpack. This design enables flexible deployment across diverse environments without being restricted to stable indoor labs.
硬件系统概述。Oracle Suite采用模块化架构，集成了三个主要硬件组件：H-FPVHive、H-Gloves和H-Backpack。这种设计使其能够灵活部署在各种环境中，而不受限于稳定的室内实验室。


Hardware Design. The Oracle Suite incorporates significant optimizations in ergonomics, sensing, and reliability.
硬件设计。Oracle Suite在人体工程学、传感和可靠性方面进行了重大优化。


- H-FPVHive supports a chest-mounted configuration with two fisheye cameras, two pinhole cameras, and four infrared lenses integrated into the chest unit, where all cameras achieve hardware synchronization, and the infrared lenses facilitate better localization of the relative pose of H-Gloves.
- H-FPVHive支持胸挂式配置，胸部单元集成了两个鱼眼相机、两个针孔相机和四个红外镜头，所有相机均实现硬件同步，红外镜头有助于更好地定位H-Gloves的相对姿态。


- H-Gloves include six IMUs, five fingertip pressure sensors with a resolution of $5\mathrm{{mN}}$ and a range of ${0.2}\mathrm{\;N}$ to ${50}\mathrm{\;N}$ ,and an onboard MCU per glove to capture hand motions and tactile information while also containing three fisheye cameras for visual tracking and observation.
- H-Gloves包含六个IMU、五个分辨率为 $5\mathrm{{mN}}$ 且量程为 ${0.2}\mathrm{\;N}$ 至 ${50}\mathrm{\;N}$ 的指尖压力传感器，以及每个手套内置的用于捕获手部动作和触觉信息的MCU，同时还包含三个用于视觉追踪和观察的鱼眼相机。


- H-Backpack provides data storage, computation, and power supply, utilizing an NVIDIA Orin computing core.
- H-Backpack提供数据存储、计算和电源供应，采用NVIDIA Orin计算核心。


Autolabeling Pipeline. The autolabeling pipeline involves online and offline algorithms, as shown on the right side of Figure 2 By fusing multimodal sensor streams (IR, IMU, and RGB), it maintains robust localization in in-the-wild scenarios and ultimately outputs 6D wrist pose trajectories with millimeter-scale translational accuracy (mean positional error below $5\mathrm{\;{mm}}$ ).
自动标注流水线。自动标注流水线涉及在线和离线算法，如图2右侧所示。通过融合多模态传感器流（红外、IMU和RGB），它在野外场景中保持了稳健的定位，并最终输出平移精度达到毫米级（平均位置误差低于 $5\mathrm{\;{mm}}$ ）的6D手腕姿态轨迹。


Efficiency of Oracle Suite. Compared to teleoperation, Oracle Suite achieves 5×higher collection efficiency (720 vs. 150 episodes/day) without extensive operator training. Compared to VR-based hand tracking, Oracle Suite provides more accurate 3D hand skeletons, particularly under visual occlusion.
Oracle Suite 的效率。与遥操作相比，Oracle Suite 在无需大量操作员培训的情况下，实现了 5 倍的采集效率（720 vs. 150 个片段/天）。与基于 VR 的手部追踪相比，Oracle Suite 提供了更精确的 3D 手部骨架，尤其是在视觉遮挡情况下。


### 2.2 Data Validation System
### 2.2 数据验证系统


We employ motion capture tests and projection intersection methods to validate data quality. Figure 3 illustrates the testing effects of both methods, and we retain valid high-quality data. For the motion capture test, we attach motion capture spheres to H-Gloves and complete operations identical to those in the WIYH dataset within a motion-capture room. This setup tests the deviation between the wrist trajectory generated by the Oracle Suite autolabeling pipeline and the motion capture trajectory to verify the effectiveness of the automatic annotation pipeline. Our Oracle Suite achieves an average position error of 5 $\mathrm{{mm}}$ in the motion capture room environment. For projection overlap,we obtain hand masks in perspective views of H-FPVHive using a hand segmentation algorithm. We then project the skeleton data from H-Gloves into the same view and calculate the intersection with the hand mask to determine accuracy.
我们采用动作捕捉测试和投影交集法来验证数据质量。图 3 展示了两种方法的测试效果，我们保留了有效的优质数据。对于动作捕捉测试，我们将动捕球安装在 H-Gloves 上，并在动捕室内完成与 WIYH 数据集中相同的操作。该设置通过测试 Oracle Suite 自动标注流水线生成的腕部轨迹与动捕轨迹之间的偏差，来验证自动标注流水线的有效性。我们的 Oracle Suite 在动捕室环境下的平均位置误差为 5 $\mathrm{{mm}}$。对于投影重叠，我们利用手部分割算法获取 H-FPVHive 透视图中的手部掩码，随后将 H-Gloves 的骨架数据投影到同一视角，并计算其与手部掩码的交集以确定准确性。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_758a89.jpg"/>



Fig. 3: WIYH Data Validation. The first row illustrates the motion capture test conducted in the mocap room. The dashed curve indicates the reference trajectory measured by the mocap system, while the solid curve denotes the trajectory estimated by the Oracle Suite. Color intensity encodes the deviation magnitude, with red indicating larger errors. In this controlled lab setting, the Oracle Suite achieves a mean translational error below $5\mathrm{\;{mm}}$ . The second row presents the projection-intersection check used to verify action accuracy. We compute an intersection score and flag samples whose score falls below a predefined threshold; these cases are then manually reviewed and filtered to ensure data quality.
图 3：WIYH 数据验证。第一行展示了在动捕室内进行的动作捕捉测试。虚线曲线表示动捕系统测得的参考轨迹，实线曲线表示 Oracle Suite 估计的轨迹。颜色深浅代表偏差幅度，红色表示误差较大。在此受控实验室环境下，Oracle Suite 的平均平移误差低于 $5\mathrm{\;{mm}}$。第二行展示了用于验证动作准确性的投影交集检查。我们计算交集得分，并标记得分低于预设阈值的样本；这些样本随后会经过人工审核和过滤，以确保数据质量。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_a47169.jpg"/>



Fig. 4: Overview of WiYH dataset annotation pipeline. It consists of three . Finally, all annotation information is manually reviewed by humans to ensure accuracy and consistency.
图 4：WIYH 数据集标注流水线概览。它由三个阶段组成。最后，所有标注信息均经过人工审核，以确保准确性和一致性。


## 3 World In Your Hands Dataset
## 3 World In Your Hands 数据集


### 3.1 Data Collection and Annotation
### 3.1 数据采集与标注


Data collection: During data collection, collectors wear the Oracle Suite in various scenarios to record natural manipulation behaviors. Each task is recorded once according to the Standard Operating Procedure of the scenario. Recording sessions last approximately 10 minutes and include start and end event markers. Captured signals include tactile signals from pressure sensors, wrist images, first-person chest images, corresponding camera calibration parameters, left and right hand trajectory data, and left and right hand skeleton information. Samples of data annotations are shown in Figure 6
数据采集：在数据采集过程中，采集员佩戴 Oracle Suite 在各种场景中记录自然操作行为。每个任务均按照场景的标准操作程序记录一次。记录过程持续约 10 分钟，并包含开始和结束事件标记。采集的信号包括压力传感器的触觉信号、腕部图像、第一人称胸部图像、相应的相机标定参数、左右手轨迹数据以及左右手骨架信息。数据标注示例如图 6 所示。


After data collection, we curate the WIYH dataset via a three-stage annotation pipeline (Figure 4), comprising atomic action annotation, perception annotation, and vision-language annotation.
数据采集完成后，我们通过三阶段标注流水线（图 4）对 WIYH 数据集进行整理，包括原子动作标注、感知标注和视觉语言标注。


Atomic action annotation: To construct data suitable for robotic foundation model training, we design a four-stage hybrid annotation process for atomic action annotation that decomposes a task in the standard operating procedure into multiple atomic segments, where each segment is termed a subtask.
原子动作标注：为构建适用于机器人基础模型训练的数据，我们设计了一个四阶段混合标注流程，将标准操作程序中的任务分解为多个原子片段，每个片段称为一个子任务。


- Video Segmentation: The process initiates with video segmentation, where Qwen2.5-VL-72B 3 segments the recorded data for each task into semantic atomic fragments serving as basic units for annotation.
- 视频分割：流程始于视频分割，利用 Qwen2.5-VL-72B 3 将每个任务的记录数据分割为语义原子片段，作为标注的基本单元。


- Task definition: Humans define a preset action termed a "skill" and an operation target object for each atomic fragment.
- 任务定义：人工为每个原子片段定义预设动作（称为“技能”）及操作目标对象。


- Data checking: Manual reviewers check boundary and task definition accuracy to correct errors.
- 数据核对：人工审核员检查边界和任务定义的准确性，以修正错误。


- Instruction Augmentation: Post-processing combines the annotated preset actions and target object names into complete atomic instructions, which are augmented and enhanced using Qwen2.5-VL-72B 3.
- 指令增强：后处理阶段将标注的预设动作与目标对象名称组合成完整的原子指令，并使用 Qwen2.5-VL-72B 3 进行增强与优化。


Perception annotation: We use vision foundation models 27, 50 to pre-annotate scene elements, followed by manual selection and correction. This includes generating instance masks by applying SAM 27 to human-provided prompt points and estimating scene depth from first-person binocular images using the FoundationStereo [50] model.
感知标注：我们使用视觉基础模型 27, 50 对场景元素进行预标注，随后进行人工筛选与修正。这包括通过应用 SAM 27 对人工提供的提示点生成实例掩码，以及利用 FoundationStereo [50] 模型从第一人称双目图像中估计场景深度。


Vision-language annotation is generated through manual curation and includes several components.
视觉语言标注通过人工整理生成，包含多个组成部分。


- CoT Reasoning: We use Qwen2.5-VL-72B [3] to generate a thought process indicating the position of each subtask within a complete task and when to enter the next subtask for long-horizon task decomposition, where generated thought data is reviewed and corrected by humans.
- 思维链推理（CoT Reasoning）：我们使用 Qwen2.5-VL-72B [3] 生成思维过程，指明长程任务分解中每个子任务在完整任务中的位置及进入下一子任务的时机，生成的思维数据经由人工审核与修正。


- Spatial Referring (SR): We select key frames near the action start where the operation target object is clearly visible for some subtasks and add spatial relation terms based on the atomic instruction to uniquely refer to a single target area in the scene.
- 空间指代（SR）：我们选取动作开始附近、操作目标物体清晰可见的关键帧，并基于原子指令添加空间关系术语，以唯一指代场景中的单个目标区域。


- Subtask Prediction (SP): We select tasks containing 4 to 5 subtasks and use the next subtask after the current one as the correct option, while generating three logically reasonable incorrect options based on the task description.
- 子任务预测（SP）：我们选取包含 4 到 5 个子任务的任务，将当前子任务之后的子任务作为正确选项，并根据任务描述生成三个逻辑合理的错误选项。


- Completion Verification (CV): We randomly crop 0% to 35% of each subtask for a task and use the atomic instruction of each subtask, as well as the cropped fragment, as input to label the task progress status regarding whether the task is completed.
- 完成度验证（CV）：我们随机裁剪每个任务中 0% 到 35% 的子任务片段，并以各子任务的原子指令及裁剪片段作为输入，标注任务是否完成的进度状态。


Notably, subtask prediction, spatial referring, and progress verification are annotated on only a subset of the whole WIYH data to construct an evaluation set for validating the capabilities of existing VLMs in human-centric scenarios.
值得注意的是，子任务预测、空间指代和进度验证仅在 WIYH 全部数据的子集上进行标注，旨在构建评估集，以验证现有 VLM 在以人为中心的场景中的能力。


### 3.2 Data Statistics
### 3.2 数据统计


Figure 5 summarizes key statistics of our WIYH dataset. We first analyze the distribution of tasks across scenes. As shown in Figure 5(a), WIYH contains a diverse set of daily environments, including banquet, laundry, logistics, hotel, department, office, supermarket, industry, cleaning, and candlelight settings. The Sankey diagram illustrates the total accumulated duration of each task within its corresponding scene.
图 5 总结了 WIYH 数据集的关键统计信息。我们首先分析了任务在不同场景下的分布。如图 5(a) 所示，WIYH 涵盖了多样化的日常环境，包括宴会、洗衣、物流、酒店、百货、办公室、超市、工业、清洁和烛光场景。桑基图展示了每个任务在其对应场景中的累计总时长。


Scenes such as supermarkets and laundry include long-horizon activities with substantial execution time, while hotels and candlelight contain shorter but highly structured routines. This diversity in task-scene composition highlights the wide operational range covered by WIYH. Action-level temporal characteristics are shown in Figure 5(b). For each scene, we compute the average execution duration of fine-grained actions. The dataset exhibits substantial variation across action categories, from short, instantaneous operations such as "wipe" and "insert" to long-horizon manipulation such as "smooth out" or "fold lay". These temporal differences reflect the heterogeneous complexity of real-world household and industrial workflows.
超市和洗衣等场景包含执行时间较长的长程活动，而酒店和烛光场景则包含较短但高度结构化的日常流程。这种任务-场景构成的多样性凸显了 WIYH 覆盖的广泛操作范围。动作级的时间特征如图 5(b) 所示。对于每个场景，我们计算了细粒度动作的平均执行时长。数据集在动作类别间表现出显著差异，从“擦拭”和“插入”等短时瞬时操作，到“抚平”或“折叠”等长程操作。这些时间差异反映了现实生活中家庭和工业工作流程的异构复杂性。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_850cf6.jpg"/>



Fig. 5: Overview of dataset statistics, including task-scene relationships, action durations, annotation distributions, and word clouds of manipulation target objects and skills. The dataset spans a wide spectrum of real-world scenarios, from industrial to daily life (e.g., factories, hotels, apartments, supermarkets). For each scenario, it provides task and subtask annotations crucial for instruction-action alignment and task decomposition in robot learning. The chart presents multi-dimensional statistics of these annotations.
图 5：数据集统计概览，包括任务-场景关系、动作时长、标注分布以及操作目标物体和技能的词云。该数据集涵盖了从工业到日常生活的广泛现实场景（如工厂、酒店、公寓、超市）。对于每个场景，它提供了对机器人学习中的指令-动作对齐和任务分解至关重要的任务及子任务标注。图表展示了这些标注的多维统计信息。


Figure 5(c) reports the distribution of annotation types. WIYH includes rich multimodal and task-level supervision, consisting of RGB, depth, action, instruction, reasoning, mask, tactile, and calibration annotations. Among them, RGB and calibration frames appear most frequently, while reasoning and tactile annotations, though smaller in quantity, provide high-level semantic cues crucial for complex manipulation and planning tasks.
图 5(c) 报告了标注类型的分布。WIYH 包含丰富的多模态和任务级监督信息，由 RGB、深度、动作、指令、推理、掩码、触觉和标定标注组成。其中，RGB 和标定帧出现频率最高，而推理和触觉标注虽然数量较少，但提供了对复杂操作和规划任务至关重要的深层语义线索。


To better understand the manipulation space in WIYH, Figure 5(d-e) visualizes word clouds of the annotated target objects and skills. The dataset covers a broad range of manipulable objects-including clothes, linens, cartons, cups, and tableware - spanning deformable, rigid, and articulated categories. Likewise, the skill vocabulary encompasses diverse manipulation primitives such as take, place, rotate, unfold, align, push, and search, representing both low-level motor actions and higher-level task strategies. These distributions further demonstrate that WIYH captures a wide spectrum of real-world manipulation behaviors. More data, demonstrations, and statistics will be provided in the supplementary materials.
为了更好地理解 WIYH 中的操作空间，图 5(d-e) 可视化了标注目标物体和技能的词云。数据集涵盖了广泛的可操作物体，包括衣物、布草、纸箱、杯子和餐具，跨越了可变形、刚性和关节类物体。同样，技能词汇涵盖了多种操作原语，如拿取、放置、旋转、展开、对齐、推动和搜索，既代表了低级运动动作，也代表了更高级的任务策略。这些分布进一步证明了 WIYH 捕捉到了广泛的现实世界操作行为。更多数据、演示和统计信息将在补充材料中提供。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_1e3047.jpg"/>



Fig. 6: Data annotation samples cross different scenes. The example of human-centric data annotations, including depth, mask, action, and task descriptions in four scenarios.
图 6：不同场景下的数据标注样本。展示了四个场景中以人为中心的数据标注示例，包括深度、掩码、动作和任务描述。


## 4 Benchmark and Application
## 4 基准测试与应用


### 4.1 Benchmark: Human-centric Vision-Language (HVL)
### 4.1 基准测试：以人为中心的视觉-语言（HVL）


In this section, we construct three evaluation tasks based on the vision-language annotations introduced in Section 3 These tasks are designed to evaluate whether existing Multimodal Large Language Models (MLLMs) can achieve a genuine understanding of human-centric manipulation tasks in highly generalized, in-the-wild environments.
在本节中，我们基于第 3 节中介绍的视觉-语言标注构建了三个评估任务。这些任务旨在评估现有的多模态大语言模型（MLLM）是否能在高度泛化的野外环境中，实现对以人为中心的操作任务的真正理解。


We define the following three evaluation tasks and test several closed-source and open-source MLLMs, including GPT-40 [23], Doubao-seed [4], and Qwen [55]. Detailed experimental setups and data statistics are provided in the supplementary material. (1) Spatial Referring (S.R.): Given a human manipulation image and a description of the spatial relationships among multiple objects in the scene, the model is required to output the pixel coordinates of a specified region. Performance is measured by the probability that the predicted point lies within the target area. (2) Subtask Prediction (S.P.): Given a piece of human manipulation video, the current task description, and four candidate options for the next action, the model must select the correct subsequent action. (3) Completion Verification (C.V.): Given a piece of human manipulation video and the current task description, the model needs to determine whether the task has been completed. Evaluation details will be provided in the supplementary material. As shown in Table 2, scores for the S.R. task and C.V. task are generally low, whereas scores for the S.P. task remain relatively high. This discrepancy reflects the characteristics of existing VLMs, which demonstrate competent performance in high-level task decomposition but exhibit limitations in dynamic progress modeling and spatial understanding in specific embodied tasks. Specifically, on the S.R. task, scores of all VLMs are below 0.5 , and on the C.V. task, all models perform slightly above 0.5 , suggesting that their abilities in such contexts remain limited. Notably, Qwen3-VL-Plus, which was pre-trained on spatial-understanding data, achieves the best performance on the S.R. task. These experimental results collectively demonstrate that general-purpose, vanilla VLMs still fall short in embodied tasks. Our WIYH dataset, with its rich annotations of human activities, scene depth, and object masks, presents a promising foundation for future embodied pre-training.
我们定义了以下三项评估任务，并测试了多个闭源和开源多模态大模型（MLLM），包括 GPT-4o [23]、豆包-seed [4] 和通义千问 [55]。详细的实验设置和数据统计见补充材料。(1) 空间指代（S.R.）：给定一张人类操作图像及场景中多个物体间空间关系的描述，模型需输出指定区域的像素坐标。性能通过预测点落在目标区域内的概率来衡量。(2) 子任务预测（S.P.）：给定一段人类操作视频、当前任务描述以及四个后续动作候选选项，模型必须选择正确的后续动作。(3) 完成度验证（C.V.）：给定一段人类操作视频和当前任务描述，模型需判断任务是否已完成。评估细节将在补充材料中提供。如表 2 所示，S.R. 任务和 C.V. 任务的得分普遍较低，而 S.P. 任务的得分则相对较高。这种差异反映了现有视觉语言模型（VLM）的特性：它们在高级任务分解方面表现出色，但在特定具身任务的动态进度建模和空间理解上存在局限。具体而言，在 S.R. 任务上，所有 VLM 的得分均低于 0.5；在 C.V. 任务上，所有模型的表现略高于 0.5，表明它们在此类场景下的能力依然有限。值得注意的是，经过空间理解数据预训练的 Qwen2-VL-Plus 在 S.R. 任务上取得了最佳表现。这些实验结果共同表明，通用的基础 VLM 在具身任务中仍有不足。我们的 WIYH 数据集凭借其丰富的人类活动标注、场景深度和物体掩码信息，为未来的具身预训练奠定了坚实基础。


Table 2: Comparison of VLMs on spatial referring, subtask prediction, and completion verification tasks.
表 2：VLM 在空间指代、子任务预测和完成度验证任务上的对比。


<table><tr><td>Model</td><td>S.R. (%)</td><td>S.P. (%)</td><td>C.V. (%)</td></tr><tr><td>GPT-40 35</td><td>10.04</td><td>73.40</td><td>51.18</td></tr><tr><td>Qwen3-VL-Plus [43]</td><td>46.53</td><td>71.75</td><td>55.89</td></tr><tr><td>Doubao-Seed-1.6-Vision [9]</td><td>39.83</td><td>76.84</td><td>51.96</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td>25.58</td><td>69.21</td><td>56.73</td></tr></table>
<table><tbody><tr><td>模型</td><td>成功率 (%)</td><td>成功精度 (%)</td><td>综合得分 (%)</td></tr><tr><td>GPT-4o 35</td><td>10.04</td><td>73.40</td><td>51.18</td></tr><tr><td>Qwen3-VL-Plus [43]</td><td>46.53</td><td>71.75</td><td>55.89</td></tr><tr><td>豆包-Seed-1.6-Vision [9]</td><td>39.83</td><td>76.84</td><td>51.96</td></tr><tr><td>Qwen2-VL-4B-Instruct</td><td>25.58</td><td>69.21</td><td>56.73</td></tr></tbody></table>


### 4.2 Application: Human-centric World Modeling
### 4.2 应用：以人为中心的世界建模


This subsection evaluates the dataset's utility for supervised end-to-end 4D reconstruction using Gaussian Splatting (GS). By providing high-quality RGB images, depth maps, and camera poses, WIYH supports shape and motion modeling for dynamic scenes. We extracted geometry using MegaSAM 33, tracked pixel trajectories of dynamic objects with TAPIR 14, and performed scene reconstruction with Shape of Motion 47. The successful application of our dataset in spatial perception tasks like 4D Gaussian Splatting (4DGS) demonstrates the critical role of its rich annotations, particularly depth information, in enhancing spatial understanding and representation.
本小节评估了该数据集在使用高斯 splatting（GS）进行有监督的端到端 4D 重建方面的效用。通过提供高质量的 RGB 图像、深度图和相机位姿，WIYH 支持动态场景的形状和运动建模。我们使用 MegaSAM 33 提取几何信息，使用 TAPIR 14 跟踪动态对象的像素轨迹，并使用 Shape of Motion 47 进行场景重建。我们的数据集在 4D 高斯 splatting（4DGS）等空间感知任务中的成功应用，证明了其丰富的标注（尤其是深度信息）在增强空间理解和表示方面的关键作用。


As shown in Table 3, photometric metrics remain relatively consistent, whereas geometric metrics vary notably across tasks. These differences highlight the inherent challenges in reconstructing dynamic scenes, where accurate geometry prediction is particularly complex in motion-rich environments. This observation suggests promising directions for future research in 4D reconstruction for robotic manipulation, particularly methods that improve geometric understanding in complex dynamic scenes.
如表3所示，光度指标保持相对一致，而几何指标在不同任务间存在显著差异。这些差异凸显了重建动态场景的固有挑战，即在运动丰富的环境中，精确的几何预测尤为复杂。这一观察结果为机器人操作的4D重建研究指明了方向，特别是那些旨在提升复杂动态场景中几何理解能力的方法。


Figure 7 presents the 4D reconstruction results at various time steps, showcasing the dataset's potential for dynamic scene modeling. By capturing high-quality dynamic scenes, WIYH can help leverage Gaussian Splatting (GS) as an effective 3D tokenizer, advance spatial understanding, and enrich Real2Sim pipelines and neural-network-based simulation.
图7展示了不同时间步下的4D重建结果，体现了该数据集在动态场景建模方面的潜力。通过捕捉高质量的动态场景，WIYH有助于将高斯泼溅（GS）作为有效的3D分词器，推动空间理解的发展，并丰富Real2Sim流程及基于神经网络的仿真。


Table 3: Comparison of photometric and geometric metrics for different tasks.
表3：不同任务的光度与几何指标对比。


<table><tr><td rowspan="2">Scene</td><td rowspan="2">Task</td><td colspan="3">Photometry</td><td colspan="2">Geometry</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS $\downarrow$</td><td>A.R.↓</td><td>${\delta }_{1} \uparrow$</td></tr><tr><td>1</td><td>Pouring Wine</td><td>20.15</td><td>0.815</td><td>0.206</td><td>1.04</td><td>52.01</td></tr><tr><td>2</td><td>Packing Clothes</td><td>20.02</td><td>0.656</td><td>0.423</td><td>2.89</td><td>17.83</td></tr><tr><td>3</td><td>Packing Cargo</td><td>18.83</td><td>0.749</td><td>0.552</td><td>3.52</td><td>13.79</td></tr></table>
<table><tbody><tr><td rowspan="2">场景</td><td rowspan="2">任务</td><td colspan="3">光度</td><td colspan="2">几何</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS $\downarrow$</td><td>A.R.↓</td><td>${\delta }_{1} \uparrow$</td></tr><tr><td>1</td><td>倒酒</td><td>20.15</td><td>0.815</td><td>0.206</td><td>1.04</td><td>52.01</td></tr><tr><td>2</td><td>打包衣物</td><td>20.02</td><td>0.656</td><td>0.423</td><td>2.89</td><td>17.83</td></tr><tr><td>3</td><td>装载货物</td><td>18.83</td><td>0.749</td><td>0.552</td><td>3.52</td><td>13.79</td></tr></tbody></table>


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c93195.jpg"/>



Fig. 7: 4D Reconstruction Result. For the pouring wine task, we present the 4DGS reconstruction results across multiple timestamps. The visualizations include the rendered image, estimated depth map, and predicted 4D motion field. The results demonstrate that our WIYH dataset enables clean and accurate 4D reconstruction, even in challenging, dynamic action scenarios.
图 7：4D 重建结果。针对倒酒任务，我们展示了多个时间戳下的 4DGS 重建结果。可视化内容包括渲染图像、估计深度图以及预测的 4D 运动场。结果表明，即使在具有挑战性的动态动作场景中，我们的 WIYH 数据集也能实现清晰且准确的 4D 重建。


In addition, we further present the experimental results of video generation in the supplementary materials.
此外，我们还在补充材料中进一步展示了视频生成的实验结果。


## 5 Human-centric Manipulation
## 5 以人为中心的操纵


### 5.1 Experiment Objective and Implementation
### 5.1 实验目标与实现


In this section, our objective is to validate the efficacy of the WIYH data collection system and dataset for robotics manipulation. To this end, we formulate two core research questions. (1) Does utilizing the WIYH dataset as pre-training data enhance the embodied manipulation capabilities of the VLA model? (2) Does retargeting WIYH data to dexterous hand actions improve dexterous manipulation performance? To address these questions, we design two real-world manipulation experiments: (1) cross-embodiment pretraining and (2) re-targeting co-training.
本节旨在验证 WIYH 数据采集系统及数据集在机器人操纵任务中的有效性。为此，我们提出了两个核心研究问题：（1）使用 WIYH 数据集作为预训练数据能否增强 VLA 模型的具身操纵能力？（2）将 WIYH 数据重定向至灵巧手动作能否提升灵巧操纵性能？为解决这些问题，我们设计了两个真实世界操纵实验：（1）跨具身预训练；（2）重定向协同训练。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_f44a9f.jpg"/>



Fig. 8: Experiment setup. The left side demonstrates samples of rose insertion and gift packing in the cross-embodiment task. The right side shows data collection in the co-training task. Robot data contains single-object manipulation data starting from a limited set of initial poses, whereas human-centric data contains data from more complex scenes and covers a broader action space.
图 8：实验设置。左侧展示了跨具身任务中玫瑰插入和礼品包装的样本。右侧展示了协同训练任务中的数据采集过程。机器人数据包含从有限初始位姿开始的单物体操纵数据，而以人为中心的数据则包含来自更复杂场景的数据，涵盖了更广阔的动作空间。


In the cross-embodiment pretraining experiment, we adopt StarVLA-PI [13] as the base model. We pre-train StarVLA-PI using WIYH data with atomic instruction annotations for both VLM and VLA data. For post-training, we select two tasks in a real-world environment: rose insertion and gift packing, and we collect 150 gripper-based teleoperation manipulation samples for each task, as shown on the left side of Figure 8. We compare the success rates across three settings, including VLM pre-training, VLA pre-training, and without pretraining. VLA pretraining refers to loading all pre-trained VLA weights during post-training, while VLM pretraining denotes loading only the VLM weights. During evaluation, each setting of each task is tested 10 times with object positions and target objects varied in each trial.
在跨具身预训练实验中，我们采用 StarVLA-PI [13] 作为基础模型。我们使用带有 VLM 和 VLA 原子指令标注的 WIYH 数据对 StarVLA-PI 进行预训练。在后训练阶段，我们选择了真实环境中的两项任务：玫瑰插入和礼品包装，并为每项任务采集了 150 个基于夹爪的遥操作操纵样本，如图 8 左侧所示。我们比较了三种设置下的成功率，包括 VLM 预训练、VLA 预训练以及无预训练。VLA 预训练指在后训练期间加载所有预训练的 VLA 权重，而 VLM 预训练仅加载 VLM 权重。在评估过程中，每项任务的每种设置均测试 10 次，且每次试验中物体位置和目标物体均有所变化。


In the retargeting co-training experiment, we construct a combined dataset using UMI robot data ${D}_{r}$ collected in constrained environments and data ${D}_{h}$ of the same task collected using the Oracle Suite. The robot system employs a low-degree-of-freedom dexterous hand as the end-effector $H$ ,and the hand data from ${D}_{h}$ is retargeted to match the degrees of freedom of $H$ . We evaluate the effectiveness of the human-centric data collected by Oracle Suite across various dexterous grasping tasks. Implementation details will be provided in the supplementary materials. Notably, the right side of figure 8 shows scene examples and the data collection setup. The collected robot data ${D}_{r}$ always starts from a limited number of initial poses in the single-object scene. In contrast, human-centric data ${D}_{h}$ contains only manipulation sequences from multi-object cluttered scenes,making a larger observation domain and action space than ${D}_{r}$ .
在重定向协同训练实验中，我们构建了一个组合数据集，其中包含在受限环境中采集的 UMI 机器人数据 ${D}_{r}$，以及使用 Oracle Suite 采集的相同任务数据 ${D}_{h}$。机器人系统采用低自由度灵巧手作为末端执行器 $H$，并将 ${D}_{h}$ 中的手部数据重定向以匹配 $H$ 的自由度。我们评估了 Oracle Suite 采集的以人为中心的数据在各种灵巧抓取任务中的有效性。实现细节将在补充材料中提供。值得注意的是，图 8 右侧展示了场景示例和数据采集设置。采集的机器人数据 ${D}_{r}$ 始终从单物体场景中有限数量的初始位姿开始。相比之下，以人为中心的数据 ${D}_{h}$ 仅包含来自多物体杂乱场景的操纵序列，相比 ${D}_{r}$ 具有更大的观测域和动作空间。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c9473c.jpg"/>



Fig. 9: Cross-embodiment pretrain loss.
图 9：跨具身预训练损失。


Table 4: Comparison of real-world performance in cross-embodiment pretraining experiment.
表 4：跨具身预训练实验中真实世界性能的比较。


<table><tr><td>Method</td><td colspan="3">Rose Insertion Gift Packing AVG</td></tr><tr><td>w/o Pretrain</td><td>1/10</td><td>2/10</td><td>15%</td></tr><tr><td>VLM Pretrain</td><td>3/10</td><td>3/10</td><td>30%</td></tr><tr><td>VLA Pretrain</td><td>6/10</td><td>8/10</td><td>70%</td></tr></table>
<table><tbody><tr><td>方法</td><td colspan="3">玫瑰插入 礼品包装 平均值</td></tr><tr><td>无预训练</td><td>1/10</td><td>2/10</td><td>15%</td></tr><tr><td>VLM预训练</td><td>3/10</td><td>3/10</td><td>30%</td></tr><tr><td>VLA预训练</td><td>6/10</td><td>8/10</td><td>70%</td></tr></tbody></table>


### 5.2 Results and In-depth Discussion
### 5.2 结果与深入讨论


In this section, we analyze the results of two experiments to demonstrate how WIYH empowers robotic manipulation.
本节通过分析两项实验的结果，展示 WIYH 如何赋能机器人操作。


For the cross-embodiment pretraining experiments, we demonstrate the quantitative result and the training loss in Table 4 and in Figure 9, respectively. As indicated by success rates, utilizing WIYH for VLA pre-training yields the most significant performance improvements, from 15% to 70%. Besides, as illustrated in Figure 9. VLA pre-training achieves lower loss and smoother gradient convergence than VLM pre-training and no pre-training. This efficacy remains robust given the cross-embodiment gap between the pre-training data (human hands) and post-training data (robot gripper).
针对跨形态预训练实验，我们分别在表4和图9中展示了定量结果与训练损失。成功率表明，利用 WIYH 进行 VLA 预训练带来了最显著的性能提升，从 15% 提高至 70%。此外，如图9所示，相比 VLM 预训练和无预训练，VLA 预训练实现了更低的损失和更平滑的梯度收敛。尽管预训练数据（人手）与训练后数据（机器人夹爪）之间存在跨形态差异，但这种有效性依然稳健。


For the retargeting co-training experiments, we demonstrate the quantitative and qualitative results in Table 5 and in Figure 10, respectively. In the single-object scenario, the robot arm begins execution from a random initial pose during testing, which differs from the initial pose used in the robot data ${D}_{r}$ . In the multi-object cluttered scenario,we primarily evaluate the performance of the co-training strategy under conditions of varying scene height, changes in the target object's pose, and increased occlusion. The configurations of these test scenarios and qualitative experimental results are illustrated in Figure 10. Quantitative results are summarized in Table 5. (1) In the single-object scenario, co-training the policy with both robot data and human-centric data leads to an improvement in the success rate of more than 13%. Although the motion precision in the human-centric data is limited, its diverse action distribution enhances the policy's generalization capability to unseen initial states. (2) Policies trained solely on robot data exhibit minimal generalization capability in complex scenarios. Simply increasing data quantity from 200 clips to 500 clips yields marginal performance improvement from 0% to 8%. However, after co-training with human-centric data, the success rate increases by 52%. This demonstrates that incorporating human-centric data significantly improves the policy's ability to interpret complex scenes. Therefore, the co-training approach effectively introduces both the action domain and observation domain of human-centric data into the robot dataset, thereby enhancing the generalization capability of the learned policy.
针对重定向协同训练实验，我们分别在表5和图10中展示了定量与定性结果。在单物体场景中，机器人手臂在测试时从随机初始位姿开始执行，这与机器人数据 ${D}_{r}$ 中使用的初始位姿不同。在多物体杂乱场景中，我们主要评估了协同训练策略在不同场景高度、目标物体位姿变化及遮挡增加条件下的性能。这些测试场景的配置及定性实验结果如图10所示，定量结果总结于表5。(1) 在单物体场景中，将策略与机器人数据及以人为中心的数据进行协同训练，成功率提升了超过 13%。尽管以人为中心的数据在运动精度上有限，但其多样的动作分布增强了策略对未见初始状态的泛化能力。(2) 仅在机器人数据上训练的策略在复杂场景中几乎没有泛化能力。单纯将数据量从 200 段增加到 500 段，性能提升微乎其微，仅从 0% 提高到 8%。然而，在与以人为中心的数据进行协同训练后，成功率提高了 52%。这表明引入以人为中心的数据显著提升了策略对复杂场景的理解能力。因此，协同训练方法有效地将以人为中心数据的动作域和观测域引入机器人数据集，从而增强了所学策略的泛化能力。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_0eb3c7.jpg"/>



Fig. 10: Real-robot manipulation experiments. We present the comparison of manipulation policy performance under four novel task settings, where the policies are trained either exclusively on robot data collected or co-trained using both robot data and annotated human-centric data.
图10：真实机器人操作实验。我们展示了在四种新颖任务设置下操作策略性能的对比，其中策略分别仅使用采集的机器人数据进行训练，或使用机器人数据与标注的以人为中心数据进行协同训练。


Table 5: A comparison of results on two manipulation scenarios trained with data from different sources is presented. The datasets include robot data and human-centric data. And the success rate (SR) was reported.
表5：展示了使用不同来源数据训练的两种操作场景的结果对比。数据集包括机器人数据和以人为中心的数据，并报告了成功率（SR）。


<table><tr><td rowspan="2">Scenario</td><td colspan="2">Data Sources and Amounts (Clips)</td><td rowspan="2">SR (%)</td></tr><tr><td>Robot data amount</td><td>Human data amount</td></tr><tr><td rowspan="4">Single-object Scene</td><td>200</td><td>0</td><td>20</td></tr><tr><td>200</td><td>800</td><td>23.3 (+3.3)</td></tr><tr><td>500</td><td>0</td><td>43.3</td></tr><tr><td>500</td><td>800</td><td>56.7 (+13.4)</td></tr><tr><td rowspan="4">Cluttered Scene</td><td>200</td><td>0</td><td>0.0</td></tr><tr><td>200</td><td>800</td><td>32.0 (+32.0)</td></tr><tr><td>500</td><td>0</td><td>8.0</td></tr><tr><td>500</td><td>800</td><td>${60.0}\left( {+{52.0}}\right)$</td></tr></table>
<table><tbody><tr><td rowspan="2">场景</td><td colspan="2">数据来源与数量（片段）</td><td rowspan="2">成功率 (%)</td></tr><tr><td>机器人数据量</td><td>人类数据量</td></tr><tr><td rowspan="4">单物体场景</td><td>200</td><td>0</td><td>20</td></tr><tr><td>200</td><td>800</td><td>23.3 (+3.3)</td></tr><tr><td>500</td><td>0</td><td>43.3</td></tr><tr><td>500</td><td>800</td><td>56.7 (+13.4)</td></tr><tr><td rowspan="4">杂乱场景</td><td>200</td><td>0</td><td>0.0</td></tr><tr><td>200</td><td>800</td><td>32.0 (+32.0)</td></tr><tr><td>500</td><td>0</td><td>8.0</td></tr><tr><td>500</td><td>800</td><td>${60.0}\left( {+{52.0}}\right)$</td></tr></tbody></table>


## 6 Conclusion
## 6 结论


In this paper, we present the WIYH ecosystem, a cornerstone for embodied intelligence research in real-world environments. We make three key contributions: the Oracle Suite, a wearable system enabling in-the-wild data capture with integrated markerless auto-labeling; the large-scale WIYH dataset, offering diverse, skilled human manipulation sequences with rich multimodal streams; and a suite of comprehensive benchmarks, supported by extensive atomic-action and vision-language annotations, for holistic evaluation from perception to physical interaction. Our benchmarks reveal that current models still lack the fine-grained spatial and causal reasoning required for robust embodied operation. The WIYH dataset, with its unique scale and rich annotations, is designed to bridge this gap. In the future, we will open-source the whole dataset and hardware design to promote human-centric manipulation research.
在本文中，我们提出了 WIYH 生态系统，这是现实环境中具身智能研究的基石。我们做出了三项关键贡献：Oracle Suite，一套支持野外数据采集并集成无标记自动标注的可穿戴系统；大规模 WIYH 数据集，提供多样化、高技能的人类操作序列及丰富的多模态流；以及一套全面的基准测试，辅以详尽的原子动作和视觉语言标注，用于从感知到物理交互的整体评估。我们的基准测试表明，当前模型仍缺乏稳健具身操作所需的细粒度空间和因果推理能力。WIYH 数据集凭借其独特的规模和丰富的标注，旨在弥补这一差距。未来，我们将开源整个数据集和硬件设计，以推动以人为中心的操作研究。


## Appendix
## 附录


## A World In Your Hands Ecosystem
## A World In Your Hands 生态系统


### A.1 Hardware Details
### A.1 硬件细节


Camera: We primarily use two fisheye cameras below each wrist and two fisheye cameras on the chest as the raw RGB signal sources. These cameras feature a ${180}^{ \circ  }$ field of view with an original resolution of ${1536} \times  {1920}$ pixels.
相机：我们主要使用每只手腕下方两台鱼眼相机和胸前两台鱼眼相机作为原始 RGB 信号源。这些相机具有 ${180}^{ \circ  }$ 的视场角，原始分辨率为 ${1536} \times  {1920}$ 像素。


Tactile: We utilize resistive pressure sensors with a resolution of $5\mathrm{{mN}}$ and a range of ${0.2}\mathrm{\;N}$ to ${50}\mathrm{\;N}$ distributed across the fingertips of all fingers.
触觉：我们使用分辨率为 $5\mathrm{{mN}}$、量程为 ${0.2}\mathrm{\;N}$ 至 ${50}\mathrm{\;N}$ 的电阻式压力传感器，分布在所有手指的指尖。


Autolabel Pipeline: The autolabeling pipeline involves online and offline algorithms. Specifically, the online localization module performs lightweight coarse positioning using visual-inertial odometry (VIO) algorithms 37 and IR detection, while the offline stage refines these coarse estimates through Structure-from-Motion (SfM) algorithms 40].
自动标注流水线：自动标注流水线包含在线和离线算法。具体而言，在线定位模块使用视觉惯性里程计 (VIO) 算法 37 和红外检测执行轻量级粗定位，而离线阶段则通过运动恢复结构 (SfM) 算法 40] 对这些粗略估计进行精炼。


### A.2 Data Collection Efficiency
### A.2 数据采集效率


The Oracle Suite pipeline follows a Collection, Annotation, Inspection, Archiving workflow for scalable data production. A single collector operating 8 hours per day generates approximately 1.8 TB of valid multimodal data. At scale, 100 collectors over 30 days yield roughly 5.4 PB of raw data (stored in cloud repositories). Efficiency is further improved through standardized capture protocols and automated cloud-based annotation.
Oracle Suite 流水线遵循“采集、标注、检查、归档”的工作流程，以实现可扩展的数据生产。单名采集员每天工作 8 小时，可生成约 1.8 TB 的有效多模态数据。按规模计算，100 名采集员在 30 天内可产生约 5.4 PB 的原始数据（存储于云端仓库）。通过标准化的采集协议和自动化的云端标注，效率得到了进一步提升。


We compare the efficiency of different data collection paradigms in Table 6 Efficiency: taking the rose insertion task in our experiments as an example, given 8 hours as collection time, teleoperation can collect about 150 data clips; UMI-like methods, such as DexUMI [54], can collect about 400 data clips; our oracle suite can collect 720 data clips; and VR can collect around 800 data clips. Quality: VR-based hand tracking only provides 2D hand skeletons and suffers from visual occlusion. In contrast. In comparison, the Oracle Suite provides accurate $3\mathrm{D}$ hand skeletons,even if under visual occlusion.
我们在表 6 中比较了不同数据采集范式的效率。效率方面：以我们实验中的插花任务为例，在 8 小时采集时间内，遥操作可采集约 150 个数据片段；类似 UMI 的方法（如 DexUMI [54]）可采集约 400 个数据片段；我们的 Oracle Suite 可采集 720 个数据片段；而 VR 可采集约 800 个数据片段。质量方面：基于 VR 的手部追踪仅提供 2D 手部骨架，且易受视觉遮挡影响。相比之下，即使在视觉遮挡下，Oracle Suite 也能提供精确的 $3\mathrm{D}$ 手部骨架。


Table 6: Comparison of dexterous data collection paradigms.
表 6：灵巧手数据采集范式对比。


<table><tr><td></td><td>Teleop</td><td>DexUMI</td><td>VR</td><td>Oracle Suit (Ours)</td></tr><tr><td>Efficiency</td><td>☆☆☆☆☆</td><td>★☆☆☆☆</td><td>★★★★☆</td><td>★★★☆☆</td></tr><tr><td>Quality</td><td>★★★★☆</td><td>★★★☆☆</td><td>★☆☆☆☆</td><td>★★★☆☆</td></tr><tr><td>End-effector</td><td>Dext</td><td>Dexterous</td><td>Hands</td><td>Hands</td></tr></table>
<table><tbody><tr><td></td><td>遥操作</td><td>DexUMI</td><td>虚拟现实</td><td>Oracle Suit（本文方法）</td></tr><tr><td>效率</td><td>☆☆☆☆☆</td><td>★☆☆☆☆</td><td>★★★★☆</td><td>★★★☆☆</td></tr><tr><td>质量</td><td>★★★★☆</td><td>★★★☆☆</td><td>★☆☆☆☆</td><td>★★★☆☆</td></tr><tr><td>末端执行器</td><td>灵巧</td><td>灵巧</td><td>手</td><td>手</td></tr></tbody></table>


## B World In Your Hands Dataset
## B World In Your Hands 数据集


### B.1 Detailed Task-Skill Counting Statistics
### B.1 任务-技能计数统计详情


- We further supplement the visualization of the number of skills corresponding to each task. Due to space limitations, we report statistics based on only 25% of the dataset, as shown in Figure 11. Please zoom in electronically to inspect the details. The full visualization analysis will be updated subsequently on the dataset website.
- 我们进一步补充了每个任务对应技能数量的可视化。受限于篇幅，我们仅基于 25% 的数据集展示统计结果，如图 11 所示。请电子放大以查看细节。完整可视化分析将在数据集网站后续更新。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_a72a5d.jpg"/>



Fig. 11: Task-skill count visualization of 25% dataset.
图 11：25% 数据集的任务-技能计数可视化。


### B.2 Data Annotation Details.
### B.2 数据标注详情。


Raw videos are structured into a three-tier hierarchy: Scene - Task - Subtask by annotators. Each subtask corresponds to the minimal object-centered action unit or a temporally cohesive action sequence. (1) Atomic (Subtask) Annotation:
原始视频由标注员按“场景-任务-子任务”三层级结构进行组织。每个子任务对应一个以物体为中心的最小动作单元或时间上连贯的动作序列。(1) 原子（子任务）标注：


- Actions (Skills): Drawn from predefined operation sets (e.g., PICK, PLACE INTO, SEARCH).
- 动作（技能）：选自预定义的动作集（例如：拾取、放入、搜索）。


- Target Object: Semantic structure describing objects (e.g., [cloth, container]).
- 目标物体：描述物体的语义结构（例如：[布料, 容器]）。


- Temporal Bounds: Precise start/end timestamps delimiting the execution segment in the video.
- 时间边界：界定视频中执行片段的精确起始/结束时间戳。


- Status: Labeled as success, failure, irrelevant to denote execution outcome.
- 状态：标注为成功、失败或无关，以表示执行结果。


(2) Chain-of-Thought (CoT) Reasoning: With the subtask annotation, we select tasks of moderate length (containing 4-10 subtasks). We annotate them with Chain-of-Thought data, providing reasoning that decomposes the task into several logically connected subtasks. The annotation pipeline for CoT is illustrated in Figure 12 Initially, we provide the model (Qwen2.5-VL-72B 3) with both the current subtask information (video and subtask instruction) and the ground-truth instruction of the next subtask. The model is then prompted to perform posterior attribution, explaining why the specified next subtask logically follows. Subsequently, the attribution is appended to the input and presented again to Qwen2.5-VL-72B, synthesizing it into a structured causal reasoning process. Next, the CoT segment is extracted and used to prompt Qwen3 55 , which is instructed to predict an answer. This predicted answer is matched against the ground-truth answer to automatically validate the completeness and soundness of the CoT labeling. Finally, human review and correction of erroneous annotations are performed.
(2) 思维链（CoT）推理：借助子任务标注，我们选择中等长度（包含 4 - 10 个子任务）的任务。我们用思维链数据对这些任务进行标注，提供将任务分解为多个逻辑相连子任务的推理过程。图 12 展示了思维链的标注流程。首先，我们向模型（Qwen2.5 - VL - 72B 3）同时提供当前子任务信息（视频和子任务指令）以及下一个子任务的真实指令。然后，提示模型进行后验归因，解释指定的下一个子任务为何在逻辑上紧随其后。随后，将归因内容添加到输入中，并再次呈现给 Qwen2.5 - VL - 72B，将其合成一个结构化的因果推理过程。接下来，提取思维链片段并用于提示 Qwen3 55，指示其预测答案。将这个预测答案与真实答案进行匹配，以自动验证思维链标注的完整性和合理性。最后，对错误标注进行人工审核和修正。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_671452.jpg"/>



Fig. 12: The annotation pipeline for subtask prediction chain-of-thought (CoT).
图 12：子任务预测思维链（CoT）的标注流程。


### B.3 More visualization.
### B.3 更多可视化。


We present more visualizations about the annotations of our dataset in Figure 13. Annotations of the Human-centric Vision-Language tasks are shown in Figure 14.
我们在图 13 中展示了更多关于数据集标注的可视化结果。以人为中心的视觉语言任务标注见图 14。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_4a1f18.jpg"/>



Fig. 13: Data annotation samples cross different scenes. The example of human-centric data annotations, including depth, mask, action, and task descriptions, in six different scenarios.
图 13：不同场景下的数据标注样本。以人为中心的数据标注示例，包含六种不同场景下的深度、掩码、动作和任务描述。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_a7dae5.jpg"/>



Fig. 14: Examples of annotations in the Human-centric Vision-Language Benchmark.
图 14：以人为中心的视觉语言基准测试中的标注示例。


## C Benchmark And Application Details
## C 基准测试与应用详情


### C.1 Human-centric Vision-Language
### C.1 以人为中心的视觉语言


Benchmark Task Descriptions. The Human-centric Vision-Language Benchmark in Section 4.1 is constructed by sampling raw data from WIYH, followed by manual annotation, including Spatial Referring, Subtask Prediction, and Completion Verification, with the amounts and distributions illustrated in Figure 15 In these tasks, all VLMs are directly tested without any training.
基准任务描述。第 4.1 节中的“以人为中心的视觉语言基准测试”是通过对 WIYH 原始数据进行采样并人工标注构建的，包括空间指代、子任务预测和完成度验证，其数量与分布如图 15 所示。在这些任务中，所有 VLM 均在未经任何训练的情况下直接进行测试。


(1) The Spatial Referring is designed as a spatial region grounding form: given relevant image frames, annotators delineate ROI areas using polygons according to the given questions. Specifically, given the question as prompts, the VLM outputs a set of points ${P}_{a}$ representing the spatial region referenced in the question. We denote the set of points that fall within the ground-truth region as ${P}_{r}$ . We evaluate the VLM’s spatial understanding by computing the proportion of the points that fall within the ground-truth region.
(1) 空间指代任务设计为空间区域定位形式：给定相关图像帧，标注员根据提出的问题使用多边形勾勒出感兴趣区域（ROI）。具体而言，以问题作为提示，VLM 输出一组点 ${P}_{a}$，代表问题所指代的空间区域。我们将落入真值区域内的点集记为 ${P}_{r}$。我们通过计算落入真值区域内的点所占比例来评估 VLM 的空间理解能力。


$$
{\text{ score }}_{SR} = \frac{{N}_{a}}{{N}_{r}}, \tag{1}
$$



where ${N}_{a}$ is the number of ${P}_{a}$ and ${N}_{r}$ is the number of ${P}_{r}$ . score ${}_{SR}$ A is a scalar in the range $\left\lbrack  {0,1}\right\rbrack$ ,with values closer to 1 indicating stronger spatial understanding capability of the VLM. This task is designed to evaluate the capability of VLM models to understand spatial referring in human-centric scenarios.
其中 ${N}_{a}$ 是 ${P}_{a}$ 的数量，${N}_{r}$ 是 ${P}_{r}$ 的数量。得分 ${}_{SR}$ A 是一个范围在 $\left\lbrack  {0,1}\right\rbrack$ 内的标量，数值越接近 1，表明 VLM 的空间理解能力越强。该任务旨在评估 VLM 模型在以人为中心的场景中理解空间指代的能力。


(2) The Subtask Prediction adopts a multiple-choice question format. We use the Qwen2.5-VL-72B tool to first generate nine negative candidate options based on video clips and ground truth values, covering both action and target object dimensions. Then, the annotators manually select three reasonable negative samples that fit the current scenario. These samples, along with the correct answers, constitute the final candidate set. That is, each question has four options, and the accuracy of random selection was approximately 25%. This task is designed to evaluate the capability of VLM models to perform long-horizon task planning and decomposition in human-centric scenarios.
(2) 子任务预测采用多项选择题形式。我们使用 Qwen2.5-VL-72B 工具，根据视频片段和真值生成九个负样本候选选项，涵盖动作和目标对象维度。随后，标注员手动选择三个符合当前场景的合理负样本。这些样本与正确答案共同构成最终候选集。即每个问题有四个选项，随机选择的准确率约为 25%。该任务旨在评估 VLM 模型在以人为中心的场景中执行长程任务规划与分解的能力。


(3) The Completion Verification is designed as binary questions: given a video segment randomly truncated from the end, annotators determine whether the subtask has been completed. The evaluation metric is the selection accuracy rate. The accuracy of random selection was approximately ${50}\%$ . This task is designed to evaluate the capability of VLM models to understand the dynamic process of manipulation tasks in human-centric scenarios.
(3) 完成度验证设计为二元问题：给定一个从末尾随机截断的视频片段，标注员判断该子任务是否已完成。评估指标为选择准确率。随机选择的准确率约为 ${50}\%$。该任务旨在评估 VLM 模型在以人为中心的场景中理解操作任务动态过程的能力。


Tasks Visualization. We show the samples and detailed annotations of the three tasks in Figure 16.
任务可视化。我们在图 16 中展示了这三项任务的样本及详细标注。


Case Analysis. Figure 16 presents the results of the Qwen3-VL model on the Spatial Referring task. We observe that existing VLMs perform reasonably well on in-plane spatial referring relations (e.g., "between A and B" as illustrated in the first row). However, when it comes to relations that require understanding positions across different height levels, such as "beneath" or "between A and B" where A and B are situated at different heights, the models fail to grasp the correct spatial information.
案例分析。图 16 展示了 Qwen3-VL 模型在空间指代任务上的结果。我们观察到，现有的 VLM 在平面内空间指代关系（例如第一行所示的“在 A 和 B 之间”）上表现良好。然而，当涉及需要理解不同高度层级关系时，例如“下方”或 A 与 B 处于不同高度时的“在 A 和 B 之间”，模型无法准确把握空间信息。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_9ef3ef.jpg"/>



Fig. 15: Amount of three task annotations in the Human-centric Vision-Language Benchmark.
图 15：以人为中心的视觉语言基准测试中三项任务的标注数量。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_18bec9.jpg"/>



Fig. 16: Case analysis of the Spatial Referring task.
图 16：空间指代任务的案例分析。


### C.2 Human-centric World Model
### C.2 以人为中心的世界模型


Gaussian Splatting 4D Reconstruction Evaluation Metrics. In terms of metrics of 4D reconstruction, we evaluate the RGB rendering quality using PSNR, SSIM, and LPIPS. For PSNR and SSIM, higher is better, while for LPIPS, lower is better. For geometry accuracy from rendered depth maps, we use A.R. (lower is better) and ${\delta }_{1}$ (higher is better). - PSNR (Peak Signal-to-Noise Ratio),
高斯溅射（Gaussian Splatting）4D 重建评估指标。在 4D 重建指标方面，我们使用 PSNR、SSIM 和 LPIPS 评估 RGB 渲染质量。对于 PSNR 和 SSIM，数值越高越好；对于 LPIPS，数值越低越好。对于渲染深度图的几何精度，我们使用 A.R.（越低越好）和 ${\delta }_{1}$（越高越好）。- PSNR（峰值信噪比），


$$
\text{ PSNR } = {10} \cdot  {\log }_{10}\left( \frac{{\mathrm{{MAX}}}_{I}^{2}}{\mathrm{{MSE}}}\right)
$$



$$
\mathrm{{MSE}} = \frac{1}{MN}\mathop{\sum }\limits_{{i = 1}}^{M}\mathop{\sum }\limits_{{j = 1}}^{N}{\left( I\left( i,j\right)  - K\left( i,j\right) \right) }^{2},
$$



where $I$ is the reference image, $K$ is the rendered image,and ${\mathrm{{MAX}}}_{I}$ is the maximum possible pixel value. A higher PSNR indicates that the rendered image is closer to the ground truth.
其中 $I$ 是参考图像，$K$ 是渲染图像，${\mathrm{{MAX}}}_{I}$ 是最大可能像素值。PSNR 越高，表明渲染图像越接近真值。


- SSIM (Structural Similarity Index Measure).
- SSIM（结构相似性指数）。


$$
\operatorname{SSIM}\left( {x,y}\right)  = \frac{\left( {2{\mu }_{x}{\mu }_{y} + {c}_{1}}\right) \left( {2{\sigma }_{xy} + {c}_{2}}\right) }{\left( {{\mu }_{x}^{2} + {\mu }_{y}^{2} + {c}_{1}}\right) \left( {{\sigma }_{x}^{2} + {\sigma }_{y}^{2} + {c}_{2}}\right) },
$$



where $\mu ,\sigma$ ,and ${\sigma }_{xy}$ denote the mean,variance,and covariance of image patches $x$ and $y$ ,respectively. SSIM evaluates perceptual similarity in terms of luminance, contrast, and structure; values closer to 1 indicate better similarity.
其中 $\mu ,\sigma$ 和 ${\sigma }_{xy}$ 分别表示图像块 $x$ 和 $y$ 的均值、方差和协方差。SSIM 从亮度、对比度和结构三个方面评估感知相似度；数值越接近 1，表明相似度越高。


- LPIPS (Learned Perceptual Image Patch Similarity). This metric computes the perceptual distance between rendered and ground-truth images by extracting deep features from a pretrained CNN and measuring the distance in that feature space. Lower LPIPS values indicate that two images are more perceptually similar.
- LPIPS（学习感知图像块相似度）。该指标通过提取预训练 CNN 的深度特征并测量特征空间中的距离，来计算渲染图像与真实图像之间的感知距离。LPIPS 值越低，表示两张图像在感知上越相似。


- AbsRel (Absolute Relative Error).
- AbsRel（绝对相对误差）。


$$
\text{ AbsRel } = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}\frac{\left| {d}_{i} - {d}_{i}^{ * }\right| }{{d}_{i}^{ * }},
$$



where ${d}_{i}$ is the predicted depth at the pixel $i$ and ${d}_{i}^{ * }$ is the corresponding ground-truth depth. A lower value denotes a more accurate geometry.
其中 ${d}_{i}$ 是像素 $i$ 处的预测深度，${d}_{i}^{ * }$ 是对应的真实深度。数值越低表示几何结构越准确。


- ${\delta }_{1}$ (Threshold Accuracy / Delta1).
- ${\delta }_{1}$（阈值准确率 / Delta1）。


$$
{\delta }_{1} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}\mathbf{1}\left( {\max \left( {\frac{{d}_{i}}{{d}_{i}^{ * }},\frac{{d}_{i}^{ * }}{{d}_{i}}}\right)  < {1.25}}\right) ,
$$



which measures the percentage of predicted depth values that are within a factor of 1.25 of the ground-truth depth. A higher value ${\delta }_{1}$ indicates better accuracy.
该指标衡量预测深度值在真实深度值 1.25 倍因子范围内的百分比。${\delta }_{1}$ 值越高表示准确度越好。


During optimization, all input videos are processed at a resolution of 960×720. We represent each scene with ${40}\mathrm{k}$ dynamic Gaussians and ${100}\mathrm{k}$ static Gaussians and use $\mathrm{B} = {10}$ shared $\mathrm{{SE}}\left( 3\right)$ motion bases for all points. We optimize the model with Adam, running 1k iterations for the initialization stage and 500 training epochs,with learning rates of ${1.6} \times  {10}^{-4}$ for Gaussian means and $\operatorname{SE}\left( 3\right)$ bases, $5 \times  {10}^{-3}$ for scales, $1 \times  {10}^{-3}$ for rotations,and $1 \times  {10}^{-2}$ for colors,opacities,and motion coefficients. Training on a 300-frame sequence takes about 2 hours on a single A100 GPU and supports real-time rendering at around 140 FPS.
在优化过程中，所有输入视频均以 960×720 的分辨率进行处理。我们用 ${40}\mathrm{k}$ 个动态高斯和 ${100}\mathrm{k}$ 个静态高斯表示每个场景，并为所有点使用 $\mathrm{B} = {10}$ 个共享的 $\mathrm{{SE}}\left( 3\right)$ 运动基。我们使用 Adam 优化器对模型进行优化，初始化阶段运行 1k 次迭代，训练阶段运行 500 个 epoch；高斯均值和 $\operatorname{SE}\left( 3\right)$ 基的学习率为 ${1.6} \times  {10}^{-4}$，尺度为 $5 \times  {10}^{-3}$，旋转为 $1 \times  {10}^{-3}$，颜色、不透明度和运动系数为 $1 \times  {10}^{-2}$。在单张 A100 GPU 上训练 300 帧的序列大约需要 2 小时，并支持约 140 FPS 的实时渲染。


Addition Application: Language-conditioned Video Generation This section evaluates the contribution of our proposed WIYH dataset to enhancing video generation performance in human-centric manipulation tasks. We fine-tuned two baseline models, transformer-based CogVideo [19] and UNet-based DynamiCrafter [52], initialized with their original weights, using WIYH under the image-and-text-to-video setting. During fine-tuning, all input videos were resized to ${480}\mathrm{p}$ resolution. We randomly select ${90}\%$ of the data as the training set and the remaining ${10}\%$ as the test set.
附加应用：语言条件视频生成。本节评估了我们提出的 WIYH 数据集在提升以人为中心的操纵任务中视频生成性能方面的贡献。我们在图像和文本转视频的设置下，使用 WIYH 对两个基准模型（基于 Transformer 的 CogVideo [19] 和基于 UNet 的 DynamiCrafter [52]）进行了微调，并使用其原始权重进行初始化。微调期间，所有输入视频均调整为 ${480}\mathrm{p}$ 分辨率。我们随机选择 ${90}\%$ 的数据作为训练集，其余 ${10}\%$ 作为测试集。


As shown in Figure 17, models fine-tuned with WIYH generate egocentric videos with improved spatial and temporal coherence, as well as stronger alignment with both the input image and action instructions. For example, in the "Wipe the cabinet" task, the cabinet's orientation and the towel's color are reproduced more accurately. In contrast, videos generated by non-fine-tuned models exhibit noticeable object fragmentation and distortion, along with discontinuous and physically implausible hand-object interactions. Notably, after fine-tuning, the complex motions of non-rigid objects in the generated videos appear more realistic and natural. These improvements demonstrate that WIYH substantially enhances a model's ability to capture real-world dynamics.
如图 17 所示，经 WIYH 微调的模型生成的自中心视角视频具有更好的空间和时间连贯性，且与输入图像及动作指令的对齐效果更强。例如，在“擦拭橱柜”任务中，橱柜的方向和毛巾的颜色还原得更准确。相比之下，未经微调的模型生成的视频表现出明显的物体破碎和畸变，以及不连续且物理上不合理的手物交互。值得注意的是，微调后，生成视频中非刚性物体的复杂运动显得更加真实自然。这些改进证明了 WIYH 显著增强了模型捕捉现实世界动态的能力。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_deaf0b.jpg"/>



Fig. 17: Language-Conditioned Video Generation. When provided with language instructions, two baseline video prediction methods exhibited significant hallucinations without WIYH fine-tuning. However, after fine-tuning on our dataset, they demonstrated a markedly enhanced ability to imagine future states.
图 17：语言条件视频生成。在给定语言指令时，两个基准视频预测方法在未经 WIYH 微调的情况下表现出明显的幻觉。然而，在使用我们的数据集微调后，它们展现出显著增强的未来状态想象能力。


Quantitative results are summarized in Table 7. We use four metrics from VBench 22-consistency, smoothness, dynamic (motion intensity), and quality in order to assess motion plausibility, temporal coherence, and visual fidelity of the video generation models. Following VBench 22, the evaluation metrics are as follows:
定量结果汇总于表 7。我们使用 VBench 22 中的四个指标——一致性、平滑度、动态性（运动强度）和质量，以评估视频生成模型的运动合理性、时间连贯性和视觉保真度。遵循 VBench 22，评估指标如下：


- Consistency. To evaluate the temporal consistency of the target manipulation subjects, we calculate the DINO [10] feature similarity across frames. For assessing the temporal consistency of the background scenes, we compute the CLIP 38 feature similarity across frames. The overall consistency score is defined as the arithmetic mean of the subject consistency and the background consistency.
- 一致性。为了评估目标操纵对象的时间一致性，我们计算帧间的 DINO [10] 特征相似度。为了评估背景场景的时间一致性，我们计算帧间的 CLIP 38 特征相似度。总体一致性得分定义为对象一致性和背景一致性的算术平均值。


- Smoothness. To assess the temporal continuity of the generated videos, we employ a motion-consistency metric based on the AMT 32 video frame interpolation model. Given a generated video with frames $\left\{  {{f}_{0},{f}_{1},\ldots ,{f}_{{2n} - 1},{f}_{2n}}\right\}$ , we first remove all odd-indexed frames to obtain $\left\{  {{f}_{0},{f}_{2},\ldots ,{f}_{2n}}\right\}$ . The AMT model is then used to interpolate the missing frames,producing $\left\{  {{\widehat{f}}_{1},{\widehat{f}}_{3},\ldots ,{\widehat{f}}_{{2n} - 1}}\right\}$ . Finally, we compute the mean absolute error between the interpolated and original frames, where a larger value indicates smoother and more temporally coherent motion.
- 平滑度。为评估生成视频的时间连续性，我们采用基于 AMT $\left\{  {{f}_{0},{f}_{1},\ldots ,{f}_{{2n} - 1},{f}_{2n}}\right\}$ 视频帧插值模型的运动一致性指标。给定包含帧 $\left\{  {{f}_{0},{f}_{1},\ldots ,{f}_{{2n} - 1},{f}_{2n}}\right\}$ 的生成视频，我们首先移除所有奇数索引帧以获得 $\left\{  {{f}_{0},{f}_{2},\ldots ,{f}_{2n}}\right\}$ 。随后使用 AMT 模型对缺失帧进行插值，生成 $\left\{  {{\widehat{f}}_{1},{\widehat{f}}_{3},\ldots ,{\widehat{f}}_{{2n} - 1}}\right\}$ 。最后，计算插值帧与原始帧之间的平均绝对误差，数值越大表示运动越平滑且时间连贯性越好。


- Dynamic. To prevent the model from producing static videos in pursuit of higher temporal consistency scores, we employ RAFT [41] to estimate optical flow strengths between consecutive frames of each generated video. The average of the largest 5% optical flows is used to determine whether the video is static. The final dynamic score is calculated by measuring the proportion of non-static videos generated by the model.
- 动态性。为防止模型为追求更高的时间一致性得分而生成静态视频，我们采用 RAFT [41] 来估计生成视频中连续帧之间的光流强度。通过取最大 5% 光流的平均值来判断视频是否为静态。最终的动态性得分通过计算模型生成的非静态视频比例得出。


- Quality. To assess the visual quality of generated video frames, we employ the MUSIQ [25] image quality predictor trained on the SPAQ [15] dataset. This metric quantifies common distortions such as overexposure, noise, and blur, with a higher score indicating fewer artifacts.
- 质量。为评估生成视频帧的视觉质量，我们采用在 SPAQ [15] 数据集上训练的 MUSIQ [25] 图像质量预测器。该指标量化了过曝、噪声和模糊等常见失真，得分越高表示伪影越少。


After fine-tuning with the WIYH dataset, both baseline models show clear improvements across all metrics, with the most significant gains observed in dynamics and temporal consistency.
在经过 WIYH 数据集微调后，两个基准模型在所有指标上均表现出明显提升，其中在动态性和时间一致性方面的增益最为显著。


Table 7: Comparison of consistency, smoothness, dynamics, and quality metrics under different conditions.
表 7：不同条件下一致性、平滑度、动态性和质量指标的对比。


<table><tr><td rowspan="2">Metric</td><td colspan="2">CogVideo</td><td colspan="2">DynamiCrafter</td></tr><tr><td>w/o WIYH</td><td>w/ WIYH</td><td>w/o WIYH</td><td>w/ WIYH</td></tr><tr><td>Consistency↑</td><td>80.6</td><td>88.2 (+7.6)</td><td>85.3</td><td>91.5 (+6.2)</td></tr><tr><td>Smoothness↑</td><td>93.5</td><td>98.4 (+4.9)</td><td>97.4</td><td>98.6 (+1.2)</td></tr><tr><td>Dynamic $\uparrow$</td><td>62.9</td><td>78.5 (+15.6)</td><td>74.8</td><td>89.6 (+14.8)</td></tr><tr><td>Quality $\uparrow$</td><td>60.1</td><td>67.6 (+7.5)</td><td>63.5</td><td>70.6 (+7.1)</td></tr></table>
<table><tbody><tr><td rowspan="2">指标</td><td colspan="2">CogVideo</td><td colspan="2">DynamiCrafter</td></tr><tr><td>无 WIYH</td><td>有 WIYH</td><td>无 WIYH</td><td>有 WIYH</td></tr><tr><td>一致性↑</td><td>80.6</td><td>88.2 (+7.6)</td><td>85.3</td><td>91.5 (+6.2)</td></tr><tr><td>平滑度↑</td><td>93.5</td><td>98.4 (+4.9)</td><td>97.4</td><td>98.6 (+1.2)</td></tr><tr><td>动态 $\uparrow$</td><td>62.9</td><td>78.5 (+15.6)</td><td>74.8</td><td>89.6 (+14.8)</td></tr><tr><td>质量 $\uparrow$</td><td>60.1</td><td>67.6 (+7.5)</td><td>63.5</td><td>70.6 (+7.1)</td></tr></tbody></table>


## D Details of Human-centric Manipulation
## D 以人为中心的操纵细节


In this section, we present the details of our co-training experiments in Section 5, Human-centric Manipulation, of the main text. As shown in Figure 18, we demonstrate 11 task variants under two scenarios: Single-Object Scene and Multi-Object Cluttered Scene. Figures 19 and 20 illustrate the success rates and failure causes for each task in these two scenarios, respectively.
本节详细介绍了正文中第5节“以人为中心的操纵”部分的协同训练实验。如图18所示，我们展示了单物体场景和多物体杂乱场景下共11种任务变体。图19和图20分别展示了这两种场景下各项任务的成功率及失败原因。


As shown in Figures 19 and 20 on most tasks, the incorporation of human data leads to a general increase in success rate. In the Single-Object Scene, the diverse action distribution of human data enhanced the policy's generalization capability to unseen initial states, enabling successful grasping at different positions or over long distances. In the Multi-Object Cluttered Scene, policies trained solely on robot data demonstrated almost no generalization ability in complex scenarios. When factors such as object height or layout changed, policies trained only on robot data failed, whereas the inclusion of human data gradually enabled success by effectively introducing both the action and observation domains of human data into the robot dataset. Furthermore, the consistently low success rates across all methods in certain tasks highlight important directions for future research: how to further improve the collection precision of human data and how to design efficient algorithms that can better transfer knowledge from human data to dexterous manipulation tasks.
如图19和图20所示，在大多数任务中，引入人类数据普遍提升了成功率。在单物体场景中，人类数据多样的动作分布增强了策略对未见初始状态的泛化能力，使其能够成功抓取不同位置或远距离的目标。在多物体杂乱场景中，仅使用机器人数据训练的策略在复杂环境下几乎没有泛化能力。当物体高度或布局发生变化时，仅使用机器人数据的策略会失败，而引入人类数据则通过将人类数据的动作和观测域有效融入机器人数据集，逐步实现了成功。此外，某些任务中所有方法持续偏低的成功率，也指明了未来的重要研究方向：如何进一步提高人类数据的采集精度，以及如何设计高效算法，以更好地将人类数据的知识迁移到灵巧操纵任务中。


<img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c7f40f.jpg"/>



Fig.18: Task variant visualization in the co-training experiment.
图18：协同训练实验中的任务变体可视化。


## E Ethics and Privacy Statement
## E 伦理与隐私声明


Our work involves large-scale data collection of real-world human manipulation activities. We took a number of measures to ensure responsible data handling, privacy protection, and ethical compliance.
我们的工作涉及大规模真实世界人类操纵活动的数据采集。我们采取了多项措施以确保负责任的数据处理、隐私保护及伦理合规。


Data Collection and Consent. All human subjects participating in data collection were recruited voluntarily and provided informed consent in accordance with institutional guidelines. Participants were briefed on the purpose of the study, the modalities collected (RGB video, depth, motion capture, tactile sensing), the intended research use, and their right to withdraw at any time.
数据采集与知情同意。所有参与数据采集的人员均为自愿参与，并根据机构准则提供了知情同意。参与者已获悉研究目的、采集模态（RGB视频、深度、动作捕捉、触觉传感）、预期的研究用途以及随时退出的权利。


Privacy Protection. Although the dataset contains real-world human activities and may capture visible human faces or identifiable biometric patterns, several privacy-preserving strategies were implemented:
隐私保护。尽管数据集包含真实世界的人类活动，且可能捕捉到可见的人脸或可识别的生物特征模式，我们实施了多项隐私保护策略：


- Controlled subjects only: No bystanders or non-consenting individuals appear in the dataset; all scenes were recorded in controlled environments with only consenting operators present.
- 仅限受控主体：数据集中不包含旁观者或未同意的个人；所有场景均在受控环境中录制，仅有同意参与的操作员在场。


- No identity labels: We do not include personal information, demographic information, or identity annotations.
- 无身份标签：我们不包含任何个人信息、人口统计学信息或身份标注。


- Face and identity protection: When faces are visible in RGB streams, we apply optional face blurring in the public release version.
- 面部与身份保护：当RGB流中出现人脸时，我们在公开发布的版本中应用了可选的面部模糊处理。


- No audio recordings: We do not collect speech or audio signals to avoid unintended disclosure of personal information.
- 无音频记录：我们不采集语音或音频信号，以避免意外泄露个人信息。


Sensitive Content and Safety. The dataset does not include minors, hazardous behaviors, or sensitive activities. All recordings were conducted in standard workplace or daily-life environments using safety protocols.
敏感内容与安全。数据集不包含未成年人、危险行为或敏感活动。所有录制均在标准工作场所或日常生活中使用安全协议进行。


Data Usage and Licensing. The released dataset is intended exclusively for research on embodied AI, manipulation, perception, and VLA learning. Redistribution or commercial use is restricted according to the license accompanying the dataset. Users are required to comply with ethical research practices and avoid any attempts at identity recognition or misuse of human-related data.
数据使用与许可。发布的数据集仅用于具身智能、操纵、感知和VLA学习研究。根据数据集附带的许可协议，限制重新分发或商业用途。用户必须遵守伦理研究规范，避免任何身份识别尝试或对人类相关数据的滥用。


Potential Risks and Mitigations. As with any dataset containing human recordings, there is a theoretical risk of model misuse, such as identity extraction or surveillance applications. To mitigate these risks, we provide: (1) no textual identity metadata, (2) optional visual anonymization tools, and (3) a license that explicitly forbids re-identification, face recognition, or surveillance-related uses.
潜在风险与缓解措施。与任何包含人类记录的数据集一样，存在模型被滥用的理论风险，例如身份提取或监控应用。为缓解这些风险，我们提供：(1) 无文本身份元数据，(2) 可选的视觉匿名化工具，以及 (3) 明确禁止重新识别、人脸识别或监控相关用途的许可协议。


Institutional Review. All data collection procedures were reviewed and approved by an internal ethics committee to ensure compliance with privacy protection and responsible AI principles.
机构审查。所有数据采集程序均经过内部伦理委员会审查并批准，以确保符合隐私保护和负责任的AI原则。


<table><tr><td>Task Variation</td><td>Training Recipe</td><td>Failure Analysis</td><td>Success vs. Failure</td></tr><tr><td>(1) Grasping from HOME_POINT with the target in front of hand.</td><td>UMI#200</td><td>Failed to reach a reasonable position, early grasping</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_97c55e.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Grasped air</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_40eaa1.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Failed to reach a reasonable position, early grasping</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c4e02d.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>-</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_1f619f.jpg"/></td></tr><tr><td>(2) Grasping from HOME_POINT with the target in right, front of hand.</td><td>UMI#200</td><td>Far distance, can not grasping</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_a885d1.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Failed to reach a reasonable position, early grasping</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_0974f3.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Failed to reach a reasonable position, early grasping</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_cdb3bd.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>Failed to reach a reasonable position, early grasping</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_37e0a2.jpg"/></td></tr><tr><td>(3) Grasping from HOME_POINT with the target in right, front of hand, away from hand.</td><td>UMI#200</td><td>Large positional deviation</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c597b3.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Significant deviation from the target</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_d7e9f4.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Failed to reach a reasonable position, collided with target</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_9632e6.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>Reached the position, unreasonable grasp timing</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_785cb4.jpg"/></td></tr><tr><td>(4) Grasping from HOME POINT with the target in middle of platform, parallel to the hand.</td><td>UMI#200</td><td>Unable to reach the grasping position</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_56119e.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Unable to reach the grasping position</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c53c4a.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Collided with the target</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_8eb1ed.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>Failed to reach a reasonable position, early grasping, collided</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_f6ac65.jpg"/></td></tr><tr><td>(5) Grasping from HOME_POINT with the target in left of hand, close to the hand.</td><td>UMI#200</td><td>Collided with the target</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_aefaaf.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Severe finger compression</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_e9ea9b.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Collided with the target</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_dad9da.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>Failed to reach a reasonable position, early grasping, collided</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_f21dff.jpg"/></td></tr><tr><td>(6) Hand facing to the target directly.</td><td>UMI#200</td><td>Joint and motion anomalies</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c4cc7c.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Failed to reach a reasonable position, early grasping, collided</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_384458.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Collided with the target</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_6066b8.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>Collided with the target</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_28ebb2.jpg"/></td></tr></table>
<table><tbody><tr><td>任务变体</td><td>训练配方</td><td>失败分析</td><td>成功与失败对比</td></tr><tr><td>(1) 从 HOME_POINT 开始抓取，目标位于手部前方。</td><td>UMI#200</td><td>未能到达合理位置，过早抓取</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_97c55e.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>抓空</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_40eaa1.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>未能到达合理位置，过早抓取</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c4e02d.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>-</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_1f619f.jpg"/></td></tr><tr><td>(2) 从 HOME_POINT 开始抓取，目标位于手部右前方。</td><td>UMI#200</td><td>距离过远，无法抓取</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_a885d1.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>未能到达合理位置，过早抓取</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_0974f3.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>未能到达合理位置，过早抓取</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_cdb3bd.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>未能到达合理位置，过早抓取</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_37e0a2.jpg"/></td></tr><tr><td>(3) 从 HOME_POINT 开始抓取，目标位于手部右前方，且远离手部。</td><td>UMI#200</td><td>位置偏差大</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c597b3.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>与目标存在显著偏差</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_d7e9f4.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>未能到达合理位置，与目标发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_9632e6.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>已到达位置，抓取时机不合理</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_785cb4.jpg"/></td></tr><tr><td>(4) 从 HOME_POINT 开始抓取，目标位于平台中部，与手部平行。</td><td>UMI#200</td><td>无法到达抓取位置</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_56119e.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>无法到达抓取位置</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c53c4a.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>与目标发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_8eb1ed.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>未能到达合理位置，过早抓取，发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_f6ac65.jpg"/></td></tr><tr><td>(5) 从 HOME_POINT 开始抓取，目标位于手部左侧，靠近手部。</td><td>UMI#200</td><td>与目标发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_aefaaf.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>手指严重挤压</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_e9ea9b.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>与目标发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_dad9da.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>未能到达合理位置，过早抓取，发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_f21dff.jpg"/></td></tr><tr><td>(6) 手部直接朝向目标。</td><td>UMI#200</td><td>关节与运动异常</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c4cc7c.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>未能到达合理位置，过早抓取，发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_384458.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>与目标发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_6066b8.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>与目标发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_28ebb2.jpg"/></td></tr></tbody></table>


<table><tr><td>Task Variation</td><td>Training Recipe</td><td>Failure Analysis</td><td>Success vs. Failure</td></tr><tr><td>(7) Crowded objects around the target.</td><td>UMI#200</td><td>Abnormal grasping behavior</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_11e28b.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Failed to reach a reasonable position, early grasping, collided</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_b0ecb5.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Failed to reach a reasonable position, early grasping, collided</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_00e129.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>Failed to reach a reasonable position, early grasping, collided</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_90b726.jpg"/></td></tr><tr><td>(8) Obstacle in front of the target.</td><td>UMI#200</td><td>Abnormal grasping behavior</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_922600.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Unable to reach the target position</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_a425f6.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Grasping position too low, collided with other assets</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_46dca0.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>Severe finger compression</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_639e0a.jpg"/></td></tr><tr><td>(9) Elevated the target.</td><td>UMI#200</td><td>Abnormal grasping behavior</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_0a9a76.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Unable to reach the grasping position</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_669234.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Reached the grasping position, grasping failed</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c349c9.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>Severe finger compression</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_fd5e45.jpg"/></td></tr><tr><td>(10) Elevated the target, hand facing to the target directly.</td><td>UMI#200</td><td>Unable to reach the grasping position</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_db048e.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Unable to reach the grasping position</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c9379d.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Failed to reach a reasonable position, early grasping</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_f14d30.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>-</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_6a136e.jpg"/></td></tr><tr><td>(11) leaned target.</td><td>UMI#200</td><td>Unable to reach the grasping position</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_0ee8a1.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>Incorrect grasping location</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_2a8628.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>Incorrect grasping location</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_25a3b7.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>Incorrect grasping location</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_f1a04f.jpg"/></td></tr></table>
<table><tbody><tr><td>任务多样性</td><td>训练配方</td><td>失败分析</td><td>成功与失败对比</td></tr><tr><td>(7) 目标周围物体拥挤。</td><td>UMI#200</td><td>异常抓取行为</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_11e28b.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>未能到达合理位置，过早抓取，发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_b0ecb5.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>未能到达合理位置，过早抓取，发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_00e129.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>未能到达合理位置，过早抓取，发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_90b726.jpg"/></td></tr><tr><td>(8) 目标前方有障碍物。</td><td>UMI#200</td><td>异常抓取行为</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_922600.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>无法到达目标位置</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_a425f6.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>抓取位置过低，与其他资产发生碰撞</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_46dca0.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>手指严重挤压</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_639e0a.jpg"/></td></tr><tr><td>(9) 抬高目标。</td><td>UMI#200</td><td>异常抓取行为</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_0a9a76.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>无法到达抓取位置</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_669234.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>到达抓取位置，抓取失败</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c349c9.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>手指严重挤压</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_fd5e45.jpg"/></td></tr><tr><td>(10) 抬高目标，手直接朝向目标。</td><td>UMI#200</td><td>无法到达抓取位置</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_db048e.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>无法到达抓取位置</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_c9379d.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>未能到达合理位置，过早抓取</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_f14d30.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>-</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_6a136e.jpg"/></td></tr><tr><td>(11) 倾斜目标。</td><td>UMI#200</td><td>无法到达抓取位置</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_0ee8a1.jpg"/></td></tr><tr><td></td><td>UMI#500</td><td>抓取位置不正确</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_2a8628.jpg"/></td></tr><tr><td></td><td>UMI#200+WiYH#800</td><td>抓取位置不正确</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_25a3b7.jpg"/></td></tr><tr><td></td><td>UMI#500+WiYH#800</td><td>抓取位置不正确</td><td><img src="https://raw.githubusercontent.com/Tsuki-Gor/Pic_Bed_Ob/main/Mixed/M2026/05/2026_05_11__23_22_48_f1a04f.jpg"/></td></tr></tbody></table>


Fig. 20: Dexterous manipulation Details - 2.
图 20：灵巧操作细节 - 2。


## References
## 参考文献


1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
1. Achiam, J., 等：Gpt-4 技术报告。arXiv 预印本 arXiv:2303.08774 (2023)


2. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35, 23716- 23736 (2022)
2. Alayrac, J.B., 等：Flamingo：用于少样本学习的视觉语言模型。神经信息处理系统进展 35, 23716-23736 (2022)


3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)
3. Bai, S., 等：Qwen2.5-vl 技术报告。arXiv 预印本 arXiv:2502.13923 (2025)


4. Bai, Y., Chen, H., Chen, J., Chen, Z., Deng, Y., Dong, X., Hantrakul, L., Hao, W., Huang, Q., Huang, Z., et al.: Seed-music: A unified framework for high quality and controlled music generation. arXiv preprint arXiv:2409.09214 (2024)
4. Bai, Y., 等：Seed-music：用于高质量和可控音乐生成的统一框架。arXiv 预印本 arXiv:2409.09214 (2024)


5. Banerjee, P., Shkodrani, S., Moulon, P., Hampali, S., Han, S., Zhang, F., Zhang, L., Fountain, J., Miller, E., Basol, S., et al.: Hot3d: Hand and object tracking in 3d from egocentric multi-view videos. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 7061-7071 (2025)
5. Banerjee, P., 等：Hot3d：基于第一人称多视角视频的 3D 手部与物体追踪。见：计算机视觉与模式识别会议论文集。第 7061-7071 页 (2025)


6. Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al.: π0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint ARXIV.2410.24164
6. Black, K., 等：π0：一种用于通用机器人控制的视觉-语言-动作流模型。corr, abs/2410.24164, 2024. doi: 10.48550. arXiv 预印本 ARXIV.2410.24164


7. Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakr-ishnan, K., Hausman, K., Herzog, A., Hsu, J., et al.: Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817 (2022)
7. Brohan, A., 等：Rt-1：用于大规模现实世界控制的机器人 Transformer。arXiv 预印本 arXiv:2212.06817 (2022)


8. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Nee-lakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877-1901 (2020)
8. Brown, T., 等：语言模型是少样本学习者。神经信息处理系统进展 33, 1877-1901 (2020)


9. ByteDance Doubao Team: Doubao seed 1.6-vision: A hybrid-training vision-language model (2024), https://modelscope.cn/models/doubao-seed-1.6- vision accessed: 2025-11-14
9. 字节跳动豆包团队：Doubao seed 1.6-vision：一种混合训练的视觉语言模型 (2024), https://modelscope.cn/models/doubao-seed-1.6-vision 访问日期：2025-11-14


10. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650-9660 (2021)
10. Caron, M., 等：自监督视觉 Transformer 中的涌现属性。见：IEEE/CVF 国际计算机视觉会议论文集。第 9650-9660 页 (2021)


11. Chao, Y.W., Yang, W., Xiang, Y., Molchanov, P., Handa, A., Tremblay, J., Narang, Y.S., Van Wyk, K., Iqbal, U., Birchfield, S., et al.: Dexycb: A benchmark for capturing hand grasping of objects. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9044-9053 (2021)
11. Chao, Y.W., 等：Dexycb：一个用于捕捉手部抓取物体的基准测试。见：IEEE/CVF 计算机视觉与模式识别会议论文集。第 9044-9053 页 (2021)


12. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H.W., Sutton, C., Gehrmann, S., et al.: Palm: Scaling language modeling with pathways. Journal of Machine Learning Research 24(240), 1-113 (2023)
12. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H.W., Sutton, C., Gehrmann, S., 等：Palm：通过 Pathways 扩展语言建模。《机器学习研究杂志》24(240), 1-113 (2023)


13. starVLA Contributors: Starvla: A lego-like codebase for vision-language-action model developing. GitHub repository (1 2025). https://doi.org/10.5281/ zenodo.18264214 https://github.com/starVLA/starVLA
13. starVLA 贡献者：Starvla：用于视觉-语言-动作模型开发的乐高式代码库。GitHub 仓库 (2025年1月)。https://doi.org/10.5281/ zenodo.18264214 https://github.com/starVLA/starVLA


14. Doersch, C., Yang, Y., Vecerik, M., Gokay, D., Gupta, A., Aytar, Y., Carreira, J., Zisserman, A.: Tapir: Tracking any point with per-frame initialization and temporal refinement. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10061-10072 (2023)
14. Doersch, C., Yang, Y., Vecerik, M., Gokay, D., Gupta, A., Aytar, Y., Carreira, J., Zisserman, A.：Tapir：通过逐帧初始化和时间细化跟踪任意点。收录于：IEEE/CVF 国际计算机视觉会议论文集，第 10061-10072 页 (2023)


15. Fang, Y., Zhu, H., Zeng, Y., Ma, K., Wang, Z.: Perceptual quality assessment of smartphone photography. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3677-3686 (2020)
15. Fang, Y., Zhu, H., Zeng, Y., Ma, K., Wang, Z.：智能手机摄影的感知质量评估。收录于：IEEE/CVF 计算机视觉与模式识别会议论文集，第 3677-3686 页 (2020)


16. Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Hamburger, J., Jiang, H., Liu, M., Liu, X., et al.: Ego4d: Around the world in 3,000 hours of egocentric video. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18995-19012 (2022)
16. Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Hamburger, J., Jiang, H., Liu, M., Liu, X., 等：Ego4d：在 3,000 小时的第一人称视角视频中环游世界。收录于：IEEE/CVF 计算机视觉与模式识别会议论文集，第 18995-19012 页 (2022)


17. Grauman, K., Westbury, A., Torresani, L., Kitani, K., Malik, J., Afouras, T., Ashutosh, K., Baiyya, V., Bansal, S., Boote, B., et al.: Ego-ex04d: Understanding skilled human activity from first-and third-person perspectives. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19383-19400 (2024)
17. Grauman, K., Westbury, A., Torresani, L., Kitani, K., Malik, J., Afouras, T., Ashutosh, K., Baiyya, V., Bansal, S., Boote, B., 等：Ego-ex04d：从第一人称和第三人称视角理解熟练的人类活动。收录于：IEEE/CVF 计算机视觉与模式识别会议论文集，第 19383-19400 页 (2024)


18. He, Y., Liang, S., Rui, X., Cai, C., Wan, G.: Egovm: Achieving precise ego-localization using lightweight vectorized maps. In: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 12248-12255. IEEE (2024)
18. He, Y., Liang, S., Rui, X., Cai, C., Wan, G.：Egovm：利用轻量级矢量化地图实现精确的自我定位。收录于：2024 IEEE/RSJ 国际智能机器人与系统会议 (IROS)，第 12248-12255 页。IEEE (2024)


19. Hong, W., Ding, M., Zheng, W., Liu, X., Tang, J.: Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In: The Eleventh International Conference on Learning Representations (2022)
19. Hong, W., Ding, M., Zheng, W., Liu, X., Tang, J.：Cogvideo：通过 Transformer 进行大规模文本生成视频预训练。收录于：第十一届国际学习表征会议 (2022)


20. Hoque, R., Huang, P., Yoon, D.J., Sivapurapu, M., Zhang, J.: Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709 (2025)
20. Hoque, R., Huang, P., Yoon, D.J., Sivapurapu, M., Zhang, J.：Egodex：从大规模第一人称视频中学习灵巧操作。arXiv 预印本 arXiv:2505.11709 (2025)


21. Hsieh, J., Tu, K.H., Hung, K.H., Ke, T.W.: Dexman: Learning bimanual dexterous manipulation from human and generated videos. arXiv preprint arXiv:2510.08475 (2025)
21. Hsieh, J., Tu, K.H., Hung, K.H., Ke, T.W.：Dexman：从人类和生成的视频中学习双手动灵巧操作。arXiv 预印本 arXiv:2510.08475 (2025)


22. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21807-21818 (2024)
22. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., 等：Vbench：视频生成模型的综合基准测试套件。收录于：IEEE/CVF 计算机视觉与模式识别会议论文集，第 21807-21818 页 (2024)


23. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Os-trow, A., Welhinda, A., Hayes, A., Radford, A., et al.: Gpt-40 system card. arXiv preprint arXiv:2410.21276 (2024)
23. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Os-trow, A., Welhinda, A., Hayes, A., Radford, A., 等：Gpt-40 系统卡。arXiv 预印本 arXiv:2410.21276 (2024)


24. Kareer, S., Patel, D., Punamiya, R., Mathur, P., Cheng, S., Wang, C., Hoffman, J., Xu, D.: Egomimic: Scaling imitation learning via egocentric video. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 13226-13233. IEEE (2025)
24. Kareer, S., Patel, D., Punamiya, R., Mathur, P., Cheng, S., Wang, C., Hoffman, J., Xu, D.：Egomimic：通过第一人称视频扩展模仿学习。收录于：2025 IEEE 国际机器人与自动化会议 (ICRA)，第 13226-13233 页。IEEE (2025)


25. Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5148-5157 (2021)
25. Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: 多尺度图像质量 Transformer。载于：IEEE/CVF 国际计算机视觉会议论文集，第 5148-5157 页 (2021)


26. Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al.: Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246 (2024)
26. Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., 等：Openvla: 一种开源视觉-语言-动作模型。arXiv 预印本 arXiv:2406.09246 (2024)


27. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything. arXiv:2304.02643 (2023)
27. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything (分割一切)。arXiv:2304.02643 (2023)


28. Kwon, T., Tekin, B., Stühmer, J., Bogo, F., Pollefeys, M.: H2o: Two hands manipulating objects for first person interaction recognition. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10138-10148 (2021)
28. Kwon, T., Tekin, B., Stühmer, J., Bogo, F., Pollefeys, M.: H2o: 用于第一人称交互识别的双手操作物体数据集。载于：IEEE/CVF 国际计算机视觉会议论文集，第 10138-10148 页 (2021)


29. Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., Li, C.: Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895 (2024)
29. Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., Li, C.: Llava-next-interleave: 在大型多模态模型中处理多图像、视频和 3D 数据。arXiv 预印本 arXiv:2407.07895 (2024)


30. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International conference on machine learning. pp. 19730-19742. PMLR (2023)
30. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: 利用冻结图像编码器和大型语言模型进行语言-图像预训练引导。载于：国际机器学习会议论文集，第 19730-19742 页。PMLR (2023)


31. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International conference on machine learning. pp. 12888-12900. PMLR (2022)
31. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: 用于统一视觉-语言理解与生成的语言-图像预训练引导。载于：国际机器学习会议论文集，第 12888-12900 页。PMLR (2022)


32. Li, Z., Zhu, Z.L., Han, L.H., Hou, Q., Guo, C.L., Cheng, M.M.: Amt: All-pairs multi-field transforms for efficient frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9801- 9810 (2023)
32. Li, Z., Zhu, Z.L., Han, L.H., Hou, Q., Guo, C.L., Cheng, M.M.: Amt: 用于高效帧插值的全对多场变换。载于：IEEE/CVF 计算机视觉与模式识别会议论文集，第 9801-9810 页 (2023)


33. Li, Z., Tucker, R., Cole, F., Wang, Q., Jin, L., Ye, V., Kanazawa, A., Holynski, A., Snavely, N.: Megasam: Accurate, fast and robust structure and motion from casual dynamic videos. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10486-10496 (2025)
33. Li, Z., Tucker, R., Cole, F., Wang, Q., Jin, L., Ye, V., Kanazawa, A., Holynski, A., Snavely, N.: Megasam: 从非结构化动态视频中进行准确、快速且鲁棒的结构与运动恢复。载于：计算机视觉与模式识别会议论文集，第 10486-10496 页 (2025)


34. Liu, Y., Liu, Y., Jiang, C., Lyu, K., Wan, W., Shen, H., Liang, B., Fu, Z., Wang, H., Yi, L.: Hoi4d: A 4d egocentric dataset for category-level human-object interaction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21013-21022 (2022)
34. Liu, Y., Liu, Y., Jiang, C., Lyu, K., Wan, W., Shen, H., Liang, B., Fu, Z., Wang, H., Yi, L.: Hoi4d: 一个用于类别级人-物交互的 4D 自我中心数据集。载于：IEEE/CVF 计算机视觉与模式识别会议论文集，第 21013-21022 页 (2022)


35. OpenAI: Gpt-40: The omni model. Tech. rep. (May 2024), https://openai.com/ index/hello-gpt-40 accessed: 2025-11-14
35. OpenAI: Gpt-4o: 全能模型。技术报告 (2024年5月)，https://openai.com/index/hello-gpt-40 访问日期：2025-11-14


36. Peddi, R., Arya, S., Challa, B., Pallapothula, L., Vyas, A., Gouripeddi, B., Zhang, Q., Wang, J., Komaragiri, V., Ragan, E., et al.: Captaincook4d: A dataset for understanding errors in procedural activities. Advances in Neural Information Processing Systems 37, 135626-135679 (2024)
36. Peddi, R., Arya, S., Challa, B., Pallapothula, L., Vyas, A., Gouripeddi, B., Zhang, Q., Wang, J., Komaragiri, V., Ragan, E., 等：Captaincook4d: 一个用于理解程序性活动错误的数据集。神经信息处理系统进展 37, 135626-135679 (2024)


37. Qin, T., Li, P., Vins-mono, S.S.: A robust and versatile monocular visual-inertial state estimator., 2018, 34. DOI: https://doi.org/10.1109/TRO pp. 1004-1020 (2018)
37. Qin, T., Li, P., Vins-mono, S.S.: 一种鲁棒且通用的单目视觉惯性状态估计器，2018, 34。DOI: https://doi.org/10.1109/TRO 第 1004-1020 页 (2018)


38. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748-8763. PmLR (2021)
38. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J. 等：从自然语言监督中学习可迁移的视觉模型。载于：国际机器学习会议。第 8748-8763 页。PmLR (2021)


39. Routray, S., Pan, H., Jain, U., Bahl, S., Pathak, D.: Vipra: Video prediction for robot actions. arXiv preprint arXiv:2511.07732 (2025)
39. Routray, S., Pan, H., Jain, U., Bahl, S., Pathak, D.：Vipra：用于机器人动作的视频预测。arXiv 预印本 arXiv:2511.07732 (2025)


40. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2016)
40. Schönberger, J.L., Frahm, J.M.：运动恢复结构（Structure-from-motion）再探。载于：计算机视觉与模式识别会议 (CVPR) (2016)


41. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: European conference on computer vision. pp. 402-419. Springer (2020)
41. Teed, Z., Deng, J.：Raft：用于光流估计的循环全对场变换。载于：欧洲计算机视觉会议。第 402-419 页。Springer (2020)


42. Tongyi Lab, Alibaba Cloud: Qwen3-vl-4b-instruct: Instruction-tuned vision-language model (2025), https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct model version: qwen3-vl-4b-instruct. Accessed: 2025-11-14
42. 通义实验室，阿里云：Qwen3-vl-4b-instruct：指令微调视觉语言模型 (2025)，https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 模型版本：qwen3-vl-4b-instruct。访问日期：2025-11-14


43. Tongyi Lab, Alibaba Cloud: Qwen3-vl: Advancing vision-language understanding with scalable multimodal pretraining (2025), https://github.com/QwenLM/Qwen3 accessed: 2025-11-14
43. 通义实验室，阿里云：Qwen3-vl：通过可扩展多模态预训练推进视觉语言理解 (2025)，https://github.com/QwenLM/Qwen3 访问日期：2025-11-14


44. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)
44. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F. 等：Llama：开放且高效的基础语言模型。arXiv 预印本 arXiv:2302.13971 (2023)


45. Wang, J., Zhang, Q., Chao, Y.W., Wen, B., Guo, X., Xiang, Y.: Ho-cap: A capture system and dataset for 3d reconstruction and pose tracking of hand-object interaction. arXiv preprint arXiv:2406.06843 (2024)
45. Wang, J., Zhang, Q., Chao, Y.W., Wen, B., Guo, X., Xiang, Y.：Ho-cap：用于手物交互三维重建与姿态跟踪的捕捉系统及数据集。arXiv 预印本 arXiv:2406.06843 (2024)


46. Wang, M., Xing, J., Liu, Y.: Actionclip: A new paradigm for video action recognition. arXiv preprint arXiv:2109.08472 (2021)
46. Wang, M., Xing, J., Liu, Y.：Actionclip：视频动作识别的新范式。arXiv 预印本 arXiv:2109.08472 (2021)


47. Wang, Q., Ye, V., Gao, H., Zeng, W., Austin, J., Li, Z., Kanazawa, A.: Shape of motion: 4d reconstruction from a single video. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9660-9672 (2025)
47. Wang, Q., Ye, V., Gao, H., Zeng, W., Austin, J., Li, Z., Kanazawa, A.：运动的形状：从单视频进行 4D 重建。载于：IEEE/CVF 国际计算机视觉会议论文集。第 9660-9672 页 (2025)


48. Wang, X., Zhao, K., Liu, F., Wang, J., Zhao, G., Bao, X., Zhu, Z., Zhang, Y., Wang, X.: Egovid-5m: A large-scale video-action dataset for egocentric video generation. arXiv preprint arXiv:2411.08380 (2024)
48. Wang, X., Zhao, K., Liu, F., Wang, J., Zhao, G., Bao, X., Zhu, Z., Zhang, Y., Wang, X.：Egovid-5m：用于第一人称视频生成的大规模视频动作数据集。arXiv 预印本 arXiv:2411.08380 (2024)


49. Wang, X., Kwon, T., Rad, M., Pan, B., Chakraborty, I., Andrist, S., Bohus, D., Feniello, A., Tekin, B., Frujeri, F.V., et al.: Holoassist: an egocentric human interaction dataset for interactive ai assistants in the real world. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 20270-20281 (2023)
49. Wang, X., Kwon, T., Rad, M., Pan, B., Chakraborty, I., Andrist, S., Bohus, D., Feniello, A., Tekin, B., Frujeri, F.V. 等：Holoassist：用于现实世界交互式 AI 助手的第一人称人类交互数据集。载于：IEEE/CVF 国际计算机视觉会议论文集。第 20270-20281 页 (2023)


50. Wen, B., Trepte, M., Aribido, J., Kautz, J., Gallo, O., Birchfield, S.: Foundation-stereo: Zero-shot stereo matching. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5249-5260 (2025)
50. Wen, B., Trepte, M., Aribido, J., Kautz, J., Gallo, O., Birchfield, S.：Foundation-stereo：零样本立体匹配。载于：IEEE/CVF 计算机视觉与模式识别会议论文集。第 5249-5260 页 (2025)


51. Wen, J., Zhu, M., Zhu, Y., Tang, Z., Li, J., Zhou, Z., Li, C., Liu, X., Peng, Y., Shen, C., et al.: Diffusion-vla: Generalizable and interpretable robot foundation model via self-generated reasoning. arXiv preprint arXiv:2412.03293 (2024)
51. Wen, J., Zhu, M., Zhu, Y., Tang, Z., Li, J., Zhou, Z., Li, C., Liu, X., Peng, Y., Shen, C. 等：Diffusion-vla：通过自生成推理实现可泛化且可解释的机器人基础模型。arXiv 预印本 arXiv:2412.03293 (2024)


52. Xing, J., Xia, M., Zhang, Y., Chen, H., Yu, W., Liu, H., Liu, G., Wang, X., Shan, Y., Wong, T.T.: Dynamicrafter: Animating open-domain images with video diffusion priors. In: European Conference on Computer Vision. pp. 399-417. Springer (2024)
52. Xing, J., Xia, M., Zhang, Y., Chen, H., Yu, W., Liu, H., Liu, G., Wang, X., Shan, Y., Wong, T.T.: Dynamicrafter: 利用视频扩散先验实现开放域图像动画化。载于：欧洲计算机视觉国际会议，第399-417页。Springer (2024)


53. Xu, H., Ghosh, G., Huang, P.Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettle-moyer, L., Feichtenhofer, C.: Videoclip: Contrastive pre-training for zero-shot video-text understanding. arXiv preprint arXiv:2109.14084 (2021)
53. Xu, H., Ghosh, G., Huang, P.Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettle-moyer, L., Feichtenhofer, C.: Videoclip: 用于零样本视频-文本理解的对比预训练。arXiv预印本 arXiv:2109.14084 (2021)


54. Xu, M., Zhang, H., Hou, Y., Xu, Z., Fan, L., Veloso, M., Song, S.: Dexumi: Using human hand as the universal manipulation interface for dexterous manipulation. arXiv preprint arXiv:2505.21864 (2025)
54. Xu, M., Zhang, H., Hou, Y., Xu, Z., Fan, L., Veloso, M., Song, S.: Dexumi: 使用人手作为灵巧操作的通用交互接口。arXiv预印本 arXiv:2505.21864 (2025)


55. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)
55. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3技术报告。arXiv预印本 arXiv:2505.09388 (2025)


56. Zhou, Z., Zhu, Y., Wen, J., Shen, C., Xu, Y.: Vision-language-action model with open-world embodied reasoning from pretrained knowledge. arXiv preprint arXiv:2505.21906 (2025)
56. Zhou, Z., Zhu, Y., Wen, J., Shen, C., Xu, Y.: 具备预训练知识开放世界具身推理的视觉-语言-动作模型。arXiv预印本 arXiv:2505.21906 (2025)


57. Zhu, X., Liu, Y., Li, H., Chen, J.: Learning generalizable robot policy with human demonstration video as a prompt. arXiv preprint arXiv:2505.20795 (2025)
57. Zhu, X., Liu, Y., Li, H., Chen, J.: 以人类演示视频为提示学习可泛化的机器人策略。arXiv预印本 arXiv:2505.20795 (2025)


58. Zhu, Y., Feng, F.: Let me show you: Learning by retrieving from egocentric video for robotic manipulation. arXiv preprint arXiv:2511.05199 (2025)
58. Zhu, Y., Feng, F.: 让我演示给你看：通过检索第一人称视角视频进行机器人操作学习。arXiv预印本 arXiv:2511.05199 (2025)


59. Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al.: Rt-2: Vision-language-action models transfer web knowledge to robotic control. In: Conference on Robot Learning. pp. 2165-2183. PMLR (2023)
59. Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al.: Rt-2: 将网络知识迁移至机器人控制的视觉-语言-动作模型。载于：机器人学习会议，第2165-2183页。PMLR (2023)